//! A fleet node serving several models: each on an engine of its own, each session and request
//! run on the model the head names, and a head from before heads named models offered the first.
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::Value;
use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::fleet::FleetHead;
use superfluid_daemon::fleet_join::accept_nodes;
use superfluid_daemon::fleet_manager::{FleetManager, PlacementPolicy};
use superfluid_daemon::fleet_models::HeadModels;
use superfluid_daemon::nodeagent::{NodeAgent, FINISH_DECLINED};
use superfluid_daemon::registry::ModelRegistry;
use superfluid_daemon::{openai, Daemon, DaemonOptions, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_linkf::{Endpoint, Role};
use superfluid_proto::handshake::{HelloAckF, Hello, LinkRole};
use superfluid_proto::linkf::{self, msg_type, Lease};

static DIRS: AtomicU64 = AtomicU64::new(0);
const TOKEN: &[u8] = b"a fleet token";
const LEASE: Lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };

fn store(tag: &str) -> SessionStore {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fleet-node-models-{tag}-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    SessionStore::open(&dir.join("wal.log")).unwrap()
}

fn node_daemon(tag: &str) -> Arc<Daemon> {
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig { vocab: 512, ..EngineConfig::default() }), None)).unwrap();
    Arc::new(Daemon::new(store(tag), host, Box::new(MockCodec), 2))
}

/// A mock engine whose token ring holds at most `ctx` tokens, the longest stream it takes.
fn windowed(tag: &str, ctx: u32) -> Arc<Daemon> {
    use superfluid_proto::linkw::{RingKind, RingSpec};
    use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};
    let host = EngineHost::spawn(move || {
        let cfg = EngineConfig { vocab: 512, ..EngineConfig::default() };
        let vocab = cfg.vocab;
        let specs = vec![
            RingSpec { ring_id: TOKEN_RING_IN, kind: RingKind::Tokens, slot_bytes: 16 + ctx * 4, slots: 64 },
            RingSpec { ring_id: TOKEN_RING_OUT, kind: RingKind::Tokens, slot_bytes: 16 + 1024 * 4, slots: 64 },
            RingSpec { ring_id: LOGITS_RING, kind: RingKind::Logits, slot_bytes: 16 + vocab * 4, slots: 8 },
        ];
        (MockEngine::new(cfg), Some(specs))
    })
    .unwrap();
    Arc::new(Daemon::new(store(tag), host, Box::new(MockCodec), 2))
}

fn decoded(daemon: &Daemon) -> u64 {
    daemon.sched_stats().decode_tokens.load(Ordering::Relaxed)
}

/// `node`, dialing `head` once per lane of each of its two models, and again whenever a link ends.
fn join_as(head: &str, node: NodeAgent) {
    let node = node.with_max_lanes(4).with_auth(TOKEN.to_vec());
    for _ in 0..4 {
        let (head, mut conn) = (head.to_string(), node.connection());
        std::thread::spawn(move || loop {
            let Ok(sock) = TcpStream::connect(&head) else { break };
            let Ok(ep) = Endpoint::from_tcp_secure(sock, TOKEN, Role::Responder, Duration::from_secs(5)) else { break };
            let _ = conn.serve(ep);
            std::thread::sleep(Duration::from_millis(50));
        });
    }
}

/// A head serving `first` over the nodes that join it, with an HTTP API, taking on any model a
/// node brings.
fn head_of_fleet(first: &str) -> (String, String, Arc<FleetManager>) {
    let manager = Arc::new(FleetManager::with_options(Vec::new(), PlacementPolicy::LoadAware, LEASE, 90, 1));
    let over_fleet = {
        let manager = Arc::clone(&manager);
        move |name: &str| {
            Daemon::over_fleet(store("head"), Box::new(MockChatCodec), Arc::clone(&manager), name.to_string(), DaemonOptions::default())
                .map(Arc::new)
        }
    };
    let first_daemon = over_fleet(first).unwrap();
    let models = HeadModels::new(Arc::clone(&manager), vec![first.to_string()], Box::new(over_fleet));
    let registry = Arc::new(ModelRegistry::with_initial(first, first_daemon, models.loader()));
    models.attach(&registry);
    let nodes = TcpListener::bind("127.0.0.1:0").unwrap();
    let nodes_addr = nodes.local_addr().unwrap().to_string();
    let fleet = Arc::clone(&manager);
    std::thread::spawn(move || accept_nodes(nodes, manager, TOKEN.to_vec(), models));
    let http = TcpListener::bind("127.0.0.1:0").unwrap();
    let http_addr = http.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(http, registry, openai::ServeConfig::default());
    });
    (nodes_addr, http_addr, fleet)
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(20))).http_status_as_error(false).build().new_agent()
}

fn chat(addr: &str, model: &str) -> (u16, Value) {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hello there"}],
        "max_tokens": 8,
        "temperature": 0,
    });
    let mut resp = agent()
        .post(&format!("http://{addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .send(body.to_string().as_bytes())
        .unwrap();
    let text = resp.body_mut().read_to_string().unwrap();
    (resp.status().as_u16(), serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}")))
}

fn models(addr: &str) -> Vec<String> {
    let mut resp = agent().get(&format!("http://{addr}/v1/models")).call().unwrap();
    let body: Value = serde_json::from_str(&resp.body_mut().read_to_string().unwrap()).unwrap();
    let mut ids: Vec<String> = body["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap().to_string()).collect();
    ids.sort();
    ids
}

fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ok() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_node_serving_two_models_runs_each_chat_on_the_engine_of_the_model_it_names() {
    let (nodes, http, fleet) = head_of_fleet("model-a");
    let (a, b) = (node_daemon("a"), node_daemon("b"));
    join_as(&nodes, NodeAgent::new(Arc::clone(&a), "box-ab", "model-a").with_model(Arc::clone(&b), "model-b"));
    wait_for("the head takes on the second model the node brings", || models(&http).contains(&"model-b".to_string()));
    assert_eq!(models(&http), ["model-a", "model-b"]);
    assert_eq!(fleet.status().len(), 1, "one node serves both");
    assert_eq!(fleet.status()[0].models, ["model-a", "model-b"]);

    let (status, body) = chat(&http, "model-b");
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["model"], "model-b");
    assert!(decoded(&b) > 0, "model-b's engine ran it");
    assert_eq!(decoded(&a), 0, "model-a's engine did not");

    let before = decoded(&b);
    let (status, body) = chat(&http, "model-a");
    assert_eq!(status, 200, "{body}");
    assert!(decoded(&a) > 0, "model-a's engine ran it");
    assert_eq!(decoded(&b), before, "model-b's engine did not");
}

#[test]
fn a_node_serving_two_models_holds_each_to_its_own_tokenizer_and_window() {
    let (nodes, http, fleet) = head_of_fleet("model-a");
    let (wide, narrow) = (windowed("wide", 4096), windowed("narrow", 1024));
    let (wide_window, window) = (wide.max_stream_tokens(), narrow.max_stream_tokens());
    assert!(window < wide_window, "{window} < {wide_window}");
    let node = NodeAgent::new(wide, "box-ab", "model-a").with_tokenizer(7).with_model(narrow, "model-b").with_tokenizer(9);
    join_as(&nodes, node);
    wait_for("the second model is served", || models(&http).contains(&"model-b".to_string()));
    assert_eq!(fleet.advertised_context("model-b"), Some(window), "each model is held to its own window");
    assert_eq!(fleet.advertised_context("model-a"), Some(wide_window));

    // Another node with model-b on its tokenizer serves it too: the first set model-b's, not model-a's.
    join_as(&nodes, NodeAgent::new(node_daemon("b2"), "box-b", "model-b").with_tokenizer(9));
    wait_for("the second node joined", || fleet.status().iter().filter(|n| n.up).count() == 2);
    assert_eq!(fleet.serving("model-b").0, 2, "{:?}", fleet.status());
    // And one with another tokenizer does not.
    join_as(&nodes, NodeAgent::new(node_daemon("b3"), "box-other", "model-b").with_tokenizer(5));
    wait_for("the third node joined", || fleet.status().iter().filter(|n| n.up).count() == 3);
    assert_eq!(fleet.serving("model-b").0, 2, "{:?}", fleet.status());
}

/// A head's link to `node` from a release before heads named models (Link F version 3): its
/// hello, the node's answer, and the tokenizer it tells.
fn head_from_before(node: NodeAgent) -> (Endpoint, HelloAckF, linkf::TokenizerMsg) {
    let (ours, theirs) = UnixStream::pair().unwrap();
    let mut conn = node.connection();
    std::thread::spawn(move || {
        let _ = conn.serve(Endpoint::new(theirs).unwrap());
    });
    let mut ep = Endpoint::new(ours).unwrap();
    let hello = Hello { proto_versions: (2, 3), link_role: LinkRole::HeadDaemon, auth: Vec::new(), serves: None };
    ep.send(msg_type::HELLO, &hello).unwrap();
    let (mut ack, mut told) = (None, None);
    let by = Instant::now() + Duration::from_secs(10);
    while told.is_none() {
        assert!(Instant::now() < by, "the node answered and told its tokenizer");
        for f in ep.pump().unwrap() {
            match f.msg_type {
                msg_type::HELLO_ACK_F => ack = Some(HelloAckF::decode(&f.payload).unwrap()),
                msg_type::TOKENIZER => told = Some(linkf::TokenizerMsg::decode(&f.payload).unwrap()),
                _ => {}
            }
        }
    }
    (ep, ack.expect("the ack comes first"), told.unwrap())
}

/// A session assignment as a head from before `model` sends it.
#[derive(Serialize)]
struct AssignBefore {
    command_id: u64,
    session_id: u64,
    epoch: u64,
    lease: Lease,
    codec: String,
    params: Vec<u8>,
    qos_class: u8,
    policy_ref: u64,
}

#[test]
fn a_head_from_before_heads_named_models_is_offered_the_first_and_everything_runs_there() {
    let (a, b) = (node_daemon("old-a"), node_daemon("old-b"));
    let node = NodeAgent::new(Arc::clone(&a), "box-ab", "model-a")
        .with_model_id("org/model-a:q4")
        .with_tokenizer(7)
        .with_model(Arc::clone(&b), "model-b")
        .with_model_id("org/model-b:q4")
        .with_tokenizer(9)
        .with_max_lanes(2);
    let (mut ep, ack, told) = head_from_before(node);
    assert_eq!(ack.chosen_version, 3);
    assert_eq!(ack.loadable_models, ["model-a"], "only the first model, which it would not mis-route");
    assert_eq!(ack.model_ids, ["org/model-a:q4"]);
    assert_eq!((told.digest, told.models.len()), (7, 0));

    let session = 5;
    let assign = AssignBefore {
        command_id: 1,
        session_id: session,
        epoch: 1,
        lease: LEASE,
        codec: "chatml".into(),
        params: superfluid_daemon::fleet::encode_params(&GenParams::default()),
        qos_class: 0,
        policy_ref: 0,
    };
    ep.send(msg_type::SESSION_ASSIGN, &assign).unwrap();
    let append = linkf::AppendContentMsg {
        command_id: 2,
        session_id: session,
        epoch: 1,
        blocks: superfluid_daemon::fleet::encode_tokens(&(500..520).collect::<Vec<u32>>()),
        spans: Vec::new(),
        media_refs: Vec::new(),
    };
    ep.send(msg_type::APPEND_CONTENT, &append).unwrap();
    let generate = linkf::GenerateReqMsg {
        command_id: 3,
        session_id: session,
        epoch: 1,
        generation_id: 1,
        budgets: linkf::GenerationBudgetsMsg { max_new_tokens: 6, max_wall_ms: 30_000 },
        policy_ref: 0,
    };
    ep.send(msg_type::GENERATE_REQ, &generate).unwrap();
    let (mut tokens, mut finish) = (Vec::new(), None);
    let by = Instant::now() + Duration::from_secs(20);
    while finish.is_none() {
        assert!(Instant::now() < by, "the generation finished");
        for f in ep.pump().unwrap().into_iter().filter(|f| f.msg_type == msg_type::TOKEN_EVENTS) {
            let m: linkf::TokenEventsMsg = postcard::from_bytes(&f.payload).unwrap();
            for ev in m.events {
                match ev.payload {
                    linkf::TokenEventPayload::Tokens(t) => tokens.extend(t),
                    linkf::TokenEventPayload::Finish { reason } => finish = Some(reason),
                    _ => {}
                }
            }
        }
    }
    assert_ne!(finish, Some(FINISH_DECLINED));
    assert_eq!(tokens.len(), 6);
    assert!(decoded(&a) > 0, "the first model ran it");
    assert_eq!(decoded(&b), 0);
}

#[test]
fn a_head_of_this_release_is_offered_every_model_and_a_session_for_one_not_served_is_declined() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let (a, b) = (node_daemon("new-a"), node_daemon("new-b"));
    let node = NodeAgent::new(Arc::clone(&a), "box-ab", "model-a").with_model(Arc::clone(&b), "model-b").with_max_lanes(4);
    std::thread::spawn(move || {
        for sock in listener.incoming().flatten() {
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(Endpoint::from_tcp(sock).unwrap());
            });
        }
    });
    let mut head = FleetHead::connect(&addr, Vec::new()).unwrap();
    assert_eq!(head.node().loadable_models, ["model-a", "model-b"]);
    let told: Vec<&str> = head.models().iter().map(|m| m.name.as_str()).collect();
    assert_eq!(told, ["model-a", "model-b"]);

    let prompt: Vec<u32> = (500..520).collect();
    head.assign_for(1, GenParams::default(), 0, "chatml", "model-b", LEASE).unwrap();
    head.append(1, &prompt).unwrap();
    let g = head.generate(1, 6, 30_000).unwrap();
    assert_eq!(g.tokens.len(), 6);
    assert!(decoded(&b) > 0 && decoded(&a) == 0, "model-b's engine ran it");

    head.assign_for(2, GenParams::default(), 0, "chatml", "model-c", LEASE).unwrap();
    head.append(2, &prompt).unwrap();
    let g = head.generate(2, 6, 30_000).unwrap();
    assert_eq!(g.finish, FINISH_DECLINED, "a model the node does not serve runs on none");
    assert_eq!(decoded(&a), 0);

    // State operations go to the model named too: a capability record from each.
    use superfluid_proto::linkw::{StateSyncOkMsg, StateSyncReqMsg};
    assert!(matches!(head.state_sync("model-b", StateSyncReqMsg::Capabilities, &[]), Ok(StateSyncOkMsg::Capabilities { .. })));
    let refused = head.state_sync("model-c", StateSyncReqMsg::Capabilities, &[]).unwrap_err();
    assert!(refused.to_string().contains("does not serve model-c"), "{refused}");
}
