//! A fleet head takes on the models its nodes bring, with no restart: a head serving one model
//! serves another once a node that serves it joins, and loads and unloads it on request.
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::fleet_join::accept_nodes;
use superfluid_daemon::fleet_manager::{FleetManager, PlacementPolicy};
use superfluid_daemon::fleet_models::HeadModels;
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::registry::ModelRegistry;
use superfluid_daemon::{openai, Daemon, DaemonOptions, EngineHost, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_linkf::{Endpoint, Role};
use superfluid_proto::linkf::Lease;

static DIRS: AtomicU64 = AtomicU64::new(0);
const TOKEN: &[u8] = b"a fleet token";

fn store(tag: &str) -> SessionStore {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fleet-adopt-{tag}-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    SessionStore::open(&dir.join("wal.log")).unwrap()
}

fn engine() -> EngineHost {
    EngineHost::spawn(|| (MockEngine::new(EngineConfig { vocab: 512, ..EngineConfig::default() }), None)).unwrap()
}

/// A node named `identity` that serves `model`, dialing `head` once per lane and again whenever
/// a link ends.
fn join(head: &str, identity: &'static str, model: &'static str) {
    join_with(head, identity, model, 7);
}

/// As [`join`], telling the head its tokenizer by `tokenizer`.
fn join_with(head: &str, identity: &'static str, model: &'static str, tokenizer: u64) {
    let daemon = Arc::new(Daemon::new(store(identity), engine(), Box::new(MockCodec), 2));
    let node = NodeAgent::new(daemon, identity, model).with_max_lanes(2).with_auth(TOKEN.to_vec()).with_tokenizer(tokenizer);
    for _ in 0..2 {
        let (head, mut conn) = (head.to_string(), node.connection());
        std::thread::spawn(move || loop {
            let Ok(sock) = TcpStream::connect(&head) else { break };
            let Ok(ep) = Endpoint::from_tcp_secure(sock, TOKEN, Role::Responder, Duration::from_secs(5)) else { break };
            let _ = conn.serve(ep);
            std::thread::sleep(Duration::from_millis(50));
        });
    }
}

/// A head serving `first` over the nodes that join it, with an HTTP API; it builds another model
/// as the head's own `serve` does, minus the tokenizer a node shares.
fn head(first: &str) -> (String, String) {
    let (nodes, http, _) = head_of_fleet(first);
    (nodes, http)
}

/// As [`head`], with the head's fleet.
fn head_of_fleet(first: &str) -> (String, String, Arc<FleetManager>) {
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let manager = Arc::new(FleetManager::with_options(Vec::new(), PlacementPolicy::LoadAware, lease, 90, 1));
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
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(20)))
        .http_status_as_error(false)
        .build()
        .new_agent()
}

fn post(addr: &str, path: &str, body: &Value) -> (u16, Value) {
    let mut resp = agent()
        .post(&format!("http://{addr}{path}"))
        .header("content-type", "application/json")
        .send(body.to_string().as_bytes())
        .unwrap_or_else(|e| panic!("POST {path}: {e}"));
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

fn chat(addr: &str, model: &str) -> (u16, Value) {
    let user = serde_json::json!([{"role": "user", "content": "hello there"}]);
    post(addr, "/v1/chat/completions", &serde_json::json!({"model": model, "messages": user, "max_tokens": 8, "temperature": 0}))
}

#[test]
fn a_head_serving_one_model_serves_the_one_a_joining_node_brings() {
    let (nodes, http) = head("model-a");
    join(&nodes, "box-a", "model-a");
    assert_eq!(models(&http), ["model-a"]);

    join(&nodes, "box-b", "model-b");
    wait_for("the head lists the model the node brought", || models(&http).contains(&"model-b".to_string()));
    assert_eq!(models(&http), ["model-a", "model-b"]);
    let (status, body) = chat(&http, "model-b");
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["model"], "model-b");
    assert!(body["choices"][0]["message"].is_object(), "{body}");
    let (status, body) = chat(&http, "model-a");
    assert_eq!(status, 200, "the head still serves its own: {body}");
}

#[test]
fn a_head_loads_only_a_model_a_node_serves_and_an_unload_outlasts_a_join() {
    let (nodes, http) = head("model-a");
    join(&nodes, "box-a", "model-a");

    let (status, body) = post(&http, "/v1/models/load", &serde_json::json!({"id": "model-c"}));
    assert_eq!(status, 404, "no node serves it: {body}");
    assert_eq!(body["error"]["code"], "model_not_found");

    join(&nodes, "box-b", "model-b");
    wait_for("the node's model is served", || models(&http).contains(&"model-b".to_string()));
    let (status, body) = post(&http, "/v1/models/unload", &serde_json::json!({"id": "model-b"}));
    assert_eq!((status, &body["deleted"]), (200, &Value::Bool(true)), "{body}");
    assert_eq!(models(&http), ["model-a"]);

    // Another node with the unloaded model joins: the head keeps it unloaded.
    join(&nodes, "box-c", "model-b");
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(models(&http), ["model-a"], "a join does not undo an unload");

    let (status, body) = post(&http, "/v1/models/load", &serde_json::json!({"id": "model-b"}));
    assert_eq!(status, 200, "a node serves it: {body}");
    assert_eq!(models(&http), ["model-a", "model-b"]);
    let (status, body) = chat(&http, "model-b");
    assert_eq!(status, 200, "{body}");
}

#[test]
fn the_first_node_to_bring_a_model_sets_the_tokenizer_the_next_is_held_to() {
    let (nodes, http, fleet) = head_of_fleet("model-a");
    join_with(&nodes, "box-b", "model-b", 7);
    wait_for("the node's model is served", || models(&http).contains(&"model-b".to_string()));
    join_with(&nodes, "box-other", "model-b", 9);
    wait_for("the second node joined", || fleet.status().iter().filter(|n| n.up).count() == 2);
    assert_eq!(fleet.serving("model-b").0, 1, "a node that tokenizes otherwise does not serve it here");
    let (status, body) = chat(&http, "model-b");
    assert_eq!(status, 200, "{body}");
    let decoded: std::collections::HashMap<String, u64> = fleet.status().into_iter().map(|n| (n.name, n.decode_tokens)).collect();
    assert_eq!(decoded["box-other"], 0, "{decoded:?}");
}
