//! A head's connections outliving idle spells, nodes that go down and come back, and what the
//! head learns about its nodes at start.

use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use superfluid_daemon::fleet_manager::{FleetManager, NodeSpec, PlacementPolicy};
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::{Daemon, DaemonError, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_linkf::Liveness;
use superfluid_proto::linkf::Lease;

static DIRS: AtomicU64 = AtomicU64::new(0);

fn mock_daemon() -> Daemon {
    mock_daemon_with(EngineConfig::default())
}

fn mock_daemon_with(cfg: EngineConfig) -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-liveness-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

/// A head that heartbeats every 50 ms, well under the short silences these nodes keep.
fn quick(heartbeat_ms: u64, silence: Duration) -> Liveness {
    Liveness { heartbeat: Duration::from_millis(heartbeat_ms), silence }
}

struct Node {
    addr: String,
    accepts: Arc<AtomicUsize>,
}

fn serve(listener: TcpListener, identity: &'static str, idle: Duration, auth: &'static [u8]) -> Node {
    serve_lanes(listener, identity, idle, auth, 0)
}

fn serve_lanes(listener: TcpListener, identity: &'static str, idle: Duration, auth: &'static [u8], lanes: u32) -> Node {
    let addr = listener.local_addr().unwrap().to_string();
    let accepts = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&accepts);
    std::thread::spawn(move || {
        let node = NodeAgent::new(Arc::new(mock_daemon()), identity, "mock")
            .with_max_lanes(lanes)
            .with_timeouts(Duration::from_secs(2), idle)
            .with_auth(auth.to_vec());
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            counted.fetch_add(1, Ordering::SeqCst);
            let ep = if auth.is_empty() {
                superfluid_linkf::Endpoint::from_tcp(sock)
            } else {
                superfluid_linkf::Endpoint::from_tcp_secure(sock, auth, superfluid_linkf::Role::Responder, Duration::from_secs(5))
            };
            let Ok(ep) = ep else { continue };
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    Node { addr, accepts }
}

fn node(identity: &'static str, idle: Duration) -> Node {
    serve(TcpListener::bind("127.0.0.1:0").unwrap(), identity, idle, b"")
}

fn manager(addrs: &[&str], policy: PlacementPolicy, conns: usize) -> FleetManager {
    manager_with_key(addrs, policy, conns, b"")
}

fn manager_with_key(addrs: &[&str], policy: PlacementPolicy, conns: usize, key: &[u8]) -> FleetManager {
    let specs = addrs.iter().map(|a| NodeSpec { addr: a.to_string(), auth: key.to_vec() }).collect();
    let lease = Lease { duration_ms: 120_000, renew_by_ms: 90_000 };
    FleetManager::with_options(specs, policy, lease, 90, conns)
}

fn reference(prompt: &[u32], max: u32) -> Vec<u32> {
    mock_daemon().generate_tokens(prompt.to_vec(), GenParams::default(), max, 0).unwrap().0
}

fn stream(mgr: &FleetManager, session: u64, prompt: &[u32]) -> Result<Vec<u32>, DaemonError> {
    mgr.place(session, GenParams::default(), 0, "mock", "mock", prompt, 8)?;
    let mut got = Vec::new();
    let r = mgr.generate_streaming(session, 8, 60_000, |b| got.extend_from_slice(b));
    mgr.finish(session);
    r.map(|_| got)
}

fn round(mgr: &FleetManager, first: u64, n: u64) -> Vec<Result<Vec<u32>, DaemonError>> {
    std::thread::scope(|s| {
        let jobs: Vec<_> = (first..first + n)
            .map(|session| s.spawn(move || stream(mgr, session, &[1, 2, 3, session as u32])))
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    })
}

fn assert_round(results: Vec<Result<Vec<u32>, DaemonError>>, first: u64, when: &str) {
    for (k, r) in results.into_iter().enumerate() {
        let session = first + k as u64;
        let got = r.unwrap_or_else(|e| panic!("{when}: session {session} failed: {e}"));
        assert_eq!(got, reference(&[1, 2, 3, session as u32], 8), "{when}: session {session}");
    }
}

#[test]
fn streams_survive_nodes_dropping_idle_connections() {
    let idle = Duration::from_millis(300);
    let (a, b) = (node("idle-a", idle), node("idle-b", idle));
    let mgr = manager(&[&a.addr, &b.addr], PlacementPolicy::LeastLoaded, 4);

    assert_round(round(&mgr, 1, 8), 1, "warm");
    std::thread::sleep(idle * 3);
    assert_round(round(&mgr, 100, 8), 100, "after the nodes dropped every idle connection");
    assert_round(round(&mgr, 200, 8), 200, "after that");
}

#[test]
fn tending_keeps_idle_connections_open() {
    let idle = Duration::from_millis(300);
    let a = node("tended", idle);
    let mgr = manager(&[&a.addr], PlacementPolicy::LeastLoaded, 1).with_liveness(quick(50, Duration::from_secs(30)));
    assert_round(round(&mgr, 1, 1), 1, "warm");
    let opened = a.accepts.load(Ordering::SeqCst);

    let until = Instant::now() + idle * 3;
    while Instant::now() < until {
        mgr.tend();
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_round(round(&mgr, 2, 1), 2, "after an idle spell the head tended");
    assert_eq!(a.accepts.load(Ordering::SeqCst), opened, "the tended connection was kept, not reopened");
}

#[test]
fn a_node_that_was_down_is_used_again_once_it_answers() {
    // The node's listener stays bound throughout; until it is "up" it hangs up on every
    // connection, as a node that is restarting would.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let late_addr = listener.local_addr().unwrap().to_string();
    let answering = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let accepts = Arc::new(AtomicUsize::new(0));
    {
        let (answering, accepts) = (Arc::clone(&answering), Arc::clone(&accepts));
        std::thread::spawn(move || {
            let node = NodeAgent::new(Arc::new(mock_daemon()), "late", "mock")
                .with_timeouts(Duration::from_secs(2), Duration::from_secs(60));
            for sock in listener.incoming() {
                let Ok(sock) = sock else { break };
                if !answering.load(Ordering::SeqCst) {
                    continue;
                }
                accepts.fetch_add(1, Ordering::SeqCst);
                let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
                let mut conn = node.connection();
                std::thread::spawn(move || {
                    let _ = conn.serve(ep);
                });
            }
        });
    }
    let up = node("up", Duration::from_secs(60));
    let retry = Duration::from_millis(100);
    let mgr = manager(&[&late_addr, &up.addr], PlacementPolicy::RoundRobin, 1).with_retry_every(retry);

    assert_round(round(&mgr, 1, 2), 1, "with one node down");
    let placed_while_down: Vec<String> = (10..14)
        .map(|s| {
            mgr.place(s, GenParams::default(), 0, "mock", "mock", &[1, 2, 3], 8).unwrap();
            mgr.node_of(s).unwrap()
        })
        .collect();
    assert!(placed_while_down.iter().all(|n| n == "up"), "a down node takes nothing: {placed_while_down:?}");

    answering.store(true, Ordering::SeqCst);
    std::thread::sleep(retry * 2);
    mgr.retry_down();
    assert!(accepts.load(Ordering::SeqCst) > 0, "the background retry reached the node");

    let placed: Vec<String> = (20..24)
        .map(|s| {
            mgr.place(s, GenParams::default(), 0, "mock", "mock", &[1, 2, 3], 8).unwrap();
            mgr.node_of(s).unwrap()
        })
        .collect();
    assert!(placed.iter().any(|n| n == "late"), "the recovered node takes work again: {placed:?}");
}

#[test]
fn a_session_appended_after_an_idle_drop_is_bound_on_the_new_connection() {
    let idle = Duration::from_millis(300);
    let a = node("rebind", idle);
    let mgr = manager(&[&a.addr], PlacementPolicy::LeastLoaded, 1);
    let prompt = [1u32, 2, 3, 4];
    mgr.place(1, GenParams::default(), 0, "mock", "mock", &prompt, 8).unwrap();
    std::thread::sleep(idle * 3);
    let grown = [1u32, 2, 3, 4, 5, 6];
    mgr.append(1, &grown).expect("append re-binds the session on a fresh connection");
    let mut got = Vec::new();
    mgr.generate_streaming(1, 8, 60_000, |b| got.extend_from_slice(b)).expect("and generates there");
    assert_eq!(got, reference(&grown, 8));
}

#[test]
fn the_probe_names_each_node_and_its_window() {
    let a = node("probe-a", Duration::from_secs(60));
    let mgr = manager(&[&a.addr], PlacementPolicy::LeastLoaded, 1);
    let reports = mgr.probe();
    let r = reports[0].as_ref().expect("the node answers");
    assert_eq!(r.identity, "probe-a");
    assert_eq!(r.models, vec!["mock".to_string()]);
    assert!(r.max_context_tokens > 0);
    assert_eq!(mgr.advertised_context("mock"), Some(r.max_context_tokens));
}

#[test]
fn the_probe_says_a_key_mismatch_is_one() {
    let a = serve(TcpListener::bind("127.0.0.1:0").unwrap(), "keyed", Duration::from_secs(60), b"right");
    let mgr = manager_with_key(&[&a.addr], PlacementPolicy::LeastLoaded, 1, b"wrong");
    let err = mgr.probe().remove(0).expect_err("the node refuses the head");
    assert!(err.to_string().contains("--fleet-auth"), "{err}");
}

#[test]
fn the_probe_gives_up_on_an_address_that_never_answers() {
    let mute = TcpListener::bind("127.0.0.1:0").unwrap();
    let mgr = manager(&[&mute.local_addr().unwrap().to_string()], PlacementPolicy::LeastLoaded, 1);
    let start = Instant::now();
    let err = mgr.probe().remove(0).expect_err("nothing answers the handshake");
    assert!(start.elapsed() < Duration::from_secs(15), "{:?}", start.elapsed());
    assert!(err.to_string().contains("handshake"), "{err}");
}

#[test]
fn a_finished_session_is_forgotten() {
    let a = node("forget", Duration::from_secs(60));
    let mgr = manager(&[&a.addr], PlacementPolicy::LeastLoaded, 1);
    assert_round(round(&mgr, 7, 1), 7, "one request");
    assert_eq!(mgr.node_of(7), None);
    assert!(matches!(mgr.generate(7, 4, 60_000), Err(DaemonError::UnknownSession(7))));
}

#[test]
fn the_head_opens_a_connection_per_node_lane_by_default() {
    let a = serve_lanes(TcpListener::bind("127.0.0.1:0").unwrap(), "lanes", Duration::from_secs(60), b"", 4);
    let mgr = manager(&[&a.addr], PlacementPolicy::LeastLoaded, superfluid_daemon::fleet_manager::DEFAULT_CONNS_PER_NODE);
    mgr.probe().remove(0).expect("the node answers");
    for s in 1..=8 {
        assert_round(round(&mgr, s, 1), s, "one at a time");
    }
    assert_eq!(a.accepts.load(Ordering::SeqCst), 4, "one connection per lane, each reused");

    let b = serve_lanes(TcpListener::bind("127.0.0.1:0").unwrap(), "pinned", Duration::from_secs(60), b"", 4);
    let mgr = manager(&[&b.addr], PlacementPolicy::LeastLoaded, 1);
    for s in 1..=4 {
        assert_round(round(&mgr, s, 1), s, "pinned to one connection");
    }
    assert_eq!(b.accepts.load(Ordering::SeqCst), 1, "--fleet-conns-per-node 1 keeps one");
}

/// A node whose engine takes `tick` a step, with `idle` as its silence limit; counts the
/// connections it served to the end.
fn slow_node(identity: &'static str, tick: Duration, idle: Duration) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let closed = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&closed);
    std::thread::spawn(move || {
        let daemon = mock_daemon_with(EngineConfig { tick_delay: tick, ..EngineConfig::default() });
        let node = NodeAgent::new(Arc::new(daemon), identity, "mock").with_timeouts(Duration::from_secs(2), idle);
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let (mut conn, closed) = (node.connection(), Arc::clone(&counted));
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
                closed.fetch_add(1, Ordering::SeqCst);
            });
        }
    });
    (addr, closed)
}

/// Runs the head's background tend every 50 ms until dropped.
struct Tender(Arc<std::sync::atomic::AtomicBool>, Option<std::thread::JoinHandle<()>>);

fn tend(mgr: &Arc<FleetManager>) -> Tender {
    let on = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let (mgr, running) = (Arc::clone(mgr), Arc::clone(&on));
    Tender(
        on,
        Some(std::thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                mgr.tend();
                std::thread::sleep(Duration::from_millis(50));
            }
        })),
    )
}

impl Drop for Tender {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
        if let Some(t) = self.1.take() {
            let _ = t.join();
        }
    }
}

#[test]
fn a_generation_longer_than_its_lease_and_the_silence_limit_finishes_on_its_connection() {
    // The lease is renewed over the connection the session was placed on while it generates,
    // and the time spent generating is not silence on either end.
    let idle = Duration::from_millis(500);
    let (addr, closed) = slow_node("slow", Duration::from_millis(400), idle);
    let specs = vec![NodeSpec { addr, auth: Vec::new() }];
    let lease = Lease { duration_ms: 600, renew_by_ms: 400 };
    // Several connections, as a node's lanes take: an idle one must not fence a busy one's lease.
    let mgr = Arc::new(
        FleetManager::with_options(specs, PlacementPolicy::LeastLoaded, lease, 90, 4).with_liveness(quick(50, idle)),
    );
    let _tend = tend(&mgr);

    let max: u32 = 240;
    for (session, prompt) in [(1u64, [1u32, 2, 3, 7]), (2, [1, 2, 3, 8])] {
        mgr.place(session, GenParams::default(), 0, "mock", "mock", &prompt, max as u64).unwrap();
        let started = Instant::now();
        let mut got = Vec::new();
        mgr.generate_streaming(session, max as u64, 60_000, |b| got.extend_from_slice(b))
            .unwrap_or_else(|e| panic!("session {session}: {e}"));
        let took = started.elapsed();
        mgr.finish(session);
        assert!(took > idle * 2, "session {session} ran past its lease and the silence limit ({took:?})");
        assert_eq!(got, reference(&prompt, max), "session {session}: every token, once");
    }
    assert_eq!(closed.load(Ordering::SeqCst), 0, "the node closed none of its connections");
}

#[test]
fn a_step_slower_than_the_silence_limit_keeps_the_link() {
    // While the engine is inside one step, longer than either end waits on silence, the node's
    // connection goes on heartbeating and the head's goes on waiting.
    let idle = Duration::from_millis(400);
    let (addr, closed) = slow_node("stalled", Duration::from_millis(900), idle);
    let mgr = FleetManager::with_options(
        vec![NodeSpec { addr, auth: Vec::new() }],
        PlacementPolicy::LeastLoaded,
        Lease { duration_ms: 60_000, renew_by_ms: 40_000 },
        90,
        1,
    )
    .with_liveness(quick(50, idle));
    let prompt = [1u32, 2, 3, 9];
    mgr.place(1, GenParams::default(), 0, "mock", "mock", &prompt, 3).unwrap();
    let mut got = Vec::new();
    mgr.generate_streaming(1, 3, 60_000, |b| got.extend_from_slice(b)).expect("the slow steps finish");
    assert_eq!(got, reference(&prompt, 3));
    assert_eq!(closed.load(Ordering::SeqCst), 0, "the node closed none of its connections");
}

#[test]
fn a_placed_session_waiting_past_its_lease_is_kept() {
    // Between placement and generation the session's connection sits idle; the head's tend is
    // what answers its renewals.
    let (addr, _) = slow_node("waiting", Duration::ZERO, Duration::from_secs(30));
    let lease = Lease { duration_ms: 400, renew_by_ms: 200 };
    let mgr = Arc::new(
        FleetManager::with_options(vec![NodeSpec { addr, auth: Vec::new() }], PlacementPolicy::LeastLoaded, lease, 90, 1)
            .with_liveness(quick(50, Duration::from_secs(30))),
    );
    let _tend = tend(&mgr);
    let prompt = [1u32, 2, 3, 10];
    mgr.place(1, GenParams::default(), 0, "mock", "mock", &prompt, 8).unwrap();
    std::thread::sleep(Duration::from_millis(1_200));
    let mut got = Vec::new();
    mgr.generate_streaming(1, 8, 60_000, |b| got.extend_from_slice(b)).expect("the lease was kept, so the node takes it");
    assert_eq!(got, reference(&prompt, 8));
}

#[test]
fn a_node_that_goes_silent_mid_generation_is_given_up_on() {
    use superfluid_proto::handshake::{CapacitySummary, HelloAckF};
    use superfluid_proto::linkf::LinkFLimits;
    // Answers the handshake, then reads and never says anything again, as a machine that slept
    // with its sockets open looks to the head.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mute = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let ack = HelloAckF {
            chosen_version: 0,
            node_identity: "mute".into(),
            capacity: CapacitySummary {
                gpu_memory_bytes: 0,
                gpu_memory_free_bytes: 0,
                host_memory_bytes: 0,
                active_sessions: 0,
                max_sessions: u32::MAX,
                max_context_tokens: 4096,
                kv_blocks_used: 0,
                kv_blocks_total: 0,
                lanes_active: 0,
                max_lanes: 1,
                queue_depth: 0,
                decode_tokens_per_s: 0,
                prefill_tokens_per_s: 0,
                sampled_at_ms: 0,
            },
            loadable_models: vec!["mock".into()],
            feature_flags: 0,
            limits: LinkFLimits { max_frame: 1 << 20, transfer_window: 8, max_transfer: 1 << 26, transfer_budget: 1 << 28 },
        };
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ack = ack.clone();
            std::thread::spawn(move || {
                let mut ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
                if ep.answer_hello_as_agent(&ack, b"").is_ok() {
                    while ep.pump().is_ok() {}
                }
            });
        }
    });
    let up = node("speaks", Duration::from_secs(60));
    let mgr = FleetManager::with_options(
        vec![NodeSpec { addr: mute, auth: Vec::new() }, NodeSpec { addr: up.addr.clone(), auth: Vec::new() }],
        PlacementPolicy::RoundRobin,
        Lease { duration_ms: 60_000, renew_by_ms: 40_000 },
        90,
        1,
    )
    .with_liveness(quick(50, Duration::from_millis(300)));
    let prompt = [1u32, 2, 3, 11];
    mgr.place(1, GenParams::default(), 0, "mock", "mock", &prompt, 8).unwrap();
    assert_eq!(mgr.node_of(1).as_deref(), Some("mute"));
    let started = Instant::now();
    let mut got = Vec::new();
    mgr.generate_streaming(1, 8, 60_000, |b| got.extend_from_slice(b)).expect("moved to the node that answers");
    assert!(started.elapsed() < Duration::from_secs(10), "{:?}", started.elapsed());
    assert_eq!(got, reference(&prompt, 8));
    assert_eq!(mgr.node_of(1).as_deref(), Some("speaks"));
}

fn mute_ack() -> superfluid_proto::handshake::HelloAckF {
    use superfluid_proto::handshake::{CapacitySummary, HelloAckF};
    use superfluid_proto::linkf::LinkFLimits;
    HelloAckF {
        chosen_version: 2,
        node_identity: "mute-v2".into(),
        capacity: CapacitySummary {
            gpu_memory_bytes: 0,
            gpu_memory_free_bytes: 0,
            host_memory_bytes: 0,
            active_sessions: 0,
            max_sessions: u32::MAX,
            max_context_tokens: 4096,
            kv_blocks_used: 0,
            kv_blocks_total: 0,
            lanes_active: 0,
            max_lanes: 1,
            queue_depth: 0,
            decode_tokens_per_s: 0,
            prefill_tokens_per_s: 0,
            sampled_at_ms: 0,
        },
        loadable_models: vec!["mock".into()],
        feature_flags: 0,
        limits: LinkFLimits { max_frame: 1 << 20, transfer_window: 8, max_transfer: 1 << 26, transfer_budget: 1 << 28 },
    }
}

#[test]
fn a_version_2_node_that_goes_silent_mid_generation_is_given_up_at_its_wall_budget() {
    use superfluid_proto::linkf::msg_type;
    // A node from before heartbeats, so its silence ends nothing; it answers the hello at version
    // 2 and then never speaks again.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mute = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            std::thread::spawn(move || {
                let mut ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
                while let Ok(frames) = ep.pump() {
                    if frames.iter().any(|f| f.msg_type == msg_type::HELLO) {
                        ep.send(msg_type::HELLO_ACK_F, &mute_ack()).unwrap();
                    }
                }
            });
        }
    });
    let up = node("speaks-v3", Duration::from_secs(60));
    let mgr = FleetManager::with_options(
        vec![NodeSpec { addr: mute, auth: Vec::new() }, NodeSpec { addr: up.addr.clone(), auth: Vec::new() }],
        PlacementPolicy::RoundRobin,
        Lease { duration_ms: 60_000, renew_by_ms: 40_000 },
        90,
        1,
    )
    .with_liveness(quick(50, Duration::from_millis(300)));
    let prompt = [1u32, 2, 3, 12];
    mgr.place(1, GenParams::default(), 0, "mock", "mock", &prompt, 8).unwrap();
    assert_eq!(mgr.node_of(1).as_deref(), Some("mute-v2"));
    let started = Instant::now();
    let mut got = Vec::new();
    mgr.generate_streaming(1, 8, 500, |b| got.extend_from_slice(b)).expect("moved to the node that answers");
    assert!(started.elapsed() < Duration::from_secs(10), "{:?}", started.elapsed());
    assert_eq!(got, reference(&prompt, 8));
    assert_eq!(mgr.node_of(1).as_deref(), Some("speaks-v3"));
}

#[test]
fn a_stage_forward_slower_than_the_silence_limit_keeps_the_link() {
    let idle = Duration::from_millis(300);
    let (addr, closed) = slow_node("stage", Duration::from_millis(900), idle);
    let mut head = superfluid_daemon::fleet::FleetHead::connect(&addr, Vec::new()).unwrap();
    head.keep_alive(quick(50, idle));
    let out = head.forward_stage(&[1, 2, 3], 0, 24, None).expect("the slow stage answers");
    assert!(matches!(out, superfluid_engine::StageOutput::Token(_)), "the last stage gives a token");
    assert_eq!(closed.load(Ordering::SeqCst), 0, "the node closed none of its connections");
}

#[test]
fn a_node_back_under_another_model_name_is_learned_again() {
    // Each connection is served under the name the node has when it is accepted, so the node
    // can come back as another model, as one restarted with another --model-name does.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let name = Arc::new(std::sync::Mutex::new("old"));
    {
        let name = Arc::clone(&name);
        std::thread::spawn(move || {
            let daemon = Arc::new(mock_daemon());
            for sock in listener.incoming() {
                let Ok(sock) = sock else { break };
                let model = *name.lock().unwrap();
                let mut conn = NodeAgent::new(Arc::clone(&daemon), "renamed", model)
                    .with_timeouts(Duration::from_secs(2), Duration::from_millis(300));
                let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
                std::thread::spawn(move || {
                    let _ = conn.serve(ep);
                });
            }
        });
    }
    let retry = Duration::from_millis(100);
    let mgr = manager(&[&addr], PlacementPolicy::LeastLoaded, 1).with_retry_every(retry);
    mgr.probe().remove(0).expect("the node answers");
    assert_eq!(mgr.serving("old"), (1, 1));

    *name.lock().unwrap() = "new";
    // The head said nothing for longer than the node waits, so the node closed the connection;
    // the head's tend sees it go, and its retry reaches the node again.
    std::thread::sleep(Duration::from_millis(900));
    let deadline = Instant::now() + Duration::from_secs(5);
    while mgr.serving("new") != (1, 1) && Instant::now() < deadline {
        mgr.tend();
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(mgr.serving("new"), (1, 1), "the retry learned the name it serves now");
    assert_eq!(mgr.serving("old"), (0, 1));
}
