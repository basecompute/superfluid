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
use superfluid_proto::linkf::Lease;

static DIRS: AtomicU64 = AtomicU64::new(0);

fn mock_daemon() -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-liveness-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
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
    let mgr = manager(&[&a.addr], PlacementPolicy::LeastLoaded, 1);
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
