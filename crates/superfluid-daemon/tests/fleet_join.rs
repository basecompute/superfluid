//! A node that dials the head, over the encrypted link.
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use superfluid_daemon::fleet_join::accept_nodes;
use superfluid_daemon::fleet_manager::{FleetManager, PlacementPolicy};
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::{Daemon, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_linkf::{Endpoint, Role};
use superfluid_proto::linkf::Lease;

static DIRS: AtomicU64 = AtomicU64::new(0);
const TOKEN: &[u8] = b"a fleet token";

fn mock_daemon() -> Daemon {
    mock_daemon_ticking(Duration::ZERO)
}

/// A daemon whose engine takes `tick` a step.
fn mock_daemon_ticking(tick: Duration) -> Daemon {
    let dir = std::env::temp_dir().join(format!("superfluid-join-{}-{}", std::process::id(), DIRS.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || (MockEngine::new(EngineConfig { tick_delay: tick, ..EngineConfig::default() }), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

/// Dials `head` and serves until the link ends, then dials again.
fn keep_dialing(head: String, mut conn: NodeAgent) {
    std::thread::spawn(move || loop {
        let Ok(sock) = TcpStream::connect(&head) else { break };
        let Ok(ep) = Endpoint::from_tcp_secure(sock, TOKEN, Role::Responder, Duration::from_secs(5)) else { break };
        let _ = conn.serve(ep);
        std::thread::sleep(Duration::from_millis(50));
    });
}

/// A node with `lanes` lanes that dials `head` once per lane and serves until a link ends,
/// then dials again.
fn join(head: String, identity: &'static str, lanes: u32) {
    let node = NodeAgent::new(Arc::new(mock_daemon()), identity, "mock").with_max_lanes(lanes).with_auth(TOKEN.to_vec());
    for _ in 0..lanes {
        keep_dialing(head.clone(), node.connection());
    }
}

fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ok() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_node_that_dials_in_is_adopted_and_serves_a_session() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let head_addr = listener.local_addr().unwrap().to_string();
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let manager = Arc::new(FleetManager::with_options(Vec::new(), PlacementPolicy::LoadAware, lease, 90, 1));
    let m = Arc::clone(&manager);
    std::thread::spawn(move || accept_nodes(listener, m, TOKEN.to_vec(), vec!["mock".into()]));
    assert!(manager.load().is_empty(), "no node yet");

    join(head_addr.clone(), "box-a", 1);
    wait_for("the node joined", || manager.load().len() == 1);
    let report = manager.probe();
    assert_eq!(report.len(), 1);
    let node = report[0].as_ref().expect("the joined node answers");
    assert_eq!(node.identity, "box-a");
    assert_eq!(node.models, vec!["mock".to_string()]);

    let prompt: Vec<u32> = (400..432).collect();
    let params = GenParams::default();
    let (local, _) = mock_daemon().generate_tokens(prompt.clone(), params, 8, 0).unwrap();
    let placed = manager.place(1, params, 0, "chatml", "mock", &prompt, 8).expect("placed on the joined node");
    assert_eq!(placed, 0);
    let gen = manager.generate(1, 8, 60_000).expect("generated on the joined node");
    assert_eq!(gen.tokens, local, "the joined node serves what a local daemon does");
    manager.finish(1);

    join(head_addr, "box-b", 1);
    wait_for("a second node joined", || manager.load().len() == 2);
}

#[test]
fn a_node_dials_once_per_lane_and_serves_that_many_at_once() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let head_addr = listener.local_addr().unwrap().to_string();
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let manager = Arc::new(FleetManager::with_options(Vec::new(), PlacementPolicy::LoadAware, lease, 90, 1));
    let m = Arc::clone(&manager);
    std::thread::spawn(move || accept_nodes(listener, m, TOKEN.to_vec(), vec!["mock".into()]));
    join(head_addr, "box-c", 3);
    wait_for("three connections", || manager.load().len() == 1 && manager.connections(0) == 3);
    let params = GenParams::default();
    let prompts: Vec<Vec<u32>> = (0..3).map(|k| (100 * k..100 * k + 24).collect()).collect();
    for (k, p) in prompts.iter().enumerate() {
        manager.place(k as u64 + 1, params, 0, "chatml", "mock", p, 8).expect("placed");
    }
    let started = Instant::now();
    let runs: Vec<_> = (0..3)
        .map(|k| {
            let m = Arc::clone(&manager);
            std::thread::spawn(move || m.generate(k + 1, 8, 60_000).expect("generated").tokens)
        })
        .collect();
    let outs: Vec<Vec<u32>> = runs.into_iter().map(|r| r.join().unwrap()).collect();
    assert!(started.elapsed() < Duration::from_secs(30));
    for (k, p) in prompts.iter().enumerate() {
        let (local, _) = mock_daemon().generate_tokens(p.clone(), params, 8, 0).unwrap();
        assert_eq!(outs[k], local, "session {} matches a local daemon", k + 1);
        manager.finish(k as u64 + 1);
    }
}

#[test]
fn a_joined_node_keeps_a_connection_for_short_calls_when_its_head_asks() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let head_addr = listener.local_addr().unwrap().to_string();
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let manager = Arc::new(FleetManager::with_options(Vec::new(), PlacementPolicy::LoadAware, lease, 90, 1));
    let m = Arc::clone(&manager);
    std::thread::spawn(move || accept_nodes(listener, m, TOKEN.to_vec(), vec!["mock".into()]));

    // As `superfluid node join` does: one connection per lane, and one more the first time the
    // head asks.
    let (ask, asked) = std::sync::mpsc::sync_channel::<()>(1);
    let node = NodeAgent::new(Arc::new(mock_daemon_ticking(Duration::from_millis(50))), "box-d", "mock")
        .with_max_lanes(2)
        .with_auth(TOKEN.to_vec())
        .with_on_control_ask(move || {
            let _ = ask.try_send(());
        });
    for _ in 0..2 {
        keep_dialing(head_addr.clone(), node.connection());
    }
    asked.recv_timeout(Duration::from_secs(10)).expect("the head asks for a connection for short calls");
    keep_dialing(head_addr, node.connection());
    wait_for("both lanes' connections", || manager.load().len() == 1 && manager.connections(0) == 2);

    // Both lanes generating, a state operation does not wait for either to end.
    let params = GenParams::default();
    for k in 0..2u64 {
        let prompt: Vec<u32> = (100 * k as u32..100 * k as u32 + 24).collect();
        manager.place(k + 1, params, 0, "chatml", "mock", &prompt, 8).expect("placed");
    }
    let runs: Vec<_> = (0..2)
        .map(|k| {
            let m = Arc::clone(&manager);
            std::thread::spawn(move || {
                m.generate(k + 1, 2000, 60_000).expect("generated");
                Instant::now()
            })
        })
        .collect();
    std::thread::sleep(Duration::from_millis(300));
    manager
        .state_sync("mock", superfluid_proto::linkw::StateSyncReqMsg::Capabilities, &[])
        .expect("the node's capabilities");
    let answered = Instant::now();
    for run in runs {
        assert!(answered < run.join().unwrap(), "the state operation waited for a generation to end");
    }
}

#[test]
fn a_node_with_another_token_does_not_join() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let head_addr = listener.local_addr().unwrap().to_string();
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let manager = Arc::new(FleetManager::with_options(Vec::new(), PlacementPolicy::LoadAware, lease, 90, 1));
    let m = Arc::clone(&manager);
    std::thread::spawn(move || accept_nodes(listener, m, b"the head's token".to_vec(), vec!["mock".into()]));
    let sock = TcpStream::connect(&head_addr).unwrap();
    let ep = Endpoint::from_tcp_secure(sock, TOKEN, Role::Responder, Duration::from_secs(5));
    assert!(ep.is_err(), "the handshake with another token must fail");
    std::thread::sleep(Duration::from_millis(200));
    assert!(manager.load().is_empty(), "nothing joined");
}
