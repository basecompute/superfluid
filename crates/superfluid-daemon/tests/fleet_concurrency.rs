//! Concurrency on the fleet edge.

use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use superfluid_daemon::fleet::FleetHead;
use superfluid_daemon::fleet_manager::{FleetManager, NodeSpec, PlacementPolicy};
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::{Daemon, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkf::Lease;

static DIRS: AtomicU64 = AtomicU64::new(0);

fn spec(addr: String) -> NodeSpec {
    NodeSpec { addr, auth: Vec::new() }
}

fn mock_daemon() -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fleetc-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

fn reference(prompt: &[u32], max: u32) -> Vec<u32> {
    let (t, _) = mock_daemon().generate_tokens(prompt.to_vec(), GenParams::default(), max, 0).unwrap();
    t
}

fn spawn_node(identity: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let node = NodeAgent::new(daemon, identity, "mock")
            .with_timeouts(Duration::from_secs(5), Duration::from_secs(2));
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    addr
}

fn lease() -> Lease {
    Lease { duration_ms: 120_000, renew_by_ms: 90_000 }
}

#[test]
fn concurrent_generations_across_distinct_nodes() {
    const N: usize = 4;
    let addrs: Vec<String> = (0..N).map(|_| spawn_node("node-multi")).collect();
    let prompts: Vec<Vec<u32>> = (0..N).map(|i| (i as u32 * 50..i as u32 * 50 + 20).collect()).collect();
    let refs: Vec<Vec<u32>> = prompts.iter().map(|p| reference(p, 16)).collect();

    let barrier = Arc::new(Barrier::new(N));
    let mut handles = Vec::new();
    for (i, (addr, prompt)) in addrs.iter().zip(prompts.iter()).enumerate() {
        let _ = i;
        let addr = addr.clone();
        let prompt = prompt.clone();
        let b = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            let mut head = FleetHead::connect(&addr, Vec::new()).unwrap();
            head.assign(1, GenParams::default(), 0, "raw", lease()).unwrap();
            head.append(1, &prompt).unwrap();
            b.wait();
            head.generate(1, 16, 120_000).unwrap().tokens
        }));
    }
    for (i, h) in handles.into_iter().enumerate() {
        assert_eq!(h.join().unwrap(), refs[i], "node {i} diverged under concurrent load");
    }
}

#[test]
fn concurrent_heads_to_one_node_are_all_served() {
    const K: usize = 4;
    let addr = spawn_node("node-shared");
    let prompts: Vec<Vec<u32>> = (0..K).map(|i| (i as u32 * 7 + 1..i as u32 * 7 + 15).collect()).collect();
    let refs: Vec<Vec<u32>> = prompts.iter().map(|p| reference(p, 12)).collect();

    let barrier = Arc::new(Barrier::new(K));
    let mut handles = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let addr = addr.clone();
        let prompt = prompt.clone();
        let b = Arc::clone(&barrier);
        let session = (i as u64) + 1;
        handles.push(std::thread::spawn(move || {
            b.wait();
            let mut head = FleetHead::connect(&addr, Vec::new()).unwrap();
            head.assign(session, GenParams::default(), 0, "raw", lease()).unwrap();
            head.append(session, &prompt).unwrap();
            head.generate(session, 12, 120_000).unwrap().tokens
        }));
    }
    for (i, h) in handles.into_iter().enumerate() {
        assert_eq!(h.join().unwrap(), refs[i], "head {i} got wrong output from the shared node");
    }
}

fn spawn_counting_node(identity: &'static str) -> (String, Arc<AtomicU64>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let accepted = Arc::new(AtomicU64::new(0));
    let seen = Arc::clone(&accepted);
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let node = NodeAgent::new(daemon, identity, "mock")
            .with_timeouts(Duration::from_secs(5), Duration::from_secs(30));
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            seen.fetch_add(1, Ordering::Relaxed);
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    (addr, accepted)
}

#[test]
fn concurrent_generations_on_one_node_open_more_than_one_connection() {
    let (addr, accepted) = spawn_counting_node("pool-node");
    let prompt: Vec<u32> = (0..24).collect();
    const MAX_NEW: u64 = 32;
    const N: u64 = 4;

    let mgr = Arc::new(FleetManager::with_options(
        vec![spec(addr)],
        PlacementPolicy::LeastLoaded,
        lease(),
        superfluid_daemon::fleet_manager::DEFAULT_POOL_HIGH_PCT,
        N as usize,
    ));
    for s in 0..N {
        mgr.place(s + 1, GenParams::default(), 0, "raw", "mock", &prompt, MAX_NEW)
            .expect("placement");
    }

    let barrier = Arc::new(Barrier::new(N as usize));
    let handles: Vec<_> = (0..N)
        .map(|s| {
            let mgr = Arc::clone(&mgr);
            let b = Arc::clone(&barrier);
            std::thread::spawn(move || {
                b.wait();
                mgr.generate(s + 1, MAX_NEW, 120_000).map(|g| g.tokens.len())
            })
        })
        .collect();
    for h in handles {
        let r = h.join().unwrap();
        assert!(r.is_ok(), "every concurrent generation must be served: {r:?}");
    }

    let conns = accepted.load(Ordering::Relaxed);
    assert!(
        conns > 1,
        "concurrent work on one node still used a single connection ({conns}); \
         the node's extra lanes are unreachable"
    );
    assert!(
        conns <= N,
        "opened {conns} connections for {N} concurrent generations — the pool \
         should be bounded, not unbounded"
    );
}
