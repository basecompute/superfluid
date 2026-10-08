//! Fleet reconnect recovery.

use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use superfluid_daemon::fleet::FleetHead;
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::{Daemon, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkf::Lease;

static DIRS: AtomicU64 = AtomicU64::new(0);

fn mock_daemon() -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fleet-rec-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

fn spawn_node() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let mut node = NodeAgent::new(daemon, "node-a", "mock");
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            if node.serve(ep).is_err() {
                break;
            }
        }
    });
    addr
}

#[test]
fn reconnect_holds_session_and_resumes_generation() {
    let addr = spawn_node();
    let lease = Lease {
        duration_ms: 60_000,
        renew_by_ms: 40_000,
    };
    let params = GenParams::default();
    let prompt: Vec<u32> = (300..340).collect();
    let session = 11u64;

    let mut head = FleetHead::connect(&addr, Vec::new()).unwrap();
    head.assign(session, params, 0, "chatml", lease).unwrap();
    head.append(session, &prompt).unwrap();
    let g1 = head.generate(session, 8, 60_000).unwrap();
    assert!(!g1.tokens.is_empty());
    let committed_before = head.wal().total_committed();

    head.reconnect(&addr, Vec::new()).expect("reconnect");
    assert_eq!(
        head.wal().total_committed(),
        committed_before,
        "reconnect committed phantom bytes"
    );

    let g2 = head.generate(session, 8, 60_000).unwrap();
    assert_eq!(g2.tokens, g1.tokens, "resumed generation diverged");
    head.wal().assert_single_contiguous_streams();
}
