//! Robustness regression.

use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use superfluid_daemon::fleet::FleetHead;
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::{Daemon, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkf::Lease;

static DIRS: AtomicU64 = AtomicU64::new(0);

fn mock_daemon() -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-wedge-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

#[test]
fn silent_connection_does_not_wedge_the_node() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let mut node = NodeAgent::new(daemon, "node-w", "mock")
            .with_timeouts(Duration::from_millis(300), Duration::from_secs(2));
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let _ = node.serve(ep);
        }
    });

    let _silent = TcpStream::connect(&addr).unwrap();

    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let params = GenParams::default();
    let prompt: Vec<u32> = (10..26).collect();
    let mut head = FleetHead::connect(&addr, Vec::new()).expect("connect past the silent peer");
    let session = 1u64;
    head.assign(session, params, 0, "raw", lease).unwrap();
    head.append(session, &prompt).unwrap();
    let gen = head.generate(session, 8, 60_000).unwrap();
    assert!(!gen.tokens.is_empty(), "node served the real head after the silent one");
    head.wal().assert_single_contiguous_streams();
}

#[test]
fn idle_drop_is_transparent_to_the_caller() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let mut node = NodeAgent::new(daemon, "node-i", "mock")
            .with_timeouts(Duration::from_millis(300), Duration::from_millis(400));
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let _ = node.serve(ep);
        }
    });

    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let params = GenParams::default();
    let prompt: Vec<u32> = (10..26).collect();
    let mut head = FleetHead::connect(&addr, Vec::new()).unwrap();
    let session = 5u64;
    head.assign(session, params, 0, "raw", lease).unwrap();
    head.append(session, &prompt).unwrap();
    let g1 = head.generate(session, 8, 60_000).unwrap();

    std::thread::sleep(Duration::from_millis(700));

    let g2 = head.generate(session, 8, 60_000).expect("generate after idle drop");
    assert_eq!(g2.tokens, g1.tokens, "transparent resume produced different output");
    head.wal().assert_single_contiguous_streams();
}

#[test]
fn node_rejects_wrong_credential_and_non_head() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let mut node = NodeAgent::new(daemon, "node-auth", "mock")
            .with_timeouts(Duration::from_millis(500), Duration::from_secs(5))
            .with_auth(b"the-shared-secret".to_vec());
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp_secure(
                sock,
                b"the-shared-secret",
                superfluid_linkf::Role::Responder,
                Duration::from_secs(5),
            );
            let Ok(ep) = ep else { continue };
            let _ = node.serve(ep);
        }
    });

    assert!(
        FleetHead::connect(&addr, b"wrong".to_vec()).is_err(),
        "node accepted a bad credential"
    );
    assert!(FleetHead::connect(&addr, Vec::new()).is_err());
    let mut head = FleetHead::connect(&addr, b"the-shared-secret".to_vec())
        .expect("correct credential must be accepted");
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    head.assign(1, GenParams::default(), 0, "raw", lease).unwrap();
    head.append(1, &(10..26).collect::<Vec<u32>>()).unwrap();
    assert!(!head.generate(1, 8, 60_000).unwrap().tokens.is_empty());
}
