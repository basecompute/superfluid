//! Pipeline-parallel fleet.

use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use superfluid_daemon::fleet::PipelineFleet;
use superfluid_daemon::{Daemon, EngineHost, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};

const N_LAYERS: u32 = 24;

static DIRS: AtomicU64 = AtomicU64::new(0);

fn mock_daemon() -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-pipe-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

fn spawn_node(identity: &'static str) -> (String, Vec<u8>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let mut node = superfluid_daemon::nodeagent::NodeAgent::new(daemon, identity, "mock");
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            if node.serve(ep).is_err() {
                break;
            }
        }
    });
    (addr, Vec::new())
}

#[test]
fn two_stage_pipeline_matches_single_full_node() {
    let context: Vec<u32> = (100..112).collect();
    let max = 8u32;

    let mut full = PipelineFleet::connect(vec![spawn_node("full")], N_LAYERS).unwrap();
    assert_eq!(full.boundaries(), &[N_LAYERS]);
    let reference = full.generate(&context, max).unwrap();
    assert_eq!(reference.len(), max as usize);

    let mut pipe =
        PipelineFleet::connect(vec![spawn_node("stage0"), spawn_node("stage1")], N_LAYERS).unwrap();
    assert_eq!(pipe.boundaries(), &[12, 24]);
    let piped = pipe.generate(&context, max).unwrap();

    assert_eq!(piped, reference, "2-stage pipeline diverged from the full-node output");
}

#[test]
fn three_stage_pipeline_matches_single_full_node() {
    let context: Vec<u32> = (7..15).collect();
    let max = 6u32;

    let mut full = PipelineFleet::connect(vec![spawn_node("f")], N_LAYERS).unwrap();
    let reference = full.generate(&context, max).unwrap();

    let mut pipe = PipelineFleet::connect(
        vec![spawn_node("s0"), spawn_node("s1"), spawn_node("s2")],
        N_LAYERS,
    )
    .unwrap();
    assert_eq!(pipe.boundaries(), &[8, 16, 24]);
    let piped = pipe.generate(&context, max).unwrap();

    assert_eq!(piped, reference, "3-stage pipeline diverged from the full-node output");
}
