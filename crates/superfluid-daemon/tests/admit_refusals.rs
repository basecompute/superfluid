//! An admit the engine refuses fails that request, never the tick.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use superfluid_abi::{LaneAdmit, Status};
use superfluid_daemon::{Daemon, DaemonError, DaemonOptions, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};

static DIRS: AtomicU64 = AtomicU64::new(0);

fn refusing_daemon() -> Daemon {
    let dir = std::env::temp_dir().join(format!("superfluid-admit-refusals-{}-{}", std::process::id(), DIRS.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    fn thirty_seven(a: &LaneAdmit) -> bool {
        a.prompt.count == 37
    }
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()).refusing_admits(thirty_seven, Status::RejectCertUnmatched), None))
        .expect("spawn");
    Daemon::with_options(store, host, Box::new(MockCodec), DaemonOptions { max_lanes: 4, ..Default::default() }).unwrap()
}

#[test]
fn a_refused_admit_fails_its_own_request_and_the_others_run_on() {
    let d = Arc::new(refusing_daemon());
    let run = |prompt: u32, seed: u32| {
        let d = Arc::clone(&d);
        std::thread::spawn(move || {
            let s = d.create(None, GenParams::default()).unwrap();
            d.append(s, None, (seed..seed + prompt).collect()).unwrap();
            d.generate(s, 24)
        })
    };
    let others: Vec<_> = (0..3).map(|i| run(40, 1000 * (i + 1))).collect();
    let refused = run(37, 9000);
    match refused.join().unwrap() {
        Err(DaemonError::Generation(why)) => {
            assert!(why.contains("the engine refused to admit this request"), "{why}");
            assert!(why.contains("RejectCertUnmatched"), "{why}");
        }
        other => panic!("the refused request must fail with the engine's reason, got {other:?}"),
    }
    for o in others {
        let out = o.join().unwrap().expect("a lane the engine took runs to the end");
        assert_eq!(out.tokens_generated, 24);
    }
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..16).collect()).unwrap();
    assert_eq!(d.generate(s, 8).unwrap().tokens_generated, 8);
}
