//! An engine that cannot take a tick's admissions (NeedsReplan) gets fewer of
//! them, not a failed tick.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use superfluid_daemon::{Daemon, DaemonOptions, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_executor::fake::{FakeConfig, FakePrimitives};
use superfluid_executor::{Executor, ExecutorConfig};

fn daemon(cfg: FakeConfig, max_lanes: usize) -> Arc<Daemon> {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-admit-replan-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || (Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()), None))
        .expect("spawn");
    Arc::new(Daemon::with_options(store, host, Box::new(MockCodec), DaemonOptions { max_lanes, ..Default::default() }).unwrap())
}

#[test]
fn a_burst_seeded_from_one_entry_is_admitted_in_turns_when_sequences_run_short() {
    // A buffer per sequence, as llama.cpp's streams: four sequence ids in
    // all, so four lanes copying from one cached entry need one id too many.
    let cfg = FakeConfig {
        max_seqs: 4,
        takeover_preferred: true,
        copy_shares_cells: false,
        page: 16,
        step_delay: std::time::Duration::from_millis(10),
        ..FakeConfig::default()
    };
    let d = daemon(cfg, 4);
    let stats = d.sched_stats();
    let prefix: Vec<u32> = (0..1024).map(|t| t % 900 + 100).collect();
    let session = |i: u32| {
        let mut p = prefix.clone();
        p.extend([2000 + i, 2100 + i, 2200 + i, 2300 + i]);
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, p).unwrap();
        s
    };

    // The leader runs a while; the burst sharing its prefix queues behind
    // it and is admitted in the one tick after it retires, all four copying
    // from its entry: one sequence id more than there are.
    let leader = session(99);
    let dl = Arc::clone(&d);
    let leader = std::thread::spawn(move || dl.generate(leader, 48));
    while stats.lanes_active.load(Ordering::Relaxed) < 1 {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let burst: Vec<_> = (0..4u32)
        .map(|i| {
            let s = session(i);
            let dd = Arc::clone(&d);
            std::thread::spawn(move || dd.generate(s, 4))
        })
        .collect();
    leader.join().unwrap().unwrap();
    for h in burst {
        let out = h.join().unwrap().expect("no request fails for want of a sequence id");
        assert_eq!(out.tokens_generated, 4);
    }
    assert!(
        stats.admits_deferred.load(Ordering::Relaxed) > 0,
        "the burst did not fit one tick, and some of it waited for the next"
    );
    assert!(
        stats.warm_prefix_tokens.load(Ordering::Relaxed) >= 4 * 1024,
        "every request in the burst started from the shared prefix"
    );
}

// The clock alone repeats within a microsecond on macOS, and two tests on one
// log are refused while both stores are open.
fn unique() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    format!("{t}-{}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}
