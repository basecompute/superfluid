//! A running tick ends at the engine's next step boundary when the daemon
//! needs it to (a cancel, a chat to admit), not when its plan runs out.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use superfluid_daemon::{qos, Daemon, DaemonOptions, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_executor::fake::{FakeConfig, FakePrimitives};
use superfluid_executor::{Executor, ExecutorConfig};

fn daemon(max_lanes: usize) -> Arc<Daemon> {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!("superfluid-tick-yield-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    // Steps of 512 prompt tokens at 200 ms each: a 4096-token prefill tick
    // runs 1.6 s whole.
    let cfg = FakeConfig {
        prefill_step_tokens: 512,
        step_delay: Duration::from_millis(200),
        max_seq_len: 16384,
        ..FakeConfig::default()
    };
    let host = EngineHost::spawn(move || (Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()), None)).expect("spawn");
    let opts = DaemonOptions { max_lanes, prefill_budget: 4096, ..Default::default() };
    Arc::new(Daemon::with_options(store, host, Box::new(MockCodec), opts).unwrap())
}

#[test]
fn a_cancel_ends_a_long_prefill_tick_within_a_step() {
    let d = daemon(1);
    let stats = d.sched_stats();
    let s = d.create(None, GenParams::default()).unwrap();
    d.set_qos(s, qos::INTERACTIVE_CHAT, false).unwrap();
    d.append(s, None, (0..4000).map(|t| t % 900 + 50).collect()).unwrap();
    let dd = Arc::clone(&d);
    let running = std::thread::spawn(move || dd.generate(s, 4));
    while stats.lanes_active.load(Ordering::Relaxed) < 1 {
        assert!(!running.is_finished(), "never admitted: {:?}", running.join());
        std::thread::sleep(Duration::from_millis(1));
    }
    // Well inside the prompt's one prefill tick (8 steps, 1.6 s whole):
    // about 1.15 s of it is left.
    std::thread::sleep(Duration::from_millis(450));
    let t0 = Instant::now();
    d.cancel_registry().cancel(s);
    let _ = running.join().unwrap();
    let took = t0.elapsed();
    assert!(took < Duration::from_millis(800), "stopped within a step or two, not the rest of a 1.6 s tick: {took:?}");
}
