//! A runtime's length limit through the daemon, over the executor.

use std::path::{Path, PathBuf};

use superfluid_daemon::codec::MockCodec;
use superfluid_daemon::runtime::EngineHost;
use superfluid_daemon::{Daemon, GenParams, SessionStore};
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives};

fn test_dir(tag: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("superfluid-executor-limits-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn daemon(wal: &Path, cfg: FakeConfig) -> Daemon {
    let store = SessionStore::open(wal).expect("open store");
    let host = EngineHost::spawn(move || (Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()), None)).expect("spawn");
    Daemon::with_park(store, host, Box::new(MockCodec), 8, None)
}

#[test]
fn a_prompt_that_fills_the_context_ends_with_length_and_the_daemon_serves_on() {
    let dir = test_dir("full");
    let d = daemon(&dir.join("wal.log"), FakeConfig { max_seq_len: 48, ..Default::default() });
    let full = d.create(None, GenParams::default()).unwrap();
    d.append(full, None, (0..48).collect()).unwrap();
    let out = d.generate(full, 8).expect("a full context is a finish, not a failed tick");
    assert_eq!((out.tokens_generated, out.finish), (0, superfluid_abi::finish::LENGTH));
    let near = d.create(None, GenParams::default()).unwrap();
    d.append(near, None, (100..147).collect()).unwrap();
    let out = d.generate(near, 8).unwrap();
    assert_eq!((out.tokens_generated, out.finish), (1, superfluid_abi::finish::LENGTH));
    let out = d.generate(near, 8).expect("generating again at the limit");
    assert_eq!((out.tokens_generated, out.finish), (0, superfluid_abi::finish::LENGTH));
    let _ = std::fs::remove_dir_all(&dir);
}
