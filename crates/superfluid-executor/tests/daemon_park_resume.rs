//! The daemon's park / resume path end to end over the executor.

use std::path::{Path, PathBuf};

use superfluid_daemon::codec::MockCodec;
use superfluid_daemon::runtime::EngineHost;
use superfluid_daemon::{Daemon, GenParams, SessionStore};
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives};

fn test_dir(tag: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let d = std::env::temp_dir().join(format!("superfluid-executor-{tag}-{}-{nanos}-{n}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn daemon(wal: &Path, park: Option<&Path>, cfg: FakeConfig) -> Daemon {
    let store = SessionStore::open(wal).expect("open store");
    let host = EngineHost::spawn(move || {
        (Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()), None)
    })
    .expect("spawn");
    Daemon::with_park(store, host, Box::new(MockCodec), 8, park.map(|p| p.to_path_buf()))
}

fn tokens_of(d: &Daemon, session: u64) -> Vec<u32> {
    let store = d.store();
    let store = store.lock().unwrap();
    store.session(session).unwrap().tokens.clone()
}

fn park_resume(cfg: FakeConfig, min_warm: u64, tag: &str) {
    let run_life1 = |dir: &Path, park: Option<&Path>| -> u64 {
        let d = daemon(&dir.join("wal.log"), park, cfg.clone());
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, (0..32).collect()).unwrap();
        let out = d.generate(session, 8).unwrap();
        assert_eq!(out.warm_prefix, 0, "first generation is cold");
        assert_eq!(out.tokens_generated, 8);
        session
    };

    let warm_dir = test_dir(&format!("{tag}-warm"));
    let park = warm_dir.join("park");
    let session = run_life1(&warm_dir, Some(&park));
    assert!(park.join(format!("{session}.park")).exists(), "retire parked a sealed artifact");

    let cold_dir = test_dir(&format!("{tag}-cold"));
    let cold_session = run_life1(&cold_dir, None);

    let (warm, warm_tokens) = {
        let d = daemon(&warm_dir.join("wal.log"), Some(&park), cfg.clone());
        let out = d.generate(session, 4).unwrap();
        (out, tokens_of(&d, session))
    };
    let (cold, cold_tokens) = {
        let d = daemon(&cold_dir.join("wal.log"), None, cfg.clone());
        let out = d.generate(cold_session, 4).unwrap();
        (out, tokens_of(&d, cold_session))
    };
    assert!(warm.warm_prefix >= min_warm, "restart resumed warm from the park artifact (got {})", warm.warm_prefix);
    assert_eq!(cold.warm_prefix, 0, "control restart is cold");
    assert_eq!(warm_tokens, cold_tokens, "resume-from-park continues the exact stream");
}

#[test]
fn park_resume_survives_daemon_restart_attention_only() {
    park_resume(FakeConfig::default(), 16, "attention");
}

#[test]
fn park_resume_survives_daemon_restart_recurrent() {
    park_resume(FakeConfig { truncate_partial: false, page: 1, ..Default::default() }, 39, "recurrent");
}

#[test]
fn corrupt_park_artifact_resumes_cold() {
    let dir = test_dir("corrupt");
    let wal = dir.join("wal.log");
    let park = dir.join("park");
    let session = {
        let d = daemon(&wal, Some(&park), FakeConfig::default());
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, (0..32).collect()).unwrap();
        d.generate(session, 8).unwrap();
        session
    };
    let artifact = park.join(format!("{session}.park"));
    let mut bytes = std::fs::read(&artifact).unwrap();
    assert!(bytes.len() > 64);
    let mid = 24 + (bytes.len() - 24) / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&artifact, &bytes).unwrap();

    let d = daemon(&wal, Some(&park), FakeConfig::default());
    let out = d.generate(session, 4).unwrap();
    assert_eq!(out.warm_prefix, 0, "corrupt artifact resumes cold");
    assert_eq!(out.tokens_generated, 4);
}
