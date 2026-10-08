#![cfg(feature = "basert")]

use std::path::PathBuf;

use superfluid_daemon::{BundleCodec, Daemon, EngineHost, SessionStore, WorkerSpec};

fn model_path() -> Option<PathBuf> {
    let p = model_path_found()?;
    if p.extension().is_some_and(|e| e == "base") {
        superfluid_engine_ffi::libbasert::require()?;
    }
    Some(p)
}

fn model_path_found() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_MODEL") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    ["models/Qwen3-0.6B-Q4_K_M.base", "models/Qwen3-0.6B-Q4_K_M.gguf"]
        .iter()
        .map(|c| root.join(c))
        .find(|p| p.exists())
}

fn spawn_native_process(model: &std::path::Path) -> EngineHost {
    EngineHost::spawn_process(WorkerSpec {
        bin: env!("CARGO_BIN_EXE_superfluid-workerd").into(),
        args: vec![
            "--engine".into(),
            "native".into(),
            "--model".into(),
            model.to_string_lossy().into_owned(),
            "--max-context".into(),
            "4096".into(),
            "--max-batch".into(),
            "8".into(),
        ],
        stderr: None,
    })
    .expect("spawn native worker process")
}

#[test]
fn worker_process_crash_resumes_warm_from_park() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-procreal-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let park = dir.join("park");

    let host = spawn_native_process(&model);
    let pid = host.worker_pid().expect("process backend");
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let codec = BundleCodec::load(&model).expect("tokenizer load");
    let daemon = Daemon::with_park(store, host, Box::new(codec), 8, Some(park.clone()));

    let session = daemon.create(None, Default::default()).unwrap();
    let prompt: Vec<u32> = (0..64).map(|i| 1000 + i * 7).collect();
    daemon.append(session, None, prompt).unwrap();
    let first = daemon.generate(session, 8).unwrap();
    assert!(first.tokens_generated >= 1);
    assert_eq!(first.warm_prefix, 0, "first generation is cold");
    {
        let artifact = park.join(format!("{session}.park"));
        let mut ok = artifact.exists();
        for _ in 0..200 {
            if ok {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            ok = artifact.exists();
        }
        assert!(ok, "retire parked a sealed artifact (background writer)");
    }

    let _ = std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .expect("kill");

    let failed = daemon.generate(session, 4);
    assert!(failed.is_err(), "a murdered engine fails typed");

    let out = daemon.generate(session, 4).unwrap();
    assert!(out.tokens_generated >= 1);
    assert!(
        out.warm_prefix >= 16,
        "resume after worker murder is warm from the park artifact (got {})",
        out.warm_prefix
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn worker_respawn_reregisters_speculation() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-worker-spec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = spawn_native_process(&model);
    let pid = host.worker_pid().expect("process backend");
    let codec = BundleCodec::load(&model).expect("tokenizer load");
    let daemon = Daemon::with_options(
        store,
        host,
        Box::new(codec),
        superfluid_daemon::DaemonOptions {
            speculate: Some("prompt-lookup".into()),
            ..Default::default()
        },
    )
    .expect("options");
    let stats = daemon.sched_stats();
    let mut prompt = Vec::new();
    for _ in 0..6 {
        prompt.extend_from_slice(&[3000, 3005, 3010, 3015, 3020, 3025, 3030, 3035]);
    }
    let session = daemon.create(None, superfluid_daemon::GenParams::default()).unwrap();
    daemon.append(session, None, prompt).unwrap();
    daemon.generate(session, 24).unwrap();
    let before = stats.spec_accepted.load(std::sync::atomic::Ordering::Relaxed);
    assert!(before > 0, "speculating before the murder");
    let _ = std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .expect("kill");
    let failed = daemon.generate(session, 4);
    assert!(failed.is_err(), "a murdered engine fails typed");
    let out = daemon.generate(session, 24).unwrap();
    assert_eq!(out.tokens_generated, 24);
    assert!(daemon.speculation().is_some());
    let after = stats.spec_accepted.load(std::sync::atomic::Ordering::Relaxed);
    assert!(after > before, "drafts accepted on the respawned worker: {after} > {before}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fim_satellite_in_its_own_worker_serves_completions() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-fimsat-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let primary_host = EngineHost::spawn(|| {
        (superfluid_engine::MockEngine::new(superfluid_engine::EngineConfig::default()), None)
    })
    .unwrap();
    let primary = Daemon::new(
        SessionStore::open(&dir.join("wal.log")).unwrap(),
        primary_host,
        Box::new(superfluid_daemon::MockCodec),
        8,
    );
    let sat = Daemon::new(
        SessionStore::ephemeral(),
        spawn_native_process(&model),
        Box::new(BundleCodec::load(&model).expect("tokenizer load")),
        2,
    );
    primary
        .attach_fim_satellite("qwen3-fim", std::sync::Arc::new(sat))
        .expect("Qwen3's tokenizer renders FIM");
    let pre = "def add(a, b):\n    return ";
    let suf = "\n\nprint(add(1, 2))\n";
    let c1 = primary.complete(pre, suf, superfluid_daemon::codec::fim_mode::PSM, 12).unwrap();
    assert!(!c1.cached && !c1.expired);
    assert_eq!(c1.tokens.len(), 12);
    assert!(!c1.text.is_empty());
    let c2 = primary.complete(pre, suf, superfluid_daemon::codec::fim_mode::PSM, 12).unwrap();
    assert!(c2.cached, "the satellite's composition cache answers the repeat");
    assert_eq!(c2.tokens, c1.tokens);
    let primary_decoded =
        primary.sched_stats().decode_tokens.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(primary_decoded, 0, "the primary's engine never ran");
    assert!(primary.session_ids().is_empty());
    assert!(primary.metrics_text().contains("superfluid_fim_satellite_info{model=\"qwen3-fim\"} 1"));
    let _ = std::fs::remove_dir_all(&dir);
}
