//! Hybrid (Gated-DeltaNet, Qwen3.5/3.6) bundles through the daemon on the REAL engine (P2.5).

#![cfg(feature = "basert")]

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use superfluid_daemon::{BundleCodec, Daemon, EngineHost, GenParams, SessionStore};
use superfluid_engine_ffi::{FfiEngineConfig, NativeEngine};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};

fn hybrid_path() -> Option<PathBuf> {
    let p = hybrid_path_found()?;
    if p.extension().is_some_and(|e| e == "base") {
        superfluid_engine_ffi::libbasert::require()?;
    }
    Some(p)
}

fn hybrid_path_found() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_HYBRID") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let home = std::env::var("HOME").unwrap_or_default();
    [
        root.join("models/Qwen3.5-2B-Base-Q4.base"),
        PathBuf::from(&home).join(".cache/baseRT/models/Qwen/Qwen3.5-2B-Base/default-q4/model.base"),
        PathBuf::from(&home)
            .join("Library/Caches/baseRT/models/Qwen/Qwen3.5-2B-Base/default-q4/model.base"),
    ]
    .into_iter()
    .find(|p| p.exists())
}

fn spawn_hybrid(model: PathBuf) -> EngineHost {
    EngineHost::spawn(move || {
        let engine = NativeEngine::load(FfiEngineConfig {
            model_path: model,
            max_context: 4096,
            max_batch_size: 2,
            seed_ttl_ticks: 64,
        })
        .expect("model load");
        let vocab = engine.vocab();
        let specs = vec![
            linkw::RingSpec {
                ring_id: TOKEN_RING_IN,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + 4096 * 4,
                slots: 64,
            },
            linkw::RingSpec {
                ring_id: TOKEN_RING_OUT,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + 1024 * 4,
                slots: 64,
            },
            linkw::RingSpec {
                ring_id: LOGITS_RING,
                kind: linkw::RingKind::Logits,
                slot_bytes: 16 + vocab * 4,
                slots: 8,
            },
        ];
        (engine, Some(specs))
    })
    .expect("spawn engine worker")
}

#[test]
fn hybrid_parks_every_space_and_resumes_by_adoption_real() {
    let Some(model) = hybrid_path() else {
        eprintln!("SKIP: no hybrid GDN bundle (set BASERT_TEST_HYBRID)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-hybrid-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let park = dir.join("park");
    let wal = dir.join("wal.log");
    let vocab_hint: u32 = std::env::var("BASERT_TEST_HYBRID_VOCAB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(150_000);
    let base = if vocab_hint > 3300 { 3000 } else { 8 };
    let step = if vocab_hint > 3300 { 5 } else { 2 };
    let prompt: Vec<u32> = (0..48).map(|i| base + i * step).collect();

    let (a, first_tokens, covered) = {
        let store = SessionStore::open(&wal).unwrap();
        let codec = BundleCodec::load(&model).expect("tokenizer load");
        let host = spawn_hybrid(model.clone());
        let daemon = Daemon::with_park(store, host, Box::new(codec), 2, Some(park.clone()));
        let stats = daemon.sched_stats();
        let tokens_of = |session: u64| {
            let store = daemon.store();
            let store = store.lock().unwrap();
            store.session(session).unwrap().tokens.clone()
        };
        let a = daemon.create(None, GenParams::default()).unwrap();
        daemon.append(a, None, prompt.clone()).unwrap();
        let a1 = daemon.generate(a, 16).unwrap();
        assert_eq!(a1.warm_prefix, 0);
        assert_eq!(a1.tokens_generated, 16);
        let path = park.join(format!("{a}.park"));
        for _ in 0..200 {
            if path.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let parsed = superfluid_daemon::park::read(&park, a).expect("v3 artifact");
        assert!(parsed.multi_space);
        assert_eq!(parsed.spaces.len(), 2, "KV + GDN blob");
        assert!(parsed.covered > 0, "durable boundary exists");
        assert!(parsed.covered < 48, "cut inside the prompt, not at the head");
        assert_eq!(stats.unservable_park_refusals.load(Ordering::Relaxed), 0);
        assert_eq!(stats.parks_lossless.load(Ordering::Relaxed), 1);

        let a2 = daemon.generate(a, 8).unwrap();
        assert_eq!(a2.warm_prefix, parsed.covered, "adopted whole, exact length");
        assert_eq!(a2.tokens_generated, 8);
        assert_eq!(stats.resumes.load(Ordering::Relaxed), 1);
        assert_eq!(stats.unservable_cold_admissions.load(Ordering::Relaxed), 0);

        let b = daemon.create(None, GenParams::default()).unwrap();
        daemon.append(b, None, prompt.clone()).unwrap();
        daemon.generate(b, 16).unwrap();
        daemon.generate(b, 8).unwrap();
        assert_eq!(tokens_of(a), tokens_of(b), "adopt-resume is token-exact");
        (a, tokens_of(a), parsed.covered)
    };

    let store = SessionStore::open(&wal).unwrap();
    let codec = BundleCodec::load(&model).expect("tokenizer load");
    let host = spawn_hybrid(model);
    let daemon = Daemon::with_park(store, host, Box::new(codec), 2, Some(park.clone()));
    let out = daemon.generate(a, 4).unwrap();
    assert!(out.warm_prefix > covered, "resumed warm from the newest artifact: {}", out.warm_prefix);
    assert_eq!(out.tokens_generated, 4);
    let reference = daemon.create(None, GenParams::default()).unwrap();
    daemon.append(reference, None, prompt).unwrap();
    daemon.generate(reference, 16).unwrap();
    daemon.generate(reference, 8).unwrap();
    daemon.generate(reference, 4).unwrap();
    let store = daemon.store();
    let store = store.lock().unwrap();
    assert_eq!(&store.session(a).unwrap().tokens[..first_tokens.len()], &first_tokens[..]);
    assert_eq!(store.session(a).unwrap().tokens, store.session(reference).unwrap().tokens);
    let _ = std::fs::remove_dir_all(&dir);
}
