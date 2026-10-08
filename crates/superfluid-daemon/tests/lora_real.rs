#![cfg(feature = "basert")]

use superfluid_daemon::{BundleCodec, Daemon, EngineHost, EventBody, GenParams, SessionStore, TextCodec};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn spawn(model: PathBuf) -> Daemon {
    let dir = std::env::temp_dir().join(format!("lora-real-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let mp = model;
    let host = EngineHost::spawn(move || {
        let e = superfluid_engine_ffi::NativeEngine::load(superfluid_engine_ffi::FfiEngineConfig {
            model_path: mp,
            max_context: 2048,
            max_batch_size: 2,
            seed_ttl_ticks: 64,
        })
        .expect("engine load");
        let vocab = e.vocab();
        let specs = vec![
            linkw::RingSpec { ring_id: TOKEN_RING_IN, kind: linkw::RingKind::Tokens, slot_bytes: 16 + 4096 * 4, slots: 64 },
            linkw::RingSpec { ring_id: TOKEN_RING_OUT, kind: linkw::RingKind::Tokens, slot_bytes: 16 + 1024 * 4, slots: 64 },
            linkw::RingSpec { ring_id: LOGITS_RING, kind: linkw::RingKind::Logits, slot_bytes: 16 + vocab * 4, slots: 8 },
        ];
        (e, Some(specs))
    })
    .expect("spawn engine");
    Daemon::new(store, host, Box::new(BundleCodec::load(&root().join("models/Llama-3.2-1B-cuda-q4mix.base")).unwrap()), 2)
}

fn run(d: &Daemon, prompt: &[u32]) -> Vec<u32> {
    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, Some("p".into()), prompt.to_vec()).unwrap();
    let out = d.generate(session, 12).expect("generate");
    let mut toks = Vec::new();
    for e in &out.events {
        if let EventBody::Generated { span, .. } = &e.body {
            toks.extend(span.iter().copied());
        }
    }
    toks
}

#[test]
fn real_adapter_file_changes_generation_and_unload_restores() {
    let model = root().join("models/Llama-3.2-1B-cuda-q4mix.base");
    let adapter = root().join("models/test-lora-adapter.base");
    if !model.exists() || !adapter.exists() {
        eprintln!("::warning::SKIP lora_real — model or adapter fixture missing (see file header to regenerate)");
        return;
    }
    let d = spawn(model);
    let codec = BundleCodec::load(&root().join("models/Llama-3.2-1B-cuda-q4mix.base")).unwrap();
    let prompt = codec.encode("The capital of France is");

    let baseline = run(&d, &prompt);
    assert!(baseline.len() >= 4, "baseline generated");

    let id = d.lora_load(adapter.to_str().unwrap()).expect("lora_load accepts the converted adapter");
    assert!(id.is_some_and(|s| !s.is_empty()), "an adapter id is active");
    let adapted = run(&d, &prompt);
    assert_ne!(adapted, baseline, "the adapter must change generation");

    let id = d.lora_unload().expect("lora_unload");
    assert!(id.is_none() || id.as_deref() == Some(""), "no adapter after unload");
    let restored = run(&d, &prompt);
    assert_eq!(restored, baseline, "unload must restore the baseline exactly");
}
