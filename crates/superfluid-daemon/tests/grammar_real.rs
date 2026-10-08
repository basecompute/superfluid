//! Grammar-constrained decoding end to end on the real engine.

#![cfg(feature = "basert")]

use std::path::PathBuf;

use superfluid_daemon::{BundleCodec, Daemon, EngineHost, GenParams, NoCodec, SessionStore, TextCodec};
use superfluid_engine_ffi::{FfiEngineConfig, NativeEngine};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};

static MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

fn real_daemon(model: PathBuf) -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-grammar-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || {
        let engine = NativeEngine::load(FfiEngineConfig {
            model_path: model,
            max_context: 4096,
            max_batch_size: 8,
            seed_ttl_ticks: 64,
        })
        .expect("model load");
        let vocab = engine.vocab();
        let specs = vec![
            linkw::RingSpec { ring_id: TOKEN_RING_IN, kind: linkw::RingKind::Tokens, slot_bytes: 16 + 4096 * 4, slots: 64 },
            linkw::RingSpec { ring_id: TOKEN_RING_OUT, kind: linkw::RingKind::Tokens, slot_bytes: 16 + 1024 * 4, slots: 64 },
            linkw::RingSpec { ring_id: LOGITS_RING, kind: linkw::RingKind::Logits, slot_bytes: 16 + vocab * 4, slots: 8 },
        ];
        (engine, Some(specs))
    })
    .expect("spawn engine");
    Daemon::new(store, host, Box::new(NoCodec), 8)
}

#[test]
fn constrained_decode_yields_schema_valid_json() {
    let _g = MODEL_LOCK.lock().unwrap();
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let d = real_daemon(model.clone());
    let codec = BundleCodec::load(&model).expect("tokenizer");

    let schema = r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}"#;
    let prompt = codec.encode("Output a JSON object with the capital city of France.\n");

    let (tokens, expired) = d
        .generate_tokens_constrained(prompt, GenParams::default(), 64, 0, schema)
        .expect("constrained generate");
    assert!(!expired, "constrained generation expired");
    assert!(!tokens.is_empty(), "no tokens produced");

    let text = codec.decode(&tokens);
    let v: serde_json::Value =
        serde_json::from_str(text.trim()).unwrap_or_else(|e| panic!("not valid JSON ({e}): {text:?}"));
    assert!(v.is_object(), "not an object: {text:?}");
    assert!(v.get("city").and_then(|c| c.as_str()).is_some(), "no string 'city': {text:?}");
    assert_eq!(v.as_object().unwrap().len(), 1, "extra keys: {text:?}");
}

#[test]
fn concurrent_constrained_lanes_stay_isolated() {
    let _g = MODEL_LOCK.lock().unwrap();
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let d = std::sync::Arc::new(real_daemon(model.clone()));
    let codec = std::sync::Arc::new(BundleCodec::load(&model).expect("tokenizer"));

    let specs: Vec<(&str, &str, &str)> = vec![
        (
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}"#,
            "Output a JSON object with a city.\n",
            "city",
        ),
        (
            r#"{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"],"additionalProperties":false}"#,
            "Output a JSON object with a count.\n",
            "count",
        ),
    ];

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(specs.len()));
    let mut handles = Vec::new();
    for (schema, prompt, key) in specs {
        let d = std::sync::Arc::clone(&d);
        let codec = std::sync::Arc::clone(&codec);
        let b = std::sync::Arc::clone(&barrier);
        let (schema, prompt, key) = (schema.to_string(), prompt.to_string(), key.to_string());
        handles.push(std::thread::spawn(move || {
            let toks = codec.encode(&prompt);
            b.wait();
            let (out, expired) = d
                .generate_tokens_constrained(toks, GenParams::default(), 64, 0, &schema)
                .expect("constrained generate");
            assert!(!expired);
            let text = codec.decode(&out);
            let v: serde_json::Value = serde_json::from_str(text.trim())
                .unwrap_or_else(|e| panic!("[{key}] not JSON ({e}): {text:?}"));
            assert!(v.is_object(), "[{key}] not object: {text:?}");
            assert_eq!(v.as_object().unwrap().len(), 1, "[{key}] extra keys: {text:?}");
            assert!(v.get(&key).is_some(), "[{key}] wrong schema — got {text:?}");
        }));
    }
    for h in handles {
        h.join().expect("a constrained lane panicked");
    }
}

#[test]
fn undersized_budget_refuses_truncated_json() {
    let _g = MODEL_LOCK.lock().unwrap();
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let d = real_daemon(model.clone());
    let codec = BundleCodec::load(&model).expect("tokenizer");

    let schema = r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}"#;
    let toks = codec.encode("Output a JSON object with a city.\n");
    let r = d.generate_tokens_constrained(toks, GenParams::default(), 2, 0, schema);
    assert!(
        r.is_err(),
        "an undersized budget must refuse (grammar incomplete), got Ok: {:?}",
        r.ok().map(|(t, _)| codec.decode(&t))
    );
}
