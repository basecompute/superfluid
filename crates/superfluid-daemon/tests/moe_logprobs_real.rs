#![cfg(feature = "basert")]

use superfluid_daemon::{BundleCodec, Daemon, EngineHost, EventBody, GenParams, SessionStore, TextCodec};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};
use std::path::PathBuf;

fn model() -> Option<PathBuf> {
    let p = model_found()?;
    if p.extension().is_some_and(|e| e == "base") {
        superfluid_engine_ffi::libbasert::require()?;
    }
    Some(p)
}

fn model_found() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_MODEL_MOE") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models/tiny-qwen35moe-q4mix.base");
    p.exists().then_some(p)
}

#[test]
fn moe_hybrid_logprobs_ride_the_decomposed_host_logit_path() {
    let Some(m) = model() else {
        eprintln!("::warning::SKIP moe_hybrid_logprobs — no tiny MoE fixture (tools/make_tiny_hybrid_fixture.py)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("moe-lp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let mp = m.clone();
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
    let d = Daemon::new(store, host, Box::new(BundleCodec::load(&m).unwrap()), 2);
    let codec = BundleCodec::load(&m).unwrap();
    let prompt = codec.encode("The capital of France is");

    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, Some("p".into()), prompt).unwrap();
    let extras = superfluid_daemon::scheduler::GenExtras {
        want_logprobs: true,
        top_logprobs: 3,
        ..Default::default()
    };
    let mut lps: Vec<superfluid_daemon::scheduler::TokenLogprob> = Vec::new();
    let out = d
        .generate_streaming_ex_lp(session, 12, extras, |_| Ok(()), |batch| lps.extend(batch.iter().cloned()))
        .expect("logprobs generation on a MoE-hybrid");
    let mut toks: Vec<u32> = Vec::new();
    for e in &out.events {
        if let EventBody::Generated { span, .. } = &e.body {
            toks.extend(span.iter().copied());
        }
    }
    assert!(!toks.is_empty(), "no tokens generated");
    assert_eq!(lps.len(), toks.len(), "one logprob per generated token");
    for (lp, tok) in lps.iter().zip(&toks) {
        assert_eq!(lp.token, *tok, "logprob entries follow the generated stream");
        assert!(lp.logprob.is_finite() && lp.logprob <= 0.0, "log-probability sanity: {}", lp.logprob);
        assert_eq!(lp.top.len(), 3, "top-k alternatives present");
        assert!(
            lp.top.iter().all(|&(_, l)| l <= lp.logprob + 1e-3),
            "chosen token must be top-1 under greedy: {:?}",
            lp
        );
    }
}
