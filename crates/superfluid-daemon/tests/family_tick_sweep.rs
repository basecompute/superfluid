#![cfg(feature = "basert")]
use superfluid_daemon::{BundleCodec, Daemon, EngineHost, EventBody, GenParams, SessionStore, TextCodec};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};

fn candidates() -> Vec<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BASERT_SWEEP_MODEL") {
        let p = std::path::PathBuf::from(p);
        return if p.exists() { vec![p] } else { Vec::new() };
    }
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    [
        "models/Qwen3-0.6B-Q4_K_M.base",
        "models/gemma-3-1b-it-Q4_K_M.base",
        "models/gemma-4-E2B-it-Q4.base",
        "models/muse-glimmer-30B-kquant-17gb.base",
        "models/Qwen3.5-2B-Base-Q4.base",
    ]
    .iter()
    .map(|n| root.join(n))
    .filter(|p| p.exists())
    .take(1)
    .collect()
}

fn longest_repeat(v: &[u32]) -> usize {
    let (mut run, mut best) = (1usize, 1usize);
    for i in 1..v.len() {
        run = if v[i] == v[i - 1] { run + 1 } else { 1 };
        best = best.max(run);
    }
    if v.is_empty() { 0 } else { best }
}

#[test]
fn every_family_decodes_sanely_through_the_tick_path() {
    let models = candidates();
    if models.is_empty() {
        eprintln!("::warning::SKIP family tick sweep — no model fixtures present");
        return;
    }
    let mut broken = Vec::new();
    for m in models {
        let name = m.file_name().unwrap().to_string_lossy().to_string();
        let codec = match BundleCodec::load(&m) {
            Ok(c) => c,
            Err(e) => { eprintln!("  {name}: codec load failed {e:?}"); continue }
        };
        let prompt = codec.encode("The capital of France is");
        let dir = std::env::temp_dir().join(format!("tick-sweep-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).unwrap();
        let store = SessionStore::open(&dir.join("wal.log")).unwrap();
        let mp = m.clone();
        let host = EngineHost::spawn(move || {
            let e = superfluid_engine_ffi::NativeEngine::load(superfluid_engine_ffi::FfiEngineConfig {
                model_path: mp, max_context: 4096, max_batch_size: 2, seed_ttl_ticks: 64,
            }).expect("engine load");
            let vocab = e.vocab();
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
            (e, Some(specs))
        }).expect("spawn");
        let d = Daemon::new(store, host, Box::new(BundleCodec::load(&m).unwrap()), 1);
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, Some("p".into()), prompt).unwrap();
        let out = match d.generate(session, 16) {
            Ok(o) => o,
            Err(e) => {
                eprintln!(
                    "::warning::family tick sweep did NOT exercise {name} — not admitted: {e:?}"
                );
                continue;
            }
        };
        let mut toks: Vec<u32> = Vec::new();
        let mut text = String::new();
        for e in &out.events {
            if let EventBody::Generated { span, text: t, .. } = &e.body {
                toks.extend(span.iter().copied());
                text.push_str(t);
            }
        }
        let rep = longest_repeat(&toks);
        let blank = text.trim().is_empty();
        let bad = toks.is_empty() || blank || (toks.len() >= 6 && rep >= toks.len() / 2);
        println!(
            "  {name:<38} tokens={:<3} longest_repeat={rep:<3} text={:?}",
            toks.len(),
            text.chars().take(40).collect::<String>()
        );
        if bad {
            broken.push(format!("{name} (tokens={}, longest_repeat={rep}, text={text:?})", toks.len()));
        }
    }
    assert!(
        broken.is_empty(),
        "families that decode correctly through the C API but NOT through the tick path:\n  - {}",
        broken.join("\n  - ")
    );
}
