#![cfg(feature = "basert")]

use superfluid_daemon::codec::ChatMlCodec;
use superfluid_daemon::{Daemon, DaemonOptions, EngineHost, EventBody, GenParams, SessionStore};
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
    if let Ok(p) = std::env::var("BASERT_TEST_MODEL") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    ["models/Qwen3-0.6B-Q4_0.base", "models/Qwen3.5-2B-Base-Q4.base"]
        .iter()
        .map(|c| root.join(c))
        .find(|p| p.exists())
}

#[test]
fn provisional_deltas_stream_at_token_cadence() {
    let Some(m) = model() else {
        eprintln!("::warning::SKIP provisional_deltas_stream_at_token_cadence — no fixture");
        return;
    };
    let codec = match ChatMlCodec::load(&m) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("::warning::SKIP provisional_deltas_stream_at_token_cadence — codec: {e:?}");
            return;
        }
    };
    let dir = std::env::temp_dir().join(format!("stream-delta-real-{}", std::process::id()));
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
            linkw::RingSpec { ring_id: TOKEN_RING_IN, kind: linkw::RingKind::Tokens, slot_bytes: 16 + 2048 * 4, slots: 64 },
            linkw::RingSpec { ring_id: TOKEN_RING_OUT, kind: linkw::RingKind::Tokens, slot_bytes: 16 + 1024 * 4, slots: 64 },
            linkw::RingSpec { ring_id: LOGITS_RING, kind: linkw::RingKind::Logits, slot_bytes: 16 + vocab * 4, slots: 8 },
        ];
        (e, Some(specs))
    })
    .expect("spawn engine");
    let d = Daemon::with_options(store, host, Box::new(codec), DaemonOptions { max_lanes: 2, ..Default::default() })
        .unwrap();

    let session = d.create(None, GenParams::default()).unwrap();
    d.append_message(session, superfluid_daemon::wal::role::USER, "Write two sentences about rivers.".into())
        .unwrap();

    #[derive(Default)]
    struct Trace {
        deltas: Vec<(u32, String)>,
        committed: Vec<(u32, String)>,
        commit_before_delta: bool,
    }
    let trace = std::cell::RefCell::new(Trace::default());
    let out = d
        .generate_streaming_delta(
            session,
            48,
            Default::default(),
            |ev| {
                if let EventBody::Generated { text, channel, .. } = &ev.body {
                    if !text.is_empty() {
                        trace.borrow_mut().committed.push((*channel, text.clone()));
                    }
                }
                Ok(())
            },
            |_| {},
            |channel, text, _produced| {
                let mut t = trace.borrow_mut();
                if !t.committed.is_empty() && t.deltas.is_empty() {
                    t.commit_before_delta = true;
                }
                t.deltas.push((channel, text.to_string()));
            },
        )
        .expect("generation");
    let Trace { deltas, committed, commit_before_delta: commit_seen_before_delta } = trace.into_inner();
    assert!(out.tokens_generated >= 16, "enough tokens to span cadences");
    assert!(!commit_seen_before_delta, "a commit landed before any delta");

    for ch in [superfluid_daemon::wal::channel::TEXT, superfluid_daemon::wal::channel::REASONING] {
        let d_cat: String = deltas.iter().filter(|(c, _)| *c == ch).map(|(_, t)| t.as_str()).collect();
        let c_cat: String = committed.iter().filter(|(c, _)| *c == ch).map(|(_, t)| t.as_str()).collect();
        assert_eq!(d_cat, c_cat, "channel {ch}: deltas must reconcile with commits");
    }
    assert!(
        deltas.len() > committed.len(),
        "expected finer-than-tick delivery: {} deltas vs {} commits",
        deltas.len(),
        committed.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
