#![cfg(feature = "basert")]

use superfluid_daemon::{
    wal, Daemon, DaemonOptions, EngineHost, EventBody, GenParams, SessionStore, TextCodec,
};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};
use std::path::PathBuf;

fn fixture_wav() -> Vec<u8> {
    let n: usize = 32000;
    let mut pcm = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / 16000.0;
        let env = 0.5 + 0.4 * (2.0 * std::f32::consts::PI * t).sin();
        let s = env * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        pcm.extend_from_slice(&((s * 32767.0) as i16).to_le_bytes());
    }
    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&32000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    wav.extend_from_slice(&pcm);
    wav
}

fn model_from(env: &str, default_rel: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env) {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(default_rel);
    p.exists().then_some(p)
}

fn spawn_daemon(model: PathBuf, codec: Box<dyn TextCodec + Send + Sync>, tag: &str) -> Daemon {
    let dir = std::env::temp_dir().join(format!("audio-real-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let mp = model;
    let host = EngineHost::spawn(move || {
        let e = superfluid_engine_ffi::NativeEngine::load(superfluid_engine_ffi::FfiEngineConfig {
            model_path: mp,
            max_context: 4096,
            max_batch_size: 2,
            seed_ttl_ticks: 64,
        })
        .expect("engine load");
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
    })
    .expect("spawn engine");
    Daemon::with_options(
        store,
        host,
        codec,
        DaemonOptions {
            max_lanes: 2,
            media_dir: Some(dir.join("media")),
            ..Default::default()
        },
    )
    .unwrap()
}

fn assert_audio_chat(d: &Daemon, session: u64) {
    let out = d.generate(session, 12).expect("generation after an audio turn");
    let mut toks = 0usize;
    for e in &out.events {
        if let EventBody::Generated { span, .. } = &e.body {
            toks += span.len();
        }
    }
    assert!(toks > 0, "no tokens decoded after the audio turn");

    let store = d.store();
    let store = store.lock().unwrap();
    let s = store.session(session).unwrap();
    let mut saw = false;
    for e in &s.events {
        if let EventBody::Block { kind, payload, .. } = &e.body {
            if *kind != wal::block_kind::IMAGE {
                continue;
            }
            saw = true;
            let p: serde_json::Value = serde_json::from_str(payload).unwrap();
            let n = p["n_tokens"].as_u64().expect("n_tokens") as usize;
            let tok = p["image_token_id"].as_u64().expect("marker id") as u32;
            let off = p["offset"].as_u64().expect("offset") as usize;
            assert!(n > 0, "empty placeholder run: {p}");
            assert_eq!(
                s.tokens[off..off + n].iter().filter(|&&t| t == tok).count(),
                n,
                "placeholder run not at its recorded offset: {p}"
            );
        }
    }
    assert!(saw, "no media block recorded");
}

#[test]
fn gemma4_conformer_serves_audio_chat_via_template() {
    let Some(m) = model_from("BASERT_TEST_MODEL_GEMMA4_VL", "models/gemma-4-E2B-it-vl.base") else {
        eprintln!("::warning::SKIP gemma4_conformer_serves_audio_chat_via_template — no Gemma-4 bundle");
        return;
    };
    let codec = match superfluid_daemon::template_codec::TemplateCodec::load(&m) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("::warning::SKIP gemma4_conformer_serves_audio_chat_via_template — template load: {e:?}");
            return;
        }
    };
    let d = spawn_daemon(m, Box::new(codec), "tmpl");
    let session = d.create(None, GenParams::default()).unwrap();
    let blob = d.put_media(&fixture_wav()).unwrap();
    use superfluid_daemon::codec::{ChatMessage, ContentPart};
    let mut msg = ChatMessage::new(wal::role::USER, String::new());
    msg.parts = vec![
        ContentPart::Text("What do you hear in ".to_string()),
        ContentPart::Audio { blob },
        ContentPart::Text(" ? Answer briefly.".to_string()),
    ];
    d.append_conversation(session, &[msg], &[], &serde_json::Map::new())
        .expect("audio conversation through the model's own template");
    assert_audio_chat(&d, session);
}
