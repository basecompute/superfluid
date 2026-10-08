#![cfg(feature = "basert")]

use superfluid_daemon::codec::ChatMlCodec;
use superfluid_daemon::{
    wal, Daemon, DaemonOptions, EngineHost, EventBody, GenParams, SessionStore, TextCodec,
};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};
use std::path::PathBuf;

fn fixture_png() -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/images/gradient-64.png");
    std::fs::read(p).expect("tests/fixtures/images/gradient-64.png")
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
    let dir = std::env::temp_dir().join(format!("vision-real-{}-{tag}", std::process::id()));
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

fn assert_image_chat(d: &Daemon, session: u64) {
    let out = d.generate(session, 12).expect("generation after an image turn");
    let mut toks = 0usize;
    for e in &out.events {
        if let EventBody::Generated { span, .. } = &e.body {
            toks += span.len();
        }
    }
    assert!(toks > 0, "no tokens decoded after the image turn");

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
            let tok = p["image_token_id"].as_u64().expect("image_token_id") as u32;
            let off = p["offset"].as_u64().expect("offset") as usize;
            assert!(n > 0, "empty placeholder run: {p}");
            assert_eq!(
                s.tokens[off..off + n].iter().filter(|&&t| t == tok).count(),
                n,
                "placeholder run not at its recorded offset: {p}"
            );
        }
    }
    assert!(saw, "no IMAGE block recorded");
}

#[test]
fn qwen_vl_tower_serves_image_chat() {
    let Some(m) = model_from("BASERT_TEST_MODEL_HYBRID", "models/Qwen3.5-2B-Base-Q4.base") else {
        eprintln!("::warning::SKIP qwen_vl_tower_serves_image_chat — no Qwen3.5-VL bundle");
        return;
    };
    let codec = match ChatMlCodec::load(&m) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("::warning::SKIP qwen_vl_tower_serves_image_chat — ChatML codec refused: {e:?}");
            return;
        }
    };
    let d = spawn_daemon(m, Box::new(codec), "qwenvl");
    let session = d.create(None, GenParams::default()).unwrap();
    let blob = d.put_media(&fixture_png()).unwrap();
    d.append_image(session, wal::role::USER, &blob, "What is in ", " ? Answer briefly.")
        .expect("image turn through the ChatML codec");
    assert_image_chat(&d, session);
}

#[test]
fn muse_perception_tower_serves_image_chat() {
    let Some(m) = model_from("BASERT_TEST_MODEL_MUSE_VL", "models/muse-glimmer-30B-cuda-q4mix.base") else {
        eprintln!("::warning::SKIP muse_perception_tower_serves_image_chat — no Muse bundle");
        return;
    };
    let codec = match superfluid_daemon::atem::AtemCodec::load(&m) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("::warning::SKIP muse_perception_tower_serves_image_chat — ATEM codec refused: {e:?}");
            return;
        }
    };
    let d = spawn_daemon(m, Box::new(codec), "musevl");
    let session = d.create(None, GenParams::default()).unwrap();
    let blob = d.put_media(&fixture_png()).unwrap();
    d.append_image(session, wal::role::USER, &blob, "Describe ", " in one word.")
        .expect("image turn through the ATEM codec");
    assert_image_chat(&d, session);
}

#[test]
fn gemma4_tower_serves_image_chat_via_template() {
    let Some(m) = model_from("BASERT_TEST_MODEL_GEMMA4_VL", "models/gemma-4-E2B-it-vl.base") else {
        eprintln!("::warning::SKIP gemma4_tower_serves_image_chat_via_template — no Gemma-4 VISION bundle");
        return;
    };
    let codec = match superfluid_daemon::template_codec::TemplateCodec::load(&m) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("::warning::SKIP gemma4_tower_serves_image_chat_via_template — template load: {e:?}");
            return;
        }
    };
    let d = spawn_daemon(m, Box::new(codec), "gemma4vl");
    let session = d.create(None, GenParams::default()).unwrap();
    let blob = d.put_media(&fixture_png()).unwrap();
    use superfluid_daemon::codec::{ChatMessage, ContentPart};
    let mut msg = ChatMessage::new(wal::role::USER, String::new());
    msg.parts = vec![
        ContentPart::Text("What is in ".to_string()),
        ContentPart::Image { blob },
        ContentPart::Text(" ? Answer briefly.".to_string()),
    ];
    d.append_conversation(session, &[msg], &[], &serde_json::Map::new())
        .expect("image conversation through the model's own template");
    assert_image_chat(&d, session);
}
