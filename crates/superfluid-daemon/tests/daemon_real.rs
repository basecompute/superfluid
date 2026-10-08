//! The daemon against the REAL engine (feature `basert`).

#![cfg(feature = "basert")]

use std::path::PathBuf;

use superfluid_abi::finish;
use superfluid_daemon::{codec::ChatMlCodec, BundleCodec, Daemon, EngineHost, EventBody, GenParams, SessionStore};
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

fn ring_specs_for(vocab: u32) -> Vec<linkw::RingSpec> {
    vec![
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
    ]
}

fn real_daemon(wal: &std::path::Path, model: PathBuf) -> Daemon {
    let store = SessionStore::open(wal).expect("open store");
    let model_for_codec = model.clone();
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
    .expect("spawn engine worker");
    let codec = BundleCodec::load(&model_for_codec).expect("tokenizer load");
    Daemon::new(store, host, Box::new(codec), 8)
}

fn real_daemon_with(wal: &std::path::Path, model: PathBuf, opts: superfluid_daemon::DaemonOptions) -> Daemon {
    let store = SessionStore::open(wal).expect("open store");
    let model_for_codec = model.clone();
    let host = EngineHost::spawn(move || {
        let engine = NativeEngine::load(FfiEngineConfig {
            model_path: model,
            max_context: 4096,
            max_batch_size: 8,
            seed_ttl_ticks: 64,
        })
        .expect("model load");
        let vocab = engine.vocab();
        (engine, Some(ring_specs_for(vocab)))
    })
    .expect("spawn engine worker");
    let codec = BundleCodec::load(&model_for_codec).expect("tokenizer load");
    Daemon::with_options(store, host, Box::new(codec), opts).expect("options")
}

fn wait_for_file(p: &std::path::Path) -> bool {
    for _ in 0..200 {
        if p.exists() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    p.exists()
}

fn generated_tokens(events: &[superfluid_daemon::CommittedEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Generated { span, .. } => Some(span.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

#[test]
fn sessions_survive_restart_and_resume_warm() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model (set BASERT_TEST_MODEL or place models/Qwen3-0.6B-Q4_K_M.*)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-real-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("wal.log");
    let prompt: Vec<u32> = (0..64).map(|i| 1000 + i * 7).collect();

    let (session, first_gen, epoch_1) = {
        let d = real_daemon(&wal, model.clone());
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, prompt.clone()).unwrap();
        let out = d.generate(session, 8).unwrap();
        assert!(out.tokens_generated >= 1, "real decode produced tokens");
        assert_eq!(out.warm_prefix, 0, "first generation is cold");
        let store = d.store();
        let store = store.lock().unwrap();
        let s = store.session(session).unwrap();
        (session, generated_tokens(&s.events), s.epoch)
    };

    {
        let d = real_daemon(&wal, model);
        {
            let store = d.store();
            let store = store.lock().unwrap();
            let s = store.session(session).unwrap();
            let mut expect = prompt.clone();
            expect.extend_from_slice(&first_gen);
            assert_eq!(s.tokens, expect, "stream replayed verbatim from the WAL");
            assert_eq!(s.epoch, epoch_1 + 1, "recovery bumped the epoch");
        }

        let cold = d.generate(session, 4).unwrap();
        assert!(cold.tokens_generated >= 1);
        assert_eq!(cold.warm_prefix, 0, "fresh engine: cold re-prefill");
        if cold.finish == finish::NONE {
            let warm = d.generate(session, 4).unwrap();
            assert!(
                warm.warm_prefix >= 16,
                "resume seeded {} tokens from the real prefix cache",
                warm.warm_prefix
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn text_in_text_out_and_replay() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-text-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("wal.log");

    let (session, text_1) = {
        let d = real_daemon(&wal, model.clone());
        let session = d.create(None, GenParams::default()).unwrap();
        let e = d
            .append(session, Some("The capital of France is".into()), Vec::new())
            .unwrap();
        let EventBody::Appended { span, .. } = &e.body else {
            panic!("wrong body");
        };
        assert!(!span.is_empty(), "real tokenizer produced the span");

        let out = d.generate(session, 12).unwrap();
        let text: String = out
            .events
            .iter()
            .filter_map(|e| match &e.body {
                EventBody::Generated { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(!text.is_empty(), "decoded generation text");
        assert!(text.is_ascii() || text.chars().all(|c| c != '\u{FFFD}'),
            "no torn characters in streamed text: {text:?}");
        (session, text)
    };

    let d = real_daemon(&wal, model);
    let events = d.read(session, 0).unwrap();
    let replayed: String = events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Generated { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(replayed, text_1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn concurrent_real_sessions_batch_invariant() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use std::sync::Arc;
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-conc-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let twin_prompt: Vec<u32> = (0..48).map(|i| 4000 + i * 3).collect();
    let other_prompt: Vec<u32> = (0..32).map(|i| 9000 + i * 5).collect();

    let batched = {
        let d = Arc::new(real_daemon(&dir.join("wal.log"), model.clone()));
        let mk = |prompt: &[u32]| {
            let sid = d.create(None, GenParams::default()).unwrap();
            d.append(sid, None, prompt.to_vec()).unwrap();
            sid
        };
        let (a, b, c) = (mk(&twin_prompt), mk(&twin_prompt), mk(&other_prompt));
        let handles: Vec<_> = [a, b, c]
            .into_iter()
            .map(|sid| {
                let d = Arc::clone(&d);
                std::thread::spawn(move || {
                    let out = d.generate(sid, 8).unwrap();
                    assert!(out.tokens_generated >= 1);
                    out.events
                        .iter()
                        .filter_map(|e| match &e.body {
                            EventBody::Generated { span, .. } => Some(span.clone()),
                            _ => None,
                        })
                        .flatten()
                        .collect::<Vec<u32>>()
                })
            })
            .collect();
        let outs: Vec<Vec<u32>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(outs[0], outs[1], "twin prompts in one batch diverged");
        outs[0].clone()
    };

    let solo = {
        let d = real_daemon(&dir.join("wal2.log"), model);
        let sid = d.create(None, GenParams::default()).unwrap();
        d.append(sid, None, twin_prompt).unwrap();
        let out = d.generate(sid, 8).unwrap();
        out.events
            .iter()
            .filter_map(|e| match &e.body {
                EventBody::Generated { span, .. } => Some(span.clone()),
                _ => None,
            })
            .flatten()
            .collect::<Vec<u32>>()
    };
    assert_eq!(batched, solo, "batched output diverged from solo run");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn chatml_chat_end_to_end() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-chat-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("wal.log");

    let (session, tokens_before, text_before) = {
        let store = SessionStore::open(&wal).unwrap();
        let model2 = model.clone();
        let host = EngineHost::spawn(move || {
            let engine = superfluid_engine_ffi::NativeEngine::load(
                superfluid_engine_ffi::FfiEngineConfig {
                    model_path: model2,
                    max_context: 4096,
                    max_batch_size: 8,
                    seed_ttl_ticks: 64,
                },
            )
            .expect("model load");
            let vocab = engine.vocab();
            (engine, Some(ring_specs_for(vocab)))
        })
        .expect("spawn");
        let codec = ChatMlCodec::load(&model).expect("chatml codec");
        let d = Daemon::new(store, host, Box::new(codec), 8);

        let session = d.create(None, GenParams::default()).unwrap();
        d.append_message(
            session,
            1,
            "What is the capital of France? Reply with just the city name.".into(),
        )
        .unwrap();
        let out = d.generate(session, 1024).unwrap();
        assert_eq!(out.finish, superfluid_abi::finish::EOS, "turn ended at <|im_end|>");

        let mut reasoning = String::new();
        let mut text = String::new();
        for e in &out.events {
            if let EventBody::Generated { text: t, channel, .. } = &e.body {
                if *channel == 1 {
                    reasoning.push_str(t);
                } else {
                    text.push_str(t);
                }
            }
        }
        assert!(!reasoning.is_empty(), "Qwen3 thinks by default");
        assert!(
            text.contains("Paris"),
            "visible answer should name Paris, got: {text:?}"
        );
        let store = d.store();
        let store = store.lock().unwrap();
        let s = store.session(session).unwrap();
        (session, s.tokens.clone(), text)
    };

    let store = SessionStore::open(&wal).unwrap();
    let s = store.session(session).unwrap();
    assert_eq!(s.tokens, tokens_before);
    let replayed: String = s
        .events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Generated { text, channel: 0, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(replayed, text_before);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn openai_api_end_to_end_with_stateless_warm_reuse() {
    use std::io::{Read, Write};
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-oai-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let model2 = model.clone();
    let host = EngineHost::spawn(move || {
        let engine = superfluid_engine_ffi::NativeEngine::load(superfluid_engine_ffi::FfiEngineConfig {
            model_path: model2,
            max_context: 4096,
            max_batch_size: 8,
            seed_ttl_ticks: 64,
        })
        .expect("model load");
        let vocab = engine.vocab();
        (engine, Some(ring_specs_for(vocab)))
    })
    .expect("spawn");
    let codec = ChatMlCodec::load(&model).expect("chatml codec");
    let daemon = std::sync::Arc::new(Daemon::new(store, host, Box::new(codec), 8));
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = superfluid_daemon::openai::serve_blocking(listener, daemon, "qwen3-test".into());
    });

    let post = |body: &str| -> (String, String) {
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        write!(
            s,
            "POST /v1/chat/completions HTTP/1.1\r\nhost: {addr}\r\nconnection: close\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut raw = String::new();
        s.read_to_string(&mut raw).unwrap();
        let (head, payload) = raw.split_once("\r\n\r\n").unwrap();
        (head.to_string(), payload.to_string())
    };
    let req = r#"{"model":"qwen3-test","messages":[{"role":"user","content":"What is the capital of France? Reply with just the city name."}],"max_tokens":1024,"temperature":0}"#;

    let (head, body) = post(req);
    assert!(head.starts_with("HTTP/1.1 200"));
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.contains("Paris"), "got: {content:?}");
    assert!(
        v["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .map(|r| !r.is_empty())
            .unwrap_or(false),
        "reasoning separated from the answer"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "stop");

    let (head2, body2) = post(req);
    let warm: u64 = head2
        .lines()
        .find_map(|l| l.to_ascii_lowercase().strip_prefix("x-superfluid-warm:").map(|v| v.trim().parse().unwrap_or(0)))
        .unwrap_or(0);
    assert!(warm > 0, "repeat request should seed warm, head: {head2}");
    let v2: serde_json::Value = serde_json::from_str(&body2).unwrap();
    assert_eq!(
        v2["choices"][0]["message"]["content"], v["choices"][0]["message"]["content"],
        "warm-seeded continuation must be token-identical to cold"
    );

    let sreq = req.replace(r#""max_tokens""#, r#""stream":true,"max_tokens""#);
    let (_, sbody_raw) = post(&sreq);
    let sbody = dechunk(&sbody_raw);
    let mut streamed = String::new();
    for e in sbody.split("\n\n").filter_map(|b| b.strip_prefix("data: ")) {
        if e == "[DONE]" {
            break;
        }
        let c: serde_json::Value = serde_json::from_str(e).unwrap();
        if let Some(t) = c["choices"][0]["delta"]["content"].as_str() {
            streamed.push_str(t);
        }
    }
    assert_eq!(streamed, content, "streamed content equals non-streamed");
    let _ = std::fs::remove_dir_all(&dir);
}

fn dechunk(payload: &str) -> String {
    let mut out = String::new();
    let mut rest = payload;
    while let Some((size_line, tail)) = rest.split_once("\r\n") {
        let Ok(size) = usize::from_str_radix(size_line.trim(), 16) else {
            break;
        };
        if size == 0 {
            break;
        }
        if tail.len() < size {
            out.push_str(tail);
            break;
        }
        out.push_str(&tail[..size]);
        rest = tail[size..].strip_prefix("\r\n").unwrap_or(&tail[size..]);
    }
    out
}

#[test]
fn real_tool_call_round_trip() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-tool-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let model2 = model.clone();
    let host = EngineHost::spawn(move || {
        let engine = superfluid_engine_ffi::NativeEngine::load(superfluid_engine_ffi::FfiEngineConfig {
            model_path: model2,
            max_context: 4096,
            max_batch_size: 8,
            seed_ttl_ticks: 64,
        })
        .expect("model load");
        let vocab = engine.vocab();
        (engine, Some(ring_specs_for(vocab)))
    })
    .expect("spawn");
    let codec = ChatMlCodec::load(&model).expect("chatml codec").with_thinking(false);
    let d = Daemon::new(store, host, Box::new(codec), 8);

    let session = d.create(None, GenParams::default()).unwrap();
    d.append_system_with_tools(
        session,
        None,
        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        })
        .to_string()],
    )
    .unwrap();
    d.append_message(session, 1, "What is the weather in Paris right now? Use the tool.".into())
        .unwrap();

    let out = d.generate(session, 1024).unwrap();
    let calls: Vec<(u64, String, String)> = out
        .events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::ToolUse { name, arguments } => {
                Some((e.event_id, name.clone(), arguments.clone()))
            }
            _ => None,
        })
        .collect();
    if calls.is_empty() {
        for e in &out.events {
            if let EventBody::Generated { text, channel, .. } = &e.body {
                eprintln!("[ch{channel}] {text:?}");
            }
        }
    }
    assert!(!calls.is_empty(), "the model should call the tool");
    let (call_id, name, args) = &calls[0];
    assert_eq!(name, "get_weather");
    assert!(args.contains("Paris"), "arguments: {args:?}");
    assert_eq!(d.open_tool_calls(session).unwrap().len(), calls.len());

    d.append_tool_result(
        session,
        *call_id,
        r#"{"temperature": "18C", "condition": "sunny"}"#.into(),
    )
    .unwrap();
    assert_eq!(
        d.open_tool_calls(session).unwrap().len(),
        calls.len() - 1,
        "result closed its ledger entry"
    );

    let out2 = d.generate(session, 1024).unwrap();
    let text: String = out2
        .events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Generated { text, channel: 0, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if !(text.contains("18") || text.to_lowercase().contains("sunny")) {
        for e in &out2.events {
            if let EventBody::Generated { text, channel, finish, .. } = &e.body {
                eprintln!("[2:ch{channel} fin{finish}] {text:?}");
            }
        }
        eprintln!("[out2 fin={} n={}]", out2.finish, out2.tokens_generated);
    }
    assert!(
        text.contains("18") || text.to_lowercase().contains("sunny"),
        "final answer should use the tool result, got: {text:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn anthropic_api_end_to_end() {
    use std::io::{Read, Write};
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-anthropic-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let model2 = model.clone();
    let host = EngineHost::spawn(move || {
        let engine = superfluid_engine_ffi::NativeEngine::load(superfluid_engine_ffi::FfiEngineConfig {
            model_path: model2,
            max_context: 4096,
            max_batch_size: 8,
            seed_ttl_ticks: 64,
        })
        .expect("model load");
        let vocab = engine.vocab();
        (engine, Some(ring_specs_for(vocab)))
    })
    .expect("spawn");
    let codec = ChatMlCodec::load(&model).expect("chatml codec");
    let daemon = std::sync::Arc::new(Daemon::new(store, host, Box::new(codec), 8));
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = superfluid_daemon::openai::serve_blocking(listener, daemon, "qwen3-test".into());
    });

    let post = |body: &str| -> String {
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        write!(
            s,
            "POST /v1/messages HTTP/1.1\r\nhost: {addr}\r\nconnection: close\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut raw = String::new();
        s.read_to_string(&mut raw).unwrap();
        let (head, payload) = raw.split_once("\r\n\r\n").unwrap();
        if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
            dechunk(payload)
        } else {
            payload.to_string()
        }
    };

    let body = post(
        r#"{"model":"qwen3-test","max_tokens":1024,"temperature":0,
            "messages":[{"role":"user","content":"What is the capital of France? Reply with just the city name."}]}"#,
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["type"], "message", "body: {body}");
    let blocks = v["content"].as_array().unwrap();
    assert!(
        blocks.iter().any(|b| b["type"] == "thinking"),
        "thinking block present"
    );
    let text: String = blocks
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect();
    assert!(text.contains("Paris"), "got: {text:?}");
    assert_eq!(v["stop_reason"], "end_turn");

    let sbody = post(
        r#"{"model":"qwen3-test","max_tokens":1024,"stream":true,"temperature":0,
            "messages":[{"role":"user","content":"What is the capital of France? Reply with just the city name."}]}"#,
    );
    let mut streamed_text = String::new();
    let mut saw_thinking_delta = false;
    for chunk in sbody.split("\n\n") {
        let Some(data) = chunk.lines().find_map(|l| l.strip_prefix("data: ")) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        if v["type"] == "content_block_delta" {
            match v["delta"]["type"].as_str() {
                Some("text_delta") => {
                    streamed_text.push_str(v["delta"]["text"].as_str().unwrap_or(""))
                }
                Some("thinking_delta") => saw_thinking_delta = true,
                _ => {}
            }
        }
    }
    assert!(saw_thinking_delta, "thinking streamed on its own block");
    assert_eq!(streamed_text, text, "streamed text equals non-streamed");
    let _ = std::fs::remove_dir_all(&dir);
}

fn real_daemon_parked(wal: &std::path::Path, model: PathBuf, park: &std::path::Path) -> Daemon {
    let store = SessionStore::open(wal).expect("open store");
    let model_for_codec = model.clone();
    let host = EngineHost::spawn(move || {
        let engine = NativeEngine::load(FfiEngineConfig {
            model_path: model,
            max_context: 4096,
            max_batch_size: 8,
            seed_ttl_ticks: 64,
        })
        .expect("model load");
        let vocab = engine.vocab();
        let specs = ring_specs_for(vocab);
        (engine, Some(specs))
    })
    .expect("spawn engine worker");
    let codec = BundleCodec::load(&model_for_codec).expect("tokenizer load");
    Daemon::with_park(store, host, Box::new(codec), 8, Some(park.to_path_buf()))
}

#[test]
fn park_resume_survives_restart_real() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let base = std::env::temp_dir().join(format!("superfluid-park-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let warm_home = base.join("warm");
    let cold_home = base.join("cold");
    std::fs::create_dir_all(&warm_home).unwrap();
    std::fs::create_dir_all(&cold_home).unwrap();
    let park = warm_home.join("park");
    let prompt: Vec<u32> = (0..64).map(|i| 1000 + i * 7).collect();

    let run_life1 = |wal: &std::path::Path, park: Option<&std::path::Path>| -> u64 {
        let d = match park {
            Some(p) => real_daemon_parked(wal, model.clone(), p),
            None => real_daemon(wal, model.clone()),
        };
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, prompt.clone()).unwrap();
        let out = d.generate(session, 8).unwrap();
        assert!(out.tokens_generated >= 1);
        assert_eq!(out.warm_prefix, 0, "first generation is cold");
        session
    };
    let session = run_life1(&warm_home.join("wal.log"), Some(&park));
    assert!(
        park.join(format!("{session}.park")).exists(),
        "retire parked a sealed artifact"
    );
    let cold_session = run_life1(&cold_home.join("wal.log"), None);

    let warm = {
        let d = real_daemon_parked(&warm_home.join("wal.log"), model.clone(), &park);
        d.generate(session, 4).unwrap()
    };
    let cold = {
        let d = real_daemon(&cold_home.join("wal.log"), model);
        d.generate(cold_session, 4).unwrap()
    };
    assert!(
        warm.warm_prefix >= 16,
        "restart resumed warm from the park artifact (got {})",
        warm.warm_prefix
    );
    assert_eq!(cold.warm_prefix, 0, "control restart is cold");
    assert_eq!(
        generated_tokens(&warm.events),
        generated_tokens(&cold.events),
        "resume-from-park continues the exact greedy stream"
    );
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn observer_subscription_with_provisional_deltas_real() {
    use superfluid_daemon::api;
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-sub-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let socket = dir.join("superfluid.sock");
    let daemon = std::sync::Arc::new(real_daemon(&dir.join("wal.log"), model));
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let serving = std::sync::Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = api::serve(listener, serving);
    });

    let session = daemon.create(None, GenParams::default()).unwrap();
    daemon
        .append(session, Some("The capital of France is".into()), Vec::new())
        .unwrap();
    let cursor = daemon.read(session, 0).unwrap().len() as u64;

    let mut obs = api::NativeClient::connect(&socket).expect("connect observer");
    let sub = obs.subscribe(session, cursor, true).unwrap();

    let out = daemon.generate(session, 12).unwrap();
    let committed_texts: Vec<String> = out
        .events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Generated { text, .. } if !text.is_empty() => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert!(!committed_texts.is_empty(), "real decode produced text");

    let last_id = out.events.last().unwrap().event_id;
    let mut provisional: Vec<String> = Vec::new();
    let mut committed: Vec<String> = Vec::new();
    let mut done = false;
    while !done {
        match obs.next_frame().unwrap() {
            api::Response::SubProvisional { sub: s, text, .. } => {
                assert_eq!(s, sub);
                provisional.push(text);
            }
            api::Response::SubEvent { sub: s, event } => {
                assert_eq!(s, sub);
                done = event.event_id == last_id;
                if let EventBody::Generated { text, .. } = &event.body {
                    if !text.is_empty() {
                        committed.push(text.clone());
                    }
                }
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(
            provisional.len() >= committed.len(),
            "a committed slice arrived before its provisional preview"
        );
    }
    assert_eq!(
        provisional.concat(),
        committed_texts.concat(),
        "provisional text reconciles exactly with the committed record"
    );
    assert!(
        provisional.len() >= committed_texts.len(),
        "token-cadence deltas are at least as fine as tick-cadence commits"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pressure_relief_real_evicts_lossless_with_park() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-pressure-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let park = dir.join("park");

    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let model_for_codec = model.clone();
    let host = superfluid_daemon::EngineHost::spawn(move || {
        let engine = NativeEngine::load(FfiEngineConfig {
            model_path: model,
            max_context: 4096,
            max_batch_size: 8,
            seed_ttl_ticks: 64,
        })
        .expect("model load");
        let vocab = engine.vocab();
        (engine, Some(ring_specs_for(vocab)))
    })
    .expect("spawn engine worker");
    let codec = BundleCodec::load(&model_for_codec).expect("tokenizer load");
    let d = superfluid_daemon::Daemon::with_options(
        store,
        host,
        Box::new(codec),
        superfluid_daemon::DaemonOptions {
            max_lanes: 8,
            park_dir: Some(park.clone()),
            pressure_high_pct: 1,
            pressure_low_pct: 0,
            ..Default::default()
        },
    )
    .unwrap();

    let s1 = d.create(None, GenParams::default()).unwrap();
    d.append(s1, None, (0..400).map(|i| 1000 + i * 7).collect()).unwrap();
    assert!(d.generate(s1, 8).unwrap().tokens_generated >= 1);
    assert!(wait_for_file(&park.join(format!("{s1}.park"))));

    let s2 = d.create(None, GenParams::default()).unwrap();
    d.append(s2, None, (0..400).map(|i| 2000 + i * 5).collect()).unwrap();
    assert!(d.generate(s2, 8).unwrap().tokens_generated >= 1);

    let stats = d.sched_stats();
    assert!(
        stats
            .pressure_evictions
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0,
        "aggressive watermarks force relief on the real cache"
    );

    let out = d.generate(s1, 4).unwrap();
    assert!(
        out.warm_prefix >= 16,
        "evicted-under-pressure session resumes warm from its artifact (got {})",
        out.warm_prefix
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lossy_park_real_round_trip() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-lossy-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    superfluid_engine_ffi::set_kv_bits(16);
    let park = dir.join("park");
    let wal = dir.join("wal.log");
    let mk = |park_lossy: bool| -> Daemon {
        let store = SessionStore::open(&wal).unwrap();
        let m = model.clone();
        let host = EngineHost::spawn(move || {
            let engine = NativeEngine::load(FfiEngineConfig {
                model_path: m,
                max_context: 4096,
                max_batch_size: 8,
                seed_ttl_ticks: 64,
            })
            .expect("model load");
            let vocab = engine.vocab();
            (engine, Some(ring_specs_for(vocab)))
        })
        .expect("spawn");
        let codec = BundleCodec::load(&model).expect("tokenizer load");
        Daemon::with_options(
            store,
            host,
            Box::new(codec),
            superfluid_daemon::DaemonOptions {
                max_lanes: 8,
                park_dir: Some(park.clone()),
                park_lossy,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let prompt: Vec<u32> = (0..64).map(|i| 1000 + i * 7).collect();
    let session = {
        let d = mk(true);
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, prompt.clone()).unwrap();
        assert!(d.generate(session, 8).unwrap().tokens_generated >= 1);
        session
    };
    let lossy = superfluid_daemon::park::read(&park, session).expect("artifact");
    assert_eq!(lossy.encoding, superfluid_abi::encoding::Q8);

    let lossless_size = {
        {
            let d = mk(false);
            d.generate(session, 4).unwrap();
        }
        superfluid_daemon::park::read(&park, session).unwrap().sealed.len()
    };
    assert!(
        (lossy.sealed.len() as f64) < 0.6 * lossless_size as f64,
        "lossy artifact compresses ({} vs {} lossless)",
        lossy.sealed.len(),
        lossless_size
    );

    {
        let d = mk(true);
        d.generate(session, 4).unwrap();
    }
    let d = mk(true);
    let out = d.generate(session, 4).unwrap();
    assert!(out.tokens_generated >= 1, "lossy-resumed KV still generates");
    assert!(
        out.warm_prefix >= 16,
        "restart resumes warm from the LOSSY artifact (got {})",
        out.warm_prefix
    );
    drop(d);
    superfluid_engine_ffi::set_kv_bits(0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sampled_runs_are_shape_deterministic_and_forks_continue_identically() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-rng-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let d = real_daemon(&dir.join("wal.log"), model);
    let params = GenParams {
        temperature: 0.9,
        top_p: 0.95,
        top_k: 40,
        seed: 4242,
        ..Default::default()
    };
    let prompt: Vec<u32> = (0..40).map(|i| 3000 + i * 5).collect();
    let tokens_of = |session: u64| {
        let store = d.store();
        let store = store.lock().unwrap();
        store.session(session).unwrap().tokens.clone()
    };
    let warmup = d.create(None, params).unwrap();
    d.append(warmup, None, prompt.clone()).unwrap();
    d.generate(warmup, 1).unwrap();

    let a = d.create(None, params).unwrap();
    d.append(a, None, prompt.clone()).unwrap();
    let ra = d.generate(a, 24).unwrap();
    assert_eq!(ra.tokens_generated, 24);
    assert!(ra.warm_prefix > 0);
    let b = d.create(None, params).unwrap();
    d.append(b, None, prompt.clone()).unwrap();
    d.generate(b, 24).unwrap();
    assert_eq!(tokens_of(a), tokens_of(b), "same shape + seed → same sampled tokens");
    let c = d
        .create(
            None,
            GenParams {
                seed: 99,
                ..params
            },
        )
        .unwrap();
    d.append(c, None, prompt.clone()).unwrap();
    d.generate(c, 24).unwrap();
    assert_ne!(tokens_of(a), tokens_of(c), "the seed is live");

    let cut = d.create(None, params).unwrap();
    d.append(cut, None, prompt).unwrap();
    d.generate(cut, 10).unwrap();
    let at = d.read(cut, 0).unwrap().len() as u64;
    let branch = d.fork(cut, at, None).unwrap();
    d.generate(cut, 14).unwrap();
    d.generate(branch, 14).unwrap();
    assert_eq!(tokens_of(branch), tokens_of(cut), "a fork continues exactly as its parent");
    assert_eq!(tokens_of(cut)[..50], tokens_of(a)[..50], "the pre-cut prefix is the whole run's");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn preemption_park_resume_is_token_exact_real() {
    use superfluid_daemon::qos;
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-preempt-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let d = std::sync::Arc::new(real_daemon_with(
        &dir.join("wal.log"),
        model,
        superfluid_daemon::DaemonOptions {
            max_lanes: 1,
            park_dir: Some(dir.join("park")),
            ..Default::default()
        },
    ));
    let stats = d.sched_stats();
    let prompt: Vec<u32> = (0..40).map(|i| 3000 + i * 5).collect();
    let tokens_of = |session: u64| {
        let store = d.store();
        let store = store.lock().unwrap();
        store.session(session).unwrap().tokens.clone()
    };
    const AGENT_TOKENS: u32 = 512;
    let warmup = d.create(None, GenParams::default()).unwrap();
    d.append(warmup, None, prompt.clone()).unwrap();
    d.generate(warmup, 4).unwrap();
    let reference = d.create(None, GenParams::default()).unwrap();
    d.append(reference, None, prompt.clone()).unwrap();
    d.generate(reference, AGENT_TOKENS).unwrap();

    let agent = d.create(None, GenParams::default()).unwrap();
    d.append(agent, None, prompt.clone()).unwrap();
    d.set_qos(agent, qos::BACKGROUND_AGENT, false).unwrap();
    let dd = std::sync::Arc::clone(&d);
    let t = std::thread::spawn(move || dd.generate(agent, AGENT_TOKENS));
    for _ in 0..1000 {
        if d.active_registry().lock().unwrap().contains(&agent) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    for _ in 0..1000 {
        if !d.active_registry().lock().unwrap().contains(&agent) {
            break;
        }
        if d.sched_stats().decode_tokens.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let chat = d.create(None, GenParams::default()).unwrap();
    d.append(chat, None, (0..24).map(|i| 5000 + i * 3).collect()).unwrap();
    d.set_qos(chat, qos::INTERACTIVE_CHAT, false).unwrap();
    let agent_live_when_chat_ran = {
        let active = d.active_registry();
        let mut live = false;
        d.generate_streaming(chat, 4, |_| {
            live |= active.lock().unwrap().contains(&agent);
            Ok(())
        })
        .unwrap();
        live
    };
    let out = t.join().unwrap().unwrap();
    assert_eq!(out.tokens_generated, AGENT_TOKENS);
    assert!(agent_live_when_chat_ran, "the chat ran while the agent call was in flight");
    assert!(stats.preemptions.load(std::sync::atomic::Ordering::Relaxed) >= 1, "the agent was preempted");
    assert_eq!(tokens_of(agent), tokens_of(reference), "preempt/park/resume is token-exact");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn speculation_native_is_exact_and_fingerprinted_real() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-spec-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let plain_dir = dir.join("plain");
    let spec_dir = dir.join("spec");
    std::fs::create_dir_all(&plain_dir).unwrap();
    std::fs::create_dir_all(&spec_dir).unwrap();
    let mut prompt = Vec::new();
    for _ in 0..6 {
        prompt.extend_from_slice(&[3000, 3005, 3010, 3015, 3020, 3025, 3030, 3035]);
    }
    let run = |d: &Daemon| -> (Vec<u32>, Option<u64>) {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, prompt.clone()).unwrap();
        d.generate(s, 48).unwrap();
        let store = d.store();
        let store = store.lock().unwrap();
        let st = store.session(s).unwrap();
        (st.tokens.clone(), st.behavior_fp)
    };
    let plain = real_daemon(&plain_dir.join("wal.log"), model.clone());
    let (plain_tokens, plain_fp) = run(&plain);
    drop(plain);
    let spec = real_daemon_with(
        &spec_dir.join("wal.log"),
        model,
        superfluid_daemon::DaemonOptions {
            speculate: Some("prompt-lookup".into()),
            ..Default::default()
        },
    );
    let grant = spec.speculation().expect("prompt-lookup registered natively");
    assert_eq!(grant.class, superfluid_abi::exactness::DISTRIBUTION_EXACT);
    assert_eq!(
        grant.modes,
        superfluid_abi::cert_mode::GREEDY | superfluid_abi::cert_mode::GUMBEL
    );
    let (spec_tokens, spec_fp) = run(&spec);
    assert_eq!(spec_tokens, plain_tokens, "speculation never changes greedy output");
    assert_ne!(spec_fp, plain_fp, "a non-SPI strategy joins the generation fingerprint");
    let st = spec.sched_stats();
    let proposed = st.spec_proposed.load(std::sync::atomic::Ordering::Relaxed);
    let accepted = st.spec_accepted.load(std::sync::atomic::Ordering::Relaxed);
    assert!(proposed > 0 && accepted > 0, "drafts proposed {proposed} accepted {accepted}");
    assert!(spec.metrics_text().contains(&format!("superfluid_spec_accepted_total {accepted}")));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fim_completion_real() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-fim-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let model_for_codec = model.clone();
    let host = EngineHost::spawn(move || {
        let engine = NativeEngine::load(FfiEngineConfig {
            model_path: model,
            max_context: 4096,
            max_batch_size: 8,
            seed_ttl_ticks: 64,
        })
        .expect("model load");
        let vocab = engine.vocab();
        (engine, Some(ring_specs_for(vocab)))
    })
    .expect("spawn");
    let codec = ChatMlCodec::load(&model_for_codec).expect("chatml codec");
    let d = Daemon::new(store, host, Box::new(codec), 8);
    let before = d.session_ids();
    let pre = "def add(a, b):\n    return ";
    let suf = "\n\nprint(add(1, 2))\n";
    let c1 = d.complete(pre, suf, superfluid_daemon::codec::fim_mode::PSM, 12).unwrap();
    assert!(!c1.cached && !c1.expired);
    assert_eq!(c1.tokens.len(), 12);
    assert!(!c1.text.is_empty());
    let c2 = d.complete(pre, suf, superfluid_daemon::codec::fim_mode::PSM, 12).unwrap();
    assert!(c2.cached);
    assert_eq!(c2.tokens, c1.tokens);
    assert_eq!(d.session_ids(), before);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn draft_model_speculation_real() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("superfluid-draft-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prompt: Vec<u32> = (0..40).map(|i| 3000 + i * 5).collect();
    let run = |d: &Daemon| -> (Vec<u32>, Option<u64>) {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, prompt.clone()).unwrap();
        d.generate(s, 40).unwrap();
        let store = d.store();
        let store = store.lock().unwrap();
        let st = store.session(s).unwrap();
        (st.tokens.clone(), st.behavior_fp)
    };
    let plain = real_daemon(&dir.join("plain.log"), model.clone());
    let (plain_tokens, plain_fp) = run(&plain);
    drop(plain);

    let spec = real_daemon_with(
        &dir.join("spec.log"),
        model.clone(),
        superfluid_daemon::DaemonOptions {
            speculate: Some(format!("draft-model:{}", model.display())),
            ..Default::default()
        },
    );
    let grant = spec.speculation().expect("draft-model registered");
    assert_eq!(grant.strategy_id, "draft-model");
    assert_eq!(grant.class, superfluid_abi::exactness::DISTRIBUTION_EXACT);
    let (spec_tokens, spec_fp) = run(&spec);
    assert_eq!(spec_tokens, plain_tokens, "draft-model speculation is token-exact");
    assert_ne!(spec_fp, plain_fp, "a non-SPI strategy joins the fingerprint");
    let st = spec.sched_stats();
    let proposed = st.spec_proposed.load(std::sync::atomic::Ordering::Relaxed);
    let accepted = st.spec_accepted.load(std::sync::atomic::Ordering::Relaxed);
    assert!(proposed > 0 && accepted == proposed, "self-draft: all {proposed} proposals accepted");
    let _ = std::fs::remove_dir_all(&dir);
}
