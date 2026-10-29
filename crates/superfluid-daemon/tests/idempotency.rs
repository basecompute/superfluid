//! `Idempotency-Key`: a request sent again with its key gets the first one's answer, whole.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde_json::Value;
use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::registry::ModelRegistry;
use superfluid_daemon::{openai, Daemon, EngineHost, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};

static DIRS: AtomicU64 = AtomicU64::new(0);

fn temp(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("superfluid-idem-{tag}-{}-{}", std::process::id(), DIRS.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A server whose engine takes `tick` a step, recording keyed answers under `keep`.
fn server(keep: Option<std::path::PathBuf>, tick: std::time::Duration, cfg: openai::ServeConfig) -> String {
    let dir = temp("store");
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || (MockEngine::new(EngineConfig { vocab: 512, tick_delay: tick, ..EngineConfig::default() }), None)).unwrap();
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let registry = Arc::new(ModelRegistry::single("mock-model", daemon));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, openai::ServeConfig { idempotency_dir: keep, ..cfg });
    });
    for _ in 0..100 {
        if std::net::TcpStream::connect(&addr).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    addr
}

fn quick(keep: Option<std::path::PathBuf>) -> String {
    server(keep, std::time::Duration::ZERO, openai::ServeConfig::default())
}

/// Sampled, so two runs of it say different things.
fn chat(stream: bool) -> Value {
    serde_json::json!({"model": "mock-model", "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 24, "temperature": 1.0, "stream": stream})
}

fn post(addr: &str, body: &Value, key: Option<&str>, auth: Option<&str>) -> (u16, Option<String>, String) {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(20)))
        .http_status_as_error(false)
        .build()
        .new_agent();
    let mut req = agent.post(&format!("http://{addr}/v1/chat/completions")).header("content-type", "application/json");
    if let Some(key) = key {
        req = req.header("Idempotency-Key", key);
    }
    if let Some(auth) = auth {
        req = req.header("Authorization", &format!("Bearer {auth}"));
    }
    let mut resp = req.send(body.to_string().as_bytes()).unwrap();
    let replayed = resp.headers().get("idempotent-replayed").and_then(|v| v.to_str().ok()).map(str::to_string);
    (resp.status().as_u16(), replayed, resp.body_mut().read_to_string().unwrap())
}

#[test]
fn a_request_sent_again_with_its_key_gets_the_first_answer() {
    let addr = quick(None);
    let (s1, r1, b1) = post(&addr, &chat(false), Some("k-1"), None);
    let (s2, r2, b2) = post(&addr, &chat(false), Some("k-1"), None);
    assert_eq!((s1, s2), (200, 200), "{b1} / {b2}");
    assert_eq!(b2, b1, "the second is the first's answer, not a new one");
    assert_eq!((r1, r2.as_deref()), (None, Some("true")));
    let (_, _, b3) = post(&addr, &chat(false), Some("k-2"), None);
    let (_, _, b4) = post(&addr, &chat(false), None, None);
    assert_ne!(b3, b1, "another key runs again");
    assert_ne!(b4, b3, "no key runs again");
}

#[test]
fn a_key_sent_with_another_request_is_refused() {
    let addr = quick(None);
    post(&addr, &chat(false), Some("k"), None);
    let mut other = chat(false);
    other["max_tokens"] = 5.into();
    let (status, _, body) = post(&addr, &other, Some("k"), None);
    assert_eq!(status, 422, "{body}");
    assert!(body.contains("Idempotency-Key"), "{body}");
}

#[test]
fn a_stream_cut_short_is_replayed_whole_from_its_start() {
    let addr = server(None, std::time::Duration::from_millis(40), openai::ServeConfig::default());
    // The first client reads a little of the stream and goes.
    let body = chat(true).to_string();
    let mut s = std::net::TcpStream::connect(&addr).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: localhost\r\ncontent-type: application/json\r\nidempotency-key: cut\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut first = [0u8; 256];
    let n = s.read(&mut first).unwrap();
    assert!(n > 0);
    drop(s);

    let (status, replayed, again) = post(&addr, &chat(true), Some("cut"), None);
    assert_eq!((status, replayed.as_deref()), (200, Some("true")), "{again}");
    assert!(again.trim_end().ends_with("data: [DONE]"), "the whole stream, to its end: {again}");
    let (_, _, once_more) = post(&addr, &chat(true), Some("cut"), None);
    assert_eq!(once_more, again, "every retry reads the same recording");
}

#[test]
fn a_finished_answer_is_replayed_after_a_restart() {
    let keep = temp("keep");
    let first = quick(Some(keep.clone()));
    let (_, _, answer) = post(&first, &chat(false), Some("across"), None);
    let second = quick(Some(keep));
    let (status, replayed, again) = post(&second, &chat(false), Some("across"), None);
    assert_eq!((status, replayed.as_deref()), (200, Some("true")), "{again}");
    assert_eq!(again, answer);
}

#[test]
fn a_key_is_the_credentials_it_came_with() {
    let keys = Arc::new(
        superfluid_daemon::keypolicy::KeyTable::parse(
            r#"{"keys":[{"key":"sk-alice","name":"alice"},{"key":"sk-bob","name":"bob"}]}"#,
            |_| None,
        )
        .unwrap(),
    );
    let addr = server(None, std::time::Duration::ZERO, openai::ServeConfig { keys: Some(keys), ..Default::default() });
    let (_, _, alice) = post(&addr, &chat(false), Some("shared"), Some("sk-alice"));
    let (status, replayed, bob) = post(&addr, &chat(false), Some("shared"), Some("sk-bob"));
    assert_eq!((status, replayed), (200, None), "bob's request is his own: {bob}");
    assert_ne!(bob, alice);
}

#[test]
fn a_request_a_restart_cut_short_runs_again_under_its_key() {
    let keep = temp("cut-short");
    let first = quick(Some(keep.clone()));
    let (_, _, answer) = post(&first, &chat(false), Some("crash"), None);
    // As a run that stopped mid-answer leaves it: the record not done, its body partial.
    for e in std::fs::read_dir(&keep).unwrap().flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "json") {
            let mut v: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
            v["done"] = false.into();
            std::fs::write(&p, v.to_string()).unwrap();
        } else {
            std::fs::write(&p, &answer.as_bytes()[..answer.len() / 2]).unwrap();
        }
    }
    let second = quick(Some(keep));
    let (status, replayed, again) = post(&second, &chat(false), Some("crash"), None);
    assert_eq!((status, replayed), (200, None), "run again, not the half it had: {again}");
    let v: Value = serde_json::from_str(&again).expect("a whole answer");
    assert!(v["choices"][0]["message"]["content"].is_string(), "{again}");
    let (_, replayed, once_more) = post(&second, &chat(false), Some("crash"), None);
    assert_eq!((replayed.as_deref(), once_more), (Some("true"), again), "and kept for its key from then on");
}
