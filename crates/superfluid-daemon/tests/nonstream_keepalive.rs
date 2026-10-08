//! `--nonstream-keepalive <secs>` over the full mock stack, raw HTTP/1.1.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::{openai, Daemon, EngineHost, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};

fn spawn(keepalive_secs: u64, tick: Duration, ctx: Option<u32>) -> (std::net::SocketAddr, Arc<Daemon>) {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-keepalive-test-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || {
        let cfg = EngineConfig { tick_delay: tick, ..EngineConfig::default() };
        let vocab = cfg.vocab;
        let e = MockEngine::new(cfg);
        let specs = ctx.map(|ctx| {
            vec![
                linkw::RingSpec { ring_id: TOKEN_RING_IN, kind: linkw::RingKind::Tokens, slot_bytes: 16 + ctx * 4, slots: 64 },
                linkw::RingSpec { ring_id: TOKEN_RING_OUT, kind: linkw::RingKind::Tokens, slot_bytes: 16 + 1024 * 4, slots: 64 },
                linkw::RingSpec { ring_id: LOGITS_RING, kind: linkw::RingKind::Logits, slot_bytes: 16 + vocab * 4, slots: 8 },
            ]
        });
        (e, specs)
    })
    .expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let registry =
        Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", Arc::clone(&daemon)));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = openai::ServeConfig { nonstream_keepalive_secs: keepalive_secs, ..Default::default() };
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    (addr, daemon)
}

struct Reply {
    status: u16,
    head: String,
    body: String,
    head_at: Duration,
    total: Duration,
}

impl Reply {
    fn header(&self, name: &str) -> Option<String> {
        self.head.lines().skip(1).find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim().eq_ignore_ascii_case(name).then(|| v.trim().to_string())
        })
    }
}

fn post(addr: std::net::SocketAddr, path: &str, body: &str) -> Reply {
    let mut s = TcpStream::connect(addr).unwrap();
    let t0 = Instant::now();
    write!(
        s,
        "POST {path} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = Vec::new();
    let mut head_at = None;
    let mut buf = [0u8; 4096];
    loop {
        let n = s.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
        if head_at.is_none() && raw.windows(4).any(|w| w == b"\r\n\r\n") {
            head_at = Some(t0.elapsed());
        }
    }
    let total = t0.elapsed();
    let raw = String::from_utf8(raw).unwrap();
    let (head, payload) = raw.split_once("\r\n\r\n").expect("head");
    let status = head.lines().next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        dechunk(payload)
    } else {
        payload.to_string()
    };
    Reply { status, head: head.to_string(), body, head_at: head_at.unwrap(), total }
}

fn dechunk(mut rest: &str) -> String {
    let mut out = String::new();
    while let Some((size, tail)) = rest.split_once("\r\n") {
        let size = usize::from_str_radix(size.trim(), 16).expect("chunk size");
        if size == 0 {
            break;
        }
        out.push_str(&tail[..size]);
        rest = &tail[size + 2..];
    }
    out
}

fn stable(v: &serde_json::Value) -> serde_json::Value {
    let mut v = v.clone();
    if let Some(o) = v.as_object_mut() {
        o.remove("id");
        o.remove("created");
    }
    v
}

const TICK: Duration = Duration::from_millis(400);

const SLOW_CHAT: &str = r#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"max_tokens":1400,"ignore_eos":true,"temperature":0}"#;
const SLOW_COMPLETION: &str = r#"{"model":"mock-model","prompt":"hello","max_tokens":1400,"ignore_eos":true,"temperature":0}"#;
const SLOW_MESSAGES: &str = r#"{"model":"mock-model","max_tokens":1400,"temperature":0,"messages":[{"role":"user","content":"hello"}]}"#;

fn assert_keepalive_matches_off(path: &str, body: &str) {
    let (on, _) = spawn(1, TICK, None);
    let (off, _) = spawn(0, TICK, None);
    let k = post(on, path, body);
    let p = post(off, path, body);
    assert_eq!(k.status, 200, "{path}: {}", k.body);
    assert_eq!(p.status, 200, "{path}: {}", p.body);
    assert!(
        k.total >= Duration::from_millis(2500),
        "{path}: the fixture must outlast several intervals, took {:?}",
        k.total
    );
    assert!(
        k.head_at + Duration::from_millis(1500) < k.total,
        "{path}: head at {:?} of {:?}",
        k.head_at,
        k.total
    );
    assert_eq!(k.header("x-superfluid-keepalive").as_deref(), Some("whitespace"), "{}", k.head);
    assert_eq!(k.header("content-type").as_deref(), Some("application/json"), "{}", k.head);
    let spaces = k.body.len() - k.body.trim_start().len();
    assert!(spaces >= 1, "{path}: at least one keep-alive space: {:?}", &k.body[..k.body.len().min(40)]);
    assert!(k.body[..spaces].bytes().all(|b| b == b' '), "{path}: only ASCII spaces lead");
    let kv: serde_json::Value = serde_json::from_str(&k.body).expect("whitespace-led JSON parses");
    let pv: serde_json::Value = serde_json::from_str(&p.body).unwrap();
    assert_eq!(stable(&kv), stable(&pv), "{path}: same document as the flag-off server");
    assert!(p.header("content-length").is_some(), "{}", p.head);
    assert!(p.header("x-superfluid-keepalive").is_none(), "{}", p.head);
    assert!(p.body.starts_with('{'), "flag off: no whitespace");
}

#[test]
fn slow_chat_commits_at_admission_then_spaces_then_the_same_json() {
    assert_keepalive_matches_off("/v1/chat/completions", SLOW_CHAT);
}

#[test]
fn slow_completion_commits_at_admission_then_spaces_then_the_same_json() {
    assert_keepalive_matches_off("/v1/completions", SLOW_COMPLETION);
}

#[test]
fn slow_anthropic_message_commits_at_admission_then_spaces_then_the_same_json() {
    assert_keepalive_matches_off("/v1/messages", SLOW_MESSAGES);
}

#[test]
fn keepalive_body_carries_the_usage() {
    let (on, _) = spawn(1, Duration::ZERO, None);
    let r = post(on, "/v1/chat/completions", r#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"max_tokens":7,"ignore_eos":true}"#);
    assert_eq!(r.status, 200);
    let v: serde_json::Value = serde_json::from_str(&r.body).unwrap();
    assert_eq!(v["usage"]["completion_tokens"], 7, "{v}");
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0, "{v}");
}

#[test]
fn refusals_before_admission_keep_their_status() {
    let (on, _) = spawn(1, Duration::ZERO, Some(64));
    let no_keepalive = |r: &Reply| {
        assert!(r.header("x-superfluid-keepalive").is_none(), "{}", r.head);
        assert!(r.body.starts_with('{'), "no whitespace: {:?}", r.body);
    };

    let r = post(on, "/v1/chat/completions", r#"{"model":"mock-model","messages":[]}"#);
    assert_eq!(r.status, 400, "{}", r.body);
    no_keepalive(&r);

    let r = post(on, "/v1/chat/completions", r#"{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(r.status, 404, "{}", r.body);
    no_keepalive(&r);

    let r = post(on, "/v1/chat/completions", "not json");
    assert_eq!(r.status, 400, "{}", r.body);
    no_keepalive(&r);

    let long = "the session stream keeps growing and growing past the window. ".repeat(6);
    let chat = format!(r#"{{"model":"mock-model","messages":[{{"role":"user","content":"{long}"}}]}}"#);
    let r = post(on, "/v1/chat/completions", &chat);
    assert_eq!(r.status, 400, "{}", r.body);
    no_keepalive(&r);
    let v: serde_json::Value = serde_json::from_str(&r.body).unwrap();
    assert_eq!(v["error"]["code"], "context_length_exceeded", "{v}");

    let completion = format!(r#"{{"model":"mock-model","prompt":"{long}"}}"#);
    let r = post(on, "/v1/completions", &completion);
    assert_eq!(r.status, 400, "{}", r.body);
    no_keepalive(&r);
    let v: serde_json::Value = serde_json::from_str(&r.body).unwrap();
    assert_eq!(v["error"]["code"], "context_length_exceeded", "{v}");

    let messages = format!(r#"{{"model":"mock-model","max_tokens":8,"messages":[{{"role":"user","content":"{long}"}}]}}"#);
    let r = post(on, "/v1/messages", &messages);
    assert_eq!(r.status, 400, "{}", r.body);
    no_keepalive(&r);
    let v: serde_json::Value = serde_json::from_str(&r.body).unwrap();
    assert_eq!(v["type"], "error", "Anthropic's shape: {v}");

    let r = post(on, "/v1/chat/completions", r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.header("x-superfluid-keepalive").as_deref(), Some("whitespace"));
    serde_json::from_str::<serde_json::Value>(&r.body).unwrap();
}

#[test]
fn streaming_is_untouched_by_the_flag() {
    let (on, _) = spawn(1, Duration::ZERO, None);
    for (path, body) in [
        ("/v1/chat/completions", r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#),
        ("/v1/completions", r#"{"model":"mock-model","prompt":"hi","max_tokens":4,"stream":true}"#),
        ("/v1/messages", r#"{"model":"mock-model","max_tokens":4,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#),
    ] {
        let r = post(on, path, body);
        assert_eq!(r.status, 200, "{path}: {}", r.body);
        assert!(r.header("x-superfluid-keepalive").is_none(), "{path}: {}", r.head);
        assert_eq!(r.header("content-type").as_deref(), Some("text/event-stream"), "{path}: {}", r.head);
        assert!(r.body.starts_with("data: ") || r.body.starts_with("event: "), "{path}: {:?}", r.body);
    }
}

#[test]
fn fim_completions_are_not_kept_alive() {
    let (on, _) = spawn(1, Duration::ZERO, None);
    let r = post(on, "/v1/completions", r#"{"model":"mock-model","prompt":"def f(","suffix":"):\n  pass","max_tokens":4}"#);
    assert_eq!(r.status, 200, "{}", r.body);
    assert!(r.header("x-superfluid-keepalive").is_none(), "{}", r.head);
    assert!(r.header("content-length").is_some(), "{}", r.head);
    serde_json::from_str::<serde_json::Value>(&r.body).unwrap();
}

#[test]
fn a_disconnect_cancels_the_generation_without_keepalive() {
    let (off, daemon) = spawn(0, Duration::from_millis(50), None);
    let active = daemon.active_registry();
    let body = r#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"max_tokens":3500,"ignore_eos":true}"#;
    let mut s = TcpStream::connect(off).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let t0 = Instant::now();
    while active.lock().unwrap().is_empty() {
        assert!(t0.elapsed() < Duration::from_secs(10), "generation never started");
        std::thread::sleep(Duration::from_millis(10));
    }
    let hung_up = Instant::now();
    drop(s);
    while !active.lock().unwrap().is_empty() {
        assert!(
            hung_up.elapsed() < Duration::from_millis(4500),
            "the generation outlived its client by {:?} (3500 tokens at 50 ms a tick would take minutes)",
            hung_up.elapsed()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let r = post(off, "/v1/chat/completions", r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#);
    assert_eq!(r.status, 200, "{}", r.body);
}

#[test]
fn a_disconnect_cancels_the_generation() {
    let (on, daemon) = spawn(1, Duration::from_millis(500), None);
    let active = daemon.active_registry();
    let body = r#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"max_tokens":3500,"ignore_eos":true}"#;
    let mut s = TcpStream::connect(on).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut head = [0u8; 256];
    let n = s.read(&mut head).unwrap();
    assert!(String::from_utf8_lossy(&head[..n]).starts_with("HTTP/1.1 200"), "{}", String::from_utf8_lossy(&head[..n]));
    let t0 = Instant::now();
    while active.lock().unwrap().is_empty() {
        assert!(t0.elapsed() < Duration::from_secs(10), "generation never started");
        std::thread::sleep(Duration::from_millis(10));
    }
    let hung_up = Instant::now();
    drop(s);
    while !active.lock().unwrap().is_empty() {
        assert!(
            hung_up.elapsed() < Duration::from_millis(4500),
            "the generation outlived its client by {:?}",
            hung_up.elapsed()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let r = post(on, "/v1/chat/completions", r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#);
    assert_eq!(r.status, 200, "{}", r.body);
}

// The clock alone repeats within a microsecond on macOS, and two tests on one
// log are refused while both stores are open.
fn unique() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    format!("{t}-{}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}
