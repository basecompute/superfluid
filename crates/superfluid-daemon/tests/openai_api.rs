//! The OpenAI adapter over the full mock stack.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::{openai, Daemon, EngineHost, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};

fn spawn_stack() -> std::net::SocketAddr {
    spawn_stack_with(Box::new(MockChatCodec))
}

fn spawn_stack_with(codec: Box<dyn superfluid_daemon::TextCodec + Send + Sync>) -> std::net::SocketAddr {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-test-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host =
        EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, codec, 8));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking(listener, daemon, "mock-model".into());
    });
    addr
}

fn spawn_auth_stack(key: &str) -> std::net::SocketAddr {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-auth-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", daemon));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = openai::ServeConfig { api_key: Some(key.to_string()), ..Default::default() };
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    addr
}

fn spawn_config_stack(cfg: openai::ServeConfig) -> std::net::SocketAddr {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-config-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", daemon));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    addr
}

fn spawn_qos_stack(qos: openai::HttpQosPolicy) -> (std::net::SocketAddr, Arc<Daemon>) {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-qos-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let registry =
        Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", Arc::clone(&daemon)));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = openai::ServeConfig { qos, ..Default::default() };
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    (addr, daemon)
}

fn post_hdrs(addr: std::net::SocketAddr, path: &str, body: &str, hdrs: &[(&str, &str)]) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    let extra: String = hdrs.iter().map(|(k, v)| format!("{k}: {v}\r\n")).collect();
    write!(
        s,
        "POST {path} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\n{extra}content-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let raw = String::from_utf8_lossy(&raw).into_owned();
    let (head, payload) = raw.split_once("\r\n\r\n").expect("split");
    let status: u16 = head.lines().next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") { dechunk(payload) } else { payload.to_string() };
    (status, body)
}

fn newest_qos(daemon: &Daemon) -> (u8, bool) {
    let newest = daemon.sessions(true).into_iter().map(|s| s.id).max().expect("a session");
    let i = daemon.inspect(newest).unwrap();
    (i.qos_class, i.batch_invariant)
}

const QOS_CHAT: &str = r#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"max_tokens":3,"temperature":0}"#;

#[test]
fn qos_headers_set_the_class_on_every_generating_route() {
    use superfluid_daemon::qos;
    let (addr, daemon) = spawn_qos_stack(openai::HttpQosPolicy {
        allow_batch_invariant: true,
        ..Default::default()
    });

    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::FOREGROUND_AGENT, false), "no header: the default class");

    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[("x-superfluid-qos", "interactive")]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::INTERACTIVE_CHAT, false));

    let stream = r#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"max_tokens":3,"temperature":0,"stream":true}"#;
    let (st, body) = post_hdrs(
        addr,
        "/v1/chat/completions",
        stream,
        &[("X-Superfluid-Qos", "Background"), ("x-superfluid-batch-invariant", "true")],
    );
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::BACKGROUND_AGENT, true), "headers are case-insensitive");

    let cmpl = r#"{"model":"mock-model","prompt":"once upon a time","max_tokens":3,"temperature":0}"#;
    let (st, body) = post_hdrs(addr, "/v1/completions", cmpl, &[("x-superfluid-qos", "completion")]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::INLINE_COMPLETION, false));

    let cmpl_stream = r#"{"model":"mock-model","prompt":"once upon a time","max_tokens":3,"temperature":0,"stream":true}"#;
    let (st, body) = post_hdrs(addr, "/v1/completions", cmpl_stream, &[("x-superfluid-qos", "3")]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::BACKGROUND_AGENT, false));

    let msgs = r#"{"model":"mock-model","max_tokens":3,"temperature":0,"messages":[{"role":"user","content":"hello"}]}"#;
    let (st, body) = post_hdrs(addr, "/v1/messages", msgs, &[("x-superfluid-qos", "interactive")]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::INTERACTIVE_CHAT, false));

    let msgs_stream = r#"{"model":"mock-model","max_tokens":3,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hello"}]}"#;
    let (st, body) = post_hdrs(addr, "/v1/messages", msgs_stream, &[("x-superfluid-qos", "background")]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::BACKGROUND_AGENT, false));
}

#[test]
fn bad_qos_header_is_a_400_before_any_session() {
    let (addr, daemon) = spawn_qos_stack(openai::HttpQosPolicy::default());
    let before = daemon.sessions(true).len();
    for (path, body) in [
        ("/v1/chat/completions", QOS_CHAT),
        ("/v1/completions", r#"{"model":"mock-model","prompt":"hi","max_tokens":3}"#),
        ("/v1/messages", r#"{"model":"mock-model","max_tokens":3,"messages":[{"role":"user","content":"hi"}]}"#),
    ] {
        let (st, resp) = post_hdrs(addr, path, body, &[("x-superfluid-qos", "urgent")]);
        assert_eq!(st, 400, "{path}: {resp}");
        assert!(resp.contains("interactive, completion, agent, background"), "{path}: {resp}");
    }
    let (st, resp) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[("x-superfluid-batch-invariant", "maybe")]);
    assert_eq!(st, 400, "{resp}");
    assert!(resp.contains("expected true or false"), "{resp}");
    assert_eq!(daemon.sessions(true).len(), before, "a refused request creates no session");
}

#[test]
fn batch_invariant_header_is_refused_unless_the_operator_allows_it() {
    use superfluid_daemon::qos;
    let (addr, daemon) = spawn_qos_stack(openai::HttpQosPolicy::default());
    let before = daemon.sessions(true).len();
    for (path, body) in [
        ("/v1/chat/completions", QOS_CHAT),
        ("/v1/completions", r#"{"model":"mock-model","prompt":"hi","max_tokens":3}"#),
        ("/v1/messages", r#"{"model":"mock-model","max_tokens":3,"messages":[{"role":"user","content":"hi"}]}"#),
    ] {
        let (st, resp) = post_hdrs(addr, path, body, &[("x-superfluid-batch-invariant", "true")]);
        assert_eq!(st, 403, "{path}: {resp}");
        assert!(resp.contains("--http-allow-batch-invariant"), "{path}: {resp}");
    }
    assert_eq!(daemon.sessions(true).len(), before, "a refused request creates no session");

    let (st, body) =
        post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[("x-superfluid-batch-invariant", "false")]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::FOREGROUND_AGENT, false));
}

#[test]
fn operator_default_applies_and_headers_can_be_ignored() {
    use superfluid_daemon::qos;
    let background = openai::RequestQos { class: qos::BACKGROUND_AGENT, batch_invariant: false };

    let (addr, daemon) = spawn_qos_stack(openai::HttpQosPolicy {
        default: background,
        honor_headers: true,
        allow_batch_invariant: false,
    });
    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::BACKGROUND_AGENT, false), "operator default");
    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[("x-superfluid-qos", "agent")]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::FOREGROUND_AGENT, false), "a header overrides the default");

    let (addr, daemon) = spawn_qos_stack(openai::HttpQosPolicy {
        default: background,
        honor_headers: false,
        allow_batch_invariant: false,
    });
    for hdrs in [
        &[("x-superfluid-qos", "interactive"), ("x-superfluid-batch-invariant", "true")][..],
        &[("x-superfluid-qos", "urgent")][..],
    ] {
        let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, hdrs);
        assert_eq!(st, 200, "{body}");
        assert_eq!(newest_qos(&daemon), (qos::BACKGROUND_AGENT, false), "headers ignored: {hdrs:?}");
    }
}

#[test]
fn a_qos_class_does_not_change_the_rendered_prompt() {
    let (addr, _daemon) = spawn_qos_stack(openai::HttpQosPolicy::default());
    let body = r#"{"model":"mock-model","messages":[{"role":"user","content":"hello there"}],"max_tokens":8,"temperature":0}"#;
    let content = |hdrs: &[(&str, &str)]| {
        let (st, resp) = post_hdrs(addr, "/v1/chat/completions", body, hdrs);
        assert_eq!(st, 200, "{resp}");
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        (v["choices"][0]["message"]["content"].clone(), v["usage"]["prompt_tokens"].clone())
    };
    let base = content(&[]);
    for class in ["interactive", "background"] {
        assert_eq!(content(&[("x-superfluid-qos", class)]), base, "class {class} changed the prompt or the reply");
    }
}

fn http_hdr(
    addr: std::net::SocketAddr,
    method: &str,
    path: &str,
    hdr: Option<(&str, &str)>,
) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    let extra = hdr.map(|(k, v)| format!("{k}: {v}\r\n")).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\n{extra}content-type: application/json\r\ncontent-length: 2\r\n\r\n{{}}").unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let raw = String::from_utf8_lossy(&raw).into_owned();
    let (head, payload) = raw.split_once("\r\n\r\n").expect("split");
    let status: u16 = head.lines().next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") { dechunk(payload) } else { payload.to_string() };
    (status, body)
}

#[test]
fn api_key_guards_every_route_except_health() {
    let addr = spawn_auth_stack("sk-secret-key");

    let (st, body) = http_hdr(addr, "GET", "/v1/models", None);
    assert_eq!(st, 401, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "authentication_error");
    assert_eq!(v["error"]["code"], "invalid_api_key");
    assert_eq!(v["error"]["message"], "Invalid or missing API key");

    let (st, _b) = http_hdr(addr, "GET", "/v1/models", Some(("authorization", "Bearer wrong")));
    assert_eq!(st, 401);
    let (st, _b) = http_hdr(addr, "GET", "/v1/models", Some(("authorization", "Bearer sk-secret")));
    assert_eq!(st, 401);

    let (st, body) = http_hdr(addr, "GET", "/v1/models", Some(("authorization", "Bearer sk-secret-key")));
    assert_eq!(st, 200, "{body}");
    let (st, body) = http_hdr(addr, "GET", "/v1/models", Some(("x-api-key", "sk-secret-key")));
    assert_eq!(st, 200, "{body}");

    let (st, body) = http_hdr(addr, "GET", "/health", None);
    assert_eq!(st, 200, "{body}");

    let (st, _b) = http_hdr(addr, "GET", "/v1/nope", None);
    assert_eq!(st, 401);
}

#[test]
fn an_openai_only_surface_serves_no_anthropic_or_ollama_route() {
    let all = spawn_config_stack(openai::ServeConfig::default());
    let openai_only =
        spawn_config_stack(openai::ServeConfig { surface: openai::HttpSurface::OPENAI_ONLY, ..Default::default() });
    let messages = r#"{"model":"mock-model","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}"#;
    for (method, path, body) in
        [("POST", "/v1/messages", messages), ("GET", "/api/tags", ""), ("GET", "/api/version", ""), ("GET", "/", "")]
    {
        let (st, _, b) = http(all, method, path, body);
        assert_ne!(st, 404, "{method} {path} is served by default: {b}");
        let (st, _, b) = http(openai_only, method, path, body);
        assert_eq!(st, 404, "{method} {path} is not served beside the OpenAI API alone: {b}");
    }
    let (st, _, b) = http(openai_only, "GET", "/v1/models", "");
    assert_eq!(st, 200, "{b}");
    let chat = r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
    let (st, _, b) = http(openai_only, "POST", "/v1/chat/completions", chat);
    assert_eq!(st, 200, "{b}");
}

#[test]
fn the_server_names_itself_as_its_embedder_says() {
    let default = spawn_config_stack(openai::ServeConfig::default());
    let named = spawn_config_stack(openai::ServeConfig {
        identity: openai::ServerIdentity { name: "baseRT".into(), version: "9.9.9".into() },
        ..Default::default()
    });
    for (addr, name, version) in [(default, "superfluid", env!("CARGO_PKG_VERSION")), (named, "baseRT", "9.9.9")] {
        let (st, _, b) = http(addr, "GET", "/v1/models", "");
        assert_eq!(st, 200, "{b}");
        let models: serde_json::Value = serde_json::from_str(&b).unwrap();
        assert_eq!(models["data"][0]["owned_by"], name, "{b}");
        let (st, _, b) = http(addr, "GET", "/v1/models/mock-model", "");
        assert_eq!(st, 200, "{b}");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&b).unwrap()["owned_by"], name, "{b}");
        let (st, _, b) = http(addr, "GET", "/props", "");
        assert_eq!(st, 200, "{b}");
        let props: serde_json::Value = serde_json::from_str(&b).unwrap();
        assert_eq!((props["server"].as_str(), props["build"]["version"].as_str()), (Some(name), Some(version)), "{b}");
    }
}

#[test]
fn loose_model_naming_takes_no_name_any_case_or_part_of_one() {
    let exact = spawn_config_stack(openai::ServeConfig::default());
    let loose = spawn_config_stack(openai::ServeConfig { model_naming: openai::ModelNaming::Loose, ..Default::default() });
    let chat = |model: Option<&str>| {
        let mut body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4});
        if let Some(m) = model {
            body["model"] = m.into();
        }
        body.to_string()
    };
    for (model, by_name, loosely) in [(None, 400, 200), (Some("MOCK-MODEL"), 404, 200), (Some("mock"), 404, 200), (Some("nope"), 404, 404)] {
        let (st, _, b) = http(exact, "POST", "/v1/chat/completions", &chat(model));
        assert_eq!(st, by_name, "{model:?} named exactly: {b}");
        let (st, _, b) = http(loose, "POST", "/v1/chat/completions", &chat(model));
        assert_eq!(st, loosely, "{model:?} named loosely: {b}");
        if st == 200 {
            assert!(b.contains(r#""model":"mock-model""#), "the reply names the model that served it: {b}");
        }
    }
}

#[test]
fn model_dir_autoloads_on_first_use_and_idle_timeout_frees_it() {
    use superfluid_daemon::registry::{ModelLoader, ModelRegistry};
    let loader: ModelLoader = Box::new(|_id: &str, _runtime| Ok(mock_daemon()));
    let registry = ModelRegistry::with_initial("model-a", mock_daemon(), loader);
    registry.register_known("model-b", std::path::Path::new("/models/model-b.base"));

    assert!(registry.is_known("model-b"));
    assert!(!registry.is_loaded("model-b"));
    assert_eq!(registry.all_names(), vec!["model-a", "model-b"]);
    assert_eq!(registry.names(), vec!["model-a"], "not loaded yet");

    let (name, _d) = registry.resolve(Some("model-b")).expect("autoloads on first use");
    assert_eq!(name, "model-b");
    assert!(registry.is_loaded("model-b"));

    assert!(registry.resolve(Some("model-zzz")).is_none());

    registry.track_idle();
    let freed = registry.sweep_idle(std::time::Duration::from_secs(0));
    assert_eq!(freed, vec!["model-b"]);
    assert!(!registry.is_loaded("model-b"));
    assert!(registry.resolve(Some("model-b")).is_some());

    let freed = registry.sweep_idle(std::time::Duration::from_secs(0));
    assert!(!freed.contains(&"model-a".to_string()), "swept the default: {freed:?}");
    assert!(registry.resolve(None).is_some());
}

#[test]
fn default_max_tokens_fills_the_remaining_context() {
    use superfluid_daemon::openai::TokenLimits;
    let fill = TokenLimits { default_max_tokens: None, max_context: 8192 };
    assert_eq!(fill.resolve(None, || 100), 8092);
    assert_eq!(fill.resolve(None, || 0), 8192);
    assert_eq!(fill.resolve(Some(16), || 100), 16);
    assert_eq!(fill.resolve(None, || 8192), 1);
    assert_eq!(fill.resolve(None, || 99999), 1);

    let pinned = TokenLimits { default_max_tokens: Some(256), max_context: 8192 };
    assert_eq!(pinned.resolve(None, || 100), 256);
    assert_eq!(pinned.resolve(Some(9), || 100), 9);
}

#[test]
fn rate_limit_throttles_then_exempts_probes_and_metrics() {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-rl-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", daemon));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = openai::ServeConfig { rate_limit_per_minute: 3, ..Default::default() };
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });

    for i in 0..3 {
        let (st, body) = http_hdr(addr, "GET", "/v1/models", None);
        assert_eq!(st, 200, "request {i} should be within the allowance: {body}");
    }
    let (st, body) = http_hdr(addr, "GET", "/v1/models", None);
    assert_eq!(st, 429, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "rate_limit_error");
    assert_eq!(v["error"]["code"], "rate_limit_exceeded");

    let (st, _b) = http_hdr(addr, "GET", "/health", None);
    assert_eq!(st, 200);
    let (st, _b) = http_hdr(addr, "GET", "/metrics", None);
    assert_eq!(st, 200);
    let (st, _b) = http_hdr(addr, "GET", "/v1/metrics", None);
    assert_eq!(st, 200);
}

fn spawn_key_stack(
    policy: &str,
    api_key: Option<&str>,
    rate_limit_per_minute: u32,
) -> (std::net::SocketAddr, Arc<Daemon>, Arc<superfluid_daemon::keypolicy::KeyTable>) {
    spawn_key_stack_on(policy, api_key, rate_limit_per_minute, EngineConfig::default())
}

fn spawn_key_stack_on(
    policy: &str,
    api_key: Option<&str>,
    rate_limit_per_minute: u32,
    engine: EngineConfig,
) -> (std::net::SocketAddr, Arc<Daemon>, Arc<superfluid_daemon::keypolicy::KeyTable>) {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-keys-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || (MockEngine::new(engine), None)).expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let registry =
        Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", Arc::clone(&daemon)));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let table = Arc::new(superfluid_daemon::keypolicy::KeyTable::parse(policy, |_| None).unwrap());
    let cfg = openai::ServeConfig {
        api_key: api_key.map(str::to_string),
        rate_limit_per_minute,
        qos: openai::HttpQosPolicy { honor_headers: false, ..Default::default() },
        keys: Some(Arc::clone(&table)),
        ..Default::default()
    };
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    (addr, daemon, table)
}

const KEYS: &str = r#"{"keys":[
    {"name":"ui","key":"sk-ui","class":"interactive"},
    {"name":"agents","key":"sk-agents","class":"agent","max_class":"completion","batch_invariant":true},
    {"name":"batch","key":"sk-batch","class":"background"}
]}"#;

#[test]
fn key_policy_authenticates_and_assigns_each_key_its_class() {
    use superfluid_daemon::qos;
    let (addr, daemon, _t) = spawn_key_stack(KEYS, Some("sk-global"), 0);

    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[]);
    assert_eq!(st, 401, "{body}");
    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[("authorization", "Bearer sk-nope")]);
    assert_eq!(st, 401, "{body}");
    let (st, _b) = http_hdr(addr, "GET", "/health", None);
    assert_eq!(st, 200);

    for (key, class) in [
        ("sk-ui", qos::INTERACTIVE_CHAT),
        ("sk-agents", qos::FOREGROUND_AGENT),
        ("sk-batch", qos::BACKGROUND_AGENT),
    ] {
        let bearer = format!("Bearer {key}");
        let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[("authorization", &bearer)]);
        assert_eq!(st, 200, "{key}: {body}");
        assert_eq!(newest_qos(&daemon).0, class, "{key}");
    }
    let msgs = r#"{"model":"mock-model","max_tokens":3,"messages":[{"role":"user","content":"hi"}]}"#;
    let (st, body) = post_hdrs(addr, "/v1/messages", msgs, &[("x-api-key", "sk-batch")]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon).0, qos::BACKGROUND_AGENT);

    let (st, body) = post_hdrs(
        addr,
        "/v1/chat/completions",
        QOS_CHAT,
        &[("authorization", "Bearer sk-global"), ("x-superfluid-qos", "interactive")],
    );
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon).0, qos::FOREGROUND_AGENT);
}

#[test]
fn key_policy_caps_the_qos_header_per_key() {
    use superfluid_daemon::qos;
    let (addr, daemon, table) = spawn_key_stack(KEYS, None, 0);
    let agents = [("authorization", "Bearer sk-agents")];

    let with = |extra: (&'static str, &'static str)| [agents[0], extra];
    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &with(("x-superfluid-qos", "completion")));
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::INLINE_COMPLETION, false), "promotion up to max_class");
    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &with(("x-superfluid-qos", "background")));
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon).0, qos::BACKGROUND_AGENT, "demotion is always allowed");

    let before = daemon.sessions(true).len();
    for path in ["/v1/chat/completions", "/v1/completions"] {
        let body = if path == "/v1/completions" {
            r#"{"model":"mock-model","prompt":"hi","max_tokens":3}"#
        } else {
            QOS_CHAT
        };
        let (st, resp) = post_hdrs(addr, path, body, &with(("x-superfluid-qos", "interactive")));
        assert_eq!(st, 403, "{path}: {resp}");
        assert!(resp.contains("at most the completion class"), "{resp}");
    }
    assert_eq!(daemon.sessions(true).len(), before, "a refused request starts no session");

    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &with(("x-superfluid-batch-invariant", "true")));
    assert_eq!(st, 200, "{body}");
    assert_eq!(newest_qos(&daemon), (qos::FOREGROUND_AGENT, true), "this key may ask for batch invariance");
    let (st, body) = post_hdrs(
        addr,
        "/v1/chat/completions",
        QOS_CHAT,
        &[("authorization", "Bearer sk-ui"), ("x-superfluid-batch-invariant", "true")],
    );
    assert_eq!(st, 403, "{body}");
    assert!(
        body.contains("\\\"batch_invariant\\\": true on this key") && !body.contains("--http-allow-batch-invariant"),
        "a keyed denial names the key's permission, not the global flag: {body}"
    );

    let m = table.metrics_text();
    assert!(m.contains("superfluid_key_rejected_total{key=\"agents\",reason=\"qos\"} 2"), "{m}");
    assert!(m.contains("superfluid_key_rejected_total{key=\"ui\",reason=\"qos\"} 1"), "{m}");
}

#[test]
fn key_policy_rate_limits_per_key_and_releases_slots() {
    let policy = r#"{"keys":[
        {"name":"gw","key":"sk-gw","rate_limit_rpm":4,"max_concurrent":2},
        {"name":"plain","key":"sk-plain"}
    ]}"#;
    let (addr, _daemon, table) = spawn_key_stack(policy, None, 2);
    let gw = [("authorization", "Bearer sk-gw")];
    for i in 0..4 {
        let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &gw);
        assert_eq!(st, 200, "gw request {i} is within the key's allowance, past the IP's: {body}");
    }
    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &gw);
    assert_eq!(st, 429, "{body}");
    assert!(body.contains("API key 'gw' exceeded its rate limit (4/min)"), "{body}");

    let plain = [("authorization", "Bearer sk-plain")];
    for i in 0..2 {
        let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &plain);
        assert_eq!(st, 200, "plain request {i}: {body}");
    }
    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &plain);
    assert_eq!(st, 429, "a key with no allowance of its own pays the per-IP limit: {body}");

    let gw_key = table.keys().find(|k| k.name == "gw").unwrap();
    assert_eq!(gw_key.in_flight(), 0);

    let (st, _b) = http_hdr(addr, "GET", "/metrics", Some(("authorization", "Bearer sk-gw")));
    assert_eq!(st, 200, "metrics spend no key allowance");
    let (st, m) = http_hdr(addr, "GET", "/metrics", Some(("authorization", "Bearer sk-gw")));
    assert_eq!(st, 200);
    assert!(m.contains("superfluid_key_requests_total{key=\"gw\"} 5"), "{m}");
    assert!(m.contains("superfluid_key_rejected_total{key=\"gw\",reason=\"rate_limit\"} 1"), "{m}");
    assert!(m.contains("superfluid_key_in_flight{key=\"gw\"} 0"), "{m}");
    assert!(!m.contains("sk-gw"), "the secret never reaches metrics");
}

#[test]
fn key_policy_slot_outlives_a_disconnect_while_work_runs() {
    let policy = r#"{"keys":[{"name":"one","key":"sk-one","max_concurrent":1}]}"#;
    let engine = EngineConfig { tick_delay: std::time::Duration::from_millis(50), ..EngineConfig::default() };
    let (addr, daemon, table) = spawn_key_stack_on(policy, None, 0, engine);
    let active = daemon.active_registry();
    let key = table.keys().next().unwrap().clone();
    let body = r#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"max_tokens":3500,"ignore_eos":true}"#;
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\nauthorization: Bearer sk-one\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let t0 = std::time::Instant::now();
    while key.in_flight() == 0 {
        assert!(t0.elapsed() < std::time::Duration::from_secs(20), "request never took its slot");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    while active.lock().unwrap().is_empty() {
        assert!(t0.elapsed() < std::time::Duration::from_secs(20), "generation never started");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    drop(s);
    // The disconnect cancels the work; the slot belongs to the work, not the
    // connection, so it frees only once the generation has wound down.
    let t0 = std::time::Instant::now();
    loop {
        let working = !active.lock().unwrap().is_empty();
        let held = key.in_flight() != 0;
        assert!(held || !working, "the slot was released while its generation still ran");
        if !held {
            break;
        }
        assert!(t0.elapsed() < std::time::Duration::from_secs(10), "slot never released after the disconnect cancelled the work");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let (st, resp) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[("authorization", "Bearer sk-one")]);
    assert_eq!(st, 200, "{resp}");
}

#[test]
fn key_policy_stream_holds_its_slot_until_the_body_ends() {
    let policy = r#"{"keys":[{"name":"one","key":"sk-one","max_concurrent":1}]}"#;
    let (addr, _daemon, table) = spawn_key_stack(policy, None, 0);
    let key = table.keys().next().unwrap().clone();
    let body = r#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"max_tokens":4000,"stream":true}"#;
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\nauthorization: Bearer sk-one\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut first = [0u8; 64];
    let n = s.read(&mut first).unwrap();
    assert!(String::from_utf8_lossy(&first[..n]).starts_with("HTTP/1.1 200"));
    assert_eq!(key.in_flight(), 1, "the unread stream holds the slot");
    let (st, resp) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[("authorization", "Bearer sk-one")]);
    assert_eq!(st, 429, "{resp}");
    assert!(resp.contains("concurrent-request limit (1)"), "{resp}");
    drop(s);
    let t0 = std::time::Instant::now();
    while key.in_flight() != 0 {
        assert!(t0.elapsed() < std::time::Duration::from_secs(20), "slot leaked after disconnect");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let (st, resp) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[("authorization", "Bearer sk-one")]);
    assert_eq!(st, 200, "{resp}");
}

#[test]
fn file_store_quota_refuses_413_and_expiry_reclaims() {
    use superfluid_daemon::files::FileStore;
    let dir = std::env::temp_dir().join(format!(
        "superfluid-files-quota-{}-{}",
        std::process::id(),
        unique()
    ));
    let store = FileStore::with_limits(dir.clone(), Some(100), None);
    store.put("a.bin", "batch", &[0u8; 60], None).expect("first fits");
    assert_eq!(store.stored_bytes(), 60);
    let err = store.put("b.bin", "batch", &[0u8; 60], None).unwrap_err();
    assert!(matches!(err, superfluid_daemon::DaemonError::FilesQuota { max: 100, .. }), "{err}");
    assert_eq!(store.stored_bytes(), 60);
    store.put("c.bin", "batch", &[0u8; 40], None).expect("exactly at the quota fits");
    assert_eq!(store.stored_bytes(), 100);

    let expiring = FileStore::with_limits(dir.clone(), Some(100), Some(60));
    assert_eq!(expiring.sweep(), 0, "nothing is stale yet");
    for m in expiring.list() {
        let path = dir.join(format!("{}.json", m.id));
        let mut v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        v["created_at"] = serde_json::json!(m.created_at - 3600);
        std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
    }
    assert_eq!(expiring.sweep(), 2);
    assert_eq!(expiring.stored_bytes(), 0);

    expiring.put("d.bin", "batch", &[0u8; 80], None).expect("space was reclaimed");
}

fn spawn_template_media_stack() -> (std::net::SocketAddr, Arc<Daemon>) {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-tmpl-media-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| {
        (
            MockEngine::new(EngineConfig {
                image_token_id: 999,
                media_tokens_per_image: 4,
                ..EngineConfig::default()
            }),
            None,
        )
    })
    .expect("spawn");
    let daemon = Arc::new(
        Daemon::with_options(
            store,
            host,
            Box::new(superfluid_daemon::codec::MockTemplateCodec),
            superfluid_daemon::DaemonOptions {
                media_dir: Some(dir.join("media")),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let d = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = openai::serve_blocking(listener, d, "mock-model".into());
    });
    (addr, daemon)
}

fn spawn_media_stack() -> (std::net::SocketAddr, Arc<Daemon>) {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-media-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| {
        (
            MockEngine::new(EngineConfig {
                image_token_id: 999,
                media_tokens_per_image: 4,
                ..EngineConfig::default()
            }),
            None,
        )
    })
    .expect("spawn");
    let daemon = Arc::new(
        Daemon::with_options(
            store,
            host,
            Box::new(MockChatCodec),
            superfluid_daemon::DaemonOptions {
                media_dir: Some(dir.join("media")),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let d = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = openai::serve_blocking(listener, d, "mock-model".into());
    });
    (addr, daemon)
}

fn http(addr: std::net::SocketAddr, method: &str, path: &str, body: &str) -> (u16, String, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).unwrap();
    let (head, payload) = raw.split_once("\r\n\r\n").expect("header split");
    let status: u16 = head
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        dechunk(payload)
    } else {
        payload.to_string()
    };
    (status, head.to_string(), body)
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
fn non_streaming_completion_shape_and_usage() {
    let addr = spawn_stack();
    let (status, head, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hello"}],"max_tokens":6,"temperature":0}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    assert!(head.to_ascii_lowercase().contains("x-superfluid-warm:"));
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert!(v["choices"][0]["finish_reason"].is_string());
    let u = &v["usage"];
    assert_eq!(u["completion_tokens"], 6);
    assert_eq!(
        u["total_tokens"].as_u64().unwrap(),
        u["prompt_tokens"].as_u64().unwrap() + 6
    );
}

#[test]
fn usage_reports_cached_tokens_as_the_warm_prefix() {
    let addr = spawn_stack();
    let req = r#"{"model":"mock-model","messages":[{"role":"user","content":"a shared prompt for the cache"}],"max_tokens":4,"temperature":0}"#;
    for _ in 0..2 {
        let (status, head, body) = http(addr, "POST", "/v1/chat/completions", req);
        assert_eq!(status, 200, "body: {body}");
        let warm: u64 = head
            .lines()
            .find_map(|l| l.to_ascii_lowercase().strip_prefix("x-superfluid-warm:").map(|v| v.trim().to_string()))
            .expect("x-superfluid-warm header")
            .parse()
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let u = &v["usage"];
        let cached = u["prompt_tokens_details"]["cached_tokens"].as_u64().expect("cached_tokens");
        assert_eq!(cached, warm.min(u["prompt_tokens"].as_u64().unwrap()));
    }
    let stream_req = req.replace(r#""temperature":0"#, r#""temperature":0,"stream":true"#);
    let (status, _, body) = http(addr, "POST", "/v1/chat/completions", &stream_req);
    assert_eq!(status, 200);
    let events: Vec<&str> = body.split("\n\n").filter_map(|b| b.strip_prefix("data: ")).collect();
    let terminal: serde_json::Value = serde_json::from_str(events[events.len() - 2]).unwrap();
    let u = &terminal["usage"];
    assert_eq!(
        u["prompt_tokens_details"]["cached_tokens"].as_u64().expect("cached_tokens"),
        terminal["superfluid"]["warm"].as_u64().unwrap().min(u["prompt_tokens"].as_u64().unwrap())
    );
}

#[test]
fn streaming_sse_framing() {
    let addr = spawn_stack();
    let (status, head, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":40,"stream":true,"temperature":0}"#,
    );
    assert_eq!(status, 200);
    assert!(head.to_ascii_lowercase().contains("text/event-stream"));
    let events: Vec<&str> = body
        .split("\n\n")
        .filter_map(|b| b.strip_prefix("data: "))
        .collect();
    assert!(events.len() >= 3, "role chunk + terminal + DONE, got {events:?}");
    assert_eq!(*events.last().unwrap(), "[DONE]");
    let first: serde_json::Value = serde_json::from_str(events[0]).unwrap();
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    let terminal: serde_json::Value =
        serde_json::from_str(events[events.len() - 2]).unwrap();
    assert!(terminal["choices"][0]["finish_reason"].is_string());
    assert_eq!(terminal["usage"]["completion_tokens"], 40);
    for e in &events[..events.len() - 1] {
        let v: serde_json::Value = serde_json::from_str(e).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["id"], first["id"]);
    }
}

#[test]
fn chat_stream_delivers_token_cadence_deltas() {
    let script: Vec<u32> = "streaming!".bytes().map(|b| 0x100 + b as u32).collect();
    let addr = spawn_scripted_stack(script);
    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":40,"stream":true,"temperature":0}"#,
    );
    assert_eq!(status, 200);
    let mut streamed = String::new();
    let mut content_chunks = 0usize;
    for e in body.split("\n\n").filter_map(|b| b.strip_prefix("data: ")) {
        if e == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(e).unwrap();
        if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
            streamed.push_str(t);
            content_chunks += 1;
        }
    }
    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":40,"temperature":0}"#,
    );
    assert_eq!(status, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let full = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert_eq!(streamed, full, "streamed concatenation equals the non-streaming result");
    assert_eq!(streamed, "streaming!");
    assert!(
        content_chunks >= 5,
        "token-cadence deltas expected (got {content_chunks} content chunks for: {streamed:?})"
    );
}

#[test]
fn chat_stream_continuous_usage_counts_tokens_per_chunk() {
    let script: Vec<u32> = "streaming!".bytes().map(|b| 0x100 + b as u32).collect();
    let addr = spawn_scripted_stack(script);
    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":40,"stream":true,"temperature":0,"stream_options":{"include_usage":true,"continuous_usage_stats":true}}"#,
    );
    assert_eq!(status, 200);
    let (mut counts, mut terminal) = (Vec::new(), None);
    for e in body.split("\n\n").filter_map(|b| b.strip_prefix("data: ")) {
        if e == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(e).unwrap();
        if v["choices"][0]["delta"]["content"].as_str().is_some() {
            counts.push(v["usage"]["completion_tokens"].as_u64().expect("usage on every content chunk"));
        }
        if v["choices"][0]["finish_reason"].is_string() {
            terminal = v["usage"]["completion_tokens"].as_u64();
        }
    }
    assert!(counts.len() >= 5, "token-cadence chunks expected: {counts:?}");
    assert!(counts.windows(2).all(|w| w[0] <= w[1]), "monotonic: {counts:?}");
    assert_eq!(counts.last().copied(), terminal, "last chunk count equals terminal usage");

    let (_, _, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":40,"stream":true,"temperature":0}"#,
    );
    for e in body.split("\n\n").filter_map(|b| b.strip_prefix("data: ")) {
        if e == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(e).unwrap();
        if v["choices"][0]["delta"]["content"].as_str().is_some() {
            assert!(v["usage"].is_null(), "no per-chunk usage unless asked: {v}");
        }
    }
}

#[test]
fn ignore_eos_decodes_to_max_tokens() {
    let script: Vec<u32> = "streaming!".bytes().map(|b| 0x100 + b as u32).collect();
    let addr = spawn_scripted_stack(script);
    let completion = |body: &str| -> (u64, String) {
        let (status, _, resp) = http(addr, "POST", "/v1/chat/completions", body);
        assert_eq!(status, 200, "{resp}");
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        (
            v["usage"]["completion_tokens"].as_u64().unwrap(),
            v["choices"][0]["finish_reason"].as_str().unwrap().to_string(),
        )
    };
    let (n, fin) = completion(
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":20,"temperature":0}"#,
    );
    assert!(n < 20 && fin == "stop", "without ignore_eos the script's EOS ends it: {n} {fin}");
    let (n, fin) = completion(
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":20,"temperature":0,"ignore_eos":true}"#,
    );
    assert_eq!((n, fin.as_str()), (20, "length"));

    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":20,"temperature":0,"ignore_eos":true,"stream":true,"stream_options":{"include_usage":true}}"#,
    );
    assert_eq!(status, 200);
    let terminal = body
        .split("\n\n")
        .filter_map(|b| b.strip_prefix("data: "))
        .filter(|e| *e != "[DONE]")
        .map(|e| serde_json::from_str::<serde_json::Value>(e).unwrap())
        .find(|v| v["choices"][0]["finish_reason"].is_string())
        .expect("terminal chunk");
    assert_eq!(terminal["usage"]["completion_tokens"], 20);
    assert_eq!(terminal["choices"][0]["finish_reason"], "length");
}

#[test]
fn chat_and_completions_accept_ignore_eos() {
    let addr = spawn_stack();
    for (path, body) in [
        ("/v1/chat/completions", r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"ignore_eos":true}"#),
        ("/v1/completions", r#"{"model":"mock-model","prompt":"hi","max_tokens":4,"ignore_eos":true}"#),
    ] {
        let (status, _, resp) = http(addr, "POST", path, body);
        assert_eq!(status, 200, "{path}: {resp}");
    }
}

#[test]
fn penalties_outside_openai_bounds_are_refused() {
    let addr = spawn_stack();
    let chat = |extra: &str| {
        format!(r#"{{"model":"mock-model","messages":[{{"role":"user","content":"hi"}}],"max_tokens":4,"ignore_eos":true,{extra}}}"#)
    };
    for (path, body) in [
        ("/v1/chat/completions", chat(r#""frequency_penalty":-2.5"#)),
        ("/v1/chat/completions", chat(r#""presence_penalty":2.5"#)),
        (
            "/v1/completions",
            r#"{"model":"mock-model","prompt":"hi","max_tokens":4,"ignore_eos":true,"frequency_penalty":-3}"#.to_string(),
        ),
    ] {
        let (status, _, resp) = http(addr, "POST", path, &body);
        assert_eq!(status, 400, "{path}: {resp}");
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error", "{resp}");
        assert!(v["error"]["message"].as_str().unwrap().contains("must be in [-2, 2]"), "{resp}");
    }
    // The bounds themselves, and a null (an unset optional, as OpenAI clients
    // send it), are taken.
    for extra in [r#""frequency_penalty":-2,"presence_penalty":2"#, r#""frequency_penalty":null"#] {
        let (status, _, resp) = http(addr, "POST", "/v1/chat/completions", &chat(extra));
        assert_eq!(status, 200, "{extra}: {resp}");
    }
}

#[test]
fn models_route_and_typed_errors() {
    let addr = spawn_stack();
    let (status, _, body) = http(addr, "GET", "/v1/models", "");
    assert_eq!(status, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["data"][0]["id"], "mock-model");

    let (status, _, body) = http(addr, "POST", "/v1/chat/completions", "{not json");
    assert_eq!(status, 400);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");

    let (status, _, _) = http(addr, "GET", "/v1/nope", "");
    assert_eq!(status, 404);

    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[]}"#,
    );
    assert_eq!(status, 400, "body: {body}");

    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(status, 400, "missing model must be refused: {body}");
    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(status, 404, "unserved model must be refused: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("mock-model"));
}

fn spawn_scripted_stack(script: Vec<u32>) -> std::net::SocketAddr {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-script-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || {
        let cfg = EngineConfig { scripted: script.clone(), ..Default::default() };
        (MockEngine::new(cfg), None)
    })
    .expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking(listener, daemon, "mock-model".into());
    });
    addr
}

fn spawn_scripted_stack_with(
    script: Vec<u32>,
    codec: Box<dyn superfluid_daemon::codec::TextCodec + Send + Sync>,
) -> std::net::SocketAddr {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-script-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || {
        let cfg = EngineConfig { scripted: script.clone(), ..Default::default() };
        (MockEngine::new(cfg), None)
    })
    .expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, codec, 8));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking(listener, daemon, "mock-model".into());
    });
    addr
}

#[test]
fn template_generation_prompt_primes_the_reasoning_channel() {
    use superfluid_daemon::codec::{MockThinkingTemplateCodec, TextCodec as _, MOCK_THINK_CLOSE};
    let mut script = superfluid_daemon::codec::MockCodec.encode("thoughts");
    script.push(MOCK_THINK_CLOSE);
    script.extend(superfluid_daemon::codec::MockCodec.encode("Hello"));
    let addr = spawn_scripted_stack_with(script, Box::new(MockThinkingTemplateCodec));
    let body = r#"{"model":"mock-model","max_tokens":64,"temperature":0,
        "messages":[{"role":"user","content":"hi"}]}"#;
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["reasoning_content"], "thoughts", "the lane did not start in REASONING: {msg}");
    assert_eq!(msg["content"], "Hello", "{msg}");
}

#[test]
fn template_generation_prompt_primes_the_reasoning_channel_with_media() {
    use superfluid_daemon::codec::{MockThinkingTemplateCodec, TextCodec as _, MOCK_THINK_CLOSE};
    use superfluid_daemon::EventBody;
    let mut script = superfluid_daemon::codec::MockCodec.encode("thoughts");
    script.push(MOCK_THINK_CLOSE);
    script.extend(superfluid_daemon::codec::MockCodec.encode("Hello"));
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-think-media-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || {
        let cfg = EngineConfig {
            scripted: script.clone(),
            image_token_id: 999,
            media_tokens_per_image: 4,
            ..EngineConfig::default()
        };
        (MockEngine::new(cfg), None)
    })
    .expect("spawn");
    let daemon = Arc::new(
        Daemon::with_options(
            store,
            host,
            Box::new(MockThinkingTemplateCodec),
            superfluid_daemon::DaemonOptions { media_dir: Some(dir.join("media")), ..Default::default() },
        )
        .unwrap(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let d = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = openai::serve_blocking(listener, d, "mock-model".into());
    });
    let body = r#"{"model":"mock-model","max_tokens":64,"temperature":0,
        "messages":[{"role":"user","content":[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8gaW1hZ2U="}}]}]}"#;
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["reasoning_content"], "thoughts", "the lane did not start in REASONING: {msg}");
    assert_eq!(msg["content"], "Hello", "{msg}");
    let store = daemon.store();
    let store = store.lock().unwrap();
    let id = store.session_ids().into_iter().next().expect("a session");
    let kinds: Vec<&str> = store.session(id).unwrap().events.iter().map(|e| match &e.body {
        EventBody::Appended { .. } => "appended",
        EventBody::Block { .. } => "block",
        EventBody::GenerationPrompt { .. } => "opener",
        EventBody::Generated { .. } => "generated",
        _ => "other",
    }).collect();
    let opener_at = kinds.iter().position(|k| *k == "opener").expect("opener recorded: {kinds:?}");
    let block_at = kinds.iter().position(|k| *k == "block").expect("image block recorded: {kinds:?}");
    assert!(block_at < opener_at, "opener must follow the media blocks: {kinds:?}");
}

fn tool_envelope(json: &str) -> Vec<u32> {
    use superfluid_daemon::codec::{TextCodec, MOCK_TOOL_CLOSE, MOCK_TOOL_OPEN};
    let mut v = vec![MOCK_TOOL_OPEN];
    v.extend(superfluid_daemon::codec::MockCodec.encode(json));
    v.push(MOCK_TOOL_CLOSE);
    v
}

fn chat_with_tools(addr: std::net::SocketAddr, extra: &str) -> serde_json::Value {
    let body = format!(
        r#"{{"model":"mock-model","max_tokens":512,"temperature":0,
            "tools":[{{"type":"function","function":{{"name":"get_weather",
              "parameters":{{"type":"object","properties":{{"city":{{"type":"string"}}}}}}}}}}],
            "messages":[{{"role":"user","content":"weather in Paris?"}}]{extra}}}"#
    );
    let (status, _h, body) = http(addr, "POST", "/v1/chat/completions", &body);
    assert_eq!(status, 200, "body: {body}");
    serde_json::from_str(&body).unwrap()
}

#[test]
fn chat_returns_a_tool_call_in_openai_shape() {
    let addr = spawn_scripted_stack(tool_envelope(
        r#"{"name":"get_weather","arguments":{"city":"Paris","units":"c"}}"#,
    ));
    let v = chat_with_tools(addr, "");

    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["role"], "assistant");
    let calls = msg["tool_calls"].as_array().expect("tool_calls array");
    assert_eq!(calls.len(), 1, "{msg}");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["index"], 0);
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args = calls[0]["function"]["arguments"].as_str().expect("arguments is a string");
    let parsed: serde_json::Value = serde_json::from_str(args).expect("arguments parse as JSON");
    assert_eq!(parsed["city"], "Paris");
    assert_eq!(parsed["units"], "c");
    assert!(calls[0]["id"].as_str().is_some_and(|s| !s.is_empty()), "{msg}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn chat_returns_parallel_tool_calls_with_distinct_ids() {
    let mut script = tool_envelope(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#);
    script.extend(tool_envelope(r#"{"name":"get_weather","arguments":{"city":"Rome"}}"#));
    let addr = spawn_scripted_stack(script);
    let v = chat_with_tools(addr, "");

    let calls = v["choices"][0]["message"]["tool_calls"].as_array().expect("tool_calls");
    assert_eq!(calls.len(), 2, "{}", v["choices"][0]["message"]);
    assert_eq!(calls[0]["index"], 0);
    assert_eq!(calls[1]["index"], 1);
    let a: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    let b: serde_json::Value =
        serde_json::from_str(calls[1]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(a["city"], "Paris");
    assert_eq!(b["city"], "Rome");
    assert_ne!(calls[0]["id"], calls[1]["id"]);
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn chat_keeps_content_alongside_a_tool_call() {
    use superfluid_daemon::codec::TextCodec as _;
    let mut script = superfluid_daemon::codec::MockCodec.encode("Let me check.");
    script.extend(tool_envelope(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#));
    let addr = spawn_scripted_stack(script);
    let v = chat_with_tools(addr, "");

    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["content"].as_str().unwrap_or_default(), "Let me check.");
    assert_eq!(msg["tool_calls"].as_array().expect("tool_calls").len(), 1);
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn chat_survives_a_malformed_tool_call() {
    let addr = spawn_scripted_stack(tool_envelope("{not json at all"));
    let v = chat_with_tools(addr, "");

    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["role"], "assistant");
    let calls = msg["tool_calls"].as_array().map(|a| a.len()).unwrap_or(0);
    assert_eq!(calls, 0, "unparseable text must not become a tool call: {msg}");
    assert!(
        msg["content"].as_str().is_some_and(|c| c.contains("not json")),
        "the raw text should survive as content: {msg}"
    );
    assert_ne!(v["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn chat_streams_tool_call_deltas() {
    let addr = spawn_scripted_stack(tool_envelope(
        r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#,
    ));
    let body = r#"{"model":"mock-model","max_tokens":512,"temperature":0,"stream":true,
        "tools":[{"type":"function","function":{"name":"get_weather","parameters":{}}}],
        "messages":[{"role":"user","content":"weather in Paris?"}]}"#;
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 200, "{raw}");

    let events: Vec<serde_json::Value> = raw
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data is JSON"))
        .collect();
    assert!(!events.is_empty(), "no SSE events: {raw}");
    for e in &events {
        assert_eq!(e["object"], "chat.completion.chunk");
    }

    let entries: Vec<&serde_json::Value> = events
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
        .collect();
    assert!(!entries.is_empty(), "no tool_calls delta in stream: {raw}");
    assert!(entries.len() > 1, "arguments were not fragmented: {raw}");
    let opening = entries[0];
    assert_eq!(opening["index"], 0);
    assert_eq!(opening["type"], "function");
    assert!(
        opening["id"].as_str().is_some_and(|s| !s.is_empty()),
        "opening delta identifies the call: {raw}"
    );
    assert_eq!(opening["function"]["name"], "get_weather", "the name comes first: {raw}");
    for e in &entries[1..] {
        assert_eq!(e["index"], 0, "fragments repeat the call's index: {raw}");
        assert!(e["function"]["name"].is_null(), "only the opening delta names the call: {raw}");
        assert!(e["id"].is_null(), "only the opening delta carries the id: {raw}");
    }
    let args: String = entries
        .iter()
        .filter_map(|e| e["function"]["arguments"].as_str())
        .collect();
    let parsed: serde_json::Value = serde_json::from_str(&args).expect("arguments parse");
    assert_eq!(parsed["city"], "Paris");

    let fin = events
        .iter()
        .rev()
        .find_map(|e| e["choices"][0]["finish_reason"].as_str())
        .expect("no finish_reason in stream");
    assert_eq!(fin, "tool_calls");
    assert!(raw.contains("data: [DONE]"), "{raw}");
}

fn stream_events(raw: &str) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let events: Vec<serde_json::Value> = raw
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data is JSON"))
        .collect();
    let entries = events
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
        .cloned()
        .collect();
    (events, entries)
}

fn stream_finish(events: &[serde_json::Value]) -> String {
    events
        .iter()
        .rev()
        .find_map(|e| e["choices"][0]["finish_reason"].as_str())
        .expect("no finish_reason in stream")
        .to_string()
}

const WEATHER_STREAM: &str = r#"{"model":"mock-model","max_tokens":512,"temperature":0,"stream":true,
    "tools":[{"type":"function","function":{"name":"get_weather","parameters":{}}}],
    "messages":[{"role":"user","content":"weather?"}]}"#;

#[test]
fn a_streamed_call_that_fails_to_parse_is_not_left_runnable() {
    let mut script = tool_envelope(r#"{"name":"get_weather","arguments":{"city":"Paris"""Texas"""}}"#);
    script.extend(tool_envelope(r#"{"name":"get_weather","arguments":{"city":"Lyon"}}"#));
    let addr = spawn_scripted_stack(script);
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", WEATHER_STREAM);
    assert_eq!(status, 200, "{raw}");
    let (events, entries) = stream_events(&raw);

    let openings: Vec<&serde_json::Value> = entries.iter().filter(|e| e["function"]["name"].is_string()).collect();
    assert_eq!(openings.len(), 2, "both calls were announced: {raw}");
    assert_eq!(openings[0]["index"], 0, "{raw}");
    assert_eq!(openings[1]["index"], 1, "a later call must not be merged into the broken one: {raw}");
    let second: String = entries
        .iter()
        .filter(|e| e["index"] == 1)
        .filter_map(|e| e["function"]["arguments"].as_str())
        .collect();
    let parsed: serde_json::Value = serde_json::from_str(&second).expect("the good call's arguments parse");
    assert_eq!(parsed["city"], "Lyon");

    let content: String = events.iter().filter_map(|e| e["choices"][0]["delta"]["content"].as_str()).collect();
    assert!(content.contains("Texas"), "the unparsed body surfaces as content: {raw}");
    assert_eq!(
        stream_finish(&events),
        "length",
        "a reply holding an incomplete call must not tell the client to run it: {raw}"
    );
}

#[test]
fn an_unparseable_block_that_never_named_a_call_finishes_normally() {
    let addr = spawn_scripted_stack(tool_envelope("{not json"));
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", WEATHER_STREAM);
    assert_eq!(status, 200, "{raw}");
    let (events, entries) = stream_events(&raw);
    assert!(entries.is_empty(), "nothing was announced: {raw}");
    let content: String = events.iter().filter_map(|e| e["choices"][0]["delta"]["content"].as_str()).collect();
    assert!(content.contains("not json"), "{raw}");
    assert_eq!(stream_finish(&events), "stop", "{raw}");
}

struct HarmonyTok;

const HARMONY_MARKERS: [&str; 7] = ["<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>", "<|call|>", "<|constrain|>"];

const HARMONY_TEMPLATE: &str = "{%- for message in messages %}{%- if message.role == 'assistant' %}<|start|>assistant<|channel|>final<|message|>{{ message.content }}<|end|>{%- else %}<|start|>{{ message.role }}<|message|>{{ message.content }}<|end|>{%- endif %}{%- endfor %}{%- if add_generation_prompt %}<|start|>assistant{%- endif %}";

impl superfluid_engine::Tokenizer for HarmonyTok {
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            match HARMONY_MARKERS.iter().position(|m| rest.starts_with(m)) {
                Some(i) => {
                    out.push(300 + i as u32);
                    rest = &rest[HARMONY_MARKERS[i].len()..];
                }
                None => {
                    out.push(u32::from(rest.as_bytes()[0]));
                    rest = &rest[1..];
                }
            }
        }
        out
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        if token < 256 {
            vec![token as u8]
        } else {
            Vec::new()
        }
    }
    fn vocab_size(&self) -> u32 {
        400
    }
    fn special_tokens(&self) -> Vec<(String, u32)> {
        HARMONY_MARKERS.iter().enumerate().map(|(i, m)| (m.to_string(), 300 + i as u32)).collect()
    }
    fn bos_token(&self) -> Option<u32> {
        None
    }
    fn eos_token(&self) -> u32 {
        304
    }
    fn chat_template_jinja(&self) -> String {
        HARMONY_TEMPLATE.to_string()
    }
}

fn spawn_scripted_harmony_stack(script: &str) -> std::net::SocketAddr {
    use superfluid_engine::Tokenizer;
    let codec = superfluid_daemon::template_codec::TemplateCodec::from_tokenizer(Arc::new(HarmonyTok)).expect("a Harmony codec");
    let script = HarmonyTok.encode(script);
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-harmony-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || {
        let cfg = EngineConfig { scripted: script.clone(), ..Default::default() };
        (MockEngine::new(cfg), None)
    })
    .expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(codec), 8));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking(listener, daemon, "mock-model".into());
    });
    addr
}

const WRITE_TOOL_CHAT: &str = r#"{"model":"mock-model","max_tokens":512,"temperature":0,"stream":STREAM,TOOL_CHOICE
    "tools":[{"type":"function","function":{"name":"write","parameters":{"type":"object",
      "properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}],
    "messages":[{"role":"user","content":"Make a server."}]}"#;

fn write_tool_chat(stream: bool, tool_choice: &str) -> String {
    WRITE_TOOL_CHAT.replace("STREAM", &stream.to_string()).replace("TOOL_CHOICE", tool_choice)
}

const BROKEN_WRITE: &str = r#"<|channel|>analysis<|message|>Write it.<|end|><|start|>assistant<|channel|>commentary to=functions.write <|constrain|>json<|message|>{"path":"server.py","content":"main()\n","}<|call|>"#;
const BROKEN_WRITE_ARGUMENTS: &str = r#"{"path":"server.py","content":"main()\n","}"#;

#[test]
fn a_harmony_call_whose_arguments_break_goes_out_as_that_call_cut_short() {
    let addr = spawn_scripted_harmony_stack(BROKEN_WRITE);
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", &write_tool_chat(true, ""));
    assert_eq!(status, 200, "{raw}");
    let (events, entries) = stream_events(&raw);
    let openings: Vec<&serde_json::Value> = entries.iter().filter(|e| e["function"]["name"].is_string()).collect();
    assert_eq!(openings.len(), 1, "the call went out: {raw}");
    assert_eq!(openings[0]["function"]["name"], "write", "{raw}");
    assert!(openings[0]["id"].is_string(), "{raw}");
    let arguments: String = entries.iter().filter_map(|e| e["function"]["arguments"].as_str()).collect();
    assert_eq!(arguments, BROKEN_WRITE_ARGUMENTS, "the arguments as written: {raw}");
    let content: String = events.iter().filter_map(|e| e["choices"][0]["delta"]["content"].as_str()).collect();
    assert!(!content.contains("to=functions"), "the call is not the reply's text: {raw}");
    assert_eq!(stream_finish(&events), "length", "{raw}");

    let addr = spawn_scripted_harmony_stack(BROKEN_WRITE);
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", &write_tool_chat(false, ""));
    assert_eq!(status, 200, "{raw}");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let choice = &v["choices"][0];
    assert_eq!(choice["finish_reason"], "length", "{raw}");
    let calls = choice["message"]["tool_calls"].as_array().expect("the call is in tool_calls");
    assert_eq!(calls.len(), 1, "{raw}");
    assert_eq!(calls[0]["function"]["name"], "write", "{raw}");
    assert_eq!(calls[0]["function"]["arguments"], BROKEN_WRITE_ARGUMENTS, "{raw}");
    assert!(!choice["message"]["content"].as_str().unwrap_or("").contains("to=functions"), "{raw}");
}

#[test]
fn under_tool_choice_none_a_broken_harmony_call_stays_text() {
    let addr = spawn_scripted_harmony_stack(BROKEN_WRITE);
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", &write_tool_chat(true, r#""tool_choice":"none","#));
    assert_eq!(status, 200, "{raw}");
    let (events, entries) = stream_events(&raw);
    assert!(entries.is_empty(), "{raw}");
    let content: String = events.iter().filter_map(|e| e["choices"][0]["delta"]["content"].as_str()).collect();
    assert!(content.contains(BROKEN_WRITE_ARGUMENTS), "{raw}");
    assert_eq!(stream_finish(&events), "stop", "{raw}");
}

#[test]
fn a_harmony_call_that_parses_goes_out_whole() {
    let script = r#"<|channel|>commentary to=functions.write <|constrain|>json<|message|>{"path":"server.py","content":"main()\n"}<|call|>"#;
    let addr = spawn_scripted_harmony_stack(script);
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", &write_tool_chat(true, ""));
    assert_eq!(status, 200, "{raw}");
    let (events, entries) = stream_events(&raw);
    assert_eq!(entries.len(), 1, "{raw}");
    assert_eq!(entries[0]["function"]["name"], "write");
    let args: serde_json::Value = serde_json::from_str(entries[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, serde_json::json!({"path": "server.py", "content": "main()\n"}));
    assert_eq!(stream_finish(&events), "tool_calls", "{raw}");
}

#[test]
fn a_call_the_token_limit_cut_short_finishes_length() {
    use superfluid_daemon::codec::{TextCodec, MOCK_TOOL_OPEN};
    let cut = r#"{"name":"get_weather","arguments":{"city":"Pa"#;
    let mut script = vec![MOCK_TOOL_OPEN];
    script.extend(superfluid_daemon::codec::MockCodec.encode(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#));
    let max_tokens = 1 + superfluid_daemon::codec::MockCodec.encode(cut).len();
    for stream in [true, false] {
        let addr = spawn_scripted_stack(script.clone());
        let body = WEATHER_STREAM
            .replace(r#""max_tokens":512"#, &format!(r#""max_tokens":{max_tokens}"#))
            .replace(r#""stream":true"#, &format!(r#""stream":{stream}"#));
        let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", &body);
        assert_eq!(status, 200, "{raw}");
        let finish = if stream {
            let (events, entries) = stream_events(&raw);
            assert!(!entries.is_empty(), "the cut call still goes out, for the client to fail: {raw}");
            stream_finish(&events)
        } else {
            let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
            assert!(v["choices"][0]["message"]["tool_calls"].is_array(), "{raw}");
            v["choices"][0]["finish_reason"].as_str().unwrap().to_string()
        };
        assert_eq!(finish, "length", "a call the limit cut must not be run (stream: {stream}): {raw}");
    }
}

#[test]
fn anthropic_messages_returns_a_tool_use_block() {
    let addr = spawn_scripted_stack(tool_envelope(
        r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#,
    ));
    let body = r#"{"model":"mock-model","max_tokens":512,
        "tools":[{"name":"get_weather","description":"d","input_schema":{"type":"object"}}],
        "messages":[{"role":"user","content":"weather in Paris?"}]}"#;
    let (status, _h, raw) = http(addr, "POST", "/v1/messages", body);
    assert_eq!(status, 200, "{raw}");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

    let blocks = v["content"].as_array().expect("content blocks");
    let tu = blocks
        .iter()
        .find(|b| b["type"] == "tool_use")
        .unwrap_or_else(|| panic!("no tool_use block: {v}"));
    assert_eq!(tu["name"], "get_weather");
    assert_eq!(tu["input"]["city"], "Paris");
    assert!(tu["id"].as_str().is_some_and(|s| !s.is_empty()), "{tu}");
    assert_eq!(v["stop_reason"], "tool_use");
}

#[test]
fn anthropic_a_tool_use_the_token_limit_cut_short_stops_max_tokens() {
    use superfluid_daemon::codec::{TextCodec, MOCK_TOOL_OPEN};
    let mut script = vec![MOCK_TOOL_OPEN];
    script.extend(superfluid_daemon::codec::MockCodec.encode(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#));
    let max_tokens = 1 + superfluid_daemon::codec::MockCodec.encode(r#"{"name":"get_weather","arguments":{"city":"Pa"#).len();
    for stream in [false, true] {
        let addr = spawn_scripted_stack(script.clone());
        let body = format!(
            r#"{{"model":"mock-model","max_tokens":{max_tokens},"stream":{stream},
            "tools":[{{"name":"get_weather","description":"d","input_schema":{{"type":"object"}}}}],
            "messages":[{{"role":"user","content":"weather in Paris?"}}]}}"#
        );
        let (status, _h, raw) = http(addr, "POST", "/v1/messages", &body);
        assert_eq!(status, 200, "{raw}");
        let stop = if stream {
            raw.lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
                .find_map(|e| e["delta"]["stop_reason"].as_str().map(str::to_string))
                .expect("a message_delta with the stop reason")
        } else {
            let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
            v["stop_reason"].as_str().unwrap().to_string()
        };
        assert_eq!(stop, "max_tokens", "a tool_use the limit cut must not be run (stream: {stream}): {raw}");
    }
}

#[test]
fn tools_request_paths_render() {
    let addr = spawn_stack();
    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","max_tokens":4,"temperature":0,
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
            "messages":[
                {"role":"system","content":"sys"},
                {"role":"user","content":"go"},
                {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]},
                {"role":"tool","tool_call_id":"call_1","content":"result"},
                {"role":"user","content":"and?"}
            ]}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
}

#[test]
fn anthropic_messages_shape_and_streaming_grammar() {
    let addr = spawn_stack();
    let (status, head, body) = http(
        addr,
        "POST",
        "/v1/messages",
        r#"{"model":"mock-model","max_tokens":6,"temperature":0,
            "system":"be terse",
            "messages":[{"role":"user","content":"hello"}]}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    assert!(head.to_ascii_lowercase().contains("x-superfluid-warm:"));
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert!(v["content"].is_array());
    assert!(v["stop_reason"].is_string());
    assert_eq!(
        v["usage"]["output_tokens"].as_u64().unwrap(),
        6,
        "usage from stream==spans"
    );

    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/messages",
        r#"{"model":"mock-model","max_tokens":40,"stream":true,"temperature":0,
            "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
    );
    assert_eq!(status, 200);
    let names: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("event: "))
        .collect();
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(names.last(), Some(&"message_stop"));
    assert!(names.contains(&"content_block_start"));
    assert!(names.contains(&"content_block_stop"));
    let delta_pos = names.iter().position(|n| *n == "message_delta").unwrap();
    assert!(delta_pos == names.len() - 2, "message_delta precedes stop");
    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/messages",
        r#"{"model":"mock-model","max_tokens":4,"temperature":0,
            "tools":[{"name":"f","description":"d","input_schema":{"type":"object"}}],
            "messages":[
                {"role":"user","content":"go"},
                {"role":"assistant","content":[{"type":"text","text":"on it"},{"type":"tool_use","id":"toolu_1","name":"f","input":{"x":1}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"done"}]}
            ]}"#,
    );
    assert_eq!(status, 200, "body: {body}");
}

fn spawn_stack_with_daemon(
    codec: Box<dyn superfluid_daemon::TextCodec + Send + Sync>,
) -> (std::net::SocketAddr, Arc<Daemon>) {
    spawn_stack_with_options(codec, 50)
}

fn spawn_stack_with_options(
    codec: Box<dyn superfluid_daemon::TextCodec + Send + Sync>,
    pin_budget_pct: u8,
) -> (std::net::SocketAddr, Arc<Daemon>) {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-anthropic-cc-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let mut cfg = EngineConfig::default();
    cfg.spaces
        .retain(|s| s.kind == superfluid_abi::space_kind::PAGED_TOKEN_KV);
    let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).expect("spawn");
    let daemon = Arc::new(
        Daemon::with_options(
            store,
            host,
            codec,
            superfluid_daemon::DaemonOptions {
                media_dir: Some(dir.join("media")),
                pin_budget_pct,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let d = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = openai::serve_blocking(listener, d, "mock-model".into());
    });
    (addr, daemon)
}

fn cache_control_request(stream: bool) -> String {
    let system = "You are a careful assistant. ".repeat(8);
    let user = "Please summarize the following notes in one line. ".repeat(6);
    serde_json::json!({
        "model": "mock-model", "max_tokens": 4, "temperature": 0, "stream": stream,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
    })
    .to_string()
}

fn cache_usage(u: &serde_json::Value) -> (u64, u64, u64) {
    (
        u["input_tokens"].as_u64().expect("input_tokens"),
        u["cache_creation_input_tokens"]
            .as_u64()
            .expect("cache_creation_input_tokens"),
        u["cache_read_input_tokens"]
            .as_u64()
            .expect("cache_read_input_tokens"),
    )
}

fn cache_control_pins_the_breakpoint_prefix(
    codec: Box<dyn superfluid_daemon::TextCodec + Send + Sync>,
) {
    use std::sync::atomic::Ordering;
    let (addr, daemon) = spawn_stack_with_daemon(codec);
    let stats = daemon.sched_stats();
    let (status, _, body) = http(addr, "POST", "/v1/messages", &cache_control_request(false));
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let (input, created, read) = cache_usage(&v["usage"]);
    let prompt = input + created + read;
    assert_eq!(read, 0, "nothing cached before the first request");
    assert!(created > 0, "the breakpoint prefix was pinned: {v}");
    assert!(
        created < prompt / 2,
        "only the system turn is pinned, not the user turn: {v}"
    );
    assert_eq!(created % 16, 0, "pins are block-aligned");
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    assert!(stats.pinned_bytes.load(Ordering::Relaxed) > 0);

    let (status, _, body) = http(addr, "POST", "/v1/messages", &cache_control_request(false));
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let (input2, created2, read2) = cache_usage(&v["usage"]);
    assert_eq!(input2 + created2 + read2, prompt);
    assert!(read2 >= created, "the pinned prefix was read back: {v}");
    assert_eq!(created2, 0, "nothing new to create");
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        1,
        "one pin per prefix, refreshed"
    );

    let (status, _, body) = http(addr, "POST", "/v1/messages", &cache_control_request(true));
    assert_eq!(status, 200);
    let delta: serde_json::Value = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        .find(|v| v["type"] == "message_delta")
        .expect("message_delta");
    let (i3, c3, r3) = cache_usage(&delta["usage"]);
    assert_eq!(i3 + c3 + r3, prompt);
    assert!(r3 > 0);
}

#[test]
fn anthropic_cache_control_pins_and_reports_usage() {
    cache_control_pins_the_breakpoint_prefix(Box::new(MockChatCodec));
}

#[test]
fn anthropic_cache_control_on_a_template_codec() {
    cache_control_pins_the_breakpoint_prefix(Box::new(superfluid_daemon::codec::MockTemplateCodec));
}

#[test]
fn anthropic_tool_breakpoint_never_pins_the_first_user_turn() {
    let req = serde_json::json!({
        "model": "mock-model", "max_tokens": 2, "temperature": 0,
        "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"},
                   "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "Summarize these notes. ".repeat(12)}],
    })
    .to_string();
    let (addr, _d) = spawn_stack_with_daemon(Box::new(superfluid_daemon::codec::MockTemplateCodec));
    let (status, _, body) = http(addr, "POST", "/v1/messages", &req);
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        cache_usage(&v["usage"]).1,
        0,
        "no tools-only prefix exists: {v}"
    );
    let (addr, _d) = spawn_stack_with_daemon(Box::new(MockChatCodec));
    let (status, _, body) = http(addr, "POST", "/v1/messages", &req);
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let (input, created, read) = cache_usage(&v["usage"]);
    assert!(created > 0, "the tools turn is pinned: {v}");
    assert!(
        created < (input + created + read) / 2,
        "the user turn is not: {v}"
    );
}

#[test]
fn anthropic_usage_reports_pins_after_the_budget_settles() {
    let (addr, daemon) = spawn_stack_with_options(Box::new(MockChatCodec), 1);
    let req = serde_json::json!({
        "model": "mock-model", "max_tokens": 2, "temperature": 0,
        "system": [{"type": "text", "text": "Be terse. ".repeat(6), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Summarize these notes. ".repeat(12),
                      "cache_control": {"type": "ephemeral"}}]}],
    })
    .to_string();
    let (status, _, body) = http(addr, "POST", "/v1/messages", &req);
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let stats = daemon.sched_stats();
    assert!(
        stats
            .pins_yielded
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 2,
        "both yielded"
    );
    assert_eq!(
        stats.pins_held.load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(cache_usage(&v["usage"]).1, 0, "nothing stayed pinned: {v}");
}

#[test]
fn anthropic_usage_without_cache_control_reports_reads_only() {
    let (addr, _daemon) = spawn_stack_with_daemon(Box::new(MockChatCodec));
    let req = r#"{"model":"mock-model","max_tokens":4,"temperature":0,
        "system":"a system prompt long enough to fill a few blocks of cache for the reuse path",
        "messages":[{"role":"user","content":"hi"}]}"#;
    let mut reads = Vec::new();
    for _ in 0..2 {
        let (status, _, body) = http(addr, "POST", "/v1/messages", req);
        assert_eq!(status, 200, "body: {body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let (_, created, read) = cache_usage(&v["usage"]);
        assert_eq!(created, 0);
        reads.push(read);
    }
    assert_eq!(reads[0], 0);
    assert!(reads[1] > 0, "the second request is served warm");
}

#[test]
fn anthropic_cache_control_refusals() {
    let (addr, daemon) = spawn_stack_with_daemon(Box::new(MockChatCodec));
    let text = |cc: serde_json::Value| serde_json::json!({"type": "text", "text": "x", "cache_control": cc});
    let eph = serde_json::json!({"type": "ephemeral"});
    let cases: Vec<(serde_json::Value, &str)> = vec![
        (
            serde_json::json!([{"role": "user", "content": [
                text(eph.clone()), text(eph.clone()), text(eph.clone()), text(eph.clone()), text(eph.clone())
            ]}]),
            "A maximum of 4 blocks with cache_control may be provided. Found 5.",
        ),
        (
            serde_json::json!([{"role": "user", "content": [text(serde_json::json!({"type": "persistent"}))]}]),
            "Input should be 'ephemeral'",
        ),
        (
            serde_json::json!([{"role": "user", "content": [text(serde_json::json!({"type": "ephemeral", "ttl": "10m"}))]}]),
            "Input should be '5m' or '1h'",
        ),
        (
            serde_json::json!([{"role": "user", "content": [text(serde_json::json!({"type": "ephemeral", "scope": "global"}))]}]),
            "Extra inputs are not permitted",
        ),
        (
            serde_json::json!([{"role": "user", "content": [
                text(eph.clone()), text(serde_json::json!({"type": "ephemeral", "ttl": "1h"}))
            ]}]),
            "must not come after a ttl='5m'",
        ),
        (
            serde_json::json!([{"role": "user", "content": [{"type": "text", "text": "", "cache_control": eph}]}]),
            "empty text blocks",
        ),
        (
            serde_json::json!([
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "", "cache_control": {"type": "ephemeral"}}]},
                {"role": "user", "content": "r"}
            ]),
            "thinking blocks",
        ),
    ];
    let before = daemon.session_ids().len();
    for (messages, want) in cases {
        let body =
            serde_json::json!({"model": "mock-model", "max_tokens": 2, "messages": messages})
                .to_string();
        let (status, _, resp) = http(addr, "POST", "/v1/messages", &body);
        assert_eq!(status, 400, "{want}: {resp}");
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains(want), "expected {want:?} in {msg:?}");
    }
    assert_eq!(
        daemon.session_ids().len(),
        before,
        "refused before any session exists"
    );
    let tool_result = |inner_ttl: &str, outer_ttl: &str| {
        serde_json::json!({
            "model": "mock-model", "max_tokens": 2,
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
                ]},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": "t1",
                    "cache_control": {"type": "ephemeral", "ttl": outer_ttl},
                    "content": [{"type": "text", "text": "a",
                                 "cache_control": {"type": "ephemeral", "ttl": inner_ttl}}]
                }]}
            ]
        })
        .to_string()
    };
    let (status, _, resp) = http(addr, "POST", "/v1/messages", &tool_result("1h", "5m"));
    assert_eq!(status, 200, "{resp}");
    let (status, _, resp) = http(addr, "POST", "/v1/messages", &tool_result("5m", "1h"));
    assert_eq!(status, 400, "{resp}");
    assert!(resp.contains("must not come after"), "{resp}");
    let body = serde_json::json!({
        "model": "mock-model", "max_tokens": 2,
        "tools": [{"name": "f", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}, "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}
            ]}]}
        ]
    })
    .to_string();
    let (status, _, resp) = http(addr, "POST", "/v1/messages", &body);
    assert_eq!(status, 400, "{resp}");
    assert!(resp.contains("Found 5."), "{resp}");
}

#[test]
fn template_mode_serves_image_chat_via_whole_render() {
    use superfluid_daemon::EventBody;
    let (addr, daemon) = spawn_template_media_stack();
    let png_b64 = "aGVsbG8gaW1hZ2U=";
    let body = format!(
        r#"{{"model":"mock-model","max_tokens":4,"messages":[{{"role":"user","content":[{{"type":"text","text":"What is this?"}},{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{png_b64}"}}}}]}}]}}"#
    );
    let (status, _, resp) = http(addr, "POST", "/v1/chat/completions", &body);
    assert_eq!(status, 200, "{resp}");
    let store = daemon.store();
    let store = store.lock().unwrap();
    let mut saw = false;
    for id in store.session_ids() {
        let s = store.session(id).unwrap();
        for e in &s.events {
            if let EventBody::Block { kind, payload, .. } = &e.body {
                if *kind != superfluid_daemon::wal::block_kind::IMAGE {
                    continue;
                }
                saw = true;
                let p: serde_json::Value = serde_json::from_str(payload).unwrap();
                assert_eq!(p["n_tokens"], 4, "{p}");
                assert_eq!(p["image_token_id"], 999, "{p}");
                let off = p["offset"].as_u64().unwrap() as usize;
                assert_eq!(&s.tokens[off..off + 4], &[999, 999, 999, 999], "{p}");
                assert_ne!(s.tokens.get(off + 4), Some(&999), "run must be exactly n_tokens");
            }
        }
    }
    assert!(saw, "an IMAGE block was recorded on the whole-render path");
}

#[test]
fn template_mode_accepts_input_audio_parts() {
    use superfluid_daemon::EventBody;
    let (addr, daemon) = spawn_template_media_stack();
    let wav_b64 = "aGVsbG8gYXVkaW8=";
    let body = format!(
        r#"{{"model":"mock-model","max_tokens":4,"messages":[{{"role":"user","content":[{{"type":"text","text":"What do you hear?"}},{{"type":"input_audio","input_audio":{{"data":"{wav_b64}","format":"wav"}}}}]}}]}}"#
    );
    let (status, _, resp) = http(addr, "POST", "/v1/chat/completions", &body);
    assert_eq!(status, 200, "{resp}");
    let store = daemon.store();
    let store = store.lock().unwrap();
    let mut saw = false;
    for id in store.session_ids() {
        let s = store.session(id).unwrap();
        for e in &s.events {
            if let EventBody::Block { kind, payload, .. } = &e.body {
                if *kind != superfluid_daemon::wal::block_kind::IMAGE {
                    continue;
                }
                saw = true;
                let p: serde_json::Value = serde_json::from_str(payload).unwrap();
                let off = p["offset"].as_u64().unwrap() as usize;
                assert_eq!(&s.tokens[off..off + 4], &[999, 999, 999, 999], "{p}");
            }
        }
    }
    assert!(saw, "a media block was recorded for the audio part");
}

#[test]
fn input_audio_refuses_non_wav_formats() {
    let (addr, _daemon) = spawn_template_media_stack();
    let body = r#"{"model":"mock-model","max_tokens":4,"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"aGVsbG8=","format":"mp3"}}]}]}"#;
    let (status, _, resp) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 400, "{resp}");
    assert!(resp.contains("wav"), "the refusal names the accepted format: {resp}");
}

#[test]
fn template_mode_serves_anthropic_messages_with_images() {
    use superfluid_daemon::EventBody;
    let (addr, daemon) = spawn_template_media_stack();
    let png_b64 = "aGVsbG8gaW1hZ2U=";
    let body = format!(
        r#"{{"model":"mock-model","max_tokens":4,"messages":[{{"role":"user","content":[{{"type":"text","text":"Describe"}},{{"type":"image","source":{{"type":"base64","media_type":"image/png","data":"{png_b64}"}}}}]}}]}}"#
    );
    let (status, _, resp) = http(addr, "POST", "/v1/messages", &body);
    assert_eq!(status, 200, "{resp}");
    let store = daemon.store();
    let store = store.lock().unwrap();
    let mut saw = false;
    for id in store.session_ids() {
        let s = store.session(id).unwrap();
        for e in &s.events {
            if let EventBody::Block { kind, payload, .. } = &e.body {
                if *kind != superfluid_daemon::wal::block_kind::IMAGE {
                    continue;
                }
                saw = true;
                let p: serde_json::Value = serde_json::from_str(payload).unwrap();
                let off = p["offset"].as_u64().unwrap() as usize;
                assert_eq!(&s.tokens[off..off + 4], &[999, 999, 999, 999], "{p}");
            }
        }
    }
    assert!(saw, "an IMAGE block was recorded through /v1/messages");
    drop(store);
    let body = r#"{"model":"mock-model","max_tokens":4,"messages":[{"role":"user","content":"hello"}]}"#;
    let (status, _, resp) = http(addr, "POST", "/v1/messages", body);
    assert_eq!(status, 200, "{resp}");
}

#[test]
fn image_parts_enter_through_both_adapters() {
    use superfluid_daemon::EventBody;
    let (addr, daemon) = spawn_media_stack();
    let png_b64 = "aGVsbG8gaW1hZ2U=";
    let body = format!(
        r#"{{"model":"mock-model","max_tokens":4,"messages":[{{"role":"user","content":[{{"type":"text","text":"What is this?"}},{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{png_b64}"}}}}]}}]}}"#
    );
    let (status, _, resp) = http(addr, "POST", "/v1/chat/completions", &body);
    assert_eq!(status, 200, "{resp}");
    let store = daemon.store();
    let store = store.lock().unwrap();
    let mut saw = false;
    for id in store.session_ids() {
        let s = store.session(id).unwrap();
        for e in &s.events {
            if let EventBody::Block { kind, span, .. } = &e.body {
                if *kind == superfluid_daemon::wal::block_kind::IMAGE {
                    assert_eq!(span.iter().filter(|&&t| t == 999).count(), 4);
                    saw = true;
                }
            }
        }
    }
    assert!(saw, "an image turn was recorded");
    drop(store);
    let body = format!(
        r#"{{"model":"mock-model","max_tokens":4,"messages":[{{"role":"user","content":[{{"type":"image","source":{{"type":"base64","media_type":"image/png","data":"{png_b64}"}}}},{{"type":"text","text":"Describe."}}]}}]}}"#
    );
    let (status, _, resp) = http(addr, "POST", "/v1/messages", &body);
    assert_eq!(status, 200, "{resp}");
    let body = r#"{"model":"mock-model","max_tokens":4,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}]}"#;
    let (status, _, resp) = http(addr, "POST", "/v1/chat/completions", body);
    assert_ne!(status, 200);
    assert!(resp.contains("not fetched"), "{resp}");
    let body = r#"{"model":"mock-model","max_tokens":4,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/x.png"}}]}]}"#;
    let (status, _, _) = http(addr, "POST", "/v1/messages", body);
    assert_ne!(status, 200);
}

#[test]
fn health_route_ok() {
    let addr = spawn_stack();
    let (status, _, body) = http(addr, "GET", "/health", "");
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "ok");
}

#[test]
fn tokenize_route_returns_tokens_and_count() {
    let addr = spawn_stack();
    let (status, _, body) = http(addr, "POST", "/v1/tokenize", r#"{"text":"hello world"}"#);
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let toks = v["tokens"].as_array().expect("tokens array");
    assert!(!toks.is_empty(), "expected non-empty token ids");
    assert_eq!(v["count"].as_u64().unwrap(), toks.len() as u64);
    let (status2, _, body2) = http(addr, "POST", "/v1/tokenize", r#"{"content":"hello world"}"#);
    assert_eq!(status2, 200);
    let v2: serde_json::Value = serde_json::from_str(&body2).unwrap();
    assert_eq!(v2["tokens"], v["tokens"]);
}

#[test]
fn chat_accepts_sampling_penalties() {
    let addr = spawn_stack();
    let (status, _, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":6,"temperature":0.7,"presence_penalty":0.5,"frequency_penalty":0.3,"repeat_penalty":1.1}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["usage"]["completion_tokens"], 6);
}

#[test]
fn completions_route_shape_and_echo() {
    let addr = spawn_stack();
    let (status, _, body) = http(
        addr, "POST", "/v1/completions",
        r#"{"model":"mock-model","prompt":"once upon a time","max_tokens":5,"temperature":0}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert!(v["choices"][0]["text"].is_string());
    assert_eq!(v["usage"]["completion_tokens"], 5);
    assert!(v["choices"][0]["finish_reason"].is_string());
    let (status2, _, body2) = http(
        addr, "POST", "/v1/completions",
        r#"{"model":"mock-model","prompt":"PROMPTX","max_tokens":3,"temperature":0,"echo":true}"#,
    );
    assert_eq!(status2, 200, "body: {body2}");
    let v2: serde_json::Value = serde_json::from_str(&body2).unwrap();
    assert!(v2["choices"][0]["text"].as_str().unwrap().starts_with("PROMPTX"));
}

#[test]
fn completions_streaming_frames() {
    let addr = spawn_stack();
    let (status, head, body) = http(
        addr, "POST", "/v1/completions",
        r#"{"model":"mock-model","prompt":"hello","max_tokens":4,"stream":true}"#,
    );
    assert_eq!(status, 200, "body: {body}");
    assert!(head.to_ascii_lowercase().contains("text/event-stream"));
    assert!(body.contains("text_completion"), "no completion chunks: {body}");
    assert!(body.contains("[DONE]"), "stream not terminated: {body}");
}

#[test]
fn embeddings_route_float_and_base64_and_batch() {
    let addr = spawn_stack();
    let (status, _, body) = http(addr, "POST", "/v1/embeddings",
        r#"{"model":"mock-model","input":"hello world"}"#);
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    let emb = v["data"][0]["embedding"].as_array().expect("float embedding array");
    assert_eq!(emb.len(), 8, "mock embedding dim");
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    let (s2, _, b2) = http(addr, "POST", "/v1/embeddings",
        r#"{"model":"mock-model","input":["a","b","c"]}"#);
    assert_eq!(s2, 200, "body: {b2}");
    let v2: serde_json::Value = serde_json::from_str(&b2).unwrap();
    assert_eq!(v2["data"].as_array().unwrap().len(), 3);
    assert_eq!(v2["data"][2]["index"], 2);
    let (s3, _, b3) = http(addr, "POST", "/v1/embeddings",
        r#"{"model":"mock-model","input":"x","encoding_format":"base64"}"#);
    assert_eq!(s3, 200, "body: {b3}");
    let v3: serde_json::Value = serde_json::from_str(&b3).unwrap();
    assert!(v3["data"][0]["embedding"].is_string(), "base64 embedding string");
}

#[test]
fn chat_n_produces_n_choices() {
    let addr = spawn_stack();
    let (status, _, body) = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0.7,"n":3}"#);
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let choices = v["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 3, "n=3 -> 3 choices");
    assert_eq!(choices[0]["index"], 0);
    assert_eq!(choices[2]["index"], 2);
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() >= 3);
}

#[test]
fn chat_stop_truncates_content() {
    let addr = spawn_stack();
    let base = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":12,"temperature":0}"#);
    let bv: serde_json::Value = serde_json::from_str(&base.2).unwrap();
    let full = bv["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    if full.len() < 4 {
        return;
    }
    let stop = &full[2..4];
    let body = format!(
        r#"{{"model":"mock-model","messages":[{{"role":"user","content":"hi"}}],"max_tokens":12,"temperature":0,"stop":"{}"}}"#,
        stop.replace('\\', "\\\\").replace('"', "\\\"")
    );
    let (status, _, resp) = http(addr, "POST", "/v1/chat/completions", &body);
    assert_eq!(status, 200, "resp: {resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let out = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!out.contains(stop), "stop string must be excluded: {out:?}");
    assert!(out.len() < full.len(), "stopped output must be shorter");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[test]
fn chat_accepts_response_format() {
    let addr = spawn_stack();
    for rf in [
        r#"{"type":"json_object"}"#,
        r#"{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"x":{"type":"number"}}}}}"#,
    ] {
        let body = format!(
            r#"{{"model":"mock-model","messages":[{{"role":"user","content":"hi"}}],"max_tokens":6,"temperature":0,"response_format":{rf}}}"#
        );
        let (status, _, resp) = http(addr, "POST", "/v1/chat/completions", &body);
        assert_eq!(status, 200, "resp: {resp}");
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion");
    }
}

#[test]
fn capability_descriptor_rides_props_and_models() {
    let addr = spawn_stack();
    let (s1, _, b1) = http(addr, "GET", "/props", "");
    assert_eq!(s1, 200, "{b1}");
    let v: serde_json::Value = serde_json::from_str(&b1).unwrap();
    assert!(v.get("kv_cache").is_none(), "{v}");
    let caps = &v["capabilities"];
    assert_eq!(caps["descriptor_version"], 1, "{caps}");
    assert_eq!(caps["workload"]["causal_generation"], true, "{caps}");
    assert!(caps["state"]["recurrent_snapshot"].is_string(), "{caps}");

    let (s2, _, b2) = http(addr, "GET", "/v1/models", "");
    assert_eq!(s2, 200, "{b2}");
    let v: serde_json::Value = serde_json::from_str(&b2).unwrap();
    let entry = &v["data"][0];
    assert_eq!(entry["loaded"], true, "{entry}");
    assert_eq!(entry["capabilities"]["descriptor_version"], 1, "{entry}");

    let (s3, _, b3) = http(addr, "GET", "/v1/models/mock-model", "");
    assert_eq!(s3, 200, "{b3}");
    let v: serde_json::Value = serde_json::from_str(&b3).unwrap();
    assert_eq!(v["capabilities"]["descriptor_version"], 1, "{v}");
}

#[test]
fn input_modalities_agree_with_the_capability_descriptor() {
    let addr = spawn_stack();
    let (s, _, b) = http(addr, "GET", "/v1/models", "");
    assert_eq!(s, 200, "{b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    let entry = &v["data"][0];

    let mods = &entry["capabilities"]["modalities"];
    assert_eq!(mods["image_encode"], true, "mock stack serves image chat: {entry}");
    assert_eq!(mods["gemma_audio_encode"], true, "mock stack serves audio parts: {entry}");

    let advertised = &entry["architecture"]["input_modalities"];
    assert_eq!(
        advertised,
        &serde_json::json!(["text", "image", "audio"]),
        "input_modalities must track the descriptor, not a constant: {entry}"
    );

    let (s, _, b) = http(addr, "GET", "/v1/models/mock-model", "");
    assert_eq!(s, 200, "{b}");
    let one: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(&one["architecture"]["input_modalities"], advertised, "{one}");
}

#[test]
fn props_and_slots_routes() {
    let addr = spawn_stack();
    let (s1, _, b1) = http(addr, "GET", "/props", "");
    assert_eq!(s1, 200, "{b1}");
    let v1: serde_json::Value = serde_json::from_str(&b1).unwrap();
    assert_eq!(v1["model"], "mock-model");
    assert!(v1["default_generation_settings"]["n_predict"].is_number());
    let ctx = v1["max_context"].as_u64().expect("max_context is a number");
    assert!(ctx >= 512, "{v1}");
    assert_eq!(v1["default_generation_settings"]["n_ctx"], ctx, "{v1}");
    assert_eq!(v1["server"], "superfluid", "{v1}");
    assert!(v1["max_request_bytes"].as_u64().unwrap_or(0) >= 1_000_000, "{v1}");
    let (s2, _, b2) = http(addr, "GET", "/slots", "");
    assert_eq!(s2, 200, "{b2}");
    let v2: serde_json::Value = serde_json::from_str(&b2).unwrap();
    assert!(v2.is_array());
    assert!(v2[0]["kv_pool_total"].is_number());
}

#[test]
fn lora_load_unload_list_routes() {
    let addr = spawn_stack();
    let (s1, _, b1) = http(addr, "POST", "/v1/lora/load", r#"{"path":"/tmp/adapter.base"}"#);
    assert_eq!(s1, 200, "{b1}");
    assert_eq!(serde_json::from_str::<serde_json::Value>(&b1).unwrap()["loaded"], true);
    let (s2, _, _) = http(addr, "POST", "/v1/lora/load", r#"{}"#);
    assert_eq!(s2, 400);
    let (s3, _, b3) = http(addr, "GET", "/v1/lora", "");
    assert_eq!(s3, 200, "{b3}");
    let v3: serde_json::Value = serde_json::from_str(&b3).unwrap();
    assert_eq!(v3["object"], "list");
    let row = &v3["data"][0];
    assert!(row["model"].is_string());
    assert!(row["adapter"].is_string());
    assert!(row["active"].is_boolean());
    let (s4, _, b4) = http(addr, "POST", "/v1/lora/unload", "");
    assert_eq!(s4, 200, "{b4}");
    assert_eq!(serde_json::from_str::<serde_json::Value>(&b4).unwrap()["unloaded"], true);
}

#[test]
fn rerank_route_scores_and_sorts() {
    let addr = spawn_stack();
    let (status, _, body) = http(addr, "POST", "/v1/rerank",
        r#"{"model":"mock-model","query":"animals","documents":["a cat and a dog","stock market news","a lion in the wild"],"return_documents":true}"#);
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);
    let s0 = results[0]["relevance_score"].as_f64().unwrap();
    let s1 = results[1]["relevance_score"].as_f64().unwrap();
    assert!(s0 >= s1, "results must be sorted desc");
    assert!(results[0]["index"].is_number());
    assert!(results[0]["document"]["text"].is_string());
    let (_, _, b2) = http(addr, "POST", "/v1/rerank",
        r#"{"model":"mock-model","query":"x","documents":["a","b","c"],"top_n":1}"#);
    let v2: serde_json::Value = serde_json::from_str(&b2).unwrap();
    assert_eq!(v2["results"].as_array().unwrap().len(), 1);
}

fn http_ct(addr: std::net::SocketAddr, method: &str, path: &str, ct: &str, body: &[u8]) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(s, "{method} {path} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: {ct}\r\ncontent-length: {}\r\n\r\n", body.len()).unwrap();
    s.write_all(body).unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let raw = String::from_utf8_lossy(&raw).into_owned();
    let (head, payload) = raw.split_once("\r\n\r\n").expect("split");
    let status: u16 = head.lines().next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") { dechunk(payload) } else { payload.to_string() };
    (status, body)
}

#[test]
fn files_upload_list_get_content_delete() {
    let addr = spawn_stack();
    let boundary = "BOUNDARYXYZ";
    let content = "line one\nline two\n";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\nContent-Type: application/octet-stream\r\n\r\n{c}\r\n--{b}--\r\n",
        b = boundary, c = content
    );
    let ct = format!("multipart/form-data; boundary={boundary}");
    let (status, resp) = http_ct(addr, "POST", "/v1/files", &ct, body.as_bytes());
    assert_eq!(status, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["object"], "file");
    assert_eq!(v["filename"], "in.jsonl");
    assert_eq!(v["purpose"], "batch");
    assert_eq!(v["bytes"].as_u64().unwrap(), content.len() as u64);
    let id = v["id"].as_str().unwrap().to_string();
    let (_, lresp) = http_ct(addr, "GET", "/v1/files", "application/json", b"");
    let lv: serde_json::Value = serde_json::from_str(&lresp).unwrap();
    assert!(lv["data"].as_array().unwrap().iter().any(|f| f["id"] == id));
    let (gs, gr) = http_ct(addr, "GET", &format!("/v1/files/{id}"), "application/json", b"");
    assert_eq!(gs, 200, "{gr}");
    let (cs, cr) = http_ct(addr, "GET", &format!("/v1/files/{id}/content"), "application/json", b"");
    assert_eq!(cs, 200);
    assert_eq!(cr, content);
    let (ds, dr) = http_ct(addr, "DELETE", &format!("/v1/files/{id}"), "application/json", b"");
    assert_eq!(ds, 200, "{dr}");
    assert_eq!(serde_json::from_str::<serde_json::Value>(&dr).unwrap()["deleted"], true);
    let (gs2, _) = http_ct(addr, "GET", &format!("/v1/files/{id}"), "application/json", b"");
    assert_eq!(gs2, 404);
}

#[test]
fn batches_end_to_end() {
    let addr = spawn_stack();
    let l1 = r#"{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4}}"#;
    let l2 = r#"{"custom_id":"b","method":"POST","url":"/v1/chat/completions","body":{"model":"mock-model","messages":[{"role":"user","content":"yo"}],"max_tokens":4}}"#;
    let content = format!("{l1}\n{l2}\n");
    let boundary = "BND";
    let up = format!("--{b}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\n\r\n{c}\r\n--{b}--\r\n", b=boundary, c=content);
    let (us, ur) = http_ct(addr, "POST", "/v1/files", &format!("multipart/form-data; boundary={boundary}"), up.as_bytes());
    assert_eq!(us, 200, "{ur}");
    let input_id = serde_json::from_str::<serde_json::Value>(&ur).unwrap()["id"].as_str().unwrap().to_string();

    let (cs, cr) = http_ct(addr, "POST", "/v1/batches", "application/json",
        format!(r#"{{"input_file_id":"{input_id}","endpoint":"/v1/chat/completions"}}"#).as_bytes());
    assert_eq!(cs, 200, "{cr}");
    let cv: serde_json::Value = serde_json::from_str(&cr).unwrap();
    assert_eq!(cv["object"], "batch");
    assert_eq!(cv["request_counts"]["total"], 2);
    let bid = cv["id"].as_str().unwrap().to_string();

    let mut done = None;
    for _ in 0..200 {
        let (_, gr) = http_ct(addr, "GET", &format!("/v1/batches/{bid}"), "application/json", b"");
        let gv: serde_json::Value = serde_json::from_str(&gr).unwrap();
        if gv["status"] == "completed" { done = Some(gv); break; }
        if gv["status"] == "failed" { panic!("batch failed: {gr}"); }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    let bv = done.expect("batch did not complete");
    assert_eq!(bv["request_counts"]["completed"], 2);
    let out_id = bv["output_file_id"].as_str().expect("output file id");

    let (os, orr) = http_ct(addr, "GET", &format!("/v1/files/{out_id}/content"), "application/json", b"");
    assert_eq!(os, 200);
    let lines: Vec<&str> = orr.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 2, "output: {orr}");
    let r0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(r0["custom_id"], "a");
    assert_eq!(r0["response"]["status_code"], 200);
    assert_eq!(r0["response"]["body"]["object"], "chat.completion");
}

fn http_ct_auth(addr: std::net::SocketAddr, method: &str, path: &str, ct: &str, body: &[u8], key: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\nauthorization: Bearer {key}\r\ncontent-type: {ct}\r\ncontent-length: {}\r\n\r\n",
        body.len()
    )
    .unwrap();
    s.write_all(body).unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let raw = String::from_utf8_lossy(&raw).into_owned();
    let (head, payload) = raw.split_once("\r\n\r\n").expect("split");
    let status: u16 = head.lines().next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") { dechunk(payload) } else { payload.to_string() };
    (status, body)
}

#[test]
fn batch_lines_run_under_the_creating_keys_class() {
    use superfluid_daemon::qos;
    let (addr, daemon, _t) = spawn_key_stack(KEYS, None, 0);
    let l1 = r#"{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4}}"#;
    let l2 = r#"{"custom_id":"b","method":"POST","url":"/v1/chat/completions","body":{"model":"mock-model","messages":[{"role":"user","content":"yo"}],"max_tokens":4}}"#;
    let up = format!("--BND\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--BND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\n\r\n{l1}\n{l2}\n\r\n--BND--\r\n");
    let (us, ur) = http_ct_auth(addr, "POST", "/v1/files", "multipart/form-data; boundary=BND", up.as_bytes(), "sk-batch");
    assert_eq!(us, 200, "{ur}");
    let input_id = serde_json::from_str::<serde_json::Value>(&ur).unwrap()["id"].as_str().unwrap().to_string();
    let before = daemon.sessions(true).len();
    let (cs, cr) = http_ct_auth(addr, "POST", "/v1/batches", "application/json",
        format!(r#"{{"input_file_id":"{input_id}","endpoint":"/v1/chat/completions"}}"#).as_bytes(), "sk-batch");
    assert_eq!(cs, 200, "{cr}");
    let bid = serde_json::from_str::<serde_json::Value>(&cr).unwrap()["id"].as_str().unwrap().to_string();
    for _ in 0..400 {
        let (_, gr) = http_ct_auth(addr, "GET", &format!("/v1/batches/{bid}"), "application/json", b"", "sk-batch");
        let gv: serde_json::Value = serde_json::from_str(&gr).unwrap();
        if gv["status"] == "completed" { break; }
        assert_ne!(gv["status"], "failed", "{gr}");
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    let sessions = daemon.sessions(true);
    assert_eq!(sessions.len(), before + 2, "one session per line");
    let mut ids: Vec<u64> = sessions.iter().map(|s| s.id).collect();
    ids.sort();
    for id in &ids[before..] {
        assert_eq!(daemon.inspect(*id).unwrap().qos_class, qos::BACKGROUND_AGENT, "batch session {id}");
    }
}

#[test]
fn a_capped_key_can_manage_its_running_batch() {
    let (addr, _daemon, table) =
        spawn_key_stack(r#"{"keys":[{"name":"one","key":"sk-one","max_concurrent":1}]}"#, None, 0);
    let key = table.keys().next().unwrap().clone();
    let line = r#"{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":3500,"ignore_eos":true}}"#;
    let up = format!("--BND\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--BND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\n\r\n{line}\n{line}\n{line}\n\r\n--BND--\r\n");
    let (us, ur) = http_ct_auth(addr, "POST", "/v1/files", "multipart/form-data; boundary=BND", up.as_bytes(), "sk-one");
    assert_eq!(us, 200, "{ur}");
    let input_id = serde_json::from_str::<serde_json::Value>(&ur).unwrap()["id"].as_str().unwrap().to_string();
    let (cs, cr) = http_ct_auth(addr, "POST", "/v1/batches", "application/json",
        format!(r#"{{"input_file_id":"{input_id}","endpoint":"/v1/chat/completions"}}"#).as_bytes(), "sk-one");
    assert_eq!(cs, 200, "{cr}");
    let bid = serde_json::from_str::<serde_json::Value>(&cr).unwrap()["id"].as_str().unwrap().to_string();
    assert_eq!(key.in_flight(), 1, "the running batch holds the key's slot");
    let (st, body) = post_hdrs(addr, "/v1/chat/completions", QOS_CHAT, &[("authorization", "Bearer sk-one")]);
    assert_eq!(st, 429, "new work is still capped: {body}");
    let (st, body) = http_ct_auth(addr, "GET", &format!("/v1/batches/{bid}"), "application/json", b"", "sk-one");
    assert_eq!(st, 200, "status is readable: {body}");
    let (st, body) = http_ct_auth(addr, "GET", "/v1/files", "application/json", b"", "sk-one");
    assert_eq!(st, 200, "file management takes no slot: {body}");
    let (st, body) = http_ct_auth(addr, "GET", "/props", "application/json", b"", "sk-one");
    assert_eq!(st, 429, "an engine-backed GET is capped like any work: {body}");
    let (st, body) = http_ct_auth(addr, "POST", &format!("/v1/batches/{bid}/cancel"), "application/json", b"", "sk-one");
    assert_eq!(st, 200, "the owner can cancel: {body}");
}

#[test]
fn serve_config_refuses_a_key_colliding_with_the_global_key() {
    let dir = std::env::temp_dir().join(format!("superfluid-openai-collide-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", daemon));
    let table = superfluid_daemon::keypolicy::KeyTable::parse(r#"{"keys":[{"name":"a","key":"sk-same"}]}"#, |_| None).unwrap();
    let cfg = openai::ServeConfig {
        api_key: Some("sk-same".into()),
        keys: Some(Arc::new(table)),
        ..Default::default()
    };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let err = openai::serve_blocking_config(listener, registry, cfg).unwrap_err();
    assert!(err.to_string().contains("--api-key"), "{err}");
}

#[test]
fn files_and_batches_are_scoped_to_their_key() {
    let (addr, _daemon, _t) = spawn_key_stack(r#"{"keys":[{"name":"a","key":"sk-a"},{"name":"b","key":"sk-b"}]}"#, Some("sk-global"), 0);
    let line = r#"{"custom_id":"x","method":"POST","url":"/v1/chat/completions","body":{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4}}"#;
    let up = format!("--BND\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--BND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\n\r\n{line}\n\r\n--BND--\r\n");
    let (us, ur) = http_ct_auth(addr, "POST", "/v1/files", "multipart/form-data; boundary=BND", up.as_bytes(), "sk-a");
    assert_eq!(us, 200, "{ur}");
    assert!(!ur.contains("owner"), "internal owner never returned: {ur}");
    let fid = serde_json::from_str::<serde_json::Value>(&ur).unwrap()["id"].as_str().unwrap().to_string();
    let (cs, cr) = http_ct_auth(addr, "POST", "/v1/batches", "application/json",
        format!(r#"{{"input_file_id":"{fid}","endpoint":"/v1/chat/completions"}}"#).as_bytes(), "sk-a");
    assert_eq!(cs, 200, "{cr}");
    assert!(!cr.contains("owner"), "{cr}");
    let bid = serde_json::from_str::<serde_json::Value>(&cr).unwrap()["id"].as_str().unwrap().to_string();

    let get = |m: &str, path: &str, key: &str| http_ct_auth(addr, m, path, "application/json", b"", key);
    let (_, l) = get("GET", "/v1/files", "sk-b");
    assert!(!l.contains(&fid), "B lists A's file: {l}");
    assert_eq!(get("GET", &format!("/v1/files/{fid}"), "sk-b").0, 404);
    assert_eq!(get("GET", &format!("/v1/files/{fid}/content"), "sk-b").0, 404);
    let (_, l) = get("GET", "/v1/batches", "sk-b");
    assert!(!l.contains(&bid), "B lists A's batch: {l}");
    assert_eq!(get("GET", &format!("/v1/batches/{bid}"), "sk-b").0, 404);
    assert_eq!(get("POST", &format!("/v1/batches/{bid}/cancel"), "sk-b").0, 404);
    let (st, body) = http_ct_auth(addr, "POST", "/v1/batches", "application/json",
        format!(r#"{{"input_file_id":"{fid}","endpoint":"/v1/chat/completions"}}"#).as_bytes(), "sk-b");
    assert_eq!(st, 400, "B cannot batch over A's file: {body}");
    let (st, body) = get("DELETE", &format!("/v1/files/{fid}"), "sk-b");
    assert_eq!(st, 404, "{body}");
    assert_eq!(get("GET", &format!("/v1/files/{fid}/content"), "sk-a").0, 200);
    assert_eq!(get("GET", &format!("/v1/batches/{bid}"), "sk-a").0, 200);
    let (_, l) = get("GET", "/v1/files", "sk-global");
    assert!(l.contains(&fid) && !l.contains("owner"), "the global key sees every file: {l}");
    assert_eq!(get("GET", &format!("/v1/batches/{bid}"), "sk-global").0, 200);
}

#[test]
fn tenant_keys_cannot_administer_the_server() {
    let (addr, _d, _t) = spawn_key_stack(
        r#"{"keys":[{"name":"t","key":"sk-t"},{"name":"ops","key":"sk-ops","admin":true}]}"#,
        Some("sk-global"),
        0,
    );
    let unload = r#"{"model":"nope"}"#;
    let post = |key: &str| post_hdrs(addr, "/v1/models/unload", unload, &[("authorization", &format!("Bearer {key}"))]);
    let (st, body) = post("sk-t");
    assert_eq!(st, 403, "{body}");
    assert!(body.contains("admin_required"), "{body}");
    for key in ["sk-ops", "sk-global"] {
        let (st, body) = post(key);
        assert_ne!(st, 403, "{key} may administer: {body}");
    }
    let (st, body) = http_ct_auth(addr, "DELETE", "/v1/models/mock-model", "application/json", b"", "sk-t");
    assert_eq!(st, 403, "{body}");
    let (st, _) = http_ct_auth(addr, "GET", "/v1/batches/", "application/json", b"", "sk-t");
    assert!((400..500).contains(&st), "a bare /v1/batches/ is a 4xx, got {st}");
    let (_, m) = http_hdr(addr, "GET", "/metrics", Some(("authorization", "Bearer sk-global")));
    assert!(m.contains("superfluid_key_rejected_total{key=\"t\",reason=\"admin\"} 2"), "{m}");
    assert!(m.contains("superfluid_key_rejected_total{key=\"t\",reason=\"qos\"} 0"), "{m}");
    assert!(m.contains("superfluid_key_requests_total{key=\"t\"} 3"), "{m}");
}

#[test]
fn an_exhausted_address_still_lets_an_unmetered_owner_cancel() {
    let (addr, _d, table) = spawn_key_stack(r#"{"keys":[{"name":"t","key":"sk-t"}]}"#, Some("sk-global"), 3);
    let line = r#"{"custom_id":"x","method":"POST","url":"/v1/chat/completions","body":{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":3000,"ignore_eos":true}}"#;
    let lines: String = (0..40).map(|_| format!("{line}\n")).collect();
    let up = format!("--BND\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--BND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\n\r\n{lines}\r\n--BND--\r\n");
    let (us, ur) = http_ct_auth(addr, "POST", "/v1/files", "multipart/form-data; boundary=BND", up.as_bytes(), "sk-t");
    assert_eq!(us, 200, "{ur}");
    let fid = serde_json::from_str::<serde_json::Value>(&ur).unwrap()["id"].as_str().unwrap().to_string();
    let (cs, cr) = http_ct_auth(addr, "POST", "/v1/batches", "application/json",
        format!(r#"{{"input_file_id":"{fid}","endpoint":"/v1/chat/completions"}}"#).as_bytes(), "sk-t");
    assert_eq!(cs, 200, "{cr}");
    let bid = serde_json::from_str::<serde_json::Value>(&cr).unwrap()["id"].as_str().unwrap().to_string();
    assert_eq!(http_ct_auth(addr, "GET", "/v1/batches", "application/json", b"", "sk-t").0, 200);
    assert_eq!(http_ct_auth(addr, "GET", "/v1/batches", "application/json", b"", "sk-t").0, 429);
    let before = table.metrics_text();
    let unload = r#"{"model":"nope"}"#;
    let (st, body) = post_hdrs(addr, "/v1/models/unload", unload, &[("authorization", "Bearer sk-t")]);
    assert_eq!(st, 429, "{body}");
    assert!(body.contains("Rate limit exceeded") && !body.contains("/min"), "the per-IP refusal: {body}");
    let count = |m: &str| {
        m.lines()
            .find(|l| l.starts_with("superfluid_key_requests_total{key=\"t\"}"))
            .and_then(|l| l.rsplit(' ').next()?.parse::<u64>().ok())
            .unwrap()
    };
    assert_eq!(count(&table.metrics_text()), count(&before) + 1, "the admin attempt is counted");
    let (st, body) = http_ct_auth(addr, "POST", &format!("/v1/batches/{bid}/cancel"), "application/json", b"", "sk-t");
    assert_eq!(st, 200, "the owner can stop its batch: {body}");
    let (st, body) = http_ct_auth(addr, "POST", "/v1/batches/batch_nope/cancel", "application/json", b"", "sk-t");
    assert_eq!(st, 429, "{body}");
}

#[test]
fn tenant_keys_see_only_their_own_key_metrics() {
    let (addr, _d, _t) = spawn_key_stack(
        r#"{"keys":[{"name":"t1","key":"sk-t1"},{"name":"t2","key":"sk-t2"},{"name":"ops","key":"sk-ops","admin":true}]}"#,
        Some("sk-global"),
        0,
    );
    let (st, m) = http_hdr(addr, "GET", "/metrics", Some(("authorization", "Bearer sk-t1")));
    assert_eq!(st, 200);
    assert!(m.contains("superfluid_key_requests_total{key=\"t1\"}"), "{m}");
    assert!(!m.contains("key=\"t2\"") && !m.contains("key=\"ops\""), "no other tenant: {m}");
    for admin in ["sk-ops", "sk-global"] {
        let (_, m) = http_hdr(addr, "GET", "/v1/metrics", Some(("authorization", &format!("Bearer {admin}"))));
        assert!(m.contains("key=\"t1\"") && m.contains("key=\"t2\"") && m.contains("key=\"ops\""), "{admin}: {m}");
    }
}

#[test]
fn sync_routes_resolve_the_qos_header_against_the_key() {
    let (addr, _d, table) = spawn_key_stack(KEYS, None, 0);
    let emb = r#"{"model":"mock-model","input":"hello"}"#;
    let agents = ("authorization", "Bearer sk-agents");
    let (st, body) = post_hdrs(addr, "/v1/embeddings", emb, &[agents, ("x-superfluid-qos", "interactive")]);
    assert_eq!(st, 403, "above the key's max_class: {body}");
    assert!(body.contains("at most the completion class"), "{body}");
    let (st, body) = post_hdrs(addr, "/v1/embeddings", emb, &[agents, ("x-superfluid-qos", "urgent")]);
    assert_eq!(st, 400, "{body}");
    let (st, body) = post_hdrs(addr, "/v1/embeddings", emb, &[agents, ("x-superfluid-qos", "completion")]);
    assert_ne!(st, 403, "within the ceiling: {body}");
    let m = table.metrics_text();
    assert!(m.contains("superfluid_key_rejected_total{key=\"agents\",reason=\"qos\"} 1"), "{m}");
}

#[test]
fn batch_lines_are_paced_by_the_keys_rate_limit() {
    let (addr, _d, _t) =
        spawn_key_stack(r#"{"keys":[{"name":"slow","key":"sk-slow","rate_limit_rpm":2}]}"#, Some("sk-global"), 0);
    let line = r#"{"custom_id":"x","method":"POST","url":"/v1/chat/completions","body":{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":2}}"#;
    let up = format!("--BND\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--BND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\n\r\n{line}\n{line}\n{line}\n\r\n--BND--\r\n");
    let (us, ur) = http_ct_auth(addr, "POST", "/v1/files", "multipart/form-data; boundary=BND", up.as_bytes(), "sk-slow");
    assert_eq!(us, 200, "{ur}");
    let fid = serde_json::from_str::<serde_json::Value>(&ur).unwrap()["id"].as_str().unwrap().to_string();
    let (cs, cr) = http_ct_auth(addr, "POST", "/v1/batches", "application/json",
        format!(r#"{{"input_file_id":"{fid}","endpoint":"/v1/chat/completions"}}"#).as_bytes(), "sk-slow");
    assert_eq!(cs, 200, "{cr}");
    let bid = serde_json::from_str::<serde_json::Value>(&cr).unwrap()["id"].as_str().unwrap().to_string();
    std::thread::sleep(std::time::Duration::from_millis(400));
    let (_, gr) = http_ct_auth(addr, "GET", &format!("/v1/batches/{bid}"), "application/json", b"", "sk-global");
    let gv: serde_json::Value = serde_json::from_str(&gr).unwrap();
    assert_eq!(gv["request_counts"]["completed"], 0, "lines wait for the key's tokens: {gr}");
    let (st, body) = http_ct_auth(addr, "POST", &format!("/v1/batches/{bid}/cancel"), "application/json", b"", "sk-slow");
    assert_eq!(st, 200, "{body}");
    let mut status = String::new();
    for _ in 0..100 {
        let (_, gr) = http_ct_auth(addr, "GET", &format!("/v1/batches/{bid}"), "application/json", b"", "sk-global");
        status = serde_json::from_str::<serde_json::Value>(&gr).unwrap()["status"].as_str().unwrap_or("").to_string();
        if status == "cancelled" {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!(status, "cancelled", "a paced batch is still cancellable");
    let (st, body) = http_ct_auth(addr, "POST", "/v1/batches/batch_nope/cancel", "application/json", b"", "sk-slow");
    assert_eq!(st, 429, "{body}");
}

fn mock_daemon() -> Arc<Daemon> {
    mock_daemon_with(Box::new(MockChatCodec))
}

fn mock_daemon_with(codec: Box<dyn superfluid_daemon::TextCodec + Send + Sync>) -> Arc<Daemon> {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-reg-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host =
        EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    Arc::new(Daemon::new(store, host, codec, 8))
}

fn spawn_registry_stack() -> std::net::SocketAddr {
    use superfluid_daemon::registry::{ModelLoader, ModelRegistry};
    let loader: ModelLoader = Box::new(|_id: &str, _runtime| Ok(mock_daemon()));
    let registry = Arc::new(ModelRegistry::with_initial("model-a", mock_daemon(), loader));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_registry(listener, registry);
    });
    addr
}

struct TemplatedMockCodec(&'static str);

impl superfluid_daemon::TextCodec for TemplatedMockCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        superfluid_daemon::TextCodec::encode(&MockChatCodec, text)
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        superfluid_daemon::TextCodec::token_bytes(&MockChatCodec, token)
    }
    fn chat_template_source(&self) -> Option<String> {
        Some(self.0.to_string())
    }
}

fn serve_registry(registry: Arc<superfluid_daemon::registry::ModelRegistry>) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_registry(listener, registry);
    });
    addr
}

const THINKING_TEMPLATE: &str =
    "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}";

#[test]
fn props_reports_the_chat_template_of_the_model_it_describes() {
    let daemon = mock_daemon_with(Box::new(TemplatedMockCodec(THINKING_TEMPLATE)));
    let addr = serve_registry(Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", daemon)));
    for path in ["/props", "/props?model=mock-model", "/props?model=mock-model&autoload=1"] {
        let (s, _, b) = http(addr, "GET", path, "");
        assert_eq!(s, 200, "{path}: {b}");
        let v: serde_json::Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v["model"], "mock-model", "{path}: {v}");
        assert_eq!(v["loaded"], true, "{path}: {v}");
        assert_eq!(v["chat_template"], THINKING_TEMPLATE, "{path}: {v}");
    }
    let (s, _, b) = http(addr, "GET", "/props?model=nope&autoload=1", "");
    assert_eq!(s, 404, "{b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["error"]["code"], "model_not_found", "{v}");
}

#[test]
fn props_omits_the_chat_template_when_the_codec_has_none() {
    let addr = spawn_stack();
    let (s, _, b) = http(addr, "GET", "/props?model=mock-model", "");
    assert_eq!(s, 200, "{b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["loaded"], true, "{v}");
    assert!(v.get("chat_template").is_none(), "{v}");
}

#[test]
fn a_known_model_that_fails_to_load_says_why() {
    use superfluid_daemon::registry::{ModelLoader, ModelRegistry};
    use superfluid_daemon::DaemonError;
    let loader: ModelLoader = Box::new(|source: &str, _runtime| {
        if source.ends_with("model-bad.base") {
            Err(DaemonError::Generation("the engine worker ran out of memory".into()))
        } else if source.ends_with("model-refused.base") {
            Err(DaemonError::Config("--park-lossy: this runtime exports no lossy encoding"))
        } else {
            Ok(mock_daemon())
        }
    });
    let registry = Arc::new(ModelRegistry::with_initial("model-a", mock_daemon(), loader));
    registry.register_known("model-bad", std::path::Path::new("/models/model-bad.base"));
    registry.register_known("model-refused", std::path::Path::new("/models/model-refused.base"));
    let addr = serve_registry(Arc::clone(&registry));

    let body = r#"{"model":"model-bad","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
    let (s, _, b) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(s, 503, "{b}");
    assert!(b.contains("model 'model-bad' could not be loaded: "), "{b}");
    assert!(b.contains("the engine worker ran out of memory"), "{b}");
    assert!(!registry.is_loaded("model-bad"));

    let (s, _, b) = http(addr, "GET", "/props?model=model-bad&autoload=1", "");
    assert_eq!(s, 503, "the same through discovery: {b}");

    let body = r#"{"model":"model-refused","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
    let (s, _, b) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(s, 400, "{b}");
    assert!(b.contains("model 'model-refused' could not be loaded: "), "{b}");

    let body = r#"{"model":"model-zzz","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
    let (s, _, b) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(s, 404, "an unknown name is still not found: {b}");
}

#[test]
fn a_capability_refusal_of_a_load_is_the_clients() {
    use superfluid_daemon::registry::{ModelLoader, ModelRegistry};
    use superfluid_daemon::DaemonError;
    let loader: ModelLoader = Box::new(|_source: &str, _runtime| {
        Err(DaemonError::Unsupported(superfluid_daemon::Refusal::new(
            "unsupported_runtime",
            None,
            "--kv-bits: the llamacpp runtime keeps its KV type per context",
        )))
    });
    let registry = Arc::new(ModelRegistry::with_initial("model-a", mock_daemon(), loader));
    registry.register_known("model-kv", std::path::Path::new("/models/model-kv.gguf"));
    let addr = serve_registry(Arc::clone(&registry));
    let body = r#"{"model":"model-kv","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
    let (s, _, b) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(s, 400, "{b}");
    assert!(b.contains("model 'model-kv' could not be loaded: --kv-bits: the llamacpp runtime"), "{b}");
    assert!(b.contains(r#""code":"unsupported_runtime""#), "the refusal's code survives the autoload: {b}");
    let (s, _, b) = http(addr, "GET", "/props?model=model-kv&autoload=1", "");
    assert_eq!(s, 400, "the same through discovery: {b}");
}

#[test]
fn props_describes_an_unloaded_model_and_autoload_loads_it() {
    use superfluid_daemon::registry::{ModelLoader, ModelRegistry};
    let loader: ModelLoader =
        Box::new(|_id: &str, _runtime| Ok(mock_daemon_with(Box::new(TemplatedMockCodec("template-b")))));
    let registry = Arc::new(ModelRegistry::with_initial(
        "model-a",
        mock_daemon_with(Box::new(TemplatedMockCodec("template-a"))),
        loader,
    ));
    registry.register_known("model-b", std::path::Path::new("/models/model-b.base"));
    let addr = serve_registry(Arc::clone(&registry));

    let (s, _, b) = http(addr, "GET", "/props?model=model-b", "");
    assert_eq!(s, 200, "{b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["model"], "model-b", "{v}");
    assert_eq!(v["default_model"], "model-a", "{v}");
    assert_eq!(v["loaded"], false, "{v}");
    assert!(v.get("chat_template").is_none(), "an unloaded model's template is not known yet: {v}");
    assert!(v["capabilities"].is_null(), "no descriptor has been read for it: {v}");
    assert!(
        v["default_generation_settings"].get("top_p").is_none(),
        "nothing the model publishes is known yet, so the default model's must not stand in: {v}"
    );
    assert!(!registry.is_loaded("model-b"), "discovery alone must not load a model");

    let (s, _, b) = http(addr, "GET", "/props?model=model-b&autoload=1", "");
    assert_eq!(s, 200, "{b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["model"], "model-b", "{v}");
    assert_eq!(v["loaded"], true, "{v}");
    assert_eq!(v["chat_template"], "template-b", "{v}");
    assert!(registry.is_loaded("model-b"));

    let (s, _, b) = http(addr, "GET", "/props", "");
    assert_eq!(s, 200, "{b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["model"], "model-a", "{v}");
    assert_eq!(v["chat_template"], "template-a", "{v}");
}

#[test]
fn startup_multi_model_keys_by_stem_and_keeps_the_first_default() {
    use superfluid_daemon::registry::{ModelLoader, ModelRegistry};
    let loader: ModelLoader = Box::new(|_id: &str, _runtime| Ok(mock_daemon()));
    let registry = ModelRegistry::with_initial("model-a", mock_daemon(), loader);

    assert!(registry.load_as("model-b", "/models/model-b.base").unwrap());
    assert!(registry.load_as("model-c", "/models/model-c.base").unwrap());
    assert!(!registry.load_as("model-b", "/models/model-b.base").unwrap());

    let mut names = registry.names();
    names.sort();
    assert_eq!(names, vec!["model-a", "model-b", "model-c"]);
    assert!(registry.is_loaded("model-b"));
    assert!(!registry.is_loaded("/models/model-b.base"));
    assert_eq!(registry.default_name(), "model-a");
    assert_eq!(registry.resolve(None).unwrap().0, "model-a");

    registry.load("model-d").unwrap();
    assert_eq!(registry.default_name(), "model-d");
}

#[test]
fn an_unloaded_model_comes_back_by_its_name() {
    use superfluid_daemon::registry::{ModelLoader, ModelRegistry};
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let loader: ModelLoader = Box::new(move |source: &str, _runtime| {
        log.lock().unwrap().push(source.to_string());
        Ok(mock_daemon())
    });
    let registry = ModelRegistry::with_initial("model-a", mock_daemon(), loader);
    let dir = std::env::temp_dir().join(format!("sf-unload-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("model-b.gguf");
    std::fs::write(&file, b"x").unwrap();
    let path = file.to_str().unwrap().to_string();

    assert!(registry.load_as("model-b", &path).unwrap());
    assert!(registry.unload("model-b").unwrap());
    assert!(!registry.is_loaded("model-b"));
    assert!(registry.is_known("model-b"), "an unloaded model stays known by its name");

    let (name, _) = registry.try_resolve(Some("model-b")).unwrap();
    assert_eq!(name, "model-b", "a request by name loads it again under that name");
    assert!(registry.is_loaded("model-b"));

    assert!(registry.unload("model-b").unwrap());
    registry.load_on("model-b", None).unwrap();
    assert!(registry.is_loaded("model-b"), "a load by name finds its source");
    assert!(!registry.is_loaded(&path));
    assert_eq!(*seen.lock().unwrap(), vec![path.clone(), path.clone(), path]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn registry_load_list_route_and_unload() {
    let addr = spawn_registry_stack();

    let (st, _h, body) = http(addr, "GET", "/v1/models", "");
    assert_eq!(st, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let ids: Vec<&str> = v["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec!["model-a"]);

    let (st, _h, _b) = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"model-b","messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(st, 404);

    let (st, _h, _b) = http(addr, "POST", "/v1/models/load", r#"{"id":"model-b"}"#);
    assert_eq!(st, 200);
    let (st, _h, body) = http(addr, "GET", "/v1/models", "");
    assert_eq!(st, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let mut ids: Vec<&str> = v["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    ids.sort();
    assert_eq!(ids, vec!["model-a", "model-b"]);

    let (st, _h, body) = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"model-b","messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(st, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["model"], "model-b");

    let (st, _h, body) = http(addr, "POST", "/v1/models/unload", r#"{"id":"model-b"}"#);
    assert_eq!(st, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["deleted"], true);
    let (st, _h, _b) = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"model-b","messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(st, 404);

    let (st, _h, _b) = http(addr, "POST", "/v1/models/unload", r#"{"id":"model-a"}"#);
    assert_eq!(st, 400);
}

#[test]
fn single_model_server_refuses_dynamic_load() {
    let addr = spawn_stack();
    let (st, _h, _b) = http(addr, "POST", "/v1/models/load", r#"{"id":"whatever"}"#);
    assert_eq!(st, 400);
}

fn audio_multipart(model: &str, response_format: Option<&str>) -> (String, Vec<u8>) {
    let boundary = "AUDIOBND";
    let audio = b"RIFF....WAVEfake";
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n").as_bytes());
    if let Some(rf) = response_format {
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\n{rf}\r\n").as_bytes());
    }
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend_from_slice(audio);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

fn audio_multipart_fields(model: &str, fields: &[(&str, &str)]) -> (String, Vec<u8>) {
    let boundary = "AUDIOBND2";
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n").as_bytes());
    for (k, v) in fields {
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
    }
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend_from_slice(b"RIFF....WAVEfake");
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

#[test]
fn audio_transcription_json_and_verbose() {
    let addr = spawn_stack();
    let (ct, body) = audio_multipart("mock-model", None);
    let (st, resp) = http_ct(addr, "POST", "/v1/audio/transcriptions", &ct, &body);
    assert_eq!(st, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["text"], "mock transcription");

    let (ct, body) = audio_multipart("mock-model", Some("verbose_json"));
    let (st, resp) = http_ct(addr, "POST", "/v1/audio/transcriptions", &ct, &body);
    assert_eq!(st, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["task"], "transcribe");
    assert_eq!(v["language"], "en");
    assert_eq!(v["duration"].as_f64().unwrap(), 1.0);
    assert_eq!(v["segments"].as_array().unwrap().len(), 1);
    assert_eq!(v["segments"][0]["text"], "mock transcription");
}

#[test]
fn audio_translation_and_missing_model() {
    let addr = spawn_stack();
    let (ct, body) = audio_multipart("mock-model", None);
    let (st, resp) = http_ct(addr, "POST", "/v1/audio/translations", &ct, &body);
    assert_eq!(st, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["text"], "mock translation");

    let (ct, body) = audio_multipart("", None);
    let _ = (ct, body);
    let boundary = "NOMODEL";
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFFfake\r\n--{boundary}--\r\n").as_bytes());
    let (st, _resp) = http_ct(addr, "POST", "/v1/audio/transcriptions", &format!("multipart/form-data; boundary={boundary}"), &b);
    assert_eq!(st, 400);
}

#[test]
fn audio_prompt_task_and_timestamp_granularities() {
    let addr = spawn_stack();

    let (ct, body) = audio_multipart_fields("mock-model", &[("prompt", "Zyzzyva")]);
    let (st, resp) = http_ct(addr, "POST", "/v1/audio/transcriptions", &ct, &body);
    assert_eq!(st, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["text"], "Zyzzyva mock transcription");

    let (ct, body) =
        audio_multipart_fields("mock-model", &[("timestamp_granularities[]", "segment")]);
    let (st, resp) = http_ct(addr, "POST", "/v1/audio/transcriptions", &ct, &body);
    assert_eq!(st, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["task"], "transcribe");
    assert_eq!(v["segments"].as_array().unwrap().len(), 1);

    let (ct, body) =
        audio_multipart_fields("mock-model", &[("task", "translate"), ("temperature", "0.2")]);
    let (st, resp) = http_ct(addr, "POST", "/v1/audio/transcriptions", &ct, &body);
    assert_eq!(st, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["text"], "mock translation");

    let (ct, body) = audio_multipart_fields("mock-model", &[("task", "transcribe")]);
    let (st, resp) = http_ct(addr, "POST", "/v1/audio/translations", &ct, &body);
    assert_eq!(st, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["text"], "mock translation");
}

#[test]
fn audio_streaming_emits_deltas_then_done() {
    let addr = spawn_stack();
    let (ct, body) = audio_multipart_fields("mock-model", &[("stream", "true")]);
    let (st, resp) = http_ct(addr, "POST", "/v1/audio/transcriptions", &ct, &body);
    assert_eq!(st, 200, "{resp}");

    let events: Vec<serde_json::Value> = resp
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data is JSON"))
        .collect();
    assert!(events.len() >= 2, "want deltas + done, got {events:?}");

    assert_eq!(events[0]["type"], "transcript.text.delta");
    assert_eq!(events[0]["delta"], "mock");
    assert_eq!(events[1]["delta"], " transcription");

    let done = events.last().unwrap();
    assert_eq!(done["type"], "transcript.text.done");
    assert_eq!(done["text"], "mock transcription");
    assert_eq!(done["languages"][0]["code"], "en");

    let joined: String = events
        .iter()
        .filter(|e| e["type"] == "transcript.text.delta")
        .map(|e| e["delta"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(joined, done["text"].as_str().unwrap());

    assert!(resp.contains("data: [DONE]"), "{resp}");
}

#[test]
fn retired_slot_save_restore_answers_501_with_a_pointer() {
    let addr = spawn_stack();
    for path in ["/slots/mock-model/save", "/slots/mock-model/restore"] {
        let (st, _h, body) = http(addr, "POST", path, "{}");
        assert_eq!(st, 501, "{path}: {body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["code"], "route_retired");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("retired"), "{msg}");
        assert!(msg.contains("WAL"), "message must name what is durable: {msg}");
        assert!(msg.contains("prefix cache"), "{msg}");
        assert!(msg.contains("--park"), "message must name the flag: {msg}");
        assert!(
            !msg.contains("automatic"),
            "persistence is NOT automatic; the old wording sent operators away \
             believing their KV was saved: {msg}"
        );
    }
    let (st, _h, _b) = http(addr, "POST", "/slots/basecompute%2FQwen3-0.6B/save", "{}");
    assert_eq!(st, 501);
}

#[test]
fn audio_upload_past_the_default_body_limit_is_accepted() {
    let addr = spawn_stack();
    let boundary = "BIGAUDIO";
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nmock-model\r\n").as_bytes());
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend_from_slice(b"RIFF....WAVEfake");
    body.resize(body.len() + 3 * 1024 * 1024, 0u8);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let ct = format!("multipart/form-data; boundary={boundary}");
    let (st, resp) = http_ct(addr, "POST", "/v1/audio/transcriptions", &ct, &body);
    assert_eq!(st, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["text"], "mock transcription");
}

#[test]
fn audio_streaming_client_hangup_leaves_the_daemon_serving() {
    let addr = spawn_stack();
    let (ct, body) = audio_multipart_fields(
        "mock-model",
        &[("stream", "true"), ("prompt", "one two three four five six seven eight")],
    );
    {
        let mut s = TcpStream::connect(addr).unwrap();
        write!(s, "POST /v1/audio/transcriptions HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: {ct}\r\ncontent-length: {}\r\n\r\n", body.len()).unwrap();
        s.write_all(&body).unwrap();
        let mut buf = [0u8; 256];
        let _ = s.read(&mut buf).unwrap();
    }

    let (ct, body) = audio_multipart("mock-model", None);
    let (st, resp) = http_ct(addr, "POST", "/v1/audio/transcriptions", &ct, &body);
    assert_eq!(st, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["text"], "mock transcription");
}

#[test]
fn chat_accepts_logit_bias() {
    let addr = spawn_stack();
    let (st, _h, body) = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"logit_bias":{"100":50.0,"200":-100.0}}"#);
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert!(v["choices"][0]["message"]["content"].is_string());
}

#[test]
fn chat_logprobs_shape() {
    let addr = spawn_stack();
    let (st, _h, body) = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0,"logprobs":true}"#);
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let content = v["choices"][0]["logprobs"]["content"].as_array().expect("logprobs.content array");
    assert!(!content.is_empty(), "expected per-token logprobs");
    let first = &content[0];
    assert!(first["token"].is_string());
    assert!(first["logprob"].is_number());
    assert!(first["bytes"].is_array());
    assert!(first["top_logprobs"].is_array());
    let (st, _h, body) = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"temperature":0}"#);
    assert_eq!(st, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["choices"][0].get("logprobs").is_none() || v["choices"][0]["logprobs"].is_null());
}

#[test]
fn chat_streaming_with_logprobs_completes() {
    let addr = spawn_stack();
    let (status, head, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true,"temperature":0,"logprobs":true,"top_logprobs":2}"#,
    );
    assert_eq!(status, 200);
    assert!(head.to_ascii_lowercase().contains("text/event-stream"));
    let events: Vec<&str> = body
        .split("\n\n")
        .filter_map(|b| b.strip_prefix("data: "))
        .collect();
    assert_eq!(*events.last().unwrap(), "[DONE]");
    let first: serde_json::Value = serde_json::from_str(events[0]).unwrap();
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    let terminal: serde_json::Value = serde_json::from_str(events[events.len() - 2]).unwrap();
    assert!(terminal["choices"][0]["finish_reason"].is_string());
    assert_eq!(terminal["usage"]["completion_tokens"], 16);
    for e in &events[..events.len() - 1] {
        let v: serde_json::Value = serde_json::from_str(e).unwrap();
        if v["choices"][0]["delta"].get("content").is_some() {
            assert!(v["choices"][0]["logprobs"]["content"].is_array());
        }
    }
}

#[test]
fn completions_logprobs_shape() {
    let addr = spawn_stack();
    let (st, _h, body) = http(addr, "POST", "/v1/completions",
        r#"{"model":"mock-model","prompt":"hello","max_tokens":6,"temperature":0,"logprobs":2}"#);
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let lp = &v["choices"][0]["logprobs"];
    assert!(lp["tokens"].is_array(), "tokens array");
    assert!(lp["token_logprobs"].is_array());
    assert!(lp["top_logprobs"].is_array());
    assert!(lp["text_offset"].is_array());
    let n = lp["tokens"].as_array().unwrap().len();
    assert_eq!(n, lp["token_logprobs"].as_array().unwrap().len());
    assert_eq!(n, lp["text_offset"].as_array().unwrap().len());
    assert!(n >= 1);
    let (_st, _h, body) = http(addr, "POST", "/v1/completions",
        r#"{"model":"mock-model","prompt":"hi","max_tokens":3,"temperature":0}"#);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["choices"][0]["logprobs"].is_null());
}

#[test]
fn models_retrieve_and_delete_openai() {
    let addr = spawn_registry_stack();
    let (st, _h, body) = http(addr, "GET", "/v1/models/model-a", "");
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["id"], "model-a");
    assert_eq!(v["object"], "model");
    assert!(v["created"].is_number());
    assert!(v["owned_by"].is_string());
    let (st, _h, _b) = http(addr, "GET", "/v1/models/nope", "");
    assert_eq!(st, 404);
    let (st, _h, _b) = http(addr, "POST", "/v1/models/load", r#"{"id":"model-b"}"#);
    assert_eq!(st, 200);
    let (st, _h, body) = http(addr, "DELETE", "/v1/models/model-b", "");
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["id"], "model-b");
    assert_eq!(v["object"], "model");
    assert_eq!(v["deleted"], true);
    let (st, _h, _b) = http(addr, "DELETE", "/v1/models/model-a", "");
    assert_eq!(st, 400);
}

#[test]
fn grammar_handles_are_distinct_and_free_exactly_once() {
    let d = mock_daemon();
    let a = d.grammar_create(r#"{"type":"object"}"#).expect("create a");
    let b = d.grammar_create(r#"{"type":"object","properties":{"x":{"type":"number"}}}"#)
        .expect("create b");
    assert_ne!(a, 0, "a zero handle means 'no grammar' — never a live one");
    assert_ne!(b, 0);
    assert_ne!(a, b, "two grammars must not share a handle");

    d.grammar_free(a);
    d.grammar_free(b);
    let c = d.grammar_create(r#"{"type":"object"}"#).expect("create after free");
    assert_ne!(c, a);
    assert_ne!(c, b);
}

#[test]
fn grammar_create_refuses_a_malformed_schema() {
    let d = mock_daemon();
    for bad in [
        "",
        "not json",
        r#"{"type":"object""#,
        r#"{"type":"object"}}"#,
        "[]",
    ] {
        assert!(d.grammar_create(bad).is_err(), "should refuse {bad:?}");
    }
    assert!(d.grammar_create(r#"{"type":"object"}"#).is_ok());
}

#[test]
fn response_format_variants_all_produce_a_completion() {
    let addr = spawn_stack();
    for rf in [
        r#"{"type":"json_object"}"#,
        r#"{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"x":{"type":"number"}}}}}"#,
        r#"{"type":"json_schema","json_schema":{"type":"object"}}"#,
        r#"{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"array","items":{"type":"string"}}}}}}}}"#,
        r#"{"type":"text"}"#,
    ] {
        let body = format!(
            r#"{{"model":"mock-model","messages":[{{"role":"user","content":"hi"}}],
                "max_tokens":8,"temperature":0,"response_format":{rf}}}"#
        );
        let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", &body);
        assert_eq!(status, 200, "response_format {rf} -> {resp}");
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion", "{rf}");
        assert!(v["choices"][0]["message"]["content"].is_string(), "{rf}");
    }
}

#[test]
fn response_format_does_not_alter_the_generated_text() {
    use superfluid_daemon::codec::TextCodec as _;
    let script = superfluid_daemon::codec::MockCodec.encode("{\"x\":1}");
    let addr = spawn_scripted_stack(script);
    let body = r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],
        "max_tokens":64,"temperature":0,"response_format":{"type":"json_object"}}"#;
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "{\"x\":1}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[test]
fn response_format_works_with_n_and_streaming() {
    let addr = spawn_stack();
    let body = r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],
        "max_tokens":8,"temperature":0,"n":3,"response_format":{"type":"json_object"}}"#;
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(v["choices"].as_array().unwrap().len(), 3, "{resp}");

    let body = r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],
        "max_tokens":8,"temperature":0,"stream":true,"response_format":{"type":"json_object"}}"#;
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 200, "{raw}");
    assert!(raw.contains("chat.completion.chunk"), "{raw}");
    assert!(raw.contains("data: [DONE]"), "{raw}");

    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"again"}],"max_tokens":4}"#);
    assert_eq!(status, 200, "{resp}");
}

fn prompt_tokens_of(v: &serde_json::Value) -> u64 {
    v["usage"]["prompt_tokens"].as_u64().expect("usage.prompt_tokens")
}

fn chat(addr: std::net::SocketAddr, messages: &str) -> serde_json::Value {
    let body = format!(
        r#"{{"model":"mock-model","max_tokens":8,"temperature":0,"messages":{messages}}}"#
    );
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", &body);
    assert_eq!(status, 200, "{resp}");
    serde_json::from_str(&resp).unwrap()
}

#[test]
fn conversation_history_grows_the_prompt_turn_by_turn() {
    let addr = spawn_stack();
    let one = chat(addr, r#"[{"role":"user","content":"first question"}]"#);
    let three = chat(
        addr,
        r#"[{"role":"user","content":"first question"},
            {"role":"assistant","content":"first answer"},
            {"role":"user","content":"second question"}]"#,
    );
    let five = chat(
        addr,
        r#"[{"role":"user","content":"first question"},
            {"role":"assistant","content":"first answer"},
            {"role":"user","content":"second question"},
            {"role":"assistant","content":"second answer"},
            {"role":"user","content":"third question"}]"#,
    );
    assert!(
        prompt_tokens_of(&one) < prompt_tokens_of(&three),
        "history was dropped: {} vs {}",
        prompt_tokens_of(&one),
        prompt_tokens_of(&three)
    );
    assert!(
        prompt_tokens_of(&three) < prompt_tokens_of(&five),
        "later turns were dropped: {} vs {}",
        prompt_tokens_of(&three),
        prompt_tokens_of(&five)
    );
}

#[test]
fn system_prompt_reaches_the_model() {
    let addr = spawn_stack();
    let without = chat(addr, r#"[{"role":"user","content":"hi"}]"#);
    let with = chat(
        addr,
        r#"[{"role":"system","content":"You are a laconic assistant with strict rules."},
            {"role":"user","content":"hi"}]"#,
    );
    assert!(
        prompt_tokens_of(&with) > prompt_tokens_of(&without),
        "system prompt did not reach the model: {} vs {}",
        prompt_tokens_of(&with),
        prompt_tokens_of(&without)
    );
}

#[test]
fn tool_result_round_trip_reaches_the_next_prompt() {
    let addr = spawn_stack();
    let before_tool = chat(
        addr,
        r#"[{"role":"user","content":"weather in Paris?"},
            {"role":"assistant","tool_calls":[{"id":"call_1","type":"function",
              "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}]"#,
    );
    let with_result = chat(
        addr,
        r#"[{"role":"user","content":"weather in Paris?"},
            {"role":"assistant","tool_calls":[{"id":"call_1","type":"function",
              "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
            {"role":"tool","tool_call_id":"call_1",
             "content":"{\"temp_c\":18,\"conditions\":\"overcast with light rain\"}"}]"#,
    );
    assert!(
        prompt_tokens_of(&with_result) > prompt_tokens_of(&before_tool),
        "the tool result never reached the prompt: {} vs {}",
        prompt_tokens_of(&with_result),
        prompt_tokens_of(&before_tool)
    );
    assert_eq!(with_result["choices"][0]["message"]["role"], "assistant");
    assert!(with_result["choices"][0]["message"]["content"].is_string());
}

#[test]
fn conversation_ending_on_an_assistant_turn_is_served() {
    let addr = spawn_stack();
    let v = chat(
        addr,
        r#"[{"role":"user","content":"tell me a story"},
            {"role":"assistant","content":"Once upon a"}]"#,
    );
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
}

#[test]
fn marker_shaped_tool_result_does_not_prime_the_next_turn() {
    use superfluid_daemon::codec::{TextCodec as _, MOCK_TOOL_CLOSE, MOCK_TOOL_OPEN};
    let mut script = superfluid_daemon::codec::MockCodec.encode("Looking at the model.");
    script.push(MOCK_TOOL_OPEN);
    script.extend(
        superfluid_daemon::codec::MockCodec
            .encode(r#"{"name": "read", "arguments": {"path": "/a/AppModel.swift"}}"#),
    );
    script.push(MOCK_TOOL_CLOSE);
    let addr = spawn_scripted_stack(script);
    let body = r#"{"model":"mock-model","max_tokens":128,"temperature":0,
        "tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],
        "messages":[{"role":"user","content":"add a delete-all button"},
            {"role":"assistant","tool_calls":[{"id":"call_1","type":"function",
              "function":{"name":"read","arguments":"{\"path\":\"/a/ChatView.swift\"}"}}]},
            {"role":"tool","tool_call_id":"call_1",
             "content":"/// Collapsed-by-default reveal for a model's `\u0002` trace.\nstruct ThinkingReveal {}"}]}"#;
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let msg = &v["choices"][0]["message"];
    assert!(
        msg["reasoning_content"].is_null(),
        "the tool result primed the turn into reasoning: {msg}"
    );
    assert_eq!(msg["content"], "Looking at the model.", "{msg}");
    assert_eq!(msg["tool_calls"][0]["function"]["name"], "read", "the call leaked as text: {msg}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls", "{v}");
}

#[test]
fn marker_shaped_user_content_does_not_prime_the_next_turn() {
    use superfluid_daemon::codec::TextCodec as _;
    let script = superfluid_daemon::codec::MockCodec.encode("Hello!");
    let addr = spawn_scripted_stack(script);
    let body = r#"{"model":"mock-model","max_tokens":64,"temperature":0,
        "messages":[{"role":"user","content":"what does \u0002 mean?"}]}"#;
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let msg = &v["choices"][0]["message"];
    assert!(msg["reasoning_content"].is_null(), "primed into reasoning: {msg}");
    assert_eq!(msg["content"], "Hello!", "{msg}");
}

#[test]
fn reasoning_is_separated_from_content() {
    use superfluid_daemon::codec::{TextCodec as _, MOCK_THINK_CLOSE, MOCK_THINK_OPEN};
    let mut script = vec![MOCK_THINK_OPEN];
    script.extend(superfluid_daemon::codec::MockCodec.encode("the user wants a greeting"));
    script.push(MOCK_THINK_CLOSE);
    script.extend(superfluid_daemon::codec::MockCodec.encode("Hello!"));
    let addr = spawn_scripted_stack(script);

    let body = r#"{"model":"mock-model","max_tokens":64,"temperature":0,
        "messages":[{"role":"user","content":"hi"}]}"#;
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 200, "{resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["content"], "Hello!", "reasoning leaked into content: {msg}");
    assert_eq!(
        msg["reasoning_content"], "the user wants a greeting",
        "reasoning was dropped: {msg}"
    );
}

#[test]
fn streaming_separates_reasoning_from_content() {
    use superfluid_daemon::codec::{TextCodec as _, MOCK_THINK_CLOSE, MOCK_THINK_OPEN};
    let mut script = vec![MOCK_THINK_OPEN];
    script.extend(superfluid_daemon::codec::MockCodec.encode("thinking"));
    script.push(MOCK_THINK_CLOSE);
    script.extend(superfluid_daemon::codec::MockCodec.encode("Hello!"));
    let addr = spawn_scripted_stack(script);

    let body = r#"{"model":"mock-model","max_tokens":64,"temperature":0,"stream":true,
        "messages":[{"role":"user","content":"hi"}]}"#;
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 200, "{raw}");
    let events: Vec<serde_json::Value> = raw
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data is JSON"))
        .collect();

    let joined = |field: &str| -> String {
        events
            .iter()
            .filter_map(|e| e["choices"][0]["delta"][field].as_str())
            .collect()
    };
    assert_eq!(joined("content"), "Hello!", "{raw}");
    assert_eq!(joined("reasoning_content"), "thinking", "{raw}");
}

#[test]
fn tool_choice_never_rejects_a_request() {
    let addr = spawn_stack();
    for tc in [
        r#""auto""#,
        r#""none""#,
        r#""required""#,
        r#"{"type":"function","function":{"name":"f"}}"#,
        r#"{"type":"function","name":"f"}"#,
        r#""banana""#,
        r#"{"type":"nonsense"}"#,
        r#"{}"#,
    ] {
        let body = format!(
            r#"{{"model":"mock-model","max_tokens":8,"temperature":0,"tool_choice":{tc},
                "tools":[{{"type":"function","function":{{"name":"f","parameters":{{}}}}}}],
                "messages":[{{"role":"user","content":"go"}}]}}"#
        );
        let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", &body);
        assert_eq!(status, 200, "tool_choice {tc} -> {resp}");
    }
}

#[test]
fn demanded_constraints_fail_closed() {
    let addr = spawn_stack();
    let post = |body: &str| -> (u16, serde_json::Value) {
        let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", body);
        (status, serde_json::from_str(&resp).unwrap_or(serde_json::Value::Null))
    };
    let (status, v) = post(
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":6,
            "response_format":{"type":"json_schema","json_schema":{"schema":"not-an-object"}}}"#,
    );
    assert_eq!(status, 400, "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    let (status, v) = post(
        r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":6,
            "response_format":{"type":"json_schema"}}"#,
    );
    assert_eq!(status, 400, "{v}");
    let (status, v) = post(
        r#"{"model":"mock-model","max_tokens":6,
            "tool_choice":{"type":"function","function":{"name":"not_declared"}},
            "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
            "messages":[{"role":"user","content":"go"}]}"#,
    );
    assert_eq!(status, 400, "{v}");
    assert!(
        v["error"]["message"].as_str().is_some_and(|m| m.contains("not_declared")),
        "error should name the missing tool: {v}"
    );
    let (status, v) = post(
        r#"{"model":"mock-model","max_tokens":6,"tool_choice":"required","tools":[],
            "messages":[{"role":"user","content":"go"}]}"#,
    );
    assert_eq!(status, 400, "{v}");
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"again"}],"max_tokens":4}"#);
    assert_eq!(status, 200, "{resp}");
}

#[test]
fn demanded_constraints_fail_closed_when_streaming() {
    let addr = spawn_stack();
    let body = r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],
        "max_tokens":6,"stream":true,
        "response_format":{"type":"json_schema","json_schema":{"schema":"not-an-object"}}}"#;
    let (_status, _h, raw) = http(addr, "POST", "/v1/chat/completions", body);
    assert!(raw.contains("error"), "stream must surface the refusal: {raw}");
    assert!(
        !raw.contains("\"content\":"),
        "no content deltas may precede the refusal: {raw}"
    );
}

fn chat_tool_choice(addr: std::net::SocketAddr, tc: &str, stream: bool) -> String {
    let body = format!(
        r#"{{"model":"mock-model","max_tokens":512,"temperature":0,"stream":{stream},
            "tool_choice":{tc},
            "tools":[{{"type":"function","function":{{"name":"get_weather",
              "parameters":{{"type":"object","properties":{{"city":{{"type":"string"}}}}}}}}}},
                     {{"type":"function","function":{{"name":"get_time","parameters":{{}}}}}}],
            "messages":[{{"role":"user","content":"weather in Paris?"}}]}}"#
    );
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions", &body);
    assert_eq!(status, 200, "tool_choice {tc} -> {resp}");
    resp
}

#[test]
fn tool_choice_none_never_surfaces_a_tool_call() {
    let addr = spawn_scripted_stack(tool_envelope(
        r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#,
    ));

    let v: serde_json::Value =
        serde_json::from_str(&chat_tool_choice(addr, r#""none""#, false)).unwrap();
    let msg = &v["choices"][0]["message"];
    assert!(msg["tool_calls"].is_null(), "a call surfaced under `none`: {msg}");
    assert_ne!(v["choices"][0]["finish_reason"], "tool_calls", "{v}");
    assert!(
        msg["content"].as_str().is_some_and(|c| c.contains("get_weather")),
        "the call text should survive as content: {msg}"
    );

    let raw = chat_tool_choice(addr, r#""none""#, true);
    assert!(!raw.contains("\"tool_calls\""), "a tool_calls delta under `none`: {raw}");
    assert!(raw.contains("get_weather"), "the call text should survive: {raw}");
}

#[test]
fn tool_choice_none_withholds_the_declarations() {
    let addr = spawn_stack();
    let tokens = |tc: &str| -> u64 {
        let v: serde_json::Value =
            serde_json::from_str(&chat_tool_choice(addr, tc, false)).unwrap();
        v["usage"]["prompt_tokens"].as_u64().expect("prompt_tokens")
    };
    let auto = tokens(r#""auto""#);
    let none = tokens(r#""none""#);
    let required = tokens(r#""required""#);
    assert!(none < auto, "`none` still declared the tools: {none} vs {auto}");
    assert_eq!(required, auto, "`required` must still declare them: {required} vs {auto}");
}

#[test]
fn tool_choice_required_takes_the_constrained_path() {
    let addr = spawn_stack();
    for tc in [r#""required""#, r#"{"type":"function","function":{"name":"get_time"}}"#] {
        let v: serde_json::Value =
            serde_json::from_str(&chat_tool_choice(addr, tc, false)).unwrap();
        assert_eq!(v["object"], "chat.completion", "{tc}");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant", "{tc}");
        let raw = chat_tool_choice(addr, tc, true);
        assert!(raw.contains("data: [DONE]"), "{tc}: {raw}");
    }
    let (status, _h, resp) = http(addr, "POST", "/v1/chat/completions",
        r#"{"model":"mock-model","messages":[{"role":"user","content":"again"}],"max_tokens":4}"#);
    assert_eq!(status, 200, "{resp}");
}

#[test]
fn anthropic_tool_result_round_trip_reaches_the_next_prompt() {
    let addr = spawn_stack();
    let msgs = |extra: &str| {
        format!(
            r#"{{"model":"mock-model","max_tokens":8,
                "tools":[{{"name":"get_weather","description":"d","input_schema":{{"type":"object"}}}}],
                "messages":[{{"role":"user","content":"weather?"}},
                  {{"role":"assistant","content":[{{"type":"tool_use","id":"toolu_1",
                     "name":"get_weather","input":{{"city":"Paris"}}}}]}}{extra}]}}"#
        )
    };
    let (st, _h, a) = http(addr, "POST", "/v1/messages", &msgs(""));
    assert_eq!(st, 200, "{a}");
    let (st, _h, b) = http(
        addr,
        "POST",
        "/v1/messages",
        &msgs(
            r#",{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1",
               "content":"{\"temp_c\":18,\"conditions\":\"overcast with light rain\"}"}]}"#,
        ),
    );
    assert_eq!(st, 200, "{b}");
    let a: serde_json::Value = serde_json::from_str(&a).unwrap();
    let b: serde_json::Value = serde_json::from_str(&b).unwrap();
    let tokens = |v: &serde_json::Value| {
        let u = &v["usage"];
        ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]
            .iter()
            .map(|k| u[*k].as_u64().expect(k))
            .sum::<u64>()
    };
    assert!(
        tokens(&b) > tokens(&a),
        "the tool_result never reached the prompt: {} vs {}",
        tokens(&b),
        tokens(&a)
    );
}

#[test]
fn an_unterminated_tool_block_is_still_committed() {
    use superfluid_daemon::codec::{TextCodec as _, MOCK_TOOL_OPEN};
    let mut script = vec![MOCK_TOOL_OPEN];
    script.extend(
        superfluid_daemon::codec::MockCodec
            .encode(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#),
    );
    let addr = spawn_scripted_stack(script);
    let v = chat_with_tools(addr, "");

    let calls = v["choices"][0]["message"]["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("unterminated block was dropped: {}", v["choices"][0]["message"]));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn an_unterminated_unparseable_tool_block_surfaces_as_content() {
    use superfluid_daemon::codec::{TextCodec as _, MOCK_TOOL_OPEN};
    let mut script = vec![MOCK_TOOL_OPEN];
    script.extend(superfluid_daemon::codec::MockCodec.encode("{not json"));
    let addr = spawn_scripted_stack(script);
    let v = chat_with_tools(addr, "");

    let msg = &v["choices"][0]["message"];
    assert!(msg["tool_calls"].is_null(), "unparseable text became a call: {msg}");
    assert!(
        msg["content"].as_str().is_some_and(|c| c.contains("not json")),
        "the raw block vanished: {msg}"
    );
}

#[test]
fn tool_choice_required_routes_unmarked_output_as_a_call() {
    use superfluid_daemon::codec::TextCodec as _;
    let script = superfluid_daemon::codec::MockCodec
        .encode(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#);
    let addr = spawn_scripted_stack(script);

    let raw = chat_tool_choice(addr, r#""required""#, false);
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let calls = v["choices"][0]["message"]["tool_calls"]
        .as_array()
        .unwrap_or_else(|| panic!("unmarked output was not routed as a call: {raw}"));
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Paris");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");

    let raw = chat_tool_choice(addr, r#""required""#, true);
    assert!(raw.contains("\"tool_calls\""), "no tool_calls delta: {raw}");
    assert!(raw.contains("get_weather"), "{raw}");
}

fn call_from(addr: std::net::SocketAddr) -> serde_json::Value {
    chat_with_tools(addr, "")["choices"][0]["message"].clone()
}

fn scripted_call(payload: &str) -> std::net::SocketAddr {
    spawn_scripted_stack(tool_envelope(payload))
}

#[test]
fn tool_call_arguments_given_as_a_json_string_are_unwrapped() {
    let addr = scripted_call(r#"{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}"#);
    let msg = call_from(addr);
    let calls = msg["tool_calls"].as_array().unwrap_or_else(|| panic!("{msg}"));
    let args = calls[0]["function"]["arguments"].as_str().expect("arguments string");
    let parsed: serde_json::Value = serde_json::from_str(args).expect("arguments parse");
    assert!(parsed.is_object(), "double-encoded: json.loads gave {parsed:?}");
    assert_eq!(parsed["city"], "Paris");
}

#[test]
fn tool_call_inside_a_markdown_fence_is_salvaged() {
    let addr = scripted_call("```json\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n```");
    let msg = call_from(addr);
    let calls = msg["tool_calls"].as_array().unwrap_or_else(|| panic!("{msg}"));
    assert_eq!(calls[0]["function"]["name"], "get_weather");
}

#[test]
fn tool_call_wrapped_in_prose_is_salvaged() {
    let addr = scripted_call(
        r#"Sure, let me look that up: {"name":"get_weather","arguments":{"city":"Paris"}} Hope that helps!"#,
    );
    let msg = call_from(addr);
    let calls = msg["tool_calls"].as_array().unwrap_or_else(|| panic!("{msg}"));
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Paris");
}

#[test]
fn tool_call_with_a_trailing_comma_is_salvaged() {
    let addr = scripted_call(r#"{"name":"get_weather","arguments":{"city":"Paris",},}"#);
    let msg = call_from(addr);
    let calls = msg["tool_calls"].as_array().unwrap_or_else(|| panic!("{msg}"));
    assert_eq!(calls[0]["function"]["name"], "get_weather");
}

#[test]
fn tool_call_truncated_by_the_token_cap_is_closed_and_salvaged() {
    let addr = scripted_call(r#"{"name":"get_weather","arguments":{"city":"Paris"#);
    let msg = call_from(addr);
    let calls = msg["tool_calls"].as_array().unwrap_or_else(|| panic!("{msg}"));
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Paris");
}

#[test]
fn tool_call_in_the_function_wrapper_is_salvaged() {
    let addr = scripted_call(
        r#"{"type":"function","function":{"name":"get_weather","arguments":{"city":"Paris"}}}"#,
    );
    let msg = call_from(addr);
    let calls = msg["tool_calls"].as_array().unwrap_or_else(|| panic!("{msg}"));
    assert_eq!(calls[0]["function"]["name"], "get_weather");
}

#[test]
fn salvage_never_invents_a_tool_call() {
    for payload in [
        "I think the weather in Paris is nice today.",
        "{}",
        r#"{"arguments":{"city":"Paris"}}"#,
        r#"{"name":"","arguments":{}}"#,
        r#"{"name":"get_weather","arguments":"not json"}"#,
        "{{{{",
    ] {
        let addr = scripted_call(payload);
        let msg = call_from(addr);
        assert!(
            msg["tool_calls"].is_null(),
            "salvage invented a call from {payload:?}: {msg}"
        );
        assert!(
            !msg["content"].as_str().unwrap_or_default().is_empty(),
            "the raw text vanished for {payload:?}: {msg}"
        );
    }
}

#[test]
fn envelope_grammar_and_parser_are_the_same_definition() {
    use superfluid_daemon::codec::{MockChatCodec, TextCodec as _, ToolEnvelope};
    let codec = MockChatCodec;
    assert_eq!(codec.tool_envelope(), ToolEnvelope::Json);

    let tools = vec![
        r#"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}"#.to_string(),
    ];
    let schema = codec.call_grammar(&tools).expect("json envelope has a grammar");
    let schema: serde_json::Value = serde_json::from_str(&schema).unwrap();
    let name_const = schema["properties"]["name"]["const"].as_str().expect("name const");
    assert_eq!(name_const, "get_weather");

    let instance = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let (name, args) = codec.parse_tool_call(instance).expect("parser reads its own grammar");
    assert_eq!(name, name_const);
    assert_eq!(serde_json::from_str::<serde_json::Value>(&args).unwrap()["city"], "Paris");
}

struct ThinkingDialect;

impl superfluid_daemon::TextCodec for ThinkingDialect {
    fn encode(&self, text: &str) -> Vec<u32> {
        superfluid_daemon::TextCodec::encode(&MockChatCodec, text)
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        superfluid_daemon::TextCodec::token_bytes(&MockChatCodec, token)
    }
    fn supports_enable_thinking(&self) -> bool {
        true
    }
}

#[test]
fn models_advertise_enable_thinking_from_the_dialect() {
    use superfluid_daemon::codec::{ToolEnvelope, WithToolEnvelope};
    let dialect = |codec: Box<dyn superfluid_daemon::TextCodec + Send + Sync>| {
        let (st, body) = http_hdr(spawn_stack_with(codec), "GET", "/v1/models", None);
        assert_eq!(st, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        v["data"][0]["capabilities"]["dialect"].clone()
    };

    let plain = dialect(Box::new(MockChatCodec));
    assert_eq!(plain["enable_thinking"], false, "{plain}");
    assert_eq!(plain["reasoning_effort"], false, "{plain}");

    let thinking = dialect(Box::new(ThinkingDialect));
    assert_eq!(thinking["enable_thinking"], true, "{thinking}");
    assert_eq!(thinking["reasoning_effort"], false, "{thinking}");

    let wrapped = dialect(Box::new(WithToolEnvelope::new(Box::new(ThinkingDialect), ToolEnvelope::Json)));
    assert_eq!(wrapped["enable_thinking"], true, "{wrapped}");
}

#[test]
fn overriding_the_envelope_moves_parser_and_grammar_together() {
    use superfluid_daemon::codec::{MockChatCodec, TextCodec as _, ToolEnvelope, WithToolEnvelope};
    let plain = MockChatCodec;
    let wrapped = WithToolEnvelope::new(Box::new(MockChatCodec), ToolEnvelope::Json);
    assert_eq!(wrapped.tool_envelope(), ToolEnvelope::Json);

    let tools = vec![r#"{"name":"f","parameters":{"type":"object"}}"#.to_string()];
    assert_eq!(wrapped.call_grammar(&tools), plain.call_grammar(&tools));
    assert_eq!(
        wrapped.parse_tool_call(r#"{"name":"f","arguments":{}}"#),
        plain.parse_tool_call(r#"{"name":"f","arguments":{}}"#)
    );
    assert_eq!(wrapped.encode("hello"), plain.encode("hello"));
    assert_eq!(wrapped.channel_markers(), plain.channel_markers());
    assert_eq!(wrapped.render_message(1, "hi"), plain.render_message(1, "hi"));
}

#[test]
fn tool_call_parser_names_only_real_envelopes() {
    use superfluid_daemon::codec::ToolEnvelope;
    assert_eq!(ToolEnvelope::from_name("json"), Some(ToolEnvelope::Json));
    for bad in ["", "auto", "hermes", "mistral", "llama3_json", "JSON"] {
        assert!(ToolEnvelope::from_name(bad).is_none(), "accepted {bad:?}");
    }
    for n in ToolEnvelope::names() {
        assert!(ToolEnvelope::from_name(n).is_some(), "advertised but unknown: {n}");
        assert_eq!(ToolEnvelope::from_name(n).unwrap().name(), *n);
    }
}

fn weather_tool() -> String {
    r#"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}"#.to_string()
}

#[test]
fn structural_tag_frame_is_the_channelizer_frame() {
    use superfluid_daemon::codec::{MockChatCodec, TextCodec as _};
    let codec = MockChatCodec;
    let (begin, end) = codec.tool_call_delimiters().expect("dialect has tool markers");

    let (open, close, _) = codec
        .channel_markers()
        .into_iter()
        .find(|&(_, _, ch)| ch == 2 )
        .expect("a TOOL_CALL marker pair");
    assert_eq!(begin, codec.decode(&[open]));
    assert_eq!(end, codec.decode(&[close]));

    let tag: serde_json::Value =
        serde_json::from_str(&codec.structural_tag(&[weather_tool()], false).expect("tag")).unwrap();
    assert_eq!(tag["type"], "structural_tag");
    let fmt = &tag["format"];
    assert_eq!(fmt["type"], "triggered_tags");
    assert_eq!(fmt["triggers"][0], begin, "trigger must open the router's region");
    assert_eq!(fmt["tags"][0]["begin"], begin);
    assert_eq!(fmt["tags"][0]["end"], end);
}

#[test]
fn structural_tag_carries_every_declared_tool() {
    use superfluid_daemon::codec::{MockChatCodec, TextCodec as _};
    let time_tool = r#"{"type":"function","function":{"name":"get_time","parameters":{"type":"object"}}}"#.to_string();
    let tools = vec![weather_tool(), time_tool];
    let tag: serde_json::Value =
        serde_json::from_str(&MockChatCodec.structural_tag(&tools, false).expect("tag")).unwrap();

    let tags = tag["format"]["tags"].as_array().expect("tags");
    assert_eq!(tags.len(), 2);
    let names: Vec<&str> = tags
        .iter()
        .map(|t| t["content"]["json_schema"]["properties"]["name"]["const"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["get_weather", "get_time"]);
    let params = &tags[0]["content"]["json_schema"]["properties"]["arguments"];
    assert_eq!(params["properties"]["city"]["type"], "string");
    assert_eq!(params["required"][0], "city");
    let trigger = tag["format"]["triggers"][0].as_str().unwrap();
    for t in tags {
        assert!(t["begin"].as_str().unwrap().starts_with(trigger));
    }
}

#[test]
fn structural_tag_at_least_one_expresses_required() {
    use superfluid_daemon::codec::{MockChatCodec, TextCodec as _};
    let of = |req: bool| -> serde_json::Value {
        serde_json::from_str(&MockChatCodec.structural_tag(&[weather_tool()], req).expect("tag"))
            .unwrap()
    };
    assert_eq!(of(false)["format"]["at_least_one"], false);
    assert_eq!(of(true)["format"]["at_least_one"], true);
    assert_eq!(of(false)["format"]["stop_after_first"], false);
    assert_eq!(of(true)["format"]["stop_after_first"], true);
}

#[test]
fn structural_tag_is_absent_without_tools() {
    use superfluid_daemon::codec::{MockChatCodec, TextCodec as _};
    assert!(MockChatCodec.structural_tag(&[], false).is_none());
    assert!(MockChatCodec.structural_tag(&["not json".to_string()], false).is_none());
}

#[test]
fn structural_tag_admits_exactly_what_the_parser_reads() {
    use superfluid_daemon::codec::{MockChatCodec, TextCodec as _};
    let codec = MockChatCodec;
    let tag: serde_json::Value =
        serde_json::from_str(&codec.structural_tag(&[weather_tool()], false).expect("tag")).unwrap();
    let schema = &tag["format"]["tags"][0]["content"]["json_schema"];

    let instance = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let v: serde_json::Value = serde_json::from_str(instance).unwrap();
    assert_eq!(v["name"], schema["properties"]["name"]["const"]);

    let (name, args) = codec.parse_tool_call(instance).expect("parser reads the framed shape");
    assert_eq!(name, "get_weather");
    assert_eq!(serde_json::from_str::<serde_json::Value>(&args).unwrap()["city"], "Paris");
}

#[test]
fn auto_tool_choice_still_serves_when_the_tag_is_in_play() {
    let addr = spawn_stack();
    let v = chat_with_tools(addr, r#","tool_choice":"auto""#);
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    let v = chat_with_tools(addr, "");
    assert_eq!(v["object"], "chat.completion");
}

#[test]
fn hub_models_are_served_under_their_real_id() {
    use superfluid_daemon::registry::model_id_for;
    let dir = std::env::temp_dir().join(format!(
        "superfluid-modelid-{}-{}",
        std::process::id(),
        unique()
    ));
    let variant = dir.join("basecompute/gemma-4-E4B-it/default-q8");
    std::fs::create_dir_all(&variant).unwrap();
    let model = variant.join("model.base");
    std::fs::write(&model, b"x").unwrap();

    assert_eq!(model_id_for(&model), "model");

    std::fs::write(
        variant.join("hub.json"),
        br#"{"id":"basecompute/gemma-4-E4B-it","variant":"default-q8"}"#,
    )
    .unwrap();
    assert_eq!(model_id_for(&model), "basecompute/gemma-4-E4B-it");

    let loose = dir.join("Qwen3-0.6B-Q4_K_M.base");
    std::fs::write(&loose, b"x").unwrap();
    assert_eq!(model_id_for(&loose), "Qwen3-0.6B-Q4_K_M");

    for bad in [&b"not json"[..], br#"{}"#, br#"{"id":""}"#, br#"{"id":"   "}"#] {
        std::fs::write(variant.join("hub.json"), bad).unwrap();
        assert_eq!(model_id_for(&model), "model", "bad sidecar {bad:?}");
    }
}

fn spawn_narrow_stack(ctx: u32) -> std::net::SocketAddr {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-narrow-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || {
        let cfg = EngineConfig::default();
        let vocab = cfg.vocab;
        let e = MockEngine::new(cfg);
        let specs = vec![
            linkw::RingSpec {
                ring_id: TOKEN_RING_IN,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + ctx * 4,
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
    .expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking(listener, daemon, "mock-model".into());
    });
    addr
}

#[test]
fn overlong_prompt_is_400_context_length_exceeded() {
    let addr = spawn_narrow_stack(64);
    let long = "the session stream keeps growing and growing past the window. ".repeat(6);
    let body = format!(
        r#"{{"model":"mock-model","messages":[{{"role":"user","content":"{long}"}}]}}"#
    );
    let (status, _head, payload) = http(addr, "POST", "/v1/chat/completions", &body);
    assert_eq!(status, 400, "context overflow is the client's error: {payload}");
    let v: serde_json::Value = serde_json::from_str(&payload).expect("json error body");
    assert_eq!(v["error"]["code"], "context_length_exceeded", "body: {payload}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body: {payload}");
    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("64"), "message names the ceiling: {msg}");
    assert!(
        msg.chars().filter(|c| c.is_ascii_digit()).count() > 2,
        "message names the request size too: {msg}"
    );

    let ok = r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
    let (status, _h, payload) = http(addr, "POST", "/v1/chat/completions", ok);
    assert_eq!(status, 200, "a prompt within the window still serves: {payload}");
}

#[test]
fn overlong_prompt_streaming_is_400_before_the_body() {
    let addr = spawn_narrow_stack(64);
    let long = "the session stream keeps growing and growing past the window. ".repeat(6);
    let body = format!(
        r#"{{"model":"mock-model","messages":[{{"role":"user","content":"{long}"}}],"stream":true}}"#
    );
    let (status, head, payload) = http(addr, "POST", "/v1/chat/completions", &body);
    assert_eq!(status, 400, "context overflow is a 400 even when streaming: {payload}");
    assert!(
        !head.to_ascii_lowercase().contains("text/event-stream"),
        "the refusal is a JSON body, not a stream: {head}"
    );
    assert!(!payload.contains("data:"), "no SSE frames precede the refusal: {payload}");
    let v: serde_json::Value = serde_json::from_str(&payload).expect("json error body");
    assert_eq!(v["error"]["code"], "context_length_exceeded", "body: {payload}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body: {payload}");
    assert!(
        v["error"]["message"].as_str().unwrap_or_default().contains("64"),
        "message names the ceiling: {payload}"
    );

    let ok = r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#;
    let (status, head, payload) = http(addr, "POST", "/v1/chat/completions", ok);
    assert_eq!(status, 200, "a prompt within the window still streams: {payload}");
    assert!(head.to_ascii_lowercase().contains("text/event-stream"), "head: {head}");
    assert!(payload.contains("data: [DONE]"), "payload: {payload}");
}

#[test]
fn streaming_admission_counts_the_generation_opener() {
    let addr = spawn_narrow_stack(64);
    let req = |n: usize, stream: bool| {
        format!(
            r#"{{"model":"mock-model","messages":[{{"role":"user","content":"{}"}}],"max_tokens":1,"stream":{stream}}}"#,
            "x".repeat(n)
        )
    };
    let first_refused = (1..64)
        .find(|&n| http(addr, "POST", "/v1/chat/completions", &req(n, false)).0 == 400)
        .expect("a 64-token window refuses some prompt under 64 bytes");
    let (status, head, payload) = http(addr, "POST", "/v1/chat/completions", &req(first_refused, true));
    assert_eq!(status, 400, "content of {first_refused}: the opener overflows, so a 400 before the body: {payload}");
    assert!(!head.to_ascii_lowercase().contains("text/event-stream"), "head: {head}");
    assert!(payload.contains("context_length_exceeded"), "body: {payload}");
    let (status, _h, payload) = http(addr, "POST", "/v1/chat/completions", &req(first_refused - 1, true));
    assert_eq!(status, 200, "content of {}: fits with its opener: {payload}", first_refused - 1);
    assert!(payload.contains("data: [DONE]"), "payload: {payload}");
}

#[test]
fn props_reports_the_daemons_own_window() {
    let addr = spawn_narrow_stack(64);
    let (status, _h, body) = http(addr, "GET", "/props", "");
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["max_context"], 64, "{v}");
    assert_eq!(v["default_generation_settings"]["n_ctx"], 64, "{v}");
    assert_eq!(v["default_generation_settings"]["max_context"], 64, "{v}");
}

#[test]
fn overlong_completion_streaming_is_400_before_the_body() {
    let addr = spawn_narrow_stack(64);
    let long = "the session stream keeps growing and growing past the window. ".repeat(6);
    let body = format!(r#"{{"model":"mock-model","prompt":"{long}","stream":true,"echo":true,"max_tokens":4}}"#);
    let (status, head, payload) = http(addr, "POST", "/v1/completions", &body);
    assert_eq!(status, 400, "context overflow is a 400 even when streaming: {payload}");
    assert!(!head.to_ascii_lowercase().contains("text/event-stream"), "head: {head}");
    assert!(!payload.contains("data:"), "nothing was echoed before the refusal: {payload}");
    let v: serde_json::Value = serde_json::from_str(&payload).expect("json error body");
    assert_eq!(v["error"]["code"], "context_length_exceeded", "body: {payload}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body: {payload}");
}

#[test]
fn anthropic_streaming_overlong_prompt_is_400_before_the_body() {
    let addr = spawn_narrow_stack(64);
    let long = "the session stream keeps growing and growing past the window. ".repeat(6);
    let body = format!(
        r#"{{"model":"mock-model","max_tokens":8,"stream":true,"messages":[{{"role":"user","content":"{long}"}}]}}"#
    );
    let (status, head, payload) = http(addr, "POST", "/v1/messages", &body);
    assert_eq!(status, 400, "an over-context prompt is the client's error: {payload}");
    assert!(!head.to_ascii_lowercase().contains("text/event-stream"), "head: {head}");
    assert!(
        payload.contains("invalid_request_error") && !payload.contains("api_error"),
        "api_error reads as retryable, which is the livelock: {payload}"
    );
}

#[test]
fn spec_max_temperature_follows_the_routed_models_speculation() {
    let mk = |speculate: Option<&str>| {
        let dir = std::env::temp_dir().join(format!(
            "superfluid-specmax-{}-{}-{}",
            std::process::id(),
            speculate.unwrap_or("plain"),
            unique()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let store = SessionStore::open(&dir.join("wal.log")).unwrap();
        let mut cfg = EngineConfig::default();
        cfg.spaces.retain(|s| s.kind == superfluid_abi::space_kind::PAGED_TOKEN_KV);
        let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).expect("spawn");
        Arc::new(
            Daemon::with_options(
                store,
                host,
                Box::new(MockChatCodec),
                superfluid_daemon::DaemonOptions {
                    speculate: speculate.map(str::to_string),
                    ..Default::default()
                },
            )
            .unwrap(),
        )
    };
    let plain = mk(None);
    let spec = mk(Some("prompt-lookup"));
    assert!(spec.speculation().is_some(), "the mock registers prompt-lookup");
    let op = openai::SamplingOverrides { spec_max_temperature: Some(0.6), ..Default::default() };
    assert_eq!(op.for_daemon(&plain).spec_max_temperature, None, "a plain model is never capped");
    assert_eq!(op.for_daemon(&spec).spec_max_temperature, Some(0.6));

    let serve = |d: Arc<Daemon>| {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let cfg = openai::ServeConfig { sampling: op, ..Default::default() };
        let reg = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", d));
        std::thread::spawn(move || {
            let _ = openai::serve_blocking_config(listener, reg, cfg);
        });
        addr
    };
    let props_temp = |addr| {
        let (status, _, body) = http(addr, "GET", "/props", "");
        assert_eq!(status, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        v["default_generation_settings"]["temperature"].as_f64()
    };
    let t_spec = props_temp(serve(Arc::clone(&spec))).expect("temperature advertised");
    assert!((t_spec - 0.6).abs() < 1e-6, "speculating model capped: {t_spec}");
    let t_plain = props_temp(serve(Arc::clone(&plain))).expect("temperature advertised");
    assert!((t_plain - 1.0).abs() < 1e-6, "plain model left alone: {t_plain}");
}

struct RefusingTemplateCodec;

const REFUSAL: &str = "Conversation roles must alternate user/assistant/user/assistant/...";

impl superfluid_daemon::TextCodec for RefusingTemplateCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        superfluid_daemon::TextCodec::encode(&MockChatCodec, text)
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        superfluid_daemon::TextCodec::token_bytes(&MockChatCodec, token)
    }
    fn renders_per_message(&self) -> bool {
        false
    }
    fn render_prompt_structured_with(
        &self,
        _messages: &[superfluid_daemon::codec::ChatMessage],
        _tools: &[String],
        _kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        None
    }
    fn template_refusal(
        &self,
        _messages: &[superfluid_daemon::codec::ChatMessage],
        _tools: &[String],
        _kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<String> {
        Some(REFUSAL.to_string())
    }
}

#[test]
fn a_conversation_the_template_refuses_says_why() {
    let (addr, _d) = spawn_stack_with_daemon(Box::new(RefusingTemplateCodec));
    let body = r#"{"model":"mock-model","max_tokens":8,"temperature":0,
        "messages":[{"role":"user","content":"one"},{"role":"user","content":"two"}]}"#;
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", body);
    assert_eq!(status, 400, "{raw}");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains(REFUSAL), "the template's own words reach the client: {raw}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{raw}");

    let stream = body.replace(r#""temperature":0"#, r#""temperature":0,"stream":true"#);
    let (status, _h, raw) = http(addr, "POST", "/v1/chat/completions", &stream);
    assert_eq!(status, 400, "{raw}");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert!(v["error"]["message"].as_str().unwrap_or_default().contains(REFUSAL), "{raw}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{raw}");
}

type Rendered = Arc<std::sync::Mutex<Vec<(u32, String, Option<String>)>>>;

struct CapturingChatCodec(Rendered);

impl superfluid_daemon::TextCodec for CapturingChatCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        superfluid_daemon::TextCodec::encode(&MockChatCodec, text)
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        superfluid_daemon::TextCodec::token_bytes(&MockChatCodec, token)
    }
    fn renders_per_message(&self) -> bool {
        false
    }
    fn render_prompt_structured_with(
        &self,
        messages: &[superfluid_daemon::codec::ChatMessage],
        _tools: &[String],
        _kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        *self.0.lock().unwrap() = messages.iter().map(|m| (m.role, m.content.clone(), m.reasoning.clone())).collect();
        Some(superfluid_daemon::TextCodec::encode(&MockChatCodec, "rendered"))
    }
}

#[test]
fn a_thinking_only_assistant_turn_stays_in_the_history() {
    use superfluid_daemon::wal::role;
    let seen: Rendered = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (addr, _d) = spawn_stack_with_daemon(Box::new(CapturingChatCodec(Arc::clone(&seen))));
    let body = serde_json::json!({"model": "mock-model", "max_tokens": 2, "messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "work it out", "signature": ""}]},
        {"role": "user", "content": "go on"}
    ]})
    .to_string();
    let (status, _, resp) = http(addr, "POST", "/v1/messages", &body);
    assert_eq!(status, 200, "{resp}");
    let seen = seen.lock().unwrap().clone();
    let roles: Vec<u32> = seen.iter().map(|m| m.0).collect();
    assert_eq!(roles, vec![role::USER, role::ASSISTANT, role::USER], "{seen:?}");
    assert_eq!(seen[1].2.as_deref(), Some("work it out"), "{seen:?}");
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

fn spawn_unspecified_bind(api_key: Option<&str>) -> std::net::SocketAddr {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-openai-rebind-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", daemon));
    let listener = TcpListener::bind("0.0.0.0:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let cfg = openai::ServeConfig { api_key: api_key.map(str::to_string), ..Default::default() };
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    std::net::SocketAddr::from(([127, 0, 0, 1], port))
}

fn get_with_host(addr: std::net::SocketAddr, host: &str, extra: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(s, "GET /v1/models HTTP/1.1\r\nhost: {host}\r\nconnection: close\r\n{extra}\r\n").unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let raw = String::from_utf8_lossy(&raw).into_owned();
    let (head, payload) = raw.split_once("\r\n\r\n").expect("split");
    let status: u16 = head.lines().next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") { dechunk(payload) } else { payload.to_string() };
    (status, body)
}

#[test]
fn a_keyless_server_off_loopback_refuses_a_rebound_host_name() {
    let addr = spawn_unspecified_bind(None);
    let port = addr.port();
    let (st, body) = get_with_host(addr, &format!("rebound.evil.example:{port}"), "");
    assert_eq!(st, 403, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], "origin_refused");
    assert!(v["error"]["message"].as_str().unwrap().contains("--api-key"), "{body}");
    for host in [format!("127.0.0.1:{port}"), format!("192.168.1.5:{port}"), "localhost".to_string()] {
        let (st, body) = get_with_host(addr, &host, "");
        assert_eq!(st, 200, "{host}: {body}");
    }

    let keyed = spawn_unspecified_bind(Some("sk-k"));
    let rebound = format!("rebound.evil.example:{}", keyed.port());
    assert_eq!(get_with_host(keyed, &rebound, "").0, 401, "a keyed server leaves the name to the key");
    let (st, body) = get_with_host(keyed, &rebound, "authorization: Bearer sk-k\r\n");
    assert_eq!(st, 200, "{body}");
}

#[test]
fn batches_beyond_the_worker_cap_wait_in_validating_and_cancel_cleanly() {
    let addr = spawn_stack();
    let line = r#"{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":3500,"ignore_eos":true}}"#;
    let up = format!("--BND\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--BND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\n\r\n{line}\n{line}\n{line}\n{line}\n\r\n--BND--\r\n");
    let (us, ur) = http_ct(addr, "POST", "/v1/files", "multipart/form-data; boundary=BND", up.as_bytes());
    assert_eq!(us, 200, "{ur}");
    let input_id = serde_json::from_str::<serde_json::Value>(&ur).unwrap()["id"].as_str().unwrap().to_string();
    let cap = superfluid_daemon::batches::MAX_RUNNING_BATCHES;
    let ids: Vec<String> = (0..cap + 2)
        .map(|_| {
            let (cs, cr) = http_ct(addr, "POST", "/v1/batches", "application/json",
                format!(r#"{{"input_file_id":"{input_id}","endpoint":"/v1/chat/completions"}}"#).as_bytes());
            assert_eq!(cs, 200, "{cr}");
            serde_json::from_str::<serde_json::Value>(&cr).unwrap()["id"].as_str().unwrap().to_string()
        })
        .collect();
    let status = |id: &str| {
        let (_, gr) = http_ct(addr, "GET", &format!("/v1/batches/{id}"), "application/json", b"");
        serde_json::from_str::<serde_json::Value>(&gr).unwrap()["status"].as_str().unwrap_or("").to_string()
    };
    std::thread::sleep(std::time::Duration::from_millis(200));
    let statuses: Vec<String> = ids.iter().map(|id| status(id)).collect();
    let running = statuses.iter().filter(|s| *s == "in_progress").count();
    assert!(running <= cap, "at most {cap} batches run at once: {statuses:?}");
    assert_eq!(statuses.iter().filter(|s| *s == "validating").count(), 2, "{statuses:?}");

    for id in &ids {
        let (st, body) = http_ct(addr, "POST", &format!("/v1/batches/{id}/cancel"), "application/json", b"");
        assert_eq!(st, 200, "{body}");
    }
    for id in &ids {
        let mut last = String::new();
        for _ in 0..400 {
            last = status(id);
            if last == "cancelled" {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        assert_eq!(last, "cancelled", "{id}");
    }
}
