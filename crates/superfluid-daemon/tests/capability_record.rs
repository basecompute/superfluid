//! The capability record (Link W v6).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::registry::ModelRegistry;
use superfluid_daemon::runtime_pick::RuntimeId;
use superfluid_daemon::{openai, Daemon, DaemonError, DaemonOptions, EngineHost, SessionStore};
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives, SamplingDefaults};

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-caps-{tag}-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn executor_daemon(tag: &str, cfg: FakeConfig, runtime_id: Option<&'static str>) -> Arc<Daemon> {
    let dir = scratch(tag);
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || (Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()), None))
        .expect("spawn");
    Arc::new(
        Daemon::with_options(
            store,
            host,
            Box::new(MockChatCodec),
            DaemonOptions {
                media_dir: Some(dir.join("media")),
                runtime_id,
                trace_scope: superfluid_daemon::otlp::TraceScope::model("fake-model"),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

fn serve(daemon: Arc<Daemon>) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking(listener, daemon, "fake-model".into());
    });
    addr
}

fn http(addr: std::net::SocketAddr, method: &str, path: &str, body: &str) -> (u16, serde_json::Value) {
    let (status, raw) = http_raw(addr, method, path, body);
    (status, serde_json::from_str(raw.trim()).unwrap_or(serde_json::Value::Null))
}

fn http_raw(addr: std::net::SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
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
    let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        payload.split_once("\r\n").map(|(_, rest)| rest.rsplit_once("\r\n0\r\n").map_or(rest, |(b, _)| b)).unwrap_or(payload)
    } else {
        payload
    };
    (status, body.to_string())
}

const PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

#[test]
fn an_executor_runtime_describes_itself_in_the_hello() {
    let d = executor_daemon("describe", FakeConfig::default(), None);
    let caps = d.capability_descriptor().expect("the executor describes its runtime");
    assert_eq!(caps["runtime"], serde_json::json!({"id": "fake", "version": "1"}), "{caps}");
    assert_eq!(caps["modalities"]["image_encode"], "the fake runtime has no media path");
    assert_eq!(caps["workload"]["embedding"], "the fake runtime serves generation; it has no embedding path");
    assert_eq!(caps["serving"]["park_lossy"], "the fake runtime exports no lossy encoding");
    assert_eq!(caps["serving"]["park_lossless"], true);

    let addr = serve(d);
    let (status, list) = http(addr, "GET", "/v1/models", "");
    assert_eq!(status, 200, "{list}");
    let entry = &list["data"][0];
    assert_eq!(entry["runtime"], serde_json::json!({"id": "fake", "version": "1"}), "{entry}");
    assert_eq!(entry["architecture"]["input_modalities"], serde_json::json!(["text"]), "{entry}");
}

#[test]
fn what_the_record_refuses_is_a_client_error_in_its_words() {
    let addr = serve(executor_daemon("refuse", FakeConfig::default(), None));
    let chat = serde_json::json!({"model": "fake-model", "max_tokens": 4, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": PNG}}]}]});
    let (status, body) = http(addr, "POST", "/v1/chat/completions", &chat.to_string());
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    let msg = body["error"]["message"].as_str().unwrap_or_default();
    assert_eq!(msg, "model 'fake-model' on runtime fake does not accept images: the fake runtime has no media path", "{body}");
    assert_eq!((body["error"]["code"].as_str(), body["error"]["param"].as_str()), (Some("unsupported_input"), Some("messages[0].content")));

    let (status, body) = http(addr, "POST", "/v1/embeddings", r#"{"model":"fake-model","input":"hello"}"#);
    assert_eq!(status, 400, "{body}");
    let msg = body["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.starts_with("model 'fake-model' on runtime fake serves no embeddings: the fake runtime"), "{body}");
    assert_eq!((body["error"]["code"].as_str(), body["error"]["param"].as_str()), (Some("unsupported_model"), Some("model")));

    let mut streamed = chat.clone();
    streamed["stream"] = true.into();
    let (status, body) = http(addr, "POST", "/v1/chat/completions", &streamed.to_string());
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert_eq!((body["error"]["code"].as_str(), body["error"]["param"].as_str()), (Some("unsupported_input"), Some("messages[0].content")));

    let msgs = serde_json::json!({"model": "fake-model", "max_tokens": 4, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG.trim_start_matches("data:image/png;base64,")}}]}]});
    let (status, body) = http(addr, "POST", "/v1/messages", &msgs.to_string());
    assert_eq!(status, 400, "{body}");
    assert!(body["error"]["message"].as_str().unwrap_or_default().starts_with("model 'fake-model' on runtime fake does not accept images"), "{body}");

    let text = r#"{"model":"fake-model","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}"#;
    let (status, body) = http(addr, "POST", "/v1/chat/completions", text);
    assert_eq!(status, 200, "{body}");
}

#[test]
fn the_record_is_read_without_asking_the_worker() {
    let published = SamplingDefaults { temperature: Some(0.6), top_k: Some(20), ..Default::default() };
    let d = executor_daemon("offline", FakeConfig { sampling_defaults: published, ..Default::default() }, None);
    d.shutdown();
    let caps = d.capability_descriptor().expect("the record answers without the scheduler");
    assert_eq!(caps["sampling_defaults"], serde_json::json!({"temperature": 0.6, "top_k": 20}), "{caps}");
    let sd = d.model_sampling_defaults();
    assert_eq!((sd.temperature, sd.top_k), (Some(0.6), Some(20)));
}

#[test]
fn a_load_names_its_runtime_and_a_loaded_model_keeps_its_own() {
    let addr = serve(executor_daemon("load", FakeConfig::default(), None));
    let (status, body) = http(addr, "POST", "/v1/models/load", r#"{"model":"fake-model","runtime":"No Such"}"#);
    assert_eq!(status, 400, "{body}");
    let msg = body["error"]["message"].as_str().unwrap_or_default();
    assert_eq!(msg, "'No Such' is not a runtime id", "{body}");
    assert_eq!((body["error"]["code"].as_str(), body["error"]["param"].as_str()), (Some("unsupported_runtime"), Some("runtime")));

    let on_llamacpp = executor_daemon("loaded", FakeConfig::default(), Some("llamacpp"));
    let registry = ModelRegistry::with_initial("m", on_llamacpp, Box::new(|_: &str, _| Err(DaemonError::Config("no loads"))));
    registry.set_runtime_names(Box::new(|name: &str| match name {
        "llamacpp" | "llama.cpp" => Ok(RuntimeId::new("llamacpp")),
        other => Err(format!("no runtime '{other}' here (runtimes here: llamacpp reads a GGUF file)")),
    }));
    assert_eq!(registry.runtime_named("llama.cpp").unwrap(), RuntimeId::new("llamacpp"), "an alias");
    assert_eq!(registry.runtime_named("vllm").unwrap_err(), "no runtime 'vllm' here (runtimes here: llamacpp reads a GGUF file)");
    assert!(registry.load_on("m", Some(RuntimeId::new("llamacpp"))).is_ok(), "the same runtime is a no-op");
    let e = registry.load_on("m", Some(RuntimeId::new("mlx"))).unwrap_err();
    assert!(
        matches!(&e, DaemonError::Unsupported(r) if r.message == "model 'm' is loaded on runtime llamacpp; unload it to load it on mlx"
            && r.code == "unsupported_runtime" && r.param.as_deref() == Some("runtime")),
        "{e:?}"
    );
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
