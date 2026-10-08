//! P2.8 FIM completions against the mock stack.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use superfluid_daemon::codec::{fim_mode, MockChatCodec};
use superfluid_daemon::completion_bucket::BucketConfig;
use superfluid_daemon::{api, openai, Daemon, DaemonError, DaemonOptions, EngineHost, GenParams, MockCodec, SessionStore, TextCodec};
use superfluid_engine::{EngineConfig, MockEngine};

static DIRS: AtomicU64 = AtomicU64::new(0);

fn test_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fim-test-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("test dir");
    dir
}

fn kv_only_config() -> EngineConfig {
    let mut cfg = EngineConfig::default();
    cfg.spaces.retain(|s| s.kind == superfluid_abi::space_kind::PAGED_TOKEN_KV);
    cfg
}

fn daemon_with(codec: Box<dyn TextCodec + Send + Sync>, opts: DaemonOptions) -> Arc<Daemon> {
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(kv_only_config()), None)).unwrap();
    Arc::new(Daemon::with_options(store, host, codec, opts).unwrap())
}

fn satellite(codec: Box<dyn TextCodec + Send + Sync>) -> Arc<Daemon> {
    let host = EngineHost::spawn(|| (MockEngine::new(kv_only_config()), None)).unwrap();
    Arc::new(
        Daemon::with_options(
            SessionStore::ephemeral(),
            host,
            codec,
            DaemonOptions { max_lanes: 2, ..Default::default() },
        )
        .unwrap(),
    )
}

fn scripted_satellite(script: Vec<u32>) -> Arc<Daemon> {
    let host = EngineHost::spawn(move || {
        let mut cfg = kv_only_config();
        cfg.scripted = script;
        (MockEngine::new(cfg), None)
    })
    .unwrap();
    Arc::new(
        Daemon::with_options(SessionStore::ephemeral(), host, Box::new(MockChatCodec), DaemonOptions::default())
            .unwrap(),
    )
}

fn decoded(d: &Daemon) -> u64 {
    d.sched_stats().decode_tokens.load(Ordering::Relaxed)
}

const PRE: &str = "fn main() {\n    let x = ";
const SUF: &str = "\n}\n";

#[test]
fn satellite_serves_completions_and_primary_falls_back_without_one() {
    let primary = daemon_with(Box::new(MockCodec), DaemonOptions::default());
    assert!(!primary.fim_supported());
    assert!(matches!(primary.complete(PRE, SUF, fim_mode::PSM, 4), Err(DaemonError::NoFimDialect)));

    let bad = satellite(Box::new(MockCodec));
    assert!(matches!(primary.attach_fim_satellite("bad", bad), Err(DaemonError::NoFimDialect)));
    assert_eq!(primary.fim_satellite_name(), None);

    let sat = satellite(Box::new(MockChatCodec));
    primary.attach_fim_satellite("fim-sat", Arc::clone(&sat)).unwrap();
    assert!(primary.fim_supported());
    assert_eq!(primary.fim_satellite_name().as_deref(), Some("fim-sat"));
    assert!(
        matches!(primary.attach_fim_satellite("again", satellite(Box::new(MockChatCodec))), Err(DaemonError::Config(_))),
        "one satellite per daemon"
    );

    let before = decoded(&primary);
    let c = primary.complete(PRE, SUF, fim_mode::PSM, 6).unwrap();
    assert!(!c.cached && !c.expired);
    assert_eq!(c.tokens.len(), 6);
    assert_eq!(decoded(&primary), before, "the primary's engine never ran the completion");
    assert!(decoded(&sat) >= 6, "the satellite's engine did");
    assert_eq!(c.prompt_tokens as usize, MockChatCodec.render_fim(PRE, SUF, fim_mode::PSM).unwrap().len());
    let again = primary.complete(PRE, SUF, fim_mode::PSM, 6).unwrap();
    assert!(again.cached);
    assert_eq!(again.tokens, c.tokens);
    assert_eq!(again.finish, c.finish, "a cache hit reports the generation's own finish");
    assert!(primary.session_ids().is_empty());

    let m = primary.metrics_text();
    assert!(m.contains("superfluid_fim_completions_total{result=\"generated\"} 1"), "{m}");
    assert!(m.contains("superfluid_fim_completions_total{result=\"cached\"} 1"), "{m}");
    assert!(m.contains("superfluid_fim_satellite_info{model=\"fim-sat\"} 1"), "{m}");
    assert!(m.contains("superfluid_fim_satellite_decode_tokens_total"), "{m}");

    let solo = daemon_with(Box::new(MockChatCodec), DaemonOptions::default());
    let before = decoded(&solo);
    let c = solo.complete(PRE, SUF, fim_mode::SPM, 5).unwrap();
    assert_eq!(c.tokens.len(), 5);
    assert!(decoded(&solo) >= before + 5, "served on the primary's own engine");
    assert!(!solo.metrics_text().contains("superfluid_fim_satellite_info"));
}

fn serve_uds(daemon: Arc<Daemon>) -> PathBuf {
    let socket = test_dir().join("b.sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    std::thread::spawn(move || {
        let _ = api::serve(listener, daemon);
    });
    socket
}

fn complete_req(max_tokens: u32) -> api::Request {
    api::Request::Complete {
        prefix: PRE.into(),
        suffix: SUF.into(),
        mode: fim_mode::PSM,
        max_tokens,
    }
}

#[test]
fn uds_bucket_is_per_connection_refuses_typed_and_refills() {
    let d = daemon_with(
        Box::new(MockChatCodec),
        DaemonOptions {
            completion_bucket: BucketConfig { rate_per_sec: 5.0, burst: 2 },
            ..Default::default()
        },
    );
    let socket = serve_uds(Arc::clone(&d));
    let mut a = api::NativeClient::connect(&socket).unwrap();
    for _ in 0..2 {
        assert!(matches!(a.request(&complete_req(3)).unwrap(), api::Response::Completion { .. }));
    }
    let wait = match a.request(&complete_req(3)).unwrap() {
        api::Response::Throttled { retry_after_ms } => retry_after_ms,
        other => panic!("expected Throttled, got {other:?}"),
    };
    assert!((1..=200).contains(&wait), "one token refills in 200 ms at 5/s, got {wait}");
    assert!(matches!(a.request(&api::Request::List).unwrap(), api::Response::Sessions { .. }));
    let mut b = api::NativeClient::connect(&socket).unwrap();
    assert!(matches!(b.request(&complete_req(3)).unwrap(), api::Response::Completion { .. }));
    std::thread::sleep(std::time::Duration::from_millis(wait + 20));
    assert!(matches!(a.request(&complete_req(3)).unwrap(), api::Response::Completion { .. }));
    assert!(
        d.metrics_text().contains("superfluid_fim_completions_total{result=\"throttled\"} 1"),
        "{}",
        d.metrics_text()
    );
}

#[test]
fn keystroke_storm_is_throttled_and_session_lanes_keep_decoding() {
    let d = daemon_with(
        Box::new(MockChatCodec),
        DaemonOptions {
            completion_bucket: BucketConfig { rate_per_sec: 2.0, burst: 4 },
            completion_deadline_ms: 50,
            ..Default::default()
        },
    );
    let socket = serve_uds(Arc::clone(&d));
    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, None, (0..32).collect()).unwrap();
    let gen = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || {
            let extras = superfluid_daemon::scheduler::GenExtras { ignore_eos: true, ..Default::default() };
            d.generate_streaming_ex(session, 256, extras, |_| Ok(())).unwrap()
        })
    };
    let mut editor = api::NativeClient::connect(&socket).unwrap();
    let (mut served, mut expired, mut throttled) = (0, 0, 0);
    for i in 0..60u32 {
        let req = api::Request::Complete {
            prefix: format!("{PRE}{i}"),
            suffix: SUF.into(),
            mode: fim_mode::PSM,
            max_tokens: 4,
        };
        match editor.request(&req).unwrap() {
            api::Response::Completion { expired: true, .. } => expired += 1,
            api::Response::Completion { .. } => served += 1,
            api::Response::Throttled { .. } => throttled += 1,
            other => panic!("unexpected reply {other:?}"),
        }
    }
    let out = gen.join().unwrap();
    assert_eq!(out.tokens_generated, 256, "the session lane decoded everything it asked for");
    assert_eq!(served + expired + throttled, 60);
    assert!(throttled >= 40, "the storm is refused at the bucket, not queued (throttled {throttled})");
    assert!(served + expired >= 4, "the burst is admitted");
    assert_eq!(d.session_ids(), vec![session], "completions minted no durable session");
}

fn spawn_http(daemon: Arc<Daemon>, key: Option<&str>) -> std::net::SocketAddr {
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", daemon));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = openai::ServeConfig { api_key: key.map(str::to_string), ..Default::default() };
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    addr
}

fn post(addr: std::net::SocketAddr, path: &str, body: &str, hdrs: &[(&str, &str)]) -> (u16, String, String) {
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

fn fim_body(extra: &str) -> String {
    let t = if extra.contains("temperature") { "" } else { r#","temperature":0"# };
    format!(
        r#"{{"model":"mock-model","prompt":{},"suffix":{},"max_tokens":5{t}{extra}}}"#,
        serde_json::to_string(PRE).unwrap(),
        serde_json::to_string(SUF).unwrap()
    )
}

#[test]
fn v1_completions_suffix_serves_fim_stream_and_non_stream() {
    let primary = daemon_with(Box::new(MockChatCodec), DaemonOptions::default());
    let sat = scripted_satellite(MockCodec.encode("é€xyz"));
    primary.attach_fim_satellite("fim-sat", Arc::clone(&sat)).unwrap();
    let addr = spawn_http(Arc::clone(&primary), None);

    let (st, _h, body) = post(addr, "/v1/completions", &fim_body(""), &[]);
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["model"], "mock-model");
    let text = v["choices"][0]["text"].as_str().unwrap().to_string();
    assert_eq!(text, "é€");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 5);
    let prompt = MockChatCodec.render_fim(PRE, SUF, fim_mode::PSM).unwrap().len();
    assert_eq!(v["usage"]["prompt_tokens"], prompt);
    assert_eq!(v["usage"]["total_tokens"], prompt + 5);
    assert_eq!(v["superfluid"]["served_by"], "fim-sat");
    assert_eq!(v["superfluid"]["cached"], false);
    assert!(decoded(&sat) >= 5, "the satellite generated it");
    assert!(primary.session_ids().is_empty(), "a FIM request mints no durable session");

    let (st, head, body) = post(addr, "/v1/completions", &fim_body(r#","stream":true,"temperature":0.7,"seed":1"#), &[]);
    assert_eq!(st, 200, "{body}");
    assert!(head.to_ascii_lowercase().contains("text/event-stream"), "{head}");
    let frames: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(body.contains("data: [DONE]"), "{body}");
    let last = frames.last().expect("a finish frame");
    assert_eq!(last["choices"][0]["finish_reason"], "length");
    assert_eq!(last["superfluid"]["cached"], false, "a sampled request bypasses the cache");
    assert!(frames.iter().all(|f| f["object"] == "text_completion"));
    let streamed: String = frames.iter().map(|f| f["choices"][0]["text"].as_str().unwrap_or("")).collect();
    assert_eq!(streamed, "é€", "{body}");
    assert!(!body.contains('\u{FFFD}'), "a split UTF-8 sequence leaked: {body}");

    let (_st, _h, body) = post(addr, "/v1/completions", &fim_body(r#","stream":true"#), &[]);
    let frames: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let streamed: String = frames.iter().map(|f| f["choices"][0]["text"].as_str().unwrap_or("")).collect();
    assert_eq!(streamed, text);
    assert_eq!(frames.last().unwrap()["superfluid"]["cached"], true);

    for (extra, what) in [
        (r#","echo":true"#, "echo"),
        (r#","logprobs":2"#, "logprobs"),
        (r#","fim_mode":"xyz""#, "fim_mode"),
    ] {
        let (st, _h, body) = post(addr, "/v1/completions", &fim_body(extra), &[]);
        assert_eq!(st, 400, "{what}: {body}");
        assert!(body.contains(what), "{body}");
    }
    let (st, _h, body) = post(
        addr,
        "/v1/completions",
        r#"{"model":"mock-model","prompt":"hi","max_tokens":2,"fim_mode":"spm"}"#,
        &[],
    );
    assert_eq!(st, 400, "fim_mode without suffix: {body}");
    let (st, _h, body) = post(addr, "/v1/completions", &fim_body(r#","fim_mode":"spm""#), &[]);
    assert_eq!(st, 200, "{body}");

    let (st, _h, body) =
        post(addr, "/v1/completions", r#"{"model":"mock-model","prompt":"hi","max_tokens":3}"#, &[]);
    assert_eq!(st, 200, "{body}");
    assert_eq!(primary.session_ids().len(), 1, "non-FIM completions keep their durable session");

    let mut s = TcpStream::connect(addr).unwrap();
    write!(s, "GET /v1/models HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\n\r\n").unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).unwrap();
    let (_, payload) = raw.split_once("\r\n\r\n").unwrap();
    let payload = if raw.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        dechunk(payload)
    } else {
        payload.to_string()
    };
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    let dialect = &v["data"][0]["capabilities"]["dialect"];
    assert_eq!(dialect["fim"], true, "{v}");
    assert_eq!(dialect["fim_model"], "fim-sat", "{v}");
}

#[test]
fn v1_completions_suffix_without_fim_dialect_is_a_typed_400() {
    let d = daemon_with(Box::new(MockCodec), DaemonOptions::default());
    let addr = spawn_http(d, None);
    let (st, _h, body) = post(addr, "/v1/completions", &fim_body(""), &[]);
    assert_eq!(st, 400, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], "fim_unsupported", "{body}");
}

#[test]
fn http_bucket_is_per_client_429_with_retry_after_and_refills() {
    let d = daemon_with(
        Box::new(MockChatCodec),
        DaemonOptions {
            completion_bucket: BucketConfig { rate_per_sec: 4.0, burst: 2 },
            ..Default::default()
        },
    );
    let addr = spawn_http(Arc::clone(&d), None);
    let key_a = [("authorization", "Bearer editor-a")];
    let key_b = [("authorization", "Bearer editor-b")];
    for _ in 0..2 {
        let (st, _h, body) = post(addr, "/v1/completions", &fim_body(""), &key_a);
        assert_eq!(st, 200, "{body}");
    }
    let (st, head, body) = post(addr, "/v1/completions", &fim_body(""), &key_a);
    assert_eq!(st, 429, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "rate_limit_error");
    assert_eq!(v["error"]["code"], "completion_rate_limited");
    let retry = head
        .lines()
        .find_map(|l| l.to_ascii_lowercase().strip_prefix("retry-after: ").map(str::to_string))
        .expect("Retry-After header");
    assert_eq!(retry.trim(), "1", "{head}");
    let (st, _h, _body) = post(addr, "/v1/completions", &fim_body(r#","stream":true"#), &key_a);
    assert_eq!(st, 429);
    let (st, _h, body) = post(addr, "/v1/completions", &fim_body(""), &key_b);
    assert_eq!(st, 429, "a rotated unauthenticated header must not mint a bucket: {body}");
    let (st, _h, body) =
        post(addr, "/v1/completions", r#"{"model":"mock-model","prompt":"hi","max_tokens":2}"#, &key_a);
    assert_eq!(st, 200, "{body}");
    std::thread::sleep(std::time::Duration::from_millis(300));
    let (st, _h, body) = post(addr, "/v1/completions", &fim_body(""), &key_a);
    assert_eq!(st, 200, "{body}");
    assert!(d.metrics_text().contains("superfluid_fim_completions_total{result=\"throttled\"} 3"));
}

#[test]
fn http_bucket_keys_on_the_authenticated_credential() {
    let d = daemon_with(
        Box::new(MockChatCodec),
        DaemonOptions {
            completion_bucket: BucketConfig { rate_per_sec: 0.5, burst: 2 },
            ..Default::default()
        },
    );
    let addr = spawn_http(Arc::clone(&d), Some("sk-real"));
    let mut statuses = Vec::new();
    for i in 0..4 {
        let bogus = format!("Bearer rotated-{i}");
        let (st, _h, _body) = post(
            addr,
            "/v1/completions",
            &fim_body(""),
            &[("x-api-key", "sk-real"), ("authorization", bogus.as_str())],
        );
        statuses.push(st);
    }
    assert_eq!(statuses, vec![200, 200, 429, 429], "rotation did not reset the allowance");
}

#[test]
fn http_bucket_keys_on_the_policy_key() {
    let d = daemon_with(
        Box::new(MockChatCodec),
        DaemonOptions {
            completion_bucket: BucketConfig { rate_per_sec: 0.5, burst: 2 },
            ..Default::default()
        },
    );
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", Arc::clone(&d)));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let table = superfluid_daemon::keypolicy::KeyTable::parse(
        r#"{"keys":[{"name":"a","key":"sk-a"},{"name":"b","key":"sk-b"}]}"#,
        |_| None,
    )
    .unwrap();
    let cfg = openai::ServeConfig { keys: Some(Arc::new(table)), ..Default::default() };
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    let status = |key: &str| post(addr, "/v1/completions", &fim_body(""), &[("x-api-key", key)]).0;
    let a: Vec<u16> = (0..3).map(|_| status("sk-a")).collect();
    assert_eq!(a, vec![200, 200, 429], "key a spends its own allowance");
    assert_eq!(status("sk-b"), 200, "key b has an allowance of its own");
}

#[test]
fn fim_is_exempt_from_the_keys_class_ceiling() {
    let d = daemon_with(Box::new(MockChatCodec), DaemonOptions::default());
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", Arc::clone(&d)));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let table = superfluid_daemon::keypolicy::KeyTable::parse(
        r#"{"keys":[{"name":"bg","key":"sk-bg","class":"background"}]}"#,
        |_| None,
    )
    .unwrap();
    let cfg = openai::ServeConfig { keys: Some(Arc::new(table)), ..Default::default() };
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    let (st, _h, body) = post(addr, "/v1/completions", &fim_body(""), &[("x-api-key", "sk-bg")]);
    assert_eq!(st, 200, "{body}");
    let (st, _h, body) = post(
        addr,
        "/v1/completions",
        r#"{"model":"mock-model","prompt":"hi","max_tokens":2}"#,
        &[("x-api-key", "sk-bg"), ("x-superfluid-qos", "completion")],
    );
    assert_eq!(st, 403, "{body}");
}

#[test]
fn api_web_complete_is_metered() {
    let d = daemon_with(
        Box::new(MockChatCodec),
        DaemonOptions {
            completion_bucket: BucketConfig { rate_per_sec: 0.5, burst: 1 },
            ..Default::default()
        },
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let token = "web-token-for-test".to_string();
    {
        let (d, t) = (Arc::clone(&d), token.clone());
        std::thread::spawn(move || {
            let _ = superfluid_daemon::apiweb::serve_blocking(listener, d, t, Vec::new());
        });
    }
    std::thread::sleep(std::time::Duration::from_millis(100));
    let body = format!(
        r#"{{"Complete":{{"prefix":{},"suffix":{},"mode":0,"max_tokens":3}}}}"#,
        serde_json::to_string(PRE).unwrap(),
        serde_json::to_string(SUF).unwrap()
    );
    let auth = format!("Bearer {token}");
    let hdrs = [("authorization", auth.as_str()), ("x-superfluid-csrf", token.as_str())];
    let (st, _h, b) = post(addr, "/web/rpc", &body, &hdrs);
    assert_eq!(st, 200, "{b}");
    assert!(b.contains("Completion"), "{b}");
    let (st, head, b) = post(addr, "/web/rpc", &body, &hdrs);
    assert_eq!(st, 429, "{b}");
    assert!(b.contains("Throttled"), "{b}");
    assert!(head.to_ascii_lowercase().contains("retry-after: 2"), "{head}");
    let (st, _h, b) = post(addr, "/web/rpc", r#""List""#, &hdrs);
    assert_eq!(st, 200, "{b}");
}

#[test]
fn a_throttled_fim_request_never_autoloads_its_model() {
    use superfluid_daemon::registry::{ModelLoader, ModelRegistry};
    let opts = || DaemonOptions {
        completion_bucket: BucketConfig { rate_per_sec: 0.5, burst: 1 },
        ..Default::default()
    };
    let loads = Arc::new(AtomicU64::new(0));
    let loader: ModelLoader = {
        let loads = Arc::clone(&loads);
        Box::new(move |_id: &str, _runtime| {
            loads.fetch_add(1, Ordering::Relaxed);
            Ok(daemon_with(Box::new(MockChatCodec), opts()))
        })
    };
    let registry = Arc::new(ModelRegistry::with_initial(
        "mock-model",
        daemon_with(Box::new(MockChatCodec), opts()),
        loader,
    ));
    registry.register_known("cold-model", std::path::Path::new("/models/cold-model.base"));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    {
        let registry = Arc::clone(&registry);
        std::thread::spawn(move || {
            let _ = openai::serve_blocking_config(listener, registry, openai::ServeConfig::default());
        });
    }
    let (st, _h, body) = post(addr, "/v1/completions", &fim_body(""), &[]);
    assert_eq!(st, 200, "{body}");
    let cold = fim_body("").replace("\"mock-model\"", "\"cold-model\"");
    let (st, _h, body) = post(addr, "/v1/completions", &cold, &[]);
    assert_eq!(st, 429, "{body}");
    assert_eq!(loads.load(Ordering::Relaxed), 0, "a refused request autoloaded a model");
    assert!(!registry.is_loaded("cold-model"));
    let bad = fim_body(r#","echo":true"#).replace("\"mock-model\"", "\"cold-model\"");
    let (st, _h, _body) = post(addr, "/v1/completions", &bad, &[]);
    assert_eq!(st, 400);
    assert_eq!(loads.load(Ordering::Relaxed), 0);
}

#[test]
fn spm_uses_the_trained_layout() {
    const PRE_T: u32 = 4_000_000;
    const SUF_T: u32 = 4_000_001;
    const MID_T: u32 = 4_000_002;
    let span = MockChatCodec.render_fim("ab", "cd", fim_mode::SPM).unwrap();
    let mut want = vec![PRE_T, SUF_T];
    want.extend(MockCodec.encode("cd"));
    want.push(MID_T);
    want.extend(MockCodec.encode("ab"));
    assert_eq!(span, want);
    let psm = MockChatCodec.render_fim("ab", "cd", fim_mode::PSM).unwrap();
    let mut want = vec![PRE_T];
    want.extend(MockCodec.encode("ab"));
    want.push(SUF_T);
    want.extend(MockCodec.encode("cd"));
    want.push(MID_T);
    assert_eq!(psm, want);
}
