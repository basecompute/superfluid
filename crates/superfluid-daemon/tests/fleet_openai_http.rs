//! Fleet-head OpenAI HTTP (whole-API routing).

use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::fleet_manager::{FleetManager, NodeSpec, PlacementPolicy};
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::{Daemon, EngineHost, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkf::Lease;
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};

static DIRS: AtomicU64 = AtomicU64::new(0);

fn mock_daemon(engine: EngineConfig) -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fopenai-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || (MockEngine::new(engine), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

fn spawn_node() -> String {
    spawn_node_with(EngineConfig::default())
}

fn spawn_node_with(engine: EngineConfig) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon(engine));
        let node = NodeAgent::new(daemon, "openai-node", "mock");
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    addr
}

fn start_head(node_addr: String) -> String {
    start_head_with_ctx(node_addr, 8192)
}

fn start_head_with_ctx(node_addr: String, max_context: u32) -> String {
    start_head_cfg(node_addr, max_context, Default::default())
}

fn start_head_cfg(node_addr: String, max_context: u32, cfg: superfluid_daemon::openai::ServeConfig) -> String {
    let specs = vec![NodeSpec { addr: node_addr, auth: Vec::new() }];
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let manager = Arc::new(FleetManager::new(specs, PlacementPolicy::LeastLoaded, lease));
    // As `serve` does, so the head knows what each node serves before the first request.
    let _ = manager.probe();
    let dir = std::env::temp_dir().join(format!("superfluid-fopenai-head-{}-{}", std::process::id(), DIRS.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let daemon = Daemon::over_fleet(store, Box::new(MockChatCodec), manager, "mock".into(), Default::default()).unwrap();
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock", Arc::new(daemon)));
    let http = TcpListener::bind("127.0.0.1:0").unwrap();
    let http_addr = http.local_addr().unwrap().to_string();
    let cfg = superfluid_daemon::openai::ServeConfig { max_context, ..cfg };
    std::thread::spawn(move || {
        if let Err(e) = superfluid_daemon::openai::serve_blocking_config(http, registry, cfg) {
            eprintln!("fleet head: {e}");
        }
    });
    http_addr
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(10)))
        .build()
        .new_agent()
}

fn wait_ready(a: &ureq::Agent, http_addr: &str) {
    let url = format!("http://{http_addr}/v1/models");
    for _ in 0..100 {
        if a.get(&url).call().is_ok() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("fleet head never became ready");
}

#[test]
fn fleet_openai_non_streaming_chat_completion() {
    let http_addr = start_head(spawn_node());
    let a = agent();
    wait_ready(&a, &http_addr);

    let body = serde_json::json!({
        "model": "mock",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello there"}
        ],
        "max_tokens": 12,
        "seed": 0
    })
    .to_string();
    let mut resp = a
        .post(&format!("http://{http_addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .send(body.as_bytes())
        .expect("chat request");
    assert_eq!(resp.status().as_u16(), 200, "status");
    let text = resp.body_mut().read_to_string().unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "mock");
    let content = json["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.is_empty(), "fleet generated no content: {json}");
    let completion = json["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(completion > 0, "no completion tokens counted");
    assert!(
        json["choices"][0]["finish_reason"].is_string(),
        "missing finish_reason"
    );
}

fn head_metrics(a: &ureq::Agent, http_addr: &str) -> String {
    let mut resp = a.get(&format!("http://{http_addr}/metrics")).call().expect("metrics");
    assert_eq!(resp.status().as_u16(), 200);
    resp.body_mut().read_to_string().unwrap()
}

fn series(metrics: &str, name: &str, node_addr: &str) -> u64 {
    let key = format!("{name}{{node=\"openai-node\",addr=\"{node_addr}\"}} ");
    let line = metrics.lines().find(|l| l.starts_with(&key)).unwrap_or_else(|| panic!("no {key}in {metrics}"));
    line[key.len()..].parse().unwrap()
}

#[test]
fn fleet_head_metrics_list_each_node_and_count_its_tokens() {
    let node_addr = spawn_node();
    let http_addr = start_head(node_addr.clone());
    let a = agent();
    wait_ready(&a, &http_addr);
    let chat = |stream: bool| {
        let body = serde_json::json!({
            "model": "mock",
            "messages": [{"role": "user", "content": "hello there"}],
            "max_tokens": 12,
            "stream": stream
        })
        .to_string();
        let mut resp = a
            .post(&format!("http://{http_addr}/v1/chat/completions"))
            .header("content-type", "application/json")
            .send(body.as_bytes())
            .expect("chat request");
        assert_eq!(resp.status().as_u16(), 200);
        resp.body_mut().read_to_string().unwrap()
    };

    let json: serde_json::Value = serde_json::from_str(&chat(false)).unwrap();
    let completion = json["usage"]["completion_tokens"].as_u64().unwrap();
    let m = head_metrics(&a, &http_addr);
    assert!(m.contains("superfluid_fleet_head_info{"), "{m}");
    assert_eq!(series(&m, "superfluid_fleet_node_up", &node_addr), 1);
    assert_eq!(series(&m, "superfluid_fleet_node_lanes_active", &node_addr), 0);
    assert_eq!(series(&m, "superfluid_fleet_node_decode_tokens_total", &node_addr), completion);

    chat(true);
    let m = head_metrics(&a, &http_addr);
    assert!(series(&m, "superfluid_fleet_node_decode_tokens_total", &node_addr) > completion, "{m}");
}

#[test]
fn a_connection_tells_its_status_hook_what_the_node_reports() {
    let node_addr = spawn_node();
    let mut head = superfluid_daemon::fleet::FleetHead::connect(&node_addr, Vec::new()).expect("connect");
    let seen = Arc::new(AtomicU64::new(0));
    let hook_seen = Arc::clone(&seen);
    head.on_status(Arc::new(move |_| {
        hook_seen.fetch_add(1, Ordering::Relaxed);
    }));
    // The node reports every 2 s; the head reads it on the next call that pumps the link.
    std::thread::sleep(std::time::Duration::from_millis(2_500));
    head.drain_status().expect("node still there");
    assert!(seen.load(Ordering::Relaxed) > 0, "no status reached the hook");
}

#[test]
fn fleet_openai_rejects_unknown_model() {
    let http_addr = start_head(spawn_node());
    let a = agent();
    wait_ready(&a, &http_addr);

    let body = serde_json::json!({
        "model": "not-the-model",
        "messages": [{"role": "user", "content": "hi"}]
    })
    .to_string();
    let resp = a
        .post(&format!("http://{http_addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .send(body.as_bytes());
    let status = match resp {
        Ok(r) => r.status().as_u16(),
        Err(ureq::Error::StatusCode(code)) => code,
        Err(e) => panic!("unexpected transport error: {e}"),
    };
    assert_eq!(status, 404, "unknown model must 404");
}

#[test]
fn fleet_openai_streaming_chat_completion() {
    let http_addr = start_head(spawn_node());
    let a = agent();
    wait_ready(&a, &http_addr);

    let body = serde_json::json!({
        "model": "mock",
        "messages": [{"role": "user", "content": "hello there"}],
        "max_tokens": 12,
        "stream": true
    })
    .to_string();
    let mut resp = a
        .post(&format!("http://{http_addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .send(body.as_bytes())
        .expect("stream request");
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.body_mut().read_to_string().unwrap();
    assert!(text.contains("chat.completion.chunk"), "no chunk objects: {text}");
    assert!(text.contains("[DONE]"), "stream not terminated with [DONE]");
    assert!(
        text.contains("\"role\":\"assistant\""),
        "no assistant role opener in stream"
    );
}

#[test]
fn fleet_openai_models_carries_capabilities() {
    let http_addr = start_head(spawn_node());
    let a = agent();
    wait_ready(&a, &http_addr);

    let mut resp = a
        .get(&format!("http://{http_addr}/v1/models"))
        .call()
        .expect("models");
    assert_eq!(resp.status().as_u16(), 200);
    let v: serde_json::Value =
        serde_json::from_str(&resp.body_mut().read_to_string().unwrap()).unwrap();
    let entry = &v["data"][0];

    assert_eq!(entry["id"], "mock", "{entry}");
    assert_eq!(entry["object"], "model", "{entry}");
    assert_eq!(entry["owned_by"], "superfluid", "{entry}");

    assert!(entry["meta"].is_object(), "{entry}");
    assert!(entry["meta"]["n_ctx"].as_u64().unwrap_or(0) > 0, "{entry}");

    let caps = &entry["capabilities"];
    assert_eq!(caps["descriptor_version"], 1, "{caps}");
    assert_eq!(caps["workload"]["causal_generation"], true, "{caps}");
    assert!(!caps["workload"]["embedding"].is_string(), "embeddings reach a node, so they are not refused: {caps}");
    assert!(
        caps["dialect"]["reasoning_effort"].is_boolean(),
        "the head owns the template, so it answers this first-hand: {caps}"
    );
    assert!(caps["dialect"]["enable_thinking"].is_boolean(), "{caps}");

    assert!(caps["serving"].is_null(), "unobservable block guessed: {caps}");
    assert!(caps["ops"].is_null(), "unobservable block guessed: {caps}");
}

#[test]
fn the_head_advertises_what_its_node_does_and_answers_an_image_as_one_machine_would() {
    let http_addr = start_head(spawn_node());
    let a = agent();
    wait_ready(&a, &http_addr);
    // The head takes its node's descriptor once the node answers.
    let mut caps = serde_json::Value::Null;
    for _ in 0..100 {
        let mut resp = a.get(&format!("http://{http_addr}/v1/models")).call().expect("models");
        let v: serde_json::Value = serde_json::from_str(&resp.body_mut().read_to_string().unwrap()).unwrap();
        caps = v["data"][0]["capabilities"].clone();
        if caps["architecture"] == "mock" {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!(caps["architecture"], "mock", "the node's descriptor: {caps}");
    assert_eq!(caps["backend"], "fleet", "{caps}");

    let local = {
        let dir = std::env::temp_dir().join(format!("superfluid-fopenai-local-{}-{}", std::process::id(), DIRS.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&dir).unwrap();
        let store = SessionStore::open(&dir.join("wal.log")).unwrap();
        let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
        let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
        let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock", daemon));
        let http = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = http.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let _ = superfluid_daemon::openai::serve_blocking_config(http, registry, Default::default());
        });
        addr
    };
    wait_ready(&a, &local);
    let body = serde_json::json!({
        "model": "mock",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url":
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}},
        ]}],
        "max_tokens": 4,
    })
    .to_string();
    let status = |addr: &str| {
        let reader = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(10)))
            .http_status_as_error(false)
            .build()
            .new_agent();
        let resp = reader
            .post(&format!("http://{addr}/v1/chat/completions"))
            .header("content-type", "application/json")
            .send(body.as_bytes())
            .expect("image request");
        resp.status().as_u16()
    };
    assert_eq!(status(&http_addr), status(&local), "an engine that cannot take the image refuses it alike");
}

fn spawn_narrow_node(ctx: u32) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let dir = std::env::temp_dir().join(format!(
            "superfluid-fopenai-narrow-{}-{}",
            std::process::id(),
            DIRS.fetch_add(1, Ordering::Relaxed)
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
        .unwrap();
        let daemon = Arc::new(Daemon::new(store, host, Box::new(MockCodec), 8));
        let node = NodeAgent::new(daemon, "narrow-node", "mock");
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    addr
}

#[test]
fn fleet_non_streaming_overlong_prompt_is_400_context_length_exceeded() {
    const CTX: u32 = 64;
    let http_addr = start_head(spawn_narrow_node(CTX));
    wait_ready(&agent(), &http_addr);
    let a = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(10)))
        .http_status_as_error(false)
        .build()
        .new_agent();

    let long = "the session stream keeps growing and growing past the window. ".repeat(6);
    let body = serde_json::json!({
        "model": "mock",
        "messages": [{"role": "user", "content": long}],
        "max_tokens": 8
    })
    .to_string();
    let mut resp = a
        .post(&format!("http://{http_addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .send(body.as_bytes())
        .expect("the head answers rather than hanging");
    let status = resp.status().as_u16();
    let text = resp.body_mut().read_to_string().unwrap();
    assert_eq!(status, 400, "context overflow is the client's error: {text}");
    let v: serde_json::Value = serde_json::from_str(&text).expect("json error body");
    assert_eq!(v["error"]["code"], "context_length_exceeded", "body: {text}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body: {text}");
}

#[test]
fn models_reports_the_nodes_context_not_the_heads() {
    const NODE_CTX: u32 = 64;
    let http_addr = start_head_with_ctx(spawn_narrow_node(NODE_CTX), 32768);
    let a = agent();
    wait_ready(&a, &http_addr);

    let body = serde_json::json!({
        "model": "mock",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4
    })
    .to_string();
    let _ = a
        .post(&format!("http://{http_addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .send(body.as_bytes());

    let mut resp = a.get(&format!("http://{http_addr}/v1/models")).call().unwrap();
    let text = resp.body_mut().read_to_string().unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let n_ctx = v["data"][0]["meta"]["n_ctx"].as_u64().unwrap_or(0);
    assert_eq!(
        n_ctx, NODE_CTX as u64,
        "advertised {n_ctx}, but no node can hold more than {NODE_CTX}: {text}"
    );
}

#[test]
fn fleet_head_requires_the_api_key_when_one_is_set() {
    let http_addr = start_head_cfg(
        spawn_node(),
        8192,
        superfluid_daemon::openai::ServeConfig { api_key: Some("fleet-secret".into()), ..Default::default() },
    );
    let a = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(10)))
        .http_status_as_error(false)
        .build()
        .new_agent();
    let url = format!("http://{http_addr}/v1/models");
    let mut ready = false;
    for _ in 0..100 {
        if a.get(&url).call().is_ok() {
            ready = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(ready, "fleet head never became ready");

    let mut resp = a.get(&url).call().unwrap();
    assert_eq!(resp.status().as_u16(), 401, "no key");
    let json: serde_json::Value = serde_json::from_str(&resp.body_mut().read_to_string().unwrap()).unwrap();
    assert_eq!(json["error"]["type"], "authentication_error");
    assert_eq!(json["error"]["code"], "invalid_api_key");

    let resp = a.get(&url).header("authorization", "Bearer wrong").call().unwrap();
    assert_eq!(resp.status().as_u16(), 401, "wrong key");

    let mut resp = a.get(&url).header("authorization", "Bearer fleet-secret").call().unwrap();
    assert_eq!(resp.status().as_u16(), 200, "the key");
    let json: serde_json::Value = serde_json::from_str(&resp.body_mut().read_to_string().unwrap()).unwrap();
    assert_eq!(json["data"][0]["owned_by"], "superfluid");

    let body = serde_json::json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
        .to_string();
    let chat = format!("http://{http_addr}/v1/chat/completions");
    let resp = a.post(&chat).header("content-type", "application/json").send(body.as_bytes()).unwrap();
    assert_eq!(resp.status().as_u16(), 401, "chat without the key");
    let resp = a
        .post(&chat)
        .header("content-type", "application/json")
        .header("x-api-key", "fleet-secret")
        .send(body.as_bytes())
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200, "chat with the key in X-Api-Key");
}

#[test]
fn fleet_head_cancels_a_disconnected_stream_and_holds_its_slot_until_then() {
    use std::io::{Read, Write};
    let node = spawn_node_with(EngineConfig {
        tick_delay: std::time::Duration::from_millis(50),
        ..EngineConfig::default()
    });
    let policy = r#"{"keys":[{"name":"one","key":"sk-one","max_concurrent":1}]}"#;
    let table = Arc::new(superfluid_daemon::keypolicy::KeyTable::parse(policy, |_| None).unwrap());
    let key = table.keys().next().unwrap().clone();
    let http_addr = start_head_cfg(
        node,
        8192,
        superfluid_daemon::openai::ServeConfig { keys: Some(Arc::clone(&table)), ..Default::default() },
    );
    let a = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(10)))
        .http_status_as_error(false)
        .build()
        .new_agent();
    let models = format!("http://{http_addr}/v1/models");
    let t0 = std::time::Instant::now();
    while a.get(&models).call().is_err() {
        assert!(t0.elapsed() < std::time::Duration::from_secs(5), "fleet head never became ready");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hello"}],"max_tokens":4000,"stream":true,"temperature":0,"seed":0}"#;
    let mut s = std::net::TcpStream::connect(&http_addr).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\nauthorization: Bearer sk-one\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut got = String::new();
    let mut buf = [0u8; 1024];
    let t0 = std::time::Instant::now();
    while !got.contains("\"content\"") {
        assert!(t0.elapsed() < std::time::Duration::from_secs(10), "no content chunk: {got}");
        let n = s.read(&mut buf).unwrap();
        assert!(n > 0, "stream ended before any content: {got}");
        got.push_str(&String::from_utf8_lossy(&buf[..n]));
    }
    assert!(got.starts_with("HTTP/1.1 200"));
    assert_eq!(key.in_flight(), 1, "the open stream holds the slot");
    drop(s);
    let left = std::time::Instant::now();
    let mut saw_held = false;
    loop {
        let n = key.in_flight();
        assert!(n <= 1, "slot count never exceeds the cap");
        if n == 0 {
            break;
        }
        saw_held = true;
        assert!(left.elapsed() < std::time::Duration::from_secs(1), "the node was not cancelled: slot held {:?}", left.elapsed());
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(saw_held || left.elapsed() < std::time::Duration::from_millis(5), "the slot was released with the body");
    let chat = format!("http://{http_addr}/v1/chat/completions");
    let second = r#"{"model":"mock","messages":[{"role":"user","content":"again"}],"max_tokens":4,"temperature":0,"seed":0}"#;
    let resp = a
        .post(&chat)
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-one")
        .send(second.as_bytes())
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
}

#[test]
fn fleet_head_health_says_whether_a_node_serves_the_model() {
    let a: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(10)))
        .http_status_as_error(false)
        .build()
        .new_agent();
    let health = |addr: &str| {
        let mut resp = a.get(&format!("http://{addr}/health")).call().expect("health");
        let json: serde_json::Value = serde_json::from_str(&resp.body_mut().read_to_string().unwrap()).unwrap();
        (resp.status().as_u16(), json)
    };

    let http_addr = start_head(spawn_node());
    wait_ready(&agent(), &http_addr);
    let (status, json) = health(&http_addr);
    assert_eq!(status, 200, "{json}");
    assert_eq!(json["status"], "ok");
    assert_eq!(json["nodes_up"], 1);

    let gone = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string();
    let http_addr = start_head(gone);
    wait_ready(&agent(), &http_addr);
    let (status, json) = health(&http_addr);
    assert_eq!(status, 503, "{json}");
    assert_eq!(json["status"], "no_nodes");
    assert_eq!(json["nodes_known"], 1);
}
