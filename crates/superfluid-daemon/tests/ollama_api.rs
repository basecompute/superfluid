//! Ollama's HTTP contract over the real daemon/scheduler and a scripted engine.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use superfluid_daemon::codec::{
    MockChatCodec, MockCodec, TextCodec, MOCK_THINK_CLOSE, MOCK_THINK_OPEN, MOCK_TOOL_CLOSE,
    MOCK_TOOL_OPEN,
};
use superfluid_daemon::keypolicy::KeyTable;
use superfluid_daemon::registry::ModelRegistry;
use superfluid_daemon::{openai, Daemon, EngineHost, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use serde_json::{json, Value};

static NEXT: AtomicUsize = AtomicUsize::new(1);
fn daemon(cfg: EngineConfig) -> Arc<Daemon> {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-ollama-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || (MockEngine::new(cfg.clone()), None)).unwrap();
    Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8))
}
fn serve(registry: Arc<ModelRegistry>, cfg: openai::ServeConfig) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || openai::serve_blocking_config(listener, registry, cfg).unwrap());
    addr
}
fn scripted(script: Vec<u32>) -> SocketAddr {
    serve(
        Arc::new(ModelRegistry::single(
            "mock-model",
            daemon(EngineConfig {
                scripted: script,
                ..Default::default()
            }),
        )),
        openai::ServeConfig::default(),
    )
}
fn request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: &str,
    headers: &[(&str, &str)],
) -> (u16, String, String) {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let extra: String = headers
        .iter()
        .map(|(k, v)| format!("{k}: {v}\r\n"))
        .collect();
    write!(stream,"{method} {path} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\n{extra}content-length: {}\r\n\r\n{body}",body.len()).unwrap();
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes).unwrap();
    let raw = String::from_utf8(bytes).unwrap();
    let (head, payload) = raw.split_once("\r\n\r\n").unwrap();
    let status = head
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let payload = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        let mut rest = payload.as_bytes();
        let mut output = Vec::new();
        loop {
            let at = rest.windows(2).position(|w| w == b"\r\n").unwrap();
            let size = usize::from_str_radix(
                std::str::from_utf8(&rest[..at])
                    .unwrap()
                    .split(';')
                    .next()
                    .unwrap(),
                16,
            )
            .unwrap();
            rest = &rest[at + 2..];
            if size == 0 {
                break;
            }
            output.extend_from_slice(&rest[..size]);
            rest = &rest[size + 2..];
        }
        String::from_utf8(output).unwrap()
    } else {
        payload.to_owned()
    };
    (status, head.to_ascii_lowercase(), payload)
}
fn post(addr: SocketAddr, path: &str, body: Value) -> (u16, String, String) {
    request(addr, "POST", path, &body.to_string(), &[])
}
fn frames(body: &str) -> Vec<Value> {
    body.lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
fn query(stream: bool) -> Value {
    json!({"model":"mock-model","messages":[{"role":"user","content":"hi"}],"stream":stream,"options":{"num_predict":128,"temperature":0}})
}

#[test]
fn chat_and_generate_have_ollama_shapes_and_stream_by_default() {
    let addr = scripted(MockCodec.encode("Hello!"));
    for path in ["/api/chat", "/api/generate"] {
        let mut body = if path.ends_with("chat") {
            query(false)
        } else {
            json!({"model":"mock-model","prompt":"hi","stream":false,"options":{"num_predict":128,"temperature":0}})
        };
        let (status, head, text) = post(addr, path, body.clone());
        assert_eq!(status, 200, "{text}");
        assert!(head.contains("application/json"));
        let value: Value = serde_json::from_str(&text).unwrap();
        let content = if path.ends_with("chat") {
            &value["message"]["content"]
        } else {
            &value["response"]
        };
        assert_eq!(content, "Hello!");
        assert_eq!(value["done"], true);
        assert_eq!(value["done_reason"], "stop");
        assert!(value["eval_count"].as_u64().unwrap() > 0);
        assert!(value["prompt_eval_count"].as_u64().unwrap() > 0);
        assert!(value["created_at"].as_str().unwrap().ends_with('Z'));
        assert!(value.get("choices").is_none());
        body.as_object_mut().unwrap().remove("stream");
        let (status, head, text) = post(addr, path, body);
        assert_eq!(status, 200, "{text}");
        assert!(head.contains("application/x-ndjson"));
        assert!(!text.contains("data:"));
        let values = frames(&text);
        assert_eq!(values.iter().filter(|v| v["done"] == true).count(), 1);
        assert_eq!(values.last().unwrap()["done"], true);
        let joined: String = values
            .iter()
            .filter_map(|v| {
                if path.ends_with("chat") {
                    v["message"]["content"].as_str()
                } else {
                    v["response"].as_str()
                }
            })
            .collect();
        assert_eq!(joined, "Hello!");
        assert!(values.last().unwrap()["eval_count"].as_u64().unwrap() > 0);
    }
}

#[test]
fn thinking_is_separate_in_chat_and_generate_on_both_transports() {
    let mut script = vec![MOCK_THINK_OPEN];
    script.extend(MockCodec.encode("reason"));
    script.push(MOCK_THINK_CLOSE);
    script.extend(MockCodec.encode("answer"));
    let addr = scripted(script);
    for path in ["/api/chat", "/api/generate"] {
        for stream in [false, true] {
            let body = if path.ends_with("chat") {
                query(stream)
            } else {
                json!({"model":"mock-model","prompt":"hi","stream":stream,"options":{"num_predict":128,"temperature":0}})
            };
            let (status, _, text) = post(addr, path, body);
            assert_eq!(status, 200, "{text}");
            let values = frames(&text);
            let messages: Vec<_> = values
                .iter()
                .map(|v| {
                    if path.ends_with("chat") {
                        &v["message"]
                    } else {
                        v
                    }
                })
                .collect();
            assert_eq!(
                messages
                    .iter()
                    .filter_map(|v| v["thinking"].as_str())
                    .collect::<String>(),
                "reason"
            );
            assert_eq!(
                messages
                    .iter()
                    .filter_map(|v| v[if path.ends_with("chat") {
                        "content"
                    } else {
                        "response"
                    }]
                    .as_str())
                    .collect::<String>(),
                "answer"
            );
        }
    }
}

#[test]
fn tool_calls_arrive_as_objects_once_and_can_be_returned_with_results() {
    let mut script = vec![MOCK_TOOL_OPEN];
    script.extend(MockCodec.encode(r#"{"name":"weather","arguments":{"city":"Paris"}}"#));
    script.push(MOCK_TOOL_CLOSE);
    let addr = scripted(script);
    for stream in [false, true] {
        let mut body = query(stream);
        body["tools"] = json!([{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]);
        let (status, _, text) = post(addr, "/api/chat", body.clone());
        assert_eq!(status, 200, "{text}");
        let values = frames(&text);
        let calls: Vec<Value> = values
            .iter()
            .filter_map(|v| v["message"]["tool_calls"].as_array())
            .flatten()
            .cloned()
            .collect();
        assert_eq!(calls.len(), 1, "{text}");
        assert_eq!(calls[0]["function"]["name"], "weather");
        assert_eq!(calls[0]["function"]["arguments"], json!({"city":"Paris"}));
        assert_eq!(values.last().unwrap()["done_reason"], "stop");
        body["messages"].as_array_mut().unwrap().extend([
            json!({"role":"assistant","content":"","tool_calls":calls}),
            json!({"role":"tool","tool_name":"weather","content":"sunny"}),
        ]);
        body["stream"] = json!(false);
        let (status, _, reply) = post(addr, "/api/chat", body);
        assert_eq!(status, 200, "{reply}");
    }
}

#[test]
fn stop_sequences_are_held_back_across_ndjson_records() {
    let addr = scripted(MockCodec.encode("hello STOP hidden"));
    for stream in [false, true] {
        let mut body = query(stream);
        body["options"]["stop"] = json!(["STOP"]);
        let (status, _, text) = post(addr, "/api/chat", body);
        assert_eq!(status, 200, "{text}");
        let values = frames(&text);
        assert_eq!(
            values
                .iter()
                .filter_map(|v| v["message"]["content"].as_str())
                .collect::<String>(),
            "hello "
        );
        assert_eq!(values.last().unwrap()["done_reason"], "stop");
    }
}

#[test]
fn raw_generate_uses_plain_completion_and_reports_usage() {
    let addr = scripted(MockCodec.encode("raw answer"));
    for stream in [false, true] {
        let (status, _, text) = post(
            addr,
            "/api/generate",
            json!({"model":"mock-model","prompt":"hello","raw":true,"stream":stream,"options":{"num_predict":128,"temperature":0}}),
        );
        assert_eq!(status, 200, "{text}");
        let values = frames(&text);
        assert_eq!(
            values
                .iter()
                .filter_map(|v| v["response"].as_str())
                .collect::<String>(),
            "raw answer"
        );
        assert!(values.last().unwrap()["eval_count"].as_u64().unwrap() > 0);
    }
}

#[test]
fn suffix_generation_uses_fim_and_reports_usage_on_both_transports() {
    let addr = scripted(MockCodec.encode("42"));
    for stream in [false, true] {
        let (status, _, text) = post(
            addr,
            "/api/generate",
            json!({
                "model":"mock-model", "prompt":"let n = ", "suffix":";",
                "stream":stream, "options":{"num_predict":16,"temperature":0}
            }),
        );
        assert_eq!(status, 200, "{text}");
        let values = frames(&text);
        assert_eq!(
            values
                .iter()
                .map(|v| v["response"].as_str().unwrap())
                .collect::<String>(),
            "42"
        );
        let last = values.last().unwrap();
        assert_eq!(last["done"], true);
        assert_eq!(last["eval_count"], 2);
        assert!(last["prompt_eval_count"].as_u64().unwrap() > 0);
    }
}

#[test]
fn discovery_is_read_only_and_latest_alias_preloads_only_known_models() {
    let loads = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&loads);
    let registry = Arc::new(ModelRegistry::with_initial(
        "mock-model",
        daemon(EngineConfig::default()),
        Box::new(move |_, _| {
            count.fetch_add(1, Ordering::Relaxed);
            Ok(daemon(EngineConfig::default()))
        }),
    ));
    registry.register_known("cold", std::path::Path::new("/a/local/model.base"));
    let addr = serve(Arc::clone(&registry), openai::ServeConfig::default());
    for path in ["/api/tags", "/api/ps", "/api/version"] {
        let (status, _, body) = request(addr, "GET", path, "", &[]);
        assert_eq!(status, 200, "{body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        if path.ends_with("tags") {
            assert_eq!(value["models"].as_array().unwrap().len(), 2);
        }
        if path.ends_with("ps") {
            assert_eq!(value["models"].as_array().unwrap().len(), 1);
        }
        if path.ends_with("version") {
            assert_eq!(value["server"], "superfluid");
        }
    }
    for name in ["mock-model", "cold:latest"] {
        let (status, _, body) = post(addr, "/api/show", json!({"model":name}));
        assert_eq!(status, 200, "{body}");
    }
    assert_eq!(loads.load(Ordering::Relaxed), 0);
    let (status, _, body) = post(addr, "/api/generate", json!({"model":"cold:latest"}));
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["done_reason"],
        "load"
    );
    assert_eq!(loads.load(Ordering::Relaxed), 1);
    let (status, _, body) = post(addr, "/api/generate", json!({"model":"/etc/passwd"}));
    assert_eq!(status, 404, "{body}");
    assert_eq!(loads.load(Ordering::Relaxed), 1);
    let (status, _, body) = request(addr, "HEAD", "/api/tags", "", &[]);
    assert_eq!(status, 200);
    assert!(body.is_empty());
}

#[test]
fn embedding_batch_dimensions_normalization_and_truncation_are_real() {
    let addr = serve(
        Arc::new(ModelRegistry::single(
            "mock-model",
            daemon(EngineConfig::default()),
        )),
        openai::ServeConfig {
            max_context: 8,
            ..Default::default()
        },
    );
    let (status, _, body) = post(
        addr,
        "/api/embed",
        json!({"model":"mock-model","input":["hello","world"],"dimensions":3}),
    );
    assert_eq!(status, 200, "{body}");
    let value: Value = serde_json::from_str(&body).unwrap();
    let vectors = value["embeddings"].as_array().unwrap();
    assert_eq!(vectors.len(), 2);
    for v in vectors {
        let v = v.as_array().unwrap();
        assert_eq!(v.len(), 3);
        let norm: f64 = v.iter().map(|x| x.as_f64().unwrap().powi(2)).sum();
        assert!((norm - 1.0).abs() < 1e-5);
    }
    let text = "a".repeat(100);
    let (status, _, body) = post(
        addr,
        "/api/embed",
        json!({"model":"mock-model","input":text,"truncate":false}),
    );
    assert_eq!(status, 400, "{body}");
    let (status, _, body) = post(
        addr,
        "/api/embed",
        json!({"model":"mock-model","input":text}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["prompt_eval_count"],
        8
    );
    let (status, _, body) = post(
        addr,
        "/api/embed",
        json!({"model":"mock-model","input":[],"truncate":false}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["embeddings"],
        json!([])
    );
    let (status, _, body) = post(
        addr,
        "/api/embeddings",
        json!({"model":"mock-model","prompt":"hi"}),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["embedding"]
            .as_array()
            .unwrap()
            .len(),
        8
    );
}

#[test]
fn refusals_auth_and_rate_limits_use_ollama_error_envelopes() {
    let addr = scripted(MockCodec.encode("hello"));
    for (path, body, expected) in [
        ("/api/chat", json!({}), 400),
        ("/api/chat", json!({"model":"missing","messages":[]}), 404),
        (
            "/api/chat",
            json!({"model":"mock-model","keep_alive":0}),
            400,
        ),
        (
            "/api/generate",
            json!({"model":"mock-model","template":"bad"}),
            400,
        ),
        (
            "/api/generate",
            json!({"model":"mock-model","options":{"num_gpu":1}}),
            400,
        ),
        (
            "/api/generate",
            json!({"model":"mock-model","raw":true,"prompt":"hi","format":"json"}),
            400,
        ),
        ("/api/pull", json!({"model":"llama3"}), 501),
        (
            "/api/chat",
            json!({"model":"mock-model","messages":[{"role":"user","content":"describe","images":["not base64!"]}]}),
            400,
        ),
        ("/api/unknown", json!({}), 404),
        (
            "/api/embed",
            json!({"model":"mock-model","input":"hi","dimensions":99}),
            400,
        ),
    ] {
        let (status, _, text) = post(addr, path, body);
        assert_eq!(status, expected, "{path}: {text}");
        assert!(
            serde_json::from_str::<Value>(&text).unwrap()["error"].is_string(),
            "{text}"
        );
    }
    let d = daemon(EngineConfig::default());
    let addr = serve(
        Arc::new(ModelRegistry::single("mock-model", d)),
        openai::ServeConfig {
            api_key: Some("secret".into()),
            rate_limit_per_minute: 1,
            ..Default::default()
        },
    );
    let (status, _, body) = post(addr, "/api/chat", query(false));
    assert_eq!(status, 401);
    assert!(serde_json::from_str::<Value>(&body).unwrap()["error"].is_string());
    let h = [("authorization", "Bearer secret")];
    assert_eq!(request(addr, "GET", "/api/tags", "", &h).0, 200);
    let (status, _, body) = request(addr, "GET", "/api/tags", "", &h);
    assert_eq!(status, 429, "{body}");
    assert!(serde_json::from_str::<Value>(&body).unwrap()["error"].is_string());
}

#[test]
fn streaming_holds_key_quota_and_disconnect_releases_it() {
    let keys = Arc::new(
        KeyTable::parse(
            r#"{"keys":[{"name":"one","key":"secret","max_concurrent":1}]}"#,
            |_| None,
        )
        .unwrap(),
    );
    let policy = keys.keys().next().unwrap().clone();
    let d = daemon(EngineConfig {
        tick_delay: Duration::from_millis(15),
        ..Default::default()
    });
    let addr = serve(
        Arc::new(ModelRegistry::single("mock-model", d)),
        openai::ServeConfig {
            keys: Some(keys),
            ..Default::default()
        },
    );
    let mut body = query(true);
    body["options"]["num_predict"] = json!(4000);
    let body = body.to_string();
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(20)))
        .unwrap();
    write!(stream,"POST /api/chat HTTP/1.1\r\nhost: localhost\r\nauthorization: Bearer secret\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",body.len()).unwrap();
    let mut first = [0u8; 64];
    let n = stream.read(&mut first).unwrap();
    assert!(String::from_utf8_lossy(&first[..n]).starts_with("HTTP/1.1 200"));
    assert_eq!(policy.in_flight(), 1);
    let (status, _, text) = request(
        addr,
        "POST",
        "/api/chat",
        &query(false).to_string(),
        &[("authorization", "Bearer secret")],
    );
    assert_eq!(status, 429, "{text}");
    assert!(serde_json::from_str::<Value>(&text).unwrap()["error"].is_string());
    drop(stream);
    let start = Instant::now();
    while policy.in_flight() != 0 {
        assert!(start.elapsed() < Duration::from_secs(20), "quota leaked");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn an_explicit_unlimited_budget_overrides_only_the_default_token_count() {
    let d = daemon(EngineConfig {
        scripted: MockCodec.encode("full answer"),
        ..Default::default()
    });
    let addr = serve(
        Arc::new(ModelRegistry::single("mock-model", d)),
        openai::ServeConfig {
            default_max_tokens: Some(2),
            ..Default::default()
        },
    );
    for endpoint in ["/api/chat", "/api/generate"] {
        let mut body = if endpoint.ends_with("chat") {
            query(false)
        } else {
            json!({"model":"mock-model","prompt":"hi","stream":false,"options":{"temperature":0}})
        };
        body["options"]["num_predict"] = json!(-1);
        let (status, _, text) = post(addr, endpoint, body);
        assert_eq!(status, 200, "{text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            if endpoint.ends_with("chat") {
                &v["message"]["content"]
            } else {
                &v["response"]
            },
            "full answer"
        );
    }
}

#[test]
fn multiple_tool_calls_keep_their_names_arguments_and_result_pairing() {
    let mut script = Vec::new();
    for call in [
        json!({"name":"weather","arguments":{"city":"東京"}}),
        json!({"name":"clock","arguments":{"zone":"UTC"}}),
    ] {
        script.push(MOCK_TOOL_OPEN);
        script.extend(MockCodec.encode(&call.to_string()));
        script.push(MOCK_TOOL_CLOSE);
    }
    let addr = scripted(script);
    for stream in [false, true] {
        let mut body = query(stream);
        body["options"]["num_predict"] = json!(256);
        body["tools"] = json!([
            {"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},
            {"type":"function","function":{"name":"clock","parameters":{"type":"object","properties":{"zone":{"type":"string"}}}}}
        ]);
        let (status, _, text) = post(addr, "/api/chat", body.clone());
        assert_eq!(status, 200, "{text}");
        let values = frames(&text);
        let calls: Vec<_> = values
            .iter()
            .filter_map(|v| v["message"]["tool_calls"].as_array())
            .flatten()
            .cloned()
            .collect();
        assert_eq!(calls.len(), 2, "{text}");
        assert_eq!(calls[0]["function"]["arguments"], json!({"city":"東京"}));
        assert_eq!(calls[1]["function"]["arguments"], json!({"zone":"UTC"}));
        assert_ne!(calls[0]["id"], calls[1]["id"]);
        body["messages"].as_array_mut().unwrap().extend([
            json!({"role":"assistant","tool_calls":calls}),
            json!({"role":"tool","tool_name":"clock","content":"noon"}),
            json!({"role":"tool","tool_name":"weather","content":"rain"}),
        ]);
        body["stream"] = json!(false);
        let (status, _, reply) = post(addr, "/api/chat", body);
        assert_eq!(status, 200, "{reply}");
    }
}

#[test]
fn ndjson_preserves_unicode_newlines_and_json_looking_content() {
    let expected = "東京 café 🦀\n\n{\"done\":true}\r\ndata: [DONE]";
    let addr = scripted(MockCodec.encode(expected));
    for stream in [false, true] {
        for path in ["/api/chat", "/api/generate"] {
            let body = if path.ends_with("chat") {
                query(stream)
            } else {
                json!({"model":"mock-model","prompt":"hi","stream":stream,"options":{"num_predict":128}})
            };
            let (status, _, text) = post(addr, path, body);
            assert_eq!(status, 200, "{text}");
            let values = frames(&text);
            let actual: String = values
                .iter()
                .map(|v| {
                    if path.ends_with("chat") {
                        v["message"]["content"].as_str().unwrap()
                    } else {
                        v["response"].as_str().unwrap()
                    }
                })
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(values.iter().filter(|v| v["done"] == true).count(), 1);
        }
    }
}

#[test]
fn malformed_and_overflowing_inputs_are_client_errors_and_do_not_poison_serving() {
    let addr = scripted(MockCodec.encode("ok"));
    for body in [
        "",
        "{",
        "null",
        "[]",
        r#"{"model":"mock-model","stream":"true"}"#,
        r#"{"model":"mock-model","options":{"temperature":1e100}}"#,
        r#"{"model":"mock-model","options":{"top_k":4294967296}}"#,
        r#"{"model":"mock-model","options":{"num_predict":4294967296}}"#,
        r#"{"model":"mock-model","think":7}"#,
        r#"{"model":"mock-model","format":[]}"#,
        r#"{"model":"mock-model","messages":[{"role":"tool","content":"orphan"}]}"#,
        r#"{"model":"mock-model","messages":[{"role":"assistant","tool_calls":[{"function":{"name":"x","arguments":"{}"}}]}]}"#,
    ] {
        let (status, _, text) = request(addr, "POST", "/api/chat", body, &[]);
        assert_eq!(status, 400, "{body}: {text}");
        assert!(serde_json::from_str::<Value>(&text).unwrap()["error"].is_string());
    }
    let (status, _, text) = post(addr, "/api/chat", query(false));
    assert_eq!(status, 200, "{text}");
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap()["message"]["content"],
        "ok"
    );
}

#[test]
fn empty_embedding_requests_preload_without_embedding_text() {
    let addr = scripted(MockCodec.encode("ok"));
    for mut body in [
        json!({}),
        json!({"input":""}),
        json!({"input":null}),
        json!({"input":[]}),
    ] {
        body["model"] = json!("mock-model");
        let (status, _, text) = post(addr, "/api/embed", body);
        assert_eq!(status, 200, "{text}");
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["embeddings"], json!([]), "{text}");
        assert_eq!(value["prompt_eval_count"], 0);
    }
    for body in [
        json!({"model":"mock-model"}),
        json!({"model":"mock-model","prompt":""}),
    ] {
        let (status, _, text) = post(addr, "/api/embeddings", body);
        assert_eq!(status, 200, "{text}");
        assert_eq!(
            serde_json::from_str::<Value>(&text).unwrap()["embedding"],
            json!([])
        );
    }
    let (status, _, text) = post(
        addr,
        "/api/embed",
        json!({"model":"mock-model","input":[""]}),
    );
    assert_eq!(status, 200, "{text}");
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap()["embeddings"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn cohosted_protocols_keep_their_wire_contracts_under_concurrent_requests() {
    let addr = scripted(MockCodec.encode("hello"));
    std::thread::scope(|scope| {
        let mut requests = Vec::new();
        for stream in [false, true] {
            for path in ["/api/chat", "/v1/chat/completions", "/v1/messages"] {
                requests.push(scope.spawn(move || {
                    let mut body = json!({"model":"mock-model","messages":[{"role":"user","content":"hi"}],"stream":stream});
                    if path == "/api/chat" {
                        body["options"] = json!({"num_predict":32});
                    } else {
                        body["max_tokens"] = json!(32);
                    }
                    let (status, headers, text) = post(
                        addr,
                        path,
                        body,
                    );
                    assert_eq!(status, 200, "{path}: {text}");
                    if !stream {
                        let value: Value = serde_json::from_str(&text).unwrap();
                        let content = match path {
                            "/api/chat" => &value["message"]["content"],
                            "/v1/chat/completions" => &value["choices"][0]["message"]["content"],
                            _ => &value["content"][0]["text"],
                        };
                        assert_eq!(content, "hello", "{path}: {text}");
                    } else if path == "/api/chat" {
                        assert!(headers.contains("application/x-ndjson"));
                        let values = frames(&text);
                        assert_eq!(values.iter().filter(|v| v["done"] == true).count(), 1);
                        let content: String = values
                            .iter()
                            .filter_map(|v| v["message"]["content"].as_str())
                            .collect();
                        assert_eq!(content, "hello");
                    } else {
                        assert!(headers.contains("text/event-stream"));
                        if path == "/v1/messages" {
                            assert_eq!(text.matches("event: message_stop").count(), 1);
                            assert!(!text.contains("data: [DONE]"));
                        } else {
                            assert_eq!(text.matches("data: [DONE]").count(), 1);
                            assert!(!text.contains("event: message_stop"));
                        }
                    }
                }));
            }
        }
        for request in requests {
            request.join().unwrap();
        }
    });
}
