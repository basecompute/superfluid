//! A fleet head serves every API format the single-node server does, with the same answers:
//! each request here goes to a server on a local engine and to a head whose node runs the
//! same engine, and what each format says back is compared.

use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::fleet_manager::{FleetManager, NodeSpec, PlacementPolicy};
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::registry::ModelRegistry;
use superfluid_daemon::{openai, Daemon, DaemonOptions, EngineHost, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkf::Lease;
use serde_json::Value;

static DIRS: AtomicU64 = AtomicU64::new(0);

fn store(tag: &str) -> SessionStore {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fleet-head-{tag}-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    SessionStore::open(&dir.join("wal.log")).unwrap()
}

/// A small vocabulary, so half of what the mock engine says decodes to text.
fn engine() -> EngineHost {
    EngineHost::spawn(|| (MockEngine::new(EngineConfig { vocab: 512, ..EngineConfig::default() }), None)).unwrap()
}

fn serve(daemon: Daemon) -> String {
    let registry = Arc::new(ModelRegistry::single("mock-model", Arc::new(daemon)));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, openai::ServeConfig::default());
    });
    addr
}

fn local() -> String {
    serve(Daemon::new(store("local"), engine(), Box::new(MockChatCodec), 8))
}

fn node() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(Daemon::new(store("node"), engine(), Box::new(MockCodec), 8));
        let node = NodeAgent::new(daemon, "api-node", "mock-model").with_max_lanes(8);
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

fn head(node: String) -> String {
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let fleet = Arc::new(FleetManager::new(vec![NodeSpec { addr: node, auth: Vec::new() }], PlacementPolicy::LeastLoaded, lease));
    let _ = fleet.probe();
    let daemon = Daemon::over_fleet(store("head"), Box::new(MockChatCodec), fleet, "mock-model".into(), DaemonOptions::default())
        .expect("a daemon over the fleet");
    serve(daemon)
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(20)))
        .http_status_as_error(false)
        .build()
        .new_agent()
}

fn post(addr: &str, path: &str, body: &Value) -> (u16, String) {
    let mut resp = agent()
        .post(&format!("http://{addr}{path}"))
        .header("content-type", "application/json")
        .send(body.to_string().as_bytes())
        .unwrap_or_else(|e| panic!("POST {path}: {e}"));
    (resp.status().as_u16(), resp.body_mut().read_to_string().unwrap())
}

fn json(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

/// The `data:` payloads of an SSE body, as JSON, the end marker left out.
fn events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(json)
        .collect()
}

/// What a format's answer says, without the ids and times that differ between servers.
fn said(path: &str, stream: bool, body: &str) -> Value {
    match (path, stream) {
        ("/v1/chat/completions", false) => {
            let j = json(body);
            serde_json::json!([j["choices"][0]["message"]["content"], j["choices"][0]["finish_reason"], j["usage"]["completion_tokens"]])
        }
        ("/v1/chat/completions", true) => {
            let text: String = events(body).iter().filter_map(|e| e["choices"][0]["delta"]["content"].as_str().map(str::to_string)).collect();
            let fin = events(body).iter().find_map(|e| e["choices"][0]["finish_reason"].as_str().map(str::to_string));
            serde_json::json!([text, fin])
        }
        ("/v1/completions", false) => {
            let j = json(body);
            serde_json::json!([j["choices"][0]["text"], j["choices"][0]["finish_reason"]])
        }
        ("/v1/completions", true) => {
            let text: String = events(body).iter().filter_map(|e| e["choices"][0]["text"].as_str().map(str::to_string)).collect();
            serde_json::json!([text])
        }
        ("/v1/responses", false) => {
            let j = json(body);
            serde_json::json!([j["status"], j["output"].as_array().map(|o| o.iter().map(|i| i["content"].clone()).collect::<Vec<_>>())])
        }
        ("/v1/messages", false) => {
            let j = json(body);
            serde_json::json!([j["content"], j["stop_reason"], j["usage"]["output_tokens"]])
        }
        ("/v1/messages", true) => {
            let text: String = body
                .lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .map(json)
                .filter_map(|e| e["delta"]["text"].as_str().map(str::to_string))
                .collect();
            serde_json::json!([text])
        }
        ("/api/chat", false) => {
            let j = json(body);
            serde_json::json!([j["message"]["content"], j["done_reason"], j["eval_count"]])
        }
        ("/api/generate", false) => {
            let j = json(body);
            serde_json::json!([j["response"], j["done_reason"], j["eval_count"]])
        }
        ("/api/chat", true) | ("/api/generate", true) => {
            let text: String = body
                .lines()
                .filter(|l| !l.is_empty())
                .map(json)
                .filter_map(|e| e["message"]["content"].as_str().or(e["response"].as_str()).map(str::to_string))
                .collect();
            serde_json::json!([text])
        }
        _ => unreachable!("{path}"),
    }
}

#[test]
fn every_api_format_answers_through_a_fleet_head_as_it_does_on_one_machine() {
    let (local, head) = (local(), head(node()));
    let chat = serde_json::json!({"model": "mock-model", "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 12, "temperature": 0});
    let completion = serde_json::json!({"model": "mock-model", "prompt": "once upon a time", "max_tokens": 12, "temperature": 0});
    let responses = serde_json::json!({"model": "mock-model", "input": "hello there", "max_output_tokens": 12, "temperature": 0});
    let messages = serde_json::json!({"model": "mock-model", "max_tokens": 12, "temperature": 0, "messages": [{"role": "user", "content": "hello there"}]});
    let ollama_chat = serde_json::json!({"model": "mock-model", "messages": [{"role": "user", "content": "hello there"}], "options": {"num_predict": 12, "temperature": 0}});
    let ollama_generate = serde_json::json!({"model": "mock-model", "prompt": "once upon a time", "options": {"num_predict": 12, "temperature": 0}});

    let cases: Vec<(&str, &Value)> = vec![
        ("/v1/chat/completions", &chat),
        ("/v1/completions", &completion),
        ("/v1/responses", &responses),
        ("/v1/messages", &messages),
        ("/api/chat", &ollama_chat),
        ("/api/generate", &ollama_generate),
    ];
    for (path, body) in cases {
        for stream in [false, true] {
            if path == "/v1/responses" && stream {
                continue;
            }
            let mut body = body.clone();
            body["stream"] = Value::Bool(stream);
            let (ls, lb) = post(&local, path, &body);
            let (hs, hb) = post(&head, path, &body);
            assert_eq!(ls, 200, "{path} stream={stream} on one machine: {lb}");
            assert_eq!(hs, 200, "{path} stream={stream} through the head: {hb}");
            let (l, h) = (said(path, stream, &lb), said(path, stream, &hb));
            assert_eq!(h, l, "{path} stream={stream}: the head said otherwise");
            // The mock engine's raw stream can say nothing; what both say must still agree.
            if !stream {
                let text = l[0].as_str().map(str::len).unwrap_or_else(|| l[0].to_string().len());
                assert!(text > 4, "{path}: nothing said ({l})");
            }
        }
    }
}

#[test]
fn what_a_node_cannot_do_yet_is_refused_as_a_400() {
    let head = head(node());
    let body = serde_json::json!({"model": "mock-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "presence_penalty": 0.5});
    let (status, text) = post(&head, "/v1/chat/completions", &body);
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("fleet"), "{text}");
}
