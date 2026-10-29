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

/// What differs between two servers that say the same thing: ids, times, durations, and what a
/// server's own prompt cache held.
const VOLATILE: &[&str] = &[
    "id",
    "item_id",
    "created",
    "created_at",
    "completed_at",
    "system_fingerprint",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
    "cached_tokens",
    "prompt_eval_cached_count",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "previous_response_id",
    "sequence_number",
];

fn normalize(v: &mut Value) {
    match v {
        Value::Object(m) => {
            m.retain(|k, _| !VOLATILE.contains(&k.as_str()));
            m.values_mut().for_each(normalize);
        }
        Value::Array(a) => a.iter_mut().for_each(normalize),
        _ => {}
    }
}

/// The fields a stream's events carry text (or logprobs) in, a piece each: where a stream cuts
/// them is how fast tokens came, not what was said.
const TEXT: &[&str] = &["content", "text", "response", "thinking", "reasoning_content", "delta", "partial_json", "arguments"];

/// Folds a stream's events into one value: text fields joined, every other field as it last was.
fn fold(into: &mut Value, event: &Value) {
    let (Value::Object(into), Value::Object(event)) = (into, event) else { return };
    for (k, v) in event {
        let had = into.contains_key(k);
        match (into.get_mut(k), v) {
            (Some(Value::String(a)), Value::String(b)) if TEXT.contains(&k.as_str()) => a.push_str(b),
            (Some(a @ Value::Object(_)), Value::Object(_)) => fold(a, v),
            (Some(Value::Array(a)), Value::Array(b)) if TEXT.contains(&k.as_str()) => a.extend(b.iter().cloned()),
            (Some(Value::Array(a)), Value::Array(b)) => {
                for (i, item) in b.iter().enumerate() {
                    match a.get_mut(i) {
                        Some(slot @ Value::Object(_)) => fold(slot, item),
                        Some(slot) => *slot = item.clone(),
                        None => a.push(item.clone()),
                    }
                }
            }
            (_, Value::Null) if had => {}
            _ => {
                into.insert(k.clone(), v.clone());
            }
        }
    }
}

/// A body as JSON without what differs between servers: one value, or a stream's events folded.
fn said(body: &str) -> Value {
    let lines: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
    let events: Vec<Value> = if lines.iter().any(|l| l.starts_with("data: ")) {
        lines.iter().filter_map(|l| l.strip_prefix("data: ")).filter(|d| *d != "[DONE]").map(json).collect()
    } else if lines.len() > 1 {
        lines.iter().map(|l| json(l)).collect()
    } else {
        vec![json(body)]
    };
    let mut v = serde_json::json!({});
    for e in &events {
        fold(&mut v, e);
    }
    normalize(&mut v);
    v
}

/// Posts `body` to both servers and asserts they answer alike.
fn same(local: &str, head: &str, path: &str, body: &Value) -> Value {
    let (ls, lb) = post(local, path, body);
    let (hs, hb) = post(head, path, body);
    assert_eq!(hs, ls, "{path} {body}: status on one machine {ls} ({lb}), through the head {hs} ({hb})");
    let (l, h) = (said(&lb), said(&hb));
    assert_eq!(h, l, "{path} {body}: the head said otherwise");
    l
}

#[test]
fn every_api_format_answers_through_a_fleet_head_as_it_does_on_one_machine() {
    let (local, head) = (local(), head(node()));
    let user = serde_json::json!([{"role": "user", "content": "hello there"}]);
    let cases: Vec<(&str, Value)> = vec![
        ("/v1/chat/completions", serde_json::json!({"model": "mock-model", "messages": user, "max_tokens": 12, "temperature": 0})),
        ("/v1/completions", serde_json::json!({"model": "mock-model", "prompt": "once upon a time", "max_tokens": 12, "temperature": 0})),
        ("/v1/responses", serde_json::json!({"model": "mock-model", "input": "hello there", "max_output_tokens": 12, "temperature": 0})),
        ("/v1/messages", serde_json::json!({"model": "mock-model", "max_tokens": 12, "temperature": 0, "messages": user})),
        ("/api/chat", serde_json::json!({"model": "mock-model", "messages": user, "options": {"num_predict": 12, "temperature": 0}})),
        ("/api/generate", serde_json::json!({"model": "mock-model", "prompt": "once upon a time", "options": {"num_predict": 12, "temperature": 0}})),
    ];
    for (path, body) in cases {
        for stream in [false, true] {
            let mut body = body.clone();
            body["stream"] = Value::Bool(stream);
            let said = same(&local, &head, path, &body);
            assert!(said.to_string().len() > 40, "{path} stream={stream}: nothing said ({said})");
        }
    }
}

#[test]
fn what_a_request_asks_beyond_sampling_reaches_the_node() {
    let (local, head) = (local(), head(node()));
    let chat = |extra: Value| {
        let mut body = serde_json::json!({"model": "mock-model", "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 12, "temperature": 0});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        body
    };
    for extra in [
        serde_json::json!({"presence_penalty": 0.5, "frequency_penalty": 0.25}),
        serde_json::json!({"repeat_penalty": 1.3}),
        serde_json::json!({"ignore_eos": true}),
        serde_json::json!({"logprobs": true, "top_logprobs": 2}),
        serde_json::json!({"logit_bias": {"300": 5, "301": -100}}),
        serde_json::json!({"response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}}}),
    ] {
        for stream in [false, true] {
            let mut body = chat(extra.clone());
            body["stream"] = Value::Bool(stream);
            same(&local, &head, "/v1/chat/completions", &body);
        }
    }
}

#[test]
fn embeddings_and_lora_reach_a_node() {
    let (local, head) = (local(), head(node()));
    let embedded = same(&local, &head, "/v1/embeddings", &serde_json::json!({"model": "mock-model", "input": ["hello there", "general kenobi"]}));
    same(&local, &head, "/api/embed", &serde_json::json!({"model": "mock-model", "input": "hello there"}));
    assert!(embedded["data"][1]["embedding"].as_array().is_some_and(|e| !e.is_empty()), "{embedded}");
    let lora = same(&local, &head, "/v1/lora/load", &serde_json::json!({"model": "mock-model", "path": "/nonexistent/adapter.gguf"}));
    assert_eq!(lora["loaded"], true, "{lora}");
}

/// An engine that takes images, four tokens each.
fn media_engine() -> EngineHost {
    EngineHost::spawn(|| {
        let cfg = EngineConfig { vocab: 512, image_token_id: 999, media_tokens_per_image: 4, ..EngineConfig::default() };
        (MockEngine::new(cfg), None)
    })
    .unwrap()
}

fn media_options(dir: &std::path::Path) -> DaemonOptions {
    DaemonOptions { media_dir: Some(dir.join("media")), ..DaemonOptions::default() }
}

fn temp(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("superfluid-fleet-head-{tag}-{}-{}", std::process::id(), DIRS.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A head and a local server whose engines take images; the node's media pool is returned.
fn media_pair() -> (String, String, std::path::PathBuf) {
    let local_dir = temp("local-media");
    let local = serve(Daemon::with_options(store("local"), media_engine(), Box::new(MockChatCodec), media_options(&local_dir)).unwrap());
    let node_dir = temp("node-media");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let node_addr = listener.local_addr().unwrap().to_string();
    let opts = media_options(&node_dir);
    std::thread::spawn(move || {
        let daemon = Arc::new(Daemon::with_options(store("node"), media_engine(), Box::new(MockCodec), opts).unwrap());
        let node = NodeAgent::new(daemon, "media-node", "mock-model").with_max_lanes(8);
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    let head_dir = temp("head-media");
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let fleet = Arc::new(FleetManager::new(vec![NodeSpec { addr: node_addr, auth: Vec::new() }], PlacementPolicy::LeastLoaded, lease));
    let _ = fleet.probe();
    let head = serve(Daemon::over_fleet(store("head"), Box::new(MockChatCodec), fleet, "mock-model".into(), media_options(&head_dir)).unwrap());
    (local, head, node_dir.join("media"))
}

const IMAGE: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

#[test]
fn an_image_reaches_the_node_and_is_answered_as_on_one_machine() {
    let (local, head, node_media) = media_pair();
    for stream in [false, true] {
        let body = serde_json::json!({"model": "mock-model", "max_tokens": 8, "temperature": 0, "stream": stream, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": IMAGE}},
        ]}]});
        let said = same(&local, &head, "/v1/chat/completions", &body);
        assert!(said.to_string().contains("prompt_tokens"), "{said}");
    }
    let blobs: Vec<String> = std::fs::read_dir(&node_media)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    assert!(
        blobs.iter().any(|n| superfluid_daemon::media::MediaPool::is_valid_id(n)),
        "the image is in the node's media pool by its hash: {blobs:?}"
    );
}

fn post_multipart(addr: &str, path: &str, fields: &[(&str, &str)], file: &[u8]) -> (u16, String) {
    let boundary = "FLEETBND";
    let mut body: Vec<u8> = Vec::new();
    for (k, v) in fields {
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
    }
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let mut resp = agent()
        .post(&format!("http://{addr}{path}"))
        .header("content-type", &format!("multipart/form-data; boundary={boundary}"))
        .send(&body[..])
        .unwrap();
    (resp.status().as_u16(), resp.body_mut().read_to_string().unwrap())
}

#[test]
fn a_transcription_runs_on_a_node_as_on_one_machine() {
    let (local, head) = (local(), head(node()));
    for (format, stream) in [("json", "false"), ("verbose_json", "false"), ("json", "true")] {
        let fields = [("model", "mock-model"), ("response_format", format), ("stream", stream)];
        let (ls, lb) = post_multipart(&local, "/v1/audio/transcriptions", &fields, b"RIFF....WAVEfake");
        let (hs, hb) = post_multipart(&head, "/v1/audio/transcriptions", &fields, b"RIFF....WAVEfake");
        assert_eq!((hs, ls), (200, 200), "one machine: {lb}; the head: {hb}");
        assert_eq!(said(&hb), said(&lb), "{format} stream={stream}");
        assert!(lb.contains("mock transcription"), "{lb}");
    }
}

#[test]
fn every_route_answers_through_the_head_with_the_status_it_has_on_one_machine() {
    let (local, head) = (local(), head(node()));
    let mut differ = Vec::new();
    let get = |addr: &str, path: &str| {
        let mut r = agent().get(&format!("http://{addr}{path}")).call().unwrap();
        (r.status().as_u16(), r.body_mut().read_to_string().unwrap())
    };
    let gets = ["/health", "/metrics", "/props", "/slots", "/v1/models", "/v1/models/mock-model", "/v1/lora", "/v1/batches", "/v1/files", "/api/tags", "/api/ps", "/api/version", "/"];
    for p in gets {
        let (l, lb) = get(&local, p);
        let (h, hb) = get(&head, p);
        if l != h {
            differ.push(format!("GET {p}: one machine {l} ({lb}), the head {h} ({hb})"));
        }
    }
    let posts: Vec<(&str, Value)> = vec![
        ("/v1/tokenize", serde_json::json!({"model": "mock-model", "content": "hello"})),
        ("/v1/rerank", serde_json::json!({"model": "mock-model", "query": "q", "documents": ["a", "b"]})),
        ("/api/show", serde_json::json!({"model": "mock-model"})),
        ("/api/embeddings", serde_json::json!({"model": "mock-model", "prompt": "hi"})),
        ("/slots/0/save", serde_json::json!({"filename": "a.bin"})),
        ("/slots/0/restore", serde_json::json!({"filename": "a.bin"})),
        ("/v1/lora/unload", serde_json::json!({"model": "mock-model"})),
        ("/v1/models/load", serde_json::json!({"id": "other"})),
        ("/v1/models/unload", serde_json::json!({"id": "mock-model"})),
        ("/api/pull", serde_json::json!({"model": "x", "stream": false})),
        ("/api/copy", serde_json::json!({"source": "mock-model", "destination": "y"})),
        ("/api/delete", serde_json::json!({"model": "y"})),
    ];
    for (p, body) in posts {
        let (l, lb) = post(&local, p, &body);
        let (h, hb) = post(&head, p, &body);
        if l != h {
            differ.push(format!("POST {p}: one machine {l} ({lb}), the head {h} ({hb})"));
        }
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// A node serving `model` under `identity`, telling the head its tokenizer by `tokenizer`.
fn named_node(identity: &'static str, model: &'static str, tokenizer: u64) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(Daemon::new(store("node"), engine(), Box::new(MockCodec), 8));
        let node = NodeAgent::new(daemon, identity, model).with_max_lanes(8).with_tokenizer(tokenizer);
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

/// A head serving each of `models` over one fleet of `nodes`, with the tokenizer digests given.
fn multi_head(nodes: Vec<String>, models: &[(&str, u64)]) -> (String, Arc<FleetManager>) {
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let specs = nodes.into_iter().map(|addr| NodeSpec { addr, auth: Vec::new() }).collect();
    let fleet = Arc::new(FleetManager::new(specs, PlacementPolicy::LeastLoaded, lease));
    for (model, digest) in models {
        fleet.expect_tokenizer(model, *digest);
    }
    let _ = fleet.probe();
    let mut registry: Option<Arc<ModelRegistry>> = None;
    for (model, _) in models {
        let d = Arc::new(
            Daemon::over_fleet(store("head"), Box::new(MockChatCodec), Arc::clone(&fleet), model.to_string(), DaemonOptions::default()).unwrap(),
        );
        match &registry {
            None => registry = Some(Arc::new(ModelRegistry::single(model.to_string(), d))),
            Some(r) => r.insert_loaded(model, d),
        }
    }
    let registry = registry.unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, openai::ServeConfig::default());
    });
    (addr, fleet)
}

fn chat_to(addr: &str, model: &str) -> (u16, String) {
    let body = serde_json::json!({"model": model, "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 6, "temperature": 0});
    post(addr, "/v1/chat/completions", &body)
}

/// Tokens each node has generated for the head, by node name.
fn decoded(fleet: &FleetManager) -> std::collections::HashMap<String, u64> {
    fleet.status().into_iter().map(|n| (n.name, n.decode_tokens)).collect()
}

#[test]
fn two_models_on_two_nodes_are_routed_by_model() {
    let (a, b) = (named_node("alpha-node", "alpha", 7), named_node("beta-node", "beta", 7));
    let (head, fleet) = multi_head(vec![a, b], &[("alpha", 7), ("beta", 7)]);

    let mut resp = agent().get(&format!("http://{head}/v1/models")).call().unwrap();
    let listed: Value = serde_json::from_str(&resp.body_mut().read_to_string().unwrap()).unwrap();
    let ids: Vec<&str> = listed["data"].as_array().unwrap().iter().filter_map(|m| m["id"].as_str()).collect();
    assert!(ids.contains(&"alpha") && ids.contains(&"beta"), "{listed}");

    for _ in 0..3 {
        let (status, body) = chat_to(&head, "alpha");
        assert_eq!(status, 200, "{body}");
    }
    let after_alpha = decoded(&fleet);
    assert!(after_alpha["alpha-node"] > 0 && after_alpha["beta-node"] == 0, "{after_alpha:?}");
    let (status, body) = chat_to(&head, "beta");
    assert_eq!(status, 200, "{body}");
    let after_beta = decoded(&fleet);
    assert_eq!(after_beta["alpha-node"], after_alpha["alpha-node"], "beta went to the alpha node: {after_beta:?}");
    assert!(after_beta["beta-node"] > 0, "{after_beta:?}");
}

#[test]
fn a_node_whose_tokenizer_differs_is_not_used_for_the_model() {
    let (same, other, unknown) = (named_node("same", "alpha", 7), named_node("other", "alpha", 8), named_node("silent", "beta", 0));
    let (head, fleet) = multi_head(vec![other.clone(), same], &[("alpha", 7), ("beta", 9)]);
    for _ in 0..4 {
        let (status, body) = chat_to(&head, "alpha");
        assert_eq!(status, 200, "{body}");
    }
    let by_node = decoded(&fleet);
    assert!(by_node["same"] > 0 && by_node["other"] == 0, "the node with another tokenizer took work: {by_node:?}");

    // A node that does not say what it tokenizes with is taken at its word.
    let (head, _) = multi_head(vec![unknown], &[("beta", 9)]);
    let (status, body) = chat_to(&head, "beta");
    assert_eq!(status, 200, "{body}");
}

#[test]
fn a_request_sent_again_with_its_key_through_the_head_gets_the_first_answer() {
    let head = head(node());
    let body = serde_json::json!({"model": "mock-model", "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 12, "temperature": 1.0});
    let send = || {
        let mut resp = agent()
            .post(&format!("http://{head}/v1/chat/completions"))
            .header("content-type", "application/json")
            .header("Idempotency-Key", "through-the-head")
            .send(body.to_string().as_bytes())
            .unwrap();
        let replayed = resp.headers().get("idempotent-replayed").is_some();
        (replayed, resp.body_mut().read_to_string().unwrap())
    };
    let (first_replayed, first) = send();
    let (again_replayed, again) = send();
    assert_eq!((first_replayed, again_replayed), (false, true));
    assert_eq!(again, first);
}
