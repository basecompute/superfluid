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
use superfluid_proto::handshake::BATCH_INVARIANT;
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
    serve_with(Arc::new(daemon), openai::ServeConfig::default())
}

fn serve_with(daemon: Arc<Daemon>, cfg: openai::ServeConfig) -> String {
    let registry = Arc::new(ModelRegistry::single("mock-model", daemon));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    addr
}

fn local() -> String {
    serve(Daemon::new(store("local"), engine(), Box::new(MockChatCodec), 8))
}

fn node() -> String {
    node_serving(Daemon::new(store("node"), engine(), Box::new(MockCodec), 8))
}

fn node_serving(daemon: Daemon) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(daemon);
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
    serve(head_daemon(node))
}

fn head_daemon(node: String) -> Daemon {
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let fleet = Arc::new(FleetManager::new(vec![NodeSpec { addr: node, auth: Vec::new() }], PlacementPolicy::LeastLoaded, lease));
    let _ = fleet.probe();
    Daemon::over_fleet(store("head"), Box::new(MockChatCodec), fleet, "mock-model".into(), DaemonOptions::default())
        .expect("a daemon over the fleet")
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

fn send_multipart(addr: &str, path: &str, fields: &[(&str, &str)], file: &[u8]) -> ureq::http::Response<ureq::Body> {
    let boundary = "FLEETBND";
    let mut body: Vec<u8> = Vec::new();
    for (k, v) in fields {
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
    }
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    agent()
        .post(&format!("http://{addr}{path}"))
        .header("content-type", &format!("multipart/form-data; boundary={boundary}"))
        .send(&body[..])
        .unwrap()
}

fn post_multipart(addr: &str, path: &str, fields: &[(&str, &str)], file: &[u8]) -> (u16, String) {
    let mut resp = send_multipart(addr, path, fields, file);
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
fn a_streamed_transcription_tells_each_segment_as_the_node_has_it() {
    // The mock engine takes a moment for each word, so 400 of them run for most of a second.
    let words = vec!["word"; 400].join(" ");
    let head = head(node());
    let fields = [("model", "mock-model"), ("stream", "true"), ("prompt", words.as_str())];
    let mut resp = send_multipart(&head, "/v1/audio/transcriptions", &fields, b"RIFF....WAVEfake");
    assert_eq!(resp.status().as_u16(), 200);
    let start = std::time::Instant::now();
    let (mut first, mut done, mut deltas) = (None, None, 0);
    for line in std::io::BufRead::lines(std::io::BufReader::new(resp.body_mut().as_reader())) {
        let line = line.unwrap();
        if line.contains("transcript.text.delta") {
            deltas += 1;
            first.get_or_insert(start.elapsed());
        } else if line.contains("transcript.text.done") {
            done = Some(start.elapsed());
        }
    }
    let (first, done) = (first.expect("deltas"), done.expect("a done event"));
    assert_eq!(deltas, 402, "a delta for each word");
    assert!(done - first > std::time::Duration::from_millis(300), "the first delta came at {first:?}, the end at {done:?}");
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
fn health_counts_every_node_and_each_models_own() {
    let gone = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string();
    let (head, _) = multi_head(vec![named_node("alpha-node", "alpha", 7), gone], &[("alpha", 7), ("beta", 7)]);
    let mut resp = agent().get(&format!("http://{head}/health")).call().unwrap();
    let health = json(&resp.body_mut().read_to_string().unwrap());
    assert_eq!(resp.status().as_u16(), 200, "a model still served keeps the head in: {health}");
    assert_eq!((&health["nodes_up"], &health["nodes_known"]), (&1.into(), &2.into()), "{health}");
    assert_eq!((&health["models"]["alpha"]["nodes_up"], &health["models"]["beta"]["nodes_up"]), (&1.into(), &0.into()), "{health}");
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

/// A byte-level tokenizer with ChatML's and Qwen's tool-call markers, for a head to take from its
/// node.
struct Bytes;

impl superfluid_engine::Tokenizer for Bytes {
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(r) = rest.strip_prefix("<|im_start|>") {
                out.push(256);
                rest = r;
            } else if let Some(r) = rest.strip_prefix("<|im_end|>") {
                out.push(257);
                rest = r;
            } else if let Some(r) = rest.strip_prefix("<tool_call>") {
                out.push(258);
                rest = r;
            } else if let Some(r) = rest.strip_prefix("</tool_call>") {
                out.push(259);
                rest = r;
            } else {
                let c = rest.chars().next().unwrap();
                out.extend(c.to_string().bytes().map(u32::from));
                rest = &rest[c.len_utf8()..];
            }
        }
        out
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        match token {
            0..=255 => vec![token as u8],
            256 => b"<|im_start|>".to_vec(),
            257 => b"<|im_end|>".to_vec(),
            258 => b"<tool_call>".to_vec(),
            259 => b"</tool_call>".to_vec(),
            _ => Vec::new(),
        }
    }
    fn vocab_size(&self) -> u32 {
        512
    }
    fn special_tokens(&self) -> Vec<(String, u32)> {
        vec![
            ("<|im_start|>".into(), 256),
            ("<|im_end|>".into(), 257),
            ("<tool_call>".into(), 258),
            ("</tool_call>".into(), 259),
        ]
    }
    fn bos_token(&self) -> Option<u32> {
        None
    }
    fn eos_token(&self) -> u32 {
        257
    }
    fn chat_template_jinja(&self) -> String {
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}".into()
    }
}

/// A node that shares its tokenizer, with `lanes` lanes, its engine taking `tick` a step.
fn sharing_node(tok: Arc<dyn superfluid_engine::Tokenizer>, lanes: u32, tick: std::time::Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let host = EngineHost::spawn(move || (MockEngine::new(EngineConfig { vocab: 512, tick_delay: tick, ..EngineConfig::default() }), None)).unwrap();
        let daemon = Arc::new(Daemon::new(store("node"), host, Box::new(MockCodec), lanes as usize));
        let node = NodeAgent::new(daemon, "sharer", "mock-model").with_max_lanes(lanes).with_tokenizer_of(tok);
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

fn fleet_of(node: String) -> Arc<FleetManager> {
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let fleet = Arc::new(FleetManager::new(vec![NodeSpec { addr: node, auth: Vec::new() }], PlacementPolicy::LeastLoaded, lease));
    let _ = fleet.probe();
    fleet
}

#[test]
fn a_head_with_no_copy_of_the_model_takes_its_tokenizer_from_a_node() {
    use superfluid_engine::Tokenizer;
    let tok: Arc<dyn Tokenizer> = Arc::new(Bytes);
    let fleet = fleet_of(sharing_node(Arc::clone(&tok), 8, std::time::Duration::ZERO));
    let remote = fleet.remote_tokenizer("mock-model").expect("the node shares its tokenizer");
    assert_eq!(remote.digest(), superfluid_daemon::fleet::tokenizer_digest(&*tok));
    for text in ["hello there", "<|im_start|>user\nhi<|im_end|>", "naïve ✓"] {
        assert_eq!(remote.encode(text), tok.encode(text), "{text}");
    }
    assert_eq!((remote.vocab_size(), remote.eos_token(), remote.special_tokens()), (512, 257, tok.special_tokens()));
    assert_eq!(remote.chat_template_jinja(), tok.chat_template_jinja());

    // A head built on what it took answers as one machine with the tokenizer itself does.
    let codec = |t: Arc<dyn Tokenizer>| superfluid_daemon::codec::auto_codec_from_tokenizer(t).0;
    let local = serve(Daemon::new(store("local"), engine(), codec(Arc::clone(&tok)), 8));
    let head = serve(
        Daemon::over_fleet(store("head"), codec(Arc::new(remote)), fleet, "mock-model".into(), DaemonOptions::default()).unwrap(),
    );
    for stream in [false, true] {
        let body = serde_json::json!({"model": "mock-model", "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 12, "temperature": 0, "stream": stream});
        let said = same(&local, &head, "/v1/chat/completions", &body);
        assert!(said.to_string().contains("completion_tokens"), "{said}");
    }
}

#[test]
fn requests_past_every_lane_wait_their_turn_on_a_head_that_encodes_on_a_node() {
    // Every lane's connection is held by a generation that commits as it goes, while more
    // sessions start; each one's turn opener is encoded on the node, which must not happen
    // under a lock those generations need.
    use superfluid_daemon::{scheduler::GenExtras, wal::role, GenParams};
    let fleet = fleet_of(sharing_node(Arc::new(Bytes), 2, std::time::Duration::from_millis(50)));
    let remote = fleet.remote_tokenizer("mock-model").unwrap();
    // The curated ChatML codec, as Qwen's models get: it renders each message alone and the
    // turn's opener when the turn is generated.
    let codec = Box::new(superfluid_daemon::codec::ChatMlCodec::from_tokenizer(Arc::new(remote)).unwrap());
    let head = Arc::new(Daemon::over_fleet(store("head"), codec, fleet, "mock-model".into(), DaemonOptions::default()).unwrap());
    let turn = |text: &str| {
        let s = head.create(None, GenParams::default()).unwrap();
        head.append_message(s, role::USER, text.into()).unwrap();
        s
    };
    let (started, first) = std::sync::mpsc::channel();
    let long: Vec<_> = (0..2)
        .map(|i| {
            let (head, started, s) = (Arc::clone(&head), started.clone(), turn(&format!("long {i}")));
            std::thread::spawn(move || {
                let extras = GenExtras { ignore_eos: true, ..GenExtras::default() };
                let mut told = false;
                head.generate_streaming_ex(s, 1000, extras, |_| {
                    if !std::mem::replace(&mut told, true) {
                        let _ = started.send(());
                    }
                    Ok(())
                })
                .unwrap()
            })
        })
        .collect();
    for _ in 0..2 {
        first.recv_timeout(std::time::Duration::from_secs(20)).expect("both lanes generating");
    }
    let (done, finished) = std::sync::mpsc::channel();
    for i in 0..3 {
        let (head, done) = (Arc::clone(&head), done.clone());
        std::thread::spawn(move || {
            let s = head.create(None, GenParams::default()).unwrap();
            head.append_message(s, role::USER, format!("short {i}")).unwrap();
            let _ = done.send(head.generate(s, 10).map(|o| o.tokens_generated));
        });
    }
    for _ in 0..3 {
        let generated = finished.recv_timeout(std::time::Duration::from_secs(20)).expect("a session past every lane is answered");
        assert!(generated.unwrap() > 0);
    }
    for run in long {
        assert!(run.join().unwrap().tokens_generated > 0);
    }
}

#[test]
fn a_node_runs_a_generation_in_the_qos_class_its_head_gave_the_session() {
    // The node caps its agent class at one lane: two agent sessions take turns there, where in
    // another class they would run side by side.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let config = EngineConfig { vocab: 512, tick_delay: std::time::Duration::from_millis(20), ..EngineConfig::default() };
        let host = EngineHost::spawn(move || (MockEngine::new(config), None)).unwrap();
        let opts = DaemonOptions { max_lanes: 2, class_lanes: [0, 0, 1, 0], ..DaemonOptions::default() };
        let daemon = Arc::new(Daemon::with_options(store("node"), host, Box::new(MockCodec), opts).unwrap());
        let node = NodeAgent::new(daemon, "classes", "mock-model").with_max_lanes(2);
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    let fleet = fleet_of(addr);
    let agent = superfluid_daemon::qos::FOREGROUND_AGENT;
    for s in 1..=2u64 {
        fleet.place(s, superfluid_daemon::GenParams::default(), agent, "raw", "mock-model", &[1, 2, 3], 64).unwrap();
    }
    let runs: Vec<_> = (1..=2u64)
        .map(|s| {
            let fleet = Arc::clone(&fleet);
            std::thread::spawn(move || {
                let mut first = None;
                fleet
                    .generate_streaming(s, 64, 60_000, |_| {
                        first.get_or_insert_with(std::time::Instant::now);
                    })
                    .unwrap();
                (first.expect("tokens"), std::time::Instant::now())
            })
        })
        .collect();
    let mut ran: Vec<_> = runs.into_iter().map(|r| r.join().unwrap()).collect();
    ran.sort();
    assert!(ran[1].0 >= ran[0].1, "two agent sessions ran at once on a node that gives the class one lane");
}

#[test]
fn a_head_with_no_node_up_answers_503_in_every_format_streamed_or_not() {
    // A node address nothing listens on: the head knows a node, none is up.
    let gone = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string();
    let head = head(gone);
    let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
    for stream in [false, true] {
        for (path, body) in [
            ("/v1/chat/completions", serde_json::json!({"model": "mock-model", "messages": messages, "max_tokens": 4, "stream": stream})),
            ("/v1/messages", serde_json::json!({"model": "mock-model", "messages": messages, "max_tokens": 4, "stream": stream})),
            ("/api/chat", serde_json::json!({"model": "mock-model", "messages": messages, "options": {"num_predict": 4}, "stream": stream})),
            ("/v1/completions", serde_json::json!({"model": "mock-model", "prompt": "hi", "max_tokens": 4, "stream": stream})),
        ] {
            let (status, said) = post(&head, path, &body);
            assert_eq!(status, 503, "{path} stream={stream}: {said}");
            assert!(said.contains("no node"), "{path}: {said}");
        }
    }
}

#[test]
fn a_completion_whose_deadline_passes_while_it_waits_for_a_connection_expires_unstarted() {
    // The node's one lane is held by a long generation; a completion with a short deadline
    // waits for its connection, and the deadline passes while it waits.
    use std::ops::ControlFlow;
    use std::sync::atomic::AtomicBool;
    use superfluid_daemon::{scheduler::GenExtras, GenParams};
    let fleet = fleet_of(sharing_node(Arc::new(Bytes), 1, std::time::Duration::from_millis(50)));
    let head = Arc::new(Daemon::over_fleet(store("head"), Box::new(MockChatCodec), fleet, "mock-model".into(), DaemonOptions::default()).unwrap());
    let (stop, long_done) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let (started, first) = std::sync::mpsc::channel();
    let long = {
        let (head, stop, long_done) = (Arc::clone(&head), Arc::clone(&stop), Arc::clone(&long_done));
        std::thread::spawn(move || {
            let extras = GenExtras { ignore_eos: true, ..GenExtras::default() };
            let r = head.generate_tokens_streaming(vec![1, 2, 3], GenParams::default(), 10_000, 0, extras, |_| {
                let _ = started.send(());
                if stop.load(Ordering::Relaxed) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            });
            long_done.store(true, Ordering::Relaxed);
            r
        })
    };
    first.recv_timeout(std::time::Duration::from_secs(20)).expect("the long generation holds the lane");
    let asked = std::time::Instant::now();
    let (tokens, expired) = head.generate_tokens(vec![4, 5, 6], GenParams::default(), 8, 50).unwrap();
    let (took, waited_out) = (asked.elapsed(), long_done.load(Ordering::Relaxed));
    stop.store(true, Ordering::Relaxed);
    assert_eq!((tokens.len(), expired), (0, true), "a completion past its deadline is not started (answered after {took:?})");
    assert!(!waited_out, "the expired completion waited out the generation holding the lane ({took:?})");
    long.join().unwrap().unwrap();
}

#[test]
fn a_placed_completion_whose_connection_is_taken_past_its_deadline_expires_unstarted() {
    // Placed in time; then another session placed on the node's one connection takes it before
    // this one's generation does, and holds it past the deadline.
    use std::sync::atomic::AtomicBool;
    use superfluid_daemon::GenParams;
    // One connection to the node, so both sessions are placed on it.
    let node = NodeSpec { addr: sharing_node(Arc::new(Bytes), 1, std::time::Duration::from_millis(50)), auth: Vec::new() };
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let fleet = Arc::new(FleetManager::with_options(vec![node], PlacementPolicy::LeastLoaded, lease, 90, 1));
    let _ = fleet.probe();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(1_000);
    let placed = fleet.place_by(1, GenParams::default(), 0, 0, "raw", "mock-model", &[1, 2, 3], 8, Some(deadline)).unwrap();
    assert!(placed.is_some(), "placed before its deadline");
    fleet.place(2, GenParams::default(), 0, "raw", "mock-model", &[4, 5, 6], 20_000).unwrap();
    let long_done = Arc::new(AtomicBool::new(false));
    let (started, first) = std::sync::mpsc::channel();
    let long = {
        let (fleet, long_done) = (Arc::clone(&fleet), Arc::clone(&long_done));
        std::thread::spawn(move || {
            // Well past the deadline: cancelled once the other call answers.
            let r = fleet.generate_streaming(2, 20_000, 60_000, |_| {
                let _ = started.send(());
            });
            long_done.store(true, Ordering::Relaxed);
            r
        })
    };
    first.recv_timeout(std::time::Duration::from_secs(20)).expect("the other session holds the connection");
    // What the dispatcher has once placed: the budget left then, and the deadline itself.
    let left = deadline.saturating_duration_since(std::time::Instant::now()).as_millis().max(1) as u64;
    let g = fleet.generate_streaming_by(1, 8, left, Some(deadline), None, &mut |_| {}, |_| {}).unwrap();
    let waited_out = long_done.load(Ordering::Relaxed);
    let _ = fleet.cancel(2);
    assert_eq!((g.tokens.len(), g.expired), (0, true), "a generation whose deadline passed while it waited is not started");
    assert!(!waited_out, "the expired generation waited out the one holding its connection");
    long.join().unwrap().unwrap();
    fleet.finish(1);
    fleet.finish(2);
}

/// A node on the mock engine with `lanes` lanes and `class_lanes` caps, ticking every `tick`:
/// its address, and its scheduler's counters. An `old` node does not state what a node from
/// before batch-invariant lanes does not.
fn qos_node(
    identity: &'static str,
    lanes: u32,
    class_lanes: [usize; 4],
    tick: std::time::Duration,
    old: bool,
) -> (String, Arc<superfluid_daemon::scheduler::SchedStats>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let config = EngineConfig { vocab: 512, tick_delay: tick, ..EngineConfig::default() };
    let host = EngineHost::spawn(move || (MockEngine::new(config), None)).unwrap();
    let opts = DaemonOptions { max_lanes: lanes as usize, class_lanes, ..DaemonOptions::default() };
    let daemon = Arc::new(Daemon::with_options(store("node"), host, Box::new(MockCodec), opts).unwrap());
    let stats = daemon.sched_stats();
    let mut node = NodeAgent::new(daemon, identity, "mock-model").with_max_lanes(lanes);
    if old {
        node = node.without_feature(BATCH_INVARIANT);
    }
    std::thread::spawn(move || {
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    (addr, stats)
}

/// The options of a generation that asks only to run alone.
fn alone() -> superfluid_proto::linkf::GenerateOptsMsg {
    superfluid_proto::linkf::GenerateOptsMsg {
        session_id: 0,
        generation_id: 0,
        presence_penalty_bits: 0,
        frequency_penalty_bits: 0,
        repeat_penalty_bits: superfluid_daemon::scheduler::GenExtras::default().repeat_penalty.to_bits(),
        ignore_eos: false,
        top_logprobs: None,
        grammar: None,
        logit_bias: None,
        media: Vec::new(),
        batch_invariant: true,
    }
}

#[test]
fn a_batch_invariant_generation_runs_alone_on_its_node() {
    // Two lanes, and a generation asked to run alone: one beside it waits until it ends, as on a
    // single server.
    let (addr, _) = qos_node("alone", 2, [0; 4], std::time::Duration::from_millis(20), false);
    let fleet = fleet_of(addr);
    let agent = superfluid_daemon::qos::FOREGROUND_AGENT;
    fleet.place_by(1, superfluid_daemon::GenParams::default(), agent, BATCH_INVARIANT, "raw", "mock-model", &[1, 2, 3], 600, None).unwrap();
    fleet.place(2, superfluid_daemon::GenParams::default(), agent, "raw", "mock-model", &[4, 5, 6], 48).unwrap();
    let (started, first) = std::sync::mpsc::channel();
    let alone_run = {
        let fleet = Arc::clone(&fleet);
        std::thread::spawn(move || {
            let opts = alone();
            let ask = superfluid_daemon::fleet::Ask { opts: &opts, blobs: &[] };
            fleet
                .generate_streaming_by(1, 600, 60_000, None, Some(&ask), &mut |_| {}, |_| {
                    let _ = started.send(());
                })
                .unwrap();
            std::time::Instant::now()
        })
    };
    first.recv_timeout(std::time::Duration::from_secs(20)).expect("the batch-invariant generation runs");
    let mut beside = None;
    fleet
        .generate_streaming(2, 8, 60_000, |_| {
            beside.get_or_insert_with(std::time::Instant::now);
        })
        .unwrap();
    let ended = alone_run.join().unwrap();
    assert!(beside.expect("tokens") >= ended, "a generation ran beside one that asked to run alone");
}

#[test]
fn a_batch_invariant_lane_goes_to_a_node_that_can_run_it_alone() {
    // An old node and a new one: round robin takes them in turn, but a batch-invariant session
    // only ever lands on the new one.
    use superfluid_daemon::{qos, GenParams};
    let (old, _) = qos_node("old", 2, [0; 4], std::time::Duration::ZERO, true);
    let (new, _) = qos_node("new", 2, [0; 4], std::time::Duration::ZERO, false);
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let specs = vec![NodeSpec { addr: old, auth: Vec::new() }, NodeSpec { addr: new, auth: Vec::new() }];
    let fleet = FleetManager::new(specs, PlacementPolicy::RoundRobin, lease);
    let _ = fleet.probe();
    for s in 1..=4u64 {
        fleet.place_by(s, GenParams::default(), qos::FOREGROUND_AGENT, BATCH_INVARIANT, "raw", "mock-model", &[1, 2, 3], 8, None).unwrap();
        assert_eq!(fleet.node_of(s).as_deref(), Some("new"), "session {s}");
        fleet.finish(s);
    }
    let mut plain: Vec<_> = (5..=8u64)
        .map(|s| {
            fleet.place(s, GenParams::default(), qos::FOREGROUND_AGENT, "raw", "mock-model", &[1, 2, 3], 8).unwrap();
            fleet.node_of(s).unwrap()
        })
        .collect();
    plain.sort();
    plain.dedup();
    assert_eq!(plain, ["new", "old"], "any other session goes to either");
}

#[test]
fn a_fleet_head_runs_a_batch_invariant_lane_on_a_node_that_can_and_names_one_that_cannot() {
    use superfluid_daemon::{qos, wal::role, GenParams};
    let serve_keyed = |daemon: Daemon| {
        let registry = Arc::new(ModelRegistry::single("mock-model", Arc::new(daemon)));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let keys = superfluid_daemon::keypolicy::KeyTable::parse(r#"{"keys":[{"key":"sk-det","name":"det","batch_invariant":true}]}"#, |_| None).unwrap();
        std::thread::spawn(move || {
            let _ = openai::serve_blocking_config(listener, registry, openai::ServeConfig { keys: Some(Arc::new(keys)), ..Default::default() });
        });
        addr
    };
    let ask = |addr: &str, invariant: bool, stream: bool| {
        let body = serde_json::json!({"model": "mock-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stream": stream});
        let mut req = agent()
            .post(&format!("http://{addr}/v1/chat/completions"))
            .header("content-type", "application/json")
            .header("authorization", "Bearer sk-det");
        if invariant {
            req = req.header("x-superfluid-batch-invariant", "true");
        }
        let mut resp = req.send(body.to_string().as_bytes()).unwrap();
        (resp.status().as_u16(), resp.body_mut().read_to_string().unwrap())
    };

    // A node of this release takes the lane, by SetQos and over HTTP, streamed or not.
    let (new, _) = qos_node("new", 2, [0; 4], std::time::Duration::ZERO, false);
    let head = Daemon::over_fleet(store("head"), Box::new(MockChatCodec), fleet_of(new), "mock-model".into(), DaemonOptions::default()).unwrap();
    let s = head.create(None, GenParams::default()).unwrap();
    head.set_qos(s, qos::FOREGROUND_AGENT, true).expect("a node that can run it alone is up");
    head.append_message(s, role::USER, "hi".into()).unwrap();
    assert!(head.generate(s, 4).unwrap().tokens_generated > 0);
    let addr = serve_keyed(head);
    for stream in [false, true] {
        let (status, said) = ask(&addr, true, stream);
        assert_eq!(status, 200, "stream={stream}: {said}");
    }

    // A node from before is named in the refusal, which comes before a stream starts.
    let (old, _) = qos_node("box-old", 2, [0; 4], std::time::Duration::ZERO, true);
    let head = Daemon::over_fleet(store("head"), Box::new(MockChatCodec), fleet_of(old.clone()), "mock-model".into(), DaemonOptions::default()).unwrap();
    let s = head.create(None, GenParams::default()).unwrap();
    match head.set_qos(s, qos::FOREGROUND_AGENT, true) {
        Err(superfluid_daemon::DaemonError::Unsupported(r)) => {
            assert_eq!(r.code, "unsupported_on_fleet");
            assert!(r.message.contains("box-old") && r.message.contains("batch-invariant"), "{}", r.message);
        }
        other => panic!("a node that cannot run alone was given a batch-invariant lane: {:?}", other.map(|_| ()).map_err(|e| e.to_string())),
    }
    head.set_qos(s, qos::FOREGROUND_AGENT, false).expect("any other QoS is taken");

    // A session the store already holds as batch-invariant is refused there too.
    let mut kept = store("kept");
    let held = kept.create(None, GenParams::default()).unwrap();
    kept.append(held, None, vec![1, 2, 3]).unwrap();
    kept.set_qos(held, qos::FOREGROUND_AGENT, true).unwrap();
    let over = Daemon::over_fleet(kept, Box::new(MockChatCodec), fleet_of(old), "mock-model".into(), DaemonOptions::default()).unwrap();
    let refused = over.generate(held, 4).map(|o| o.tokens_generated).map_err(|e| e.to_string());
    assert!(refused.as_ref().is_err_and(|e| e.contains("box-old") && e.contains("batch-invariant")), "{refused:?}");

    let addr = serve_keyed(head);
    for (invariant, stream) in [(true, false), (true, true), (false, false)] {
        let (status, said) = ask(&addr, invariant, stream);
        if invariant {
            assert_eq!(status, 400, "stream={stream}: {said}");
            assert!(said.contains("unsupported_on_fleet") && said.contains("box-old"), "stream={stream}: {said}");
        } else {
            assert_eq!(status, 200, "{said}");
        }
    }
}

#[test]
fn a_fleet_head_sends_each_request_in_the_class_it_resolves() {
    // Each class may hold one of the node's two lanes: of two requests at once in a class, one
    // waits in that class's queue on the node, which says what class the head sent.
    use superfluid_daemon::qos;
    let (addr, stats) = qos_node("classes", 2, [1, 1, 1, 1], std::time::Duration::from_millis(20), false);
    let fleet = fleet_of(addr);
    let queued_in = |head: &str, headers: &[(&'static str, &'static str)]| -> usize {
        let runs: Vec<_> = (0..2)
            .map(|i| {
                let (head, headers) = (head.to_string(), headers.to_vec());
                std::thread::spawn(move || {
                    let body = serde_json::json!({"model": "mock-model", "messages": [{"role": "user", "content": format!("hello {i}")}], "max_tokens": 300, "ignore_eos": true});
                    let mut req = agent().post(&format!("http://{head}/v1/chat/completions")).header("content-type", "application/json");
                    for (k, v) in headers {
                        req = req.header(k, v);
                    }
                    let mut resp = req.send(body.to_string().as_bytes()).unwrap();
                    (resp.status().as_u16(), resp.body_mut().read_to_string().unwrap())
                })
            })
            .collect();
        let by = std::time::Instant::now() + std::time::Duration::from_secs(20);
        let class = loop {
            if let Some(c) = (0..4).find(|&c| stats.queue_depth[c].load(Ordering::Relaxed) > 0) {
                break c;
            }
            assert!(std::time::Instant::now() < by, "neither request waited in a class queue on the node");
            std::thread::sleep(std::time::Duration::from_millis(2));
        };
        for run in runs {
            let (status, said) = run.join().unwrap();
            assert_eq!(status, 200, "{said}");
        }
        class
    };
    let serving = |cfg: openai::ServeConfig| {
        let daemon = Daemon::over_fleet(store("head"), Box::new(MockChatCodec), Arc::clone(&fleet), "mock-model".into(), DaemonOptions::default()).unwrap();
        let registry = Arc::new(ModelRegistry::single("mock-model", Arc::new(daemon)));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let _ = openai::serve_blocking_config(listener, registry, cfg);
        });
        addr
    };

    // --http-default-qos background, and the header over it.
    let background = openai::RequestQos { class: qos::BACKGROUND_AGENT, batch_invariant: false };
    let head = serving(openai::ServeConfig { qos: openai::HttpQosPolicy { default: background, ..Default::default() }, ..Default::default() });
    assert_eq!(queued_in(&head, &[]), qos::BACKGROUND_AGENT as usize, "the default class");
    assert_eq!(queued_in(&head, &[("x-superfluid-qos", "interactive")]), qos::INTERACTIVE_CHAT as usize, "the header's class");

    // --no-http-qos-header: the default, whatever the header says.
    let head = serving(openai::ServeConfig {
        qos: openai::HttpQosPolicy { default: background, honor_headers: false, ..Default::default() },
        ..Default::default()
    });
    assert_eq!(queued_in(&head, &[("x-superfluid-qos", "interactive")]), qos::BACKGROUND_AGENT as usize, "the header is ignored");

    // A key policy's class.
    let keys = superfluid_daemon::keypolicy::KeyTable::parse(r#"{"keys":[{"key":"sk-tab","name":"tab","class":"completion"}]}"#, |_| None).unwrap();
    let head = serving(openai::ServeConfig { keys: Some(Arc::new(keys)), ..Default::default() });
    assert_eq!(queued_in(&head, &[("authorization", "Bearer sk-tab")]), qos::INLINE_COMPLETION as usize, "the key's class");
}

/// The operator's sampling defaults (`--repeat-penalty`, `--presence-penalty`,
/// `--spec-max-temperature`) are the head's: it resolves each request before a node sees it, and
/// a node applies only what the request carries, so a request's own value still wins. The head
/// takes from the node's capability record what the node's own server would from its engine:
/// whether the engine applies penalties itself (the presence default) and whether it speculates
/// (the temperature cap), and answers as one machine would.
#[test]
fn the_head_resolves_the_operators_sampling_defaults_as_its_nodes_own_server_would() {
    let op = openai::SamplingOverrides {
        repeat_penalty: Some(1.3),
        presence_penalty: Some(0.5),
        spec_max_temperature: Some(0.2),
        ..Default::default()
    };
    for (speculates, penalties) in [(true, true), (false, false), (true, false), (false, true)] {
        let record = if penalties {
            r#"{"descriptor_version":1,"architecture":"mock","workload":{"causal_generation":true},"sampling":{"engine_penalties":true}}"#
        } else {
            r#"{"descriptor_version":1,"architecture":"mock","workload":{"causal_generation":true}}"#
        };
        let engine = || {
            EngineHost::spawn(move || (MockEngine::new(EngineConfig { vocab: 512, ..EngineConfig::default() }).with_capability_descriptor(record), None))
                .unwrap()
        };
        let opts = || DaemonOptions { speculate: speculates.then(|| "prompt-lookup".to_string()), ..DaemonOptions::default() };
        let local = Arc::new(Daemon::with_options(store("op-local"), engine(), Box::new(MockChatCodec), opts()).unwrap());
        let node = Daemon::with_options(store("op-node"), engine(), Box::new(MockCodec), opts()).unwrap();
        assert_eq!(node.speculation().is_some(), speculates);
        let head = Arc::new(head_daemon(node_serving(node)));
        let by = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while head.capabilities().record().and_then(|r| r.get("architecture")).is_none() {
            assert!(std::time::Instant::now() < by, "the head learned its node's capability record");
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let (here, there) = (op.for_daemon(&local), op.for_daemon(&head));
        assert_eq!(there, here, "speculates={speculates} penalties={penalties}: resolved as one machine would");
        assert_eq!((there.spec_max_temperature.is_some(), there.presence_penalty.is_some()), (speculates, penalties));
        assert_eq!(there.repeat_penalty, Some(1.3));
        if !(speculates && penalties) {
            continue;
        }
        let cfg = || openai::ServeConfig { sampling: op, ..Default::default() };
        let (local, head) = (serve_with(local, cfg()), serve_with(head, cfg()));
        let props = |addr: &str| {
            let mut resp = agent().get(&format!("http://{addr}/props")).call().unwrap();
            json(&resp.body_mut().read_to_string().unwrap())["default_generation_settings"].clone()
        };
        let (l, h) = (props(&local), props(&head));
        assert_eq!((h["repeat_penalty"].as_f64(), h["presence_penalty"].as_f64()), (Some(1.3f32 as f64), Some(0.5)), "{h}");
        assert_eq!((&h["repeat_penalty"], &h["presence_penalty"]), (&l["repeat_penalty"], &l["presence_penalty"]));
        // No temperature or penalty named: the defaults apply, through the head as on one machine;
        // one named wins over them. (What each says of its speculation is the engine's own state.)
        for extra in [serde_json::json!({}), serde_json::json!({"repeat_penalty": 1.0, "presence_penalty": 0, "temperature": 0})] {
            let mut body = serde_json::json!({"model": "mock-model", "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 12, "seed": 7});
            body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            let (l, h) = (post(&local, "/v1/chat/completions", &body), post(&head, "/v1/chat/completions", &body));
            assert_eq!((h.0, l.0), (200, 200), "{body}: {} / {}", l.1, h.1);
            assert_eq!(said(&h.1)["choices"], said(&l.1)["choices"], "{body}: the head said otherwise");
        }
    }
}

/// What the head takes from its nodes' capability record is the record of the node serving now,
/// not of the first that answered: a node can be removed and another join in its place, started
/// otherwise. The head reads the record again when its nodes change, so the temperature cap
/// (`--spec-max-temperature`) comes with a node that speculates and goes with one that does not.
#[test]
fn the_head_reads_its_nodes_record_again_when_another_node_takes_over() {
    const TOKEN: &[u8] = b"a fleet token";
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let fleet = Arc::new(FleetManager::with_options(Vec::new(), PlacementPolicy::LeastLoaded, lease, 90, 0));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let join_addr = listener.local_addr().unwrap().to_string();
    let models = superfluid_daemon::fleet_models::HeadModels::fixed(Arc::clone(&fleet), vec!["mock-model".into()]);
    let accepting = Arc::clone(&fleet);
    std::thread::spawn(move || superfluid_daemon::fleet_join::accept_nodes(listener, accepting, TOKEN.to_vec(), models));
    let head = Daemon::over_fleet(store("head"), Box::new(MockChatCodec), Arc::clone(&fleet), "mock-model".into(), DaemonOptions::default())
        .expect("a daemon over the fleet");

    // A node that dials the head as `identity`, speculating or not, until the head removes it.
    let join = |identity: &'static str, speculates: bool| {
        let record = r#"{"descriptor_version":1,"architecture":"mock","workload":{"causal_generation":true}}"#;
        let engine = EngineHost::spawn(move || {
            (MockEngine::new(EngineConfig { vocab: 512, ..EngineConfig::default() }).with_capability_descriptor(record), None)
        })
        .unwrap();
        let opts = DaemonOptions { speculate: speculates.then(|| "prompt-lookup".to_string()), ..DaemonOptions::default() };
        let daemon = Daemon::with_options(store("joined"), engine, Box::new(MockCodec), opts).unwrap();
        assert_eq!(daemon.speculation().is_some(), speculates);
        let mut conn = NodeAgent::new(Arc::new(daemon), identity, "mock-model").with_max_lanes(1).with_auth(TOKEN.to_vec()).connection();
        let head = join_addr.clone();
        std::thread::spawn(move || loop {
            let Ok(sock) = std::net::TcpStream::connect(&head) else { break };
            let secure = superfluid_linkf::Endpoint::from_tcp_secure(sock, TOKEN, superfluid_linkf::Role::Responder, std::time::Duration::from_secs(5));
            let Ok(ep) = secure else { break };
            if matches!(conn.serve(ep), Err(superfluid_daemon::DaemonError::Fleet(superfluid_linkf::LinkFError::Removed))) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        });
    };
    let wait_for = |what: &str, ok: &dyn Fn() -> bool| {
        let by = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while !ok() {
            assert!(std::time::Instant::now() < by, "{what}");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    };
    let op = openai::SamplingOverrides { spec_max_temperature: Some(0.2), ..Default::default() };
    let capped = || op.for_daemon(&head).spec_max_temperature.is_some();

    join("plain", false);
    wait_for("the head learned its node's capability record", &|| head.capabilities().record().and_then(|r| r.get("architecture")).is_some());
    assert!(!capped(), "no cap over a node that does not speculate");

    fleet.remove("plain").expect("the plain node is listed");
    join("speculating", true);
    wait_for("the cap applies once a node that speculates serves", &capped);

    fleet.remove("speculating").expect("the speculating node is listed");
    join("plain-again", false);
    wait_for("the cap is gone once the node serving does not speculate", &|| !capped());
}
