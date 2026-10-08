//! The Responses API over the full mock stack: items in and out, stored
//! responses continued by `previous_response_id`, streaming, and the store.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};
use superfluid_daemon::codec::{MockChatCodec, MockCodec, MockTemplateCodec, TextCodec, MOCK_TOOL_CLOSE, MOCK_TOOL_OPEN};
use superfluid_daemon::{openai, Daemon, DaemonOptions, EngineHost, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};

fn unique() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos();
    format!("{t}-{}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

fn fresh_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("superfluid-responses-{}-{}", std::process::id(), unique()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A server over `dir`: its session log, and the stored responses beside it.
fn serve_on(
    dir: &Path,
    codec: Box<dyn TextCodec + Send + Sync>,
    engine: EngineConfig,
    cfg: openai::ServeConfig,
) -> (SocketAddr, Arc<Daemon>) {
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || (MockEngine::new(engine.clone()), None)).expect("spawn");
    let daemon = Arc::new(
        Daemon::with_options(store, host, codec, DaemonOptions { media_dir: Some(dir.join("media")), ..Default::default() })
            .unwrap(),
    );
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock-model", Arc::clone(&daemon)));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let _ = openai::serve_blocking_config(listener, registry, cfg);
    });
    (addr, daemon)
}

/// An engine that answers every request with `tokens`, then ends its turn.
fn scripted(tokens: Vec<u32>) -> EngineConfig {
    EngineConfig { scripted: tokens, ..Default::default() }
}

fn serve(script: &str) -> (SocketAddr, Arc<Daemon>) {
    serve_on(&fresh_dir(), Box::new(MockChatCodec), scripted(MockCodec.encode(script)), Default::default())
}

fn dechunk(payload: &str) -> String {
    let mut out = String::new();
    let mut rest = payload;
    while let Some((size_line, tail)) = rest.split_once("\r\n") {
        let Ok(size) = usize::from_str_radix(size_line.trim(), 16) else { break };
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

fn call(addr: SocketAddr, method: &str, path: &str, body: &str, bearer: Option<&str>) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    let auth = bearer.map(|k| format!("authorization: Bearer {k}\r\n")).unwrap_or_default();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\n{auth}content-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
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

fn create(addr: SocketAddr, body: Value) -> Value {
    let (status, resp) = call(addr, "POST", "/v1/responses", &body.to_string(), None);
    assert_eq!(status, 200, "{resp}");
    serde_json::from_str(&resp).unwrap()
}

fn tool_envelope(json: &str) -> Vec<u32> {
    let mut v = vec![MOCK_TOOL_OPEN];
    v.extend(MockCodec.encode(json));
    v.push(MOCK_TOOL_CLOSE);
    v
}

fn weather_tool() -> Value {
    json!([{"type": "function", "name": "get_weather", "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}])
}

fn newest(daemon: &Daemon) -> u64 {
    daemon.session_ids().into_iter().max().expect("a session")
}

/// A session's stream as text, its own generation included.
fn prompt_of(daemon: &Daemon, session: u64) -> String {
    let i = daemon.inspect(session).unwrap();
    i.tokens.iter().filter(|t| (0x100..0x200).contains(*t)).map(|t| (t - 0x100) as u8 as char).collect()
}

fn parent_of(daemon: &Daemon, session: u64) -> Option<u64> {
    daemon.sessions(true).into_iter().find(|s| s.id == session).and_then(|s| s.parent)
}

fn session_of(daemon: &Daemon, id: &str) -> u64 {
    daemon.responses().get(id).expect("a stored response").session
}

fn text_of(response: &Value) -> String {
    response["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|i| i["type"] == "message")
        .flat_map(|i| i["content"].as_array().unwrap().iter())
        .filter_map(|p| p["text"].as_str())
        .collect()
}

#[test]
fn a_string_input_returns_a_completed_response() {
    let (addr, _d) = serve("Hello!");
    let r = create(addr, json!({"model": "mock-model", "input": "hi", "temperature": 0, "metadata": {"k": "v"}}));
    let id = r["id"].as_str().unwrap();
    assert!(id.starts_with("resp_") && id.len() == 37, "{id}");
    assert_eq!(r["object"], "response");
    assert_eq!(r["status"], "completed");
    assert_eq!(r["model"], "mock-model");
    assert_eq!(r["store"], true);
    assert_eq!(r["metadata"], json!({"k": "v"}));
    assert!(r["incomplete_details"].is_null());
    assert_eq!(r["output"][0]["type"], "message");
    assert_eq!(r["output"][0]["role"], "assistant");
    assert_eq!(r["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(text_of(&r), "Hello!");
    let u = &r["usage"];
    assert!(u["input_tokens"].as_u64().unwrap() > 0, "{u}");
    assert_eq!(
        u["total_tokens"].as_u64().unwrap(),
        u["input_tokens"].as_u64().unwrap() + u["output_tokens"].as_u64().unwrap()
    );
    assert!(u["input_tokens_details"]["cached_tokens"].is_u64(), "{u}");

    let cut = create(addr, json!({"model": "mock-model", "input": "hi", "max_output_tokens": 3}));
    assert_eq!(cut["status"], "incomplete", "{cut}");
    assert_eq!(cut["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(cut["output"][0]["status"], "incomplete");
}

#[test]
fn a_function_call_round_trips_through_items() {
    let (addr, daemon) = serve_on(
        &fresh_dir(),
        Box::new(MockChatCodec),
        scripted(tool_envelope(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#)),
        Default::default(),
    );
    let r = create(addr, json!({"model": "mock-model", "input": "weather in Paris?", "tools": weather_tool()}));
    let call = r["output"].as_array().unwrap().iter().find(|i| i["type"] == "function_call").expect("a function_call item").clone();
    assert_eq!(call["name"], "get_weather");
    assert_eq!(call["status"], "completed");
    let args: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Paris");
    let call_id = call["call_id"].as_str().unwrap();
    assert!(call_id.starts_with("call_"), "{call_id}");

    // Stateless: the client sends the whole conversation back.
    let r2 = create(addr, json!({"model": "mock-model", "store": false, "tools": weather_tool(), "input": [
        {"role": "user", "content": "weather in Paris?"},
        call,
        {"type": "function_call_output", "call_id": call_id, "output": "sunny, 21C"},
    ]}));
    assert_eq!(r2["status"], "completed", "{r2}");
    let prompt = prompt_of(&daemon, newest(&daemon));
    assert!(prompt.contains("weather in Paris?") && prompt.contains("get_weather") && prompt.contains("sunny, 21C"), "{prompt}");

    // Stateful: only the output goes back, after the stored response.
    let r3 = create(addr, json!({"model": "mock-model", "previous_response_id": r["id"], "tools": weather_tool(), "input": [
        {"type": "function_call_output", "call_id": call_id, "output": "rainy, 12C"},
    ]}));
    assert_eq!(r3["previous_response_id"], r["id"]);
    let s3 = newest(&daemon);
    assert_eq!(parent_of(&daemon, s3), Some(session_of(&daemon, r["id"].as_str().unwrap())), "forked from the stored session");
    let prompt = prompt_of(&daemon, s3);
    assert!(prompt.contains("weather in Paris?") && prompt.contains("rainy, 12C"), "{prompt}");
}

#[test]
fn previous_response_id_forks_the_stored_session_warm() {
    let (addr, daemon) = serve("Noted.");
    let r1 = create(addr, json!({"model": "mock-model", "instructions": "Be brief.", "input": "My name is Ada.", "temperature": 0}));
    let s1 = session_of(&daemon, r1["id"].as_str().unwrap());
    let r2 = create(addr, json!({"model": "mock-model", "instructions": "Be brief.", "previous_response_id": r1["id"],
        "input": "What is my name?", "temperature": 0}));
    let s2 = session_of(&daemon, r2["id"].as_str().unwrap());
    assert_eq!(parent_of(&daemon, s2), Some(s1), "same instructions and tools: a fork");
    let prompt = prompt_of(&daemon, s2);
    let at = |s: &str| prompt.find(s).unwrap_or_else(|| panic!("{s:?} not in {prompt:?}"));
    assert!(at("Be brief.") < at("My name is Ada.") && at("My name is Ada.") < at("Noted.") && at("Noted.") < at("What is my name?"), "{prompt}");
    assert_eq!(prompt.matches("Be brief.").count(), 1, "{prompt}");
    let cached = r2["usage"]["input_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert!(cached > 0, "the stored conversation is served from the prefix cache: {}", r2["usage"]);

    // Instructions are not carried over: other ones rebuild the conversation.
    let r3 = create(addr, json!({"model": "mock-model", "instructions": "Be verbose.", "previous_response_id": r2["id"], "input": "Again?"}));
    let s3 = session_of(&daemon, r3["id"].as_str().unwrap());
    assert_eq!(parent_of(&daemon, s3), None, "another head: rebuilt, not forked");
    let prompt = prompt_of(&daemon, s3);
    assert!(!prompt.contains("Be brief.") && prompt.starts_with("<0>Be verbose."), "{prompt}");
    for turn in ["My name is Ada.", "What is my name?", "Again?"] {
        assert!(prompt.contains(turn), "{turn:?} not in {prompt:?}");
    }
    let asked = &prompt[..prompt.find("Again?").unwrap()];
    assert_eq!(asked.matches("Noted.").count(), 2, "both earlier answers: {prompt}");

    // None at all: the earlier ones are gone too.
    let r4 = create(addr, json!({"model": "mock-model", "previous_response_id": r2["id"], "input": "And now?"}));
    let prompt = prompt_of(&daemon, session_of(&daemon, r4["id"].as_str().unwrap()));
    assert!(!prompt.contains("Be brief."), "{prompt}");
}

#[test]
fn a_whole_conversation_template_is_rebuilt_not_appended() {
    let (addr, daemon) = serve_on(&fresh_dir(), Box::new(MockTemplateCodec), scripted(MockCodec.encode("Noted.")), Default::default());
    let r1 = create(addr, json!({"model": "mock-model", "instructions": "Be brief.", "input": "My name is Ada."}));
    let r2 = create(addr, json!({"model": "mock-model", "instructions": "Be brief.", "previous_response_id": r1["id"], "input": "What is my name?"}));
    let s2 = session_of(&daemon, r2["id"].as_str().unwrap());
    assert_eq!(parent_of(&daemon, s2), None, "rendered fresh");
    let prompt = prompt_of(&daemon, s2);
    let at = |s: &str| prompt.find(s).unwrap_or_else(|| panic!("{s:?} not in {prompt:?}"));
    assert!(at("Be brief.") < at("My name is Ada.") && at("My name is Ada.") < at("Noted.") && at("Noted.") < at("What is my name?"), "{prompt}");
}

#[test]
fn store_false_is_answered_but_not_kept() {
    let (addr, _d) = serve("Ok.");
    let r = create(addr, json!({"model": "mock-model", "input": "hi", "store": false}));
    assert_eq!(r["store"], false);
    let id = r["id"].as_str().unwrap();
    let (status, _) = call(addr, "GET", &format!("/v1/responses/{id}"), "", None);
    assert_eq!(status, 404);
    let (status, body) = call(addr, "POST", "/v1/responses",
        &json!({"model": "mock-model", "input": "again", "previous_response_id": id}).to_string(), None);
    assert_eq!(status, 400, "an unknown previous response is a 400, as on OpenAI: {body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["code"], "previous_response_not_found");
    assert_eq!(v["error"]["message"], format!("Previous response with id '{id}' not found."));
    assert_eq!(v["error"]["param"], "previous_response_id");
}

#[test]
fn a_stored_response_is_retrieved_listed_and_deleted() {
    let (addr, daemon) = serve("Ok.");
    let r = create(addr, json!({"model": "mock-model", "input": [
        {"role": "developer", "content": "Answer in French."},
        {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
    ]}));
    let id = r["id"].as_str().unwrap();
    let session = session_of(&daemon, id);

    let (status, body) = call(addr, "GET", &format!("/v1/responses/{id}"), "", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap(), r, "retrieved as created");

    let (status, body) = call(addr, "GET", &format!("/v1/responses/{id}/input_items"), "", None);
    assert_eq!(status, 200, "{body}");
    let list: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(list["object"], "list");
    assert_eq!(list["data"].as_array().unwrap().len(), 2);
    assert_eq!(list["data"][0]["role"], "user", "newest first by default");
    assert_eq!(list["data"][0]["content"][0]["text"], "Hello");
    assert_eq!(list["has_more"], false);
    let (_, body) = call(addr, "GET", &format!("/v1/responses/{id}/input_items?order=asc&limit=1"), "", None);
    let page: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(page["data"][0]["role"], "developer");
    assert_eq!(page["has_more"], true);
    let (_, body) = call(addr, "GET", &format!("/v1/responses/{id}/input_items?order=asc&after={}", page["last_id"].as_str().unwrap()), "", None);
    let rest: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(rest["data"][0]["role"], "user", "{rest}");

    let (status, body) = call(addr, "POST", &format!("/v1/responses/{id}/cancel"), "", None);
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("background"), "{body}");

    assert!(daemon.sessions(false).iter().any(|s| s.id == session));
    let (status, body) = call(addr, "DELETE", &format!("/v1/responses/{id}"), "", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap(), json!({"id": id, "object": "response", "deleted": true}));
    assert!(!daemon.sessions(false).iter().any(|s| s.id == session), "its session is purged");
    let (status, body) = call(addr, "POST", "/v1/responses",
        &json!({"model": "mock-model", "input": "again", "previous_response_id": id}).to_string(), None);
    assert_eq!(status, 400, "a deleted previous response: {body}");
    assert!(body.contains("previous_response_not_found"), "{body}");
    for (method, path) in [("GET", format!("/v1/responses/{id}")), ("DELETE", format!("/v1/responses/{id}")),
                           ("GET", format!("/v1/responses/{id}/input_items")), ("POST", format!("/v1/responses/{id}/cancel"))] {
        let (status, body) = call(addr, method, &path, "", None);
        assert_eq!(status, 404, "{method} {path}: {body}");
    }
}

#[test]
fn deleting_a_response_keeps_the_history_a_later_one_continues() {
    let (addr, daemon) = serve_on(&fresh_dir(), Box::new(MockTemplateCodec), scripted(MockCodec.encode("Noted.")), Default::default());
    let r1 = create(addr, json!({"model": "mock-model", "input": "My name is Ada."}));
    let r2 = create(addr, json!({"model": "mock-model", "previous_response_id": r1["id"], "input": "Hi again."}));
    let (status, _) = call(addr, "DELETE", &format!("/v1/responses/{}", r1["id"].as_str().unwrap()), "", None);
    assert_eq!(status, 200);
    let r3 = create(addr, json!({"model": "mock-model", "previous_response_id": r2["id"], "input": "What is my name?"}));
    let prompt = prompt_of(&daemon, session_of(&daemon, r3["id"].as_str().unwrap()));
    assert!(prompt.contains("My name is Ada.") && prompt.contains("Hi again."), "{prompt}");
}

fn sse_events(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter(|b| !b.trim().is_empty())
        .filter_map(|block| {
            let mut name = None;
            let mut data = None;
            for line in block.lines() {
                if let Some(n) = line.strip_prefix("event: ") {
                    name = Some(n.to_string());
                }
                if let Some(d) = line.strip_prefix("data: ") {
                    data = Some(serde_json::from_str::<Value>(d).unwrap());
                }
            }
            Some((name?, data?))
        })
        .collect()
}

#[test]
fn a_streamed_response_emits_ordered_numbered_events_and_is_stored() {
    let (addr, _d) = serve("Hi there");
    let (status, body) = call(addr, "POST", "/v1/responses",
        &json!({"model": "mock-model", "input": "hello", "stream": true, "temperature": 0}).to_string(), None);
    assert_eq!(status, 200, "{body}");
    let events = sse_events(&body);
    for (i, (name, data)) in events.iter().enumerate() {
        assert_eq!(&data["type"], name, "event name matches its type");
        assert_eq!(data["sequence_number"], i as u64, "{name}");
    }
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
    let deltas = names.iter().filter(|n| **n == "response.output_text.delta").count();
    assert!(deltas >= 2, "token-cadence deltas: {names:?}");
    let mut expect = vec!["response.created", "response.in_progress", "response.output_item.added", "response.content_part.added"];
    expect.extend(std::iter::repeat_n("response.output_text.delta", deltas));
    expect.extend(["response.output_text.done", "response.content_part.done", "response.output_item.done", "response.completed"]);
    assert_eq!(names, expect);
    let streamed: String = events.iter().filter(|(n, _)| n == "response.output_text.delta").map(|(_, d)| d["delta"].as_str().unwrap()).collect();
    assert_eq!(streamed, "Hi there");
    let done = &events.iter().find(|(n, _)| n == "response.output_text.done").unwrap().1;
    assert_eq!(done["text"], "Hi there");
    let completed = &events.last().unwrap().1["response"];
    assert_eq!(completed["status"], "completed");
    assert_eq!(text_of(completed), "Hi there");
    assert!(completed["usage"]["output_tokens"].as_u64().unwrap() > 0);
    assert_eq!(events[0].1["response"]["status"], "in_progress");

    let (status, body) = call(addr, "GET", &format!("/v1/responses/{}", completed["id"].as_str().unwrap()), "", None);
    assert_eq!(status, 200);
    assert_eq!(&serde_json::from_str::<Value>(&body).unwrap(), completed);
}

#[test]
fn a_streamed_function_call_streams_its_arguments() {
    let (addr, _d) = serve_on(
        &fresh_dir(),
        Box::new(MockChatCodec),
        scripted(tool_envelope(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#)),
        Default::default(),
    );
    let (status, body) = call(addr, "POST", "/v1/responses",
        &json!({"model": "mock-model", "input": "weather?", "tools": weather_tool(), "stream": true}).to_string(), None);
    assert_eq!(status, 200, "{body}");
    let events = sse_events(&body);
    let added = events.iter().find(|(n, d)| n == "response.output_item.added" && d["item"]["type"] == "function_call").expect("call added");
    assert_eq!(added.1["item"]["name"], "get_weather");
    let args: String = events.iter().filter(|(n, _)| n == "response.function_call_arguments.delta").map(|(_, d)| d["delta"].as_str().unwrap()).collect();
    let done = &events.iter().find(|(n, _)| n == "response.function_call_arguments.done").expect("arguments done").1;
    assert_eq!(done["arguments"].as_str().unwrap(), args);
    assert_eq!(serde_json::from_str::<Value>(&args).unwrap()["city"], "Paris");
    let completed = &events.last().unwrap();
    assert_eq!(completed.0, "response.completed");
    let call = completed.1["response"]["output"].as_array().unwrap().iter().find(|i| i["type"] == "function_call").unwrap();
    assert_eq!(call["arguments"].as_str().unwrap(), args);
    assert_eq!(call["call_id"], added.1["item"]["call_id"]);
}

#[test]
fn hosted_tools_and_background_mode_are_refused_before_any_session() {
    let (addr, daemon) = serve("Ok.");
    let before = daemon.session_ids().len();
    for (body, param, code) in [
        (json!({"model": "mock-model", "input": "hi", "tools": [{"type": "web_search"}]}), "tools[0]", "unsupported_tool"),
        (json!({"model": "mock-model", "input": "hi", "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}]}), "tools[0]", "unsupported_tool"),
        (json!({"model": "mock-model", "input": "hi", "background": true}), "background", "unsupported_parameter"),
        (json!({"model": "mock-model", "input": "hi", "tool_choice": {"type": "web_search_preview"}}), "tool_choice", "unsupported_parameter"),
        (json!({"model": "mock-model", "input": [{"type": "item_reference", "id": "msg_1"}]}), "input[0]", "unsupported_parameter"),
        (json!({"model": "mock-model", "input": []}), "input", "missing_required_parameter"),
    ] {
        let (status, resp) = call(addr, "POST", "/v1/responses", &body.to_string(), None);
        assert_eq!(status, 400, "{body}: {resp}");
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!((v["error"]["param"].as_str(), v["error"]["code"].as_str()), (Some(param), Some(code)), "{resp}");
    }
    assert_eq!(daemon.session_ids().len(), before);
}

#[test]
fn a_key_sees_only_its_own_responses() {
    let table = superfluid_daemon::keypolicy::KeyTable::parse(
        r#"{"keys":[{"name":"alice","key":"sk-alice"},{"name":"bob","key":"sk-bob"}]}"#,
        |_| None,
    )
    .unwrap();
    let cfg = openai::ServeConfig { keys: Some(Arc::new(table)), ..Default::default() };
    let (addr, _d) = serve_on(&fresh_dir(), Box::new(MockChatCodec), scripted(MockCodec.encode("Ok.")), cfg);
    let (status, body) = call(addr, "POST", "/v1/responses", &json!({"model": "mock-model", "input": "hi"}).to_string(), Some("sk-alice"));
    assert_eq!(status, 200, "{body}");
    let id = serde_json::from_str::<Value>(&body).unwrap()["id"].as_str().unwrap().to_string();
    let path = format!("/v1/responses/{id}");
    assert_eq!(call(addr, "GET", &path, "", Some("sk-bob")).0, 404);
    assert_eq!(call(addr, "GET", &format!("{path}/input_items"), "", Some("sk-bob")).0, 404);
    assert_eq!(call(addr, "DELETE", &path, "", Some("sk-bob")).0, 404);
    let cont = json!({"model": "mock-model", "input": "again", "previous_response_id": id}).to_string();
    assert_eq!(call(addr, "POST", "/v1/responses", &cont, Some("sk-bob")).0, 400, "another key's response is unknown to bob");
    assert_eq!(call(addr, "GET", &path, "", Some("sk-alice")).0, 200);
    assert_eq!(call(addr, "POST", "/v1/responses", &cont, Some("sk-alice")).0, 200);
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let target = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &target);
        } else {
            std::fs::copy(e.path(), target).unwrap();
        }
    }
}

#[test]
fn stored_responses_survive_a_restart() {
    let dir = fresh_dir();
    let (addr, first) = serve_on(&dir, Box::new(MockChatCodec), scripted(MockCodec.encode("Noted.")), Default::default());
    let r1 = create(addr, json!({"model": "mock-model", "input": "My name is Ada."}));
    let id = r1["id"].as_str().unwrap();
    let s1 = session_of(&first, id);

    // The directory as a crash leaves it, under a server started afresh: one
    // process cannot open a session log twice, so the restart runs on a copy.
    let restarted = fresh_dir();
    copy_dir(&dir, &restarted);
    let (addr, second) = serve_on(&restarted, Box::new(MockChatCodec), scripted(MockCodec.encode("Ada.")), Default::default());
    let (status, body) = call(addr, "GET", &format!("/v1/responses/{id}"), "", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap(), r1);
    let r2 = create(addr, json!({"model": "mock-model", "previous_response_id": id, "input": "What is my name?"}));
    let s2 = session_of(&second, r2["id"].as_str().unwrap());
    assert_eq!(parent_of(&second, s2), Some(s1), "forked from the session the first server stored");
    let prompt = prompt_of(&second, s2);
    assert!(prompt.contains("My name is Ada.") && prompt.contains("Noted.") && prompt.contains("What is my name?"), "{prompt}");
}

#[test]
fn a_client_that_leaves_a_stream_cancels_it_and_nothing_is_stored() {
    let dir = fresh_dir();
    let engine = EngineConfig { tick_delay: std::time::Duration::from_millis(20), ..Default::default() };
    let (addr, daemon) = serve_on(&dir, Box::new(MockChatCodec), engine, Default::default());
    let active = daemon.active_registry();
    let body = json!({"model": "mock-model", "input": "hello", "stream": true, "max_output_tokens": 4000}).to_string();
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "POST /v1/responses HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut first = [0u8; 256];
    let n = s.read(&mut first).unwrap();
    assert!(String::from_utf8_lossy(&first[..n]).starts_with("HTTP/1.1 200"));
    let t0 = std::time::Instant::now();
    while active.lock().unwrap().is_empty() {
        assert!(t0.elapsed() < std::time::Duration::from_secs(20), "generation never started");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    drop(s);
    let t0 = std::time::Instant::now();
    while !active.lock().unwrap().is_empty() {
        assert!(t0.elapsed() < std::time::Duration::from_secs(10), "the generation outlived its client");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    std::thread::sleep(std::time::Duration::from_millis(100));
    let stored = std::fs::read_dir(dir.join("responses")).unwrap().count();
    assert_eq!(stored, 0, "a response its client left is not stored");
}

#[test]
fn reasoning_comes_out_as_a_reasoning_item() {
    use superfluid_daemon::codec::{MockThinkingTemplateCodec, MOCK_THINK_CLOSE};
    let mut script = MockCodec.encode("thoughts");
    script.push(MOCK_THINK_CLOSE);
    script.extend(MockCodec.encode("Hello"));
    let (addr, _d) = serve_on(&fresh_dir(), Box::new(MockThinkingTemplateCodec), scripted(script), Default::default());
    let r = create(addr, json!({"model": "mock-model", "input": "hi"}));
    assert_eq!(r["output"][0]["type"], "reasoning", "{r}");
    assert_eq!(r["output"][0]["content"][0], json!({"type": "reasoning_text", "text": "thoughts"}));
    assert_eq!(text_of(&r), "Hello");
    assert!(r["usage"]["output_tokens_details"]["reasoning_tokens"].as_u64().unwrap() > 0, "{}", r["usage"]);

    let (status, body) = call(addr, "POST", "/v1/responses", &json!({"model": "mock-model", "input": "hi", "stream": true}).to_string(), None);
    assert_eq!(status, 200);
    let events = sse_events(&body);
    let thought: String = events.iter().filter(|(n, _)| n == "response.reasoning_text.delta").map(|(_, d)| d["delta"].as_str().unwrap()).collect();
    assert_eq!(thought, "thoughts");
    let done = &events.last().unwrap().1["response"];
    assert_eq!(done["output"][0]["type"], "reasoning");
    assert_eq!(done["output"][1]["type"], "message");
}
