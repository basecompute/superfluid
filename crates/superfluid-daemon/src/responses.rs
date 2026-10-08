//! api-openai-responses: OpenAI's Responses API, a dialect over the chat path.
//!
//! A stored response names the session it ran in. `previous_response_id`
//! forks that session after the response's last event and appends only the
//! new input, so the conversation continues warm. Where a fork would not
//! render what a fresh request renders (a template that renders whole
//! conversations, other instructions or tools, another model, a turn cut
//! short), the conversation is rebuilt from the stored items instead.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::generation_output::{Body, Frame, Generation, Message, StreamFailure, Usage};
use crate::http_state::{error_response, refusal_response, AppState};
use crate::inference::{self, ChatMessage, ChatRequest, Continuation, ReqFunction, ReqToolCall, RequestPolicy};
use crate::response_store::{new_response_id, ResponseBody, ResponseRecord, ResponseStore};
use crate::{Daemon, DaemonError, Refusal};

type Key = Option<axum::Extension<crate::keypolicy::ClientKey>>;

#[derive(Deserialize)]
struct CreateRequest {
    model: Option<String>,
    #[serde(default)]
    input: Value,
    instructions: Option<String>,
    #[serde(default)]
    tools: Vec<Value>,
    tool_choice: Option<Value>,
    text: Option<Value>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_output_tokens: Option<u32>,
    seed: Option<u64>,
    parallel_tool_calls: Option<bool>,
    reasoning: Option<Value>,
    metadata: Option<Value>,
    store: Option<bool>,
    #[serde(default)]
    stream: bool,
    previous_response_id: Option<String>,
    background: Option<bool>,
    conversation: Option<Value>,
    prompt: Option<Value>,
    truncation: Option<String>,
}

fn caller(key: &Key) -> Option<&str> {
    key.as_ref().map(|k| k.0 .0.name.as_str())
}

/// With API keys on, a key sees only the responses it created.
fn visible(rec: &ResponseRecord, caller: Option<&str>) -> bool {
    caller.is_none_or(|c| rec.owner.as_deref() == Some(c))
}

fn refuse(param: &str, code: &'static str, message: impl Into<String>) -> Response {
    refusal_response(StatusCode::BAD_REQUEST, &Refusal::new(code, Some(param), message))
}

fn not_found(id: &str) -> Response {
    error_response(StatusCode::NOT_FOUND, format!("Response with id '{id}' not found."))
}

fn str_of<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

/// The request's input as items, each with the id it is listed under.
#[allow(clippy::result_large_err)]
fn input_items(input: &Value, suffix: &str) -> Result<Vec<Value>, Response> {
    let raw: Vec<Value> = match input {
        Value::String(s) => vec![json!({"type": "message", "role": "user", "content": s})],
        Value::Array(a) => a.clone(),
        Value::Null => Vec::new(),
        _ => return Err(refuse("input", "invalid_type", "input must be a string or an array of input items")),
    };
    let mut out = Vec::with_capacity(raw.len());
    for (i, mut item) in raw.into_iter().enumerate() {
        let Some(obj) = item.as_object_mut() else {
            return Err(refuse(&format!("input[{i}]"), "invalid_type", "an input item must be an object"));
        };
        if !obj.contains_key("type") && obj.contains_key("role") {
            obj.insert("type".into(), "message".into());
        }
        let kind = obj.get("type").and_then(Value::as_str).unwrap_or("").to_string();
        if kind == "message" {
            let assistant = obj.get("role").and_then(Value::as_str) == Some("assistant");
            if let Some(Value::String(text)) = obj.get("content") {
                let part = if assistant {
                    json!({"type": "output_text", "text": text, "annotations": []})
                } else {
                    json!({"type": "input_text", "text": text})
                };
                obj.insert("content".into(), json!([part]));
            }
            obj.entry("status").or_insert_with(|| "completed".into());
        }
        if !obj.contains_key("id") {
            let prefix = match kind.as_str() {
                "message" => "msg",
                "function_call" => "fc",
                "function_call_output" => "fco",
                "reasoning" => "rs",
                _ => "item",
            };
            obj.insert("id".into(), format!("{prefix}_{suffix}_in{i}").into());
        }
        out.push(item);
    }
    Ok(out)
}

/// Responses items, as chat messages: a function call joins the assistant
/// message before it, and reasoning joins the assistant turn after it.
#[derive(Default)]
struct Conversation {
    messages: Vec<ChatMessage>,
    reasoning: Option<String>,
    call_names: HashMap<String, String>,
}

impl Conversation {
    #[allow(clippy::result_large_err)]
    fn push_items(&mut self, items: &[Value], at: &str) -> Result<(), Response> {
        for (i, item) in items.iter().enumerate() {
            let param = format!("{at}[{i}]");
            match item.get("type").and_then(Value::as_str) {
                Some("message") => self.message(item, &param)?,
                Some("function_call") => {
                    let name = str_of(item, "name");
                    if name.is_empty() {
                        return Err(refuse(&format!("{param}.name"), "missing_required_parameter", "a function_call item needs a name"));
                    }
                    let name = match str_of(item, "namespace") {
                        "" => name.to_string(),
                        ns => qualified(ns, name),
                    };
                    let name = name.as_str();
                    let call_id = str_of(item, "call_id").to_string();
                    let arguments = match item.get("arguments") {
                        Some(Value::String(s)) => s.clone(),
                        Some(v) if !v.is_null() => v.to_string(),
                        _ => "{}".into(),
                    };
                    self.call_names.insert(call_id.clone(), name.to_string());
                    let call = ReqToolCall {
                        id: Some(call_id),
                        function: Some(ReqFunction { name: name.to_string(), arguments: Value::String(arguments) }),
                    };
                    match self.messages.last_mut() {
                        Some(m) if m.role == "assistant" => m.tool_calls.push(call),
                        _ => {
                            let mut m = self.turn("assistant", Value::String(String::new()));
                            m.tool_calls.push(call);
                            self.messages.push(m);
                        }
                    }
                }
                Some("function_call_output") => {
                    let call_id = str_of(item, "call_id").to_string();
                    let output = match item.get("output") {
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Array(parts)) => parts.iter().map(|p| str_of(p, "text")).collect::<Vec<_>>().join(""),
                        Some(v) if !v.is_null() => v.to_string(),
                        _ => String::new(),
                    };
                    self.messages.push(ChatMessage {
                        role: "tool".into(),
                        content: Some(Value::String(output)),
                        name: self.call_names.get(&call_id).cloned(),
                        tool_call_id: Some(call_id),
                        ..Default::default()
                    });
                }
                Some("reasoning") => {
                    let texts = |key: &str| -> Vec<String> {
                        item.get(key)
                            .and_then(Value::as_array)
                            .map(|parts| parts.iter().map(|p| str_of(p, "text").to_string()).filter(|t| !t.is_empty()).collect())
                            .unwrap_or_default()
                    };
                    let mut text = texts("content");
                    if text.is_empty() {
                        text = texts("summary");
                    }
                    if !text.is_empty() {
                        let text = text.join("\n");
                        self.reasoning = Some(match self.reasoning.take() {
                            Some(before) => format!("{before}\n{text}"),
                            None => text,
                        });
                    }
                }
                Some("item_reference") => {
                    return Err(refuse(&param, "unsupported_parameter", "item_reference input items are not supported: send the item itself, or continue with previous_response_id"));
                }
                Some(other) => {
                    return Err(refuse(&param, "unsupported_input", format!(
                        "input items of type '{other}' are not supported: send message, function_call, function_call_output and reasoning items"
                    )));
                }
                None => return Err(refuse(&param, "invalid_type", "an input item needs a type")),
            }
        }
        Ok(())
    }

    fn turn(&mut self, role: &str, content: Value) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: Some(content),
            reasoning_content: if role == "assistant" { self.reasoning.take() } else { None },
            ..Default::default()
        }
    }

    #[allow(clippy::result_large_err)]
    fn message(&mut self, item: &Value, param: &str) -> Result<(), Response> {
        let role = match item.get("role").and_then(Value::as_str) {
            Some("user") => "user",
            Some("system" | "developer") => "system",
            Some("assistant") => "assistant",
            other => {
                return Err(refuse(&format!("{param}.role"), "invalid_value", format!(
                    "role must be user, system, developer or assistant, got {}",
                    other.unwrap_or("nothing")
                )))
            }
        };
        let content = match item.get("content") {
            Some(Value::String(s)) => Value::String(s.clone()),
            Some(Value::Array(parts)) => {
                let mut out = Vec::with_capacity(parts.len());
                for (j, p) in parts.iter().enumerate() {
                    let at = format!("{param}.content[{j}]");
                    match p.get("type").and_then(Value::as_str) {
                        Some("input_text" | "output_text" | "text") => out.push(json!({"type": "text", "text": str_of(p, "text")})),
                        Some("refusal") => out.push(json!({"type": "text", "text": str_of(p, "refusal")})),
                        Some("input_image") => match p.get("image_url").and_then(Value::as_str) {
                            Some(url) => out.push(json!({"type": "image_url", "image_url": {"url": url}})),
                            None => {
                                return Err(refuse(&at, "unsupported_parameter", "an input_image must carry the image inline as a data: URL in image_url; file_id is not supported"))
                            }
                        },
                        Some("input_audio") => out.push(json!({"type": "input_audio", "input_audio": p.get("input_audio").cloned().unwrap_or(Value::Null)})),
                        Some("input_file") => {
                            return Err(refuse(&at, "unsupported_parameter", "input_file content is not supported: send the text as input_text"))
                        }
                        other => {
                            return Err(refuse(&at, "unsupported_input", format!(
                                "content parts of type '{}' are not supported",
                                other.unwrap_or("none")
                            )))
                        }
                    }
                }
                Value::Array(out)
            }
            None | Some(Value::Null) => Value::String(String::new()),
            Some(_) => return Err(refuse(&format!("{param}.content"), "invalid_type", "content must be a string or an array of parts")),
        };
        let m = self.turn(role, content);
        self.messages.push(m);
        Ok(())
    }
}

/// The chat name of a namespace's member: `namespace.member`.
fn qualified(namespace: &str, member: &str) -> String {
    if member.starts_with(&format!("{namespace}.")) {
        member.to_string()
    } else {
        format!("{namespace}.{member}")
    }
}

/// The namespace and member a chat tool name stands for, when it is a member of one of `tools`'
/// namespaces.
fn namespace_of(tools: &[Value], name: &str) -> Option<(String, String)> {
    tools.iter().filter(|t| str_of(t, "type") == "namespace").find_map(|ns| {
        let ns_name = str_of(ns, "name");
        ns.get("tools")?.as_array()?.iter().find_map(|m| {
            let member = str_of(m, "name");
            (!member.is_empty() && qualified(ns_name, member) == name).then(|| (ns_name.to_string(), member.to_string()))
        })
    })
}

#[allow(clippy::result_large_err)]
fn function_tool(t: &Value, param: &str, name: Option<String>) -> Result<Value, Response> {
    if let Some(f) = t.get("function").filter(|f| f.is_object()) {
        let mut f = f.clone();
        if let Some(name) = name {
            f["name"] = Value::String(name);
        }
        return Ok(json!({"type": "function", "function": f}));
    }
    let own = str_of(t, "name");
    if own.is_empty() {
        return Err(refuse(&format!("{param}.name"), "missing_required_parameter", "a function tool needs a name"));
    }
    let mut f = json!({"name": name.unwrap_or_else(|| own.to_string())});
    for k in ["description", "parameters", "strict"] {
        if let Some(v) = t.get(k).filter(|v| !v.is_null()) {
            f[k] = v.clone();
        }
    }
    Ok(json!({"type": "function", "function": f}))
}

/// The chat tools a request's tools come to: a namespace is flattened to `namespace.member`
/// functions, and a call to one goes back out as a `function_call` with its `namespace`.
#[allow(clippy::result_large_err)]
fn chat_tools(tools: &[Value]) -> Result<Vec<Value>, Response> {
    let mut out = Vec::with_capacity(tools.len());
    for (i, t) in tools.iter().enumerate() {
        let param = format!("tools[{i}]");
        match t.get("type").and_then(Value::as_str) {
            Some("function") => out.push(function_tool(t, &param, None)?),
            Some("namespace") => {
                let ns = str_of(t, "name");
                if ns.is_empty() {
                    return Err(refuse(&format!("{param}.name"), "missing_required_parameter", "a namespace tool needs a name"));
                }
                for (j, m) in t.get("tools").and_then(Value::as_array).into_iter().flatten().enumerate() {
                    let param = format!("{param}.tools[{j}]");
                    if str_of(m, "type") != "function" {
                        return Err(refuse(&param, "unsupported_tool", "a namespace's members must be function tools"));
                    }
                    let name = qualified(ns, str_of(m, "name"));
                    out.push(function_tool(m, &param, Some(name))?);
                }
            }
            Some(other) => {
                return Err(refuse(&param, "unsupported_tool", format!(
                    "tool type '{other}' is not supported: superfluid runs function tools, which your code executes; \
                     built-in tools (web_search, file_search, computer_use, code_interpreter, image_generation, mcp) \
                     run on OpenAI's servers"
                )))
            }
            None => return Err(refuse(&param, "invalid_type", "a tool needs a type")),
        }
    }
    Ok(out)
}

#[allow(clippy::result_large_err)]
fn chat_tool_choice(v: &Option<Value>) -> Result<Option<Value>, Response> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if matches!(s.as_str(), "auto" | "none" | "required") => Ok(v.clone()),
        Some(Value::String(s)) => Err(refuse("tool_choice", "invalid_value", format!(
            "tool_choice must be auto, none, required or a function, got '{s}'"
        ))),
        Some(o) => match o.get("type").and_then(Value::as_str) {
            Some("function") if !str_of(o, "name").is_empty() => Ok(Some(json!({"type": "function", "function": {"name": str_of(o, "name")}}))),
            Some("function") => Err(refuse("tool_choice.name", "missing_required_parameter", "a function tool_choice needs a name")),
            other => Err(refuse("tool_choice", "unsupported_parameter", format!(
                "tool_choice of type '{}' is not supported: use auto, none, required or a function",
                other.unwrap_or("none")
            ))),
        },
    }
}

/// `text.format` as the chat path's `response_format`, which compiles it to
/// a grammar.
#[allow(clippy::result_large_err)]
fn response_format(text: &Option<Value>) -> Result<Option<Value>, Response> {
    let Some(f) = text.as_ref().and_then(|t| t.get("format")).filter(|f| !f.is_null()) else {
        return Ok(None);
    };
    match f.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_object") => Ok(Some(json!({"type": "json_object"}))),
        Some("json_schema") => {
            let Some(schema) = f.get("schema").filter(|s| s.is_object()) else {
                return Err(refuse("text.format.schema", "missing_required_parameter", "a json_schema format needs a schema"));
            };
            Ok(Some(json!({"type": "json_schema", "json_schema": {
                "name": f.get("name").cloned().unwrap_or_else(|| "response".into()),
                "schema": schema,
                "strict": f.get("strict").cloned().unwrap_or(Value::Null),
            }})))
        }
        other => Err(refuse("text.format.type", "invalid_value", format!(
            "text.format.type must be text, json_object or json_schema, got '{}'",
            other.unwrap_or("none")
        ))),
    }
}

/// `reasoning.effort` as the model's own knob: its effort levels when its
/// template has them, else its thinking switch (off for none and minimal).
/// A model with neither ignores it.
fn thinking(reasoning: &Option<Value>, daemon: &Daemon) -> (Option<bool>, Option<String>) {
    let Some(effort) = reasoning.as_ref().and_then(|r| r.get("effort")).and_then(Value::as_str) else {
        return (None, None);
    };
    if daemon.supports_reasoning_effort() && daemon.reasoning_effort_levels().contains(&effort) {
        return (None, Some(effort.to_string()));
    }
    if daemon.supports_enable_thinking() {
        return (Some(!matches!(effort, "none" | "minimal")), None);
    }
    (None, None)
}

fn system_message(text: &str) -> ChatMessage {
    ChatMessage { role: "system".into(), content: Some(Value::String(text.into())), ..Default::default() }
}

/// Digest of the conversation's head (system text and whether it came from instructions):
/// a continuation forks only onto the same head.
fn head(instructions: Option<&str>, root_system: Option<&str>, chat: &ChatRequest) -> String {
    let system = instructions.or(root_system);
    let probe = ChatRequest {
        messages: system.map(|s| vec![system_message(s)]).unwrap_or_default(),
        tools: chat.tools.clone(),
        tool_choice: chat.tool_choice.clone(),
        ..Default::default()
    };
    let (system, tools) = inference::system_head(&probe).unwrap_or_default();
    let from_instructions = instructions.is_some();
    let digest =
        crate::keypolicy::sha256(&json!([system, from_instructions, tools, chat.template_kwargs()]).to_string());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The stored items of a conversation up to and including `last`, oldest
/// first: each response's input, then its output.
fn history(store: &ResponseStore, last: &str) -> Result<Vec<Value>, String> {
    let mut chain = Vec::new();
    let mut next = Some(last.to_string());
    while let Some(id) = next {
        let missing = || format!("response {id} in this conversation's history is no longer stored");
        let rec = store.record(&id).ok_or_else(missing)?;
        let body = store.body(&id).ok_or_else(missing)?;
        let mut items = body.input_items;
        items.extend(body.response.get("output").and_then(Value::as_array).cloned().unwrap_or_default());
        chain.push(items);
        next = rec.previous_response_id;
    }
    Ok(chain.into_iter().rev().flatten().collect())
}

/// What the response object echoes of its request, and what its record keeps.
struct Meta {
    id: String,
    created_at: u64,
    model: String,
    owner: Option<String>,
    store: bool,
    previous_response_id: Option<String>,
    instructions: Option<String>,
    tools: Vec<Value>,
    tool_choice: Value,
    text: Value,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_output_tokens: Option<u32>,
    parallel_tool_calls: bool,
    reasoning: Value,
    metadata: Value,
    truncation: String,
    input_items: Vec<Value>,
    head: String,
    root_system: Option<String>,
}

impl Meta {
    fn object(&self, status: &str, output: Vec<Value>, usage: Value, incomplete: Option<&str>, error: Value) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "background": false,
            "error": error,
            "incomplete_details": incomplete.map(|r| json!({"reason": r})),
            "instructions": self.instructions,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_calls": null,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": self.parallel_tool_calls,
            "previous_response_id": self.previous_response_id,
            "reasoning": self.reasoning,
            "service_tier": "default",
            "store": self.store,
            "temperature": self.temperature,
            "text": self.text,
            "tool_choice": self.tool_choice,
            "tools": self.tools,
            "top_p": self.top_p,
            "truncation": self.truncation,
            "usage": usage,
            "user": null,
            "metadata": self.metadata,
        })
    }

    fn persist(&self, model: &Daemon, store: &ResponseStore, session: Option<u64>, response: &Value) -> Result<(), DaemonError> {
        let session = session.ok_or(DaemonError::Protocol("the generation reported no session"))?;
        let (end_event, turn_closed) = model.session_end(session).ok_or(DaemonError::UnknownSession(session))?;
        store.put(
            &ResponseRecord {
                id: self.id.clone(),
                owner: self.owner.clone(),
                model: self.model.clone(),
                session,
                end_event,
                turn_closed,
                previous_response_id: self.previous_response_id.clone(),
                head: self.head.clone(),
                root_system: self.root_system.clone(),
                created_at: self.created_at,
                deleted: false,
            },
            &ResponseBody { response: response.clone(), input_items: self.input_items.clone() },
        )
    }
}

enum Open {
    Nothing,
    Reasoning(String),
    Text(String),
    Call { call_id: String, name: String, namespace: Option<String>, arguments: String },
}

/// Output items built from chat frames, and the stream events that build
/// them. The complete response goes through it too, so both carry the same
/// items.
struct Output {
    /// The request's tools, to tell a namespace member's call from a plain function's.
    tools: Vec<Value>,
    suffix: String,
    seq: u64,
    items: Vec<Value>,
    open: Open,
    events: Vec<(&'static str, Value)>,
}

impl Output {
    fn new(id: &str, tools: &[Value]) -> Output {
        Output {
            tools: tools.iter().filter(|t| str_of(t, "type") == "namespace").cloned().collect(),
            suffix: id.trim_start_matches("resp_").to_string(),
            seq: 0,
            items: Vec::new(),
            open: Open::Nothing,
            events: Vec::new(),
        }
    }

    fn emit(&mut self, kind: &'static str, fields: Value) {
        let mut v = json!({"type": kind, "sequence_number": self.seq});
        if let (Some(o), Value::Object(f)) = (v.as_object_mut(), fields) {
            o.extend(f);
        }
        self.seq += 1;
        self.events.push((kind, v));
    }

    fn index(&self) -> usize {
        self.items.len()
    }

    fn item_id(&self, prefix: &str) -> String {
        format!("{prefix}_{}_{}", self.suffix, self.index())
    }

    fn reasoning(&mut self, delta: &str) {
        if !matches!(self.open, Open::Reasoning(_)) {
            self.close("completed");
            let (id, at) = (self.item_id("rs"), self.index());
            self.emit("response.output_item.added", json!({"output_index": at, "item": {
                "id": id, "type": "reasoning", "summary": [], "content": [], "status": "in_progress",
            }}));
            self.emit("response.content_part.added", json!({
                "item_id": id, "output_index": at, "content_index": 0, "part": {"type": "reasoning_text", "text": ""},
            }));
            self.open = Open::Reasoning(String::new());
        }
        if let Open::Reasoning(text) = &mut self.open {
            text.push_str(delta);
        }
        let (id, at) = (self.item_id("rs"), self.index());
        self.emit("response.reasoning_text.delta", json!({"item_id": id, "output_index": at, "content_index": 0, "delta": delta}));
    }

    fn open_text(&mut self) {
        if matches!(self.open, Open::Text(_)) {
            return;
        }
        self.close("completed");
        let (id, at) = (self.item_id("msg"), self.index());
        self.emit("response.output_item.added", json!({"output_index": at, "item": {
            "id": id, "type": "message", "status": "in_progress", "role": "assistant", "content": [],
        }}));
        self.emit("response.content_part.added", json!({
            "item_id": id, "output_index": at, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        }));
        self.open = Open::Text(String::new());
    }

    fn text(&mut self, delta: &str) {
        self.open_text();
        if let Open::Text(text) = &mut self.open {
            text.push_str(delta);
        }
        let (id, at) = (self.item_id("msg"), self.index());
        self.emit("response.output_text.delta", json!({
            "item_id": id, "output_index": at, "content_index": 0, "delta": delta, "logprobs": [],
        }));
    }

    fn call(&mut self, name: &str, arguments: &str) {
        self.close("completed");
        let (id, at) = (self.item_id("fc"), self.index());
        let call_id = format!("call_{}_{at}", self.suffix);
        let (namespace, name) = match namespace_of(&self.tools, name) {
            Some((ns, member)) => (Some(ns), member),
            None => (None, name.to_string()),
        };
        let mut item = json!({"id": id, "type": "function_call", "status": "in_progress", "call_id": call_id, "name": name, "arguments": ""});
        if let Some(ns) = &namespace {
            item["namespace"] = Value::String(ns.clone());
        }
        self.emit("response.output_item.added", json!({"output_index": at, "item": item}));
        self.open = Open::Call { call_id, name, namespace, arguments: String::new() };
        self.arguments(arguments);
    }

    fn arguments(&mut self, delta: &str) {
        let Open::Call { arguments, .. } = &mut self.open else { return };
        if delta.is_empty() {
            return;
        }
        arguments.push_str(delta);
        let (id, at) = (self.item_id("fc"), self.index());
        self.emit("response.function_call_arguments.delta", json!({"item_id": id, "output_index": at, "delta": delta}));
    }

    fn close(&mut self, status: &str) {
        let at = self.index();
        let item = match std::mem::replace(&mut self.open, Open::Nothing) {
            Open::Nothing => return,
            Open::Reasoning(text) => {
                let id = self.item_id("rs");
                self.emit("response.reasoning_text.done", json!({"item_id": id, "output_index": at, "content_index": 0, "text": text}));
                let part = json!({"type": "reasoning_text", "text": text});
                self.emit("response.content_part.done", json!({"item_id": id, "output_index": at, "content_index": 0, "part": part}));
                json!({"id": id, "type": "reasoning", "summary": [], "content": [part], "status": status})
            }
            Open::Text(text) => {
                let id = self.item_id("msg");
                self.emit("response.output_text.done", json!({"item_id": id, "output_index": at, "content_index": 0, "text": text, "logprobs": []}));
                let part = json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []});
                self.emit("response.content_part.done", json!({"item_id": id, "output_index": at, "content_index": 0, "part": part}));
                json!({"id": id, "type": "message", "status": status, "role": "assistant", "content": [part]})
            }
            Open::Call { call_id, name, namespace, arguments } => {
                let id = self.item_id("fc");
                self.emit("response.function_call_arguments.done", json!({"item_id": id, "output_index": at, "name": name, "arguments": arguments}));
                let mut item = json!({"id": id, "type": "function_call", "status": status, "call_id": call_id, "name": name, "arguments": arguments});
                if let Some(ns) = namespace {
                    item["namespace"] = Value::String(ns);
                }
                item
            }
        };
        self.emit("response.output_item.done", json!({"output_index": at, "item": item}));
        self.items.push(item);
    }

    /// One chat frame's message: its reasoning, its text, then its calls.
    fn feed(&mut self, m: &Message) {
        if let Some(t) = m.reasoning.as_deref().filter(|t| !t.is_empty()) {
            self.reasoning(t);
        }
        if let Some(t) = m.content.as_deref().filter(|t| !t.is_empty()) {
            self.text(t);
        }
        for t in &m.tools {
            match &t.name {
                Some(name) => self.call(name, &t.arguments),
                None => self.arguments(&t.arguments),
            }
        }
    }

    /// Closes the last item; a response always has one, if only an empty
    /// message.
    fn finish(&mut self, incomplete: bool) -> Vec<Value> {
        let status = if incomplete { "incomplete" } else { "completed" };
        if self.items.is_empty() && matches!(self.open, Open::Nothing) {
            self.open_text();
        }
        self.close(status);
        self.items.clone()
    }
}

fn usage_json(usage: Option<&Usage>, reasoning_tokens: u64) -> Value {
    let Some(u) = usage else { return Value::Null };
    let input = u.prompt.unwrap_or(0);
    json!({
        "input_tokens": input,
        "input_tokens_details": {"cached_tokens": u.cached.unwrap_or(0).min(input)},
        "output_tokens": u.completion,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens.min(u.completion)},
        "total_tokens": u.total.unwrap_or(input + u.completion),
    })
}

/// The response a finished generation makes, stored when asked to be.
/// Blocking: it tokenizes and writes to disk.
fn finalize(
    meta: &Meta,
    model: &Daemon,
    store: &ResponseStore,
    out: &mut Output,
    frame: &Frame,
) -> Result<Value, DaemonError> {
    let incomplete = frame.choices.first().and_then(|c| c.finish) == Some("length");
    let output = out.finish(incomplete);
    let reasoning_tokens: u64 = output
        .iter()
        .filter(|i| i["type"] == "reasoning")
        .map(|i| model.tokenize(i["content"][0]["text"].as_str().unwrap_or("")).len() as u64)
        .sum();
    let response = meta.object(
        if incomplete { "incomplete" } else { "completed" },
        output,
        usage_json(frame.usage.as_ref(), reasoning_tokens),
        incomplete.then_some("max_output_tokens"),
        Value::Null,
    );
    if meta.store {
        meta.persist(model, store, frame.session, &response)?;
    }
    Ok(response)
}

pub(crate) async fn create(
    State(st): State<AppState>,
    key: Key,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    match start(st, key, headers, body).await {
        Ok(r) | Err(r) => r,
    }
}

#[allow(clippy::result_large_err)]
async fn start(st: AppState, key: Key, headers: HeaderMap, body: axum::body::Bytes) -> Result<Response, Response> {
    let req: CreateRequest = serde_json::from_slice(&body)
        .map_err(|e| error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")))?;
    if req.background == Some(true) {
        return Err(refuse("background", "unsupported_parameter",
            "background mode is not supported: a response runs while its request is open; stream it to watch it, close the connection to cancel it"));
    }
    if req.conversation.as_ref().is_some_and(|c| !c.is_null()) {
        return Err(refuse("conversation", "unsupported_parameter", "conversation objects are not supported: chain responses with previous_response_id"));
    }
    if req.prompt.as_ref().is_some_and(|p| !p.is_null()) {
        return Err(refuse("prompt", "unsupported_parameter", "prompt templates are not supported: send instructions and input"));
    }
    if req.metadata.as_ref().is_some_and(|m| !m.is_null() && !m.is_object()) {
        return Err(refuse("metadata", "invalid_type", "metadata must be an object of strings"));
    }
    let owner = caller(&key).map(str::to_string);
    let front = st.default_daemon();
    let prev = match &req.previous_response_id {
        None => None,
        Some(id) => match front.responses().get(id).filter(|r| visible(r, owner.as_deref())) {
            Some(r) => Some(r),
            None => {
                // A 400, as OpenAI answers it; GET and DELETE of an unknown id are 404.
                return Err(refuse("previous_response_id", "previous_response_not_found", format!(
                    "Previous response with id '{id}' not found."
                )));
            }
        },
    };
    let model = req.model.clone().or_else(|| prev.as_ref().map(|p| p.model.clone()));
    if model.is_none() && st.requires_model() {
        return Err(error_response(StatusCode::BAD_REQUEST, "missing required field: model".into()));
    }
    let qos = st.qos.resolve(&headers, key.as_ref().map(|k| &*k.0 .0))?;
    let (model_name, daemon) = st.resolve(model.as_deref())?;

    let id = new_response_id().map_err(crate::http_state::daemon_error)?;
    let suffix = id.trim_start_matches("resp_").to_string();
    let items = input_items(&req.input, &suffix)?;
    if items.is_empty() {
        return Err(refuse("input", "missing_required_parameter", "input must be non-empty"));
    }
    let mut turn = Conversation::default();
    turn.push_items(&items, "input")?;
    let (enable_thinking, reasoning_effort) = thinking(&req.reasoning, &daemon);
    let instructions = req.instructions.clone().filter(|i| !i.is_empty());
    let mut chat = ChatRequest {
        model: Some(model_name.clone()),
        tools: chat_tools(&req.tools)?,
        tool_choice: chat_tool_choice(&req.tool_choice)?,
        response_format: response_format(&req.text)?,
        max_completion_tokens: req.max_output_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        seed: req.seed,
        enable_thinking,
        reasoning_effort,
        stream: req.stream,
        route: Some("POST /v1/responses"),
        ..Default::default()
    };

    // The conversation opens with the instructions, else with the first
    // input of its first response; that is what the session renders first.
    let root_system = match &prev {
        Some(p) => p.root_system.clone(),
        None => turn.messages.first().map(inference::absorbable_system_text).transpose().map_err(crate::http_state::daemon_error)?.flatten(),
    };
    let head = head(instructions.as_deref(), root_system.as_deref(), &chat);
    let fork = prev.as_ref().filter(|p| {
        p.model == model_name && daemon.renders_per_message() && p.turn_closed && p.head == head
    });
    let fork = fork.and_then(|p| {
        daemon
            .session_end(p.session)
            .filter(|(end, _)| *end >= p.end_event)
            .map(|_| Continuation { session: p.session, at_event: p.end_event })
    });
    match (fork, &prev) {
        (Some(c), _) => {
            chat.messages = turn.messages;
            chat.continues = Some(c);
        }
        (None, prev) => {
            let mut all = Conversation::default();
            if let Some(i) = &instructions {
                all.messages.push(system_message(i));
            }
            if let Some(p) = prev {
                let store_daemon = Arc::clone(&front);
                let last = p.id.clone();
                let chain = tokio::task::spawn_blocking(move || history(store_daemon.responses(), &last))
                    .await
                    .map_err(|e| error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
                    .map_err(|why| error_response(StatusCode::BAD_REQUEST, why))?;
                all.push_items(&chain, "previous_response_id")?;
            }
            all.push_items(&items, "input")?;
            chat.messages = all.messages;
        }
    }

    let meta = Arc::new(Meta {
        id,
        created_at: inference::unix_now(),
        model: model_name.clone(),
        owner,
        store: req.store.unwrap_or(true),
        previous_response_id: req.previous_response_id.clone(),
        instructions,
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone().unwrap_or_else(|| "auto".into()),
        text: req.text.clone().unwrap_or_else(|| json!({"format": {"type": "text"}})),
        temperature: req.temperature,
        top_p: req.top_p,
        max_output_tokens: req.max_output_tokens,
        parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
        reasoning: json!({
            "effort": req.reasoning.as_ref().and_then(|r| r.get("effort")).cloned().unwrap_or(Value::Null),
            "summary": null,
        }),
        metadata: req.metadata.clone().filter(|m| !m.is_null()).unwrap_or_else(|| json!({})),
        truncation: req.truncation.clone().unwrap_or_else(|| "disabled".into()),
        input_items: items,
        head,
        root_system,
    });
    let policy = RequestPolicy { limits: st.limits, sampling: st.sampling, qos };
    let hold = key.map(|k| k.0);
    if req.stream {
        let generation = inference::chat(Arc::clone(&daemon), model_name, chat, policy, hold, Default::default())
            .await
            .map_err(crate::http_generation::run_error)?;
        return Ok(stream(generation, meta, daemon, front));
    }
    Ok(crate::keepalive::respond(st.nonstream_keepalive, move |hooks| async move {
        match inference::chat(Arc::clone(&daemon), model_name, chat, policy, hold, hooks).await {
            Ok(generation) => complete(generation, meta, daemon, front).await,
            Err(e) => crate::http_generation::run_error(e),
        }
    })
    .await)
}

async fn complete(generation: Generation, meta: Arc<Meta>, model: Arc<Daemon>, front: Arc<Daemon>) -> Response {
    let Body::Complete(frame) = generation.body else {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "a complete response came as a stream".into());
    };
    let done = tokio::task::spawn_blocking(move || {
        let mut out = Output::new(&meta.id, &meta.tools);
        if let Some(c) = frame.choices.first() {
            out.feed(&c.message);
        }
        finalize(&meta, &model, front.responses(), &mut out, &frame)
    })
    .await;
    let response = match done {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("the response could not be stored: {e}")),
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let mut headers = HeaderMap::new();
    if let Some(v) = generation.warm_header.and_then(|w| w.to_string().parse().ok()) {
        headers.insert("x-superfluid-warm", v);
    }
    (StatusCode::OK, headers, Json(response)).into_response()
}

fn sse(kind: &str, v: &Value) -> Event {
    Event::default().event(kind).data(v.to_string())
}

/// Chat frames as Responses events. The generation is cancelled when the
/// client goes: this task then drops the frame receiver, which the chat
/// stream watches.
fn stream(generation: Generation, meta: Arc<Meta>, model: Arc<Daemon>, front: Arc<Daemon>) -> Response {
    let Body::Stream(mut rx) = generation.body else {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "a streamed response came complete".into());
    };
    let (tx, events) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::spawn(async move {
        let mut out = Output::new(&meta.id, &meta.tools);
        let started = meta.object("in_progress", Vec::new(), Value::Null, None, Value::Null);
        out.emit("response.created", json!({"response": started}));
        out.emit("response.in_progress", json!({"response": started}));
        loop {
            for (kind, v) in std::mem::take(&mut out.events) {
                if tx.send(sse(kind, &v)).await.is_err() {
                    return;
                }
            }
            let frame = tokio::select! {
                biased;
                _ = tx.closed() => return,
                f = rx.recv() => f,
            };
            let failure = match frame {
                Some(Ok(frame)) => {
                    if let Some(c) = frame.choices.first() {
                        out.feed(&c.message);
                    }
                    if frame.choices.first().and_then(|c| c.finish).is_none() {
                        continue;
                    }
                    let (meta, model, front) = (Arc::clone(&meta), Arc::clone(&model), Arc::clone(&front));
                    let done = tokio::task::spawn_blocking(move || {
                        let r = finalize(&meta, &model, front.responses(), &mut out, &frame);
                        (out, r)
                    })
                    .await;
                    match done {
                        Ok((done_out, Ok(response))) => {
                            out = done_out;
                            let kind = if response["status"] == "incomplete" { "response.incomplete" } else { "response.completed" };
                            out.emit(kind, json!({"response": response}));
                            for (kind, v) in std::mem::take(&mut out.events) {
                                let _ = tx.send(sse(kind, &v)).await;
                            }
                            return;
                        }
                        Ok((done_out, Err(e))) => {
                            out = done_out;
                            StreamFailure::new(e, false)
                        }
                        Err(e) => {
                            let _ = tx.send(sse("error", &json!({"type": "error", "message": e.to_string()}))).await;
                            return;
                        }
                    }
                }
                Some(Err(failure)) => failure,
                None => StreamFailure::new(DaemonError::Protocol("the generation ended without a result"), false),
            };
            let error = json!({"code": failure.code.unwrap_or(if failure.client_error { "invalid_request" } else { "server_error" }), "message": failure.message});
            out.emit("error", json!({"code": error["code"], "message": failure.message, "param": failure.param}));
            let failed = meta.object("failed", out.items.clone(), Value::Null, None, error);
            out.emit("response.failed", json!({"response": failed}));
            for (kind, v) in std::mem::take(&mut out.events) {
                let _ = tx.send(sse(kind, &v)).await;
            }
            return;
        }
    });
    let stream = ReceiverStream::new(events).map(Ok::<_, std::convert::Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

pub(crate) async fn retrieve(State(st): State<AppState>, key: Key, Path(id): Path<String>) -> Response {
    let front = st.default_daemon();
    let found = tokio::task::spawn_blocking(move || {
        let store = front.responses();
        store.get(&id).filter(|r| visible(r, caller(&key))).and_then(|_| store.body(&id)).ok_or(id)
    })
    .await;
    match found {
        Ok(Ok(body)) => Json(body.response).into_response(),
        Ok(Err(id)) => not_found(&id),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// Deletes a response and purges the session it ran in. A purge is a logical
/// removal: the session log keeps the events until its file is removed.
pub(crate) async fn delete(State(st): State<AppState>, key: Key, Path(id): Path<String>) -> Response {
    let front = st.default_daemon();
    let registry = Arc::clone(&st.registry);
    let asked = id.clone();
    let gone = tokio::task::spawn_blocking(move || {
        let store = front.responses();
        let Some(rec) = store.get(&id).filter(|r| visible(r, caller(&key))) else {
            return Ok(false);
        };
        if !store.delete(&id)? {
            return Ok(false);
        }
        // A model that is not loaded is not loaded to purge; its session stays.
        if let Some((_, d)) = registry.is_loaded(&rec.model).then(|| registry.resolve(Some(&rec.model))).flatten() {
            let generation = d.store().lock().expect("store").session(rec.session).map(|s| s.generation);
            if let Ok(g) = generation {
                if let Err(e) = d.purge(rec.session, g, crate::wal::PurgeMode::Reroot) {
                    tracing::warn!(response = %id, session = rec.session, "could not purge a deleted response's session: {e}");
                }
            }
        }
        Ok::<_, DaemonError>(true)
    })
    .await;
    match gone {
        Ok(Ok(true)) => Json(json!({"id": asked, "object": "response", "deleted": true})).into_response(),
        Ok(Ok(false)) => not_found(&asked),
        Ok(Err(e)) => crate::http_state::daemon_error(e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Deserialize, Default)]
pub(crate) struct ListQuery {
    limit: Option<usize>,
    order: Option<String>,
    after: Option<String>,
}

pub(crate) async fn input_items_list(
    State(st): State<AppState>,
    key: Key,
    Path(id): Path<String>,
    Query(q): Query<ListQuery>,
) -> Response {
    let limit = q.limit.unwrap_or(20);
    if !(1..=100).contains(&limit) {
        return refuse("limit", "invalid_value", "limit must be between 1 and 100");
    }
    let front = st.default_daemon();
    let found = tokio::task::spawn_blocking(move || {
        let store = front.responses();
        store.get(&id).filter(|r| visible(r, caller(&key))).and_then(|_| store.body(&id)).ok_or(id)
    })
    .await;
    let mut items = match found {
        Ok(Ok(body)) => body.input_items,
        Ok(Err(id)) => return not_found(&id),
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    match q.order.as_deref() {
        None | Some("desc") => items.reverse(),
        Some("asc") => {}
        Some(other) => return refuse("order", "invalid_value", format!("order must be asc or desc, got '{other}'")),
    }
    if let Some(after) = &q.after {
        let Some(at) = items.iter().position(|i| str_of(i, "id") == after) else {
            return refuse("after", "invalid_value", format!("no input item '{after}' in this response"));
        };
        items.drain(..=at);
    }
    let has_more = items.len() > limit;
    items.truncate(limit);
    let first = items.first().map(|i| i["id"].clone()).unwrap_or(Value::Null);
    let last = items.last().map(|i| i["id"].clone()).unwrap_or(Value::Null);
    Json(json!({"object": "list", "data": items, "first_id": first, "last_id": last, "has_more": has_more})).into_response()
}

pub(crate) async fn cancel(State(st): State<AppState>, key: Key, Path(id): Path<String>) -> Response {
    let front = st.default_daemon();
    if front.responses().get(&id).filter(|r| visible(r, caller(&key))).is_none() {
        return not_found(&id);
    }
    refuse("response_id", "unsupported_parameter",
        "only a background response can be cancelled, and background mode is not supported: a response is stored once it has finished")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(items: Value) -> Vec<ChatMessage> {
        let items = input_items(&items, "t").unwrap_or_else(|_| panic!("items"));
        let mut c = Conversation::default();
        assert!(c.push_items(&items, "input").is_ok());
        c.messages
    }

    #[test]
    fn a_string_input_is_one_user_message() {
        let items = input_items(&json!("hi"), "abc").unwrap_or_else(|_| panic!("items"));
        assert_eq!(items, vec![json!({
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}],
            "status": "completed", "id": "msg_abc_in0",
        })]);
    }

    #[test]
    fn calls_join_their_assistant_turn_and_reasoning_leads_it() {
        let m = conv(json!([
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
            {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "look it up"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
            {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call", "call_id": "c2", "name": "get_weather", "arguments": "{\"city\":\"Rome\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        ]));
        let roles: Vec<&str> = m.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool"]);
        assert_eq!(m[2].tool_calls.len(), 2);
        assert_eq!(m[2].reasoning_content.as_deref(), Some("look it up"));
        assert_eq!(m[3].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(m[3].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn built_in_tools_and_hosted_choices_are_refused() {
        assert!(chat_tools(&[json!({"type": "web_search"})]).is_err());
        assert!(chat_tool_choice(&Some(json!({"type": "file_search"}))).is_err());
        let t = chat_tools(&[json!({"type": "function", "name": "f", "parameters": {"type": "object"}})]).unwrap_or_default();
        assert_eq!(t, vec![json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}})]);
    }

    #[test]
    fn a_namespace_is_flattened_and_its_calls_go_back_out_under_it() {
        let tools = vec![
            json!({"type": "function", "name": "exec_command", "parameters": {"type": "object"}}),
            json!({"type": "namespace", "name": "multi_agent_v1", "description": "sub-agents", "tools": [
                {"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}},
                {"type": "function", "name": "close_agent", "parameters": {"type": "object"}},
            ]}),
        ];
        let chat = chat_tools(&tools).unwrap_or_default();
        let names: Vec<&str> = chat.iter().map(|t| t["function"]["name"].as_str().unwrap_or("")).collect();
        assert_eq!(names, ["exec_command", "multi_agent_v1.spawn_agent", "multi_agent_v1.close_agent"]);
        assert!(chat_tools(&[json!({"type": "namespace", "name": "n", "tools": [{"type": "web_search"}]})]).is_err());

        let mut out = Output::new("resp_x", &tools);
        out.feed(&Message::complete_tool(0, "c".into(), "multi_agent_v1.spawn_agent".into(), "{}".into()));
        out.feed(&Message::complete_tool(1, "d".into(), "exec_command".into(), "{}".into()));
        let items = out.finish(false);
        assert_eq!((items[0]["name"].as_str(), items[0]["namespace"].as_str()), (Some("spawn_agent"), Some("multi_agent_v1")));
        assert_eq!((items[1]["name"].as_str(), items[1].get("namespace")), (Some("exec_command"), None));
        let added = out.events.iter().find(|(k, _)| *k == "response.output_item.added").map(|(_, v)| v.clone()).unwrap_or_default();
        assert_eq!(added["item"]["namespace"], "multi_agent_v1");

        let m = conv(json!([
            {"role": "user", "content": "go"},
            {"type": "function_call", "call_id": "c", "namespace": "multi_agent_v1", "name": "spawn_agent", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": "ok"},
        ]));
        let call = m[1].tool_calls[0].function.as_ref().map(|f| f.name.clone());
        assert_eq!(call.as_deref(), Some("multi_agent_v1.spawn_agent"));
        assert_eq!(m[2].name.as_deref(), Some("multi_agent_v1.spawn_agent"));
    }

    #[test]
    fn stream_events_balance_and_number_in_order() {
        let mut out = Output::new("resp_x", &[]);
        out.feed(&Message::reasoning("hmm"));
        out.feed(&Message::content("Hel"));
        out.feed(&Message::content("lo"));
        out.feed(&Message::complete_tool(0, "call_9".into(), "f".into(), "{\"a\":".into()));
        out.feed(&Message { tools: vec![crate::generation_output::ToolDelta { index: 0, id: None, name: None, arguments: "1}".into() }], ..Message::default() });
        let items = out.finish(false);
        assert_eq!(items.len(), 3);
        assert_eq!(items[1]["content"][0]["text"], "Hello");
        assert_eq!(items[2]["arguments"], "{\"a\":1}");
        assert_eq!(items[2]["call_id"], "call_x_2");
        let seqs: Vec<u64> = out.events.iter().map(|(_, v)| v["sequence_number"].as_u64().unwrap_or(u64::MAX)).collect();
        assert_eq!(seqs, (0..seqs.len() as u64).collect::<Vec<_>>());
        let added = out.events.iter().filter(|(k, _)| *k == "response.output_item.added").count();
        let done = out.events.iter().filter(|(k, _)| *k == "response.output_item.done").count();
        assert_eq!((added, done), (3, 3));
    }
}
