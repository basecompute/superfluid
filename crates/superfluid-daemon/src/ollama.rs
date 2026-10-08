//! Ollama inference over the shared daemon, policy checks and generation core.

use std::collections::{BTreeMap, VecDeque};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::generation_output::{
    Body as GenerationBody, Frame, Generation, StreamFailure, ToolDelta, Usage,
};
use crate::http_generation;
use crate::http_state::{self, AppState, AuthenticatedCredential};
use crate::keypolicy::ClientKey;

pub(crate) fn routes() -> Router<AppState> {
    Router::new()
        .route("/", get(|| async { "superfluid is running" }))
        .route("/api/version", get(version))
        .route("/api/tags", get(tags))
        .route("/api/ps", get(ps))
        .route("/api/show", post(show))
        .route("/api/chat", post(chat))
        .route("/api/generate", post(generate))
        .route("/api/embed", post(embed))
        .route("/api/embeddings", post(legacy_embeddings))
        .route("/api/pull", any(unsupported_management))
        .route("/api/push", any(unsupported_management))
        .route("/api/create", any(unsupported_management))
        .route("/api/copy", any(unsupported_management))
        .route("/api/delete", any(unsupported_management))
        .route("/api/blobs/{digest}", any(unsupported_management))
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({"error": message.into()}))).into_response()
}

pub(crate) async fn error_layer(req: Request, next: Next) -> Response {
    let ollama = req.uri().path().starts_with("/api/");
    let response = next.run(req).await;
    if !ollama || response.status().is_success() {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, 1024 * 1024)
        .await
        .unwrap_or_default();
    let value = serde_json::from_slice::<Value>(&bytes).ok();
    let message = value
        .as_ref()
        .and_then(|v| {
            v["error"]
                .as_str()
                .or_else(|| v["error"]["message"].as_str())
        })
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if bytes.is_empty() {
                parts
                    .status
                    .canonical_reason()
                    .unwrap_or("request failed")
                    .to_owned()
            } else {
                String::from_utf8_lossy(&bytes).into_owned()
            }
        });
    parts.headers.remove(header::CONTENT_LENGTH);
    parts
        .headers
        .insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
    Response::from_parts(parts, Body::from(json!({"error": message}).to_string()))
}

async fn unsupported_management() -> Response {
    error(StatusCode::NOT_IMPLEMENTED, "Ollama model distribution and Modelfiles are not supported; load models with superfluid and use a model listed by /api/tags")
}

async fn version() -> Json<Value> {
    Json(
        json!({"version": env!("CARGO_PKG_VERSION"), "server": "superfluid", "compatibility": "ollama-inference"}),
    )
}

#[allow(clippy::result_large_err)]
fn parse<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, Response> {
    serde_json::from_slice(body)
        .map_err(|e| error(StatusCode::BAD_REQUEST, format!("invalid request: {e}")))
}

#[allow(clippy::result_large_err)]
fn model_name(st: &AppState, requested: &str) -> Result<String, Response> {
    if requested.trim().is_empty() {
        return Err(error(StatusCode::BAD_REQUEST, "model is required"));
    }
    let exists = |s: &str| st.registry.is_loaded(s) || st.registry.is_known(s);
    if exists(requested) {
        return Ok(requested.to_owned());
    }
    if let Some(base) = requested.strip_suffix(":latest").filter(|s| exists(s)) {
        return Ok(base.to_owned());
    }
    Err(error(
        StatusCode::NOT_FOUND,
        format!("model '{requested}' not found; choose a model from /api/tags"),
    ))
}

/// A map field sent as `null` is an empty one.
fn null_is_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Map<String, Value>, D::Error> {
    Ok(Option::<Map<String, Value>>::deserialize(d)?.unwrap_or_default())
}

fn reject_fields(extra: &Map<String, Value>) -> Result<(), String> {
    if let Some((key, _)) = extra.iter().find(|(_, v)| !v.is_null()) {
        return Err(format!("unsupported parameter: {key}"));
    }
    Ok(())
}

fn residency(value: &Option<Value>) -> Result<(), String> {
    if value.as_ref().is_some_and(|v| !v.is_null()) {
        return Err("keep_alive is not supported yet; model lifetime is controlled by superfluid --idle-timeout and its model-management API".into());
    }
    Ok(())
}

fn options(value: &Map<String, Value>, context: u32) -> Result<Map<String, Value>, String> {
    let mut mapped = Map::new();
    for (name, v) in value.iter().filter(|(_, v)| !v.is_null()) {
        match name.as_str() {
            "num_predict" => {
                let n = v.as_i64().ok_or("options.num_predict must be an integer")?;
                if n == -1 || n == -2 {
                    continue;
                }
                if !(1..=u32::MAX as i64).contains(&n) {
                    return Err("options.num_predict must be positive, -1 or -2 (fill the available context)".into());
                }
                mapped.insert("max_tokens".into(), json!(n));
            }
            "num_ctx" => {
                if v.as_u64() != Some(context as u64) {
                    return Err(format!("per-request options.num_ctx is not supported; this server uses {context} (--max-context)"));
                }
            }
            "seed" => {
                if v.as_i64() == Some(-1) {
                    continue;
                }
                if v.as_u64().is_none() {
                    return Err("options.seed must be non-negative or -1 (random)".into());
                }
                mapped.insert(name.clone(), v.clone());
            }
            "top_k" => {
                if v.as_u64().is_none_or(|n| n > u32::MAX as u64) {
                    return Err("options.top_k must be a non-negative 32-bit integer".into());
                }
                mapped.insert(name.clone(), v.clone());
            }
            "temperature" | "top_p" | "min_p" | "repeat_penalty" | "presence_penalty"
            | "frequency_penalty" => {
                let n = v
                    .as_f64()
                    .ok_or_else(|| format!("options.{name} must be a number"))?;
                let valid = match name.as_str() {
                    "top_p" | "min_p" => (0.0..=1.0).contains(&n),
                    "presence_penalty" | "frequency_penalty" => (-2.0..=2.0).contains(&n),
                    "repeat_penalty" => n > 0.0 && n <= f32::MAX as f64,
                    _ => n >= 0.0 && n <= f32::MAX as f64,
                };
                if !valid {
                    return Err(format!("options.{name} is out of range"));
                }
                mapped.insert(name.clone(), v.clone());
            }
            "stop" => {
                let stops = v
                    .as_array()
                    .ok_or("options.stop must be an array of strings")?;
                if stops.len() > 4 || stops.iter().any(|s| s.as_str().is_none_or(str::is_empty)) {
                    return Err("options.stop accepts up to four non-empty strings".into());
                }
                mapped.insert("stop".into(), v.clone());
            }
            _ => return Err(format!("unsupported option: {name}")),
        }
    }
    Ok(mapped)
}

fn format_and_thinking(
    format: Option<&Value>,
    think: Option<&Value>,
    out: &mut Map<String, Value>,
) -> Result<(), String> {
    if let Some(f) = format.filter(|v| !v.is_null() && v.as_str() != Some("")) {
        let format = if f == "json" {
            json!({"type":"json_schema", "json_schema":{"name":"ollama", "schema":{}}})
        } else if f.is_object() {
            json!({"type":"json_schema", "json_schema":{"name":"ollama", "schema":f}})
        } else {
            return Err("format must be 'json' or a JSON schema object".into());
        };
        out.insert("response_format".into(), format);
    }
    if let Some(t) = think.filter(|v| !v.is_null()) {
        if t.is_boolean() {
            out.insert("enable_thinking".into(), t.clone());
        } else if matches!(t.as_str(), Some("minimal" | "low" | "medium" | "high")) {
            out.insert("reasoning_effort".into(), t.clone());
        } else {
            return Err("think must be a boolean or a supported reasoning level (minimal, low, medium, high)".into());
        }
    }
    Ok(())
}

fn images_content(content: String, images: Vec<String>) -> Result<Value, String> {
    if images.is_empty() {
        return Ok(json!(content));
    }
    let mut parts = vec![json!({"type":"text", "text":content})];
    for image in images {
        crate::inference::base64_decode(&image)
            .map_err(|_| "images must contain base64-encoded image data")?;
        parts.push(json!({"type":"image_url", "image_url":{"url":format!("data:image/jpeg;base64,{image}")}}));
    }
    Ok(json!(parts))
}

#[derive(Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    images: Vec<String>,
    thinking: Option<String>,
    #[serde(default)]
    tool_calls: Vec<Value>,
    tool_name: Option<String>,
    tool_call_id: Option<String>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

fn messages(messages: Vec<Message>) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    let mut pending: Vec<(String, String)> = Vec::new();
    for (index, m) in messages.into_iter().enumerate() {
        reject_fields(&m.extra)?;
        let role = m.role.to_ascii_lowercase();
        if !matches!(role.as_str(), "system" | "user" | "assistant" | "tool") {
            return Err(format!("unsupported message role: {role}"));
        }
        if role != "assistant"
            && (!m.tool_calls.is_empty() || m.thinking.as_ref().is_some_and(|s| !s.is_empty()))
        {
            return Err("only assistant messages may contain thinking or tool_calls".into());
        }
        let mut v = json!({"role":role, "content":images_content(m.content, m.images)?});
        if let Some(t) = m.thinking {
            v["reasoning_content"] = json!(t);
        }
        let mut calls = Vec::new();
        for (i, call) in m.tool_calls.iter().enumerate() {
            let name = call["function"]["name"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or("tool_calls require function.name")?;
            let args = &call["function"]["arguments"];
            if !args.is_object() {
                return Err("tool call function.arguments must be an object".into());
            }
            let id = call["id"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("ollama_{index}_{i}"));
            if pending.iter().any(|(other, _)| other == &id) {
                return Err("duplicate pending tool call id".into());
            }
            pending.push((id.clone(), name.to_owned()));
            calls.push(json!({"id":id, "type":"function", "function":{"name":name,"arguments":args.to_string()}}));
        }
        if !calls.is_empty() {
            v["tool_calls"] = json!(calls);
        }
        if role == "tool" {
            let at = if let Some(id) = &m.tool_call_id {
                pending.iter().position(|(p, _)| p == id)
            } else if let Some(name) = &m.tool_name {
                pending.iter().position(|(_, n)| n == name)
            } else if pending.len() == 1 {
                Some(0)
            } else {
                None
            };
            let at = at.ok_or("tool result must identify a pending call using tool_name or tool_call_id (or have exactly one pending call)")?;
            let (id, name) = pending.remove(at);
            v["tool_call_id"] = json!(id);
            v["name"] = json!(name);
        }
        out.push(v);
    }
    Ok(out)
}

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    #[serde(default)]
    messages: Vec<Message>,
    #[serde(default)]
    tools: Vec<Value>,
    stream: Option<bool>,
    format: Option<Value>,
    think: Option<Value>,
    keep_alive: Option<Value>,
    #[serde(default, deserialize_with = "null_is_empty")]
    options: Map<String, Value>,
    logprobs: Option<bool>,
    top_logprobs: Option<u32>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

impl ChatRequest {
    fn into_generation(self, context: u32) -> Result<Value, String> {
        reject_fields(&self.extra)?;
        residency(&self.keep_alive)?;
        let mut out = options(&self.options, context)?;
        format_and_thinking(self.format.as_ref(), self.think.as_ref(), &mut out)?;
        for tool in &self.tools {
            if tool["type"] != "function"
                || tool["function"]["name"].as_str().is_none_or(str::is_empty)
                || !tool["function"]["parameters"].is_object()
            {
                return Err("tools require type 'function', function.name and an object function.parameters".into());
            }
        }
        if self.top_logprobs.is_some_and(|n| n > 20) {
            return Err("top_logprobs must be between 0 and 20".into());
        }
        out.insert("model".into(), json!(self.model));
        out.insert("messages".into(), json!(messages(self.messages)?));
        out.insert("tools".into(), json!(self.tools));
        out.insert("stream".into(), json!(self.stream.unwrap_or(true)));
        out.insert("logprobs".into(), json!(self.logprobs));
        out.insert("top_logprobs".into(), json!(self.top_logprobs));
        Ok(Value::Object(out))
    }
}

pub(crate) async fn chat(
    State(mut st): State<AppState>,
    key: Option<Extension<ClientKey>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: ChatRequest = match parse(&body) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let started = Instant::now();
    if matches!(
        req.options.get("num_predict").and_then(Value::as_i64),
        Some(-1 | -2)
    ) {
        st.limits.default_max_tokens = None;
    }
    let mut request = match req.into_generation(st.limits.max_context) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let model = match model_name(&st, request["model"].as_str().unwrap()) {
        Ok(v) => v,
        Err(e) => return e,
    };
    request["model"] = json!(model);
    if request["messages"].as_array().is_some_and(Vec::is_empty) {
        return preload(st, key, headers, model, Kind::Chat, started).await;
    }
    let request = match serde_json::from_value(request) {
        Ok(request) => request,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    let (hooks, watch) = crate::keepalive::watch();
    let generation = http_generation::chat(State(st), key, headers, request, hooks).await;
    watch.disarm();
    match generation {
        Ok(generation) => Reply::new(Kind::Chat, started).respond(generation),
        Err(response) => response,
    }
}

#[derive(Deserialize)]
struct GenerateRequest {
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    system: String,
    #[serde(default)]
    images: Vec<String>,
    suffix: Option<String>,
    /// A prompt template in place of the model's; only empty (the model's own) is served.
    #[serde(default)]
    template: String,
    #[serde(default)]
    raw: bool,
    stream: Option<bool>,
    format: Option<Value>,
    think: Option<Value>,
    keep_alive: Option<Value>,
    #[serde(default, deserialize_with = "null_is_empty")]
    options: Map<String, Value>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

async fn generate(
    State(mut st): State<AppState>,
    key: Option<Extension<ClientKey>>,
    peer: ConnectInfo<SocketAddr>,
    cred: Option<Extension<AuthenticatedCredential>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let mut req: GenerateRequest = match parse(&body) {
        Ok(v) => v,
        Err(e) => return e,
    };
    // The ollama CLI sends every string field, empty when unset.
    if req.suffix.as_deref() == Some("") {
        req.suffix = None;
    }
    if !req.template.is_empty() {
        return error(StatusCode::BAD_REQUEST, "template is not supported: the model's own chat template is used");
    }
    let started = Instant::now();
    if matches!(
        req.options.get("num_predict").and_then(Value::as_i64),
        Some(-1 | -2)
    ) {
        st.limits.default_max_tokens = None;
    }
    let translated = (|| -> Result<Value, String> {
        reject_fields(&req.extra)?;
        residency(&req.keep_alive)?;
        let mut out = options(&req.options, st.limits.max_context)?;
        out.insert("model".into(), json!(req.model));
        out.insert("stream".into(), json!(req.stream.unwrap_or(true)));
        if req.raw || req.suffix.is_some() {
            if !req.system.is_empty()
                || !req.images.is_empty()
                || req.format.as_ref().is_some_and(|v| !v.is_null())
                || req.think.as_ref().is_some_and(|v| !v.is_null())
                || out.contains_key("stop")
            {
                return Err("raw and suffix generation do not support system, images, format, think or options.stop".into());
            }
            out.insert("prompt".into(), json!(req.prompt));
            if let Some(s) = &req.suffix {
                out.insert("suffix".into(), json!(s));
            }
        } else {
            format_and_thinking(req.format.as_ref(), req.think.as_ref(), &mut out)?;
            let mut msgs = Vec::new();
            if !req.system.is_empty() {
                msgs.push(json!({"role":"system","content":req.system}));
            }
            msgs.push(json!({"role":"user","content":images_content(req.prompt.clone(), req.images.clone())?}));
            out.insert("messages".into(), json!(msgs));
        }
        Ok(Value::Object(out))
    })();
    let mut request = match translated {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let model = match model_name(&st, &req.model) {
        Ok(v) => v,
        Err(e) => return e,
    };
    request["model"] = json!(model);
    if req.prompt.is_empty()
        && req.system.is_empty()
        && req.images.is_empty()
        && req.suffix.is_none()
    {
        return preload(st, key, headers, model, Kind::Generate, started).await;
    }
    let (hooks, watch) = crate::keepalive::watch();
    let generation = if req.raw || req.suffix.is_some() {
        let request = match serde_json::from_value(request) {
            Ok(request) => request,
            Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
        };
        http_generation::completions(State(st), key, peer, cred, headers, request, hooks).await
    } else {
        let request = match serde_json::from_value(request) {
            Ok(request) => request,
            Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
        };
        http_generation::chat(State(st), key, headers, request, hooks).await
    };
    watch.disarm();
    match generation {
        Ok(generation) => Reply::new(Kind::Generate, started).respond(generation),
        Err(response) => response,
    }
}

async fn preload(
    st: AppState,
    key: Option<Extension<ClientKey>>,
    headers: HeaderMap,
    model: String,
    kind: Kind,
    started: Instant,
) -> Response {
    if let Err(e) = st.sync_class(&headers, &key) {
        return e;
    }
    if let Err(e) = st.resolve(Some(&model)) {
        return e;
    }
    let mut value = Reply::new(kind, started).base(&model, true);
    value["done_reason"] = json!("load");
    value["total_duration"] = json!(nanos(started.elapsed()));
    Json(value).into_response()
}

async fn tags(State(st): State<AppState>) -> Json<Value> {
    let modified = timestamp(SystemTime::now());
    Json(json!({"models": st.registry.all_names().iter().map(|name| tag_entry(name, &modified)).collect::<Vec<_>>()}))
}

/// One `/api/tags` row: a digest of the name (there is no manifest) and a size of 0.
fn tag_entry(name: &str, modified: &str) -> Value {
    use sha2::Digest;
    let digest: String = sha2::Sha256::digest(name.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    json!({"name": name, "model": name, "modified_at": modified, "size": 0, "digest": digest, "details": {"format": "", "family": "", "families": null, "parameter_size": "", "quantization_level": ""}})
}

async fn ps(State(st): State<AppState>) -> Json<Value> {
    Json(
        json!({"models":st.registry.names().iter().map(|name| json!({"name":name,"model":name,"context_length":st.limits.max_context})).collect::<Vec<_>>()}),
    )
}

#[derive(Deserialize)]
struct ShowRequest {
    // The ollama CLI sends both, the same; `model` is the current name, `name` the old one.
    #[serde(default)]
    model: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    verbose: bool,
    // What a modelfile would be with these; the CLI sends them (often empty) on every show.
    #[serde(default, rename = "system")]
    _system: Option<Value>,
    #[serde(default, rename = "template")]
    _template: Option<Value>,
    #[serde(default, rename = "options")]
    _options: Option<Value>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

async fn show(
    State(st): State<AppState>,
    key: Option<Extension<ClientKey>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: ShowRequest = match parse(&body) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if let Err(e) = reject_fields(&req.extra) {
        return error(StatusCode::BAD_REQUEST, e);
    }
    let requested = if req.model.is_empty() { &req.name } else { &req.model };
    let model = match model_name(&st, requested) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let class = match st.sync_class(&headers, &key) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let daemon = st.registry.peek_loaded(&model);
    let loaded = daemon.is_some();
    let result = tokio::task::spawn_blocking(move || {
        let _hold = key;
        crate::with_sync_class(class, || {
            let mut result = json!({"model_info":{},"details":{},"superfluid":{"model":model,"loaded":loaded}});
            if let Some(d) = daemon {
                let caps = d.capability_descriptor();
                let mut supported = Vec::new();
                if let Some(c) = &caps {
                    if c["workload"]["causal_generation"] == true { supported.push("completion"); }
                    if c["workload"]["embedding"] == true { supported.push("embedding"); }
                    if c["modalities"]["image_encode"] == true { supported.push("vision"); }
                    if let Some(arch) = c["architecture"].as_str() {
                        result["details"]["family"] = json!(arch);
                        result["model_info"]["general.architecture"] = json!(arch);
                        result["model_info"][format!("{arch}.context_length")] = json!(st.limits.max_context);
                    }
                }
                let probe = r#"{"type":"function","function":{"name":"capability_probe","parameters":{"type":"object","properties":{}}}}"#.to_owned();
                if d.tool_structural_tag(std::slice::from_ref(&probe), false).is_some() || d.tool_call_grammar(&[probe]).is_some() { supported.push("tools"); }
                if d.supports_enable_thinking() || d.supports_reasoning_effort() { supported.push("thinking"); }
                if d.fim_supported() { supported.push("insert"); }
                result["capabilities"] = json!(supported);
                if let Some(template) = d.chat_template() { result["template"] = json!(template); }
                if req.verbose { result["superfluid"]["capabilities"] = json!(caps); }
            }
            result
        })
    }).await;
    match result {
        Ok(v) => Json(v).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Deserialize)]
struct EmbedRequest {
    model: String,
    input: Option<Value>,
    prompt: Option<String>,
    truncate: Option<bool>,
    dimensions: Option<usize>,
    keep_alive: Option<Value>,
    #[serde(default, deserialize_with = "null_is_empty")]
    options: Map<String, Value>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

async fn embed(
    State(st): State<AppState>,
    key: Option<Extension<ClientKey>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    embeddings(st, key, headers, body, false).await
}
async fn legacy_embeddings(
    State(st): State<AppState>,
    key: Option<Extension<ClientKey>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    embeddings(st, key, headers, body, true).await
}
async fn embeddings(
    st: AppState,
    key: Option<Extension<ClientKey>>,
    headers: HeaderMap,
    body: Bytes,
    legacy: bool,
) -> Response {
    let started = Instant::now();
    let req: EmbedRequest = match parse(&body) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let valid = (|| -> Result<Vec<String>, String> {
        reject_fields(&req.extra)?;
        residency(&req.keep_alive)?;
        if req.options.iter().any(|(k, v)| {
            !v.is_null() && (k != "num_ctx" || v.as_u64() != Some(st.limits.max_context as u64))
        }) {
            return Err(
                "embedding options only support num_ctx equal to the configured --max-context"
                    .into(),
            );
        }
        if req.dimensions == Some(0) {
            return Err("dimensions must be positive".into());
        }
        if legacy {
            if req.input.is_some() || req.truncate.is_some() || req.dimensions.is_some() {
                return Err("use /api/embed for input, truncate and dimensions".into());
            }
            return Ok(req
                .prompt
                .clone()
                .filter(|s| !s.is_empty())
                .into_iter()
                .collect());
        }
        if req.prompt.is_some() {
            return Err("use input with /api/embed".into());
        }
        match &req.input {
            None => Ok(Vec::new()),
            Some(Value::String(s)) if s.is_empty() => Ok(Vec::new()),
            Some(Value::String(s)) => Ok(vec![s.clone()]),
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "input must contain only strings".into())
                })
                .collect(),
            _ => Err("input must be a string or an array of strings".into()),
        }
    })();
    let inputs = match valid {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let class = match st.sync_class(&headers, &key) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let name = match model_name(&st, &req.model) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let load_started = Instant::now();
    let (_, daemon) = match st.resolve(Some(&name)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let load = load_started.elapsed();
    let result = tokio::task::spawn_blocking(move || {
        let _hold = key;
        let mut vectors = Vec::new();
        let mut count = 0u64;
        for input in inputs {
            let mut tokens = daemon.tokenize(&input);
            if tokens.len() > st.limits.max_context as usize {
                if legacy || req.truncate == Some(false) {
                    return Err(crate::DaemonError::Constraint(
                        "input exceeds the configured context length".into(),
                    ));
                }
                tokens.truncate(st.limits.max_context as usize);
            }
            count += tokens.len() as u64;
            let mut vector = daemon.embed_tokens_as(tokens, class).map_err(|err| {
                let descriptor = daemon.capability_descriptor();
                match descriptor.as_ref().map(|d| &d["workload"]["embedding"]) {
                    Some(Value::String(reason)) => crate::DaemonError::Constraint(format!(
                        "model does not support embeddings: {reason}"
                    )),
                    Some(Value::Bool(false)) => {
                        crate::DaemonError::Constraint("model does not support embeddings".into())
                    }
                    _ => err,
                }
            })?;
            if !legacy {
                if let Some(dimensions) = req.dimensions {
                    if dimensions > vector.len() {
                        return Err(crate::DaemonError::Constraint(format!(
                            "dimensions exceeds this model's embedding size ({})",
                            vector.len()
                        )));
                    }
                    vector.truncate(dimensions);
                }
                let norm = vector
                    .iter()
                    .map(|v| (*v as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if norm > 0.0 {
                    for v in &mut vector {
                        *v = (*v as f64 / norm) as f32;
                    }
                }
            }
            vectors.push(vector);
        }
        Ok::<_, crate::DaemonError>((vectors, count))
    })
    .await;
    match result {
        Ok(Ok((vectors, count))) => Json(if legacy {
            json!({"embedding":vectors.into_iter().next().unwrap_or_default()})
        } else {
            json!({"model":name,"embeddings":vectors,"total_duration":nanos(started.elapsed()),"load_duration":nanos(load),"prompt_eval_count":count})
        }).into_response(),
        Ok(Err(e)) => http_state::daemon_error(e),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Chat,
    Generate,
}

struct Reply {
    kind: Kind,
    started: Instant,
    tools: BTreeMap<u64, Tool>,
    tool_bytes: usize,
    ended: bool,
}
#[derive(Default)]
struct Tool {
    id: String,
    name: String,
    arguments: String,
}

impl Reply {
    fn new(kind: Kind, started: Instant) -> Self {
        Self {
            kind,
            started,
            tools: BTreeMap::new(),
            tool_bytes: 0,
            ended: false,
        }
    }

    fn base(&self, model: &str, done: bool) -> Value {
        let mut value =
            json!({"model":model, "created_at":timestamp(SystemTime::now()), "done":done});
        match self.kind {
            Kind::Chat => value["message"] = json!({"role":"assistant","content":""}),
            Kind::Generate => value["response"] = json!(""),
        }
        value
    }

    fn content(&self, value: &mut Value, content: &str, thinking: Option<&str>) {
        match self.kind {
            Kind::Chat => {
                value["message"]["content"] = json!(content);
                if let Some(s) = thinking.filter(|s| !s.is_empty()) {
                    value["message"]["thinking"] = json!(s);
                }
            }
            Kind::Generate => {
                value["response"] = json!(content);
                if let Some(s) = thinking.filter(|s| !s.is_empty()) {
                    value["thinking"] = json!(s);
                }
            }
        }
    }

    fn terminal(&mut self, value: &mut Value, usage: Option<&Usage>, reason: &str) {
        value["done"] = json!(true);
        value["done_reason"] = json!(if reason == "tool_calls" {
            "stop"
        } else {
            reason
        });
        value["total_duration"] = json!(nanos(self.started.elapsed()));
        if let Some(usage) = usage {
            if let Some(prompt) = usage.prompt {
                value["prompt_eval_count"] = json!(prompt);
            }
            value["eval_count"] = json!(usage.completion);
            if let Some(cached) = usage.cached {
                value["prompt_eval_cached_count"] = json!(cached);
            }
        }
        self.ended = true;
    }

    fn tool_values(&mut self) -> Result<Vec<Value>, String> {
        std::mem::take(&mut self.tools).into_iter().map(|(index, tool)| {
            let arguments: Value = serde_json::from_str(&tool.arguments).map_err(|_| "model produced incomplete or invalid tool-call arguments; no tool call was emitted")?;
            if tool.name.is_empty() || !arguments.is_object() { return Err("model produced an invalid tool call".into()); }
            Ok(json!({"id":tool.id,"function":{"index":index,"name":tool.name,"arguments":arguments}}))
        }).collect()
    }

    fn accept_tools(&mut self, calls: &[ToolDelta]) -> Result<(), String> {
        for call in calls {
            let tool = self.tools.entry(call.index as u64).or_default();
            if let Some(id) = &call.id {
                tool.id.clone_from(id);
            }
            if let Some(name) = &call.name {
                tool.name.push_str(name);
                self.tool_bytes += name.len();
            }
            tool.arguments.push_str(&call.arguments);
            self.tool_bytes += call.arguments.len();
            if self.tool_bytes > 8 * 1024 * 1024 || self.tools.len() > 1024 {
                return Err("tool-call output exceeds the response limit".into());
            }
        }
        Ok(())
    }

    fn encode(
        &mut self,
        model: &str,
        source: Frame,
        streaming: bool,
    ) -> Result<Vec<Value>, String> {
        let choice = source
            .choices
            .first()
            .ok_or("generation returned no choice")?;
        let message = &choice.message;
        let text = choice
            .text
            .as_deref()
            .or(message.content.as_deref())
            .unwrap_or("");
        let mut value = self.base(model, false);
        self.content(&mut value, text, message.reasoning.as_deref());
        if choice.text.is_none() {
            if let Some(lp) = &choice.logprobs {
                let token = |text: &str, lp: f32| json!({"token":text,"logprob":lp,"bytes":text.as_bytes()});
                value["logprobs"] = json!(lp
                    .tokens
                    .iter()
                    .map(|t| {
                        let mut v = token(&t.token, t.logprob);
                        v["top_logprobs"] = json!(t
                            .alternatives
                            .iter()
                            .map(|(text, lp)| token(text, *lp))
                            .collect::<Vec<_>>());
                        v
                    })
                    .collect::<Vec<_>>());
            }
        }
        self.accept_tools(&message.tools)?;
        if let Some(reason) = choice.finish {
            let tools = self.tool_values()?;
            let mut frames = Vec::new();
            if !tools.is_empty() {
                if !matches!(self.kind, Kind::Chat) {
                    return Err("a generate response cannot contain tool calls".into());
                }
                if streaming {
                    let mut calls = self.base(model, false);
                    calls["message"]["tool_calls"] = json!(tools);
                    frames.push(calls);
                } else {
                    value["message"]["tool_calls"] = json!(tools);
                }
            }
            self.terminal(&mut value, source.usage.as_ref(), reason);
            frames.push(value);
            Ok(frames)
        } else if !text.is_empty()
            || message.reasoning.as_deref().is_some_and(|s| !s.is_empty())
            || message.assistant
        {
            Ok(vec![value])
        } else {
            Ok(Vec::new())
        }
    }

    fn respond(self, generation: Generation) -> Response {
        match generation.body {
            GenerationBody::Complete(frame) => {
                let mut headers = HeaderMap::new();
                if let Some(warm) = generation.warm_header {
                    if let Ok(value) = warm.to_string().parse() {
                        headers.insert("x-superfluid-warm", value);
                    }
                }
                self.json(&generation.model, frame, headers)
            }
            GenerationBody::Stream(rx) => self.stream(generation.model, rx),
        }
    }

    fn json(mut self, model: &str, source: Frame, headers: HeaderMap) -> Response {
        match self.encode(model, source, false) {
            Ok(mut values) if self.ended && values.len() == 1 => {
                (headers, Json(values.remove(0))).into_response()
            }
            Ok(_) => error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation ended without a final response",
            ),
            Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }

    fn stream(
        self,
        model: String,
        rx: tokio::sync::mpsc::Receiver<Result<Frame, StreamFailure>>,
    ) -> Response {
        let stream = futures_util::stream::unfold(
            (self, model, rx, VecDeque::new()),
            |(mut reply, model, mut rx, mut pending)| async move {
                loop {
                    if let Some(value) = pending.pop_front() {
                        let bytes = Bytes::from(format!("{value}\n"));
                        return Some((Ok::<_, Infallible>(bytes), (reply, model, rx, pending)));
                    }
                    if reply.ended {
                        return None;
                    }
                    let encoded = match rx.recv().await {
                        Some(Ok(value)) => reply.encode(&model, value, true),
                        Some(Err(error)) => Err(error.message),
                        None => Err("generation ended without a final response".into()),
                    };
                    match encoded {
                        Ok(values) => pending.extend(values),
                        Err(message) => {
                            reply.ended = true;
                            rx.close();
                            pending.push_back(json!({"error":message}));
                        }
                    }
                }
            },
        );
        (
            [(header::CONTENT_TYPE, "application/x-ndjson")],
            Body::from_stream(stream),
        )
            .into_response()
    }
}

fn nanos(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

pub(crate) fn timestamp(time: SystemTime) -> String {
    let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = duration.as_secs().min(libc::time_t::MAX as u64) as libc::time_t;
    let mut tm = std::mem::MaybeUninit::<libc::tm>::uninit();
    // SAFETY: seconds and tm are valid pointers to appropriately sized
    // objects; gmtime_r initializes tm on success and retains neither.
    let tm = unsafe {
        if libc::gmtime_r(&seconds, tm.as_mut_ptr()).is_null() {
            return "1970-01-01T00:00:00Z".into();
        }
        tm.assume_init()
    };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}Z",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        duration.subsec_nanos()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation_output::Message as OutputMessage;

    #[test]
    fn tool_history_preserves_structured_arguments_and_pairs_results() {
        let messages: Vec<Message> = serde_json::from_value(json!([
            {"role":"user","content":"weather"},
            {"role":"assistant","thinking":"check both","tool_calls":[
                {"function":{"name":"weather","arguments":{"city":"東京"}}},
                {"function":{"name":"clock","arguments":{}}}
            ]},
            {"role":"tool","tool_name":"clock","content":"noon"},
            {"role":"tool","content":"sunny"}
        ]))
        .unwrap();
        let translated = super::messages(messages).unwrap();
        assert_eq!(translated[1]["reasoning_content"], "check both");
        assert_eq!(
            translated[2]["tool_call_id"],
            translated[1]["tool_calls"][1]["id"]
        );
        assert_eq!(
            translated[3]["tool_call_id"],
            translated[1]["tool_calls"][0]["id"]
        );
        assert_eq!(
            serde_json::from_str::<Value>(
                translated[1]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap()
            )
            .unwrap(),
            json!({"city":"東京"})
        );
    }

    #[test]
    fn incomplete_tools_never_become_executable_calls() {
        let mut reply = Reply::new(Kind::Chat, Instant::now());
        assert!(reply
            .encode(
                "m",
                Frame::chat(
                    OutputMessage::complete_tool(0, "c".into(), "edit".into(), "{\"path\":".into()),
                    None
                ),
                true
            )
            .unwrap()
            .is_empty());
        let error = reply
            .encode(
                "m",
                Frame::chat(OutputMessage::default(), Some("length")),
                true,
            )
            .unwrap_err();
        assert!(error.contains("incomplete or invalid"));
    }

    #[test]
    fn tool_fragments_become_one_object_before_the_terminal_record() {
        let mut reply = Reply::new(Kind::Chat, Instant::now());
        for (name, arguments) in [("weather", "{\"city\":"), ("", "\"Paris\"}")] {
            assert!(reply
                .encode(
                    "m",
                    Frame::chat(
                        OutputMessage::complete_tool(0, "c".into(), name.into(), arguments.into()),
                        None
                    ),
                    true
                )
                .unwrap()
                .is_empty());
        }
        let mut terminal = Frame::chat(OutputMessage::default(), Some("tool_calls"));
        terminal.usage = Some(Usage::complete(3, 7, None));
        let frames = reply.encode("m", terminal, true).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0]["done"], false);
        assert_eq!(
            frames[0]["message"]["tool_calls"][0]["function"]["arguments"],
            json!({"city":"Paris"})
        );
        assert_eq!(frames[1]["done"], true);
        assert_eq!(frames[1]["done_reason"], "stop");
        assert_eq!(frames[1]["eval_count"], 7);
    }

    #[test]
    fn generated_thinking_and_json_schema_are_mapped_without_stringifying_the_schema() {
        let request: ChatRequest = serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":"hi"}],"think":false,"format":{"type":"object","properties":{"x":{"type":"integer"}}},"options":{"num_predict":32,"seed":-1,"temperature":0}})).unwrap();
        let mapped = request.into_generation(128).unwrap();
        assert_eq!(mapped["enable_thinking"], false);
        assert_eq!(
            mapped["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
        assert_eq!(mapped["max_tokens"], 32);
        assert!(mapped.get("seed").is_none());
        assert_eq!(mapped["stream"], true);
        let mut reply = Reply::new(Kind::Generate, Instant::now());
        let mut result = Frame::chat(
            OutputMessage {
                content: Some("answer".into()),
                reasoning: Some("reason".into()),
                ..OutputMessage::default()
            },
            Some("stop"),
        );
        result.usage = Some(Usage::complete(9, 5, None));
        let output = reply.encode("m", result, false).unwrap();
        assert_eq!(output[0]["response"], "answer");
        assert_eq!(output[0]["thinking"], "reason");
        assert!(output[0].get("message").is_none());
    }

    #[tokio::test]
    async fn dropping_the_ndjson_body_closes_the_generation_channel() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let response = Reply::new(Kind::Chat, Instant::now()).stream("m".into(), rx);
        drop(response);
        assert!(tx.send(Ok(Frame::default())).await.is_err());
    }

    #[tokio::test]
    async fn a_producer_error_or_early_close_is_a_single_ndjson_error() {
        for event in [
            None,
            Some(Err(StreamFailure {
                code: None,
                message: "worker failed".into(),
                client_error: false,
                param: None,
            })),
        ] {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            if let Some(event) = event {
                tx.send(event).await.unwrap();
            }
            drop(tx);
            let response = Reply::new(Kind::Generate, Instant::now()).stream("m".into(), rx);
            assert_eq!(
                response.headers()[header::CONTENT_TYPE],
                "application/x-ndjson"
            );
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert_eq!(text.lines().count(), 1);
            let value: Value = serde_json::from_str(text).unwrap();
            assert!(value["error"].is_string());
            assert!(value.get("done").is_none());
        }
    }

    #[test]
    fn options_reject_unsupported_or_invalid_values() {
        for bad in [
            json!({"num_gpu":1}),
            json!({"num_ctx":64}),
            json!({"temperature":-1}),
            json!({"min_p":2}),
            json!({"num_predict":0}),
            json!({"seed":-2}),
            json!({"stop":[""]}),
        ] {
            assert!(options(bad.as_object().unwrap(), 128).is_err(), "{bad}");
        }
        assert!(options(
            json!({"num_ctx":128,"num_predict":-1,"seed":-1})
                .as_object()
                .unwrap(),
            128
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn null_options_are_no_options() {
        let req: ChatRequest = serde_json::from_str(r#"{"model":"m","messages":[],"options":null}"#).unwrap();
        assert!(req.options.is_empty());
        let req: GenerateRequest = serde_json::from_str(r#"{"model":"m","options":null,"template":"","suffix":""}"#).unwrap();
        assert!(req.options.is_empty() && req.extra.is_empty());
    }

    #[test]
    fn a_tag_row_carries_what_the_ollama_cli_prints() {
        let row = tag_entry("Qwen3-4B", "2026-01-01T00:00:00.000000000Z");
        assert_eq!(row["digest"].as_str().map(str::len), Some(64));
        assert_eq!(row["size"], 0);
        assert_eq!(row["modified_at"], "2026-01-01T00:00:00.000000000Z");
    }

    #[test]
    fn show_takes_the_name_and_the_model_together_as_the_cli_sends_them() {
        let req: ShowRequest = serde_json::from_str(r#"{"name":"m","model":"m","system":"","options":{}}"#).unwrap();
        assert!(req.extra.is_empty());
        assert_eq!((req.model.as_str(), req.name.as_str()), ("m", "m"));
        let req: ShowRequest = serde_json::from_str(r#"{"name":"old"}"#).unwrap();
        assert_eq!((req.model.as_str(), req.name.as_str()), ("", "old"));
    }

    #[test]
    fn timestamps_are_utc_rfc3339_including_leap_days() {
        assert_eq!(timestamp(UNIX_EPOCH), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(
            timestamp(UNIX_EPOCH + std::time::Duration::from_secs(1709164800)),
            "2024-02-29T00:00:00.000000000Z"
        );
    }
}
