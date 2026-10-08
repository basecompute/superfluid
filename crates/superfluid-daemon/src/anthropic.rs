//! api-anthropic.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use superfluid_abi::finish;

use crate::http_state::AppState;
use crate::block_inference::{self as blocks, Breakpoint, BreakpointAt, Built};
use crate::wal::{channel, role, EventBody};
use crate::{Daemon, DaemonError};

#[derive(Deserialize)]
struct MessagesRequest {
    model: Option<String>,
    max_tokens: u32,
    #[serde(default)]
    messages: Vec<Message>,
    #[serde(default)]
    system: Option<serde_json::Value>,
    #[serde(default)]
    tools: Vec<AnthropicTool>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    content: serde_json::Value,
}

#[derive(Deserialize)]
struct AnthropicTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_schema: serde_json::Value,
    #[serde(default)]
    cache_control: Option<serde_json::Value>,
}

const MAX_BREAKPOINTS: usize = 4;
const TTL_5M_MS: u64 = 5 * 60 * 1000;
const TTL_1H_MS: u64 = 60 * 60 * 1000;

fn cache_control_ttl(v: &serde_json::Value) -> Result<u64, String> {
    let obj = v
        .as_object()
        .ok_or_else(|| "cache_control: expected an object".to_string())?;
    for key in obj.keys() {
        if key != "type" && key != "ttl" {
            return Err(format!(
                "cache_control.{key}: Extra inputs are not permitted"
            ));
        }
    }
    match obj.get("type").and_then(|t| t.as_str()) {
        Some("ephemeral") => {}
        Some(other) => {
            return Err(format!(
                "cache_control.type: Input should be 'ephemeral' (got '{other}')"
            ))
        }
        None => return Err("cache_control.type: Field required".into()),
    }
    match obj.get("ttl") {
        None => Ok(TTL_5M_MS),
        Some(t) => match t.as_str() {
            Some("5m") => Ok(TTL_5M_MS),
            Some("1h") => Ok(TTL_1H_MS),
            _ => Err(format!(
                "cache_control.ttl: Input should be '5m' or '1h' (got {t})"
            )),
        },
    }
}

fn block_breakpoints(
    blocks: &[serde_json::Value],
    at: BreakpointAt,
    out: &mut Vec<Breakpoint>,
) -> Result<(), String> {
    for b in blocks {
        if b.get("type").and_then(|t| t.as_str()) == Some("tool_result") {
            if let Some(serde_json::Value::Array(inner)) = b.get("content") {
                block_breakpoints(inner, at, out)?;
            }
        }
        if let Some(cc) = b.get("cache_control").filter(|v| !v.is_null()) {
            let kind = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if kind == "thinking" || kind == "redacted_thinking" {
                return Err(format!("cache_control cannot be set on {kind} blocks"));
            }
            if kind == "text"
                && b.get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .is_empty()
            {
                return Err("cache_control cannot be set for empty text blocks".into());
            }
            out.push(Breakpoint {
                at,
                ttl_ms: cache_control_ttl(cc)?,
            });
        }
    }
    Ok(())
}

fn cache_breakpoints(req: &MessagesRequest) -> Result<Vec<Breakpoint>, String> {
    let mut out = Vec::new();
    for t in &req.tools {
        if let Some(cc) = t.cache_control.as_ref().filter(|v| !v.is_null()) {
            out.push(Breakpoint {
                at: BreakpointAt::FirstTurn,
                ttl_ms: cache_control_ttl(cc)?,
            });
        }
    }
    if let Some(serde_json::Value::Array(blocks)) = &req.system {
        block_breakpoints(blocks, BreakpointAt::FirstTurn, &mut out)?;
    }
    for (k, m) in req.messages.iter().enumerate() {
        if let serde_json::Value::Array(blocks) = &m.content {
            block_breakpoints(blocks, BreakpointAt::Message(k), &mut out)?;
        }
    }
    if out.len() > MAX_BREAKPOINTS {
        return Err(format!(
            "A maximum of {MAX_BREAKPOINTS} blocks with cache_control may be provided. Found {}.",
            out.len()
        ));
    }
    let mut seen_5m = false;
    for b in &out {
        if b.ttl_ms == TTL_5M_MS {
            seen_5m = true;
        } else if seen_5m {
            return Err(
                "a ttl='1h' cache_control block must not come after a ttl='5m' cache_control block"
                    .into(),
            );
        }
    }
    Ok(out)
}

pub(crate) fn error_response(status: StatusCode, message: String) -> Response {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "api_error"
    };
    (
        status,
        axum::Json(serde_json::json!({
            "type": "error",
            "error": {"type": kind, "message": message},
        })),
    )
        .into_response()
}

fn daemon_error(e: DaemonError) -> Response {
    error_response(StatusCode::BAD_REQUEST, e.to_string())
}

fn system_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn build_session(daemon: &Daemon, req: &MessagesRequest, op: &crate::inference::SamplingOverrides, model: &crate::ModelSamplingDefaults, breakpoints: &[Breakpoint]) -> Result<Built, DaemonError> {
    blocks::build_session(daemon, &normalize(req)?, op, model, breakpoints)
}

fn normalize(req: &MessagesRequest) -> Result<blocks::BlockPrompt, DaemonError> {
    use blocks::{Block, BlockMessage, BlockPrompt, Content};
    let mut messages = Vec::with_capacity(req.messages.len());
    for message in &req.messages {
        let content = match &message.content {
            serde_json::Value::String(text) => Content::Text(text.clone()),
            serde_json::Value::Array(values) => {
                let mut blocks = Vec::new();
                for value in values {
                    let text = |key: &str| value.get(key).and_then(|v| v.as_str()).unwrap_or_default().to_owned();
                    let block = match value.get("type").and_then(|v| v.as_str()) {
                        Some("text") => Block::Text(text("text")),
                        Some("image") => {
                            let source = &value["source"];
                            if source["type"].as_str() != Some("base64") {
                                return Err(DaemonError::Protocol("image blocks must carry a base64 source (URLs are not fetched)"));
                            }
                            Block::Image(crate::inference::base64_decode(source["data"].as_str().unwrap_or_default())?)
                        }
                        Some("tool_use") => Block::ToolCall { id: text("id"), name: text("name"), arguments: value.get("input").cloned().unwrap_or(serde_json::json!({})).to_string() },
                        Some("tool_result") => Block::ToolResult { id: value.get("tool_use_id").and_then(|v| v.as_str()).map(str::to_owned), text: value.get("content").map(system_text).unwrap_or_default() },
                        Some("thinking") => Block::Reasoning(value.get("thinking").and_then(|v| v.as_str()).map(str::to_owned)),
                        _ => continue,
                    };
                    blocks.push(block);
                }
                Content::Blocks(blocks)
            }
            _ => Content::Empty,
        };
        messages.push(BlockMessage { role: if message.role == "assistant" { role::ASSISTANT } else { role::USER }, content });
    }
    Ok(BlockPrompt {
        system: req.system.as_ref().map(system_text), messages,
        tools: req.tools.iter().map(|t| serde_json::json!({"type":"function","function":{"name":t.name,"description":t.description.clone().unwrap_or_default(),"parameters":t.input_schema}}).to_string()).collect(),
        temperature: req.temperature, top_p: req.top_p, top_k: req.top_k,
    })
}

fn blocks_of(events: &[crate::CommittedEvent]) -> (Vec<serde_json::Value>, bool) {
    let mut blocks: Vec<serde_json::Value> = Vec::new();
    let mut had_tool = false;
    let mut cur_channel: Option<u32> = None;
    let mut buf = String::new();
    let flush = |blocks: &mut Vec<serde_json::Value>, ch: Option<u32>, buf: &mut String| {
        if buf.is_empty() {
            return;
        }
        let b = std::mem::take(buf);
        match ch {
            Some(channel::REASONING) => blocks.push(serde_json::json!({
                "type": "thinking", "thinking": b, "signature": "",
            })),
            Some(channel::TEXT) => blocks.push(serde_json::json!({
                "type": "text", "text": b,
            })),
            _ => {}
        }
    };
    for e in events {
        match &e.body {
            EventBody::Generated { text, channel: ch, .. } => {
                if *ch == channel::TOOL_CALL {
                    continue;
                }
                if cur_channel != Some(*ch) {
                    flush(&mut blocks, cur_channel, &mut buf);
                    cur_channel = Some(*ch);
                }
                buf.push_str(text);
            }
            EventBody::ToolUse { name, arguments } => {
                flush(&mut blocks, cur_channel, &mut buf);
                cur_channel = None;
                had_tool = true;
                let input: serde_json::Value =
                    serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));
                blocks.push(serde_json::json!({
                    "type": "tool_use",
                    "id": format!("toolu_{}", e.event_id),
                    "name": name,
                    "input": input,
                }));
            }
            EventBody::ToolParseFailure { raw } => {
                flush(&mut blocks, cur_channel, &mut buf);
                cur_channel = None;
                blocks.push(serde_json::json!({"type": "text", "text": raw}));
            }
            _ => {}
        }
    }
    flush(&mut blocks, cur_channel, &mut buf);
    (blocks, had_tool)
}

fn stop_reason(fin: u32, produced: u32, max_tokens: u32, had_tool: bool) -> &'static str {
    let cut = fin == finish::LENGTH || (fin != finish::EOS && produced >= max_tokens);
    if cut {
        "max_tokens"
    } else if had_tool {
        "tool_use"
    } else {
        "end_turn"
    }
}

fn usage_json(usage: &crate::generation_output::Usage, pinned: u64) -> serde_json::Value {
    let prompt = usage.prompt.unwrap_or(0);
    let read = usage.cached.unwrap_or(0).min(prompt);
    let created = pinned.min(prompt).saturating_sub(read);
    serde_json::json!({"input_tokens":prompt - read - created,"cache_creation_input_tokens":created,"cache_read_input_tokens":read,"output_tokens":usage.completion})
}

fn next_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(1);
    format!("msg_superfluid_{}", N.fetch_add(1, Ordering::Relaxed))
}

pub(crate) async fn messages(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let req: MessagesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    if req.model.is_none() {
        return error_response(StatusCode::BAD_REQUEST, "missing required field: model".into());
    }
    if req.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages must be non-empty".into());
    }
    let breakpoints = match cache_breakpoints(&req) {
        Ok(b) => b,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, message),
    };
    let qos = match st.qos.resolve(&headers, key.as_ref().map(|k| &*k.0 .0)) {
        Ok(q) => q,
        Err(resp) => return resp,
    };
    let (model_name, daemon) = match st.resolve(req.model.as_deref()) {
        Ok(pair) => pair,
        Err(resp) => return resp,
    };
    let id = next_id();

    if req.stream {
        return messages_stream(daemon, model_name, req, id, st.sampling, qos, breakpoints, key.map(|k| k.0)).await;
    }

    let op = st.sampling;
    let hold = key.map(|k| k.0);
    crate::keepalive::respond(st.nonstream_keepalive, move |hooks| {
        messages_complete(daemon, model_name, req, id, op, qos, breakpoints, hold, hooks)
    })
    .await
}

#[allow(clippy::too_many_arguments)]
async fn messages_complete(
    daemon: Arc<Daemon>, model_name: String, req: MessagesRequest, id: String,
    op: crate::inference::SamplingOverrides, qos: crate::inference::RequestQos,
    breakpoints: Vec<Breakpoint>, hold: Option<crate::keypolicy::ClientKey>,
    hooks: crate::keepalive::Hooks,
) -> Response {
    let max_tokens = req.max_tokens;
    let out = crate::inference::blocking(qos, hold, move || {
        let mut hooks = hooks;
        let model_defaults = daemon.model_sampling_defaults();
        let built = build_session(&daemon, &req, &op, &model_defaults, &breakpoints)?;
        qos.apply(&daemon, built.session)?;
        if hooks.is_on() {
            daemon.session_fits(built.session, None)?;
        }
        hooks.admitted(&daemon, built.session);
        hooks.check()?;
        blocks::generate(&daemon, &built, &op, &model_defaults, max_tokens, |_| {
            hooks.cancel_if_gone();
            Ok(())
        })
    }).await;
    let blocks::Outcome { generated: out, usage, pinned } = match out {
        Ok(out) => out,
        Err(crate::inference::RunError::Daemon(e)) => return daemon_error(e),
        Err(crate::inference::RunError::Worker(e)) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let usage = usage_json(&usage, pinned);
    let (content, had_tool) = blocks_of(&out.events);
    let json = serde_json::json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model_name.clone(),
        "content": content,
        "stop_reason": stop_reason(out.finish, out.tokens_generated, max_tokens, had_tool),
        "stop_sequence": null,
        "usage": usage,
    });
    let mut headers = HeaderMap::new();
    if let Ok(v) = out.warm_prefix.to_string().parse() {
        headers.insert("x-superfluid-warm", v);
    }
    (StatusCode::OK, headers, axum::Json(json)).into_response()
}

struct BlockStream {
    index: i64,
    open: bool,
    open_channel: Option<u32>,
    had_tool: bool,
}

type SseEvents = Vec<(&'static str, serde_json::Value)>;

impl BlockStream {
    fn new() -> BlockStream {
        BlockStream {
            index: -1,
            open: false,
            open_channel: None,
            had_tool: false,
        }
    }

    fn close_if_open(&mut self, out: &mut SseEvents) {
        if self.open {
            out.push((
                "content_block_stop",
                serde_json::json!({"type": "content_block_stop", "index": self.index}),
            ));
            self.open = false;
            self.open_channel = None;
        }
    }

    fn open_block(&mut self, block: serde_json::Value, channel: Option<u32>, out: &mut SseEvents) {
        self.index += 1;
        self.open = true;
        self.open_channel = channel;
        out.push((
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": self.index,
                "content_block": block,
            }),
        ));
    }

    fn on_event(&mut self, event_id: u64, body: &EventBody) -> SseEvents {
        let mut out = SseEvents::new();
        match body {
            EventBody::Generated { text, channel: ch, .. } => {
                if *ch == channel::TOOL_CALL {
                    return out;
                }
                if self.open_channel != Some(*ch) || !self.open {
                    self.close_if_open(&mut out);
                    let block = if *ch == channel::REASONING {
                        serde_json::json!({"type": "thinking", "thinking": "", "signature": ""})
                    } else {
                        serde_json::json!({"type": "text", "text": ""})
                    };
                    self.open_block(block, Some(*ch), &mut out);
                }
                if !text.is_empty() {
                    let delta = if *ch == channel::REASONING {
                        serde_json::json!({"type": "thinking_delta", "thinking": text})
                    } else {
                        serde_json::json!({"type": "text_delta", "text": text})
                    };
                    out.push((
                        "content_block_delta",
                        serde_json::json!({
                            "type": "content_block_delta",
                            "index": self.index,
                            "delta": delta,
                        }),
                    ));
                }
            }
            EventBody::ToolUse { name, arguments } => {
                self.close_if_open(&mut out);
                self.had_tool = true;
                self.open_block(
                    serde_json::json!({
                        "type": "tool_use",
                        "id": format!("toolu_{event_id}"),
                        "name": name,
                        "input": {},
                    }),
                    None,
                    &mut out,
                );
                out.push((
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": self.index,
                        "delta": {"type": "input_json_delta", "partial_json": arguments},
                    }),
                ));
                self.close_if_open(&mut out);
            }
            EventBody::ToolParseFailure { raw } => {
                self.close_if_open(&mut out);
                self.open_block(serde_json::json!({"type": "text", "text": ""}), None, &mut out);
                out.push((
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": self.index,
                        "delta": {"type": "text_delta", "text": raw},
                    }),
                ));
                self.close_if_open(&mut out);
            }
            _ => {}
        }
        out
    }

    fn finish(&mut self) -> SseEvents {
        let mut out = SseEvents::new();
        self.close_if_open(&mut out);
        out
    }
}

#[allow(clippy::too_many_arguments)]
async fn messages_stream(
    daemon: Arc<Daemon>, model_name: String, req: MessagesRequest, id: String,
    op: crate::inference::SamplingOverrides, qos: crate::inference::RequestQos,
    breakpoints: Vec<Breakpoint>, hold: Option<crate::keypolicy::ClientKey>,
) -> Response {
    let max_tokens = req.max_tokens;
    let prepare_daemon = Arc::clone(&daemon);
    let admitted = crate::inference::blocking(qos, hold.clone(), move || {
        let model = prepare_daemon.model_sampling_defaults();
        let built = build_session(&prepare_daemon, &req, &op, &model, &breakpoints)?;
        qos.apply(&prepare_daemon, built.session)?;
        prepare_daemon.session_fits(built.session, None)?;
        Ok((built, model))
    }).await;
    let (built, model) = match admitted {
        Ok(value) => value,
        Err(crate::inference::RunError::Daemon(e)) => return daemon_error(e),
        Err(crate::inference::RunError::Worker(e)) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let rx = blocks::stream(daemon, built, op, model, qos, max_tokens, hold);
    let mut blocks = BlockStream::new();
    let frames = ReceiverStream::new(rx).map(move |event| {
        match event {
            Ok(blocks::Event::Start) => vec![("message_start", serde_json::json!({
                "type":"message_start", "message":{"id":id,"type":"message","role":"assistant","model":model_name,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}
            }))],
            Ok(blocks::Event::Committed(event)) => blocks.on_event(event.event_id, &event.body),
            Ok(blocks::Event::Finished { finish, tokens, usage, pinned }) => {
                let mut events = blocks.finish();
                events.push(("message_delta", serde_json::json!({"type":"message_delta","delta":{"stop_reason":stop_reason(finish,tokens,max_tokens,blocks.had_tool),"stop_sequence":null},"usage":usage_json(&usage,pinned)})));
                events.push(("message_stop", serde_json::json!({"type":"message_stop"})));
                events
            }
            Err(error) => vec![("error", serde_json::json!({"type":"error","error":{"type":if error.client_error {"invalid_request_error"} else {"api_error"},"message":error.message}}))],
        }
    });
    let stream = futures_util::StreamExt::flat_map(frames, |events| {
        tokio_stream::iter(events).map(|(name, value)| Ok::<_, std::convert::Infallible>(Event::default().event(name).data(value.to_string())))
    });
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen(text: &str, ch: u32) -> EventBody {
        EventBody::Generated {
            span: vec![],
            text: text.into(),
            channel: ch,
            finish: 0,
        }
    }

    fn assert_grammar(events: &[(&'static str, serde_json::Value)]) {
        let mut open: Option<i64> = None;
        let mut last_index: i64 = -1;
        for (name, v) in events {
            let idx = v.get("index").and_then(|i| i.as_i64());
            match *name {
                "content_block_start" => {
                    assert!(open.is_none(), "start while a block is open: {v}");
                    let idx = idx.expect("index");
                    assert_eq!(idx, last_index + 1, "indices strictly increase");
                    last_index = idx;
                    open = Some(idx);
                }
                "content_block_delta" => {
                    assert_eq!(idx, open, "delta outside its open block: {v}");
                }
                "content_block_stop" => {
                    assert_eq!(idx, open, "stop must close the OPEN block: {v}");
                    open = None;
                }
                _ => {}
            }
        }
        assert!(open.is_none(), "a block was left open");
    }

    #[test]
    fn tool_use_followed_by_text_never_double_stops() {
        let mut b = BlockStream::new();
        let mut all = Vec::new();
        all.extend(b.on_event(0, &gen("thinking...", channel::REASONING)));
        all.extend(b.on_event(1, &EventBody::ToolUse {
            name: "get_weather".into(),
            arguments: "{}".into(),
        }));
        all.extend(b.on_event(2, &gen("and the answer", channel::TEXT)));
        all.extend(b.finish());
        assert_grammar(&all);
        assert!(b.had_tool);
    }

    #[test]
    fn multiple_tool_uses_in_sequence() {
        let mut b = BlockStream::new();
        let mut all = Vec::new();
        for i in 0..3u64 {
            all.extend(b.on_event(i, &EventBody::ToolUse {
                name: format!("tool{i}"),
                arguments: "{}".into(),
            }));
        }
        all.extend(b.finish());
        assert_grammar(&all);
        let stops = all.iter().filter(|(n, _)| *n == "content_block_stop").count();
        assert_eq!(stops, 3, "one stop per tool block, no extras");
    }

    #[test]
    fn parse_failure_is_an_ordered_text_block() {
        let mut b = BlockStream::new();
        let mut all = Vec::new();
        all.extend(b.on_event(0, &gen("", channel::TOOL_CALL)));
        all.extend(b.on_event(1, &EventBody::ToolParseFailure {
            raw: "<atem:garbage".into(),
        }));
        all.extend(b.on_event(2, &gen("recovered", channel::TEXT)));
        all.extend(b.finish());
        assert_grammar(&all);
        let texts: Vec<&str> = all
            .iter()
            .filter(|(n, _)| *n == "content_block_delta")
            .filter_map(|(_, v)| v["delta"]["text"].as_str())
            .collect();
        assert_eq!(texts, vec!["<atem:garbage", "recovered"]);
    }

    #[test]
    fn empty_slices_and_channel_flips_balance() {
        let mut b = BlockStream::new();
        let mut all = Vec::new();
        all.extend(b.on_event(0, &gen("", channel::REASONING)));
        all.extend(b.on_event(1, &gen("think", channel::REASONING)));
        all.extend(b.on_event(2, &gen("", channel::TEXT)));
        all.extend(b.on_event(3, &gen("answer", channel::TEXT)));
        all.extend(b.finish());
        assert_grammar(&all);
    }
}
