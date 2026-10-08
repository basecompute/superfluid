//! api-openai over a FLEET (multi-node head).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use superfluid_abi::finish;

use crate::fleet_manager::FleetManager;
use crate::openai::ServeConfig;
use crate::http_state::error_response;
use crate::inference::{
    finish_reason, next_id, split_content, unix_now, ChatRequest,
    DEFAULT_MAX_TOKENS,
};
use crate::wal::{channel, role};
use crate::{DaemonError, GenParams, TextCodec};

const FLEET_WALL_MS: u64 = 10 * 60 * 1000;

#[derive(Clone)]
pub(crate) struct FleetState {
    pub(crate) fleet: Arc<FleetManager>,
    pub(crate) sampling: crate::inference::SamplingOverrides,
    pub(crate) codec: Arc<dyn TextCodec + Send + Sync>,
    pub(crate) codec_name: Arc<String>,
    pub(crate) model_name: Arc<String>,
    pub(crate) max_context: u32,
}

pub fn serve_blocking(
    listener: std::net::TcpListener,
    fleet: Arc<FleetManager>,
    codec: Arc<dyn TextCodec + Send + Sync>,
    codec_name: String,
    model_name: String,
    max_context: u32,
    cfg: ServeConfig,
) -> std::io::Result<()> {
    if let Some(keys) = &cfg.keys {
        keys.check_global_key(cfg.api_key.as_deref())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    }
    let listens_on_loopback = listener.local_addr()?.ip().is_loopback();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        listener.set_nonblocking(true)?;
        let listener = tokio::net::TcpListener::from_std(listener)?;
        let ip_limiter = Arc::new(crate::ratelimit::RateLimiter::new(cfg.rate_limit_per_minute));
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/models", get(models))
            .with_state(FleetState {
                fleet,
                codec,
                codec_name: Arc::new(codec_name),
                model_name: Arc::new(model_name),
                max_context,
                sampling: cfg.sampling,
            });
        let app = crate::openai::access_layers(app, &cfg, ip_limiter, listens_on_loopback);
        axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).await
    })
}

const MEDIA_REASON: &str =
    "fleet serve is text-only; image and audio parts are not routed to nodes";

fn fleet_capabilities(st: &FleetState) -> serde_json::Value {
    serde_json::json!({
        "descriptor_version": 1,
        "backend": "fleet",
        "workload": {
            "causal_generation": true,
            "embedding": "the fleet head serves chat completions only; no /v1/embeddings route",
            "speech_to_text": "the fleet head serves chat completions only; no /v1/audio route",
        },
        "modalities": {
            "image_encode": MEDIA_REASON,
            "gemma_audio_encode": MEDIA_REASON,
            "whisper_transcribe": "the fleet head serves chat completions only; no /v1/audio route",
            "whisper_translate": "the fleet head serves chat completions only; no /v1/audio route",
        },
        "dialect": {
            "enable_thinking": st.codec.supports_enable_thinking(),
            "reasoning_effort": st.codec.supports_reasoning_effort(),
            "reasoning_effort_levels": st.codec.reasoning_effort_levels(),
        },
    })
}

async fn models(State(st): State<FleetState>) -> Json<serde_json::Value> {
    // Unknown until a node answers, unless --max-context pinned it.
    let n_ctx = st
        .fleet
        .advertised_context(&st.model_name)
        .or((st.max_context > 0).then_some(st.max_context as u64));
    let entry = crate::openai::ModelObject {
        id: &st.model_name,
        created: unix_now(),
        n_ctx: n_ctx.map(|n| u32::try_from(n).unwrap_or(u32::MAX)),
        owned_by: "superfluid-fleet",
        loaded: true,
        capabilities: Some(fleet_capabilities(&st)),
    }
    .build();
    Json(serde_json::json!({"object": "list", "data": [entry]}))
}

fn next_session() -> u64 {
    static N: std::sync::OnceLock<AtomicU64> = std::sync::OnceLock::new();
    N.get_or_init(|| {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        AtomicU64::new((1u64 << 40) | (seed & ((1 << 40) - 1)))
    })
    .fetch_add(1, Ordering::Relaxed)
}

fn render(
    op: &crate::inference::SamplingOverrides,
    codec: &dyn TextCodec,
    req: &ChatRequest,
) -> Result<(GenParams, Vec<u32>), DaemonError> {
    let mut params = crate::inference::params_of_for(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.seed,
        &crate::ModelSamplingDefaults::default(),
        op,
    );
    let _ = crate::inference::draw_seed_if_unseeded(&mut params, req.seed);
    let mut messages: Vec<crate::codec::ChatMessage> = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        let split = split_content(&m.content)?;
        if !split.images.is_empty() {
            return Err(DaemonError::Protocol(MEDIA_REASON));
        }
        let r = match m.role.as_str() {
            "assistant" => role::ASSISTANT,
            "system" => role::SYSTEM,
            "tool" => role::TOOL,
            _ => role::USER,
        };
        let mut msg = crate::codec::ChatMessage::new(r, split.text);
        for c in &m.tool_calls {
            let Some(f) = &c.function else { continue };
            let args = match &f.arguments {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            msg.tool_calls.push(crate::codec::ToolCallMsg {
                id: c.id.clone().unwrap_or_default(),
                name: f.name.clone(),
                arguments: args,
            });
        }
        if r == role::TOOL {
            msg.tool_call_id = m.tool_call_id.clone();
            msg.name = m.name.clone();
        }
        msg.reasoning = crate::inference::reasoning_of(m, r);
        messages.push(msg);
    }
    let tools: Vec<String> = req.tools.iter().map(|t| t.to_string()).collect();
    let kwargs = req.template_kwargs();
    let tokens = codec
        .render_prompt_structured_with(&messages, &tools, &kwargs)
        .ok_or_else(|| crate::render_refusal(codec, &messages, &tools, &kwargs, "codec has no chat dialect for this model"))?;
    Ok((params, tokens))
}

struct Seg {
    channel: u32,
    text: String,
    tool: Option<(String, String)>,
}

fn channelize(
    codec: &dyn TextCodec,
    prompt: &[u32],
    tokens: &[u32],
    schemas: Option<&crate::codec::ToolSchemas>,
) -> Vec<Seg> {
    let mut chan = crate::codec::primed_channelizer(codec, prompt);
    let mut segs = Vec::new();
    for run in chan.split(tokens) {
        let text = codec.decode(&run.text);
        let tool = if run.channel == channel::TOOL_CALL && run.closes {
            codec.parse_tool_call_with(&text, schemas)
        } else {
            None
        };
        if !text.is_empty() || tool.is_some() {
            segs.push(Seg { channel: run.channel, text, tool });
        }
    }
    segs
}

fn map_err(e: DaemonError) -> Response {
    if matches!(e, DaemonError::StreamTooLong { .. }) {
        return crate::http_state::error_response_typed(
            StatusCode::BAD_REQUEST,
            e.to_string(),
            "invalid_request_error",
            "context_length_exceeded",
        );
    }
    if let DaemonError::Unsupported(r) = &e {
        return crate::http_state::refusal_response(StatusCode::BAD_REQUEST, r);
    }
    let status = match e {
        DaemonError::Config(_) | DaemonError::FleetNode { .. } => StatusCode::SERVICE_UNAVAILABLE,
        DaemonError::Protocol(_) | DaemonError::TemplateRefused(_) | DaemonError::Unsupported(_) => {
            StatusCode::BAD_REQUEST
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error_response(status, e.to_string())
}

fn fleet_unsupported_penalty(req: &ChatRequest) -> Option<String> {
    let named = |name: &str| {
        format!(
            "{name} is not supported on a fleet endpoint: the fleet protocol carries no \
             per-request penalty, so a fleet generation always runs unpenalized. Until the \
             wire carries one, use a single-node endpoint if you need a penalty — setting it \
             on the nodes would break the fleet's token-for-token parity with the local path."
        )
    };
    if req.repeat_penalty.is_some_and(|v| v != 1.0 && v != 0.0) {
        return Some(named("repeat_penalty"));
    }
    if req.presence_penalty.is_some_and(|v| v != 0.0) {
        return Some(named("presence_penalty"));
    }
    if req.frequency_penalty.is_some_and(|v| v != 0.0) {
        return Some(named("frequency_penalty"));
    }
    None
}

fn fleet_unsupported_option(req: &ChatRequest) -> Option<String> {
    if req.ignore_eos {
        return Some(
            "ignore_eos is not supported on a fleet endpoint: the fleet protocol does not \
             carry it, so nodes would still stop at end-of-sequence. Use a single-node \
             endpoint for fixed-length generations."
                .into(),
        );
    }
    if req.stream_options.as_ref().is_some_and(|o| o.continuous_usage_stats) {
        return Some(
            "stream_options.continuous_usage_stats is not supported on a fleet endpoint: the \
             fleet stream carries no per-chunk token counts. Use a single-node endpoint, or \
             read usage from the final chunk."
                .into(),
        );
    }
    None
}

async fn chat_completions(
    State(st): State<FleetState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    body: axum::body::Bytes,
) -> Response {
    let req: ChatRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    if let Some(reason) = fleet_unsupported_penalty(&req).or_else(|| fleet_unsupported_option(&req)) {
        return error_response(StatusCode::BAD_REQUEST, reason);
    }
    match req.model.as_deref() {
        None => return error_response(StatusCode::BAD_REQUEST, "missing required field: model".into()),
        Some(m) if m != st.model_name.as_str() => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!("model '{m}' not found; this fleet serves '{}'", st.model_name),
            )
        }
        Some(_) => {}
    }
    if req.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages must be non-empty".into());
    }
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let id = next_id();
    let created = unix_now();
    let hold = key.as_ref().map(|k| k.0.clone());

    if req.stream {
        return chat_stream(st, req, max_tokens, id, created, hold).await;
    }

    let out = tokio::task::spawn_blocking(move || {
        let _hold = hold;
        let (params, tokens) = render(&st.sampling, st.codec.as_ref(), &req)?;
        let session = next_session();
        let gen = st
            .fleet
            .place(session, params, 0, &st.codec_name, &st.model_name, &tokens, max_tokens as u64)
            .and_then(|_| st.fleet.generate(session, max_tokens as u64, FLEET_WALL_MS));
        st.fleet.finish(session);
        let gen = gen?;
        let schemas = crate::codec::ToolSchemas::from_values(req.tools.iter());
        let segs = channelize(st.codec.as_ref(), &tokens, &gen.tokens, Some(&schemas));
        Ok::<_, DaemonError>((gen, segs, tokens.len(), st))
    })
    .await;
    let (gen, segs, prompt_len, st) = match out {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return map_err(e),
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<(String, String)> = Vec::new();
    for s in &segs {
        match s.channel {
            channel::REASONING => reasoning.push_str(&s.text),
            channel::TOOL_CALL => {
                if let Some(tc) = &s.tool {
                    tool_calls.push(tc.clone());
                } else {
                    content.push_str(&s.text);
                }
            }
            _ => content.push_str(&s.text),
        }
    }
    let mut message = serde_json::json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning_content"] = serde_json::Value::String(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = tool_calls_json(&tool_calls);
    }
    let completion = gen.tokens.len() as u32;
    let json = serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": *st.model_name,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason(gen.finish, completion, max_tokens, !tool_calls.is_empty()),
        }],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion,
            "total_tokens": prompt_len as u64 + completion as u64,
        },
    });
    (StatusCode::OK, Json(json)).into_response()
}

fn tool_calls_json(calls: &[(String, String)]) -> serde_json::Value {
    serde_json::Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, (name, args))| {
                serde_json::json!({
                    "index": i,
                    "id": format!("call_{i}"),
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                })
            })
            .collect(),
    )
}

async fn chat_stream(
    st: FleetState,
    req: ChatRequest,
    max_tokens: u32,
    id: String,
    created: u64,
    hold: Option<crate::keypolicy::ClientKey>,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<serde_json::Value>(64);
    let model = st.model_name.to_string();
    let chunk = move |delta: serde_json::Value, fin: Option<&str>| {
        serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": fin}],
        })
    };
    tokio::task::spawn_blocking(move || {
        let _hold = hold;
        let send = |v: serde_json::Value| -> Result<(), DaemonError> {
            tx.blocking_send(v).map_err(|_| DaemonError::Protocol("sse client gone"))
        };
        let session = next_session();
        let run = || -> Result<(), DaemonError> {
            let (params, tokens) = render(&st.sampling, st.codec.as_ref(), &req)?;
            let prompt_len = tokens.len();
            st.fleet
                .place(session, params, 0, &st.codec_name, &st.model_name, &tokens, max_tokens as u64)?;
            send(chunk(serde_json::json!({"role": "assistant"}), None))?;

            let mut chan = crate::codec::primed_channelizer(st.codec.as_ref(), &tokens);
            let schemas = crate::codec::ToolSchemas::from_values(req.tools.iter());
            let mut tool_buf = String::new();
            let mut tool_index = 0usize;
            let mut frag = crate::tool_fragment::ToolCallStream::new(st.codec.tool_fragment_mode())
                .with_schemas(crate::codec::ToolSchemas::of_values(&req.tools));
            let mut open_call_id = String::new();
            let mut stream_err: Option<DaemonError> = None;
            let gen = st.fleet.generate_streaming(
                session,
                max_tokens as u64,
                FLEET_WALL_MS,
                |batch| {
                    if stream_err.is_some() {
                        return;
                    }
                    for run in chan.split(batch) {
                        let text = st.codec.decode(&run.text);
                        let r = match run.channel {
                            channel::REASONING => {
                                if text.is_empty() {
                                    continue;
                                }
                                send(chunk(serde_json::json!({"reasoning_content": text}), None))
                            }
                            channel::TOOL_CALL => {
                                tool_buf.push_str(&text);
                                let mut r = Ok(());
                                for d in frag.feed(&text) {
                                    if d.opening {
                                        open_call_id = format!("call_{tool_index}");
                                    }
                                    r = send(chunk(
                                        crate::openai::tool_call_delta(&d, tool_index, &open_call_id),
                                        None,
                                    ));
                                    if r.is_err() {
                                        break;
                                    }
                                }
                                if r.is_ok() && run.closes {
                                    let raw = std::mem::take(&mut tool_buf);
                                    if let Some((name, args)) = st.codec.parse_tool_call_with(&raw, Some(&schemas)) {
                                        if frag.opened() {
                                            for d in frag.finish(&args) {
                                                r = send(chunk(
                                                    crate::openai::tool_call_delta(
                                                        &d,
                                                        tool_index,
                                                        &open_call_id,
                                                    ),
                                                    None,
                                                ));
                                                if r.is_err() {
                                                    break;
                                                }
                                            }
                                            frag.reset();
                                            tool_index += 1;
                                            r
                                        } else {
                                            let d = serde_json::json!({"tool_calls": [{
                                                "index": tool_index,
                                                "id": format!("call_{tool_index}"),
                                                "type": "function",
                                                "function": {"name": name, "arguments": args},
                                            }]});
                                            tool_index += 1;
                                            send(chunk(d, None))
                                        }
                                    } else {
                                        frag.reset();
                                        send(chunk(serde_json::json!({"content": raw}), None))
                                    }
                                } else {
                                    r
                                }
                            }
                            _ => {
                                if text.is_empty() {
                                    continue;
                                }
                                send(chunk(serde_json::json!({"content": text}), None))
                            }
                        };
                        if let Err(e) = r {
                            let _ = st.fleet.cancel(session);
                            stream_err = Some(e);
                            return;
                        }
                    }
                },
            )?;
            if let Some(e) = stream_err {
                return Err(e);
            }
            let completion = gen.tokens.len() as u32;
            let fin = finish_reason(gen.finish, completion, max_tokens, tool_index > 0);
            let mut terminal = chunk(serde_json::json!({}), Some(fin));
            terminal["usage"] = serde_json::json!({
                "prompt_tokens": prompt_len,
                "completion_tokens": completion,
                "total_tokens": prompt_len as u64 + completion as u64,
            });
            send(terminal)
        };
        let ran = run();
        st.fleet.finish(session);
        if let Err(e) = ran {
            let kind = if matches!(
                e,
                DaemonError::Constraint(_)
                    | DaemonError::TemplateRefused(_)
                    | DaemonError::Unsupported(_)
                    | DaemonError::StreamTooLong { .. }
            ) {
                "invalid_request_error"
            } else {
                "server_error"
            };
            let _ = tx.blocking_send(crate::http_state::sse_error_frame(&e, kind));
        }
    });

    let stream = ReceiverStream::new(rx)
        .map(|v| Event::default().data(v.to_string()))
        .chain(tokio_stream::once(Event::default().data("[DONE]")))
        .map(Ok::<_, std::convert::Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

const _: u32 = finish::EOS;

#[cfg(test)]
mod penalty_refusal_tests {
    use super::{fleet_unsupported_option, fleet_unsupported_penalty};
    use crate::inference::ChatRequest;

    fn req(json: &str) -> ChatRequest {
        serde_json::from_str(json).expect("request")
    }

    #[test]
    fn a_penalty_the_wire_cannot_carry_is_refused() {
        assert!(fleet_unsupported_penalty(&req(r#"{"messages":[],"repeat_penalty":1.3}"#)).is_some());
        assert!(fleet_unsupported_penalty(&req(r#"{"messages":[],"presence_penalty":0.5}"#)).is_some());
        assert!(fleet_unsupported_penalty(&req(r#"{"messages":[],"frequency_penalty":-0.5}"#)).is_some());
    }

    #[test]
    fn benchmark_options_the_wire_cannot_carry_are_refused() {
        assert!(fleet_unsupported_option(&req(r#"{"messages":[],"ignore_eos":true}"#)).is_some());
        assert!(fleet_unsupported_option(&req(
            r#"{"messages":[],"stream_options":{"include_usage":true,"continuous_usage_stats":true}}"#
        ))
        .is_some());
        assert!(fleet_unsupported_option(&req(r#"{"messages":[],"ignore_eos":false}"#)).is_none());
        assert!(fleet_unsupported_option(&req(
            r#"{"messages":[],"stream_options":{"include_usage":true}}"#
        ))
        .is_none());
    }

    #[test]
    fn an_absent_or_no_op_penalty_is_served() {
        assert!(fleet_unsupported_penalty(&req(r#"{"messages":[]}"#)).is_none());
        assert!(fleet_unsupported_penalty(&req(r#"{"messages":[],"repeat_penalty":1.0}"#)).is_none());
        assert!(fleet_unsupported_penalty(&req(r#"{"messages":[],"repeat_penalty":0.0}"#)).is_none());
        assert!(fleet_unsupported_penalty(
            &req(r#"{"messages":[],"presence_penalty":0.0,"frequency_penalty":0.0}"#)
        )
        .is_none());
    }
}

#[cfg(test)]
mod template_refusal_tests {
    use super::{map_err, render};
    use crate::codec::{ChatMessage, MockChatCodec};
    use crate::inference::{ChatRequest, SamplingOverrides};
    use crate::{DaemonError, TextCodec};

    const REFUSAL: &str = "Conversation roles must alternate user/assistant/user/assistant/...";

    struct Refusing;

    impl TextCodec for Refusing {
        fn encode(&self, text: &str) -> Vec<u32> {
            MockChatCodec.encode(text)
        }
        fn token_bytes(&self, token: u32) -> Vec<u8> {
            MockChatCodec.token_bytes(token)
        }
        fn renders_per_message(&self) -> bool {
            false
        }
        fn render_prompt_structured_with(
            &self,
            _messages: &[ChatMessage],
            _tools: &[String],
            _kwargs: &serde_json::Map<String, serde_json::Value>,
        ) -> Option<Vec<u32>> {
            None
        }
        fn template_refusal(
            &self,
            _messages: &[ChatMessage],
            _tools: &[String],
            _kwargs: &serde_json::Map<String, serde_json::Value>,
        ) -> Option<String> {
            Some(REFUSAL.to_string())
        }
    }

    #[test]
    fn a_template_refusal_on_the_fleet_is_a_client_error_in_its_words() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"one"},{"role":"user","content":"two"}]}"#,
        )
        .expect("request");
        let e = render(&SamplingOverrides::default(), &Refusing, &req).expect_err("the template refuses");
        assert!(matches!(&e, DaemonError::TemplateRefused(why) if why == REFUSAL), "{e:?}");

        let resp = map_err(e);
        assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
        let body = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime")
            .block_on(axum::body::to_bytes(resp.into_body(), usize::MAX))
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(v["error"]["message"].as_str().unwrap_or_default().contains(REFUSAL), "{v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    }
}

#[cfg(test)]
mod reasoning_history_tests {
    use std::sync::Mutex;

    use super::render;
    use crate::codec::{ChatMessage, MockChatCodec};
    use crate::inference::{ChatRequest, SamplingOverrides};
    use crate::TextCodec;

    #[derive(Default)]
    struct Capturing(Mutex<Vec<(u32, Option<String>)>>);

    impl TextCodec for Capturing {
        fn encode(&self, text: &str) -> Vec<u32> {
            MockChatCodec.encode(text)
        }
        fn token_bytes(&self, token: u32) -> Vec<u8> {
            MockChatCodec.token_bytes(token)
        }
        fn renders_per_message(&self) -> bool {
            false
        }
        fn render_prompt_structured_with(
            &self,
            messages: &[ChatMessage],
            _tools: &[String],
            _kwargs: &serde_json::Map<String, serde_json::Value>,
        ) -> Option<Vec<u32>> {
            *self.0.lock().unwrap() = messages.iter().map(|m| (m.role, m.reasoning.clone())).collect();
            Some(vec![1])
        }
    }

    #[test]
    fn a_fleet_render_carries_the_reasoning_an_assistant_turn_hands_back() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"messages":[
                {"role":"user","content":"list files","reasoning_content":"not mine"},
                {"role":"assistant","content":"","reasoning_content":"call ls first"},
                {"role":"user","content":"and then?"},
                {"role":"assistant","content":"done","reasoning":"the other spelling"}
            ]}"#,
        )
        .expect("request");
        let codec = Capturing::default();
        render(&SamplingOverrides::default(), &codec, &req).expect("rendered");
        let seen = codec.0.lock().unwrap().clone();
        let reasoning: Vec<Option<&str>> = seen.iter().map(|(_, r)| r.as_deref()).collect();
        assert_eq!(reasoning, vec![None, Some("call ls first"), None, Some("the other spelling")], "{seen:?}");
    }
}
