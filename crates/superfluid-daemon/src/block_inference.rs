//! Protocol-neutral inference for ordered content blocks and pinned prompt cuts.

use crate::generation_output::{usage, StreamFailure, Usage};
use crate::inference::{self, RequestQos, SamplingOverrides};
use crate::wal::role;
use crate::{CommittedEvent, Daemon, DaemonError, GenerateOutcome, ModelSamplingDefaults};
use std::sync::Arc;

pub(crate) struct BlockPrompt {
    pub system: Option<String>,
    pub tools: Vec<String>,
    pub messages: Vec<BlockMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
}
pub(crate) struct BlockMessage {
    pub role: u32,
    pub content: Content,
}
pub(crate) enum Content {
    Text(String),
    Blocks(Vec<Block>),
    Empty,
}
pub(crate) enum Block {
    Text(String),
    Image(Vec<u8>),
    Reasoning(Option<String>),
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        id: Option<String>,
        text: String,
    },
}
pub(crate) struct Outcome {
    pub generated: GenerateOutcome,
    pub usage: Usage,
    pub pinned: u64,
}
pub(crate) enum Event {
    Start,
    Committed(CommittedEvent),
    Finished {
        finish: u32,
        tokens: u32,
        usage: Usage,
        pinned: u64,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BreakpointAt {
    FirstTurn,
    Message(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Breakpoint {
    pub at: BreakpointAt,
    pub ttl_ms: u64,
}

pub(crate) struct Built {
    pub session: u64,
    pub seed_drawn: bool,
    pub pins: Vec<(usize, u64)>,
    pub tool_schemas: Option<std::sync::Arc<crate::codec::ToolSchemas>>,
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

pub(crate) fn build_session(
    daemon: &Daemon,
    req: &BlockPrompt,
    op: &crate::inference::SamplingOverrides,
    model: &crate::ModelSamplingDefaults,
    breakpoints: &[Breakpoint],
) -> Result<Built, DaemonError> {
    let mut params = crate::inference::params_of_for(
        req.temperature,
        req.top_p,
        req.top_k,
        None,
        None,
        model,
        &op.for_daemon(daemon),
    );
    let seed_drawn = crate::inference::draw_seed_if_unseeded(&mut params, None);
    let session = daemon.create(None, params)?;
    let stream_len = || -> Result<usize, DaemonError> {
        Ok(daemon
            .store()
            .lock()
            .expect("store")
            .session(session)?
            .tokens
            .len())
    };

    let system = req.system.clone();
    let tool_jsons = req.tools.clone();
    let tool_schemas = (!tool_jsons.is_empty())
        .then(|| std::sync::Arc::new(crate::codec::ToolSchemas::from_jsons(&tool_jsons)));

    if !daemon.renders_per_message() {
        use crate::codec::{ChatMessage, ContentPart};
        let mut msgs: Vec<ChatMessage> = Vec::new();
        if let Some(sys) = &system {
            if !sys.is_empty() {
                msgs.push(ChatMessage::new(role::SYSTEM, sys.clone()));
            }
        }
        let mut cuts: Vec<usize> = Vec::with_capacity(req.messages.len());
        let mut any_image = false;
        for m in &req.messages {
            let r = m.role;
            match &m.content {
                Content::Text(text) => msgs.push(ChatMessage::new(r, text.clone())),
                Content::Blocks(blocks) => {
                    let mut msg = ChatMessage::new(r, String::new());
                    let mut parts: Vec<ContentPart> = Vec::new();
                    let mut has_image = false;
                    for b in blocks {
                        match b {
                            Block::Text(t) => parts.push(ContentPart::Text(t.clone())),
                            Block::Image(bytes) => {
                                daemon.require("modalities", "image_encode", "unsupported_input", Some("messages"), "does not accept images")?;
                                parts.push(ContentPart::Image {
                                    blob: daemon.put_media(bytes)?,
                                });
                                has_image = true;
                            }
                            Block::ToolCall {
                                id,
                                name,
                                arguments,
                            } => msg.tool_calls.push(crate::codec::ToolCallMsg {
                                id: id.clone(),
                                name: name.clone(),
                                arguments: arguments.clone(),
                            }),
                            Block::ToolResult { id, text } => {
                                let mut tr = ChatMessage::new(role::TOOL, text.clone());
                                tr.tool_call_id = id.clone();
                                msgs.push(tr);
                            }
                            Block::Reasoning(t) if r == role::ASSISTANT => {
                                if let Some(t) =
                                    t.as_deref().map(str::trim).filter(|t| !t.is_empty())
                                {
                                    let acc = msg.reasoning.get_or_insert_with(String::new);
                                    if !acc.is_empty() {
                                        acc.push('\n');
                                    }
                                    acc.push_str(t);
                                }
                            }
                            _ => {}
                        }
                    }
                    if has_image {
                        any_image = true;
                        msg.parts = parts;
                    } else {
                        msg.content = parts
                            .into_iter()
                            .map(|p| match p {
                                ContentPart::Text(t) => t,
                                ContentPart::Image { .. } | ContentPart::Audio { .. } => {
                                    unreachable!()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("");
                    }
                    if !msg.content.is_empty()
                        || !msg.parts.is_empty()
                        || !msg.tool_calls.is_empty()
                        || msg.reasoning.is_some()
                    {
                        msgs.push(msg);
                    }
                }
                _ => {}
            }
            cuts.push(msgs.len());
        }
        daemon.append_conversation(session, &msgs, &tool_jsons, &serde_json::Map::new())?;
        let mut pins = Vec::new();
        if !breakpoints.is_empty() && !any_image {
            let full = daemon
                .store()
                .lock()
                .expect("store")
                .session(session)?
                .tokens
                .clone();
            let first_cut = if msgs.first().map(|m| m.role) == Some(role::SYSTEM) {
                1
            } else {
                0
            };
            for bp in breakpoints {
                let cut = match bp.at {
                    BreakpointAt::FirstTurn => first_cut,
                    BreakpointAt::Message(k) => cuts.get(k).copied().unwrap_or(msgs.len()),
                };
                if cut == 0 && tool_jsons.is_empty() {
                    continue;
                }
                if let Some(prefix) = daemon.render_conversation(&msgs[..cut], &tool_jsons) {
                    pins.push((common_prefix(&prefix, &full), bp.ttl_ms));
                }
            }
        }
        return Ok(Built {
            session,
            seed_drawn,
            pins,
            tool_schemas,
        });
    }

    if !tool_jsons.is_empty() {
        daemon.append_system_with_tools(session, system, tool_jsons)?;
    } else if let Some(sys) = system {
        if !sys.is_empty() {
            daemon.append_message(session, role::SYSTEM, sys)?;
        }
    }
    let first_turn_end = stream_len()?;
    let mut message_ends: Vec<usize> = Vec::with_capacity(req.messages.len());
    let mut any_image = false;

    // A turn after the last user query (a user message with text, not only
    // tool results) is a step of a tool loop.
    let last_query = req.messages.iter().rposition(|m| {
        m.role == role::USER
            && match &m.content {
                Content::Text(_) => true,
                Content::Blocks(b) => b.iter().any(|x| matches!(x, Block::Text(_))),
                Content::Empty => false,
            }
    });
    for (i, m) in req.messages.iter().enumerate() {
        let r = m.role;
        let after_query = last_query.is_some_and(|q| i > q);
        match &m.content {
            Content::Text(text) => {
                daemon.append_message(session, r, text.clone())?;
            }
            Content::Blocks(blocks) => {
                let mut text = String::new();
                let mut calls: Vec<(String, String)> = Vec::new();
                let mut tool_results: Vec<String> = Vec::new();
                let mut images: Vec<(String, String)> = Vec::new();
                for b in blocks {
                    match b {
                        Block::Text(t) => text.push_str(t),
                        Block::Image(bytes) => {
                            daemon.require("modalities", "image_encode", "unsupported_input", Some("messages"), "does not accept images")?;
                            images.push((std::mem::take(&mut text), daemon.put_media(bytes)?))
                        }
                        Block::ToolCall {
                            name, arguments, ..
                        } => calls.push((name.clone(), arguments.clone())),
                        Block::ToolResult { text, .. } => tool_results.push(text.clone()),
                        _ => {}
                    }
                }
                for content in tool_results {
                    daemon.append_tool_history(session, content)?;
                }
                if !images.is_empty() {
                    any_image = true;
                    let n = images.len();
                    for (i, (pre, hash)) in images.into_iter().enumerate() {
                        let post = if i + 1 == n {
                            std::mem::take(&mut text)
                        } else {
                            String::new()
                        };
                        daemon.append_image(session, r, &hash, &pre, &post)?;
                    }
                } else if !calls.is_empty() {
                    daemon.append_assistant_with_tool_calls(session, text.clone(), calls, after_query)?;
                } else if !text.is_empty() {
                    daemon.append_message(session, r, text)?;
                }
            }
            _ => {}
        }
        message_ends.push(stream_len()?);
    }
    let mut pins = Vec::new();
    if !any_image {
        for bp in breakpoints {
            let end = match bp.at {
                BreakpointAt::FirstTurn => first_turn_end,
                BreakpointAt::Message(k) => message_ends.get(k).copied().unwrap_or(0),
            };
            if end > 0 {
                pins.push((end, bp.ttl_ms));
            }
        }
    }
    Ok(Built {
        session,
        seed_drawn,
        pins,
        tool_schemas,
    })
}

fn pin_breakpoints(daemon: &Daemon, built: &Built) {
    let Some(stream) = breakpoint_stream(daemon, built) else {
        return;
    };
    for &(upto, ttl_ms) in &built.pins {
        let _ = daemon.pin_prefix(Some(built.session), &stream, upto, ttl_ms);
    }
}

struct Unanchor<'a> {
    daemon: &'a Daemon,
    session: u64,
    armed: bool,
}

impl<'a> Unanchor<'a> {
    fn new(daemon: &'a Daemon, built: &Built) -> Unanchor<'a> {
        Unanchor {
            daemon,
            session: built.session,
            armed: !built.pins.is_empty(),
        }
    }
}

impl Drop for Unanchor<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.daemon.unanchor_pins(self.session);
        }
    }
}

fn pinned_breakpoints(daemon: &Daemon, built: &Built) -> u64 {
    let Some(stream) = breakpoint_stream(daemon, built) else {
        return 0;
    };
    let uptos: Vec<usize> = built.pins.iter().map(|&(upto, _)| upto).collect();
    daemon.pinned_prefix_tokens(&stream, &uptos).unwrap_or(0)
}

fn breakpoint_stream(daemon: &Daemon, built: &Built) -> Option<Vec<u32>> {
    if built.pins.is_empty() {
        return None;
    }
    let store = daemon.store();
    let store = store.lock().expect("store");
    store.session(built.session).ok().map(|s| s.tokens.clone())
}

pub(crate) fn generate(
    daemon: &Daemon,
    built: &Built,
    op: &SamplingOverrides,
    model: &ModelSamplingDefaults,
    max_tokens: u32,
    mut on_event: impl FnMut(&CommittedEvent) -> Result<(), DaemonError>,
) -> Result<Outcome, DaemonError> {
    pin_breakpoints(daemon, built);
    let _unanchor = Unanchor::new(daemon, built);
    let mut extras = inference::extras_of(None, None, None, op, model);
    extras.seed_drawn = built.seed_drawn;
    extras.tool_schemas = built.tool_schemas.clone();
    let generated =
        daemon.generate_streaming_ex(built.session, max_tokens, extras, |ev| on_event(ev))?;
    let pinned = pinned_breakpoints(daemon, built);
    let usage = usage(
        daemon,
        built.session,
        generated.tokens_generated,
        generated.warm_prefix,
    );
    Ok(Outcome {
        generated,
        usage,
        pinned,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn stream(
    daemon: Arc<Daemon>,
    built: Built,
    op: SamplingOverrides,
    model: ModelSamplingDefaults,
    qos: RequestQos,
    max_tokens: u32,
    hold: Option<crate::keypolicy::ClientKey>,
) -> tokio::sync::mpsc::Receiver<Result<Event, StreamFailure>> {
    inference::spawn_stream(qos, hold, move |tx| {
        let _closed = crate::keepalive::cancel_when_closed(&daemon, built.session, tx);
        let send = |event| {
            tx.blocking_send(Ok(event))
                .map_err(|_| DaemonError::Protocol("generation consumer gone"))
        };
        pin_breakpoints(&daemon, &built);
        let _unanchor = Unanchor::new(&daemon, &built);
        send(Event::Start)?;
        let mut extras = inference::extras_of(None, None, None, &op, &model);
        extras.seed_drawn = built.seed_drawn;
        extras.tool_schemas = built.tool_schemas.clone();
        let generated =
            daemon.generate_streaming_ex(built.session, max_tokens, extras, |event| {
                send(Event::Committed(event.clone()))
            })?;
        send(Event::Finished {
            finish: generated.finish,
            tokens: generated.tokens_generated,
            usage: usage(
                &daemon,
                built.session,
                generated.tokens_generated,
                generated.warm_prefix,
            ),
            pinned: pinned_breakpoints(&daemon, &built),
        })
    })
}
