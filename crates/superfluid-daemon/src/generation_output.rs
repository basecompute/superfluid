//! Values passed from inference to protocol encoders.

use crate::{Daemon, DaemonError};

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Chat,
    Completion,
}

pub(crate) struct Generation {
    pub kind: Kind,
    pub id: String,
    pub model: String,
    pub created: u64,
    pub warm_header: Option<u64>,
    pub body: Body,
}

pub(crate) enum Body {
    Complete(Frame),
    Stream(tokio::sync::mpsc::Receiver<Result<Frame, StreamFailure>>),
}

pub(crate) struct StreamFailure {
    pub message: String,
    pub client_error: bool,
    pub code: Option<&'static str>,
    pub param: Option<String>,
}

impl StreamFailure {
    pub fn new(error: DaemonError, client_error: bool) -> Self {
        let (code, param) = match &error {
            DaemonError::StreamTooLong { .. } => (Some("context_length_exceeded"), None),
            DaemonError::Unsupported(r) => (Some(r.code), r.param.clone()),
            _ => (None, None),
        };
        Self {
            code,
            param,
            message: error.to_string(),
            client_error,
        }
    }
}

#[derive(Default)]
pub(crate) struct Frame {
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
    pub warm: Option<u64>,
    pub spec: Option<(u64, u64)>,
    pub fim: Option<FimMetadata>,
    /// The session the choice ran in, for a dialect that keeps it.
    pub session: Option<u64>,
}

#[derive(Default)]
pub(crate) struct Choice {
    pub index: u32,
    pub message: Message,
    pub text: Option<String>,
    pub logprobs: Option<Logprobs>,
    pub finish: Option<&'static str>,
}

#[derive(Default)]
pub(crate) struct Message {
    pub assistant: bool,
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tools: Vec<ToolDelta>,
}

pub(crate) struct ToolDelta {
    pub index: usize,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
}

impl Message {
    pub fn content(text: impl Into<String>) -> Self {
        Self {
            content: Some(text.into()),
            ..Self::default()
        }
    }

    pub fn reasoning(text: impl Into<String>) -> Self {
        Self {
            reasoning: Some(text.into()),
            ..Self::default()
        }
    }

    pub fn tool(delta: &crate::tool_fragment::ToolDelta, index: usize, id: &str) -> Self {
        Self {
            tools: vec![ToolDelta {
                index,
                id: delta.opening.then(|| id.to_owned()),
                name: delta.opening.then(|| delta.name.clone()),
                arguments: delta.arguments.clone(),
            }],
            ..Self::default()
        }
    }

    pub fn complete_tool(index: usize, id: String, name: String, arguments: String) -> Self {
        Self {
            tools: vec![ToolDelta {
                index,
                id: Some(id),
                name: Some(name),
                arguments,
            }],
            ..Self::default()
        }
    }
}

impl Frame {
    pub fn chat(message: Message, finish: Option<&'static str>) -> Self {
        Self {
            choices: vec![Choice {
                message,
                finish,
                ..Choice::default()
            }],
            ..Self::default()
        }
    }

    pub fn text(
        text: impl Into<String>,
        logprobs: Option<Logprobs>,
        finish: Option<&'static str>,
    ) -> Self {
        Self {
            choices: vec![Choice {
                text: Some(text.into()),
                logprobs,
                finish,
                ..Choice::default()
            }],
            ..Self::default()
        }
    }
}

#[derive(Default)]
pub(crate) struct Usage {
    pub prompt: Option<u64>,
    pub completion: u64,
    pub total: Option<u64>,
    pub cached: Option<u64>,
}

impl Usage {
    pub fn complete(prompt: u64, completion: u64, cached: Option<u64>) -> Self {
        Self {
            prompt: Some(prompt),
            completion,
            total: Some(prompt + completion),
            cached,
        }
    }

    pub fn partial(completion: u32) -> Self {
        Self {
            completion: completion as u64,
            ..Self::default()
        }
    }
}

pub(crate) struct FimMetadata {
    pub cached: bool,
    pub expired: bool,
    pub served_by: String,
}

pub(crate) struct Logprobs {
    pub tokens: Vec<TokenProbability>,
    pub offset: usize,
}

pub(crate) struct TokenProbability {
    pub token: String,
    pub logprob: f32,
    pub alternatives: Vec<(String, f32)>,
}

pub(crate) fn decode_logprobs(
    daemon: &Daemon,
    items: &[crate::scheduler::TokenLogprob],
    offset: usize,
) -> Logprobs {
    Logprobs {
        tokens: items
            .iter()
            .map(|item| TokenProbability {
                token: daemon.detokenize(&[item.token]),
                logprob: item.logprob,
                alternatives: item
                    .top
                    .iter()
                    .map(|(id, lp)| (daemon.detokenize(&[*id]), *lp))
                    .collect(),
            })
            .collect(),
        offset,
    }
}

pub(crate) fn usage(daemon: &Daemon, session: u64, completion: u32, warm: u64) -> Usage {
    let total = daemon
        .store()
        .lock()
        .expect("store")
        .session(session)
        .map(|s| s.tokens.len() as u64)
        .unwrap_or(0);
    let prompt = total.saturating_sub(completion as u64);
    Usage {
        prompt: Some(prompt),
        completion: completion as u64,
        total: Some(total),
        cached: Some(warm.min(prompt)),
    }
}

pub(crate) fn message(
    c: &crate::inference::Collected,
    choice: &crate::inference::ToolChoice,
) -> Message {
    let mut content = c.content.clone();
    content.push_str(&c.unparsed_tool);
    let suppress = *choice == crate::inference::ToolChoice::None;
    if suppress {
        for (_, name, args) in &c.tool_calls {
            content.push_str(&format!(
                "{{\"name\":{},\"arguments\":{}}}",
                serde_json::Value::String(name.clone()),
                args
            ));
        }
    }
    let mut calls: Vec<&(u64, String, String)> = c.tool_calls.iter().chain(c.broken_calls.iter()).collect();
    calls.sort_by_key(|(id, _, _)| *id);
    Message {
        assistant: true,
        content: Some(content),
        reasoning: (!c.reasoning.is_empty()).then(|| c.reasoning.clone()),
        tools: if suppress {
            vec![]
        } else {
            calls
                .into_iter()
                .enumerate()
                .map(|(index, (id, name, args))| ToolDelta {
                    index,
                    id: Some(format!("call_{id}")),
                    name: Some(name.clone()),
                    arguments: args.clone(),
                })
                .collect()
        },
    }
}
