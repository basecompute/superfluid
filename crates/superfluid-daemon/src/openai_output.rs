//! OpenAI JSON/SSE encoding of protocol-neutral generation results.

use crate::generation_output::*;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

pub(crate) fn message_json(message: &Message) -> Value {
    let mut value = json!({});
    if message.assistant {
        value["role"] = json!("assistant");
    }
    if let Some(text) = &message.content {
        value["content"] = json!(text);
    }
    if let Some(text) = &message.reasoning {
        value["reasoning_content"] = json!(text);
    }
    if !message.tools.is_empty() {
        value["tool_calls"] = Value::Array(
            message
                .tools
                .iter()
                .map(|tool| {
                    let mut call =
                        json!({"index":tool.index,"function":{"arguments":tool.arguments}});
                    if let Some(id) = &tool.id {
                        call["id"] = json!(id);
                        call["type"] = json!("function");
                    }
                    if let Some(name) = &tool.name {
                        call["function"]["name"] = json!(name);
                    }
                    call
                })
                .collect(),
        );
    }
    value
}

pub(crate) fn usage_json(usage: &Usage) -> Value {
    let mut value = json!({"completion_tokens":usage.completion});
    if let Some(n) = usage.prompt {
        value["prompt_tokens"] = json!(n);
    }
    if let Some(n) = usage.total {
        value["total_tokens"] = json!(n);
    }
    if let Some(n) = usage.cached {
        value["prompt_tokens_details"] = json!({"cached_tokens":n});
    }
    value
}

pub(crate) fn chat_logprobs(logprobs: &Logprobs) -> Value {
    let token = |text: &str, lp: f32| json!({"token":text,"logprob":lp,"bytes":text.as_bytes()});
    json!({"content":logprobs.tokens.iter().map(|t| {
        let mut value = token(&t.token, t.logprob);
        value["top_logprobs"] = Value::Array(t.alternatives.iter().map(|(text, lp)| token(text, *lp)).collect());
        value
    }).collect::<Vec<_>>()})
}

pub(crate) fn completion_logprobs(logprobs: &Logprobs) -> Value {
    let mut offset = logprobs.offset;
    let offsets = logprobs
        .tokens
        .iter()
        .map(|t| {
            let at = offset;
            offset += t.token.len();
            at
        })
        .collect::<Vec<_>>();
    json!({
        "tokens":logprobs.tokens.iter().map(|t| &t.token).collect::<Vec<_>>(),
        "token_logprobs":logprobs.tokens.iter().map(|t| t.logprob).collect::<Vec<_>>(),
        "top_logprobs":logprobs.tokens.iter().map(|t| {
            if t.alternatives.is_empty() { Value::Null } else {
                let mut map = serde_json::Map::new();
                for (text, lp) in &t.alternatives { map.insert(text.clone(), json!(lp)); }
                Value::Object(map)
            }
        }).collect::<Vec<_>>(),
        "text_offset":offsets,
    })
}

pub(crate) fn frame_json(
    kind: Kind,
    id: &str,
    model: &str,
    created: u64,
    frame: Frame,
    streaming: bool,
) -> Value {
    let chat = matches!(kind, Kind::Chat);
    let choices = frame
        .choices
        .into_iter()
        .map(|choice| {
            let mut value = json!({"index":choice.index,"finish_reason":choice.finish});
            if chat {
                value[if streaming { "delta" } else { "message" }] = message_json(&choice.message);
                if let Some(lp) = &choice.logprobs {
                    value["logprobs"] = chat_logprobs(lp);
                }
            } else {
                value["text"] = json!(choice.text.unwrap_or_default());
                value["logprobs"] = choice
                    .logprobs
                    .as_ref()
                    .map(completion_logprobs)
                    .unwrap_or(Value::Null);
            }
            value
        })
        .collect::<Vec<_>>();
    let object = match (chat, streaming) {
        (true, true) => "chat.completion.chunk",
        (true, false) => "chat.completion",
        _ => "text_completion",
    };
    let mut value =
        json!({"id":id,"object":object,"created":created,"model":model,"choices":choices});
    if chat {
        value["system_fingerprint"] = json!("superfluid");
    }
    if chat || !streaming {
        if let Some(usage) = &frame.usage {
            value["usage"] = usage_json(usage);
        }
    }
    let mut extra = json!({});
    if let Some(warm) = frame.warm {
        extra["warm"] = json!(warm);
    }
    if let Some((proposed, accepted)) = frame.spec {
        extra["spec"] = json!({"proposed":proposed,"accepted":accepted,"acceptance_rate":if proposed > 0 { accepted as f64 / proposed as f64 } else { 0.0 }});
    }
    if let Some(fim) = frame.fim {
        extra["cached"] = json!(fim.cached);
        extra["expired"] = json!(fim.expired);
        extra["served_by"] = json!(fim.served_by);
    }
    if extra.as_object().is_some_and(|m| !m.is_empty()) {
        value["superfluid"] = extra;
    }
    value
}

pub(crate) fn generation_response(generation: Generation) -> Response {
    let Generation {
        kind,
        id,
        model,
        created,
        warm_header,
        body,
    } = generation;
    match body {
        Body::Complete(frame) => {
            let mut headers = HeaderMap::new();
            if let Some(warm) = warm_header {
                if let Ok(value) = warm.to_string().parse() {
                    headers.insert("x-superfluid-warm", value);
                }
            }
            (
                StatusCode::OK,
                headers,
                Json(frame_json(kind, &id, &model, created, frame, false)),
            )
                .into_response()
        }
        Body::Stream(rx) => {
            let stream = ReceiverStream::new(rx).map(move |frame| {
                let value = match frame {
                    Ok(frame) => frame_json(kind, &id, &model, created, frame, true),
                    Err(error) => {
                        let mut value = json!({"error":{"message":error.message,"type":if error.client_error { "invalid_request_error" } else { "server_error" }}});
                        if let Some(code) = error.code { value["error"]["code"] = json!(code); }
                        if let Some(param) = error.param { value["error"]["param"] = json!(param); }
                        value
                    },
                };
                Event::default().data(value.to_string())
            }).chain(tokio_stream::once(Event::default().data("[DONE]")))
              .map(Ok::<_, std::convert::Infallible>);
            Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_choices_keep_tools_reasoning_logprobs_and_usage() {
        let mut message = Message::complete_tool(
            0,
            "call_1".into(),
            "weather".into(),
            "{\"city\":\"東京\"}".into(),
        );
        message.assistant = true;
        message.content = Some(String::new());
        message.reasoning = Some("check weather".into());
        let frame = Frame {
            choices: vec![
                Choice {
                    message,
                    finish: Some("tool_calls"),
                    ..Choice::default()
                },
                Choice {
                    index: 1,
                    message: Message::content("other answer"),
                    finish: Some("length"),
                    logprobs: Some(Logprobs {
                        tokens: vec![],
                        offset: 0,
                    }),
                    ..Choice::default()
                },
            ],
            usage: Some(Usage::complete(12, 8, Some(4))),
            spec: Some((4, 2)),
            ..Frame::default()
        };
        let value = frame_json(Kind::Chat, "id", "model", 123, frame, false);
        assert_eq!(
            value,
            json!({
                "id":"id","object":"chat.completion","created":123,"model":"model","system_fingerprint":"superfluid",
                "choices":[
                    {"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"","reasoning_content":"check weather","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"東京\"}"}}]}},
                    {"index":1,"finish_reason":"length","message":{"content":"other answer"},"logprobs":{"content":[]}}
                ],
                "usage":{"prompt_tokens":12,"completion_tokens":8,"total_tokens":20,"prompt_tokens_details":{"cached_tokens":4}},
                "superfluid":{"spec":{"proposed":4,"accepted":2,"acceptance_rate":0.5}}
            })
        );
    }

    #[test]
    fn completion_logprobs_keep_byte_offsets_and_null_alternatives() {
        let lp = Logprobs {
            offset: 3,
            tokens: vec![
                TokenProbability {
                    token: "é".into(),
                    logprob: -0.5,
                    alternatives: vec![],
                },
                TokenProbability {
                    token: "x".into(),
                    logprob: -1.0,
                    alternatives: vec![("y".into(), -2.0)],
                },
            ],
        };
        assert_eq!(
            completion_logprobs(&lp),
            json!({"tokens":["é","x"],"token_logprobs":[-0.5,-1.0],"top_logprobs":[null,{"y":-2.0}],"text_offset":[3,5]})
        );
        assert_eq!(chat_logprobs(&lp)["content"][0]["bytes"], json!([195, 169]));
    }

    #[tokio::test]
    async fn completion_sse_keeps_its_wire_contract_and_error_type() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let mut last = Frame::text("", None, Some("stop"));
        last.usage = Some(Usage::complete(2, 3, Some(0)));
        tx.send(Ok(Frame::text("hi", None, None))).await.unwrap();
        tx.send(Ok(last)).await.unwrap();
        tx.send(Err(StreamFailure {
            message: "too long".into(),
            client_error: true,
            code: None,
            param: None,
        }))
        .await
        .unwrap();
        drop(tx);
        let response = generation_response(Generation {
            kind: Kind::Completion,
            id: "id".into(),
            model: "m".into(),
            created: 123,
            warm_header: None,
            body: Body::Stream(rx),
        });
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let bytes = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        let data = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect::<Vec<_>>();
        assert_eq!(data.len(), 4);
        assert_eq!(data[3], "[DONE]");
        let first: Value = serde_json::from_str(data[0]).unwrap();
        assert_eq!(
            first["choices"][0],
            json!({"index":0,"text":"hi","logprobs":null,"finish_reason":null})
        );
        let last: Value = serde_json::from_str(data[1]).unwrap();
        assert!(last.get("usage").is_none());
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        let error: Value = serde_json::from_str(data[2]).unwrap();
        assert_eq!(
            error,
            json!({"error":{"message":"too long","type":"invalid_request_error"}})
        );
    }

    #[tokio::test]
    async fn dropping_sse_closes_the_inference_channel() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let response = generation_response(Generation {
            kind: Kind::Chat,
            id: "id".into(),
            model: "m".into(),
            created: 0,
            warm_header: None,
            body: Body::Stream(rx),
        });
        drop(response);
        assert!(tx.send(Ok(Frame::default())).await.is_err());
    }
}
