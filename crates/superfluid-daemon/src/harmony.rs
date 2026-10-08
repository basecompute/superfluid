//! Harmony, the response format of OpenAI's gpt-oss models.

use std::sync::Arc;

use superfluid_engine::Tokenizer;

use crate::codec::{Channelizer, Run};
use crate::wal::channel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HarmonyIds {
    pub start: u32,
    pub channel: u32,
    pub message: u32,
    pub end: u32,
    pub ret: u32,
    pub call: u32,
    pub constrain: Option<u32>,
}

impl HarmonyIds {
    pub fn discover(specials: &[(String, u32)]) -> Option<HarmonyIds> {
        let find = |name: &str| specials.iter().find(|(s, _)| s == name).map(|(_, i)| *i);
        Some(HarmonyIds {
            start: find("<|start|>")?,
            channel: find("<|channel|>")?,
            message: find("<|message|>")?,
            end: find("<|end|>")?,
            ret: find("<|return|>")?,
            call: find("<|call|>")?,
            constrain: find("<|constrain|>"),
        })
    }

    fn is_marker(&self, t: u32) -> bool {
        t == self.start
            || t == self.channel
            || t == self.message
            || t == self.end
            || t == self.ret
            || t == self.call
            || Some(t) == self.constrain
    }
}

fn classify(header: &str) -> u32 {
    let h = header.trim_start();
    if h.contains("to=") {
        channel::TOOL_CALL
    } else if h.starts_with("analysis") {
        channel::REASONING
    } else {
        channel::TEXT
    }
}

enum State {
    Header,
    Body(u32),
}

pub struct HarmonyChannelizer {
    tok: Arc<dyn Tokenizer>,
    ids: HarmonyIds,
    state: State,
    header: String,
    header_tokens: Vec<u32>,
    header_cut: Option<usize>,
    recipient: Vec<u32>,
}

impl HarmonyChannelizer {
    pub fn new(tok: Arc<dyn Tokenizer>, ids: HarmonyIds) -> HarmonyChannelizer {
        HarmonyChannelizer {
            tok,
            ids,
            state: State::Header,
            header: String::new(),
            header_tokens: Vec::new(),
            header_cut: None,
            recipient: Vec::new(),
        }
    }

    fn reset_header(&mut self) {
        self.header.clear();
        self.header_tokens.clear();
        self.header_cut = None;
    }

    fn recipient_tokens(&self) -> Vec<u32> {
        let end = self.header_cut.unwrap_or(self.header_tokens.len());
        let Some(pos) = self.header.find("to=") else {
            return Vec::new();
        };
        let mut at = 0;
        for (i, &t) in self.header_tokens[..end].iter().enumerate() {
            at += String::from_utf8_lossy(&self.tok.token_bytes(t)).len();
            if at > pos {
                return self.header_tokens[i..end].to_vec();
            }
        }
        Vec::new()
    }
}

impl Channelizer for HarmonyChannelizer {
    fn split(&mut self, tokens: &[u32]) -> Vec<Run> {
        let mut out: Vec<Run> = Vec::new();
        let mut span: Vec<u32> = Vec::new();
        let mut text: Vec<u32> = Vec::new();
        for &t in tokens {
            match self.state {
                State::Header => {
                    span.push(t);
                    if t == self.ids.message {
                        let mut header = self.header.clone();
                        if !self.recipient.is_empty() {
                            header.push(' ');
                            header.push_str(&String::from_utf8_lossy(&self.recipient.iter().flat_map(|&r| self.tok.token_bytes(r)).collect::<Vec<u8>>()));
                        }
                        let ch = classify(&header);
                        if ch == channel::TOOL_CALL {
                            let cut = self.header_cut.unwrap_or(self.header_tokens.len());
                            text.extend(self.header_tokens[..cut].iter().copied());
                            text.extend(self.recipient.iter().copied());
                        }
                        self.reset_header();
                        self.recipient.clear();
                        self.state = State::Body(ch);
                    } else if t == self.ids.start {
                        self.reset_header();
                        self.recipient.clear();
                    } else if t == self.ids.channel {
                        let carried = self.recipient_tokens();
                        self.reset_header();
                        if !carried.is_empty() {
                            self.recipient = carried;
                        }
                    } else if Some(t) == self.ids.constrain {
                        self.header_cut.get_or_insert(self.header_tokens.len());
                    } else if self.ids.is_marker(t) {
                    } else {
                        self.header.push_str(&String::from_utf8_lossy(&self.tok.token_bytes(t)));
                        self.header_tokens.push(t);
                    }
                }
                State::Body(ch) => {
                    if t == self.ids.end || t == self.ids.ret || t == self.ids.call {
                        span.push(t);
                        out.push(Run {
                            span: std::mem::take(&mut span),
                            text: std::mem::take(&mut text),
                            channel: ch,
                            closes: true,
                        });
                        self.state = State::Header;
                    } else if t == self.ids.start || t == self.ids.channel {
                        if !span.is_empty() {
                            out.push(Run {
                                span: std::mem::take(&mut span),
                                text: std::mem::take(&mut text),
                                channel: ch,
                                closes: false,
                            });
                        }
                        span.push(t);
                        self.reset_header();
                        self.state = State::Header;
                    } else if self.ids.is_marker(t) {
                        span.push(t);
                    } else {
                        span.push(t);
                        text.push(t);
                    }
                }
            }
        }
        if !span.is_empty() {
            out.push(Run { span, text, channel: self.current_channel(), closes: false });
        }
        out
    }

    fn current_channel(&self) -> u32 {
        match self.state {
            State::Header => channel::TEXT,
            State::Body(ch) => ch,
        }
    }

    fn prime(&mut self, channel: u32) {
        self.state = State::Body(channel);
    }
}

pub fn parse_harmony_tool_call(raw: &str) -> Option<(String, String)> {
    let (name, after) = recipient(raw)?;
    let body = after.get(after.find('{')?..)?;
    let args = crate::codec::repair_json_object(body)?;
    Some((name, args))
}

fn recipient(raw: &str) -> Option<(String, &str)> {
    let p = raw.find("to=")?;
    let rest = &raw[p + 3..];
    let len = rest.find(|c: char| c.is_whitespace() || c == '<' || c == '{').unwrap_or(rest.len());
    let recipient = &rest[..len];
    let name = recipient.strip_prefix("functions.").unwrap_or(recipient);
    if name.is_empty() {
        return None;
    }
    Some((name.to_string(), &rest[len..]))
}

pub fn named_call(raw: &str) -> Option<(String, String)> {
    let (name, after) = recipient(raw)?;
    let args = after.find('{').map_or("", |i| &after[i..]);
    Some((name, args.to_string()))
}

fn tag_token(id: u32) -> serde_json::Value {
    serde_json::json!({ "type": "token", "token": id })
}

fn tag_text(s: &str) -> serde_json::Value {
    serde_json::json!({ "type": "const_string", "value": s })
}

fn markers(ids: HarmonyIds) -> Vec<u32> {
    let mut markers = vec![ids.start, ids.channel, ids.message, ids.end, ids.ret, ids.call];
    markers.extend(ids.constrain);
    markers
}

fn body_then_next_message(ids: HarmonyIds) -> [serde_json::Value; 3] {
    [
        serde_json::json!({ "type": "tag", "begin": tag_token(ids.message),
          "content": { "type": "any_tokens", "exclude_tokens": markers(ids) },
          "end": tag_token(ids.end) }),
        tag_token(ids.start),
        tag_text("assistant"),
    ]
}

fn optional_analysis(ids: HarmonyIds) -> serde_json::Value {
    let [body, start, assistant] = body_then_next_message(ids);
    serde_json::json!({ "type": "optional", "content": { "type": "sequence", "elements": [
        tag_token(ids.channel), tag_text("analysis"), body, start, assistant,
    ]}})
}

fn optional_content_type(ids: HarmonyIds) -> serde_json::Value {
    let written = match ids.constrain {
        Some(c) => serde_json::json!({ "type": "or", "elements": [
            { "type": "sequence", "elements": [
                { "type": "optional", "content": tag_text(" ") }, tag_token(c), tag_text("json"),
            ]},
            tag_text(" json"),
        ]}),
        None => tag_text(" json"),
    };
    serde_json::json!({ "type": "optional", "content": written })
}

fn structural(elements: Vec<serde_json::Value>) -> String {
    serde_json::json!({ "type": "structural_tag", "format": { "type": "sequence", "elements": elements } }).to_string()
}

pub fn forced_call_tag(ids: HarmonyIds, tool_jsons: &[String]) -> Option<String> {
    use serde_json::json;
    let mut calls: Vec<serde_json::Value> = declared_functions(tool_jsons)?
        .into_iter()
        .map(|(name, params)| {
            json!({ "type": "sequence", "elements": [
                tag_token(ids.channel),
                tag_text(&format!("commentary to=functions.{name}")),
                optional_content_type(ids),
                tag_token(ids.message),
                { "type": "json_schema", "json_schema": params },
            ]})
        })
        .collect();
    let call = if calls.len() == 1 { calls.remove(0) } else { json!({ "type": "or", "elements": calls }) };
    Some(structural(vec![optional_analysis(ids), call]))
}

fn declared_functions(tool_jsons: &[String]) -> Option<Vec<(String, serde_json::Value)>> {
    if tool_jsons.is_empty() {
        return None;
    }
    tool_jsons
        .iter()
        .map(|t| {
            let v: serde_json::Value = serde_json::from_str(t).ok()?;
            let f = v.get("function").unwrap_or(&v);
            let name = f.get("name").and_then(|n| n.as_str())?.to_string();
            let params = f.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({ "type": "object" }));
            Some((name, params))
        })
        .collect()
}

pub const FREE_MESSAGES: usize = 3;

pub fn auto_call_tag(ids: HarmonyIds, tool_jsons: &[String]) -> Option<String> {
    use serde_json::json;
    let channel_of_call = || json!({ "type": "or", "elements": [tag_text("commentary"), tag_text("analysis")] });
    let mut last = vec![json!({ "type": "sequence", "elements": [
        tag_token(ids.channel), tag_text("final"), optional_content_type(ids), tag_token(ids.message),
        { "type": "any_tokens", "exclude_tokens": markers(ids) },
    ]})];
    for (name, params) in declared_functions(tool_jsons)? {
        let to = tag_text(&format!(" to=functions.{name}"));
        last.push(json!({ "type": "sequence", "elements": [
            { "type": "or", "elements": [
                { "type": "sequence", "elements": [tag_token(ids.channel), channel_of_call(), to.clone()] },
                { "type": "sequence", "elements": [to, tag_token(ids.channel), channel_of_call()] },
            ]},
            optional_content_type(ids),
            tag_token(ids.message),
            { "type": "json_schema", "json_schema": params },
            { "type": "star", "content": tag_text(" ") },
        ]}));
    }
    let free_message = || {
        let [body, start, assistant] = body_then_next_message(ids);
        json!({ "type": "sequence", "elements": [
            tag_token(ids.channel),
            { "type": "or", "elements": [tag_text("analysis"), tag_text("commentary")] },
            optional_content_type(ids),
            body, start, assistant,
        ]})
    };
    let mut before = json!({ "type": "optional", "content": free_message() });
    for _ in 1..FREE_MESSAGES {
        before = json!({ "type": "optional", "content": { "type": "sequence", "elements": [free_message(), before] } });
    }
    Some(structural(vec![before, json!({ "type": "or", "elements": last })]))
}

pub fn final_json_tag(ids: HarmonyIds, json_schema: &str) -> Option<String> {
    let schema: serde_json::Value = serde_json::from_str(json_schema).ok()?;
    Some(structural(vec![
        optional_analysis(ids),
        tag_token(ids.channel),
        tag_text("final"),
        optional_content_type(ids),
        tag_token(ids.message),
        serde_json::json!({ "type": "json_schema", "json_schema": schema }),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tok;
    const START: u32 = 1000;
    const CHANNEL: u32 = 1001;
    const MESSAGE: u32 = 1002;
    const END: u32 = 1003;
    const RETURN: u32 = 1004;
    const CALL: u32 = 1005;
    const CONSTRAIN: u32 = 1006;

    fn ids() -> HarmonyIds {
        HarmonyIds { start: START, channel: CHANNEL, message: MESSAGE, end: END, ret: RETURN, call: CALL, constrain: Some(CONSTRAIN) }
    }

    impl Tokenizer for Tok {
        fn encode(&self, text: &str) -> Vec<u32> {
            text.bytes().map(u32::from).collect()
        }
        fn token_bytes(&self, token: u32) -> Vec<u8> {
            if token < 256 {
                vec![token as u8]
            } else {
                Vec::new()
            }
        }
        fn vocab_size(&self) -> u32 {
            1100
        }
        fn special_tokens(&self) -> Vec<(String, u32)> {
            vec![
                ("<|start|>".into(), START),
                ("<|channel|>".into(), CHANNEL),
                ("<|message|>".into(), MESSAGE),
                ("<|end|>".into(), END),
                ("<|return|>".into(), RETURN),
                ("<|call|>".into(), CALL),
                ("<|constrain|>".into(), CONSTRAIN),
            ]
        }
        fn bos_token(&self) -> Option<u32> {
            None
        }
        fn eos_token(&self) -> u32 {
            RETURN
        }
        fn chat_template_jinja(&self) -> String {
            String::new()
        }
    }

    fn bytes(s: &str) -> Vec<u32> {
        s.bytes().map(u32::from).collect()
    }

    fn stream(parts: &[&[u32]]) -> Vec<u32> {
        parts.iter().flat_map(|p| p.iter().copied()).collect()
    }

    fn decode(text: &[u32]) -> String {
        String::from_utf8(text.iter().flat_map(|&t| Tok.token_bytes(t)).collect()).unwrap()
    }

    #[test]
    fn discovers_the_markers_from_the_specials() {
        assert_eq!(HarmonyIds::discover(&Tok.special_tokens()), Some(ids()));
        let mut fewer = Tok.special_tokens();
        fewer.retain(|(s, _)| s != "<|return|>");
        assert_eq!(HarmonyIds::discover(&fewer), None, "every marker the format needs");
        fewer = Tok.special_tokens();
        fewer.retain(|(s, _)| s != "<|constrain|>");
        assert!(HarmonyIds::discover(&fewer).is_some(), "the constraint marker is optional");
    }

    #[test]
    fn routes_analysis_and_final() {
        let toks = stream(&[
            &[CHANNEL],
            &bytes("analysis"),
            &[MESSAGE],
            &bytes("think hard"),
            &[END, START],
            &bytes("assistant"),
            &[CHANNEL],
            &bytes("final"),
            &[MESSAGE],
            &bytes("Hello!"),
            &[RETURN],
        ]);
        let mut ch = HarmonyChannelizer::new(Arc::new(Tok), ids());
        let runs = ch.split(&toks);
        let joined: Vec<u32> = runs.iter().flat_map(|r| r.span.iter().copied()).collect();
        assert_eq!(joined, toks, "spans partition the input");
        let routed: Vec<(u32, String, bool)> =
            runs.iter().map(|r| (r.channel, decode(&r.text), r.closes)).collect();
        assert_eq!(
            routed,
            vec![
                (channel::REASONING, "think hard".into(), true),
                (channel::TEXT, "Hello!".into(), true),
            ]
        );
    }

    #[test]
    fn routes_a_tool_call_with_its_header() {
        let toks = stream(&[
            &[CHANNEL],
            &bytes("commentary to=functions.get_weather "),
            &[CONSTRAIN],
            &bytes("json"),
            &[MESSAGE],
            &bytes(r#"{"city":"Paris"}"#),
            &[CALL],
        ]);
        let mut ch = HarmonyChannelizer::new(Arc::new(Tok), ids());
        let runs = ch.split(&toks);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].channel, channel::TOOL_CALL);
        assert!(runs[0].closes);
        let raw = decode(&runs[0].text);
        assert_eq!(raw, r#"commentary to=functions.get_weather {"city":"Paris"}"#, "the type is dropped");
        assert_eq!(parse_harmony_tool_call(&raw), Some(("get_weather".into(), r#"{"city":"Paris"}"#.into())));
    }

    #[test]
    fn a_recipient_glued_to_the_constraint_marker_keeps_its_name() {
        let toks = stream(&[
            &[CHANNEL],
            &bytes("commentary to=functions.bash"),
            &[CONSTRAIN],
            &bytes("json"),
            &[MESSAGE],
            &bytes(r#"{"command":"ls -la"}"#),
            &[CALL],
        ]);
        let mut ch = HarmonyChannelizer::new(Arc::new(Tok), ids());
        let runs = ch.split(&toks);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].channel, channel::TOOL_CALL);
        let raw = decode(&runs[0].text);
        assert_eq!(raw, r#"commentary to=functions.bash{"command":"ls -la"}"#);
        assert_eq!(parse_harmony_tool_call(&raw), Some(("bash".into(), r#"{"command":"ls -la"}"#.into())));
        assert_eq!(parse_harmony_tool_call(r#"commentary to=functions.bash json{"command":"ls"}"#), Some(("bash".into(), r#"{"command":"ls"}"#.into())));
    }

    #[test]
    fn a_recipient_before_the_channel_still_makes_a_call() {
        let calls = |toks: &[u32]| -> Vec<(u32, String, bool)> {
            let mut ch = HarmonyChannelizer::new(Arc::new(Tok), ids());
            ch.split(toks).into_iter().filter(|r| !r.text.is_empty()).map(|r| (r.channel, decode(&r.text), r.closes)).collect()
        };
        let role_header = stream(&[&[START], &bytes("assistant to=functions.bash"), &[CHANNEL], &bytes("commentary json"), &[MESSAGE], &bytes(r#"{"command":"ls"}"#), &[CALL]]);
        let got = calls(&role_header);
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].0, got[0].2), (channel::TOOL_CALL, true), "{got:?}");
        assert_eq!(parse_harmony_tool_call(&got[0].1), Some(("bash".into(), r#"{"command":"ls"}"#.into())), "{got:?}");

        let restarted = stream(&[&[CHANNEL], &bytes("commentary to=functions.read"), &[CHANNEL], &bytes("analysis code"), &[MESSAGE], &bytes(r#"{"path":"a"}"#), &[CALL]]);
        let got = calls(&restarted);
        assert_eq!(got[0].0, channel::TOOL_CALL, "{got:?}");
        assert_eq!(parse_harmony_tool_call(&got[0].1), Some(("read".into(), r#"{"path":"a"}"#.into())), "{got:?}");

        let mut ch = HarmonyChannelizer::new(Arc::new(Tok), ids());
        let sliced: Vec<Run> = role_header.iter().flat_map(|&t| ch.split(&[t])).collect();
        let text: Vec<u32> = sliced.iter().filter(|r| r.channel == channel::TOOL_CALL).flat_map(|r| r.text.clone()).collect();
        assert_eq!(parse_harmony_tool_call(&decode(&text)).map(|(n, _)| n), Some("bash".into()));

        let answer = stream(&[&[START], &bytes("assistant"), &[CHANNEL], &bytes("final"), &[MESSAGE], &bytes("hi"), &[RETURN]]);
        assert_eq!(calls(&answer), vec![(channel::TEXT, "hi".into(), true)]);
        let forgotten = stream(&[&[START], &bytes("assistant to=functions.f"), &[START], &bytes("assistant"), &[CHANNEL], &bytes("final"), &[MESSAGE], &bytes("x"), &[RETURN]]);
        assert_eq!(calls(&forgotten), vec![(channel::TEXT, "x".into(), true)]);
    }

    #[test]
    fn preambles_are_text_and_builtin_tools_are_calls() {
        let toks = stream(&[
            &[CHANNEL],
            &bytes("commentary"),
            &[MESSAGE],
            &bytes("Let me check."),
            &[END, START],
            &bytes("assistant"),
            &[CHANNEL],
            &bytes("analysis to=browser.search "),
            &[CONSTRAIN],
            &bytes("json"),
            &[MESSAGE],
            &bytes(r#"{"query":"weather"}"#),
            &[CALL],
        ]);
        let mut ch = HarmonyChannelizer::new(Arc::new(Tok), ids());
        let runs = ch.split(&toks);
        assert_eq!(runs[0].channel, channel::TEXT);
        assert_eq!(decode(&runs[0].text), "Let me check.");
        assert_eq!(runs[1].channel, channel::TOOL_CALL);
        assert_eq!(parse_harmony_tool_call(&decode(&runs[1].text)), Some(("browser.search".into(), r#"{"query":"weather"}"#.into())));
    }

    #[test]
    fn slicing_anywhere_routes_like_the_whole() {
        let toks = stream(&[
            &[CHANNEL],
            &bytes("analysis"),
            &[MESSAGE],
            &bytes("hmm"),
            &[END, START],
            &bytes("assistant"),
            &[CHANNEL],
            &bytes("commentary to=functions.f "),
            &[CONSTRAIN],
            &bytes("json"),
            &[MESSAGE],
            &bytes(r#"{"a":1}"#),
            &[CALL],
        ]);
        let whole: Vec<(u32, Vec<u32>)> = {
            let mut ch = HarmonyChannelizer::new(Arc::new(Tok), ids());
            merge(ch.split(&toks))
        };
        for cut in 1..toks.len() {
            let mut ch = HarmonyChannelizer::new(Arc::new(Tok), ids());
            let mut runs = ch.split(&toks[..cut]);
            runs.extend(ch.split(&toks[cut..]));
            assert_eq!(merge(runs), whole, "cut at {cut}");
        }
    }

    fn merge(runs: Vec<Run>) -> Vec<(u32, Vec<u32>)> {
        let mut out: Vec<(u32, Vec<u32>)> = Vec::new();
        for r in runs {
            if r.text.is_empty() {
                continue;
            }
            match out.last_mut() {
                Some((ch, text)) if *ch == r.channel => text.extend(r.text),
                _ => out.push((r.channel, r.text)),
            }
        }
        out
    }

    #[test]
    fn a_stop_mid_message_closes_nothing_and_a_new_header_recovers() {
        let toks = stream(&[&[CHANNEL], &bytes("final"), &[MESSAGE], &bytes("partial"), &[START], &bytes("assistant"), &[CHANNEL], &bytes("final"), &[MESSAGE], &bytes("again")]);
        let mut ch = HarmonyChannelizer::new(Arc::new(Tok), ids());
        let runs = ch.split(&toks);
        let texts: Vec<(String, bool)> = runs.iter().filter(|r| !r.text.is_empty()).map(|r| (decode(&r.text), r.closes)).collect();
        assert_eq!(texts, vec![("partial".into(), false), ("again".into(), false)]);
        assert_eq!(ch.current_channel(), channel::TEXT);
    }

    #[test]
    fn the_forced_call_tag_matches_markers_by_id_and_bodies_by_schema() {
        use serde_json::json;
        let weather = r#"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}"#.to_string();
        let now = r#"{"name":"now"}"#.to_string();
        let tag: serde_json::Value = serde_json::from_str(&forced_call_tag(ids(), &[weather.clone(), now]).expect("a tag")).unwrap();
        assert_eq!(tag["type"], "structural_tag");
        let seq = tag["format"]["elements"].as_array().expect("a sequence");
        assert_eq!(seq.len(), 2, "an optional analysis, then the call");

        assert_eq!(seq[0]["type"], "optional");
        let analysis = seq[0]["content"]["elements"].as_array().expect("the analysis message");
        assert_eq!(analysis[0], json!({"type": "token", "token": CHANNEL}), "markers are matched by id");
        assert_eq!(analysis[1], json!({"type": "const_string", "value": "analysis"}));
        let body = &analysis[2];
        assert_eq!(body["type"], "tag", "the unbounded body is closed by its end marker");
        assert_eq!(body["begin"], json!({"type": "token", "token": MESSAGE}));
        assert_eq!(body["end"], json!({"type": "token", "token": END}));
        assert_eq!(body["content"]["type"], "any_tokens");
        let excluded: Vec<u32> = serde_json::from_value(body["content"]["exclude_tokens"].clone()).unwrap();
        for marker in [START, CHANNEL, MESSAGE, END, RETURN, CALL, CONSTRAIN] {
            assert!(excluded.contains(&marker), "the analysis body cannot contain marker {marker}");
        }
        assert_eq!(&analysis[3..], &[json!({"type": "token", "token": START}), json!({"type": "const_string", "value": "assistant"})]);

        let calls = seq[1]["elements"].as_array().expect("one alternative per tool");
        assert_eq!(seq[1]["type"], "or");
        assert_eq!(calls.len(), 2);
        let first = calls[0]["elements"].as_array().unwrap();
        assert_eq!(first[0], json!({"type": "token", "token": CHANNEL}));
        assert_eq!(first[1], json!({"type": "const_string", "value": "commentary to=functions.get_weather"}));
        assert_eq!(first[2]["type"], "optional", "the content type may be written or left out");
        assert_eq!(first[3], json!({"type": "token", "token": MESSAGE}));
        assert_eq!(first[4]["json_schema"]["required"], json!(["city"]), "the body is the tool's own parameters");
        assert_eq!(first.len(), 5, "the tag ends with the body; the turn's stop token ends the call");
        assert_eq!(calls[1]["elements"][4]["json_schema"], json!({"type": "object"}), "no parameters: any object");

        let one: serde_json::Value = serde_json::from_str(&forced_call_tag(ids(), &[weather]).unwrap()).unwrap();
        assert_eq!(one["format"]["elements"][1]["type"], "sequence", "one tool needs no alternation");
        assert!(forced_call_tag(ids(), &[]).is_none(), "no tools, no tag");
        assert!(forced_call_tag(ids(), &[r#"{"parameters":{}}"#.to_string()]).is_none(), "a nameless tool, no tag");
    }

    #[test]
    fn the_answer_tag_holds_the_schema_in_the_final_channel() {
        use serde_json::json;
        let schema = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}"#;
        let tag: serde_json::Value = serde_json::from_str(&final_json_tag(ids(), schema).expect("a tag")).unwrap();
        let seq = tag["format"]["elements"].as_array().expect("a sequence");
        assert_eq!(seq[0]["type"], "optional", "the model may reason first");
        assert_eq!(seq[0]["content"]["elements"][1], json!({"type": "const_string", "value": "analysis"}));
        assert_eq!(&seq[1..3], &[json!({"type": "token", "token": CHANNEL}), json!({"type": "const_string", "value": "final"})]);
        assert_eq!(seq[3]["type"], "optional", "the content type may be written or left out");
        assert_eq!(seq[4], json!({"type": "token", "token": MESSAGE}));
        assert_eq!(seq[5], json!({"type": "json_schema", "json_schema": serde_json::from_str::<serde_json::Value>(schema).unwrap()}));
        assert_eq!(seq.len(), 6, "the value is the last thing; `<|return|>` is the stop that ends it");
        assert!(final_json_tag(ids(), "{not json").is_none());
    }

    #[test]
    fn the_parser_repairs_a_truncated_body_and_refuses_the_rest() {
        assert_eq!(parse_harmony_tool_call(r#"commentary to=functions.f json{"a":1,"b":[1,2"#), Some(("f".into(), r#"{"a":1,"b":[1,2]}"#.into())));
        assert_eq!(parse_harmony_tool_call("final no recipient here {}"), None);
        assert_eq!(parse_harmony_tool_call("commentary to=functions.f json"), None, "no body");
        assert_eq!(parse_harmony_tool_call("commentary to= json{}"), None, "no name");
    }

    #[test]
    fn a_call_whose_arguments_do_not_parse_is_still_named_by_its_header() {
        let raw = r#"commentary to=functions.write {"path":"server.py","content":"main()\n","}"#;
        assert_eq!(parse_harmony_tool_call(raw), None, "the arguments are not JSON");
        assert_eq!(named_call(raw), Some(("write".into(), r#"{"path":"server.py","content":"main()\n","}"#.into())));
        assert_eq!(named_call(r#"commentary to=functions.bash json{"command":"ls"#), Some(("bash".into(), r#"{"command":"ls"#.into())), "the type is not the arguments");
        assert_eq!(named_call("commentary to=functions.f json"), Some(("f".into(), String::new())), "no body, no arguments");
        assert_eq!(named_call("final no recipient here {}"), None);
        assert_eq!(named_call("commentary to= json{}"), None, "no name");
    }

    #[test]
    fn the_auto_tag_holds_only_a_calls_arguments() {
        use serde_json::json;
        let weather = r#"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}"#.to_string();
        let now = r#"{"name":"now"}"#.to_string();
        let tag: serde_json::Value = serde_json::from_str(&auto_call_tag(ids(), &[weather, now]).expect("a tag")).unwrap();
        let seq = tag["format"]["elements"].as_array().expect("a sequence");
        assert_eq!(seq.len(), 2, "the messages before the last, then the last");

        let none = json!(null);
        let mut depth = 0;
        let mut free = &seq[0];
        while free["type"] == "optional" {
            depth += 1;
            let inner = &free["content"];
            let message = if inner["elements"][0]["type"] == "sequence" { &inner["elements"][0] } else { inner };
            let parts = message["elements"].as_array().expect("a message");
            assert_eq!(parts[0], json!({"type": "token", "token": CHANNEL}));
            assert_eq!(parts[1], json!({"type": "or", "elements": [
                {"type": "const_string", "value": "analysis"}, {"type": "const_string", "value": "commentary"}]}));
            assert_eq!(parts[3]["type"], "tag", "an unbounded body closed by its end marker");
            assert_eq!(parts[3]["end"], json!({"type": "token", "token": END}));
            assert_eq!(&parts[4..], &[json!({"type": "token", "token": START}), json!({"type": "const_string", "value": "assistant"})]);
            free = if inner["elements"][0]["type"] == "sequence" { &inner["elements"][1] } else { &none };
        }
        assert_eq!(depth, FREE_MESSAGES);

        assert_eq!(seq[1]["type"], "or");
        let last = seq[1]["elements"].as_array().unwrap();
        assert_eq!(last.len(), 3, "the answer, then one call per tool");
        let answer = last[0]["elements"].as_array().unwrap();
        assert_eq!(&answer[..2], &[json!({"type": "token", "token": CHANNEL}), json!({"type": "const_string", "value": "final"})]);
        assert_eq!(answer.last().unwrap()["type"], "any_tokens", "the answer runs to the stop token");
        for (call, params) in last[1..].iter().zip([json!(["city"]), json!(null)]) {
            let parts = call["elements"].as_array().unwrap();
            assert_eq!(parts[0]["type"], "or", "the recipient after the channel, or before it");
            let body = parts.iter().find(|p| p["type"] == "json_schema").expect("the arguments");
            assert_eq!(body["json_schema"]["required"], params);
            assert_eq!(parts.last().unwrap()["type"], "star", "every branch of the last message runs to the stop token");
        }
        assert!(auto_call_tag(ids(), &[]).is_none(), "no tools, no tag");
        assert!(auto_call_tag(ids(), &[r#"{"parameters":{}}"#.to_string()]).is_none(), "a nameless tool, no tag");
    }
}
