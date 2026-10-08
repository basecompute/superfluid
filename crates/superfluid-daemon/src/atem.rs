use std::sync::Arc;

use superfluid_abi::Status;
use superfluid_engine::Tokenizer;

use crate::codec::{BundleCodec, Channelizer, Run, TextCodec, TurnState};
use crate::wal::{channel, role};

pub struct AtemCodec {
    inner: BundleCodec,
    tok_start: u32,
    tok_message: u32,
    tok_eom: u32,
    tok_eot: u32,
    bos: Option<u32>,
}

impl AtemCodec {
    #[cfg(feature = "basert")]
    pub fn load(model_path: &std::path::Path) -> Result<AtemCodec, Status> {
        let tok = superfluid_engine_ffi::TokenizerHandle::load(model_path).map_err(|_| Status::Fatal)?;
        Self::from_tokenizer(Arc::new(tok))
    }

    pub fn from_tokenizer(tok: Arc<dyn Tokenizer>) -> Result<AtemCodec, Status> {
        let inner = BundleCodec::from_tokenizer(tok);
        let bos = inner.tokenizer().bos_token();
        let one = |s: &str| -> Result<u32, Status> {
            let mut t = inner.encode(s);
            if bos.is_some() && t.first() == bos.as_ref() {
                t.remove(0);
            }
            if t.len() == 1 {
                Ok(t[0])
            } else {
                Err(Status::Unsupported)
            }
        };
        Ok(AtemCodec {
            tok_start: one("<|start|>")?,
            tok_message: one("<|message|>")?,
            tok_eom: one("<|eom|>")?,
            tok_eot: one("<|eot|>")?,
            bos,
            inner,
        })
    }

    fn enc(&self, text: &str) -> Vec<u32> {
        let mut t = self.inner.encode(text);
        if self.bos.is_some() && t.first() == self.bos.as_ref() {
            t.remove(0);
        }
        t
    }

    fn enc_content(&self, text: &str) -> Vec<u32> {
        let mut t = self.inner.encode_content(text);
        if self.bos.is_some() && t.first() == self.bos.as_ref() {
            t.remove(0);
        }
        t
    }

    fn enc_pieces(&self, pieces: &[(&str, bool)]) -> Vec<u32> {
        let mut t = self.inner.encode_pieces(pieces);
        if self.bos.is_some() && t.first() == self.bos.as_ref() {
            t.remove(0);
        }
        t
    }

    fn turn(&self, header: &[(&str, bool)], body: &[(&str, bool)]) -> Vec<u32> {
        let mut span = vec![self.tok_start];
        span.extend(self.enc_pieces(header));
        span.push(self.tok_message);
        span.extend(self.enc_pieces(body));
        span.push(self.tok_eot);
        span
    }

    fn system_tail(&self, tool_jsons: &[String]) -> String {
        let mut body = String::new();
        body.push_str("\n\nReasoning strength: high.");
        let mut namespaces: Vec<String> = Vec::new();
        if !tool_jsons.is_empty() {
            body.push_str("\n\n");
            body.push_str(&render_tool_defs(tool_jsons, &mut namespaces));
        }
        body.push_str("\n\n# Valid recipients: \"self\"");
        for ns in &namespaces {
            body.push_str(&format!(", \"{ns}.*\""));
        }
        body.push_str(", \"user\".");
        body
    }
}

fn render_tool_defs(tool_jsons: &[String], namespaces: &mut Vec<String>) -> String {
    let mut s = String::new();
    s.push_str(
        "In this environment you have access to a set of tools you can use to answer \
         the user's question.\n\n",
    );
    s.push_str(
        "You can invoke a function by writing a \"<atem:function_calls>\" block like \
         the following:\n<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n\
         <atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n\
         </atem:invoke>\n</atem:function_calls>\n\n",
    );
    s.push_str(
        "String and scalar parameters should be specified as is, while lists and objects \
         should use JSON format. Note that spaces for string values are not stripped. The \
         output is not expected to be valid XML and is parsed with regular expressions.\n",
    );
    s.push_str("Here are the functions available in JSONSchema format:\n");
    s.push_str("// Tool metadata\n");
    let tools: Vec<serde_json::Value> = tool_jsons
        .iter()
        .filter_map(|t| serde_json::from_str::<serde_json::Value>(t).ok())
        .map(|t| t.get("function").cloned().unwrap_or(t))
        .collect();
    for t in &tools {
        if let Some(name) = t.get("name").and_then(|n| n.as_str()) {
            let ns = name.split('.').next().unwrap_or(name).to_string();
            if !namespaces.contains(&ns) {
                namespaces.push(ns);
            }
        }
    }
    for ns in namespaces.iter() {
        s.push_str(&format!(
            "{{\"name\": {}, \"description\": \"\"}}\n",
            serde_json::json!(ns)
        ));
    }
    s.push_str("// Function schemas");
    for t in &tools {
        let name = t.get("name").cloned().unwrap_or(serde_json::json!(""));
        let desc = t
            .get("description")
            .cloned()
            .unwrap_or(serde_json::json!(""));
        let params = t
            .get("parameters")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        s.push_str(&format!(
            "\n{{\"name\": {name}, \"description\": {desc}, \"parameters\": {params}}}"
        ));
    }
    s.push_str("\n\nHere's an example of how to call a function in the tool set:\n");
    s.push_str(
        "(If the tool namespace is not specified, invoke the function directly as \
         `example_function_name` rather than `example_tool_name.example_function_name`)\n\n",
    );
    s.push_str("to=example_tool_name.example_function_name\n\n");
    s.push_str(
        "<atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n\
         <atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n\
         <atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\n\
         that can span\n\"multiple\" lines\n</atem:parameter>\n\
         </atem:invoke>\n</atem:function_calls>",
    );
    s
}

pub fn render_atem_for_test(name: &str, arguments: &str) -> String {
    render_atem(name, arguments)
}

fn render_atem(name: &str, arguments: &str) -> String {
    render_atem_pieces(name, arguments).into_iter().map(|(s, _)| s).collect()
}

type Pieces<'a, const N: usize> = [(&'a str, bool); N];

fn tool_result_pieces<'a>(name: &'a str, content: &'a str) -> (Pieces<'a, 2>, Pieces<'a, 5>) {
    (
        [("tool ", false), (name, true)],
        [
            ("<tool_output name=\"", false),
            (name, true),
            ("\">\n", false),
            (content, true),
            ("\n</tool_output>", false),
        ],
    )
}

fn render_atem_pieces(name: &str, arguments: &str) -> Vec<(String, bool)> {
    let mut p: Vec<(String, bool)> = vec![
        ("<atem:function_calls>\n<atem:invoke name=\"".to_string(), false),
        (name.to_string(), true),
        ("\">\n".to_string(), false),
    ];
    if let Ok(serde_json::Value::Object(args)) = serde_json::from_str(arguments) {
        for (k, v) in args {
            p.push(("<atem:parameter name=\"".to_string(), false));
            p.push((k, true));
            p.push(("\">".to_string(), false));
            let value = match v {
                serde_json::Value::Bool(true) => "true".to_string(),
                serde_json::Value::Bool(false) => "false".to_string(),
                serde_json::Value::Null => "null".to_string(),
                serde_json::Value::String(v) => v,
                serde_json::Value::Number(n) => n.to_string(),
                other => other.to_string(),
            };
            p.push((value, true));
            p.push(("</atem:parameter>\n".to_string(), false));
        }
    }
    p.push(("</atem:invoke>\n</atem:function_calls>".to_string(), false));
    p
}

impl TextCodec for AtemCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.enc(text)
    }
    fn encode_content(&self, text: &str) -> Vec<u32> {
        self.enc_content(text)
    }
    fn encode_pieces(&self, pieces: &[(&str, bool)]) -> Vec<u32> {
        self.enc_pieces(pieces)
    }
    fn chat_template_source(&self) -> Option<String> {
        self.inner.chat_template_source()
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        self.inner.token_bytes(token)
    }
    fn marker_literal(&self, token: u32) -> String {
        self.inner.marker_literal(token)
    }

    fn stream_prologue(&self) -> Vec<u32> {
        self.bos.into_iter().collect()
    }

    fn render_message(&self, msg_role: u32, text: &str) -> Option<Vec<u32>> {
        match msg_role {
            role::SYSTEM => {
                let tail = self.system_tail(&[]);
                Some(self.turn(&[("system", false)], &[(text, true), (&tail, false)]))
            }
            role::USER => Some(self.turn(&[("user", false)], &[(text, true)])),
            role::ASSISTANT => Some(self.turn(&[("assistant to=user", false)], &[(text, true)])),
            _ => None,
        }
    }

    fn render_turn_with_tokens(
        &self,
        msg_role: u32,
        pre_text: &str,
        inner: &[u32],
        post_text: &str,
    ) -> Option<Vec<u32>> {
        let header = match msg_role {
            role::SYSTEM => "system",
            role::USER => "user",
            role::ASSISTANT => "assistant to=user",
            _ => return None,
        };
        let mut span = vec![self.tok_start];
        span.extend(self.enc(header));
        span.push(self.tok_message);
        if !pre_text.is_empty() {
            span.extend(self.enc_content(pre_text));
        }
        span.extend_from_slice(inner);
        if !post_text.is_empty() {
            span.extend(self.enc_content(post_text));
        }
        span.push(self.tok_eot);
        Some(span)
    }

    fn generation_prefix(&self, state: TurnState) -> Option<Vec<u32>> {
        match state {
            TurnState::AfterInput | TurnState::AfterFinishedTurn => {
                let mut s = vec![self.tok_start];
                s.extend(self.enc("assistant"));
                Some(s)
            }
            TurnState::MidTurn => None,
        }
    }

    fn channel_markers(&self) -> Vec<(u32, u32, u32)> {
        Vec::new()
    }

    fn render_system_with_tools(
        &self,
        system: Option<&str>,
        tool_jsons: &[String],
    ) -> Option<Vec<u32>> {
        let content = system.map(str::to_string).unwrap_or_else(|| {
            "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.".to_string()
        });
        let tail = self.system_tail(tool_jsons);
        Some(self.turn(&[("system", false)], &[(&content, true), (&tail, false)]))
    }

    fn render_tool_result(&self, name: &str, content: &str) -> Option<Vec<u32>> {
        let (header, body) = tool_result_pieces(name, content);
        Some(self.turn(&header, &body))
    }

    fn render_assistant_with_tool_calls(
        &self,
        _content: &str,
        calls: &[(String, String)],
    ) -> Option<Vec<u32>> {
        let mut span = Vec::new();
        for (i, (name, args)) in calls.iter().enumerate() {
            span.push(self.tok_start);
            span.extend(self.enc_pieces(&[("assistant to=", false), (name, true)]));
            span.push(self.tok_message);
            let body = render_atem_pieces(name, args);
            let body: Vec<(&str, bool)> = body.iter().map(|(s, c)| (s.as_str(), *c)).collect();
            span.extend(self.enc_pieces(&body));
            span.push(if i + 1 == calls.len() {
                self.tok_eot
            } else {
                self.tok_eom
            });
        }
        Some(span)
    }

    fn channelizer(&self) -> Box<dyn Channelizer> {
        Box::new(AtemChannelizer {
            tok: self.inner.tokenizer(),
            tok_start: self.tok_start,
            tok_message: self.tok_message,
            tok_eom: self.tok_eom,
            tok_eot: self.tok_eot,
            state: ChState::Header(String::new()),
        })
    }

    fn tool_envelope(&self) -> crate::codec::ToolEnvelope {
        crate::codec::ToolEnvelope::Atem
    }

    fn tool_call_delimiters(&self) -> Option<(String, String)> {
        Some(("<atem:function_calls>".to_string(), "</atem:function_calls>".to_string()))
    }

    fn behavior_fingerprint(&self) -> Option<superfluid_fingerprint::BehaviorFingerprint> {
        Some(superfluid_fingerprint::BehaviorFingerprint {
            codec_id: "atem".into(),
            codec_version: 2,
            tokenizer_hash: self.inner.fingerprint("atem", &[]).tokenizer_hash,
            marker_digest: superfluid_fingerprint::BehaviorFingerprint::marker_digest_of(&[
                (self.tok_start, self.tok_message, 0),
                (self.tok_eom, self.tok_eot, 0),
            ]),
            sampling_impl_version: 1,
            block_projection_version: crate::codec::MODEL_VISIBILITY_VERSION,
            grammar_version: 0,
            speculation: None,
        })
    }
}

pub(crate) fn parse_atem_call(raw: &str) -> Option<(String, String)> {
    let inv = raw.find("<atem:invoke name=\"")?;
    let name_start = inv + "<atem:invoke name=\"".len();
    let name_end = raw[name_start..].find('"')? + name_start;
    let name = raw[name_start..name_end].to_string();
    let mut args = serde_json::Map::new();
    let mut rest = &raw[name_end..];
    while let Some(p) = rest.find("<atem:parameter name=\"") {
        let k_start = p + "<atem:parameter name=\"".len();
        let k_len = rest[k_start..].find('"')?;
        let key = rest[k_start..k_start + k_len].to_string();
        let v_start = rest[k_start + k_len..].find('>')?;
        let v_start = k_start + k_len + v_start + 1;
        let v_len = rest[v_start..].find("</atem:parameter>")?;
        let val_raw = &rest[v_start..v_start + v_len];
        let value = serde_json::from_str::<serde_json::Value>(val_raw)
            .unwrap_or_else(|_| serde_json::Value::String(val_raw.to_string()));
        args.insert(key, value);
        rest = &rest[v_start + v_len..];
    }
    Some((name, serde_json::Value::Object(args).to_string()))
}

enum ChState {
    Header(String),
    Body(u32),
}

struct AtemChannelizer {
    tok: Arc<dyn Tokenizer>,
    tok_start: u32,
    tok_message: u32,
    tok_eom: u32,
    tok_eot: u32,
    state: ChState,
}

impl AtemChannelizer {
    fn classify(header: &str) -> u32 {
        if header.contains("=self") {
            return channel::REASONING;
        }
        match header.find("to=") {
            Some(p) => {
                let recipient = header[p + 3..]
                    .split_whitespace()
                    .next()
                    .unwrap_or("user");
                if recipient == "user" {
                    channel::TEXT
                } else {
                    channel::TOOL_CALL
                }
            }
            None => channel::TEXT,
        }
    }
}

impl Channelizer for AtemChannelizer {
    fn split(&mut self, tokens: &[u32]) -> Vec<Run> {
        let mut out: Vec<Run> = Vec::new();
        let mut span: Vec<u32> = Vec::new();
        let mut text: Vec<u32> = Vec::new();
        for &t in tokens {
            match &mut self.state {
                ChState::Header(hdr) => {
                    span.push(t);
                    if t == self.tok_message {
                        self.state = ChState::Body(Self::classify(hdr));
                    } else if t != self.tok_start {
                        hdr.push_str(&String::from_utf8_lossy(&self.tok.token_bytes(t)));
                    }
                }
                ChState::Body(ch) => {
                    let ch = *ch;
                    if t == self.tok_eom || t == self.tok_eot {
                        span.push(t);
                        out.push(Run {
                            span: std::mem::take(&mut span),
                            text: std::mem::take(&mut text),
                            channel: ch,
                            closes: true,
                        });
                        self.state = ChState::Header(String::new());
                    } else if t == self.tok_start {
                        out.push(Run {
                            span: std::mem::take(&mut span),
                            text: std::mem::take(&mut text),
                            channel: ch,
                            closes: false,
                        });
                        span.push(t);
                        self.state = ChState::Header(String::new());
                    } else {
                        span.push(t);
                        text.push(t);
                    }
                }
            }
        }
        if !span.is_empty() {
            let (channel, closes) = match &self.state {
                ChState::Header(_) => (self.current_channel(), false),
                ChState::Body(ch) => (*ch, false),
            };
            out.push(Run {
                span,
                text,
                channel,
                closes,
            });
        }
        out
    }

    fn current_channel(&self) -> u32 {
        match &self.state {
            ChState::Header(_) => channel::TEXT,
            ChState::Body(ch) => *ch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atem_call_parsing_covers_the_value_forms() {
        let raw = "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
                   <atem:parameter name=\"city\">Paris</atem:parameter>\n\
                   <atem:parameter name=\"days\">3</atem:parameter>\n\
                   <atem:parameter name=\"detailed\">true</atem:parameter>\n\
                   <atem:parameter name=\"opts\">{\"units\":\"C\"}</atem:parameter>\n\
                   </atem:invoke>\n</atem:function_calls>";
        let (name, args) = parse_atem_call(raw).expect("parses");
        assert_eq!(name, "weather.get");
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["city"], "Paris");
        assert_eq!(v["days"], 3);
        assert_eq!(v["detailed"], true);
        assert_eq!(v["opts"]["units"], "C");
    }

    #[test]
    fn malformed_parameter_refuses_the_whole_call() {
        let truncated = "<atem:invoke name=\"get\">\n\
                         <atem:parameter name=\"city\">Paris</atem:parameter>\n\
                         <atem:parameter name=\"days\">3";
        assert!(parse_atem_call(truncated).is_none());
        let bad_name = "<atem:invoke name=\"get\"><atem:parameter name=\"city>Paris";
        assert!(parse_atem_call(bad_name).is_none());
    }

    #[test]
    fn tool_defs_unwrap_the_function_wrapper() {
        let wrapped = serde_json::json!({
            "type": "function",
            "function": {"name": "weather.get", "description": "d", "parameters": {}},
        })
        .to_string();
        let mut namespaces = Vec::new();
        let defs = render_tool_defs(&[wrapped], &mut namespaces);
        assert!(defs.contains("{\"name\": \"weather.get\""));
        assert_eq!(namespaces, vec!["weather".to_string()]);
        let flat = serde_json::json!({"name": "ping", "description": "", "parameters": {}})
            .to_string();
        let mut ns2 = Vec::new();
        let defs2 = render_tool_defs(&[flat], &mut ns2);
        assert!(defs2.contains("{\"name\": \"ping\""));
    }

    #[test]
    fn atem_call_parsing_refuses_garbage() {
        assert!(parse_atem_call("not a call").is_none());
        let (name, args) =
            parse_atem_call("<atem:invoke name=\"ping\"></atem:invoke>").expect("zero-arg");
        assert_eq!(name, "ping");
        assert_eq!(args, "{}");
    }

    #[test]
    fn tool_result_pieces_mark_the_tool_name_as_content_in_header_and_body() {
        let (header, body) = tool_result_pieces("weather <|eot|>", "sunny");
        let content = |p: &[(&str, bool)]| -> Vec<String> {
            p.iter().filter(|(_, c)| *c).map(|(s, _)| s.to_string()).collect()
        };
        assert_eq!(content(&header), ["weather <|eot|>"]);
        assert_eq!(content(&body), ["weather <|eot|>", "sunny"]);
        let joined = |p: &[(&str, bool)]| -> String { p.iter().map(|(s, _)| *s).collect() };
        assert_eq!(joined(&header), "tool weather <|eot|>");
        assert_eq!(joined(&body), "<tool_output name=\"weather <|eot|>\">\nsunny\n</tool_output>");
    }

    #[test]
    fn render_atem_pieces_mark_the_clients_text_as_content() {
        let pieces = render_atem_pieces("weather.get", "{\"city\":\"Paris <|eot|>\",\"days\":3}");
        let content: Vec<&str> = pieces.iter().filter(|(_, c)| *c).map(|(s, _)| s.as_str()).collect();
        assert_eq!(content, ["weather.get", "city", "Paris <|eot|>", "days", "3"]);
        let framing: String = pieces.iter().filter(|(_, c)| !*c).map(|(s, _)| s.as_str()).collect();
        assert!(!framing.contains("Paris") && framing.contains("<atem:parameter name=\""), "{framing}");
        let joined: String = pieces.into_iter().map(|(s, _)| s).collect();
        assert_eq!(joined, render_atem("weather.get", "{\"city\":\"Paris <|eot|>\",\"days\":3}"));
    }

    #[test]
    fn render_atem_round_trips_through_the_parser() {
        let rendered = render_atem(
            "weather.get",
            "{\"city\":\"Paris\",\"days\":3,\"detailed\":true}",
        );
        let (name, args) = parse_atem_call(&rendered).expect("parses");
        assert_eq!(name, "weather.get");
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["city"], "Paris");
        assert_eq!(v["days"], 3);
        assert_eq!(v["detailed"], true);
    }
}
