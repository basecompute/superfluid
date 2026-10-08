//! The daemon-side text codec seam.

use superfluid_abi::Status;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    AfterInput,
    MidTurn,
    AfterFinishedTurn,
}

#[derive(Debug, Clone, Default)]
pub struct ToolCallMsg {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    pub role: u32,
    pub content: String,
    pub parts: Vec<ContentPart>,
    pub tool_calls: Vec<ToolCallMsg>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentPart {
    Text(String),
    Image { blob: String },
    Audio { blob: String },
}

impl ChatMessage {
    pub fn new(role: u32, content: impl Into<String>) -> ChatMessage {
        ChatMessage { role, content: content.into(), ..Default::default() }
    }
}

pub(crate) fn thinking_probe_contexts() -> Vec<(Vec<ChatMessage>, Vec<String>)> {
    let user = ChatMessage::new(crate::wal::role::USER, "probe");
    let system = ChatMessage::new(crate::wal::role::SYSTEM, "probe");
    let tool = r#"{"type":"function","function":{"name":"probe","description":"probe","parameters":{"type":"object","properties":{}}}}"#;
    vec![
        (vec![user.clone()], Vec::new()),
        (vec![system.clone(), user.clone()], Vec::new()),
        (vec![system, user], vec![tool.to_string()]),
    ]
}

#[cfg_attr(not(feature = "basert"), allow(dead_code))]
pub(crate) fn thinking_switch_changes<T: PartialEq>(
    render: impl Fn(&[ChatMessage], &[String], bool) -> Option<T>,
) -> bool {
    thinking_probe_contexts().iter().any(|(msgs, tools)| {
        match (render(msgs, tools, true), render(msgs, tools, false)) {
            (Some(on), Some(off)) => on != off,
            (Some(_), None) | (None, Some(_)) => true,
            (None, None) => false,
        }
    })
}

pub trait TextCodec: Send {
    fn encode(&self, text: &str) -> Vec<u32>;

    fn encode_content(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    fn encode_pieces(&self, pieces: &[(&str, bool)]) -> Vec<u32> {
        let mut out = Vec::new();
        for (text, content) in pieces {
            out.extend(if *content { self.encode_content(text) } else { self.encode(text) });
        }
        out
    }
    fn token_bytes(&self, token: u32) -> Vec<u8>;

    fn decode(&self, tokens: &[u32]) -> String {
        let mut s = Utf8Stream::default();
        let mut out = String::new();
        for &t in tokens {
            out.push_str(&s.push(&self.token_bytes(t)));
        }
        out.push_str(&s.flush());
        out
    }

    fn stream_prologue(&self) -> Vec<u32> {
        Vec::new()
    }

    fn behavior_fingerprint(&self) -> Option<superfluid_fingerprint::BehaviorFingerprint> {
        None
    }

    fn chat_template_source(&self) -> Option<String> {
        None
    }

    fn render_message(&self, _role: u32, _text: &str) -> Option<Vec<u32>> {
        None
    }

    fn generation_prefix(&self, _state: TurnState) -> Option<Vec<u32>> {
        None
    }

    fn trailing_generation_prompt(
        &self,
        _messages: &[ChatMessage],
        _tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
        span: &[u32],
    ) -> usize {
        let thinking = kwargs.get("enable_thinking").and_then(|v| v.as_bool());
        [
            self.generation_prefix_with(TurnState::AfterInput, thinking),
            self.generation_prefix(TurnState::AfterInput),
        ]
        .into_iter()
        .flatten()
        .find(|op| !op.is_empty() && span.len() > op.len() && span.ends_with(op))
        .map_or(0, |op| op.len())
    }

    fn empty_think_tokens(&self) -> Option<Vec<u32>> {
        None
    }

    fn generation_prefix_with(&self, state: TurnState, thinking: Option<bool>) -> Option<Vec<u32>> {
        let mut op = self.generation_prefix(state)?;
        if thinking == Some(false) {
            if let Some(block) = self.empty_think_tokens() {
                op.extend(block);
            }
        }
        Some(op)
    }

    fn render_system_full(
        &self,
        system: Option<&str>,
        tool_jsons: &[String],
        _kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        self.render_system_with_tools(system, tool_jsons)
    }

    fn supports_reasoning_effort(&self) -> bool {
        false
    }

    fn supports_enable_thinking(&self) -> bool {
        false
    }

    fn reasoning_effort_levels(&self) -> Vec<&'static str> {
        Vec::new()
    }

    fn validate_template_kwargs(
        &self,
        _kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        Ok(())
    }

    fn render_prompt_structured_with(
        &self,
        messages: &[ChatMessage],
        tools: &[String],
        _kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        self.render_prompt_structured(messages, tools)
    }

    fn template_refusal(
        &self,
        _messages: &[ChatMessage],
        _tools: &[String],
        _kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<String> {
        None
    }

    fn turn_separator(&self) -> Option<Vec<u32>> {
        None
    }

    fn channel_markers(&self) -> Vec<(u32, u32, u32)> {
        Vec::new()
    }

    fn render_system_with_tools(
        &self,
        _system: Option<&str>,
        _tool_jsons: &[String],
    ) -> Option<Vec<u32>> {
        None
    }

    fn render_tool_result(&self, _name: &str, _content: &str) -> Option<Vec<u32>> {
        None
    }

    fn render_assistant_with_tool_calls(
        &self,
        _content: &str,
        _calls: &[(String, String)],
    ) -> Option<Vec<u32>> {
        None
    }

    /// The same turn where no user query follows it in the conversation (a
    /// step of a tool loop): some templates render that turn differently from
    /// one a later query put in history (Qwen's open its reasoning block).
    fn render_assistant_with_tool_calls_after_query(
        &self,
        content: &str,
        calls: &[(String, String)],
    ) -> Option<Vec<u32>> {
        self.render_assistant_with_tool_calls(content, calls)
    }

    fn render_fim(&self, _prefix: &str, _suffix: &str, _mode: u8) -> Option<Vec<u32>> {
        None
    }

    fn render_turn_with_tokens(
        &self,
        _role: u32,
        _pre_text: &str,
        _inner: &[u32],
        _post_text: &str,
    ) -> Option<Vec<u32>> {
        None
    }

    fn render_block(&self, role: u32, kind: u32, payload: &str) -> Option<Vec<u32>> {
        let text = project_block_text(kind, payload)?;
        if text.is_empty() {
            return Some(Vec::new());
        }
        self.render_message(role, &text)
    }

    fn channelizer(&self) -> Box<dyn Channelizer> {
        Box::new(Segmenter::new(self.channel_markers()))
    }

    fn turn_terminators(&self) -> Vec<u32> {
        Vec::new()
    }

    fn render_prompt_structured(
        &self,
        messages: &[ChatMessage],
        tools: &[String],
    ) -> Option<Vec<u32>> {
        let flat: Vec<(u32, String)> =
            messages.iter().map(|m| (m.role, m.content.clone())).collect();
        self.render_prompt(&flat, tools)
    }

    fn render_prompt(&self, messages: &[(u32, String)], tools: &[String]) -> Option<Vec<u32>> {
        let mut out = self.stream_prologue();
        let mut tools_emitted = tools.is_empty();
        for (role, text) in messages {
            if !tools_emitted && *role == crate::wal::role::SYSTEM {
                out.extend(self.render_system_with_tools(Some(text), tools)?);
                tools_emitted = true;
            } else {
                out.extend(self.render_message(*role, text)?);
            }
        }
        if !tools_emitted {
            let mut with = self.render_system_with_tools(None, tools)?;
            with.extend(std::mem::take(&mut out));
            out = with;
        }
        if let Some(op) = self.generation_prefix(TurnState::AfterInput) {
            out.extend(op);
        }
        Some(out)
    }

    fn call_grammar(&self, tool_jsons: &[String]) -> Option<String> {
        self.tool_envelope().call_grammar(tool_jsons)
    }

    fn renders_per_message(&self) -> bool {
        true
    }

    fn tool_envelope(&self) -> ToolEnvelope {
        ToolEnvelope::Json
    }

    fn tool_fragment_mode(&self) -> crate::tool_fragment::ToolFragmentMode {
        match self.tool_envelope() {
            ToolEnvelope::Json => crate::tool_fragment::ToolFragmentMode::Json,
            _ => crate::tool_fragment::ToolFragmentMode::None,
        }
    }

    fn tool_call_delimiters(&self) -> Option<(String, String)> {
        let (open, close, _) = self
            .channel_markers()
            .into_iter()
            .find(|&(_, _, ch)| ch == crate::wal::channel::TOOL_CALL)?;
        Some((self.marker_literal(open), self.marker_literal(close)))
    }

    fn marker_literal(&self, token: u32) -> String {
        self.decode(&[token])
    }

    fn structural_tag(&self, tool_jsons: &[String], at_least_one: bool) -> Option<String> {
        let (begin, end) = self.tool_call_delimiters()?;
        self.tool_envelope().structural_tag(tool_jsons, &begin, &end, at_least_one)
    }

    fn response_format_tag(&self, json_schema: &str) -> Option<String> {
        let _ = json_schema;
        None
    }

    fn parse_tool_call(&self, raw: &str) -> Option<(String, String)> {
        self.parse_tool_call_with(raw, None)
    }

    fn parse_tool_call_with(&self, raw: &str, schemas: Option<&ToolSchemas>) -> Option<(String, String)> {
        self.tool_envelope().parse_call(raw, schemas)
    }

    fn named_tool_call(&self, raw: &str) -> Option<(String, String)> {
        self.tool_envelope().named_call(raw)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub span: Vec<u32>,
    pub text: Vec<u32>,
    pub channel: u32,
    pub closes: bool,
}

pub(crate) fn repair_json_object(raw: &str) -> Option<String> {
    let unfenced = strip_code_fence(raw).trim();
    let mut attempts: Vec<String> = vec![unfenced.to_string()];
    if let Some(obj) = first_json_object(unfenced) {
        attempts.push(obj.to_string());
    }
    for i in 0..attempts.len() {
        let a = attempts[i].clone();
        let fixed = drop_trailing_commas(&a);
        if fixed != a {
            attempts.push(fixed.clone());
        }
        if let Some(closed) = close_unterminated(&a) {
            attempts.push(closed.clone());
            let closed_fixed = drop_trailing_commas(&closed);
            if closed_fixed != closed {
                attempts.push(closed_fixed);
            }
        }
    }
    attempts.iter().find_map(|a| match serde_json::from_str::<serde_json::Value>(a) {
        Ok(v) if v.is_object() => Some(v.to_string()),
        _ => None,
    })
}

fn strip_code_fence(raw: &str) -> &str {
    let t = raw.trim();
    let Some(rest) = t.strip_prefix("```") else { return t };
    let rest = match rest.find('\n') {
        Some(i) => &rest[i + 1..],
        None => rest,
    };
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

fn first_json_object(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for (i, c) in raw.char_indices().skip_while(|(i, _)| *i < start) {
        if esc {
            esc = false;
            continue;
        }
        match c {
            '\\' if in_str => esc = true,
            '"' => in_str = !in_str,
            '{' if !in_str => depth += 1,
            '}' if !in_str => {
                depth -= 1;
                if depth == 0 {
                    return raw.get(start..i + c.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

fn close_unterminated(raw: &str) -> Option<String> {
    let mut stack: Vec<char> = Vec::new();
    let (mut in_str, mut esc) = (false, false);
    for c in raw.chars() {
        if esc {
            esc = false;
            continue;
        }
        match c {
            '\\' if in_str => esc = true,
            '"' => in_str = !in_str,
            '{' | '[' if !in_str => stack.push(c),
            '}' | ']' if !in_str => {
                stack.pop()?;
            }
            _ => {}
        }
    }
    if stack.is_empty() && !in_str {
        return None;
    }
    let mut out = raw.to_string();
    if esc {
        out.pop();
    }
    if in_str {
        out.push('"');
    }
    while let Some(open) = stack.pop() {
        out.push(if open == '{' { '}' } else { ']' });
    }
    Some(out)
}

fn drop_trailing_commas(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let (mut in_str, mut esc) = (false, false);
    let chars: Vec<char> = raw.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if esc {
            esc = false;
            out.push(c);
            continue;
        }
        match c {
            '\\' if in_str => {
                esc = true;
                out.push(c);
            }
            '"' => {
                in_str = !in_str;
                out.push(c);
            }
            ',' if !in_str => {
                let next = chars[i + 1..].iter().find(|c| !c.is_whitespace());
                if matches!(next, Some('}') | Some(']')) {
                    continue;
                }
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

fn call_fields(v: &serde_json::Value) -> Option<(String, String)> {
    let v = v.get("function").filter(|f| f.is_object()).unwrap_or(v);
    let name = v.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let args = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .or_else(|| v.get("args"))
        .or_else(|| v.get("input"));
    let args = match args {
        None => "{}".to_string(),
        Some(serde_json::Value::String(s)) => {
            let inner: serde_json::Value = serde_json::from_str(s).ok()?;
            inner.to_string()
        }
        Some(other) => other.to_string(),
    };
    Some((name, args))
}

pub fn parse_json_tool_call(raw: &str) -> Option<(String, String)> {
    let unfenced = strip_code_fence(raw);
    let mut attempts: Vec<String> = vec![unfenced.to_string()];
    if let Some(obj) = first_json_object(unfenced) {
        attempts.push(obj.to_string());
    }
    for i in 0..attempts.len() {
        let a = attempts[i].clone();
        let fixed = drop_trailing_commas(&a);
        if fixed != a {
            attempts.push(fixed.clone());
        }
        if let Some(closed) = close_unterminated(&a) {
            attempts.push(closed.clone());
            let closed_fixed = drop_trailing_commas(&closed);
            if closed_fixed != closed {
                attempts.push(closed_fixed);
            }
        }
    }
    for a in attempts {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(a.trim()) {
            if let Some(call) = call_fields(&v) {
                return Some(call);
            }
        }
    }
    None
}
const GEMMA_QUOTE: [char; 5] = ['<', '|', '"', '|', '>'];

fn gemma_at(cs: &[char], p: usize, pat: &[char]) -> bool {
    p + pat.len() <= cs.len() && cs[p..p + pat.len()] == *pat
}

fn gemma_skip_ws(cs: &[char], p: &mut usize) {
    while *p < cs.len() && matches!(cs[*p], ' ' | '\n' | '\t') {
        *p += 1;
    }
}

fn gemma_parse_key(cs: &[char], p: &mut usize) -> String {
    let start = *p;
    while *p < cs.len() && !matches!(cs[*p], ':' | ',' | '}' | ']') {
        *p += 1;
    }
    cs[start..*p].iter().collect()
}

fn gemma_parse_string(cs: &[char], p: &mut usize) -> String {
    *p += GEMMA_QUOTE.len();
    let mut i = *p;
    while i < cs.len() && !gemma_at(cs, i, &GEMMA_QUOTE) {
        i += 1;
    }
    if i >= cs.len() {
        let v: String = cs[*p..].iter().collect();
        *p = cs.len();
        return v;
    }
    let v: String = cs[*p..i].iter().collect();
    *p = i + GEMMA_QUOTE.len();
    v
}

const GEMMA_MAX_DEPTH: usize = 128;

fn gemma_parse_value(cs: &[char], p: &mut usize, in_array: bool) -> serde_json::Value {
    gemma_parse_value_at(cs, p, in_array, 0)
}

fn gemma_parse_value_at(cs: &[char], p: &mut usize, in_array: bool, depth: usize) -> serde_json::Value {
    use serde_json::Value;
    gemma_skip_ws(cs, p);
    if *p >= cs.len() {
        return Value::Null;
    }
    if depth > GEMMA_MAX_DEPTH {
        *p = cs.len();
        return Value::Null;
    }
    if gemma_at(cs, *p, &GEMMA_QUOTE) {
        return Value::String(gemma_parse_string(cs, p));
    }
    if cs[*p] == '{' {
        *p += 1;
        let mut obj = serde_json::Map::new();
        while *p < cs.len() && cs[*p] != '}' {
            while *p < cs.len() && matches!(cs[*p], ' ' | ',' | '\n') {
                *p += 1;
            }
            if *p >= cs.len() || cs[*p] == '}' {
                break;
            }
            let mut k = gemma_parse_key(cs, p);
            while k.ends_with(' ') || k.ends_with('\n') {
                k.pop();
            }
            if *p >= cs.len() || cs[*p] != ':' {
                break;
            }
            *p += 1;
            let v = gemma_parse_value_at(cs, p, false, depth + 1);
            obj.insert(k, v);
        }
        if *p < cs.len() && cs[*p] == '}' {
            *p += 1;
        }
        return Value::Object(obj);
    }
    if cs[*p] == '[' {
        *p += 1;
        let mut arr = Vec::new();
        while *p < cs.len() && cs[*p] != ']' {
            while *p < cs.len() && matches!(cs[*p], ' ' | ',' | '\n') {
                *p += 1;
            }
            if *p >= cs.len() || cs[*p] == ']' {
                break;
            }
            arr.push(gemma_parse_value_at(cs, p, true, depth + 1));
        }
        if *p < cs.len() && cs[*p] == ']' {
            *p += 1;
        }
        return Value::Array(arr);
    }
    let start = *p;
    let (mut brace, mut bracket) = (0i32, 0i32);
    while *p < cs.len() {
        match cs[*p] {
            '{' => brace += 1,
            '}' => {
                if brace == 0 {
                    break;
                }
                brace -= 1;
            }
            '[' => bracket += 1,
            ']' => {
                if bracket == 0 {
                    break;
                }
                bracket -= 1;
            }
            ',' if brace == 0 && bracket == 0 => {
                if in_array {
                    break;
                }
                let mut q = *p + 1;
                while q < cs.len() && matches!(cs[q], ' ' | '\n' | '\t') {
                    q += 1;
                }
                let ks = q;
                while q < cs.len() && (cs[q].is_ascii_alphanumeric() || cs[q] == '_') {
                    q += 1;
                }
                if q > ks && q < cs.len() && cs[q] == ':' {
                    break;
                }
            }
            _ => {}
        }
        *p += 1;
    }
    let lit: String = cs[start..*p].iter().collect();
    let t = lit.trim();
    match t {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" | "" => Value::Null,
        _ => {
            if let Ok(i) = t.parse::<i64>() {
                return Value::Number(i.into());
            }
            if let Ok(f) = t.parse::<f64>() {
                if let Some(n) = serde_json::Number::from_f64(f) {
                    return Value::Number(n);
                }
            }
            Value::String(t.to_string())
        }
    }
}

pub fn trailing_open_channel(tokens: &[u32], markers: &[(u32, u32, u32)]) -> u32 {
    let mut cur = crate::wal::channel::TEXT;
    for &t in tokens {
        if cur == crate::wal::channel::TEXT {
            if let Some((_, _, ch)) = markers.iter().find(|(o, _, _)| *o == t) {
                cur = *ch;
            }
        } else if markers.iter().any(|(_, c, ch)| *c == t && *ch == cur) {
            cur = crate::wal::channel::TEXT;
        }
    }
    cur
}

pub fn primed_channelizer(codec: &dyn TextCodec, prompt: &[u32]) -> Box<dyn Channelizer> {
    let mut chan = codec.channelizer();
    let open = trailing_open_channel(prompt, &codec.channel_markers());
    if open != crate::wal::channel::TEXT {
        chan.prime(open);
    }
    chan
}

#[cfg(test)]
mod trailing_channel_tests {
    use super::trailing_open_channel;
    use crate::wal::channel;

    #[test]
    fn reports_the_channel_a_prompt_leaves_open() {
        let m = [(10, 11, channel::REASONING), (20, 21, channel::TOOL_CALL)];
        assert_eq!(trailing_open_channel(&[1, 2, 3], &m), channel::TEXT);
        assert_eq!(trailing_open_channel(&[1, 2, 10, 5], &m), channel::REASONING);
        assert_eq!(trailing_open_channel(&[1, 10, 5, 11, 5], &m), channel::TEXT);
        assert_eq!(trailing_open_channel(&[10, 7, 11, 8, 20, 9, 21, 10], &m), channel::REASONING);
        assert_eq!(trailing_open_channel(&[10, 21], &m), channel::REASONING);
    }

    #[test]
    fn primed_channelizer_starts_where_the_prompt_left_off() {
        use super::{primed_channelizer, MockChatCodec, MOCK_THINK_CLOSE, MOCK_THINK_OPEN};
        let codec = MockChatCodec;
        let generated = [7u32, 8, MOCK_THINK_CLOSE, 9];
        let open = primed_channelizer(&codec, &[1, 2, MOCK_THINK_OPEN]).split(&generated);
        assert_eq!(open[0].channel, channel::REASONING, "{open:?}");
        assert_eq!(open.last().unwrap().channel, channel::TEXT, "{open:?}");
        let closed =
            primed_channelizer(&codec, &[1, MOCK_THINK_OPEN, MOCK_THINK_CLOSE]).split(&generated[..2]);
        assert!(closed.iter().all(|r| r.channel == channel::TEXT), "{closed:?}");
    }
}

#[derive(Clone)]
pub struct Delims {
    name_open: String,
    name_close: String,
    key_open: String,
    key_close: String,
    val_close: String,
}

pub(crate) const S_NAME: &str = "N1nfx";
pub(crate) const S_K1: &str = "K1kax";
pub(crate) const S_V1: &str = "V1vax";
pub(crate) const S_K2: &str = "J2kbx";
pub(crate) const S_V2: &str = "U2vbx";

pub(crate) fn common_prefix(a: &str, b: &str) -> String {
    let mut end = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        end += ca.len_utf8();
    }
    a[..end].to_string()
}

pub(crate) fn common_suffix(a: &str, b: &str) -> String {
    let mut n = 0;
    for (ca, cb) in a.chars().rev().zip(b.chars().rev()) {
        if ca != cb {
            break;
        }
        n += ca.len_utf8();
    }
    a[a.len() - n..].to_string()
}

pub(crate) fn learn_delims(two: &str, zero: Option<&str>, frame: Option<(&str, &str)>) -> Option<Delims> {
    let at = |s: &str| two.find(s);
    let inm = at(S_NAME)?;
    let mut pairs = [
        (at(S_K1)?, S_K1.len(), at(S_V1)?, S_V1.len()),
        (at(S_K2)?, S_K2.len(), at(S_V2)?, S_V2.len()),
    ];
    pairs.sort_by_key(|p| p.0);
    let (ika, kalen, iva, valen) = pairs[0];
    let (ikb, kblen, ivb, vblen) = pairs[1];
    if !(ika < iva && iva < ikb && ikb < ivb) {
        return None;
    }

    let anchor_end = frame
        .and_then(|(o, _)| two[..inm].rfind(o).map(|i| i + o.len()))
        .unwrap_or_else(|| two[..inm].rfind('\n').map(|i| i + 1).unwrap_or(0));
    let name_open = two[anchor_end..inm].trim().to_string();

    let after_name_two = &two[inm + S_NAME.len()..];
    let name_close = match zero.and_then(|z| z.find(S_NAME).map(|i| &z[i + S_NAME.len()..])) {
        Some(after_name_zero) => common_prefix(after_name_two, after_name_zero),
        None => {
            after_name_two[..after_name_two.find(&two[ika..ika + kalen]).unwrap_or(after_name_two.len())].to_string()
        }
    };

    let mut key_open = common_suffix(&two[..ika], &two[..ikb]);
    let key_close = common_prefix(&two[ika + kalen..], &two[ikb + kblen..]);
    let val_close = common_prefix(&two[iva + valen..], &two[ivb + vblen..]);

    while !name_close.is_empty() && key_open.starts_with(&name_close) {
        key_open = key_open[name_close.len()..].to_string();
    }

    if name_open.is_empty() || name_close.is_empty() || key_open.is_empty() || key_close.is_empty() {
        return None;
    }
    Some(Delims {
        name_open,
        name_close,
        key_open,
        key_close,
        val_close,
    })
}

pub(crate) fn typed_value(s: &str) -> serde_json::Value {
    let t = s.trim();
    match serde_json::from_str::<serde_json::Value>(t) {
        Ok(v @ (serde_json::Value::Object(_) | serde_json::Value::Array(_))) => v,
        Ok(v @ (serde_json::Value::Number(_) | serde_json::Value::Bool(_) | serde_json::Value::Null))
            if !t.is_empty() =>
        {
            v
        }
        _ => serde_json::Value::String(s.to_string()),
    }
}

impl Delims {
    pub(crate) fn structural_tag(
        &self,
        tool_jsons: &[String],
        begin: &str,
        end: &str,
        at_least_one: bool,
    ) -> Option<String> {
        if tool_jsons.is_empty() || begin.is_empty() || end.is_empty() || !at_least_one {
            return None;
        }
        let mut tags = Vec::new();
        for t in tool_jsons {
            let v: serde_json::Value = serde_json::from_str(t).ok()?;
            let f = v.get("function").unwrap_or(&v);
            let name = f.get("name").and_then(|n| n.as_str())?;
            let params = f
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
            tags.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": self.call_body_format(name, &params),
                "end": end,
            }));
        }
        Some(wrap_structural_tag(serde_json::json!({
            "type": "triggered_tags",
            "triggers": [begin],
            "tags": tags,
            "at_least_one": at_least_one,
            "stop_after_first": stop_after_first(at_least_one),
        })))
    }

    fn call_body_format(&self, name: &str, params: &serde_json::Value) -> serde_json::Value {
        let props = params.get("properties").and_then(|p| p.as_object());
        let required: Vec<String> = params
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let param = |key: &str| -> serde_json::Value {
            let schema = props.and_then(|p| p.get(key));
            let open = format!("{}{key}{}", self.key_open, self.key_close);
            if self.val_close.is_empty() {
                let schema = schema.cloned().unwrap_or_else(|| serde_json::json!({ "type": "string" }));
                return serde_json::json!({
                    "type": "sequence",
                    "elements": [
                        { "type": "const_string", "value": open },
                        value_format(&schema, ATEM_STRING),
                    ]
                });
            }
            serde_json::json!({
                "type": "tag",
                "begin": open,
                "content": glm_value_format(schema, params),
                "end": self.val_close,
            })
        };
        let all: Vec<String> = props.map(|p| p.keys().cloned().collect()).unwrap_or_default();
        let mut elements = vec![
            serde_json::json!({ "type": "regex", "pattern": "\\s*" }),
            serde_json::json!({
                "type": "const_string",
                "value": format!("{}{name}{}", self.name_open, self.name_close),
            }),
        ];
        for key in &all {
            let e = param(key);
            if required.contains(key) {
                elements.push(e);
            } else {
                elements.push(serde_json::json!({ "type": "optional", "content": e }));
            }
        }
        elements.push(serde_json::json!({ "type": "any_text" }));
        serde_json::json!({ "type": "sequence", "elements": elements })
    }

    fn value(&self, name: &str, key: &str, raw: &str, schemas: Option<&ToolSchemas>) -> serde_json::Value {
        let prop = schemas.and_then(|s| s.property(name, key));
        let root = schemas.and_then(|s| s.params(name)).unwrap_or(&serde_json::Value::Null);
        declared_value(raw, prop, root, StringUnion::Verbatim)
    }

    pub(crate) fn parse(&self, raw: &str, schemas: Option<&ToolSchemas>) -> Option<(String, String)> {
        let i = raw.find(&self.name_open)?;
        let j = i + self.name_open.len();
        let ne = raw[j..].find(&self.name_close).unwrap_or(raw.len() - j);
        let name = raw[j..j + ne].trim();
        if name.is_empty() || name.contains('<') || name.contains(char::is_whitespace) {
            return None;
        }
        let mut args = serde_json::Map::new();
        let mut cursor = j + ne;
        while let Some(k) = raw[cursor..].find(&self.key_open) {
            let ks = cursor + k + self.key_open.len();
            let ke = match raw[ks..].find(&self.key_close) {
                Some(e) => e,
                None => break,
            };
            let key = raw[ks..ks + ke].trim().to_string();
            let vs = ks + ke + self.key_close.len();
            let ve = if self.val_close.is_empty() {
                raw.len() - vs
            } else {
                raw[vs..].find(&self.val_close).unwrap_or(raw.len() - vs)
            };
            let val = &raw[vs..vs + ve];
            if !key.is_empty() {
                let v = self.value(name, &key, val, schemas);
                args.insert(key, v);
            }
            cursor = vs + ve;
        }
        Some((name.to_string(), serde_json::Value::Object(args).to_string()))
    }

    #[cfg(test)]
    pub(crate) fn for_test(name_open: &str, name_close: &str, key_open: &str, key_close: &str, val_close: &str) -> Self {
        Self {
            name_open: name_open.into(),
            name_close: name_close.into(),
            key_open: key_open.into(),
            key_close: key_close.into(),
            val_close: val_close.into(),
        }
    }

    pub(crate) fn peek_name(&self, raw: &str) -> Option<String> {
        let i = raw.find(&self.name_open)?;
        let j = i + self.name_open.len();
        let ne = raw[j..].find(&self.name_close)?;
        let name = raw[j..j + ne].trim();
        if name.is_empty() || name.contains('<') || name.contains(char::is_whitespace) {
            return None;
        }
        Some(name.to_string())
    }

    pub(crate) fn closed_args(
        &self,
        raw: &str,
        cursor: &mut Option<usize>,
        schemas: Option<&ToolSchemas>,
    ) -> Vec<(String, serde_json::Value)> {
        let mut out = Vec::new();
        let name = schemas.and_then(|_| self.peek_name(raw)).unwrap_or_default();
        if cursor.is_none() {
            let Some(i) = raw.find(&self.name_open) else {
                return out;
            };
            let j = i + self.name_open.len();
            let Some(ne) = raw[j..].find(&self.name_close) else {
                return out;
            };
            *cursor = Some(j + ne);
        }
        let cursor = cursor.as_mut().expect("set above");
        while let Some(k) = raw[*cursor..].find(&self.key_open) {
            let ks = *cursor + k + self.key_open.len();
            let Some(ke) = raw[ks..].find(&self.key_close) else {
                break;
            };
            let key = raw[ks..ks + ke].trim().to_string();
            let vs = ks + ke + self.key_close.len();
            if self.val_close.is_empty() {
                break;
            }
            let Some(ve) = raw[vs..].find(&self.val_close) else {
                break;
            };
            if !key.is_empty() {
                let v = self.value(&name, &key, &raw[vs..vs + ve], schemas);
                out.push((key, v));
            }
            *cursor = vs + ve;
        }
        out
    }
}

pub fn parse_gemma_tool_call(raw: &str) -> Option<(String, String)> {
    let mut body = raw.trim();
    if let Some(i) = body.find(GEMMA_TOOL_OPEN) {
        body = &body[i + GEMMA_TOOL_OPEN.len()..];
    }
    if let Some(i) = body.find(GEMMA_TOOL_CLOSE) {
        body = &body[..i];
    }
    let rest = body.trim().strip_prefix("call:")?;
    let brace = rest.find('{')?;
    let name = rest[..brace].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let cs: Vec<char> = rest.chars().collect();
    let mut p = rest[..brace].chars().count();
    let v = gemma_parse_value(&cs, &mut p, false);
    if v.is_null() {
        return Some((name, "{}".to_string()));
    }
    Some((name, serde_json::to_string(&v).ok()?))
}

// Re-serialised as the HTTP routes do, so a tool declared natively renders the tokens the same
// tool sent to /v1/chat/completions does. A bare function or an Anthropic tool is wrapped the
// way the Anthropic route wraps it.
pub fn function_tool(json: &str) -> Result<String, String> {
    use serde_json::{json, Value};
    let Value::Object(mut tool) =
        serde_json::from_str::<Value>(json).map_err(|e| format!("not JSON: {e}"))?
    else {
        return Err("not a JSON object".into());
    };
    let named = |f: &serde_json::Map<String, Value>| -> Result<(), String> {
        match f.get("name").and_then(Value::as_str) {
            Some(n) if !n.is_empty() => {}
            _ => return Err("it has no name".into()),
        }
        match f.get("parameters") {
            None | Some(Value::Object(_)) => Ok(()),
            Some(_) => Err("its parameters are not a JSON schema object".into()),
        }
    };
    if tool.get("type").is_some_and(|t| t != "function") {
        return Err("its type is not \"function\"".into());
    }
    if let Some(f) = tool.get("function") {
        let Value::Object(f) = f else {
            return Err("its function is not an object".into());
        };
        named(f)?;
        return Ok(Value::Object(tool).to_string());
    }
    tool.remove("type");
    let function = match tool.remove("input_schema") {
        Some(schema) => json!({
            "name": tool.get("name").cloned().unwrap_or(Value::Null),
            "description": tool.get("description").and_then(Value::as_str).unwrap_or_default(),
            "parameters": schema,
        }),
        None => Value::Object(tool),
    };
    named(function.as_object().expect("built as an object"))?;
    Ok(json!({"type": "function", "function": function}).to_string())
}

#[derive(Debug, Clone, Default)]
pub struct ToolSchemas(std::collections::HashMap<String, ToolSchema>);

#[derive(Debug, Clone)]
struct ToolSchema {
    root: serde_json::Value,
    args: serde_json::Value,
}

impl ToolSchemas {
    pub fn from_values<'a>(tools: impl IntoIterator<Item = &'a serde_json::Value>) -> ToolSchemas {
        let mut m = std::collections::HashMap::new();
        for t in tools {
            let f = t.get("function").unwrap_or(t);
            let Some(name) = f.get("name").and_then(|n| n.as_str()) else { continue };
            let params = f
                .get("parameters")
                .or_else(|| f.get("input_schema"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let args = resolve_local_ref(&params, &params).into_owned();
            m.insert(name.to_string(), ToolSchema { root: params, args });
        }
        ToolSchemas(m)
    }

    pub fn from_jsons(tools: &[String]) -> ToolSchemas {
        let vals: Vec<serde_json::Value> = tools.iter().filter_map(|t| serde_json::from_str(t).ok()).collect();
        ToolSchemas::from_values(vals.iter())
    }

    pub fn of_values(tools: &[serde_json::Value]) -> Option<std::sync::Arc<ToolSchemas>> {
        (!tools.is_empty()).then(|| std::sync::Arc::new(ToolSchemas::from_values(tools.iter())))
    }

    pub fn params(&self, tool: &str) -> Option<&serde_json::Value> {
        self.0.get(tool).map(|t| &t.root)
    }

    pub fn args_schema(&self, tool: &str) -> Option<&serde_json::Value> {
        self.0.get(tool).map(|t| &t.args)
    }

    pub fn property(&self, tool: &str, key: &str) -> Option<&serde_json::Value> {
        self.args_schema(tool)?.get("properties")?.get(key)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

fn resolve_local_ref<'a>(p: &'a serde_json::Value, root: &'a serde_json::Value) -> std::borrow::Cow<'a, serde_json::Value> {
    use std::borrow::Cow;
    let mut cur: Cow<'a, serde_json::Value> = Cow::Borrowed(p);
    for _ in 0..16 {
        let Some(r) = cur.get("$ref").and_then(|r| r.as_str()) else { break };
        let Some(target) = r.strip_prefix('#').and_then(|ptr| root.pointer(ptr)) else { break };
        let siblings: Vec<(String, serde_json::Value)> = cur
            .as_object()
            .map(|o| o.iter().filter(|(k, _)| *k != "$ref").map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        if siblings.is_empty() {
            cur = Cow::Borrowed(target);
            continue;
        }
        let mut merged = target.clone();
        if let Some(obj) = merged.as_object_mut() {
            for (k, v) in siblings {
                match (k.as_str(), obj.get_mut(&k), v) {
                    ("required", Some(serde_json::Value::Array(have)), serde_json::Value::Array(add)) => {
                        for x in add {
                            if !have.contains(&x) {
                                have.push(x);
                            }
                        }
                    }
                    ("properties", Some(serde_json::Value::Object(have)), serde_json::Value::Object(add)) => {
                        have.extend(add);
                    }
                    (_, _, v) => {
                        obj.insert(k, v);
                    }
                }
            }
        }
        cur = Cow::Owned(merged);
    }
    cur
}

fn static_type_name(t: &str) -> Option<&'static str> {
    ["null", "boolean", "integer", "number", "string", "array", "object"].into_iter().find(|n| *n == t)
}

fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn schema_types(p: &serde_json::Value, root: &serde_json::Value) -> Vec<&'static str> {
    fn walk(p: &serde_json::Value, root: &serde_json::Value, depth: u32, out: &mut Vec<&'static str>) {
        if depth > 16 {
            return;
        }
        let p = resolve_local_ref(p, root);
        match p.get("type") {
            Some(serde_json::Value::String(t)) => out.extend(static_type_name(t)),
            Some(serde_json::Value::Array(ts)) => {
                out.extend(ts.iter().filter_map(|t| t.as_str()).filter_map(static_type_name))
            }
            _ => {}
        }
        for k in ["anyOf", "oneOf", "allOf"] {
            if let Some(alts) = p.get(k).and_then(|a| a.as_array()) {
                for a in alts {
                    walk(a, root, depth + 1, out);
                }
            }
        }
        if let Some(vals) = p.get("enum").and_then(|e| e.as_array()) {
            for v in vals {
                out.push(json_type_name(v));
            }
        }
        if let Some(c) = p.get("const") {
            out.push(json_type_name(c));
        }
    }
    let mut out = Vec::new();
    walk(p, root, 0, &mut out);
    out
}

fn json_is_one_of(v: &serde_json::Value, types: &[&str]) -> bool {
    use serde_json::Value as J;
    types.iter().any(|t| match (*t, v) {
        ("null", J::Null) | ("boolean", J::Bool(_)) | ("array", J::Array(_)) | ("object", J::Object(_)) => true,
        ("number", J::Number(_)) => true,
        ("integer", J::Number(n)) => n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0),
        _ => false,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StringUnion {
    Typed,
    #[cfg_attr(not(any(feature = "basert", test)), allow(dead_code))]
    Verbatim,
}

fn declared_value(
    raw: &str,
    prop: Option<&serde_json::Value>,
    root: &serde_json::Value,
    union: StringUnion,
) -> serde_json::Value {
    let types = prop.map(|p| schema_types(p, root)).unwrap_or_default();
    if types.is_empty() {
        return typed_value(raw);
    }
    let text = types.contains(&"string");
    let padded = raw.len() != raw.trim().len();
    if text && (padded || union == StringUnion::Verbatim) {
        return serde_json::Value::String(raw.to_string());
    }
    let t = raw.trim();
    let parsed = serde_json::from_str::<serde_json::Value>(t).ok().or_else(|| {
        let py = match t {
            "True" => serde_json::Value::Bool(true),
            "False" => serde_json::Value::Bool(false),
            "None" => serde_json::Value::Null,
            _ => return None,
        };
        (!text && json_is_one_of(&py, &types)).then_some(py)
    });
    match parsed {
        Some(j) if !j.is_string() && json_is_one_of(&j, &types) => j,
        _ if text => serde_json::Value::String(raw.to_string()),
        Some(j) => j,
        None => serde_json::Value::String(raw.to_string()),
    }
}

fn glm_value(raw: &str, prop: Option<&serde_json::Value>, root: &serde_json::Value) -> serde_json::Value {
    declared_value(raw, prop, root, StringUnion::Typed)
}

fn glm_bare_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(char::is_whitespace) && !name.contains(['{', '}', '<', '>', '"'])
}

pub fn parse_glm_tool_call(raw: &str, schemas: Option<&ToolSchemas>) -> Option<(String, String)> {
    const AK_O: &str = GLM_ARG_KEY_OPEN;
    const AK_C: &str = GLM_ARG_KEY_CLOSE;
    const AV_O: &str = GLM_ARG_VALUE_OPEN;
    const AV_C: &str = GLM_ARG_VALUE_CLOSE;
    let mut body = raw;
    if let Some(i) = body.find("<tool_call>") {
        body = &body[i + "<tool_call>".len()..];
    }
    if let Some(i) = body.find("</tool_call>") {
        body = &body[..i];
    }
    let first_key = body.find(AK_O);
    let name = body[..first_key.unwrap_or(body.len())].trim();
    if !glm_bare_name(name) {
        return None;
    }
    let mut args = serde_json::Map::new();
    let mut pos = first_key;
    while let Some(p) = pos {
        let k_start = p + AK_O.len();
        let k_end = k_start + body[k_start..].find(AK_C)?;
        let key = body[k_start..k_end].trim();
        let after_key = k_end + AK_C.len();
        let v_open = after_key + body[after_key..].find(AV_O)?;
        if !body[after_key..v_open].trim().is_empty() {
            return None;
        }
        let v_start = v_open + AV_O.len();
        let v_end = v_start + body[v_start..].find(AV_C)?;
        if !key.is_empty() {
            let prop = schemas.and_then(|s| s.property(name, key));
            let root = schemas.and_then(|s| s.params(name)).unwrap_or(&serde_json::Value::Null);
            args.insert(key.to_string(), glm_value(&body[v_start..v_end], prop, root));
        }
        let after = &body[v_end + AV_C.len()..];
        let next = after.trim_start();
        if next.is_empty() {
            break;
        }
        if !next.starts_with(AK_O) {
            return None;
        }
        pos = Some(body.len() - next.len());
    }
    if let Some(required) = schemas
        .and_then(|s| s.args_schema(name))
        .and_then(|p| p.get("required"))
        .and_then(|r| r.as_array())
    {
        if required.iter().filter_map(|k| k.as_str()).any(|k| !args.contains_key(k)) {
            return None;
        }
    }
    Some((name.to_string(), serde_json::Value::Object(args).to_string()))
}

const GLM_ARG_KEY_OPEN: &str = "<arg_key>";
const GLM_ARG_KEY_CLOSE: &str = "</arg_key>";
const GLM_ARG_VALUE_OPEN: &str = "<arg_value>";
const GLM_ARG_VALUE_CLOSE: &str = "</arg_value>";

fn glm_value_format(schema: Option<&serde_json::Value>, params: &serde_json::Value) -> serde_json::Value {
    let any_text = serde_json::json!({ "type": "any_text" });
    let Some(schema) = schema else { return any_text };
    let schema = resolve_local_ref(schema, params);
    if let Some(vals) = fixed_values(&schema, params, 0) {
        let elems: Vec<serde_json::Value> = vals
            .into_iter()
            .map(|v| match v {
                serde_json::Value::String(s) => serde_json::json!({ "type": "const_string", "value": s }),
                other => serde_json::json!({ "type": "const_string", "value": python_json(&other) }),
            })
            .collect();
        return if elems.len() == 1 {
            elems.into_iter().next().unwrap_or(any_text)
        } else {
            serde_json::json!({ "type": "or", "elements": elems })
        };
    }
    let types = schema_types(&schema, params);
    if types.is_empty() || types.contains(&"string") {
        return any_text;
    }
    let mut schema = schema.into_owned();
    if let Some(obj) = schema.as_object_mut() {
        for defs in ["$defs", "definitions"] {
            if let (Some(d), false) = (params.get(defs), obj.contains_key(defs)) {
                obj.insert(defs.to_string(), d.clone());
            }
        }
    }
    serde_json::json!({ "type": "json_schema", "json_schema": schema })
}

fn fixed_values(schema: &serde_json::Value, root: &serde_json::Value, depth: u32) -> Option<Vec<serde_json::Value>> {
    if depth > 16 {
        return None;
    }
    let schema = resolve_local_ref(schema, root);
    if let Some(c) = schema.get("const") {
        return Some(vec![c.clone()]);
    }
    if let Some(vals) = schema.get("enum").and_then(|e| e.as_array()).filter(|v| !v.is_empty()) {
        return Some(vals.clone());
    }
    for k in ["anyOf", "oneOf"] {
        if let Some(alts) = schema.get(k).and_then(|a| a.as_array()).filter(|a| !a.is_empty()) {
            let mut out: Vec<serde_json::Value> = Vec::new();
            for a in alts {
                for v in fixed_values(a, root, depth + 1)? {
                    if !out.contains(&v) {
                        out.push(v);
                    }
                }
            }
            return Some(out);
        }
    }
    None
}

fn python_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Array(a) => format!("[{}]", a.iter().map(python_json).collect::<Vec<_>>().join(", ")),
        serde_json::Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", serde_json::Value::String(k.clone()), python_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        scalar => scalar.to_string(),
    }
}

fn glm_call_body_format(name: &str, params: &serde_json::Value) -> serde_json::Value {
    let args = resolve_local_ref(params, params);
    let props = args.get("properties").and_then(|p| p.as_object());
    let required: Vec<String> = args
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let arg = |key: &str, schema: Option<&serde_json::Value>| {
        serde_json::json!({
            "type": "sequence",
            "elements": [
                { "type": "const_string", "value": format!("{GLM_ARG_KEY_OPEN}{key}{GLM_ARG_KEY_CLOSE}") },
                {
                    "type": "tag",
                    "begin": GLM_ARG_VALUE_OPEN,
                    "content": glm_value_format(schema, params),
                    "end": GLM_ARG_VALUE_CLOSE,
                },
            ]
        })
    };
    let mut elements = vec![serde_json::json!({ "type": "const_string", "value": name })];
    for (key, schema) in props.into_iter().flatten() {
        if required.contains(key) {
            elements.push(arg(key, Some(schema)));
        } else {
            elements.push(serde_json::json!({ "type": "optional", "content": arg(key, Some(schema)) }));
        }
    }
    for key in required.iter().filter(|k| !props.is_some_and(|p| p.contains_key(k.as_str()))) {
        elements.push(arg(key, None));
    }
    serde_json::json!({ "type": "sequence", "elements": elements })
}

fn wrap_structural_tag(format: serde_json::Value) -> String {
    serde_json::json!({ "type": "structural_tag", "format": format }).to_string()
}

fn stop_after_first(at_least_one: bool) -> bool {
    at_least_one
}

fn value_format(schema: &serde_json::Value, string_pattern: &str) -> serde_json::Value {
    if let Some(vals) = schema.get("enum").and_then(|e| e.as_array()) {
        let elems: Vec<serde_json::Value> = vals
            .iter()
            .map(|v| {
                let lit = v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string());
                serde_json::json!({ "type": "const_string", "value": lit })
            })
            .collect();
        if !elems.is_empty() {
            return serde_json::json!({ "type": "or", "elements": elems });
        }
    }
    match schema.get("type").and_then(|t| t.as_str()).unwrap_or("string") {
        "integer" => serde_json::json!({ "type": "regex", "pattern": r"-?[0-9]+" }),
        "number" => serde_json::json!({ "type": "regex", "pattern": r"-?[0-9]+(\.[0-9]+)?" }),
        "boolean" => serde_json::json!({
            "type": "or",
            "elements": [
                { "type": "const_string", "value": "true" },
                { "type": "const_string", "value": "false" },
            ]
        }),
        "array" => serde_json::json!({
            "type": "regex",
            "pattern": r"\[[^{}\[\]\n]*\]"
        }),
        "object" => serde_json::json!({ "type": "regex", "pattern": r"\{[^{}\n]*\}" }),
        _ => serde_json::json!({ "type": "regex", "pattern": string_pattern }),
    }
}

const GEMMA_STRING: &str = r"[A-Za-z0-9_.:/=@+-][A-Za-z0-9 _.:/=@+-]*";

const ATEM_STRING: &str = r"[A-Za-z0-9_.:/=@+,(){}\[\]!?'-][A-Za-z0-9 _.:/=@+,(){}\[\]!?'-]*";

fn gemma_value_format(schema: &serde_json::Value) -> serde_json::Value {
    value_format(schema, GEMMA_STRING)
}

fn atem_call_body_format(name: &str, params: &serde_json::Value) -> serde_json::Value {
    let props = params.get("properties").and_then(|p| p.as_object());
    let required: Vec<String> = params
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let param = |key: &str| -> serde_json::Value {
        let schema = props
            .and_then(|p| p.get(key))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({ "type": "string" }));
        serde_json::json!({
            "type": "sequence",
            "elements": [
                { "type": "const_string", "value": format!("<atem:parameter name=\"{key}\">") },
                value_format(&schema, ATEM_STRING),
                { "type": "const_string", "value": "</atem:parameter>\n" },
            ]
        })
    };
    let all: Vec<String> = props.map(|p| p.keys().cloned().collect()).unwrap_or_default();

    let mut elements = vec![serde_json::json!({
        "type": "const_string",
        "value": format!("\n<atem:invoke name=\"{name}\">\n"),
    })];
    for key in &all {
        let e = param(key);
        if required.contains(key) {
            elements.push(e);
        } else {
            elements.push(serde_json::json!({ "type": "optional", "content": e }));
        }
    }
    elements.push(serde_json::json!({ "type": "const_string", "value": "</atem:invoke>\n" }));
    serde_json::json!({ "type": "sequence", "elements": elements })
}

fn gemma_call_body_format(name: &str, params: &serde_json::Value) -> serde_json::Value {
    let props = params.get("properties").and_then(|p| p.as_object());
    let required: Vec<String> = params
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let pair = |key: &str| -> serde_json::Value {
        let schema = props
            .and_then(|p| p.get(key))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({ "type": "string" }));
        serde_json::json!({
            "type": "sequence",
            "elements": [
                { "type": "const_string", "value": format!("{key}:") },
                gemma_value_format(&schema),
            ]
        })
    };

    let mut all: Vec<String> = props.map(|p| p.keys().cloned().collect()).unwrap_or_default();
    all.sort();

    let mut elements = vec![serde_json::json!({
        "type": "const_string",
        "value": format!("call:{name}{{"),
    })];
    let mut first = true;
    for key in all.iter().filter(|k| required.contains(k)) {
        if !first {
            elements.push(serde_json::json!({ "type": "const_string", "value": "," }));
        }
        first = false;
        elements.push(pair(key));
    }
    elements.push(serde_json::json!({ "type": "const_string", "value": "}" }));
    serde_json::json!({ "type": "sequence", "elements": elements })
}

pub const GEMMA_TOOL_OPEN: &str = "<|tool_call>";
pub const GEMMA_TOOL_CLOSE: &str = "<tool_call|>";

pub const GEMMA_TOOL_RESPONSE_OPEN: &str = "<|tool_response>";
pub const GEMMA_TOOL_RESPONSE_CLOSE: &str = "<tool_response|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolEnvelope {
    #[default]
    Json,
    Atem,
    Harmony,
    Gemma,
    Glm,
}

impl ToolEnvelope {
    pub fn from_name(s: &str) -> Option<ToolEnvelope> {
        match s {
            "json" => Some(ToolEnvelope::Json),
            "atem" => Some(ToolEnvelope::Atem),
            "gemma" => Some(ToolEnvelope::Gemma),
            "harmony" => Some(ToolEnvelope::Harmony),
            "glm" => Some(ToolEnvelope::Glm),
            _ => None,
        }
    }

    pub fn names() -> &'static [&'static str] {
        &["json", "atem", "gemma", "harmony", "glm"]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ToolEnvelope::Json => "json",
            ToolEnvelope::Atem => "atem",
            ToolEnvelope::Gemma => "gemma",
            ToolEnvelope::Harmony => "harmony",
            ToolEnvelope::Glm => "glm",
        }
    }

    pub fn parse_call(&self, raw: &str, schemas: Option<&ToolSchemas>) -> Option<(String, String)> {
        match self {
            ToolEnvelope::Json => parse_json_tool_call(raw),
            ToolEnvelope::Atem => crate::atem::parse_atem_call(raw),
            ToolEnvelope::Harmony => crate::harmony::parse_harmony_tool_call(raw),
            ToolEnvelope::Gemma => parse_gemma_tool_call(raw),
            ToolEnvelope::Glm => parse_glm_tool_call(raw, schemas),
        }
    }

    pub fn named_call(&self, raw: &str) -> Option<(String, String)> {
        match self {
            ToolEnvelope::Harmony => crate::harmony::named_call(raw),
            ToolEnvelope::Json | ToolEnvelope::Atem | ToolEnvelope::Gemma | ToolEnvelope::Glm => None,
        }
    }

    pub fn structural_tag(
        &self,
        tool_jsons: &[String],
        begin: &str,
        end: &str,
        at_least_one: bool,
    ) -> Option<String> {
        if tool_jsons.is_empty() || begin.is_empty() || end.is_empty() {
            return None;
        }
        match self {
            ToolEnvelope::Harmony => None,
            ToolEnvelope::Atem => {
                let mut tags = Vec::new();
                for t in tool_jsons {
                    let v: serde_json::Value = serde_json::from_str(t).ok()?;
                    let f = v.get("function").unwrap_or(&v);
                    let name = f.get("name").and_then(|n| n.as_str())?;
                    let params = f
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
                    tags.push(serde_json::json!({
                        "type": "tag",
                        "begin": begin,
                        "content": atem_call_body_format(name, &params),
                        "end": end,
                    }));
                }
                Some(wrap_structural_tag(serde_json::json!({
                    "type": "triggered_tags",
                    "triggers": [begin],
                    "tags": tags,
                    "at_least_one": at_least_one,
                    "stop_after_first": stop_after_first(at_least_one),
                })))
            }
            ToolEnvelope::Glm => {
                if !at_least_one {
                    return None;
                }
                let mut tags = Vec::new();
                for t in tool_jsons {
                    let v: serde_json::Value = serde_json::from_str(t).ok()?;
                    let f = v.get("function").unwrap_or(&v);
                    let name = f.get("name").and_then(|n| n.as_str())?;
                    let params = f
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
                    tags.push(serde_json::json!({
                        "type": "tag",
                        "begin": begin,
                        "content": glm_call_body_format(name, &params),
                        "end": end,
                    }));
                }
                Some(wrap_structural_tag(serde_json::json!({
                    "type": "triggered_tags",
                    "triggers": [begin],
                    "tags": tags,
                    "at_least_one": at_least_one,
                    "stop_after_first": stop_after_first(at_least_one),
                })))
            }
            ToolEnvelope::Gemma => {
                if !at_least_one {
                    return None;
                }
                let mut tags = Vec::new();
                for t in tool_jsons {
                    let v: serde_json::Value = serde_json::from_str(t).ok()?;
                    let f = v.get("function").unwrap_or(&v);
                    let name = f.get("name").and_then(|n| n.as_str())?;
                    let params = f
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
                    tags.push(serde_json::json!({
                        "type": "tag",
                        "begin": begin,
                        "content": gemma_call_body_format(name, &params),
                        "end": end,
                    }));
                }
                Some(wrap_structural_tag(serde_json::json!({
                    "type": "triggered_tags",
                    "triggers": [begin],
                    "tags": tags,
                    "at_least_one": at_least_one,
                    "stop_after_first": stop_after_first(at_least_one),
                })))
            }
            ToolEnvelope::Json => {
                let mut tags = Vec::new();
                for t in tool_jsons {
                    let v: serde_json::Value = serde_json::from_str(t).ok()?;
                    let f = v.get("function").unwrap_or(&v);
                    let name = f.get("name").and_then(|n| n.as_str())?;
                    let params = f
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
                    tags.push(serde_json::json!({
                        "type": "tag",
                        "begin": begin,
                        "content": {
                            "type": "json_schema",
                            "json_schema": {
                                "type": "object",
                                "properties": {
                                    "name": { "const": name },
                                    "arguments": params,
                                },
                                "required": ["name", "arguments"],
                                "additionalProperties": false,
                            },
                        },
                        "end": end,
                    }));
                }
                Some(wrap_structural_tag(serde_json::json!({
                    "type": "triggered_tags",
                    "triggers": [begin],
                    "tags": tags,
                    "at_least_one": at_least_one,
                    "stop_after_first": stop_after_first(at_least_one),
                })))
            }
        }
    }

    pub fn call_grammar(&self, tool_jsons: &[String]) -> Option<String> {
        match self {
            ToolEnvelope::Atem => None,
            ToolEnvelope::Gemma => None,
            ToolEnvelope::Harmony => None,
            ToolEnvelope::Glm => None,
            ToolEnvelope::Json => {
                if tool_jsons.is_empty() {
                    return None;
                }
                let mut variants = Vec::new();
                for t in tool_jsons {
                    let v: serde_json::Value = serde_json::from_str(t).ok()?;
                    let f = v.get("function").unwrap_or(&v);
                    let name = f.get("name").and_then(|n| n.as_str())?;
                    let params = f
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
                    variants.push(serde_json::json!({
                        "type": "object",
                        "properties": { "name": { "const": name }, "arguments": params },
                        "required": ["name", "arguments"],
                        "additionalProperties": false,
                    }));
                }
                let schema = if variants.len() == 1 {
                    variants.pop().unwrap()
                } else {
                    serde_json::json!({ "anyOf": variants })
                };
                Some(schema.to_string())
            }
        }
    }
}

pub fn tool_envelope_override_conflict(codec: &dyn TextCodec, env: ToolEnvelope) -> Option<String> {
    let own = codec.tool_envelope();
    if env == own {
        return None;
    }
    if env == ToolEnvelope::Glm
        && codec
            .chat_template_source()
            .is_some_and(|t| t.contains(GLM_ARG_KEY_OPEN) && t.contains(GLM_ARG_VALUE_OPEN))
    {
        return None;
    }
    Some(format!(
        "--tool-call-parser {} conflicts with this model's own tool wire ({}). The dialect renders \
         its instructions from the checkpoint's template, so the override would instruct the model \
         in one format while parsing another. Drop the flag to use the derived wire{}.",
        env.name(),
        own.name(),
        if env == ToolEnvelope::Glm {
            " (`glm` is accepted only where the chat template itself spells the GLM \
             <arg_key>/<arg_value> wire)"
        } else {
            ""
        }
    ))
}

pub struct WithToolEnvelope {
    inner: Box<dyn TextCodec + Send + Sync>,
    envelope: ToolEnvelope,
}

impl WithToolEnvelope {
    pub fn new(inner: Box<dyn TextCodec + Send + Sync>, envelope: ToolEnvelope) -> Self {
        WithToolEnvelope { inner, envelope }
    }
}

impl TextCodec for WithToolEnvelope {
    fn tool_envelope(&self) -> ToolEnvelope {
        self.envelope
    }
    fn renders_per_message(&self) -> bool {
        self.inner.renders_per_message()
    }
    fn tool_call_delimiters(&self) -> Option<(String, String)> {
        self.inner.tool_call_delimiters()
    }
    fn marker_literal(&self, token: u32) -> String {
        self.inner.marker_literal(token)
    }
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }
    fn encode_content(&self, text: &str) -> Vec<u32> {
        self.inner.encode_content(text)
    }
    fn encode_pieces(&self, pieces: &[(&str, bool)]) -> Vec<u32> {
        self.inner.encode_pieces(pieces)
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        self.inner.token_bytes(token)
    }
    fn chat_template_source(&self) -> Option<String> {
        self.inner.chat_template_source()
    }
    fn decode(&self, tokens: &[u32]) -> String {
        self.inner.decode(tokens)
    }
    fn stream_prologue(&self) -> Vec<u32> {
        self.inner.stream_prologue()
    }
    fn behavior_fingerprint(&self) -> Option<superfluid_fingerprint::BehaviorFingerprint> {
        self.inner.behavior_fingerprint()
    }
    fn render_message(&self, role: u32, text: &str) -> Option<Vec<u32>> {
        self.inner.render_message(role, text)
    }
    fn generation_prefix(&self, state: TurnState) -> Option<Vec<u32>> {
        self.inner.generation_prefix(state)
    }
    fn generation_prefix_with(&self, state: TurnState, thinking: Option<bool>) -> Option<Vec<u32>> {
        self.inner.generation_prefix_with(state, thinking)
    }
    fn turn_separator(&self) -> Option<Vec<u32>> {
        self.inner.turn_separator()
    }
    fn channel_markers(&self) -> Vec<(u32, u32, u32)> {
        self.inner.channel_markers()
    }
    fn response_format_tag(&self, json_schema: &str) -> Option<String> {
        self.inner.response_format_tag(json_schema)
    }
    fn render_system_with_tools(
        &self,
        system: Option<&str>,
        tool_jsons: &[String],
    ) -> Option<Vec<u32>> {
        self.inner.render_system_with_tools(system, tool_jsons)
    }
    fn render_tool_result(&self, name: &str, content: &str) -> Option<Vec<u32>> {
        self.inner.render_tool_result(name, content)
    }
    fn render_assistant_with_tool_calls(
        &self,
        content: &str,
        calls: &[(String, String)],
    ) -> Option<Vec<u32>> {
        self.inner.render_assistant_with_tool_calls(content, calls)
    }
    fn render_assistant_with_tool_calls_after_query(
        &self,
        content: &str,
        calls: &[(String, String)],
    ) -> Option<Vec<u32>> {
        self.inner.render_assistant_with_tool_calls_after_query(content, calls)
    }
    fn render_fim(&self, prefix: &str, suffix: &str, mode: u8) -> Option<Vec<u32>> {
        self.inner.render_fim(prefix, suffix, mode)
    }
    fn render_turn_with_tokens(
        &self,
        role: u32,
        pre_text: &str,
        inner: &[u32],
        post_text: &str,
    ) -> Option<Vec<u32>> {
        self.inner.render_turn_with_tokens(role, pre_text, inner, post_text)
    }
    fn render_block(&self, role: u32, kind: u32, payload: &str) -> Option<Vec<u32>> {
        self.inner.render_block(role, kind, payload)
    }
    fn channelizer(&self) -> Box<dyn Channelizer> {
        self.inner.channelizer()
    }
    fn render_prompt(&self, messages: &[(u32, String)], tools: &[String]) -> Option<Vec<u32>> {
        self.inner.render_prompt(messages, tools)
    }
    fn turn_terminators(&self) -> Vec<u32> {
        self.inner.turn_terminators()
    }
    fn render_prompt_structured(
        &self,
        messages: &[ChatMessage],
        tools: &[String],
    ) -> Option<Vec<u32>> {
        self.inner.render_prompt_structured(messages, tools)
    }
    fn render_prompt_structured_with(
        &self,
        messages: &[ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        self.inner.render_prompt_structured_with(messages, tools, kwargs)
    }
    fn template_refusal(
        &self,
        messages: &[ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<String> {
        self.inner.template_refusal(messages, tools, kwargs)
    }
    fn render_system_full(
        &self,
        system: Option<&str>,
        tool_jsons: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        self.inner.render_system_full(system, tool_jsons, kwargs)
    }
    fn validate_template_kwargs(
        &self,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        self.inner.validate_template_kwargs(kwargs)
    }
    fn supports_reasoning_effort(&self) -> bool {
        self.inner.supports_reasoning_effort()
    }
    fn reasoning_effort_levels(&self) -> Vec<&'static str> {
        self.inner.reasoning_effort_levels()
    }
    fn supports_enable_thinking(&self) -> bool {
        self.inner.supports_enable_thinking()
    }
    fn empty_think_tokens(&self) -> Option<Vec<u32>> {
        self.inner.empty_think_tokens()
    }
}

pub trait Channelizer: Send {
    fn split(&mut self, tokens: &[u32]) -> Vec<Run>;
    fn current_channel(&self) -> u32;
    fn prime(&mut self, _channel: u32) {}
}

pub mod fim_mode {
    pub const PSM: u8 = 0;
    pub const SPM: u8 = 1;
}

pub const MODEL_VISIBILITY_VERSION: u32 = 2;

pub fn project_block_text(kind: u32, payload: &str) -> Option<String> {
    use crate::wal::block_kind;
    let v: serde_json::Value = serde_json::from_str(payload).unwrap_or(serde_json::Value::Null);
    let str_of = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let cap = |mut t: String, max: usize| {
        if t.len() > max {
            let mut cut = max;
            while !t.is_char_boundary(cut) {
                cut -= 1;
            }
            t.truncate(cut);
            t.push_str("\n…[truncated]");
        }
        t
    };
    Some(match kind {
        block_kind::FILE_DIFF => {
            let path = str_of("path");
            let diff = str_of("diff");
            cap(format!("```diff\n# {path}\n{diff}\n```"), 16 * 1024)
        }
        block_kind::ARTIFACT => {
            let title = str_of("title");
            let content = str_of("content");
            cap(format!("[artifact: {title}]\n{content}"), 16 * 1024)
        }
        block_kind::CITATION => {
            let title = str_of("title");
            let url = str_of("url");
            let quote = str_of("quote");
            if quote.is_empty() {
                format!("[citation] {title} — {url}")
            } else {
                format!("[citation] {title} — {url}\n> {quote}")
            }
        }
        block_kind::PROGRESS => String::new(),
        block_kind::DIAGNOSTIC => {
            let severity = str_of("severity");
            let message = str_of("message");
            let path = str_of("path");
            let line = v.get("line").and_then(|x| x.as_u64()).unwrap_or(0);
            if path.is_empty() {
                format!("[diagnostic {severity}] {message}")
            } else {
                format!("[diagnostic {severity}] {path}:{line}: {message}")
            }
        }
        block_kind::WORKSPACE_REF => format!("@{}", str_of("path")),
        block_kind::IMAGE => String::new(),
        _ => return None,
    })
}

pub struct Segmenter {
    pairs: Vec<(u32, u32, u32)>,
    active: Option<usize>,
    names: Vec<(u32, Vec<u32>)>,
    name_at: Option<usize>,
}

impl Segmenter {
    pub fn new(pairs: Vec<(u32, u32, u32)>) -> Segmenter {
        Segmenter {
            pairs,
            active: None,
            names: Vec::new(),
            name_at: None,
        }
    }

    pub fn with_names(mut self, names: Vec<(u32, Vec<u32>)>) -> Segmenter {
        self.names = names;
        self
    }

    fn name_of(&self, open: u32) -> Option<&[u32]> {
        self.names.iter().find(|(o, _)| *o == open).map(|(_, n)| n.as_slice())
    }

    fn takes_name(&mut self, open: u32, t: u32) -> bool {
        let (Some(k), Some(name)) = (self.name_at, self.name_of(open)) else {
            return false;
        };
        let named = name.get(k) == Some(&t);
        let len = name.len();
        self.name_at = (named && k + 1 < len).then_some(k + 1);
        named
    }

    fn is_marker(&self, t: u32) -> bool {
        self.pairs.iter().any(|&(o, c, _)| o == t || c == t)
    }
}

impl Segmenter {
    fn pair_for(&self, channel: u32) -> Option<usize> {
        self.pairs.iter().position(|&(_, _, c)| c == channel)
    }
}

impl Channelizer for Segmenter {
    fn split(&mut self, tokens: &[u32]) -> Vec<Run> {
        if self.pairs.is_empty() {
            return vec![Run {
                span: tokens.to_vec(),
                text: tokens.to_vec(),
                channel: crate::wal::channel::TEXT,
                closes: false,
            }];
        }
        let mut out: Vec<Run> = Vec::new();
        let mut run: Vec<u32> = Vec::new();
        let mut text: Vec<u32> = Vec::new();
        let mut run_channel = self.current_channel();
        fn flush(
            run: &mut Vec<u32>,
            text: &mut Vec<u32>,
            ch: u32,
            closes: bool,
            out: &mut Vec<Run>,
        ) {
            if !run.is_empty() {
                out.push(Run {
                    span: std::mem::take(run),
                    text: std::mem::take(text),
                    channel: ch,
                    closes,
                });
            }
        }
        for &t in tokens {
            match self.active {
                None => {
                    if let Some(i) = self.pairs.iter().position(|&(open, _, _)| open == t) {
                        flush(&mut run, &mut text, run_channel, false, &mut out);
                        self.active = Some(i);
                        self.name_at = self.name_of(t).map(|_| 0);
                        run_channel = self.pairs[i].2;
                        run.push(t);
                    } else {
                        run.push(t);
                        if !self.is_marker(t) {
                            text.push(t);
                        }
                    }
                }
                Some(i) => {
                    if self.pairs[i].1 == t {
                        run.push(t);
                        flush(&mut run, &mut text, run_channel, true, &mut out);
                        self.active = None;
                        self.name_at = None;
                        run_channel = crate::wal::channel::TEXT;
                    } else {
                        run.push(t);
                        let named = self.takes_name(self.pairs[i].0, t);
                        if !named && !self.is_marker(t) {
                            text.push(t);
                        }
                    }
                }
            }
        }
        flush(&mut run, &mut text, run_channel, false, &mut out);
        out
    }

    fn current_channel(&self) -> u32 {
        self.active
            .map(|i| self.pairs[i].2)
            .unwrap_or(crate::wal::channel::TEXT)
    }

    fn prime(&mut self, channel: u32) {
        if let Some(i) = self.pair_for(channel) {
            self.active = Some(i);
            self.name_at = None;
        }
    }
}

pub struct MockCodec;

const MOCK_BASE: u32 = 0x100;

impl TextCodec for MockCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| MOCK_BASE + b as u32).collect()
    }

    fn token_bytes(&self, token: u32) -> Vec<u8> {
        if (MOCK_BASE..MOCK_BASE + 256).contains(&token) {
            vec![(token - MOCK_BASE) as u8]
        } else {
            Vec::new()
        }
    }
}

pub struct MockChatCodec;

pub const MOCK_TURN_END: u32 = MOCK_BASE + 0x1E;
pub const MOCK_THINK_OPEN: u32 = MOCK_BASE + 0x02;
pub const MOCK_THINK_CLOSE: u32 = MOCK_BASE + 0x03;
pub const MOCK_TOOL_OPEN: u32 = MOCK_BASE + 0x04;
pub const MOCK_TOOL_CLOSE: u32 = MOCK_BASE + 0x05;

impl TextCodec for MockChatCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        MockCodec.encode(text)
    }
    fn encode_content(&self, text: &str) -> Vec<u32> {
        MockCodec
            .encode(text)
            .into_iter()
            .map(|t| {
                if [MOCK_TURN_END, MOCK_THINK_OPEN, MOCK_THINK_CLOSE, MOCK_TOOL_OPEN, MOCK_TOOL_CLOSE]
                    .contains(&t)
                {
                    MOCK_BASE + b'?' as u32
                } else {
                    t
                }
            })
            .collect()
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        MockCodec.token_bytes(token)
    }
    fn behavior_fingerprint(&self) -> Option<superfluid_fingerprint::BehaviorFingerprint> {
        Some(superfluid_fingerprint::BehaviorFingerprint {
            codec_id: "mock-chat".into(),
            codec_version: 2,
            tokenizer_hash: superfluid_fingerprint::vocab_hash(2 * MOCK_BASE, |t| {
                MockCodec.token_bytes(t)
            }),
            marker_digest: superfluid_fingerprint::BehaviorFingerprint::marker_digest_of(
                &self.channel_markers(),
            ),
            sampling_impl_version: 1,
            block_projection_version: MODEL_VISIBILITY_VERSION,
            grammar_version: 0,
            speculation: None,
        })
    }
    fn render_message(&self, role: u32, text: &str) -> Option<Vec<u32>> {
        let mut span = MockCodec.encode(&format!("<{role}>"));
        span.extend(self.encode_content(text));
        span.push(MOCK_TURN_END);
        Some(span)
    }
    fn render_turn_with_tokens(
        &self,
        role: u32,
        pre_text: &str,
        inner: &[u32],
        post_text: &str,
    ) -> Option<Vec<u32>> {
        let mut span = MockCodec.encode(&format!("<{role}>"));
        span.extend(self.encode_content(pre_text));
        span.extend_from_slice(inner);
        span.extend(self.encode_content(post_text));
        span.push(MOCK_TURN_END);
        Some(span)
    }
    fn render_fim(&self, prefix: &str, suffix: &str, mode: u8) -> Option<Vec<u32>> {
        const FIM_PRE: u32 = 4_000_000;
        const FIM_SUF: u32 = 4_000_001;
        const FIM_MID: u32 = 4_000_002;
        let mut span = Vec::new();
        if mode == fim_mode::SPM {
            span.extend([FIM_PRE, FIM_SUF]);
            span.extend(MockCodec.encode(suffix));
            span.push(FIM_MID);
            span.extend(MockCodec.encode(prefix));
        } else {
            span.push(FIM_PRE);
            span.extend(MockCodec.encode(prefix));
            span.push(FIM_SUF);
            span.extend(MockCodec.encode(suffix));
            span.push(FIM_MID);
        }
        Some(span)
    }
    fn generation_prefix(&self, state: TurnState) -> Option<Vec<u32>> {
        match state {
            TurnState::AfterInput => Some(MockCodec.encode("<a>")),
            TurnState::MidTurn => None,
            TurnState::AfterFinishedTurn => {
                let mut s = vec![MOCK_TURN_END];
                s.extend(MockCodec.encode("<a>"));
                Some(s)
            }
        }
    }
    fn channel_markers(&self) -> Vec<(u32, u32, u32)> {
        vec![
            (
                MOCK_THINK_OPEN,
                MOCK_THINK_CLOSE,
                crate::wal::channel::REASONING,
            ),
            (
                MOCK_TOOL_OPEN,
                MOCK_TOOL_CLOSE,
                crate::wal::channel::TOOL_CALL,
            ),
        ]
    }
    fn render_system_with_tools(
        &self,
        system: Option<&str>,
        tool_jsons: &[String],
    ) -> Option<Vec<u32>> {
        let mut span = MockCodec.encode("<0>");
        span.extend(self.encode_content(system.unwrap_or("")));
        for t in tool_jsons {
            span.extend(MockCodec.encode(t));
        }
        span.push(MOCK_TURN_END);
        Some(span)
    }
    fn render_tool_result(&self, _name: &str, content: &str) -> Option<Vec<u32>> {
        let mut span = MockCodec.encode("<r>");
        span.extend(self.encode_content(content));
        span.push(MOCK_TURN_END);
        Some(span)
    }
    fn render_assistant_with_tool_calls(
        &self,
        content: &str,
        calls: &[(String, String)],
    ) -> Option<Vec<u32>> {
        let mut span = MockCodec.encode("<a>");
        span.extend(self.encode_content(content));
        for (name, args) in calls {
            span.push(MOCK_TOOL_OPEN);
            span.extend(MockCodec.encode(&format!(
                "{{\"name\": \"{name}\", \"arguments\": {args}}}"
            )));
            span.push(MOCK_TOOL_CLOSE);
        }
        span.push(MOCK_TURN_END);
        Some(span)
    }
}

pub struct MockTemplateCodec;

pub const MOCK_IMAGE_MARKER: u32 = 999;

impl TextCodec for MockTemplateCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        MockCodec.encode(text)
    }
    fn decode(&self, tokens: &[u32]) -> String {
        MockCodec.decode(tokens)
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        MockCodec.token_bytes(token)
    }
    fn renders_per_message(&self) -> bool {
        false
    }
    fn render_prompt_structured_with(
        &self,
        messages: &[ChatMessage],
        tools: &[String],
        _kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        let mut span = Vec::new();
        for m in messages {
            span.extend(MockCodec.encode(&format!("<{}>", m.role)));
            if m.parts.is_empty() {
                span.extend(MockCodec.encode(&m.content));
            } else {
                for p in &m.parts {
                    match p {
                        ContentPart::Text(t) => span.extend(MockCodec.encode(t)),
                        ContentPart::Image { .. } => span.push(MOCK_IMAGE_MARKER),
                        ContentPart::Audio { .. } => span.push(MOCK_IMAGE_MARKER),
                    }
                }
            }
            span.push(MOCK_TURN_END);
        }
        for t in tools {
            span.extend(MockCodec.encode(t));
        }
        span.extend(MockCodec.encode("<a>"));
        Some(span)
    }
}

pub struct MockThinkingTemplateCodec;

impl TextCodec for MockThinkingTemplateCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        MockCodec.encode(text)
    }
    fn decode(&self, tokens: &[u32]) -> String {
        MockCodec.decode(tokens)
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        MockCodec.token_bytes(token)
    }
    fn renders_per_message(&self) -> bool {
        false
    }
    fn channel_markers(&self) -> Vec<(u32, u32, u32)> {
        MockChatCodec.channel_markers()
    }
    fn render_prompt_structured_with(
        &self,
        messages: &[ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        let mut span = MockTemplateCodec.render_prompt_structured_with(messages, tools, kwargs)?;
        span.push(MOCK_THINK_OPEN);
        Some(span)
    }
    fn trailing_generation_prompt(
        &self,
        _messages: &[ChatMessage],
        _tools: &[String],
        _kwargs: &serde_json::Map<String, serde_json::Value>,
        span: &[u32],
    ) -> usize {
        let mut opener = MockCodec.encode("<a>");
        opener.push(MOCK_THINK_OPEN);
        if span.len() > opener.len() && span.ends_with(&opener) {
            opener.len()
        } else {
            0
        }
    }
}

pub struct NoCodec;

impl TextCodec for NoCodec {
    fn encode(&self, _text: &str) -> Vec<u32> {
        Vec::new()
    }
    fn token_bytes(&self, _token: u32) -> Vec<u8> {
        Vec::new()
    }
}

#[derive(Default)]
pub struct Utf8Stream {
    buf: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.buf.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // SAFETY-free: valid_up_to bounds a checked-valid
                    // prefix.
                    out.push_str(std::str::from_utf8(&self.buf[..valid]).expect("checked"));
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.buf.drain(..valid + n);
                        }
                        None => {
                            self.buf.drain(..valid);
                            return out;
                        }
                    }
                }
            }
        }
    }

    pub fn flush(&mut self) -> String {
        if self.buf.is_empty() {
            return String::new();
        }
        self.buf.clear();
        "\u{FFFD}".to_string()
    }
}

pub struct BundleCodec {
    handle: std::sync::Arc<dyn superfluid_engine::Tokenizer>,
    vocab_hash: u64,
    markers: std::sync::OnceLock<superfluid_engine::Markers>,
}

impl BundleCodec {
    #[cfg(feature = "basert")]
    pub fn load(model_path: &std::path::Path) -> Result<BundleCodec, Status> {
        let handle =
            superfluid_engine_ffi::TokenizerHandle::load(model_path).map_err(|_| Status::Fatal)?;
        Ok(Self::from_tokenizer(std::sync::Arc::new(handle)))
    }

    pub fn from_tokenizer(tok: std::sync::Arc<dyn superfluid_engine::Tokenizer>) -> BundleCodec {
        let vocab_hash = superfluid_fingerprint::vocab_hash(tok.vocab_size(), |t| tok.token_bytes(t));
        BundleCodec {
            handle: tok,
            vocab_hash,
            markers: std::sync::OnceLock::new(),
        }
    }

    fn markers(&self) -> &superfluid_engine::Markers {
        self.markers.get_or_init(|| superfluid_engine::Markers::of(self.handle.as_ref()))
    }

    pub(crate) fn tokenizer(&self) -> std::sync::Arc<dyn superfluid_engine::Tokenizer> {
        std::sync::Arc::clone(&self.handle)
    }

    pub(crate) fn fingerprint(&self, codec_id: &str, markers: &[(u32, u32, u32)])
        -> superfluid_fingerprint::BehaviorFingerprint {
        superfluid_fingerprint::BehaviorFingerprint {
            codec_id: codec_id.to_string(),
            codec_version: 2,
            tokenizer_hash: self.vocab_hash,
            marker_digest: superfluid_fingerprint::BehaviorFingerprint::marker_digest_of(markers),
            sampling_impl_version: 1,
            block_projection_version: MODEL_VISIBILITY_VERSION,
            grammar_version: 0,
            speculation: None,
        }
    }
}

impl TextCodec for BundleCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.handle.encode(text)
    }
    fn encode_content(&self, text: &str) -> Vec<u32> {
        self.handle
            .encode_plain(text)
            .unwrap_or_else(|| self.markers().encode_plain(self.handle.as_ref(), text))
    }
    fn encode_pieces(&self, pieces: &[(&str, bool)]) -> Vec<u32> {
        self.handle
            .encode_pieces(pieces)
            .unwrap_or_else(|| self.markers().encode_pieces(self.handle.as_ref(), pieces))
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        self.handle.token_bytes(token)
    }
    fn marker_literal(&self, token: u32) -> String {
        let text = self.decode(&[token]);
        if !text.is_empty() {
            return text;
        }
        self.handle
            .special_tokens()
            .into_iter()
            .find(|(_, id)| *id == token)
            .map(|(s, _)| s)
            .unwrap_or_default()
    }
    fn behavior_fingerprint(&self) -> Option<superfluid_fingerprint::BehaviorFingerprint> {
        Some(self.fingerprint("raw", &[]))
    }
    fn chat_template_source(&self) -> Option<String> {
        let source = self.handle.chat_template_jinja();
        (!source.trim().is_empty()).then_some(source)
    }
    fn render_fim(&self, prefix: &str, suffix: &str, mode: u8) -> Option<Vec<u32>> {
        let bos = self.handle.bos_token();
        let atomic = |m: &str| -> bool {
            let mut t = self.encode(m);
            if bos.is_some() && t.first() == bos.as_ref() && t.len() > 1 {
                t.remove(0);
            }
            t.len() == 1
        };
        const PRE: &str = "<|fim_prefix|>";
        const SUF: &str = "<|fim_suffix|>";
        const MID: &str = "<|fim_middle|>";
        if !(atomic(PRE) && atomic(SUF) && atomic(MID)) {
            return None;
        }
        let pieces: [(&str, bool); 5] = if mode == fim_mode::SPM {
            [(PRE, false), (SUF, false), (suffix, true), (MID, false), (prefix, true)]
        } else {
            [(PRE, false), (prefix, true), (SUF, false), (suffix, true), (MID, false)]
        };
        Some(self.encode_pieces(&pieces))
    }
}

#[cfg(feature = "basert")]
pub fn chatml_codec(model: &std::path::Path) -> Result<ChatMlCodec, &'static str> {
    let tok = superfluid_engine_ffi::TokenizerHandle::load(model).map_err(|_| "tokenizer load failed")?;
    chatml_codec_from_tokenizer(std::sync::Arc::new(tok))
}

pub fn chatml_codec_from_tokenizer(
    tok: std::sync::Arc<dyn superfluid_engine::Tokenizer>,
) -> Result<ChatMlCodec, &'static str> {
    let codec = ChatMlCodec::from_tokenizer(std::sync::Arc::clone(&tok))
        .map_err(|_| "tokenizer lacks the ChatML markers")?;
    if let Ok(t) = crate::template_codec::TemplateCodec::from_tokenizer(tok) {
        if !codec.frames_like(&t) {
            return Err(
                "this model's chat template does not frame turns the way the curated ChatML \
                 dialect does, so it would send every turn in a shape the checkpoint was not \
                 trained on. Use --dialect auto or --dialect template.",
            );
        }
    }
    Ok(codec)
}

#[cfg(feature = "basert")]
pub fn auto_codec(
    model: &std::path::Path,
) -> Result<(Box<dyn TextCodec + Send + Sync>, &'static str), &'static str> {
    let tok = superfluid_engine_ffi::TokenizerHandle::load(model)
        .map_err(|_| "no curated codec, no usable chat template, and no raw tokenizer")?;
    Ok(auto_codec_from_tokenizer(std::sync::Arc::new(tok)))
}

pub fn auto_codec_from_tokenizer(
    tok: std::sync::Arc<dyn superfluid_engine::Tokenizer>,
) -> (Box<dyn TextCodec + Send + Sync>, &'static str) {
    let tmpl = crate::template_codec::TemplateCodec::from_tokenizer(std::sync::Arc::clone(&tok)).ok();
    let curated = chatml_codec_from_tokenizer(std::sync::Arc::clone(&tok)).ok();
    if let Some(c) = curated {
        (Box::new(c), "chatml")
    } else if let Some(c) = tmpl {
        (Box::new(c), "template")
    } else {
        (Box::new(BundleCodec::from_tokenizer(tok)), "raw")
    }
}

pub(crate) const CONTENT_STAND_IN: &str = "BASERTCONTENTSTANDIN9f2c";

pub(crate) fn split_once_exact<'a>(text: &'a str, needle: &str) -> Option<(&'a str, &'a str)> {
    let (before, after) = text.split_once(needle)?;
    (!after.contains(needle)).then_some((before, after))
}

pub(crate) fn stand_in_key(n: usize) -> String {
    format!("{CONTENT_STAND_IN}{n}Z")
}

pub(crate) fn stand_in_strings(v: &mut serde_json::Value, subs: &mut Vec<(String, String)>) {
    match v {
        serde_json::Value::String(text) => {
            let key = stand_in_key(subs.len());
            subs.push((key.clone(), std::mem::take(text)));
            *text = key;
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(|i| stand_in_strings(i, subs)),
        serde_json::Value::Object(map) => map.values_mut().for_each(|i| stand_in_strings(i, subs)),
        _ => {}
    }
}

fn place_known_form(rest: &str, original: &str, follows: impl Fn(&str) -> bool) -> Option<usize> {
    let fits = |form: &str| rest.starts_with(form) && follows(&rest[form.len()..]);
    if fits(original) {
        return Some(original.len());
    }
    let trimmed = original.trim();
    if fits(trimmed) {
        return Some(trimmed.len());
    }
    let quoted = serde_json::to_string(original).unwrap_or_default();
    let json = quoted.get(1..quoted.len().saturating_sub(1)).unwrap_or("");
    let html_safe = json
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\'', "\\u0027");
    if fits(&html_safe) {
        return Some(html_safe.len());
    }
    if fits(json) {
        return Some(json.len());
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Placement {
    SearchFraming,
    KnownFormsOnly,
}

pub(crate) fn split_around_stand_ins(
    stand_ins: &str,
    subs: &[(String, String)],
    real: &str,
    placement: Placement,
) -> Option<Vec<(String, bool)>> {
    let mut hits: Vec<(usize, usize)> = Vec::with_capacity(subs.len());
    for (i, (key, _)) in subs.iter().enumerate() {
        let mut it = stand_ins.match_indices(key.as_str());
        match (it.next(), it.next()) {
            (Some((pos, _)), None) => hits.push((pos, i)),
            _ => return None,
        }
    }
    hits.sort_unstable();
    let mut pieces: Vec<(String, bool)> = Vec::with_capacity(2 * hits.len() + 1);
    let (mut si_at, mut real_at) = (0usize, 0usize);
    for (n, (pos, i)) in hits.iter().enumerate() {
        let framing = &stand_ins[si_at..*pos];
        if !real[real_at..].starts_with(framing) {
            return None;
        }
        real_at += framing.len();
        si_at = pos + subs[*i].0.len();
        let last = n + 1 == hits.len();
        let next_end = hits.get(n + 1).map(|(p, _)| *p).unwrap_or(stand_ins.len());
        let next_framing = &stand_ins[si_at..next_end];
        let rest = &real[real_at..];
        let follows = |after: &str| -> bool {
            if last {
                after == next_framing
            } else {
                after.starts_with(next_framing)
            }
        };
        let known = place_known_form(rest, &subs[*i].1, follows);
        let content_end = match known {
            Some(end) => end,
            None if placement == Placement::KnownFormsOnly => return None,
            None if last => {
                if !rest.ends_with(next_framing) {
                    return None;
                }
                rest.len() - next_framing.len()
            }
            None => {
                if next_framing.is_empty() {
                    return None;
                }
                let mut it = rest.match_indices(next_framing);
                match (it.next(), it.next()) {
                    (Some((p, _)), None) => p,
                    _ => return None,
                }
            }
        };
        pieces.push((framing.to_string(), false));
        pieces.push((rest[..content_end].to_string(), true));
        real_at += content_end;
    }
    pieces.push((stand_ins[si_at..].to_string(), false));
    Some(pieces)
}

#[cfg(test)]
mod glm_call_tests {
    use super::*;

    fn args(raw: &str) -> (String, serde_json::Value) {
        args_with(raw, None)
    }

    fn args_with(raw: &str, schemas: Option<&ToolSchemas>) -> (String, serde_json::Value) {
        let (n, a) = parse_glm_tool_call(raw, schemas).expect("parses");
        (n, serde_json::from_str(&a).unwrap())
    }

    fn schemas(tools: serde_json::Value) -> ToolSchemas {
        ToolSchemas::from_values(tools.as_array().unwrap().iter())
    }

    #[test]
    fn reads_glm_bodies_framed_or_not() {
        let (n, a) = args("get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>");
        assert_eq!(n, "get_weather");
        assert_eq!(a, serde_json::json!({"city": "Paris"}));
        let (n2, a2) = args("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>");
        assert_eq!((n2, a2), (n, a));
    }

    #[test]
    fn non_strings_come_back_typed_strings_stay_raw() {
        let (_, a) = args(
            "f<arg_key>n</arg_key><arg_value>3</arg_value><arg_key>flag</arg_key><arg_value>true</arg_value>\
             <arg_key>o</arg_key><arg_value>{\"x\": [1, 2]}</arg_value><arg_key>s</arg_key><arg_value>\"quoted\"</arg_value>\
             <arg_key>txt</arg_key><arg_value>a < b & {not json</arg_value>",
        );
        assert_eq!(
            a,
            serde_json::json!({"n": 3, "flag": true, "o": {"x": [1, 2]}, "s": "\"quoted\"", "txt": "a < b & {not json"})
        );
    }

    #[test]
    fn declared_strings_stay_strings_and_declared_types_are_honoured() {
        let s = schemas(serde_json::json!([{
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object", "properties": {
                "id": {"type": "string"}, "flag": {"type": "string"}, "nil": {"type": "string"},
                "obj": {"type": "string"},
                "n": {"type": "integer"}, "on": {"type": "boolean"}, "list": {"type": "array"},
                "maybe": {"type": ["integer", "null"]}, "either": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                "pick": {"enum": ["1", "2"]},
                "loose": {}
            }}}
        }]));
        let raw = "f<arg_key>id</arg_key><arg_value>3</arg_value><arg_key>flag</arg_key><arg_value>true</arg_value>\
                   <arg_key>nil</arg_key><arg_value>null</arg_value><arg_key>obj</arg_key><arg_value>{\"a\": 1}</arg_value>\
                   <arg_key>n</arg_key><arg_value>42</arg_value><arg_key>on</arg_key><arg_value>false</arg_value>\
                   <arg_key>list</arg_key><arg_value>[1, \"x\"]</arg_value><arg_key>maybe</arg_key><arg_value>null</arg_value>\
                   <arg_key>either</arg_key><arg_value>7</arg_value><arg_key>pick</arg_key><arg_value>2</arg_value>\
                   <arg_key>loose</arg_key><arg_value>5</arg_value><arg_key>undeclared</arg_key><arg_value>false</arg_value>";
        let (_, a) = args_with(raw, Some(&s));
        assert_eq!(
            a,
            serde_json::json!({
                "id": "3", "flag": "true", "nil": "null", "obj": "{\"a\": 1}",
                "n": 42, "on": false, "list": [1, "x"], "maybe": null, "either": 7, "pick": "2",
                "loose": 5, "undeclared": false
            })
        );
        let (_, b) = args_with("g<arg_key>id</arg_key><arg_value>3</arg_value>", Some(&s));
        assert_eq!(b, serde_json::json!({"id": 3}));
        let anth = schemas(serde_json::json!([{"name": "f", "input_schema": {"properties": {"id": {"type": "string"}}}}]));
        assert_eq!(args_with("f<arg_key>id</arg_key><arg_value>3</arg_value>", Some(&anth)).1, serde_json::json!({"id": "3"}));
    }

    #[test]
    fn string_values_keep_their_boundary_whitespace() {
        let raw = "write<arg_key>content</arg_key><arg_value>\n    indented\n</arg_value>\
                   <arg_key>pad</arg_key><arg_value>  x  </arg_value><arg_key>n</arg_key><arg_value> 3 </arg_value>";
        let (_, a) = args(raw);
        assert_eq!(a, serde_json::json!({"content": "\n    indented\n", "pad": "  x  ", "n": 3}));
        let s = schemas(serde_json::json!([{"name": "write", "parameters": {"properties": {"n": {"type": "string"}}}}]));
        assert_eq!(args_with(raw, Some(&s)).1["n"], " 3 ");
        let (n, a) = args("<tool_call>f\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n</tool_call>");
        assert_eq!((n.as_str(), a), ("f", serde_json::json!({"k": "v"})));
    }

    #[test]
    fn a_call_cut_inside_an_argument_is_not_a_call() {
        for cut in [
            "f<arg_key>path</arg_key><arg_value>/etc/pas",
            "f<arg_key>path</arg_key><arg_value>",
            "f<arg_key>path</arg_key>",
            "f<arg_key>pa",
            "f<arg_key>",
            "f<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key><arg_value>par",
            "f<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b",
        ] {
            assert_eq!(parse_glm_tool_call(cut, None), None, "{cut:?}");
        }
        assert!(parse_glm_tool_call("f<arg_key>a</arg_key><arg_value>1</arg_value>", None).is_some());
        assert_eq!(
            parse_glm_tool_call("f<arg_key>a</arg_key>junk<arg_value>1</arg_value>", None),
            None
        );
        for junk in [
            "f<arg_key>a</arg_key><arg_value>1</arg_value>junk",
            "f<arg_key>a</arg_key><arg_value>1</arg_value>junk<arg_key>b</arg_key><arg_value>2</arg_value>",
        ] {
            assert_eq!(parse_glm_tool_call(junk, None), None, "{junk:?}");
        }
        assert!(parse_glm_tool_call("f<arg_key>a</arg_key><arg_value>1</arg_value>\n<arg_key>b</arg_key><arg_value>2</arg_value>\n", None).is_some());
    }

    #[test]
    fn refs_and_non_string_value_sets_are_honoured() {
        let tool = serde_json::json!({"name": "f", "parameters": {
            "type": "object",
            "$defs": {"Id": {"type": "string"}, "Level": {"enum": [1, 2, 3]}, "Alias": {"$ref": "#/$defs/Id"}},
            "properties": {
                "id": {"$ref": "#/$defs/Id"}, "alias": {"$ref": "#/$defs/Alias"},
                "level": {"$ref": "#/$defs/Level"}, "one": {"const": 1}, "mixed": {"enum": ["a", 2, null]},
                "either": {"anyOf": [{"$ref": "#/$defs/Id"}, {"type": "null"}]}
            }
        }});
        let s = schemas(serde_json::json!([tool.clone()]));
        let raw = "f<arg_key>id</arg_key><arg_value>3</arg_value><arg_key>alias</arg_key><arg_value>true</arg_value>\
                   <arg_key>level</arg_key><arg_value>2</arg_value><arg_key>one</arg_key><arg_value>1</arg_value>\
                   <arg_key>mixed</arg_key><arg_value>a</arg_value><arg_key>either</arg_key><arg_value>null</arg_value>";
        assert_eq!(
            args_with(raw, Some(&s)).1,
            serde_json::json!({"id": "3", "alias": "true", "level": 2, "one": 1, "mixed": "a", "either": null})
        );

        let tag: serde_json::Value = serde_json::from_str(
            &ToolEnvelope::Glm.structural_tag(&[tool.to_string()], "<tool_call>", "</tool_call>", true).unwrap(),
        )
        .unwrap();
        let el = tag["format"]["tags"][0]["content"]["elements"].as_array().unwrap().clone();
        let value = |i: usize| el[i]["content"]["elements"][1]["content"].clone();
        assert_eq!(value(1)["type"], "any_text", "id: a string through its ref");
        assert_eq!(value(2)["type"], "any_text", "alias: a ref to a ref to a string");
        assert_eq!(
            value(3),
            serde_json::json!({"type": "or", "elements": [
                {"type": "const_string", "value": "1"}, {"type": "const_string", "value": "2"}, {"type": "const_string", "value": "3"}
            ]})
        );
        assert_eq!(value(4), serde_json::json!({"type": "const_string", "value": "1"}));
        assert_eq!(
            value(5),
            serde_json::json!({"type": "or", "elements": [
                {"type": "const_string", "value": "a"}, {"type": "const_string", "value": "2"}, {"type": "const_string", "value": "null"}
            ]})
        );
        assert_eq!(python_json(&serde_json::json!({"k": [1, "é"], "n": null})), "{\"k\": [1, \"é\"], \"n\": null}");
    }

    #[test]
    fn a_root_level_ref_names_the_argument_object() {
        let tool = serde_json::json!({"name": "f", "parameters": {
            "$ref": "#/$defs/Args",
            "$defs": {
                "Args": {"type": "object", "properties": {"id": {"$ref": "#/$defs/Id"}, "n": {"type": "integer"}}, "required": ["id"]},
                "Id": {"type": "string"}
            }
        }});
        let s = schemas(serde_json::json!([tool.clone()]));
        assert_eq!(parse_glm_tool_call("f", Some(&s)), None, "`id` is required");
        assert_eq!(
            args_with("f<arg_key>id</arg_key><arg_value>3</arg_value><arg_key>n</arg_key><arg_value>4</arg_value>", Some(&s)).1,
            serde_json::json!({"id": "3", "n": 4})
        );
        let tag: serde_json::Value = serde_json::from_str(
            &ToolEnvelope::Glm.structural_tag(&[tool.to_string()], "<tool_call>", "</tool_call>", true).unwrap(),
        )
        .unwrap();
        let el = tag["format"]["tags"][0]["content"]["elements"].as_array().unwrap().clone();
        assert_eq!(el[1]["elements"][0]["value"], "<arg_key>id</arg_key>", "required, so not optional");
        assert_eq!(el[1]["elements"][1]["content"]["type"], "any_text");
        assert_eq!(el[2]["type"], "optional");
        assert_eq!(el[2]["content"]["elements"][1]["content"]["json_schema"]["type"], "integer");
    }

    #[test]
    fn ref_siblings_composed_sets_and_padded_nullable_strings() {
        let tool = serde_json::json!({"name": "f", "parameters": {
            "$ref": "#/$defs/Args",
            "required": ["extra"],
            "properties": {"extra": {"type": "string"}},
            "$defs": {
                "Args": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
                "Mode": {"type": "string"}
            }
        }});
        let s = schemas(serde_json::json!([tool.clone()]));
        assert_eq!(
            parse_glm_tool_call("f<arg_key>id</arg_key><arg_value>1</arg_value>", Some(&s)),
            None,
            "`extra`, required beside the $ref, is missing"
        );
        assert!(parse_glm_tool_call(
            "f<arg_key>id</arg_key><arg_value>1</arg_value><arg_key>extra</arg_key><arg_value>x</arg_value>",
            Some(&s)
        )
        .is_some());
        let tag: serde_json::Value = serde_json::from_str(
            &ToolEnvelope::Glm.structural_tag(&[tool.to_string()], "<tool_call>", "</tool_call>", true).unwrap(),
        )
        .unwrap();
        let keys: Vec<String> = tag["format"]["tags"][0]["content"]["elements"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|e| e["elements"][0]["value"].as_str().map(String::from))
            .collect();
        assert_eq!(keys, ["<arg_key>id</arg_key>", "<arg_key>extra</arg_key>"], "both required");

        let tool = serde_json::json!({"name": "g", "parameters": {"type": "object",
            "$defs": {"Mode": {"type": "string"}},
            "properties": {
                "mode": {"$ref": "#/$defs/Mode", "enum": ["r", "w"]},
                "pick": {"anyOf": [{"const": "a"}, {"oneOf": [{"const": "b"}, {"enum": [1]}]}]},
                "open": {"anyOf": [{"const": "a"}, {"type": "string"}]},
                "maybe": {"type": ["string", "null"]}
            },
            "required": ["mode", "pick", "open", "maybe"]
        }});
        let tag: serde_json::Value = serde_json::from_str(
            &ToolEnvelope::Glm.structural_tag(&[tool.to_string()], "<tool_call>", "</tool_call>", true).unwrap(),
        )
        .unwrap();
        let el = tag["format"]["tags"][0]["content"]["elements"].as_array().unwrap().clone();
        let value = |i: usize| el[i]["elements"][1]["content"].clone();
        let consts = |vs: &[&str]| {
            serde_json::json!({"type": "or", "elements": vs.iter().map(|v| serde_json::json!({"type": "const_string", "value": v})).collect::<Vec<_>>()})
        };
        assert_eq!(value(1), consts(&["r", "w"]));
        assert_eq!(value(2), consts(&["a", "b", "1"]));
        assert_eq!(value(3)["type"], "any_text", "an open alternative keeps it open");

        let s = schemas(serde_json::json!([tool]));
        let read = |v: &str| {
            args_with(
                &format!("g<arg_key>mode</arg_key><arg_value>r</arg_value><arg_key>pick</arg_key><arg_value>a</arg_value>\
                          <arg_key>open</arg_key><arg_value>a</arg_value><arg_key>maybe</arg_key><arg_value>{v}</arg_value>"),
                Some(&s),
            )
            .1["maybe"]
                .clone()
        };
        assert_eq!(read("null"), serde_json::Value::Null, "tojson's null");
        assert_eq!(read(" null "), " null ", "padded: written raw, so the string");
        assert_eq!(read("\nnull"), "\nnull");
    }

    #[test]
    fn real_glm_5_3_calls_read_exactly() {
        let s = schemas(serde_json::json!([
            {"type": "function", "function": {"name": "write", "parameters": {"type": "object",
                "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filePath", "content"]}}},
            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}, "description": {"type": "string"}},
                "required": ["command", "description"]}}},
            {"type": "function", "function": {"name": "todowrite", "parameters": {"type": "object",
                "properties": {"todos": {"type": "array", "items": {"type": "object"}}}, "required": ["todos"]}}},
            {"type": "function", "function": {"name": "set_port", "parameters": {"type": "object",
                "properties": {"port": {"type": "string"}, "enabled": {"type": "boolean"}}, "required": ["port", "enabled"]}}}
        ]));
        let cases = [
            (
                "set_port<arg_key>port</arg_key><arg_value>8080</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value>",
                serde_json::json!({"port": "8080", "enabled": true}),
            ),
            (
                "write<arg_key>filePath</arg_key><arg_value>/tmp/demo/app.py</arg_value><arg_key>content</arg_key><arg_value>from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n\n@app.route(\"/\")\ndef index():\n    return jsonify(message=\"Hello, World!\")\n\n\nif __name__ == \"__main__\":\n    app.run(host=\"0.0.0.0\", port=5000)\n</arg_value>",
                serde_json::json!({"filePath": "/tmp/demo/app.py", "content": "from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n\n@app.route(\"/\")\ndef index():\n    return jsonify(message=\"Hello, World!\")\n\n\nif __name__ == \"__main__\":\n    app.run(host=\"0.0.0.0\", port=5000)\n"}),
            ),
            (
                "todowrite<arg_key>todos</arg_key><arg_value>[{\"content\": \"Read and analyze the RFI PDF (requirements, deadlines, submission format)\", \"status\": \"in_progress\", \"priority\": \"high\"}, {\"content\": \"Extract key questions and map them to our capabilities/evidence\", \"status\": \"pending\", \"priority\": \"high\"}]</arg_value>",
                serde_json::json!({"todos": [
                    {"content": "Read and analyze the RFI PDF (requirements, deadlines, submission format)", "status": "in_progress", "priority": "high"},
                    {"content": "Extract key questions and map them to our capabilities/evidence", "status": "pending", "priority": "high"}
                ]}),
            ),
            (
                "bash<arg_key>command</arg_key><arg_value>ls -lh ./inputs</arg_value><arg_key>description</arg_key><arg_value>List files in ./inputs with sizes</arg_value>",
                serde_json::json!({"command": "ls -lh ./inputs", "description": "List files in ./inputs with sizes"}),
            ),
        ];
        for (body, want) in cases {
            assert_eq!(args_with(body, Some(&s)).1, want, "{body}");
        }
    }

    #[test]
    fn required_arguments_must_be_present_when_the_schema_is_known() {
        let s = schemas(serde_json::json!([
            {"name": "write_file", "parameters": {"type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "mode": {"type": "string"}},
                "required": ["path", "content"]}},
            {"name": "list_files", "parameters": {"type": "object", "properties": {"dir": {"type": "string"}}}}
        ]));
        for cut in [
            "<tool_call>write_file",
            "write_file<arg_key>path</arg_key><arg_value>/a</arg_value>",
            "write_file<arg_key>content</arg_key><arg_value>x</arg_value><arg_key>mode</arg_key><arg_value>w</arg_value>",
        ] {
            assert_eq!(parse_glm_tool_call(cut, Some(&s)), None, "{cut:?}");
        }
        assert!(parse_glm_tool_call(
            "write_file<arg_key>path</arg_key><arg_value>/a</arg_value><arg_key>content</arg_key><arg_value></arg_value>",
            Some(&s)
        )
        .is_some());
        assert_eq!(parse_glm_tool_call("list_files", Some(&s)), Some(("list_files".into(), "{}".into())));
        assert!(parse_glm_tool_call("write_file", None).is_some());
    }

    #[test]
    fn no_arguments_and_bad_names() {
        assert_eq!(parse_glm_tool_call("<tool_call>ping</tool_call>", None), Some(("ping".into(), "{}".into())));
        assert_eq!(parse_glm_tool_call("<tool_call>\nping\n</tool_call>", None), Some(("ping".into(), "{}".into())));
        assert_eq!(parse_glm_tool_call("{\"name\": \"f\"}", None), None, "JSON is not a GLM body");
        assert_eq!(parse_glm_tool_call("  ", None), None);
        assert_eq!(parse_glm_tool_call("<tool_call>I cannot call that</tool_call>", None), None);
        assert_eq!(
            parse_glm_tool_call("I will call f<arg_key>a</arg_key><arg_value>1</arg_value>", None),
            None
        );
        assert_eq!(
            parse_glm_tool_call("mcp__fs.read-file<arg_key>a</arg_key><arg_value>1</arg_value>", None).map(|c| c.0),
            Some("mcp__fs.read-file".into())
        );
    }

    #[test]
    fn forced_choice_gets_a_structural_tag_and_auto_stays_free() {
        let tool = serde_json::json!({"type": "function", "function": {"name": "write_file", "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "mode": {"enum": ["w", "a"]}, "lines": {"type": "integer"}},
            "required": ["path"]
        }}})
        .to_string();
        let e = ToolEnvelope::Glm;
        assert_eq!(e.structural_tag(std::slice::from_ref(&tool), "<tool_call>", "</tool_call>", false), None);
        let tag: serde_json::Value = serde_json::from_str(
            &e.structural_tag(std::slice::from_ref(&tool), "<tool_call>", "</tool_call>", true).expect("forced tag"),
        )
        .unwrap();
        assert_eq!(tag["type"], "structural_tag");
        let f = &tag["format"];
        assert_eq!((f["type"].as_str(), f["at_least_one"].as_bool()), (Some("triggered_tags"), Some(true)));
        assert_eq!(f["stop_after_first"], true);
        assert_eq!(f["triggers"], serde_json::json!(["<tool_call>"]));
        let t = &f["tags"][0];
        assert_eq!((t["begin"].as_str(), t["end"].as_str()), (Some("<tool_call>"), Some("</tool_call>")));
        let el = t["content"]["elements"].as_array().unwrap();
        assert_eq!(el[0], serde_json::json!({"type": "const_string", "value": "write_file"}));
        assert_eq!(el[1]["elements"][0]["value"], "<arg_key>path</arg_key>");
        assert_eq!(el[1]["elements"][1]["begin"], "<arg_value>");
        assert_eq!(el[1]["elements"][1]["content"]["type"], "any_text");
        assert_eq!(el[1]["elements"][1]["end"], "</arg_value>");
        assert_eq!(el[2]["type"], "optional");
        assert_eq!(el[2]["content"]["elements"][1]["content"]["type"], "or");
        assert_eq!(el[3]["content"]["elements"][1]["content"]["type"], "json_schema");
        assert_eq!(el[3]["content"]["elements"][1]["content"]["json_schema"], serde_json::json!({"type": "integer"}));
        assert_eq!(e.call_grammar(&[tool]), None);

        let tool = serde_json::json!({"name": "op", "parameters": {
            "type": "object",
            "$defs": {"P": {"type": "object", "properties": {"x": {"type": "integer"}}}},
            "properties": {"kind": {"type": "string", "const": "move"}, "to": {"type": "array", "items": {"$ref": "#/$defs/P"}}},
            "required": ["kind", "to", "why"]
        }})
        .to_string();
        let tag: serde_json::Value =
            serde_json::from_str(&e.structural_tag(&[tool], "<tool_call>", "</tool_call>", true).unwrap()).unwrap();
        let el = tag["format"]["tags"][0]["content"]["elements"].as_array().unwrap().clone();
        assert_eq!(el[1]["elements"][1]["content"], serde_json::json!({"type": "const_string", "value": "move"}));
        assert_eq!(el[2]["elements"][1]["content"]["json_schema"]["$defs"]["P"]["type"], "object");
        assert_eq!(el[3]["elements"][0]["value"], "<arg_key>why</arg_key>");
        assert_eq!(el[3]["elements"][1]["content"]["type"], "any_text");
    }

    #[test]
    fn glm_override_is_taken_where_the_template_spells_the_glm_wire() {
        struct T(Option<&'static str>);
        impl TextCodec for T {
            fn encode(&self, text: &str) -> Vec<u32> {
                MockCodec.encode(text)
            }
            fn token_bytes(&self, token: u32) -> Vec<u8> {
                MockCodec.token_bytes(token)
            }
            fn chat_template_source(&self) -> Option<String> {
                self.0.map(str::to_string)
            }
        }
        let glm = T(Some(include_str!("../../../tests/fixtures/chat_templates/glm_5_3.jinja")));
        assert_eq!(glm.tool_envelope(), ToolEnvelope::Json, "a wire the learner missed reads as JSON");
        assert_eq!(tool_envelope_override_conflict(&glm, ToolEnvelope::Glm), None);
        assert_eq!(tool_envelope_override_conflict(&glm, ToolEnvelope::Json), None, "a no-op");
        assert!(tool_envelope_override_conflict(&glm, ToolEnvelope::Gemma).is_some());
        let chatml = T(Some("{% for m in messages %}<|im_start|>{{ m.content }}{% endfor %}<tool_call>"));
        assert!(tool_envelope_override_conflict(&chatml, ToolEnvelope::Glm).is_some());
        assert!(tool_envelope_override_conflict(&T(None), ToolEnvelope::Glm).is_some());
    }
}

#[cfg(all(test, feature = "basert"))]
mod stand_in_tests {
    use super::*;
    use Placement::*;

    fn subs(values: &[&str]) -> Vec<(String, String)> {
        values.iter().enumerate().map(|(i, v)| (stand_in_key(i), v.to_string())).collect()
    }

    fn rebuilt(pieces: &[(String, bool)]) -> String {
        pieces.iter().map(|(t, _)| t.as_str()).collect()
    }

    #[test]
    fn repeated_framing_between_array_items_is_placed_by_the_known_forms() {
        let s = subs(&["a", "b", "c <|im_end|>"]);
        let stand_ins = format!(r#"{{"paths": ["{}", "{}", "{}"]}}"#, s[0].0, s[1].0, s[2].0);
        let real = r#"{"paths": ["a", "b", "c <|im_end|>"]}"#;
        let pieces = split_around_stand_ins(&stand_ins, &s, real, KnownFormsOnly).expect("placed");
        assert_eq!(rebuilt(&pieces), real);
        let content: Vec<&str> = pieces.iter().filter(|(_, c)| *c).map(|(t, _)| t.as_str()).collect();
        assert_eq!(content, ["a", "b", r"c <|im_end|>"]);
        let framing: Vec<&str> = pieces.iter().filter(|(_, c)| !*c).map(|(t, _)| t.as_str()).collect();
        assert_eq!(framing, [r#"{"paths": [""#, r#"", ""#, r#"", ""#, r#""]}"#]);
    }

    #[test]
    fn a_trimmed_message_is_placed_where_the_framing_follows_it() {
        let s = subs(&["abc\n"]);
        let stand_ins = format!("<start_of_turn>user\n{}\n<end_of_turn>", s[0].0);
        let real = "<start_of_turn>user\nabc\n<end_of_turn>";
        let pieces = split_around_stand_ins(&stand_ins, &s, real, KnownFormsOnly).expect("placed");
        assert_eq!(rebuilt(&pieces), real);
        assert_eq!(pieces[1], ("abc".to_string(), true));
        assert_eq!(pieces[2], ("\n<end_of_turn>".to_string(), false));
    }

    #[test]
    fn unknown_forms_fall_back_to_a_unique_search_or_none() {
        let s = subs(&["x", "y"]);
        let stand_ins = format!("<a>{}</a><b>{}</b>", s[0].0, s[1].0);
        let pieces = split_around_stand_ins(&stand_ins, &s, "<a>X!</a><b>Y!</b>", SearchFraming).expect("placed");
        assert_eq!(pieces[1], ("X!".to_string(), true));
        assert_eq!(pieces[3], ("Y!".to_string(), true));
        assert!(split_around_stand_ins(&stand_ins, &s, "<a>X!</a><b>Y!</b>", KnownFormsOnly).is_none());
        assert!(split_around_stand_ins(&stand_ins, &s, "<a>X</a><b></a><b>Y</b>", SearchFraming).is_none());
        assert!(split_around_stand_ins("<a></a>", &s, "<a>x</a>", SearchFraming).is_none());
    }

    #[test]
    fn framing_a_template_emits_only_for_the_real_text_is_never_content() {
        let s = subs(&["<tool_response>r</tool_response>"]);
        let stand_ins = format!("<|im_start|>user\n{}<|im_end|>", s[0].0);
        let real = "<|im_start|>user\n<wrapped><tool_response>r</tool_response></wrapped><|im_end|>";
        assert!(split_around_stand_ins(&stand_ins, &s, real, KnownFormsOnly).is_none());
        let pieces = split_around_stand_ins(&stand_ins, &s, real, SearchFraming).expect("placed");
        assert_eq!(pieces[1].0, "<wrapped><tool_response>r</tool_response></wrapped>");
    }

    #[test]
    fn known_forms_cover_tojson_and_trim() {
        let always = |_: &str| true;
        let original = " <x> & 'y'\n";
        assert_eq!(place_known_form(" <x> & 'y'\n|", original, always), Some(original.len()));
        assert_eq!(place_known_form("<x> & 'y'|", original, always), Some("<x> & 'y'".len()));
        let html_safe = r" <x> & 'y'\n";
        assert_eq!(place_known_form(&format!("{html_safe}|"), original, always), Some(html_safe.len()));
        assert_eq!(place_known_form(" <x> & 'y'\\n|", original, always), Some(" <x> & 'y'\\n".len()));
        assert_eq!(place_known_form("other", "plain", always), None);
        assert_eq!(place_known_form("abc\n<e>", "abc\n", |after: &str| after == "\n<e>"), Some(3));
    }
}

#[derive(Debug, Clone, Copy)]
enum Piece<'a> {
    Framing(&'a str),
    Content(&'a str),
}

impl<'a> Piece<'a> {
    fn as_pair(&self) -> (&'a str, bool) {
        match *self {
            Piece::Framing(t) => (t, false),
            Piece::Content(t) => (t, true),
        }
    }
}

fn pieces_of(owned: &[(String, bool)]) -> Vec<Piece<'_>> {
    owned
        .iter()
        .map(|(s, content)| {
            if *content {
                Piece::Content(s)
            } else {
                Piece::Framing(s)
            }
        })
        .collect()
}

pub struct ChatMlCodec {
    inner: BundleCodec,
    im_start: u32,
    im_end: u32,
    newline: u32,
    think_open: Option<u32>,
    think_close: Option<u32>,
    tool_open: u32,
    tool_close: u32,
    think: ThinkSuffixes,
    template: Option<Box<crate::template_codec::TemplateCodec>>,
    derive_tool_wire: bool,
    thinking_switch: bool,
    /// What the template puts between an assistant turn's role line and its
    /// content: in history (a user query follows it), and where none does.
    assistant_openers: std::sync::OnceLock<AssistantOpeners>,
}

/// See `ChatMlCodec::assistant_openers`. Qwen3.5 opens only a turn no
/// query follows with its (empty) reasoning block; Qwen3.8 keeps the block
/// on every turn unless told not to preserve it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct AssistantOpeners {
    in_history: String,
    last: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ThinkSuffixes {
    default: Vec<u32>,
    on: Vec<u32>,
    off: Vec<u32>,
}

impl ChatMlCodec {
    #[cfg(feature = "basert")]
    pub fn load(model_path: &std::path::Path) -> Result<ChatMlCodec, Status> {
        let tok = superfluid_engine_ffi::TokenizerHandle::load(model_path).map_err(|_| Status::Fatal)?;
        Self::from_tokenizer(std::sync::Arc::new(tok))
    }

    pub fn from_tokenizer(
        tok: std::sync::Arc<dyn superfluid_engine::Tokenizer>,
    ) -> Result<ChatMlCodec, Status> {
        let inner = BundleCodec::from_tokenizer(std::sync::Arc::clone(&tok));
        let one = |s: &str| -> Result<u32, Status> {
            let t = inner.encode(s);
            if t.len() == 1 {
                Ok(t[0])
            } else {
                Err(Status::Unsupported)
            }
        };
        let tmpl = crate::template_codec::TemplateCodec::from_tokenizer(tok).ok();
        let (think_open, think_close) = match (one("<think>"), one("</think>")) {
            (Ok(o), Ok(c)) => (Some(o), Some(c)),
            _ => (None, None),
        };
        let mut codec = ChatMlCodec {
            im_start: one("<|im_start|>")?,
            im_end: one("<|im_end|>")?,
            newline: one("\n")?,
            think_open,
            think_close,
            tool_open: one("<tool_call>")?,
            tool_close: one("</tool_call>")?,
            inner,
            think: ThinkSuffixes { default: Vec::new(), on: Vec::new(), off: Vec::new() },
            derive_tool_wire: tmpl.as_ref().is_some_and(|t| !t.tool_body_is_json()),
            template: tmpl.map(Box::new),
            thinking_switch: false,
            assistant_openers: std::sync::OnceLock::new(),
        };
        codec.think = codec.derive_think_suffixes().unwrap_or_else(|| codec.assumed_think_suffixes());
        codec.thinking_switch = thinking_switch_changes(|msgs, tools, on| {
            let mut kw = serde_json::Map::new();
            kw.insert("enable_thinking".into(), serde_json::Value::Bool(on));
            codec.render_prompt_structured_with(msgs, tools, &kw)
        });
        Ok(codec)
    }

    pub fn with_thinking(mut self, enabled: bool) -> ChatMlCodec {
        self.think.default = if enabled { self.think.on.clone() } else { self.think.off.clone() };
        self
    }

    fn derive_think_suffixes(&self) -> Option<ThinkSuffixes> {
        let t = self.template.as_deref()?;
        let (open, close) = (self.think_open?, self.think_close?);
        let mut opener = vec![self.im_start];
        opener.extend(self.inner.encode("assistant\n"));
        let probe = [ChatMessage::new(crate::wal::role::USER, "probe".to_string())];
        let suffix = |thinking: Option<bool>| -> Option<Vec<u32>> {
            let mut kw = serde_json::Map::new();
            if let Some(on) = thinking {
                kw.insert("enable_thinking".to_string(), serde_json::Value::Bool(on));
            }
            let with = t.render_prompt_structured_with(&probe, &[], &kw)?;
            let without = t.render_turns_structured_with(&probe, &[], &kw)?;
            let s = with.strip_prefix(without.as_slice())?.strip_prefix(opener.as_slice())?;
            let markup_only = s.iter().all(|&tok| {
                tok == open
                    || tok == close
                    || self.inner.token_bytes(tok).iter().all(u8::is_ascii_whitespace)
            });
            markup_only.then(|| s.to_vec())
        };
        Some(ThinkSuffixes { default: suffix(None)?, on: suffix(Some(true))?, off: suffix(Some(false))? })
    }

    fn reopen_for(&self, mut prompt: Vec<u32>, kwargs: &serde_json::Map<String, serde_json::Value>) -> Vec<u32> {
        let Some(thinking) = kwargs.get("enable_thinking").and_then(|v| v.as_bool()) else {
            return prompt;
        };
        let unset = self.generation_prefix_with(TurnState::AfterInput, None);
        let want = self.generation_prefix_with(TurnState::AfterInput, Some(thinking));
        if let (Some(unset), Some(want)) = (unset, want) {
            if prompt.ends_with(&unset) {
                prompt.truncate(prompt.len() - unset.len());
                prompt.extend(want);
            }
        }
        prompt
    }

    fn assumed_think_suffixes(&self) -> ThinkSuffixes {
        let off = if self.think_open.is_some() {
            self.inner.encode("<think>\n\n</think>\n\n")
        } else {
            Vec::new()
        };
        ThinkSuffixes { default: Vec::new(), on: Vec::new(), off }
    }

    fn role_str(role: u32) -> &'static str {
        match role {
            crate::wal::role::SYSTEM => "system",
            crate::wal::role::ASSISTANT => "assistant",
            _ => "user",
        }
    }

    fn inserted_text(
        t: &crate::template_codec::TemplateCodec,
        with: &[u32],
        without: &[u32],
    ) -> Option<String> {
        let pre = with
            .iter()
            .zip(without.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let suf = with[pre..]
            .iter()
            .rev()
            .zip(without[pre..].iter().rev())
            .take_while(|(a, b)| a == b)
            .count();
        if with.len() <= pre + suf {
            return None;
        }
        let text = t.decode(&with[pre..with.len() - suf]);
        let trimmed = text.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }

    fn tool_source(&self) -> Option<&crate::template_codec::TemplateCodec> {
        if self.derive_tool_wire {
            self.template.as_deref()
        } else {
            None
        }
    }

    fn turn(&self, role: u32, pieces: &[Piece<'_>]) -> Vec<u32> {
        let role_line = format!("{}\n", Self::role_str(role));
        let mut body: Vec<(&str, bool)> = Vec::with_capacity(pieces.len() + 1);
        body.push((&role_line, false));
        body.extend(pieces.iter().map(Piece::as_pair));
        let mut span = vec![self.im_start];
        span.extend(self.inner.encode_pieces(&body));
        span.push(self.im_end);
        span.push(self.newline);
        span
    }

    /// The template's text between an assistant turn's role line and its
    /// content, for a turn a later user query puts in history and for one
    /// that ends the conversation. Empty where there is no template, or the
    /// template carries the content somewhere else.
    fn derived_assistant_openers(&self) -> AssistantOpeners {
        let Some(t) = self.template.as_deref() else { return AssistantOpeners::default() };
        let before_content = |msgs: &[ChatMessage]| -> Option<String> {
            let text = self.inner.decode(&t.render_turns_structured(msgs, &[])?);
            let at = text.find(CONTENT_STAND_IN)?;
            let head = &text[..at];
            let role = format!("{}\n", Self::role_str(crate::wal::role::ASSISTANT));
            Some(head[head.rfind(&role)? + role.len()..].to_string())
        };
        let user = ChatMessage::new(crate::wal::role::USER, "x");
        let turn = ChatMessage::new(crate::wal::role::ASSISTANT, CONTENT_STAND_IN);
        AssistantOpeners {
            in_history: before_content(&[user.clone(), turn.clone(), ChatMessage::new(crate::wal::role::USER, "w")])
                .unwrap_or_default(),
            last: before_content(&[user, turn]).unwrap_or_default(),
        }
    }

    fn assistant_openers(&self) -> &AssistantOpeners {
        self.assistant_openers.get_or_init(|| self.derived_assistant_openers())
    }

    /// An assistant turn with `opener` after its role line; the turn as it is
    /// where the opener is empty or the turn does not begin as one.
    fn opened(&self, turn: Vec<u32>, opener: &str) -> Vec<u32> {
        if opener.is_empty() {
            return turn;
        }
        let role_line = self.inner.encode_pieces(&[(&format!("{}\n", Self::role_str(crate::wal::role::ASSISTANT)), false)]);
        let head = 1 + role_line.len();
        if turn.len() < head || turn[0] != self.im_start || turn[1..head] != role_line[..] {
            return turn;
        }
        let mut span = turn[..head].to_vec();
        span.extend(self.inner.encode_pieces(&[(opener, false)]));
        span.extend_from_slice(&turn[head..]);
        span
    }

    fn frames_like(&self, t: &crate::template_codec::TemplateCodec) -> bool {
        let probe = [
            ChatMessage::new(crate::wal::role::USER, "x"),
            ChatMessage::new(crate::wal::role::ASSISTANT, "y"),
        ];
        match t.render_turns_structured(&probe, &[]) {
            None => false,
            Some(toks) => toks.contains(&self.im_start) && toks.contains(&self.im_end),
        }
    }

    fn derived_system_body(
        &self,
        t: &crate::template_codec::TemplateCodec,
        system: Option<&str>,
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<(String, bool)>> {
        let body_for = |sys: &str| -> Option<String> {
            let msgs = [
                ChatMessage::new(crate::wal::role::SYSTEM, sys),
                ChatMessage::new(crate::wal::role::USER, "x"),
            ];
            let toks = t.render_turns_structured_with(&msgs, tools, kwargs)?;
            let open = toks.iter().position(|&x| x == self.im_start)?;
            let end = toks[open..].iter().position(|&x| x == self.im_end)? + open;
            let body = &toks[open + 1..end];
            let text = self.inner.decode(body);
            let role = format!("{}\n", Self::role_str(crate::wal::role::SYSTEM));
            Some(text.strip_prefix(&role).unwrap_or(&text).to_string())
        };
        let non_empty = |pieces: Vec<(String, bool)>| -> Option<Vec<(String, bool)>> {
            let joined: String = pieces.iter().map(|(s, _)| s.as_str()).collect();
            (!joined.trim().is_empty()).then_some(pieces)
        };
        let Some(sys) = system.filter(|s| !s.is_empty()) else {
            return non_empty(vec![(body_for("")?, false)]);
        };
        let reference = body_for(sys)?;
        let body = body_for(CONTENT_STAND_IN)?;
        let placed = split_once_exact(&body, CONTENT_STAND_IN).and_then(|(before, after)| {
            [sys, sys.trim()]
                .into_iter()
                .find(|used| {
                    reference.len() == before.len() + used.len() + after.len()
                        && reference.starts_with(before)
                        && reference.ends_with(after)
                        && &reference[before.len()..before.len() + used.len()] == *used
                })
                .map(|used| {
                    vec![
                        (before.to_string(), false),
                        (used.to_string(), true),
                        (after.to_string(), false),
                    ]
                })
        });
        match placed {
            Some(pieces) => non_empty(pieces),
            None => {
                tracing::debug!(
                    "chat template does not carry the system text verbatim; \
                     rendering it as framing"
                );
                non_empty(vec![(reference, false)])
            }
        }
    }

    fn derived_tool_instructions(
        t: &crate::template_codec::TemplateCodec,
        tool_jsons: &[String],
    ) -> Option<String> {
        let msgs = [
            ChatMessage::new(crate::wal::role::SYSTEM, ""),
            ChatMessage::new(crate::wal::role::USER, "x"),
        ];
        let with = t.render_turns_structured(&msgs, tool_jsons)?;
        let without = t.render_turns_structured(&msgs, &[])?;
        Self::inserted_text(t, &with, &without)
    }

    fn render_without_calls(t: &crate::template_codec::TemplateCodec, content: &str) -> Option<Vec<u32>> {
        let user = ChatMessage::new(crate::wal::role::USER, "x");
        let plain = ChatMessage::new(crate::wal::role::ASSISTANT, content);
        t.render_turns_structured(&[user, plain], &[])
    }

    fn derived_call_block_against(
        t: &crate::template_codec::TemplateCodec,
        content: &str,
        calls: &[(String, String)],
        without: &[u32],
    ) -> Option<String> {
        let user = ChatMessage::new(crate::wal::role::USER, "x");
        let mut calling = ChatMessage::new(crate::wal::role::ASSISTANT, content);
        for (name, args) in calls {
            calling.tool_calls.push(ToolCallMsg {
                id: String::new(),
                name: name.clone(),
                arguments: args.clone(),
            });
        }
        let with = t.render_turns_structured(&[user, calling], &[])?;
        Self::inserted_text(t, &with, without)
    }

    fn derived_call_pieces(
        t: &crate::template_codec::TemplateCodec,
        content: &str,
        calls: &[(String, String)],
    ) -> Option<Vec<(String, bool)>> {
        let without = Self::render_without_calls(t, content)?;
        let block = Self::derived_call_block_against(t, content, calls, &without)?;
        let mut subs: Vec<(String, String)> = Vec::new();
        let stand_in_calls: Vec<(String, String)> = calls
            .iter()
            .map(|(name, args)| {
                let args = match serde_json::from_str::<serde_json::Value>(args) {
                    Ok(mut v) => {
                        stand_in_strings(&mut v, &mut subs);
                        v.to_string()
                    }
                    Err(_) => args.clone(),
                };
                (name.clone(), args)
            })
            .collect();
        if subs.is_empty() {
            return Some(vec![(block, false)]);
        }
        let with_stand_ins = Self::derived_call_block_against(t, content, &stand_in_calls, &without)?;
        Some(split_around_stand_ins(&with_stand_ins, &subs, &block, Placement::SearchFraming).unwrap_or_else(|| {
            tracing::debug!(
                "chat template does not carry the call arguments verbatim; \
                 the call block is rendered as framing"
            );
            vec![(block, false)]
        }))
    }
}

impl TextCodec for ChatMlCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        self.inner.token_bytes(token)
    }
    fn chat_template_source(&self) -> Option<String> {
        self.inner.chat_template_source()
    }
    fn behavior_fingerprint(&self) -> Option<superfluid_fingerprint::BehaviorFingerprint> {
        let base = if self.think.default.is_empty() {
            "chatml"
        } else if self.think.default == self.think.off {
            "chatml/nothink"
        } else {
            "chatml/think"
        };
        let id = match &self.template {
            Some(t) => format!("{base}/tmpl:{}", t.render_identity()),
            None => base.to_string(),
        };
        Some(self.inner.fingerprint(&id, &self.channel_markers()))
    }
    fn encode_content(&self, text: &str) -> Vec<u32> {
        self.inner.encode_content(text)
    }
    fn encode_pieces(&self, pieces: &[(&str, bool)]) -> Vec<u32> {
        self.inner.encode_pieces(pieces)
    }
    fn render_message(&self, role: u32, text: &str) -> Option<Vec<u32>> {
        let turn = self.turn(role, &[Piece::Content(text)]);
        if role == crate::wal::role::ASSISTANT {
            return Some(self.opened(turn, &self.assistant_openers().in_history));
        }
        Some(turn)
    }
    fn render_turn_with_tokens(
        &self,
        role: u32,
        pre_text: &str,
        inner: &[u32],
        post_text: &str,
    ) -> Option<Vec<u32>> {
        let role_line = format!("{}\n", Self::role_str(role));
        let mut span = vec![self.im_start];
        span.extend(self.inner.encode_pieces(&[(&role_line, false), (pre_text, true)]));
        span.extend_from_slice(inner);
        if !post_text.is_empty() {
            span.extend(self.inner.encode_content(post_text));
        }
        span.push(self.im_end);
        span.push(self.newline);
        Some(span)
    }
    fn render_fim(&self, prefix: &str, suffix: &str, mode: u8) -> Option<Vec<u32>> {
        self.inner.render_fim(prefix, suffix, mode)
    }
    fn empty_think_tokens(&self) -> Option<Vec<u32>> {
        self.think_open?;
        Some(self.inner.encode("<think>\n\n</think>\n\n"))
    }
    fn generation_prefix(&self, state: TurnState) -> Option<Vec<u32>> {
        self.generation_prefix_with(state, None)
    }
    fn generation_prefix_with(&self, state: TurnState, thinking: Option<bool>) -> Option<Vec<u32>> {
        let suffix = match thinking {
            None => &self.think.default,
            Some(true) => &self.think.on,
            Some(false) => &self.think.off,
        };
        let opener = || {
            let mut s = vec![self.im_start];
            s.extend(self.inner.encode("assistant\n"));
            s.extend_from_slice(suffix);
            s
        };
        match state {
            TurnState::AfterInput => Some(opener()),
            TurnState::MidTurn => None,
            TurnState::AfterFinishedTurn => {
                let mut s = vec![self.newline];
                s.extend(opener());
                Some(s)
            }
        }
    }
    fn channel_markers(&self) -> Vec<(u32, u32, u32)> {
        let mut markers = Vec::new();
        if let (Some(o), Some(c)) = (self.think_open, self.think_close) {
            markers.push((o, c, crate::wal::channel::REASONING));
        }
        markers.push((
            self.tool_open,
            self.tool_close,
            crate::wal::channel::TOOL_CALL,
        ));
        markers
    }

    fn turn_separator(&self) -> Option<Vec<u32>> {
        Some(vec![self.newline])
    }

    fn render_system_with_tools(
        &self,
        system: Option<&str>,
        tool_jsons: &[String],
    ) -> Option<Vec<u32>> {
        let derived = self
            .tool_source()
            .and_then(|t| Self::derived_tool_instructions(t, tool_jsons));
        let mut pieces: Vec<Piece<'_>> = Vec::new();
        if let Some(sys) = system {
            pieces.push(Piece::Content(sys));
            pieces.push(Piece::Framing("\n\n"));
        }
        if let Some(tools_text) = derived {
            pieces.push(Piece::Framing(&tools_text));
            return Some(self.turn(crate::wal::role::SYSTEM, &pieces));
        }
        let mut body = String::from(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        );
        for t in tool_jsons {
            body.push('\n');
            body.push_str(t);
        }
        body.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
        );
        pieces.push(Piece::Framing(&body));
        Some(self.turn(crate::wal::role::SYSTEM, &pieces))
    }

    fn tool_envelope(&self) -> ToolEnvelope {
        match self.tool_source() {
            Some(t) => t.tool_envelope(),
            None => ToolEnvelope::Json,
        }
    }

    fn tool_call_delimiters(&self) -> Option<(String, String)> {
        match self.tool_source() {
            Some(t) => t.tool_call_delimiters(),
            None => Some((self.marker_literal(self.tool_open), self.marker_literal(self.tool_close))),
        }
    }
    fn marker_literal(&self, token: u32) -> String {
        self.inner.marker_literal(token)
    }

    fn structural_tag(&self, tool_jsons: &[String], at_least_one: bool) -> Option<String> {
        match self.tool_source() {
            Some(t) => t.structural_tag(tool_jsons, at_least_one),
            None => {
                let (begin, end) = self.tool_call_delimiters()?;
                self.tool_envelope()
                    .structural_tag(tool_jsons, &begin, &end, at_least_one)
            }
        }
    }

    fn call_grammar(&self, tool_jsons: &[String]) -> Option<String> {
        match self.tool_source() {
            Some(t) => t.call_grammar(tool_jsons),
            None => self.tool_envelope().call_grammar(tool_jsons),
        }
    }

    fn named_tool_call(&self, raw: &str) -> Option<(String, String)> {
        self.tool_source()?.named_tool_call(raw)
    }

    fn parse_tool_call_with(&self, raw: &str, schemas: Option<&ToolSchemas>) -> Option<(String, String)> {
        match self.tool_source() {
            Some(t) => t.parse_tool_call_with(raw, schemas),
            None => parse_json_tool_call(raw),
        }
    }

    fn validate_template_kwargs(
        &self,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        match (&self.template, kwargs.is_empty()) {
            (Some(t), false) => t.validate_template_kwargs(kwargs),
            _ => Ok(()),
        }
    }

    fn supports_reasoning_effort(&self) -> bool {
        self.template.as_deref().is_some_and(|t| t.supports_reasoning_effort())
    }

    fn reasoning_effort_levels(&self) -> Vec<&'static str> {
        self.template.as_deref().map(|t| t.reasoning_effort_levels()).unwrap_or_default()
    }

    fn supports_enable_thinking(&self) -> bool {
        self.thinking_switch
    }

    fn render_system_full(
        &self,
        system: Option<&str>,
        tool_jsons: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        let derived = self
            .template
            .as_deref()
            .and_then(|t| self.derived_system_body(t, system, tool_jsons, kwargs));
        match (derived, tool_jsons.is_empty()) {
            (Some(body), _) => Some(self.turn(crate::wal::role::SYSTEM, &pieces_of(&body))),
            (None, true) => match system {
                Some(text) if !text.is_empty() => {
                    self.render_message(crate::wal::role::SYSTEM, text)
                }
                _ => Some(Vec::new()),
            },
            (None, false) => self.render_system_with_tools(system, tool_jsons),
        }
    }

    fn render_prompt_structured_with(
        &self,
        messages: &[ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        let sys_text = messages
            .iter()
            .find(|m| m.role == crate::wal::role::SYSTEM)
            .map(|m| m.content.as_str());
        let derived = self
            .template
            .as_deref()
            .and_then(|t| self.derived_system_body(t, sys_text, tools, kwargs));
        let Some(body) = derived else {
            return self.render_prompt_structured(messages, tools).map(|p| self.reopen_for(p, kwargs));
        };
        let mut out = self.stream_prologue();
        let mut derived_emitted = false;
        for m in messages {
            if !derived_emitted && m.role == crate::wal::role::SYSTEM {
                out.extend(self.turn(crate::wal::role::SYSTEM, &pieces_of(&body)));
                derived_emitted = true;
            } else {
                out.extend(self.render_message(m.role, &m.content)?);
            }
        }
        if !derived_emitted {
            let mut with = self.turn(crate::wal::role::SYSTEM, &pieces_of(&body));
            with.extend(std::mem::take(&mut out));
            out = with;
        }
        if let Some(op) = self.generation_prefix(TurnState::AfterInput) {
            out.extend(op);
        }
        Some(self.reopen_for(out, kwargs))
    }

    fn render_tool_result(&self, _name: &str, content: &str) -> Option<Vec<u32>> {
        Some(self.turn(
            crate::wal::role::USER,
            &[
                Piece::Framing("<tool_response>\n"),
                Piece::Content(content),
                Piece::Framing("\n</tool_response>"),
            ],
        ))
    }

    fn render_assistant_with_tool_calls(
        &self,
        content: &str,
        calls: &[(String, String)],
    ) -> Option<Vec<u32>> {
        let turn = self.assistant_tool_turn(content, calls)?;
        Some(self.opened(turn, &self.assistant_openers().in_history))
    }

    fn render_assistant_with_tool_calls_after_query(
        &self,
        content: &str,
        calls: &[(String, String)],
    ) -> Option<Vec<u32>> {
        let turn = self.assistant_tool_turn(content, calls)?;
        Some(self.opened(turn, &self.assistant_openers().last))
    }
}

impl ChatMlCodec {
    /// An assistant turn with tool calls, nothing between its role line and
    /// its content (the openers go there).
    fn assistant_tool_turn(&self, content: &str, calls: &[(String, String)]) -> Option<Vec<u32>> {
        let derived = self
            .tool_source()
            .and_then(|t| Self::derived_call_pieces(t, content, calls));
        if let Some(block) = derived {
            let sep = if content.is_empty() { "" } else { "\n" };
            let mut pieces = vec![Piece::Content(content), Piece::Framing(sep)];
            pieces.extend(pieces_of(&block));
            return Some(self.turn(crate::wal::role::ASSISTANT, &pieces));
        }
        let mut span = vec![self.im_start];
        span.extend(self.inner.encode_pieces(&[("assistant\n", false), (content, true)]));
        for (i, (name, args)) in calls.iter().enumerate() {
            if (i == 0 && !content.is_empty()) || i > 0 {
                span.push(self.newline);
            }
            span.push(self.tool_open);
            span.extend(self.inner.encode_content(&format!(
                "\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n"
            )));
            span.push(self.tool_close);
        }
        span.push(self.im_end);
        span.push(self.newline);
        Some(span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn switch(template: &str) -> bool {
        let mut env = minijinja::Environment::new();
        env.add_function("raise_exception", |m: String| -> Result<minijinja::Value, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m))
        });
        env.add_template("chat", template).unwrap();
        let t = env.get_template("chat").unwrap();
        thinking_switch_changes(|msgs, tools, on| {
            let messages: Vec<_> = msgs
                .iter()
                .map(|m| {
                    let role = if m.role == crate::wal::role::SYSTEM { "system" } else { "user" };
                    minijinja::context! { role => role, content => m.content.clone() }
                })
                .collect();
            let tools: Vec<serde_json::Value> = tools.iter().map(|t| serde_json::from_str(t).unwrap()).collect();
            t.render(minijinja::context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => true,
                enable_thinking => on,
            })
            .ok()
        })
    }

    const TURNS: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}";

    #[test]
    fn a_template_that_branches_on_enable_thinking_has_the_switch() {
        let qwen = format!(
            "{TURNS}{{% if add_generation_prompt %}}<|im_start|>assistant\n\
             {{% if enable_thinking is defined and enable_thinking is false %}}<think>\n\n</think>\n\n{{% endif %}}{{% endif %}}"
        );
        assert!(switch(&qwen));
        let strict = format!("{{% if not enable_thinking %}}{{{{ raise_exception('thinking is required') }}}}{{% endif %}}{TURNS}");
        assert!(switch(&strict));
        let with_tools = format!("{{% if tools and enable_thinking %}}<think>{{% endif %}}{TURNS}");
        assert!(switch(&with_tools));
        let with_system =
            format!("{{% if messages[0].role == 'system' and enable_thinking %}}<think>{{% endif %}}{TURNS}");
        assert!(switch(&with_system));
    }

    #[test]
    fn a_template_that_only_names_enable_thinking_has_no_switch() {
        for template in [
            format!("{{# enable_thinking is not supported #}}{TURNS}"),
            format!("{TURNS}{{{{ 'enable_thinking' if false }}}}"),
            format!("{{% set enable_thinking = true %}}{{% if enable_thinking %}}{{% endif %}}{TURNS}"),
        ] {
            assert!(!switch(&template), "{template}");
        }
    }

    fn parse(raw: &str) -> Option<(String, serde_json::Value)> {
        let (n, a) = parse_gemma_tool_call(raw)?;
        Some((n, serde_json::from_str(&a).unwrap()))
    }

    #[test]
    fn gemma_call_with_markers_parses() {
        let (name, args) = parse("<|tool_call>call:bash{command:ls -R}<tool_call|>").unwrap();
        assert_eq!(name, "bash");
        assert_eq!(args, serde_json::json!({"command": "ls -R"}));
    }

    #[test]
    fn gemma_call_without_markers_parses() {
        let (name, args) = parse("call:bash{command:ls -R}").unwrap();
        assert_eq!(name, "bash");
        assert_eq!(args, serde_json::json!({"command": "ls -R"}));
    }

    #[test]
    fn gemma_bare_value_may_contain_a_comma() {
        let (_, args) = parse("call:grep{pattern:hello, world, path:notes.txt}").unwrap();
        assert_eq!(args, serde_json::json!({"pattern": "hello, world", "path": "notes.txt"}));
    }

    #[test]
    fn gemma_scalars_keep_their_types() {
        let (_, args) =
            parse("call:f{a:true, b:false, c:5, d:1.5, e:null, s:plain text}").unwrap();
        assert_eq!(
            args,
            serde_json::json!({"a": true, "b": false, "c": 5, "d": 1.5, "e": null, "s": "plain text"})
        );
    }

    #[test]
    fn gemma_quoted_string_form_parses() {
        let (_, args) = parse("call:f{a:<|\"|>x, y} z<|\"|>}").unwrap();
        assert_eq!(args, serde_json::json!({"a": "x, y} z"}));
    }

    #[test]
    fn gemma_nested_object_parses() {
        let (_, args) = parse("call:f{o:{k:v}}").unwrap();
        assert_eq!(args, serde_json::json!({"o": {"k": "v"}}));
    }

    #[test]
    fn gemma_array_elements_split_on_commas() {
        let (_, args) = parse("call:f{l:[1, 2, 3]}").unwrap();
        assert_eq!(args, serde_json::json!({"l": [1, 2, 3]}));
    }

    #[test]
    fn gemma_array_of_strings_keeps_each_element() {
        let (_, args) = parse("call:f{l:[alpha, beta gamma, delta]}").unwrap();
        assert_eq!(args, serde_json::json!({"l": ["alpha", "beta gamma", "delta"]}));
    }

    #[test]
    fn gemma_object_value_may_still_contain_a_comma() {
        let (_, args) = parse("call:f{pattern:hello, world, path:n.txt}").unwrap();
        assert_eq!(args, serde_json::json!({"pattern": "hello, world", "path": "n.txt"}));
    }

    #[test]
    fn gemma_integers_stay_integers() {
        let (_, args) = parse("call:f{n:42}").unwrap();
        assert_eq!(args, serde_json::json!({"n": 42}));
        assert_eq!(serde_json::to_string(&args).unwrap(), r#"{"n":42}"#);
    }

    #[test]
    fn gemma_no_args_is_an_empty_object_not_a_failure() {
        let (name, args) = parse("call:now{}").unwrap();
        assert_eq!(name, "now");
        assert_eq!(args, serde_json::json!({}));
    }

    #[test]
    fn gemma_rejects_what_is_not_a_call() {
        assert!(parse_gemma_tool_call("I will run ls -R for you.").is_none());
        assert!(parse_gemma_tool_call(r#"{"name":"bash","arguments":{}}"#).is_none());
        assert!(parse_gemma_tool_call("call:bash").is_none());
        assert!(parse_gemma_tool_call("call:{command:ls}").is_none());
    }

    #[test]
    fn gemma_truncated_body_still_names_the_call() {
        let (name, args) = parse("<|tool_call>call:bash{command:ls -R").unwrap();
        assert_eq!(name, "bash");
        assert_eq!(args, serde_json::json!({"command": "ls -R"}));
    }

    #[test]
    fn gemma_call_grammar_orders_arguments_as_the_template_renders_them() {
        let params = serde_json::json!({"type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern", "path"]});
        let body = gemma_call_body_format("grep", &params);
        let keys: Vec<String> = body["elements"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|e| e["elements"][0]["value"].as_str().map(String::from))
            .collect();
        assert_eq!(keys, vec!["path:", "pattern:"], "{body}");
    }

    #[test]
    fn gemma_envelope_is_selected_by_name_and_round_trips() {
        let e = ToolEnvelope::from_name("gemma").expect("gemma envelope available");
        assert_eq!(e.name(), "gemma");
        assert!(ToolEnvelope::names().contains(&"gemma"));
        let (name, _) = e.parse_call("<|tool_call>call:bash{command:ls}<tool_call|>", None).unwrap();
        assert_eq!(name, "bash");
        assert!(e.call_grammar(&[r#"{"function":{"name":"bash"}}"#.to_string()]).is_none());
    }

    #[test]
    fn every_structural_tag_is_a_structural_tag_document() {
        let tool = r#"{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}"#.to_string();
                let envelopes = vec![ToolEnvelope::Json, ToolEnvelope::Gemma, ToolEnvelope::Atem];
        for e in envelopes {
            let Some(tag) =
                e.structural_tag(std::slice::from_ref(&tool), "<|tool_call>", "<tool_call|>", true)
            else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&tag).expect("tag is JSON");
            assert_eq!(
                v.get("type").and_then(|t| t.as_str()),
                Some("structural_tag"),
                "{} tag root type: {tag}",
                e.name()
            );
            assert!(
                v.get("format").is_some(),
                "{} tag has no `format` field — xgrammar rejects such a document, and the caller then decodes unconstrained: {tag}",
                e.name()
            );
        }
    }

    #[test]
    fn gemma_structural_tag_frames_the_call_without_json() {
        let tool = r#"{"name":"grep","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"n":{"type":"integer"},"deep":{"type":"boolean"}},"required":["pattern","n","deep"]}}"#.to_string();
        let tag = ToolEnvelope::Gemma
            .structural_tag(&[tool], "<|tool_call>", "<tool_call|>", true)
            .expect("gemma tag");
        let v: serde_json::Value = serde_json::from_str(&tag).unwrap();
        let f = &v["format"];
        assert_eq!(f["type"], "triggered_tags");
        assert_eq!(f["at_least_one"], true, "tool_choice=required must reach the tag");
        assert_eq!(f["stop_after_first"], true, "a forced call is one call");
        assert_eq!(f["triggers"][0], "<|tool_call>");
        let content = &f["tags"][0]["content"];
        assert_eq!(content["type"], "sequence");
        let dump = content.to_string();
        assert!(dump.contains(r#""value":"call:grep{""#), "call prefix: {dump}");
        assert!(!dump.contains("json_schema"), "gemma body must not be JSON: {dump}");
        assert!(dump.contains(r#""value":"n:""#), "integer key framed: {dump}");
        assert!(dump.contains("-?[0-9]+"), "integer value pattern: {dump}");
        assert!(dump.contains(r#""value":"true""#), "boolean rendered as const: {dump}");
    }

    #[test]
    fn a_dialect_without_terminators_is_unaffected() {
        assert!(MockChatCodec.turn_terminators().is_empty());
    }

    #[test]
    fn json_envelope_still_rejects_the_gemma_body() {
        assert!(parse_json_tool_call("call:bash{command:ls -R}").is_none());
    }

    #[test]
    fn utf8_stream_reassembles_split_characters() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(b"caf"), "caf");
        assert_eq!(s.push(&[0xC3]), "");
        assert_eq!(s.push(&[0xA9]), "\u{e9}");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn utf8_stream_replaces_invalid_and_flushes_partial() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(&[0xFF, b'a']), "\u{FFFD}a");
        assert_eq!(s.push(&[0xE2, 0x82]), "");
        assert_eq!(s.flush(), "\u{FFFD}");
    }

    fn shape(runs: &[Run]) -> Vec<(Vec<u32>, u32, bool)> {
        runs.iter()
            .map(|r| (r.span.clone(), r.channel, r.closes))
            .collect()
    }

    #[test]
    fn segmenter_partitions_at_markers_across_slices() {
        let think = (100, 101, crate::wal::channel::REASONING);
        let tool = (200, 201, crate::wal::channel::TOOL_CALL);
        let mut seg = Segmenter::new(vec![think, tool]);
        let runs = seg.split(&[1, 2, 100, 3]);
        assert_eq!(
            shape(&runs),
            vec![
                (vec![1, 2], crate::wal::channel::TEXT, false),
                (vec![100, 3], crate::wal::channel::REASONING, false),
            ]
        );
        let runs = seg.split(&[4, 101, 200, 5, 201, 6]);
        assert_eq!(
            shape(&runs),
            vec![
                (vec![4, 101], crate::wal::channel::REASONING, true),
                (vec![200, 5, 201], crate::wal::channel::TOOL_CALL, true),
                (vec![6], crate::wal::channel::TEXT, false),
            ]
        );
        let input = vec![100, 200, 7, 101, 8];
        let runs = seg.split(&input);
        let concat: Vec<u32> = runs.iter().flat_map(|r| r.span.clone()).collect();
        assert_eq!(concat, input);
        assert_eq!(runs[0].channel, crate::wal::channel::REASONING);
    }

    #[test]
    fn a_named_block_keeps_its_name_out_of_the_text() {
        use crate::wal::channel;
        const OPEN: u32 = 100;
        const CLOSE: u32 = 101;
        let (thought, nl) = (50, 51);
        let named = || Segmenter::new(vec![(OPEN, CLOSE, channel::REASONING)]).with_names(vec![(OPEN, vec![thought, nl])]);
        let texts = |runs: Vec<Run>| -> Vec<(u32, Vec<u32>, bool)> { runs.into_iter().map(|r| (r.channel, r.text, r.closes)).collect() };

        let whole = [OPEN, thought, nl, 7, 8, CLOSE, 9];
        let runs = named().split(&whole);
        assert_eq!(runs.iter().flat_map(|r| r.span.clone()).collect::<Vec<_>>(), whole, "spans partition the stream");
        assert_eq!(texts(runs), vec![(channel::REASONING, vec![7, 8], true), (channel::TEXT, vec![9], false)]);

        let mut seg = named();
        let mut runs = seg.split(&[OPEN, thought]);
        runs.extend(seg.split(&[nl, 7, CLOSE]));
        let reasoning: Vec<u32> = runs.iter().filter(|r| r.channel == channel::REASONING).flat_map(|r| r.text.clone()).collect();
        assert_eq!(reasoning, vec![7]);

        let runs = named().split(&[OPEN, thought, 7, nl, CLOSE]);
        assert_eq!(texts(runs), vec![(channel::REASONING, vec![7, nl], true)]);

        let mut seg = named();
        seg.split(&[OPEN, thought, nl, 7, CLOSE]);
        assert_eq!(texts(seg.split(&[OPEN, thought, nl, 8, CLOSE])), vec![(channel::REASONING, vec![8], true)]);
        let mut primed = named();
        primed.prime(channel::REASONING);
        assert_eq!(texts(primed.split(&[thought, CLOSE])), vec![(channel::REASONING, vec![thought], true)]);
    }

    #[test]
    fn segmenter_without_markers_is_single_channel() {
        let mut seg = Segmenter::new(Vec::new());
        let runs = seg.split(&[1, 2, 3]);
        assert_eq!(
            shape(&runs),
            vec![(vec![1, 2, 3], crate::wal::channel::TEXT, false)]
        );
    }

    #[test]
    fn mock_codec_round_trips_multibyte_text() {
        let c = MockCodec;
        let text = "héllo — ok";
        let toks = c.encode(text);
        assert_eq!(c.decode(&toks), text);
    }
}

#[cfg(test)]
mod learned_tool_wire_tests {
    use super::{learn_delims, S_K1, S_K2, S_NAME, S_V1, S_V2};

    fn render(name_open: &str, name_close: &str, key_open: &str, key_close: &str, val_close: &str, args: &[(&str, &str)]) -> String {
        let mut s = String::from(name_open);
        s.push_str(S_NAME);
        s.push_str(name_close);
        for (k, v) in args {
            s.push_str(key_open);
            s.push_str(k);
            s.push_str(key_close);
            s.push_str(v);
            s.push_str(val_close);
        }
        s
    }

    #[test]
    fn learns_a_qwen_coder_xml_body_and_inverts_it() {
        let frame = ("<tool_call>", "</tool_call>");
        let two = render("<function=", ">\n", "<parameter=", ">\n", "\n</parameter>\n", &[(S_K1, S_V1), (S_K2, S_V2)]) + "</function>";
        let zero = render("<function=", ">\n", "", "", "", &[]) + "</function>";
        let d = learn_delims(&two, Some(&(zero)), Some(frame)).expect("learns");
        let raw = "<function=bash>\n<parameter=command>\nls -la /tmp\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>";
        let (name, args) = d.parse(raw, None).expect("parses");
        assert_eq!(name, "bash");
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["command"], "ls -la /tmp");
        assert_eq!(v["timeout"], 30);
        let cut = "<function=read>\n<parameter=path>\n/a\n</parameter>\n<parameter=limit>\n5";
        let (n2, a2) = d.parse(cut, None).unwrap();
        assert_eq!(n2, "read");
        let v2: serde_json::Value = serde_json::from_str(&a2).unwrap();
        assert_eq!(v2["path"], "/a");
    }

    #[test]
    fn learns_a_zero_arg_call_terminator() {
        let two = render("<function=", ">\n", "<parameter=", ">\n", "\n</parameter>\n", &[(S_K1, S_V1), (S_K2, S_V2)]) + "</function>";
        let zero = render("<function=", ">\n", "", "", "", &[]) + "</function>";
        let d = learn_delims(&two, Some(&zero), Some(("<tool_call>", "</tool_call>"))).unwrap();
        let (name, args) = d.parse("<function=now>\n</function>", None).expect("zero-arg call");
        assert_eq!(name, "now");
        assert_eq!(args, "{}");
    }

    fn qwen_delims() -> super::Delims {
        let two = render("<function=", ">\n", "<parameter=", ">\n", "\n</parameter>\n", &[(S_K1, S_V1), (S_K2, S_V2)]) + "</function>";
        let zero = render("<function=", ">\n", "", "", "", &[]) + "</function>";
        learn_delims(&two, Some(&zero), Some(("<tool_call>", "</tool_call>"))).expect("learns")
    }

    fn serve_schemas() -> super::ToolSchemas {
        super::ToolSchemas::from_values(
            serde_json::json!([{"type": "function", "function": {"name": "serve", "parameters": {
                "type": "object",
                "properties": {
                    "port": {"type": "string"}, "host": {"type": "string"},
                    "workers": {"type": "integer"}, "ratio": {"type": "number"},
                    "debug": {"type": "boolean"}, "verbose": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "env": {"type": "object"},
                    "id": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                    "note": {"type": ["string", "null"]},
                    "limit": {"$ref": "#/$defs/Count"},
                    "mode": {"enum": ["1", "2"]},
                    "loose": {}
                },
                "$defs": {"Count": {"type": "integer"}}
            }}}])
            .as_array()
            .unwrap(),
        )
    }

    fn parsed(d: &super::Delims, raw: &str, schemas: Option<&super::ToolSchemas>) -> serde_json::Value {
        let (name, args) = d.parse(raw, schemas).expect("parses");
        assert_eq!(name, "serve");
        serde_json::from_str(&args).unwrap()
    }

    #[test]
    fn learned_xml_wire_types_values_from_the_declared_schema() {
        let d = qwen_delims();
        let s = serve_schemas();
        let raw = "<function=serve>\n\
            <parameter=port>\n8080\n</parameter>\n\
            <parameter=host>\nnull\n</parameter>\n\
            <parameter=workers>\n4\n</parameter>\n\
            <parameter=ratio>\n0.5\n</parameter>\n\
            <parameter=debug>\ntrue\n</parameter>\n\
            <parameter=verbose>\nFalse\n</parameter>\n\
            <parameter=tags>\n[\"a\", \"b c\"]\n</parameter>\n\
            <parameter=env>\n{\"K\": 1}\n</parameter>\n\
            <parameter=id>\n42\n</parameter>\n\
            <parameter=note>\nnull\n</parameter>\n\
            <parameter=limit>\n10\n</parameter>\n\
            <parameter=mode>\n2\n</parameter>\n\
            <parameter=loose>\n7\n</parameter>\n\
            <parameter=undeclared>\ntrue\n</parameter>\n\
            </function>";
        assert_eq!(
            parsed(&d, raw, Some(&s)),
            serde_json::json!({
                "port": "8080", "host": "null",
                "workers": 4, "ratio": 0.5, "debug": true, "verbose": false,
                "tags": ["a", "b c"], "env": {"K": 1},
                "id": "42", "note": "null",
                "limit": 10, "mode": "2",
                "loose": 7, "undeclared": true
            })
        );
        let odd = parsed(&d, "<function=serve>\n<parameter=workers>\nfour\n</parameter>\n</function>", Some(&s));
        assert_eq!(odd, serde_json::json!({"workers": "four"}));
    }

    #[test]
    fn learned_xml_wire_without_a_schema_keeps_the_heuristic() {
        let d = qwen_delims();
        let raw = "<function=serve>\n<parameter=port>\n8080\n</parameter>\n<parameter=host>\nlocalhost\n</parameter>\n</function>";
        assert_eq!(parsed(&d, raw, None), serde_json::json!({"port": 8080, "host": "localhost"}));
        let other = super::ToolSchemas::from_values(
            serde_json::json!([{"name": "other", "parameters": {"properties": {"port": {"type": "string"}}}}])
                .as_array()
                .unwrap(),
        );
        assert_eq!(parsed(&d, raw, Some(&other)), serde_json::json!({"port": 8080, "host": "localhost"}));
    }

    #[test]
    fn learned_xml_wire_keeps_payload_whitespace() {
        let d = qwen_delims();
        let s = super::ToolSchemas::from_values(
            serde_json::json!([{"name": "write", "parameters": {"properties": {
                "path": {"type": "string"}, "content": {"type": "string"}
            }}}])
            .as_array()
            .unwrap(),
        );
        let raw = "<function=write>\n<parameter=path>\n/tmp/a.py\n</parameter>\n\
                   <parameter=content>\n    def f():\n        return 1\n\n</parameter>\n</function>";
        for schemas in [Some(&s), None] {
            let (_, args) = d.parse(raw, schemas).unwrap();
            let v: serde_json::Value = serde_json::from_str(&args).unwrap();
            assert_eq!(v["path"], "/tmp/a.py");
            assert_eq!(v["content"], "    def f():\n        return 1\n", "schemas={}", schemas.is_some());
        }
    }

    #[test]
    fn closed_args_type_from_the_same_schema() {
        let d = qwen_delims();
        let s = serve_schemas();
        let raw = "<function=serve>\n<parameter=port>\n8080\n</parameter>\n<parameter=workers>\n4\n</parameter>\n<parameter=id>\n9";
        let mut cursor = None;
        let got = d.closed_args(raw, &mut cursor, Some(&s));
        assert_eq!(
            got,
            vec![("port".to_string(), serde_json::json!("8080")), ("workers".to_string(), serde_json::json!(4))]
        );
        let mut cursor = None;
        assert_eq!(d.closed_args(raw, &mut cursor, None)[0].1, serde_json::json!(8080));
    }

    #[cfg(feature = "basert")]
    #[test]
    fn learned_xml_wire_forced_tag_is_one_call_of_free_text_values() {
        let d = qwen_delims();
        let tool = serde_json::json!({"type": "function", "function": {"name": "write", "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}, "mode": {"type": "integer"}},
            "required": ["content"]
        }}})
        .to_string();
        let tag: serde_json::Value = serde_json::from_str(
            &d.structural_tag(std::slice::from_ref(&tool), "<tool_call>", "</tool_call>", true).expect("forced tag"),
        )
        .unwrap();
        let f = &tag["format"];
        assert_eq!(f["at_least_one"], true);
        assert_eq!(f["stop_after_first"], true);
        let el = f["tags"][0]["content"]["elements"].as_array().unwrap();
        assert_eq!(el[1]["value"], "<function=write>\n<");
        let content = &el[2];
        assert_eq!(
            content,
            &serde_json::json!({
                "type": "tag", "begin": "parameter=content>\n",
                "content": {"type": "any_text"}, "end": "\n</parameter>\n<"
            })
        );
        let mode = &el[3]["content"];
        assert_eq!(el[3]["type"], "optional");
        assert_eq!(mode["begin"], "parameter=mode>\n");
        assert_eq!(mode["content"]["type"], "json_schema");
        assert_eq!(mode["content"]["json_schema"]["type"], "integer");
        assert_eq!(el.last().unwrap(), &serde_json::json!({"type": "any_text"}));
    }

    #[test]
    fn bails_when_the_body_has_no_repeating_key_opener() {
        let two = format!("call:{S_NAME}{{{S_K1}:{S_V1},{S_K2}:{S_V2}}}");
        assert!(learn_delims(&two, None, Some(("<|tool_call>", "<tool_call|>"))).is_none());
    }
}
