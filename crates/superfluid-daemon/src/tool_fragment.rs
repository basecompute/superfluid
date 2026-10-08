//! Streaming a tool call while the model is still writing it.

use crate::codec::Delims;

#[derive(Clone)]
pub enum ToolFragmentMode {
    Json,
    Delimited(Delims),
    None,
}

#[derive(Debug, PartialEq)]
pub struct ToolDelta {
    pub opening: bool,
    pub name: String,
    pub arguments: String,
}

pub struct ToolCallStream {
    mode: ToolFragmentMode,
    body: String,
    opened: bool,
    complete: bool,
    sent: String,
    args_start: Option<usize>,
    depth: usize,
    scan_pos: usize,
    scan_in_str: bool,
    scan_esc: bool,
    arg_cursor: Option<usize>,
    sent_keys: Vec<String>,
    schemas: Option<std::sync::Arc<crate::codec::ToolSchemas>>,
}

impl ToolCallStream {
    pub fn new(mode: ToolFragmentMode) -> Self {
        Self {
            mode,
            body: String::new(),
            opened: false,
            complete: false,
            sent: String::new(),
            args_start: None,
            depth: 0,
            scan_pos: 0,
            scan_in_str: false,
            scan_esc: false,
            arg_cursor: None,
            sent_keys: Vec::new(),
            schemas: None,
        }
    }

    pub fn with_schemas(mut self, schemas: Option<std::sync::Arc<crate::codec::ToolSchemas>>) -> Self {
        self.schemas = schemas;
        self
    }

    pub fn opened(&self) -> bool {
        self.opened
    }

    pub fn reset(&mut self) {
        let mode = self.mode.clone();
        let schemas = self.schemas.take();
        *self = Self::new(mode).with_schemas(schemas);
    }

    pub fn feed(&mut self, text: &str) -> Vec<ToolDelta> {
        self.body.push_str(text);
        let mut out = Vec::new();
        if self.complete {
            return out;
        }
        match self.mode.clone() {
            ToolFragmentMode::Json => self.step_json(&mut out),
            ToolFragmentMode::Delimited(d) => self.step_delimited(&d, &mut out),
            ToolFragmentMode::None => {}
        }
        out
    }

    pub fn finish(&mut self, arguments: &str) -> Vec<ToolDelta> {
        let mut out = Vec::new();
        if self.complete {
            return out;
        }
        match self.mode.clone() {
            ToolFragmentMode::Json => self.step_json(&mut out),
            ToolFragmentMode::Delimited(d) => self.step_delimited(&d, &mut out),
            ToolFragmentMode::None => {}
        }
        if !self.opened || self.complete {
            return out;
        }
        if self.sent.is_empty() {
            self.emit(&mut out, arguments.to_string());
        } else {
            match self.mode {
                ToolFragmentMode::Json => {
                    self.emit(&mut out, "}".repeat(self.depth));
                }
                _ => {
                    for (key, value) in missing_members(arguments, &self.sent_keys) {
                        self.emit(
                            &mut out,
                            format!(",{}:{}", serde_json::Value::String(key), value),
                        );
                    }
                    self.emit(&mut out, "}".to_string());
                }
            }
        }
        self.complete = true;
        out
    }

    fn step_json(&mut self, out: &mut Vec<ToolDelta>) {
        if !self.opened {
            match json_member_string(&self.body, "name") {
                Some(name) if !name.is_empty() => self.open(out, name),
                _ => return,
            }
        }
        if self.args_start.is_none() {
            let v = json_member_value_start(&self.body, "arguments")
                .or_else(|| json_member_value_start(&self.body, "parameters"));
            let Some(v) = v else { return };
            let first = self.body.as_bytes()[v];
            if first != b'{' && first != b'[' {
                self.mode = ToolFragmentMode::None;
                return;
            }
            self.args_start = Some(v);
            self.scan_pos = v;
        }
        let start = self.args_start.expect("located above");
        let end = self.scan_json_value();
        let already = start + self.sent.len();
        if end > already {
            let fragment = self.body[already..end].to_string();
            self.emit(out, fragment);
        }
    }

    fn scan_json_value(&mut self) -> usize {
        let b = self.body.as_bytes();
        while self.scan_pos < b.len() {
            let c = b[self.scan_pos];
            self.scan_pos += 1;
            if self.scan_in_str {
                if self.scan_esc {
                    self.scan_esc = false;
                } else if c == b'\\' {
                    self.scan_esc = true;
                } else if c == b'"' {
                    self.scan_in_str = false;
                }
                continue;
            }
            match c {
                b'"' => self.scan_in_str = true,
                b'{' | b'[' => self.depth += 1,
                b'}' | b']' => {
                    self.depth -= 1;
                    if self.depth == 0 {
                        self.complete = true;
                        return self.scan_pos;
                    }
                }
                _ => {}
            }
        }
        if self.scan_esc {
            b.len() - 1
        } else {
            b.len()
        }
    }

    fn step_delimited(&mut self, d: &Delims, out: &mut Vec<ToolDelta>) {
        if !self.opened {
            match d.peek_name(&self.body) {
                Some(name) => self.open(out, name),
                None => return,
            }
        }
        let mut cursor = self.arg_cursor;
        let closed = d.closed_args(&self.body, &mut cursor, self.schemas.as_deref());
        self.arg_cursor = cursor;
        for (key, value) in closed {
            if self.sent_keys.contains(&key) {
                continue;
            }
            let sep = if self.sent_keys.is_empty() { "{" } else { "," };
            let fragment = format!("{sep}{}:{}", serde_json::Value::String(key.clone()), value);
            self.sent_keys.push(key);
            self.emit(out, fragment);
        }
    }

    fn open(&mut self, out: &mut Vec<ToolDelta>, name: String) {
        out.push(ToolDelta {
            opening: true,
            name,
            arguments: String::new(),
        });
        self.opened = true;
    }

    fn emit(&mut self, out: &mut Vec<ToolDelta>, fragment: String) {
        if fragment.is_empty() {
            return;
        }
        self.sent.push_str(&fragment);
        if let Some(last) = out.last_mut() {
            if last.opening && last.arguments.is_empty() {
                last.arguments = fragment;
                return;
            }
        }
        out.push(ToolDelta {
            opening: false,
            name: String::new(),
            arguments: fragment,
        });
    }
}

fn missing_members(arguments: &str, sent: &[String]) -> Vec<(String, serde_json::Value)> {
    let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(arguments)
    else {
        return Vec::new();
    };
    map.into_iter()
        .filter(|(k, _)| !sent.iter().any(|s| s == k))
        .collect()
}

fn json_member_value_start(body: &str, key: &str) -> Option<usize> {
    let quoted = format!("\"{key}\"");
    let b = body.as_bytes();
    let (mut depth, mut in_str, mut esc, mut expect_key) = (0i32, false, false, false);
    let mut found = None;
    for (i, &c) in b.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' => {
                if depth == 1 && expect_key && body[i..].starts_with(&quoted) {
                    found = Some(i);
                    break;
                }
                in_str = true;
            }
            b'{' => {
                depth += 1;
                expect_key = depth == 1;
            }
            b'[' => {
                depth += 1;
                expect_key = false;
            }
            b'}' | b']' => {
                depth -= 1;
                expect_key = false;
            }
            b',' => expect_key = depth == 1,
            b':' => expect_key = false,
            _ => {}
        }
    }
    let k = found?;
    let colon = k + body[k..].find(':')?;
    body[colon + 1..]
        .find(|c: char| !c.is_whitespace())
        .map(|off| colon + 1 + off)
}

fn json_member_string(body: &str, key: &str) -> Option<String> {
    let p = json_member_value_start(body, key)?;
    if *body.as_bytes().get(p)? != b'"' {
        return None;
    }
    let mut esc = false;
    for (off, c) in body[p + 1..].char_indices() {
        if esc {
            esc = false;
        } else if c == '\\' {
            esc = true;
        } else if c == '"' {
            return serde_json::from_str::<String>(&body[p..p + off + 2]).ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(mode: ToolFragmentMode, chunks: &[&str], parsed_args: &str) -> (Vec<String>, String) {
        let mut s = ToolCallStream::new(mode);
        let mut names = Vec::new();
        let mut args = String::new();
        let mut open_seen = false;
        let mut take = |deltas: Vec<ToolDelta>, names: &mut Vec<String>, args: &mut String| {
            for d in deltas {
                if d.opening {
                    open_seen = true;
                    names.push(d.name);
                } else {
                    assert!(open_seen, "an arguments fragment preceded the name");
                }
                args.push_str(&d.arguments);
            }
        };
        for c in chunks {
            take(s.feed(c), &mut names, &mut args);
        }
        take(s.finish(parsed_args), &mut names, &mut args);
        (names, args)
    }

    #[test]
    fn json_streams_raw_argument_bytes_after_the_name() {
        let (names, args) = drive(
            ToolFragmentMode::Json,
            &[
                "{\"name\": \"get_wea",
                "ther\", \"arguments\": {\"city\": \"Tok",
                "yo\", \"unit\": \"c\"}}",
            ],
            "{\"city\":\"Tokyo\",\"unit\":\"c\"}",
        );
        assert_eq!(names, vec!["get_weather"]);
        assert_eq!(args, "{\"city\": \"Tokyo\", \"unit\": \"c\"}");
    }

    #[test]
    fn json_opens_before_any_argument_byte() {
        let mut s = ToolCallStream::new(ToolFragmentMode::Json);
        let d = s.feed("{\"name\": \"ping\", \"argu");
        assert_eq!(d.len(), 1);
        assert!(d[0].opening && d[0].name == "ping" && d[0].arguments.is_empty());
        assert!(s.opened());
    }

    #[test]
    fn json_holds_back_a_partial_escape() {
        let mut s = ToolCallStream::new(ToolFragmentMode::Json);
        let mut sent = String::new();
        for d in s.feed("{\"name\":\"f\",\"arguments\":{\"s\":\"a\\") {
            assert!(!d.arguments.ends_with('\\'), "fragment ended mid-escape");
            sent.push_str(&d.arguments);
        }
        for d in s.feed("n b\"}}") {
            sent.push_str(&d.arguments);
        }
        assert_eq!(sent, "{\"s\":\"a\\n b\"}");
    }

    #[test]
    fn json_closes_a_call_the_parser_salvaged_from_a_cut_body() {
        let (_, args) = drive(
            ToolFragmentMode::Json,
            &["{\"name\":\"f\",\"arguments\":{\"a\":{\"b\":1"],
            "{\"a\":{\"b\":1}}",
        );
        assert_eq!(args, "{\"a\":{\"b\":1}}");
        assert!(serde_json::from_str::<serde_json::Value>(&args).is_ok());
    }

    #[test]
    fn json_argument_less_call_still_gets_an_object() {
        let (names, args) = drive(ToolFragmentMode::Json, &["{\"name\":\"ping\"}"], "{}");
        assert_eq!(names, vec!["ping"]);
        assert_eq!(args, "{}");
    }

    #[test]
    fn json_string_valued_arguments_stay_one_shot() {
        let mut s = ToolCallStream::new(ToolFragmentMode::Json);
        let deltas = s.feed("{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}");
        assert!(s.opened());
        assert!(deltas.iter().all(|d| d.arguments.is_empty()));
        let rest = s.finish("{\"a\":1}");
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].arguments, "{\"a\":1}");
    }

    fn qwen_wire() -> ToolFragmentMode {
        ToolFragmentMode::Delimited(Delims::for_test(
            "<function=",
            ">\n",
            "<parameter=",
            ">\n",
            "\n</parameter>\n",
        ))
    }

    #[test]
    fn delimited_streams_one_closed_argument_at_a_time() {
        let mut s = ToolCallStream::new(qwen_wire());
        let mut args = String::new();
        let d = s.feed("<function=get_weather>\n<parameter=city>\nTok");
        assert_eq!(d.len(), 1);
        assert!(d[0].opening && d[0].name == "get_weather" && d[0].arguments.is_empty());
        for d in s.feed("yo\n</parameter>\n") {
            args.push_str(&d.arguments);
        }
        assert_eq!(
            args, "{\"city\":\"Tokyo\"",
            "a closed value streams, the object stays open"
        );
        for d in s.feed("<parameter=days>\n3\n</parameter>\n") {
            args.push_str(&d.arguments);
        }
        assert_eq!(
            args, "{\"city\":\"Tokyo\",\"days\":3",
            "typed on the way out"
        );
        for d in s.finish("{\"city\":\"Tokyo\",\"days\":3}") {
            args.push_str(&d.arguments);
        }
        assert_eq!(args, "{\"city\":\"Tokyo\",\"days\":3}");
        let v: serde_json::Value =
            serde_json::from_str(&args).expect("fragments reassemble to JSON");
        assert_eq!(v["days"], 3);
    }

    #[test]
    fn delimited_types_streamed_values_from_the_declared_schema() {
        let schemas = crate::codec::ToolSchemas::of_values(&[serde_json::json!({
            "type": "function",
            "function": {"name": "serve", "parameters": {"type": "object", "properties": {
                "port": {"type": "string"}, "workers": {"type": "integer"}
            }}}
        })]);
        let mut s = ToolCallStream::new(qwen_wire()).with_schemas(schemas);
        for round in 0..2 {
            let mut args = String::new();
            for chunk in [
                "<function=serve>\n<parameter=port>\n8080\n</parameter>\n",
                "<parameter=workers>\n4\n</parameter>\n</function>",
            ] {
                for d in s.feed(chunk) {
                    args.push_str(&d.arguments);
                }
            }
            assert_eq!(args, "{\"port\":\"8080\",\"workers\":4", "round {round}");
            s.reset();
        }
        let mut blind = ToolCallStream::new(qwen_wire());
        let args: String = blind
            .feed("<function=serve>\n<parameter=port>\n8080\n</parameter>\n")
            .into_iter()
            .map(|d| d.arguments)
            .collect();
        assert_eq!(args, "{\"port\":8080");
    }

    #[test]
    fn delimited_zero_argument_call_gets_an_object() {
        let mut s = ToolCallStream::new(qwen_wire());
        let _ = s.feed("<function=ping>\n");
        let rest = s.finish("{}");
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].arguments, "{}");
    }

    #[test]
    fn delimited_hands_over_a_value_only_the_final_parse_recovered() {
        let mut s = ToolCallStream::new(qwen_wire());
        let mut args = String::new();
        for chunk in [
            "<function=f>\n<parameter=a>\n1\n</parameter>\n",
            "<parameter=b>\n22",
        ] {
            for d in s.feed(chunk) {
                args.push_str(&d.arguments);
            }
        }
        assert_eq!(args, "{\"a\":1");
        for d in s.finish("{\"a\":1,\"b\":22}") {
            args.push_str(&d.arguments);
        }
        assert_eq!(
            args, "{\"a\":1,\"b\":22}",
            "the value only the final parse recovered is handed over at the close"
        );
        assert!(serde_json::from_str::<serde_json::Value>(&args).is_ok());
    }

    #[test]
    fn json_name_is_decoded_by_the_json_parser() {
        let mut s = ToolCallStream::new(ToolFragmentMode::Json);
        let d = s.feed(r#"{"name":"get\u005fweather","arguments":{}}"#);
        assert_eq!(d[0].name, "get_weather");
    }

    #[test]
    fn json_name_comes_from_a_key_not_a_value() {
        let mut s = ToolCallStream::new(ToolFragmentMode::Json);
        let d = s.feed(r#"{"note":"name","alias":"wrong","name":"right","arguments":{}}"#);
        assert_eq!(d[0].name, "right");
    }

    #[test]
    fn none_mode_never_fragments() {
        let mut s = ToolCallStream::new(ToolFragmentMode::None);
        assert!(s.feed("call:f{x:1}").is_empty());
        assert!(s.finish("{\"x\":1}").is_empty());
        assert!(
            !s.opened(),
            "an unfragmented call is emitted whole by the route"
        );
    }

    #[test]
    fn a_reset_stream_announces_the_next_call() {
        let mut s = ToolCallStream::new(ToolFragmentMode::Json);
        let _ = s.feed("{\"name\":\"a\",\"arguments\":{\"x\":1}}");
        let _ = s.finish("{\"x\":1}");
        s.reset();
        assert!(!s.opened());
        let d = s.feed("{\"name\":\"b\",\"arguments\":{\"y\":2}}");
        assert!(d[0].opening && d[0].name == "b");
    }
}
