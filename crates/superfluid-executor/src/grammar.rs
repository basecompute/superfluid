//! Grammar-constrained decoding for primitive runtimes, through llguidance.

use std::collections::HashMap;
use std::sync::Arc;

use superfluid_abi::Status;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory, TokenParser};
use serde_json::Value;

use crate::primitives::Vocabulary;

struct Compiler {
    factory: ParserFactory,
    specials: Vec<(String, u32)>,
    eos: Arc<Vec<u32>>,
}

impl Compiler {
    fn new(vocab: Vocabulary) -> Result<Compiler, Status> {
        let Some(&primary) = vocab.eos.first() else {
            return Err(Status::Unsupported);
        };
        let info = TokRxInfo {
            vocab_size: vocab.tokens.len() as u32,
            tok_eos: primary,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let trie = TokTrie::from(&info, &vocab.tokens);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&env).map_err(|_| Status::Fatal)?;
        factory.quiet();
        let mut specials = vocab.specials;
        specials.retain(|(s, _)| !s.is_empty());
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        Ok(Compiler { factory, specials, eos: Arc::new(vocab.eos) })
    }

    fn compile(&self, grammar: TopLevelGrammar) -> Result<TokenParser, Status> {
        self.factory.create_parser(grammar).map_err(|_| Status::RejectBadStruct)
    }
}

struct Entry {
    proto: TokenParser,
    open_ended: bool,
    in_use: bool,
    pending_free: bool,
}

#[derive(Default)]
pub(crate) struct Grammars {
    compiler: Option<Compiler>,
    entries: HashMap<u32, Entry>,
    next: u32,
}

impl Grammars {
    fn compiler(&mut self, vocab: impl FnOnce() -> Option<Vocabulary>) -> Result<&Compiler, Status> {
        if self.compiler.is_none() {
            let v = vocab().ok_or(Status::Unsupported)?;
            self.compiler = Some(Compiler::new(v)?);
        }
        Ok(self.compiler.as_ref().expect("built above"))
    }

    fn register(&mut self, proto: TokenParser, open_ended: bool) -> u32 {
        self.next += 1;
        let h = self.next;
        self.entries.insert(h, Entry { proto, open_ended, in_use: false, pending_free: false });
        h
    }

    pub(crate) fn create_schema(
        &mut self,
        vocab: impl FnOnce() -> Option<Vocabulary>,
        json_schema: &str,
    ) -> Result<u32, Status> {
        let schema: Value = serde_json::from_str(json_schema).map_err(|_| Status::RejectBadStruct)?;
        let proto = self.compiler(vocab)?.compile(TopLevelGrammar::from_json_schema(prepared(schema)))?;
        Ok(self.register(proto, false))
    }

    pub(crate) fn create_structural(
        &mut self,
        vocab: impl FnOnce() -> Option<Vocabulary>,
        tag_json: &str,
    ) -> Result<u32, Status> {
        let tag: Value = serde_json::from_str(tag_json).map_err(|_| Status::RejectBadStruct)?;
        let c = self.compiler(vocab)?;
        let lark = structural_to_lark(&tag, &c.specials, &c.eos)?;
        let proto = c.compile(TopLevelGrammar::from_lark(lark))?;
        Ok(self.register(proto, true))
    }

    pub(crate) fn free(&mut self, handle: u32) -> Result<(), Status> {
        let e = self.entries.get_mut(&handle).ok_or(Status::UnknownHandle)?;
        if e.pending_free {
            return Err(Status::UnknownHandle);
        }
        if e.in_use {
            e.pending_free = true;
        } else {
            self.entries.remove(&handle);
        }
        Ok(())
    }

    pub(crate) fn available(&self, handle: u32) -> bool {
        self.entries.get(&handle).is_some_and(|e| !e.in_use && !e.pending_free)
    }

    pub(crate) fn cursor(&self, handle: u32, replay: &[u32]) -> Option<Cursor> {
        let e = self.entries.get(&handle)?;
        let mut matcher = Matcher::new(Ok(e.proto.deep_clone()));
        for &t in replay {
            matcher.consume_token(t).ok()?;
        }
        Some(Cursor(matcher))
    }

    pub(crate) fn borrow(&mut self, handle: u32, cursor: Cursor) -> Option<LaneGrammar> {
        let eos = Arc::clone(&self.compiler.as_ref()?.eos);
        let e = self.entries.get_mut(&handle)?;
        e.in_use = true;
        Some(LaneGrammar { handle, matcher: cursor.0, eos, open_ended: e.open_ended, next: None })
    }

    pub(crate) fn release(&mut self, handle: u32) {
        if let Some(e) = self.entries.get_mut(&handle) {
            e.in_use = false;
            if e.pending_free {
                self.entries.remove(&handle);
            }
        }
    }
}

pub(crate) struct Cursor(Matcher);

pub(crate) struct LaneGrammar {
    pub(crate) handle: u32,
    matcher: Matcher,
    eos: Arc<Vec<u32>>,
    open_ended: bool,
    next: Option<SimpleVob>,
}

impl LaneGrammar {
    fn allowed(&mut self) -> Result<SimpleVob, Status> {
        if let Some(m) = self.next.take() {
            return Ok(m);
        }
        let stopped = self.matcher.is_stopped();
        let mut m = self.matcher.compute_mask_or_eos().map_err(|_| Status::Fatal)?;
        if stopped || self.matcher.is_accepting().unwrap_or(false) {
            for &id in self.eos.iter() {
                if (id as usize) < m.len() {
                    m.allow_token(id);
                }
            }
        }
        Ok(m)
    }

    pub(crate) fn mask(&mut self, row: &mut [f32]) -> Result<(), Status> {
        let allowed = self.allowed()?;
        if allowed.is_zero() {
            return Err(Status::Fatal);
        }
        let n = allowed.len().min(row.len());
        allowed.iter_unset_entries(|i| {
            if i < n {
                row[i] = f32::NEG_INFINITY;
            }
        });
        for v in row.iter_mut().skip(n) {
            *v = f32::NEG_INFINITY;
        }
        Ok(())
    }

    pub(crate) fn accept(&mut self, token: u32) -> Result<bool, Status> {
        if self.eos.contains(&token) {
            return Ok(true);
        }
        self.matcher.consume_token(token).map_err(|_| Status::Fatal)?;
        if self.matcher.is_stopped() && !self.open_ended {
            return Ok(true);
        }
        let next = self.allowed()?;
        let ending = self.eos.iter().filter(|&&id| (id as usize) < next.len() && next.is_allowed(id)).count();
        let only_ending = next.num_set() == ending;
        self.next = Some(next);
        Ok(only_ending && !self.open_ended)
    }
}

pub fn structural_to_lark(tag: &Value, specials: &[(String, u32)], eos: &[u32]) -> Result<String, Status> {
    let mut t = Translator { specials, eos, ends: Vec::new(), rules: Vec::new() };
    let format = match tag.get("type").and_then(Value::as_str) {
        Some("structural_tag") => tag.get("format").ok_or(Status::RejectBadStruct)?,
        _ => tag,
    };
    let start = t.format(format)?;
    let mut out = format!("start: {start}\n");
    for (name, body) in &t.rules {
        out.push_str(&format!("{name}: {body}\n"));
    }
    Ok(out)
}

const WS: &str = r"(/[ \t\r\n]{1,4}/)";

const JSON_WHITESPACE: &str = r"[\x20\x0A\x0D\x09]{1,64}";

const ANNOTATIONS: [&str; 9] =
    ["title", "default", "description", "examples", "deprecated", "readOnly", "writeOnly", "$comment", "$schema"];

fn native_strict(schema: &mut Value) {
    let Value::Object(o) = schema else { return };
    for defs in ["$defs", "definitions"] {
        if let Some(Value::Object(d)) = o.get_mut(defs) {
            d.values_mut().for_each(native_strict);
        }
    }
    if ["$ref", "const", "enum"].iter().any(|k| o.contains_key(*k)) {
        return;
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(Value::Array(branches)) = o.get_mut(key) {
            branches.iter_mut().for_each(native_strict);
            return;
        }
    }
    if let Some(Value::Array(branches)) = o.get_mut("allOf") {
        if let [only] = branches.as_mut_slice() {
            native_strict(only);
        }
        return;
    }
    if o.keys().all(|k| k == "type" || ANNOTATIONS.contains(&k.as_str())) {
        return;
    }
    let (object, array) = {
        let types: Vec<&str> = match o.get("type") {
            Some(Value::String(t)) => vec![t.as_str()],
            Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        let untyped = !o.contains_key("type");
        let has = |keys: &[&str]| keys.iter().any(|k| o.contains_key(*k));
        let object = types.contains(&"object")
            || (untyped && has(&["properties", "additionalProperties", "unevaluatedProperties"]));
        let array = types.contains(&"array") || (untyped && !object && has(&["items", "prefixItems", "unevaluatedItems"]));
        (object, array)
    };
    if object {
        for key in ["properties", "patternProperties"] {
            if let Some(Value::Object(props)) = o.get_mut(key) {
                props.values_mut().for_each(native_strict);
            }
        }
        for key in ["additionalProperties", "unevaluatedProperties"] {
            if let Some(v @ Value::Object(_)) = o.get_mut(key) {
                native_strict(v);
            }
        }
        if !o.contains_key("additionalProperties") && !o.contains_key("unevaluatedProperties") {
            o.insert("additionalProperties".to_string(), Value::Bool(false));
        }
    }
    if array {
        if let Some(Value::Array(prefix)) = o.get_mut("prefixItems") {
            prefix.iter_mut().for_each(native_strict);
        }
        for key in ["items", "unevaluatedItems"] {
            if let Some(v @ Value::Object(_)) = o.get_mut(key) {
                native_strict(v);
            }
        }
        if !o.contains_key("items") && !o.contains_key("unevaluatedItems") {
            o.insert("items".to_string(), Value::Bool(false));
        }
    }
}

fn prepared(mut schema: Value) -> Value {
    native_strict(&mut schema);
    bounded_whitespace(schema)
}

fn bounded_whitespace(mut schema: Value) -> Value {
    if let Value::Object(o) = &mut schema {
        let guidance = o.entry("x-guidance").or_insert_with(|| Value::Object(Default::default()));
        if let Value::Object(g) = guidance {
            if !g.contains_key("whitespace_pattern") && !g.contains_key("whitespace_flexible") {
                g.insert("whitespace_pattern".to_string(), Value::String(JSON_WHITESPACE.to_string()));
            }
        }
    }
    schema
}

struct Translator<'a> {
    specials: &'a [(String, u32)],
    eos: &'a [u32],
    ends: Vec<u32>,
    rules: Vec<(String, String)>,
}

impl Translator<'_> {
    fn rule(&mut self, stem: &str, body: String) -> String {
        let name = format!("{stem}_{}", self.rules.len());
        self.rules.push((name.clone(), body));
        name
    }

    fn literal(&self, s: &str) -> String {
        let mut parts = Vec::new();
        let mut text = String::new();
        let mut rest = s;
        while !rest.is_empty() {
            if let Some((lit, id)) = self.specials.iter().find(|(lit, _)| rest.starts_with(lit.as_str())) {
                if !text.is_empty() {
                    parts.push(serde_json::to_string(&text).expect("a string serializes"));
                    text.clear();
                }
                parts.push(format!("<[{id}]>"));
                rest = &rest[lit.len()..];
            } else {
                let ch = rest.chars().next().expect("non-empty");
                text.push(ch);
                rest = &rest[ch.len_utf8()..];
            }
        }
        if !text.is_empty() {
            parts.push(serde_json::to_string(&text).expect("a string serializes"));
        }
        if parts.is_empty() {
            "\"\"".to_string()
        } else {
            parts.join(" ")
        }
    }

    fn token_id(&self, v: &Value) -> Result<u32, Status> {
        match v {
            Value::Number(n) => n.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or(Status::RejectBadStruct),
            Value::String(s) => {
                self.specials.iter().find(|(lit, _)| lit == s).map(|(_, id)| *id).ok_or(Status::Unsupported)
            }
            _ => Err(Status::RejectBadStruct),
        }
    }

    fn token_of(&self, v: &Value) -> Result<Option<u32>, Status> {
        match v {
            Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("token") => {
                self.token_id(o.get("token").ok_or(Status::RejectBadStruct)?).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn boundary(&self, v: &Value) -> Result<String, Status> {
        if let Value::String(s) = v {
            return Ok(self.literal(s));
        }
        let id = self.token_of(v)?.ok_or(Status::RejectBadStruct)?;
        Ok(format!("<[{id}]>"))
    }

    fn elements(&mut self, v: &Value) -> Result<Vec<String>, Status> {
        let els = v.get("elements").and_then(Value::as_array).ok_or(Status::RejectBadStruct)?;
        els.iter().map(|e| self.format(e)).collect()
    }

    fn content(&mut self, v: &Value) -> Result<String, Status> {
        self.format(v.get("content").ok_or(Status::RejectBadStruct)?)
    }

    fn tag(&mut self, v: &Value) -> Result<String, Status> {
        let begin = self.boundary(v.get("begin").ok_or(Status::RejectBadStruct)?)?;
        let end_v = v.get("end").ok_or(Status::RejectBadStruct)?;
        let end = self.boundary(end_v)?;
        let end_token = self.token_of(end_v)?;
        self.ends.extend(end_token);
        let content = self.content(v);
        if end_token.is_some() {
            self.ends.pop();
        }
        Ok(self.rule("tag", format!("{begin} {} {end}", content?)))
    }

    fn format(&mut self, v: &Value) -> Result<String, Status> {
        let kind = v.get("type").and_then(Value::as_str).ok_or(Status::RejectBadStruct)?;
        Ok(match kind {
            "const_string" => self.literal(v.get("value").and_then(Value::as_str).ok_or(Status::RejectBadStruct)?),
            "json_schema" => {
                if v.get("style").and_then(Value::as_str).is_some_and(|s| s != "json") {
                    return Err(Status::Unsupported);
                }
                let schema = v.get("json_schema").ok_or(Status::RejectBadStruct)?;
                let json = self.rule("json", format!("%json {}", prepared(schema.clone())));
                format!("{WS}? {json} {WS}?")
            }
            "regex" => {
                let p = v.get("pattern").and_then(Value::as_str).ok_or(Status::RejectBadStruct)?;
                format!("/{}/", p.replace('/', "\\/"))
            }
            "sequence" => format!("({})", self.elements(v)?.join(" ")),
            "or" => format!("({})", self.elements(v)?.join(" | ")),
            "optional" => format!("({})?", self.content(v)?),
            "plus" => format!("({})+", self.content(v)?),
            "star" => format!("({})*", self.content(v)?),
            "tag" => self.tag(v)?,
            "triggered_tags" => self.triggered(v)?,
            "token" => format!("<[{}]>", self.token_id(v.get("token").ok_or(Status::RejectBadStruct)?)?),
            "any_tokens" => {
                let mut excluded: Vec<u32> = self.eos.to_vec();
                excluded.extend(self.ends.last().copied());
                if let Some(list) = v.get("exclude_tokens") {
                    for t in list.as_array().ok_or(Status::RejectBadStruct)? {
                        excluded.push(self.token_id(t)?);
                    }
                }
                excluded.sort_unstable();
                excluded.dedup();
                let ids: Vec<String> = excluded.iter().map(u32::to_string).collect();
                format!("<[^{}]>*", ids.join(","))
            }
            _ => return Err(Status::Unsupported),
        })
    }

    fn triggered(&mut self, v: &Value) -> Result<String, Status> {
        let triggers = v.get("triggers").and_then(Value::as_array).ok_or(Status::RejectBadStruct)?;
        let mut excluded: Vec<u32> = self.eos.to_vec();
        for trig in triggers {
            let s = trig.as_str().ok_or(Status::RejectBadStruct)?;
            let id = self
                .specials
                .iter()
                .find(|(lit, _)| s.starts_with(lit.as_str()))
                .map(|(_, id)| *id)
                .ok_or(Status::Unsupported)?;
            if !excluded.contains(&id) {
                excluded.push(id);
            }
        }
        let tags = v.get("tags").and_then(Value::as_array).ok_or(Status::RejectBadStruct)?;
        if tags.is_empty() {
            return Err(Status::RejectBadStruct);
        }
        let calls: Vec<String> = tags.iter().map(|t| self.tag(t)).collect::<Result<_, _>>()?;
        let call = self.rule("call", calls.join(" | "));
        let ids: Vec<String> = excluded.iter().map(u32::to_string).collect();
        let text = self.rule("text", format!("<[^{}]>*", ids.join(",")));
        let at_least_one = v.get("at_least_one").and_then(Value::as_bool).unwrap_or(false);
        let stop_after_first = v.get("stop_after_first").and_then(Value::as_bool).unwrap_or(false);
        Ok(match (at_least_one, stop_after_first) {
            (false, false) => format!("{text} ({call} {text})*"),
            (true, false) => format!("{call} ({text} {call})* {text}"),
            (false, true) => format!("{text} {call}?"),
            (true, true) => call,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specials() -> Vec<(String, u32)> {
        vec![("</tool_call>".to_string(), 101), ("<tool_call>".to_string(), 100)]
    }

    fn envelope_tag(at_least_one: bool) -> Value {
        serde_json::json!({"type": "structural_tag", "format": {
            "type": "triggered_tags",
            "triggers": ["<tool_call>"],
            "tags": [{"type": "tag", "begin": "<tool_call>", "content": {"type": "json_schema",
                "json_schema": {"type": "object", "properties": {"name": {"const": "f"}}}}, "end": "</tool_call>"}],
            "at_least_one": at_least_one,
            "stop_after_first": false,
        }})
    }

    #[test]
    fn a_tool_call_tag_becomes_free_text_around_framed_calls() {
        let lark = structural_to_lark(&envelope_tag(false), &specials(), &[7]).unwrap();
        assert!(lark.starts_with("start: text_3 (call_2 text_3)*\n"), "{lark}");
        assert!(
            lark.contains(&format!("tag_1: <[100]> {WS}? json_0 {WS}? <[101]>")),
            "the frame is the marker tokens: {lark}"
        );
        assert!(lark.contains("text_3: <[^7,100]>*"), "free text excludes the stop set and the trigger: {lark}");
    }

    #[test]
    fn a_forced_call_starts_with_the_call() {
        let lark = structural_to_lark(&envelope_tag(true), &specials(), &[7]).unwrap();
        assert!(lark.starts_with("start: call_2 (text_3 call_2)* text_3\n"), "{lark}");
    }

    #[test]
    fn text_between_markers_stays_text() {
        let sp = specials();
        let t = Translator { specials: &sp, eos: &[], ends: Vec::new(), rules: Vec::new() };
        assert_eq!(t.literal("<tool_call>\n{\"name\": \"f\""), r#"<[100]> "\n{\"name\": \"f\"""#);
        assert_eq!(t.literal("plain"), "\"plain\"");
        assert_eq!(t.literal(""), "\"\"");
    }

    #[test]
    fn a_trigger_that_is_not_a_marker_is_refused() {
        let mut tag = envelope_tag(false);
        tag["format"]["triggers"] = serde_json::json!(["<function="]);
        assert_eq!(structural_to_lark(&tag, &specials(), &[7]).unwrap_err(), Status::Unsupported);
    }

    #[test]
    fn an_xml_style_body_is_refused() {
        let mut tag = envelope_tag(false);
        tag["format"]["tags"][0]["content"]["style"] = serde_json::json!("qwen_xml");
        assert_eq!(structural_to_lark(&tag, &specials(), &[7]).unwrap_err(), Status::Unsupported);
    }

    #[test]
    fn markers_named_by_id_and_a_body_up_to_its_end_token() {
        let tag = serde_json::json!({"type": "structural_tag", "format": {"type": "sequence", "elements": [
            {"type": "optional", "content": {"type": "sequence", "elements": [
                {"type": "token", "token": 200}, {"type": "const_string", "value": "analysis"},
                {"type": "tag", "begin": {"type": "token", "token": 201},
                 "content": {"type": "any_tokens", "exclude_tokens": [200, 203]},
                 "end": {"type": "token", "token": 202}},
                {"type": "token", "token": 203}, {"type": "const_string", "value": "assistant"}]}},
            {"type": "sequence", "elements": [
                {"type": "token", "token": 200}, {"type": "const_string", "value": "commentary to=functions.f"},
                {"type": "token", "token": 201}, {"type": "json_schema", "json_schema": {"type": "object"}}]}]}});
        let lark = structural_to_lark(&tag, &specials(), &[7]).unwrap();
        assert!(lark.contains(r#"((<[200]> "analysis" tag_0 <[203]> "assistant"))?"#), "the analysis may be left out: {lark}");
        assert!(
            lark.contains("tag_0: <[201]> <[^7,200,202,203]>* <[202]>"),
            "the body stops at its end token and at the stop set: {lark}"
        );
        assert!(lark.contains(r#"(<[200]> "commentary to=functions.f" <[201]> "#), "{lark}");

        let named = serde_json::json!({"type": "token", "token": "<tool_call>"});
        assert_eq!(structural_to_lark(&named, &specials(), &[7]).unwrap(), "start: <[100]>\n");
        let unknown = serde_json::json!({"type": "token", "token": "<nope>"});
        assert_eq!(structural_to_lark(&unknown, &specials(), &[7]).unwrap_err(), Status::Unsupported);
    }

    #[test]
    fn composite_bodies_translate() {
        let tag = serde_json::json!({"type": "triggered_tags", "triggers": ["<tool_call>"], "tags": [{
            "begin": "<tool_call>call:f{", "end": "}</tool_call>",
            "content": {"type": "sequence", "elements": [
                {"type": "const_string", "value": "city:"},
                {"type": "or", "elements": [{"type": "regex", "pattern": "[^,}]*"}, {"type": "const_string", "value": "x/y"}]},
            ]}}], "at_least_one": true});
        let lark = structural_to_lark(&tag, &specials(), &[7]).unwrap();
        assert!(lark.contains(r#"<[100]> "call:f{" ("city:" (/[^,}]*/ | "x/y")) "}" <[101]>"#), "{lark}");
    }

    fn strict(v: Value) -> Value {
        let mut v = v;
        native_strict(&mut v);
        v
    }

    #[test]
    fn schemas_close_where_xgrammar_closes_them() {
        use serde_json::json;
        let cases = [
            (json!({"type": "object", "properties": {"a": {}}}),
             json!({"type": "object", "properties": {"a": {}}, "additionalProperties": false})),
            (json!({"properties": {"a": {}}}), json!({"properties": {"a": {}}, "additionalProperties": false})),
            (json!({"type": "object", "unevaluatedProperties": true}), json!({"type": "object", "unevaluatedProperties": true})),
            (json!({"type": "object", "title": "t", "description": "d"}), json!({"type": "object", "title": "t", "description": "d"})),
            (json!({"type": "object", "$defs": {}}), json!({"type": "object", "$defs": {}, "additionalProperties": false})),
            (json!({"type": "array", "minItems": 0}), json!({"type": "array", "minItems": 0, "items": false})),
            (json!({"prefixItems": [{"type": "object", "required": []}]}),
             json!({"prefixItems": [{"type": "object", "required": [], "additionalProperties": false}], "items": false})),
            (json!({"type": "array", "items": {"type": "object", "properties": {}}}),
             json!({"type": "array", "items": {"type": "object", "properties": {}, "additionalProperties": false}})),
            (json!({"type": ["object", "null"]}), json!({"type": ["object", "null"]})),
            (json!({"type": ["object", "null"], "required": ["a"]}),
             json!({"type": ["object", "null"], "required": ["a"], "additionalProperties": false})),
            (json!({"$ref": "#/$defs/p", "properties": {}}), json!({"$ref": "#/$defs/p", "properties": {}})),
            (json!({"const": {"a": 1}}), json!({"const": {"a": 1}})),
            (json!({"enum": [{"a": 1}]}), json!({"enum": [{"a": 1}]})),
            (json!({"anyOf": [{"type": "object", "required": []}, {"type": "null"}], "properties": {}}),
             json!({"anyOf": [{"type": "object", "required": [], "additionalProperties": false}, {"type": "null"}], "properties": {}})),
            (json!({"allOf": [{"type": "object", "required": []}]}),
             json!({"allOf": [{"type": "object", "required": [], "additionalProperties": false}]})),
            (json!({"allOf": [{"type": "object", "required": []}, {"required": []}]}),
             json!({"allOf": [{"type": "object", "required": []}, {"required": []}]})),
            (json!({"$defs": {"p": {"type": "object", "required": []}}, "$ref": "#/$defs/p"}),
             json!({"$defs": {"p": {"type": "object", "required": [], "additionalProperties": false}}, "$ref": "#/$defs/p"})),
            (json!({"type": "object", "additionalProperties": {"type": "object", "required": []}}),
             json!({"type": "object", "additionalProperties": {"type": "object", "required": [], "additionalProperties": false}})),
            (json!(true), json!(true)),
        ];
        for (input, want) in cases {
            assert_eq!(strict(input.clone()), want, "{input}");
        }
    }
}
