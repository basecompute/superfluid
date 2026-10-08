//! Grammar-constrained decoding in the executor, on the fake runtime with a small byte-level
//! vocabulary.

use superfluid_abi::*;
use superfluid_engine::testing::Harness;
use superfluid_engine::Engine;
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives, Vocabulary};

const CALL: u32 = 256;
const END_CALL: u32 = 257;
const EOS: u32 = 299;

fn vocabulary() -> Vocabulary {
    let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let special = |s: &str| {
        let mut v = vec![0xFF];
        v.extend_from_slice(s.as_bytes());
        v
    };
    tokens.push(special("<call>"));
    tokens.push(special("</call>"));
    for i in 258..EOS {
        tokens.push(special(&format!("<unused{i}>")));
    }
    tokens.push(special("</s>"));
    Vocabulary {
        tokens,
        specials: vec![("<call>".into(), CALL), ("</call>".into(), END_CALL)],
        eos: vec![EOS],
    }
}

fn harness_with(vocabulary: Option<Vocabulary>) -> Harness<Executor<FakePrimitives>> {
    let cfg = FakeConfig { vocab: EOS + 1, eos: EOS, vocabulary, ..Default::default() };
    Harness::with_engine(Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()))
}

fn harness() -> Harness<Executor<FakePrimitives>> {
    harness_with(Some(vocabulary()))
}

const OBJECT: &str = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false,"x-guidance":{"whitespace_flexible":false}}"#;

fn structural_tag(at_least_one: bool) -> String {
    serde_json::json!({"type": "structural_tag", "format": {
        "type": "triggered_tags",
        "triggers": ["<call>"],
        "tags": [{"type": "tag", "begin": "<call>",
            "content": {"type": "json_schema", "json_schema": serde_json::from_str::<serde_json::Value>(OBJECT).unwrap()},
            "end": "</call>"}],
        "at_least_one": at_least_one,
        "stop_after_first": false,
    }})
    .to_string()
}

fn constrained(grammar: u32) -> impl FnOnce(LaneAdmit) -> LaneAdmit {
    move |mut a| {
        a.grammar_handle = grammar;
        a.params.flags = 0;
        a
    }
}

fn text(tokens: &[u32]) -> Vec<u8> {
    tokens.iter().filter(|&&t| t < 256).map(|&t| t as u8).collect()
}

#[test]
fn a_schema_grammar_holds_every_draw_and_finishes_the_lane() {
    let mut h = harness();
    let g = h.engine.grammar_create(OBJECT).expect("compiles");
    let p = h.prompt(&[1, 2, 3]);
    let ev = h.tick_ok(h.plan().admit_with(constrained(g), 1, p).prefill(1, 0, 3).decode(1, 64));
    let e = *ev.emit_for(1);
    assert_eq!(e.finish, finish::GRAMMAR, "the grammar ended the lane");
    let out = h.rings_tokens(&e.token_ref);
    let v: serde_json::Value = serde_json::from_slice(&text(&out)).expect("valid JSON");
    assert!(v["ok"].is_boolean(), "{v}");
    assert_eq!(v.as_object().unwrap().len(), 1, "nothing but the declared key: {v}");
}

#[test]
fn whitespace_inside_a_value_is_bounded() {
    let mut h = harness();
    let schema = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#;
    let g = h.engine.grammar_create(schema).expect("compiles");
    let p = h.prompt(&[1, 2, 3]);
    h.tick_ok(h.plan().admit_with(constrained(g), 1, p).prefill(1, 0, 3));
    let seq = h.engine.lane_sequence(1).unwrap();
    let mut script: Vec<u32> = br#"{"ok":true"#.iter().map(|&b| u32::from(b)).collect();
    script.extend(std::iter::repeat_n(u32::from(b' '), 200));
    h.engine.primitives_mut().script(seq, script);
    let ev = h.tick_ok(h.plan().decode(1, 120));
    let e = *ev.emit_for(1);
    assert_eq!(e.finish, finish::GRAMMAR, "the value had to end");
    let out = text(&h.rings_tokens(&e.token_ref));
    let spaces = out.iter().filter(|&&b| b == b' ').count();
    assert!((1..=64).contains(&spaces), "{spaces} spaces in one gap");
    let v: serde_json::Value = serde_json::from_slice(&out).expect("valid JSON");
    assert_eq!(v["ok"], true);
}

fn resume(h: &mut Harness<Executor<FakePrimitives>>, lane: u64, prompt: &[u32], grammar: u32, replay: u32) -> Vec<u32> {
    let p = h.prompt(prompt);
    let admit = move |a| {
        let mut a = constrained(grammar)(a);
        a.grammar_replay = replay;
        a
    };
    let ev = h.tick_ok(h.plan().admit_with(admit, lane, p).prefill(lane, 0, prompt.len() as u32).decode(lane, 64));
    let out = h.rings_tokens(&ev.emit_for(lane).token_ref);
    h.tick_ok(h.plan().retire(lane, false));
    out
}

#[test]
fn a_continuation_resumes_its_grammar_where_the_generation_stopped() {
    let mut h = harness();
    let g = h.engine.grammar_create(OBJECT).expect("compiles");
    let prompt = [1u32, 2, 3];
    let p = h.prompt(&prompt);
    let ev = h.tick_ok(h.plan().admit_with(constrained(g), 1, p).prefill(1, 0, 3).decode(1, 5));
    let head = h.rings_tokens(&ev.emit_for(1).token_ref);
    assert_eq!(head.len(), 5);
    h.tick_ok(h.plan().retire(1, false));
    let mut cont = prompt.to_vec();
    cont.extend_from_slice(&head);

    let whole = text(&[head.clone(), resume(&mut h, 2, &cont, g, 5)].concat());
    let v: serde_json::Value = serde_json::from_slice(&whole)
        .unwrap_or_else(|e| panic!("not one object ({e}): {}", String::from_utf8_lossy(&whole)));
    assert!(v["ok"].is_boolean(), "{v}");
    let restarted = text(&resume(&mut h, 3, &cont, g, 0));
    assert_eq!(restarted.first(), Some(&b'{'), "from the start the grammar opens another object");

    let refused = |h: &mut Harness<Executor<FakePrimitives>>, grammar: u32, replay: u32| {
        let p = h.prompt(&cont);
        let admit = move |mut a: LaneAdmit| {
            a.grammar_handle = grammar;
            a.grammar_replay = replay;
            a
        };
        h.tick_err(h.plan().admit_with(admit, 4, p))
    };
    assert_eq!(refused(&mut h, g, 3), Status::RejectIllegalCombination);
    assert_eq!(refused(&mut h, g, cont.len() as u32 + 1), Status::RejectIllegalCombination);
    assert_eq!(refused(&mut h, 0, 5), Status::RejectIllegalCombination);
    let again = text(&[head, resume(&mut h, 5, &cont, g, 5)].concat());
    assert!(serde_json::from_slice::<serde_json::Value>(&again).is_ok());
}

#[test]
fn a_structural_tag_holds_a_call_and_frees_the_text_around_it() {
    let mut h = harness();
    let g = h.engine.grammar_create_structural(&structural_tag(false)).expect("compiles");
    let p = h.prompt(&[1, 2, 3]);
    h.tick_ok(h.plan().admit_with(constrained(g), 1, p).prefill(1, 0, 3));
    let seq = h.engine.lane_sequence(1).unwrap();
    h.engine.primitives_mut().script(seq, vec![b'h' as u32, b'i' as u32, CALL]);
    let ev = h.tick_ok(h.plan().decode(1, 40));
    let e = *ev.emit_for(1);
    let out = h.rings_tokens(&e.token_ref);
    assert_eq!(&out[..3], &[b'h' as u32, b'i' as u32, CALL], "free text, then the call opens");
    let close = out.iter().position(|&t| t == END_CALL).expect("the call closed");
    let body: serde_json::Value = serde_json::from_slice(&text(&out[3..close])).expect("the call body is JSON");
    assert!(body["ok"].is_boolean(), "{body}");
    assert!(out.len() > close + 1, "free text again after the call");
    assert_ne!(e.finish, finish::GRAMMAR, "free text can always go on");
}

#[test]
fn a_forced_call_starts_with_its_marker() {
    let mut h = harness();
    let g = h.engine.grammar_create_structural(&structural_tag(true)).expect("compiles");
    let p = h.prompt(&[1, 2, 3]);
    let ev = h.tick_ok(h.plan().admit_with(constrained(g), 1, p).prefill(1, 0, 3).decode(1, 20));
    let out = h.rings_tokens(&ev.emit_for(1).token_ref);
    assert_eq!(out[0], CALL, "{out:?}");
}

const CHANNEL: u32 = 258;
const MESSAGE: u32 = 259;
const END: u32 = 260;
const START: u32 = 261;

fn harmony_vocabulary() -> Vocabulary {
    let mut v = vocabulary();
    for (id, name) in [(CHANNEL, "<|channel|>"), (MESSAGE, "<|message|>"), (END, "<|end|>"), (START, "<|start|>")] {
        let mut bytes = vec![0xFF];
        bytes.extend_from_slice(name.as_bytes());
        v.tokens[id as usize] = bytes;
        v.specials.push((name.to_string(), id));
    }
    v
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

#[test]
fn a_header_framed_call_is_held_and_ends_on_the_models_stop_token() {
    let mut h = harness_with(Some(harmony_vocabulary()));
    let schema: serde_json::Value = serde_json::from_str(OBJECT).unwrap();
    let tag = serde_json::json!({"type": "structural_tag", "format": {"type": "sequence", "elements": [
        {"type": "optional", "content": {"type": "sequence", "elements": [
            {"type": "token", "token": CHANNEL}, {"type": "const_string", "value": "analysis"},
            {"type": "tag", "begin": {"type": "token", "token": MESSAGE},
             "content": {"type": "any_tokens", "exclude_tokens": [CHANNEL, MESSAGE, END, START]},
             "end": {"type": "token", "token": END}},
            {"type": "token", "token": START}, {"type": "const_string", "value": "assistant"}]}},
        {"type": "sequence", "elements": [
            {"type": "token", "token": CHANNEL}, {"type": "const_string", "value": "commentary to=functions.f"},
            {"type": "token", "token": MESSAGE}, {"type": "json_schema", "json_schema": schema}]}]}});
    let g = h.engine.grammar_create_structural(&tag.to_string()).expect("compiles");
    let p = h.prompt(&[1, 2, 3]);
    h.tick_ok(h.plan().admit_with(constrained(g), 1, p).prefill(1, 0, 3));
    let seq = h.engine.lane_sequence(1).unwrap();
    let reasoning: Vec<u32> = [vec![CHANNEL], bytes("analysis"), vec![MESSAGE], bytes("hm"), vec![END, START], bytes("assistant")].concat();
    h.engine.primitives_mut().script(seq, reasoning.clone());
    let ev = h.tick_ok(h.plan().decode(1, 120));
    let e = *ev.emit_for(1);
    let out = h.rings_tokens(&e.token_ref);
    assert_eq!(&out[..reasoning.len()], &reasoning[..], "the analysis runs as the model wrote it");
    let call = &out[reasoning.len()..];
    assert_eq!(call[0], CHANNEL, "then the call's header: {call:?}");
    let body_at = call.iter().position(|&t| t == MESSAGE).expect("the header ends");
    assert_eq!(text(&call[1..body_at]), b"commentary to=functions.f");
    assert_eq!(*call.last().unwrap(), EOS, "the model's stop token ends the call: {call:?}");
    let body: serde_json::Value = serde_json::from_slice(&text(&call[body_at + 1..call.len() - 1])).expect("the body is JSON");
    assert!(body["ok"].is_boolean(), "{body}");
    assert_eq!(e.finish, finish::EOS, "an open-ended grammar ends on the stop token, not on completion");
}

#[test]
fn a_grammar_handle_serves_one_lane_at_a_time() {
    let mut h = harness();
    let g = h.engine.grammar_create(OBJECT).unwrap();
    let p = h.prompt(&[1, 2]);
    h.tick_ok(h.plan().admit_with(constrained(g), 1, p).prefill(1, 0, 2));

    let p2 = h.prompt(&[3, 4]);
    let err = h.tick_err(h.plan().admit_with(constrained(g), 2, p2).prefill(2, 0, 2));
    assert_eq!(err, Status::RejectIllegalCombination, "a borrowed handle is busy");

    let p3 = h.prompt(&[5]);
    let host = move |mut a: LaneAdmit| {
        a.grammar_handle = g;
        a.sampling = sampling::HOST;
        a
    };
    assert_eq!(h.tick_err(h.plan().admit_with(host, 3, p3)), Status::RejectIllegalCombination);

    let p4 = h.prompt(&[6]);
    assert_eq!(
        h.tick_err(h.plan().admit_with(constrained(9999), 4, p4)),
        Status::RejectIllegalCombination,
        "an unknown handle"
    );

    h.engine.grammar_free(g).expect("deferred free");
    assert_eq!(h.engine.grammar_free(g), Err(Status::UnknownHandle), "a double free");
    h.tick_ok(h.plan().retire(1, false));
    assert_eq!(h.engine.grammar_free(g), Err(Status::UnknownHandle));
    let p5 = h.prompt(&[7]);
    assert_eq!(h.tick_err(h.plan().admit_with(constrained(g), 5, p5)), Status::RejectIllegalCombination);

    let g2 = h.engine.grammar_create(OBJECT).unwrap();
    let p6 = h.prompt(&[8]);
    h.tick_ok(h.plan().admit_with(constrained(g2), 6, p6).prefill(6, 0, 1));
    h.tick_ok(h.plan().retire(6, false));
    let p7 = h.prompt(&[9]);
    h.tick_ok(h.plan().admit_with(constrained(g2), 7, p7).prefill(7, 0, 1));
}

#[test]
fn lanes_under_different_grammars_decode_together() {
    let mut h = harness();
    let a = h.engine.grammar_create(OBJECT).unwrap();
    let b = h.engine.grammar_create(r#"{"type":"string","enum":["yes","no"]}"#).unwrap();
    let p1 = h.prompt(&[1, 2]);
    let p2 = h.prompt(&[3, 4]);
    let ev = h.tick_ok(
        h.plan()
            .admit_with(constrained(a), 1, p1)
            .admit_with(constrained(b), 2, p2)
            .prefill(1, 0, 2)
            .prefill(2, 0, 2)
            .decode(1, 64)
            .decode(2, 64),
    );
    let one: serde_json::Value = serde_json::from_slice(&text(&h.rings_tokens(&ev.emit_for(1).token_ref))).unwrap();
    assert!(one["ok"].is_boolean(), "{one}");
    let two: String = serde_json::from_slice(&text(&h.rings_tokens(&ev.emit_for(2).token_ref))).unwrap();
    assert!(two == "yes" || two == "no", "{two}");
    assert_eq!(ev.emit_for(2).finish, finish::GRAMMAR);
}

#[test]
fn no_vocabulary_no_grammar() {
    let mut h = harness_with(None);
    assert_eq!(h.engine.grammar_create(OBJECT), Err(Status::Unsupported));
    assert_eq!(h.engine.grammar_create_structural(&structural_tag(false)), Err(Status::Unsupported));
}

#[test]
fn an_inexpressible_tag_is_refused() {
    let mut h = harness();
    let tag = serde_json::json!({"type": "triggered_tags", "triggers": ["<fn="], "tags": [{"begin": "<fn=f>",
        "content": {"type": "json_schema", "json_schema": {"type": "object"}}, "end": "</fn>"}]});
    assert_eq!(h.engine.grammar_create_structural(&tag.to_string()), Err(Status::Unsupported));
    assert_eq!(h.engine.grammar_create("{not json"), Err(Status::RejectBadStruct));
}

fn accepts(h: &mut Harness<Executor<FakePrimitives>>, grammar: u32, prefix: &[u32]) -> bool {
    let p = h.prompt(prefix);
    let n = prefix.len() as u32;
    let admit = move |mut a: LaneAdmit| {
        a.grammar_handle = grammar;
        a.grammar_replay = n;
        a
    };
    let (plan, arena) = h.plan().admit_with(admit, 1, p).build();
    match h.engine.tick(&plan, &arena, &mut h.rings) {
        Ok(_) => {
            h.tick_ok(h.plan().retire(1, false));
            true
        }
        Err(Status::RejectIllegalCombination) => false,
        Err(s) => panic!("admit failed otherwise: {s:?}"),
    }
}

fn bytes_of(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

#[test]
fn a_schema_admits_only_what_it_declares_as_the_native_engine_does() {
    let mut h = harness();
    let declared = h
        .engine
        .grammar_create(&serde_json::json!({"type": "object",
            "properties": {"a": {"type": "integer"}}, "required": ["a"]}).to_string())
        .expect("compiles");
    assert!(accepts(&mut h, declared, &bytes_of(r#"{"a":1}"#)));
    assert!(!accepts(&mut h, declared, &bytes_of(r#"{"a":1,"b"#)), "an undeclared key");

    let open = h
        .engine
        .grammar_create(&serde_json::json!({"type": "object",
            "properties": {"a": {"type": "integer"}}, "additionalProperties": true}).to_string())
        .expect("compiles");
    assert!(accepts(&mut h, open, &bytes_of(r#"{"a":1,"b":"#)), "declared open stays open");

    let bare = h
        .engine
        .grammar_create(&serde_json::json!({"type": "object", "description": "anything"}).to_string())
        .expect("compiles");
    assert!(accepts(&mut h, bare, &bytes_of(r#"{"b":1,"c":"#)), "a bare object is any object");

    let nested = h
        .engine
        .grammar_create(&serde_json::json!({
            "$defs": {"point": {"type": "object", "properties": {"x": {"type": "integer"}}}},
            "type": "object", "properties": {"p": {"$ref": "#/$defs/point"}}}).to_string())
        .expect("compiles");
    assert!(accepts(&mut h, nested, &bytes_of(r#"{"p":{"x":1}"#)));
    assert!(!accepts(&mut h, nested, &bytes_of(r#"{"p":{"x":1,"y"#)), "through a $ref too");

    let tuple = h
        .engine
        .grammar_create(&serde_json::json!({"type": "object",
            "properties": {"t": {"type": "array", "prefixItems": [{"type": "integer"}]}}}).to_string())
        .expect("compiles");
    assert!(accepts(&mut h, tuple, &bytes_of(r#"{"t":[1]"#)));
    assert!(!accepts(&mut h, tuple, &bytes_of(r#"{"t":[1,2"#)), "an item past the declared ones");

    let any_array = h.engine.grammar_create(r#"{"type":"array"}"#).expect("compiles");
    assert!(accepts(&mut h, any_array, &bytes_of(r#"[1,"x","#)), "a bare array is any array");

    let tag = serde_json::json!({"type": "structural_tag", "format": {
        "type": "triggered_tags", "triggers": ["<call>"],
        "tags": [{"type": "tag", "begin": "<call>",
            "content": {"type": "json_schema", "json_schema": {"type": "object",
                "properties": {"ok": {"type": "boolean"}}}},
            "end": "</call>"}],
        "at_least_one": false, "stop_after_first": false}});
    let call = h.engine.grammar_create_structural(&tag.to_string()).expect("compiles");
    let mut ok = vec![CALL];
    ok.extend(bytes_of(r#"{"ok":true}"#));
    assert!(accepts(&mut h, call, &ok));
    let mut extra = vec![CALL];
    extra.extend(bytes_of(r#"{"ok":true,"x"#));
    assert!(!accepts(&mut h, call, &extra), "an undeclared key inside a call");
}
