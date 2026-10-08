//! Harmony's tool-call tags (`harmony.rs`) as the generic executor holds them for llama.cpp and
//! MLX: the xgrammar structural tag translated to llguidance.

use superfluid_abi::*;
use superfluid_daemon::harmony::{auto_call_tag, HarmonyIds};
use superfluid_engine::testing::Harness;
use superfluid_engine::Engine;
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives, Vocabulary};

const WRITE: &str = r#"{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}"#;
const BASH: &str = r#"{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}"#;

const ANALYSIS: &str = "<|channel|>analysis<|message|>Write the server, then answer.<|end|><|start|>assistant";
const PREAMBLE: &str = "<|channel|>commentary<|message|>Writing server.py.<|end|><|start|>assistant";
const WRITE_CALL: &str = r#"<|channel|>commentary to=functions.write <|constrain|>json<|message|>{"path":"server.py","content":"main()\n"}<|call|>"#;
const BROKEN_CALL: &str = r#"<|channel|>commentary to=functions.write <|constrain|>json<|message|>{"path":"server.py","content":"main()\n","}<|call|>"#;

fn admitted() -> Vec<String> {
    vec![
        "<|channel|>final<|message|>The server is in server.py.<|return|>".to_string(),
        format!("{ANALYSIS}<|channel|>final<|message|>Done.<|return|>"),
        WRITE_CALL.to_string(),
        format!("{ANALYSIS}{WRITE_CALL}"),
        format!("{PREAMBLE}{WRITE_CALL}"),
        format!("{ANALYSIS}{PREAMBLE}{WRITE_CALL}"),
        r#"<|channel|>commentary to=functions.bash<|constrain|>json<|message|>{"command":"ls"}<|call|>"#.to_string(),
        r#"<|channel|>commentary to=functions.bash json<|message|>{"command":"ls"}<|call|>"#.to_string(),
        r#"<|channel|>commentary to=functions.bash<|message|>{"command":"ls"}<|call|>"#.to_string(),
        r#" to=functions.bash<|channel|>commentary json<|message|>{"command":"ls"}<|call|>"#.to_string(),
        format!(r#"{ANALYSIS} to=functions.bash<|channel|>commentary json<|message|>{{"command":"ls"}}<|call|>"#),
        r#"<|channel|>analysis to=functions.bash<|constrain|>json<|message|>{"command":"ls"}<|call|>"#.to_string(),
    ]
}

fn refused() -> Vec<&'static str> {
    vec![
        BROKEN_CALL,
        r#"<|channel|>commentary to=functions.read<|constrain|>json<|message|>{"path":"a"}<|call|>"#,
        r#"<|channel|>commentary to=functions.bash<|constrain|>json<|message|>{"command":1}<|call|>"#,
        r#"<|channel|>commentary to=functions.write<|constrain|>json<|message|>{"path":"server.py"}<|call|>"#,
        r#"<|channel|>commentary to=functions.bash<|constrain|>json<|message|>{"command":"ls"}<|end|>"#,
    ]
}

const MARKERS: [&str; 7] = ["<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>", "<|call|>", "<|constrain|>"];

fn byte_vocabulary() -> Vocabulary {
    let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let mut specials = Vec::new();
    for m in MARKERS {
        specials.push((m.to_string(), tokens.len() as u32));
        tokens.push([&[0xFF][..], m.as_bytes()].concat());
    }
    let ids = HarmonyIds::discover(&specials).expect("the markers");
    Vocabulary { tokens, specials, eos: vec![ids.ret, ids.call] }
}

fn byte_encode(v: &Vocabulary, text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        if let Some((m, id)) = v.specials.iter().find(|(m, _)| rest.starts_with(m.as_str())) {
            out.push(*id);
            rest = &rest[m.len()..];
        } else {
            out.push(u32::from(rest.as_bytes()[0]));
            rest = &rest[1..];
        }
    }
    out
}

fn harness(v: Vocabulary) -> Harness<Executor<FakePrimitives>> {
    let cfg = FakeConfig { vocab: v.tokens.len() as u32, eos: v.eos[0], vocabulary: Some(v), ..Default::default() };
    Harness::with_engine(Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()))
}

fn admits(h: &mut Harness<Executor<FakePrimitives>>, grammar: u32, stops: &[u32], tokens: &[u32]) -> bool {
    let mut tokens = tokens.to_vec();
    if let Some(last) = tokens.last_mut() {
        if stops.contains(last) {
            *last = stops[0];
        }
    }
    let tokens = &tokens[..];
    let p = h.prompt(tokens);
    let n = tokens.len() as u32;
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

#[test]
fn the_auto_tag_holds_a_calls_arguments_and_frees_the_rest() {
    let v = byte_vocabulary();
    let ids = HarmonyIds::discover(&v.specials).unwrap();
    let tag = auto_call_tag(ids, &[WRITE.to_string(), BASH.to_string()]).expect("a tag");
    let stops = v.eos.clone();
    let mut h = harness(v.clone());
    let g = h.engine.grammar_create_structural(&tag).expect("the tag compiles");
    for text in admitted() {
        assert!(admits(&mut h, g, &stops, &byte_encode(&v, &text)), "the tag must admit:\n{text}");
    }
    for text in refused() {
        assert!(!admits(&mut h, g, &stops, &byte_encode(&v, text)), "the tag must refuse:\n{text}");
    }
    let at = BROKEN_CALL.find(r#"","}"#).unwrap() + 1;
    assert!(admits(&mut h, g, &stops, &byte_encode(&v, &BROKEN_CALL[..at])), "the call up to its last string");
    let closed = format!("{}}}<|call|>", &BROKEN_CALL[..at]);
    assert!(admits(&mut h, g, &stops, &byte_encode(&v, &closed)), "the call closed there:\n{closed}");
}

#[cfg(feature = "llamacpp")]
#[test]
fn gpt_oss_vocabulary_holds_a_calls_arguments_under_auto() {
    let Some(gguf) = std::env::var_os("SUPERFLUID_TEST_GPT_OSS").map(std::path::PathBuf::from).filter(|p| p.is_file()) else {
        eprintln!("SKIP: set SUPERFLUID_TEST_GPT_OSS to a gpt-oss GGUF");
        return;
    };
    use superfluid_engine::Tokenizer;
    let tok = superfluid_adapter_llamacpp::LlamaTokenizer::load(&gguf).expect("tokenizer");
    let specials = tok.special_tokens();
    let ids = HarmonyIds::discover(&specials).expect("gpt-oss speaks Harmony");
    let mut tokens: Vec<Vec<u8>> = (0..tok.vocab_size()).map(|t| tok.token_bytes(t)).collect();
    for (m, id) in &specials {
        tokens[*id as usize] = [&[0xFF][..], m.as_bytes()].concat();
    }
    let stops = vec![ids.ret, ids.call];
    let v = Vocabulary { tokens, specials, eos: stops.clone() };
    let tag = auto_call_tag(ids, &[WRITE.to_string(), BASH.to_string()]).expect("a tag");
    let mut h = harness(v);
    let g = h.engine.grammar_create_structural(&tag).expect("the tag compiles against gpt-oss's vocabulary");
    let encode = |text: &str| {
        let mut t = tok.encode(text);
        if t.first().copied() == tok.bos_token() {
            t.remove(0);
        }
        t
    };
    for text in admitted() {
        assert!(admits(&mut h, g, &stops, &encode(&text)), "the tag must admit:\n{text}");
    }
    for text in refused() {
        assert!(!admits(&mut h, g, &stops, &encode(text)), "the tag must refuse:\n{text}");
    }
}
