//! Golden template tests.

#![cfg(feature = "basert")]

use std::path::PathBuf;

use superfluid_daemon::atem::AtemCodec;
use superfluid_daemon::codec::{ChatMlCodec, TextCodec, TurnState};
use superfluid_daemon::wal::{channel, role};

fn qwen_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_MODEL") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    ["models/Qwen3-0.6B-Q4_K_M.base", "models/Qwen3-0.6B-Q4_K_M.gguf"]
        .iter()
        .map(|c| root.join(c))
        .find(|p| p.exists())
}

fn qwen35_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_QWEN35") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    ["models/Qwen3.5-2B-Q4.base"].iter().map(|c| root.join(c)).find(|p| p.exists())
}

fn glimmer_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_GLIMMER") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    ["models/muse-glimmer-30B-kquant-17gb.base"]
        .iter()
        .map(|c| root.join(c))
        .find(|p| p.exists())
}

fn marked<'a>(
    codec: &'a dyn TextCodec,
    markers: &[&'static str],
) -> impl Fn(&[u32]) -> String + 'a {
    let map: std::collections::HashMap<u32, &'static str> = markers
        .iter()
        .map(|m| {
            let t = codec.encode(m);
            assert_eq!(t.len(), 1, "marker {m} must be atomic");
            (t[0], *m)
        })
        .collect();
    move |span: &[u32]| {
        let mut out = String::new();
        let mut stream = superfluid_daemon::codec::Utf8Stream::default();
        for &t in span {
            if let Some(m) = map.get(&t) {
                out.push_str(&stream.flush());
                out.push_str(m);
            } else {
                out.push_str(&stream.push(&codec.token_bytes(t)));
            }
        }
        out.push_str(&stream.flush());
        out
    }
}

#[test]
fn chatml_rendering_is_pinned() {
    let Some(model) = qwen_path() else {
        eprintln!("SKIP: no Qwen3 test model");
        return;
    };
    let c = ChatMlCodec::load(&model).expect("codec load");
    let tx = marked(&c, &["<|im_start|>", "<|im_end|>"]);

    assert_eq!(
        tx(&c.render_message(role::USER, "Hello").unwrap()),
        "<|im_start|>user\nHello<|im_end|>\n"
    );
    assert_eq!(
        tx(&c.render_message(role::SYSTEM, "Be brief.").unwrap()),
        "<|im_start|>system\nBe brief.<|im_end|>\n"
    );
    assert_eq!(
        tx(&c.render_message(role::ASSISTANT, "Hi!").unwrap()),
        "<|im_start|>assistant\nHi!<|im_end|>\n"
    );
    assert_eq!(
        tx(&c.generation_prefix(TurnState::AfterInput).unwrap()),
        "<|im_start|>assistant\n"
    );
    assert_eq!(
        tx(&c.generation_prefix(TurnState::AfterFinishedTurn).unwrap()),
        "\n<|im_start|>assistant\n"
    );
    assert_eq!(
        tx(&c.render_tool_result("get_weather", "22C").unwrap()),
        "<|im_start|>user\n<tool_response>\n22C\n</tool_response><|im_end|>\n"
    );
    let tool = r#"{"name": "get_weather", "description": "d", "parameters": {}}"#.to_string();
    assert_eq!(
        tx(&c.render_system_with_tools(Some("Sys."), &[tool]).unwrap()),
        "<|im_start|>system\nSys.\n\n# Tools\n\nYou may call one or more functions to assist \
         with the user query.\n\nYou are provided with function signatures within \
         <tools></tools> XML tags:\n<tools>\n{\"name\": \"get_weather\", \"description\": \
         \"d\", \"parameters\": {}}\n</tools>\n\nFor each function call, return a json \
         object with function name and arguments within <tool_call></tool_call> XML \
         tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": \
         <args-json-object>}\n</tool_call><|im_end|>\n"
    );
    assert_eq!(
        tx(&c.render_assistant_with_tool_calls(
            "",
            &[("get_weather".into(), r#"{"city":"Paris"}"#.into())]
        )
        .unwrap()),
        "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
         {\"city\":\"Paris\"}}\n</tool_call><|im_end|>\n"
    );

    let base_fp = c.behavior_fingerprint().unwrap().digest();
    let nothink = ChatMlCodec::load(&model).unwrap().with_thinking(false);
    let ntx = marked(&nothink, &["<|im_start|>", "<|im_end|>"]);
    assert_eq!(
        ntx(&nothink.generation_prefix(TurnState::AfterInput).unwrap()),
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_ne!(
        nothink.behavior_fingerprint().unwrap().digest(),
        base_fp,
        "think-injection is part of the generation-behavior identity"
    );
}

#[test]
fn chatml_opener_follows_the_template_thinking_default() {
    let empty = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let open = "<|im_start|>assistant\n<think>\n";
    let bare = "<|im_start|>assistant\n";
    for (name, path, want) in [
        ("Qwen3", qwen_path(), [bare, bare, empty]),
        ("Qwen3.5", qwen35_path(), [empty, open, empty]),
    ] {
        let Some(model) = path else {
            eprintln!("SKIP {name}: no bundle (BASERT_TEST_MODEL / BASERT_TEST_QWEN35)");
            continue;
        };
        let c = ChatMlCodec::load(&model).expect("codec load");
        let tx = marked(&c, &["<|im_start|>", "<|im_end|>", "<think>", "</think>"]);
        for (thinking, want) in [None, Some(true), Some(false)].into_iter().zip(want) {
            assert_eq!(
                tx(&c.generation_prefix_with(TurnState::AfterInput, thinking).unwrap()),
                want,
                "{name}, enable_thinking {thinking:?}"
            );
        }
        assert_eq!(
            c.generation_prefix(TurnState::AfterInput),
            c.generation_prefix_with(TurnState::AfterInput, None),
            "{name}"
        );
        assert_eq!(
            tx(&c.generation_prefix_with(TurnState::AfterFinishedTurn, Some(false)).unwrap()),
            format!("\n{empty}"),
            "{name}"
        );
        let msgs = [superfluid_daemon::codec::ChatMessage::new(role::USER, "hi".to_string())];
        for (thinking, want) in [None, Some(true), Some(false)].into_iter().zip(want) {
            let mut kw = serde_json::Map::new();
            if let Some(t) = thinking {
                kw.insert("enable_thinking".to_string(), serde_json::Value::Bool(t));
            }
            let span = c.render_prompt_structured_with(&msgs, &[], &kw).unwrap();
            let prompt = tx(&span);
            assert!(
                prompt.ends_with(&format!("hi<|im_end|>\n{want}")),
                "{name}, enable_thinking {thinking:?}: {prompt:?}"
            );
            let opener = c.trailing_generation_prompt(&msgs, &[], &kw, &span);
            assert_eq!(
                tx(&span[span.len() - opener..]),
                want,
                "{name}, enable_thinking {thinking:?}: opener split"
            );
        }
    }
}

#[test]
fn atem_rendering_is_pinned() {
    let Some(model) = glimmer_path() else {
        eprintln!("SKIP: no Muse Glimmer bundle (set BASERT_TEST_GLIMMER)");
        return;
    };
    let c = AtemCodec::load(&model).expect("codec load");
    let tx = marked(&c, &["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"]);

    assert_eq!(
        tx(&c.render_message(role::USER, "Hello").unwrap()),
        "<|start|>user<|message|>Hello<|eot|>"
    );
    assert_eq!(
        tx(&c.render_message(role::SYSTEM, "Be brief.").unwrap()),
        "<|start|>system<|message|>Be brief.\n\nReasoning strength: high.\n\n\
         # Valid recipients: \"self\", \"user\".<|eot|>"
    );
    assert_eq!(
        tx(&c.render_message(role::ASSISTANT, "Hi!").unwrap()),
        "<|start|>assistant to=user<|message|>Hi!<|eot|>"
    );
    assert_eq!(
        tx(&c.generation_prefix(TurnState::AfterInput).unwrap()),
        "<|start|>assistant"
    );
    assert_eq!(
        tx(&c.render_tool_result("get_weather", "22C").unwrap()),
        "<|start|>tool get_weather<|message|><tool_output name=\"get_weather\">\n22C\n\
         </tool_output><|eot|>"
    );
    assert_eq!(
        tx(&c.render_assistant_with_tool_calls(
            "",
            &[("get_weather".into(), r#"{"city":"Paris"}"#.into())]
        )
        .unwrap()),
        "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n\
         <atem:invoke name=\"get_weather\">\n\
         <atem:parameter name=\"city\">Paris</atem:parameter>\n\
         </atem:invoke>\n</atem:function_calls><|eot|>"
    );

    let tool =
        r#"{"name": "weather.get", "description": "Get weather", "parameters": {}}"#.to_string();
    let sys = tx(&c.render_system_with_tools(None, &[tool]).unwrap());
    assert!(sys.starts_with(
        "<|start|>system<|message|>You are a helpful AI assistant.\n\
         Knowledge cutoff: 2026-01-04.\n\nReasoning strength: high.\n\n\
         In this environment you have access to a set of tools"
    ));
    assert!(sys.contains("// Tool metadata\n{\"name\": \"weather\", \"description\": \"\"}\n"));
    assert!(sys.contains(
        "// Function schemas\n{\"name\": \"weather.get\", \"description\": \"Get weather\", \
         \"parameters\": {}}"
    ));
    assert!(sys.ends_with(
        "# Valid recipients: \"self\", \"weather.*\", \"user\".<|eot|>"
    ));

    let mut z = c.channelizer();
    let mut stream = c.encode(" to=self");
    stream.extend(c.encode("<|message|>"));
    stream.extend(c.encode("I should answer plainly."));
    stream.extend(c.encode("<|eom|>"));
    stream.extend(c.encode("<|start|>"));
    stream.extend(c.encode("assistant to=user"));
    stream.extend(c.encode("<|message|>"));
    stream.extend(c.encode("Paris."));
    stream.extend(c.encode("<|eot|>"));
    let runs = z.split(&stream);
    let significant: Vec<(u32, String, bool)> = runs
        .iter()
        .filter(|r| !r.text.is_empty() || r.closes)
        .map(|r| (r.channel, c.decode(&r.text), r.closes))
        .collect();
    assert_eq!(
        significant,
        vec![
            (channel::REASONING, "I should answer plainly.".to_string(), true),
            (channel::TEXT, "Paris.".to_string(), true),
        ]
    );
    let rejoined: Vec<u32> = runs.iter().flat_map(|r| r.span.clone()).collect();
    assert_eq!(rejoined, stream);
}

#[test]
fn chatml_append_stability_property() {
    use superfluid_daemon::codec::project_block_text;
    use superfluid_daemon::wal::block_kind;
    let Some(model) = qwen_path() else {
        eprintln!("SKIP: no Qwen3 test model");
        return;
    };
    let c = ChatMlCodec::load(&model).expect("codec load");
    let tx = marked(&c, &["<|im_start|>", "<|im_end|>"]);
    let role_name = |r: u32| match r {
        role::SYSTEM => "system",
        role::USER => "user",
        _ => "assistant",
    };
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    let texts = [
        "Hello", "Fix the bug in main.rs", "Sure — here is a plan:\n1. read\n2. write",
        "naïve café 日本語 🙂", "```rust\nfn f() {}\n```", "", "trailing space ",
        "<tool_call>not really</tool_call>", "line\n\nbreaks\n",
    ];
    let payloads = [
        (block_kind::FILE_DIFF, r#"{"path":"src/x.rs","diff":"-a\n+b\n+c"}"#),
        (block_kind::ARTIFACT, r#"{"title":"notes","content":"alpha\nbeta"}"#),
        (block_kind::CITATION, r#"{"title":"Doc","url":"https://x/y","quote":"q"}"#),
        (block_kind::PROGRESS, r#"{"percent":10}"#),
        (block_kind::DIAGNOSTIC, r#"{"severity":"error","message":"E0308","path":"a.rs","line":3}"#),
        (block_kind::WORKSPACE_REF, r#"{"path":"src/lib.rs"}"#),
    ];
    for _case in 0..64 {
        let n = 1 + rnd(8) as usize;
        let mut spans: Vec<u32> = Vec::new();
        let mut text = String::new();
        for i in 0..n {
            let r = if i == 0 && rnd(2) == 0 {
                role::SYSTEM
            } else if rnd(2) == 0 {
                role::USER
            } else {
                role::ASSISTANT
            };
            if rnd(3) == 0 {
                let (kind, payload) = payloads[rnd(payloads.len() as u64) as usize];
                let span = c.render_block(r, kind, payload).unwrap();
                let projected = project_block_text(kind, payload).unwrap();
                if projected.is_empty() {
                    assert!(span.is_empty());
                    continue;
                }
                spans.extend(&span);
                text.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role_name(r), projected));
            } else {
                let t = texts[rnd(texts.len() as u64) as usize];
                let span = c.render_message(r, t).unwrap();
                spans.extend(&span);
                text.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role_name(r), t));
            }
        }
        assert_eq!(tx(&spans), text, "spans decode to the whole conversation");
        let quotes_marker = text.contains("<tool_call>") || text.contains("</tool_call>");
        if !quotes_marker {
            let whole = c.encode(&text);
            assert_eq!(whole, spans, "whole-render tokens == concatenated block spans");
        } else {
            let tool_open = c.encode("<tool_call>");
            let tool_close = c.encode("</tool_call>");
            assert!(
                !spans.contains(&tool_open[0]) && !spans.contains(&tool_close[0]),
                "a quoted <tool_call> in content leaked as a control id"
            );
        }
    }
}
