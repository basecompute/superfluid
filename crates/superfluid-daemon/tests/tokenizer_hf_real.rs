//! The daemon's built-in HF tokenizer against libbaseRT's tokenizer for the SAME checkpoint.

#![cfg(feature = "basert")]

use std::collections::BTreeSet;
use std::path::PathBuf;

use superfluid_engine::Tokenizer;

fn bundle() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_MODEL") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    [
        "models/Qwen3-0.6B-Q4_0.base",
        "models/Qwen3-0.6B-Q8.base",
        "models/Qwen3-0.6B-Q4_K_M.base",
    ]
    .iter()
    .map(|c| root.join(c))
    .find(|p| p.is_file())
}

fn hf_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SUPERFLUID_TEST_HF_TOKENIZER") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    std::fs::read_dir(snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("tokenizer.json").is_file())
}

fn looks_like_a_control_token(bytes: &[u8]) -> bool {
    bytes.len() >= 3
        && ((bytes[0] == b'<' && bytes[bytes.len() - 1] == b'>')
            || (bytes[0] == b'[' && bytes[bytes.len() - 1] == b']'))
}

#[test]
fn hf_tokenizer_matches_the_bundle_tokenizer_for_the_same_checkpoint() {
    let (Some(bundle), Some(hf)) = (bundle(), hf_dir()) else {
        eprintln!(
            "SKIP: need a Qwen3-0.6B .base under models/ and its HF snapshot in the HF cache"
        );
        return;
    };
    let a: Box<dyn Tokenizer> =
        Box::new(superfluid_engine_ffi::TokenizerHandle::load(&bundle).expect("bundle tokenizer"));
    let b: Box<dyn Tokenizer> =
        Box::new(superfluid_tokenizer_hf::HfTokenizer::load(&hf).expect("hf tokenizer"));

    assert_eq!(a.vocab_size(), b.vocab_size(), "id space (bundle vs hf)");

    let mut hard: Vec<(u32, Vec<u8>, Vec<u8>)> = Vec::new();
    let mut tolerated: Vec<(u32, &'static str, String)> = Vec::new();
    for t in 0..a.vocab_size() {
        let (x, y) = (a.token_bytes(t), b.token_bytes(t));
        if x == y {
            continue;
        }
        if x.is_empty() && looks_like_a_control_token(&y) {
            tolerated.push((
                t,
                "bundle: control, hf: text",
                String::from_utf8_lossy(&y).into_owned(),
            ));
        } else if y.is_empty() && looks_like_a_control_token(&x) {
            tolerated.push((
                t,
                "hf: control, bundle: text",
                String::from_utf8_lossy(&x).into_owned(),
            ));
        } else {
            hard.push((t, x, y));
        }
    }
    eprintln!("shape-only control deviations (tolerated): {tolerated:?}");
    assert!(
        hard.is_empty(),
        "{} ids decode differently; first: {:?}",
        hard.len(),
        &hard[..hard.len().min(5)]
    );
    assert!(
        tolerated.len() <= 8,
        "too many shape-only deviations: {tolerated:?}"
    );

    let corpus = [
        "hi",
        "Hello, world!",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "  leading spaces and\ttabs\n\nnewlines",
        "日本語のテキスト and émojis 🚀🔥",
        "def f(x):\n    return x * 2  # comment",
        "1234567890 3.14159 1e-9",
        "<think>\nreasoning\n</think>\n\nanswer",
        "unicode: naïve café — “quotes” …",
        "{\"name\": \"f\", \"arguments\": {\"a\": [1, 2, 3]}}",
    ];
    for s in corpus {
        assert_eq!(a.encode(s), b.encode(s), "encode {s:?}");
    }

    assert_eq!(a.eos_token(), b.eos_token(), "eos");
    match (a.bos_token(), b.bos_token()) {
        (x, Some(y)) => assert_eq!(x, Some(y), "bos"),
        (Some(x), None) => eprintln!("bundle names a nominal BOS {x} the checkpoint does not declare (inert: encode parity holds)"),
        (None, None) => {}
    }

    let sa: BTreeSet<(String, u32)> = a.special_tokens().into_iter().collect();
    let sb: BTreeSet<(String, u32)> = b.special_tokens().into_iter().collect();
    let missing: Vec<_> = sb.difference(&sa).collect();
    let extra: Vec<_> = sa.difference(&sb).collect();
    eprintln!("bundle-only special tokens (shape-based extras): {extra:?}");
    assert!(
        missing.is_empty(),
        "declared added tokens the bundle does not split on: {missing:?}"
    );
}

#[test]
fn content_encodes_the_same_through_the_hf_tokenizer_as_through_libbasert() {
    use superfluid_daemon::{BundleCodec, TextCodec};
    use std::sync::Arc;
    let (Some(bundle), Some(hf)) = (bundle(), hf_dir()) else {
        eprintln!("SKIP: need a Qwen3-0.6B .base under models/ and its HF snapshot in the HF cache");
        return;
    };
    let native = BundleCodec::from_tokenizer(Arc::new(superfluid_engine_ffi::TokenizerHandle::load(&bundle).expect("bundle tokenizer")));
    let hf_tok = Arc::new(superfluid_tokenizer_hf::HfTokenizer::load(&hf).expect("hf tokenizer"));
    let composed = BundleCodec::from_tokenizer(hf_tok.clone());
    let id = |text: &str| hf_tok.special_tokens().into_iter().find(|(s, _)| s == text).map(|(_, id)| id).expect("a Qwen3 marker");

    let clean: [&[(&str, bool)]; 3] = [
        &[("<|im_start|>user\n", false), ("hello there\n", true), ("<|im_end|>\n", false)],
        &[("<|im_start|>system\n", false), ("\nBe brief.  \n", true), ("<|im_end|>\n<|im_start|>assistant\n", false)],
        &[("<|im_start|>user\n", false), ("naïve café, 日本語, 🦀 and < | not a marker | >", true), ("<|im_end|>\n", false)],
    ];
    for pieces in clean {
        assert_eq!(composed.encode_pieces(pieces), native.encode_pieces(pieces), "{pieces:?}");
    }
    for text in ["plain words", "\n\nleading newlines", "trailing space ", ""] {
        assert_eq!(composed.encode_content(text), native.encode_content(text), "{text:?}");
    }

    for marker in ["<|im_end|>", "<|im_start|>", "<think>", "</think>", "<tool_call>", "<|endoftext|>"] {
        let text = format!("quote {marker} here");
        assert!(hf_tok.encode(&text).contains(&id(marker)), "{marker}: `encode` parses it");
        for (name, codec) in [("libbaseRT", &native), ("composed", &composed)] {
            let toks = codec.encode_content(&text);
            assert!(!toks.contains(&id(marker)), "{name}: {marker} is words in content: {toks:?}");
            assert_eq!(codec.decode(&toks), text, "{name}: the text is whole");
            let turn = codec.encode_pieces(&[("<|im_start|>user\n", false), (&text, true), ("<|im_end|>\n", false)]);
            assert_eq!(turn.iter().filter(|&&t| t == id("<|im_end|>")).count(), 1, "{name}: only the dialect's turn end");
            assert_eq!(turn.iter().filter(|&&t| t == id("<|im_start|>")).count(), 1, "{name}: only the dialect's opener");
        }
    }
}
