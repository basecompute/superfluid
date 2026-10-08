//! The llama.cpp package's tokenizer library tokenizes a GGUF exactly as llama.cpp linked into this
//! build does.

#![cfg(feature = "llamacpp")]

use std::path::{Path, PathBuf};

use superfluid_adapter_llamacpp::LlamaTokenizer;
use superfluid_daemon::dylib_tokenizer::{library_file, DylibTokenizer};
use superfluid_engine::Tokenizer;

fn library() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_BIN_EXE_superfluid")).with_file_name(library_file("superfluid_tokenizer_llamacpp"));
    p.is_file().then_some(p)
}

#[test]
fn the_library_tokenizes_as_the_linked_llama_cpp_does() {
    let Some(lib) = library() else {
        eprintln!("SKIP: build superfluid-tokenizer-llamacpp first");
        return;
    };
    let Some(model) = std::env::var_os("SUPERFLUID_TEST_GGUF").map(PathBuf::from).filter(|p| p.is_file()) else {
        eprintln!("SKIP: set SUPERFLUID_TEST_GGUF");
        return;
    };
    let t0 = std::time::Instant::now();
    let loaded = DylibTokenizer::open(&lib, &model).expect("the library opens the GGUF's tokenizer");
    eprintln!("library open: {:?}", t0.elapsed());
    let t1 = std::time::Instant::now();
    let linked = LlamaTokenizer::load(&model).expect("linked tokenizer");
    eprintln!("linked open: {:?}", t1.elapsed());
    let t2 = std::time::Instant::now();
    let again = DylibTokenizer::open(&lib, &model).expect("a second open");
    eprintln!("library open, second time: {:?}", t2.elapsed());
    drop(again);

    assert_eq!(loaded.vocab_size(), linked.vocab_size());
    assert_eq!(loaded.bos_token(), linked.bos_token());
    assert_eq!(loaded.eos_token(), linked.eos_token());
    assert_eq!(loaded.special_tokens(), linked.special_tokens());
    assert_eq!(loaded.chat_template_jinja(), linked.chat_template_jinja());
    assert!(!loaded.chat_template_jinja().is_empty(), "a chat model carries a template");
    assert_eq!(loaded.chat_template_named("tool_use"), linked.chat_template_named("tool_use"));

    let texts = [
        "",
        "The capital of France is",
        "  leading spaces, tabs\tand\nnewlines\r\n",
        "naïve café, 日本語のテキスト, emoji 🦀🔥 and a ZWJ family 👨‍👩‍👧",
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "fn main() { println!(\"{}\", 1 + 2); } // code",
        &"long ".repeat(3000),
    ];
    for t in texts {
        assert_eq!(loaded.encode(t), linked.encode(t), "encode {:?}", &t[..t.len().min(40)]);
    }
    let t0 = std::time::Instant::now();
    for id in 0..linked.vocab_size() + 4 {
        assert_eq!(loaded.token_bytes(id), linked.token_bytes(id), "token {id}");
    }
    eprintln!("token_bytes over the vocabulary through the library: {:?}", t0.elapsed());
    let e = DylibTokenizer::open(&lib, Path::new("/nonexistent/model.gguf")).err().expect("refused");
    assert!(e.starts_with("tokenizer load failed for /nonexistent/model.gguf"), "{e}");
    let e = DylibTokenizer::open(Path::new("/usr/lib/libz.dylib"), &model).err().expect("refused");
    assert!(e.contains("is not a superfluid tokenizer library") || e.contains("could not load"), "{e}");
}

#[test]
fn quoted_markers_in_content_are_words_on_llama_cpps_tokenizer() {
    use superfluid_daemon::{BundleCodec, TextCodec};
    use std::sync::Arc;
    let Some(lib) = library() else {
        eprintln!("SKIP: build superfluid-tokenizer-llamacpp first");
        return;
    };
    let Some(model) = std::env::var_os("SUPERFLUID_TEST_GGUF").map(PathBuf::from).filter(|p| p.is_file()) else {
        eprintln!("SKIP: set SUPERFLUID_TEST_GGUF");
        return;
    };
    let linked: Arc<dyn Tokenizer> = Arc::new(LlamaTokenizer::load(&model).expect("linked tokenizer"));
    let loaded: Arc<dyn Tokenizer> = Arc::new(DylibTokenizer::open(&lib, &model).expect("the library opens the GGUF's tokenizer"));
    let specials = linked.special_tokens();
    let markers: Vec<(String, u32)> = ["<|im_end|>", "<|im_start|>", "<think>", "</think>", "<tool_call>"]
        .iter()
        .filter_map(|m| specials.iter().find(|(s, _)| s == m).cloned())
        .collect();
    assert!(markers.len() >= 2, "a chat GGUF declares its markers: {markers:?}");
    for (name, tok) in [("linked", linked), ("library", loaded)] {
        let codec = BundleCodec::from_tokenizer(Arc::clone(&tok));
        let turn: &[(&str, bool)] = &[("<|im_start|>user\n", false), ("hello there\n", true), ("<|im_end|>\n", false)];
        assert_eq!(codec.encode_pieces(turn), tok.encode("<|im_start|>user\nhello there\n<|im_end|>\n"), "{name}: one encode");
        assert_eq!(codec.encode_content("plain words"), tok.encode("plain words"), "{name}");
        for (marker, id) in &markers {
            let text = format!("quote {marker} here");
            assert!(tok.encode(&text).contains(id), "{name}: `encode` parses {marker}");
            let toks = codec.encode_content(&text);
            assert!(!toks.contains(id), "{name}: {marker} is words in content: {toks:?}");
            let bos = tok.bos_token();
            let body: Vec<u32> = toks.iter().copied().filter(|t| Some(*t) != bos).collect();
            assert_eq!(codec.decode(&body), text, "{name}: the text is whole");
        }
    }
}
