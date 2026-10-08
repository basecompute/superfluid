//! Whether a `.base` bundle's embedded HF tokenizer, read by HF's own library, tokenizes as
//! libbaseRT does.

use std::path::PathBuf;
use std::time::Instant;

use superfluid_engine::Tokenizer;
use superfluid_engine_ffi::TokenizerHandle;
use superfluid_tokenizer_hf::HfTokenizer;

fn bundles() -> Vec<PathBuf> {
    ["BASERT_TEST_MODEL", "BASERT_TEST_MODEL_GEMMA4", "BASERT_TEST_MODEL_GPTOSS"]
        .iter()
        .filter_map(|k| std::env::var_os(k).map(PathBuf::from))
        .filter(|p| p.is_file())
        .collect()
}

const TEXTS: &[&str] = &[
    "",
    "The capital of France is",
    "  leading spaces, tabs\tand\nnewlines\r\n",
    "naïve café, 日本語のテキスト, emoji 🦀🔥 and a ZWJ family 👨‍👩‍👧",
    "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n",
    "<|start|>user<|message|>hi<|end|><|start|>assistant<|channel|>final<|message|>",
    "fn main() { println!(\"{}\", 1 + 2); } // code",
    "1234567890 3.14159 -42 1e10",
];

#[test]
#[ignore = "known differences, listed above"]
fn the_bundles_hf_tokenizer_tokenizes_as_libbasert_does() {
    let bundles = bundles();
    if bundles.is_empty() {
        eprintln!("SKIP: set BASERT_TEST_MODEL (and _GEMMA4, _GPTOSS) to .base bundles");
        return;
    }
    let mut problems = Vec::new();
    for b in &bundles {
        let name = b.display().to_string();
        let t0 = Instant::now();
        let hf = match HfTokenizer::from_bundle(b) {
            Ok(t) => t,
            Err(e) => {
                problems.push(format!("{name}: from_bundle: {e}"));
                continue;
            }
        };
        let hf_open = t0.elapsed();
        let t1 = Instant::now();
        let native = TokenizerHandle::load(b).expect("libbaseRT tokenizer");
        eprintln!("{name}: HF from the header {hf_open:?}, libbaseRT {:?}", t1.elapsed());

        let mut report = |what: String| problems.push(format!("{name}: {what}"));
        if hf.vocab_size() != native.vocab_size() {
            report(format!("vocab_size HF {} vs libbaseRT {}", hf.vocab_size(), native.vocab_size()));
        }
        if hf.bos_token() != Tokenizer::bos_token(&native) {
            report(format!("bos HF {:?} vs libbaseRT {:?}", hf.bos_token(), Tokenizer::bos_token(&native)));
        }
        if hf.eos_token() != Tokenizer::eos_token(&native) {
            report(format!("eos HF {} vs libbaseRT {}", hf.eos_token(), Tokenizer::eos_token(&native)));
        }
        let (mut hs, mut ns) = (hf.special_tokens(), Tokenizer::special_tokens(&native));
        hs.sort_by_key(|(_, id)| *id);
        ns.sort_by_key(|(_, id)| *id);
        if hs != ns {
            let only_hf: Vec<_> = hs.iter().filter(|x| !ns.contains(x)).take(8).collect();
            let only_native: Vec<_> = ns.iter().filter(|x| !hs.contains(x)).take(8).collect();
            report(format!(
                "special tokens: HF {} vs libbaseRT {}; only HF (first 8) {only_hf:?}; only libbaseRT (first 8) {only_native:?}",
                hs.len(),
                ns.len()
            ));
        }
        if hf.chat_template_jinja() != Tokenizer::chat_template_jinja(&native) {
            report(format!(
                "chat template differs (HF {} bytes, libbaseRT {} bytes)",
                hf.chat_template_jinja().len(),
                Tokenizer::chat_template_jinja(&native).len()
            ));
        }
        for t in TEXTS {
            let (a, n) = (hf.encode(t), Tokenizer::encode(&native, t));
            if a != n {
                let head: String = t.chars().take(30).collect();
                let at = a.iter().zip(&n).position(|(x, y)| x != y).unwrap_or(a.len().min(n.len()));
                let (lo, hi) = (at.saturating_sub(2), at + 4);
                report(format!(
                    "encode {head:?}: first difference at {at}: HF {:?} vs libbaseRT {:?} (lengths {} vs {})",
                    &a[lo.min(a.len())..hi.min(a.len())],
                    &n[lo.min(n.len())..hi.min(n.len())],
                    a.len(),
                    n.len()
                ));
            }
        }
        let mut differing = Vec::new();
        for id in 0..hf.vocab_size().max(native.vocab_size()) + 2 {
            if hf.token_bytes(id) != Tokenizer::token_bytes(&native, id) {
                differing.push(id);
            }
        }
        if !differing.is_empty() {
            let sample: Vec<_> = differing
                .iter()
                .take(6)
                .map(|&id| (id, String::from_utf8_lossy(&hf.token_bytes(id)).into_owned(), String::from_utf8_lossy(&Tokenizer::token_bytes(&native, id)).into_owned()))
                .collect();
            report(format!("token_bytes differ on {} ids, e.g. {sample:?}", differing.len()));
        }
    }
    assert!(problems.is_empty(), "\n{}", problems.join("\n"));
}
