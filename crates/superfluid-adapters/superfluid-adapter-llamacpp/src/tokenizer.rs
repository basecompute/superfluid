//! `LlamaTokenizer`.

use std::ffi::{CStr, CString};
use std::path::Path;

use superfluid_engine::Tokenizer;
use crate::sys;

use crate::runtime::LlamaError;

pub struct LlamaTokenizer {
    model: *mut sys::llama_model,
    vocab: *const sys::llama_vocab,
    n_tokens: u32,
}

// SAFETY: a vocab-only model is an immutable table; every call here is a
// const read (`llama_tokenize`, `llama_token_to_piece`, attribute and id
// queries), so concurrent `&self` use is safe.
unsafe impl Send for LlamaTokenizer {}
// SAFETY: as above.
unsafe impl Sync for LlamaTokenizer {}

impl LlamaTokenizer {
    pub fn load(path: &Path) -> Result<LlamaTokenizer, LlamaError> {
        sys::load().map_err(LlamaError::Runtime)?;
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| LlamaError::Config("NUL in model path"))?;
        sys::clear_errors();
        // SAFETY: plain FFI; vocab_only skips every tensor.
        let model = unsafe {
            sys::llama_backend_init();
            let mut mp = sys::llama_model_default_params();
            mp.vocab_only = true;
            sys::llama_model_load_from_file(c.as_ptr(), mp)
        };
        if model.is_null() {
            return Err(LlamaError::Load(crate::runtime::with_reason(path.display().to_string())));
        }
        // SAFETY: model is live.
        let vocab = unsafe { sys::llama_model_get_vocab(model) };
        // SAFETY: vocab is live.
        let n_tokens = unsafe { sys::llama_vocab_n_tokens(vocab) } as u32;
        Ok(LlamaTokenizer { model, vocab, n_tokens })
    }

    pub(crate) fn model(&self) -> *const sys::llama_model {
        self.model
    }

    fn attr(&self, id: u32) -> u32 {
        // SAFETY: vocab is live; id < n_tokens is checked by callers.
        unsafe { sys::llama_vocab_get_attr(self.vocab, id as sys::llama_token) as u32 }
    }
}

impl Drop for LlamaTokenizer {
    fn drop(&mut self) {
        // SAFETY: owned, freed once.
        unsafe { sys::llama_model_free(self.model) };
    }
}

const TOKEN_NULL: sys::llama_token = -1;
const CONTROL: u32 = sys::LLAMA_TOKEN_ATTR_CONTROL;
const USER_DEFINED: u32 = sys::LLAMA_TOKEN_ATTR_USER_DEFINED;
const UNUSED: u32 = sys::LLAMA_TOKEN_ATTR_UNUSED;

impl Tokenizer for LlamaTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut cap = bytes.len() + 8;
        loop {
            let mut out = vec![0 as sys::llama_token; cap];
            // SAFETY: text is a live slice (length passed, no NUL needed);
            // out has `cap` slots. add_special = the family's BOS policy;
            // parse_special = special-token strings encode to their ids.
            let n = unsafe {
                sys::llama_tokenize(
                    self.vocab,
                    bytes.as_ptr() as *const _,
                    bytes.len() as i32,
                    out.as_mut_ptr(),
                    cap as i32,
                    true,
                    true,
                )
            };
            if n < 0 {
                cap = (-n) as usize;
                continue;
            }
            out.truncate(n as usize);
            return out.into_iter().map(|t| t as u32).collect();
        }
    }

    fn token_bytes(&self, token: u32) -> Vec<u8> {
        if token >= self.n_tokens {
            return Vec::new();
        }
        let attr = self.attr(token);
        if attr & (CONTROL | UNUSED) != 0 {
            return Vec::new();
        }
        let mut cap = 64usize;
        loop {
            let mut buf = vec![0u8; cap];
            // SAFETY: buf has `cap` bytes; a negative return is the size needed.
            let n = unsafe {
                sys::llama_token_to_piece(self.vocab, token as sys::llama_token, buf.as_mut_ptr() as *mut _, cap as i32, 0, true)
            };
            if n < 0 {
                cap = (-n) as usize;
                continue;
            }
            buf.truncate(n as usize);
            return buf;
        }
    }

    fn vocab_size(&self) -> u32 {
        self.n_tokens
    }

    fn special_tokens(&self) -> Vec<(String, u32)> {
        (0..self.n_tokens)
            .filter(|&id| self.attr(id) & (CONTROL | USER_DEFINED) != 0)
            .filter_map(|id| {
                // SAFETY: vocab is live; the text is a NUL-terminated C string it owns.
                let p = unsafe { sys::llama_vocab_get_text(self.vocab, id as sys::llama_token) };
                if p.is_null() {
                    return None;
                }
                // SAFETY: as above.
                let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
                (!s.is_empty()).then_some((s, id))
            })
            .collect()
    }

    fn bos_token(&self) -> Option<u32> {
        // SAFETY: vocab is live.
        let b = unsafe { sys::llama_vocab_bos(self.vocab) };
        (b != TOKEN_NULL).then_some(b as u32)
    }

    fn eos_token(&self) -> u32 {
        // SAFETY: vocab is live.
        let e = unsafe { sys::llama_vocab_eos(self.vocab) };
        if e == TOKEN_NULL { 0 } else { e as u32 }
    }

    fn chat_template_jinja(&self) -> String {
        // SAFETY: model is live; the template string is model-owned.
        let p = unsafe { sys::llama_model_chat_template(self.model, std::ptr::null()) };
        if p.is_null() {
            return String::new();
        }
        // SAFETY: as above.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    fn chat_template_named(&self, name: &str) -> Option<String> {
        let name = CString::new(name).ok()?;
        // SAFETY: model is live, the name NUL-terminated; a template string
        // is model-owned, and there is none of that name when NULL.
        let p = unsafe { sys::llama_model_chat_template(self.model, name.as_ptr()) };
        if p.is_null() {
            return None;
        }
        // SAFETY: as above.
        let template = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        (!template.is_empty()).then_some(template)
    }
}
