//! `TokenizerHandle`.

use std::ffi::CString;
use std::path::Path;

use crate::engine::{last_error, FfiError};
use crate::libbasert::sys;

pub struct TokenizerHandle {
    handle: sys::baseRT_model_t,
}

impl TokenizerHandle {
    pub fn load(model_path: &Path) -> Result<TokenizerHandle, FfiError> {
        crate::libbasert::load().map_err(FfiError::Load)?;
        let path = CString::new(model_path.to_string_lossy().as_bytes())
            .map_err(|_| FfiError::Load("NUL in model path".into()))?;
        // SAFETY: load copies the path; a null return carries the typed
        // error state.
        let handle = unsafe { sys::baseRT_load_tokenizer_only(path.as_ptr()) };
        if handle.is_null() {
            return Err(FfiError::Load(last_error()));
        }
        Ok(TokenizerHandle { handle })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_with(text, true)
    }

    pub fn encode_plain(&self, text: &str) -> Vec<u32> {
        self.encode_with(text, false)
    }

    pub fn encode_pieces(&self, pieces: &[(&str, bool)]) -> Vec<u32> {
        let mut texts: Vec<CString> = Vec::with_capacity(pieces.len());
        let mut plain: Vec<std::os::raw::c_int> = Vec::with_capacity(pieces.len());
        let mut bytes = 0usize;
        for (text, content) in pieces {
            let c = CString::new(*text).unwrap_or_else(|_| {
                let cleaned: String = text.chars().filter(|&ch| ch != '\0').collect();
                CString::new(cleaned).expect("NULs removed")
            });
            bytes += c.as_bytes().len();
            texts.push(c);
            plain.push(*content as std::os::raw::c_int);
        }
        let ptrs: Vec<*const std::os::raw::c_char> = texts.iter().map(|c| c.as_ptr()).collect();
        let mut out = vec![0u32; bytes + 8];
        // SAFETY: `ptrs` and `plain` outlive the call; `out` holds the cap given, and
        // encode truncates to it.
        let n = unsafe {
            sys::baseRT_encode_pieces(
                self.handle,
                ptrs.as_ptr(),
                plain.as_ptr(),
                ptrs.len() as std::os::raw::c_int,
                out.as_mut_ptr(),
                out.len() as std::os::raw::c_int,
            )
        };
        out.truncate(n.max(0) as usize);
        out
    }

    fn encode_with(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let Ok(c) = CString::new(text) else {
            let cleaned: String = text.chars().filter(|&ch| ch != '\0').collect();
            return self.encode_with(&cleaned, parse_special);
        };
        let mut out = vec![0u32; text.len() + 8];
        // SAFETY: out is sized generously (BPE never yields more tokens
        // than bytes + specials); encode truncates to the cap it is
        // given and returns the count.
        let n = unsafe {
            if parse_special {
                sys::baseRT_encode(self.handle, c.as_ptr(), out.as_mut_ptr(), out.len() as i32)
            } else {
                sys::baseRT_encode_plain(
                    self.handle,
                    c.as_ptr(),
                    out.as_mut_ptr(),
                    out.len() as i32,
                )
            }
        };
        out.truncate(n.max(0) as usize);
        out
    }

    pub fn chat_template_jinja(&self) -> String {
        // SAFETY: returns a thread-local C string valid until the next
        // call on this thread; copied immediately.
        let p = unsafe { sys::baseRT_chat_template_jinja(self.handle) };
        if p.is_null() {
            return String::new();
        }
        // SAFETY: p is a NUL-terminated C string from the engine.
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }

    pub fn special_tokens(&self) -> Vec<(String, u32)> {
        // SAFETY: count is a pure read on the loaded tokenizer.
        let n = unsafe { sys::baseRT_special_token_count(self.handle) };
        let mut out = Vec::with_capacity(n.max(0) as usize);
        for i in 0..n.max(0) {
            let mut id: u32 = u32::MAX;
            // SAFETY: i is in [0,n); the returned pointer is a thread-local
            // C string valid until the next call, copied immediately.
            let p = unsafe { sys::baseRT_special_token(self.handle, i, &mut id) };
            if p.is_null() {
                continue;
            }
            // SAFETY: p is a non-null NUL-terminated C string from the
            // engine, valid until the next call on this thread; copied now.
            let s = unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned();
            if !s.is_empty() {
                out.push((s, id));
            }
        }
        out
    }

    pub fn grammar_compiles_from_schema(&self, json_schema: &str) -> bool {
        let Ok(c) = CString::new(json_schema) else {
            return false;
        };
        // SAFETY: create returns NULL on error, else an owned grammar we
        // free immediately (we only care that it parsed).
        let g = unsafe { sys::baseRT_grammar_create_from_schema(self.handle, c.as_ptr()) };
        if g.is_null() {
            return false;
        }
        // SAFETY: g is a non-null grammar from create; free it once.
        unsafe { sys::baseRT_grammar_free(g) };
        true
    }

    pub fn grammar_compiles_from_structural_tag(&self, tag_json: &str) -> bool {
        let Ok(c) = CString::new(tag_json) else {
            return false;
        };
        // SAFETY: create returns NULL on error, else an owned grammar we
        // free immediately (we only care that it parsed).
        let g = unsafe { sys::baseRT_grammar_create_from_structural_tag(self.handle, c.as_ptr()) };
        if g.is_null() {
            return false;
        }
        // SAFETY: g is a non-null grammar from create; free it once.
        unsafe { sys::baseRT_grammar_free(g) };
        true
    }

    pub fn structural_tag_admits(&self, tag_json: &str, text: &str) -> bool {
        let Ok(c) = CString::new(tag_json) else { return false };
        // SAFETY: create returns NULL on error, else an owned grammar.
        let g = unsafe { sys::baseRT_grammar_create_from_structural_tag(self.handle, c.as_ptr()) };
        if g.is_null() {
            return false;
        }
        let toks = self.encode(text);
        let skip = usize::from(toks.first() == Some(&self.bos_token()));
        let mut ok = true;
        for t in toks.iter().skip(skip) {
            // SAFETY: g is a live grammar for this model's vocab.
            if unsafe { sys::baseRT_grammar_accept_token(g, *t) } == 0 {
                ok = false;
                break;
            }
        }
        // SAFETY: g came from create above; free it exactly once.
        unsafe { sys::baseRT_grammar_free(g) };
        ok
    }

    pub fn token_bytes(&self, token: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        // SAFETY: length-preserving stateless decode into a caller
        // buffer; returns the true length.
        let n = unsafe {
            sys::baseRT_decode_token_raw(
                self.handle,
                token,
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                buf.len() as i32,
            )
        };
        let n = n.max(0) as usize;
        if n > buf.len() {
            buf.resize(n, 0);
            // SAFETY: as above, now with the exact required capacity.
            let m = unsafe {
                sys::baseRT_decode_token_raw(
                    self.handle,
                    token,
                    buf.as_mut_ptr() as *mut std::os::raw::c_char,
                    buf.len() as i32,
                )
            };
            buf.truncate((m.max(0) as usize).min(buf.len()));
            return buf;
        }
        buf.truncate(n);
        buf
    }

    pub fn eos_token(&self) -> u32 {
        // SAFETY: handle is live.
        unsafe { sys::baseRT_eos_token_id(self.handle) }
    }

    pub fn vocab(&self) -> u32 {
        // SAFETY: handle is live; get_config is a const read of the
        // parsed metadata.
        unsafe { sys::baseRT_get_config(self.handle) }.vocab_size
    }

    pub fn bos_token(&self) -> u32 {
        // SAFETY: handle is live; const read.
        unsafe { sys::baseRT_bos_id(self.handle) }
    }
}

// SAFETY: the handle wraps immutable tables and the entry points used keep no per-call
// state, so concurrent `&self` calls are safe.
unsafe impl Send for TokenizerHandle {}
// SAFETY: as above — every method used is a const read of immutable
// tables.
unsafe impl Sync for TokenizerHandle {}

impl Drop for TokenizerHandle {
    fn drop(&mut self) {
        // SAFETY: dropped exactly once; free handles partial states.
        unsafe { sys::baseRT_free_model(self.handle) };
    }
}

impl superfluid_engine::Tokenizer for TokenizerHandle {
    fn encode(&self, text: &str) -> Vec<u32> {
        TokenizerHandle::encode(self, text)
    }
    fn encode_plain(&self, text: &str) -> Option<Vec<u32>> {
        Some(TokenizerHandle::encode_plain(self, text))
    }
    fn encode_pieces(&self, pieces: &[(&str, bool)]) -> Option<Vec<u32>> {
        Some(TokenizerHandle::encode_pieces(self, pieces))
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        TokenizerHandle::token_bytes(self, token)
    }
    fn vocab_size(&self) -> u32 {
        self.vocab()
    }
    fn special_tokens(&self) -> Vec<(String, u32)> {
        TokenizerHandle::special_tokens(self)
    }
    fn bos_token(&self) -> Option<u32> {
        let bos = TokenizerHandle::bos_token(self);
        (bos != u32::MAX).then_some(bos)
    }
    fn eos_token(&self) -> u32 {
        TokenizerHandle::eos_token(self)
    }
    fn chat_template_jinja(&self) -> String {
        TokenizerHandle::chat_template_jinja(self)
    }
}
