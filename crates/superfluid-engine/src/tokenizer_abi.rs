//! A [`Tokenizer`] across a shared-library boundary.

use std::ffi::{c_char, c_void, CStr};
use std::path::Path;

use crate::Tokenizer;

pub const ABI_VERSION: u32 = 1;

pub mod symbol {
    pub const ABI_VERSION: &[u8] = b"superfluid_tokenizer_abi_version\0";
    pub const OPEN: &[u8] = b"superfluid_tokenizer_open\0";
    pub const CLOSE: &[u8] = b"superfluid_tokenizer_close\0";
    pub const ENCODE: &[u8] = b"superfluid_tokenizer_encode\0";
    pub const TOKEN_BYTES: &[u8] = b"superfluid_tokenizer_token_bytes\0";
    pub const VOCAB_SIZE: &[u8] = b"superfluid_tokenizer_vocab_size\0";
    pub const SPECIAL_TOKENS: &[u8] = b"superfluid_tokenizer_special_tokens\0";
    pub const BOS: &[u8] = b"superfluid_tokenizer_bos\0";
    pub const EOS: &[u8] = b"superfluid_tokenizer_eos\0";
    pub const CHAT_TEMPLATE: &[u8] = b"superfluid_tokenizer_chat_template\0";
}

pub type AbiVersionFn = unsafe extern "C" fn() -> u32;
pub type OpenFn = unsafe extern "C" fn(path: *const c_char, err: *mut c_char, err_len: usize) -> *mut c_void;
pub type CloseFn = unsafe extern "C" fn(tok: *mut c_void);
pub type EncodeFn = unsafe extern "C" fn(tok: *const c_void, text: *const u8, len: usize, out: *mut u32, cap: usize) -> usize;
pub type TokenBytesFn = unsafe extern "C" fn(tok: *const c_void, id: u32, out: *mut u8, cap: usize) -> usize;
pub type VocabSizeFn = unsafe extern "C" fn(tok: *const c_void) -> u32;
pub type SpecialTokensFn = unsafe extern "C" fn(tok: *const c_void, out: *mut u8, cap: usize) -> usize;
pub type BosFn = unsafe extern "C" fn(tok: *const c_void) -> i64;
pub type EosFn = unsafe extern "C" fn(tok: *const c_void) -> u32;
pub type ChatTemplateFn = unsafe extern "C" fn(tok: *const c_void, name: *const c_char, out: *mut u8, cap: usize) -> usize;

pub fn pack_special(tokens: &[(String, u32)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (text, id) in tokens {
        out.extend_from_slice(&id.to_le_bytes());
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    out
}

pub fn unpack_special(mut bytes: &[u8]) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    while bytes.len() >= 8 {
        let id = u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes"));
        let len = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes")) as usize;
        let Some(text) = bytes.get(8..8 + len) else { break };
        out.push((String::from_utf8_lossy(text).into_owned(), id));
        bytes = &bytes[8 + len..];
    }
    out
}

#[doc(hidden)]
pub mod export {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    type Boxed = Box<dyn Tokenizer>;

    fn fill(bytes: &[u8], out: *mut u8, cap: usize) -> usize {
        let n = bytes.len().min(cap);
        if n > 0 && !out.is_null() {
            // SAFETY: the caller's buffer holds `cap` bytes; `n <= cap`.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, n) };
        }
        bytes.len()
    }

    fn write_err(err: *mut c_char, err_len: usize, msg: &str) {
        if err.is_null() || err_len == 0 {
            return;
        }
        let n = msg.len().min(err_len - 1);
        // SAFETY: `err` holds `err_len` bytes; `n + 1 <= err_len`.
        unsafe {
            std::ptr::copy_nonoverlapping(msg.as_ptr(), err.cast::<u8>(), n);
            *err.add(n) = 0;
        }
    }

    /// # Safety
    /// `tok` is a live handle [`open`] returned, or NULL.
    unsafe fn tok<'a>(tok: *const c_void) -> Option<&'a dyn Tokenizer> {
        // SAFETY: per this function's contract; NULL is refused.
        (!tok.is_null()).then(|| unsafe { &**(tok as *const Boxed) })
    }

    /// # Safety
    /// `path` is NUL-terminated; `err` holds `err_len` bytes (or is NULL).
    pub unsafe fn open<T, E>(
        path: *const c_char,
        err: *mut c_char,
        err_len: usize,
        load: impl Fn(&Path) -> Result<T, E>,
    ) -> *mut c_void
    where
        T: Tokenizer + 'static,
        E: std::fmt::Display,
    {
        if path.is_null() {
            write_err(err, err_len, "no path");
            return std::ptr::null_mut();
        }
        // SAFETY: per this function's contract.
        let path = unsafe { CStr::from_ptr(path) }.to_string_lossy().into_owned();
        match catch_unwind(AssertUnwindSafe(|| load(Path::new(&path)))) {
            Ok(Ok(t)) => Box::into_raw(Box::new(Box::new(t) as Boxed)) as *mut c_void,
            Ok(Err(e)) => {
                write_err(err, err_len, &e.to_string());
                std::ptr::null_mut()
            }
            Err(_) => {
                write_err(err, err_len, "the tokenizer panicked while loading");
                std::ptr::null_mut()
            }
        }
    }

    /// # Safety
    /// `t` is a live handle [`open`] returned, released once.
    pub unsafe fn close(t: *mut c_void) {
        if !t.is_null() {
            // SAFETY: per this function's contract.
            let boxed = unsafe { Box::from_raw(t as *mut Boxed) };
            let _ = catch_unwind(AssertUnwindSafe(|| drop(boxed)));
        }
    }

    /// # Safety
    /// `t` is live; `text` holds `len` bytes; `out` holds `cap` ids.
    pub unsafe fn encode(t: *const c_void, text: *const u8, len: usize, out: *mut u32, cap: usize) -> usize {
        catch_unwind(AssertUnwindSafe(|| {
            let bytes = if len == 0 {
                &[][..]
            } else {
                // SAFETY: per this function's contract.
                unsafe { std::slice::from_raw_parts(text, len) }
            };
            // SAFETY: per this function's contract.
            let Some(tk) = (unsafe { tok(t) }) else { return 0 };
            let ids = tk.encode(&String::from_utf8_lossy(bytes));
            let n = ids.len().min(cap);
            if n > 0 && !out.is_null() {
                // SAFETY: `out` holds `cap` ids; `n <= cap`.
                unsafe { std::ptr::copy_nonoverlapping(ids.as_ptr(), out, n) };
            }
            ids.len()
        }))
        .unwrap_or(0)
    }

    /// # Safety
    /// `t` is live; `out` holds `cap` bytes.
    pub unsafe fn token_bytes(t: *const c_void, id: u32, out: *mut u8, cap: usize) -> usize {
        // SAFETY: per this function's contract.
        catch_unwind(AssertUnwindSafe(|| unsafe { tok(t) }.map_or(0, |tk| fill(&tk.token_bytes(id), out, cap))))
            .unwrap_or(0)
    }

    /// # Safety
    /// `t` is live.
    pub unsafe fn vocab_size(t: *const c_void) -> u32 {
        // SAFETY: per this function's contract.
        catch_unwind(AssertUnwindSafe(|| unsafe { tok(t) }.map_or(0, |tk| tk.vocab_size()))).unwrap_or(0)
    }

    /// # Safety
    /// `t` is live; `out` holds `cap` bytes.
    pub unsafe fn special_tokens(t: *const c_void, out: *mut u8, cap: usize) -> usize {
        // SAFETY: per this function's contract.
        catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: per this function's contract.
            unsafe { tok(t) }.map_or(0, |tk| fill(&pack_special(&tk.special_tokens()), out, cap))
        }))
        .unwrap_or(0)
    }

    /// # Safety
    /// `t` is live.
    pub unsafe fn bos(t: *const c_void) -> i64 {
        // SAFETY: per this function's contract.
        catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: per this function's contract.
            unsafe { tok(t) }.and_then(|tk| tk.bos_token()).map_or(-1, i64::from)
        }))
        .unwrap_or(-1)
    }

    /// # Safety
    /// `t` is live.
    pub unsafe fn eos(t: *const c_void) -> u32 {
        // SAFETY: per this function's contract.
        catch_unwind(AssertUnwindSafe(|| unsafe { tok(t) }.map_or(0, |tk| tk.eos_token()))).unwrap_or(0)
    }

    /// # Safety
    /// `t` is live; `name` is NUL-terminated or NULL; `out` holds `cap` bytes.
    pub unsafe fn chat_template(t: *const c_void, name: *const c_char, out: *mut u8, cap: usize) -> usize {
        catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: per this function's contract.
            let Some(tk) = (unsafe { tok(t) }) else { return usize::MAX };
            let template = if name.is_null() {
                Some(tk.chat_template_jinja())
            } else {
                // SAFETY: per this function's contract.
                tk.chat_template_named(&unsafe { CStr::from_ptr(name) }.to_string_lossy())
            };
            match template {
                Some(s) => fill(s.as_bytes(), out, cap),
                None => usize::MAX,
            }
        }))
        .unwrap_or(usize::MAX)
    }
}

#[macro_export]
macro_rules! export_tokenizer {
    ($load:expr) => {
        #[no_mangle]
        pub extern "C" fn superfluid_tokenizer_abi_version() -> u32 {
            $crate::tokenizer_abi::ABI_VERSION
        }

        /// # Safety
        /// See `superfluid_engine::tokenizer_abi::OpenFn`.
        #[no_mangle]
        pub unsafe extern "C" fn superfluid_tokenizer_open(
            path: *const ::std::ffi::c_char,
            err: *mut ::std::ffi::c_char,
            err_len: usize,
        ) -> *mut ::std::ffi::c_void {
            // SAFETY: forwarded under the same contract.
            unsafe { $crate::tokenizer_abi::export::open(path, err, err_len, $load) }
        }

        /// # Safety
        /// See `superfluid_engine::tokenizer_abi::CloseFn`.
        #[no_mangle]
        pub unsafe extern "C" fn superfluid_tokenizer_close(tok: *mut ::std::ffi::c_void) {
            // SAFETY: forwarded under the same contract.
            unsafe { $crate::tokenizer_abi::export::close(tok) }
        }

        /// # Safety
        /// See `superfluid_engine::tokenizer_abi::EncodeFn`.
        #[no_mangle]
        pub unsafe extern "C" fn superfluid_tokenizer_encode(
            tok: *const ::std::ffi::c_void,
            text: *const u8,
            len: usize,
            out: *mut u32,
            cap: usize,
        ) -> usize {
            // SAFETY: forwarded under the same contract.
            unsafe { $crate::tokenizer_abi::export::encode(tok, text, len, out, cap) }
        }

        /// # Safety
        /// See `superfluid_engine::tokenizer_abi::TokenBytesFn`.
        #[no_mangle]
        pub unsafe extern "C" fn superfluid_tokenizer_token_bytes(
            tok: *const ::std::ffi::c_void,
            id: u32,
            out: *mut u8,
            cap: usize,
        ) -> usize {
            // SAFETY: forwarded under the same contract.
            unsafe { $crate::tokenizer_abi::export::token_bytes(tok, id, out, cap) }
        }

        /// # Safety
        /// See `superfluid_engine::tokenizer_abi::VocabSizeFn`.
        #[no_mangle]
        pub unsafe extern "C" fn superfluid_tokenizer_vocab_size(tok: *const ::std::ffi::c_void) -> u32 {
            // SAFETY: forwarded under the same contract.
            unsafe { $crate::tokenizer_abi::export::vocab_size(tok) }
        }

        /// # Safety
        /// See `superfluid_engine::tokenizer_abi::SpecialTokensFn`.
        #[no_mangle]
        pub unsafe extern "C" fn superfluid_tokenizer_special_tokens(
            tok: *const ::std::ffi::c_void,
            out: *mut u8,
            cap: usize,
        ) -> usize {
            // SAFETY: forwarded under the same contract.
            unsafe { $crate::tokenizer_abi::export::special_tokens(tok, out, cap) }
        }

        /// # Safety
        /// See `superfluid_engine::tokenizer_abi::BosFn`.
        #[no_mangle]
        pub unsafe extern "C" fn superfluid_tokenizer_bos(tok: *const ::std::ffi::c_void) -> i64 {
            // SAFETY: forwarded under the same contract.
            unsafe { $crate::tokenizer_abi::export::bos(tok) }
        }

        /// # Safety
        /// See `superfluid_engine::tokenizer_abi::EosFn`.
        #[no_mangle]
        pub unsafe extern "C" fn superfluid_tokenizer_eos(tok: *const ::std::ffi::c_void) -> u32 {
            // SAFETY: forwarded under the same contract.
            unsafe { $crate::tokenizer_abi::export::eos(tok) }
        }

        /// # Safety
        /// See `superfluid_engine::tokenizer_abi::ChatTemplateFn`.
        #[no_mangle]
        pub unsafe extern "C" fn superfluid_tokenizer_chat_template(
            tok: *const ::std::ffi::c_void,
            name: *const ::std::ffi::c_char,
            out: *mut u8,
            cap: usize,
        ) -> usize {
            // SAFETY: forwarded under the same contract.
            unsafe { $crate::tokenizer_abi::export::chat_template(tok, name, out, cap) }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PanicsOnDrop;

    impl Tokenizer for PanicsOnDrop {
        fn encode(&self, _text: &str) -> Vec<u32> {
            Vec::new()
        }
        fn token_bytes(&self, _token: u32) -> Vec<u8> {
            Vec::new()
        }
        fn vocab_size(&self) -> u32 {
            1
        }
        fn special_tokens(&self) -> Vec<(String, u32)> {
            Vec::new()
        }
        fn bos_token(&self) -> Option<u32> {
            None
        }
        fn eos_token(&self) -> u32 {
            0
        }
        fn chat_template_jinja(&self) -> String {
            String::new()
        }
    }

    impl Drop for PanicsOnDrop {
        fn drop(&mut self) {
            panic!("a tokenizer that panics on drop");
        }
    }

    #[test]
    fn closing_a_tokenizer_whose_drop_panics_does_not_unwind_into_the_caller() {
        // SAFETY: a NUL-terminated path and no error buffer; the handle is
        // closed once.
        unsafe {
            let h = export::open(c"unused".as_ptr(), std::ptr::null_mut(), 0, |_| Ok::<_, String>(PanicsOnDrop));
            assert!(!h.is_null());
            assert_eq!(export::vocab_size(h), 1);
            export::close(h);
        }
    }

    #[test]
    fn a_null_handle_answers_each_call_with_its_failure_value() {
        let null = std::ptr::null();
        // SAFETY: every call is handed NULL, which each refuses.
        unsafe {
            assert_eq!(export::encode(null, b"hi".as_ptr(), 2, std::ptr::null_mut(), 0), 0);
            assert_eq!(export::token_bytes(null, 1, std::ptr::null_mut(), 0), 0);
            assert_eq!(export::vocab_size(null), 0);
            assert_eq!(export::special_tokens(null, std::ptr::null_mut(), 0), 0);
            assert_eq!(export::bos(null), -1);
            assert_eq!(export::eos(null), 0);
            assert_eq!(export::chat_template(null, std::ptr::null(), std::ptr::null_mut(), 0), usize::MAX);
            export::close(std::ptr::null_mut());
        }
    }

    #[test]
    fn special_tokens_pack_and_unpack() {
        let t = vec![("<|im_start|>".to_string(), 151644), ("".to_string(), 7), ("é".to_string(), 3)];
        assert_eq!(unpack_special(&pack_special(&t)), t);
        let packed = pack_special(&t);
        assert_eq!(unpack_special(&packed[..packed.len() - 1]).len(), 2, "a truncated tail is dropped");
    }
}
