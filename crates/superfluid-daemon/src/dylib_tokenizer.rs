//! A runtime's tokenizer, loaded from its package at run time.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};

use superfluid_engine::tokenizer_abi::{self as abi, symbol};
use superfluid_engine::Tokenizer;

struct Fns {
    close: abi::CloseFn,
    encode: abi::EncodeFn,
    token_bytes: abi::TokenBytesFn,
    vocab_size: abi::VocabSizeFn,
    special_tokens: abi::SpecialTokensFn,
    bos: abi::BosFn,
    eos: abi::EosFn,
    chat_template: abi::ChatTemplateFn,
}

pub struct DylibTokenizer {
    tok: *mut c_void,
    f: Fns,
    library: PathBuf,
}

// SAFETY: the interface's tokenizers are immutable once open — every call
// but `close` is a read, safe from any thread concurrently (the contract
// `superfluid_engine::Tokenizer: Send + Sync` states for every tokenizer).
unsafe impl Send for DylibTokenizer {}
// SAFETY: as above.
unsafe impl Sync for DylibTokenizer {}

fn dl_error() -> String {
    // SAFETY: dlerror returns NULL or a NUL-terminated message owned by the
    // loader, valid until the next dl* call on this thread.
    let e = unsafe { libc::dlerror() };
    if e.is_null() {
        "unknown loader error".into()
    } else {
        // SAFETY: as above.
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

impl DylibTokenizer {
    pub fn open(library: &Path, model: &Path) -> Result<DylibTokenizer, String> {
        let lib_c = CString::new(library.to_string_lossy().as_bytes()).map_err(|_| "NUL in library path".to_string())?;
        // SAFETY: a NUL-terminated path; RTLD_LOCAL keeps the library's
        // symbols (its own llama.cpp) out of the daemon's namespace.
        let lib = unsafe { libc::dlopen(lib_c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if lib.is_null() {
            return Err(format!("could not load the tokenizer library {}: {}", library.display(), dl_error()));
        }
        let sym = |name: &[u8]| -> Result<*mut c_void, String> {
            // SAFETY: `lib` is a live handle; `name` is NUL-terminated.
            let p = unsafe { libc::dlsym(lib, name.as_ptr().cast::<c_char>()) };
            if p.is_null() {
                let name = String::from_utf8_lossy(&name[..name.len() - 1]).into_owned();
                return Err(format!("{} is not a superfluid tokenizer library (no {name})", library.display()));
            }
            Ok(p)
        };
        // SAFETY: each symbol is the exported function of that name, whose
        // signature the interface fixes (so for every transmute below).
        let version: abi::AbiVersionFn = unsafe { std::mem::transmute(sym(symbol::ABI_VERSION)?) };
        // SAFETY: as above.
        let have = unsafe { version() };
        if have != abi::ABI_VERSION {
            return Err(format!(
                "the tokenizer library {} speaks interface {have}; this superfluid speaks {}",
                library.display(),
                abi::ABI_VERSION
            ));
        }
        // SAFETY: as above.
        let open: abi::OpenFn = unsafe { std::mem::transmute(sym(symbol::OPEN)?) };
        // SAFETY: as above, for each field.
        let f = unsafe {
            Fns {
                close: std::mem::transmute::<*mut c_void, abi::CloseFn>(sym(symbol::CLOSE)?),
                encode: std::mem::transmute::<*mut c_void, abi::EncodeFn>(sym(symbol::ENCODE)?),
                token_bytes: std::mem::transmute::<*mut c_void, abi::TokenBytesFn>(sym(symbol::TOKEN_BYTES)?),
                vocab_size: std::mem::transmute::<*mut c_void, abi::VocabSizeFn>(sym(symbol::VOCAB_SIZE)?),
                special_tokens: std::mem::transmute::<*mut c_void, abi::SpecialTokensFn>(sym(symbol::SPECIAL_TOKENS)?),
                bos: std::mem::transmute::<*mut c_void, abi::BosFn>(sym(symbol::BOS)?),
                eos: std::mem::transmute::<*mut c_void, abi::EosFn>(sym(symbol::EOS)?),
                chat_template: std::mem::transmute::<*mut c_void, abi::ChatTemplateFn>(sym(symbol::CHAT_TEMPLATE)?),
            }
        };
        let model_c = CString::new(model.to_string_lossy().as_bytes()).map_err(|_| "NUL in model path".to_string())?;
        let mut err = vec![0 as c_char; 1024];
        // SAFETY: a NUL-terminated path and an error buffer of its length.
        let tok = unsafe { open(model_c.as_ptr(), err.as_mut_ptr(), err.len()) };
        if tok.is_null() {
            // SAFETY: the library NUL-terminates what it writes; the buffer
            // was zeroed, so an unwritten one reads as empty.
            let why = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
            return Err(format!("tokenizer load failed for {}: {why}", model.display()));
        }
        Ok(DylibTokenizer { tok, f, library: library.to_path_buf() })
    }

    pub fn library(&self) -> &Path {
        &self.library
    }

    fn filled<T: Copy + Default>(guess: usize, mut call: impl FnMut(*mut T, usize) -> usize) -> Option<Vec<T>> {
        let mut buf = vec![T::default(); guess];
        let n = call(buf.as_mut_ptr(), buf.len());
        if n == usize::MAX {
            return None;
        }
        if n > buf.len() {
            buf = vec![T::default(); n];
            let again = call(buf.as_mut_ptr(), buf.len());
            buf.truncate(again.min(n));
            return Some(buf);
        }
        buf.truncate(n);
        Some(buf)
    }
}

impl Drop for DylibTokenizer {
    fn drop(&mut self) {
        // SAFETY: the handle is live and released once; the library stays
        // loaded (see `open`).
        unsafe { (self.f.close)(self.tok) };
    }
}

impl Tokenizer for DylibTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        Self::filled(text.len() + 8, |out, cap| {
            // SAFETY: `text` is a live slice of `len` bytes; `out` holds `cap` ids.
            unsafe { (self.f.encode)(self.tok, text.as_ptr(), text.len(), out, cap) }
        })
        .unwrap_or_default()
    }

    fn token_bytes(&self, token: u32) -> Vec<u8> {
        Self::filled(64, |out, cap| {
            // SAFETY: `out` holds `cap` bytes.
            unsafe { (self.f.token_bytes)(self.tok, token, out, cap) }
        })
        .unwrap_or_default()
    }

    fn vocab_size(&self) -> u32 {
        // SAFETY: the handle is live.
        unsafe { (self.f.vocab_size)(self.tok) }
    }

    fn special_tokens(&self) -> Vec<(String, u32)> {
        let packed = Self::filled(16 << 10, |out, cap| {
            // SAFETY: `out` holds `cap` bytes.
            unsafe { (self.f.special_tokens)(self.tok, out, cap) }
        })
        .unwrap_or_default();
        abi::unpack_special(&packed)
    }

    fn bos_token(&self) -> Option<u32> {
        // SAFETY: the handle is live.
        let b = unsafe { (self.f.bos)(self.tok) };
        u32::try_from(b).ok()
    }

    fn eos_token(&self) -> u32 {
        // SAFETY: the handle is live.
        unsafe { (self.f.eos)(self.tok) }
    }

    fn chat_template_jinja(&self) -> String {
        let bytes = Self::filled(16 << 10, |out, cap| {
            // SAFETY: a NULL name asks for the default; `out` holds `cap` bytes.
            unsafe { (self.f.chat_template)(self.tok, std::ptr::null(), out, cap) }
        });
        bytes.map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default()
    }

    fn chat_template_named(&self, name: &str) -> Option<String> {
        let name = CString::new(name).ok()?;
        let bytes = Self::filled(16 << 10, |out, cap| {
            // SAFETY: a NUL-terminated name; `out` holds `cap` bytes.
            unsafe { (self.f.chat_template)(self.tok, name.as_ptr(), out, cap) }
        })?;
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }
}

pub fn library_file(stem: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
}

pub fn find_library(worker: &Path, stem: &str) -> Option<PathBuf> {
    let file = library_file(stem);
    let dir = worker.parent()?;
    [dir.join(&file), dir.join("..").join("lib").join(&file)].into_iter().find(|p| p.is_file())
}
