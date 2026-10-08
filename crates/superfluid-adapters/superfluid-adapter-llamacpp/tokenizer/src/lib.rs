//! The llama.cpp runtime's tokenizer library.

superfluid_adapter_kit::export_tokenizer!(|path: &std::path::Path| {
    if let Some(dir) = own_directory() {
        superfluid_adapter_llamacpp::sys::set_own_directory(dir);
    }
    superfluid_adapter_llamacpp::quiet_logs();
    superfluid_adapter_llamacpp::LlamaTokenizer::load(path)
});

fn own_directory() -> Option<std::path::PathBuf> {
    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
    // SAFETY: dladdr fills `info` for an address inside a loaded image, and
    // this function's own address is one; the name it returns lives as long
    // as the image, which outlives this call.
    let file = unsafe {
        if libc::dladdr(own_directory as *const libc::c_void, info.as_mut_ptr()) == 0 {
            return None;
        }
        let info = info.assume_init();
        if info.dli_fname.is_null() {
            return None;
        }
        std::ffi::CStr::from_ptr(info.dli_fname).to_str().ok()?.to_string()
    };
    std::path::Path::new(&file).parent().map(std::path::Path::to_path_buf)
}
