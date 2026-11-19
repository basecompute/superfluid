//! The llama.cpp runtime's tokenizer library.

superfluid_adapter_kit::export_tokenizer!(|path: &std::path::Path| {
    if let Some(dir) = own_directory() {
        superfluid_adapter_llamacpp::sys::set_own_directory(dir);
    }
    superfluid_adapter_llamacpp::quiet_logs();
    superfluid_adapter_llamacpp::LlamaTokenizer::load(path)
});

#[cfg(unix)]
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

#[cfg(windows)]
fn own_directory() -> Option<std::path::PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    };

    let mut module: HMODULE = std::ptr::null_mut();
    let flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    // SAFETY: asks which loaded image holds this function's own address, without
    // taking a reference to it.
    if unsafe { GetModuleHandleExW(flags, own_directory as *const u16, &mut module) } == 0 {
        return None;
    }
    let mut name = vec![0u16; 1024];
    loop {
        // SAFETY: writes at most `name.len()` units into `name`.
        let n = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) } as usize;
        if n == 0 {
            return None;
        }
        if n < name.len() {
            name.truncate(n);
            break;
        }
        // A name cut short fills the buffer: ask again with more room.
        name.resize(name.len() * 2, 0);
    }
    std::path::PathBuf::from(std::ffi::OsString::from_wide(&name)).parent().map(std::path::Path::to_path_buf)
}
