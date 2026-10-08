//! `FfiEngine`.

mod engine;
mod native;
mod tick;
mod tokenizer;

pub mod libbasert;

pub use engine::{FfiEngine, FfiEngineConfig, FfiError};
pub use native::NativeEngine;
pub use tokenizer::TokenizerHandle;

pub fn set_kv_bits(bits: i32) {
    if libbasert::load().is_ok() {
        // SAFETY: plain global setter, serialized by callers (the same
        // process-global discipline as the other pre-load setters).
        unsafe { libbasert::sys::baseRT_set_kv_bits(bits) };
    }
}

pub fn suggest_max_context(
    models: &[std::path::PathBuf],
    speculators: &[(std::path::PathBuf, bool, Option<usize>)],
    max_batch: i32,
    kv_bits: i32,
) -> Option<i32> {
    use std::os::unix::ffi::OsStrExt;
    let cstrs = |ps: &[std::path::PathBuf]| -> Option<Vec<std::ffi::CString>> {
        ps.iter().map(|p| std::ffi::CString::new(p.as_os_str().as_bytes())).collect::<Result<_, _>>().ok()
    };
    let spec_paths: Vec<std::path::PathBuf> = speculators.iter().map(|(p, _, _)| p.clone()).collect();
    let embedded: Vec<std::os::raw::c_int> = speculators.iter().map(|&(_, e, _)| e as std::os::raw::c_int).collect();
    let target: Vec<std::os::raw::c_int> = speculators
        .iter()
        .map(|&(_, _, t)| t.filter(|&i| i < models.len()).map_or(-1, |i| i as std::os::raw::c_int))
        .collect();
    let (owned, owned_spec) = (cstrs(models)?, cstrs(&spec_paths)?);
    let ptrs: Vec<*const std::os::raw::c_char> = owned.iter().map(|c| c.as_ptr()).collect();
    let spec: Vec<*const std::os::raw::c_char> = owned_spec.iter().map(|c| c.as_ptr()).collect();
    libbasert::load().ok()?;
    // SAFETY: `ptrs` / `spec` point into `owned` / `owned_spec` and `embedded`
    // / `target` have one entry per speculator; all outlive the call, and the
    // engine reads them and returns a plain integer.
    let n = unsafe {
        libbasert::sys::baseRT_suggest_max_context_spec(
            ptrs.as_ptr(),
            ptrs.len() as i32,
            spec.as_ptr(),
            embedded.as_ptr(),
            target.as_ptr(),
            spec.len() as i32,
            max_batch,
            kv_bits,
            1,
        )
    };
    (n > 0).then_some(n)
}

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            let mut e: i32 = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn fnv_identity_spec_is_pinned() {
        use superfluid_fingerprint::{fnv1a64, FNV_BASIS};
        assert_eq!(fnv1a64(b"", FNV_BASIS), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv1a64(b"a", FNV_BASIS), 0xAF63_DC4C_8601_EC8C);
        assert_eq!(fnv1a64(b"foobar", FNV_BASIS), 0x85944171F73967E8);
    }

    #[test]
    fn f16_conversion_covers_the_classes() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0400), 6.103_515_6e-5);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7c01).is_nan());
    }
}
