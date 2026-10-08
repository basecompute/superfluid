//! The four normative `Array` decode rules and the writer-side arena.

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::status::AbiError;
use crate::types::Array;

/// # Safety
/// Implementors are `#[repr(C)]` with every byte pattern valid, so a zeroed, partially
/// copied value is valid.
pub unsafe trait AbiRecord: Copy {
    const MIN_PREFIX: usize;
}

#[derive(Debug, Clone, Copy)]
pub struct ArenaBounds {
    pub base: usize,
    pub len: usize,
}

impl ArenaBounds {
    pub fn of_slice(s: &[u8]) -> ArenaBounds {
        ArenaBounds {
            base: s.as_ptr() as usize,
            len: s.len(),
        }
    }

    fn contains(&self, ptr: usize, len: usize) -> bool {
        let end = match ptr.checked_add(len) {
            Some(e) => e,
            None => return false,
        };
        let arena_end = self.base + self.len;
        ptr >= self.base && end <= arena_end
    }
}

/// # Safety
/// `arena` must be readable for the iterator's lifetime and written by the array's rules.
pub unsafe fn read_array<T: AbiRecord>(
    arr: &Array,
    arena: &ArenaBounds,
) -> Result<RecordIter<T>, AbiError> {
    if (arr.elem_size != 0 || arr.count != 0) && (arr.elem_size as usize) < T::MIN_PREFIX {
        return Err(AbiError::BadStride {
            elem_size: arr.elem_size,
            min_prefix: T::MIN_PREFIX,
        });
    }
    let ptr = arr.data as usize;
    if ptr == 0 {
        if arr.count != 0 {
            return Err(AbiError::BadAlignment);
        }
    } else if !ptr.is_multiple_of(8) {
        return Err(AbiError::BadAlignment);
    }
    if arr.count == 0 {
        return Ok(RecordIter {
            data: std::ptr::null(),
            count: 0,
            index: 0,
            elem_size: 0,
            _marker: PhantomData,
        });
    }
    let total = (arr.count as usize)
        .checked_mul(arr.elem_size as usize)
        .ok_or(AbiError::Bounds)?;
    if !arena.contains(ptr, total) {
        return Err(AbiError::Bounds);
    }
    Ok(RecordIter {
        data: arr.data as *const u8,
        count: arr.count,
        index: 0,
        elem_size: arr.elem_size,
        _marker: PhantomData,
    })
}

#[derive(Debug)]
pub struct RecordIter<T: AbiRecord> {
    data: *const u8,
    count: u32,
    index: u32,
    elem_size: u32,
    _marker: PhantomData<T>,
}

impl<T: AbiRecord> Iterator for RecordIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.index >= self.count {
            return None;
        }
        let src =
            self.data.wrapping_add(self.index as usize * self.elem_size as usize);
        self.index += 1;
        let copy_len = std::cmp::min(self.elem_size as usize, std::mem::size_of::<T>());
        let mut out = MaybeUninit::<T>::zeroed();
        // SAFETY: the read stays inside the bounds-checked slot, the destination holds
        // `copy_len` bytes, and `AbiRecord` makes any byte pattern valid.
        unsafe {
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr() as *mut u8, copy_len);
            Some(out.assume_init())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = (self.count - self.index) as usize;
        (rem, Some(rem))
    }
}

impl<T: AbiRecord> ExactSizeIterator for RecordIter<T> {}

/// # Safety
/// `ptr` must be valid for reads of `struct_size` bytes.
pub unsafe fn read_versioned_struct<T: AbiRecord>(
    ptr: *const T,
    struct_size: u64,
) -> Result<T, AbiError> {
    if (struct_size as usize) < T::MIN_PREFIX {
        return Err(AbiError::BadStruct {
            got: struct_size,
            min: T::MIN_PREFIX,
        });
    }
    let copy_len = std::cmp::min(struct_size as usize, std::mem::size_of::<T>());
    let mut out = MaybeUninit::<T>::zeroed();
    // SAFETY: caller guarantees struct_size readable bytes; copy_len is
    // bounded by both sides; zero-init satisfies rule 1.
    unsafe {
        std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr() as *mut u8, copy_len);
        Ok(out.assume_init())
    }
}

#[derive(Default)]
pub struct RecordArena {
    segments: Vec<Box<[u64]>>,
}

impl RecordArena {
    pub fn new() -> RecordArena {
        RecordArena::default()
    }

    pub fn push_records<T: AbiRecord>(&mut self, records: &[T]) -> Array {
        if records.is_empty() {
            return Array::EMPTY;
        }
        let elem = std::mem::size_of::<T>();
        assert!(
            std::mem::align_of::<T>() <= 8,
            "ABI records must not require alignment beyond 8"
        );
        assert!(
            u32::try_from(records.len()).is_ok(),
            "Array.count is u32; {} records cannot be published",
            records.len()
        );
        let bytes = std::mem::size_of_val(records);
        let words = bytes.div_ceil(8);
        let mut seg = vec![0u64; words].into_boxed_slice();
        // SAFETY: seg holds >= bytes bytes; T is a plain repr(C) record;
        // the copy writes records verbatim and leaves any tail as the
        // zero-fill (writer rule 2).
        unsafe {
            std::ptr::copy_nonoverlapping(
                records.as_ptr() as *const u8,
                seg.as_mut_ptr() as *mut u8,
                bytes,
            );
        }
        let data = seg.as_ptr() as *const std::ffi::c_void;
        self.segments.push(seg);
        Array {
            data,
            count: records.len() as u32,
            elem_size: elem as u32,
        }
    }

    pub fn contains(&self, arr: &Array) -> bool {
        if arr.count == 0 {
            return true;
        }
        let ptr = arr.data as usize;
        let len = (arr.count as usize).saturating_mul(arr.elem_size as usize);
        self.segments.iter().any(|s| {
            let b = ArenaBounds {
                base: s.as_ptr() as usize,
                len: s.len() * 8,
            };
            b.contains(ptr, len)
        })
    }

    pub fn bounds_of(&self, arr: &Array) -> Option<ArenaBounds> {
        if arr.count == 0 {
            return Some(ArenaBounds { base: 0, len: 0 });
        }
        let ptr = arr.data as usize;
        let len = (arr.count as usize).saturating_mul(arr.elem_size as usize);
        self.segments
            .iter()
            .map(|s| ArenaBounds {
                base: s.as_ptr() as usize,
                len: s.len() * 8,
            })
            .find(|b| b.contains(ptr, len))
    }
}

pub struct ArrayBuilder<T: AbiRecord> {
    records: Vec<T>,
}

impl<T: AbiRecord> Default for ArrayBuilder<T> {
    fn default() -> Self {
        ArrayBuilder {
            records: Vec::new(),
        }
    }
}

impl<T: AbiRecord> ArrayBuilder<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, r: T) -> &mut Self {
        self.records.push(r);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn build(self, arena: &mut RecordArena) -> Array {
        arena.push_records(&self.records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LaneAdmit, LaneCommit, LanePrefill};

    #[test]
    fn empty_array_is_legal_with_null_ptr() {
        let arr = Array::EMPTY;
        let arena = ArenaBounds { base: 0, len: 0 };
        // SAFETY: empty array reads nothing.
        let it = unsafe { read_array::<LaneCommit>(&arr, &arena) }.unwrap();
        assert_eq!(it.count(), 0);
    }

    #[test]
    fn empty_array_with_undersized_stride_still_rejected() {
        let arr = Array {
            data: std::ptr::null(),
            count: 0,
            elem_size: 8,
        };
        let arena = ArenaBounds { base: 0, len: 0 };
        // SAFETY: rejected before any read.
        let err = unsafe { read_array::<LaneCommit>(&arr, &arena) }.unwrap_err();
        assert!(matches!(err, AbiError::BadStride { .. }));
    }

    #[test]
    fn empty_array_with_misaligned_ptr_rejected() {
        let backing = [0u64; 2];
        let arr = Array {
            data: (backing.as_ptr() as usize + 4) as *const _,
            count: 0,
            elem_size: std::mem::size_of::<LaneCommit>() as u32,
        };
        let arena = ArenaBounds::of_slice(&[]);
        // SAFETY: rejected before any read.
        let err = unsafe { read_array::<LaneCommit>(&arr, &arena) }.unwrap_err();
        assert_eq!(err, AbiError::BadAlignment);
    }

    #[test]
    fn roundtrip_same_version() {
        let mut arena = RecordArena::new();
        let recs = [
            LanePrefill {
                lane_tag: 7,
                token_offset: 0,
                token_count: 128,
            },
            LanePrefill {
                lane_tag: 9,
                token_offset: 128,
                token_count: 64,
            },
        ];
        let arr = arena.push_records(&recs);
        let bounds = arena.bounds_of(&arr).unwrap();
        // SAFETY: arr was built into arena and bounds cover it.
        let got: Vec<_> = unsafe { read_array::<LanePrefill>(&arr, &bounds) }
            .unwrap()
            .collect();
        assert_eq!(got, recs);
    }

    #[test]
    fn newer_writer_larger_stride_reads_known_prefix() {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct PrefillV2 {
            lane_tag: u64,
            token_offset: u32,
            token_count: u32,
            future_field: u64,
        }
        // SAFETY: plain integers.
        unsafe impl AbiRecord for PrefillV2 {
            const MIN_PREFIX: usize = 16;
        }
        let mut arena = RecordArena::new();
        let arr = arena.push_records(&[PrefillV2 {
            lane_tag: 3,
            token_offset: 1,
            token_count: 2,
            future_field: 0xDEAD,
        }]);
        assert_eq!(arr.elem_size, 24);
        let bounds = arena.bounds_of(&arr).unwrap();
        // SAFETY: arr was built into arena.
        let got: Vec<_> = unsafe { read_array::<LanePrefill>(&arr, &bounds) }
            .unwrap()
            .collect();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].lane_tag, 3);
        assert_eq!(got[0].token_count, 2);
    }

    #[test]
    fn a_v1_admit_decodes_with_no_grammar_replay() {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct AdmitV1([u64; 17]);
        // SAFETY: plain integers.
        unsafe impl AbiRecord for AdmitV1 {
            const MIN_PREFIX: usize = 136;
        }
        let mut words = [0u64; 17];
        words[0] = 11;
        let mut arena = RecordArena::new();
        let arr = arena.push_records(&[AdmitV1(words)]);
        assert_eq!(arr.elem_size, 136);
        let bounds = arena.bounds_of(&arr).unwrap();
        // SAFETY: arr was built into arena.
        let got: Vec<LaneAdmit> =
            unsafe { read_array::<LaneAdmit>(&arr, &bounds) }.unwrap().collect();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].lane_tag, 11);
        assert_eq!(got[0].grammar_replay, 0);
    }

    #[test]
    fn older_writer_smaller_stride_zero_fills_tail() {
        let mut arena = RecordArena::new();
        let tags: [u64; 2] = [11, 22];
        let arr = arena.push_records(&tags);
        assert_eq!(arr.elem_size, 8);
        let bounds = arena.bounds_of(&arr).unwrap();
        // SAFETY: arr was built into arena.
        let err = unsafe { read_array::<LanePrefill>(&arr, &bounds) }.unwrap_err();
        assert_eq!(
            err,
            AbiError::BadStride {
                elem_size: 8,
                min_prefix: 16
            }
        );
    }

    #[test]
    fn misaligned_data_rejected() {
        let mut arena = RecordArena::new();
        let arr = arena.push_records(&[LaneCommit::default()]);
        let bounds = arena.bounds_of(&arr).unwrap();
        let bad = Array {
            data: (arr.data as usize + 4) as *const _,
            ..arr
        };
        // SAFETY: rejected before any read.
        let err = unsafe { read_array::<LaneCommit>(&bad, &bounds) }.unwrap_err();
        assert_eq!(err, AbiError::BadAlignment);
    }

    #[test]
    fn bounds_overflow_rejected() {
        let mut arena = RecordArena::new();
        let arr = arena.push_records(&[LaneCommit::default()]);
        let bounds = arena.bounds_of(&arr).unwrap();
        let bad = Array {
            count: u32::MAX,
            elem_size: u32::MAX,
            ..arr
        };
        // SAFETY: rejected before any read.
        let err = unsafe { read_array::<LaneCommit>(&bad, &bounds) }.unwrap_err();
        assert_eq!(err, AbiError::Bounds);
    }

    #[test]
    fn out_of_arena_rejected() {
        let mut arena = RecordArena::new();
        let arr = arena.push_records(&[LaneCommit::default()]);
        let bounds = arena.bounds_of(&arr).unwrap();
        let outside = [0u64; 4];
        let bad = Array {
            data: outside.as_ptr() as *const _,
            ..arr
        };
        // SAFETY: rejected before any read.
        let err = unsafe { read_array::<LaneCommit>(&bad, &bounds) }.unwrap_err();
        assert_eq!(err, AbiError::Bounds);
    }

    #[test]
    fn versioned_struct_rules() {
        use crate::types::OpStatus;
        let v1 = OpStatus {
            struct_size: std::mem::size_of::<OpStatus>() as u64,
            state: 2,
            error: 0,
            bytes_moved: 4096,
            ..Default::default()
        };
        // SAFETY: v1 is a live local.
        let got = unsafe { read_versioned_struct(&v1, v1.struct_size) }.unwrap();
        assert_eq!(got, v1);

        // SAFETY: rejected before any read.
        let err = unsafe { read_versioned_struct(&v1, 8) }.unwrap_err();
        assert!(matches!(err, AbiError::BadStruct { got: 8, .. }));
    }
}
