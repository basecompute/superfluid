//! `baseRT_status` codes and the Rust-side error type.

pub type StatusRaw = i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Status {
    Ok = 0,

    RejectBadStride = -100,
    RejectBadAlignment = -101,
    RejectBounds = -102,
    RejectBadRefGeneration = -103,
    RejectUnknownLane = -104,
    RejectDuplicateLane = -105,
    RejectIllegalCombination = -106,
    RejectBudget = -107,
    RejectStaleSeed = -108,
    RejectHostRules = -110,
    RejectTransferLocked = -111,
    RejectStaleNonce = -112,
    NeedsReplan = -113,
    RejectBadStruct = -114,

    Busy = -120,
    BufferTooSmall = -121,
    StaleSizing = -122,
    OutOfBoundary = -123,
    EnvelopeMismatch = -124,
    Checksum = -125,
    IdentityMismatch = -126,
    Unsupported = -127,
    RegistrationRefused = -128,
    UnknownHandle = -129,
    SeedUnservable = -130,
    RingLayout = -131,

    RejectCertUnmatched = -140,

    Fatal = -1,
}

pub mod fault_code {
    pub const UNSPECIFIED: u32 = 0;
    pub const GRAMMAR_OVERFLOW: u32 = 1;
    pub const REF_SPAN_OOB: u32 = 2;
    pub const STRATEGY: u32 = 3;
    pub const OOM_TENTATIVE: u32 = 4;
    pub const INTERNAL: u32 = 5;
}

impl Status {
    pub fn from_raw(raw: StatusRaw) -> Option<Status> {
        Some(match raw {
            0 => Status::Ok,
            -100 => Status::RejectBadStride,
            -101 => Status::RejectBadAlignment,
            -102 => Status::RejectBounds,
            -103 => Status::RejectBadRefGeneration,
            -104 => Status::RejectUnknownLane,
            -105 => Status::RejectDuplicateLane,
            -106 => Status::RejectIllegalCombination,
            -107 => Status::RejectBudget,
            -108 => Status::RejectStaleSeed,
            -110 => Status::RejectHostRules,
            -111 => Status::RejectTransferLocked,
            -112 => Status::RejectStaleNonce,
            -113 => Status::NeedsReplan,
            -114 => Status::RejectBadStruct,
            -120 => Status::Busy,
            -121 => Status::BufferTooSmall,
            -122 => Status::StaleSizing,
            -123 => Status::OutOfBoundary,
            -124 => Status::EnvelopeMismatch,
            -125 => Status::Checksum,
            -126 => Status::IdentityMismatch,
            -127 => Status::Unsupported,
            -128 => Status::RegistrationRefused,
            -129 => Status::UnknownHandle,
            -130 => Status::SeedUnservable,
            -131 => Status::RingLayout,
            -140 => Status::RejectCertUnmatched,
            -1 => Status::Fatal,
            _ => return None,
        })
    }

    pub const fn raw(self) -> StatusRaw {
        self as i32
    }

    pub fn is_plan_rejection(self) -> bool {
        (-114..=-100).contains(&self.raw())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AbiError {
    #[error("record stride {elem_size} below the type's minimum required prefix {min_prefix}")]
    BadStride { elem_size: u32, min_prefix: usize },
    #[error("array data pointer is not 8-byte aligned")]
    BadAlignment,
    #[error("array bounds violation (count x elem_size overflows or exceeds its arena)")]
    Bounds,
    #[error("top-level struct_size {got} below the v1 minimum {min}")]
    BadStruct { got: u64, min: usize },
}

impl AbiError {
    pub fn status(self) -> Status {
        match self {
            AbiError::BadStride { .. } => Status::RejectBadStride,
            AbiError::BadAlignment => Status::RejectBadAlignment,
            AbiError::Bounds => Status::RejectBounds,
            AbiError::BadStruct { .. } => Status::RejectBadStruct,
        }
    }
}
