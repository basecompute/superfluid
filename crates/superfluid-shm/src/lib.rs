//! Link W local-adapter mechanics.

pub mod control;
pub mod fdpass;
pub mod rings;
pub mod segment;

pub use control::{ControlSegment, YieldSignal, CONTROL_BYTES, CONTROL_TAG};
pub use rings::{RingRole, SharedRing, SharedRings, LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};
pub use segment::ShmSegment;

#[derive(Debug, thiserror::Error)]
pub enum ShmError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("segment too small for the requested ring layout")]
    TooSmall,
    #[error("bad ring magic (segment is not a ring or layout disagrees)")]
    BadMagic,
    #[error("payload exceeds the ring's slot capacity")]
    PayloadTooLarge,
    #[error("segment is {actual} bytes, but {announced} were announced")]
    Short { announced: usize, actual: u64 },
    #[error("shm word at offset {0} is not 8-byte aligned (ring slot_bytes must be a multiple of 8)")]
    Misaligned(usize),
}
