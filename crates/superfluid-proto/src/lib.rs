//! superfluid wire protocol.

pub mod envelope;
pub mod handshake;
pub mod idempotency;
pub mod linkf;
pub mod linkw;

pub use envelope::{encode_frame, EnvelopeError, Frame, FrameClass, FrameDecoder, MAX_FRAME};
pub use handshake::{negotiate, Hello, HelloAckF, HelloAckW, LinkRole};
