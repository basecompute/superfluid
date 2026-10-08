//! Engine-worker skeleton.

pub mod check;
pub mod materialize;
pub mod process;
pub mod server;

pub use process::{ring_specs_for, WorkerArgs};
pub use server::{WorkerConfig, WorkerServer};

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("envelope: {0}")]
    Envelope(#[from] superfluid_proto::EnvelopeError),
    #[error("payload decode: {0}")]
    Payload(#[from] postcard::Error),
    #[error("shm: {0}")]
    Shm(#[from] superfluid_shm::ShmError),
    #[error("protocol violation: {0}")]
    Protocol(&'static str),
    #[error("peer closed the connection")]
    Closed,
}
