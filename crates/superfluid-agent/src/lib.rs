//! Node-agent skeleton.

pub mod client;
pub mod sampler;

pub use client::WorkerClient;
pub use sampler::argmax;

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
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
    #[error("worker rejected the request with status {0}")]
    Rejected(i32),
    #[error("peer closed the connection")]
    Closed,
}
