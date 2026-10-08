//! The Link F session layer.

pub mod agent;
pub mod driver;
pub mod head;
pub mod secure;
pub mod wal;

pub use agent::{AgentAction, AgentSession};
pub use driver::Endpoint;
pub use secure::{key_from, Role, SecureStream};
pub use head::HeadSession;
pub use wal::WalStub;

#[derive(Debug, thiserror::Error)]
pub enum LinkFError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("envelope: {0}")]
    Envelope(#[from] superfluid_proto::EnvelopeError),
    #[error("payload decode: {0}")]
    Payload(#[from] postcard::Error),
    #[error("protocol violation: {0}")]
    Protocol(&'static str),
    #[error("peer closed the connection")]
    Closed,
    #[error("timed out waiting for the peer")]
    TimedOut,
    #[error("peer failed authentication")]
    Unauthorized,
}
