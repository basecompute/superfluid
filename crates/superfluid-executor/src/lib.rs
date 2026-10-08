//! The generic executor.

pub mod cache;
pub mod capabilities;
pub mod envelope;
pub mod executor;
pub mod fake;
pub mod grammar;
pub mod primitives;
pub mod sampling;
pub mod sizing;
mod speculate;
mod tick;

pub use executor::{exit_on_fatal, Executor, ExecutorConfig};
pub use fake::FakePrimitives;
pub use primitives::{
    Feed, Input, MemCounters, PrimError, RuntimeDescriptor, RuntimePrimitives, SampleSpec, SamplingDefaults, Seq,
    Vocabulary,
};
