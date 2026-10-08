//! Shm-ring abstraction.

use superfluid_abi::{RingRef, TokenRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RingError {
    #[error("unknown ring id {0}")]
    UnknownRing(u32),
    #[error("stale generation (ref {reference}, current {current})")]
    StaleGeneration { reference: u64, current: u64 },
    #[error("span out of ring bounds")]
    OutOfBounds,
}

#[derive(Debug, Clone, Copy)]
pub struct RingAttachment {
    pub ring_id: u32,
    pub kind: u32,
    pub role: u32,
    pub slots: u32,
    pub slot_bytes: u32,
    pub base: *mut u8,
    pub len: usize,
}

pub trait Rings {
    fn read_tokens(&self, r: &TokenRef) -> Result<Vec<u32>, RingError>;

    fn write_tokens(&mut self, tokens: &[u32]) -> Result<TokenRef, RingError>;

    fn write_logits_row(&mut self, row: &[f32]) -> RingRef;

    fn read_logits_row(&self, r: &RingRef) -> Result<Vec<f32>, RingError>;

    /// The daemon wants the running tick to end at its next step boundary
    /// (a request is waiting to be admitted, or a lane was cancelled).
    fn yield_requested(&self) -> bool {
        false
    }

    /// Latency-sensitive requests have arrived lately: run prefill in short
    /// steps, so a yield comes soon after it is asked for.
    fn latency_wanted(&self) -> bool {
        false
    }
}

pub const TOKEN_RING_IN: u32 = 1;
pub const TOKEN_RING_OUT: u32 = 2;
pub const LOGITS_RING: u32 = 3;

#[derive(Debug)]
pub struct InMemoryRings {
    token_in: RingBuf<Vec<u32>>,
    token_out: RingBuf<Vec<u32>>,
    logits: RingBuf<Vec<f32>>,
    yield_requested: bool,
    latency_wanted: bool,
}

#[derive(Debug)]
struct RingBuf<T> {
    slots: Vec<Option<(u64, T)>>,
    next_index: usize,
    next_generation: u64,
}

impl<T: Clone> RingBuf<T> {
    fn new(slots: usize) -> Self {
        RingBuf {
            slots: vec![None; slots],
            next_index: 0,
            next_generation: 1,
        }
    }

    fn push(&mut self, payload: T) -> (u32, u64) {
        let index = self.next_index;
        let generation = self.next_generation;
        self.slots[index] = Some((generation, payload));
        self.next_index = (index + 1) % self.slots.len();
        self.next_generation += 1;
        (index as u32, generation)
    }

    fn get(&self, index: u32, generation: u64) -> Result<&T, RingError> {
        let slot = self
            .slots
            .get(index as usize)
            .ok_or(RingError::OutOfBounds)?;
        match slot {
            Some((gen, payload)) if *gen == generation => Ok(payload),
            Some((gen, _)) => Err(RingError::StaleGeneration {
                reference: generation,
                current: *gen,
            }),
            None => Err(RingError::StaleGeneration {
                reference: generation,
                current: 0,
            }),
        }
    }
}

impl InMemoryRings {
    pub fn new() -> Self {
        InMemoryRings {
            token_in: RingBuf::new(64),
            token_out: RingBuf::new(64),
            logits: RingBuf::new(16),
            yield_requested: false,
            latency_wanted: false,
        }
    }

    /// What `yield_requested` answers, for tests of an engine's early stop.
    pub fn set_yield(&mut self, on: bool) {
        self.yield_requested = on;
    }

    /// What `latency_wanted` answers, for tests.
    pub fn set_latency(&mut self, on: bool) {
        self.latency_wanted = on;
    }

    pub fn stage_prompt(&mut self, tokens: &[u32]) -> TokenRef {
        let (index, generation) = self.token_in.push(tokens.to_vec());
        TokenRef {
            ring_id: TOKEN_RING_IN,
            index,
            count: tokens.len() as u32,
            _pad0: 0,
            generation,
        }
    }
}

impl Default for InMemoryRings {
    fn default() -> Self {
        Self::new()
    }
}

impl Rings for InMemoryRings {
    fn read_tokens(&self, r: &TokenRef) -> Result<Vec<u32>, RingError> {
        let buf = match r.ring_id {
            TOKEN_RING_IN => &self.token_in,
            TOKEN_RING_OUT => &self.token_out,
            other => return Err(RingError::UnknownRing(other)),
        };
        let tokens = buf.get(r.index, r.generation)?;
        if tokens.len() != r.count as usize {
            return Err(RingError::OutOfBounds);
        }
        Ok(tokens.clone())
    }

    fn write_tokens(&mut self, tokens: &[u32]) -> Result<TokenRef, RingError> {
        let (index, generation) = self.token_out.push(tokens.to_vec());
        Ok(TokenRef {
            ring_id: TOKEN_RING_OUT,
            index,
            count: tokens.len() as u32,
            _pad0: 0,
            generation,
        })
    }

    fn write_logits_row(&mut self, row: &[f32]) -> RingRef {
        let (index, generation) = self.logits.push(row.to_vec());
        RingRef {
            ring_id: LOGITS_RING,
            index,
            generation,
        }
    }

    fn read_logits_row(&self, r: &RingRef) -> Result<Vec<f32>, RingError> {
        if r.ring_id != LOGITS_RING {
            return Err(RingError::UnknownRing(r.ring_id));
        }
        self.logits.get(r.index, r.generation).cloned()
    }

    fn yield_requested(&self) -> bool {
        self.yield_requested
    }

    fn latency_wanted(&self) -> bool {
        self.latency_wanted
    }
}
