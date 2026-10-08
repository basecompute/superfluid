//! Mock-engine state model.

use std::collections::{HashMap, HashSet};

use superfluid_abi::{space_kind, RingRef, SamplingParams};

use crate::engine::{SeqHandle, SpaceConfig};

pub fn chain_digest(parent: &[u8; 32], op: u64, shape: u64) -> [u8; 32] {
    let mut input = [0u8; 24];
    input[..8].copy_from_slice(&parent[..8]);
    input[8..16].copy_from_slice(&op.to_le_bytes());
    input[16..24].copy_from_slice(&shape.to_le_bytes());
    let h = xxhash_rust::xxh3::xxh3_64(&input);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&h.to_le_bytes());
    out
}

pub mod prov_op {
    pub const PREFILL: u64 = 1;
    pub const DECODE: u64 = 2;
    pub const PROMOTE: u64 = 4;
    pub const SEED: u64 = 5;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Busy {
    Idle,
    Ticking,
    OpActive(HashSet<u32>),
}

#[derive(Debug, Clone)]
pub struct SpaceState {
    pub kind: u32,
    pub version_tag: u64,
    pub valid_len: u64,
    pub demoted: Vec<(u64, u64, u8)>,
    pub provenance: [u8; 32],
    pub taint_bits: u32,
    pub content_gen: u64,
    pub boundaries: Vec<u64>,
}

impl SpaceState {
    pub fn new(cfg: &SpaceConfig) -> SpaceState {
        SpaceState {
            kind: cfg.kind,
            version_tag: cfg.version_tag,
            valid_len: 0,
            demoted: Vec::new(),
            provenance: [0u8; 32],
            taint_bits: 0,
            content_gen: 1,
            boundaries: vec![0],
        }
    }

    pub fn is_blob_like(&self) -> bool {
        matches!(self.kind, space_kind::RECURRENT_BLOB | space_kind::RING_KV)
    }

    pub fn advance(&mut self, new_len: u64, op: u64, interval: u32) {
        self.provenance = chain_digest(&self.provenance, op, new_len);
        self.valid_len = new_len;
        self.content_gen += 1;
        if self.is_blob_like() && interval > 0 {
            let interval = interval as u64;
            let last = *self.boundaries.last().unwrap_or(&0);
            let mut b = (last / interval + 1) * interval;
            while b <= new_len {
                self.boundaries.push(b);
                b += interval;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub spaces: HashMap<u32, SpaceState>,
    pub busy: Busy,
}

impl Sequence {
    pub fn op_active_on(&self, space_id: u32) -> bool {
        matches!(&self.busy, Busy::OpActive(s) if s.contains(&space_id))
    }

    pub fn any_op_active(&self) -> bool {
        matches!(&self.busy, Busy::OpActive(s) if !s.is_empty())
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Lane {
    pub seq: SeqHandle,
    pub sampling: u32,
    pub params: SamplingParams,
    pub determinism: u8,
    pub strategy_slot: u32,
    pub cert_id: u32,
    pub granted_class: u8,
    pub rng_counter: u64,
    pub prompt: Vec<u32>,
    pub prefilled: u64,
    pub prefill_end: u64,
    pub committed: Vec<u32>,
    pub pending_logits: Option<RingRef>,
    pub finished: bool,
    pub media_binds: Vec<(u32, u64)>,
    pub script_pos: usize,
    pub grammar: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub id: u64,
    pub space_id: u32,
    pub tokens: Vec<u32>,
    pub provenance: [u8; 32],
    pub taint_bits: u32,
    pub tier: u8,
    pub bytes: u64,
    pub cache_class: u64,
    pub pinned: u32,
    pub boundaries: Vec<u64>,
    pub lru: u64,
}

#[derive(Debug, Default)]
pub struct Cache {
    pub entries: Vec<CacheEntry>,
    next_id: u64,
    lru_clock: u64,
    pub variant_quota: usize,
}

#[derive(Debug, Default)]
pub struct PublishOutcome {
    pub stored: Option<u64>,
    pub evicted: Vec<(u32, u64)>,
}

impl Cache {
    pub fn new() -> Cache {
        Cache {
            entries: Vec::new(),
            next_id: 1,
            lru_clock: 0,
            variant_quota: 2,
        }
    }

    pub fn publish(&mut self, mut e: CacheEntry) -> PublishOutcome {
        self.lru_clock += 1;
        e.lru = self.lru_clock;
        e.id = self.next_id;
        self.next_id += 1;
        let mut outcome = PublishOutcome::default();

        if self.entries.iter().any(|x| {
            x.space_id == e.space_id && x.tokens == e.tokens && x.provenance == e.provenance
        }) {
            return outcome;
        }
        let variants: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, x)| x.space_id == e.space_id && x.tokens == e.tokens)
            .map(|(i, _)| i)
            .collect();
        if variants.len() >= self.variant_quota {
            let victim = variants
                .iter()
                .copied()
                .filter(|&i| self.entries[i].taint_bits != 0 && self.entries[i].pinned == 0)
                .min_by_key(|&i| self.entries[i].lru);
            match victim {
                Some(v) => {
                    let removed = self.entries.remove(v);
                    outcome.evicted.push((removed.space_id, removed.bytes));
                }
                None => {
                    if e.taint_bits != 0 {
                        return outcome;
                    }
                    let Some(v) = variants
                        .iter()
                        .copied()
                        .filter(|&i| self.entries[i].pinned == 0)
                        .min_by_key(|&i| self.entries[i].lru)
                    else {
                        return outcome;
                    };
                    let removed = self.entries.remove(v);
                    outcome.evicted.push((removed.space_id, removed.bytes));
                }
            }
        }
        outcome.stored = Some(e.id);
        self.entries.push(e);
        outcome
    }

    pub fn get(&self, id: u64) -> Option<&CacheEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut CacheEntry> {
        self.entries.iter_mut().find(|e| e.id == id)
    }

    pub fn touch(&mut self, id: u64) {
        self.lru_clock += 1;
        let clock = self.lru_clock;
        if let Some(e) = self.get_mut(id) {
            e.lru = clock;
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SeedLease {
    pub handle: u64,
    pub prefix: Vec<u32>,
    pub determinism: u8,
    pub pinned_entries: Vec<(u32, u64)>,
    pub expires_at_tick: u64,
    pub adopted: Option<crate::engine::SeqHandle>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Snapshot,
    Restore,
    Demote,
    Promote,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct OpState {
    pub handle: u64,
    pub seq: SeqHandle,
    pub space_id: u32,
    pub kind: OpKind,
    pub state: u8,
    pub error: u32,
    pub bytes_total: u64,
    pub bytes_moved: u64,
    pub output: Option<Vec<u8>>,
    pub staged: Option<StagedImport>,
    pub demote_range: Option<(u64, u64, u8)>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StagedImport {
    pub valid_len: u64,
    pub provenance: [u8; 32],
    pub taint_bits: u32,
    pub boundaries: Vec<u64>,
    pub promote_range: Option<(u64, u64)>,
    pub encoding: u8,
}
