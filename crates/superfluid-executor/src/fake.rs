//! `FakePrimitives`.

use std::collections::{HashMap, HashSet};

use superfluid_abi::encoding;

use crate::primitives::{SampleSpec,
    Feed, Input, MemCounters, PrimError, RuntimeDescriptor, RuntimePrimitives, SamplingDefaults, Seq, Vocabulary,
};

#[derive(Debug, Clone)]
pub struct FakeConfig {
    pub vocab: u32,
    pub eos: u32,
    pub page: u32,
    pub max_batch: u32,
    pub max_seq_len: u64,
    pub kv_bytes_per_token: u64,
    pub cells_total: u64,
    pub truncate_partial: bool,
    pub max_seqs: u32,
    pub identity: u8,
    pub engine_sampling: bool,
    pub copy_shares_cells: bool,
    pub takeover_preferred: bool,
    pub seed_min_tokens: u64,
    /// Cells the cache may keep in the pool, and tokens it may hold exported
    /// (the descriptor's limits of the same names); 0 and 0, the default, is
    /// a runtime whose resting sequences cost nothing.
    pub cache_resident_cells: u64,
    pub cache_exported_cells: u64,
    pub vocabulary: Option<Vocabulary>,
    pub sampling_defaults: SamplingDefaults,
    /// Wall time each model step takes, for tests that need ticks to last.
    pub step_delay: std::time::Duration,
    /// Serve `step_rows` (a row after every token of a feed).
    pub verify_rows: bool,
    /// Predict the token that followed the last earlier occurrence of the
    /// current one: text that repeats itself, as a prompt lookup drafts.
    pub copy_model: bool,
    /// With `copy_model`, every this-many positions predict a fresh token
    /// instead (0: never): drafts that are taken in part.
    pub copy_miss_every: usize,
    /// The descriptor's `prefill_step_tokens`.
    pub prefill_step_tokens: u32,
}

impl Default for FakeConfig {
    fn default() -> Self {
        FakeConfig {
            vocab: 32000,
            eos: 31999,
            page: 16,
            max_batch: 8,
            max_seq_len: 4096,
            kv_bytes_per_token: 64,
            cells_total: 0,
            truncate_partial: true,
            max_seqs: 0,
            identity: 0xFA,
            engine_sampling: true,
            copy_shares_cells: false,
            takeover_preferred: false,
            seed_min_tokens: 0,
            cache_resident_cells: 0,
            cache_exported_cells: 0,
            vocabulary: None,
            sampling_defaults: SamplingDefaults::default(),
            step_delay: std::time::Duration::ZERO,
            verify_rows: false,
            copy_model: false,
            copy_miss_every: 0,
            prefill_step_tokens: 0,
        }
    }
}

pub struct FakePrimitives {
    pub cfg: FakeConfig,
    seqs: HashMap<Seq, Vec<u32>>,
    next_seq: Seq,
    scripted: HashMap<Seq, Vec<u32>>,
    fail_next: Option<PrimError>,
    poison_import: bool,
    pub feeds: Vec<(usize, bool)>,
    /// For every step, the cells the pool held for sequences the step did
    /// not feed: what a runtime that attends over its whole pool (llama.cpp's
    /// unified cache) pays for on each one.
    pub beside: Vec<u64>,
    cells: HashMap<Seq, Vec<u64>>,
    holders: HashMap<u64, u32>,
    next_cell: u64,
    /// What the executor last asked of `sync_steps`.
    pub synced: Option<bool>,
}

impl Default for FakePrimitives {
    fn default() -> Self {
        FakePrimitives::new(FakeConfig::default())
    }
}

const SEED: u64 = 0xCBF2_9CE4_8422_2325;

fn hash(content: &[u32], salt: u64) -> u64 {
    let mut h = superfluid_fingerprint::fnv1a64(&salt.to_le_bytes(), SEED);
    for &t in content {
        h = superfluid_fingerprint::fnv1a64(&t.to_le_bytes(), h);
    }
    h
}

impl FakePrimitives {
    pub fn new(cfg: FakeConfig) -> FakePrimitives {
        FakePrimitives {
            cfg,
            seqs: HashMap::new(),
            next_seq: 1,
            scripted: HashMap::new(),
            fail_next: None,
            poison_import: false,
            feeds: Vec::new(),
            beside: Vec::new(),
            cells: HashMap::new(),
            holders: HashMap::new(),
            next_cell: 1,
            synced: None,
        }
    }

    fn room_for(&self, more: u64, freed: u64) -> Result<(), PrimError> {
        let used = (self.holders.len() as u64).saturating_sub(freed);
        if self.cfg.cells_total > 0 && used + more > self.cfg.cells_total {
            return Err(PrimError::Capacity);
        }
        Ok(())
    }

    fn take_cells(&mut self, seq: Seq, n: u64) {
        let ids: Vec<u64> = (0..n).map(|k| self.next_cell + k).collect();
        self.next_cell += n;
        for &id in &ids {
            self.holders.insert(id, 1);
        }
        self.cells.entry(seq).or_default().extend(ids);
    }

    fn drop_cells(&mut self, seq: Seq, from: usize) {
        let Some(own) = self.cells.get_mut(&seq) else { return };
        let gone: Vec<u64> = own.drain(from.min(own.len())..).collect();
        for id in gone {
            if let Some(n) = self.holders.get_mut(&id) {
                *n -= 1;
                if *n == 0 {
                    self.holders.remove(&id);
                }
            }
        }
    }

    pub fn token_for(&self, content: &[u32]) -> u32 {
        let miss = self.cfg.copy_miss_every > 0 && content.len().is_multiple_of(self.cfg.copy_miss_every);
        if self.cfg.copy_model && !miss {
            if let Some((&last, before)) = content.split_last() {
                if let Some(i) = before.iter().rposition(|&t| t == last) {
                    return content[i + 1];
                }
            }
        }
        (hash(content, 1) % (self.cfg.vocab as u64 - 1)) as u32
    }

    pub fn row_for(&self, content: &[u32]) -> Vec<f32> {
        let best = self.token_for(content);
        let base = hash(content, 2);
        let mut row: Vec<f32> = (0..self.cfg.vocab)
            .map(|i| ((base.wrapping_mul(i as u64 + 1) >> 40) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        row[best as usize] = 10.0;
        row
    }

    pub fn contents(&self, seq: Seq) -> Option<&[u32]> {
        self.seqs.get(&seq).map(|v| v.as_slice())
    }

    pub fn script(&mut self, seq: Seq, tokens: Vec<u32>) {
        self.scripted.insert(seq, tokens);
    }

    pub fn fault_next_step(&mut self, code: u32) {
        self.fail_next = Some(PrimError::Fault(code));
    }

    pub fn fail_next_step(&mut self, e: PrimError) {
        self.fail_next = Some(e);
    }

    fn count_beside(&mut self, feeds: &[Feed<'_>]) {
        let fed: HashSet<u64> = feeds.iter().flat_map(|f| self.cells.get(&f.seq).into_iter().flatten().copied()).collect();
        self.beside.push(self.holders.keys().filter(|id| !fed.contains(id)).count() as u64);
    }

    pub fn poison_next_import(&mut self) {
        self.poison_import = true;
    }

    fn cut_ok(&self, len: u64, cur: u64) -> bool {
        len == 0 || (len <= cur && (self.cfg.truncate_partial || len == cur))
    }
}

impl RuntimePrimitives for FakePrimitives {
    fn sync_steps(&mut self, on: bool) {
        self.synced = Some(on);
    }

    fn describe(&self) -> RuntimeDescriptor {
        RuntimeDescriptor {
            runtime_id: "fake".into(),
            runtime_version: "1".into(),
            max_batch: self.cfg.max_batch,
            max_seqs: self.cfg.max_seqs,
            max_seq_len: self.cfg.max_seq_len,
            vocab_size: self.cfg.vocab,
            page_size_tokens: self.cfg.page,
            kv_bytes_per_token: self.cfg.kv_bytes_per_token,
            cells_total: self.cfg.cells_total,
            truncate_partial: self.cfg.truncate_partial,
            copy_shares_cells: self.cfg.copy_shares_cells,
            takeover_preferred: self.cfg.takeover_preferred,
            seed_min_tokens: self.cfg.seed_min_tokens,
            cache_resident_cells: self.cfg.cache_resident_cells,
            cache_exported_cells: self.cfg.cache_exported_cells,
            recurrent: !self.cfg.truncate_partial,
            export_encodings: vec![encoding::LOSSLESS],
            engine_sampling: self.cfg.engine_sampling,
            engine_draws: false,
            verify_rows: self.cfg.verify_rows,
            verify_batches: true,
            prefill_step_tokens: self.cfg.prefill_step_tokens,
            weights_identity: [self.cfg.identity; 32],
            architecture: "fake".into(),
            backend: "cpu".into(),
            sampling_defaults: self.cfg.sampling_defaults,
        }
    }

    fn is_eos(&self, token: u32) -> bool {
        token == self.cfg.eos
    }

    fn mem_counters(&self) -> MemCounters {
        let cells_used = self.holders.len() as u64;
        MemCounters {
            allocated_bytes: cells_used * self.cfg.kv_bytes_per_token,
            cells_used,
            cells_total: self.cfg.cells_total,
        }
    }

    fn seq_create(&mut self) -> Result<Seq, PrimError> {
        if self.cfg.max_seqs > 0 && self.seqs.len() as u32 >= self.cfg.max_seqs {
            return Err(PrimError::Capacity);
        }
        let s = self.next_seq;
        self.next_seq += 1;
        self.seqs.insert(s, Vec::new());
        self.cells.insert(s, Vec::new());
        Ok(s)
    }

    fn seq_free(&mut self, seq: Seq) {
        self.drop_cells(seq, 0);
        self.cells.remove(&seq);
        self.seqs.remove(&seq);
        self.scripted.remove(&seq);
    }

    fn seq_len(&self, seq: Seq) -> u64 {
        self.seqs.get(&seq).map(|v| v.len() as u64).unwrap_or(0)
    }

    fn cells_released(&self, seqs: &[Seq]) -> u64 {
        let set: HashSet<Seq> = seqs.iter().copied().collect();
        let mut held: HashMap<u64, u32> = HashMap::new();
        for s in &set {
            for &id in self.cells.get(s).into_iter().flatten() {
                *held.entry(id).or_default() += 1;
            }
        }
        held.iter().filter(|(id, n)| self.holders.get(id) == Some(n)).count() as u64
    }

    fn seq_copy(&mut self, src: Seq, dst: Seq, len: u64) -> Result<(), PrimError> {
        let s = self.seqs.get(&src).ok_or(PrimError::UnknownSeq)?;
        if !self.cut_ok(len, s.len() as u64) {
            return Err(PrimError::OutOfBoundary);
        }
        if !self.seqs.contains_key(&dst) {
            return Err(PrimError::UnknownSeq);
        }
        let prefix = s[..len as usize].to_vec();
        if self.cfg.copy_shares_cells {
            let shared: Vec<u64> = self.cells[&src][..len as usize].to_vec();
            self.drop_cells(dst, 0);
            for &id in &shared {
                *self.holders.get_mut(&id).expect("held by src") += 1;
            }
            self.cells.insert(dst, shared);
        } else {
            self.room_for(len, self.cells_released(&[dst]))?;
            self.drop_cells(dst, 0);
            self.take_cells(dst, len);
        }
        *self.seqs.get_mut(&dst).expect("checked") = prefix;
        Ok(())
    }

    fn seq_truncate(&mut self, seq: Seq, new_len: u64) -> Result<(), PrimError> {
        let cur = self.seq_len(seq);
        if !self.seqs.contains_key(&seq) {
            return Err(PrimError::UnknownSeq);
        }
        if !self.cut_ok(new_len, cur) {
            return Err(PrimError::OutOfBoundary);
        }
        self.seqs.get_mut(&seq).expect("checked").truncate(new_len as usize);
        self.drop_cells(seq, new_len as usize);
        Ok(())
    }

    fn seq_boundary(&self, seq: Seq, cap: u64) -> u64 {
        let len = self.seq_len(seq);
        if self.cfg.truncate_partial {
            cap.min(len)
        } else if len <= cap {
            len
        } else {
            0
        }
    }

    fn seq_export(&mut self, seq: Seq, len: u64, enc: u8) -> Result<Vec<u8>, PrimError> {
        if enc != encoding::LOSSLESS {
            return Err(PrimError::Unsupported);
        }
        let s = self.seqs.get(&seq).ok_or(PrimError::UnknownSeq)?;
        if !self.cut_ok(len, s.len() as u64) {
            return Err(PrimError::OutOfBoundary);
        }
        Ok(s[..len as usize].iter().flat_map(|t| t.to_le_bytes()).collect())
    }

    fn seq_import(&mut self, seq: Seq, payload: &[u8]) -> Result<u64, PrimError> {
        let s = self.seqs.get(&seq).ok_or(PrimError::UnknownSeq)?;
        if !s.is_empty() || !payload.len().is_multiple_of(4) {
            return Err(PrimError::OutOfBoundary);
        }
        self.room_for(payload.len() as u64 / 4, 0)?;
        self.take_cells(seq, payload.len() as u64 / 4);
        let s = self.seqs.get_mut(&seq).expect("checked");
        *s = payload.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
        if std::mem::take(&mut self.poison_import) {
            return Err(PrimError::Fatal);
        }
        Ok(s.len() as u64)
    }

    fn step(&mut self, feeds: &[Feed<'_>]) -> Result<Vec<Vec<f32>>, PrimError> {
        if let Some(e) = self.fail_next.take() {
            return Err(e);
        }
        if !self.cfg.step_delay.is_zero() {
            std::thread::sleep(self.cfg.step_delay);
        }
        let mut more = 0u64;
        for f in feeds {
            let Input::Tokens(t) = f.input else { return Err(PrimError::Unsupported) };
            if !self.seqs.contains_key(&f.seq) {
                return Err(PrimError::UnknownSeq);
            }
            more += t.len() as u64;
        }
        self.room_for(more, 0)?;
        self.count_beside(feeds);
        let mut rows = Vec::new();
        for f in feeds {
            let Input::Tokens(t) = f.input else { return Err(PrimError::Unsupported) };
            self.take_cells(f.seq, t.len() as u64);
            self.feeds.push((t.len(), f.wants_row));
            let s = self.seqs.get_mut(&f.seq).expect("checked");
            s.extend_from_slice(t);
            if f.wants_row {
                let content = s.clone();
                let mut row = self.row_for(&content);
                if let Some(script) = self.scripted.get_mut(&f.seq) {
                    if !script.is_empty() {
                        let next = script.remove(0);
                        for x in row.iter_mut() {
                            *x = -1.0;
                        }
                        row[next as usize] = 10.0;
                    }
                }
                rows.push(row);
            }
        }
        Ok(rows)
    }

    fn step_rows(&mut self, feeds: &[Feed<'_>]) -> Result<Option<Vec<Vec<Vec<f32>>>>, PrimError> {
        if !self.cfg.verify_rows {
            return Ok(None);
        }
        if let Some(e) = self.fail_next.take() {
            return Err(e);
        }
        if !self.cfg.step_delay.is_zero() {
            std::thread::sleep(self.cfg.step_delay);
        }
        let mut more = 0u64;
        for f in feeds {
            let Input::Tokens(t) = f.input else { return Err(PrimError::Unsupported) };
            if !self.seqs.contains_key(&f.seq) {
                return Err(PrimError::UnknownSeq);
            }
            more += t.len() as u64;
        }
        self.room_for(more, 0)?;
        self.count_beside(feeds);
        let mut out = Vec::new();
        for f in feeds {
            let Input::Tokens(t) = f.input else { return Err(PrimError::Unsupported) };
            self.take_cells(f.seq, t.len() as u64);
            self.feeds.push((t.len(), f.wants_row));
            let s = self.seqs.get_mut(&f.seq).expect("checked");
            let base = s.len();
            s.extend_from_slice(t);
            if f.wants_row {
                let content = s.clone();
                out.push((1..=t.len()).map(|i| self.row_for(&content[..base + i])).collect());
            }
        }
        Ok(Some(out))
    }

    fn step_sampled(&mut self, feeds: &[Feed<'_>], specs: &[SampleSpec]) -> Result<Option<Vec<u32>>, PrimError> {
        if !self.cfg.engine_sampling || specs.len() != feeds.len() || specs.iter().any(|s| s.temperature > 0.0) {
            return Ok(None);
        }
        let rows = self.step(feeds)?;
        Ok(Some(rows.iter().map(|r| crate::sampling::argmax(r)).collect()))
    }

    fn vocabulary(&self) -> Option<Vocabulary> {
        self.cfg.vocabulary.clone()
    }
}
