//! `RuntimePrimitives`.

pub type Seq = u64;

/// The most rows `step_rows` answers for one feed: a verified draft of
/// `VERIFY_ROWS_MAX - 1` tokens and the token before it.
pub const VERIFY_ROWS_MAX: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimError {
    Unsupported,
    UnknownSeq,
    OutOfBoundary,
    Capacity,
    Fault(u32),
    Fatal,
}

#[derive(Debug, Clone)]
pub struct RuntimeDescriptor {
    pub runtime_id: String,
    pub runtime_version: String,
    pub max_batch: u32,
    pub max_seqs: u32,
    pub max_seq_len: u64,
    pub vocab_size: u32,
    pub page_size_tokens: u32,
    pub kv_bytes_per_token: u64,
    pub cells_total: u64,
    pub truncate_partial: bool,
    pub copy_shares_cells: bool,
    pub takeover_preferred: bool,
    pub seed_min_tokens: u64,
    /// Cells the prefix cache may keep in the pool: the state its entries
    /// hold that no lane or bare sequence shares. `0` = no bound: an entry at
    /// rest costs the other sequences nothing. A runtime whose steps cost by
    /// what its pool holds names what it affords (llama.cpp's unified cache
    /// runs every step's attention over the cells up to the highest one in
    /// use, whoever holds them); past it the least recently used entries
    /// leave the pool, exported or evicted.
    pub cache_resident_cells: u64,
    /// Tokens of state the prefix cache may hold out of the pool as lossless exports.
    /// `0` = none: an entry that leaves the pool is evicted.
    pub cache_exported_cells: u64,
    pub recurrent: bool,
    pub export_encodings: Vec<u8>,
    pub engine_sampling: bool,
    pub engine_draws: bool,
    /// `step_rows` answers a row for every position of a feed: what
    /// verifying a draft in one pass needs.
    pub verify_rows: bool,
    /// `step_rows` runs every lane's draft in one pass; a runtime that runs
    /// them one lane at a time drafts only while one lane decodes.
    pub verify_batches: bool,
    /// The most prompt tokens one prefill step runs (the runtime's own
    /// micro-batch); a tick asked to yield stops between such steps. 0: a
    /// tick's prefill is one step.
    pub prefill_step_tokens: u32,
    pub weights_identity: [u8; 32],
    pub architecture: String,
    pub backend: String,
    pub sampling_defaults: SamplingDefaults,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub do_sample: Option<bool>,
}

impl SamplingDefaults {
    pub fn from_generation_config(v: &serde_json::Value) -> SamplingDefaults {
        let f = |k: &str| v.get(k).and_then(serde_json::Value::as_f64);
        SamplingDefaults {
            temperature: f("temperature"),
            top_p: f("top_p"),
            top_k: v.get("top_k").and_then(serde_json::Value::as_u64).map(|x| x as u32),
            min_p: f("min_p"),
            repetition_penalty: f("repetition_penalty"),
            do_sample: v.get("do_sample").and_then(serde_json::Value::as_bool),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemCounters {
    pub allocated_bytes: u64,
    pub cells_used: u64,
    pub cells_total: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Vocabulary {
    pub tokens: Vec<Vec<u8>>,
    pub specials: Vec<(String, u32)>,
    pub eos: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct Feed<'a> {
    pub seq: Seq,
    pub input: Input<'a>,
    pub wants_row: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Input<'a> {
    Tokens(&'a [u32]),
    Media { handle: u64, offset: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleSpec {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub rng_position: u64,
}

pub trait RuntimePrimitives {
    fn describe(&self) -> RuntimeDescriptor;
    fn is_eos(&self, token: u32) -> bool;
    fn mem_counters(&self) -> MemCounters;

    fn seq_create(&mut self) -> Result<Seq, PrimError>;
    fn seq_free(&mut self, seq: Seq);
    fn seq_len(&self, seq: Seq) -> u64;
    fn cells_released(&self, seqs: &[Seq]) -> u64 {
        let mut seen = std::collections::HashSet::new();
        seqs.iter().filter(|s| seen.insert(**s)).map(|&s| self.seq_len(s)).sum()
    }
    fn seq_copy(&mut self, src: Seq, dst: Seq, len: u64) -> Result<(), PrimError>;
    fn seq_truncate(&mut self, seq: Seq, new_len: u64) -> Result<(), PrimError>;
    fn seq_boundary(&self, seq: Seq, cap: u64) -> u64;
    fn seq_export(&mut self, seq: Seq, len: u64, encoding: u8) -> Result<Vec<u8>, PrimError>;
    fn seq_import(&mut self, seq: Seq, payload: &[u8]) -> Result<u64, PrimError>;

    fn step(&mut self, feeds: &[Feed<'_>]) -> Result<Vec<Vec<f32>>, PrimError>;
    fn step_sampled(&mut self, _feeds: &[Feed<'_>], _specs: &[SampleSpec]) -> Result<Option<Vec<u32>>, PrimError> {
        Ok(None)
    }

    /// `step`, answering for each feed that wants rows a row after every
    /// one of its tokens (in order), not only its last. `Ok(None)`: not
    /// served, and nothing moved.
    fn step_rows(&mut self, _feeds: &[Feed<'_>]) -> Result<Option<Vec<Vec<Vec<f32>>>>, PrimError> {
        Ok(None)
    }

    /// Whether a step that reads no row is done when it returns, rather than queued.
    fn sync_steps(&mut self, _on: bool) {}

    fn shed(&mut self, _bytes_target: u64) -> u64 {
        0
    }

    fn vocabulary(&self) -> Option<Vocabulary> {
        None
    }
}
