//! `#[repr(C)]` mirrors of every struct in `include/baseRT/baseRT_tick.h`, plus the named constants
//! those structs carry in fixed-width fields.

use std::ffi::c_void;

use crate::array::AbiRecord;

pub mod sampling {
    pub const GPU_GREEDY: u32 = 0;
    pub const GPU_GUMBEL: u32 = 1;
    pub const HOST: u32 = 2;
}

pub mod determinism {
    pub const BEST_EFFORT: u8 = 0;
    pub const DETERMINISTIC: u8 = 1;
}

pub mod exactness {
    pub const APPROXIMATE: u8 = 0;
    pub const DISTRIBUTION_EXACT: u8 = 1;
    pub const SEED_PATH_INVARIANT: u8 = 2;
}

pub mod tick_flags {
    pub const DRAIN: u32 = 1 << 0;
    pub const CAPTURE_OK: u32 = 1 << 1;
    pub const PARTIAL_EMITS: u32 = 1 << 2;
}

pub mod tick_status {
    pub const OK: u32 = 0;
    pub const SHED: u32 = 1;
    pub const LANE_ERRORS: u32 = 2;
    pub const FATAL: u32 = 3;
}

pub mod admit_status {
    pub const UNSPECIFIED: u32 = 0;
    pub const ADMITTED: u32 = 1;
    pub const REJECTED: u32 = 2;
}

pub mod finish {
    pub const NONE: u32 = 0;
    pub const EOS: u32 = 1;
    pub const LENGTH: u32 = 2;
    pub const GRAMMAR: u32 = 3;
    pub const CANCELLED: u32 = 4;
    pub const ERROR: u32 = 5;
}

pub mod op_state {
    pub const PENDING: u8 = 0;
    pub const RUNNING: u8 = 1;
    pub const DONE: u8 = 2;
    pub const FAILED: u8 = 3;
    pub const CANCELLED: u8 = 4;
}

pub mod encoding {
    pub const LOSSLESS: u8 = 0;
    pub const Q8: u8 = 1;
    pub const Q4: u8 = 2;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaInfo {
    pub n_tokens: u32,
    pub image_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub preprocess_fp: u64,
}

pub mod space_kind {
    pub const UNSPECIFIED: u32 = 0;
    pub const PAGED_TOKEN_KV: u32 = 1;
    pub const RING_KV: u32 = 2;
    pub const RECURRENT_BLOB: u32 = 3;
    pub const DEPTH_PAGED_KV: u32 = 4;
    pub const ENCODER_CACHE: u32 = 5;
}

pub mod space_flag {
    pub const OPS_UNAVAILABLE: u32 = 1 << 0;
    pub const NO_PREFIX_CACHE: u32 = 1 << 1;
}

pub mod artifact_role {
    pub const UNSPECIFIED: u32 = 0;
    pub const DRAFT_WEIGHTS: u32 = 1;
    pub const AUX_HEAD: u32 = 2;
    pub const PROJECTION: u32 = 3;
    pub const DRAFT_TOKENIZER: u32 = 4;
}

pub mod tap_tensor {
    pub const UNSPECIFIED: u32 = 0;
    pub const HIDDEN: u32 = 1;
    pub const ATTN_OUT: u32 = 2;
    pub const EMBED: u32 = 3;
}

pub mod cert_mode {
    pub const GREEDY: u8 = 1 << 0;
    pub const GUMBEL: u8 = 1 << 1;
    pub const HOST: u8 = 1 << 2;
}

pub mod cert_domain {
    pub const NONE: u32 = 0;
    pub const GREEDY_ONLY: u32 = 1;
    pub const PENALTY_FREE: u32 = 2;
    pub const ANY: u32 = 3;
}

pub mod strategy_cap {
    pub const UNSPECIFIED: u32 = 0;
    pub const PROPOSAL_LINEAR: u32 = 1;
    pub const PROPOSAL_BLOCK: u32 = 2;
    pub const PROPOSAL_TREE: u32 = 3;
    pub const MASK_ANCESTOR: u32 = 4;
    pub const HOST_COMPATIBLE: u32 = 5;
}

pub mod taint {
    pub const QUANTIZED_DEMOTION: u32 = 1 << 0;
    pub const DECOMPOSITION: u32 = 1 << 1;
}

pub mod tier {
    pub const GPU: u8 = 0;
    pub const HOST: u8 = 1;
    pub const GONE: u8 = 2;
}

pub mod fault {
    pub const UNSPECIFIED: u32 = 0;
    pub const GRAMMAR_OVERFLOW: u32 = 1;
    pub const REF_SPAN_OOB: u32 = 2;
    pub const STRATEGY: u32 = 3;
    pub const OOM_TENTATIVE: u32 = 4;
    pub const INTERNAL: u32 = 5;
    pub const MEDIA: u32 = 6;
}

pub mod shed_kind {
    pub const UNSPECIFIED: u32 = 0;
    pub const PREFILL_CHUNK: u32 = 1;
    pub const CACHE_EVICTION: u32 = 2;
}

pub mod fork_flags {
    pub const EAGER: u32 = 1 << 0;
    /// superfluid's own, outside the C ABI: a copy for one request (a chat's next turn),
    /// sent only to a runtime that asks for prefix checkpoints.
    pub const PRIVATE: u32 = 1 << 16;
}

pub mod checksum_kind {
    pub const XXH3_64: u32 = 1;
    pub const FNV1A64: u32 = 2;
}

pub const ALL_SPACES: u32 = 0xFFFF_FFFF;
pub const STATE_ENVELOPE_MAGIC: u64 = 0x4554_4154_5354_5242;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Array {
    pub data: *const c_void,
    pub count: u32,
    pub elem_size: u32,
}

impl Array {
    pub const EMPTY: Array = Array {
        data: std::ptr::null(),
        count: 0,
        elem_size: 0,
    };
}

// SAFETY: Array is a plain (pointer, count, stride) view; sending it
// across threads is safe — dereferencing it is the unsafe act, guarded by
// `array::read_array`'s contract.
unsafe impl Send for Array {}
// SAFETY: as above; shared references never mutate through the view.
unsafe impl Sync for Array {}

impl Default for Array {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenRef {
    pub ring_id: u32,
    pub index: u32,
    pub count: u32,
    pub _pad0: u32,
    pub generation: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RingRef {
    pub ring_id: u32,
    pub index: u32,
    pub generation: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Buf {
    pub ptr: *mut c_void,
    pub len: u64,
}

// SAFETY: plain (pointer, len) view; dereferencing is the guarded act.
unsafe impl Send for Buf {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenRange {
    pub start: u64,
    pub end: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub freq_penalty: f32,
    pub presence_penalty: f32,
    pub repeat_penalty: f32,
    pub flags: u32,
}

pub mod sampling_flag {
    pub const IGNORE_EOS: u32 = 1 << 0;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ShedPolicy {
    pub victim_lanes: Array,
    pub evictable_cache_classes: u64,
    pub protected_quota_bytes: u64,
    pub max_evict_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LaneAdmit {
    pub lane_tag: u64,
    pub prompt: TokenRef,
    pub seed_handle: u64,
    pub sampling: u32,
    pub logit_bias_handle: u32,
    pub params: SamplingParams,
    pub rng_counter_base: u64,
    pub grammar_handle: u32,
    pub strategy_slot: u32,
    pub determinism_class: u8,
    pub minimum_exactness: u8,
    pub allow_approximate: u8,
    pub want_logprobs: u8,
    pub required_cert_id: u32,
    pub host_sampler_identity: [u8; 32],
    pub grammar_replay: u32,
    pub decode_replay: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneCommit {
    pub lane_tag: u64,
    pub token_id: u32,
    pub _pad0: u32,
    pub logits_nonce: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LanePrefill {
    pub lane_tag: u64,
    pub token_offset: u32,
    pub token_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneDecode {
    pub lane_tag: u64,
    pub max_new_tokens: u16,
    pub overshoot: u16,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneRetire {
    pub lane_tag: u64,
    pub publish_to_cache: u8,
    pub _pad0: [u8; 7],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TickPlan {
    pub struct_size: u64,
    pub plan_seq: u64,
    pub flags: u32,
    pub _pad0: u32,

    pub prefill_token_budget: u32,
    pub max_decode_lanes: u32,

    pub admits: Array,
    pub commits: Array,
    pub prefills: Array,
    pub decodes: Array,
    pub retires: Array,

    pub shed_policy: ShedPolicy,

    pub on_partial_emit:
        Option<unsafe extern "C" fn(user: *mut c_void, lane_tag: u64, tokens: *const u32, n_tokens: u32)>,
    pub partial_emit_user: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdmitResult {
    pub lane_tag: u64,
    pub status: u32,
    pub reject_code: u32,
    pub cert_id: u32,
    pub granted_class: u8,
    pub _pad0: [u8; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecStats {
    pub proposed: u32,
    pub accepted: u32,
    pub cert_id_in_effect: u32,
    pub _pad0: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneEmit {
    pub lane_tag: u64,
    pub token_ref: TokenRef,
    pub n_tokens: u32,
    pub finish: u32,
    pub logits_row: RingRef,
    pub spec: SpecStats,
}

pub const MAX_TOP_LOGPROBS: usize = 20;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneLogprob {
    pub lane_tag: u64,
    pub logprob: f32,
    pub n_top: u32,
    pub top_ids: [u32; MAX_TOP_LOGPROBS],
    pub top_logprobs: [f32; MAX_TOP_LOGPROBS],
}

impl Default for LaneLogprob {
    fn default() -> Self {
        LaneLogprob {
            lane_tag: 0,
            logprob: 0.0,
            n_top: 0,
            top_ids: [0; MAX_TOP_LOGPROBS],
            top_logprobs: [0.0; MAX_TOP_LOGPROBS],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneFault {
    pub lane_tag: u64,
    pub code: u32,
    pub _pad0: u32,
    pub detail: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShedEntry {
    pub lane_tag: u64,
    pub kind: u32,
    pub reason: u32,
    pub bytes: u64,
    pub tokens: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ShedReport {
    pub prefill_chunks_dropped: u32,
    pub evictions_performed: u32,
    pub bytes_evicted: u64,
    pub entries: Array,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpComplete {
    pub op: u64,
    pub state: u8,
    pub _pad0: [u8; 3],
    pub error: u32,
    pub bytes_moved: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TickTimings {
    pub wall_ns: u64,
    pub prefill_ns: u64,
    pub decode_ns: u64,
    pub graph_hits: u32,
    pub graph_misses: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TickMemCounters {
    pub allocated_bytes: u64,
    pub host_retained_bytes: u64,
    pub pool_blocks_total: u64,
    pub pool_blocks_used: u64,
    pub pool_bytes_evictable: u64,
    pub tentative_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TickEvents {
    pub struct_size: u64,
    pub plan_seq: u64,
    pub tick_status: u32,
    pub _pad0: u32,

    pub admit_results: Array,
    pub emits: Array,
    pub faults: Array,

    pub shed: ShedReport,
    pub op_completions: Array,

    pub timings: TickTimings,
    pub mem: TickMemCounters,
    pub logprobs: Array,
}

pub const RING_MAGIC: u64 = 0x4252_5452_494E_4731;

pub mod ring_kind {
    pub const UNSPECIFIED: u32 = 0;
    pub const TOKENS: u32 = 1;
    pub const LOGITS: u32 = 2;
}

pub mod ring_role {
    pub const UNSPECIFIED: u32 = 0;
    pub const ENGINE_READS: u32 = 1;
    pub const ENGINE_WRITES: u32 = 2;
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RingDesc {
    pub struct_size: u64,
    pub ring_id: u32,
    pub kind: u32,
    pub role: u32,
    pub slots: u32,
    pub slot_bytes: u32,
    pub _pad0: u32,
    pub base: *mut c_void,
    pub len: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StateSpaceDesc {
    pub space_id: u32,
    pub kind: u32,
    pub version_tag: u64,
    pub bytes_per_token: u64,
    pub blob_bytes: u64,
    pub page_size_tokens: u32,
    pub fork_cost_class: u32,
    pub fork_cost_bytes: u64,
    pub snapshot_cadence: u32,
    pub snapshot_interval_tokens: u32,
    pub placement: u32,
    pub flags: u32,
    pub name: [u8; 32],
}

impl Default for StateSpaceDesc {
    fn default() -> Self {
        // SAFETY: all-integer struct; the all-zero pattern is valid.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Artifact {
    pub role: u32,
    pub _pad0: u32,
    pub content_hash: [u8; 32],
    pub byte_size: u64,
    pub load_path: *const std::ffi::c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TapSpec {
    pub layer: u32,
    pub tensor: u32,
    pub dtype: u32,
    pub layout: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CapabilityReq {
    pub kind_id: u32,
    pub _pad0: u32,
    pub params: Array,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StrategyRegistration {
    pub struct_size: u64,
    pub strategy_id: *const std::ffi::c_char,
    pub impl_version: *const std::ffi::c_char,
    pub impl_hash: [u8; 32],
    pub config_hash: [u8; 32],

    pub artifacts: Array,
    pub taps: Array,
    pub capabilities: Array,
    pub target_archs: Array,

    pub kernel_caps_required: u32,
    pub _pad0: u32,
    pub est_state_bytes: u64,

    pub claimed_exactness: u8,
    pub _pad1: [u8; 3],
    pub rng_contract_version: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ExactnessCert {
    pub cert_id: u32,
    pub exactness: u8,
    pub sampling_modes: u8,
    pub grammar_allowed: u8,
    pub logit_bias_allowed: u8,
    pub param_domain: u32,
    pub verification_shape_class: u32,
    pub admissible_host_identities: Array,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StrategyGrant {
    pub struct_size: u64,
    pub strategy_slot: u32,
    pub _pad0: u32,
    pub certificates: Array,
    pub state_spaces: Array,
    pub reserved_bytes: u64,
}

pub type OpHandle = u64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpStatus {
    pub struct_size: u64,
    pub state: u8,
    pub _pad0: [u8; 3],
    pub error: u32,
    pub bytes_moved: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MatchCandidate {
    pub prefix_len: u64,
    pub provenance_digest: [u8; 32],
    pub taint_bits: u32,
    pub resident_tier: u8,
    pub _pad0: [u8; 3],
}

impl Default for MatchCandidate {
    fn default() -> Self {
        // SAFETY: all-integer struct; the all-zero pattern is valid.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SpaceMatch {
    pub space_id: u32,
    pub _pad0: u32,
    pub candidates: Array,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MatchResult {
    pub struct_size: u64,
    pub spaces: Array,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StateEnvelope {
    pub magic: u64,
    pub envelope_version: u32,
    pub space_kind: u32,
    pub compat_identity: [u8; 32],
    pub version_tag: u64,
    pub provenance_digest: [u8; 32],
    pub taint_bits: u32,
    pub _pad0: u32,
    pub range: TokenRange,
    pub encoding: u8,
    pub _pad1: [u8; 3],
    pub checksum_kind: u32,
    pub payload_len: u64,
    pub content_checksum: u64,
}

impl Default for StateEnvelope {
    fn default() -> Self {
        // SAFETY: all-integer struct; the all-zero pattern is valid.
        unsafe { std::mem::zeroed() }
    }
}

macro_rules! abi_record {
    ($($t:ty),+ $(,)?) => {
        $(
            // SAFETY: every field is a fixed-width integer/float; any bit
            // pattern (including all-zeros) is a valid value, and the type
            // is #[repr(C)] with explicit padding.
            unsafe impl AbiRecord for $t {
                const MIN_PREFIX: usize = std::mem::size_of::<$t>();
            }
        )+
    };
}

// SAFETY: as `abi_record!`. The admit grew past its v1 size (136 bytes,
// through `host_sampler_identity`); a writer of that size still decodes,
// with the appended fields zero.
unsafe impl AbiRecord for LaneAdmit {
    const MIN_PREFIX: usize = std::mem::offset_of!(LaneAdmit, grammar_replay);
}

abi_record!(
    LaneCommit,
    LanePrefill,
    LaneDecode,
    LaneRetire,
    AdmitResult,
    LaneEmit,
    LaneLogprob,
    LaneFault,
    ShedEntry,
    OpComplete,
    StateSpaceDesc,
    TapSpec,
    MatchCandidate,
    OpStatus,
);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheKey {
    pub compat_key: [u8; 32],
    pub provenance_digest: [u8; 32],
}

impl Default for CacheKey {
    fn default() -> Self {
        // SAFETY: all-integer struct; the all-zero pattern is valid.
        unsafe { std::mem::zeroed() }
    }
}

// SAFETY: u64 lane tags / handles ride in ABI arrays directly.
unsafe impl AbiRecord for u64 {
    const MIN_PREFIX: usize = 8;
}

// SAFETY: raw byte payloads (capability params) ride in ABI arrays with
// elem_size 1; the alignment rule constrains only the array base pointer.
unsafe impl AbiRecord for u8 {
    const MIN_PREFIX: usize = 1;
}

// SAFETY: a 32-byte hash is plain bytes; every bit pattern is valid.
unsafe impl AbiRecord for [u8; 32] {
    const MIN_PREFIX: usize = 32;
}

abi_record!(
    Artifact,
    CapabilityReq,
    ExactnessCert,
    SpaceMatch,
    ShedPolicy,
    CacheKey,
);

abi_record!(
    TickPlan,
    TickEvents,
    StrategyRegistration,
    StrategyGrant,
    MatchResult,
    StateEnvelope,
);
