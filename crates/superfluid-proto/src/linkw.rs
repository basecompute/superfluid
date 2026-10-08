use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::handshake::StateSpaceDescMsg;

pub mod msg_type {
    pub const HELLO: u16 = 0x0001;
    pub const HELLO_ACK_W: u16 = 0x0002;
    pub const RING_ATTACH: u16 = 0x0003;
    pub const RING_ACK: u16 = 0x0004;

    pub const TICK_SUBMIT: u16 = 0x0005;
    pub const TICK_RESULT: u16 = 0x0006;

    pub const MATCH_REQ: u16 = 0x0007;
    pub const MATCH_RES: u16 = 0x0008;
    pub const SEED_ACQUIRE: u16 = 0x0009;
    pub const SEED_GRANT: u16 = 0x000A;
    pub const SEED_RELEASE: u16 = 0x000B;

    pub const STATE_OP_SUBMIT: u16 = 0x000C;
    pub const STATE_OP_ACCEPTED: u16 = 0x000D;
    pub const STATE_OP_DONE: u16 = 0x000E;
    pub const OP_CANCEL: u16 = 0x000F;

    pub const STRATEGY_REGISTER: u16 = 0x0010;
    pub const STRATEGY_GRANT: u16 = 0x0011;

    pub const STATE_SYNC_REQ: u16 = 0x0012;
    pub const STATE_SYNC_RES: u16 = 0x0013;

    pub const BUF_ALLOC: u16 = 0x0014;
    pub const BUF_ATTACH: u16 = 0x0015;
    pub const BUF_RELEASE: u16 = 0x0016;

    pub const PING: u16 = 0x0017;
    pub const PONG: u16 = 0x0018;
    pub const DRAIN_REQ: u16 = 0x0019;
    pub const DRAINED: u16 = 0x001A;

    pub const TRANSCRIBE_SEGMENT: u16 = 0x001B;

    pub const TRANSCRIBE_CANCEL: u16 = 0x001C;
    pub const TICK_EMIT: u16 = 0x001D;
    /// The daemon's control words for the worker, a segment of their own
    /// (the rings' headers are the engine ABI's); its fd follows on the fd
    /// channel under `superfluid_shm::CONTROL_TAG`.
    pub const CONTROL_ATTACH: u16 = 0x001E;
    pub const CONTROL_ACK: u16 = 0x001F;

    pub const TELEMETRY: u16 = 0x8001;

    pub const ALL: &[u16] = &[
        HELLO,
        HELLO_ACK_W,
        RING_ATTACH,
        RING_ACK,
        TICK_SUBMIT,
        TICK_RESULT,
        MATCH_REQ,
        MATCH_RES,
        SEED_ACQUIRE,
        SEED_GRANT,
        SEED_RELEASE,
        STATE_OP_SUBMIT,
        STATE_OP_ACCEPTED,
        STATE_OP_DONE,
        OP_CANCEL,
        STRATEGY_REGISTER,
        STRATEGY_GRANT,
        STATE_SYNC_REQ,
        STATE_SYNC_RES,
        BUF_ALLOC,
        BUF_ATTACH,
        BUF_RELEASE,
        PING,
        PONG,
        DRAIN_REQ,
        DRAINED,
        TRANSCRIBE_SEGMENT,
        TICK_EMIT,
        TRANSCRIBE_CANCEL,
        CONTROL_ATTACH,
        CONTROL_ACK,
        TELEMETRY,
    ];
}

pub fn is_known_type(msg_type: u16) -> bool {
    msg_type::ALL.contains(&msg_type)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RingKind {
    Tokens,
    Logits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingSpec {
    pub ring_id: u32,
    pub kind: RingKind,
    pub slot_bytes: u32,
    pub slots: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkWLimits {
    pub max_frame: u32,
    pub op_window: u32,
    pub ring_specs: Vec<RingSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingAttachMsg {
    pub ring_id: u32,
    pub layout: RingSpec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingAckMsg {
    pub ring_id: u32,
    pub generation_base: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlAttachMsg {
    pub bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlAckMsg {
    pub bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenRefMsg {
    pub ring_id: u32,
    pub index: u32,
    pub count: u32,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingRefMsg {
    pub ring_id: u32,
    pub index: u32,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenRangeMsg {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SamplingParamsMsg {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub freq_penalty: f32,
    pub presence_penalty: f32,
    pub repeat_penalty: f32,
    #[serde(default)]
    pub flags: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaneAdmitMsg {
    pub lane_tag: u64,
    pub prompt: TokenRefMsg,
    pub seed_handle: u64,
    pub sampling: u32,
    pub params: SamplingParamsMsg,
    pub rng_counter_base: u64,
    pub grammar_handle: u32,
    pub logit_bias_handle: u32,
    pub want_logprobs: bool,
    pub top_logprobs: u8,
    pub strategy_slot: u32,
    pub determinism_class: u8,
    pub minimum_exactness: u8,
    pub allow_approximate: bool,
    pub required_cert_id: u32,
    pub host_sampler_identity: [u8; 32],
    pub grammar_replay: u32,
    pub decode_replay: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneCommitMsg {
    pub lane_tag: u64,
    pub token_id: u32,
    pub logits_nonce: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanePrefillMsg {
    pub lane_tag: u64,
    pub token_offset: u32,
    pub token_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneDecodeMsg {
    pub lane_tag: u64,
    pub max_new_tokens: u16,
    pub overshoot: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneRetireMsg {
    pub lane_tag: u64,
    pub publish_to_cache: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShedPolicyMsg {
    pub victim_lanes: Vec<u64>,
    pub evictable_cache_classes: u64,
    pub protected_quota_bytes: u64,
    pub max_evict_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickPlanMsg {
    pub plan_seq: u64,
    pub flags: u32,
    pub prefill_token_budget: u32,
    pub max_decode_lanes: u32,
    pub admits: Vec<LaneAdmitMsg>,
    pub commits: Vec<LaneCommitMsg>,
    pub prefills: Vec<LanePrefillMsg>,
    pub decodes: Vec<LaneDecodeMsg>,
    pub retires: Vec<LaneRetireMsg>,
    pub shed_policy: ShedPolicyMsg,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TickSubmitMsg {
    pub plan: TickPlanMsg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmitResultMsg {
    pub lane_tag: u64,
    pub status: u32,
    pub reject_code: u32,
    pub cert_id: u32,
    pub granted_class: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecStatsMsg {
    pub proposed: u32,
    pub accepted: u32,
    pub cert_id_in_effect: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneEmitMsg {
    pub lane_tag: u64,
    pub token_ref: TokenRefMsg,
    pub n_tokens: u32,
    pub finish: u32,
    pub logits_row: RingRefMsg,
    pub spec: SpecStatsMsg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneFaultMsg {
    pub lane_tag: u64,
    pub code: u32,
    pub detail: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShedEntryMsg {
    pub lane_tag: u64,
    pub kind: u32,
    pub reason: u32,
    pub bytes: u64,
    pub tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShedReportMsg {
    pub prefill_chunks_dropped: u32,
    pub evictions_performed: u32,
    pub bytes_evicted: u64,
    pub entries: Vec<ShedEntryMsg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpCompleteMsg {
    pub op: u64,
    pub state: u8,
    pub error: u32,
    pub bytes_moved: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickTimingsMsg {
    pub wall_ns: u64,
    pub prefill_ns: u64,
    pub decode_ns: u64,
    pub graph_hits: u32,
    pub graph_misses: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemCountersMsg {
    pub allocated_bytes: u64,
    pub host_retained_bytes: u64,
    pub pool_blocks_total: u64,
    pub pool_blocks_used: u64,
    pub pool_bytes_evictable: u64,
    pub tentative_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickEventsMsg {
    pub plan_seq: u64,
    pub tick_status: u32,
    pub admit_results: Vec<AdmitResultMsg>,
    pub emits: Vec<LaneEmitMsg>,
    pub faults: Vec<LaneFaultMsg>,
    pub shed: ShedReportMsg,
    pub op_completions: Vec<OpCompleteMsg>,
    pub timings: TickTimingsMsg,
    pub mem: MemCountersMsg,
    #[serde(default)]
    pub logprobs: Vec<LaneLogprobMsg>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneLogprobMsg {
    pub lane_tag: u64,
    pub logprob_bits: u32,
    #[serde(default)]
    pub top_ids: Vec<u32>,
    #[serde(default)]
    pub top_logprob_bits: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum TickResultMsg {
    Events(TickEventsMsg),
    Rejected { status: i32 },
}

pub const ALL_SPACES: u32 = 0xFFFF_FFFF;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchReqMsg {
    pub space_id: u32,
    pub span: TokenRefMsg,
    pub media_deps: Vec<[u8; 32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchCandidateMsg {
    pub prefix_len: u64,
    pub provenance_digest: [u8; 32],
    pub taint_bits: u32,
    pub resident_tier: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpaceMatchMsg {
    pub space_id: u32,
    pub candidates: Vec<MatchCandidateMsg>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchResMsg {
    pub spaces: Vec<SpaceMatchMsg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedAcquireMsg {
    pub span: TokenRefMsg,
    pub prefix_len: u64,
    pub determinism_class: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeedGrantMsg {
    Granted {
        seed_handle: u64,
        ttl_ms: u32,
        ttl_ticks: Option<u64>,
    },
    Refused {
        status: i32,
    },
}

#[derive(Deserialize)]
enum LegacySeedGrantMsg {
    Granted { seed_handle: u64, ttl_ms: u32 },
    Refused { status: i32 },
}

impl SeedGrantMsg {
    pub fn decode(bytes: &[u8]) -> Result<SeedGrantMsg, postcard::Error> {
        match postcard::from_bytes::<SeedGrantMsg>(bytes) {
            Ok(g) => Ok(g),
            Err(e) => match postcard::from_bytes::<LegacySeedGrantMsg>(bytes) {
                Ok(LegacySeedGrantMsg::Granted {
                    seed_handle,
                    ttl_ms,
                }) => Ok(SeedGrantMsg::Granted {
                    seed_handle,
                    ttl_ms,
                    ttl_ticks: None,
                }),
                Ok(LegacySeedGrantMsg::Refused { status }) => Ok(SeedGrantMsg::Refused { status }),
                Err(_) => Err(e),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedReleaseMsg {
    pub seed_handle: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateOpKind {
    Snapshot {
        sequence: u64,
        space_id: u32,
        boundary_pos: u64,
        buf_id: u64,
        sizing_gen: u64,
    },
    Restore {
        sequence: u64,
        space_id: u32,
        buf_id: u64,
    },
    Demote {
        sequence: u64,
        space_id: u32,
        range: TokenRangeMsg,
        encoding: u8,
        buf_id: u64,
        sizing_gen: u64,
    },
    Promote {
        sequence: u64,
        space_id: u32,
        range: TokenRangeMsg,
        buf_id: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateOpSubmitMsg {
    pub op: StateOpKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateOpAcceptedMsg {
    Accepted { op_handle: u64 },
    Rejected { status: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateOpDoneMsg {
    pub op_handle: u64,
    pub state: u8,
    pub error: u32,
    pub bytes_moved: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpCancelMsg {
    pub op_handle: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactMsg {
    pub role: u32,
    pub content_hash: [u8; 32],
    pub byte_size: u64,
    pub load_path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TapSpecMsg {
    pub layer: u32,
    pub tensor: u32,
    pub dtype: u32,
    pub layout: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityReqMsg {
    pub kind_id: u32,
    pub params: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrategyRegisterMsg {
    pub strategy_id: String,
    pub impl_version: String,
    pub impl_hash: [u8; 32],
    pub config_hash: [u8; 32],
    pub artifacts: Vec<ArtifactMsg>,
    pub taps: Vec<TapSpecMsg>,
    pub capabilities: Vec<CapabilityReqMsg>,
    pub target_archs: Vec<String>,
    pub kernel_caps_required: u32,
    pub est_state_bytes: u64,
    pub claimed_exactness: u8,
    pub rng_contract_version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaInfoMsg {
    pub n_tokens: u32,
    pub image_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub preprocess_fp: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactnessCertMsg {
    pub cert_id: u32,
    pub exactness: u8,
    pub sampling_modes: u8,
    pub param_domain: u32,
    pub grammar_allowed: bool,
    pub logit_bias_allowed: bool,
    pub verification_shape_class: u32,
    pub admissible_host_identities: Vec<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyGrantMsg {
    Granted(StrategyGrantOkMsg),
    Refused { status: i32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrategyGrantOkMsg {
    pub strategy_slot: u32,
    pub certificates: Vec<ExactnessCertMsg>,
    pub state_spaces: Vec<StateSpaceDescMsg>,
    pub reserved_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptSegmentMsg {
    pub start_ms: i32,
    pub end_ms: i32,
    pub text: String,
    pub avg_logprob_bits: u32,
    pub no_speech_prob_bits: u32,
    pub compression_ratio_bits: u32,
    pub temperature_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscribeSegmentMsg {
    pub start_ms: i32,
    pub end_ms: i32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscribeCancelMsg {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickEmitMsg {
    pub lane_tag: u64,
    pub tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateSyncReqMsg {
    Fork {
        parent_sequence: u64,
        flags: u32,
    },
    Trim {
        sequence: u64,
        space_id: u32,
        new_len: u64,
    },
    ExportSize {
        sequence: u64,
        space_id: u32,
        range: TokenRangeMsg,
        encoding: u8,
    },
    CacheEvict {
        bytes_target: u64,
        policy: ShedPolicyMsg,
    },
    LaneSequence { lane_tag: u64 },
    CreateSequence,
    FreeSequence { sequence: u64 },
    PublishSequence { sequence: u64, span: TokenRefMsg },
    AdoptSequence { sequence: u64, span: TokenRefMsg },
    MediaProbe { image_path: String },
    MediaEncode { image_path: String },
    MediaRelease { media_handle: u64 },
    MediaBind { lane_tag: u64, media_handle: u64, token_offset: u32 },
    CreateGrammar { json_schema: String },
    CreateGrammarStructural { tag_json: String },
    FreeGrammar { grammar_handle: u32 },
    CreateLogitBias { tokens: Vec<i32>, values: Vec<u32> },
    FreeLogitBias { handle: u32 },
    ForwardStage {
        tokens: Vec<u32>,
        start_layer: u32,
        end_layer: u32,
        hidden_in: Option<Vec<u8>>,
    },
    Embed { tokens: Vec<u32> },
    Transcribe {
        audio_path: String,
        language: Option<String>,
        translate: bool,
        timestamps: bool,
        prompt: Option<String>,
        stream: bool,
    },
    LoraLoad { adapter_path: String },
    LoraUnload,
    LoraId,
    Capabilities,
    SnapshotBoundary { sequence: u64, space_id: u32, cap: u64 },
    CacheEvictEntries { keys: Vec<CacheKeyMsg> },
    /// Drops the prefix-cache entry published under exactly these tokens,
    /// unless a seed holds it.
    Unpublish { span: TokenRefMsg },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKeyMsg {
    pub compat_key: [u8; 32],
    pub provenance_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateSyncResMsg {
    Ok(StateSyncOkMsg),
    Err { status: i32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateSyncOkMsg {
    Fork {
        child_sequence: u64,
    },
    Trim,
    ExportSize {
        required_bytes: u64,
        sizing_gen: u64,
    },
    CacheEvict {
        bytes_freed: u64,
    },
    LaneSequence {
        sequence: u64,
    },
    CreateSequence {
        sequence: u64,
    },
    FreeSequence,
    PublishSequence,
    AdoptSequence { seed_handle: u64 },
    MediaProbe { info: MediaInfoMsg },
    MediaEncode { media_handle: u64, info: MediaInfoMsg },
    MediaRelease,
    MediaBind,
    CreateGrammar { grammar_handle: u32 },
    FreeGrammar,
    CreateLogitBias { handle: u32 },
    FreeLogitBias,
    ForwardStage {
        hidden: Option<Vec<u8>>,
        token: Option<u32>,
    },
    Transcribe {
        text: String,
        language: String,
        duration_ms: i32,
        segments: Vec<TranscriptSegmentMsg>,
    },
    Embed { embedding: Vec<u8> },
    LoraId { id: Option<String> },
    Capabilities { json: String },
    SnapshotBoundary { boundary: u64 },
    CacheEvictEntries,
    Unpublish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufAllocMsg {
    pub buf_id: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufAttachMsg {
    pub buf_id: u64,
    pub len: u64,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufReleaseMsg {
    pub buf_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PingMsg {
    pub queue_depth: u32,
    pub mem: MemCountersMsg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PongMsg {
    pub queue_depth: u32,
    pub mem: MemCountersMsg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DrainReqMsg;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DrainedMsg;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryMsg {
    pub spans: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TickWindowError {
    #[error("bundle {bundle_id} already has an outstanding TickSubmit (plan_seq {outstanding_plan_seq}); exactly one is allowed per bundle")]
    AlreadyOutstanding {
        bundle_id: u64,
        outstanding_plan_seq: u64,
    },

    #[error("TickResult for bundle {bundle_id} plan_seq {got} does not match the outstanding submit's plan_seq {expected}")]
    PlanSeqMismatch {
        bundle_id: u64,
        expected: u64,
        got: u64,
    },

    #[error("TickResult for bundle {bundle_id} with no outstanding TickSubmit")]
    NotOutstanding { bundle_id: u64 },
}

#[derive(Debug, Default)]
pub struct TickWindow {
    outstanding: HashMap<u64, u64>,
}

impl TickWindow {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn submit(&mut self, bundle_id: u64, plan_seq: u64) -> Result<(), TickWindowError> {
        if let Some(&outstanding_plan_seq) = self.outstanding.get(&bundle_id) {
            return Err(TickWindowError::AlreadyOutstanding {
                bundle_id,
                outstanding_plan_seq,
            });
        }
        self.outstanding.insert(bundle_id, plan_seq);
        Ok(())
    }

    pub fn complete(&mut self, bundle_id: u64, plan_seq: u64) -> Result<(), TickWindowError> {
        match self.outstanding.get(&bundle_id) {
            None => Err(TickWindowError::NotOutstanding { bundle_id }),
            Some(&expected) if expected != plan_seq => Err(TickWindowError::PlanSeqMismatch {
                bundle_id,
                expected,
                got: plan_seq,
            }),
            Some(_) => {
                self.outstanding.remove(&bundle_id);
                Ok(())
            }
        }
    }

    pub fn is_outstanding(&self, bundle_id: u64) -> bool {
        self.outstanding.contains_key(&bundle_id)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn seed_grant_ttl_ticks_is_backward_compatible() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        enum OldGrant {
            Granted { seed_handle: u64, ttl_ms: u32 },
            Refused { status: i32 },
        }
        let old = postcard::to_stdvec(&OldGrant::Granted {
            seed_handle: 7,
            ttl_ms: 1000,
        })
        .unwrap();
        assert_eq!(
            SeedGrantMsg::decode(&old).unwrap(),
            SeedGrantMsg::Granted {
                seed_handle: 7,
                ttl_ms: 1000,
                ttl_ticks: None
            }
        );
        let refused = postcard::to_stdvec(&OldGrant::Refused { status: -130 }).unwrap();
        assert_eq!(
            SeedGrantMsg::decode(&refused).unwrap(),
            SeedGrantMsg::Refused { status: -130 }
        );
        let new = postcard::to_stdvec(&SeedGrantMsg::Granted {
            seed_handle: 9,
            ttl_ms: 1000,
            ttl_ticks: Some(64),
        })
        .unwrap();
        assert_eq!(
            SeedGrantMsg::decode(&new).unwrap(),
            SeedGrantMsg::Granted {
                seed_handle: 9,
                ttl_ms: 1000,
                ttl_ticks: Some(64)
            }
        );
        assert_eq!(
            postcard::from_bytes::<OldGrant>(&new).unwrap(),
            OldGrant::Granted {
                seed_handle: 9,
                ttl_ms: 1000
            }
        );
        assert!(SeedGrantMsg::decode(&[0xff, 0xff]).is_err());
    }

    use super::*;

    #[test]
    fn tick_window_double_submit_rejected() {
        let mut w = TickWindow::new();
        w.submit(1, 100).unwrap();
        let err = w.submit(1, 101).unwrap_err();
        assert_eq!(
            err,
            TickWindowError::AlreadyOutstanding {
                bundle_id: 1,
                outstanding_plan_seq: 100
            }
        );
    }

    #[test]
    fn tick_window_submit_complete_alternation() {
        let mut w = TickWindow::new();
        w.submit(1, 100).unwrap();
        assert!(w.is_outstanding(1));
        w.complete(1, 100).unwrap();
        assert!(!w.is_outstanding(1));
        w.submit(1, 101).unwrap();
        w.complete(1, 101).unwrap();
    }

    #[test]
    fn tick_window_complete_without_submit() {
        let mut w = TickWindow::new();
        let err = w.complete(7, 1).unwrap_err();
        assert_eq!(err, TickWindowError::NotOutstanding { bundle_id: 7 });
    }

    #[test]
    fn tick_window_plan_seq_mismatch() {
        let mut w = TickWindow::new();
        w.submit(1, 100).unwrap();
        let err = w.complete(1, 999).unwrap_err();
        assert_eq!(
            err,
            TickWindowError::PlanSeqMismatch {
                bundle_id: 1,
                expected: 100,
                got: 999
            }
        );
    }

    #[test]
    fn independent_bundles_do_not_interfere() {
        let mut w = TickWindow::new();
        w.submit(1, 10).unwrap();
        w.submit(2, 20).unwrap();
        w.complete(1, 10).unwrap();
        w.complete(2, 20).unwrap();
    }

    #[test]
    fn all_msg_types_are_known() {
        for &t in msg_type::ALL {
            assert!(is_known_type(t));
        }
        assert!(!is_known_type(0x0002 + 0x1000));
    }
}
