use serde::{Deserialize, Serialize};

use crate::handshake::CapacitySummary;
use crate::linkw::{MemCountersMsg, ShedPolicyMsg};

pub mod msg_type {
    pub const HELLO: u16 = 0x0001;
    pub const HELLO_ACK_F: u16 = 0x0002;
    pub const RECONCILE: u16 = 0x0003;
    pub const RECONCILE_ACK: u16 = 0x0004;

    pub const SESSION_ASSIGN: u16 = 0x0005;
    pub const SESSION_REVOKE: u16 = 0x0006;
    pub const POLICY_UPDATE: u16 = 0x0007;
    pub const LEASE_RENEW: u16 = 0x0008;
    pub const LEASE_GRANT: u16 = 0x0009;

    pub const APPEND_CONTENT: u16 = 0x000A;
    pub const GENERATE_REQ: u16 = 0x000B;
    pub const PARK_REQ: u16 = 0x000C;
    pub const RESUME_REQ: u16 = 0x000D;
    pub const GENERATE_CANCEL: u16 = 0x0018;

    pub const TOKEN_EVENTS: u16 = 0x000E;
    pub const WATERMARK_ACK: u16 = 0x000F;

    pub const XFER_BEGIN: u16 = 0x0010;
    pub const XFER_CHUNK: u16 = 0x0011;
    pub const XFER_END: u16 = 0x0012;
    pub const XFER_ABORT: u16 = 0x0013;

    pub const STATE_SHIP: u16 = 0x0014;
    pub const STATE_FETCH: u16 = 0x0015;
    pub const BLOB_FETCH: u16 = 0x0016;

    pub const NODE_STATUS: u16 = 0x0017;
    pub const STAGE_FORWARD_REQ: u16 = 0x0019;
    pub const STAGE_FORWARD_RESP: u16 = 0x001A;
    /// What a generation asks beyond its sampling, just before its GENERATE_REQ (version 3).
    pub const GENERATE_OPTS: u16 = 0x001B;
    /// A Link W state operation run on the node's engine, and its answer (version 3).
    pub const STATE_SYNC_REQ: u16 = 0x001C;
    pub const STATE_SYNC_RESP: u16 = 0x001D;
    /// The node's tokenizer, by digest, just after the handshake (version 3).
    pub const TOKENIZER: u16 = 0x001E;
    /// The node's tokenizer whole, for a head that has no copy of the model (version 3).
    pub const TOKENIZER_EXPORT_REQ: u16 = 0x001F;
    pub const TOKENIZER_EXPORT: u16 = 0x0020;
    /// Text the node's tokenizer encodes for the head (version 3).
    pub const TOKENIZE_REQ: u16 = 0x0021;
    pub const TOKENIZE_RESP: u16 = 0x0022;
    /// A streamed transcription's segment, told as the node has it, ahead of its answer
    /// (version 3).
    pub const STATE_SYNC_SEGMENT: u16 = 0x0023;

    pub const TELEMETRY: u16 = 0x8001;
    /// An idle head's sign of life; droppable, so a node that predates it skips it.
    pub const KEEPALIVE: u16 = 0x8002;
    /// The head asks a node that dialed it for one connection more, kept for short calls
    /// while generations hold the rest; droppable, so a node that predates it dials no more.
    pub const CONTROL_ASK: u16 = 0x8003;
    /// The head removed the node from its fleet and refuses it until the head restarts: sent
    /// on each of the node's links before the head closes it, and on a link the node opens
    /// after. Droppable, so a node that predates it sees only the link close, and dials again.
    pub const REMOVED: u16 = 0x8004;

    pub const ALL: &[u16] = &[
        HELLO,
        HELLO_ACK_F,
        RECONCILE,
        RECONCILE_ACK,
        SESSION_ASSIGN,
        SESSION_REVOKE,
        POLICY_UPDATE,
        LEASE_RENEW,
        LEASE_GRANT,
        APPEND_CONTENT,
        GENERATE_REQ,
        PARK_REQ,
        RESUME_REQ,
        TOKEN_EVENTS,
        WATERMARK_ACK,
        XFER_BEGIN,
        XFER_CHUNK,
        XFER_END,
        XFER_ABORT,
        STATE_SHIP,
        STATE_FETCH,
        BLOB_FETCH,
        NODE_STATUS,
        GENERATE_CANCEL,
        STAGE_FORWARD_REQ,
        STAGE_FORWARD_RESP,
        GENERATE_OPTS,
        STATE_SYNC_REQ,
        STATE_SYNC_RESP,
        TOKENIZER,
        TOKENIZER_EXPORT_REQ,
        TOKENIZER_EXPORT,
        TOKENIZE_REQ,
        TOKENIZE_RESP,
        STATE_SYNC_SEGMENT,
        TELEMETRY,
        KEEPALIVE,
        CONTROL_ASK,
        REMOVED,
    ];
}

pub fn is_known_type(msg_type: u16) -> bool {
    msg_type::ALL.contains(&msg_type)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkFLimits {
    pub max_frame: u32,
    pub transfer_window: u32,
    pub max_transfer: u64,
    pub transfer_budget: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    pub duration_ms: u64,
    pub renew_by_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationWatermark {
    pub generation_id: u64,
    pub watermark: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseStateMsg {
    pub lease: Lease,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionReconcileState {
    pub session_id: u64,
    pub epoch: u64,
    pub lease_state: LeaseStateMsg,
    pub command_watermark: u64,
    pub generation_watermarks: Vec<GenerationWatermark>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconcileMsg {
    pub sessions: Vec<SessionReconcileState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryRange {
    pub generation_id: u64,
    pub from_seq: u64,
    pub to_seq: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionReconcileAck {
    pub session_id: u64,
    pub held: bool,
    pub retry_ranges: Vec<RetryRange>,
    pub lease_view: LeaseStateMsg,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconcileAckMsg {
    pub sessions: Vec<SessionReconcileAck>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionAssignMsg {
    pub command_id: u64,
    pub session_id: u64,
    pub epoch: u64,
    pub lease: Lease,
    pub codec: String,
    pub params: Vec<u8>,
    pub qos_class: u8,
    pub policy_ref: u64,
    /// The model the session is for, as the node told it in `loadable_models`; empty for the
    /// node's first. A node serving several runs the session on that model's engine. Last, so a
    /// node from before it reads the message and leaves it; [`SessionAssignMsg::decode`] reads
    /// one from a head before it (version 4).
    pub model: String,
}

/// A [`SessionAssignMsg`] from a head released before [`SessionAssignMsg::model`].
#[derive(Deserialize)]
struct SessionAssignBefore {
    command_id: u64,
    session_id: u64,
    epoch: u64,
    lease: Lease,
    codec: String,
    params: Vec<u8>,
    qos_class: u8,
    policy_ref: u64,
}

impl SessionAssignMsg {
    /// An assignment from a head of any release; one from before `model` names none.
    pub fn decode(bytes: &[u8]) -> Result<SessionAssignMsg, postcard::Error> {
        or_before(bytes, |m: SessionAssignBefore| SessionAssignMsg {
            command_id: m.command_id,
            session_id: m.session_id,
            epoch: m.epoch,
            lease: m.lease,
            codec: m.codec,
            params: m.params,
            qos_class: m.qos_class,
            policy_ref: m.policy_ref,
            model: String::new(),
        })
    }
}

/// `bytes` as a `T`, or, when they are too short for one, as the `B` a peer released before
/// `T`'s last field sends, made a `T` by `from`.
fn or_before<T: serde::de::DeserializeOwned, B: serde::de::DeserializeOwned>(
    bytes: &[u8],
    from: impl FnOnce(B) -> T,
) -> Result<T, postcard::Error> {
    match postcard::from_bytes::<T>(bytes) {
        Ok(m) => Ok(m),
        Err(e) => postcard::from_bytes::<B>(bytes).map(from).map_err(|_| e),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionRevokeMsg {
    pub command_id: u64,
    pub session_id: u64,
    pub epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyUpdateMsg {
    pub policy_version: u64,
    pub envelope: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseRenewMsg {
    pub session_id: u64,
    pub epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseGrantMsg {
    pub session_id: u64,
    pub epoch: u64,
    pub duration_ms: u64,
    pub renew_by_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenSpan {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppendContentMsg {
    pub command_id: u64,
    pub session_id: u64,
    pub epoch: u64,
    pub blocks: Vec<u8>,
    pub spans: Vec<TokenSpan>,
    pub media_refs: Vec<[u8; 32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationBudgetsMsg {
    pub max_new_tokens: u64,
    pub max_wall_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateReqMsg {
    pub command_id: u64,
    pub session_id: u64,
    pub epoch: u64,
    pub generation_id: u64,
    pub budgets: GenerationBudgetsMsg,
    pub policy_ref: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateCancelMsg {
    pub session_id: u64,
    pub generation_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageForwardReqMsg {
    pub request_id: u64,
    pub tokens: Vec<u32>,
    pub start_layer: u32,
    pub end_layer: u32,
    pub hidden_in: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageForwardRespMsg {
    pub request_id: u64,
    pub hidden: Option<Vec<u8>>,
    pub token: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParkReqMsg {
    pub command_id: u64,
    pub session_id: u64,
    pub epoch: u64,
    pub generation_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResumeReqMsg {
    pub command_id: u64,
    pub session_id: u64,
    pub epoch: u64,
    pub generation_id: u64,
    pub budgets: GenerationBudgetsMsg,
    pub policy_ref: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentKind {
    TextDelta,
    ReasoningDelta,
    ToolCallDelta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageMsg {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenEventPayload {
    Tokens(Vec<u32>),
    Segment { kind: SegmentKind, data: Vec<u8> },
    Usage(UsageMsg),
    Finish { reason: u32 },
    /// The log-probabilities of tokens already sent, for a generation that asked (version 3).
    Logprobs(Vec<TokenLogprobMsg>),
}

/// A token's log-probability and its most likely alternatives, as f32 bits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenLogprobMsg {
    pub token: u32,
    pub logprob_bits: u32,
    pub top: Vec<(u32, u32)>,
}

/// A grammar a generation is held to, as the node compiles it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrammarSpecMsg {
    JsonSchema(String),
    Structural(String),
}

/// What a generation asks beyond its sampling parameters. Floats are f32 bits; the defaults
/// (no penalty, EOS ends it, no logprobs, no grammar or bias) are what a generation with no
/// options gets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateOptsMsg {
    pub session_id: u64,
    pub generation_id: u64,
    pub presence_penalty_bits: u32,
    pub frequency_penalty_bits: u32,
    pub repeat_penalty_bits: u32,
    pub ignore_eos: bool,
    pub top_logprobs: Option<u8>,
    pub grammar: Option<GrammarSpecMsg>,
    pub logit_bias: Option<(Vec<i32>, Vec<u32>)>,
    /// The images and audio in the context: where each starts, the blob it is (sent before, by
    /// its SHA-256 in hex), and the tokens it takes.
    pub media: Vec<(u64, String, u32)>,
}

/// What a node's model tokenizes with, by digest (0: not known), so a head refuses a node whose
/// tokens would mean other text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMsg {
    /// The first model's.
    pub digest: u64,
    /// Each model the node serves, in `loadable_models` order, with its own tokenizer and window:
    /// told to a head on version 4, which names the model of each session and request; empty to
    /// one before it, which is offered the first model only. Last, so a head from before it reads
    /// the message and leaves it; [`TokenizerMsg::decode`] reads one from a node before it.
    pub models: Vec<NodeModelMsg>,
}

/// One model a node serves, as [`TokenizerMsg::models`] tells it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeModelMsg {
    pub name: String,
    /// Its tokenizer's digest, 0 when not known.
    pub tokenizer: u64,
    /// The longest context it holds, 0 when not known.
    pub max_context_tokens: u64,
}

impl TokenizerMsg {
    /// The digest from a node of any release; one from before `models` tells none.
    pub fn decode(bytes: &[u8]) -> Result<TokenizerMsg, postcard::Error> {
        #[derive(Deserialize)]
        struct Before {
            digest: u64,
        }
        or_before(bytes, |m: Before| TokenizerMsg { digest: m.digest, models: Vec::new() })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerExportReqMsg {
    pub request_id: u64,
    /// The model whose tokenizer is asked for; empty for the node's first (version 4, last).
    pub model: String,
}

impl TokenizerExportReqMsg {
    /// A request from a head of any release; one from before `model` names none.
    pub fn decode(bytes: &[u8]) -> Result<TokenizerExportReqMsg, postcard::Error> {
        #[derive(Deserialize)]
        struct Before {
            request_id: u64,
        }
        or_before(bytes, |m: Before| TokenizerExportReqMsg { request_id: m.request_id, model: String::new() })
    }
}

/// What a tokenizer says without being asked to encode: every token's bytes, its specials and
/// ends, and its chat templates. A head with this and a node to encode has the tokenizer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerExportMsg {
    pub request_id: u64,
    pub digest: u64,
    pub vocab: Vec<Vec<u8>>,
    pub specials: Vec<(String, u32)>,
    pub bos: Option<u32>,
    pub eos: u32,
    pub chat_template: String,
    /// The templates asked for by name (`tool_use`), each the node has.
    pub named_templates: Vec<(String, String)>,
    /// Whether the tokenizer encodes plainly (no specials parsed) and by pieces.
    pub plain: bool,
    pub pieces: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizeMsg {
    Encode(String),
    Plain(String),
    Pieces(Vec<(String, bool)>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizeReqMsg {
    pub request_id: u64,
    pub text: TokenizeMsg,
    /// The model whose tokenizer encodes it; empty for the node's first (version 4, last).
    pub model: String,
}

impl TokenizeReqMsg {
    /// A request from a head of any release; one from before `model` names none.
    pub fn decode(bytes: &[u8]) -> Result<TokenizeReqMsg, postcard::Error> {
        #[derive(Deserialize)]
        struct Before {
            request_id: u64,
            text: TokenizeMsg,
        }
        or_before(bytes, |m: Before| TokenizeReqMsg { request_id: m.request_id, text: m.text, model: String::new() })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizeRespMsg {
    pub request_id: u64,
    pub tokens: Option<Vec<u32>>,
}

/// A media path the head sent as a blob: the node reads it from its own media pool.
pub const BLOB_PREFIX: &str = "fleet-blob:";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSyncReqF {
    pub request_id: u64,
    pub req: crate::linkw::StateSyncReqMsg,
    /// The model whose engine runs it; empty for the node's first (version 4, last).
    pub model: String,
}

impl StateSyncReqF {
    /// A request from a head of any release; one from before `model` names none.
    pub fn decode(bytes: &[u8]) -> Result<StateSyncReqF, postcard::Error> {
        #[derive(Deserialize)]
        struct Before {
            request_id: u64,
            req: crate::linkw::StateSyncReqMsg,
        }
        or_before(bytes, |m: Before| StateSyncReqF { request_id: m.request_id, req: m.req, model: String::new() })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSyncRespF {
    pub request_id: u64,
    pub result: Result<crate::linkw::StateSyncOkMsg, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSyncSegmentF {
    pub request_id: u64,
    pub segment: crate::linkw::TranscribeSegmentMsg,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenEvent {
    pub session_id: u64,
    pub epoch: u64,
    pub generation_id: u64,
    pub event_seq: u64,
    pub payload: TokenEventPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenEventsMsg {
    pub events: Vec<TokenEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatermarkAckMsg {
    pub session_id: u64,
    pub epoch: u64,
    pub generation_id: u64,
    pub watermark: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct XferBeginMsg {
    pub transfer_id: u64,
    pub total_len: u64,
    pub checksum: u64,
    pub content_type: u32,
    pub content_hash: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XferChunkMsg {
    pub transfer_id: u64,
    pub offset: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct XferEndMsg {
    pub transfer_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct XferAbortMsg {
    pub transfer_id: u64,
    pub reason: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateShipMsg {
    pub envelope: Vec<u8>,
    pub transfer_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateFetchMsg {
    pub envelope: Vec<u8>,
    pub transfer_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobFetchMsg {
    pub content_hash: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeStatusMsg {
    pub capacity: CapacitySummary,
    pub loadable_models: Vec<String>,
    pub mem: MemCountersMsg,
    /// The node's scheduler counters; `None` from a node released before them. Last, so a head
    /// from before it reads the status and leaves it; [`NodeStatusMsg::decode`] reads a status
    /// from before it.
    pub stats: Option<NodeStatsMsg>,
}

/// What a node's scheduler has counted since its counters last started from 0, as its own
/// `/metrics` counts it, so a head that only routes can report what its nodes measure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeStatsMsg {
    /// Which start of the node's counters these are: another value means they started again
    /// from 0. A node with several connections reports on each, so a status read late can be
    /// older than one read before it; within one run a lower count is such a status, not a
    /// restart.
    pub run: u64,
    pub prefill_tokens: u64,
    pub warm_prefix_tokens: u64,
    pub cold_admissions: u64,
    pub unservable_cold_admissions: u64,
    pub unservable_park_refusals: u64,
    pub preemptions: u64,
    pub admits_deferred: u64,
    pub starvation_grants: u64,
    pub os_pressure_events: u64,
    pub pressure_evictions: u64,
    pub pressure_bytes_evicted: u64,
    pub pins_yielded: u64,
    pub pins_expired: u64,
    pub worker_respawns: u64,
    pub spec_proposed: u64,
    pub spec_accepted: u64,
    pub parks_lossless: u64,
    pub parks_lossy: u64,
    pub resumes: u64,
}

/// A status from a node released before [`NodeStatusMsg::stats`].
#[derive(Deserialize)]
struct NodeStatusBefore {
    capacity: CapacitySummary,
    loadable_models: Vec<String>,
    mem: MemCountersMsg,
}

impl NodeStatusMsg {
    /// A status from a node of any release; one from before `stats` has none.
    pub fn decode(bytes: &[u8]) -> Result<NodeStatusMsg, postcard::Error> {
        match postcard::from_bytes::<NodeStatusMsg>(bytes) {
            Ok(status) => Ok(status),
            Err(e) => match postcard::from_bytes::<NodeStatusBefore>(bytes) {
                Ok(s) => Ok(NodeStatusMsg { capacity: s.capacity, loadable_models: s.loadable_models, mem: s.mem, stats: None }),
                Err(_) => Err(e),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryMsg {
    pub spans: Vec<u8>,
}

pub type SharedShedPolicyMsg = ShedPolicyMsg;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_msg_types_are_known() {
        for &t in msg_type::ALL {
            assert!(is_known_type(t));
        }
        assert!(!is_known_type(0x00FF));
    }

    #[test]
    fn a_node_that_predates_removal_skips_the_head_saying_so() {
        use crate::envelope::{encode_frame, FrameClass, FrameDecoder};
        assert_ne!(msg_type::REMOVED & 0x8000, 0, "sent droppable, as every 0x8000 type is");
        let mut said = encode_frame(3, msg_type::REMOVED, 7, 0, FrameClass::Droppable, &[]).unwrap();
        said.extend(encode_frame(3, msg_type::KEEPALIVE, 8, 0, FrameClass::Droppable, &[]).unwrap());
        let mut old = FrameDecoder::new(|t: u16| t != msg_type::REMOVED && is_known_type(t));
        old.feed(&said);
        assert_eq!(old.decode_next().unwrap().map(|f| f.msg_type), Some(msg_type::KEEPALIVE), "skipped, and what follows read");
        let mut new = FrameDecoder::new(is_known_type);
        new.feed(&said);
        assert_eq!(new.decode_next().unwrap().map(|f| f.msg_type), Some(msg_type::REMOVED));
    }

    fn status(stats: Option<NodeStatsMsg>) -> NodeStatusMsg {
        NodeStatusMsg {
            capacity: CapacitySummary { max_lanes: 4, kv_blocks_used: 3, kv_blocks_total: 64, ..Default::default() },
            loadable_models: vec!["org/model".into()],
            mem: MemCountersMsg {
                allocated_bytes: 0,
                host_retained_bytes: 0,
                pool_blocks_total: 64,
                pool_blocks_used: 3,
                pool_bytes_evictable: 0,
                tentative_bytes: 0,
            },
            stats,
        }
    }

    /// A status as a node released before [`NodeStatusMsg::stats`] sends it, and as a head
    /// released before it reads one.
    #[derive(Serialize, Deserialize)]
    struct StatusFrom {
        capacity: CapacitySummary,
        loadable_models: Vec<String>,
        mem: MemCountersMsg,
    }

    #[test]
    fn a_head_from_before_node_stats_reads_a_status_that_carries_them() {
        let stats = NodeStatsMsg { run: 9, spec_proposed: 40, spec_accepted: 31, warm_prefix_tokens: 512, ..Default::default() };
        let new = postcard::to_stdvec(&status(Some(stats))).unwrap();
        let old: StatusFrom = postcard::from_bytes(&new).unwrap();
        assert_eq!((old.capacity.kv_blocks_used, old.loadable_models.as_slice()), (3, &["org/model".to_string()][..]));
        assert_eq!(NodeStatusMsg::decode(&new).unwrap(), status(Some(stats)));
    }

    #[test]
    fn a_status_from_before_node_stats_reads_as_one_with_none() {
        let s = status(None);
        let old = postcard::to_stdvec(&StatusFrom { capacity: s.capacity, loadable_models: s.loadable_models.clone(), mem: s.mem }).unwrap();
        assert_eq!(NodeStatusMsg::decode(&old).unwrap(), s);
        assert!(NodeStatusMsg::decode(&old[..old.len() - 2]).is_err(), "a cut status is still refused");
    }

    #[test]
    fn a_head_and_a_node_from_before_the_model_named_read_each_other() {
        #[derive(Serialize)]
        struct OldAssign {
            command_id: u64,
            session_id: u64,
            epoch: u64,
            lease: Lease,
            codec: String,
            params: Vec<u8>,
            qos_class: u8,
            policy_ref: u64,
        }
        let lease = Lease { duration_ms: 30_000, renew_by_ms: 20_000 };
        let assign = SessionAssignMsg {
            command_id: 7,
            session_id: 9,
            epoch: 3,
            lease,
            codec: "raw".into(),
            params: vec![1, 2],
            qos_class: 1,
            policy_ref: 0,
            model: "org/b".into(),
        };
        // A node from before reads a new head's assignment, the model left unread.
        let theirs: SessionAssignBefore = postcard::from_bytes(&postcard::to_allocvec(&assign).unwrap()).unwrap();
        assert_eq!((theirs.session_id, theirs.epoch, theirs.policy_ref), (9, 3, 0));
        assert_eq!(SessionAssignMsg::decode(&postcard::to_allocvec(&assign).unwrap()).unwrap(), assign);
        // A head from before names no model, which is the node's first.
        let old = OldAssign { command_id: 7, session_id: 9, epoch: 3, lease, codec: "raw".into(), params: vec![1, 2], qos_class: 1, policy_ref: 0 };
        let read = SessionAssignMsg::decode(&postcard::to_allocvec(&old).unwrap()).unwrap();
        assert_eq!(read, SessionAssignMsg { model: String::new(), ..assign });

        #[derive(Serialize)]
        struct OldSync {
            request_id: u64,
            req: crate::linkw::StateSyncReqMsg,
        }
        let old = OldSync { request_id: 4, req: crate::linkw::StateSyncReqMsg::Capabilities };
        let read = StateSyncReqF::decode(&postcard::to_allocvec(&old).unwrap()).unwrap();
        assert_eq!((read.request_id, read.model.as_str()), (4, ""));
        let named = StateSyncReqF { request_id: 4, req: crate::linkw::StateSyncReqMsg::Capabilities, model: "org/b".into() };
        assert_eq!(StateSyncReqF::decode(&postcard::to_allocvec(&named).unwrap()).unwrap(), named);

        let read = TokenizerExportReqMsg::decode(&postcard::to_allocvec(&4u64).unwrap()).unwrap();
        assert_eq!(read, TokenizerExportReqMsg { request_id: 4, model: String::new() });
        let text = TokenizeMsg::Encode("hi".into());
        let read = TokenizeReqMsg::decode(&postcard::to_allocvec(&(4u64, text.clone())).unwrap()).unwrap();
        assert_eq!(read, TokenizeReqMsg { request_id: 4, text, model: String::new() });

        // A head from before reads the first model's digest from a node that tells each model's.
        let told = TokenizerMsg {
            digest: 11,
            models: vec![NodeModelMsg { name: "org/b".into(), tokenizer: 12, max_context_tokens: 4096 }],
        };
        let bytes = postcard::to_allocvec(&told).unwrap();
        assert_eq!(postcard::from_bytes::<u64>(&bytes).unwrap(), 11);
        assert_eq!(TokenizerMsg::decode(&bytes).unwrap(), told);
        assert_eq!(TokenizerMsg::decode(&postcard::to_allocvec(&11u64).unwrap()).unwrap(), TokenizerMsg { digest: 11, models: Vec::new() });
    }

    #[test]
    fn linkw_and_linkf_catalogs_may_overlap_numerically() {
        assert_eq!(crate::linkw::msg_type::HELLO, msg_type::HELLO);
    }
}
