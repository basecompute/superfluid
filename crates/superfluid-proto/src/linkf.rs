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

    pub const TELEMETRY: u16 = 0x8001;
    /// An idle head's sign of life; droppable, so a node that predates it skips it.
    pub const KEEPALIVE: u16 = 0x8002;

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
        TELEMETRY,
        KEEPALIVE,
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
    fn linkw_and_linkf_catalogs_may_overlap_numerically() {
        assert_eq!(crate::linkw::msg_type::HELLO, msg_type::HELLO);
    }
}
