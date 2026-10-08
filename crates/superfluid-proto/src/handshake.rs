//! Handshake types shared in shape (not in payload) by Link W and Link F: `Hello` is common; the
//! ack is link-specific because each catalog versions and describes itself independently.

use serde::{Deserialize, Serialize};

use crate::linkf::LinkFLimits;
use crate::linkw::LinkWLimits;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LinkRole {
    NodeAgent,
    Worker,
    HeadDaemon,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    pub proto_versions: (u16, u16),
    pub link_role: LinkRole,
    pub auth: Vec<u8>,
}

pub fn negotiate(offered: (u16, u16), supported: (u16, u16)) -> Option<u16> {
    let lo = offered.0.max(supported.0);
    let hi = offered.1.min(supported.1);
    if lo <= hi {
        Some(hi)
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSpaceDescMsg {
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
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrategySummary {
    pub strategy_id: String,
    pub strategy_slot: u32,
    pub impl_version: String,
    pub claimed_exactness: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelloAckW {
    pub chosen_version: u16,
    pub engine_bundle_hash: [u8; 32],
    pub capabilities: String,
    pub state_space_descriptors: Vec<StateSpaceDescMsg>,
    pub registered_strategies: Vec<StrategySummary>,
    pub limits: LinkWLimits,
    pub kv_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacitySummary {
    pub gpu_memory_bytes: u64,
    pub gpu_memory_free_bytes: u64,
    pub host_memory_bytes: u64,
    pub active_sessions: u32,
    pub max_sessions: u32,
    pub max_context_tokens: u64,

    pub kv_blocks_used: u64,
    pub kv_blocks_total: u64,
    pub lanes_active: u32,
    pub max_lanes: u32,
    pub queue_depth: u32,
    pub decode_tokens_per_s: u32,
    pub prefill_tokens_per_s: u32,
    pub sampled_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelloAckF {
    pub chosen_version: u16,
    pub node_identity: String,
    pub capacity: CapacitySummary,
    pub loadable_models: Vec<String>,
    pub feature_flags: u64,
    pub limits: LinkFLimits,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negotiate_overlap_picks_highest_common() {
        assert_eq!(negotiate((1, 5), (3, 7)), Some(5));
        assert_eq!(negotiate((1, 3), (1, 3)), Some(3));
        assert_eq!(negotiate((1, 10), (10, 10)), Some(10));
    }

    #[test]
    fn negotiate_no_overlap_is_none() {
        assert_eq!(negotiate((1, 2), (3, 4)), None);
        assert_eq!(negotiate((5, 5), (1, 4)), None);
    }

    #[test]
    fn negotiate_touching_ranges() {
        assert_eq!(negotiate((1, 3), (3, 5)), Some(3));
    }
}
