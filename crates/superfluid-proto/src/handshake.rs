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
    /// The id each of `loadable_models` is pulled by (`org/model:tag`), in the same order, as
    /// far as the node knows them: shorter, or empty, where it does not. Last, so a head from
    /// before it reads the ack and leaves it; [`HelloAckF::decode`] reads an ack from before it.
    pub model_ids: Vec<String>,
}

/// [`HelloAckF::feature_flags`]: the node serves whatever model its head serves first, and
/// follows it to another, so a head does not take on the model such a node brings. A head from
/// before it reads no flags.
pub const FOLLOWS_HEAD: u64 = 1;

/// An ack from a node released before [`HelloAckF::model_ids`].
#[derive(Deserialize)]
struct HelloAckFBefore {
    chosen_version: u16,
    node_identity: String,
    capacity: CapacitySummary,
    loadable_models: Vec<String>,
    feature_flags: u64,
    limits: LinkFLimits,
}

impl HelloAckF {
    /// An ack from a node of any release; one from before `model_ids` has none.
    pub fn decode(bytes: &[u8]) -> Result<HelloAckF, postcard::Error> {
        match postcard::from_bytes::<HelloAckF>(bytes) {
            Ok(ack) => Ok(ack),
            Err(e) => match postcard::from_bytes::<HelloAckFBefore>(bytes) {
                Ok(a) => Ok(HelloAckF {
                    chosen_version: a.chosen_version,
                    node_identity: a.node_identity,
                    capacity: a.capacity,
                    loadable_models: a.loadable_models,
                    feature_flags: a.feature_flags,
                    limits: a.limits,
                    model_ids: Vec::new(),
                }),
                Err(_) => Err(e),
            },
        }
    }
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

    fn ack(model_ids: Vec<String>) -> HelloAckF {
        HelloAckF {
            chosen_version: 3,
            node_identity: "box-a".into(),
            capacity: CapacitySummary {
                gpu_memory_bytes: 0,
                gpu_memory_free_bytes: 0,
                host_memory_bytes: 0,
                active_sessions: 0,
                max_sessions: 1,
                max_context_tokens: 4096,
                kv_blocks_used: 0,
                kv_blocks_total: 0,
                lanes_active: 0,
                max_lanes: 8,
                queue_depth: 0,
                decode_tokens_per_s: 0,
                prefill_tokens_per_s: 0,
                sampled_at_ms: 0,
            },
            loadable_models: vec!["org/model".into()],
            feature_flags: 0,
            limits: LinkFLimits { max_frame: 1 << 20, transfer_window: 8, max_transfer: 1 << 26, transfer_budget: 1 << 28 },
            model_ids,
        }
    }

    #[derive(Serialize, Deserialize)]
    struct Before {
        chosen_version: u16,
        node_identity: String,
        capacity: CapacitySummary,
        loadable_models: Vec<String>,
        feature_flags: u64,
        limits: LinkFLimits,
    }

    #[test]
    fn a_head_from_before_model_ids_reads_an_ack_that_carries_them() {
        let new = postcard::to_stdvec(&ack(vec!["org/model:q4".into()])).unwrap();
        let read: Before = postcard::from_bytes(&new).unwrap();
        assert_eq!((read.node_identity.as_str(), read.loadable_models.as_slice()), ("box-a", &["org/model".to_string()][..]));
    }

    #[test]
    fn an_ack_from_before_model_ids_reads_as_one_with_none() {
        let a = ack(Vec::new());
        let before = Before {
            chosen_version: a.chosen_version,
            node_identity: a.node_identity.clone(),
            capacity: a.capacity,
            loadable_models: a.loadable_models.clone(),
            feature_flags: a.feature_flags,
            limits: a.limits,
        };
        let old = postcard::to_stdvec(&before).unwrap();
        assert_eq!(HelloAckF::decode(&old).unwrap(), a);
        let new = ack(vec!["org/model:q4".into()]);
        assert_eq!(HelloAckF::decode(&postcard::to_stdvec(&new).unwrap()).unwrap(), new);
        assert!(HelloAckF::decode(&old[..old.len() - 3]).is_err(), "a cut ack is still refused");
    }
}
