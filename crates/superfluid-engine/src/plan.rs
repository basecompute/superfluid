//! Plan decoding + structural validation.

use std::collections::HashSet;

use superfluid_abi::{
    array::{read_array, read_versioned_struct},
    sampling, LaneAdmit, LaneCommit, LaneDecode, LanePrefill,
    LaneRetire, TickPlan, RecordArena, Status,
};

#[derive(Debug)]
pub struct DecodedPlan {
    pub plan_seq: u64,
    pub flags: u32,
    pub prefill_token_budget: u32,
    pub max_decode_lanes: u32,
    pub admits: Vec<LaneAdmit>,
    pub commits: Vec<LaneCommit>,
    pub prefills: Vec<LanePrefill>,
    pub decodes: Vec<LaneDecode>,
    pub retires: Vec<LaneRetire>,
    pub victim_lanes: Vec<u64>,
    pub evictable_cache_classes: u64,
    pub protected_quota_bytes: u64,
    pub max_evict_bytes: u64,
    pub on_partial_emit: Option<
        unsafe extern "C" fn(user: *mut core::ffi::c_void, lane_tag: u64, tokens: *const u32, n_tokens: u32),
    >,
    pub partial_emit_user: *mut core::ffi::c_void,
}

fn decode_records<T: superfluid_abi::AbiRecord>(
    arr: &superfluid_abi::Array,
    arena: &RecordArena,
) -> Result<Vec<T>, Status> {
    let bounds = arena.bounds_of(arr).ok_or(Status::RejectBounds)?;
    // SAFETY: bounds_of proved the array lies inside one of the arena's
    // live segments, which outlive this call; read_array enforces the
    // remaining normative rules.
    let iter = unsafe { read_array::<T>(arr, &bounds) }.map_err(|e| e.status())?;
    Ok(iter.collect())
}

fn unique_tags(tags: impl Iterator<Item = u64>) -> Result<HashSet<u64>, Status> {
    let mut set = HashSet::new();
    for t in tags {
        if !set.insert(t) {
            return Err(Status::RejectDuplicateLane);
        }
    }
    Ok(set)
}

impl DecodedPlan {
    pub fn decode(plan: &TickPlan, arena: &RecordArena) -> Result<DecodedPlan, Status> {
        // SAFETY: `plan` is a live reference; struct_size is its writer's
        // declared size, bounded below by the min-prefix check inside.
        let plan = unsafe { read_versioned_struct(plan, plan.struct_size.max(1)) }
            .map_err(|e| e.status())?;

        let admits: Vec<LaneAdmit> = decode_records(&plan.admits, arena)?;
        let commits: Vec<LaneCommit> = decode_records(&plan.commits, arena)?;
        let prefills: Vec<LanePrefill> = decode_records(&plan.prefills, arena)?;
        let decodes: Vec<LaneDecode> = decode_records(&plan.decodes, arena)?;
        let retires: Vec<LaneRetire> = decode_records(&plan.retires, arena)?;
        let victim_lanes: Vec<u64> = decode_records(&plan.shed_policy.victim_lanes, arena)?;

        let admit_tags = unique_tags(admits.iter().map(|r| r.lane_tag))?;
        let commit_tags = unique_tags(commits.iter().map(|r| r.lane_tag))?;
        let prefill_tags = unique_tags(prefills.iter().map(|r| r.lane_tag))?;
        let decode_tags = unique_tags(decodes.iter().map(|r| r.lane_tag))?;
        let retire_tags = unique_tags(retires.iter().map(|r| r.lane_tag))?;

        if admit_tags.intersection(&retire_tags).next().is_some() {
            return Err(Status::RejectIllegalCombination);
        }
        if admit_tags.intersection(&commit_tags).next().is_some() {
            return Err(Status::RejectIllegalCombination);
        }
        if commit_tags.intersection(&retire_tags).next().is_some() {
            return Err(Status::RejectIllegalCombination);
        }
        if decode_tags.intersection(&retire_tags).next().is_some() {
            return Err(Status::RejectIllegalCombination);
        }
        if prefill_tags.intersection(&retire_tags).next().is_some() {
            return Err(Status::RejectIllegalCombination);
        }

        for a in &admits {
            if a.sampling == sampling::HOST {
                if let Some(d) = decodes.iter().find(|d| d.lane_tag == a.lane_tag) {
                    if d.max_new_tokens != 1 {
                        return Err(Status::RejectHostRules);
                    }
                }
            }
        }

        if decodes.len() as u32 > plan.max_decode_lanes {
            return Err(Status::RejectBudget);
        }

        Ok(DecodedPlan {
            plan_seq: plan.plan_seq,
            flags: plan.flags,
            prefill_token_budget: plan.prefill_token_budget,
            max_decode_lanes: plan.max_decode_lanes,
            admits,
            commits,
            prefills,
            decodes,
            retires,
            victim_lanes,
            evictable_cache_classes: plan.shed_policy.evictable_cache_classes,
            protected_quota_bytes: plan.shed_policy.protected_quota_bytes,
            max_evict_bytes: plan.shed_policy.max_evict_bytes,
            on_partial_emit: plan.on_partial_emit,
            partial_emit_user: plan.partial_emit_user,
        })
    }
}
