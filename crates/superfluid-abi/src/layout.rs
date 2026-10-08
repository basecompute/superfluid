//! Every size, offset, field width and constant the C header's layout probe
//! reports, as the Rust mirror has it.

use std::mem::{offset_of, size_of};

use crate::status::fault_code;
use crate::types::*;
use crate::types::TickMemCounters as MemCounters;
use crate::Status;

const fn pointee_size<F>(_: *const F) -> u64 {
    size_of::<F>() as u64
}

macro_rules! layout {
    ($( $t:ident { $($f:ident),* $(,)? } )* ; $( $c:ident = $v:expr, )*) => {
        &[
            $(
                (concat!("sizeof(BaseRT", stringify!($t), ")"), size_of::<$t>() as u64),
                $(
                    (concat!("offsetof(BaseRT", stringify!($t), ",", stringify!($f), ")"), offset_of!($t, $f) as u64),
                    (
                        concat!("fieldsize(BaseRT", stringify!($t), ",", stringify!($f), ")"),
                        {
                            let v = std::mem::MaybeUninit::<$t>::uninit();
                            // SAFETY: only the field's address is formed; nothing is read.
                            pointee_size(unsafe { &raw const (*v.as_ptr()).$f })
                        },
                    ),
                )*
            )*
            $( (concat!("const(", stringify!($c), ")"), $v), )*
        ]
    };
}

pub const RUST_LAYOUT: &[(&str, u64)] = layout!(
    Array { data, count, elem_size }
    TokenRef { ring_id, index, count, _pad0, generation }
    RingRef { ring_id, index, generation }
    Buf { ptr, len }
    TokenRange { start, end }
    SamplingParams {
        temperature, top_p, min_p, top_k,
        freq_penalty, presence_penalty, repeat_penalty, flags,
    }
    ShedPolicy {
        victim_lanes, evictable_cache_classes,
        protected_quota_bytes, max_evict_bytes,
    }
    LaneAdmit {
        lane_tag, prompt, seed_handle, sampling, logit_bias_handle, params,
        rng_counter_base, grammar_handle, strategy_slot,
        determinism_class, minimum_exactness, allow_approximate, want_logprobs,
        required_cert_id, host_sampler_identity, grammar_replay, decode_replay,
    }
    LaneCommit { lane_tag, token_id, _pad0, logits_nonce }
    LanePrefill { lane_tag, token_offset, token_count }
    LaneDecode { lane_tag, max_new_tokens, overshoot, _pad1 }
    LaneRetire { lane_tag, publish_to_cache, _pad0 }
    TickPlan {
        struct_size, plan_seq, flags, _pad0,
        prefill_token_budget, max_decode_lanes,
        admits, commits, prefills, decodes, retires, shed_policy,
        on_partial_emit, partial_emit_user,
    }
    AdmitResult {
        lane_tag, status, reject_code, cert_id, granted_class, _pad0,
    }
    SpecStats { proposed, accepted, cert_id_in_effect, _pad0 }
    LaneEmit { lane_tag, token_ref, n_tokens, finish, logits_row, spec }
    LaneLogprob { lane_tag, logprob, n_top, top_ids, top_logprobs }
    LaneFault { lane_tag, code, _pad0, detail }
    ShedEntry { lane_tag, kind, reason, bytes, tokens }
    ShedReport {
        prefill_chunks_dropped, evictions_performed, bytes_evicted, entries,
    }
    OpComplete { op, state, _pad0, error, bytes_moved }
    TickTimings { wall_ns, prefill_ns, decode_ns, graph_hits, graph_misses }
    MemCounters {
        allocated_bytes, host_retained_bytes, pool_blocks_total,
        pool_blocks_used, pool_bytes_evictable, tentative_bytes,
    }
    TickEvents {
        struct_size, plan_seq, tick_status, _pad0,
        admit_results, emits, faults, shed, op_completions, timings, mem, logprobs,
    }
    StateSpaceDesc {
        space_id, kind, version_tag, bytes_per_token, blob_bytes,
        page_size_tokens, fork_cost_class, fork_cost_bytes,
        snapshot_cadence, snapshot_interval_tokens, placement, flags, name,
    }
    Artifact { role, _pad0, content_hash, byte_size, load_path }
    TapSpec { layer, tensor, dtype, layout }
    CapabilityReq { kind_id, _pad0, params }
    StrategyRegistration {
        struct_size, strategy_id, impl_version, impl_hash, config_hash,
        artifacts, taps, capabilities, target_archs,
        kernel_caps_required, _pad0, est_state_bytes,
        claimed_exactness, _pad1, rng_contract_version,
    }
    ExactnessCert {
        cert_id, exactness, sampling_modes, grammar_allowed,
        logit_bias_allowed, param_domain, verification_shape_class,
        admissible_host_identities,
    }
    StrategyGrant {
        struct_size, strategy_slot, _pad0,
        certificates, state_spaces, reserved_bytes,
    }
    OpStatus { struct_size, state, _pad0, error, bytes_moved }
    MatchCandidate {
        prefix_len, provenance_digest, taint_bits, resident_tier, _pad0,
    }
    SpaceMatch { space_id, _pad0, candidates }
    MatchResult { struct_size, spaces }
    StateEnvelope {
        magic, envelope_version, space_kind, compat_identity, version_tag,
        provenance_digest, taint_bits, _pad0, range, encoding, _pad1,
        checksum_kind, payload_len, content_checksum,
    }
    CacheKey { compat_key, provenance_digest }
    RingDesc {
        struct_size, ring_id, kind, role, slots, slot_bytes, _pad0,
        base, len,
    }
    ;
    BASERT_STATUS_OK = Status::Ok.raw() as i64 as u64,
    BASERT_REJECT_BAD_STRIDE = Status::RejectBadStride.raw() as i64 as u64,
    BASERT_REJECT_BAD_ALIGNMENT = Status::RejectBadAlignment.raw() as i64 as u64,
    BASERT_REJECT_BOUNDS = Status::RejectBounds.raw() as i64 as u64,
    BASERT_REJECT_BAD_REF_GENERATION = Status::RejectBadRefGeneration.raw() as i64 as u64,
    BASERT_REJECT_UNKNOWN_LANE = Status::RejectUnknownLane.raw() as i64 as u64,
    BASERT_REJECT_DUPLICATE_LANE = Status::RejectDuplicateLane.raw() as i64 as u64,
    BASERT_REJECT_ILLEGAL_COMBINATION = Status::RejectIllegalCombination.raw() as i64 as u64,
    BASERT_REJECT_BUDGET = Status::RejectBudget.raw() as i64 as u64,
    BASERT_REJECT_STALE_SEED = Status::RejectStaleSeed.raw() as i64 as u64,
    BASERT_REJECT_HOST_RULES = Status::RejectHostRules.raw() as i64 as u64,
    BASERT_REJECT_TRANSFER_LOCKED = Status::RejectTransferLocked.raw() as i64 as u64,
    BASERT_REJECT_STALE_NONCE = Status::RejectStaleNonce.raw() as i64 as u64,
    BASERT_NEEDS_REPLAN = Status::NeedsReplan.raw() as i64 as u64,
    BASERT_REJECT_BAD_STRUCT = Status::RejectBadStruct.raw() as i64 as u64,
    BASERT_EBUSY = Status::Busy.raw() as i64 as u64,
    BASERT_ERR_BUFFER_TOO_SMALL = Status::BufferTooSmall.raw() as i64 as u64,
    BASERT_ERR_STALE_SIZING = Status::StaleSizing.raw() as i64 as u64,
    BASERT_ERR_OUT_OF_BOUNDARY = Status::OutOfBoundary.raw() as i64 as u64,
    BASERT_ERR_ENVELOPE_MISMATCH = Status::EnvelopeMismatch.raw() as i64 as u64,
    BASERT_ERR_CHECKSUM = Status::Checksum.raw() as i64 as u64,
    BASERT_ERR_IDENTITY_MISMATCH = Status::IdentityMismatch.raw() as i64 as u64,
    BASERT_ERR_OP_UNSUPPORTED = Status::Unsupported.raw() as i64 as u64,
    BASERT_ERR_REGISTRATION_REFUSED = Status::RegistrationRefused.raw() as i64 as u64,
    BASERT_ERR_UNKNOWN_HANDLE = Status::UnknownHandle.raw() as i64 as u64,
    BASERT_ERR_SEED_UNSERVABLE = Status::SeedUnservable.raw() as i64 as u64,
    BASERT_ERR_RING_LAYOUT = Status::RingLayout.raw() as i64 as u64,
    BASERT_REJECT_CERT_UNMATCHED = Status::RejectCertUnmatched.raw() as i64 as u64,
    BASERT_ERR_FATAL = Status::Fatal.raw() as i64 as u64,
    BASERT_FAULT_UNSPECIFIED = fault_code::UNSPECIFIED as u64,
    BASERT_FAULT_GRAMMAR_OVERFLOW = fault_code::GRAMMAR_OVERFLOW as u64,
    BASERT_FAULT_REF_SPAN_OOB = fault_code::REF_SPAN_OOB as u64,
    BASERT_FAULT_STRATEGY = fault_code::STRATEGY as u64,
    BASERT_FAULT_OOM_TENTATIVE = fault_code::OOM_TENTATIVE as u64,
    BASERT_FAULT_INTERNAL = fault_code::INTERNAL as u64,
    BASERT_SAMPLING_GPU_GREEDY = sampling::GPU_GREEDY as u64,
    BASERT_SAMPLING_GPU_GUMBEL = sampling::GPU_GUMBEL as u64,
    BASERT_SAMPLING_HOST = sampling::HOST as u64,
    BASERT_DETERMINISM_BEST_EFFORT = determinism::BEST_EFFORT as u64,
    BASERT_DETERMINISM_DETERMINISTIC = determinism::DETERMINISTIC as u64,
    BASERT_EXACTNESS_APPROXIMATE = exactness::APPROXIMATE as u64,
    BASERT_EXACTNESS_DISTRIBUTION_EXACT = exactness::DISTRIBUTION_EXACT as u64,
    BASERT_EXACTNESS_SEED_PATH_INVARIANT = exactness::SEED_PATH_INVARIANT as u64,
    BASERT_TICK_DRAIN = tick_flags::DRAIN as u64,
    BASERT_TICK_CAPTURE_OK = tick_flags::CAPTURE_OK as u64,
    BASERT_TICK_OK = tick_status::OK as u64,
    BASERT_TICK_SHED = tick_status::SHED as u64,
    BASERT_TICK_LANE_ERRORS = tick_status::LANE_ERRORS as u64,
    BASERT_TICK_FATAL = tick_status::FATAL as u64,
    BASERT_ADMIT_UNSPECIFIED = admit_status::UNSPECIFIED as u64,
    BASERT_ADMIT_ADMITTED = admit_status::ADMITTED as u64,
    BASERT_ADMIT_REJECTED = admit_status::REJECTED as u64,
    BASERT_FINISH_NONE = finish::NONE as u64,
    BASERT_FINISH_EOS = finish::EOS as u64,
    BASERT_FINISH_LENGTH = finish::LENGTH as u64,
    BASERT_FINISH_GRAMMAR = finish::GRAMMAR as u64,
    BASERT_FINISH_CANCELLED = finish::CANCELLED as u64,
    BASERT_FINISH_ERROR = finish::ERROR as u64,
    BASERT_OP_PENDING = op_state::PENDING as u64,
    BASERT_OP_RUNNING = op_state::RUNNING as u64,
    BASERT_OP_DONE = op_state::DONE as u64,
    BASERT_OP_FAILED = op_state::FAILED as u64,
    BASERT_OP_CANCELLED = op_state::CANCELLED as u64,
    BASERT_ENCODING_LOSSLESS = encoding::LOSSLESS as u64,
    BASERT_ENCODING_Q8 = encoding::Q8 as u64,
    BASERT_ENCODING_Q4 = encoding::Q4 as u64,
    BASERT_SPACE_KIND_UNSPECIFIED = space_kind::UNSPECIFIED as u64,
    BASERT_SPACE_KIND_PAGED_TOKEN_KV = space_kind::PAGED_TOKEN_KV as u64,
    BASERT_SPACE_KIND_RING_KV = space_kind::RING_KV as u64,
    BASERT_SPACE_KIND_RECURRENT_BLOB = space_kind::RECURRENT_BLOB as u64,
    BASERT_SPACE_KIND_DEPTH_PAGED_KV = space_kind::DEPTH_PAGED_KV as u64,
    BASERT_SPACE_KIND_ENCODER_CACHE = space_kind::ENCODER_CACHE as u64,
    BASERT_ARTIFACT_UNSPECIFIED = artifact_role::UNSPECIFIED as u64,
    BASERT_ARTIFACT_DRAFT_WEIGHTS = artifact_role::DRAFT_WEIGHTS as u64,
    BASERT_ARTIFACT_AUX_HEAD = artifact_role::AUX_HEAD as u64,
    BASERT_ARTIFACT_PROJECTION = artifact_role::PROJECTION as u64,
    BASERT_ARTIFACT_DRAFT_TOKENIZER = artifact_role::DRAFT_TOKENIZER as u64,
    BASERT_TAP_TENSOR_UNSPECIFIED = tap_tensor::UNSPECIFIED as u64,
    BASERT_TAP_TENSOR_HIDDEN = tap_tensor::HIDDEN as u64,
    BASERT_TAP_TENSOR_ATTN_OUT = tap_tensor::ATTN_OUT as u64,
    BASERT_TAP_TENSOR_EMBED = tap_tensor::EMBED as u64,
    BASERT_CERT_MODE_GREEDY = cert_mode::GREEDY as u64,
    BASERT_CERT_MODE_GUMBEL = cert_mode::GUMBEL as u64,
    BASERT_CERT_MODE_HOST = cert_mode::HOST as u64,
    BASERT_CERT_DOMAIN_NONE = cert_domain::NONE as u64,
    BASERT_CERT_DOMAIN_GREEDY_ONLY = cert_domain::GREEDY_ONLY as u64,
    BASERT_CERT_DOMAIN_PENALTY_FREE = cert_domain::PENALTY_FREE as u64,
    BASERT_CERT_DOMAIN_ANY = cert_domain::ANY as u64,
    BASERT_STRATEGY_CAP_UNSPECIFIED = strategy_cap::UNSPECIFIED as u64,
    BASERT_STRATEGY_CAP_PROPOSAL_LINEAR = strategy_cap::PROPOSAL_LINEAR as u64,
    BASERT_STRATEGY_CAP_PROPOSAL_BLOCK = strategy_cap::PROPOSAL_BLOCK as u64,
    BASERT_STRATEGY_CAP_PROPOSAL_TREE = strategy_cap::PROPOSAL_TREE as u64,
    BASERT_STRATEGY_CAP_MASK_ANCESTOR = strategy_cap::MASK_ANCESTOR as u64,
    BASERT_STRATEGY_CAP_HOST_COMPATIBLE = strategy_cap::HOST_COMPATIBLE as u64,
    BASERT_TAINT_QUANTIZED_DEMOTION = taint::QUANTIZED_DEMOTION as u64,
    BASERT_TAINT_DECOMPOSITION = taint::DECOMPOSITION as u64,
    BASERT_TIER_GPU = tier::GPU as u64,
    BASERT_TIER_HOST = tier::HOST as u64,
    BASERT_TIER_GONE = tier::GONE as u64,
    BASERT_SHED_UNSPECIFIED = shed_kind::UNSPECIFIED as u64,
    BASERT_SHED_PREFILL_CHUNK = shed_kind::PREFILL_CHUNK as u64,
    BASERT_SHED_CACHE_EVICTION = shed_kind::CACHE_EVICTION as u64,
    BASERT_FORK_EAGER = fork_flags::EAGER as u64,
    BASERT_CHECKSUM_XXH3_64 = checksum_kind::XXH3_64 as u64,
    BASERT_CHECKSUM_FNV1A64 = checksum_kind::FNV1A64 as u64,
    BASERT_ALL_SPACES = ALL_SPACES as u64,
    BASERT_STATE_ENVELOPE_MAGIC = STATE_ENVELOPE_MAGIC,
    BASERT_RING_MAGIC = RING_MAGIC,
    BASERT_RING_KIND_UNSPECIFIED = ring_kind::UNSPECIFIED as u64,
    BASERT_RING_KIND_TOKENS = ring_kind::TOKENS as u64,
    BASERT_RING_KIND_LOGITS = ring_kind::LOGITS as u64,
    BASERT_RING_ROLE_UNSPECIFIED = ring_role::UNSPECIFIED as u64,
    BASERT_RING_ROLE_ENGINE_READS = ring_role::ENGINE_READS as u64,
    BASERT_RING_ROLE_ENGINE_WRITES = ring_role::ENGINE_WRITES as u64,
);

#[cfg(abi_c_layout)]
mod probed {
    include!(concat!(env!("OUT_DIR"), "/c_layout.rs"));

    const fn order(a: &str, b: &str) -> std::cmp::Ordering {
        let (a, b) = (a.as_bytes(), b.as_bytes());
        let mut i = 0;
        while i < a.len() && i < b.len() {
            if a[i] != b[i] {
                return if a[i] < b[i] { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater };
            }
            i += 1;
        }
        if a.len() < b.len() {
            std::cmp::Ordering::Less
        } else if a.len() > b.len() {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }

    /// `sorted` is in byte order of its names, as the build script writes it.
    const fn find(sorted: &[(&str, u64)], key: &str) -> Option<u64> {
        let (mut lo, mut hi) = (0, sorted.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            match order(sorted[mid].0, key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(sorted[mid].1),
            }
        }
        None
    }

    // The build fails here, not at a later `cargo test`, when the Rust mirror
    // and the header disagree; the panic names the entry.
    const _: () = {
        let mut i = 0;
        while i < super::RUST_LAYOUT.len() {
            let (name, ours) = super::RUST_LAYOUT[i];
            match find(C_LAYOUT, name) {
                Some(theirs) if theirs == ours => {}
                _ => panic!("{}", name),
            }
            i += 1;
        }
        assert!(C_LAYOUT.len() == super::RUST_LAYOUT.len(), "the header has entries the Rust mirror does not check");
    };
}
