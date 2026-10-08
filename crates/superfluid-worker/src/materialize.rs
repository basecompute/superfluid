//! Wire ⇄ ABI materialization.

use superfluid_abi::{
    array::read_array, ArenaBounds, AdmitResult, LaneAdmit, LaneCommit,
    LaneDecode, LaneEmit, LaneFault, LanePrefill, LaneRetire,
    MatchCandidate, MatchResult, OpComplete, RingRef, SamplingParams,
    ShedEntry, ShedPolicy, SpaceMatch, StateSpaceDesc, TickEvents,
    TickPlan, TokenRef, RecordArena,
};
use superfluid_proto::handshake::StateSpaceDescMsg;
use superfluid_proto::linkw::*;

pub fn token_ref_from_msg(m: &TokenRefMsg) -> TokenRef {
    TokenRef {
        ring_id: m.ring_id,
        index: m.index,
        count: m.count,
        _pad0: 0,
        generation: m.generation,
    }
}

pub fn token_ref_to_msg(r: &TokenRef) -> TokenRefMsg {
    TokenRefMsg {
        ring_id: r.ring_id,
        index: r.index,
        count: r.count,
        generation: r.generation,
    }
}

pub fn ring_ref_to_msg(r: &RingRef) -> RingRefMsg {
    RingRefMsg {
        ring_id: r.ring_id,
        index: r.index,
        generation: r.generation,
    }
}

fn sampling_params(m: &SamplingParamsMsg) -> SamplingParams {
    SamplingParams {
        temperature: m.temperature,
        top_p: m.top_p,
        min_p: m.min_p,
        top_k: m.top_k,
        freq_penalty: m.freq_penalty,
        presence_penalty: m.presence_penalty,
        repeat_penalty: m.repeat_penalty,
        flags: m.flags,
    }
}

pub fn materialize_plan(msg: &TickPlanMsg) -> (TickPlan, RecordArena) {
    let mut arena = RecordArena::new();

    let admits: Vec<LaneAdmit> = msg
        .admits
        .iter()
        .map(|a| LaneAdmit {
            lane_tag: a.lane_tag,
            prompt: token_ref_from_msg(&a.prompt),
            seed_handle: a.seed_handle,
            sampling: a.sampling,
            logit_bias_handle: a.logit_bias_handle,
            params: sampling_params(&a.params),
            rng_counter_base: a.rng_counter_base,
            grammar_handle: a.grammar_handle,
            strategy_slot: a.strategy_slot,
            determinism_class: a.determinism_class,
            minimum_exactness: a.minimum_exactness,
            allow_approximate: a.allow_approximate as u8,
            want_logprobs: if a.want_logprobs { 1 + a.top_logprobs.min(20) } else { 0 },
            required_cert_id: a.required_cert_id,
            host_sampler_identity: a.host_sampler_identity,
            grammar_replay: a.grammar_replay,
            decode_replay: a.decode_replay,
        })
        .collect();
    let commits: Vec<LaneCommit> = msg
        .commits
        .iter()
        .map(|c| LaneCommit {
            lane_tag: c.lane_tag,
            token_id: c.token_id,
            _pad0: 0,
            logits_nonce: c.logits_nonce,
        })
        .collect();
    let prefills: Vec<LanePrefill> = msg
        .prefills
        .iter()
        .map(|p| LanePrefill {
            lane_tag: p.lane_tag,
            token_offset: p.token_offset,
            token_count: p.token_count,
        })
        .collect();
    let decodes: Vec<LaneDecode> = msg
        .decodes
        .iter()
        .map(|d| LaneDecode {
            lane_tag: d.lane_tag,
            max_new_tokens: d.max_new_tokens,
            overshoot: d.overshoot,
            _pad1: 0,
        })
        .collect();
    let retires: Vec<LaneRetire> = msg
        .retires
        .iter()
        .map(|r| LaneRetire {
            lane_tag: r.lane_tag,
            publish_to_cache: r.publish_to_cache as u8,
            _pad0: [0; 7],
        })
        .collect();

    let plan = TickPlan {
        struct_size: std::mem::size_of::<TickPlan>() as u64,
        plan_seq: msg.plan_seq,
        flags: msg.flags,
        _pad0: 0,
        prefill_token_budget: msg.prefill_token_budget,
        max_decode_lanes: msg.max_decode_lanes,
        admits: arena.push_records(&admits),
        commits: arena.push_records(&commits),
        prefills: arena.push_records(&prefills),
        decodes: arena.push_records(&decodes),
        retires: arena.push_records(&retires),
        shed_policy: ShedPolicy {
            victim_lanes: arena.push_records(&msg.shed_policy.victim_lanes),
            evictable_cache_classes: msg.shed_policy.evictable_cache_classes,
            protected_quota_bytes: msg.shed_policy.protected_quota_bytes,
            max_evict_bytes: msg.shed_policy.max_evict_bytes,
        },
        on_partial_emit: None,
        partial_emit_user: std::ptr::null_mut(),
    };
    (plan, arena)
}

pub fn events_to_msg(ev: &TickEvents) -> TickEventsMsg {
    let bounds = ArenaBounds {
        base: 0,
        len: usize::MAX,
    };
    // SAFETY: engine-owned arrays valid until the next tick on this
    // bundle; serialization happens before any next tick by construction.
    unsafe {
        TickEventsMsg {
            plan_seq: ev.plan_seq,
            tick_status: ev.tick_status,
            admit_results: read_array::<AdmitResult>(&ev.admit_results, &bounds)
                .map(|it| {
                    it.map(|r| AdmitResultMsg {
                        lane_tag: r.lane_tag,
                        status: r.status,
                        reject_code: r.reject_code,
                        cert_id: r.cert_id,
                        granted_class: r.granted_class,
                    })
                    .collect()
                })
                .unwrap_or_default(),
            emits: read_array::<LaneEmit>(&ev.emits, &bounds)
                .map(|it| {
                    it.map(|e| LaneEmitMsg {
                        lane_tag: e.lane_tag,
                        token_ref: token_ref_to_msg(&e.token_ref),
                        n_tokens: e.n_tokens,
                        finish: e.finish,
                        logits_row: ring_ref_to_msg(&e.logits_row),
                        spec: SpecStatsMsg {
                            proposed: e.spec.proposed,
                            accepted: e.spec.accepted,
                            cert_id_in_effect: e.spec.cert_id_in_effect,
                        },
                    })
                    .collect()
                })
                .unwrap_or_default(),
            faults: read_array::<LaneFault>(&ev.faults, &bounds)
                .map(|it| {
                    it.map(|f| LaneFaultMsg {
                        lane_tag: f.lane_tag,
                        code: f.code,
                        detail: f.detail,
                    })
                    .collect()
                })
                .unwrap_or_default(),
            shed: ShedReportMsg {
                prefill_chunks_dropped: ev.shed.prefill_chunks_dropped,
                evictions_performed: ev.shed.evictions_performed,
                bytes_evicted: ev.shed.bytes_evicted,
                entries: read_array::<ShedEntry>(&ev.shed.entries, &bounds)
                    .map(|it| {
                        it.map(|s| ShedEntryMsg {
                            lane_tag: s.lane_tag,
                            kind: s.kind,
                            reason: s.reason,
                            bytes: s.bytes,
                            tokens: s.tokens,
                        })
                        .collect()
                    })
                    .unwrap_or_default(),
            },
            op_completions: read_array::<OpComplete>(&ev.op_completions, &bounds)
                .map(|it| {
                    it.map(|o| OpCompleteMsg {
                        op: o.op,
                        state: o.state,
                        error: o.error,
                        bytes_moved: o.bytes_moved,
                    })
                    .collect()
                })
                .unwrap_or_default(),
            timings: TickTimingsMsg {
                wall_ns: ev.timings.wall_ns,
                prefill_ns: ev.timings.prefill_ns,
                decode_ns: ev.timings.decode_ns,
                graph_hits: ev.timings.graph_hits,
                graph_misses: ev.timings.graph_misses,
            },
            mem: MemCountersMsg {
                allocated_bytes: ev.mem.allocated_bytes,
                host_retained_bytes: ev.mem.host_retained_bytes,
                pool_blocks_total: ev.mem.pool_blocks_total,
                pool_blocks_used: ev.mem.pool_blocks_used,
                pool_bytes_evictable: ev.mem.pool_bytes_evictable,
                tentative_bytes: ev.mem.tentative_bytes,
            },
            logprobs: read_array::<superfluid_abi::LaneLogprob>(&ev.logprobs, &bounds)
                .map(|it| {
                    it.map(|l| {
                        let n = (l.n_top as usize).min(superfluid_abi::MAX_TOP_LOGPROBS);
                        superfluid_proto::linkw::LaneLogprobMsg {
                            lane_tag: l.lane_tag,
                            logprob_bits: l.logprob.to_bits(),
                            top_ids: l.top_ids[..n].to_vec(),
                            top_logprob_bits: l.top_logprobs[..n].iter().map(|f| f.to_bits()).collect(),
                        }
                    })
                    .collect()
                })
                .unwrap_or_default(),
        }
    }
}

pub fn desc_to_msg(d: &StateSpaceDesc) -> StateSpaceDescMsg {
    let name_end = d.name.iter().position(|b| *b == 0).unwrap_or(d.name.len());
    StateSpaceDescMsg {
        space_id: d.space_id,
        kind: d.kind,
        version_tag: d.version_tag,
        bytes_per_token: d.bytes_per_token,
        blob_bytes: d.blob_bytes,
        page_size_tokens: d.page_size_tokens,
        fork_cost_class: d.fork_cost_class,
        fork_cost_bytes: d.fork_cost_bytes,
        snapshot_cadence: d.snapshot_cadence,
        snapshot_interval_tokens: d.snapshot_interval_tokens,
        placement: d.placement,
        flags: d.flags,
        name: String::from_utf8_lossy(&d.name[..name_end]).into_owned(),
    }
}

pub fn match_result_to_msg(m: &MatchResult) -> MatchResMsg {
    let bounds = ArenaBounds {
        base: 0,
        len: usize::MAX,
    };
    // SAFETY: engine-owned match result, next-call lifetime.
    let spaces = unsafe {
        read_array::<SpaceMatch>(&m.spaces, &bounds)
            .map(|it| {
                it.map(|s| SpaceMatchMsg {
                    space_id: s.space_id,
                    candidates: read_array::<MatchCandidate>(&s.candidates, &bounds)
                        .map(|cands| {
                            cands
                                .map(|c| MatchCandidateMsg {
                                    prefix_len: c.prefix_len,
                                    provenance_digest: c.provenance_digest,
                                    taint_bits: c.taint_bits,
                                    resident_tier: c.resident_tier,
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                })
                .collect()
            })
            .unwrap_or_default()
    };
    MatchResMsg { spaces }
}
