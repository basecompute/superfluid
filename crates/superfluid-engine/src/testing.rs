use superfluid_abi::*;

use crate::{Engine, EngineConfig, InMemoryRings, MockEngine, Rings};

pub trait StageRings: Rings {
    fn stage_prompt(&mut self, tokens: &[u32]) -> TokenRef;
}

impl StageRings for InMemoryRings {
    fn stage_prompt(&mut self, tokens: &[u32]) -> TokenRef {
        InMemoryRings::stage_prompt(self, tokens)
    }
}

pub const DEFAULT_MAX_DECODE_LANES: u32 = 8;

pub struct Harness<E: Engine = MockEngine, R: StageRings = InMemoryRings> {
    pub engine: E,
    pub rings: R,
    pub next_seq: std::cell::Cell<u64>,
    pub max_decode_lanes: u32,
}

impl Harness<MockEngine> {
    pub fn new() -> Harness<MockEngine> {
        Harness::with_config(EngineConfig::default())
    }

    pub fn with_config(cfg: EngineConfig) -> Harness<MockEngine> {
        Harness::with_engine(MockEngine::new(cfg))
    }
}

impl Default for Harness<MockEngine> {
    fn default() -> Self {
        Harness::new()
    }
}

impl<E: Engine> Harness<E> {
    pub fn with_engine(engine: E) -> Harness<E> {
        Harness::with_engine_and_rings(engine, InMemoryRings::new())
    }
}

impl<E: Engine, R: StageRings> Harness<E, R> {
    pub fn with_engine_and_rings(engine: E, rings: R) -> Harness<E, R> {
        Harness {
            engine,
            rings,
            next_seq: std::cell::Cell::new(1),
            max_decode_lanes: DEFAULT_MAX_DECODE_LANES,
        }
    }

    pub fn with_max_decode_lanes(mut self, n: u32) -> Self {
        self.max_decode_lanes = n;
        self
    }

    pub fn plan(&self) -> PlanBuilder {
        let seq = self.next_seq.get();
        self.next_seq.set(seq + 1);
        PlanBuilder::new(seq).max_decode_lanes(self.max_decode_lanes)
    }

    pub fn tick_ok(&mut self, pb: PlanBuilder) -> Events {
        let (plan, arena) = pb.build();
        let ev = self
            .engine
            .tick(&plan, &arena, &mut self.rings)
            .unwrap_or_else(|s| panic!("tick rejected: {s:?}"));
        Events::decode(ev)
    }

    pub fn tick_err(&mut self, pb: PlanBuilder) -> Status {
        let (plan, arena) = pb.build();
        match self.engine.tick(&plan, &arena, &mut self.rings) {
            Ok(_) => panic!("tick unexpectedly accepted"),
            Err(s) => s,
        }
    }

    pub fn prompt(&mut self, tokens: &[u32]) -> TokenRef {
        StageRings::stage_prompt(&mut self.rings, tokens)
    }

    pub fn rings_tokens(&self, r: &TokenRef) -> Vec<u32> {
        self.rings.read_tokens(r).unwrap()
    }
}

pub trait EngineInspect: Engine {
    fn lane_committed(&self, lane_tag: u64) -> Option<&[u32]>;
    fn inject_fault(&mut self, lane_tag: u64, code: u32);
}

impl EngineInspect for MockEngine {
    fn lane_committed(&self, lane_tag: u64) -> Option<&[u32]> {
        MockEngine::lane_committed(self, lane_tag)
    }
    fn inject_fault(&mut self, lane_tag: u64, code: u32) {
        MockEngine::inject_fault(self, lane_tag, code)
    }
}

pub struct Events {
    pub plan_seq: u64,
    pub tick_status: u32,
    pub admit_results: Vec<AdmitResult>,
    pub emits: Vec<LaneEmit>,
    pub faults: Vec<LaneFault>,
    pub shed_chunks: u32,
    pub shed_entries: Vec<ShedEntry>,
    pub op_completions: Vec<OpComplete>,
    pub logprobs: Vec<LaneLogprob>,
}

impl Events {
    pub fn decode(ev: &TickEvents) -> Events {
        let bounds = ArenaBounds {
            base: 0,
            len: usize::MAX,
        };
        // SAFETY: engine-owned arrays, valid until the next tick.
        unsafe {
            Events {
                plan_seq: ev.plan_seq,
                tick_status: ev.tick_status,
                admit_results: read_array(&ev.admit_results, &bounds).unwrap().collect(),
                emits: read_array(&ev.emits, &bounds).unwrap().collect(),
                faults: read_array(&ev.faults, &bounds).unwrap().collect(),
                shed_chunks: ev.shed.prefill_chunks_dropped,
                shed_entries: read_array(&ev.shed.entries, &bounds).unwrap().collect(),
                op_completions: read_array(&ev.op_completions, &bounds).unwrap().collect(),
                logprobs: read_array(&ev.logprobs, &bounds).unwrap().collect(),
            }
        }
    }

    pub fn emit_for(&self, lane: u64) -> &LaneEmit {
        self.emits
            .iter()
            .find(|e| e.lane_tag == lane)
            .expect("no emit for lane")
    }

    pub fn logprobs_for(&self, lane: u64) -> Vec<&LaneLogprob> {
        self.logprobs.iter().filter(|l| l.lane_tag == lane).collect()
    }

    pub fn admit_for(&self, lane: u64) -> &AdmitResult {
        self.admit_results
            .iter()
            .find(|a| a.lane_tag == lane)
            .expect("no admit result for lane")
    }
}

use superfluid_abi::array::read_array;

pub struct PlanBuilder {
    plan_seq: u64,
    flags: u32,
    prefill_budget: u32,
    max_decode_lanes: u32,
    admits: Vec<LaneAdmit>,
    commits: Vec<LaneCommit>,
    prefills: Vec<LanePrefill>,
    decodes: Vec<LaneDecode>,
    retires: Vec<LaneRetire>,
    victims: Vec<u64>,
    evictable_classes: u64,
    max_evict_bytes: u64,
    protected_quota_bytes: u64,
    on_partial_emit:
        Option<unsafe extern "C" fn(user: *mut std::ffi::c_void, lane_tag: u64, tokens: *const u32, n_tokens: u32)>,
    partial_emit_user: *mut std::ffi::c_void,
}

impl PlanBuilder {
    pub fn new(plan_seq: u64) -> PlanBuilder {
        PlanBuilder {
            plan_seq,
            flags: 0,
            prefill_budget: 4096,
            max_decode_lanes: 8,
            admits: Vec::new(),
            commits: Vec::new(),
            prefills: Vec::new(),
            decodes: Vec::new(),
            retires: Vec::new(),
            victims: Vec::new(),
            evictable_classes: u64::MAX,
            max_evict_bytes: u64::MAX,
            protected_quota_bytes: 0,
            on_partial_emit: None,
            partial_emit_user: std::ptr::null_mut(),
        }
    }

    pub fn flags(mut self, f: u32) -> Self {
        self.flags = f;
        self
    }

    pub fn prefill_budget(mut self, b: u32) -> Self {
        self.prefill_budget = b;
        self
    }

    pub fn max_decode_lanes(mut self, n: u32) -> Self {
        self.max_decode_lanes = n;
        self
    }

    pub fn evict_envelope(mut self, classes: u64, max_bytes: u64) -> Self {
        self.evictable_classes = classes;
        self.max_evict_bytes = max_bytes;
        self
    }

    pub fn protected_quota(mut self, bytes: u64) -> Self {
        self.protected_quota_bytes = bytes;
        self
    }

    pub fn partial_emit(
        mut self,
        cb: unsafe extern "C" fn(user: *mut std::ffi::c_void, lane_tag: u64, tokens: *const u32, n_tokens: u32),
        user: *mut std::ffi::c_void,
    ) -> Self {
        self.on_partial_emit = Some(cb);
        self.partial_emit_user = user;
        self
    }

    pub fn admit(mut self, lane: u64, prompt: TokenRef) -> Self {
        self.admits.push(basic_admit(lane, prompt));
        self
    }

    pub fn admit_with(
        mut self,
        f: impl FnOnce(LaneAdmit) -> LaneAdmit,
        lane: u64,
        prompt: TokenRef,
    ) -> Self {
        self.admits.push(f(basic_admit(lane, prompt)));
        self
    }

    pub fn commit(mut self, lane: u64, token: u32, nonce: u64) -> Self {
        self.commits.push(LaneCommit {
            lane_tag: lane,
            token_id: token,
            _pad0: 0,
            logits_nonce: nonce,
        });
        self
    }

    pub fn prefill(mut self, lane: u64, offset: u32, count: u32) -> Self {
        self.prefills.push(LanePrefill {
            lane_tag: lane,
            token_offset: offset,
            token_count: count,
        });
        self
    }

    pub fn decode(mut self, lane: u64, max_new: u16) -> Self {
        self.decodes.push(LaneDecode {
            lane_tag: lane,
            max_new_tokens: max_new,
            overshoot: 0,
            _pad1: 0,
        });
        self
    }

    pub fn retire(mut self, lane: u64, publish: bool) -> Self {
        self.retires.push(LaneRetire {
            lane_tag: lane,
            publish_to_cache: publish as u8,
            _pad0: [0; 7],
        });
        self
    }

    pub fn build(self) -> (TickPlan, RecordArena) {
        let mut arena = RecordArena::new();
        let plan = TickPlan {
            struct_size: std::mem::size_of::<TickPlan>() as u64,
            plan_seq: self.plan_seq,
            flags: self.flags,
            _pad0: 0,
            prefill_token_budget: self.prefill_budget,
            max_decode_lanes: self.max_decode_lanes,
            admits: arena.push_records(&self.admits),
            commits: arena.push_records(&self.commits),
            prefills: arena.push_records(&self.prefills),
            decodes: arena.push_records(&self.decodes),
            retires: arena.push_records(&self.retires),
            shed_policy: ShedPolicy {
                victim_lanes: arena.push_records(&self.victims),
                evictable_cache_classes: self.evictable_classes,
                protected_quota_bytes: self.protected_quota_bytes,
                max_evict_bytes: self.max_evict_bytes,
            },
            on_partial_emit: self.on_partial_emit,
            partial_emit_user: self.partial_emit_user,
        };
        (plan, arena)
    }
}

pub fn register_prompt_lookup_with<E: Engine>(
    h: &mut Harness<E>,
    impl_hash: [u8; 32],
    config_hash: [u8; 32],
    kernel_caps_required: u32,
) -> Result<(u32, Vec<ExactnessCert>), Status> {
    let strategy_id = std::ffi::CString::new("prompt-lookup").unwrap();
    let impl_version = std::ffi::CString::new("1.0.0").unwrap();
    let mut arena = RecordArena::new();
    let caps = [CapabilityReq {
        kind_id: strategy_cap::PROPOSAL_LINEAR,
        _pad0: 0,
        params: Array::EMPTY,
    }];
    let reg = StrategyRegistration {
        struct_size: std::mem::size_of::<StrategyRegistration>() as u64,
        strategy_id: strategy_id.as_ptr(),
        impl_version: impl_version.as_ptr(),
        impl_hash,
        config_hash,
        artifacts: Array::EMPTY,
        taps: Array::EMPTY,
        capabilities: arena.push_records(&caps),
        target_archs: Array::EMPTY,
        kernel_caps_required,
        _pad0: 0,
        est_state_bytes: 0,
        claimed_exactness: exactness::SEED_PATH_INVARIANT,
        _pad1: [0; 3],
        rng_contract_version: 1,
    };
    let grant = h.engine.strategy_register(&reg)?;
    let bounds = ArenaBounds {
        base: 0,
        len: usize::MAX,
    };
    // SAFETY: engine-owned grant arrays, next-call lifetime.
    let certs: Vec<ExactnessCert> =
        unsafe { superfluid_abi::array::read_array(&grant.certificates, &bounds) }
            .unwrap()
            .collect();
    Ok((grant.strategy_slot, certs))
}

pub fn basic_admit(lane: u64, prompt: TokenRef) -> LaneAdmit {
    LaneAdmit {
        lane_tag: lane,
        prompt,
        seed_handle: 0,
        sampling: sampling::GPU_GREEDY,
        logit_bias_handle: 0,
        params: SamplingParams { flags: sampling_flag::IGNORE_EOS, ..Default::default() },
        rng_counter_base: 0,
        grammar_handle: 0,
        strategy_slot: 0,
        determinism_class: determinism::BEST_EFFORT,
        minimum_exactness: exactness::APPROXIMATE,
        allow_approximate: 0,
        want_logprobs: 0,
        required_cert_id: 0,
        host_sampler_identity: [0; 32],
        grammar_replay: 0,
        decode_replay: 0,
    }
}

pub fn rows_agree(reference: &[f32], other: &[f32], tolerance: f32) -> Result<(), String> {
    if reference.len() != other.len() {
        return Err(format!("rows of {} and {} logits", reference.len(), other.len()));
    }
    let (mut at, mut diff) = (0, 0f32);
    for (i, (&a, &b)) in reference.iter().zip(other).enumerate() {
        if a.is_nan() || b.is_nan() {
            return Err(format!("logit at id {i} is not a number ({a} vs {b})"));
        }
        if a.is_infinite() || b.is_infinite() {
            if a == b {
                continue;
            }
            return Err(format!("logit at id {i}: {a} vs {b}"));
        }
        let d = (a - b).abs();
        if d > diff {
            (at, diff) = (i, d);
        }
    }
    if diff > tolerance {
        return Err(format!(
            "largest logit difference {diff} at id {at} ({} vs {}) exceeds {tolerance}",
            reference[at], other[at]
        ));
    }
    Ok(())
}

pub fn top2_margin(record: &LaneLogprob) -> f32 {
    match record.n_top {
        0 | 1 => f32::INFINITY,
        _ => record.top_logprobs[0] - record.top_logprobs[1],
    }
}

pub fn tokens_agree(reference: &[u32], margins: &[f32], other: &[u32], tolerance: f32) -> Result<(), String> {
    for (k, (a, b)) in reference.iter().zip(other).enumerate() {
        if a == b {
            continue;
        }
        let margin = margins.get(k).copied().unwrap_or(f32::INFINITY);
        if margin <= 2.0 * tolerance {
            return Ok(());
        }
        return Err(format!(
            "step {k}: {b} where the reference took {a} by a margin of {margin} (tolerance {tolerance}): {reference:?} vs {other:?}"
        ));
    }
    if reference.len() != other.len() {
        return Err(format!("{} tokens where the reference has {}: {reference:?} vs {other:?}", other.len(), reference.len()));
    }
    Ok(())
}

pub mod tick_contract {
    use super::*;

    pub fn admit_prefill_decode_in_one_tick<E: EngineInspect, R: StageRings>(mut h: Harness<E, R>) {
        let p = h.prompt(&[1, 2, 3, 4]);
        let ev = h.tick_ok(h.plan().admit(7, p).prefill(7, 0, 4).decode(7, 3));
        assert_eq!(ev.tick_status, tick_status::OK);
        assert_eq!(ev.admit_for(7).status, admit_status::ADMITTED);
        let e = ev.emit_for(7);
        assert_eq!(e.n_tokens, 3);
        assert_eq!(e.finish, finish::NONE);
        assert_eq!(h.engine.lane_committed(7).unwrap().len(), 7);
    }

    pub fn same_log_same_seed_same_tokens<E: Engine, R: StageRings>(make: impl Fn() -> Harness<E, R>) {
        let run = || {
            let mut h = make();
            let p = h.prompt(&[10, 20, 30]);
            let ev = h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 3).decode(1, 8));
            let e = *ev.emit_for(1);
            h.rings_tokens(&e.token_ref)
        };
        assert_eq!(run(), run());
    }

    pub fn plan_seq_must_be_monotonic<E: Engine, R: StageRings>(mut h: Harness<E, R>) {
        let p = h.prompt(&[1]);
        h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 1));
        let stale = PlanBuilder::new(1);
        assert_eq!(h.tick_err(stale), Status::RejectIllegalCombination);
    }

    pub fn structural_rejections<E: Engine, R: StageRings>(mut h: Harness<E, R>) {
        let p = h.prompt(&[1, 2]);
        let err = h.tick_err(h.plan().admit(1, p).admit(1, p));
        assert_eq!(err, Status::RejectDuplicateLane);
        let p2 = h.prompt(&[1, 2]);
        let err = h.tick_err(h.plan().admit(2, p2).retire(2, false));
        assert_eq!(err, Status::RejectIllegalCombination);
        let err = h.tick_err(h.plan().commit(99, 5, 1));
        assert_eq!(err, Status::RejectUnknownLane);
        let err = h.tick_err(h.plan().decode(98, 1));
        assert_eq!(err, Status::RejectUnknownLane);
        let err = h.tick_err(h.plan().max_decode_lanes(1000));
        assert_eq!(err, Status::RejectBudget);
        let p3 = h.prompt(&[5]);
        let ev = h.tick_ok(h.plan().admit(3, p3).prefill(3, 0, 1).decode(3, 1));
        assert_eq!(ev.tick_status, tick_status::OK);
    }

    pub fn stale_ring_generation_rejected<E: Engine, R: StageRings>(mut h: Harness<E, R>) {
        let mut p = h.prompt(&[1, 2, 3]);
        p.generation += 7;
        let err = h.tick_err(h.plan().admit(1, p));
        assert_eq!(err, Status::RejectBadRefGeneration);
    }

    pub fn drain_takes_no_new_work<E: Engine, R: StageRings>(mut h: Harness<E, R>) {
        let p = h.prompt(&[1]);
        let err = h.tick_err(h.plan().flags(tick_flags::DRAIN).admit(1, p));
        assert_eq!(err, Status::RejectIllegalCombination);
        let p2 = h.prompt(&[1, 2]);
        h.tick_ok(h.plan().admit(2, p2).prefill(2, 0, 2));
        let ev = h.tick_ok(h.plan().flags(tick_flags::DRAIN).decode(2, 1));
        assert_eq!(ev.emit_for(2).n_tokens, 1);
    }

    pub fn a_replayed_decode_tail_resumes_the_lane<E: Engine, R: StageRings>(mut h: Harness<E, R>) {
        const CUT: usize = 5;
        let prompt: Vec<u32> = vec![11, 12, 13, 14, 15, 16];
        const SEED: u64 = 1000;
        let admitted_at = |len: usize| {
            move |mut a: LaneAdmit| {
                a.params.flags |= sampling_flag::IGNORE_EOS;
                a.rng_counter_base = SEED + len as u64;
                a
            }
        };
        let n = prompt.len() as u32;
        let plain = admitted_at(prompt.len());
        let resumed_len = prompt.len() + CUT;
        let replayed = move |a: LaneAdmit| {
            let mut a = admitted_at(resumed_len)(a);
            a.decode_replay = CUT as u32;
            a
        };

        let p = h.prompt(&prompt);
        let ev = h.tick_ok(h.plan().admit_with(plain, 1, p).prefill(1, 0, n).decode(1, 8));
        let e = *ev.emit_for(1);
        let full = h.rings_tokens(&e.token_ref);
        assert_eq!(full.len(), 8);
        h.tick_ok(h.plan().retire(1, false));

        let mut resumed = prompt.clone();
        resumed.extend_from_slice(&full[..CUT]);

        let p = h.prompt(&resumed);
        h.tick_err(h.plan().admit_with(replayed, 2, p).prefill(2, 0, resumed.len() as u32).decode(2, 3));

        let p = h.prompt(&resumed);
        let ev = h.tick_ok(h.plan().admit_with(replayed, 2, p).prefill(2, 0, n).decode(2, CUT as u16 + 3));
        let e = *ev.emit_for(2);
        assert_eq!(e.finish, finish::NONE);
        assert_eq!(h.rings_tokens(&e.token_ref), full[CUT..], "the continuation is the lane it resumes");
        h.tick_ok(h.plan().retire(2, false));

        let p = h.prompt(&resumed);
        let ev = h.tick_ok(h.plan().admit_with(replayed, 3, p).prefill(3, 0, n).decode(3, 3));
        assert_eq!((ev.emit_for(3).n_tokens, ev.emit_for(3).finish), (0, finish::NONE), "inside the tail");
        let ev = h.tick_ok(h.plan().decode(3, CUT as u16 - 3));
        assert_eq!(ev.emit_for(3).n_tokens, 0, "the tail's last token takes the grant's last unit");
        let ev = h.tick_ok(h.plan().decode(3, 3));
        let e = *ev.emit_for(3);
        assert_eq!(h.rings_tokens(&e.token_ref), full[CUT..], "paced or not, the same tokens");
        h.tick_ok(h.plan().retire(3, false));

        let p = h.prompt(&resumed);
        let whole = move |a: LaneAdmit| {
            let mut a = admitted_at(resumed_len)(a);
            a.decode_replay = resumed_len as u32;
            a
        };
        h.tick_err(h.plan().admit_with(whole, 4, p).decode(4, 1));
    }

    pub fn host_alternation_full_loop<E: EngineInspect, R: StageRings>(mut h: Harness<E, R>) {
        let p = h.prompt(&[1, 2]);
        let ev = h.tick_ok(
            h.plan()
                .admit_with(
                    |mut a| {
                        a.sampling = sampling::HOST;
                        a
                    },
                    1,
                    p,
                )
                .prefill(1, 0, 2)
                .decode(1, 1),
        );
        let e = ev.emit_for(1);
        assert_eq!(e.n_tokens, 0);
        let nonce = e.logits_row.generation;
        assert_ne!(nonce, 0);

        let err = h.tick_err(h.plan().decode(1, 1));
        assert_eq!(err, Status::RejectHostRules);

        let err = h.tick_err(h.plan().commit(1, 42, nonce + 1).decode(1, 1));
        assert_eq!(err, Status::RejectStaleNonce);

        let ev = h.tick_ok(h.plan().commit(1, 42, nonce).decode(1, 1));
        let committed = h.engine.lane_committed(1).unwrap();
        assert_eq!(*committed.last().unwrap(), 42);
        let e2 = ev.emit_for(1);
        assert_eq!(e2.n_tokens, 0);
        assert_ne!(e2.logits_row.generation, nonce, "fresh row, fresh nonce");
    }

    pub fn host_lane_max_new_tokens_must_be_one<E: Engine, R: StageRings>(mut h: Harness<E, R>) {
        let p = h.prompt(&[1]);
        let err = h.tick_err(
            h.plan()
                .admit_with(
                    |mut a| {
                        a.sampling = sampling::HOST;
                        a
                    },
                    1,
                    p,
                )
                .prefill(1, 0, 1)
                .decode(1, 4),
        );
        assert_eq!(err, Status::RejectHostRules);
    }

    pub fn commit_on_gpu_lane_rejected<E: Engine, R: StageRings>(mut h: Harness<E, R>) {
        let p = h.prompt(&[1]);
        h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 1));
        let err = h.tick_err(h.plan().commit(1, 5, 1));
        assert_eq!(err, Status::RejectIllegalCombination);
    }

    pub fn prefill_budget_sheds_and_reports<E: Engine, R: StageRings>(mut h: Harness<E, R>) {
        let tokens: Vec<u32> = (0..100).collect();
        let p = h.prompt(&tokens);

        let err = h.tick_err(
            h.plan()
                .prefill_budget(10)
                .admit(1, p)
                .prefill(1, 0, 100)
                .decode(1, 1),
        );
        assert_eq!(err, Status::RejectIllegalCombination);

        let ev = h.tick_ok(h.plan().prefill_budget(10).admit(1, p).prefill(1, 0, 100));
        assert_eq!(ev.tick_status, tick_status::SHED);
        assert_eq!(ev.shed_chunks, 1);
        let entry = ev
            .shed_entries
            .iter()
            .find(|e| e.kind == shed_kind::PREFILL_CHUNK)
            .unwrap();
        assert_eq!(entry.tokens, 90);
        assert_eq!(entry.lane_tag, 1);

        let ev = h.tick_ok(h.plan().prefill_budget(200).prefill(1, 10, 90).decode(1, 1));
        assert_eq!(ev.tick_status, tick_status::OK);
        assert_eq!(ev.emit_for(1).n_tokens, 1);
    }

    pub fn noncontiguous_prefill_rejected<E: Engine, R: StageRings>(mut h: Harness<E, R>) {
        let p = h.prompt(&[1, 2, 3, 4]);
        h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 2));
        let err = h.tick_err(h.plan().prefill(1, 3, 1));
        assert_eq!(err, Status::RejectIllegalCombination);
    }

    pub fn per_lane_fault_does_not_fail_tick<E: EngineInspect, R: StageRings>(mut h: Harness<E, R>) {
        let p1 = h.prompt(&[1, 2]);
        let p2 = h.prompt(&[3, 4]);
        h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 2));
        h.tick_ok(h.plan().admit(2, p2).prefill(2, 0, 2));
        h.engine
            .inject_fault(1, superfluid_abi::status::fault_code::GRAMMAR_OVERFLOW);
        let ev = h.tick_ok(h.plan().decode(1, 1).decode(2, 1));
        assert_eq!(ev.tick_status, tick_status::LANE_ERRORS);
        assert_eq!(ev.faults.len(), 1);
        assert_eq!(ev.faults[0].lane_tag, 1);
        assert_eq!(
            ev.faults[0].code,
            superfluid_abi::status::fault_code::GRAMMAR_OVERFLOW
        );
        assert_eq!(ev.emit_for(2).n_tokens, 1);
        let err = h.tick_err(h.plan().decode(1, 1));
        assert_eq!(err, Status::RejectUnknownLane);
        let ev = h.tick_ok(h.plan().retire(1, true).decode(2, 1));
        assert_eq!(ev.emit_for(2).n_tokens, 1);
        let err = h.tick_err(h.plan().retire(1, false));
        assert_eq!(err, Status::RejectUnknownLane);
    }

    pub fn retire_publish_then_seeded_readmission<E: EngineInspect, R: StageRings>(mut h: Harness<E, R>) {
        let paged_desc = h
            .engine
            .state_spaces()
            .iter()
            .find(|d| d.kind == space_kind::PAGED_TOKEN_KV)
            .copied()
            .expect("the engine advertises a paged token-KV space");
        let space = paged_desc.space_id;
        let unit = lcm(paged_desc.page_size_tokens.max(1) as u64, paged_desc.snapshot_interval_tokens.max(1) as u64);
        let n = 64u64.div_ceil(unit) * unit;
        let tokens: Vec<u32> = (100..100 + n as u32).collect();
        let p = h.prompt(&tokens);
        h.tick_ok(h.plan().admit(1, p).prefill(1, 0, n as u32));
        h.tick_ok(h.plan().retire(1, true));

        let mut longer = tokens.clone();
        longer.push(100 + n as u32);
        let m = h.engine.space_match(ALL_SPACES, &longer, &[]).unwrap();
        let bounds = ArenaBounds {
            base: 0,
            len: usize::MAX,
        };
        // SAFETY: engine-owned result, next-call lifetime.
        let spaces: Vec<SpaceMatch> =
            unsafe { superfluid_abi::array::read_array(&m.spaces, &bounds) }
                .unwrap()
                .collect();
        let paged = spaces.iter().find(|s| s.space_id == space).expect("the paged space in the match");
        // SAFETY: as above.
        let cands: Vec<MatchCandidate> =
            unsafe { superfluid_abi::array::read_array(&paged.candidates, &bounds) }
                .unwrap()
                .collect();
        assert!(cands.iter().any(|c| c.prefix_len == n), "candidates {:?} carry the {n}-token prefix", cands.iter().map(|c| c.prefix_len).collect::<Vec<_>>());

        let handle = h
            .engine
            .seed_acquire(&longer, n, determinism::BEST_EFFORT)
            .unwrap();

        let p2 = h.prompt(&longer);
        let ev = h.tick_ok(
            h.plan()
                .admit_with(
                    |mut a| {
                        a.seed_handle = handle;
                        a
                    },
                    2,
                    p2,
                )
                .prefill(2, n as u32, 1)
                .decode(2, 1),
        );
        assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
        assert_eq!(ev.emit_for(2).n_tokens, 1);
        assert_eq!(h.engine.lane_committed(2).unwrap().len(), n as usize + 2);

        let p3 = h.prompt(&longer);
        let err = h.tick_err(h.plan().admit_with(
            |mut a| {
                a.seed_handle = handle;
                a
            },
            3,
            p3,
        ));
        assert_eq!(err, Status::RejectStaleSeed);
    }

    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    fn lcm(a: u64, b: u64) -> u64 {
        a / gcd(a, b) * b
    }

    pub fn events_echo_plan_seq<E: Engine, R: StageRings>(mut h: Harness<E, R>) {
        let p = h.prompt(&[1]);
        let ev = h.tick_ok(h.plan().admit(9, p).prefill(9, 0, 1));
        assert_eq!(ev.plan_seq, 1);
        let ev = h.tick_ok(h.plan());
        assert_eq!(ev.plan_seq, 2);
    }
}
