//! What the executor adds beyond the tick contract.

use std::ffi::c_void;

use superfluid_abi::*;
use superfluid_engine::testing::Harness;
use superfluid_engine::Engine;
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives, PrimError, RuntimePrimitives};

fn with(cfg: FakeConfig) -> Harness<Executor<FakePrimitives>> {
    Harness::with_engine(Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()))
}

fn harness() -> Harness<Executor<FakePrimitives>> {
    with(FakeConfig::default())
}

fn sampled(mut a: LaneAdmit, base: u64) -> LaneAdmit {
    a.sampling = sampling::GPU_GUMBEL;
    a.params = SamplingParams { temperature: 0.9, top_p: 0.95, top_k: 40, ..Default::default() };
    a.rng_counter_base = base;
    a
}

#[test]
fn sampled_continuation_is_position_indexed() {
    const SEED: u64 = 1000;
    let prompt: Vec<u32> = vec![5, 6, 7];
    let mut h = harness();
    let p = h.prompt(&prompt);
    let ev = h.tick_ok(
        h.plan().admit_with(|a| sampled(a, SEED + prompt.len() as u64), 1, p).prefill(1, 0, 3).decode(1, 8),
    );
    let full = h.rings_tokens(&ev.emit_for(1).token_ref);
    assert_eq!(full.len(), 8);

    let mut h2 = harness();
    let mut cont = prompt.clone();
    cont.extend_from_slice(&full[..4]);
    let p2 = h2.prompt(&cont);
    let ev = h2.tick_ok(
        h2.plan().admit_with(|a| sampled(a, SEED + cont.len() as u64), 1, p2).prefill(1, 0, 7).decode(1, 4),
    );
    let tail = h2.rings_tokens(&ev.emit_for(1).token_ref);
    assert_eq!(tail, full[4..].to_vec(), "the continuation must draw what the whole lane drew");
}

#[test]
fn a_replayed_tail_is_fed_one_token_a_step() {
    let prompt: Vec<u32> = (10..20).collect();
    let replayed = |mut a: LaneAdmit| {
        a.decode_replay = 4;
        a
    };
    let mut h = harness();
    let p = h.prompt(&prompt);
    let ev = h.tick_ok(h.plan().admit_with(replayed, 1, p).prefill(1, 0, 6).decode(1, 6));
    assert_eq!(ev.emit_for(1).n_tokens, 2);
    assert_eq!(
        h.engine.primitives().feeds,
        [(6, false), (1, false), (1, false), (1, false), (1, true), (1, true)],
        "the slice, the tail a token at a time, then the first drawn token fed for the next row"
    );
    assert_eq!(h.engine.lane_committed(1).unwrap()[..10], prompt[..]);

    let mut h = harness();
    let p = h.prompt(&prompt);
    let ev = h.tick_ok(h.plan().admit_with(replayed, 1, p).prefill(1, 0, 6).decode(1, 4));
    assert_eq!(ev.emit_for(1).n_tokens, 0);
    assert_eq!(h.engine.primitives().feeds, [(6, false), (1, false), (1, false), (1, false)]);
    let ev = h.tick_ok(h.plan().decode(1, 1));
    assert_eq!(ev.emit_for(1).n_tokens, 1);
    assert_eq!(h.engine.primitives().feeds.last(), Some(&(1, true)), "the held-back token, for its row");

    let mut h = harness();
    let p = h.prompt(&prompt);
    assert_eq!(h.tick_err(h.plan().admit_with(replayed, 1, p).prefill(1, 0, 7).decode(1, 4)), Status::RejectBounds);
    let p = h.prompt(&prompt);
    assert_eq!(
        h.tick_err(h.plan().admit_with(replayed, 1, p).prefill(1, 0, 5).decode(1, 4)),
        Status::RejectIllegalCombination
    );
    assert!(h.engine.primitives().feeds.is_empty(), "a refused plan runs no forward");
}

#[test]
fn greedy_rounds_take_the_runtime_argmax_and_match_host_rows() {
    let run = |engine_sampling: bool, second: fn(LaneAdmit) -> LaneAdmit| {
        let mut h = with(FakeConfig { engine_sampling, ..Default::default() });
        let p1 = h.prompt(&[3, 1, 4, 1, 5]);
        let p2 = h.prompt(&[9, 2, 6, 5, 3, 5, 8]);
        let ev = h.tick_ok(
            h.plan().admit(1, p1).admit_with(second, 2, p2).prefill(1, 0, 5).prefill(2, 0, 7).decode(1, 12).decode(2, 12),
        );
        (h.rings_tokens(&ev.emit_for(1).token_ref), h.rings_tokens(&ev.emit_for(2).token_ref))
    };
    let greedy = |a: LaneAdmit| a;
    let (a1, a2) = run(true, greedy);
    let (b1, b2) = run(false, greedy);
    assert_eq!((&a1, &a2), (&b1, &b2), "runtime argmax and host argmax agree");
    assert_eq!(a1.len(), 12);
    let (c1, c2) = run(true, |a| sampled(a, 77));
    let (d1, d2) = run(false, |a| sampled(a, 77));
    assert_eq!((&c1, &c2), (&d1, &d2));
    assert_eq!(c1, a1, "the greedy lane is unchanged by its neighbour's sampling");
    let with_lp = |mut a: LaneAdmit| {
        a.want_logprobs = 3;
        a
    };
    let (e1, e2) = run(true, with_lp);
    let (f1, f2) = run(false, with_lp);
    assert_eq!((&e1, &e2), (&f1, &f2));
    assert_eq!(e2, a2);
}

unsafe extern "C" fn collect(user: *mut c_void, lane: u64, tokens: *const u32, n: u32) {
    // SAFETY: the test passes a live Vec behind `user`; the engine hands a
    // live token slice for the duration of the call.
    let v = unsafe { &mut *(user as *mut Vec<(u64, u32)>) };
    for i in 0..n as usize {
        // SAFETY: i < n, the slice the engine passed.
        v.push((lane, unsafe { *tokens.add(i) }));
    }
}

#[test]
fn partial_emits_mirror_the_terminal_emit() {
    let mut h = harness();
    let mut seen: Vec<(u64, u32)> = Vec::new();
    let p = h.prompt(&[1, 2]);
    let pb = h
        .plan()
        .flags(tick_flags::PARTIAL_EMITS)
        .partial_emit(collect, &mut seen as *mut _ as *mut c_void)
        .admit(3, p)
        .prefill(3, 0, 2)
        .decode(3, 5);
    let ev = h.tick_ok(pb);
    let terminal = h.rings_tokens(&ev.emit_for(3).token_ref);
    assert_eq!(seen.iter().map(|(_, t)| *t).collect::<Vec<_>>(), terminal);
    assert!(seen.iter().all(|(l, _)| *l == 3));
}

#[test]
fn logprobs_ride_the_events_for_lanes_that_asked() {
    let mut h = harness();
    let p = h.prompt(&[1, 2]);
    let (plan, arena) = h
        .plan()
        .admit_with(
            |mut a| {
                a.want_logprobs = 1;
                a
            },
            1,
            p,
        )
        .prefill(1, 0, 2)
        .decode(1, 3)
        .build();
    let ev = h.engine.tick(&plan, &arena, &mut h.rings).expect("tick");
    let bounds = ArenaBounds { base: 0, len: usize::MAX };
    // SAFETY: engine-owned arrays, valid until the next tick.
    let lps: Vec<LaneLogprob> = unsafe { array::read_array(&ev.logprobs, &bounds) }.unwrap().collect();
    assert_eq!(lps.len(), 3);
    for lp in &lps {
        assert_eq!(lp.lane_tag, 1);
        assert!(lp.logprob <= 0.0);
        assert!(lp.n_top as usize <= MAX_TOP_LOGPROBS);
        assert!(lp.top_logprobs[0] >= lp.logprob - 1e-6, "the chosen token is greedy here");
    }
}

#[test]
fn logit_bias_steers_the_greedy_choice() {
    let mut h = harness();
    let want = h.engine.primitives().token_for(&[1, 2]);
    let other = if want == 7 { 8 } else { 7 };
    let handle = h.engine.logit_bias_create(&[other as i32], &[100.0]).unwrap();
    let p = h.prompt(&[1, 2]);
    let ev = h.tick_ok(
        h.plan()
            .admit_with(
                |mut a| {
                    a.logit_bias_handle = handle;
                    a
                },
                1,
                p,
            )
            .prefill(1, 0, 2)
            .decode(1, 1),
    );
    assert_eq!(h.rings_tokens(&ev.emit_for(1).token_ref), vec![other]);
    assert!(h.engine.logit_bias_free(handle).is_ok());
    assert_eq!(h.engine.logit_bias_free(handle), Err(Status::UnknownHandle));
    let p2 = h.prompt(&[1]);
    let err = h.tick_err(h.plan().admit_with(
        |mut a| {
            a.logit_bias_handle = 99;
            a
        },
        2,
        p2,
    ));
    assert_eq!(err, Status::RejectIllegalCombination);
}

#[test]
fn repeat_penalty_changes_a_repeated_greedy_token() {
    for (penalty, takes_it) in [(2.0, false), (1.0, true)] {
        let mut h = harness();
        let repeated: u32 = 77;
        let prompt = vec![repeated, 5, 6];
        let best = h.engine.primitives().token_for(&prompt);
        assert_ne!(best, repeated);
        let bias = h.engine.logit_bias_create(&[repeated as i32], &[18.0]).unwrap();
        let p = h.prompt(&prompt);
        let ev = h.tick_ok(
            h.plan()
                .admit_with(
                    |mut a| {
                        a.logit_bias_handle = bias;
                        a.params.repeat_penalty = penalty;
                        a
                    },
                    1,
                    p,
                )
                .prefill(1, 0, 3)
                .decode(1, 1),
        );
        let want = if takes_it { repeated } else { best };
        assert_eq!(h.rings_tokens(&ev.emit_for(1).token_ref), vec![want], "repeat penalty {penalty}");
    }
}

#[test]
fn needs_replan_when_the_pool_cannot_serve_and_eviction_frees_it() {
    let cfg = FakeConfig { cells_total: 100, ..Default::default() };
    let mut h = with(cfg);
    let tokens: Vec<u32> = (0..200).collect();
    let p = h.prompt(&tokens);
    let err = h.tick_err(h.plan().admit(1, p).prefill(1, 0, 200));
    assert_eq!(err, Status::NeedsReplan);
    assert!(h.engine.lane_sequence(1).is_none());

    let t1: Vec<u32> = (0..64).collect();
    let p1 = h.prompt(&t1);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 64));
    h.tick_ok(h.plan().retire(1, true));
    let t2: Vec<u32> = (1000..1090).collect();
    let p2 = h.prompt(&t2);
    let ev = h.tick_ok(h.plan().admit(2, p2).prefill(2, 0, 90));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(ev.tick_status, tick_status::SHED, "an eviction is a reported shed");
    let t3: Vec<u32> = (2000..2050).collect();
    let p3 = h.prompt(&t3);
    let err = h.tick_err(h.plan().evict_envelope(u64::MAX, 0).admit(3, p3).prefill(3, 0, 50));
    assert_eq!(err, Status::NeedsReplan);
}

#[test]
fn length_finish_at_the_runtime_ceiling() {
    let cfg = FakeConfig { max_seq_len: 6, ..Default::default() };
    let mut h = with(cfg);
    let p = h.prompt(&[1, 2, 3]);
    let ev = h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 3).decode(1, 10));
    let e = ev.emit_for(1);
    assert_eq!(e.n_tokens, 3, "3 prompt + 3 generated reaches the ceiling of 6");
    assert_eq!(e.finish, finish::LENGTH);
    let p2 = h.prompt(&[1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(h.tick_err(h.plan().admit(2, p2)), Status::RejectBudget);
}

#[test]
fn eos_finishes_unless_ignored() {
    let eos = FakeConfig::default().eos;
    let mut h = harness();
    let p = h.prompt(&[1, 2]);
    h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.params.flags = 0;
            a
        },
        1,
        p,
    ).prefill(1, 0, 2));
    let seq = h.engine.lane_sequence(1).unwrap();
    h.engine.primitives_mut().script(seq, vec![eos, 5]);
    let ev = h.tick_ok(h.plan().decode(1, 4));
    let e = ev.emit_for(1);
    assert_eq!(e.n_tokens, 1);
    assert_eq!(e.finish, finish::EOS);

    let mut h = harness();
    let p = h.prompt(&[1, 2]);
    h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.params.flags = sampling_flag::IGNORE_EOS;
            a
        },
        1,
        p,
    ).prefill(1, 0, 2));
    let seq = h.engine.lane_sequence(1).unwrap();
    h.engine.primitives_mut().script(seq, vec![eos, 5]);
    let ev = h.tick_ok(h.plan().decode(1, 4));
    let e = ev.emit_for(1);
    assert_eq!(e.n_tokens, 4, "ignore_eos decodes to the budget");
    assert_eq!(e.finish, finish::NONE);
}

#[test]
fn recurrent_runtime_seeds_only_at_exact_entry_length() {
    let cfg = FakeConfig { truncate_partial: false, page: 1, ..Default::default() };
    let mut h = with(cfg);
    let t1: Vec<u32> = (0..40).collect();
    let p1 = h.prompt(&t1);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 40));
    h.tick_ok(h.plan().retire(1, true));
    let mut t2 = t1.clone();
    t2.extend([100, 101]);
    assert!(h.engine.seed_acquire(&t2, 40, determinism::BEST_EFFORT).is_ok());
    assert_eq!(h.engine.seed_acquire(&t2, 32, determinism::BEST_EFFORT), Err(Status::SeedUnservable));
}

#[test]
fn a_retire_in_the_same_plan_credits_its_cells_to_the_admit() {
    let mut h = with(FakeConfig { cells_total: 100, ..Default::default() });
    let t1: Vec<u32> = (0..80).collect();
    let p1 = h.prompt(&t1);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 80));
    let t2: Vec<u32> = (1000..1060).collect();
    let p2 = h.prompt(&t2);
    let ev = h.tick_ok(h.plan().retire(1, false).admit(2, p2).prefill(2, 0, 60));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(ev.tick_status, tick_status::OK, "no eviction was needed");
    let mut h = with(FakeConfig { cells_total: 100, ..Default::default() });
    let p1 = h.prompt(&t1);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 80));
    let p2 = h.prompt(&t2);
    let err = h.tick_err(h.plan().retire(1, true).admit(2, p2).prefill(2, 0, 60));
    assert_eq!(err, Status::NeedsReplan);
    h.tick_ok(h.plan().retire(1, true));
    let p2 = h.prompt(&t2);
    let ev = h.tick_ok(h.plan().admit(2, p2).prefill(2, 0, 60));
    assert_eq!(ev.tick_status, tick_status::SHED, "fits by evicting the published entry");
}

#[test]
fn a_seeded_copy_is_charged_to_a_bounded_pool() {
    let mut h = with(FakeConfig { cells_total: 100, ..Default::default() });
    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
    h.tick_ok(h.plan().retire(1, true));
    let mut longer = tokens.clone();
    longer.extend(64..94);
    let handle = h.engine.seed_acquire(&longer, 64, determinism::BEST_EFFORT).unwrap();
    let other = h.engine.seed_acquire(&longer, 64, determinism::BEST_EFFORT).unwrap();
    let p2 = h.prompt(&longer);
    let err = h.tick_err(
        h.plan().admit_with(|mut a| { a.seed_handle = handle; a }, 2, p2).prefill(2, 64, 30),
    );
    assert_eq!(err, Status::NeedsReplan);
    assert!(h.engine.lane_sequence(2).is_none(), "no state change on refusal");
    assert_eq!(h.engine.cache_bytes(), 64 * 64, "the entry is still cached");
    h.engine.seed_release(other).unwrap();
    let p2 = h.prompt(&longer);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = handle; a }, 2, p2).prefill(2, 64, 30));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!((cells_used(&h), h.engine.cache_bytes()), (93, 0));
}

#[test]
fn a_runtime_fault_in_a_packed_step_stops_its_lanes_not_the_tick() {
    let mut h = harness();
    let p1 = h.prompt(&[1, 2, 3]);
    let p2 = h.prompt(&[4, 5, 6]);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 3));
    h.tick_ok(h.plan().admit(2, p2).prefill(2, 0, 3));
    h.engine.primitives_mut().fault_next_step(status::fault_code::GRAMMAR_OVERFLOW);
    let ev = h.tick_ok(h.plan().decode(1, 2).decode(2, 2));
    assert_eq!(ev.tick_status, tick_status::LANE_ERRORS);
    let mut tags: Vec<u64> = ev.faults.iter().map(|f| f.lane_tag).collect();
    tags.sort_unstable();
    assert_eq!(tags, vec![1, 2], "every lane of the faulting step stops");
    assert!(ev.faults.iter().all(|f| f.code == status::fault_code::GRAMMAR_OVERFLOW));
    assert_eq!(h.engine.lane_sequence(1), None);
    let err = h.tick_err(h.plan().decode(1, 1));
    assert_eq!(err, Status::RejectUnknownLane);
    let p3 = h.prompt(&[7, 8]);
    let ev = h.tick_ok(h.plan().retire(1, true).retire(2, true).admit(3, p3).prefill(3, 0, 2).decode(3, 1));
    assert_eq!(ev.emit_for(3).n_tokens, 1);
    assert_eq!(h.tick_err(h.plan().retire(1, true)), Status::RejectUnknownLane, "retired once");
    let p4 = h.prompt(&[9, 10]);
    h.engine.primitives_mut().fault_next_step(status::fault_code::GRAMMAR_OVERFLOW);
    let ev = h.tick_ok(h.plan().admit(4, p4).prefill(4, 0, 2).decode(4, 1));
    assert_eq!(ev.tick_status, tick_status::LANE_ERRORS);
    assert_eq!(ev.faults[0].lane_tag, 4);
    assert!(ev.emits.iter().all(|e| e.lane_tag != 4));
}

#[test]
fn eviction_never_takes_an_entry_past_the_byte_cap() {
    let mut h = with(FakeConfig { cells_total: 96, ..Default::default() });
    let a: Vec<u32> = (0..64).collect();
    let pa = h.prompt(&a);
    h.tick_ok(h.plan().admit(1, pa).prefill(1, 0, 64));
    h.tick_ok(h.plan().retire(1, true));
    let b: Vec<u32> = (500..516).collect();
    let pb = h.prompt(&b);
    h.tick_ok(h.plan().admit(2, pb).prefill(2, 0, 16));
    h.tick_ok(h.plan().retire(2, true));
    let c: Vec<u32> = (900..940).collect();
    let pc = h.prompt(&c);
    let err = h.tick_err(h.plan().evict_envelope(u64::MAX, 2048).admit(3, pc).prefill(3, 0, 40));
    assert_eq!(err, Status::NeedsReplan);
    assert_eq!(h.engine.cache_bytes(), 5120, "nothing was evicted on refusal");
    assert_eq!(h.engine.cache_evict(1, 0, 2048, 1536).unwrap(), 1024);
    assert_eq!(h.engine.cache_bytes(), 4096);
}

#[test]
fn eviction_honours_the_cache_class_mask() {
    let mut h = with(FakeConfig { cells_total: 100, ..Default::default() });
    let a: Vec<u32> = (0..64).collect();
    let pa = h.prompt(&a);
    h.tick_ok(h.plan().admit(1, pa).prefill(1, 0, 64));
    h.tick_ok(h.plan().retire(1, true));
    assert_eq!(h.engine.cache_evict(0, 0, u64::MAX, u64::MAX).unwrap(), 0, "class excluded");
    assert_eq!(h.engine.cache_bytes(), 4096);
    let c: Vec<u32> = (900..950).collect();
    let pc = h.prompt(&c);
    let err = h.tick_err(h.plan().evict_envelope(0, u64::MAX).admit(3, pc).prefill(3, 0, 50));
    assert_eq!(err, Status::NeedsReplan);
    assert_eq!(h.engine.cache_evict(superfluid_executor::cache::RESIDENT_PREFIX_CLASS, 0, u64::MAX, u64::MAX).unwrap(), 4096);
}

#[test]
fn enumerated_eviction_refuses_a_foreign_chain_key() {
    let mut h = harness();
    let a: Vec<u32> = (0..64).collect();
    let pa = h.prompt(&a);
    h.tick_ok(h.plan().admit(1, pa).prefill(1, 0, 64));
    h.tick_ok(h.plan().retire(1, true));
    let digest = superfluid_fingerprint::content_digest(&a);
    let mine = h.engine.compat_identity();
    let foreign = [0xEE; 32];
    assert_eq!(h.engine.cache_evict_entries(&[(mine, digest), (foreign, digest)]), Err(Status::IdentityMismatch));
    assert_eq!(h.engine.cache_bytes(), 4096, "refused before any eviction");
    h.engine.cache_evict_entries(&[(mine, [0x11; 32])]).unwrap();
    assert_eq!(h.engine.cache_bytes(), 4096, "an absent digest is idempotent");
    h.engine.cache_evict_entries(&[(mine, digest)]).unwrap();
    assert_eq!(h.engine.cache_bytes(), 0);
}

#[test]
fn logprobs_carry_the_requested_number_of_alternatives() {
    for (want, expect_top) in [(1u8, 0u32), (4, 3), (25, 20)] {
        let mut h = harness();
        let p = h.prompt(&[1, 2, 3]);
        let ev = h.tick_ok(
            h.plan().admit_with(|mut a| { a.want_logprobs = want; a }, 1, p).prefill(1, 0, 3).decode(1, 1),
        );
        let rec = h.engine.last_logprobs();
        assert_eq!(rec.len(), 1, "want {want}");
        assert_eq!(rec[0].n_top, expect_top, "want {want}");
        assert_eq!(ev.emit_for(1).n_tokens, 1);
    }
}

#[test]
fn logit_bias_applies_before_the_repeat_penalty() {
    let mut h = harness();
    let repeated: u32 = 77;
    let prompt = vec![repeated, 5, 6];
    let best = h.engine.primitives().token_for(&prompt);
    assert_ne!(best, repeated);
    let bias = h.engine.logit_bias_create(&[repeated as i32], &[18.0]).unwrap();
    let p = h.prompt(&prompt);
    let ev = h.tick_ok(
        h.plan()
            .admit_with(
                |mut a| {
                    a.logit_bias_handle = bias;
                    a.params.repeat_penalty = 2.0;
                    a
                },
                1,
                p,
            )
            .prefill(1, 0, 3)
            .decode(1, 1),
    );
    assert_eq!(h.rings_tokens(&ev.emit_for(1).token_ref), vec![best]);
}

#[test]
fn a_published_sequence_is_cut_to_its_aligned_key() {
    let mut h = harness();
    let tokens: Vec<u32> = (0..40).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 40));
    let seq = h.engine.lane_sequence(1).unwrap();
    h.tick_ok(h.plan().retire(1, true));
    assert_eq!(h.engine.primitives().contents(seq).map(|c| c.len()), Some(32));
    assert_eq!(h.engine.cache_bytes(), 32 * 64);
    let bare = h.engine.create_sequence().unwrap();
    let restored: Vec<u32> = (100..137).collect();
    h.engine
        .primitives_mut()
        .step(&[superfluid_executor::Feed { seq: bare, input: superfluid_executor::Input::Tokens(&restored), wants_row: false }])
        .unwrap();
    h.engine.publish_sequence(bare, &restored).unwrap();
    assert_eq!(h.engine.primitives().contents(bare).map(|c| c.len()), Some(32));
}

#[test]
fn a_recurrent_prompt_is_checkpointed_one_token_short_of_its_end_and_decodes_the_same() {
    let tokens: Vec<u32> = (0..200).map(|i| i % 97).collect();
    let run = |truncate_partial: bool| {
        let mut h = with(FakeConfig { truncate_partial, page: 1, ..Default::default() });
        let p = h.prompt(&tokens);
        let ev = h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 200).decode(1, 8));
        let out = h.rings_tokens(&ev.emit_for(1).token_ref);
        (h, out)
    };
    let (plain, plain_out) = run(true);
    let (h, out) = run(false);
    assert_eq!(out, plain_out, "holding the last prompt token back changes nothing decoded");
    assert_eq!(out.len(), 8);
    assert_eq!(h.engine.match_lengths(&tokens), vec![199], "the same prompt again starts from the checkpoint");
    let mut next = tokens.clone();
    next.extend([500, 501, 502]);
    assert!(h.engine.match_lengths(&next).contains(&199), "so does a longer one that extends it");
    assert!(plain.engine.match_lengths(&tokens).is_empty(), "a cuttable runtime publishes on retire as before");
    let short: Vec<u32> = (0..100).collect();
    let mut h = with(FakeConfig { truncate_partial: false, page: 1, ..Default::default() });
    let p = h.prompt(&short);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 100).decode(1, 1));
    assert!(h.engine.match_lengths(&short).is_empty(), "a short prompt is not checkpointed");
}

#[test]
fn a_tick_asked_to_yield_stops_after_a_prefill_step_and_the_rest_runs_next_tick() {
    let tokens: Vec<u32> = (0..2048).map(|i| i % 300 + 10).collect();
    let cfg = FakeConfig { prefill_step_tokens: 512, ..Default::default() };
    let mut plain = with(cfg.clone());
    let p = plain.prompt(&tokens);
    let ev = plain.tick_ok(plain.plan().admit(1, p).prefill(1, 0, 2048).decode(1, 8));
    let want = plain.rings_tokens(&ev.emit_for(1).token_ref);

    let mut h = with(cfg);
    let p = h.prompt(&tokens);
    h.rings.set_yield(true);
    let ev = h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 2048).decode(1, 8));
    let shed: u64 = ev.shed_entries.iter().filter(|e| e.kind == shed_kind::PREFILL_CHUNK).map(|e| e.tokens).sum();
    assert_eq!(shed, 1536, "one 512-token step ran; the rest goes back to the scheduler");
    assert_eq!(ev.emit_for(1).n_tokens, 0, "a lane whose prompt was cut does not decode");
    assert_eq!(h.engine.lane_committed(1).unwrap().len(), 512);
    h.rings.set_yield(false);
    let ev = h.tick_ok(h.plan().prefill(1, 512, 1536).decode(1, 8));
    assert_eq!(h.rings_tokens(&ev.emit_for(1).token_ref), want, "the same reply as a tick that ran whole");
}

#[test]
fn a_tick_asked_to_yield_stops_between_decode_rounds() {
    let tokens: Vec<u32> = (0..40).collect();
    let run = |yield_at: Option<usize>| {
        let mut h = with(FakeConfig::default());
        let p = h.prompt(&tokens);
        let mut out = Vec::new();
        let mut plan = h.plan().admit(1, p).prefill(1, 0, 40).decode(1, 16);
        for tick in 0..4 {
            h.rings.set_yield(yield_at == Some(tick));
            let ev = h.tick_ok(plan);
            let got = h.rings_tokens(&ev.emit_for(1).token_ref);
            if yield_at == Some(tick) {
                assert_eq!(got.len(), 1, "one round, then the tick ends");
            }
            out.extend(got);
            plan = h.plan().decode(1, 16);
        }
        out
    };
    let whole = run(None);
    let cut = run(Some(1));
    assert_eq!(cut[..], whole[..cut.len()], "a yielded tick changes nothing decoded, only when");
    assert_eq!(cut.len(), whole.len() - 15);
}

#[test]
fn while_latency_matters_a_prefill_step_is_a_short_one() {
    // 500 ms a 512-token step: once a tick has measured that, a step in
    // latency mode is what 0.3 s holds in whole 128-token units (256 at the
    // delay alone, 128 where a loaded machine adds to it), and a tick asked to
    // yield stops after it.
    let shed_after = |latency: bool| {
        let mut h = with(FakeConfig {
            prefill_step_tokens: 512,
            step_delay: std::time::Duration::from_millis(500),
            ..Default::default()
        });
        let warm: Vec<u32> = (0..512).map(|i| i % 300 + 10).collect();
        let p = h.prompt(&warm);
        h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 512));
        let long: Vec<u32> = (0..1024).map(|i| i % 300 + 10).collect();
        let p = h.prompt(&long);
        h.rings.set_latency(latency);
        h.rings.set_yield(true);
        let ev = h.tick_ok(h.plan().admit(2, p).prefill(2, 0, 1024));
        ev.shed_entries.iter().filter(|e| e.kind == shed_kind::PREFILL_CHUNK).map(|e| e.tokens).sum::<u64>()
    };
    assert_eq!(shed_after(false), 512, "the runtime's whole micro-batch");
    let step = 1024 - shed_after(true);
    assert!(step < 512 && step % 128 == 0 && step > 0, "a shorter step in whole 128-token units: {step}");
}

#[test]
fn a_prefill_step_is_waited_out_only_where_it_is_long() {
    // Steps of 512 tokens: until a tick has measured how long one takes, and
    // when one takes 600 ms, each is waited out; at 10 ms one is not, unless
    // latency-sensitive requests are around.
    let synced = |delay_ms: u64, latency: bool| {
        let mut h = with(FakeConfig {
            prefill_step_tokens: 512,
            step_delay: std::time::Duration::from_millis(delay_ms),
            ..Default::default()
        });
        let warm: Vec<u32> = (0..512).map(|i| i % 300 + 10).collect();
        let p = h.prompt(&warm);
        h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 512));
        let first = h.engine.primitives().synced;
        let next: Vec<u32> = (0..512).map(|i| i % 300 + 20).collect();
        let p = h.prompt(&next);
        h.rings.set_latency(latency);
        h.tick_ok(h.plan().admit(2, p).prefill(2, 0, 512));
        (first, h.engine.primitives().synced)
    };
    assert_eq!(synced(600, false), (Some(true), Some(true)));
    assert_eq!(synced(10, false), (Some(true), Some(false)), "a short step is queued behind the last");
    assert_eq!(synced(10, true), (Some(true), Some(true)), "while a chat may come, every step is waited out");
}

#[test]
fn exact_only_candidates_are_servable() {
    let mut h = with(FakeConfig { truncate_partial: false, page: 1, ..Default::default() });
    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
    h.tick_ok(h.plan().retire(1, true));
    assert_eq!(h.engine.match_lengths(&tokens), Vec::<u64>::new(), "the whole span is unservable");
    let mut longer = tokens.clone();
    longer.push(64);
    assert_eq!(h.engine.match_lengths(&longer), vec![64]);
    assert!(h.engine.seed_acquire(&longer, 64, determinism::BEST_EFFORT).is_ok());
}

#[test]
fn a_host_lane_finishes_with_length_at_the_context_limit() {
    let mut h = with(FakeConfig { max_seq_len: 6, ..Default::default() });
    let p = h.prompt(&[1, 2, 3, 4]);
    let ev = h.tick_ok(
        h.plan().admit_with(|mut a| { a.sampling = sampling::HOST; a }, 1, p).prefill(1, 0, 4).decode(1, 1),
    );
    let nonce = ev.emit_for(1).logits_row.generation;
    assert_ne!(nonce, 0, "a row at length 4");
    let ev = h.tick_ok(h.plan().commit(1, 7, nonce).decode(1, 1));
    let nonce = ev.emit_for(1).logits_row.generation;
    assert_ne!(nonce, 0, "a row at length 5");
    let ev = h.tick_ok(h.plan().commit(1, 8, nonce).decode(1, 1));
    let e = ev.emit_for(1);
    assert_eq!(e.finish, finish::LENGTH);
    assert_eq!(e.logits_row.generation, 0, "no row past the limit");
    assert_eq!(h.engine.lane_committed(1).unwrap().len(), 6);
    assert_eq!(h.tick_err(h.plan().decode(1, 1)), Status::RejectIllegalCombination);
    assert_eq!(h.tick_err(h.plan().commit(1, 9, nonce)), Status::RejectIllegalCombination);
}

#[test]
fn decode_reservations_are_clamped_to_the_remaining_context() {
    let mut h = with(FakeConfig { cells_total: 100, max_seq_len: 100, ..Default::default() });
    let prompt: Vec<u32> = (0..99).collect();
    let p = h.prompt(&prompt);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 99));
    let ev = h.tick_ok(h.plan().decode(1, 10));
    let e = ev.emit_for(1);
    assert_eq!(e.n_tokens, 1);
    assert_eq!(e.finish, finish::LENGTH);
}

#[test]
fn evictions_are_listed_in_the_shed_report() {
    let mut h = with(FakeConfig { cells_total: 100, ..Default::default() });
    let a: Vec<u32> = (0..64).collect();
    let pa = h.prompt(&a);
    h.tick_ok(h.plan().admit(1, pa).prefill(1, 0, 64));
    h.tick_ok(h.plan().retire(1, true));
    let c: Vec<u32> = (900..990).collect();
    let pc = h.prompt(&c);
    let ev = h.tick_ok(h.plan().admit(2, pc).prefill(2, 0, 90));
    assert_eq!(ev.tick_status, tick_status::SHED);
    let evictions: Vec<&ShedEntry> = ev.shed_entries.iter().filter(|e| e.kind == shed_kind::CACHE_EVICTION).collect();
    assert_eq!(evictions.len(), 1);
    assert_eq!(evictions[0].bytes, 64 * 64);
    assert_eq!(evictions[0].tokens, 64);
}

#[test]
fn an_empty_prompt_is_refused_typed() {
    let mut h = harness();
    let p = h.prompt(&[]);
    assert_eq!(h.tick_err(h.plan().admit(1, p)), Status::RejectIllegalCombination);
    assert!(h.engine.lane_sequence(1).is_none());
}

fn full_of_two_entries() -> Harness<Executor<FakePrimitives>> {
    let mut h = with(FakeConfig { cells_total: 8, max_seqs: 2, kv_bytes_per_token: 1, page: 1, ..Default::default() });
    for (lane, base) in [(1u64, 0u32), (2, 100)] {
        let t: Vec<u32> = (base..base + 4).collect();
        let p = h.prompt(&t);
        h.tick_ok(h.plan().admit(lane, p).prefill(lane, 0, 4));
        h.tick_ok(h.plan().retire(lane, true));
    }
    assert_eq!(h.engine.cache_bytes(), 8);
    h
}

#[test]
fn cell_and_slot_pressure_share_one_eviction_envelope() {
    let mut h = full_of_two_entries();
    let (p3, p4) = (h.prompt(&[200, 201]), h.prompt(&[300, 301]));
    let err = h.tick_err(
        h.plan().evict_envelope(u64::MAX, 4).admit(3, p3).admit(4, p4).prefill(3, 0, 2).prefill(4, 0, 2),
    );
    assert_eq!(err, Status::NeedsReplan);
    assert_eq!(h.engine.cache_bytes(), 8, "a refused plan evicts nothing");
    let (p3, p4) = (h.prompt(&[200, 201]), h.prompt(&[300, 301]));
    let ev = h.tick_ok(
        h.plan().evict_envelope(u64::MAX, 8).admit(3, p3).admit(4, p4).prefill(3, 0, 2).prefill(4, 0, 2),
    );
    assert_eq!(ev.admit_for(3).status, admit_status::ADMITTED);
    assert_eq!(ev.admit_for(4).status, admit_status::ADMITTED);
    assert_eq!(h.engine.cache_bytes(), 0);
    assert_eq!(ev.shed_entries.iter().filter(|e| e.kind == shed_kind::CACHE_EVICTION).count(), 2);
}

#[test]
fn a_plan_refused_for_slots_evicts_nothing_for_its_cells() {
    let mut h = full_of_two_entries();
    let lease = h.engine.seed_acquire(&[0, 1, 2, 3, 9], 4, determinism::BEST_EFFORT).expect("lease on the first entry");
    let (p3, p4) = (h.prompt(&[200, 201]), h.prompt(&[300, 301]));
    let err = h.tick_err(h.plan().admit(3, p3).admit(4, p4).prefill(3, 0, 2).prefill(4, 0, 2));
    assert_eq!(err, Status::NeedsReplan);
    assert_eq!(h.engine.cache_bytes(), 8, "a refused plan evicts nothing");
    assert_eq!(h.engine.match_lengths(&[100, 101, 102, 103, 9]), vec![4], "the unpinned entry is still served");
    h.engine.seed_release(lease).unwrap();
}

#[test]
fn a_lane_at_the_limit_commits_no_token_however_it_samples() {
    let greedy = |a: LaneAdmit| a;
    let gumbel = |a: LaneAdmit| sampled(a, 4);
    let host = |mut a: LaneAdmit| {
        a.sampling = sampling::HOST;
        a
    };
    type Admit = fn(LaneAdmit) -> LaneAdmit;
    let modes: [(&str, bool, Admit); 4] = [
        ("greedy, the runtime's pick", true, greedy),
        ("greedy, a host row", false, greedy),
        ("sampled", true, gumbel),
        ("HOST", true, host),
    ];
    for (name, engine_sampling, admit) in modes {
        let mut h = with(FakeConfig { max_seq_len: 4, engine_sampling, ..Default::default() });
        let p = h.prompt(&[1, 2, 3, 4]);
        let ev = h.tick_ok(h.plan().admit_with(admit, 1, p).prefill(1, 0, 4).decode(1, 1));
        let e = ev.emit_for(1);
        assert_eq!(e.n_tokens, 0, "{name}: no token past the limit");
        assert_eq!(e.logits_row.generation, 0, "{name}: no row past the limit");
        assert_eq!(e.finish, finish::LENGTH, "{name}");
        assert_eq!(h.engine.lane_committed(1).unwrap().len(), 4, "{name}");
    }
}

fn cells_used(h: &Harness<Executor<FakePrimitives>>) -> u64 {
    h.engine.primitives().mem_counters().cells_used
}

#[test]
fn a_publishing_retire_reserves_the_token_it_ingests() {
    let mut h = with(FakeConfig { cells_total: 8, page: 1, kv_bytes_per_token: 1, ..Default::default() });
    let p1 = h.prompt(&[1, 2, 3]);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 3).decode(1, 1));
    assert_eq!(cells_used(&h), 3, "the prompt is ingested; the generated token is pending");
    let prompt: Vec<u32> = (100..106).collect();
    let p2 = h.prompt(&prompt);
    let err = h.tick_err(h.plan().retire(1, true).admit(2, p2).prefill(2, 0, 5));
    assert_eq!(err, Status::NeedsReplan);
    assert_eq!(cells_used(&h), 3, "a refused plan moves nothing");
    assert!(h.engine.lane_sequence(1).is_some(), "the lane is still live");
    let p2 = h.prompt(&prompt);
    h.tick_ok(h.plan().retire(1, true).admit(2, p2).prefill(2, 0, 4));
    assert_eq!(cells_used(&h), 8);
}

#[test]
fn a_shared_prefix_frees_nothing_while_another_sequence_holds_it() {
    let cfg = FakeConfig { cells_total: 8, page: 1, kv_bytes_per_token: 1, copy_shares_cells: true, ..Default::default() };
    let mut h = with(cfg);
    let prefix: Vec<u32> = vec![10, 11, 12, 13];
    let p1 = h.prompt(&prefix);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 4));
    h.tick_ok(h.plan().retire(1, true));
    let mut seeded = prefix.clone();
    seeded.push(20);
    let lease = h.engine.seed_acquire(&seeded, 4, determinism::BEST_EFFORT).expect("a lease on the cached prefix");
    let p2 = h.prompt(&seeded);
    h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p2).prefill(2, 4, 1));
    assert_eq!(cells_used(&h), 4, "the lane shares the cached prefix's cells");
    let wants_five = |h: &mut Harness<Executor<FakePrimitives>>| h.prompt(&(200..206).collect::<Vec<u32>>());

    let p3 = wants_five(&mut h);
    let err = h.tick_err(h.plan().evict_envelope(u64::MAX, 0).retire(2, false).admit(3, p3).prefill(3, 0, 5));
    assert_eq!(err, Status::NeedsReplan);
    let p3 = wants_five(&mut h);
    let err = h.tick_err(h.plan().admit(3, p3).prefill(3, 0, 5));
    assert_eq!(err, Status::NeedsReplan);
    assert_eq!((cells_used(&h), h.engine.cache_bytes()), (4, 4), "the refusals moved nothing");
    assert!(h.engine.lane_sequence(2).is_some());

    let p3 = wants_five(&mut h);
    let ev = h.tick_ok(h.plan().retire(2, false).admit(3, p3).prefill(3, 0, 5));
    assert_eq!(ev.admit_for(3).status, admit_status::ADMITTED);
    assert_eq!(h.engine.cache_bytes(), 0, "the prefix was evicted to make the room");
    assert_eq!(cells_used(&h), 5);
}

#[test]
fn a_broken_runtime_ends_a_worker_process() {
    let broken_tick = || {
        let mut h = harness();
        let p = h.prompt(&[1, 2, 3]);
        h.engine.primitives_mut().fail_next_step(PrimError::Fatal);
        h.tick_err(h.plan().admit(1, p).prefill(1, 0, 3).decode(1, 1))
    };
    if std::env::var_os("SUPERFLUID_TEST_PLAYS_WORKER").is_some() {
        superfluid_executor::exit_on_fatal(true);
        broken_tick();
        std::process::exit(1);
    }
    assert_eq!(broken_tick(), Status::Fatal, "not a worker process: the tick answers");
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "a_broken_runtime_ends_a_worker_process", "--nocapture", "--test-threads=1"])
        .env("SUPERFLUID_TEST_PLAYS_WORKER", "1")
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(70), "{said}");
    assert!(said.contains("this worker exits so that the daemon starts a clean one"), "{said}");
}

#[test]
fn a_lease_says_how_long_it_lives() {
    let h = harness();
    assert_eq!(h.engine.seed_lease_ticks(), Some(ExecutorConfig::default().seed_ttl_ticks));
}

#[test]
fn a_sequence_created_outside_a_tick_takes_a_cache_entrys_slot() {
    let mut h = with(FakeConfig { max_seqs: 2, page: 1, kv_bytes_per_token: 1, ..Default::default() });
    for (lane, first) in [(1u64, 10u32), (2, 20)] {
        let p = h.prompt(&[first, first + 1, first + 2]);
        h.tick_ok(h.plan().admit(lane, p).prefill(lane, 0, 3));
        h.tick_ok(h.plan().retire(lane, true));
    }
    assert_eq!(h.engine.cache_bytes(), 6, "two entries hold the two slots");
    let lease = h.engine.seed_acquire(&[20, 21, 22, 99], 3, determinism::BEST_EFFORT).expect("a lease on the newer entry");
    let fresh = h.engine.create_sequence().expect("the older entry gives up its slot");
    assert_eq!(h.engine.cache_bytes(), 3, "one entry was evicted");
    assert!(h.engine.seed_acquire(&[10, 11, 12, 99], 3, determinism::BEST_EFFORT).is_err(), "the least recently used one");
    assert_eq!(h.engine.create_sequence(), Err(Status::NeedsReplan));
    h.engine.free_sequence(fresh).unwrap();
    h.engine.seed_release(lease).unwrap();
}

#[test]
fn a_seed_with_no_room_for_a_copy_takes_its_entry_over() {
    let cfg = || FakeConfig { cells_total: 8, page: 1, kv_bytes_per_token: 1, ..Default::default() };
    let turn2: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 20];
    let mut cold = with(cfg());
    let p = cold.prompt(&turn2);
    let ev = cold.tick_ok(cold.plan().admit(1, p).prefill(1, 0, 7).decode(1, 1));
    let want = cold.rings_tokens(&ev.emit_for(1).token_ref);

    let mut h = with(cfg());
    let p1 = h.prompt(&turn2[..6]);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 6));
    h.tick_ok(h.plan().retire(1, true));
    assert_eq!((cells_used(&h), h.engine.cache_bytes()), (6, 6), "the first turn is a cache entry");
    let lease = h.engine.seed_acquire(&turn2, 6, determinism::BEST_EFFORT).expect("a lease on the cached turn");
    let p2 = h.prompt(&turn2);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p2).prefill(2, 6, 1).decode(2, 1));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(h.rings_tokens(&ev.emit_for(2).token_ref), want, "the lane continues as one that prefilled the prompt cold");
    assert_eq!(h.engine.cache_bytes(), 0, "the entry is the lane's sequence now");
    assert_eq!(cells_used(&h), 7, "six seeded cells and the new token, and no copy");
    let left: Vec<(u32, u64, u64)> = ev.shed_entries.iter().map(|e| (e.kind, e.bytes, e.tokens)).collect();
    assert_eq!(left, vec![(shed_kind::CACHE_EVICTION, 6, 6)], "the report says an entry left the cache");
}

#[test]
fn a_runtime_that_prefers_it_takes_the_entry_over_with_room_to_spare() {
    let cfg = |prefer| FakeConfig { cells_total: 64, page: 1, kv_bytes_per_token: 1, takeover_preferred: prefer, ..Default::default() };
    let turn2: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 20];
    let run = |prefer: bool| {
        let mut h = with(cfg(prefer));
        let p1 = h.prompt(&turn2[..6]);
        h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 6));
        h.tick_ok(h.plan().retire(1, true));
        let lease = h.engine.seed_acquire(&turn2, 6, determinism::BEST_EFFORT).expect("a lease on the cached turn");
        let p2 = h.prompt(&turn2);
        let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p2).prefill(2, 6, 1).decode(2, 1));
        assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
        (h.rings_tokens(&ev.emit_for(2).token_ref), cells_used(&h), h.engine.cache_bytes())
    };
    let (copied, copied_cells, copied_cache) = run(false);
    let (taken, taken_cells, taken_cache) = run(true);
    assert_eq!(taken, copied, "the lane decodes the same either way");
    assert_eq!((copied_cells, copied_cache), (13, 6), "a copy beside the entry");
    assert_eq!((taken_cells, taken_cache), (7, 0), "the entry itself, and nothing in the cache meanwhile");

    let mut h = with(cfg(true));
    let p1 = h.prompt(&turn2[..6]);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 6));
    h.tick_ok(h.plan().retire(1, true));
    let first = h.engine.seed_acquire(&turn2, 6, determinism::BEST_EFFORT).expect("a lease");
    let second = h.engine.seed_acquire(&turn2, 6, determinism::BEST_EFFORT).expect("a second lease on the same entry");
    let p2 = h.prompt(&turn2);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = first; a }, 2, p2).prefill(2, 6, 1).decode(2, 1));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(h.engine.cache_bytes(), 6, "a shared entry stays in the cache");
    h.engine.seed_release(second).unwrap();
}

#[test]
fn a_branch_from_a_cached_conversation_copies_it_and_leaves_it_for_its_siblings() {
    let cfg = || FakeConfig { cells_total: 64, page: 1, kv_bytes_per_token: 1, takeover_preferred: true, ..Default::default() };
    let document: Vec<u32> = vec![10, 11, 12, 13, 14, 15];
    let primer: Vec<u32> = document.iter().copied().chain([40, 41]).collect();
    let mut h = with(cfg());
    let p1 = h.prompt(&primer);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 8).decode(1, 1));
    h.tick_ok(h.plan().retire(1, true));
    assert_eq!(h.engine.cache_bytes(), 9, "the primer's prompt and its reply");

    let branch = |q: u32| -> Vec<u32> { document.iter().copied().chain([q, q + 1]).collect() };
    let first = branch(50);
    let lease = h.engine.seed_acquire(&first, 6, determinism::BEST_EFFORT).expect("a lease on the primer's entry");
    let p2 = h.prompt(&first);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p2).prefill(2, 6, 2).decode(2, 1));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(h.engine.cache_bytes(), 9, "a branch admitted alone copies: the entry stays for the next one");
    let copied = h.rings_tokens(&ev.emit_for(2).token_ref);
    let mut cold = with(cfg());
    let pc = cold.prompt(&first);
    let ev = cold.tick_ok(cold.plan().admit(1, pc).prefill(1, 0, 8).decode(1, 1));
    assert_eq!(copied, cold.rings_tokens(&ev.emit_for(1).token_ref), "the copy decodes as a cold run of the branch does");
    for q in [60, 70, 80] {
        let next = h.engine.seed_acquire(&branch(q), 6, determinism::BEST_EFFORT).expect("a sibling seeds the document too");
        h.engine.seed_release(next).unwrap();
    }

    let mut h = with(cfg());
    let p1 = h.prompt(&primer);
    let ev = h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 8).decode(1, 1));
    let reply = h.rings_tokens(&ev.emit_for(1).token_ref);
    h.tick_ok(h.plan().retire(1, true));
    let turn2: Vec<u32> = primer.iter().chain(&reply).copied().chain([90]).collect();
    let lease = h.engine.seed_acquire(&turn2, 9, determinism::BEST_EFFORT).expect("a lease on the first turn");
    let p2 = h.prompt(&turn2);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p2).prefill(2, 9, 1).decode(2, 1));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(h.engine.cache_bytes(), 0, "the conversation's next turn still takes its entry over");
}

#[test]
fn a_lanes_state_kept_for_a_shared_prefix_is_copied_from_by_a_lane_admitted_alone() {
    let cfg = FakeConfig {
        truncate_partial: false,
        takeover_preferred: true,
        cells_total: 64,
        page: 1,
        kv_bytes_per_token: 1,
        ..Default::default()
    };
    let mut h = with(cfg);
    let leader: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 20, 21];
    let p1 = h.prompt(&leader);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 6));
    let parent = h.engine.lane_sequence(1).expect("the leader's sequence");
    let copy = h.engine.seq_fork(parent, 0).expect("a fork where the shared prefix ends");
    h.engine.publish_sequence(copy, &leader[..6]).expect("kept for the requests sharing it");

    let follower: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 30, 31];
    let lease = h.engine.seed_acquire(&follower, 6, determinism::BEST_EFFORT).expect("a lease on the kept state");
    let p2 = h.prompt(&follower);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p2).prefill(2, 6, 2).decode(2, 1));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(h.engine.cache_bytes(), 6, "the kept state is still there for the next request");
    let next = h.engine.seed_acquire(&follower, 6, determinism::BEST_EFFORT).expect("and seeds it");
    h.engine.seed_release(next).unwrap();
}

#[test]
fn a_private_copy_of_a_lanes_state_is_kept_as_any_entry() {
    // A turn's checkpoint serves the chat's next turn alone: forked private,
    // that turn takes it over where a shared prefix would be copied from.
    let cfg = FakeConfig {
        truncate_partial: false,
        takeover_preferred: true,
        cells_total: 64,
        page: 1,
        kv_bytes_per_token: 1,
        ..Default::default()
    };
    let mut h = with(cfg);
    let turn: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 20, 21];
    let p1 = h.prompt(&turn);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 6));
    let parent = h.engine.lane_sequence(1).expect("the lane's sequence");
    let copy = h.engine.seq_fork(parent, superfluid_abi::fork_flags::PRIVATE).expect("a private fork");
    h.engine.publish_sequence(copy, &turn[..6]).expect("kept for the next turn");

    let next: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 30, 31];
    let lease = h.engine.seed_acquire(&next, 6, determinism::BEST_EFFORT).expect("a lease on the kept state");
    let p2 = h.prompt(&next);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p2).prefill(2, 6, 2).decode(2, 1));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(h.engine.cache_bytes(), 0, "the next turn took the private copy over");
}

#[test]
fn a_seed_under_the_runtimes_floor_is_refused_unless_it_is_the_entry_whole() {
    let mut h = with(FakeConfig { page: 1, seed_min_tokens: 4, ..Default::default() });
    let long: Vec<u32> = vec![10, 11, 12, 13, 14, 15];
    let p = h.prompt(&long);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 6));
    h.tick_ok(h.plan().retire(1, true));
    assert!(h.engine.seed_acquire(&long, 3, determinism::BEST_EFFORT).is_err(), "three tokens of a six-token entry");
    let at_floor = h.engine.seed_acquire(&long, 4, determinism::BEST_EFFORT).expect("four tokens: the floor");
    h.engine.seed_release(at_floor).unwrap();

    let short: Vec<u32> = vec![20, 21];
    let p = h.prompt(&short);
    h.tick_ok(h.plan().admit(2, p).prefill(2, 0, 2));
    h.tick_ok(h.plan().retire(2, true));
    let next: Vec<u32> = vec![20, 21, 22];
    let whole = h.engine.seed_acquire(&next, 2, determinism::BEST_EFFORT).expect("a two-token entry, seeded whole");
    h.engine.seed_release(whole).unwrap();
}

#[test]
fn an_entry_taken_over_is_cut_to_the_seed_and_a_shared_one_is_left() {
    let mut h = with(FakeConfig { cells_total: 8, page: 1, kv_bytes_per_token: 1, ..Default::default() });
    let cached: Vec<u32> = vec![10, 11, 12, 13, 14, 15];
    let p1 = h.prompt(&cached);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 6));
    h.tick_ok(h.plan().retire(1, true));
    let next: Vec<u32> = vec![10, 11, 12, 13, 30, 31];

    let shared = h.engine.seed_acquire(&next, 4, determinism::BEST_EFFORT).expect("a lease");
    let other = h.engine.seed_acquire(&next, 4, determinism::BEST_EFFORT).expect("a second lease on the same entry");
    let p2 = h.prompt(&next);
    let err = h.tick_err(h.plan().admit_with(|mut a| { a.seed_handle = shared; a }, 2, p2).prefill(2, 4, 2).decode(2, 1));
    assert_eq!(err, Status::NeedsReplan);
    assert_eq!((cells_used(&h), h.engine.cache_bytes()), (6, 6), "a refused plan moves nothing");
    h.engine.seed_release(other).unwrap();

    let p2 = h.prompt(&next);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = shared; a }, 2, p2).prefill(2, 4, 2).decode(2, 1));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(h.engine.cache_bytes(), 0);
    assert_eq!(cells_used(&h), 6, "four seeded cells and the two new tokens");
    let mut cold = with(FakeConfig { cells_total: 8, page: 1, kv_bytes_per_token: 1, ..Default::default() });
    let p = cold.prompt(&next);
    let want = cold.tick_ok(cold.plan().admit(1, p).prefill(1, 0, 6).decode(1, 1));
    assert_eq!(h.rings_tokens(&ev.emit_for(2).token_ref), cold.rings_tokens(&want.emit_for(1).token_ref));
}

#[test]
fn an_entry_taken_over_gives_its_room_to_the_whole_plan() {
    let mut h = with(FakeConfig { cells_total: 16, page: 1, kv_bytes_per_token: 1, ..Default::default() });
    let shared: Vec<u32> = (10..14).collect();
    let long: Vec<u32> = (50..60).collect();
    for (lane, tokens) in [(1u64, &shared), (2, &long)] {
        let p = h.prompt(tokens);
        h.tick_ok(h.plan().admit(lane, p).prefill(lane, 0, tokens.len() as u32));
        h.tick_ok(h.plan().retire(lane, true));
    }
    assert_eq!(cells_used(&h), 14);
    let on_shared: Vec<u32> = shared.iter().copied().chain([90]).collect();
    let on_long: Vec<u32> = vec![50, 51, 91];
    let copied = h.engine.seed_acquire(&on_shared, 4, determinism::BEST_EFFORT).expect("a lease on the shared entry");
    let held = h.engine.seed_acquire(&on_shared, 4, determinism::BEST_EFFORT).expect("a second lease keeps it shared");
    let taken = h.engine.seed_acquire(&on_long, 2, determinism::BEST_EFFORT).expect("a lease on the long entry");
    let (p3, p4) = (h.prompt(&on_shared), h.prompt(&on_long));
    let ev = h.tick_ok(
        h.plan()
            .admit_with(|mut a| { a.seed_handle = copied; a }, 3, p3)
            .prefill(3, 4, 1)
            .admit_with(|mut a| { a.seed_handle = taken; a }, 4, p4)
            .prefill(4, 2, 1),
    );
    assert_eq!((ev.admit_for(3).status, ev.admit_for(4).status), (admit_status::ADMITTED, admit_status::ADMITTED));
    assert_eq!(cells_used(&h), 10);
    assert_eq!(h.engine.cache_bytes(), 4, "the shared entry stays; the long one is the second lane's sequence");
    h.engine.seed_release(held).unwrap();
}

#[test]
fn a_step_the_runtime_refuses_stops_its_lanes_not_the_tick() {
    let mut h = harness();
    let p1 = h.prompt(&[1, 2, 3]);
    let p2 = h.prompt(&[4, 5, 6]);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 3));
    h.engine.primitives_mut().fail_next_step(PrimError::Capacity);
    let ev = h.tick_ok(h.plan().admit(2, p2).prefill(2, 0, 3).decode(1, 2));
    assert_eq!(ev.tick_status, tick_status::LANE_ERRORS);
    assert_eq!(ev.faults.len(), 1);
    assert_eq!((ev.faults[0].lane_tag, ev.faults[0].code), (2, status::fault_code::OOM_TENTATIVE));
    assert_eq!(ev.emit_for(1).n_tokens, 2, "the lane beside it decoded");
    assert_eq!(h.engine.lane_sequence(2), None);
    assert_eq!(cells_used(&h), 4, "lane 1's prompt and one generated token; nothing of lane 2 is held");
    h.tick_ok(h.plan().retire(2, true));
    assert_eq!(cells_used(&h), 4);
    assert!(h.engine.seed_acquire(&[4, 5, 6, 7], 3, 0).is_err(), "no entry over a sequence that never held the prompt");
    h.engine.primitives_mut().fail_next_step(PrimError::Fatal);
    assert_eq!(h.tick_err(h.plan().decode(1, 1)), Status::Fatal);
}

#[test]
fn the_repetition_penalties_skip_special_and_stop_tokens() {
    let marker: u32 = 77;
    let fake = FakeConfig::default();
    let vocabulary = superfluid_executor::Vocabulary {
        tokens: (0..fake.vocab).map(|i| format!("t{i}").into_bytes()).collect(),
        specials: vec![("<|turn|>".to_string(), marker)],
        eos: vec![fake.eos],
    };
    for (cfg, structure) in [(FakeConfig { vocabulary: Some(vocabulary), ..Default::default() }, true), (FakeConfig::default(), false)] {
        let mut h = with(cfg);
        let prompt = vec![marker, 5, 6];
        let best = h.engine.primitives().token_for(&prompt);
        assert_ne!(best, marker);
        let bias = h.engine.logit_bias_create(&[marker as i32], &[18.0]).unwrap();
        let p = h.prompt(&prompt);
        let ev = h.tick_ok(
            h.plan()
                .admit_with(
                    |mut a| {
                        a.logit_bias_handle = bias;
                        a.params.repeat_penalty = 2.0;
                        a
                    },
                    1,
                    p,
                )
                .prefill(1, 0, 3)
                .decode(1, 1),
        );
        let want = if structure { marker } else { best };
        assert_eq!(h.rings_tokens(&ev.emit_for(1).token_ref), vec![want], "a special token: {structure}");
    }
}

struct NarrowOut(superfluid_engine::InMemoryRings, usize);

impl superfluid_engine::Rings for NarrowOut {
    fn read_tokens(&self, r: &TokenRef) -> Result<Vec<u32>, superfluid_engine::rings::RingError> {
        self.0.read_tokens(r)
    }
    fn write_tokens(&mut self, tokens: &[u32]) -> Result<TokenRef, superfluid_engine::rings::RingError> {
        if tokens.len() > self.1 {
            return Err(superfluid_engine::rings::RingError::OutOfBounds);
        }
        self.0.write_tokens(tokens)
    }
    fn write_logits_row(&mut self, row: &[f32]) -> RingRef {
        self.0.write_logits_row(row)
    }
    fn read_logits_row(&self, r: &RingRef) -> Result<Vec<f32>, superfluid_engine::rings::RingError> {
        self.0.read_logits_row(r)
    }
}

impl superfluid_engine::testing::StageRings for NarrowOut {
    fn stage_prompt(&mut self, tokens: &[u32]) -> TokenRef {
        self.0.stage_prompt(tokens)
    }
}

#[test]
fn an_emit_the_out_ring_cannot_hold_faults_its_lane_not_the_tick() {
    let mut h = Harness::with_engine_and_rings(
        Executor::new(FakePrimitives::new(FakeConfig::default()), ExecutorConfig::default()),
        NarrowOut(superfluid_engine::InMemoryRings::new(), 4),
    );
    let (p1, p2) = (h.prompt(&[5, 6, 7]), h.prompt(&[8, 9]));
    let ev = h.tick_ok(h.plan().admit(1, p1).admit(2, p2).prefill(1, 0, 3).prefill(2, 0, 2).decode(1, 8).decode(2, 2));
    assert_eq!(ev.tick_status, tick_status::LANE_ERRORS);
    assert_eq!(ev.faults.len(), 1);
    assert_eq!((ev.faults[0].lane_tag, ev.faults[0].code), (1, fault::REF_SPAN_OOB));
    assert!(ev.emits.iter().all(|e| e.lane_tag != 1), "a faulted lane emits nothing");
    assert_eq!(h.rings_tokens(&ev.emit_for(2).token_ref).len(), 2);
    h.tick_ok(h.plan().retire(1, false).retire(2, false));
}
