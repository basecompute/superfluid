//! Prompt-lookup speculation in the executor: drafts verified in one pass
//! change nothing a plain decode would have produced.

use superfluid_abi::{cert_mode, exactness, finish as finish_code, sampling, sampling_flag, SamplingParams, Status};
use superfluid_engine::testing::{register_prompt_lookup_with, Harness};
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives};

fn harness(cfg: FakeConfig) -> Harness<Executor<FakePrimitives>> {
    Harness::with_engine(Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()))
}

fn copying() -> FakeConfig {
    FakeConfig { verify_rows: true, copy_model: true, vocab: 512, eos: 511, ..FakeConfig::default() }
}

struct Run {
    tokens: Vec<u32>,
    finish: u32,
    proposed: u32,
    accepted: u32,
}

/// `ticks` ticks of `per_tick` new tokens on one lane, speculating or not.
fn generate(cfg: FakeConfig, speculate: bool, params: SamplingParams, mode: u32, prompt: &[u32], ticks: usize, per_tick: u16) -> Run {
    let mut h = harness(cfg);
    let slot = if speculate {
        register_prompt_lookup_with(&mut h, [1; 32], [2; 32], 0).expect("granted").0
    } else {
        0
    };
    let p = h.prompt(prompt);
    let mut run = Run { tokens: Vec::new(), finish: finish_code::NONE, proposed: 0, accepted: 0 };
    let ev = h.tick_ok(
        h.plan()
            .admit_with(
                |mut a| {
                    a.strategy_slot = slot;
                    a.sampling = mode;
                    a.params = params;
                    a.rng_counter_base = 77;
                    a
                },
                1,
                p,
            )
            .prefill(1, 0, prompt.len() as u32)
            .decode(1, per_tick),
    );
    take(&mut run, &h, &ev);
    for _ in 1..ticks {
        if run.finish != finish_code::NONE {
            break;
        }
        let ev = h.tick_ok(h.plan().decode(1, per_tick));
        take(&mut run, &h, &ev);
    }
    run
}

fn take(run: &mut Run, h: &Harness<Executor<FakePrimitives>>, ev: &superfluid_engine::testing::Events) {
    let e = ev.emit_for(1);
    run.tokens.extend(h.rings_tokens(&e.token_ref));
    run.finish = e.finish;
    run.proposed += e.spec.proposed;
    run.accepted += e.spec.accepted;
}

fn greedy() -> SamplingParams {
    SamplingParams { flags: sampling_flag::IGNORE_EOS, ..Default::default() }
}

#[test]
fn greedy_drafts_change_nothing_and_are_taken_when_the_text_repeats() {
    let prompt = [10, 11, 12, 13, 14, 15, 10, 11, 12];
    let plain = generate(copying(), false, greedy(), sampling::GPU_GREEDY, &prompt, 4, 16);
    let spec = generate(copying(), true, greedy(), sampling::GPU_GREEDY, &prompt, 4, 16);
    assert_eq!(plain.tokens.len(), 64);
    assert_eq!(spec.tokens, plain.tokens, "a verified draft is the plain decode");
    assert_eq!(plain.proposed, 0);
    assert!(spec.proposed > 0 && spec.accepted > spec.proposed / 2, "{} of {} drafts taken", spec.accepted, spec.proposed);
}

#[test]
fn sampled_drafts_draw_what_a_plain_decode_draws() {
    let params = SamplingParams { temperature: 0.8, top_p: 0.95, flags: sampling_flag::IGNORE_EOS, ..Default::default() };
    let prompt = [20, 21, 22, 23, 20, 21, 22, 23, 20, 21];
    let plain = generate(copying(), false, params, sampling::GPU_GUMBEL, &prompt, 3, 24);
    let spec = generate(copying(), true, params, sampling::GPU_GUMBEL, &prompt, 3, 24);
    assert_eq!(spec.tokens, plain.tokens, "each row is drawn at the position a plain decode draws it");
    assert!(spec.proposed > 0);
}

#[test]
fn rejected_drafts_roll_back_cleanly() {
    // A model that copies but misses every third position: drafts are taken
    // in part, and a rejected tail left in the state would change every row
    // after it.
    let cfg = FakeConfig { copy_miss_every: 3, ..copying() };
    let prompt = [3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4];
    let plain = generate(cfg.clone(), false, greedy(), sampling::GPU_GREEDY, &prompt, 4, 16);
    let spec = generate(cfg, true, greedy(), sampling::GPU_GREEDY, &prompt, 4, 16);
    assert_eq!(spec.tokens, plain.tokens);
    assert!(spec.accepted > 0 && spec.accepted < spec.proposed, "taken in part: {} of {}", spec.accepted, spec.proposed);
}

#[test]
fn an_end_of_sequence_inside_a_draft_ends_the_lane_there() {
    // The model goes 32, 33, 511 (EOS); the history holds that run after
    // [30, 31, 32], so once 32 is out the draft is [33, 511, ...].
    let cfg = copying();
    let prompt = [30, 31, 32, 33, 511, 40, 30, 31];
    let params = SamplingParams::default();
    let plain = generate(cfg.clone(), false, params, sampling::GPU_GREEDY, &prompt, 2, 16);
    let spec = generate(cfg, true, params, sampling::GPU_GREEDY, &prompt, 2, 16);
    assert_eq!(plain.finish, finish_code::EOS);
    assert_eq!(plain.tokens, vec![32, 33, 511]);
    assert!(spec.proposed > 0, "the EOS came inside a draft");
    assert_eq!((spec.tokens, spec.finish), (plain.tokens, plain.finish), "nothing after the EOS");
}

#[test]
fn registration_needs_rows_at_every_position_and_a_partial_rollback() {
    let mut h = harness(FakeConfig { verify_rows: false, ..copying() });
    assert_eq!(register_prompt_lookup_with(&mut h, [1; 32], [2; 32], 0).err(), Some(Status::RegistrationRefused));
    let mut h = harness(FakeConfig { truncate_partial: false, ..copying() });
    assert_eq!(register_prompt_lookup_with(&mut h, [1; 32], [2; 32], 0).err(), Some(Status::RegistrationRefused));
    let mut h = harness(copying());
    assert_eq!(
        register_prompt_lookup_with(&mut h, [1; 32], [2; 32], 1).err(),
        Some(Status::RegistrationRefused),
        "no kernel capability is on offer"
    );
    let (slot, certs) = register_prompt_lookup_with(&mut h, [1; 32], [2; 32], 0).expect("granted");
    assert_ne!(slot, 0);
    assert_eq!(certs.len(), 1);
    assert_eq!(certs[0].exactness, exactness::DISTRIBUTION_EXACT);
    assert_eq!(certs[0].sampling_modes, cert_mode::GREEDY | cert_mode::GUMBEL);
}

#[test]
fn a_slot_that_was_never_granted_is_refused_at_admission() {
    let mut h = harness(copying());
    let p = h.prompt(&[1, 2, 3]);
    let st = h.tick_err(
        h.plan()
            .admit_with(
                |mut a| {
                    a.strategy_slot = 9;
                    a
                },
                1,
                p,
            )
            .prefill(1, 0, 3),
    );
    assert_eq!(st, Status::RejectIllegalCombination);
}
