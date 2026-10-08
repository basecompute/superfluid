mod common;

use superfluid_abi::*;
use superfluid_engine::{Engine, EngineConfig};
use common::{tick_contract, Harness};

macro_rules! contract_case {
    ($name:ident) => {
        #[test]
        fn $name() {
            tick_contract::$name(Harness::new());
        }
    };
}

contract_case!(admit_prefill_decode_in_one_tick);
contract_case!(plan_seq_must_be_monotonic);
contract_case!(structural_rejections);
contract_case!(stale_ring_generation_rejected);
contract_case!(drain_takes_no_new_work);
contract_case!(host_alternation_full_loop);
contract_case!(host_lane_max_new_tokens_must_be_one);
contract_case!(commit_on_gpu_lane_rejected);
contract_case!(prefill_budget_sheds_and_reports);
contract_case!(noncontiguous_prefill_rejected);
contract_case!(per_lane_fault_does_not_fail_tick);
contract_case!(retire_publish_then_seeded_readmission);
contract_case!(events_echo_plan_seq);
contract_case!(a_replayed_decode_tail_resumes_the_lane);

#[test]
fn same_log_same_seed_same_tokens() {
    tick_contract::same_log_same_seed_same_tokens(Harness::new);
}

#[test]
fn seed_lease_expires_by_ttl() {
    let cfg = EngineConfig {
        seed_ttl_ticks: 1,
        ..Default::default()
    };
    let mut h = Harness::with_config(cfg);
    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
    h.tick_ok(h.plan().retire(1, true));
    let handle = h
        .engine
        .seed_acquire(&tokens, 64, determinism::BEST_EFFORT)
        .unwrap();
    h.tick_ok(h.plan());
    h.tick_ok(h.plan());
    let p2 = h.prompt(&tokens);
    let err = h.tick_err(h.plan().admit_with(
        |mut a| {
            a.seed_handle = handle;
            a
        },
        2,
        p2,
    ));
    assert_eq!(err, Status::RejectStaleSeed);
}

#[test]
fn needs_replan_when_envelope_cannot_serve() {
    let cfg = EngineConfig {
        pool_bytes: 64 * 100,
        ..Default::default()
    };
    let mut h = Harness::with_config(cfg);
    let tokens: Vec<u32> = (0..200).collect();
    let p = h.prompt(&tokens);
    let err = h.tick_err(h.plan().admit(1, p).prefill(1, 0, 200));
    assert_eq!(err, Status::NeedsReplan);
    assert!(h.engine.lane_sequence(1).is_none());
}

#[test]
fn eviction_within_envelope_frees_pool() {
    let cfg = EngineConfig {
        pool_bytes: 64 * 100,
        ..Default::default()
    };
    let mut h = Harness::with_config(cfg);
    let t1: Vec<u32> = (0..64).collect();
    let p1 = h.prompt(&t1);
    h.tick_ok(h.plan().admit(1, p1).prefill(1, 0, 64));
    h.tick_ok(h.plan().retire(1, true));

    let t2: Vec<u32> = (1000..1090).collect();
    let p2 = h.prompt(&t2);
    let ev = h.tick_ok(h.plan().admit(2, p2).prefill(2, 0, 90));
    assert!(ev.admit_results[0].status == admit_status::ADMITTED);
}

#[test]
fn a_grammar_continuation_resumes_where_the_generation_stopped() {
    const SCHEMA: &str = r#"{"type":"object"}"#;
    let prompt = [10u32, 20, 30];
    let mut h = Harness::new();
    let g = h.engine.grammar_create(SCHEMA).unwrap();
    let p = h.prompt(&prompt);
    let ev = h.tick_ok(
        h.plan().admit_with(|mut a| { a.grammar_handle = g; a }, 1, p).prefill(1, 0, 3).decode(1, 8),
    );
    let full = h.rings_tokens(&ev.emit_for(1).token_ref);
    assert_eq!(full.len(), 8);

    let mut cont = prompt.to_vec();
    cont.extend_from_slice(&full[..5]);
    let resume = |replay: u32| {
        let mut h = Harness::new();
        let g = h.engine.grammar_create(SCHEMA).unwrap();
        let p = h.prompt(&cont);
        let ev = h.tick_ok(
            h.plan()
                .admit_with(
                    |mut a| {
                        a.grammar_handle = g;
                        a.grammar_replay = replay;
                        a.rng_counter_base = 5;
                        a
                    },
                    1,
                    p,
                )
                .prefill(1, 0, 8)
                .decode(1, 3),
        );
        h.rings_tokens(&ev.emit_for(1).token_ref)
    };
    assert_eq!(resume(5), full[5..].to_vec(), "resumed where the grammar stopped");
    assert_ne!(resume(0), full[5..].to_vec(), "restarted at the grammar's start");

    let mut h = Harness::new();
    let g = h.engine.grammar_create(SCHEMA).unwrap();
    let p = h.prompt(&cont);
    let no_grammar = h.tick_err(h.plan().admit_with(|mut a| { a.grammar_replay = 5; a }, 1, p));
    assert_eq!(no_grammar, Status::RejectIllegalCombination);
    let past_the_prompt =
        h.tick_err(h.plan().admit_with(|mut a| { a.grammar_handle = g; a.grammar_replay = 9; a }, 1, p));
    assert_eq!(past_the_prompt, Status::RejectIllegalCombination);
}
