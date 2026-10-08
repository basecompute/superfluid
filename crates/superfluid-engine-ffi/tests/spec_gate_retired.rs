//! A lane retired from speculation for the rest of its request (here by the per-lane yield floor)
//! stays retired when a losing gate runs untapped plain ticks.

#![allow(clippy::too_many_arguments)]

mod common;
use common::*;

#[test]
fn a_retired_lane_stays_retired_across_untapped_ticks() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    if !bundle_has_speculator(&path) {
        eprintln!("SKIP: bundle has no speculator section");
        return;
    }
    std::env::set_var("SUPERFLUID_SPEC_MIN_SPEEDUP", "100");
    std::env::set_var("SUPERFLUID_SPEC_MIN_YIELD", "100");
    std::env::set_var("SUPERFLUID_SPEC_LANE_YIELD_ROUNDS", "2");
    let (mut client, worker) = spawn_native(path);
    let mut seq = 1u64;
    let mut reg = prompt_lookup_reg();
    reg.strategy_id = "mtp-head".into();
    reg.impl_hash = [0x44; 32];
    client.register_strategy(reg).expect("mtp-head registers");

    let (off, _, _) = generate(&mut client, &mut seq, 1, &repetitive_prompt(), 640, false, 0, 0);
    let (on, proposed, _) = generate(&mut client, &mut seq, 2, &repetitive_prompt(), 640, false, 0, 1);
    eprintln!("retired lane: proposed {proposed} over {} tokens", on.len());
    assert!(proposed <= 32, "a retired lane was re-armed: {proposed} drafts proposed");
    assert_greedy_matches("mtp-head (retired lane, losing gate)", &on, &off);

    drop(client);
    worker.join().unwrap();
}
