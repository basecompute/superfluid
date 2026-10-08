//! A speculation strategy the throughput gate judges LOSING decodes its plain ticks untapped
//! (SpecGate::untapped_now).

#![allow(clippy::too_many_arguments)]

mod common;
use common::*;

#[test]
fn losing_gate_decodes_untapped_and_stays_exact() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    if !bundle_has_speculator(&path) {
        eprintln!("SKIP: bundle has no speculator section (untapped plain needs a tapped strategy)");
        return;
    }
    std::env::set_var("SUPERFLUID_SPEC_MIN_SPEEDUP", "100");
    std::env::set_var("SUPERFLUID_SPEC_MIN_YIELD", "0");
    let (mut client, worker) = spawn_native(path);
    let mut seq = 1u64;
    let mut reg = prompt_lookup_reg();
    reg.strategy_id = "mtp-head".into();
    reg.impl_hash = [0x44; 32];
    client.register_strategy(reg).expect("mtp-head registers");

    let prompt: Vec<u32> = match std::env::var("BASERT_TEST_PROMPT_IDS") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => repetitive_prompt(),
    };
    let (off, _, _) = generate(&mut client, &mut seq, 1, &prompt, 640, false, 0, 0);
    let (on, proposed, accepted) = generate(&mut client, &mut seq, 2, &prompt, 640, false, 0, 1);
    eprintln!("losing gate: proposed {proposed} accepted {accepted} over {} tokens", on.len());
    assert_greedy_matches("mtp-head (losing gate, untapped plain ticks)", &on, &off);
    assert!(proposed > 0, "speculating ticks (before the verdict, then re-checks) still drafted");

    drop(client);
    worker.join().unwrap();
}
