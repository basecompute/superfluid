//! Sampled speculation with the GPU sampler (the tick samples a verify round's rows on the device,
//! gumbel_topk_f16_rows).

#![allow(clippy::too_many_arguments)]

mod common;
use common::*;

#[test]
fn sampled_speculation_draws_the_plain_gpu_stream() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    std::env::remove_var("BASERTD_NO_GPU_SAMPLE");
    std::env::remove_var("BASERTD_NO_GPU_ROW_SAMPLE");
    std::env::set_var("BASERTD_GPU_SAMPLE_ALONE", "1");
    let mtp = bundle_has_speculator(&path);
    let (mut client, worker) = spawn_native(path);
    let mut seq = 1u64;
    let mut reg = prompt_lookup_reg();
    if mtp {
        reg.strategy_id = "mtp-head".into();
        reg.impl_hash = [0x44; 32];
    }
    client.register_strategy(reg).expect("strategy registers");

    let prompt: Vec<u32> = match std::env::var("BASERT_TEST_PROMPT_IDS") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) if mtp => NATURAL_PROMPT.to_vec(),
        Err(_) => repetitive_prompt(),
    };
    let (off, _, _) = generate(&mut client, &mut seq, 1, &prompt, 64, true, 4242, 0);
    let (on, proposed, accepted) = generate(&mut client, &mut seq, 2, &prompt, 64, true, 4242, 1);
    let common = off.iter().zip(on.iter()).take_while(|(a, b)| a == b).count();
    eprintln!(
        "{} sampled on GPU: proposed {proposed} accepted {accepted}; identical prefix {common}/{}",
        if mtp { "mtp-head" } else { "prompt-lookup" },
        off.len().min(on.len())
    );
    assert!(proposed > 0 && accepted > 0, "drafts were proposed and accepted ({accepted}/{proposed})");
    assert!(common >= COMMON_MIN, "one noise stream on and off ({common} identical)");

    drop(client);
    worker.join().unwrap();
}

const NATURAL_PROMPT: [u32; 32] = [
    760, 2235, 324, 55965, 51624, 1669, 264, 16002, 383, 279, 22317, 799, 19177, 6353, 13, 26482, 557, 264, 6321,
    11, 5158, 303, 264, 1375, 551, 6661, 506, 2957, 11, 321, 424, 1018,
];

const COMMON_MIN: usize = 48;
