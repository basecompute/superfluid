mod common;

use superfluid_engine::testing::tick_contract;
use common::{harness, mlx_dir, MODEL_LOCK};

macro_rules! contract_case {
    ($name:ident) => {
        #[test]
        fn $name() {
            let Some(path) = mlx_dir() else {
                eprintln!("SKIP: no MLX model");
                return;
            };
            let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            tick_contract::$name(harness(&path));
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
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    tick_contract::same_log_same_seed_same_tokens(|| harness(&path));
}
