use superfluid_engine::testing::{tick_contract, Harness};
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives};

fn h() -> Harness<Executor<FakePrimitives>> {
    Harness::with_engine(Executor::new(FakePrimitives::default(), ExecutorConfig::default()))
}

/// A runtime that affords the cache one cell of its pool: every entry is
/// exported as it is published, and every seed is an import.
fn exporting() -> Harness<Executor<FakePrimitives>> {
    let cfg = FakeConfig { copy_shares_cells: true, cache_resident_cells: 1, cache_exported_cells: 1 << 20, ..Default::default() };
    Harness::with_engine(Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()))
}

macro_rules! contract_case {
    ($name:ident) => {
        #[test]
        fn $name() {
            tick_contract::$name(h());
        }

        mod $name {
            #[test]
            fn with_the_cache_out_of_the_pool() {
                super::tick_contract::$name(super::exporting());
            }
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
    tick_contract::same_log_same_seed_same_tokens(h);
    tick_contract::same_log_same_seed_same_tokens(exporting);
}
