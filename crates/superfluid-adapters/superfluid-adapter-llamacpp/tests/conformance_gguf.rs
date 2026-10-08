use std::path::{Path, PathBuf};

use superfluid_adapter_llamacpp::{LlamaConfig, LlamaRuntime};
use superfluid_engine::testing::{tick_contract, Harness};
use superfluid_executor::{Executor, ExecutorConfig, RuntimePrimitives};

fn gguf() -> Option<PathBuf> {
    if let Err(why) = superfluid_adapter_llamacpp::sys::load() {
        eprintln!("SKIP: {why}");
        return None;
    }
    if let Ok(p) = std::env::var("SUPERFLUID_TEST_GGUF") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let p = root.join("models/Qwen3-0.6B-Q4_K_M.gguf");
    p.is_file().then_some(p)
}

fn harness(path: &Path, streams: bool) -> Harness<Executor<LlamaRuntime>> {
    let layout = if streams {
        LlamaConfig { streams: Some(true), ..Default::default() }
    } else {
        LlamaConfig { n_seq_max: 16, ..Default::default() }
    };
    let rt = LlamaRuntime::open(LlamaConfig { model_path: path.to_path_buf(), max_seq_len: 512, max_batch: 8, ..layout }).expect("open gguf");
    assert_eq!(rt.describe().copy_shares_cells, !streams, "the layout asked for");
    Harness::with_engine(Executor::new(rt, ExecutorConfig::default()))
}

macro_rules! contract_case {
    ($name:ident) => {
        #[test]
        fn $name() {
            let Some(path) = gguf() else {
                eprintln!("SKIP: no GGUF");
                return;
            };
            tick_contract::$name(harness(&path, false));
        }

        mod $name {
            #[test]
            fn with_a_buffer_per_sequence() {
                let Some(path) = super::gguf() else {
                    eprintln!("SKIP: no GGUF");
                    return;
                };
                super::tick_contract::$name(super::harness(&path, true));
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
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    for streams in [false, true] {
        tick_contract::same_log_same_seed_same_tokens(|| harness(&path, streams));
    }
}
