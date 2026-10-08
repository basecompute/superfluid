//! The conformance kit IN-PROCESS on the native engine.

mod common;

use superfluid_abi::{ring_kind, ring_role};
use superfluid_engine::rings::RingAttachment;
use superfluid_engine::testing::{tick_contract, Harness};
use superfluid_engine::Engine;
use superfluid_shm::{RingRole, SharedRing, SharedRings, LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};
use common::{model_path, FfiEngineConfig, NativeEngine, MODEL_LOCK};

fn ring(engine: &mut NativeEngine, ring_id: u32, kind: u32, role: u32, slots: u32, slot_bytes: u32) -> SharedRing {
    let ring = SharedRing::create(ring_id, slots, slot_bytes, RingRole::Writer).expect("create ring");
    engine
        .attach_ring(&RingAttachment {
            ring_id,
            kind,
            role,
            slots,
            slot_bytes,
            base: ring.base_ptr(),
            len: ring.seg_len(),
        })
        .expect("engine attaches the ring");
    ring
}

fn harness(path: &std::path::Path) -> Harness<NativeEngine, SharedRings> {
    let mut engine = NativeEngine::load(FfiEngineConfig {
        model_path: path.to_path_buf(),
        max_context: 4096,
        max_batch_size: 8,
        seed_ttl_ticks: 64,
    })
    .expect("model load");
    let vocab = engine.vocab();
    let token_in = ring(&mut engine, TOKEN_RING_IN, ring_kind::TOKENS, ring_role::ENGINE_READS, 64, 16 + 4096 * 4);
    let token_out = ring(&mut engine, TOKEN_RING_OUT, ring_kind::TOKENS, ring_role::ENGINE_WRITES, 64, 16 + 1024 * 4);
    let logits = ring(&mut engine, LOGITS_RING, ring_kind::LOGITS, ring_role::ENGINE_WRITES, 8, 16 + vocab * 4);
    Harness::with_engine_and_rings(engine, SharedRings { token_in, token_out, logits, control: None })
}

macro_rules! contract_case {
    ($name:ident) => {
        #[test]
        fn $name() {
            let Some(path) = model_path() else {
                eprintln!("SKIP: no test model");
                return;
            };
            let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            tick_contract::$name(harness(&path));
        }
    };
}

contract_case!(plan_seq_must_be_monotonic);
contract_case!(structural_rejections);
contract_case!(stale_ring_generation_rejected);
contract_case!(drain_takes_no_new_work);
contract_case!(host_lane_max_new_tokens_must_be_one);
contract_case!(commit_on_gpu_lane_rejected);
contract_case!(prefill_budget_sheds_and_reports);
contract_case!(noncontiguous_prefill_rejected);
contract_case!(events_echo_plan_seq);
contract_case!(a_replayed_decode_tail_resumes_the_lane);

#[test]
fn same_log_same_seed_same_tokens() {
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    tick_contract::same_log_same_seed_same_tokens(|| harness(&path));
}
