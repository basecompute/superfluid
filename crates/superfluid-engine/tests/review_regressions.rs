//! Regression tests for the independent review round.

mod common;

use superfluid_abi::*;
use superfluid_engine::{Engine, EngineConfig, SpaceConfig};
use common::Harness;

const KV: u32 = 1;
const GDN: u32 = 2;

fn range(a: u64, b: u64) -> TokenRange {
    TokenRange { start: a, end: b }
}

fn publish_prefix(h: &mut Harness, lane: u64) -> Vec<u32> {
    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(lane, p).prefill(lane, 0, 64));
    h.tick_ok(h.plan().retire(lane, true));
    tokens
}

#[test]
fn duplicate_seed_handle_in_one_plan_rejected() {
    let mut h = Harness::new();
    let tokens = publish_prefix(&mut h, 1);
    let seed = h
        .engine
        .seed_acquire(&tokens, 64, determinism::BEST_EFFORT)
        .unwrap();
    let p1 = h.prompt(&tokens);
    let p2 = h.prompt(&tokens);
    let err = h.tick_err(
        h.plan()
            .admit_with(
                |mut a| {
                    a.seed_handle = seed;
                    a
                },
                2,
                p1,
            )
            .admit_with(
                |mut a| {
                    a.seed_handle = seed;
                    a
                },
                3,
                p2,
            ),
    );
    assert_eq!(err, Status::RejectStaleSeed);
    let p3 = h.prompt(&tokens);
    let ev = h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.seed_handle = seed;
            a
        },
        4,
        p3,
    ));
    assert_eq!(ev.admit_for(4).status, admit_status::ADMITTED);
    assert_eq!(
        h.engine.seed_release(seed).unwrap_err(),
        Status::UnknownHandle
    );
}

#[test]
fn taint_survives_lossless_snapshot_restore() {
    let mut h = Harness::new();
    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
    let seq = h.engine.lane_sequence(1).unwrap();

    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 64), encoding::Q8)
        .unwrap();
    let op = h
        .engine
        .space_demote(seq, KV, range(0, 64), encoding::Q8, bytes, gen)
        .unwrap();
    h.engine.pump();
    let payload = h.engine.take_op_output(op).unwrap();
    let op = h
        .engine
        .space_promote(seq, KV, range(0, 64), &payload)
        .unwrap();
    h.engine.pump();
    let _ = op;
    assert_eq!(
        h.engine.space_taint(seq, KV),
        Some(taint::QUANTIZED_DEMOTION)
    );

    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 64), encoding::LOSSLESS)
        .unwrap();
    let op = h
        .engine
        .space_demote(seq, KV, range(0, 64), encoding::LOSSLESS, bytes, gen)
        .unwrap();
    h.engine.pump();
    let sealed = h.engine.take_op_output(op).unwrap();
    let fresh = h.engine.create_sequence();
    let op = h.engine.space_restore(fresh, KV, &sealed).unwrap();
    h.engine.pump();
    let _ = op;
    assert_eq!(
        h.engine.space_taint(fresh, KV),
        Some(taint::QUANTIZED_DEMOTION),
        "lossless restore laundered taint"
    );
}

#[test]
fn promote_is_tier_movement_not_restore() {
    let mut h = Harness::new();
    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
    let seq = h.engine.lane_sequence(1).unwrap();

    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 16), encoding::LOSSLESS)
        .unwrap();
    let op = h
        .engine
        .space_demote(seq, KV, range(0, 16), encoding::LOSSLESS, bytes, gen)
        .unwrap();
    h.engine.pump();
    let payload = h.engine.take_op_output(op).unwrap();
    assert_eq!(h.engine.space_valid_len(seq, KV), Some(64));
    let op = h
        .engine
        .space_promote(seq, KV, range(0, 16), &payload)
        .unwrap();
    h.engine.pump();
    let _ = op;
    assert_eq!(
        h.engine.space_valid_len(seq, KV),
        Some(64),
        "promote truncated the sequence"
    );

    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 16), encoding::LOSSLESS)
        .unwrap();
    let op = h
        .engine
        .space_demote(seq, KV, range(0, 16), encoding::LOSSLESS, bytes, gen)
        .unwrap();
    h.engine.pump();
    let payload = h.engine.take_op_output(op).unwrap();
    let other = h.engine.create_sequence();
    assert_eq!(
        h.engine
            .space_promote(other, KV, range(0, 16), &payload)
            .unwrap_err(),
        Status::RejectBounds
    );
}

#[test]
fn re_demote_of_demoted_range_rejected() {
    let mut h = Harness::new();
    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
    let seq = h.engine.lane_sequence(1).unwrap();
    let before = h.engine.pool_used();

    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 16), encoding::LOSSLESS)
        .unwrap();
    let op = h
        .engine
        .space_demote(seq, KV, range(0, 16), encoding::LOSSLESS, bytes, gen)
        .unwrap();
    h.engine.pump();
    let payload = h.engine.take_op_output(op).unwrap();

    let (bytes2, gen2) = h
        .engine
        .space_export_size(seq, KV, range(0, 16), encoding::LOSSLESS)
        .unwrap();
    assert_eq!(
        h.engine
            .space_demote(seq, KV, range(0, 16), encoding::LOSSLESS, bytes2, gen2)
            .unwrap_err(),
        Status::RejectBounds
    );

    let op = h
        .engine
        .space_promote(seq, KV, range(0, 16), &payload)
        .unwrap();
    h.engine.pump();
    let _ = op;
    assert_eq!(
        h.engine.pool_used(),
        before,
        "demote/promote cycle drained the ledger"
    );
}

#[test]
fn fault_releases_pool_bytes() {
    let mut h = Harness::new();
    let baseline = h.engine.pool_used();
    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
    assert!(h.engine.pool_used() > baseline);
    h.engine
        .inject_fault(1, superfluid_abi::status::fault_code::INTERNAL);
    let ev = h.tick_ok(h.plan().decode(1, 1));
    assert_eq!(ev.faults.len(), 1);
    assert_eq!(
        h.engine.pool_used(),
        baseline,
        "fault leaked the lane's pool bytes"
    );
}

#[test]
fn publish_dedup_does_not_strand_pool_bytes() {
    let mut h = Harness::new();
    let baseline = h.engine.pool_used();
    publish_prefix(&mut h, 1);
    let after_first = h.engine.pool_used();
    publish_prefix(&mut h, 2);
    assert_eq!(
        h.engine.pool_used(),
        after_first,
        "dedup stranded the second copy's bytes"
    );
    h.engine
        .cache_evict(u64::MAX, 0, u64::MAX, u64::MAX)
        .unwrap();
    assert_eq!(h.engine.pool_used(), baseline);
}

#[test]
fn protected_quota_floors_eviction_and_feasibility() {
    let cfg = EngineConfig {
        pool_bytes: 64 * 100,
        ..Default::default()
    };
    let mut h = Harness::with_config(cfg);
    publish_prefix(&mut h, 1);

    let freed = h
        .engine
        .cache_evict(u64::MAX, u64::MAX, u64::MAX, u64::MAX)
        .unwrap();
    assert_eq!(freed, 0);

    let t2: Vec<u32> = (1000..1090).collect();
    let p2 = h.prompt(&t2);
    let err = h.tick_err(
        h.plan()
            .evict_envelope(u64::MAX, u64::MAX)
            .protected_quota(u64::MAX)
            .admit(2, p2)
            .prefill(2, 0, 90),
    );
    assert_eq!(err, Status::NeedsReplan);
    assert!(h.engine.lane_sequence(2).is_none());
}

#[test]
fn commit_tokens_count_toward_feasibility() {
    let cfg = EngineConfig {
        pool_bytes: 64 * 8,
        ..Default::default()
    };
    let mut h = Harness::with_config(cfg);
    let tokens: Vec<u32> = (0..8).collect();
    let p = h.prompt(&tokens);
    let ev = h.tick_ok(
        h.plan()
            .admit_with(
                |mut a| {
                    a.sampling = sampling::HOST;
                    a
                },
                1,
                p,
            )
            .prefill(1, 0, 8)
            .decode(1, 1),
    );
    let nonce = ev.emit_for(1).logits_row.generation;
    let err = h.tick_err(h.plan().commit(1, 42, nonce));
    assert_eq!(err, Status::NeedsReplan);
}

#[test]
fn host_replay_tokens_count_toward_feasibility() {
    let cfg = EngineConfig {
        pool_bytes: 64 * 6,
        ..Default::default()
    };
    let mut h = Harness::with_config(cfg);
    let baseline = h.engine.pool_used();
    let tokens: Vec<u32> = (0..8).collect();
    let p = h.prompt(&tokens);
    let replayed = |mut a: superfluid_abi::LaneAdmit| {
        a.sampling = sampling::HOST;
        a.decode_replay = 2;
        a
    };
    let err = h.tick_err(h.plan().admit_with(replayed, 1, p).prefill(1, 0, 6).decode(1, 1));
    assert_eq!(err, Status::NeedsReplan);
    assert!(h.engine.lane_sequence(1).is_none());
    assert_eq!(h.engine.pool_used(), baseline, "a refused plan charges nothing");

    let cfg = EngineConfig {
        pool_bytes: 64 * 7,
        ..Default::default()
    };
    let mut h = Harness::with_config(cfg);
    let p = h.prompt(&tokens);
    let ev = h.tick_ok(h.plan().admit_with(replayed, 1, p).prefill(1, 0, 6).decode(1, 1));
    assert_eq!(ev.emit_for(1).n_tokens, 0, "mid-tail: no row, no token");
    assert_eq!(h.engine.pool_used(), 64 * 7);
    let err = h.tick_err(h.plan().decode(1, 1));
    assert_eq!(err, Status::NeedsReplan);
    assert_eq!(h.engine.pool_used(), 64 * 7);
}

#[test]
fn finished_lane_decode_and_prefill_retire_conflicts_rejected() {
    let mut h = Harness::new();
    let tokens: Vec<u32> = (0..4).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 4));
    h.engine.force_finish(1, finish::EOS);
    let ev = h.tick_ok(h.plan().decode(1, 1));
    assert_eq!(ev.emit_for(1).finish, finish::EOS);

    assert_eq!(
        h.tick_err(h.plan().decode(1, 1)),
        Status::RejectIllegalCombination
    );

    let t2: Vec<u32> = (0..8).collect();
    let p2 = h.prompt(&t2);
    h.tick_ok(h.plan().admit(2, p2).prefill(2, 0, 4));
    assert_eq!(
        h.tick_err(h.plan().prefill(2, 4, 4).retire(2, false)),
        Status::RejectIllegalCombination
    );
}

#[test]
fn seeded_admit_inherits_source_taint() {
    let mut h = Harness::new();
    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
    let seq = h.engine.lane_sequence(1).unwrap();

    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 64), encoding::Q8)
        .unwrap();
    let op = h
        .engine
        .space_demote(seq, KV, range(0, 64), encoding::Q8, bytes, gen)
        .unwrap();
    h.engine.pump();
    let payload = h.engine.take_op_output(op).unwrap();
    let op = h
        .engine
        .space_promote(seq, KV, range(0, 64), &payload)
        .unwrap();
    h.engine.pump();
    let _ = op;
    h.tick_ok(h.plan().retire(1, true));

    let seed = h
        .engine
        .seed_acquire(&tokens, 64, determinism::BEST_EFFORT)
        .unwrap();
    let p2 = h.prompt(&tokens);
    h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.seed_handle = seed;
            a
        },
        2,
        p2,
    ));
    let seq2 = h.engine.lane_sequence(2).unwrap();
    assert_eq!(
        h.engine.space_taint(seq2, KV),
        Some(taint::QUANTIZED_DEMOTION),
        "seeded admission laundered the source entry's taint"
    );
}

#[test]
fn expired_lease_pins_are_reclaimed() {
    let cfg = EngineConfig {
        seed_ttl_ticks: 1,
        ..Default::default()
    };
    let mut h = Harness::with_config(cfg);
    let tokens = publish_prefix(&mut h, 1);
    let _seed = h
        .engine
        .seed_acquire(&tokens, 64, determinism::BEST_EFFORT)
        .unwrap();
    h.tick_ok(h.plan());
    h.tick_ok(h.plan());
    h.tick_ok(h.plan());
    let freed = h
        .engine
        .cache_evict(u64::MAX, 0, u64::MAX, u64::MAX)
        .unwrap();
    assert!(freed > 0, "expired lease still pins the cache");
}

#[test]
fn ringkv_follows_its_own_matrix_row() {
    let cfg = EngineConfig {
        spaces: vec![SpaceConfig {
            space_id: 7,
            kind: space_kind::RING_KV,
            name: "swa.ring",
            version_tag: 1,
            bytes_per_token: 0,
            blob_bytes: 2048,
            page_size_tokens: 0,
            snapshot_cadence: 2,
            snapshot_interval_tokens: 32,
            flags: 0,
        }],
        ..Default::default()
    };
    let mut h = Harness::with_config(cfg);
    let tokens: Vec<u32> = (0..48).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 48));
    let seq = h.engine.lane_sequence(1).unwrap();

    let (bytes, gen) = h
        .engine
        .space_export_size(seq, 7, range(0, 41), encoding::LOSSLESS)
        .unwrap();
    assert!(h.engine.space_snapshot(seq, 7, 41, bytes, gen).is_ok());
    h.engine.pump();

    assert!(h
        .engine
        .space_export_size(seq, 7, range(0, 48), encoding::Q8)
        .is_ok());

    assert_eq!(
        h.engine.space_trim(seq, 7, 17).unwrap_err(),
        Status::OutOfBoundary
    );
}

#[test]
fn export_payload_sized_exactly_to_contract() {
    for (space, enc, snapshot) in [
        (KV, encoding::LOSSLESS, false),
        (KV, encoding::Q8, false),
        (GDN, encoding::LOSSLESS, true),
    ] {
        let mut h = Harness::new();
        let tokens: Vec<u32> = (0..64).collect();
        let p = h.prompt(&tokens);
        h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
        let seq = h.engine.lane_sequence(1).unwrap();
        let (required, gen) = h
            .engine
            .space_export_size(seq, space, range(0, 32), enc)
            .unwrap();
        let op = if snapshot {
            h.engine
                .space_snapshot(seq, space, 32, required, gen)
                .unwrap()
        } else {
            h.engine
                .space_demote(seq, space, range(0, 32), enc, required, gen)
                .unwrap()
        };
        h.engine.pump();
        let sealed = h.engine.take_op_output(op).unwrap();
        assert_eq!(
            sealed.len() as u64,
            required,
            "sealed export != sized bytes for space {space} enc {enc}"
        );
    }
}

#[test]
fn registration_scoping_and_kernel_caps() {
    let mut h = Harness::new();
    let err =
        common::register_prompt_lookup_with(&mut h, [1; 32], [2; 32], 0xF000_0000).unwrap_err();
    assert_eq!(err, Status::RegistrationRefused);

    let grant = common::register_prompt_lookup_with(&mut h, [1; 32], [2; 32], 0).unwrap();
    assert_eq!(grant.1.len(), 1);

    let grant = common::register_prompt_lookup_with(&mut h, [9; 32], [9; 32], 0).unwrap();
    assert!(grant.1.is_empty(), "cert granted by name label alone");
}
