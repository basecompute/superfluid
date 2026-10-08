mod common;

use superfluid_abi::*;
use superfluid_engine::Engine;
use common::Harness;

const KV: u32 = 1;
const GDN: u32 = 2;

fn seeded_seq(h: &mut Harness) -> u64 {
    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
    h.engine.lane_sequence(1).unwrap()
}

fn range(a: u64, b: u64) -> TokenRange {
    TokenRange { start: a, end: b }
}

#[test]
fn export_sizing_is_mandatory_and_generation_stamped() {
    let mut h = Harness::new();
    let seq = seeded_seq(&mut h);
    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 64), encoding::LOSSLESS)
        .unwrap();
    assert!(bytes > 128);

    h.tick_ok(h.plan().decode(1, 1));

    let err = h
        .engine
        .space_demote(seq, KV, range(0, 64), encoding::LOSSLESS, bytes, gen)
        .unwrap_err();
    assert_eq!(err, Status::StaleSizing);

    let (bytes2, gen2) = h
        .engine
        .space_export_size(seq, KV, range(0, 64), encoding::LOSSLESS)
        .unwrap();
    let err = h
        .engine
        .space_demote(seq, KV, range(0, 64), encoding::LOSSLESS, bytes2 - 1, gen2)
        .unwrap_err();
    assert_eq!(err, Status::BufferTooSmall);
}

#[test]
fn transfer_lock_rejects_ticks_and_conflicting_ops() {
    let mut h = Harness::new();
    let seq = seeded_seq(&mut h);
    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 16), encoding::LOSSLESS)
        .unwrap();
    let op = h
        .engine
        .space_demote(seq, KV, range(0, 16), encoding::LOSSLESS, bytes, gen)
        .unwrap();

    let err = h.tick_err(h.plan().decode(1, 1));
    assert_eq!(err, Status::RejectTransferLocked);

    let err = h
        .engine
        .space_demote(seq, KV, range(0, 16), encoding::LOSSLESS, bytes, gen)
        .unwrap_err();
    assert_eq!(err, Status::Busy);

    assert_eq!(h.engine.seq_fork(seq, 0).unwrap_err(), Status::Busy);
    assert_eq!(h.engine.space_trim(seq, KV, 0).unwrap_err(), Status::Busy);

    h.engine.pump();
    let st = h.engine.op_poll(op).unwrap();
    assert_eq!(st.state, op_state::DONE);
    let ev = h.tick_ok(h.plan().decode(1, 1));
    assert!(ev.op_completions.iter().any(|c| c.op == op));
    assert!(h.engine.seq_fork(seq, 0).is_ok());
}

#[test]
fn snapshot_restore_roundtrip_via_envelope() {
    let mut h = Harness::new();
    let seq = seeded_seq(&mut h);
    let (bytes, gen) = h
        .engine
        .space_export_size(seq, GDN, range(0, 64), encoding::LOSSLESS)
        .unwrap();
    let op = h.engine.space_snapshot(seq, GDN, 64, bytes, gen).unwrap();
    h.engine.pump();
    let payload = h.engine.take_op_output(op).expect("snapshot payload");

    let fresh = h.engine.create_sequence();
    let op2 = h.engine.space_restore(fresh, GDN, &payload).unwrap();
    h.engine.pump();
    assert_eq!(h.engine.op_poll(op2).unwrap().state, op_state::DONE);
    assert_eq!(h.engine.space_valid_len(fresh, GDN), Some(64));
}

#[test]
fn restore_rejects_corruption_and_wrong_target() {
    let mut h = Harness::new();
    let seq = seeded_seq(&mut h);
    let (bytes, gen) = h
        .engine
        .space_export_size(seq, GDN, range(0, 64), encoding::LOSSLESS)
        .unwrap();
    let op = h.engine.space_snapshot(seq, GDN, 64, bytes, gen).unwrap();
    h.engine.pump();
    let payload = h.engine.take_op_output(op).unwrap();

    let fresh = h.engine.create_sequence();

    let mut corrupt = payload.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0xFF;
    assert_eq!(
        h.engine.space_restore(fresh, GDN, &corrupt).unwrap_err(),
        Status::Checksum
    );

    assert_eq!(
        h.engine.space_restore(fresh, KV, &payload).unwrap_err(),
        Status::EnvelopeMismatch
    );

    assert_eq!(
        h.engine
            .space_restore(fresh, GDN, &payload[..64])
            .unwrap_err(),
        Status::EnvelopeMismatch
    );

    assert_eq!(h.engine.space_valid_len(fresh, GDN), Some(0));
}

#[test]
fn lossy_demote_taints_promoted_state() {
    let mut h = Harness::new();
    let seq = seeded_seq(&mut h);

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
    assert_eq!(
        h.engine.space_taint(seq, KV),
        Some(0),
        "demote alone taints nothing"
    );

    let op2 = h
        .engine
        .space_promote(seq, KV, range(0, 64), &payload)
        .unwrap();
    h.engine.pump();
    assert_eq!(h.engine.op_poll(op2).unwrap().state, op_state::DONE);
    assert_eq!(
        h.engine.space_taint(seq, KV),
        Some(taint::QUANTIZED_DEMOTION)
    );

    h.tick_ok(h.plan().retire(1, true));
    let tokens: Vec<u32> = (0..64).collect();
    assert_eq!(
        h.engine
            .seed_acquire(&tokens, 64, determinism::DETERMINISTIC)
            .unwrap_err(),
        Status::SeedUnservable
    );
    assert!(h
        .engine
        .seed_acquire(&tokens, 64, determinism::BEST_EFFORT)
        .is_ok());
}

#[test]
fn recurrent_blob_is_lossless_only() {
    let mut h = Harness::new();
    let seq = seeded_seq(&mut h);
    let err = h
        .engine
        .space_export_size(seq, GDN, range(0, 64), encoding::Q8)
        .unwrap_err();
    assert_eq!(err, Status::Unsupported);
}

#[test]
fn trim_rules_per_kind() {
    let mut h = Harness::new();
    let seq = seeded_seq(&mut h);
    assert!(h.engine.space_trim(seq, KV, 3).is_ok());
    assert_eq!(
        h.engine.space_trim(seq, GDN, 17).unwrap_err(),
        Status::OutOfBoundary
    );
    assert!(h.engine.space_trim(seq, GDN, 32).is_ok());
    assert_eq!(
        h.engine.space_trim(seq, GDN, 999).unwrap_err(),
        Status::RejectBounds
    );
}

#[test]
fn cancel_semantics() {
    let mut h = Harness::new();
    let seq = seeded_seq(&mut h);
    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 16), encoding::LOSSLESS)
        .unwrap();
    let op = h
        .engine
        .space_demote(seq, KV, range(0, 16), encoding::LOSSLESS, bytes, gen)
        .unwrap();
    h.engine.op_cancel(op).unwrap();
    assert_eq!(h.engine.op_poll(op).unwrap().state, op_state::CANCELLED);
    assert!(h.engine.take_op_output(op).is_none());
    assert!(h.engine.seq_fork(seq, 0).is_ok(), "claim released");

    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 16), encoding::LOSSLESS)
        .unwrap();
    let op2 = h
        .engine
        .space_demote(seq, KV, range(0, 16), encoding::LOSSLESS, bytes, gen)
        .unwrap();
    h.engine.pump();
    h.engine.op_cancel(op2).unwrap();
    assert_eq!(h.engine.op_poll(op2).unwrap().state, op_state::DONE);

    assert_eq!(h.engine.op_poll(9999).unwrap_err(), Status::UnknownHandle);
}

#[test]
fn demote_frees_pool_only_at_commit() {
    let mut h = Harness::new();
    let _seq = seeded_seq(&mut h);
    let used_before = h.engine.pool_used();
    let seq = h.engine.lane_sequence(1).unwrap();
    let (bytes, gen) = h
        .engine
        .space_export_size(seq, KV, range(0, 64), encoding::LOSSLESS)
        .unwrap();
    let op = h
        .engine
        .space_demote(seq, KV, range(0, 64), encoding::LOSSLESS, bytes, gen)
        .unwrap();
    assert_eq!(h.engine.pool_used(), used_before);
    h.engine.pump();
    let _ = op;
    assert!(h.engine.pool_used() < used_before);
}

#[test]
fn ops_unavailable_space_refuses_its_ops_and_every_seed() {
    let mut cfg = superfluid_engine::EngineConfig::default();
    for s in cfg.spaces.iter_mut() {
        if s.kind == space_kind::RECURRENT_BLOB {
            s.flags |= space_flag::OPS_UNAVAILABLE;
        }
    }
    let mut h = Harness::with_config(cfg);
    let gdn_desc = h
        .engine
        .state_spaces()
        .iter()
        .find(|d| d.space_id == GDN)
        .expect("the flagged space is still advertised");
    assert_eq!(gdn_desc.flags & space_flag::OPS_UNAVAILABLE, space_flag::OPS_UNAVAILABLE);

    let tokens: Vec<u32> = (0..64).collect();
    let p = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, p).prefill(1, 0, 64));
    let seq = h.engine.lane_sequence(1).unwrap();

    assert_eq!(
        h.engine
            .space_export_size(seq, GDN, range(0, 64), encoding::LOSSLESS)
            .unwrap_err(),
        Status::Unsupported
    );
    assert_eq!(h.engine.space_trim(seq, GDN, 32).unwrap_err(), Status::Unsupported);
    assert!(h
        .engine
        .space_export_size(seq, KV, range(0, 64), encoding::LOSSLESS)
        .is_ok());
    assert_eq!(
        h.engine
            .space_export_size(seq, 99, range(0, 64), encoding::LOSSLESS)
            .unwrap_err(),
        Status::UnknownHandle
    );

    h.tick_ok(h.plan().retire(1, true));
    let mut longer = tokens.clone();
    longer.push(64);
    assert_eq!(
        h.engine
            .seed_acquire(&longer, 64, determinism::BEST_EFFORT)
            .unwrap_err(),
        Status::SeedUnservable,
        "no seed can cover a space with no ops: refuse whole, never partial"
    );
}
