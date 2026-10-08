//! The executor's state ops over the fake runtime.

use superfluid_abi::*;
use superfluid_engine::testing::Harness;
use superfluid_engine::Engine;
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives};

const KV: u32 = 1;

fn with(cfg: FakeConfig) -> Harness<Executor<FakePrimitives>> {
    Harness::with_engine(Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()))
}

fn harness() -> Harness<Executor<FakePrimitives>> {
    with(FakeConfig::default())
}

fn seeded_seq(h: &mut Harness<Executor<FakePrimitives>>) -> u64 {
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
    let mut h = harness();
    let seq = seeded_seq(&mut h);
    let (bytes, gen) = h.engine.space_export_size(seq, KV, range(0, 32), encoding::LOSSLESS).unwrap();
    assert!(bytes > 32 * 4, "envelope + payload");
    assert_eq!(h.engine.space_export_size(seq, KV, range(0, 64), encoding::LOSSLESS), Err(Status::RejectBounds));

    h.tick_ok(h.plan().decode(1, 1));
    assert_eq!(h.engine.space_snapshot(seq, KV, 32, bytes, gen), Err(Status::StaleSizing));

    let (bytes2, gen2) = h.engine.space_export_size(seq, KV, range(0, 32), encoding::LOSSLESS).unwrap();
    assert_eq!(h.engine.space_snapshot(seq, KV, 32, bytes2 - 1, gen2), Err(Status::BufferTooSmall));
    assert_eq!(h.engine.space_export_size(seq, 99, range(0, 32), encoding::LOSSLESS), Err(Status::UnknownHandle));
    assert_eq!(h.engine.space_export_size(999, KV, range(0, 32), encoding::LOSSLESS), Err(Status::UnknownHandle));
    assert_eq!(h.engine.space_export_size(seq, KV, range(0, 32), encoding::Q8), Err(Status::Unsupported));
}

#[test]
fn snapshot_restore_adopt_roundtrip() {
    let mut h = harness();
    let seq = seeded_seq(&mut h);
    let (bytes, gen) = h.engine.space_export_size(seq, KV, range(0, 48), encoding::LOSSLESS).unwrap();
    let op = h.engine.space_snapshot(seq, KV, 48, bytes, gen).unwrap();
    assert_eq!(h.engine.op_poll(op).unwrap().state, op_state::DONE);
    let payload = h.engine.take_op_output(op).expect("sealed payload");
    assert_eq!(payload.len() as u64, bytes);
    assert!(h.engine.take_op_output(op).is_none(), "taken once");
    let ev = h.tick_ok(h.plan());
    assert!(ev.op_completions.iter().any(|c| c.op == op && c.state == op_state::DONE));

    let fresh = h.engine.create_sequence().unwrap();
    let op2 = h.engine.space_restore(fresh, KV, &payload).unwrap();
    assert_eq!(h.engine.op_poll(op2).unwrap().state, op_state::DONE);
    assert_eq!(h.engine.primitives().contents(fresh).unwrap().len(), 48);
    let covered: Vec<u32> = (0..48).collect();
    let handle = h.engine.seed_adopt(fresh, &covered).unwrap();
    let mut prompt = covered.clone();
    prompt.extend(48..70);
    let p = h.prompt(&prompt);
    let ev = h.tick_ok(
        h.plan()
            .admit_with(
                |mut a| {
                    a.seed_handle = handle;
                    a
                },
                2,
                p,
            )
            .prefill(2, 48, 22)
            .decode(2, 1),
    );
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(ev.emit_for(2).n_tokens, 1);
    assert_eq!(h.engine.lane_sequence(2), Some(fresh), "the lane owns the adopted sequence");
    assert_eq!(h.engine.lane_committed(2).unwrap().len(), 71);
    assert_eq!(h.engine.free_sequence(fresh), Err(Status::UnknownHandle), "no longer bare");
}

#[test]
fn restore_refuses_corruption_wrong_kind_and_foreign_runtime() {
    let mut h = harness();
    let seq = seeded_seq(&mut h);
    let (bytes, gen) = h.engine.space_export_size(seq, KV, range(0, 16), encoding::LOSSLESS).unwrap();
    let op = h.engine.space_snapshot(seq, KV, 16, bytes, gen).unwrap();
    let payload = h.engine.take_op_output(op).unwrap();
    let fresh = h.engine.create_sequence().unwrap();

    let mut corrupt = payload.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0xFF;
    assert_eq!(h.engine.space_restore(fresh, KV, &corrupt), Err(Status::Checksum));
    assert_eq!(h.engine.space_restore(fresh, KV, &payload[..64]), Err(Status::EnvelopeMismatch));
    assert_eq!(h.engine.space_restore(fresh, 2, &payload), Err(Status::UnknownHandle));
    assert_eq!(h.engine.primitives().contents(fresh).unwrap().len(), 0);

    let foreign = {
        let mut hh = with(FakeConfig { identity: 0x11, ..Default::default() });
        let s = seeded_seq(&mut hh);
        let (b, g) = hh.engine.space_export_size(s, KV, range(0, 16), encoding::LOSSLESS).unwrap();
        let op = hh.engine.space_snapshot(s, KV, 16, b, g).unwrap();
        hh.engine.take_op_output(op).unwrap()
    };
    assert_eq!(h.engine.space_restore(fresh, KV, &foreign), Err(Status::IdentityMismatch));
    assert!(h.engine.space_restore(fresh, KV, &payload).is_ok());
    assert_eq!(h.engine.primitives().contents(fresh).unwrap().len(), 16);
}

#[test]
fn trim_and_fork_rules_by_runtime_kind() {
    let mut h = harness();
    let seq = seeded_seq(&mut h);
    assert_eq!(h.engine.space_trim(seq, KV, 10), Err(Status::Busy), "a lane's sequence is never trimmed under it");
    let child = h.engine.seq_fork(seq, 0).unwrap();
    assert_eq!(h.engine.primitives().contents(child).unwrap().len(), 63);
    assert!(h.engine.space_trim(child, KV, 10).is_ok());
    assert_eq!(h.engine.primitives().contents(child).unwrap().len(), 10);
    assert_eq!(h.engine.space_trim(child, KV, 999), Err(Status::RejectBounds));
    assert!(h.engine.free_sequence(child).is_ok());

    let mut h = with(FakeConfig { truncate_partial: false, page: 1, ..Default::default() });
    let seq = seeded_seq(&mut h);
    let child = h.engine.seq_fork(seq, 0).unwrap();
    assert_eq!(h.engine.space_trim(child, KV, 10), Err(Status::OutOfBoundary));
    assert!(h.engine.space_trim(child, KV, 63).is_ok(), "the head is always a boundary");
    assert_eq!(h.engine.space_snapshot_boundary(child, KV, 40), Ok(0), "no boundary below the head");
    assert_eq!(h.engine.space_snapshot_boundary(child, KV, 63), Ok(63));
    let (bytes, gen) = h.engine.space_export_size(child, KV, range(0, 63), encoding::LOSSLESS).unwrap();
    assert_eq!(h.engine.space_snapshot(child, KV, 40, bytes, gen), Err(Status::OutOfBoundary));
    assert!(h.engine.space_snapshot(child, KV, 63, bytes, gen).is_ok());
}

#[test]
fn ops_complete_at_submission_and_done_stands() {
    let mut h = harness();
    let seq = seeded_seq(&mut h);
    let (bytes, gen) = h.engine.space_export_size(seq, KV, range(0, 16), encoding::LOSSLESS).unwrap();
    let op = h.engine.space_demote(seq, KV, range(0, 16), encoding::LOSSLESS, bytes, gen).unwrap();
    assert!(h.engine.op_cancel(op).is_ok());
    assert_eq!(h.engine.op_poll(op).unwrap().state, op_state::DONE);
    assert_eq!(h.engine.op_poll(9999), Err(Status::UnknownHandle));
    assert_eq!(h.engine.space_promote(seq, KV, range(0, 16), &[]), Err(Status::Unsupported));
}

fn sealed(h: &Harness<Executor<FakePrimitives>>, tokens: &[u32], range_end: u64, taint: u32) -> Vec<u8> {
    let payload: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    superfluid_executor::envelope::seal(superfluid_executor::envelope::SealArgs {
        kind: space_kind::PAGED_TOKEN_KV,
        version_tag: 1,
        compat: h.engine.compat_identity(),
        provenance: superfluid_fingerprint::content_digest(tokens),
        taint_bits: taint,
        range: range(0, range_end),
        encoding: encoding::LOSSLESS,
        payload: &payload,
    })
}

fn op_result(h: &mut Harness<Executor<FakePrimitives>>, op: u64) -> (u8, u32) {
    let st = h.engine.op_poll(op).unwrap();
    (st.state, st.error)
}

fn exported(h: &mut Harness<Executor<FakePrimitives>>, seq: u64, len: u64) -> StateEnvelope {
    let (bytes, gen) = h.engine.space_export_size(seq, KV, range(0, len), encoding::LOSSLESS).unwrap();
    let op = h.engine.space_snapshot(seq, KV, len, bytes, gen).unwrap();
    let sealed = h.engine.take_op_output(op).unwrap();
    let compat = h.engine.compat_identity();
    superfluid_executor::envelope::open(&sealed, space_kind::PAGED_TOKEN_KV, 1, &compat).unwrap().0
}

#[test]
fn an_adopt_lease_is_exclusive() {
    let mut h = harness();
    let tokens: Vec<u32> = (0..16).collect();
    let seq = h.engine.create_sequence().unwrap();
    let op = h.engine.space_restore(seq, KV, &sealed(&h, &tokens, 16, 0)).unwrap();
    assert_eq!(op_result(&mut h, op).0, op_state::DONE);
    let lease = h.engine.seed_adopt(seq, &tokens).unwrap();
    assert_eq!(h.engine.seed_adopt(seq, &tokens), Err(Status::Busy));
    assert_eq!(h.engine.free_sequence(seq), Err(Status::Busy));
    assert_eq!(h.engine.space_trim(seq, KV, 8), Err(Status::Busy));
    assert_eq!(h.engine.publish_sequence(seq, &tokens), Err(Status::Busy));
    h.engine.seed_release(lease).unwrap();
    let again = h.engine.seed_adopt(seq, &tokens).unwrap();
    assert_ne!(again, lease);
    h.engine.seed_release(again).unwrap();
    h.engine.free_sequence(seq).unwrap();
}

#[test]
fn a_whole_prompt_adoption_decodes_or_is_refused_typed() {
    let tokens: Vec<u32> = (0..16).collect();
    let mut h = harness();
    let seq = h.engine.create_sequence().unwrap();
    h.engine.space_restore(seq, KV, &sealed(&h, &tokens, 16, 0)).unwrap();
    let lease = h.engine.seed_adopt(seq, &tokens).unwrap();
    let p = h.prompt(&tokens);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p).decode(2, 1));
    assert_eq!(ev.emit_for(2).n_tokens, 1);
    assert_eq!(h.engine.lane_committed(2).unwrap().len(), 17);

    let mut h = with(FakeConfig { truncate_partial: false, page: 1, ..Default::default() });
    let seq = h.engine.create_sequence().unwrap();
    h.engine.space_restore(seq, KV, &sealed(&h, &tokens, 16, 0)).unwrap();
    let lease = h.engine.seed_adopt(seq, &tokens).unwrap();
    let p = h.prompt(&tokens);
    let err = h.tick_err(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p).decode(2, 1));
    assert_eq!(err, Status::RejectIllegalCombination);
    let mut longer = tokens.clone();
    longer.push(16);
    let p = h.prompt(&longer);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p).prefill(2, 16, 1).decode(2, 1));
    assert_eq!(ev.emit_for(2).n_tokens, 1);
}

#[test]
fn adoption_allocates_no_spare_sequence() {
    let mut h = with(FakeConfig { max_seqs: 1, ..Default::default() });
    let tokens: Vec<u32> = (0..16).collect();
    let seq = h.engine.create_sequence().unwrap();
    assert_eq!(h.engine.create_sequence(), Err(Status::NeedsReplan), "the table is full");
    h.engine.space_restore(seq, KV, &sealed(&h, &tokens, 16, 0)).unwrap();
    let lease = h.engine.seed_adopt(seq, &tokens).unwrap();
    let mut longer = tokens.clone();
    longer.push(16);
    let p = h.prompt(&longer);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p).prefill(2, 16, 1).decode(2, 1));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(h.engine.lane_sequence(2), Some(seq));
}

#[test]
fn restore_refuses_a_range_past_the_runtime_limit() {
    let mut h = with(FakeConfig { max_seq_len: 64, ..Default::default() });
    let seq = h.engine.create_sequence().unwrap();
    let big: Vec<u32> = (0..65).collect();
    let op = h.engine.space_restore(seq, KV, &sealed(&h, &big, 65, 0)).unwrap();
    assert_eq!(op_result(&mut h, op), (op_state::FAILED, Status::RejectBounds.raw() as u32));
    assert_eq!(h.engine.primitives().contents(seq).map(|c| c.len()), Some(0));
    let ok: Vec<u32> = (0..16).collect();
    let op = h.engine.space_restore(seq, KV, &sealed(&h, &ok, 16, 0)).unwrap();
    assert_eq!(op_result(&mut h, op).0, op_state::DONE);
}

#[test]
fn a_failed_restore_leaves_the_sequence_empty_on_recurrent_memory() {
    let mut h = with(FakeConfig { truncate_partial: false, page: 1, ..Default::default() });
    let seq = h.engine.create_sequence().unwrap();
    let eight: Vec<u32> = (0..8).collect();
    let op = h.engine.space_restore(seq, KV, &sealed(&h, &eight, 10, 0)).unwrap();
    assert_eq!(op_result(&mut h, op), (op_state::FAILED, Status::EnvelopeMismatch.raw() as u32));
    assert_eq!(h.engine.primitives().contents(seq).map(|c| c.len()), Some(0), "rolled back");
    let op = h.engine.space_restore(seq, KV, &sealed(&h, &eight, 8, 0)).unwrap();
    assert_eq!(op_result(&mut h, op).0, op_state::DONE, "the empty sequence takes the next restore");
}

#[test]
fn sizings_of_one_generation_stay_valid_together() {
    let mut h = harness();
    let seq = seeded_seq(&mut h);
    let (b1, g1) = h.engine.space_export_size(seq, KV, range(0, 32), encoding::LOSSLESS).unwrap();
    let (b2, g2) = h.engine.space_export_size(seq, KV, range(0, 48), encoding::LOSSLESS).unwrap();
    assert_eq!(g1, g2);
    let op = h.engine.space_snapshot(seq, KV, 32, b1, g1).unwrap();
    assert_eq!(op_result(&mut h, op).0, op_state::DONE, "the first sizing survived the second");
    let op = h.engine.space_snapshot(seq, KV, 48, b2, g2).unwrap();
    assert_eq!(op_result(&mut h, op).0, op_state::DONE);
}

#[test]
fn imported_taint_survives_re_export_and_forks() {
    let mut h = harness();
    let tokens: Vec<u32> = (0..16).collect();
    let seq = h.engine.create_sequence().unwrap();
    let op = h.engine.space_restore(seq, KV, &sealed(&h, &tokens, 16, taint::QUANTIZED_DEMOTION)).unwrap();
    assert_eq!(op_result(&mut h, op).0, op_state::DONE);
    assert_ne!(exported(&mut h, seq, 16).taint_bits & taint::QUANTIZED_DEMOTION, 0);
    let child = h.engine.seq_fork(seq, 0).unwrap();
    assert_ne!(exported(&mut h, child, 16).taint_bits & taint::QUANTIZED_DEMOTION, 0);
    let clean = h.engine.create_sequence().unwrap();
    h.engine.space_restore(clean, KV, &sealed(&h, &tokens, 16, 0)).unwrap();
    assert_eq!(exported(&mut h, clean, 16).taint_bits, 0);
}

#[test]
fn trim_refuses_a_cached_sequence() {
    let mut h = harness();
    let seq = seeded_seq(&mut h);
    h.tick_ok(h.plan().retire(1, true));
    assert_eq!(h.engine.space_trim(seq, KV, 8), Err(Status::Busy));
}

#[test]
fn the_paged_boundary_is_page_aligned() {
    let mut h = harness();
    let seq = seeded_seq(&mut h);
    assert_eq!(h.engine.space_snapshot_boundary(seq, KV, 39).unwrap(), 32);
    assert_eq!(h.engine.space_snapshot_boundary(seq, KV, 100).unwrap(), 48);
    let mut h = with(FakeConfig { truncate_partial: false, page: 1, ..Default::default() });
    let seq = seeded_seq(&mut h);
    assert_eq!(h.engine.space_snapshot_boundary(seq, KV, 100).unwrap(), 63);
    assert_eq!(h.engine.space_snapshot_boundary(seq, KV, 39).unwrap(), 0);
}

#[test]
fn a_fork_carries_its_parents_provenance() {
    let mut h = harness();
    let seq = seeded_seq(&mut h);
    let child = h.engine.seq_fork(seq, 0).unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    let env = exported(&mut h, child, 32);
    assert_eq!(env.provenance_digest, superfluid_fingerprint::content_digest(&tokens[..32]));
    assert_ne!(env.provenance_digest, [0u8; 32]);
}

#[test]
fn adopted_seeds_are_not_charged_a_copy() {
    let mut h = with(FakeConfig { cells_total: 32, ..Default::default() });
    let tokens: Vec<u32> = (0..16).collect();
    let seq = h.engine.create_sequence().unwrap();
    h.engine.space_restore(seq, KV, &sealed(&h, &tokens, 16, 0)).unwrap();
    let lease = h.engine.seed_adopt(seq, &tokens).unwrap();
    let mut longer = tokens.clone();
    longer.push(16);
    let p = h.prompt(&longer);
    let ev = h.tick_ok(h.plan().admit_with(|mut a| { a.seed_handle = lease; a }, 2, p).prefill(2, 16, 1).decode(2, 1));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
}

#[test]
fn match_candidates_carry_their_entrys_taint() {
    let mut h = harness();
    let tokens: Vec<u32> = (0..16).collect();
    let seq = h.engine.create_sequence().unwrap();
    h.engine.space_restore(seq, KV, &sealed(&h, &tokens, 16, taint::QUANTIZED_DEMOTION)).unwrap();
    h.engine.publish_sequence(seq, &tokens).unwrap();
    let mut longer = tokens.clone();
    longer.push(16);
    let m = h.engine.space_match(ALL_SPACES, &longer, &[]).unwrap();
    let bounds = ArenaBounds { base: 0, len: usize::MAX };
    // SAFETY: engine-owned result, next-call lifetime.
    let spaces: Vec<SpaceMatch> = unsafe { superfluid_abi::array::read_array(&m.spaces, &bounds) }.unwrap().collect();
    // SAFETY: as above.
    let cands: Vec<MatchCandidate> =
        unsafe { superfluid_abi::array::read_array(&spaces[0].candidates, &bounds) }.unwrap().collect();
    assert_eq!(cands.len(), 1);
    assert_eq!(cands[0].prefix_len, 16);
    assert_ne!(cands[0].taint_bits & taint::QUANTIZED_DEMOTION, 0);
}

#[test]
fn a_replacement_restore_replaces_the_metadata() {
    let mut h = harness();
    let first: Vec<u32> = (0..16).collect();
    let second: Vec<u32> = (100..116).collect();
    let seq = h.engine.create_sequence().unwrap();
    h.engine.space_restore(seq, KV, &sealed(&h, &first, 16, taint::QUANTIZED_DEMOTION)).unwrap();
    let lease = h.engine.seed_adopt(seq, &first).unwrap();
    h.engine.seed_release(lease).unwrap();
    h.engine.space_trim(seq, KV, 0).unwrap();
    let op = h.engine.space_restore(seq, KV, &sealed(&h, &second, 16, 0)).unwrap();
    assert_eq!(op_result(&mut h, op).0, op_state::DONE);
    let env = exported(&mut h, seq, 16);
    assert_eq!(env.taint_bits, 0, "the old taint is gone");
    assert_ne!(env.provenance_digest, superfluid_fingerprint::content_digest(&first), "the old lineage is gone");
}

#[test]
fn a_refused_import_leaves_the_sequence_empty() {
    let mut h = harness();
    let tokens: Vec<u32> = (0..16).collect();
    let seq = h.engine.create_sequence().unwrap();
    h.engine.primitives_mut().poison_next_import();
    let op = h.engine.space_restore(seq, KV, &sealed(&h, &tokens, 16, 0)).unwrap();
    assert_eq!(op_result(&mut h, op).0, op_state::FAILED);
    assert_eq!(h.engine.primitives().contents(seq).map(|c| c.len()), Some(0), "rolled back");
    let op = h.engine.space_restore(seq, KV, &sealed(&h, &tokens, 16, 0)).unwrap();
    assert_eq!(op_result(&mut h, op).0, op_state::DONE);
}

#[test]
fn zero_length_exports_are_refused() {
    let mut h = harness();
    let seq = seeded_seq(&mut h);
    assert_eq!(h.engine.space_export_size(seq, KV, range(0, 0), encoding::LOSSLESS), Err(Status::RejectBounds));
    let (bytes, gen) = h.engine.space_export_size(seq, KV, range(0, 16), encoding::LOSSLESS).unwrap();
    assert_eq!(h.engine.space_snapshot(seq, KV, 0, bytes, gen), Err(Status::RejectBounds));
}

#[test]
fn exact_only_publication_needs_the_whole_sequence() {
    let mut h = with(FakeConfig { truncate_partial: false, page: 1, ..Default::default() });
    let tokens: Vec<u32> = (0..16).collect();
    let seq = h.engine.create_sequence().unwrap();
    h.engine.space_restore(seq, KV, &sealed(&h, &tokens, 16, 0)).unwrap();
    assert_eq!(h.engine.publish_sequence(seq, &tokens[..8]), Err(Status::OutOfBoundary));
    h.engine.publish_sequence(seq, &tokens).unwrap();
    let mut longer = tokens.clone();
    longer.push(16);
    assert_eq!(h.engine.match_lengths(&longer), vec![16]);
}

#[test]
fn slot_pressure_is_kept_inside_the_plan() {
    let mut h = with(FakeConfig { max_seqs: 2, ..Default::default() });
    let a: Vec<u32> = (0..32).collect();
    let pa = h.prompt(&a);
    h.tick_ok(h.plan().admit(1, pa).prefill(1, 0, 32));
    h.tick_ok(h.plan().retire(1, true));
    let b: Vec<u32> = (100..132).collect();
    let pb = h.prompt(&b);
    h.tick_ok(h.plan().admit(2, pb).prefill(2, 0, 32));
    let c: Vec<u32> = (200..232).collect();
    let pc = h.prompt(&c);
    assert_eq!(h.tick_err(h.plan().evict_envelope(u64::MAX, 0).admit(3, pc).prefill(3, 0, 32)), Status::NeedsReplan);
    assert_eq!(h.engine.live_sequences(), 2);
    let pc = h.prompt(&c);
    let ev = h.tick_ok(h.plan().admit(3, pc).prefill(3, 0, 32));
    assert_eq!(ev.tick_status, tick_status::SHED);
    assert_eq!(ev.shed_entries.iter().filter(|e| e.kind == shed_kind::CACHE_EVICTION).count(), 1);
    assert_eq!(h.engine.live_sequences(), 2);
    let d: Vec<u32> = (300..332).collect();
    let pd = h.prompt(&d);
    let ev = h.tick_ok(h.plan().retire(2, false).admit(4, pd).prefill(4, 0, 32));
    assert_eq!(ev.tick_status, tick_status::OK);
}
