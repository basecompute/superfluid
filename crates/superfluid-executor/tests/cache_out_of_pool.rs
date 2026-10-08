//! A runtime whose steps cost by what its pool holds (`cache_resident_cells`:
//! llama.cpp's shared KV pool runs every step's attention over the cells up
//! to the highest one in use, whoever holds them) keeps only that much of
//! the cache in the pool, and a runtime that gives each lane a buffer of its
//! own shares its table of sequences between the lanes and the cache. Either
//! way the entries that leave the pool are held as exports (up to
//! `cache_exported_cells`), and still served: a request that seeds from one
//! imports it into its own sequence.
//!
//! The fake counts, for every step, the cells the pool held for sequences
//! the step did not feed (`beside`): what a shared pool pays for.

use superfluid_abi::*;
use superfluid_engine::testing::{Events, Harness};
use superfluid_engine::Engine;
use superfluid_executor::cache::{HOST_EXPORT_CLASS, RESIDENT_PREFIX_CLASS};
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives, RuntimePrimitives};

type H = Harness<Executor<FakePrimitives>>;

fn with(cfg: FakeConfig) -> H {
    Harness::with_engine(Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()))
}

/// A runtime whose copies share cells, with one-token pages and one-byte
/// cells, that affords the cache `resident` cells of its pool and `exported`
/// tokens of exports.
fn pool(resident: u64, exported: u64) -> FakeConfig {
    FakeConfig {
        page: 1,
        kv_bytes_per_token: 1,
        copy_shares_cells: true,
        cache_resident_cells: resident,
        cache_exported_cells: exported,
        ..Default::default()
    }
}

/// Run `tokens` through a lane and leave them in the cache.
fn publish(h: &mut H, lane: u64, tokens: &[u32]) -> Events {
    let p = h.prompt(tokens);
    h.tick_ok(h.plan().admit(lane, p).prefill(lane, 0, tokens.len() as u32));
    h.tick_ok(h.plan().retire(lane, true))
}

/// Admit `prompt` on a seed of its first `seed` tokens, prefill the rest and
/// decode `n`: the tick's events.
fn seeded(h: &mut H, lane: u64, prompt: &[u32], seed: u64, n: u16) -> Events {
    let lease = h.engine.seed_acquire(prompt, seed, determinism::BEST_EFFORT).expect("a lease on the cached prefix");
    let p = h.prompt(prompt);
    let rest = prompt.len() as u32 - seed as u32;
    let plan = h.plan().admit_with(
        |mut a| {
            a.seed_handle = lease;
            a
        },
        lane,
        p,
    );
    h.tick_ok(plan.prefill(lane, seed as u32, rest).decode(lane, n))
}

/// What a lane that prefills `prompt` cold decodes.
fn cold(cfg: FakeConfig, prompt: &[u32], n: u16) -> Vec<u32> {
    let mut h = with(cfg);
    let p = h.prompt(prompt);
    let ev = h.tick_ok(h.plan().admit(1, p).prefill(1, 0, prompt.len() as u32).decode(1, n));
    h.rings_tokens(&ev.emit_for(1).token_ref)
}

fn cells_used(h: &H) -> u64 {
    h.engine.primitives().mem_counters().cells_used
}

/// The length the cache offers for a prompt that continues `tokens`.
fn offered(h: &H, tokens: &[u32]) -> Vec<u64> {
    let mut next = tokens.to_vec();
    next.push(7);
    h.engine.match_lengths(&next)
}

fn left(ev: &Events) -> Vec<(u32, u64, u64)> {
    ev.shed_entries.iter().map(|e| (e.kind, e.bytes, e.tokens)).collect()
}

fn prompts(n: u32) -> Vec<Vec<u32>> {
    (0..n).map(|k| (100 * (k + 1)..100 * (k + 1) + 6).collect()).collect()
}

/// Four six-token prompts answered one after another leave 24 cells of
/// cache. A request that shares nothing with them then stepped beside all
/// 24, on a runtime that affords 8: every one of its steps paid for a cache
/// it made no use of, and the more the cache held the slower it ran. The
/// three entries past the bound leave the pool as they are published, the
/// lane steps beside the one that is left, and all four prompts are still
/// served.
#[test]
fn a_lane_steps_beside_no_more_of_the_cache_than_the_runtime_affords() {
    let mut h = with(pool(8, 64));
    let prompts = prompts(4);
    for (k, t) in prompts.iter().enumerate() {
        publish(&mut h, 1 + k as u64, t);
    }
    let mark = h.engine.primitives().beside.len();
    let p = h.prompt(&[900, 901, 902]);
    h.tick_ok(h.plan().admit(9, p).prefill(9, 0, 3).decode(9, 4));
    let beside = h.engine.primitives().beside[mark..].to_vec();
    assert_eq!(beside.len(), 4, "the prefill and three decode steps");
    assert!(beside.iter().all(|&cells| cells <= 8), "each step ran beside at most the 8 cells the runtime affords: {beside:?}");
    for t in &prompts {
        assert_eq!(offered(&h, t), vec![6], "the cache still serves {t:?}");
    }
    assert_eq!((h.engine.cache_bytes(), h.engine.cache_exported_tokens()), (6, 18), "one entry in the pool, three exported");
}

/// An entry out of the pool seeds what it seeded in it: a lane that takes
/// its first six tokens from the export continues as one that prefilled the
/// whole prompt cold, in cells of its own, and a seed shorter than the entry
/// is the entry imported and cut.
#[test]
fn an_entry_out_of_the_pool_seeds_what_it_seeded_in_it() {
    let turn: Vec<u32> = (10..16).collect();
    let other: Vec<u32> = (200..206).collect();
    let longer: Vec<u32> = turn.iter().copied().chain([40, 41]).collect();
    let mut h = with(pool(8, 64));
    publish(&mut h, 1, &turn);
    publish(&mut h, 2, &other);
    assert_eq!((h.engine.cache_bytes(), h.engine.cache_exported_tokens()), (6, 6), "the older entry left the pool");
    assert_eq!(h.engine.match_lengths(&longer), vec![6], "and is still offered");
    let ev = seeded(&mut h, 3, &longer, 6, 3);
    assert_eq!(ev.admit_for(3).status, admit_status::ADMITTED);
    assert_eq!(h.rings_tokens(&ev.emit_for(3).token_ref), cold(pool(8, 64), &longer, 3), "the lane continues as one that prefilled the prompt cold");
    // The entry in the pool (6), and the lane's own: six imported, two
    // prefilled, two of the three generated ingested.
    assert_eq!(cells_used(&h), 6 + 6 + 2 + 2);
    assert_eq!(h.engine.cache_exported_tokens(), 6, "the export stays for the next request");

    // Four of the six tokens shared: the entry comes in whole and is cut.
    let partly: Vec<u32> = vec![10, 11, 12, 13, 50, 51];
    let mut h = with(pool(8, 64));
    publish(&mut h, 1, &turn);
    publish(&mut h, 2, &other);
    assert_eq!(h.engine.match_lengths(&partly), vec![4]);
    let ev = seeded(&mut h, 3, &partly, 4, 2);
    assert_eq!(h.rings_tokens(&ev.emit_for(3).token_ref), cold(pool(8, 64), &partly, 2));
    assert_eq!(cells_used(&h), 6 + 4 + 2 + 1, "four seeded cells, two prefilled, one generated token ingested");
}

/// An import is planned by its peak. The whole entry comes in and is cut to
/// the seed before the lane prefills and decodes, so the plan needs the
/// larger of the two, not their sum: in a pool of one eight-token lane, a
/// request that seeds four tokens from a six-token export, prefills two and
/// decodes two holds eight cells at most, and is admitted (planned as a sum
/// it was ten, and refused). Short of that, the entry in the pool leaves it
/// to make the room, and with nothing to move out the plan is refused
/// before anything moves.
#[test]
fn an_import_is_planned_by_its_peak() {
    let turn: Vec<u32> = (10..16).collect();
    let other: Vec<u32> = (200..206).collect();
    let partly: Vec<u32> = vec![10, 11, 12, 13, 50, 51];
    // Every entry is exported as it is published (one cell afforded).
    let mut h = with(FakeConfig { cells_total: 8, max_seq_len: 8, ..pool(1, 64) });
    publish(&mut h, 1, &turn);
    assert_eq!((cells_used(&h), h.engine.cache_exported_tokens()), (0, 6));
    let ev = seeded(&mut h, 3, &partly, 4, 2);
    assert_eq!(ev.admit_for(3).status, admit_status::ADMITTED);
    assert_eq!(h.rings_tokens(&ev.emit_for(3).token_ref), cold(pool(1, 64), &partly, 2));
    assert_eq!(cells_used(&h), 7, "four seeded cells, two prefilled, one generated token ingested");

    // Twelve cells, six of them an entry in the pool: seven are needed (the
    // seed's four, then the larger of the import's other two and the three
    // the lane adds), and the entry leaves the pool to make the room.
    let mut h = with(FakeConfig { cells_total: 12, ..pool(8, 64) });
    publish(&mut h, 1, &turn);
    publish(&mut h, 2, &other);
    assert_eq!((cells_used(&h), h.engine.cache_exported_tokens()), (6, 6));
    let ev = seeded(&mut h, 3, &partly, 4, 1);
    assert_eq!(ev.admit_for(3).status, admit_status::ADMITTED);
    assert_eq!(ev.tick_status, tick_status::OK, "nothing was shed: both entries are still cached");
    assert_eq!(h.engine.cache_exported_tokens(), 12);
    assert_eq!((offered(&h, &turn), offered(&h, &other)), (vec![6], vec![6]));

    // Seven cells, six of them a live lane's: six are needed and one is free.
    let mut h = with(FakeConfig { cells_total: 7, ..pool(8, 64) });
    publish(&mut h, 1, &turn);
    publish(&mut h, 2, &other);
    let big: Vec<u32> = (300..307).collect();
    let pb = h.prompt(&big);
    h.tick_ok(h.plan().admit(5, pb).prefill(5, 0, 7));
    assert_eq!((cells_used(&h), h.engine.cache_exported_tokens()), (6, 12), "a lane holds six cells; both entries are exports");
    let lease = h.engine.seed_acquire(&partly, 4, determinism::BEST_EFFORT).unwrap();
    let p = h.prompt(&partly);
    let plan = h.plan().admit_with(
        |mut a| {
            a.seed_handle = lease;
            a
        },
        3,
        p,
    );
    assert_eq!(h.tick_err(plan.prefill(3, 4, 2)), Status::NeedsReplan);
    assert_eq!(cells_used(&h), 6, "a refused plan moves nothing");
}

/// A seed from an export must cover at least an eighth of the entry; from an entry in the
/// pool any seed is served.
#[test]
fn a_seed_from_an_export_is_taken_when_it_is_worth_the_import() {
    let entry: Vec<u32> = (10..26).collect();
    let shares = |n: usize| -> Vec<u32> { entry[..n].iter().copied().chain([900, 901]).collect() };
    let mut h = with(pool(1, 64));
    publish(&mut h, 1, &entry);
    assert_eq!(h.engine.cache_exported_tokens(), 16);
    assert_eq!(h.engine.match_lengths(&shares(1)), Vec::<u64>::new(), "one token of sixteen is not offered");
    assert_eq!(h.engine.seed_acquire(&shares(1), 1, determinism::BEST_EFFORT), Err(Status::SeedUnservable));
    assert_eq!(h.engine.match_lengths(&shares(2)), vec![2], "an eighth of it is");
    let ev = seeded(&mut h, 2, &shares(2), 2, 1);
    assert_eq!(h.rings_tokens(&ev.emit_for(2).token_ref), cold(pool(1, 64), &shares(2), 1));

    let mut h = with(pool(64, 64));
    publish(&mut h, 1, &entry);
    assert_eq!((h.engine.cache_bytes(), h.engine.match_lengths(&shares(1))), (16, vec![1]), "in the pool, any seed is a copy");
}

/// Exports stay inside the allowance, least recently used out first; a pinned export stays,
/// and an entry larger than the allowance is evicted.
#[test]
fn exports_stay_inside_the_runtimes_allowance() {
    let p = prompts(4);
    let mut h = with(pool(6, 12));
    for (k, t) in p[..3].iter().enumerate() {
        publish(&mut h, 1 + k as u64, t);
    }
    assert_eq!((h.engine.cache_bytes(), h.engine.cache_exported_tokens()), (6, 12));
    let ev = publish(&mut h, 4, &p[3]);
    assert_eq!(h.engine.cache_exported_tokens(), 12);
    assert_eq!(offered(&h, &p[0]), Vec::<u64>::new(), "the oldest export made room");
    for t in &p[1..] {
        assert_eq!(offered(&h, t), vec![6]);
    }
    assert_eq!(ev.tick_status, tick_status::SHED);
    assert_eq!(left(&ev), vec![(shed_kind::CACHE_EVICTION, 0, 6)], "an entry left the cache, and no pool bytes with it");

    // A lease on the oldest export: the next oldest goes instead.
    let mut h = with(pool(6, 12));
    for (k, t) in p[..3].iter().enumerate() {
        publish(&mut h, 1 + k as u64, t);
    }
    let mut next = p[0].clone();
    next.push(7);
    let lease = h.engine.seed_acquire(&next, 6, determinism::BEST_EFFORT).expect("a lease on the oldest export");
    publish(&mut h, 4, &p[3]);
    assert_eq!((offered(&h, &p[0]), offered(&h, &p[1])), (vec![6], vec![]));
    h.engine.seed_release(lease).unwrap();

    // Room for four tokens of exports: a six-token entry is not kept.
    let mut h = with(pool(6, 4));
    publish(&mut h, 1, &p[0]);
    let ev = publish(&mut h, 2, &p[1]);
    assert_eq!((offered(&h, &p[0]), h.engine.cache_exported_tokens()), (vec![], 0));
    assert_eq!(left(&ev), vec![(shed_kind::CACHE_EVICTION, 6, 6)]);
}

/// An export that has seeded several lanes holds a prefix requests share,
/// and is worth more than one that has seeded none, as an entry in the pool
/// is: making room for a newer export takes the export that seeded nothing,
/// though it is the more recently used.
#[test]
fn an_export_requests_share_outlives_newer_ones_that_seeded_nothing() {
    let shared: Vec<u32> = (10..16).collect();
    let p = prompts(2);
    let mut h = with(pool(1, 12));
    publish(&mut h, 1, &shared);
    for lane in [2u64, 3] {
        let next: Vec<u32> = shared.iter().copied().chain([50 + lane as u32]).collect();
        seeded(&mut h, lane, &next, 6, 1);
        h.tick_ok(h.plan().retire(lane, false));
    }
    publish(&mut h, 4, &p[0]);
    assert_eq!(h.engine.cache_exported_tokens(), 12, "the shared prefix and the first prompt fill the room");
    let ev = publish(&mut h, 5, &p[1]);
    assert_eq!(left(&ev), vec![(shed_kind::CACHE_EVICTION, 0, 6)]);
    assert_eq!(offered(&h, &shared), vec![6], "the prefix two requests shared is still served");
    assert_eq!((offered(&h, &p[0]), offered(&h, &p[1])), (vec![], vec![6]), "the export that seeded nothing made the room");
}

/// A runtime with no room for exports evicts what its pool cannot afford:
/// the lane still steps beside no more than the bound, and each eviction is
/// in the shed report of the tick that made it.
#[test]
fn with_no_room_for_exports_the_entries_past_the_bound_are_evicted() {
    let mut h = with(pool(8, 0));
    let p = prompts(3);
    publish(&mut h, 1, &p[0]);
    let ev = publish(&mut h, 2, &p[1]);
    assert_eq!(ev.tick_status, tick_status::SHED);
    assert_eq!(left(&ev), vec![(shed_kind::CACHE_EVICTION, 6, 6)]);
    publish(&mut h, 3, &p[2]);
    assert_eq!((h.engine.cache_bytes(), h.engine.cache_exported_tokens(), cells_used(&h)), (6, 0, 6));
    assert_eq!((offered(&h, &p[0]), offered(&h, &p[1]), offered(&h, &p[2])), (vec![], vec![], vec![6]));
}

/// A conversation's turns share cells; past the bound the earlier turns are dropped, not
/// exported, since the last turn covers them.
#[test]
fn a_conversations_earlier_turns_are_dropped_not_exported() {
    let turn1: Vec<u32> = (10..16).collect();
    let turn2: Vec<u32> = (10..19).collect();
    let turn3: Vec<u32> = (10..22).collect();
    let mut h = with(pool(10, 64));
    publish(&mut h, 1, &turn1);
    seeded(&mut h, 2, &turn2, 6, 0);
    h.tick_ok(h.plan().retire(2, true));
    assert_eq!((cells_used(&h), h.engine.cache_exported_tokens()), (9, 0), "two turns in nine cells, inside the ten afforded");
    seeded(&mut h, 3, &turn3, 9, 0);
    let ev = h.tick_ok(h.plan().retire(3, true));
    assert_eq!((cells_used(&h), h.engine.cache_exported_tokens()), (0, 12), "one export, of the whole conversation");
    assert_eq!(left(&ev), vec![(shed_kind::CACHE_EVICTION, 6, 6), (shed_kind::CACHE_EVICTION, 9, 9)], "the two earlier turns left the cache");
    let next: Vec<u32> = (10..23).collect();
    assert_eq!(h.engine.match_lengths(&next), vec![12]);
    // The export serves a seed at an earlier turn's length as well.
    let branch: Vec<u32> = turn1.iter().copied().chain([77, 78]).collect();
    let ev = seeded(&mut h, 4, &branch, 6, 2);
    assert_eq!(h.rings_tokens(&ev.emit_for(4).token_ref), cold(pool(10, 64), &branch, 2));
}

/// A reply is not always rendered in the history as it was generated (Qwen3's
/// template drops the empty think block), so a conversation's turn and the
/// next turn's stream part ways a few tokens before the turn ends, and
/// neither begins with the other. Each turn would then be an export of its
/// own, pushing out entries that nothing else covers. A turn that a later
/// one covers all but an eighth of is dropped; an entry that shares less
/// with another is kept.
#[test]
fn a_turn_that_a_later_one_all_but_covers_is_not_kept_twice() {
    let turn1: Vec<u32> = (10..26).collect();
    let turn2: Vec<u32> = (10..25).chain([70, 71, 72]).collect();
    let other: Vec<u32> = (10..18).chain(200..208).collect();
    let mut h = with(pool(1, 64));
    publish(&mut h, 1, &turn1);
    assert_eq!(h.engine.cache_exported_tokens(), 16);
    let ev = publish(&mut h, 2, &turn2);
    assert_eq!(h.engine.cache_exported_tokens(), 18, "the later turn's export stands for both");
    assert_eq!(left(&ev), vec![(shed_kind::CACHE_EVICTION, 0, 16)], "the earlier one left the cache, and no pool bytes with it");
    assert_eq!(offered(&h, &turn1), vec![15], "and serves the earlier turn up to where they part");
    publish(&mut h, 3, &other);
    assert_eq!(h.engine.cache_exported_tokens(), 34, "an entry that shares half of another is its own");
}

/// An entry is dropped for one that begins with it only when that one would
/// serve its seeds out of the pool: a six-token prompt beside a 56-token
/// stream that begins with it stays, because a six-token seed is less than
/// an eighth of the longer entry, and from an export that is not served.
#[test]
fn a_short_entry_is_kept_beside_a_long_one_that_begins_with_it() {
    let short: Vec<u32> = (10..16).collect();
    let long: Vec<u32> = (10..66).collect();
    let mut h = with(pool(1, 64));
    publish(&mut h, 1, &short);
    publish(&mut h, 2, &long);
    assert_eq!(h.engine.cache_exported_tokens(), 62, "both are exports");
    assert_eq!(offered(&h, &short), vec![6], "and the short one still serves its own prompt");
}

/// The few tokens every prompt shares with every entry are no use of an
/// entry. A one-token seed copied from the oldest entry made it the most
/// recently used, and the entry after it left the pool in its place.
#[test]
fn a_seed_of_a_few_tokens_is_no_use_of_its_entry() {
    let old: Vec<u32> = (10..26).collect();
    let newer: Vec<u32> = (100..112).collect();
    let mut h = with(pool(32, 64));
    publish(&mut h, 1, &old);
    publish(&mut h, 2, &newer);
    seeded(&mut h, 3, &[10, 900, 901], 1, 1);
    h.tick_ok(h.plan().retire(3, false));
    publish(&mut h, 4, &(300..308).collect::<Vec<u32>>());
    assert_eq!(
        (h.engine.cache_bytes(), h.engine.cache_exported_tokens()),
        (20, 16),
        "36 cells are past the 32 afforded: the oldest entry leaves the pool, not the one after it"
    );
}

/// An export that does not come back is state lost, not a request lost: the
/// lane computes its seed afresh and continues as a cold one does, and the
/// entry is gone from the cache.
#[test]
fn a_seed_whose_export_does_not_come_back_is_computed_afresh() {
    let turn: Vec<u32> = (10..16).collect();
    let longer: Vec<u32> = turn.iter().copied().chain([40, 41]).collect();
    let mut h = with(pool(8, 64));
    publish(&mut h, 1, &turn);
    publish(&mut h, 2, &(200..206).collect::<Vec<u32>>());
    h.engine.primitives_mut().poison_next_import();
    let ev = seeded(&mut h, 3, &longer, 6, 3);
    assert_eq!(ev.admit_for(3).status, admit_status::ADMITTED);
    assert_eq!(h.rings_tokens(&ev.emit_for(3).token_ref), cold(pool(8, 64), &longer, 3));
    assert_eq!(h.engine.match_lengths(&longer), Vec::<u64>::new(), "nothing seeds from that export again");
    assert_eq!(h.engine.cache_exported_tokens(), 0);
    let left: Vec<(u64, u32, u64, u64)> = ev.shed_entries.iter().map(|e| (e.lane_tag, e.kind, e.bytes, e.tokens)).collect();
    assert_eq!(left, vec![(3, shed_kind::CACHE_EVICTION, 0, 6)], "and the tick says the entry left the cache");
}

/// Room for a lane is made by exporting the oldest entry, not evicting it; `cache_evict`
/// frees pool bytes the same way.
#[test]
fn room_in_the_pool_is_made_by_exporting() {
    let p = prompts(2);
    let mut h = with(FakeConfig { cells_total: 16, ..pool(12, 64) });
    publish(&mut h, 1, &p[0]);
    publish(&mut h, 2, &p[1]);
    assert_eq!((cells_used(&h), h.engine.cache_exported_tokens()), (12, 0), "both entries fit the pool's allowance");
    let prompt: Vec<u32> = (900..908).collect();
    let pp = h.prompt(&prompt);
    let ev = h.tick_ok(h.plan().admit(3, pp).prefill(3, 0, 8));
    assert_eq!(ev.admit_for(3).status, admit_status::ADMITTED);
    assert_eq!(ev.tick_status, tick_status::OK, "the entry that made the room is still cached");
    assert_eq!((h.engine.cache_bytes(), h.engine.cache_exported_tokens()), (6, 6));
    assert_eq!((offered(&h, &p[0]), offered(&h, &p[1])), (vec![6], vec![6]));

    let mut h = with(pool(12, 64));
    publish(&mut h, 1, &p[0]);
    publish(&mut h, 2, &p[1]);
    assert_eq!(h.engine.cache_evict(RESIDENT_PREFIX_CLASS, 0, u64::MAX, 6), Ok(6), "six pool bytes freed");
    assert_eq!((h.engine.cache_bytes(), h.engine.cache_exported_tokens(), cells_used(&h)), (6, 6, 6));
    assert_eq!((offered(&h, &p[0]), offered(&h, &p[1])), (vec![6], vec![6]));
}

/// When the machine is short of memory the daemon names the exports' class
/// in a relief round, and they go: they free nothing in the pool, and only
/// the machine's pressure takes them, never the pool's. For a while after,
/// an entry that leaves the pool is evicted rather than kept in host memory.
#[test]
fn memory_short_on_the_machine_takes_the_exports() {
    let p = prompts(3);
    let mut h = with(pool(6, 64));
    publish(&mut h, 1, &p[0]);
    publish(&mut h, 2, &p[1]);
    assert_eq!(h.engine.cache_exported_tokens(), 6);
    assert_eq!(h.engine.cache_evict(RESIDENT_PREFIX_CLASS, 0, u64::MAX, 6), Ok(6), "the pool's pressure");
    assert_eq!((h.engine.cache_bytes(), h.engine.cache_exported_tokens()), (0, 12), "exports the entry that leaves it");
    assert_eq!(h.engine.cache_evict(HOST_EXPORT_CLASS, 0, 0, 0), Ok(0), "the machine's frees no pool bytes");
    assert_eq!(h.engine.cache_exported_tokens(), 0, "and takes the exports");
    assert_eq!((offered(&h, &p[0]), offered(&h, &p[1])), (vec![], vec![]));
    publish(&mut h, 3, &p[2]);
    let ev = publish(&mut h, 4, &p[0]);
    assert_eq!(left(&ev), vec![(shed_kind::CACHE_EVICTION, 6, 6)], "an entry past the bound is evicted");
    assert_eq!(h.engine.cache_exported_tokens(), 0);
}

/// With a buffer per lane, an entry that gives its slot to a new lane is exported, and a
/// later seed imports it into its own slot.
#[test]
fn an_entry_that_gives_up_its_slot_is_kept_as_its_export() {
    let cfg = || FakeConfig { max_seqs: 2, page: 1, kv_bytes_per_token: 1, takeover_preferred: true, cache_exported_cells: 64, ..Default::default() };
    let turn: Vec<u32> = (10..16).collect();
    let longer: Vec<u32> = turn.iter().copied().chain([40, 41]).collect();
    let mut h = with(cfg());
    publish(&mut h, 1, &turn);
    publish(&mut h, 2, &(200..206).collect::<Vec<u32>>());
    assert_eq!((h.engine.live_sequences(), h.engine.cache_exported_tokens()), (2, 0), "two entries hold both slots");
    let p = h.prompt(&[900, 901, 902]);
    let ev = h.tick_ok(h.plan().admit(3, p).prefill(3, 0, 3).decode(3, 1));
    assert_eq!(ev.admit_for(3).status, admit_status::ADMITTED);
    assert_eq!(ev.tick_status, tick_status::OK, "the older entry gave up its slot and nothing left the cache");
    assert_eq!(h.engine.cache_exported_tokens(), 6);
    h.tick_ok(h.plan().retire(3, false));
    assert_eq!(h.engine.match_lengths(&longer), vec![6]);
    let ev = seeded(&mut h, 4, &longer, 6, 3);
    assert_eq!(ev.admit_for(4).status, admit_status::ADMITTED);
    assert_eq!(h.rings_tokens(&ev.emit_for(4).token_ref), cold(cfg(), &longer, 3), "the lane continues as one that prefilled the prompt cold");

    // With no room for exports the entry is evicted, as it always was.
    let mut h = with(FakeConfig { cache_exported_cells: 0, ..cfg() });
    publish(&mut h, 1, &turn);
    publish(&mut h, 2, &(200..206).collect::<Vec<u32>>());
    let p = h.prompt(&[900, 901, 902]);
    let ev = h.tick_ok(h.plan().admit(3, p).prefill(3, 0, 3).decode(3, 1));
    assert_eq!(left(&ev), vec![(shed_kind::CACHE_EVICTION, 6, 6)]);
    assert_eq!(h.engine.match_lengths(&longer), Vec::<u64>::new());
}

/// A runtime that cuts only at a sequence's head serves each entry at its
/// own length, so a conversation's earlier turn is not covered by its later
/// one: both are exported, the one whose cells the other shares after it,
/// and a seed at the earlier turn's length is served from its export.
#[test]
fn a_head_only_runtime_keeps_each_turn_it_cannot_cut_to() {
    let cfg = || FakeConfig { truncate_partial: false, ..pool(8, 64) };
    let turn1: Vec<u32> = (10..16).collect();
    let turn2: Vec<u32> = (10..18).collect();
    let mut h = with(cfg());
    publish(&mut h, 1, &turn1);
    seeded(&mut h, 2, &turn2, 6, 0);
    h.tick_ok(h.plan().retire(2, true));
    assert_eq!((cells_used(&h), h.engine.cache_exported_tokens()), (8, 0), "two turns in eight cells");
    publish(&mut h, 3, &(200..206).collect::<Vec<u32>>());
    assert_eq!((cells_used(&h), h.engine.cache_exported_tokens()), (6, 14), "both turns exported, each at its length");
    let branch: Vec<u32> = turn1.iter().copied().chain([77]).collect();
    assert_eq!(h.engine.match_lengths(&branch), vec![6]);
    let ev = seeded(&mut h, 4, &branch, 6, 2);
    assert_eq!(h.rings_tokens(&ev.emit_for(4).token_ref), cold(cfg(), &branch, 2));
}

/// An entry out of the pool holds no sequence of the runtime's: with a table
/// of two sequences both held by cache entries, a sequence created outside a
/// tick takes the older entry's slot, and the entry is exported, not lost.
/// A sequence published outside a tick settles the cache as a retire does.
#[test]
fn an_entry_out_of_the_pool_holds_no_slot() {
    let p = prompts(2);
    let mut h = with(FakeConfig { max_seqs: 2, ..pool(64, 64) });
    publish(&mut h, 1, &p[0]);
    publish(&mut h, 2, &p[1]);
    assert_eq!(h.engine.live_sequences(), 2);
    let fresh = h.engine.create_sequence().expect("the older entry gives up its slot");
    assert_eq!((h.engine.live_sequences(), h.engine.cache_exported_tokens()), (2, 6));
    assert_eq!((offered(&h, &p[0]), offered(&h, &p[1])), (vec![6], vec![6]));
    h.engine.free_sequence(fresh).unwrap();

    // A fork of a lane, published by hand into a pool that affords four cells.
    let mut h = with(pool(4, 64));
    let tokens: Vec<u32> = (10..16).collect();
    let pt = h.prompt(&tokens);
    h.tick_ok(h.plan().admit(1, pt).prefill(1, 0, 6));
    let child = h.engine.seq_fork(h.engine.lane_sequence(1).unwrap(), 0).unwrap();
    h.engine.publish_sequence(child, &tokens[..5]).unwrap();
    h.tick_ok(h.plan().retire(1, false));
    assert_eq!((cells_used(&h), h.engine.cache_exported_tokens()), (0, 5), "five cells are more than the four afforded");
    assert_eq!(h.engine.match_lengths(&tokens), vec![5]);
}
