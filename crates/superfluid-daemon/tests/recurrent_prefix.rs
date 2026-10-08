//! On a runtime whose state cannot be cut back (recurrent or hybrid models),
//! a prefix is reused only from a state kept exactly where it ends.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use superfluid_daemon::codec::{MockChatCodec, TextCodec};
use superfluid_daemon::wal::role;
use superfluid_daemon::{Daemon, DaemonOptions, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_executor::fake::{FakeConfig, FakePrimitives};
use superfluid_executor::{Executor, ExecutorConfig};

fn daemon(max_lanes: usize, prefill_budget: u32) -> Arc<Daemon> {
    daemon_with(Box::new(MockCodec), max_lanes, prefill_budget)
}

fn daemon_with(codec: Box<dyn TextCodec + Send + Sync>, max_lanes: usize, prefill_budget: u32) -> Arc<Daemon> {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "superfluid-recurrent-prefix-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let cfg = FakeConfig {
        truncate_partial: false,
        page: 1,
        step_delay: std::time::Duration::from_millis(2),
        ..FakeConfig::default()
    };
    let host = EngineHost::spawn(move || (Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()), None))
        .expect("spawn");
    let opts = DaemonOptions { max_lanes, prefill_budget, ..Default::default() };
    Arc::new(Daemon::with_options(store, host, codec, opts).unwrap())
}

#[test]
fn a_burst_sharing_a_long_prefix_starts_from_the_state_kept_where_it_ends() {
    let d = daemon(4, 64);
    let stats = d.sched_stats();
    const PREFIX: u32 = 2048;
    let sessions: Vec<u64> = (0..5u32)
        .map(|i| {
            let s = d.create(None, GenParams::default()).unwrap();
            let mut prompt: Vec<u32> = (0..PREFIX).map(|t| t % 900 + 50).collect();
            prompt.extend((0..8).map(|t| 1000 + i * 8 + t));
            d.append(s, None, prompt).unwrap();
            s
        })
        .collect();
    let calls: Vec<_> = sessions
        .iter()
        .map(|&s| {
            let dd = Arc::clone(&d);
            std::thread::spawn(move || dd.generate(s, 8))
        })
        .collect();
    for h in calls {
        assert_eq!(h.join().unwrap().unwrap().tokens_generated, 8);
    }
    let cold = stats.cold_admissions.load(Ordering::Relaxed);
    let warm = stats.warm_prefix_tokens.load(Ordering::Relaxed);
    assert_eq!(cold, 1, "one request prefilled the shared prefix; the rest started from its state (warm {warm})");
    assert!(warm >= 4 * PREFIX as u64, "the other four were seeded with the whole prefix: {warm}");
}

#[test]
fn a_resent_prompt_starts_one_token_short_of_its_end() {
    let d = daemon(2, 4096);
    let stats = d.sched_stats();
    let prompt: Vec<u32> = (0..600).map(|t| t % 700 + 30).collect();
    let run = || {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, prompt.clone()).unwrap();
        d.generate(s, 6).unwrap()
    };
    let first = run();
    assert_eq!(stats.warm_prefix_tokens.load(Ordering::Relaxed), 0);
    let again = run();
    assert_eq!(stats.warm_prefix_tokens.load(Ordering::Relaxed), 599, "seeded from the checkpoint");
    assert_eq!(first.tokens_generated, again.tokens_generated);
}

/// A session of `history`, and the length of that input.
fn open_chat(d: &Daemon, history: &[(u32, String)]) -> (u64, u64) {
    let s = d.create(None, GenParams::default()).unwrap();
    for (r, text) in history {
        d.append_message(s, *r, text.clone()).unwrap();
    }
    (s, d.inspect(s).unwrap().tokens.len() as u64)
}

/// One turn of a chat: a session of `history`, generated from. Returns the
/// length of its input (before the generation prompt) and the outcome.
fn chat_turn(d: &Daemon, history: &[(u32, String)]) -> (u64, superfluid_daemon::GenerateOutcome) {
    let (s, input) = open_chat(d, history);
    (input, d.generate(s, 6).unwrap())
}

fn history(turns: &[(u32, &str)]) -> Vec<(u32, String)> {
    turns.iter().map(|(r, t)| (*r, t.to_string())).collect()
}

fn then(h: &[(u32, String)], reply: &str, ask: &str) -> Vec<(u32, String)> {
    let mut h = h.to_vec();
    h.push((role::ASSISTANT, reply.to_string()));
    h.push((role::USER, ask.to_string()));
    h
}

fn text(n: usize, salt: u8) -> String {
    (0..n).map(|i| char::from(b'a' + ((i + salt as usize) % 26) as u8)).collect()
}

fn spans(o: &superfluid_daemon::GenerateOutcome) -> Vec<u32> {
    o.events.iter().filter_map(|e| e.body.span().map(<[u32]>::to_vec)).flatten().collect()
}

/// A chat's next turn renders the history again, and the assistant turn it
/// renders does not begin as the generation prompt did (a template drops the
/// reasoning opener from history), so the state kept where the prompt ends
/// is no prefix of it. The state where the last input ended is.
#[test]
fn the_next_turn_starts_from_the_state_kept_where_the_last_input_ended() {
    let system = text(700, 0);
    let first = history(&[(role::SYSTEM, &system), (role::USER, "why does the build fail?")]);
    let next = then(&first, "a missing import", "fix it");

    let d = daemon_with(Box::new(MockChatCodec), 2, 4096);
    let (boundary, _) = chat_turn(&d, &first);
    let (next_boundary, warm) = chat_turn(&d, &next);
    assert_eq!(warm.warm_prefix, boundary, "turn two started from the state where turn one's input ended");
    let (_, last) = chat_turn(&d, &then(&next, "done", "thanks"));
    assert_eq!(last.warm_prefix, next_boundary, "and turn three from where turn two's ended");

    let cold = daemon_with(Box::new(MockChatCodec), 2, 4096);
    let (_, cold) = chat_turn(&cold, &next);
    assert_eq!(cold.warm_prefix, 0);
    assert_eq!(spans(&warm), spans(&cold), "the warm turn decodes as the cold one");
    assert!(!spans(&cold).is_empty());
}

/// A turn's checkpoint serves the next turn only; once that turn keeps its
/// own, the old one is dropped rather than held (a forked checkpoint is a
/// shared prefix, the last thing evicted), so a long chat holds one.
#[test]
fn a_turn_checkpoint_is_dropped_once_the_next_turn_keeps_its_own() {
    let d = daemon_with(Box::new(MockChatCodec), 2, 4096);
    let system = text(700, 0);
    let mut turns = vec![history(&[(role::SYSTEM, &system), (role::USER, "why does the build fail?")])];
    for i in 1..4 {
        let h = then(&turns[i - 1], &format!("answer {i}"), &format!("question {i}"));
        turns.push(h);
    }
    let mut boundaries = Vec::new();
    for (i, h) in turns.iter().enumerate() {
        let (input, out) = chat_turn(&d, h);
        if i > 0 {
            assert_eq!(out.warm_prefix, boundaries[i - 1], "turn {} starts where turn {i}'s input ended", i + 1);
        }
        boundaries.push(input);
    }
    let (_, branch) = chat_turn(&d, &then(&turns[0], "another answer", "and then?"));
    assert_eq!(branch.warm_prefix, 0, "turn one's checkpoint went when turn two kept its own");
    let (_, branch) = chat_turn(&d, &then(&turns[2], "another answer", "and then?"));
    assert_eq!(branch.warm_prefix, 0, "and turn three's when turn four kept its own");
    let (_, branch) = chat_turn(&d, &then(&turns[3], "another answer", "and then?"));
    assert_eq!(branch.warm_prefix, boundaries[3], "the last turn's stays");
}

/// A lane may stop twice in one prefill: where requests queued behind it
/// stop sharing its prompt, and where its input ends. Here the shared part
/// (a system prompt) ends first.
#[test]
fn a_prefill_keeps_a_shared_prefix_and_then_its_turn_checkpoint() {
    let d = daemon_with(Box::new(MockChatCodec), 4, 64);
    let stats = d.sched_stats();
    let system = text(1500, 0);
    let chats: Vec<Vec<(u32, String)>> = ["alpha: what fails?", "bravo: what passes?"]
        .iter()
        .map(|q| history(&[(role::SYSTEM, &system), (role::USER, q)]))
        .collect();
    let opened: Vec<(u64, u64)> = chats.iter().map(|h| open_chat(&d, h)).collect();
    let calls: Vec<_> = opened
        .iter()
        .map(|&(s, input)| {
            let dd = Arc::clone(&d);
            std::thread::spawn(move || (input, dd.generate(s, 6).unwrap()))
        })
        .collect();
    let firsts: Vec<_> = calls.into_iter().map(|c| c.join().unwrap()).collect();
    assert_eq!(stats.cold_admissions.load(Ordering::Relaxed), 1, "one prefilled the system prompt");
    let shared = system.len() as u64;
    let warm: Vec<u64> = firsts.iter().map(|(_, o)| o.warm_prefix).collect();
    assert!(warm.contains(&0) && warm.iter().any(|&w| w > shared), "the other started past it: {warm:?}");
    for (h, (input, _)) in chats.iter().zip(&firsts) {
        let (_, next) = chat_turn(&d, &then(h, "ok", "go on"));
        assert_eq!(next.warm_prefix, *input, "each chat's next turn starts where its input ended");
    }
}

/// Here the turn checkpoint comes first: the request queued behind the lane
/// is the chat's next turn, which shares its input and a token of its
/// generation prompt. The state where the input ends serves it, and no
/// second state is kept a token later.
#[test]
fn a_turn_checkpoint_kept_first_serves_a_request_queued_behind_it() {
    let d = daemon_with(Box::new(MockChatCodec), 4, 64);
    let stats = d.sched_stats();
    let first = history(&[(role::SYSTEM, &text(1500, 0)), (role::USER, "why does the build fail?")]);
    let next = then(&first, "a missing import", "fix it");
    let ((lead, input), (queued, _)) = (open_chat(&d, &first), open_chat(&d, &next));
    let lead = {
        let dd = Arc::clone(&d);
        std::thread::spawn(move || dd.generate(lead, 6).unwrap())
    };
    std::thread::sleep(std::time::Duration::from_millis(15));
    let queued = d.generate(queued, 6).unwrap();
    let lead = lead.join().unwrap();
    assert_eq!(lead.warm_prefix, 0);
    assert_eq!(queued.warm_prefix, input, "the queued turn started where the lead's input ended");
    assert_eq!(stats.cold_admissions.load(Ordering::Relaxed), 1);
}

fn daemon_pooled(resident: u64, exported: u64) -> Arc<Daemon> {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "superfluid-recurrent-pooled-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    // As llama.cpp's shared pool for a hybrid model: copies share cells, the
    // cache keeps `resident` cells in the pool and `exported` tokens out of it,
    // and there are 31 sequences, which entries no later one covers fill.
    let cfg = FakeConfig {
        truncate_partial: false,
        page: 1,
        kv_bytes_per_token: 1,
        copy_shares_cells: true,
        cache_resident_cells: resident,
        cache_exported_cells: exported,
        max_seqs: 31,
        ..FakeConfig::default()
    };
    let host = EngineHost::spawn(move || (Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()), None))
        .expect("spawn");
    let opts = DaemonOptions { max_lanes: 2, prefill_budget: 4096, ..Default::default() };
    Arc::new(Daemon::with_options(store, host, Box::new(MockChatCodec), opts).unwrap())
}

/// A long chat on a runtime with a fixed number of sequences: the entries
/// its turns leave fill them (on a recurrent model no later entry covers an
/// earlier one), and each turn still starts from the state where the last
/// turn's input ended.
#[test]
fn a_long_chat_keeps_starting_from_its_last_turn_when_its_entries_fill_the_sequences() {
    let d = daemon_pooled(4096, 8000);
    let mut h = history(&[(role::SYSTEM, &text(700, 0)), (role::USER, "look up the code for account 1")]);
    let mut last_input = 0;
    let mut stalls = Vec::new();
    for turn in 0..30 {
        let (input, out) = chat_turn(&d, &h);
        if turn > 0 && out.warm_prefix != last_input {
            stalls.push((turn, input, out.warm_prefix, last_input));
        }
        last_input = input;
        h = then(&h, &format!("lookup {turn} {}", text(12, turn as u8)), &format!("code {}; next {}?", 1000 + turn, turn + 1));
    }
    assert!(stalls.is_empty(), "turns that did not start from the last turn's input (turn, input, warm, wanted): {stalls:?}");
}
