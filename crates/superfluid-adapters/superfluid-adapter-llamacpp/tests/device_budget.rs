//! The tests whose outcome turns on how much device memory is free: the free
//! figure itself, the room for exports past the headroom, and a layout of
//! gigabytes of buffers. llama.cpp's Metal backend reports as free the
//! device's working set less what this process has allocated, and the tests
//! of one file run side by side in one process, each opening and dropping
//! contexts of its own, so beside them the figure moves while a test reads
//! it. In `real_gguf` these tests failed now and then whatever the build: on
//! the nightly's m4-pro-mini-1 free memory rose 362 MB while a 397 MB model
//! loaded (#133), and on a 48 GB M5 Pro a run in four granted the exports no
//! room. Run one at a time, they passed every time.
//!
//! Cargo runs one test binary at a time, so these have the process to
//! themselves, and each takes the device alone while it runs.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use superfluid_abi::*;
use superfluid_adapter_llamacpp::{LlamaConfig, LlamaRuntime, LlamaTokenizer};
use superfluid_engine::testing::{tokens_agree, top2_margin, Harness};
use superfluid_engine::{Engine, Tokenizer};
use superfluid_executor::{Executor, ExecutorConfig, Feed, Input, PrimError, RuntimePrimitives};

/// The device, to one test at a time.
static DEVICE: Mutex<()> = Mutex::new(());

/// The device until the guard drops. Taken first in a test, it is let go
/// after everything the test opened; a test that failed holding it leaves the
/// device as free as one that passed.
fn device_alone() -> MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(PoisonError::into_inner)
}

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

fn runtime(path: &Path) -> LlamaRuntime {
    LlamaRuntime::open(LlamaConfig {
        model_path: path.to_path_buf(),
        max_seq_len: 512,
        max_batch: 8,
        n_seq_max: 16,
        ..Default::default()
    })
    .expect("open gguf")
}

fn tolerances(rt: &LlamaRuntime) -> (f32, f32) {
    if rt.expert_count() > 0 {
        (2.5, 2.5)
    } else {
        (0.5, 0.05)
    }
}

/// Publish each prompt from a lane of its own, one after another.
fn publish_each(h: &mut Harness<Executor<LlamaRuntime>>, first_lane: u64, prompts: &[Vec<u32>]) {
    for (k, prompt) in prompts.iter().enumerate() {
        let lane = first_lane + k as u64;
        let p = h.prompt(prompt);
        h.tick_ok(h.plan().admit(lane, p).prefill(lane, 0, prompt.len() as u32));
        h.tick_ok(h.plan().retire(lane, true));
    }
}

/// `seed` tokens of `prompt` cold, then the rest and `n` decoded: the tokens
/// a seed of that length computes after it, and their top-2 margins.
fn cold_in_chunks(h: &mut Harness<Executor<LlamaRuntime>>, lane: u64, prompt: &[u32], seed: u32, n: u16) -> (Vec<u32>, Vec<f32>) {
    let top2 = |mut a: LaneAdmit| {
        a.want_logprobs = 3;
        a
    };
    let p = h.prompt(prompt);
    h.tick_ok(h.plan().admit_with(top2, lane, p).prefill(lane, 0, seed));
    let ev = h.tick_ok(h.plan().prefill(lane, seed, prompt.len() as u32 - seed).decode(lane, n));
    let tokens = h.rings_tokens(&ev.emit_for(lane).token_ref);
    let margins = ev.logprobs_for(lane).into_iter().map(top2_margin).collect();
    h.tick_ok(h.plan().retire(lane, false));
    (tokens, margins)
}

/// `prompt` seeded with its first `seed` tokens from the cache, then the rest
/// and `n` decoded.
fn warm(h: &mut Harness<Executor<LlamaRuntime>>, lane: u64, prompt: &[u32], seed: u32, n: u16) -> Vec<u32> {
    let handle = h.engine.seed_acquire(prompt, seed as u64, determinism::BEST_EFFORT).expect("a seed from the cache");
    let p = h.prompt(prompt);
    let plan = h.plan().admit_with(
        |mut ad| {
            ad.seed_handle = handle;
            ad
        },
        lane,
        p,
    );
    let ev = h.tick_ok(plan.prefill(lane, seed, prompt.len() as u32 - seed).decode(lane, n));
    assert_eq!(ev.admit_for(lane).status, admit_status::ADMITTED);
    h.rings_tokens(&ev.emit_for(lane).token_ref)
}

/// What the prefix cache keeps in a shared KV pool is bounded. Every step's
/// attention there runs over the cells up to the highest one in use,
/// whoever holds them, so a cache left to fill the pool made each lane
/// slower the more it held (Qwen3-4B on an M5 Pro, four lanes: 19.8 ms a
/// round beside an empty cache, 21.9 ms beside 24k cells of it, and beside
/// 64 cells placed above 24k freed ones). Past the bound the entries are the
/// executor's exports, in host memory: the pool holds no more of the cache
/// than the bound, every prompt is still offered, and a request that seeds
/// from an exported one continues as the same chunks prefilled cold.
#[test]
fn the_cache_keeps_a_bounded_part_of_itself_in_a_shared_pool() {
    let _device = device_alone();
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let texts = [
        "The capital of France is Paris, and the capital of Germany is Berlin. The capital of Italy is Rome, and the capital of Spain is",
        "Once upon a time, in a small village by the sea, there lived an old fisherman who went out every morning before the sun rose to",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nprint(fibonacci(",
        "The quick brown fox jumps over the lazy dog near the quiet river bank while the sun sets behind the hills and the birds",
    ];
    let prompts: Vec<Vec<u32>> = texts.iter().map(|t| tok.encode(t)).collect();
    let total: usize = prompts.iter().map(Vec::len).sum();
    let bound = 2 * prompts.iter().map(Vec::len).max().unwrap() as u32;
    assert!(total as u32 > bound, "the four prompts ({total} tokens) are more than the cache keeps in the pool ({bound})");
    let rt = LlamaRuntime::open(LlamaConfig {
        model_path: path.clone(),
        max_seq_len: 512,
        max_batch: 8,
        n_seq_max: 16,
        streams: Some(false),
        cache_resident_cells: bound,
        ..Default::default()
    })
    .expect("open gguf");
    let d = rt.describe();
    assert_eq!(d.cache_resident_cells, bound as u64);
    assert!(d.cache_exported_cells >= total as u64, "room for the exports of a pool this small: {}", d.cache_exported_cells);
    let mut h = Harness::with_engine(Executor::new(rt, ExecutorConfig::default()));
    let (_, same_state) = tolerances(h.engine.primitives());

    let n0 = prompts[0].len() as u32;
    let mut longer = prompts[0].clone();
    longer.extend(tok.encode(" Madrid. The capital of Portugal is"));
    let (cold, margins) = cold_in_chunks(&mut h, 20, &longer, n0, 6);

    publish_each(&mut h, 1, &prompts);
    let held = h.engine.primitives().mem_counters().cells_used;
    assert!(held <= bound as u64, "the pool holds {held} cells of cache, past the {bound} it keeps there");
    assert!(held > 0, "and the newest entries are still in it");
    let exported = h.engine.cache_exported_tokens();
    assert_eq!(held + exported, total as u64, "every prompt is cached: in the pool, or exported");
    assert!(exported >= n0 as u64, "the oldest entry left the pool first");
    for prompt in &prompts {
        let mut next = prompt.clone();
        next.push(longer[n0 as usize]);
        assert_eq!(h.engine.match_lengths(&next).first(), Some(&(prompt.len() as u64)), "each prompt is still offered whole");
    }

    let seeded = warm(&mut h, 30, &longer, n0, 6);
    tokens_agree(&cold, &margins, &seeded, same_state)
        .unwrap_or_else(|e| panic!("a seed from an exported entry continues as the same chunks prefilled cold: {e}"));
}

/// With a buffer per lane, an entry that gives its slot to a new lane is exported and a
/// later seed imports it.
#[test]
fn with_a_buffer_per_lane_an_entry_that_gives_up_its_slot_is_kept_as_its_export() {
    let _device = device_alone();
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let rt = LlamaRuntime::open(LlamaConfig { model_path: path.clone(), max_seq_len: 512, max_batch: 2, streams: Some(true), ..Default::default() })
        .expect("open gguf");
    let d = rt.describe();
    if d.copy_shares_cells {
        eprintln!("SKIP: this model keeps one shared pool (a buffer per lane is for plain attention)");
        return;
    }
    assert_eq!((d.max_seqs, d.cache_resident_cells), (3, 0), "four buffers, one kept free for moves, and no bound on what rests in them");
    assert!(d.cache_exported_cells > 0);
    let mut h = Harness::with_engine(Executor::new(rt, ExecutorConfig::default())).with_max_decode_lanes(2);
    let (_, same_state) = tolerances(h.engine.primitives());
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let prompts: Vec<Vec<u32>> = [
        "The capital of France is Paris, and the capital of Germany is Berlin. The capital of Italy is Rome, and the capital of Spain is",
        "Once upon a time, in a small village by the sea, there lived an old fisherman who went out every morning before the sun rose to",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nprint(fibonacci(",
    ]
    .iter()
    .map(|t| tok.encode(t))
    .collect();
    let n0 = prompts[0].len() as u32;
    let mut longer = prompts[0].clone();
    longer.extend(tok.encode(" Madrid. The capital of Portugal is"));
    let (cold, margins) = cold_in_chunks(&mut h, 20, &longer, n0, 6);

    publish_each(&mut h, 1, &prompts);
    assert_eq!((h.engine.live_sequences(), h.engine.cache_exported_tokens()), (3, 0), "three entries hold every slot");
    let other = tok.encode("Water boils at a hundred degrees Celsius at sea level, and freezes at");
    let p = h.prompt(&other);
    let ev = h.tick_ok(h.plan().admit(9, p).prefill(9, 0, other.len() as u32).decode(9, 2));
    assert_eq!(ev.admit_for(9).status, admit_status::ADMITTED);
    assert_eq!(ev.tick_status, tick_status::OK, "the oldest entry gave up its slot, and nothing left the cache");
    assert_eq!(h.engine.cache_exported_tokens(), n0 as u64, "it is the oldest entry's export");
    h.tick_ok(h.plan().retire(9, false));

    let seeded = warm(&mut h, 30, &longer, n0, 6);
    tokens_agree(&cold, &margins, &seeded, same_state)
        .unwrap_or_else(|e| panic!("a seed from the export continues as the same chunks prefilled cold: {e}"));
}

/// A sliding-window model exports whole entries, both of its caches included, and a seed
/// from one continues as the same chunks prefilled cold.
#[test]
fn a_sliding_window_models_exports_carry_the_window() {
    let _device = device_alone();
    if let Err(why) = superfluid_adapter_llamacpp::sys::load() {
        eprintln!("SKIP: {why}");
        return;
    }
    let Some(path) = std::env::var_os("SUPERFLUID_TEST_GGUF_SWA").map(PathBuf::from).filter(|p| p.is_file()) else {
        eprintln!("SKIP: set SUPERFLUID_TEST_GGUF_SWA to a sliding-window GGUF (gemma-3)");
        return;
    };
    let rt = LlamaRuntime::open(LlamaConfig { model_path: path.clone(), max_seq_len: 2048, max_batch: 2, ..Default::default() })
        .expect("open gguf");
    let d = rt.describe();
    assert!(d.copy_shares_cells && !d.truncate_partial, "one shared pool, cut only at a sequence's head");
    assert_eq!(d.cache_resident_cells, d.max_seq_len, "a lane's worth of cache in the pool");
    assert!(d.cache_exported_cells >= d.max_seq_len, "and room for more than that exported: {}", d.cache_exported_cells);
    let mut h = Harness::with_engine(Executor::new(rt, ExecutorConfig::default())).with_max_decode_lanes(2);
    let (_, same_state) = tolerances(h.engine.primitives());
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    // Past gemma-3's 1,024-token window.
    let long = |about: &str| {
        let mut text = String::new();
        while tok.encode(&text).len() < 1200 {
            text.push_str(&format!("Notes on {about}, written down one more time. ").repeat(20));
        }
        tok.encode(&text)
    };
    let prompts: Vec<Vec<u32>> = ["rivers", "mountains", "deserts"].iter().map(|a| long(a)).collect();
    let n0 = prompts[0].len() as u32;
    let mut longer = prompts[0].clone();
    longer.extend(tok.encode(" In short, rivers are"));
    let (cold, margins) = cold_in_chunks(&mut h, 20, &longer, n0, 6);

    publish_each(&mut h, 1, &prompts);
    let held = h.engine.primitives().mem_counters().cells_used;
    assert!(held <= d.max_seq_len, "the pool holds {held} cells of cache, past the {} it keeps there", d.max_seq_len);
    assert!(h.engine.cache_exported_tokens() >= n0 as u64, "the oldest entry is an export");
    let seeded = warm(&mut h, 30, &longer, n0, 6);
    tokens_agree(&cold, &margins, &seeded, same_state)
        .unwrap_or_else(|e| panic!("a seed from a sliding-window export continues as the same chunks prefilled cold: {e}"));
}

#[test]
fn the_ledger_follows_copy_chains_and_the_limits_follow_the_config() {
    let _device = device_alone();
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let open = |streams: Option<bool>| {
        LlamaRuntime::open(LlamaConfig { model_path: path.clone(), max_seq_len: 8192, max_batch: 8, streams, ..Default::default() })
            .expect("open gguf")
    };
    let d = open(Some(false)).describe();
    assert_eq!(d.max_seq_len, 8192, "the per-sequence context is the configured one, not n_ctx / n_seq_max");
    assert_eq!(d.max_seqs, 31, "twice the batch, at least 32, minus the export scratch");
    assert!(d.copy_shares_cells && !d.takeover_preferred && d.seed_min_tokens == 0);
    let d = open(None).describe();
    assert_eq!(
        (d.max_seq_len, d.max_seqs, d.cells_total),
        (8192, 9, 10 * 8192),
        "ten buffers: one per lane, one for a cached prefix, one kept free for moves"
    );
    assert!(!d.copy_shares_cells && d.takeover_preferred && d.seed_min_tokens == 16);

    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let prompt = tok.encode("The quick brown fox jumps over the lazy dog");
    let n = prompt.len() as u64;
    let mut rt = runtime(&path);
    let a = rt.seq_create().unwrap();
    rt.step(&[Feed { seq: a, input: Input::Tokens(&prompt), wants_row: false }]).unwrap();
    let b = rt.seq_create().unwrap();
    rt.seq_copy(a, b, n).unwrap();
    rt.step(&[Feed { seq: b, input: Input::Tokens(&prompt[..2]), wants_row: false }]).unwrap();
    let c = rt.seq_create().unwrap();
    rt.seq_copy(b, c, n + 2).unwrap();
    assert_eq!(rt.mem_counters().cells_used, n + 2, "A, B and C hold n + 2 distinct cells");
    rt.seq_free(b);
    assert_eq!(rt.mem_counters().cells_used, n + 2);
    if rt.describe().recurrent {
        assert_eq!(rt.seq_truncate(a, 3), Err(PrimError::OutOfBoundary), "head-only: no cut below the head");
    } else {
        rt.seq_truncate(a, 3).unwrap();
    }
    assert_eq!(rt.mem_counters().cells_used, n + 2);
    rt.seq_free(a);
    assert_eq!(rt.mem_counters().cells_used, n + 2);
    rt.seq_free(c);
    assert_eq!(rt.mem_counters().cells_used, 0);
}

/// A shared pool is sized by what its context takes on the device, which is
/// more than its KV. Eight lanes of Qwen3-0.6B's trained 40,960 on a 48 GB Mac
/// were given a pool of 208k cells, 60% of the free memory by their KV; the
/// context llama.cpp b12503 made for it reserved another 28.6 GB to gather K
/// and V rows into, 52 GB in all on a device of 40. Metal allocated it, the
/// device's free figure read zero, and every request failed: Metal refused the
/// first step's command buffers (`kIOGPUCommandBufferCallbackErrorOutOfMemory`)
/// and llama.cpp the step after it. Sized by what a context takes, the pool
/// leaves the device its headroom and its steps compute what a small pool's do.
#[test]
fn a_shared_pool_is_sized_by_what_its_context_takes_on_the_device() {
    let _device = device_alone();
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF model");
        return;
    };
    let before = superfluid_adapter_llamacpp::runtime::device_free_bytes();
    let Some(gpu) = superfluid_adapter_llamacpp::devices().into_iter().find(|d| d.budget > 0).filter(|_| before > 0) else {
        eprintln!("SKIP: no GPU device reports a budget");
        return;
    };
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let prompt = tok.encode("The capital of France is Paris, and the capital of Germany is");
    let next_two = |rt: &mut LlamaRuntime| {
        let seq = rt.seq_create().unwrap();
        let row = rt.step(&[Feed { seq, input: Input::Tokens(&prompt), wants_row: true }]).expect("the prompt's step").remove(0);
        let first = superfluid_executor::sampling::argmax(&row);
        let row = rt.step(&[Feed { seq, input: Input::Tokens(&[first]), wants_row: true }]).expect("the step after it").remove(0);
        (first, superfluid_executor::sampling::argmax(&row))
    };
    let expected = next_two(&mut runtime(&path));

    let (lanes, window) = (8u32, 40_960u32);
    let mut rt = LlamaRuntime::open(LlamaConfig {
        model_path: path.clone(),
        max_seq_len: window,
        max_batch: lanes,
        streams: Some(false),
        ..Default::default()
    })
    .expect("open gguf");
    let left = superfluid_adapter_llamacpp::runtime::device_free_bytes();
    let cells = rt.describe().cells_total;
    let headroom = superfluid_executor::sizing::headroom_bytes(gpu.budget);
    assert!(left >= headroom, "a pool of {cells} cells left {left} bytes of the device free, under the {headroom} it keeps clear");
    if cells > window as u64 && cells < lanes as u64 * window as u64 {
        // The pool's share of what the weights left free, the weights themselves, and the
        // cells llama.cpp rounds a context up by.
        let weights = std::fs::metadata(&path).unwrap().len();
        let share = (superfluid_executor::sizing::POOL_SHARE_OF_FREE * before as f64) as u64 + weights + (256 << 20);
        assert!(before - left <= share, "the model and a pool of {cells} cells took {} bytes of the {before} free", before - left);
    }
    assert_eq!(next_two(&mut rt), expected, "the pool's steps compute what a small pool's do");
}

#[test]
fn the_pool_budget_counts_the_weights() {
    let _device = device_alone();
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF model");
        return;
    };
    let before = superfluid_adapter_llamacpp::runtime::device_free_bytes();
    if before == 0 {
        eprintln!("SKIP: no GPU device reports a budget");
        return;
    }
    let rt = runtime(&path);
    let after = superfluid_adapter_llamacpp::runtime::device_free_bytes();
    let weights = std::fs::metadata(&path).unwrap().len();
    assert!(
        before.saturating_sub(after) >= weights / 2,
        "free device memory went {before} -> {after} with a {weights}-byte model open"
    );
    assert_eq!(rt.describe().cells_total, 8 * 512, "a shape this small is granted whole");
}
