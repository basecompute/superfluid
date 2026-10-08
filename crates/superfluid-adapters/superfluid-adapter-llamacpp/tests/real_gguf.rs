use std::path::{Path, PathBuf};
use std::sync::Arc;

use superfluid_abi::*;
use superfluid_adapter_llamacpp::{LlamaConfig, LlamaRuntime, LlamaTokenizer};
use superfluid_engine::testing::{rows_agree, tokens_agree, top2_margin, Harness};
use superfluid_engine::{Engine, Tokenizer};
use superfluid_executor::{Executor, ExecutorConfig, Feed, Input, PrimError, RuntimePrimitives, SampleSpec};

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

#[test]
fn the_sequence_table_stops_at_llama_cpps_limit() {
    let Some(path) = gguf() else { return };
    let open = |max_batch: u32, streams: Option<bool>| {
        LlamaRuntime::open(LlamaConfig { model_path: path.clone(), max_seq_len: 32, max_batch, streams, ..Default::default() })
    };
    let rt = open(160, Some(false)).unwrap_or_else(|e| panic!("160 lanes fit a table of 256: {e}"));
    assert_eq!(rt.describe().max_seqs, 255, "the shared pool: every id but the scratch one");
    drop(rt);
    let rt = open(160, None).unwrap_or_else(|e| panic!("160 lanes fit 162 buffers: {e}"));
    assert_eq!(rt.describe().max_seqs, 161, "a buffer per lane and two more: every buffer but the one kept free for moves");
    drop(rt);
    let err = open(256, None).err().expect("256 lanes and the scratch id do not fit").to_string();
    assert!(err.contains("at most 255 lanes"), "{err}");
}

#[test]
fn a_failed_load_says_what_llama_cpp_said() {
    if gguf().is_none() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("superfluid-llama-badload-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bad = dir.join("not-a-model.gguf");
    std::fs::write(&bad, b"GGUF\x03\x00\x00\x00 this is no model at all").unwrap();
    let err = LlamaRuntime::open(LlamaConfig { model_path: bad.clone(), ..Default::default() }).err().expect("no model").to_string();
    let said = err.strip_prefix(&format!("model load failed: {}", bad.display())).unwrap_or_else(|| panic!("{err}"));
    assert!(said.starts_with(": ") && said.len() > 12, "llama.cpp's reason follows the file: {err}");
    eprintln!("a failed load reads: {err}");
    let err = LlamaTokenizer::load(&bad).err().expect("no vocabulary either").to_string();
    assert!(err.len() > format!("model load failed: {}", bad.display()).len() + 12, "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

fn harness(path: &Path) -> Harness<Executor<LlamaRuntime>> {
    Harness::with_engine(Executor::new(runtime(path), ExecutorConfig::default()))
}

fn greedy(h: &mut Harness<Executor<LlamaRuntime>>, lane: u64, prompt: &[u32], n: u16) -> Vec<u32> {
    let p = h.prompt(prompt);
    let ev = h.tick_ok(h.plan().admit(lane, p).prefill(lane, 0, prompt.len() as u32).decode(lane, n));
    h.rings_tokens(&ev.emit_for(lane).token_ref)
}

fn greedy_with_margins(h: &mut Harness<Executor<LlamaRuntime>>, lane: u64, prompt: &[u32], n: u16) -> (Vec<u32>, Vec<f32>) {
    let p = h.prompt(prompt);
    let top2 = |mut a: LaneAdmit| {
        a.want_logprobs = 3;
        a
    };
    let ev = h.tick_ok(h.plan().admit_with(top2, lane, p).prefill(lane, 0, prompt.len() as u32).decode(lane, n));
    let tokens = h.rings_tokens(&ev.emit_for(lane).token_ref);
    let margins = ev.logprobs_for(lane).into_iter().map(top2_margin).collect();
    (tokens, margins)
}

fn tolerances(rt: &LlamaRuntime) -> (f32, f32) {
    if rt.expert_count() > 0 {
        (2.5, 2.5)
    } else {
        (0.5, 0.05)
    }
}

fn row(rt: &mut LlamaRuntime, seq: u64, tokens: &[u32]) -> Vec<f32> {
    rt.step(&[Feed { seq, input: Input::Tokens(tokens), wants_row: true }]).unwrap().remove(0)
}

#[test]
fn a_real_models_tool_call_is_held_to_the_tools_schema() {
    use superfluid_daemon::wal::role;
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let tok = Arc::new(LlamaTokenizer::load(&path).expect("tokenizer"));
    let (codec, kind) = superfluid_daemon::codec::auto_codec_from_tokenizer(tok.clone());
    let tool = r#"{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}"#.to_string();
    let Some(tag) = codec.structural_tag(std::slice::from_ref(&tool), true) else {
        eprintln!("SKIP: the {kind} dialect writes no structural tag");
        return;
    };
    let Some((open, close)) = codec.tool_call_delimiters() else {
        eprintln!("SKIP: no call frame");
        return;
    };
    let id = |s: &str| tok.special_tokens().into_iter().find(|(t, _)| t == s.trim()).map(|(_, i)| i);
    let (Some(open_id), Some(close_id)) = (id(&open), id(&close)) else {
        eprintln!("SKIP: the call frame is not a pair of marker tokens");
        return;
    };
    let mut h = harness(&path);
    let g = h.engine.grammar_create_structural(&tag).expect("the daemon's tag compiles");
    let mut prompt = codec
        .render_prompt(&[(role::USER, "What is the weather in Paris right now?".into())], &[tool])
        .expect("tools prompt");
    prompt.extend(codec.empty_think_tokens().unwrap_or_default());
    let p = h.prompt(&prompt);
    let ev = h.tick_ok(
        h.plan()
            .admit_with(
                |mut a| {
                    a.grammar_handle = g;
                    a.params.flags = 0;
                    a
                },
                1,
                p,
            )
            .prefill(1, 0, prompt.len() as u32)
            .decode(1, 96),
    );
    let out = h.rings_tokens(&ev.emit_for(1).token_ref);
    let text = String::from_utf8_lossy(&out.iter().flat_map(|&t| tok.token_bytes(t)).collect::<Vec<_>>()).into_owned();
    eprintln!("constrained call: {out:?} {text:?}");
    let start = out.iter().position(|&t| t == open_id).expect("the call opens with its marker");
    let end = out.iter().position(|&t| t == close_id).expect("the call closes with its marker");
    let body: Vec<u8> = out[start + 1..end].iter().flat_map(|&t| tok.token_bytes(t)).collect();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{e}: {:?}", String::from_utf8_lossy(&body)));
    assert_eq!(v["name"], "get_weather", "{v}");
    assert!(v["arguments"]["city"].is_string(), "{v}");
}

#[test]
fn greedy_is_stable_and_warm_seed_continues_exactly() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let prompt = tok.encode("The capital of France is");
    let mut h = harness(&path);
    let (reshape, same_state) = tolerances(h.engine.primitives());
    let (a, a_margins) = greedy_with_margins(&mut h, 1, &prompt, 12);
    let b = greedy(&mut h, 2, &prompt, 12);
    tokens_agree(&a, &a_margins, &b, same_state)
        .unwrap_or_else(|e| panic!("greedy decode is stable across lanes: {e}"));
    let text = tok.chat_template_jinja();
    assert!(!text.is_empty(), "a chat model carries a template");
    assert_eq!(tok.chat_template_named("no-such-template"), None);
    if let Some(tool_use) = tok.chat_template_named("tool_use") {
        assert!(!tool_use.is_empty() && tool_use != "no-such-template");
    }
    eprintln!("greedy: {:?}", String::from_utf8_lossy(&a.iter().flat_map(|&t| tok.token_bytes(t)).collect::<Vec<_>>()));

    h.tick_ok(h.plan().retire(1, true).retire(2, false));
    let mut longer = prompt.clone();
    longer.extend_from_slice(&a[..4]);
    let cold = greedy(&mut h, 3, &longer, 6);
    h.tick_ok(h.plan().retire(3, false));
    tokens_agree(&a[4..10], &a_margins[4..10], &cold, reshape)
        .unwrap_or_else(|e| panic!("the cold continuation is the original tail up to a near-tie: {e}"));
    if h.engine.descriptor().recurrent {
        assert_eq!(
            h.engine.seed_acquire(&longer, prompt.len() as u64, determinism::BEST_EFFORT).err(),
            Some(Status::SeedUnservable),
            "head-only: a seed below the head is unservable"
        );
        let (reference, reference_margins) = greedy_with_margins(&mut h, 5, &prompt, 18);
        h.tick_ok(h.plan().retire(5, false));
        let mut span = prompt.clone();
        span.extend_from_slice(&a);
        let mut next = span.clone();
        next.push(reference[12]);
        let handle = h.engine.seed_acquire(&next, span.len() as u64, determinism::BEST_EFFORT).expect("warm seed at the head");
        let p = h.prompt(&next);
        let ev = h.tick_ok(
            h.plan()
                .admit_with(
                    |mut ad| {
                        ad.seed_handle = handle;
                        ad
                    },
                    6,
                    p,
                )
                .prefill(6, span.len() as u32, 1)
                .decode(6, 5),
        );
        let warm = h.rings_tokens(&ev.emit_for(6).token_ref);
        tokens_agree(&reference[13..18], &reference_margins[13..18], &warm, same_state)
            .unwrap_or_else(|e| panic!("a seed at the head continues the stream: {e}"));
        return;
    }
    let p = h.prompt(&longer);
    let top2 = |mut a: LaneAdmit| {
        a.want_logprobs = 3;
        a
    };
    h.tick_ok(h.plan().admit_with(top2, 8, p).prefill(8, 0, prompt.len() as u32));
    let ev = h.tick_ok(h.plan().prefill(8, prompt.len() as u32, 4).decode(8, 6));
    let cold_chunks = h.rings_tokens(&ev.emit_for(8).token_ref);
    let chunk_margins: Vec<f32> = ev.logprobs_for(8).into_iter().map(top2_margin).collect();
    h.tick_ok(h.plan().retire(8, false));
    let handle = h.engine.seed_acquire(&longer, prompt.len() as u64, determinism::BEST_EFFORT).expect("warm seed");
    let p = h.prompt(&longer);
    let ev = h.tick_ok(
        h.plan()
            .admit_with(
                |mut ad| {
                    ad.seed_handle = handle;
                    ad
                },
                4,
                p,
            )
            .prefill(4, prompt.len() as u32, 4)
            .decode(4, 6),
    );
    let warm = h.rings_tokens(&ev.emit_for(4).token_ref);
    tokens_agree(&cold_chunks, &chunk_margins, &warm, same_state)
        .unwrap_or_else(|e| panic!("a warm seed continues as the same chunks prefilled cold: {e}"));
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

/// A prompt costs what its tokens cost, whatever their count. llama.cpp's
/// Metal attention kernel takes a batch of twenty tokens or more eight
/// queries at a time and skips the cache blocks a group is masked from; a
/// last group short of eight it computes against every cell up to the
/// highest one in use. Beside 12k cells of other sequences a 41-token prompt
/// took 33 ms on Qwen3-0.6B where one of 40 took 12 (118 ms and 54 on
/// Qwen3-4B beside 17k), so a request's first token came later the more the
/// pool held. On a shared pool the adapter fills such a batch to whole
/// groups with tokens of its scratch sequence, which change nothing a
/// sequence computes.
#[test]
fn a_prompt_costs_the_same_whatever_its_token_count() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let open = |fill_batches: bool| {
        LlamaRuntime::open(LlamaConfig {
            model_path: path.clone(),
            max_seq_len: 2048,
            max_batch: 8,
            streams: Some(false),
            fill_batches,
            ..Default::default()
        })
        .expect("open gguf")
    };
    let mut rt = open(true);
    let d = rt.describe();
    if d.backend != "metal" || d.recurrent {
        eprintln!("SKIP: the fill is for llama.cpp's Metal kernel on a model whose batches are not split by sequence");
        return;
    }
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let text = tok.encode(&"The quick brown fox jumps over the lazy dog near the quiet river bank while the sun sets. ".repeat(120));
    // Six sequences of 2,000 tokens: 12k cells no prompt below attends to.
    for i in 0..6 {
        let s = rt.seq_create().unwrap();
        rt.step(&[Feed { seq: s, input: Input::Tokens(&text[i..i + 2000]), wants_row: false }]).unwrap();
    }
    assert_eq!(rt.mem_counters().cells_used, 12_000);
    let mut prompt_ms = |n: usize, at: usize| -> f64 {
        let s = rt.seq_create().unwrap();
        let t0 = std::time::Instant::now();
        rt.step(&[Feed { seq: s, input: Input::Tokens(&text[at..at + n]), wants_row: true }]).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        rt.seq_free(s);
        ms
    };
    // Once each unmeasured: the device finishes the fill above, and each
    // batch shape has run.
    prompt_ms(40, 30);
    prompt_ms(41, 30);
    let (mut whole, mut odd) = (Vec::new(), Vec::new());
    for i in 0..7 {
        whole.push(prompt_ms(40, 50 + 3 * i));
        odd.push(prompt_ms(41, 50 + 3 * i));
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (whole, odd) = (median(whole), median(odd));
    eprintln!("beside 12k cells: a 40-token prompt {whole:.1} ms, a 41-token one {odd:.1} ms");
    assert!(odd < 1.5 * whole, "a 41-token prompt took {odd:.1} ms beside 12k cells of other sequences, one of 40 tokens {whole:.1} ms");

    // The filled batch answers what an unfilled one does.
    let s = rt.seq_create().unwrap();
    let filled = row(&mut rt, s, &text[50..91]);
    drop(rt);
    let mut plain = open(false);
    let s = plain.seq_create().unwrap();
    let unfilled = row(&mut plain, s, &text[50..91]);
    let (row_tol, _) = tolerances(&plain);
    rows_agree(&unfilled, &filled, row_tol).expect("a 41-token prompt filled to 48 answers as it does unfilled");
}

#[test]
fn park_export_and_resume_by_adoption_continue_exactly() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let prompt = tok.encode("Once upon a time, in a small village,");
    let mut h = harness(&path);
    let full = greedy(&mut h, 1, &prompt, 10);
    let seq = h.engine.lane_sequence(1).unwrap();
    let ingested = prompt.len() as u64 + 9;
    let (bytes, gen) = h.engine.space_export_size(seq, 1, TokenRange { start: 0, end: ingested }, encoding::LOSSLESS).unwrap();
    let op = h.engine.space_snapshot(seq, 1, ingested, bytes, gen).unwrap();
    let sealed = h.engine.take_op_output(op).unwrap();
    drop(h);

    let mut h2 = harness(&path);
    let fresh = h2.engine.create_sequence().unwrap();
    let op = h2.engine.space_restore(fresh, 1, &sealed).unwrap();
    assert_eq!(h2.engine.op_poll(op).unwrap().state, op_state::DONE);
    let mut covered = prompt.clone();
    covered.extend_from_slice(&full[..9]);
    let handle = h2.engine.seed_adopt(fresh, &covered).unwrap();
    let mut stream = covered.clone();
    stream.push(full[9]);
    let p = h2.prompt(&stream);
    let ev = h2.tick_ok(
        h2.plan()
            .admit_with(
                |mut ad| {
                    ad.seed_handle = handle;
                    ad
                },
                7,
                p,
            )
            .prefill(7, covered.len() as u32, 1)
            .decode(7, 4),
    );
    let resumed = h2.rings_tokens(&ev.emit_for(7).token_ref);
    drop(h2);
    let mut h3 = harness(&path);
    let (uninterrupted, margins) = greedy_with_margins(&mut h3, 1, &prompt, 14);
    let (_, same_state) = tolerances(h3.engine.primitives());
    tokens_agree(&uninterrupted[10..], &margins[10..], &resumed, same_state)
        .unwrap_or_else(|e| panic!("resume from a parked artifact continues the stream: {e}"));
}

#[test]
fn batched_rounds_agree_with_solo_decode() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let prompts: Vec<Vec<u32>> = ["The capital of France is", "Once upon a time, in a small village by the sea,", "def fibonacci(n):"]
        .iter()
        .map(|t| tok.encode(t))
        .collect();
    let mut rt = runtime(&path);
    let (reshape, _) = tolerances(&rt);
    let rounds = 8;
    let mut solo: Vec<(Vec<u32>, Vec<Vec<f32>>)> = Vec::new();
    for p in &prompts {
        let s = rt.seq_create().unwrap();
        let mut r = row(&mut rt, s, p);
        let (mut toks, mut rows) = (Vec::new(), Vec::new());
        for _ in 0..rounds {
            let t = host_argmax(&r);
            toks.push(t);
            r = row(&mut rt, s, &[t]);
            rows.push(r.clone());
        }
        solo.push((toks, rows));
        rt.seq_free(s);
    }
    let seqs: Vec<u64> = prompts.iter().map(|_| rt.seq_create().unwrap()).collect();
    for (s, p) in seqs.iter().zip(&prompts) {
        row(&mut rt, *s, p);
    }
    for k in 0..rounds {
        let inputs: Vec<[u32; 1]> = solo.iter().map(|(t, _)| [t[k]]).collect();
        let feeds: Vec<Feed> =
            seqs.iter().zip(&inputs).map(|(s, t)| Feed { seq: *s, input: Input::Tokens(t), wants_row: true }).collect();
        let rows = rt.step(&feeds).unwrap();
        assert_eq!(rows.len(), seqs.len());
        for (i, r) in rows.iter().enumerate() {
            rows_agree(&solo[i].1[k], r, reshape)
                .unwrap_or_else(|e| panic!("member {i}, round {k}: a batched round decodes what the sequence decodes alone: {e}"));
        }
    }
    for (s, p) in seqs.iter().zip(&prompts) {
        assert_eq!(rt.seq_len(*s), (p.len() + rounds) as u64);
    }
    let swapped = rt.seq_create().unwrap();
    row(&mut rt, swapped, &prompts[1]);
    let caught = (0..rounds).any(|k| {
        let r = row(&mut rt, swapped, &[solo[0].0[k]]);
        rows_agree(&solo[0].1[k], &r, reshape).is_err()
    });
    assert!(caught, "a lane decoding on another member's history must fall outside the tolerance");
    for s in seqs.into_iter().chain([swapped]) {
        rt.seq_free(s);
    }
}

#[test]
fn tokenizer_agrees_with_the_hf_tokenizer_of_the_same_checkpoint() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let Ok(hf_dir) = std::env::var("SUPERFLUID_TEST_GGUF_HF") else {
        eprintln!("SKIP: set SUPERFLUID_TEST_GGUF_HF to the checkpoint's HF snapshot");
        return;
    };
    let a: Arc<dyn Tokenizer> = Arc::new(LlamaTokenizer::load(&path).unwrap());
    let b: Arc<dyn Tokenizer> = Arc::new(superfluid_tokenizer_hf::HfTokenizer::load(std::path::Path::new(&hf_dir)).unwrap());
    let (va, vb) = (a.vocab_size(), b.vocab_size());
    assert!(va <= vb && vb - va <= 64, "id space: llama {va}, hf {vb}");
    let corpus = [
        "hi",
        "Hello, world!",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "  leading spaces and\ttabs\n\nnewlines",
        "日本語のテキスト and émojis 🚀🔥",
        "def f(x):\n    return x * 2  # comment",
    ];
    for s in corpus {
        assert_eq!(a.encode(s), b.encode(s), "encode {s:?}");
    }
    let (mut diff, mut control) = (0u32, 0u32);
    for t in 0..a.vocab_size() {
        let (x, y) = (a.token_bytes(t), b.token_bytes(t));
        if x == y {
            continue;
        }
        if x.is_empty() || y.is_empty() {
            control += 1;
            continue;
        }
        diff += 1;
        if diff <= 5 {
            eprintln!("token {t}: llama {x:?} hf {y:?}");
        }
    }
    eprintln!("{control} ids are control on one side only");
    assert!(diff <= 32, "{diff} ids decode differently");
    assert_eq!(a.eos_token(), b.eos_token(), "eos");
}

fn host_argmax(r: &[f32]) -> u32 {
    r.iter().enumerate().fold(0usize, |m, (i, &v)| if v > r[m] { i } else { m }) as u32
}

#[test]
fn engine_argmax_matches_the_host_argmax() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let prompts: Vec<Vec<u32>> =
        ["The capital of France is", "Once upon a time,", "def fibonacci(n):"].iter().map(|t| tok.encode(t)).collect();
    let mut rt = runtime(&path);
    assert!(rt.describe().engine_sampling);
    let greedy = SampleSpec { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, rng_position: 0 };
    let mut streams = Vec::new();
    for engine in [false, true] {
        let seqs: Vec<u64> = prompts.iter().map(|_| rt.seq_create().unwrap()).collect();
        let mut next = Vec::new();
        for (s, p) in seqs.iter().zip(&prompts) {
            let row = rt.step(&[Feed { seq: *s, input: Input::Tokens(p), wants_row: true }]).unwrap().remove(0);
            next.push(host_argmax(&row));
        }
        let mut toks: Vec<Vec<u32>> = vec![Vec::new(); seqs.len()];
        for _ in 0..6 {
            for (i, t) in next.iter().enumerate() {
                toks[i].push(*t);
            }
            let feeds: Vec<Feed> = seqs
                .iter()
                .zip(&next)
                .map(|(s, t)| Feed { seq: *s, input: Input::Tokens(std::slice::from_ref(t)), wants_row: true })
                .collect();
            next = if engine {
                rt.step_sampled(&feeds, &vec![greedy; feeds.len()]).unwrap().expect("greedy specs are honoured")
            } else {
                rt.step(&feeds).unwrap().iter().map(|r| host_argmax(r)).collect()
            };
        }
        streams.push(toks);
        for s in seqs {
            rt.seq_free(s);
        }
    }
    assert_eq!(streams[1], streams[0], "the engine's argmax is the host's");
    let s = rt.seq_create().unwrap();
    rt.step(&[Feed { seq: s, input: Input::Tokens(&prompts[0]), wants_row: false }]).unwrap();
    let len = rt.seq_len(s);
    let warm = SampleSpec { temperature: 0.7, ..greedy };
    let one = [prompts[0][0]];
    assert_eq!(rt.step_sampled(&[Feed { seq: s, input: Input::Tokens(&one), wants_row: true }], &[warm]).unwrap(), None);
    assert_eq!(rt.seq_len(s), len, "a refused spec moves no state");
    rt.seq_free(s);
}

#[test]
fn handles_are_unique_and_shared_prefixes_count_once() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let prompt = tok.encode("The quick brown fox jumps over the lazy dog");
    let mut rt = runtime(&path);
    let d = rt.describe();
    assert!(d.copy_shares_cells);
    assert!(d.runtime_version.starts_with("llama.cpp "), "{}", d.runtime_version);
    let a = rt.seq_create().unwrap();
    rt.step(&[Feed { seq: a, input: Input::Tokens(&prompt), wants_row: false }]).unwrap();
    let n = prompt.len() as u64;
    assert_eq!(rt.mem_counters().cells_used, n);
    let copies: Vec<u64> = (0..3)
        .map(|_| {
            let c = rt.seq_create().unwrap();
            rt.seq_copy(a, c, n).unwrap();
            c
        })
        .collect();
    assert_eq!(rt.mem_counters().cells_used, n, "three copies share the source's cells");
    for &c in &copies {
        rt.step(&[Feed { seq: c, input: Input::Tokens(&prompt[..2]), wants_row: false }]).unwrap();
    }
    assert_eq!(rt.mem_counters().cells_used, n + 6);
    rt.seq_free(a);
    assert_eq!(rt.mem_counters().cells_used, n + 6);
    assert_eq!(rt.seq_len(a), 0);
    assert_eq!(rt.seq_copy(a, copies[0], 1), Err(PrimError::UnknownSeq), "a freed handle is unknown");
    let b = rt.seq_create().unwrap();
    assert!(b > copies[2], "handles are monotonic: {b} after {}", copies[2]);
    assert_ne!(b, a);
    if d.recurrent {
        assert_eq!(rt.seq_truncate(copies[1], 2), Err(PrimError::OutOfBoundary), "head-only: no cut below the head");
        assert_eq!(rt.mem_counters().cells_used, n + 6);
    } else {
        rt.seq_truncate(copies[1], 2).unwrap();
        assert_eq!(rt.mem_counters().cells_used, n + 4);
    }
    for c in copies {
        rt.seq_free(c);
    }
    rt.seq_free(b);
    assert_eq!(rt.mem_counters().cells_used, 0);
}

#[test]
fn the_ledger_follows_copy_chains_and_the_limits_follow_the_config() {
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

#[test]
fn the_pool_budget_counts_the_weights() {
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

#[test]
fn a_kv_cell_is_priced_at_what_the_context_stores() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF model");
        return;
    };
    let mut rt = runtime(&path);
    let per_token = rt.describe().kv_bytes_per_token;
    let stored = stored_per_cell(&mut rt);
    eprintln!("{}: a cell stores {stored} bytes, priced at {per_token}", path.file_name().unwrap_or_default().to_string_lossy());
    assert!(
        (per_token..=per_token + 64).contains(&stored),
        "a cell stores {stored} bytes, and the descriptor prices it at {per_token}"
    );
}

#[test]
fn a_window_is_sized_from_the_header_without_the_weights() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF model");
        return;
    };
    let sized = match superfluid_adapter_llamacpp::runtime::sizing(&path) {
        Ok(s) => s,
        Err(why) if why.contains("sizes no window on") || why.contains("registers no GPU") => {
            eprintln!("SKIP: {why}");
            return;
        }
        Err(why) => panic!("{why}"),
    };
    let rt = runtime(&path);
    assert_eq!(sized.kv_bytes_per_token, rt.describe().kv_bytes_per_token, "the header prices a cell as the load does");
    assert_eq!(sized.weight_bytes, std::fs::metadata(&path).unwrap().len());
    assert!(sized.trained_context >= 2048, "{sized:?}");
    let device = superfluid_adapter_llamacpp::devices().into_iter().next().expect("a GPU");
    assert!(sized.budget_bytes > 0 && sized.budget_bytes <= device.memory, "{sized:?} on {device:?}");
    let window = superfluid_executor::sizing::window(&[sized], 4).expect("a window");
    assert!(window <= sized.trained_context && (window.is_multiple_of(1024) || window == sized.trained_context), "{window}");
    eprintln!("{}: {window} tokens at 4 lanes from {sized:?}", path.display());
}

fn stored_per_cell(rt: &mut LlamaRuntime) -> u64 {
    let toks: Vec<u32> = (0..64u32).map(|i| 100 + i).collect();
    let s = rt.seq_create().unwrap();
    rt.step(&[Feed { seq: s, input: Input::Tokens(&toks[..16]), wants_row: false }]).unwrap();
    let short = rt.seq_export(s, 16, encoding::LOSSLESS).unwrap().len() as u64;
    rt.step(&[Feed { seq: s, input: Input::Tokens(&toks[16..]), wants_row: false }]).unwrap();
    let long = rt.seq_export(s, 64, encoding::LOSSLESS).unwrap().len() as u64;
    rt.seq_free(s);
    (long - short) / 48
}

#[test]
fn a_hybrids_kv_cell_is_priced_by_the_layers_that_attend() {
    if let Err(why) = superfluid_adapter_llamacpp::sys::load() {
        eprintln!("SKIP: {why}");
        return;
    }
    let Some(path) = std::env::var_os("SUPERFLUID_TEST_GGUF_HYBRID").map(PathBuf::from).filter(|p| p.is_file()) else {
        eprintln!("SKIP: set SUPERFLUID_TEST_GGUF_HYBRID to a hybrid GGUF (Qwen3.5, Qwen3.6)");
        return;
    };
    let mut rt = runtime(&path);
    let d = rt.describe();
    assert!(d.recurrent, "a hybrid keeps recurrent state");
    let per_token = d.kv_bytes_per_token;
    let stored = stored_per_cell(&mut rt);
    eprintln!("{}: a cell stores {stored} bytes, priced at {per_token}", path.file_name().unwrap_or_default().to_string_lossy());
    assert!(
        (per_token..=per_token + 64).contains(&stored),
        "a cell stores {stored} bytes, and the descriptor prices it at {per_token}"
    );
}

#[test]
fn a_hybrids_layer_map_is_read_before_its_interval() {
    use tiny_gguf::{qwen35, Said};
    if let Err(why) = superfluid_adapter_llamacpp::sys::load() {
        eprintln!("SKIP: {why}");
        return;
    }
    let dir = std::env::temp_dir().join(format!("superfluid-llama-layer-map-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cases = [
        (
            "map-and-interval",
            qwen35(&[true, false, true, false])
                .says("attention.recurrent_layers", Said::Bools(&[true, false, true, false]))
                .says("full_attention_interval", Said::U32(4)),
            2,
        ),
        ("interval-alone", qwen35(&[true, false, true, false]).says("full_attention_interval", Said::U32(2)), 2),
        ("map-alone", qwen35(&[true, false, false, true]).says("attention.recurrent_layers", Said::U32s(&[1, 0, 0, 1])), 2),
    ];
    let mut wrong = Vec::new();
    for (name, model, attending) in cases {
        let path = dir.join(format!("{name}.gguf"));
        model.write(&path);
        let mut rt = runtime(&path);
        let d = rt.describe();
        assert!(d.recurrent, "{name}: a hybrid's architecture");
        let (priced, stored) = (d.kv_bytes_per_token, stored_per_cell(&mut rt));
        eprintln!("{name}: a cell stores {stored} bytes, priced at {priced}");
        if priced != attending * 512 || !(priced..=priced + 64).contains(&stored) {
            wrong.push(format!("{name}: a cell stores {stored} bytes, priced at {priced}; {attending} layers attend"));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_layout_llama_cpp_does_not_read_is_not_read_here() {
    use tiny_gguf::{llama, Said};
    if let Err(why) = superfluid_adapter_llamacpp::sys::load() {
        eprintln!("SKIP: {why}");
        return;
    }
    let dir = std::env::temp_dir().join(format!("superfluid-llama-stray-layout-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("stray-layout.gguf");
    llama(4).says("attention.recurrent_layers", Said::U32(1)).says("full_attention_interval", Said::U32(4)).write(&path);
    let mut rt = runtime(&path);
    let d = rt.describe();
    assert!(!d.recurrent, "attention on every layer");
    let (priced, stored) = (d.kv_bytes_per_token, stored_per_cell(&mut rt));
    eprintln!("stray-layout: a cell stores {stored} bytes, priced at {priced}");
    assert_eq!(priced, 4 * 256, "every layer holds KV, whatever the two keys say");
    assert!((priced..=priced + 64).contains(&stored), "a cell stores {stored} bytes, and the descriptor prices it at {priced}");
    drop(rt);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cells_released_counts_only_what_no_other_sequence_holds() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF model");
        return;
    };
    let mut rt = runtime(&path);
    let toks: Vec<u32> = (0..11u32).map(|i| 100 + i).collect();
    let a = rt.seq_create().unwrap();
    rt.step(&[Feed { seq: a, input: Input::Tokens(&toks[..8]), wants_row: false }]).unwrap();
    let b = rt.seq_create().unwrap();
    rt.seq_copy(a, b, 8).unwrap();
    assert_eq!(rt.cells_released(&[b]), 0, "the copy holds nothing of its own");
    assert_eq!(rt.cells_released(&[a]), 0, "the copy keeps the prefix alive");
    assert_eq!(rt.cells_released(&[a, b]), 8);
    rt.step(&[Feed { seq: b, input: Input::Tokens(&toks[8..]), wants_row: false }]).unwrap();
    assert_eq!(rt.cells_released(&[b]), 3, "only what the copy grew");
    assert_eq!(rt.cells_released(&[a, b]), 11);
    for (seq, expect) in [(b, 3), (a, 8)] {
        let before = rt.mem_counters().cells_used;
        assert_eq!(rt.cells_released(&[seq]), expect);
        rt.seq_free(seq);
        assert_eq!(before - rt.mem_counters().cells_used, expect, "the pool gives back what was predicted");
    }
}

#[test]
fn a_sliding_window_model_is_reused_only_at_its_head() {
    if let Err(why) = superfluid_adapter_llamacpp::sys::load() {
        eprintln!("SKIP: {why}");
        return;
    }
    let Some(path) = std::env::var_os("SUPERFLUID_TEST_GGUF_SWA").map(PathBuf::from).filter(|p| p.is_file()) else {
        eprintln!("SKIP: set SUPERFLUID_TEST_GGUF_SWA to a sliding-window GGUF (gemma-3)");
        return;
    };
    let mut rt = runtime(&path);
    let d = rt.describe();
    assert!(!d.recurrent, "attention all the way: not a recurrent model");
    assert!(!d.truncate_partial, "and still cut only at its head");
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let prompt = tok.encode("The quick brown fox jumps over the lazy dog near the quiet river bank.");
    let n = prompt.len() as u64;
    let a = rt.seq_create().unwrap();
    rt.step(&[Feed { seq: a, input: Input::Tokens(&prompt), wants_row: false }]).unwrap();
    let b = rt.seq_create().unwrap();
    assert_eq!(rt.seq_copy(a, b, n - 2), Err(PrimError::OutOfBoundary), "no copy below the head");
    assert_eq!(rt.seq_truncate(a, n - 2), Err(PrimError::OutOfBoundary), "no cut below the head");
    assert_eq!(rt.seq_boundary(a, n - 2), 0, "nothing of it serves a shorter prefix");
    assert_eq!(rt.seq_boundary(a, n + 8), n);
    rt.seq_copy(a, b, n).unwrap();
    let warm = row(&mut rt, b, &prompt[..1]);
    let c = rt.seq_create().unwrap();
    let mut cold_input = prompt.clone();
    cold_input.push(prompt[0]);
    let cold = row(&mut rt, c, &cold_input);
    let (row_tol, _) = tolerances(&rt);
    rows_agree(&cold, &warm, row_tol).expect("a head copy continues as the cold sequence does");
}

mod tiny_gguf {
    use std::path::Path;

    const N_EMBD: u64 = 64;
    const N_FF: u64 = 128;
    const N_VOCAB: u64 = 256;

    pub enum Said {
        U32(u32),
        F32(f32),
        Str(&'static str),
        U32s(&'static [u32]),
        I32s(&'static [i32]),
        Bools(&'static [bool]),
    }

    pub struct Model {
        arch: &'static str,
        said: Vec<(String, Said)>,
        tensors: Vec<(String, Vec<u64>, f32)>,
    }

    impl Model {
        fn new(arch: &'static str, n_layer: usize) -> Model {
            let mut m = Model { arch, said: Vec::new(), tensors: Vec::new() };
            m.said.push(("general.architecture".into(), Said::Str(arch)));
            m.said.push(("tokenizer.ggml.model".into(), Said::Str("none")));
            for (key, value) in [
                ("vocab_size", N_VOCAB),
                ("context_length", 4096),
                ("embedding_length", N_EMBD),
                ("block_count", n_layer as u64),
                ("feed_forward_length", N_FF),
                ("attention.head_count", 1),
                ("attention.head_count_kv", 1),
            ] {
                m = m.says(key, Said::U32(value as u32));
            }
            m = m.says("attention.layer_norm_rms_epsilon", Said::F32(1e-6));
            m.tensors.push(("token_embd.weight".into(), vec![N_EMBD, N_VOCAB], 0.0));
            m.tensors.push(("output_norm.weight".into(), vec![N_EMBD], 1.0));
            m
        }

        pub fn says(mut self, key: &str, value: Said) -> Model {
            self.said.push((format!("{}.{key}", self.arch), value));
            self
        }

        fn block(&mut self, il: usize, norms: &[(&str, u64)], weights: &[(&str, [u64; 2])]) {
            for (name, n) in norms {
                self.tensors.push((format!("blk.{il}.{name}"), vec![*n], 1.0));
            }
            for (name, shape) in weights {
                self.tensors.push((format!("blk.{il}.{name}"), shape.to_vec(), 0.0));
            }
        }

        pub fn write(&self, path: &Path) {
            fn string(out: &mut Vec<u8>, s: &str) {
                out.extend((s.len() as u64).to_le_bytes());
                out.extend(s.as_bytes());
            }
            let padded = |n: usize| n.div_ceil(32) * 32;
            let mut out = Vec::new();
            out.extend(b"GGUF");
            out.extend(3u32.to_le_bytes());
            out.extend((self.tensors.len() as u64).to_le_bytes());
            out.extend((self.said.len() as u64).to_le_bytes());
            for (key, value) in &self.said {
                string(&mut out, key);
                match value {
                    Said::U32(v) => {
                        out.extend(4u32.to_le_bytes());
                        out.extend(v.to_le_bytes());
                    }
                    Said::F32(v) => {
                        out.extend(6u32.to_le_bytes());
                        out.extend(v.to_le_bytes());
                    }
                    Said::Str(s) => {
                        out.extend(8u32.to_le_bytes());
                        string(&mut out, s);
                    }
                    Said::U32s(a) => {
                        out.extend(9u32.to_le_bytes());
                        out.extend(4u32.to_le_bytes());
                        out.extend((a.len() as u64).to_le_bytes());
                        a.iter().for_each(|v| out.extend(v.to_le_bytes()));
                    }
                    Said::I32s(a) => {
                        out.extend(9u32.to_le_bytes());
                        out.extend(5u32.to_le_bytes());
                        out.extend((a.len() as u64).to_le_bytes());
                        a.iter().for_each(|v| out.extend(v.to_le_bytes()));
                    }
                    Said::Bools(a) => {
                        out.extend(9u32.to_le_bytes());
                        out.extend(7u32.to_le_bytes());
                        out.extend((a.len() as u64).to_le_bytes());
                        out.extend(a.iter().map(|&b| b as u8));
                    }
                }
            }
            let mut offset = 0;
            for (name, shape, _) in &self.tensors {
                string(&mut out, name);
                out.extend((shape.len() as u32).to_le_bytes());
                shape.iter().for_each(|d| out.extend(d.to_le_bytes()));
                out.extend(0u32.to_le_bytes());
                out.extend((offset as u64).to_le_bytes());
                offset += padded(shape.iter().product::<u64>() as usize * 4);
            }
            out.resize(padded(out.len()), 0);
            for (_, shape, fill) in &self.tensors {
                for _ in 0..shape.iter().product::<u64>() {
                    out.extend(fill.to_le_bytes());
                }
                out.resize(padded(out.len()), 0);
            }
            std::fs::write(path, out).expect("write the model");
        }
    }

    pub fn llama(n_layer: usize) -> Model {
        let mut m = Model::new("llama", n_layer);
        for il in 0..n_layer {
            m.block(
                il,
                &[("attn_norm.weight", N_EMBD), ("ffn_norm.weight", N_EMBD)],
                &[
                    ("attn_q.weight", [N_EMBD, N_EMBD]),
                    ("attn_k.weight", [N_EMBD, N_EMBD]),
                    ("attn_v.weight", [N_EMBD, N_EMBD]),
                    ("attn_output.weight", [N_EMBD, N_EMBD]),
                    ("ffn_gate.weight", [N_EMBD, N_FF]),
                    ("ffn_down.weight", [N_FF, N_EMBD]),
                    ("ffn_up.weight", [N_EMBD, N_FF]),
                ],
            );
        }
        m
    }

    pub fn qwen35(recurrent: &[bool]) -> Model {
        const HEAD: u64 = 128;
        let mut m = Model::new("qwen35", recurrent.len())
            .says("attention.key_length", Said::U32(HEAD as u32))
            .says("attention.value_length", Said::U32(HEAD as u32))
            .says("rope.dimension_count", Said::U32(32))
            .says("rope.dimension_sections", Said::I32s(&[4, 4, 8, 0]))
            .says("ssm.conv_kernel", Said::U32(4))
            .says("ssm.inner_size", Said::U32(HEAD as u32))
            .says("ssm.state_size", Said::U32(HEAD as u32))
            .says("ssm.time_step_rank", Said::U32(1))
            .says("ssm.group_count", Said::U32(1));
        for (il, &recurrent) in recurrent.iter().enumerate() {
            m.block(
                il,
                &[("attn_norm.weight", N_EMBD), ("post_attention_norm.weight", N_EMBD)],
                &[("ffn_gate.weight", [N_EMBD, N_FF]), ("ffn_down.weight", [N_FF, N_EMBD]), ("ffn_up.weight", [N_EMBD, N_FF])],
            );
            if recurrent {
                m.block(
                    il,
                    &[("ssm_norm.weight", HEAD)],
                    &[
                        ("attn_qkv.weight", [N_EMBD, 3 * HEAD]),
                        ("attn_gate.weight", [N_EMBD, HEAD]),
                        ("ssm_conv1d.weight", [4, 3 * HEAD]),
                        ("ssm_dt.bias", [1, 1]),
                        ("ssm_a", [1, 1]),
                        ("ssm_beta.weight", [N_EMBD, 1]),
                        ("ssm_alpha.weight", [N_EMBD, 1]),
                        ("ssm_out.weight", [HEAD, N_EMBD]),
                    ],
                );
            } else {
                m.block(
                    il,
                    &[("attn_q_norm.weight", HEAD), ("attn_k_norm.weight", HEAD)],
                    &[
                        ("attn_q.weight", [N_EMBD, 2 * HEAD]),
                        ("attn_k.weight", [N_EMBD, HEAD]),
                        ("attn_v.weight", [N_EMBD, HEAD]),
                        ("attn_output.weight", [HEAD, N_EMBD]),
                    ],
                );
            }
        }
        m
    }
}

#[test]
fn lanes_moved_into_adjacent_buffers_decode_as_they_were() {
    let Some(path) = gguf() else {
        eprintln!("SKIP: no GGUF");
        return;
    };
    let tok = LlamaTokenizer::load(&path).expect("tokenizer");
    let texts = ["The quick brown fox", "Once upon a time there was", "In the beginning", "A list of prime numbers:", "The capital of France is"];
    let prompts: Vec<Vec<u32>> = texts.iter().map(|t| tok.encode(t)).collect();
    let open = || {
        // Six lanes: eight buffers, room for the copies and imports below.
        LlamaRuntime::open(LlamaConfig { model_path: path.clone(), max_seq_len: 256, max_batch: 6, streams: Some(true), ..Default::default() })
            .expect("open gguf")
    };
    let round = |rt: &mut LlamaRuntime, seqs: &[u64]| -> Vec<Vec<f32>> {
        let feeds: Vec<Feed<'_>> = seqs.iter().map(|s| Feed { seq: *s, input: Input::Tokens(&[42]), wants_row: true }).collect();
        rt.step(&feeds).unwrap()
    };
    let close = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);

    let mut together = open();
    let kept = [0usize, 2, 4];
    let mut seqs = Vec::new();
    for &k in &kept {
        let s = together.seq_create().unwrap();
        together.step(&[Feed { seq: s, input: Input::Tokens(&prompts[k]), wants_row: false }]).unwrap();
        seqs.push(s);
    }
    let want = round(&mut together, &seqs);
    let want_next = round(&mut together, &seqs);

    let mut apart = open();
    assert!(!apart.describe().copy_shares_cells, "a buffer per sequence");
    let mut all = Vec::new();
    for p in &prompts {
        let s = apart.seq_create().unwrap();
        apart.step(&[Feed { seq: s, input: Input::Tokens(p), wants_row: false }]).unwrap();
        all.push(s);
    }
    apart.seq_free(all[1]);
    apart.seq_free(all[3]);
    let left = [all[0], all[2], all[4]];
    let got = round(&mut apart, &left);
    let got_next = round(&mut apart, &left);
    let (reshape, _) = tolerances(&apart);
    for (i, (g, w)) in got.iter().zip(&want).chain(got_next.iter().zip(&want_next)).enumerate() {
        let d = close(g, w);
        assert!(d < reshape, "row {i} moved by {d} when its sequence changed buffers (tolerance {reshape})");
    }
    for (s, k) in left.iter().zip(kept) {
        assert_eq!(apart.seq_len(*s), prompts[k].len() as u64 + 2, "its length came with it");
    }

    let n = prompts[0].len() as u64;
    let copy = apart.seq_create().unwrap();
    apart.seq_copy(left[0], copy, n).unwrap();
    assert_eq!(apart.seq_len(copy), n);
    let fresh = apart.seq_create().unwrap();
    apart.step(&[Feed { seq: fresh, input: Input::Tokens(&prompts[0]), wants_row: false }]).unwrap();
    let rows = apart.step(&[
        Feed { seq: copy, input: Input::Tokens(&[7]), wants_row: true },
        Feed { seq: fresh, input: Input::Tokens(&[7]), wants_row: true },
    ])
    .unwrap();
    let d = close(&rows[0], &rows[1]);
    assert!(d < reshape, "the cut copy and a sequence fed the same tokens differ by {d} (tolerance {reshape})");
    assert_eq!(apart.seq_boundary(left[0], n), n);
    let prefix = apart.seq_export(left[0], n, superfluid_abi::encoding::LOSSLESS).unwrap();
    let back = apart.seq_create().unwrap();
    assert_eq!(apart.seq_import(back, &prefix).unwrap(), n);
    let rows = apart.step(&[
        Feed { seq: back, input: Input::Tokens(&[9]), wants_row: true },
        Feed { seq: copy, input: Input::Tokens(&[9]), wants_row: false },
    ])
    .unwrap();
    let again = apart.seq_create().unwrap();
    apart.seq_copy(fresh, again, n).unwrap();
    let want = apart.step(&[Feed { seq: again, input: Input::Tokens(&[9]), wants_row: true }]).unwrap();
    let d = close(&rows[0], &want[0]);
    assert!(d < reshape, "an exported prefix, imported, and a cut copy differ by {d} (tolerance {reshape})");
}
