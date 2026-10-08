//! MLX behind the generic executor, on a REAL model.

mod common;

use superfluid_abi::*;
use superfluid_adapter_mlx::MlxRuntime;
use superfluid_engine::testing::{rows_agree, tokens_agree, top2_margin, Harness};
use superfluid_engine::{Engine, Tokenizer};
use superfluid_executor::{Executor, Feed, Input, PrimError, RuntimePrimitives, SampleSpec};
use superfluid_tokenizer_hf::HfTokenizer;
use common::{harness, mlx_dir, runtime, MODEL_LOCK};

fn greedy(h: &mut Harness<Executor<MlxRuntime>>, lane: u64, prompt: &[u32], n: u16) -> Vec<u32> {
    let p = h.prompt(prompt);
    let ev = h.tick_ok(h.plan().admit(lane, p).prefill(lane, 0, prompt.len() as u32).decode(lane, n));
    h.rings_tokens(&ev.emit_for(lane).token_ref)
}

fn greedy_with_margins(h: &mut Harness<Executor<MlxRuntime>>, lane: u64, prompt: &[u32], n: u16) -> (Vec<u32>, Vec<f32>) {
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

fn argmax(r: &[f32]) -> usize {
    r.iter().enumerate().fold(0usize, |m, (i, &v)| if v > r[m] { i } else { m })
}

const RESHAPE_TOLERANCE: f32 = 2.5;

fn row(rt: &mut MlxRuntime, seq: u64, tokens: &[u32]) -> Vec<f32> {
    rt.step(&[Feed { seq, input: Input::Tokens(tokens), wants_row: true }]).unwrap().remove(0)
}

#[test]
fn a_real_models_tool_call_is_held_to_the_tools_schema() {
    use superfluid_daemon::wal::role;
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let tok = std::sync::Arc::new(HfTokenizer::load(&path).expect("tokenizer"));
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
fn descriptor_matches_the_artifact() {
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let tok = HfTokenizer::load(&path).expect("tokenizer");
    let rt = runtime(&path);
    let d = rt.describe();
    assert_eq!(d.runtime_id, "mlx");
    assert!(d.runtime_version.starts_with("mlx "), "{}", d.runtime_version);
    assert_eq!(d.vocab_size, tok.vocab_size());
    assert_eq!(d.page_size_tokens, 1);
    assert!(d.kv_bytes_per_token > 0);
    assert_eq!(d.export_encodings, vec![encoding::LOSSLESS]);
    assert!(rt.is_eos(tok.eos_token()));
    assert!(!rt.eos().is_empty());
    assert_ne!(d.weights_identity, [0u8; 32]);
}

#[test]
fn a_window_is_sized_from_the_model_without_its_weights() {
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let sized = superfluid_adapter_mlx::runtime::sizing(superfluid_adapter_mlx::venv_or_env(None).as_deref(), &path).expect("sizing");
    let rt = runtime(&path);
    // The load sizes its pool by what a token measured when that is more
    // than the dimensions price it at; the price is never above it.
    assert!(
        sized.kv_bytes_per_token <= rt.describe().kv_bytes_per_token,
        "the lazy model prices a token at most as the load does: {} against {}",
        sized.kv_bytes_per_token,
        rt.describe().kv_bytes_per_token
    );
    let weights: u64 = std::fs::read_dir(&path)
        .unwrap()
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
        .map(|e| std::fs::metadata(e.path()).unwrap().len())
        .sum();
    assert_eq!(sized.weight_bytes, weights);
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let trained = config.get("max_position_embeddings").or_else(|| config["text_config"].get("max_position_embeddings"));
    assert_eq!(Some(sized.trained_context), trained.and_then(|v| v.as_u64()));
    assert!(sized.budget_bytes > 0, "{sized:?}");
    let window = superfluid_executor::sizing::window(&[sized], 4).expect("a window");
    eprintln!("{}: {window} tokens at 4 lanes from {sized:?}", path.display());
}

#[test]
fn greedy_is_stable_and_warm_seed_continues_exactly() {
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let tok = HfTokenizer::load(&path).expect("tokenizer");
    // Longer than the adapter's 16-token seed minimum, so the warm seed below
    // is servable.
    let prompt = tok.encode("Here is a short fact about European geography, stated plainly. The capital of France is");
    assert!(prompt.len() >= 16, "{} tokens", prompt.len());
    let mut h = harness(&path);
    let (a, a_margins) = greedy_with_margins(&mut h, 1, &prompt, 12);
    let b = greedy(&mut h, 2, &prompt, 12);
    assert_eq!(a, b, "greedy decode is stable across lanes");
    let text = String::from_utf8_lossy(&a.iter().flat_map(|&t| tok.token_bytes(t)).collect::<Vec<_>>()).into_owned();
    eprintln!("greedy: {text:?}");
    assert!(text.contains("Paris"), "a chat model names Paris: {text:?}");

    h.tick_ok(h.plan().retire(1, true).retire(2, false));
    let mut longer = prompt.clone();
    longer.extend_from_slice(&a[..4]);
    let cold = greedy(&mut h, 3, &longer, 6);
    h.tick_ok(h.plan().retire(3, false));
    if h.engine.descriptor().recurrent {
        assert_eq!(
            h.engine.seed_acquire(&longer, prompt.len() as u64, determinism::BEST_EFFORT).err(),
            Some(Status::SeedUnservable),
            "head-only: a seed below the head is unservable"
        );
        let reference = greedy(&mut h, 5, &prompt, 18);
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
        assert_eq!(warm, reference[13..18].to_vec(), "a seed at the head continues the exact stream");
        return;
    }
    tokens_agree(&a[4..10], &a_margins[4..10], &cold, RESHAPE_TOLERANCE)
        .unwrap_or_else(|e| panic!("the cold continuation is the original tail up to a near-tie: {e}"));
    let p = h.prompt(&longer);
    h.tick_ok(h.plan().admit(8, p).prefill(8, 0, prompt.len() as u32));
    let ev = h.tick_ok(h.plan().prefill(8, prompt.len() as u32, 4).decode(8, 6));
    let cold_chunks = h.rings_tokens(&ev.emit_for(8).token_ref);
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
    assert_eq!(warm, cold_chunks, "a warm seed continues exactly as the same chunks prefilled cold");
}

#[test]
fn park_export_and_resume_by_adoption_continue_exactly() {
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let tok = HfTokenizer::load(&path).expect("tokenizer");
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
    let uninterrupted = greedy(&mut h3, 1, &prompt, 14);
    assert_eq!(resumed, uninterrupted[10..].to_vec(), "resume from a parked artifact continues the exact stream");
}

#[test]
fn batched_decode_rounds_match_solo_decode() {
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let tok = HfTokenizer::load(&path).expect("tokenizer");
    let prompts: Vec<Vec<u32>> = ["The capital of France is", "Once upon a time, in a small village by the sea,", "def fibonacci(n):"]
        .iter()
        .map(|t| tok.encode(t))
        .collect();
    let mut rt = runtime(&path);
    let rounds = 8;
    let solo: Vec<(Vec<u32>, Vec<Vec<f32>>)> = prompts.iter().map(|p| solo_decode(&mut rt, p, rounds)).collect();
    let seqs: Vec<u64> = prompts.iter().map(|_| rt.seq_create().unwrap()).collect();
    for (s, p) in seqs.iter().zip(&prompts) {
        row(&mut rt, *s, p);
    }
    let mut last = Vec::new();
    for k in 0..rounds {
        let inputs: Vec<[u32; 1]> = solo.iter().map(|(t, _)| [t[k]]).collect();
        let feeds: Vec<Feed> =
            seqs.iter().zip(&inputs).map(|(s, t)| Feed { seq: *s, input: Input::Tokens(t), wants_row: true }).collect();
        let rows = rt.step(&feeds).unwrap();
        assert_eq!(rows.len(), seqs.len());
        for (i, r) in rows.iter().enumerate() {
            rows_agree(&solo[i].1[k], r, RESHAPE_TOLERANCE)
                .unwrap_or_else(|e| panic!("member {i}, round {k}: a batched round decodes what the sequence decodes alone: {e}"));
        }
        last = rows;
    }
    for (s, p) in seqs.iter().zip(&prompts) {
        assert_eq!(rt.seq_len(*s), (p.len() + rounds) as u64);
    }
    let swapped = rt.seq_create().unwrap();
    row(&mut rt, swapped, &prompts[1]);
    let caught = (0..rounds).any(|k| {
        let r = row(&mut rt, swapped, &[solo[0].0[k]]);
        rows_agree(&solo[0].1[k], &r, RESHAPE_TOLERANCE).is_err()
    });
    assert!(caught, "a lane decoding on another member's cache must fall outside the tolerance");
    rt.seq_free(swapped);
    let next: Vec<u32> = last.iter().map(|r| argmax(r) as u32).collect();
    let len0 = rt.seq_len(seqs[0]);
    let payload = rt.seq_export(seqs[0], len0, encoding::LOSSLESS).unwrap();
    let c = rt.seq_create().unwrap();
    assert_eq!(rt.seq_import(c, &payload).unwrap(), len0);
    let d = rt.seq_create().unwrap();
    rt.seq_copy(seqs[1], d, rt.seq_len(seqs[1])).unwrap();
    let t0 = next[0];
    let t1 = next[1];
    let from_export = row(&mut rt, c, &[t0]);
    let member0 = row(&mut rt, seqs[0], &[t0]);
    rows_agree(&member0, &from_export, 0.0)
        .unwrap_or_else(|e| panic!("an export taken after batched rounds continues bit for bit: {e}"));
    let from_copy = row(&mut rt, d, &[t1]);
    let member1 = row(&mut rt, seqs[1], &[t1]);
    rows_agree(&member1, &from_copy, 0.0)
        .unwrap_or_else(|e| panic!("a copy taken after batched rounds continues bit for bit: {e}"));
    let (a1, a2) = ([t1], [next[2]]);
    let sub: Vec<Feed> = vec![
        Feed { seq: seqs[1], input: Input::Tokens(&a1), wants_row: true },
        Feed { seq: seqs[2], input: Input::Tokens(&a2), wants_row: true },
    ];
    let rows = rt.step(&sub).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rt.seq_len(seqs[0]), len0 + 1);
    let m = rt.mem_counters();
    assert_eq!(m.cells_used, [seqs[0], seqs[1], seqs[2], c, d].iter().map(|s| rt.seq_len(*s)).sum::<u64>());
    for s in seqs.into_iter().chain([c, d]) {
        rt.seq_free(s);
    }
    assert_eq!(rt.mem_counters().cells_used, 0);
}

#[test]
fn a_round_the_pool_cannot_pad_decodes_every_lane_as_it_would_alone() {
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let tok = HfTokenizer::load(&path).expect("tokenizer");
    let long = "The quick brown fox jumps over the lazy dog. ".repeat(24);
    let prompts: Vec<Vec<u32>> =
        [long.as_str(), "The capital of France is", "def fibonacci(n):"].iter().map(|t| tok.encode(t)).collect();
    let rounds = 6;
    let own: usize = prompts.iter().map(|p| p.len() + rounds).sum();
    assert!(3 * prompts[0].len() > own + 64, "the prompts must differ enough for padding to matter");
    for (cells_total, merged) in [((own + 64) as u32, 2), (0, 3)] {
        let mut rt = MlxRuntime::open(superfluid_adapter_mlx::MlxConfig {
            model_path: path.clone(),
            max_seq_len: 512,
            max_batch: 8,
            cells_total,
            ..Default::default()
        })
        .expect("open mlx model");
        let solo: Vec<(Vec<u32>, Vec<Vec<f32>>)> = prompts.iter().map(|p| solo_decode(&mut rt, p, rounds)).collect();
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
            assert_eq!(rt.merged_rows(), merged, "pool of {cells_total} cells, round {k}");
            for (i, r) in rows.iter().enumerate() {
                rows_agree(&solo[i].1[k], r, RESHAPE_TOLERANCE)
                    .unwrap_or_else(|e| panic!("pool of {cells_total} cells, member {i}, round {k}: {e}"));
            }
        }
        for (s, p) in seqs.iter().zip(&prompts) {
            assert_eq!(rt.seq_len(*s), (p.len() + rounds) as u64);
        }
        for s in seqs {
            rt.seq_free(s);
        }
        assert_eq!(rt.mem_counters().cells_used, 0);
    }
}

#[test]
fn engine_argmax_matches_the_host_argmax() {
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let tok = HfTokenizer::load(&path).expect("tokenizer");
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
            next.push(argmax(&row) as u32);
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
                rt.step(&feeds).unwrap().iter().map(|r| argmax(r) as u32).collect()
            };
        }
        streams.push(toks);
        for s in seqs {
            rt.seq_free(s);
        }
    }
    assert_eq!(streams[1], streams[0], "the device's argmax is the host's");
    let s = rt.seq_create().unwrap();
    let row = rt.step(&[Feed { seq: s, input: Input::Tokens(&prompts[1]), wants_row: true }]).unwrap().remove(0);
    let one = [argmax(&row) as u32];
    let c = rt.seq_create().unwrap();
    rt.seq_copy(s, c, rt.seq_len(s)).unwrap();
    let host = argmax(&rt.step(&[Feed { seq: s, input: Input::Tokens(&one), wants_row: true }]).unwrap()[0]) as u32;
    let dev = rt.step_sampled(&[Feed { seq: c, input: Input::Tokens(&one), wants_row: true }], &[greedy]).unwrap().unwrap()[0];
    assert_eq!(dev, host);
    assert!(rt.describe().engine_draws);
    let warm = SampleSpec { temperature: 0.7, top_k: 20, top_p: 0.8, min_p: 0.0, rng_position: 41 };
    let two = [dev];
    let h = rt.seq_create().unwrap();
    rt.seq_copy(c, h, rt.seq_len(c)).unwrap();
    let mut row = rt.step(&[Feed { seq: h, input: Input::Tokens(&two), wants_row: true }]).unwrap().remove(0);
    let logits = row.clone();
    let params = superfluid_abi::SamplingParams { temperature: 0.7, top_k: 20, top_p: 0.8, ..Default::default() };
    let host = superfluid_executor::sampling::sample(&mut row, &params, superfluid_executor::sampling::uniform(41));
    let drawn = rt
        .step_sampled(&[Feed { seq: c, input: Input::Tokens(&two), wants_row: true }], &[warm])
        .unwrap()
        .expect("one sampling for every lane is drawn on the device")[0];
    assert!(drawn == host || logits[drawn as usize] == logits[host as usize], "the device drew {drawn}, the host draws {host}");
    assert_eq!(rt.seq_len(c), rt.seq_len(h), "the draw's pass is the step's");
    let (len_c, len_h) = (rt.seq_len(c), rt.seq_len(h));
    let other = SampleSpec { temperature: 1.1, ..warm };
    let feeds = [
        Feed { seq: c, input: Input::Tokens(&one), wants_row: true },
        Feed { seq: h, input: Input::Tokens(&one), wants_row: true },
    ];
    assert_eq!(rt.step_sampled(&feeds, &[warm, other]).unwrap(), None);
    assert_eq!((rt.seq_len(c), rt.seq_len(h)), (len_c, len_h), "a refused round moves no state");
    rt.seq_free(s);
    rt.seq_free(c);
    rt.seq_free(h);
}

#[test]
fn state_primitives_cut_copy_and_round_trip() {
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let tok = HfTokenizer::load(&path).expect("tokenizer");
    let prompt = tok.encode("Numbers: one two three four five six seven");
    let mut rt = runtime(&path);
    let a = rt.seq_create().unwrap();
    let rows = rt.step(&[Feed { seq: a, input: Input::Tokens(&prompt), wants_row: true }]).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].len(), rt.describe().vocab_size as usize);
    assert_eq!(rt.seq_len(a), prompt.len() as u64);
    if rt.describe().recurrent {
        let plen = prompt.len() as u64;
        let b = rt.seq_create().unwrap();
        assert_eq!(rt.seq_copy(a, b, 3), Err(PrimError::OutOfBoundary), "no copy below the head");
        rt.seq_copy(a, b, plen).unwrap();
        assert_eq!(rt.seq_export(a, 5, encoding::LOSSLESS).err(), Some(PrimError::OutOfBoundary), "no export below the head");
        let payload = rt.seq_export(a, plen, encoding::LOSSLESS).unwrap();
        let c = rt.seq_create().unwrap();
        assert_eq!(rt.seq_import(c, &payload).unwrap(), plen);
        let next = [prompt[0]];
        let ra = rt.step(&[Feed { seq: a, input: Input::Tokens(&next), wants_row: true }]).unwrap();
        let rb = rt.step(&[Feed { seq: b, input: Input::Tokens(&next), wants_row: true }]).unwrap();
        let rc = rt.step(&[Feed { seq: c, input: Input::Tokens(&next), wants_row: true }]).unwrap();
        assert_eq!(argmax(&rb[0]), argmax(&ra[0]), "a head copy continues like its source");
        assert_eq!(argmax(&rc[0]), argmax(&ra[0]), "an imported head continues like the original");
        assert_eq!(rt.seq_truncate(c, 2), Err(PrimError::OutOfBoundary), "no cut below the head");
        rt.seq_truncate(c, 0).unwrap();
        assert_eq!(rt.seq_len(c), 0);
        assert_eq!(rt.mem_counters().cells_used, rt.seq_len(a) + rt.seq_len(b));
        rt.seq_free(a);
        rt.seq_free(b);
        rt.seq_free(c);
        assert_eq!(rt.mem_counters().cells_used, 0);
        return;
    }
    let b = rt.seq_create().unwrap();
    rt.seq_copy(a, b, 3).unwrap();
    assert_eq!(rt.seq_len(b), 3);
    let rb = rt.step(&[Feed { seq: b, input: Input::Tokens(&prompt[3..]), wants_row: true }]).unwrap();
    assert_eq!(rt.seq_len(b), prompt.len() as u64);
    assert_eq!(rt.seq_len(a), prompt.len() as u64, "the source is untouched");
    assert_eq!(argmax(&rb[0]), argmax(&rows[0]), "the copy's continuation agrees with the source's");
    let payload = rt.seq_export(a, 5, encoding::LOSSLESS).unwrap();
    let c = rt.seq_create().unwrap();
    assert_eq!(rt.seq_import(c, &payload).unwrap(), 5);
    let rc = rt.step(&[Feed { seq: c, input: Input::Tokens(&prompt[5..]), wants_row: true }]).unwrap();
    assert_eq!(argmax(&rc[0]), argmax(&rows[0]), "an imported prefix continues like the original");
    rt.seq_truncate(c, 2).unwrap();
    assert_eq!(rt.seq_len(c), 2);
    assert_eq!(rt.seq_truncate(c, 3), Err(PrimError::OutOfBoundary));
    assert_eq!(rt.seq_export(c, 3, encoding::LOSSLESS).err(), Some(PrimError::OutOfBoundary));
    assert_eq!(rt.seq_export(c, 2, encoding::Q8).err(), Some(PrimError::Unsupported));
    assert_eq!(rt.seq_import(c, &payload).err(), Some(PrimError::OutOfBoundary), "import needs an empty sequence");
    let m = rt.mem_counters();
    assert_eq!(m.cells_used, rt.seq_len(a) + rt.seq_len(b) + rt.seq_len(c));
    assert!(m.allocated_bytes > 0);
    rt.seq_free(a);
    rt.seq_free(b);
    rt.seq_free(c);
    assert_eq!(rt.mem_counters().cells_used, 0);
    assert_eq!(rt.seq_len(a), 0);
}

fn solo_decode(rt: &mut MlxRuntime, prompt: &[u32], rounds: usize) -> (Vec<u32>, Vec<Vec<f32>>) {
    let s = rt.seq_create().unwrap();
    let mut r = row(rt, s, prompt);
    let (mut toks, mut rows) = (Vec::new(), Vec::new());
    for _ in 0..rounds {
        let t = argmax(&r) as u32;
        toks.push(t);
        r = row(rt, s, &[t]);
        rows.push(r.clone());
    }
    rt.seq_free(s);
    (toks, rows)
}

#[test]
fn a_hybrids_kv_cell_is_priced_by_the_layers_that_attend() {
    let Some(path) = std::env::var_os("SUPERFLUID_TEST_MLX_HYBRID").map(std::path::PathBuf::from).filter(|p| p.join("config.json").is_file()) else {
        eprintln!("SKIP: set SUPERFLUID_TEST_MLX_HYBRID to a hybrid MLX model directory (LFM2)");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let mut rt = runtime(&path);
    let d = rt.describe();
    assert!(d.recurrent, "a hybrid keeps recurrent state");
    let per_token = d.kv_bytes_per_token;
    let toks: Vec<u32> = (0..64u32).map(|i| 100 + i).collect();
    let s = rt.seq_create().unwrap();
    rt.step(&[Feed { seq: s, input: Input::Tokens(&toks[..16]), wants_row: false }]).unwrap();
    let short = rt.seq_export(s, 16, encoding::LOSSLESS).unwrap().len() as u64;
    rt.step(&[Feed { seq: s, input: Input::Tokens(&toks[16..]), wants_row: false }]).unwrap();
    let long = rt.seq_export(s, 64, encoding::LOSSLESS).unwrap().len() as u64;
    let stored = (long - short) / 48;
    eprintln!("{}: a cell stores {stored} bytes, priced at {per_token}", path.file_name().unwrap_or_default().to_string_lossy());
    assert!(
        (per_token..=per_token + 64).contains(&stored),
        "a cell stores {stored} bytes, and the descriptor prices it at {per_token}"
    );
    rt.seq_free(s);
}

#[test]
fn freed_and_cut_state_goes_back() {
    let Some(path) = mlx_dir() else {
        eprintln!("SKIP: no MLX model");
        return;
    };
    let _g = MODEL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let tok = HfTokenizer::load(&path).expect("tokenizer");
    let mut rt = runtime(&path);
    let per_token = rt.describe().kv_bytes_per_token;
    let prompts: Vec<Vec<u32>> =
        ["The capital of France is", "Once upon a time,", "def fibonacci(n):"].iter().map(|t| tok.encode(t)).collect();
    let rounds = 8;
    let solo: Vec<(Vec<u32>, Vec<Vec<f32>>)> = prompts.iter().map(|p| solo_decode(&mut rt, p, rounds)).collect();
    let baseline = rt.mem_counters().allocated_bytes;

    let seqs: Vec<u64> = prompts.iter().map(|_| rt.seq_create().unwrap()).collect();
    for (s, p) in seqs.iter().zip(&prompts) {
        row(&mut rt, *s, p);
    }
    let mut live = vec![0usize, 1, 2];
    for round in 0..rounds {
        if round == rounds / 2 {
            rt.seq_free(seqs[1]);
            live = vec![0, 2];
        }
        let inputs: Vec<[u32; 1]> = live.iter().map(|&i| [solo[i].0[round]]).collect();
        let feeds: Vec<Feed> = live
            .iter()
            .zip(&inputs)
            .map(|(&i, t)| Feed { seq: seqs[i], input: Input::Tokens(t), wants_row: true })
            .collect();
        let rows = rt.step(&feeds).unwrap();
        for (k, &i) in live.iter().enumerate() {
            let what = if round < rounds / 2 { "a member decodes as it would alone" } else { "a survivor decodes on as it would alone" };
            rows_agree(&solo[i].1[round], &rows[k], RESHAPE_TOLERANCE)
                .unwrap_or_else(|e| panic!("member {i}, round {round}: {what}: {e}"));
        }
    }
    rt.seq_free(seqs[0]);
    rt.seq_free(seqs[2]);
    let m = rt.mem_counters();
    assert_eq!(m.cells_used, 0);
    assert!(
        m.allocated_bytes <= baseline + 16 * per_token,
        "every freed sequence's state went back: {} bytes against a {baseline}-byte baseline",
        m.allocated_bytes
    );

    if !rt.describe().truncate_partial {
        return;
    }
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(60);
    let long = tok.encode(&text);
    let long = &long[..long.len().min(448)];
    let keep = 64u64;
    assert!(long.len() as u64 >= keep + 256, "a cut that frees a growth step needs a longer prompt");
    let a = rt.seq_create().unwrap();
    rt.step(&[Feed { seq: a, input: Input::Tokens(long), wants_row: false }]).unwrap();
    let b = rt.seq_create().unwrap();
    rt.seq_copy(a, b, keep).unwrap();
    rt.seq_truncate(a, keep).unwrap();
    let m = rt.mem_counters();
    assert_eq!(m.cells_used, 2 * keep);
    assert!(
        m.allocated_bytes <= baseline + (2 * keep + 256) * per_token,
        "a cut and a partial copy hold their prefix, not the {}-token buffer: {} bytes over the baseline",
        long.len(),
        m.allocated_bytes.saturating_sub(baseline)
    );
    let c = rt.seq_create().unwrap();
    rt.step(&[Feed { seq: c, input: Input::Tokens(&long[..keep as usize]), wants_row: false }]).unwrap();
    let t = [long[keep as usize]];
    let rows: Vec<Vec<f32>> = [a, b, c]
        .iter()
        .map(|s| rt.step(&[Feed { seq: *s, input: Input::Tokens(&t), wants_row: true }]).unwrap().remove(0))
        .collect();
    rows_agree(&rows[2], &rows[0], RESHAPE_TOLERANCE)
        .unwrap_or_else(|e| panic!("a cut continues like a fresh prefill of its prefix: {e}"));
    rows_agree(&rows[0], &rows[1], 0.0)
        .unwrap_or_else(|e| panic!("a partial copy continues bit for bit as the cut holding the same prefix: {e}"));
    for s in [a, b, c] {
        rt.seq_free(s);
    }
    let d = rt.seq_create().unwrap();
    rt.step(&[Feed { seq: d, input: Input::Tokens(long), wants_row: false }]).unwrap();
    let mut len = long.len() as u64;
    while len > keep {
        len = len.saturating_sub(32).max(keep);
        rt.seq_truncate(d, len).unwrap();
    }
    let m = rt.mem_counters();
    assert_eq!(m.cells_used, keep);
    assert!(
        m.allocated_bytes <= baseline + (keep + 256) * per_token,
        "32-token cuts down to {keep} tokens hold under a growth step of slack: {} bytes over the baseline",
        m.allocated_bytes.saturating_sub(baseline)
    );
    rt.seq_free(d);
    assert!(rt.mem_counters().allocated_bytes <= baseline + 16 * per_token);
}
