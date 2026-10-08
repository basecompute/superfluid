//! A probe, not a gate: what a llama.cpp step costs against what its KV
//! cache holds, and where it holds it. Prints tables; asserts nothing. The
//! numbers behind `LlamaConfig`'s cache limits come from it (b11284, Metal,
//! Qwen3-4B Q4_K_M); run it again on another backend or build before
//! trusting them there.
//!
//! ```text
//! SUPERFLUID_TEST_GGUF=<model.gguf> cargo test --release -p superfluid-adapter-llamacpp \
//!     --test kv_cost_probe -- --ignored --nocapture --test-threads 1
//! ```
//!
//! The pool is `PROBE_LANES` x `PROBE_MAX_SEQ_LEN` cells (4 x 8192 unless
//! set), one shared pool unless `PROBE_STREAMS=1` asks for a buffer per
//! sequence. `PROBE_ONLY` picks experiments by name (`held`, `lengths`,
//! `export`; comma separated). `PROBE_FILL=0` runs batches as they are fed
//! (see `LlamaConfig::fill_batches`).
//!
//! A step that asks for no row returns before the device has run it, and
//! the next call that reads a result waits for both: every timing here
//! starts after a step that read one.

use std::path::PathBuf;
use std::time::Instant;

use superfluid_abi::encoding;
use superfluid_adapter_llamacpp::{LlamaConfig, LlamaRuntime, LlamaTokenizer};
use superfluid_engine::Tokenizer;
use superfluid_executor::{Feed, Input, RuntimePrimitives, SampleSpec, Seq};

const PARA: &str = "The quick brown fox jumps over the lazy dog near the quiet river bank while the sun sets behind the hills. ";
const GREEDY: SampleSpec = SampleSpec { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, rng_position: 0 };

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn wanted(name: &str) -> bool {
    match std::env::var("PROBE_ONLY") {
        Ok(only) => only.split(',').any(|n| n.trim() == name),
        Err(_) => true,
    }
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Probe {
    rt: LlamaRuntime,
    /// A long run of ordinary text, as tokens: every sequence is a slice of it.
    text: Vec<u32>,
    lanes: usize,
}

impl Probe {
    fn open() -> Option<Probe> {
        if let Err(why) = superfluid_adapter_llamacpp::sys::load() {
            eprintln!("SKIP: {why}");
            return None;
        }
        let path = PathBuf::from(std::env::var("SUPERFLUID_TEST_GGUF").ok()?);
        let lanes = env_u32("PROBE_LANES", 4);
        let rt = LlamaRuntime::open(LlamaConfig {
            model_path: path.clone(),
            max_seq_len: env_u32("PROBE_MAX_SEQ_LEN", 8192),
            max_batch: lanes,
            streams: Some(env_u32("PROBE_STREAMS", 0) != 0),
            fill_batches: env_u32("PROBE_FILL", 1) != 0,
            ..Default::default()
        })
        .expect("open gguf");
        let tok = LlamaTokenizer::load(&path).expect("tokenizer");
        let text = tok.encode(&PARA.repeat(420));
        let d = rt.describe();
        eprintln!(
            "# {} on {} ({}), pool {} cells{}, {} lanes, {} bytes a cell",
            path.file_name().unwrap().to_string_lossy(),
            d.runtime_version,
            d.backend,
            d.cells_total,
            if d.copy_shares_cells { " shared" } else { " in a buffer per sequence" },
            d.max_batch,
            d.kv_bytes_per_token
        );
        Some(Probe { rt, text, lanes: lanes as usize })
    }

    /// A sequence holding `n` tokens of state.
    fn held(&mut self, n: usize, salt: usize) -> Seq {
        let s = self.rt.seq_create().expect("a free sequence");
        let start = salt % 16;
        self.rt.step(&[Feed { seq: s, input: Input::Tokens(&self.text[start..start + n]), wants_row: false }]).expect("fill");
        s
    }

    /// One step of a fresh sequence over `n` tokens, with its row read: the
    /// milliseconds it took.
    fn prompt_ms(&mut self, n: usize, at: usize) -> f64 {
        let s = self.rt.seq_create().expect("a free sequence");
        let t0 = Instant::now();
        let feeds = [Feed { seq: s, input: Input::Tokens(&self.text[at..at + n]), wants_row: true }];
        self.rt.step_sampled(&feeds, &[GREEDY]).expect("prefill").expect("greedy");
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        self.rt.seq_free(s);
        ms
    }

    /// `lanes` fresh sequences with a 64-token prompt each, then 112 packed
    /// greedy rounds: the median milliseconds of a round after the first 16.
    fn round_ms(&mut self, lanes: usize) -> f64 {
        let seqs: Vec<Seq> = (0..lanes).map(|_| self.rt.seq_create().expect("a free sequence")).collect();
        let mut next: Vec<u32> = Vec::new();
        for (i, &s) in seqs.iter().enumerate() {
            let at = 32 + 7 * i;
            let feeds = [Feed { seq: s, input: Input::Tokens(&self.text[at..at + 64]), wants_row: true }];
            next.push(self.rt.step_sampled(&feeds, &[GREEDY]).expect("prefill").expect("greedy")[0]);
        }
        let mut ms = Vec::new();
        for _ in 0..112 {
            let feeds: Vec<Feed> =
                seqs.iter().zip(&next).map(|(s, t)| Feed { seq: *s, input: Input::Tokens(std::slice::from_ref(t)), wants_row: true }).collect();
            let t0 = Instant::now();
            let picked = self.rt.step_sampled(&feeds, &vec![GREEDY; feeds.len()]).expect("decode").expect("greedy");
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
            next = picked;
        }
        for s in seqs {
            self.rt.seq_free(s);
        }
        median(&ms[16..])
    }

    fn line(&mut self, what: &str) {
        let used = self.rt.mem_counters().cells_used;
        let (many, one) = (self.round_ms(self.lanes), self.round_ms(1));
        eprintln!(
            "{what:<40} {used:>6} cells in use   {} lanes {many:6.2} ms a round, {:6.1} tok/s   1 lane {one:6.2} ms, {:5.1} tok/s",
            self.lanes,
            self.lanes as f64 * 1e3 / many,
            1e3 / one
        );
    }
}

/// A decode round against the cells other sequences hold, and against where
/// they hold them: the cells in use, or the highest cell in use. A small
/// sequence is placed above 24k cells of others, which are then freed: 64
/// cells are in use, and the highest of them is past 24k.
#[test]
#[ignore = "a measurement: needs SUPERFLUID_TEST_GGUF and a quiet machine"]
fn a_decode_round_against_cells_held() {
    if !wanted("held") {
        return;
    }
    let Some(mut p) = Probe::open() else { return };
    p.round_ms(p.lanes); // warm the device
    p.line("nothing held");
    let mut fillers: Vec<Seq> = Vec::new();
    for target in [4096usize, 8192, 16384, 24576] {
        while fillers.len() * 2048 < target {
            let f = p.held(2048, fillers.len());
            fillers.push(f);
        }
        p.line(&format!("{} sequences of 2048 held", fillers.len()));
    }
    let above = p.held(64, 3);
    for f in fillers.drain(..) {
        p.rt.seq_free(f);
    }
    p.line("those freed, 64 cells left above them");
    p.rt.seq_free(above);
    p.line("nothing held");
    // Every other one of eight freed: 8k in use, the highest near 16k.
    let eight: Vec<Seq> = (0..8).map(|i| p.held(2048, i)).collect();
    for s in eight.iter().step_by(2) {
        p.rt.seq_free(*s);
    }
    p.line("8 x 2048 held, every other one freed");
    for s in eight.iter().skip(1).step_by(2) {
        p.rt.seq_free(*s);
    }
    let four: Vec<Seq> = (0..4).map(|i| p.held(2048, i)).collect();
    p.line("4 x 2048 held, packed");
    for s in four {
        p.rt.seq_free(s);
    }
}

/// A prompt's time by its token count, against cells held: which counts pay
/// for the held cells, and how much. `PROBE_LENGTHS` and `PROBE_HELD` (in
/// sequences of `PROBE_FILLER` tokens) replace the defaults.
#[test]
#[ignore = "a measurement: needs SUPERFLUID_TEST_GGUF and a quiet machine"]
fn a_prompt_by_its_length_against_cells_held() {
    if !wanted("lengths") {
        return;
    }
    let list = |name: &str, default: &[usize]| -> Vec<usize> {
        match std::env::var(name) {
            Ok(l) => l.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
            Err(_) => default.to_vec(),
        }
    };
    let Some(mut p) = Probe::open() else { return };
    p.round_ms(p.lanes);
    let filler = env_u32("PROBE_FILLER", 2900) as usize;
    let lengths = list("PROBE_LENGTHS", &[1, 4, 8, 16, 19, 20, 21, 24, 40, 41, 47, 48, 100, 132, 340, 512, 2896, 2900]);
    let mut fillers: Vec<Seq> = Vec::new();
    for target in list("PROBE_HELD", &[0, 2, 4, 6, 8]) {
        while fillers.len() < target {
            let f = p.held(filler, fillers.len());
            fillers.push(f);
        }
        let used = p.rt.mem_counters().cells_used;
        p.prompt_ms(1, 0); // the device finishes the fill
        let mut out = format!("{used:>6} held, ms by tokens:");
        for &n in &lengths {
            let ms: Vec<f64> = (0..5).map(|i| p.prompt_ms(n, 300 + 3 * i)).collect();
            out.push_str(&format!(" {n}:{:.1}", median(&ms[1..])));
        }
        eprintln!("{out}");
    }
}

/// What moving a sequence out of the cache and back costs: the export of
/// its state to host memory, and the import into a fresh sequence.
#[test]
#[ignore = "a measurement: needs SUPERFLUID_TEST_GGUF and a quiet machine"]
fn export_and_import() {
    if !wanted("export") {
        return;
    }
    let Some(mut p) = Probe::open() else { return };
    p.round_ms(p.lanes);
    for n in [256usize, 1024, 2900, 2900, 6000, 6000] {
        let s = p.held(n, 1);
        p.prompt_ms(1, 0); // the device finishes the fill
        let t0 = Instant::now();
        let blob = p.rt.seq_export(s, n as u64, encoding::LOSSLESS).unwrap();
        let out = t0.elapsed().as_secs_f64() * 1e3;
        p.rt.seq_free(s);
        let fresh = p.rt.seq_create().unwrap();
        let t0 = Instant::now();
        assert_eq!(p.rt.seq_import(fresh, &blob).unwrap(), n as u64);
        let back = t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        p.rt.step_sampled(&[Feed { seq: fresh, input: Input::Tokens(&p.text[..1]), wants_row: true }], &[GREEDY]).unwrap();
        let step = t0.elapsed().as_secs_f64() * 1e3;
        p.rt.seq_free(fresh);
        eprintln!("{n:>5} tokens, {:7.1} MB: export {out:6.1} ms, import {back:6.1} ms, the step after {step:5.1} ms", blob.len() as f64 / 1e6);
    }
}
