//! Shared rig for the basert speculation tests (spec_conformance, spec_gpu_sampling).

#![allow(dead_code, clippy::too_many_arguments)]
pub use std::os::unix::net::UnixStream;
pub use std::path::PathBuf;
pub use std::sync::Mutex;
pub use std::thread;

pub use superfluid_agent::WorkerClient;
pub use superfluid_engine::Engine;
pub use superfluid_engine_ffi::{FfiEngine, FfiEngineConfig, NativeEngine};
pub use superfluid_proto::linkw;
pub use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};
pub use superfluid_worker::{WorkerConfig, WorkerServer};

pub static MODEL_LOCK: Mutex<()> = Mutex::new(());

pub fn model_path() -> Option<PathBuf> {
    let p = model_path_found()?;
    if p.extension().is_some_and(|e| e == "base") {
        superfluid_engine_ffi::libbasert::require()?;
    }
    Some(p)
}

fn model_path_found() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_MODEL") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    [
        "models/Qwen3-0.6B-Q4_K_M.base",
        "models/Qwen3-0.6B-Q4_K_M.gguf",
    ]
    .iter()
    .map(|c| root.join(c))
    .find(|p| p.exists())
}

pub fn spawn_native(path: PathBuf) -> (WorkerClient, thread::JoinHandle<()>) {
    spawn_engine(move || {
        let engine = NativeEngine::load(config(path)).expect("model load");
        let vocab = engine.vocab();
        (engine, vocab)
    })
}

pub fn spawn_ffi(path: PathBuf) -> (WorkerClient, thread::JoinHandle<()>) {
    spawn_engine(move || {
        let engine = FfiEngine::load(config(path)).expect("model load");
        let vocab = engine.vocab();
        (engine, vocab)
    })
}

fn config(path: PathBuf) -> FfiEngineConfig {
    FfiEngineConfig {
        model_path: path,
        max_context: 4096,
        max_batch_size: 8,
        seed_ttl_ticks: 64,
    }
}

pub fn spawn_engine<E, F>(make: F) -> (WorkerClient, thread::JoinHandle<()>)
where
    E: superfluid_engine::Engine,
    F: FnOnce() -> (E, u32) + Send + 'static,
{
    let (frames_a, frames_w) = UnixStream::pair().expect("socketpair");
    let (fds_a, fds_w) = UnixStream::pair().expect("fd socketpair");
    let handle = thread::spawn(move || {
        let (engine, vocab) = make();
        let specs = vec![
            linkw::RingSpec {
                ring_id: TOKEN_RING_IN,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + 4096 * 4,
                slots: 64,
            },
            linkw::RingSpec {
                ring_id: TOKEN_RING_OUT,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + 1024 * 4,
                slots: 64,
            },
            linkw::RingSpec {
                ring_id: LOGITS_RING,
                kind: linkw::RingKind::Logits,
                slot_bytes: 16 + vocab * 4,
                slots: 8,
            },
        ];
        let server = WorkerServer::new(
            engine,
            WorkerConfig {
                engine_bundle_hash: [0xBA; 32],
            },
            frames_w,
            fds_w,
        )
        .with_ring_specs(specs);
        server.serve().expect("worker serve");
    });
    let client = WorkerClient::connect(frames_a, fds_a).expect("connect");
    (client, handle)
}

pub fn plan(plan_seq: u64) -> linkw::TickPlanMsg {
    linkw::TickPlanMsg {
        plan_seq,
        flags: 0,
        prefill_token_budget: 4096,
        max_decode_lanes: 8,
        admits: vec![],
        commits: vec![],
        prefills: vec![],
        decodes: vec![],
        retires: vec![],
        shed_policy: linkw::ShedPolicyMsg {
            victim_lanes: vec![],
            evictable_cache_classes: u64::MAX,
            protected_quota_bytes: 0,
            max_evict_bytes: u64::MAX,
        },
    }
}

pub fn admit(
    lane: u64,
    prompt: linkw::TokenRefMsg,
    sampled: bool,
    seed: u64,
    slot: u32,
) -> linkw::LaneAdmitMsg {
    linkw::LaneAdmitMsg {
        lane_tag: lane,
        prompt,
        seed_handle: 0,
        sampling: if sampled { 1 } else { 0 },
        params: linkw::SamplingParamsMsg {
            temperature: if sampled { 0.8 } else { 0.0 },
            top_p: 0.95,
            min_p: 0.0,
            top_k: 40,
            freq_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_penalty: 0.0,
            flags: 0,
        },
        rng_counter_base: seed,
        grammar_handle: 0,
        logit_bias_handle: 0,
        want_logprobs: false,
        top_logprobs: 0,
        strategy_slot: slot,
        determinism_class: 0,
        minimum_exactness: 0,
        allow_approximate: false,
        required_cert_id: 0,
        host_sampler_identity: [0; 32],
        grammar_replay: 0,
        decode_replay: 0,
    }
}

pub fn assert_greedy_matches(id: &str, on: &[u32], off: &[u32]) {
    let bitexact = std::env::var("BASERT_SPEC_BITEXACT")
        .map(|v| v != "0")
        .unwrap_or(false);
    let common = off
        .iter()
        .zip(on.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let n = off.len().min(on.len());
    if bitexact {
        if on != off {
            eprintln!("{id} greedy (bit-exact): identical prefix {common}/{n}\n  off={off:?}\n  on ={on:?}");
        }
        assert_eq!(
            on, off,
            "{id} speculation must not change greedy output (bit-exact routing)"
        );
        return;
    }
    if common < n {
        eprintln!("{id} greedy: identical prefix {common}/{n} (a near-tie flipped under the tile-class numerics)");
    }
    assert!(
        common * 2 >= n,
        "{id} greedy streams diverge early ({common}/{n})"
    );
}

pub fn prompt_lookup_reg() -> linkw::StrategyRegisterMsg {
    linkw::StrategyRegisterMsg {
        strategy_id: "prompt-lookup".into(),
        impl_version: "1.0.0".into(),
        impl_hash: [0x11; 32],
        config_hash: [0x22; 32],
        artifacts: vec![],
        taps: vec![],
        capabilities: vec![linkw::CapabilityReqMsg {
            kind_id: superfluid_abi::strategy_cap::PROPOSAL_LINEAR,
            params: vec![],
        }],
        target_archs: vec![],
        kernel_caps_required: 0,
        est_state_bytes: 0,
        claimed_exactness: superfluid_abi::exactness::SEED_PATH_INVARIANT,
        rng_contract_version: 1,
    }
}

pub fn generate(
    client: &mut WorkerClient,
    seq: &mut u64,
    lane: u64,
    prompt: &[u32],
    n: u32,
    sampled: bool,
    seed: u64,
    slot: u32,
) -> (Vec<u32>, u32, u32) {
    let pref = client.stage_prompt(prompt).unwrap();
    let mut p = plan(*seq);
    *seq += 1;
    p.admits.push(admit(lane, pref, sampled, seed, slot));
    p.prefills.push(linkw::LanePrefillMsg {
        lane_tag: lane,
        token_offset: 0,
        token_count: prompt.len() as u32,
    });
    let ev = client.tick(p).unwrap();
    let ar = ev
        .admit_results
        .iter()
        .find(|a| a.lane_tag == lane)
        .unwrap();
    assert_eq!(
        ar.status,
        superfluid_abi::admit_status::ADMITTED,
        "admit: {ar:?}"
    );
    let mut out = Vec::new();
    let (mut proposed, mut accepted) = (0u32, 0u32);
    while (out.len() as u32) < n {
        let want = (n - out.len() as u32).min(32) as u16;
        let mut p = plan(*seq);
        *seq += 1;
        p.decodes.push(linkw::LaneDecodeMsg {
            lane_tag: lane,
            max_new_tokens: want,
            overshoot: 0,
    });
        let ev = client.tick(p).unwrap();
        let emit = ev.emits.iter().find(|e| e.lane_tag == lane).unwrap();
        let toks = client.read_tokens(&emit.token_ref).unwrap();
        out.extend(toks);
        proposed += emit.spec.proposed;
        accepted += emit.spec.accepted;
        if emit.finish != superfluid_abi::finish::NONE {
            break;
        }
    }
    let mut p = plan(*seq);
    *seq += 1;
    p.retires.push(linkw::LaneRetireMsg {
        lane_tag: lane,
        publish_to_cache: false,
    });
    client.tick(p).unwrap();
    (out, proposed, accepted)
}

pub fn arch_is_recurrent(arch: &str) -> bool {
    arch.starts_with("qwen35")
        || arch.starts_with("qwen36")
        || arch.starts_with("qwen3next")
        || arch.starts_with("nemotron")
        || arch.starts_with("mamba")
}

pub fn gguf_architecture(path: &std::path::Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 24 || &bytes[0..4] != b"GGUF" {
        return None;
    }
    let mut at = 4usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = bytes.get(at..at + n)?;
        at += n;
        Some(s)
    };
    let _version = u32::from_le_bytes(take(4)?.try_into().ok()?);
    let _n_tensors = u64::from_le_bytes(take(8)?.try_into().ok()?);
    let n_kv = u64::from_le_bytes(take(8)?.try_into().ok()?);
    fn scalar_size(ty: u32) -> Option<usize> {
        Some(match ty {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            _ => return None,
        })
    }
    fn value<'a>(take: &mut impl FnMut(usize) -> Option<&'a [u8]>, ty: u32) -> Option<Option<String>> {
        match ty {
            8 => {
                let len = u64::from_le_bytes(take(8)?.try_into().ok()?) as usize;
                let s = take(len)?;
                Some(Some(String::from_utf8_lossy(s).into_owned()))
            }
            9 => {
                let ety = u32::from_le_bytes(take(4)?.try_into().ok()?);
                let count = u64::from_le_bytes(take(8)?.try_into().ok()?);
                for _ in 0..count {
                    value(take, ety)?;
                }
                Some(None)
            }
            _ => {
                take(scalar_size(ty)?)?;
                Some(None)
            }
        }
    }
    for _ in 0..n_kv {
        let klen = u64::from_le_bytes(take(8)?.try_into().ok()?) as usize;
        let key = String::from_utf8_lossy(take(klen)?).into_owned();
        let ty = u32::from_le_bytes(take(4)?.try_into().ok()?);
        let v = value(&mut take, ty)?;
        if key == "general.architecture" {
            return v;
        }
    }
    None
}

pub fn bundle_is_recurrent(path: &std::path::Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut hdr = [0u8; 16];
    if f.read_exact(&mut hdr).is_err() {
        return false;
    }
    if &hdr[0..4] == b"GGUF" {
        return gguf_architecture(path)
            .map(|a| arch_is_recurrent(&a))
            .unwrap_or(false);
    }
    if &hdr[0..4] != b"BASE" {
        return false;
    }
    let len = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
    let mut json = vec![0u8; len.min(64 << 20)];
    if f.read_exact(&mut json).is_err() {
        return false;
    }
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&json) else {
        return false;
    };
    arch_is_recurrent(v.get("arch").and_then(|a| a.as_str()).unwrap_or(""))
}

pub fn bundle_has_speculator(path: &std::path::Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut hdr = [0u8; 16];
    if f.read_exact(&mut hdr).is_err() || &hdr[0..4] != b"BASE" {
        return false;
    }
    let len = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
    let mut json = vec![0u8; len.min(64 << 20)];
    if f.read_exact(&mut json).is_err() {
        return false;
    }
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&json) else {
        return false;
    };
    if v.get("speculator").is_some() {
        return true;
    }
    let layers = v
        .get("config")
        .and_then(|c| c.get("num_hidden_layers"))
        .and_then(|n| n.as_u64());
    let want = layers.map(|l| format!("layers.{l}.eh_proj.weight"));
    v.get("arch").and_then(|a| a.as_str()) == Some("glm_dsa")
        && v.get("tensors")
            .and_then(|t| t.as_array())
            .is_some_and(|t| {
                t.iter()
                    .any(|e| e.get("name").and_then(|n| n.as_str()) == want.as_deref())
            })
}

pub fn repetitive_prompt() -> Vec<u32> {
    let mut v = Vec::new();
    for _ in 0..6 {
        v.extend_from_slice(&[3000, 3005, 3010, 3015, 3020, 3025, 3030, 3035]);
    }
    v
}
