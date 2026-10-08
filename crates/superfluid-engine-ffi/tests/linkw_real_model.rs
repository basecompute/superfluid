//! Link W end to end against the REAL engine, parameterized over BOTH back ends.

use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::thread;

use std::sync::Mutex;

use superfluid_agent::{argmax, WorkerClient};
use superfluid_engine::Engine;
use superfluid_engine_ffi::{FfiEngine, FfiEngineConfig, NativeEngine};
use superfluid_proto::linkw::{self, ALL_SPACES};
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};
use superfluid_worker::{WorkerConfig, WorkerServer};

const KV: u32 = 1;

fn model_path() -> Option<PathBuf> {
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
    ["models/Qwen3-0.6B-Q4_K_M.base", "models/Qwen3-0.6B-Q4_K_M.gguf"]
        .iter()
        .map(|c| root.join(c))
        .find(|p| p.exists())
}

static MODEL_LOCK: Mutex<()> = Mutex::new(());

fn hybrid_path() -> Option<PathBuf> {
    let p = hybrid_path_found()?;
    if p.extension().is_some_and(|e| e == "base") {
        superfluid_engine_ffi::libbasert::require()?;
    }
    Some(p)
}

fn hybrid_path_found() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_HYBRID") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let home = std::env::var("HOME").unwrap_or_default();
    [
        root.join("models/Qwen3.5-2B-Base-Q4.base"),
        PathBuf::from(&home).join(".cache/baseRT/models/Qwen/Qwen3.5-2B-Base/default-q4/model.base"),
        PathBuf::from(&home)
            .join("Library/Caches/baseRT/models/Qwen/Qwen3.5-2B-Base/default-q4/model.base"),
    ]
    .into_iter()
    .find(|p| p.exists())
}

#[derive(Clone, Copy, PartialEq)]
enum Backend {
    Ffi,
    Native,
}

fn engine_config(path: PathBuf) -> FfiEngineConfig {
    FfiEngineConfig {
        model_path: path,
        max_context: 4096,
        max_batch_size: 8,
        seed_ttl_ticks: 64,
    }
}

fn serve<E: Engine + 'static>(
    engine: E,
    vocab: u32,
    frames_w: UnixStream,
    fds_w: UnixStream,
) {
    let specs = ring_specs(vocab);
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
}

fn spawn_worker(path: PathBuf, backend: Backend) -> (WorkerClient, thread::JoinHandle<()>) {
    let (frames_a, frames_w) = UnixStream::pair().expect("socketpair");
    let (fds_a, fds_w) = UnixStream::pair().expect("fd socketpair");
    let handle = thread::spawn(move || match backend {
        Backend::Ffi => {
            let engine = FfiEngine::load(engine_config(path)).expect("model load");
            let vocab = engine.vocab();
            serve(engine, vocab, frames_w, fds_w);
        }
        Backend::Native => {
            let engine = NativeEngine::load(engine_config(path)).expect("model load");
            let vocab = engine.vocab();
            serve(engine, vocab, frames_w, fds_w);
        }
    });
    let client = WorkerClient::connect(frames_a, fds_a).expect("connect");
    (client, handle)
}

fn ring_specs(vocab: u32) -> Vec<linkw::RingSpec> {
    vec![
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
    ]
}

fn plan(plan_seq: u64) -> linkw::TickPlanMsg {
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

fn admit(lane: u64, prompt: linkw::TokenRefMsg) -> linkw::LaneAdmitMsg {
    linkw::LaneAdmitMsg {
        lane_tag: lane,
        prompt,
        seed_handle: 0,
        sampling: 0,
        params: linkw::SamplingParamsMsg {
            temperature: 0.0,
            top_p: 1.0,
            min_p: 0.0,
            top_k: 0,
            freq_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_penalty: 0.0,
            flags: 0,
        },
        rng_counter_base: 0,
        grammar_handle: 0,
        logit_bias_handle: 0,
        want_logprobs: false,
                top_logprobs: 0,
        strategy_slot: 0,
        determinism_class: 0,
        minimum_exactness: 0,
        allow_approximate: false,
        required_cert_id: 0,
        host_sampler_identity: [0; 32],
        grammar_replay: 0,
        decode_replay: 0,
    }
}

fn run_full_lifecycle(backend: Backend) {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model (set BASERT_TEST_MODEL or place models/Qwen3-0.6B-Q4_K_M.*)");
        return;
    };
    let (mut client, worker) = spawn_worker(path, backend);

    assert_eq!(client.hello.chosen_version, superfluid_agent::client::PROTO_VERSION);
    let kv_desc = client
        .hello
        .state_space_descriptors
        .iter()
        .find(|d| d.space_id == KV)
        .expect("kv.full advertised");
    assert_eq!(kv_desc.kind, 1);
    let page = kv_desc.page_size_tokens as u64;
    assert!(page > 0, "paged KV must advertise its block size");
    assert!(kv_desc.bytes_per_token > 0);
    let logits_spec = client
        .hello
        .limits
        .ring_specs
        .iter()
        .find(|s| s.ring_id == LOGITS_RING)
        .expect("logits ring advertised");
    let vocab = ((logits_spec.slot_bytes - 16) / 4) as usize;
    assert!(vocab > 10_000, "vocab-sized logits slots");

    let prompt: Vec<u32> = (0..64).map(|i| 1000 + i * 7).collect();
    let pref = client.stage_prompt(&prompt).unwrap();
    let mut p1 = plan(1);
    p1.admits.push(admit(7, pref));
    p1.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 7,
        token_offset: 0,
        token_count: 64,
    });
    p1.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 7,
        max_new_tokens: 4,
        overshoot: 0,
});
    let ev = client.tick(p1).unwrap();
    assert_eq!(ev.admit_results[0].status, 1);
    let emit = &ev.emits[0];
    assert!(
        (1..=4).contains(&emit.n_tokens),
        "greedy decode emitted {} tokens",
        emit.n_tokens
    );
    let generated = client.read_tokens(&emit.token_ref).unwrap();
    assert_eq!(generated.len(), emit.n_tokens as usize);
    assert!(generated.iter().all(|&t| (t as usize) < vocab));

    let pref2 = client.stage_prompt(&prompt).unwrap();
    let mut p2 = plan(2);
    p2.admits.push(admit(11, pref2));
    p2.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 11,
        token_offset: 0,
        token_count: 64,
    });
    p2.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 11,
        max_new_tokens: 4,
        overshoot: 0,
});
    let ev = client.tick(p2).unwrap();
    let emit2 = ev.emits.iter().find(|e| e.lane_tag == 11).unwrap();
    let generated2 = client.read_tokens(&emit2.token_ref).unwrap();
    assert_eq!(
        generated, generated2,
        "greedy decode must be deterministic across identical lanes"
    );

    let bad_prompt = vec![u32::MAX, 1, 2];
    let bad_ref = client.stage_prompt(&bad_prompt).unwrap();
    let mut pbad = plan(100);
    pbad.admits.push(admit(99, bad_ref));
    pbad.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 99,
        token_offset: 0,
        token_count: 3,
    });
    match client.tick(pbad).unwrap_err() {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -102),
        other => panic!("expected bounds rejection, got {other:?}"),
    }

    {
        let replay_ref = client.stage_prompt(&[1, 2, 3, 4]).unwrap();
        let mut preplay = plan(101);
        let mut a = admit(98, replay_ref);
        a.decode_replay = 2;
        preplay.admits.push(a);
        preplay.prefills.push(linkw::LanePrefillMsg {
            lane_tag: 98,
            token_offset: 0,
            token_count: 4,
        });
        preplay.decodes.push(linkw::LaneDecodeMsg {
            lane_tag: 98,
            max_new_tokens: 3,
            overshoot: 0,
        });
        match client.tick(preplay).unwrap_err() {
            superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -102),
            other => panic!("expected a bounds rejection for a prefill into the tail, got {other:?}"),
        }
    }

    let mut bad = plan(2);
    bad.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 7,
        max_new_tokens: 1,
        overshoot: 0,
});
    match client.tick(bad).unwrap_err() {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -106),
        other => panic!("expected typed rejection, got {other:?}"),
    }

    let mut p3 = plan(3);
    p3.retires.push(linkw::LaneRetireMsg {
        lane_tag: 7,
        publish_to_cache: true,
    });
    p3.retires.push(linkw::LaneRetireMsg {
        lane_tag: 11,
        publish_to_cache: false,
    });
    client.tick(p3).unwrap();

    let full: Vec<u32> = prompt.iter().copied().chain(generated.clone()).collect();
    let span = client.stage_prompt(&full).unwrap();
    let m = client.match_prefix(ALL_SPACES, span).unwrap();
    let kv_space = m.spaces.iter().find(|s| s.space_id == KV).unwrap();
    let cand = kv_space
        .candidates
        .iter()
        .max_by_key(|c| c.prefix_len)
        .expect("published prefix must match");
    assert!(cand.prefix_len >= page, "at least one cached block");
    assert_eq!(cand.prefix_len % page, 0, "matches are block-aligned");
    assert!(cand.prefix_len < full.len() as u64);

    let seed = client.seed_acquire(span, cand.prefix_len, 0).unwrap();
    let mut p4 = plan(4);
    let mut seeded = admit(8, span);
    seeded.seed_handle = seed;
    p4.admits.push(seeded);
    p4.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 8,
        token_offset: cand.prefix_len as u32,
        token_count: (full.len() as u64 - cand.prefix_len) as u32,
    });
    p4.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 8,
        max_new_tokens: 2,
        overshoot: 0,
});
    let ev = client.tick(p4).unwrap();
    assert_eq!(ev.admit_results[0].status, 1);
    let emit = ev.emits.iter().find(|e| e.lane_tag == 8).unwrap();
    assert!(emit.n_tokens >= 1);

    let span2 = client.stage_prompt(&full).unwrap();
    let mut p5 = plan(5);
    let mut reseed = admit(12, span2);
    reseed.seed_handle = seed;
    p5.admits.push(reseed);
    match client.tick(p5).unwrap_err() {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -108),
        other => panic!("expected stale-seed rejection, got {other:?}"),
    }

    let host_prompt: Vec<u32> = (0..8).map(|i| 2000 + i * 3).collect();
    let hp = client.stage_prompt(&host_prompt).unwrap();
    let mut p6 = plan(6);
    let mut host = admit(9, hp);
    host.sampling = 2;
    p6.admits.push(host);
    p6.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 9,
        token_offset: 0,
        token_count: 8,
    });
    p6.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 9,
        max_new_tokens: 1,
        overshoot: 0,
});
    let ev = client.tick(p6).unwrap();
    let emit = ev.emits.iter().find(|e| e.lane_tag == 9).unwrap();
    assert_eq!(emit.n_tokens, 0, "HOST decode emits a row and stops");
    let row = client.read_logits(&emit.logits_row).unwrap();
    assert_eq!(row.len(), vocab, "the full real logits row crossed shm");
    assert!(row.iter().all(|v| v.is_finite()));
    let choice = argmax(&row);
    let nonce = emit.logits_row.generation;

    let mut p7 = plan(7);
    p7.commits.push(linkw::LaneCommitMsg {
        lane_tag: 9,
        token_id: choice,
        logits_nonce: nonce,
    });
    p7.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 9,
        max_new_tokens: 1,
        overshoot: 0,
});
    let ev = client.tick(p7).unwrap();
    let emit2 = ev.emits.iter().find(|e| e.lane_tag == 9).unwrap();
    assert_ne!(
        emit2.logits_row.generation, nonce,
        "commit + decode produced a fresh row"
    );

    let mut p8 = plan(8);
    p8.commits.push(linkw::LaneCommitMsg {
        lane_tag: 9,
        token_id: 0,
        logits_nonce: nonce,
    });
    p8.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 9,
        max_new_tokens: 1,
        overshoot: 0,
});
    match client.tick(p8).unwrap_err() {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -112),
        other => panic!("expected stale-nonce rejection, got {other:?}"),
    }

    let expect_export = match backend {
        Backend::Ffi => -127,
        Backend::Native => -129,
    };
    match client
        .state_sync(linkw::StateSyncReqMsg::ExportSize {
            sequence: 3,
            space_id: KV,
            range: linkw::TokenRangeMsg { start: 0, end: 32 },
            encoding: 0,
        })
        .unwrap_err()
    {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, expect_export),
        other => panic!("expected typed refusal, got {other:?}"),
    }
    match client
        .state_sync(linkw::StateSyncReqMsg::Fork {
            parent_sequence: 3,
            flags: 0,
        })
        .unwrap_err()
    {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, expect_export),
        other => panic!("expected typed refusal, got {other:?}"),
    }
    match client
        .register_strategy(linkw::StrategyRegisterMsg {
            strategy_id: "prompt-lookup".into(),
            impl_version: "1.0.0".into(),
            impl_hash: [1; 32],
            config_hash: [2; 32],
            artifacts: vec![],
            taps: vec![],
            capabilities: vec![linkw::CapabilityReqMsg {
                kind_id: 1,
                params: vec![],
            }],
            target_archs: vec!["qwen3".into()],
            kernel_caps_required: 0,
            est_state_bytes: 1 << 16,
            claimed_exactness: 2,
            rng_contract_version: 1,
        })
        .unwrap_err()
    {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -128),
        other => panic!("expected refusal, got {other:?}"),
    }

    let evict = client
        .state_sync(linkw::StateSyncReqMsg::CacheEvict {
            bytes_target: 1 << 20,
            policy: linkw::ShedPolicyMsg {
                victim_lanes: vec![],
                evictable_cache_classes: u64::MAX,
                protected_quota_bytes: 0,
                max_evict_bytes: u64::MAX,
            },
        })
        .unwrap();
    assert!(matches!(evict, linkw::StateSyncOkMsg::CacheEvict { .. }));

    let pong = client.ping().unwrap();
    assert!(pong.mem.allocated_bytes > 0, "real model memory reported");
    client.drain().unwrap();

    drop(client);
    worker.join().expect("worker thread");
}

#[test]
fn full_lifecycle_over_link_w_ffi_engine() {
    run_full_lifecycle(Backend::Ffi);
}

#[test]
fn full_lifecycle_over_link_w_native_engine() {
    run_full_lifecycle(Backend::Native);
}

#[test]
fn cross_backend_greedy_determinism() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let greedy = |backend: Backend| -> (Vec<u32>, u64) {
        let (mut client, worker) = spawn_worker(path.clone(), backend);
        let version_tag = client
            .hello
            .state_space_descriptors
            .iter()
            .find(|d| d.space_id == 1)
            .expect("kv desc")
            .version_tag;
        let prompt: Vec<u32> = (0..48).map(|i| 3000 + i * 5).collect();
        let pref = client.stage_prompt(&prompt).unwrap();
        let mut p = plan(1);
        p.admits.push(admit(1, pref));
        p.prefills.push(linkw::LanePrefillMsg {
            lane_tag: 1,
            token_offset: 0,
            token_count: 48,
        });
        p.decodes.push(linkw::LaneDecodeMsg {
            lane_tag: 1,
            max_new_tokens: 6,
            overshoot: 0,
    });
        let ev = client.tick(p).unwrap();
        let emit = ev.emits.iter().find(|e| e.lane_tag == 1).unwrap();
        let tokens = client.read_tokens(&emit.token_ref).unwrap();
        drop(client);
        worker.join().expect("worker thread");
        (tokens, version_tag)
    };
    let (ffi, ffi_tag) = greedy(Backend::Ffi);
    let (native, native_tag) = greedy(Backend::Native);
    assert_eq!(ffi, native, "the two tick executors diverged on greedy decode");
    assert_eq!(
        ffi_tag, native_tag,
        "state identity must be canonical across back ends"
    );
}

#[test]
fn native_structural_plan_rejections() {
    use superfluid_abi::{Array, LaneDecode, TickPlan, RecordArena, Status};
    use superfluid_engine::InMemoryRings;

    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let mut engine = NativeEngine::load(engine_config(path)).expect("model load");
    let mut rings = InMemoryRings::new();
    let arena = RecordArena::new();

    let base_plan = || TickPlan {
        struct_size: std::mem::size_of::<TickPlan>() as u64,
        plan_seq: 1,
        flags: 0,
        _pad0: 0,
        prefill_token_budget: 4096,
        max_decode_lanes: 8,
        admits: Array::default(),
        commits: Array::default(),
        prefills: Array::default(),
        decodes: Array::default(),
        retires: Array::default(),
        shed_policy: superfluid_abi::ShedPolicy {
            victim_lanes: Array::default(),
            evictable_cache_classes: u64::MAX,
            protected_quota_bytes: 0,
            max_evict_bytes: u64::MAX,
        },
        on_partial_emit: None,
        partial_emit_user: std::ptr::null_mut(),
    };

    let mut p = base_plan();
    p.struct_size = 8;
    assert_eq!(
        engine.tick(&p, &arena, &mut rings).err(),
        Some(Status::RejectBadStruct)
    );

    let decodes = [LaneDecode {
        lane_tag: 1,
        max_new_tokens: 1,
        overshoot: 0,
        _pad1: 0,
    }];
    let mut p = base_plan();
    p.decodes = Array {
        data: decodes.as_ptr() as *const _,
        count: 1,
        elem_size: 8,
    };
    assert_eq!(
        engine.tick(&p, &arena, &mut rings).err(),
        Some(Status::RejectBadStride)
    );

    let raw = [0u8; 64];
    let mut p = base_plan();
    p.decodes = Array {
        // SAFETY: offset 4 stays inside the 64-byte buffer; the pointer
        // is only handed to the engine, whose alignment check rejects it.
        data: unsafe { raw.as_ptr().add(4) } as *const _,
        count: 1,
        elem_size: std::mem::size_of::<LaneDecode>() as u32,
    };
    assert_eq!(
        engine.tick(&p, &arena, &mut rings).err(),
        Some(Status::RejectBadAlignment)
    );

    let dup = [
        LaneDecode {
            lane_tag: 9,
            max_new_tokens: 1,
            overshoot: 0,
            _pad1: 0,
        },
        LaneDecode {
            lane_tag: 9,
            max_new_tokens: 1,
            overshoot: 0,
            _pad1: 0,
        },
    ];
    let mut p = base_plan();
    p.decodes = Array {
        data: dup.as_ptr() as *const _,
        count: 2,
        elem_size: std::mem::size_of::<LaneDecode>() as u32,
    };
    assert_eq!(
        engine.tick(&p, &arena, &mut rings).err(),
        Some(Status::RejectDuplicateLane)
    );

    let one = [LaneDecode {
        lane_tag: 1,
        max_new_tokens: 1,
        overshoot: 0,
        _pad1: 0,
    }];
    let mut p = base_plan();
    p.decodes = Array {
        data: one.as_ptr() as *const _,
        count: 1,
        elem_size: std::mem::size_of::<LaneDecode>() as u32,
    };
    assert_eq!(
        engine.tick(&p, &arena, &mut rings).err(),
        Some(Status::RejectBounds)
    );
}

#[test]
fn native_kv_snapshot_restore_round_trip() {
    use superfluid_abi::{
        checksum_kind, encoding, op_state, space_kind, Array, LaneAdmit,
        LaneDecode, LanePrefill, ShedPolicy, StateEnvelope,
        TickPlan, TokenRange, TokenRef, RecordArena, Status,
    };
    use superfluid_engine::rings::RingAttachment;
    use superfluid_engine::InMemoryRings;
    use superfluid_shm::{RingRole, SharedRing};

    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let mut engine = NativeEngine::load(engine_config(path)).expect("model load");
    let vocab = engine.vocab();

    let mut token_in = SharedRing::create(TOKEN_RING_IN, 64, 16 + 4096 * 4, RingRole::Writer)
        .expect("token in");
    let token_out =
        SharedRing::create(TOKEN_RING_OUT, 64, 16 + 1024 * 4, RingRole::Reader).expect("token out");
    let logits =
        SharedRing::create(LOGITS_RING, 8, 16 + vocab * 4, RingRole::Reader).expect("logits");
    for (ring, kind, role) in [
        (&token_in, superfluid_abi::ring_kind::TOKENS, superfluid_abi::ring_role::ENGINE_READS),
        (&token_out, superfluid_abi::ring_kind::TOKENS, superfluid_abi::ring_role::ENGINE_WRITES),
        (&logits, superfluid_abi::ring_kind::LOGITS, superfluid_abi::ring_role::ENGINE_WRITES),
    ] {
        engine
            .attach_ring(&RingAttachment {
                ring_id: match kind {
                    superfluid_abi::ring_kind::LOGITS => LOGITS_RING,
                    _ if role == superfluid_abi::ring_role::ENGINE_READS => TOKEN_RING_IN,
                    _ => TOKEN_RING_OUT,
                },
                kind,
                role,
                slots: if kind == superfluid_abi::ring_kind::LOGITS { 8 } else { 64 },
                slot_bytes: if kind == superfluid_abi::ring_kind::LOGITS {
                    16 + vocab * 4
                } else if role == superfluid_abi::ring_role::ENGINE_READS {
                    16 + 4096 * 4
                } else {
                    16 + 1024 * 4
                },
                base: ring.base_ptr(),
                len: ring.seg_len(),
            })
            .expect("attach");
    }

    let prompt: Vec<u32> = (0..48).map(|i| 5000 + i * 3).collect();
    let bytes: Vec<u8> = prompt.iter().flat_map(|t| t.to_le_bytes()).collect();
    let (idx, gen) = token_in.push(&bytes).expect("stage");
    let span = TokenRef {
        ring_id: TOKEN_RING_IN,
        index: idx,
        count: 48,
        _pad0: 0,
        generation: gen,
    };
    let mut arena = RecordArena::new();
    let admits = [LaneAdmit {
        lane_tag: 1,
        prompt: span,
        seed_handle: 0,
        sampling: 0,
        logit_bias_handle: 0,
        params: Default::default(),
        rng_counter_base: 0,
        grammar_handle: 0,
        strategy_slot: 0,
        determinism_class: 0,
        minimum_exactness: 0,
        allow_approximate: 0,
        want_logprobs: 0,
        required_cert_id: 0,
        host_sampler_identity: [0; 32],
        grammar_replay: 0,
        decode_replay: 0,
    }];
    let prefills = [LanePrefill {
        lane_tag: 1,
        token_offset: 0,
        token_count: 48,
    }];
    let decodes = [LaneDecode {
        lane_tag: 1,
        max_new_tokens: 4,
        overshoot: 0,
        _pad1: 0,
    }];
    let plan = |seq: u64, a: &mut RecordArena, with_work: bool| TickPlan {
        struct_size: std::mem::size_of::<TickPlan>() as u64,
        plan_seq: seq,
        flags: 0,
        _pad0: 0,
        prefill_token_budget: 4096,
        max_decode_lanes: 8,
        admits: if with_work { a.push_records(&admits) } else { Array::default() },
        commits: Array::default(),
        prefills: if with_work { a.push_records(&prefills) } else { Array::default() },
        decodes: if with_work { a.push_records(&decodes) } else { Array::default() },
        retires: Array::default(),
        shed_policy: ShedPolicy {
            victim_lanes: Array::default(),
            evictable_cache_classes: u64::MAX,
            protected_quota_bytes: 0,
            max_evict_bytes: u64::MAX,
        },
        on_partial_emit: None,
        partial_emit_user: std::ptr::null_mut(),
    };
    let mut dummy = InMemoryRings::new();
    let p1 = plan(1, &mut arena, true);
    engine.tick(&p1, &arena, &mut dummy).expect("tick");

    let seq = engine.lane_sequence(1);
    assert_ne!(seq, 0, "lane owns a sequence");
    let kv = 1u32;
    let range = TokenRange { start: 0, end: 48 };

    let (size, sgen) = engine
        .space_export_size(seq, kv, range, encoding::LOSSLESS)
        .expect("export size");
    assert!(size > std::mem::size_of::<StateEnvelope>() as u64);

    let op = engine
        .space_snapshot(seq, kv, 48, size, sgen)
        .expect("snapshot");
    assert_eq!(engine.op_poll(op).unwrap().state, op_state::DONE);
    let sealed = engine.take_op_output(op).expect("staged bytes");
    assert_eq!(sealed.len() as u64, size);
    let env_bytes = &sealed[..std::mem::size_of::<StateEnvelope>()];
    // SAFETY: env_bytes holds exactly sizeof(StateEnvelope) bytes
    // written by the engine; read_unaligned tolerates the Vec offset.
    let env: StateEnvelope =
        unsafe { std::ptr::read_unaligned(env_bytes.as_ptr() as *const _) };
    assert_eq!(env.magic, superfluid_abi::STATE_ENVELOPE_MAGIC);
    assert_eq!(env.space_kind, space_kind::PAGED_TOKEN_KV);
    assert_eq!(env.checksum_kind, checksum_kind::FNV1A64);
    assert_eq!(env.range, TokenRange { start: 0, end: 48 });

    let mut arena2 = RecordArena::new();
    let decodes2 = [LaneDecode {
        lane_tag: 1,
        max_new_tokens: 1,
        overshoot: 0,
        _pad1: 0,
    }];
    let mut p2 = plan(2, &mut arena2, false);
    p2.decodes = arena2.push_records(&decodes2);
    engine.tick(&p2, &arena2, &mut dummy).expect("tick 2");
    assert_eq!(
        engine.space_snapshot(seq, kv, 48, size, sgen).err(),
        Some(Status::StaleSizing),
        "old sizing_gen must refuse synchronously"
    );

    let target = engine.create_sequence();
    assert_ne!(target, 0);
    let op = engine.space_restore(target, kv, &sealed).expect("restore");
    assert_eq!(engine.op_poll(op).unwrap().state, op_state::DONE);
    let (size2, sgen2) = engine
        .space_export_size(target, kv, range, encoding::LOSSLESS)
        .expect("export size 2");
    assert_eq!(size2, size);
    let op = engine
        .space_snapshot(target, kv, 48, size2, sgen2)
        .expect("snapshot 2");
    let sealed2 = engine.take_op_output(op).expect("staged 2");
    let env_len = std::mem::size_of::<StateEnvelope>();
    assert_eq!(
        sealed[env_len..],
        sealed2[env_len..],
        "KV payload must round-trip byte exactly"
    );

    let target2 = engine.create_sequence();
    let mut corrupt = sealed.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0xFF;
    assert_eq!(
        engine.space_restore(target2, kv, &corrupt).err(),
        Some(Status::Checksum)
    );
    assert_eq!(
        engine.space_snapshot(target, kv, 48, size - 1, sgen2).err(),
        Some(Status::BufferTooSmall)
    );
    engine.space_trim(seq, kv, 32).expect("trim");
    assert_eq!(
        engine.space_trim(seq, kv, 10_000).err(),
        Some(Status::OutOfBoundary)
    );
}

#[test]
fn hybrid_bundle_advertises_blob_space_and_serves_boundary_seeds() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = hybrid_path() else {
        eprintln!("SKIP: no hybrid GDN bundle (set BASERT_TEST_HYBRID)");
        return;
    };
    let probe = |backend: Backend| {
        let rig = backend == Backend::Ffi;
        let (mut client, worker) = spawn_worker(path.clone(), backend);
        let descs = client.hello.state_space_descriptors.clone();
        assert!(descs.len() >= 2, "KV + GDN blob advertised");
        let kv = descs.iter().find(|d| d.space_id == KV).expect("kv desc");
        let gdn = descs.iter().find(|d| d.space_id == 2).expect("gdn desc");
        if !rig {
            assert!(
                descs.iter().any(|d| d.kind == superfluid_abi::space_kind::ENCODER_CACHE),
                "hybrid+vision bundle advertises the encoder cache"
            );
        }
        assert_eq!(kv.kind, superfluid_abi::space_kind::PAGED_TOKEN_KV);
        assert_eq!(gdn.kind, superfluid_abi::space_kind::RECURRENT_BLOB);
        assert!(gdn.blob_bytes > 0, "the blob size is the engine's real snapshot size");
        assert_eq!(
            gdn.flags & superfluid_abi::space_flag::OPS_UNAVAILABLE != 0,
            rig,
            "ops-unavailable only on the rig"
        );
        assert_eq!(
            gdn.flags & superfluid_abi::space_flag::NO_PREFIX_CACHE != 0,
            rig,
            "the boundary-snapshot cache serves the native blob; the rig's cannot"
        );
        assert_eq!(gdn.snapshot_cadence, 1, "tool-boundary cadence (§7.3)");
        assert_eq!(kv.flags, 0);
        let page = kv.page_size_tokens as u64;

        let prompt: Vec<u32> = (0..48).map(|i| 3000 + i * 5).collect();
        let pref = client.stage_prompt(&prompt).unwrap();
        let mut p = plan(1);
        p.admits.push(admit(1, pref));
        p.prefills.push(linkw::LanePrefillMsg {
            lane_tag: 1,
            token_offset: 0,
            token_count: 48,
        });
        client.tick(p).unwrap();
        let mut p2 = plan(2);
        p2.retires.push(linkw::LaneRetireMsg {
            lane_tag: 1,
            publish_to_cache: true,
        });
        client.tick(p2).unwrap();
        let mut longer = prompt.clone();
        longer.push(9);
        let span = client.stage_prompt(&longer).unwrap();
        let boundary = (47 / page) * page;
        assert!(boundary >= page);
        if rig {
            match client.seed_acquire(span, boundary, 0).unwrap_err() {
                superfluid_agent::AgentError::Rejected(status) => {
                    assert_eq!(status, superfluid_abi::Status::SeedUnservable as i32)
                }
                other => panic!("expected typed SEED_UNSERVABLE, got {other:?}"),
            }
        } else {
            let handle = client
                .seed_acquire(span, boundary, 0)
                .expect("native hybrid seed at the attached boundary");
            client.seed_release(handle).unwrap();
            match client.seed_acquire(span, boundary + page, 0).unwrap_err() {
                superfluid_agent::AgentError::Rejected(status) => {
                    assert_eq!(status, superfluid_abi::Status::SeedUnservable as i32)
                }
                other => panic!("expected typed SEED_UNSERVABLE, got {other:?}"),
            }
        }
        drop(client);
        worker.join().expect("worker thread");
    };
    probe(Backend::Native);
    probe(Backend::Ffi);
}

#[test]
fn fork_promote_and_keyed_eviction_are_real() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let (mut client, worker) = spawn_worker(path, Backend::Native);
    let descs = client.hello.state_space_descriptors.clone();
    let kv = descs.iter().find(|d| d.space_id == KV).expect("kv desc");
    let page = kv.page_size_tokens as u64;

    let prompt: Vec<u32> = (0..48).map(|i| 3000 + i * 5).collect();
    let pref = client.stage_prompt(&prompt).unwrap();
    let mut p = plan(1);
    p.admits.push(admit(1, pref));
    p.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 1,
        token_offset: 0,
        token_count: 48,
    });
    client.tick(p).unwrap();
    let seq = match client
        .state_sync(linkw::StateSyncReqMsg::LaneSequence { lane_tag: 1 })
        .unwrap()
    {
        linkw::StateSyncOkMsg::LaneSequence { sequence } => sequence,
        other => panic!("unexpected reply {other:?}"),
    };
    let b = (47 / page) * page;
    assert!(b >= page);
    let range = linkw::TokenRangeMsg { start: 0, end: b };

    let export_kv = |client: &mut WorkerClient, sequence: u64| -> Vec<u8> {
        let (bytes, sizing_gen) = match client
            .state_sync(linkw::StateSyncReqMsg::ExportSize {
                sequence,
                space_id: KV,
                range,
                encoding: superfluid_abi::encoding::LOSSLESS,
            })
            .unwrap()
        {
            linkw::StateSyncOkMsg::ExportSize { required_bytes, sizing_gen } => (required_bytes, sizing_gen),
            other => panic!("unexpected reply {other:?}"),
        };
        let buf = client.buf_create(bytes).unwrap();
        let op = client
            .state_op(linkw::StateOpKind::Snapshot {
                sequence,
                space_id: KV,
                boundary_pos: b,
                buf_id: buf,
                sizing_gen,
            })
            .unwrap();
        let done = client.wait_op_done(op).unwrap();
        assert_eq!(done.state, superfluid_abi::op_state::DONE);
        let sealed = client.buf_read(buf, bytes as usize).unwrap();
        client.buf_release(buf).unwrap();
        sealed
    };

    let parent_kv = export_kv(&mut client, seq);
    for flags in [0u32, superfluid_abi::fork_flags::EAGER] {
        let child = match client
            .state_sync(linkw::StateSyncReqMsg::Fork { parent_sequence: seq, flags })
            .unwrap()
        {
            linkw::StateSyncOkMsg::Fork { child_sequence } => child_sequence,
            other => panic!("unexpected reply {other:?}"),
        };
        let child_kv = export_kv(&mut client, child);
        let hdr = std::mem::size_of::<superfluid_abi::StateEnvelope>();
        assert_eq!(&child_kv[hdr..], &parent_kv[hdr..], "fork (flags={flags}) KV differs from parent");
        match client.state_sync(linkw::StateSyncReqMsg::FreeSequence { sequence: child }).unwrap() {
            linkw::StateSyncOkMsg::FreeSequence => {}
            other => panic!("unexpected reply {other:?}"),
        }
        let again = export_kv(&mut client, seq);
        assert_eq!(again[hdr..], parent_kv[hdr..], "parent KV changed after freeing a fork (flags={flags})");
    }

    let (bytes, sizing_gen) = match client
        .state_sync(linkw::StateSyncReqMsg::ExportSize {
            sequence: seq,
            space_id: KV,
            range,
            encoding: superfluid_abi::encoding::LOSSLESS,
        })
        .unwrap()
    {
        linkw::StateSyncOkMsg::ExportSize { required_bytes, sizing_gen } => (required_bytes, sizing_gen),
        other => panic!("unexpected reply {other:?}"),
    };
    let buf = client.buf_create(bytes).unwrap();
    let op = client
        .state_op(linkw::StateOpKind::Demote {
            sequence: seq,
            space_id: KV,
            range,
            encoding: superfluid_abi::encoding::LOSSLESS,
            buf_id: buf,
            sizing_gen,
        })
        .unwrap();
    let done = client.wait_op_done(op).unwrap();
    assert_eq!(done.state, superfluid_abi::op_state::DONE);
    let demoted = client.buf_read(buf, bytes as usize).unwrap();
    client.buf_release(buf).unwrap();
    match client.state_sync(linkw::StateSyncReqMsg::ExportSize {
        sequence: seq,
        space_id: KV,
        range,
        encoding: superfluid_abi::encoding::LOSSLESS,
    }) {
        Err(_) => {}
        Ok(other) => panic!("export of a demoted sequence must refuse, got {other:?}"),
    }
    let pbuf = client.buf_create(demoted.len() as u64).unwrap();
    client.buf_write(pbuf, &demoted).unwrap();
    let op = client
        .state_op(linkw::StateOpKind::Promote {
            sequence: seq,
            space_id: KV,
            range,
            buf_id: pbuf,
        })
        .unwrap();
    let done = client.wait_op_done(op).unwrap();
    assert_eq!(done.state, superfluid_abi::op_state::DONE, "promote error {}", done.error);
    client.buf_release(pbuf).unwrap();
    let hdr = std::mem::size_of::<superfluid_abi::StateEnvelope>();
    let promoted_kv = export_kv(&mut client, seq);
    assert_eq!(promoted_kv[hdr..], parent_kv[hdr..], "lossless demote→promote must round-trip bit-exact");

    let mut p2 = plan(2);
    p2.retires.push(linkw::LaneRetireMsg {
        lane_tag: 1,
        publish_to_cache: true,
    });
    client.tick(p2).unwrap();
    let mut longer = prompt.clone();
    longer.push(9);
    let span = client.stage_prompt(&longer).unwrap();
    let (handle, ttl) = client.seed_acquire_leased(span, b, 0).expect("published prefix seeds");
    assert_eq!(ttl, Some(64), "SeedGrant.ttl_ticks from baseRT_seed_lease_ticks");
    client.seed_release(handle).unwrap();
    let bad = linkw::CacheKeyMsg {
        compat_key: [0xEE; 32],
        provenance_digest: [0; 32],
    };
    match client.state_sync(linkw::StateSyncReqMsg::CacheEvictEntries { keys: vec![bad] }) {
        Err(superfluid_agent::AgentError::Rejected(status)) => {
            assert_eq!(status, superfluid_abi::Status::IdentityMismatch as i32)
        }
        other => panic!("expected typed IDENTITY_MISMATCH, got {other:?}"),
    }
    let env: superfluid_abi::StateEnvelope =
        // SAFETY: parent_kv holds a full envelope written by the engine.
        unsafe { std::ptr::read_unaligned(parent_kv.as_ptr() as *const _) };
    let mut compat = [0u8; 32];
    compat[..8].copy_from_slice(&env.version_tag.to_le_bytes());
    let keys: Vec<linkw::CacheKeyMsg> = [47usize, b as usize]
        .iter()
        .map(|&n| linkw::CacheKeyMsg {
            compat_key: compat,
            provenance_digest: superfluid_fingerprint::content_digest(&prompt[..n]),
        })
        .collect();
    match client.state_sync(linkw::StateSyncReqMsg::CacheEvictEntries { keys }).unwrap() {
        linkw::StateSyncOkMsg::CacheEvictEntries => {}
        other => panic!("unexpected reply {other:?}"),
    }
    let span2 = client.stage_prompt(&longer).unwrap();
    match client.seed_acquire(span2, b, 0) {
        Err(_) => {}
        Ok(h) => {
            client.seed_release(h).unwrap();
            panic!("evicted prefix must not seed");
        }
    }
    let keys2 = vec![linkw::CacheKeyMsg {
        compat_key: compat,
        provenance_digest: superfluid_fingerprint::content_digest(&prompt[..b as usize]),
    }];
    match client.state_sync(linkw::StateSyncReqMsg::CacheEvictEntries { keys: keys2 }).unwrap() {
        linkw::StateSyncOkMsg::CacheEvictEntries => {}
        other => panic!("unexpected reply {other:?}"),
    }

    drop(client);
    worker.join().expect("worker thread");
}

#[test]
fn hybrid_snapshot_boundary_serves_the_durable_cut() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = hybrid_path() else {
        eprintln!("SKIP: no hybrid GDN bundle (set BASERT_TEST_HYBRID)");
        return;
    };
    let (mut client, worker) = spawn_worker(path, Backend::Native);
    let descs = client.hello.state_space_descriptors.clone();
    let kv = descs.iter().find(|d| d.space_id == KV).expect("kv desc");
    let page = kv.page_size_tokens as u64;
    let gdn_id = 2u32;

    let prompt: Vec<u32> = (0..48).map(|i| 3000 + i * 5).collect();
    let pref = client.stage_prompt(&prompt).unwrap();
    let mut p = plan(1);
    p.admits.push(admit(1, pref));
    p.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 1,
        token_offset: 0,
        token_count: 48,
    });
    client.tick(p).unwrap();

    let seq = match client
        .state_sync(linkw::StateSyncReqMsg::LaneSequence { lane_tag: 1 })
        .unwrap()
    {
        linkw::StateSyncOkMsg::LaneSequence { sequence } => sequence,
        other => panic!("unexpected reply {other:?}"),
    };
    let expect = (47 / page) * page;
    assert!(expect >= page, "prompt spans at least one page");
    let boundary = match client
        .state_sync(linkw::StateSyncReqMsg::SnapshotBoundary {
            sequence: seq,
            space_id: gdn_id,
            cap: 47,
        })
        .unwrap()
    {
        linkw::StateSyncOkMsg::SnapshotBoundary { boundary } => boundary,
        other => panic!("unexpected reply {other:?}"),
    };
    assert_eq!(boundary, expect, "GDN durable boundary = last whole-page prompt position");
    match client
        .state_sync(linkw::StateSyncReqMsg::SnapshotBoundary {
            sequence: seq,
            space_id: gdn_id,
            cap: boundary - 1,
        })
        .unwrap()
    {
        linkw::StateSyncOkMsg::SnapshotBoundary { boundary: b } => assert_eq!(b, 0),
        other => panic!("unexpected reply {other:?}"),
    }
    match client
        .state_sync(linkw::StateSyncReqMsg::SnapshotBoundary {
            sequence: seq,
            space_id: KV,
            cap: 47,
        })
        .unwrap()
    {
        linkw::StateSyncOkMsg::SnapshotBoundary { boundary: b } => assert_eq!(b, (47 / page) * page),
        other => panic!("unexpected reply {other:?}"),
    }

    let (bytes, sizing_gen) = match client
        .state_sync(linkw::StateSyncReqMsg::ExportSize {
            sequence: seq,
            space_id: gdn_id,
            range: linkw::TokenRangeMsg { start: boundary, end: boundary },
            encoding: superfluid_abi::encoding::LOSSLESS,
        })
        .unwrap()
    {
        linkw::StateSyncOkMsg::ExportSize { required_bytes, sizing_gen } => (required_bytes, sizing_gen),
        other => panic!("unexpected reply {other:?}"),
    };
    let buf = client.buf_create(bytes).unwrap();
    let op = client
        .state_op(linkw::StateOpKind::Snapshot {
            sequence: seq,
            space_id: gdn_id,
            boundary_pos: boundary,
            buf_id: buf,
            sizing_gen,
        })
        .unwrap();
    let done = client.wait_op_done(op).unwrap();
    assert_eq!(done.state, superfluid_abi::op_state::DONE);
    assert_eq!(done.error, 0);
    let sealed = client.buf_read(buf, bytes as usize).unwrap();
    client.buf_release(buf).unwrap();
    let env_bytes = &sealed[..std::mem::size_of::<superfluid_abi::StateEnvelope>()];
    // SAFETY: env_bytes holds exactly sizeof(StateEnvelope) bytes
    // written by the engine; read_unaligned tolerates the Vec offset.
    let env: superfluid_abi::StateEnvelope =
        unsafe { std::ptr::read_unaligned(env_bytes.as_ptr() as *const _) };
    assert_eq!(env.magic, superfluid_abi::STATE_ENVELOPE_MAGIC);
    assert_eq!(env.space_kind, superfluid_abi::space_kind::RECURRENT_BLOB);
    assert_eq!(env.range.start, boundary, "envelope records the durable cut");
    assert_eq!(env.range.end, boundary);

    match client
        .state_sync(linkw::StateSyncReqMsg::ExportSize {
            sequence: seq,
            space_id: gdn_id,
            range: linkw::TokenRangeMsg { start: boundary - 1, end: boundary - 1 },
            encoding: superfluid_abi::encoding::LOSSLESS,
        })
        .unwrap_err()
    {
        superfluid_agent::AgentError::Rejected(status) => {
            assert_eq!(status, superfluid_abi::Status::OutOfBoundary as i32)
        }
        other => panic!("expected typed OUT_OF_BOUNDARY, got {other:?}"),
    }

    drop(client);
    worker.join().expect("worker thread");
}
