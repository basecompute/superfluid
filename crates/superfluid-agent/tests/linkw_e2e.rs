use std::os::unix::net::UnixStream;
use std::thread;

use superfluid_agent::{argmax, WorkerClient};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkw::{self, ALL_SPACES};
use superfluid_worker::{WorkerConfig, WorkerServer};

const KV: u32 = 1;
const GDN: u32 = 2;

fn spawn_worker() -> (WorkerClient, thread::JoinHandle<()>) {
    let (frames_a, frames_w) = UnixStream::pair().expect("socketpair");
    let (fds_a, fds_w) = UnixStream::pair().expect("fd socketpair");
    let handle = thread::spawn(move || {
        let engine = MockEngine::new(EngineConfig::default());
        let server = WorkerServer::new(
            engine,
            WorkerConfig {
                engine_bundle_hash: [0xBA; 32],
            },
            frames_w,
            fds_w,
        );
        server.serve().expect("worker serve");
    });
    let client = WorkerClient::connect(frames_a, fds_a).expect("connect");
    (client, handle)
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

#[test]
fn full_lifecycle_over_link_w() {
    let (mut client, worker) = spawn_worker();

    assert_eq!(client.hello.chosen_version, superfluid_agent::client::PROTO_VERSION);
    assert_eq!(client.hello.engine_bundle_hash, [0xBA; 32]);
    let kinds: Vec<u32> = client
        .hello
        .state_space_descriptors
        .iter()
        .map(|d| d.kind)
        .collect();
    assert!(kinds.contains(&1) && kinds.contains(&3));

    let prompt: Vec<u32> = (100..164).collect();
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
    assert_eq!(ev.tick_status, 0);
    let emit = &ev.emits[0];
    assert_eq!(emit.n_tokens, 4);
    let generated = client.read_tokens(&emit.token_ref).unwrap();
    assert_eq!(generated.len(), 4);

    let mut bad = plan(1);
    bad.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 7,
        max_new_tokens: 1,
        overshoot: 0,
});
    let err = client.tick(bad).unwrap_err();
    match err {
        superfluid_agent::AgentError::Rejected(status) => {
            assert_eq!(status, -106);
        }
        other => panic!("expected typed rejection, got {other:?}"),
    }

    let mut p2 = plan(2);
    p2.retires.push(linkw::LaneRetireMsg {
        lane_tag: 7,
        publish_to_cache: true,
    });
    client.tick(p2).unwrap();

    let full: Vec<u32> = prompt.iter().copied().chain(generated.clone()).collect();
    let span = client.stage_prompt(&full).unwrap();
    let m = client.match_prefix(ALL_SPACES, span).unwrap();
    let kv_space = m.spaces.iter().find(|s| s.space_id == KV).unwrap();
    assert!(kv_space.candidates.iter().any(|c| c.prefix_len >= 64));

    let seed = client.seed_acquire(span, 64, 0).unwrap();
    let mut p3 = plan(3);
    let mut seeded = admit(8, span);
    seeded.seed_handle = seed;
    p3.admits.push(seeded);
    p3.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 8,
        token_offset: 64,
        token_count: full.len() as u32 - 64,
    });
    p3.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 8,
        max_new_tokens: 2,
        overshoot: 0,
});
    let ev = client.tick(p3).unwrap();
    assert_eq!(ev.admit_results[0].status, 1);
    assert_eq!(ev.emits[0].n_tokens, 2);

    let host_prompt: Vec<u32> = (500..508).collect();
    let hp = client.stage_prompt(&host_prompt).unwrap();
    let mut p4 = plan(4);
    let mut host = admit(9, hp);
    host.sampling = 2;
    p4.admits.push(host);
    p4.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 9,
        token_offset: 0,
        token_count: 8,
    });
    p4.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 9,
        max_new_tokens: 1,
        overshoot: 0,
});
    let ev = client.tick(p4).unwrap();
    let emit = ev.emits.iter().find(|e| e.lane_tag == 9).unwrap();
    assert_eq!(emit.n_tokens, 0, "HOST decode emits a row and stops");
    let row = client.read_logits(&emit.logits_row).unwrap();
    assert_eq!(row.len(), 8);
    let choice = argmax(&row);
    let nonce = emit.logits_row.generation;

    let mut p5 = plan(5);
    p5.commits.push(linkw::LaneCommitMsg {
        lane_tag: 9,
        token_id: choice,
        logits_nonce: nonce,
    });
    p5.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 9,
        max_new_tokens: 1,
        overshoot: 0,
});
    let ev = client.tick(p5).unwrap();
    let emit2 = ev.emits.iter().find(|e| e.lane_tag == 9).unwrap();
    assert_ne!(
        emit2.logits_row.generation, nonce,
        "commit+decode produced a fresh row"
    );

    let mut p6 = plan(6);
    p6.commits.push(linkw::LaneCommitMsg {
        lane_tag: 9,
        token_id: 0,
        logits_nonce: nonce,
    });
    p6.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 9,
        max_new_tokens: 1,
        overshoot: 0,
});
    match client.tick(p6).unwrap_err() {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -112),
        other => panic!("expected stale-nonce rejection, got {other:?}"),
    }

    let sizing = client
        .state_sync(linkw::StateSyncReqMsg::ExportSize {
            sequence: 2,
            space_id: KV,
            range: linkw::TokenRangeMsg { start: 0, end: 32 },
            encoding: 0,
        })
        .unwrap();
    let (required, sizing_gen) = match sizing {
        linkw::StateSyncOkMsg::ExportSize {
            required_bytes,
            sizing_gen,
        } => (required_bytes, sizing_gen),
        other => panic!("wrong sync result {other:?}"),
    };

    let buf = client.buf_create(required).unwrap();
    let op = client
        .state_op(linkw::StateOpKind::Demote {
            sequence: 2,
            space_id: KV,
            range: linkw::TokenRangeMsg { start: 0, end: 32 },
            encoding: 0,
            buf_id: buf,
            sizing_gen,
        })
        .unwrap();
    let done = client.wait_op_done(op).unwrap();
    assert_eq!(done.state, 2);
    let sealed = client.buf_read(buf, required as usize).unwrap();
    assert_eq!(
        u64::from_le_bytes(sealed[0..8].try_into().unwrap()),
        0x4554415453545242
    );

    let err = client
        .state_op(linkw::StateOpKind::Demote {
            sequence: 2,
            space_id: KV,
            range: linkw::TokenRangeMsg { start: 0, end: 32 },
            encoding: 0,
            buf_id: buf,
            sizing_gen,
        })
        .unwrap_err();
    match err {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -122),
        other => panic!("expected stale-sizing rejection, got {other:?}"),
    }

    let op = client
        .state_op(linkw::StateOpKind::Promote {
            sequence: 2,
            space_id: KV,
            range: linkw::TokenRangeMsg { start: 0, end: 32 },
            buf_id: buf,
        })
        .unwrap();
    let done = client.wait_op_done(op).unwrap();
    assert_eq!(done.state, 2);
    client.buf_release(buf).unwrap();

    let sizing = client
        .state_sync(linkw::StateSyncReqMsg::ExportSize {
            sequence: 2,
            space_id: GDN,
            range: linkw::TokenRangeMsg { start: 0, end: 64 },
            encoding: 0,
        })
        .unwrap();
    let (required, sizing_gen) = match sizing {
        linkw::StateSyncOkMsg::ExportSize {
            required_bytes,
            sizing_gen,
        } => (required_bytes, sizing_gen),
        other => panic!("wrong sync result {other:?}"),
    };
    let buf = client.buf_create(required).unwrap();
    let op = client
        .state_op(linkw::StateOpKind::Snapshot {
            sequence: 2,
            space_id: GDN,
            boundary_pos: 64,
            buf_id: buf,
            sizing_gen,
        })
        .unwrap();
    assert_eq!(client.wait_op_done(op).unwrap().state, 2);

    let pong = client.ping().unwrap();
    assert!(pong.mem.allocated_bytes > 0);
    client.drain().unwrap();

    drop(client);
    worker.join().expect("worker thread");
}

#[test]
fn strategy_registration_over_link_w() {
    let (mut client, worker) = spawn_worker();

    let grant = client
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
            target_archs: vec!["mock-arch".into()],
            kernel_caps_required: 0,
            est_state_bytes: 1 << 16,
            claimed_exactness: 2,
            rng_contract_version: 1,
        })
        .unwrap();
    assert!(grant.strategy_slot > 0);
    assert_eq!(grant.certificates.len(), 1);
    assert_eq!(grant.certificates[0].exactness, 2);
    assert_eq!(grant.reserved_bytes, 1 << 16);

    let err = client
        .register_strategy(linkw::StrategyRegisterMsg {
            strategy_id: "future-method".into(),
            impl_version: "0.1.0".into(),
            impl_hash: [3; 32],
            config_hash: [4; 32],
            artifacts: vec![],
            taps: vec![],
            capabilities: vec![linkw::CapabilityReqMsg {
                kind_id: 0xDEAD,
                params: vec![],
            }],
            target_archs: vec![],
            kernel_caps_required: 0,
            est_state_bytes: 0,
            claimed_exactness: 0,
            rng_contract_version: 1,
        })
        .unwrap_err();
    match err {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -128),
        other => panic!("expected refusal, got {other:?}"),
    }

    let prompt: Vec<u32> = (0..4).collect();
    let pref = client.stage_prompt(&prompt).unwrap();
    let mut p = plan(1);
    let mut a = admit(1, pref);
    a.strategy_slot = grant.strategy_slot;
    a.minimum_exactness = 2;
    p.admits.push(a);
    p.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 1,
        token_offset: 0,
        token_count: 4,
    });
    p.decodes.push(linkw::LaneDecodeMsg {
        lane_tag: 1,
        max_new_tokens: 2,
        overshoot: 0,
});
    let ev = client.tick(p).unwrap();
    assert_eq!(ev.admit_results[0].status, 1);
    assert_eq!(ev.admit_results[0].granted_class, 2);
    assert_eq!(
        ev.emits[0].spec.cert_id_in_effect,
        ev.admit_results[0].cert_id
    );

    drop(client);
    worker.join().expect("worker thread");
}

#[test]
fn frame_version_mismatch_fails_closed() {
    use superfluid_proto::envelope::FrameClass;
    use superfluid_proto::linkw::msg_type;
    use std::io::Write;
    use std::os::unix::net::UnixStream;

    let (frames_a, frames_w) = UnixStream::pair().expect("socketpair");
    let (fds_a, fds_w) = UnixStream::pair().expect("fd socketpair");
    let worker = thread::spawn(move || {
        let engine = MockEngine::new(EngineConfig::default());
        let server = WorkerServer::new(
            engine,
            WorkerConfig {
                engine_bundle_hash: [0; 32],
            },
            frames_w,
            fds_w,
        );
        server.serve()
    });
    let mut client = superfluid_agent::WorkerClient::connect(frames_a, fds_a).expect("connect");
    assert_eq!(client.hello.chosen_version, superfluid_agent::client::PROTO_VERSION);

    let bytes = superfluid_proto::encode_frame(
        1,
        msg_type::PING,
        999,
        0,
        FrameClass::Required,
        &postcard::to_stdvec(&superfluid_proto::linkw::PingMsg {
            queue_depth: 0,
            mem: superfluid_proto::linkw::MemCountersMsg {
                allocated_bytes: 0,
                host_retained_bytes: 0,
                pool_blocks_total: 0,
                pool_blocks_used: 0,
                pool_bytes_evictable: 0,
                tentative_bytes: 0,
            },
        })
        .unwrap(),
    )
    .unwrap();
    client.raw_frames_stream().write_all(&bytes).unwrap();

    let err = worker.join().expect("join");
    assert!(err.is_err(), "worker must reject the mismatched frame");
    drop(client);
}

#[test]
fn oversized_envelope_payload_len_is_typed_refusal() {
    let (mut client, worker) = spawn_worker();
    let buf = client.buf_create(4096).unwrap();
    let mut sealed = vec![0u8; 4096];
    sealed[0..8].copy_from_slice(&0x4554415453545242u64.to_le_bytes());
    let off = std::mem::offset_of!(superfluid_abi::StateEnvelope, payload_len);
    sealed[off..off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    client.buf_write(buf, &sealed).unwrap();
    let err = client
        .state_op(superfluid_proto::linkw::StateOpKind::Restore {
            sequence: 1,
            space_id: 1,
            buf_id: buf,
        })
        .unwrap_err();
    match err {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -124),
        other => panic!("expected typed refusal, got {other:?}"),
    }
    drop(client);
    worker.join().expect("worker thread");
}

#[test]
fn streaming_transcribe_stops_when_the_listener_hangs_up() {
    let prompt = "one two three four five six seven eight nine ten";
    let total = prompt.split_whitespace().count() + 2;
    let req = |stream| linkw::StateSyncReqMsg::Transcribe {
        audio_path: "unused-by-the-mock.wav".into(),
        language: None,
        translate: false,
        timestamps: true,
        prompt: Some(prompt.into()),
        stream,
    };
    let (mut client, worker) = spawn_worker();

    let mut all = Vec::new();
    client
        .state_sync_streaming(req(true), &mut |seg| {
            all.push(seg.text);
            true
        })
        .expect("streaming transcribe");
    assert_eq!(all.len(), total, "control run should decode the whole file");

    let mut seen = Vec::new();
    let ok = client
        .state_sync_streaming(req(true), &mut |seg| {
            seen.push(seg.text);
            false
        })
        .expect("a cancelled transcription still replies");
    assert_eq!(seen[0], "one");
    assert!(
        seen.len() <= 3,
        "want an early stop, decoded {} of {total}: {seen:?}",
        seen.len()
    );
    match ok {
        linkw::StateSyncOkMsg::Transcribe { text, segments, .. } => {
            assert_eq!(segments.len(), seen.len());
            assert_eq!(text, seen.join(" "));
        }
        other => panic!("expected a Transcribe reply, got {other:?}"),
    }

    let ok = client.state_sync(req(false)).expect("plain transcribe after a cancel");
    match ok {
        linkw::StateSyncOkMsg::Transcribe { segments, .. } => assert_eq!(segments.len(), 1),
        other => panic!("expected a Transcribe reply, got {other:?}"),
    }

    drop(client);
    worker.join().expect("worker thread");
}

#[test]
fn a_cancel_arriving_after_the_decode_does_not_kill_the_worker() {
    let (mut client, worker) = spawn_worker();
    let req = |stream| linkw::StateSyncReqMsg::Transcribe {
        audio_path: "unused-by-the-mock.wav".into(),
        language: None,
        translate: false,
        timestamps: true,
        prompt: None,
        stream,
    };
    client
        .state_sync_streaming(req(true), &mut |_| true)
        .expect("streaming transcribe");

    client.send_transcribe_cancel().expect("send cancel");

    match client.state_sync(req(false)).expect("worker still serves after a late cancel") {
        linkw::StateSyncOkMsg::Transcribe { segments, .. } => assert_eq!(segments.len(), 1),
        other => panic!("expected a Transcribe reply, got {other:?}"),
    }
    drop(client);
    worker.join().expect("worker thread");
}
