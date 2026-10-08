//! Postcard round-trip tests for representative messages from both catalogs.

use superfluid_proto::linkf::*;
use superfluid_proto::linkw::*;

fn roundtrip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let bytes = postcard::to_allocvec(value).expect("serialize");
    postcard::from_bytes(&bytes).expect("deserialize")
}

#[test]
fn tick_submit_with_several_lanes_roundtrips() {
    let plan = TickPlanMsg {
        plan_seq: 42,
        flags: 0b11,
        prefill_token_budget: 4096,
        max_decode_lanes: 8,
        admits: vec![
            LaneAdmitMsg {
                lane_tag: 1,
                prompt: TokenRefMsg {
                    ring_id: 0,
                    index: 10,
                    count: 3,
                    generation: 7,
                },
                seed_handle: 0,
                sampling: 0,
                params: SamplingParamsMsg {
                    temperature: 0.8,
                    top_p: 0.95,
                    min_p: 0.0,
                    top_k: 40,
                    freq_penalty: 0.0,
                    presence_penalty: 0.0,
                    repeat_penalty: 1.1,
                    flags: 0,
                },
                rng_counter_base: 100,
                grammar_handle: 0,
                logit_bias_handle: 0,
                want_logprobs: false,
                top_logprobs: 0,
                strategy_slot: 0,
                determinism_class: 0,
                minimum_exactness: 0,
                allow_approximate: true,
                required_cert_id: 0,
                host_sampler_identity: [0u8; 32],
                grammar_replay: 0,
                decode_replay: 0,
            },
            LaneAdmitMsg {
                lane_tag: 2,
                prompt: TokenRefMsg {
                    ring_id: 0,
                    index: 20,
                    count: 5,
                    generation: 7,
                },
                seed_handle: 55,
                sampling: 1,
                params: SamplingParamsMsg {
                    temperature: 0.0,
                    top_p: 1.0,
                    min_p: 0.0,
                    top_k: 0,
                    freq_penalty: 0.0,
                    presence_penalty: 0.0,
                    repeat_penalty: 1.0,
                    flags: 0,
                },
                rng_counter_base: 0,
                grammar_handle: 3,
                logit_bias_handle: 0,
                want_logprobs: false,
                top_logprobs: 0,
                strategy_slot: 1,
                determinism_class: 1,
                minimum_exactness: 2,
                allow_approximate: false,
                required_cert_id: 9,
                host_sampler_identity: [1u8; 32],
                grammar_replay: 12,
                decode_replay: 7,
            },
        ],
        commits: vec![LaneCommitMsg {
            lane_tag: 1,
            token_id: 555,
            logits_nonce: 999,
        }],
        prefills: vec![LanePrefillMsg {
            lane_tag: 2,
            token_offset: 0,
            token_count: 128,
        }],
        decodes: vec![
            LaneDecodeMsg {
                lane_tag: 1,
                max_new_tokens: 64,
                overshoot: 0,
        },
            LaneDecodeMsg {
                lane_tag: 2,
                max_new_tokens: 32,
                overshoot: 0,
        },
        ],
        retires: vec![LaneRetireMsg {
            lane_tag: 3,
            publish_to_cache: true,
        }],
        shed_policy: ShedPolicyMsg {
            victim_lanes: vec![9, 10, 11],
            evictable_cache_classes: 0xFF,
            protected_quota_bytes: 1 << 20,
            max_evict_bytes: 1 << 24,
        },
    };
    let msg = TickSubmitMsg { plan };
    assert_eq!(roundtrip(&msg), msg);
}

#[test]
fn tick_result_roundtrips() {
    let msg = TickResultMsg::Events(TickEventsMsg {
        plan_seq: 42,
        tick_status: 0,
        admit_results: vec![AdmitResultMsg {
            lane_tag: 1,
            status: 1,
            reject_code: 0,
            cert_id: 3,
            granted_class: 2,
        }],
        emits: vec![LaneEmitMsg {
            lane_tag: 1,
            token_ref: TokenRefMsg {
                ring_id: 0,
                index: 1,
                count: 1,
                generation: 1,
            },
            n_tokens: 1,
            finish: 0,
            logits_row: RingRefMsg {
                ring_id: 1,
                index: 2,
                generation: 1,
            },
            spec: SpecStatsMsg {
                proposed: 4,
                accepted: 3,
                cert_id_in_effect: 3,
            },
        }],
        faults: vec![],
        shed: ShedReportMsg {
            prefill_chunks_dropped: 0,
            evictions_performed: 1,
            bytes_evicted: 4096,
            entries: vec![ShedEntryMsg {
                lane_tag: 5,
                kind: 1,
                reason: 0,
                bytes: 4096,
                tokens: 32,
            }],
        },
        op_completions: vec![OpCompleteMsg {
            op: 77,
            state: 2,
            error: 0,
            bytes_moved: 8192,
        }],
        timings: TickTimingsMsg {
            wall_ns: 1_000_000,
            prefill_ns: 400_000,
            decode_ns: 600_000,
            graph_hits: 10,
            graph_misses: 1,
        },
        mem: MemCountersMsg {
            allocated_bytes: 1 << 30,
            host_retained_bytes: 1 << 20,
            pool_blocks_total: 100,
            pool_blocks_used: 40,
            pool_bytes_evictable: 1 << 16,
            tentative_bytes: 0,
        },
        logprobs: vec![LaneLogprobMsg { lane_tag: 1, logprob_bits: (-0.5f32).to_bits(), top_ids: vec![7, 9], top_logprob_bits: vec![(-0.5f32).to_bits(), (-1.2f32).to_bits()] }],
    });
    assert_eq!(roundtrip(&msg), msg);
    let rejected = TickResultMsg::Rejected { status: -113 };
    assert_eq!(roundtrip(&rejected), rejected);
}

#[test]
fn strategy_register_and_grant_roundtrip() {
    let reg = StrategyRegisterMsg {
        strategy_id: "eagle3".into(),
        impl_version: "1.2.3".into(),
        impl_hash: [2u8; 32],
        config_hash: [3u8; 32],
        artifacts: vec![ArtifactMsg {
            role: 1,
            content_hash: [4u8; 32],
            byte_size: 123_456,
            load_path: "/models/draft.bin".into(),
        }],
        taps: vec![TapSpecMsg {
            layer: 5,
            tensor: 1,
            dtype: 0,
            layout: 0,
        }],
        capabilities: vec![CapabilityReqMsg {
            kind_id: 1,
            params: vec![1, 2, 3],
        }],
        target_archs: vec!["cuda_sm90".into(), "metal".into()],
        kernel_caps_required: 0b101,
        est_state_bytes: 4096,
        claimed_exactness: 2,
        rng_contract_version: 1,
    };
    assert_eq!(roundtrip(&reg), reg);

    let grant = StrategyGrantMsg::Granted(StrategyGrantOkMsg {
        strategy_slot: 0,
        certificates: vec![ExactnessCertMsg {
            cert_id: 1,
            exactness: 1,
            sampling_modes: 0b011,
            param_domain: 1,
            grammar_allowed: false,
            logit_bias_allowed: true,
            verification_shape_class: 0,
            admissible_host_identities: vec![[9u8; 32]],
        }],
        state_spaces: vec![],
        reserved_bytes: 0,
    });
    assert_eq!(roundtrip(&grant), grant);
    let refused = StrategyGrantMsg::Refused { status: -128 };
    assert_eq!(roundtrip(&refused), refused);
}

#[test]
fn transcribe_segment_frame_roundtrips() {
    let seg = superfluid_proto::linkw::TranscribeSegmentMsg {
        start_ms: 0,
        end_ms: 1_500,
        text: " And so, my fellow Americans".into(),
    };
    assert_eq!(roundtrip(&seg), seg);
}

#[test]
fn transcribe_cancel_frame_roundtrips() {
    let c = superfluid_proto::linkw::TranscribeCancelMsg {};
    assert_eq!(roundtrip(&c), c);
}

#[test]
fn state_sync_union_roundtrips_each_variant() {
    let variants = vec![
        StateSyncReqMsg::Fork {
            parent_sequence: 1,
            flags: 1,
        },
        StateSyncReqMsg::Trim {
            sequence: 1,
            space_id: 2,
            new_len: 100,
        },
        StateSyncReqMsg::ExportSize {
            sequence: 1,
            space_id: 2,
            range: TokenRangeMsg { start: 0, end: 10 },
            encoding: 0,
        },
        StateSyncReqMsg::CacheEvict {
            bytes_target: 1000,
            policy: ShedPolicyMsg {
                victim_lanes: vec![1],
                evictable_cache_classes: 1,
                protected_quota_bytes: 0,
                max_evict_bytes: 1000,
            },
        },
        StateSyncReqMsg::Transcribe {
            audio_path: "/tmp/a.wav".into(),
            language: Some("en".into()),
            translate: true,
            timestamps: true,
            prompt: Some("Zyzzyva".into()),
            stream: true,
        },
        StateSyncReqMsg::Transcribe {
            audio_path: "/tmp/b.wav".into(),
            language: None,
            translate: false,
            timestamps: false,
            prompt: None,
            stream: false,
        },
        StateSyncReqMsg::SnapshotBoundary {
            sequence: 1,
            space_id: 2,
            cap: 63,
        },
        StateSyncReqMsg::CacheEvictEntries {
            keys: vec![CacheKeyMsg {
                compat_key: [7; 32],
                provenance_digest: [9; 32],
            }],
        },
        StateSyncReqMsg::Unpublish {
            span: TokenRefMsg {
                ring_id: 1,
                index: 128,
                count: 64,
                generation: 2,
            },
        },
    ];
    for v in variants {
        assert_eq!(roundtrip(&v), v);
    }

    let results = vec![
        StateSyncResMsg::Ok(StateSyncOkMsg::Fork { child_sequence: 2 }),
        StateSyncResMsg::Ok(StateSyncOkMsg::Trim),
        StateSyncResMsg::Ok(StateSyncOkMsg::ExportSize {
            required_bytes: 4096,
            sizing_gen: 7,
        }),
        StateSyncResMsg::Ok(StateSyncOkMsg::CacheEvict { bytes_freed: 500 }),
        StateSyncResMsg::Ok(StateSyncOkMsg::SnapshotBoundary { boundary: 40 }),
        StateSyncResMsg::Ok(StateSyncOkMsg::CacheEvictEntries),
        StateSyncResMsg::Ok(StateSyncOkMsg::Unpublish),
        StateSyncResMsg::Err { status: -123 },
    ];
    for r in results {
        assert_eq!(roundtrip(&r), r);
    }
}

#[test]
fn session_assign_roundtrips() {
    let msg = SessionAssignMsg {
        command_id: 1001,
        session_id: 55,
        epoch: 3,
        lease: Lease {
            duration_ms: 30_000,
            renew_by_ms: 20_000,
        },
        codec: "anthropic-messages".into(),
        params: vec![9, 9, 9],
        qos_class: 1,
        policy_ref: 42,
    };
    assert_eq!(roundtrip(&msg), msg);
}

#[test]
fn token_events_roundtrips_all_payload_kinds() {
    let msg = TokenEventsMsg {
        events: vec![
            TokenEvent {
                session_id: 1,
                epoch: 1,
                generation_id: 1,
                event_seq: 0,
                payload: TokenEventPayload::Tokens(vec![10, 20, 30]),
            },
            TokenEvent {
                session_id: 1,
                epoch: 1,
                generation_id: 1,
                event_seq: 1,
                payload: TokenEventPayload::Segment {
                    kind: SegmentKind::TextDelta,
                    data: b"hello".to_vec(),
                },
            },
            TokenEvent {
                session_id: 1,
                epoch: 1,
                generation_id: 1,
                event_seq: 2,
                payload: TokenEventPayload::Usage(UsageMsg {
                    input_tokens: 12,
                    output_tokens: 34,
                }),
            },
            TokenEvent {
                session_id: 1,
                epoch: 1,
                generation_id: 1,
                event_seq: 3,
                payload: TokenEventPayload::Finish { reason: 1 },
            },
        ],
    };
    assert_eq!(roundtrip(&msg), msg);
}

#[test]
fn reconcile_and_reconcile_ack_roundtrip() {
    let reconcile = ReconcileMsg {
        sessions: vec![SessionReconcileState {
            session_id: 1,
            epoch: 4,
            lease_state: LeaseStateMsg {
                lease: Lease {
                    duration_ms: 10_000,
                    renew_by_ms: 8_000,
                },
                elapsed_ms: 1_000,
            },
            command_watermark: 500,
            generation_watermarks: vec![GenerationWatermark {
                generation_id: 1,
                watermark: Some(100),
            }],
        }],
    };
    assert_eq!(roundtrip(&reconcile), reconcile);

    let ack = ReconcileAckMsg {
        sessions: vec![SessionReconcileAck {
            session_id: 1,
            held: true,
            retry_ranges: vec![RetryRange {
                generation_id: 1,
                from_seq: 90,
                to_seq: 100,
            }],
            lease_view: LeaseStateMsg {
                lease: Lease {
                    duration_ms: 10_000,
                    renew_by_ms: 8_000,
                },
                elapsed_ms: 1_500,
            },
        }],
    };
    assert_eq!(roundtrip(&ack), ack);
}

#[test]
fn xfer_and_state_ship_roundtrip() {
    let begin = XferBeginMsg {
        transfer_id: 1,
        total_len: 4096,
        checksum: 0xDEAD_BEEF,
        content_type: 1,
        content_hash: [7u8; 32],
    };
    assert_eq!(roundtrip(&begin), begin);

    let chunk = XferChunkMsg {
        transfer_id: 1,
        offset: 0,
        data: vec![1, 2, 3, 4],
    };
    assert_eq!(roundtrip(&chunk), chunk);

    let ship = StateShipMsg {
        envelope: vec![9, 9, 9],
        transfer_id: 1,
    };
    assert_eq!(roundtrip(&ship), ship);
}

#[test]
fn handshake_hello_and_acks_roundtrip() {
    use superfluid_proto::handshake::*;

    let hello = Hello {
        proto_versions: (1, 3),
        link_role: LinkRole::NodeAgent,
        auth: vec![1, 2, 3],
    };
    assert_eq!(roundtrip(&hello), hello);

    let ack_w = HelloAckW {
        chosen_version: 4,
        engine_bundle_hash: [1u8; 32],
        capabilities: r#"{"descriptor_version":1,"serving":{"park_lossy":"no lossy encoding"}}"#.to_string(),
        state_space_descriptors: vec![StateSpaceDescMsg {
            space_id: 0,
            kind: 1,
            version_tag: 1,
            bytes_per_token: 128,
            blob_bytes: 0,
            page_size_tokens: 16,
            fork_cost_class: 0,
            fork_cost_bytes: 0,
            snapshot_cadence: 0,
            snapshot_interval_tokens: 0,
            placement: 0,
            flags: 0,
            name: "kv0".into(),
        }],
        registered_strategies: vec![StrategySummary {
            strategy_id: "eagle3".into(),
            strategy_slot: 0,
            impl_version: "1.0.0".into(),
            claimed_exactness: 1,
        }],
        limits: LinkWLimits {
            max_frame: MAX_FRAME_TEST,
            op_window: 8,
            ring_specs: vec![RingSpec {
                ring_id: 0,
                kind: RingKind::Tokens,
                slot_bytes: 4,
                slots: 1024,
            }],
        },
        kv_bits: 16,
    };
    assert_eq!(roundtrip(&ack_w), ack_w);

    let ack_f = HelloAckF {
        chosen_version: 3,
        node_identity: "node-a".into(),
        capacity: CapacitySummary {
            gpu_memory_bytes: 1 << 34,
            gpu_memory_free_bytes: 1 << 30,
            host_memory_bytes: 1 << 36,
            active_sessions: 2,
            max_sessions: 16,
            max_context_tokens: 4096,
            kv_blocks_used: 0,
            kv_blocks_total: 0,
            lanes_active: 0,
            max_lanes: 0,
            queue_depth: 0,
            decode_tokens_per_s: 0,
            prefill_tokens_per_s: 0,
            sampled_at_ms: 0,
        },
        loadable_models: vec!["glm-5.2".into()],
        feature_flags: 0,
        limits: LinkFLimits {
            max_frame: MAX_FRAME_TEST,
            transfer_window: 4,
            max_transfer: 1 << 30,
            transfer_budget: 1 << 31,
        },
    };
    assert_eq!(roundtrip(&ack_f), ack_f);
}

const MAX_FRAME_TEST: u32 = 16 * 1024 * 1024;
