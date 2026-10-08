use std::os::unix::net::UnixStream;
use std::time::Duration;

use superfluid_linkf::{AgentAction, AgentSession, Endpoint, HeadSession};
use superfluid_proto::handshake::{CapacitySummary, HelloAckF};
use superfluid_proto::linkf::{self, msg_type};

fn t(ms: u64) -> Duration {
    Duration::from_millis(ms)
}

fn ackf() -> HelloAckF {
    HelloAckF {
        chosen_version: 1,
        node_identity: "node-a".into(),
        capacity: CapacitySummary {
            gpu_memory_bytes: 1 << 31,
            gpu_memory_free_bytes: 1 << 30,
            host_memory_bytes: 1 << 33,
            active_sessions: 0,
            max_sessions: 8,
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
        loadable_models: vec!["mock".into()],
        feature_flags: 0,
        limits: linkf::LinkFLimits {
            max_frame: superfluid_proto::MAX_FRAME,
            transfer_window: 4,
            max_transfer: 1 << 24,
            transfer_budget: 1 << 26,
        },
    }
}

fn connect() -> (Endpoint, Endpoint) {
    let (h, a) = UnixStream::pair().expect("socketpair");
    let mut head_ep = Endpoint::new(h).unwrap();
    let mut agent_ep = Endpoint::new(a).unwrap();
    head_ep
        .send(
            msg_type::HELLO,
            &superfluid_proto::handshake::Hello {
                proto_versions: (
                    superfluid_linkf::driver::PROTO_VERSION,
                    superfluid_linkf::driver::PROTO_VERSION,
                ),
                link_role: superfluid_proto::handshake::LinkRole::HeadDaemon,
                auth: Vec::new(),
            },
        )
        .unwrap();
    agent_ep.answer_hello_as_agent(&ackf(), &[]).unwrap();
    let frames = head_ep.wait_frames().unwrap();
    assert!(frames.iter().any(|f| f.msg_type == msg_type::HELLO_ACK_F));
    (head_ep, agent_ep)
}

#[test]
fn session_lifecycle_recovery_and_fencing() {
    let mut head = HeadSession::new();
    let mut agent = AgentSession::new();

    let (mut head_ep, mut agent_ep) = connect();

    let lease = linkf::Lease {
        duration_ms: 10_000,
        renew_by_ms: 6_000,
    };
    let assign = head.assign_session(1, 5, lease, 0, "chatml");
    head_ep.send(msg_type::SESSION_ASSIGN, &assign).unwrap();
    let gen_req = head
        .generate(
            1,
            100,
            linkf::GenerationBudgetsMsg {
                max_new_tokens: 8,
                max_wall_ms: 1000,
            },
        )
        .unwrap();
    head_ep.send(msg_type::GENERATE_REQ, &gen_req).unwrap();

    for frame in agent_ep.wait_frames().unwrap() {
        match frame.msg_type {
            msg_type::SESSION_ASSIGN => {
                let msg: linkf::SessionAssignMsg = postcard::from_bytes(&frame.payload).unwrap();
                assert!(agent.on_assign(&msg, t(0)));
            }
            msg_type::GENERATE_REQ => {
                let msg: linkf::GenerateReqMsg = postcard::from_bytes(&frame.payload).unwrap();
                assert!(agent.on_generate(&msg));
            }
            other => panic!("unexpected frame 0x{other:04x}"),
        }
    }

    for k in 0..3u32 {
        let batch = agent
            .emit(1, 100, linkf::TokenEventPayload::Tokens(vec![k]))
            .unwrap();
        agent_ep.send(msg_type::TOKEN_EVENTS, &batch).unwrap();
    }
    let mut acks = Vec::new();
    while acks.len() < 3 {
        for frame in head_ep.wait_frames().unwrap() {
            assert_eq!(frame.msg_type, msg_type::TOKEN_EVENTS);
            let msg: linkf::TokenEventsMsg = postcard::from_bytes(&frame.payload).unwrap();
            acks.extend(head.on_token_events(&msg));
        }
    }
    assert_eq!(head.wal().total_committed(), 3);
    agent.on_watermark_ack(&acks[0]);
    assert_eq!(agent.unacked_events(1, 100), 2);

    drop(head_ep);
    drop(agent_ep);

    let (mut head_ep, mut agent_ep) = connect();
    let rec = head.reconcile();
    head_ep.send(msg_type::RECONCILE, &rec).unwrap();
    let frames = agent_ep.wait_frames().unwrap();
    let rec_frame = &frames[0];
    assert_eq!(rec_frame.msg_type, msg_type::RECONCILE);
    let rec_msg: linkf::ReconcileMsg = postcard::from_bytes(&rec_frame.payload).unwrap();
    assert_eq!(
        rec_msg.sessions[0].generation_watermarks[0].watermark,
        Some(2)
    );
    let (ack, retransmissions) = agent.on_reconcile(&rec_msg, t(1000));
    agent_ep.send(msg_type::RECONCILE_ACK, &ack).unwrap();
    assert!(ack.sessions[0].held);
    assert!(retransmissions.is_empty());

    let b4 = agent
        .emit(1, 100, linkf::TokenEventPayload::Tokens(vec![3]))
        .unwrap();
    let _b5_lost = agent
        .emit(1, 100, linkf::TokenEventPayload::Tokens(vec![4]))
        .unwrap();
    let acks = head.on_token_events(&b4);
    assert_eq!(acks[0].watermark, 3);
    let rec = head.reconcile();
    let (_ack, retransmissions) = agent.on_reconcile(&rec, t(1100));
    assert_eq!(retransmissions.len(), 1);
    assert_eq!(retransmissions[0].events.len(), 1);
    assert_eq!(retransmissions[0].events[0].event_seq, 4);
    let before = head.wal().total_committed();
    head.on_token_events(&retransmissions[0]);
    assert_eq!(head.wal().total_committed(), before + 1);
    let acks = head.on_token_events(&retransmissions[0]);
    assert_eq!(head.wal().total_committed(), before + 1);
    assert_eq!(acks[0].watermark, 4);

    let mut old_agent_view = agent;
    let mut new_agent = AgentSession::new();
    let assign2 = head.assign_session(1, 6, lease, 0, "chatml");
    assert!(new_agent.on_assign(&assign2, t(2000)));
    let gen2 = head
        .generate(
            1,
            101,
            linkf::GenerationBudgetsMsg {
                max_new_tokens: 4,
                max_wall_ms: 500,
            },
        )
        .unwrap();
    assert!(new_agent.on_generate(&gen2));

    let late = old_agent_view
        .emit(1, 100, linkf::TokenEventPayload::Tokens(vec![99]))
        .unwrap();
    let before = head.wal().total_committed();
    let acks = head.on_token_events(&late);
    assert!(acks.is_empty());
    assert_eq!(head.wal().total_committed(), before);

    let rec = head.reconcile();
    let (ack, _r) = old_agent_view.on_reconcile(&rec, t(2100));
    assert!(!ack.sessions[0].held);
    assert_eq!(old_agent_view.is_fenced(1), Some(true));

    let batch = new_agent
        .emit(1, 101, linkf::TokenEventPayload::Tokens(vec![7]))
        .unwrap();
    let acks = head.on_token_events(&batch);
    assert_eq!(acks[0].epoch, 6);

    assert_eq!(new_agent.needs_renew(t(8_500)).len(), 1);
    let renew = &new_agent.needs_renew(t(8_500))[0];
    let grant = head.on_lease_renew(renew).unwrap();
    new_agent.on_lease_grant(&grant, t(8_500));
    assert!(
        new_agent.poll(t(12_000)).is_empty(),
        "renewed lease expired early"
    );
    let actions = new_agent.poll(t(19_000));
    assert_eq!(actions, vec![AgentAction::SelfFence { session_id: 1 }]);
    assert!(new_agent
        .emit(1, 101, linkf::TokenEventPayload::Tokens(vec![8]))
        .is_none());
    assert_eq!(new_agent.unacked_events(1, 101), 1);

    head.wal().assert_single_contiguous_streams();
}

#[test]
fn stale_lease_grant_for_superseded_epoch_rejected() {
    let mut head = HeadSession::new();
    let mut agent = AgentSession::new();
    let lease = linkf::Lease {
        duration_ms: 1000,
        renew_by_ms: 600,
    };
    let assign = head.assign_session(1, 5, lease, 0, "chatml");
    agent.on_assign(&assign, t(0));
    let renew = linkf::LeaseRenewMsg {
        session_id: 1,
        epoch: 5,
    };
    let grant = head.on_lease_renew(&renew).unwrap();

    let assign2 = head.assign_session(1, 6, lease, 0, "chatml");
    agent.on_assign(&assign2, t(100));

    agent.on_lease_grant(&grant, t(100));
    let actions = agent.poll(t(1200));
    assert_eq!(actions, vec![AgentAction::SelfFence { session_id: 1 }]);

    assert!(head.on_lease_renew(&renew).is_none());
}
