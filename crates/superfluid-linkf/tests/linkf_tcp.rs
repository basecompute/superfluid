//! Link F over TCP: the SAME state machines and frames as `linkf_e2e`, but across the network
//! transport that reaches Stage 0's other boxes.

use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use superfluid_linkf::{AgentSession, Endpoint, HeadSession};
use superfluid_proto::handshake::{CapacitySummary, HelloAckF};
use superfluid_proto::linkf::{self, msg_type};

fn t(ms: u64) -> Duration {
    Duration::from_millis(ms)
}

fn ackf() -> HelloAckF {
    HelloAckF {
        chosen_version: 1,
        node_identity: "node-tcp".into(),
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
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().unwrap();
    let accept = std::thread::spawn(move || {
        let (sock, _) = listener.accept().expect("accept");
        Endpoint::from_tcp(sock).unwrap()
    });
    let head_sock = TcpStream::connect(addr).expect("connect");
    let mut head_ep = Endpoint::from_tcp(head_sock).unwrap();
    let mut agent_ep = accept.join().unwrap();

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
fn assign_generate_stream_ack_over_tcp() {
    let mut head = HeadSession::new();
    let mut agent = AgentSession::new();
    let (mut head_ep, mut agent_ep) = connect();

    let lease = linkf::Lease {
        duration_ms: 10_000,
        renew_by_ms: 6_000,
    };
    let assign = head.assign_session(7, 3, lease, 0, "chatml");
    head_ep.send(msg_type::SESSION_ASSIGN, &assign).unwrap();
    let gen = head
        .generate(
            7,
            200,
            linkf::GenerationBudgetsMsg {
                max_new_tokens: 4,
                max_wall_ms: 1000,
            },
        )
        .unwrap();
    head_ep.send(msg_type::GENERATE_REQ, &gen).unwrap();

    let mut saw_assign = false;
    let mut saw_gen = false;
    while !(saw_assign && saw_gen) {
        for frame in agent_ep.wait_frames().unwrap() {
            match frame.msg_type {
                msg_type::SESSION_ASSIGN => {
                    let m: linkf::SessionAssignMsg = postcard::from_bytes(&frame.payload).unwrap();
                    assert!(agent.on_assign(&m, t(0)));
                    saw_assign = true;
                }
                msg_type::GENERATE_REQ => {
                    let m: linkf::GenerateReqMsg = postcard::from_bytes(&frame.payload).unwrap();
                    assert!(agent.on_generate(&m));
                    saw_gen = true;
                }
                other => panic!("unexpected 0x{other:04x}"),
            }
        }
    }

    for k in 0..4u32 {
        let batch = agent
            .emit(7, 200, linkf::TokenEventPayload::Tokens(vec![k]))
            .unwrap();
        agent_ep.send(msg_type::TOKEN_EVENTS, &batch).unwrap();
    }
    let mut acks = Vec::new();
    while acks.len() < 4 {
        for frame in head_ep.wait_frames().unwrap() {
            assert_eq!(frame.msg_type, msg_type::TOKEN_EVENTS);
            let m: linkf::TokenEventsMsg = postcard::from_bytes(&frame.payload).unwrap();
            acks.extend(head.on_token_events(&m));
        }
    }
    assert_eq!(head.wal().total_committed(), 4);
    assert_eq!(acks.last().unwrap().watermark, 3);

    drop(head_ep);
    drop(agent_ep);
    let (mut head_ep, mut agent_ep) = connect();
    let rec = head.reconcile();
    head_ep.send(msg_type::RECONCILE, &rec).unwrap();
    let frames = agent_ep.wait_frames().unwrap();
    let rec_msg: linkf::ReconcileMsg =
        postcard::from_bytes(&frames.iter().find(|f| f.msg_type == msg_type::RECONCILE).unwrap().payload)
            .unwrap();
    let (ack, retransmissions) = agent.on_reconcile(&rec_msg, t(1000));
    agent_ep.send(msg_type::RECONCILE_ACK, &ack).unwrap();
    assert!(ack.sessions[0].held);
    assert!(retransmissions.is_empty());

    head.wal().assert_single_contiguous_streams();
}
