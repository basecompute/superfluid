//! Fleet CHAT generation through the head codec.

use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::fleet::FleetHead;
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::wal::role;
use superfluid_daemon::{Daemon, EngineHost, GenParams, MockCodec, SessionStore, TextCodec};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkf::Lease;

static DIRS: AtomicU64 = AtomicU64::new(0);

fn mock_daemon() -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fchat-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

fn spawn_node() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let mut node = NodeAgent::new(daemon, "chat-node", "mock");
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            if node.serve(ep).is_err() {
                break;
            }
        }
    });
    addr
}

#[test]
fn fleet_chat_renders_generates_and_channelizes_via_head_codec() {
    let addr = spawn_node();
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let codec = MockChatCodec;
    let messages = vec![
        superfluid_daemon::codec::ChatMessage::new(role::SYSTEM, "You are helpful."),
        superfluid_daemon::codec::ChatMessage::new(role::USER, "hello there"),
    ];

    let mut head = FleetHead::connect(&addr, Vec::new()).unwrap();

    head.assign(1, GenParams::default(), 0, "chatml", lease).unwrap();
    let segs = head.generate_chat(1, &codec, &messages, &[], 12, 60_000).unwrap();
    assert!(!segs.is_empty(), "chat generation produced no segments");
    let seg_text: String = segs.iter().map(|s| s.text.as_str()).collect();

    let prompt = codec
        .render_prompt_structured(&messages, &[])
        .expect("render_prompt_structured");
    assert!(
        prompt.windows(1).count() >= messages.len(),
        "render_prompt_structured applied the chat template"
    );
    head.assign(2, GenParams::default(), 0, "chatml", lease).unwrap();
    head.append(2, &prompt).unwrap();
    let raw = head.generate(2, 12, 60_000).unwrap();
    let mut chan = codec.channelizer();
    let manual: String = chan
        .split(&raw.tokens)
        .iter()
        .map(|r| codec.decode(&r.text))
        .collect();

    assert_eq!(seg_text, manual, "generate_chat != render+generate+channelize");
}
