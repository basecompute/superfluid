//! One node, end to end.

use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use superfluid_daemon::fleet::FleetHead;
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::{Daemon, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkf::{Lease, TokenEventPayload};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};

static DIRS: AtomicU64 = AtomicU64::new(0);

fn wal_path() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fleet-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("wal.log")
}

fn mock_daemon() -> Daemon {
    let store = SessionStore::open(&wal_path()).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

fn spawn_node() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let node = NodeAgent::new(daemon, "node-a", "mock");
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    addr
}

fn spawn_keyed_node(token: &'static [u8]) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let node = NodeAgent::new(daemon, "keyed", "mock").with_auth(token.to_vec());
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp_secure(
                sock,
                token,
                superfluid_linkf::Role::Responder,
                std::time::Duration::from_secs(5),
            );
            let Ok(ep) = ep else { continue };
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    addr
}

#[test]
fn a_keyed_node_serves_over_an_encrypted_link_and_refuses_another_key() {
    let prompt: Vec<u32> = (300..332).collect();
    let params = GenParams::default();
    let (local_tokens, _) = mock_daemon().generate_tokens(prompt.clone(), params, 8, 0).unwrap();
    let addr = spawn_keyed_node(b"the fleet token");
    assert!(FleetHead::connect(&addr, b"another token".to_vec()).is_err());
    let mut head = FleetHead::connect(&addr, b"the fleet token".to_vec()).expect("connect with the token");
    head.assign(7, params, 0, "chatml", test_lease()).unwrap();
    head.append(7, &prompt).unwrap();
    let gen = head.generate(7, 8, 60_000).unwrap();
    assert_eq!(gen.tokens, local_tokens);
}

fn spawn_narrow_node(ctx: u32) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let store = SessionStore::open(&wal_path()).unwrap();
        let host = EngineHost::spawn(move || {
            let cfg = EngineConfig::default();
            let vocab = cfg.vocab;
            let e = MockEngine::new(cfg);
            let specs = vec![
                linkw::RingSpec {
                    ring_id: TOKEN_RING_IN,
                    kind: linkw::RingKind::Tokens,
                    slot_bytes: 16 + ctx * 4,
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
            (e, Some(specs))
        })
        .unwrap();
        let daemon = Arc::new(Daemon::new(store, host, Box::new(MockCodec), 8));
        let node = NodeAgent::new(daemon, "narrow", "mock");
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let mut conn = node.connection();
            std::thread::spawn(move || {
                let _ = conn.serve(ep);
            });
        }
    });
    addr
}

fn test_lease() -> Lease {
    Lease {
        duration_ms: 60_000,
        renew_by_ms: 40_000,
    }
}

#[test]
fn fleet_generation_matches_local_token_for_token() {
    let prompt: Vec<u32> = (200..232).collect();
    let params = GenParams::default();
    let max = 16u32;

    let local = mock_daemon();
    let (local_tokens, expired) = local.generate_tokens(prompt.clone(), params, max, 0).unwrap();
    assert!(!expired);
    assert!(!local_tokens.is_empty());

    let addr = spawn_node();
    let lease = Lease {
        duration_ms: 60_000,
        renew_by_ms: 40_000,
    };
    let mut head = FleetHead::connect(&addr, Vec::new()).expect("connect node");
    assert_eq!(head.node().loadable_models, vec!["mock".to_string()]);

    let session = 42u64;
    head.assign(session, params, 0, "chatml", lease).unwrap();
    head.append(session, &prompt).unwrap();
    let gen = head.generate(session, max as u64, 60_000).unwrap();

    assert!(!gen.expired);
    assert_eq!(
        gen.tokens, local_tokens,
        "fleet path diverged from the local path"
    );
    head.wal().assert_single_contiguous_streams();
    let events = head.wal().session_events(session);
    let mut committed_tokens = Vec::new();
    let mut finishes = 0;
    for c in events {
        match &c.event.payload {
            TokenEventPayload::Tokens(t) => committed_tokens.extend_from_slice(t),
            TokenEventPayload::Finish { .. } => finishes += 1,
            _ => {}
        }
    }
    assert_eq!(committed_tokens, local_tokens, "committed tokens diverged");
    assert_eq!(finishes, 1, "exactly one Finish event");
    assert!(
        events.len() <= local_tokens.len() + 1,
        "coalescing must not INCREASE the frame count"
    );
}

#[test]
fn fleet_reassignment_fences_and_second_generation_wins() {
    let addr = spawn_node();
    let lease = Lease {
        duration_ms: 60_000,
        renew_by_ms: 40_000,
    };
    let mut head = FleetHead::connect(&addr, Vec::new()).unwrap();
    let session = 7u64;
    let params = GenParams::default();
    let prompt: Vec<u32> = (10..26).collect();

    let e1 = head.assign(session, params, 0, "chatml", lease).unwrap();
    head.append(session, &prompt).unwrap();
    let g1 = head.generate(session, 8, 60_000).unwrap();
    assert!(!g1.tokens.is_empty());

    let e2 = head.assign(session, params, 0, "chatml", lease).unwrap();
    assert!(e2 > e1);
    head.append(session, &prompt).unwrap();
    let g2 = head.generate(session, 8, 60_000).unwrap();
    assert_eq!(g2.tokens, g1.tokens, "same prompt, same greedy output");

    head.wal().assert_single_contiguous_streams();
}

#[test]
fn a_node_refusal_reports_a_finish_reason_instead_of_dropping_the_link() {
    const CTX: u32 = 64;
    let addr = spawn_narrow_node(CTX);
    let mut head = superfluid_daemon::fleet::FleetHead::connect(&addr, Vec::new()).expect("connect");
    assert_eq!(
        head.node().capacity.max_context_tokens,
        CTX as u64,
        "the node advertises its real ceiling, which is what lets the head pre-check"
    );

    let session = 1u64;
    head.assign(session, GenParams::default(), 0, "mock", test_lease()).unwrap();
    let overlong: Vec<u32> = (0..CTX + 32).collect();
    head.append(session, &overlong).unwrap();
    let g = head.generate(session, 8, 60_000).expect("the link survives the refusal");
    assert_eq!(
        g.finish,
        superfluid_daemon::nodeagent::FINISH_REFUSED,
        "the node reports WHY it refused"
    );
    assert!(g.tokens.is_empty(), "a refused generation produces nothing");

    let session = 2u64;
    head.assign(session, GenParams::default(), 0, "mock", test_lease()).unwrap();
    head.append(session, &(0..8).collect::<Vec<u32>>()).unwrap();
    let g = head.generate(session, 4, 60_000).expect("the connection still serves");
    assert_eq!(g.tokens.len(), 4);
}

#[test]
fn a_second_head_reusing_a_session_id_is_never_left_hanging() {
    let addr = spawn_node();
    let session = 1u64 << 40;

    {
        let mut first = FleetHead::connect(&addr, Vec::new()).expect("connect");
        first.assign(session, GenParams::default(), 0, "mock", test_lease()).unwrap();
        first.append(session, &(0..8).collect::<Vec<u32>>()).unwrap();
        for _ in 0..3 {
            let g = first.generate(session, 4, 60_000).expect("first head generates");
            assert_eq!(g.tokens.len(), 4);
        }
    }

    let mut second = FleetHead::connect(&addr, Vec::new()).expect("reconnect");
    second.assign(session, GenParams::default(), 0, "mock", test_lease()).unwrap();
    second.append(session, &(0..8).collect::<Vec<u32>>()).unwrap();
    let g = second
        .generate(session, 4, 60_000)
        .expect("the second head must not hang on a node that remembers the id");
    assert!(
        g.finish == superfluid_daemon::nodeagent::FINISH_DECLINED || !g.tokens.is_empty(),
        "either it generated, or it was told why it did not — never silence"
    );

    let fresh = session + 9_999;
    second.assign(fresh, GenParams::default(), 0, "mock", test_lease()).unwrap();
    second.append(fresh, &(0..8).collect::<Vec<u32>>()).unwrap();
    let g = second.generate(fresh, 4, 60_000).expect("the connection still serves");
    assert_eq!(g.tokens.len(), 4);
}
