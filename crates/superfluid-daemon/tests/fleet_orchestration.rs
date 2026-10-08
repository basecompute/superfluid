//! Head-side multi-node orchestration.

use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use superfluid_daemon::fleet_manager::{FleetManager, NodeSpec, PlacementPolicy};
use superfluid_daemon::{Daemon, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkf::Lease;
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};

static DIRS: AtomicU64 = AtomicU64::new(0);

fn mock_daemon() -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-orch-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

fn reference(prompt: &[u32], max: u32) -> Vec<u32> {
    let (t, _) = mock_daemon()
        .generate_tokens(prompt.to_vec(), GenParams::default(), max, 0)
        .unwrap();
    t
}

fn spawn_node(identity: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let node = superfluid_daemon::nodeagent::NodeAgent::new(daemon, identity, "mock");
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

fn spawn_node_model(identity: &'static str, model: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let node = superfluid_daemon::nodeagent::NodeAgent::new(daemon, identity, model);
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

fn dead_addr() -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let a = l.local_addr().unwrap().to_string();
    drop(l);
    a
}

fn lease() -> Lease {
    Lease { duration_ms: 120_000, renew_by_ms: 90_000 }
}

fn spec(addr: String) -> NodeSpec {
    NodeSpec { addr, auth: Vec::new() }
}

#[test]
fn least_loaded_placement_spreads_sessions() {
    let nodes = vec![
        spec(spawn_node("n0")),
        spec(spawn_node("n1")),
        spec(spawn_node("n2")),
    ];
    let mgr = FleetManager::new(nodes, PlacementPolicy::LeastLoaded, lease());
    for s in 0..6u64 {
        mgr.place(s + 1, GenParams::default(), 0, "raw", "mock", &(0..8).collect::<Vec<u32>>(), 64)
            .unwrap();
    }
    let load = mgr.load();
    assert_eq!(load.iter().sum::<usize>(), 6);
    assert_eq!(load, vec![2, 2, 2], "load not balanced: {load:?}");
}

#[test]
fn generation_routes_to_the_owning_node_and_is_correct() {
    let nodes = vec![spec(spawn_node("a")), spec(spawn_node("b"))];
    let mgr = FleetManager::new(nodes, PlacementPolicy::LeastLoaded, lease());
    for i in 0..4u64 {
        let prompt: Vec<u32> = (i as u32 * 30..i as u32 * 30 + 12).collect();
        mgr.place(i + 1, GenParams::default(), 0, "raw", "mock", &prompt, 64).unwrap();
        let got = mgr.generate(i + 1, 12, 120_000).unwrap();
        assert_eq!(got.tokens, reference(&prompt, 12), "session {i} wrong output");
    }
}

#[test]
fn round_robin_rotates_nodes() {
    let nodes = vec![spec(spawn_node("r0")), spec(spawn_node("r1")), spec(spawn_node("r2"))];
    let mgr = FleetManager::new(nodes, PlacementPolicy::RoundRobin, lease());
    let mut seen = Vec::new();
    for s in 0..3u64 {
        mgr.place(s + 1, GenParams::default(), 0, "raw", "mock", &(0..8).collect::<Vec<u32>>(), 64)
            .unwrap();
        seen.push(mgr.node_of(s + 1).unwrap().to_string());
    }
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), 3, "round-robin did not spread across all nodes");
}

#[test]
fn placement_skips_a_dead_node() {
    let live = spawn_node("live");
    let nodes = vec![spec(dead_addr()), spec(live)];
    let mgr = FleetManager::new(nodes, PlacementPolicy::RoundRobin, lease());
    let prompt: Vec<u32> = (5..17).collect();
    mgr.place(1, GenParams::default(), 0, "raw", "mock", &prompt, 64).unwrap();
    assert_eq!(mgr.node_of(1).as_deref(), Some("live"));
    let got = mgr.generate(1, 10, 120_000).unwrap();
    assert_eq!(got.tokens, reference(&prompt, 10));
}

#[test]
fn generation_fails_over_when_the_owning_node_dies() {
    use std::time::Duration;
    let listener_a = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr_a = listener_a.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let node = superfluid_daemon::nodeagent::NodeAgent::new(daemon, "a", "mock")
            .with_timeouts(Duration::from_secs(5), Duration::from_millis(200));
        if let Ok((sock, _)) = listener_a.accept() {
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let _ = node.connection().serve(ep);
        }
    });
    let addr_b = spawn_node("b");

    let mgr = FleetManager::new(
        vec![spec(addr_a), spec(addr_b)],
        PlacementPolicy::RoundRobin,
        lease(),
    );
    let prompt: Vec<u32> = (9..21).collect();
    mgr.place(1, GenParams::default(), 0, "raw", "mock", &prompt, 64).unwrap();
    assert_eq!(mgr.node_of(1).as_deref(), Some("a"), "expected initial placement on node a");

    std::thread::sleep(Duration::from_millis(500));

    let got = mgr.generate(1, 10, 120_000).expect("failover generate");
    assert_eq!(got.tokens, reference(&prompt, 10), "failover produced wrong output");
    assert_eq!(mgr.node_of(1).as_deref(), Some("b"), "session did not move to the survivor");
}

#[test]
fn capability_miss_does_not_poison_a_healthy_node() {
    let a = spawn_node_model("node-a", "model-a");
    let b = spawn_node_model("node-b", "model-b");
    let mgr = FleetManager::new(vec![spec(b), spec(a)], PlacementPolicy::RoundRobin, lease());
    let prompt: Vec<u32> = (0..10).collect();

    mgr.place(1, GenParams::default(), 0, "raw", "model-a", &prompt, 64).unwrap();
    assert_eq!(mgr.node_of(1).as_deref(), Some("node-a"));
    assert_eq!(mgr.generate(1, 8, 120_000).unwrap().tokens, reference(&prompt, 8));

    mgr.place(2, GenParams::default(), 0, "raw", "model-b", &prompt, 64).unwrap();
    assert_eq!(mgr.node_of(2).as_deref(), Some("node-b"));
    assert_eq!(mgr.generate(2, 8, 120_000).unwrap().tokens, reference(&prompt, 8));
}

#[test]
fn concurrent_generation_across_nodes_is_correct() {
    let nodes = vec![
        spec(spawn_node("c0")),
        spec(spawn_node("c1")),
        spec(spawn_node("c2")),
    ];
    let mgr = Arc::new(FleetManager::new(nodes, PlacementPolicy::LeastLoaded, lease()));

    let n = 12u64;
    let mut prompts = Vec::new();
    let mut expected = Vec::new();
    for i in 0..n {
        let prompt: Vec<u32> = (i as u32 * 30..i as u32 * 30 + 12).collect();
        mgr.place(i + 1, GenParams::default(), 0, "raw", "mock", &prompt, 64).unwrap();
        expected.push(reference(&prompt, 12));
        prompts.push(prompt);
    }
    assert_eq!(mgr.load().iter().sum::<usize>(), n as usize);
    assert!(
        mgr.load().iter().all(|&c| c > 0),
        "sessions did not spread across all nodes: {:?}",
        mgr.load()
    );

    let barrier = Arc::new(std::sync::Barrier::new(n as usize));
    let mut handles = Vec::new();
    for i in 0..n {
        let mgr = Arc::clone(&mgr);
        let b = Arc::clone(&barrier);
        let want = expected[i as usize].clone();
        handles.push(std::thread::spawn(move || {
            b.wait();
            let got = mgr.generate(i + 1, 12, 120_000).expect("concurrent generate");
            assert_eq!(got.tokens, want, "session {i} produced wrong output under concurrency");
        }));
    }
    for h in handles {
        h.join().expect("a concurrent generation panicked");
    }
}

#[test]
fn streaming_generation_delivers_incrementally() {
    let nodes = vec![spec(spawn_node("s0"))];
    let mgr = FleetManager::new(nodes, PlacementPolicy::LeastLoaded, lease());
    let prompt: Vec<u32> = (10..22).collect();
    mgr.place(1, GenParams::default(), 0, "raw", "mock", &prompt, 64).unwrap();

    const N: u64 = 40;
    let mut batches = 0usize;
    let mut collected: Vec<u32> = Vec::new();
    let g = mgr
        .generate_streaming(1, N, 120_000, |b| {
            batches += 1;
            collected.extend_from_slice(b);
        })
        .unwrap();
    assert_eq!(collected, reference(&prompt, N as u32), "streamed tokens wrong");
    assert_eq!(g.tokens, collected, "returned vector must equal what was streamed");
    assert_eq!(g.finish, superfluid_abi::finish::LENGTH);
    assert!(
        batches >= 2,
        "expected incremental delivery across coalesced frames, got {batches} batch(es)"
    );
}

#[test]
fn streaming_generation_cancels_mid_stream() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let nodes = vec![spec(spawn_node("x0"))];
    let mgr = Arc::new(FleetManager::new(nodes, PlacementPolicy::LeastLoaded, lease()));
    let prompt: Vec<u32> = (0..8).collect();
    mgr.place(1, GenParams::default(), 0, "raw", "mock", &prompt, 64).unwrap();

    let seen = AtomicUsize::new(0);
    let mgr2 = Arc::clone(&mgr);
    let g = mgr
        .generate_streaming(1, 1_000_000, 120_000, |b| {
            if seen.fetch_add(b.len(), Ordering::Relaxed) + b.len() >= 5 {
                mgr2.cancel(1).unwrap();
            }
        })
        .unwrap();
    assert_eq!(g.finish, superfluid_abi::finish::CANCELLED, "cancel must end the stream CANCELLED");
    assert!(g.tokens.len() >= 5, "some tokens should stream before the cancel lands");
    assert!(
        g.tokens.len() < 1_000_000,
        "cancel must stop the node early, not run the full budget ({} tokens)",
        g.tokens.len()
    );
}

#[test]
fn park_frees_the_node_and_resume_re_places() {
    let nodes = vec![spec(spawn_node("p0")), spec(spawn_node("p1"))];
    let mgr = FleetManager::new(nodes, PlacementPolicy::LeastLoaded, lease());
    let prompt: Vec<u32> = (5..17).collect();
    mgr.place(1, GenParams::default(), 0, "raw", "mock", &prompt, 64).unwrap();
    assert_eq!(mgr.load().iter().sum::<usize>(), 1, "one session placed");

    mgr.park(1).unwrap();
    assert_eq!(mgr.load().iter().sum::<usize>(), 0, "parked session frees node load");
    assert!(mgr.generate(1, 4, 120_000).is_err(), "a parked session must refuse generate");
    assert!(mgr.append(1, &[99]).is_err(), "a parked session must refuse append");
    mgr.park(1).unwrap();

    let node = mgr.resume(1).unwrap();
    assert_eq!(mgr.load().iter().sum::<usize>(), 1, "resume restores node load");
    assert_eq!(mgr.load()[node], 1, "resumed session sits on the returned node");
    let got = mgr.generate(1, 12, 120_000).unwrap();
    assert_eq!(got.tokens, reference(&prompt, 12), "resumed generation wrong");

    assert!(mgr.resume(1).is_err(), "resuming a live (unparked) session errors");
}

#[test]
fn streaming_generation_fails_over_when_the_owning_node_dies() {
    use std::time::Duration;
    let listener_a = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr_a = listener_a.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(mock_daemon());
        let node = superfluid_daemon::nodeagent::NodeAgent::new(daemon, "sa", "mock")
            .with_timeouts(Duration::from_secs(5), Duration::from_millis(200));
        if let Ok((sock, _)) = listener_a.accept() {
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            let _ = node.connection().serve(ep);
        }
    });
    let addr_b = spawn_node("sb");
    let mgr = FleetManager::new(
        vec![spec(addr_a), spec(addr_b)],
        PlacementPolicy::RoundRobin,
        lease(),
    );
    let prompt: Vec<u32> = (9..21).collect();
    mgr.place(1, GenParams::default(), 0, "raw", "mock", &prompt, 64).unwrap();
    assert_eq!(mgr.node_of(1).as_deref(), Some("sa"));
    std::thread::sleep(Duration::from_millis(500));

    let mut collected = Vec::new();
    let got = mgr
        .generate_streaming(1, 10, 120_000, |b| collected.extend_from_slice(b))
        .expect("streaming failover");
    assert_eq!(got.tokens, reference(&prompt, 10), "streaming failover produced wrong output");
    assert_eq!(collected, got.tokens, "callback saw a different stream than the return");
    assert_eq!(mgr.node_of(1).as_deref(), Some("sb"), "session did not move to the survivor");
}

#[test]
fn continuation_from_prefix_matches_full_generation() {
    let prompt: Vec<u32> = (3..15).collect();
    let full = reference(&prompt, 10);
    let k = 4;
    let head = reference(&prompt, k as u32);
    assert_eq!(head, full[..k], "prefix of the full run");
    let mut ctx = prompt.clone();
    ctx.extend_from_slice(&head);
    let tail = reference(&ctx, (10 - k) as u32);
    let stitched: Vec<u32> = head.iter().chain(tail.iter()).copied().collect();
    assert_eq!(stitched, full, "prefix + continuation must equal the full generation");
}

fn narrow_daemon(ctx: u32) -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-orch-narrow-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
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
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

fn spawn_narrow_node_model(identity: &'static str, ctx: u32, model: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(narrow_daemon(ctx));
        let node = superfluid_daemon::nodeagent::NodeAgent::new(daemon, identity, model);
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

fn spawn_narrow_node(identity: &'static str, ctx: u32) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let daemon = Arc::new(narrow_daemon(ctx));
        let node = superfluid_daemon::nodeagent::NodeAgent::new(daemon, identity, "mock");
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

#[test]
fn a_prompt_no_node_can_hold_is_refused_at_the_head() {
    const CTX: u32 = 64;
    let nodes = vec![spec(spawn_narrow_node("narrow-a", CTX))];
    let mgr = FleetManager::new(nodes, PlacementPolicy::LeastLoaded, lease());

    let small: Vec<u32> = (0..8).collect();
    mgr.place(1, GenParams::default(), 0, "mock", "mock", &small, 64).expect("a short prompt places");

    let overlong: Vec<u32> = (0..CTX + 32).collect();
    let err = mgr
        .place(2, GenParams::default(), 0, "mock", "mock", &overlong, 64)
        .expect_err("no node can hold this");
    match err {
        superfluid_daemon::DaemonError::StreamTooLong { len, max } => {
            assert_eq!(len, overlong.len() as u64);
            assert_eq!(max, CTX as u64, "the answer is the widest window in the fleet");
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }

    mgr.place(3, GenParams::default(), 0, "mock", "mock", &small, 64)
        .expect("the node still serves prompts that fit");
}

#[test]
fn placement_routes_around_a_node_too_narrow_for_the_prompt() {
    const NARROW: u32 = 64;
    const WIDE: u32 = 4096;
    let nodes = vec![
        spec(spawn_narrow_node("narrow-b", NARROW)),
        spec(spawn_narrow_node("wide-b", WIDE)),
    ];
    let mgr = FleetManager::new(nodes, PlacementPolicy::RoundRobin, lease());

    let small: Vec<u32> = (0..8).collect();
    mgr.place(1, GenParams::default(), 0, "mock", "mock", &small, 64).unwrap();
    mgr.place(2, GenParams::default(), 0, "mock", "mock", &small, 64).unwrap();

    let big: Vec<u32> = (0..NARROW + 32).collect();
    mgr.place(3, GenParams::default(), 0, "mock", "mock", &big, 64)
        .expect("the wide node can hold it");
    assert_eq!(
        mgr.node_of(3).as_deref(),
        Some("wide-b"),
        "placement must ROUTE to a node that fits, not discover the misfit by failing over"
    );
    let gen = mgr.generate(3, 4, 60_000).expect("and can generate on it");
    assert_eq!(gen.tokens.len(), 4);

    mgr.place(4, GenParams::default(), 0, "mock", "mock", &small, 64).unwrap();
    mgr.place(5, GenParams::default(), 0, "mock", "mock", &small, 64).unwrap();
    let landed = [mgr.node_of(4), mgr.node_of(5)];
    assert!(
        landed.iter().any(|n| n.as_deref() == Some("narrow-b")),
        "a skipped node must stay in the rotation, got {landed:?}"
    );
}

#[test]
fn a_cold_fleet_still_refuses_an_oversized_prompt_by_type() {
    const CTX: u32 = 64;
    let nodes = vec![spec(spawn_narrow_node("cold-narrow", CTX))];
    let mgr = FleetManager::new(nodes, PlacementPolicy::LeastLoaded, lease());

    let overlong: Vec<u32> = (0..CTX + 32).collect();
    let err = mgr
        .place(1, GenParams::default(), 0, "mock", "mock", &overlong, 64)
        .expect_err("no node can hold this");
    match err {
        superfluid_daemon::DaemonError::StreamTooLong { len, max } => {
            assert_eq!(len, overlong.len() as u64);
            assert_eq!(max, CTX as u64, "learned during the attempt, not guessed");
        }
        other => panic!("a cold fleet must still answer by type, got {other:?}"),
    }

    mgr.place(2, GenParams::default(), 0, "mock", "mock", &(0..8).collect::<Vec<u32>>(), 64)
        .expect("the node serves prompts that fit");
}

#[test]
fn the_quoted_limit_ignores_nodes_that_serve_another_model() {
    const NARROW: u32 = 64;
    let narrow = spawn_narrow_node_model("narrow-c", NARROW, "model-a");
    let wide = spawn_narrow_node_model("wide-c", 4096, "model-b");
    let mgr = FleetManager::new(
        vec![spec(narrow), spec(wide)],
        PlacementPolicy::RoundRobin,
        lease(),
    );

    let overlong: Vec<u32> = (0..NARROW + 32).collect();
    let err = mgr
        .place(1, GenParams::default(), 0, "mock", "model-a", &overlong, 64)
        .expect_err("no model-a node can hold this");
    match err {
        superfluid_daemon::DaemonError::StreamTooLong { max, .. } => {
            assert_eq!(
                max, NARROW as u64,
                "quoting the model-b node's 4096 would be unreachable advice"
            );
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

#[test]
fn a_session_that_outgrows_its_node_is_moved_to_one_that_fits() {
    const NARROW: u32 = 64;
    const WIDE: u32 = 4096;
    let nodes = vec![
        spec(spawn_narrow_node("narrow-d", NARROW)),
        spec(spawn_narrow_node("wide-d", WIDE)),
    ];
    let mgr = FleetManager::new(nodes, PlacementPolicy::LeastLoaded, lease());

    let small: Vec<u32> = (0..8).collect();
    mgr.place(1, GenParams::default(), 0, "mock", "mock", &small, 64).unwrap();
    let home = mgr.node_of(1);
    if home.as_deref() != Some("narrow-d") {
        mgr.place(2, GenParams::default(), 0, "mock", "mock", &small, 64).unwrap();
        assert_eq!(mgr.node_of(2).as_deref(), Some("narrow-d"), "one of them must be narrow");
    }
    let session = if home.as_deref() == Some("narrow-d") { 1 } else { 2 };

    let grown: Vec<u32> = (0..NARROW + 32).collect();
    mgr.append(session, &grown).expect("the fleet can still hold this, elsewhere");
    assert_eq!(
        mgr.node_of(session).as_deref(),
        Some("wide-d"),
        "the session must MOVE to a node that fits, not wait to be refused"
    );
    let gen = mgr.generate(session, 4, 60_000).expect("and generate there");
    assert_eq!(gen.tokens.len(), 4);
}

fn spawn_load_node(identity: &'static str, ctx: u32) -> String {
    spawn_narrow_node(identity, ctx)
}

#[test]
fn load_aware_degrades_to_session_count_when_nothing_is_reported() {
    let nodes = vec![spec(spawn_load_node("la-a", 4096)), spec(spawn_load_node("la-b", 4096))];
    let mgr = FleetManager::new(nodes, PlacementPolicy::LoadAware, lease());
    let prompt: Vec<u32> = (0..8).collect();

    for s in 1..=4u64 {
        mgr.place(s, GenParams::default(), 0, "mock", "mock", &prompt, 64).unwrap();
    }
    let mut counts = std::collections::HashMap::new();
    for s in 1..=4u64 {
        *counts.entry(mgr.node_of(s).unwrap_or_default()).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 2, "both nodes must be used: {counts:?}");
    for (node, n) in &counts {
        assert!(*n >= 1, "node {node} took {n} of 4");
    }
}

#[test]
fn a_silent_node_does_not_keep_looking_idle() {
    let live = spawn_load_node("la-live", 4096);
    let nodes = vec![spec(dead_addr()), spec(live)];
    let mgr = FleetManager::new(nodes, PlacementPolicy::LoadAware, lease());
    let prompt: Vec<u32> = (0..8).collect();

    for s in 1..=3u64 {
        mgr.place(s, GenParams::default(), 0, "mock", "mock", &prompt, 64)
            .expect("the live node serves");
        assert_eq!(
            mgr.node_of(s).as_deref(),
            Some("la-live"),
            "a node that answers nothing must never win on an empty score"
        );
    }
}

#[test]
fn load_aware_still_refuses_a_prompt_no_node_can_hold() {
    const CTX: u32 = 64;
    let nodes = vec![spec(spawn_load_node("la-narrow", CTX))];
    let mgr = FleetManager::new(nodes, PlacementPolicy::LoadAware, lease());

    let overlong: Vec<u32> = (0..CTX + 32).collect();
    let err = mgr
        .place(1, GenParams::default(), 0, "mock", "mock", &overlong, 64)
        .expect_err("the ceiling is a filter, not a preference");
    assert!(
        matches!(err, superfluid_daemon::DaemonError::StreamTooLong { .. }),
        "expected a typed refusal, got {err:?}"
    );

    mgr.place(2, GenParams::default(), 0, "mock", "mock", &(0..8).collect::<Vec<u32>>(), 64)
        .expect("a fitting prompt still places");
}
