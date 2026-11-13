//! A fleet head's nodes: listed over HTTP, and one removed until the head restarts.

use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::fleet_join::accept_nodes;
use superfluid_daemon::fleet_manager::{FleetManager, NodeSpec, PlacementPolicy, RemoveError};
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::{Daemon, DaemonError, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_linkf::{Endpoint, LinkFError, Role};
use superfluid_proto::linkf::Lease;

static DIRS: AtomicU64 = AtomicU64::new(0);
const TOKEN: &[u8] = b"a fleet token";
const LEASE: Lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };

fn dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("superfluid-fleet-nodes-{tag}-{}-{}", std::process::id(), DIRS.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A daemon whose engine takes `tick` a step.
fn mock_daemon(tick: Duration) -> Daemon {
    let store = SessionStore::open(&dir("node").join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || (MockEngine::new(EngineConfig { tick_delay: tick, ..EngineConfig::default() }), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

/// What each serve of a node's link ended with, oldest first.
type Outcomes = Arc<Mutex<Vec<Result<(), String>>>>;

fn outcome(r: Result<(), DaemonError>) -> Result<(), String> {
    match r {
        Err(DaemonError::Fleet(LinkFError::Removed)) => Err("removed".into()),
        Err(e) => Err(e.to_string()),
        Ok(()) => Ok(()),
    }
}

fn told_removed(r: &Result<(), String>) -> bool {
    r.as_ref().is_err_and(|e| e == "removed")
}

/// A node that dials `head` once per lane and, whatever a link ends with, dials again, as a
/// node that predates being told it was removed does.
fn join(head: &str, identity: &'static str, lanes: u32, tick: Duration) -> Outcomes {
    let node = NodeAgent::new(Arc::new(mock_daemon(tick)), identity, "mock").with_max_lanes(lanes).with_auth(TOKEN.to_vec());
    let outcomes: Outcomes = Arc::default();
    for _ in 0..lanes {
        let (head, mut conn, outcomes) = (head.to_string(), node.connection(), Arc::clone(&outcomes));
        std::thread::spawn(move || loop {
            let Ok(sock) = TcpStream::connect(&head) else { break };
            let Ok(ep) = Endpoint::from_tcp_secure(sock, TOKEN, Role::Responder, Duration::from_secs(5)) else { break };
            let r = outcome(conn.serve(ep));
            outcomes.lock().unwrap().push(r);
            std::thread::sleep(Duration::from_millis(50));
        });
    }
    outcomes
}

/// A node the head dials, counting the connections it took.
fn dialed(identity: &'static str) -> (String, Arc<AtomicUsize>, Outcomes) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let accepted = Arc::new(AtomicUsize::new(0));
    let outcomes: Outcomes = Arc::default();
    let (counted, told) = (Arc::clone(&accepted), Arc::clone(&outcomes));
    std::thread::spawn(move || {
        let node = NodeAgent::new(Arc::new(mock_daemon(Duration::ZERO)), identity, "mock").with_max_lanes(2);
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            counted.fetch_add(1, Ordering::SeqCst);
            let (mut conn, told) = (node.connection(), Arc::clone(&told));
            std::thread::spawn(move || {
                let r = outcome(conn.serve(Endpoint::from_tcp(sock).unwrap()));
                told.lock().unwrap().push(r);
            });
        }
    });
    (addr, accepted, outcomes)
}

/// A head over `specs` that also takes nodes dialing in; its fleet and the address nodes join.
fn head(specs: Vec<NodeSpec>, policy: PlacementPolicy) -> (Arc<FleetManager>, String) {
    let fleet = Arc::new(FleetManager::with_options(specs, policy, LEASE, 90, 0));
    let _ = fleet.probe();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let f = Arc::clone(&fleet);
    std::thread::spawn(move || accept_nodes(listener, f, TOKEN.to_vec(), vec!["mock".into()]));
    (fleet, addr)
}

/// The head's HTTP API on `bind`, over `fleet`; its address.
fn http(fleet: Arc<FleetManager>, bind: &str, cfg: superfluid_daemon::openai::ServeConfig) -> String {
    let store = SessionStore::open(&dir("head").join("wal.log")).unwrap();
    let daemon = Daemon::over_fleet(store, Box::new(MockChatCodec), fleet, "mock".into(), Default::default()).unwrap();
    serve(daemon, bind, cfg)
}

fn serve(daemon: Daemon, bind: &str, cfg: superfluid_daemon::openai::ServeConfig) -> String {
    let registry = Arc::new(superfluid_daemon::registry::ModelRegistry::single("mock", Arc::new(daemon)));
    let listener = TcpListener::bind(bind).unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || superfluid_daemon::openai::serve_blocking_config(listener, registry, cfg));
    addr
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(10)))
        .http_status_as_error(false)
        .build()
        .new_agent()
}

/// `method` on `path`, with `key` as the bearer when given: the status and the JSON body.
fn call(addr: &str, method: &str, path: &str, key: Option<&str>) -> (u16, Value) {
    let url = format!("http://{addr}{path}");
    let a = agent();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut resp = loop {
        let r = match (method, key) {
            ("GET", None) => a.get(&url).call(),
            ("GET", Some(k)) => a.get(&url).header("authorization", &format!("Bearer {k}")).call(),
            (_, None) => a.delete(&url).call(),
            (_, Some(k)) => a.delete(&url).header("authorization", &format!("Bearer {k}")).call(),
        };
        match r {
            Ok(resp) => break resp,
            Err(e) if Instant::now() < deadline => {
                let _ = e;
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => panic!("{method} {url}: {e}"),
        }
    };
    let status = resp.status().as_u16();
    let body = resp.body_mut().read_to_string().unwrap();
    (status, serde_json::from_str(&body).unwrap_or(Value::String(body)))
}

fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !ok() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn listed(fleet: &FleetManager) -> Vec<String> {
    fleet.nodes().into_iter().map(|n| n.status.name).collect()
}

#[test]
fn a_head_lists_each_node_with_how_it_is_connected_and_what_it_reported() {
    let (node_addr, _, _) = dialed("dialed-a");
    let (fleet, join_addr) = head(vec![NodeSpec { addr: node_addr.clone(), auth: Vec::new() }], PlacementPolicy::LoadAware);
    join(&join_addr, "joined-b", 2, Duration::ZERO);
    wait_for("the second node joined", || listed(&fleet).len() == 2);
    let api = http(Arc::clone(&fleet), "127.0.0.1:0", Default::default());

    let (status, body) = call(&api, "GET", "/v1/fleet/nodes", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!((body["object"].as_str(), body["schema"].as_u64()), (Some("list"), Some(1)));
    let data = body["data"].as_array().unwrap();
    let node = |name: &str| data.iter().find(|n| n["name"] == name).unwrap_or_else(|| panic!("{name} in {body}"));
    let (a, b) = (node("dialed-a"), node("joined-b"));
    assert_eq!(a["addr"], node_addr.as_str());
    assert_eq!(a["connection"], "dialed");
    assert_eq!(b["addr"], "127.0.0.1", "a joined node is listed at the address it dialed from");
    assert_eq!(b["connection"], "joined");
    for n in [a, b] {
        assert_eq!(n["object"], "fleet.node");
        assert_eq!(n["up"], true, "{n}");
        assert_eq!(n["models"], serde_json::json!(["mock"]));
        assert_eq!(n["max_lanes"], 2);
        assert!(n["max_context_tokens"].as_u64().unwrap() > 0, "{n}");
        assert_eq!(n["sessions"], 0);
        let since = n["since"].as_u64().unwrap();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        assert!(since <= now && since + 60 > now, "up since it joined: {n}");
        for field in ["lanes", "lanes_active", "queue_depth", "kv_blocks_used", "kv_blocks_total", "decode_tokens"] {
            assert!(n[field].is_u64(), "{field} in {n}");
        }
    }
}

#[test]
fn removing_a_node_answers_its_entry_and_leaves_it_out_of_the_list_metrics_and_health() {
    let (node_addr, _, _) = dialed("dialed-c");
    let (fleet, join_addr) = head(vec![NodeSpec { addr: node_addr, auth: Vec::new() }], PlacementPolicy::LoadAware);
    join(&join_addr, "joined-d", 1, Duration::ZERO);
    wait_for("the second node joined", || listed(&fleet).len() == 2);
    let api = http(Arc::clone(&fleet), "127.0.0.1:0", Default::default());

    let (status, body) = call(&api, "DELETE", "/v1/fleet/nodes/no-such-node", None);
    assert_eq!((status, body["error"]["code"].as_str()), (404, Some("node_not_found")), "{body}");

    let (status, body) = call(&api, "DELETE", "/v1/fleet/nodes/joined-d", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!((body["name"].as_str(), body["removed"].as_bool(), body["schema"].as_u64()), (Some("joined-d"), Some(true), Some(1)));
    assert_eq!(body["connection"], "joined");

    let (_, list) = call(&api, "GET", "/v1/fleet/nodes", None);
    let names: Vec<&str> = list["data"].as_array().unwrap().iter().filter_map(|n| n["name"].as_str()).collect();
    assert_eq!(names, ["dialed-c"]);
    let mut resp = agent().get(&format!("http://{api}/metrics")).call().unwrap();
    let metrics = resp.body_mut().read_to_string().unwrap();
    assert!(!metrics.contains("joined-d"), "{metrics}");
    assert!(metrics.contains("node=\"dialed-c\""), "{metrics}");
    let (_, health) = call(&api, "GET", "/health", None);
    assert_eq!((health["nodes_up"].as_u64(), health["nodes_known"].as_u64()), (Some(1), Some(1)), "{health}");

    let (status, _) = call(&api, "DELETE", "/v1/fleet/nodes/joined-d", None);
    assert_eq!(status, 404, "a node removed is no longer there to remove");
    let (status, body) = call(&api, "DELETE", &format!("/v1/fleet/nodes/{}", "127.0.0.1"), None);
    assert_eq!(status, 404, "the address a removed node joined from names nothing now: {body}");
}

#[test]
fn a_server_that_is_not_a_fleet_head_has_no_nodes() {
    let api = serve(mock_daemon(Duration::ZERO), "127.0.0.1:0", Default::default());
    for method in ["GET", "DELETE"] {
        let path = if method == "GET" { "/v1/fleet/nodes" } else { "/v1/fleet/nodes/any" };
        let (status, body) = call(&api, method, path, None);
        assert_eq!((status, body["error"]["code"].as_str()), (404, Some("not_a_fleet_head")), "{method}: {body}");
    }
}

#[test]
fn removing_a_joined_node_moves_its_session_and_refuses_the_node_when_it_dials_again() {
    let (fleet, join_addr) = head(Vec::new(), PlacementPolicy::RoundRobin);
    // Steps slow enough that the generation is still running when its node is removed.
    let removed = join(&join_addr, "box-a", 1, Duration::from_millis(40));
    wait_for("box-a joined", || listed(&fleet) == ["box-a"]);
    join(&join_addr, "box-b", 1, Duration::from_millis(5));
    wait_for("box-b joined", || listed(&fleet).len() == 2);

    let prompt: Vec<u32> = (200..232).collect();
    fleet.place(1, GenParams::default(), 0, "chatml", "mock", &prompt, 60).expect("placed");
    assert_eq!(fleet.node_of(1).as_deref(), Some("box-a"));
    assert_eq!(fleet.nodes().iter().find(|n| n.status.name == "box-a").map(|n| n.sessions), Some(1));
    let got = Arc::new(AtomicUsize::new(0));
    let run = {
        let (fleet, got) = (Arc::clone(&fleet), Arc::clone(&got));
        std::thread::spawn(move || {
            fleet.generate_streaming(1, 60, 60_000, |b| {
                got.fetch_add(b.len(), Ordering::SeqCst);
            })
        })
    };
    wait_for("tokens from box-a", || got.load(Ordering::SeqCst) > 0);
    let entry = fleet.remove("box-a").expect("box-a is listed");
    assert_eq!(entry.status.name, "box-a");
    let generated = run.join().unwrap().expect("the generation finished on another node");
    assert!(!generated.expired);
    assert_eq!(got.load(Ordering::SeqCst), 60, "every token, once");
    assert_eq!(fleet.node_of(1).as_deref(), Some("box-b"), "the session moved");
    fleet.finish(1);

    // The node is told on its link, then dials again, as a node that predates being told
    // does, and is turned away each time, told again.
    wait_for("box-a told twice", || removed.lock().unwrap().iter().filter(|r| told_removed(r)).count() >= 2);
    assert_eq!(listed(&fleet), ["box-b"], "box-a is not taken back");
    assert_eq!(fleet.remove("box-a").unwrap_err(), RemoveError::Unknown);
    let (up, known) = fleet.nodes_up();
    assert_eq!((up, known), (1, 1));
}

#[test]
fn removing_a_dialed_node_tells_it_and_the_head_dials_it_no_more() {
    let (addr, accepted, outcomes) = dialed("dialed-e");
    let fleet = Arc::new(
        FleetManager::with_options(vec![NodeSpec { addr, auth: Vec::new() }], PlacementPolicy::LoadAware, LEASE, 90, 0)
            .with_retry_every(Duration::from_millis(20)),
    );
    let _ = fleet.probe();
    fleet.place(1, GenParams::default(), 0, "chatml", "mock", &[1, 2, 3], 4).expect("placed");
    fleet.generate(1, 4, 60_000).expect("generated");
    fleet.finish(1);

    fleet.remove("dialed-e").expect("listed by its identity");
    wait_for("the node was told", || outcomes.lock().unwrap().iter().any(told_removed));
    let dials = accepted.load(Ordering::SeqCst);
    for _ in 0..10 {
        fleet.tend();
        std::thread::sleep(Duration::from_millis(30));
    }
    assert_eq!(accepted.load(Ordering::SeqCst), dials, "the head dialed a removed node");
    let e = fleet.place(2, GenParams::default(), 0, "chatml", "mock", &[1, 2, 3], 4).unwrap_err();
    assert!(matches!(e, DaemonError::NoNode(_)), "{e}");
    assert!(fleet.status().is_empty());
}

#[test]
fn removing_a_node_takes_a_key_with_admin_when_the_head_has_a_key_policy() {
    let (node_addr, _, _) = dialed("dialed-f");
    let (fleet, _) = head(vec![NodeSpec { addr: node_addr, auth: Vec::new() }], PlacementPolicy::LoadAware);
    let keys = superfluid_daemon::keypolicy::KeyTable::parse(
        r#"{"keys":[{"name":"app","key":"sk-app"},{"name":"ops","key":"sk-ops","admin":true}]}"#,
        |_| None,
    )
    .unwrap();
    let cfg = superfluid_daemon::openai::ServeConfig { keys: Some(Arc::new(keys)), ..Default::default() };
    let api = http(fleet, "127.0.0.1:0", cfg);

    assert_eq!(call(&api, "GET", "/v1/fleet/nodes", None).0, 401, "the list follows the API's auth");
    assert_eq!(call(&api, "GET", "/v1/fleet/nodes", Some("sk-app")).0, 200, "any key lists");
    let (status, body) = call(&api, "DELETE", "/v1/fleet/nodes/dialed-f", Some("sk-app"));
    assert_eq!((status, body["error"]["code"].as_str()), (403, Some("admin_required")), "{body}");
    let (status, body) = call(&api, "DELETE", "/v1/fleet/nodes/dialed-f", Some("sk-ops"));
    assert_eq!((status, body["removed"].as_bool()), (200, Some(true)), "{body}");
}

/// An address of this machine's other than loopback, when it has one with a route out.
fn own_address() -> Option<std::net::IpAddr> {
    let probe = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    probe.connect("192.0.2.1:9").ok()?;
    Some(probe.local_addr().ok()?.ip()).filter(|ip| !ip.is_loopback() && !ip.is_unspecified())
}

#[test]
fn a_head_with_no_key_takes_removal_only_from_this_machine() {
    let Some(ip) = own_address() else {
        eprintln!("skipped: this machine has no address but loopback to ask from");
        return;
    };
    let (node_addr, _, _) = dialed("dialed-g");
    let (fleet, _) = head(vec![NodeSpec { addr: node_addr, auth: Vec::new() }], PlacementPolicy::LoadAware);
    let api = http(Arc::clone(&fleet), "0.0.0.0:0", Default::default());
    let port = api.rsplit(':').next().unwrap();
    let elsewhere = format!("{ip}:{port}");

    let (status, body) = call(&elsewhere, "GET", "/v1/fleet/nodes", None);
    assert_eq!(status, 200, "listing follows the API's auth, which has no key here: {body}");
    let (status, body) = call(&elsewhere, "DELETE", "/v1/fleet/nodes/dialed-g", None);
    assert_eq!((status, body["error"]["code"].as_str()), (403, Some("admin_required")), "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("--api-key"), "{body}");
    assert_eq!(listed(&fleet), ["dialed-g"], "nothing was removed");
    let (status, _) = call(&format!("127.0.0.1:{port}"), "DELETE", "/v1/fleet/nodes/dialed-g", None);
    assert_eq!(status, 200, "from this machine it is");
}

struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn a_node_join_process_the_head_removes_exits_saying_so() {
    use std::io::Read;
    let (fleet, join_addr) = {
        // No token: the head takes nodes from this machine only, and the node dials without one.
        let fleet = Arc::new(FleetManager::with_options(Vec::new(), PlacementPolicy::LoadAware, LEASE, 90, 0));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let f = Arc::clone(&fleet);
        std::thread::spawn(move || accept_nodes(listener, f, Vec::new(), vec!["mock".into()]));
        (fleet, addr)
    };
    let home = dir("join-home");
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_superfluid"))
        .args(["node", "join", &join_addr, "--model", "mock", "--runtime", "mock", "--identity", "proc-join", "--max-batch", "2"])
        .env("HOME", &home)
        .env("SUPERFLUID_HOME", home.join(".superfluid"))
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn superfluid node join");
    let mut stderr = child.stderr.take().unwrap();
    let mut child = Child(child);
    wait_for("the node joined", || listed(&fleet) == ["proc-join"]);

    fleet.remove("proc-join").expect("listed");
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "the node is still running after its head removed it");
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut said = String::new();
    stderr.read_to_string(&mut said).unwrap();
    assert_eq!(status.code(), Some(superfluid_daemon::fleet_join::REMOVED_EXIT), "{said}");
    assert!(
        said.contains(&format!("the head at {join_addr} removed this node; it will refuse it until it restarts")),
        "{said}"
    );
}

#[test]
fn the_fleet_command_lists_a_heads_nodes_and_removes_one() {
    let (node_addr, _, _) = dialed("dialed-h");
    let (fleet, _) = head(vec![NodeSpec { addr: node_addr, auth: Vec::new() }], PlacementPolicy::LoadAware);
    let api = http(Arc::clone(&fleet), "127.0.0.1:0", Default::default());
    let home = dir("cli-home");
    let run = |args: &[&str]| {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_superfluid"))
            .arg("fleet")
            .args(args)
            .env("HOME", &home)
            .env("SUPERFLUID_HOME", home.join(".superfluid"))
            .env_remove("SUPERFLUID_API_KEY")
            .output()
            .expect("run superfluid fleet");
        (out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
    };
    // Answering once it is up.
    call(&api, "GET", "/v1/fleet/nodes", None);

    let (code, table, said) = run(&["nodes", &api]);
    assert_eq!(code, Some(0), "{said}");
    let lines: Vec<&str> = table.lines().collect();
    assert_eq!(lines.len(), 2, "{table}");
    assert!(lines[0].starts_with("NAME"), "{table}");
    let row: Vec<&str> = lines[1].split_whitespace().collect();
    assert_eq!((row[0], row[2], row[3], row[5]), ("dialed-h", "dialed", "up", "mock"), "{table}");
    let (code, json, _) = run(&["nodes", &format!("http://{api}/"), "--json"]);
    assert_eq!(code, Some(0));
    assert_eq!(serde_json::from_str::<Value>(&json).unwrap()["data"][0]["name"], "dialed-h", "{json}");

    let (code, _, said) = run(&["remove", "no-such-node", &api]);
    assert_eq!(code, Some(1));
    assert!(said.contains("no fleet node named 'no-such-node'"), "{said}");
    let (code, _, said) = run(&["remove", "dialed-h", &api]);
    assert_eq!(code, Some(0), "{said}");
    assert!(said.contains("fleet node 'dialed-h' removed"), "{said}");
    assert!(listed(&fleet).is_empty());
    let (code, _, said) = run(&["nodes", "127.0.0.1:1"]);
    assert_eq!(code, Some(1));
    assert!(said.contains("no fleet head answers at 127.0.0.1:1"), "{said}");
}
