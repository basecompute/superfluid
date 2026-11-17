//! A node with no `--model` follows only what a head states over a link the token
//! authenticates: a peer without the token, a forged one, or a head that states nothing gets
//! nothing pulled, and a real head's models reach the node in its hello.
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use superfluid_daemon::fleet::FleetHead;
use superfluid_daemon::fleet_join::{accept_nodes, ask_head};
use superfluid_daemon::fleet_manager::{FleetManager, PlacementPolicy};
use superfluid_daemon::fleet_models::HeadModels;
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::{Daemon, EngineHost, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_linkf::{Endpoint, LinkFError, Role};
use superfluid_proto::handshake::{Hello, LinkRole, Served};
use superfluid_proto::linkf::{msg_type, Lease};

static DIRS: AtomicU64 = AtomicU64::new(0);
const TOKEN: &[u8] = b"the fleet's token";

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("superfluid-follow-{tag}-{}-{}", std::process::id(), DIRS.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn served(models: &[&str], ids: &[&str]) -> Served {
    Served { models: models.iter().map(|m| m.to_string()).collect(), ids: ids.iter().map(|m| m.to_string()).collect() }
}

/// A head with `token` that serves `serves`, taking nodes that dial it.
fn head(token: &[u8], serves: &Served) -> (String, Arc<FleetManager>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let manager = Arc::new(FleetManager::with_options(Vec::new(), PlacementPolicy::LoadAware, lease, 90, 1));
    let models = HeadModels::fixed(Arc::clone(&manager), serves.models.clone());
    models.know_ids(&serves.models, &serves.ids);
    let (m, token) = (Arc::clone(&manager), token.to_vec());
    std::thread::spawn(move || accept_nodes(listener, m, token, models));
    (addr, manager)
}

/// A head released before heads stated what they serve: its hello states nothing.
fn head_from_before(token: &[u8]) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let token = token.to_vec();
    std::thread::spawn(move || {
        for sock in listener.incoming().flatten() {
            let token = token.clone();
            std::thread::spawn(move || {
                let _ = FleetHead::accept(sock, token);
            });
        }
    });
    addr
}

/// A peer with no token that speaks first, in the clear, a hello naming a model of its choosing.
fn forged_head() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        for sock in listener.incoming().flatten() {
            std::thread::spawn(move || {
                let Ok(mut ep) = Endpoint::from_tcp(sock) else { return };
                let hello = Hello {
                    proto_versions: (superfluid_linkf::driver::PROTO_VERSION, superfluid_linkf::driver::PROTO_VERSION),
                    link_role: LinkRole::HeadDaemon,
                    auth: TOKEN.to_vec(),
                    serves: Some(served(&["review/attacker-selected-model"], &["review/attacker-selected-model:x"])),
                };
                let _ = ep.send(msg_type::HELLO, &hello);
                std::thread::sleep(Duration::from_secs(5));
            });
        }
    });
    addr
}

fn addrs(addr: &str) -> Vec<std::net::SocketAddr> {
    vec![addr.parse().unwrap()]
}

#[test]
fn a_node_learns_what_a_head_serves_from_its_hello_and_is_not_taken_in_for_asking() {
    let serves = served(&["review/real-model", "local-model"], &["review/real-model:q4"]);
    let (addr, manager) = head(TOKEN, &serves);
    let stated = ask_head(&addrs(&addr), TOKEN, "asker").unwrap().expect("a head of this release states what it serves");
    assert_eq!(stated, served(&["review/real-model", "local-model"], &["review/real-model:q4", ""]));
    std::thread::sleep(Duration::from_millis(300));
    assert!(manager.status().is_empty(), "a node that only asked is not one of the head's");
}

#[test]
fn a_lane_hears_what_the_head_serves_at_each_handshake() {
    let serves = served(&["review/real-model"], &["review/real-model:q4"]);
    let (addr, manager) = head(TOKEN, &serves);
    let dir = scratch("lane");
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    let daemon = Arc::new(Daemon::new(SessionStore::open(&dir.join("wal.log")).unwrap(), host, Box::new(MockCodec), 2));
    let heard: Arc<Mutex<Vec<Option<Served>>>> = Arc::default();
    let h = Arc::clone(&heard);
    let node = NodeAgent::new(daemon, "follower", "review/real-model")
        .with_max_lanes(1)
        .with_auth(TOKEN.to_vec())
        .following()
        .with_on_head(move |s| h.lock().unwrap().push(s.cloned()));
    let mut conn = node.connection();
    std::thread::spawn(move || {
        let sock = TcpStream::connect(&addr).unwrap();
        let ep = Endpoint::from_tcp_secure(sock, TOKEN, Role::Responder, Duration::from_secs(5)).unwrap();
        let _ = conn.serve(ep);
    });
    let by = Instant::now() + Duration::from_secs(10);
    while heard.lock().unwrap().is_empty() || manager.status().is_empty() {
        assert!(Instant::now() < by, "the lane joined and heard the head");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(heard.lock().unwrap()[0], Some(served(&["review/real-model"], &["review/real-model:q4"])));
}

#[test]
fn a_peer_without_the_token_or_a_head_from_before_states_nothing_a_node_acts_on() {
    let (other, _) = head(b"another network's token", &served(&["review/attacker-selected-model"], &["review/attacker-selected-model:x"]));
    assert!(matches!(ask_head(&addrs(&other), TOKEN, "asker"), Err(LinkFError::Unauthorized)));
    assert!(ask_head(&addrs(&forged_head()), TOKEN, "asker").is_err(), "a peer that cannot run the token's handshake");
    assert_eq!(ask_head(&addrs(&head_from_before(TOKEN)), TOKEN, "asker").unwrap(), None);
}

/// A basert runtime whose worker records each pull in `pulls` and fails it.
fn recording_runtime(dir: &Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let v = superfluid_agent::client::PROTO_VERSION;
    let bin = dir.join("superfluid-worker-basert");
    let script = format!(
        r#"#!/bin/sh
case "$1" in
  check) echo '{{"runtime":{{"id":"basert","version":"1"}},"link_w":[{v}],"available":true,"formats":[{{"id":"toy","describe":"a toy file","file":{{"extensions":["toy"]}}}}],"tokenizer":{{"from":"huggingface"}},"pull":{{"options":[]}}}}' ;;
  pull) shift; echo "$@" >> "{pulls}"; exit 1 ;;
esac
"#,
        pulls = dir.join("pulls").display()
    );
    std::fs::write(&bin, script).unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    bin
}

/// `superfluid node join` with no `--model`, at `head` (or found on the local network), run
/// for at most `wait`: its exit status if it ended, its stderr, and the pulls it asked for.
fn follow(head: Option<&str>, wait: Duration) -> (Option<std::process::ExitStatus>, String, Option<String>) {
    let dir = scratch("node");
    let runtime = recording_runtime(&dir);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_superfluid"));
    cmd.args(["node", "join"]).args(head).args(["--token", std::str::from_utf8(TOKEN).unwrap(), "--identity", "follower", "--offline"]);
    cmd.env("SUPERFLUID_HOME", dir.join("home")).env("SUPERFLUID_WORKER_BASERT", &runtime).env_remove("RUST_LOG");
    let log = std::fs::File::create(dir.join("stderr")).unwrap();
    let mut child = cmd.stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let by = Instant::now() + wait;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if Instant::now() >= by {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let stderr = std::fs::read_to_string(dir.join("stderr")).unwrap_or_default();
    (status, stderr, std::fs::read_to_string(dir.join("pulls")).ok())
}

#[test]
fn nothing_is_pulled_on_the_word_of_a_peer_without_the_token_or_a_head_that_states_nothing() {
    let (other, _) = head(b"another network's token", &served(&["review/attacker-selected-model"], &["review/attacker-selected-model:x"]));
    let (status, stderr, pulls) = follow(Some(&other), Duration::from_secs(20));
    assert_eq!(pulls, None, "{stderr}");
    assert_eq!(status.and_then(|s| s.code()), Some(2), "refused: {stderr}");
    assert!(stderr.contains("refused this node's token"), "{stderr}");

    let (_, stderr, pulls) = follow(Some(&forged_head()), Duration::from_secs(3));
    assert_eq!(pulls, None, "a forged peer's hello is never read as a head's: {stderr}");

    let (status, stderr, pulls) = follow(Some(&head_from_before(TOKEN)), Duration::from_secs(20));
    assert_eq!(pulls, None, "{stderr}");
    assert_eq!(status.and_then(|s| s.code()), Some(2), "{stderr}");
    assert!(stderr.contains("states nothing over the link") && stderr.contains("give this node --model"), "{stderr}");
}

#[test]
fn a_node_pulls_the_model_a_head_states_over_the_link() {
    let (addr, _) = head(TOKEN, &served(&["review/real-model"], &["review/real-model:q4"]));
    let (status, stderr, pulls) = follow(Some(&addr), Duration::from_secs(20));
    assert!(status.is_some(), "the recording runtime fails the pull, which ends the node: {stderr}");
    assert_eq!(pulls.as_deref().map(str::trim), Some("review/real-model:q4 --offline"), "{stderr}");
}

#[test]
#[ignore = "needs multicast on an interface: cargo test -p superfluid --test fleet_follow -- --ignored"]
fn nothing_is_pulled_on_a_forged_announcement_of_this_network() {
    // Anyone on the network can copy a head's `net` and announce any model under it, with the
    // `ids` property heads announced before they stated their models over the link.
    let port = TcpListener::bind("0.0.0.0:0").unwrap().local_addr().unwrap().port();
    let net = superfluid_daemon::fleet_mdns::network_id(TOKEN);
    let mdns = mdns_sd::ServiceDaemon::new().unwrap();
    mdns.set_service_name_len_max(16).unwrap();
    let attacker = "review/attacker-selected-model";
    let props = [("model", attacker), ("models", attacker), ("ids", "review/attacker-selected-model:x"), ("net", net.as_str())];
    let forged = mdns_sd::ServiceInfo::new(superfluid_daemon::fleet_mdns::SERVICE, "superfluid forged", "forged-host.local.", (), port, &props[..])
        .unwrap()
        .enable_addr_auto();
    mdns.register(forged).unwrap();
    let (_, stderr, pulls) = follow(None, Duration::from_secs(8));
    let _ = mdns.shutdown();
    assert_eq!(pulls, None, "{stderr}");
    assert!(stderr.contains("found the head") && stderr.contains(attacker), "the forged announcement was heard: {stderr}");
}
