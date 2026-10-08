//! `superfluid node join`: this machine serves a model to a fleet head it dials, and the head
//! takes nodes that dial in.

use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use superfluid_linkf::{Endpoint, LinkFError, Role};

use crate::fleet::FleetHead;
use crate::fleet_manager::FleetManager;
use crate::nodeagent::NodeAgent;
use crate::runtime_pick::RuntimeId;
use crate::runtimes::Catalog;
use crate::{Daemon, EngineHost, NoCodec, SessionStore};

pub const HELP: &str = "Serve a model to a fleet head from this machine.
Usage: superfluid node join [<head-host:port>] --model <path|org/model[:tag]> [options]
  Dials the head (found on the local network when no address is given), serves the
  model to it, and dials again whenever the link drops. The head's token, from
  `superfluid fleet token` there, authenticates both ends and encrypts the link.
Options:
  --model <m>             The model; an id is pulled first, as `serve` pulls it
  --token <value>         The head's fleet token, kept at $SUPERFLUID_HOME/fleet/token
  --runtime <id>          The runtime, when the model's format does not decide it
  --max-context <N>       This node's context window [default: sized for this device]
  --max-batch <N>         This node's lanes [default: 8]
  --identity <name>       The name the head shows for this node [default: the host name]
  --model-name <id>       The name the head matches on [default: what the head announces,
                          else the model's id]
  --offline               Pull nothing; the model must be in the runtime's cache
";

const CONNECT: Duration = Duration::from_secs(5);
const HANDSHAKE: Duration = Duration::from_secs(10);
const DISCOVER: Duration = Duration::from_secs(30);
const RETRY_MAX: Duration = Duration::from_secs(30);
const REFUSED: &str = "the head refused this node's token; copy the head's `superfluid fleet token` here with --token";
/// Peers in the handshake at once; more wait in the backlog, so a burst of connects cannot
/// take a thread each.
const HANDSHAKES_AT_ONCE: usize = 8;

pub fn run(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("join") => join(&args[1..]),
        _ => {
            eprint!("{HELP}");
            std::process::exit(2);
        }
    }
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("superfluid: {msg}");
    std::process::exit(2);
}

fn join(args: &[String]) {
    let (mut head, mut model, mut token, mut runtime, mut identity, mut model_name) = (None, None, None, None, None, None);
    let (mut max_context, mut max_batch, mut offline) = (None, 8u32, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_else(|| fail(format!("{a} takes a value")));
        match a.as_str() {
            "--model" => model = Some(val()),
            "--token" => token = Some(val()),
            "--runtime" => runtime = Some(val()),
            "--identity" => identity = Some(val()),
            "--model-name" => model_name = Some(val()),
            "--max-context" => max_context = Some(val().parse().unwrap_or_else(|_| fail("--max-context takes a number"))),
            "--max-batch" => max_batch = val().parse().unwrap_or_else(|_| fail("--max-batch takes a number")),
            "--offline" => offline = true,
            "-h" | "--help" => {
                print!("{HELP}");
                return;
            }
            s if s.starts_with('-') => fail(format!("unknown flag {s}")),
            s if head.is_none() => head = Some(s.to_string()),
            s => fail(format!("unexpected argument {s}")),
        }
    }
    let Some(model) = model else { fail("--model is required") };
    let home = crate::runtimes::default_home();
    if let Some(t) = &token {
        crate::fleet_token::write(&home, t.trim().as_bytes()).unwrap_or_else(|e| fail(e));
    }
    let token = crate::fleet_token::read(&home).unwrap_or_default();
    let (target, announced) = match head {
        Some(h) => (Target { given: Some(h), found: Mutex::new(Vec::new()) }, None),
        None => {
            let (addrs, served) = discover();
            (Target { given: None, found: Mutex::new(addrs) }, Some(served))
        }
    };
    let addrs = target.addrs();
    if addrs.is_empty() {
        fail(format!("cannot resolve {target}"));
    }
    if token.is_empty() && !addrs.iter().all(|a| a.ip().is_loopback()) {
        fail("no fleet token: run `superfluid fleet token` on the head and pass --token <value> here");
    }
    let identity = identity.unwrap_or_else(crate::metrics::hostname);
    let max_batch = max_batch.max(1);
    let loaded = load_node(&model, runtime.as_deref(), max_context, max_batch, offline, &node_wal(&home, &identity))
        .unwrap_or_else(|e| fail(e));
    eprintln!("superfluid: {}", loaded.summary);
    let model_name = model_name.or(announced.filter(|m| !m.is_empty())).unwrap_or(loaded.model_name);
    let node = NodeAgent::new(Arc::new(loaded.daemon), identity.clone(), model_name.clone())
        .with_max_lanes(max_batch)
        .with_auth(token.clone());
    eprintln!("superfluid: node '{identity}' serves {model_name}; dialing the head at {target} once per lane ({max_batch})");
    // One connection per lane, each its own generation at a time on the head.
    let target = Arc::new(target);
    let links: Vec<_> = (0..max_batch)
        .map(|lane| {
            let (target, token, node) = (Arc::clone(&target), token.clone(), node.connection());
            std::thread::spawn(move || keep_dialing(&target, &token, node, lane == 0))
        })
        .collect();
    for link in links {
        let _ = link.join();
    }
}

/// Where the head is: an address given on the command line, resolved at each dial, or the
/// addresses the local network announced, looked for again when none of them answers.
struct Target {
    given: Option<String>,
    found: Mutex<Vec<SocketAddr>>,
}

impl Target {
    fn addrs(&self) -> Vec<SocketAddr> {
        match &self.given {
            Some(h) => h.to_socket_addrs().map(|a| a.collect()).unwrap_or_default(),
            None => self.found.lock().expect("head addrs").clone(),
        }
    }

    /// After every address in `tried` failed: looks on the network again, unless the head was
    /// given or another lane already found it anew. Lanes wait on the one that is looking.
    fn refind(&self, tried: &[SocketAddr]) {
        if self.given.is_some() {
            return;
        }
        let mut found = self.found.lock().expect("head addrs");
        if *found == tried {
            if let Ok((addrs, _)) = crate::fleet_mdns::find_head(DISCOVER) {
                *found = addrs;
            }
        }
    }
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.given {
            Some(h) => f.write_str(h),
            None => match self.found.lock().expect("head addrs").first() {
                Some(a) => write!(f, "{a}"),
                None => f.write_str("the fleet head"),
            },
        }
    }
}

/// The head on the local network, however long it takes to appear.
fn discover() -> (Vec<SocketAddr>, String) {
    eprintln!("superfluid: looking for a fleet head on the local network");
    loop {
        match crate::fleet_mdns::find_head(DISCOVER) {
            Ok((addrs, served)) => {
                eprintln!("superfluid: found the head at {}, serving {served}", addrs[0]);
                return (addrs, served);
            }
            Err(e) => eprintln!("superfluid: {e}; looking again"),
        }
    }
}

/// Dials the head and serves until the link ends, then dials again; the first lane speaks
/// for the node in the log.
fn keep_dialing(target: &Target, token: &[u8], mut conn: NodeAgent, first: bool) {
    let say = |msg: String| {
        if first {
            eprintln!("superfluid: {msg}");
        }
    };
    let mut backoff = Duration::from_secs(2);
    loop {
        let addrs = target.addrs();
        match dial(&addrs, token) {
            Ok(ep) => {
                say(format!("joined the head at {target}"));
                backoff = Duration::from_secs(2);
                match conn.serve(ep) {
                    Ok(()) => say("the head closed the link".into()),
                    Err(crate::DaemonError::Fleet(LinkFError::Unauthorized)) => fail(REFUSED),
                    Err(e) => say(format!("the link to the head ended: {e}")),
                }
            }
            Err(LinkFError::Unauthorized) => fail(REFUSED),
            Err(e) => {
                say(format!("cannot join the head at {target}: {e}"));
                target.refind(&addrs);
            }
        }
        say(format!("dialing again in {} s", backoff.as_secs()));
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(RETRY_MAX);
    }
}

pub struct LoadedNode {
    pub daemon: Daemon,
    /// The name the head sees the model under.
    pub model_name: String,
    /// The runtime, model, window and lanes, for the node's start line.
    pub summary: String,
}

/// A fleet node's model by path or id, as `superfluid serve` loads one: pulled when given by
/// id, on its runtime (installed first if it never was), in a window sized for this device
/// unless `max_context` pins it.
pub fn load_node(
    model: &str,
    runtime: Option<&str>,
    max_context: Option<i32>,
    lanes: u32,
    offline: bool,
    wal: &Path,
) -> Result<LoadedNode, String> {
    let lanes = lanes.max(1);
    let catalog = Catalog::from_env();
    let runtime = runtime.map(|r| if r == "native" { "basert" } else { r });
    let (path, model_name, id) = resolve_model(&catalog, model, runtime, offline)?;
    let runtime = catalog.require(id).or_else(|_| crate::pulls::ensure_installed(&catalog, id, offline))?;
    let (ctx, sized) = match max_context {
        Some(n) => (n, format!("--max-context {n}")),
        None => match device_window(&runtime, &path, lanes) {
            Ok(n) => (n, "sized for this device".to_string()),
            Err(why) => (8192, format!("{why}; --max-context N pins it")),
        },
    };
    let ctx = ctx.max(512);
    let host = if runtime.worker.bin.is_file() {
        EngineHost::spawn_process(crate::WorkerSpec {
            bin: runtime.worker.bin.clone(),
            args: runtime.serve_args(&path, ctx, lanes, 0),
            stderr: None,
        })
    } else {
        let rt = crate::linked::engine(runtime.id.name())
            .ok_or_else(|| format!("the {} runtime has no worker here", runtime.id.name()))?;
        let args = superfluid_adapter_kit::WorkerArgs {
            model: Some(path.clone()),
            max_context: ctx,
            max_batch: lanes,
            ..Default::default()
        };
        EngineHost::try_spawn_loaded(move || rt.open(&args))
    }
    .map_err(|e| format!("model load failed: {e}"))?;
    let summary = format!(
        "runtime {} serves {}: context window {ctx} tokens ({sized}), {lanes} lanes",
        runtime.id.name(),
        path.display()
    );
    loaded(host, Box::new(NoCodec), lanes, model_name, summary, wal)
}

/// A node on the test engine, which needs no model.
pub fn load_mock_node(max_context: Option<i32>, lanes: u32, wal: &Path) -> Result<LoadedNode, String> {
    let lanes = lanes.max(1);
    let engine = crate::linked::engine("mock").ok_or("this build does not link the mock engine")?;
    let args = superfluid_adapter_kit::WorkerArgs {
        max_context: max_context.unwrap_or(4096).max(512),
        max_batch: lanes,
        ..Default::default()
    };
    let host = EngineHost::try_spawn_loaded(move || engine.open(&args)).map_err(|e| format!("model load failed: {e}"))?;
    loaded(host, Box::new(crate::MockCodec), lanes, "mock".into(), "the mock engine".into(), wal)
}

fn loaded(
    host: EngineHost,
    codec: Box<dyn crate::TextCodec + Send + Sync>,
    lanes: u32,
    model_name: String,
    summary: String,
    wal: &Path,
) -> Result<LoadedNode, String> {
    let store = SessionStore::open(wal).map_err(|e| format!("node store: {e}"))?;
    Ok(LoadedNode { daemon: Daemon::new(store, host, codec, lanes as usize), model_name, summary })
}

/// The window this device holds for `lanes` lanes of the model, as `superfluid serve` sizes it.
fn device_window(runtime: &crate::runtimes::Runtime, model: &std::path::Path, lanes: u32) -> Result<i32, String> {
    let id = runtime.id;
    if crate::linked::sizes_context(id) {
        return crate::linked::suggest_max_context(id, &[model.to_path_buf()], &[], lanes as i32, 0)?
            .ok_or_else(|| format!("the {} runtime could not size a window for this device", id.base().name()));
    }
    let sizing = crate::runtimes::sizing_of(runtime, model)?;
    let trained = sizing.trained_context;
    let per_lane = superfluid_adapter_kit::sizing::window(&[sizing], lanes)
        .ok_or_else(|| format!("the {} runtime reported no device budget to size a window against", id.base().name()))?;
    let whole = per_lane.saturating_mul(u64::from(lanes)).min(trained.max(per_lane));
    Ok(i32::try_from(whole).unwrap_or(i32::MAX))
}

/// The node's end of the link: it dials the first address that answers, and the head still
/// opens the handshake.
fn dial(addrs: &[SocketAddr], token: &[u8]) -> Result<Endpoint, LinkFError> {
    let mut last = None;
    for addr in addrs {
        match TcpStream::connect_timeout(addr, CONNECT) {
            Ok(sock) => {
                return if token.is_empty() {
                    Endpoint::from_tcp(sock)
                } else {
                    Endpoint::from_tcp_secure(sock, token, Role::Responder, HANDSHAKE)
                };
            }
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "address resolved to nothing")).into())
}

pub fn loopback(addr: &str) -> bool {
    addr.starts_with("127.") || addr.starts_with("localhost:") || addr.starts_with("[::1]:")
}

/// A session store under the system temp dir, for `superfluid-noded`.
pub fn scratch_wal() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("superfluid-node-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(format!("{}: {e}", dir.display())));
    dir.join("wal.log")
}

/// The node's session store, under `$SUPERFLUID_HOME/fleet/nodes/<identity>`; a node has no
/// state worth keeping across runs, so it starts empty.
fn node_wal(home: &Path, identity: &str) -> PathBuf {
    let dir = home.join("fleet").join("nodes").join(identity.replace(['/', '\\'], "_"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(format!("{}: {e}", dir.display())));
    dir.join("wal.log")
}

/// The model's path (pulled when given by id), the name the head sees it under, and its runtime.
fn resolve_model(catalog: &Catalog, model: &str, runtime: Option<&str>, offline: bool) -> Result<(PathBuf, String, RuntimeId), String> {
    let named = |name: &str| {
        catalog.named(name).ok().or_else(|| RuntimeId::parse(name)).ok_or_else(|| format!("--runtime '{name}' is not a runtime id"))
    };
    if crate::pulls::looks_like_id(model) {
        let id = match runtime {
            Some(name) => named(name)?,
            None => RuntimeId::new(crate::cli::runtime_for_id(model)),
        };
        let flags = crate::pulls::PullFlags { offline, options: Vec::new() };
        let path = crate::pulls::pull(catalog, id, model, &flags)?;
        return Ok((path, crate::pulls::served_name(model), id));
    }
    let path = PathBuf::from(model);
    if !path.exists() {
        return Err(format!("{}: no such file or directory", path.display()));
    }
    let id = match runtime {
        Some(name) => named(name)?,
        None => catalog.reader_of(&path)?,
    };
    Ok((path.clone(), crate::registry::model_id_for(&path), id))
}

/// The head's end: every node that dials in with the token is taken into the fleet. With no
/// token, only nodes on this machine are; `model` is what the head serves, and a node that
/// serves something else is taken but warned about, since no session will go to it.
pub fn accept_nodes(listener: TcpListener, manager: Arc<FleetManager>, auth: Vec<u8>, model: String) {
    let in_handshake = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    for sock in listener.incoming() {
        let Ok(sock) = sock else { continue };
        let Ok(peer) = sock.peer_addr() else { continue };
        if auth.is_empty() && !peer.ip().is_loopback() {
            eprintln!("superfluid: a node at {peer} was refused: this head has no fleet token (`superfluid fleet token` makes one)");
            continue;
        }
        while in_handshake.load(std::sync::atomic::Ordering::Acquire) >= HANDSHAKES_AT_ONCE {
            std::thread::sleep(Duration::from_millis(20));
        }
        in_handshake.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let (manager, auth, model, in_handshake) = (Arc::clone(&manager), auth.clone(), model.clone(), Arc::clone(&in_handshake));
        std::thread::spawn(move || {
            let opened = FleetHead::accept(sock, auth);
            in_handshake.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            match opened {
                Ok(head) => {
                    let node = head.node();
                    let (identity, models, ctx) = (node.node_identity.clone(), node.loadable_models.join(", "), node.capacity.max_context_tokens);
                    let serves_it = node.loadable_models.contains(&model);
                    match manager.adopt(head, peer.ip()) {
                        Ok((_, true, _)) => {
                            eprintln!("superfluid: fleet node '{identity}' joined from {peer}: {models}, context window {ctx} tokens");
                            if !serves_it {
                                eprintln!("superfluid: fleet node '{identity}' serves {models}, not {model}; no session will go to it (set --model-name {model} there)");
                            }
                        }
                        Ok((_, false, 1)) => eprintln!("superfluid: fleet node '{identity}' rejoined from {peer}"),
                        Ok(_) => {}
                        Err(why) => eprintln!("superfluid: a node at {peer} did not join: {why}"),
                    }
                }
                Err(e) => eprintln!("superfluid: a node at {peer} did not join: {e}"),
            }
        });
    }
}
