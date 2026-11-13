//! `superfluid node join`: this machine serves a model to a fleet head it dials, and the head
//! takes nodes that dial in.

use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use superfluid_linkf::{Endpoint, LinkFError, Role};

use crate::fleet::FleetHead;
use crate::fleet_manager::FleetManager;
use crate::fleet_mdns::Search;
use crate::fleet_models::HeadModels;
use crate::nodeagent::NodeAgent;
use crate::runtime_pick::RuntimeId;
use crate::runtimes::Catalog;
use crate::{Daemon, EngineHost, NoCodec, SessionStore};

pub const HELP: &str = "Serve a model to a fleet head from this machine.
Usage: superfluid node join [<head-host:port>] [--model <path|org/model[:tag]>] [options]
  Dials the head (found on the local network when no address is given), serves the
  model to it, and dials again whenever the link drops. The head's token, from
  `superfluid fleet token` there, authenticates both ends and encrypts the link.
  A head found on the local network must be of this token's network: a head of another
  network is never joined, and one that announces none (released before network ids)
  only when no head of this network answers. When the head is lost, the node looks
  again and joins whichever head of its network answers, which may be another machine
  that took over; a head given by address is dialed again there, never looked for.
  With no --model, the node serves the model its network's head serves first, pulled
  by the id the head announces and served under the head's name, and follows it: when
  the head serves another model first (a head that took over, or one restarted with
  another model), the node lets what it runs finish, closes its links and engine, and
  starts again with the same arguments to serve the new one. A head that announces no
  id for its model (one it has by path) is waited on until it does. Such a node finds
  its head on the local network, so it takes no head address.
  With --model, the model never changes: the node joins any head of its network, and a
  head that does not serve the model takes it on.
  A model given by id is served under its own name, org/model without the tag, and a
  head that does not serve it takes it on when the node joins. A model given by path
  is served under its file's name when the head serves a model by that name, and
  otherwise under the head's one model when it serves one, as the same model under
  another file name; --model-name names it otherwise.
Options:
  --model <m>             The model, fixed; an id is pulled first, as `serve` pulls it
                          [default: the head's first model, followed]
  --token <value>         The head's fleet token, kept at $SUPERFLUID_HOME/fleet/token
  --runtime <id>          The runtime, when the model's format does not decide it
  --max-context <N>       This node's context window [default: sized for this device]
  --max-batch <N>         This node's lanes [default: 8]
  --identity <name>       The name the head shows for this node [default: the host name]
  --model-name <id>       The name the head matches on [default: an id's own name; a
                          path's own name, or the head's one model when it serves one]
  --offline               Pull nothing; the model must be in the runtime's cache
  --park                  Keep each prompt's prefix on disk, under
                          $SUPERFLUID_HOME/fleet/nodes/<identity>/park, so a restarted node
                          starts the head's sessions warm
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
    // The node-agent's warnings (a fenced session, a silent head) go to stderr, as noded's do.
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".to_string());
    if let Err(e) = crate::telemetry::init_stderr(&filter) {
        eprintln!("superfluid: logging: {e}");
    }
    let (mut head, mut model, mut token, mut runtime, mut identity, mut model_name) = (None, None, None, None, None, None);
    let (mut max_context, mut max_batch, mut offline, mut park) = (None, 8u32, false, false);
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
            "--park" => park = true,
            "-h" | "--help" => {
                print!("{HELP}");
                return;
            }
            s if s.starts_with('-') => fail(format!("unknown flag {s}")),
            s if head.is_none() => head = Some(s.to_string()),
            s => fail(format!("unexpected argument {s}")),
        }
    }
    if model.is_none() && head.is_some() {
        fail("a node with no --model serves the model its network's head announces on the local network; give --model to join a head by address");
    }
    if model.is_none() && model_name.is_some() {
        fail("--model-name goes with --model; a node with no --model serves the head's model under the head's name");
    }
    let home = crate::runtimes::default_home();
    if let Some(t) = &token {
        crate::fleet_token::write(&home, t.trim().as_bytes()).unwrap_or_else(|e| fail(e));
    }
    let token = crate::fleet_token::read(&home).unwrap_or_default();
    let (events, heard) = std::sync::mpsc::sync_channel::<Event>(8);
    let mut search = Search::for_token(&token);
    let (target, announced, follows) = match (head, &model) {
        (Some(h), _) => (Target::new(Some(h), Vec::new(), search, None), None, None),
        (None, Some(_)) => {
            let found = discover(&mut search);
            (Target::new(None, found.nodes, search, None), Some(found.models), None)
        }
        (None, None) => {
            let (found, serving) = discover_model(&mut search);
            let follower = Follower { serving: serving.clone(), events: events.clone() };
            (Target::new(None, found.nodes, search, Some(follower)), None, Some(serving))
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
    let state = NodeState::open(&home, &identity, park).unwrap_or_else(|e| fail(e));
    let source = match (&follows, &model) {
        (Some(serving), _) => serving.id.clone(),
        (None, Some(m)) => m.clone(),
        (None, None) => unreachable!("a node with no --model follows its head"),
    };
    let loaded = load_node(&source, runtime.as_deref(), max_context, max_batch, offline, &state).unwrap_or_else(|e| fail(e));
    eprintln!("superfluid: {}", loaded.summary);
    let model_name = match (&follows, model_name) {
        (Some(serving), _) => serving.name.clone(),
        (None, Some(name)) => name,
        (None, None) => name_among(announced.as_deref().unwrap_or_default(), loaded.model_name, loaded.id.is_some()),
    };
    let daemon = Arc::new(loaded.daemon);
    let mut node = NodeAgent::new(Arc::clone(&daemon), identity.clone(), model_name.clone()).with_max_lanes(max_batch);
    if let Some(tok) = loaded.tokenizer {
        node = node.with_tokenizer_of(tok);
    }
    if let Some(id) = loaded.id {
        node = node.with_model_id(id);
    }
    if follows.is_some() {
        node = node.following();
    }
    // A head that keeps a connection for short calls asks for it; one connection more, once.
    let ask = events.clone();
    let node = node.with_auth(token.clone()).with_on_control_ask(move || {
        let _ = ask.try_send(Event::ControlAsk);
    });
    let follow_note = if follows.is_some() { ", following the head's model" } else { "" };
    eprintln!("superfluid: node '{identity}' serves {model_name}{follow_note}; dialing the head at {target} once per lane ({max_batch})");
    // One connection per lane, each its own generation at a time on the head.
    let (target, links) = (Arc::new(target), Arc::new(Links::default()));
    if follows.is_some() {
        watch_head(Arc::clone(&target));
    }
    let dial = |lane: u32| {
        let (target, token, node, links) = (Arc::clone(&target), token.clone(), node.connection(), Arc::clone(&links));
        std::thread::spawn(move || keep_dialing(&target, &token, node, lane, &links))
    };
    let mut lanes: Vec<_> = (0..max_batch).map(dial).collect();
    let mut asked = false;
    let followed = loop {
        match heard.recv_timeout(Duration::from_secs(1)) {
            Ok(Event::ControlAsk) if !asked => {
                asked = true;
                lanes.push(dial(max_batch));
            }
            Ok(Event::ControlAsk) => {}
            Ok(Event::Follow(wanted)) => break Some(wanted),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) if lanes.iter().all(|l| l.is_finished()) => break None,
            Err(_) => {}
        }
    };
    let Some(wanted) = followed else { return };
    eprintln!("superfluid: {}", wanted.change_from(&model_name));
    // What the node runs ends, then its links, then its engine; it starts again as it was
    // started, and pulls and loads the head's model as it did at first.
    let by = std::time::Instant::now() + DRAIN;
    while busy(&daemon) && std::time::Instant::now() < by {
        std::thread::sleep(Duration::from_millis(200));
    }
    links.stop();
    for lane in lanes {
        let _ = lane.join();
    }
    drop(node);
    daemon.signal_shutdown();
    daemon.shutdown();
    restart();
}

/// What the node's main thread is told while its lanes run.
enum Event {
    /// The head asked for a connection for short calls.
    ControlAsk,
    /// The head this node follows serves another model now.
    Follow(Follow),
}

/// What a node that follows its head serves: the head's first model, by the id it announces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Serving {
    /// The name the head serves the model under, which the node serves it under too.
    pub name: String,
    /// The id the node pulls it by, `org/model:tag`.
    pub id: String,
}

/// What a node that follows its head makes of the head's announcement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Follow {
    Serve(Serving),
    /// The head serves this model first with no id announced, so a node cannot pull it: it
    /// serves a model by path, or was released before heads announced ids.
    Unpullable(String),
    /// The head serves no model.
    Nothing,
}

impl Follow {
    /// What a node that follows `head` serves: the head's first model, the one it announces as
    /// `model`, by the first of its ids.
    pub fn of(head: &crate::fleet_mdns::Head) -> Follow {
        match (head.models.first(), head.ids.first().filter(|id| !id.is_empty())) {
            (None, _) => Follow::Nothing,
            (Some(name), Some(id)) => Follow::Serve(Serving { name: name.clone(), id: id.clone() }),
            (Some(name), None) => Follow::Unpullable(name.clone()),
        }
    }

    /// What the node says when its head's model is this, and it served `was`.
    fn change_from(&self, was: &str) -> String {
        match self {
            Follow::Serve(s) => format!("the head serves {} now (by {}); following it from {was}", s.name, s.id),
            Follow::Unpullable(name) => format!(
                "the head serves {name} now, by no id a node can pull it by; leaving {was} to wait for a model this node can pull"
            ),
            Follow::Nothing => format!("the head serves no model now; leaving {was} to wait for one"),
        }
    }

    /// What the node says while it waits on a head that serves this.
    fn waiting(&self) -> String {
        match self {
            Follow::Serve(s) => format!("the head serves {}", s.name),
            Follow::Unpullable(name) => format!(
                "the head serves {name} with no id to pull it by (it has it by path, or predates announcing ids); \
                 waiting for it to serve a model by id, or give this node --model"
            ),
            Follow::Nothing => "the head serves no model yet; waiting".to_string(),
        }
    }
}

/// How long a node that follows its head waits before it looks again at a head that serves
/// nothing it can pull.
const LOOK_AGAIN: Duration = Duration::from_secs(10);
/// How long a node that follows its head watches the announcement before it browses anew.
const WATCH: Duration = Duration::from_secs(600);
/// How long a node that follows another model waits for the generations it runs to end.
const DRAIN: Duration = Duration::from_secs(30);

/// A node that follows its head's model: what it serves, and where it is told when the head
/// serves another.
struct Follower {
    serving: Serving,
    events: std::sync::mpsc::SyncSender<Event>,
}

impl Follower {
    /// Tells the node when `head` serves another model first than the one it serves; `true`
    /// when it did.
    fn hear(&self, head: &crate::fleet_mdns::Head) -> bool {
        let wanted = Follow::of(head);
        if wanted == Follow::Serve(self.serving.clone()) {
            return false;
        }
        let _ = self.events.try_send(Event::Follow(wanted));
        true
    }
}

/// Where the head is: an address given on the command line, resolved at each dial, or the
/// addresses the local network announced, looked for again when none of them answers. A look
/// takes only a head of this node's network, which may be another machine that took over.
struct Target {
    given: Option<String>,
    found: Mutex<Vec<SocketAddr>>,
    search: Mutex<Search>,
    /// The network looked for, as the search has it.
    net: Option<String>,
    follower: Option<Follower>,
}

impl Target {
    fn new(given: Option<String>, found: Vec<SocketAddr>, search: Search, follower: Option<Follower>) -> Target {
        let net = search.net().map(str::to_string);
        Target { given, found: Mutex::new(found), search: Mutex::new(search), net, follower }
    }

    fn addrs(&self) -> Vec<SocketAddr> {
        match &self.given {
            Some(h) => h.to_socket_addrs().map(|a| a.collect()).unwrap_or_default(),
            None => self.found.lock().expect("head addrs").clone(),
        }
    }

    /// After every address in `tried` failed: looks on the network again for a head of this
    /// node's network, unless another lane already found one anew. Lanes wait on the one that is
    /// looking. A head given by address is not looked for: that address is the head asked for,
    /// and it is dialed again until it answers. A node that follows its head's model follows
    /// the model of the head it finds.
    fn refind(&self, tried: &[SocketAddr]) {
        if self.given.is_some() {
            return;
        }
        let mut found = self.found.lock().expect("head addrs");
        if *found != tried {
            return;
        }
        if let Some(at) = tried.first() {
            eprintln!("superfluid: the head at {at} does not answer; looking for a head of this network");
        }
        let mut search = self.search.lock().expect("head search");
        match crate::fleet_mdns::find_head(&mut search, DISCOVER, &|msg| eprintln!("superfluid: {msg}")) {
            Ok(head) => {
                if head.nodes != *found {
                    eprintln!("superfluid: found the head at {}, serving {}", head.nodes[0], head.models.join(", "));
                }
                *found = head.nodes.clone();
                if let Some(f) = &self.follower {
                    f.hear(&head);
                }
            }
            Err(e) => eprintln!("superfluid: {e}"),
        }
    }

    /// Whether `head` is the one this node dials: of its network, or announcing none, at an
    /// address it dials.
    fn is(&self, head: &crate::fleet_mdns::Head) -> bool {
        let kin = crate::fleet_mdns::kin(self.net.as_deref(), head.net.as_deref());
        kin != crate::fleet_mdns::Kin::Foreign && head.nodes.iter().any(|a| self.addrs().contains(a))
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

/// Watches the announcement of the head a following node dials, and tells the node when it
/// serves another model first.
fn watch_head(target: Arc<Target>) {
    std::thread::spawn(move || loop {
        let Some(follower) = &target.follower else { return };
        match crate::fleet_mdns::watch(WATCH, |head| (target.is(&head) && follower.hear(&head)).then_some(())) {
            Ok(Some(())) => return,
            Ok(None) => {}
            Err(e) => {
                eprintln!("superfluid: cannot watch the head's announcement ({e}); looking again in {} s", LOOK_AGAIN.as_secs());
                std::thread::sleep(LOOK_AGAIN);
            }
        }
    });
}

/// The head of this node's network on the local network, however long it takes to appear.
fn discover(search: &mut Search) -> crate::fleet_mdns::Head {
    eprintln!("superfluid: looking for a fleet head on the local network");
    loop {
        match crate::fleet_mdns::find_head(search, DISCOVER, &|msg| eprintln!("superfluid: {msg}")) {
            Ok(head) => {
                eprintln!("superfluid: found the head at {}, serving {}", head.nodes[0], head.models.join(", "));
                return head;
            }
            Err(e) => eprintln!("superfluid: {e}; looking again"),
        }
    }
}

/// The head of this node's network and its first model, once that is a model a node can pull.
fn discover_model(search: &mut Search) -> (crate::fleet_mdns::Head, Serving) {
    let mut head = discover(search);
    let mut said = None;
    loop {
        match Follow::of(&head) {
            Follow::Serve(serving) => return (head, serving),
            other => {
                if said.as_ref() != Some(&other) {
                    eprintln!("superfluid: {}", other.waiting());
                    said = Some(other);
                }
                std::thread::sleep(LOOK_AGAIN);
                if let Ok(found) = crate::fleet_mdns::find_head(search, DISCOVER, &|msg| eprintln!("superfluid: {msg}")) {
                    head = found;
                }
            }
        }
    }
}

/// Whether the node runs or holds a generation.
fn busy(daemon: &Daemon) -> bool {
    let stats = daemon.sched_stats();
    stats.lanes_active.load(std::sync::atomic::Ordering::Relaxed) > 0
        || stats.queue_depth.iter().any(|q| q.load(std::sync::atomic::Ordering::Relaxed) > 0)
}

/// Starts this node again as it was started: with no `--model`, it serves the model its head
/// serves now, pulled and loaded as at the first start.
fn restart() -> ! {
    use std::os::unix::process::CommandExt;
    let exe = std::env::current_exe().unwrap_or_else(|e| fail(format!("cannot find this program to start it again: {e}")));
    eprintln!("superfluid: starting again to serve the head's model");
    let e = std::process::Command::new(exe).args(std::env::args_os().skip(1)).exec();
    fail(format!("cannot start again: {e}"))
}

/// The name a node serves its model under when `--model-name` does not give it. A model given by
/// id keeps its own name, `org/model` without the tag: the head serves it under that name, or
/// takes it on when the node joins. A model given by path keeps its own name when the head
/// serves a model by it, and otherwise takes the head's one model when the head serves one,
/// as the same model under another file name; the head holds it to that model's tokenizer.
fn name_among(served: &[String], own: String, by_id: bool) -> String {
    match served {
        _ if by_id || served.contains(&own) => own,
        [only] => only.clone(),
        _ => own,
    }
}

/// The links a node's lanes hold to its head, which a node that follows another model closes.
#[derive(Default)]
struct Links {
    stopping: std::sync::atomic::AtomicBool,
    open: Mutex<std::collections::HashMap<u32, TcpStream>>,
}

impl Links {
    fn stopping(&self) -> bool {
        self.stopping.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Keeps `sock`, `lane`'s link, to close when the links stop; `false`, and closed now, once
    /// they have.
    fn open(&self, lane: u32, sock: TcpStream) -> bool {
        let mut open = self.open.lock().expect("links");
        if self.stopping() {
            let _ = sock.shutdown(std::net::Shutdown::Both);
            return false;
        }
        open.insert(lane, sock);
        true
    }

    fn close(&self, lane: u32) {
        self.open.lock().expect("links").remove(&lane);
    }

    /// Closes every link, and keeps the lanes from dialing again.
    fn stop(&self) {
        let open = self.open.lock().expect("links");
        self.stopping.store(true, std::sync::atomic::Ordering::Release);
        for sock in open.values() {
            let _ = sock.shutdown(std::net::Shutdown::Both);
        }
    }

    /// Waits `d`, or until the links stop: `false` when they did.
    fn sleep(&self, d: Duration) -> bool {
        let by = std::time::Instant::now() + d;
        while std::time::Instant::now() < by {
            if self.stopping() {
                return false;
            }
            std::thread::sleep(Duration::from_millis(100).min(by.saturating_duration_since(std::time::Instant::now())));
        }
        !self.stopping()
    }
}

/// Dials the head and serves until the link ends, then dials again, until the links stop; the
/// first lane speaks for the node in the log.
fn keep_dialing(target: &Target, token: &[u8], mut conn: NodeAgent, lane: u32, links: &Links) {
    let say = |msg: String| {
        if lane == 0 {
            eprintln!("superfluid: {msg}");
        }
    };
    let mut backoff = Duration::from_secs(2);
    while !links.stopping() {
        let addrs = target.addrs();
        match dial(&addrs, token) {
            Ok((ep, sock)) => {
                if !links.open(lane, sock) {
                    return;
                }
                say(format!("joined the head at {target}"));
                backoff = Duration::from_secs(2);
                let served = conn.serve(ep);
                links.close(lane);
                if links.stopping() {
                    return;
                }
                match served {
                    Ok(()) => {
                        tracing::debug!(lane, "the link to the head closed");
                        say("the head closed the link".into())
                    }
                    Err(crate::DaemonError::Fleet(LinkFError::Unauthorized)) => fail(REFUSED),
                    Err(e) => {
                        tracing::debug!(lane, error = %e, "the link to the head ended");
                        say(format!("the link to the head ended: {e}"))
                    }
                }
            }
            Err(LinkFError::Unauthorized) => fail(REFUSED),
            Err(e) => {
                say(format!("cannot join the head at {target}: {e}"));
                target.refind(&addrs);
            }
        }
        if links.stopping() {
            return;
        }
        say(format!("dialing again in {} s", backoff.as_secs()));
        if !links.sleep(backoff) {
            return;
        }
        backoff = (backoff * 2).min(RETRY_MAX);
    }
}

pub struct LoadedNode {
    pub daemon: Daemon,
    /// The name the head sees the model under.
    pub model_name: String,
    /// The id the model was pulled by, `org/model:tag`; none for one given by path.
    pub id: Option<String>,
    /// The runtime, model, window and lanes, for the node's start line.
    pub summary: String,
    /// The model's tokenizer, for the head to check and, with no copy of the model, to use.
    pub tokenizer: Option<Arc<dyn superfluid_engine::Tokenizer>>,
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
    state: &NodeState,
) -> Result<LoadedNode, String> {
    let lanes = lanes.max(1);
    let catalog = Catalog::from_env();
    let runtime = runtime.map(|r| if r == "native" { "basert" } else { r });
    let (path, model_name, id) = resolve_model(&catalog, model, runtime, offline)?;
    let runtime = catalog.require(id).or_else(|_| crate::pulls::ensure_installed(&catalog, id, offline).map_err(String::from))?;
    let pulled = crate::pulls::looks_like_id(model).then(|| model.to_string());
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
        let args = crate::runtime_options::worker_args(runtime.id, &path, ctx, lanes, 0).map_err(|e| format!("model load failed: {e}"))?;
        EngineHost::try_spawn_loaded(move || rt.open(&args))
    }
    .map_err(|e| format!("model load failed: {e}"))?;
    let summary = format!(
        "runtime {} serves {}: context window {ctx} tokens ({sized}), {lanes} lanes",
        runtime.id.name(),
        path.display()
    );
    let mut node = loaded(host, Box::new(NoCodec), lanes, model_name, summary, state)?;
    node.tokenizer = crate::serve::open_tokenizer(&runtime, &path).ok();
    node.id = pulled;
    Ok(node)
}

/// A node on the test engine, which needs no model.
pub fn load_mock_node(max_context: Option<i32>, lanes: u32, state: &NodeState) -> Result<LoadedNode, String> {
    let lanes = lanes.max(1);
    let engine = crate::linked::engine("mock").ok_or("this build does not link the mock engine")?;
    let args = superfluid_adapter_kit::WorkerArgs {
        max_context: max_context.unwrap_or(4096).max(512),
        max_batch: lanes,
        ..Default::default()
    };
    let host = EngineHost::try_spawn_loaded(move || engine.open(&args)).map_err(|e| format!("model load failed: {e}"))?;
    loaded(host, Box::new(crate::MockCodec), lanes, "mock".into(), "the mock engine".into(), state)
}

fn loaded(
    host: EngineHost,
    codec: Box<dyn crate::TextCodec + Send + Sync>,
    lanes: u32,
    model_name: String,
    summary: String,
    state: &NodeState,
) -> Result<LoadedNode, String> {
    let store = SessionStore::open(&state.wal).map_err(|e| format!("node store: {e}"))?;
    let daemon = Daemon::with_park(store, host, codec, lanes as usize, state.park.clone());
    Ok(LoadedNode { daemon, model_name, id: None, summary, tokenizer: None })
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
/// opens the handshake. The socket comes back too, for the node to close the link by.
fn dial(addrs: &[SocketAddr], token: &[u8]) -> Result<(Endpoint, TcpStream), LinkFError> {
    let mut last = None;
    for addr in addrs {
        match TcpStream::connect_timeout(addr, CONNECT) {
            Ok(sock) => {
                let held = sock.try_clone()?;
                let ep = if token.is_empty() {
                    Endpoint::from_tcp(sock)
                } else {
                    Endpoint::from_tcp_secure(sock, token, Role::Responder, HANDSHAKE)
                }?;
                return Ok((ep, held));
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
/// Where a node keeps its state, under `$SUPERFLUID_HOME/fleet/nodes/<identity>`. Its session
/// store starts empty each run: a node's sessions last one generation, and the head keeps the
/// conversations. Its parked prefixes, with `--park`, are kept across runs.
pub struct NodeState {
    pub wal: PathBuf,
    pub park: Option<PathBuf>,
}

impl NodeState {
    pub fn open(home: &Path, identity: &str, park: bool) -> Result<NodeState, String> {
        let dir = home.join("fleet").join("nodes").join(identity.replace(['/', '\\'], "_"));
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let wal = dir.join("wal.log");
        for stale in [wal.clone(), dir.join("wal.log.prev")] {
            let _ = std::fs::remove_file(stale);
        }
        Ok(NodeState { wal, park: park.then(|| dir.join("park")) })
    }

    /// A session store of this process's own, and no parking.
    pub fn scratch() -> Result<NodeState, String> {
        let dir = std::env::temp_dir().join(format!("superfluid-node-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(NodeState { wal: dir.join("wal.log"), park: None })
    }
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
/// token, only nodes on this machine are. A node that serves a model the head does not serve
/// brings it ([`HeadModels::bring`]), unless it follows the head's first model and is about to
/// serve that; the first node to bring a model sets the tokenizer the head holds the others
/// serving it to, and the first to report the id it pulled a model by sets the id announced.
pub fn accept_nodes(listener: TcpListener, manager: Arc<FleetManager>, auth: Vec<u8>, models: Arc<HeadModels>) {
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
        let (manager, auth, models, in_handshake) = (Arc::clone(&manager), auth.clone(), Arc::clone(&models), Arc::clone(&in_handshake));
        std::thread::spawn(move || {
            let opened = FleetHead::accept(sock, auth);
            in_handshake.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            match opened {
                Ok(head) => {
                    let node = head.node();
                    let (identity, serves, ctx) = (node.node_identity.clone(), node.loadable_models.clone(), node.capacity.max_context_tokens);
                    let (ids, follows) = (node.model_ids.clone(), node.feature_flags & superfluid_proto::handshake::FOLLOWS_HEAD != 0);
                    for m in &serves {
                        manager.expect_tokenizer_first(m, head.tokenizer());
                    }
                    let joined = manager.adopt(head, peer.ip());
                    match &joined {
                        Ok((_, true, _)) => eprintln!(
                            "superfluid: fleet node '{identity}' joined from {peer}: {}, context window {ctx} tokens",
                            serves.join(", ")
                        ),
                        Ok((_, false, 1)) => eprintln!("superfluid: fleet node '{identity}' rejoined from {peer}"),
                        Ok(_) => {}
                        Err(why) => eprintln!("superfluid: a node at {peer} did not join: {why}"),
                    }
                    if joined.is_ok() {
                        models.know_ids(&serves, &ids);
                        if !follows {
                            models.bring(&identity, &serves);
                        }
                    }
                }
                Err(e) => eprintln!("superfluid: a node at {peer} did not join: {e}"),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fleet_mdns::Head;

    fn head(models: &[&str], ids: &[&str]) -> Head {
        Head {
            instance: "superfluid on box-a".into(),
            nodes: vec![SocketAddr::from(([192, 168, 1, 20], 48456))],
            models: models.iter().map(|m| m.to_string()).collect(),
            ids: ids.iter().map(|m| m.to_string()).collect(),
            http: Vec::new(),
            net: None,
        }
    }

    fn serving(name: &str, id: &str) -> Follow {
        Follow::Serve(Serving { name: name.into(), id: id.into() })
    }

    #[test]
    fn a_following_node_serves_the_heads_first_model_by_its_announced_id() {
        assert_eq!(Follow::of(&head(&["org/a", "org/b"], &["org/a:q4", "org/b:q8"])), serving("org/a", "org/a:q4"));
        assert_eq!(Follow::of(&head(&["local-a", "org/b"], &["", "org/b:q8"])), Follow::Unpullable("local-a".into()), "the first, even when another could be pulled");
        assert_eq!(Follow::of(&head(&["org/a"], &[])), Follow::Unpullable("org/a".into()), "a head from before ids");
        assert_eq!(Follow::of(&head(&[], &[])), Follow::Nothing);
    }

    #[test]
    fn a_follower_is_told_only_when_the_head_serves_another_model_first() {
        let (events, heard) = std::sync::mpsc::sync_channel(4);
        let follower = Follower { serving: Serving { name: "org/a".into(), id: "org/a:q4".into() }, events };
        assert!(!follower.hear(&head(&["org/a", "org/b"], &["org/a:q4", "org/b:q8"])), "the head took on another model: no change");
        assert!(heard.try_recv().is_err());
        assert!(follower.hear(&head(&["org/a"], &["org/a:q8"])), "another tag of the same model");
        assert!(matches!(heard.try_recv(), Ok(Event::Follow(f)) if f == serving("org/a", "org/a:q8")));
        assert!(follower.hear(&head(&["org/b", "org/a"], &["org/b:q8", "org/a:q4"])));
        assert!(matches!(heard.try_recv(), Ok(Event::Follow(f)) if f == serving("org/b", "org/b:q8")));
        assert!(follower.hear(&head(&["local-c"], &[])));
        assert!(matches!(heard.try_recv(), Ok(Event::Follow(Follow::Unpullable(n))) if n == "local-c"));
    }

    #[test]
    fn stopping_the_links_closes_each_and_keeps_lanes_from_dialing_again() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let links = Links::default();
        let sock = TcpStream::connect(addr).unwrap();
        let (mut theirs, _) = listener.accept().unwrap();
        assert!(links.open(0, sock));
        assert!(links.sleep(Duration::from_millis(10)), "nothing stops it yet");
        links.stop();
        use std::io::Read;
        assert_eq!(theirs.read(&mut [0u8; 8]).unwrap(), 0, "the head sees the link close");
        assert!(links.stopping());
        assert!(!links.sleep(Duration::from_secs(5)), "a lane waiting to dial again stops at once");
        let late = TcpStream::connect(addr).unwrap();
        let (mut theirs, _) = listener.accept().unwrap();
        assert!(!links.open(1, late), "a link opened after the stop is closed");
        assert_eq!(theirs.read(&mut [0u8; 8]).unwrap(), 0);
    }

    #[test]
    fn a_node_that_stops_its_links_leaves_its_head_and_its_lanes_end() {
        let token = b"a fleet token".to_vec();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let lease = superfluid_proto::linkf::Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
        let manager = Arc::new(FleetManager::with_options(Vec::new(), crate::fleet_manager::PlacementPolicy::LoadAware, lease, 90, 1));
        let models = HeadModels::fixed(Arc::clone(&manager), vec!["mock".into()]);
        let (m, t) = (Arc::clone(&manager), token.clone());
        std::thread::spawn(move || accept_nodes(listener, m, t, models));

        let state = NodeState::scratch().unwrap();
        let node = load_mock_node(None, 2, &state).unwrap();
        let agent = NodeAgent::new(Arc::new(node.daemon), "box-a", "mock").with_max_lanes(2).with_auth(token.clone());
        let target = Arc::new(Target::new(Some(addr), Vec::new(), Search::for_token(&token), None));
        let links = Arc::new(Links::default());
        let lanes: Vec<_> = (0..2)
            .map(|lane| {
                let (target, token, conn, links) = (Arc::clone(&target), token.clone(), agent.connection(), Arc::clone(&links));
                std::thread::spawn(move || keep_dialing(&target, &token, conn, lane, &links))
            })
            .collect();
        let up = || manager.status().first().is_some_and(|n| n.up && n.lanes == 2);
        let by = std::time::Instant::now() + Duration::from_secs(10);
        while !up() {
            assert!(std::time::Instant::now() < by, "the node joined on both lanes");
            std::thread::sleep(Duration::from_millis(20));
        }
        links.stop();
        let by = std::time::Instant::now() + Duration::from_secs(10);
        while !lanes.iter().all(|l| l.is_finished()) {
            assert!(std::time::Instant::now() < by, "every lane ended once its link closed");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn a_node_given_a_path_takes_its_own_name_when_served_else_the_heads_only_one() {
        let own = || "qwen".to_string();
        assert_eq!(name_among(&["llama".into(), "qwen".into()], own(), false), "qwen");
        assert_eq!(name_among(&["Qwen3-0.6B".into()], own(), false), "Qwen3-0.6B", "the same model under another file name");
        assert_eq!(name_among(&["llama".into(), "gemma".into()], own(), false), "qwen", "several, none it: its own");
        assert_eq!(name_among(&[], own(), false), "qwen");
    }

    #[test]
    fn a_node_given_an_id_keeps_its_name_for_the_head_to_take_on() {
        let own = || "Qwen/Qwen3-0.6B-GGUF".to_string();
        assert_eq!(name_among(&["llama".into()], own(), true), "Qwen/Qwen3-0.6B-GGUF", "not renamed to the head's one model");
        assert_eq!(name_among(&["Qwen/Qwen3-0.6B-GGUF".into()], own(), true), "Qwen/Qwen3-0.6B-GGUF");
        assert_eq!(name_among(&[], own(), true), "Qwen/Qwen3-0.6B-GGUF");
    }
}
