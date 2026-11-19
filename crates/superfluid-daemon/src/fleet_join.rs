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
use superfluid_proto::handshake::Served;
use crate::nodeagent::NodeAgent;
use crate::runtime_pick::RuntimeId;
use crate::runtimes::Catalog;
use crate::{Daemon, EngineHost, NoCodec, SessionStore};

pub const HELP: &str = "Serve a model, or several, to a fleet head from this machine.
Usage: superfluid node join [<head-host:port>] [--model <path|org/model[:tag]>]... [options]
  Dials the head (found on the local network when no address is given), serves the
  model to it, and dials again whenever the link drops. The head's token, from
  `superfluid fleet token` there, authenticates both ends and encrypts the link.
  A head found on the local network must be of this token's network: a head of another
  network is never joined, and one that announces none (released before network ids)
  only when no head of this network answers. When the head is lost, the node looks
  again and joins whichever head of its network answers, which may be another machine
  that took over; a head given by address is dialed again there, never looked for.
  With no --model, the node asks the head (found on the local network, or at the
  address given) what it serves, over the link the token authenticates: an announcement
  on the network only says where to ask, and nothing is pulled on its word. It serves the
  head's first model, pulled by the id the head states and served under the head's name,
  and follows it: when the head states another first model (a head that took over, one
  restarted with another model, or one whose announcement changed and that is asked
  again), the node lets what it runs finish, closes its links and engine, and starts again
  with the same arguments to serve the new one. A head that states no id for its model
  (one it has by path) is waited on; a head that states nothing over the link (released
  before heads did) is not followed: give --model.
  With --model, the model never changes: the node joins any head of its network, and a
  head that does not serve the model takes it on.
  The head resolves each request's sampling before a node sees it, with its own
  --temperature, --repeat-penalty, --presence-penalty and --spec-max-temperature as the
  defaults for a request that names none; a node applies what the request carries.
  A model given by id is served under its own name, org/model without the tag, and a
  head that does not serve it takes it on when the node joins. A model given by path
  is served under its file's name when the head serves a model by that name, and
  otherwise under the head's one model when it serves one, as the same model under
  another file name; --model-name names it otherwise.
  --model again serves several models at once, as `serve` serves several: each on its own
  engine with --max-batch lanes, their windows sized together for this device (each one's
  drafter counted), one connection to the head per lane of each. The head takes on each, and
  runs each session and request on the model it names. Each is served under its own name,
  or --model-name given once per --model, in order. A head released before heads named the
  model of each request is offered the first model only. A node that follows the head
  serves one model.
Options:
  --model <m>             The model, fixed; an id is pulled first, as `serve` pulls it;
                          again for each more model to serve
                          [default: the head's first model, followed]
  --token <value>         The head's fleet token, kept at $SUPERFLUID_HOME/fleet/token
  --runtime <id>          The runtime, when the model's format does not decide it
  --max-context <N>       This node's context window [default: sized for this device]
  --max-batch <N>         This node's lanes, each model's [default: 8]
  --class-lanes <class>=N[,...]
                          Caps this node's lanes per QoS class, each model's, as
                          `serve --class-lanes` does: the head sends each generation
                          in its request's class [default: uncapped]
  --identity <name>       The name the head shows for this node [default: the host name]
  --model-name <id>       The name the head matches on, once per --model [default: an
                          id's own name; a path's own name, or the head's one model when
                          it serves one and this node serves one]
  --offline               Pull nothing; the model must be in the runtime's cache
  --speculate <auto|prompt-lookup|mtp-head|<drafter>|off>
                          Draft tokens and verify them, as `serve` does; also dflash:,
                          dspark:, eagle3:, draft-model: <path or id>
                          [default: auto on basert, else off]
  --park                  Keep each prompt's prefix on disk, under
                          $SUPERFLUID_HOME/fleet/nodes/<identity>/park, so a restarted node
                          starts the head's sessions warm
  --park-lossy            Park in the lossy Q8 tier (about half the size); implies --park
  --park-budget-gb <N>    Disk budget for the parked prefixes, oldest evicted first; 0 is
                          unbounded [default: 20]
  --kv-bits <0|4|8|16|84> KV cache precision, on runtimes that take it (basert)
                          [default: 0, auto]
  --<runtime>-<name> <value>
                          A knob of that runtime, e.g. --llamacpp-n-gpu-layers 20;
                          `superfluid runtime options <id>` lists them
  --prefill-budget <N>, --tick-decode-budget <N>, --tick-target-ms <N>
                          The scheduler's budgets per tick, as `serve` takes them
  --spec-draft-tokens <N>, --[no-]spec-adaptive, --spec-min-yield <F>,
  --spec-yield-rounds <N>, --[no-]spec-throughput-gate, --spec-gate-probe-tokens <N>,
  --spec-min-speedup <F>, --spec-gate-reprobe <N>, --spec-gate-reprobe-max <N>,
  --dspark-confidence <F>, --[no-]spec-bitexact
                          Tune speculation, as `serve` takes them
  --otlp-endpoint <url>   Push this node's operational traces (never content) to an
                          OTLP/HTTP collector; --otlp-header, --otlp-service-name,
                          --otlp-filter, --otlp-queue, --otlp-batch-ms and
                          --otlp-timeout-ms as `serve` takes them
  --log-format <text|json>
                          Stderr as prose, or one JSON event a line for a program that runs
                          this [default: text, or SUPERFLUID_LOG_FORMAT]
";

const CONNECT: Duration = Duration::from_secs(5);
const HANDSHAKE: Duration = Duration::from_secs(10);
const DISCOVER: Duration = Duration::from_secs(30);
const RETRY_MAX: Duration = Duration::from_secs(30);
const REFUSED: &str = "the head refused this node's token; copy the head's `superfluid fleet token` here with --token";
/// The exit code of a node the head removed, told apart from a flag it got wrong (2) and a
/// failure along the way (1).
pub const REMOVED_EXIT: i32 = 5;
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

/// Ends a node the head at `head` removed, with [`REMOVED_EXIT`]: it neither dials that head
/// again nor starts again to follow it.
fn removed(head: impl std::fmt::Display) -> ! {
    ending();
    crate::events::emit("node.removed", serde_json::json!({"head": head.to_string()}));
    crate::events::fail(REMOVED_EXIT, format!("the head at {head} removed this node; it will refuse it until it restarts"))
}

fn fail(msg: impl std::fmt::Display) -> ! {
    crate::events::fail(2, msg)
}

/// The head refused this node's token: no dialing again will change that.
fn refused() -> ! {
    ending();
    crate::events::emit("node.refused", serde_json::json!({"reason": REFUSED}));
    fail(REFUSED)
}

/// Lets the first of the node's threads that ends it say why: the head tells every lane, and the
/// others wait for the exit, so the node says it once.
fn ending() {
    static ENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if ENDING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        loop {
            std::thread::park();
        }
    }
}

fn join(args: &[String]) {
    crate::events::init("node join", args).unwrap_or_else(|e| fail(e));
    let (mut head, mut token, mut identity) = (None, None, None);
    let (mut models, mut names): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    let mut park = false;
    let mut opts = NodeOptions::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_else(|| fail(format!("{a} takes a value")));
        match a.as_str() {
            "--model" => models.push(val()),
            "--token" => token = Some(val()),
            "--runtime" => opts.runtime = Some(val()),
            "--identity" => identity = Some(val()),
            "--model-name" => names.push(val()),
            "--max-context" => opts.max_context = Some(val().parse().unwrap_or_else(|_| fail("--max-context takes a number"))),
            "--max-batch" => opts.lanes = val().parse().unwrap_or_else(|_| fail("--max-batch takes a number")),
            "--offline" => opts.offline = true,
            "--park" => park = true,
            // Read before the rest by events::init.
            "--log-format" => drop(val()),
            s if s.starts_with("--log-format=") => {}
            "-h" | "--help" => {
                print!("{HELP}");
                return;
            }
            // The flags `serve` takes for the engine, a runtime's own among them: after every other,
            // since any --<word>-<word> may name a runtime's option.
            s if s.starts_with('-') => {
                if !opts.take(s, &mut it).unwrap_or_else(|e| fail(e)) {
                    fail(format!("unknown flag {s}"))
                }
            }
            s if head.is_none() => head = Some(s.to_string()),
            s => fail(format!("unexpected argument {s}")),
        }
    }
    if models.is_empty() && !names.is_empty() {
        fail("--model-name goes with --model; a node with no --model serves the head's model under the head's name");
    }
    if names.len() > models.len() || (models.len() > 1 && !names.is_empty() && names.len() != models.len()) {
        fail("give --model-name once for each --model, in the same order, or not at all");
    }
    let model = models.first().cloned();
    // The node-agent's warnings (a fenced session, a silent head) go to stderr, as noded's do;
    // its scheduler's spans to a collector with --otlp-endpoint, as `serve`'s do.
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".to_string());
    let otlp = opts.engine.otlp_traces().unwrap_or_else(|e| fail(e));
    if let Err(e) = crate::telemetry::init_sinks(crate::telemetry::LogSink::Stderr(filter), None, otlp) {
        eprintln!("superfluid: logging: {e}");
    }
    let home = crate::runtimes::default_home();
    if let Some(t) = &token {
        crate::fleet_token::write(&home, t.trim().as_bytes()).unwrap_or_else(|e| fail(e));
    }
    let token = crate::fleet_token::read(&home).unwrap_or_default();
    let identity = identity.unwrap_or_else(crate::metrics::hostname);
    let (events, heard) = std::sync::mpsc::sync_channel::<Event>(8);
    let mut search = Search::for_token(&token);
    let (target, announced, follows) = match (head, &model) {
        (given, None) => {
            let (found, serving) = learn_head(given.as_deref(), &mut search, &token, &identity);
            let follower = Arc::new(Follower::new(serving.clone(), events.clone()));
            let found = if given.is_some() { Vec::new() } else { found };
            (Target::new(given, found, search, Some(follower)), None, Some(serving))
        }
        (Some(h), Some(_)) => (Target::new(Some(h), Vec::new(), search, None), None, None),
        (None, Some(_)) => {
            let found = discover(&mut search);
            (Target::new(None, found.nodes, search, None), Some(found.models), None)
        }
    };
    let addrs = target.addrs();
    if addrs.is_empty() {
        fail(format!("cannot resolve {target}"));
    }
    if token.is_empty() && !addrs.iter().all(|a| a.ip().is_loopback()) {
        fail("no fleet token: run `superfluid fleet token` on the head and pass --token <value> here");
    }
    opts.lanes = opts.lanes.max(1);
    let max_batch = opts.lanes;
    if let Err(e) = opts.class_lanes() {
        fail(e);
    }
    let state = NodeState::open(&home, &identity, opts.parks(park)).unwrap_or_else(|e| fail(e));
    let sources = match &follows {
        Some(serving) => vec![serving.id.clone()],
        None => models.clone(),
    };
    let loaded = load_nodes(&sources, &opts, &state).unwrap_or_else(|e| fail(e));
    let several = loaded.len() > 1;
    let mut node: Option<NodeAgent> = None;
    let (mut served, mut daemons) = (Vec::new(), Vec::new());
    for (i, loaded) in loaded.into_iter().enumerate() {
        eprintln!("superfluid: {}", loaded.summary);
        // Several models are each served under their own name, or the one --model-name gives it.
        let name = match (&follows, names.get(i)) {
            (Some(serving), _) => serving.name.clone(),
            (None, Some(name)) => name.clone(),
            (None, None) if several => loaded.model_name,
            (None, None) => name_among(announced.as_deref().unwrap_or_default(), loaded.model_name, loaded.id.is_some()),
        };
        let daemon = Arc::new(loaded.daemon);
        let mut agent = match node.take() {
            None => NodeAgent::new(Arc::clone(&daemon), identity.clone(), name.clone()),
            Some(agent) => agent.with_model(Arc::clone(&daemon), name.clone()),
        };
        if let Some(tok) = loaded.tokenizer {
            agent = agent.with_tokenizer_of(tok);
        }
        if let Some(id) = loaded.id {
            agent = agent.with_model_id(id);
        }
        node = Some(agent);
        served.push(name);
        daemons.push(daemon);
    }
    if let Some(name) = repeated(served.iter().map(String::as_str)) {
        fail(format!("two of the models are served as {name}; give each its own --model-name"));
    }
    let model_name = served.join(", ");
    // One connection per lane of each model, so each model's lanes can all be busy at once.
    let max_batch = max_batch * served.len() as u32;
    let mut node = node.expect("a node serves a model").with_max_lanes(max_batch);
    if let Some(follower) = target.follower.clone() {
        node = node.following().with_on_head(move |serves| {
            follower.hear(serves);
        });
    }
    // A head that keeps a connection for short calls asks for it; one connection more, once.
    let ask = events.clone();
    let node = node.with_auth(token.clone()).with_on_control_ask(move || {
        let _ = ask.try_send(Event::ControlAsk);
    });
    let follow_note = if follows.is_some() { ", following the head's model" } else { "" };
    eprintln!("superfluid: node '{identity}' serves {model_name}{follow_note}; dialing the head at {target} once per lane ({max_batch})");
    let mut dialing = serde_json::json!({"head": target.to_string(), "node": identity, "model": served[0], "lanes": max_batch});
    if several {
        dialing["models"] = serde_json::json!(served);
    }
    crate::events::emit("node.dialing", dialing);
    // One connection per lane, each its own generation at a time on the head.
    let (target, links) = (Arc::new(target), Arc::new(Links::default()));
    if follows.is_some() {
        watch_head(Arc::clone(&target), token.clone(), identity.clone());
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
    let (next, id) = match &wanted {
        Follow::Serve(s) => (Some(&s.name), Some(&s.id)),
        Follow::Unpullable(name) => (Some(name), None),
        Follow::Nothing => (None, None),
    };
    crate::events::emit("node.head.changed", serde_json::json!({"model": next, "id": id, "from": model_name}));
    // What the node runs ends, then its links, then its engine; it starts again as it was
    // started, and pulls and loads the head's model as it did at first.
    let by = std::time::Instant::now() + DRAIN;
    while daemons.iter().any(|d| busy(d)) && std::time::Instant::now() < by {
        std::thread::sleep(Duration::from_millis(200));
    }
    links.stop();
    for lane in lanes {
        let _ = lane.join();
    }
    drop(node);
    // The node that starts next opens the session logs this one holds, and loads its model into
    // the memory this one's worker has. An exec lets go of both. Windows has none: this process
    // waits there for the next as its child, so it lets go of them itself, and each daemon ends
    // here with its store's lock and its engine's worker.
    for daemon in daemons {
        daemon.signal_shutdown();
        daemon.shutdown();
    }
    restart();
}

/// What the node's main thread is told while its lanes run.
enum Event {
    /// The head asked for a connection for short calls.
    ControlAsk,
    /// The head this node follows serves another model now.
    Follow(Follow),
}

/// What a node that follows its head serves: the head's first model, by the id it states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Serving {
    /// The name the head serves the model under, which the node serves it under too.
    pub name: String,
    /// The id the node pulls it by, `org/model:tag`.
    pub id: String,
}

/// What a node that follows its head makes of what the head states it serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Follow {
    Serve(Serving),
    /// The head serves this model first with no id stated, so a node cannot pull it: the head
    /// has it by path.
    Unpullable(String),
    /// The head serves no model.
    Nothing,
}

impl Follow {
    /// What a node that follows a head serves, from what the head stated it serves over a link
    /// the token authenticates: the head's first model, by the first of its ids.
    pub fn of(serves: &Served) -> Follow {
        match (serves.models.first(), serves.ids.first().filter(|id| !id.is_empty())) {
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
                "the head serves {name} with no id to pull it by (it has it by path); \
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
    /// Whether the node said its head states nothing over the link.
    told_unstated: std::sync::atomic::AtomicBool,
}

impl Follower {
    fn new(serving: Serving, events: std::sync::mpsc::SyncSender<Event>) -> Follower {
        Follower { serving, events, told_unstated: std::sync::atomic::AtomicBool::new(false) }
    }

    /// Tells the node when its head stated, over a link the token authenticates, that it serves
    /// another model first than the one this node serves; `true` when it did. A head that states
    /// nothing is not followed.
    fn hear(&self, serves: Option<&Served>) -> bool {
        let Some(serves) = serves else {
            if !self.told_unstated.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "superfluid: the head states nothing over the link of what it serves (it was released before heads did); \
                     this node keeps serving {} and follows it no further",
                    self.serving.name
                );
            }
            return false;
        };
        let wanted = Follow::of(serves);
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
    follower: Option<Arc<Follower>>,
}

impl Target {
    fn new(given: Option<String>, found: Vec<SocketAddr>, search: Search, follower: Option<Arc<Follower>>) -> Target {
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
    /// and it is dialed again until it answers. What the head found serves is learned when a
    /// lane joins it, from its hello, not from its announcement.
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
            crate::events::emit("node.head.lost", serde_json::json!({"head": at.to_string()}));
            crate::events::emit("node.discover.start", serde_json::json!({}));
        }
        let mut search = self.search.lock().expect("head search");
        match crate::fleet_mdns::find_head(&mut search, DISCOVER, &|msg| eprintln!("superfluid: {msg}")) {
            Ok(head) => {
                if head.nodes != *found {
                    eprintln!("superfluid: found the head at {}, serving {}", head.nodes[0], head.models.join(", "));
                    crate::events::emit("node.head.found", serde_json::json!({"addr": head.nodes[0].to_string(), "models": head.models}));
                }
                *found = head.nodes;
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

/// Watches the announcement of the head a following node dials. Anyone on the network can
/// announce, so a change there only has the node ask the head it dials, over a link the token
/// authenticates, what it serves; it follows what the head states. A query the head does not
/// answer is asked again ([`Asking`]), announced again or not, since a head whose lanes stay
/// connected may not announce again.
fn watch_head(target: Arc<Target>, token: Vec<u8>, identity: String) {
    std::thread::spawn(move || {
        let Some(follower) = target.follower.clone() else { return };
        let mut asking = Asking::new(ASK_BACKOFF);
        loop {
            let wait = asking.next().map_or(WATCH, |due| due.saturating_duration_since(std::time::Instant::now()).max(Duration::from_millis(50)));
            // Ends with `true` once the node is told to follow, and with `false` after a query that
            // got no answer, to wait no longer than its next try.
            let heard = crate::fleet_mdns::watch(wait, |head| {
                let changed = target.is(&head) && head.models.first() != Some(&follower.serving.name);
                if !(changed && asking.hint(&head.models, std::time::Instant::now())) {
                    return None;
                }
                let followed = ask(&target, &follower, &token, &identity, &mut asking, &head.models);
                (followed || asking.next().is_some()).then_some(followed)
            });
            match heard {
                Ok(Some(true)) => return,
                Ok(Some(false)) => {}
                Ok(None) => {
                    if let Some(models) = asking.due(std::time::Instant::now()) {
                        if ask(&target, &follower, &token, &identity, &mut asking, &models) {
                            return;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("superfluid: cannot watch the head's announcement ({e}); looking again in {} s", LOOK_AGAIN.as_secs());
                    std::thread::sleep(LOOK_AGAIN);
                }
            }
        }
    });
}

/// Asks the head a following node dials what it serves, after its announcement changed to list
/// `models`, and tells the follower; `true` once the node is told to follow. A query with no
/// answer, or with an answer that does not settle the announcement ([`settles`]), is noted in
/// `asking`, to be asked again; only the last try that gets no answer says so.
fn ask(target: &Target, follower: &Follower, token: &[u8], identity: &str, asking: &mut Asking, models: &[String]) -> bool {
    match ask_head(&target.addrs(), token, identity) {
        Err(LinkFError::Removed) => removed(target),
        Ok(serves) => {
            if settles(serves.as_ref(), models) {
                asking.answered(models);
            } else {
                let again = asking.failed(models, std::time::Instant::now());
                tracing::debug!(again, "the head's answer does not bear out its announcement; asking again");
            }
            follower.hear(serves.as_ref())
        }
        Err(e) => {
            let again = asking.failed(models, std::time::Instant::now());
            tracing::debug!(error = %e, again, "the head did not say what it serves after its announcement changed");
            if !again {
                eprintln!(
                    "superfluid: the head did not say what it serves after its announcement changed ({ASK_TRIES} tries, the last: {e}); \
                     this node keeps serving {}",
                    follower.serving.name
                );
            }
            false
        }
    }
}

/// Whether the head's answer settles an announcement listing `models`: it states the same first
/// model. The first model is what a node follows, and an announcement may list fewer models
/// than the head states (a name too long for a TXT property is left out), so the rest is not
/// compared. An answer naming another first model leaves the announcement unsettled: it may be
/// a stale one, and a real change behind it is still to be caught, so it is asked again, a
/// bounded number of times ([`ASK_TRIES`]). A head that states nothing settles it: it will
/// never state more.
fn settles(answer: Option<&Served>, models: &[String]) -> bool {
    answer.is_none_or(|served| served.models.first() == models.first())
}

/// How many times a following node asks its head what it serves, for one announcement change
/// the head does not answer for, or does not bear out.
const ASK_TRIES: u32 = 5;
/// The wait before a following node asks its head again, doubling with each try.
const ASK_BACKOFF: Duration = Duration::from_secs(2);

/// When a following node asks its head what it serves after the head's announcement changes:
/// once for each model list announced, and, while a query gets no answer, again up to
/// [`ASK_TRIES`] times with a wait that doubles, whether or not the head announces again. A
/// list counts as asked only once the head answered for it.
struct Asking {
    /// The announced list the head last answered for.
    answered: Option<Vec<String>>,
    /// The announced list not answered for yet: its tries so far, and when it is next asked.
    pending: Option<(Vec<String>, u32, std::time::Instant)>,
    backoff: Duration,
}

impl Asking {
    fn new(backoff: Duration) -> Asking {
        Asking { answered: None, pending: None, backoff }
    }

    /// Whether to ask now, for an announcement listing `models` heard at `now`.
    fn hint(&mut self, models: &[String], now: std::time::Instant) -> bool {
        if self.answered.as_deref() == Some(models) {
            return false;
        }
        match &self.pending {
            Some((asked, _, due)) if asked[..] == *models => *due <= now,
            _ => {
                self.pending = Some((models.to_vec(), 0, now));
                true
            }
        }
    }

    /// The list a query is due for at `now`, with no announcement heard.
    fn due(&self, now: std::time::Instant) -> Option<Vec<String>> {
        self.pending.as_ref().filter(|(_, _, due)| *due <= now).map(|(models, _, _)| models.clone())
    }

    /// When a query is next due.
    fn next(&self) -> Option<std::time::Instant> {
        self.pending.as_ref().map(|(_, _, due)| *due)
    }

    /// The head answered for `models`.
    fn answered(&mut self, models: &[String]) {
        self.answered = Some(models.to_vec());
        if self.pending.as_ref().is_some_and(|(asked, _, _)| asked[..] == *models) {
            self.pending = None;
        }
    }

    /// A query for `models` got no answer at `now`: `true` while it is to be asked again.
    fn failed(&mut self, models: &[String], now: std::time::Instant) -> bool {
        let Some((_, tries, due)) = self.pending.as_mut().filter(|(asked, _, _)| asked[..] == *models) else {
            return false;
        };
        *tries += 1;
        if *tries >= ASK_TRIES {
            self.pending = None;
            return false;
        }
        *due = now + self.backoff * 2u32.pow(*tries - 1);
        true
    }
}

/// The head of this node's network on the local network, however long it takes to appear.
fn discover(search: &mut Search) -> crate::fleet_mdns::Head {
    eprintln!("superfluid: looking for a fleet head on the local network");
    crate::events::emit("node.discover.start", serde_json::json!({}));
    loop {
        match crate::fleet_mdns::find_head(search, DISCOVER, &|msg| eprintln!("superfluid: {msg}")) {
            Ok(head) => {
                eprintln!("superfluid: found the head at {}, serving {}", head.nodes[0], head.models.join(", "));
                crate::events::emit("node.head.found", serde_json::json!({"addr": head.nodes[0].to_string(), "models": head.models}));
                return head;
            }
            Err(e) => eprintln!("superfluid: {e}; looking again"),
        }
    }
}

/// The head a node with no `--model` follows, and the model it states it serves first, once
/// that is one a node can pull: the head at `given`, or the one of this node's network found on
/// the local network, where an announcement only says where to ask. Nothing is pulled or loaded
/// on an announcement's word: the head is asked over a link the token authenticates. A head that
/// states nothing there (released before heads did) is not followed.
fn learn_head(given: Option<&str>, search: &mut Search, token: &[u8], identity: &str) -> (Vec<SocketAddr>, Serving) {
    let mut said: Option<String> = None;
    let mut say_once = |msg: String| {
        if said.as_ref() != Some(&msg) {
            eprintln!("superfluid: {msg}");
            said = Some(msg);
        }
    };
    let mut rounds = 0u32;
    loop {
        if rounds > 0 {
            std::thread::sleep(LOOK_AGAIN);
        }
        rounds += 1;
        let addrs = match given {
            Some(h) => h.to_socket_addrs().map(|a| a.collect::<Vec<_>>()).unwrap_or_default(),
            None if rounds == 1 => discover(search).nodes,
            None => match crate::fleet_mdns::find_head(search, DISCOVER, &|msg| eprintln!("superfluid: {msg}")) {
                Ok(head) => head.nodes,
                Err(_) => Vec::new(),
            },
        };
        let Some(at) = addrs.first().copied() else {
            say_once(format!("cannot find {}; looking again", given.unwrap_or("a fleet head of this network")));
            continue;
        };
        if token.is_empty() && !addrs.iter().all(|a| a.ip().is_loopback()) {
            fail("no fleet token: run `superfluid fleet token` on the head and pass --token <value> here");
        }
        match ask_head(&addrs, token, identity) {
            Ok(Some(serves)) => match Follow::of(&serves) {
                Follow::Serve(serving) => return (addrs, serving),
                other => say_once(other.waiting()),
            },
            Ok(None) => fail(format!(
                "the head at {at} states nothing over the link of what it serves (it was released before heads did), \
                 so a node with no --model does not follow it; give this node --model"
            )),
            Err(LinkFError::Unauthorized) if given.is_some() => refused(),
            Err(LinkFError::Removed) => removed(at),
            Err(LinkFError::Unauthorized) => say_once(format!(
                "the head announced at {at} does not hold this network's token, so it is not followed \
                 (anyone on the network can copy an announcement)"
            )),
            Err(e) => say_once(format!("cannot ask the head at {at} what it serves: {e}; asking again")),
        }
    }
}

/// What the head at `addrs` states it serves in its hello, read over a link `token`
/// authenticates and answered as a node that only asks ([`superfluid_proto::handshake::ASKS_ONLY`]),
/// which the head does not take in; `None` from a head that states nothing (one released before
/// heads did), which is left unanswered. [`LinkFError::Removed`] from a head that removed this
/// node, which says so once it has the node's answer; what it stated is then not to be acted on.
pub fn ask_head(addrs: &[SocketAddr], token: &[u8], identity: &str) -> Result<Option<Served>, LinkFError> {
    let (mut ep, _sock) = dial(addrs, token)?;
    let hello = ep.read_hello_within(HANDSHAKE)?;
    let Some(serves) = hello.serves.clone() else { return Ok(None) };
    let ack = superfluid_proto::handshake::HelloAckF {
        chosen_version: superfluid_linkf::driver::PROTO_VERSION,
        node_identity: identity.to_string(),
        capacity: Default::default(),
        loadable_models: Vec::new(),
        feature_flags: superfluid_proto::handshake::ASKS_ONLY,
        limits: superfluid_proto::linkf::LinkFLimits {
            max_frame: superfluid_proto::MAX_FRAME,
            transfer_window: 8,
            max_transfer: 1 << 26,
            transfer_budget: 1 << 28,
        },
        model_ids: Vec::new(),
    };
    ep.answer_hello(&hello, &ack, token)?;
    // A head that removed this node says so before it closes the link; any other just closes it.
    let by = std::time::Instant::now() + REMOVED_WAIT;
    while std::time::Instant::now() < by {
        match ep.pump() {
            Ok(frames) if frames.iter().any(|f| f.msg_type == superfluid_proto::linkf::msg_type::REMOVED) => {
                return Err(LinkFError::Removed)
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    Ok(Some(serves))
}

/// How long [`ask_head`] waits, once answered, for the head to say it removed this node before
/// taking what it stated: a head closes the link to a node that only asks as soon as it has read
/// its answer.
const REMOVED_WAIT: Duration = Duration::from_secs(2);

/// Whether the node runs or holds a generation.
fn busy(daemon: &Daemon) -> bool {
    let stats = daemon.sched_stats();
    stats.lanes_active.load(std::sync::atomic::Ordering::Relaxed) > 0
        || stats.queue_depth.iter().any(|q| q.load(std::sync::atomic::Ordering::Relaxed) > 0)
}

/// Starts this node again as it was started: with no `--model`, it serves the model its head
/// serves now, pulled and loaded as at the first start.
fn restart() -> ! {
    let exe = std::env::current_exe().unwrap_or_else(|e| fail(format!("cannot find this program to start it again: {e}")));
    eprintln!("superfluid: starting again to serve the head's model");
    // The next image writes its events to the stream this one wrote to, from its `start` on.
    crate::events::prepare_exec();
    let e = crate::sys::replace_process(std::process::Command::new(exe).args(std::env::args_os().skip(1)));
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
    let tell = |event: &str, fields: serde_json::Value| {
        if lane == 0 {
            crate::events::emit(event, fields);
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
                tell("node.joined", serde_json::json!({"head": target.to_string()}));
                backoff = Duration::from_secs(2);
                let served = conn.serve(ep);
                links.close(lane);
                // Before the links stopping: a node that was about to follow another model ends.
                if let Err(crate::DaemonError::Fleet(LinkFError::Removed)) = served {
                    removed(target);
                }
                if links.stopping() {
                    return;
                }
                match served {
                    Ok(()) => {
                        tracing::debug!(lane, "the link to the head closed");
                        say("the head closed the link".into());
                        tell("node.link.ended", serde_json::json!({"reason": "the head closed the link"}))
                    }
                    Err(crate::DaemonError::Fleet(LinkFError::Unauthorized)) => refused(),
                    Err(e) => {
                        tracing::debug!(lane, error = %e, "the link to the head ended");
                        say(format!("the link to the head ended: {e}"));
                        tell("node.link.ended", serde_json::json!({"reason": format!("the link to the head ended: {e}")}))
                    }
                }
            }
            Err(LinkFError::Unauthorized) => refused(),
            Err(e) => {
                say(format!("cannot join the head at {target}: {e}"));
                tell("node.dial.failed", serde_json::json!({"head": target.to_string(), "error": e.to_string()}));
                target.refind(&addrs);
            }
        }
        if links.stopping() {
            return;
        }
        say(format!("dialing again in {} s", backoff.as_secs()));
        tell("node.retry", serde_json::json!({"in_secs": backoff.as_secs()}));
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

/// How a node runs its model, beyond the model itself: what `superfluid node join` and
/// `superfluid-noded` take for the engine, each as `superfluid serve` takes it.
#[derive(Debug, Clone)]
pub struct NodeOptions {
    /// `--runtime`, when the model's format does not decide it; `mock` is the test engine.
    pub runtime: Option<String>,
    /// `--max-context`; none sizes the window for this device.
    pub max_context: Option<i32>,
    /// `--max-batch`.
    pub lanes: u32,
    /// `--class-lanes`, as given; [`NodeOptions::class_lanes`] reads it against the lanes.
    pub class_lanes: Option<String>,
    pub offline: bool,
    /// `--speculate`, kept as named, `off` included.
    pub speculate: Option<String>,
    /// `--park-lossy`: parked prefixes in the lossy Q8 tier, on F16 KV pools; it parks, as
    /// `--park` does.
    pub park_lossy: bool,
    /// The engine flags `serve` takes too: `--kv-bits`, the scheduler's budgets, the speculation
    /// knobs, `--park-budget-gb`, the trace exporter.
    pub engine: crate::engine_flags::EngineFlags,
    /// `--<runtime>-<name> <value>`, as given; checked against the runtime once it is known.
    pub runtime_options: Vec<(String, Option<String>)>,
}

impl Default for NodeOptions {
    fn default() -> NodeOptions {
        NodeOptions {
            runtime: None,
            max_context: None,
            lanes: 8,
            class_lanes: None,
            offline: false,
            speculate: None,
            park_lossy: false,
            engine: Default::default(),
            runtime_options: Vec::new(),
        }
    }
}

impl NodeOptions {
    /// Reads `arg` when it is one of the flags both nodes take from `serve` (`--speculate`,
    /// `--park-lossy`, [`crate::engine_flags::EngineFlags`], a runtime's option), its value from
    /// `rest`: `Ok(true)` when it was one, `Ok(false)` when not.
    pub fn take(&mut self, arg: &str, rest: &mut std::slice::Iter<'_, String>) -> Result<bool, String> {
        match arg {
            "--speculate" => self.speculate = Some(rest.next().cloned().ok_or_else(|| format!("{arg} takes a value"))?),
            "--class-lanes" => self.class_lanes = Some(rest.next().cloned().ok_or_else(|| format!("{arg} takes a value"))?),
            "--park-lossy" => self.park_lossy = true,
            _ => {
                let mut missing = false;
                let took = self.engine.take(arg, &mut || {
                    rest.next().cloned().unwrap_or_else(|| {
                        missing = true;
                        String::new()
                    })
                });
                if missing {
                    return Err(format!("{arg} takes a value"));
                }
                if !took.map_err(|e| e.to_string())? {
                    let Some(option) = crate::runtime_options::candidate(arg, rest) else { return Ok(false) };
                    self.runtime_options.push(option);
                }
            }
        }
        Ok(true)
    }

    /// Whether the node keeps its prompts' prefixes on disk: `--park`, or `--park-lossy`.
    pub fn parks(&self, park: bool) -> bool {
        park || self.park_lossy
    }

    /// The lanes each QoS class may hold of the node's, each model's with several
    /// ([`class_lanes_of`]); an error for a `--class-lanes` that does not read against them.
    pub fn class_lanes(&self) -> Result<[usize; 4], String> {
        class_lanes_of(self.class_lanes.as_deref(), self.lanes)
    }
}

/// `--class-lanes` on a node: the lanes each QoS class may hold of `lanes`, as
/// [`crate::qos::parse_class_lanes`] reads them for `serve`; uncapped when not given.
pub fn class_lanes_of(spec: Option<&str>, lanes: u32) -> Result<[usize; 4], String> {
    match spec {
        None => Ok([0; 4]),
        Some(spec) => crate::qos::parse_class_lanes(spec, lanes.max(1) as usize).map_err(|e| format!("--class-lanes: {e}")),
    }
}

/// A fleet node's model by path or id, as `superfluid serve` loads one: pulled when given by
/// id, on its runtime (installed first if it never was), with the runtime's options checked
/// against what it declares, in a window sized for this device unless `--max-context` pins it,
/// and speculating as `serve` does: with the strategy `--speculate` names (`off` is none), else
/// `auto` on the baseRT runtime, its drafter counted in the window.
pub fn load_node(model: &str, opts: &NodeOptions, state: &NodeState) -> Result<LoadedNode, String> {
    Ok(load_nodes(&[model.to_string()], opts, state)?.remove(0))
}

/// What [`load_nodes`] settles for one model before any is loaded.
struct Planned {
    path: PathBuf,
    model_name: String,
    pulled: Option<String>,
    runtime: crate::runtimes::Runtime,
    spec: NodeSpeculation,
}

/// Several models on one node, each loaded as [`load_node`] loads one, on an engine and daemon of
/// its own, `--max-batch` lanes each: their windows are sized together, as `serve` sizes the
/// models it serves at once on one device, every model's drafter charged; with `--max-context`,
/// each has that window. The first model keeps `state`; each other keeps its sessions and
/// parked prefixes in a directory of its own there, by the `--model` it was given
/// ([`NodeState::of_model`]).
pub fn load_nodes(models: &[String], opts: &NodeOptions, state: &NodeState) -> Result<Vec<LoadedNode>, String> {
    let lanes = opts.lanes.max(1);
    opts.class_lanes()?;
    if models.is_empty() && opts.runtime.as_deref() != Some("mock") {
        return Err("no model to serve: give --model".into());
    }
    if let Some(model) = repeated(models.iter().map(String::as_str)) {
        return Err(format!("--model {model} is given twice"));
    }
    // The test engine, which needs no model, as `superfluid-noded --engine mock` serves it.
    if opts.runtime.as_deref() == Some("mock") {
        return load_mock_nodes(models, opts, state);
    }
    let catalog = Catalog::from_env();
    use crate::runtime_options::Unresolved;
    let given = crate::runtime_options::resolve_candidates(&opts.runtime_options, &catalog.ids()).map_err(|e| match e {
        Unresolved::Unknown(flag) => format!("unknown flag {flag}"),
        Unresolved::NoValue(flag) => format!("{flag} takes a value"),
    })?;
    // Before the runtime is first asked for, as `serve` sets them: one may be what makes it usable.
    crate::runtime_options::adopt(&given)?;
    let kv_bits = opts.engine.kv_bits_for(opts.park_lossy)?;
    let runtime = opts.runtime.as_deref().map(|r| if r == "native" { "basert" } else { r });
    let mut planned = Vec::with_capacity(models.len());
    for model in models {
        let (path, model_name, id) = resolve_model(&catalog, model, runtime, opts.offline)?;
        let runtime = catalog.require(id).or_else(|_| match crate::runtime_options::blamed(&catalog, id, &given) {
            Some(why) => Err(why),
            None => crate::pulls::ensure_installed(&catalog, id, opts.offline).map_err(String::from),
        })?;
        crate::serve::refuse_statically(&runtime, opts.park_lossy, opts.engine.kv_bits, opts.speculate.as_deref())?;
        crate::runtime_options::settle(&given, &runtime, &catalog)?;
        let pulled = crate::pulls::looks_like_id(model).then(|| model.to_string());
        let spec = node_speculation(opts.speculate.as_deref(), &catalog, &runtime, &path, opts.offline)?;
        planned.push(Planned { path, model_name, pulled, runtime, spec });
    }
    let (ctx, sized) = match opts.max_context {
        Some(n) => (n, format!("--max-context {n}")),
        None => match device_window(&planned, lanes, kv_bits) {
            Ok(n) if planned.len() > 1 => (n, format!("sized for this device, {} models together", planned.len())),
            Ok(n) => (n, "sized for this device".to_string()),
            Err(why) => (8192, format!("{why}; --max-context N pins it")),
        },
    };
    let ctx = ctx.max(512);
    opts.engine.export_spec_bitexact();
    let mut loaded_nodes = Vec::with_capacity(planned.len());
    for (i, p) in planned.into_iter().enumerate() {
        let Planned { path, model_name, pulled, runtime, spec } = p;
        let own;
        let state = match i {
            0 => state,
            _ => {
                own = state.of_model(&models[i])?;
                &own
            }
        };
        crate::events::emit("model.load.start", serde_json::json!({"model": model_name, "runtime": runtime.id.name(), "path": path.display().to_string()}));
        let host = if runtime.worker.bin.is_file() {
            EngineHost::spawn_process(crate::WorkerSpec {
                bin: runtime.worker.bin.clone(),
                args: runtime.serve_args(&path, ctx, lanes, kv_bits),
                stderr: None,
            })
        } else {
            let rt = crate::linked::engine(runtime.id.name())
                .ok_or_else(|| format!("the {} runtime has no worker here", runtime.id.name()))?;
            let args = crate::runtime_options::worker_args(runtime.id, &path, ctx, lanes, kv_bits).map_err(|e| format!("model load failed: {e}"))?;
            EngineHost::try_spawn_loaded(move || rt.open(&args))
        }
        .map_err(|e| format!("model load failed: {e}"))
        .inspect_err(|e| crate::events::emit("model.load.failed", serde_json::json!({"model": model_name, "error": e})))?;
        crate::events::emit("model.load.done", serde_json::json!({"model": model_name, "runtime": runtime.id.name(), "context": ctx, "lanes": lanes}));
        let speculators = if spec.ranked.is_empty() { "" } else { ", 1 speculator" };
        let daemon = loaded(host, Box::new(NoCodec), state, spec, &model_name, opts)?;
        let kv = crate::serve::kv_cache(&daemon, kv_bits).map(|said| format!(", {said}")).unwrap_or_default();
        let summary = format!("runtime {} serves {}: context window {ctx} tokens ({sized}), {lanes} lanes{speculators}{kv}", runtime.id.name(), path.display());
        let tokenizer = crate::serve::open_tokenizer(&runtime, &path).ok();
        loaded_nodes.push(LoadedNode { daemon, model_name, id: pulled, summary, tokenizer });
    }
    Ok(loaded_nodes)
}

/// A name given more than once among `names`.
fn repeated<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let mut seen = std::collections::HashSet::new();
    names.into_iter().find(|n| !seen.insert(*n))
}

/// A node on the test engine, which needs no model. It speculates only when `--speculate` names
/// a strategy, as a runtime other than baseRT does; its `auto` is prompt-lookup.
pub fn load_mock_node(opts: &NodeOptions, state: &NodeState) -> Result<LoadedNode, String> {
    load_mock_as(opts, state, "mock")
}

/// Mock engines for `models`: one is served as `mock`, as [`load_mock_node`] serves it; several
/// each under the name given, the first keeping `state` and each other its own under it.
fn load_mock_nodes(models: &[String], opts: &NodeOptions, state: &NodeState) -> Result<Vec<LoadedNode>, String> {
    if models.len() <= 1 {
        return Ok(vec![load_mock_node(opts, state)?]);
    }
    models
        .iter()
        .enumerate()
        .map(|(i, name)| match i {
            0 => load_mock_as(opts, state, name),
            _ => load_mock_as(opts, &state.of_model(name)?, name),
        })
        .collect()
}

fn load_mock_as(opts: &NodeOptions, state: &NodeState, name: &str) -> Result<LoadedNode, String> {
    if let Some((flag, _)) = opts.runtime_options.first() {
        return Err(format!("{flag}: the mock engine takes no runtime option"));
    }
    let kv_bits = opts.engine.kv_bits_for(opts.park_lossy)?;
    let lanes = opts.lanes.max(1);
    opts.class_lanes()?;
    let engine = crate::linked::engine("mock").ok_or("this build does not link the mock engine")?;
    let args = superfluid_adapter_kit::WorkerArgs {
        max_context: opts.max_context.unwrap_or(4096).max(512),
        max_batch: lanes,
        kv_bits,
        ..Default::default()
    };
    let host = EngineHost::try_spawn_loaded(move || engine.open(&args)).map_err(|e| format!("model load failed: {e}"))?;
    let spec = match crate::serve::speculate_directive(opts.speculate.as_deref(), "mock") {
        None => NodeSpeculation::default(),
        Some(d) => NodeSpeculation { ranked: vec![if d.trim() == "auto" { "prompt-lookup".into() } else { d.to_string() }], auto: false },
    };
    let daemon = loaded(host, Box::new(crate::MockCodec), state, spec, name, opts)?;
    let summary = if name == "mock" { "the mock engine".to_string() } else { format!("the mock engine as {name}") };
    Ok(LoadedNode { daemon, model_name: name.into(), id: None, summary, tokenizer: None })
}

/// The node's daemon over the engine `host` loaded, refused as `serve` refuses a flag the engine
/// says it cannot take, its parked prefixes swept to `--park-budget-gb` first.
fn loaded(
    host: EngineHost,
    codec: Box<dyn crate::TextCodec + Send + Sync>,
    state: &NodeState,
    spec: NodeSpeculation,
    model_name: &str,
    opts: &NodeOptions,
) -> Result<Daemon, String> {
    let caps = host.capabilities().read().expect("capability record").clone();
    caps.check_startup_flags(opts.park_lossy, opts.engine.kv_bits, opts.speculate.as_deref())?;
    if let Some(dir) = &state.park {
        crate::serve::settle_park_dir(dir, true, opts.engine.park_budget_gb);
    }
    let store = SessionStore::open(&state.wal).map_err(|e| format!("node store: {e}"))?;
    Daemon::with_options(store, host, codec, daemon_options(opts, state, spec, model_name)).map_err(|e| format!("model load failed: {e}"))
}

/// What the node's daemon is built with: its lanes, its parking, its speculation, and the engine
/// flags, as `serve` builds its own.
fn daemon_options(opts: &NodeOptions, state: &NodeState, spec: NodeSpeculation, model_name: &str) -> crate::DaemonOptions {
    let mut ranked = spec.ranked.into_iter();
    let engine = &opts.engine;
    crate::DaemonOptions {
        max_lanes: opts.lanes.max(1) as usize,
        // Read before any model is loaded (load_nodes), so it reads here.
        class_lanes: opts.class_lanes().unwrap_or_default(),
        park_dir: state.park.clone(),
        park_lossy: opts.park_lossy,
        park_budget_bytes: engine.park_budget_gb.saturating_mul(1 << 30),
        tick_decode_budget: engine.tick_decode_budget,
        tick_target_ms: engine.tick_target_ms,
        prefill_budget: engine.prefill_budget,
        speculate: ranked.next(),
        speculate_auto: spec.auto,
        speculate_fallbacks: ranked.collect(),
        speculation: engine.speculation.clone(),
        trace_scope: crate::otlp::TraceScope::model(model_name),
        ..crate::DaemonOptions::default()
    }
}

/// The drafters a node speculates with, ranked: the first is registered, and under `auto` the
/// rest are tried in turn when the engine refuses it.
#[derive(Default)]
struct NodeSpeculation {
    ranked: Vec<String>,
    auto: bool,
}

/// What `superfluid serve` would speculate with for this model, chosen the same way: the strategy
/// `named` (a drafter given by id pulled first), else `auto` on the baseRT runtime, settled
/// against the engine's capabilities and the drafters installed (or paired in the hub catalog,
/// pulled unless `offline`).
fn node_speculation(
    named: Option<&str>,
    catalog: &Catalog,
    runtime: &crate::runtimes::Runtime,
    path: &Path,
    offline: bool,
) -> Result<NodeSpeculation, String> {
    crate::speculate_resolve::allow_catalog_fetch(!offline);
    let mut named = named.map(str::to_string);
    if let Some((kind, id)) = named.as_deref().and_then(crate::speculate_resolve::drafter_id) {
        let flags = crate::pulls::PullFlags { offline, ..Default::default() };
        let drafter = crate::pulls::pull(catalog, runtime.id, id, &flags).map_err(|e| format!("--speculate {kind}:{id}: {e}"))?;
        named = Some(format!("{kind}:{}", drafter.display()));
    }
    let Some(directive) = crate::serve::speculation_for(named.as_deref(), &runtime.static_capabilities(), path, runtime.id.name()) else {
        return Ok(NodeSpeculation::default());
    };
    let pull = crate::serve::auto_drafter_pull(catalog, runtime.id, path);
    let pull = (!offline).then_some(&pull as &dyn Fn(&str));
    let ranked = crate::speculate_resolve::resolve_all(directive, path, pull)?;
    match ranked.first() {
        Some(r) => eprintln!("superfluid: speculation: {r}"),
        None => {
            let dirs: Vec<_> = crate::speculate_resolve::auto_search_dirs(path).iter().map(|(d, _)| d.display().to_string()).collect();
            eprintln!(
                "superfluid: speculation: auto found no head in {} and no fitting drafter in {} — decoding plainly",
                path.display(),
                dirs.join(", ")
            );
        }
    }
    Ok(NodeSpeculation { ranked, auto: directive.trim() == "auto" })
}

/// A drafter as the sizing charges it: its weights, whether they are a head the model embeds, and
/// the model it drafts for when it is loaded beside it.
type Speculator = (PathBuf, bool, Option<usize>);

/// The window this device holds for `lanes` lanes of each of the models, as `superfluid serve`
/// sizes the models it serves together: with the drafter each speculates with charged too, the
/// smallest window of any combination of drafters they may fall back to (or, past 16
/// combinations, every candidate charged at once). Models on runtimes of more than one family
/// are not sized, as `serve` does not size them.
fn device_window(models: &[Planned], lanes: u32, kv_bits: i32) -> Result<i32, String> {
    let id = models[0].runtime.id;
    if let Some(other) = models.iter().find(|m| m.runtime.id.base() != id.base()) {
        return Err(format!(
            "not sized for this device: the {} runtime sizes it, and {} is served by {}",
            id.base().name(),
            other.path.display(),
            other.runtime.id.base().name()
        ));
    }
    if crate::linked::sizes_context(id) {
        use crate::speculate_resolve::{sizing_path, SpecLoad};
        const MAX_CANDIDATES: usize = 16;
        // Each model's choices: the drafters it may speculate with, or none.
        let each: Vec<Vec<Option<Speculator>>> = models
            .iter()
            .enumerate()
            .map(|(i, m)| {
                if m.spec.ranked.is_empty() {
                    return vec![None];
                }
                m.spec
                    .ranked
                    .iter()
                    .take(MAX_CANDIDATES)
                    .map(|r| {
                        sizing_path(r, &m.path)
                            .map(|(path, load)| (path, load == SpecLoad::EmbeddedHead, (load == SpecLoad::Sidecar).then_some(i)))
                    })
                    .collect()
            })
            .collect();
        let few_enough = each.iter().try_fold(1usize, |p, c| p.checked_mul(c.len()).filter(|p| *p <= MAX_CANDIDATES)).is_some();
        let choices: Vec<Vec<Speculator>> = if few_enough || models.len() == 1 {
            let mut combos: Vec<Vec<Speculator>> = vec![Vec::new()];
            for options in &each {
                combos = combos
                    .iter()
                    .flat_map(|combo| options.iter().map(move |o| combo.iter().cloned().chain(o.clone()).collect()))
                    .collect();
            }
            combos
        } else {
            vec![each.into_iter().flatten().flatten().collect()]
        };
        let paths: Vec<PathBuf> = models.iter().map(|m| m.path.clone()).collect();
        let mut window: Option<i32> = None;
        for speculators in &choices {
            if let Some(n) = crate::linked::suggest_max_context(id, &paths, speculators, lanes as i32, kv_bits)? {
                window = Some(window.map_or(n, |w| w.min(n)));
            }
        }
        return window.ok_or_else(|| format!("the {} runtime could not size a window for this device", id.base().name()));
    }
    let each = models.iter().map(|m| crate::runtimes::sizing_of(&m.runtime, &m.path)).collect::<Result<Vec<_>, String>>()?;
    let trained = each.iter().map(|s| s.trained_context).min().unwrap_or(0);
    let per_lane = superfluid_adapter_kit::sizing::window(&each, lanes)
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
                // Asked before the clone: on Windows a connect with a timeout leaves the socket's
                // own record of being connected to be brought up to date, a clone made first
                // keeps the stale one, and a shutdown through it (`Links::stop`) is then refused
                // as "not connected", so the lane it was to end never hears.
                #[cfg(windows)]
                sock.peer_addr()?;
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

    /// Where the node keeps the state of `model`, another model than its first: in a directory
    /// of its own under `models`, as a server with several models keeps each, its session store
    /// started empty and its prefixes parked there when the node parks.
    pub fn of_model(&self, model: &str) -> Result<NodeState, String> {
        let base = self.wal.parent().unwrap_or(Path::new(".")).join("models");
        let dir = crate::registry::state_dir(&base, model).map_err(|e| format!("{}: {e}", base.display()))?;
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let wal = dir.join("wal.log");
        for stale in [wal.clone(), dir.join("wal.log.prev")] {
            let _ = std::fs::remove_file(stale);
        }
        Ok(NodeState { wal, park: self.park.as_ref().map(|_| dir.join("park")) })
    }

    /// A session store of this process's own, and no parking. Each one is in a directory of
    /// its own, so two in one process never open the same log.
    pub fn scratch() -> Result<NodeState, String> {
        static NTH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nth = NTH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("superfluid-node-{}-{nth}", std::process::id()));
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
            crate::events::emit("fleet.node.refused", serde_json::json!({"addr": peer.to_string(), "reason": "this head has no fleet token (`superfluid fleet token` makes one)"}));
            continue;
        }
        while in_handshake.load(std::sync::atomic::Ordering::Acquire) >= HANDSHAKES_AT_ONCE {
            std::thread::sleep(Duration::from_millis(20));
        }
        in_handshake.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let (manager, auth, models, in_handshake) = (Arc::clone(&manager), auth.clone(), Arc::clone(&models), Arc::clone(&in_handshake));
        std::thread::spawn(move || {
            let opened = FleetHead::accept_serving(sock, auth, models.served(), &|identity| manager.refuses(identity, peer.ip()));
            in_handshake.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            match opened {
                Ok(None) => tracing::debug!(%peer, "a node asked what this head serves, or was refused"),
                Ok(Some(head)) => {
                    let node = head.node();
                    let (identity, serves, ctx) = (node.node_identity.clone(), node.loadable_models.clone(), node.capacity.max_context_tokens);
                    let (ids, follows) = (node.model_ids.clone(), node.feature_flags & superfluid_proto::handshake::FOLLOWS_HEAD != 0);
                    for m in &serves {
                        manager.expect_tokenizer_first(m, head.tokenizer_of(m));
                    }
                    let joined = manager.adopt(head, peer.ip());
                    match &joined {
                        Ok((_, true, _)) => {
                            eprintln!(
                                "superfluid: fleet node '{identity}' joined from {peer}: {}, context window {ctx} tokens",
                                serves.join(", ")
                            );
                            crate::events::emit("fleet.node.joined", serde_json::json!({"node": identity, "addr": peer.to_string(), "models": serves, "context": ctx}));
                        }
                        Ok((_, false, 1)) => {
                            eprintln!("superfluid: fleet node '{identity}' rejoined from {peer}");
                            crate::events::emit("fleet.node.rejoined", serde_json::json!({"node": identity, "addr": peer.to_string()}));
                        }
                        Ok(_) => {}
                        Err(why) => {
                            eprintln!("superfluid: a node at {peer} did not join: {why}");
                            crate::events::emit("fleet.node.refused", serde_json::json!({"addr": peer.to_string(), "reason": why.to_string()}));
                        }
                    }
                    if joined.is_ok() {
                        models.know_ids(&serves, &ids);
                        if !follows {
                            models.bring(&identity, &serves);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("superfluid: a node at {peer} did not join: {e}");
                    crate::events::emit("fleet.node.refused", serde_json::json!({"addr": peer.to_string(), "reason": e.to_string()}));
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(models: &[&str], ids: &[&str]) -> Served {
        Served { models: models.iter().map(|m| m.to_string()).collect(), ids: ids.iter().map(|m| m.to_string()).collect() }
    }

    fn serving(name: &str, id: &str) -> Follow {
        Follow::Serve(Serving { name: name.into(), id: id.into() })
    }

    #[test]
    fn a_following_node_serves_the_heads_first_model_by_its_stated_id() {
        assert_eq!(Follow::of(&head(&["org/a", "org/b"], &["org/a:q4", "org/b:q8"])), serving("org/a", "org/a:q4"));
        assert_eq!(Follow::of(&head(&["local-a", "org/b"], &["", "org/b:q8"])), Follow::Unpullable("local-a".into()), "the first, even when another could be pulled");
        assert_eq!(Follow::of(&head(&["org/a"], &[])), Follow::Unpullable("org/a".into()), "no id stated");
        assert_eq!(Follow::of(&head(&[], &[])), Follow::Nothing);
    }

    #[test]
    fn a_follower_is_told_only_when_the_head_serves_another_model_first() {
        let (events, heard) = std::sync::mpsc::sync_channel(4);
        let follower = Follower::new(Serving { name: "org/a".into(), id: "org/a:q4".into() }, events);
        assert!(!follower.hear(Some(&head(&["org/a", "org/b"], &["org/a:q4", "org/b:q8"]))), "the head took on another model: no change");
        assert!(heard.try_recv().is_err());
        assert!(follower.hear(Some(&head(&["org/a"], &["org/a:q8"]))), "another tag of the same model");
        assert!(matches!(heard.try_recv(), Ok(Event::Follow(f)) if f == serving("org/a", "org/a:q8")));
        assert!(follower.hear(Some(&head(&["org/b", "org/a"], &["org/b:q8", "org/a:q4"]))));
        assert!(matches!(heard.try_recv(), Ok(Event::Follow(f)) if f == serving("org/b", "org/b:q8")));
        assert!(follower.hear(Some(&head(&["local-c"], &[]))));
        assert!(matches!(heard.try_recv(), Ok(Event::Follow(Follow::Unpullable(n))) if n == "local-c"));
        assert!(!follower.hear(None), "a head that states nothing is not followed");
        assert!(!follower.hear(None));
        assert!(heard.try_recv().is_err());
    }

    fn list(models: &[&str]) -> Vec<String> {
        models.iter().map(|m| m.to_string()).collect()
    }

    #[test]
    fn a_change_the_head_did_not_answer_for_is_asked_again_and_one_it_answered_is_not() {
        let (b, c) = (list(&["org/b"]), list(&["org/c"]));
        let t0 = std::time::Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut asking = Asking::new(Duration::from_millis(100));
        assert!(asking.hint(&b, t0), "a new list is asked at once");
        assert!(asking.failed(&b, t0), "no answer: asked again");
        assert!(!asking.hint(&b, at(50)), "announced again before the wait is out");
        assert_eq!(asking.due(at(50)), None);
        assert_eq!(asking.next(), Some(at(100)));
        assert_eq!(asking.due(at(100)), Some(b.clone()), "due with no announcement heard");
        assert!(asking.failed(&b, at(100)));
        assert_eq!(asking.next(), Some(at(300)), "the wait doubles");
        assert!(asking.hint(&b, at(300)), "or when it is announced again once due");
        asking.answered(&b);
        assert!(!asking.hint(&b, at(400)), "an answered list is not asked again");
        assert_eq!(asking.due(at(10_000)), None);
        assert!(asking.hint(&c, at(400)), "another list is");
        for i in 1..ASK_TRIES {
            assert!(asking.failed(&c, at(400)), "try {i} of {ASK_TRIES}");
        }
        assert!(!asking.failed(&c, at(400)), "given up after {ASK_TRIES} tries");
        assert_eq!(asking.next(), None);
        assert!(asking.hint(&c, at(500)), "until the head announces it again");
    }

    /// A head serving `org/b` that drops the first connection, as a head busy or restarting
    /// might, and answers the rest.
    fn head_dropping_the_first_ask(bind: &str, token: &[u8]) -> std::net::SocketAddr {
        let listener = std::net::TcpListener::bind(bind).unwrap();
        let addr = listener.local_addr().unwrap();
        let lease = superfluid_proto::linkf::Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
        let manager = Arc::new(FleetManager::with_options(Vec::new(), crate::fleet_manager::PlacementPolicy::LoadAware, lease, 90, 1));
        let models = HeadModels::fixed(Arc::clone(&manager), vec!["org/b".into()]);
        models.know_ids(&["org/b".into()], &["org/b:q4".into()]);
        let token = token.to_vec();
        std::thread::spawn(move || {
            drop(listener.accept());
            accept_nodes(listener, manager, token, models)
        });
        addr
    }

    #[test]
    fn a_follower_asks_again_a_head_that_refused_its_first_ask_and_follows_its_answer() {
        let token = b"a fleet token".to_vec();
        let addr = head_dropping_the_first_ask("127.0.0.1:0", &token).to_string();
        let (events, heard) = std::sync::mpsc::sync_channel(4);
        let follower = Arc::new(Follower::new(Serving { name: "org/a".into(), id: "org/a:q4".into() }, events));
        let target = Target::new(Some(addr), Vec::new(), Search::for_token(&token), Some(Arc::clone(&follower)));
        let mut asking = Asking::new(Duration::from_millis(50));
        let announced = list(&["org/b"]);

        assert!(asking.hint(&announced, std::time::Instant::now()));
        assert!(!ask(&target, &follower, &token, "follower", &mut asking, &announced), "the first ask gets no answer");
        assert!(heard.try_recv().is_err());
        assert!(!asking.hint(&announced, std::time::Instant::now()), "not handled, only waiting");
        std::thread::sleep(Duration::from_millis(60));
        let again = asking.due(std::time::Instant::now()).expect("asked again with no new announcement");
        assert!(ask(&target, &follower, &token, "follower", &mut asking, &again), "the head answers the next");
        assert!(matches!(heard.try_recv(), Ok(Event::Follow(f)) if f == serving("org/b", "org/b:q4")));
    }

    #[test]
    fn an_answer_settles_an_announcement_only_with_the_same_first_model() {
        let b = list(&["org/b"]);
        assert!(!settles(Some(&head(&["org/a"], &["org/a:q4"])), &b), "a stale announcement");
        assert!(settles(Some(&head(&["org/b"], &["org/b:q4"])), &b));
        assert!(settles(Some(&head(&["org/b", "org/very-long-name"], &[])), &b), "a name the announcement left out");
        assert!(!settles(Some(&head(&["org/c", "org/b"], &[])), &b), "the first is what a node follows");
        assert!(settles(None, &b), "a head that states nothing never will");
    }

    #[test]
    fn a_stale_announcement_the_head_contradicts_does_not_hide_its_later_real_change() {
        let token = b"a fleet token".to_vec();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let lease = superfluid_proto::linkf::Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
        let manager = Arc::new(FleetManager::with_options(Vec::new(), crate::fleet_manager::PlacementPolicy::LoadAware, lease, 90, 1));
        let models = HeadModels::fixed(Arc::clone(&manager), vec!["org/a".into()]);
        models.know_ids(&["org/a".into(), "org/b".into()], &["org/a:q4".into(), "org/b:q4".into()]);
        let daemon = || {
            let dir = std::env::temp_dir().join(format!("superfluid-stale-{}-{}", std::process::id(), rand_tag()));
            std::fs::create_dir_all(&dir).unwrap();
            let host = crate::EngineHost::spawn(|| (superfluid_engine::MockEngine::new(superfluid_engine::EngineConfig::default()), None)).unwrap();
            Arc::new(crate::Daemon::new(crate::SessionStore::open(&dir.join("wal.log")).unwrap(), host, Box::new(crate::MockCodec), 1))
        };
        let registry = Arc::new(crate::registry::ModelRegistry::with_initial("org/a", daemon(), models.loader()));
        models.attach(&registry);
        let (m, t, served) = (Arc::clone(&manager), token.clone(), Arc::clone(&models));
        std::thread::spawn(move || accept_nodes(listener, m, t, served));
        let (events, heard) = std::sync::mpsc::sync_channel(4);
        let follower = Arc::new(Follower::new(Serving { name: "org/a".into(), id: "org/a:q4".into() }, events));
        let target = Target::new(Some(addr), Vec::new(), Search::for_token(&token), Some(Arc::clone(&follower)));
        let mut asking = Asking::new(Duration::from_millis(50));
        let b = list(&["org/b"]);

        // A late announcement of org/b, heard while the head serves org/a again.
        assert!(asking.hint(&b, std::time::Instant::now()));
        assert!(!ask(&target, &follower, &token, "follower", &mut asking, &b), "the head states org/a: nothing to follow");
        assert!(heard.try_recv().is_err());
        // The same head now serves org/b first; no lane reconnects and the list announced is the same.
        registry.insert_loaded("org/b", daemon());
        registry.unload("org/a").unwrap();
        assert_eq!(models.served().models, ["org/b"]);
        std::thread::sleep(Duration::from_millis(60));
        let again = asking.due(std::time::Instant::now()).expect("the unsettled announcement is asked again");
        assert!(ask(&target, &follower, &token, "follower", &mut asking, &again));
        assert!(matches!(heard.try_recv(), Ok(Event::Follow(f)) if f == serving("org/b", "org/b:q4")));
    }

    fn rand_tag() -> u64 {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    #[test]
    #[ignore = "needs multicast on an interface: cargo test -p superfluid --lib fleet_join -- --ignored"]
    fn a_watching_follower_asks_again_with_no_new_announcement_and_follows() {
        let token = b"a_watching_follower_asks_again".to_vec();
        let at = head_dropping_the_first_ask("0.0.0.0:0", &token);
        let net = crate::fleet_mdns::network_id(&token);
        let _announced = crate::fleet_mdns::advertise(at.port(), &["org/b".into()], None, Some(&net)).unwrap();
        let mut search = Search::for_token(&token);
        let head = crate::fleet_mdns::find_head(&mut search, Duration::from_secs(10), &|_| {}).unwrap();
        let (events, heard) = std::sync::mpsc::sync_channel(4);
        let follower = Arc::new(Follower::new(Serving { name: "org/a".into(), id: "org/a:q4".into() }, events));
        let target = Arc::new(Target::new(None, head.nodes, search, Some(follower)));
        watch_head(target, token, "follower".into());
        // The head announces once; the first ask is dropped, the next, 2 s on, answered.
        match heard.recv_timeout(Duration::from_secs(15)) {
            Ok(Event::Follow(f)) => assert_eq!(f, serving("org/b", "org/b:q4")),
            _ => panic!("the follower did not follow the head's answer"),
        }
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
        let node = load_mock_node(&NodeOptions { lanes: 2, ..NodeOptions::default() }, &state).unwrap();
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

    /// A node speculates as `serve` does: with the strategy named, `off` being none, and on a
    /// runtime other than baseRT (the mock engine here) not unless one is named.
    #[test]
    fn a_node_speculates_with_the_strategy_named_and_not_otherwise_off_basert() {
        let strategy = |speculate: Option<&str>| {
            let state = NodeState::scratch().unwrap();
            let opts = NodeOptions { lanes: 2, speculate: speculate.map(str::to_string), ..NodeOptions::default() };
            let node = load_mock_node(&opts, &state).unwrap();
            node.daemon.speculation().map(|g| g.strategy_id.to_string())
        };
        assert_eq!(strategy(None), None, "the mock engine is not baseRT: no default");
        assert_eq!(strategy(Some("off")), None);
        assert_eq!(strategy(Some("prompt-lookup")).as_deref(), Some("prompt-lookup"));
        assert_eq!(strategy(Some("auto")).as_deref(), Some("prompt-lookup"), "auto off baseRT is prompt-lookup");
        let state = NodeState::scratch().unwrap();
        let opts = NodeOptions { lanes: 2, speculate: Some("eagle3".into()), ..NodeOptions::default() };
        assert!(load_mock_node(&opts, &state).is_err(), "a strategy the engine refuses fails the load");
    }

    /// Two nodes of one process, each on a scratch store, run side by side: neither finds its
    /// session log held by the other.
    #[test]
    fn scratch_stores_of_one_process_do_not_share_a_log() {
        let opts = NodeOptions { lanes: 2, ..NodeOptions::default() };
        let first = load_mock_node(&opts, &NodeState::scratch().unwrap()).unwrap();
        let second = load_mock_node(&opts, &NodeState::scratch().unwrap());
        assert!(second.is_ok(), "the second loads while the first runs: {:?}", second.err());
        drop(first);
    }

    fn node_options(args: &[&str]) -> Result<NodeOptions, String> {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let mut opts = NodeOptions::default();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            assert!(opts.take(a, &mut it)?, "{a} is a node's engine flag");
        }
        Ok(opts)
    }

    /// A node reads the engine flags `serve` takes, and its daemon is built with them as `serve`
    /// builds its own.
    #[test]
    fn a_node_builds_its_daemon_with_the_engine_flags_serve_takes() {
        let opts = node_options(&[
            "--speculate", "prompt-lookup", "--park-lossy", "--park-budget-gb", "3", "--prefill-budget", "256",
            "--tick-decode-budget", "64", "--tick-target-ms", "120", "--spec-draft-tokens", "5", "--no-spec-throughput-gate",
            "--llamacpp-n-gpu-layers", "20", "--mlx-venv=/env",
        ])
        .unwrap();
        assert!(opts.parks(false), "--park-lossy parks");
        assert_eq!(opts.engine.kv_bits_for(opts.park_lossy), Ok(16), "on F16 pools, as serve's lossy parking is");
        assert_eq!(
            opts.runtime_options,
            vec![("--llamacpp-n-gpu-layers".to_string(), Some("20".to_string())), ("--mlx-venv".to_string(), Some("/env".to_string()))]
        );
        let state = NodeState { wal: PathBuf::from("/unused"), park: Some(PathBuf::from("/park")) };
        let spec = NodeSpeculation { ranked: vec!["prompt-lookup".into()], auto: false };
        let d = daemon_options(&NodeOptions { lanes: 3, ..opts.clone() }, &state, spec, "qwen");
        assert_eq!((d.max_lanes, d.park_dir.as_deref(), d.park_lossy, d.park_budget_bytes), (3, Some(Path::new("/park")), true, 3 << 30));
        assert_eq!((d.prefill_budget, d.tick_decode_budget, d.tick_target_ms), (256, 64, 120));
        assert_eq!(d.speculate.as_deref(), Some("prompt-lookup"));
        assert_eq!((d.speculation.max_draft, d.speculation.throughput_gate), (Some(5), Some(false)));
        assert_eq!(d.trace_scope.model, "qwen");

        let defaults = daemon_options(&NodeOptions::default(), &state, NodeSpeculation::default(), "m");
        let serve = crate::DaemonOptions::default();
        assert_eq!((defaults.prefill_budget, defaults.park_budget_bytes, defaults.tick_target_ms), (serve.prefill_budget, serve.park_budget_bytes, serve.tick_target_ms));

        // The mock engine loads with them, and refuses what serve refuses.
        let node = load_mock_node(&NodeOptions { lanes: 2, runtime_options: Vec::new(), ..opts }, &NodeState::scratch().unwrap()).unwrap();
        assert_eq!(node.daemon.speculation().map(|g| g.strategy_id), Some("prompt-lookup".to_string()));
        let refused = |args: &[&str]| load_mock_node(&node_options(args).unwrap(), &NodeState::scratch().unwrap()).err().unwrap_or_default();
        assert!(refused(&["--kv-bits", "5"]).starts_with("--kv-bits 5: expected"));
        assert!(refused(&["--park-lossy", "--kv-bits", "8"]).contains("--park-lossy requires F16 KV pools"));
        assert!(refused(&["--mlx-venv", "/env"]).contains("the mock engine takes no runtime option"));
        assert_eq!(node_options(&["--spec-draft-tokens", "16"]).unwrap_err(), "--spec-draft-tokens must be in 1..=15");
        assert_eq!(node_options(&["--prefill-budget"]).unwrap_err(), "--prefill-budget takes a value");
        assert!(!NodeOptions::default().take("-x", &mut [].iter()).unwrap(), "not one of them");
    }

    #[test]
    fn a_node_caps_its_lanes_per_class_as_serve_does() {
        assert_eq!(class_lanes_of(None, 8), Ok([0; 4]), "uncapped");
        assert_eq!(class_lanes_of(Some("background=2,agent=6"), 8), Ok([0, 0, 6, 2]));
        let e = class_lanes_of(Some("agent=9"), 8).unwrap_err();
        assert!(e.starts_with("--class-lanes: ") && e.contains("exceeds"), "{e}");
        let state = NodeState::scratch().unwrap();
        let capped = |spec: &str| NodeOptions { lanes: 2, class_lanes: Some(spec.into()), ..NodeOptions::default() };
        assert_eq!(capped("agent=1").class_lanes(), Ok([0, 0, 1, 0]));
        load_mock_node(&capped("agent=1"), &state).expect("a capped node loads");
        let e = load_mock_node(&capped("agent=3"), &state).map(|_| ()).unwrap_err();
        assert!(e.starts_with("--class-lanes: "), "refused before the engine opens: {e}");
        let mut opts = NodeOptions::default();
        let args = ["--class-lanes".to_string(), "background=2".to_string()];
        let mut it = args[1..].iter();
        assert!(opts.take(&args[0], &mut it).unwrap(), "read with the flags a node takes from serve");
        assert_eq!(opts.class_lanes.as_deref(), Some("background=2"));
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
    fn a_node_given_several_models_loads_each_on_its_own_engine_and_state() {
        let home = std::env::temp_dir().join(format!("superfluid-node-several-{}-{}", std::process::id(), rand_tag()));
        let state = NodeState::open(&home, "box-ab", true).unwrap();
        let opts = NodeOptions { runtime: Some("mock".into()), lanes: 2, ..NodeOptions::default() };
        let one = load_nodes(&["only".into()], &opts, &state).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].model_name, "mock", "one mock model is served as the mock engine always was");
        drop(one);

        let models: Vec<String> = vec!["model-a".into(), "org/model-b".into()];
        let both = load_nodes(&models, &opts, &state).unwrap();
        let names: Vec<&str> = both.iter().map(|n| n.model_name.as_str()).collect();
        assert_eq!(names, ["model-a", "org/model-b"]);
        let other = state.of_model("org/model-b").unwrap();
        assert_ne!(other.wal, state.wal, "each model keeps its own session store");
        assert!(other.wal.starts_with(home.join("fleet/nodes/box-ab/models")), "{}", other.wal.display());
        assert_eq!(other.park, Some(other.wal.parent().unwrap().join("park")), "and parks its own prefixes");

        let twice = load_nodes(&["model-a".into(), "model-a".into()], &opts, &state).err().unwrap_or_default();
        assert!(twice.contains("given twice"), "{twice}");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_node_given_an_id_keeps_its_name_for_the_head_to_take_on() {
        let own = || "Qwen/Qwen3-0.6B-GGUF".to_string();
        assert_eq!(name_among(&["llama".into()], own(), true), "Qwen/Qwen3-0.6B-GGUF", "not renamed to the head's one model");
        assert_eq!(name_among(&["Qwen/Qwen3-0.6B-GGUF".into()], own(), true), "Qwen/Qwen3-0.6B-GGUF");
        assert_eq!(name_among(&[], own(), true), "Qwen/Qwen3-0.6B-GGUF");
    }
}
