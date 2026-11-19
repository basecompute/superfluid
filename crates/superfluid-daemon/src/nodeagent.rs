//! The node-agent.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use superfluid_abi::finish;
use superfluid_linkf::{AgentAction, AgentSession, Endpoint, LinkFError, Liveness};
use superfluid_proto::handshake::{CapacitySummary, HelloAckF};
use superfluid_proto::linkf::{self, msg_type, LinkFLimits, TokenEventPayload};

use crate::fleet::{decode_params, decode_tokens};
use crate::wal::GenParams;
use crate::{Daemon, DaemonError};

pub const FINISH_EXPIRED: u32 = u32::MAX;

pub const FINISH_REFUSED: u32 = u32::MAX - 1;

pub const FINISH_DECLINED: u32 = u32::MAX - 2;

pub const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// How long the head may be silent on a connection before the node drops it: six of the
/// head's heartbeats.
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

struct AgentShared {
    agent: AgentSession,
    /// The connection each session was placed over, the one the head answers its renewals on.
    owners: HashMap<u64, u64>,
    contexts: HashMap<u64, Vec<u32>>,
    /// Each session's sampling and the QoS class the head gave it, which its generations run in.
    params: HashMap<u64, (GenParams, u8)>,
    /// The model each session runs on, by its place in [`NodeAgent::models`]; none for one the
    /// head named a model this node does not serve.
    models: HashMap<u64, Option<usize>>,
    last_load_sample: Option<(u64, u64, u64)>,
    decode_rate_ewma: u32,
    prefill_rate_ewma: u32,
    busy: Busy,
}

/// Time with at least one generation running, which the node's rates are measured over: a
/// rate over wall time would read an idle node as a slow one.
#[derive(Default)]
struct Busy {
    running: u32,
    since: Option<Instant>,
    total: Duration,
}

impl Busy {
    fn start(&mut self) {
        if self.running == 0 {
            self.since = Some(Instant::now());
        }
        self.running += 1;
    }

    fn stop(&mut self) {
        self.running = self.running.saturating_sub(1);
        if self.running == 0 {
            if let Some(t) = self.since.take() {
                self.total += t.elapsed();
            }
        }
    }

    fn ms(&self) -> u64 {
        (self.total + self.since.map(|t| t.elapsed()).unwrap_or_default()).as_millis() as u64
    }
}

/// What a node is told of its head's models at each handshake ([`NodeAgent::with_on_head`]).
pub type HeadHook = Arc<dyn Fn(Option<&superfluid_proto::handshake::Served>) + Send + Sync>;

/// One model a node serves, on an engine and daemon of its own.
#[derive(Clone)]
struct Model {
    daemon: Arc<Daemon>,
    /// The name the head matches it on.
    name: String,
    /// The id it is pulled by, told to the head, which announces it to nodes that follow.
    id: Option<String>,
    /// Its tokenizer digest, told to the head (0: not known).
    tokenizer: u64,
    /// Its tokenizer, for a head that has no copy of the model to export and encode with.
    tokenizer_impl: Option<Arc<dyn superfluid_engine::Tokenizer>>,
}

pub struct NodeAgent {
    /// The models served, the first first: a head before [`superfluid_linkf::driver::NAMES_MODELS`]
    /// is offered only it, and everything it asks runs on it.
    models: Vec<Model>,
    /// Whether this connection's head names the model of each session and request, and so is
    /// offered every model.
    names_models: bool,
    shared: Arc<Mutex<AgentShared>>,
    identity: String,
    /// What the node tells the head of itself ([`superfluid_proto::handshake::FOLLOWS_HEAD`],
    /// [`superfluid_proto::handshake::BATCH_INVARIANT`]).
    feature_flags: u64,
    start: Instant,
    /// Which start of the daemon's counters this is, told to the head with them
    /// ([`linkf::NodeStatsMsg::run`]); every connection of this node shares it.
    run: u64,
    handshake_timeout: Duration,
    idle_timeout: Duration,
    auth: Vec<u8>,
    max_lanes: u32,
    last_status_ms: u64,
    conn: u64,
    /// When this connection last asked for each lease it has not had granted yet.
    asked: HashMap<u64, Duration>,
    /// What each generation the head is about to start asks beyond its sampling.
    opts: HashMap<(u64, u64), linkf::GenerateOptsMsg>,
    /// Blobs the head is sending: the SHA-256 each should have, and the bytes so far.
    incoming: HashMap<u64, ([u8; 32], Vec<u8>)>,
    /// Called when a head this node dialed asks for a connection for short calls.
    on_control_ask: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Told, at each handshake, what the head stated it serves over the link.
    on_head: Option<HeadHook>,
}

/// The largest blob a node takes from its head.
const MAX_BLOB: u64 = 1 << 30;

static NEXT_CONN: AtomicU64 = AtomicU64::new(1);

const STATUS_INTERVAL_MS: u64 = 2_000;

/// A value no earlier start of a node's counters is likely to have had: a node agent is made
/// with its daemon, so a new one means counters from 0.
fn new_run() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
    h.write_u32(std::process::id());
    h.finish()
}

/// A renewal the head has not answered is asked for again after this long.
const RENEW_RETRY: Duration = Duration::from_secs(1);

/// How long one read of the link waits while a generation runs, which bounds how late its
/// tokens leave.
const POLL_RUNNING: Duration = Duration::from_millis(2);

const POLL_IDLE: Duration = Duration::from_millis(20);

/// Tokens are sent in batches of this many, or after [`EMIT_COALESCE_MS`]; a generation's first
/// tokens go at once, so the head's first token waits on nothing.
const EMIT_COALESCE_TOKENS: usize = 16;

const EMIT_COALESCE_MS: u64 = 50;

impl NodeAgent {
    pub fn new(daemon: Arc<Daemon>, identity: impl Into<String>, model_name: impl Into<String>) -> NodeAgent {
        NodeAgent {
            max_lanes: 0,
            last_status_ms: 0,
            conn: NEXT_CONN.fetch_add(1, Ordering::Relaxed),
            asked: HashMap::new(),
            opts: HashMap::new(),
            incoming: HashMap::new(),
            on_control_ask: None,
            on_head: None,
            models: vec![Model { daemon, name: model_name.into(), id: None, tokenizer: 0, tokenizer_impl: None }],
            names_models: false,
            shared: Arc::new(Mutex::new(AgentShared {
                agent: AgentSession::new(),
                owners: HashMap::new(),
                contexts: HashMap::new(),
                params: HashMap::new(),
                models: HashMap::new(),
                last_load_sample: None,
                decode_rate_ewma: 0,
                prefill_rate_ewma: 0,
                busy: Busy::default(),
            })),
            identity: identity.into(),
            feature_flags: superfluid_proto::handshake::BATCH_INVARIANT,
            start: Instant::now(),
            run: new_run(),
            handshake_timeout: DEFAULT_HANDSHAKE_TIMEOUT,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
            auth: Vec::new(),
        }
    }

    pub fn with_max_lanes(mut self, lanes: u32) -> NodeAgent {
        self.max_lanes = lanes;
        self
    }

    pub fn connection(&self) -> NodeAgent {
        NodeAgent {
            models: self.models.clone(),
            names_models: false,
            shared: Arc::clone(&self.shared),
            identity: self.identity.clone(),
            feature_flags: self.feature_flags,
            start: self.start,
            run: self.run,
            handshake_timeout: self.handshake_timeout,
            idle_timeout: self.idle_timeout,
            auth: self.auth.clone(),
            max_lanes: self.max_lanes,
            last_status_ms: 0,
            conn: NEXT_CONN.fetch_add(1, Ordering::Relaxed),
            asked: HashMap::new(),
            opts: HashMap::new(),
            incoming: HashMap::new(),
            on_control_ask: self.on_control_ask.clone(),
            on_head: self.on_head.clone(),
        }
    }

    /// `idle` is how long the head may be silent before a connection is dropped; zero keeps
    /// every connection. Keep it well above the head's heartbeat, 5 s.
    pub fn with_timeouts(mut self, handshake: Duration, idle: Duration) -> NodeAgent {
        self.handshake_timeout = handshake;
        self.idle_timeout = idle;
        self
    }

    /// Serves `model_name` too, on `daemon`, its own engine: a head that names the model of each
    /// session and request ([`superfluid_linkf::driver::NAMES_MODELS`]) is offered it after the
    /// models before it, and each session for it runs there; a head before that is offered the
    /// first model only. What [`Self::with_model_id`] and the tokenizer setters tell from here on
    /// is this model's.
    pub fn with_model(mut self, daemon: Arc<Daemon>, model_name: impl Into<String>) -> NodeAgent {
        self.models.push(Model { daemon, name: model_name.into(), id: None, tokenizer: 0, tokenizer_impl: None });
        self
    }

    fn last(&mut self) -> &mut Model {
        self.models.last_mut().expect("a node serves a model")
    }

    /// Tells each head the id the model is pulled by, `org/model:tag`.
    pub fn with_model_id(mut self, id: impl Into<String>) -> NodeAgent {
        self.last().id = Some(id.into());
        self
    }

    /// Tells each head this node follows its first model, so the head does not take on the
    /// model the node serves.
    pub fn following(mut self) -> NodeAgent {
        self.feature_flags |= superfluid_proto::handshake::FOLLOWS_HEAD;
        self
    }

    /// A node that does not state `flag` to its head, as one released before it does; for
    /// testing a head against such a node.
    #[doc(hidden)]
    pub fn without_feature(mut self, flag: u64) -> NodeAgent {
        self.feature_flags &= !flag;
        self
    }

    /// Tells each head the model's tokenizer by `digest` ([`crate::fleet::tokenizer_digest`]),
    /// so one whose tokenizer differs places nothing here.
    pub fn with_tokenizer(mut self, digest: u64) -> NodeAgent {
        self.last().tokenizer = digest;
        self
    }

    /// Serves the model's tokenizer to a head: its digest, all it says without encoding, and
    /// the text it encodes, so a head with no copy of the model tokenizes as this node does.
    pub fn with_tokenizer_of(mut self, tok: Arc<dyn superfluid_engine::Tokenizer>) -> NodeAgent {
        let model = self.last();
        model.tokenizer = crate::fleet::tokenizer_digest(&*tok);
        model.tokenizer_impl = Some(tok);
        self
    }

    /// The models this connection's head is offered: every one to a head that names the model
    /// of each request, the first to one before.
    fn served(&self) -> &[Model] {
        if self.names_models {
            &self.models
        } else {
            &self.models[..1]
        }
    }

    /// The model the head means by `name`: on a node serving one, or to a head that names none,
    /// the first; otherwise the one served by that name, the first for an empty one, and none
    /// for a name this node does not serve.
    fn model_of(&self, name: &str) -> Option<usize> {
        if self.served().len() == 1 || name.is_empty() {
            return Some(0);
        }
        self.models.iter().position(|m| m.name == name)
    }

    fn first(&self) -> &Arc<Daemon> {
        &self.models[0].daemon
    }

    /// For a node that dials its head: `dial` is called when the head asks for one connection
    /// more, which it keeps for short calls; it may be called more than once.
    /// Tells `hook`, at each handshake, what the head stated it serves in its hello, over a
    /// link the token authenticates: `None` from a head that states nothing.
    pub fn with_on_head(mut self, hook: impl Fn(Option<&superfluid_proto::handshake::Served>) + Send + Sync + 'static) -> NodeAgent {
        self.on_head = Some(Arc::new(hook));
        self
    }

    pub fn with_on_control_ask(mut self, dial: impl Fn() + Send + Sync + 'static) -> NodeAgent {
        self.on_control_ask = Some(Arc::new(dial));
        self
    }

    pub fn with_auth(mut self, auth: Vec<u8>) -> NodeAgent {
        self.auth = auth;
        self
    }

    fn sample_load(&mut self) -> (u64, u64, u32, u32, u32, u32, u32, u64) {
        let mut sh = self.shared.lock().expect("agent shared");
        // The node's load is every model's: they share its device, and its connections.
        let sum = |of: &dyn Fn(&crate::scheduler::SchedStats) -> u64| -> u64 {
            self.models.iter().map(|m| of(&m.daemon.sched_stats())).sum()
        };
        let now = self.start.elapsed().as_millis() as u64;
        let decoded = sum(&|s| s.decode_tokens.load(Ordering::Relaxed));
        let prefilled = sum(&|s| s.prefill_tokens.load(Ordering::Relaxed));

        let smooth = |prev: u32, inst: u32| -> u32 {
            match prev {
                0 => inst,
                p => ((p as u64 * 3 + inst as u64) / 4) as u32,
            }
        };
        const MIN_BUSY_MS: u64 = 200;
        let busy = sh.busy.ms();
        match sh.last_load_sample {
            Some((prev_busy, prev_decoded, prev_prefilled)) => {
                let dt_ms = busy.saturating_sub(prev_busy);
                if dt_ms >= MIN_BUSY_MS {
                    let rate = |n: u64, prev: u64| (n.saturating_sub(prev).saturating_mul(1000) / dt_ms) as u32;
                    let (d, pf) = (rate(decoded, prev_decoded), rate(prefilled, prev_prefilled));
                    if d > 0 {
                        sh.decode_rate_ewma = smooth(sh.decode_rate_ewma, d);
                    }
                    if pf > 0 {
                        sh.prefill_rate_ewma = smooth(sh.prefill_rate_ewma, pf);
                    }
                    sh.last_load_sample = Some((busy, decoded, prefilled));
                }
            }
            None => sh.last_load_sample = Some((busy, decoded, prefilled)),
        }
        let (rate, prefill_rate) = (sh.decode_rate_ewma, sh.prefill_rate_ewma);

        let queue = sum(&|s| s.queue_depth.iter().map(|q| q.load(Ordering::Relaxed)).sum());
        (
            sum(&|s| s.pool_blocks_used.load(Ordering::Relaxed)),
            sum(&|s| s.pool_blocks_total.load(Ordering::Relaxed)),
            sum(&|s| s.lanes_active.load(Ordering::Relaxed)) as u32,
            self.max_lanes,
            queue as u32,
            rate,
            prefill_rate,
            now,
        )
    }

    fn capacity(&mut self) -> CapacitySummary {
        let (kv_used, kv_total, lanes, max_lanes, queue, rate, prefill_rate, at) =
            self.sample_load();
        CapacitySummary {
            gpu_memory_bytes: 0,
            gpu_memory_free_bytes: 0,
            host_memory_bytes: 0,
            active_sessions: self.shared.lock().expect("agent shared").contexts.len() as u32,
            max_sessions: u32::MAX,
            max_context_tokens: self.first().max_stream_tokens(),
            kv_blocks_used: kv_used,
            kv_blocks_total: kv_total,
            lanes_active: lanes,
            max_lanes,
            queue_depth: queue,
            decode_tokens_per_s: rate,
            prefill_tokens_per_s: prefill_rate,
            sampled_at_ms: at,
        }
    }

    fn send_status(&mut self, endpoint: &mut Endpoint) {
        let capacity = self.capacity();
        let msg = linkf::NodeStatusMsg {
            mem: superfluid_proto::linkw::MemCountersMsg {
                allocated_bytes: 0,
                host_retained_bytes: 0,
                pool_blocks_total: capacity.kv_blocks_total,
                pool_blocks_used: capacity.kv_blocks_used,
                pool_bytes_evictable: 0,
                tentative_bytes: 0,
            },
            capacity,
            loadable_models: self.served().iter().map(|m| m.name.clone()).collect(),
            stats: Some(self.stats()),
        };
        let _ = endpoint.send(msg_type::NODE_STATUS, &msg);
        self.last_status_ms = self.start.elapsed().as_millis() as u64;
    }

    /// The node's scheduler counters, for the head's `/metrics`: every model's summed, as its
    /// load is, since the head lists a node's series once, under all the models it serves.
    fn stats(&self) -> linkf::NodeStatsMsg {
        let models: Vec<Arc<crate::scheduler::SchedStats>> = self.models.iter().map(|m| m.daemon.sched_stats()).collect();
        let sum = |of: fn(&crate::scheduler::SchedStats) -> &AtomicU64| -> u64 {
            models.iter().map(|s| of(s).load(Ordering::Relaxed)).fold(0, u64::saturating_add)
        };
        linkf::NodeStatsMsg {
            run: self.run,
            prefill_tokens: sum(|s| &s.prefill_tokens),
            warm_prefix_tokens: sum(|s| &s.warm_prefix_tokens),
            cold_admissions: sum(|s| &s.cold_admissions),
            unservable_cold_admissions: sum(|s| &s.unservable_cold_admissions),
            unservable_park_refusals: sum(|s| &s.unservable_park_refusals),
            preemptions: sum(|s| &s.preemptions),
            admits_deferred: sum(|s| &s.admits_deferred),
            starvation_grants: sum(|s| &s.starvation_grants),
            os_pressure_events: sum(|s| &s.os_pressure_events),
            pressure_evictions: sum(|s| &s.pressure_evictions),
            pressure_bytes_evicted: sum(|s| &s.pressure_bytes_evicted),
            pins_yielded: sum(|s| &s.pins_yielded),
            pins_expired: sum(|s| &s.pins_expired),
            worker_respawns: sum(|s| &s.worker_respawns),
            spec_proposed: sum(|s| &s.spec_proposed),
            spec_accepted: sum(|s| &s.spec_accepted),
            parks_lossless: sum(|s| &s.parks_lossless),
            parks_lossy: sum(|s| &s.parks_lossy),
            resumes: sum(|s| &s.resumes),
        }
    }

    fn ackf(&mut self) -> HelloAckF {
        // In the models' order, empty where one has no id, and none past the last that has one.
        let mut model_ids: Vec<String> = self.served().iter().map(|m| m.id.clone().unwrap_or_default()).collect();
        while model_ids.last().is_some_and(String::is_empty) {
            model_ids.pop();
        }
        HelloAckF {
            chosen_version: superfluid_linkf::driver::PROTO_VERSION,
            node_identity: self.identity.clone(),
            capacity: self.capacity(),
            loadable_models: self.served().iter().map(|m| m.name.clone()).collect(),
            feature_flags: self.feature_flags,
            model_ids,
            limits: LinkFLimits {
                max_frame: superfluid_proto::MAX_FRAME,
                transfer_window: 8,
                max_transfer: 1 << 26,
                transfer_budget: 1 << 28,
            },
        }
    }

    pub fn serve(&mut self, mut endpoint: Endpoint) -> Result<(), DaemonError> {
        // The hello says the version the head speaks, which says what it is offered.
        let hello = if self.handshake_timeout.is_zero() {
            endpoint.read_hello()
        } else {
            endpoint.read_hello_within(self.handshake_timeout)
        };
        let hs = hello.and_then(|hello| {
            self.names_models = Endpoint::shared_version(&hello).is_some_and(|v| v >= superfluid_linkf::driver::NAMES_MODELS);
            let ack = self.ackf();
            endpoint.answer_hello(&hello, &ack, &self.auth)
        });
        match hs {
            Ok(()) => {}
            Err(LinkFError::TimedOut) => {
                tracing::warn!("node-agent dropped a silent connection (no Hello)");
                return Ok(());
            }
            Err(LinkFError::Unauthorized) => return Err(LinkFError::Unauthorized.into()),
            Err(e) => return Err(e.into()),
        }
        if let Some(hook) = &self.on_head {
            hook(endpoint.head_serves());
        }
        let heartbeat = match self.idle_timeout {
            t if t.is_zero() => Liveness::DEFAULT.heartbeat,
            t => Liveness::DEFAULT.heartbeat.min(t / 3),
        };
        endpoint.keep_alive(Liveness { heartbeat, silence: self.idle_timeout });
        if endpoint.version() >= 3 {
            let models = match self.names_models {
                true => self
                    .models
                    .iter()
                    .map(|m| linkf::NodeModelMsg {
                        name: m.name.clone(),
                        tokenizer: m.tokenizer,
                        max_context_tokens: m.daemon.max_stream_tokens(),
                    })
                    .collect(),
                false => Vec::new(),
            };
            endpoint.send(msg_type::TOKENIZER, &linkf::TokenizerMsg { digest: self.models[0].tokenizer, models })?;
        }
        // Dropped on the way out, which stops a generation still running.
        let mut work = Work::default();
        self.serve_link(&mut endpoint, &mut work)
    }

    fn serve_link(&mut self, endpoint: &mut Endpoint, work: &mut Work) -> Result<(), DaemonError> {
        loop {
            let now = self.start.elapsed();
            self.renew(endpoint, now)?;
            self.fence(now);
            if now.as_millis() as u64 >= self.last_status_ms + STATUS_INTERVAL_MS {
                self.send_status(endpoint);
            }
            let was_busy = work.busy();
            if let Some(run) = work.generation.as_mut() {
                if self.forward(endpoint, run, now)? {
                    work.generation = None;
                }
            }
            work.syncs.retain(|rx| loop {
                match rx.try_recv() {
                    Ok(Synced::Segment(seg)) => {
                        let _ = endpoint.send(msg_type::STATE_SYNC_SEGMENT, &seg);
                    }
                    Ok(Synced::Done(resp)) => {
                        let _ = endpoint.send(msg_type::STATE_SYNC_RESP, &resp);
                        break false;
                    }
                    Err(mpsc::TryRecvError::Empty) => break true,
                    Err(mpsc::TryRecvError::Disconnected) => break false,
                }
            });
            if let Some(stage) = &work.stage {
                match stage.try_recv() {
                    Ok(resp) => {
                        work.stage = None;
                        endpoint.send(msg_type::STAGE_FORWARD_RESP, &resp)?;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => work.stage = None,
                }
            }
            if was_busy && !work.busy() {
                endpoint.poll_every(POLL_IDLE)?;
            }
            let frames = match endpoint.pump() {
                Ok(f) => f,
                Err(LinkFError::Closed) => return Ok(()),
                Err(LinkFError::TimedOut) => {
                    tracing::warn!(
                        quiet_ms = endpoint.quiet_for().as_millis() as u64,
                        "node-agent dropped a connection the head went silent on"
                    );
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            };
            for frame in frames {
                self.handle(endpoint, frame, work)?;
            }
            // A head on version 2 sends no heartbeat while it waits on a generation, so only
            // a connection with nothing running is idle.
            if endpoint.version() < 3
                && !work.busy()
                && !self.idle_timeout.is_zero()
                && endpoint.quiet_for() >= self.idle_timeout
            {
                tracing::info!("node-agent dropped an idle connection");
                return Ok(());
            }
        }
    }

    /// Asks the head to renew the leases of the sessions placed over this connection, which is
    /// where it answers them; again after [`RENEW_RETRY`] until a grant comes.
    fn renew(&mut self, endpoint: &mut Endpoint, now: Duration) -> Result<(), DaemonError> {
        let due: Vec<_> = {
            let sh = self.shared.lock().expect("agent shared");
            sh.agent
                .needs_renew(now)
                .into_iter()
                .filter(|r| sh.owners.get(&r.session_id) == Some(&self.conn))
                .collect()
        };
        for renew in due {
            if self.asked.get(&renew.session_id).is_some_and(|&at| now.saturating_sub(at) < RENEW_RETRY) {
                continue;
            }
            endpoint.send(msg_type::LEASE_RENEW, &renew)?;
            self.asked.insert(renew.session_id, now);
        }
        Ok(())
    }

    fn fence(&mut self, now: Duration) {
        let mut sh = self.shared.lock().expect("agent shared");
        for action in sh.agent.poll(now) {
            match action {
                AgentAction::SelfFence { session_id } => {
                    tracing::warn!(session_id, "node-agent self-fenced on lease expiry");
                    sh.contexts.remove(&session_id);
                    sh.params.remove(&session_id);
                    sh.owners.remove(&session_id);
                    sh.models.remove(&session_id);
                }
            }
        }
    }

    fn handle(
        &mut self,
        endpoint: &mut Endpoint,
        frame: superfluid_proto::Frame,
        work: &mut Work,
    ) -> Result<(), DaemonError> {
        let now = self.start.elapsed();
        match frame.msg_type {
            msg_type::RECONCILE => {
                let m: linkf::ReconcileMsg = postcard::from_bytes(&frame.payload)?;
                let mut sh = self.shared.lock().expect("agent shared");
                let (ack, retransmissions) = sh.agent.on_reconcile(&m, now);
                for s in ack.sessions.iter().filter(|s| s.held) {
                    sh.owners.insert(s.session_id, self.conn);
                }
                drop(sh);
                endpoint.send(msg_type::RECONCILE_ACK, &ack)?;
                for batch in retransmissions {
                    endpoint.send(msg_type::TOKEN_EVENTS, &batch)?;
                }
            }
            msg_type::SESSION_ASSIGN => {
                let m = linkf::SessionAssignMsg::decode(&frame.payload)?;
                let model = self.model_of(&m.model);
                if model.is_none() {
                    tracing::warn!(session = m.session_id, model = %m.model, "the head placed a session for a model this node does not serve; its generations are declined");
                }
                let mut sh = self.shared.lock().expect("agent shared");
                if sh.agent.on_assign(&m, now) {
                    sh.params.insert(m.session_id, (decode_params(&m.params), m.qos_class.min(crate::qos::BACKGROUND_AGENT)));
                    sh.models.insert(m.session_id, model);
                    sh.owners.insert(m.session_id, self.conn);
                    self.asked.remove(&m.session_id);
                }
            }
            msg_type::SESSION_REVOKE => {
                let m: linkf::SessionRevokeMsg = postcard::from_bytes(&frame.payload)?;
                let mut sh = self.shared.lock().expect("agent shared");
                if sh.agent.on_revoke(&m) {
                    sh.contexts.remove(&m.session_id);
                    sh.params.remove(&m.session_id);
                    sh.owners.remove(&m.session_id);
                    sh.models.remove(&m.session_id);
                    self.asked.remove(&m.session_id);
                }
            }
            msg_type::APPEND_CONTENT => {
                let m: linkf::AppendContentMsg = postcard::from_bytes(&frame.payload)?;
                let mut sh = self.shared.lock().expect("agent shared");
                if sh.agent.on_append(m.session_id, m.epoch, m.command_id) {
                    let tokens = decode_tokens(&m.blocks)?;
                    sh.contexts.insert(m.session_id, tokens);
                }
            }
            msg_type::GENERATE_REQ => {
                let m: linkf::GenerateReqMsg = postcard::from_bytes(&frame.payload)?;
                let model = {
                    let sh = self.shared.lock().expect("agent shared");
                    // A session placed before this node told its models is the first's.
                    sh.models.get(&m.session_id).copied().unwrap_or(Some(0))
                };
                let declined = work.generation.is_some()
                    || model.is_none()
                    || !self.shared.lock().expect("agent shared").agent.on_generate(&m);
                if declined {
                    tracing::warn!(
                        session = m.session_id,
                        generation = m.generation_id,
                        epoch = m.epoch,
                        "declining a generate (unknown session or model, superseded epoch, \
                         replayed command, or one already running on this connection); \
                         reporting it rather than going silent"
                    );
                    let msg = linkf::TokenEventsMsg {
                        events: vec![linkf::TokenEvent {
                            session_id: m.session_id,
                            epoch: m.epoch,
                            generation_id: m.generation_id,
                            event_seq: 0,
                            payload: TokenEventPayload::Finish {
                                reason: FINISH_DECLINED,
                            },
                        }],
                    };
                    endpoint.send(msg_type::TOKEN_EVENTS, &msg)?;
                } else {
                    work.generation = Some(self.start_generation(&m, model.expect("checked above")));
                    endpoint.poll_every(POLL_RUNNING)?;
                }
            }
            msg_type::CONTROL_ASK => {
                if let Some(dial) = &self.on_control_ask {
                    dial();
                }
            }
            // The head removed this node and closes the link; whoever serves it says so.
            msg_type::REMOVED => return Err(LinkFError::Removed.into()),
            msg_type::TOKENIZER_EXPORT_REQ => {
                let m = linkf::TokenizerExportReqMsg::decode(&frame.payload)?;
                endpoint.send(msg_type::TOKENIZER_EXPORT, &self.export(m.request_id, self.model_of(&m.model)))?;
            }
            msg_type::TOKENIZE_REQ => {
                let m = linkf::TokenizeReqMsg::decode(&frame.payload)?;
                let tok = self.model_of(&m.model).and_then(|i| self.models[i].tokenizer_impl.as_ref());
                let tokens = tok.and_then(|t| match &m.text {
                    linkf::TokenizeMsg::Encode(text) => Some(t.encode(text)),
                    linkf::TokenizeMsg::Plain(text) => t.encode_plain(text),
                    linkf::TokenizeMsg::Pieces(pieces) => {
                        let pieces: Vec<(&str, bool)> = pieces.iter().map(|(p, s)| (p.as_str(), *s)).collect();
                        t.encode_pieces(&pieces)
                    }
                });
                endpoint.send(msg_type::TOKENIZE_RESP, &linkf::TokenizeRespMsg { request_id: m.request_id, tokens })?;
            }
            msg_type::GENERATE_OPTS => {
                let m = linkf::GenerateOptsMsg::decode(&frame.payload)?;
                self.opts.insert((m.session_id, m.generation_id), m);
            }
            msg_type::XFER_BEGIN => {
                let m: linkf::XferBeginMsg = postcard::from_bytes(&frame.payload)?;
                if m.total_len > MAX_BLOB {
                    return Err(DaemonError::Protocol("the head sent a blob larger than a node takes"));
                }
                self.incoming.insert(m.transfer_id, (m.content_hash, Vec::with_capacity(m.total_len as usize)));
            }
            msg_type::XFER_CHUNK => {
                let m: linkf::XferChunkMsg = postcard::from_bytes(&frame.payload)?;
                if let Some((_, bytes)) = self.incoming.get_mut(&m.transfer_id) {
                    if m.offset != bytes.len() as u64 || bytes.len() as u64 + m.data.len() as u64 > MAX_BLOB {
                        return Err(DaemonError::Protocol("a blob's pieces came out of order"));
                    }
                    bytes.extend_from_slice(&m.data);
                }
            }
            msg_type::XFER_END => {
                let m: linkf::XferEndMsg = postcard::from_bytes(&frame.payload)?;
                if let Some((want, bytes)) = self.incoming.remove(&m.transfer_id) {
                    let want: String = want.iter().map(|b| format!("{b:02x}")).collect();
                    if crate::media::content_hash(&bytes) != want {
                        return Err(DaemonError::Protocol("a blob from the head is not what its hash says"));
                    }
                    // Kept by each model's daemon, as the request it comes for may run on any; a blob
                    // this node cannot keep fails the request that brings it, not the link.
                    for model in &self.models {
                        if let Err(e) = model.daemon.media_put(&bytes) {
                            tracing::warn!("node-agent cannot keep a blob from the head: {e}");
                        }
                    }
                }
            }
            msg_type::STATE_SYNC_REQ => {
                let mut m = linkf::StateSyncReqF::decode(&frame.payload)?;
                let Some(model) = self.model_of(&m.model) else {
                    let result = Err(format!("this node does not serve {}", m.model));
                    endpoint.send(msg_type::STATE_SYNC_RESP, &linkf::StateSyncRespF { request_id: m.request_id, result })?;
                    return Ok(());
                };
                let daemon = Arc::clone(&self.models[model].daemon);
                Self::localize(&daemon, &mut m.req);
                let (tx, rx) = mpsc::channel();
                std::thread::Builder::new()
                    .name("node-state-sync".into())
                    .spawn(move || {
                        let request_id = m.request_id;
                        let streamed = matches!(m.req, superfluid_proto::linkw::StateSyncReqMsg::Transcribe { stream: true, .. });
                        let capabilities = matches!(m.req, superfluid_proto::linkw::StateSyncReqMsg::Capabilities);
                        let result = if streamed {
                            let (seg_tx, seg_rx) = mpsc::channel();
                            let told = tx.clone();
                            let tell = std::thread::spawn(move || {
                                for segment in seg_rx {
                                    let _ = told.send(Synced::Segment(linkf::StateSyncSegmentF { request_id, segment }));
                                }
                            });
                            let result = daemon.state_sync_stream(m.req, seg_tx);
                            let _ = tell.join();
                            result
                        } else {
                            daemon.state_sync(m.req)
                        };
                        let result = match result {
                            Ok(superfluid_proto::linkw::StateSyncOkMsg::Capabilities { json }) if capabilities => {
                                Ok(superfluid_proto::linkw::StateSyncOkMsg::Capabilities { json: as_served(&daemon, json) })
                            }
                            other => other,
                        };
                        let result = result.map_err(crate::fleet_daemon::message);
                        let _ = tx.send(Synced::Done(linkf::StateSyncRespF { request_id, result }));
                    })
                    .expect("spawn a node state operation");
                work.syncs.push(rx);
                endpoint.poll_every(POLL_RUNNING)?;
            }
            msg_type::GENERATE_CANCEL => {
                let m: linkf::GenerateCancelMsg = postcard::from_bytes(&frame.payload)?;
                if let Some(run) = work
                    .generation
                    .as_mut()
                    .filter(|r| r.session_id == m.session_id && r.generation_id == m.generation_id)
                {
                    run.cancelled = true;
                    run.cancel.store(true, Ordering::Relaxed);
                }
            }
            msg_type::STAGE_FORWARD_REQ => {
                let m: linkf::StageForwardReqMsg = postcard::from_bytes(&frame.payload)?;
                if work.stage.is_some() {
                    // The head waits on one stage per connection; another is answered as one
                    // the node could not run.
                    let resp = linkf::StageForwardRespMsg { request_id: m.request_id, hidden: None, token: None };
                    endpoint.send(msg_type::STAGE_FORWARD_RESP, &resp)?;
                } else {
                    work.stage = Some(self.start_stage(m));
                    endpoint.poll_every(POLL_RUNNING)?;
                }
            }
            msg_type::WATERMARK_ACK => {
                let m: linkf::WatermarkAckMsg = postcard::from_bytes(&frame.payload)?;
                self.shared.lock().expect("agent shared").agent.on_watermark_ack(&m);
            }
            msg_type::LEASE_GRANT => {
                let m: linkf::LeaseGrantMsg = postcard::from_bytes(&frame.payload)?;
                self.shared.lock().expect("agent shared").agent.on_lease_grant(&m, now);
                self.asked.remove(&m.session_id);
            }
            _ => {  }
        }
        Ok(())
    }

    /// The tokenizer of `model` as a head without the model takes it; no vocabulary when this
    /// node has none, or does not serve the model.
    fn export(&self, request_id: u64, model: Option<usize>) -> linkf::TokenizerExportMsg {
        let model = model.map(|i| &self.models[i]);
        let Some((t, digest)) = model.and_then(|m| Some((m.tokenizer_impl.as_ref()?, m.tokenizer))) else {
            return linkf::TokenizerExportMsg {
                request_id,
                digest: 0,
                vocab: Vec::new(),
                specials: Vec::new(),
                bos: None,
                eos: 0,
                chat_template: String::new(),
                named_templates: Vec::new(),
                plain: false,
                pieces: false,
            };
        };
        linkf::TokenizerExportMsg {
            request_id,
            digest,
            vocab: (0..t.vocab_size()).map(|i| t.token_bytes(i)).collect(),
            specials: t.special_tokens(),
            bos: t.bos_token(),
            eos: t.eos_token(),
            chat_template: t.chat_template_jinja(),
            named_templates: ["tool_use"].iter().filter_map(|n| Some((n.to_string(), t.chat_template_named(n)?))).collect(),
            plain: t.encode_plain("a").is_some(),
            pieces: t.encode_pieces(&[("a", false)]).is_some(),
        }
    }

    /// A path the head sent as a blob, where the blob sits in `daemon`'s media pool.
    fn local(daemon: &Daemon, path: &str) -> String {
        match path.strip_prefix(linkf::BLOB_PREFIX) {
            Some(hash) if crate::media::MediaPool::is_valid_id(hash) => daemon.media_path(hash).to_string_lossy().into_owned(),
            _ => path.to_string(),
        }
    }

    fn localize(daemon: &Daemon, req: &mut superfluid_proto::linkw::StateSyncReqMsg) {
        use superfluid_proto::linkw::StateSyncReqMsg as R;
        match req {
            R::MediaProbe { image_path } | R::MediaEncode { image_path } => *image_path = Self::local(daemon, image_path),
            R::Transcribe { audio_path, .. } => *audio_path = Self::local(daemon, audio_path),
            _ => {}
        }
    }

    /// Runs a stage forward on a thread of its own, as a generation runs, so a long one does not
    /// silence the connection; on the first model, as a pipeline serves one.
    fn start_stage(&self, m: linkf::StageForwardReqMsg) -> mpsc::Receiver<linkf::StageForwardRespMsg> {
        let (tx, rx) = mpsc::channel();
        let daemon = Arc::clone(self.first());
        std::thread::Builder::new()
            .name("node-stage".into())
            .spawn(move || {
                let (hidden, token) = match daemon.forward_stage(m.tokens, m.start_layer, m.end_layer, m.hidden_in) {
                    Ok(crate::StageOutput::Hidden(h)) => (Some(h), None),
                    Ok(crate::StageOutput::Token(t)) => (None, Some(t)),
                    Err(_) => (None, None),
                };
                let _ = tx.send(linkf::StageForwardRespMsg { request_id: m.request_id, hidden, token });
            })
            .expect("spawn a node stage");
        rx
    }

    /// Starts `req` on a thread of its own, on the engine of the `model`-th model, so this
    /// connection goes on reading the head (cancels, lease grants, heartbeats) and sending status
    /// while it runs.
    fn start_generation(&mut self, req: &linkf::GenerateReqMsg, model: usize) -> Running {
        let daemon = Arc::clone(&self.models[model].daemon);
        let (context, (params, class)) = {
            let sh = self.shared.lock().expect("agent shared");
            (
                sh.contexts.get(&req.session_id).cloned().unwrap_or_default(),
                sh.params.get(&req.session_id).copied().unwrap_or((GenParams::default(), crate::qos::FOREGROUND_AGENT)),
            )
        };
        let max_new = req.budgets.max_new_tokens.min(u32::MAX as u64) as u32;
        let max_wall_ms = req.budgets.max_wall_ms;
        let opts = self.opts.remove(&(req.session_id, req.generation_id));
        self.opts.clear();
        // Asked only of a node that states it can, as a single server runs one: alone.
        let batch_invariant = opts.as_ref().is_some_and(|o| o.batch_invariant);
        let media: Vec<(u64, std::path::PathBuf, u32)> = opts
            .iter()
            .flat_map(|o| &o.media)
            .map(|(offset, hash, n)| (*offset, std::path::PathBuf::from(Self::local(&daemon, &format!("{}{hash}", linkf::BLOB_PREFIX))), *n))
            .collect();
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, out) = mpsc::channel();
        let (shared, stop) = (Arc::clone(&self.shared), Arc::clone(&cancel));
        shared.lock().expect("agent shared").busy.start();
        std::thread::Builder::new()
            .name("node-generation".into())
            .spawn(move || {
                let generated = match Extras::of(&daemon, opts.as_ref()) {
                    Ok(extras) => {
                        let lp = tx.clone();
                        let generated = daemon.generate_tokens_streaming_lp(
                            context,
                            params,
                            max_new,
                            max_wall_ms,
                            class,
                            batch_invariant,
                            extras.extras.clone(),
                            media,
                            |batch| {
                                if stop.load(Ordering::Relaxed) || tx.send(Out::Tokens(batch.to_vec())).is_err() {
                                    return std::ops::ControlFlow::Break(());
                                }
                                std::ops::ControlFlow::Continue(())
                            },
                            |items| {
                                let _ = lp.send(Out::Logprobs(items.to_vec()));
                            },
                        );
                        extras.free(&daemon);
                        generated
                    }
                    Err(e) => Err(e),
                };
                shared.lock().expect("agent shared").busy.stop();
                let _ = tx.send(Out::Done(generated));
            })
            .expect("spawn a node generation");
        Running {
            session_id: req.session_id,
            generation_id: req.generation_id,
            max_new,
            cancel,
            cancelled: false,
            out,
            pending: Vec::with_capacity(EMIT_COALESCE_TOKENS),
            last_flush: self.start.elapsed(),
            sent: false,
        }
    }

    /// Sends what `run` produced since the last pass, coalesced, and its finish once it ends;
    /// `true` once it has ended, or was fenced and goes silent.
    fn forward(&mut self, endpoint: &mut Endpoint, run: &mut Running, now: Duration) -> Result<bool, DaemonError> {
        let mut done = None;
        loop {
            match run.out.try_recv() {
                Ok(Out::Tokens(t)) => {
                    run.pending.extend_from_slice(&t);
                    if run.pending.len() >= EMIT_COALESCE_TOKENS && !self.flush(endpoint, run, now)? {
                        return Ok(true);
                    }
                }
                Ok(Out::Logprobs(items)) => {
                    let msgs = items
                        .into_iter()
                        .map(|lp| linkf::TokenLogprobMsg {
                            token: lp.token,
                            logprob_bits: lp.logprob.to_bits(),
                            top: lp.top.into_iter().map(|(t, p)| (t, p.to_bits())).collect(),
                        })
                        .collect();
                    let emitted = self.shared.lock().expect("agent shared").agent.emit(
                        run.session_id,
                        run.generation_id,
                        TokenEventPayload::Logprobs(msgs),
                    );
                    match emitted {
                        Some(batch) => endpoint.send(msg_type::TOKEN_EVENTS, &batch)?,
                        None => return Ok(true),
                    }
                }
                Ok(Out::Done(r)) => {
                    done = Some(r);
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    done = Some(Err(DaemonError::Generation("the generation ended without a result".into())));
                    break;
                }
            }
        }
        let due = done.is_some() || !run.sent || now.saturating_sub(run.last_flush).as_millis() as u64 >= EMIT_COALESCE_MS;
        if due && !run.pending.is_empty() && !self.flush(endpoint, run, now)? {
            return Ok(true);
        }
        let mut sh = self.shared.lock().expect("agent shared");
        if sh.agent.is_fenced(run.session_id) != Some(false) {
            return Ok(true);
        }
        let Some(generated) = done else { return Ok(false) };
        let reason = match generated {
            Ok(_) if run.cancelled => finish::CANCELLED,
            Ok((_, _, true)) => FINISH_EXPIRED,
            Ok((tokens, _, _)) if tokens.len() as u32 >= run.max_new => finish::LENGTH,
            Ok(_) => finish::EOS,
            Err(e) if is_client_refusal(&e) => {
                tracing::warn!(
                    session = run.session_id,
                    generation = run.generation_id,
                    error = %e,
                    "refusing a generation the node cannot stage; \
                     the head should have caught this before dispatch"
                );
                FINISH_REFUSED
            }
            Err(e) => return Err(e),
        };
        if let Some(batch) = sh.agent.emit(run.session_id, run.generation_id, TokenEventPayload::Finish { reason }) {
            endpoint.send(msg_type::TOKEN_EVENTS, &batch)?;
        }
        Ok(true)
    }

    /// Sends `run`'s pending tokens; `false` when its session is fenced, which sends nothing.
    fn flush(&mut self, endpoint: &mut Endpoint, run: &mut Running, now: Duration) -> Result<bool, DaemonError> {
        run.last_flush = now;
        run.sent = true;
        let tokens = std::mem::take(&mut run.pending);
        let batch = self.shared.lock().expect("agent shared").agent.emit(
            run.session_id,
            run.generation_id,
            TokenEventPayload::Tokens(tokens),
        );
        match batch {
            Some(batch) => {
                endpoint.send(msg_type::TOKEN_EVENTS, &batch)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

/// What a connection has running off its own thread.
#[derive(Default)]
struct Work {
    generation: Option<Running>,
    /// A stage forward's answer, once it has one.
    stage: Option<mpsc::Receiver<linkf::StageForwardRespMsg>>,
    /// State operations' answers, once each has one.
    syncs: Vec<mpsc::Receiver<Synced>>,
}

/// What a state operation has to tell the head: a streamed transcription's segments, then its
/// answer.
enum Synced {
    Segment(linkf::StateSyncSegmentF),
    Done(linkf::StateSyncRespF),
}

impl Work {
    fn busy(&self) -> bool {
        self.generation.is_some() || self.stage.is_some() || !self.syncs.is_empty()
    }
}

/// A generation running on a connection; dropping it stops the generation at its next step.
struct Running {
    session_id: u64,
    generation_id: u64,
    max_new: u32,
    cancel: Arc<AtomicBool>,
    /// The head asked for the stop.
    cancelled: bool,
    out: mpsc::Receiver<Out>,
    pending: Vec<u32>,
    last_flush: Duration,
    /// Tokens have gone to the head.
    sent: bool,
}

impl Drop for Running {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// The engine's capability record, as the head learns what this node does: with whether the node
/// speculates, which the head needs to resolve a request's default temperature
/// (`--spec-max-temperature`) and which the engine's record does not say. A record that is not an
/// object (an engine that describes nothing) is left as it is.
fn as_served(daemon: &Daemon, json: String) -> String {
    let Ok(mut record) = serde_json::from_str::<serde_json::Value>(&json) else { return json };
    let Some(obj) = record.as_object_mut().filter(|_| daemon.speculation().is_some()) else { return json };
    let (section, key) = crate::NODE_SPECULATES;
    match obj.entry(section).or_insert_with(|| serde_json::json!({})).as_object_mut() {
        Some(s) => s.insert(key.into(), true.into()),
        None => return json,
    };
    record.to_string()
}

enum Out {
    Tokens(Vec<u32>),
    Logprobs(Vec<crate::scheduler::TokenLogprob>),
    Done(Result<(Vec<u32>, u32, bool), DaemonError>),
}

/// A generation's extras as its options ask, with the grammar and logit bias compiled on this
/// node's engine and freed once it ends.
struct Extras {
    extras: crate::scheduler::GenExtras,
}

impl Extras {
    fn of(daemon: &Daemon, opts: Option<&linkf::GenerateOptsMsg>) -> Result<Extras, DaemonError> {
        // Each prompt stays in the cache for the conversation's next turn, and with --park on
        // disk, so a restarted node starts it warm too.
        let mut extras = crate::scheduler::GenExtras { keep_prefix: true, ..Default::default() };
        let Some(o) = opts else { return Ok(Extras { extras }) };
        extras.presence_penalty = f32::from_bits(o.presence_penalty_bits);
        extras.frequency_penalty = f32::from_bits(o.frequency_penalty_bits);
        extras.repeat_penalty = f32::from_bits(o.repeat_penalty_bits);
        extras.ignore_eos = o.ignore_eos;
        if let Some(top) = o.top_logprobs {
            extras.want_logprobs = true;
            extras.top_logprobs = top;
        }
        extras.grammar_handle = match &o.grammar {
            Some(linkf::GrammarSpecMsg::JsonSchema(schema)) => daemon.grammar_create(schema)?,
            Some(linkf::GrammarSpecMsg::Structural(tag)) => daemon.grammar_create_structural(tag)?,
            None => 0,
        };
        if let Some((tokens, values)) = &o.logit_bias {
            let values: Vec<f32> = values.iter().map(|b| f32::from_bits(*b)).collect();
            match daemon.logit_bias_create(tokens, &values) {
                Ok(h) => extras.logit_bias_handle = h,
                Err(e) => {
                    if extras.grammar_handle != 0 {
                        daemon.grammar_free(extras.grammar_handle);
                    }
                    return Err(e);
                }
            }
        }
        Ok(Extras { extras })
    }

    fn free(&self, daemon: &Daemon) {
        if self.extras.grammar_handle != 0 {
            daemon.grammar_free(self.extras.grammar_handle);
        }
        if self.extras.logit_bias_handle != 0 {
            daemon.logit_bias_free(self.extras.logit_bias_handle);
        }
    }
}

fn is_client_refusal(e: &DaemonError) -> bool {
    matches!(
        e,
        DaemonError::StreamTooLong { .. } | DaemonError::Constraint(_)
    )
}
