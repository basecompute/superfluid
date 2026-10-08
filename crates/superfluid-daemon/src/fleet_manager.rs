//! Head-side multi-node orchestration.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use superfluid_proto::linkf::Lease;

use crate::fleet::{FleetGeneration, FleetHead};
use crate::wal::GenParams;
use crate::DaemonError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementPolicy {
    LoadAware,
    LeastLoaded,
    RoundRobin,
}

#[derive(Clone, Copy, Default)]
struct NodeLoad {
    kv_used: u64,
    kv_total: u64,
    lanes_active: u32,
    queue_depth: u32,
    decode_tokens_per_s: u32,
    prefill_tokens_per_s: u32,
    at: Option<std::time::Instant>,
}

impl NodeLoad {
    const TTL: std::time::Duration = std::time::Duration::from_secs(10);

    fn fresh(&self) -> bool {
        self.at.is_some_and(|t| t.elapsed() < Self::TTL)
    }

    fn pool_pct(&self) -> Option<u64> {
        (self.fresh() && self.kv_total > 0).then(|| self.kv_used * 100 / self.kv_total)
    }

    /// `typical` stands in for a rate this node has not reported yet: the mean of the nodes
    /// that have, so a node that has served nothing is neither shunned nor flooded.
    fn eta_ms(&self, inflight: u32, prompt_tokens: u64, expected_out: u64, typical: (u64, u64)) -> u64 {
        let ahead = if self.fresh() {
            self.lanes_active as u64 + self.queue_depth as u64 + inflight as u64
        } else {
            inflight as u64
        };
        const NOMINAL_DECODE: u64 = 20;
        const NOMINAL_PREFILL: u64 = 200;
        // A rate is what the node does when busy, so it stays true while the node idles; only the
        // load that piles up ahead of a request goes stale.
        let or = |reported: u32, typical: u64, nominal: u64| match (reported, typical) {
            (0, 0) => nominal,
            (0, t) => t,
            (r, _) => r as u64,
        };
        let decode = or(self.decode_tokens_per_s, typical.0, NOMINAL_DECODE);
        let prefill = or(self.prefill_tokens_per_s, typical.1, NOMINAL_PREFILL);
        let queued = ahead.saturating_mul(expected_out).saturating_mul(1_000) / decode;
        let mine = prompt_tokens.saturating_mul(1_000) / prefill
            + expected_out.saturating_mul(1_000) / decode;
        queued.saturating_add(mine)
    }
}

#[derive(Debug, Clone)]
pub struct NodeSpec {
    pub addr: String,
    pub auth: Vec<u8>,
}

struct NodeConn {
    spec: NodeSpec,
    conns: Vec<Mutex<Option<FleetHead>>>,
    /// How many of `conns` are in use: the count asked for, or the node's lanes once learned.
    open: std::sync::atomic::AtomicUsize,
    cursor: std::sync::atomic::AtomicUsize,
}

impl NodeMeta {
    fn new() -> NodeMeta {
        NodeMeta {
            models: Vec::new(),
            identity: String::new(),
            peer: None,
            down_since: None,
            last_retry: None,
            learned: false,
            max_context_tokens: 0,
            load: NodeLoad::default(),
        }
    }

    fn up(&self) -> bool {
        self.down_since.is_none()
    }
}

impl NodeConn {
    fn checkout_next(&self) -> (usize, std::sync::MutexGuard<'_, Option<FleetHead>>) {
        let open = self.open.load(std::sync::atomic::Ordering::Relaxed).clamp(1, self.conns.len());
        let i = self.cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % open;
        (i, self.conns[i].lock().expect("node conn"))
    }

    fn checkout_at(&self, i: usize) -> std::sync::MutexGuard<'_, Option<FleetHead>> {
        self.conns[i.min(self.conns.len() - 1)].lock().expect("node conn")
    }

    fn reset(&self) {
        for slot in &self.conns {
            if let Ok(mut g) = slot.try_lock() {
                *g = None;
            }
        }
    }
}

#[derive(Clone)]
struct NodeMeta {
    models: Vec<String>,
    identity: String,
    /// Where a joined node dialed from, so a second machine with its name is told apart.
    peer: Option<std::net::IpAddr>,
    /// Set when the node failed; requests pass it over until the background retry connects.
    down_since: Option<std::time::Instant>,
    last_retry: Option<std::time::Instant>,
    learned: bool,
    max_context_tokens: u64,
    load: NodeLoad,
}

#[derive(Debug, Clone)]
struct Placed {
    node: usize,
    params: GenParams,
    qos_class: u8,
    codec: String,
    model: String,
    tokens: Vec<u32>,
    slot: usize,
    epoch: u64,
    parked: bool,
}

struct Shared {
    meta: Vec<NodeMeta>,
    /// Finished sessions to revoke on a node: (node, slot, session, epoch).
    forgets: Vec<(usize, usize, u64, u64)>,
    sessions: BTreeMap<u64, Placed>,
    rr_cursor: usize,
    cancels: BTreeMap<u64, Arc<AtomicBool>>,
    inflight: Vec<u32>,
    claims: BTreeMap<u64, usize>,
}

pub struct FleetManager {
    nodes: std::sync::RwLock<Vec<Arc<NodeConn>>>,
    conns_by_lanes: bool,
    retry_every: std::time::Duration,
    last_keepalive: Mutex<std::time::Instant>,
    shared: Mutex<Shared>,
    policy: PlacementPolicy,
    lease: Lease,
    pool_high_pct: u32,
}

struct InFlight<'a> {
    shared: &'a Mutex<Shared>,
    node: usize,
    session: u64,
}

impl<'a> InFlight<'a> {
    fn claim(shared: &'a Mutex<Shared>, node: usize, session: u64) {
        let mut sh = shared.lock().expect("shared");
        if let Some(prev) = sh.claims.insert(session, node) {
            if prev != node {
                if let Some(n) = sh.inflight.get_mut(prev) {
                    *n = n.saturating_sub(1);
                }
            } else {
                return;
            }
        }
        if let Some(n) = sh.inflight.get_mut(node) {
            *n = n.saturating_add(1);
        }
    }

    fn release(shared: &Mutex<Shared>, session: u64) {
        let mut sh = shared.lock().expect("shared");
        if let Some(node) = sh.claims.remove(&session) {
            if let Some(n) = sh.inflight.get_mut(node) {
                *n = n.saturating_sub(1);
            }
        }
    }

    fn guard(shared: &'a Mutex<Shared>, node: usize, session: u64) -> InFlight<'a> {
        Self::claim(shared, node, session);
        InFlight { shared, node, session }
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        let _ = self.node;
        Self::release(self.shared, self.session);
    }
}

#[derive(Clone, Copy, Default)]
pub struct RequestShape {
    pub prompt_tokens: u64,
    pub expected_out: u64,
}

pub const DEFAULT_POOL_HIGH_PCT: u32 = 90;

/// 0: as many connections to a node as it has lanes, so each lane can be busy at once.
pub const DEFAULT_CONNS_PER_NODE: usize = 0;

const MAX_CONNS_PER_NODE: usize = 64;

/// The address a joined node is listed under: it dialed in, so the head cannot dial it.
const JOINED: &str = "joined:";

/// How often the background retry tries a node that failed.
pub const RETRY_DOWN_AFTER: std::time::Duration = std::time::Duration::from_secs(10);

/// Under the node's idle timeout (120 s by default), so an idle head keeps its connections.
pub const KEEPALIVE_EVERY: std::time::Duration = std::time::Duration::from_secs(15);

#[derive(Debug, Clone)]
pub struct NodeReport {
    pub addr: String,
    pub identity: String,
    pub models: Vec<String>,
    pub max_context_tokens: u64,
    pub max_lanes: u32,
}

impl FleetManager {
    pub fn new(nodes: Vec<NodeSpec>, policy: PlacementPolicy, lease: Lease) -> FleetManager {
        Self::with_pool_high(nodes, policy, lease, DEFAULT_POOL_HIGH_PCT)
    }

    pub fn with_pool_high(
        nodes: Vec<NodeSpec>,
        policy: PlacementPolicy,
        lease: Lease,
        pool_high_pct: u32,
    ) -> FleetManager {
        Self::with_options(nodes, policy, lease, pool_high_pct, DEFAULT_CONNS_PER_NODE)
    }

    pub fn with_options(
        nodes: Vec<NodeSpec>,
        policy: PlacementPolicy,
        lease: Lease,
        pool_high_pct: u32,
        conns_per_node: usize,
    ) -> FleetManager {
        let meta = vec![NodeMeta::new(); nodes.len()];
        let node_count = nodes.len();
        let nodes = nodes
            .into_iter()
            .map(|spec| {
                Arc::new(NodeConn {
                    spec,
                    conns: (0..if conns_per_node == 0 { MAX_CONNS_PER_NODE } else { conns_per_node })
                        .map(|_| Mutex::new(None))
                        .collect(),
                    open: std::sync::atomic::AtomicUsize::new(conns_per_node.max(1)),
                    cursor: std::sync::atomic::AtomicUsize::new(0),
                })
            })
            .collect();
        FleetManager {
            nodes: std::sync::RwLock::new(nodes),
            conns_by_lanes: conns_per_node == 0,
            retry_every: RETRY_DOWN_AFTER,
            last_keepalive: Mutex::new(std::time::Instant::now()),
            shared: Mutex::new(Shared {
                meta,
                sessions: BTreeMap::new(),
                forgets: Vec::new(),
                rr_cursor: 0,
            inflight: vec![0; node_count],
                claims: BTreeMap::new(),
                cancels: BTreeMap::new(),
            }),
            policy,
            lease,
            pool_high_pct: pool_high_pct.clamp(1, 100),
        }
    }

    fn node(&self, i: usize) -> Arc<NodeConn> {
        Arc::clone(&self.nodes.read().expect("nodes")[i])
    }

    fn node_count(&self) -> usize {
        self.nodes.read().expect("nodes").len()
    }

    /// Takes in a node that dialed the head: one seen before, by identity, gets its slot
    /// back; a new one gets a slot of its own. `true` for a new one.
    /// Takes in a connection from a node that dialed the head. The node has a slot per lane it
    /// advertised and fills one per connection it opens, so it runs as many generations at
    /// once as it dialed; a node seen before, by identity, takes its slots back, unless it is
    /// still connected from another address, which is a second machine with the same name.
    /// Returns the node's index, whether it is new, and its open connections.
    pub fn adopt(&self, head: FleetHead, peer: std::net::IpAddr) -> Result<(usize, bool, usize), String> {
        let identity = head.node().node_identity.clone();
        let lanes = (head.node().capacity.max_lanes as usize).max(1);
        let (i, new) = {
            let mut sh = self.shared.lock().expect("shared");
            match sh.meta.iter().position(|m| m.identity == identity) {
                Some(i) => {
                    let node = self.node(i);
                    let connected = node.conns.iter().any(|c| c.lock().expect("node conn").is_some());
                    if connected && sh.meta[i].peer.is_some_and(|p| p != peer) {
                        return Err(format!(
                            "another node named '{identity}' is joined from {}; give this one --identity",
                            sh.meta[i].peer.expect("peer")
                        ));
                    }
                    sh.meta[i].peer = Some(peer);
                    (i, false)
                }
                None => {
                    let mut nodes = self.nodes.write().expect("nodes");
                    nodes.push(Arc::new(NodeConn {
                        spec: NodeSpec { addr: format!("{JOINED}{identity}"), auth: Vec::new() },
                        conns: (0..lanes).map(|_| Mutex::new(None)).collect(),
                        open: std::sync::atomic::AtomicUsize::new(1),
                        cursor: std::sync::atomic::AtomicUsize::new(0),
                    }));
                    // Listed as down until its connection is in place, so no request lands
                    // on an empty slot meanwhile.
                    let mut meta = NodeMeta::new();
                    meta.identity = identity.clone();
                    meta.peer = Some(peer);
                    meta.down_since = Some(std::time::Instant::now());
                    sh.meta.push(meta);
                    sh.inflight.push(0);
                    (nodes.len() - 1, true)
                }
            }
        };
        self.sync_meta(i, &head);
        let node = self.node(i);
        let mut slot = 0;
        for (s, conn) in node.conns.iter().enumerate() {
            if conn.lock().expect("node conn").is_none() {
                slot = s;
                break;
            }
        }
        *node.checkout_at(slot) = Some(head);
        let open = node.conns.iter().filter(|c| c.lock().expect("node conn").is_some()).count();
        node.open.store(open.max(1), std::sync::atomic::Ordering::Relaxed);
        if new {
            // Listed as down only until this connection was in place: not a return.
            self.shared.lock().expect("shared").meta[i].down_since = None;
        } else {
            self.mark_up(i);
        }
        Ok((i, new, open))
    }

    /// The connections a node has open.
    pub fn connections(&self, i: usize) -> usize {
        self.node(i).open.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn load(&self) -> Vec<usize> {
        let sh = self.shared.lock().expect("shared");
        let mut counts = vec![0usize; self.node_count()];
        for p in sh.sessions.values() {
            if !p.parked {
                counts[p.node] += 1;
            }
        }
        counts
    }

    pub fn node_of(&self, session: u64) -> Option<String> {
        let sh = self.shared.lock().expect("shared");
        let node = sh.sessions.get(&session)?.node;
        Some(sh.meta[node].identity.clone())
    }

    /// How often a node that failed is tried again in the background (default
    /// [`RETRY_DOWN_AFTER`]).
    pub fn with_retry_every(mut self, every: std::time::Duration) -> FleetManager {
        self.retry_every = every;
        self
    }

    fn mark_unhealthy(&self, i: usize, why: &dyn std::fmt::Display) {
        let mut sh = self.shared.lock().expect("shared");
        if sh.meta[i].down_since.is_none() {
            let now = std::time::Instant::now();
            sh.meta[i].down_since = Some(now);
            sh.meta[i].last_retry = Some(now);
            tracing::warn!("fleet: node {} is down ({why}); retrying it in the background", self.label(&sh, i));
        }
    }

    fn mark_up(&self, i: usize) {
        let mut sh = self.shared.lock().expect("shared");
        if let Some(since) = sh.meta[i].down_since.take() {
            tracing::info!("fleet: node {} is back after {} s", self.label(&sh, i), since.elapsed().as_secs());
        }
    }

    fn label(&self, sh: &Shared, i: usize) -> String {
        let node = self.node(i);
        let addr = &node.spec.addr;
        match sh.meta[i].identity.as_str() {
            "" => addr.clone(),
            id => format!("{id} ({addr})"),
        }
    }

    fn identity(&self, i: usize) -> String {
        let sh = self.shared.lock().expect("shared");
        self.label(&sh, i)
    }

    fn active_node(&self, session: u64) -> Result<usize, DaemonError> {
        let sh = self.shared.lock().expect("shared");
        let p = sh
            .sessions
            .get(&session)
            .ok_or(DaemonError::UnknownSession(session))?;
        if p.parked {
            return Err(DaemonError::Config("session is parked; resume it first"));
        }
        Ok(p.node)
    }

    fn ensure_connected(&self, i: usize, guard: &mut Option<FleetHead>) -> Result<(), DaemonError> {
        if guard.is_some() {
            return Ok(());
        }
        let node = self.node(i);
        let spec = &node.spec;
        if spec.addr.starts_with(JOINED) {
            let e = DaemonError::FleetNode { addr: spec.addr.clone(), why: "left; waiting for it to join again".into() };
            self.mark_unhealthy(i, &e);
            return Err(e);
        }
        match FleetHead::connect(&spec.addr, spec.auth.clone()) {
            Ok(head) => {
                self.sync_meta(i, &head);
                *guard = Some(head);
                Ok(())
            }
            Err(e) => {
                self.mark_unhealthy(i, &e);
                Err(e)
            }
        }
    }

    /// A fresh connection knows none of the sessions placed over the one it replaced; this
    /// places the session on it again from what the head kept.
    fn rebind(&self, session: u64, head: &mut FleetHead) -> Result<(), DaemonError> {
        if head.knows(session) {
            return Ok(());
        }
        let p = self
            .shared
            .lock()
            .expect("shared")
            .sessions
            .get(&session)
            .cloned()
            .ok_or(DaemonError::UnknownSession(session))?;
        let epoch = head.assign(session, p.params, p.qos_class, &p.codec, self.lease)?;
        if !p.tokens.is_empty() {
            head.append(session, &p.tokens)?;
        }
        if let Some(p) = self.shared.lock().expect("shared").sessions.get_mut(&session) {
            p.epoch = epoch;
        }
        Ok(())
    }

    /// Rebinds the session on the slot's connection, dropping the connection only when the link
    /// itself failed.
    fn rebind_on(&self, session: u64, guard: &mut Option<FleetHead>) -> Result<(), DaemonError> {
        let r = self.rebind(session, guard.as_mut().expect("connected"));
        if let Err(e) = &r {
            if is_fatal_node(e) {
                *guard = None;
            }
        }
        r
    }

    pub fn advertised_context(&self, model: &str) -> Option<u64> {
        let sh = self.shared.lock().expect("shared");
        (0..self.node_count())
            .filter(|&i| sh.meta[i].up() && Self::serves(&sh, i, model))
            .map(|i| sh.meta[i].max_context_tokens)
            .filter(|&c| c > 0)
            .max()
    }

    fn serves(sh: &Shared, i: usize, model: &str) -> bool {
        !sh.meta[i].learned || sh.meta[i].models.iter().any(|m| m == model)
    }

    fn holds(sh: &Shared, i: usize, len: u64) -> bool {
        let max = sh.meta[i].max_context_tokens;
        max == 0 || len <= max
    }

    fn length_refusal(&self, cand: &[usize], model: &str, len: u64) -> Option<DaemonError> {
        let sh = self.shared.lock().expect("shared");
        let eligible: Vec<usize> =
            cand.iter().copied().filter(|&i| Self::serves(&sh, i, model)).collect();
        if eligible.is_empty() || eligible.iter().any(|&i| Self::holds(&sh, i, len)) {
            return None;
        }
        Some(DaemonError::StreamTooLong {
            len,
            max: eligible.iter().map(|&i| sh.meta[i].max_context_tokens).max().unwrap_or(0),
        })
    }

    fn sync_meta(&self, i: usize, head: &FleetHead) {
        let mut sh = self.shared.lock().expect("shared");
        let cap = &head.node().capacity;
        sh.meta[i].models = head.node().loadable_models.clone();
        sh.meta[i].max_context_tokens = cap.max_context_tokens;
        sh.meta[i].identity = head.node().node_identity.clone();
        sh.meta[i].learned = true;
        sh.meta[i].load = Self::load_from(cap);
        let node = self.node(i);
        if self.conns_by_lanes && !node.spec.addr.starts_with(JOINED) {
            let (lanes, pool) = (cap.max_lanes.max(1) as usize, node.conns.len());
            let was = node.open.swap(lanes.min(pool), std::sync::atomic::Ordering::Relaxed);
            if lanes > pool && was != pool {
                tracing::warn!(
                    "fleet: node {} has {lanes} lanes; the head opens at most {pool} connections to it",
                    self.label(&sh, i)
                );
            }
        }
    }

    fn load_from(cap: &superfluid_proto::handshake::CapacitySummary) -> NodeLoad {
        NodeLoad {
            kv_used: cap.kv_blocks_used,
            kv_total: cap.kv_blocks_total,
            lanes_active: cap.lanes_active,
            queue_depth: cap.queue_depth,
            decode_tokens_per_s: cap.decode_tokens_per_s,
            prefill_tokens_per_s: cap.prefill_tokens_per_s,
            at: Some(std::time::Instant::now()),
        }
    }

    fn replacement_shape(tokens: &[u32], expected_out: u64) -> RequestShape {
        RequestShape {
            prompt_tokens: tokens.len() as u64,
            expected_out,
        }
    }

    fn candidates_claiming(
        &self,
        model: &str,
        claim_for: Option<u64>,
        shape: RequestShape,
    ) -> Vec<usize> {
        let mut sh = self.shared.lock().expect("shared");
        let mut load = vec![0usize; self.node_count()];
        for p in sh.sessions.values() {
            if !p.parked {
                load[p.node] += 1;
            }
        }
        let mut cand: Vec<usize> = (0..self.node_count())
            .filter(|&i| sh.meta[i].up() && Self::serves(&sh, i, model))
            .collect();
        match self.policy {
            PlacementPolicy::LoadAware => {
                let fits: Vec<usize> = cand
                    .iter()
                    .copied()
                    .filter(|&i| Self::holds(&sh, i, shape.prompt_tokens))
                    .collect();
                let judged: &[usize] = if fits.is_empty() { &cand } else { &fits };
                let hot: Vec<usize> = judged
                    .iter()
                    .copied()
                    .filter(|&i| {
                        sh.meta[i].load.pool_pct().is_some_and(|p| p >= self.pool_high_pct as u64)
                    })
                    .collect();
                if hot.len() < judged.len() {
                    cand.retain(|i| !hot.contains(i));
                }
                let inflight = sh.inflight.clone();
                let mean = |rate: fn(&NodeLoad) -> u32| {
                    let known: Vec<u64> =
                        cand.iter().map(|&i| rate(&sh.meta[i].load) as u64).filter(|&r| r > 0).collect();
                    known.iter().sum::<u64>() / (known.len() as u64).max(1)
                };
                let typical = (mean(|l| l.decode_tokens_per_s), mean(|l| l.prefill_tokens_per_s));
                cand.sort_by_key(|&i| {
                    (
                        sh.meta[i].load.eta_ms(inflight[i], shape.prompt_tokens, shape.expected_out, typical),
                        load[i],
                    )
                });
            }
            PlacementPolicy::LeastLoaded => {
                let inflight = sh.inflight.clone();
                cand.sort_by_key(|&i| load[i] + inflight[i] as usize);
            }
            PlacementPolicy::RoundRobin => {
                let n = self.node_count().max(1);
                let c = sh.rr_cursor % n;
                cand.sort_by_key(|&i| (i + n - c) % n);
                sh.rr_cursor = sh.rr_cursor.wrapping_add(1);
            }
        }
        if let (Some(session), Some(&winner)) = (claim_for, cand.first()) {
            Self::claim_locked(&mut sh, winner, session);
        }
        cand
    }

    /// The slot's connection, dropped first if the node closed it while it sat idle: a write
    /// into a closed socket can look like it succeeded.
    fn checkout_live(node_conn: &NodeConn, slot: usize) -> std::sync::MutexGuard<'_, Option<FleetHead>> {
        let mut guard = node_conn.checkout_at(slot);
        if guard.as_mut().is_some_and(|h| h.drain_status().is_err()) {
            *guard = None;
        }
        guard
    }

    fn slot_of(&self, session: u64) -> usize {
        self.shared
            .lock()
            .expect("shared")
            .sessions
            .get(&session)
            .map(|p| p.slot)
            .unwrap_or(0)
    }

    fn claim_locked(sh: &mut Shared, node: usize, session: u64) {
        if let Some(prev) = sh.claims.insert(session, node) {
            if prev == node {
                return;
            }
            if let Some(n) = sh.inflight.get_mut(prev) {
                *n = n.saturating_sub(1);
            }
        }
        if let Some(n) = sh.inflight.get_mut(node) {
            *n = n.saturating_add(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn place(
        &self,
        session: u64,
        params: GenParams,
        qos_class: u8,
        codec: &str,
        model: &str,
        tokens: &[u32],
        expected_out: u64,
    ) -> Result<usize, DaemonError> {
        let cand = self.candidates_claiming(
            model,
            Some(session),
            RequestShape {
                prompt_tokens: tokens.len() as u64,
                expected_out,
            },
        );
        let release_on_error = |e: DaemonError| {
            InFlight::release(&self.shared, session);
            e
        };
        if cand.is_empty() {
            return Err(release_on_error(DaemonError::Config(
                "no healthy node can serve the model",
            )));
        }
        let len = tokens.len() as u64;
        if let Some(e) = self.length_refusal(&cand, model, len) {
            return Err(release_on_error(e));
        }
        let mut last_err = None;
        for &i in &cand {
            match self.try_place_on(i, session, params, qos_class, codec, model, tokens) {
                Ok(true) => {
                    tracing::info!(
                        "fleet: session {session} ({} prompt tokens) placed on {}",
                        tokens.len(),
                        self.identity(i)
                    );
                    return Ok(i);
                }
                Ok(false) => continue,
                Err(e) => {
                    self.mark_unhealthy(i, &e);
                    self.node(i).reset();
                    last_err = Some(e);
                }
            }
        }
        if let Some(e) = self.length_refusal(&cand, model, len) {
            return Err(release_on_error(e));
        }
        Err(release_on_error(last_err.unwrap_or(DaemonError::Config(
            "no reachable node serves the model",
        ))))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_place_on(
        &self,
        i: usize,
        session: u64,
        params: GenParams,
        qos_class: u8,
        codec: &str,
        model: &str,
        tokens: &[u32],
    ) -> Result<bool, DaemonError> {
        let node = self.node(i);
        let (slot, mut guard) = node.checkout_next();
        if let Some(h) = guard.as_mut() {
            if h.drain_status().is_err() {
                *guard = None;
            }
        }
        self.ensure_connected(i, &mut guard)?;
        self.flush_forgets(i, slot, guard.as_mut().expect("connected"));
        self.sync_meta(i, guard.as_ref().expect("connected"));
        {
            let sh = self.shared.lock().expect("shared");
            if !Self::serves(&sh, i, model) {
                return Ok(false);
            }
            if !Self::holds(&sh, i, tokens.len() as u64) {
                return Ok(false);
            }
        }
        let head = guard.as_mut().expect("connected");
        let epoch = match head.assign(session, params, qos_class, codec, self.lease) {
            Ok(epoch) => epoch,
            Err(e) => {
                *guard = None;
                return Err(e);
            }
        };
        if !tokens.is_empty() {
            if let Err(e) = head.append(session, tokens) {
                *guard = None;
                return Err(e);
            }
        }
        InFlight::claim(&self.shared, i, session);
        self.shared.lock().expect("shared").sessions.insert(
            session,
            Placed {
                node: i,
                params,
                qos_class,
                codec: codec.to_string(),
                model: model.to_string(),
                tokens: tokens.to_vec(),
                slot,
                epoch,
                parked: false,
            },
        );
        Ok(true)
    }

    pub fn append(&self, session: u64, tokens: &[u32]) -> Result<(), DaemonError> {
        let node = self.active_node(session)?;
        let len = tokens.len() as u64;
        let fits = {
            let sh = self.shared.lock().expect("shared");
            Self::holds(&sh, node, len)
        };
        if !fits {
            return self.re_place_grown(session, node, tokens);
        }
        {
            let node_conn = self.node(node);
            let mut guard = Self::checkout_live(&node_conn, self.slot_of(session));
            self.ensure_connected(node, &mut guard)?;
            self.rebind_on(session, &mut guard)?;
            let head = guard.as_mut().expect("connected");
            if let Err(e) = head.append(session, tokens) {
                *guard = None;
                return Err(e);
            }
            self.sync_meta(node, head);
        }
        if let Some(p) = self.shared.lock().expect("shared").sessions.get_mut(&session) {
            p.tokens = tokens.to_vec();
        }
        Ok(())
    }

    fn re_place_grown(
        &self,
        session: u64,
        from: usize,
        tokens: &[u32],
    ) -> Result<(), DaemonError> {
        let placed = {
            let sh = self.shared.lock().expect("shared");
            sh.sessions.get(&session).cloned().ok_or(DaemonError::UnknownSession(session))?
        };
        let cand: Vec<usize> = self
            .candidates_claiming(
                &placed.model,
                Some(session),
                Self::replacement_shape(tokens, 0),
            )
            .into_iter()
            .filter(|&i| i != from)
            .collect();
        let len = tokens.len() as u64;
        if let Some(e) = self.length_refusal(&cand, &placed.model, len) {
            return Err(e);
        }
        for &i in &cand {
            match self.try_place_on(
                i,
                session,
                placed.params,
                placed.qos_class,
                &placed.codec,
                &placed.model,
                tokens,
            ) {
                Ok(true) => return Ok(()),
                Ok(false) => continue,
                Err(e) => {
                    self.mark_unhealthy(i, &e);
                    self.node(i).reset();
                }
            }
        }
        if let Some(e) = self.length_refusal(&cand, &placed.model, len) {
            return Err(e);
        }
        Err(DaemonError::Config(
            "no reachable node can hold the session's grown context",
        ))
    }

    pub fn generate(
        &self,
        session: u64,
        max_new_tokens: u64,
        max_wall_ms: u64,
    ) -> Result<FleetGeneration, DaemonError> {
        let node = self.active_node(session)?;

        match self.generate_on(node, session, max_new_tokens, max_wall_ms) {
            Ok(g) => return Ok(g),
            Err(e) if !is_fatal_node(&e) => return Err(e),
            Err(e) => self.mark_unhealthy(node, &e),
        }
        self.node(node).reset();
        let placed = self
            .shared
            .lock()
            .expect("shared")
            .sessions
            .get(&session)
            .cloned()
            .expect("placed session");
        let cand: Vec<usize> = self
            .candidates_claiming(
                &placed.model,
                Some(session),
                Self::replacement_shape(&placed.tokens, max_new_tokens),
            )
            .into_iter()
            .filter(|&i| i != node)
            .collect();
        for i in cand {
            match self.try_place_on(
                i,
                session,
                placed.params,
                placed.qos_class,
                &placed.codec,
                &placed.model,
                &placed.tokens,
            ) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(e) => {
                    self.mark_unhealthy(i, &e);
                    self.node(i).reset();
                    continue;
                }
            }
            tracing::warn!("fleet: session {session} moved to {}", self.identity(i));
            match self.generate_on(i, session, max_new_tokens, max_wall_ms) {
                Ok(g) => return Ok(g),
                Err(e) => self.mark_unhealthy(i, &e),
            }
            self.node(i).reset();
        }
        Err(DaemonError::Generation(
            "generation failed and no healthy node could take over the session".into(),
        ))
    }

    fn map_refusal(
        &self,
        node: usize,
        session: u64,
        r: Result<FleetGeneration, DaemonError>,
    ) -> Result<FleetGeneration, DaemonError> {
        let g = r?;
        if g.finish == crate::nodeagent::FINISH_DECLINED {
            return Err(DaemonError::Generation(format!(
                "node {} declined session {session} (stale epoch or replayed command)",
                self.shared.lock().expect("shared").meta[node].identity
            )));
        }
        if g.finish != crate::nodeagent::FINISH_REFUSED {
            return Ok(g);
        }
        let sh = self.shared.lock().expect("shared");
        let len = sh.sessions.get(&session).map(|p| p.tokens.len() as u64).unwrap_or(0);
        Err(DaemonError::StreamTooLong {
            len,
            max: sh.meta[node].max_context_tokens,
        })
    }

    fn generate_on(
        &self,
        node: usize,
        session: u64,
        max_new_tokens: u64,
        max_wall_ms: u64,
    ) -> Result<FleetGeneration, DaemonError> {
        let _inflight = InFlight::guard(&self.shared, node, session);
        let node_conn = self.node(node);
        let mut guard = Self::checkout_live(&node_conn, self.slot_of(session));
        self.ensure_connected(node, &mut guard)?;
        self.rebind_on(session, &mut guard)?;
        let head = guard.as_mut().expect("connected");
        let r = head.generate(session, max_new_tokens, max_wall_ms);
        if let Err(e) = &r {
            if is_fatal_node(e) {
                *guard = None;
            }
        }
        if let Some(head) = guard.as_ref() {
            self.sync_meta(node, head);
        }
        drop(guard);
        self.map_refusal(node, session, r)
    }

    pub fn generate_streaming(
        &self,
        session: u64,
        max_new_tokens: u64,
        max_wall_ms: u64,
        mut on_tokens: impl FnMut(&[u32]),
    ) -> Result<FleetGeneration, DaemonError> {
        let (node, flag) = {
            let mut sh = self.shared.lock().expect("shared");
            let p = sh
                .sessions
                .get(&session)
                .ok_or(DaemonError::UnknownSession(session))?;
            if p.parked {
                return Err(DaemonError::Config("session is parked; resume it first"));
            }
            let node = p.node;
            let flag = sh
                .cancels
                .entry(session)
                .or_insert_with(|| Arc::new(AtomicBool::new(false)))
                .clone();
            flag.store(false, Ordering::Relaxed);
            (node, flag)
        };

        let mut all_tokens: Vec<u32> = Vec::new();

        match self.generate_streaming_on(node, session, max_new_tokens, max_wall_ms, &flag, |b| {
            all_tokens.extend_from_slice(b);
            on_tokens(b);
        }) {
            Ok(g) => return Ok(g),
            Err(e) if !is_fatal_node(&e) => return Err(e),
            Err(e) => self.mark_unhealthy(node, &e),
        }

        let placed = self
            .shared
            .lock()
            .expect("shared")
            .sessions
            .get(&session)
            .cloned()
            .expect("placed session");
        let input = placed.tokens.clone();
        let mut tried = vec![node];
        loop {
            let remaining = max_new_tokens.saturating_sub(all_tokens.len() as u64);
            if remaining == 0 {
                return Ok(FleetGeneration {
                    tokens: all_tokens,
                    finish: superfluid_abi::finish::LENGTH,
                    expired: false,
                });
            }
            let cand: Vec<usize> = self
                .candidates_claiming(
                    &placed.model,
                    Some(session),
                    Self::replacement_shape(&input, remaining),
                )
                .into_iter()
                .filter(|i| !tried.contains(i))
                .collect();
            if cand.is_empty() {
                return Err(DaemonError::Generation(
                    "streaming generation failed and no healthy node could take over".into(),
                ));
            }
            let i = cand[0];
            tried.push(i);
            let mut ctx = input.clone();
            ctx.extend_from_slice(&all_tokens);
            match self.try_place_on(
                i,
                session,
                placed.params,
                placed.qos_class,
                &placed.codec,
                &placed.model,
                &ctx,
            ) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(e) => {
                    self.mark_unhealthy(i, &e);
                    self.node(i).reset();
                    continue;
                }
            }
            tracing::warn!(
                "fleet: session {session} moved to {} after {} streamed tokens",
                self.identity(i),
                all_tokens.len()
            );
            match self.generate_streaming_on(i, session, remaining, max_wall_ms, &flag, |b| {
                all_tokens.extend_from_slice(b);
                on_tokens(b);
            }) {
                Ok(g) => {
                    return Ok(FleetGeneration {
                        tokens: all_tokens,
                        finish: g.finish,
                        expired: g.expired,
                    })
                }
                Err(e) if !is_fatal_node(&e) => return Err(e),
                Err(e) => self.mark_unhealthy(i, &e),
            }
        }
    }

    fn generate_streaming_on(
        &self,
        node: usize,
        session: u64,
        max_new_tokens: u64,
        max_wall_ms: u64,
        flag: &AtomicBool,
        on_tokens: impl FnMut(&[u32]),
    ) -> Result<FleetGeneration, DaemonError> {
        let _inflight = InFlight::guard(&self.shared, node, session);
        let node_conn = self.node(node);
        let mut guard = Self::checkout_live(&node_conn, self.slot_of(session));
        self.ensure_connected(node, &mut guard)?;
        self.rebind_on(session, &mut guard)?;
        let head = guard.as_mut().expect("connected");
        let r = head.generate_streaming(session, max_new_tokens, max_wall_ms, flag, on_tokens);
        if let Err(e) = &r {
            if is_fatal_node(e) {
                *guard = None;
            }
        }
        if let Some(head) = guard.as_ref() {
            self.sync_meta(node, head);
        }
        drop(guard);
        self.map_refusal(node, session, r)
    }

    pub fn cancel(&self, session: u64) -> Result<(), DaemonError> {
        let sh = self.shared.lock().expect("shared");
        if !sh.sessions.contains_key(&session) {
            return Err(DaemonError::UnknownSession(session));
        }
        if let Some(flag) = sh.cancels.get(&session) {
            flag.store(true, Ordering::Relaxed);
        }
        Ok(())
    }

    pub fn park(&self, session: u64) -> Result<(), DaemonError> {
        let node = {
            let mut sh = self.shared.lock().expect("shared");
            let p = sh
                .sessions
                .get_mut(&session)
                .ok_or(DaemonError::UnknownSession(session))?;
            if p.parked {
                return Ok(());
            }
            p.parked = true;
            (p.node, p.slot, p.epoch)
        };
        let (node, slot, epoch) = node;
        let node_conn = self.node(node);
        let mut guard = Self::checkout_live(&node_conn, slot);
        if self.ensure_connected(node, &mut guard).is_ok() {
            if let Err(e) = guard.as_mut().expect("connected").revoke(session, epoch) {
                *guard = None;
                self.mark_unhealthy(node, &e);
            }
        }
        Ok(())
    }

    pub fn resume(&self, session: u64) -> Result<usize, DaemonError> {
        let placed = {
            let sh = self.shared.lock().expect("shared");
            let p = sh
                .sessions
                .get(&session)
                .ok_or(DaemonError::UnknownSession(session))?;
            if !p.parked {
                return Err(DaemonError::Config("session is not parked"));
            }
            p.clone()
        };
        let cand = self.candidates_claiming(
            &placed.model,
            Some(session),
            Self::replacement_shape(&placed.tokens, 0),
        );
        if cand.is_empty() {
            InFlight::release(&self.shared, session);
            return Err(DaemonError::Config("no healthy node can serve the model"));
        }
        let mut last_err = None;
        for i in cand {
            match self.try_place_on(
                i,
                session,
                placed.params,
                placed.qos_class,
                &placed.codec,
                &placed.model,
                &placed.tokens,
            ) {
                Ok(true) => return Ok(i),
                Ok(false) => continue,
                Err(e) => {
                    self.mark_unhealthy(i, &e);
                    self.node(i).reset();
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or(DaemonError::Config("no reachable node serves the model")))
    }

    /// Ends a session the caller is done with: the head drops its record, and the node the
    /// context it kept (now if its connection is free, else the next time it is).
    pub fn finish(&self, session: u64) {
        let placed = {
            let mut sh = self.shared.lock().expect("shared");
            sh.cancels.remove(&session);
            sh.sessions.remove(&session)
        };
        InFlight::release(&self.shared, session);
        let Some(p) = placed else { return };
        self.shared.lock().expect("shared").forgets.push((p.node, p.slot, session, p.epoch));
        let node = self.node(p.node);
        let lock = node.conns[p.slot].try_lock();
        if let Ok(mut guard) = lock {
            if let Some(head) = guard.as_mut() {
                self.flush_forgets(p.node, p.slot, head);
            }
        }
    }

    fn flush_forgets(&self, node: usize, slot: usize, head: &mut FleetHead) {
        let due: Vec<(u64, u64)> = {
            let mut sh = self.shared.lock().expect("shared");
            if sh.forgets.is_empty() {
                return;
            }
            let (due, keep) = sh.forgets.drain(..).partition(|&(n, s, _, _)| n == node && s == slot);
            sh.forgets = keep;
            due.into_iter().map(|(_, _, session, epoch)| (session, epoch)).collect()
        };
        for (session, epoch) in due {
            let _ = head.revoke(session, epoch);
        }
    }

    /// Connects to every node once, so the head learns each one (or why it cannot) at start.
    pub fn probe(&self) -> Vec<Result<NodeReport, DaemonError>> {
        std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..self.node_count())
                .map(|i| {
                    scope.spawn(move || {
                        let node = self.node(i);
                        let mut guard = node.checkout_at(0);
                        self.ensure_connected(i, &mut guard)?;
                        let node = guard.as_ref().expect("connected").node();
                        Ok(NodeReport {
                            addr: self.node(i).spec.addr.clone(),
                            identity: node.node_identity.clone(),
                            models: node.loadable_models.clone(),
                            max_context_tokens: node.capacity.max_context_tokens,
                            max_lanes: node.capacity.max_lanes,
                        })
                    })
                })
                .collect();
            jobs.into_iter().map(|j| j.join().expect("probe thread")).collect()
        })
    }

    /// Keeps idle connections open, drops ones a node closed, and finishes deferred session ends.
    pub fn keepalive(&self) {
        let nodes: Vec<Arc<NodeConn>> = self.nodes.read().expect("nodes").clone();
        for (i, node) in nodes.iter().enumerate() {
            for (slot, conn) in node.conns.iter().enumerate() {
                let Ok(mut guard) = conn.try_lock() else { continue };
                let Some(head) = guard.as_mut() else { continue };
                self.flush_forgets(i, slot, head);
                match head.keepalive() {
                    Ok(()) => self.sync_meta(i, head),
                    Err(_) => *guard = None,
                }
            }
        }
    }

    /// Tries every node that failed and is due a retry, in parallel and outside the connection
    /// slots; one that answers is placed on again. Requests never wait on a down node.
    pub fn retry_down(&self) {
        let due: Vec<usize> = {
            let mut sh = self.shared.lock().expect("shared");
            let now = std::time::Instant::now();
            let mut due = Vec::new();
            for (i, m) in sh.meta.iter_mut().enumerate() {
                if m.down_since.is_some() && m.last_retry.is_none_or(|t| now.duration_since(t) >= self.retry_every) {
                    m.last_retry = Some(now);
                    due.push(i);
                }
            }
            due
        };
        std::thread::scope(|scope| {
            for i in due {
                scope.spawn(move || {
                    let node = self.node(i);
                    let spec = &node.spec;
                    // A joined node comes back when it dials in again.
                    if spec.addr.starts_with(JOINED) {
                        return;
                    }
                    match FleetHead::connect(&spec.addr, spec.auth.clone()) {
                        Ok(head) => {
                            self.sync_meta(i, &head);
                            self.mark_up(i);
                            if let Ok(mut slot) = node.conns[0].try_lock() {
                                if slot.is_none() {
                                    *slot = Some(head);
                                }
                            }
                        }
                        Err(e) => tracing::debug!("fleet: {e}"),
                    }
                });
            }
        });
    }

    pub fn tend(&self) {
        self.keepalive();
        self.retry_down();
    }

    pub fn spawn_keepalive(self: &Arc<Self>) {
        let mgr = Arc::downgrade(self);
        let tick = self.retry_every.min(KEEPALIVE_EVERY);
        std::thread::Builder::new()
            .name("fleet-keepalive".into())
            .spawn(move || loop {
                std::thread::sleep(tick);
                let Some(mgr) = mgr.upgrade() else { return };
                mgr.retry_down();
                let due = mgr.last_keepalive.lock().expect("keepalive clock").elapsed() >= KEEPALIVE_EVERY;
                if due {
                    mgr.keepalive();
                    *mgr.last_keepalive.lock().expect("keepalive clock") = std::time::Instant::now();
                }
            })
            .expect("spawn the fleet keepalive");
    }
}

fn is_fatal_node(e: &DaemonError) -> bool {
    matches!(
        e,
        DaemonError::FleetNode { .. }
            | DaemonError::Fleet(superfluid_linkf::LinkFError::Closed)
            | DaemonError::Fleet(superfluid_linkf::LinkFError::TimedOut)
            | DaemonError::Fleet(superfluid_linkf::LinkFError::Io(_))
    )
}
