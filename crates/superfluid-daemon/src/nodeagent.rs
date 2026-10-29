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
    params: HashMap<u64, GenParams>,
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

pub struct NodeAgent {
    daemon: Arc<Daemon>,
    shared: Arc<Mutex<AgentShared>>,
    identity: String,
    model_name: String,
    start: Instant,
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
}

static NEXT_CONN: AtomicU64 = AtomicU64::new(1);

const STATUS_INTERVAL_MS: u64 = 2_000;

/// A renewal the head has not answered is asked for again after this long.
const RENEW_RETRY: Duration = Duration::from_secs(1);

/// How long one read of the link waits while a generation runs, which bounds how late its
/// tokens leave.
const POLL_RUNNING: Duration = Duration::from_millis(2);

const POLL_IDLE: Duration = Duration::from_millis(20);

const EMIT_COALESCE_TOKENS: usize = 16;

const EMIT_COALESCE_MS: u64 = 200;

impl NodeAgent {
    pub fn new(daemon: Arc<Daemon>, identity: impl Into<String>, model_name: impl Into<String>) -> NodeAgent {
        NodeAgent {
            max_lanes: 0,
            last_status_ms: 0,
            conn: NEXT_CONN.fetch_add(1, Ordering::Relaxed),
            asked: HashMap::new(),
            opts: HashMap::new(),
            daemon,
            shared: Arc::new(Mutex::new(AgentShared {
                agent: AgentSession::new(),
                owners: HashMap::new(),
                contexts: HashMap::new(),
                params: HashMap::new(),
                last_load_sample: None,
                decode_rate_ewma: 0,
                prefill_rate_ewma: 0,
                busy: Busy::default(),
            })),
            identity: identity.into(),
            model_name: model_name.into(),
            start: Instant::now(),
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
            daemon: Arc::clone(&self.daemon),
            shared: Arc::clone(&self.shared),
            identity: self.identity.clone(),
            model_name: self.model_name.clone(),
            start: self.start,
            handshake_timeout: self.handshake_timeout,
            idle_timeout: self.idle_timeout,
            auth: self.auth.clone(),
            max_lanes: self.max_lanes,
            last_status_ms: 0,
            conn: NEXT_CONN.fetch_add(1, Ordering::Relaxed),
            asked: HashMap::new(),
            opts: HashMap::new(),
        }
    }

    /// `idle` is how long the head may be silent before a connection is dropped; zero keeps
    /// every connection. Keep it well above the head's heartbeat, 5 s.
    pub fn with_timeouts(mut self, handshake: Duration, idle: Duration) -> NodeAgent {
        self.handshake_timeout = handshake;
        self.idle_timeout = idle;
        self
    }

    pub fn with_auth(mut self, auth: Vec<u8>) -> NodeAgent {
        self.auth = auth;
        self
    }

    fn sample_load(&mut self) -> (u64, u64, u32, u32, u32, u32, u32, u64) {
        let mut sh = self.shared.lock().expect("agent shared");
        let stats = self.daemon.sched_stats();
        let now = self.start.elapsed().as_millis() as u64;
        let decoded = stats.decode_tokens.load(std::sync::atomic::Ordering::Relaxed);
        let prefilled = stats.prefill_tokens.load(std::sync::atomic::Ordering::Relaxed);

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

        let queue: u64 = stats.queue_depth.iter().map(|q| q.load(std::sync::atomic::Ordering::Relaxed)).sum();
        (
            stats.pool_blocks_used.load(std::sync::atomic::Ordering::Relaxed),
            stats.pool_blocks_total.load(std::sync::atomic::Ordering::Relaxed),
            stats.lanes_active.load(std::sync::atomic::Ordering::Relaxed) as u32,
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
            max_context_tokens: self.daemon.max_stream_tokens(),
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
            loadable_models: vec![self.model_name.clone()],
        };
        let _ = endpoint.send(msg_type::NODE_STATUS, &msg);
        self.last_status_ms = self.start.elapsed().as_millis() as u64;
    }

    fn ackf(&mut self) -> HelloAckF {
        HelloAckF {
            chosen_version: superfluid_linkf::driver::PROTO_VERSION,
            node_identity: self.identity.clone(),
            capacity: self.capacity(),
            loadable_models: vec![self.model_name.clone()],
            feature_flags: 0,
            limits: LinkFLimits {
                max_frame: superfluid_proto::MAX_FRAME,
                transfer_window: 8,
                max_transfer: 1 << 26,
                transfer_budget: 1 << 28,
            },
        }
    }

    pub fn serve(&mut self, mut endpoint: Endpoint) -> Result<(), DaemonError> {
        let hs = if self.handshake_timeout.is_zero() {
            endpoint.answer_hello_as_agent(&self.ackf(), &self.auth)
        } else {
            endpoint.answer_hello_as_agent_within(&self.ackf(), self.handshake_timeout, &self.auth)
        };
        match hs {
            Ok(()) => {}
            Err(LinkFError::TimedOut) => {
                tracing::warn!("node-agent dropped a silent connection (no Hello)");
                return Ok(());
            }
            Err(LinkFError::Unauthorized) => return Err(LinkFError::Unauthorized.into()),
            Err(e) => return Err(e.into()),
        }
        let heartbeat = match self.idle_timeout {
            t if t.is_zero() => Liveness::DEFAULT.heartbeat,
            t => Liveness::DEFAULT.heartbeat.min(t / 3),
        };
        endpoint.keep_alive(Liveness { heartbeat, silence: self.idle_timeout });
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
            work.syncs.retain(|rx| match rx.try_recv() {
                Ok(resp) => {
                    let _ = endpoint.send(msg_type::STATE_SYNC_RESP, &resp);
                    false
                }
                Err(mpsc::TryRecvError::Empty) => true,
                Err(mpsc::TryRecvError::Disconnected) => false,
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
                let m: linkf::SessionAssignMsg = postcard::from_bytes(&frame.payload)?;
                let mut sh = self.shared.lock().expect("agent shared");
                if sh.agent.on_assign(&m, now) {
                    sh.params.insert(m.session_id, decode_params(&m.params));
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
                let declined = work.generation.is_some() || !self.shared.lock().expect("agent shared").agent.on_generate(&m);
                if declined {
                    tracing::warn!(
                        session = m.session_id,
                        generation = m.generation_id,
                        epoch = m.epoch,
                        "declining a generate (unknown session, superseded epoch, replayed \
                         command, or one already running on this connection); reporting it \
                         rather than going silent"
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
                    work.generation = Some(self.start_generation(&m));
                    endpoint.poll_every(POLL_RUNNING)?;
                }
            }
            msg_type::GENERATE_OPTS => {
                let m: linkf::GenerateOptsMsg = postcard::from_bytes(&frame.payload)?;
                self.opts.insert((m.session_id, m.generation_id), m);
            }
            msg_type::STATE_SYNC_REQ => {
                let m: linkf::StateSyncReqF = postcard::from_bytes(&frame.payload)?;
                let (tx, rx) = mpsc::channel();
                let daemon = Arc::clone(&self.daemon);
                std::thread::Builder::new()
                    .name("node-state-sync".into())
                    .spawn(move || {
                        let result = daemon.state_sync(m.req).map_err(|e| e.to_string());
                        let _ = tx.send(linkf::StateSyncRespF { request_id: m.request_id, result });
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

    /// Runs a stage forward on a thread of its own, as a generation runs, so a long one does not
    /// silence the connection.
    fn start_stage(&self, m: linkf::StageForwardReqMsg) -> mpsc::Receiver<linkf::StageForwardRespMsg> {
        let (tx, rx) = mpsc::channel();
        let daemon = Arc::clone(&self.daemon);
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

    /// Starts `req` on a thread of its own, so this connection goes on reading the head
    /// (cancels, lease grants, heartbeats) and sending status while it runs.
    fn start_generation(&mut self, req: &linkf::GenerateReqMsg) -> Running {
        let (context, params) = {
            let sh = self.shared.lock().expect("agent shared");
            (
                sh.contexts.get(&req.session_id).cloned().unwrap_or_default(),
                sh.params.get(&req.session_id).copied().unwrap_or_default(),
            )
        };
        let max_new = req.budgets.max_new_tokens.min(u32::MAX as u64) as u32;
        let max_wall_ms = req.budgets.max_wall_ms;
        let opts = self.opts.remove(&(req.session_id, req.generation_id));
        self.opts.clear();
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, out) = mpsc::channel();
        let (daemon, shared, stop) = (Arc::clone(&self.daemon), Arc::clone(&self.shared), Arc::clone(&cancel));
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
                            extras.extras.clone(),
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
        let due = done.is_some() || now.saturating_sub(run.last_flush).as_millis() as u64 >= EMIT_COALESCE_MS;
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
    syncs: Vec<mpsc::Receiver<linkf::StateSyncRespF>>,
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
}

impl Drop for Running {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
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
        let mut extras = crate::scheduler::GenExtras::default();
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
