//! The node-agent.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use superfluid_abi::finish;
use superfluid_linkf::{AgentAction, AgentSession, Endpoint};
use superfluid_proto::handshake::{CapacitySummary, HelloAckF};
use superfluid_proto::linkf::{self, msg_type, LinkFLimits, TokenEventPayload};

use crate::fleet::{decode_params, decode_tokens};
use crate::wal::GenParams;
use crate::{Daemon, DaemonError};

pub const FINISH_EXPIRED: u32 = u32::MAX;

pub const FINISH_REFUSED: u32 = u32::MAX - 1;

pub const FINISH_DECLINED: u32 = u32::MAX - 2;

pub const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

struct AgentShared {
    agent: AgentSession,
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
}

const STATUS_INTERVAL_MS: u64 = 2_000;

const EMIT_COALESCE_TOKENS: usize = 16;

const EMIT_COALESCE_MS: u64 = 200;

impl NodeAgent {
    pub fn new(daemon: Arc<Daemon>, identity: impl Into<String>, model_name: impl Into<String>) -> NodeAgent {
        NodeAgent {
            max_lanes: 0,
            last_status_ms: 0,
            daemon,
            shared: Arc::new(Mutex::new(AgentShared {
                agent: AgentSession::new(),
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
        }
    }

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
            Err(superfluid_linkf::LinkFError::TimedOut) => {
                tracing::warn!("node-agent dropped a silent connection (no Hello)");
                return Ok(());
            }
            Err(superfluid_linkf::LinkFError::Unauthorized) => {
                return Err(superfluid_linkf::LinkFError::Unauthorized.into());
            }
            Err(e) => return Err(e.into()),
        }
        let mut last_activity = Instant::now();
        loop {
            let now = self.start.elapsed();
            for renew in self.shared.lock().expect("agent shared").agent.needs_renew(now) {
                endpoint.send(msg_type::LEASE_RENEW, &renew)?;
            }
            {
                let mut sh = self.shared.lock().expect("agent shared");
                for action in sh.agent.poll(now) {
                    match action {
                        AgentAction::SelfFence { session_id } => {
                            tracing::warn!(session_id, "node-agent self-fenced on lease expiry");
                            sh.contexts.remove(&session_id);
                            sh.params.remove(&session_id);
                        }
                    }
                }
            }
            if now.as_millis() as u64 >= self.last_status_ms + STATUS_INTERVAL_MS {
                self.send_status(&mut endpoint);
            }
            let frames = match endpoint.pump() {
                Ok(f) => f,
                Err(superfluid_linkf::LinkFError::Closed) => return Ok(()),
                Err(e) => return Err(e.into()),
            };
            if frames.is_empty() {
                if !self.idle_timeout.is_zero() && last_activity.elapsed() >= self.idle_timeout {
                    tracing::info!("node-agent dropped an idle connection");
                    return Ok(());
                }
            } else {
                last_activity = Instant::now();
                for frame in frames {
                    self.handle(&mut endpoint, frame)?;
                }
            }
        }
    }

    fn handle(
        &mut self,
        endpoint: &mut Endpoint,
        frame: superfluid_proto::Frame,
    ) -> Result<(), DaemonError> {
        let now = self.start.elapsed();
        match frame.msg_type {
            msg_type::RECONCILE => {
                let m: linkf::ReconcileMsg = postcard::from_bytes(&frame.payload)?;
                let (ack, retransmissions) =
                    self.shared.lock().expect("agent shared").agent.on_reconcile(&m, now);
                endpoint.send(msg_type::RECONCILE_ACK, &ack)?;
                for batch in retransmissions {
                    endpoint.send(msg_type::TOKEN_EVENTS, &batch)?;
                }
            }
            msg_type::SESSION_ASSIGN => {
                let m: linkf::SessionAssignMsg = postcard::from_bytes(&frame.payload)?;
                {
                    let mut sh = self.shared.lock().expect("agent shared");
                    if sh.agent.on_assign(&m, now) {
                        sh.params.insert(m.session_id, decode_params(&m.params));
                    }
                }
            }
            msg_type::SESSION_REVOKE => {
                let m: linkf::SessionRevokeMsg = postcard::from_bytes(&frame.payload)?;
                let mut sh = self.shared.lock().expect("agent shared");
                if sh.agent.on_revoke(&m) {
                    sh.contexts.remove(&m.session_id);
                    sh.params.remove(&m.session_id);
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
                let declined = !self.shared.lock().expect("agent shared").agent.on_generate(&m);
                if declined {
                    tracing::warn!(
                        session = m.session_id,
                        generation = m.generation_id,
                        epoch = m.epoch,
                        "declining a generate (unknown session, superseded epoch, or \
                         replayed command); reporting it rather than going silent"
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
                    match self.run_generation(endpoint, &m) {
                        Ok(()) => {}
                        Err(e) if is_client_refusal(&e) => {
                            tracing::warn!(
                                session = m.session_id,
                                generation = m.generation_id,
                                error = %e,
                                "refusing a generation the node cannot stage; \
                                 the head should have caught this before dispatch"
                            );
                            let batch = self.shared.lock().expect("agent shared").agent.emit(
                                m.session_id,
                                m.generation_id,
                                TokenEventPayload::Finish {
                                    reason: FINISH_REFUSED,
                                },
                            );
                            if let Some(batch) = batch {
                                endpoint.send(msg_type::TOKEN_EVENTS, &batch)?;
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
            msg_type::STAGE_FORWARD_REQ => {
                let m: linkf::StageForwardReqMsg = postcard::from_bytes(&frame.payload)?;
                let resp = match self.daemon.forward_stage(
                    m.tokens.clone(),
                    m.start_layer,
                    m.end_layer,
                    m.hidden_in.clone(),
                ) {
                    Ok(crate::StageOutput::Hidden(h)) => linkf::StageForwardRespMsg {
                        request_id: m.request_id,
                        hidden: Some(h),
                        token: None,
                    },
                    Ok(crate::StageOutput::Token(t)) => linkf::StageForwardRespMsg {
                        request_id: m.request_id,
                        hidden: None,
                        token: Some(t),
                    },
                    Err(_) => linkf::StageForwardRespMsg {
                        request_id: m.request_id,
                        hidden: None,
                        token: None,
                    },
                };
                endpoint.send(msg_type::STAGE_FORWARD_RESP, &resp)?;
            }
            msg_type::WATERMARK_ACK => {
                let m: linkf::WatermarkAckMsg = postcard::from_bytes(&frame.payload)?;
                self.shared.lock().expect("agent shared").agent.on_watermark_ack(&m);
            }
            msg_type::LEASE_GRANT => {
                let m: linkf::LeaseGrantMsg = postcard::from_bytes(&frame.payload)?;
                self.shared.lock().expect("agent shared").agent.on_lease_grant(&m, now);
            }
            _ => {  }
        }
        Ok(())
    }

    fn flush_emit(
        endpoint: &mut Endpoint,
        shared: &Arc<Mutex<AgentShared>>,
        session_id: u64,
        generation_id: u64,
        pending: &mut Vec<u32>,
        now: Duration,
    ) -> Result<(), Stop> {
        let emitted = shared.lock().expect("agent shared").agent.emit(
            session_id,
            generation_id,
            TokenEventPayload::Tokens(std::mem::take(pending)),
        );
        match emitted {
            Some(b) => {
                if endpoint.send(msg_type::TOKEN_EVENTS, &b).is_err() {
                    return Err(Stop::Peer);
                }
            }
            None => return Err(Stop::Fenced),
        }
        let frames = endpoint.pump().map_err(|_| Stop::Peer)?;
        for frame in frames {
            match frame.msg_type {
                msg_type::GENERATE_CANCEL => {
                    if let Ok(m) = postcard::from_bytes::<linkf::GenerateCancelMsg>(&frame.payload) {
                        if m.session_id == session_id && m.generation_id == generation_id {
                            return Err(Stop::Cancelled);
                        }
                    }
                }
                msg_type::WATERMARK_ACK => {
                    if let Ok(m) = postcard::from_bytes::<linkf::WatermarkAckMsg>(&frame.payload) {
                        shared.lock().expect("agent shared").agent.on_watermark_ack(&m);
                    }
                }
                msg_type::LEASE_GRANT => {
                    if let Ok(m) = postcard::from_bytes::<linkf::LeaseGrantMsg>(&frame.payload) {
                        shared.lock().expect("agent shared").agent.on_lease_grant(&m, now);
                    }
                }
                _ => {  }
            }
        }
        Ok(())
    }

    fn run_generation(
        &mut self,
        endpoint: &mut Endpoint,
        req: &linkf::GenerateReqMsg,
    ) -> Result<(), DaemonError> {
        let session_id = req.session_id;
        let generation_id = req.generation_id;
        let (context, params) = {
            let sh = self.shared.lock().expect("agent shared");
            (
                sh.contexts.get(&session_id).cloned().unwrap_or_default(),
                sh.params.get(&session_id).copied().unwrap_or_default(),
            )
        };
        let max_new = req.budgets.max_new_tokens.min(u32::MAX as u64) as u32;

        let daemon = Arc::clone(&self.daemon);
        let start = self.start;
        let shared = Arc::clone(&self.shared);
        let mut stop: Option<Stop> = None;
        let mut pending: Vec<u32> = Vec::with_capacity(EMIT_COALESCE_TOKENS);
        let mut last_flush = start.elapsed();
        let extras = crate::scheduler::GenExtras::default();
        self.shared.lock().expect("agent shared").busy.start();
        let generated = daemon.generate_tokens_streaming(
            context,
            params,
            max_new,
            req.budgets.max_wall_ms,
            extras,
            |batch| {
                pending.extend_from_slice(batch);
                let now = start.elapsed();
                let due = pending.len() >= EMIT_COALESCE_TOKENS
                    || now.saturating_sub(last_flush).as_millis() as u64 >= EMIT_COALESCE_MS;
                if !due {
                    return std::ops::ControlFlow::Continue(());
                }
                last_flush = now;
                match Self::flush_emit(
                    endpoint,
                    &shared,
                    session_id,
                    generation_id,
                    &mut pending,
                    now,
                ) {
                    Ok(()) => std::ops::ControlFlow::Continue(()),
                    Err(s) => {
                        stop = Some(s);
                        std::ops::ControlFlow::Break(())
                    }
                }
            },
        );
        self.shared.lock().expect("agent shared").busy.stop();
        let (tokens, _finish, expired) = generated?;

        if stop.is_none() && !pending.is_empty() {
            let now = start.elapsed();
            if let Err(s) = Self::flush_emit(
                endpoint,
                &self.shared,
                session_id,
                generation_id,
                &mut pending,
                now,
            ) {
                stop = Some(s);
            }
        }

        let reason = match stop {
            Some(Stop::Fenced) | Some(Stop::Peer) => return Ok(()),
            Some(Stop::Cancelled) => finish::CANCELLED,
            None if expired => FINISH_EXPIRED,
            None if tokens.len() as u32 >= max_new => finish::LENGTH,
            None => finish::EOS,
        };
        if let Some(batch) =
            self.shared
                .lock()
                .expect("agent shared")
                .agent
                .emit(session_id, generation_id, TokenEventPayload::Finish { reason })
        {
            endpoint.send(msg_type::TOKEN_EVENTS, &batch)?;
        }
        Ok(())
    }
}

fn is_client_refusal(e: &DaemonError) -> bool {
    matches!(
        e,
        DaemonError::StreamTooLong { .. } | DaemonError::Constraint(_)
    )
}

enum Stop {
    Cancelled,
    Fenced,
    Peer,
}
