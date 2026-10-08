//! The head daemon's fleet edge.

use std::collections::BTreeMap;

use superfluid_linkf::{Endpoint, HeadSession};
use superfluid_proto::handshake::HelloAckF;
use superfluid_proto::linkf::{
    self, msg_type, AppendContentMsg, GenerateCancelMsg, GenerationBudgetsMsg, Lease, TokenEventPayload,
};

use crate::codec::TextCodec;
use superfluid_engine::StageOutput;
use crate::wal::GenParams;
use crate::DaemonError;

pub fn encode_params(params: &GenParams) -> Vec<u8> {
    postcard::to_stdvec(params).expect("GenParams postcard")
}

pub fn decode_params(bytes: &[u8]) -> GenParams {
    postcard::from_bytes(bytes).unwrap_or_default()
}

pub fn encode_tokens(tokens: &[u32]) -> Vec<u8> {
    postcard::to_stdvec(&tokens.to_vec()).expect("tokens postcard")
}

pub fn decode_tokens(bytes: &[u8]) -> Result<Vec<u32>, DaemonError> {
    postcard::from_bytes(bytes).map_err(DaemonError::Codec)
}

#[derive(Debug, Clone)]
pub struct FleetGeneration {
    pub tokens: Vec<u32>,
    pub finish: u32,
    pub expired: bool,
}

#[derive(Debug, Clone)]
pub struct FleetSegment {
    pub channel: u32,
    pub text: String,
    pub tool: Option<(String, String)>,
}

#[derive(Debug, Clone)]
struct Placement {
    params: GenParams,
    qos_class: u8,
    codec: String,
    lease: Lease,
    tokens: Vec<u32>,
}

pub struct FleetHead {
    head: HeadSession,
    endpoint: Endpoint,
    node: HelloAckF,
    next_generation: u64,
    placements: BTreeMap<u64, Placement>,
    addr: String,
    auth: Vec<u8>,
}

/// Epochs and command ids are numbered across every connection this head makes: a node keeps
/// a session's across connections, and refuses an epoch or command it has already seen.
static NEXT_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static NEXT_COMMANDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn fresh_session_state() -> HeadSession {
    HeadSession::starting_at(NEXT_COMMANDS.fetch_add(1 << 32, std::sync::atomic::Ordering::Relaxed))
}

pub const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

pub const HANDSHAKE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

fn closed_at_handshake() -> String {
    "closed the connection at the handshake; check that --fleet-auth holds the token in the node's --auth-file".into()
}

/// Dials `addr` and opens the link: encrypted and keyed by the token when there is one.
fn handshake(addr: &str, auth: Vec<u8>) -> Result<(Endpoint, HelloAckF), DaemonError> {
    let node = |why: String| DaemonError::FleetNode { addr: addr.to_string(), why };
    let stream = Endpoint::tcp_connect_within(addr, CONNECT_TIMEOUT).map_err(|e| node(format!("unreachable ({e})")))?;
    let opened = if auth.is_empty() {
        Endpoint::from_tcp(stream)
    } else {
        Endpoint::from_tcp_secure(stream, &auth, superfluid_linkf::Role::Initiator, HANDSHAKE_TIMEOUT)
    };
    hello(opened, addr, auth, "--fleet-auth and the node's --auth-file differ")
}

/// The head's side of the handshake on a stream just opened to `peer`, dialed or accepted,
/// with each way it can fail named for the operator; `mismatch` says what differs when the
/// token is refused.
fn hello(opened: Result<Endpoint, superfluid_linkf::LinkFError>, peer: &str, auth: Vec<u8>, mismatch: &str) -> Result<(Endpoint, HelloAckF), DaemonError> {
    use superfluid_linkf::LinkFError;
    let node = |why: String| DaemonError::FleetNode { addr: peer.to_string(), why };
    let mut endpoint = match opened {
        Ok(ep) => ep,
        Err(LinkFError::Unauthorized) => return Err(node(format!("refused the fleet token: {mismatch}"))),
        Err(LinkFError::TimedOut) => {
            return Err(node(format!("did not finish the handshake within {} s", HANDSHAKE_TIMEOUT.as_secs())))
        }
        Err(LinkFError::Closed | LinkFError::Io(_)) if !auth.is_empty() => return Err(node(closed_at_handshake())),
        Err(e) => return Err(node(e.to_string())),
    };
    match endpoint.hello_as_head_within(auth, HANDSHAKE_TIMEOUT) {
        Ok(ack) => Ok((endpoint, ack)),
        Err(LinkFError::Unauthorized) => Err(node(format!("refused the fleet token: {mismatch}"))),
        Err(LinkFError::Closed | LinkFError::Io(_)) => Err(node(closed_at_handshake())),
        Err(LinkFError::TimedOut) => Err(node(format!(
            "did not answer the handshake within {} s; is superfluid-noded listening there?",
            HANDSHAKE_TIMEOUT.as_secs()
        ))),
        Err(e) => Err(node(e.to_string())),
    }
}

fn is_transient(e: &DaemonError) -> bool {
    matches!(
        e,
        DaemonError::Fleet(superfluid_linkf::LinkFError::Closed)
            | DaemonError::Fleet(superfluid_linkf::LinkFError::TimedOut)
            | DaemonError::Fleet(superfluid_linkf::LinkFError::Io(_))
    )
}

impl FleetHead {
    pub fn connect(addr: &str, auth: Vec<u8>) -> Result<FleetHead, DaemonError> {
        let (endpoint, node) = handshake(addr, auth.clone())?;
        Ok(FleetHead::from_parts(endpoint, node, addr.to_string(), auth))
    }

    /// A node that dialed in; the head still opens the handshake, encrypted when it has a token.
    pub fn accept(sock: std::net::TcpStream, auth: Vec<u8>) -> Result<FleetHead, DaemonError> {
        let peer = sock.peer_addr().map(|a| a.to_string()).unwrap_or_default();
        let opened = if auth.is_empty() {
            Endpoint::from_tcp(sock)
        } else {
            Endpoint::from_tcp_secure(sock, &auth, superfluid_linkf::Role::Initiator, HANDSHAKE_TIMEOUT)
        };
        let (endpoint, node) = hello(opened, &peer, auth.clone(), "its token differs from this head's")?;
        Ok(FleetHead::from_parts(endpoint, node, String::new(), auth))
    }

    fn from_parts(endpoint: Endpoint, node: HelloAckF, addr: String, auth: Vec<u8>) -> FleetHead {
        FleetHead {
            head: fresh_session_state(),
            endpoint,
            node,
            next_generation: 1,
            placements: BTreeMap::new(),
            addr,
            auth,
        }
    }

    fn reconnect_self(&mut self) -> Result<(), DaemonError> {
        let (addr, auth) = (self.addr.clone(), self.auth.clone());
        self.reconnect(&addr, auth)?;
        Ok(())
    }

    pub fn node(&self) -> &HelloAckF {
        &self.node
    }

    /// Reads what the node sent while the connection sat idle; an error means it is gone.
    pub fn drain_status(&mut self) -> Result<(), DaemonError> {
        for frame in self.endpoint.pump()? {
            if frame.msg_type == msg_type::NODE_STATUS {
                if let Ok(m) = postcard::from_bytes::<linkf::NodeStatusMsg>(&frame.payload) {
                    self.node.capacity = m.capacity;
                    if !m.loadable_models.is_empty() {
                        self.node.loadable_models = m.loadable_models;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn keepalive(&mut self) -> Result<(), DaemonError> {
        self.endpoint.send(msg_type::KEEPALIVE, &())?;
        self.drain_status()
    }

    pub fn knows(&self, session_id: u64) -> bool {
        self.head.current_epoch(session_id).is_some()
    }


    pub fn reconnect(&mut self, addr: &str, auth: Vec<u8>) -> Result<Vec<u64>, DaemonError> {
        let _ = self.endpoint.shutdown();
        if addr.is_empty() {
            return Err(DaemonError::FleetNode {
                addr: self.node.node_identity.clone(),
                why: "dialed in on its own; waiting for it to join again".into(),
            });
        }
        let (endpoint, node) = handshake(addr, auth)?;
        self.node = node;
        self.endpoint = endpoint;
        let rec = self.head.reconcile();
        self.endpoint.send(msg_type::RECONCILE, &rec)?;

        let mut received: std::collections::BTreeSet<(u64, u64, u64)> =
            std::collections::BTreeSet::new();
        let ack = loop {
            let mut got = None;
            for frame in self.endpoint.wait_frames()? {
                match frame.msg_type {
                    msg_type::RECONCILE_ACK => {
                        got = Some(postcard::from_bytes::<linkf::ReconcileAckMsg>(&frame.payload)?);
                    }
                    msg_type::TOKEN_EVENTS => {
                        let m: linkf::TokenEventsMsg = postcard::from_bytes(&frame.payload)?;
                        for ev in &m.events {
                            received.insert((ev.session_id, ev.generation_id, ev.event_seq));
                        }
                        for a in self.head.on_token_events(&m) {
                            self.endpoint.send(msg_type::WATERMARK_ACK, &a)?;
                        }
                    }
                    _ => {}
                }
            }
            if let Some(a) = got {
                break a;
            }
        };

        let mut expected: std::collections::BTreeSet<(u64, u64, u64)> =
            std::collections::BTreeSet::new();
        for s in &ack.sessions {
            if self.head.current_epoch(s.session_id).is_some() {
                for r in &s.retry_ranges {
                    for seq in r.from_seq..=r.to_seq {
                        let id = (s.session_id, r.generation_id, seq);
                        if !received.contains(&id) {
                            expected.insert(id);
                        }
                    }
                }
            }
        }
        while !expected.is_empty() {
            for frame in self.endpoint.wait_frames()? {
                if frame.msg_type != msg_type::TOKEN_EVENTS {
                    continue;
                }
                let m: linkf::TokenEventsMsg = postcard::from_bytes(&frame.payload)?;
                for ev in &m.events {
                    expected.remove(&(ev.session_id, ev.generation_id, ev.event_seq));
                }
                for a in self.head.on_token_events(&m) {
                    self.endpoint.send(msg_type::WATERMARK_ACK, &a)?;
                }
            }
        }

        // A session the node still holds may have missed the last append, written into the
        // connection that dropped; resend each one's latest content.
        let reassigned = self.head.on_reconcile_ack(&ack);
        for s in ack.sessions.iter().filter(|s| s.held && !reassigned.contains(&s.session_id)) {
            if let (Some(p), Some(epoch)) =
                (self.placements.get(&s.session_id).cloned(), self.head.current_epoch(s.session_id))
            {
                if !p.tokens.is_empty() {
                    self.send_append(s.session_id, epoch, &p.tokens)?;
                }
            }
        }
        let mut replaced = Vec::new();
        for session_id in reassigned {
            if let Some(p) = self.placements.get(&session_id).cloned() {
                if let Some(epoch) = self.head.current_epoch(session_id) {
                    let mut msg = self.head.assign_session(
                        session_id,
                        epoch,
                        p.lease,
                        p.qos_class,
                        &p.codec,
                    );
                    msg.params = encode_params(&p.params);
                    self.endpoint.send(msg_type::SESSION_ASSIGN, &msg)?;
                    if !p.tokens.is_empty() {
                        self.send_append(session_id, epoch, &p.tokens)?;
                    }
                    replaced.push(session_id);
                }
            }
        }
        Ok(replaced)
    }

    pub fn assign(
        &mut self,
        session_id: u64,
        params: GenParams,
        qos_class: u8,
        codec: &str,
        lease: Lease,
    ) -> Result<u64, DaemonError> {
        let entry = self.placements.entry(session_id).or_insert(Placement {
            params,
            qos_class,
            codec: codec.to_string(),
            lease,
            tokens: Vec::new(),
        });
        entry.params = params;
        entry.qos_class = qos_class;
        entry.codec = codec.to_string();
        entry.lease = lease;

        let epoch = NEXT_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut msg = self.head.assign_session(session_id, epoch, lease, qos_class, codec);
        msg.params = encode_params(&params);
        match self.endpoint.send(msg_type::SESSION_ASSIGN, &msg) {
            Ok(()) => Ok(epoch),
            Err(e) => {
                let e: DaemonError = e.into();
                if is_transient(&e) {
                    self.reconnect_self()?;
                    Ok(self.head.current_epoch(session_id).unwrap_or(epoch))
                } else {
                    Err(e)
                }
            }
        }
    }

    pub fn park(&mut self, session_id: u64) -> Result<(), DaemonError> {
        let epoch = self.head.current_epoch(session_id).unwrap_or(0);
        self.revoke(session_id, epoch)
    }

    /// Ends the session on the node, which frees the context it kept. `epoch` is the one it was
    /// last assigned, so this also reaches a session placed over another connection.
    pub fn revoke(&mut self, session_id: u64, epoch: u64) -> Result<(), DaemonError> {
        self.placements.remove(&session_id);
        let msg = match self.head.revoke(session_id) {
            Some(msg) => msg,
            None => linkf::SessionRevokeMsg { command_id: self.head.next_command(session_id), session_id, epoch },
        };
        self.endpoint.send(msg_type::SESSION_REVOKE, &msg)?;
        Ok(())
    }

    pub fn append(&mut self, session_id: u64, tokens: &[u32]) -> Result<(), DaemonError> {
        let epoch = self
            .head
            .current_epoch(session_id)
            .ok_or(DaemonError::Protocol("append before assign"))?;
        if let Some(p) = self.placements.get_mut(&session_id) {
            p.tokens = tokens.to_vec();
        }
        match self.send_append(session_id, epoch, tokens) {
            Ok(()) => Ok(()),
            Err(e) if is_transient(&e) => {
                self.reconnect_self()
            }
            Err(e) => Err(e),
        }
    }

    fn send_append(&mut self, session_id: u64, epoch: u64, tokens: &[u32]) -> Result<(), DaemonError> {
        let msg = AppendContentMsg {
            command_id: self.head.next_command(session_id),
            session_id,
            epoch,
            blocks: encode_tokens(tokens),
            spans: Vec::new(),
            media_refs: Vec::new(),
        };
        self.endpoint.send(msg_type::APPEND_CONTENT, &msg)?;
        Ok(())
    }

    pub fn generate(
        &mut self,
        session_id: u64,
        max_new_tokens: u64,
        max_wall_ms: u64,
    ) -> Result<FleetGeneration, DaemonError> {
        let generation_id = self.next_generation;
        self.next_generation += 1;
        for attempt in 0..2 {
            match self.generate_once(session_id, generation_id, max_new_tokens, max_wall_ms) {
                Ok(g) => return Ok(g),
                Err(e) if attempt == 0 && is_transient(&e) => {
                    self.reconnect_self()?;
                    if let Some(g) = self.recovered_generation(session_id, generation_id) {
                        self.head.retire_generation(session_id, generation_id);
                        return Ok(g);
                    }
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!("generate retried at most once")
    }

    pub fn generate_streaming(
        &mut self,
        session_id: u64,
        max_new_tokens: u64,
        max_wall_ms: u64,
        cancel: &std::sync::atomic::AtomicBool,
        mut on_tokens: impl FnMut(&[u32]),
    ) -> Result<FleetGeneration, DaemonError> {
        let mut delivered = false;
        let first = self.generate_streaming_once(session_id, max_new_tokens, max_wall_ms, cancel, |t| {
            delivered = true;
            on_tokens(t)
        });
        match first {
            // A connection the node dropped while idle fails before anything streams; nothing
            // reached the caller, so the generation can start over on a fresh one.
            Err(e) if !delivered && is_transient(&e) => {
                self.reconnect_self()?;
                self.generate_streaming_once(session_id, max_new_tokens, max_wall_ms, cancel, on_tokens)
            }
            r => r,
        }
    }

    fn generate_streaming_once(
        &mut self,
        session_id: u64,
        max_new_tokens: u64,
        max_wall_ms: u64,
        cancel: &std::sync::atomic::AtomicBool,
        mut on_tokens: impl FnMut(&[u32]),
    ) -> Result<FleetGeneration, DaemonError> {
        let generation_id = self.next_generation;
        self.next_generation += 1;
        let req = self
            .head
            .generate(
                session_id,
                generation_id,
                GenerationBudgetsMsg { max_new_tokens, max_wall_ms },
            )
            .ok_or(DaemonError::Protocol("generate before assign"))?;
        self.endpoint.send(msg_type::GENERATE_REQ, &req)?;

        let mut tokens = Vec::new();
        let mut finish = 0u32;
        let mut done = false;
        let mut cancel_sent = false;
        while !done {
            if !cancel_sent && cancel.load(std::sync::atomic::Ordering::Relaxed) {
                self.endpoint.send(
                    msg_type::GENERATE_CANCEL,
                    &GenerateCancelMsg { session_id, generation_id },
                )?;
                cancel_sent = true;
            }
            for frame in self.endpoint.wait_frames()? {
                match frame.msg_type {
                    msg_type::TOKEN_EVENTS => {
                        let m: linkf::TokenEventsMsg = postcard::from_bytes(&frame.payload)?;
                        for ev in &m.events {
                            if ev.session_id == session_id && ev.generation_id == generation_id {
                                match &ev.payload {
                                    TokenEventPayload::Tokens(t) => {
                                        tokens.extend_from_slice(t);
                                        on_tokens(t);
                                    }
                                    TokenEventPayload::Finish { reason } => {
                                        finish = *reason;
                                        done = true;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        for ack in self.head.on_token_events(&m) {
                            self.endpoint.send(msg_type::WATERMARK_ACK, &ack)?;
                        }
                    }
                    msg_type::LEASE_RENEW => {
                        let m: linkf::LeaseRenewMsg = postcard::from_bytes(&frame.payload)?;
                        if let Some(grant) = self.head.on_lease_renew(&m) {
                            self.endpoint.send(msg_type::LEASE_GRANT, &grant)?;
                        }
                    }
                    msg_type::NODE_STATUS => {
                        if let Ok(m) = postcard::from_bytes::<linkf::NodeStatusMsg>(&frame.payload)
                        {
                            self.node.capacity = m.capacity;
                            if !m.loadable_models.is_empty() {
                                self.node.loadable_models = m.loadable_models;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let expired = finish == u32::MAX;
        self.head.retire_generation(session_id, generation_id);
        Ok(FleetGeneration { tokens, finish, expired })
    }

    pub fn forward_stage(
        &mut self,
        tokens: &[u32],
        start_layer: u32,
        end_layer: u32,
        hidden_in: Option<Vec<u8>>,
    ) -> Result<superfluid_engine::StageOutput, DaemonError> {
        let request_id = self.next_generation;
        self.next_generation += 1;
        let req = linkf::StageForwardReqMsg {
            request_id,
            tokens: tokens.to_vec(),
            start_layer,
            end_layer,
            hidden_in,
        };
        self.endpoint.send(msg_type::STAGE_FORWARD_REQ, &req)?;
        loop {
            for frame in self.endpoint.wait_frames()? {
                if frame.msg_type == msg_type::STAGE_FORWARD_RESP {
                    let m: linkf::StageForwardRespMsg = postcard::from_bytes(&frame.payload)?;
                    if m.request_id != request_id {
                        continue;
                    }
                    return match (m.hidden, m.token) {
                        (Some(h), None) => Ok(superfluid_engine::StageOutput::Hidden(h)),
                        (None, Some(t)) => Ok(superfluid_engine::StageOutput::Token(t)),
                        _ => Err(DaemonError::Generation(
                            "stage node could not run the forward (engine lacks staged forward)".into(),
                        )),
                    };
                }
            }
        }
    }

    pub fn generate_chat(
        &mut self,
        session_id: u64,
        codec: &dyn TextCodec,
        messages: &[crate::codec::ChatMessage],
        tools: &[String],
        max_new_tokens: u64,
        max_wall_ms: u64,
    ) -> Result<Vec<FleetSegment>, DaemonError> {
        let prompt = codec
            .render_prompt_structured(messages, tools)
            .ok_or(DaemonError::Protocol("codec has no chat dialect"))?;
        self.append(session_id, &prompt)?;
        let gen = self.generate(session_id, max_new_tokens, max_wall_ms)?;

        let mut chan = crate::codec::primed_channelizer(codec, &prompt);
        let schemas = crate::codec::ToolSchemas::from_jsons(tools);
        let mut segments = Vec::new();
        for run in chan.split(&gen.tokens) {
            let text = codec.decode(&run.text);
            let tool = if run.channel == crate::wal::channel::TOOL_CALL && run.closes {
                codec.parse_tool_call_with(&text, Some(&schemas))
            } else {
                None
            };
            if !text.is_empty() || tool.is_some() {
                segments.push(FleetSegment {
                    channel: run.channel,
                    text,
                    tool,
                });
            }
        }
        Ok(segments)
    }

    fn recovered_generation(
        &self,
        session_id: u64,
        generation_id: u64,
    ) -> Option<FleetGeneration> {
        let epoch = self.head.current_epoch(session_id)?;
        let mut events: Vec<_> = self
            .head
            .wal()
            .session_events(session_id)
            .iter()
            .filter(|c| c.event.epoch == epoch && c.event.generation_id == generation_id)
            .collect();
        events.sort_by_key(|c| c.event.event_seq);
        let mut tokens = Vec::new();
        let mut finish = None;
        for c in events {
            match &c.event.payload {
                TokenEventPayload::Tokens(t) => tokens.extend_from_slice(t),
                TokenEventPayload::Finish { reason } => finish = Some(*reason),
                _ => {}
            }
        }
        let finish = finish?;
        Some(FleetGeneration {
            tokens,
            finish,
            expired: finish == crate::nodeagent::FINISH_EXPIRED,
        })
    }

    fn generate_once(
        &mut self,
        session_id: u64,
        generation_id: u64,
        max_new_tokens: u64,
        max_wall_ms: u64,
    ) -> Result<FleetGeneration, DaemonError> {
        let req = self
            .head
            .generate(
                session_id,
                generation_id,
                GenerationBudgetsMsg {
                    max_new_tokens,
                    max_wall_ms,
                },
            )
            .ok_or(DaemonError::Protocol("generate before assign"))?;
        self.endpoint.send(msg_type::GENERATE_REQ, &req)?;

        let mut tokens = Vec::new();
        let mut finish = 0u32;
        let mut expired = false;
        let mut done = false;
        while !done {
            for frame in self.endpoint.wait_frames()? {
                match frame.msg_type {
                    msg_type::TOKEN_EVENTS => {
                        let m: linkf::TokenEventsMsg = postcard::from_bytes(&frame.payload)?;
                        for ev in &m.events {
                            if ev.session_id == session_id && ev.generation_id == generation_id {
                                match &ev.payload {
                                    TokenEventPayload::Tokens(t) => tokens.extend_from_slice(t),
                                    TokenEventPayload::Finish { reason } => {
                                        finish = *reason;
                                        done = true;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        for ack in self.head.on_token_events(&m) {
                            self.endpoint.send(msg_type::WATERMARK_ACK, &ack)?;
                        }
                    }
                    msg_type::LEASE_RENEW => {
                        let m: linkf::LeaseRenewMsg = postcard::from_bytes(&frame.payload)?;
                        if let Some(grant) = self.head.on_lease_renew(&m) {
                            self.endpoint.send(msg_type::LEASE_GRANT, &grant)?;
                        }
                    }
                    msg_type::NODE_STATUS => {
                        if let Ok(m) = postcard::from_bytes::<linkf::NodeStatusMsg>(&frame.payload)
                        {
                            self.node.capacity = m.capacity;
                            if !m.loadable_models.is_empty() {
                                self.node.loadable_models = m.loadable_models;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        if finish == u32::MAX {
            expired = true;
        }
        self.head.retire_generation(session_id, generation_id);
        Ok(FleetGeneration {
            tokens,
            finish,
            expired,
        })
    }

    pub fn wal(&self) -> &superfluid_linkf::WalStub {
        self.head.wal()
    }
}

pub struct PipelineFleet {
    stages: Vec<FleetHead>,
    boundaries: Vec<u32>,
}

impl PipelineFleet {
    pub fn connect(nodes: Vec<(String, Vec<u8>)>, n_layers: u32) -> Result<PipelineFleet, DaemonError> {
        if nodes.is_empty() {
            return Err(DaemonError::Config("a pipeline needs at least one stage node"));
        }
        let p = nodes.len() as u32;
        let boundaries: Vec<u32> = (0..p).map(|i| ((i + 1) * n_layers) / p).collect();
        let mut stages = Vec::with_capacity(nodes.len());
        for (addr, auth) in nodes {
            stages.push(FleetHead::connect(&addr, auth)?);
        }
        Ok(PipelineFleet { stages, boundaries })
    }

    pub fn boundaries(&self) -> &[u32] {
        &self.boundaries
    }

    fn forward_one(&mut self, context: &[u32]) -> Result<u32, DaemonError> {
        let mut hidden: Option<Vec<u8>> = None;
        let mut start = 0u32;
        let n = self.stages.len();
        for i in 0..n {
            let end = self.boundaries[i];
            let empty: Vec<u32> = Vec::new();
            let tokens: &[u32] = if i == 0 { context } else { &empty };
            match self.stages[i].forward_stage(tokens, start, end, hidden.take())? {
                StageOutput::Hidden(h) => hidden = Some(h),
                StageOutput::Token(t) => return Ok(t),
            }
            start = end;
        }
        Err(DaemonError::Protocol(
            "pipeline finished every stage without a token — last stage must reach n_layers",
        ))
    }

    pub fn generate(&mut self, context: &[u32], max_tokens: u32) -> Result<Vec<u32>, DaemonError> {
        let mut ctx = context.to_vec();
        let mut out = Vec::with_capacity(max_tokens as usize);
        for _ in 0..max_tokens {
            let tok = self.forward_one(&ctx)?;
            out.push(tok);
            ctx.push(tok);
        }
        Ok(out)
    }
}
