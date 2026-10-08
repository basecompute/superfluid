//! The Link W server loop.

use std::collections::HashMap;
use std::ffi::CString;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

use superfluid_abi::{
    op_state, Artifact, CapabilityReq, StrategyRegistration, TapSpec,
    TokenRange, RecordArena, Status,
};
use superfluid_engine::{Engine, StageOutput};
use superfluid_proto::envelope::FrameClass;
use superfluid_proto::handshake::{negotiate, Hello, HelloAckW};
use superfluid_proto::linkw::{self, msg_type};
use superfluid_proto::{encode_frame, Frame, FrameDecoder};
use superfluid_shm::{
    fdpass, ControlSegment, RingRole, SharedRing, SharedRings, ShmSegment, CONTROL_TAG, LOGITS_RING,
    TOKEN_RING_IN, TOKEN_RING_OUT,
};

use crate::materialize::{
    desc_to_msg, events_to_msg, match_result_to_msg, materialize_plan, token_ref_from_msg,
};
use crate::WorkerError;

pub const PROTO_VERSION: u16 = 8;

// A decoding lane writes one token-out slot a tick and the daemon reads them
// all after it, so the ring bounds the lanes a tick can carry: room for every
// sequence a runtime holds (llama.cpp declares 255).
pub const TOKEN_OUT_SLOTS: u32 = 256;

pub fn default_ring_specs() -> Vec<linkw::RingSpec> {
    vec![
        linkw::RingSpec {
            ring_id: TOKEN_RING_IN,
            kind: linkw::RingKind::Tokens,
            slot_bytes: 16 + 4096 * 4,
            slots: 64,
        },
        linkw::RingSpec {
            ring_id: TOKEN_RING_OUT,
            kind: linkw::RingKind::Tokens,
            slot_bytes: 16 + 1024 * 4,
            slots: TOKEN_OUT_SLOTS,
        },
        linkw::RingSpec {
            ring_id: LOGITS_RING,
            kind: linkw::RingKind::Logits,
            slot_bytes: 16 + 64 * 4,
            slots: 32,
        },
    ]
}

pub struct WorkerConfig {
    pub engine_bundle_hash: [u8; 32],
}

enum OpBinding {
    Export { buf_id: u64 },
    Import,
}

pub struct WorkerServer<E: Engine> {
    engine: E,
    cfg: WorkerConfig,
    frames: UnixStream,
    fd_channel: UnixStream,
    seq: u64,
    rings: Option<SharedRings>,
    pending_rings: HashMap<u32, SharedRing>,
    pending_control: Option<ControlSegment>,
    ring_specs: Vec<linkw::RingSpec>,
    bufs: HashMap<u64, ShmSegment>,
    pending_buf_release: std::collections::HashSet<u64>,
    open_ops: HashMap<u64, OpBinding>,
    op_window: usize,
    decoder: FrameDecoder<fn(u16) -> bool>,
    pending: std::collections::VecDeque<Frame>,
    handshaken: bool,
    chosen_version: u16,
    last_mem: linkw::MemCountersMsg,
}

struct EmitSink<'a> {
    frames: &'a mut UnixStream,
    seq: &'a mut u64,
    ok: bool,
}

// Called by the engine, possibly from C: a panic must not unwind out of it,
// and the pointers are the engine's to get wrong.
unsafe extern "C" fn emit_tramp(
    user: *mut std::ffi::c_void,
    lane_tag: u64,
    tokens: *const u32,
    n_tokens: u32,
) {
    if user.is_null() || n_tokens == 0 {
        return;
    }
    // SAFETY: a non-null `user` is the EmitSink living across the enclosing
    // engine.tick call, which hands it to no one else.
    let sink = unsafe { &mut *(user as *mut EmitSink) };
    if !sink.ok {
        return;
    }
    if tokens.is_null() {
        sink.ok = false;
        return;
    }
    // SAFETY: the engine contract keeps `tokens` valid for `n_tokens`
    // elements for the duration of the callback; it is non-null.
    let toks = unsafe { std::slice::from_raw_parts(tokens, n_tokens as usize) };
    let sent = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let msg = linkw::TickEmitMsg { lane_tag, tokens: toks.to_vec() };
        postcard::to_stdvec(&msg).is_ok_and(|p| {
            send_frame(sink.frames, sink.seq, msg_type::TICK_EMIT, 0, &p).is_ok()
        })
    }));
    sink.ok = sent.unwrap_or(false);
}

fn send_frame(
    frames: &mut UnixStream,
    seq: &mut u64,
    msg_type: u16,
    correlation: u64,
    payload: &[u8],
) -> Result<(), WorkerError> {
    let class = if msg_type & 0x8000 != 0 {
        FrameClass::Droppable
    } else {
        FrameClass::Required
    };
    let bytes = encode_frame(PROTO_VERSION, msg_type, *seq, correlation, class, payload)?;
    *seq += 1;
    frames.write_all(&bytes)?;
    Ok(())
}

fn poll_transcribe_cancel(
    frames: &mut UnixStream,
    decoder: &mut FrameDecoder<fn(u16) -> bool>,
    pending: &mut std::collections::VecDeque<Frame>,
) -> bool {
    if frames.set_nonblocking(true).is_err() {
        return false;
    }
    let mut buf = [0u8; 64 * 1024];
    let read = frames.read(&mut buf);
    if frames.set_nonblocking(false).is_err() {
        return true;
    }
    match read {
        Ok(0) => return true,
        Ok(n) => decoder.feed(&buf[..n]),
        Err(e)
            if e.kind() == std::io::ErrorKind::WouldBlock
                || e.kind() == std::io::ErrorKind::TimedOut => {}
        Err(_) => return true,
    }
    let mut cancel = false;
    loop {
        match decoder.decode_next() {
            Ok(Some(frame)) => {
                if frame.msg_type == msg_type::TRANSCRIBE_CANCEL {
                    cancel = true;
                } else {
                    pending.push_back(frame);
                }
            }
            Ok(None) => return cancel,
            Err(_) => return true,
        }
    }
}

impl<E: Engine> WorkerServer<E> {
    pub fn new(engine: E, cfg: WorkerConfig, frames: UnixStream, fd_channel: UnixStream) -> Self {
        WorkerServer {
            engine,
            cfg,
            frames,
            fd_channel,
            seq: 1,
            rings: None,
            pending_rings: HashMap::new(),
            pending_control: None,
            ring_specs: default_ring_specs(),
            bufs: HashMap::new(),
            pending_buf_release: std::collections::HashSet::new(),
            open_ops: HashMap::new(),
            op_window: 16,
            decoder: FrameDecoder::new(linkw::is_known_type as fn(u16) -> bool),
            pending: std::collections::VecDeque::new(),
            handshaken: false,
            chosen_version: 0,
            last_mem: zero_mem(),
        }
    }

    pub fn with_ring_specs(mut self, specs: Vec<linkw::RingSpec>) -> Self {
        self.ring_specs = specs;
        self
    }

    pub fn serve(mut self) -> Result<(), WorkerError> {
        self.frames
            .set_read_timeout(Some(Duration::from_millis(5)))?;
        let mut buf = [0u8; 64 * 1024];
        loop {
            while let Some(frame) = self.pending.pop_front() {
                self.dispatch(frame)?;
            }
            match self.frames.read(&mut buf) {
                Ok(0) => return Ok(()),
                Ok(n) => self.decoder.feed(&buf[..n]),
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(e.into()),
            }
            while let Some(frame) = self.decoder.decode_next()? {
                self.dispatch(frame)?;
            }
            self.engine.pump();
            self.flush_op_completions()?;
        }
    }

    fn send(&mut self, msg_type: u16, correlation: u64, payload: &[u8]) -> Result<(), WorkerError> {
        send_frame(
            &mut self.frames,
            &mut self.seq,
            msg_type,
            correlation,
            payload,
        )
    }

    fn reply<T: serde::Serialize>(
        &mut self,
        msg_type: u16,
        correlation: u64,
        msg: &T,
    ) -> Result<(), WorkerError> {
        let payload = postcard::to_stdvec(msg)?;
        self.send(msg_type, correlation, &payload)
    }

    fn flush_op_completions(&mut self) -> Result<(), WorkerError> {
        let handles: Vec<u64> = self.open_ops.keys().copied().collect();
        for op in handles {
            let st = match self.engine.op_poll(op) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if st.state == op_state::PENDING || st.state == op_state::RUNNING {
                continue;
            }
            let binding = self.open_ops.remove(&op).expect("present");
            self.finalize_op(op, binding, st.state, st.error, st.bytes_moved)?;
        }
        Ok(())
    }

    fn finalize_op(
        &mut self,
        op: u64,
        binding: OpBinding,
        state: u8,
        error: u32,
        bytes_moved: u64,
    ) -> Result<(), WorkerError> {
        if state == op_state::DONE {
            if let OpBinding::Export { buf_id } = binding {
                if let Some(payload) = self.engine.take_op_output(op) {
                    if let Some(seg) = self.bufs.get(&buf_id) {
                        let _ = seg.write_at(0, &payload);
                    }
                }
            }
        }
        self.reply(
            msg_type::STATE_OP_DONE,
            0,
            &linkw::StateOpDoneMsg {
                op_handle: op,
                state,
                error,
                bytes_moved,
            },
        )?;
        if let OpBinding::Export { buf_id } = binding {
            if self.pending_buf_release.remove(&buf_id) {
                self.bufs.remove(&buf_id);
            }
        }
        Ok(())
    }

    fn dispatch(&mut self, frame: Frame) -> Result<(), WorkerError> {
        let corr = frame.seq;
        if !self.handshaken && frame.msg_type != msg_type::HELLO {
            return Err(WorkerError::Protocol("message before handshake"));
        }
        if self.handshaken && frame.proto_version != self.chosen_version {
            return Err(WorkerError::Protocol(
                "frame version differs from the negotiated version",
            ));
        }
        match frame.msg_type {
            msg_type::HELLO => {
                let hello: Hello = postcard::from_bytes(&frame.payload)?;
                let chosen = negotiate(hello.proto_versions, (PROTO_VERSION, PROTO_VERSION))
                    .ok_or(WorkerError::Protocol("no common proto version"))?;
                let descs: Vec<_> = self.engine.state_spaces().iter().map(desc_to_msg).collect();
                let ack = HelloAckW {
                    chosen_version: chosen,
                    engine_bundle_hash: self.cfg.engine_bundle_hash,
                    capabilities: self.engine.capability_descriptor().unwrap_or_default(),
                    state_space_descriptors: descs,
                    registered_strategies: Vec::new(),
                    limits: linkw::LinkWLimits {
                        max_frame: superfluid_proto::MAX_FRAME,
                        op_window: self.op_window as u32,
                        ring_specs: self.ring_specs.clone(),
                    },
                    kv_bits: self.engine.kv_bits(),
                };
                self.handshaken = true;
                self.chosen_version = chosen;
                self.reply(msg_type::HELLO_ACK_W, corr, &ack)
            }
            msg_type::RING_ATTACH => {
                let msg: linkw::RingAttachMsg = postcard::from_bytes(&frame.payload)?;
                let spec = self
                    .ring_specs
                    .iter()
                    .find(|s| s.ring_id == msg.ring_id)
                    .copied()
                    .ok_or(WorkerError::Protocol("attach for unknown ring id"))?;
                if msg.layout != spec {
                    return Err(WorkerError::Protocol("ring layout disagrees with spec"));
                }
                let (fd, tag) = fdpass::recv_fd(&self.fd_channel)?;
                if tag != msg.ring_id as u64 {
                    return Err(WorkerError::Protocol("fd tag does not match ring id"));
                }
                let seg =
                    ShmSegment::from_fd(fd, SharedRing::required_len(spec.slots, spec.slot_bytes))?;
                let role = if msg.ring_id == TOKEN_RING_IN {
                    RingRole::Reader
                } else {
                    RingRole::Writer
                };
                let ring = SharedRing::attach(seg, msg.ring_id, spec.slots, spec.slot_bytes, role)?;
                let attach = superfluid_engine::rings::RingAttachment {
                    ring_id: msg.ring_id,
                    kind: match spec.kind {
                        linkw::RingKind::Tokens => superfluid_abi::ring_kind::TOKENS,
                        linkw::RingKind::Logits => superfluid_abi::ring_kind::LOGITS,
                    },
                    role: if msg.ring_id == TOKEN_RING_IN {
                        superfluid_abi::ring_role::ENGINE_READS
                    } else {
                        superfluid_abi::ring_role::ENGINE_WRITES
                    },
                    slots: spec.slots,
                    slot_bytes: spec.slot_bytes,
                    base: ring.base_ptr(),
                    len: ring.seg_len(),
                };
                self.engine
                    .attach_ring(&attach)
                    .map_err(|_| WorkerError::Protocol("engine refused ring attachment"))?;
                self.pending_rings.insert(msg.ring_id, ring);
                if self.rings.is_none()
                    && [TOKEN_RING_IN, TOKEN_RING_OUT, LOGITS_RING]
                        .iter()
                        .all(|id| self.pending_rings.contains_key(id))
                {
                    self.rings = Some(SharedRings {
                        token_in: self.pending_rings.remove(&TOKEN_RING_IN).expect("checked"),
                        token_out: self.pending_rings.remove(&TOKEN_RING_OUT).expect("checked"),
                        logits: self.pending_rings.remove(&LOGITS_RING).expect("checked"),
                        control: self.pending_control.take(),
                    });
                }
                self.reply(
                    msg_type::RING_ACK,
                    corr,
                    &linkw::RingAckMsg {
                        ring_id: msg.ring_id,
                        generation_base: 1,
                    },
                )
            }
            msg_type::CONTROL_ATTACH => {
                let msg: linkw::ControlAttachMsg = postcard::from_bytes(&frame.payload)?;
                let (fd, tag) = fdpass::recv_fd(&self.fd_channel)?;
                if tag != CONTROL_TAG {
                    return Err(WorkerError::Protocol("fd tag is not the control segment's"));
                }
                let control = ControlSegment::attach(ShmSegment::from_fd(fd, msg.bytes as usize)?)?;
                match self.rings.as_mut() {
                    Some(rings) => rings.control = Some(control),
                    None => self.pending_control = Some(control),
                }
                self.reply(msg_type::CONTROL_ACK, corr, &linkw::ControlAckMsg { bytes: msg.bytes })
            }
            msg_type::TICK_SUBMIT => {
                let msg: linkw::TickSubmitMsg = postcard::from_bytes(&frame.payload)?;
                let Self { engine, frames, seq, rings, .. } = self;
                let rings = rings
                    .as_mut()
                    .ok_or(WorkerError::Protocol("tick before ring attach"))?;
                let (mut plan, arena) = materialize_plan(&msg.plan);
                let mut sink = EmitSink { frames, seq, ok: true };
                if msg.plan.flags & superfluid_abi::tick_flags::PARTIAL_EMITS != 0 {
                    plan.on_partial_emit = Some(emit_tramp);
                    plan.partial_emit_user = (&mut sink as *mut EmitSink).cast();
                }
                let result = match engine.tick(&plan, &arena, rings) {
                    Ok(ev) => linkw::TickResultMsg::Events(events_to_msg(ev)),
                    Err(s) => linkw::TickResultMsg::Rejected { status: s.raw() },
                };
                if let linkw::TickResultMsg::Events(ev) = &result {
                    self.last_mem = ev.mem;
                    let done: Vec<linkw::OpCompleteMsg> = ev.op_completions.clone();
                    for c in done {
                        if let Some(binding) = self.open_ops.remove(&c.op) {
                            self.finalize_op(c.op, binding, c.state, c.error, c.bytes_moved)?;
                        }
                    }
                }
                self.reply(msg_type::TICK_RESULT, corr, &result)
            }
            msg_type::MATCH_REQ => {
                let msg: linkw::MatchReqMsg = postcard::from_bytes(&frame.payload)?;
                let rings = self
                    .rings
                    .as_ref()
                    .ok_or(WorkerError::Protocol("match before ring attach"))?;
                let span = token_ref_from_msg(&msg.span);
                use superfluid_engine::Rings;
                let tokens = rings
                    .read_tokens(&span)
                    .map_err(|_| WorkerError::Protocol("stale span in match"))?;
                let res = self
                    .engine
                    .space_match_ref(msg.space_id, &span, &tokens, &msg.media_deps)
                    .map(match_result_to_msg)
                    .unwrap_or(linkw::MatchResMsg { spaces: Vec::new() });
                self.reply(msg_type::MATCH_RES, corr, &res)
            }
            msg_type::SEED_ACQUIRE => {
                let msg: linkw::SeedAcquireMsg = postcard::from_bytes(&frame.payload)?;
                let rings = self
                    .rings
                    .as_ref()
                    .ok_or(WorkerError::Protocol("seed before ring attach"))?;
                use superfluid_engine::Rings;
                let span = token_ref_from_msg(&msg.span);
                let grant = match rings.read_tokens(&span) {
                    Ok(tokens) => match self.engine.seed_acquire_ref(
                        &span,
                        &tokens,
                        msg.prefix_len,
                        msg.determinism_class,
                    ) {
                        Ok(handle) => linkw::SeedGrantMsg::Granted {
                            seed_handle: handle,
                            ttl_ms: 1000,
                            ttl_ticks: self.engine.seed_lease_ticks(),
                        },
                        Err(s) => linkw::SeedGrantMsg::Refused { status: s.raw() },
                    },
                    Err(_) => linkw::SeedGrantMsg::Refused {
                        status: Status::RejectBadRefGeneration.raw(),
                    },
                };
                self.reply(msg_type::SEED_GRANT, corr, &grant)
            }
            msg_type::SEED_RELEASE => {
                let msg: linkw::SeedReleaseMsg = postcard::from_bytes(&frame.payload)?;
                let _ = self.engine.seed_release(msg.seed_handle);
                Ok(())
            }
            msg_type::STATE_OP_SUBMIT => {
                let msg: linkw::StateOpSubmitMsg = postcard::from_bytes(&frame.payload)?;
                let reply = if self.open_ops.len() >= self.op_window {
                    linkw::StateOpAcceptedMsg::Rejected {
                        status: Status::Busy.raw(),
                    }
                } else {
                    self.submit_state_op(&msg.op)
                };
                self.reply(msg_type::STATE_OP_ACCEPTED, corr, &reply)
            }
            msg_type::OP_CANCEL => {
                let msg: linkw::OpCancelMsg = postcard::from_bytes(&frame.payload)?;
                let _ = self.engine.op_cancel(msg.op_handle);
                Ok(())
            }
            msg_type::STATE_SYNC_REQ => {
                let msg: linkw::StateSyncReqMsg = postcard::from_bytes(&frame.payload)?;
                let res = self.state_sync(&msg);
                self.reply(msg_type::STATE_SYNC_RES, corr, &res)
            }
            msg_type::STRATEGY_REGISTER => {
                let msg: linkw::StrategyRegisterMsg = postcard::from_bytes(&frame.payload)?;
                let res = self.register_strategy(&msg);
                self.reply(msg_type::STRATEGY_GRANT, corr, &res)
            }
            msg_type::BUF_ALLOC => {
                let _msg: linkw::BufAllocMsg = postcard::from_bytes(&frame.payload)?;
                Ok(())
            }
            msg_type::BUF_ATTACH => {
                let msg: linkw::BufAttachMsg = postcard::from_bytes(&frame.payload)?;
                let (fd, tag) = fdpass::recv_fd(&self.fd_channel)?;
                if tag != msg.buf_id {
                    return Err(WorkerError::Protocol("fd tag does not match buf id"));
                }
                let seg = ShmSegment::from_fd(fd, msg.len as usize)?;
                self.bufs.insert(msg.buf_id, seg);
                Ok(())
            }
            msg_type::BUF_RELEASE => {
                let msg: linkw::BufReleaseMsg = postcard::from_bytes(&frame.payload)?;
                let bound = self
                    .open_ops
                    .values()
                    .any(|b| matches!(b, OpBinding::Export { buf_id } if *buf_id == msg.buf_id));
                if bound {
                    self.pending_buf_release.insert(msg.buf_id);
                } else {
                    self.bufs.remove(&msg.buf_id);
                }
                Ok(())
            }
            msg_type::PING => {
                let _msg: linkw::PingMsg = postcard::from_bytes(&frame.payload)?;
                let pong = linkw::PongMsg {
                    queue_depth: self.open_ops.len() as u32,
                    mem: self.last_mem,
                };
                self.reply(msg_type::PONG, corr, &pong)
            }
            msg_type::DRAIN_REQ => {
                while !self.open_ops.is_empty() {
                    self.engine.pump();
                    self.flush_op_completions()?;
                }
                self.reply(msg_type::DRAINED, corr, &linkw::DrainedMsg)
            }
            msg_type::TELEMETRY => Ok(()),
            msg_type::TRANSCRIBE_CANCEL => Ok(()),
            _ => Err(WorkerError::Protocol("unhandled required message")),
        }
    }

    fn submit_state_op(&mut self, op: &linkw::StateOpKind) -> linkw::StateOpAcceptedMsg {
        let result: Result<(u64, OpBinding), Status> = match *op {
            linkw::StateOpKind::Snapshot {
                sequence,
                space_id,
                boundary_pos,
                buf_id,
                sizing_gen,
            } => {
                let Some(seg) = self.bufs.get(&buf_id) else {
                    return linkw::StateOpAcceptedMsg::Rejected {
                        status: Status::UnknownHandle.raw(),
                    };
                };
                let len = seg.len() as u64;
                self.engine
                    .space_snapshot(sequence, space_id, boundary_pos, len, sizing_gen)
                    .map(|h| (h, OpBinding::Export { buf_id }))
            }
            linkw::StateOpKind::Demote {
                sequence,
                space_id,
                range,
                encoding,
                buf_id,
                sizing_gen,
            } => {
                let Some(seg) = self.bufs.get(&buf_id) else {
                    return linkw::StateOpAcceptedMsg::Rejected {
                        status: Status::UnknownHandle.raw(),
                    };
                };
                let len = seg.len() as u64;
                self.engine
                    .space_demote(
                        sequence,
                        space_id,
                        TokenRange {
                            start: range.start,
                            end: range.end,
                        },
                        encoding,
                        len,
                        sizing_gen,
                    )
                    .map(|h| (h, OpBinding::Export { buf_id }))
            }
            linkw::StateOpKind::Restore {
                sequence,
                space_id,
                buf_id,
            } => match self.read_sealed(buf_id) {
                Ok(bytes) => self
                    .engine
                    .space_restore(sequence, space_id, &bytes)
                    .map(|h| (h, OpBinding::Import)),
                Err(s) => Err(s),
            },
            linkw::StateOpKind::Promote {
                sequence,
                space_id,
                range,
                buf_id,
            } => match self.read_sealed(buf_id) {
                Ok(bytes) => self
                    .engine
                    .space_promote(
                        sequence,
                        space_id,
                        TokenRange {
                            start: range.start,
                            end: range.end,
                        },
                        &bytes,
                    )
                    .map(|h| (h, OpBinding::Import)),
                Err(s) => Err(s),
            },
        };
        match result {
            Ok((handle, binding)) => {
                self.open_ops.insert(handle, binding);
                linkw::StateOpAcceptedMsg::Accepted { op_handle: handle }
            }
            Err(s) => linkw::StateOpAcceptedMsg::Rejected { status: s.raw() },
        }
    }

    fn read_sealed(&self, buf_id: u64) -> Result<Vec<u8>, Status> {
        const ENVELOPE_BYTES: usize = std::mem::size_of::<superfluid_abi::StateEnvelope>();
        let seg = self.bufs.get(&buf_id).ok_or(Status::UnknownHandle)?;
        if seg.len() < ENVELOPE_BYTES {
            return Err(Status::EnvelopeMismatch);
        }
        let mut head = vec![0u8; ENVELOPE_BYTES];
        seg.read_at(0, &mut head)
            .map_err(|_| Status::EnvelopeMismatch)?;
        let off = std::mem::offset_of!(superfluid_abi::StateEnvelope, payload_len);
        let payload_len = u64::from_le_bytes(head[off..off + 8].try_into().expect("8 bytes"));
        let payload_len = usize::try_from(payload_len).map_err(|_| Status::EnvelopeMismatch)?;
        let total = ENVELOPE_BYTES
            .checked_add(payload_len)
            .ok_or(Status::EnvelopeMismatch)?;
        if seg.len() < total {
            return Err(Status::EnvelopeMismatch);
        }
        let mut out = vec![0u8; total];
        seg.read_at(0, &mut out)
            .map_err(|_| Status::EnvelopeMismatch)?;
        Ok(out)
    }

    fn state_sync(&mut self, msg: &linkw::StateSyncReqMsg) -> linkw::StateSyncResMsg {
        let result = match *msg {
            linkw::StateSyncReqMsg::Fork {
                parent_sequence,
                flags,
            } => self.engine.seq_fork(parent_sequence, flags).map(|child| {
                linkw::StateSyncOkMsg::Fork {
                    child_sequence: child,
                }
            }),
            linkw::StateSyncReqMsg::Trim {
                sequence,
                space_id,
                new_len,
            } => self
                .engine
                .space_trim(sequence, space_id, new_len)
                .map(|()| linkw::StateSyncOkMsg::Trim),
            linkw::StateSyncReqMsg::ExportSize {
                sequence,
                space_id,
                range,
                encoding,
            } => self
                .engine
                .space_export_size(
                    sequence,
                    space_id,
                    TokenRange {
                        start: range.start,
                        end: range.end,
                    },
                    encoding,
                )
                .map(
                    |(required_bytes, sizing_gen)| linkw::StateSyncOkMsg::ExportSize {
                        required_bytes,
                        sizing_gen,
                    },
                ),
            linkw::StateSyncReqMsg::CacheEvict {
                bytes_target,
                ref policy,
            } => self
                .engine
                .cache_evict(
                    policy.evictable_cache_classes,
                    policy.protected_quota_bytes,
                    policy.max_evict_bytes,
                    bytes_target,
                )
                .map(|bytes_freed| linkw::StateSyncOkMsg::CacheEvict { bytes_freed }),
            linkw::StateSyncReqMsg::CacheEvictEntries { ref keys } => {
                let raw: Vec<([u8; 32], [u8; 32])> =
                    keys.iter().map(|k| (k.compat_key, k.provenance_digest)).collect();
                self.engine
                    .cache_evict_entries(&raw)
                    .map(|()| linkw::StateSyncOkMsg::CacheEvictEntries)
            }
            linkw::StateSyncReqMsg::SnapshotBoundary {
                sequence,
                space_id,
                cap,
            } => self
                .engine
                .space_snapshot_boundary(sequence, space_id, cap)
                .map(|boundary| linkw::StateSyncOkMsg::SnapshotBoundary { boundary }),
            linkw::StateSyncReqMsg::LaneSequence { lane_tag } => self
                .engine
                .lane_sequence(lane_tag)
                .map(|sequence| linkw::StateSyncOkMsg::LaneSequence { sequence })
                .ok_or(Status::UnknownHandle),
            linkw::StateSyncReqMsg::CreateSequence => self
                .engine
                .create_sequence()
                .map(|sequence| linkw::StateSyncOkMsg::CreateSequence { sequence }),
            linkw::StateSyncReqMsg::FreeSequence { sequence } => self
                .engine
                .free_sequence(sequence)
                .map(|()| linkw::StateSyncOkMsg::FreeSequence),
            linkw::StateSyncReqMsg::CreateGrammarStructural { ref tag_json } => self
                .engine
                .grammar_create_structural(tag_json)
                .map(|grammar_handle| linkw::StateSyncOkMsg::CreateGrammar { grammar_handle }),
            linkw::StateSyncReqMsg::CreateGrammar { ref json_schema } => self
                .engine
                .grammar_create(json_schema)
                .map(|grammar_handle| linkw::StateSyncOkMsg::CreateGrammar { grammar_handle }),
            linkw::StateSyncReqMsg::FreeGrammar { grammar_handle } => self
                .engine
                .grammar_free(grammar_handle)
                .map(|()| linkw::StateSyncOkMsg::FreeGrammar),
            linkw::StateSyncReqMsg::CreateLogitBias { ref tokens, ref values } => {
                let vals: Vec<f32> = values.iter().map(|b| f32::from_bits(*b)).collect();
                self.engine
                    .logit_bias_create(tokens, &vals)
                    .map(|handle| linkw::StateSyncOkMsg::CreateLogitBias { handle })
            }
            linkw::StateSyncReqMsg::FreeLogitBias { handle } => self
                .engine
                .logit_bias_free(handle)
                .map(|()| linkw::StateSyncOkMsg::FreeLogitBias),
            linkw::StateSyncReqMsg::Embed { ref tokens } => self.engine.embed(tokens).map(|v| {
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for f in v {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                linkw::StateSyncOkMsg::Embed { embedding: bytes }
            }),
            linkw::StateSyncReqMsg::Transcribe {
                ref audio_path,
                ref language,
                translate,
                timestamps,
                ref prompt,
                stream,
            } => {
                let params = superfluid_engine::TranscribeParams {
                    language: language.clone(),
                    translate,
                    timestamps,
                    prompt: prompt.clone(),
                };
                let Self { engine, frames, seq, decoder, pending, .. } = self;
                let mut sink = |start_ms: i32, end_ms: i32, text: &str| -> bool {
                    let msg = linkw::TranscribeSegmentMsg {
                        start_ms,
                        end_ms,
                        text: text.to_string(),
                    };
                    let sent = postcard::to_stdvec(&msg).is_ok_and(|p| {
                        send_frame(frames, seq, msg_type::TRANSCRIBE_SEGMENT, 0, &p).is_ok()
                    });
                    sent && !poll_transcribe_cancel(frames, decoder, pending)
                };
                if stream {
                    engine.transcribe(audio_path, &params, Some(&mut sink))
                } else {
                    engine.transcribe(audio_path, &params, None)
                }
            }
            .map(|t| linkw::StateSyncOkMsg::Transcribe {
                    text: t.text,
                    language: t.language,
                    duration_ms: t.duration_ms,
                    segments: t
                        .segments
                        .into_iter()
                        .map(|s| linkw::TranscriptSegmentMsg {
                            start_ms: s.start_ms,
                            end_ms: s.end_ms,
                            text: s.text,
                            avg_logprob_bits: s.avg_logprob.to_bits(),
                            no_speech_prob_bits: s.no_speech_prob.to_bits(),
                            compression_ratio_bits: s.compression_ratio.to_bits(),
                            temperature_bits: s.temperature.to_bits(),
                        })
                        .collect(),
                }),
            linkw::StateSyncReqMsg::LoraLoad { ref adapter_path } => self
                .engine
                .lora_load(adapter_path)
                .map(|()| linkw::StateSyncOkMsg::LoraId { id: self.engine.lora_id() }),
            linkw::StateSyncReqMsg::LoraUnload => self
                .engine
                .lora_unload()
                .map(|()| linkw::StateSyncOkMsg::LoraId { id: self.engine.lora_id() }),
            linkw::StateSyncReqMsg::LoraId => {
                Ok(linkw::StateSyncOkMsg::LoraId { id: self.engine.lora_id() })
            }
            linkw::StateSyncReqMsg::Capabilities => Ok(linkw::StateSyncOkMsg::Capabilities {
                json: self.engine.capability_descriptor().unwrap_or_default(),
            }),
            linkw::StateSyncReqMsg::ForwardStage {
                ref tokens,
                start_layer,
                end_layer,
                ref hidden_in,
            } => self
                .engine
                .forward_stage(tokens, start_layer, end_layer, hidden_in.clone())
                .map(|out| match out {
                    StageOutput::Hidden(h) => linkw::StateSyncOkMsg::ForwardStage {
                        hidden: Some(h),
                        token: None,
                    },
                    StageOutput::Token(t) => linkw::StateSyncOkMsg::ForwardStage {
                        hidden: None,
                        token: Some(t),
                    },
                }),
            linkw::StateSyncReqMsg::MediaProbe { ref image_path } => {
                self.engine.media_probe(image_path).map(|i| linkw::StateSyncOkMsg::MediaProbe {
                    info: linkw::MediaInfoMsg {
                        n_tokens: i.n_tokens,
                        image_token_id: i.image_token_id,
                        boi_token_id: i.boi_token_id,
                        eoi_token_id: i.eoi_token_id,
                        preprocess_fp: i.preprocess_fp,
                    },
                })
            }
            linkw::StateSyncReqMsg::MediaEncode { ref image_path } => {
                self.engine
                    .media_encode(image_path)
                    .map(|(media_handle, i)| linkw::StateSyncOkMsg::MediaEncode {
                        media_handle,
                        info: linkw::MediaInfoMsg {
                            n_tokens: i.n_tokens,
                            image_token_id: i.image_token_id,
                            boi_token_id: i.boi_token_id,
                            eoi_token_id: i.eoi_token_id,
                            preprocess_fp: i.preprocess_fp,
                        },
                    })
            }
            linkw::StateSyncReqMsg::MediaRelease { media_handle } => self
                .engine
                .media_release(media_handle)
                .map(|()| linkw::StateSyncOkMsg::MediaRelease),
            linkw::StateSyncReqMsg::MediaBind {
                lane_tag,
                media_handle,
                token_offset,
            } => self
                .engine
                .media_bind(lane_tag, media_handle, token_offset)
                .map(|()| linkw::StateSyncOkMsg::MediaBind),
            linkw::StateSyncReqMsg::AdoptSequence { sequence, ref span } => {
                use superfluid_engine::Rings;
                let span_ref = token_ref_from_msg(span);
                match self.rings.as_ref() {
                    None => Err(Status::RejectBadRefGeneration),
                    Some(rings) => match rings.read_tokens(&span_ref) {
                        Ok(tokens) => self
                            .engine
                            .seed_adopt_ref(sequence, &span_ref, &tokens)
                            .map(|seed_handle| linkw::StateSyncOkMsg::AdoptSequence { seed_handle }),
                        Err(_) => Err(Status::RejectBadRefGeneration),
                    },
                }
            }
            linkw::StateSyncReqMsg::PublishSequence { sequence, ref span } => {
                use superfluid_engine::Rings;
                let span = token_ref_from_msg(span);
                match self.rings.as_ref() {
                    None => Err(Status::RejectBadRefGeneration),
                    Some(rings) => match rings.read_tokens(&span) {
                        Ok(tokens) => self
                            .engine
                            .publish_sequence(sequence, &tokens)
                            .map(|()| linkw::StateSyncOkMsg::PublishSequence),
                        Err(_) => Err(Status::RejectBadRefGeneration),
                    },
                }
            }
            linkw::StateSyncReqMsg::Unpublish { ref span } => {
                use superfluid_engine::Rings;
                let span = token_ref_from_msg(span);
                match self.rings.as_ref() {
                    None => Err(Status::RejectBadRefGeneration),
                    Some(rings) => match rings.read_tokens(&span) {
                        Ok(tokens) => self.engine.unpublish(&tokens).map(|()| linkw::StateSyncOkMsg::Unpublish),
                        Err(_) => Err(Status::RejectBadRefGeneration),
                    },
                }
            }
        };
        match result {
            Ok(ok) => linkw::StateSyncResMsg::Ok(ok),
            Err(s) => linkw::StateSyncResMsg::Err { status: s.raw() },
        }
    }

    fn register_strategy(&mut self, msg: &linkw::StrategyRegisterMsg) -> linkw::StrategyGrantMsg {
        let strategy_id = CString::new(msg.strategy_id.as_str()).unwrap_or_default();
        let impl_version = CString::new(msg.impl_version.as_str()).unwrap_or_default();
        let load_paths: Vec<CString> = msg
            .artifacts
            .iter()
            .map(|a| CString::new(a.load_path.as_str()).unwrap_or_default())
            .collect();
        let archs: Vec<CString> = msg
            .target_archs
            .iter()
            .map(|a| CString::new(a.as_str()).unwrap_or_default())
            .collect();
        let mut arena = RecordArena::new();
        let artifacts: Vec<Artifact> = msg
            .artifacts
            .iter()
            .zip(&load_paths)
            .map(|(a, p)| Artifact {
                role: a.role,
                _pad0: 0,
                content_hash: a.content_hash,
                byte_size: a.byte_size,
                load_path: p.as_ptr(),
            })
            .collect();
        let taps: Vec<TapSpec> = msg
            .taps
            .iter()
            .map(|t| TapSpec {
                layer: t.layer,
                tensor: t.tensor,
                dtype: t.dtype,
                layout: t.layout,
            })
            .collect();
        let capabilities: Vec<CapabilityReq> = msg
            .capabilities
            .iter()
            .map(|c| CapabilityReq {
                kind_id: c.kind_id,
                _pad0: 0,
                params: arena.push_records(&c.params),
            })
            .collect();
        let arch_ptrs: Vec<u64> = archs.iter().map(|a| a.as_ptr() as u64).collect();

        let reg = StrategyRegistration {
            struct_size: std::mem::size_of::<StrategyRegistration>() as u64,
            strategy_id: strategy_id.as_ptr(),
            impl_version: impl_version.as_ptr(),
            impl_hash: msg.impl_hash,
            config_hash: msg.config_hash,
            artifacts: arena.push_records(&artifacts),
            taps: arena.push_records(&taps),
            capabilities: arena.push_records(&capabilities),
            target_archs: arena.push_records(&arch_ptrs),
            kernel_caps_required: msg.kernel_caps_required,
            _pad0: 0,
            est_state_bytes: msg.est_state_bytes,
            claimed_exactness: msg.claimed_exactness,
            _pad1: [0; 3],
            rng_contract_version: msg.rng_contract_version,
        };

        match self.engine.strategy_register(&reg) {
            Ok(grant) => {
                let bounds = superfluid_abi::ArenaBounds {
                    base: 0,
                    len: usize::MAX,
                };
                // SAFETY: engine-owned grant arrays, next-call lifetime;
                // serialized immediately.
                let certificates = unsafe {
                    superfluid_abi::array::read_array::<superfluid_abi::ExactnessCert>(
                        &grant.certificates,
                        &bounds,
                    )
                    .map(|it| {
                        it.map(|c| {
                            let ids = superfluid_abi::array::read_array::<[u8; 32]>(
                                &c.admissible_host_identities,
                                &bounds,
                            )
                            .map(|h| h.collect())
                            .unwrap_or_default();
                            linkw::ExactnessCertMsg {
                                cert_id: c.cert_id,
                                exactness: c.exactness,
                                sampling_modes: c.sampling_modes,
                                param_domain: c.param_domain,
                                grammar_allowed: c.grammar_allowed != 0,
                                logit_bias_allowed: c.logit_bias_allowed != 0,
                                verification_shape_class: c.verification_shape_class,
                                admissible_host_identities: ids,
                            }
                        })
                        .collect()
                    })
                    .unwrap_or_default()
                };
                let state_spaces = {
                    // SAFETY: as above.
                    unsafe {
                        superfluid_abi::array::read_array::<superfluid_abi::StateSpaceDesc>(
                            &grant.state_spaces,
                            &bounds,
                        )
                        .map(|it| it.map(|d| desc_to_msg(&d)).collect())
                        .unwrap_or_default()
                    }
                };
                linkw::StrategyGrantMsg::Granted(linkw::StrategyGrantOkMsg {
                    strategy_slot: grant.strategy_slot,
                    certificates,
                    state_spaces,
                    reserved_bytes: grant.reserved_bytes,
                })
            }
            Err(s) => linkw::StrategyGrantMsg::Refused { status: s.raw() },
        }
    }
}

fn zero_mem() -> linkw::MemCountersMsg {
    linkw::MemCountersMsg {
        allocated_bytes: 0,
        host_retained_bytes: 0,
        pool_blocks_total: 0,
        pool_blocks_used: 0,
        pool_bytes_evictable: 0,
        tentative_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_emit_trampoline_refuses_null_pointers_from_the_engine() {
        let (mut frames, mut peer) = UnixStream::pair().unwrap();
        let mut seq = 1;
        let toks = [4u32, 5, 6];
        // SAFETY: the trampoline is handed exactly what a careless engine
        // could pass; every pointer that is not null is live.
        unsafe {
            emit_tramp(std::ptr::null_mut(), 7, toks.as_ptr(), 3);
            let mut sink = EmitSink { frames: &mut frames, seq: &mut seq, ok: true };
            emit_tramp((&mut sink as *mut EmitSink).cast(), 7, std::ptr::null(), 3);
            assert!(!sink.ok, "a null token pointer ends the partial emits");
            let mut sink = EmitSink { frames: &mut frames, seq: &mut seq, ok: true };
            emit_tramp((&mut sink as *mut EmitSink).cast(), 7, toks.as_ptr(), 3);
            assert!(sink.ok, "a good emit is sent");
        }
        drop(frames);
        let mut sent = Vec::new();
        peer.read_to_end(&mut sent).unwrap();
        assert!(!sent.is_empty(), "the good emit reached the link");
    }
}
