//! The Link W client.

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

use superfluid_proto::envelope::FrameClass;
use superfluid_proto::handshake::{Hello, HelloAckW, LinkRole};
use superfluid_proto::linkw::{self, msg_type};
use superfluid_proto::{encode_frame, Frame, FrameDecoder};
use superfluid_shm::{
    fdpass, ControlSegment, RingRole, SharedRing, SharedRings, ShmSegment, CONTROL_BYTES, CONTROL_TAG,
    LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT,
};

use crate::AgentError;

pub const PROTO_VERSION: u16 = 8;

pub struct WorkerClient {
    frames: UnixStream,
    fd_channel: UnixStream,
    decoder: FrameDecoder<fn(u16) -> bool>,
    seq: u64,
    pub hello: HelloAckW,
    rings: SharedRings,
    bufs: HashMap<u64, ShmSegment>,
    next_buf_id: u64,
    pub op_done: VecDeque<linkw::StateOpDoneMsg>,
}

impl WorkerClient {
    pub fn connect(frames: UnixStream, fd_channel: UnixStream) -> Result<WorkerClient, AgentError> {
        frames.set_read_timeout(Some(Duration::from_millis(50)))?;
        let mut client = WorkerClient {
            frames,
            fd_channel,
            decoder: FrameDecoder::new(linkw::is_known_type as fn(u16) -> bool),
            seq: 1,
            hello: HelloAckW {
                chosen_version: 0,
                engine_bundle_hash: [0; 32],
                capabilities: String::new(),
                state_space_descriptors: Vec::new(),
                registered_strategies: Vec::new(),
                limits: linkw::LinkWLimits {
                    max_frame: 0,
                    op_window: 0,
                    ring_specs: Vec::new(),
                },
                kv_bits: 0,
            },
            rings: SharedRings {
                token_in: SharedRing::create(TOKEN_RING_IN, 1, 32, RingRole::Writer)?,
                token_out: SharedRing::create(TOKEN_RING_OUT, 1, 32, RingRole::Reader)?,
                logits: SharedRing::create(LOGITS_RING, 1, 32, RingRole::Reader)?,
                control: None,
            },
            bufs: HashMap::new(),
            next_buf_id: 1,
            op_done: VecDeque::new(),
        };

        let hello = Hello {
            proto_versions: (PROTO_VERSION, PROTO_VERSION),
            link_role: LinkRole::NodeAgent,
            auth: Vec::new(),
        };
        let ack: HelloAckW = client.request(msg_type::HELLO, &hello, msg_type::HELLO_ACK_W)?;
        client.hello = ack;

        for spec in client.hello.limits.ring_specs.clone() {
            let role = if spec.ring_id == TOKEN_RING_IN {
                RingRole::Writer
            } else {
                RingRole::Reader
            };
            let ring = SharedRing::create(spec.ring_id, spec.slots, spec.slot_bytes, role)?;
            let attach = linkw::RingAttachMsg {
                ring_id: spec.ring_id,
                layout: spec,
            };
            let payload = postcard::to_stdvec(&attach)?;
            let sent_seq = client.send(msg_type::RING_ATTACH, 0, &payload)?;
            fdpass::send_fd(
                client.frames_fd_channel(),
                ring.raw_fd(),
                spec.ring_id as u64,
            )?;
            let ack_frame = client.wait_correlated(sent_seq)?;
            let _ack: linkw::RingAckMsg = postcard::from_bytes(&ack_frame.payload)?;
            match spec.ring_id {
                TOKEN_RING_IN => client.rings.token_in = ring,
                TOKEN_RING_OUT => client.rings.token_out = ring,
                LOGITS_RING => client.rings.logits = ring,
                _ => return Err(AgentError::Protocol("worker advertised unknown ring id")),
            }
        }

        let control = ControlSegment::create()?;
        let payload = postcard::to_stdvec(&linkw::ControlAttachMsg { bytes: CONTROL_BYTES as u32 })?;
        let sent_seq = client.send(msg_type::CONTROL_ATTACH, 0, &payload)?;
        fdpass::send_fd(client.frames_fd_channel(), control.raw_fd(), CONTROL_TAG)?;
        let ack_frame = client.wait_correlated(sent_seq)?;
        let _ack: linkw::ControlAckMsg = postcard::from_bytes(&ack_frame.payload)?;
        client.rings.control = Some(control);
        Ok(client)
    }

    /// The daemon's control words for this worker (end the running tick at
    /// the next step boundary; prefill in short steps), settable from any
    /// thread.
    pub fn yield_signal(&self) -> Option<superfluid_shm::YieldSignal> {
        self.rings.control.as_ref()?.signal().ok()
    }

    fn frames_fd_channel(&self) -> &UnixStream {
        &self.fd_channel
    }

    pub fn raw_frames_stream(&mut self) -> &mut UnixStream {
        &mut self.frames
    }

    fn send(&mut self, msg_type: u16, correlation: u64, payload: &[u8]) -> Result<u64, AgentError> {
        let class = if msg_type & 0x8000 != 0 {
            FrameClass::Droppable
        } else {
            FrameClass::Required
        };
        let seq = self.seq;
        let bytes = encode_frame(PROTO_VERSION, msg_type, seq, correlation, class, payload)?;
        self.seq += 1;
        self.frames.write_all(&bytes)?;
        Ok(seq)
    }

    fn wait_correlated(&mut self, sent_seq: u64) -> Result<Frame, AgentError> {
        self.wait_correlated_streaming(sent_seq, &mut |_| true)
    }

    fn wait_correlated_emitting(
        &mut self,
        sent_seq: u64,
        on_emit: &mut dyn FnMut(linkw::TickEmitMsg),
    ) -> Result<Frame, AgentError> {
        let mut buf = [0u8; 64 * 1024];
        loop {
            while let Some(frame) = self.decoder.decode_next()? {
                if self.hello.chosen_version != 0
                    && frame.proto_version != self.hello.chosen_version
                {
                    return Err(AgentError::Protocol(
                        "frame version differs from the negotiated version",
                    ));
                }
                if frame.correlation_id == sent_seq {
                    return Ok(frame);
                }
                if frame.msg_type == msg_type::TICK_EMIT {
                    match postcard::from_bytes(&frame.payload) {
                        Ok(em) => on_emit(em),
                        Err(e) => {
                            tracing::warn!(error = %e, "dropping malformed TICK_EMIT frame")
                        }
                    }
                    continue;
                }
                self.note_async(frame)?;
            }
            match self.frames.read(&mut buf) {
                Ok(0) => return Err(AgentError::Closed),
                Ok(n) => self.decoder.feed(&buf[..n]),
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn wait_correlated_streaming(
        &mut self,
        sent_seq: u64,
        on_segment: &mut dyn FnMut(linkw::TranscribeSegmentMsg) -> bool,
    ) -> Result<Frame, AgentError> {
        let mut buf = [0u8; 64 * 1024];
        let mut cancelled = false;
        loop {
            while let Some(frame) = self.decoder.decode_next()? {
                if self.hello.chosen_version != 0
                    && frame.proto_version != self.hello.chosen_version
                {
                    return Err(AgentError::Protocol(
                        "frame version differs from the negotiated version",
                    ));
                }
                if frame.correlation_id == sent_seq {
                    return Ok(frame);
                }
                if frame.msg_type == msg_type::TRANSCRIBE_SEGMENT {
                    if !on_segment(postcard::from_bytes(&frame.payload)?) && !cancelled {
                        cancelled = true;
                        let payload = postcard::to_stdvec(&linkw::TranscribeCancelMsg {})?;
                        self.send(msg_type::TRANSCRIBE_CANCEL, 0, &payload)?;
                    }
                    continue;
                }
                self.note_async(frame)?;
            }
            match self.frames.read(&mut buf) {
                Ok(0) => return Err(AgentError::Closed),
                Ok(n) => self.decoder.feed(&buf[..n]),
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn note_async(&mut self, frame: Frame) -> Result<(), AgentError> {
        if frame.msg_type == msg_type::STATE_OP_DONE {
            let done: linkw::StateOpDoneMsg = postcard::from_bytes(&frame.payload)?;
            self.op_done.push_back(done);
        }
        Ok(())
    }

    fn request<Req: serde::Serialize, Res: serde::de::DeserializeOwned>(
        &mut self,
        req_type: u16,
        req: &Req,
        res_type: u16,
    ) -> Result<Res, AgentError> {
        let payload = postcard::to_stdvec(req)?;
        let sent = self.send(req_type, 0, &payload)?;
        let frame = self.wait_correlated(sent)?;
        if frame.msg_type != res_type {
            return Err(AgentError::Protocol("unexpected response type"));
        }
        Ok(postcard::from_bytes(&frame.payload)?)
    }

    pub fn stage_prompt(&mut self, tokens: &[u32]) -> Result<linkw::TokenRefMsg, AgentError> {
        let r = self.rings.stage_prompt(tokens)?;
        Ok(linkw::TokenRefMsg {
            ring_id: r.ring_id,
            index: r.index,
            count: r.count,
            generation: r.generation,
        })
    }

    pub fn read_tokens(&self, r: &linkw::TokenRefMsg) -> Result<Vec<u32>, AgentError> {
        use superfluid_engine::Rings;
        self.rings
            .read_tokens(&superfluid_abi::TokenRef {
                ring_id: r.ring_id,
                index: r.index,
                count: r.count,
                _pad0: 0,
                generation: r.generation,
            })
            .map_err(|_| AgentError::Protocol("stale emitted-token reference"))
    }

    pub fn read_logits(&self, r: &linkw::RingRefMsg) -> Result<Vec<f32>, AgentError> {
        use superfluid_engine::Rings;
        self.rings
            .read_logits_row(&superfluid_abi::RingRef {
                ring_id: r.ring_id,
                index: r.index,
                generation: r.generation,
            })
            .map_err(|_| AgentError::Protocol("stale logits reference"))
    }

    pub fn tick(&mut self, plan: linkw::TickPlanMsg) -> Result<linkw::TickEventsMsg, AgentError> {
        let res: linkw::TickResultMsg = self.request(
            msg_type::TICK_SUBMIT,
            &linkw::TickSubmitMsg { plan },
            msg_type::TICK_RESULT,
        )?;
        match res {
            linkw::TickResultMsg::Events(ev) => Ok(ev),
            linkw::TickResultMsg::Rejected { status } => Err(AgentError::Rejected(status)),
        }
    }

    pub fn tick_streaming(
        &mut self,
        plan: linkw::TickPlanMsg,
        on_emit: &mut dyn FnMut(linkw::TickEmitMsg),
    ) -> Result<linkw::TickEventsMsg, AgentError> {
        let payload = postcard::to_stdvec(&linkw::TickSubmitMsg { plan })?;
        let sent = self.send(msg_type::TICK_SUBMIT, 0, &payload)?;
        let frame = self.wait_correlated_emitting(sent, on_emit)?;
        if frame.msg_type != msg_type::TICK_RESULT {
            return Err(AgentError::Protocol("unexpected response type"));
        }
        match postcard::from_bytes::<linkw::TickResultMsg>(&frame.payload)? {
            linkw::TickResultMsg::Events(ev) => Ok(ev),
            linkw::TickResultMsg::Rejected { status } => Err(AgentError::Rejected(status)),
        }
    }

    pub fn match_prefix(
        &mut self,
        space_id: u32,
        span: linkw::TokenRefMsg,
    ) -> Result<linkw::MatchResMsg, AgentError> {
        self.request(
            msg_type::MATCH_REQ,
            &linkw::MatchReqMsg {
                space_id,
                span,
                media_deps: Vec::new(),
            },
            msg_type::MATCH_RES,
        )
    }

    pub fn seed_release(&mut self, seed_handle: u64) -> Result<(), AgentError> {
        let payload = postcard::to_stdvec(&linkw::SeedReleaseMsg { seed_handle })?;
        self.send(msg_type::SEED_RELEASE, 0, &payload)?;
        Ok(())
    }

    pub fn seed_acquire(
        &mut self,
        span: linkw::TokenRefMsg,
        prefix_len: u64,
        determinism_class: u8,
    ) -> Result<u64, AgentError> {
        self.seed_acquire_leased(span, prefix_len, determinism_class)
            .map(|(handle, _)| handle)
    }

    pub fn seed_acquire_leased(
        &mut self,
        span: linkw::TokenRefMsg,
        prefix_len: u64,
        determinism_class: u8,
    ) -> Result<(u64, Option<u64>), AgentError> {
        let payload = postcard::to_stdvec(&linkw::SeedAcquireMsg {
            span,
            prefix_len,
            determinism_class,
        })?;
        let sent = self.send(msg_type::SEED_ACQUIRE, 0, &payload)?;
        let frame = self.wait_correlated(sent)?;
        if frame.msg_type != msg_type::SEED_GRANT {
            return Err(AgentError::Protocol("unexpected response type"));
        }
        match linkw::SeedGrantMsg::decode(&frame.payload)? {
            linkw::SeedGrantMsg::Granted {
                seed_handle,
                ttl_ticks,
                ..
            } => Ok((seed_handle, ttl_ticks)),
            linkw::SeedGrantMsg::Refused { status } => Err(AgentError::Rejected(status)),
        }
    }

    pub fn buf_create(&mut self, len: u64) -> Result<u64, AgentError> {
        let buf_id = self.next_buf_id;
        self.next_buf_id += 1;
        let seg = ShmSegment::create(len as usize)?;
        let alloc = postcard::to_stdvec(&linkw::BufAllocMsg { buf_id, len })?;
        self.send(msg_type::BUF_ALLOC, 0, &alloc)?;
        let attach = postcard::to_stdvec(&linkw::BufAttachMsg {
            buf_id,
            len,
            generation: 1,
        })?;
        self.send(msg_type::BUF_ATTACH, 0, &attach)?;
        fdpass::send_fd(&self.fd_channel, seg.raw_fd(), buf_id)?;
        self.bufs.insert(buf_id, seg);
        Ok(buf_id)
    }

    pub fn buf_read(&self, buf_id: u64, len: usize) -> Result<Vec<u8>, AgentError> {
        let seg = self
            .bufs
            .get(&buf_id)
            .ok_or(AgentError::Protocol("unknown buffer"))?;
        let mut out = vec![0u8; len.min(seg.len())];
        seg.read_at(0, &mut out)?;
        Ok(out)
    }

    pub fn buf_write(&self, buf_id: u64, data: &[u8]) -> Result<(), AgentError> {
        let seg = self
            .bufs
            .get(&buf_id)
            .ok_or(AgentError::Protocol("unknown buffer"))?;
        seg.write_at(0, data)?;
        Ok(())
    }

    pub fn buf_release(&mut self, buf_id: u64) -> Result<(), AgentError> {
        let payload = postcard::to_stdvec(&linkw::BufReleaseMsg { buf_id })?;
        self.send(msg_type::BUF_RELEASE, 0, &payload)?;
        self.bufs.remove(&buf_id);
        Ok(())
    }

    pub fn state_op(&mut self, op: linkw::StateOpKind) -> Result<u64, AgentError> {
        let res: linkw::StateOpAcceptedMsg = self.request(
            msg_type::STATE_OP_SUBMIT,
            &linkw::StateOpSubmitMsg { op },
            msg_type::STATE_OP_ACCEPTED,
        )?;
        match res {
            linkw::StateOpAcceptedMsg::Accepted { op_handle } => Ok(op_handle),
            linkw::StateOpAcceptedMsg::Rejected { status } => Err(AgentError::Rejected(status)),
        }
    }

    pub fn wait_op_done(&mut self, op: u64) -> Result<linkw::StateOpDoneMsg, AgentError> {
        loop {
            if let Some(pos) = self.op_done.iter().position(|d| d.op_handle == op) {
                return Ok(self.op_done.remove(pos).expect("indexed"));
            }
            let mut buf = [0u8; 64 * 1024];
            while let Some(frame) = self.decoder.decode_next()? {
                self.note_async(frame)?;
            }
            match self.frames.read(&mut buf) {
                Ok(0) => return Err(AgentError::Closed),
                Ok(n) => self.decoder.feed(&buf[..n]),
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(e.into()),
            }
        }
    }

    pub fn state_sync(
        &mut self,
        req: linkw::StateSyncReqMsg,
    ) -> Result<linkw::StateSyncOkMsg, AgentError> {
        let res: linkw::StateSyncResMsg =
            self.request(msg_type::STATE_SYNC_REQ, &req, msg_type::STATE_SYNC_RES)?;
        match res {
            linkw::StateSyncResMsg::Ok(ok) => Ok(ok),
            linkw::StateSyncResMsg::Err { status } => Err(AgentError::Rejected(status)),
        }
    }

    pub fn send_transcribe_cancel(&mut self) -> Result<(), AgentError> {
        let payload = postcard::to_stdvec(&linkw::TranscribeCancelMsg {})?;
        self.send(msg_type::TRANSCRIBE_CANCEL, 0, &payload)?;
        Ok(())
    }

    pub fn state_sync_streaming(
        &mut self,
        req: linkw::StateSyncReqMsg,
        on_segment: &mut dyn FnMut(linkw::TranscribeSegmentMsg) -> bool,
    ) -> Result<linkw::StateSyncOkMsg, AgentError> {
        let payload = postcard::to_stdvec(&req)?;
        let sent = self.send(msg_type::STATE_SYNC_REQ, 0, &payload)?;
        let frame = self.wait_correlated_streaming(sent, on_segment)?;
        if frame.msg_type != msg_type::STATE_SYNC_RES {
            return Err(AgentError::Protocol("unexpected response type"));
        }
        match postcard::from_bytes::<linkw::StateSyncResMsg>(&frame.payload)? {
            linkw::StateSyncResMsg::Ok(ok) => Ok(ok),
            linkw::StateSyncResMsg::Err { status } => Err(AgentError::Rejected(status)),
        }
    }

    pub fn register_strategy(
        &mut self,
        reg: linkw::StrategyRegisterMsg,
    ) -> Result<linkw::StrategyGrantOkMsg, AgentError> {
        let res: linkw::StrategyGrantMsg =
            self.request(msg_type::STRATEGY_REGISTER, &reg, msg_type::STRATEGY_GRANT)?;
        match res {
            linkw::StrategyGrantMsg::Granted(ok) => Ok(ok),
            linkw::StrategyGrantMsg::Refused { status } => Err(AgentError::Rejected(status)),
        }
    }

    pub fn ping(&mut self) -> Result<linkw::PongMsg, AgentError> {
        let ping = linkw::PingMsg {
            queue_depth: 0,
            mem: linkw::MemCountersMsg {
                allocated_bytes: 0,
                host_retained_bytes: 0,
                pool_blocks_total: 0,
                pool_blocks_used: 0,
                pool_bytes_evictable: 0,
                tentative_bytes: 0,
            },
        };
        self.request(msg_type::PING, &ping, msg_type::PONG)
    }

    pub fn drain(&mut self) -> Result<(), AgentError> {
        let _: linkw::DrainedMsg =
            self.request(msg_type::DRAIN_REQ, &linkw::DrainReqMsg, msg_type::DRAINED)?;
        Ok(())
    }
}
