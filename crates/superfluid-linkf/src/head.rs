//! The head-daemon endpoint.

use std::collections::{BTreeMap, BTreeSet};

use superfluid_proto::idempotency::{DedupOutcome, EventDedup, GenerationKey};
use superfluid_proto::linkf::*;

#[derive(Debug, Clone)]
struct SessionState {
    epoch: u64,
    lease: Lease,
    command_watermark: u64,
}

#[derive(Debug, Default)]
pub struct HeadSession {
    sessions: BTreeMap<u64, SessionState>,
    generations: BTreeSet<(u64, u64, u64)>,
    issued_generates: BTreeMap<(u64, u64), GenerateReqMsg>,
    dedup: EventDedup,
    wal: crate::WalStub,
    next_command_id: u64,
}

impl HeadSession {
    pub fn new() -> HeadSession {
        HeadSession {
            next_command_id: 1,
            ..Default::default()
        }
    }

    /// A session whose command ids start at `first_command`, so a head that reconnects can keep
    /// them past the ones its earlier connection used: a node remembers a session's commands
    /// across connections.
    pub fn starting_at(first_command: u64) -> HeadSession {
        HeadSession {
            next_command_id: first_command.max(1),
            ..Default::default()
        }
    }

    fn alloc_command(&mut self, session_id: u64) -> u64 {
        let id = self.next_command_id;
        self.next_command_id += 1;
        if let Some(s) = self.sessions.get_mut(&session_id) {
            s.command_watermark = id;
        }
        id
    }

    pub fn next_command(&mut self, session_id: u64) -> u64 {
        self.alloc_command(session_id)
    }

    pub fn assign_session(
        &mut self,
        session_id: u64,
        epoch: u64,
        lease: Lease,
        qos_class: u8,
        codec: &str,
    ) -> SessionAssignMsg {
        self.sessions.insert(
            session_id,
            SessionState { epoch, lease, command_watermark: 0 },
        );
        self.dedup.set_session_epoch(session_id, epoch);
        self.issued_generates
            .retain(|(sid, _), msg| *sid != session_id || msg.epoch >= epoch);
        let command_id = self.alloc_command(session_id);
        SessionAssignMsg {
            command_id,
            session_id,
            epoch,
            lease,
            codec: codec.to_string(),
            params: Vec::new(),
            qos_class,
            policy_ref: 0,
        }
    }

    pub fn revoke(&mut self, session_id: u64) -> Option<SessionRevokeMsg> {
        let epoch = self.sessions.get(&session_id)?.epoch;
        let command_id = self.alloc_command(session_id);
        self.sessions.remove(&session_id);
        self.generations.retain(|(sid, _, _)| *sid != session_id);
        self.issued_generates.retain(|(sid, _), _| *sid != session_id);
        self.dedup.set_session_epoch(session_id, epoch);
        Some(SessionRevokeMsg {
            command_id,
            session_id,
            epoch,
        })
    }

    pub fn generate(
        &mut self,
        session_id: u64,
        generation_id: u64,
        budgets: GenerationBudgetsMsg,
    ) -> Option<GenerateReqMsg> {
        let epoch = self.sessions.get(&session_id)?.epoch;
        if let Some(msg) = self.issued_generates.get(&(session_id, generation_id)) {
            return Some(*msg);
        }
        let command_id = self.alloc_command(session_id);
        let msg = GenerateReqMsg {
            command_id,
            session_id,
            epoch,
            generation_id,
            budgets,
            policy_ref: 0,
        };
        self.issued_generates
            .insert((session_id, generation_id), msg);
        self.generations.insert((session_id, epoch, generation_id));
        Some(msg)
    }

    pub fn on_token_events(&mut self, msg: &TokenEventsMsg) -> Vec<WatermarkAckMsg> {
        let mut touched: BTreeSet<(u64, u64, u64)> = BTreeSet::new();
        for ev in &msg.events {
            let key = GenerationKey {
                session_id: ev.session_id,
                epoch: ev.epoch,
                generation_id: ev.generation_id,
            };
            match self.dedup.accept(key, ev.event_seq) {
                Ok(DedupOutcome::Accepted) => {
                    self.wal.commit(ev.clone());
                    self.generations
                        .insert((ev.session_id, ev.epoch, ev.generation_id));
                    touched.insert((ev.session_id, ev.epoch, ev.generation_id));
                }
                Ok(DedupOutcome::AlreadyCommitted) => {
                    touched.insert((ev.session_id, ev.epoch, ev.generation_id));
                }
                Ok(DedupOutcome::StaleEpoch) => {  }
                Err(_gap) => {  }
            }
        }
        touched
            .into_iter()
            .filter_map(|(session_id, epoch, generation_id)| {
                let key = GenerationKey {
                    session_id,
                    epoch,
                    generation_id,
                };
                self.dedup.watermark(&key).map(|watermark| WatermarkAckMsg {
                    session_id,
                    epoch,
                    generation_id,
                    watermark,
                })
            })
            .collect()
    }

    pub fn on_lease_renew(&mut self, msg: &LeaseRenewMsg) -> Option<LeaseGrantMsg> {
        let s = self.sessions.get(&msg.session_id)?;
        if s.epoch != msg.epoch {
            return None;
        }
        Some(LeaseGrantMsg {
            session_id: msg.session_id,
            epoch: s.epoch,
            duration_ms: s.lease.duration_ms,
            renew_by_ms: s.lease.renew_by_ms,
        })
    }

    pub fn reconcile(&self) -> ReconcileMsg {
        let sessions = self
            .sessions
            .iter()
            .map(|(id, s)| {
                let generation_watermarks = self
                    .generations
                    .iter()
                    .filter(|(sid, e, _)| sid == id && *e == s.epoch)
                    .map(|(sid, e, g)| GenerationWatermark {
                        generation_id: *g,
                        watermark: self.dedup.watermark(&GenerationKey {
                            session_id: *sid,
                            epoch: *e,
                            generation_id: *g,
                        }),
                    })
                    .collect();
                SessionReconcileState {
                    session_id: *id,
                    epoch: s.epoch,
                    lease_state: LeaseStateMsg {
                        lease: s.lease,
                        elapsed_ms: 0,
                    },
                    command_watermark: s.command_watermark,
                    generation_watermarks,
                }
            })
            .collect();
        ReconcileMsg { sessions }
    }

    pub fn on_reconcile_ack(&mut self, ack: &ReconcileAckMsg) -> Vec<u64> {
        ack.sessions
            .iter()
            .filter(|s| !s.held)
            .map(|s| s.session_id)
            .collect()
    }

    pub fn retire_generation(&mut self, session_id: u64, generation_id: u64) {
        if let Some(s) = self.sessions.get(&session_id) {
            self.dedup.retire_generation(&GenerationKey {
                session_id,
                epoch: s.epoch,
                generation_id,
            });
            self.generations
                .remove(&(session_id, s.epoch, generation_id));
            self.issued_generates.remove(&(session_id, generation_id));
        }
    }

    pub fn current_epoch(&self, session_id: u64) -> Option<u64> {
        self.sessions.get(&session_id).map(|s| s.epoch)
    }

    pub fn wal(&self) -> &crate::WalStub {
        &self.wal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use superfluid_proto::linkf::TokenEventPayload;

    fn lease() -> Lease {
        Lease {
            duration_ms: 1000,
            renew_by_ms: 600,
        }
    }

    fn ev(session: u64, epoch: u64, generation: u64, seq: u64) -> TokenEvent {
        TokenEvent {
            session_id: session,
            epoch,
            generation_id: generation,
            event_seq: seq,
            payload: TokenEventPayload::Tokens(vec![seq as u32]),
        }
    }

    #[test]
    fn generate_retry_reuses_ids() {
        let mut h = HeadSession::new();
        h.assign_session(1, 5, lease(), 0, "chatml");
        let budgets = GenerationBudgetsMsg {
            max_new_tokens: 64,
            max_wall_ms: 1000,
        };
        let g1 = h.generate(1, 100, budgets).unwrap();
        let g2 = h.generate(1, 100, budgets).unwrap();
        assert_eq!(g1.command_id, g2.command_id);
        assert_eq!(g1.generation_id, g2.generation_id);
    }

    #[test]
    fn retransmit_commits_zero_bytes_and_reacks() {
        let mut h = HeadSession::new();
        h.assign_session(1, 5, lease(), 0, "chatml");
        let batch = TokenEventsMsg {
            events: vec![ev(1, 5, 100, 0), ev(1, 5, 100, 1)],
        };
        let acks = h.on_token_events(&batch);
        assert_eq!(acks.len(), 1);
        assert_eq!(acks[0].watermark, 1);
        assert_eq!(h.wal().total_committed(), 2);

        let acks = h.on_token_events(&batch);
        assert_eq!(acks.len(), 1);
        assert_eq!(acks[0].watermark, 1);
        assert_eq!(h.wal().total_committed(), 2);
    }

    #[test]
    fn stale_epoch_events_never_commit() {
        let mut h = HeadSession::new();
        h.assign_session(1, 5, lease(), 0, "chatml");
        h.on_token_events(&TokenEventsMsg {
            events: vec![ev(1, 5, 100, 0)],
        });
        h.assign_session(1, 6, lease(), 0, "chatml");
        let acks = h.on_token_events(&TokenEventsMsg {
            events: vec![ev(1, 5, 100, 1)],
        });
        assert!(acks.is_empty());
        assert_eq!(h.wal().total_committed(), 1);
        h.wal().assert_single_contiguous_streams();
    }

    #[test]
    fn reconcile_reports_none_when_nothing_committed() {
        let mut h = HeadSession::new();
        h.assign_session(1, 5, lease(), 0, "chatml");
        let budgets = GenerationBudgetsMsg {
            max_new_tokens: 4,
            max_wall_ms: 100,
        };
        h.generate(1, 100, budgets).unwrap();
        let rec = h.reconcile();
        let s = &rec.sessions[0];
        assert_eq!(s.generation_watermarks.len(), 1);
        assert_eq!(s.generation_watermarks[0].watermark, None);
    }
}
