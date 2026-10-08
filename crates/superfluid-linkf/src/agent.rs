//! The node-agent endpoint.

use std::collections::BTreeMap;
use std::time::Duration;

use superfluid_proto::idempotency::{CommandTable, LeaseClock, RetryBuffer};
use superfluid_proto::linkf::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentAction {
    SelfFence { session_id: u64 },
}

#[derive(Debug)]
struct SessionCtx {
    epoch: u64,
    clock: LeaseClock,
    commands: CommandTable<()>,
    fenced: bool,
}

#[derive(Debug)]
struct GenerationCtx {
    epoch: u64,
    next_seq: u64,
    retry: RetryBuffer<TokenEvent>,
}

#[derive(Debug, Default)]
pub struct AgentSession {
    sessions: BTreeMap<u64, SessionCtx>,
    generations: BTreeMap<(u64, u64), GenerationCtx>,
}

impl AgentSession {
    pub fn new() -> AgentSession {
        AgentSession::default()
    }

    pub fn on_revoke(&mut self, msg: &SessionRevokeMsg) -> bool {
        match self.sessions.get(&msg.session_id) {
            Some(s) if msg.epoch < s.epoch => return false,
            None => return false,
            _ => {}
        }
        self.sessions.remove(&msg.session_id);
        self.generations.retain(|(sid, _), _| *sid != msg.session_id);
        true
    }

    pub fn on_assign(&mut self, msg: &SessionAssignMsg, now: Duration) -> bool {
        if let Some(s) = self.sessions.get_mut(&msg.session_id) {
            if msg.epoch < s.epoch {
                return false;
            }
            let (_, fresh) = s.commands.apply(msg.command_id, || ());
            if !fresh {
                return false;
            }
            if msg.epoch > s.epoch {
                self.generations
                    .retain(|(sid, _), g| *sid != msg.session_id || g.epoch >= msg.epoch);
            }
            s.epoch = msg.epoch;
            s.fenced = false;
            s.clock = LeaseClock::new(now, msg.lease.duration_ms, msg.lease.renew_by_ms);
            true
        } else {
            let mut commands = CommandTable::new();
            commands.apply(msg.command_id, || ());
            self.sessions.insert(
                msg.session_id,
                SessionCtx {
                    epoch: msg.epoch,
                    clock: LeaseClock::new(now, msg.lease.duration_ms, msg.lease.renew_by_ms),
                    commands,
                    fenced: false,
                },
            );
            true
        }
    }

    pub fn on_generate(&mut self, msg: &GenerateReqMsg) -> bool {
        let Some(s) = self.sessions.get_mut(&msg.session_id) else {
            return false;
        };
        if msg.epoch != s.epoch || s.fenced {
            return false;
        }
        let (_, fresh) = s.commands.apply(msg.command_id, || ());
        if !fresh {
            return false;
        }
        self.generations
            .entry((msg.session_id, msg.generation_id))
            .or_insert(GenerationCtx {
                epoch: msg.epoch,
                next_seq: 0,
                retry: RetryBuffer::new(),
            });
        true
    }

    pub fn on_append(&mut self, session_id: u64, epoch: u64, command_id: u64) -> bool {
        let Some(s) = self.sessions.get_mut(&session_id) else {
            return false;
        };
        if epoch != s.epoch || s.fenced {
            return false;
        }
        let (_, fresh) = s.commands.apply(command_id, || ());
        fresh
    }

    pub fn emit(
        &mut self,
        session_id: u64,
        generation_id: u64,
        payload: TokenEventPayload,
    ) -> Option<TokenEventsMsg> {
        let s = self.sessions.get(&session_id)?;
        if s.fenced {
            return None;
        }
        let g = self.generations.get_mut(&(session_id, generation_id))?;
        if g.epoch != s.epoch {
            return None;
        }
        let event = TokenEvent {
            session_id,
            epoch: g.epoch,
            generation_id,
            event_seq: g.next_seq,
            payload,
        };
        g.next_seq += 1;
        g.retry.push(event.event_seq, event.clone());
        Some(TokenEventsMsg {
            events: vec![event],
        })
    }

    pub fn on_watermark_ack(&mut self, msg: &WatermarkAckMsg) {
        if let Some(g) = self
            .generations
            .get_mut(&(msg.session_id, msg.generation_id))
        {
            if g.epoch == msg.epoch {
                g.retry.ack(msg.watermark);
            }
        }
    }

    pub fn needs_renew(&self, now: Duration) -> Vec<LeaseRenewMsg> {
        self.sessions
            .iter()
            .filter(|(_, s)| !s.fenced && s.clock.must_renew(now) && !s.clock.expired(now))
            .map(|(id, s)| LeaseRenewMsg {
                session_id: *id,
                epoch: s.epoch,
            })
            .collect()
    }

    pub fn on_lease_grant(&mut self, msg: &LeaseGrantMsg, now: Duration) {
        if let Some(s) = self.sessions.get_mut(&msg.session_id) {
            if s.epoch == msg.epoch && !s.fenced {
                s.clock = LeaseClock::new(now, msg.duration_ms, msg.renew_by_ms);
            }
        }
    }

    pub fn poll(&mut self, now: Duration) -> Vec<AgentAction> {
        let mut actions = Vec::new();
        for (id, s) in self.sessions.iter_mut() {
            if !s.fenced && s.clock.expired(now) {
                s.fenced = true;
                actions.push(AgentAction::SelfFence { session_id: *id });
            }
        }
        actions
    }

    pub fn on_reconcile(
        &mut self,
        msg: &ReconcileMsg,
        now: Duration,
    ) -> (ReconcileAckMsg, Vec<TokenEventsMsg>) {
        let mut acks = Vec::new();
        let mut retransmissions = Vec::new();
        for head_state in &msg.sessions {
            let Some(s) = self.sessions.get_mut(&head_state.session_id) else {
                acks.push(SessionReconcileAck {
                    session_id: head_state.session_id,
                    held: false,
                    retry_ranges: Vec::new(),
                    lease_view: head_state.lease_state,
                });
                continue;
            };
            if head_state.epoch > s.epoch {
                s.fenced = true;
            }
            s.commands
                .trim(head_state.command_watermark.saturating_sub(1));
            let held = head_state.epoch == s.epoch && !s.fenced;
            let mut retry_ranges = Vec::new();
            if held {
                for gw in &head_state.generation_watermarks {
                    if let Some(g) = self
                        .generations
                        .get(&(head_state.session_id, gw.generation_id))
                    {
                        let from = gw.watermark.map(|w| w + 1).unwrap_or(0);
                        let tail = g.retry.retransmit_from(from);
                        if let (Some(first), Some(last)) = (tail.first(), tail.last()) {
                            retry_ranges.push(RetryRange {
                                generation_id: gw.generation_id,
                                from_seq: first.event_seq,
                                to_seq: last.event_seq,
                            });
                            retransmissions.push(TokenEventsMsg { events: tail });
                        }
                    }
                }
            }
            acks.push(SessionReconcileAck {
                session_id: head_state.session_id,
                held,
                retry_ranges,
                lease_view: LeaseStateMsg {
                    lease: Lease {
                        duration_ms: 0,
                        renew_by_ms: 0,
                    },
                    elapsed_ms: now.as_millis() as u64,
                },
            });
        }
        (ReconcileAckMsg { sessions: acks }, retransmissions)
    }

    pub fn is_fenced(&self, session_id: u64) -> Option<bool> {
        self.sessions.get(&session_id).map(|s| s.fenced)
    }

    pub fn unacked_events(&self, session_id: u64, generation_id: u64) -> usize {
        self.generations
            .get(&(session_id, generation_id))
            .map(|g| g.retry.len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assign(session: u64, epoch: u64, command: u64) -> SessionAssignMsg {
        SessionAssignMsg {
            command_id: command,
            session_id: session,
            epoch,
            lease: Lease {
                duration_ms: 1000,
                renew_by_ms: 600,
            },
            codec: "chatml".into(),
            params: Vec::new(),
            qos_class: 0,
            policy_ref: 0,
        }
    }

    fn t(ms: u64) -> Duration {
        Duration::from_millis(ms)
    }

    #[test]
    fn assign_is_idempotent_and_epoch_is_absolute() {
        let mut a = AgentSession::new();
        let msg = assign(1, 5, 10);
        assert!(a.on_assign(&msg, t(0)));
        assert!(!a.on_assign(&msg, t(1)));
        let mut old = assign(1, 4, 11);
        old.command_id = 11;
        assert!(!a.on_assign(&old, t(2)));
    }

    #[test]
    fn generate_replay_does_not_restart() {
        let mut a = AgentSession::new();
        a.on_assign(&assign(1, 5, 10), t(0));
        let g = GenerateReqMsg {
            command_id: 11,
            session_id: 1,
            epoch: 5,
            generation_id: 100,
            budgets: GenerationBudgetsMsg {
                max_new_tokens: 4,
                max_wall_ms: 100,
            },
            policy_ref: 0,
        };
        assert!(a.on_generate(&g));
        a.emit(1, 100, TokenEventPayload::Tokens(vec![1])).unwrap();
        assert!(!a.on_generate(&g));
        let batch = a.emit(1, 100, TokenEventPayload::Tokens(vec![2])).unwrap();
        assert_eq!(batch.events[0].event_seq, 1, "stream restarted on replay");
    }

    #[test]
    fn append_is_gated_on_epoch_fence_and_command_id() {
        let mut a = AgentSession::new();
        a.on_assign(&assign(1, 5, 10), t(0));
        assert!(a.on_append(1, 5, 20));
        assert!(!a.on_append(1, 5, 20));
        assert!(!a.on_append(1, 4, 21));
        assert!(!a.on_append(2, 5, 22));
        a.on_generate(&GenerateReqMsg {
            command_id: 30,
            session_id: 1,
            epoch: 5,
            generation_id: 100,
            budgets: GenerationBudgetsMsg { max_new_tokens: 4, max_wall_ms: 100 },
            policy_ref: 0,
        });
        a.poll(t(5000));
        assert!(!a.on_append(1, 5, 31), "fenced session must refuse appends");
    }

    #[test]
    fn lease_expiry_self_fences_but_retains_state() {
        let mut a = AgentSession::new();
        a.on_assign(&assign(1, 5, 10), t(0));
        a.on_generate(&GenerateReqMsg {
            command_id: 11,
            session_id: 1,
            epoch: 5,
            generation_id: 100,
            budgets: GenerationBudgetsMsg {
                max_new_tokens: 4,
                max_wall_ms: 100,
            },
            policy_ref: 0,
        });
        a.emit(1, 100, TokenEventPayload::Tokens(vec![1])).unwrap();

        assert_eq!(a.needs_renew(t(700)).len(), 1);
        let actions = a.poll(t(1500));
        assert_eq!(actions, vec![AgentAction::SelfFence { session_id: 1 }]);
        assert!(a.emit(1, 100, TokenEventPayload::Tokens(vec![2])).is_none());
        assert_eq!(a.unacked_events(1, 100), 1);
    }
}
