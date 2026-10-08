use std::collections::BTreeMap;

use superfluid_proto::linkf::TokenEvent;

#[derive(Debug, Clone)]
pub struct CommittedEvent {
    pub committed_id: u64,
    pub event: TokenEvent,
}

#[derive(Debug, Default)]
pub struct WalStub {
    sessions: BTreeMap<u64, Vec<CommittedEvent>>,
}

impl WalStub {
    pub fn new() -> WalStub {
        WalStub::default()
    }

    pub fn commit(&mut self, event: TokenEvent) -> u64 {
        let log = self.sessions.entry(event.session_id).or_default();
        let committed_id = log.len() as u64;
        log.push(CommittedEvent {
            committed_id,
            event,
        });
        committed_id
    }

    pub fn session_events(&self, session_id: u64) -> &[CommittedEvent] {
        self.sessions
            .get(&session_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn total_committed(&self) -> usize {
        self.sessions.values().map(|v| v.len()).sum()
    }

    pub fn assert_single_contiguous_streams(&self) {
        let mut per_gen: BTreeMap<(u64, u64, u64), Vec<u64>> = BTreeMap::new();
        for log in self.sessions.values() {
            for c in log {
                per_gen
                    .entry((c.event.session_id, c.event.epoch, c.event.generation_id))
                    .or_default()
                    .push(c.event.event_seq);
            }
        }
        for ((s, e, g), mut seqs) in per_gen {
            seqs.sort_unstable();
            for (i, seq) in seqs.iter().enumerate() {
                assert_eq!(
                    *seq, i as u64,
                    "session {s} epoch {e} generation {g}: committed stream has a gap or duplicate"
                );
            }
        }
    }
}
