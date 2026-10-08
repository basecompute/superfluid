//! Idempotency helpers for Link F: pure logic, no I/O..

use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct CommandTable<R> {
    applied: BTreeMap<u64, R>,
}

impl<R> Default for CommandTable<R> {
    fn default() -> Self {
        Self {
            applied: BTreeMap::new(),
        }
    }
}

impl<R: Clone> CommandTable<R> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self, command_id: u64, compute: impl FnOnce() -> R) -> (R, bool) {
        if let Some(existing) = self.applied.get(&command_id) {
            return (existing.clone(), false);
        }
        let result = compute();
        self.applied.insert(command_id, result.clone());
        (result, true)
    }

    pub fn get(&self, command_id: u64) -> Option<&R> {
        self.applied.get(&command_id)
    }

    pub fn trim(&mut self, watermark: u64) {
        match watermark.checked_add(1) {
            Some(cut) => self.applied = self.applied.split_off(&cut),
            None => self.applied.clear(),
        }
    }

    pub fn len(&self) -> usize {
        self.applied.len()
    }

    pub fn is_empty(&self) -> bool {
        self.applied.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GenerationKey {
    pub session_id: u64,
    pub epoch: u64,
    pub generation_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupOutcome {
    Accepted,
    AlreadyCommitted,
    StaleEpoch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("gap in event_seq for {key:?}: expected {expected}, got {got}")]
pub struct Gap {
    pub key: GenerationKey,
    pub expected: u64,
    pub got: u64,
}

#[derive(Debug, Clone, Default)]
pub struct EventDedup {
    next_seq: BTreeMap<GenerationKey, u64>,
    session_epoch: BTreeMap<u64, u64>,
}

impl EventDedup {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_session_epoch(&mut self, session_id: u64, epoch: u64) {
        let prev = self.session_epoch.insert(session_id, epoch);
        if prev.is_none_or(|p| p < epoch) {
            self.next_seq
                .retain(|k, _| k.session_id != session_id || k.epoch >= epoch);
        }
    }

    pub fn retire_generation(&mut self, key: &GenerationKey) {
        self.next_seq.remove(key);
    }

    pub fn retire_session(&mut self, session_id: u64) {
        self.next_seq.retain(|k, _| k.session_id != session_id);
        self.session_epoch.remove(&session_id);
    }

    pub fn accept(&mut self, key: GenerationKey, event_seq: u64) -> Result<DedupOutcome, Gap> {
        if let Some(&current) = self.session_epoch.get(&key.session_id) {
            if key.epoch < current {
                return Ok(DedupOutcome::StaleEpoch);
            }
        }
        let next = *self.next_seq.get(&key).unwrap_or(&0);
        match event_seq.cmp(&next) {
            std::cmp::Ordering::Less => Ok(DedupOutcome::AlreadyCommitted),
            std::cmp::Ordering::Equal => {
                self.next_seq.insert(key, next + 1);
                Ok(DedupOutcome::Accepted)
            }
            std::cmp::Ordering::Greater => Err(Gap {
                key,
                expected: next,
                got: event_seq,
            }),
        }
    }

    pub fn watermark(&self, key: &GenerationKey) -> Option<u64> {
        self.next_seq.get(key).and_then(|&n| n.checked_sub(1))
    }

    pub fn tracked_generations(&self) -> usize {
        self.next_seq.len()
    }
}

#[derive(Debug, Clone)]
pub struct RetryBuffer<E> {
    events: VecDeque<(u64, E)>,
}

impl<E> Default for RetryBuffer<E> {
    fn default() -> Self {
        Self {
            events: VecDeque::new(),
        }
    }
}

impl<E: Clone> RetryBuffer<E> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, event_seq: u64, event: E) {
        self.events.push_back((event_seq, event));
    }

    pub fn ack(&mut self, watermark: u64) {
        while let Some(&(seq, _)) = self.events.front() {
            if seq <= watermark {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn retransmit_from(&self, from: u64) -> Vec<E> {
        self.events
            .iter()
            .filter(|&&(seq, _)| seq >= from)
            .map(|(_, e)| e.clone())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseClock {
    received_at: Duration,
    duration: Duration,
    renew_by: Duration,
}

impl LeaseClock {
    pub fn new(received_at: Duration, duration_ms: u64, renew_by_ms: u64) -> Self {
        Self {
            received_at,
            duration: Duration::from_millis(duration_ms),
            renew_by: Duration::from_millis(renew_by_ms),
        }
    }

    pub fn must_renew(&self, now: Duration) -> bool {
        now.saturating_sub(self.received_at) >= self.renew_by
    }

    pub fn expired(&self, now: Duration) -> bool {
        now.saturating_sub(self.received_at) >= self.duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(session_id: u64, epoch: u64, generation_id: u64) -> GenerationKey {
        GenerationKey {
            session_id,
            epoch,
            generation_id,
        }
    }

    #[test]
    fn command_table_replay_returns_original_result() {
        let mut t: CommandTable<u64> = CommandTable::new();
        let mut calls = 0;
        let (r1, new1) = t.apply(1, || {
            calls += 1;
            42
        });
        assert_eq!(r1, 42);
        assert!(new1);
        assert_eq!(calls, 1);

        let (r2, new2) = t.apply(1, || {
            calls += 1;
            999
        });
        assert_eq!(r2, 42);
        assert!(!new2);
        assert_eq!(calls, 1);
    }

    #[test]
    fn command_table_trim_drops_at_and_below_watermark() {
        let mut t: CommandTable<()> = CommandTable::new();
        for id in 1..=5 {
            t.apply(id, || ());
        }
        assert_eq!(t.len(), 5);
        t.trim(3);
        assert_eq!(t.len(), 2);
        assert!(t.get(3).is_none());
        assert!(t.get(4).is_some());
        assert!(t.get(5).is_some());
    }

    #[test]
    fn event_dedup_fences_stale_epochs() {
        let mut d = EventDedup::new();
        d.set_session_epoch(1, 5);
        assert_eq!(d.accept(key(1, 4, 1), 0), Ok(DedupOutcome::StaleEpoch));
        assert_eq!(d.watermark(&key(1, 4, 1)), None);
        assert_eq!(d.accept(key(1, 5, 1), 0), Ok(DedupOutcome::Accepted));

        assert_eq!(d.tracked_generations(), 1);
        d.set_session_epoch(1, 6);
        assert_eq!(d.tracked_generations(), 0);
        assert_eq!(d.accept(key(1, 5, 1), 1), Ok(DedupOutcome::StaleEpoch));
    }

    #[test]
    fn event_dedup_gc_is_bounded() {
        let mut d = EventDedup::new();
        for g in 0..10 {
            d.accept(key(1, 1, g), 0).unwrap();
        }
        assert_eq!(d.tracked_generations(), 10);
        for g in 0..10 {
            d.retire_generation(&key(1, 1, g));
        }
        assert_eq!(d.tracked_generations(), 0);

        d.accept(key(2, 1, 1), 0).unwrap();
        d.accept(key(3, 1, 1), 0).unwrap();
        d.retire_session(2);
        assert_eq!(d.tracked_generations(), 1);
    }

    #[test]
    fn event_dedup_contiguous_accept() {
        let mut d = EventDedup::new();
        let k = key(1, 1, 1);
        assert_eq!(d.accept(k, 0), Ok(DedupOutcome::Accepted));
        assert_eq!(d.accept(k, 1), Ok(DedupOutcome::Accepted));
        assert_eq!(d.accept(k, 2), Ok(DedupOutcome::Accepted));
        assert_eq!(d.watermark(&k), Some(2));
    }

    #[test]
    fn event_dedup_retransmit_after_lost_ack_is_deduped() {
        let mut d = EventDedup::new();
        let k = key(1, 1, 1);
        d.accept(k, 0).unwrap();
        d.accept(k, 1).unwrap();
        assert_eq!(d.accept(k, 0), Ok(DedupOutcome::AlreadyCommitted));
        assert_eq!(d.accept(k, 1), Ok(DedupOutcome::AlreadyCommitted));
        assert_eq!(d.watermark(&k), Some(1));
    }

    #[test]
    fn event_dedup_gap_rejected() {
        let mut d = EventDedup::new();
        let k = key(1, 1, 1);
        d.accept(k, 0).unwrap();
        let err = d.accept(k, 2).unwrap_err();
        assert_eq!(
            err,
            Gap {
                key: k,
                expected: 1,
                got: 2
            }
        );
        assert_eq!(d.watermark(&k), Some(0));
    }

    #[test]
    fn event_dedup_generations_are_independent() {
        let mut d = EventDedup::new();
        let g1 = key(1, 1, 1);
        let g2 = key(1, 1, 2);
        d.accept(g1, 0).unwrap();
        d.accept(g1, 1).unwrap();
        d.accept(g2, 0).unwrap();
        assert_eq!(d.watermark(&g1), Some(1));
        assert_eq!(d.watermark(&g2), Some(0));
    }

    #[test]
    fn retry_buffer_ack_trims_contiguous_prefix() {
        let mut b: RetryBuffer<&'static str> = RetryBuffer::new();
        b.push(0, "a");
        b.push(1, "b");
        b.push(2, "c");
        b.ack(1);
        assert_eq!(b.len(), 1);
        assert_eq!(b.retransmit_from(0), vec!["c"]);
    }

    #[test]
    fn retry_buffer_retransmit_from_yields_tail() {
        let mut b: RetryBuffer<u32> = RetryBuffer::new();
        for i in 0..5 {
            b.push(i, i as u32 * 10);
        }
        assert_eq!(b.retransmit_from(3), vec![30, 40]);
        assert_eq!(b.retransmit_from(0), vec![0, 10, 20, 30, 40]);
        assert_eq!(b.retransmit_from(100), Vec::<u32>::new());
    }

    #[test]
    fn retry_buffer_ack_all_empties_buffer() {
        let mut b: RetryBuffer<u32> = RetryBuffer::new();
        b.push(0, 1);
        b.push(1, 2);
        b.ack(1);
        assert!(b.is_empty());
    }

    #[test]
    fn lease_clock_renew_and_expiry_boundaries() {
        let received = Duration::from_secs(100);
        let clock = LeaseClock::new(received, 10_000, 8_000);

        assert!(!clock.must_renew(received + Duration::from_millis(7_999)));
        assert!(clock.must_renew(received + Duration::from_millis(8_000)));
        assert!(clock.must_renew(received + Duration::from_millis(8_001)));

        assert!(!clock.expired(received + Duration::from_millis(9_999)));
        assert!(clock.expired(received + Duration::from_millis(10_000)));
        assert!(clock.expired(received + Duration::from_millis(20_000)));
    }

    #[test]
    fn lease_clock_now_before_receipt_is_never_expired() {
        let received = Duration::from_secs(100);
        let clock = LeaseClock::new(received, 10_000, 8_000);
        let earlier = Duration::from_secs(50);
        assert!(!clock.must_renew(earlier));
        assert!(!clock.expired(earlier));
    }
}
