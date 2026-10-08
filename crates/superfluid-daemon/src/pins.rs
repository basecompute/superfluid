//! Pinned prefixes.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;

use superfluid_proto::linkw::ALL_SPACES;

use crate::runtime::EngineHost;
use crate::scheduler::SchedStats;

pub(crate) const LEGACY_RENEW_TICKS: u64 = 3;

pub(crate) fn renew_every(ttl_ticks: Option<u64>) -> Option<u64> {
    match ttl_ticks {
        None => Some(LEGACY_RENEW_TICKS),
        Some(0) => None,
        Some(ttl) => Some((ttl / 2).max(1)),
    }
}

pub(crate) const MAX_PINS: usize = 256;

const ACQUIRE_TRIES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PinKey {
    Session(u64),
    Prefix(u64),
}

pub struct PinReq {
    pub key: PinKey,
    pub tokens: Vec<u32>,
    pub want: u64,
    pub deadline_unix_ms: u64,
    pub anchor: Option<u64>,
    pub reply: Option<Sender<u64>>,
}

const LOOKAHEAD: u32 = u32::MAX;

fn stageable<'a>(host: &EngineHost, tokens: &'a [u32]) -> &'a [u32] {
    match host.max_stream_tokens() as usize {
        0 => tokens,
        max => &tokens[..tokens.len().min(max)],
    }
}

pub fn session_target(stream: &[u32]) -> (Vec<u32>, u64) {
    let mut tokens = Vec::with_capacity(stream.len() + 1);
    tokens.extend_from_slice(stream);
    tokens.push(LOOKAHEAD);
    (tokens, stream.len() as u64)
}

pub(crate) fn common_seed_lengths(
    host: &EngineHost,
    m: &superfluid_proto::linkw::MatchResMsg,
    cap: u64,
) -> Vec<u64> {
    enum Serves {
        UpTo(u64, u64),
        Exactly(HashSet<u64>),
    }
    let mut spaces = Vec::new();
    let exact = host.exact_seeds_only();
    for s in &m.spaces {
        let (kind, page) = host.space_shape(s.space_id).unwrap_or((0, 0));
        if kind == superfluid_abi::space_kind::ENCODER_CACHE {
            continue;
        }
        let lens = s.candidates.iter().map(|c| c.prefix_len);
        spaces.push(if page > 0 && !exact {
            let page = page as u64;
            let longest = lens.map(|l| l / page * page).max().unwrap_or(0);
            Serves::UpTo(page, longest)
        } else {
            Serves::Exactly(lens.collect())
        });
    }
    if spaces.is_empty() {
        return Vec::new();
    }
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let mut unit = 0u64;
    let mut paged_max: Option<u64> = None;
    for sp in &spaces {
        if let Serves::UpTo(page, longest) = sp {
            unit = if unit == 0 {
                *page
            } else {
                unit / gcd(unit, *page) * page
            };
            paged_max = Some(paged_max.map_or(*longest, |m| m.min(*longest)));
        }
    }
    let paged_max = paged_max.map(|m| m.min(cap) / unit * unit);
    let mut lens: Vec<u64> = paged_max
        .into_iter()
        .chain(spaces.iter().flat_map(|sp| match sp {
            Serves::UpTo(..) => Vec::new(),
            Serves::Exactly(set) => set.iter().copied().collect(),
        }))
        .filter(|&l| l > 0 && l <= cap)
        .filter(|&l| {
            spaces.iter().all(|sp| match sp {
                Serves::UpTo(page, longest) => l % page == 0 && l <= *longest,
                Serves::Exactly(set) => set.contains(&l),
            })
        })
        .collect();
    lens.sort_unstable_by(|a, b| b.cmp(a));
    lens.dedup();
    lens
}

pub fn prefix_digest(tokens: &[u32]) -> u64 {
    prefix_digest_from(0xCBF2_9CE4_8422_2325, tokens)
}

fn prefix_digest_from(seed: u64, tokens: &[u32]) -> u64 {
    let mut h = seed;
    for t in tokens {
        h = superfluid_fingerprint::fnv1a64(&t.to_le_bytes(), h);
    }
    h
}

struct Lease {
    handle: u64,
    len: u64,
    blocks: Vec<(u64, u64)>,
    renewed_tick: u64,
    renew_every: u64,
}

struct Pin {
    tokens: Vec<u32>,
    want: u64,
    deadline_unix_ms: u64,
    anchors: HashSet<u64>,
    touched: u64,
    lease: Option<Lease>,
}

pub struct Pins {
    map: HashMap<PinKey, Pin>,
    clock: u64,
    budget_pct: u64,
    fallback_pool_tokens: u64,
    host_generation: u64,
    lease_too_short: bool,
    bytes: u64,
    dirty: bool,
}

fn lease_blocks(host: &EngineHost, tokens: &[u32]) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = host
        .paged_spaces()
        .into_iter()
        .flat_map(|(space_id, page, bytes)| {
            let seed = superfluid_fingerprint::fnv1a64(&space_id.to_le_bytes(), 0);
            block_chain_from(seed, tokens, page)
                .into_iter()
                .map(move |h| (h, bytes))
        })
        .collect();
    for (space_id, bytes) in host.blob_spaces() {
        let seed = superfluid_fingerprint::fnv1a64(&space_id.to_le_bytes(), 1);
        out.push((prefix_digest_from(seed, tokens), bytes));
    }
    out
}

#[cfg(test)]
fn block_chain(tokens: &[u32], page: u64) -> Vec<u64> {
    block_chain_from(0xCBF2_9CE4_8422_2325, tokens, page)
}

fn block_chain_from(seed: u64, tokens: &[u32], page: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(tokens.len() / page.max(1) as usize);
    let mut h = seed;
    for block in tokens.chunks_exact(page.max(1) as usize) {
        for t in block {
            h = superfluid_fingerprint::fnv1a64(&t.to_le_bytes(), h);
        }
        out.push(h);
    }
    out
}

impl Pins {
    pub fn new(budget_pct: u8, fallback_pool_tokens: u64) -> Pins {
        Pins {
            map: HashMap::new(),
            clock: 0,
            budget_pct: budget_pct as u64,
            fallback_pool_tokens,
            host_generation: 0,
            lease_too_short: false,
            bytes: 0,
            dirty: false,
        }
    }

    pub fn set_fallback_pool_tokens(&mut self, tokens: u64) {
        self.fallback_pool_tokens = tokens;
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn until_next_deadline(&self) -> Option<std::time::Duration> {
        let next = self.map.values().map(|p| p.deadline_unix_ms).min()?;
        let now = crate::wal::now_unix_ms();
        Some(std::time::Duration::from_millis(next.saturating_sub(now)))
    }

    pub fn any_held(&self) -> bool {
        self.map.values().any(|p| p.lease.is_some())
    }

    pub fn enforceable(&self, host: &EngineHost) -> bool {
        self.budget_pct > 0
            && !self.lease_too_short
            && host.page_size().is_some()
            && host.is_alive()
            && !host.has_unservable_space()
            && host.cache_serves_all_spaces()
    }

    fn pinned_bytes(&mut self) -> u64 {
        if self.dirty {
            let mut seen: HashMap<u64, u64> = HashMap::new();
            for p in self.map.values() {
                if let Some(l) = &p.lease {
                    seen.extend(l.blocks.iter().copied());
                }
            }
            self.bytes = seen.values().sum();
            self.dirty = false;
        }
        self.bytes
    }

    pub fn is_anchor(&self, session: u64) -> bool {
        self.map.values().any(|p| p.anchors.contains(&session))
    }

    pub fn awaits_lease(&self, host: &EngineHost, session: u64) -> bool {
        self.enforceable(host) && self.is_anchor(session)
    }

    pub fn held_max(&self, host: &EngineHost, keys: &[PinKey]) -> u64 {
        if host.generation() != self.host_generation || !host.is_alive() {
            return 0;
        }
        let now = crate::wal::now_unix_ms();
        keys.iter()
            .filter(|k| self.map.get(k).is_some_and(|p| p.deadline_unix_ms > now))
            .map(|k| self.held(*k))
            .max()
            .unwrap_or(0)
    }

    fn held(&self, key: PinKey) -> u64 {
        self.map
            .get(&key)
            .and_then(|p| p.lease.as_ref())
            .map(|l| l.len)
            .unwrap_or(0)
    }

    fn publish_gauges(&mut self, stats: &SchedStats) {
        let held = self.map.values().filter(|p| p.lease.is_some()).count() as u64;
        let bytes = self.pinned_bytes();
        stats.pins_held.store(held, Ordering::Relaxed);
        stats.pinned_bytes.store(bytes, Ordering::Relaxed);
    }

    fn check_generation(&mut self, host: &EngineHost) {
        if host.generation() == self.host_generation && host.is_alive() {
            return;
        }
        if host.generation() != self.host_generation {
            self.lease_too_short = false;
        }
        self.host_generation = host.generation();
        for p in self.map.values_mut() {
            p.lease = None;
        }
        self.dirty = true;
    }

    fn release_lease(host: &mut EngineHost, lease: Lease) {
        if host.is_alive() {
            let _ = host.client().seed_release(lease.handle);
        }
    }

    pub fn handle(&mut self, host: &mut EngineHost, stats: &SchedStats, tick: u64, req: PinReq) {
        self.check_generation(host);
        self.sweep(host, stats);
        let held = if req.deadline_unix_ms == 0 {
            if let Some(p) = self.map.remove(&req.key) {
                if let Some(l) = p.lease {
                    Self::release_lease(host, l);
                    self.dirty = true;
                }
            }
            0
        } else {
            self.clock += 1;
            let clock = self.clock;
            let pin = self.map.entry(req.key).or_insert_with(|| Pin {
                tokens: Vec::new(),
                want: 0,
                deadline_unix_ms: 0,
                anchors: HashSet::new(),
                touched: 0,
                lease: None,
            });
            pin.deadline_unix_ms = match req.key {
                PinKey::Prefix(_) => pin.deadline_unix_ms.max(req.deadline_unix_ms),
                PinKey::Session(_) => req.deadline_unix_ms,
            };
            pin.touched = clock;
            if let Some(a) = req.anchor {
                pin.anchors.insert(a);
            }
            if !req.tokens.is_empty() && req.want >= pin.want {
                pin.tokens = req.tokens;
                pin.want = req.want;
            }
            self.cap_entries(host, stats, req.key);
            self.acquire(host, tick, req.key);
            self.enforce_budget(host, stats, Some(req.key));
            self.held(req.key)
        };
        self.publish_gauges(stats);
        if let Some(reply) = req.reply {
            let _ = reply.send(held);
        }
    }

    pub fn unanchor(&mut self, session: u64) {
        for (k, p) in self.map.iter_mut() {
            if matches!(k, PinKey::Prefix(_)) {
                p.anchors.remove(&session);
            }
        }
    }

    pub fn sweep(&mut self, host: &mut EngineHost, stats: &SchedStats) {
        if self.map.is_empty() {
            return;
        }
        if host.generation() != self.host_generation || !host.is_alive() {
            self.check_generation(host);
            self.publish_gauges(stats);
        }
        let now = crate::wal::now_unix_ms();
        let expired: Vec<PinKey> = self
            .map
            .iter()
            .filter(|(_, p)| p.deadline_unix_ms <= now)
            .map(|(k, _)| *k)
            .collect();
        if expired.is_empty() {
            return;
        }
        for key in expired {
            if let Some(p) = self.map.remove(&key) {
                if let Some(l) = p.lease {
                    Self::release_lease(host, l);
                    self.dirty = true;
                }
                stats.pins_expired.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.publish_gauges(stats);
    }

    pub fn after_tick(
        &mut self,
        host: &mut EngineHost,
        stats: &SchedStats,
        tick: u64,
        retired: &[(u64, Vec<u32>)],
    ) {
        if self.map.is_empty() {
            return;
        }
        self.sweep(host, stats);
        for (session, stream) in retired {
            let keys: Vec<PinKey> = self
                .map
                .iter()
                .filter(|(_, p)| p.anchors.contains(session))
                .map(|(k, _)| *k)
                .collect();
            for key in keys {
                if key == PinKey::Session(*session) {
                    if let Some(p) = self.map.get_mut(&key) {
                        (p.tokens, p.want) = session_target(stream);
                    }
                } else if let Some(p) = self.map.get_mut(&key) {
                    p.anchors.remove(session);
                }
                self.acquire(host, tick, key);
            }
        }
        self.renew(host, tick);
        self.enforce_budget(host, stats, None);
        self.publish_gauges(stats);
    }

    fn acquire(&mut self, host: &mut EngineHost, tick: u64, key: PinKey) {
        if !self.enforceable(host) {
            return;
        }
        let Some(page) = host.page_size() else { return };
        let Some(pin) = self.map.get(&key) else {
            return;
        };
        let tokens = stageable(host, &pin.tokens).to_vec();
        let n = tokens.len() as u64;
        let cap = pin.want.min(n.saturating_sub(1)) / page * page;
        let current = pin.lease.as_ref().map(|l| l.len).unwrap_or(0);
        if cap < page || current >= cap {
            return;
        }
        let Ok(span) = host.client().stage_prompt(&tokens) else {
            return;
        };
        let Ok(m) = host.client().match_prefix(ALL_SPACES, span) else {
            return;
        };
        let lens = common_seed_lengths(host, &m, cap);
        for len in lens
            .into_iter()
            .filter(|&l| l >= page && l > current)
            .take(ACQUIRE_TRIES)
        {
            let Ok((handle, ttl)) = host.client().seed_acquire_leased(span, len, 0) else {
                continue;
            };
            let Some(renew_every) = renew_every(ttl) else {
                self.refuse_short_leases(host, handle);
                return;
            };
            let lease = Lease {
                handle,
                len,
                blocks: lease_blocks(host, &tokens[..len as usize]),
                renewed_tick: tick,
                renew_every,
            };
            let old = self.map.get_mut(&key).and_then(|p| p.lease.replace(lease));
            if let Some(old) = old {
                Self::release_lease(host, old);
            }
            self.dirty = true;
            return;
        }
    }

    fn renew(&mut self, host: &mut EngineHost, tick: u64) {
        let due: Vec<PinKey> = self
            .map
            .iter()
            .filter(|(_, p)| {
                p.lease
                    .as_ref()
                    .is_some_and(|l| tick.saturating_sub(l.renewed_tick) >= l.renew_every)
            })
            .map(|(k, _)| *k)
            .collect();
        for key in due {
            let Some(pin) = self.map.get(&key) else {
                continue;
            };
            let Some(len) = pin.lease.as_ref().map(|l| l.len) else {
                continue;
            };
            let tokens = stageable(host, &pin.tokens).to_vec();
            let fresh = if host.is_alive() {
                host.client()
                    .stage_prompt(&tokens)
                    .ok()
                    .and_then(|span| host.client().seed_acquire_leased(span, len, 0).ok())
            } else {
                None
            };
            let Some(pin) = self.map.get_mut(&key) else {
                continue;
            };
            match fresh {
                Some((handle, ttl)) => {
                    let Some(every) = renew_every(ttl) else {
                        if let Some(l) = pin.lease.take() {
                            Self::release_lease(host, l);
                        }
                        self.refuse_short_leases(host, handle);
                        continue;
                    };
                    let lease = pin.lease.as_mut().expect("checked above");
                    let old = std::mem::replace(&mut lease.handle, handle);
                    lease.renewed_tick = tick;
                    lease.renew_every = every;
                    let _ = host.client().seed_release(old);
                }
                None => {
                    if let Some(l) = pin.lease.take() {
                        Self::release_lease(host, l);
                        self.dirty = true;
                    }
                }
            }
        }
    }

    fn refuse_short_leases(&mut self, host: &mut EngineHost, handle: u64) {
        let _ = host.client().seed_release(handle);
        for p in self.map.values_mut() {
            if let Some(l) = p.lease.take() {
                Self::release_lease(host, l);
            }
        }
        self.dirty = true;
        if !self.lease_too_short {
            tracing::warn!(
                "engine seed leases live 0 ticks: pins cannot be held across ticks and are left \
                 unenforced (recorded only) under this worker"
            );
        }
        self.lease_too_short = true;
    }

    fn yield_oldest(
        &mut self,
        host: &mut EngineHost,
        stats: &SchedStats,
        keep: Option<PinKey>,
    ) -> bool {
        let victim = self
            .map
            .iter()
            .filter(|(_, p)| p.lease.is_some())
            .min_by_key(|(k, p)| (Some(**k) == keep, p.touched))
            .map(|(k, _)| *k);
        let Some(key) = victim else { return false };
        if let Some(l) = self.map.get_mut(&key).and_then(|p| p.lease.take()) {
            Self::release_lease(host, l);
        }
        self.dirty = true;
        stats.pins_yielded.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn enforce_budget(&mut self, host: &mut EngineHost, stats: &SchedStats, keep: Option<PinKey>) {
        let Some(block_bytes) = host.kv_block_bytes() else {
            return;
        };
        let mut total = stats.pool_blocks_total.load(Ordering::Relaxed);
        if total == 0 {
            total = self.fallback_pool_tokens / host.page_size().unwrap_or(1).max(1);
        }
        if total == 0 {
            return;
        }
        let budget = total * block_bytes * self.budget_pct / 100;
        while self.pinned_bytes() > budget {
            if !self.yield_oldest(host, stats, keep) {
                break;
            }
        }
    }

    fn cap_entries(&mut self, host: &mut EngineHost, stats: &SchedStats, keep: PinKey) {
        let is_prefix = |k: &PinKey| matches!(k, PinKey::Prefix(_));
        while self.map.keys().filter(|k| is_prefix(k)).count() > MAX_PINS {
            let victim = self
                .map
                .iter()
                .filter(|(k, _)| is_prefix(k) && **k != keep)
                .min_by_key(|(_, p)| p.touched)
                .map(|(k, _)| *k);
            let Some(key) = victim else { return };
            if let Some(p) = self.map.remove(&key) {
                if let Some(l) = p.lease {
                    Self::release_lease(host, l);
                    self.dirty = true;
                    stats.pins_yielded.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    pub fn yield_for(&mut self, host: &mut EngineHost, stats: &SchedStats, bytes: u64) -> u64 {
        let before = self.pinned_bytes();
        while before - self.pinned_bytes() < bytes {
            if !self.yield_oldest(host, stats, None) {
                break;
            }
        }
        self.publish_gauges(stats);
        before - self.pinned_bytes()
    }

    pub fn yield_all(&mut self, host: &mut EngineHost, stats: &SchedStats) -> bool {
        if !self.any_held() {
            return false;
        }
        while self.yield_oldest(host, stats, None) {}
        self.publish_gauges(stats);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_head_only_runtime_is_seeded_only_at_an_entry_length() {
        use superfluid_executor::fake::FakeConfig;
        use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives};
        let host = |truncate_partial: bool| {
            EngineHost::spawn(move || {
                let cfg = FakeConfig { truncate_partial, page: 16, ..Default::default() };
                (Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default()), None)
            })
            .expect("spawn")
        };
        let cached_40 = superfluid_proto::linkw::MatchResMsg {
            spaces: vec![superfluid_proto::linkw::SpaceMatchMsg {
                space_id: 1,
                candidates: vec![superfluid_proto::linkw::MatchCandidateMsg {
                    prefix_len: 40,
                    provenance_digest: [0; 32],
                    taint_bits: 0,
                    resident_tier: 0,
                }],
            }],
        };
        let paged = host(true);
        assert!(!paged.exact_seeds_only());
        assert_eq!(common_seed_lengths(&paged, &cached_40, 64), [32], "any page boundary within the match");
        let head_only = host(false);
        assert!(head_only.exact_seeds_only());
        assert_eq!(common_seed_lengths(&head_only, &cached_40, 64), [40], "the entry's own length");
        assert_eq!(common_seed_lengths(&head_only, &cached_40, 32), Vec::<u64>::new(), "no cut below the head");
    }

    #[test]
    fn nested_prefixes_share_their_leading_blocks() {
        let long: Vec<u32> = (0..64).collect();
        let a = block_chain(&long[..32], 16);
        let b = block_chain(&long, 16);
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 4);
        assert_eq!(
            a[..],
            b[..2],
            "a prefix's blocks are the longer pin's leading blocks"
        );
        let other: Vec<u32> = (1..33).collect();
        assert_ne!(block_chain(&other, 16)[0], a[0]);
        assert_eq!(block_chain(&long[..40], 16).len(), 2);
    }

    #[test]
    fn renewal_cadence_follows_the_granted_lease() {
        assert_eq!(renew_every(Some(64)), Some(32));
        assert_eq!(renew_every(Some(8)), Some(4));
        assert_eq!(renew_every(Some(3)), Some(1));
        assert_eq!(renew_every(Some(1)), Some(1));
        assert_eq!(renew_every(Some(0)), None);
        assert_eq!(renew_every(None), Some(LEGACY_RENEW_TICKS));
        for ttl in 1..200u64 {
            assert!(renew_every(Some(ttl)).unwrap() < ttl + 1);
        }
    }

    #[test]
    fn prefix_digest_is_content_identity() {
        assert_eq!(prefix_digest(&[1, 2, 3]), prefix_digest(&[1, 2, 3]));
        assert_ne!(prefix_digest(&[1, 2, 3]), prefix_digest(&[1, 2]));
    }
}
