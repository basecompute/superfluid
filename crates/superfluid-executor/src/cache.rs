//! The executor's prefix / reuse cache: entries in the runtime's pool, and
//! entries held out of it as the runtime's lossless exports.

use crate::primitives::Seq;

pub const RESIDENT_PREFIX_CLASS: u64 = 1;
/// The resident entries that have started two lanes or more: a prefix
/// requests share. Evicted only when this class is named too.
pub const SHARED_PREFIX_CLASS: u64 = 2;
/// The entries held out of the pool, as exports in host memory. Named when
/// memory is short on the machine: they go, and for a while no entry that
/// leaves the pool is kept as one.
pub const HOST_EXPORT_CLASS: u64 = 4;

/// One part in eight of an entry: what a seed must take of it to count as a use, and what
/// two entries may differ by and still be one.
pub const ENTRY_SHARE: u64 = 8;

pub struct CacheEntry {
    /// The entry's sequence while it is in the pool, and its name either way
    /// (a runtime never hands a handle out twice).
    pub seq: Seq,
    pub tokens: Vec<u32>,
    pub pins: u32,
    pub last_used: u64,
    pub bytes: u64,
    /// Lanes started from this entry.
    pub seeded: u32,
    /// A copy of a live lane's state, published for the requests that share
    /// its prefix: a lane seeded from it copies, never takes it over.
    pub shared: bool,
    /// `Some`: the entry is out of the pool, held as the runtime's lossless
    /// export of its `tokens.len()` tokens of state; its sequence is freed.
    pub exported: Option<Vec<u8>>,
    /// How much of `tokens` was the prompt of the lane that left the entry
    /// (the rest it generated); 0 where none is known.
    pub prompt: u64,
}

impl CacheEntry {
    /// Whether `len` tokens are a real part of this entry.
    pub fn share(&self, len: u64) -> bool {
        len * ENTRY_SHARE >= self.tokens.len() as u64
    }

    /// Whether a seed of `len` tokens is served from this entry: any, from
    /// one in the pool (a copy); a real part of it, from an export.
    pub fn serves(&self, len: u64) -> bool {
        self.exported.is_none() || self.share(len)
    }

    /// An entry holds a prefix requests share when it is a lane's state
    /// published for them, or has started two lanes or more: it goes after
    /// every entry that does not.
    pub fn holds_shared_prefix(&self) -> bool {
        self.shared || self.seeded >= 2
    }

    /// Whether `other` serves what this entry does: its tokens begin with all of this
    /// entry's, or all but a share of them when it was used later.
    pub fn covered_by(&self, other: &CacheEntry) -> bool {
        let common = self.tokens.iter().zip(&other.tokens).take_while(|(a, b)| a == b).count();
        let (n, later) = (self.tokens.len(), (other.last_used, other.seq) > (self.last_used, self.seq));
        let serves = other.share(common as u64);
        if common == n {
            // Of two entries with the same tokens, the later one stays.
            serves && (other.tokens.len() > n || later)
        } else {
            serves && later && (n - common) as u64 * ENTRY_SHARE <= n as u64
        }
    }
}

#[derive(Default)]
pub struct PrefixCache {
    pub entries: Vec<CacheEntry>,
    /// An entry came in or was used since the cache last settled, so an
    /// export may have come to be covered.
    pub changed: bool,
}

impl PrefixCache {
    pub fn insert(&mut self, seq: Seq, tokens: Vec<u32>, prompt: u64, bytes: u64, now: u64) {
        let prompt = prompt.min(tokens.len() as u64);
        self.entries.push(CacheEntry { seq, tokens, pins: 0, last_used: now, bytes, seeded: 0, shared: false, exported: None, prompt });
        self.changed = true;
    }

    pub fn get(&self, seq: Seq) -> Option<&CacheEntry> {
        self.entries.iter().find(|e| e.seq == seq)
    }

    /// Take the entry out of the cache.
    pub fn remove(&mut self, seq: Seq) -> Option<CacheEntry> {
        let at = self.entries.iter().position(|e| e.seq == seq)?;
        Some(self.entries.remove(at))
    }

    /// The sequences of the entries in the pool.
    pub fn resident(&self) -> Vec<Seq> {
        self.entries.iter().filter(|e| e.exported.is_none()).map(|e| e.seq).collect()
    }

    /// Tokens of state held out of the pool.
    pub fn exported_tokens(&self) -> u64 {
        self.entries.iter().filter(|e| e.exported.is_some()).map(|e| e.tokens.len() as u64).sum()
    }

    /// Host bytes the exports hold.
    pub fn exported_bytes(&self) -> u64 {
        self.entries.iter().filter_map(|e| e.exported.as_ref()).map(|p| p.len() as u64).sum()
    }

    /// The entries in the pool, then those out of it: of two that serve the
    /// same seed, the one in the pool copies it, the other imports.
    fn by_tier(&self) -> impl Iterator<Item = (usize, &CacheEntry)> {
        let in_pool = self.entries.iter().enumerate().filter(|(_, e)| e.exported.is_none());
        in_pool.chain(self.entries.iter().enumerate().filter(|(_, e)| e.exported.is_some()))
    }

    pub fn best_prefix(&self, span: &[u32], page: u64, exact_only: bool) -> Option<(usize, u64)> {
        let mut best: Option<(usize, u64)> = None;
        for (i, e) in self.by_tier() {
            let common = e.tokens.iter().zip(span.iter()).take_while(|(a, b)| a == b).count() as u64;
            let len = if exact_only {
                if common == e.tokens.len() as u64 { common } else { 0 }
            } else {
                common / page * page
            };
            if len >= page && len < span.len() as u64 && e.serves(len) && best.is_none_or(|(_, b)| len > b) {
                best = Some((i, len));
            }
        }
        best
    }

    pub fn candidates(&self, span: &[u32], page: u64, exact_only: bool) -> Vec<(u64, Seq)> {
        let mut out: Vec<(u64, Seq)> = Vec::new();
        for (_, e) in self.by_tier() {
            let common = e.tokens.iter().zip(span.iter()).take_while(|(a, b)| a == b).count() as u64;
            let len = if exact_only {
                if common == e.tokens.len() as u64 && common < span.len() as u64 { common } else { 0 }
            } else {
                common / page * page
            };
            if len >= page && len.is_multiple_of(page) && e.serves(len) && !out.iter().any(|(l, _)| *l == len) {
                out.push((len, e.seq));
            }
        }
        out.sort_by_key(|(l, _)| std::cmp::Reverse(*l));
        out
    }

    /// Pool bytes an eviction could free: the entries in the pool no lease
    /// pins.
    pub fn unpinned_bytes(&self) -> u64 {
        self.entries.iter().filter(|e| e.pins == 0 && e.exported.is_none()).map(|e| e.bytes).sum()
    }

    /// Pool bytes the cache accounts for.
    pub fn resident_bytes(&self) -> u64 {
        self.entries.iter().filter(|e| e.exported.is_none()).map(|e| e.bytes).sum()
    }

    pub fn plan_evict(&self, target: u64, cap: u64, shared_too: bool) -> (u64, Vec<usize>) {
        let (picked, _) = self.plan_evict_where(cap, shared_too, |p| self.bytes_of(p) >= target);
        (self.bytes_of(&picked), picked)
    }

    pub fn plan_evict_until(&self, cap: u64, enough: impl FnMut(&[usize]) -> bool) -> (Vec<usize>, bool) {
        self.plan_evict_where(cap, true, enough)
    }

    /// The entries in the pool one eviction takes out of it.
    fn plan_evict_where(&self, cap: u64, shared_too: bool, mut enough: impl FnMut(&[usize]) -> bool) -> (Vec<usize>, bool) {
        let mut order: Vec<usize> = (0..self.entries.len())
            .filter(|&i| {
                let e = &self.entries[i];
                e.pins == 0 && e.exported.is_none() && (shared_too || !e.holds_shared_prefix())
            })
            .collect();
        // An entry that has started two lanes or more holds a prefix that
        // requests share (a system prompt, a document, a burst's common
        // part): it goes after every entry that has not, oldest first in each.
        order.sort_by_key(|&i| (self.entries[i].holds_shared_prefix(), self.entries[i].last_used));
        let mut used = 0u64;
        let mut picked = Vec::new();
        if enough(&picked) {
            return (picked, true);
        }
        for i in order {
            let bytes = self.entries[i].bytes;
            if used + bytes > cap {
                continue;
            }
            used += bytes;
            picked.push(i);
            if enough(&picked) {
                return (picked, true);
            }
        }
        (picked, false)
    }

    pub fn bytes_of(&self, indices: &[usize]) -> u64 {
        indices.iter().map(|&i| self.entries[i].bytes).sum()
    }

    /// Takes the unpinned entries whose tokens carry one of `digests` out of
    /// the cache, and names them with whether each was in the pool.
    pub fn evict_digests(&mut self, digests: &[[u8; 32]]) -> Vec<(Seq, bool)> {
        let mut gone = Vec::new();
        self.entries.retain(|e| {
            let d = superfluid_fingerprint::content_digest(&e.tokens);
            if e.pins == 0 && digests.contains(&d) {
                gone.push((e.seq, e.exported.is_none()));
                false
            } else {
                true
            }
        });
        gone
    }

    /// Removes the entries keyed by exactly `tokens` that no seed holds.
    pub fn evict_exact(&mut self, tokens: &[u32]) -> Vec<(Seq, bool)> {
        let mut gone = Vec::new();
        self.entries.retain(|e| {
            if e.pins == 0 && e.tokens[..] == *tokens {
                gone.push((e.seq, e.exported.is_none()));
                false
            } else {
                true
            }
        });
        gone
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheEntry, PrefixCache};

    fn evict(c: &mut PrefixCache, target: u64, shared_too: bool) -> (u64, Vec<u64>) {
        let (freed, picked) = c.plan_evict(target, u64::MAX, shared_too);
        let seqs: Vec<u64> = picked.iter().map(|&i| c.entries[i].seq).collect();
        for &s in &seqs {
            c.remove(s);
        }
        (freed, seqs)
    }

    #[test]
    fn an_entry_that_seeded_several_lanes_outlives_newer_ones_that_seeded_none() {
        let mut c = PrefixCache::default();
        c.insert(1, vec![1, 2, 3], 0, 10, 1);
        c.entries[0].seeded = 7;
        c.insert(2, vec![4, 5, 6], 0, 10, 5);
        c.insert(3, vec![7, 8, 9], 0, 10, 6);
        let (_, gone) = evict(&mut c, 20, true);
        assert_eq!(gone, vec![2, 3], "the two that seeded nothing go first");
        c.insert(4, vec![1], 0, 10, 9);
        c.entries.last_mut().unwrap().seeded = 1;
        let (freed, gone) = evict(&mut c, 100, false);
        assert_eq!((freed, gone), (10, vec![4]), "one lane is not a shared prefix; the shared one is spared");
        let (_, gone) = evict(&mut c, 100, true);
        assert_eq!(gone, vec![1], "unless shared prefixes may go too");
    }

    #[test]
    fn an_entry_out_of_the_pool_is_offered_after_one_in_it_and_only_for_a_real_part_of_it() {
        let mut c = PrefixCache::default();
        c.insert(1, (10..26).collect(), 0, 16, 1);
        c.insert(2, (10..18).collect(), 0, 8, 2);
        c.entries[0].exported = Some(vec![0; 64]);
        let span: Vec<u32> = (10..30).collect();
        assert_eq!(c.candidates(&span, 1, false), vec![(16, 1), (8, 2)]);
        let branch: Vec<u32> = (10..18).chain([99]).collect();
        assert_eq!(c.candidates(&branch, 1, false), vec![(8, 2)], "the eight both hold are offered from the pool");
        let short: Vec<u32> = vec![10, 900];
        assert_eq!(c.candidates(&short, 1, false), vec![(1, 2)], "one token is no real part of the sixteen exported");
        assert_eq!((c.resident(), c.exported_tokens(), c.exported_bytes(), c.resident_bytes()), (vec![2], 16, 64, 8));
        let (_, gone) = evict(&mut c, 100, true);
        assert_eq!(gone, vec![2], "an eviction takes from the pool only");
    }

    #[test]
    fn a_later_turn_covers_an_earlier_one_and_a_different_conversation_does_not() {
        let entry = |seq: u64, tokens: Vec<u32>, used: u64| CacheEntry {
            seq,
            tokens,
            pins: 0,
            last_used: used,
            bytes: 0,
            seeded: 0,
            shared: false,
            exported: None,
            prompt: 0,
        };
        let turn1 = entry(1, (10..26).collect(), 1);
        let turn2 = entry(2, (10..26).chain([40, 41]).collect(), 2);
        assert!(turn1.covered_by(&turn2) && !turn2.covered_by(&turn1));
        let reworded = entry(3, (10..25).chain([70, 71, 72]).collect(), 3);
        assert!(turn1.covered_by(&reworded), "all but its last token, used later");
        assert!(!reworded.covered_by(&turn1), "used earlier, and its own last three tokens");
        let other = entry(4, (10..18).chain(200..208).collect(), 4);
        assert!(!turn1.covered_by(&other), "half of it is another conversation's");
        let long = entry(5, (10..300).collect(), 5);
        assert!(!turn1.covered_by(&long), "sixteen tokens are no real part of 290: a seed of them is not served from its export");
    }
}
