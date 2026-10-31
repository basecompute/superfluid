//! Which node last ran each prompt prefix, so a conversation's next turn is placed where its
//! prefix is warm: in the node's cache, or on its disk with `--park`. Kept on disk by the head,
//! so it holds across a head restart; nodes are named by identity, which a restart keeps.

use std::collections::HashMap;
use std::path::PathBuf;

/// Tokens per block a prefix is matched in.
const BLOCK: usize = 64;
/// A prefix shorter than this is not worth sending a request out of its way for.
const MIN_BLOCKS: usize = 2;
/// Prefixes remembered at most; the oldest eighth goes when it is full.
const MAX: usize = 1 << 16;

#[derive(Default)]
pub struct Warm {
    /// A hash of a model and a prompt's first blocks → the node that ran it, and when.
    by_prefix: HashMap<u64, (String, u64)>,
    clock: u64,
    path: Option<PathBuf>,
    dirty: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Stored {
    prefixes: Vec<(u64, String, u64)>,
}

/// The hash of `model` and each whole block of `tokens` so far, one per block.
fn chain<'a>(model: &str, tokens: &'a [u32]) -> impl Iterator<Item = u64> + 'a {
    let seed = model.bytes().fold(0xCBF2_9CE4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3));
    tokens.as_chunks::<BLOCK>().0.iter().scan(seed, |h, block| {
        for t in block {
            for b in t.to_le_bytes() {
                *h = (*h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        Some(*h)
    })
}

impl Warm {
    /// Remembered in `path`, read back from it when it is there.
    pub fn kept_in(path: PathBuf) -> Warm {
        let stored: Option<Stored> = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok());
        let mut warm = Warm { path: Some(path), ..Warm::default() };
        for (hash, node, at) in stored.map(|s| s.prefixes).unwrap_or_default() {
            warm.clock = warm.clock.max(at);
            warm.by_prefix.insert(hash, (node, at));
        }
        warm
    }

    /// The node that ran the longest prefix of `tokens` the head remembers, and how many tokens
    /// that prefix is.
    pub fn warmest(&self, model: &str, tokens: &[u32]) -> Option<(&str, usize)> {
        let mut best = None;
        for (n, hash) in chain(model, tokens).enumerate() {
            match self.by_prefix.get(&hash) {
                Some((node, _)) => best = Some((node.as_str(), (n + 1) * BLOCK)),
                None => break,
            }
        }
        best.filter(|&(_, len)| len >= MIN_BLOCKS * BLOCK)
    }

    pub fn ran(&mut self, model: &str, tokens: &[u32], node: &str) {
        self.clock += 1;
        let at = self.clock;
        for hash in chain(model, tokens) {
            self.by_prefix.insert(hash, (node.to_string(), at));
        }
        if self.by_prefix.len() > MAX {
            let mut ages: Vec<u64> = self.by_prefix.values().map(|&(_, at)| at).collect();
            ages.sort_unstable();
            let cut = ages[MAX / 8];
            self.by_prefix.retain(|_, &mut (_, at)| at > cut);
        }
        self.dirty = true;
    }

    /// Writes what changed since the last save.
    pub fn save(&mut self) {
        let Some(path) = &self.path else { return };
        if !self.dirty {
            return;
        }
        let stored = Stored { prefixes: self.by_prefix.iter().map(|(&h, (node, at))| (h, node.clone(), *at)).collect() };
        let Ok(bytes) = serde_json::to_vec(&stored) else { return };
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path)).is_ok() {
            self.dirty = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_longest_prefix_a_node_ran_wins_and_is_kept_across_a_restart() {
        let path = std::env::temp_dir().join(format!("superfluid-warm-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let turn1: Vec<u32> = (0..300).collect();
        let mut turn2 = turn1.clone();
        turn2.extend(1000..1200);
        let mut warm = Warm::kept_in(path.clone());
        warm.ran("m", &turn1, "a");
        warm.ran("m", &turn2[..400], "b");
        assert_eq!(warm.warmest("m", &turn2), Some(("b", 384)), "b ran more of it");
        assert_eq!(warm.warmest("m", &turn1), Some(("b", 256)), "b ran the same first blocks last");
        assert_eq!(warm.warmest("other", &turn2), None, "another model's prefix is its own");
        assert_eq!(warm.warmest("m", &(5000..5300).collect::<Vec<u32>>()), None);
        assert_eq!(warm.warmest("m", &turn1[..100]), None, "one block is too short to matter");
        warm.save();
        let again = Warm::kept_in(path.clone());
        assert_eq!(again.warmest("m", &turn2), Some(("b", 384)));
        let _ = std::fs::remove_file(&path);
    }
}
