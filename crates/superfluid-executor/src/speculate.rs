//! Prompt-lookup speculation: drafts copied from the lane's own history,
//! verified in one pass of the target (`step_rows`).

use std::collections::HashMap;

use crate::primitives::VERIFY_ROWS_MAX;

pub const STRATEGY_ID: &str = "prompt-lookup";

/// The daemon's proposal configuration (`key=value,...`); unknown keys and
/// an empty string leave the defaults.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpecConfig {
    pub max_draft: usize,
    pub adaptive: bool,
    pub min_yield: f64,
    pub yield_rounds: u32,
    pub throughput_gate: bool,
    pub min_speedup: f64,
    pub gate_reprobe: u32,
    pub gate_reprobe_max: u32,
}

impl Default for SpecConfig {
    fn default() -> Self {
        SpecConfig {
            max_draft: 4,
            adaptive: true,
            min_yield: 0.75,
            yield_rounds: 24,
            throughput_gate: true,
            min_speedup: 1.08,
            gate_reprobe: 32,
            gate_reprobe_max: 1024,
        }
    }
}

impl SpecConfig {
    pub fn parse(params: &str) -> SpecConfig {
        let mut c = SpecConfig::default();
        for kv in params.split(',') {
            let Some((k, v)) = kv.split_once('=') else { continue };
            let v = v.trim();
            match k.trim() {
                "max_draft" => c.max_draft = v.parse().unwrap_or(c.max_draft),
                "adaptive" => c.adaptive = v != "0",
                "min_yield" => c.min_yield = v.parse().unwrap_or(c.min_yield),
                "yield_rounds" => c.yield_rounds = v.parse().unwrap_or(c.yield_rounds),
                "throughput_gate" => c.throughput_gate = v != "0",
                "min_speedup" => c.min_speedup = v.parse().unwrap_or(c.min_speedup),
                "gate_reprobe" => c.gate_reprobe = v.parse().unwrap_or(c.gate_reprobe),
                "gate_reprobe_max" => c.gate_reprobe_max = v.parse().unwrap_or(c.gate_reprobe_max),
                _ => {}
            }
        }
        c.max_draft = c.max_draft.clamp(1, VERIFY_ROWS_MAX as usize - 1);
        c.yield_rounds = c.yield_rounds.max(1);
        c.gate_reprobe = c.gate_reprobe.max(1);
        c.gate_reprobe_max = c.gate_reprobe_max.max(c.gate_reprobe);
        c
    }
}

/// Longest n-gram first: the most recent earlier occurrence of the
/// history's last `n` tokens, and up to `k` tokens that followed it.
pub(crate) fn propose(history: &[u32], k: usize) -> Vec<u32> {
    const NGRAM_MAX: usize = 3;
    const NGRAM_MIN: usize = 2;
    if k == 0 {
        return Vec::new();
    }
    for n in (NGRAM_MIN..=NGRAM_MAX).rev() {
        if history.len() <= n {
            continue;
        }
        let tail = &history[history.len() - n..];
        let found = (0..history.len() - n).rev().find(|&i| &history[i..i + n] == tail);
        if let Some(i) = found {
            let from = i + n;
            let to = (from + k).min(history.len());
            if from < to {
                return history[from..to].to_vec();
            }
        }
    }
    Vec::new()
}

/// A lane's speculation: the draft depth it is at, and the yield floor.
#[derive(Debug, Clone)]
pub(crate) struct LaneSpec {
    pub depth: usize,
    accepted_ema: Option<f64>,
    window_rounds: u32,
    window_accepted: u32,
    off_rounds: u32,
    off_next: u32,
}

impl LaneSpec {
    pub fn new(cfg: &SpecConfig) -> LaneSpec {
        LaneSpec {
            depth: cfg.max_draft,
            accepted_ema: None,
            window_rounds: 0,
            window_accepted: 0,
            off_rounds: 0,
            off_next: cfg.gate_reprobe,
        }
    }

    /// Whether this lane drafts this round; a lane that is resting counts
    /// the round down.
    pub fn drafting(&mut self) -> bool {
        if self.off_rounds > 0 {
            self.off_rounds -= 1;
            return false;
        }
        true
    }

    /// After a verified round of `proposed` drafts, `accepted` of them taken.
    pub fn record(&mut self, cfg: &SpecConfig, proposed: usize, accepted: usize) {
        if proposed == 0 {
            return;
        }
        let a = accepted as f64;
        self.accepted_ema = Some(match self.accepted_ema {
            Some(e) => 0.8 * e + 0.2 * a,
            None => a,
        });
        if cfg.adaptive {
            let want = self.accepted_ema.unwrap_or(a).ceil() as usize + 1;
            self.depth = want.clamp(1, cfg.max_draft);
        }
        self.window_rounds += 1;
        self.window_accepted += accepted as u32;
        if self.window_rounds >= cfg.yield_rounds {
            let yielded = self.window_accepted as f64 / self.window_rounds as f64;
            if yielded < cfg.min_yield {
                self.off_rounds = self.off_next;
                self.off_next = (self.off_next * 2).min(cfg.gate_reprobe_max);
            } else {
                self.off_next = cfg.gate_reprobe;
            }
            self.window_rounds = 0;
            self.window_accepted = 0;
        }
    }
}

/// Whether drafting pays at a lane count: decode tokens per second of ticks that draft
/// against ticks that do not.
#[derive(Debug, Default)]
pub(crate) struct ThroughputGate {
    buckets: HashMap<usize, Bucket>,
}

/// Ticks a drafting probe runs (the first at a lane count is not measured:
/// its passes compile kernels for the new shape) and a plain one runs.
const SPEC_PROBE_TICKS: u32 = 2;
const PLAIN_PROBE_TICKS: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    /// Measuring drafting (`spec`) or plain decoding.
    Probe { spec: bool, left: u32 },
    /// Running the faster mode until the other is measured again.
    Run { spec: bool, left: u32 },
}

#[derive(Debug)]
struct Bucket {
    plain: Option<f64>,
    spec: Option<f64>,
    phase: Phase,
    on_next: u32,
    off_next: u32,
    warm: bool,
}

impl Bucket {
    fn new(cfg: &SpecConfig) -> Bucket {
        Bucket {
            plain: None,
            spec: None,
            phase: Phase::Probe { spec: true, left: SPEC_PROBE_TICKS },
            on_next: cfg.gate_reprobe,
            off_next: cfg.gate_reprobe,
            warm: false,
        }
    }

    fn drafting(&self) -> bool {
        matches!(self.phase, Phase::Probe { spec: true, .. } | Phase::Run { spec: true, .. })
    }

    /// Run the faster mode, for longer each time the same mode wins again.
    fn decide(&mut self, cfg: &SpecConfig) -> Phase {
        let spec_wins = match (self.spec, self.plain) {
            (Some(s), Some(p)) => s >= p * cfg.min_speedup,
            _ => true,
        };
        let grow = |n: u32| (n * 2).min(cfg.gate_reprobe_max);
        if spec_wins {
            let left = self.on_next;
            self.on_next = grow(self.on_next);
            self.off_next = cfg.gate_reprobe;
            Phase::Run { spec: true, left }
        } else {
            let left = self.off_next;
            self.off_next = grow(self.off_next);
            self.on_next = cfg.gate_reprobe;
            Phase::Run { spec: false, left }
        }
    }
}

fn ema(slot: &mut Option<f64>, sample: f64) {
    *slot = Some(match *slot {
        Some(e) => 0.7 * e + 0.3 * sample,
        None => sample,
    });
}

impl ThroughputGate {
    /// Whether a tick with `lanes` decoding lanes drafts.
    pub fn allows(&mut self, cfg: &SpecConfig, lanes: usize) -> bool {
        !cfg.throughput_gate || self.buckets.entry(lanes).or_insert_with(|| Bucket::new(cfg)).drafting()
    }

    /// A tick's decode phase: `tokens` over `secs`, and whether it drafted.
    pub fn record(&mut self, cfg: &SpecConfig, lanes: usize, drafted: bool, secs: f64, tokens: usize) {
        if !cfg.throughput_gate || tokens == 0 || secs <= 0.0 {
            return;
        }
        let b = self.buckets.entry(lanes).or_insert_with(|| Bucket::new(cfg));
        let rate = tokens as f64 / secs;
        if drafted {
            if b.warm {
                ema(&mut b.spec, rate);
            }
            b.warm = true;
        } else {
            ema(&mut b.plain, rate);
        }
        // Ticks without drafts measure plain decoding on the same traffic: drafting that
        // falls behind by its own winning margin stops now.
        if let (Phase::Run { spec: true, .. }, Some(s), Some(p)) = (b.phase, b.spec, b.plain) {
            if drafted && s * cfg.min_speedup < p {
                b.phase = b.decide(cfg);
                return;
            }
        }
        b.phase = match b.phase {
            Phase::Probe { spec, left } if left > 1 => Phase::Probe { spec, left: left - 1 },
            Phase::Run { spec, left } if left > 1 => Phase::Run { spec, left: left - 1 },
            Phase::Probe { spec: true, .. } if b.plain.is_none() => Phase::Probe { spec: false, left: PLAIN_PROBE_TICKS },
            Phase::Probe { .. } => b.decide(cfg),
            Phase::Run { spec: true, .. } => Phase::Probe { spec: false, left: PLAIN_PROBE_TICKS },
            Phase::Run { spec: false, .. } => Phase::Probe { spec: true, left: SPEC_PROBE_TICKS },
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_draft_is_what_followed_the_last_occurrence_of_the_tail() {
        let h = [1, 2, 3, 4, 9, 9, 2, 3];
        assert_eq!(propose(&h, 3), vec![4, 9, 9], "the 2-gram [2, 3] last occurred at 1");
        let h = [5, 6, 7, 8, 5, 6, 7];
        assert_eq!(propose(&h, 4), vec![8, 5, 6, 7], "the 3-gram wins; the draft stops at the history's end");
        assert_eq!(propose(&[1, 2, 3, 4], 4), Vec::<u32>::new(), "nothing repeats");
        assert_eq!(propose(&[1, 1, 1], 2), vec![1], "a run copies itself");
        assert_eq!(propose(&[1, 2, 1, 2], 0), Vec::<u32>::new());
    }

    #[test]
    fn the_daemon_config_parses_and_an_empty_one_is_the_default() {
        assert_eq!(SpecConfig::parse(""), SpecConfig::default());
        let c = SpecConfig::parse("max_draft=6,adaptive=0,min_yield=0.500000,yield_rounds=8,throughput_gate=0,gate_probe_tokens=16,min_speedup=1.2,gate_reprobe=4,gate_reprobe_max=64,bitexact=0");
        assert_eq!((c.max_draft, c.adaptive, c.yield_rounds, c.throughput_gate, c.gate_reprobe, c.gate_reprobe_max), (6, false, 8, false, 4, 64));
        assert!((c.min_yield - 0.5).abs() < 1e-9 && (c.min_speedup - 1.2).abs() < 1e-9);
        assert_eq!(SpecConfig::parse("max_draft=99").max_draft, VERIFY_ROWS_MAX as usize - 1, "never more than a pass verifies");
    }

    #[test]
    fn a_lane_that_yields_too_little_rests_and_tries_again() {
        let cfg = SpecConfig { yield_rounds: 4, min_yield: 1.0, gate_reprobe: 3, ..SpecConfig::default() };
        let mut l = LaneSpec::new(&cfg);
        for _ in 0..4 {
            assert!(l.drafting());
            l.record(&cfg, 4, 0);
        }
        assert_eq!(l.depth, 1, "adaptive depth follows what is accepted");
        assert!(!l.drafting() && !l.drafting() && !l.drafting(), "rests three rounds");
        assert!(l.drafting(), "then probes again");
    }

    #[test]
    fn the_gate_closes_a_lane_count_where_drafting_is_slower_and_keeps_it_open_where_faster() {
        let cfg = SpecConfig { gate_reprobe: 4, min_speedup: 1.05, ..SpecConfig::default() };
        let mut g = ThroughputGate::default();
        // Lane count 8: drafting ticks run 80 tok/s, plain ones 100.
        let mut log = Vec::new();
        for _ in 0..40 {
            let on = g.allows(&cfg, 8);
            g.record(&cfg, 8, on, 1.0, if on { 80 } else { 100 });
            log.push(on);
        }
        let tail_on = log[20..].iter().filter(|x| **x).count();
        assert!(tail_on <= 6, "mostly plain once measured slower: {log:?}");
        // Lane count 1: drafting 150, plain 100.
        let mut log = Vec::new();
        for _ in 0..40 {
            let on = g.allows(&cfg, 1);
            g.record(&cfg, 1, on, 1.0, if on { 150 } else { 100 });
            log.push(on);
        }
        let tail_on = log[20..].iter().filter(|x| **x).count();
        assert!(tail_on >= 14, "mostly drafting once measured faster: {log:?}");
        // Then the traffic changes and drafting falls clearly behind (by the
        // margin it had to win by): it stops before the window runs out.
        assert!(g.allows(&cfg, 1));
        g.record(&cfg, 1, false, 1.0, 100);
        for _ in 0..3 {
            g.record(&cfg, 1, true, 1.0, 30);
        }
        assert!(!g.allows(&cfg, 1), "switched off before the window ran out");
    }

    #[test]
    fn the_first_drafting_tick_is_not_measured() {
        let cfg = SpecConfig::default();
        let mut g = ThroughputGate::default();
        assert!(g.allows(&cfg, 2));
        g.record(&cfg, 2, true, 10.0, 1);
        for _ in 0..2 {
            g.record(&cfg, 2, true, 1.0, 150);
        }
        for _ in 0..3 {
            g.record(&cfg, 2, false, 1.0, 100);
        }
        assert!(g.allows(&cfg, 2), "a slow first tick (kernels compiling) does not close the gate");
    }
}
