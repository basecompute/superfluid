//! Keystroke-storm protection for FIM completions.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const DEFAULT_RATE_PER_SEC: f64 = 10.0;
pub const DEFAULT_BURST: u32 = 20;

const MAX_CLIENTS: usize = 100_000;
const IDLE_EVICT: Duration = Duration::from_secs(300);
const SWEEP_EVERY: u32 = 1000;
const MIN_RATE_PER_SEC: f64 = 0.001;
const MAX_WAIT: Duration = Duration::from_secs(24 * 3600);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BucketConfig {
    pub rate_per_sec: f64,
    pub burst: u32,
}

impl Default for BucketConfig {
    fn default() -> BucketConfig {
        BucketConfig {
            rate_per_sec: DEFAULT_RATE_PER_SEC,
            burst: DEFAULT_BURST,
        }
    }
}

impl BucketConfig {
    pub const OFF: BucketConfig = BucketConfig {
        rate_per_sec: 0.0,
        burst: 0,
    };

    pub fn enabled(&self) -> bool {
        self.rate_per_sec > 0.0
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.rate_per_sec.is_finite() || self.rate_per_sec < 0.0 {
            return Err("completion rate must be finite and non-negative");
        }
        if self.enabled() && self.rate_per_sec < MIN_RATE_PER_SEC {
            return Err("completion rate must be 0 (off) or at least 0.001 per second");
        }
        if self.enabled() && self.burst == 0 {
            return Err("completion burst must be at least 1 when the rate is non-zero");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TokenBucket {
    tokens: f64,
    last: Instant,
}

impl TokenBucket {
    pub fn new(cfg: BucketConfig, now: Instant) -> TokenBucket {
        TokenBucket {
            tokens: cfg.burst as f64,
            last: now,
        }
    }

    pub fn try_take(&mut self, cfg: BucketConfig, now: Instant) -> Result<(), Duration> {
        if !cfg.enabled() {
            return Ok(());
        }
        let elapsed = now.saturating_duration_since(self.last).as_secs_f64();
        self.tokens = (self.tokens + elapsed * cfg.rate_per_sec).min(cfg.burst as f64);
        self.last = now;
        if self.tokens >= 1.0 - 1e-9 {
            self.tokens = (self.tokens - 1.0).max(0.0);
            return Ok(());
        }
        let wait = (1.0 - self.tokens) / cfg.rate_per_sec;
        Err(Duration::try_from_secs_f64(wait).map_or(MAX_WAIT, |d| d.min(MAX_WAIT)))
    }
}

pub struct ClientBuckets {
    state: Mutex<State>,
}

struct State {
    buckets: HashMap<u64, TokenBucket>,
    sweep_counter: u32,
}

impl Default for ClientBuckets {
    fn default() -> ClientBuckets {
        ClientBuckets::new()
    }
}

impl ClientBuckets {
    pub fn new() -> ClientBuckets {
        ClientBuckets {
            state: Mutex::new(State {
                buckets: HashMap::new(),
                sweep_counter: 0,
            }),
        }
    }

    pub fn try_take(&self, client: u64, cfg: BucketConfig) -> Result<(), Duration> {
        if !cfg.enabled() {
            return Ok(());
        }
        let now = Instant::now();
        let mut st = self.state.lock().expect("completion buckets");
        st.sweep_counter += 1;
        if st.sweep_counter >= SWEEP_EVERY {
            st.sweep_counter = 0;
            st.buckets.retain(|_, b| now.saturating_duration_since(b.last) <= IDLE_EVICT);
        }
        if st.buckets.len() >= MAX_CLIENTS && !st.buckets.contains_key(&client) {
            st.buckets.retain(|_, b| now.saturating_duration_since(b.last) <= IDLE_EVICT);
            if st.buckets.len() >= MAX_CLIENTS {
                let mut stamps: Vec<Instant> = st.buckets.values().map(|b| b.last).collect();
                stamps.sort_unstable();
                let cutoff = stamps[(stamps.len() / 10).min(stamps.len() - 1)];
                st.buckets.retain(|_, b| b.last > cutoff);
            }
        }
        st.buckets
            .entry(client)
            .or_insert_with(|| TokenBucket::new(cfg, now))
            .try_take(cfg, now)
    }

    pub fn len(&self) -> usize {
        self.state.lock().expect("completion buckets").buckets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub fn http_client_key(peer: std::net::IpAddr, credential: Option<&str>) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    peer.hash(&mut h);
    credential.hash(&mut h);
    h.finish()
}

pub fn retry_after_secs(wait: Duration) -> u64 {
    wait.as_secs_f64().ceil().max(1.0) as u64
}

pub fn retry_after_ms(wait: Duration) -> u64 {
    let ms = wait.as_nanos().div_ceil(1_000_000);
    u64::try_from(ms).unwrap_or(u64::MAX).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: BucketConfig = BucketConfig {
        rate_per_sec: 2.0,
        burst: 3,
    };

    #[test]
    fn burst_then_refuse_then_refill() {
        let t0 = Instant::now();
        let mut b = TokenBucket::new(CFG, t0);
        for _ in 0..3 {
            assert!(b.try_take(CFG, t0).is_ok());
        }
        let wait = b.try_take(CFG, t0).unwrap_err();
        assert!((wait.as_secs_f64() - 0.5).abs() < 1e-6, "{wait:?}");
        let t1 = t0 + Duration::from_millis(500);
        assert!(b.try_take(CFG, t1).is_ok());
        assert!(b.try_take(CFG, t1).is_err());
        let t2 = t1 + Duration::from_secs(60);
        for _ in 0..3 {
            assert!(b.try_take(CFG, t2).is_ok());
        }
        assert!(b.try_take(CFG, t2).is_err());
    }

    #[test]
    fn the_advertised_wait_is_enough() {
        for rate in [3.0, 7.0, 9.7, 13.0] {
            let cfg = BucketConfig { rate_per_sec: rate, burst: 1 };
            let t0 = Instant::now();
            let mut b = TokenBucket::new(cfg, t0);
            b.try_take(cfg, t0).unwrap();
            let wait = b.try_take(cfg, t0).unwrap_err();
            let t1 = t0 + Duration::from_millis(retry_after_ms(wait));
            assert!(b.try_take(cfg, t1).is_ok(), "rate {rate}: {wait:?}");
        }
    }

    #[test]
    fn a_denormal_rate_saturates_instead_of_panicking() {
        let cfg = BucketConfig { rate_per_sec: 1e-320, burst: 1 };
        let t0 = Instant::now();
        let mut b = TokenBucket::new(cfg, t0);
        b.try_take(cfg, t0).unwrap();
        let wait = b.try_take(cfg, t0).unwrap_err();
        assert_eq!(wait, MAX_WAIT);
        assert_eq!(retry_after_secs(wait), 24 * 3600);
    }

    #[test]
    fn off_never_refuses() {
        let t0 = Instant::now();
        let mut b = TokenBucket::new(BucketConfig::OFF, t0);
        for _ in 0..1000 {
            assert!(b.try_take(BucketConfig::OFF, t0).is_ok());
        }
        let m = ClientBuckets::new();
        assert!(m.try_take(1, BucketConfig::OFF).is_ok());
        assert!(m.is_empty(), "a disabled bucket tracks nobody");
    }

    #[test]
    fn clients_are_independent() {
        let m = ClientBuckets::new();
        let ip: std::net::IpAddr = "127.0.0.1".parse().unwrap();
        let a = http_client_key(ip, Some("key-a"));
        let b = http_client_key(ip, Some("key-b"));
        assert_ne!(a, b, "two credentials on one host are two clients");
        assert_eq!(a, http_client_key(ip, Some("key-a")), "stable across requests");
        for _ in 0..3 {
            assert!(m.try_take(a, CFG).is_ok());
        }
        assert!(m.try_take(a, CFG).is_err());
        assert!(m.try_take(b, CFG).is_ok(), "b is not charged for a's storm");
    }

    #[test]
    fn validation() {
        assert!(BucketConfig::default().validate().is_ok());
        assert!(BucketConfig::OFF.validate().is_ok());
        assert!(BucketConfig { rate_per_sec: 1.0, burst: 0 }.validate().is_err());
        assert!(BucketConfig { rate_per_sec: f64::NAN, burst: 1 }.validate().is_err());
        assert!(BucketConfig { rate_per_sec: -1.0, burst: 1 }.validate().is_err());
        assert!(BucketConfig { rate_per_sec: 1e-320, burst: 1 }.validate().is_err());
        assert!(BucketConfig { rate_per_sec: 0.001, burst: 1 }.validate().is_ok());
        assert_eq!(retry_after_secs(Duration::from_millis(1)), 1);
        assert_eq!(retry_after_secs(Duration::from_millis(1500)), 2);
        assert_eq!(retry_after_secs(Duration::from_nanos(1_000_000_001)), 2);
        assert_eq!(retry_after_ms(Duration::from_nanos(99_000_001)), 100);
        assert_eq!(retry_after_ms(Duration::from_millis(99)), 99);
        assert_eq!(retry_after_ms(Duration::ZERO), 1);
    }
}
