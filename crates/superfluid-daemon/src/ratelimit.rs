//! Token-bucket rate limiter keyed by client IP (`--rate-limit N`).

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::Instant;

const MAX_BUCKETS: usize = 100_000;
const IDLE_EVICT_SECS: f64 = 300.0;
const SWEEP_EVERY: u32 = 1000;

struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

impl Bucket {
    fn take(&mut self, rpm: f64, now: Instant) -> bool {
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * (rpm / 60.0)).min(rpm);
        self.last_refill = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            return true;
        }
        false
    }
}

pub struct TokenBucket {
    rpm: u32,
    bucket: Mutex<Bucket>,
}

impl TokenBucket {
    pub fn new(rpm: u32) -> TokenBucket {
        TokenBucket { rpm, bucket: Mutex::new(Bucket { tokens: rpm as f64, last_refill: Instant::now() }) }
    }

    pub fn allow(&self) -> bool {
        self.rpm == 0 || self.bucket.lock().expect("token bucket").take(self.rpm as f64, Instant::now())
    }

    pub fn next_token_in(&self) -> std::time::Duration {
        if self.rpm == 0 {
            return std::time::Duration::ZERO;
        }
        let b = self.bucket.lock().expect("token bucket");
        let rpm = self.rpm as f64;
        let now_tokens = (b.tokens + Instant::now().duration_since(b.last_refill).as_secs_f64() * rpm / 60.0).min(rpm);
        std::time::Duration::from_secs_f64(((1.0 - now_tokens).max(0.0)) * 60.0 / rpm)
    }
}

pub struct RateLimiter {
    rpm: u32,
    state: Mutex<State>,
}

struct State {
    buckets: HashMap<IpAddr, Bucket>,
    sweep_counter: u32,
}

impl RateLimiter {
    pub fn new(rpm: u32) -> RateLimiter {
        RateLimiter {
            rpm,
            state: Mutex::new(State { buckets: HashMap::new(), sweep_counter: 0 }),
        }
    }

    pub fn allow(&self, client: IpAddr) -> bool {
        if self.rpm == 0 {
            return true;
        }
        let now = Instant::now();
        let mut st = self.state.lock().expect("rate limiter");
        st.sweep_counter += 1;
        if st.sweep_counter >= SWEEP_EVERY {
            st.sweep_counter = 0;
            evict_idle(&mut st.buckets, now);
        }
        if st.buckets.len() >= MAX_BUCKETS && !st.buckets.contains_key(&client) {
            evict_idle(&mut st.buckets, now);
            if st.buckets.len() >= MAX_BUCKETS {
                evict_least_recent(&mut st.buckets);
            }
        }
        let rpm = self.rpm as f64;
        st.buckets
            .entry(client)
            .or_insert(Bucket { tokens: rpm, last_refill: now })
            .take(rpm, now)
    }

    pub fn bucket_count(&self) -> usize {
        self.state.lock().expect("rate limiter").buckets.len()
    }
}

fn evict_idle(buckets: &mut HashMap<IpAddr, Bucket>, now: Instant) {
    buckets.retain(|_, b| now.duration_since(b.last_refill).as_secs_f64() <= IDLE_EVICT_SECS);
}

fn evict_least_recent(buckets: &mut HashMap<IpAddr, Bucket>) {
    if buckets.is_empty() {
        return;
    }
    let mut stamps: Vec<Instant> = buckets.values().map(|b| b.last_refill).collect();
    let drop = (stamps.len() / 10 + 1).min(stamps.len() - 1);
    stamps.sort_unstable();
    let cutoff = stamps[drop];
    buckets.retain(|_, b| b.last_refill >= cutoff);
}
