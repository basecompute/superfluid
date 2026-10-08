//! The multi-lane scheduler.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

use superfluid_abi::{encoding, finish, op_state};
use superfluid_proto::linkw::{self, ALL_SPACES};

use crate::bus::SessionBus;
use crate::codec::{Channelizer, TextCodec, Utf8Stream};
use crate::park;
use crate::runtime::EngineHost;
use crate::store::{CommittedEvent, SessionStore};
use crate::wal::GenParams;
use crate::DaemonError;

const KV_SPACE: u32 = 1;
const MAX_WORKER_RESPAWNS: u32 = 3;
// An admission stages at most two token-in slots: its prompt and a resumed prefix.
const STAGES_PER_ADMISSION: u32 = 2;

fn lane_cap(configured: usize, max_seqs: Option<u64>, emit_slots: Option<u32>) -> usize {
    let lanes = match max_seqs {
        Some(n) if (n as usize) < configured => {
            eprintln!("[superfluid] {configured} lanes asked for; the runtime holds {n} sequences, so {n} run");
            (n as usize).max(1)
        }
        _ => configured,
    };
    // Every decoding lane writes one token-out slot a tick, read back after
    // it: more lanes than slots would overwrite the first emits.
    match emit_slots {
        Some(n) if (n as usize) < lanes => {
            eprintln!("[superfluid] {lanes} lanes asked for; the worker's token-out ring holds {n} emits a tick, so {n} run");
            (n as usize).max(1)
        }
        _ => lanes,
    }
}

fn ring_slots(host: &mut EngineHost, ring_id: u32) -> Option<u32> {
    if !host.is_alive() {
        return None;
    }
    host.client()
        .hello
        .limits
        .ring_specs
        .iter()
        .find(|r| r.ring_id == ring_id)
        .map(|r| r.slots)
}

fn worker_is_gone(e: &DaemonError) -> bool {
    matches!(
        e,
        DaemonError::Agent(
            superfluid_agent::AgentError::Io(_)
                | superfluid_agent::AgentError::Closed
                | superfluid_agent::AgentError::Envelope(_)
        )
    )
}

pub const PREFILL_BUDGET: u32 = 4096;
pub const SOLO_PREFILL_BUDGET: u32 = 16384;
const TICK_DECODE: u16 = 32;
const TICK_DECODE_MAX: u16 = 256;
const DECODE_SHARE_BESIDE_PREFILL: f64 = 0.25;
/// The same while a chat or an inline completion decodes: its reply is
/// short and watched, so the prefill beside it waits more, not the reply.
const DECODE_SHARE_BESIDE_PREFILL_INTERACTIVE: f64 = 2.0;
const TICK_DECODE_BESIDE_PREFILL: u16 = 8;
const TARGET_TICK_TOTAL_SECS: f64 = 2.0;
const OPEN_TICK_SECS: f64 = 0.25;
const MAX_PREFILL_TICK_SECS: f64 = 8.0;
const MAX_LONE_PREFILL_TICK_SECS: f64 = 16.0;
/// A tick's prefill while a request could still join: a newcomer waits for
/// the tick in flight, so this bounds its time to first token.
const OPEN_PREFILL_TICK_SECS: f64 = 0.5;
/// And in tokens, whatever the learned rate says: an engine that returns
/// before its GPU work ends makes an early tick look all but free.
const OPEN_PREFILL_TICK_MAX: u32 = 1024;

/// The open tick's target on an engine that ends a tick at its next step when a
/// preempting request arrives.
const OPEN_PREFILL_TICK_SECS_YIELDING: f64 = 2.0;

/// How long after the last chat or inline completion the engine keeps its
/// prefill steps short.
const LATENCY_HOLD: std::time::Duration = std::time::Duration::from_secs(300);

/// A queued request waits for a lane prefilling the same prefix (at least this long) when
/// that lane retires soon, and starts from its cached prefix.
const SHARED_PREFIX_MIN_TOKENS: u64 = 1024;
/// "Soon": at most this many tokens left to generate.
const SHARED_PREFIX_TAIL_TOKENS: u32 = 64;
/// Never hold a request longer than this.
const SHARED_PREFIX_MAX_HOLD_TICKS: u64 = 64;
/// On a runtime whose state cannot be cut back, a chat's input of at least
/// this many tokens is checkpointed where it ends (the executor's floor for
/// the checkpoint where a prompt ends): less costs little to run again.
const TURN_CHECKPOINT_MIN_TOKENS: u64 = 128;
/// The turn checkpoints remembered so that a later turn's supersedes them.
/// One forgotten stays in the cache until evicted.
const TURN_CHECKPOINTS_TRACKED: usize = 64;

const PREFILL_RATE_MIN_TOKENS: u32 = 256;


/// The daemon's cache classes (a bitset the engine maps its entries onto):
/// a resident prefix, and one that has started two lanes or more.
const SHARED_PREFIX_CLASS: u64 = 2;
/// The prefix cache's entries held in host memory (llama.cpp's exports): they
/// free nothing in the pool, and are named only when the machine is short of
/// memory.
const HOST_EXPORT_CLASS: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
struct TickShape {
    open_secs: f64,
    full_secs: f64,
    decode_floor: u16,
    beside_prefill: u16,
}

const NATIVE_TICKS: TickShape =
    TickShape { open_secs: OPEN_TICK_SECS, full_secs: f64::INFINITY, decode_floor: TICK_DECODE, beside_prefill: TICK_DECODE_BESIDE_PREFILL };
const ROUND_TICKS: TickShape = TickShape { open_secs: 0.05, full_secs: 0.25, decode_floor: 4, beside_prefill: 1 };

fn tick_target(shape: &TickShape, configured: f64, lanes: usize, max_lanes: usize, finishing: bool) -> f64 {
    if lanes < max_lanes || finishing {
        configured.min(shape.open_secs)
    } else {
        configured.min(shape.full_secs)
    }
}

pub struct GenerateJob {
    pub session: u64,
    pub stream: Vec<u32>,
    /// Where the conversation's input ends in `stream`, before the
    /// generation prompt the daemon appended to it (0: none appended).
    pub input_end: u64,
    pub params: GenParams,
    pub max_tokens: u32,
    pub events: Sender<LaneEvent>,
    pub class: u8,
    pub batch_invariant: bool,
    pub carry: Option<LaneCarry>,
    pub queued_at: u64,
    pub completion: bool,
    pub deadline: Option<std::time::Instant>,
    pub media: Vec<(u64, std::path::PathBuf, u32)>,
    pub grammar_handle: u32,
    pub extras: GenExtras,
    pub trace: tracing::Span,
}

#[derive(Debug, Clone, Default)]
pub struct GenExtras {
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub grammar_handle: u32,
    pub logit_bias_handle: u32,
    pub want_logprobs: bool,
    pub top_logprobs: u8,
    pub wants_deltas: bool,
    pub ephemeral: bool,
    pub open_channel: u32,
    pub thinking: Option<bool>,
    pub ignore_eos: bool,
    pub seed_drawn: bool,
    pub tool_schemas: Option<std::sync::Arc<crate::codec::ToolSchemas>>,
}

pub struct LaneCarry {
    seg: Box<dyn Channelizer>,
    utf8: Utf8Stream,
    disp_seg: Box<dyn Channelizer>,
    disp_utf8: Utf8Stream,
    tool_buf: String,
    produced_before: u32,
    spec_before: (u64, u64),
    warm_prefix_first: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TokenLogprob {
    pub token: u32,
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

pub enum LaneEvent {
    Committed(CommittedEvent),
    Provisional { channel: u32, text: String, produced: u32 },
    TokenLogprobs(Vec<TokenLogprob>),
    CompletionTokens(Vec<u32>),
    Expired,
    Done {
        tokens_generated: u32,
        finish: u32,
        warm_prefix: u64,
        spec: (u64, u64),
    },
    Failed(String),
}

fn provisional_ingest(
    codec: &Arc<dyn TextCodec + Send + Sync>,
    bus: &SessionBus,
    lane: &mut Lane,
    tokens: &[u32],
) {
    if tokens.is_empty() {
        return;
    }
    lane.disp_tokens += tokens.len() as u32;
    let produced = lane.disp_tokens;
    for run in lane.disp_seg.split(tokens) {
        let mut text = String::new();
        for &t in &run.text {
            text.push_str(&lane.disp_utf8.push(&codec.token_bytes(t)));
        }
        if text.is_empty() {
            continue;
        }
        let channel = run.channel;
        if channel == crate::wal::channel::TEXT || channel == crate::wal::channel::REASONING {
            bus.publish_provisional(lane.session, channel, &text);
            let _ = lane.job.events.send(LaneEvent::Provisional { channel, text, produced });
        }
    }
}

fn provisional_flush(bus: &SessionBus, lane: &mut Lane) {
    let text = lane.disp_utf8.flush();
    if text.is_empty() {
        return;
    }
    let channel = lane.disp_seg.current_channel();
    if channel == crate::wal::channel::TEXT || channel == crate::wal::channel::REASONING {
        bus.publish_provisional(lane.session, channel, &text);
        let produced = lane.disp_tokens;
        let _ = lane.job.events.send(LaneEvent::Provisional { channel, text, produced });
    }
}

pub enum SchedCmd {
    Submit(Box<GenerateJob>),
    Sync(linkw::StateSyncReqMsg, Sender<Result<linkw::StateSyncOkMsg, String>>),
    SyncAs(u8, Box<SchedCmd>),
    SyncStream(
        linkw::StateSyncReqMsg,
        Sender<linkw::TranscribeSegmentMsg>,
        Sender<Result<linkw::StateSyncOkMsg, String>>,
    ),
    Pin(crate::pins::PinReq),
    PinHeld(Vec<crate::pins::PinKey>, Sender<u64>),
    PinUnanchor(u64),
    Shutdown,
}

pub struct SchedulerHandle {
    pub tx: Sender<SchedCmd>,
    join: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl SchedulerHandle {
    pub fn signal_shutdown(&self) {
        let _ = self.tx.send(SchedCmd::Shutdown);
    }

    pub fn shutdown(&self) {
        let mut guard = self.join.lock().expect("join");
        if let Some(j) = guard.take() {
            self.signal_shutdown();
            let _ = j.join();
        }
    }
}

impl Drop for SchedulerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

enum LanePhase {
    Prefilling,
    Decoding,
    Finishing,
}

struct Lane {
    tag: u64,
    session: u64,
    job: GenerateJob,
    phase: LanePhase,
    prefilled: u64,
    produced: u32,
    fin: u32,
    warm_prefix: u64,
    utf8: Utf8Stream,
    seg: Box<dyn Channelizer>,
    disp_seg: Box<dyn Channelizer>,
    disp_utf8: Utf8Stream,
    frame_fed: usize,
    emit_dead: bool,
    tool_buf: String,
    admitted: bool,
    seed_handle: u64,
    span: linkw::TokenRefMsg,
    all_tokens: Vec<u32>,
    parked: bool,
    parked_lossy: bool,
    preempted: bool,
    starved_ticks: u32,
    prefill_starved_ticks: u32,
    /// On a recurrent runtime: where this lane's prefill stops to keep a state for queued
    /// requests sharing that prefix (0: nowhere), whether this chunk ends there, and the
    /// longest state kept.
    checkpoint_at: u64,
    checkpoint_due: bool,
    checkpointed: u64,
    /// Whether the state where the job's input ends has been kept (or tried).
    turn_kept: bool,
    produced_before: u32,
    spec: (u64, u64),
    disp_tokens: u32,
    warm_prefix_first: Option<u64>,
    admitted_at: std::time::Instant,
    ttft_observed: bool,
    ttft_frame_ms: Option<u64>,
    fault_code: Option<u32>,
}

#[derive(Default)]
struct ClassQueue {
    jobs: VecDeque<GenerateJob>,
}

pub fn kv_bits_label(bits: u64) -> Option<&'static str> {
    match bits {
        16 => Some("f16"),
        8 => Some("q8_0"),
        4 => Some("q4_0"),
        84 => Some("K q8_0 / V q4_0"),
        _ => None,
    }
}

#[derive(Debug, Default)]
pub struct SchedStats {
    pub pressure_evictions: std::sync::atomic::AtomicU64,
    pub pressure_bytes_evicted: std::sync::atomic::AtomicU64,
    pub unservable_cold_admissions: std::sync::atomic::AtomicU64,
    pub unservable_park_refusals: std::sync::atomic::AtomicU64,
    pub preemptions: std::sync::atomic::AtomicU64,
    pub admits_deferred: std::sync::atomic::AtomicU64,
    pub starvation_grants: std::sync::atomic::AtomicU64,
    pub os_pressure_events: std::sync::atomic::AtomicU64,
    pub ticks: std::sync::atomic::AtomicU64,
    pub lanes_active: std::sync::atomic::AtomicU64,
    pub queue_depth: [std::sync::atomic::AtomicU64; 4],
    pub prefill_tokens: std::sync::atomic::AtomicU64,
    pub decode_tokens: std::sync::atomic::AtomicU64,
    pub warm_prefix_tokens: std::sync::atomic::AtomicU64,
    pub cold_admissions: std::sync::atomic::AtomicU64,
    pub parks_lossless: std::sync::atomic::AtomicU64,
    pub parks_lossy: std::sync::atomic::AtomicU64,
    pub resumes: std::sync::atomic::AtomicU64,
    pub worker_respawns: std::sync::atomic::AtomicU64,
    pub spec_proposed: std::sync::atomic::AtomicU64,
    pub spec_accepted: std::sync::atomic::AtomicU64,
    pub ttft_buckets: [std::sync::atomic::AtomicU64; TTFT_BUCKETS_MS.len() + 1],
    pub ttft_sum_ms: std::sync::atomic::AtomicU64,
    pub ttft_count: std::sync::atomic::AtomicU64,
    pub ttft_last_ms: std::sync::atomic::AtomicU64,
    pub pool_blocks_used: std::sync::atomic::AtomicU64,
    pub prefill_budget_live: std::sync::atomic::AtomicU64,
    pub decode_grant_live: std::sync::atomic::AtomicU64,
    pub tick_target_live_ms: std::sync::atomic::AtomicU64,
    pub pool_blocks_total: std::sync::atomic::AtomicU64,
    pub kv_bits: std::sync::atomic::AtomicU64,
    pub pins_held: std::sync::atomic::AtomicU64,
    pub pinned_bytes: std::sync::atomic::AtomicU64,
    pub pins_yielded: std::sync::atomic::AtomicU64,
    pub pins_expired: std::sync::atomic::AtomicU64,
    pub tick_wall_ms_live: std::sync::atomic::AtomicU64,
    pub prefill_rate_live: std::sync::atomic::AtomicU64,
    pub decode_rate_live: std::sync::atomic::AtomicU64,
    pub work_ticks: std::sync::atomic::AtomicU64,
    tick_seq: std::sync::atomic::AtomicU64,
    tick_since_ms: std::sync::atomic::AtomicU64,
    tick_prefill_tokens: std::sync::atomic::AtomicU64,
    tick_prefill_lanes: std::sync::atomic::AtomicU64,
    tick_decode_lanes: std::sync::atomic::AtomicU64,
    tick_admits: std::sync::atomic::AtomicU64,
    tick_retires: std::sync::atomic::AtomicU64,
    tick_target_ms: std::sync::atomic::AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TickInFlight {
    pub since_ms: u64,
    pub prefill_tokens: u64,
    pub prefill_lanes: u64,
    pub decode_lanes: u64,
    pub admits: u64,
    pub retires: u64,
    pub target_ms: u64,
}

impl SchedStats {
    pub fn set_tick_in_flight(&self, t: TickInFlight) {
        use std::sync::atomic::Ordering::SeqCst;
        self.tick_seq.fetch_add(1, SeqCst);
        self.tick_since_ms.store(t.since_ms, SeqCst);
        self.tick_prefill_tokens.store(t.prefill_tokens, SeqCst);
        self.tick_prefill_lanes.store(t.prefill_lanes, SeqCst);
        self.tick_decode_lanes.store(t.decode_lanes, SeqCst);
        self.tick_admits.store(t.admits, SeqCst);
        self.tick_retires.store(t.retires, SeqCst);
        self.tick_target_ms.store(t.target_ms, SeqCst);
        self.tick_seq.fetch_add(1, SeqCst);
    }

    pub fn clear_tick_in_flight(&self) {
        self.set_tick_in_flight(TickInFlight::default());
    }

    pub fn tick_in_flight(&self) -> Option<TickInFlight> {
        use std::sync::atomic::Ordering::SeqCst;
        loop {
            let s1 = self.tick_seq.load(SeqCst);
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let t = TickInFlight {
                since_ms: self.tick_since_ms.load(SeqCst),
                prefill_tokens: self.tick_prefill_tokens.load(SeqCst),
                prefill_lanes: self.tick_prefill_lanes.load(SeqCst),
                decode_lanes: self.tick_decode_lanes.load(SeqCst),
                admits: self.tick_admits.load(SeqCst),
                retires: self.tick_retires.load(SeqCst),
                target_ms: self.tick_target_ms.load(SeqCst),
            };
            if self.tick_seq.load(SeqCst) == s1 {
                return (t.since_ms != 0).then_some(t);
            }
        }
    }
}

pub fn uptime_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

fn per_sec(tokens: u64, secs: f64) -> u64 {
    if tokens == 0 || secs <= 0.0 {
        0
    } else {
        (tokens as f64 / secs) as u64
    }
}

pub const TTFT_BUCKETS_MS: [u64; 13] =
    [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10_000, 20_000, 40_000];

impl SchedStats {
    pub fn observe_ttft(&self, ms: u64) {
        let idx = TTFT_BUCKETS_MS
            .iter()
            .position(|&b| ms <= b)
            .unwrap_or(TTFT_BUCKETS_MS.len());
        self.ttft_buckets[idx].fetch_add(1, Ordering::Relaxed);
        self.ttft_sum_ms.fetch_add(ms, Ordering::Relaxed);
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
        self.ttft_last_ms.store(ms, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculationGrant {
    pub strategy_id: String,
    pub slot: u32,
    pub class: u8,
    pub cert_id: u32,
    pub identity: u64,
    pub modes: u8,
}

fn proposal_config(id: &str, opts: &crate::SpeculationOptions) -> String {
    fn env_depth(key: &str) -> Option<u32> {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|n| (1..=15).contains(n))
    }
    fn env_bool(key: &str, default: bool) -> bool {
        std::env::var(key)
            .ok()
            .map(|v| v.trim().parse::<i32>().unwrap_or(i32::from(default)) != 0)
            .unwrap_or(default)
    }
    let strategy_depth = || match id {
        "mtp-head" => env_depth("SUPERFLUID_MTP_DRAFTS").unwrap_or(0),
        "eagle3" => env_depth("SUPERFLUID_EAGLE_DRAFTS").unwrap_or(2),
        "dflash" | "dspark" => env_depth("SUPERFLUID_DFLASH_DRAFTS").unwrap_or(0),
        _ => 4,
    };
    let max_draft = opts.max_draft.unwrap_or_else(strategy_depth);
    let adaptive = opts
        .adaptive
        .unwrap_or_else(|| env_bool("SUPERFLUID_SPEC_ADAPTIVE", true));
    let min_yield = opts.min_yield.unwrap_or_else(|| {
        std::env::var("SUPERFLUID_SPEC_MIN_YIELD")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.75)
    });
    let yield_rounds = opts.yield_rounds.unwrap_or_else(|| {
        std::env::var("SUPERFLUID_SPEC_LANE_YIELD_ROUNDS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(24)
    });
    let throughput_gate = opts
        .throughput_gate
        .unwrap_or_else(|| env_bool("SUPERFLUID_SPEC_GATE", true));
    let gate_probe_tokens = opts.gate_probe_tokens.unwrap_or_else(|| {
        std::env::var("SUPERFLUID_SPEC_GATE_PROBE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(16)
    });
    let min_speedup = opts.min_speedup.unwrap_or_else(|| {
        std::env::var("SUPERFLUID_SPEC_MIN_SPEEDUP")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 1.0)
            .unwrap_or(1.08)
    });
    let env_u32 = |key: &str, default: u32| {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(default)
    };
    let gate_reprobe = opts
        .gate_reprobe
        .unwrap_or_else(|| env_u32("SUPERFLUID_SPEC_GATE_REPROBE", 32));
    let gate_reprobe_max = opts
        .gate_reprobe_max
        .unwrap_or_else(|| env_u32("SUPERFLUID_SPEC_GATE_REPROBE_MAX", 1024));
    let bitexact = env_bool("BASERT_SPEC_BITEXACT", false);
    let mut cfg = format!(
        "max_draft={max_draft},adaptive={},min_yield={min_yield:.6},yield_rounds={yield_rounds},throughput_gate={},gate_probe_tokens={gate_probe_tokens},min_speedup={min_speedup:.6},gate_reprobe={gate_reprobe},gate_reprobe_max={gate_reprobe_max},bitexact={}",
        u8::from(adaptive),
        u8::from(throughput_gate),
        u8::from(bitexact),
    );
    if id == "dspark" {
        let confidence = opts.dspark_confidence.or_else(|| {
            std::env::var("SUPERFLUID_DSPARK_CONFIDENCE")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|t| t.is_finite() && *t > 0.0)
        });
        if let Some(t) = confidence {
            cfg.push_str(&format!(",confidence={t:.6}"));
        }
    }
    cfg
}

fn prefill_tick_cap(secs_per_token: f64, lone: bool) -> u32 {
    let secs = if lone { MAX_LONE_PREFILL_TICK_SECS } else { MAX_PREFILL_TICK_SECS };
    (secs / secs_per_token).clamp(512.0, PREFILL_BUDGET as f64) as u32
}

/// Whole 512-token chunks; whole 128-token steps on a model where one chunk passes the
/// half second.
/// How long an open tick's prefill may run: longer on an engine that yields at its next
/// step, unless a latency-sensitive request is around.
fn open_prefill_tick_target(yields: bool, latency: bool) -> f64 {
    if yields && !latency {
        OPEN_PREFILL_TICK_SECS_YIELDING
    } else {
        OPEN_PREFILL_TICK_SECS
    }
}

fn open_prefill_tick_cap(secs_per_token: f64, target_secs: f64) -> u32 {
    let tokens = (target_secs / secs_per_token).clamp(128.0, OPEN_PREFILL_TICK_MAX as f64) as u32;
    let step = if tokens < 512 { 128 } else { 512 };
    tokens / step * step
}

/// Why a shared-prefix checkpoint was not kept: the engine forks no lane's
/// sequence (none will be), or this one failed.
enum CheckpointError {
    Refused(String),
    Failed(String),
}

impl CheckpointError {
    /// Only an engine that says it cannot fork refuses for good; anything
    /// else (a lane that faulted, a full sequence table) fails this one.
    fn of(e: superfluid_agent::AgentError) -> CheckpointError {
        if matches!(e, superfluid_agent::AgentError::Rejected(s) if s == superfluid_abi::Status::Unsupported as i32) {
            CheckpointError::Refused(e.to_string())
        } else {
            CheckpointError::Failed(e.to_string())
        }
    }
}

struct PrefillBudget {
    left: u64,
    tail_capped: bool,
    held_below: Option<u8>,
}

impl PrefillBudget {
    const TAIL_FLOOR: u64 = 256;

    fn new(budget: u64) -> Self {
        PrefillBudget { left: budget, tail_capped: false, held_below: None }
    }

    fn offer(&self, want: u64) -> u64 {
        want.min(self.left)
    }

    fn offer_to(&self, want: u64, class: u8) -> u64 {
        if self.held_below.is_some_and(|c| class > c) {
            return 0;
        }
        self.offer(want)
    }

    /// A chat or an inline completion finished its prompt this tick: nothing
    /// of a lower class prefills beside it, so its first token is not held
    /// behind someone else's prompt.
    fn hold_below(&mut self, class: u8) {
        self.held_below = Some(self.held_below.map_or(class, |c| c.min(class)));
    }

    fn take(&mut self, chunk: u64, completes: bool) {
        self.left -= chunk;
        if completes && !self.tail_capped {
            self.tail_capped = true;
            self.left = self.left.min(chunk.max(Self::TAIL_FLOOR));
        }
    }
}

/// `plan` with only its first `keep` admissions, and none of the work of the
/// lanes it drops.
fn with_first_admits(plan: &linkw::TickPlanMsg, keep: usize) -> linkw::TickPlanMsg {
    let mut p = plan.clone();
    let dropped: std::collections::HashSet<u64> = p.admits.split_off(keep.min(p.admits.len())).iter().map(|a| a.lane_tag).collect();
    p.prefills.retain(|x| !dropped.contains(&x.lane_tag));
    p.decodes.retain(|x| !dropped.contains(&x.lane_tag));
    p.commits.retain(|x| !dropped.contains(&x.lane_tag));
    p
}

/// A lane's starved-tick count after a tick: `waiting` (prefill left, none
/// this tick) counts only when the youngest lane served was admitted after
/// it (lane tags rise with admission).
fn starved_after(prev: u32, waiting: bool, youngest_served: Option<u64>, tag: u64) -> u32 {
    if waiting && youngest_served.is_some_and(|y| y > tag) {
        prev.saturating_add(1)
    } else {
        prev
    }
}

fn prefill_rank(class: u8, remaining: u64, tag: u64, starved_ticks: u32, bound: u32) -> (bool, u8, u64, u64) {
    let starving = remaining > 0 && starved_ticks >= bound;
    (!starving, class, remaining, tag)
}

#[cfg(test)]
mod prefill_order_tests {
    use super::{prefill_rank, prefill_tick_cap, starved_after, PrefillBudget, PREFILL_BUDGET};

    #[test]
    fn prompts_that_arrived_together_prefill_one_after_another() {
        // Eight equal prompts, one tick's budget each time: the shortest
        // left goes first, and nobody is passed over by a younger lane, so
        // they finish in order instead of in turns.
        const BOUND: u32 = 8;
        let mut lanes: Vec<(u64, u64, u32)> = (0..8).map(|t| (t, 7680, 0)).collect();
        let mut finished_at = Vec::new();
        for tick in 0..400u32 {
            let mut order: Vec<usize> = (0..lanes.len()).filter(|&i| lanes[i].1 > 0).collect();
            if order.is_empty() {
                break;
            }
            order.sort_by_key(|&i| prefill_rank(3, lanes[i].1, lanes[i].0, lanes[i].2, BOUND));
            let served = order[0];
            lanes[served].1 -= 512;
            lanes[served].2 = 0;
            if lanes[served].1 == 0 {
                finished_at.push(tick + 1);
            }
            let youngest = Some(lanes[served].0);
            for (i, l) in lanes.iter_mut().enumerate() {
                l.2 = starved_after(l.2, i != served && l.1 > 0, youngest, l.0);
            }
        }
        assert_eq!(finished_at, (1..=8).map(|k| 15 * k).collect::<Vec<_>>(), "fifteen ticks each, in turn");
    }

    #[test]
    fn only_an_engine_that_cannot_fork_refuses_checkpoints_for_good() {
        use super::CheckpointError;
        use superfluid_abi::Status;
        use superfluid_agent::AgentError;
        let refused = |e| matches!(CheckpointError::of(e), CheckpointError::Refused(_));
        assert!(refused(AgentError::Rejected(Status::Unsupported as i32)));
        assert!(!refused(AgentError::Rejected(Status::UnknownHandle as i32)), "a lane that faulted in its tick");
        assert!(!refused(AgentError::Rejected(Status::NeedsReplan as i32)), "a full sequence table");
        assert!(!refused(AgentError::Closed), "a worker that went away");
    }

    #[test]
    fn an_open_tick_prefills_whole_chunks_for_about_half_a_second() {
        use super::open_prefill_tick_cap;
        assert_eq!(open_prefill_tick_cap(1.0 / 1200.0, super::OPEN_PREFILL_TICK_SECS), 512, "600 tokens round down to one chunk");
        assert_eq!(open_prefill_tick_cap(1.0 / 5000.0, super::OPEN_PREFILL_TICK_SECS), 1024, "2500 tokens, at most two chunks");
        assert_eq!(open_prefill_tick_cap(1.0, super::OPEN_PREFILL_TICK_SECS), 128, "never below one step");
        assert_eq!(open_prefill_tick_cap(1.0 / 330.0, super::OPEN_PREFILL_TICK_SECS), 128, "a 27B: 165 tokens round down to one step");
        assert_eq!(open_prefill_tick_cap(1.0 / 700.0, super::OPEN_PREFILL_TICK_SECS), 256, "350 tokens, two steps");
        assert_eq!(open_prefill_tick_cap(1e-6, super::OPEN_PREFILL_TICK_SECS), 1024, "never above two chunks, whatever an early tick suggests");
        assert_eq!(
            open_prefill_tick_cap(1.0 / 330.0, super::OPEN_PREFILL_TICK_SECS_YIELDING),
            512,
            "an engine that yields mid-tick: a 27B's open tick is a whole chunk"
        );
        use super::open_prefill_tick_target as target;
        assert_eq!(target(true, false), super::OPEN_PREFILL_TICK_SECS_YIELDING, "agents alone, on an engine that yields");
        assert_eq!(target(true, true), super::OPEN_PREFILL_TICK_SECS, "a chat around: its next tokens wait out each tick");
        assert_eq!(target(false, false), super::OPEN_PREFILL_TICK_SECS, "an engine that runs a tick whole");
    }

    #[test]
    fn lone_lane_plans_longer_prefill_ticks() {
        let spt = 0.0078125;
        assert_eq!(prefill_tick_cap(spt, false), 1024);
        assert_eq!(prefill_tick_cap(spt, true), 2048);
        assert_eq!(prefill_tick_cap(1.0, true), 512);
        assert_eq!(prefill_tick_cap(1e-6, false), PREFILL_BUDGET);
        assert_eq!(prefill_tick_cap(1e-6, true), PREFILL_BUDGET);
    }

    fn grants(budget: u64, remaining: &[u64]) -> Vec<u64> {
        let mut b = PrefillBudget::new(budget);
        remaining
            .iter()
            .map(|&r| {
                let c = b.offer(r);
                if c > 0 {
                    b.take(c, c == r);
                }
                c
            })
            .collect()
    }

    #[test]
    fn the_tail_after_the_first_completion_is_capped_in_aggregate() {
        let g = grants(4096, &[500, 4000, 4000, 4000, 4000, 4000, 4000, 4000]);
        assert_eq!(g, vec![500, 500, 0, 0, 0, 0, 0, 0]);
        let g = grants(4096, &[40, 4000, 4000]);
        assert_eq!(g, vec![40, 256, 0]);
        let g = grants(4096, &[1000, 300, 200, 4000]);
        assert_eq!(g, vec![1000, 300, 200, 500]);
        assert_eq!(grants(4096, &[3000, 3000]), vec![3000, 1096]);
    }

    #[test]
    fn nothing_of_a_lower_class_prefills_beside_a_chat_that_finishes_its_prompt() {
        let mut b = PrefillBudget::new(4096);
        assert_eq!(b.offer_to(22, 0), 22);
        b.take(22, true);
        b.hold_below(0);
        assert_eq!(b.offer_to(4000, 3), 0, "an agent waits a tick");
        assert_eq!(b.offer_to(100, 0), 100, "another chat does not");
        let mut b = PrefillBudget::new(4096);
        b.take(500, false);
        assert_eq!(b.offer_to(4000, 3), 3596, "a chat still prefilling holds nothing back");
    }

    #[test]
    fn a_long_prefill_progresses_under_a_stream_of_short_prompts() {
        const BUDGET: u64 = 512;
        const BOUND: u32 = 3;
        let mut lanes: Vec<(u64, u64, u32)> = vec![(0, 8 * BUDGET, 0)];
        let mut next_tag = 1;
        let mut long_ticks = Vec::new();
        for tick in 0..64u32 {
            lanes.retain(|l| l.0 == 0 || l.1 > 0);
            while lanes.len() < 3 {
                lanes.push((next_tag, BUDGET, 0));
                next_tag += 1;
            }
            let mut order: Vec<usize> = (0..lanes.len()).collect();
            order.sort_by_key(|&i| prefill_rank(0, lanes[i].1, lanes[i].0, lanes[i].2, BOUND));
            let mut budget = PrefillBudget::new(BUDGET);
            let mut served = Vec::new();
            for i in order {
                let l = &mut lanes[i];
                if l.1 == 0 {
                    continue;
                }
                let chunk = budget.offer(l.1);
                l.1 -= chunk;
                if chunk > 0 {
                    l.2 = 0;
                    served.push(l.0);
                    budget.take(chunk, l.1 == 0);
                }
                if l.0 == 0 && chunk > 0 {
                    long_ticks.push(tick);
                }
            }
            let youngest = served.iter().copied().max();
            for l in lanes.iter_mut() {
                l.2 = starved_after(l.2, l.1 > 0 && !served.contains(&l.0), youngest, l.0);
            }
            assert!(lanes[0].2 <= BOUND, "tick {tick}: the long lane waited past the bound");
            if lanes[0].1 == 0 {
                break;
            }
        }
        assert_eq!(lanes[0], (0, 0, 0), "the long prompt finished prefilling: {long_ticks:?}");
        assert!(long_ticks.windows(2).all(|w| w[1] - w[0] <= BOUND + 1), "bounded gaps: {long_ticks:?}");
        assert!(prefill_rank(0, 8 * BUDGET, 0, 0, BOUND) > prefill_rank(0, BUDGET, 9, 0, BOUND));
        assert!(prefill_rank(5, 8 * BUDGET, 0, BOUND, BOUND) < prefill_rank(0, 1, 9, 0, BOUND));
        assert!(prefill_rank(5, 0, 0, BOUND, BOUND) > prefill_rank(0, 1, 9, 0, BOUND));
    }
}

#[cfg(test)]
mod tick_target_tests {
    use super::{tick_target, NATIVE_TICKS, OPEN_TICK_SECS, ROUND_TICKS};

    #[test]
    fn a_tick_is_short_while_a_joiner_could_be_admitted_next() {
        assert_eq!(tick_target(&NATIVE_TICKS, 2.0, 1, 2, false), OPEN_TICK_SECS, "a slot is free");
        assert_eq!(tick_target(&NATIVE_TICKS, 2.0, 2, 2, false), 2.0, "full and staying full: nothing could admit a joiner");
        assert_eq!(tick_target(&ROUND_TICKS, 2.0, 1, 2, false), 0.05, "a round-granular runtime's open tick is a few rounds");
        assert_eq!(tick_target(&ROUND_TICKS, 2.0, 2, 2, false), 0.25, "and its full one is never long");
        assert_eq!(
            tick_target(&NATIVE_TICKS, 2.0, 2, 2, true),
            OPEN_TICK_SECS,
            "a finishing lane frees a slot, and its client waits on the retire"
        );
        assert_eq!(tick_target(&NATIVE_TICKS, 0.1, 2, 2, true), 0.1, "a configured target under the open one stands");
    }
}

#[cfg(test)]
mod speculation_config_tests {
    use super::proposal_config;
    use crate::SpeculationOptions;

    #[test]
    fn cli_policy_is_serialized_for_the_engine_and_fingerprint() {
        let cfg = proposal_config(
            "dspark",
            &SpeculationOptions {
                max_draft: Some(2),
                adaptive: Some(false),
                min_yield: Some(0.5),
                yield_rounds: Some(12),
                throughput_gate: Some(false),
                gate_probe_tokens: Some(8),
                min_speedup: Some(1.12),
                gate_reprobe: Some(0),
                gate_reprobe_max: Some(64),
                dspark_confidence: Some(0.42),
            },
        );
        assert!(cfg.starts_with(
            "max_draft=2,adaptive=0,min_yield=0.500000,yield_rounds=12,throughput_gate=0,gate_probe_tokens=8,min_speedup=1.120000,gate_reprobe=0,gate_reprobe_max=64,"
        ));
        assert!(cfg.ends_with(",confidence=0.420000"));
    }

    #[test]
    fn one_draft_override_applies_to_mtp() {
        let cfg = proposal_config(
            "mtp-head",
            &SpeculationOptions {
                max_draft: Some(1),
                ..SpeculationOptions::default()
            },
        );
        assert!(cfg.starts_with("max_draft=1,"));
        assert!(!cfg.contains("confidence="));
    }
}

#[derive(Default)]
pub struct Speculation {
    reg: Mutex<Option<linkw::StrategyRegisterMsg>>,
    grant: Mutex<Option<SpeculationGrant>>,
}

impl Speculation {
    pub fn register(
        this: &Arc<Speculation>,
        host: &mut EngineHost,
        strategy: &str,
        opts: &crate::SpeculationOptions,
    ) -> Result<(), DaemonError> {
        let (id, draft_path) = match strategy.split_once(':') {
            Some(("draft-model", path)) => ("draft-model", Some(path.to_string())),
            Some(("dflash", path)) => ("dflash", Some(path.to_string())),
            Some(("dspark", path)) => ("dspark", Some(path.to_string())),
            Some(("eagle3", path)) => ("eagle3", Some(path.to_string())),
            _ if strategy == "prompt-lookup" => ("prompt-lookup", None),
            _ if strategy == "mtp-head" => ("mtp-head", None),
            _ => {
                return Err(DaemonError::Config(
                    "unknown speculation strategy (prompt-lookup | mtp-head | dflash / dspark / eagle3 / draft-model followed by ':' and a model id or .base path)",
                ))
            }
        };
        let draft_identity = match draft_path.as_ref() {
            Some(p) => Some(draft_weights_identity(p)?),
            None => None,
        };
        let impl_hash = {
            let mut h = [0u8; 32];
            let tag = if id == "dflash" {
                "dflash/1"
            } else if id == "dspark" {
                "dspark/1"
            } else if id == "eagle3" {
                "eagle3/1"
            } else if draft_path.is_some() {
                "draft-model/1"
            } else if id == "mtp-head" {
                "mtp-head/1"
            } else {
                "prompt-lookup/1"
            };
            h[..8].copy_from_slice(&superfluid_fingerprint::fnv1a64(tag.as_bytes(), 0xCBF2_9CE4_8422_2325).to_le_bytes());
            h
        };
        let proposal_cfg = proposal_config(id, opts);
        let config_hash = {
            let mut h = [0u8; 32];
            let cfg = match (&draft_path, &draft_identity) {
                (Some(p), Some((fnv, size))) => {
                    format!("draft={p},weights={fnv:016x},bytes={size},{proposal_cfg}")
                }
                _ => format!("n_max=4,{proposal_cfg}"),
            };
            h[..8].copy_from_slice(&superfluid_fingerprint::fnv1a64(cfg.as_bytes(), 0xCBF2_9CE4_8422_2325).to_le_bytes());
            h
        };
        let artifacts = match (&draft_path, &draft_identity) {
            (Some(path), Some((fnv, size))) => {
                let mut content_hash = [0u8; 32];
                content_hash[..8].copy_from_slice(&fnv.to_le_bytes());
                vec![linkw::ArtifactMsg {
                    role: superfluid_abi::artifact_role::DRAFT_WEIGHTS,
                    content_hash,
                    byte_size: *size,
                    load_path: path.clone(),
                }]
            }
            _ => vec![],
        };
        let reg = linkw::StrategyRegisterMsg {
            strategy_id: id.to_string(),
            impl_version: "1.0.0".into(),
            impl_hash,
            config_hash,
            artifacts,
            taps: vec![],
            capabilities: vec![linkw::CapabilityReqMsg {
                kind_id: superfluid_abi::strategy_cap::PROPOSAL_LINEAR,
                params: proposal_cfg.into_bytes(),
            }],
            target_archs: vec![],
            kernel_caps_required: 0,
            est_state_bytes: 0,
            claimed_exactness: superfluid_abi::exactness::SEED_PATH_INVARIANT,
            rng_contract_version: 1,
        };
        *this.reg.lock().expect("spec reg") = Some(reg);
        this.reregister(host)
    }

    pub fn reregister(&self, host: &mut EngineHost) -> Result<(), DaemonError> {
        let Some(reg) = self.reg.lock().expect("spec reg").clone() else {
            return Ok(());
        };
        let ok = host.client().register_strategy(reg.clone()).map_err(|e| {
            DaemonError::Generation(format!("speculation strategy {} refused: {e}", reg.strategy_id))
        })?;
        let best = ok
            .certificates
            .iter()
            .filter(|c| c.sampling_modes & (superfluid_abi::cert_mode::GREEDY | superfluid_abi::cert_mode::GUMBEL) != 0)
            .max_by_key(|c| c.exactness);
        let identity = {
            let mut h = superfluid_fingerprint::RecordHasher::new();
            h.field(1, reg.strategy_id.as_bytes())
                .field(2, &reg.impl_hash)
                .field(3, &reg.config_hash);
            h.finish()
        };
        let modes = ok.certificates.iter().fold(0u8, |m, c| m | c.sampling_modes);
        *self.grant.lock().expect("spec grant") = Some(SpeculationGrant {
            strategy_id: reg.strategy_id.clone(),
            slot: ok.strategy_slot,
            class: best.map(|c| c.exactness).unwrap_or(superfluid_abi::exactness::APPROXIMATE),
            cert_id: best.map(|c| c.cert_id).unwrap_or(0),
            identity,
            modes,
        });
        Ok(())
    }

    pub fn grant(&self) -> Option<SpeculationGrant> {
        self.grant.lock().expect("spec grant").clone()
    }

    pub fn slot_for(&self, sampled: bool) -> u32 {
        let mode = if sampled {
            superfluid_abi::cert_mode::GUMBEL
        } else {
            superfluid_abi::cert_mode::GREEDY
        };
        self.grant
            .lock()
            .expect("spec grant")
            .as_ref()
            .filter(|g| g.modes & mode != 0)
            .map(|g| g.slot)
            .unwrap_or(0)
    }

    pub fn identity_for_fingerprint(&self) -> Option<u64> {
        self.grant
            .lock()
            .expect("spec grant")
            .as_ref()
            .filter(|g| g.class < superfluid_abi::exactness::SEED_PATH_INVARIANT)
            .map(|g| g.identity)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParkRefused {
    SpaceUnservable,
    Failed(String),
}

impl std::fmt::Display for ParkRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParkRefused::SpaceUnservable => {
                write!(f, "a state space has no ops (hybrid GDN before P2.5); park refused")
            }
            ParkRefused::Failed(m) => f.write_str(m),
        }
    }
}

impl From<String> for ParkRefused {
    fn from(m: String) -> Self {
        ParkRefused::Failed(m)
    }
}

impl From<&str> for ParkRefused {
    fn from(m: &str) -> Self {
        ParkRefused::Failed(m.to_string())
    }
}

enum ParkTask {
    Write(Box<ParkJob>),
    Remove { dir: PathBuf, session: u64 },
}

struct ParkJob {
    dir: PathBuf,
    session: u64,
    covered: u64,
    digest: u64,
    encoding: u8,
    sealed: Vec<u8>,
    spaces: Option<Vec<park::ParkedSpace>>,
}

pub struct Scheduler {
    host: EngineHost,
    store: Arc<Mutex<SessionStore>>,
    codec: Arc<dyn TextCodec + Send + Sync>,
    cancels: Arc<crate::CancelSet>,
    active: Arc<Mutex<HashSet<u64>>>,
    stats: Arc<SchedStats>,
    rx: Receiver<SchedCmd>,
    max_lanes: usize,
    configured_lanes: usize,
    retired: Arc<Mutex<Option<String>>>,
    park_dir: Option<PathBuf>,
    pressure_high_pct: u8,
    pressure_low_pct: u8,
    park_lossy: bool,
    trace_scope: crate::otlp::TraceScope,
    park_tx: Option<std::sync::mpsc::SyncSender<ParkTask>>,
    ephemeral_parked: std::collections::HashSet<u64>,
    park_removals_pending: Vec<u64>,
    park_writer: Option<std::thread::JoinHandle<()>>,
    bus: Arc<SessionBus>,

    plan_seq: u64,
    next_lane: u64,
    lanes: Vec<Lane>,
    queues: [ClassQueue; 4],
    class_lanes: [usize; 4],
    tick_decode_budget: u32,
    tick_target_secs: f64,
    tick_shape: TickShape,
    prefill_budget: u32,
    prefill_budget_fixed: bool,
    decode_round_secs: Option<f64>,
    prefill_secs_per_token: Option<f64>,
    just_retired: bool,
    prefill_secs_per_token_bulk: Option<f64>,
    prefill_secs_per_token_last: Option<f64>,
    prefill_rate_samples: u32,
    /// The engine refused to fork a lane's sequence: no checkpoint where a
    /// shared prefix ends is kept, and no request is held for one.
    prefix_checkpoints_off: bool,
    /// The turn checkpoints kept, oldest first, by the length and digest of the tokens each
    /// covers; a turn's checkpoint is dropped once its next turn keeps one.
    turn_checkpoints: std::collections::VecDeque<(u64, [u8; 32])>,
    tick_decode_budget_fixed: bool,
    starvation_ticks: u32,
    background_tick_divisor: Arc<std::sync::atomic::AtomicU32>,
    pressure_source: Option<Box<dyn crate::pressure::PressureSource + Send>>,
    tick_count: u64,
    speculation: Arc<Speculation>,
    tool_lease_ms: u64,
    respawn_streak: u32,
    staged_unread: u32,
    deferred_sync: VecDeque<(u8, u64, SchedCmd)>,
    admit_retry: bool,
    pins: crate::pins::Pins,
}

impl Scheduler {
    fn respawn_worker(&mut self) -> Result<(), DaemonError> {
        self.host.respawn()?;
        self.stats.kv_bits.store(self.host.kv_bits() as u64, Ordering::Relaxed);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        host: EngineHost,
        store: Arc<Mutex<SessionStore>>,
        codec: Arc<dyn TextCodec + Send + Sync>,
        cancels: Arc<crate::CancelSet>,
        active: Arc<Mutex<HashSet<u64>>>,
        stats: Arc<SchedStats>,
        background_tick_divisor: Arc<std::sync::atomic::AtomicU32>,
        speculation: Arc<Speculation>,
        retired: Arc<Mutex<Option<String>>>,
        mut opts: crate::DaemonOptions,
    ) -> SchedulerHandle {
        let (tx, rx) = std::sync::mpsc::channel();
        let join = std::thread::Builder::new()
            .name("superfluid-scheduler".into())
            .spawn(move || {
                let bus = store.lock().expect("store").bus();
                let host_max_stream = host.max_stream_tokens();
                let mut host = host;
                let max_lanes = lane_cap(
                    opts.max_lanes,
                    host.max_seqs(),
                    ring_slots(&mut host, superfluid_shm::TOKEN_RING_OUT),
                );
                let park_refused = host.park_refused();
                let lossy_allowed = host.lossy_park_allowed();
                let tick_shape = if host.round_granular() { ROUND_TICKS } else { NATIVE_TICKS };
                let (park_tx, park_writer) = if opts.park_dir.is_some() && park_refused.is_none() {
                    let (tx, rx) = std::sync::mpsc::sync_channel::<ParkTask>(4);
                    let park_budget = opts.park_budget_bytes;
                    let writer = std::thread::Builder::new()
                        .name("superfluid-park-writer".into())
                        .spawn(move || {
                            while let Ok(task) = rx.recv() {
                                let job = match task {
                                    ParkTask::Write(job) => job,
                                    ParkTask::Remove { dir, session } => {
                                        park::remove(&dir, session);
                                        continue;
                                    }
                                };
                                let res = match &job.spaces {
                                    Some(spaces) => park::write_v3(
                                        &job.dir,
                                        job.session,
                                        job.covered,
                                        job.digest,
                                        spaces,
                                    ),
                                    None => park::write(
                                        &job.dir,
                                        job.session,
                                        job.covered,
                                        job.digest,
                                        job.encoding,
                                        &job.sealed,
                                    ),
                                };
                                if let Err(e) = res {
                                    eprintln!(
                                        "[superfluid] park write failed (session {}): {e}",
                                        job.session
                                    );
                                }
                                if park_budget > 0 {
                                    let freed = park::sweep(&job.dir, park_budget);
                                    if freed > 0 {
                                        tracing::debug!(
                                            freed_bytes = freed,
                                            budget_bytes = park_budget,
                                            "park retention sweep"
                                        );
                                    }
                                }
                            }
                        })
                        .expect("spawn park writer");
                    (Some(tx), Some(writer))
                } else {
                    (None, None)
                };
                Scheduler {
                    host,
                    store,
                    codec,
                    cancels,
                    active,
                    stats,
                    rx,
                    max_lanes,
                    configured_lanes: opts.max_lanes,
                    retired,
                    park_dir: match park_refused {
                        Some(why) if opts.park_dir.is_some() => {
                            eprintln!("[superfluid] parking is off for this model: {why}");
                            None
                        }
                        _ => opts.park_dir,
                    },
                    pressure_high_pct: opts.pressure_high_pct,
                    pressure_low_pct: opts.pressure_low_pct,
                    park_lossy: opts.park_lossy && lossy_allowed,
                    trace_scope: opts.trace_scope.clone(),
                    park_tx,
                    park_writer,
                    ephemeral_parked: std::collections::HashSet::new(),
                    park_removals_pending: Vec::new(),
                    bus,
                    plan_seq: 0,
                    next_lane: 1,
                    lanes: Vec::new(),
                    queues: Default::default(),
                    class_lanes: opts.class_lanes,
                    tick_target_secs: if opts.tick_target_ms == 0 {
                        TARGET_TICK_TOTAL_SECS
                    } else {
                        opts.tick_target_ms as f64 / 1000.0
                    },
                    tick_shape,
                    tick_decode_budget: if opts.tick_decode_budget == 0 {
                        (max_lanes as u32).saturating_mul(TICK_DECODE as u32)
                    } else {
                        opts.tick_decode_budget
                    },
                    prefill_budget: if opts.prefill_budget == 0 {
                        1024
                    } else {
                        if opts.prefill_budget > PREFILL_BUDGET {
                            eprintln!(
                                "superfluid: --prefill-budget {} is above the {PREFILL_BUDGET}-token per-tick \
                                 ceiling (the media run-size limit); using {PREFILL_BUDGET}",
                                opts.prefill_budget
                            );
                        }
                        opts.prefill_budget.clamp(64, PREFILL_BUDGET)
                    },
                    prefill_budget_fixed: opts.prefill_budget != 0,
                    decode_round_secs: None,
                    prefill_secs_per_token: None,
                    just_retired: false,
                    prefill_secs_per_token_bulk: None,
                    prefill_secs_per_token_last: None,
                    prefill_rate_samples: 0,
                    prefix_checkpoints_off: false,
                    turn_checkpoints: std::collections::VecDeque::new(),
                    tick_decode_budget_fixed: opts.tick_decode_budget != 0,
                    starvation_ticks: opts.agent_starvation_ticks.max(1),
                    background_tick_divisor,
                    pressure_source: opts.pressure_source.take_source(),
                    tick_count: 0,
                    speculation,
                    tool_lease_ms: opts.tool_lease_ms,
                    respawn_streak: 0,
                    staged_unread: 0,
                    deferred_sync: VecDeque::new(),
                    admit_retry: false,
                    pins: crate::pins::Pins::new(
                        opts.pin_budget_pct,
                        host_max_stream * max_lanes as u64,
                    ),
                }
                .run();
            })
            .expect("spawn scheduler");
        SchedulerHandle {
            tx,
            join: std::sync::Mutex::new(Some(join)),
        }
    }

    fn run(mut self) {
        self.run_loop();
        drop(self.park_tx.take());
        if let Some(w) = self.park_writer.take() {
            let _ = w.join();
        }
        if let Some(dir) = self.park_dir.clone() {
            let pending = std::mem::take(&mut self.park_removals_pending);
            let tracked = std::mem::take(&mut self.ephemeral_parked);
            for session in pending.into_iter().chain(tracked) {
                park::remove(&dir, session);
            }
        }
    }

    fn run_loop(&mut self) {
        loop {
            if self.lanes.is_empty() && self.queued() == 0 && self.deferred_sync.is_empty() {
                self.retry_park_removals();
                let retry = (!self.park_removals_pending.is_empty())
                    .then(|| std::time::Duration::from_millis(50));
                let wait = match (self.pins.until_next_deadline(), retry) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                let cmd = match wait {
                    None => self.rx.recv().map_err(|_| ()),
                    Some(wait) => match self.rx.recv_timeout(wait) {
                        Ok(cmd) => Ok(cmd),
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            self.pins.sweep(&mut self.host, &self.stats);
                            continue;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(()),
                    },
                };
                match cmd {
                    Ok(SchedCmd::Shutdown) | Err(()) => return,
                    Ok(cmd) => self.dispatch(cmd),
                }
            }
            // Lowered before what is waiting is taken in: anything that
            // arrives after raises it for the tick about to run.
            self.host.clear_yield();
            // A request that arrives between the look and the store raised
            // the word itself: looked at again, it is raised again.
            let slot = self.host.yield_slot();
            let latency = self.latency_sensitive();
            crate::runtime::set_latency(&slot, latency);
            if !latency && self.cancels.latency_recent(LATENCY_HOLD) {
                crate::runtime::set_latency(&slot, true);
            }
            loop {
                match self.rx.try_recv() {
                    Ok(SchedCmd::Shutdown) => return,
                    Ok(cmd) => self.dispatch(cmd),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return,
                }
            }
            if !self.host.is_alive() {
                if self.respawn_streak < MAX_WORKER_RESPAWNS {
                    self.respawn_streak += 1;
                    if self.respawn_worker().is_ok() {
                        let _ = self.speculation.reregister(&mut self.host);
                        self.after_respawn();
                    }
                }
                if !self.host.is_alive() {
                    if self.respawn_streak >= MAX_WORKER_RESPAWNS {
                        self.give_up_on_worker();
                    }
                    let why = self
                        .retired
                        .lock()
                        .expect("retired")
                        .clone()
                        .unwrap_or_else(|| "engine worker unavailable (respawn failed)".into());
                    let orphaned: Vec<GenerateJob> =
                        self.queues.iter_mut().flat_map(|q| q.jobs.drain(..)).collect();
                    for job in orphaned {
                        self.finish_failed(&job, &why);
                    }
                    for (_, _, cmd) in std::mem::take(&mut self.deferred_sync) {
                        self.dispatch(cmd);
                    }
                    continue;
                }
            }
            self.pins.sweep(&mut self.host, &self.stats);
            let tick_span = if self.lanes.is_empty() && self.queued() == 0 {
                tracing::Span::none()
            } else {
                tracing::debug_span!(
                    parent: None,
                    "tick",
                    tick_seq = tracing::field::Empty,
                    no_tick = tracing::field::Empty,
                    lanes = tracing::field::Empty,
                    queued = self.queued(),
                    model = self.trace_scope.model.as_str(),
                    store_ns = self.trace_scope.ns
                )
            };
            let _tick = tick_span.enter();
            self.poll_os_pressure();
            self.preempt_for_queued();
            self.admit_from_queue();
            self.run_deferred_syncs();
            let ticks_before = self.tick_count;
            let tick_result = self.tick_once(&tick_span);
            if !tick_span.is_disabled() {
                if self.tick_count != ticks_before {
                    tick_span.record("tick_seq", self.tick_count);
                } else {
                    tick_span.record("no_tick", true);
                }
            }
            if tick_result.is_ok() {
                self.respawn_streak = 0;
            }
            if let Err(e) = tick_result {
                let admitted: Vec<u64> = self
                    .lanes
                    .iter()
                    .filter(|l| l.admitted)
                    .map(|l| l.tag)
                    .collect();
                if !admitted.is_empty() {
                    let mut p = linkw::TickPlanMsg {
                        plan_seq: {
                            self.plan_seq += 1;
                            self.plan_seq
                        },
                        flags: 0,
                        prefill_token_budget: PREFILL_BUDGET,
                        max_decode_lanes: self.max_lanes as u32,
                        admits: vec![],
                        commits: vec![],
                        prefills: vec![],
                        decodes: vec![],
                        retires: vec![],
                        shed_policy: linkw::ShedPolicyMsg {
                            victim_lanes: vec![],
                            evictable_cache_classes: u64::MAX,
                            protected_quota_bytes: 0,
                            max_evict_bytes: u64::MAX,
                        },
                    };
                    for tag in admitted {
                        p.retires.push(linkw::LaneRetireMsg {
                            lane_tag: tag,
                            publish_to_cache: true,
                        });
                    }
                    self.stats.set_tick_in_flight(TickInFlight {
                        since_ms: uptime_ms().max(1),
                        retires: p.retires.len() as u64,
                        target_ms: self.stats.tick_target_live_ms.load(Ordering::Relaxed),
                        ..Default::default()
                    });
                    let _ = self.host.client().tick(p);
                    self.stats.clear_tick_in_flight();
                }
                let msg = format!("scheduler tick failed: {e}");
                let drained: Vec<u64> = self.lanes.iter().map(|l| l.session).collect();
                for lane in self.lanes.drain(..) {
                    self.active.lock().expect("active set").remove(&lane.session);
                    let _ = lane.job.events.send(LaneEvent::Failed(msg.clone()));
                }
                for session in drained {
                    self.drop_ephemeral_artifact(session);
                }
                self.publish_occupancy();
                if worker_is_gone(&e) && self.host.is_process() {
                    let mut fresh = false;
                    if self.respawn_streak < MAX_WORKER_RESPAWNS {
                        self.respawn_streak += 1;
                        self.stats.worker_respawns.fetch_add(1, Ordering::Relaxed);
                        match self.respawn_worker().and_then(|()| self.speculation.reregister(&mut self.host)) {
                            Ok(()) => {
                                eprintln!("[superfluid] engine worker died ({e}); respawned a fresh one");
                                self.after_respawn();
                                fresh = true;
                            }
                            Err(err) => eprintln!(
                                "[superfluid] engine worker died ({e}); respawn FAILED: {err}"
                            ),
                        }
                    }
                    if !fresh && self.respawn_streak >= MAX_WORKER_RESPAWNS {
                        self.give_up_on_worker();
                    }
                }
            }
            self.pins.sweep(&mut self.host, &self.stats);
        }
    }

    fn dispatch(&mut self, cmd: SchedCmd) {
        match cmd {
            SchedCmd::Submit(job) => self.enqueue(*job),
            SchedCmd::Sync(req, reply) => self.run_sync(req, reply),
            SchedCmd::SyncStream(req, segs, reply) => self.run_sync_stream(req, segs, reply),
            SchedCmd::Pin(req) => self.pins.handle(&mut self.host, &self.stats, self.tick_count, req),
            SchedCmd::PinHeld(keys, reply) => {
                let _ = reply.send(self.pins.held_max(&self.host, &keys));
            }
            SchedCmd::PinUnanchor(session) => self.pins.unanchor(session),
            SchedCmd::SyncAs(class, inner) => {
                self.deferred_sync.push_back((class, self.tick_count, *inner));
            }
            SchedCmd::Shutdown => {}
        }
    }

    fn sync_must_wait(&self, class: u8) -> bool {
        self.lanes.iter().any(|l| l.job.class < class)
            || self.queues.iter().take(class as usize).any(|q| !q.jobs.is_empty())
    }

    fn run_deferred_syncs(&mut self) {
        let period = self.starvation_ticks as u64;
        let stuck = self.lanes.is_empty() && !self.admit_retry;
        let starved = |queued: u64| self.tick_count.saturating_sub(queued) >= period;
        let pick = self
            .deferred_sync
            .iter()
            .enumerate()
            .filter(|(_, (class, queued, _))| stuck || !self.sync_must_wait(*class) || starved(*queued))
            .min_by_key(|&(i, (class, queued, _))| {
                let aged = starved(*queued);
                (!aged, if aged { 0 } else { self.sync_rank(*class) }, i)
            })
            .map(|(i, _)| i);
        if let Some((_, _, cmd)) = pick.and_then(|i| self.deferred_sync.remove(i)) {
            self.dispatch(cmd);
        }
    }

    fn sync_rank(&self, class: u8) -> usize {
        class as usize
    }

    fn run_sync(&mut self, req: linkw::StateSyncReqMsg, reply: Sender<Result<linkw::StateSyncOkMsg, String>>) {
        let res = if self.host.is_alive() {
            self.host.client().state_sync(req).map_err(|e| e.to_string())
        } else {
            Err("engine worker unavailable".into())
        };
        let _ = reply.send(res);
    }

    fn run_sync_stream(
        &mut self,
        req: linkw::StateSyncReqMsg,
        segs: Sender<linkw::TranscribeSegmentMsg>,
        reply: Sender<Result<linkw::StateSyncOkMsg, String>>,
    ) {
        let res = if self.host.is_alive() {
            self.host
                .client()
                .state_sync_streaming(req, &mut |seg| {
                    segs.send(seg).is_ok()
                })
                .map_err(|e| e.to_string())
        } else {
            Err("engine worker unavailable".into())
        };
        let _ = reply.send(res);
    }

    fn queued(&self) -> usize {
        self.queues.iter().map(|q| q.jobs.len()).sum()
    }

    fn enqueue(&mut self, mut job: GenerateJob) {
        let c = (job.class as usize).min(3);
        job.queued_at = self.tick_count;
        self.queues[c].jobs.push_back(job);
    }

    fn engine_exclusive(&self, job: &GenerateJob) -> bool {
        job.batch_invariant || (!job.media.is_empty() && self.host.has_recurrent_space())
    }

    fn strategy_slot_for(speculation: &Speculation, job: &GenerateJob) -> u32 {
        if job.grammar_handle != 0 || job.extras.logit_bias_handle != 0 || job.extras.want_logprobs {
            0
        } else {
            speculation.slot_for(job.params.temperature != 0.0)
        }
    }

    fn decode_replay_for(strategy_slot: u32, produced_before: u32) -> u32 {
        if strategy_slot == 0 { produced_before } else { 0 }
    }

    fn prefill_end_of(speculation: &Speculation, lane: &Lane) -> u64 {
        let replay = Self::decode_replay_for(Self::strategy_slot_for(speculation, &lane.job), lane.produced_before);
        (lane.job.stream.len() as u64)
            .saturating_sub(replay as u64)
            .max(lane.prefilled)
    }

    fn pop_admissible(&mut self) -> Option<GenerateJob> {
        if self.lanes.iter().any(|l| self.engine_exclusive(&l.job)) {
            return None;
        }
        let mut per_class = [0usize; 4];
        for l in &self.lanes {
            per_class[(l.job.class as usize).min(3)] += 1;
        }
        let now = self.tick_count;
        let period = self.starvation_ticks as u64;
        let mut best: Option<(u8, usize)> = None;
        let mut excl_waiting: Option<u8> = None;
        for (c, q) in self.queues.iter().enumerate() {
            let Some(head) = q.jobs.front() else {
                continue;
            };
            let cap = if self.class_lanes[c] == 0 {
                self.max_lanes
            } else {
                self.class_lanes[c]
            };
            if per_class[c] >= cap {
                continue;
            }
            let waited = now.saturating_sub(head.queued_at);
            let aged = (waited / period).min(c as u64) as u8;
            let eff = c as u8 - aged;
            if self.engine_exclusive(head) && !self.lanes.is_empty() {
                if excl_waiting.is_none_or(|b| eff < b) {
                    excl_waiting = Some(eff);
                }
                continue;
            }
            if best.is_none_or(|(b, _)| eff < b) {
                best = Some((eff, c));
            }
        }
        if let (Some(e), Some((b, _))) = (excl_waiting, best) {
            if e <= b {
                return None;
            }
        }
        let (_, c) = best?;
        self.queues[c].jobs.pop_front()
    }

    /// Whether `job` should wait for a lane that is prefilling a long prefix
    /// it shares and will publish that prefix to the cache shortly.
    fn awaits_a_shared_prefix(&self, job: &GenerateJob) -> bool {
        if job.carry.is_some()
            || job.class <= crate::qos::INLINE_COMPLETION
            || !job.media.is_empty()
            || self.tick_count.saturating_sub(job.queued_at) >= SHARED_PREFIX_MAX_HOLD_TICKS
            || self.host.has_unservable_space()
            || !self.host.cache_serves_all_spaces()
        {
            return false;
        }
        let Some(page) = self.host.page_size().filter(|&p| p > 0) else {
            return false;
        };
        if self.host.needs_prefix_checkpoints() {
            if self.prefix_checkpoints_off {
                return false;
            }
            // A retired lane's entry covers its reply and seeds nothing that
            // only shares its prompt: wait instead for the state kept where
            // the shared part ends, while a lane can still stop there.
            return self.lanes.iter().any(|l| {
                if l.preempted || l.job.completion || !l.job.media.is_empty() {
                    return false;
                }
                let shared = job.stream.iter().zip(&l.all_tokens).take_while(|(a, b)| a == b).count() as u64;
                let total = Self::prefill_end_of(&self.speculation, l);
                shared >= SHARED_PREFIX_MIN_TOKENS && shared + 1 < total && !Self::kept_for(l, shared) && l.prefilled < shared
            });
        }
        self.lanes.iter().any(|l| {
            if l.preempted
                || l.parked_lossy
                || l.job.completion
                || !l.job.media.is_empty()
                || l.job.max_tokens.saturating_sub(l.produced) > SHARED_PREFIX_TAIL_TOKENS
            {
                return false;
            }
            let shared = job.stream.iter().zip(&l.all_tokens).take_while(|(a, b)| a == b).count() as u64;
            let shared = shared / page * page;
            shared >= SHARED_PREFIX_MIN_TOKENS && shared > l.warm_prefix
        })
    }

    /// On a runtime whose state cannot be cut back, a prefilling lane stops
    /// where the requests queued behind it stop sharing its prompt (the
    /// shortest such share of at least SHARED_PREFIX_MIN_TOKENS), so its
    /// state there can be kept and seed them all; and where a chat's input
    /// ends, before the generation prompt. The next turn renders that input
    /// again but not the generation prompt as it was (a template drops the
    /// reasoning opener from an assistant turn in history), so the state the
    /// executor keeps where the prompt ends is no prefix of it. Of the points
    /// past what is prefilled, the nearest comes first; the next is planned
    /// once the prefill has passed it.
    fn plan_prefix_checkpoints(&mut self) {
        if !self.host.needs_prefix_checkpoints() || self.prefix_checkpoints_off {
            for l in self.lanes.iter_mut() {
                l.checkpoint_at = 0;
            }
            return;
        }
        // A prefill that stopped where the input ends without planning to
        // (a shed chunk planned again from there, or a budget that ran out
        // there) keeps the state there before it goes on.
        let mut landed = false;
        for l in self.lanes.iter_mut() {
            if Self::turn_point(&self.speculation, l) == Some(l.prefilled) {
                (l.checkpoint_at, l.checkpoint_due, landed) = (l.prefilled, true, true);
            }
        }
        if landed {
            self.publish_prefix_checkpoints();
        }
        for i in 0..self.lanes.len() {
            let l = &self.lanes[i];
            let total = Self::prefill_end_of(&self.speculation, l);
            let mut at = 0u64;
            if Self::checkpoint_eligible(l, total) {
                for job in self.queues.iter().flat_map(|q| q.jobs.iter()) {
                    let shared = job.stream.iter().zip(&l.all_tokens).take_while(|(a, b)| a == b).count() as u64;
                    let useful = shared >= SHARED_PREFIX_MIN_TOKENS
                        && shared + 1 < total
                        && shared > l.prefilled
                        && !Self::kept_for(l, shared);
                    if useful && (at == 0 || shared < at) {
                        at = shared;
                    }
                }
                if let Some(end) = Self::turn_point(&self.speculation, l).filter(|&end| end > l.prefilled) {
                    if at == 0 || end < at {
                        at = end;
                    }
                }
            }
            self.lanes[i].checkpoint_at = at;
        }
    }

    fn checkpoint_eligible(l: &Lane, total: u64) -> bool {
        !l.preempted
            && !l.job.completion
            && l.job.media.is_empty()
            && !matches!(l.phase, LanePhase::Finishing)
            && l.prefilled < total
    }

    /// Where this lane's state is kept for the chat's next turn: the end of the input,
    /// unless the lane started there or already prefilled past it.
    fn turn_point(speculation: &Speculation, l: &Lane) -> Option<u64> {
        let end = l.job.input_end;
        let total = Self::prefill_end_of(speculation, l);
        (Self::checkpoint_eligible(l, total)
            && !l.turn_kept
            && end >= TURN_CHECKPOINT_MIN_TOKENS
            && end + 1 < total
            && end >= l.prefilled
            && end != l.warm_prefix)
            .then_some(end)
    }

    /// Whether a state that seeds a request sharing `shared` tokens of this
    /// lane's prompt is in the cache already: the lane kept one, or started
    /// from one.
    fn kept_for(l: &Lane, shared: u64) -> bool {
        let serves = |at: u64| at >= SHARED_PREFIX_MIN_TOKENS && at <= shared;
        serves(l.checkpointed) || serves(l.warm_prefix)
    }

    /// Keeps a copy of each lane's state where this tick's prefill stopped
    /// for a checkpoint, in the prefix cache under the tokens it covers.
    fn publish_prefix_checkpoints(&mut self) {
        for i in 0..self.lanes.len() {
            if !std::mem::take(&mut self.lanes[i].checkpoint_due) {
                continue;
            }
            let (tag, at) = (self.lanes[i].tag, self.lanes[i].checkpoint_at);
            let tokens = self.lanes[i].all_tokens[..at as usize].to_vec();
            let turn = at == self.lanes[i].job.input_end;
            self.lanes[i].turn_kept |= turn;
            // A turn's checkpoint serves the chat's next turn alone, and goes
            // before a prefix requests share when the cache is short of room.
            match self.fork_into_cache(tag, &tokens, turn) {
                Ok(()) => {
                    self.lanes[i].checkpointed = at;
                    if turn {
                        self.supersede_turn_checkpoints(&tokens);
                    }
                }
                Err(CheckpointError::Refused(e)) => {
                    tracing::info!(lane = tag, "the engine forks no lane's sequence ({e}): no shared-prefix checkpoints");
                    self.prefix_checkpoints_off = true;
                }
                Err(CheckpointError::Failed(e)) => tracing::debug!(lane = tag, at, "prefix checkpoint not kept: {e}"),
            }
        }
    }

    /// Records the turn checkpoint just kept under `tokens`, and drops those
    /// it extends: earlier turns of the same chat.
    fn supersede_turn_checkpoints(&mut self, tokens: &[u32]) {
        let digest = superfluid_fingerprint::content_digest(tokens);
        let mut kept = std::collections::VecDeque::with_capacity(self.turn_checkpoints.len() + 1);
        for (len, d) in std::mem::take(&mut self.turn_checkpoints) {
            let n = len as usize;
            if n == tokens.len() && d == digest {
                continue;
            }
            if n < tokens.len() && superfluid_fingerprint::content_digest(&tokens[..n]) == d {
                self.unpublish(&tokens[..n]);
                continue;
            }
            kept.push_back((len, d));
        }
        kept.push_back((tokens.len() as u64, digest));
        while kept.len() > TURN_CHECKPOINTS_TRACKED {
            kept.pop_front();
        }
        self.turn_checkpoints = kept;
    }

    fn unpublish(&mut self, tokens: &[u32]) {
        let res = match self.stage(tokens) {
            Ok(span) => self.host.client().state_sync(linkw::StateSyncReqMsg::Unpublish { span }).map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        };
        if let Err(e) = res {
            tracing::debug!(len = tokens.len(), "superseded turn checkpoint not dropped: {e}");
        }
    }

    fn fork_into_cache(&mut self, tag: u64, tokens: &[u32], private: bool) -> Result<(), CheckpointError> {
        use CheckpointError::Failed;
        let seq = match self.host.client().state_sync(linkw::StateSyncReqMsg::LaneSequence { lane_tag: tag }) {
            Ok(linkw::StateSyncOkMsg::LaneSequence { sequence }) => sequence,
            Ok(_) => return Err(Failed("mismatched LaneSequence reply".into())),
            Err(e) => return Err(CheckpointError::of(e)),
        };
        let child = match self.host.client().state_sync(linkw::StateSyncReqMsg::Fork {
            parent_sequence: seq,
            flags: if private { superfluid_abi::fork_flags::PRIVATE } else { 0 },
        }) {
            Ok(linkw::StateSyncOkMsg::Fork { child_sequence }) => child_sequence,
            Ok(_) => return Err(Failed("mismatched Fork reply".into())),
            Err(e) => return Err(CheckpointError::of(e)),
        };
        let published = match self.stage(tokens) {
            Ok(span) => match self.host.client().state_sync(linkw::StateSyncReqMsg::PublishSequence { sequence: child, span }) {
                Ok(linkw::StateSyncOkMsg::PublishSequence) => Ok(()),
                Ok(_) => Err("mismatched PublishSequence reply".into()),
                Err(e) => Err(e.to_string()),
            },
            Err(e) => Err(e.to_string()),
        };
        if published.is_err() {
            let _ = self.host.client().state_sync(linkw::StateSyncReqMsg::FreeSequence { sequence: child });
        }
        published.map_err(Failed)
    }

    /// Whether a chat or inline completion is queued, being served, or arrived within
    /// LATENCY_HOLD: such a tick stays short.
    fn latency_sensitive(&self) -> bool {
        self.cancels.latency_recent(LATENCY_HOLD)
            || self.queues[..=crate::qos::INLINE_COMPLETION as usize].iter().any(|q| !q.jobs.is_empty())
            || self.lanes.iter().any(|l| l.job.class <= crate::qos::INLINE_COMPLETION && !l.preempted)
    }

    fn joiner_waits_on_this_tick(&self) -> bool {
        if self.max_lanes <= 1 {
            return false;
        }
        self.live_lanes() < self.max_lanes
            || self.queues[..=crate::qos::INLINE_COMPLETION as usize]
                .iter()
                .any(|q| !q.jobs.is_empty())
            || self.lanes.iter().any(|l| {
                l.job.class > crate::qos::INLINE_COMPLETION
                    && !l.preempted
                    && l.job.media.is_empty()
                    && !self.engine_exclusive(&l.job)
            })
    }

    /// Lanes holding a slot past this tick: a preempted lane retires in it,
    /// and the engine retires before it admits, so its slot is already free.
    fn live_lanes(&self) -> usize {
        self.lanes
            .iter()
            .filter(|l| !(l.preempted && matches!(l.phase, LanePhase::Finishing)))
            .count()
    }

    fn preempt_for_queued(&mut self) {
        if self.live_lanes() < self.max_lanes {
            return;
        }
        let Some(want) = self
            .queues
            .iter()
            .enumerate()
            .find(|(_, q)| !q.jobs.is_empty())
            .map(|(c, _)| c as u8)
        else {
            return;
        };
        if want > crate::qos::INLINE_COMPLETION {
            return;
        }
        // A decoding lane first; else one still prefilling, whose prefilled
        // pages go to the prefix cache on retire and seed it when it resumes.
        let victim_in = |phase_decoding: bool| {
            self.lanes
                .iter()
                .enumerate()
                .filter(|(_, l)| {
                    l.admitted
                        && !l.preempted
                        && !self.engine_exclusive(&l.job)
                        && l.job.class > want
                        && l.job.media.is_empty()
                        && if phase_decoding {
                            matches!(l.phase, LanePhase::Decoding)
                        } else {
                            matches!(l.phase, LanePhase::Prefilling)
                        }
                })
                .max_by_key(|(_, l)| (l.job.class, l.tag))
                .map(|(i, _)| i)
        };
        let victim = victim_in(true).or_else(|| victim_in(false));
        if let Some(i) = victim {
            self.preempt_lane(i);
        }
    }

    fn preempt_lane(&mut self, i: usize) {
        let lane = &mut self.lanes[i];
        lane.preempted = true;
        lane.phase = LanePhase::Finishing;
        lane.fin = finish::NONE;
        tracing::info!(
            session = lane.session,
            lane = lane.tag,
            class = lane.job.class,
            produced = lane.produced,
            "lane preempted"
        );
        self.stats.preemptions.fetch_add(1, Ordering::Relaxed);
    }

    fn poll_os_pressure(&mut self) {
        let Some(src) = self.pressure_source.as_mut() else {
            return;
        };
        match src.level() {
            crate::pressure::PressureLevel::Normal => {}
            crate::pressure::PressureLevel::Warning => {
                self.stats.os_pressure_events.fetch_add(1, Ordering::Relaxed);
                self.relief_round_now();
            }
            crate::pressure::PressureLevel::Critical => {
                self.stats.os_pressure_events.fetch_add(1, Ordering::Relaxed);
                self.relief_round_now();
                let victims: Vec<usize> = self
                    .lanes
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| {
                        l.admitted
                            && !l.preempted
                            && l.job.class == crate::qos::BACKGROUND_AGENT
                            && matches!(l.phase, LanePhase::Decoding)
                    })
                    .map(|(i, _)| i)
                    .collect();
                for i in victims {
                    self.preempt_lane(i);
                }
            }
        }
    }

    fn relief_round_now(&mut self) {
        // Memory is short on the machine: what the cache holds in host memory
        // goes first, whatever the pool holds.
        self.evict_cache(0, HOST_EXPORT_CLASS);
        if let Ok(pong) = self.host.client().ping() {
            let mem = pong.mem;
            let low = self.pressure_low_pct as u64;
            let (used, total) = (mem.pool_blocks_used, mem.pool_blocks_total);
            if total == 0 || used * 100 <= total * low {
                return;
            }
            let saved = self.pressure_high_pct;
            self.pressure_high_pct = 1;
            self.relieve_pressure(&mem, true);
            self.pressure_high_pct = saved;
        }
    }

    fn after_respawn(&mut self) {
        self.max_lanes = lane_cap(
            self.configured_lanes,
            self.host.max_seqs(),
            ring_slots(&mut self.host, superfluid_shm::TOKEN_RING_OUT),
        );
        self.pins.set_fallback_pool_tokens(self.host.max_stream_tokens() * self.max_lanes as u64);
        if let Some(why) = self.host.take_record_change() {
            self.retire(&why);
        }
    }

    // Respawning stops after a streak; without a retire nothing would load
    // the model again and every request would fail until a restart.
    fn give_up_on_worker(&mut self) {
        if self.retired.lock().expect("retired").is_none() {
            self.retire(&format!(
                "the engine worker died and {MAX_WORKER_RESPAWNS} respawns in a row did not keep one up"
            ));
        }
    }

    fn retire(&mut self, why: &str) {
        let msg = format!("{why}; the model loads again on its next request, so retry");
        eprintln!("[superfluid] {msg}");
        *self.retired.lock().expect("retired") = Some(msg.clone());
        let drained: Vec<u64> = self.lanes.iter().map(|l| l.session).collect();
        for lane in self.lanes.drain(..) {
            self.active.lock().expect("active set").remove(&lane.session);
            let _ = lane.job.events.send(LaneEvent::Failed(msg.clone()));
        }
        for session in drained {
            self.drop_ephemeral_artifact(session);
        }
        let queued: Vec<GenerateJob> = self.queues.iter_mut().flat_map(|q| q.jobs.drain(..)).collect();
        for job in queued {
            self.finish_failed(&job, &msg);
        }
    }

    fn stage(&mut self, tokens: &[u32]) -> Result<linkw::TokenRefMsg, superfluid_agent::AgentError> {
        self.staged_unread += 1;
        self.host.client().stage_prompt(tokens)
    }

    fn admit_from_queue(&mut self) {
        self.admit_retry = false;
        let retired = self.retired.lock().expect("retired").clone();
        if let Some(msg) = retired {
            let queued: Vec<GenerateJob> = self.queues.iter_mut().flat_map(|q| q.jobs.drain(..)).collect();
            for job in queued {
                self.finish_failed(&job, &msg);
            }
            return;
        }
        let mut held: Vec<GenerateJob> = Vec::new();
        if self.lanes.iter().all(|l| l.admitted) {
            self.staged_unread = 0;
        }
        let stage_slots = ring_slots(&mut self.host, superfluid_shm::TOKEN_RING_IN).unwrap_or(u32::MAX);
        while self.live_lanes() < self.max_lanes {
            // A staged prompt is read when its tick admits it; staging past
            // the ring's size would overwrite one still waiting, and fail the
            // tick. The rest wait a tick in their queue.
            if self.staged_unread > 0 && self.staged_unread.saturating_add(STAGES_PER_ADMISSION) > stage_slots {
                break;
            }
            let Some(job) = self.pop_admissible() else {
                break;
            };
            if job.stream.is_empty() {
                self.finish_failed(&job, "session has no content to generate from");
                continue;
            }
            if self.awaits_a_shared_prefix(&job) {
                held.push(job);
                continue;
            }
            if let Some(deadline) = job.deadline {
                if std::time::Instant::now() > deadline {
                    let _ = job.events.send(LaneEvent::Expired);
                    self.active.lock().expect("active set").remove(&job.session);
                    self.drop_ephemeral_artifact(job.session);
                    continue;
                }
            }
            let max_stream = self.host.max_stream_tokens();
            if job.stream.len() as u64 > max_stream {
                let msg = DaemonError::StreamTooLong {
                    len: job.stream.len() as u64,
                    max: max_stream,
                }
                .to_string();
                self.finish_failed(&job, &msg);
                continue;
            }
            let span = match self.stage(&job.stream) {
                Ok(s) => s,
                Err(e) => {
                    let transport = matches!(
                        e,
                        superfluid_agent::AgentError::Io(_)
                            | superfluid_agent::AgentError::Closed
                            | superfluid_agent::AgentError::Shm(_)
                    );
                    if transport && self.host.is_process() && self.respawn_streak < MAX_WORKER_RESPAWNS {
                        self.respawn_streak += 1;
                        eprintln!(
                            "[superfluid] engine worker gone at admission ({e}); respawning"
                        );
                        self.stats.worker_respawns.fetch_add(1, Ordering::Relaxed);
                        if self
                            .respawn_worker()
                            .and_then(|()| self.speculation.reregister(&mut self.host))
                            .is_ok()
                        {
                            let c = (job.class as usize).min(3);
                            self.queues[c].jobs.push_front(job);
                            self.after_respawn();
                            self.admit_retry = true;
                            break;
                        }
                    }
                    self.finish_failed(&job, &format!("stage failed: {e}"));
                    continue;
                }
            };
            let mut seed_handle = 0u64;
            let mut warm_prefix = 0u64;
            if self.host.has_unservable_space() {
                self.stats
                    .unservable_cold_admissions
                    .fetch_add(1, Ordering::Relaxed);
            } else if let Some(page) = self.host.page_size() {
                if self.host.cache_serves_all_spaces() && job.media.is_empty() {
                if let Ok(m) = self.host.client().match_prefix(ALL_SPACES, span) {
                    let align_down = |l: u64| l.checked_div(page).unwrap_or(0) * page;
                    let replay = Self::decode_replay_for(
                        Self::strategy_slot_for(&self.speculation, &job),
                        job.carry.as_ref().map(|c| c.produced_before).unwrap_or(0),
                    );
                    let stream_len = job.stream.len() as u64;
                    let aligned_cap =
                        align_down(stream_len.saturating_sub(1).min(stream_len.saturating_sub(replay as u64)));
                    let best = crate::pins::common_seed_lengths(&self.host, &m, aligned_cap)
                        .first()
                        .copied()
                        .unwrap_or(0);
                    if best >= page {
                        if let Ok(h) = self.host.client().seed_acquire(span, best, 0) {
                            seed_handle = h;
                            warm_prefix = best;
                        }
                    }
                }
                }
                (seed_handle, warm_prefix) =
                    self.resume_from_park(&job, span, page, seed_handle, warm_prefix);
            }
            if warm_prefix > 0 {
                self.stats
                    .warm_prefix_tokens
                    .fetch_add(warm_prefix, Ordering::Relaxed);
            } else {
                self.stats.cold_admissions.fetch_add(1, Ordering::Relaxed);
            }
            let tag = self.next_lane;
            self.next_lane += 1;
            let mut job = job;
            let mut media_ok = true;
            let mut bound: Vec<u64> = Vec::new();
            for (offset, path, _n) in job.media.iter().filter(|(o, _, _)| *o >= warm_prefix) {
                let path_s = path.to_string_lossy().into_owned();
                let enc = self.host.client().state_sync(linkw::StateSyncReqMsg::MediaEncode {
                    image_path: path_s.clone(),
                });
                let handle = match enc {
                    Ok(linkw::StateSyncOkMsg::MediaEncode { media_handle, .. }) => media_handle,
                    other => {
                        eprintln!("[superfluid] media encode failed for {path_s}: {other:?}");
                        media_ok = false;
                        break;
                    }
                };
                let bind = self.host.client().state_sync(linkw::StateSyncReqMsg::MediaBind {
                    lane_tag: tag,
                    media_handle: handle,
                    token_offset: *offset as u32,
                });
                if matches!(bind, Ok(linkw::StateSyncOkMsg::MediaBind)) {
                    bound.push(handle);
                } else {
                    let _ = self
                        .host
                        .client()
                        .state_sync(linkw::StateSyncReqMsg::MediaRelease { media_handle: handle });
                    media_ok = false;
                    break;
                }
            }
            if !media_ok {
                for handle in &bound {
                    let _ = self.host.client().state_sync(
                        linkw::StateSyncReqMsg::MediaRelease { media_handle: *handle },
                    );
                }
                if seed_handle != 0 {
                    let _ = self.host.client().seed_release(seed_handle);
                }
                self.finish_failed(&job, "media encode/bind failed (see daemon log)");
                continue;
            }
            let spec_before = job.carry.as_ref().map(|c| c.spec_before).unwrap_or_default();
            let (seg, utf8, disp_seg, disp_utf8, tool_buf, produced_before, warm_prefix_first) =
                match job.carry.take() {
                    Some(c) => (
                        c.seg,
                        c.utf8,
                        c.disp_seg,
                        c.disp_utf8,
                        c.tool_buf,
                        c.produced_before,
                        c.warm_prefix_first,
                    ),
                    None => {
                        let mk = || {
                            let mut seg = self.codec.channelizer();
                            if job.extras.open_channel != 0 {
                                seg.prime(job.extras.open_channel);
                            }
                            seg
                        };
                        (
                            mk(),
                            Utf8Stream::default(),
                            mk(),
                            Utf8Stream::default(),
                            String::new(),
                            0,
                            None,
                        )
                    }
                };
            let all_tokens = job.stream.clone();
            self.lanes.push(Lane {
                tag,
                session: job.session,
                job,
                phase: LanePhase::Prefilling,
                prefilled: warm_prefix,
                produced: 0,
                fin: finish::NONE,
                warm_prefix,
                utf8,
                seg,
                disp_seg,
                disp_utf8,
                frame_fed: 0,
                emit_dead: false,
                tool_buf,
                admitted: false,
                seed_handle,
                span,
                all_tokens,
                parked: false,
                parked_lossy: false,
                preempted: false,
                starved_ticks: 0,
                prefill_starved_ticks: 0,
                checkpoint_at: 0,
                checkpoint_due: false,
                checkpointed: 0,
                turn_kept: false,
                produced_before,
                disp_tokens: produced_before,
                spec: spec_before,
                warm_prefix_first,
                admitted_at: std::time::Instant::now(),
                ttft_observed: false,
                ttft_frame_ms: None,
                fault_code: None,
            });
        }
        for job in held.into_iter().rev() {
            let c = (job.class as usize).min(3);
            self.queues[c].jobs.push_front(job);
        }
    }

    fn resume_from_park(
        &mut self,
        job: &GenerateJob,
        span: linkw::TokenRefMsg,
        page: u64,
        seed_handle: u64,
        warm_prefix: u64,
    ) -> (u64, u64) {
        let Some(dir) = self.park_dir.clone() else {
            return (seed_handle, warm_prefix);
        };
        let Some(p) = park::read(&dir, job.session) else {
            return (seed_handle, warm_prefix);
        };
        if p.multi_space {
            if !self.host.multi_space() && self.host.parkable_spaces().len() != p.spaces.len() {
                return (seed_handle, warm_prefix);
            }
            return match self.resume_adopt(job, &p) {
                Ok((h, covered)) => {
                    if seed_handle != 0 {
                        let _ = self.host.client().seed_release(seed_handle);
                    }
                    self.stats.resumes.fetch_add(1, Ordering::Relaxed);
                    (h, covered)
                }
                Err(msg) => {
                    eprintln!("[superfluid] park resume fell back cold (session {}): {msg}", job.session);
                    (seed_handle, warm_prefix)
                }
            };
        }
        if self.host.multi_space() || !job.media.is_empty() {
            return (seed_handle, warm_prefix);
        }
        let align_down = |l: u64| l.checked_div(page).unwrap_or(0) * page;
        let aligned_cap = align_down(job.stream.len() as u64 - 1);
        let usable = p.covered.min(aligned_cap);
        if p.covered % page != 0
            || usable <= warm_prefix
            || p.covered as usize > job.stream.len()
            || park::stream_digest(&job.stream[..p.covered as usize]) != p.stream_digest
        {
            return (seed_handle, warm_prefix);
        }
        let covered_span = match self.stage(&job.stream[..p.covered as usize]) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "[superfluid] park resume fell back cold (session {}): stage: {e}",
                    job.session
                );
                return (seed_handle, warm_prefix);
            }
        };
        if let Err(msg) = self.restore_to_cache(job.session, &p, covered_span) {
            eprintln!(
                "[superfluid] park resume fell back cold (session {}): {msg}",
                job.session
            );
            return (seed_handle, warm_prefix);
        }
        match self.host.client().seed_acquire(span, usable, 0) {
            Ok(h) => {
                if seed_handle != 0 {
                    let _ = self.host.client().seed_release(seed_handle);
                }
                self.stats.resumes.fetch_add(1, Ordering::Relaxed);
                (h, usable)
            }
            Err(e) => {
                eprintln!(
                    "[superfluid] park resume restored but could not seed (session {}): {e}",
                    job.session
                );
                (seed_handle, warm_prefix)
            }
        }
    }

    fn restore_to_cache(
        &mut self,
        session: u64,
        p: &park::Parked,
        span: linkw::TokenRefMsg,
    ) -> Result<(), String> {
        let _op_span = tracing::info_span!(
            "restore",
            session,
            model = self.trace_scope.model.as_str(),
            store_ns = self.trace_scope.ns,
            covered = p.covered,
            bytes = p.sealed.len(),
            encoding = p.encoding
        )
        .entered();
        let client = self.host.client();
        let seq = match client.state_sync(linkw::StateSyncReqMsg::CreateSequence) {
            Ok(linkw::StateSyncOkMsg::CreateSequence { sequence }) => sequence,
            Ok(_) => return Err("mismatched CreateSequence reply".into()),
            Err(e) => return Err(e.to_string()),
        };
        let free_seq = |client: &mut superfluid_agent::WorkerClient| {
            let _ = client.state_sync(linkw::StateSyncReqMsg::FreeSequence { sequence: seq });
        };
        let buf = match client.buf_create(p.sealed.len() as u64) {
            Ok(b) => b,
            Err(e) => {
                free_seq(client);
                return Err(e.to_string());
            }
        };
        let result = (|| {
            client
                .buf_write(buf, &p.sealed)
                .map_err(|e| e.to_string())?;
            let op = client
                .state_op(linkw::StateOpKind::Restore {
                    sequence: seq,
                    space_id: KV_SPACE,
                    buf_id: buf,
                })
                .map_err(|e| e.to_string())?;
            let done = client.wait_op_done(op).map_err(|e| e.to_string())?;
            if done.state != op_state::DONE || done.error != 0 {
                return Err(format!(
                    "restore terminal state {} error {}",
                    done.state, done.error
                ));
            }
            Ok(())
        })();
        let _ = client.buf_release(buf);
        if let Err(e) = result {
            free_seq(client);
            return Err(e);
        }
        match client.state_sync(linkw::StateSyncReqMsg::PublishSequence {
            sequence: seq,
            span,
        }) {
            Ok(linkw::StateSyncOkMsg::PublishSequence) => Ok(()),
            Ok(_) => {
                free_seq(client);
                Err("mismatched PublishSequence reply".into())
            }
            Err(e) => {
                free_seq(client);
                Err(e.to_string())
            }
        }
    }

    fn relieve_pressure(&mut self, mem: &linkw::MemCountersMsg, host_short: bool) {
        let (high, low) = (self.pressure_high_pct as u64, self.pressure_low_pct as u64);
        if std::env::var_os("SUPERFLUID_DEBUG_PRESSURE").is_some() {
            eprintln!(
                "[pressure] used={} total={} evictable={}",
                mem.pool_blocks_used, mem.pool_blocks_total, mem.pool_bytes_evictable
            );
        }
        if mem.pool_blocks_total == 0 || high == 0 {
            return;
        }
        let Some(block_bytes) = self.host.kv_block_bytes() else {
            return;
        };
        let (used, total) = (mem.pool_blocks_used, mem.pool_blocks_total);
        if used * 100 < total * high {
            return;
        }
        let low_blocks = total * low / 100;
        let excess = used.saturating_sub(low_blocks);
        if excess == 0 {
            return;
        }
        let slack = total * block_bytes / 4;
        let mut evictable = mem.pool_bytes_evictable;
        let need = excess * block_bytes;
        if need > evictable && self.pins.any_held() {
            evictable += self
                .pins
                .yield_for(&mut self.host, &self.stats, need - evictable);
        }
        let bytes_target = (excess * block_bytes + slack).min(evictable.max(1));
        // The pool's pressure leaves what the cache holds in host memory
        // alone, and the entries that leave the pool may be kept there.
        let all = if host_short { u64::MAX } else { !HOST_EXPORT_CLASS };
        // The slack is taken from entries no two requests have shared; a
        // shared prefix goes only for what the pool needs.
        let mut freed = self.evict_cache(bytes_target, all & !SHARED_PREFIX_CLASS);
        if freed < need {
            freed += self.evict_cache(need - freed, all);
        }
        if freed < need && self.pins.any_held() {
            let yielded = self
                .pins
                .yield_for(&mut self.host, &self.stats, need - freed);
            if yielded > 0 {
                freed += self.evict_cache((need - freed + slack).min(yielded), all);
            }
        }
        if freed > 0 {
            self.stats.pressure_evictions.fetch_add(1, Ordering::Relaxed);
            self.stats
                .pressure_bytes_evicted
                .fetch_add(freed, Ordering::Relaxed);
        }
    }

    fn evict_cache(&mut self, bytes_target: u64, classes: u64) -> u64 {
        let res = self
            .host
            .client()
            .state_sync(linkw::StateSyncReqMsg::CacheEvict {
                bytes_target,
                policy: linkw::ShedPolicyMsg {
                    victim_lanes: vec![],
                    evictable_cache_classes: classes,
                    protected_quota_bytes: 0,
                    max_evict_bytes: bytes_target,
                },
            });
        match res {
            Ok(linkw::StateSyncOkMsg::CacheEvict { bytes_freed }) => bytes_freed,
            _ => 0,
        }
    }

    fn retry_park_removals(&mut self) {
        if self.park_removals_pending.is_empty() {
            return;
        }
        let (Some(dir), Some(tx)) = (self.park_dir.clone(), self.park_tx.as_ref()) else {
            self.park_removals_pending.clear();
            return;
        };
        self.park_removals_pending.retain(|&session| {
            matches!(
                tx.try_send(ParkTask::Remove {
                    dir: dir.clone(),
                    session,
                }),
                Err(std::sync::mpsc::TrySendError::Full(_))
            )
        });
    }

    /// Lanes whose admission the engine could not take this tick go back to
    /// the head of their class queue, their seed leases released, to be
    /// admitted on a later tick.
    fn defer_admits(&mut self, tags: &[u64]) {
        let mut back: Vec<GenerateJob> = Vec::new();
        let mut seeds: Vec<u64> = Vec::new();
        let mut i = 0;
        while i < self.lanes.len() {
            if tags.contains(&self.lanes[i].tag) {
                let lane = self.lanes.remove(i);
                if lane.seed_handle != 0 {
                    seeds.push(lane.seed_handle);
                }
                back.push(lane.job);
            } else {
                i += 1;
            }
        }
        for seed in seeds {
            let _ = self.host.client().seed_release(seed);
        }
        for job in back.into_iter().rev() {
            let c = (job.class as usize).min(3);
            self.queues[c].jobs.push_front(job);
        }
    }

    fn drop_refused(&mut self, refused: &HashMap<u64, u32>) {
        let mut gone: Vec<(u64, u64)> = Vec::new();
        let mut active = self.active.lock().expect("active set");
        self.lanes.retain(|lane| {
            let Some(&code) = refused.get(&lane.tag) else { return true };
            active.remove(&lane.session);
            let status = superfluid_abi::Status::from_raw(code as i32)
                .map(|s| format!("{s:?}"))
                .unwrap_or_else(|| format!("status {}", code as i32));
            tracing::warn!(session = lane.session, lane = lane.tag, status = %status, "engine refused an admit");
            let _ = lane.job.events.send(LaneEvent::Failed(format!("the engine refused to admit this request ({status})")));
            gone.push((lane.session, lane.seed_handle));
            false
        });
        drop(active);
        for (session, seed) in gone {
            if seed != 0 {
                let _ = self.host.client().seed_release(seed);
            }
            self.drop_ephemeral_artifact(session);
        }
    }

    fn drop_ephemeral_artifact(&mut self, session: u64) {
        if !self.ephemeral_parked.remove(&session) {
            return;
        }
        let Some(dir) = self.park_dir.clone() else {
            return;
        };
        let Some(tx) = self.park_tx.as_ref() else {
            return;
        };
        if tx.try_send(ParkTask::Remove { dir, session }).is_err() {
            self.park_removals_pending.push(session);
        }
    }

    fn park_lane(&mut self, i: usize) -> Result<(), ParkRefused> {
        let Some(dir) = self.park_dir.clone() else {
            return Ok(());
        };
        if self.lanes[i].job.completion {
            return Ok(());
        }
        if (self.host.multi_space() || !self.lanes[i].job.media.is_empty()) && !self.host.has_unservable_space() {
            return self.park_lane_multi(i, dir);
        }
        if self.host.has_unservable_space() {
            self.stats
                .unservable_park_refusals
                .fetch_add(1, Ordering::Relaxed);
            return Err(ParkRefused::SpaceUnservable);
        }
        let Some(page) = self.host.page_size() else {
            return Ok(());
        };
        let (tag, session) = (self.lanes[i].tag, self.lanes[i].session);
        let n = self.lanes[i].all_tokens.len() as u64;
        if n <= 1 {
            return Ok(());
        }
        let mut aligned = ((n - 1) / page) * page;
        if aligned < page {
            return Ok(());
        }
        let park_encoding = if self.park_lossy {
            encoding::Q8
        } else {
            encoding::LOSSLESS
        };
        let client = self.host.client();
        let seq = match client.state_sync(linkw::StateSyncReqMsg::LaneSequence { lane_tag: tag }) {
            Ok(linkw::StateSyncOkMsg::LaneSequence { sequence }) => sequence,
            Ok(_) => return Err("mismatched LaneSequence reply".into()),
            Err(e) => return Err(e.to_string().into()),
        };
        if let Ok(linkw::StateSyncOkMsg::SnapshotBoundary { boundary }) =
            client.state_sync(linkw::StateSyncReqMsg::SnapshotBoundary { sequence: seq, space_id: KV_SPACE, cap: n - 1 })
        {
            aligned = aligned.min(boundary);
            if aligned < page {
                return Ok(());
            }
        }
        let digest = park::stream_digest(&self.lanes[i].all_tokens[..aligned as usize]);
        let _span = tracing::info_span!(
            "park",
            session,
            model = self.trace_scope.model.as_str(),
            store_ns = self.trace_scope.ns,
            covered = aligned,
            lossy = self.park_lossy
        )
        .entered();
        let (bytes, sizing_gen) = match client.state_sync(linkw::StateSyncReqMsg::ExportSize {
            sequence: seq,
            space_id: KV_SPACE,
            range: linkw::TokenRangeMsg {
                start: 0,
                end: aligned,
            },
            encoding: park_encoding,
        }) {
            Ok(linkw::StateSyncOkMsg::ExportSize {
                required_bytes,
                sizing_gen,
            }) => (required_bytes, sizing_gen),
            Ok(_) => return Err("mismatched ExportSize reply".into()),
            Err(e) => return Err(e.to_string().into()),
        };
        let buf = client.buf_create(bytes).map_err(|e| ParkRefused::from(e.to_string()))?;
        let result = (|| {
            let op_kind = if self.park_lossy {
                linkw::StateOpKind::Demote {
                    sequence: seq,
                    space_id: KV_SPACE,
                    range: linkw::TokenRangeMsg {
                        start: 0,
                        end: aligned,
                    },
                    encoding: encoding::Q8,
                    buf_id: buf,
                    sizing_gen,
                }
            } else {
                linkw::StateOpKind::Snapshot {
                    sequence: seq,
                    space_id: KV_SPACE,
                    boundary_pos: aligned,
                    buf_id: buf,
                    sizing_gen,
                }
            };
            let op = client.state_op(op_kind).map_err(|e| e.to_string())?;
            let done = client.wait_op_done(op).map_err(|e| e.to_string())?;
            if done.state != op_state::DONE || done.error != 0 {
                return Err(format!(
                    "snapshot terminal state {} error {}",
                    done.state, done.error
                ));
            }
            let take = if done.bytes_moved > 0 {
                done.bytes_moved.min(bytes)
            } else {
                bytes
            };
            client
                .buf_read(buf, take as usize)
                .map_err(|e| e.to_string())
        })();
        let _ = client.buf_release(buf);
        let sealed = result.map_err(ParkRefused::from)?;
        let job = ParkJob {
            dir,
            session,
            covered: aligned,
            digest,
            encoding: park_encoding,
            sealed,
            spaces: None,
        };
        match self
            .park_tx
            .as_ref()
            .expect("writer exists with park_dir")
            .try_send(ParkTask::Write(Box::new(job)))
        {
            Ok(()) => {
                if self.lanes[i].job.extras.ephemeral {
                    self.ephemeral_parked.insert(session);
                }
                if self.park_lossy {
                    self.stats.parks_lossy.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.stats.parks_lossless.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                return Err("park writer backlogged; artifact dropped".into());
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                return Err("park writer gone".into());
            }
        }
        self.lanes[i].parked_lossy = self.park_lossy;
        Ok(())
    }

    fn park_lane_multi(&mut self, i: usize, dir: PathBuf) -> Result<(), ParkRefused> {
        let (tag, session) = (self.lanes[i].tag, self.lanes[i].session);
        let n = self.lanes[i].all_tokens.len() as u64;
        if n <= 1 {
            return Ok(());
        }
        let spaces = self.host.parkable_spaces();
        let lossy = self.park_lossy;
        let fallback_boundary = self.host.park_boundary(n - 1);
        let client = self.host.client();
        let seq = match client.state_sync(linkw::StateSyncReqMsg::LaneSequence { lane_tag: tag }) {
            Ok(linkw::StateSyncOkMsg::LaneSequence { sequence }) => sequence,
            Ok(_) => return Err("mismatched LaneSequence reply".into()),
            Err(e) => return Err(e.to_string().into()),
        };
        let cap = n - 1;
        let mut covered = cap;
        let mut queried = false;
        for &(space_id, _, _, _) in &spaces {
            match client.state_sync(linkw::StateSyncReqMsg::SnapshotBoundary { sequence: seq, space_id, cap }) {
                Ok(linkw::StateSyncOkMsg::SnapshotBoundary { boundary }) => {
                    queried = true;
                    covered = covered.min(boundary);
                }
                Ok(_) => return Err("mismatched SnapshotBoundary reply".into()),
                Err(_) => break,
            }
        }
        if !queried {
            covered = fallback_boundary;
        }
        if covered == 0 {
            return Ok(());
        }
        let digest = park::stream_digest(&self.lanes[i].all_tokens[..covered as usize]);
        let _span = tracing::info_span!(
            "park",
            session,
            model = self.trace_scope.model.as_str(),
            store_ns = self.trace_scope.ns,
            covered,
            lossy = self.park_lossy,
            multi = true
        )
        .entered();
        let mut parked: Vec<park::ParkedSpace> = Vec::with_capacity(spaces.len());
        for &(space_id, kind, _, _) in &spaces {
            let paged = kind == superfluid_abi::space_kind::PAGED_TOKEN_KV;
            let encoding = if paged && lossy { encoding::Q8 } else { encoding::LOSSLESS };
            let range = if paged {
                linkw::TokenRangeMsg { start: 0, end: covered }
            } else {
                linkw::TokenRangeMsg { start: covered, end: covered }
            };
            let (bytes, sizing_gen) = match client.state_sync(linkw::StateSyncReqMsg::ExportSize {
                sequence: seq,
                space_id,
                range,
                encoding,
            }) {
                Ok(linkw::StateSyncOkMsg::ExportSize { required_bytes, sizing_gen }) => {
                    (required_bytes, sizing_gen)
                }
                Ok(_) => return Err("mismatched ExportSize reply".into()),
                Err(e) => return Err(format!("export size (space {space_id}): {e}").into()),
            };
            let buf = client.buf_create(bytes).map_err(|e| ParkRefused::from(e.to_string()))?;
            let result = (|| {
                let op_kind = if paged && lossy {
                    linkw::StateOpKind::Demote {
                        sequence: seq,
                        space_id,
                        range,
                        encoding: encoding::Q8,
                        buf_id: buf,
                        sizing_gen,
                    }
                } else {
                    linkw::StateOpKind::Snapshot {
                        sequence: seq,
                        space_id,
                        boundary_pos: covered,
                        buf_id: buf,
                        sizing_gen,
                    }
                };
                let op = client.state_op(op_kind).map_err(|e| e.to_string())?;
                let done = client.wait_op_done(op).map_err(|e| e.to_string())?;
                if done.state != op_state::DONE || done.error != 0 {
                    return Err(format!(
                        "space {space_id} export terminal state {} error {}",
                        done.state, done.error
                    ));
                }
                let take = if done.bytes_moved > 0 { done.bytes_moved.min(bytes) } else { bytes };
                client.buf_read(buf, take as usize).map_err(|e| e.to_string())
            })();
            let _ = client.buf_release(buf);
            let sealed = result.map_err(ParkRefused::from)?;
            parked.push(park::ParkedSpace {
                space_id,
                kind,
                encoding,
                sealed,
            });
        }
        let job = ParkJob {
            dir,
            session,
            covered,
            digest,
            encoding: if lossy { encoding::Q8 } else { encoding::LOSSLESS },
            sealed: Vec::new(),
            spaces: Some(parked),
        };
        match self
            .park_tx
            .as_ref()
            .expect("writer exists with park_dir")
            .try_send(ParkTask::Write(Box::new(job)))
        {
            Ok(()) => {
                if self.lanes[i].job.extras.ephemeral {
                    self.ephemeral_parked.insert(session);
                }
                if lossy {
                    self.stats.parks_lossy.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.stats.parks_lossless.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                return Err("park writer backlogged; artifact dropped".into());
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                return Err("park writer gone".into());
            }
        }
        self.lanes[i].parked_lossy = lossy;
        Ok(())
    }

    fn resume_adopt(&mut self, job: &GenerateJob, p: &park::Parked) -> Result<(u64, u64), String> {
        if p.covered as usize >= job.stream.len() {
            return Err("artifact covers the whole stream (nothing to prefill)".into());
        }
        if park::stream_digest(&job.stream[..p.covered as usize]) != p.stream_digest {
            return Err("stream digest mismatch (rewritten history)".into());
        }
        let _op_span = tracing::info_span!(
            "restore",
            session = job.session,
            model = self.trace_scope.model.as_str(),
            store_ns = self.trace_scope.ns,
            covered = p.covered,
            spaces = p.spaces.len(),
            multi = true
        )
        .entered();
        let want = self.host.parkable_spaces();
        for &(id, _, _, _) in &want {
            if !p.spaces.iter().any(|s| s.space_id == id) {
                return Err(format!("artifact lacks space {id}"));
            }
        }
        let covered_span = self
            .stage(&job.stream[..p.covered as usize])
            .map_err(|e| e.to_string())?;
        let client = self.host.client();
        let seq = match client.state_sync(linkw::StateSyncReqMsg::CreateSequence) {
            Ok(linkw::StateSyncOkMsg::CreateSequence { sequence }) => sequence,
            Ok(_) => return Err("mismatched CreateSequence reply".into()),
            Err(e) => return Err(e.to_string()),
        };
        let free_seq = |client: &mut superfluid_agent::WorkerClient| {
            let _ = client.state_sync(linkw::StateSyncReqMsg::FreeSequence { sequence: seq });
        };
        let mut order: Vec<&park::ParkedSpace> = p.spaces.iter().collect();
        order.sort_by_key(|s| (s.kind != superfluid_abi::space_kind::PAGED_TOKEN_KV, s.space_id));
        for sp in order {
            let buf = match client.buf_create(sp.sealed.len() as u64) {
                Ok(b) => b,
                Err(e) => {
                    free_seq(client);
                    return Err(e.to_string());
                }
            };
            let result = (|| {
                client.buf_write(buf, &sp.sealed).map_err(|e| e.to_string())?;
                let op = client
                    .state_op(linkw::StateOpKind::Restore {
                        sequence: seq,
                        space_id: sp.space_id,
                        buf_id: buf,
                    })
                    .map_err(|e| e.to_string())?;
                let done = client.wait_op_done(op).map_err(|e| e.to_string())?;
                if done.state != op_state::DONE || done.error != 0 {
                    return Err(format!(
                        "restore space {} terminal state {} error {}",
                        sp.space_id, done.state, done.error
                    ));
                }
                Ok(())
            })();
            let _ = client.buf_release(buf);
            if let Err(e) = result {
                free_seq(client);
                return Err(e);
            }
        }
        match client.state_sync(linkw::StateSyncReqMsg::AdoptSequence {
            sequence: seq,
            span: covered_span,
        }) {
            Ok(linkw::StateSyncOkMsg::AdoptSequence { seed_handle }) => Ok((seed_handle, p.covered)),
            Ok(_) => {
                free_seq(client);
                Err("mismatched AdoptSequence reply".into())
            }
            Err(e) => {
                free_seq(client);
                Err(e.to_string())
            }
        }
    }

    fn commit_tool_block(
        codec: &Arc<dyn TextCodec + Send + Sync>,
        store: &Arc<Mutex<SessionStore>>,
        tool_lease_ms: u64,
        session: u64,
        raw: String,
        schemas: Option<&crate::codec::ToolSchemas>,
    ) -> Result<Option<crate::CommittedEvent>, DaemonError> {
        match codec.parse_tool_call_with(&raw, schemas) {
            Some((name, args)) => {
                let mut store = store.lock().expect("store");
                let e = store.commit_tool_use(session, name, args)?;
                if tool_lease_ms > 0 {
                    let deadline = crate::wal::now_unix_ms() + tool_lease_ms;
                    let _ = store.commit_tool_lease(session, e.event_id, deadline);
                }
                Ok(Some(e))
            }
            None => Ok(Some(
                store.lock().expect("store").commit_tool_parse_failure(session, raw)?,
            )),
        }
    }

    fn finish_failed(&mut self, job: &GenerateJob, msg: &str) {
        self.drop_ephemeral_artifact(job.session);
        self.active.lock().expect("active set").remove(&job.session);
        let _ = job.events.send(LaneEvent::Failed(msg.to_string()));
    }

    fn publish_occupancy(&self) {
        self.stats.clear_tick_in_flight();
        self.stats
            .lanes_active
            .store(self.lanes.len() as u64, Ordering::Relaxed);
        for (c, q) in self.queues.iter().enumerate() {
            self.stats.queue_depth[c].store(q.jobs.len() as u64, Ordering::Relaxed);
        }
    }

    fn tick_once(&mut self, tick_span: &tracing::Span) -> Result<(), DaemonError> {
        self.publish_occupancy();
        if self.lanes.is_empty() {
            return Ok(());
        }

        let mut locally_done: Vec<u64> = Vec::new();
        for i in 0..self.lanes.len() {
            let session = self.lanes[i].session;
            let is_active = !matches!(self.lanes[i].phase, LanePhase::Finishing);
            if is_active && self.cancels.lock().expect("cancels").remove(&session) {
                let lane = &mut self.lanes[i];
                lane.fin = finish::CANCELLED;
                provisional_flush(&self.bus, lane);
                let text = lane.utf8.flush();
                let e = self.store.lock().expect("store").commit_generated(
                    session,
                    Vec::new(),
                    text,
                    lane.seg.current_channel(),
                    finish::CANCELLED,
                )?;
                let _ = lane.job.events.send(LaneEvent::Committed(e));
                if lane.admitted {
                    lane.phase = LanePhase::Finishing;
                } else {
                    if lane.seed_handle != 0 {
                        let _ = self.host.client().seed_release(lane.seed_handle);
                    }
                    locally_done.push(lane.tag);
                }
            }
        }
        let now_dl = std::time::Instant::now();
        for i in 0..self.lanes.len() {
            if matches!(self.lanes[i].phase, LanePhase::Finishing) {
                continue;
            }
            let Some(deadline) = self.lanes[i].job.deadline else {
                continue;
            };
            if now_dl <= deadline {
                continue;
            }
            let completion = self.lanes[i].job.completion;
            let session = self.lanes[i].session;
            if !completion {
                provisional_flush(&self.bus, &mut self.lanes[i]);
                let text = self.lanes[i].utf8.flush();
                let channel = self.lanes[i].seg.current_channel();
                let e = self.store.lock().expect("store").commit_generated(
                    session,
                    Vec::new(),
                    text,
                    channel,
                    finish::LENGTH,
                )?;
                let _ = self.lanes[i].job.events.send(LaneEvent::Committed(e));
                self.lanes[i].fin = finish::LENGTH;
            } else {
                let _ = self.lanes[i].job.events.send(LaneEvent::Expired);
            }
            let lane = &mut self.lanes[i];
            if lane.admitted {
                lane.phase = LanePhase::Finishing;
            } else {
                if lane.seed_handle != 0 {
                    let _ = self.host.client().seed_release(lane.seed_handle);
                }
                locally_done.push(lane.tag);
            }
        }
        if !locally_done.is_empty() {
            let mut locally_retired: Vec<u64> = Vec::new();
            let mut active = self.active.lock().expect("active set");
            self.lanes.retain(|lane| {
                if locally_done.contains(&lane.tag) {
                    active.remove(&lane.session);
                    let _ = lane.job.events.send(LaneEvent::Done {
                        tokens_generated: lane.produced,
                        finish: lane.fin,
                        warm_prefix: lane.warm_prefix,
                        spec: lane.spec,
                    });
                    locally_retired.push(lane.session);
                    false
                } else {
                    true
                }
            });
            drop(active);
            for session in locally_retired {
                self.drop_ephemeral_artifact(session);
            }
        }
        self.publish_occupancy();
        if self.lanes.is_empty() {
            return Ok(());
        }

        self.tick_count += 1;
        self.stats.ticks.fetch_add(1, Ordering::Relaxed);

        if self.park_lossy {
            for lane in &mut self.lanes {
                if lane.preempted
                    && !lane.parked
                    && lane.job.params.seed != 0
                    && !lane.job.extras.seed_drawn
                {
                    lane.parked = true;
                }
            }
        }
        if self.park_lossy && !self.pins.is_empty() && self.pins.enforceable(&self.host) {
            for lane in &mut self.lanes {
                if matches!(lane.phase, LanePhase::Finishing)
                    && !lane.parked
                    && lane.job.media.is_empty()
                    && !lane.job.completion
                    && self.pins.awaits_lease(&self.host, lane.session)
                {
                    lane.parked = true;
                }
            }
        }

        if !self.park_removals_pending.is_empty() {
            if let (Some(dir), Some(tx)) = (self.park_dir.clone(), self.park_tx.as_ref()) {
                self.park_removals_pending.retain(|&session| {
                    matches!(
                        tx.try_send(ParkTask::Remove {
                            dir: dir.clone(),
                            session,
                        }),
                        Err(std::sync::mpsc::TrySendError::Full(_))
                    )
                });
            } else {
                self.park_removals_pending.clear();
            }
        }

        if self.park_dir.is_some() {
            for i in 0..self.lanes.len() {
                if matches!(self.lanes[i].phase, LanePhase::Finishing)
                    && self.lanes[i].admitted
                    && !self.lanes[i].parked
                    && (!self.lanes[i].job.extras.ephemeral || self.lanes[i].preempted)
                {
                    self.lanes[i].parked = true;
                    if let Err(msg) = self.park_lane(i) {
                        eprintln!(
                            "[superfluid] park skipped (session {}): {msg}",
                            self.lanes[i].session
                        );
                    }
                }
            }
        }

        let rate_cap = |ceiling: u32| match self.prefill_secs_per_token_bulk {
            Some(spt) if spt > 0.0 => Some((MAX_PREFILL_TICK_SECS / spt).clamp(512.0, ceiling as f64) as u32),
            _ => None,
        };
        let solo = !self.prefill_budget_fixed
            && self.lanes.len() == 1
            && self.queued() == 0
            && self
                .lanes
                .iter()
                .all(|l| matches!(l.phase, LanePhase::Prefilling));
        let solo_budget = if solo { rate_cap(SOLO_PREFILL_BUDGET) } else { None };
        let open = self.joiner_waits_on_this_tick();
        let tick_prefill_budget = if self
            .lanes
            .iter()
            .any(|l| !l.job.media.is_empty() && matches!(l.phase, LanePhase::Prefilling))
        {
            PREFILL_BUDGET
        } else if open {
            // One chunk until two ticks have taught the rate; then the slower of the
            // average and the last tick.
            let spt = match (self.prefill_secs_per_token_bulk, self.prefill_secs_per_token_last) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            let cap = match spt.filter(|&t| t > 0.0 && self.prefill_rate_samples >= 2) {
                Some(spt) => {
                    let yields = matches!(
                        self.host.capabilities().read().expect("capability record").get("serving", "tick_yield"),
                        crate::capabilities::Cap::Yes
                    );
                    open_prefill_tick_cap(spt, open_prefill_tick_target(yields, self.latency_sensitive()))
                }
                None => 512,
            };
            self.prefill_budget.min(cap)
        } else if let Some(cap) = solo_budget.filter(|&cap| cap > self.prefill_budget) {
            cap
        } else {
            match self.prefill_secs_per_token_bulk {
                Some(spt) if spt > 0.0 => {
                    let lone = self.lanes.len() == 1
                        && self.queues.iter().all(|q| q.jobs.is_empty())
                        && self.lanes.iter().all(|l| {
                            Self::prefill_end_of(&self.speculation, l).saturating_sub(l.prefilled)
                                > prefill_tick_cap(spt, true) as u64
                        });
                    self.prefill_budget.min(prefill_tick_cap(spt, lone))
                }
                _ => self.prefill_budget,
            }
        };
        let mut plan = linkw::TickPlanMsg {
            plan_seq: {
                self.plan_seq += 1;
                self.plan_seq
            },
            flags: 0,
            prefill_token_budget: tick_prefill_budget,
            max_decode_lanes: self.max_lanes as u32,
            admits: vec![],
            commits: vec![],
            prefills: vec![],
            decodes: vec![],
            retires: vec![],
            shed_policy: linkw::ShedPolicyMsg {
                victim_lanes: vec![],
                evictable_cache_classes: u64::MAX,
                protected_quota_bytes: 0,
                max_evict_bytes: u64::MAX,
            },
        };

        let prefill_demand_lanes = self
            .lanes
            .iter()
            .filter(|l| {
                !matches!(l.phase, LanePhase::Finishing)
                    && l.prefilled < Self::prefill_end_of(&self.speculation, l)
            })
            .count()
            + self.queues.iter().map(|q| q.jobs.len()).sum::<usize>();
        self.plan_prefix_checkpoints();
        let mut budget = PrefillBudget::new(tick_prefill_budget as u64);
        let mut retiring: Vec<u64> = Vec::new();
        let starvation = self.starvation_ticks;
        let mut order: Vec<usize> = (0..self.lanes.len()).collect();
        order.sort_by_key(|&i| {
            let l = &self.lanes[i];
            let remaining = Self::prefill_end_of(&self.speculation, l).saturating_sub(l.prefilled);
            prefill_rank(l.job.class, remaining, l.tag, l.prefill_starved_ticks, starvation)
        });
        let mut decode_candidates: Vec<usize> = Vec::new();
        let speculation = Arc::clone(&self.speculation);
        for &li in &order {
            let lane = &mut self.lanes[li];
            if matches!(lane.phase, LanePhase::Finishing) {
                plan.retires.push(linkw::LaneRetireMsg {
                    lane_tag: lane.tag,
                    publish_to_cache: !lane.parked_lossy
                        && lane.job.media.is_empty()
                        && !lane.job.completion,
                });
                retiring.push(lane.tag);
                continue;
            }
            if !lane.admitted {
                let strategy_slot = Self::strategy_slot_for(&speculation, &lane.job);
                plan.admits.push(linkw::LaneAdmitMsg {
                    lane_tag: lane.tag,
                    prompt: lane.span,
                    seed_handle: lane.seed_handle,
                    sampling: if lane.job.params.temperature == 0.0 { 0 } else { 1 },
                    params: linkw::SamplingParamsMsg {
                        temperature: lane.job.params.temperature,
                        top_p: lane.job.params.top_p,
                        min_p: lane.job.params.min_p,
                        top_k: lane.job.params.top_k,
                        freq_penalty: lane.job.extras.frequency_penalty,
                        presence_penalty: lane.job.extras.presence_penalty,
                        repeat_penalty: lane.job.extras.repeat_penalty,
                        flags: if lane.job.extras.ignore_eos {
                            superfluid_abi::sampling_flag::IGNORE_EOS
                        } else {
                            0
                        },
                    },
                    rng_counter_base: lane
                        .job
                        .params
                        .seed
                        .wrapping_add(lane.job.stream.len() as u64),
                    grammar_handle: lane.job.grammar_handle,
                    grammar_replay: if lane.job.grammar_handle != 0 { lane.produced_before } else { 0 },
                    decode_replay: Self::decode_replay_for(strategy_slot, lane.produced_before),
                    logit_bias_handle: lane.job.extras.logit_bias_handle,
                    want_logprobs: lane.job.extras.want_logprobs,
                    top_logprobs: lane.job.extras.top_logprobs,
                    strategy_slot,
                    determinism_class: 0,
                    minimum_exactness: 0,
                    allow_approximate: false,
                    required_cert_id: 0,
                    host_sampler_identity: [0; 32],
                });
            }
            let total = Self::prefill_end_of(&speculation, lane);
            if lane.prefilled < total {
                let mut chunk = budget.offer_to(total - lane.prefilled, lane.job.class) as u32;
                if lane.checkpoint_at > lane.prefilled {
                    chunk = chunk.min((lane.checkpoint_at - lane.prefilled) as u32);
                }
                for &(off, _, n) in &lane.job.media {
                    let (rs, re) = (off, off + n as u64);
                    let (cs, ce) = (lane.prefilled, lane.prefilled + chunk as u64);
                    if rs < ce && re > ce {
                        chunk = if rs > cs { (rs - cs) as u32 } else { 0 };
                    }
                }
                if chunk > 0 {
                    lane.prefill_starved_ticks = 0;
                }
                if chunk > 0 {
                    plan.prefills.push(linkw::LanePrefillMsg {
                        lane_tag: lane.tag,
                        token_offset: lane.prefilled as u32,
                        token_count: chunk,
                    });
                    lane.prefilled += chunk as u64;
                    lane.checkpoint_due = lane.checkpoint_at > 0 && lane.prefilled == lane.checkpoint_at;
                    budget.take(chunk as u64, lane.prefilled >= total);
                    if lane.prefilled >= total && lane.job.class <= crate::qos::INLINE_COMPLETION {
                        budget.hold_below(lane.job.class);
                    }
                }
            }
            if lane.prefilled >= total {
                lane.phase = LanePhase::Decoding;
                if lane.job.max_tokens > lane.produced {
                    decode_candidates.push(li);
                }
            }
        }
        // A lane counts a starved tick only when the prefill went to a younger lane.
        let youngest_served = plan.prefills.iter().map(|p| p.lane_tag).max();
        for lane in self.lanes.iter_mut() {
            let waiting = lane.prefilled < Self::prefill_end_of(&self.speculation, lane)
                && !matches!(lane.phase, LanePhase::Finishing)
                && !plan.prefills.iter().any(|p| p.lane_tag == lane.tag);
            lane.prefill_starved_ticks = starved_after(lane.prefill_starved_ticks, waiting, youngest_served, lane.tag);
        }
        let divisor = self.background_tick_divisor.load(Ordering::Relaxed).max(1);
        let tick_no = self.tick_count;
        decode_candidates.sort_by_key(|&i| {
            let l = &self.lanes[i];
            let starving = l.starved_ticks >= starvation;
            (!starving, l.job.class, l.tag)
        });
        let planned_prefill_tokens: f64 = plan.prefills.iter().map(|p| p.token_count as f64).sum();
        let finishing = self.lanes.iter().any(|l| matches!(l.phase, LanePhase::Finishing));
        let shape = self.tick_shape;
        let target_secs = tick_target(&shape, self.tick_target_secs, self.lanes.len(), self.max_lanes, finishing);
        self.stats.tick_target_live_ms.store((target_secs * 1000.0).round() as u64, Ordering::Relaxed);
        let grant_cap: u16 = match (plan.prefills.len(), self.decode_round_secs) {
            (p, Some(c)) if c > 0.0 && !decode_candidates.is_empty() => {
                if p > 0 {
                    let prefill_secs = self.prefill_secs_per_token.map_or(0.0, |s| s * planned_prefill_tokens);
                    let interactive =
                        decode_candidates.iter().any(|&i| self.lanes[i].job.class <= crate::qos::INLINE_COMPLETION);
                    let share =
                        if interactive { DECODE_SHARE_BESIDE_PREFILL_INTERACTIVE } else { DECODE_SHARE_BESIDE_PREFILL };
                    ((prefill_secs * share / c) as u32)
                        .clamp(shape.beside_prefill as u32, TICK_DECODE_MAX as u32) as u16
                } else {
                    let rounds = ((target_secs / c) as u32).clamp(shape.decode_floor as u32, TICK_DECODE_MAX as u32);
                    let first_out = decode_candidates
                        .iter()
                        .map(|&i| self.lanes[i].job.max_tokens.saturating_sub(self.lanes[i].produced))
                        .min()
                        .unwrap_or(rounds);
                    let rounds = rounds.min(first_out.max(shape.beside_prefill as u32));
                    if self.just_retired {
                        rounds.min(shape.beside_prefill as u32) as u16
                    } else {
                        rounds as u16
                    }
                }
            }
            _ => shape.decode_floor,
        };
        self.stats.decode_grant_live.store(grant_cap as u64, Ordering::Relaxed);
        let mut decode_left = if self.tick_decode_budget_fixed {
            self.tick_decode_budget
        } else {
            (self.max_lanes as u32).saturating_mul(grant_cap as u32)
        };
        let retire_only = plan.prefills.is_empty() && !plan.retires.is_empty();
        if retire_only {
            decode_candidates.clear();
        }
        self.just_retired = retire_only;
        for &li in &decode_candidates {
            let lane = &mut self.lanes[li];
            let background = lane.job.class >= crate::qos::FOREGROUND_AGENT;
            let throttled = background && divisor > 1 && !tick_no.is_multiple_of(divisor as u64);
            let want = if throttled {
                0
            } else {
                (lane.job.max_tokens - lane.produced)
                    .min(grant_cap as u32)
                    .min(decode_left) as u16
            };
            if want > 0 {
                if lane.starved_ticks >= starvation {
                    self.stats.starvation_grants.fetch_add(1, Ordering::Relaxed);
                }
                lane.starved_ticks = 0;
                decode_left -= want as u32;
                let overshoot = (lane.job.max_tokens - lane.produced - want as u32).min(16) as u16;
                plan.decodes.push(linkw::LaneDecodeMsg {
                    lane_tag: lane.tag,
                    max_new_tokens: want,
                    overshoot,
                });
            } else {
                lane.starved_ticks = lane.starved_ticks.saturating_add(1);
            }
        }

        let wants_partial = self.lanes.iter().any(|l| {
            !matches!(l.phase, LanePhase::Finishing)
                && (l.job.extras.wants_deltas
                    || l.job.completion
                    || self.bus.has_provisional_subscriber(l.session))
        });
        if wants_partial {
            plan.flags |= superfluid_abi::tick_flags::PARTIAL_EMITS;
        }
        let mut plan_prefill_tokens: u32 = plan.prefills.iter().map(|p| p.token_count).sum();
        let prefill_by_tag: HashMap<u64, u32> =
            plan.prefills.iter().map(|p| (p.lane_tag, p.token_count)).collect();
        let plan_decode_rounds: u32 =
            plan.decodes.iter().map(|d| d.max_new_tokens as u32).max().unwrap_or(0);
        let plan_prefill_lanes_planned = plan.prefills.len();
        let plan_prefill_lanes = plan.prefills.len().max(prefill_demand_lanes);
        let plan_decode_lanes = plan.decodes.len();
        if !tick_span.is_disabled() {
            let planned: HashSet<u64> = plan
                .admits
                .iter()
                .map(|a| a.lane_tag)
                .chain(plan.prefills.iter().map(|p| p.lane_tag))
                .chain(plan.decodes.iter().map(|d| d.lane_tag))
                .chain(plan.retires.iter().map(|r| r.lane_tag))
                .collect();
            tick_span.record("lanes", planned.len());
            for lane in self.lanes.iter().filter(|l| planned.contains(&l.tag)) {
                tick_span.follows_from(&lane.job.trace);
            }
        }
        let tick_started = std::time::Instant::now();
        self.stats.set_tick_in_flight(TickInFlight {
            since_ms: uptime_ms().max(1),
            prefill_tokens: plan_prefill_tokens as u64,
            prefill_lanes: plan_prefill_lanes_planned as u64,
            decode_lanes: plan_decode_lanes as u64,
            admits: plan.admits.len() as u64,
            retires: plan.retires.len() as u64,
            target_ms: (target_secs * 1000.0).round() as u64,
        });
        let mut deferred: Vec<u64> = Vec::new();
        let ev = {
            let Scheduler {
                host,
                lanes,
                bus,
                codec,
                pins,
                stats,
                ..
            } = self;
            let by_tag: std::collections::HashMap<u64, usize> =
                lanes.iter().enumerate().map(|(i, l)| (l.tag, i)).collect();
            // The engine answers NeedsReplan when this tick's admissions do
            // not fit (sequence slots or cells it cannot free); it has run
            // nothing. Admit fewer rather than fail every lane.
            let fewer = (!plan.admits.is_empty()
                && plan.admits.iter().all(|a| {
                    by_tag.get(&a.lane_tag).is_some_and(|&i| lanes[i].job.media.is_empty())
                }))
            .then(|| plan.clone());
            let mut on_emit = |em: linkw::TickEmitMsg| {
                let Some(&i) = by_tag.get(&em.lane_tag) else { return };
                let lane = &mut lanes[i];
                lane.frame_fed += em.tokens.len();
                stats
                    .decode_tokens
                    .fetch_add(em.tokens.len() as u64, Ordering::Relaxed);
                if !lane.ttft_observed && lane.ttft_frame_ms.is_none() && !em.tokens.is_empty() {
                    lane.ttft_frame_ms = Some(lane.admitted_at.elapsed().as_millis() as u64);
                }
                if lane.job.completion {
                    if !lane.emit_dead
                        && lane.job.events.send(LaneEvent::CompletionTokens(em.tokens)).is_err()
                    {
                        lane.emit_dead = true;
                    }
                } else {
                    provisional_ingest(codec, bus, lane, &em.tokens);
                }
            };
            let retry = pins.any_held().then(|| plan.clone());
            let needs_replan = |r: &Result<linkw::TickEventsMsg, superfluid_agent::AgentError>| {
                matches!(r, Err(superfluid_agent::AgentError::Rejected(s)) if *s == superfluid_abi::Status::NeedsReplan as i32)
            };
            let mut res = match (host.client().tick_streaming(plan, &mut on_emit), retry) {
                (Err(superfluid_agent::AgentError::Rejected(status)), Some(plan))
                    if status == superfluid_abi::Status::NeedsReplan as i32
                        && pins.yield_all(host, stats) =>
                {
                    host.client().tick_streaming(plan, &mut on_emit)
                }
                (res, _) => res,
            };
            if let Some(full) = fewer.filter(|_| needs_replan(&res)) {
                let mut keep = full.admits.len();
                while keep > 0 && needs_replan(&res) {
                    keep /= 2;
                    res = host.client().tick_streaming(with_first_admits(&full, keep), &mut on_emit);
                }
                if res.is_ok() {
                    deferred = full.admits[keep..].iter().map(|a| a.lane_tag).collect();
                }
            }
            stats.clear_tick_in_flight();
            res?
        };
        if !deferred.is_empty() {
            plan_prefill_tokens = plan_prefill_tokens
                .saturating_sub(deferred.iter().filter_map(|t| prefill_by_tag.get(t)).sum());
            self.stats.admits_deferred.fetch_add(deferred.len() as u64, Ordering::Relaxed);
            self.defer_admits(&deferred);
        }
        let tick_secs = tick_started.elapsed().as_secs_f64().max(1e-3);
        // Prefill the engine shed (a tick that yielded between steps) is
        // planned again from where it stopped.
        for e in ev.shed.entries.iter().filter(|e| e.kind == superfluid_abi::shed_kind::PREFILL_CHUNK) {
            if let Some(lane) = self.lanes.iter_mut().find(|l| l.tag == e.lane_tag) {
                lane.prefilled = lane.prefilled.saturating_sub(e.tokens);
                if matches!(lane.phase, LanePhase::Decoding) {
                    lane.phase = LanePhase::Prefilling;
                }
                if lane.checkpoint_due && lane.prefilled != lane.checkpoint_at {
                    lane.checkpoint_due = false;
                }
            }
        }
        let prefill_tokens_ran: u32 = plan_prefill_tokens
            .saturating_sub(
                ev.admit_results
                    .iter()
                    .filter(|a| a.status != superfluid_abi::admit_status::ADMITTED)
                    .filter_map(|a| prefill_by_tag.get(&a.lane_tag))
                    .sum(),
            )
            .saturating_sub(
                ev.faults
                    .iter()
                    .filter(|f| f.code == superfluid_abi::fault::MEDIA)
                    .filter_map(|f| prefill_by_tag.get(&f.lane_tag))
                    .sum(),
            )
            .saturating_sub(
                ev.shed
                    .entries
                    .iter()
                    .filter(|e| e.kind == superfluid_abi::shed_kind::PREFILL_CHUNK)
                    .map(|e| e.tokens.min(u32::MAX as u64) as u32)
                    .sum(),
            );
        let tick_rounds_exec: u32 = ev
            .emits
            .iter()
            .filter(|e| e.n_tokens > 0)
            .map(|e| e.n_tokens.saturating_sub(e.spec.accepted).max(1))
            .max()
            .unwrap_or(0);
        self.stats
            .prefill_tokens
            .fetch_add(prefill_tokens_ran as u64, Ordering::Relaxed);
        let phase_split = plan_prefill_tokens == 0 || ev.timings.prefill_ns > 0;
        let (prefill_secs_meas, decode_secs_meas) = if phase_split {
            (ev.timings.prefill_ns as f64 / 1e9, ev.timings.decode_ns as f64 / 1e9)
        } else {
            let decode_est = self.decode_round_secs.unwrap_or(0.0) * tick_rounds_exec as f64;
            ((tick_secs - decode_est).max(1e-3), decode_est)
        };
        // A tick the daemon ended early ran fewer steps than planned: the
        // rates are of what ran.
        let decode_rounds_ran = if tick_rounds_exec > 0 { tick_rounds_exec.min(plan_decode_rounds) } else { plan_decode_rounds };
        if decode_rounds_ran > 0 && phase_split && decode_secs_meas > 0.0 {
            let sample = decode_secs_meas / decode_rounds_ran as f64;
            self.decode_round_secs = Some(match self.decode_round_secs {
                Some(c) => 0.7 * c + 0.3 * sample,
                None => sample,
            });
        } else if plan_prefill_tokens == 0 && decode_rounds_ran > 0 {
            let sample = tick_secs / decode_rounds_ran as f64;
            self.decode_round_secs = Some(match self.decode_round_secs {
                Some(c) => 0.7 * c + 0.3 * sample,
                None => sample,
            });
        }
        if prefill_tokens_ran >= PREFILL_RATE_MIN_TOKENS && phase_split {
            let sample = prefill_secs_meas.max(1e-3) / prefill_tokens_ran as f64;
            self.prefill_secs_per_token = Some(match self.prefill_secs_per_token {
                Some(c) => 0.7 * c + 0.3 * sample,
                None => sample,
            });
            if prefill_tokens_ran >= 512 {
                self.prefill_secs_per_token_last = Some(sample);
                self.prefill_rate_samples = self.prefill_rate_samples.saturating_add(1);
                self.prefill_secs_per_token_bulk = Some(match self.prefill_secs_per_token_bulk {
                    Some(c) => 0.7 * c + 0.3 * sample,
                    None => sample,
                });
            }
        }
        if !self.prefill_budget_fixed && prefill_tokens_ran > 0 {
            // What the whole planned prefill would have taken at the rate of
            // what ran.
            let prefill_secs = prefill_secs_meas.max(1e-3) * plan_prefill_tokens as f64 / prefill_tokens_ran as f64;
            let share = plan_prefill_lanes as f64
                / (plan_prefill_lanes as f64 + plan_decode_lanes as f64);
            let factor = (self.tick_target_secs * share / prefill_secs).clamp(0.5, 2.0);
            self.prefill_budget =
                ((self.prefill_budget as f64 * factor) as u32).clamp(256, PREFILL_BUDGET);
        }
        if std::env::var_os("SUPERFLUID_DEBUG_TICK").is_some() {
            eprintln!(
                "[tick] secs={:.3} pf_secs={:.3} dec_secs={:.3} prefill_tok={} rounds={} P_eff={} P_plan={} D={} round_secs={:?} pf_secs_per_tok={:?} budget={} grant={}",
                tick_secs, prefill_secs_meas, decode_secs_meas, plan_prefill_tokens, plan_decode_rounds, plan_prefill_lanes, plan_prefill_lanes_planned,
                plan_decode_lanes, self.decode_round_secs, self.prefill_secs_per_token, self.prefill_budget, grant_cap
            );
        }
        self.stats
            .prefill_budget_live
            .store(self.prefill_budget as u64, Ordering::Relaxed);
        if !self.pins.is_empty() {
            let retired: Vec<(u64, Vec<u32>)> = self
                .lanes
                .iter()
                .filter(|l| retiring.contains(&l.tag) && !l.job.completion)
                .map(|l| (l.session, l.all_tokens.clone()))
                .collect();
            self.pins
                .after_tick(&mut self.host, &self.stats, self.tick_count, &retired);
        }
        self.publish_prefix_checkpoints();
        self.relieve_pressure(&ev.mem, false);
        self.stats
            .pool_blocks_used
            .store(ev.mem.pool_blocks_used, Ordering::Relaxed);
        self.stats
            .pool_blocks_total
            .store(ev.mem.pool_blocks_total, Ordering::Relaxed);

        let refused: HashMap<u64, u32> = ev
            .admit_results
            .iter()
            .filter(|r| r.status == superfluid_abi::admit_status::REJECTED)
            .map(|r| (r.lane_tag, r.reject_code))
            .collect();
        if !refused.is_empty() {
            self.drop_refused(&refused);
        }

        for f in &ev.faults {
            if let Some(lane) = self.lanes.iter_mut().find(|l| l.tag == f.lane_tag) {
                if lane.fault_code.is_some() {
                    continue;
                }
                lane.fault_code = Some(f.code);
                lane.fin = finish::ERROR;
                provisional_flush(&self.bus, lane);
                let text = lane.utf8.flush();
                let e = self.store.lock().expect("store").commit_generated(
                    lane.session,
                    Vec::new(),
                    text,
                    lane.seg.current_channel(),
                    finish::ERROR,
                )?;
                let _ = lane.job.events.send(LaneEvent::Committed(e));
                lane.phase = LanePhase::Finishing;
                tracing::warn!(session = lane.session, lane = lane.tag, code = f.code, "engine lane fault");
            }
        }

        #[allow(clippy::type_complexity)]
        let mut logprobs_by_lane: HashMap<u64, Vec<(f32, Vec<(u32, f32)>)>> = HashMap::new();
        for lp in &ev.logprobs {
            let top: Vec<(u32, f32)> = lp
                .top_ids
                .iter()
                .zip(&lp.top_logprob_bits)
                .map(|(id, b)| (*id, f32::from_bits(*b)))
                .collect();
            logprobs_by_lane
                .entry(lp.lane_tag)
                .or_default()
                .push((f32::from_bits(lp.logprob_bits), top));
        }
        let mut emitted: HashMap<u64, (Vec<u32>, u32)> = HashMap::new();
        for emit in &ev.emits {
            if emit.spec.proposed > 0 {
                self.stats
                    .spec_proposed
                    .fetch_add(emit.spec.proposed as u64, Ordering::Relaxed);
                self.stats
                    .spec_accepted
                    .fetch_add(emit.spec.accepted as u64, Ordering::Relaxed);
                if let Some(lane) = self.lanes.iter_mut().find(|l| l.tag == emit.lane_tag) {
                    lane.spec.0 += emit.spec.proposed as u64;
                    lane.spec.1 += emit.spec.accepted as u64;
                }
            }
            if emit.n_tokens > 0 {
                let tokens = self.host.client().read_tokens(&emit.token_ref)?;
                emitted.insert(emit.lane_tag, (tokens, emit.finish));
            } else if emit.finish != finish::NONE {
                emitted.insert(emit.lane_tag, (Vec::new(), emit.finish));
            }
        }
        let tick_decode_tokens: u64 = emitted.values().map(|(t, _)| t.len() as u64).sum();
        let ttft_tail_secs = if tick_rounds_exec > 1 {
            decode_secs_meas * (tick_rounds_exec as f64 - 1.0) / tick_rounds_exec as f64
        } else {
            0.0
        };
        if prefill_tokens_ran > 0 || tick_decode_tokens > 0 {
            self.stats
                .tick_wall_ms_live
                .store((tick_secs * 1e3) as u64, Ordering::Relaxed);
            self.stats.prefill_rate_live.store(
                per_sec(prefill_tokens_ran as u64, prefill_secs_meas),
                Ordering::Relaxed,
            );
            let decode_secs_for_rate =
                if decode_secs_meas > 0.0 { decode_secs_meas } else { tick_secs };
            self.stats.decode_rate_live.store(
                per_sec(tick_decode_tokens, decode_secs_for_rate),
                Ordering::Relaxed,
            );
            self.stats.work_ticks.fetch_add(1, Ordering::Relaxed);
        }
        let terminators = self.codec.turn_terminators();
        for lane in &mut self.lanes {
            lane.admitted = true;
            let Some((tokens, fin)) = emitted.remove(&lane.tag) else {
                continue;
            };
            if tokens.is_empty() {
                lane.fin = fin;
                if !lane.job.completion {
                    provisional_flush(&self.bus, lane);
                    let text = lane.utf8.flush();
                    let e = self.store.lock().expect("store").commit_generated(
                        lane.session,
                        Vec::new(),
                        text,
                        lane.seg.current_channel(),
                        fin,
                    )?;
                    let _ = lane.job.events.send(LaneEvent::Committed(e));
                }
                lane.phase = LanePhase::Finishing;
                continue;
            }
            lane.all_tokens.extend_from_slice(&tokens);
            lane.produced += tokens.len() as u32;
            lane.fin = fin;
            let frame_fed = std::mem::take(&mut lane.frame_fed).min(tokens.len());
            if !lane.job.completion {
                provisional_ingest(&self.codec, &self.bus, lane, &tokens[frame_fed..]);
            }
            if lane.fin == finish::NONE
                && !lane.job.extras.ignore_eos
                && !terminators.is_empty()
                && tokens.iter().any(|t| terminators.contains(t))
            {
                lane.fin = finish::EOS;
            }
            if lane.job.extras.want_logprobs {
                if let Some(recs) = logprobs_by_lane.remove(&lane.tag) {
                    if !recs.is_empty() {
                        let items: Vec<TokenLogprob> = tokens
                            .iter()
                            .zip(recs)
                            .map(|(t, (lp, top))| TokenLogprob { token: *t, logprob: lp, top })
                            .collect();
                        let _ = lane.job.events.send(LaneEvent::TokenLogprobs(items));
                    }
                }
            }
            self.stats
                .decode_tokens
                .fetch_add((tokens.len() - frame_fed) as u64, Ordering::Relaxed);
            if !lane.ttft_observed && !tokens.is_empty() {
                lane.ttft_observed = true;
                let ms = lane.ttft_frame_ms.take().unwrap_or_else(|| {
                    let elapsed = lane.admitted_at.elapsed().as_secs_f64();
                    ((elapsed - ttft_tail_secs).max(0.0) * 1e3) as u64
                });
                self.stats.observe_ttft(ms);
            }
            let terminal_now = lane.fin != finish::NONE || lane.produced >= lane.job.max_tokens;
            if lane.job.completion {
                let leftover = tokens[frame_fed..].to_vec();
                if lane.emit_dead
                    || (!leftover.is_empty()
                        && lane.job.events.send(LaneEvent::CompletionTokens(leftover)).is_err())
                {
                    lane.phase = LanePhase::Finishing;
                    continue;
                }
                if terminal_now {
                    lane.phase = LanePhase::Finishing;
                }
                continue;
            }
            if terminal_now {
                provisional_flush(&self.bus, lane);
            }
            let runs = lane.seg.split(&tokens);
            let n_runs = runs.len().max(1);
            let mut dead = false;
            for (i, run) in runs.into_iter().enumerate() {
                let last = i + 1 == n_runs;
                let mut text = String::new();
                for &t in &run.text {
                    text.push_str(&lane.utf8.push(&self.codec.token_bytes(t)));
                }
                if last && terminal_now {
                    text.push_str(&lane.utf8.flush());
                }
                let channel = run.channel;
                let tool_closed = run.closes && channel == crate::wal::channel::TOOL_CALL;
                if channel == crate::wal::channel::TOOL_CALL {
                    lane.tool_buf.push_str(&text);
                }
                let e = self.store.lock().expect("store").commit_generated(
                    lane.session,
                    run.span,
                    text,
                    channel,
                    if last { fin } else { finish::NONE },
                )?;
                let tool_event = if tool_closed {
                    let raw = std::mem::take(&mut lane.tool_buf);
                    Self::commit_tool_block(
                        &self.codec,
                        &self.store,
                        self.tool_lease_ms,
                        lane.session,
                        raw,
                        lane.job.extras.tool_schemas.as_deref(),
                    )?
                } else {
                    None
                };
                if lane.job.events.send(LaneEvent::Committed(e)).is_err() {
                    dead = true;
                    break;
                }
                if let Some(te) = tool_event {
                    if lane.job.events.send(LaneEvent::Committed(te)).is_err() {
                        dead = true;
                        break;
                    }
                }
            }
            if dead {
                lane.fin = finish::CANCELLED;
                provisional_flush(&self.bus, lane);
                let text = lane.utf8.flush();
                let _ = self.store.lock().expect("store").commit_generated(
                    lane.session,
                    Vec::new(),
                    text,
                    lane.seg.current_channel(),
                    finish::CANCELLED,
                )?;
                lane.phase = LanePhase::Finishing;
                continue;
            }
            if terminal_now {
                if !lane.tool_buf.is_empty() {
                    let raw = std::mem::take(&mut lane.tool_buf);
                    let te = Self::commit_tool_block(
                        &self.codec,
                        &self.store,
                        self.tool_lease_ms,
                        lane.session,
                        raw,
                        lane.job.extras.tool_schemas.as_deref(),
                    )?;
                    if let Some(te) = te {
                        let _ = lane.job.events.send(LaneEvent::Committed(te));
                    }
                }
                lane.phase = LanePhase::Finishing;
            }
        }

        if !retiring.is_empty() {
            let mut i = 0;
            let mut requeue: Vec<GenerateJob> = Vec::new();
            while i < self.lanes.len() {
                if !retiring.contains(&self.lanes[i].tag) {
                    i += 1;
                    continue;
                }
                let lane = self.lanes.remove(i);
                if lane.preempted && lane.fin == finish::NONE {
                    let mut job = lane.job;
                    job.stream = lane.all_tokens;
                    job.max_tokens -= lane.produced;
                    job.carry = Some(LaneCarry {
                        seg: lane.seg,
                        utf8: lane.utf8,
                        disp_seg: lane.disp_seg,
                        disp_utf8: lane.disp_utf8,
                        tool_buf: lane.tool_buf,
                        produced_before: lane.produced_before + lane.produced,
                        spec_before: lane.spec,
                        warm_prefix_first: Some(lane.warm_prefix_first.unwrap_or(lane.warm_prefix)),
                    });
                    requeue.push(job);
                } else {
                    self.drop_ephemeral_artifact(lane.session);
                    self.active.lock().expect("active set").remove(&lane.session);
                    if let Some(code) = lane.fault_code {
                        let _ = lane
                            .job
                            .events
                            .send(LaneEvent::Failed(format!("engine lane fault (code {code})")));
                    } else {
                        let _ = lane.job.events.send(LaneEvent::Done {
                            tokens_generated: lane.produced_before + lane.produced,
                            finish: lane.fin,
                            warm_prefix: lane.warm_prefix_first.unwrap_or(lane.warm_prefix),
                            spec: lane.spec,
                        });
                    }
                }
            }
            for job in requeue {
                let c = (job.class as usize).min(3);
                self.queues[c].jobs.push_front(job);
            }
        }
        self.publish_occupancy();
        Ok(())
    }
}

fn draft_weights_identity(path: &str) -> Result<(u64, u64), DaemonError> {
    use std::io::Read;
    const SEED: u64 = 0xCBF2_9CE4_8422_2325;
    let mut f = std::fs::File::open(path).map_err(|_| {
        DaemonError::Config(
            "draft-model weights are not readable by the daemon; an exact draft identity \
             requires the daemon to read the weights file",
        )
    })?;
    let size = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut h = SEED;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => h = superfluid_fingerprint::fnv1a64(&buf[..n], h),
            Err(_) => {
                return Err(DaemonError::Config(
                    "failed reading draft-model weights for identity",
                ))
            }
        }
    }
    Ok((h, size))
}

pub fn run_completion(
    tx: &Sender<SchedCmd>,
    stream: Vec<u32>,
    params: GenParams,
    max_tokens: u32,
    deadline_ms: u64,
    grammar_handle: u32,
) -> Result<(Vec<u32>, u32, bool), DaemonError> {
    run_completion_streaming(
        tx,
        stream,
        params,
        max_tokens,
        deadline_ms,
        GenExtras { grammar_handle, ..GenExtras::default() },
        |_| std::ops::ControlFlow::Continue(()),
    )
}

pub fn run_completion_streaming(
    tx: &Sender<SchedCmd>,
    stream: Vec<u32>,
    params: GenParams,
    max_tokens: u32,
    deadline_ms: u64,
    extras: GenExtras,
    mut on_tokens: impl FnMut(&[u32]) -> std::ops::ControlFlow<()>,
) -> Result<(Vec<u32>, u32, bool), DaemonError> {
    let (etx, erx) = std::sync::mpsc::channel();
    tx.send(SchedCmd::Submit(Box::new(GenerateJob {
        session: 0,
        stream,
        input_end: 0,
        params,
        max_tokens,
        events: etx,
        class: crate::qos::INLINE_COMPLETION,
        batch_invariant: false,
        carry: None,
        queued_at: 0,
        media: Vec::new(),
        completion: true,
        deadline: (deadline_ms > 0).then(|| std::time::Instant::now() + std::time::Duration::from_millis(deadline_ms)),
        grammar_handle: extras.grammar_handle,
        extras,
        trace: tracing::Span::current(),
    })))
    .map_err(|_| DaemonError::Protocol("scheduler is gone"))?;
    let mut tokens = Vec::new();
    loop {
        match erx.recv() {
            Ok(LaneEvent::Provisional { .. }) => {}
            Ok(LaneEvent::CompletionTokens(t)) => {
                if on_tokens(&t).is_break() {
                    tokens.extend(t);
                    return Ok((tokens, finish::CANCELLED, false));
                }
                tokens.extend(t);
            }
            Ok(LaneEvent::Done { finish, .. }) => return Ok((tokens, finish, false)),
            Ok(LaneEvent::Expired) => return Ok((Vec::new(), finish::NONE, true)),
            Ok(LaneEvent::Committed(_)) => {}
            Ok(LaneEvent::TokenLogprobs(_)) => {}
            Ok(LaneEvent::Failed(msg)) => return Err(DaemonError::Generation(msg)),
            Err(_) => return Err(DaemonError::Protocol("scheduler dropped the lane")),
        }
    }
}

pub type RunJobResult = (Vec<CommittedEvent>, u32, u32, u64, Vec<TokenLogprob>, (u64, u64));

#[allow(clippy::too_many_arguments)]
pub fn run_job(
    tx: &Sender<SchedCmd>,
    job_session: u64,
    stream: Vec<u32>,
    input_end: u64,
    params: GenParams,
    max_tokens: u32,
    class: u8,
    batch_invariant: bool,
    media: Vec<(u64, std::path::PathBuf, u32)>,
    extras: GenExtras,
    cancels: &Arc<crate::CancelSet>,
    mut on_event: impl FnMut(&CommittedEvent) -> Result<(), DaemonError>,
    mut on_logprobs: impl FnMut(&[TokenLogprob]),
    mut on_delta: impl FnMut(u32, &str, u32),
) -> Result<RunJobResult, DaemonError> {
    let (etx, erx) = std::sync::mpsc::channel();
    let grammar_handle = extras.grammar_handle;
    tx.send(SchedCmd::Submit(Box::new(GenerateJob {
        session: job_session,
        stream,
        input_end,
        params,
        max_tokens,
        events: etx,
        class,
        batch_invariant,
        carry: None,
        queued_at: 0,
        media,
        completion: false,
        deadline: None,
        grammar_handle,
        extras,
        trace: tracing::Span::current(),
    })))
    .map_err(|_| DaemonError::Protocol("scheduler is gone"))?;
    if class <= crate::qos::INLINE_COMPLETION {
        // A chat or a completion that may preempt: the running tick ends at
        // its next step instead of making it wait, and steps stay short while
        // such requests keep coming.
        cancels.latency_request();
    }
    let mut events = Vec::new();
    let mut logprobs: Vec<TokenLogprob> = Vec::new();
    let mut delivery_err: Option<DaemonError> = None;
    loop {
        match erx.recv() {
            Ok(LaneEvent::Provisional { channel, text, produced }) => {
                if delivery_err.is_none() {
                    on_delta(channel, &text, produced);
                }
            }
            Ok(LaneEvent::TokenLogprobs(items)) => {
                if delivery_err.is_none() {
                    on_logprobs(&items);
                }
                logprobs.extend(items);
            }
            Ok(LaneEvent::Committed(e)) => {
                if delivery_err.is_none() {
                    if let Err(err) = on_event(&e) {
                        cancels.cancel(job_session);
                        delivery_err = Some(err);
                    } else {
                        events.push(e);
                    }
                }
            }
            Ok(LaneEvent::Done {
                tokens_generated,
                finish,
                warm_prefix,
                spec,
            }) => {
                return match delivery_err {
                    Some(err) => Err(err),
                    None => Ok((events, tokens_generated, finish, warm_prefix, logprobs, spec)),
                };
            }
            Ok(LaneEvent::Failed(msg)) => {
                return Err(delivery_err.unwrap_or(DaemonError::Generation(msg)));
            }
            Ok(LaneEvent::CompletionTokens(_)) | Ok(LaneEvent::Expired) => {
                return Err(DaemonError::Protocol("completion event on a generate lane"));
            }
            Err(_) => {
                return Err(delivery_err.unwrap_or(DaemonError::Protocol("scheduler dropped the lane")));
            }
        }
    }
}
