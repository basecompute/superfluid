//! superfluid — the daemon.

pub mod anthropic;
pub mod api;
pub mod apiweb;
pub mod atem;
pub mod bus;
pub mod capabilities;
pub mod cli;
pub mod codec;
mod exec;
pub mod completion_bucket;
pub mod files;
pub mod harmony;
pub mod keypolicy;
pub mod launch;
pub mod batches;
pub mod response_store;
pub mod ratelimit;
pub mod registry;
pub mod template_codec;
pub mod tool_fragment;
pub mod fleet;
pub mod fleet_manager;
pub mod fleet_join;
pub mod fleet_mdns;
pub mod fleet_token;
#[cfg(feature = "tui")]
pub mod tui;
#[cfg(feature = "tui")]
pub mod tui_top;
pub mod nodeagent;
mod inference;
mod block_inference;
mod generation_output;
mod openai_output;
mod http_state;
mod http_generation;
mod keepalive;
mod responses;
pub mod openai;
pub mod ollama;
pub mod fleet_openai;
pub mod park;
pub mod pins;
pub mod runtime;
pub mod runtime_pick;
pub mod installs;
pub mod pulls;
pub mod runtimes;
pub mod runtimes_report;
pub mod linked;
pub mod dylib_tokenizer;
pub mod export;
pub mod otlp;
pub mod media;
pub mod metrics;
pub mod pressure;
pub mod scheduler;
pub mod serve;
pub mod speculate_resolve;
use superfluid_proto::linkw as proto_linkw;
pub mod telemetry;
pub mod store;
pub mod store_id;
pub mod wal;
pub mod workerd;

pub mod qos {
    pub const INTERACTIVE_CHAT: u8 = 0;
    pub const INLINE_COMPLETION: u8 = 1;
    pub const FOREGROUND_AGENT: u8 = 2;
    pub const BACKGROUND_AGENT: u8 = 3;

    pub const NAMES: [&str; 4] = ["interactive", "completion", "agent", "background"];

    pub fn parse(s: &str) -> Option<u8> {
        match s.trim().to_ascii_lowercase().as_str() {
            "interactive" | "interactive-chat" | "interactive_chat" | "0" => Some(INTERACTIVE_CHAT),
            "completion" | "inline-completion" | "inline_completion" | "1" => Some(INLINE_COMPLETION),
            "agent" | "foreground" | "foreground-agent" | "foreground_agent" | "2" => Some(FOREGROUND_AGENT),
            "background" | "background-agent" | "background_agent" | "3" => Some(BACKGROUND_AGENT),
            _ => None,
        }
    }

    pub fn parse_class_lanes(spec: &str, max_lanes: usize) -> Result<[usize; 4], String> {
        let mut caps = [0usize; 4];
        let mut seen = [false; 4];
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (name, n) = part
                .split_once('=')
                .ok_or_else(|| format!("expected class=N, got {part:?}"))?;
            let class = parse(name).ok_or_else(|| {
                format!("unknown class {:?} (expected one of {})", name.trim(), NAMES.join(", "))
            })? as usize;
            if seen[class] {
                return Err(format!("class {} is capped twice", NAMES[class]));
            }
            seen[class] = true;
            let n: usize = n
                .trim()
                .parse()
                .map_err(|_| format!("cap for {} must be a whole number, got {:?}", NAMES[class], n.trim()))?;
            if n == 0 {
                return Err(format!(
                    "cap for {} must be at least 1 (omit the class to leave it uncapped)",
                    NAMES[class]
                ));
            }
            if n > max_lanes {
                return Err(format!(
                    "cap {n} for {} exceeds --max-batch {max_lanes}; it could never bind",
                    NAMES[class]
                ));
            }
            caps[class] = n;
        }
        Ok(caps)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn names_round_trip_and_aliases_parse() {
            for (c, name) in NAMES.iter().enumerate() {
                assert_eq!(parse(name), Some(c as u8));
                assert_eq!(parse(&c.to_string()), Some(c as u8));
            }
            assert_eq!(parse(" Interactive-Chat "), Some(INTERACTIVE_CHAT));
            assert_eq!(parse("foreground_agent"), Some(FOREGROUND_AGENT));
            assert_eq!(parse("urgent"), None);
            assert_eq!(parse("4"), None);
        }

        #[test]
        fn class_lanes_spec_parses_and_refuses_bad_caps() {
            assert_eq!(parse_class_lanes("", 8), Ok([0; 4]));
            assert_eq!(parse_class_lanes("background=2, agent=6", 8), Ok([0, 0, 6, 2]));
            assert_eq!(parse_class_lanes("interactive=8", 8), Ok([8, 0, 0, 0]));
            assert!(parse_class_lanes("background", 8).unwrap_err().contains("class=N"));
            assert!(parse_class_lanes("urgent=1", 8).unwrap_err().contains("unknown class"));
            assert!(parse_class_lanes("agent=-1", 8).unwrap_err().contains("whole number"));
            assert!(parse_class_lanes("agent=9", 8).unwrap_err().contains("exceeds"));
            assert!(parse_class_lanes("background=0", 8).unwrap_err().contains("at least 1"));
            assert!(parse_class_lanes("agent=1,foreground=2", 8).unwrap_err().contains("twice"));
        }
    }
}

pub use codec::{MockCodec, NoCodec, TextCodec, Utf8Stream};
pub use superfluid_engine::StageOutput;
pub use runtime::{EngineHost, WorkerSpec};
pub use store::{CommittedEvent, SessionStore};
pub use wal::{EventBody, GenParams};

pub use codec::BundleCodec;

#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("wal encode/decode: {0}")]
    Codec(#[from] postcard::Error),
    #[error("wal record failed checksum")]
    WalCorrupt,
    #[error("unknown session {0}")]
    UnknownSession(u64),
    #[error("session {0} has no content to generate from")]
    EmptySession(u64),
    #[error(
        "this model's maximum context length is {max} tokens, but your request has {len} \
         tokens; shorten the prompt or drop earlier turns"
    )]
    StreamTooLong { len: u64, max: u64 },
    #[error("model load: {0}")]
    EngineLoad(String),
    #[error("engine link: {0}")]
    Agent(#[from] superfluid_agent::AgentError),
    #[error("fleet link (Link F): {0}")]
    Fleet(#[from] superfluid_linkf::LinkFError),
    #[error("fleet node {addr}: {why}")]
    FleetNode { addr: String, why: String },
    #[error("engine rejected the plan with status {0}")]
    Rejected(i32),
    #[error("generation failed: {0}")]
    Generation(String),
    #[error("session {0} is generating; append/generate after it finishes or cancel it")]
    SessionBusy(u64),
    #[error("tool call {0} is unknown or already closed")]
    UnknownToolCall(u64),
    #[error("unknown message role {0}")]
    UnknownRole(u32),
    #[error("tool {index} is not a function declaration: {why}")]
    InvalidTool { index: usize, why: String },
    #[error("session {0} already has a prompt; AppendSystem must come before any other turn")]
    SystemNotFirst(u64),
    #[error("fork point {fork_at} is outside session {session}'s committed events (1..={max})")]
    ForkPoint { session: u64, fork_at: u64, max: u64 },
    #[error("tool call {0} belongs to an ancestor's timeline; only that session can close it")]
    InheritedToolCall(u64),
    #[error("session {session} metadata version is {actual}, command expected {expected}")]
    MetaConflict { session: u64, expected: u64, actual: u64 },
    #[error("session {session} generation is {actual}, command expected {expected}")]
    GenerationConflict { session: u64, expected: u64, actual: u64 },
    #[error("session {0} was purged")]
    Purged(u64),
    #[error("files quota exceeded: {stored} bytes stored, {incoming} more would pass the {max} limit")]
    FilesQuota { stored: u64, incoming: u64, max: u64 },
    #[error("rebase edit spec invalid: {0}")]
    RebaseSpec(&'static str),
    #[error("session {0} did not quiesce for purge (lane still running)")]
    PurgeTimeout(u64),
    #[error("permission request {0} is unknown or already answered")]
    UnknownPermission(u64),
    #[error("session {0} has an open permission request; answer it before generating")]
    PermissionPending(u64),
    #[error("tool call {0}'s executor lease expired; the result was recorded as expired, not applied")]
    ToolLeaseExpired(u64),
    #[error("tool call {0} already reached a terminal outcome; the late result was recorded as a reconciliation")]
    ToolLateEffect(u64),
    #[error("unknown block kind {0}")]
    UnknownBlockKind(u32),
    #[error("media blob {0} is not in the pool (put it first)")]
    UnknownMedia(String),
    #[error("image expands to {0} placeholders, beyond one prefill chunk")]
    MediaTooLarge(u32),
    #[error("this codec has no FIM dialect")]
    NoFimDialect,
    #[error("completion rate limit exceeded; retry in {retry_after_ms} ms")]
    CompletionThrottled { retry_after_ms: u64 },
    #[error("protocol: {0}")]
    Protocol(&'static str),
    #[error("constraint unsatisfiable: {0}")]
    Constraint(String),
    #[error("the model's chat template refused this conversation: {0}")]
    TemplateRefused(String),
    #[error("{0}")]
    Unsupported(Refusal),
    #[error("configuration: {0}")]
    Config(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub code: &'static str,
    pub param: Option<String>,
    pub message: String,
}

impl Refusal {
    pub fn new(code: &'static str, param: Option<&str>, message: impl Into<String>) -> Refusal {
        Refusal { code, param: param.map(str::to_string), message: message.into() }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SessionSummary {
    pub id: u64,
    pub parent: Option<u64>,
    pub fork_at: u64,
    pub title: Option<String>,
    pub archived: bool,
    pub meta_version: u64,
    pub generation: u64,
    pub events: u64,
    pub tokens: u64,
    pub open_tool_calls: u64,
}

#[derive(Debug)]
pub struct GenerateOutcome {
    pub events: Vec<CommittedEvent>,
    pub tokens_generated: u32,
    pub finish: u32,
    pub warm_prefix: u64,
    pub logprobs: Vec<scheduler::TokenLogprob>,
    pub spec: (u64, u64),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelSamplingDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub do_sample: Option<bool>,
    pub repetition_penalty: Option<f32>,
}

impl ModelSamplingDefaults {
    pub fn from_json(v: &serde_json::Value) -> Self {
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
        Self {
            temperature: f("temperature"),
            top_p: f("top_p"),
            top_k: v.get("top_k").and_then(|x| x.as_u64()).map(|x| x as u32),
            min_p: f("min_p"),
            do_sample: v.get("do_sample").and_then(|x| x.as_bool()),
            repetition_penalty: f("repetition_penalty"),
        }
    }
}

thread_local! {
    static SYNC_CLASS: std::cell::Cell<Option<u8>> = const { std::cell::Cell::new(None) };
}

pub fn with_sync_class<R>(class: Option<u8>, f: impl FnOnce() -> R) -> R {
    let _scope = SyncClassScope::enter(class);
    f()
}

pub struct SyncClassScope {
    prev: Option<Option<u8>>,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl SyncClassScope {
    pub fn enter(class: Option<u8>) -> SyncClassScope {
        let prev = class.map(|c| SYNC_CLASS.with(|cell| cell.replace(Some(c))));
        SyncClassScope { prev, _not_send: std::marker::PhantomData }
    }
}

impl Drop for SyncClassScope {
    fn drop(&mut self) {
        if let Some(prev) = self.prev {
            SYNC_CLASS.with(|c| c.set(prev));
        }
    }
}

fn ambient_sync_class() -> Option<u8> {
    SYNC_CLASS.with(|c| c.get())
}

struct Serving {
    capabilities: runtime::SharedCapabilities,
    retired: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    model_sampling: std::sync::OnceLock<ModelSamplingDefaults>,
    codec: std::sync::Arc<dyn TextCodec + Send + Sync>,
    sched: scheduler::SchedulerHandle,
    speculation: std::sync::Arc<scheduler::Speculation>,
    staged_tokens: u64,
}

/// Sessions whose generation should stop. A cancel also asks the running
/// tick to end at its next step boundary, so the lane stops within a step;
/// a queued chat asks the same.
#[derive(Default)]
pub struct CancelSet {
    set: std::sync::Mutex<std::collections::HashSet<u64>>,
    wake: std::sync::RwLock<Option<runtime::YieldSlot>>,
    /// When a latency-sensitive request last arrived (`monotonic_ms`; 0:
    /// never).
    latency_at_ms: std::sync::atomic::AtomicU64,
}

/// Milliseconds on a monotonic clock that starts at 1 with the process: a
/// wall clock stepped back would hold latency mode on until it caught up.
fn monotonic_ms() -> u64 {
    static BASE: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    BASE.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}

impl std::ops::Deref for CancelSet {
    type Target = std::sync::Mutex<std::collections::HashSet<u64>>;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

impl CancelSet {
    pub fn cancel(&self, session: u64) {
        self.set.lock().expect("cancel registry").insert(session);
        self.wake();
    }

    /// Asks the running tick to end at its next step boundary.
    pub fn wake(&self) {
        if let Some(slot) = self.wake.read().expect("wake slot").as_ref() {
            runtime::raise_yield(slot);
        }
    }

    pub(crate) fn wake_with(&self, slot: runtime::YieldSlot) {
        *self.wake.write().expect("wake slot") = Some(slot);
    }

    /// A chat or an inline completion arrived: the worker runs short
    /// prefill steps for a while, and the running tick ends at its next one.
    pub fn latency_request(&self) {
        self.latency_at_ms.store(monotonic_ms(), std::sync::atomic::Ordering::Release);
        if let Some(slot) = self.wake.read().expect("wake slot").as_ref() {
            runtime::set_latency(slot, true);
        }
        self.wake();
    }

    /// Whether a latency-sensitive request arrived within `hold`.
    pub(crate) fn latency_recent(&self, hold: std::time::Duration) -> bool {
        let at = self.latency_at_ms.load(std::sync::atomic::Ordering::Acquire);
        at > 0 && monotonic_ms().saturating_sub(at) < hold.as_millis() as u64
    }
}

struct Ground<'a> {
    store: &'a std::sync::Arc<std::sync::Mutex<SessionStore>>,
    cancels: &'a std::sync::Arc<CancelSet>,
    active: &'a std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u64>>>,
    stats: &'a std::sync::Arc<scheduler::SchedStats>,
    background_tick_divisor: &'a std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl Serving {
    fn start(
        host: EngineHost,
        codec: std::sync::Arc<dyn TextCodec + Send + Sync>,
        ground: Ground<'_>,
        opts: DaemonOptions,
    ) -> Result<Serving, DaemonError> {
        let mut opts = opts;
        let mut speculation = std::sync::Arc::new(scheduler::Speculation::default());
        let mut host = host;
        ground.cancels.wake_with(host.yield_slot());
        ground
            .stats
            .kv_bits
            .store(host.kv_bits() as u64, std::sync::atomic::Ordering::Relaxed);
        if let Some(first) = opts.speculate.clone() {
            let mut candidates = vec![first];
            if opts.speculate_auto {
                candidates.extend(opts.speculate_fallbacks.iter().cloned());
            }
            let picked = first_registrable(
                &candidates,
                opts.speculate_auto,
                |strategy| {
                    speculation = std::sync::Arc::new(scheduler::Speculation::default());
                    scheduler::Speculation::register(&speculation, &mut host, strategy, &opts.speculation)
                },
                |strategy, e, next| {
                    eprintln!(
                        "superfluid: speculation: auto picked {strategy}, but the engine refused it ({e}) — {}",
                        match next {
                            Some(n) => format!("trying {n}"),
                            None => "decoding plainly".to_string(),
                        }
                    );
                },
            )?;
            if picked.is_none() {
                speculation = std::sync::Arc::new(scheduler::Speculation::default());
            } else if picked != opts.speculate {
                eprintln!("superfluid: speculation: {}", picked.as_deref().unwrap_or_default());
            }
            opts.speculate = picked;
        }
        let staged_tokens = host.staged_tokens();
        let capabilities = host.capabilities();
        let retired = std::sync::Arc::new(std::sync::Mutex::new(None));
        let sched = scheduler::Scheduler::spawn(
            host,
            std::sync::Arc::clone(ground.store),
            std::sync::Arc::clone(&codec),
            std::sync::Arc::clone(ground.cancels),
            std::sync::Arc::clone(ground.active),
            std::sync::Arc::clone(ground.stats),
            std::sync::Arc::clone(ground.background_tick_divisor),
            std::sync::Arc::clone(&speculation),
            std::sync::Arc::clone(&retired),
            opts,
        );
        {
            let st = ground.store.lock().expect("store");
            let now = wal::now_unix_ms();
            for id in st.session_ids() {
                let Ok(s) = st.session(id) else { continue };
                if s.pin_deadline_unix_ms > now {
                    let (tokens, want) = pins::session_target(&s.tokens);
                    let _ = sched.tx.send(scheduler::SchedCmd::Pin(pins::PinReq {
                        key: pins::PinKey::Session(id),
                        tokens,
                        want,
                        deadline_unix_ms: s.pin_deadline_unix_ms,
                        anchor: Some(id),
                        reply: None,
                    }));
                }
            }
        }
        Ok(Serving { capabilities, retired, model_sampling: std::sync::OnceLock::new(), codec, sched, speculation, staged_tokens })
    }
}

pub struct Daemon {
    serving: std::sync::RwLock<std::sync::Arc<Serving>>,
    reloading: std::sync::Mutex<()>,
    reload_hook: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>>,
    runtime_id: Option<&'static str>,
    store: std::sync::Arc<std::sync::Mutex<SessionStore>>,
    cancels: std::sync::Arc<CancelSet>,
    active: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u64>>>,
    purging: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u64>>>,
    stats: std::sync::Arc<scheduler::SchedStats>,
    trace_scope: otlp::TraceScope,
    background_tick_divisor: std::sync::Arc<std::sync::atomic::AtomicU32>,
    park_dir: Option<std::path::PathBuf>,
    tool_lease_ms: u64,
    media: media::MediaPool,
    files: files::FileStore,
    batches: std::sync::Arc<batches::BatchStore>,
    responses: response_store::ResponseStore,
    completion: std::sync::Arc<CompletionSatellite>,
    fim_satellite: std::sync::OnceLock<FimSatellite>,
    built_with: DaemonOptions,
}

impl CompletionSatellite {
    fn cache_get(&self, key: u64) -> Option<(Vec<u32>, String, u32)> {
        let mut c = self.cache.lock().expect("completion cache");
        if let Some(v) = c.map.get(&key).cloned() {
            c.lru.retain(|&k| k != key);
            c.lru.push_back(key);
            Some(v)
        } else {
            None
        }
    }
    fn cache_put(&self, key: u64, tokens: Vec<u32>, text: String, finish: u32) {
        let mut c = self.cache.lock().expect("completion cache");
        if c.map.insert(key, (tokens, text, finish)).is_none() {
            c.lru.push_back(key);
        }
        while c.lru.len() > self.cache_cap {
            if let Some(old) = c.lru.pop_front() {
                c.map.remove(&old);
            }
        }
    }
}

struct FimSatellite {
    name: String,
    daemon: std::sync::Arc<Daemon>,
}

#[derive(Default)]
struct CompletionCounters {
    generated: std::sync::atomic::AtomicU64,
    cached: std::sync::atomic::AtomicU64,
    expired: std::sync::atomic::AtomicU64,
    throttled: std::sync::atomic::AtomicU64,
}

struct CompletionSatellite {
    store: std::sync::Mutex<SessionStore>,
    cache: std::sync::Mutex<CompletionCache>,
    deadline_ms: u64,
    cache_cap: usize,
    bucket: completion_bucket::BucketConfig,
    counters: CompletionCounters,
}

#[derive(Default)]
struct CompletionCache {
    map: std::collections::HashMap<u64, (Vec<u32>, String, u32)>,
    lru: std::collections::VecDeque<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    pub text: String,
    pub tokens: Vec<u32>,
    pub cached: bool,
    pub expired: bool,
    pub finish: u32,
    pub prompt_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct FimRequest {
    pub prefix: String,
    pub suffix: String,
    pub mode: u8,
    pub max_tokens: u32,
    pub params: GenParams,
    pub extras: scheduler::GenExtras,
}

impl FimRequest {
    pub fn greedy(prefix: &str, suffix: &str, mode: u8, max_tokens: u32) -> FimRequest {
        FimRequest {
            prefix: prefix.to_string(),
            suffix: suffix.to_string(),
            mode,
            max_tokens,
            params: GenParams::default(),
            extras: scheduler::GenExtras::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Tier {
    Resident,
    Parked { covered: u64, encoding: u8 },
    Cold,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Inspection {
    pub summary: SessionSummary,
    pub params: GenParams,
    pub epoch: u64,
    pub last_finish: u32,
    pub qos_class: u8,
    pub batch_invariant: bool,
    pub pin_deadline_unix_ms: u64,
    pub behavior_fingerprint: Option<u64>,
    pub open_tool_calls: Vec<(u64, String, String)>,
    pub tier: Tier,
    pub tokens: Vec<u32>,
    pub events: Vec<api::EventMsg>,
    pub wal_version: u8,
    #[serde(default, deserialize_with = "de_trailing_opt")]
    pub store_id: Option<store::StoreId>,
}

fn de_trailing_opt<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    use std::cell::Cell;
    struct V<'a, T>(&'a Cell<bool>, std::marker::PhantomData<T>);
    impl<'de, T: serde::Deserialize<'de>> serde::de::Visitor<'de> for V<'_, T> {
        type Value = Option<T>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("an optional trailing field")
        }
        fn visit_none<E: serde::de::Error>(self) -> Result<Option<T>, E> {
            self.0.set(true);
            Ok(None)
        }
        fn visit_some<D2: serde::Deserializer<'de>>(self, d: D2) -> Result<Option<T>, D2::Error> {
            self.0.set(true);
            T::deserialize(d).map(Some)
        }
    }
    let reached = Cell::new(false);
    match d.deserialize_option(V(&reached, std::marker::PhantomData)) {
        Ok(v) => Ok(v),
        Err(e) if !reached.get() && e.to_string() == POSTCARD_UNEXPECTED_END => Ok(None),
        Err(e) => Err(e),
    }
}

const POSTCARD_UNEXPECTED_END: &str = "Hit the end of buffer, expected more data";

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeculationOptions {
    pub max_draft: Option<u32>,
    pub adaptive: Option<bool>,
    pub min_yield: Option<f64>,
    pub yield_rounds: Option<u32>,
    pub throughput_gate: Option<bool>,
    pub gate_probe_tokens: Option<u32>,
    pub min_speedup: Option<f64>,
    pub gate_reprobe: Option<u32>,
    pub gate_reprobe_max: Option<u32>,
    pub dspark_confidence: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct DaemonOptions {
    pub max_lanes: usize,
    pub park_dir: Option<std::path::PathBuf>,
    pub pressure_high_pct: u8,
    pub pressure_low_pct: u8,
    pub pin_budget_pct: u8,
    pub park_lossy: bool,
    pub park_budget_bytes: u64,
    pub class_lanes: [usize; 4],
    pub tick_decode_budget: u32,
    pub tick_target_ms: u32,
    pub runtime_id: Option<&'static str>,
    pub prefill_budget: u32,
    pub agent_starvation_ticks: u32,
    pub pressure_source: pressure::PressureConfig,
    pub completion_deadline_ms: u64,
    pub completion_bucket: completion_bucket::BucketConfig,
    pub media_dir: Option<std::path::PathBuf>,
    pub files_max_bytes: Option<u64>,
    pub files_expiry_secs: Option<u64>,
    pub tool_lease_ms: u64,
    pub speculate: Option<String>,
    pub speculate_auto: bool,
    pub speculate_fallbacks: Vec<String>,
    pub speculation: SpeculationOptions,
    pub trace_scope: otlp::TraceScope,
}

impl Default for DaemonOptions {
    fn default() -> DaemonOptions {
        DaemonOptions {
            max_lanes: 8,
            park_dir: None,
            pressure_high_pct: 85,
            pressure_low_pct: 70,
            pin_budget_pct: 50,
            park_lossy: false,
            park_budget_bytes: 20 << 30,
            class_lanes: [0; 4],
            tick_decode_budget: 0,
            tick_target_ms: 0,
            runtime_id: None,
            prefill_budget: scheduler::PREFILL_BUDGET,
            agent_starvation_ticks: 8,
            pressure_source: pressure::PressureConfig::None,
            tool_lease_ms: 0,
            speculate: None,
            speculate_auto: false,
            speculate_fallbacks: Vec::new(),
            speculation: SpeculationOptions::default(),
            files_max_bytes: None,
            files_expiry_secs: None,
            media_dir: None,
            completion_deadline_ms: 0,
            trace_scope: otlp::TraceScope::default(),
            completion_bucket: completion_bucket::BucketConfig::default(),
        }
    }
}

struct PurgeFence {
    set: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u64>>>,
    ids: Vec<u64>,
}

impl Drop for PurgeFence {
    fn drop(&mut self) {
        let mut p = self.set.lock().expect("purging");
        for id in &self.ids {
            p.remove(id);
        }
    }
}

pub(crate) fn render_refusal(
    codec: &dyn TextCodec,
    messages: &[codec::ChatMessage],
    tools: &[String],
    kwargs: &serde_json::Map<String, serde_json::Value>,
    fallback: &'static str,
) -> DaemonError {
    match codec.template_refusal(messages, tools, kwargs) {
        Some(why) => DaemonError::TemplateRefused(why),
        None => DaemonError::Protocol(fallback),
    }
}

impl Daemon {
    pub fn new(
        store: SessionStore,
        host: EngineHost,
        codec: Box<dyn TextCodec + Send + Sync>,
        max_lanes: usize,
    ) -> Daemon {
        Daemon::with_park(store, host, codec, max_lanes, None)
    }

    pub fn with_park(
        store: SessionStore,
        host: EngineHost,
        codec: Box<dyn TextCodec + Send + Sync>,
        max_lanes: usize,
        park_dir: Option<std::path::PathBuf>,
    ) -> Daemon {
        Daemon::with_options(
            store,
            host,
            codec,
            DaemonOptions {
                max_lanes,
                park_dir,
                ..DaemonOptions::default()
            },
        )
        .expect("default watermarks are valid")
    }

    pub fn with_options(
        store: SessionStore,
        host: EngineHost,
        codec: Box<dyn TextCodec + Send + Sync>,
        opts: DaemonOptions,
    ) -> Result<Daemon, DaemonError> {
        if opts.pressure_high_pct == 0 || opts.pressure_high_pct > 100 {
            return Err(DaemonError::Config(
                "pressure_high_pct must be in 1..=100",
            ));
        }
        if opts.pressure_low_pct >= opts.pressure_high_pct {
            return Err(DaemonError::Config(
                "pressure_low_pct must be below pressure_high_pct",
            ));
        }
        if opts.max_lanes == 0 {
            return Err(DaemonError::Config("max_lanes must be at least 1"));
        }
        let mut opts = opts;
        let runtime_id = opts.runtime_id;
        opts.pin_budget_pct = opts.pin_budget_pct.min(opts.pressure_low_pct);
        opts.completion_bucket.validate().map_err(DaemonError::Config)?;
        if matches!(opts.speculation.max_draft, Some(0 | 16..)) {
            return Err(DaemonError::Config("speculation max_draft must be in 1..=15"));
        }
        if opts.speculation.min_yield.is_some_and(|v| !v.is_finite() || v < 0.0) {
            return Err(DaemonError::Config("speculation min_yield must be finite and non-negative"));
        }
        if opts.speculation.yield_rounds == Some(0) {
            return Err(DaemonError::Config("speculation yield_rounds must be at least 1"));
        }
        if opts.speculation.min_speedup.is_some_and(|v| !v.is_finite() || v < 1.0) {
            return Err(DaemonError::Config(
                "speculation min_speedup must be finite and at least 1.0",
            ));
        }
        if opts.speculation.dspark_confidence.is_some_and(|v| !v.is_finite() || v <= 0.0) {
            return Err(DaemonError::Config(
                "speculation dspark_confidence must be finite and positive",
            ));
        }
        opts.trace_scope.ns = export::store_trace_ns(&store.store_id());
        let store = std::sync::Arc::new(std::sync::Mutex::new(store));
        let built_with = opts.clone();
        let cancels = std::sync::Arc::new(CancelSet::default());
        let active = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        let purging = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        let stats = std::sync::Arc::new(scheduler::SchedStats::default());
        let background_tick_divisor = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(1));
        let park_dir = opts.park_dir.clone();
        let tool_lease_ms = opts.tool_lease_ms;
        let files_max_bytes = opts.files_max_bytes;
        let files_expiry_secs = opts.files_expiry_secs;
        let completion_deadline_ms = opts.completion_deadline_ms;
        let completion_bucket = opts.completion_bucket;
        let media_dir = opts.media_dir.clone().unwrap_or_else(|| {
            park_dir
                .as_ref()
                .and_then(|p| p.parent().map(|d| d.join("media")))
                .unwrap_or_else(|| {
                    let uniq = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0);
                    std::env::temp_dir().join(format!("superfluid-{}-{}", std::process::id(), uniq)).join("media")
                })
        });
        let trace_scope = opts.trace_scope.clone();
        let serving = Serving::start(
            host,
            std::sync::Arc::from(codec),
            Ground {
                store: &store,
                cancels: &cancels,
                active: &active,
                stats: &stats,
                background_tick_divisor: &background_tick_divisor,
            },
            opts,
        )?;
        Ok(Daemon {
            trace_scope,
            serving: std::sync::RwLock::new(std::sync::Arc::new(serving)),
            reloading: std::sync::Mutex::new(()),
            reload_hook: std::sync::OnceLock::new(),
            runtime_id,
            store,
            cancels,
            active,
            purging,
            stats,
            background_tick_divisor,
            park_dir,
            tool_lease_ms,
            files: files::FileStore::with_limits(
                files::files_dir_for(media_dir.parent().unwrap_or(media_dir.as_path())),
                files_max_bytes,
                files_expiry_secs,
            ),
            batches: std::sync::Arc::new(batches::BatchStore::new(batches::batches_dir_for(
                media_dir.parent().unwrap_or(media_dir.as_path()),
            ))),
            responses: response_store::ResponseStore::new(response_store::responses_dir_for(
                media_dir.parent().unwrap_or(media_dir.as_path()),
            )),
            media: media::MediaPool::new(media_dir),
            completion: std::sync::Arc::new(CompletionSatellite {
                store: std::sync::Mutex::new(SessionStore::ephemeral()),
                cache: std::sync::Mutex::new(CompletionCache::default()),
                deadline_ms: completion_deadline_ms,
                cache_cap: 256,
                bucket: completion_bucket,
                counters: CompletionCounters::default(),
            }),
            fim_satellite: std::sync::OnceLock::new(),
            built_with,
        })
    }

    fn serving(&self) -> std::sync::Arc<Serving> {
        std::sync::Arc::clone(&self.serving.read().expect("serving"))
    }

    fn codec(&self) -> std::sync::Arc<dyn TextCodec + Send + Sync> {
        std::sync::Arc::clone(&self.serving().codec)
    }

    fn sched_tx(&self) -> std::sync::mpsc::Sender<scheduler::SchedCmd> {
        self.serving().sched.tx.clone()
    }

    pub fn reload_with(
        &self,
        build: impl FnOnce() -> Result<(EngineHost, Box<dyn TextCodec + Send + Sync>), DaemonError>,
    ) -> Result<(), DaemonError> {
        let _one_at_a_time = self.reloading.lock().expect("reloading");
        self.shutdown();
        {
            let mut cancels = self.cancels.lock().expect("cancel registry");
            let active = self.active.lock().expect("active set");
            cancels.retain(|session| active.contains(session));
        }
        let started = build().and_then(|(host, codec)| {
            Serving::start(
                host,
                std::sync::Arc::from(codec),
                Ground {
                    store: &self.store,
                    cancels: &self.cancels,
                    active: &self.active,
                    stats: &self.stats,
                    background_tick_divisor: &self.background_tick_divisor,
                },
                self.built_with.clone(),
            )
        });
        match started {
            Ok(serving) => {
                *self.serving.write().expect("serving") = std::sync::Arc::new(serving);
                Ok(())
            }
            Err(e) => {
                let stopped = self.serving();
                let mut why = stopped.retired.lock().expect("retired");
                if why.is_none() {
                    *why = Some(format!("the model's engine was stopped to load it again, which failed ({e}); it loads again on its next request, so retry"));
                }
                Err(e)
            }
        }
    }

    pub fn on_retired(&self, reload: Box<dyn Fn() + Send + Sync>) {
        let _ = self.reload_hook.set(reload);
    }

    fn ensure_serving(&self) -> Result<(), DaemonError> {
        if self.needs_reload().is_none() {
            return Ok(());
        }
        if let Some(reload) = self.reload_hook.get() {
            reload();
        }
        match self.needs_reload() {
            None => Ok(()),
            Some(why) => Err(DaemonError::Generation(why)),
        }
    }

    pub fn max_stream_tokens(&self) -> u64 {
        let serving = self.serving();
        let record = serving.capabilities.read().expect("capability record");
        runtime::stream_ceiling(serving.staged_tokens, &record)
    }

    fn check_stream_fits(&self, len: usize) -> Result<(), DaemonError> {
        let len = len as u64;
        let max = self.max_stream_tokens();
        if max > 0 && len > max {
            return Err(DaemonError::StreamTooLong { len, max });
        }
        Ok(())
    }

    pub fn session_fits(&self, session: u64, thinking: Option<bool>) -> Result<(), DaemonError> {
        self.ensure_serving()?;
        let len = {
            let store = self.store.lock().expect("store");
            let s = store.session(session)?;
            s.tokens.len() + self.generation_opener(&s.events, thinking).map_or(0, |op| op.len())
        };
        self.check_stream_fits(len)
    }

    pub fn fim_fits(&self, req: &FimRequest) -> Result<(), DaemonError> {
        let d = match self.fim_satellite.get() {
            Some(sat) => &sat.daemon,
            None => self,
        };
        d.ensure_serving()?;
        let fim = d
            .codec()
            .render_fim(&req.prefix, &req.suffix, req.mode)
            .ok_or(DaemonError::NoFimDialect)?;
        d.check_stream_fits(fim.len())
    }

    pub fn complete(
        &self,
        prefix: &str,
        suffix: &str,
        mode: u8,
        max_tokens: u32,
    ) -> Result<Completion, DaemonError> {
        self.complete_streaming(&FimRequest::greedy(prefix, suffix, mode, max_tokens), |_, _| {
            std::ops::ControlFlow::Continue(())
        })
    }

    pub fn complete_streaming(
        &self,
        req: &FimRequest,
        on_text: impl FnMut(&str, usize) -> std::ops::ControlFlow<()>,
    ) -> Result<Completion, DaemonError> {
        let result = match self.fim_satellite.get() {
            Some(sat) => sat.daemon.complete_here(req, on_text),
            None => self.complete_here(req, on_text),
        };
        if let Ok(c) = &result {
            use std::sync::atomic::Ordering::Relaxed;
            let n = &self.completion.counters;
            if c.expired {
                n.expired.fetch_add(1, Relaxed);
            } else if c.cached {
                n.cached.fetch_add(1, Relaxed);
            } else {
                n.generated.fetch_add(1, Relaxed);
            }
        }
        result
    }

    pub fn attach_fim_satellite(
        &self,
        name: impl Into<String>,
        satellite: std::sync::Arc<Daemon>,
    ) -> Result<(), DaemonError> {
        if satellite.codec().render_fim("", "", codec::fim_mode::PSM).is_none() {
            return Err(DaemonError::NoFimDialect);
        }
        if satellite.fim_satellite.get().is_some() {
            return Err(DaemonError::Config("a FIM satellite cannot itself route to another"));
        }
        self.fim_satellite
            .set(FimSatellite {
                name: name.into(),
                daemon: satellite,
            })
            .map_err(|_| DaemonError::Config("a FIM satellite is already attached"))
    }

    pub fn fim_satellite_name(&self) -> Option<String> {
        self.fim_satellite.get().map(|s| s.name.clone())
    }

    pub fn fim_satellite(&self) -> Option<std::sync::Arc<Daemon>> {
        self.fim_satellite.get().map(|s| std::sync::Arc::clone(&s.daemon))
    }

    pub fn fim_supported(&self) -> bool {
        self.fim_satellite.get().is_some()
            || self.codec().render_fim("", "", codec::fim_mode::PSM).is_some()
    }

    pub fn completion_bucket(&self) -> completion_bucket::BucketConfig {
        self.completion.bucket
    }

    pub fn note_completion_throttled(&self) {
        self.completion
            .counters
            .throttled
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn complete_here(
        &self,
        req: &FimRequest,
        mut on_text: impl FnMut(&str, usize) -> std::ops::ControlFlow<()>,
    ) -> Result<Completion, DaemonError> {
        self.ensure_serving()?;
        let _span = self.raw_generate_span(req.max_tokens);
        let submitted = std::time::Instant::now();
        let fim = self
            .codec()
            .render_fim(&req.prefix, &req.suffix, req.mode)
            .ok_or(DaemonError::NoFimDialect)?;
        self.check_stream_fits(fim.len())?;
        let prompt_tokens = fim.len() as u32;
        let expired = || Completion {
            text: String::new(),
            tokens: Vec::new(),
            cached: false,
            expired: true,
            finish: superfluid_abi::finish::NONE,
            prompt_tokens,
        };
        let cacheable = req.params.temperature <= 0.0;
        let key = {
            let mut h = superfluid_fingerprint::RecordHasher::new();
            for t in &fim {
                h.field_u32(1, *t);
            }
            h.field_u32(2, req.mode as u32);
            h.field_u32(3, req.max_tokens);
            if let Some(fp) = self.codec().behavior_fingerprint() {
                h.field_u64(4, fp.digest());
            }
            let x = &req.extras;
            h.field_u32(5, x.presence_penalty.to_bits());
            h.field_u32(6, x.frequency_penalty.to_bits());
            h.field_u32(7, x.repeat_penalty.to_bits());
            h.field_u32(8, x.ignore_eos as u32);
            h.finish()
        };
        if cacheable {
            if let Some((tokens, text, finish)) = self.completion.cache_get(key) {
                let _ = on_text(&text, tokens.len());
                return Ok(Completion {
                    text,
                    tokens,
                    cached: true,
                    expired: false,
                    finish,
                    prompt_tokens,
                });
            }
        }
        if self.completion.deadline_ms > 0
            && submitted.elapsed().as_millis() as u64 > self.completion.deadline_ms
        {
            return Ok(expired());
        }
        let session = {
            let mut store = self.completion.store.lock().expect("completion store");
            let s = store.create(None, req.params)?;
            store.set_qos(s, qos::INLINE_COMPLETION, false)?;
            store.append(s, None, fim.clone())?;
            s
        };
        let (stream, params) = {
            let store = self.completion.store.lock().expect("completion store");
            let st = store.session(session)?;
            (st.tokens.clone(), st.params)
        };
        let mut produced: Vec<u32> = Vec::new();
        let mut emitted = 0usize;
        let result = scheduler::run_completion_streaming(
            &self.sched_tx(),
            stream,
            params,
            req.max_tokens,
            self.completion.deadline_ms,
            scheduler::GenExtras {
                grammar_handle: 0,
                logit_bias_handle: 0,
                want_logprobs: false,
                ..req.extras.clone()
            },
            |toks| {
                produced.extend_from_slice(toks);
                let text = self.codec().decode(&produced);
                let stable = text.trim_end_matches('\u{FFFD}');
                if stable.len() > emitted && stable.is_char_boundary(emitted) {
                    let piece = &stable[emitted..];
                    emitted = stable.len();
                    return on_text(piece, produced.len());
                }
                std::ops::ControlFlow::Continue(())
            },
        );
        {
            let mut store = self.completion.store.lock().expect("completion store");
            let _ = store.drop_session(session);
        }
        let (tokens, finish, was_expired) = result?;
        if was_expired {
            return Ok(expired());
        }
        let text = self.codec().decode(&tokens);
        if text.len() > emitted && text.is_char_boundary(emitted) {
            let _ = on_text(&text[emitted..], tokens.len());
        }
        if cacheable && finish != superfluid_abi::finish::CANCELLED {
            self.completion.cache_put(key, tokens.clone(), text.clone(), finish);
        }
        Ok(Completion {
            text,
            tokens,
            cached: false,
            expired: false,
            finish,
            prompt_tokens,
        })
    }

    pub fn generate_tokens(
        &self,
        stream: Vec<u32>,
        params: GenParams,
        max_tokens: u32,
        deadline_ms: u64,
    ) -> Result<(Vec<u32>, bool), DaemonError> {
        self.ensure_serving()?;
        let _span = self.raw_generate_span(max_tokens);
        self.check_stream_fits(stream.len())?;
        let deadline = self.effective_deadline(deadline_ms);
        let (tokens, _finish, expired) =
            scheduler::run_completion(&self.sched_tx(), stream, params, max_tokens, deadline, 0)?;
        Ok((tokens, expired))
    }

    pub fn generate_tokens_streaming(
        &self,
        stream: Vec<u32>,
        params: GenParams,
        max_tokens: u32,
        deadline_ms: u64,
        extras: scheduler::GenExtras,
        on_tokens: impl FnMut(&[u32]) -> std::ops::ControlFlow<()>,
    ) -> Result<(Vec<u32>, u32, bool), DaemonError> {
        self.ensure_serving()?;
        let _span = self.raw_generate_span(max_tokens);
        self.check_stream_fits(stream.len())?;
        let deadline = self.effective_deadline(deadline_ms);
        scheduler::run_completion_streaming(
            &self.sched_tx(),
            stream,
            params,
            max_tokens,
            deadline,
            extras,
            on_tokens,
        )
    }

    pub fn forward_stage(
        &self,
        tokens: Vec<u32>,
        start_layer: u32,
        end_layer: u32,
        hidden_in: Option<Vec<u8>>,
    ) -> Result<superfluid_engine::StageOutput, DaemonError> {
        self.ensure_serving()?;
        match self.engine_sync(proto_linkw::StateSyncReqMsg::ForwardStage {
            tokens,
            start_layer,
            end_layer,
            hidden_in,
        })? {
            proto_linkw::StateSyncOkMsg::ForwardStage {
                hidden: Some(h),
                token: None,
            } => Ok(superfluid_engine::StageOutput::Hidden(h)),
            proto_linkw::StateSyncOkMsg::ForwardStage {
                hidden: None,
                token: Some(t),
            } => Ok(superfluid_engine::StageOutput::Token(t)),
            _ => Err(DaemonError::Protocol("mismatched ForwardStage reply")),
        }
    }

    fn effective_deadline(&self, deadline_ms: u64) -> u64 {
        if deadline_ms == 0 {
            self.completion.deadline_ms
        } else if self.completion.deadline_ms == 0 {
            deadline_ms
        } else {
            deadline_ms.min(self.completion.deadline_ms)
        }
    }

    pub fn generate_tokens_constrained(
        &self,
        stream: Vec<u32>,
        params: GenParams,
        max_tokens: u32,
        deadline_ms: u64,
        json_schema: &str,
    ) -> Result<(Vec<u32>, bool), DaemonError> {
        self.ensure_serving()?;
        let _span = self.raw_generate_span(max_tokens);
        let handle = match self.engine_sync(proto_linkw::StateSyncReqMsg::CreateGrammar {
            json_schema: json_schema.to_string(),
        })? {
            proto_linkw::StateSyncOkMsg::CreateGrammar { grammar_handle } => grammar_handle,
            _ => return Err(DaemonError::Protocol("mismatched CreateGrammar reply")),
        };
        let deadline = self.effective_deadline(deadline_ms);
        let result =
            scheduler::run_completion(&self.sched_tx(), stream, params, max_tokens, deadline, handle);
        let _ = self.engine_sync(proto_linkw::StateSyncReqMsg::FreeGrammar {
            grammar_handle: handle,
        });
        let (tokens, finish, expired) = result?;
        if expired {
            return Err(DaemonError::Generation(format!(
                "constrained generation hit the wall deadline with the grammar incomplete                  ({} tokens)",
                tokens.len()
            )));
        }
        if finish != superfluid_abi::finish::GRAMMAR {
            return Err(DaemonError::Generation(format!(
                "constrained generation did not complete the grammar (finish={finish},                  {} tokens); raise max_tokens",
                tokens.len()
            )));
        }
        Ok((tokens, false))
    }

    pub fn put_media(&self, bytes: &[u8]) -> Result<String, DaemonError> {
        Ok(self.media.put(bytes)?)
    }

    pub fn media_pool(&self) -> &media::MediaPool {
        &self.media
    }

    fn engine_sync(&self, req: proto_linkw::StateSyncReqMsg) -> Result<proto_linkw::StateSyncOkMsg, DaemonError> {
        self.engine_sync_as(req, None)
    }

    fn engine_sync_as(
        &self,
        req: proto_linkw::StateSyncReqMsg,
        class: Option<u8>,
    ) -> Result<proto_linkw::StateSyncOkMsg, DaemonError> {
        let class = class.or_else(ambient_sync_class);
        let (tx, rx) = std::sync::mpsc::channel();
        let cmd = scheduler::SchedCmd::Sync(req, tx);
        self.sched_tx()
            .send(match class {
                Some(c) => scheduler::SchedCmd::SyncAs(c, Box::new(cmd)),
                None => cmd,
            })
            .map_err(|_| DaemonError::Protocol("scheduler is gone"))?;
        rx.recv()
            .map_err(|_| DaemonError::Protocol("scheduler dropped the query"))?
            .map_err(DaemonError::Generation)
    }

    pub fn append_image(
        &self,
        session: u64,
        role: u32,
        blob: &str,
        pre_text: &str,
        post_text: &str,
    ) -> Result<CommittedEvent, DaemonError> {
        if self.active.lock().expect("active set").contains(&session) {
            return Err(DaemonError::SessionBusy(session));
        }
        if !matches!(role, wal::role::SYSTEM | wal::role::USER | wal::role::ASSISTANT) {
            return Err(DaemonError::UnknownRole(role));
        }
        if !self.media.exists(blob) {
            return Err(DaemonError::UnknownMedia(blob.to_string()));
        }
        let path = self.media.path_of(blob);
        let info = match self.engine_sync(proto_linkw::StateSyncReqMsg::MediaProbe {
            image_path: path.to_string_lossy().into_owned(),
        })? {
            proto_linkw::StateSyncOkMsg::MediaProbe { info } => info,
            _ => return Err(DaemonError::Protocol("mismatched MediaProbe reply")),
        };
        if info.n_tokens == 0 || info.n_tokens as u64 > scheduler::PREFILL_BUDGET as u64 {
            return Err(DaemonError::MediaTooLarge(info.n_tokens));
        }
        let mut run = Vec::with_capacity(info.n_tokens as usize + 2);
        if info.boi_token_id != 0 {
            run.push(info.boi_token_id);
        }
        run.extend(std::iter::repeat_n(info.image_token_id, info.n_tokens as usize));
        if info.eoi_token_id != 0 {
            run.push(info.eoi_token_id);
        }
        let mut span = self.separator_for(session);
        let turn = self
            .codec()
            .render_turn_with_tokens(role, pre_text, &run, post_text)
            .ok_or(DaemonError::Protocol("codec has no chat dialect"))?;
        span.extend(turn);
        let first_in_span = span
            .iter()
            .position(|&t| t == info.image_token_id)
            .ok_or(DaemonError::Protocol("codec dropped the placeholder run"))?;
        let mut store = self.store.lock().expect("store");
        let stream_len = store.session(session)?.tokens.len() as u64;
        let payload = serde_json::json!({
            "blob": blob,
            "n_tokens": info.n_tokens,
            "image_token_id": info.image_token_id,
            "offset": stream_len + first_in_span as u64,
            "pre_text": pre_text,
            "post_text": post_text,
            "media_identity": {
                "preprocessing_fp": format!("{:016x}", info.preprocess_fp),
                "encoder": "bundle-weights-identity",
            },
        })
        .to_string();
        store.commit_block(
            session,
            role,
            wal::block_kind::IMAGE,
            payload,
            span,
            codec::MODEL_VISIBILITY_VERSION,
        )
    }

    fn media_of(&self, s: &store::SessionState) -> Vec<(u64, std::path::PathBuf, u32)> {
        let mut v = Vec::new();
        for e in &s.events {
            if let EventBody::Block {
                kind, payload, ..
            } = &e.body
            {
                if *kind != wal::block_kind::IMAGE {
                    continue;
                }
                let p: serde_json::Value = serde_json::from_str(payload).unwrap_or_default();
                let blob = p.get("blob").and_then(|b| b.as_str()).unwrap_or("");
                let off = p.get("offset").and_then(|o| o.as_u64()).unwrap_or(0);
                let n = p.get("n_tokens").and_then(|o| o.as_u64()).unwrap_or(0) as u32;
                if !blob.is_empty() && n > 0 {
                    v.push((off, self.media.path_of(blob), n));
                }
            }
        }
        v
    }

    pub fn gc_media(&self) -> Result<(usize, usize), DaemonError> {
        let store = self.store.lock().expect("store");
        let mut live = std::collections::HashSet::new();
        for id in store.session_ids() {
            let Ok(s) = store.session(id) else { continue };
            for e in &s.events {
                if let EventBody::Block { kind, payload, .. } = &e.body {
                    if *kind == wal::block_kind::IMAGE {
                        let p: serde_json::Value = serde_json::from_str(payload).unwrap_or_default();
                        if let Some(b) = p.get("blob").and_then(|b| b.as_str()) {
                            live.insert(b.to_string());
                        }
                    }
                }
            }
        }
        Ok(self.media.sweep(&live)?)
    }

    pub fn speculation(&self) -> Option<scheduler::SpeculationGrant> {
        self.serving().speculation.grant()
    }

    pub fn tool_lease_ms(&self) -> u64 {
        self.tool_lease_ms
    }

    pub fn inspect(&self, session: u64) -> Result<Inspection, DaemonError> {
        // The active set before the store, in the order a generation takes them.
        let resident = self.active.lock().expect("active set").contains(&session);
        let store = self.store.lock().expect("store");
        let s = store.session(session)?;
        let tier = if resident {
            Tier::Resident
        } else {
            match self
                .park_dir
                .as_deref()
                .and_then(|d| park::read(d, session))
            {
                Some(p) => Tier::Parked {
                    covered: p.covered,
                    encoding: p.encoding,
                },
                None => Tier::Cold,
            }
        };
        Ok(Inspection {
            summary: SessionSummary {
                id: s.id,
                parent: if s.rerooted { None } else { s.parent },
                fork_at: s.base,
                title: s.title.clone(),
                archived: s.archived,
                meta_version: s.meta_version,
                generation: s.generation,
                events: s.events.len() as u64,
                tokens: s.tokens.len() as u64,
                open_tool_calls: s.open_tool_calls.len() as u64,
            },
            params: s.params,
            epoch: s.epoch,
            last_finish: s.last_finish,
            qos_class: s.qos_class,
            batch_invariant: s.batch_invariant,
            pin_deadline_unix_ms: s.pin_deadline_unix_ms,
            behavior_fingerprint: s.behavior_fp,
            open_tool_calls: s
                .open_tool_calls
                .iter()
                .map(|(id, (n, a))| (*id, n.clone(), a.clone()))
                .collect(),
            tier,
            tokens: s.tokens.clone(),
            events: s.events.iter().cloned().map(Into::into).collect(),
            wal_version: store.wal_version(),
            store_id: Some(store.store_id()),
        })
    }

    pub fn files(&self) -> &files::FileStore {
        &self.files
    }

    pub fn batches(&self) -> std::sync::Arc<batches::BatchStore> {
        std::sync::Arc::clone(&self.batches)
    }

    pub fn responses(&self) -> &response_store::ResponseStore {
        &self.responses
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.codec().encode(text)
    }

    pub fn grammar_create(&self, json_schema: &str) -> Result<u32, DaemonError> {
        match self.engine_sync(proto_linkw::StateSyncReqMsg::CreateGrammar {
            json_schema: json_schema.to_string(),
        })? {
            proto_linkw::StateSyncOkMsg::CreateGrammar { grammar_handle } => Ok(grammar_handle),
            _ => Err(DaemonError::Protocol("mismatched CreateGrammar reply")),
        }
    }

    pub fn grammar_free(&self, grammar_handle: u32) {
        let _ = self.engine_sync(proto_linkw::StateSyncReqMsg::FreeGrammar { grammar_handle });
    }

    pub fn logit_bias_create(&self, tokens: &[i32], values: &[f32]) -> Result<u32, DaemonError> {
        match self.engine_sync(proto_linkw::StateSyncReqMsg::CreateLogitBias {
            tokens: tokens.to_vec(),
            values: values.iter().map(|v| v.to_bits()).collect(),
        })? {
            proto_linkw::StateSyncOkMsg::CreateLogitBias { handle } => Ok(handle),
            _ => Err(DaemonError::Protocol("mismatched CreateLogitBias reply")),
        }
    }
    pub fn logit_bias_free(&self, handle: u32) {
        let _ = self.engine_sync(proto_linkw::StateSyncReqMsg::FreeLogitBias { handle });
    }

    pub fn lora_load(&self, adapter_path: &str) -> Result<Option<String>, DaemonError> {
        self.lora_reply(proto_linkw::StateSyncReqMsg::LoraLoad {
            adapter_path: adapter_path.to_string(),
        })
    }

    pub fn lora_unload(&self) -> Result<Option<String>, DaemonError> {
        self.lora_reply(proto_linkw::StateSyncReqMsg::LoraUnload)
    }

    pub fn lora_id(&self) -> Result<Option<String>, DaemonError> {
        self.lora_reply(proto_linkw::StateSyncReqMsg::LoraId)
    }

    fn lora_reply(&self, req: proto_linkw::StateSyncReqMsg) -> Result<Option<String>, DaemonError> {
        match self.engine_sync(req)? {
            proto_linkw::StateSyncOkMsg::LoraId { id } => Ok(id),
            _ => Err(DaemonError::Protocol("mismatched LoraId reply")),
        }
    }

    pub fn try_model_sampling_defaults(&self) -> Option<ModelSamplingDefaults> {
        let serving = self.serving();
        if let Some(cached) = serving.model_sampling.get() {
            return Some(cached.clone());
        }
        let descriptor = self.capability_descriptor()?;
        let resolved = descriptor
            .get("sampling_defaults")
            .map(ModelSamplingDefaults::from_json)
            .unwrap_or_default();
        let _ = serving.model_sampling.set(resolved.clone());
        Some(resolved)
    }

    pub fn model_sampling_defaults(&self) -> ModelSamplingDefaults {
        self.try_model_sampling_defaults().unwrap_or_default()
    }

    pub fn capability_descriptor(&self) -> Option<serde_json::Value> {
        self.serving().capabilities.read().expect("capability record").with_runtime(self.runtime_id)
    }

    pub fn runtime_id(&self) -> Option<&'static str> {
        self.runtime_id
    }

    pub fn needs_reload(&self) -> Option<String> {
        self.serving().retired.lock().expect("retired").clone()
    }

    pub fn capabilities(&self) -> capabilities::Capabilities {
        self.serving().capabilities.read().expect("capability record").clone()
    }

    pub fn require(&self, section: &str, key: &str, code: &'static str, param: Option<&str>, what: &str) -> Result<(), DaemonError> {
        let why = match self.serving().capabilities.read().expect("capability record").get(section, key) {
            capabilities::Cap::No(why) => why.to_string(),
            capabilities::Cap::Yes | capabilities::Cap::Unknown => return Ok(()),
        };
        Err(DaemonError::Unsupported(Refusal::new(code, param, format!("{} {what}: {why}", self.described()))))
    }

    fn described(&self) -> String {
        let model = &self.trace_scope.model;
        let runtime = self.capability_descriptor().and_then(|r| r["runtime"]["id"].as_str().map(str::to_string));
        match (model.is_empty(), runtime) {
            (false, Some(r)) => format!("model '{model}' on runtime {r}"),
            (false, None) => format!("model '{model}'"),
            (true, Some(r)) => format!("this model (runtime {r})"),
            (true, None) => "this model".to_string(),
        }
    }

    pub fn chat_template(&self) -> Option<String> {
        self.codec().chat_template_source()
    }

    pub fn transcribe(
        &self,
        audio_path: &str,
        params: &superfluid_engine::TranscribeParams,
    ) -> Result<superfluid_engine::Transcription, DaemonError> {
        self.transcribe_as(audio_path, params, None)
    }

    pub fn transcribe_as(
        &self,
        audio_path: &str,
        params: &superfluid_engine::TranscribeParams,
        class: Option<u8>,
    ) -> Result<superfluid_engine::Transcription, DaemonError> {
        self.ensure_serving()?;
        self.require_transcription(params)?;
        Self::decode_transcription(self.engine_sync_as(Self::transcribe_req(audio_path, params, false), class)?)
    }

    fn require_transcription(&self, params: &superfluid_engine::TranscribeParams) -> Result<(), DaemonError> {
        if params.translate {
            self.require("modalities", "whisper_translate", "unsupported_model", Some("model"), "does not translate speech")
        } else {
            self.require("modalities", "whisper_transcribe", "unsupported_model", Some("model"), "does not transcribe speech")
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn transcribe_streaming(
        &self,
        audio_path: &str,
        params: &superfluid_engine::TranscribeParams,
    ) -> Result<
        (
            std::sync::mpsc::Receiver<proto_linkw::TranscribeSegmentMsg>,
            std::sync::mpsc::Receiver<Result<superfluid_engine::Transcription, DaemonError>>,
        ),
        DaemonError,
    > {
        self.transcribe_streaming_as(audio_path, params, None)
    }

    #[allow(clippy::type_complexity)]
    pub fn transcribe_streaming_as(
        &self,
        audio_path: &str,
        params: &superfluid_engine::TranscribeParams,
        class: Option<u8>,
    ) -> Result<
        (
            std::sync::mpsc::Receiver<proto_linkw::TranscribeSegmentMsg>,
            std::sync::mpsc::Receiver<Result<superfluid_engine::Transcription, DaemonError>>,
        ),
        DaemonError,
    > {
        self.ensure_serving()?;
        self.require_transcription(params)?;
        let (seg_tx, seg_rx) = std::sync::mpsc::channel();
        let (res_tx, res_rx) = std::sync::mpsc::channel();
        let (raw_tx, raw_rx) = std::sync::mpsc::channel();
        self.sched_tx()
            .send({
                let cmd = scheduler::SchedCmd::SyncStream(Self::transcribe_req(audio_path, params, true), seg_tx, raw_tx);
                match class.or_else(ambient_sync_class) {
                    Some(c) => scheduler::SchedCmd::SyncAs(c, Box::new(cmd)),
                    None => cmd,
                }
            })
            .map_err(|_| DaemonError::Protocol("scheduler is gone"))?;
        std::thread::spawn(move || {
            let out = match raw_rx.recv() {
                Ok(Ok(ok)) => Self::decode_transcription(ok),
                Ok(Err(e)) => Err(DaemonError::Generation(e)),
                Err(_) => Err(DaemonError::Protocol("scheduler dropped the query")),
            };
            let _ = res_tx.send(out);
        });
        Ok((seg_rx, res_rx))
    }

    fn transcribe_req(
        audio_path: &str,
        params: &superfluid_engine::TranscribeParams,
        stream: bool,
    ) -> proto_linkw::StateSyncReqMsg {
        proto_linkw::StateSyncReqMsg::Transcribe {
            audio_path: audio_path.to_string(),
            language: params.language.clone(),
            translate: params.translate,
            timestamps: params.timestamps,
            prompt: params.prompt.clone(),
            stream,
        }
    }

    fn decode_transcription(
        ok: proto_linkw::StateSyncOkMsg,
    ) -> Result<superfluid_engine::Transcription, DaemonError> {
        match ok {
            proto_linkw::StateSyncOkMsg::Transcribe { text, language, duration_ms, segments } => {
                Ok(superfluid_engine::Transcription {
                    text,
                    language,
                    duration_ms,
                    segments: segments
                        .into_iter()
                        .map(|s| superfluid_engine::TranscriptSegment {
                            start_ms: s.start_ms,
                            end_ms: s.end_ms,
                            text: s.text,
                            avg_logprob: f32::from_bits(s.avg_logprob_bits),
                            no_speech_prob: f32::from_bits(s.no_speech_prob_bits),
                            compression_ratio: f32::from_bits(s.compression_ratio_bits),
                            temperature: f32::from_bits(s.temperature_bits),
                        })
                        .collect(),
                })
            }
            _ => Err(DaemonError::Protocol("engine returned a non-Transcribe reply")),
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, DaemonError> {
        self.embed_as(text, None)
    }

    pub fn embed_as(&self, text: &str, class: Option<u8>) -> Result<Vec<f32>, DaemonError> {
        self.ensure_serving()?;
        self.require("workload", "embedding", "unsupported_model", Some("model"), "serves no embeddings")?;
        self.embed_tokens_as(self.codec().encode(text), class)
    }

    pub(crate) fn embed_tokens_as(&self, tokens: Vec<u32>, class: Option<u8>) -> Result<Vec<f32>, DaemonError> {
        self.ensure_serving()?;
        self.require("workload", "embedding", "unsupported_model", Some("model"), "serves no embeddings")?;
        match self.engine_sync_as(proto_linkw::StateSyncReqMsg::Embed { tokens }, class)? {
            proto_linkw::StateSyncOkMsg::Embed { embedding } => Ok(embedding
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()),
            _ => Err(DaemonError::Protocol("engine returned a non-Embed reply")),
        }
    }

    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.codec().decode(tokens)
    }

    fn raw_generate_span(&self, max_tokens: u32) -> tracing::span::EnteredSpan {
        tracing::info_span!(
            parent: None,
            "generate",
            max_tokens,
            raw = true,
            model = self.trace_scope.model.as_str(),
            store_ns = self.trace_scope.ns
        )
        .entered()
    }

    pub fn metrics_text(&self) -> String {
        let mut out = metrics::render_prometheus(&self.stats);
        out.push_str(&format!(
            "# HELP superfluid_telemetry_dropped_total Telemetry records dropped by the bounded sinks (log lines and OTLP spans).\n# TYPE superfluid_telemetry_dropped_total counter\nsuperfluid_telemetry_dropped_total {}\n",
            telemetry::dropped_total()
        ));
        let g = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
        out.push_str(&format!(
            "# HELP superfluid_otlp_spans_exported_total Operational spans accepted by the OTLP collector.\n# TYPE superfluid_otlp_spans_exported_total counter\nsuperfluid_otlp_spans_exported_total {}\n",
            g(&otlp::EXPORTED)
        ));
        out.push_str(&format!(
            "# HELP superfluid_otlp_spans_dropped_total Operational spans dropped instead of delaying a tick.\n# TYPE superfluid_otlp_spans_dropped_total counter\nsuperfluid_otlp_spans_dropped_total{{reason=\"queue_full\"}} {}\nsuperfluid_otlp_spans_dropped_total{{reason=\"export_failed\"}} {}\nsuperfluid_otlp_spans_dropped_total{{reason=\"rejected\"}} {}\n",
            g(&otlp::DROPPED_QUEUE_FULL),
            g(&otlp::DROPPED_EXPORT_FAILED),
            g(&otlp::DROPPED_REJECTED)
        ));
        out.push_str(&self.completion_metrics_text());
        out
    }

    fn completion_metrics_text(&self) -> String {
        use std::sync::atomic::Ordering::Relaxed;
        let n = &self.completion.counters;
        let mut out = String::from(
            "# HELP superfluid_fim_completions_total FIM completions by outcome.\n\
             # TYPE superfluid_fim_completions_total counter\n",
        );
        for (label, v) in [
            ("generated", &n.generated),
            ("cached", &n.cached),
            ("expired", &n.expired),
            ("throttled", &n.throttled),
        ] {
            out.push_str(&format!(
                "superfluid_fim_completions_total{{result=\"{label}\"}} {}\n",
                v.load(Relaxed)
            ));
        }
        if let Some(sat) = self.fim_satellite.get() {
            let s = &sat.daemon.stats;
            let model = sat.name.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
            out.push_str(&format!(
                "# HELP superfluid_fim_satellite_info The --fim-model satellite serving completions.\n\
                 # TYPE superfluid_fim_satellite_info gauge\n\
                 superfluid_fim_satellite_info{{model=\"{model}\"}} 1\n\
                 # HELP superfluid_fim_satellite_decode_tokens_total Tokens the satellite decoded.\n\
                 # TYPE superfluid_fim_satellite_decode_tokens_total counter\n\
                 superfluid_fim_satellite_decode_tokens_total {}\n\
                 # HELP superfluid_fim_satellite_lanes_active Satellite lanes resident after its last tick.\n\
                 # TYPE superfluid_fim_satellite_lanes_active gauge\n\
                 superfluid_fim_satellite_lanes_active {}\n\
                 # HELP superfluid_fim_satellite_pool_blocks_used Satellite KV paged blocks in use.\n\
                 # TYPE superfluid_fim_satellite_pool_blocks_used gauge\n\
                 superfluid_fim_satellite_pool_blocks_used {}\n\
                 # HELP superfluid_fim_satellite_pool_blocks_total Satellite KV paged blocks total.\n\
                 # TYPE superfluid_fim_satellite_pool_blocks_total gauge\n\
                 superfluid_fim_satellite_pool_blocks_total {}\n",
                s.decode_tokens.load(Relaxed),
                s.lanes_active.load(Relaxed),
                s.pool_blocks_used.load(Relaxed),
                s.pool_blocks_total.load(Relaxed),
            ));
        }
        out
    }

    pub fn set_background_tick_divisor(&self, divisor: u32) {
        self.background_tick_divisor
            .store(divisor.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    pub fn set_qos(
        &self,
        session: u64,
        class: u8,
        batch_invariant: bool,
    ) -> Result<CommittedEvent, DaemonError> {
        self.store
            .lock()
            .expect("store")
            .set_qos(session, class, batch_invariant)
    }

    pub fn sched_stats(&self) -> std::sync::Arc<scheduler::SchedStats> {
        std::sync::Arc::clone(&self.stats)
    }

    pub fn cancel_registry(&self) -> std::sync::Arc<CancelSet> {
        std::sync::Arc::clone(&self.cancels)
    }

    pub fn active_registry(
        &self,
    ) -> std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u64>>> {
        std::sync::Arc::clone(&self.active)
    }

    pub fn store(&self) -> std::sync::Arc<std::sync::Mutex<SessionStore>> {
        std::sync::Arc::clone(&self.store)
    }

    pub fn bus(&self) -> std::sync::Arc<bus::SessionBus> {
        self.store.lock().expect("store").bus()
    }

    pub fn shutdown(&self) {
        self.serving().sched.shutdown();
    }

    pub fn signal_shutdown(&self) {
        self.serving().sched.signal_shutdown();
    }

    pub fn create(&self, parent: Option<u64>, params: GenParams) -> Result<u64, DaemonError> {
        self.store.lock().expect("store").create(parent, params)
    }

    pub fn fork(
        &self,
        parent: u64,
        at_event: u64,
        params: Option<GenParams>,
    ) -> Result<u64, DaemonError> {
        self.refuse_if_purging(parent)?;
        self.store.lock().expect("store").fork(parent, at_event, params)
    }

    pub fn tree(&self) -> Vec<(u64, Option<u64>, u64)> {
        self.sessions(false)
            .into_iter()
            .map(|s| (s.id, s.parent, s.fork_at))
            .collect()
    }

    pub fn sessions(&self, include_archived: bool) -> Vec<SessionSummary> {
        let store = self.store.lock().expect("store");
        store
            .session_ids()
            .into_iter()
            .filter_map(|id| store.session(id).ok())
            .filter(|s| include_archived || !s.archived)
            .map(|s| SessionSummary {
                id: s.id,
                parent: if s.rerooted { None } else { s.parent },
                fork_at: s.base,
                title: s.title.clone(),
                archived: s.archived,
                meta_version: s.meta_version,
                generation: s.generation,
                events: s.events.len() as u64,
                tokens: s.tokens.len() as u64,
                open_tool_calls: s.open_tool_calls.len() as u64,
            })
            .collect()
    }

    pub fn set_meta(
        &self,
        session: u64,
        expected_version: u64,
        title: Option<String>,
        archived: Option<bool>,
    ) -> Result<u64, DaemonError> {
        self.store
            .lock()
            .expect("store")
            .set_meta(session, expected_version, title, archived)
    }

    pub fn rebase(
        &self,
        parent: u64,
        edits: Vec<wal::RebaseEdit>,
        params: Option<GenParams>,
    ) -> Result<u64, DaemonError> {
        use wal::RebaseEdit;
        if edits.is_empty() {
            return Err(DaemonError::RebaseSpec("no edits"));
        }
        self.refuse_if_purging(parent)?;
        let range = |e: &RebaseEdit| match e {
            RebaseEdit::Drop { from, to } | RebaseEdit::Replace { from, to, .. } => (*from, *to),
        };
        let mut store = self.store.lock().expect("store");
        let (max, parent_events) = {
            let p = store.session(parent)?;
            (p.next_event_id(), p.events.clone())
        };
        let mut last_to = 0;
        for e in &edits {
            let (from, to) = range(e);
            if from == 0 {
                return Err(DaemonError::RebaseSpec("edits never include Created (id 0)"));
            }
            if from >= to {
                return Err(DaemonError::RebaseSpec("empty range"));
            }
            if to > max {
                return Err(DaemonError::RebaseSpec("range beyond the parent's committed events"));
            }
            if from < last_to {
                return Err(DaemonError::RebaseSpec("ranges must be sorted and non-overlapping"));
            }
            if let RebaseEdit::Replace { role, .. } = e {
                if !matches!(*role, wal::role::SYSTEM | wal::role::USER | wal::role::ASSISTANT) {
                    return Err(DaemonError::UnknownRole(*role));
                }
            }
            last_to = to;
        }
        let first = range(&edits[0]).0;
        let child = store.rebase_branch(parent, first, params, edits.clone())?;
        let mut tool_ids: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        let mut i = first as usize;
        while i < parent_events.len() {
            let id = i as u64;
            if let Some(e) = edits.iter().find(|e| {
                let (from, to) = range(e);
                id >= from && id < to
            }) {
                let (from, to) = range(e);
                if let RebaseEdit::Replace { role, text, .. } = e {
                    if id == from {
                        let span = self.render_for(&store, child, *role, text)?;
                        store.append_message(child, *role, text.clone(), span)?;
                    }
                }
                i = to as usize;
                continue;
            }
            match &parent_events[i].body {
                EventBody::Appended { text, span } => {
                    store.append(child, text.clone(), span.clone())?;
                }
                EventBody::Message { role, text, span } => {
                    store.append_message(child, *role, text.clone(), span.clone())?;
                }
                EventBody::GenerationPrompt { span } => {
                    store.commit_generation_prompt(child, span.clone())?;
                }
                EventBody::Generated {
                    span,
                    text,
                    channel,
                    finish,
                } => {
                    store.commit_generated(child, span.clone(), text.clone(), *channel, *finish)?;
                }
                EventBody::ToolUse { name, arguments } => {
                    let e = store.commit_tool_use(child, name.clone(), arguments.clone())?;
                    tool_ids.insert(id, e.event_id);
                }
                EventBody::ToolResult {
                    call_id,
                    content,
                    span,
                } => match tool_ids.get(call_id) {
                    Some(new_id) => {
                        store.commit_tool_result(child, *new_id, content.clone(), span.clone())?;
                    }
                    None => {
                        store.append_message(child, wal::role::TOOL, content.clone(), span.clone())?;
                    }
                },
                EventBody::ToolParseFailure { raw } => {
                    store.commit_tool_parse_failure(child, raw.clone())?;
                }
                EventBody::Created { .. }
                | EventBody::EpochBump
                | EventBody::GenerationFingerprint { .. }
                | EventBody::Forked { .. }
                | EventBody::Rebased { .. }
                | EventBody::MetaUpdated { .. }
                | EventBody::ToolExpired { .. }
                | EventBody::Purged { .. }
                | EventBody::Rerooted { .. }
                | EventBody::QosSet { .. } => {}
                EventBody::Block {
                    role,
                    kind,
                    payload,
                    span,
                    visibility_version,
                } => {
                    store.commit_block(child, *role, *kind, payload.clone(), span.clone(), *visibility_version)?;
                }
                EventBody::PermissionRequest { .. }
                | EventBody::PermissionResponse { .. }
                | EventBody::ToolCancelRequested { .. }
                | EventBody::ToolOutcome { .. }
                | EventBody::ToolReconciliation { .. }
                | EventBody::ToolLease { .. }
                | EventBody::Pinned { .. } => {}
            }
            i += 1;
        }
        Ok(child)
    }

    pub fn trim(&self, session: u64, to_event: u64) -> Result<u64, DaemonError> {
        let max = {
            let store = self.store.lock().expect("store");
            store.session(session)?.next_event_id()
        };
        if to_event == 0 || to_event > max {
            return Err(DaemonError::ForkPoint {
                session,
                fork_at: to_event,
                max,
            });
        }
        if to_event == max {
            return self.fork(session, to_event, None);
        }
        self.rebase(
            session,
            vec![wal::RebaseEdit::Drop {
                from: to_event,
                to: max,
            }],
            None,
        )
    }

    pub fn pin(&self, session: u64, ttl_ms: u64) -> Result<u64, DaemonError> {
        let deadline = if ttl_ms == 0 {
            0
        } else {
            wal::now_unix_ms().saturating_add(ttl_ms)
        };
        let (tokens, want) = {
            let mut store = self.store.lock().expect("store");
            store.set_pin(session, deadline)?;
            pins::session_target(&store.session(session)?.tokens)
        };
        self.send_pin(
            pins::PinKey::Session(session),
            tokens,
            want,
            deadline,
            Some(session),
        )?;
        Ok(deadline)
    }

    pub fn pin_prefix(
        &self,
        anchor: Option<u64>,
        stream: &[u32],
        upto: usize,
        ttl_ms: u64,
    ) -> Result<u64, DaemonError> {
        let upto = upto.min(stream.len());
        let key = pins::PinKey::Prefix(pins::prefix_digest(&stream[..upto]));
        let tokens = if upto < stream.len() {
            stream[..=upto].to_vec()
        } else {
            pins::session_target(stream).0
        };
        let deadline = wal::now_unix_ms().saturating_add(ttl_ms.max(1));
        self.send_pin(key, tokens, upto as u64, deadline, anchor)
    }

    pub fn pinned_prefix_tokens(
        &self,
        stream: &[u32],
        uptos: &[usize],
    ) -> Result<u64, DaemonError> {
        let keys = uptos
            .iter()
            .map(|&u| {
                let upto = u.min(stream.len());
                pins::PinKey::Prefix(pins::prefix_digest(&stream[..upto]))
            })
            .collect();
        let (tx, rx) = std::sync::mpsc::channel();
        self.sched_tx()
            .send(scheduler::SchedCmd::PinHeld(keys, tx))
            .map_err(|_| DaemonError::Protocol("scheduler is gone"))?;
        rx.recv()
            .map_err(|_| DaemonError::Protocol("scheduler dropped the query"))
    }

    pub fn unanchor_pins(&self, session: u64) {
        let _ = self.sched_tx().send(scheduler::SchedCmd::PinUnanchor(session));
    }

    fn send_pin(
        &self,
        key: pins::PinKey,
        tokens: Vec<u32>,
        want: u64,
        deadline_unix_ms: u64,
        anchor: Option<u64>,
    ) -> Result<u64, DaemonError> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.sched_tx()
            .send(scheduler::SchedCmd::Pin(pins::PinReq {
                key,
                tokens,
                want,
                deadline_unix_ms,
                anchor,
                reply: Some(tx),
            }))
            .map_err(|_| DaemonError::Protocol("scheduler is gone"))?;
        rx.recv()
            .map_err(|_| DaemonError::Protocol("scheduler dropped the pin"))
    }

    fn render_for(
        &self,
        store: &SessionStore,
        session: u64,
        role: u32,
        text: &str,
    ) -> Result<Vec<u32>, DaemonError> {
        let s = store.session(session)?;
        let mut span = if s.tokens.is_empty() {
            self.codec().stream_prologue()
        } else {
            match Self::last_stream_record(&s.events) {
                Some(EventBody::Generated { finish, .. })
                    if *finish == superfluid_abi::finish::EOS =>
                {
                    self.codec().turn_separator().unwrap_or_default()
                }
                _ => Vec::new(),
            }
        };
        match self.codec().render_message(role, text) {
            Some(rendered) => span.extend(rendered),
            None => {
                let encoded = self.codec().encode(text);
                if encoded.is_empty() {
                    return Err(DaemonError::Protocol("codec produced no span for the replacement"));
                }
                span.extend(encoded);
            }
        }
        Ok(span)
    }

    pub fn purge(
        &self,
        session: u64,
        expected_generation: u64,
        mode: wal::PurgeMode,
    ) -> Result<Vec<u64>, DaemonError> {
        let targets: Vec<u64> = {
            let store = self.store.lock().expect("store");
            let s = store.session(session)?;
            if s.generation != expected_generation {
                return Err(DaemonError::GenerationConflict {
                    session,
                    expected: expected_generation,
                    actual: s.generation,
                });
            }
            let mut t = vec![session];
            if mode == wal::PurgeMode::Cascade {
                t.extend(store.descendants_of(session));
            }
            t
        };
        {
            let _a = self.active.lock().expect("active set");
            let mut p = self.purging.lock().expect("purging");
            p.extend(targets.iter().copied());
        }
        let _fence = PurgeFence {
            set: std::sync::Arc::clone(&self.purging),
            ids: targets.clone(),
        };
        for &t in &targets {
            self.quiesce(t)?;
        }
        let purged = {
            let mut store = self.store.lock().expect("store");
            for &t in &targets {
                let open: Vec<u64> = store.session(t)?.open_tool_calls.keys().copied().collect();
                for call in open {
                    store.expire_tool_call(t, call, "purged")?;
                }
            }
            store.purge(session, expected_generation, mode)?
        };
        let bus = self.bus();
        for &p in &purged {
            bus.end_session(p, bus::EndReason::Purged);
        }
        for &p in &purged {
            let _ = self.send_pin(pins::PinKey::Session(p), Vec::new(), 0, 0, None);
        }
        Ok(purged)
    }

    fn refuse_if_purging(&self, session: u64) -> Result<(), DaemonError> {
        let _a = self.active.lock().expect("active set");
        if self.purging.lock().expect("purging").contains(&session) {
            return Err(DaemonError::Purged(session));
        }
        Ok(())
    }

    fn quiesce(&self, session: u64) -> Result<(), DaemonError> {
        {
            let mut c = self.cancels.lock().expect("cancel registry");
            if self.active.lock().expect("active set").contains(&session) {
                c.insert(session);
            } else {
                return Ok(());
            }
        }
        self.cancels.wake();
        for _ in 0..2000 {
            if !self.active.lock().expect("active set").contains(&session) {
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        Err(DaemonError::PurgeTimeout(session))
    }

    pub fn append(
        &self,
        session: u64,
        text: Option<String>,
        span: Vec<u32>,
    ) -> Result<CommittedEvent, DaemonError> {
        {
            let active = self.active.lock().expect("active set");
            if active.contains(&session) {
                return Err(DaemonError::SessionBusy(session));
            }
            if self.purging.lock().expect("purging").contains(&session) {
                return Err(DaemonError::Purged(session));
            }
        }
        let span = if span.is_empty() {
            match &text {
                Some(t) => {
                    let encoded = self.codec().encode(t);
                    if encoded.is_empty() {
                        return Err(DaemonError::Protocol(
                            "append had no span and the codec produced none",
                        ));
                    }
                    encoded
                }
                None => return Err(DaemonError::Protocol("append carried neither text nor span")),
            }
        } else {
            span
        };
        self.store.lock().expect("store").append(session, text, span)
    }

    fn separator_for(&self, session: u64) -> Vec<u32> {
        let store = self.store.lock().expect("store");
        let Ok(s) = store.session(session) else {
            return Vec::new();
        };
        if s.tokens.is_empty() {
            return self.codec().stream_prologue();
        }
        match Self::last_stream_record(&s.events) {
            Some(EventBody::Generated { finish, .. })
                if *finish == superfluid_abi::finish::EOS =>
            {
                self.codec().turn_separator().unwrap_or_default()
            }
            _ => Vec::new(),
        }
    }

    pub fn append_message(
        &self,
        session: u64,
        role: u32,
        text: String,
    ) -> Result<CommittedEvent, DaemonError> {
        if self.active.lock().expect("active set").contains(&session) {
            return Err(DaemonError::SessionBusy(session));
        }
        if !matches!(role, wal::role::SYSTEM | wal::role::USER | wal::role::ASSISTANT) {
            return Err(DaemonError::UnknownRole(role));
        }
        let mut span = self.separator_for(session);
        span.extend(
            self.codec()
                .render_message(role, &text)
                .ok_or(DaemonError::Protocol("codec has no chat dialect"))?,
        );
        self.store
            .lock()
            .expect("store")
            .append_message(session, role, text, span)
    }

    pub fn renders_per_message(&self) -> bool {
        self.codec().renders_per_message()
    }

    pub fn append_system_full(
        &self,
        session: u64,
        system: Option<String>,
        tool_jsons: Vec<String>,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<CommittedEvent, DaemonError> {
        if self.active.lock().expect("active set").contains(&session) {
            return Err(DaemonError::SessionBusy(session));
        }
        let span = self
            .codec()
            .render_system_full(system.as_deref(), &tool_jsons, kwargs)
            .ok_or(DaemonError::Protocol("codec has no chat dialect"))?;
        self.store.lock().expect("store").append_message(
            session,
            wal::role::SYSTEM,
            system.unwrap_or_default(),
            span,
        )
    }

    // Chat templates declare tools only in the opening system turn, so one rendered later would
    // put the model's input in a shape it was not trained on; it is refused instead.
    pub fn append_system(
        &self,
        session: u64,
        system: Option<String>,
        tools: Vec<String>,
    ) -> Result<CommittedEvent, DaemonError> {
        if system.is_none() && tools.is_empty() {
            return Err(DaemonError::Protocol("AppendSystem carried neither text nor tools"));
        }
        let tool_jsons = tools
            .iter()
            .enumerate()
            .map(|(index, t)| {
                codec::function_tool(t).map_err(|why| DaemonError::InvalidTool { index, why })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if self.active.lock().expect("active set").contains(&session) {
            return Err(DaemonError::SessionBusy(session));
        }
        let codec = self.codec();
        let mut span = codec.stream_prologue();
        span.extend(
            codec
                .render_system_full(system.as_deref(), &tool_jsons, &serde_json::Map::new())
                .ok_or(DaemonError::Protocol("codec has no chat dialect"))?,
        );
        let mut store = self.store.lock().expect("store");
        if !store.session(session)?.tokens.is_empty() {
            return Err(DaemonError::SystemNotFirst(session));
        }
        store.append_message(session, wal::role::SYSTEM, system.unwrap_or_default(), span)
    }

    pub(crate) fn render_conversation(
        &self,
        messages: &[codec::ChatMessage],
        tools: &[String],
    ) -> Option<Vec<u32>> {
        self.codec()
            .render_prompt_structured_with(messages, tools, &serde_json::Map::new())
    }

    pub fn supports_reasoning_effort(&self) -> bool {
        self.codec().supports_reasoning_effort()
    }

    pub fn reasoning_effort_levels(&self) -> Vec<&'static str> {
        self.codec().reasoning_effort_levels()
    }

    pub fn supports_enable_thinking(&self) -> bool {
        self.codec().supports_enable_thinking()
    }

    pub fn validate_template_kwargs(
        &self,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        self.codec().validate_template_kwargs(kwargs)
    }

    fn commit_opener(&self, session: u64, opener: Vec<u32>) -> Result<(), DaemonError> {
        if opener.is_empty() {
            return Ok(());
        }
        self.store
            .lock()
            .expect("store")
            .commit_generation_prompt(session, opener)
            .map(|_| ())
    }

    pub fn append_conversation(
        &self,
        session: u64,
        messages: &[crate::codec::ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<CommittedEvent, DaemonError> {
        let codec = self.codec();
        let mut span = codec
            .render_prompt_structured_with(messages, tools, kwargs)
            .ok_or_else(|| render_refusal(&*codec, messages, tools, kwargs, "codec has no chat dialect"))?;
        let opener_len = codec.trailing_generation_prompt(messages, tools, kwargs, &span);
        let opener: Vec<u32> = if opener_len > 0 && opener_len < span.len() {
            span.split_off(span.len() - opener_len)
        } else {
            Vec::new()
        };
        let text = messages.last().map(|m| m.content.clone());
        let blobs: Vec<(u32, String)> = messages
            .iter()
            .flat_map(|m| {
                m.parts.iter().filter_map(move |p| match p {
                    crate::codec::ContentPart::Image { blob } => Some((m.role, blob.clone())),
                    crate::codec::ContentPart::Audio { blob } => Some((m.role, blob.clone())),
                    crate::codec::ContentPart::Text(_) => None,
                })
            })
            .collect();
        if blobs.is_empty() {
            let ev = self.append(session, text, span)?;
            self.commit_opener(session, opener)?;
            return Ok(ev);
        }
        let mut probes = Vec::with_capacity(blobs.len());
        for (_, blob) in &blobs {
            if !self.media.exists(blob) {
                return Err(DaemonError::UnknownMedia(blob.clone()));
            }
            let path = self.media.path_of(blob);
            let info = match self.engine_sync(proto_linkw::StateSyncReqMsg::MediaProbe {
                image_path: path.to_string_lossy().into_owned(),
            })? {
                proto_linkw::StateSyncOkMsg::MediaProbe { info } => info,
                _ => return Err(DaemonError::Protocol("mismatched MediaProbe reply")),
            };
            if info.n_tokens == 0 || info.n_tokens as u64 > scheduler::PREFILL_BUDGET as u64 {
                return Err(DaemonError::MediaTooLarge(info.n_tokens));
            }
            probes.push(info);
        }
        let mut out = Vec::with_capacity(span.len() + 256);
        let mut offsets: Vec<usize> = Vec::with_capacity(probes.len());
        let mut img = 0usize;
        for (idx, &t) in span.iter().enumerate() {
            if img == probes.len() || t != probes[img].image_token_id {
                if probes.iter().any(|p| p.image_token_id == t) {
                    return Err(DaemonError::Protocol(
                        "rendered prompt contains a media marker with no matching pending part",
                    ));
                }
                out.push(t);
                continue;
            }
            let info = &probes[img];
            if info.boi_token_id != 0 && out.last() != Some(&info.boi_token_id) {
                out.push(info.boi_token_id);
            }
            offsets.push(out.len());
            out.extend(std::iter::repeat_n(info.image_token_id, info.n_tokens as usize));
            if info.eoi_token_id != 0 && span.get(idx + 1) != Some(&info.eoi_token_id) {
                out.push(info.eoi_token_id);
            }
            img += 1;
        }
        if img != probes.len() {
            return Err(DaemonError::Protocol(
                "this model's chat template did not emit an image marker for every image part;                  it cannot place media",
            ));
        }
        let stream_len = {
            let store = self.store.lock().expect("store");
            store.session(session)?.tokens.len() as u64
        };
        let ev = self.append(session, text, out)?;
        for (((role, blob), info), off) in blobs.iter().zip(&probes).zip(&offsets) {
            let payload = serde_json::json!({
                "blob": blob,
                "n_tokens": info.n_tokens,
                "image_token_id": info.image_token_id,
                "offset": stream_len + *off as u64,
                "pre_text": "",
                "post_text": "",
                "media_identity": {
                    "preprocessing_fp": format!("{:016x}", info.preprocess_fp),
                    "encoder": "bundle-weights-identity",
                },
            })
            .to_string();
            let mut store = self.store.lock().expect("store");
            store.commit_block(
                session,
                *role,
                wal::block_kind::IMAGE,
                payload,
                Vec::new(),
                codec::MODEL_VISIBILITY_VERSION,
            )?;
        }
        self.commit_opener(session, opener)?;
        Ok(ev)
    }

    pub fn response_format_tag(&self, json_schema: &str) -> Option<String> {
        self.codec().response_format_tag(json_schema)
    }

    pub fn tool_structural_tag(&self, tool_jsons: &[String], at_least_one: bool) -> Option<String> {
        self.codec().structural_tag(tool_jsons, at_least_one)
    }

    pub fn grammar_create_structural(&self, tag_json: &str) -> Result<u32, DaemonError> {
        match self.engine_sync(proto_linkw::StateSyncReqMsg::CreateGrammarStructural {
            tag_json: tag_json.to_string(),
        })? {
            proto_linkw::StateSyncOkMsg::CreateGrammar { grammar_handle } => Ok(grammar_handle),
            _ => Err(DaemonError::Protocol("mismatched CreateGrammar reply")),
        }
    }

    pub fn parse_tool_call(&self, raw: &str) -> Option<(String, String)> {
        self.codec().parse_tool_call(raw)
    }

    pub fn named_tool_call(&self, raw: &str) -> Option<(String, String)> {
        self.codec().named_tool_call(raw)
    }

    pub fn parse_tool_call_with(
        &self,
        raw: &str,
        schemas: Option<&crate::codec::ToolSchemas>,
    ) -> Option<(String, String)> {
        self.codec().parse_tool_call_with(raw, schemas)
    }

    pub fn tool_fragment_mode(&self) -> crate::tool_fragment::ToolFragmentMode {
        self.codec().tool_fragment_mode()
    }

    pub fn tool_call_grammar(&self, tool_jsons: &[String]) -> Option<String> {
        self.codec().call_grammar(tool_jsons)
    }

    pub fn append_system_with_tools(
        &self,
        session: u64,
        system: Option<String>,
        tool_jsons: Vec<String>,
    ) -> Result<CommittedEvent, DaemonError> {
        if self.active.lock().expect("active set").contains(&session) {
            return Err(DaemonError::SessionBusy(session));
        }
        let span = self
            .codec()
            .render_system_with_tools(system.as_deref(), &tool_jsons)
            .ok_or(DaemonError::Protocol("codec has no tool dialect"))?;
        self.store.lock().expect("store").append_message(
            session,
            wal::role::SYSTEM,
            system.unwrap_or_default(),
            span,
        )
    }

    /// `after_query`: no user query follows this turn in the conversation
    /// (a step of a tool loop), which some templates render differently.
    pub fn append_assistant_with_tool_calls(
        &self,
        session: u64,
        content: String,
        calls: Vec<(String, String)>,
        after_query: bool,
    ) -> Result<CommittedEvent, DaemonError> {
        if self.active.lock().expect("active set").contains(&session) {
            return Err(DaemonError::SessionBusy(session));
        }
        let mut span = self.separator_for(session);
        let codec = self.codec();
        let turn = if after_query {
            codec.render_assistant_with_tool_calls_after_query(&content, &calls)
        } else {
            codec.render_assistant_with_tool_calls(&content, &calls)
        };
        span.extend(turn.ok_or(DaemonError::Protocol("codec has no tool dialect"))?);
        self.store.lock().expect("store").append_message(
            session,
            wal::role::ASSISTANT,
            content,
            span,
        )
    }

    pub fn append_tool_result(
        &self,
        session: u64,
        call_id: u64,
        content: String,
    ) -> Result<CommittedEvent, DaemonError> {
        if self.active.lock().expect("active set").contains(&session) {
            return Err(DaemonError::SessionBusy(session));
        }
        let name = {
            let mut store = self.store.lock().expect("store");
            let s = store.session(session)?;
            if call_id < s.base {
                return Err(DaemonError::InheritedToolCall(call_id));
            }
            match s.ledger.get(&call_id) {
                None => return Err(DaemonError::UnknownToolCall(call_id)),
                Some(e)
                    if e.state == wal::ledger_state::OPEN
                        || e.state == wal::ledger_state::CANCEL_REQUESTED =>
                {
                    if let Some(deadline) = e.lease_deadline_unix_ms {
                        if wal::now_unix_ms() > deadline {
                            store.expire_tool_call(session, call_id, "lease expired")?;
                            return Err(DaemonError::ToolLeaseExpired(call_id));
                        }
                    }
                    e.name.clone()
                }
                Some(e) if e.state == wal::ledger_state::SUCCEEDED => {
                    return Err(DaemonError::UnknownToolCall(call_id));
                }
                Some(_) => {
                    store.commit_tool_reconciliation(
                        session,
                        call_id,
                        format!("late result: {} bytes", content.len()),
                    )?;
                    return Err(DaemonError::ToolLateEffect(call_id));
                }
            }
        };
        let mut span = self.separator_for(session);
        span.extend(
            self.codec()
                .render_tool_result(&name, &content)
                .ok_or(DaemonError::Protocol("codec has no tool dialect"))?,
        );
        self.store
            .lock()
            .expect("store")
            .commit_tool_result(session, call_id, content, span)
    }

    pub fn append_block(
        &self,
        session: u64,
        role: u32,
        kind: u32,
        payload: String,
    ) -> Result<CommittedEvent, DaemonError> {
        if self.active.lock().expect("active set").contains(&session) {
            return Err(DaemonError::SessionBusy(session));
        }
        if !matches!(role, wal::role::SYSTEM | wal::role::USER | wal::role::ASSISTANT | wal::role::TOOL) {
            return Err(DaemonError::UnknownRole(role));
        }
        if codec::project_block_text(kind, &payload).is_none() {
            return Err(DaemonError::UnknownBlockKind(kind));
        }
        let projected = self
            .codec()
            .render_block(role, kind, &payload)
            .ok_or(DaemonError::Protocol("codec has no chat dialect"))?;
        let span = if projected.is_empty() {
            Vec::new()
        } else {
            let mut span = self.separator_for(session);
            span.extend(projected);
            span
        };
        self.store.lock().expect("store").commit_block(
            session,
            role,
            kind,
            payload,
            span,
            codec::MODEL_VISIBILITY_VERSION,
        )
    }

    pub fn request_permission(
        &self,
        session: u64,
        call_id: u64,
        text: String,
    ) -> Result<CommittedEvent, DaemonError> {
        let active = self.active.lock().expect("active set");
        if active.contains(&session) {
            return Err(DaemonError::SessionBusy(session));
        }
        if self.purging.lock().expect("purging").contains(&session) {
            return Err(DaemonError::Purged(session));
        }
        self.store
            .lock()
            .expect("store")
            .commit_permission_request(session, call_id, text)
    }

    pub fn respond_permission(
        &self,
        session: u64,
        request_id: u64,
        granted: bool,
    ) -> Result<CommittedEvent, DaemonError> {
        self.store
            .lock()
            .expect("store")
            .commit_permission_response(session, request_id, granted)
    }

    pub fn request_tool_cancel(&self, session: u64, call_id: u64) -> Result<CommittedEvent, DaemonError> {
        self.store
            .lock()
            .expect("store")
            .commit_tool_cancel_request(session, call_id)
    }

    pub fn append_tool_outcome(
        &self,
        session: u64,
        call_id: u64,
        outcome: u8,
        note: String,
    ) -> Result<CommittedEvent, DaemonError> {
        self.store
            .lock()
            .expect("store")
            .commit_tool_outcome(session, call_id, outcome, note)
    }

    pub fn ledger(&self, session: u64) -> Result<Vec<(u64, store::LedgerEntry)>, DaemonError> {
        Ok(self
            .store
            .lock()
            .expect("store")
            .session(session)?
            .ledger
            .iter()
            .map(|(id, e)| (*id, e.clone()))
            .collect())
    }

    pub fn append_tool_history(
        &self,
        session: u64,
        content: String,
    ) -> Result<CommittedEvent, DaemonError> {
        if self.active.lock().expect("active set").contains(&session) {
            return Err(DaemonError::SessionBusy(session));
        }
        let mut span = self.separator_for(session);
        span.extend(
            self.codec()
                .render_tool_result("", &content)
                .ok_or(DaemonError::Protocol("codec has no tool dialect"))?,
        );
        self.store.lock().expect("store").append_message(
            session,
            wal::role::TOOL,
            content,
            span,
        )
    }

    pub fn open_tool_calls(
        &self,
        session: u64,
    ) -> Result<Vec<(u64, String, String)>, DaemonError> {
        Ok(self
            .store
            .lock()
            .expect("store")
            .session(session)?
            .open_tool_calls
            .iter()
            .map(|(id, (n, a))| (*id, n.clone(), a.clone()))
            .collect())
    }

    pub fn generate(&self, session: u64, max_tokens: u32) -> Result<GenerateOutcome, DaemonError> {
        self.generate_streaming(session, max_tokens, |_| Ok(()))
    }

    pub fn generate_streaming(
        &self,
        session: u64,
        max_tokens: u32,
        on_event: impl FnMut(&CommittedEvent) -> Result<(), DaemonError>,
    ) -> Result<GenerateOutcome, DaemonError> {
        self.generate_streaming_ex(session, max_tokens, scheduler::GenExtras::default(), on_event)
    }

    pub fn generate_streaming_ex(
        &self,
        session: u64,
        max_tokens: u32,
        extras: scheduler::GenExtras,
        on_event: impl FnMut(&CommittedEvent) -> Result<(), DaemonError>,
    ) -> Result<GenerateOutcome, DaemonError> {
        self.generate_streaming_ex_lp(session, max_tokens, extras, on_event, |_| {})
    }

    pub fn generate_streaming_ex_lp(
        &self,
        session: u64,
        max_tokens: u32,
        extras: scheduler::GenExtras,
        on_event: impl FnMut(&CommittedEvent) -> Result<(), DaemonError>,
        on_logprobs: impl FnMut(&[scheduler::TokenLogprob]),
    ) -> Result<GenerateOutcome, DaemonError> {
        self.generate_streaming_inner(session, max_tokens, extras, on_event, on_logprobs, |_, _, _| {})
    }

    pub fn generate_streaming_delta(
        &self,
        session: u64,
        max_tokens: u32,
        mut extras: scheduler::GenExtras,
        on_event: impl FnMut(&CommittedEvent) -> Result<(), DaemonError>,
        on_logprobs: impl FnMut(&[scheduler::TokenLogprob]),
        on_delta: impl FnMut(u32, &str, u32),
    ) -> Result<GenerateOutcome, DaemonError> {
        extras.wants_deltas = true;
        self.generate_streaming_inner(session, max_tokens, extras, on_event, on_logprobs, on_delta)
    }

    fn generation_opener(&self, events: &[store::CommittedEvent], thinking: Option<bool>) -> Option<Vec<u32>> {
        match Self::last_stream_record(events) {
            Some(EventBody::Message { .. }) | Some(EventBody::ToolResult { .. }) => self
                .codec()
                .generation_prefix_with(codec::TurnState::AfterInput, thinking),
            Some(EventBody::Generated { finish, .. }) if *finish == superfluid_abi::finish::EOS => self
                .codec()
                .generation_prefix_with(codec::TurnState::AfterFinishedTurn, thinking),
            _ => None,
        }
    }

    /// Where a continuation of `session` forks (one past its last event),
    /// and whether its last turn ended: an append after an unended turn
    /// lands inside it.
    pub(crate) fn session_end(&self, session: u64) -> Option<(u64, bool)> {
        let store = self.store.lock().expect("store");
        let s = store.session(session).ok()?;
        let ended = matches!(
            Self::last_stream_record(&s.events),
            Some(EventBody::Generated { finish, .. }) if *finish == superfluid_abi::finish::EOS
        );
        Some((s.next_event_id(), ended))
    }

    fn last_stream_record(events: &[store::CommittedEvent]) -> Option<&EventBody> {
        events
            .iter()
            .rev()
            .map(|e| &e.body)
            .find(|b| b.span().is_some_and(|s| !s.is_empty()))
    }

    fn open_turn_tokens(events: &[store::CommittedEvent]) -> Vec<u32> {
        let mut spans: Vec<&[u32]> = Vec::new();
        for e in events.iter().rev() {
            match &e.body {
                EventBody::Generated { span, finish, .. } => {
                    if spans.is_empty() && *finish == superfluid_abi::finish::EOS {
                        return Vec::new();
                    }
                    spans.push(span);
                }
                EventBody::GenerationPrompt { span } => {
                    spans.push(span);
                    break;
                }
                b => match b.span() {
                    None | Some([]) => continue,
                    Some(_) => break,
                },
            }
        }
        spans.into_iter().rev().flatten().copied().collect()
    }

    fn generate_streaming_inner(
        &self,
        session: u64,
        max_tokens: u32,
        extras: scheduler::GenExtras,
        on_event: impl FnMut(&CommittedEvent) -> Result<(), DaemonError>,
        on_logprobs: impl FnMut(&[scheduler::TokenLogprob]),
        on_delta: impl FnMut(u32, &str, u32),
    ) -> Result<GenerateOutcome, DaemonError> {
        self.ensure_serving()?;
        let _span = tracing::info_span!(
            parent: None,
            "generate",
            session,
            max_tokens,
            model = self.trace_scope.model.as_str(),
            store_ns = self.trace_scope.ns
        )
        .entered();
        let (stream, params, class, batch_invariant, media) = {
            let mut cancels = self.cancels.lock().expect("cancel registry");
            let mut active = self.active.lock().expect("active set");
            if self.purging.lock().expect("purging").contains(&session) {
                return Err(DaemonError::Purged(session));
            }
            if self.tool_lease_ms > 0 {
                let now = crate::wal::now_unix_ms();
                let mut store = self.store.lock().expect("store");
                let elapsed: Vec<u64> = {
                    let s = store.session(session)?;
                    s.open_tool_calls
                        .keys()
                        .filter(|id| {
                            s.ledger
                                .get(id)
                                .and_then(|e| e.lease_deadline_unix_ms)
                                .is_some_and(|d| d <= now)
                        })
                        .copied()
                        .collect()
                };
                for call_id in elapsed {
                    let _ = store.expire_tool_call(session, call_id, "lease elapsed");
                }
            }
            let snap = {
                let store = self.store.lock().expect("store");
                let s = store.session(session)?;
                if !s.open_permissions.is_empty() {
                    return Err(DaemonError::PermissionPending(session));
                }
                if !s.open_tool_calls.is_empty() {
                    tracing::warn!(
                        session,
                        open_tool_calls = s.open_tool_calls.len(),
                        "generating with unanswered tool calls: the model cannot see its own \
                         tool output and will re-issue the call. Return a result for each open \
                         call id (OpenAI: a `role: \"tool\"` message carrying its `tool_call_id`), \
                         or run with --tool-lease-ms so unanswered calls expire instead of \
                         accumulating."
                    );
                }
                (
                    s.tokens.clone(),
                    s.params,
                    s.qos_class,
                    s.batch_invariant,
                    self.media_of(s),
                )
            };
            if snap.0.is_empty() {
                return Err(DaemonError::EmptySession(session));
            }
            self.check_stream_fits(snap.0.len())?;
            if max_tokens == 0 {
                return Ok(GenerateOutcome {
                    events: Vec::new(),
                    tokens_generated: 0,
                    finish: superfluid_abi::finish::NONE,
                    warm_prefix: 0,
                    logprobs: Vec::new(),
                    spec: (0, 0),
                });
            }
            if !active.insert(session) {
                return Err(DaemonError::SessionBusy(session));
            }
            // A cancel aimed at the previous generation can land after its
            // lane stopped reading cancels (finishing, or already retired);
            // it must not stop this one.
            cancels.remove(&session);
            snap
        };
        if let Some(mut fp) = self.codec().behavior_fingerprint() {
            fp.speculation = self.serving().speculation.identity_for_fingerprint();
            let commit = self
                .store
                .lock()
                .expect("store")
                .commit_generation_fingerprint(session, fp.digest());
            if let Err(e) = commit {
                self.active.lock().expect("active set").remove(&session);
                return Err(e);
            }
        }
        let mut stream = stream;
        let (opener, open_turn) = {
            let store = self.store.lock().expect("store");
            let events = &store.session(session)?.events;
            let opener = self.generation_opener(events, extras.thinking);
            let open_turn: Vec<u32> = if opener.is_some() {
                Vec::new()
            } else {
                Self::open_turn_tokens(events)
            };
            (opener, open_turn)
        };
        let opener_len = opener.as_ref().map(Vec::len);
        let input_end = if opener.is_some() { stream.len() as u64 } else { 0 };
        if let Some(op) = opener {
            if let Err(e) = self.check_stream_fits(stream.len() + op.len()) {
                self.active.lock().expect("active set").remove(&session);
                return Err(e);
            }
            let commit = self
                .store
                .lock()
                .expect("store")
                .commit_generation_prompt(session, op.clone());
            if let Err(e) = commit {
                self.active.lock().expect("active set").remove(&session);
                return Err(e);
            }
            stream.extend_from_slice(&op);
        }
        let mut extras = extras;
        if extras.open_channel == 0 {
            let scope: &[u32] = match opener_len {
                Some(n) => &stream[stream.len() - n..],
                None => &open_turn,
            };
            let open = codec::trailing_open_channel(scope, &self.codec().channel_markers());
            if open != 0 {
                extras.open_channel = open;
            }
        }
        let result = scheduler::run_job(
            &self.sched_tx(),
            session,
            stream,
            input_end,
            params,
            max_tokens,
            class,
            batch_invariant,
            media,
            extras,
            &self.cancels,
            on_event,
            on_logprobs,
            on_delta,
        );
        match result {
            Ok((events, tokens_generated, finish, warm_prefix, logprobs, spec)) => Ok(GenerateOutcome {
                events,
                tokens_generated,
                finish,
                warm_prefix,
                logprobs,
                spec,
            }),
            Err(e) => {
                self.active.lock().expect("active set").remove(&session);
                Err(e)
            }
        }
    }

    pub fn read(&self, session: u64, cursor: u64) -> Result<Vec<CommittedEvent>, DaemonError> {
        Ok(self.store.lock().expect("store").read(session, cursor)?.to_vec())
    }

    pub fn session_ids(&self) -> Vec<u64> {
        self.store.lock().expect("store").session_ids()
    }
}

#[cfg(test)]
mod open_turn_tests {
    use super::*;
    use superfluid_abi::finish;

    fn ev(event_id: u64, body: EventBody) -> store::CommittedEvent {
        store::CommittedEvent { event_id, epoch: 0, ts_unix_ms: 0, body }
    }

    fn generated(span: &[u32], finish: u32) -> EventBody {
        EventBody::Generated { span: span.to_vec(), text: String::new(), channel: 0, finish }
    }

    #[test]
    fn open_turn_skips_every_token_less_record() {
        let events = vec![
            ev(0, EventBody::Message { role: 1, text: "hi".into(), span: vec![900, 901] }),
            ev(1, EventBody::GenerationPrompt { span: vec![1, 2] }),
            ev(2, generated(&[3, 4], finish::NONE)),
            ev(3, EventBody::ToolUse { name: "x".into(), arguments: "{".into() }),
            ev(4, EventBody::ToolLease { call_id: 3, deadline_unix_ms: 1 }),
            ev(5, EventBody::ToolReconciliation { call_id: 3, note: String::new() }),
            ev(6, EventBody::EpochBump),
            ev(7, EventBody::Pinned { deadline_unix_ms: 1 }),
            ev(8, EventBody::Block { role: 1, kind: 7, payload: "{}".into(), span: vec![], visibility_version: 1 }),
            ev(9, generated(&[5], finish::LENGTH)),
        ];
        assert_eq!(Daemon::open_turn_tokens(&events), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn token_less_records_do_not_reopen_a_closed_turn() {
        let closed = vec![
            ev(0, EventBody::GenerationPrompt { span: vec![1] }),
            ev(1, generated(&[2], finish::EOS)),
            ev(2, EventBody::ToolLease { call_id: 9, deadline_unix_ms: 1 }),
            ev(3, EventBody::QosSet { class: 0, batch_invariant: false }),
        ];
        assert!(Daemon::open_turn_tokens(&closed).is_empty());
        let after_result = vec![
            ev(0, EventBody::GenerationPrompt { span: vec![1] }),
            ev(1, generated(&[2], finish::NONE)),
            ev(2, EventBody::ToolUse { name: "x".into(), arguments: "{}".into() }),
            ev(3, EventBody::ToolResult { call_id: 2, content: "ok".into(), span: vec![7, 8] }),
            ev(4, EventBody::ToolLease { call_id: 2, deadline_unix_ms: 1 }),
        ];
        assert!(Daemon::open_turn_tokens(&after_result).is_empty());
    }

    #[test]
    fn the_last_stream_record_is_the_last_one_with_tokens() {
        let events = vec![
            ev(0, EventBody::GenerationPrompt { span: vec![1] }),
            ev(1, generated(&[2], finish::EOS)),
            ev(2, EventBody::ToolUse { name: "x".into(), arguments: "{}".into() }),
            ev(3, EventBody::ToolLease { call_id: 2, deadline_unix_ms: 1 }),
            ev(4, EventBody::Block { role: 1, kind: 7, payload: "{}".into(), span: vec![], visibility_version: 1 }),
            ev(5, EventBody::GenerationFingerprint { digest: 9 }),
        ];
        assert!(matches!(
            Daemon::last_stream_record(&events),
            Some(EventBody::Generated { finish: finish::EOS, .. })
        ));
        assert!(Daemon::last_stream_record(&events[2..]).is_none());
    }
}

fn first_registrable<E>(
    candidates: &[String],
    auto: bool,
    mut register: impl FnMut(&str) -> Result<(), E>,
    mut refused: impl FnMut(&str, &E, Option<&str>),
) -> Result<Option<String>, E> {
    for (i, c) in candidates.iter().enumerate() {
        match register(c) {
            Ok(()) => return Ok(Some(c.clone())),
            Err(e) if auto => refused(c, &e, candidates.get(i + 1).map(String::as_str)),
            Err(e) => return Err(e),
        }
    }
    Ok(None)
}

#[cfg(test)]
mod speculate_fallback_tests {
    use super::first_registrable;

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn auto_tries_each_candidate_in_order_until_one_registers() {
        let mut tried = Vec::new();
        let mut reported = Vec::new();
        let picked = first_registrable(
            &strs(&["dspark:a", "dflash:b", "eagle3:c"]),
            true,
            |s| {
                tried.push(s.to_string());
                if s == "dflash:b" { Ok(()) } else { Err(format!("no {s}")) }
            },
            |s, e, next| reported.push((s.to_string(), e.clone(), next.map(str::to_string))),
        )
        .unwrap();
        assert_eq!(picked.as_deref(), Some("dflash:b"));
        assert_eq!(tried, ["dspark:a", "dflash:b"]);
        assert_eq!(reported, [("dspark:a".to_string(), "no dspark:a".to_string(), Some("dflash:b".to_string()))]);
    }

    #[test]
    fn auto_with_every_candidate_refused_serves_plainly() {
        let mut last_next = Some("x".to_string());
        let picked = first_registrable(
            &strs(&["dspark:a", "dflash:b"]),
            true,
            |_| Err::<(), _>("refused"),
            |_, _, next| last_next = next.map(str::to_string),
        )
        .unwrap();
        assert_eq!(picked, None);
        assert_eq!(last_next, None, "the last refusal says it is decoding plainly");
    }

    #[test]
    fn an_explicit_strategy_fails_on_the_first_refusal() {
        let mut reported = 0;
        let r = first_registrable(
            &strs(&["dspark:a", "dflash:b"]),
            false,
            |s| if s == "dspark:a" { Err("refused") } else { Ok(()) },
            |_, _, _| reported += 1,
        );
        assert_eq!(r, Err("refused"));
        assert_eq!(reported, 0);
    }
}

#[cfg(test)]
mod latency_tests {
    use super::CancelSet;
    use std::time::Duration;

    #[test]
    fn a_latency_request_raises_the_word_and_holds_it_for_a_while() {
        let ctl = superfluid_shm::ControlSegment::create().unwrap();
        let worker = superfluid_shm::ControlSegment::attach(
            superfluid_shm::ShmSegment::from_fd(
                {
                    use std::os::fd::FromRawFd;
                    // SAFETY: a fresh duplicate of the segment's fd, owned here.
                    unsafe { std::os::fd::OwnedFd::from_raw_fd(libc::dup(ctl.raw_fd())) }
                },
                superfluid_shm::CONTROL_BYTES,
            )
            .unwrap(),
        )
        .unwrap();
        let cancels = CancelSet::default();
        cancels.wake_with(std::sync::Arc::new(std::sync::RwLock::new(Some(ctl.signal().unwrap()))));
        assert!(!cancels.latency_recent(Duration::from_secs(300)), "nothing has come");
        cancels.latency_request();
        assert!(worker.latency_wanted() && worker.yield_requested(), "short steps, and the running tick ends at its next");
        assert!(cancels.latency_recent(Duration::from_secs(300)));
        assert!(!cancels.latency_recent(Duration::ZERO), "past the hold");
    }
}
