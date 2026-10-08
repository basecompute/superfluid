//! `superfluid serve`: the server, as a library function so that another
//! program can run it.

use crate as superfluid_daemon;

use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use superfluid_daemon::runtime_pick::{RuntimeId, RuntimePick};
use superfluid_daemon::{api, Daemon, EngineHost, SessionStore};

fn retired(flag: &str, why: &str) {
    eprintln!("superfluid: {flag} is retired — {why}");
}

fn sampling_f32(flag: &str, raw: &str, lo: f32, hi: f32) -> f32 {
    let v: f32 = raw.parse().unwrap_or_else(|_| {
        eprintln!("superfluid: {flag} expects a number, got {raw:?}");
        std::process::exit(2);
    });
    if !v.is_finite() || v < lo || v > hi {
        eprintln!("superfluid: {flag} must be a finite number in [{lo}, {hi}], got {v}");
        std::process::exit(2);
    }
    v
}

fn usage_error(msg: &str, command: &str) -> ! {
    eprintln!("superfluid: {msg}");
    eprintln!("Run 'superfluid {command} --help' for its options.");
    std::process::exit(2);
}

fn needs_value(flag: &str) -> ! {
    usage_error(&format!("{flag} needs a value"), "serve")
}

fn num<T: std::str::FromStr>(flag: &str, raw: String) -> T {
    raw.parse().unwrap_or_else(|_| usage_error(&format!("{flag} expects a number, got {raw:?}"), "serve"))
}

fn unknown_serve_flag(flag: &str) -> ! {
    let hint = superfluid_daemon::cli::suggest(flag, superfluid_daemon::cli::serve_flag_names())
        .map(|s| format!(" (did you mean {s}?)"))
        .unwrap_or_default();
    usage_error(&format!("serve has no option {flag}{hint}"), "serve")
}

fn human_bytes(n: u64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    if n as f64 >= GIB {
        format!("{:.1} GiB", n as f64 / GIB)
    } else {
        format!("{:.0} MiB", n as f64 / MIB)
    }
}

pub(crate) fn settle_park_dir(dir: &std::path::Path, parking: bool, budget_gb: u64) {
    if parking {
        let budget = budget_gb.saturating_mul(1 << 30);
        if budget == 0 {
            return;
        }
        let freed = superfluid_daemon::park::sweep(dir, budget);
        if freed > 0 {
            eprintln!(
                "superfluid: park retention sweep freed {} in {} at startup (budget {budget_gb} GiB)",
                human_bytes(freed),
                dir.display()
            );
        }
        return;
    }
    let stale: u64 = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| {
                    let n = e.file_name();
                    let n = n.to_string_lossy();
                    n.ends_with(".park") || n.ends_with(".park.tmp")
                })
                .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                .sum()
        })
        .unwrap_or(0);
    if stale > 0 {
        eprintln!(
            "superfluid: {} of park artifacts in {} are from an earlier run with --park; \
             nothing reads or reclaims them while parking is off. Delete them, or pass \
             --park to put them back under --park-budget-gb.",
            human_bytes(stale),
            dir.display()
        );
    }
}

pub const SUN_PATH_LEN: usize = if cfg!(target_os = "linux") { 108 } else { 104 };

fn listen_or_exit(flag: &str, addr: &str) -> std::net::TcpListener {
    std::net::TcpListener::bind(addr).unwrap_or_else(|e| {
        let hint = match e.kind() {
            std::io::ErrorKind::AddrInUse => " (another process listens there: stop it or pick another port, e.g. --port 8454)",
            std::io::ErrorKind::AddrNotAvailable => " (no interface on this machine has that address)",
            std::io::ErrorKind::PermissionDenied => " (a port under 1024 needs privileges)",
            _ => "",
        };
        eprintln!("superfluid: cannot listen on {flag} {addr}: {e}{hint}");
        std::process::exit(1);
    })
}

/// What a program running [`run`] in place of `superfluid serve` changes.
/// [`Embedding::default`] is `superfluid serve` itself.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The APIs served beside the OpenAI-compatible routes.
    pub http: superfluid_daemon::openai::HttpSurface,
    /// How a request's `model` picks the model that serves it.
    pub model_naming: superfluid_daemon::openai::ModelNaming,
    /// Names registered artifacts (initial, additional and model-dir models).
    /// Receives already registered names so an embedding can disambiguate
    /// variants. Explicit HTTP model IDs are not rewritten.
    pub model_name: fn(&Path, &std::collections::HashSet<String>) -> String,
    /// Sampling policy for omitted parameters; None is superfluid's default.
    pub sampling_fallback: Option<superfluid_daemon::openai::SamplingFallback>,
    /// The name and version the HTTP API reports for the server.
    pub identity: superfluid_daemon::openai::ServerIdentity,
    /// Whether the native session API listens on a unix socket (`--socket`).
    /// Without it the HTTP API is all there is, and `--no-http` is refused.
    pub session_socket: bool,
    /// The program that runs a linked runtime's worker, in place of the `superfluid-workerd`
    /// beside this executable; it passes [`crate::workerd::main`]'s arguments to it.
    pub workerd: Option<PathBuf>,
}

impl Default for Embedding {
    fn default() -> Embedding {
        Embedding {
            http: superfluid_daemon::openai::HttpSurface::default(),
            model_naming: superfluid_daemon::openai::ModelNaming::default(),
            model_name: superfluid_daemon::registry::distinct_name,
            sampling_fallback: None,
            identity: superfluid_daemon::openai::ServerIdentity::default(),
            session_socket: true,
            workerd: None,
        }
    }
}

/// Runs `superfluid serve`; `args` is its command line after `serve`.
pub fn run(args: &[String], embedding: &Embedding) {
    let mut models: Vec<PathBuf> = Vec::new();
    let mut pull_flags = superfluid_daemon::pulls::PullFlags::default();
    let mut runtime_pick = RuntimePick::default();
    let mut api_key: Option<String> = None;
    let mut rate_limit: u32 = 0;
    let mut host: Option<String> = None;
    let mut port: Option<u16> = None;
    let mut files_max_bytes: Option<u64> = None;
    let mut files_expiry: Option<u64> = None;
    let mut files_sweep: u64 = 300;
    let mut drain_timeout: u64 = 60;
    let mut nonstream_keepalive: u64 = 0;
    let mut default_max_tokens: Option<u32> = None;
    let mut log_file: Option<PathBuf> = None;
    let mut model_dir: Option<PathBuf> = None;
    let mut tool_envelope: Option<superfluid_daemon::codec::ToolEnvelope> = None;
    let mut idle_timeout: u64 = 0;
    let mut sessions: Option<PathBuf> = None;
    let mut socket: Option<PathBuf> = None;
    let mut http: Option<String> = None;
    let mut no_http = false;
    let mut max_context: Option<i32> = None;
    let mut max_context_auto_flag = false;
    let mut max_batch: u32 = 8;
    let mut op_temperature: Option<f32> = None;
    let mut op_top_p: Option<f32> = None;
    let mut op_top_k: Option<u32> = None;
    let mut op_min_p: Option<f32> = None;
    let mut op_repeat_penalty: Option<f32> = None;
    let mut op_spec_max_temperature: Option<f32> = None;
    let mut park = false;
    let mut sessions_cache = false;
    let mut park_lossy = false;
    let mut park_budget_gb: u64 = 20;
    let mut tool_lease_ms: u64 = 0;
    let mut prefix_blob_budget_mb: u64 = 0;
    let mut kv_bits_arg: Option<i32> = None;
    let mut worker_process = true;
    let mut os_pressure = true;
    let mut dialect = String::from("auto");
    let mut pressure_high: u8 = 0;
    let mut pressure_low: u8 = 0;
    let mut pin_budget_pct: u8 = superfluid_daemon::DaemonOptions::default().pin_budget_pct;
    let mut tick_decode_budget: u32 = 0;
    let mut tick_target_ms: u32 = 0;
    let mut prefill_budget: u32 = superfluid_daemon::DaemonOptions::default().prefill_budget;
    let mut starvation_ticks: u32 = 8;
    let mut class_lanes_spec: Option<String> = None;
    let mut http_default_qos = superfluid_daemon::openai::RequestQos::AGENT;
    let mut http_qos_header = true;
    let mut http_allow_batch_invariant = false;
    let mut key_policy_path: Option<PathBuf> = None;
    let mut qos_flags: Vec<&'static str> = Vec::new();
    let mut log_dir: Option<PathBuf> = None;
    let mut log_filter = String::from("info");
    let mut speculate: Option<String> = None;
    let mut speculation = superfluid_daemon::SpeculationOptions::default();
    let mut spec_bitexact: Option<bool> = None;
    let mut completion_deadline_ms: u64 = 0;
    let mut completion_bucket = superfluid_daemon::completion_bucket::BucketConfig::default();
    let mut fim_model: Option<PathBuf> = None;
    let mut fim_max_context: Option<i32> = None;
    let mut fim_max_batch: u32 = 2;
    let mut web: Option<String> = None;
    let mut web_origins: Vec<String> = Vec::new();
    let mut otlp_metrics: Option<String> = None;
    let mut otlp_interval_ms: u64 = 15_000;
    let mut otlp_endpoint: Option<String> = None;
    let mut otlp_cfg = superfluid_daemon::otlp::OtlpConfig::default();
    let mut tui_flag = false;
    let mut no_tui = false;
    let mut fleet: Option<String> = None;
    let mut fleet_auth: Option<PathBuf> = None;
    let mut fleet_listen: Option<String> = None;
    let mut fleet_policy = String::from("load-aware");
    let mut fleet_pool_high: u32 = superfluid_daemon::fleet_manager::DEFAULT_POOL_HIGH_PCT;
    let mut fleet_conns: usize = superfluid_daemon::fleet_manager::DEFAULT_CONNS_PER_NODE;
    let args = superfluid_daemon::cli::split_equals(args, superfluid_daemon::cli::serve_takes_value);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_else(|| needs_value(a));
        // SERVE-FLAGS-BEGIN: every flag `serve` reads is listed in superfluid_daemon::cli (a test checks).
        match a.as_str() {
            "--model" => models.push(PathBuf::from(val())),
            "--offline" => pull_flags.offline = true,
            flag if flag.starts_with("--pull-") => {
                let spec = &flag["--pull-".len()..];
                let (name, value) = match spec.split_once('=') {
                    Some((n, v)) => (n.to_string(), Some(v.to_string())),
                    None => (spec.to_string(), it.clone().next().filter(|v| !v.starts_with("--")).cloned()),
                };
                if value.is_some() && !spec.contains('=') {
                    it.next();
                }
                if name.is_empty() {
                    unknown_serve_flag(a);
                }
                pull_flags.options.push((name, value));
            }
            "--runtime" => {
                let v = val();
                if let Err(e) = runtime_pick.add(&v) {
                    eprintln!("superfluid: --runtime: {e}");
                    std::process::exit(2);
                }
            }
            "--api-key" => api_key = Some(val()),
            "--rate-limit" => rate_limit = num(a, val()),
            "--files-max-bytes" => {
                files_max_bytes = Some(num(a, val()))
            }
            "--files-expiry" => files_expiry = Some(num(a, val())),
            "--files-sweep" => files_sweep = num(a, val()),
            "--drain-timeout" => drain_timeout = num(a, val()),
            "--nonstream-keepalive" => {
                nonstream_keepalive = num(a, val())
            }
            "--max-tokens" => {
                default_max_tokens = Some(num(a, val()))
            }
            "--log-file" => log_file = Some(PathBuf::from(val())),
            "--model-dir" => model_dir = Some(PathBuf::from(val())),
            "--tool-call-parser" => {
                let v = val();
                if v != "auto" {
                    tool_envelope = Some(
                        superfluid_daemon::codec::ToolEnvelope::from_name(&v).unwrap_or_else(|| {
                            eprintln!(
                                "superfluid: unknown --tool-call-parser '{v}' (have: auto, {})",
                                superfluid_daemon::codec::ToolEnvelope::names().join(", ")
                            );
                            std::process::exit(2);
                        }),
                    );
                }
            }
            "--idle-timeout" => idle_timeout = num(a, val()),
            "--host" => host = Some(val()),
            "--port" => port = Some(num(a, val())),
            "--max-batch-size" => max_batch = num(a, val()),
            "--request-timeout" => {
                completion_deadline_ms = num(a, val())
            }
            "--verbose" | "-v" => log_filter = "debug".into(),
            "--help" | "-h" => {
                print!("{}", superfluid_daemon::cli::serve_help());
                return;
            }
            "--sessions" => sessions = Some(PathBuf::from(val())),
            "--socket" => socket = Some(PathBuf::from(val())),
            "--http" => http = Some(val()),
            "--no-http" => no_http = true,
            "--max-context" => {
                let v = val();
                max_context_auto_flag = v == "auto";
                max_context = if v == "auto" { None } else { Some(num(a, v)) };
            }
            "--max-batch" => max_batch = num(a, val()),
            "--temperature" => op_temperature = Some(sampling_f32("--temperature", &val(), 0.0, 2.0)),
            "--top-p" => op_top_p = Some(sampling_f32("--top-p", &val(), 0.0, 1.0)),
            "--top-k" => op_top_k = Some(num(a, val())),
            "--min-p" => op_min_p = Some(sampling_f32("--min-p", &val(), 0.0, 1.0)),
            "--repeat-penalty" => {
                op_repeat_penalty = Some(sampling_f32("--repeat-penalty", &val(), 0.0, 2.0))
            }
            "--park" => park = true,
            "--no-park" => park = false,
            "--sessions-cache" => sessions_cache = true,
            "--park-lossy" => {
                park = true;
                park_lossy = true;
            }
            "--park-budget-gb" => park_budget_gb = num(a, val()),
            "--tool-lease-ms" => tool_lease_ms = num(a, val()),
            "--prefix-blob-budget-mb" => {
                prefix_blob_budget_mb = num(a, val())
            }
            "--kv-bits" => kv_bits_arg = Some(num(a, val())),
            "--dialect" => dialect = val(),
            "--worker-process" => worker_process = true,
            "--no-worker-process" => worker_process = false,
            "--os-pressure" => os_pressure = true,
            "--no-os-pressure" => os_pressure = false,
            "--pressure-high" => pressure_high = num(a, val()),
            "--pressure-low" => pressure_low = num(a, val()),
            "--pin-budget-pct" => pin_budget_pct = num(a, val()),
            "--log-dir" => log_dir = Some(PathBuf::from(val())),
            "--basert-lib" => std::env::set_var("BASERT_LIB", val()),
            "--speculate" => {
                let v = val();
                speculate = if v == "off" { None } else { Some(v) };
            }
            "--spec-draft-tokens" => {
                let n: u32 = num(a, val());
                if !(1..=15).contains(&n) {
                    eprintln!("superfluid: --spec-draft-tokens must be in 1..=15");
                    std::process::exit(2);
                }
                speculation.max_draft = Some(n);
            }
            "--spec-adaptive" => speculation.adaptive = Some(true),
            "--no-spec-adaptive" => speculation.adaptive = Some(false),
            "--spec-min-yield" => {
                let v: f64 = num(a, val());
                if !v.is_finite() || v < 0.0 {
                    eprintln!("superfluid: --spec-min-yield must be finite and non-negative");
                    std::process::exit(2);
                }
                speculation.min_yield = Some(v);
            }
            "--spec-yield-rounds" => {
                let n: u32 = num(a, val());
                if n == 0 {
                    eprintln!("superfluid: --spec-yield-rounds must be at least 1");
                    std::process::exit(2);
                }
                speculation.yield_rounds = Some(n);
            }
            "--spec-throughput-gate" => speculation.throughput_gate = Some(true),
            "--no-spec-throughput-gate" => speculation.throughput_gate = Some(false),
            "--spec-gate-probe-tokens" => {
                speculation.gate_probe_tokens = Some(num(a, val()))
            }
            "--spec-max-temperature" => {
                op_spec_max_temperature = Some(sampling_f32("--spec-max-temperature", &val(), 0.0, 2.0))
            }
            "--spec-min-speedup" => {
                let v: f64 = num(a, val());
                if !v.is_finite() || v < 1.0 {
                    eprintln!("superfluid: --spec-min-speedup must be finite and at least 1.0");
                    std::process::exit(2);
                }
                speculation.min_speedup = Some(v);
            }
            "--spec-gate-reprobe" => {
                speculation.gate_reprobe = Some(num(a, val()))
            }
            "--spec-gate-reprobe-max" => {
                speculation.gate_reprobe_max = Some(num(a, val()))
            }
            "--dspark-confidence" => {
                let v: f32 = num(a, val());
                if !v.is_finite() || v <= 0.0 {
                    eprintln!("superfluid: --dspark-confidence must be finite and positive");
                    std::process::exit(2);
                }
                speculation.dspark_confidence = Some(v);
            }
            "--spec-bitexact" => spec_bitexact = Some(true),
            "--no-spec-bitexact" => spec_bitexact = Some(false),
            "--log-filter" => log_filter = val(),
            "--tick-decode-budget" => {
                tick_decode_budget = num(a, val())
            }
            "--tick-target-ms" => tick_target_ms = num(a, val()),
            "--prefill-budget" => prefill_budget = num(a, val()),
            "--starvation-ticks" => starvation_ticks = num(a, val()),
            "--class-lanes" => {
                class_lanes_spec = Some(val());
                qos_flags.push("--class-lanes");
            }
            "--http-default-qos" => {
                qos_flags.push("--http-default-qos");
                let v = val();
                http_default_qos.class = superfluid_daemon::qos::parse(&v).unwrap_or_else(|| {
                    eprintln!(
                        "superfluid: --http-default-qos expects one of {}, got {v:?}",
                        superfluid_daemon::qos::NAMES.join(", ")
                    );
                    std::process::exit(2);
                });
            }
            "--no-http-qos-header" => {
                http_qos_header = false;
                qos_flags.push("--no-http-qos-header");
            }
            "--http-allow-batch-invariant" => {
                http_allow_batch_invariant = true;
                qos_flags.push("--http-allow-batch-invariant");
            }
            "--key-policy" => key_policy_path = Some(PathBuf::from(val())),
            "--completion-deadline-ms" => {
                completion_deadline_ms = num(a, val())
            }
            "--completion-rate" => {
                completion_bucket.rate_per_sec = num(a, val())
            }
            "--completion-burst" => {
                completion_bucket.burst = num(a, val())
            }
            "--fim-model" => fim_model = Some(PathBuf::from(val())),
            "--fim-max-context" => {
                fim_max_context = Some(num(a, val()))
            }
            "--fim-max-batch" => fim_max_batch = num(a, val()),
            "--web" => web = Some(val()),
            "--web-origin" => web_origins.push(val()),
            "--otlp-metrics" => otlp_metrics = Some(val()),
            "--tui" => tui_flag = true,
            "--no-tui" => no_tui = true,
            "--otlp-interval-ms" => otlp_interval_ms = num(a, val()),
            "--otlp-endpoint" => otlp_endpoint = Some(val()),
            "--otlp-header" => {
                let kv = val();
                let (k, v) = kv.split_once('=').unwrap_or_else(|| usage_error(&format!("--otlp-header expects k=v, got {kv:?}"), "serve"));
                otlp_cfg.headers.push((k.trim().to_string(), v.trim().to_string()));
            }
            "--otlp-service-name" => otlp_cfg.service_name = val(),
            "--otlp-filter" => otlp_cfg.filter = val(),
            "--otlp-queue" => otlp_cfg.queue = num(a, val()),
            "--otlp-batch-ms" => {
                let ms = num(a, val());
                otlp_cfg.interval = std::time::Duration::from_millis(ms);
            }
            "--otlp-timeout-ms" => {
                let ms = num(a, val());
                otlp_cfg.timeout = std::time::Duration::from_millis(ms);
            }
            "--fleet" => fleet = Some(val()),
            "--fleet-auth" => fleet_auth = Some(PathBuf::from(val())),
            "--fleet-listen" => fleet_listen = Some(val()),
            "--fleet-policy" => fleet_policy = val(),
            "--fleet-pool-high" => fleet_pool_high = num(a, val()),
            "--fleet-conns-per-node" => {
                fleet_conns = num(a, val())
            }
            "--paged-kv" => retired(a, "paged KV is always on"),
            "--prefix-cache" => retired(a, "prefix reuse is always on"),
            "--continuous-batching" => {
                if let Some(n) = it.clone().next().and_then(|v| v.parse::<u32>().ok()) {
                    let _ = it.next();
                    max_batch = n;
                    retired(a, "always on; its width became --max-batch");
                } else {
                    retired(a, "continuous batching is always on");
                }
            }
            "--prefix-cache-file" | "--prefix-cache-save-interval" => {
                let _ = val();
                retired(
                    a,
                    "superseded by the WAL session log (tokens, always durable) and the \
                     in-memory prefix cache; KV on disk is --park, which is OFF by default \
                     and applies to sessions resumed by id through the session API",
                );
            }
            "--files-dir" => {
                let _ = val();
                retired(a, "the file store lives under --sessions");
            }
            "--media-dir" => {
                let _ = val();
                retired(
                    a,
                    "superfluid accepts only data: image URLs and never reads a local path, so there is nothing to allow-list",
                );
            }
            "--metallib" | "--prefill-chunk" | "--gpu-wait-timeout-ms"
            | "--decode-replay" => {
                let _ = val();
                retired(a, "an engine-level knob superfluid does not expose");
            }
            "--paged-weights" | "--no-paged-weights-retry" | "--no-baked-decode" => {
                retired(a, "an engine-level knob superfluid does not expose")
            }
            other if !other.starts_with('-') => models.push(PathBuf::from(other)),
            _ => unknown_serve_flag(a),
        }
        // SERVE-FLAGS-END
    }
    if models.is_empty() {
        usage_error(
            "serve needs a model: superfluid serve <path|org/model[:tag]>, e.g. \
             superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
            "serve",
        );
    }
    superfluid_daemon::speculate_resolve::allow_catalog_fetch(!pull_flags.offline);
    let catalog = Arc::new(match &embedding.workerd {
        Some(workerd) => superfluid_daemon::runtimes::Catalog::with_workerd(workerd.clone()),
        None => superfluid_daemon::runtimes::Catalog::from_env(),
    });
    let sessions = Some(sessions.unwrap_or_else(|| superfluid_daemon::cli::default_sessions(catalog.home())));
    let socket = if embedding.session_socket {
        Some(socket.unwrap_or_else(|| superfluid_daemon::cli::default_socket(catalog.home(), SUN_PATH_LEN)))
    } else {
        if socket.is_some() {
            usage_error("--socket: this server has no session socket", "serve");
        }
        if no_http {
            usage_error("--no-http leaves nothing to serve: this server has no session socket", "serve");
        }
        None
    };
    if no_http && (http.is_some() || host.is_some() || port.is_some()) {
        usage_error("--no-http contradicts --http/--host/--port", "serve");
    }
    let mut pulled_names: std::collections::HashMap<PathBuf, String> = std::collections::HashMap::new();
    for m in models.iter_mut() {
        let token = m.to_string_lossy().into_owned();
        if !superfluid_daemon::pulls::looks_like_id(&token) {
            continue;
        }
        let runtime = match runtime_pick.explicit(m) {
            Some(name) => catalog.named(name).ok().or_else(|| RuntimeId::parse(name)).unwrap_or_else(|| {
                eprintln!("superfluid: --runtime '{name}' is not a runtime id");
                std::process::exit(2);
            }),
            None => RuntimeId::new(superfluid_daemon::cli::runtime_for_id(&token)),
        };
        let path = superfluid_daemon::pulls::pull(&catalog, runtime, &token, &pull_flags).unwrap_or_else(|e| {
            eprintln!("superfluid: {e}");
            std::process::exit(1);
        });
        if let Err(e) = runtime_pick.add(&format!("{}={}", path.display(), runtime.name())) {
            eprintln!("superfluid: {e}");
            std::process::exit(2);
        }
        pulled_names.insert(path.clone(), superfluid_daemon::pulls::served_name(&token));
        *m = path;
    }
    if !pull_flags.options.is_empty() && pulled_names.is_empty() {
        eprintln!("superfluid: --pull-* options apply to models given by id; no --model here is one");
        std::process::exit(2);
    }
    let (Some(model), Some(sessions)) = (models.first().cloned(), sessions) else {
        unreachable!("a model and sessions are both set above");
    };
    if http.is_none() && !no_http {
        http = Some(format!(
            "{}:{}",
            host.unwrap_or_else(|| "127.0.0.1".into()),
            port.unwrap_or(8453)
        ));
    }
    if let Some(socket) = socket.as_ref().filter(|s| s.as_os_str().len() >= SUN_PATH_LEN) {
        eprintln!(
            "superfluid: --socket {} is {} bytes; a unix socket path must be under {SUN_PATH_LEN} bytes here \
             (pick a shorter one)",
            socket.display(),
            socket.as_os_str().len()
        );
        std::process::exit(2);
    }
    for (flag, addr) in [("--http", &http), ("--web", &web)] {
        if let Some(addr) = addr {
            drop(listen_or_exit(flag, addr));
        }
    }
    let picked = runtime_pick.explicit(&model).is_some();
    let runtime = runtime_pick
        .resolve(&model, &catalog)
        .map_err(|e| if picked { e } else { not_there(&model).unwrap_or(e) })
        .and_then(|rt| match (catalog.require(rt), not_there(&model)) {
            (Ok(rt), _) => Ok(rt),
            (Err(_), Some(why)) => Err(why),
            (Err(_), None) => superfluid_daemon::pulls::ensure_installed(&catalog, rt, pull_flags.offline),
        })
        .and_then(|rt| refuse_statically(&rt, park_lossy, kv_bits_arg, speculate.as_deref()).map(|()| rt))
        .unwrap_or_else(|e| {
            eprintln!("superfluid: {e}");
            std::process::exit(2);
        });
    eprintln!("superfluid: runtime {} serves {} ({})", runtime.id.name(), model.display(), describe_worker(&runtime));
    if let Some(on) = spec_bitexact {
        std::env::set_var("BASERT_SPEC_BITEXACT", if on { "1" } else { "0" });
    }
    let (speculate_primary, speculate_fallbacks): (Option<String>, Vec<String>) = match speculation_for(
        speculate.as_deref(),
        &runtime.static_capabilities(),
        &model,
        runtime.id.name(),
    )
    .map(|d| superfluid_daemon::speculate_resolve::resolve_all(d, &model))
    {
        None => (None, Vec::new()),
        Some(Ok(mut ranked)) => {
            if let Some(r) = ranked.first() {
                eprintln!("superfluid: speculation: {r}");
            } else {
                let dirs = superfluid_daemon::speculate_resolve::auto_search_dirs(&model);
                let dirs: Vec<_> = dirs.iter().map(|(d, _)| d.display().to_string()).collect();
                eprintln!(
                    "superfluid: speculation: auto found no head in {} and no fitting drafter in {} — decoding plainly",
                    model.display(),
                    dirs.join(", ")
                );
            }
            let primary = (!ranked.is_empty()).then(|| ranked.remove(0));
            (primary, ranked)
        }
        Some(Err(e)) => {
            eprintln!("superfluid: {e}");
            std::process::exit(2);
        }
    };
    if op_spec_max_temperature.is_some() && speculate.is_none() {
        eprintln!("superfluid: --spec-max-temperature ignored: no --speculate strategy is configured");
    }
    if tui_flag && !cfg!(feature = "tui") {
        eprintln!(
            "superfluid: --tui: this build carries no TUI (build superfluid-daemon with the `tui` feature, \
             which `basert` includes)"
        );
        std::process::exit(2);
    }
    let class_lanes = match class_lanes_spec.as_deref() {
        None => [0; 4],
        Some(spec) => superfluid_daemon::qos::parse_class_lanes(spec, max_batch as usize).unwrap_or_else(|e| {
            eprintln!("superfluid: --class-lanes: {e}");
            std::process::exit(2);
        }),
    };
    let key_policy = key_policy_path.as_deref().map(|p| {
        let table = superfluid_daemon::keypolicy::KeyTable::load(p, api_key.as_deref())
            .unwrap_or_else(|e| {
                eprintln!("superfluid: --key-policy {e}");
                std::process::exit(2);
            });
        eprintln!("superfluid: key policy: {} key(s) from {}", table.len(), p.display());
        Arc::new(table)
    });
    if let Err(e) = completion_bucket.validate() {
        eprintln!("superfluid: --completion-rate/--completion-burst: {e}");
        std::process::exit(2);
    }
    if fim_max_batch == 0 {
        eprintln!("superfluid: --fim-max-batch must be at least 1");
        std::process::exit(2);
    }
    if sessions_cache && park {
        eprintln!(
            "superfluid: --sessions-cache discards the session log at each start, which \
             --park{} needs to resume parked sessions. Use one or the other.",
            if park_lossy { "-lossy" } else { "" }
        );
        std::process::exit(2);
    }
    if models.len() > 1 && !worker_process {
        eprintln!("superfluid: {} models — enabling --worker-process (each needs its own address space)", models.len());
        worker_process = true;
    }
    if fim_model.is_some() && !worker_process {
        eprintln!("superfluid: --fim-model needs its own worker — enabling --worker-process");
        worker_process = true;
    }
    if model_dir.is_some() && !worker_process {
        eprintln!("superfluid: --model-dir needs a per-model worker — enabling --worker-process");
        worker_process = true;
    }
    if let Some(path) = &log_file {
        use std::os::fd::AsRawFd;
        match std::fs::OpenOptions::new().create(true).append(true).open(path) {
            Ok(f) => {
                // SAFETY: `f` is an open, valid fd for the duration of the
                // call; dup2 onto STDERR_FILENO is the documented way to
                // redirect a stream. The original stderr is closed by dup2.
                unsafe {
                    libc::dup2(f.as_raw_fd(), libc::STDERR_FILENO);
                }
                drop(f);
            }
            Err(e) => {
                eprintln!("superfluid: could not open --log-file {}: {e}", path.display());
                std::process::exit(1);
            }
        }
    }
    if api_key.is_none() && key_policy.is_none() {
        let loopback = |addr: &str| {
            addr.starts_with("localhost:") || addr.parse::<std::net::SocketAddr>().is_ok_and(|a| a.ip().is_loopback())
        };
        if http.as_deref().is_none_or(loopback) && web.as_deref().is_none_or(loopback) {
            eprintln!("superfluid: no --api-key: the HTTP API is open to every process on this machine");
        } else {
            eprintln!(
                "superfluid: WARNING --api-key not set. Authentication is OFF — anyone \
                 who can reach this port can use the API. Bind loopback or set --api-key."
            );
        }
    }
    let fleet_mode = fleet.is_some() || fleet_listen.is_some();
    if park && !fleet_mode {
        let budget = if park_budget_gb == 0 {
            "NO budget: --park-budget-gb 0 disables the retention sweep, so the \
             directory grows without limit"
                .to_string()
        } else {
            format!("budget {park_budget_gb} GiB, oldest evicted first")
        };
        eprintln!(
            "superfluid: WARNING --park is on. Every finishing DURABLE session copies its \
             whole KV off the GPU on the tick thread before it retires — seconds and \
             gigabytes at agent context, and every other lane waits — then keeps an \
             artifact under <sessions>/park ({budget}). A /v1/* request mints a fresh \
             session and can never resume it, so its TERMINAL park is skipped; one \
             PREEMPTED under load still parks, because it is requeued under the same id \
             and adopts that artifact. This pays off only for sessions you resume by id \
             through the session API."
        );
    }

    let model_name = pulled_names.get(&model).cloned().unwrap_or_else(|| (embedding.model_name)(&model, &Default::default()));
    if prefix_blob_budget_mb > 0 {
        std::env::set_var("BASERT_PREFIX_BLOB_BUDGET_MB", prefix_blob_budget_mb.to_string());
    }
    if std::env::var_os("BASERT_YARN").is_some() {
        eprintln!(
            "superfluid: WARNING: BASERT_YARN is set but has no effect on this path — the rope \
             kernels disable YaRN under paged/batched dispatch, which serving always uses. \
             Long requests run UNSCALED past the trained window."
        );
    }

    let tui_on =
        cfg!(feature = "tui") && !no_tui && (tui_flag || std::io::IsTerminal::is_terminal(&std::io::stdout()));
    let tui_ring = if tui_on {
        Some(superfluid_daemon::telemetry::new_log_ring())
    } else {
        None
    };
    #[cfg(feature = "tui")]
    let stderr_capture = tui_ring.as_ref().and_then(|ring| {
        match superfluid_daemon::tui::capture_stderr(ring.clone()) {
            Ok(guard) => Some(guard),
            Err(e) => {
                eprintln!("superfluid: stderr capture unavailable, engine output may overdraw the TUI: {e}");
                None
            }
        }
    });
    #[cfg(feature = "tui")]
    let worker_stderr = stderr_capture.as_ref().map(|c| c.worker_stderr());
    #[cfg(not(feature = "tui"))]
    let worker_stderr: Option<std::sync::Arc<std::os::fd::OwnedFd>> = None;
    let log_dir_eff = log_dir.or(if tui_on { Some(sessions.join("logs")) } else { None });
    if let Some(url) = &otlp_metrics {
        if let Err(e) = superfluid_daemon::otlp::validate_http_url(url) {
            eprintln!("superfluid: --otlp-metrics: {e}");
            std::process::exit(2);
        }
        if otlp_interval_ms == 0 {
            eprintln!("superfluid: --otlp-interval-ms must be > 0");
            std::process::exit(2);
        }
    }
    if fleet_mode && otlp_endpoint.is_some() {
        eprintln!(
            "superfluid: --otlp-endpoint is not supported in fleet-head mode (the fleet head \
             has no scheduler to trace); set it on each node's superfluid instead"
        );
        std::process::exit(2);
    }
    let otlp = otlp_endpoint.map(|endpoint| {
        let mut cfg = otlp_cfg;
        cfg.endpoint = endpoint;
        if let Err(e) = cfg.validate() {
            eprintln!("superfluid: --otlp-endpoint: {e}");
            std::process::exit(2);
        }
        eprintln!(
            "superfluid: pushing operational traces to {} ({}an EGRESS: ids, lengths, timings and \
             codes only, never session content; spans: {})",
            cfg.traces_url(),
            if cfg.is_loopback() { "" } else { "NON-LOOPBACK, " },
            cfg.filter
        );
        cfg
    });
    use superfluid_daemon::telemetry::{LogConfig, LogSink};
    let sink = match log_dir_eff {
        Some(dir) => LogSink::File(LogConfig {
            dir,
            filter: log_filter,
            ..Default::default()
        }),
        None => LogSink::Stderr(log_filter),
    };
    if let Err(e) = superfluid_daemon::telemetry::init_sinks(sink, tui_ring.clone(), otlp) {
        eprintln!("superfluid: telemetry: {e}");
    }
    std::fs::create_dir_all(&sessions).expect("create sessions dir");
    let _sessions_lock = if !fleet_mode { Some(lock_sessions_dir(&sessions)) } else { None };
    if sessions_cache && !fleet_mode {
        rotate_session_wal(&sessions);
        if let Ok(rd) = std::fs::read_dir(sessions.join("models")) {
            for dir in rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|d| d.is_dir()) {
                rotate_session_wal(&dir);
            }
        }
    }
    if !fleet_mode {
        settle_park_dir(&sessions.join("park"), park, park_budget_gb);
        if let Ok(rd) = std::fs::read_dir(sessions.join("models")) {
            let mut dirs: Vec<std::path::PathBuf> = rd
                .filter_map(|e| e.ok())
                .map(|e| e.path().join("park"))
                .filter(|d| d.is_dir())
                .collect();
            dirs.sort();
            for dir in dirs {
                settle_park_dir(&dir, park, park_budget_gb);
            }
        }
    }

    if let Some(b) = kv_bits_arg {
        if ![0, 4, 8, 16, 84].contains(&b) {
            eprintln!("--kv-bits {b}: expected 0 (auto), 4 (q4_0), 8 (q8_0), 16 (f16) or 84 (K q8_0 / V q4_0)");
            std::process::exit(2);
        }
    }
    let kv_bits = match (park_lossy, kv_bits_arg) {
        (true, None) | (true, Some(16)) => 16,
        (true, Some(other)) => {
            eprintln!("--park-lossy requires F16 KV pools; --kv-bits {other} conflicts");
            std::process::exit(2);
        }
        (false, v) => v.unwrap_or(0),
    };

    let dir_default = max_context.is_none() && !max_context_auto_flag && model_dir.is_some();
    // A fleet head holds no model; its window is the nodes', learned when it reaches them.
    let max_context_auto = max_context.is_none() && !dir_default && !fleet_mode;
    let sizing_lanes = if fim_model.is_some() { max_batch.max(fim_max_batch) } else { max_batch };
    // (the window each lane's share of the device holds, the model's
    // trained window)
    let mut worker_sized: Option<(u64, u64)> = None;
    let unsized_why: Option<(String, String)> = if max_context_auto {
        let sizing = runtime.id.base();
        let elsewhere = models.iter().chain(fim_model.iter()).find_map(|m| {
            let on = if fim_model.as_ref() == Some(m) { catalog.reader_of(m) } else { runtime_pick.resolve(m, &catalog) };
            on.ok().filter(|id| id.base() != sizing).map(|id| (m, id))
        });
        match elsewhere {
            Some((m, id)) => Some((
                format!(
                    "--max-context auto sizes the window for the {} runtime, and {} is served by {}",
                    sizing.name(),
                    m.display(),
                    id.base().name()
                ),
                format!(
                    "not sized for this device: the {} runtime sizes it, and {} is served by {}",
                    sizing.name(),
                    m.display(),
                    id.base().name()
                ),
            )),
            None if !superfluid_daemon::linked::sizes_context(sizing) => {
                let set: Vec<&PathBuf> = models.iter().chain(fim_model.iter()).collect();
                let sized = set
                    .iter()
                    .map(|m| superfluid_daemon::runtimes::sizing_of(&runtime, m))
                    .collect::<Result<Vec<_>, String>>()
                    .and_then(|each| {
                        let trained = each.iter().map(|s| s.trained_context).min().unwrap_or(0);
                        superfluid_adapter_kit::sizing::window(&each, sizing_lanes).map(|n| (n, trained)).ok_or_else(|| {
                            format!("the {} runtime reported no device budget to size a window against", sizing.name())
                        })
                    });
                match sized {
                    Ok(n) => {
                        worker_sized = Some(n);
                        None
                    }
                    Err(why) => Some((format!("--max-context auto: {why}"), why)),
                }
            }
            None => None,
        }
    } else {
        None
    };
    let max_context_auto = max_context_auto && unsized_why.is_none();
    let ctx = if let Some(n) = max_context {
        n
    } else if fleet_mode {
        8192
    } else if dir_default {
        eprintln!(
            "superfluid: context window 8192 tokens (--model-dir loads models on demand at this window, so it is not \
             sized to fill the device with the startup model; --max-context N|auto overrides)"
        );
        8192
    } else if let Some((asked, kept)) = unsized_why {
        if max_context_auto_flag {
            eprintln!("superfluid: {asked}: pass --max-context <tokens>");
            std::process::exit(2);
        }
        eprintln!("superfluid: context window 8192 tokens ({kept}; --max-context N pins it)");
        8192
    } else if let Some((per_lane, trained)) = worker_sized {
        // One conversation may grow past its share of the pool, up to the trained window.
        let whole = per_lane.saturating_mul(u64::from(sizing_lanes.max(1)));
        let n = i32::try_from(whole.min(trained.max(per_lane))).unwrap_or(i32::MAX);
        eprintln!(
            "superfluid: context window {n} tokens (auto: sized for this device, {per_lane} for each of {sizing_lanes} lanes{}{}, \
             one conversation up to the trained {trained}; --max-context N pins it)",
            if models.len() > 1 { format!(", {} models", models.len()) } else { String::new() },
            if fim_model.is_some() { ", FIM satellite included" } else { "" },
        );
        n
    } else {
        use superfluid_daemon::speculate_resolve::{resolve_all, sizing_path, SpecLoad};
        let ranked: Vec<Vec<String>> = models
            .iter()
            .map(|m| {
                if *m == model {
                    speculate_primary.iter().chain(speculate_fallbacks.iter()).cloned().collect()
                } else {
                    speculate.as_deref().and_then(|d| resolve_all(d, m).ok()).unwrap_or_default()
                }
            })
            .collect();
        let sizing_models: Vec<PathBuf> = models.iter().cloned().chain(fim_model.iter().cloned()).collect();
        let lanes = if fim_model.is_some() { max_batch.max(fim_max_batch) } else { max_batch } as i32;
        let speculator_entry = |i: usize, r: &str| -> Option<(PathBuf, bool, Option<usize>)> {
            let (path, load) = sizing_path(r, &models[i])?;
            Some((path, load == SpecLoad::EmbeddedHead, (load == SpecLoad::Sidecar).then_some(i)))
        };
        let speculators_for = |choice: &[Option<&String>]| -> Vec<(PathBuf, bool, Option<usize>)> {
            (0..models.len()).filter_map(|i| speculator_entry(i, choice[i].or(ranked[i].first())?)).collect()
        };
        let sizing = runtime.id.base();
        let size = |speculators: &[(PathBuf, bool, Option<usize>)]| {
            superfluid_daemon::linked::suggest_max_context(sizing, &sizing_models, speculators, lanes, kv_bits).unwrap_or_else(|e| {
                eprintln!("superfluid: {e}");
                std::process::exit(2);
            })
        };
        let first_picks: Vec<Option<&String>> = vec![None; models.len()];
        let n_spec = speculators_for(&first_picks).len();
        const MAX_COMBINATIONS: usize = 16;
        let counts: Vec<usize> = ranked.iter().map(|c| c.len().max(1)).collect();
        let few_enough = counts
            .iter()
            .try_fold(1usize, |p, &c| p.checked_mul(c).filter(|p| *p <= MAX_COMBINATIONS))
            .is_some();
        let (sized, note) = if few_enough {
            let mut best: Option<(i32, Vec<Option<&String>>)> = None;
            let mut idx = vec![0usize; models.len()];
            loop {
                let choice: Vec<Option<&String>> =
                    idx.iter().enumerate().map(|(i, &k)| (k > 0).then(|| &ranked[i][k])).collect();
                if let Some(n) = size(&speculators_for(&choice)) {
                    if best.as_ref().is_none_or(|(b, _)| n < *b) {
                        best = Some((n, choice));
                    }
                }
                let mut i = 0;
                while i < idx.len() {
                    idx[i] += 1;
                    if idx[i] < counts[i] {
                        break;
                    }
                    idx[i] = 0;
                    i += 1;
                }
                if i == idx.len() {
                    break;
                }
            }
            let fallbacks: Vec<&str> =
                best.iter().flat_map(|(_, c)| c.iter().flatten().map(|s| s.as_str())).collect();
            let note = if fallbacks.is_empty() {
                String::new()
            } else {
                format!(", sized for the fallback {}", fallbacks.join(" + "))
            };
            (best.map(|(n, _)| n), note)
        } else {
            let all: Vec<(PathBuf, bool, Option<usize>)> = (0..models.len())
                .flat_map(|i| ranked[i].iter().filter_map(move |r| speculator_entry(i, r)))
                .collect();
            let note = format!(", every one of {} drafter candidates charged at once", all.len());
            (size(&all), note)
        };
        match sized {
            Some(n) => {
                eprintln!(
                    "superfluid: context window {n} tokens (auto: sized for this device, {lanes} lanes{}{}{}; --max-context N pins it)",
                    if n_spec == 0 { String::new() } else { format!(", {n_spec} speculator(s)") },
                    if fim_model.is_some() { ", FIM satellite included" } else { "" },
                    note
                );
                n
            }
            None => {
                eprintln!(
                    "superfluid: context window 8192 tokens (auto could not size this device/model set; --max-context N pins it)"
                );
                8192
            }
        }
    }
    .max(512);
    if let Some(n) = max_context {
        eprintln!("superfluid: context window {ctx} tokens (--max-context {n})");
    }
    // A worker process starts loading now, while the tokenizer and chat
    // template are built: the two do not depend on each other, and the load
    // is most of startup.
    let spawns_worker = !runtime.in_process_capable() || (worker_process && runtime.worker.bin.is_file());
    let early_host = (!fleet_mode && spawns_worker).then(|| {
        let spec = superfluid_daemon::WorkerSpec {
            bin: runtime.worker.bin.clone(),
            args: runtime.serve_args(&model, ctx, max_batch, kv_bits),
            stderr: worker_stderr.clone(),
        };
        std::thread::spawn(move || EngineHost::spawn_process(spec))
    });
    let tokenizer = open_tokenizer(&runtime, &model).unwrap_or_else(|e| {
        eprintln!("superfluid: {e}");
        std::process::exit(2);
    });
    let codec: Box<dyn superfluid_daemon::TextCodec + Send + Sync> = match codec_for_dialect(&dialect, tokenizer) {
        Ok((c, kind)) => {
            if dialect == "auto" {
                match kind {
                    "template" => eprintln!("superfluid: serving via the model's own chat template"),
                    "raw" => eprintln!(
                        "superfluid: model has no chat template; serving raw (chat routes will refuse)"
                    ),
                    _ => {}
                }
            }
            c
        }
        Err(e) => {
            eprintln!("superfluid: {e}");
            std::process::exit(2);
        }
    };

    if fleet_mode {
        let fleet_addrs = fleet.clone().unwrap_or_default();
        use superfluid_daemon::fleet_manager::{FleetManager, NodeSpec, PlacementPolicy};
        let auth: Vec<u8> = match &fleet_auth {
            Some(pth) => std::fs::read(pth)
                .map(|t| t.trim_ascii().to_vec())
                .unwrap_or_else(|e| {
                    eprintln!("superfluid: cannot read --fleet-auth {}: {e}", pth.display());
                    std::process::exit(2);
                }),
            None => superfluid_daemon::fleet_token::read(catalog.home()).unwrap_or_default(),
        };
        if auth.is_empty() {
            eprintln!("superfluid: no fleet token (`superfluid fleet token` makes one): the links to the nodes are not encrypted, and only nodes on this machine may join");
        }
        let specs: Vec<NodeSpec> = fleet_addrs
            .split(',')
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .map(|a| NodeSpec { addr: a.to_string(), auth: auth.clone() })
            .collect();
        if specs.is_empty() && fleet_listen.is_none() {
            eprintln!("superfluid: --fleet needs at least one host:port, or --fleet-listen for nodes that join");
            std::process::exit(2);
        }
        let policy = match fleet_policy.as_str() {
            "load-aware" => PlacementPolicy::LoadAware,
            "least-loaded" => PlacementPolicy::LeastLoaded,
            "round-robin" => PlacementPolicy::RoundRobin,
            other => {
                eprintln!(
                    "superfluid: unknown --fleet-policy '{other}' \
                     (load-aware|least-loaded|round-robin)"
                );
                std::process::exit(2);
            }
        };
        let Some(addr) = http else {
            eprintln!("superfluid: --fleet requires --http <addr> (the fleet OpenAI endpoint)");
            std::process::exit(2);
        };
        let lease = superfluid_proto::linkf::Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
        let n_nodes = specs.len();
        let manager = Arc::new(FleetManager::with_options(
            specs,
            policy,
            lease,
            fleet_pool_high,
            fleet_conns,
        ));
        let mut reached = 0;
        for node in manager.probe() {
            match node {
                Ok(n) => {
                    reached += 1;
                    eprintln!(
                        "superfluid: fleet node '{}' at {}: {}, context window {} tokens, {} lanes",
                        n.identity,
                        n.addr,
                        n.models.join(", "),
                        n.max_context_tokens,
                        n.max_lanes
                    );
                    if !n.models.contains(&model_name) {
                        eprintln!(
                            "superfluid: fleet node '{}' serves {}, not {model_name}, so nothing is placed \
                             there; start it with --model-name {model_name}",
                            n.identity,
                            n.models.join(", ")
                        );
                    }
                }
                Err(e) => eprintln!("superfluid: {e}; retrying it in the background"),
            }
        }
        if max_context.is_none() {
            match manager.advertised_context(&model_name) {
                Some(n) => eprintln!("superfluid: context window {n} tokens (the largest node's)"),
                None => eprintln!(
                    "superfluid: context window unknown until a node serving {model_name} answers"
                ),
            }
        }
        manager.spawn_keepalive();
        let _announced = fleet_listen.as_deref().map(|listen| {
            let listener = std::net::TcpListener::bind(listen).unwrap_or_else(|e| {
                eprintln!("superfluid: --fleet-listen {listen}: {e}");
                std::process::exit(1);
            });
            let port = listener.local_addr().map(|a| a.port()).unwrap_or_default();
            eprintln!("superfluid: fleet head takes joining nodes on {listen}");
            let (m, a, name) = (Arc::clone(&manager), auth.clone(), model_name.clone());
            std::thread::spawn(move || superfluid_daemon::fleet_join::accept_nodes(listener, m, a, name));
            match superfluid_daemon::fleet_mdns::advertise(port, &model_name) {
                Ok(d) => Some(d),
                Err(e) => {
                    eprintln!("superfluid: not announced on the local network ({e}); give nodes this head's address");
                    None
                }
            }
        });
        if !qos_flags.is_empty() {
            eprintln!(
                "superfluid: {} not supported in fleet-head mode (the fleet head has no \
                 local scheduler and Link F carries no QoS class); set them on each \
                 node's own `superfluid serve` instead",
                qos_flags.join(", ")
            );
            std::process::exit(2);
        }
        if op_spec_max_temperature.is_some() {
            eprintln!(
                "superfluid: --spec-max-temperature is not supported in fleet-head mode; \
                 set it on each node's own `superfluid serve` instead"
            );
            std::process::exit(2);
        }
        if fim_model.is_some() {
            eprintln!(
                "superfluid: --fim-model is not supported in fleet-head mode (the head hosts no \
                 engine); set it on a node's own `superfluid serve` instead"
            );
            std::process::exit(2);
        }
        if op_repeat_penalty.is_some() {
            eprintln!(
                "superfluid: --repeat-penalty is not supported in fleet-head mode \
                 (the fleet protocol carries no per-request penalty); \
                 set it on each node's own `superfluid serve` instead"
            );
            std::process::exit(2);
        }
        let http_listener = listen_or_exit("--http", &addr);
        eprintln!(
            "superfluid: fleet-head openai api on http://{addr} over {n_nodes} node(s), {reached} reachable"
        );
        if key_policy.is_some() {
            eprintln!(
                "superfluid: --key-policy at the fleet head: keys, rate and concurrency limits \
                 apply; class and max_class do not (the head has no local scheduler)"
            );
        }
        let codec_arc: Arc<dyn superfluid_daemon::TextCodec + Send + Sync> = Arc::from(codec);
        superfluid_daemon::fleet_openai::serve_blocking(
            http_listener,
            manager,
            codec_arc,
            "raw".to_string(),
            model_name.clone(),
            max_context.map_or(0, |n| n.max(512) as u32),
            superfluid_daemon::openai::ServeConfig {
                api_key: api_key.clone(),
                rate_limit_per_minute: rate_limit,
                keys: key_policy.clone(),
                sampling: superfluid_daemon::openai::SamplingOverrides {
                    temperature: op_temperature,
                    top_p: op_top_p,
                    top_k: op_top_k,
                    min_p: op_min_p,
                    repeat_penalty: op_repeat_penalty,
                    spec_max_temperature: None,
                    fallback: embedding.sampling_fallback,
                },
                ..Default::default()
            },
        )
        .expect("fleet serve");
        return;
    }

    let wal_path = sessions.join("wal.log");
    let store = match SessionStore::open(&wal_path) {
        Ok(store) => store,
        Err(e @ (superfluid_daemon::DaemonError::WalCorrupt | superfluid_daemon::DaemonError::Codec(_))) => {
            eprintln!(
                "superfluid: the session log {} is damaged ({e}). A record with more of the log \
                 after it fails its checksum, or a whole record does not decode, so this was not a \
                 crash mid-write (a torn last record is dropped on its own); this server won't guess \
                 which history to keep. Move the file aside to start fresh, or run with \
                 --sessions-cache if sessions don't need to survive a restart.",
                wal_path.display()
            );
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("superfluid: could not open the session log {}: {e}", wal_path.display());
            std::process::exit(1);
        }
    };

    let workerd_bin = catalog.workerd().to_path_buf();
    let worker_process = if !runtime.in_process_capable() {
        if !worker_process {
            eprintln!(
                "superfluid: --no-worker-process: the {} runtime runs in its own worker ({})",
                runtime.id.name(),
                runtime.worker.bin.display()
            );
        }
        true
    } else if worker_process && !runtime.worker.bin.is_file() {
        if fim_model.is_some() {
            eprintln!(
                "superfluid: --fim-model needs superfluid-workerd (the satellite runs in its own \
                 worker), not found at {}",
                workerd_bin.display()
            );
            std::process::exit(1);
        }
        if models.len() > 1 {
            eprintln!(
                "superfluid: {} models need superfluid-workerd (one address space each), \
                 not found at {}",
                models.len(),
                workerd_bin.display()
            );
            std::process::exit(1);
        }
        eprintln!(
            "superfluid: superfluid-workerd not found at {}; running the engine in-process \
             (install the bundled workerd or pass --no-worker-process to silence)",
            workerd_bin.display()
        );
        false
    } else {
        worker_process
    };
    let model_display = model.display().to_string();
    let model_source = model.to_string_lossy().into_owned();
    let host = if worker_process {
        match early_host {
            Some(started) => started.join().expect("worker start thread"),
            None => EngineHost::spawn_process(superfluid_daemon::WorkerSpec {
                bin: runtime.worker.bin.clone(),
                args: runtime.serve_args(&model, ctx, max_batch, kv_bits),
                stderr: worker_stderr.clone(),
            }),
        }
        .unwrap_or_else(|e| {
            #[cfg(feature = "tui")]
            if let Some(c) = stderr_capture.as_ref() {
                c.drain(std::time::Duration::from_secs(2));
            }
            eprintln!(
                "superfluid: could not start the engine worker for '{model_display}': {e} \
                 (the worker's stderr above carries the engine's reason)"
            );
            std::process::exit(1);
        })
    } else {
        superfluid_daemon::linked::spawn(runtime.id, model, ctx, max_batch, kv_bits)
            .unwrap_or_else(|| Err(format!("this build does not link the {} runtime", runtime.id)))
            .unwrap_or_else(|e| {
                eprintln!("superfluid: could not load '{model_display}': {e}");
                std::process::exit(1);
            })
    };

    let caps = host.capabilities().read().expect("capability record").clone();
    if let Err(e) = caps.check_startup_flags(park_lossy, kv_bits_arg, speculate.as_deref()) {
        eprintln!("superfluid: {e}");
        std::process::exit(2);
    }
    if prefix_blob_budget_mb > 0 {
        if let superfluid_daemon::capabilities::Cap::No(why) = caps.get("ops", "space_promote") {
            eprintln!("superfluid: --prefix-blob-budget-mb has no effect on runtime {}: {why}", runtime.id.name());
        }
    }

    let listener = socket.as_ref().map(|socket| {
        let _ = std::fs::remove_file(socket);
        let listener = UnixListener::bind(socket).unwrap_or_else(|e| {
            eprintln!("superfluid: cannot listen on --socket {}: {e}", socket.display());
            std::process::exit(1);
        });
        eprintln!("superfluid: serving on {}", socket.display());
        listener
    });
    let park_dir = park.then(|| sessions.join("park"));

    let codec: Box<dyn superfluid_daemon::TextCodec + Send + Sync> = match tool_envelope {
        Some(env) => {
            if let Some(why) = superfluid_daemon::codec::tool_envelope_override_conflict(&*codec, env) {
                eprintln!("superfluid: {why}");
                std::process::exit(2);
            }
            eprintln!("superfluid: tool-call envelope overridden to '{}'", env.name());
            Box::new(superfluid_daemon::codec::WithToolEnvelope::new(codec, env))
        }
        None => codec,
    };
    let daemon = Arc::new(
        Daemon::with_options(
            store,
            host,
            codec,
            superfluid_daemon::DaemonOptions {
                max_lanes: max_batch as usize,
                park_dir,
                park_lossy,
                park_budget_bytes: park_budget_gb.saturating_mul(1 << 30),
                tool_lease_ms,
                tick_decode_budget,
                tick_target_ms,
                runtime_id: Some(runtime.id.name()),
                prefill_budget,
                agent_starvation_ticks: starvation_ticks,
                class_lanes,
                pressure_high_pct: if pressure_high > 0 { pressure_high } else { 85 },
                pressure_low_pct: if pressure_low > 0 { pressure_low } else { 70 },
                pin_budget_pct,
                pressure_source: if os_pressure {
                    superfluid_daemon::pressure::PressureConfig::Os
                } else {
                    superfluid_daemon::pressure::PressureConfig::None
                },
                speculate: speculate_primary.clone(),
                speculate_auto: speculate.as_deref().map(str::trim) == Some("auto"),
                speculate_fallbacks: speculate_fallbacks.clone(),
                speculation: speculation.clone(),
                media_dir: Some(sessions.join("media")),
                files_max_bytes,
                files_expiry_secs: files_expiry,
                completion_deadline_ms,
                trace_scope: superfluid_daemon::otlp::TraceScope::model(&model_name),
                completion_bucket,
            },
        )
        .unwrap_or_else(|e| {
            #[cfg(feature = "tui")]
            if let Some(c) = stderr_capture.as_ref() {
                c.drain(std::time::Duration::from_secs(2));
            }
            eprintln!("superfluid: cannot serve '{model_display}': {e}");
            std::process::exit(2);
        }),
    );
    {
        let bits = daemon.sched_stats().kv_bits.load(std::sync::atomic::Ordering::Relaxed);
        match superfluid_daemon::scheduler::kv_bits_label(bits) {
            Some(label) if kv_bits != 0 && bits != kv_bits as u64 => {
                eprintln!("superfluid: kv cache {label} (--kv-bits {kv_bits} overridden by the engine, see above)")
            }
            Some(label) => eprintln!("superfluid: kv cache {label}"),
            None => {}
        }
    }
    let fim_satellite: Option<(String, Arc<Daemon>)> = fim_model.as_ref().map(|path| {
        let runtime = catalog.reader_of(path).and_then(|id| catalog.require(id)).unwrap_or_else(|e| {
            eprintln!("superfluid: --fim-model {}: {e}", path.display());
            std::process::exit(2);
        });
        let mut fim_ctx = fim_max_context.unwrap_or(ctx.min(8192)).max(512);
        if max_context_auto && fim_ctx > ctx {
            eprintln!(
                "superfluid: --fim-max-context {fim_ctx} is above the window sized for this device; \
                 the FIM satellite runs at {ctx}"
            );
            fim_ctx = ctx;
        }
        let sat = build_fim_satellite(&FimSatelliteCfg {
            model: path.clone(),
            runtime,
            sessions: sessions.clone(),
            ctx: fim_ctx,
            max_batch: fim_max_batch,
            kv_bits,
            pressure_high,
            pressure_low,
            os_pressure,
            completion_deadline_ms,
            worker_stderr: worker_stderr.clone(),
        })
        .unwrap_or_else(|e| {
            #[cfg(feature = "tui")]
            if let Some(c) = stderr_capture.as_ref() {
                c.drain(std::time::Duration::from_secs(2));
            }
            eprintln!("superfluid: could not start the --fim-model satellite '{}': {e}", path.display());
            std::process::exit(1);
        });
        let name = superfluid_daemon::registry::model_id_for(path);
        eprintln!("superfluid: FIM completions served by satellite '{name}'");
        (name, sat)
    });
    if let Some((name, sat)) = &fim_satellite {
        if let Err(e) = daemon.attach_fim_satellite(name.clone(), Arc::clone(sat)) {
            eprintln!(
                "superfluid: --fim-model {}: {e} (its tokenizer has no atomic <|fim_prefix|>/<|fim_suffix|>/<|fim_middle|> tokens)",
                fim_model.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
            );
            std::process::exit(2);
        }
    }
    let state_names: StateNames = Arc::default();
    let registry = if worker_process {
        let cfg = ModelBuildCfg {
            model_name: embedding.model_name,
            base_sessions: sessions.clone(),
            state_names: Arc::clone(&state_names),
            catalog: Arc::clone(&catalog),
            pick: runtime_pick.clone(),
            dialect: dialect.clone(),
            ctx,
            max_batch,
            kv_bits,
            park,
            park_lossy,
            park_budget_gb,
            tool_lease_ms,
            tick_decode_budget,
            tick_target_ms,
            prefill_budget,
            starvation_ticks,
            class_lanes,
            os_pressure,
            pressure_high_pct: if pressure_high > 0 { pressure_high } else { 85 },
            pressure_low_pct: if pressure_low > 0 { pressure_low } else { 70 },
            pin_budget_pct,
            speculate: speculate.clone(),
            speculation: speculation.clone(),
            completion_deadline_ms,
            completion_bucket,
            fim_satellite: fim_satellite.clone(),
            worker_stderr: worker_stderr.clone(),
        };
        let reload_cfg = cfg.clone();
        let loader: superfluid_daemon::registry::ModelLoader =
            Box::new(move |id: &str, runtime| build_loaded_daemon(&cfg, std::path::Path::new(id), runtime));
        let registry = superfluid_daemon::registry::ModelRegistry::with_initial(model_name.clone(), Arc::clone(&daemon), loader);
        registry.remember_source_on(&model_name, &model_source, Some(runtime.id));
        let names = Arc::clone(&catalog);
        registry.set_runtime_names(Box::new(move |name: &str| {
            names.named(name).map_err(|why| format!("{why} ({})", names.here_described()))
        }));
        registry.set_reloader(Box::new(move |id: &str, runtime, retired: &Daemon| {
            reload_daemon(&reload_cfg, std::path::Path::new(id), runtime, retired)
        }));
        Arc::new(registry)
    } else {
        Arc::new(superfluid_daemon::registry::ModelRegistry::single(
            model_name.clone(),
            Arc::clone(&daemon),
        ))
    };
    {
        let (registry, name) = (Arc::downgrade(&registry), model_name.clone());
        daemon.on_retired(Box::new(move || {
            if let Some(registry) = registry.upgrade() {
                let _ = registry.resolve(Some(&name));
            }
        }));
    }
    let same = |p: &std::path::Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let mut taken: std::collections::HashSet<String> = std::collections::HashSet::from([model_name.clone()]);
    let mut given: std::collections::HashSet<PathBuf> = models.first().map(|m| same(m)).into_iter().collect();
    for extra in models.iter().skip(1) {
        if !given.insert(same(extra)) {
            eprintln!("superfluid: model '{}' already loaded, skipping", (embedding.model_name)(extra, &Default::default()));
            continue;
        }
        let name = match pulled_names.get(extra) {
            Some(name) => name.clone(),
            None => {
                let name = (embedding.model_name)(extra, &taken);
                // The worker names the model again with nothing taken, so a
                // name the collision chose is recorded for it to find.
                if name != (embedding.model_name)(extra, &Default::default()) {
                    eprintln!("superfluid: {} is registered as '{name}' (another model took its stem)", extra.display());
                    state_names.lock().expect("state names").insert(extra.clone(), name.clone());
                }
                name
            }
        };
        taken.insert(name.clone());
        match registry.load_as(&name, &extra.to_string_lossy()) {
            Ok(true) => eprintln!("superfluid: loaded model '{name}'"),
            Ok(false) => eprintln!("superfluid: model '{name}' already loaded, skipping"),
            Err(e) => {
                #[cfg(feature = "tui")]
                if let Some(c) = stderr_capture.as_ref() {
                    c.drain(std::time::Duration::from_secs(2));
                }
                eprintln!("superfluid: could not load '{}': {e}", extra.display());
                std::process::exit(1);
            }
        }
    }
    if let Some(dir) = &model_dir {
        let mut n = 0;
        let here = catalog.survey();
        match artifacts_under(dir, &here, MODEL_DIR_DEPTH) {
            Ok(paths) => {
                for path in paths {
                    let refused = runtime_pick
                        .resolve(&path, &catalog)
                        .and_then(|rt| catalog.require(rt))
                        .and_then(|rt| refuse_statically(&rt, park_lossy, kv_bits_arg, speculate.as_deref()));
                    if let Err(e) = refused {
                        eprintln!("superfluid: --model-dir: skipping {}: {e}", path.display());
                        continue;
                    }
                    if !given.insert(same(&path)) {
                        continue;
                    }
                    let name = (embedding.model_name)(&path, &taken);
                    if registry.is_loaded(&name) {
                        taken.insert(name);
                        continue;
                    }
                    if name != (embedding.model_name)(&path, &Default::default()) {
                        eprintln!(
                            "superfluid: --model-dir: {} is registered as '{name}' (another artifact took its stem)",
                            path.display()
                        );
                        state_names.lock().expect("state names").insert(path.clone(), name.clone());
                    }
                    registry.register_known(&name, &path);
                    taken.insert(name);
                    n += 1;
                }
            }
            Err(e) => {
                eprintln!("superfluid: --model-dir {}: {e}", dir.display());
                std::process::exit(1);
            }
        }
        eprintln!("superfluid: {n} model(s) known from {} (loaded on first use)", dir.display());
    }
    if idle_timeout > 0 {
        registry.track_idle();
        let reg = Arc::clone(&registry);
        let max_idle = std::time::Duration::from_secs(idle_timeout);
        let tick = std::time::Duration::from_secs((idle_timeout / 4).max(1));
        std::thread::spawn(move || loop {
            std::thread::sleep(tick);
            for name in reg.sweep_idle(max_idle) {
                eprintln!("superfluid: unloaded idle model '{name}'");
            }
        });
    }
    if files_expiry.is_some() && files_sweep > 0 {
        let d = Arc::clone(&daemon);
        std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(files_sweep));
            let n = d.files().sweep();
            if n > 0 {
                eprintln!("superfluid: files sweep removed {n} expired file(s)");
            }
        });
    }
    let mut http_thread = None;
    if let Some(addr) = http {
        let http_listener = listen_or_exit("--http", &addr);
        eprintln!("superfluid: openai api on http://{addr}{}", embedding.http.also_served());
        if nonstream_keepalive > 0 {
            eprintln!(
                "superfluid: non-streaming keep-alive on: 200 at admission, then a space every \
                 {nonstream_keepalive}s until the JSON body (x-superfluid-keepalive: whitespace)"
            );
        }
        let reg = Arc::clone(&registry);
        let serve_cfg = superfluid_daemon::openai::ServeConfig {
            api_key: api_key.clone(),
            rate_limit_per_minute: rate_limit,
            drain_timeout_secs: drain_timeout,
            default_max_tokens,
            max_context: ctx as u32,
            sampling: superfluid_daemon::openai::SamplingOverrides {
                temperature: op_temperature,
                top_p: op_top_p,
                top_k: op_top_k,
                min_p: op_min_p,
                repeat_penalty: op_repeat_penalty,
                spec_max_temperature: op_spec_max_temperature,
                fallback: embedding.sampling_fallback,
            },
            qos: superfluid_daemon::openai::HttpQosPolicy {
                default: http_default_qos,
                honor_headers: http_qos_header,
                allow_batch_invariant: http_allow_batch_invariant,
            },
            keys: key_policy.clone(),
            nonstream_keepalive_secs: nonstream_keepalive,
            surface: embedding.http,
            model_naming: embedding.model_naming,
            identity: embedding.identity.clone(),
        };
        http_thread = Some(std::thread::spawn(move || {
            superfluid_daemon::openai::serve_blocking_config(http_listener, reg, serve_cfg)
        }));
    }
    if let Some(addr) = web {
        if !addr.starts_with("127.0.0.1:") && !addr.starts_with("[::1]:") && !addr.starts_with("localhost:") {
            eprintln!("superfluid: --web must bind loopback (127.0.0.1/::1)");
            std::process::exit(2);
        }
        let token_path = sessions.join("web-token");
        let token = superfluid_daemon::apiweb::mint_token(&token_path).expect("mint web token");
        let web_listener = listen_or_exit("--web", &addr);
        eprintln!("superfluid: api-web on http://{addr} (token: {})", token_path.display());
        let d = Arc::clone(&daemon);
        std::thread::spawn(move || {
            let _ = superfluid_daemon::apiweb::serve_blocking(web_listener, d, token, web_origins);
        });
    }
    let _otlp_pusher = otlp_metrics.map(|url| {
        eprintln!(
            "superfluid: pushing metrics to {url}/v1/metrics every {otlp_interval_ms}ms (an EGRESS)"
        );
        superfluid_daemon::export::MetricsPusher::start(
            url,
            std::time::Duration::from_millis(otlp_interval_ms),
            daemon.sched_stats(),
        )
    });
    #[cfg(feature = "tui")]
    if let Some(ring) = tui_ring {
        if let Some(listener) = listener {
            let serve_daemon = Arc::clone(&daemon);
            std::thread::spawn(move || {
                let _ = api::serve(listener, serve_daemon);
            });
        }
        let node = superfluid_daemon::tui::NodeInfo {
            identity: hostname(),
            model: model_name.clone(),
            engine: if worker_process { "native (worker)".into() } else { "native".into() },
            max_lanes: max_batch,
        };
        superfluid_daemon::tui::run(daemon, ring, node, stderr_capture.as_ref()).expect("tui");
        drop(stderr_capture);
        return;
    }
    let Some(listener) = listener else {
        // No session socket: the HTTP API is the server, and the process
        // lasts as long as it does.
        match http_thread.map(std::thread::JoinHandle::join) {
            Some(Ok(Ok(()))) => return,
            Some(Ok(Err(e))) => eprintln!("superfluid: the HTTP API stopped: {e}"),
            Some(Err(_)) => eprintln!("superfluid: the HTTP API stopped: its thread panicked"),
            None => unreachable!("without a session socket --no-http is refused"),
        }
        std::process::exit(1);
    };
    api::serve(listener, daemon).expect("serve");
}

#[cfg(feature = "tui")]
fn hostname() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: `buf` is a valid 256-byte writable buffer and `buf.len()` is its
    // exact capacity; gethostname writes at most that many bytes and NUL-
    // terminates within it. We only read `buf` after checking rc == 0.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast::<libc::c_char>(), buf.len()) };
    if rc == 0 {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    } else {
        "local".into()
    }
}

type StateNames = Arc<std::sync::Mutex<std::collections::HashMap<PathBuf, String>>>;

#[derive(Clone)]
struct ModelBuildCfg {
    model_name: fn(&Path, &std::collections::HashSet<String>) -> String,
    base_sessions: std::path::PathBuf,
    state_names: StateNames,
    catalog: Arc<superfluid_daemon::runtimes::Catalog>,
    pick: RuntimePick,
    dialect: String,
    ctx: i32,
    max_batch: u32,
    kv_bits: i32,
    park: bool,
    park_lossy: bool,
    park_budget_gb: u64,
    tool_lease_ms: u64,
    tick_decode_budget: u32,
    tick_target_ms: u32,
    prefill_budget: u32,
    starvation_ticks: u32,
    class_lanes: [usize; 4],
    os_pressure: bool,
    pressure_high_pct: u8,
    pressure_low_pct: u8,
    pin_budget_pct: u8,
    speculate: Option<String>,
    speculation: superfluid_daemon::SpeculationOptions,
    completion_deadline_ms: u64,
    completion_bucket: superfluid_daemon::completion_bucket::BucketConfig,
    fim_satellite: Option<(String, Arc<Daemon>)>,
    worker_stderr: Option<std::sync::Arc<std::os::fd::OwnedFd>>,
}

struct FimSatelliteCfg {
    model: PathBuf,
    runtime: superfluid_daemon::runtimes::Runtime,
    sessions: PathBuf,
    ctx: i32,
    max_batch: u32,
    kv_bits: i32,
    pressure_high: u8,
    pressure_low: u8,
    os_pressure: bool,
    completion_deadline_ms: u64,
    worker_stderr: Option<std::sync::Arc<std::os::fd::OwnedFd>>,
}

fn build_fim_satellite(cfg: &FimSatelliteCfg) -> Result<Arc<Daemon>, superfluid_daemon::DaemonError> {
    use superfluid_daemon::DaemonError;
    let tok = open_tokenizer(&cfg.runtime, &cfg.model)
        .map_err(|e| DaemonError::Generation(format!("satellite tokenizer load failed: {e}")))?;
    let codec = superfluid_daemon::BundleCodec::from_tokenizer(tok);
    let host = EngineHost::spawn_process(superfluid_daemon::WorkerSpec {
        bin: cfg.runtime.worker.bin.clone(),
        args: cfg.runtime.serve_args(&cfg.model, cfg.ctx, cfg.max_batch, cfg.kv_bits),
        stderr: cfg.worker_stderr.clone(),
    })?;
    let daemon = Daemon::with_options(
        SessionStore::ephemeral(),
        host,
        Box::new(codec),
        superfluid_daemon::DaemonOptions {
            max_lanes: cfg.max_batch as usize,
            park_dir: None,
            pressure_high_pct: if cfg.pressure_high > 0 { cfg.pressure_high } else { 85 },
            pressure_low_pct: if cfg.pressure_low > 0 { cfg.pressure_low } else { 70 },
            pressure_source: if cfg.os_pressure {
                superfluid_daemon::pressure::PressureConfig::Os
            } else {
                superfluid_daemon::pressure::PressureConfig::None
            },
            media_dir: Some(cfg.sessions.join("fim").join("media")),
            completion_deadline_ms: cfg.completion_deadline_ms,
            trace_scope: superfluid_daemon::otlp::TraceScope::model(
                &superfluid_daemon::registry::model_id_for(&cfg.model),
            ),
            ..Default::default()
        },
    )?;
    Ok(Arc::new(daemon))
}

fn build_codec_for(
    dialect: &str,
    runtime: &superfluid_daemon::runtimes::Runtime,
    model: &std::path::Path,
) -> Result<Box<dyn superfluid_daemon::TextCodec + Send + Sync>, superfluid_daemon::DaemonError> {
    use superfluid_daemon::DaemonError;
    let tok = open_tokenizer(runtime, model).map_err(DaemonError::Generation)?;
    let (codec, _) = codec_for_dialect(dialect, tok).map_err(DaemonError::Generation)?;
    Ok(codec)
}

fn refuse_statically(
    runtime: &superfluid_daemon::runtimes::Runtime,
    park_lossy: bool,
    kv_bits: Option<i32>,
    speculate: Option<&str>,
) -> Result<(), String> {
    runtime.static_capabilities().check_startup_flags(park_lossy, kv_bits, speculate)
}

fn describe_worker(runtime: &superfluid_daemon::runtimes::Runtime) -> String {
    let version = runtime.version.as_deref().map(|v| format!("{v}, ")).unwrap_or_default();
    format!("{version}{}: {}", runtime.worker.source.name(), runtime.worker.bin.display())
}

fn open_tokenizer(
    runtime: &superfluid_daemon::runtimes::Runtime,
    model: &std::path::Path,
) -> Result<Arc<dyn superfluid_engine::Tokenizer>, String> {
    use superfluid_engine::artifact::TokenizerSource;
    if let Some(why) = not_there(model) {
        return Err(why);
    }
    if let Some(lib) = &runtime.tokenizer {
        let tok = superfluid_daemon::dylib_tokenizer::DylibTokenizer::open(lib, model)?;
        return Ok(Arc::new(tok));
    }
    if runtime.info.tokenizer == Some(TokenizerSource::HuggingFace) {
        let tok = superfluid_tokenizer_hf::HfTokenizer::load(model)
            .map_err(|e| format!("tokenizer load failed for {}: {e}", model.display()))?;
        return Ok(Arc::new(tok));
    }
    superfluid_daemon::linked::tokenizer(runtime.id, model).unwrap_or_else(|| {
        Err(format!(
            "no tokenizer for {} here: the {} runtime ships none beside its worker, and this superfluid does not link it",
            model.display(),
            runtime.id
        ))
    })
}

/// How deep `--model-dir` looks: a hub cache nests a model three levels
/// down (`<org>/<repo>/<variant>/model.base`).
const MODEL_DIR_DEPTH: usize = 8;

/// The artifacts under `dir` that a runtime here reads, in name order. An
/// entry that one reads is taken whole (an MLX model is a directory); a
/// directory that none reads is looked into.
fn artifacts_under(
    dir: &std::path::Path,
    here: &[superfluid_daemon::runtimes::Found],
    depth: usize,
) -> std::io::Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?.flatten().map(|e| e.path()).collect();
    paths.sort();
    let mut found = Vec::new();
    for path in paths {
        if !superfluid_daemon::runtimes::readers_in(here, &path).is_empty() {
            found.push(path);
        } else if depth > 0 && path.is_dir() {
            found.extend(artifacts_under(&path, here, depth - 1).unwrap_or_default());
        }
    }
    Ok(found)
}

fn not_there(model: &std::path::Path) -> Option<String> {
    (!model.exists()).then(|| format!("{}: no such file or directory", model.display()))
}

fn codec_for_dialect(
    dialect: &str,
    tok: Arc<dyn superfluid_engine::Tokenizer>,
) -> Result<(Box<dyn superfluid_daemon::TextCodec + Send + Sync>, &'static str), String> {
    Ok(match dialect {
        "chatml" => (
            Box::new(
                superfluid_daemon::codec::chatml_codec_from_tokenizer(tok)
                    .map_err(|e| format!("--dialect chatml refused: {e}"))?,
            ),
            "chatml",
        ),
        "atem" => (
            Box::new(
                superfluid_daemon::atem::AtemCodec::from_tokenizer(tok)
                    .map_err(|_| "--dialect atem refused: the tokenizer lacks the ATEM markers".to_string())?,
            ),
            "atem",
        ),
        "raw" => (Box::new(superfluid_daemon::BundleCodec::from_tokenizer(tok)), "raw"),
        "template" => (
            Box::new(
                superfluid_daemon::template_codec::TemplateCodec::from_tokenizer(tok)
                    .map_err(|_| "--dialect template refused: the model carries no usable chat template".to_string())?,
            ),
            "template",
        ),
        "auto" => superfluid_daemon::codec::auto_codec_from_tokenizer(tok),
        other => return Err(format!("unknown dialect '{other}' (chatml|atem|raw|template|auto)")),
    })
}

fn speculation_for<'a>(
    directive: Option<&'a str>,
    caps: &superfluid_daemon::capabilities::Capabilities,
    model: &std::path::Path,
    runtime: &str,
) -> Option<&'a str> {
    let d = directive?;
    if d.trim() != "auto" {
        return Some(d);
    }
    match caps.declines_auto_speculation() {
        Some(why) => {
            eprintln!("superfluid: speculation: auto: {} on the {runtime} runtime verifies no drafts ({why}) — decoding plainly", model.display());
            None
        }
        // The other runtimes draft from the prompt alone: there is no
        // drafter artifact for `auto` to look for.
        None if runtime != "basert" => {
            eprintln!("superfluid: speculation: auto: prompt-lookup on the {runtime} runtime");
            Some("prompt-lookup")
        }
        None => Some(d),
    }
}

fn unsupported_runtime(param: Option<&'static str>) -> impl Fn(String) -> superfluid_daemon::DaemonError {
    move |m: String| superfluid_daemon::DaemonError::Unsupported(superfluid_daemon::Refusal::new("unsupported_runtime", param, m))
}

fn runtime_for(
    cfg: &ModelBuildCfg,
    model_path: &std::path::Path,
    requested: Option<RuntimeId>,
) -> Result<superfluid_daemon::runtimes::Runtime, superfluid_daemon::DaemonError> {
    let runtime = cfg
        .pick
        .resolve_with(model_path, requested, &cfg.catalog)
        .and_then(|rt| cfg.catalog.require(rt))
        .map_err(unsupported_runtime(Some(if requested.is_some() { "runtime" } else { "model" })))?;
    refuse_statically(&runtime, cfg.park_lossy, Some(cfg.kv_bits), cfg.speculate.as_deref()).map_err(unsupported_runtime(None))?;
    Ok(runtime)
}

fn engine_on(
    cfg: &ModelBuildCfg,
    runtime: &superfluid_daemon::runtimes::Runtime,
    model_path: &std::path::Path,
) -> Result<(Box<dyn superfluid_daemon::TextCodec + Send + Sync>, EngineHost), superfluid_daemon::DaemonError> {
    use superfluid_daemon::DaemonError;
    let codec = build_codec_for(&cfg.dialect, runtime, model_path)?;
    let host = EngineHost::spawn_process(superfluid_daemon::WorkerSpec {
        bin: runtime.worker.bin.clone(),
        args: runtime.serve_args(model_path, cfg.ctx.max(512), cfg.max_batch, cfg.kv_bits),
        stderr: cfg.worker_stderr.clone(),
    })
    .map_err(|_| DaemonError::Generation("engine worker spawn failed".to_string()))?;
    host.capabilities()
        .read()
        .expect("capability record")
        .check_startup_flags(cfg.park_lossy, Some(cfg.kv_bits), cfg.speculate.as_deref())
        .map_err(unsupported_runtime(None))?;
    Ok((codec, host))
}

fn reload_daemon(
    cfg: &ModelBuildCfg,
    model_path: &std::path::Path,
    requested: Option<RuntimeId>,
    retired: &Daemon,
) -> Result<(), superfluid_daemon::DaemonError> {
    retired.reload_with(|| {
        let runtime = runtime_for(cfg, model_path, requested)?;
        let (codec, host) = engine_on(cfg, &runtime, model_path)?;
        Ok((host, codec))
    })
}

fn build_loaded_daemon(
    cfg: &ModelBuildCfg,
    model_path: &std::path::Path,
    requested: Option<RuntimeId>,
) -> Result<Arc<Daemon>, superfluid_daemon::DaemonError> {
    use superfluid_daemon::DaemonError;
    let id: String = cfg
        .state_names
        .lock()
        .expect("state names")
        .get(model_path)
        .cloned()
        .unwrap_or_else(|| (cfg.model_name)(model_path, &Default::default()));
    let mdir = superfluid_daemon::registry::state_dir(&cfg.base_sessions.join("models"), &id)?;
    let runtime = runtime_for(cfg, model_path, requested)?;
    settle_park_dir(&mdir.join("park"), cfg.park, cfg.park_budget_gb);
    let store = SessionStore::open(&mdir.join("wal.log"))?;
    let (codec, host) = engine_on(cfg, &runtime, model_path)?;
    let park_dir = cfg.park.then(|| mdir.join("park"));
    let caps = host.capabilities().read().expect("capability record").clone();
    let mut speculate_ranked = match speculation_for(cfg.speculate.as_deref(), &caps, model_path, runtime.id.name()) {
        Some(d) => superfluid_daemon::speculate_resolve::resolve_all(d, model_path).map_err(DaemonError::Generation)?,
        None => Vec::new(),
    };
    let speculate_first = (!speculate_ranked.is_empty()).then(|| speculate_ranked.remove(0));
    let daemon = Daemon::with_options(
        store,
        host,
        codec,
        superfluid_daemon::DaemonOptions {
            max_lanes: cfg.max_batch as usize,
            park_dir,
            park_lossy: cfg.park_lossy,
            park_budget_bytes: cfg.park_budget_gb.saturating_mul(1 << 30),
            tool_lease_ms: cfg.tool_lease_ms,
            tick_decode_budget: cfg.tick_decode_budget,
            tick_target_ms: cfg.tick_target_ms,
            prefill_budget: cfg.prefill_budget,
            agent_starvation_ticks: cfg.starvation_ticks,
            class_lanes: cfg.class_lanes,
            pressure_high_pct: cfg.pressure_high_pct,
            pressure_low_pct: cfg.pressure_low_pct,
            pin_budget_pct: cfg.pin_budget_pct,
            pressure_source: if cfg.os_pressure {
                superfluid_daemon::pressure::PressureConfig::Os
            } else {
                superfluid_daemon::pressure::PressureConfig::None
            },
            speculate: speculate_first,
            speculate_auto: cfg.speculate.as_deref().map(str::trim) == Some("auto"),
            speculate_fallbacks: speculate_ranked,
            speculation: cfg.speculation.clone(),
            media_dir: Some(mdir.join("media")),
            completion_deadline_ms: cfg.completion_deadline_ms,
            trace_scope: superfluid_daemon::otlp::TraceScope::model(&id),
            completion_bucket: cfg.completion_bucket,
            runtime_id: Some(runtime.id.name()),
            ..Default::default()
        },
    )?;
    if let Some((name, sat)) = &cfg.fim_satellite {
        daemon.attach_fim_satellite(name.clone(), Arc::clone(sat))?;
    }
    Ok(Arc::new(daemon))
}

fn lock_sessions_dir(sessions: &std::path::Path) -> std::fs::File {
    use std::os::fd::AsRawFd;
    let path = sessions.join("serve.lock");
    let file = match std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("superfluid: could not open {}: {e}", path.display());
            std::process::exit(1);
        }
    };
    loop {
        // SAFETY: `file` is an open descriptor for the duration of the call.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            return file;
        }
        let err = std::io::Error::last_os_error();
        match err.kind() {
            std::io::ErrorKind::Interrupted => continue,
            std::io::ErrorKind::WouldBlock => {
                eprintln!(
                    "superfluid: another server is already using the sessions directory {}. \
                     Stop it, or give this one its own --sessions directory.",
                    sessions.display()
                );
                std::process::exit(1);
            }
            _ => {
                eprintln!("superfluid: could not lock {}: {err}", path.display());
                std::process::exit(1);
            }
        }
    }
}

fn rotate_session_wal(dir: &std::path::Path) {
    use superfluid_daemon::store_id::store_id_path;
    let wal = dir.join("wal.log");
    let prev = dir.join("wal.log.prev");
    let keep = std::fs::metadata(&wal).is_ok_and(|m| m.len() > 0);
    let aside = |from: &std::path::Path, to: Option<&std::path::Path>| -> bool {
        if !from.exists() {
            return false;
        }
        if to.is_some_and(|to| std::fs::rename(from, to).is_ok()) {
            return true;
        }
        if let Err(e) = std::fs::remove_file(from) {
            eprintln!(
                "superfluid: --sessions-cache could not set aside or remove {} ({e}); \
                 remove it, or run without --sessions-cache.",
                from.display()
            );
            std::process::exit(1);
        }
        false
    };
    if keep {
        let _ = std::fs::remove_file(store_id_path(&prev));
    }
    if keep && aside(&wal, Some(&prev)) {
        aside(&store_id_path(&wal), Some(&store_id_path(&prev)));
    } else {
        aside(&wal, None);
        aside(&store_id_path(&wal), None);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use superfluid_daemon::runtimes::{Found, Info};
    use superfluid_engine::artifact::Format;

    fn reads(id: &'static str, format: Format) -> Found {
        let info = Info { aliases: Vec::new(), formats: vec![format], tokenizer: None, named: None };
        Found { id: RuntimeId::new(id), worker: None, info: Some(info), usable: Err("not started".into()) }
    }

    #[test]
    fn model_dir_looks_into_a_hub_cache_and_takes_a_directory_model_whole() {
        let root = std::env::temp_dir().join(format!("superfluid-model-dir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let file = |rel: &str| {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b"BASE....").unwrap();
            p
        };
        let flat = file("Flat-Q4.base");
        let q4 = file("org/Model/default-q4/model.base");
        file("org/Model/default-q4/hub.json");
        let q8 = file("org/Model/default-q8/model.base");
        file("notes.txt");
        file("mlx-model/config.json");
        file("mlx-model/weights.safetensors");
        file("mlx-model/stray.base");
        let here = [
            reads("basert", Format::file("base", "a .base bundle", &["base"], None)),
            reads("mlx", Format::directory("mlx", "an MLX model", &["config.json"], &["safetensors"])),
        ];
        let found = artifacts_under(&root, &here, MODEL_DIR_DEPTH).unwrap();
        assert_eq!(found, vec![flat, root.join("mlx-model"), q4, q8]);
        assert!(artifacts_under(&root, &here, 0).unwrap().len() == 2, "depth 0 reads only the top level");
    }
}
