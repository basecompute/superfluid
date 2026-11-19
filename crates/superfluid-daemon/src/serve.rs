//! `superfluid serve`: the server, as a library function so that another
//! program can run it.

use crate as superfluid_daemon;

use superfluid_shm::os::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use superfluid_daemon::runtime_pick::{RuntimeId, RuntimePick};
use superfluid_daemon::{api, Daemon, EngineHost, SessionStore};

fn retired(flag: &str, why: &str) {
    eprintln!("superfluid: {flag} is retired — {why}");
}

fn sampling_f32(flag: &str, raw: &str, lo: f32, hi: f32) -> f32 {
    let v: f32 = raw.parse().unwrap_or_else(|_| {
        superfluid_daemon::events::fail(2, format!("{flag} expects a number, got {raw:?}"));
    });
    if !v.is_finite() || v < lo || v > hi {
        superfluid_daemon::events::fail(2, format!("{flag} must be a finite number in [{lo}, {hi}], got {v}"));
    }
    v
}

fn usage_error(msg: &str, command: &str) -> ! {
    if superfluid_daemon::events::json() {
        superfluid_daemon::events::fail(2, msg);
    }
    eprintln!("superfluid: {msg}");
    eprintln!("Run 'superfluid {command} --help' for its options.");
    std::process::exit(2);
}

/// [`superfluid_daemon::events::fail`] for a message that text mode writes without the
/// `superfluid: ` prefix.
fn fail_unprefixed(code: i32, line: String) -> ! {
    if superfluid_daemon::events::json() {
        superfluid_daemon::events::fail(code, line);
    }
    eprintln!("{line}");
    std::process::exit(code);
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

/// The session log under `sessions`, or the reason it cannot be served from, and exit.
fn open_session_store(sessions: &std::path::Path) -> SessionStore {
    let wal_path = sessions.join("wal.log");
    match SessionStore::open(&wal_path) {
        Ok(store) => store,
        Err(e @ (superfluid_daemon::DaemonError::WalCorrupt | superfluid_daemon::DaemonError::Codec(_))) => {
            superfluid_daemon::events::fail(1, format!(
                "the session log {} is damaged ({e}). A record with more of the log \
                 after it fails its checksum, or a whole record does not decode, so this was not a \
                 crash mid-write (a torn last record is dropped on its own); this server won't guess \
                 which history to keep. Move the file aside to start fresh, or run with \
                 --sessions-cache if sessions don't need to survive a restart.",
                wal_path.display()
            ));
        }
        Err(e) => {
            superfluid_daemon::events::fail(1, format!("could not open the session log {}: {e}", wal_path.display()));
        }
    }
}

fn listen_or_exit(flag: &str, addr: &str) -> std::net::TcpListener {
    std::net::TcpListener::bind(addr).unwrap_or_else(|e| {
        let hint = match e.kind() {
            std::io::ErrorKind::AddrInUse => " (another process listens there: stop it or pick another port, e.g. --port 8454)",
            std::io::ErrorKind::AddrNotAvailable => " (no interface on this machine has that address)",
            std::io::ErrorKind::PermissionDenied => " (a port under 1024 needs privileges)",
            _ => "",
        };
        superfluid_daemon::events::fail(1, format!("cannot listen on {flag} {addr}: {e}{hint}"));
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
    superfluid_daemon::events::init("serve", args).unwrap_or_else(|e| usage_error(&e, "serve"));
    let mut models: Vec<PathBuf> = Vec::new();
    let mut pull_flags = superfluid_daemon::pulls::PullFlags::default();
    let mut runtime_pick = RuntimePick::default();
    let mut api_key: Option<String> = None;
    let mut api_key_file: Option<PathBuf> = None;
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
    let mut op_presence_penalty: Option<f32> = None;
    let mut op_spec_max_temperature: Option<f32> = None;
    let mut park = false;
    let mut sessions_cache = false;
    let mut park_lossy = false;
    let mut tool_lease_ms: u64 = 0;
    let mut prefix_blob_budget_mb: u64 = 0;
    let mut worker_process = true;
    let mut os_pressure = true;
    let mut dialect = String::from("auto");
    let mut pressure_high: u8 = 0;
    let mut pressure_low: u8 = 0;
    let mut pin_budget_pct: u8 = superfluid_daemon::DaemonOptions::default().pin_budget_pct;
    let mut starvation_ticks: u32 = 8;
    let mut class_lanes_spec: Option<String> = None;
    let mut http_default_qos = superfluid_daemon::openai::RequestQos::AGENT;
    let mut http_qos_header = true;
    let mut http_allow_batch_invariant = false;
    let mut key_policy_path: Option<PathBuf> = None;
    let mut log_dir: Option<PathBuf> = None;
    let mut log_filter = String::from("info");
    let mut speculate: Option<String> = None;
    // The engine's flags a fleet node takes too, read as one (engine_flags).
    let mut engine = superfluid_daemon::engine_flags::EngineFlags::default();
    let mut completion_deadline_ms: u64 = 0;
    let mut completion_bucket = superfluid_daemon::completion_bucket::BucketConfig::default();
    let mut fim_model: Option<PathBuf> = None;
    let mut fim_max_context: Option<i32> = None;
    let mut fim_max_batch: u32 = 2;
    let mut web: Option<String> = None;
    let mut web_origins: Vec<String> = Vec::new();
    let mut otlp_metrics: Option<String> = None;
    let mut otlp_interval_ms: u64 = 15_000;
    let mut tui_flag = false;
    let mut no_tui = false;
    let mut fleet: Option<String> = None;
    let mut fleet_auth: Option<PathBuf> = None;
    let mut fleet_listen: Option<String> = None;
    let mut fleet_policy = String::from("load-aware");
    let mut fleet_pool_high: u32 = superfluid_daemon::fleet_manager::DEFAULT_POOL_HIGH_PCT;
    let mut fleet_conns: usize = superfluid_daemon::fleet_manager::DEFAULT_CONNS_PER_NODE;
    // A runtime's option, kept as given and resolved once the catalog says which runtimes are here.
    let mut option_candidates: Vec<(String, Option<String>)> = Vec::new();
    let args = superfluid_daemon::cli::split_equals(args, superfluid_daemon::cli::serve_takes_value);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_else(|| needs_value(a));
        // SERVE-FLAGS-BEGIN: every flag `serve` reads is listed in superfluid_daemon::cli (a test checks).
        match engine.take(a, &mut val) {
            Ok(true) => continue,
            Ok(false) => {}
            Err(superfluid_daemon::engine_flags::FlagError::Usage(e)) => usage_error(&e, "serve"),
            Err(superfluid_daemon::engine_flags::FlagError::Invalid(e)) => superfluid_daemon::events::fail(2, e),
        }
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
                    superfluid_daemon::events::fail(2, format!("--runtime: {e}"));
                }
            }
            "--api-key" => api_key = Some(val()),
            "--api-key-file" => api_key_file = Some(PathBuf::from(val())),
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
                            superfluid_daemon::events::fail(2, format!(
                                "unknown --tool-call-parser '{v}' (have: auto, {})",
                                superfluid_daemon::codec::ToolEnvelope::names().join(", ")
                            ));
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
            "--presence-penalty" => {
                op_presence_penalty = Some(sampling_f32("--presence-penalty", &val(), -2.0, 2.0))
            }
            "--park" => park = true,
            "--no-park" => park = false,
            "--sessions-cache" => sessions_cache = true,
            "--park-lossy" => {
                park = true;
                park_lossy = true;
            }
            "--tool-lease-ms" => tool_lease_ms = num(a, val()),
            "--prefix-blob-budget-mb" => {
                prefix_blob_budget_mb = num(a, val())
            }
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
            // Kept as named, `off` included: no `--speculate` is the default, `auto` on the baseRT
            // runtime (speculate_directive).
            "--speculate" => speculate = Some(val()),
            "--spec-max-temperature" => {
                op_spec_max_temperature = Some(sampling_f32("--spec-max-temperature", &val(), 0.0, 2.0))
            }
            "--log-filter" => log_filter = val(),
            // Read before the rest by events::init, so that a refusal here is an event too.
            "--log-format" => drop(val()),
            "--starvation-ticks" => starvation_ticks = num(a, val()),
            "--class-lanes" => {
                class_lanes_spec = Some(val());
            }
            "--http-default-qos" => {
                let v = val();
                http_default_qos.class = superfluid_daemon::qos::parse(&v).unwrap_or_else(|| {
                    superfluid_daemon::events::fail(2, format!(
                        "--http-default-qos expects one of {}, got {v:?}",
                        superfluid_daemon::qos::NAMES.join(", ")
                    ));
                });
            }
            "--no-http-qos-header" => {
                http_qos_header = false;
            }
            "--http-allow-batch-invariant" => {
                http_allow_batch_invariant = true;
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
            "--prefill-chunk" => {
                let _ = val();
                retired(a, "an engine-level knob; a runtime takes its own, --mlx-prefill-chunk or --llamacpp-ubatch-size");
            }
            "--metallib" | "--gpu-wait-timeout-ms" | "--decode-replay" => {
                let _ = val();
                retired(a, "an engine-level knob superfluid does not expose");
            }
            "--paged-weights" | "--no-paged-weights-retry" | "--no-baked-decode" => {
                retired(a, "an engine-level knob superfluid does not expose")
            }
            other if !other.starts_with('-') => models.push(PathBuf::from(other)),
            _ => match superfluid_daemon::runtime_options::candidate(a, &mut it) {
                Some(given) => option_candidates.push(given),
                None => unknown_serve_flag(a),
            },
        }
        // SERVE-FLAGS-END
    }
    let kv_bits_arg = engine.kv_bits;
    let (park_budget_gb, prefill_budget) = (engine.park_budget_gb, engine.prefill_budget);
    let (tick_decode_budget, tick_target_ms) = (engine.tick_decode_budget, engine.tick_target_ms);
    let speculation = engine.speculation.clone();
    let env_key = std::env::var(superfluid_daemon::keypolicy::API_KEY_ENV).ok().filter(|k| !k.is_empty());
    let key_from_env = api_key.is_none() && api_key_file.is_none() && env_key.is_some();
    let api_key = superfluid_daemon::keypolicy::global_key(api_key, api_key_file.as_deref(), env_key, |w| {
        eprintln!("superfluid: WARNING {w}")
    })
    .unwrap_or_else(|e| usage_error(&e, "serve"));
    if key_from_env {
        eprintln!("superfluid: authentication on, with the key in ${}", superfluid_daemon::keypolicy::API_KEY_ENV);
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
    use superfluid_daemon::runtime_options::Unresolved;
    let runtime_options = superfluid_daemon::runtime_options::resolve_candidates(&option_candidates, &catalog.ids())
        .unwrap_or_else(|e| match e {
            Unresolved::Unknown(flag) => unknown_serve_flag(&flag),
            Unresolved::NoValue(flag) => needs_value(&flag),
        });
    // Set before the runtime is required, so an option that makes it usable here counts for the first probe.
    superfluid_daemon::runtime_options::adopt(&runtime_options).unwrap_or_else(|e| usage_error(&e, "serve"));
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
    let fleet_mode = fleet.is_some() || fleet_listen.is_some();
    let mut pulled_names: std::collections::HashMap<PathBuf, String> = std::collections::HashMap::new();
    for m in models.iter_mut() {
        let token = m.to_string_lossy().into_owned();
        // A fleet head takes a model given by id from a node serving it, weights and all left there.
        if !superfluid_daemon::pulls::looks_like_id(&token) || fleet_mode {
            continue;
        }
        let runtime = match runtime_pick.explicit(m) {
            Some(name) => catalog.named(name).ok().or_else(|| RuntimeId::parse(name)).unwrap_or_else(|| {
                superfluid_daemon::events::fail(2, format!("--runtime '{name}' is not a runtime id"));
            }),
            None => RuntimeId::new(superfluid_daemon::cli::runtime_for_id(&token)),
        };
        // The runtime first, so one missing exits as it would for a model given by path (4).
        if let Err(e) = superfluid_daemon::pulls::ensure_installed(&catalog, runtime, pull_flags.offline) {
            refuse(e.exit_code(), e.to_string());
        }
        let path = superfluid_daemon::pulls::pull(&catalog, runtime, &token, &pull_flags).unwrap_or_else(|e| {
            superfluid_daemon::events::fail(1, e);
        });
        if let Err(e) = runtime_pick.add(&format!("{}={}", path.display(), runtime.name())) {
            superfluid_daemon::events::fail(2, e);
        }
        pulled_names.insert(path.clone(), superfluid_daemon::pulls::served_name(&token));
        *m = path;
    }
    if !pull_flags.options.is_empty() && pulled_names.is_empty() {
        superfluid_daemon::events::fail(2, "--pull-* options apply to models given by id; no --model here is one");
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
        superfluid_daemon::events::fail(2, format!(
            "--socket {} is {} bytes; a unix socket path must be under {SUN_PATH_LEN} bytes here \
             (pick a shorter one)",
            socket.display(),
            socket.as_os_str().len()
        ));
    }
    for (flag, addr) in [("--http", &http), ("--web", &web)] {
        if let Some(addr) = addr {
            drop(listen_or_exit(flag, addr));
        }
    }
    let key_policy = key_policy_path.as_deref().map(|p| {
        let table = superfluid_daemon::keypolicy::KeyTable::load(p, api_key.as_deref())
            .unwrap_or_else(|e| {
                superfluid_daemon::events::fail(2, format!("--key-policy {e}"));
            });
        eprintln!("superfluid: key policy: {} key(s) from {}", table.len(), p.display());
        Arc::new(table)
    });
    if fleet_mode {
        if engine.otlp_endpoint.is_some() {
            superfluid_daemon::events::fail(
                2,
                "--otlp-endpoint is not supported in fleet-head mode (the fleet head \
                 has no scheduler to trace); set it on each node: \
                 `superfluid node join … --otlp-endpoint <url>`",
            );
        }
        {
            use superfluid_daemon::telemetry::{LogConfig, LogSink};
            let sink = match log_dir.clone() {
                Some(dir) => LogSink::File(LogConfig { dir, filter: log_filter.clone(), ..Default::default() }),
                None => LogSink::Stderr(log_filter.clone()),
            };
            if let Err(e) = superfluid_daemon::telemetry::init_sinks(sink, None, None) {
                eprintln!("superfluid: telemetry: {e}");
            }
        }
        std::fs::create_dir_all(&sessions).expect("create sessions dir");
        let _sessions_lock = lock_sessions_dir(&sessions);
        if sessions_cache {
            rotate_session_wal(&sessions);
            if let Ok(rd) = std::fs::read_dir(sessions.join("models")) {
                for dir in rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|d| d.is_dir()) {
                    rotate_session_wal(&dir);
                }
            }
        }
        let fleet_addrs = fleet.clone().unwrap_or_default();
        use superfluid_daemon::fleet_manager::{FleetManager, NodeSpec, PlacementPolicy};
        let auth: Vec<u8> = match &fleet_auth {
            Some(pth) => std::fs::read(pth)
                .map(|t| t.trim_ascii().to_vec())
                .unwrap_or_else(|e| {
                    superfluid_daemon::events::fail(2, format!("cannot read --fleet-auth {}: {e}", pth.display()));
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
            superfluid_daemon::events::fail(2, "--fleet needs at least one host:port, or --fleet-listen for nodes that join");
        }
        let policy = match fleet_policy.as_str() {
            "load-aware" => PlacementPolicy::LoadAware,
            "least-loaded" => PlacementPolicy::LeastLoaded,
            "round-robin" => PlacementPolicy::RoundRobin,
            other => {
                superfluid_daemon::events::fail(2, format!("unknown --fleet-policy '{other}' (load-aware|least-loaded|round-robin)"));
            }
        };
        let Some(addr) = http else {
            superfluid_daemon::events::fail(2, "--fleet requires --http <addr> (the fleet OpenAI endpoint)");
        };
        let lease = superfluid_proto::linkf::Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
        let n_nodes = specs.len();
        let manager = Arc::new(
            FleetManager::with_options(specs, policy, lease, fleet_pool_high, fleet_conns)
                .with_warm_log(sessions.join("fleet-warm.json")),
        );
        // Every model the head serves: one given by path is opened here for its tokenizer and
        // chat template only; one given by id takes them from a node serving it, below.
        let mut served: Vec<(String, Option<Box<dyn superfluid_daemon::TextCodec + Send + Sync>>)> = Vec::new();
        let mut taken: std::collections::HashSet<String> = std::collections::HashSet::new();
        // The id a node pulls a model given by id by, which the head announces to nodes that follow it.
        let mut pulled_by: Vec<(String, String)> = Vec::new();
        for m in &models {
            let token = m.to_string_lossy().into_owned();
            let by_id = !m.exists() && superfluid_daemon::pulls::looks_like_id(&token);
            let name = if by_id { superfluid_daemon::pulls::served_name(&token) } else { (embedding.model_name)(m, &taken) };
            if !taken.insert(name.clone()) {
                eprintln!("superfluid: model '{name}' given twice, skipping");
                continue;
            }
            if by_id {
                pulled_by.push((name.clone(), token));
                served.push((name, None));
                continue;
            }
            let rt = runtime_pick.resolve(m, &catalog).and_then(|id| catalog.require(id)).unwrap_or_else(|e| {
                superfluid_daemon::events::fail(2, format!("{}: {e}", m.display()));
            });
            let tok = open_tokenizer(&rt, m).unwrap_or_else(|e| {
                superfluid_daemon::events::fail(2, e);
            });
            manager.expect_tokenizer(&name, superfluid_daemon::fleet::tokenizer_digest(&*tok));
            let codec = codec_for_dialect(&dialect, tok).map(|(c, _)| c).unwrap_or_else(|e| {
                superfluid_daemon::events::fail(2, format!("{}: {e}", m.display()));
            });
            served.push((name, Some(codec)));
        }
        let names: Vec<String> = served.iter().map(|(n, _)| n.clone()).collect();
        let model_name = names.first().cloned().unwrap_or_else(|| {
            superfluid_daemon::events::fail(2, "no model to serve");
        });
        let fleet_model = Arc::new(FleetModel {
            manager: Arc::clone(&manager),
            sessions: sessions.clone(),
            first: model_name.clone(),
            dialect: dialect.clone(),
            tool_lease_ms,
            files_max_bytes,
            files_expiry,
            completion_deadline_ms,
        });
        // A model a node brings later is served as one given by id is: with the tokenizer and
        // chat template of the node serving it.
        let head_models = {
            let fm = Arc::clone(&fleet_model);
            superfluid_daemon::fleet_models::HeadModels::new(
                Arc::clone(&manager),
                names.clone(),
                Box::new(move |name| {
                    let tok = fm.manager.remote_tokenizer(name)?;
                    let codec = fm.codec(name, tok).map_err(superfluid_daemon::DaemonError::EngineLoad)?;
                    let dir = fm.dir(name).map_err(superfluid_daemon::DaemonError::Io)?;
                    Ok(Arc::new(fm.daemon(name, SessionStore::open(&dir.join("wal.log"))?, codec)?))
                }),
            )
        };
        let (by_id, ids): (Vec<String>, Vec<String>) = pulled_by.into_iter().unzip();
        head_models.know_ids(&by_id, &ids);
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
                    if !n.models.iter().any(|m| names.contains(m)) {
                        eprintln!(
                            "superfluid: fleet node '{}' serves {}, none of {}, so nothing is placed \
                             there; start it with --model-name <one of them>, or have it join this head",
                            n.identity,
                            n.models.join(", "),
                            names.join(", ")
                        );
                    }
                }
                Err(e) => eprintln!("superfluid: {e}; retrying it in the background"),
            }
        }
        if max_context.is_none() {
            for name in &names {
                match manager.advertised_context(name) {
                    Some(n) => eprintln!("superfluid: {name}: context window {n} tokens (the largest node's)"),
                    None => eprintln!("superfluid: {name}: context window unknown until a node serving it answers"),
                }
            }
        }
        manager.spawn_keepalive();
        if let Some(listen) = fleet_listen.as_deref() {
            let listener = std::net::TcpListener::bind(listen).unwrap_or_else(|e| {
                superfluid_daemon::events::fail(1, format!("--fleet-listen {listen}: {e}"));
            });
            let port = listener.local_addr().map(|a| a.port()).unwrap_or_default();
            eprintln!("superfluid: fleet head takes joining nodes on {listen}");
            let (m, a, all) = (Arc::clone(&manager), auth.clone(), Arc::clone(&head_models));
            std::thread::spawn(move || superfluid_daemon::fleet_join::accept_nodes(listener, m, a, all));
            // `superfluid top` finds the head's API by this; one on loopback is no use off this machine.
            let http_port = addr.parse::<std::net::SocketAddr>().ok().filter(|a| !a.ip().is_loopback()).map(|a| a.port());
            // Nodes take only a head of their own network: one whose token they hold.
            let net = (!auth.is_empty()).then(|| superfluid_daemon::fleet_mdns::network_id(&auth));
            match superfluid_daemon::fleet_mdns::advertise(port, &head_models.names(), http_port, net.as_deref()) {
                Ok(announced) => head_models.announce_with(announced),
                Err(e) => eprintln!("superfluid: not announced on the local network ({e}); give nodes this head's address"),
            }
        }
        // The head resolves each request's class as a single server does and sends it to the
        // node with the session; the node's scheduler admits it there, under its own caps.
        if class_lanes_spec.is_some() {
            superfluid_daemon::events::fail(
                2,
                "--class-lanes is not supported in fleet-head mode (lane caps are a scheduler's, \
                 and each node runs its own); set it on each node instead: \
                 `superfluid node join --class-lanes` or `superfluid-noded --class-lanes`",
            );
        }
        if fim_model.is_some() {
            superfluid_daemon::events::fail(
                2,
                "--fim-model is not supported in fleet-head mode: the head hosts no \
                 engine, and it does not send FIM completions to a node; serve them \
                 from one machine's own `superfluid serve --fim-model`",
            );
        }
        let http_listener = listen_or_exit("--http", &addr);
        eprintln!(
            "superfluid: fleet-head openai api on http://{addr} over {n_nodes} node(s), {reached} reachable"
        );
        // A model given by id is served once a node serving it has shared its tokenizer.
        let served: Vec<(String, Box<dyn superfluid_daemon::TextCodec + Send + Sync>)> = served
            .into_iter()
            .map(|(name, codec)| {
                let codec = codec.unwrap_or_else(|| {
                    let mut said = false;
                    let tok = loop {
                        match manager.remote_tokenizer(&name) {
                            Ok(t) => break t,
                            Err(e) => {
                                if !said {
                                    eprintln!("superfluid: {name}: waiting for a node serving it to share its tokenizer ({e})");
                                    said = true;
                                }
                                std::thread::sleep(std::time::Duration::from_secs(2));
                            }
                        }
                    };
                    eprintln!("superfluid: {name}: tokenizer and chat template from a node serving it");
                    fleet_model.codec(&name, tok).unwrap_or_else(|e| {
                        superfluid_daemon::events::fail(2, format!("{name}: {e}"));
                    })
                });
                (name, codec)
            })
            .collect();
        let advertised = manager.advertised_context(&model_name).map(|n| n.min(u32::MAX as u64) as u32);
        let mut registry = None;
        for (name, codec) in served {
            let dir = fleet_model.dir(&name).unwrap_or_else(|e| {
                superfluid_daemon::events::fail(1, format!("{name}: cannot keep its sessions: {e}"));
            });
            let daemon = fleet_model.daemon(&name, open_session_store(&dir), codec).unwrap_or_else(|e| {
                superfluid_daemon::events::fail(2, format!("cannot serve '{name}' over the fleet: {e}"));
            });
            match &registry {
                // A model loaded later is one a node serves.
                None => registry = Some(Arc::new(superfluid_daemon::registry::ModelRegistry::with_initial(
                    name,
                    Arc::new(daemon),
                    head_models.loader(),
                ))),
                Some(r) => r.insert_loaded(&name, Arc::new(daemon)),
            }
        }
        let registry = registry.expect("the first model is served");
        head_models.attach(&registry);
        superfluid_daemon::events::emit("serve.ready", serde_json::json!({"http": http_listener.local_addr().map_or(addr.clone(), |a| a.to_string()), "socket": null, "models": head_models.names()}));
        superfluid_daemon::openai::serve_blocking_config(
            http_listener,
            registry,
            superfluid_daemon::openai::ServeConfig {
                api_key: api_key.clone(),
                rate_limit_per_minute: rate_limit,
                drain_timeout_secs: drain_timeout,
                default_max_tokens,
                max_context: max_context.map(|n| n.max(512) as u32).or(advertised).unwrap_or(8192),
                sampling: superfluid_daemon::openai::SamplingOverrides {
                    temperature: op_temperature,
                    top_p: op_top_p,
                    top_k: op_top_k,
                    min_p: op_min_p,
                    // Resolved here, where the request is, and sent to the node as the request's own,
                    // which is all a node applies: the presence default where the nodes' engine applies
                    // penalties itself and the temperature cap where they speculate, as their
                    // capability record says (SamplingOverrides::for_daemon).
                    repeat_penalty: op_repeat_penalty,
                    presence_penalty: op_presence_penalty,
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
                idempotency_dir: Some(sessions.join("idempotency")),
            },
        )
        .expect("fleet serve");
        return;
    }
    let picked = runtime_pick.explicit(&model).is_some();
    fn refuse(code: i32, e: String) -> ! {
        superfluid_daemon::events::fail(code, e)
    }
    let runtime = match runtime_pick.resolve(&model, &catalog) {
        Err(e) => refuse(2, if picked { e } else { not_there(&model).unwrap_or(e) }),
        Ok(rt) => match catalog.require(rt) {
            Ok(rt) => rt,
            Err(_) => {
                // The runtime answers without its options and refuses one of them: say which. Asked once, since
                // asking starts a worker.
                if let Some(why) = superfluid_daemon::runtime_options::blamed(&catalog, rt, &runtime_options) {
                    refuse(2, why);
                }
                if let Some(why) = not_there(&model) {
                    refuse(2, why);
                }
                // Not here, or cannot run here: exit 4, so a host can tell it from a flag it got wrong;
                // an install tried and failed is 1, a failure along the way (`pulls::Unready`).
                superfluid_daemon::pulls::ensure_installed(&catalog, rt, pull_flags.offline)
                    .unwrap_or_else(|e| refuse(e.exit_code(), e.to_string()))
            }
        },
    };
    if let Err(e) = refuse_statically(&runtime, park_lossy, kv_bits_arg, speculate.as_deref()) {
        refuse(2, e);
    }
    // Each runtime's options are checked against what its worker declares before any model loads.
    superfluid_daemon::runtime_options::settle(&runtime_options, &runtime, &catalog).unwrap_or_else(|e| refuse(2, e));
    // A drafter given by id is pulled as a model given by id is, by the runtime that drafts for it, before
    // anything sizes or loads.
    if let Some((kind, id)) = speculate.as_deref().and_then(superfluid_daemon::speculate_resolve::drafter_id) {
        let path = superfluid_daemon::pulls::pull(&catalog, runtime.id, id, &pull_flags).unwrap_or_else(|e| {
            superfluid_daemon::events::fail(1, format!("--speculate {kind}:{id}: {e}"));
        });
        speculate = Some(format!("{kind}:{}", path.display()));
    }
    eprintln!("superfluid: runtime {} serves {} ({})", runtime.id.name(), model.display(), describe_worker(&runtime));
    engine.export_spec_bitexact();
    let directive = speculation_for(speculate.as_deref(), &runtime.static_capabilities(), &model, runtime.id.name());
    let speculate_auto = directive.map(str::trim) == Some("auto");
    let resolved = directive.map(|d| {
        let pull = auto_drafter_pull(&catalog, runtime.id, &model);
        let pull = (!pull_flags.offline).then_some(&pull as &dyn Fn(&str));
        superfluid_daemon::speculate_resolve::resolve_all(d, &model, pull)
    });
    let (speculate_primary, speculate_fallbacks): (Option<String>, Vec<String>) = match resolved {
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
            superfluid_daemon::events::fail(2, e);
        }
    };
    if op_spec_max_temperature.is_some() && speculate_directive(speculate.as_deref(), runtime.id.name()).is_none() {
        eprintln!("superfluid: --spec-max-temperature ignored: no --speculate strategy is configured");
    }
    if tui_flag && !cfg!(feature = "tui") {
        superfluid_daemon::events::fail(
            2,
            "--tui: this build carries no TUI (build superfluid-daemon with the `tui` feature, \
             which `basert` includes)",
        );
    }
    let class_lanes = match class_lanes_spec.as_deref() {
        None => [0; 4],
        Some(spec) => superfluid_daemon::qos::parse_class_lanes(spec, max_batch as usize).unwrap_or_else(|e| {
            superfluid_daemon::events::fail(2, format!("--class-lanes: {e}"));
        }),
    };
    if let Err(e) = completion_bucket.validate() {
        superfluid_daemon::events::fail(2, format!("--completion-rate/--completion-burst: {e}"));
    }
    if fim_max_batch == 0 {
        superfluid_daemon::events::fail(2, "--fim-max-batch must be at least 1");
    }
    if sessions_cache && park {
        superfluid_daemon::events::fail(2, format!(
            "--sessions-cache discards the session log at each start, which \
             --park{} needs to resume parked sessions. Use one or the other.",
            if park_lossy { "-lossy" } else { "" }
        ));
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
        match std::fs::OpenOptions::new().create(true).append(true).open(path) {
            Ok(f) => {
                // What was written before goes where stderr was (JSON mode: still in the relay).
                superfluid_daemon::events::drain(std::time::Duration::from_secs(2));
                // Under --log-format json that is the stderr the events relay to.
                superfluid_daemon::events::redirect_output(f);
            }
            Err(e) => {
                superfluid_daemon::events::fail(1, format!("could not open --log-file {}: {e}", path.display()));
            }
        }
    }
    if api_key.is_none() && key_policy.is_none() {
        let loopback = |addr: &str| {
            addr.starts_with("localhost:") || addr.parse::<std::net::SocketAddr>().is_ok_and(|a| a.ip().is_loopback())
        };
        if http.as_deref().is_none_or(loopback) && web.as_deref().is_none_or(loopback) {
            eprintln!(
                "superfluid: no API key (--api-key-file, --api-key or $SUPERFLUID_API_KEY): \
                 the HTTP API is open to every process on this machine"
            );
        } else {
            eprintln!(
                "superfluid: WARNING no API key set. Authentication is OFF — anyone who can reach \
                 this port can use the API. Bind loopback or set --api-key-file (or --api-key, \
                 $SUPERFLUID_API_KEY)."
            );
        }
    }
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

    let tui_on = cfg!(feature = "tui")
        && !no_tui
        && !superfluid_daemon::events::json()
        && (tui_flag || std::io::IsTerminal::is_terminal(&std::io::stdout()));
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
    let worker_stderr: Option<superfluid_daemon::runtime::WorkerStderr> = None;
    let log_dir_eff = log_dir.or(if tui_on { Some(sessions.join("logs")) } else { None });
    if let Some(url) = &otlp_metrics {
        if let Err(e) = superfluid_daemon::otlp::validate_http_url(url) {
            superfluid_daemon::events::fail(2, format!("--otlp-metrics: {e}"));
        }
        if otlp_interval_ms == 0 {
            superfluid_daemon::events::fail(2, "--otlp-interval-ms must be > 0");
        }
    }
    let otlp = engine.otlp_traces().unwrap_or_else(|e| superfluid_daemon::events::fail(2, e));
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
    // A fleet head keeps its sessions too: its handlers commit them as a local server's do.
    let _sessions_lock = lock_sessions_dir(&sessions);
    if sessions_cache {
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

    let kv_bits = engine.kv_bits_for(park_lossy).unwrap_or_else(|e| fail_unprefixed(2, e));

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
            superfluid_daemon::events::fail(2, format!("{asked}: pass --max-context <tokens>"));
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
                    // Another model given at startup: its marked drafter is pulled before the sizing
                    // counts it, by the runtime that serves it, where that runtime drafts under `auto`.
                    // With no `--speculate` named, that runtime's default decides (speculate_directive).
                    let rt = runtime_pick.resolve(m, &catalog).and_then(|rt| catalog.require(rt)).ok();
                    let directive = speculate_directive(speculate.as_deref(), rt.as_ref().map_or("", |rt| rt.id.name()));
                    let serves = rt.filter(|rt| rt.static_capabilities().declines_auto_speculation().is_none());
                    let pull = serves.map(|rt| auto_drafter_pull(&catalog, rt.id, m));
                    let pull = pull.as_ref().filter(|_| !pull_flags.offline).map(|f| f as &dyn Fn(&str));
                    directive.and_then(|d| resolve_all(d, m, pull).ok()).unwrap_or_default()
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
                superfluid_daemon::events::fail(2, e);
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
    superfluid_daemon::events::emit("model.load.start", serde_json::json!({"model": model_name, "runtime": runtime.id.name(), "path": model.display().to_string()}));
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
        superfluid_daemon::events::emit("model.load.failed", serde_json::json!({"model": model_name, "error": e}));
        superfluid_daemon::events::fail(2, e);
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
            superfluid_daemon::events::emit("model.load.failed", serde_json::json!({"model": model_name, "error": e}));
            superfluid_daemon::events::fail(2, e);
        }
    };

    let store = open_session_store(&sessions);

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
            superfluid_daemon::events::fail(1, format!(
                "--fim-model needs superfluid-workerd (the satellite runs in its own \
                 worker), not found at {}",
                workerd_bin.display()
            ));
        }
        if models.len() > 1 {
            superfluid_daemon::events::fail(1, format!(
                "{} models need superfluid-workerd (one address space each), \
                 not found at {}",
                models.len(),
                workerd_bin.display()
            ));
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
            superfluid_daemon::events::emit("model.load.failed", serde_json::json!({"model": model_name, "error": e.to_string()}));
            superfluid_daemon::events::fail(1, format!(
                "could not start the engine worker for '{model_display}': {e} \
                 (the worker's stderr above carries the engine's reason)"
            ));
        })
    } else {
        superfluid_daemon::linked::spawn(runtime.id, model, ctx, max_batch, kv_bits)
            .unwrap_or_else(|| Err(format!("this build does not link the {} runtime", runtime.id)))
            .unwrap_or_else(|e| {
                superfluid_daemon::events::emit("model.load.failed", serde_json::json!({"model": model_name, "error": e}));
                superfluid_daemon::events::fail(1, format!("could not load '{model_display}': {e}"));
            })
    };

    let caps = host.capabilities().read().expect("capability record").clone();
    if let Err(e) = caps.check_startup_flags(park_lossy, kv_bits_arg, speculate.as_deref()) {
        superfluid_daemon::events::fail(2, e);
    }
    if prefix_blob_budget_mb > 0 {
        if let superfluid_daemon::capabilities::Cap::No(why) = caps.get("ops", "space_promote") {
            eprintln!("superfluid: --prefix-blob-budget-mb has no effect on runtime {}: {why}", runtime.id.name());
        }
    }
    if op_presence_penalty.is_some() {
        use superfluid_daemon::capabilities::Cap;
        let why = match caps.get("sampling", "engine_penalties") {
            Cap::Yes => None,
            Cap::No(why) => Some(why.to_string()),
            Cap::Unknown if runtime.worker.source == superfluid_daemon::runtimes::Source::Installed => Some(format!(
                "its worker, as installed, does not say whether its engine applies the penalties itself \
                 (`superfluid runtime repair {}` installs it again with this superfluid's worker)",
                runtime.id
            )),
            Cap::Unknown => Some("its engine does not say whether it applies the penalties itself".to_string()),
        };
        if let Some(why) = why {
            eprintln!(
                "superfluid: --presence-penalty has no effect on runtime {}: {why}; the default is for an \
                 engine that applies the penalty itself, as basert's does (a request's own presence_penalty \
                 still applies)",
                runtime.id.name()
            );
        }
    }

    let listener = socket.as_ref().map(|socket| {
        let _ = std::fs::remove_file(socket);
        let listener = UnixListener::bind(socket).unwrap_or_else(|e| {
            superfluid_daemon::events::fail(1, format!("cannot listen on --socket {}: {e}", socket.display()));
        });
        eprintln!("superfluid: serving on {}", socket.display());
        listener
    });
    let park_dir = park.then(|| sessions.join("park"));

    let codec: Box<dyn superfluid_daemon::TextCodec + Send + Sync> = match tool_envelope {
        Some(env) => {
            if let Some(why) = superfluid_daemon::codec::tool_envelope_override_conflict(&*codec, env) {
                superfluid_daemon::events::fail(2, why);
            }
            if env == codec.tool_envelope() {
                // Nothing to override, and wrapping would drop what the envelope does not
                // carry: an XML wire reports the JSON envelope but is parsed and streamed as XML.
                eprintln!("superfluid: --tool-call-parser {} is this model's own tool wire", env.name());
                codec
            } else {
                eprintln!("superfluid: tool-call envelope overridden to '{}'", env.name());
                Box::new(superfluid_daemon::codec::WithToolEnvelope::new(codec, env))
            }
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
                speculate_auto,
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
            superfluid_daemon::events::emit("model.load.failed", serde_json::json!({"model": model_name, "error": e.to_string()}));
            superfluid_daemon::events::fail(2, format!("cannot serve '{model_display}': {e}"));
        }),
    );
    if let Some(said) = kv_cache(&daemon, kv_bits) {
        eprintln!("superfluid: {said}");
    }
    superfluid_daemon::events::emit("model.load.done", serde_json::json!({"model": model_name, "runtime": runtime.id.name(), "context": ctx, "lanes": max_batch}));
    let fim_satellite: Option<(String, Arc<Daemon>)> = fim_model.as_ref().map(|path| {
        let runtime = catalog.reader_of(path).and_then(|id| catalog.require(id)).unwrap_or_else(|e| {
            superfluid_daemon::events::fail(2, format!("--fim-model {}: {e}", path.display()));
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
            superfluid_daemon::events::fail(1, format!("could not start the --fim-model satellite '{}': {e}", path.display()));
        });
        let name = superfluid_daemon::registry::model_id_for(path);
        eprintln!("superfluid: FIM completions served by satellite '{name}'");
        (name, sat)
    });
    if let Some((name, sat)) = &fim_satellite {
        if let Err(e) = daemon.attach_fim_satellite(name.clone(), Arc::clone(sat)) {
            superfluid_daemon::events::fail(2, format!(
                "--fim-model {}: {e} (its tokenizer has no atomic <|fim_prefix|>/<|fim_suffix|>/<|fim_middle|> tokens)",
                fim_model.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
            ));
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
            offline: pull_flags.offline,
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
        let on = runtime_pick.resolve(extra, &catalog).ok().map(|id| id.name());
        superfluid_daemon::events::emit("model.load.start", serde_json::json!({"model": name, "runtime": on, "path": extra.display().to_string()}));
        match registry.load_as(&name, &extra.to_string_lossy()) {
            Ok(true) => eprintln!("superfluid: loaded model '{name}'"),
            Ok(false) => eprintln!("superfluid: model '{name}' already loaded, skipping"),
            Err(e) => {
                #[cfg(feature = "tui")]
                if let Some(c) = stderr_capture.as_ref() {
                    c.drain(std::time::Duration::from_secs(2));
                }
                superfluid_daemon::events::emit("model.load.failed", serde_json::json!({"model": name, "error": e.to_string()}));
                superfluid_daemon::events::fail(1, format!("could not load '{}': {e}", extra.display()));
            }
        }
        superfluid_daemon::events::emit("model.load.done", serde_json::json!({"model": name, "runtime": on, "context": ctx, "lanes": max_batch}));
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
                superfluid_daemon::events::fail(1, format!("--model-dir {}: {e}", dir.display()));
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
    let mut http_bound = None;
    if let Some(addr) = &http {
        let http_listener = listen_or_exit("--http", addr);
        http_bound = http_listener.local_addr().ok();
        eprintln!("superfluid: openai api on http://{addr}{}", embedding.http.also_served());
        superfluid_daemon::selfupdate::check_in_background(catalog.home().to_path_buf(), pull_flags.offline);
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
                presence_penalty: op_presence_penalty,
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
            idempotency_dir: Some(sessions.join("idempotency")),
        };
        http_thread = Some(std::thread::spawn(move || {
            superfluid_daemon::openai::serve_blocking_config(http_listener, reg, serve_cfg)
        }));
    }
    if let Some(addr) = web {
        if !addr.starts_with("127.0.0.1:") && !addr.starts_with("[::1]:") && !addr.starts_with("localhost:") {
            superfluid_daemon::events::fail(2, "--web must bind loopback (127.0.0.1/::1)");
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
    let default = registry.default_name();
    let ready: Vec<String> = std::iter::once(default.clone()).chain(registry.names().into_iter().filter(|n| *n != default)).collect();
    superfluid_daemon::events::emit("serve.ready", serde_json::json!({"http": http_bound.map(|a| a.to_string()), "socket": socket.as_ref().map(|s| s.display().to_string()), "models": ready}));
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
            Some(Ok(Err(e))) => superfluid_daemon::events::fail(1, format!("the HTTP API stopped: {e}")),
            Some(Err(_)) => superfluid_daemon::events::fail(1, "the HTTP API stopped: its thread panicked"),
            None => unreachable!("without a session socket --no-http is refused"),
        }
    };
    api::serve(listener, daemon).expect("serve");
}

#[cfg(feature = "tui")]
fn hostname() -> String {
    superfluid_daemon::sys::hostname().unwrap_or_else(|| "local".into())
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
    /// `--offline`: `auto` pulls no drafter for a model loaded later either.
    offline: bool,
    speculation: superfluid_daemon::SpeculationOptions,
    completion_deadline_ms: u64,
    completion_bucket: superfluid_daemon::completion_bucket::BucketConfig,
    fim_satellite: Option<(String, Arc<Daemon>)>,
    worker_stderr: Option<superfluid_daemon::runtime::WorkerStderr>,
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
    worker_stderr: Option<superfluid_daemon::runtime::WorkerStderr>,
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

/// The KV cache precision the engine loaded `daemon` with, as the start says it: `kv_bits` is the one
/// asked for (0, auto). None when the engine does not say.
pub(crate) fn kv_cache(daemon: &Daemon, kv_bits: i32) -> Option<String> {
    let bits = daemon.sched_stats().kv_bits.load(std::sync::atomic::Ordering::Relaxed);
    let label = superfluid_daemon::scheduler::kv_bits_label(bits)?;
    Some(if kv_bits != 0 && bits != kv_bits as u64 {
        format!("kv cache {label} (--kv-bits {kv_bits} overridden by the engine, see above)")
    } else {
        format!("kv cache {label}")
    })
}

pub(crate) fn refuse_statically(
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

/// The digest of the tokenizer `model` is read with on `runtime`, as a fleet head and its nodes
/// compare them; none when no tokenizer opens here.
pub fn tokenizer_digest(runtime: &superfluid_daemon::runtimes::Runtime, model: &std::path::Path) -> Option<u64> {
    open_tokenizer(runtime, model).ok().map(|t| superfluid_daemon::fleet::tokenizer_digest(&*t))
}

pub fn open_tokenizer(
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

/// What a fleet head needs to serve a model over its nodes, at start or once a node brings it.
struct FleetModel {
    manager: Arc<superfluid_daemon::fleet_manager::FleetManager>,
    sessions: PathBuf,
    /// The model the head started with first.
    first: String,
    dialect: String,
    tool_lease_ms: u64,
    files_max_bytes: Option<u64>,
    files_expiry: Option<u64>,
    completion_deadline_ms: u64,
}

impl FleetModel {
    /// Where `name` keeps its sessions: the first model's where a single server keeps them,
    /// each other's in a directory of its own, as a server with several models keeps them.
    fn dir(&self, name: &str) -> std::io::Result<PathBuf> {
        if name == self.first {
            return Ok(self.sessions.clone());
        }
        superfluid_daemon::registry::state_dir(&self.sessions.join("models"), name)
    }

    /// The codec for `name` from the tokenizer of a node serving it, which every node serving
    /// it is held to from now on.
    fn codec(
        &self,
        name: &str,
        tok: superfluid_daemon::fleet::RemoteTokenizer,
    ) -> Result<Box<dyn superfluid_daemon::TextCodec + Send + Sync>, String> {
        self.manager.expect_tokenizer(name, tok.digest());
        codec_for_dialect(&self.dialect, Arc::new(tok)).map(|(c, _)| c)
    }

    /// The daemon that serves `name` over the fleet, its sessions in `store`.
    fn daemon(
        &self,
        name: &str,
        store: SessionStore,
        codec: Box<dyn superfluid_daemon::TextCodec + Send + Sync>,
    ) -> Result<Daemon, superfluid_daemon::DaemonError> {
        Daemon::over_fleet(
            store,
            codec,
            Arc::clone(&self.manager),
            name.to_string(),
            superfluid_daemon::DaemonOptions {
                tool_lease_ms: self.tool_lease_ms,
                media_dir: Some(self.sessions.join("media")),
                files_max_bytes: self.files_max_bytes,
                files_expiry_secs: self.files_expiry,
                completion_deadline_ms: self.completion_deadline_ms,
                trace_scope: superfluid_daemon::otlp::TraceScope::model(name),
                ..Default::default()
            },
        )
    }
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

/// `--speculate auto`'s pull (speculate_resolve::resolve_all) of the drafter the hub catalog marks
/// for `target` when no copy is installed, by `runtime`, the one that serves the target, as a drafter
/// given by id is pulled. A pull that fails is said, and `auto` settles among what is installed. The
/// default `auto` (no `--speculate`, speculate_directive) pulls as a named one does.
pub(crate) fn auto_drafter_pull<'a>(
    catalog: &'a superfluid_daemon::runtimes::Catalog,
    runtime: RuntimeId,
    target: &'a std::path::Path,
) -> impl Fn(&str) + 'a {
    move |id: &str| {
        eprintln!("superfluid: speculation: auto: the hub catalog pairs {} with {id}, which is not installed", target.display());
        if let Err(e) = superfluid_daemon::pulls::pull(catalog, runtime, id, &superfluid_daemon::pulls::PullFlags::default()) {
            eprintln!("superfluid: speculation: auto: {id}: {e} — choosing among what is installed");
        }
    }
}

/// The `--speculate` a model on `runtime` serves under: the one the command line names (`off` is
/// none), else `auto` on the baseRT runtime, where speculative decoding is on by default, and none on
/// the other runtimes, whose `auto` is prompt-lookup. A pinned installation (`basert@<install>`) is
/// its base runtime here (is_basert).
pub(crate) fn speculate_directive<'a>(named: Option<&'a str>, runtime: &str) -> Option<&'a str> {
    match named {
        Some(d) if d.trim() == "off" => None,
        Some(d) => Some(d),
        None => is_basert(runtime).then_some("auto"),
    }
}

/// Whether `runtime` (a runtime id, pinned to an installation or not) is the baseRT runtime.
fn is_basert(runtime: &str) -> bool {
    RuntimeId::new(runtime).base().name() == "basert"
}

/// The strategy a model serves with, from the `--speculate` the command line names (None: not named;
/// see speculate_directive): `auto` settled for the runtime and the engine's capabilities.
pub(crate) fn speculation_for<'a>(
    named: Option<&'a str>,
    caps: &superfluid_daemon::capabilities::Capabilities,
    model: &std::path::Path,
    runtime: &str,
) -> Option<&'a str> {
    let d = speculate_directive(named, runtime)?;
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
        None if !is_basert(runtime) => {
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
    // A model loaded after startup (another --model, --model-dir, /v1/models/load) pulls its marked
    // drafter here, as the first model does before it starts.
    let pull = auto_drafter_pull(&cfg.catalog, runtime.id, model_path);
    let pull = (!cfg.offline).then_some(&pull as &dyn Fn(&str));
    let directive = speculation_for(cfg.speculate.as_deref(), &caps, model_path, runtime.id.name());
    let mut speculate_ranked = match directive {
        Some(d) => superfluid_daemon::speculate_resolve::resolve_all(d, model_path, pull).map_err(DaemonError::Generation)?,
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
            speculate_auto: directive.map(str::trim) == Some("auto"),
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
    let path = sessions.join("serve.lock");
    let file = match std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path) {
        Ok(file) => file,
        Err(e) => {
            superfluid_daemon::events::fail(1, format!("could not open {}: {e}", path.display()));
        }
    };
    match superfluid_daemon::sys::lock_exclusive(&file, false) {
        Ok(true) => file,
        Ok(false) => superfluid_daemon::events::fail(1, format!(
            "another server is already using the sessions directory {}. \
             Stop it, or give this one its own --sessions directory.",
            sessions.display()
        )),
        Err(err) => superfluid_daemon::events::fail(1, format!("could not lock {}: {err}", path.display())),
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
            superfluid_daemon::events::fail(1, format!(
                "--sessions-cache could not set aside or remove {} ({e}); \
                 remove it, or run without --sessions-cache.",
                from.display()
            ));
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

    /// Speculative decoding is on by default on the basert runtime only: no `--speculate` is `auto`
    /// there and plain decoding elsewhere (where `auto` would be prompt-lookup); a directive named on
    /// the command line, `off` included, is the user's on every runtime, and an engine that verifies no
    /// drafts declines the default as it declines a named `auto`.
    #[test]
    fn speculation_is_auto_by_default_on_basert_only() {
        assert_eq!(speculate_directive(None, "basert"), Some("auto"));
        assert_eq!(speculate_directive(None, "llamacpp"), None);
        assert_eq!(speculate_directive(None, "mlx"), None);
        assert_eq!(speculate_directive(None, ""), None, "a runtime not resolved");
        assert_eq!(speculate_directive(Some("off"), "basert"), None);
        assert_eq!(speculate_directive(Some(" off"), "basert"), None);
        assert_eq!(speculate_directive(Some("mtp-head"), "basert"), Some("mtp-head"));
        assert_eq!(speculate_directive(Some("auto"), "llamacpp"), Some("auto"));

        let m = std::path::Path::new("m.base");
        let caps = superfluid_daemon::capabilities::Capabilities::default();
        assert_eq!(speculation_for(None, &caps, m, "basert"), Some("auto"));
        assert_eq!(speculation_for(None, &caps, m, "llamacpp"), None);
        assert_eq!(speculation_for(Some("off"), &caps, m, "basert"), None);
        assert_eq!(speculation_for(Some("auto"), &caps, m, "basert"), Some("auto"));
        assert_eq!(speculation_for(Some("auto"), &caps, m, "llamacpp"), Some("prompt-lookup"));
        let hybrid = superfluid_daemon::capabilities::Capabilities::parse(
            r#"{"serving":{"multi_row_verification":"Mamba-2 hybrid: no speculative verification"}}"#,
        );
        assert_eq!(speculation_for(None, &hybrid, m, "basert"), None);
        assert_eq!(speculation_for(Some("auto"), &hybrid, m, "basert"), None);

        // A pinned installation (`--runtime basert@<install>`) is the runtime it pins: its id keeps the
        // suffix, and the policy goes by the base runtime.
        let pinned = |id: &str| RuntimeId::parse(id).expect("a valid runtime id").name();
        assert_eq!(speculate_directive(None, pinned("basert@0.5.0-metal")), Some("auto"));
        assert_eq!(speculate_directive(None, pinned("llamacpp@b12408")), None);
        assert_eq!(speculation_for(None, &caps, m, pinned("basert@0.5.0-metal")), Some("auto"));
        assert_eq!(speculation_for(Some("auto"), &caps, m, pinned("basert@0.5.0-metal")), Some("auto"));
        assert_eq!(speculation_for(Some("auto"), &caps, m, pinned("llamacpp@b12408")), Some("prompt-lookup"));
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
