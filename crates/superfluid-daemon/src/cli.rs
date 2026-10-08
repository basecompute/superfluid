//! The `superfluid` command line: its help text, the `serve` flag table and the defaults that let
//! `superfluid serve <model>` start with no other flag.

use std::path::{Path, PathBuf};

pub const DEFAULT_HTTP: &str = "127.0.0.1:8453";

pub struct Flag {
    pub name: &'static str,
    pub value: Option<&'static str>,
    pub default: &'static str,
    pub help: &'static str,
}

pub struct Group {
    pub title: &'static str,
    pub flags: &'static [Flag],
}

const fn f(name: &'static str, value: Option<&'static str>, default: &'static str, help: &'static str) -> Flag {
    Flag { name, value, default, help }
}

pub const SERVE: &[Group] = &[
    Group {
        title: "Model",
        flags: &[
            f("--model", Some("<path|org/model[:tag]>"), "", "A model to serve, repeatable; the first is the default. Also the first bare argument."),
            f("--runtime", Some("<id>[@<install>]|<model>=<id>"), "by format", "The runtime to serve with (basert, llamacpp, mlx); `auto` clears a pick."),
            f("--offline", None, "", "Fetch nothing: a model id is only looked for in its runtime's cache."),
            f("--pull-<name>", Some("[value]"), "", "An option for the runtime that pulls a model id (e.g. --pull-file, --pull-revision)."),
            f("--model-dir", Some("<dir>"), "", "Every model in this directory is listed and loaded on first use."),
            f("--idle-timeout", Some("<sec>"), "0 (never)", "Unload a model that has been idle this long (never the default model)."),
            f("--basert-lib", Some("<path>"), "search", "Where libbaseRT is, for the basert runtime."),
        ],
    },
    Group {
        title: "Server",
        flags: &[
            f("--http", Some("<addr:port>"), DEFAULT_HTTP, "Serve the OpenAI-, Anthropic- and Ollama-compatible HTTP APIs here."),
            f("--no-http", None, "", "Serve only the native session API on the unix socket."),
            f("--host", Some("<addr>"), "127.0.0.1", "The HTTP address, as a host..."),
            f("--port", Some("<N>"), "8453", "...and a port; either fills the other from its default."),
            f("--socket", Some("<path>"), "$SUPERFLUID_HOME/superfluid.sock", "The unix socket for the native session API."),
            f("--sessions", Some("<dir>"), "$SUPERFLUID_HOME/sessions", "The session store (write-ahead log, parks, media, files). One server per directory."),
            f("--api-key", Some("<key>"), "none", "Require `Authorization: Bearer <key>` (or `X-Api-Key`) on HTTP requests."),
            f("--key-policy", Some("<file.json>"), "", "Per-key classes, rate and concurrency limits; turns authentication on."),
            f("--rate-limit", Some("<rpm>"), "0 (off)", "Requests per minute per client IP."),
            f("--drain-timeout", Some("<sec>"), "60", "On SIGUSR1, stop accepting and give in-flight requests this long."),
            f("--nonstream-keepalive", Some("<sec>"), "0 (off)", "Send 200 at admission and a space every <sec> until a non-streaming body is ready."),
            f("--web", Some("<127.0.0.1:port>"), "off", "The browser-facing API, with a token minted to <sessions>/web-token. Loopback only."),
            f("--web-origin", Some("<url>"), "", "An allowed Origin for --web, repeatable."),
        ],
    },
    Group {
        title: "Generation",
        flags: &[
            f("--max-context", Some("<N|auto>"), "auto", "The context window per lane; auto sizes it for the device where the runtime can, else 8192."),
            f("--max-batch", Some("<N>"), "8", "Concurrent decode lanes."),
            f("--max-tokens", Some("<N>"), "fill the context", "The generation cap for requests that set none."),
            f("--kv-bits", Some("<0|4|8|16|84>"), "0 (auto)", "KV cache precision, on runtimes that take it (basert)."),
            f("--dialect", Some("<auto|chatml|atem|template|raw>"), "auto", "The chat codec."),
            f("--tool-call-parser", Some("<auto|json|atem|gemma|harmony|glm>"), "auto", "Override the tool-call format the dialect speaks."),
            f("--temperature", Some("<F>"), "model's", "Default temperature (0..2) for requests that set none."),
            f("--top-p", Some("<F>"), "model's", "Default nucleus truncation (0..1)."),
            f("--top-k", Some("<N>"), "model's", "Default top-k truncation."),
            f("--min-p", Some("<F>"), "model's", "Default min-p truncation (0..1)."),
            f("--repeat-penalty", Some("<F>"), "model's", "Default repetition penalty (0..2); 1.0 disables it."),
        ],
    },
    Group {
        title: "Sessions and caching",
        flags: &[
            f("--park", None, "off", "Seal a finishing durable session's KV to disk, to resume it later by id. --no-park turns it off."),
            f("--park-lossy", None, "off", "Park in the lossy Q8 tier (about half the size); implies --park."),
            f("--park-budget-gb", Some("<N>"), "20", "Disk budget per park directory, oldest evicted first; 0 is unbounded."),
            f("--sessions-cache", None, "off", "Treat the session log as a cache: start a fresh one at each start."),
            f("--tool-lease-ms", Some("<N>"), "0 (none)", "A deadline for every tool call."),
            f("--prefix-blob-budget-mb", Some("<N>"), "engine's", "The recurrent-state snapshot cache on hybrid models (basert)."),
        ],
    },
    Group {
        title: "Files API",
        flags: &[
            f("--files-max-bytes", Some("<N>"), "unbounded", "Refuse an upload that would push stored bytes past this."),
            f("--files-expiry", Some("<sec>"), "never", "Delete a stored file this long after upload."),
            f("--files-sweep", Some("<sec>"), "300", "How often the expiry sweep runs."),
        ],
    },
    Group {
        title: "Scheduling and QoS",
        flags: &[
            f("--tick-decode-budget", Some("<N>"), "max-batch x 32", "Decode tokens granted per tick across all lanes."),
            f("--tick-target-ms", Some("<N>"), "2000", "The tick length decode grants and adaptive prefill are sized towards."),
            f("--prefill-budget", Some("<N>"), "4096", "Prompt tokens prefilled per tick (64..4096); 0 adapts it to the tick target."),
            f("--starvation-ticks", Some("<N>"), "8", "Serve an agent lane first after this many ticks without a slice."),
            f("--class-lanes", Some("<class>=N[,...]"), "uncapped", "Concurrent-lane cap per QoS class."),
            f("--http-default-qos", Some("<class>"), "agent", "QoS class for HTTP requests that name none (interactive, completion, agent, background)."),
            f("--no-http-qos-header", None, "", "Ignore the client's x-superfluid-qos header."),
            f("--http-allow-batch-invariant", None, "", "Let an HTTP request ask for a batch-invariant lane."),
            f("--worker-process", None, "on", "Run each engine in its own worker process. --no-worker-process runs it in-process."),
            f("--os-pressure", None, "on", "Relieve OS memory pressure. --no-os-pressure turns it off."),
            f("--pressure-high", Some("<pct>"), "85", "KV pool occupancy above which cache is evicted."),
            f("--pressure-low", Some("<pct>"), "70", "The occupancy eviction brings it down to."),
            f("--pin-budget-pct", Some("<pct>"), "50", "Share of the KV pool pinned prefixes may hold."),
        ],
    },
    Group {
        title: "Speculative decoding",
        flags: &[
            f("--speculate", Some("<auto|prompt-lookup|mtp-head|<drafter>|off>"), "off", "Draft tokens and verify them; also dflash:, dspark:, eagle3:, draft-model: <path>."),
            f("--spec-draft-tokens", Some("<N>"), "per strategy", "Draft depth, 1..15."),
            f("--spec-adaptive", None, "on", "Adapt the depth to the acceptance rate. --no-spec-adaptive pins it."),
            f("--spec-min-yield", Some("<F>"), "0.75", "The yield below which a lane stops drafting."),
            f("--spec-yield-rounds", Some("<N>"), "24", "Rounds over which the yield is judged."),
            f("--spec-throughput-gate", None, "on", "Check that speculation beats plain decoding. --no-spec-throughput-gate skips it."),
            f("--spec-gate-probe-tokens", Some("<N>"), "16", "The gate's probe length."),
            f("--spec-min-speedup", Some("<F>"), "1.08", "The speed-up the gate requires."),
            f("--spec-gate-reprobe", Some("<N>"), "32", "Tokens between re-probes."),
            f("--spec-gate-reprobe-max", Some("<N>"), "1024", "The ceiling the re-probe interval backs off to."),
            f("--spec-max-temperature", Some("<F>"), "none", "Cap a model-default temperature on a speculating model."),
            f("--dspark-confidence", Some("<F>"), "engine's", "The DSpark drafter's confidence threshold."),
            f("--spec-bitexact", None, "off", "Make speculated and plain decoding bit-exact. --no-spec-bitexact turns it off."),
        ],
    },
    Group {
        title: "Code completion (FIM)",
        flags: &[
            f("--fim-model", Some("<path>"), "", "A dedicated fill-in-the-middle model in its own worker."),
            f("--fim-max-context", Some("<N>"), "min(window, 8192)", "The FIM model's window."),
            f("--fim-max-batch", Some("<N>"), "2", "The FIM model's lanes."),
            f("--completion-deadline-ms", Some("<N>"), "0 (none)", "A completion that cannot start within this expires."),
            f("--completion-rate", Some("<F>"), "10", "Completions per second per client (0 = unmetered)."),
            f("--completion-burst", Some("<N>"), "20", "Completions a client may fire back to back."),
        ],
    },
    Group {
        title: "Logging and telemetry",
        flags: &[
            f("--log-filter", Some("<directive>"), "info", "Log level, or per target (superfluid_daemon::scheduler=debug,info)."),
            f("--verbose", None, "", "The same as --log-filter debug; also -v."),
            f("--log-dir", Some("<dir>"), "stderr", "Write JSON logs to rotating files here."),
            f("--log-file", Some("<path>"), "", "Send stderr to this file (appended)."),
            f("--tui", None, "on a terminal", "The live terminal monitor. --no-tui turns it off."),
            f("--otlp-metrics", Some("<url>"), "off", "Push scheduler metrics to an OTLP/HTTP collector."),
            f("--otlp-interval-ms", Some("<N>"), "15000", "The metrics push interval."),
            f("--otlp-endpoint", Some("<url>"), "off", "Push operational traces (never content) to an OTLP/HTTP collector."),
            f("--otlp-header", Some("<k=v>"), "", "A header on every OTLP request, repeatable."),
            f("--otlp-service-name", Some("<name>"), "superfluid", "The service.name resource attribute."),
            f("--otlp-filter", Some("<directive>"), "info,...scheduler=debug", "Which spans are exported."),
            f("--otlp-queue", Some("<N>"), "8192", "The span export queue; a full queue drops spans."),
            f("--otlp-batch-ms", Some("<N>"), "1000", "The span export interval."),
            f("--otlp-timeout-ms", Some("<N>"), "5000", "The export request timeout."),
        ],
    },
    Group {
        title: "Fleet head",
        flags: &[
            f("--fleet", Some("<host:port>[,...]"), "off", "Serve as a fleet head over these superfluid-noded agents."),
            f("--fleet-listen", Some("<addr:port>"), "off", "Take nodes that join (`superfluid node join`), announced on the local network."),
            f("--fleet-auth", Some("<path>"), "$SUPERFLUID_HOME/fleet/token", "The fleet token: the links to the nodes are encrypted with it."),
            f("--fleet-policy", Some("<load-aware|least-loaded|round-robin>"), "load-aware", "Session placement."),
            f("--fleet-pool-high", Some("<pct>"), "90", "Pass over a node whose KV pool is at least this full."),
            f("--fleet-conns-per-node", Some("<N>"), "lanes", "Connections to each node you list; by default as many as it has lanes."),
        ],
    },
];

/// Accepted spellings that are not rows of their own in the table: the `--no-` halves of the
/// switches, short and legacy aliases, and the retired flags `serve` still reads.
const SERVE_ALIASES: &[(&str, Option<&str>)] = &[
    ("--no-park", None),
    ("--no-worker-process", None),
    ("--no-os-pressure", None),
    ("--no-spec-adaptive", None),
    ("--no-spec-throughput-gate", None),
    ("--no-spec-bitexact", None),
    ("--no-tui", None),
    ("-v", None),
    ("--help", None),
    ("-h", None),
    ("--max-batch-size", Some("--max-batch")),
    ("--request-timeout", Some("--completion-deadline-ms")),
];

const SERVE_RETIRED_WITH_VALUE: &[&str] = &[
    "--prefix-cache-file",
    "--prefix-cache-save-interval",
    "--files-dir",
    "--media-dir",
    "--metallib",
    "--prefill-chunk",
    "--gpu-wait-timeout-ms",
    "--decode-replay",
];

const SERVE_RETIRED_SWITCHES: &[&str] =
    &["--paged-kv", "--prefix-cache", "--continuous-batching", "--paged-weights", "--no-paged-weights-retry", "--no-baked-decode"];

fn serve_flag(name: &str) -> Option<&'static Flag> {
    SERVE.iter().flat_map(|g| g.flags.iter()).find(|fl| fl.name == name)
}

/// Whether `serve` reads `name` with a value after it: `Some(true)` it does, `Some(false)` it is a
/// switch, `None` no such flag.
pub fn serve_takes_value(name: &str) -> Option<bool> {
    if let Some(fl) = serve_flag(name) {
        return Some(fl.value.is_some());
    }
    if let Some((_, alias)) = SERVE_ALIASES.iter().find(|(n, _)| *n == name) {
        return Some(alias.and_then(serve_flag).is_some_and(|fl| fl.value.is_some()));
    }
    if SERVE_RETIRED_WITH_VALUE.contains(&name) {
        return Some(true);
    }
    SERVE_RETIRED_SWITCHES.contains(&name).then_some(false)
}

/// Splits `--flag=value` into `--flag value` for every flag `takes_value` says reads a value.
/// `--pull-<name>=<value>` is left whole: its parser reads that form itself.
pub fn split_equals(args: &[String], takes_value: impl Fn(&str) -> Option<bool>) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    for a in args {
        match a.split_once('=') {
            Some((flag, value)) if flag.starts_with("--") && !flag.starts_with("--pull-") && takes_value(flag) == Some(true) => {
                out.push(flag.to_string());
                out.push(value.to_string());
            }
            _ => out.push(a.clone()),
        }
    }
    out
}

fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(ca != *cb)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

/// The known flag closest to `unknown`, when one is close enough to be what was meant.
pub fn suggest<'a>(unknown: &str, known: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let unknown = unknown.split_once('=').map_or(unknown, |(f, _)| f);
    known
        .into_iter()
        .map(|k| (distance(unknown, k), k))
        .filter(|(d, k)| *d <= 2.max(k.len() / 3) || (unknown.len() > 4 && k.starts_with(unknown)))
        .min_by_key(|(d, _)| *d)
        .map(|(_, k)| k)
}

pub fn serve_flag_names() -> impl Iterator<Item = &'static str> {
    SERVE.iter().flat_map(|g| g.flags.iter().map(|fl| fl.name)).chain(SERVE_ALIASES.iter().map(|(n, _)| *n)).filter(|n| *n != "--pull-<name>")
}

pub fn version() -> String {
    format!("superfluid {}", env!("CARGO_PKG_VERSION"))
}

pub fn overview() -> String {
    format!(
        "{}
A serving daemon for local LLM inference: one scheduler, one session log and one set of APIs
over llama.cpp, MLX and baseRT.

Usage: superfluid <command> [options]

Commands:
  serve <model>        Serve a model over HTTP (OpenAI, Anthropic, Ollama) and a unix socket
  launch <agent>       Run a coding agent (Claude Code, ...) against a local model
  stop                 Stop the server `launch` started
  fleet token          Print the fleet token, making it the first time
  node join [<head>]   Serve a model to a fleet head from this machine
  runtimes [<model>]   List the inference runtimes here, or which one serves a model
  runtime <action>     Install, update, repair, remove or pin a runtime
  session <action>     Inspect or export a session of a running server
  media <action>       Store or garbage-collect media on a running server
  top <host:port>...   Monitor running servers in the terminal
  help [<command>]     Show help for a command

Options:
  -h, --help           Show help
  -V, --version        Show the version

Get started:
  superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M     # llama.cpp; installed and pulled on first run
  superfluid serve mlx-community/Qwen3-4B-4bit        # MLX on Apple silicon
  superfluid serve ./model.gguf
  curl http://{DEFAULT_HTTP}/v1/models

Run 'superfluid <command> --help' for a command's options.
Docs: https://github.com/basecompute/superfluid/tree/main/docs
",
        version()
    )
}

pub fn serve_help() -> String {
    let mut out = String::from(
        "Serve one or more models.

Usage: superfluid serve <model> [options]

  <model> is a local path (a .gguf file, an MLX directory, a .base bundle) or a Hugging Face
  style id, org/model[:tag]. An id is pulled by its runtime on first use; the runtime is taken
  from the id (a *GGUF* repo -> llamacpp, mlx-community/* or *mlx* -> mlx, else basert) unless
  --runtime names one. A runtime that is not installed is installed first, unless --offline.

Examples:
  superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
  superfluid serve ./Qwen3-4B-Q4_K_M.gguf --port 8000 --api-key secret
  superfluid serve --model a.gguf --model b.gguf --max-batch 16

Every option that takes a value accepts --flag value and --flag=value.
",
    );
    for g in SERVE {
        out.push_str(&format!("\n{}:\n", g.title));
        for fl in g.flags {
            let head = match fl.value {
                Some(v) => format!("{} {v}", fl.name),
                None => fl.name.to_string(),
            };
            let default = if fl.default.is_empty() { String::new() } else { format!(" [default: {}]", fl.default) };
            out.push_str(&row(&head, &format!("{}{default}", fl.help)));
        }
    }
    out
}

const HEAD: usize = 32;
const WIDTH: usize = 100;

/// One option row: the flag in a column of its own, the text word-wrapped beside it.
fn row(head: &str, text: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && HEAD + 2 + line.len() + 1 + word.len() > WIDTH {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    lines.push(line);
    let pad = " ".repeat(HEAD + 2);
    let mut out = if head.chars().count() + 2 > HEAD {
        format!("  {head}\n{pad}{}\n", lines[0])
    } else {
        format!("  {head:<w$}{}\n", lines[0], w = HEAD)
    };
    for l in &lines[1..] {
        out.push_str(&format!("{pad}{l}\n"));
    }
    out
}

pub const RUNTIMES_HELP: &str = "List the inference runtimes here, or which one serves a model.

Usage: superfluid runtimes [<model>] [--json]

  With no model: one row per runtime and install, its state, version, the formats it reads and
  its device, then what to do about any that is not ready.
  With a model: which runtime reads it and what that runtime serves and refuses (exit 1 if none).

Options:
  --json    Print JSON
";

pub const RUNTIME_HELP: &str = "Install and manage inference runtimes.

Usage:
  superfluid runtime install <id> [options]     Fetch and install a runtime for this machine
  superfluid runtime update <id>                Install the newest tested build beside the current one
  superfluid runtime repair <id> [<install>]    Reinstall an install in place
  superfluid runtime remove <id> [<install>]    Remove one install, or all of them
  superfluid runtime use <id> <install>|--auto  Pin the default install, or go back to the best-ranked

  <id> is basert, llamacpp or mlx. Installs live under $SUPERFLUID_HOME/runtimes (~/.superfluid).

Install options:
  --version <V>          A runtime version (default: the newest this adapter was tested with)
  --backend <B>          metal, cuda, vulkan, rocm, cpu, ... (default: the best for this machine)
  --from <url|path>      An archive of your own (for mlx, a package index)
  --sha256 <H>           The archive's digest, checked when given
  --untested             Allow a version the adapter was not tested with
  --dry-run              Print the plan and stop

Examples:
  superfluid runtime install llamacpp
  superfluid runtime install llamacpp --backend vulkan --dry-run
";

pub const SESSION_HELP: &str = "Inspect or export a session of a running server.

Usage:
  superfluid session inspect <id> [--socket <path>] [--json]
  superfluid session export <id> [--socket <path>] --format jsonl|otlp-jsonl [options]

Options:
  --socket <path>            The server's socket [default: $SUPERFLUID_HOME/superfluid.sock]
  --json                     (inspect) Print JSON
  --format <jsonl|otlp-jsonl> (export) Events as JSON lines, or OTLP trace lines [default: jsonl]
  --out <dir>                (export) Write session-<id>.* here instead of stdout
  --include-content          (export) Include token text
  --model <name>             (export) Label the spans with this model
  --endpoint <url>           (export) Push the traces to an OTLP/HTTP collector instead
  --allow-content-egress     (export) Let --include-content cross the network with --endpoint
";

pub const MEDIA_HELP: &str = "Store or garbage-collect media on a running server.

Usage:
  superfluid media put <file> [--socket <path>]   Store a file; prints its hash
  superfluid media gc [--socket <path>]           Drop media no session refers to

Options:
  --socket <path>    The server's socket [default: $SUPERFLUID_HOME/superfluid.sock]
";

pub const TOP_HELP: &str = "Monitor running servers in the terminal.

Usage: superfluid top [--api-key <key>] <host:port> [<host:port> ...]

  Each address is a `superfluid serve` HTTP endpoint; q or Esc quits.

Options:
  --api-key <key>    Sent on every scrape, for servers behind --api-key
";

/// The help for a subcommand, by name.
pub fn command_help(command: &str) -> Option<String> {
    Some(match command {
        "serve" => serve_help(),
        "runtimes" => RUNTIMES_HELP.to_string(),
        "runtime" => RUNTIME_HELP.to_string(),
        "session" => SESSION_HELP.to_string(),
        "media" => MEDIA_HELP.to_string(),
        "top" => TOP_HELP.to_string(),
        "launch" => crate::launch::HELP.to_string(),
        "fleet" => crate::fleet_token::HELP.to_string(),
        "node" => crate::fleet_join::HELP.to_string(),
        "stop" => crate::launch::STOP_HELP.to_string(),
        _ => return None,
    })
}

pub const COMMANDS: &[&str] = &["serve", "launch", "stop", "runtimes", "runtime", "session", "media", "top", "fleet", "node", "help", "version"];

/// `$SUPERFLUID_HOME/sessions`.
pub fn default_sessions(home: &Path) -> PathBuf {
    home.join("sessions")
}

/// `$SUPERFLUID_HOME/superfluid.sock`, or a per-user socket in the temporary directory when that
/// path is too long for a unix socket (`sun_path` is 104 bytes on macOS, 108 on Linux).
pub fn default_socket(home: &Path, sun_path_len: usize) -> PathBuf {
    let sock = home.join("superfluid.sock");
    if sock.as_os_str().len() < sun_path_len {
        return sock;
    }
    // SAFETY: getuid has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    let tmp = std::env::temp_dir().join(format!("superfluid-{uid}.sock"));
    if tmp.as_os_str().len() < sun_path_len {
        tmp
    } else {
        PathBuf::from(format!("/tmp/superfluid-{uid}.sock"))
    }
}

/// The runtime a model id goes to when no `--runtime` names one: a GGUF repository to llama.cpp,
/// an MLX conversion to MLX, anything else to baseRT.
pub fn runtime_for_id(id: &str) -> &'static str {
    let id = id.split_once(':').map_or(id, |(id, _)| id).to_ascii_lowercase();
    let (org, repo) = id.split_once('/').unwrap_or(("", &id));
    if repo.contains("gguf") {
        "llamacpp"
    } else if org == "mlx-community" || repo.contains("mlx") {
        "mlx"
    } else {
        "basert"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn equals_forms_are_split_only_for_flags_that_take_a_value() {
        let got = split_equals(
            &strings(&["--port=8000", "--park", "--pull-file=m.gguf", "--model=a=b.gguf", "x=y", "--nope=1", "--no-http=1"]),
            serve_takes_value,
        );
        assert_eq!(got, strings(&["--port", "8000", "--park", "--pull-file=m.gguf", "--model", "a=b.gguf", "x=y", "--nope=1", "--no-http=1"]));
        assert_eq!(split_equals(&strings(&["--max-batch-size=4"]), serve_takes_value), strings(&["--max-batch-size", "4"]));
    }

    #[test]
    fn a_near_miss_is_suggested_and_a_far_one_is_not() {
        assert_eq!(suggest("--max-ctx", serve_flag_names()), Some("--max-context"));
        assert_eq!(suggest("--temprature", serve_flag_names()), Some("--temperature"));
        assert_eq!(suggest("--api_key", serve_flag_names()), Some("--api-key"));
        assert_eq!(suggest("--sesions=/x", serve_flag_names()), Some("--sessions"));
        assert_eq!(suggest("--frobnicate-everything", serve_flag_names()), None);
    }

    #[test]
    fn a_runtime_is_read_from_a_model_id() {
        assert_eq!(runtime_for_id("unsloth/Qwen3-0.6B-GGUF:Q4_K_M"), "llamacpp");
        assert_eq!(runtime_for_id("bartowski/Llama-3.2-1B-Instruct-gguf"), "llamacpp");
        assert_eq!(runtime_for_id("mlx-community/Qwen3-4B-4bit"), "mlx");
        assert_eq!(runtime_for_id("someone/Qwen3-4B-MLX-8bit:main"), "mlx");
        assert_eq!(runtime_for_id("basecompute/Qwen3-1.7B"), "basert");
        assert_eq!(runtime_for_id("Qwen/Qwen3-0.6B"), "basert");
    }

    #[test]
    fn the_default_socket_fits_a_unix_socket_path() {
        assert_eq!(default_socket(Path::new("/home/u/.superfluid"), 104), Path::new("/home/u/.superfluid/superfluid.sock"));
        let long = PathBuf::from(format!("/{}", "h".repeat(120)));
        let s = default_socket(&long, 104);
        assert!(s.as_os_str().len() < 104 && s.file_name().unwrap().to_string_lossy().starts_with("superfluid-"), "{}", s.display());
    }

    #[test]
    fn serve_help_names_every_flag_in_its_table() {
        let help = serve_help();
        for name in SERVE.iter().flat_map(|g| g.flags.iter().map(|fl| fl.name)) {
            assert!(help.contains(name), "{name}");
        }
        for line in help.lines() {
            assert!(line.chars().count() <= WIDTH, "a help line runs on: {line}");
        }
    }

    /// Every flag the `serve` parser matches is in the table (or an alias or a
    /// retired flag), so `serve --help`, `--flag=value` and the suggestions know it.
    #[test]
    fn every_flag_serve_reads_is_known_here() {
        let src = include_str!("serve.rs");
        let start = src.find("SERVE-FLAGS-BEGIN").expect("marker");
        let end = src.find("SERVE-FLAGS-END").expect("marker");
        let body = &src[start..end];
        let mut seen = 0;
        for piece in body.split('"').skip(1).step_by(2) {
            let long = piece.starts_with("--") && piece.len() > 2 && piece != "--pull-";
            let short = piece.len() == 2 && piece.starts_with('-') && piece.as_bytes()[1].is_ascii_alphabetic();
            if (long || short) && !piece.contains(' ') {
                seen += 1;
                assert!(serve_takes_value(piece).is_some(), "serve reads {piece}, which cli.rs does not list");
            }
        }
        assert!(seen > 100, "the scan found {seen} flags");
    }
}
