//! `superfluid launch <agent>`: run a coding agent pointed at a local server, starting one when
//! none answers. Each agent is a row of [`AGENTS`] whose [`Plan`] is what would run.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::cli::DEFAULT_HTTP;

/// The server an agent is pointed at.
#[derive(Debug, Clone)]
pub struct Target {
    /// `http://host:port`, no trailing slash.
    pub base: String,
    /// The served model id the agent asks for.
    pub model: String,
    /// The server's context window per conversation, when it says.
    pub n_ctx: Option<u64>,
    /// The key the server wants, when it runs behind `--api-key`.
    pub api_key: Option<String>,
}

impl Target {
    /// What the agent sends as its API key: the server's, or a placeholder for a keyless server
    /// (agents refuse to start without one).
    fn key(&self) -> &str {
        self.api_key.as_deref().unwrap_or("superfluid")
    }
}

/// What `launch` runs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Plan {
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    /// Variables set only where the user's environment does not already set them.
    pub env_default: Vec<(String, String)>,
    /// Variables removed from the agent's environment.
    pub env_remove: Vec<String>,
    /// Commands of the agent's own run before it starts, each to success (setup it does itself).
    pub before: Vec<Vec<String>>,
    /// Files written before the agent starts (config the agent reads by path), in launch's own
    /// directory: launch never edits an agent's own config.
    pub files: Vec<(PathBuf, String)>,
    /// Lines printed before the agent starts.
    pub notes: Vec<String>,
}

pub struct Agent {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub title: &'static str,
    pub binary: &'static str,
    pub install: &'static str,
    plan: fn(&Target, &[String]) -> Plan,
}

pub const AGENTS: &[Agent] = &[
    Agent {
        name: "claude",
        aliases: &["claude-code"],
        title: "Claude Code",
        binary: "claude",
        install: "curl -fsSL https://claude.ai/install.sh | bash",
        plan: claude,
    },
    Agent {
        name: "codex",
        aliases: &["codex-cli"],
        title: "Codex",
        binary: "codex",
        install: "npm install -g @openai/codex",
        plan: codex,
    },
    Agent {
        name: "pi",
        aliases: &["pi-coding-agent"],
        title: "pi",
        binary: "pi",
        install: "npm install -g @earendil-works/pi-coding-agent",
        plan: pi,
    },
    Agent {
        name: "opencode",
        aliases: &[],
        title: "OpenCode",
        binary: "opencode",
        install: "curl -fsSL https://opencode.ai/install | bash",
        plan: opencode,
    },
    Agent {
        name: "hermes",
        aliases: &["hermes-agent"],
        title: "Hermes Agent",
        binary: "hermes",
        install: "curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash",
        plan: hermes,
    },
    Agent {
        name: "cline",
        aliases: &["cline-cli"],
        title: "Cline",
        binary: "cline",
        install: "npm install -g cline",
        plan: cline,
    },
    Agent {
        name: "openclaw",
        aliases: &["clawdbot", "moltbot"],
        title: "OpenClaw",
        binary: "openclaw",
        install: "npm install -g openclaw@latest",
        plan: openclaw,
    },
    Agent {
        name: "ollama",
        aliases: &[],
        title: "Ollama CLI",
        binary: "ollama",
        install: "https://ollama.com/download",
        plan: ollama,
    },
];

/// Claude Code over the Anthropic Messages API; every model tier is the served model.
fn claude(t: &Target, extra: &[String]) -> Plan {
    let mut env = vec![
        ("ANTHROPIC_BASE_URL".to_string(), t.base.clone()),
        ("ANTHROPIC_AUTH_TOKEN".to_string(), t.key().to_string()),
    ];
    for var in [
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ] {
        env.push((var.to_string(), t.model.clone()));
    }
    // Without this a per-session line at the top of the system prompt defeats prefix reuse.
    env.push(("CLAUDE_CODE_ATTRIBUTION_HEADER".to_string(), "0".to_string()));
    let mut notes = Vec::new();
    // Told the server's window, so a smaller one compacts in time.
    if let Some(n) = t.n_ctx {
        env.push(("CLAUDE_CODE_MAX_CONTEXT_TOKENS".to_string(), n.to_string()));
    }
    if let Some(n) = t.n_ctx.filter(|&n| n < 200_000) {
        env.push(("CLAUDE_CODE_AUTO_COMPACT_WINDOW".to_string(), n.to_string()));
        if n < 32_768 {
            notes.push(format!(
                "the server's context window is {n} tokens; Claude Code's first request alone is about 17k. \
                 Restart the server with --max-context 65536 or more."
            ));
        }
    }
    Plan {
        program: "claude".into(),
        args: extra.to_vec(),
        env,
        // Auto mode's permission classifier runs on Anthropic's servers unless told otherwise.
        env_default: vec![("CLAUDE_CODE_AUTO_MODE_SERVER".into(), "0".into())],
        // An API key in the environment would be sent beside the token above.
        env_remove: vec!["ANTHROPIC_API_KEY".into()],
        notes,
        ..Plan::default()
    }
}

/// The window an agent is told when the server does not say.
const FALLBACK_WINDOW: u64 = 128_000;

/// A TOML string (`-c key=value` values are TOML).
fn toml_str(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}

/// Codex over the Responses API: a provider declared with `-c` overrides and a one-model catalog
/// carrying the window. Hosted web search is off; `apply_patch` is sent as a function.
fn codex(t: &Target, extra: &[String]) -> Plan {
    let catalog = state_dir().join("codex-models.json");
    let entry = serde_json::json!({
        "slug": t.model,
        "display_name": t.model,
        "context_window": t.n_ctx.unwrap_or(FALLBACK_WINDOW),
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 0,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "input_modalities": ["text"],
        "base_instructions": "",
        "support_verbosity": false,
        "supports_parallel_tool_calls": false,
        "supports_reasoning_summaries": false,
        "supported_reasoning_levels": [],
        "experimental_supported_tools": [],
    });
    let text = serde_json::to_string_pretty(&serde_json::json!({"models": [entry]})).expect("json") + "\n";
    let mut args = Vec::new();
    for (k, v) in [
        ("model_provider", toml_str("superfluid")),
        ("model_providers.superfluid.name", toml_str("Superfluid")),
        ("model_providers.superfluid.base_url", toml_str(&format!("{}/v1", t.base))),
        ("model_providers.superfluid.wire_api", toml_str("responses")),
        ("model_providers.superfluid.env_key", toml_str("SUPERFLUID_LAUNCH_KEY")),
        ("model_catalog_json", toml_str(&catalog.to_string_lossy())),
        ("web_search", toml_str("disabled")),
    ] {
        args.push("-c".to_string());
        args.push(format!("{k}={v}"));
    }
    args.push("-m".into());
    args.push(t.model.clone());
    args.extend(extra.iter().cloned());
    Plan {
        program: "codex".into(),
        args,
        env: vec![("SUPERFLUID_LAUNCH_KEY".into(), t.key().to_string())],
        files: vec![(catalog, text)],
        ..Plan::default()
    }
}

/// pi over Chat Completions, through a provider an extension loaded for this run registers.
fn pi(t: &Target, extra: &[String]) -> Plan {
    let ext = state_dir().join("pi-superfluid.ts");
    let window = t.n_ctx.unwrap_or(FALLBACK_WINDOW);
    let provider = serde_json::json!({
        "name": "Superfluid",
        "baseUrl": format!("{}/v1", t.base),
        "apiKey": t.key(),
        "api": "openai-completions",
        "models": [{
            "id": t.model,
            "name": t.model,
            "reasoning": false,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": window,
            "maxTokens": window.min(16_384),
        }],
    });
    let text = format!(
        "// Written by `superfluid launch pi` for one run: the server at {}.\n\
         export default function (pi: any) {{\n  pi.registerProvider(\"superfluid\", {});\n}}\n",
        t.base,
        serde_json::to_string_pretty(&provider).expect("json").replace('\n', "\n  ")
    );
    let mut args = vec!["-e".to_string(), ext.to_string_lossy().into_owned()];
    args.extend(["--provider".into(), "superfluid".into(), "--model".into(), t.model.clone()]);
    args.extend(extra.iter().cloned());
    Plan { program: "pi".into(), args, files: vec![(ext, text)], ..Plan::default() }
}

/// OpenCode over Chat Completions, through a provider passed in `OPENCODE_CONFIG_CONTENT`.
fn opencode(t: &Target, extra: &[String]) -> Plan {
    let window = t.n_ctx.unwrap_or(FALLBACK_WINDOW);
    let mut models = serde_json::Map::new();
    models.insert(
        t.model.clone(),
        serde_json::json!({"name": t.model, "limit": {"context": window, "output": window.min(16_384)}}),
    );
    let config = serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {"superfluid": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Superfluid",
            "options": {"baseURL": format!("{}/v1", t.base), "apiKey": t.key()},
            "models": models,
        }},
        "model": format!("superfluid/{}", t.model),
    });
    Plan {
        program: "opencode".into(),
        args: extra.to_vec(),
        env: vec![("OPENCODE_CONFIG_CONTENT".into(), config.to_string())],
        ..Plan::default()
    }
}

/// Hermes over Chat Completions, through its `custom` provider at `CUSTOM_BASE_URL`; a key is
/// sent only beside an `OPENAI_BASE_URL` naming this server.
fn hermes(t: &Target, extra: &[String]) -> Plan {
    let v1 = format!("{}/v1", t.base);
    let mut env = vec![("CUSTOM_BASE_URL".to_string(), v1.clone())];
    if let Some(key) = &t.api_key {
        env.push(("OPENAI_BASE_URL".into(), v1));
        env.push(("OPENAI_API_KEY".into(), key.clone()));
    }
    let mut notes = Vec::new();
    if let Some(n) = t.n_ctx.filter(|&n| n < 64_000) {
        notes.push(format!(
            "the server's context window is {n} tokens; Hermes refuses a model with less than 64000. \
             Restart the server with --max-context 65536 or more."
        ));
    }
    let mut args = vec!["--provider".to_string(), "custom".into(), "-m".into(), t.model.clone()];
    args.extend(extra.iter().cloned());
    Plan { program: "hermes".into(), args, env, notes, ..Plan::default() }
}

/// Cline over Chat Completions, through a provider settings file of launch's own named by
/// `CLINE_PROVIDER_SETTINGS_PATH`. It needs an `updatedAt`, and no `-P`/`-m` (saved over the file).
fn cline(t: &Target, extra: &[String]) -> Plan {
    // RFC 3339 to the millisecond, as Cline writes it.
    let mut updated = crate::ollama::timestamp(std::time::SystemTime::now());
    updated.truncate(23);
    updated.push('Z');
    let path = state_dir().join("cline-providers.json");
    let settings = serde_json::json!({
        "version": 1,
        "lastUsedProvider": "openai-compatible",
        "modes": {},
        "providers": {"openai-compatible": {
            "settings": {
                "provider": "openai-compatible",
                "apiKey": t.key(),
                "model": t.model,
                "baseUrl": format!("{}/v1", t.base),
            },
            "updatedAt": updated,
            "tokenSource": "manual",
        }},
    });
    let text = serde_json::to_string_pretty(&settings).expect("json") + "\n";
    Plan {
        program: "cline".into(),
        args: extra.to_vec(),
        env: vec![("CLINE_PROVIDER_SETTINGS_PATH".into(), path.to_string_lossy().into_owned())],
        files: vec![(path, text)],
        ..Plan::default()
    }
}

/// The OpenClaw profile launch keeps, apart from the user's own.
const OPENCLAW_PROFILE: &str = "superfluid";

/// OpenClaw over Chat Completions: the provider and default model are set in launch's own profile
/// with `config set`, then `tui --local` opens (or the arguments after `--`).
fn openclaw(t: &Target, extra: &[String]) -> Plan {
    let window = t.n_ctx.unwrap_or(FALLBACK_WINDOW);
    let provider = serde_json::json!({
        "baseUrl": format!("{}/v1", t.base),
        "apiKey": t.key(),
        "api": "openai-completions",
        "models": [{
            "id": t.model,
            "name": t.model,
            "input": ["text"],
            "reasoning": false,
            "contextWindow": window,
            "maxTokens": window.min(16_384),
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }],
    });
    let profile = |rest: &[&str]| -> Vec<String> {
        ["--profile", OPENCLAW_PROFILE].iter().chain(rest).map(|s| s.to_string()).collect()
    };
    let mut before = Vec::new();
    let config = std::env::home_dir().map(|h| h.join(format!(".openclaw-{OPENCLAW_PROFILE}")).join("openclaw.json"));
    if !config.is_some_and(|c| c.exists()) {
        before.push(profile(&["setup", "--baseline"]));
    }
    before.push(profile(&["config", "set", "models.providers.superfluid", &provider.to_string(), "--strict-json"]));
    before.push(profile(&["config", "set", "agents.defaults.model.primary", &format!("superfluid/{}", t.model)]));
    let mut args = profile(&[]);
    if extra.is_empty() {
        args.extend(["tui".to_string(), "--local".to_string()]);
    } else {
        args.extend(extra.iter().cloned());
    }
    Plan { program: "openclaw".into(), args, before, ..Plan::default() }
}

/// The `ollama` command line with `OLLAMA_HOST` set to the server: `run <model>`, or the
/// arguments after `--`.
fn ollama(t: &Target, extra: &[String]) -> Plan {
    let host = t.base.trim_start_matches("http://").to_string();
    let args = if extra.is_empty() { vec!["run".to_string(), t.model.clone()] } else { extra.to_vec() };
    Plan { program: "ollama".into(), args, env: vec![("OLLAMA_HOST".into(), host)], ..Plan::default() }
}

pub fn find(name: &str) -> Option<&'static Agent> {
    AGENTS.iter().find(|a| a.name == name || a.aliases.contains(&name))
}

pub const HELP: &str = "Run a coding agent against a local model.

Usage: superfluid launch <agent> [options] [-- <agent arguments>...]
       superfluid launch            List the agents and whether each is installed

  Agents: claude, codex, pi, opencode, hermes, cline, openclaw, ollama (its CLI's chat).

  Points the agent at the server on --http. When none answers there, starts
  `superfluid serve <model>` in the background first (its log goes to
  $SUPERFLUID_HOME/logs/serve.log) and leaves it running; `superfluid stop` stops it.
  The model is --model, or the one launch last started a server with.

Options:
  --model <path|org/model[:tag]>   The model to serve, and the one the agent uses on a server
                                   that serves several [default: the server's first]
  --http <addr:port>               The server [default: $SUPERFLUID_HTTP, else 127.0.0.1:8453]
  --api-key <key>                  The server's key, for one behind --api-key
                                   [default: $SUPERFLUID_API_KEY]
  --print                          Print what would run instead of running it

Examples:
  superfluid launch claude --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
  superfluid launch claude -- --continue
";

pub const STOP_HELP: &str = "Stop the server `superfluid launch` started.

Usage: superfluid stop

  Sends it SIGTERM and waits for it to finish what is in flight. A server you started
  yourself with `superfluid serve` is yours to stop.
";

#[derive(Debug, Default, PartialEq)]
struct Args {
    agent: Option<String>,
    model: Option<String>,
    http: Option<String>,
    api_key: Option<String>,
    print: bool,
    extra: Vec<String>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut out = Args::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (a.as_str(), None),
        };
        let mut value = |flag: &str| inline.clone().or_else(|| it.next().cloned()).ok_or(format!("{flag} needs a value"));
        match flag {
            "--" => {
                out.extra = it.cloned().collect();
                break;
            }
            "--model" => out.model = Some(value(flag)?),
            "--http" => out.http = Some(value(flag)?),
            "--api-key" => out.api_key = Some(value(flag)?),
            "--print" => out.print = true,
            f if f.starts_with('-') => {
                let known = ["--model", "--http", "--api-key", "--print"];
                return Err(match crate::cli::suggest(f, known) {
                    Some(s) => format!("unknown flag {f} (did you mean {s}?); pass the agent's own flags after --"),
                    None => format!("unknown flag {f}; pass the agent's own flags after --"),
                });
            }
            name if out.agent.is_none() => out.agent = Some(name.to_string()),
            other => return Err(format!("unexpected argument {other}; pass the agent's own arguments after --")),
        }
    }
    Ok(out)
}

/// A served model, as `/v1/models` lists it.
#[derive(Debug, Clone, PartialEq)]
struct Served {
    id: String,
    n_ctx: Option<u64>,
}

enum Probe {
    Up(Vec<Served>),
    Unauthorized,
    Down,
}

fn probe(base: &str, api_key: Option<&str>) -> Probe {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(3)))
        .http_status_as_error(false)
        .build()
        .new_agent();
    let mut req = agent.get(format!("{base}/v1/models"));
    if let Some(key) = api_key {
        req = req.header("authorization", &format!("Bearer {key}"));
    }
    let Ok(mut resp) = req.call() else { return Probe::Down };
    match resp.status().as_u16() {
        200 => {}
        401 | 403 => return Probe::Unauthorized,
        _ => return Probe::Down,
    }
    let Ok(body) = resp.body_mut().read_to_string() else { return Probe::Down };
    match parse_models(&body) {
        Some(models) => Probe::Up(models),
        None => Probe::Down,
    }
}

fn parse_models(body: &str) -> Option<Vec<Served>> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let data = v.get("data")?.as_array()?;
    Some(
        data.iter()
            .filter_map(|m| {
                Some(Served {
                    id: m.get("id")?.as_str()?.to_string(),
                    n_ctx: m.pointer("/meta/n_ctx").and_then(serde_json::Value::as_u64),
                })
            })
            .collect(),
    )
}

/// The served model `wanted` names, by id or by its file or repo name; with nothing wanted, the
/// server's first.
fn pick<'a>(served: &'a [Served], wanted: Option<&str>) -> Option<&'a Served> {
    let Some(wanted) = wanted else { return served.first() };
    if let Some(m) = served.iter().find(|m| m.id == wanted) {
        return Some(m);
    }
    let stem = |s: &str| {
        let s = s.split_once(':').map_or(s, |(s, _)| s);
        let s = s.rsplit('/').next().unwrap_or(s);
        s.strip_suffix(".gguf").unwrap_or(s).to_ascii_lowercase()
    };
    let w = stem(wanted);
    served.iter().find(|m| stem(&m.id) == w)
}

/// The id `serve <model>` lists the model under.
fn served_id(model: &str) -> String {
    if crate::pulls::looks_like_id(model) {
        crate::pulls::served_name(model)
    } else {
        crate::registry::model_id_for(Path::new(model))
    }
}

fn on_path(binary: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(binary)).find(|p| {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    })
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("superfluid launch: {msg}");
    std::process::exit(1);
}

/// `$SUPERFLUID_HOME/launch/`: the model launch last started a server with, and that server.
fn state_dir() -> PathBuf {
    crate::runtimes::default_home().join("launch")
}

/// The server `launch` started: its pid and address, one per line.
fn server_file() -> PathBuf {
    state_dir().join("server")
}

fn read_server() -> Option<(i32, String)> {
    let text = std::fs::read_to_string(server_file()).ok()?;
    let mut lines = text.lines();
    let pid = lines.next()?.trim().parse().ok()?;
    let addr = lines.next()?.trim().to_string();
    alive(pid).then_some((pid, addr))
}

fn alive(pid: i32) -> bool {
    // SAFETY: signal 0 checks for the process without sending anything.
    pid > 0 && unsafe { libc::kill(pid, 0) } == 0
}

pub fn run(args: &[String]) {
    let a = parse(args).unwrap_or_else(|e| {
        eprintln!("superfluid launch: {e}\n\n{HELP}");
        std::process::exit(2);
    });
    let Some(name) = a.agent.as_deref() else {
        list();
        return;
    };
    let agent = find(name).unwrap_or_else(|| {
        let names = AGENTS.iter().map(|a| a.name);
        match crate::cli::suggest(name, names) {
            Some(s) => fail(format!("no agent called {name} (did you mean {s}?); `superfluid launch` lists them")),
            None => fail(format!("no agent called {name}; `superfluid launch` lists them")),
        }
    });
    if !a.print && on_path(agent.binary).is_none() {
        fail(format!("{} is not installed ({} not on PATH). Install it with:\n  {}", agent.title, agent.binary, agent.install));
    }
    let addr = a
        .http
        .clone()
        .or_else(|| std::env::var("SUPERFLUID_HTTP").ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| DEFAULT_HTTP.to_string());
    let addr = addr.trim_start_matches("http://").trim_end_matches('/').to_string();
    let base = format!("http://{addr}");
    let api_key = a.api_key.clone().or_else(|| std::env::var("SUPERFLUID_API_KEY").ok().filter(|s| !s.is_empty()));
    let mut started = false;
    let served = match probe(&base, api_key.as_deref()) {
        Probe::Up(served) => served,
        Probe::Unauthorized => fail(format!("the server at {addr} wants a key: pass --api-key or set SUPERFLUID_API_KEY")),
        Probe::Down if a.print => {
            let id = a.model.clone().or_else(remembered).map_or_else(|| "<model>".into(), |m| served_id(&m));
            vec![Served { id, n_ctx: None }]
        }
        Probe::Down => {
            let model = a.model.clone().or_else(remembered).unwrap_or_else(|| {
                fail(format!(
                    "no server at {addr}, and no model to start one with. Pass one:\n  \
                     superfluid launch {} --model <path|org/model[:tag]>",
                    agent.name
                ))
            });
            started = true;
            start_server(&model, &addr, api_key.as_deref())
        }
    };
    // A server launch started serves the model it was started with, under whatever id.
    let wanted = if started { None } else { a.model.as_deref() };
    let model = pick(&served, wanted).unwrap_or_else(|| {
        let ids: Vec<&str> = served.iter().map(|m| m.id.as_str()).collect();
        fail(format!(
            "the server at {addr} does not serve {}; it serves: {}",
            a.model.as_deref().unwrap_or("any model"),
            if ids.is_empty() { "nothing".to_string() } else { ids.join(", ") }
        ))
    });
    let target = Target { base, model: model.id.clone(), n_ctx: model.n_ctx, api_key };
    let plan = (agent.plan)(&target, &a.extra);
    if a.print {
        print!("{}", render(&plan));
        return;
    }
    for (path, text) in &plan.files {
        if let Err(e) = path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(path, text)) {
            fail(format!("could not write {}: {e}", path.display()));
        }
    }
    for args in &plan.before {
        let out = std::process::Command::new(&plan.program)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap_or_else(|e| fail(format!("could not run {}: {e}", plan.program)));
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let tail: Vec<&str> = err.lines().rev().take(10).collect();
            fail(format!(
                "`{} {}` failed ({}):\n{}",
                plan.program,
                args.join(" "),
                out.status,
                tail.into_iter().rev().collect::<Vec<_>>().join("\n")
            ));
        }
    }
    for note in &plan.notes {
        eprintln!("superfluid launch: {note}");
    }
    eprintln!("superfluid launch: {} on {} at {}", agent.title, target.model, target.base);
    exec(&plan);
}

fn list() {
    println!("Agents (superfluid launch <agent>):\n");
    for a in AGENTS {
        let state = if on_path(a.binary).is_some() { "installed".to_string() } else { format!("install: {}", a.install) };
        println!("  {:<10} {:<14} {state}", a.name, a.title);
    }
    match read_server() {
        Some((pid, addr)) => println!("\nServer started by launch: {addr} (pid {pid}); `superfluid stop` stops it."),
        None => {
            if let Some(m) = remembered() {
                println!("\nWith no server running, launch starts one with {m}.");
            }
        }
    }
}

fn remembered() -> Option<String> {
    std::fs::read_to_string(state_dir().join("model")).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Starts `superfluid serve <model>` in its own session, so it outlives the agent and the
/// terminal, and waits until it lists its models.
fn start_server(model: &str, addr: &str, api_key: Option<&str>) -> Vec<Served> {
    use std::os::unix::process::CommandExt;
    let exe = std::env::current_exe().unwrap_or_else(|e| fail(format!("could not find this program: {e}")));
    let logs = crate::runtimes::default_home().join("logs");
    let log = logs.join("serve.log");
    if let Err(e) = std::fs::create_dir_all(&logs).and_then(|()| std::fs::create_dir_all(state_dir())) {
        fail(format!("could not create {}: {e}", logs.display()));
    }
    let start_len = std::fs::metadata(&log).map(|m| m.len()).unwrap_or(0);
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("serve").arg(model).args(["--http", addr, "--no-tui", "--log-file"]).arg(&log);
    if let Some(key) = api_key {
        cmd.args(["--api-key", key]);
    }
    let stdout = std::fs::OpenOptions::new().create(true).append(true).open(&log);
    cmd.stdin(std::process::Stdio::null())
        .stdout(stdout.map(std::process::Stdio::from).unwrap_or_else(|_| std::process::Stdio::null()))
        .stderr(std::process::Stdio::null());
    // SAFETY: setsid is async-signal-safe and touches only the child.
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let mut child = cmd.spawn().unwrap_or_else(|e| fail(format!("could not start the server: {e}")));
    let pid = child.id() as i32;
    let _ = std::fs::write(server_file(), format!("{pid}\n{addr}\n"));
    let _ = std::fs::write(state_dir().join("model"), format!("{model}\n"));
    eprintln!("superfluid launch: starting `superfluid serve {model}` on {addr} (log: {})", log.display());
    let base = format!("http://{addr}");
    let started = Instant::now();
    let tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            if tty {
                eprint!("\r\x1b[2K");
            }
            let _ = std::fs::remove_file(server_file());
            fail(format!("the server exited ({status}) before it was ready:\n{}", tail(&log, start_len, 15)));
        }
        if let Probe::Up(served) = probe(&base, api_key) {
            if tty {
                eprint!("\r\x1b[2K");
            }
            eprintln!("superfluid launch: server ready in {:.0} s", started.elapsed().as_secs_f64());
            return served;
        }
        if tty {
            let last = tail(&log, start_len, 1);
            let last: String = last.trim().chars().take(90).collect();
            eprint!("\r\x1b[2K  {:>4.0} s  {last}", started.elapsed().as_secs_f64());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// The last `n` lines `log` gained past byte `from`.
fn tail(log: &Path, from: u64, n: usize) -> String {
    let Ok(bytes) = std::fs::read(log) else { return String::new() };
    let text = String::from_utf8_lossy(bytes.get(from as usize..).unwrap_or_default());
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

fn exec(plan: &Plan) -> ! {
    use std::os::unix::process::CommandExt;
    let mut cmd = std::process::Command::new(&plan.program);
    cmd.args(&plan.args);
    for k in &plan.env_remove {
        cmd.env_remove(k);
    }
    cmd.envs(plan.env.iter().map(|(k, v)| (k, v)));
    for (k, v) in &plan.env_default {
        if std::env::var_os(k).is_none() {
            cmd.env(k, v);
        }
    }
    let err = cmd.exec();
    fail(format!("could not run {}: {err}", plan.program))
}

/// The plan as a shell command line, for `--print`.
pub fn render(plan: &Plan) -> String {
    let mut out = String::new();
    for k in &plan.env_remove {
        out.push_str(&format!("unset {k}\n"));
    }
    for args in &plan.before {
        let line: Vec<String> = std::iter::once(&plan.program).chain(args).map(|a| quote(a)).collect();
        out.push_str(&line.join(" "));
        out.push('\n');
    }
    for (path, text) in &plan.files {
        out.push_str(&format!("# {}:\n{}", path.display(), text.lines().map(|l| format!("#   {l}\n")).collect::<String>()));
    }
    for (k, v) in &plan.env_default {
        out.push_str(&format!("export {k}=\"${{{k}:-{v}}}\"\n"));
    }
    let mut line: Vec<String> = plan.env.iter().map(|(k, v)| format!("{k}={}", quote(v))).collect();
    line.push(quote(&plan.program));
    line.extend(plan.args.iter().map(|a| quote(a)));
    out.push_str(&line.join(" \\\n  "));
    out.push('\n');
    out
}

fn quote(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./:=@,+%".contains(c)) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

pub fn stop(args: &[String]) {
    if let Some(a) = args.first() {
        eprintln!("superfluid stop: unexpected argument {a}\n\n{STOP_HELP}");
        std::process::exit(2);
    }
    let Some((pid, addr)) = read_server() else {
        let _ = std::fs::remove_file(server_file());
        eprintln!("superfluid stop: no server started by `superfluid launch` is running");
        std::process::exit(1);
    };
    // SAFETY: pid is a live process launch started (read_server checked it).
    unsafe { libc::kill(pid, libc::SIGTERM) };
    let started = Instant::now();
    while alive(pid) && started.elapsed() < Duration::from_secs(90) {
        std::thread::sleep(Duration::from_millis(200));
    }
    if alive(pid) {
        eprintln!("superfluid stop: the server on {addr} (pid {pid}) is still finishing; it got SIGTERM");
        std::process::exit(1);
    }
    let _ = std::fs::remove_file(server_file());
    eprintln!("superfluid stop: stopped the server on {addr}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn target(n_ctx: Option<u64>) -> Target {
        Target { base: "http://127.0.0.1:8453".into(), model: "Qwen3.6-27B-Q4_K_M".into(), n_ctx, api_key: None }
    }

    fn env<'a>(p: &'a Plan, k: &str) -> Option<&'a str> {
        p.env.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
    }

    #[test]
    fn flags_go_to_launch_and_everything_after_the_dashes_to_the_agent() {
        let a = parse(&strings(&["claude", "--model=m.gguf", "--http", "127.0.0.1:9000", "--", "--continue", "-p", "hi"])).unwrap();
        assert_eq!(a.agent.as_deref(), Some("claude"));
        assert_eq!(a.model.as_deref(), Some("m.gguf"));
        assert_eq!(a.http.as_deref(), Some("127.0.0.1:9000"));
        assert_eq!(a.extra, strings(&["--continue", "-p", "hi"]));
        assert!(parse(&strings(&["claude", "--continue"])).unwrap_err().contains("after --"));
        assert!(parse(&strings(&["claude", "--modle", "x"])).unwrap_err().contains("--model"));
        assert!(parse(&strings(&["claude", "extra"])).is_err());
        assert_eq!(parse(&[]).unwrap(), Args::default());
    }

    #[test]
    fn claude_code_is_pointed_at_the_server_for_every_model_tier() {
        let p = claude(&target(Some(262_144)), &strings(&["--continue"]));
        assert_eq!(p.program, "claude");
        assert_eq!(p.args, strings(&["--continue"]));
        assert_eq!(env(&p, "ANTHROPIC_BASE_URL"), Some("http://127.0.0.1:8453"));
        assert_eq!(env(&p, "ANTHROPIC_AUTH_TOKEN"), Some("superfluid"));
        for k in ["ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"] {
            assert_eq!(env(&p, k), Some("Qwen3.6-27B-Q4_K_M"), "{k}");
        }
        assert_eq!(env(&p, "CLAUDE_CODE_AUTO_COMPACT_WINDOW"), None, "a window past 200k needs no cap");
        assert_eq!(env(&p, "CLAUDE_CODE_MAX_CONTEXT_TOKENS"), Some("262144"));
        assert_eq!(env(&p, "CLAUDE_CODE_ATTRIBUTION_HEADER"), Some("0"));
        assert_eq!(p.env_default, vec![("CLAUDE_CODE_AUTO_MODE_SERVER".to_string(), "0".to_string())]);
        assert_eq!(p.env_remove, strings(&["ANTHROPIC_API_KEY"]));
        assert!(p.notes.is_empty());
    }

    #[test]
    fn a_small_window_makes_claude_code_compact_sooner_and_a_tiny_one_is_called_out() {
        let p = claude(&target(Some(65_536)), &[]);
        assert_eq!(env(&p, "CLAUDE_CODE_AUTO_COMPACT_WINDOW"), Some("65536"));
        assert!(p.notes.is_empty());
        let p = claude(&target(Some(8192)), &[]);
        assert!(p.notes[0].contains("--max-context"), "{:?}", p.notes);
        let p = claude(&Target { api_key: Some("k1".into()), ..target(None) }, &[]);
        assert_eq!(env(&p, "ANTHROPIC_AUTH_TOKEN"), Some("k1"));
    }

    #[test]
    fn codex_gets_a_provider_of_its_own_and_a_catalog_with_the_window() {
        let p = codex(&target(Some(131_072)), &strings(&["exec", "hi"]));
        assert_eq!(p.program, "codex");
        let joined = p.args.join(" ");
        assert!(joined.contains(r#"model_provider="superfluid""#), "{joined}");
        assert!(joined.contains(r#"model_providers.superfluid.base_url="http://127.0.0.1:8453/v1""#), "{joined}");
        assert!(joined.contains(r#"model_providers.superfluid.wire_api="responses""#), "{joined}");
        assert!(joined.contains(r#"web_search="disabled""#), "{joined}");
        assert!(joined.ends_with("-m Qwen3.6-27B-Q4_K_M exec hi"), "{joined}");
        assert_eq!(env(&p, "SUPERFLUID_LAUNCH_KEY"), Some("superfluid"));
        let (path, text) = &p.files[0];
        assert!(joined.contains(&path.to_string_lossy().to_string()));
        let catalog: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(catalog["models"][0]["slug"], "Qwen3.6-27B-Q4_K_M");
        assert_eq!(catalog["models"][0]["context_window"], 131_072);
        assert!(catalog["models"][0].get("apply_patch_tool_type").is_none());
    }

    #[test]
    fn pi_loads_a_one_run_extension_that_registers_the_server() {
        let p = pi(&target(Some(262_144)), &strings(&["-p", "hi"]));
        let (path, text) = &p.files[0];
        assert_eq!(p.args[..2], [String::from("-e"), path.to_string_lossy().into_owned()]);
        assert_eq!(p.args[2..], strings(&["--provider", "superfluid", "--model", "Qwen3.6-27B-Q4_K_M", "-p", "hi"]));
        assert!(text.contains(r#"pi.registerProvider("superfluid", {"#), "{text}");
        assert!(text.contains(r#""baseUrl": "http://127.0.0.1:8453/v1""#), "{text}");
        assert!(text.contains(r#""contextWindow": 262144"#) && text.contains(r#""maxTokens": 16384"#), "{text}");
    }

    #[test]
    fn opencode_gets_a_provider_in_its_per_run_config() {
        let p = opencode(&target(Some(65_536)), &strings(&["run", "hi"]));
        assert_eq!(p.args, strings(&["run", "hi"]));
        let config: serde_json::Value = serde_json::from_str(env(&p, "OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(config["model"], "superfluid/Qwen3.6-27B-Q4_K_M");
        let provider = &config["provider"]["superfluid"];
        assert_eq!(provider["npm"], "@ai-sdk/openai-compatible");
        assert_eq!(provider["options"]["baseURL"], "http://127.0.0.1:8453/v1");
        assert_eq!(provider["models"]["Qwen3.6-27B-Q4_K_M"]["limit"]["context"], 65_536);
    }

    #[test]
    fn hermes_uses_its_custom_provider_and_a_key_only_for_this_host() {
        let p = hermes(&target(Some(262_144)), &strings(&["-z", "hi"]));
        assert_eq!(p.args, strings(&["--provider", "custom", "-m", "Qwen3.6-27B-Q4_K_M", "-z", "hi"]));
        assert_eq!(env(&p, "CUSTOM_BASE_URL"), Some("http://127.0.0.1:8453/v1"));
        assert_eq!(env(&p, "OPENAI_API_KEY"), None);
        let p = hermes(&Target { api_key: Some("k1".into()), ..target(Some(32_768)) }, &[]);
        assert_eq!((env(&p, "OPENAI_BASE_URL"), env(&p, "OPENAI_API_KEY")), (Some("http://127.0.0.1:8453/v1"), Some("k1")));
        assert!(p.notes[0].contains("64000"), "{:?}", p.notes);
    }

    #[test]
    fn cline_reads_a_provider_settings_file_of_launchs_own() {
        let p = cline(&target(None), &strings(&["-y", "hi"]));
        assert_eq!(p.args, strings(&["-y", "hi"]));
        let (path, text) = &p.files[0];
        assert_eq!(env(&p, "CLINE_PROVIDER_SETTINGS_PATH"), Some(path.to_string_lossy().as_ref()));
        let v: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(v["lastUsedProvider"], "openai-compatible");
        let s = &v["providers"]["openai-compatible"]["settings"];
        assert_eq!((s["model"].as_str(), s["baseUrl"].as_str()), (Some("Qwen3.6-27B-Q4_K_M"), Some("http://127.0.0.1:8453/v1")));
        let updated = v["providers"]["openai-compatible"]["updatedAt"].as_str().unwrap_or_default();
        assert!(updated.len() == 24 && updated.ends_with('Z') && updated.as_bytes()[19] == b'.', "{updated}");
    }

    #[test]
    fn openclaw_is_set_up_in_a_profile_of_its_own_and_opens_the_local_terminal_ui() {
        let p = openclaw(&target(Some(262_144)), &[]);
        assert_eq!(p.args, strings(&["--profile", "superfluid", "tui", "--local"]));
        let sets: Vec<&Vec<String>> = p.before.iter().filter(|b| b.get(2).map(String::as_str) == Some("config")).collect();
        assert_eq!(sets.len(), 2);
        assert!(p.before.iter().all(|b| b[..2] == strings(&["--profile", "superfluid"])));
        let provider: serde_json::Value = serde_json::from_str(&sets[0][5]).unwrap();
        assert_eq!(sets[0][4], "models.providers.superfluid");
        assert_eq!(provider["baseUrl"], "http://127.0.0.1:8453/v1");
        assert_eq!(provider["models"][0]["contextWindow"], 262_144);
        assert_eq!(sets[1][4..], strings(&["agents.defaults.model.primary", "superfluid/Qwen3.6-27B-Q4_K_M"]));
        let p = openclaw(&target(None), &strings(&["agent", "--local", "-m", "hi"]));
        assert_eq!(p.args, strings(&["--profile", "superfluid", "agent", "--local", "-m", "hi"]));
    }

    #[test]
    fn the_ollama_cli_is_pointed_at_the_server_and_chats_with_the_model() {
        let p = ollama(&target(None), &[]);
        assert_eq!(p.args, strings(&["run", "Qwen3.6-27B-Q4_K_M"]));
        assert_eq!(env(&p, "OLLAMA_HOST"), Some("127.0.0.1:8453"));
        assert_eq!(ollama(&target(None), &strings(&["ps"])).args, strings(&["ps"]));
    }

    #[test]
    fn a_model_is_found_by_id_or_by_the_name_it_was_served_from() {
        let served = vec![
            Served { id: "Qwen3.6-27B-Q4_K_M".into(), n_ctx: Some(262_144) },
            Served { id: "gemma-4-E4B".into(), n_ctx: None },
        ];
        assert_eq!(pick(&served, None).unwrap().id, "Qwen3.6-27B-Q4_K_M");
        assert_eq!(pick(&served, Some("gemma-4-E4B")).unwrap().id, "gemma-4-E4B");
        assert_eq!(pick(&served, Some("./models/Qwen3.6-27B-Q4_K_M.gguf")).unwrap().id, "Qwen3.6-27B-Q4_K_M");
        assert_eq!(pick(&served, Some("google/gemma-4-E4B")).unwrap().id, "gemma-4-E4B");
        assert_eq!(pick(&[Served { id: "unsloth/Qwen3.8-27B-GGUF".into(), n_ctx: None }], Some("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M")).unwrap().id, "unsloth/Qwen3.8-27B-GGUF");
        assert!(pick(&served, Some("llama-3")).is_none());
        assert!(pick(&served, Some("Qwen3.6")).is_none(), "a prefix is not a name");
        assert!(pick(&[], None).is_none());
    }

    #[test]
    fn models_and_their_windows_are_read_from_the_model_list() {
        let body = r#"{"object":"list","data":[{"id":"a","meta":{"n_ctx":131072}},{"id":"b","meta":{}}]}"#;
        assert_eq!(
            parse_models(body).unwrap(),
            vec![Served { id: "a".into(), n_ctx: Some(131_072) }, Served { id: "b".into(), n_ctx: None }]
        );
        assert!(parse_models("<html>").is_none());
    }

    #[test]
    fn print_renders_a_command_a_shell_runs_as_is() {
        let p = Plan {
            program: "claude".into(),
            args: strings(&["-p", "it's here"]),
            env: vec![("A".into(), "http://x:1".into()), ("B".into(), "two words".into())],
            env_remove: strings(&["K"]),
            ..Plan::default()
        };
        assert_eq!(render(&p), "unset K\nA=http://x:1 \\\n  B='two words' \\\n  claude \\\n  -p \\\n  'it'\\''s here'\n");
    }

    #[test]
    fn every_agent_is_found_by_name_and_alias() {
        for a in AGENTS {
            assert!(find(a.name).is_some_and(|f| f.name == a.name));
            for al in a.aliases {
                assert!(find(al).is_some_and(|f| f.name == a.name));
            }
        }
        assert!(find("nope").is_none());
    }
}
