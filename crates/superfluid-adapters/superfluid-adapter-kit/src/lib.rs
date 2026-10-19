//! What every runtime adapter is built from.

use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::Arc;

use superfluid_engine::Engine;
use superfluid_executor::{Executor, ExecutorConfig, RuntimePrimitives};
use superfluid_proto::linkw::RingSpec;
use superfluid_worker::{WorkerConfig, WorkerError, WorkerServer};

pub use superfluid_engine::artifact::{valid_id, Device, Format, Rule, RuntimeInfo, TokenizerSource};
pub use superfluid_engine::export_tokenizer;
pub use superfluid_engine::Tokenizer;
pub use superfluid_executor::sizing::{self, Sizing};
pub use superfluid_worker::check::{ModelCheck, OptionDecl, Report};
pub use superfluid_worker::{ring_specs_for, WorkerArgs};

pub mod conformance;
pub mod install;

use install::{Candidate, Host, Manifest, Plan, Request};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PullOption {
    pub name: &'static str,
    pub value: bool,
    pub about: &'static str,
}

/// An option of one runtime, given to `superfluid serve` as `--<runtime>-<name> <value>` and to the worker as
/// `--<name> <value>`; a knob every runtime could take is a flag of `serve` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServeOption {
    pub name: &'static str,
    pub kind: OptionKind,
    /// What the runtime does without it, as words for the help ("" when there is nothing to say).
    pub default: &'static str,
    pub about: &'static str,
}

/// What a value of an option may be; the daemon and the worker both refuse another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionKind {
    Int,
    /// An integer at or above this.
    IntFrom(i64),
    /// An integer from the first to the second, both taken: a bound past which the runtime would fail or clamp.
    IntIn(i64, i64),
    Float,
    /// A number at or above this.
    FloatFrom(i64),
    OnOff,
    Choice(&'static [&'static str]),
    Text,
}

impl OptionKind {
    /// The kind as a report spells it.
    pub fn name(self) -> &'static str {
        match self {
            OptionKind::Int | OptionKind::IntFrom(_) | OptionKind::IntIn(..) => "int",
            OptionKind::Float | OptionKind::FloatFrom(_) => "float",
            OptionKind::OnOff => "on-off",
            OptionKind::Choice(_) => "choice",
            OptionKind::Text => "text",
        }
    }

    pub fn choices(self) -> &'static [&'static str] {
        match self {
            OptionKind::Choice(c) => c,
            _ => &[],
        }
    }

    pub fn least(self) -> Option<i64> {
        match self {
            OptionKind::IntFrom(n) | OptionKind::IntIn(n, _) | OptionKind::FloatFrom(n) => Some(n),
            _ => None,
        }
    }

    pub fn most(self) -> Option<i64> {
        match self {
            OptionKind::IntIn(_, n) => Some(n),
            _ => None,
        }
    }

    pub fn check(self, value: &str) -> Result<(), String> {
        check_value(self.name(), self.choices(), self.least(), self.most(), value)
    }
}

/// Whether `value` is one an option of `kind` takes, with `kind`, `choices`, `least` and `most` as a report spells them.
pub fn check_value(kind: &str, choices: &[impl AsRef<str>], least: Option<i64>, most: Option<i64>, value: &str) -> Result<(), String> {
    let list = || choices.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    let within = |n: f64| least.is_none_or(|l| n >= l as f64) && most.is_none_or(|m| n <= m as f64);
    let number = match kind {
        "int" => value.parse::<i64>().ok().map(|n| n as f64),
        "float" => value.parse::<f64>().ok().filter(|f| f.is_finite()),
        _ => None,
    };
    let ok = match kind {
        "int" => value.parse::<i64>().is_ok_and(|n| least.is_none_or(|l| n >= l) && most.is_none_or(|m| n <= m)),
        "float" => number.is_some_and(within),
        "on-off" => matches!(value, "on" | "off"),
        "choice" => list().contains(&value),
        _ => !value.is_empty(),
    };
    if ok {
        return Ok(());
    }
    let range = match (least, most) {
        (Some(l), Some(m)) => format!("{l} to {m}"),
        (Some(l), None) => format!("at least {l}"),
        (None, Some(m)) => format!("at most {m}"),
        (None, None) => String::new(),
    };
    Err(match kind {
        "int" | "float" if number.is_some() && !range.is_empty() => format!("{range} expected, got {value}"),
        "int" => format!("an integer expected, got {value:?}"),
        "float" => format!("a number expected, got {value:?}"),
        "on-off" => format!("on or off expected, got {value:?}"),
        "choice" => format!("one of {} expected, got {value:?}", list().join(", ")),
        _ => "a value expected".to_string(),
    })
}

/// The value's place in a usage line: `<N>`, `<F>`, `<on|off>`, the choices, or `<value>`.
pub fn placeholder(kind: &str, choices: &[impl AsRef<str>]) -> String {
    match kind {
        "int" => "<N>".into(),
        "float" => "<F>".into(),
        "on-off" => "<on|off>".into(),
        "choice" => format!("<{}>", choices.iter().map(AsRef::as_ref).collect::<Vec<_>>().join("|")),
        _ => "<value>".into(),
    }
}

/// The value given for the option `name`, if any.
pub fn option<'a>(args: &'a WorkerArgs, name: &str) -> Option<&'a str> {
    args.extra(&format!("--{name}"))
}

/// Flags of every worker an option may not be named after: the parser reads them before an adapter sees them.
const RESERVED: &[&str] = &["model", "max-context", "max-batch", "kv-bits", "frames-fd", "fd-channel-fd", "engine"];

/// Refuses a declaration the daemon or the worker could not honour.
pub fn check_declaration(options: &[ServeOption]) -> Result<(), String> {
    for (i, o) in options.iter().enumerate() {
        let spelled = !o.name.is_empty()
            && !o.name.starts_with('-')
            && !o.name.ends_with('-')
            && o.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        if !spelled {
            return Err(format!("the option {:?} cannot be written --<runtime>-<name>", o.name));
        }
        if RESERVED.contains(&o.name) {
            return Err(format!("the option {} is named after a flag every worker reads", o.name));
        }
        if options[..i].iter().any(|p| p.name == o.name) {
            return Err(format!("the option {} is declared twice", o.name));
        }
        if o.kind == OptionKind::Choice(&[]) {
            return Err(format!("the option {} offers no choice", o.name));
        }
        if let OptionKind::IntIn(least, most) = o.kind {
            if least > most {
                return Err(format!("the option {} takes nothing: {least} to {most}", o.name));
            }
        }
        if o.about.is_empty() {
            return Err(format!("the option {} needs words for the help", o.name));
        }
    }
    Ok(())
}

/// The flags a worker of `rt` reads besides every worker's: its own, and one per option.
pub fn worker_flags(rt: &dyn Runtime) -> Vec<String> {
    rt.flags().iter().map(|f| f.to_string()).chain(rt.options().iter().map(|o| format!("--{}", o.name))).collect()
}

/// Refuses a value an option does not take, as the daemon does before starting a worker.
pub fn refuse_bad_options(rt: &dyn Runtime, args: &WorkerArgs) -> Result<(), String> {
    for o in rt.options() {
        if let Some(v) = option(args, o.name) {
            o.kind.check(v).map_err(|e| format!("--{}: {e}", o.name))?;
        }
    }
    Ok(())
}

/// The declarations as a report carries them.
pub fn declared(rt: &dyn Runtime) -> Vec<OptionDecl> {
    rt.options()
        .iter()
        .map(|o| OptionDecl {
            name: o.name.into(),
            kind: o.kind.name().into(),
            choices: o.kind.choices().iter().map(|c| c.to_string()).collect(),
            least: o.kind.least(),
            most: o.kind.most(),
            default: o.default.into(),
            about: o.about.into(),
        })
        .collect()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Pull {
    pub id: String,
    pub offline: bool,
    pub options: Vec<(String, Option<String>)>,
}

impl Pull {
    pub fn split(&self) -> Result<(&str, Option<&str>), String> {
        match self.id.split_once(':') {
            Some((_, "")) => Err(format!("an empty tag after ':' in {}: name one or drop the colon", self.id)),
            Some((id, tag)) => Ok((id, Some(tag))),
            None => Ok((&self.id, None)),
        }
    }

    pub fn option(&self, name: &str) -> Option<Option<&str>> {
        self.options.iter().rev().find(|(n, _)| n == name).map(|(_, v)| v.as_deref())
    }

    pub fn parse(args: &[String], declared: &[PullOption]) -> Result<Pull, String> {
        let mut it = args.iter();
        let id = it.next().filter(|a| !a.starts_with("--")).ok_or("pull needs a model id")?.clone();
        let mut pull = Pull { id, ..Pull::default() };
        while let Some(flag) = it.next() {
            let name = flag.strip_prefix("--").ok_or_else(|| format!("unexpected argument {flag:?}"))?;
            if name == "offline" {
                pull.offline = true;
                continue;
            }
            let opt = declared.iter().find(|o| o.name == name).ok_or_else(|| {
                let all: Vec<&str> = declared.iter().map(|o| o.name).collect();
                format!("the pull takes no --{name} (it takes: {})", if all.is_empty() { "none".into() } else { all.join(", ") })
            })?;
            let value = if opt.value { Some(it.next().cloned().ok_or_else(|| format!("--{name} needs a value"))?) } else { None };
            pull.options.push((name.to_string(), value));
        }
        Ok(pull)
    }

    pub fn to_args(&self) -> Vec<String> {
        let mut a = vec![self.id.clone()];
        if self.offline {
            a.push("--offline".into());
        }
        for (name, value) in &self.options {
            a.push(format!("--{name}"));
            a.extend(value.clone());
        }
        a
    }
}

pub fn run_pull_tool(cmd: &mut std::process::Command, what: &str) -> Result<std::path::PathBuf, String> {
    let stdout = run_tool(cmd, what)?;
    stdout
        .lines()
        .map(str::trim)
        .map(std::path::PathBuf::from)
        .find(|p| p.exists())
        .ok_or_else(|| format!("{what} printed no path to a model: {}", stdout.trim()))
}

/// Runs a pull tool, showing its stderr as it goes, and returns its stdout; a failure
/// carries the tail of what it said.
pub fn run_tool(cmd: &mut std::process::Command, what: &str) -> Result<String, String> {
    use std::io::Read;
    let mut child = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("{what}: {e}"))?;
    let errors = child.stderr.take().map(|from| std::thread::spawn(move || show_and_keep(from)));
    let mut stdout = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    let status = child.wait().map_err(|e| format!("{what}: {e}"))?;
    let stderr = errors.and_then(|t| t.join().ok()).unwrap_or_default();
    if !status.success() {
        return Err(pull_failed(what, &status.to_string(), &stdout, &stderr, std::env::var_os("HF_TOKEN").is_some()));
    }
    Ok(stdout)
}

const KEPT_STDERR: usize = 16 * 1024;

fn show_and_keep(mut from: impl std::io::Read) -> String {
    use std::io::Write;
    let mut kept: Vec<u8> = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match from.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let mut err = std::io::stderr().lock();
                let _ = err.write_all(&buf[..n]);
                let _ = err.flush();
                kept.extend_from_slice(&buf[..n]);
                if kept.len() > 2 * KEPT_STDERR {
                    kept.drain(..kept.len() - KEPT_STDERR);
                }
            }
        }
    }
    String::from_utf8_lossy(&kept).into_owned()
}

fn pull_failed(what: &str, how: &str, stdout: &str, stderr: &str, has_token: bool) -> String {
    let said: Vec<&str> = stdout.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let reason = if said.is_empty() {
        stderr.split(['\n', '\r']).map(str::trim).rfind(|l| !l.is_empty()).unwrap_or_default().to_string()
    } else {
        said.join(" / ")
    };
    let mut heard = format!("{stdout}\n{stderr}");
    for named in what.split_whitespace().filter(|w| w.contains('/')) {
        heard = heard.replace(named, " ");
        if let Some((id, _tag)) = named.split_once(':') {
            heard = heard.replace(id, " ");
        }
    }
    let hint = if has_token { None } else { access_hint(&heard) };
    format!("{what} failed ({how}){}{}", if reason.is_empty() { String::new() } else { format!(": {reason}") }, hint.unwrap_or_default())
}

pub fn token_hint(said: &str) -> Option<&'static str> {
    if std::env::var_os("HF_TOKEN").is_some() {
        return None;
    }
    access_hint(said)
}

fn access_hint(said: &str) -> Option<&'static str> {
    refused_for_access(said).then_some(" (a gated or private model needs a Hugging Face token: set HF_TOKEN)")
}

fn refused_for_access(said: &str) -> bool {
    let lower = said.to_ascii_lowercase();
    let phrase = ["unauthorized", "forbidden", "gated", "access to model", "authentication", "restricted"];
    let word = ["401", "403", "token", "hf_token"];
    phrase.iter().any(|w| lower.contains(w))
        || lower.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).any(|w| word.contains(&w))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    pub version: String,
    pub devices: Vec<Device>,
    /// The contract number the runtime states, when it states one (baseRT's engine ABI).
    pub abi: Option<u64>,
}

pub trait Runtime: Send + Sync {
    fn info(&self) -> RuntimeInfo;

    fn flags(&self) -> &'static [&'static str] {
        &[]
    }

    /// The options this runtime takes on `superfluid serve`, as `--<id>-<name> <value>`: the daemon checks a value
    /// against its kind, forwards it as `--<name> <value>`, and lists them in `superfluid runtime options <id>`.
    fn options(&self) -> &'static [ServeOption] {
        &[]
    }

    fn needs_model(&self) -> bool {
        true
    }

    /// What this adapter was built and tested for, as a release carries it (`superfluid compat`, `compat.json`):
    /// `tested` names the versions it installs on its own; the rest of the keys are the adapter's to define.
    fn compat(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    fn prepare(&self, _args: &WorkerArgs) {}

    fn probe(&self, args: &WorkerArgs) -> Result<Probe, String>;

    fn named(&self, _args: &WorkerArgs) -> Option<String> {
        None
    }

    fn static_capabilities(&self) -> String {
        String::new()
    }

    fn reads(&self, path: &Path) -> Result<(), String> {
        let info = self.info();
        if info.formats.iter().any(|f| f.matches(path)) {
            return Ok(());
        }
        Err(format!("{} is not {}", path.display(), describe_formats(&info.formats)))
    }

    fn model_facts(&self, _path: &Path) -> Option<Result<String, String>> {
        None
    }

    fn sizing(&self, _path: &Path, _args: &WorkerArgs) -> Option<Result<Sizing, String>> {
        None
    }

    fn open(&self, args: &WorkerArgs) -> Result<Loaded, String>;

    fn tokenizer(&self, _model: &Path) -> Option<Result<Arc<dyn Tokenizer>, String>> {
        None
    }

    fn plan(&self, _req: &Request, _host: &Host) -> Option<Result<Vec<Plan>, String>> {
        None
    }

    /// The versions this adapter installs on its own, newest first, each planned for `host` as `plan` would (`superfluid runtime
    /// versions`). Answered from the recipe alone, never the network: by default every `tested` version `compat` names.
    fn versions(&self, host: &Host) -> Vec<Candidate> {
        let tested: Vec<String> =
            self.compat()["tested"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
        tested
            .into_iter()
            .map(|version| {
                let req = Request { version: Some(version.clone()), ..Request::default() };
                let here = match self.plan(&req, host) {
                    Some(Ok(mut plans)) if !plans.is_empty() => Ok(plans.remove(0)),
                    Some(Ok(_)) => Err("nothing to install here".to_string()),
                    Some(Err(why)) => Err(why),
                    None => Err(format!("the {} runtime is not installed from a project", self.info().id)),
                };
                Candidate { version, here }
            })
            .collect()
    }

    fn pull_options(&self) -> Option<&'static [PullOption]> {
        None
    }

    fn pull(&self, _pull: &Pull, _args: &WorkerArgs) -> Result<std::path::PathBuf, String> {
        Err(format!("the {} runtime pulls no model by id", self.info().id))
    }

    fn install(&self, plan: &Plan, dir: &Path) -> Result<Vec<(String, String)>, String> {
        install::fetch_and_unpack(plan, &dir.join("lib"), dir)
    }
}

pub fn describe_formats(formats: &[Format]) -> String {
    let all: Vec<&str> = formats.iter().map(|f| f.describe.as_str()).collect();
    match all.as_slice() {
        [] => "an artifact of any format this runtime declares (it declares none)".to_string(),
        [one] => one.to_string(),
        [init @ .., last] => format!("{} or {last}", init.join(", ")),
    }
}

pub struct Loaded {
    serve: Box<dyn FnOnce(UnixStream, UnixStream) -> Result<(), WorkerError>>,
}

impl Loaded {
    pub fn engine<E: Engine + 'static>(engine: E, rings: Option<Vec<RingSpec>>) -> Loaded {
        Loaded {
            serve: Box::new(move |frames, fds| {
                let mut server = WorkerServer::new(engine, WorkerConfig { engine_bundle_hash: [0; 32] }, frames, fds);
                if let Some(rings) = rings {
                    server = server.with_ring_specs(rings);
                }
                server.serve()
            }),
        }
    }

    pub fn primitives<P: RuntimePrimitives + 'static>(runtime: P, ctx: u32) -> Loaded {
        let engine = Executor::new(runtime, ExecutorConfig::default());
        let vocab = engine.descriptor().vocab_size;
        Loaded::engine(engine, Some(ring_specs_for(ctx, vocab)))
    }

    pub fn serve(self, frames: UnixStream, fds: UnixStream) -> Result<(), WorkerError> {
        (self.serve)(frames, fds)
    }
}

pub fn context(args: &WorkerArgs) -> u32 {
    args.max_context.max(512) as u32
}

pub fn model(args: &WorkerArgs) -> Result<&Path, String> {
    args.model.as_deref().ok_or_else(|| "--model is required to serve".to_string())
}

pub fn check(rt: &dyn Runtime, args: &WorkerArgs) -> Report {
    let (version, abi, devices, available) = match rt.probe(args) {
        Ok(p) => (Some(p.version), p.abi, p.devices, Ok(())),
        Err(why) => (None, None, Vec::new(), Err(why)),
    };
    let model = args.model.as_ref().map(|path| {
        let reads = rt.reads(path);
        let (facts, sizing) = if reads.is_ok() {
            (rt.model_facts(path), rt.sizing(path, args).map(|s| s.map(|s| s.to_json().to_string())))
        } else {
            (None, None)
        };
        ModelCheck { path: path.clone(), reads, facts, sizing }
    });
    let pull = rt
        .pull_options()
        .map(|options| options.iter().map(|o| (o.name.to_string(), o.value, o.about.to_string())).collect());
    Report {
        info: rt.info(),
        version,
        abi,
        available,
        devices,
        capabilities: rt.static_capabilities(),
        model,
        pull,
        options: declared(rt),
        named: rt.named(args),
    }
}

pub fn serve_process(rt: &dyn Runtime, mut args: WorkerArgs, name: &str) -> ! {
    let refuse = |why: &str| -> ! {
        eprintln!("{name}: {why}");
        std::process::exit(2);
    };
    if rt.needs_model() && args.model.is_none() {
        refuse("--model is required to serve");
    }
    let (frames, fds) = args.streams().unwrap_or_else(|e| refuse(&e));
    superfluid_executor::exit_on_fatal(true);
    let loaded = rt.open(&args).unwrap_or_else(|e| {
        eprintln!("{name}: model load failed: {e}");
        std::process::exit(3);
    });
    match loaded.serve(frames, fds) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("{name}: exited with error: {e}");
            std::process::exit(1);
        }
    }
}

pub fn worker_main(rt: &dyn Runtime) -> ! {
    let name = format!("superfluid-worker-{}", rt.info().id);
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if let Some(command @ ("plan" | "install")) = argv.first().map(String::as_str) {
        rt.prepare(&WorkerArgs::default());
        match command {
            "plan" => plan_command(rt, &name, &argv[1..]),
            _ => install_command(rt, &name, &argv[1..]),
        }
    }
    if argv.first().map(String::as_str) == Some("pull") {
        pull_command(rt, &name, &argv[1..]);
    }
    if argv.first().map(String::as_str) == Some("compat") {
        // Always JSON; nothing of the runtime is needed.
        rt.prepare(&WorkerArgs::default());
        println!("{}", serde_json::to_string(&rt.compat()).expect("compat is JSON"));
        std::process::exit(0);
    }
    if argv.first().map(String::as_str) == Some("versions") {
        rt.prepare(&WorkerArgs::default());
        let host = Host::detect();
        let v = serde_json::json!({
            "host": host.to_json(),
            "describe": host.describe(),
            "versions": rt.versions(&host).iter().map(Candidate::to_json).collect::<Vec<_>>(),
        });
        println!("{v}");
        std::process::exit(0);
    }
    let (command, rest) = match argv.first().map(String::as_str) {
        Some(c @ ("version" | "check")) => (Some(c), &argv[1..]),
        _ => (None, &argv[..]),
    };
    let args = WorkerArgs::parse(rest).unwrap_or_else(|e| usage(rt, &name, &e));
    let reads = worker_flags(rt);
    args.refuse_unread(&reads.iter().map(String::as_str).collect::<Vec<_>>()).unwrap_or_else(|e| usage(rt, &name, &e));
    refuse_bad_options(rt, &args).unwrap_or_else(|e| usage(rt, &name, &e));
    rt.prepare(&args);
    match command {
        Some("version") => match rt.probe(&args) {
            Ok(p) => {
                println!("{}", p.version);
                std::process::exit(0)
            }
            Err(why) => {
                eprintln!("{name}: {why}");
                std::process::exit(1)
            }
        },
        Some(_) => check(rt, &args).finish(),
        None => serve_process(rt, args, &name),
    }
}

fn plan_command(rt: &dyn Runtime, name: &str, args: &[String]) -> ! {
    let req = Request::parse(args).unwrap_or_else(|e| usage(rt, name, &e));
    let host = Host::detect();
    match rt.plan(&req, &host) {
        None => fail(name, &format!("the {} runtime is not installed from a project", rt.info().id)),
        Some(Err(why)) => fail(name, &why),
        Some(Ok(plans)) => {
            let v = serde_json::json!({
                "host": host.to_json(),
                "describe": host.describe(),
                "plans": plans.iter().map(Plan::to_json).collect::<Vec<_>>(),
            });
            println!("{v}");
            std::process::exit(0)
        }
    }
}

fn install_command(rt: &dyn Runtime, name: &str, args: &[String]) -> ! {
    let (mut into, mut plan) = (None, None);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it.next().cloned().unwrap_or_else(|| usage(rt, name, &format!("{flag} needs a value")));
        match flag.as_str() {
            "--into" => into = Some(std::path::PathBuf::from(value)),
            "--plan" => plan = Some(value),
            other => usage(rt, name, &format!("unknown flag {other}")),
        }
    }
    let (Some(dir), Some(plan)) = (into, plan) else { usage(rt, name, "install needs --into and --plan") };
    let plan = serde_json::from_str(&plan).map_err(|e| e.to_string()).and_then(|v| Plan::from_json(&v)).unwrap_or_else(|e| fail(name, &e));
    match install_into(rt, &plan, &dir) {
        Ok(manifest) => {
            println!("{}", manifest.to_json());
            std::process::exit(0)
        }
        Err(why) => fail(name, &why),
    }
}

fn pull_command(rt: &dyn Runtime, name: &str, args: &[String]) -> ! {
    let Some(declared) = rt.pull_options() else { fail(name, &format!("the {} runtime pulls no model by id", rt.info().id)) };
    let (worker, pull_args) = split_pull_args(rt, args);
    let wargs = WorkerArgs::parse(&worker).unwrap_or_else(|e| usage(rt, name, &e));
    refuse_bad_options(rt, &wargs).unwrap_or_else(|e| usage(rt, name, &e));
    let pull = Pull::parse(&pull_args, declared).unwrap_or_else(|e| usage(rt, name, &e));
    rt.prepare(&wargs);
    match rt.pull(&pull, &wargs) {
        Ok(path) => {
            println!("{}", serde_json::json!({"path": path}));
            std::process::exit(0)
        }
        Err(why) => fail(name, &why),
    }
}

/// A pull's command line: the worker's own flags and options, which set up the runtime, then the pull's arguments.
pub fn split_pull_args(rt: &dyn Runtime, args: &[String]) -> (Vec<String>, Vec<String>) {
    let flags = worker_flags(rt);
    let (mut worker, mut pull_args) = (Vec::new(), Vec::new());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if flags.iter().any(|f| f == a) {
            worker.push(a.clone());
            worker.extend(it.next().cloned());
        } else {
            pull_args.push(a.clone());
        }
    }
    (worker, pull_args)
}

pub fn install_into(rt: &dyn Runtime, plan: &Plan, dir: &Path) -> Result<Manifest, String> {
    let id = rt.info().id;
    if plan.id != id {
        return Err(format!("a plan for {} given to the {id} runtime", plan.id));
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let sources = rt.install(plan, dir)?;
    install::copy_worker(dir, id)?;
    let manifest = Manifest {
        id: id.to_string(),
        install: plan.install.clone(),
        version: plan.version.clone(),
        backend: plan.backend.clone(),
        link_w: vec![superfluid_worker::server::PROTO_VERSION],
        platforms: vec![Host::detect().platform()],
        worker: format!("bin/superfluid-worker-{id}"),
        sources,
        tested: plan.tested,
    };
    manifest.write(dir)?;
    Ok(manifest)
}

fn fail(name: &str, why: &str) -> ! {
    eprintln!("{name}: {why}");
    std::process::exit(1);
}

fn usage(rt: &dyn Runtime, name: &str, why: &str) -> ! {
    let flags: String = rt.flags().iter().map(|f| format!(" [{f} <value>]")).collect();
    let options = if rt.options().is_empty() { "" } else { " [--<option> <value>]..." };
    eprintln!("{name}: {why}");
    eprintln!(
        "usage: {name} check [--model <path>]{flags}{options}\n       {name} version{flags}\n       \
         {name} plan [--version V] [--backend B] [--from <url|path>] [--sha256 H] [--untested]\n       \
         {name} install --into <dir> --plan <json>\n       \
         {name} pull <org/model[:tag]> [--offline] [--<option> [value]]...\n       \
         {name} --frames-fd N --fd-channel-fd M --model <path> [--max-context N] [--max-batch N]{flags}{options}"
    );
    if !rt.options().is_empty() {
        eprintln!("options: {}", rt.options().iter().map(|o| o.name).collect::<Vec<_>>().join(", "));
    }
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Fake {
        runs: bool,
        asked_facts: AtomicBool,
    }

    impl Runtime for Fake {
        fn info(&self) -> RuntimeInfo {
            RuntimeInfo {
                id: "fake",
                aliases: &["phony"],
                formats: vec![Format::file("fake", "a fake file", &["fake"], None)],
                tokenizer: TokenizerSource::HuggingFace,
            }
        }
        fn probe(&self, _args: &WorkerArgs) -> Result<Probe, String> {
            match self.runs {
                true => Ok(Probe { version: "fake 1".into(), devices: vec![Device { backend: "CPU".into(), name: "cpu".into(), memory: 0 }], abi: None }),
                false => Err("no fake device here".into()),
            }
        }
        fn model_facts(&self, _path: &Path) -> Option<Result<String, String>> {
            self.asked_facts.store(true, Ordering::Relaxed);
            Some(Ok(r#"{"architecture":"fake"}"#.into()))
        }
        fn open(&self, _args: &WorkerArgs) -> Result<Loaded, String> {
            Err("a fake runtime loads nothing".into())
        }
    }

    fn fake(runs: bool) -> Fake {
        Fake { runs, asked_facts: AtomicBool::new(false) }
    }

    /// A runtime that installs two versions of itself, the older one nowhere.
    struct Installs;

    impl Runtime for Installs {
        fn info(&self) -> RuntimeInfo {
            RuntimeInfo { id: "inst", aliases: &[], formats: vec![], tokenizer: TokenizerSource::HuggingFace }
        }
        fn compat(&self) -> serde_json::Value {
            serde_json::json!({"tested": ["2.0", "1.0"]})
        }
        fn probe(&self, _args: &WorkerArgs) -> Result<Probe, String> {
            Err("never".into())
        }
        fn open(&self, _args: &WorkerArgs) -> Result<Loaded, String> {
            Err("never".into())
        }
        fn plan(&self, req: &Request, host: &Host) -> Option<Result<Vec<Plan>, String>> {
            let version = req.version.clone().unwrap_or_else(|| "2.0".into());
            if version == "1.0" {
                return Some(Err(format!("inst 1.0 publishes no build for {}", host.platform())));
            }
            Some(Ok(vec![Plan {
                id: "inst".into(),
                install: format!("{version}-cpu"),
                backend: "cpu".into(),
                assets: vec![install::Asset { name: "a.tgz".into(), url: "https://x/a.tgz".into(), sha256: None, size: Some(7) }],
                tested: true,
                why: "a test".into(),
                version,
                extra: serde_json::json!({"wheels_size": 3}),
            }]))
        }
    }

    #[test]
    fn a_pull_takes_the_runtime_options_apart_from_its_own_arguments() {
        struct WithOptions;
        impl Runtime for WithOptions {
            fn info(&self) -> RuntimeInfo {
                RuntimeInfo { id: "opt", aliases: &[], formats: vec![], tokenizer: TokenizerSource::HuggingFace }
            }
            fn flags(&self) -> &'static [&'static str] {
                &["--venv"]
            }
            fn options(&self) -> &'static [ServeOption] {
                &[ServeOption { name: "lib", kind: OptionKind::Text, default: "", about: "a library" }]
            }
            fn probe(&self, _args: &WorkerArgs) -> Result<Probe, String> {
                Err("never".into())
            }
            fn open(&self, _args: &WorkerArgs) -> Result<Loaded, String> {
                Err("never".into())
            }
        }
        let argv: Vec<String> = ["org/m:q4", "--lib", "/custom/lib", "--offline", "--venv", "/custom/env", "--revision", "main"].iter().map(|s| s.to_string()).collect();
        let (worker, pull) = split_pull_args(&WithOptions, &argv);
        assert_eq!(worker, ["--lib", "/custom/lib", "--venv", "/custom/env"], "the flag and the option both set up the runtime");
        assert_eq!(pull, ["org/m:q4", "--offline", "--revision", "main"]);
        let wargs = WorkerArgs::parse(&worker).unwrap();
        assert_eq!((wargs.venv.as_deref(), option(&wargs, "lib")), (Some(Path::new("/custom/env")), Some("/custom/lib")));
    }

    #[test]
    fn the_versions_listed_are_the_tested_ones_each_planned_here() {
        let host = Host { os: "linux".into(), arch: "x64".into(), ..Host::default() };
        let listed = Installs.versions(&host);
        assert_eq!(listed.iter().map(|c| c.version.as_str()).collect::<Vec<_>>(), ["2.0", "1.0"], "the adapter's order");
        assert_eq!(listed[0].here.as_ref().map(|p| p.install.as_str()), Ok("2.0-cpu"));
        assert_eq!(listed[0].size(), 10, "the assets and the wheels");
        assert_eq!(listed[1].here, Err("inst 1.0 publishes no build for linux-x64".to_string()));
        assert_eq!(listed[1].size(), 0);
        for c in &listed {
            assert_eq!(&Candidate::from_json(&c.to_json()).unwrap(), c);
        }
        assert!(Fake { runs: true, asked_facts: AtomicBool::new(false) }.versions(&host).is_empty(), "no compat, no versions");
    }

    #[test]
    fn a_runtime_that_does_not_run_here_says_why_and_lists_no_device() {
        let report = check(&fake(false), &WorkerArgs::default());
        assert_eq!(report.version, None);
        assert_eq!(report.available, Err("no fake device here".to_string()));
        assert!(report.devices.is_empty());
        assert_eq!(report.exit_code(), 1);
        let report = check(&fake(true), &WorkerArgs::default());
        assert_eq!((report.version.as_deref(), report.devices.len(), report.exit_code()), (Some("fake 1"), 1, 0));
    }

    #[test]
    fn model_facts_are_asked_for_only_about_a_model_the_runtime_reads() {
        let dir = std::env::temp_dir().join(format!("superfluid-kit-check-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let other = dir.join("m.gguf");
        std::fs::write(&other, b"GGUF").unwrap();
        let rt = fake(true);
        let report = check(&rt, &WorkerArgs { model: Some(other.clone()), ..Default::default() });
        let m = report.model.as_ref().unwrap();
        assert_eq!(m.reads, Err(format!("{} is not a fake file", other.display())), "the default reads is by the formats");
        assert!(m.facts.is_none() && !rt.asked_facts.load(Ordering::Relaxed));
        assert_eq!(report.exit_code(), 1);

        let mine = dir.join("m.fake");
        std::fs::write(&mine, b"").unwrap();
        let report = check(&rt, &WorkerArgs { model: Some(mine), ..Default::default() });
        assert_eq!(report.model.as_ref().unwrap().facts, Some(Ok(r#"{"architecture":"fake"}"#.to_string())));
        assert_eq!(report.exit_code(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn formats_are_named_in_a_sentence() {
        let gguf = Format::file("gguf", "a GGUF file", &["gguf"], Some(b"GGUF"));
        let base = Format::file("base", "a .base bundle", &["base"], Some(b"BASE"));
        let mlx = Format::directory("mlx", "an MLX model directory", &["config.json"], &["safetensors"]);
        assert_eq!(describe_formats(std::slice::from_ref(&gguf)), "a GGUF file");
        assert_eq!(describe_formats(&[base, gguf, mlx]), "a .base bundle, a GGUF file or an MLX model directory");
        assert!(describe_formats(&[]).contains("declares none"));
    }

    #[test]
    fn a_pull_reads_its_id_tag_and_declared_options_and_refuses_the_rest() {
        const OPTS: &[PullOption] = &[
            PullOption { name: "target", value: true, about: "a quant" },
            PullOption { name: "force", value: false, about: "again" },
        ];
        let args = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let p = Pull::parse(&args(&["org/m:q8", "--target", "base-q8", "--force", "--offline"]), OPTS).unwrap();
        assert_eq!(p.split().unwrap(), ("org/m", Some("q8")));
        assert_eq!((p.option("target"), p.option("force"), p.option("profile"), p.offline), (Some(Some("base-q8")), Some(None), None, true));
        assert_eq!(Pull::parse(&p.to_args(), OPTS).unwrap(), p);
        assert!(Pull::parse(&args(&["org/m", "--revision", "x"]), OPTS).unwrap_err().contains("takes no --revision (it takes: target, force)"));
        assert!(Pull::parse(&args(&["org/m", "--target"]), OPTS).unwrap_err().contains("needs a value"));
        assert!(Pull::parse(&args(&["--force"]), OPTS).unwrap_err().contains("needs a model id"));
        assert!(Pull { id: "org/m:".into(), ..Pull::default() }.split().unwrap_err().contains("empty tag"));
    }

    #[test]
    fn the_token_hint_follows_what_the_tool_said_not_the_models_id() {
        let hint = "(a gated or private model needs a Hugging Face token: set HF_TOKEN)";
        let e = pull_failed("llama download org/gated-token-model", "exit status: 1", "", "curl: (6) Could not resolve host: huggingface.co\n", false);
        assert_eq!(e, "llama download org/gated-token-model failed (exit status: 1): curl: (6) Could not resolve host: huggingface.co");
        let e = pull_failed("llama download org/m", "exit status: 1", "", "fetching 10%\rfetching 20%\nerror: 401 Unauthorized\n", false);
        assert_eq!(e, format!("llama download org/m failed (exit status: 1): error: 401 Unauthorized {hint}"));
        let e = pull_failed("basert resolve org/m", "exit status: 2", "cannot resolve org/m\n", "Access to model org/m is restricted\n", false);
        assert_eq!(e, format!("basert resolve org/m failed (exit status: 2): cannot resolve org/m {hint}"));
        let said = "acme/token-classifier is not installed, and nothing was fetched (offline); `basert pull acme/token-classifier` fetches it\n";
        let e = pull_failed("basert resolve acme/token-classifier:default-q4 --offline", "exit status: 1", "", said, false);
        assert!(!e.contains("HF_TOKEN"), "{e}");
        assert!(!pull_failed("x", "exit status: 1", "", "403 Forbidden", true).contains("HF_TOKEN"));
        assert_eq!(pull_failed("x", "signal: 9", "", "", false), "x failed (signal: 9)");

        for refusal in ["HTTP 401", "403 Forbidden", "gated repo", "Invalid user token", "set HF_TOKEN or pass --hf-token", "requires authentication"] {
            assert!(refused_for_access(refusal), "{refusal}");
        }
        for other in ["tokenizer.json not found", "wrote 4013 bytes", "sha256 e4031f", "connection reset by peer", ""] {
            assert!(!refused_for_access(other), "{other}");
        }
    }

    #[test]
    fn a_pull_tools_stderr_is_read_for_why_it_failed() {
        let script = |body: &str| {
            let mut c = std::process::Command::new("/bin/sh");
            c.arg("-c").arg(body);
            c
        };
        let e = run_pull_tool(&mut script("echo 'fetching model' >&2; echo 'error: 403 Forbidden' >&2; exit 1"), "toy download org/m").unwrap_err();
        assert!(e.starts_with("toy download org/m failed (exit status: 1): error: 403 Forbidden"), "{e}");
        assert_eq!(e.contains("set HF_TOKEN"), std::env::var_os("HF_TOKEN").is_none(), "{e}");
        let e = run_pull_tool(&mut script("echo 'no such repository' >&2; exit 3"), "toy download org/gated-token").unwrap_err();
        assert_eq!(e, "toy download org/gated-token failed (exit status: 3): no such repository");
        let here = std::env::temp_dir();
        let ok = run_pull_tool(&mut script(&format!("echo progress >&2; echo '{}'", here.display())), "toy download org/m").unwrap();
        assert_eq!(ok, here);
        let e = run_pull_tool(&mut script("echo nothing-on-disk"), "toy download org/m").unwrap_err();
        assert!(e.contains("printed no path to a model"), "{e}");
    }

    #[test]
    fn a_primitive_runtime_opens_at_no_less_than_the_prompt_rings_floor() {
        let args = |n| WorkerArgs { max_context: n, ..Default::default() };
        assert_eq!((context(&args(0)), context(&args(4096))), (512, 4096));
        assert_eq!(model(&WorkerArgs::default()).unwrap_err(), "--model is required to serve");
    }
}

#[cfg(test)]
mod option_tests {
    use super::*;

    #[test]
    fn a_value_is_judged_by_its_kind() {
        assert!(OptionKind::Int.check("-3").is_ok());
        assert_eq!(OptionKind::Int.check("x").unwrap_err(), "an integer expected, got \"x\"");
        assert!(OptionKind::IntFrom(1).check("1").is_ok());
        assert_eq!(OptionKind::IntFrom(1).check("0").unwrap_err(), "at least 1 expected, got 0");
        assert!(OptionKind::IntIn(0, 512).check("512").is_ok() && OptionKind::IntIn(0, 512).check("0").is_ok());
        assert_eq!(OptionKind::IntIn(0, 512).check("2147483647").unwrap_err(), "0 to 512 expected, got 2147483647");
        assert_eq!(OptionKind::IntIn(0, 512).check("-1").unwrap_err(), "0 to 512 expected, got -1");
        assert_eq!(OptionKind::IntIn(0, 512).check("99999999999999999999").unwrap_err(), "an integer expected, got \"99999999999999999999\"");
        assert!(OptionKind::Float.check("1e6").is_ok() && OptionKind::Float.check("inf").is_err());
        assert!(OptionKind::FloatFrom(0).check("0").is_ok() && OptionKind::FloatFrom(0).check("0.25").is_ok());
        assert_eq!(OptionKind::FloatFrom(0).check("-1.5").unwrap_err(), "at least 0 expected, got -1.5");
        assert_eq!(OptionKind::FloatFrom(0).check("nan").unwrap_err(), "a number expected, got \"nan\"");
        assert_eq!((OptionKind::IntIn(1, 8).name(), OptionKind::FloatFrom(0).name()), ("int", "float"));
        assert_eq!((OptionKind::IntIn(1, 8).least(), OptionKind::IntIn(1, 8).most(), OptionKind::IntFrom(1).most()), (Some(1), Some(8), None));
        assert!(OptionKind::OnOff.check("on").is_ok() && OptionKind::OnOff.check("true").is_err());
        assert_eq!(OptionKind::Choice(&["a", "b"]).check("c").unwrap_err(), "one of a, b expected, got \"c\"");
        assert!(OptionKind::Text.check("").is_err() && OptionKind::Text.check("x").is_ok());
        assert_eq!(placeholder("choice", &["a", "b"]), "<a|b>");
        assert_eq!((placeholder("int", &[] as &[&str]), placeholder("on-off", &[] as &[&str])), ("<N>".into(), "<on|off>".into()));
    }

    const OK: ServeOption = ServeOption { name: "n-gpu-layers", kind: OptionKind::IntFrom(-1), default: "-1", about: "layers" };
    static ONE: [ServeOption; 1] = [OK];

    #[test]
    fn a_declaration_the_daemon_could_not_honour_is_refused() {
        let ok = OK;
        assert!(check_declaration(&[ok]).is_ok());
        for (bad, said) in [
            (ServeOption { name: "Threads", ..ok }, "cannot be written"),
            (ServeOption { name: "max-context", ..ok }, "a flag every worker reads"),
            (ServeOption { kind: OptionKind::Choice(&[]), ..ok }, "offers no choice"),
            (ServeOption { kind: OptionKind::IntIn(8, 1), ..ok }, "takes nothing"),
            (ServeOption { about: "", ..ok }, "needs words"),
        ] {
            assert!(check_declaration(&[bad]).unwrap_err().contains(said), "{bad:?}");
        }
        assert!(check_declaration(&[ok, ok]).unwrap_err().contains("declared twice"));
        assert_eq!(declared_of(&ONE)[0].least, Some(-1));
    }

    fn declared_of(options: &'static [ServeOption]) -> Vec<OptionDecl> {
        struct Toy(&'static [ServeOption]);
        impl Runtime for Toy {
            fn info(&self) -> RuntimeInfo {
                RuntimeInfo { id: "toy", aliases: &[], formats: Vec::new(), tokenizer: TokenizerSource::Engine }
            }
            fn options(&self) -> &'static [ServeOption] {
                self.0
            }
            fn probe(&self, _: &WorkerArgs) -> Result<Probe, String> {
                Err("toy".into())
            }
            fn open(&self, _: &WorkerArgs) -> Result<Loaded, String> {
                Err("toy".into())
            }
        }
        declared(&Toy(options))
    }
}
