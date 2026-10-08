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
pub use superfluid_worker::check::{ModelCheck, Report};
pub use superfluid_worker::{ring_specs_for, WorkerArgs};

pub mod conformance;
pub mod install;

use install::{Host, Manifest, Plan, Request};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PullOption {
    pub name: &'static str,
    pub value: bool,
    pub about: &'static str,
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
}

pub trait Runtime: Send + Sync {
    fn info(&self) -> RuntimeInfo;

    fn flags(&self) -> &'static [&'static str] {
        &[]
    }

    fn needs_model(&self) -> bool {
        true
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
    let (version, devices, available) = match rt.probe(args) {
        Ok(p) => (Some(p.version), p.devices, Ok(())),
        Err(why) => (None, Vec::new(), Err(why)),
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
    Report { info: rt.info(), version, available, devices, capabilities: rt.static_capabilities(), model, pull, named: rt.named(args) }
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
    let (command, rest) = match argv.first().map(String::as_str) {
        Some(c @ ("version" | "check")) => (Some(c), &argv[1..]),
        _ => (None, &argv[..]),
    };
    let args = WorkerArgs::parse(rest).unwrap_or_else(|e| usage(rt, &name, &e));
    args.refuse_unread(rt.flags()).unwrap_or_else(|e| usage(rt, &name, &e));
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
    let (mut worker, mut pull_args) = (Vec::new(), Vec::new());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if rt.flags().contains(&a.as_str()) {
            worker.push(a.clone());
            worker.extend(it.next().cloned());
        } else {
            pull_args.push(a.clone());
        }
    }
    let wargs = WorkerArgs::parse(&worker).unwrap_or_else(|e| usage(rt, name, &e));
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
    eprintln!("{name}: {why}");
    eprintln!(
        "usage: {name} check [--model <path>]{flags}\n       {name} version{flags}\n       \
         {name} plan [--version V] [--backend B] [--from <url|path>] [--sha256 H] [--untested]\n       \
         {name} install --into <dir> --plan <json>\n       \
         {name} pull <org/model[:tag]> [--offline] [--<option> [value]]...\n       \
         {name} --frames-fd N --fd-channel-fd M --model <path> [--max-context N] [--max-batch N]{flags}"
    );
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
                true => Ok(Probe { version: "fake 1".into(), devices: vec![Device { backend: "CPU".into(), name: "cpu".into(), memory: 0 }] }),
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
