//! Where each runtime's worker is, and what it says about itself.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use superfluid_engine::artifact::{Device, Format, TokenizerSource};
use serde_json::Value;

use crate::runtime_pick::RuntimeId;

const CHECK_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Env,
    BesideDaemon,
    BuiltIn,
    Installed,
    Shipped,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Env => "environment",
            Source::BesideDaemon => "beside superfluid",
            Source::BuiltIn => "built in",
            Source::Installed => "installed",
            Source::Shipped => "shipped with superfluid",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Worker {
    pub bin: PathBuf,
    pub prefix: Vec<String>,
    pub source: Source,
}

impl Worker {
    pub fn install(&self) -> Option<String> {
        if self.source != Source::Installed {
            return None;
        }
        self.bin.parent()?.parent()?.file_name().and_then(|n| n.to_str()).map(str::to_string)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Info {
    pub aliases: Vec<String>,
    pub formats: Vec<Format>,
    pub tokenizer: Option<TokenizerSource>,
    pub named: Option<String>,
}

impl Info {
    fn from_report(report: &Value) -> Info {
        use superfluid_worker::check::{formats_from, tokenizer_from};
        Info {
            aliases: report["runtime"]["aliases"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
                .unwrap_or_default(),
            formats: formats_from(&report["formats"]),
            tokenizer: tokenizer_from(&report["tokenizer"]),
            named: report["named"].as_str().map(str::to_string),
        }
    }

    pub fn format_of(&self, path: &Path) -> Option<&Format> {
        self.formats.iter().find(|f| f.matches(path))
    }

    pub fn reads_described(&self) -> String {
        let all: Vec<&str> = self.formats.iter().map(|f| f.describe.as_str()).collect();
        match all.as_slice() {
            [] => "no format: its worker declares none, so it is older than this superfluid (rebuild or update it)".to_string(),
            [one] => one.to_string(),
            [init @ .., last] => format!("{} or {last}", init.join(", ")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Runtime {
    pub id: RuntimeId,
    pub worker: Worker,
    pub version: Option<String>,
    pub capabilities: Option<String>,
    pub info: Info,
    pub devices: Vec<Device>,
    pub tokenizer: Option<PathBuf>,
    pub pull: Option<Vec<(String, bool)>>,
}

impl Runtime {
    pub fn serve_args(&self, model: &Path, ctx: i32, max_batch: u32, kv_bits: i32) -> Vec<String> {
        let mut args = self.worker.prefix.clone();
        args.extend([
            "--model".to_string(),
            model.to_string_lossy().into_owned(),
            "--max-context".to_string(),
            ctx.to_string(),
            "--max-batch".to_string(),
            max_batch.to_string(),
        ]);
        if !matches!(self.static_capabilities().get("load", "kv_bits"), crate::capabilities::Cap::No(_)) {
            args.push("--kv-bits".to_string());
            args.push(kv_bits.to_string());
        }
        args
    }

    pub fn static_capabilities(&self) -> crate::capabilities::Capabilities {
        self.capabilities.as_deref().map(crate::capabilities::Capabilities::parse).unwrap_or_default()
    }

    pub fn in_process_capable(&self) -> bool {
        crate::linked::get(self.id).is_some()
    }

    pub fn device(&self) -> Option<&Device> {
        self.devices.first()
    }
}

#[derive(Debug, Clone)]
pub struct Found {
    pub id: RuntimeId,
    pub worker: Option<Worker>,
    pub info: Option<Info>,
    pub usable: Result<Runtime, String>,
}

pub struct Catalog {
    home: PathBuf,
    beside: Option<PathBuf>,
    workerd: PathBuf,
    usable: Mutex<HashMap<RuntimeId, Runtime>>,
}

pub fn default_home() -> PathBuf {
    home_from(std::env::var_os("SUPERFLUID_HOME"), std::env::home_dir())
}

fn home_from(superfluid_home: Option<std::ffi::OsString>, user_home: Option<PathBuf>) -> PathBuf {
    if let Some(h) = superfluid_home.filter(|h| !h.is_empty()) {
        return PathBuf::from(h);
    }
    user_home.filter(|h| h.is_absolute()).unwrap_or_else(|| PathBuf::from("/")).join(".superfluid")
}

pub fn worker_name(id: RuntimeId) -> String {
    format!("superfluid-worker-{}", id.base().name())
}

pub fn worker_env(id: RuntimeId) -> String {
    format!("SUPERFLUID_WORKER_{}", id.base().name().to_ascii_uppercase().replace(['-', '.'], "_"))
}

impl Catalog {
    pub fn new(home: PathBuf, beside: Option<PathBuf>, workerd: PathBuf) -> Catalog {
        Catalog { home, beside, workerd, usable: Mutex::new(HashMap::new()) }
    }

    pub fn from_env() -> Catalog {
        let beside = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
        let workerd = beside.clone().unwrap_or_default().join("superfluid-workerd");
        Catalog::new(default_home(), beside, workerd)
    }

    /// [`Catalog::from_env`], with `workerd` running the linked runtimes'
    /// workers in place of the `superfluid-workerd` beside this executable.
    pub fn with_workerd(workerd: PathBuf) -> Catalog {
        let beside = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
        Catalog::new(default_home(), beside, workerd)
    }

    pub fn home(&self) -> &Path {
        &self.home
    }

    /// The program that runs a linked runtime's worker.
    pub fn workerd(&self) -> &Path {
        &self.workerd
    }

    pub fn beside(&self) -> Option<&Path> {
        self.beside.as_deref()
    }

    pub fn package_dir(&self, id: RuntimeId) -> PathBuf {
        self.home.join("runtimes").join(id.base().name())
    }

    fn shipped_dirs(&self) -> Vec<PathBuf> {
        let Some(beside) = &self.beside else { return Vec::new() };
        [beside.join("..").join("libexec").join("superfluid"), beside.join("libexec").join("superfluid")]
            .into_iter()
            .filter_map(|d| std::fs::canonicalize(d).ok())
            .filter(|d| d.is_dir())
            .collect()
    }

    pub fn shipped(&self, id: RuntimeId) -> Option<PathBuf> {
        let own = worker_name(id);
        self.shipped_dirs().into_iter().map(|d| d.join(&own)).find(|p| is_executable(p))
    }

    pub fn shipped_ids(&self) -> Vec<RuntimeId> {
        let mut ids: Vec<RuntimeId> = self.shipped_dirs().iter().flat_map(|d| workers_in(d)).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    pub fn ids(&self) -> Vec<RuntimeId> {
        let mut ids: Vec<RuntimeId> = Vec::new();
        for (k, _) in std::env::vars_os() {
            if let Some(id) = k.to_str().and_then(|k| k.strip_prefix("SUPERFLUID_WORKER_")) {
                ids.extend(RuntimeId::parse(&id.to_ascii_lowercase()));
            }
        }
        if let Some(dir) = &self.beside {
            ids.extend(workers_in(dir));
        }
        ids.extend(self.shipped_ids());
        ids.extend(crate::linked::ids());
        if let Ok(rd) = std::fs::read_dir(self.home.join("runtimes")) {
            ids.extend(
                rd.flatten()
                    .filter(|e| e.path().is_dir())
                    .filter_map(|e| e.file_name().to_str().and_then(RuntimeId::parse)),
            );
        }
        ids.sort();
        ids.dedup();
        ids
    }

    pub fn locate(&self, id: RuntimeId) -> Option<Worker> {
        if let Some(install) = id.install() {
            let bin = self.package_dir(id).join(install).join("bin").join(worker_name(id));
            return std::fs::canonicalize(bin).ok().filter(|p| p.is_file()).map(|bin| Worker {
                bin,
                prefix: Vec::new(),
                source: Source::Installed,
            });
        }
        if let Some(p) = std::env::var_os(worker_env(id)) {
            return Some(Worker { bin: PathBuf::from(p), prefix: Vec::new(), source: Source::Env });
        }
        if crate::linked::get(id).is_some() {
            return Some(Worker {
                bin: self.workerd.clone(),
                prefix: vec!["--engine".into(), id.name().into()],
                source: Source::BuiltIn,
            });
        }
        let own = worker_name(id);
        if let Some(bin) = self.beside.as_ref().map(|d| d.join(&own)).filter(|p| is_executable(p)) {
            return Some(Worker { bin, prefix: Vec::new(), source: Source::BesideDaemon });
        }
        let bin = self.package_dir(id).join("current").join("bin").join(&own);
        let installed = std::fs::canonicalize(bin).ok().filter(|p| p.is_file()).map(|bin| Worker {
            bin,
            prefix: Vec::new(),
            source: Source::Installed,
        });
        installed.or_else(|| self.shipped(id).map(|bin| Worker { bin, prefix: Vec::new(), source: Source::Shipped }))
    }

    pub fn require(&self, id: RuntimeId) -> Result<Runtime, String> {
        if let Some(r) = self.remembered(id) {
            return Ok(r);
        }
        let found = self.inspect(id);
        if let Ok(runtime) = &found.usable {
            self.usable.lock().expect("runtime catalog").insert(id, runtime.clone());
        }
        found.usable
    }

    fn remembered(&self, id: RuntimeId) -> Option<Runtime> {
        let mut usable = self.usable.lock().expect("runtime catalog");
        let r = usable.get(&id)?;
        if r.worker.source != Source::BuiltIn && !r.worker.bin.is_file() {
            usable.remove(&id);
            return None;
        }
        Some(r.clone())
    }

    pub fn inspect(&self, id: RuntimeId) -> Found {
        if let Some(r) = self.remembered(id) {
            return Found { id, worker: Some(r.worker.clone()), info: Some(r.info.clone()), usable: Ok(r) };
        }
        let Some(worker) = self.locate(id) else {
            return Found { id, worker: None, info: None, usable: Err(self.missing(id)) };
        };
        let report = if worker.source == Source::BuiltIn && !worker.bin.is_file() {
            crate::linked::report(id, None).ok_or_else(|| format!("this build does not link the {id} runtime"))
        } else {
            run_check(&worker)
        };
        let report = match report {
            Ok(r) => r,
            Err(why) => return Found { id, worker: Some(worker), info: None, usable: Err(why) },
        };
        let info = Info::from_report(&report);
        let usable = runtime_from_check(id, worker.clone(), &report).and_then(|r| self.with_tokenizer(r));
        // A check starts the worker (for MLX, a Python and its imports):
        // run it once per runtime, not once per question asked of it.
        if let Ok(runtime) = &usable {
            self.usable.lock().expect("runtime catalog").insert(id, runtime.clone());
        }
        Found { id, worker: Some(worker), info: Some(info), usable }
    }

    fn with_tokenizer(&self, mut runtime: Runtime) -> Result<Runtime, String> {
        let linked = crate::linked::get(runtime.id).is_some();
        match runtime.info.tokenizer.clone() {
            Some(TokenizerSource::Library(stem)) => {
                runtime.tokenizer = crate::dylib_tokenizer::find_library(&runtime.worker.bin, &stem);
                if runtime.tokenizer.is_none() && !linked {
                    return Err(format!(
                        "the {} runtime at {} ships no tokenizer library ({} beside it or in ../lib)",
                        runtime.id,
                        runtime.worker.bin.display(),
                        crate::dylib_tokenizer::library_file(&stem)
                    ));
                }
            }
            Some(TokenizerSource::HuggingFace) => {}
            Some(TokenizerSource::Engine) if linked => {}
            Some(TokenizerSource::Engine) => {
                return Err(format!(
                    "only the {id} runtime's engine tokenizes its artifacts, and this superfluid does not link it \
                     (a source build adds it with --features {feature})",
                    id = runtime.id,
                    feature = crate::linked::feature_for(runtime.id.base().name()).unwrap_or("<the runtime>"),
                ))
            }
            None => {}
        }
        Ok(runtime)
    }

    fn missing(&self, id: RuntimeId) -> String {
        if let Some(install) = id.install() {
            let have: Vec<String> = crate::installs::list(self, id).into_iter().map(|i| i.name).collect();
            return format!(
                "{} has no install {install} here (installed: {})",
                id.base(),
                if have.is_empty() { "none".to_string() } else { have.join(", ") }
            );
        }
        let shipped = self.shipped_ids();
        if !shipped.is_empty() {
            return format!(
                "the {id} runtime is not here: this superfluid ships no adapter for it (it ships: {}), and none is installed in {}",
                shipped.iter().map(|i| i.name()).collect::<Vec<_>>().join(", "),
                self.package_dir(id).display()
            );
        }
        format!(
            "the {id} runtime is not installed: no {w} in {pkg} or beside superfluid \
             (install it with `superfluid runtime install {id}`, or build {w} with `cargo build -p superfluid-adapter-{id}`)",
            w = worker_name(id),
            pkg = self.package_dir(id).join("current").join("bin").display(),
        )
    }

    pub fn survey(&self) -> Vec<Found> {
        self.ids().into_iter().map(|id| self.inspect(id)).collect()
    }

    pub fn named(&self, name: &str) -> Result<RuntimeId, String> {
        let name = name.trim();
        if let Some((base, install)) = name.split_once('@') {
            let id = self.named(base)?;
            let at = RuntimeId::parse(&format!("{id}@{install}")).ok_or_else(|| format!("'{install}' is not an install's name"))?;
            if crate::installs::list(self, id).iter().any(|i| i.name == install) {
                return Ok(at);
            }
            return Err(self.missing(at));
        }
        let ids = self.ids();
        if let Some(id) = ids.iter().copied().find(|id| id.name() == name) {
            return Ok(id);
        }
        for found in self.survey() {
            if found.info.is_some_and(|i| i.aliases.iter().any(|a| a == name)) {
                return Ok(found.id);
            }
        }
        Err(format!("no runtime '{name}' here"))
    }

    pub fn reads(&self, id: RuntimeId, path: &Path) -> Result<(), String> {
        let found = self.inspect(id);
        let Some(info) = &found.info else {
            return found.usable.map(|_| ());
        };
        if info.format_of(path).is_some() {
            return Ok(());
        }
        let mut why = format!("runtime {id} cannot serve {}: it reads {}", path.display(), info.reads_described());
        match self.readers(path).first() {
            Some((other, format)) => why.push_str(&format!(
                ", and this is {} (pass --runtime {other}, or convert the model)",
                format.describe
            )),
            None => why.push_str(", and no runtime here reads this file"),
        }
        Err(why)
    }

    pub fn readers(&self, path: &Path) -> Vec<(RuntimeId, Format)> {
        let mut known = self.known_readers(path);
        if known.is_empty() {
            let found = self.survey();
            let cannot_run = |id: &RuntimeId| found.iter().any(|f| f.id == *id && f.usable.is_err());
            known = readers_in(&found, path);
            known.sort_by_key(|(id, _)| (cannot_run(id), *id));
            return known;
        }
        known
    }

    fn known_readers(&self, path: &Path) -> Vec<(RuntimeId, Format)> {
        let mut known: Vec<(RuntimeId, Format)> = self
            .usable
            .lock()
            .expect("runtime catalog")
            .values()
            .filter(|r| r.id.install().is_none())
            .filter_map(|r| Some((r.id, r.info.format_of(path)?.clone())))
            .collect();
        known.sort_by_key(|(id, _)| *id);
        known
    }

    pub fn reader_of(&self, path: &Path) -> Result<RuntimeId, String> {
        if let Some((id, _)) = self.known_readers(path).first() {
            return Ok(*id);
        }
        let mut cannot_run = None;
        for id in self.ids() {
            let found = self.inspect(id);
            if found.info.as_ref().and_then(|info| info.format_of(path)).is_none() {
                continue;
            }
            match found.usable {
                Ok(runtime) => {
                    self.usable.lock().expect("runtime catalog").insert(id, runtime);
                    return Ok(id);
                }
                Err(_) => {
                    cannot_run.get_or_insert(id);
                }
            }
        }
        if let Some(id) = cannot_run {
            return Ok(id);
        }
        Err(format!(
            "no runtime here reads {} ({}); `superfluid runtimes {}` says more, and `superfluid runtime install <id>` adds a runtime",
            path.display(),
            self.here_described(),
            path.display()
        ))
    }

    pub fn not_here(&self, name: &str, path: &Path, why: &str) -> String {
        let mut msg = format!("{why} to serve {} ({})", path.display(), self.here_described());
        if let Some((other, format)) = self.readers(path).first() {
            msg.push_str(&format!("; it is {}, which {other} here reads: pass --runtime {other}", format.describe));
        }
        let runtime = name.split('@').next().unwrap_or(name);
        msg.push_str(&format!("; `superfluid runtime install {runtime}` installs {runtime}"));
        msg
    }

    pub fn here_described(&self) -> String {
        let all = self.survey();
        if all.is_empty() {
            return "no runtime is installed".to_string();
        }
        let parts: Vec<String> = all
            .iter()
            .map(|f| {
                let state = if f.usable.is_ok() { "" } else { " (not usable)" };
                match &f.info {
                    Some(info) => format!("{}{state} reads {}", f.id, info.reads_described()),
                    None => format!("{}{state}", f.id),
                }
            })
            .collect();
        format!("runtimes here: {}", parts.join("; "))
    }
}

fn repair_command(id: RuntimeId) -> String {
    match id.install() {
        Some(install) => format!("superfluid runtime repair {} {install}", id.base()),
        None => format!("superfluid runtime repair {id}"),
    }
}

pub fn readers_in(found: &[Found], path: &Path) -> Vec<(RuntimeId, Format)> {
    found
        .iter()
        .filter_map(|f| {
            let format = f.info.as_ref()?.format_of(path)?.clone();
            Some((f.id, format))
        })
        .collect()
}

fn workers_in(dir: &Path) -> Vec<RuntimeId> {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| is_executable(&e.path()))
                .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_prefix("superfluid-worker-")).and_then(RuntimeId::parse))
                .collect()
        })
        .unwrap_or_default()
}

fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

pub fn check_worker(id: RuntimeId, worker: &Worker) -> Result<Runtime, String> {
    let report = run_check(worker)?;
    runtime_from_check(id, worker.clone(), &report)
}

fn run_check(w: &Worker) -> Result<Value, String> {
    run_check_model(w, None)
}

pub fn run_check_model(w: &Worker, model: Option<&Path>) -> Result<Value, String> {
    let mut cmd = Command::new(&w.bin);
    crate::exec::without_secrets(&mut cmd).arg("check").args(&w.prefix);
    if let Some(m) = model {
        cmd.arg("--model").arg(m);
    }
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let child = crate::exec::when_not_busy(|| cmd.spawn()).map_err(|e| format!("{} could not start: {e}", w.bin.display()))?;
    let pid = child.id() as i32;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });
    let out = match rx.recv_timeout(CHECK_TIMEOUT) {
        Ok(out) => out.map_err(|e| format!("{} check: {e}", w.bin.display()))?,
        Err(_) => {
            kill_hard(pid);
            return Err(format!("{} check did not answer within {}s", w.bin.display(), CHECK_TIMEOUT.as_secs()));
        }
    };
    serde_json::from_slice::<Value>(&out.stdout).map_err(|_| {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(3).collect();
        format!(
            "{} check failed ({}){}",
            w.bin.display(),
            out.status,
            if tail.is_empty() {
                String::new()
            } else {
                format!(": {}", tail.into_iter().rev().collect::<Vec<_>>().join(" / "))
            }
        )
    })
}

pub fn sizing_of(rt: &Runtime, model: &Path) -> Result<superfluid_adapter_kit::Sizing, String> {
    let report = run_check_model(&rt.worker, Some(model))?;
    match &report["model"]["sizing"] {
        Value::Null => Err(format!("the {} runtime sizes no context window here", rt.id.base().name())),
        Value::String(why) => Err(why.clone()),
        said => superfluid_adapter_kit::Sizing::from_json(said)
            .ok_or_else(|| format!("{} check reported a sizing this superfluid does not read: {said}", rt.worker.bin.display())),
    }
}

fn runtime_from_check(id: RuntimeId, worker: Worker, report: &Value) -> Result<Runtime, String> {
    let reported = report["runtime"]["id"].as_str().unwrap_or_default();
    if reported != id.base().name() {
        return Err(format!("{} is not the {id} runtime's worker (it reports '{reported}')", worker.bin.display()));
    }
    let speaks: Vec<u64> = report["link_w"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
    let ours = u64::from(superfluid_agent::client::PROTO_VERSION);
    if !speaks.contains(&ours) {
        let fix = if worker.source == Source::Installed && speaks.iter().all(|v| *v < ours) {
            format!("`{}` installs it again with this superfluid's worker", repair_command(id))
        } else {
            "update the one that is older".to_string()
        };
        return Err(format!(
            "the {id} runtime at {} speaks Link W {speaks:?} and this superfluid speaks {ours}: {fix}",
            worker.bin.display()
        ));
    }
    match &report["available"] {
        Value::Bool(true) => {}
        Value::String(why) if worker.source == Source::Shipped => {
            return Err(format!("the {id} runtime is not installed: {why}"));
        }
        Value::String(why) => {
            return Err(format!(
                "the {id} runtime at {} cannot run here: {why}{}",
                worker.bin.display(),
                if worker.source == Source::Installed {
                    format!(" (`{}` reinstalls it)", repair_command(id))
                } else {
                    String::new()
                }
            ))
        }
        other => return Err(format!("{} check reported availability {other}", worker.bin.display())),
    }
    Ok(Runtime {
        id,
        version: report["runtime"]["version"].as_str().map(str::to_string),
        capabilities: report.get("capabilities").filter(|c| c.as_object().is_some_and(|o| !o.is_empty())).map(Value::to_string),
        info: Info::from_report(report),
        devices: superfluid_worker::check::devices_from(&report["devices"]),
        tokenizer: None,
        pull: report.get("pull").map(|p| {
            p["options"]
                .as_array()
                .map(|a| a.iter().filter_map(|o| Some((o["name"].as_str()?.to_string(), o["value"].as_bool().unwrap_or(false)))).collect())
                .unwrap_or_default()
        }),
        worker,
    })
}

fn kill_hard(pid: i32) {
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    // SAFETY: plain POSIX kill on a child this process spawned.
    unsafe {
        kill(pid, 9);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("superfluid-runtimes-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    pub(crate) fn fake_worker(dir: &Path, name: &str, report: &str, exit: i32) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\ncat <<'EOF'\n{report}\nEOF\nexit {exit}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    pub(crate) fn report(id: &str, link_w: u16, available: &str) -> String {
        let (aliases, formats, tokenizer) = match id {
            "llamacpp" => (
                r#"["llama.cpp"]"#,
                r#"[{"id":"gguf","describe":"a GGUF file","file":{"extensions":["gguf"],"magic":"GGUF"}}]"#,
                r#"{"from":"library","name":"superfluid_tokenizer_llamacpp"}"#,
            ),
            "mlx" => (
                "[]",
                r#"[{"id":"mlx","describe":"an MLX model directory","directory":{"files":["config.json"],"extensions":["safetensors"]}}]"#,
                r#"{"from":"huggingface"}"#,
            ),
            _ => (
                r#"["native"]"#,
                r#"[{"id":"base","describe":"a .base bundle","file":{"extensions":["base"],"magic":"BASE"}}]"#,
                r#"{"from":"engine"}"#,
            ),
        };
        format!(
            r#"{{"runtime":{{"id":"{id}","version":"v1","aliases":{aliases}}},"link_w":[{link_w}],"available":{available},"formats":{formats},"tokenizer":{tokenizer},"devices":[{{"backend":"Metal","name":"Apple M5 Pro","memory":36000000000}}],"capabilities":{{"descriptor_version":1,"load":{{"kv_bits":"no such knob"}},"serving":{{"park_lossy":"no lossy encoding"}}}}}}"#
        )
    }

    const V: u16 = superfluid_agent::client::PROTO_VERSION;

    pub(crate) fn toy_report(id: &str, available: &str) -> String {
        format!(
            r#"{{"runtime":{{"id":"{id}","version":"v1"}},"link_w":[{V}],"available":{available},"formats":[{{"id":"toy","describe":"a toy file","file":{{"extensions":["toy"]}}}}],"tokenizer":{{"from":"huggingface"}}}}"#
        )
    }

    fn catalog(home: &Path) -> Catalog {
        Catalog::new(home.to_path_buf(), None, home.join("superfluid-workerd"))
    }

    fn install_fake(home: &Path, id: &str, report_text: &str, exit: i32) -> PathBuf {
        let bin_dir = home.join("runtimes").join(id).join("current/bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        fake_worker(&bin_dir, &format!("superfluid-worker-{id}"), report_text, exit)
    }

    #[test]
    fn a_models_sizing_is_read_from_its_runtimes_check() {
        let dir = scratch("sizing");
        let with_model = |model: &str| {
            let r = report("llamacpp", V, "true");
            format!("{},\"model\":{model}}}", &r[..r.len() - 1])
        };
        let rt = |name: &str, text: &str| Runtime {
            id: RuntimeId::new("llamacpp"),
            worker: Worker { bin: fake_worker(&dir, name, text, 0), prefix: Vec::new(), source: Source::Env },
            version: None,
            capabilities: None,
            info: Info::default(),
            devices: Vec::new(),
            tokenizer: None,
            pull: None,
        };
        let model = Path::new("/m.gguf");
        let figures = r#"{"budget_bytes":34000000000,"weight_bytes":600000000,"kv_bytes_per_token":114688,"trained_context":40960}"#;
        let sized = rt("sized", &with_model(&format!(r#"{{"path":"/m.gguf","reads":true,"sizing":{figures}}}"#)));
        assert_eq!(
            sizing_of(&sized, model),
            Ok(superfluid_adapter_kit::Sizing {
                budget_bytes: 34_000_000_000,
                weight_bytes: 600_000_000,
                kv_bytes_per_token: 114_688,
                trained_context: 40_960
            })
        );
        let declined = rt(
            "declined",
            &with_model(r#"{"path":"/m.gguf","reads":true,"sizing":"the llamacpp runtime sizes no window on Vulkan yet"}"#),
        );
        assert_eq!(sizing_of(&declined, model), Err("the llamacpp runtime sizes no window on Vulkan yet".to_string()));
        let silent = rt("silent", &with_model(r#"{"path":"/m.gguf","reads":true}"#));
        assert_eq!(sizing_of(&silent, model), Err("the llamacpp runtime sizes no context window here".to_string()));
        let garbled = rt("garbled", &with_model(r#"{"path":"/m.gguf","reads":true,"sizing":{"budget_bytes":1}}"#));
        assert!(sizing_of(&garbled, model).unwrap_err().contains("reported a sizing this superfluid does not read"));
    }

    #[test]
    fn an_installed_package_is_found_checked_and_remembered() {
        let llamacpp = RuntimeId::new("llamacpp");
        if crate::linked::get(llamacpp).is_some() {
            return;
        }
        let home = scratch("installed");
        let w = install_fake(&home, "llamacpp", &report("llamacpp", V, "true"), 0);
        let cat = catalog(&home);
        let found: Vec<RuntimeId> = cat.ids().into_iter().filter(|i| crate::linked::get(*i).is_none()).collect();
        assert_eq!(found, vec![llamacpp]);
        let e = cat.require(llamacpp).unwrap_err();
        assert!(e.ends_with("ships no tokenizer library (libsuperfluid_tokenizer_llamacpp.dylib beside it or in ../lib)")
            || e.ends_with("ships no tokenizer library (libsuperfluid_tokenizer_llamacpp.so beside it or in ../lib)"), "{e}");
        let lib_dir = home.join("runtimes/llamacpp/current/lib");
        std::fs::create_dir_all(&lib_dir).unwrap();
        let lib = lib_dir.join(crate::dylib_tokenizer::library_file("superfluid_tokenizer_llamacpp"));
        std::fs::write(&lib, b"").unwrap();
        let rt = cat.require(llamacpp).unwrap();
        let pinned = w.canonicalize().unwrap();
        assert_eq!(rt.worker, Worker { bin: pinned, prefix: vec![], source: Source::Installed });
        assert_eq!(rt.version.as_deref(), Some("v1"));
        assert_eq!(rt.info.aliases, ["llama.cpp"]);
        assert_eq!(rt.device().unwrap().summary(), "Metal: Apple M5 Pro, 36.0 GB");
        assert_eq!(rt.tokenizer.as_deref().map(|p| p.canonicalize().unwrap()), Some(lib.canonicalize().unwrap()));
        assert_eq!(rt.static_capabilities().check_startup_flags(true, None, None).unwrap_err(), "--park-lossy: no lossy encoding");
        let args = rt.serve_args(Path::new("/m.gguf"), 4096, 4, 8);
        assert_eq!(args, ["--model", "/m.gguf", "--max-context", "4096", "--max-batch", "4"], "its record refuses --kv-bits");
    }

    #[test]
    fn the_home_is_never_relative_to_the_working_directory() {
        assert_eq!(home_from(Some("/srv/superfluid".into()), Some("/home/u".into())), PathBuf::from("/srv/superfluid"));
        assert_eq!(home_from(None, Some("/home/u".into())), PathBuf::from("/home/u/.superfluid"));
        assert_eq!(home_from(Some("".into()), Some("/home/u".into())), PathBuf::from("/home/u/.superfluid"), "an empty variable is unset");
        assert_eq!(home_from(None, None), PathBuf::from("/.superfluid"));
        assert_eq!(home_from(None, Some("".into())), PathBuf::from("/.superfluid"));
        assert_eq!(home_from(None, Some(".".into())), PathBuf::from("/.superfluid"));
    }

    fn counting_worker(dir: &Path, name: &str, report: &str, tally: &Path) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\necho run >> '{}'\ncat <<'EOF'\n{report}\nEOF\n", tally.display())).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    fn runs(tally: &Path) -> usize {
        std::fs::read_to_string(tally).map(|t| t.lines().count()).unwrap_or(0)
    }

    #[test]
    fn a_remembered_runtime_whose_worker_is_gone_is_looked_for_again() {
        let home = scratch("regone");
        let toyrt = RuntimeId::new("toyrt");
        let tally = home.join("tally");
        let pkg = home.join("runtimes/toyrt");
        counting_worker(&pkg.join("b1/bin"), "superfluid-worker-toyrt", &toy_report("toyrt", "true"), &tally);
        std::os::unix::fs::symlink("b1", pkg.join("current")).unwrap();
        let cat = catalog(&home);
        let first = cat.require(toyrt).unwrap();
        assert!(cat.require(toyrt).is_ok());
        assert_eq!(runs(&tally), 1, "a usable runtime is checked once");
        counting_worker(&pkg.join("b2/bin"), "superfluid-worker-toyrt", &toy_report("toyrt", "true"), &tally);
        std::fs::remove_file(pkg.join("current")).unwrap();
        std::os::unix::fs::symlink("b2", pkg.join("current")).unwrap();
        assert_eq!(cat.require(toyrt).unwrap().worker, first.worker);
        assert_eq!(runs(&tally), 1);
        std::fs::remove_dir_all(pkg.join("b1")).unwrap();
        let again = cat.require(toyrt).unwrap();
        assert_eq!(again.worker.bin, pkg.join("b2/bin/superfluid-worker-toyrt").canonicalize().unwrap());
        assert_eq!(runs(&tally), 2);
        std::fs::remove_dir_all(pkg.join("b2")).unwrap();
        let e = cat.require(toyrt).unwrap_err();
        assert!(e.starts_with("the toyrt runtime is not installed"), "{e}");
    }

    #[test]
    fn a_model_asks_no_runtime_past_the_first_that_serves_it() {
        let home = scratch("lazy");
        let (aart, zzrt) = (RuntimeId::new("aart"), RuntimeId::new("zzrt"));
        let (asked_a, asked_z) = (home.join("asked-a"), home.join("asked-z"));
        counting_worker(&home.join("runtimes/aart/current/bin"), "superfluid-worker-aart", &toy_report("aart", "true"), &asked_a);
        let reads_zz = toy_report("zzrt", "true").replace(r#""extensions":["toy"]"#, r#""extensions":["zz"]"#);
        counting_worker(&home.join("runtimes/zzrt/current/bin"), "superfluid-worker-zzrt", &reads_zz, &asked_z);
        let cat = catalog(&home);
        let toy = home.join("m.toy");
        std::fs::write(&toy, b"toy").unwrap();
        assert_eq!(cat.reader_of(&toy).unwrap(), aart);
        assert_eq!((runs(&asked_a), runs(&asked_z)), (1, 0), "the runtime after the reader was not run");
        assert!(cat.require(aart).is_ok());
        assert_eq!(runs(&asked_a), 1, "and the reader was not checked a second time");
        let zz = home.join("m.zz");
        std::fs::write(&zz, b"zz").unwrap();
        assert_eq!(cat.reader_of(&zz).unwrap(), zzrt);
        assert_eq!(runs(&asked_z), 1);
    }

    #[test]
    fn a_pick_of_one_install_leaves_later_models_on_the_default() {
        let home = scratch("pinned-reader");
        let toyrt = RuntimeId::new("toyrt");
        let pkg = home.join("runtimes/toyrt");
        for b in ["b1", "b2"] {
            let bin = pkg.join(b).join("bin");
            std::fs::create_dir_all(&bin).unwrap();
            fake_worker(&bin, "superfluid-worker-toyrt", &toy_report("toyrt", "true"), 0);
        }
        std::os::unix::fs::symlink("b2", pkg.join("current")).unwrap();
        let cat = catalog(&home);
        let pinned = cat.require(RuntimeId::parse("toyrt@b1").unwrap()).unwrap();
        assert_eq!(pinned.worker.bin, pkg.join("b1/bin/superfluid-worker-toyrt").canonicalize().unwrap());
        let model = home.join("m.toy");
        std::fs::write(&model, b"toy").unwrap();
        assert_eq!(cat.reader_of(&model).unwrap(), toyrt, "the runtime, not the install another model picked");
        assert_eq!(cat.readers(&model).into_iter().map(|(id, _)| id).collect::<Vec<_>>(), [toyrt]);
        let served = cat.require(toyrt).unwrap();
        assert_eq!(served.worker.bin, pkg.join("b2/bin/superfluid-worker-toyrt").canonicalize().unwrap());
    }

    #[test]
    fn an_adapter_a_release_ships_is_known_before_its_runtime_is_installed() {
        for flat in [false, true] {
            let root = scratch(if flat { "shipped-flat" } else { "shipped-sdk" });
            let beside = if flat { root.clone() } else { root.join("bin") };
            let libexec = root.join("libexec").join("superfluid");
            std::fs::create_dir_all(&beside).unwrap();
            std::fs::create_dir_all(&libexec).unwrap();
            let not_found = r#""toyrt not found: install it with `superfluid runtime install toyrt`""#;
            let adapter = fake_worker(&libexec, "superfluid-worker-toyrt", &toy_report("toyrt", not_found), 1).canonicalize().unwrap();
            std::fs::write(libexec.join("libsuperfluid_tokenizer_toyrt.dylib"), b"").unwrap();
            let home = root.join("home");
            let catalog = || Catalog::new(home.clone(), Some(beside.clone()), beside.join("superfluid-workerd"));
            let (toyrt, vllm) = (RuntimeId::new("toyrt"), RuntimeId::new("vllm"));

            let cat = catalog();
            assert_eq!(cat.shipped_ids(), vec![toyrt], "flat: {flat}");
            assert!(cat.ids().contains(&toyrt));
            assert_eq!(cat.named("toyrt").unwrap(), toyrt);
            assert_eq!(cat.locate(toyrt).unwrap(), Worker { bin: adapter.clone(), prefix: vec![], source: Source::Shipped });
            assert_eq!(
                cat.require(toyrt).unwrap_err(),
                "the toyrt runtime is not installed: toyrt not found: install it with `superfluid runtime install toyrt`"
            );
            let model = root.join("m.toy");
            std::fs::write(&model, b"toy").unwrap();
            assert_eq!(cat.reader_of(&model).unwrap(), toyrt);
            assert_eq!(cat.reads(toyrt, &model), Ok(()));
            assert_eq!(crate::installs::installer(&cat, toyrt).unwrap(), adapter);
            let e = cat.require(vllm).unwrap_err();
            assert!(e.starts_with("the vllm runtime is not here: this superfluid ships no adapter for it (it ships: toyrt)"), "{e}");
            assert_eq!(
                crate::installs::installer(&cat, vllm).unwrap_err(),
                "this superfluid ships no adapter to install the vllm runtime with (it ships: toyrt)"
            );

            let installed = install_fake(&home, "toyrt", &toy_report("toyrt", "true"), 0).canonicalize().unwrap();
            let cat = catalog();
            assert_eq!(cat.locate(toyrt).unwrap(), Worker { bin: installed, prefix: vec![], source: Source::Installed });
            assert!(cat.require(toyrt).is_ok());
            assert_eq!(crate::installs::installer(&cat, toyrt).unwrap(), adapter);
        }
    }

    #[test]
    fn a_source_build_ships_no_adapter() {
        let beside = scratch("no-libexec");
        let cat = Catalog::new(scratch("no-libexec-home"), Some(beside), PathBuf::from("/nonexistent/superfluid-workerd"));
        let vllm = RuntimeId::new("vllm");
        assert!(cat.shipped_ids().is_empty());
        assert_eq!(cat.shipped(vllm), None);
        let e = cat.require(vllm).unwrap_err();
        assert!(e.contains("cargo build -p superfluid-adapter-vllm"), "{e}");
        let e = crate::installs::installer(&cat, vllm).unwrap_err();
        assert!(e.starts_with("no superfluid-worker-vllm to install the vllm runtime with") && e.contains("cargo build"), "{e}");
    }

    #[test]
    fn a_worker_beside_superfluid_wins_over_an_installed_one() {
        let vllm = RuntimeId::new("vllm");
        let home = scratch("beside-home");
        let beside = scratch("beside-exe");
        install_fake(&home, "vllm", &report("vllm", V, "true"), 0);
        let mine = fake_worker(&beside, "superfluid-worker-vllm", &report("vllm", V, "true"), 0);
        std::fs::write(beside.join("superfluid-worker-vllm.d"), b"").unwrap();
        let cat = Catalog::new(home, Some(beside), PathBuf::from("/nonexistent/superfluid-workerd"));
        assert_eq!(cat.locate(vllm).unwrap(), Worker { bin: mine, prefix: vec![], source: Source::BesideDaemon });
        assert!(cat.ids().contains(&vllm) && !cat.ids().iter().any(|i| i.name().ends_with(".d")));
    }

    #[test]
    fn a_runtime_this_build_links_is_served_by_this_build() {
        let beside = scratch("linked-beside");
        let workerd = beside.join("superfluid-workerd");
        for id in crate::linked::ids() {
            fake_worker(&beside, &worker_name(id), &report(id.name(), V, "true"), 0);
            let cat = Catalog::new(scratch("linked-home"), Some(beside.clone()), workerd.clone());
            let worker = cat.locate(id).unwrap();
            assert_eq!(worker, Worker { bin: workerd.clone(), prefix: vec!["--engine".into(), id.name().into()], source: Source::BuiltIn });
        }
    }

    #[test]
    fn a_program_named_as_the_workerd_is_started_as_one() {
        let program = scratch("embedder").join("embedding-program");
        let cat = Catalog::with_workerd(program.clone());
        assert_eq!(cat.workerd(), program.as_path());
        for id in crate::linked::ids() {
            let worker = cat.locate(id).unwrap();
            assert_eq!(worker.bin, program);
            // The daemon serves a model with the prefix first, and checks the
            // runtime with `check` before it: the shapes the program must tell
            // from its own command line.
            let serve: Vec<String> = worker.prefix.iter().cloned().chain(["--model".into(), "m".into()]).collect();
            assert!(crate::workerd::is_invocation(&serve), "{serve:?}");
            let check: Vec<String> = std::iter::once("check".to_string()).chain(worker.prefix.iter().cloned()).collect();
            assert!(crate::workerd::is_invocation(&check), "{check:?}");
        }
    }

    #[test]
    fn what_is_missing_or_broken_says_why_and_how_to_fix_it() {
        let mlx = RuntimeId::new("mlx");
        if crate::linked::get(mlx).is_some() {
            return;
        }
        let home = scratch("broken");
        let cat = catalog(&home);
        let e = cat.require(mlx).unwrap_err();
        assert!(e.starts_with("the mlx runtime is not installed: no superfluid-worker-mlx in "), "{e}");
        assert!(e.contains("`superfluid runtime install mlx`") && e.contains("cargo build -p superfluid-adapter-mlx"), "{e}");

        let w = install_fake(&home, "mlx", &report("mlx", V, r#""ModuleNotFoundError: No module named 'mlx'""#), 1);
        let e = cat.require(mlx).unwrap_err();
        let pinned = w.canonicalize().unwrap();
        assert_eq!(
            e,
            format!(
                "the mlx runtime at {} cannot run here: ModuleNotFoundError: No module named 'mlx' (`superfluid runtime repair mlx` reinstalls it)",
                pinned.display()
            )
        );
        let found = cat.inspect(mlx);
        assert_eq!(found.info.unwrap().formats[0].id, "mlx");

        install_fake(&home, "mlx", &report("mlx", V + 1, "true"), 0);
        let e = cat.require(mlx).unwrap_err();
        assert!(e.contains(&format!("speaks Link W [{}] and this superfluid speaks {V}", V + 1)), "{e}");
        assert!(e.ends_with("update the one that is older"), "a newer worker: superfluid is the older one: {e}");
        install_fake(&home, "mlx", &report("mlx", V - 1, "true"), 0);
        let e = cat.require(mlx).unwrap_err();
        assert!(e.contains(&format!("speaks Link W [{}] and this superfluid speaks {V}", V - 1)), "{e}");
        assert!(e.ends_with("`superfluid runtime repair mlx` installs it again with this superfluid's worker"), "{e}");
        let old = home.join("runtimes/mlx/old/bin");
        std::fs::create_dir_all(&old).unwrap();
        fake_worker(&old, "superfluid-worker-mlx", &report("mlx", V - 1, "true"), 0);
        let e = cat.require(RuntimeId::parse("mlx@old").unwrap()).unwrap_err();
        assert!(e.ends_with("`superfluid runtime repair mlx old` installs it again with this superfluid's worker"), "{e}");
        fake_worker(&old, "superfluid-worker-mlx", &report("mlx", V, r#""no Metal device""#), 1);
        let e = cat.require(RuntimeId::parse("mlx@old").unwrap()).unwrap_err();
        assert!(e.ends_with("cannot run here: no Metal device (`superfluid runtime repair mlx old` reinstalls it)"), "{e}");
        let e = cat.not_here("mlx@gone", Path::new("/m"), "mlx has no install gone here");
        assert!(e.ends_with("`superfluid runtime install mlx` installs mlx"), "{e}");

        install_fake(&home, "mlx", &report("llamacpp", V, "true"), 0);
        assert!(cat.require(mlx).unwrap_err().contains("is not the mlx runtime's worker"));

        std::fs::write(&w, "#!/bin/sh\necho 'dyld: Library not loaded: libpython3.12.dylib' >&2\nexit 134\n").unwrap();
        let e = cat.require(mlx).unwrap_err();
        assert!(e.contains("check failed") && e.ends_with("dyld: Library not loaded: libpython3.12.dylib"), "{e}");

        install_fake(&home, "mlx", &report("mlx", V, "true"), 0);
        assert_eq!(cat.require(mlx).unwrap().version.as_deref(), Some("v1"));
    }

    #[test]
    fn a_worker_that_declares_no_formats_is_named_as_older() {
        let old = RuntimeId::new("oldrt");
        let home = scratch("old");
        install_fake(&home, "oldrt", &format!(r#"{{"runtime":{{"id":"oldrt","version":"v0"}},"link_w":[{V}],"available":true}}"#), 0);
        let cat = catalog(&home);
        assert!(cat.inspect(old).info.unwrap().formats.is_empty());
        let file = scratch("old-model").join("m.weights");
        std::fs::write(&file, b"????").unwrap();
        let e = cat.reader_of(&file).unwrap_err();
        assert!(e.contains("oldrt reads no format: its worker declares none, so it is older than this superfluid (rebuild or update it)"), "{e}");
    }

    #[test]
    fn a_model_two_runtimes_read_goes_to_the_one_that_can_run() {
        let (broken, working) = (RuntimeId::new("abroken"), RuntimeId::new("zworking"));
        let toy = |id: &str, available: &str| {
            format!(
                r#"{{"runtime":{{"id":"{id}","version":"v1"}},"link_w":[{V}],"available":{available},"formats":[{{"id":"toy","describe":"a toy file","file":{{"extensions":["toy"]}}}}],"tokenizer":{{"from":"huggingface"}}}}"#
            )
        };
        let home = scratch("two-readers");
        install_fake(&home, "abroken", &toy("abroken", r#""library missing""#), 1);
        install_fake(&home, "zworking", &toy("zworking", "true"), 0);
        let file = scratch("two-readers-model").join("m.toy");
        std::fs::write(&file, b"toy").unwrap();

        let cat = catalog(&home);
        let listed: Vec<RuntimeId> = cat.readers(&file).into_iter().map(|(id, _)| id).collect();
        assert_eq!(listed, vec![working, broken], "the one that cannot run stays listed, after the one that can");
        assert_eq!(cat.reader_of(&file).unwrap(), working, "the reader that can run, not the first by name");
        assert_eq!(crate::runtime_pick::RuntimePick::default().resolve(&file, &cat).unwrap(), working);
        assert!(cat.require(working).is_ok());
        let fresh = catalog(&home);
        let e = fresh.not_here("vllm", &file, "no runtime 'vllm' here");
        assert!(e.contains("which zworking here reads: pass --runtime zworking"), "{e}");

        let alone = scratch("one-broken-reader");
        install_fake(&alone, "abroken", &toy("abroken", r#""library missing""#), 1);
        let cat = catalog(&alone);
        assert_eq!(cat.reader_of(&file).unwrap(), broken);
        let e = cat.require(broken).unwrap_err();
        assert!(e.contains("cannot run here: library missing"), "{e}");
    }

    #[test]
    fn a_runtime_is_named_by_its_id_or_an_alias_it_declares() {
        if crate::linked::get(RuntimeId::new("llamacpp")).is_some() {
            return;
        }
        let home = scratch("named");
        install_fake(&home, "llamacpp", &report("llamacpp", V, "true"), 0);
        let cat = catalog(&home);
        assert_eq!(cat.named("llamacpp").unwrap(), RuntimeId::new("llamacpp"));
        assert_eq!(cat.named("llama.cpp").unwrap(), RuntimeId::new("llamacpp"), "an alias the runtime declares");
        assert_eq!(cat.named("vllm").unwrap_err(), "no runtime 'vllm' here");
    }

    #[test]
    fn the_runtime_for_a_model_is_the_one_whose_format_it_is() {
        let (llamacpp, mlx) = (RuntimeId::new("llamacpp"), RuntimeId::new("mlx"));
        if crate::linked::get(llamacpp).is_some() || crate::linked::get(mlx).is_some() {
            return;
        }
        let home = scratch("formats");
        install_fake(&home, "llamacpp", &report("llamacpp", V, "true"), 0);
        let lib = home.join("runtimes/llamacpp/current/lib");
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::write(lib.join(crate::dylib_tokenizer::library_file("superfluid_tokenizer_llamacpp")), b"").unwrap();
        install_fake(&home, "mlx", &report("mlx", V, "true"), 0);
        let cat = catalog(&home);
        let models = scratch("formats-models");
        let gguf = models.join("weights.bin");
        std::fs::write(&gguf, b"GGUF\x03\x00").unwrap();
        let dir = models.join("qwen-mlx");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();
        let notes = models.join("notes.txt");
        std::fs::write(&notes, b"hello").unwrap();

        assert_eq!(cat.reader_of(&gguf).unwrap(), llamacpp, "by the magic its format declares");
        assert_eq!(cat.reader_of(&dir).unwrap(), mlx);
        let e = cat.reader_of(&notes).unwrap_err();
        assert!(e.starts_with(&format!("no runtime here reads {}", notes.display())), "{e}");
        assert!(e.contains("llamacpp reads a GGUF file; mlx reads an MLX model directory"), "{e}");
        assert!(e.contains("`superfluid runtime install <id>`"), "{e}");

        let e = cat.reads(mlx, &gguf).unwrap_err();
        assert_eq!(
            e,
            format!(
                "runtime mlx cannot serve {}: it reads an MLX model directory, and this is a GGUF file (pass --runtime llamacpp, or convert the model)",
                gguf.display()
            )
        );
        let mut pick = crate::runtime_pick::RuntimePick::default();
        pick.add("vllm").unwrap();
        let e = pick.resolve(&gguf, &cat).unwrap_err();
        assert!(e.starts_with(&format!("no runtime 'vllm' here to serve {}", gguf.display())), "{e}");
        assert!(e.contains("it is a GGUF file, which llamacpp here reads: pass --runtime llamacpp"), "{e}");
        assert!(e.ends_with("`superfluid runtime install vllm` installs vllm"), "{e}");
        assert_eq!(crate::runtime_pick::RuntimePick::default().resolve(&dir, &cat).unwrap(), mlx);
        let mut pick = crate::runtime_pick::RuntimePick::default();
        pick.add("llama.cpp").unwrap();
        assert_eq!(pick.resolve(&gguf, &cat).unwrap(), llamacpp);
    }
}
