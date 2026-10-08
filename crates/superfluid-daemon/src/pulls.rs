//! Models by id.

use std::path::{Path, PathBuf};

use superfluid_adapter_kit::Pull;

use crate::runtime_pick::RuntimeId;
use crate::runtimes::{Catalog, Source};

pub fn looks_like_id(token: &str) -> bool {
    if token.is_empty() || Path::new(token).exists() {
        return false;
    }
    let file = token.to_ascii_lowercase();
    if token.starts_with(['.', '/', '~']) || file.ends_with(".base") || file.ends_with(".gguf") {
        return false;
    }
    let id = token.split_once(':').map_or(token, |(id, _)| id);
    id.contains('/')
}

pub fn served_name(id: &str) -> String {
    id.split_once(':').map_or(id, |(id, _)| id).to_string()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PullFlags {
    pub offline: bool,
    pub options: Vec<(String, Option<String>)>,
}

impl PullFlags {
    fn check(&self, runtime: RuntimeId, declared: &[(String, bool)]) -> Result<(), String> {
        for (name, value) in &self.options {
            let Some((_, takes)) = declared.iter().find(|(n, _)| n == name) else {
                let all: Vec<String> = declared.iter().map(|(n, _)| format!("--pull-{n}")).collect();
                return Err(format!(
                    "--pull-{name}: the {} runtime's pull takes no such option (it takes: {})",
                    runtime.base(),
                    if all.is_empty() { "none".to_string() } else { all.join(", ") }
                ));
            };
            match (takes, value) {
                (true, None) => return Err(format!("--pull-{name} needs a value")),
                (false, Some(v)) => return Err(format!("--pull-{name} takes no value (got {v:?})")),
                _ => {}
            }
        }
        Ok(())
    }
}

/// The runtime, installed first when it never was (and `offline` does not forbid it), as
/// `superfluid runtime install <id>` would.
pub fn ensure_installed(catalog: &Catalog, runtime: RuntimeId, offline: bool) -> Result<crate::runtimes::Runtime, String> {
    match catalog.require(runtime) {
        Ok(rt) => Ok(rt),
        Err(why) if !never_installed(catalog, runtime) => Err(why),
        Err(why) if offline => Err(format!("{why} (offline: nothing was installed)")),
        Err(why) => match named_by_the_operator(catalog, runtime) {
            Some(by) => Err(format!("{why} ({by} names the runtime to load, so none was installed)")),
            None => {
                install(catalog, runtime).map_err(|e| format!("{why}; and installing it failed: {e}"))?;
                catalog.require(runtime)
            }
        },
    }
}

pub fn pull(catalog: &Catalog, runtime: RuntimeId, id: &str, flags: &PullFlags) -> Result<PathBuf, String> {
    let rt = ensure_installed(catalog, runtime, flags.offline)?;
    let Some(declared) = &rt.pull else {
        return Err(format!("the {runtime} runtime pulls no model by id: give --model a path to one of its models"));
    };
    flags.check(runtime, declared)?;
    let request = Pull { id: id.to_string(), offline: flags.offline, options: flags.options.clone() };
    let _one_at_a_time = Lock::take(&catalog.home().join("pulls"), id, &format!("pull of {id}"))?;
    eprintln!("superfluid: pulling {id} with the {runtime} runtime{}", if flags.offline { " (offline)" } else { "" });
    let path = if rt.worker.source == Source::BuiltIn {
        let linked = crate::linked::get(runtime.base()).ok_or_else(|| format!("this build does not link the {runtime} runtime"))?;
        linked.pull(&request, &superfluid_adapter_kit::WorkerArgs::default())?
    } else {
        let mut cmd = std::process::Command::new(&rt.worker.bin);
        cmd.arg("pull").args(request.to_args()).stdin(std::process::Stdio::null()).stderr(std::process::Stdio::inherit());
        let out = crate::exec::when_not_busy(|| cmd.output()).map_err(|e| format!("{} could not start: {e}", rt.worker.bin.display()))?;
        if !out.status.success() {
            return Err(format!("pulling {id} with the {runtime} runtime failed (its reason is above)"));
        }
        let v: serde_json::Value =
            serde_json::from_slice(&out.stdout).map_err(|e| format!("{} pull printed no path: {e}", rt.worker.bin.display()))?;
        PathBuf::from(v["path"].as_str().ok_or_else(|| format!("{} pull printed no path", rt.worker.bin.display()))?)
    };
    if !path.exists() {
        return Err(format!("pulling {id} gave {}, which is not there", path.display()));
    }
    eprintln!("superfluid: {id} is {}", path.display());
    Ok(path)
}

fn never_installed(catalog: &Catalog, runtime: RuntimeId) -> bool {
    if runtime.install().is_some() {
        return false;
    }
    match catalog.locate(runtime) {
        None => true,
        Some(worker) if worker.source == Source::Env => false,
        Some(_) => crate::installs::list(catalog, runtime).is_empty(),
    }
}

fn named_by_the_operator(catalog: &Catalog, runtime: RuntimeId) -> Option<String> {
    catalog.inspect(runtime).info.and_then(|info| info.named)
}

fn install(catalog: &Catalog, runtime: RuntimeId) -> Result<(), String> {
    let id = runtime.base();
    eprintln!("superfluid: the {id} runtime is not here: installing it (`superfluid runtime install {id}`)");
    let installer = crate::installs::installer(catalog, id)?;
    let plans = crate::installs::plan(&installer, &superfluid_adapter_kit::install::Request::default())?;
    let done = crate::installs::install(catalog, id, &installer, &plans.plans)?;
    eprintln!("superfluid: installed {id} {} ({})", done.install.name, done.version);
    Ok(())
}

pub(crate) struct Lock(#[allow(dead_code)] std::fs::File);

impl Lock {
    pub(crate) fn take(dir: &Path, id: &str, what: &str) -> Result<Lock, String> {
        use std::os::unix::io::AsRawFd;
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let name: String = id.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' }).collect();
        let path = dir.join(format!("{name}.lock"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: flock on a descriptor this process owns; the lock goes
        // with the file when it closes.
        unsafe {
            if libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) != 0 {
                eprintln!("superfluid: another {what} is running; waiting for it");
                if libc::flock(file.as_raw_fd(), libc::LOCK_EX) != 0 {
                    return Err(format!("{}: {}", path.display(), std::io::Error::last_os_error()));
                }
            }
        }
        Ok(Lock(file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_told_from_paths_as_basert_tells_them() {
        for id in ["Qwen/Qwen3-0.6B", "unsloth/Qwen3-0.6B-GGUF:Q4_K_M", "mlx-community/Qwen3-4B-4bit:main"] {
            assert!(looks_like_id(id), "{id}");
        }
        for path in ["", "./m.gguf", "/models/m.gguf", "~/m", "org/m.base", "model.gguf", "Qwen3:Q4", "models/qwen3.gguf", "models/Qwen3.GGUF"] {
            assert!(!looks_like_id(path), "{path}");
        }
        let dir = std::env::temp_dir().join(format!("superfluid-pulls-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("org")).unwrap();
        std::fs::write(dir.join("org/m"), b"").unwrap();
        let on_disk = dir.join("org/m");
        assert!(!looks_like_id(&on_disk.to_string_lossy()), "a path on disk is a path");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(served_name("unsloth/Qwen3-0.6B-GGUF:Q4_K_M"), "unsloth/Qwen3-0.6B-GGUF");
    }

    #[test]
    fn pull_options_are_checked_against_what_the_runtime_declares() {
        let declared = vec![("target".to_string(), true), ("force".to_string(), false)];
        let rt = RuntimeId::new("basert");
        let flags = |options: Vec<(&str, Option<&str>)>| PullFlags {
            offline: false,
            options: options.into_iter().map(|(n, v)| (n.to_string(), v.map(str::to_string))).collect(),
        };
        assert_eq!(flags(vec![("target", Some("base-q8")), ("force", None)]).check(rt, &declared), Ok(()));
        assert!(flags(vec![("quant", Some("q8"))]).check(rt, &declared).unwrap_err().contains("takes no such option (it takes: --pull-target, --pull-force)"));
        assert!(flags(vec![("target", None)]).check(rt, &declared).unwrap_err().contains("needs a value"));
        assert!(flags(vec![("force", Some("yes"))]).check(rt, &declared).unwrap_err().contains("takes no value"));
    }

    fn toy_worker(dir: &Path, model: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let v = superfluid_agent::client::PROTO_VERSION;
        let bin = dir.join("superfluid-worker-toyrt");
        let script = format!(
            r#"#!/bin/sh
case "$1" in
  check) echo '{{"runtime":{{"id":"toyrt","version":"1"}},"link_w":[{v}],"available":true,"formats":[{{"id":"toy","describe":"a toy file","file":{{"extensions":["toy"]}}}}],"tokenizer":{{"from":"huggingface"}},"pull":{{"options":[{{"name":"target","value":true,"about":"x"}}]}}}}' ;;
  pull) shift; echo "$@" > "{args}"; echo '{{"path":"{model}"}}' ;;
esac
"#,
            args = dir.join("pull-args").display(),
            model = model.display()
        );
        std::fs::write(&bin, script).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        bin
    }

    #[test]
    fn a_model_by_id_is_pulled_by_its_runtimes_own_worker() {
        let dir = std::env::temp_dir().join(format!("superfluid-pulls-route-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.toy");
        std::fs::write(&model, b"").unwrap();
        toy_worker(&dir, &model);
        let cat = Catalog::new(dir.join("home"), Some(dir.clone()), dir.join("superfluid-workerd"));
        let toy = RuntimeId::new("toyrt");
        let flags = PullFlags { offline: true, options: vec![("target".into(), Some("q8".into()))] };
        assert_eq!(pull(&cat, toy, "org/m:q4", &flags).unwrap(), model);
        assert_eq!(std::fs::read_to_string(dir.join("pull-args")).unwrap().trim(), "org/m:q4 --offline --target q8");
        let bad = PullFlags { offline: false, options: vec![("force".into(), None)] };
        assert!(pull(&cat, toy, "org/m", &bad).unwrap_err().contains("--pull-force: the toyrt runtime's pull takes no such option"));
        let e = pull(&cat, RuntimeId::new("nowhere"), "org/m", &PullFlags { offline: true, options: Vec::new() }).unwrap_err();
        assert!(e.contains("offline: nothing was installed"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn adapter_without_its_runtime(dir: &Path, home: &Path, model: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let v = superfluid_agent::client::PROTO_VERSION;
        let bin = dir.join("superfluid-worker-toyrt");
        let script = format!(
            r#"#!/bin/sh
self=$(cd "$(dirname "$0")" && pwd)
formats='"formats":[{{"id":"toy","describe":"a toy file","file":{{"extensions":["toy"]}}}}],"tokenizer":{{"from":"huggingface"}}'
case "$1" in
  check)
    if [ -e "$self/named-build" ]; then
      echo '{{"runtime":{{"id":"toyrt","version":null}},"link_w":[{v}],"available":"/named/build holds no toyrt","named":"TOYRT_LIB",'"$formats"'}}'
      exit 1
    fi
    if [ -e "$self/../lib/payload" ] || [ -e "{home}/runtimes/toyrt/current/lib/payload" ]; then
      echo '{{"runtime":{{"id":"toyrt","version":"b1"}},"link_w":[{v}],"available":true,'"$formats"',"devices":[{{"backend":"CPU","name":"x","memory":1}}],"pull":{{"options":[]}}}}'
    else
      echo '{{"runtime":{{"id":"toyrt","version":null}},"link_w":[{v}],"available":"toyrt not found: install it with `superfluid runtime install toyrt`",'"$formats"'}}'
      exit 1
    fi ;;
  plan) echo '{{"describe":"a test machine","plans":[{{"id":"toyrt","version":"b1","backend":"cpu","install":"b1-cpu","assets":[],"tested":true,"why":"a test","extra":null}}]}}' ;;
  install)
    dir="$3"
    mkdir -p "$dir/bin" "$dir/lib"
    echo payload > "$dir/lib/payload"
    cp "$0" "$dir/bin/superfluid-worker-toyrt"
    echo '{{"id":"toyrt","install":"b1-cpu","version":"b1","backend":"cpu","link_w":[{v}],"platforms":[],"worker":"bin/superfluid-worker-toyrt","sources":[],"tested":true}}' > "$dir/runtime.json"
    cat "$dir/runtime.json" ;;
  pull) echo '{{"path":"{model}"}}' ;;
esac
"#,
            home = home.display(),
            model = model.display()
        );
        std::fs::write(&bin, script).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        bin
    }

    #[test]
    fn a_runtime_whose_adapter_is_here_is_installed_on_first_use() {
        let dir = std::env::temp_dir().join(format!("superfluid-pulls-first-use-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let beside = dir.join("bin");
        let home = dir.join("home");
        std::fs::create_dir_all(&beside).unwrap();
        let model = dir.join("m.toy");
        std::fs::write(&model, b"").unwrap();
        adapter_without_its_runtime(&beside, &home, &model);
        let catalog = || Catalog::new(home.clone(), Some(beside.clone()), beside.join("superfluid-workerd"));
        let toy = RuntimeId::new("toyrt");
        let installed = |cat: &Catalog| crate::installs::list(cat, toy).into_iter().map(|i| i.name).collect::<Vec<_>>();
        let online = PullFlags { offline: false, options: Vec::new() };

        let cat = catalog();
        let e = pull(&cat, toy, "org/m", &PullFlags { offline: true, options: Vec::new() }).unwrap_err();
        assert!(e.contains("toyrt not found") && e.ends_with("(offline: nothing was installed)"), "{e}");
        assert!(installed(&cat).is_empty());
        let e = pull(&cat, RuntimeId::new("toyrt@b9-cpu"), "org/m", &online).unwrap_err();
        assert!(e.contains("b9-cpu"), "{e}");
        assert!(installed(&cat).is_empty());

        assert_eq!(pull(&cat, toy, "org/m", &online).unwrap(), model);
        assert_eq!(installed(&cat), ["b1-cpu"]);
        assert!(cat.require(toy).is_ok());

        std::fs::remove_file(home.join("runtimes/toyrt/b1-cpu/lib/payload")).unwrap();
        let cat = catalog();
        let e = pull(&cat, toy, "org/m", &online).unwrap_err();
        assert!(e.contains("toyrt not found") && !e.contains("offline"), "{e}");
        assert_eq!(installed(&cat), ["b1-cpu"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_runtime_the_operator_named_a_build_of_is_not_installed_on_first_use() {
        let dir = std::env::temp_dir().join(format!("superfluid-pulls-named-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (beside, home) = (dir.join("bin"), dir.join("home"));
        std::fs::create_dir_all(&beside).unwrap();
        let model = dir.join("m.toy");
        std::fs::write(&model, b"").unwrap();
        adapter_without_its_runtime(&beside, &home, &model);
        std::fs::write(beside.join("named-build"), b"").unwrap();
        let catalog = || Catalog::new(home.clone(), Some(beside.clone()), beside.join("superfluid-workerd"));
        let toy = RuntimeId::new("toyrt");
        let online = PullFlags { offline: false, options: Vec::new() };

        let cat = catalog();
        let e = pull(&cat, toy, "org/m", &online).unwrap_err();
        assert!(e.contains("cannot run here: /named/build holds no toyrt"), "{e}");
        assert!(e.ends_with("(TOYRT_LIB names the runtime to load, so none was installed)"), "{e}");
        assert!(crate::installs::list(&cat, toy).is_empty(), "nothing was installed");

        std::fs::remove_file(beside.join("named-build")).unwrap();
        let cat = catalog();
        assert_eq!(pull(&cat, toy, "org/m", &online).unwrap(), model);
        assert_eq!(crate::installs::list(&cat, toy).len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_release_installs_a_runtime_it_ships_the_adapter_for_on_first_use() {
        let dir = std::env::temp_dir().join(format!("superfluid-pulls-shipped-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (beside, libexec, home) = (dir.join("bin"), dir.join("libexec").join("superfluid"), dir.join("home"));
        std::fs::create_dir_all(&beside).unwrap();
        std::fs::create_dir_all(&libexec).unwrap();
        let model = dir.join("m.toy");
        std::fs::write(&model, b"").unwrap();
        adapter_without_its_runtime(&libexec, &home, &model);
        let cat = Catalog::new(home.clone(), Some(beside.clone()), beside.join("superfluid-workerd"));
        let toy = RuntimeId::new("toyrt");
        assert_eq!(cat.locate(toy).unwrap().source, Source::Shipped);

        let e = pull(&cat, toy, "org/m", &PullFlags { offline: true, options: Vec::new() }).unwrap_err();
        assert!(e.starts_with("the toyrt runtime is not installed: toyrt not found"), "{e}");
        assert!(e.ends_with("(offline: nothing was installed)"), "{e}");
        assert!(crate::installs::list(&cat, toy).is_empty());

        assert_eq!(pull(&cat, toy, "org/m", &PullFlags { offline: false, options: Vec::new() }).unwrap(), model);
        assert_eq!(crate::installs::list(&cat, toy).into_iter().map(|i| i.name).collect::<Vec<_>>(), ["b1-cpu"]);
        assert_eq!(cat.require(toy).unwrap().worker.source, Source::Installed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_second_pull_of_one_model_waits_for_the_first() {
        let dir = std::env::temp_dir().join(format!("superfluid-pull-lock-{}", std::process::id()));
        let first = Lock::take(&dir, "org/m:q4", "pull of org/m:q4").unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let d = dir.clone();
        let waiter = std::thread::spawn(move || {
            let _second = Lock::take(&d, "org/m:q4", "pull of org/m:q4").unwrap();
            tx.send(()).unwrap();
        });
        assert!(rx.recv_timeout(std::time::Duration::from_millis(200)).is_err(), "the second waits");
        drop(first);
        assert!(rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok(), "and goes once the first is done");
        waiter.join().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
