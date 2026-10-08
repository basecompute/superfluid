//! What installing MLX fetches.

use std::path::Path;
use std::process::{Command, Stdio};

use superfluid_adapter_kit::install::{fetch, unpack, version_key, Asset, Host, Plan, Request};
use serde_json::{json, Value};

fn recipe() -> Value {
    serde_json::from_str(include_str!("../recipe.json")).expect("recipe.json is JSON")
}

pub fn tested() -> Vec<String> {
    let mut v: Vec<String> =
        recipe()["tested"].as_array().map(|a| a.iter().filter_map(|t| t["version"].as_str().map(str::to_string)).collect()).unwrap_or_default();
    v.sort_by_key(|x| std::cmp::Reverse(version_key(x)));
    v
}

fn minor(v: &str) -> String {
    v.split('.').take(2).collect::<Vec<_>>().join(".")
}

pub fn python_version() -> String {
    minor(recipe()["python"]["version"].as_str().unwrap_or_default())
}

pub fn linked_python() -> &'static str {
    env!("SUPERFLUID_MLX_PYTHON")
}

pub fn plan(req: &Request, host: &Host, linked: &str) -> Result<Vec<Plan>, String> {
    if !(host.os == "macos" && host.arch == "arm64") {
        return Err(format!("MLX runs on Apple silicon only; this is {}", host.describe()));
    }
    if let Some(b) = &req.backend {
        if b != "metal" {
            return Err(format!("MLX runs on Metal, not {b}"));
        }
    }
    let r = recipe();
    let python = &r["python"];
    let wanted = python_version();
    if linked != wanted {
        return Err(format!(
            "this MLX worker links Python {linked}, and the MLX runtime installs Python {wanted}: its worker must be built \
             against a Python {wanted} (a source build sets PYO3_PYTHON to one)"
        ));
    }
    let tested_versions = tested();
    let version = req.version.clone().or_else(|| tested_versions.first().cloned()).ok_or("the recipe lists no MLX version")?;
    let entry = r["tested"].as_array().and_then(|a| a.iter().find(|t| t["version"] == version.as_str())).cloned();
    let requirements: Vec<String> = match &entry {
        Some(t) => t["wheels"]
            .as_array()
            .map(|w| {
                w.iter()
                    .filter_map(|x| Some(format!("{}=={} --hash=sha256:{}", x["name"].as_str()?, x["version"].as_str()?, x["sha256"].as_str()?)))
                    .collect()
            })
            .unwrap_or_default(),
        None if req.untested => vec![format!("mlx=={version}"), "mlx-lm".into()],
        None => {
            return Err(format!(
                "mlx {version} is not a version this adapter was tested with (tested: {}); pass --untested to install it, \
                 unpinned, anyway",
                tested_versions.join(", ")
            ))
        }
    };
    let platform = host.platform();
    let py = &python["assets"][platform.as_str()];
    let asset = Asset {
        name: py["name"].as_str().ok_or_else(|| format!("no Python for {platform} in the recipe"))?.to_string(),
        url: py["url"].as_str().unwrap_or_default().to_string(),
        sha256: py["sha256"].as_str().map(str::to_string),
        size: py["size"].as_u64(),
    };
    let wheels_size: u64 =
        entry.as_ref().and_then(|t| t["wheels"].as_array().map(|w| w.iter().filter_map(|x| x["size"].as_u64()).sum())).unwrap_or(0);
    let index = req.from.clone().unwrap_or_else(|| r["index"].as_str().unwrap_or("https://pypi.org/simple").to_string());
    Ok(vec![Plan {
        id: "mlx".into(),
        install: format!("{version}-metal"),
        backend: "metal".into(),
        assets: vec![asset],
        tested: entry.is_some(),
        why: format!(
            "{}: Python {} and mlx {version}{} from {index}",
            host.describe(),
            python["version"].as_str().unwrap_or_default(),
            entry.as_ref().and_then(|t| t["mlx_lm"].as_str()).map(|v| format!(" with mlx-lm {v}")).unwrap_or_default()
        ),
        version,
        extra: json!({"index": index, "requirements": requirements, "wheels_size": wheels_size}),
    }])
}

pub fn install(plan: &Plan, dir: &Path) -> Result<Vec<(String, String)>, String> {
    let fetched = dir.join(".fetch");
    std::fs::create_dir_all(&fetched).map_err(|e| format!("{}: {e}", fetched.display()))?;
    let asset = plan.assets.first().ok_or("a plan without its Python")?;
    let (archive, sha) = fetch(asset, &fetched)?;
    unpack(&archive, dir)?;
    let _ = std::fs::remove_dir_all(&fetched);
    let mut sources = vec![(asset.url.clone(), sha)];
    let python = dir.join("python").join("bin").join("python3");
    if !python.is_file() {
        return Err(format!("{} holds no python/bin/python3", plan.assets[0].url));
    }
    let env = dir.join("env");
    run(own_python(&python).args(["-m", "venv"]).arg(&env), "making the environment")?;
    let requirements: Vec<String> =
        plan.extra["requirements"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let pinned = requirements.iter().all(|r| r.contains("--hash="));
    let file = dir.join(".requirements.txt");
    std::fs::write(&file, requirements.join("\n") + "\n").map_err(|e| format!("{}: {e}", file.display()))?;
    let index = plan.extra["index"].as_str().unwrap_or("https://pypi.org/simple");
    let mut pip = own_python(&env.join("bin").join("python"));
    pip.args(["-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-cache-dir", "--only-binary=:all:"])
        .args(["--index-url", index])
        .arg("-r")
        .arg(&file);
    if pinned {
        pip.args(["--require-hashes", "--no-deps"]);
    } else {
        eprintln!("warning: mlx {} is untested: pip resolves its packages from {index}, unpinned", plan.version);
    }
    eprintln!("installing {} packages from {index}", requirements.len());
    run(&mut pip, "installing mlx and mlx-lm")?;
    let _ = std::fs::remove_file(&file);
    relocatable(&env)?;
    for r in &requirements {
        let (pkg, hash) = r.split_once(" --hash=sha256:").unwrap_or((r.as_str(), ""));
        sources.push((format!("{index} {pkg}"), hash.to_string()));
    }
    place_worker(dir)?;
    Ok(sources)
}

const SCRIPT_HEAD: &[u8] = b"#!/bin/sh\n'''exec' \"$(dirname -- \"$(realpath -- \"$0\")\")/python\" \"$0\" \"$@\"\n' '''\n";

fn relocatable(env: &Path) -> Result<(), String> {
    let at = |p: &Path, e: std::io::Error| format!("{}: {e}", p.display());
    let bin = env.join("bin");
    let link = bin.join("python3");
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink("../../python/bin/python3", &link).map_err(|e| at(&link, e))?;

    let cfg = env.join("pyvenv.cfg");
    let text = std::fs::read_to_string(&cfg).map_err(|e| at(&cfg, e))?;
    let kept: Vec<&str> = text
        .lines()
        .filter(|l| !matches!(l.split('=').next().map(str::trim), Some("home" | "executable" | "command")))
        .collect();
    std::fs::write(&cfg, kept.join("\n") + "\n").map_err(|e| at(&cfg, e))?;

    for entry in std::fs::read_dir(&bin).map_err(|e| at(&bin, e))?.flatten() {
        let path = entry.path();
        if !entry.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        if entry.file_name().to_string_lossy().to_ascii_lowercase().starts_with("activate") {
            std::fs::remove_file(&path).map_err(|e| at(&path, e))?;
            continue;
        }
        let body = std::fs::read(&path).map_err(|e| at(&path, e))?;
        let line = body.iter().position(|&b| b == b'\n').unwrap_or(body.len());
        let first = &body[..line];
        let interpreter = first.rsplit(|&b| b == b'/').next().unwrap_or_default();
        if !first.starts_with(b"#!/") || !interpreter.starts_with(b"python") {
            continue;
        }
        let mut script = SCRIPT_HEAD.to_vec();
        script.extend_from_slice(body.get(line + 1..).unwrap_or_default());
        std::fs::write(&path, script).map_err(|e| at(&path, e))?;
    }
    Ok(())
}

const INSTALL_PYTHON: &str = "@executable_path/../python/lib";

fn place_worker(dir: &Path) -> Result<(), String> {
    let worker = superfluid_adapter_kit::install::copy_worker(dir, "mlx")?;
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    let lib = std::fs::read_dir(dir.join("python").join("lib"))
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .find(|n| n.starts_with("libpython3") && n.ends_with(".dylib"))
        .ok_or("the runtime's Python has no libpython")?;
    let links = crate::macho::Links::read(&worker)?;
    let python = links.python().ok_or("the MLX worker links no libpython")?;
    if links.loads_from(python, INSTALL_PYTHON) && python.name.ends_with(&format!("/{lib}")) {
        return Ok(());
    }
    let new = format!("{INSTALL_PYTHON}/{lib}");
    let tools = "the Xcode command line tools point a worker built against another Python at the runtime's (`xcode-select --install`)";
    run(Command::new("install_name_tool").args(["-change", &python.name, &new]).arg(&worker), "pointing the worker at its Python")
        .map_err(|e| format!("{e}; {tools}"))?;
    run(Command::new("codesign").args(["--force", "--sign", "-"]).arg(&worker), "signing the worker")?;
    Ok(())
}

fn own_python(python: &Path) -> Command {
    let mut cmd = Command::new(python);
    cmd.arg("-I");
    for (name, _) in std::env::vars_os() {
        if callers_pip_setting(&name.to_string_lossy()) {
            cmd.env_remove(&name);
        }
    }
    cmd
}

fn callers_pip_setting(name: &str) -> bool {
    const NETWORK: [&str; 6] = ["PIP_CERT", "PIP_CLIENT_CERT", "PIP_PROXY", "PIP_TRUSTED_HOST", "PIP_TIMEOUT", "PIP_RETRIES"];
    name.starts_with("PIP_") && !NETWORK.contains(&name)
}

fn run(cmd: &mut Command, what: &str) -> Result<(), String> {
    let status = cmd.stdout(Stdio::from(std::io::stderr())).stderr(Stdio::inherit()).status().map_err(|e| format!("{what}: {e}"))?;
    if !status.success() {
        return Err(format!("{what} failed ({status})"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_environment_names_nothing_of_where_it_was_made() {
        let dir = std::env::temp_dir().join(format!("superfluid-mlx-relocatable-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let staging = dir.join(".staging-1-0");
        let bin = staging.join("env").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(staging.join("python").join("bin")).unwrap();
        std::fs::write(staging.join("python").join("bin").join("python3"), b"").unwrap();
        let made = staging.display().to_string();
        std::os::unix::fs::symlink(format!("{made}/python/bin/python3"), bin.join("python3")).unwrap();
        std::os::unix::fs::symlink("python3", bin.join("python")).unwrap();
        std::fs::write(
            staging.join("env").join("pyvenv.cfg"),
            format!("home = {made}/python/bin\ninclude-system-site-packages = false\nversion = 3.12.14\nexecutable = {made}/python/bin/python3.12\ncommand = {made}/python/bin/python3 -m venv {made}/env\n"),
        )
        .unwrap();
        std::fs::write(bin.join("hf"), format!("#!{made}/env/bin/python\nimport sys\n")).unwrap();
        std::fs::write(bin.join("pip3.12"), format!("#!{made}/env/bin/python3.12\nimport pip\n")).unwrap();
        std::fs::write(bin.join("activate"), format!("VIRTUAL_ENV={made}/env\n")).unwrap();
        std::fs::write(bin.join("tool"), b"#!/bin/sh\necho\n").unwrap();

        relocatable(&staging.join("env")).unwrap();
        let moved = dir.join("0.32.2-metal");
        std::fs::rename(&staging, &moved).unwrap();

        let bin = moved.join("env").join("bin");
        assert!(bin.join("python").canonicalize().unwrap().ends_with("0.32.2-metal/python/bin/python3"));
        let cfg = std::fs::read_to_string(moved.join("env").join("pyvenv.cfg")).unwrap();
        assert_eq!(cfg, "include-system-site-packages = false\nversion = 3.12.14\n");
        let hf = std::fs::read_to_string(bin.join("hf")).unwrap();
        assert!(hf.starts_with("#!/bin/sh\n") && hf.ends_with("\nimport sys\n") && !hf.contains(".staging"), "{hf}");
        let pip = std::fs::read_to_string(bin.join("pip3.12")).unwrap();
        assert!(pip.starts_with("#!/bin/sh\n") && pip.ends_with("\nimport pip\n"), "{pip}");
        assert!(!bin.join("activate").exists());
        assert_eq!(std::fs::read(bin.join("tool")).unwrap(), b"#!/bin/sh\necho\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn mac() -> Host {
        Host { os: "macos".into(), arch: "arm64".into(), ..Host::default() }
    }

    #[test]
    fn the_callers_pip_settings_stay_the_callers() {
        for theirs in ["PIP_USER", "PIP_TARGET", "PIP_REQUIRE_VIRTUALENV", "PIP_CONSTRAINT", "PIP_INDEX_URL", "PIP_NO_INDEX", "PIP_CONFIG_FILE"] {
            assert!(callers_pip_setting(theirs), "{theirs}");
        }
        for kept in ["PIP_CERT", "PIP_PROXY", "PIP_TRUSTED_HOST", "HTTPS_PROXY", "SSL_CERT_FILE", "PIPX_HOME", "PATH"] {
            assert!(!callers_pip_setting(kept), "{kept}");
        }
        let cmd = own_python(Path::new("/nowhere/python3"));
        assert_eq!(cmd.get_args().next().and_then(|a| a.to_str()), Some("-I"));
    }

    #[test]
    fn the_tested_version_is_locked_to_every_wheel_and_hash() {
        let p = &plan(&Request::default(), &mac(), "3.12").unwrap()[0];
        assert_eq!((p.backend.as_str(), p.tested), ("metal", true));
        assert_eq!(p.install, format!("{}-metal", p.version));
        let reqs = p.extra["requirements"].as_array().unwrap();
        assert!(reqs.len() > 30 && reqs.iter().all(|r| r.as_str().unwrap().contains("==") && r.as_str().unwrap().contains("--hash=sha256:")));
        for pkg in ["mlx==", "mlx-metal==", "mlx-lm=="] {
            assert!(reqs.iter().any(|r| r.as_str().unwrap().starts_with(pkg)), "{pkg}");
        }
        assert!(p.assets[0].name.starts_with("cpython-3.12") && p.assets[0].sha256.as_ref().is_some_and(|s| s.len() == 64));
        assert_eq!(p.extra["index"], "https://pypi.org/simple");
    }

    #[test]
    fn the_recorded_python_is_the_one_this_build_links() {
        let running = pyo3::Python::attach(|py| {
            let v = py.version_info();
            format!("{}.{}", v.major, v.minor)
        });
        assert_eq!(linked_python(), running);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_placed_worker_runs_on_the_installs_python() {
        let dir = std::env::temp_dir().join(format!("superfluid-mlx-place-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let lib = dir.join("python").join("lib");
        std::fs::create_dir_all(&lib).unwrap();
        let name = format!("libpython{}.dylib", linked_python());
        std::fs::write(lib.join(&name), b"").unwrap();
        match place_worker(&dir) {
            Ok(()) => {}
            Err(why) if why.contains("xcode-select") => {
                eprintln!("SKIP: {why}");
                return;
            }
            Err(why) => panic!("{why}"),
        }
        let links = crate::macho::Links::read(&dir.join("bin").join("superfluid-worker-mlx")).unwrap();
        let python = links.python().expect("a libpython");
        assert!(links.loads_from(python, INSTALL_PYTHON), "{links:?}");
        assert!(python.name.ends_with(&format!("/{name}")), "{python:?}");
        place_worker(&dir).unwrap();
        assert_eq!(crate::macho::Links::read(&dir.join("bin").join("superfluid-worker-mlx")).unwrap(), links);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_worker_on_another_python_and_other_machines_are_refused() {
        assert!(plan(&Request::default(), &mac(), "3.11").unwrap_err().contains("links Python 3.11"));
        let linux = Host { os: "linux".into(), arch: "x64".into(), ..Host::default() };
        assert!(plan(&Request::default(), &linux, "3.12").unwrap_err().contains("Apple silicon only"));
        let e = plan(&Request { version: Some("9.9".into()), ..Request::default() }, &mac(), "3.12").unwrap_err();
        assert!(e.contains("--untested"), "{e}");
        let p = &plan(&Request { version: Some("9.9".into()), untested: true, ..Request::default() }, &mac(), "3.12").unwrap()[0];
        assert_eq!(p.extra["requirements"], json!(["mlx==9.9", "mlx-lm"]));
        let mirror = Request { from: Some("https://pypi.example/simple".into()), ..Request::default() };
        assert_eq!(plan(&mirror, &mac(), "3.12").unwrap()[0].extra["index"], "https://pypi.example/simple");
    }
}
