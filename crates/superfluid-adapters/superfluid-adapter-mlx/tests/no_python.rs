//! The MLX worker on a machine with no Python.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::process::{Command, Output};

use superfluid_adapter_mlx::macho::Links;

const INSTALL_PYTHON: &str = "@executable_path/../python/lib";

fn worker_without_its_python(tag: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir().join(format!("superfluid-mlx-nopython-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let bin = dir.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let worker = bin.join("superfluid-worker-mlx");
    std::fs::copy(env!("CARGO_BIN_EXE_superfluid-worker-mlx"), &worker).unwrap();
    let links = Links::read(&worker).unwrap();
    let python = links.python().expect("the worker links a libpython").clone();
    assert!(python.weak, "the worker links its Python weakly: {python:?}");
    if !links.loads_from(&python, INSTALL_PYTHON) {
        let file = python.name.rsplit('/').next().unwrap();
        let to = format!("{INSTALL_PYTHON}/{file}");
        let ran = |cmd: &mut Command| cmd.output().map(|o| o.status.success()).unwrap_or(false);
        if !ran(Command::new("install_name_tool").args(["-change", &python.name, &to]).arg(&worker))
            || !ran(Command::new("codesign").args(["--force", "--sign", "-"]).arg(&worker))
        {
            eprintln!("SKIP: no install_name_tool or codesign here to move the worker's Python");
            return None;
        }
    }
    Some(worker)
}

fn run(worker: &PathBuf, args: &[&str]) -> Output {
    Command::new(worker)
        .args(args)
        .env_remove("DYLD_LIBRARY_PATH")
        .env_remove("DYLD_FALLBACK_LIBRARY_PATH")
        .env_remove("DYLD_FRAMEWORK_PATH")
        .env_remove("SUPERFLUID_MLX_VENV")
        .env_remove("VIRTUAL_ENV")
        .env_remove("PYTHONHOME")
        .output()
        .unwrap()
}

#[test]
fn the_worker_starts_and_says_the_runtime_is_not_installed() {
    let Some(worker) = worker_without_its_python("check") else { return };
    let out = run(&worker, &["check"]);
    assert_eq!(out.status.code(), Some(1), "it exits, told; the loader does not kill it: {out:?}");
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).expect("a report");
    assert_eq!(report["runtime"]["id"], "mlx");
    let why = report["available"].as_str().expect("why it cannot run");
    assert!(why.starts_with("MLX's Python is not here"), "{why}");
    assert!(why.contains("`superfluid runtime install mlx`"), "{why}");
    assert!(why.contains(&format!("libpython{}", superfluid_adapter_mlx::recipe::linked_python())), "{why}");
    assert_eq!(report["formats"][0]["id"], "mlx");
    assert!(report["capabilities"].is_object(), "{report}");

    let out = run(&worker, &["version"]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stderr).contains("MLX's Python is not here"), "{out:?}");
    let out = run(&worker, &["pull", "mlx-community/Qwen3-0.6B-4bit", "--offline"]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stderr).contains("MLX's Python is not here"), "{out:?}");
}

#[test]
fn the_worker_plans_its_install_with_no_python() {
    let Some(worker) = worker_without_its_python("plan") else { return };
    let out = run(&worker, &["plan"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let linked = superfluid_adapter_mlx::recipe::linked_python();
    let wanted = superfluid_adapter_mlx::recipe::python_version();
    if linked == wanted {
        assert_eq!(out.status.code(), Some(0), "{stderr}");
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("a plan");
        let plan = &v["plans"][0];
        assert_eq!((plan["id"].as_str(), plan["backend"].as_str()), (Some("mlx"), Some("metal")), "{v}");
        assert!(plan["assets"][0]["name"].as_str().unwrap().starts_with(&format!("cpython-{wanted}")), "{v}");
    } else {
        assert_eq!(out.status.code(), Some(1), "{stderr}");
        assert!(stderr.contains(&format!("links Python {linked}")), "{stderr}");
    }
}
