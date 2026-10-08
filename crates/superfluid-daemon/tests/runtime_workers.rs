//! Runtimes as their own workers, end to end.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn built(name: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_BIN_EXE_superfluid")).with_file_name(name);
    p.is_file().then_some(p)
}

fn llama_worker() -> Option<PathBuf> {
    let worker = built("superfluid-worker-llamacpp")?;
    let out = Command::new(&worker).arg("version").output().ok()?;
    if !out.status.success() {
        eprintln!("SKIP: {}", String::from_utf8_lossy(&out.stderr).trim());
        return None;
    }
    Some(worker)
}

fn tokenizer_library() -> Option<PathBuf> {
    let file = if cfg!(target_os = "macos") {
        "libsuperfluid_tokenizer_llamacpp.dylib"
    } else {
        "libsuperfluid_tokenizer_llamacpp.so"
    };
    built(file)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brt-rw-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn http(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let raw = String::from_utf8_lossy(&raw).into_owned();
    let (head, payload) = raw.split_once("\r\n\r\n").expect("split");
    let status: u16 = head.lines().next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    assert!(!head.to_ascii_lowercase().contains("transfer-encoding: chunked"), "a plain JSON reply");
    (status, payload.to_string())
}

fn superfluid(home: &Path, env: &[(&str, &Path)]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_superfluid"));
    c.env("SUPERFLUID_HOME", home).env_remove("SUPERFLUID_WORKER_LLAMACPP").env_remove("SUPERFLUID_WORKER_MLX");
    for (k, v) in env {
        c.env(k, v);
    }
    c
}

fn serve(dir: &Path, model: &Path, env: &[(&str, &Path)]) -> (Child, SocketAddr, PathBuf) {
    serve_with(dir, model, env, &[])
}

fn serve_with(dir: &Path, model: &Path, env: &[(&str, &Path)], more: &[&str]) -> (Child, SocketAddr, PathBuf) {
    let addr: SocketAddr = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap()
    };
    let log = dir.join("stderr.log");
    let sock = std::env::temp_dir().join(format!("brt-rw-{}-{}.sock", addr.port(), std::process::id()));
    let mut child = Child(
        superfluid(&dir.join("home"), env)
            .args(["serve", "--model", &model.to_string_lossy(), "--sessions"])
            .arg(dir.join("sessions"))
            .arg("--socket")
            .arg(&sock)
            .args(["--http", &addr.to_string(), "--max-context", "2048", "--max-batch", "2", "--no-tui"])
            .args(more)
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(&log).unwrap())
            .spawn()
            .expect("spawn superfluid serve"),
    );
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if TcpStream::connect(addr).is_ok() && http(addr, "GET", "/v1/models", "").0 == 200 {
            break;
        }
        let stderr = std::fs::read_to_string(&log).unwrap_or_default();
        if let Ok(Some(status)) = child.0.try_wait() {
            panic!("superfluid serve exited ({status}) before it came up:\n{stderr}");
        }
        assert!(Instant::now() < deadline, "superfluid serve did not come up:\n{stderr}");
        std::thread::sleep(Duration::from_millis(200));
    }
    (child, addr, log)
}

fn greedy_paris(addr: SocketAddr, stem: &str) {
    let (status, body) = http(
        addr,
        "POST",
        "/v1/completions",
        &format!(r#"{{"model":"{stem}","prompt":"The capital of France is","max_tokens":8,"temperature":0}}"#),
    );
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("Paris"), "greedy completion names Paris: {body}");
    let (status, body) = http(
        addr,
        "POST",
        "/v1/chat/completions",
        &format!(r#"{{"model":"{stem}","messages":[{{"role":"user","content":"Say hello."}}],"max_tokens":12,"temperature":0}}"#),
    );
    assert_eq!(status, 200, "{body}");
    assert!(body.contains(r#""finish_reason""#), "{body}");
}

#[test]
fn a_model_path_that_is_not_there_is_told_as_that() {
    let dir = scratch("nomodel");
    for name in ["missing.gguf", "missing-mlx", "missing.bin"] {
        let model = dir.join(name);
        let sock = std::env::temp_dir().join(format!("brt-rw-nm-{}.sock", std::process::id()));
        let out = superfluid(&dir.join("home"), &[])
            .args(["serve", "--model", &model.to_string_lossy(), "--sessions"])
            .arg(dir.join("sessions"))
            .arg("--socket")
            .arg(&sock)
            .args(["--no-tui", "--no-http"])
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{name}: {err}");
        assert!(err.contains(&format!("superfluid: {}: no such file or directory", model.display())), "{name}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_serve_line_is_checked_before_a_model_is_pulled() {
    let dir = scratch("usage-first");
    let out = superfluid(&dir.join("home"), &[])
        .args(["serve", "--runtime", "llamacpp", "--model", "unsloth/Qwen3-0.6B-GGUF:Q4_K_M", "--offline", "--no-tui", "--max-ctx", "4096"])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.starts_with("superfluid: serve has no option --max-ctx (did you mean --max-context?)"), "{err}");
    assert!(!err.contains("llamacpp runtime") && !err.contains("unsloth/"), "nothing was looked for: {err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_model_id_names_its_runtime_and_nothing_else_is_needed() {
    let dir = scratch("zero-flags");
    let out = superfluid(&dir.join("home"), &[])
        .env_remove("SUPERFLUID_LLAMA_LIB")
        .env("HF_HOME", dir.join("hf"))
        .args(["serve", "unsloth/Qwen3-0.6B-GGUF:Q4_K_M", "--offline", "--no-tui", "--no-http"])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!err.contains("--sessions") && !err.contains("--socket") && !err.contains("usage"), "{err}");
    assert!(err.contains("llamacpp") && !err.contains("basert"), "a GGUF repository goes to llama.cpp: {err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn runtimes_names_each_worker_its_source_and_version() {
    let (Some(worker), Some(_lib)) = (llama_worker(), tokenizer_library()) else {
        eprintln!("SKIP: build superfluid-adapter-llamacpp and superfluid-tokenizer-llamacpp first");
        return;
    };
    let dir = scratch("list");
    let out = superfluid(&dir, &[("SUPERFLUID_WORKER_LLAMACPP", &worker)]).args(["runtimes", "--json"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let rows: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let llama = rows.as_array().unwrap().iter().find(|r| r["id"] == "llamacpp").expect("a llamacpp row");
    assert_eq!(llama["ready"], true, "{llama}");
    assert_eq!(llama["worker"]["source"], "environment");
    assert_eq!(llama["worker"]["path"], worker.to_string_lossy().as_ref());
    assert!(llama["version"].as_str().unwrap().starts_with("llama.cpp "), "{llama}");
    assert!(llama["capabilities"]["serving"]["park_lossy"].is_string(), "the static record rides along: {llama}");
    let out = superfluid(&dir, &[("SUPERFLUID_WORKER_MLX", &worker)]).args(["runtimes", "--json"]).output().unwrap();
    let rows: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let mlx = rows.as_array().unwrap().iter().find(|r| r["id"] == "mlx").unwrap();
    assert!(mlx["ready"].as_str().unwrap().contains("is not the mlx runtime's worker"), "{mlx}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_gguf_is_served_through_the_llamacpp_worker_and_its_tokenizer_library() {
    let (Some(worker), Some(_lib)) = (llama_worker(), tokenizer_library()) else {
        eprintln!("SKIP: build superfluid-adapter-llamacpp and superfluid-tokenizer-llamacpp first");
        return;
    };
    let Some(model) = std::env::var_os("SUPERFLUID_TEST_GGUF").map(PathBuf::from).filter(|p| p.is_file()) else {
        eprintln!("SKIP: set SUPERFLUID_TEST_GGUF");
        return;
    };
    let dir = scratch("gguf");
    let (child, addr, log) = serve(&dir, &model, &[("SUPERFLUID_WORKER_LLAMACPP", &worker)]);
    let stderr = std::fs::read_to_string(&log).unwrap();
    assert!(
        stderr.contains(&format!("runtime llamacpp serves {}", model.display()))
            && stderr.contains(&format!("environment: {}", worker.display())),
        "the startup line names the worker and where it came from:\n{stderr}"
    );
    greedy_paris(addr, &model.file_stem().unwrap().to_string_lossy());
    drop(child);
    let out = superfluid(&dir.join("home"), &[("SUPERFLUID_WORKER_LLAMACPP", &worker)])
        .args(["serve", "--model", &model.to_string_lossy(), "--sessions"])
        .arg(dir.join("s2"))
        .args(["--socket", "/nonexistent/brt.sock", "--no-http", "--park-lossy", "--no-tui"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--park-lossy: the llamacpp runtime exports no lossy encoding"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let declined = |stderr: &str| stderr.contains("the llamacpp runtime sizes no window on") || stderr.contains("registers no GPU");
    let sized_line = |stderr: &str| -> Option<u64> {
        let line = stderr.lines().find(|l| l.contains("tokens (auto: sized for this device, 2 lanes"))?;
        line.strip_prefix("superfluid: context window ")?.split(' ').next()?.parse().ok()
    };
    let out = superfluid(&dir.join("home"), &[("SUPERFLUID_WORKER_LLAMACPP", &worker)])
        .args(["serve", "--model", &model.to_string_lossy(), "--sessions"])
        .arg(dir.join("s3"))
        .args(["--socket", "/nonexistent/brt.sock", "--no-http", "--max-context", "auto", "--max-batch", "2", "--no-tui"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    if declined(&stderr) {
        assert_eq!(out.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains("--max-context auto: ") && stderr.contains("pass --max-context <tokens>"), "{stderr}");
    } else {
        let n = sized_line(&stderr).unwrap_or_else(|| panic!("a sized window:\n{stderr}"));
        assert!(n >= 2048 && (n.is_multiple_of(1024) || n < 4096), "{n}\n{stderr}");
    }
    let out = superfluid(&dir.join("home"), &[("SUPERFLUID_WORKER_LLAMACPP", &worker)])
        .args(["serve", "--model", &model.to_string_lossy(), "--sessions"])
        .arg(dir.join("s5"))
        .args(["--socket", "/nonexistent/brt.sock", "--no-http", "--max-batch", "2", "--no-tui"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("--max-context auto"), "{stderr}");
    if declined(&stderr) {
        assert!(stderr.contains("superfluid: context window 8192 tokens (the llamacpp runtime ") && stderr.contains("; --max-context N pins it)"), "{stderr}");
    } else {
        assert!(sized_line(&stderr).is_some(), "{stderr}");
    }
    let out = superfluid(&dir.join("home"), &[("SUPERFLUID_WORKER_LLAMACPP", &worker)])
        .args(["serve", "--model", &model.to_string_lossy(), "--sessions"])
        .arg(dir.join("s4"))
        .args(["--socket", "/nonexistent/brt.sock", "--no-http", "--speculate", "auto", "--no-tui"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("--speculate auto:") && !stderr.contains("cannot read the bundle header"), "{stderr}");
    assert!(stderr.contains("speculation: auto: prompt-lookup on the llamacpp runtime"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn two_artifacts_sharing_a_stem_keep_their_state_apart() {
    let (Some(worker), Some(_lib)) = (llama_worker(), tokenizer_library()) else {
        eprintln!("SKIP: build superfluid-adapter-llamacpp and superfluid-tokenizer-llamacpp first");
        return;
    };
    let Some(model) = std::env::var_os("SUPERFLUID_TEST_GGUF").map(PathBuf::from).filter(|p| p.is_file()) else {
        eprintln!("SKIP: set SUPERFLUID_TEST_GGUF");
        return;
    };
    let dir = scratch("stems");
    let models = dir.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let model = model.canonicalize().unwrap();
    for name in ["m.bin", "m.gguf", "m_gguf.gguf"] {
        std::fs::hard_link(&model, models.join(name)).or_else(|_| std::fs::copy(&model, models.join(name)).map(|_| ())).unwrap();
    }
    let (_child, addr, log) =
        serve_with(&dir, &model, &[("SUPERFLUID_WORKER_LLAMACPP", &worker)], &["--model-dir", &models.to_string_lossy()]);
    let stderr = std::fs::read_to_string(&log).unwrap();
    assert!(stderr.contains("is registered as 'm.gguf' (another artifact took its stem)"), "{stderr}");
    for name in ["m", "m.gguf", "m_gguf"] {
        let (status, body) = http(
            addr,
            "POST",
            "/v1/chat/completions",
            &format!(r#"{{"model":"{name}","messages":[{{"role":"user","content":"Say hello."}}],"max_tokens":8,"temperature":0}}"#),
        );
        assert_eq!(status, 200, "{name}: {body}");
    }
    let mut kept: Vec<(String, String)> = std::fs::read_dir(dir.join("sessions").join("models"))
        .unwrap()
        .flatten()
        .filter(|e| e.path().join("wal.log").is_file())
        .map(|e| {
            let owner = std::fs::read_to_string(e.path().join("model-id")).unwrap_or_default();
            (owner, e.file_name().to_string_lossy().into_owned())
        })
        .collect();
    kept.sort();
    let owners: Vec<&str> = kept.iter().map(|(owner, _)| owner.as_str()).collect();
    assert_eq!(owners, ["m", "m.gguf", "m_gguf"], "{kept:?}");
    assert_eq!(kept[0].1, "m");
    assert_eq!(kept[1].1, "m_gguf", "the first of the two to load has the plain name");
    assert!(kept[2].1.starts_with("m_gguf-"), "{kept:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_mlx_directory_is_served_through_the_mlx_worker() {
    let Some(worker) = built("superfluid-worker-mlx") else {
        eprintln!("SKIP: build superfluid-adapter-mlx first");
        return;
    };
    let Some(model) = std::env::var_os("SUPERFLUID_TEST_MLX").map(PathBuf::from).filter(|p| p.is_dir()) else {
        eprintln!("SKIP: set SUPERFLUID_TEST_MLX (and SUPERFLUID_MLX_VENV)");
        return;
    };
    let dir = scratch("mlx");
    let (_child, addr, log) = serve(&dir, &model, &[("SUPERFLUID_WORKER_MLX", &worker)]);
    let stderr = std::fs::read_to_string(&log).unwrap();
    assert!(stderr.contains(&format!("environment: {}", worker.display())), "{stderr}");
    greedy_paris(addr, &model.file_name().unwrap().to_string_lossy());
    let _ = std::fs::remove_dir_all(&dir);
}
