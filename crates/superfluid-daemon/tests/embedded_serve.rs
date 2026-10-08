//! `serve` run from the library by another program, the way baseRT's
//! basert-serve runs it: the OpenAI API alone, no session socket, and the
//! program as its own worker.
//!
//! This binary plays all three parts: the test; the server, when
//! EMBEDDED_SERVE is set; and the worker, when started with a worker's
//! command line. Needs a `.base` model (BASERT_TEST_MODEL, default
//! `models/Qwen3-0.6B-Q4_K_M.base` at the repository root) and skips
//! without one.

#![cfg_attr(not(feature = "basert"), allow(dead_code, unused_imports))]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const TEST: &str = "an_embedded_server_serves_the_openai_api_alone_from_its_own_worker";

#[cfg(feature = "basert")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if superfluid_daemon::workerd::is_invocation(&args) {
        superfluid_daemon::workerd::main(args, "embedded_serve");
    }
    if std::env::var_os("EMBEDDED_SERVE").is_some() {
        let mut embedding = superfluid_daemon::serve::Embedding {
            http: superfluid_daemon::openai::HttpSurface::OPENAI_ONLY,
            model_naming: superfluid_daemon::openai::ModelNaming::Exact,
            identity: superfluid_daemon::openai::ServerIdentity::default(),
            session_socket: false,
            workerd: Some(std::env::current_exe().expect("own path")),
            ..Default::default()
        };
        if std::env::var_os("EMBEDDED_CUSTOM_POLICY").is_some() {
            embedding.model_name = |_, taken| format!("embedded-{}", taken.len());
            embedding.sampling_fallback = Some(superfluid_daemon::openai::SamplingFallback {
                temperature: 0.0, repeat_penalty: 1.07,
                top_p: 0.82, top_k: 17, min_p: 0.0,
                frequency_penalty_as_repeat: false,
            });
        }
        superfluid_daemon::serve::run(&args, &embedding);
        return;
    }
    if args.iter().any(|a| a == "--list") {
        println!("{TEST}: test");
        return;
    }
    match model_path() {
        Some(model) => {
            an_embedded_server_serves_the_openai_api_alone_from_its_own_worker(&model, false);
            an_embedded_server_serves_the_openai_api_alone_from_its_own_worker(&model, true);
            println!("test {TEST} ... ok");
        }
        None => {
            eprintln!("SKIP: no test model (set BASERT_TEST_MODEL)");
            println!("test {TEST} ... ok");
        }
    }
}

#[cfg(not(feature = "basert"))]
fn main() {}

fn model_path() -> Option<PathBuf> {
    let p = model_path_found()?;
    #[cfg(feature = "basert")]
    if p.extension().is_some_and(|e| e == "base") {
        superfluid_engine_ffi::libbasert::require()?;
    }
    Some(p)
}

fn model_path_found() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_MODEL") {
        return Some(PathBuf::from(p));
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen3-0.6B-Q4_K_M.base");
    p.exists().then_some(p)
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
    let status: u16 = head
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    (status, payload.to_string())
}

fn an_embedded_server_serves_the_openai_api_alone_from_its_own_worker(model: &Path, custom: bool) {
    let me = std::env::current_exe().unwrap();
    let dir = std::env::temp_dir().join(format!("superfluid-embedded-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let home = dir.join("home");
    let addr = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let log = dir.join("stderr.log");
    let mut command = Command::new(&me);
    if custom { command.env("EMBEDDED_CUSTOM_POLICY", "1"); }
    let mut server = Child(
        command
            .env("EMBEDDED_SERVE", "1")
            .env("SUPERFLUID_HOME", &home)
            .args(["--model", &model.to_string_lossy(), "--sessions"])
            .arg(dir.join("sessions"))
            .args([
                "--http",
                &addr.to_string(),
                "--max-context",
                "2048",
                "--max-batch",
                "2",
                "--no-tui",
            ])
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(&log).unwrap())
            .spawn()
            .expect("start the embedded server"),
    );
    let deadline = Instant::now() + Duration::from_secs(300);
    while !(TcpStream::connect(addr).is_ok() && http(addr, "GET", "/v1/models", "").0 == 200) {
        let stderr = std::fs::read_to_string(&log).unwrap_or_default();
        if let Ok(Some(status)) = server.0.try_wait() {
            panic!("the embedded server exited ({status}) before it came up:\n{stderr}");
        }
        assert!(
            Instant::now() < deadline,
            "the embedded server did not come up:\n{stderr}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    let stderr = std::fs::read_to_string(&log).unwrap_or_default();

    // The model runs in a worker process, and that worker is this program.
    assert!(
        stderr.contains(&format!("built in: {})", me.display())),
        "{stderr}"
    );
    // No session socket, so nothing in the place it would have gone.
    assert!(!stderr.contains("serving on "), "{stderr}");
    assert!(!home.join("superfluid.sock").exists());

    let (_, models) = http(addr, "GET", "/v1/models", "");
    let models: serde_json::Value = serde_json::from_str(&models).unwrap();
    let id = models["data"][0]["id"]
        .as_str()
        .expect("a model id")
        .to_string();
    if custom {
        assert_eq!(id, "embedded-0");
        let (status, props) = http(addr, "GET", "/props", "");
        assert_eq!(status, 200);
        let props: serde_json::Value = serde_json::from_str(&props).unwrap();
        let defaults = &props["default_generation_settings"];
        assert_eq!(defaults["temperature"], 0.0);
        assert!((defaults["repeat_penalty"].as_f64().unwrap() - 1.07).abs() < 0.0001);
    }
    let chat = serde_json::json!({
        "model": id,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8,
        "temperature": 0
    })
    .to_string();
    let (st, body) = http(addr, "POST", "/v1/chat/completions", &chat);
    assert_eq!(st, 200, "{body}");
    assert!(body.contains(r#""choices""#), "{body}");
    let messages = serde_json::json!({
        "model": id,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Say hello."}]
    })
    .to_string();
    assert_eq!(
        http(addr, "POST", "/v1/messages", &messages).0,
        404,
        "no Anthropic API"
    );
    assert_eq!(http(addr, "GET", "/api/tags", "").0, 404, "no Ollama API");
    drop(server);
}
