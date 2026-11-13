//! `--log-format json`, from the built binary: what a program that runs `serve` or `node join`
//! reads on stderr. Every line is one JSON event; the steps of a start have names of their own,
//! and a start that fails ends with why and the exit code.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("superfluid-events-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `superfluid` in a home of its own (no fleet token or install of the developer's is read),
/// with the format left to each test.
fn superfluid(home: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_superfluid"));
    c.current_dir(home).env("HOME", home).env("SUPERFLUID_HOME", home.join(".superfluid")).env_remove("SUPERFLUID_LOG_FORMAT");
    c
}

/// Each line of `stderr` as an event: a JSON object with a timestamp and a name, the timestamps
/// rising down the stream.
fn events(stderr: &[u8]) -> Vec<Value> {
    let text = String::from_utf8_lossy(stderr);
    let all: Vec<Value> = text
        .lines()
        .map(|line| {
            let event: Value = serde_json::from_str(line).unwrap_or_else(|e| panic!("not JSON ({e}): {line:?}\nstderr:\n{text}"));
            assert!(event["timestamp"].is_string() && event["event"].is_string(), "{line}");
            event
        })
        .collect();
    let stamps: Vec<&str> = all.iter().map(|e| e["timestamp"].as_str().unwrap()).collect();
    assert!(stamps.windows(2).all(|w| w[0] <= w[1]), "{stamps:?}");
    all
}

/// The names of the events that are not `log`.
fn steps(events: &[Value]) -> Vec<&str> {
    events.iter().map(|e| e["event"].as_str().unwrap()).filter(|n| *n != "log").collect()
}

/// A runtime of the test's own, `toyrt`: its worker answers `check` itself and is otherwise
/// superfluid-workerd on the mock engine, which loads no weights. Its model is an empty
/// `model.toy` beside a tokenizer of printable ASCII. Returns the worker and the model.
fn toy_runtime(dir: &Path) -> (PathBuf, PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let report = json!({
        "runtime": {"id": "toyrt", "version": "v1"},
        "link_w": [superfluid_agent::client::PROTO_VERSION],
        "available": true,
        "formats": [{"id": "toy", "describe": "a toy file", "file": {"extensions": ["toy"]}}],
        "tokenizer": {"from": "huggingface"},
    });
    let worker = dir.join("superfluid-worker-toyrt");
    let script = format!(
        "#!/bin/sh\nif [ \"$1\" = check ]; then echo '{report}'; exit 0; fi\nexec '{}' --engine mock \"$@\"\n",
        env!("CARGO_BIN_EXE_superfluid-workerd")
    );
    std::fs::write(&worker, script).unwrap();
    std::fs::set_permissions(&worker, std::fs::Permissions::from_mode(0o755)).unwrap();
    let model_dir = dir.join("toy");
    std::fs::create_dir_all(&model_dir).unwrap();
    let vocab: serde_json::Map<String, Value> = (32u8..127).map(|c| ((c as char).to_string(), json!(c - 32))).collect();
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": null, "post_processor": null, "decoder": null,
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null, "end_of_word_suffix": null,
                  "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": []},
    });
    std::fs::write(model_dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    std::fs::write(model_dir.join("tokenizer_config.json"), r#"{"eos_token": "~"}"#).unwrap();
    std::fs::write(model_dir.join("model.toy"), b"").unwrap();
    (worker, model_dir.join("model.toy"))
}

/// `serve` of the toy model, its session socket and state under `home`, the HTTP API on a port
/// the system picks, nothing fetched.
fn serve_toy(home: &Path, more: &[&str]) -> Command {
    let (worker, model) = toy_runtime(home);
    let mut c = superfluid(home);
    c.env("SUPERFLUID_WORKER_TOYRT", worker)
        .args(["serve", "--runtime", "toyrt", "--max-context", "4096", "--max-batch", "2", "--http", "127.0.0.1:0", "--offline"])
        .args(["--sessions", "sessions", "--socket", "s.sock", "--model"])
        .arg(model)
        .args(more);
    c
}

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn a_serve_that_cannot_start_ends_its_stream_with_why_and_the_exit_code() {
    let home = scratch("missing");
    let child = superfluid(&home)
        .args(["serve", "--model", "/nonexistent/model.base", "--http", "127.0.0.1:0", "--sessions", "sessions", "--log-format", "json"])
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let events = events(&out.stderr);
    assert_eq!(steps(&events), ["start", "error"], "{events:?}");
    let start = &events[0];
    assert_eq!((&start["schema"], &start["command"], &start["pid"]), (&json!(1), &json!("serve"), &json!(pid)));
    assert_eq!(start["version"], env!("CARGO_PKG_VERSION"));
    let error = events.last().unwrap();
    assert_eq!(error["message"], "/nonexistent/model.base: no such file or directory");
    assert_eq!(error["exit_code"], 2);

    // Without the flag, stderr is what it always was.
    let out = superfluid(&home)
        .args(["serve", "--model", "/nonexistent/model.base", "--http", "127.0.0.1:0", "--sessions", "sessions"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(String::from_utf8_lossy(&out.stderr), "superfluid: /nonexistent/model.base: no such file or directory\n");
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn the_environment_asks_for_json_and_the_flag_overrides_it() {
    let home = scratch("env");
    let out = superfluid(&home).env("SUPERFLUID_LOG_FORMAT", "json").args(["serve", "m.gguf", "--bogus"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let events = events(&out.stderr);
    assert_eq!(steps(&events), ["start", "error"]);
    assert_eq!(events[1]["message"], "serve has no option --bogus", "the hint for a person is not an event");

    let out = superfluid(&home).env("SUPERFLUID_LOG_FORMAT", "json").args(["serve", "m.gguf", "--bogus", "--log-format=text"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(String::from_utf8_lossy(&out.stderr), "superfluid: serve has no option --bogus\nRun 'superfluid serve --help' for its options.\n");

    for (env, args) in [(None, &["serve", "m.gguf", "--log-format", "yaml"][..]), (Some("yaml"), &["serve", "m.gguf"][..])] {
        let mut c = superfluid(&home);
        if let Some(v) = env {
            c.env("SUPERFLUID_LOG_FORMAT", v);
        }
        let out = c.args(args).output().unwrap();
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.starts_with("superfluid: ") && err.contains("expects text or json, got \"yaml\""), "said as text: {err}");
    }
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn a_node_with_no_token_for_a_head_on_another_machine_says_so_as_an_event() {
    let home = scratch("node");
    // 192.0.2.1 is TEST-NET-1: never dialed, as the missing token stops the join first.
    let out = superfluid(&home).args(["node", "join", "192.0.2.1:9", "--model", "/nonexistent/m.gguf", "--log-format=json"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let events = events(&out.stderr);
    assert_eq!(steps(&events), ["start", "error"], "{events:?}");
    assert_eq!(events[0]["command"], "node join");
    let error = &events[1];
    assert!(error["message"].as_str().unwrap().starts_with("no fleet token: run `superfluid fleet token` on the head"), "{error}");
    assert_eq!(error["exit_code"], 2);
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn a_serve_says_each_step_up_to_ready_and_ready_names_the_port_it_bound() {
    let home = scratch("ready");
    let mut child = serve_toy(&home, &["--log-format", "json"]).stderr(Stdio::piped()).spawn().unwrap();
    let stderr = child.stderr.take().unwrap();
    let _server = Server(child);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut lines = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(120);
    while !lines.last().is_some_and(|l: &String| l.contains("\"event\":\"serve.ready\"")) {
        let left = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(left) {
            Ok(line) => lines.push(line),
            Err(_) => panic!("no serve.ready; stderr so far:\n{}", lines.join("\n")),
        }
    }
    let events = events(lines.join("\n").as_bytes());
    assert_eq!(steps(&events), ["start", "model.load.start", "model.load.done", "serve.ready"], "{events:?}");
    let step = |name: &str| events.iter().find(|e| e["event"] == name).unwrap().clone();
    assert_eq!(step("model.load.start")["runtime"], "toyrt");
    let done = step("model.load.done");
    assert_eq!((&done["model"], &done["runtime"], &done["context"], &done["lanes"]), (&json!("model"), &json!("toyrt"), &json!(4096), &json!(2)));
    assert!(events.iter().any(|e| e["event"] == "log" && e["message"].as_str().is_some_and(|m| m.starts_with("superfluid: runtime toyrt serves"))));
    let ready = step("serve.ready");
    assert_eq!((&ready["socket"], &ready["models"]), (&json!("s.sock"), &json!(["model"])));
    let http: std::net::SocketAddr = ready["http"].as_str().unwrap().parse().unwrap();
    assert_ne!(http.port(), 0, "the port bound, not the one asked for");

    // Ready means it answers.
    let mut conn = std::net::TcpStream::connect(http).unwrap();
    conn.write_all(format!("GET /v1/models HTTP/1.0\r\nHost: {http}\r\n\r\n").as_bytes()).unwrap();
    let mut reply = String::new();
    conn.read_to_string(&mut reply).unwrap();
    assert!(reply.starts_with("HTTP/1.0 200") || reply.starts_with("HTTP/1.1 200"), "{reply}");
    assert!(reply.contains("\"model\""), "{reply}");
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn a_model_that_does_not_load_says_so_before_the_error() {
    let home = scratch("load-failed");
    let mut serve = serve_toy(&home, &["--log-format", "json"]);
    // A tokenizer that names no end-of-sequence token does not load.
    std::fs::remove_file(home.join("toy/tokenizer_config.json")).unwrap();
    let out = serve.output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let events = events(&out.stderr);
    assert_eq!(steps(&events), ["start", "model.load.start", "model.load.failed", "error"], "{events:?}");
    let failed = events.iter().find(|e| e["event"] == "model.load.failed").unwrap();
    assert_eq!(failed["model"], "model");
    assert!(failed["error"].as_str().unwrap().contains("no EOS token"), "{failed}");
    assert_eq!(events.last().unwrap()["message"], failed["error"]);
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn a_panic_reaches_the_stream_as_one_log_event() {
    let home = scratch("panic");
    // A sessions directory that is a file: serve panics making it (an expect, not a refusal).
    std::fs::write(home.join("sessions"), b"").unwrap();
    let out = serve_toy(&home, &["--log-format", "json"]).output().unwrap();
    assert_eq!(out.status.code(), Some(101));
    let events = events(&out.stderr);
    let last = events.last().unwrap();
    assert_eq!((&last["event"], &last["level"]), (&json!("log"), &json!("ERROR")), "{events:?}");
    let message = last["message"].as_str().unwrap();
    assert!(message.starts_with("thread 'main' panicked at ") && message.contains("create sessions dir"), "{message}");
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn under_log_file_the_events_go_to_the_file() {
    let home = scratch("log-file");
    let out = serve_toy(&home, &["--log-format", "json", "--log-file", "serve.log", "--kv-bits", "3"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let before = events(&out.stderr);
    assert_eq!(steps(&before), ["start"], "what came before --log-file stays on stderr");
    let after = events(&std::fs::read(home.join("serve.log")).unwrap());
    let error = after.last().unwrap();
    assert_eq!(error["event"], "error");
    assert_eq!(error["message"], "--kv-bits 3: expected 0 (auto), 4 (q4_0), 8 (q8_0), 16 (f16) or 84 (K q8_0 / V q4_0)");

    // In text mode the same refusal is the line it always was, without the prefix.
    std::fs::remove_file(home.join("serve.log")).unwrap();
    let out = serve_toy(&home, &["--log-file", "serve.log", "--kv-bits", "3"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let log = std::fs::read_to_string(home.join("serve.log")).unwrap();
    assert!(log.ends_with("\n--kv-bits 3: expected 0 (auto), 4 (q4_0), 8 (q8_0), 16 (f16) or 84 (K q8_0 / V q4_0)\n"), "{log}");
    let _ = std::fs::remove_dir_all(&home);
}
