//! `--log-format json`, from the built binary: what a program that runs `serve` or `node join`
//! reads on stderr. Every line is one JSON event; the steps of a start have names of their own,
//! and a start that fails ends with why and the exit code.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use superfluid_daemon::fleet_join::accept_nodes;
use superfluid_daemon::fleet_manager::{FleetManager, PlacementPolicy};
use superfluid_daemon::fleet_models::HeadModels;
use superfluid_daemon::registry::ModelRegistry;
use superfluid_daemon::{Daemon, EngineHost, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkf::Lease;

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
        "pull": {"options": []},
    });
    let worker = dir.join("superfluid-worker-toyrt");
    // A pull says a great deal before its last progress line, which it marks as Superfluid's own,
    // then fails: the relay is still behind when serve hears of the failure.
    let pull = r#"i=0; while [ $i -lt 2000 ]; do echo "fetching part $i" >&2; i=$((i+1)); done
  printf '\036{"event":"pull.progress","done_bytes":7,"total_bytes":10,"bytes_per_sec":1,"secs_left":3}\n' >&2; exit 1"#;
    let script = format!(
        "#!/bin/sh\ncase \"$1\" in\n check) echo '{report}'; exit 0;;\n pull) {pull};;\nesac\nexec '{}' --engine mock \"$@\"\n",
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
fn a_pulls_progress_keeps_its_id_however_far_behind_the_relay_is() {
    let home = scratch("pull-id");
    let (worker, _) = toy_runtime(&home);
    let out = superfluid(&home)
        .env("SUPERFLUID_WORKER_TOYRT", worker)
        .args(["serve", "review/m", "--runtime", "toyrt", "--http", "127.0.0.1:0", "--sessions", "sessions", "--socket", "s.sock"])
        .args(["--offline", "--log-format", "json"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let events = events(&out.stderr);
    assert_eq!(steps(&events), ["start", "pull.start", "pull.progress", "pull.failed", "error"]);
    assert_eq!(events.iter().filter(|e| e["event"] == "log").count(), 2001, "the worker's lines and serve's own");
    for pull in events.iter().filter(|e| e["event"].as_str().is_some_and(|n| n.starts_with("pull."))) {
        assert_eq!(pull["id"], "review/m", "{pull}");
    }
    let _ = std::fs::remove_dir_all(&home);
}

/// Which side of the exec the test's own program is on, in
/// `a_process_that_starts_itself_again_keeps_its_events_on_the_real_stderr`.
const EXEC_SIDE: &str = "SUPERFLUID_EVENTS_TEST_EXEC_SIDE";

#[test]
fn a_process_that_starts_itself_again_keeps_its_events_on_the_real_stderr() {
    use std::os::unix::process::CommandExt;
    let json_mode = ["--log-format".to_string(), "json".to_string()];
    match std::env::var(EXEC_SIDE).as_deref() {
        // The test's program as a node that follows its head: it says something, then starts
        // itself again, as `fleet_join::restart` does.
        Ok("before") => {
            superfluid_daemon::events::init("node join", &json_mode).unwrap();
            superfluid_daemon::events::emit("before.exec", json!({}));
            eprintln!("superfluid: starting again");
            superfluid_daemon::events::prepare_exec();
            superfluid_daemon::events::emit("between.exec", json!({}));
            let e = Command::new(std::env::current_exe().unwrap()).args(std::env::args_os().skip(1)).env(EXEC_SIDE, "after").exec();
            panic!("cannot start again: {e}");
        }
        Ok("after") => {
            superfluid_daemon::events::init("node join", &json_mode).unwrap();
            superfluid_daemon::events::emit("after.exec", json!({}));
            return;
        }
        // An exec that fails after it was prepared: the reason still comes out as an event.
        Ok("fails") => {
            superfluid_daemon::events::init("node join", &json_mode).unwrap();
            superfluid_daemon::events::prepare_exec();
            superfluid_daemon::events::fail(1, "cannot start again: no such file");
        }
        _ => {}
    }
    let side = |side: &str| {
        let began = Instant::now();
        let out = Command::new(std::env::current_exe().unwrap())
            .args(["a_process_that_starts_itself_again_keeps_its_events_on_the_real_stderr", "--exact", "--nocapture", "--test-threads=1"])
            .env(EXEC_SIDE, side)
            .env_remove("SUPERFLUID_LOG_FORMAT")
            .stdout(Stdio::null())
            .output()
            .unwrap();
        (out, began.elapsed())
    };
    let (out, _) = side("before");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let both = events(&out.stderr);
    let names: Vec<&str> = both.iter().map(|e| e["event"].as_str().unwrap()).collect();
    assert_eq!(names, ["start", "before.exec", "log", "between.exec", "start", "after.exec"], "{both:?}");
    assert_eq!(both[2]["message"], "superfluid: starting again");
    assert_eq!(both[0]["pid"], both[4]["pid"], "one process, two images");

    let (out, took) = side("fails");
    assert_eq!(out.status.code(), Some(1));
    let failed = events(&out.stderr);
    assert_eq!(steps(&failed), ["start", "error"], "{failed:?}");
    assert_eq!(failed[1]["message"], "cannot start again: no such file");
    assert!(took < Duration::from_secs(2), "the exit does not wait on a relay it no longer writes through: {took:?}");
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

const TOKEN: &str = "a fleet token";

/// A fleet head in this process that serves `org/a` first, by id, and takes nodes that dial it;
/// the registry it serves through, to serve another model first; and its daemon builder.
fn fleet_head() -> (String, Arc<FleetManager>, Arc<ModelRegistry>, impl Fn() -> Arc<Daemon>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let manager = Arc::new(FleetManager::with_options(Vec::new(), PlacementPolicy::LoadAware, lease, 90, 0));
    let models = HeadModels::fixed(Arc::clone(&manager), vec!["org/a".into()]);
    models.know_ids(&["org/a".into(), "org/b".into()], &["org/a:q4".into(), "org/b:q4".into()]);
    let dir = scratch("head");
    let built = std::sync::atomic::AtomicU32::new(0);
    let daemon = move || {
        let wal = dir.join(format!("{}.wal", built.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
        Arc::new(Daemon::new(SessionStore::open(&wal).unwrap(), host, Box::new(MockCodec), 1))
    };
    let registry = Arc::new(ModelRegistry::with_initial("org/a", daemon(), models.loader()));
    models.attach(&registry);
    let m = Arc::clone(&manager);
    std::thread::spawn(move || accept_nodes(listener, m, TOKEN.as_bytes().to_vec(), models));
    (addr, manager, registry, daemon)
}

/// A relay to `head` that the test can cut, as a head that restarts would: every link through it
/// ends, and the node dials again.
struct Relay {
    addr: String,
    open: Arc<Mutex<Vec<TcpStream>>>,
}

impl Relay {
    fn to(head: String) -> Relay {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let open: Arc<Mutex<Vec<TcpStream>>> = Arc::default();
        let o = Arc::clone(&open);
        std::thread::spawn(move || {
            for node in listener.incoming().flatten() {
                let Ok(head) = TcpStream::connect(&head) else { continue };
                o.lock().unwrap().extend([node.try_clone().unwrap(), head.try_clone().unwrap()]);
                for (mut from, mut to) in [(node.try_clone().unwrap(), head.try_clone().unwrap()), (head, node)] {
                    std::thread::spawn(move || {
                        let _ = std::io::copy(&mut from, &mut to);
                        let _ = to.shutdown(Shutdown::Both);
                    });
                }
            }
        });
        Relay { addr, open }
    }

    fn cut(&self) {
        for sock in self.open.lock().unwrap().drain(..) {
            let _ = sock.shutdown(Shutdown::Both);
        }
    }
}

/// Waits, up to `within`, until what `child` wrote to `log` says `what`.
fn said(log: &Path, child: &mut Server, within: Duration, what: impl Fn(&str) -> bool) -> String {
    let by = Instant::now() + within;
    loop {
        let text = std::fs::read_to_string(log).unwrap_or_default();
        if what(&text) {
            return text;
        }
        let ended = child.0.try_wait().unwrap();
        assert!(ended.is_none() && Instant::now() < by, "not said ({ended:?}):\n{text}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_following_node_keeps_one_stream_across_its_restart_to_the_heads_new_model() {
    let (head, _manager, registry, daemon) = fleet_head();
    let relay = Relay::to(head);
    let home = scratch("follow");
    let log = home.join("stderr");
    let mut node = Server(
        superfluid(&home)
            .args(["node", "join", &relay.addr, "--token", TOKEN, "--identity", "follower-e", "--runtime", "mock", "--max-batch", "1"])
            .args(["--log-format", "json"])
            .env_remove("RUST_LOG")
            .stderr(std::fs::File::create(&log).unwrap())
            .spawn()
            .unwrap(),
    );
    let pid = node.0.id();
    said(&log, &mut node, Duration::from_secs(20), |t| t.contains(r#""event":"node.joined""#));

    // The head serves org/b first now, and its links end, as a head restarted with another model.
    registry.insert_loaded("org/b", daemon());
    registry.unload("org/a").unwrap();
    relay.cut();
    let dialing_b = r#""event":"node.dialing","head""#;
    said(&log, &mut node, Duration::from_secs(40), |t| {
        t.matches(r#""event":"start""#).count() == 2 && t.rsplit(r#""event":"start""#).next().is_some_and(|after| after.contains(dialing_b))
    });
    drop(node);

    let events = events(&std::fs::read(&log).unwrap());
    let starts: Vec<usize> = (0..events.len()).filter(|&i| events[i]["event"] == "start").collect();
    assert_eq!(starts.len(), 2, "{events:#?}");
    assert!(starts.iter().all(|&i| events[i]["pid"] == json!(pid) && events[i]["command"] == "node join"), "the same process, exec'd: {events:#?}");
    let (before, after) = events.split_at(starts[1]);
    let named = |events: &[Value], name: &str| events.iter().filter(|e| e["event"] == name).cloned().collect::<Vec<_>>();
    assert_eq!(named(before, "node.dialing")[0]["model"], "org/a");
    let changed = named(before, "node.head.changed");
    assert_eq!(changed.len(), 1, "{before:#?}");
    assert_eq!((&changed[0]["model"], &changed[0]["id"], &changed[0]["from"]), (&json!("org/b"), &json!("org/b:q4"), &json!("org/a")));
    let dialing = named(after, "node.dialing");
    assert_eq!((dialing.len(), &dialing[0]["model"], &dialing[0]["node"]), (1, &json!("org/b"), &json!("follower-e")), "{after:#?}");
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn a_node_the_head_removes_says_so_once_and_ends_with_exit_code_5() {
    let (head, manager, _registry, _daemon) = fleet_head();
    let home = scratch("removed");
    let log = home.join("stderr");
    let mut node = Server(
        superfluid(&home)
            .args(["node", "join", &head, "--token", TOKEN, "--identity", "removed-e", "--model", "mock", "--runtime", "mock"])
            .args(["--max-batch", "3", "--log-format", "json"])
            .env_remove("RUST_LOG")
            .stderr(std::fs::File::create(&log).unwrap())
            .spawn()
            .unwrap(),
    );
    said(&log, &mut node, Duration::from_secs(20), |t| t.contains(r#""event":"node.joined""#));
    let by = Instant::now() + Duration::from_secs(10);
    while manager.status().iter().filter(|n| n.name == "removed-e").count() == 0 {
        assert!(Instant::now() < by, "the head took the node");
        std::thread::sleep(Duration::from_millis(20));
    }
    manager.remove("removed-e").expect("listed");
    let by = Instant::now() + Duration::from_secs(15);
    let status = loop {
        if let Some(status) = node.0.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < by, "the node is still running after its head removed it");
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(status.code(), Some(superfluid_daemon::fleet_join::REMOVED_EXIT));
    let events = events(&std::fs::read(&log).unwrap());
    let steps = steps(&events);
    assert_eq!(steps.iter().filter(|s| **s == "node.removed").count(), 1, "every lane is told; the node says it once: {steps:?}");
    assert_eq!(steps[steps.len() - 2..], ["node.removed", "error"], "{events:#?}");
    let (removed, error) = (&events[events.len() - 2], &events[events.len() - 1]);
    assert_eq!(removed["head"], json!(head));
    assert_eq!(error["exit_code"], 5);
    assert_eq!(error["message"], format!("the head at {head} removed this node; it will refuse it until it restarts"));
    let _ = std::fs::remove_dir_all(&home);
}
