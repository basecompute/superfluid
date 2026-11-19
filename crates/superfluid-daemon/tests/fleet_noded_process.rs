//! The node-agent PROCESS, end to end.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use superfluid_daemon::fleet::FleetHead;
use superfluid_daemon::{Daemon, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkf::Lease;

static DIRS: AtomicU64 = AtomicU64::new(0);

fn local_mock() -> Daemon {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-noded-proc-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn noded_process_serves_a_fleet_generation() {
    let prompt: Vec<u32> = (500..540).collect();
    let params = GenParams::default();
    let max = 12u32;
    let (local_tokens, _) = local_mock().generate_tokens(prompt.clone(), params, max, 0).unwrap();

    // A home of its own: a fleet token in the developer's would make the node demand one.
    let home = empty_home("generation");
    let mut child = Command::new(env!("CARGO_BIN_EXE_superfluid-noded"))
        .args(["--listen", "127.0.0.1:0", "--engine", "mock", "--identity", "proc-node"])
        .env("HOME", &home)
        .env("USERPROFILE", &home)
        .env("SUPERFLUID_HOME", home.join(".superfluid"))
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn superfluid-noded");
    let stderr = child.stderr.take().unwrap();
    let child = Child(child);

    let mut addr = String::new();
    let mut reader = BufReader::new(stderr);
    let mut line = String::new();
    while reader.read_line(&mut line).unwrap() > 0 {
        if let Some(idx) = line.find(" on 127.0.0.1:") {
            let tail = &line[idx + 4..];
            addr = tail.split_whitespace().next().unwrap().to_string();
            break;
        }
        line.clear();
    }
    assert!(!addr.is_empty(), "node did not report its listen address");

    let lease = Lease {
        duration_ms: 60_000,
        renew_by_ms: 40_000,
    };
    let mut head = FleetHead::connect(&addr, Vec::new()).expect("connect noded");
    assert_eq!(head.node().node_identity, "proc-node");
    let session = 3u64;
    head.assign(session, params, 0, "chatml", lease).unwrap();
    head.append(session, &prompt).unwrap();
    let gen = head.generate(session, max as u64, 60_000).unwrap();

    assert!(!gen.expired);
    assert_eq!(gen.tokens, local_tokens, "noded process diverged from local");
    head.wal().assert_single_contiguous_streams();
    drop(head);
    drop(child);
}

/// `superfluid-noded` with `args`, and the address it reports it listens on.
fn noded_listening(args: &[&str], home: &std::path::Path) -> (Child, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_superfluid-noded"))
        .args(args)
        .env("HOME", home)
        .env("SUPERFLUID_HOME", home.join(".superfluid"))
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn superfluid-noded");
    let mut reader = BufReader::new(child.stderr.take().unwrap());
    let child = Child(child);
    let mut line = String::new();
    while reader.read_line(&mut line).unwrap() > 0 {
        if let Some(idx) = line.find(" on 127.0.0.1:") {
            let addr = line[idx + 4..].split_whitespace().next().unwrap().to_string();
            // The rest of what it says goes nowhere, so it never blocks on a full pipe.
            std::thread::spawn(move || std::io::copy(&mut reader, &mut std::io::sink()));
            return (child, addr);
        }
        line.clear();
    }
    panic!("node did not report its listen address");
}

#[test]
fn noded_serves_several_models_each_on_its_own_engine() {
    let home = empty_home("several");
    let args = ["--listen", "127.0.0.1:0", "--engine", "mock", "--identity", "two", "--max-batch", "2", "--model", "mock-a", "--model", "mock-b"];
    let (child, addr) = noded_listening(&args, &home);
    let mut head = FleetHead::connect(&addr, Vec::new()).expect("connect noded");
    assert_eq!(head.node().loadable_models, ["mock-a", "mock-b"]);
    assert_eq!(head.node().capacity.max_lanes, 4, "each model's lanes");
    let lease = Lease { duration_ms: 60_000, renew_by_ms: 40_000 };
    let prompt: Vec<u32> = (500..540).collect();
    let (local_tokens, _) = local_mock().generate_tokens(prompt.clone(), GenParams::default(), 8, 0).unwrap();
    for (session, model) in [(1, "mock-b"), (2, "mock-a")] {
        head.assign_for(session, GenParams::default(), 0, "chatml", model, lease).unwrap();
        head.append(session, &prompt).unwrap();
        let gen = head.generate(session, 8, 60_000).unwrap();
        assert_eq!(gen.tokens, local_tokens, "{model}");
    }
    drop(head);
    drop(child);

    let out = noded(&["--listen", "127.0.0.1:0", "--engine", "mock", "--model", "a", "--model", "b", "--model-name", "x"], &home);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("once for each --model"), "{}", String::from_utf8_lossy(&out.stderr));
    let out = noded(&["--listen", "127.0.0.1:0", "--engine", "mock", "--model", "a", "--model", "a"], &home);
    assert_ne!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stderr).contains("given twice"), "{}", String::from_utf8_lossy(&out.stderr));
}

fn noded(args: &[&str], home: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_superfluid-noded"))
        .args(args)
        .env("HOME", home)
        .env("USERPROFILE", home)
        .env("SUPERFLUID_HOME", home.join(".superfluid"))
        .stdin(Stdio::null())
        .output()
        .expect("run superfluid-noded")
}

fn empty_home(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("superfluid-noded-home-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn noded_refuses_an_open_listener_before_pulling_anything() {
    let home = empty_home("refuse");
    let start = std::time::Instant::now();
    let out = noded(&["--listen", "0.0.0.0:0", "--model", "nobody-here/no-such-model"], &home);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("without a fleet token"), "{err}");
    assert!(!err.contains("pulling"), "nothing is pulled before the refusal: {err}");
    // Windows scans a program just built before its first start, which can take that long by itself.
    let within = if cfg!(windows) { 60 } else { 10 };
    assert!(start.elapsed() < std::time::Duration::from_secs(within));
    assert!(!home.join(".superfluid/runtimes").exists(), "no runtime was installed either");
}

#[test]
fn noded_needs_a_model_and_names_a_missing_file() {
    let home = empty_home("model");
    let out = noded(&["--listen", "127.0.0.1:0"], &home);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--model <path|org/model[:tag]>"));

    let out = noded(&["--listen", "127.0.0.1:0", "--model", "/no/such/model.base"], &home);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_ne!(out.status.code(), Some(0), "{err}");
    assert!(err.contains("/no/such/model.base"), "{err}");
}

/// The engine flags `serve` takes, which a node takes too, refused as `serve` refuses them and
/// before the node listens; given right, the node serves as it does without them.
#[test]
fn noded_takes_the_engine_flags_serve_takes() {
    let home = empty_home("engine-flags");
    for (args, says) in [
        (&["--spec-draft-tokens", "16"][..], "--spec-draft-tokens must be in 1..=15"),
        (&["--kv-bits", "5"][..], "--kv-bits 5: expected 0 (auto)"),
        (&["--park-lossy", "--kv-bits", "8"][..], "--park-lossy requires F16 KV pools; --kv-bits 8 conflicts"),
        (&["--prefill-budget", "lots"][..], "--prefill-budget expects a number"),
        (&["--mlx-venv", "/env"][..], "the mock engine takes no runtime option"),
        (&["--otlp-endpoint", "not a url"][..], "--otlp-endpoint"),
    ] {
        let out = noded(&[&["--listen", "127.0.0.1:0", "--engine", "mock"][..], args].concat(), &home);
        let err = String::from_utf8_lossy(&out.stderr);
        assert_ne!(out.status.code(), Some(0), "{args:?}: {err}");
        assert!(err.contains(says), "{args:?}: {err}");
        assert!(!err.contains("serving 'mock'"), "{args:?} refused before the node serves: {err}");
    }

    let mut child = Command::new(env!("CARGO_BIN_EXE_superfluid-noded"))
        .args(["--listen", "127.0.0.1:0", "--engine", "mock", "--identity", "flags-node", "--speculate", "prompt-lookup"])
        .args(["--spec-draft-tokens", "4", "--no-spec-adaptive", "--prefill-budget", "256", "--tick-target-ms", "200"])
        .args(["--kv-bits", "16", "--park-budget-gb", "1"])
        .env("HOME", &home)
        .env("SUPERFLUID_HOME", home.join(".superfluid"))
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn superfluid-noded");
    let stderr = child.stderr.take().unwrap();
    let _child = Child(child);
    let mut addr = String::new();
    let mut said = String::new();
    for line in BufReader::new(stderr).lines() {
        let line = line.unwrap();
        said.push_str(&line);
        said.push('\n');
        if let Some(idx) = line.find(" on 127.0.0.1:") {
            addr = line[idx + 4..].split_whitespace().next().unwrap().to_string();
            break;
        }
    }
    assert!(!addr.is_empty(), "the node serves with the flags: {said}");
    let mut head = FleetHead::connect(&addr, Vec::new()).expect("connect noded");
    let (prompt, params) = ((500..540).collect::<Vec<u32>>(), GenParams::default());
    head.assign(1, params, 0, "chatml", Lease { duration_ms: 60_000, renew_by_ms: 40_000 }).unwrap();
    head.append(1, &prompt).unwrap();
    let gen = head.generate(1, 12, 60_000).unwrap();
    assert_eq!(gen.tokens.len(), 12, "{said}");
}

#[cfg(feature = "basert")]
#[test]
fn noded_sizes_its_window_for_the_device() {
    let model = std::env::var("BASERT_TEST_MODEL").map(std::path::PathBuf::from).ok().or_else(|| {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen3-0.6B-Q4_K_M.base");
        p.exists().then_some(p)
    });
    let Some(model) = model.filter(|m| m.extension().is_some_and(|e| e == "base")) else {
        eprintln!("SKIP: no .base fixture (set BASERT_TEST_MODEL)");
        return;
    };
    if superfluid_engine_ffi::libbasert::require().is_none() {
        eprintln!("SKIP: libbaseRT is not installed");
        return;
    }
    let home = empty_home("sizes");
    let mut child = Command::new(env!("CARGO_BIN_EXE_superfluid-noded"))
        .args(["--listen", "127.0.0.1:0", "--max-batch", "2", "--model"])
        .arg(&model)
        .env("HOME", &home)
        .env("USERPROFILE", &home)
        .env("SUPERFLUID_HOME", home.join(".superfluid"))
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn superfluid-noded");
    let stderr = child.stderr.take().unwrap();
    let _child = Child(child);
    let mut all = String::new();
    let mut summary = String::new();
    for line in BufReader::new(stderr).lines() {
        let line = line.unwrap();
        all.push_str(&line);
        all.push('\n');
        if line.contains("context window") {
            summary = line;
            break;
        }
    }
    assert!(!summary.is_empty(), "the node printed no window before it stopped:\n{all}");
    assert!(summary.contains("sized for this device") && summary.contains("2 lanes"), "{summary}");
}
