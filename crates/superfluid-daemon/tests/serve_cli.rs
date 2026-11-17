#![cfg(feature = "basert")]

use std::process::{Command, Output};

fn serve(tag: &str, args: &[&str]) -> Output {
    let dir = std::env::temp_dir().join(format!("superfluid-serve-cli-{tag}-{}", std::process::id()));
    Command::new(env!("CARGO_BIN_EXE_superfluid"))
        .args(["serve", "--model", "/nonexistent/model.base", "--sessions"])
        .arg(dir.join("sessions"))
        .args(args)
        .output()
        .expect("run superfluid")
}

#[test]
fn a_socket_path_too_long_to_bind_is_refused_before_loading() {
    let long = format!("/tmp/{}/superfluid.sock", "x".repeat(120));
    let out = serve("sock", &["--socket", &long, "--no-http"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("a unix socket path must be under"), "{err}");
    assert!(!err.contains("panicked"), "{err}");
}

#[test]
fn an_address_already_taken_is_refused_before_loading() {
    let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = taken.local_addr().unwrap().to_string();
    let sock = std::env::temp_dir().join(format!("bsc-{}.sock", std::process::id()));
    let out = serve("http", &["--socket", sock.to_str().unwrap(), "--http", &addr]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains(&format!("cannot listen on --http {addr}")), "{err}");
    assert!(err.contains("another process listens there"), "{err}");
    assert!(!err.contains("panicked"), "{err}");
    drop(taken);
}

#[test]
fn help_and_version_answer_on_stdout() {
    let run = |args: &[&str]| Command::new(env!("CARGO_BIN_EXE_superfluid")).args(args).output().expect("run superfluid");
    for args in [&["--help"][..], &["help"], &["serve", "--help"], &["help", "serve"], &["runtime", "--help"], &["session", "-h"]] {
        let out = run(args);
        assert!(out.status.success(), "{args:?}");
        assert!(String::from_utf8_lossy(&out.stdout).contains("Usage:"), "{args:?}");
    }
    let out = run(&["--version"]);
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("superfluid "));
    let out = run(&["serv"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("did you mean serve?"));
    let out = run(&["serve", "--port=nope", "m.gguf"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--port expects a number, got \"nope\""));
}

#[test]
fn an_install_as_runtimes_lists_it_reaches_the_runtime_actions() {
    let home = std::env::temp_dir().join(format!("superfluid-serve-cli-at-{}", std::process::id()));
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_superfluid")).arg("runtime").args(args).env("SUPERFLUID_HOME", &home).output().expect("run superfluid")
    };
    let out = run(&["remove", "basert@0.3.0-metal"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&out.stderr), "superfluid: basert has no install 0.3.0-metal (installed: none)\n");
    let out = run(&["install", "basert@0.3.0-metal"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("install basert --version 0.3.0 --backend metal`"));
    assert!(!home.exists(), "nothing was written");
}

#[test]
fn removing_a_runtime_with_more_than_one_install_needs_all() {
    let home = std::env::temp_dir().join(format!("superfluid-serve-cli-all-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    let place = |install: &str| {
        let (version, backend) = install.split_once('-').unwrap();
        let dir = home.join("runtimes/basert").join(install);
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = serde_json::json!({
            "id": "basert", "install": install, "version": version, "backend": backend,
            "link_w": [superfluid_agent::client::PROTO_VERSION], "platforms": [],
            "worker": "bin/superfluid-worker-basert", "sources": [], "tested": true,
        });
        std::fs::write(dir.join("runtime.json"), manifest.to_string()).unwrap();
    };
    let installed = || {
        let mut names: Vec<String> = std::fs::read_dir(home.join("runtimes/basert"))
            .map(|rd| rd.flatten().filter(|e| e.path().join("runtime.json").exists()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
            .unwrap_or_default();
        names.sort();
        names
    };
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_superfluid")).arg("runtime").args(args).env("SUPERFLUID_HOME", &home).output().expect("run superfluid")
    };
    place("0.3.0-metal");
    place("0.4.2-metal");
    let out = run(&["remove", "basert"]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "superfluid: basert has 2 installs (0.4.2-metal, 0.3.0-metal): `superfluid runtime remove basert <install>` removes one, \
         `superfluid runtime remove basert --all` every one\n"
    );
    assert_eq!(installed(), ["0.3.0-metal", "0.4.2-metal"], "nothing was removed");
    // An install after the '@' is never read as an option.
    for (action, name) in [("remove", "basert@--all"), ("use", "basert@--auto"), ("repair", "basert@--all")] {
        let out = run(&[action, name]);
        assert_eq!(out.status.code(), Some(2), "{action} {name}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.starts_with(&format!("superfluid: '{name}' names no install: an install's name does not begin with '-'")), "{err}");
    }
    assert_eq!(installed(), ["0.3.0-metal", "0.4.2-metal"], "nothing was removed");

    let out = run(&["remove", "basert@0.3.0-metal"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout), "removed basert 0.3.0-metal\nbasert serves from 0.4.2-metal by default\n");
    let out = run(&["remove", "basert"]);
    assert!(out.status.success(), "the only install goes without --all: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(installed(), Vec::<String>::new());

    place("0.3.0-metal");
    place("0.4.2-metal");
    let out = run(&["remove", "basert", "--all"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout), "removed basert 0.4.2-metal, 0.3.0-metal\n");
    assert!(!home.join("runtimes/basert").exists());
    let _ = std::fs::remove_dir_all(&home);
}

/// `serve` with its key from a file or the environment, on a model that is not there: the key is
/// read before anything loads. The environment's own key is cleared, so the developer's is not used.
fn serve_keyed(tag: &str, args: &[&str], env_key: Option<&str>) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!("superfluid-serve-cli-{tag}-{}", std::process::id()));
    let sock = std::env::temp_dir().join(format!("bsc-{tag}-{}.sock", std::process::id()));
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_superfluid"));
    cmd.args(["serve", "--model", "/nonexistent/model.base", "--no-http", "--sessions"])
        .arg(dir.join("sessions"))
        .arg("--socket")
        .arg(&sock)
        .args(args)
        .env_remove("SUPERFLUID_API_KEY");
    if let Some(key) = env_key {
        cmd.env("SUPERFLUID_API_KEY", key);
    }
    let out = cmd.output().expect("run superfluid");
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.code(), String::from_utf8_lossy(&out.stderr).into_owned())
}

fn key_file(tag: &str, text: &str, mode: u32) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("superfluid-serve-cli-keyfile-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("key");
    std::fs::write(&path, text).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
    path
}

#[test]
fn a_key_file_that_is_missing_or_empty_or_beside_api_key_is_refused_before_loading() {
    let empty = key_file("empty", " \n", 0o600);
    let missing = empty.with_file_name("nope");
    let (code, err) = serve_keyed("key-missing", &["--api-key-file", missing.to_str().unwrap()], Some("from-env"));
    assert_eq!(code, Some(2), "{err}");
    assert!(err.contains(&format!("--api-key-file {}: ", missing.display())), "{err}");
    let (code, err) = serve_keyed("key-empty", &["--api-key-file", empty.to_str().unwrap()], None);
    assert_eq!(code, Some(2), "{err}");
    assert!(err.contains("is empty"), "{err}");
    let (code, err) = serve_keyed("key-both", &["--api-key", "k", &format!("--api-key-file={}", empty.display())], None);
    assert_eq!(code, Some(2), "{err}");
    assert!(err.contains("--api-key and --api-key-file both give the key"), "{err}");
    assert!(!err.contains("panicked"), "{err}");
    let _ = std::fs::remove_dir_all(empty.parent().unwrap());
}

#[test]
fn a_key_file_open_to_others_is_warned_about_and_a_key_from_the_environment_is_said_so() {
    let open = key_file("open", "sk-cli-open\n", 0o644);
    let (_, err) = serve_keyed("key-open", &["--api-key-file", open.to_str().unwrap()], None);
    assert!(err.contains(&format!("WARNING --api-key-file {} is open to group or others (mode 644)", open.display())), "{err}");
    assert!(!err.contains("sk-cli-open"), "the key is not printed: {err}");
    assert!(!err.contains("no API key"), "{err}");
    let (_, err) = serve_keyed("key-env", &[], Some("sk-cli-env"));
    assert!(err.contains("authentication on, with the key in $SUPERFLUID_API_KEY"), "{err}");
    assert!(!err.contains("sk-cli-env"), "the key is not printed: {err}");
    let (_, err) = serve_keyed("key-flag", &["--api-key", "sk-cli-flag"], Some("sk-cli-env"));
    assert!(!err.contains("SUPERFLUID_API_KEY"), "the flag wins, quietly: {err}");
    let _ = std::fs::remove_dir_all(open.parent().unwrap());
}
