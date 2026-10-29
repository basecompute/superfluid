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
