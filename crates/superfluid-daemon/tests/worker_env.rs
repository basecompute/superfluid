//! A runtime's worker starts without the operator's credentials in its environment.
//! One test in its own process: it sets the environment before any thread reads it.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use superfluid_daemon::runtime::{EngineHost, WorkerSpec};
use superfluid_daemon::runtimes::{run_check_model, Source, Worker};

fn script(path: &Path, body: &str) {
    std::fs::write(path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn a_worker_is_not_handed_the_operators_tokens() {
    std::env::set_var("HF_TOKEN", "hf_not_for_workers");
    std::env::set_var("SUPERFLUID_TEST_PEER_API_KEY", "nor_this");
    std::env::set_var("SUPERFLUID_TEST_KEPT", "kept");
    let dir = std::env::temp_dir().join(format!("superfluid-worker-env-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let check = dir.join("check-worker");
    script(&check, r#"echo "{\"hf\":\"${HF_TOKEN-unset}\",\"peer\":\"${SUPERFLUID_TEST_PEER_API_KEY-unset}\",\"kept\":\"${SUPERFLUID_TEST_KEPT-unset}\"}""#);
    let worker = Worker { bin: check, prefix: Vec::new(), source: Source::Installed };
    let seen = run_check_model(&worker, None).unwrap();
    assert_eq!((seen["hf"].as_str(), seen["peer"].as_str(), seen["kept"].as_str()), (Some("unset"), Some("unset"), Some("kept")), "{seen}");

    let serve = dir.join("serve-worker");
    let env = dir.join("serve-env");
    script(&serve, &format!("env > '{}'\nexit 1", env.display()));
    assert!(EngineHost::spawn_process(WorkerSpec { bin: serve, args: Vec::new(), stderr: None }).is_err());
    let env = std::fs::read_to_string(&env).unwrap();
    assert!(env.lines().any(|l| l == "SUPERFLUID_TEST_KEPT=kept"), "{env}");
    assert!(!env.contains("hf_not_for_workers") && !env.contains("nor_this"), "{env}");
    let _ = std::fs::remove_dir_all(&dir);
}
