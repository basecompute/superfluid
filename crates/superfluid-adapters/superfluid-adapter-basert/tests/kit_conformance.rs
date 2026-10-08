//! The basert runtime and its worker pass the kit's conformance checks, with or without a libbaseRT
//! here.

use std::path::Path;

#[test]
fn the_runtime_conforms() {
    superfluid_adapter_kit::conformance::runtime(&superfluid_adapter_basert::Basert);
}

#[test]
fn its_worker_conforms() {
    superfluid_adapter_kit::conformance::worker(Path::new(env!("CARGO_BIN_EXE_superfluid-worker-basert")), "basert");
}

#[test]
fn its_worker_says_when_the_operator_named_the_libbasert_to_load() {
    let check = |env: Option<&str>, flag: Option<&str>| -> serde_json::Value {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_superfluid-worker-basert"));
        cmd.arg("check").env_remove("BASERT_LIB").env("SUPERFLUID_HOME", "/nonexistent/superfluid-home");
        if let Some(lib) = env {
            cmd.env("BASERT_LIB", lib);
        }
        if let Some(lib) = flag {
            cmd.args(["--basert-lib", lib]);
        }
        serde_json::from_slice(&cmd.output().unwrap().stdout).expect("a report")
    };
    let named = check(Some("/nonexistent/libbaseRT.dylib"), None);
    assert_eq!(named["named"], "BASERT_LIB", "{named}");
    assert!(named["available"].as_str().is_some_and(|why| why.contains("/nonexistent/libbaseRT.dylib")), "{named}");
    let flagged = check(None, Some("/nonexistent/libbaseRT.dylib"));
    assert_eq!(flagged["named"], "--basert-lib", "{flagged}");
    let searched = check(None, None);
    assert!(searched.get("named").is_none(), "{searched}");
}
