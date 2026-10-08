//! The llama.cpp runtime and its worker pass the kit's conformance checks.

use std::path::Path;

#[test]
fn the_runtime_conforms() {
    superfluid_adapter_kit::conformance::runtime(&superfluid_adapter_llamacpp::Llamacpp);
}

#[test]
fn its_worker_conforms() {
    superfluid_adapter_kit::conformance::worker(Path::new(env!("CARGO_BIN_EXE_superfluid-worker-llamacpp")), "llamacpp");
}

#[test]
fn its_worker_says_when_the_operator_named_the_llama_cpp_to_load() {
    let check = |named: Option<&str>| -> serde_json::Value {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_superfluid-worker-llamacpp"));
        cmd.arg("check").env_remove("SUPERFLUID_LLAMA_LIB").env("SUPERFLUID_HOME", "/nonexistent/superfluid-home");
        if let Some(dir) = named {
            cmd.env("SUPERFLUID_LLAMA_LIB", dir);
        }
        serde_json::from_slice(&cmd.output().unwrap().stdout).expect("a report")
    };
    let named = check(Some("/nonexistent/llama.cpp"));
    assert_eq!(named["named"], "SUPERFLUID_LLAMA_LIB", "{named}");
    assert!(named["available"].as_str().is_some_and(|why| why.contains("/nonexistent/llama.cpp")), "{named}");
    let searched = check(None);
    assert!(searched.get("named").is_none(), "{searched}");
}
