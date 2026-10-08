//! A model directory is data, not code (see `MlxRuntime::open`).

#[test]
fn a_model_that_brings_its_own_code_is_refused_before_it_loads() {
    use superfluid_adapter_mlx::{MlxConfig, MlxError, MlxRuntime};
    let dir = std::env::temp_dir().join(format!("superfluid-mlx-owncode-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let marker = dir.join("ran");
    std::fs::write(dir.join("config.json"), r#"{"model_type": "custom", "model_file": "model.py"}"#).unwrap();
    std::fs::write(dir.join("model.py"), format!("open({:?}, 'w').write('ran')\n", marker.to_string_lossy())).unwrap();
    let open = || MlxRuntime::open(MlxConfig { model_path: dir.clone(), max_seq_len: 512, max_batch: 2, ..Default::default() });
    let refused = |said: &str| -> String {
        let why = match open() {
            Err(MlxError::Load(why)) => why,
            other => panic!("{said}: expected a refusal, got {:?}", other.map(|_| "a runtime")),
        };
        assert!(why.contains("names model code of its own (model_file: model.py)"), "{said}: {why}");
        assert!(why.contains("SUPERFLUID_MLX_TRUST_MODEL_CODE=1"), "{said}: {why}");
        assert!(!marker.exists(), "{said}: nothing of the model ran");
        why
    };
    std::env::remove_var(TRUST);
    refused("unset");
    for no in ["", "0", "false", "no", "true"] {
        std::env::set_var(TRUST, no);
        let why = refused(&format!("{TRUST}={no}"));
        assert!(why.contains(&format!("it is set to {no:?}")), "{why}");
    }
    std::env::set_var(TRUST, "1");
    match open() {
        Err(MlxError::Load(why)) => assert!(!why.contains("names model code of its own"), "{why}"),
        Err(_) => {}
        Ok(_) => panic!("a directory with no weights loaded"),
    }
    std::env::remove_var(TRUST);
    let _ = std::fs::remove_dir_all(&dir);
}

const TRUST: &str = "SUPERFLUID_MLX_TRUST_MODEL_CODE";
