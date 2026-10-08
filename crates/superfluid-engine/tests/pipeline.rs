//! Pipeline-parallel forward contract.

use superfluid_engine::{Engine, EngineConfig, MockEngine, StageOutput};

fn full_token(eng: &mut MockEngine, tokens: &[u32]) -> u32 {
    let n = eng.n_layers();
    match eng.forward_stage(tokens, 0, n, None).unwrap() {
        StageOutput::Token(t) => t,
        StageOutput::Hidden(_) => panic!("a full [0,n) forward must reach the last layer -> Token"),
    }
}

#[test]
fn two_stage_pipeline_equals_full_forward_at_every_split() {
    let mut eng = MockEngine::new(EngineConfig::default());
    let n = eng.n_layers();
    assert!(n >= 2, "need at least two layers to split");
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    let full = full_token(&mut eng, &tokens);

    for k in 1..n {
        let h = match eng.forward_stage(&tokens, 0, k, None).unwrap() {
            StageOutput::Hidden(h) => h,
            StageOutput::Token(_) => panic!("a stage before the last must yield a boundary Hidden"),
        };
        let piped = match eng.forward_stage(&[], k, n, Some(h)).unwrap() {
            StageOutput::Token(t) => t,
            StageOutput::Hidden(_) => panic!("the last stage must yield a Token"),
        };
        assert_eq!(piped, full, "2-stage split at layer {k} != full forward");
    }
}

#[test]
fn three_stage_pipeline_equals_full_forward() {
    let mut eng = MockEngine::new(EngineConfig::default());
    let n = eng.n_layers();
    let tokens: Vec<u32> = vec![10, 20, 30];
    let full = full_token(&mut eng, &tokens);

    let (a, b) = (n / 3, 2 * n / 3);
    let h1 = match eng.forward_stage(&tokens, 0, a, None).unwrap() {
        StageOutput::Hidden(h) => h,
        _ => panic!("stage 0 -> Hidden"),
    };
    let h2 = match eng.forward_stage(&[], a, b, Some(h1)).unwrap() {
        StageOutput::Hidden(h) => h,
        _ => panic!("stage 1 -> Hidden"),
    };
    let tok = match eng.forward_stage(&[], b, n, Some(h2)).unwrap() {
        StageOutput::Token(t) => t,
        _ => panic!("stage 2 -> Token"),
    };
    assert_eq!(tok, full, "3-stage pipeline != full forward");
}

#[test]
fn different_inputs_generally_differ_and_boundary_activation_is_opaque_bytes() {
    let mut eng = MockEngine::new(EngineConfig::default());
    let n = eng.n_layers();
    let h = match eng.forward_stage(&[7, 7, 7], 0, n / 2, None).unwrap() {
        StageOutput::Hidden(h) => h,
        _ => panic!(),
    };
    assert!(!h.is_empty(), "boundary activation must carry state");
    assert_ne!(full_token(&mut eng, &[1, 2, 3]), full_token(&mut eng, &[4, 5, 6]));
}

#[test]
fn out_of_range_stage_is_rejected() {
    let mut eng = MockEngine::new(EngineConfig::default());
    let n = eng.n_layers();
    assert!(eng.forward_stage(&[1], 0, n + 1, None).is_err(), "end past n_layers rejected");
    assert!(eng.forward_stage(&[1], 5, 3, None).is_err(), "start > end rejected");
}
