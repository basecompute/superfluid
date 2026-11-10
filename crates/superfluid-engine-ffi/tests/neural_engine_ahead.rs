//! `prepare_neural_engine` against the REAL engine: the window reaches the engine, which builds the programs for the
//! prefill chunk a load at that window sizes. Set `BASERT_TEST_MODEL_ANE` to a `.base` bundle the split takes (a
//! Qwen3.6/3.8 27B, or a Qwen3.5/3.6 MoE for the long-window chunk on an M5 with 40 GiB or more to plan with), and
//! `BASERT_LIB` to a libbaseRT with `baseRT_prepare_neural_engine`.

use std::path::PathBuf;

use superfluid_engine_ffi::{libbasert, TokenizerHandle};

fn model() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("BASERT_TEST_MODEL_ANE")?);
    libbasert::require()?;
    Some(p)
}

#[test]
fn the_windows_that_share_a_prefill_chunk_share_its_programs() {
    let Some(m) = model() else {
        eprintln!("SKIP: set BASERT_TEST_MODEL_ANE (and BASERT_LIB)");
        return;
    };
    let Some(_) = libbasert::prepare_neural_engine(&m, 4096).expect("prepare at 4096") else {
        eprintln!("SKIP: the library predates the call, or its split takes nothing of {} here", m.display());
        return;
    };
    // A load given 0 picks a short window, whose chunk is the 4096 one's.
    assert_eq!(libbasert::prepare_neural_engine(&m, 0), Ok(Some(0)), "0 sizes the 4096 window's chunk");
    // Every window from 16384 up sizes one chunk (the widened one where the engine widens it, else the same).
    let trained = TokenizerHandle::load(&m).expect("tokenizer").trained_context() as i32;
    assert!(trained >= 16384, "{} is trained for {trained}; the long-window case needs 16384 or more", m.display());
    libbasert::prepare_neural_engine(&m, 32768).expect("prepare at 32768");
    for window in [16384, trained] {
        assert_eq!(libbasert::prepare_neural_engine(&m, window), Ok(Some(0)), "{window} sizes the 32768 window's chunk");
    }
}
