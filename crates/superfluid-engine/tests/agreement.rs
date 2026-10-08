//! The numeric-agreement checks the real-runtime suites compare shapes with.

use superfluid_abi::LaneLogprob;
use superfluid_engine::testing::{rows_agree, tokens_agree, top2_margin};

#[test]
fn rows_agree_within_the_tolerance_and_name_the_worst_logit_past_it() {
    let a = [1.0f32, 5.0, -2.0];
    assert!(rows_agree(&a, &[1.2, 4.9, -2.1], 0.25).is_ok());
    let e = rows_agree(&a, &[1.0, 3.0, -2.0], 0.25).unwrap_err();
    assert!(e.contains("id 1"), "{e}");
    assert!(rows_agree(&a, &a[..2], 1.0).is_err(), "rows of different widths");
}

#[test]
fn a_logit_that_is_not_a_number_agrees_with_nothing() {
    let a = [1.0f32, 2.0];
    let e = rows_agree(&a, &[f32::NAN, f32::NAN], 0.5).unwrap_err();
    assert!(e.contains("id 0") && e.contains("NaN"), "{e}");
    assert!(rows_agree(&[f32::NAN, 2.0], &a, 0.5).is_err(), "in the reference too");
    assert!(rows_agree(&[1.0, f32::NAN], &[1.0, f32::NAN], 0.5).is_err(), "NaN on both sides is not agreement");

    let masked = [1.0f32, f32::NEG_INFINITY];
    assert!(rows_agree(&masked, &[1.1, f32::NEG_INFINITY], 0.5).is_ok(), "an id masked on both sides");
    assert!(rows_agree(&masked, &[1.1, 3.0], 0.5).is_err(), "masked on one side only");
    assert!(rows_agree(&masked, &[1.1, f32::INFINITY], 0.5).is_err(), "infinities of opposite sign");
}

#[test]
fn tokens_agree_up_to_a_near_tie_and_not_past_a_confident_flip() {
    let reference = [5u32, 6, 7, 8];
    assert!(tokens_agree(&reference, &[3.0; 4], &reference, 0.5).is_ok());
    assert!(tokens_agree(&reference, &[3.0, 3.0, 0.6, 3.0], &[5, 6, 9, 1], 0.5).is_ok());
    assert!(tokens_agree(&reference, &[3.0, 3.0, 3.0, 3.0], &[5, 6, 9, 1], 0.5).is_err());
    assert!(tokens_agree(&reference, &[3.0; 4], &reference[..3], 0.5).is_err());
}

#[test]
fn a_records_top2_margin_is_its_first_two_logprobs_apart() {
    let mut r = LaneLogprob { n_top: 2, ..Default::default() };
    r.top_logprobs[0] = -0.1;
    r.top_logprobs[1] = -2.6;
    assert!((top2_margin(&r) - 2.5).abs() < 1e-6);
    r.n_top = 1;
    assert_eq!(top2_margin(&r), f32::INFINITY);
}
