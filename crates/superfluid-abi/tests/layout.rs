//! Layout parity.

use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::mem::size_of;

use superfluid_abi::*;

extern "C" {
    fn abi_probe_count() -> u64;
    fn abi_probe_name(i: u64) -> *const c_char;
    fn abi_probe_value(i: u64) -> u64;
}

fn probe_table() -> HashMap<String, u64> {
    let mut m = HashMap::new();
    // SAFETY: the probe exposes a static table; indices < count are valid.
    unsafe {
        for i in 0..abi_probe_count() {
            let name = CStr::from_ptr(abi_probe_name(i))
                .to_str()
                .unwrap()
                .to_string();
            m.insert(name, abi_probe_value(i));
        }
    }
    m
}

#[test]
fn c_and_rust_layouts_match_exactly() {
    let table = probe_table();
    let mut checked: Vec<String> = Vec::new();
    for (name, ours) in superfluid_abi::layout::RUST_LAYOUT {
        assert_eq!(table.get(*name).copied(), Some(*ours), "layout mismatch for {name}");
        checked.push(name.to_string());
    }

    let mut field_sums: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for (k, v) in &table {
        if let Some(rest) = k.strip_prefix("fieldsize(") {
            let (t, _) = rest.split_once(',').expect("fieldsize key shape");
            *field_sums.entry(t.to_string()).or_default() += v;
        }
    }
    for (t, sum) in &field_sums {
        let size = table
            .get(&format!("sizeof({t})"))
            .copied()
            .unwrap_or_else(|| panic!("no sizeof entry for {t}"));
        assert_eq!(
            *sum, size,
            "{t} has implicit padding: field widths sum to {sum}, sizeof is {size}"
        );
    }

    let mut missing: Vec<&String> = table.keys().filter(|k| !checked.contains(k)).collect();
    missing.sort();
    assert!(
        missing.is_empty(),
        "probe entries with no Rust check: {missing:?}"
    );
    assert_eq!(
        checked.len(),
        table.len(),
        "Rust checks and probe entries must be 1:1"
    );
}

#[test]
fn no_implicit_padding_anywhere() {
    assert_eq!(size_of::<Array>(), 16);
    assert_eq!(size_of::<Buf>(), 16);
    assert_eq!(size_of::<TokenRange>(), 16);
    assert_eq!(size_of::<SpecStats>(), 16);
    assert_eq!(size_of::<LaneFault>(), 24);
    assert_eq!(size_of::<ShedEntry>(), 32);
    assert_eq!(size_of::<OpComplete>(), 24);
    assert_eq!(size_of::<TickTimings>(), 32);
    assert_eq!(size_of::<TickMemCounters>(), 48);
    assert_eq!(size_of::<TapSpec>(), 16);
    assert_eq!(size_of::<CapabilityReq>(), 24);
    assert_eq!(size_of::<SpaceMatch>(), 24);
    assert_eq!(size_of::<MatchResult>(), 24);
    assert_eq!(size_of::<CacheKey>(), 64);
    assert_eq!(size_of::<TokenRef>(), 24);
    assert_eq!(size_of::<RingRef>(), 16);
    assert_eq!(size_of::<SamplingParams>(), 32);
    assert_eq!(size_of::<ShedPolicy>(), 40);
    assert_eq!(size_of::<LaneAdmit>(), 144);
    assert_eq!(<LaneAdmit as AbiRecord>::MIN_PREFIX, 136);
    assert_eq!(size_of::<LaneCommit>(), 24);
    assert_eq!(size_of::<LanePrefill>(), 16);
    assert_eq!(size_of::<LaneDecode>(), 16);
    assert_eq!(size_of::<LaneRetire>(), 16);
    assert_eq!(size_of::<TickPlan>(), 168);
    assert_eq!(size_of::<AdmitResult>(), 24);
    assert_eq!(size_of::<LaneEmit>(), 72);
    assert_eq!(size_of::<LaneLogprob>(), 176);
    assert_eq!(size_of::<ShedReport>(), 32);
    assert_eq!(size_of::<TickEvents>(), 216);
    assert_eq!(size_of::<StateSpaceDesc>(), 96);
    assert_eq!(size_of::<Artifact>(), 56);
    assert_eq!(size_of::<StrategyRegistration>(), 176);
    assert_eq!(size_of::<ExactnessCert>(), 32);
    assert_eq!(size_of::<StrategyGrant>(), 56);
    assert_eq!(size_of::<OpStatus>(), 24);
    assert_eq!(size_of::<MatchCandidate>(), 48);
    assert_eq!(size_of::<StateEnvelope>(), 136);
}
