mod common;

use std::ffi::CString;

use superfluid_abi::*;
use superfluid_engine::Engine;
use common::Harness;

struct RegBuilder {
    strategy_id: CString,
    impl_version: CString,
    arena: RecordArena,
    capabilities: Vec<CapabilityReq>,
    taps: Vec<TapSpec>,
    archs: Vec<CString>,
    claimed: u8,
}

impl RegBuilder {
    fn new(id: &str) -> RegBuilder {
        RegBuilder {
            strategy_id: CString::new(id).unwrap(),
            impl_version: CString::new("1.0.0").unwrap(),
            arena: RecordArena::new(),
            capabilities: Vec::new(),
            taps: Vec::new(),
            archs: Vec::new(),
            claimed: exactness::APPROXIMATE,
        }
    }

    fn capability(mut self, kind_id: u32) -> Self {
        self.capabilities.push(CapabilityReq {
            kind_id,
            _pad0: 0,
            params: Array::EMPTY,
        });
        self
    }

    fn tap(mut self, layer: u32) -> Self {
        self.taps.push(TapSpec {
            layer,
            tensor: tap_tensor::HIDDEN,
            dtype: 1,
            layout: 0,
        });
        self
    }

    fn arch(mut self, a: &str) -> Self {
        self.archs.push(CString::new(a).unwrap());
        self
    }

    fn build(&mut self) -> StrategyRegistration {
        let arch_ptrs: Vec<u64> = self.archs.iter().map(|c| c.as_ptr() as u64).collect();
        StrategyRegistration {
            struct_size: std::mem::size_of::<StrategyRegistration>() as u64,
            strategy_id: self.strategy_id.as_ptr(),
            impl_version: self.impl_version.as_ptr(),
            impl_hash: [1; 32],
            config_hash: [2; 32],
            artifacts: Array::EMPTY,
            taps: self.arena.push_records(&self.taps),
            capabilities: self.arena.push_records(&self.capabilities),
            target_archs: self.arena.push_records(&arch_ptrs),
            kernel_caps_required: 0,
            _pad0: 0,
            est_state_bytes: 1 << 20,
            claimed_exactness: self.claimed,
            _pad1: [0; 3],
            rng_contract_version: 1,
        }
    }
}

fn grant_slot_and_certs(grant: &StrategyGrant) -> (u32, Vec<ExactnessCert>) {
    let bounds = ArenaBounds {
        base: 0,
        len: usize::MAX,
    };
    // SAFETY: engine-owned grant arrays, next-call lifetime.
    let certs = unsafe { superfluid_abi::array::read_array(&grant.certificates, &bounds) }
        .unwrap()
        .collect();
    (grant.strategy_slot, certs)
}

#[test]
fn prompt_lookup_earns_greedy_only_cert() {
    let mut h = Harness::new();
    let mut rb = RegBuilder::new("prompt-lookup")
        .capability(strategy_cap::PROPOSAL_LINEAR)
        .arch("mock-arch");
    rb.claimed = exactness::SEED_PATH_INVARIANT;
    let reg = rb.build();
    let grant = h.engine.strategy_register(&reg).unwrap();
    let (slot, certs) = grant_slot_and_certs(grant);
    assert!(slot > 0);
    assert_eq!(certs.len(), 1);
    assert_eq!(certs[0].exactness, exactness::SEED_PATH_INVARIANT);
    assert_eq!(certs[0].sampling_modes, cert_mode::GREEDY);
    assert_eq!(grant.reserved_bytes, 1 << 20);

    let p = h.prompt(&[1, 2]);
    let cert_id = certs[0].cert_id;
    let ev = h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.strategy_slot = slot;
            a.minimum_exactness = exactness::SEED_PATH_INVARIANT;
            a
        },
        1,
        p,
    ));
    let r = ev.admit_for(1);
    assert_eq!(r.status, admit_status::ADMITTED);
    assert_eq!(r.cert_id, cert_id);
    assert_eq!(r.granted_class, exactness::SEED_PATH_INVARIANT);

    let p2 = h.prompt(&[1, 2]);
    let ev = h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.strategy_slot = slot;
            a.sampling = sampling::GPU_GUMBEL;
            a.params.temperature = 0.8;
            a.minimum_exactness = exactness::SEED_PATH_INVARIANT;
            a
        },
        2,
        p2,
    ));
    let r = ev.admit_for(2);
    assert_eq!(r.status, admit_status::REJECTED);
    assert_eq!(r.reject_code, Status::RejectCertUnmatched.raw() as u32);
    assert!(h.engine.lane_sequence(2).is_none(), "lane not created");

    let p3 = h.prompt(&[1, 2]);
    let ev = h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.strategy_slot = slot;
            a.sampling = sampling::GPU_GUMBEL;
            a.params.temperature = 0.8;
            a.minimum_exactness = exactness::APPROXIMATE;
            a.allow_approximate = 1;
            a
        },
        3,
        p3,
    ));
    let r = ev.admit_for(3);
    assert_eq!(r.status, admit_status::ADMITTED);
    assert_eq!(r.granted_class, exactness::APPROXIMATE);
    assert_eq!(r.cert_id, 0);
}

#[test]
fn registration_fails_closed() {
    let mut h = Harness::new();

    let reg = RegBuilder::new("future-method")
        .capability(0xDEAD)
        .build_owned();
    assert_eq!(
        h.engine.strategy_register(&reg.1).unwrap_err(),
        Status::RegistrationRefused
    );

    let reg = RegBuilder::new("eagle3").tap(999).build_owned();
    assert_eq!(
        h.engine.strategy_register(&reg.1).unwrap_err(),
        Status::RegistrationRefused
    );

    let reg = RegBuilder::new("draft-model")
        .arch("llama-only")
        .build_owned();
    assert_eq!(
        h.engine.strategy_register(&reg.1).unwrap_err(),
        Status::RegistrationRefused
    );

    let reg = RegBuilder::new("draft-model")
        .capability(strategy_cap::PROPOSAL_LINEAR)
        .tap(3)
        .arch("mock-arch")
        .build_owned();
    assert!(h.engine.strategy_register(&reg.1).is_ok());
}

impl RegBuilder {
    fn build_owned(mut self) -> (RegBuilder, StrategyRegistration) {
        let reg = self.build();
        (self, reg)
    }
}

#[test]
fn host_composition_rules() {
    let mut h = Harness::new();

    let reg = RegBuilder::new("draft-model")
        .capability(strategy_cap::PROPOSAL_LINEAR)
        .build_owned();
    let grant = h.engine.strategy_register(&reg.1).unwrap();
    let slot = grant.strategy_slot;
    let p = h.prompt(&[1]);
    let err = h.tick_err(h.plan().admit_with(
        |mut a| {
            a.sampling = sampling::HOST;
            a.strategy_slot = slot;
            a
        },
        1,
        p,
    ));
    assert_eq!(err, Status::RejectHostRules);

    let reg = RegBuilder::new("prompt-lookup")
        .capability(strategy_cap::HOST_COMPATIBLE)
        .build_owned();
    let grant = h.engine.strategy_register(&reg.1).unwrap();
    let slot2 = grant.strategy_slot;
    let p2 = h.prompt(&[1]);
    let ev = h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.sampling = sampling::HOST;
            a.strategy_slot = slot2;
            a.allow_approximate = 1;
            a.host_sampler_identity = [7; 32];
            a
        },
        2,
        p2,
    ));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(ev.admit_for(2).granted_class, exactness::APPROXIMATE);
}

#[test]
fn host_certificate_matches_by_identity_set() {
    let mut h = Harness::new();
    let reg = RegBuilder::new("prompt-lookup")
        .capability(strategy_cap::HOST_COMPATIBLE)
        .build_owned();
    let grant = h.engine.strategy_register(&reg.1).unwrap();
    let slot = grant.strategy_slot;

    let known = [42u8; 32];
    let cert = h.engine.make_cert(
        exactness::DISTRIBUTION_EXACT,
        cert_mode::HOST,
        cert_domain::ANY,
        false,
        vec![known],
    );
    h.engine.grant_certs_for_test("prompt-lookup", vec![cert]);

    let p = h.prompt(&[1]);
    let ev = h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.sampling = sampling::HOST;
            a.strategy_slot = slot;
            a.minimum_exactness = exactness::DISTRIBUTION_EXACT;
            a.host_sampler_identity = known;
            a
        },
        1,
        p,
    ));
    assert_eq!(ev.admit_for(1).status, admit_status::ADMITTED);
    assert_eq!(ev.admit_for(1).granted_class, exactness::DISTRIBUTION_EXACT);

    let p2 = h.prompt(&[1]);
    let ev = h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.sampling = sampling::HOST;
            a.strategy_slot = slot;
            a.minimum_exactness = exactness::DISTRIBUTION_EXACT;
            a.host_sampler_identity = [9; 32];
            a
        },
        2,
        p2,
    ));
    assert_eq!(ev.admit_for(2).status, admit_status::REJECTED);

    let p3 = h.prompt(&[1]);
    let ev = h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.sampling = sampling::HOST;
            a.strategy_slot = slot;
            a.minimum_exactness = exactness::DISTRIBUTION_EXACT;
            a
        },
        3,
        p3,
    ));
    assert_eq!(ev.admit_for(3).status, admit_status::REJECTED);
}

#[test]
fn required_cert_id_is_exact_match() {
    let mut h = Harness::new();
    let reg = RegBuilder::new("prompt-lookup")
        .capability(strategy_cap::PROPOSAL_LINEAR)
        .arch("mock-arch")
        .build_owned();
    let grant = h.engine.strategy_register(&reg.1).unwrap();
    let (slot, certs) = grant_slot_and_certs(grant);
    let real_id = certs[0].cert_id;

    let p = h.prompt(&[1]);
    let ev = h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.strategy_slot = slot;
            a.required_cert_id = real_id + 100;
            a
        },
        1,
        p,
    ));
    assert_eq!(ev.admit_for(1).status, admit_status::REJECTED);

    let p2 = h.prompt(&[1]);
    let ev = h.tick_ok(h.plan().admit_with(
        |mut a| {
            a.strategy_slot = slot;
            a.required_cert_id = real_id;
            a
        },
        2,
        p2,
    ));
    assert_eq!(ev.admit_for(2).status, admit_status::ADMITTED);
    assert_eq!(ev.admit_for(2).cert_id, real_id);
}

#[test]
fn scratch_host_compatible_spec_stats() {
    let mut h = Harness::new();
    let reg = RegBuilder::new("prompt-lookup")
        .capability(strategy_cap::HOST_COMPATIBLE)
        .build_owned();
    let grant = h.engine.strategy_register(&reg.1).unwrap();
    let slot = grant.strategy_slot;
    let p = h.prompt(&[1, 2, 3]);
    let ev = h.tick_ok(
        h.plan()
            .admit_with(
                |mut a| {
                    a.sampling = sampling::HOST;
                    a.strategy_slot = slot;
                    a.allow_approximate = 1;
                    a.host_sampler_identity = [7; 32];
                    a
                },
                2,
                p,
            )
            .prefill(2, 0, 3)
            .decode(2, 1),
    );
    let e = ev.emit_for(2);
    eprintln!(
        "n_tokens={} finish={} spec.proposed={} spec.accepted={} cert_id={}",
        e.n_tokens, e.finish, e.spec.proposed, e.spec.accepted, e.spec.cert_id_in_effect
    );
    assert_eq!(e.n_tokens, 0);
}
