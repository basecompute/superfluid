//! Strategy registration.

use superfluid_abi::{
    cert_domain, cert_mode, exactness, sampling, strategy_cap, CapabilityReq,
    ExactnessCert, LaneAdmit, SamplingParams, StrategyRegistration,
    TapSpec, RecordArena, Status,
};

#[derive(Debug, Clone)]
pub struct Cert {
    pub cert_id: u32,
    pub exactness: u8,
    pub sampling_modes: u8,
    pub grammar_allowed: bool,
    pub logit_bias_allowed: bool,
    pub param_domain: u32,
    pub verification_shape_class: u32,
    pub admissible_host_identities: Vec<[u8; 32]>,
}

impl Cert {
    pub fn to_abi(&self, arena: &mut RecordArena) -> ExactnessCert {
        ExactnessCert {
            cert_id: self.cert_id,
            exactness: self.exactness,
            sampling_modes: self.sampling_modes,
            grammar_allowed: self.grammar_allowed as u8,
            logit_bias_allowed: self.logit_bias_allowed as u8,
            param_domain: self.param_domain,
            verification_shape_class: self.verification_shape_class,
            admissible_host_identities: arena.push_records(&self.admissible_host_identities),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RegisteredStrategy {
    pub slot: u32,
    pub strategy_id: String,
    pub impl_hash: [u8; 32],
    pub config_hash: [u8; 32],
    pub host_compatible: bool,
    pub certs: Vec<Cert>,
    pub reserved_bytes: u64,
}

#[allow(dead_code)]
pub struct RegistrationCheck<'a> {
    pub reg: &'a StrategyRegistration,
    pub capabilities: Vec<CapabilityReq>,
    pub taps: Vec<TapSpec>,
    pub target_archs: Vec<String>,
}

impl RegistrationCheck<'_> {
    pub fn validate(&self, target_layers: u32, target_arch: &str) -> Result<bool, Status> {
        let mut host_compatible = false;
        for cap in &self.capabilities {
            match cap.kind_id {
                strategy_cap::PROPOSAL_LINEAR
                | strategy_cap::PROPOSAL_BLOCK
                | strategy_cap::PROPOSAL_TREE
                | strategy_cap::MASK_ANCESTOR => {}
                strategy_cap::HOST_COMPATIBLE => host_compatible = true,
                _ => return Err(Status::RegistrationRefused),
            }
        }
        for tap in &self.taps {
            if tap.layer >= target_layers {
                return Err(Status::RegistrationRefused);
            }
        }
        if !self.target_archs.is_empty() && !self.target_archs.iter().any(|a| a == target_arch) {
            return Err(Status::RegistrationRefused);
        }
        Ok(host_compatible)
    }
}

pub fn resolve_certificate(
    strategy: &RegisteredStrategy,
    admit: &LaneAdmit,
) -> Option<(u32, u8)> {
    let mode_bit = match admit.sampling {
        sampling::GPU_GREEDY => cert_mode::GREEDY,
        sampling::GPU_GUMBEL => cert_mode::GUMBEL,
        sampling::HOST => cert_mode::HOST,
        _ => return None,
    };
    let zero_identity = admit.host_sampler_identity == [0u8; 32];
    strategy
        .certs
        .iter()
        .filter(|c| {
            if admit.required_cert_id != 0 && c.cert_id != admit.required_cert_id {
                return false;
            }
            if c.sampling_modes & mode_bit == 0 {
                return false;
            }
            if admit.sampling == sampling::HOST
                && (zero_identity || !c.admissible_host_identities.contains(&admit.host_sampler_identity))
            {
                return false;
            }
            if !param_domain_matches(c.param_domain, &admit.params) {
                return false;
            }
            if admit.grammar_handle != 0 && !c.grammar_allowed {
                return false;
            }
            c.exactness >= admit.minimum_exactness
        })
        .max_by_key(|c| c.exactness)
        .map(|c| (c.cert_id, c.exactness))
}

fn param_domain_matches(domain: u32, p: &SamplingParams) -> bool {
    let penalty_free = p.freq_penalty == 0.0
        && p.presence_penalty == 0.0
        && (p.repeat_penalty == 0.0 || p.repeat_penalty == 1.0);
    match domain {
        cert_domain::GREEDY_ONLY => p.temperature == 0.0 && penalty_free,
        cert_domain::PENALTY_FREE => penalty_free,
        cert_domain::ANY => true,
        _ => false,
    }
}

pub fn default_certs(strategy_id: &str, next_cert_id: &mut u32) -> Vec<Cert> {
    match strategy_id {
        "prompt-lookup" => {
            let id = *next_cert_id;
            *next_cert_id += 1;
            vec![Cert {
                cert_id: id,
                exactness: exactness::SEED_PATH_INVARIANT,
                sampling_modes: cert_mode::GREEDY,
                grammar_allowed: false,
                logit_bias_allowed: false,
                param_domain: cert_domain::GREEDY_ONLY,
                verification_shape_class: 1,
                admissible_host_identities: Vec::new(),
            }]
        }
        _ => Vec::new(),
    }
}
