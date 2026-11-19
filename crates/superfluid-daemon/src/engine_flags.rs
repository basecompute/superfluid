//! The flags for the engine a server runs that `superfluid serve` and a fleet node (`superfluid
//! node join`, `superfluid-noded`) both take: read and checked here, so a node takes each as
//! `serve` does.

use crate::otlp::OtlpConfig;
use crate::SpeculationOptions;

/// What these flags were given, each at `serve`'s default when not.
#[derive(Debug, Clone)]
pub struct EngineFlags {
    /// `--kv-bits`, as given; [`EngineFlags::kv_bits_for`] is what the engine loads with.
    pub kv_bits: Option<i32>,
    pub park_budget_gb: u64,
    pub prefill_budget: u32,
    pub tick_decode_budget: u32,
    pub tick_target_ms: u32,
    pub speculation: SpeculationOptions,
    pub spec_bitexact: Option<bool>,
    pub otlp_endpoint: Option<String>,
    /// The rest of the trace exporter's settings (`--otlp-header`, `--otlp-filter`, ...).
    pub otlp: OtlpConfig,
}

impl Default for EngineFlags {
    fn default() -> EngineFlags {
        EngineFlags {
            kv_bits: None,
            park_budget_gb: 20,
            prefill_budget: crate::DaemonOptions::default().prefill_budget,
            tick_decode_budget: 0,
            tick_target_ms: 0,
            speculation: SpeculationOptions::default(),
            spec_bitexact: None,
            otlp_endpoint: None,
            otlp: OtlpConfig::default(),
        }
    }
}

/// Why a flag was refused: one not written as it takes (a usage error, which `serve` follows with
/// where its options are listed), or a value out of its range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlagError {
    Usage(String),
    Invalid(String),
}

impl std::fmt::Display for FlagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlagError::Usage(m) | FlagError::Invalid(m) => f.write_str(m),
        }
    }
}

fn num<T: std::str::FromStr>(flag: &str, raw: String) -> Result<T, FlagError> {
    raw.parse().map_err(|_| FlagError::Usage(format!("{flag} expects a number, got {raw:?}")))
}

fn invalid<T>(why: &str) -> Result<T, FlagError> {
    Err(FlagError::Invalid(why.to_string()))
}

impl EngineFlags {
    /// Reads `flag` when it is one of these, taking its value from `val`: `Ok(true)` when it was,
    /// `Ok(false)` when it is none of them.
    pub fn take(&mut self, flag: &str, val: &mut dyn FnMut() -> String) -> Result<bool, FlagError> {
        let spec = &mut self.speculation;
        match flag {
            "--kv-bits" => self.kv_bits = Some(num(flag, val())?),
            "--park-budget-gb" => self.park_budget_gb = num(flag, val())?,
            "--prefill-budget" => self.prefill_budget = num(flag, val())?,
            "--tick-decode-budget" => self.tick_decode_budget = num(flag, val())?,
            "--tick-target-ms" => self.tick_target_ms = num(flag, val())?,
            "--spec-draft-tokens" => {
                let n: u32 = num(flag, val())?;
                if !(1..=15).contains(&n) {
                    return invalid("--spec-draft-tokens must be in 1..=15");
                }
                spec.max_draft = Some(n);
            }
            "--spec-adaptive" => spec.adaptive = Some(true),
            "--no-spec-adaptive" => spec.adaptive = Some(false),
            "--spec-min-yield" => {
                let v: f64 = num(flag, val())?;
                if !v.is_finite() || v < 0.0 {
                    return invalid("--spec-min-yield must be finite and non-negative");
                }
                spec.min_yield = Some(v);
            }
            "--spec-yield-rounds" => {
                let n: u32 = num(flag, val())?;
                if n == 0 {
                    return invalid("--spec-yield-rounds must be at least 1");
                }
                spec.yield_rounds = Some(n);
            }
            "--spec-throughput-gate" => spec.throughput_gate = Some(true),
            "--no-spec-throughput-gate" => spec.throughput_gate = Some(false),
            "--spec-gate-probe-tokens" => spec.gate_probe_tokens = Some(num(flag, val())?),
            "--spec-min-speedup" => {
                let v: f64 = num(flag, val())?;
                if !v.is_finite() || v < 1.0 {
                    return invalid("--spec-min-speedup must be finite and at least 1.0");
                }
                spec.min_speedup = Some(v);
            }
            "--spec-gate-reprobe" => spec.gate_reprobe = Some(num(flag, val())?),
            "--spec-gate-reprobe-max" => spec.gate_reprobe_max = Some(num(flag, val())?),
            "--dspark-confidence" => {
                let v: f32 = num(flag, val())?;
                if !v.is_finite() || v <= 0.0 {
                    return invalid("--dspark-confidence must be finite and positive");
                }
                spec.dspark_confidence = Some(v);
            }
            "--spec-bitexact" => self.spec_bitexact = Some(true),
            "--no-spec-bitexact" => self.spec_bitexact = Some(false),
            "--otlp-endpoint" => self.otlp_endpoint = Some(val()),
            "--otlp-header" => {
                let kv = val();
                let (k, v) = kv.split_once('=').ok_or_else(|| FlagError::Usage(format!("--otlp-header expects k=v, got {kv:?}")))?;
                self.otlp.headers.push((k.trim().to_string(), v.trim().to_string()));
            }
            "--otlp-service-name" => self.otlp.service_name = val(),
            "--otlp-filter" => self.otlp.filter = val(),
            "--otlp-queue" => self.otlp.queue = num(flag, val())?,
            "--otlp-batch-ms" => self.otlp.interval = std::time::Duration::from_millis(num(flag, val())?),
            "--otlp-timeout-ms" => self.otlp.timeout = std::time::Duration::from_millis(num(flag, val())?),
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The KV cache precision the engine loads with: `--kv-bits` as given (0, auto, when not),
    /// and F16 under `--park-lossy`, which another precision contradicts.
    pub fn kv_bits_for(&self, park_lossy: bool) -> Result<i32, String> {
        if let Some(b) = self.kv_bits.filter(|b| ![0, 4, 8, 16, 84].contains(b)) {
            return Err(format!("--kv-bits {b}: expected 0 (auto), 4 (q4_0), 8 (q8_0), 16 (f16) or 84 (K q8_0 / V q4_0)"));
        }
        match (park_lossy, self.kv_bits) {
            (true, None) | (true, Some(16)) => Ok(16),
            (true, Some(other)) => Err(format!("--park-lossy requires F16 KV pools; --kv-bits {other} conflicts")),
            (false, v) => Ok(v.unwrap_or(0)),
        }
    }

    /// Tells the engine about to start whether speculation keeps its output bit-exact
    /// (`--spec-bitexact`), as it reads it: from this process's environment, which a worker takes.
    pub fn export_spec_bitexact(&self) {
        if let Some(on) = self.spec_bitexact {
            std::env::set_var("BASERT_SPEC_BITEXACT", if on { "1" } else { "0" });
        }
    }

    /// The trace exporter `--otlp-endpoint` asks for, checked, and said on stderr as the egress it is.
    pub fn otlp_traces(&self) -> Result<Option<OtlpConfig>, String> {
        let Some(endpoint) = &self.otlp_endpoint else { return Ok(None) };
        let mut cfg = self.otlp.clone();
        cfg.endpoint = endpoint.clone();
        cfg.validate().map_err(|e| format!("--otlp-endpoint: {e}"))?;
        eprintln!(
            "superfluid: pushing operational traces to {} ({}an EGRESS: ids, lengths, timings and \
             codes only, never session content; spans: {})",
            cfg.traces_url(),
            if cfg.is_loopback() { "" } else { "NON-LOOPBACK, " },
            cfg.filter
        );
        Ok(Some(cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn given(args: &[&str]) -> Result<EngineFlags, FlagError> {
        let mut flags = EngineFlags::default();
        let mut it = args.iter().map(|s| s.to_string());
        while let Some(a) = it.next() {
            let mut val = || it.next().expect("a value");
            assert!(flags.take(&a, &mut val)?, "{a} is one of these");
        }
        Ok(flags)
    }

    #[test]
    fn each_flag_lands_where_serve_puts_it() {
        let f = given(&[
            "--kv-bits", "8", "--park-budget-gb", "4", "--prefill-budget", "256", "--tick-decode-budget", "3",
            "--tick-target-ms", "40", "--spec-draft-tokens", "6", "--no-spec-adaptive", "--spec-min-yield", "0.5",
            "--spec-throughput-gate", "--spec-min-speedup", "1.2", "--spec-bitexact", "--otlp-endpoint",
            "http://127.0.0.1:4318", "--otlp-header", "k = v", "--otlp-batch-ms", "250",
        ])
        .unwrap();
        assert_eq!((f.kv_bits, f.park_budget_gb, f.prefill_budget, f.tick_decode_budget, f.tick_target_ms), (Some(8), 4, 256, 3, 40));
        let s = &f.speculation;
        assert_eq!((s.max_draft, s.adaptive, s.min_yield, s.throughput_gate, s.min_speedup), (Some(6), Some(false), Some(0.5), Some(true), Some(1.2)));
        assert_eq!(f.spec_bitexact, Some(true));
        assert_eq!(f.otlp.headers, vec![("k".to_string(), "v".to_string())]);
        assert_eq!(f.otlp.interval, std::time::Duration::from_millis(250));
        assert_eq!(f.otlp_traces().unwrap().map(|c| c.endpoint).as_deref(), Some("http://127.0.0.1:4318"));
        assert!(!EngineFlags::default().take("--max-batch", &mut || unreachable!()).unwrap(), "not one of these");
    }

    #[test]
    fn a_value_out_of_range_is_refused_as_serve_refuses_it() {
        assert_eq!(given(&["--spec-draft-tokens", "16"]).unwrap_err(), FlagError::Invalid("--spec-draft-tokens must be in 1..=15".into()));
        assert_eq!(given(&["--spec-min-speedup", "0.9"]).unwrap_err(), FlagError::Invalid("--spec-min-speedup must be finite and at least 1.0".into()));
        assert_eq!(given(&["--prefill-budget", "lots"]).unwrap_err(), FlagError::Usage("--prefill-budget expects a number, got \"lots\"".into()));
        assert!(matches!(given(&["--otlp-header", "novalue"]), Err(FlagError::Usage(_))));
    }

    #[test]
    fn the_kv_precision_is_checked_and_f16_under_lossy_parking() {
        let bits = |b: Option<i32>, lossy: bool| EngineFlags { kv_bits: b, ..EngineFlags::default() }.kv_bits_for(lossy);
        assert_eq!(bits(None, false), Ok(0));
        assert_eq!(bits(Some(8), false), Ok(8));
        assert_eq!(bits(None, true), Ok(16));
        assert_eq!(bits(Some(16), true), Ok(16));
        assert!(bits(Some(8), true).unwrap_err().contains("--park-lossy requires F16"));
        assert!(bits(Some(5), false).unwrap_err().starts_with("--kv-bits 5: expected"));
    }
}
