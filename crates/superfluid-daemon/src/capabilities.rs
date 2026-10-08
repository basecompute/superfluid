//! What a loaded model can do, as its runtime reported it in the Link W hello.

use serde_json::Value;

use crate::DaemonError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap<'a> {
    Yes,
    No(&'a str),
    Unknown,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Capabilities {
    record: Option<Value>,
}

impl Capabilities {
    pub fn parse(json: &str) -> Capabilities {
        Capabilities { record: serde_json::from_str::<Value>(json).ok().filter(Value::is_object) }
    }

    pub fn record(&self) -> Option<&Value> {
        self.record.as_ref()
    }

    pub fn get(&self, section: &str, key: &str) -> Cap<'_> {
        match self.record.as_ref().and_then(|r| r.get(section)).and_then(|s| s.get(key)) {
            Some(Value::Bool(true)) => Cap::Yes,
            Some(Value::String(why)) => Cap::No(why),
            Some(Value::Bool(false)) => Cap::No("not supported"),
            _ => Cap::Unknown,
        }
    }

    pub fn require(&self, section: &str, key: &str, code: &'static str, what: &str) -> Result<(), DaemonError> {
        match self.get(section, key) {
            Cap::No(why) => Err(DaemonError::Unsupported(crate::Refusal::new(code, None, format!("{what}: {why}")))),
            Cap::Yes | Cap::Unknown => Ok(()),
        }
    }

    pub fn limit(&self, key: &str) -> Option<u64> {
        self.record.as_ref()?.get("limits")?.get(key)?.as_u64().filter(|n| *n > 0)
    }

    pub fn changes(&self, other: &Capabilities) -> Vec<String> {
        fn walk(path: &str, a: &Value, b: &Value, out: &mut Vec<String>) {
            match (a, b) {
                (Value::Object(x), Value::Object(y)) => {
                    let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
                    keys.sort();
                    keys.dedup();
                    for k in keys {
                        if path.is_empty() && k == "limits" {
                            continue;
                        }
                        let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                        walk(&p, x.get(k).unwrap_or(&Value::Null), y.get(k).unwrap_or(&Value::Null), out);
                    }
                }
                _ if a != b => out.push(path.to_string()),
                _ => {}
            }
        }
        let mut out = Vec::new();
        match (&self.record, &other.record) {
            (Some(a), Some(b)) => walk("", a, b, &mut out),
            (None, None) => {}
            _ => out.push("the whole record".to_string()),
        }
        out
    }

    pub fn with_runtime(&self, picked: Option<&str>) -> Option<Value> {
        let mut r = self.record.clone()?;
        if let (Some(id), Some(obj)) = (picked, r.as_object_mut()) {
            obj.entry("runtime").or_insert_with(|| serde_json::json!({ "id": id }));
        }
        Some(r)
    }

    pub fn check_startup_flags(&self, park_lossy: bool, kv_bits: Option<i32>, speculate: Option<&str>) -> Result<(), String> {
        let generates = !matches!(self.get("workload", "causal_generation"), Cap::No(_));
        if park_lossy && generates {
            if let Cap::No(why) = self.get("serving", "park_lossy") {
                return Err(format!("--park-lossy: {why}"));
            }
        }
        if let Some(bits) = kv_bits.filter(|b| *b != 0) {
            if let Cap::No(why) = self.get("load", "kv_bits") {
                return Err(format!("--kv-bits {bits}: {why}"));
            }
        }
        if let Some(d) = speculate.filter(|d| !matches!(d.trim(), "off" | "auto")).filter(|_| generates) {
            if let Cap::No(why) = self.get("serving", "multi_row_verification") {
                return Err(format!("--speculate {d}: {why}"));
            }
        }
        Ok(())
    }

    pub fn declines_auto_speculation(&self) -> Option<&str> {
        match self.get("serving", "multi_row_verification") {
            Cap::No(why) => Some(why),
            Cap::Yes | Cap::Unknown => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXECUTOR: &str = r#"{"descriptor_version":1,"runtime":{"id":"llamacpp","version":"v"},
        "load":{"kv_bits":"no such knob"},
        "modalities":{"image_encode":"no media path","gemma_audio_encode":"no media path"},
        "serving":{"park_lossy":"no lossy encoding","multi_row_verification":"no strategy","park_lossless":true}}"#;

    #[test]
    fn a_leaf_is_yes_a_reason_or_unknown() {
        let c = Capabilities::parse(EXECUTOR);
        assert_eq!(c.get("serving", "park_lossless"), Cap::Yes);
        assert_eq!(c.get("modalities", "image_encode"), Cap::No("no media path"));
        assert_eq!(c.get("modalities", "whisper_transcribe"), Cap::Unknown);
        assert_eq!(Capabilities::parse("").get("serving", "park_lossy"), Cap::Unknown);
        assert_eq!(Capabilities::parse("[1]").record(), None, "only an object is a record");
    }

    #[test]
    fn require_refuses_in_the_runtimes_words_and_passes_the_unknown() {
        let c = Capabilities::parse(EXECUTOR);
        let e = c.require("modalities", "image_encode", "unsupported_input", "this model does not accept images").unwrap_err();
        assert!(
            matches!(&e, DaemonError::Unsupported(r) if r.message == "this model does not accept images: no media path" && r.code == "unsupported_input"),
            "{e:?}"
        );
        assert!(c.require("modalities", "whisper_transcribe", "unsupported_model", "x").is_ok(), "an undescribed key is not a refusal");
        assert!(Capabilities::default().require("modalities", "image_encode", "unsupported_input", "x").is_ok());
    }

    #[test]
    fn startup_flags_are_checked_against_the_record() {
        let c = Capabilities::parse(EXECUTOR);
        assert_eq!(c.check_startup_flags(true, None, None).unwrap_err(), "--park-lossy: no lossy encoding");
        assert_eq!(c.check_startup_flags(false, Some(8), None).unwrap_err(), "--kv-bits 8: no such knob");
        assert_eq!(c.check_startup_flags(false, None, Some("prompt-lookup")).unwrap_err(), "--speculate prompt-lookup: no strategy");
        assert!(c.check_startup_flags(false, Some(0), Some("off")).is_ok(), "0 and off are no request");
        let native = Capabilities::parse(r#"{"descriptor_version":1,"serving":{"park_lossy":true,"multi_row_verification":true}}"#);
        assert!(native.check_startup_flags(true, Some(8), Some("prompt-lookup")).is_ok());
    }

    #[test]
    fn auto_speculation_is_a_preference_and_a_model_that_generates_nothing_takes_no_generation_flags() {
        let hybrid = Capabilities::parse(
            r#"{"descriptor_version":1,"workload":{"causal_generation":true},
                "serving":{"park_lossy":true,"multi_row_verification":"Mamba-2 hybrid: no speculative verification"}}"#,
        );
        assert_eq!(hybrid.check_startup_flags(false, None, Some("auto")), Ok(()));
        assert_eq!(hybrid.check_startup_flags(false, None, Some(" auto ")), Ok(()), "as the resolver reads it, trimmed");
        assert_eq!(
            hybrid.check_startup_flags(false, None, Some("prompt-lookup")).unwrap_err(),
            "--speculate prompt-lookup: Mamba-2 hybrid: no speculative verification"
        );
        assert_eq!(hybrid.declines_auto_speculation(), Some("Mamba-2 hybrid: no speculative verification"));
        let whisper = Capabilities::parse(
            r#"{"descriptor_version":1,"workload":{"causal_generation":"speech encoder-decoder, no autoregressive text API"},
                "serving":{"park_lossy":"no autoregressive decode","multi_row_verification":"no autoregressive decode"}}"#,
        );
        assert_eq!(whisper.check_startup_flags(true, None, Some("auto")), Ok(()));
        assert_eq!(whisper.check_startup_flags(true, None, Some("dflash:/d.base")), Ok(()));
        let c = Capabilities::parse(EXECUTOR);
        assert_eq!(c.check_startup_flags(true, None, Some("auto")).unwrap_err(), "--park-lossy: no lossy encoding");
        assert_eq!(c.check_startup_flags(false, None, Some("auto")), Ok(()));
        assert_eq!(c.declines_auto_speculation(), Some("no strategy"));
        assert_eq!(Capabilities::default().declines_auto_speculation(), None, "an engine that describes nothing speculates as before");
    }

    #[test]
    fn a_respawn_that_resizes_is_the_same_model_one_that_upgrades_is_not() {
        let a = Capabilities::parse(r#"{"runtime":{"id":"llamacpp","version":"1"},"serving":{"park_lossy":"no"},"limits":{"max_seq_len":8192,"cells_total":65536}}"#);
        let resized = Capabilities::parse(r#"{"runtime":{"id":"llamacpp","version":"1"},"serving":{"park_lossy":"no"},"limits":{"max_seq_len":4096,"cells_total":32768}}"#);
        assert!(a.changes(&resized).is_empty(), "limits follow the memory free at load");
        assert_eq!((a.limit("max_seq_len"), resized.limit("cells_total")), (Some(8192), Some(32768)));
        assert_eq!(a.limit("max_seqs"), None);
        let upgraded = Capabilities::parse(r#"{"runtime":{"id":"llamacpp","version":"2"},"serving":{"park_lossy":true},"limits":{"max_seq_len":8192}}"#);
        assert_eq!(a.changes(&upgraded), ["runtime.version", "serving.park_lossy"]);
        assert_eq!(Capabilities::default().changes(&Capabilities::default()), Vec::<String>::new());
        assert_eq!(Capabilities::default().changes(&a), ["the whole record"], "a record where there was none");
    }

    #[test]
    fn the_runtime_is_named_once() {
        let exec = Capabilities::parse(EXECUTOR).with_runtime(Some("basert")).unwrap();
        assert_eq!(exec["runtime"]["id"], "llamacpp", "the record's own runtime wins");
        let native = Capabilities::parse(r#"{"descriptor_version":1}"#).with_runtime(Some("basert")).unwrap();
        assert_eq!(native["runtime"], serde_json::json!({"id": "basert"}));
        assert_eq!(Capabilities::default().with_runtime(Some("basert")), None);
    }
}
