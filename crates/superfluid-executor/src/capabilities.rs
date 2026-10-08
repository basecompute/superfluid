//! The executor's capability descriptor.

use superfluid_abi::{encoding, MAX_TOP_LOGPROBS};
use serde_json::{json, Map, Value};

use crate::primitives::{RuntimeDescriptor, SamplingDefaults};

fn cap(reason: Option<String>) -> Value {
    reason.map(Value::String).unwrap_or(Value::Bool(true))
}

fn no(reason: impl Into<String>) -> Value {
    Value::String(reason.into())
}

fn sampling_defaults(sd: &SamplingDefaults) -> Option<Value> {
    let mut m = Map::new();
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            m.insert(k.to_string(), v);
        }
    };
    put("temperature", sd.temperature.map(|x| json!(x)));
    put("top_p", sd.top_p.map(|x| json!(x)));
    put("top_k", sd.top_k.map(|x| json!(x)));
    put("min_p", sd.min_p.map(|x| json!(x)));
    put("repetition_penalty", sd.repetition_penalty.map(|x| json!(x)));
    if let Some(d) = sd.do_sample {
        m.insert("do_sample".to_string(), Value::Bool(d));
    }
    (!m.is_empty()).then_some(Value::Object(m))
}

/// `verify_rows`: the runtime answers `step_rows`, so it can verify a
/// prompt-lookup draft (models that cannot roll back are refused once loaded).
pub fn static_descriptor(id: &str, export_encodings: &[u8], verify_rows: bool) -> serde_json::Value {
    let lossy = export_encodings.iter().any(|e| *e != encoding::LOSSLESS);
    let media = || no(format!("the {id} runtime has no media path"));
    let lora = || no(format!("the {id} runtime loads no LoRA"));
    json!({
        "descriptor_version": 1,
        "runtime": {"id": id},
        "load": {
            "paged_kv": false,
            "prefix_cache": true,
            "kv_bits": no(format!("the {id} runtime keeps its KV type per context and exposes no such knob")),
        },
        "workload": {
            "causal_generation": true,
            "embedding": no(format!("the {id} runtime serves generation; it has no embedding path")),
            "speech_to_text": no("not a speech model"),
        },
        "modalities": {
            "image_encode": media(),
            "gemma_audio_encode": media(),
            "whisper_transcribe": no("not a Whisper model"),
            "whisper_translate": no("not a Whisper model"),
        },
        "serving": {
            "sequences": true,
            "host_logits": true,
            "multi_row_verification": if verify_rows {
                Value::Bool(true)
            } else {
                no(format!("the {id} runtime registers no speculation strategy"))
            },
            "speculation_strategies": if verify_rows { json!([crate::speculate::STRATEGY_ID]) } else { json!([]) },
            "prefix_seed": true,
            "park_lossless": export_encodings.contains(&encoding::LOSSLESS),
            "park_lossy": cap((!lossy).then(|| format!("the {id} runtime exports no lossy encoding"))),
            "round_granular": true,
        },
        "state": {"legacy_save_load": no("the native engine's state-file API")},
        "adaptation": {"lora_load": lora(), "lora_active_on_inference": lora()},
        "ops": {
            "snapshot_restore": true,
            "seq_fork": true,
            "space_promote": no("no demoted tier: parks are lossless exports"),
            "cache_evict_entries": true,
        },
        "sampling": {"logit_bias": true, "penalties": true, "top_logprobs_max": MAX_TOP_LOGPROBS},
        "determinism": {
            "seed_reproducible": true,
            "batch_invariant": no("a batched round or another prefill chunking moves logits by rounding"),
        },
    })
}

pub fn model_facts(architecture: &str, sd: &SamplingDefaults) -> Value {
    let mut v = json!({"architecture": architecture});
    if let Some(s) = sampling_defaults(sd) {
        v["sampling_defaults"] = s;
    }
    v
}

pub(crate) fn descriptor(d: &RuntimeDescriptor, grammar: bool) -> String {
    let id = d.runtime_id.as_str();
    let mut v = static_descriptor(id, &d.export_encodings, d.verify_rows);
    if d.verify_rows && !d.truncate_partial {
        v["serving"]["multi_row_verification"] = json!(format!(
            "prompt-lookup cuts a rejected draft off, and this model's {} state cuts only at its head",
            if d.recurrent { "recurrent" } else { "sliding-window" }
        ));
        v["serving"]["speculation_strategies"] = json!([]);
    }
    v["architecture"] = json!(d.architecture);
    v["backend"] = json!(d.backend);
    v["runtime"]["version"] = json!(d.runtime_version);
    v["load"]["max_batch_size"] = json!(d.max_batch);
    v["serving"]["tick_yield"] =
        cap((d.prefill_step_tokens == 0).then(|| format!("the {id} runtime prefills a tick as one step")));
    v["state"]["rollback"] = cap((!d.truncate_partial).then(|| "recurrent state cuts only at its head".to_string()));
    v["state"]["shared_prefix_seed"] = cap((!d.truncate_partial).then(|| {
        "a cached state seeds only a prompt that holds all of it: recurrent state cuts only at its head".to_string()
    }));
    v["state"]["recurrent_snapshot"] = cap((!d.recurrent).then(|| "no recurrent state in this architecture".to_string()));
    v["sampling"]["grammar"] =
        cap((!grammar).then(|| format!("the {id} runtime lists no vocabulary to compile grammars against")));
    v["sampling"]["engine_argmax"] = json!(d.engine_sampling);
    v["sampling"]["engine_draws"] = json!(d.engine_draws);
    v["limits"] = json!({
        "max_seq_len": d.max_seq_len,
        "max_seqs": d.max_seqs,
        "page_size_tokens": d.page_size_tokens,
        "cells_total": d.cells_total,
        "cache_resident_cells": d.cache_resident_cells,
        "cache_exported_cells": d.cache_exported_cells,
    });
    if let Some(sd) = sampling_defaults(&d.sampling_defaults) {
        v["sampling_defaults"] = sd;
    }
    v.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc() -> RuntimeDescriptor {
        crate::fake::FakePrimitives::default().describe()
    }

    use crate::primitives::RuntimePrimitives;

    #[test]
    fn a_runtime_reports_what_it_lacks_in_its_own_words() {
        let v: Value = serde_json::from_str(&descriptor(&desc(), false)).unwrap();
        assert_eq!(v["descriptor_version"], 1);
        assert_eq!(v["runtime"], json!({"id": "fake", "version": "1"}));
        assert_eq!(v["serving"]["park_lossy"], "the fake runtime exports no lossy encoding");
        assert_eq!(v["serving"]["park_lossless"], true);
        assert_eq!(v["modalities"]["image_encode"], "the fake runtime has no media path");
        assert_eq!(v["sampling"]["grammar"], "the fake runtime lists no vocabulary to compile grammars against");
        assert_eq!(v["state"]["rollback"], true, "the fake cuts anywhere by default");
        assert_eq!(v["state"]["shared_prefix_seed"], true);
        assert!(v.get("sampling_defaults").is_none(), "a checkpoint that claims none carries none");
    }

    #[test]
    fn a_models_facts_are_the_records_artifact_fields() {
        let sd = SamplingDefaults { top_p: Some(0.95), ..Default::default() };
        assert_eq!(model_facts("qwen3", &sd), json!({"architecture": "qwen3", "sampling_defaults": {"top_p": 0.95}}));
        assert_eq!(model_facts("llama", &SamplingDefaults::default()), json!({"architecture": "llama"}));
    }

    #[test]
    fn the_checkpoints_sampling_defaults_ride_along_and_only_what_it_claims() {
        let mut d = desc();
        d.sampling_defaults = SamplingDefaults { temperature: Some(0.6), top_k: Some(20), ..Default::default() };
        d.truncate_partial = false;
        let v: Value = serde_json::from_str(&descriptor(&d, true)).unwrap();
        assert_eq!(v["sampling_defaults"], json!({"temperature": 0.6, "top_k": 20}));
        assert_eq!(v["sampling"]["grammar"], true);
        assert_eq!(v["state"]["rollback"], "recurrent state cuts only at its head");
        assert!(v["state"]["shared_prefix_seed"].is_string(), "a burst on one prefix needs a kept copy");
    }

    #[test]
    fn the_static_part_is_what_every_load_reports() {
        let s = static_descriptor("fake", &[encoding::LOSSLESS], false);
        let full: Value = serde_json::from_str(&descriptor(&desc(), false)).unwrap();
        for (section, keys) in s.as_object().unwrap() {
            let Some(keys) = keys.as_object() else { continue };
            for (k, v) in keys {
                if section == "runtime" {
                    continue;
                }
                assert_eq!(&full[section][k], v, "{section}.{k}");
            }
        }
    }
}
