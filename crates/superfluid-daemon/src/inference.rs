//! Shared inference requests, session construction, sampling and execution.

pub(crate) use crate::generation_output::*;
use crate::wal::{channel, role, EventBody};
use crate::ModelSamplingDefaults;
use crate::{Daemon, DaemonError, GenParams};
use superfluid_abi::finish;
use serde::Deserialize;
use std::sync::Arc;

pub(crate) const DEFAULT_MAX_TOKENS: u32 = 2048;

#[derive(Clone, Copy)]
pub struct TokenLimits {
    pub default_max_tokens: Option<u32>,
    pub max_context: u32,
}

impl TokenLimits {
    pub fn resolve(&self, explicit: Option<u32>, prompt_tokens: impl FnOnce() -> u32) -> u32 {
        explicit
            .or(self.default_max_tokens)
            .unwrap_or_else(|| self.max_context.saturating_sub(prompt_tokens()).max(1))
    }
}
pub(crate) const DEFAULT_MAX_CONTEXT: u32 = 8192;

#[derive(Clone, Copy)]
pub(crate) struct RequestPolicy {
    pub(crate) limits: TokenLimits,
    pub(crate) sampling: SamplingOverrides,
    pub(crate) qos: RequestQos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestQos {
    pub class: u8,
    pub batch_invariant: bool,
}

impl RequestQos {
    pub const AGENT: RequestQos = RequestQos {
        class: crate::qos::FOREGROUND_AGENT,
        batch_invariant: false,
    };

    pub(crate) fn apply(self, daemon: &Daemon, session: u64) -> Result<(), DaemonError> {
        // A fork inherits its parent's class, so a continued session is
        // compared with what it holds, not with the default.
        let held = {
            let store = daemon.store();
            let store = store.lock().expect("store");
            let s = store.session(session)?;
            RequestQos { class: s.qos_class, batch_invariant: s.batch_invariant }
        };
        if self != held {
            daemon.set_qos(session, self.class, self.batch_invariant)?;
        }
        Ok(())
    }
}

impl Default for RequestQos {
    fn default() -> RequestQos {
        RequestQos::AGENT
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SamplingOverrides {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub spec_max_temperature: Option<f32>,
    /// Optional policy for an embedding whose omitted request parameters
    /// predate model-published defaults. None preserves superfluid's policy.
    pub fallback: Option<SamplingFallback>,
}

/// An embedding's defaults, below request and operator values: temperature and repetition
/// replace the model's; truncation applies only when the model publishes none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingFallback {
    pub temperature: f32,
    pub repeat_penalty: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    /// Interpret a lone frequency_penalty as multiplicative (1 + value).
    /// An explicit presence_penalty or repeat_penalty retains additive
    /// frequency semantics; presence_penalty alone disables fallback repeat.
    pub frequency_penalty_as_repeat: bool,
}

impl SamplingOverrides {
    pub fn for_daemon(&self, daemon: &Daemon) -> Self {
        let mut o = *self;
        if o.spec_max_temperature.is_some() && daemon.speculation().is_none() {
            o.spec_max_temperature = None;
        }
        o
    }

    pub fn truncates(&self) -> bool {
        self.top_p.is_some() || self.top_k.is_some() || self.min_p.is_some()
    }
}

#[derive(Deserialize, Default)]
pub(crate) struct ChatRequest {
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) messages: Vec<ChatMessage>,
    #[serde(default)]
    pub(crate) tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub(crate) tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) stream: bool,
    #[serde(default)]
    pub(crate) stream_options: Option<StreamOptions>,
    pub(crate) max_completion_tokens: Option<u32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) seed: Option<u64>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    #[serde(alias = "repetition_penalty")]
    pub(crate) repeat_penalty: Option<f32>,
    pub(crate) min_p: Option<f32>,
    pub(crate) n: Option<u32>,
    #[serde(default)]
    pub(crate) ignore_eos: bool,
    #[serde(default)]
    pub(crate) stop: StopField,
    pub(crate) response_format: Option<serde_json::Value>,
    pub(crate) logit_bias: Option<std::collections::HashMap<String, f32>>,
    pub(crate) logprobs: Option<bool>,
    pub(crate) top_logprobs: Option<u32>,
    pub(crate) chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    pub(crate) enable_thinking: Option<bool>,
    pub(crate) reasoning_effort: Option<String>,
    /// Set by `/v1/responses` only: the request continues a stored session
    /// instead of starting one, and `messages` are the turn appended to it.
    #[serde(skip)]
    pub(crate) continues: Option<Continuation>,
    /// The route the access log names, when another dialect runs the chat
    /// path; chat completions when unset.
    #[serde(skip)]
    pub(crate) route: Option<&'static str>,
}

/// A stored session forked at `at_event`, the event after its last.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Continuation {
    pub(crate) session: u64,
    pub(crate) at_event: u64,
}

impl ChatRequest {
    pub(crate) fn route(&self) -> &'static str {
        self.route.unwrap_or("POST /v1/chat/completions")
    }

    pub(crate) fn thinking(&self) -> Option<bool> {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|m| m.get("enable_thinking"))
            .and_then(|v| v.as_bool())
            .or(self.enable_thinking)
    }

    pub(crate) fn template_kwargs(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut m = self.chat_template_kwargs.clone().unwrap_or_default();
        if let Some(t) = self.enable_thinking {
            m.entry("enable_thinking".to_string())
                .or_insert(serde_json::Value::Bool(t));
        }
        if let Some(e) = &self.reasoning_effort {
            m.entry("reasoning_effort".to_string())
                .or_insert(serde_json::Value::String(e.clone()));
        }
        m
    }
}

pub(crate) fn schema_of(rf: &Option<serde_json::Value>) -> Option<String> {
    let rf = rf.as_ref()?;
    match rf.get("type").and_then(|t| t.as_str()) {
        Some("json_object") => Some(r#"{"type":"object"}"#.to_string()),
        Some("json_schema") => rf
            .get("json_schema")
            .and_then(|js| js.get("schema").or(Some(js)))
            .map(|s| s.to_string()),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function(String),
}

impl ToolChoice {
    pub(crate) fn parse(v: &Option<serde_json::Value>) -> ToolChoice {
        let Some(v) = v else { return ToolChoice::Auto };
        if let Some(s) = v.as_str() {
            return match s {
                "none" => ToolChoice::None,
                "required" => ToolChoice::Required,
                _ => ToolChoice::Auto,
            };
        }
        let name = v
            .get("function")
            .and_then(|f| f.get("name"))
            .or_else(|| v.get("name"))
            .and_then(|n| n.as_str());
        match name {
            Some(n) if !n.is_empty() => ToolChoice::Function(n.to_string()),
            _ => ToolChoice::Auto,
        }
    }

    pub(crate) fn forces_call(&self) -> bool {
        matches!(self, ToolChoice::Required | ToolChoice::Function(_))
    }

    fn declares_tools(&self) -> bool {
        !matches!(self, ToolChoice::None)
    }

    fn constrain_to<'a>(
        &self,
        tools: &'a [serde_json::Value],
    ) -> Option<Vec<&'a serde_json::Value>> {
        match self {
            ToolChoice::Auto | ToolChoice::None => None,
            ToolChoice::Required => Some(tools.iter().collect()),
            ToolChoice::Function(name) => {
                let picked: Vec<&serde_json::Value> = tools
                    .iter()
                    .filter(|t| {
                        let f = t.get("function").unwrap_or(*t);
                        f.get("name").and_then(|n| n.as_str()) == Some(name.as_str())
                    })
                    .collect();
                (!picked.is_empty()).then_some(picked)
            }
        }
    }
}

pub(crate) struct RequestHandles<'a> {
    daemon: &'a Daemon,
    grammar: Option<u32>,
    bias: Option<u32>,
}

impl<'a> RequestHandles<'a> {
    pub(crate) fn new(daemon: &'a Daemon, grammar: Option<u32>, bias: Option<u32>) -> Self {
        RequestHandles {
            daemon,
            grammar,
            bias,
        }
    }

    pub(crate) fn apply(&self, extras: &mut crate::scheduler::GenExtras) {
        if let Some(h) = self.grammar {
            extras.grammar_handle = h;
        }
        if let Some(h) = self.bias {
            extras.logit_bias_handle = h;
        }
    }
}

impl Drop for RequestHandles<'_> {
    fn drop(&mut self) {
        if let Some(h) = self.grammar.take() {
            self.daemon.grammar_free(h);
        }
        if let Some(h) = self.bias.take() {
            self.daemon.logit_bias_free(h);
        }
    }
}

pub(crate) fn grammar_handle_for(
    daemon: &Daemon,
    req: &ChatRequest,
    choice: &ToolChoice,
) -> Result<Option<u32>, DaemonError> {
    if let ToolChoice::Function(name) = choice {
        let declared = req.tools.iter().any(|t| {
            let f = t.get("function").unwrap_or(t);
            f.get("name").and_then(|n| n.as_str()) == Some(name.as_str())
        });
        if !declared {
            return Err(DaemonError::Constraint(format!(
                "tool_choice names function {name:?}, which is not among the declared tools"
            )));
        }
    }
    if !req.tools.is_empty() && *choice != ToolChoice::None {
        let jsons: Vec<String> = match choice.constrain_to(&req.tools) {
            Some(tools) => tools.iter().map(|t| t.to_string()).collect(),
            None => req.tools.iter().map(|t| t.to_string()).collect(),
        };
        if let Some(tag) = daemon.tool_structural_tag(&jsons, choice.forces_call()) {
            if let Ok(h) = daemon.grammar_create_structural(&tag) {
                if h != 0 {
                    return Ok(Some(h));
                }
            }
        }
    }
    if let Some(tools) = choice.constrain_to(&req.tools) {
        if tools.is_empty() {
            return Err(DaemonError::Constraint(
                "tool_choice demands a call but the request declares no tools".to_string(),
            ));
        }
        let jsons: Vec<String> = tools.iter().map(|t| t.to_string()).collect();
        if let Some(schema) = daemon.tool_call_grammar(&jsons) {
            if let Ok(h) = daemon.grammar_create(&schema) {
                if h != 0 {
                    return Ok(Some(h));
                }
            }
        }
        return Err(DaemonError::Constraint(
            "tool_choice demands a call, but no tool-call grammar could be compiled for this \
             model's dialect on this engine"
                .to_string(),
        ));
    }
    match schema_of(&req.response_format) {
        None if req
            .response_format
            .as_ref()
            .and_then(|rf| rf.get("type"))
            .and_then(|t| t.as_str())
            == Some("json_schema") =>
        {
            Err(DaemonError::Constraint(
                "response_format type json_schema requires a json_schema.schema block".to_string(),
            ))
        }
        None => Ok(None),
        Some(s) => {
            if let Some(tag) = daemon.response_format_tag(&s) {
                return match daemon.grammar_create_structural(&tag) {
                    Ok(h) if h != 0 => Ok(Some(h)),
                    _ => Err(DaemonError::Constraint(
                        "response_format grammar did not compile in this model's answer frame on this engine"
                            .to_string(),
                    )),
                };
            }
            match daemon.grammar_create(&s) {
                Ok(h) if h != 0 => Ok(Some(h)),
                Ok(_) => Err(DaemonError::Constraint(
                    "response_format grammar did not compile on this engine (schema rejected or \
                     constrained decoding unavailable in this build)"
                        .to_string(),
                )),
                Err(e) => Err(DaemonError::Constraint(format!(
                    "response_format grammar could not be created: {e}"
                ))),
            }
        }
    }
}

#[derive(Deserialize, Default, Clone)]
#[serde(untagged)]
pub(crate) enum StopField {
    #[default]
    None,
    One(String),
    Many(Vec<String>),
}

impl StopField {
    pub(crate) fn list(&self) -> Vec<String> {
        match self {
            StopField::None => Vec::new(),
            StopField::One(s) => vec![s.clone()],
            StopField::Many(v) => v.iter().filter(|s| !s.is_empty()).cloned().collect(),
        }
    }
}

fn stop_prefix_holdback(text: &str, stops: &[String]) -> usize {
    let mut hold = 0usize;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        let max = s.len().saturating_sub(1).min(text.len());
        for k in (1..=max).rev() {
            if !text.is_char_boundary(text.len() - k) || !s.is_char_boundary(k) {
                continue;
            }
            if text.ends_with(&s[..k]) {
                hold = hold.max(k);
                break;
            }
        }
    }
    hold
}

pub(crate) fn truncate_at_stop(text: &str, stops: &[String]) -> (String, bool) {
    let mut cut = None;
    for s in stops {
        if let Some(i) = text.find(s.as_str()) {
            cut = Some(cut.map_or(i, |c: usize| c.min(i)));
        }
    }
    match cut {
        Some(i) => (text[..i].to_string(), true),
        None => (text.to_string(), false),
    }
}

#[cfg(test)]
mod penalty_alias_tests {
    use super::{
        advertised_generation_defaults, draw_seed_if_unseeded, params_of_for, repeat_penalty_for,
        ChatRequest, CompletionRequest, SamplingOverrides, DEFAULT_REPEAT_PENALTY, DEFAULT_TOP_K,
    };
    use crate::GenParams;
    use crate::ModelSamplingDefaults;

    #[test]
    fn unseeded_sampled_requests_draw_distinct_seeds() {
        let m = ModelSamplingDefaults::default();
        let op = SamplingOverrides::default();
        let mut a = params_of_for(Some(0.7), None, None, None, None, &m, &op);
        let mut b = params_of_for(Some(0.7), None, None, None, None, &m, &op);
        assert_eq!((a.seed, b.seed), (0, 0), "params_of_for itself never draws");
        assert!(draw_seed_if_unseeded(&mut a, None));
        assert!(draw_seed_if_unseeded(&mut b, None));
        assert_ne!(a.seed, 0);
        assert_ne!(a.seed, b.seed, "two requests, two streams");
        let mut greedy = params_of_for(Some(0.0), None, None, None, None, &m, &op);
        assert!(!draw_seed_if_unseeded(&mut greedy, None));
        assert_eq!(greedy.seed, 0, "greedy has no use for a seed");
        let mut explicit = params_of_for(Some(0.7), None, None, None, Some(42), &m, &op);
        assert!(!draw_seed_if_unseeded(&mut explicit, Some(42)));
        assert_eq!(explicit.seed, 42, "the client's seed is honoured as is");
    }

    #[test]
    fn repetition_penalty_is_an_alias_for_repeat_penalty() {
        let canonical: ChatRequest =
            serde_json::from_str(r#"{"messages":[],"repeat_penalty":1.3}"#)
                .expect("canonical spelling");
        let alias: ChatRequest =
            serde_json::from_str(r#"{"messages":[],"repetition_penalty":1.3}"#)
                .expect("alias spelling");
        assert_eq!(canonical.repeat_penalty, Some(1.3));
        assert_eq!(
            alias.repeat_penalty,
            Some(1.3),
            "repetition_penalty was dropped"
        );

        let c_canon: CompletionRequest =
            serde_json::from_str(r#"{"prompt":"hi","repeat_penalty":1.3}"#).expect("canonical");
        let c_alias: CompletionRequest =
            serde_json::from_str(r#"{"prompt":"hi","repetition_penalty":1.3}"#).expect("alias");
        assert_eq!(c_canon.repeat_penalty, Some(1.3));
        assert_eq!(
            c_alias.repeat_penalty,
            Some(1.3),
            "repetition_penalty was dropped on /v1/completions"
        );
    }

    fn runtime_only(
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        seed: Option<u64>,
    ) -> GenParams {
        params_of_for(
            temperature,
            top_p,
            top_k,
            None,
            seed,
            &ModelSamplingDefaults::default(),
            &SamplingOverrides::default(),
        )
    }

    #[test]
    fn unspecified_truncation_defaults_to_top_k() {
        let p = runtime_only(None, None, None, None);
        assert_eq!(
            p.top_k, DEFAULT_TOP_K,
            "neither knob given -> default top_k"
        );
        assert_eq!(p.top_p, 1.0, "top_p still reports its nominal default");
        assert_eq!(p.temperature, 1.0);
    }

    #[test]
    fn explicit_sampling_knobs_are_never_overridden() {
        let explicit_top_p = runtime_only(None, Some(1.0), None, None);
        assert_eq!(
            explicit_top_p.top_k, 0,
            "explicit top_p disables the default"
        );
        assert_eq!(runtime_only(None, Some(0.9), None, None).top_k, 0);
        assert_eq!(
            runtime_only(None, None, Some(7), None).top_k,
            7,
            "explicit top_k wins"
        );
        assert_eq!(
            runtime_only(None, None, Some(0), None).top_k,
            0,
            "explicit 0 means disabled"
        );
    }

    #[test]
    fn greedy_gets_no_truncation_default() {
        assert_eq!(runtime_only(Some(0.0), None, None, None).top_k, 0);
    }

    #[test]
    fn advertised_defaults_match_the_effective_ones() {
        let adv = advertised_generation_defaults(
            &ModelSamplingDefaults::default(),
            &SamplingOverrides::default(),
        );
        let effective = runtime_only(None, None, None, None);
        assert_eq!(
            adv.top_k, effective.top_k,
            "advertised top_k must be the effective one"
        );
        assert_eq!(adv.top_p, effective.top_p);
        assert_eq!(adv.temperature, effective.temperature);
        assert_eq!(adv.top_k, DEFAULT_TOP_K);
    }

    #[test]
    fn round_tripping_advertised_defaults_is_stable() {
        let adv = advertised_generation_defaults(
            &ModelSamplingDefaults::default(),
            &SamplingOverrides::default(),
        );
        let echoed = runtime_only(
            Some(adv.temperature),
            Some(adv.top_p),
            Some(adv.top_k),
            None,
        );
        assert_eq!(echoed.top_k, adv.top_k, "echoed top_k survives");
        assert_eq!(echoed.top_p, adv.top_p);
        assert_eq!(echoed.temperature, adv.temperature);
    }

    #[test]
    fn model_defaults_beat_the_runtime_fallback() {
        let qwen38 = ModelSamplingDefaults {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.0),
            do_sample: Some(true),
            repetition_penalty: None,
        };
        let p = params_of_for(
            None,
            None,
            None,
            None,
            None,
            &qwen38,
            &SamplingOverrides::default(),
        );
        assert_eq!(p.top_k, 20, "the model asked for 20, not the fallback 40");
        assert_eq!(p.top_p, 0.95);
        assert_eq!(p.temperature, 1.0);
    }

    #[test]
    fn caller_beats_the_model_defaults() {
        let m = ModelSamplingDefaults {
            top_k: Some(20),
            top_p: Some(0.95),
            ..Default::default()
        };
        assert_eq!(
            params_of_for(
                None,
                None,
                Some(5),
                None,
                None,
                &m,
                &SamplingOverrides::default()
            )
            .top_k,
            5
        );
        assert_eq!(
            params_of_for(
                None,
                Some(0.5),
                None,
                None,
                None,
                &m,
                &SamplingOverrides::default()
            )
            .top_p,
            0.5
        );
        assert_eq!(
            params_of_for(
                Some(0.3),
                None,
                None,
                None,
                None,
                &m,
                &SamplingOverrides::default()
            )
            .temperature,
            0.3
        );
    }

    #[test]
    fn empty_model_defaults_keep_the_fallback() {
        let p = runtime_only(None, None, None, None);
        assert_eq!(p.top_k, DEFAULT_TOP_K);
    }

    #[test]
    fn spec_max_temperature_caps_only_the_bundle_default() {
        let m = ModelSamplingDefaults {
            temperature: Some(1.0),
            ..Default::default()
        };
        let op = SamplingOverrides {
            spec_max_temperature: Some(0.6),
            ..Default::default()
        };
        assert_eq!(
            params_of_for(None, None, None, None, None, &m, &op).temperature,
            0.6
        );
        assert_eq!(
            params_of_for(Some(0.9), None, None, None, None, &m, &op).temperature,
            0.9,
            "caller wins"
        );
        let op_t = SamplingOverrides {
            temperature: Some(0.8),
            ..op
        };
        assert_eq!(
            params_of_for(None, None, None, None, None, &m, &op_t).temperature,
            0.8,
            "--temperature wins"
        );
        let cool = ModelSamplingDefaults {
            temperature: Some(0.3),
            ..Default::default()
        };
        assert_eq!(
            params_of_for(None, None, None, None, None, &cool, &op).temperature,
            0.3
        );
        let greedy = ModelSamplingDefaults {
            do_sample: Some(false),
            ..Default::default()
        };
        assert_eq!(
            params_of_for(None, None, None, None, None, &greedy, &op).temperature,
            0.0
        );
        assert_eq!(
            params_of_for(
                None,
                None,
                None,
                None,
                None,
                &ModelSamplingDefaults::default(),
                &op
            )
            .temperature,
            0.6
        );
        assert_eq!(advertised_generation_defaults(&m, &op).temperature, 0.6);
    }

    #[test]
    fn do_sample_false_means_greedy() {
        let m = ModelSamplingDefaults {
            do_sample: Some(false),
            ..Default::default()
        };
        assert_eq!(
            params_of_for(
                None,
                None,
                None,
                None,
                None,
                &m,
                &SamplingOverrides::default()
            )
            .temperature,
            0.0
        );
        assert_eq!(
            params_of_for(
                Some(0.8),
                None,
                None,
                None,
                None,
                &m,
                &SamplingOverrides::default()
            )
            .temperature,
            0.8
        );
    }

    #[test]
    fn model_top_p_alone_suppresses_the_fallback_top_k() {
        let m = ModelSamplingDefaults {
            top_p: Some(0.9),
            ..Default::default()
        };
        let p = params_of_for(
            None,
            None,
            None,
            None,
            None,
            &m,
            &SamplingOverrides::default(),
        );
        assert_eq!(p.top_p, 0.9);
        assert_eq!(
            p.top_k, 0,
            "no fallback top_k when the model already truncates"
        );
    }

    #[test]
    fn explicit_top_p_suppresses_the_model_top_k() {
        let qwen38 = ModelSamplingDefaults {
            top_p: Some(0.95),
            top_k: Some(20),
            ..Default::default()
        };
        let p = params_of_for(
            None,
            Some(1.0),
            None,
            None,
            None,
            &qwen38,
            &SamplingOverrides::default(),
        );
        assert_eq!(p.top_p, 1.0, "caller asked for the full distribution");
        assert_eq!(p.top_k, 0, "and must not inherit the model's top_k");
    }

    #[test]
    fn explicit_top_k_suppresses_the_model_top_p() {
        let qwen38 = ModelSamplingDefaults {
            top_p: Some(0.95),
            top_k: Some(20),
            ..Default::default()
        };
        let p = params_of_for(
            None,
            None,
            Some(5),
            None,
            None,
            &qwen38,
            &SamplingOverrides::default(),
        );
        assert_eq!(p.top_k, 5);
        assert_eq!(
            p.top_p, 1.0,
            "model top_p is set aside once the caller truncates"
        );
    }

    #[test]
    fn model_min_p_alone_is_a_truncation_claim() {
        let m = ModelSamplingDefaults {
            min_p: Some(0.05),
            ..Default::default()
        };
        let p = params_of_for(
            None,
            None,
            None,
            None,
            None,
            &m,
            &SamplingOverrides::default(),
        );
        assert_eq!(p.min_p, 0.05);
        assert_eq!(
            p.top_k, 0,
            "no fallback top_k stacked on the model's policy"
        );
    }

    #[test]
    fn advertised_defaults_follow_the_model() {
        let qwen38 = ModelSamplingDefaults {
            top_p: Some(0.95),
            top_k: Some(20),
            ..Default::default()
        };
        let adv = advertised_generation_defaults(&qwen38, &SamplingOverrides::default());
        assert_eq!(adv.top_k, 20);
        assert_eq!(adv.top_p, 0.95);
        let echoed = params_of_for(
            Some(adv.temperature),
            Some(adv.top_p),
            Some(adv.top_k),
            None,
            None,
            &qwen38,
            &SamplingOverrides::default(),
        );
        assert_eq!(
            echoed.top_k, 20,
            "echoing the advertised values changes nothing"
        );
        assert_eq!(echoed.top_p, 0.95);
    }

    #[test]
    fn operator_flags_sit_between_caller_and_bundle() {
        let qwen38 = ModelSamplingDefaults {
            temperature: Some(0.6),
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.0),
            do_sample: Some(true),
            repetition_penalty: None,
        };
        let op = SamplingOverrides {
            top_k: Some(64),
            ..Default::default()
        };

        let p = params_of_for(None, None, None, None, None, &qwen38, &op);
        assert_eq!(p.top_k, 64, "operator top_k overrides the bundle's");
        assert_eq!(
            p.top_p, 1.0,
            "and takes the whole level: the bundle's top_p is set aside"
        );
        assert_eq!(
            p.temperature, 0.6,
            "temperature is a separate rung and still comes from the bundle"
        );

        let p = params_of_for(None, None, Some(7), None, None, &qwen38, &op);
        assert_eq!(p.top_k, 7, "an explicit request value wins over the flag");

        let p = params_of_for(
            None,
            None,
            None,
            None,
            None,
            &ModelSamplingDefaults::default(),
            &op,
        );
        assert_eq!(p.top_k, 64);
    }

    #[test]
    fn operator_temperature_precedence() {
        let greedy_model = ModelSamplingDefaults {
            do_sample: Some(false),
            ..Default::default()
        };
        let op = SamplingOverrides {
            temperature: Some(0.9),
            ..Default::default()
        };
        assert_eq!(
            params_of_for(None, None, None, None, None, &greedy_model, &op).temperature,
            0.9,
            "the operator outranks the checkpoint"
        );
        assert_eq!(
            params_of_for(Some(0.0), None, None, None, None, &greedy_model, &op).temperature,
            0.0,
            "an explicit request temperature still wins"
        );
    }

    #[test]
    fn an_unknown_penalty_is_omitted_but_a_pinned_one_is_not() {
        let advertise = |resolved: Option<ModelSamplingDefaults>, op: SamplingOverrides| {
            (resolved.is_some() || op.repeat_penalty.is_some())
                .then(|| repeat_penalty_for(None, &op, &resolved.clone().unwrap_or_default()))
        };
        let none = SamplingOverrides::default();

        assert_eq!(advertise(None, none), None);

        let op = SamplingOverrides {
            repeat_penalty: Some(1.2),
            ..Default::default()
        };
        assert_eq!(advertise(None, op), Some(1.2));

        let quiet = ModelSamplingDefaults::default();
        assert_eq!(advertise(Some(quiet), none), Some(DEFAULT_REPEAT_PENALTY));
        let opted_out = ModelSamplingDefaults {
            repetition_penalty: Some(1.0),
            ..Default::default()
        };
        assert_eq!(advertise(Some(opted_out), none), Some(1.0));
    }

    #[test]
    fn operator_owned_settings_survive_a_failed_bundle_lookup() {
        let advertise = |resolved: Option<ModelSamplingDefaults>, op: SamplingOverrides| {
            let eff = advertised_generation_defaults(&resolved.clone().unwrap_or_default(), &op);
            let temp = (resolved.is_some() || op.temperature.is_some()).then_some(eff.temperature);
            let trunc = (resolved.is_some() || op.truncates()).then_some((eff.top_p, eff.top_k));
            (temp, trunc)
        };

        assert_eq!(advertise(None, SamplingOverrides::default()), (None, None));

        let op = SamplingOverrides {
            top_k: Some(64),
            ..Default::default()
        };
        assert_eq!(advertise(None, op), (None, Some((1.0, 64))));

        let op = SamplingOverrides {
            temperature: Some(0.4),
            ..Default::default()
        };
        assert_eq!(advertise(None, op), (Some(0.4), None));

        let qwen38 = ModelSamplingDefaults {
            temperature: Some(0.6),
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.0),
            do_sample: Some(true),
            repetition_penalty: None,
        };
        let op = SamplingOverrides {
            top_k: Some(64),
            ..Default::default()
        };
        assert_eq!(advertise(Some(qwen38), op), (Some(0.6), Some((1.0, 64))));
    }

    #[test]
    fn repeat_penalty_resolves_caller_then_operator_then_bundle_then_default() {
        let none = SamplingOverrides::default();
        let no_model = ModelSamplingDefaults::default();

        assert_eq!(
            repeat_penalty_for(None, &none, &no_model),
            DEFAULT_REPEAT_PENALTY
        );
        assert_eq!(repeat_penalty_for(Some(1.2), &none, &no_model), 1.2);

        let model = ModelSamplingDefaults {
            repetition_penalty: Some(1.1),
            ..Default::default()
        };
        assert_eq!(
            repeat_penalty_for(None, &none, &model),
            1.1,
            "bundle beats the runtime default"
        );

        let opted_out = ModelSamplingDefaults {
            repetition_penalty: Some(1.0),
            ..Default::default()
        };
        assert_eq!(repeat_penalty_for(None, &none, &opted_out), 1.0);

        let op = SamplingOverrides {
            repeat_penalty: Some(1.2),
            ..Default::default()
        };
        assert_eq!(
            repeat_penalty_for(None, &op, &model),
            1.2,
            "flag beats the bundle"
        );
        assert_eq!(
            repeat_penalty_for(Some(1.0), &op, &model),
            1.0,
            "caller's 1.0 opts out"
        );
    }

    #[test]
    fn echoing_advertised_min_p_is_stable() {
        let model = ModelSamplingDefaults {
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.05),
            ..Default::default()
        };
        let adv = advertised_generation_defaults(&model, &SamplingOverrides::default());
        assert_eq!(adv.min_p, 0.05, "advertised min_p comes from the model");
        let echoed = params_of_for(
            Some(adv.temperature),
            Some(adv.top_p),
            Some(adv.top_k),
            Some(adv.min_p),
            None,
            &model,
            &SamplingOverrides::default(),
        );
        assert_eq!(echoed.min_p, 0.05, "echoed min_p survives the round trip");
        assert_eq!(echoed.top_p, 0.95);
        assert_eq!(echoed.top_k, 20);
    }
}

#[cfg(test)]
mod stop_tests {
    use super::{stop_prefix_holdback, truncate_at_stop};

    fn stops(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn holdback_is_the_longest_stop_prefix_at_the_tail() {
        let st = stops(&["END", "<|eot|>"]);
        assert_eq!(stop_prefix_holdback("hello", &st), 0);
        assert_eq!(stop_prefix_holdback("hello E", &st), 1);
        assert_eq!(stop_prefix_holdback("hello EN", &st), 2);
        assert_eq!(stop_prefix_holdback("hello END", &st), 0);
        assert_eq!(stop_prefix_holdback("x <|eot", &st), 5);
        assert_eq!(stop_prefix_holdback("a <|e", &stops(&["<|eot|>", "e"])), 3);
        assert_eq!(
            stop_prefix_holdback("héllo —", &stops(&["— fin"])),
            "—".len()
        );
        assert_eq!(stop_prefix_holdback("abc", &stops(&[""])), 0);
    }

    #[test]
    fn delta_stream_never_emits_a_stop_prefix_that_the_next_delta_completes() {
        let st = stops(&["END"]);
        let mut acc = String::new();
        let mut sent = 0usize;
        let mut emitted = String::new();
        for delta in ["hello E", "ND world"] {
            acc.push_str(delta);
            let (mut t, hit) = truncate_at_stop(&acc, &st);
            if !hit {
                let keep = t.len() - stop_prefix_holdback(&t, &st);
                t.truncate(keep);
            }
            emitted.push_str(t.get(sent..).unwrap_or(""));
            sent = sent.max(t.len());
            if hit {
                break;
            }
        }
        assert_eq!(emitted, "hello ");
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct DecodeSpan {
    first: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
    first_count: u64,
    last_count: u64,
    tokens: u64,
    elapsed: std::time::Duration,
}

impl DecodeSpan {
    pub(crate) fn stamp(&mut self, produced: u64) {
        self.stamp_at(std::time::Instant::now(), produced);
    }

    fn stamp_at(&mut self, now: std::time::Instant, produced: u64) {
        if self.first.is_none() {
            self.first = Some(now);
            self.first_count = produced;
        }
        self.last = Some(now);
        self.last_count = produced.max(self.last_count);
    }

    pub(crate) fn first_token(&self) -> Option<std::time::Instant> {
        self.first
    }

    pub(crate) fn finish(&mut self) {
        if let (Some(f), Some(l)) = (self.first, self.last) {
            self.tokens += self.last_count.saturating_sub(self.first_count);
            self.elapsed += l.saturating_duration_since(f);
        }
        self.first = None;
        self.last = None;
        self.first_count = 0;
        self.last_count = 0;
    }

    fn measured(&self) -> (u64, Option<std::time::Duration>) {
        if self.elapsed == std::time::Duration::ZERO {
            (0, None)
        } else {
            (self.tokens, Some(self.elapsed))
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct GenTiming {
    pub t0: std::time::Instant,
    pub first: Option<std::time::Instant>,
    pub decode: DecodeSpan,
}

fn fim_log_cached(c: &crate::Completion) -> u64 {
    if c.cached {
        c.prompt_tokens as u64
    } else {
        0
    }
}

fn prefill_rate(prompt: u64, cached: u64, ttft: Option<std::time::Duration>) -> String {
    let prefilled = prompt - cached.min(prompt);
    ttft.filter(|d| d.as_secs_f64() > 0.0 && prefilled > 0)
        .map(|d| format!("{:.0}", prefilled as f64 / d.as_secs_f64()))
        .unwrap_or_else(|| "-".into())
}

fn decode_rate(tokens: u64, span: Option<std::time::Duration>) -> String {
    span.filter(|d| tokens > 0 && d.as_millis() as u64 >= tokens)
        .map(|d| format!("{:.1}", tokens as f64 / d.as_secs_f64()))
        .unwrap_or_else(|| "-".into())
}

pub(crate) fn access_log(
    route: &str,
    model: &str,
    prompt: u64,
    cached: u64,
    completion: u64,
    timing: GenTiming,
    finish: &str,
) {
    let GenTiming { t0, first, decode } = timing;
    let total_ms = t0.elapsed().as_millis();
    let ttft = first.map(|f| f.saturating_duration_since(t0));
    let (decode_tokens, decode_span) = decode.measured();
    let prefill = prefill_rate(prompt, cached, ttft);
    let decode = decode_rate(decode_tokens, decode_span);
    tracing::info!(
        "<-- 200 {route} model={model} prompt={prompt} cached={cached} completion={completion} ttft={}ms prefill={prefill} tok/s decode={decode} tok/s total={total_ms}ms finish={finish}",
        ttft.map(|d| d.as_millis()).unwrap_or(0),
    );
}

pub(crate) fn extras_of(
    presence: Option<f32>,
    frequency: Option<f32>,
    repeat: Option<f32>,
    op: &SamplingOverrides,
    model: &ModelSamplingDefaults,
) -> crate::scheduler::GenExtras {
    let mut frequency_penalty = frequency.unwrap_or(0.0);
    let mut repeat_penalty = repeat_penalty_for(repeat, op, model);
    if op.fallback.is_some_and(|f| f.frequency_penalty_as_repeat)
        && repeat.is_none()
        && op.repeat_penalty.is_none()
    {
        if presence.is_some() {
            repeat_penalty = 1.0;
        } else if let Some(f) = frequency {
            repeat_penalty = 1.0 + f;
            frequency_penalty = 0.0;
        }
    }
    crate::scheduler::GenExtras {
        ephemeral: true,
        presence_penalty: presence.unwrap_or(0.0),
        frequency_penalty,
        repeat_penalty,
        grammar_handle: 0,
        logit_bias_handle: 0,
        want_logprobs: false,
        top_logprobs: 0,
        wants_deltas: false,
        open_channel: 0,
        thinking: None,
        ignore_eos: false,
        seed_drawn: false,
        tool_schemas: None,
    }
}

pub(crate) fn logit_bias_handle_of(
    daemon: &Daemon,
    map: &Option<std::collections::HashMap<String, f32>>,
) -> Option<u32> {
    let map = map.as_ref()?;
    if map.is_empty() {
        return None;
    }
    let mut tokens = Vec::with_capacity(map.len());
    let mut values = Vec::with_capacity(map.len());
    for (k, v) in map {
        if let Ok(t) = k.parse::<i32>() {
            tokens.push(t);
            values.push(*v);
        }
    }
    if tokens.is_empty() {
        return None;
    }
    daemon.logit_bias_create(&tokens, &values).ok()
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct StreamOptions {
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) include_usage: bool,
    #[serde(default)]
    pub(crate) continuous_usage_stats: bool,
}

#[derive(Deserialize)]
pub(crate) struct CompletionRequest {
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) ignore_eos: bool,
    #[serde(default)]
    pub(crate) prompt: String,
    #[serde(default)]
    pub(crate) stream: bool,
    #[serde(default)]
    pub(crate) echo: bool,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) seed: Option<u64>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    #[serde(alias = "repetition_penalty")]
    pub(crate) repeat_penalty: Option<f32>,
    pub(crate) min_p: Option<f32>,
    pub(crate) logprobs: Option<u32>,
    pub(crate) suffix: Option<String>,
    pub(crate) fim_mode: Option<String>,
}

pub(crate) const DEFAULT_TOP_K: u32 = 40;

pub(crate) const DEFAULT_REPEAT_PENALTY: f32 = 1.0;

#[cfg(test)]
mod embedding_sampling_tests {
    use super::*;

    fn policy() -> SamplingOverrides {
        SamplingOverrides {
            fallback: Some(SamplingFallback {
                temperature: 0.0, repeat_penalty: 1.07,
                top_p: 0.82, top_k: 17, min_p: 0.03,
                frequency_penalty_as_repeat: true,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn omitted_parameters_and_props_share_embedding_defaults() {
        let m = ModelSamplingDefaults::default();
        let op = policy();
        let p = advertised_generation_defaults(&m, &op);
        assert_eq!((p.temperature, p.top_p, p.top_k, p.min_p), (0.0, 0.82, 17, 0.03));
        assert_eq!(repeat_penalty_for(None, &op, &m), 1.07);
        let original = advertised_generation_defaults(&m, &SamplingOverrides::default());
        assert_eq!((original.temperature, original.top_p, original.top_k, original.min_p), (1.0, 1.0, 40, 0.0));
        assert_eq!(repeat_penalty_for(None, &SamplingOverrides::default(), &m), 1.0);
    }

    #[test]
    fn model_truncation_wins_as_a_set_but_not_model_temperature_or_repeat() {
        for model in [
            ModelSamplingDefaults { top_p: Some(0.95), ..Default::default() },
            ModelSamplingDefaults { top_k: Some(9), ..Default::default() },
            ModelSamplingDefaults { min_p: Some(0.1), ..Default::default() },
        ] {
            let model = ModelSamplingDefaults { temperature: Some(0.7), repetition_penalty: Some(1.2), ..model };
            let p = advertised_generation_defaults(&model, &policy());
            assert_eq!(p.temperature, 0.0);
            assert_eq!((p.top_p, p.top_k, p.min_p), (model.top_p.unwrap_or(1.0), model.top_k.unwrap_or(0), model.min_p.unwrap_or(0.0)));
            assert_eq!(repeat_penalty_for(None, &policy(), &model), 1.07);
        }
    }

    #[test]
    fn caller_and_operator_values_override_embedding_without_merging_truncation() {
        let m = ModelSamplingDefaults { top_k: Some(90), ..Default::default() };
        let op = SamplingOverrides { temperature: Some(0.6), top_k: Some(12), repeat_penalty: Some(1.3), ..policy() };
        let p = advertised_generation_defaults(&m, &op);
        assert_eq!((p.temperature, p.top_p, p.top_k, p.min_p), (0.6, 1.0, 12, 0.0));
        assert_eq!(repeat_penalty_for(None, &op, &m), 1.3);
        let p = params_of_for(Some(0.2), Some(1.0), None, None, Some(42), &m, &op);
        assert_eq!((p.temperature, p.top_p, p.top_k, p.min_p, p.seed), (0.2, 1.0, 0, 0.0, 42));
        assert_eq!(repeat_penalty_for(Some(1.0), &op, &m), 1.0);
    }

    #[test]
    fn penalty_mapping_is_opt_in_and_respects_explicit_parameters() {
        let m = ModelSamplingDefaults::default();
        for (presence, frequency, repeat, expected) in [
            (None, None, None, (0.0, 0.0, 1.07)),
            (None, Some(0.2), None, (0.0, 0.0, 1.2)),
            (None, Some(0.0), None, (0.0, 0.0, 1.0)),
            (Some(0.0), Some(0.2), None, (0.0, 0.2, 1.0)),
            (Some(0.3), None, None, (0.3, 0.0, 1.0)),
            (None, Some(0.2), Some(1.1), (0.0, 0.2, 1.1)),
        ] {
            let e = extras_of(presence, frequency, repeat, &policy(), &m);
            assert_eq!((e.presence_penalty, e.frequency_penalty, e.repeat_penalty), expected);
        }
        let e = extras_of(None, Some(0.2), None, &SamplingOverrides::default(), &m);
        assert_eq!((e.frequency_penalty, e.repeat_penalty), (0.2, 1.0));
    }
}

pub(crate) fn advertised_generation_defaults(
    model: &ModelSamplingDefaults,
    op: &SamplingOverrides,
) -> GenParams {
    params_of_for(None, None, None, None, None, model, op)
}

pub(crate) fn repeat_penalty_for(
    caller: Option<f32>,
    op: &SamplingOverrides,
    model: &ModelSamplingDefaults,
) -> f32 {
    caller
        .or(op.repeat_penalty)
        .or(op.fallback.map(|f| f.repeat_penalty))
        .or(model.repetition_penalty)
        .unwrap_or(DEFAULT_REPEAT_PENALTY)
}

pub(crate) fn draw_seed_if_unseeded(params: &mut GenParams, client_seed: Option<u64>) -> bool {
    if client_seed.is_some() || params.temperature <= 0.0 {
        return false;
    }
    params.seed = fresh_seed();
    true
}

fn fresh_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let h = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();
    if h == 0 {
        1
    } else {
        h
    }
}

pub(crate) fn params_of_for(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    seed: Option<u64>,
    model: &ModelSamplingDefaults,
    op: &SamplingOverrides,
) -> GenParams {
    let model_temperature = match model.do_sample {
        Some(false) => Some(0.0),
        _ => model.temperature,
    };
    let chosen = temperature.is_some() || op.temperature.is_some();
    let temperature = temperature
        .or(op.temperature)
        .or(op.fallback.map(|f| f.temperature))
        .or(model_temperature)
        .unwrap_or(1.0);
    let temperature = match op.spec_max_temperature {
        Some(cap) if !chosen => temperature.min(cap),
        _ => temperature,
    };

    let caller_truncates = top_p.is_some() || top_k.is_some() || min_p.is_some();
    let model_truncates = model.top_p.is_some() || model.top_k.is_some() || model.min_p.is_some();
    let (eff_top_p, eff_top_k, eff_min_p) = if caller_truncates {
        (top_p, top_k, min_p)
    } else if op.truncates() {
        (op.top_p, op.top_k, op.min_p)
    } else if model_truncates {
        (model.top_p, model.top_k, model.min_p)
    } else if let Some(f) = op.fallback {
        (Some(f.top_p), Some(f.top_k), Some(f.min_p))
    } else if temperature > 0.0 {
        (None, Some(DEFAULT_TOP_K), None)
    } else {
        (None, None, None)
    };
    GenParams {
        temperature,
        top_p: eff_top_p.unwrap_or(1.0),
        min_p: eff_min_p.unwrap_or(0.0),
        top_k: eff_top_k.unwrap_or(0),
        seed: seed.unwrap_or(0),
    }
}

#[derive(Deserialize, Default)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) tool_calls: Vec<ReqToolCall>,
    #[serde(default)]
    pub(crate) tool_call_id: Option<String>,
    #[serde(default)]
    pub(crate) name: Option<String>,
    #[serde(default)]
    pub(crate) reasoning_content: Option<String>,
    #[serde(default)]
    pub(crate) reasoning: Option<String>,
}

pub(crate) fn reasoning_of(m: &ChatMessage, role: u32) -> Option<String> {
    if role != role::ASSISTANT {
        return None;
    }
    m.reasoning_content
        .as_deref()
        .or(m.reasoning.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MediaKind {
    Image,
    Audio,
}

fn require_media(daemon: &Daemon, kind: MediaKind, param: &str) -> Result<(), DaemonError> {
    match kind {
        MediaKind::Image => daemon.require("modalities", "image_encode", "unsupported_input", Some(param), "does not accept images"),
        MediaKind::Audio => daemon.require("modalities", "gemma_audio_encode", "unsupported_input", Some(param), "does not accept audio"),
    }
}

fn content_param(messages: &[ChatMessage], m: &ChatMessage) -> String {
    match messages.iter().position(|x| std::ptr::eq(x, m)) {
        Some(i) => format!("messages[{i}].content"),
        None => "messages".to_string(),
    }
}

pub(crate) struct SplitContent {
    pub(crate) text: String,
    pub(crate) images: Vec<(String, Vec<u8>, MediaKind)>,
}

pub(crate) fn absorbable_system_text(m: &ChatMessage) -> Result<Option<String>, DaemonError> {
    if m.role != "system" {
        return Ok(None);
    }
    let split = split_content(&m.content)?;
    Ok(split.images.is_empty().then_some(split.text))
}

pub(crate) fn split_content(v: &Option<serde_json::Value>) -> Result<SplitContent, DaemonError> {
    let mut out = SplitContent {
        text: String::new(),
        images: Vec::new(),
    };
    match v {
        None => {}
        Some(serde_json::Value::String(s)) => out.text = s.clone(),
        Some(serde_json::Value::Array(parts)) => {
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                            out.text.push_str(t);
                        }
                    }
                    Some("image_url") => {
                        let url = p
                            .get("image_url")
                            .and_then(|u| u.get("url"))
                            .and_then(|u| u.as_str())
                            .unwrap_or("");
                        let Some(rest) = url.strip_prefix("data:") else {
                            return Err(DaemonError::Protocol(
                                "image_url must be a data: URL (remote images are not fetched)",
                            ));
                        };
                        let Some((_, b64)) = rest.split_once(";base64,") else {
                            return Err(DaemonError::Protocol("image_url data URL must be base64"));
                        };
                        let bytes = base64_decode(b64)?;
                        out.images
                            .push((std::mem::take(&mut out.text), bytes, MediaKind::Image));
                    }
                    Some("input_audio") => {
                        let ia = p.get("input_audio");
                        let fmt = ia
                            .and_then(|a| a.get("format"))
                            .and_then(|f| f.as_str())
                            .unwrap_or("wav");
                        if fmt != "wav" {
                            return Err(DaemonError::Protocol(
                                "input_audio format must be wav (16-bit PCM or float32)",
                            ));
                        }
                        let b64 = ia
                            .and_then(|a| a.get("data"))
                            .and_then(|d| d.as_str())
                            .ok_or(DaemonError::Protocol("input_audio requires base64 data"))?;
                        let bytes = base64_decode(b64)?;
                        out.images
                            .push((std::mem::take(&mut out.text), bytes, MediaKind::Audio));
                    }
                    _ => {}
                }
            }
        }
        Some(_) => {
            return Err(DaemonError::Protocol(
                "message content must be a string or parts",
            ))
        }
    }
    Ok(out)
}

#[derive(Deserialize)]
pub(crate) struct ReqToolCall {
    #[serde(default)]
    pub(crate) id: Option<String>,
    pub(crate) function: Option<ReqFunction>,
}

#[derive(Deserialize)]
pub(crate) struct ReqFunction {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) arguments: serde_json::Value,
}

pub(crate) fn build_session(
    daemon: &Daemon,
    req: &ChatRequest,
    op: &SamplingOverrides,
    model: &ModelSamplingDefaults,
) -> Result<(u64, bool), DaemonError> {
    if let Err(why) = daemon.validate_template_kwargs(&req.template_kwargs()) {
        return Err(DaemonError::Constraint(why));
    }
    let mut params = params_of_for(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.seed,
        model,
        &op.for_daemon(daemon),
    );
    let seed_drawn = draw_seed_if_unseeded(&mut params, req.seed);

    if let Some(c) = req.continues {
        // Only a codec that renders each message alone renders a turn
        // appended to a stored conversation as it would render it fresh.
        if !daemon.renders_per_message() {
            return Err(DaemonError::Protocol(
                "this model's chat template renders whole conversations; a stored one is not continued by appending",
            ));
        }
        let session = daemon.fork(c.session, c.at_event, Some(params))?;
        // The stored conversation ends after a query: a turn here follows
        // none of this request's user messages, or comes after the last one.
        let last_query = req.messages.iter().rposition(|m| m.role == "user");
        for (i, m) in req.messages.iter().enumerate() {
            append_chat_message(daemon, session, req, m, last_query.is_none_or(|q| i > q))?;
        }
        return Ok((session, seed_drawn));
    }

    let session = daemon.create(None, params)?;

    if !daemon.renders_per_message() {
        let mut msgs: Vec<crate::codec::ChatMessage> = Vec::with_capacity(req.messages.len());
        for m in &req.messages {
            let split = split_content(&m.content)?;
            let role = match m.role.as_str() {
                "system" => role::SYSTEM,
                "assistant" => role::ASSISTANT,
                "tool" => role::TOOL,
                _ => role::USER,
            };
            if !split.images.is_empty() {
                let mut parts: Vec<crate::codec::ContentPart> = Vec::new();
                let trailing = split.text;
                for (pre, bytes, kind) in split.images {
                    require_media(daemon, kind, &content_param(&req.messages, m))?;
                    if !pre.is_empty() {
                        parts.push(crate::codec::ContentPart::Text(pre));
                    }
                    let hash = daemon.put_media(&bytes)?;
                    parts.push(match kind {
                        MediaKind::Image => crate::codec::ContentPart::Image { blob: hash },
                        MediaKind::Audio => crate::codec::ContentPart::Audio { blob: hash },
                    });
                }
                if !trailing.is_empty() {
                    parts.push(crate::codec::ContentPart::Text(trailing));
                }
                let mut msg = crate::codec::ChatMessage::new(role, String::new());
                msg.parts = parts;
                msg.reasoning = reasoning_of(m, role);
                msgs.push(msg);
                continue;
            }
            let mut msg = crate::codec::ChatMessage::new(role, split.text);
            for c in &m.tool_calls {
                let Some(f) = &c.function else { continue };
                let args = match &f.arguments {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                msg.tool_calls.push(crate::codec::ToolCallMsg {
                    id: c.id.clone().unwrap_or_default(),
                    name: f.name.clone(),
                    arguments: args,
                });
            }
            msg.reasoning = reasoning_of(m, role);
            if role == role::TOOL {
                msg.tool_call_id = m.tool_call_id.clone();
                msg.name = m.name.clone();
            }
            msgs.push(msg);
        }
        let tools: Vec<String> = req.tools.iter().map(|t| t.to_string()).collect();
        daemon.append_conversation(session, &msgs, &tools, &req.template_kwargs())?;
        return Ok((session, seed_drawn));
    }

    let (system, tool_jsons) = system_head(req)?;
    let rest = if system.is_some() { &req.messages[1..] } else { &req.messages[..] };
    daemon.append_system_full(session, system, tool_jsons, &req.template_kwargs())?;
    // A turn after the last user query is a step of a tool loop.
    let last_query = req.messages.iter().rposition(|m| m.role == "user");
    let skipped = req.messages.len() - rest.len();
    for (i, m) in rest.iter().enumerate() {
        let after_query = last_query.is_some_and(|q| skipped + i > q);
        append_chat_message(daemon, session, req, m, after_query)?;
    }
    Ok((session, seed_drawn))
}

/// What a per-message codec renders first: the leading system message when
/// it is text only, and the tools when the request may call them. Two
/// requests with the same head render the same first turn.
pub(crate) fn system_head(req: &ChatRequest) -> Result<(Option<String>, Vec<String>), DaemonError> {
    let system = req.messages.first().map(absorbable_system_text).transpose()?.flatten();
    let tools = if !req.tools.is_empty() && ToolChoice::parse(&req.tool_choice).declares_tools() {
        req.tools.iter().map(|t| t.to_string()).collect()
    } else {
        Vec::new()
    };
    Ok((system, tools))
}

fn append_chat_message(
    daemon: &Daemon,
    session: u64,
    req: &ChatRequest,
    m: &ChatMessage,
    after_query: bool,
) -> Result<(), DaemonError> {
    let split = split_content(&m.content)?;
    if !split.images.is_empty() {
        let r = match m.role.as_str() {
            "assistant" => role::ASSISTANT,
            "system" => role::SYSTEM,
            _ => role::USER,
        };
        let n = split.images.len();
        let mut trailing = split.text;
        for (i, (pre, bytes, kind)) in split.images.into_iter().enumerate() {
            require_media(daemon, kind, &content_param(&req.messages, m))?;
            let hash = daemon.put_media(&bytes)?;
            let post = if i + 1 == n {
                std::mem::take(&mut trailing)
            } else {
                String::new()
            };
            daemon.append_image(session, r, &hash, &pre, &post)?;
        }
        return Ok(());
    }
    let content = split.text;
    match m.role.as_str() {
        "assistant" if !m.tool_calls.is_empty() => {
            let calls: Vec<(String, String)> = m
                .tool_calls
                .iter()
                .filter_map(|c| c.function.as_ref())
                .map(|f| {
                    let args = match &f.arguments {
                        serde_json::Value::String(s) => s.clone(),
                        v => v.to_string(),
                    };
                    (f.name.clone(), args)
                })
                .collect();
            daemon.append_assistant_with_tool_calls(session, content, calls, after_query)?;
        }
        "assistant" => {
            daemon.append_message(session, role::ASSISTANT, content)?;
        }
        "system" => {
            daemon.append_message(session, role::SYSTEM, content)?;
        }
        "tool" => {
            daemon.append_tool_history(session, content)?;
        }
        _ => {
            daemon.append_message(session, role::USER, content)?;
        }
    }
    Ok(())
}

pub(crate) struct Collected {
    pub(crate) content: String,
    pub(crate) reasoning: String,
    pub(crate) tool_calls: Vec<(u64, String, String)>,
    pub(crate) unparsed_tool: String,
    pub(crate) broken_calls: Vec<(u64, String, String)>,
}

pub(crate) fn promote_forced_tool_call(
    daemon: &Daemon,
    c: &mut Collected,
    session: u64,
    schemas: Option<&crate::codec::ToolSchemas>,
) {
    if !c.tool_calls.is_empty() {
        return;
    }
    let raw = c.content.trim().to_string();
    if raw.is_empty() {
        return;
    }
    if let Some((name, args)) = daemon.parse_tool_call_with(&raw, schemas) {
        c.tool_calls.push((session, name, args));
        c.content.clear();
    }
}

pub(crate) fn claim_named_calls(daemon: &Daemon, c: &mut Collected, events: &[crate::CommittedEvent], choice: &ToolChoice) {
    if *choice == ToolChoice::None {
        return;
    }
    let mut left = String::new();
    for e in events {
        if let EventBody::ToolParseFailure { raw } = &e.body {
            match daemon.named_tool_call(raw) {
                Some((name, args)) => c.broken_calls.push((e.event_id, name, args)),
                None => left.push_str(raw),
            }
        }
    }
    c.unparsed_tool = left;
}

pub(crate) fn collect(events: &[crate::CommittedEvent]) -> Collected {
    let mut c = Collected {
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
        unparsed_tool: String::new(),
        broken_calls: Vec::new(),
    };
    for e in events {
        match &e.body {
            EventBody::Generated {
                text, channel: ch, ..
            } => match *ch {
                channel::REASONING => c.reasoning.push_str(text),
                channel::TOOL_CALL => {}
                _ => c.content.push_str(text),
            },
            EventBody::ToolUse { name, arguments } => {
                c.tool_calls
                    .push((e.event_id, name.clone(), arguments.clone()));
            }
            EventBody::ToolParseFailure { raw } => c.unparsed_tool.push_str(raw),
            _ => {}
        }
    }
    c
}

pub(crate) fn finish_reason(
    fin: u32,
    produced: u32,
    max_tokens: u32,
    had_tool_calls: bool,
) -> &'static str {
    let cut = match fin {
        finish::LENGTH => true,
        finish::EOS | finish::CANCELLED => false,
        _ => produced >= max_tokens,
    };
    if cut {
        "length"
    } else if had_tool_calls {
        "tool_calls"
    } else {
        "stop"
    }
}

pub(crate) fn prompt_len(daemon: &Daemon, session: u64) -> u32 {
    daemon
        .store()
        .lock()
        .expect("store")
        .session(session)
        .map(|s| s.tokens.len().min(u32::MAX as usize) as u32)
        .unwrap_or(0)
}

pub(crate) fn next_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(1);
    format!("chatcmpl-superfluid-{}", N.fetch_add(1, Ordering::Relaxed))
}

pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod access_log_rate_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn prefill_counts_only_what_was_prefilled() {
        let cases = [
            (12961u64, 0u64, 10444u64, "1241"),
            (14481, 12800, 5716, "294"),
            (15627, 14900, 5057, "144"),
            (16045, 15900, 2138, "68"),
        ];
        for (prompt, cached, ttft_ms, want) in cases {
            let got = prefill_rate(prompt, cached, Some(Duration::from_millis(ttft_ms)));
            assert_eq!(got, want, "prompt={prompt} cached={cached}");
        }
    }

    #[test]
    fn a_fully_warm_prompt_prefilled_nothing() {
        assert_eq!(
            prefill_rate(4096, 4096, Some(Duration::from_millis(80))),
            "-"
        );
        assert_eq!(
            prefill_rate(4096, 9000, Some(Duration::from_millis(80))),
            "-"
        );
    }

    #[test]
    fn a_fim_cache_hit_logs_its_prompt_as_cached() {
        let c = |cached| crate::Completion {
            text: String::new(),
            tokens: vec![1; 16],
            cached,
            expired: false,
            finish: superfluid_abi::finish::NONE,
            prompt_tokens: 21,
        };
        assert_eq!(fim_log_cached(&c(true)), 21);
        assert_eq!(
            prefill_rate(
                21,
                fim_log_cached(&c(true)),
                Some(Duration::from_micros(60))
            ),
            "-"
        );
        assert_eq!(fim_log_cached(&c(false)), 0);
    }

    #[test]
    fn prefill_needs_a_ttft() {
        assert_eq!(prefill_rate(1024, 0, None), "-");
        assert_eq!(prefill_rate(1024, 0, Some(Duration::ZERO)), "-");
    }

    fn run(d: &mut DecodeSpan, base: std::time::Instant, stamps: &[(u64, u64)]) {
        for (ms, produced) in stamps {
            d.stamp_at(base + Duration::from_millis(*ms), *produced);
        }
        d.finish();
    }

    #[test]
    fn decode_is_the_inter_token_rate() {
        let base = std::time::Instant::now();
        let mut d = DecodeSpan::default();
        run(&mut d, base, &[(0, 1), (781, 200)]);
        let (tokens, span) = d.measured();
        assert_eq!(tokens, 199);
        assert_eq!(decode_rate(tokens, span), "254.8");
    }

    #[test]
    fn decode_excludes_the_whole_first_commit() {
        let base = std::time::Instant::now();
        let mut d = DecodeSpan::default();
        run(&mut d, base, &[(0, 5), (1000, 100)]);
        assert_eq!(d.measured().0, 95, "the first commit's 5 predate the span");
        assert_eq!(decode_rate(d.measured().0, d.measured().1), "95.0");
    }

    #[test]
    fn decode_spans_do_not_swallow_the_gaps_between_choices() {
        let base = std::time::Instant::now();
        let mut d = DecodeSpan::default();
        run(&mut d, base, &[(0, 1), (500, 51)]);
        run(&mut d, base, &[(4500, 1), (5000, 51)]);
        let (tokens, span) = d.measured();
        assert_eq!(tokens, 100);
        assert_eq!(span, Some(Duration::from_millis(1000)), "gap excluded");
        assert_eq!(decode_rate(tokens, span), "100.0");
    }

    #[test]
    fn decode_counts_only_what_was_delivered() {
        let base = std::time::Instant::now();
        let mut d = DecodeSpan::default();
        run(&mut d, base, &[(0, 1), (500, 50)]);
        let (tokens, span) = d.measured();
        assert_eq!(tokens, 49, "the undelivered tail is usage's business");
        assert_eq!(decode_rate(tokens, span), "98.0");
    }

    #[test]
    fn decode_declines_to_guess() {
        let base = std::time::Instant::now();
        let mut d = DecodeSpan::default();
        d.finish();
        assert_eq!(d.measured(), (0, None));
        assert_eq!(decode_rate(0, None), "-");

        let mut one = DecodeSpan::default();
        run(&mut one, base, &[(0, 40)]);
        assert_eq!(decode_rate(one.measured().0, one.measured().1), "-");

        assert_eq!(decode_rate(200, Some(Duration::from_millis(150))), "-");
        assert_eq!(decode_rate(100, None), "-");
    }
}

#[cfg(test)]
mod key_order_tests {
    use super::ChatRequest;

    #[test]
    fn a_tool_keeps_the_key_order_it_was_sent_in() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"messages":[],"tools":[{"type":"function","function":{"name":"replace_text",
                "parameters":{"type":"object","properties":{"path":{"type":"string"},
                "old_text":{"type":"string"},"new_text":{"type":"string"}}}}}]}"#,
        )
        .expect("request");
        let t = req.tools[0].to_string();
        let at = |k: &str| t.find(k).unwrap_or_else(|| panic!("{k} missing: {t}"));
        assert!(at("\"type\"") < at("\"function\""), "{t}");
        assert!(
            at("\"path\"") < at("\"old_text\"") && at("\"old_text\"") < at("\"new_text\""),
            "{t}"
        );
    }
}

#[cfg(test)]
mod system_absorption_tests {
    use super::*;

    #[test]
    fn assistant_reasoning_is_carried_and_other_roles_are_not() {
        let mut m = msg("assistant", serde_json::Value::Null);
        m.reasoning_content = Some("  plan  ".into());
        assert_eq!(reasoning_of(&m, role::ASSISTANT), Some("plan".into()));
        m.reasoning_content = None;
        m.reasoning = Some("alt".into());
        assert_eq!(
            reasoning_of(&m, role::ASSISTANT),
            Some("alt".into()),
            "the other spelling"
        );
        m.reasoning = Some("   ".into());
        assert_eq!(
            reasoning_of(&m, role::ASSISTANT),
            None,
            "blank reasoning is none"
        );
        let mut u = msg("user", serde_json::json!("hi"));
        u.reasoning_content = Some("x".into());
        assert_eq!(
            reasoning_of(&u, role::USER),
            None,
            "only assistant turns carry reasoning"
        );
    }

    fn msg(role: &str, content: serde_json::Value) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(content),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            reasoning: None,
        }
    }

    fn image_part() -> serde_json::Value {
        serde_json::json!({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}
        })
    }

    #[test]
    fn text_only_system_message_is_absorbed() {
        let m = msg("system", serde_json::json!("you are helpful"));
        assert_eq!(
            absorbable_system_text(&m).expect("split"),
            Some("you are helpful".to_string()),
            "a text-only system message folds into the scaffold"
        );
    }

    #[test]
    fn multimodal_system_message_is_left_for_the_media_loop() {
        let m = msg(
            "system",
            serde_json::json!([{"type": "text", "text": "look at this"}, image_part()]),
        );
        let split = split_content(&m.content).expect("split");
        assert_eq!(split.images.len(), 1, "fixture must carry one image part");

        assert_eq!(
            absorbable_system_text(&m).expect("split"),
            None,
            "a system message with media must NOT be absorbed — its images \
             would be dropped on the floor"
        );
    }

    #[test]
    fn audio_system_message_is_left_for_the_media_loop() {
        let m = msg(
            "system",
            serde_json::json!([
                {"type": "text", "text": "listen"},
                {"type": "input_audio", "input_audio": {"data": "UklGRiQAAABXQVZF", "format": "wav"}}
            ]),
        );
        let split = split_content(&m.content).expect("split");
        assert_eq!(split.images.len(), 1, "fixture must carry one audio part");
        assert_eq!(absorbable_system_text(&m).expect("split"), None);
    }

    #[test]
    fn non_system_roles_are_never_absorbed() {
        for role in ["user", "assistant", "tool", "developer"] {
            let m = msg(role, serde_json::json!("hello"));
            assert_eq!(
                absorbable_system_text(&m).expect("split"),
                None,
                "{role} must not be absorbed into the system scaffold"
            );
        }
    }
}

pub(crate) enum RunError {
    Daemon(DaemonError),
    Worker(String),
}
impl From<DaemonError> for RunError {
    fn from(error: DaemonError) -> Self {
        Self::Daemon(error)
    }
}

pub(crate) async fn completions(
    daemon: Arc<Daemon>,
    model_name: String,
    req: CompletionRequest,
    policy: RequestPolicy,
    hold: Option<crate::keypolicy::ClientKey>,
    hooks: crate::keepalive::Hooks,
) -> Result<Generation, RunError> {
    let qos = policy.qos;
    let max_tokens = policy
        .limits
        .resolve(req.max_tokens, || daemon.tokenize(&req.prompt).len() as u32);
    let id = format!(
        "cmpl-superfluid-{}",
        next_id().rsplit('-').next().unwrap_or("0")
    );
    let created = unix_now();
    let model_defaults = {
        let d = daemon.clone();
        tokio::task::spawn_blocking(move || {
            crate::with_sync_class(Some(qos.class), || d.model_sampling_defaults())
        })
        .await
        .unwrap_or_default()
    };
    let mut params = params_of_for(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.seed,
        &model_defaults,
        &policy.sampling.for_daemon(&daemon),
    );
    let seed_drawn = draw_seed_if_unseeded(&mut params, req.seed);
    let mut extras = extras_of(
        req.presence_penalty,
        req.frequency_penalty,
        req.repeat_penalty,
        &policy.sampling,
        &model_defaults,
    );
    extras.seed_drawn = seed_drawn;
    let want_logprobs = req.logprobs.is_some();
    extras.want_logprobs = want_logprobs;
    extras.top_logprobs = req.logprobs.unwrap_or(0).min(20) as u8;
    extras.ignore_eos = req.ignore_eos;
    let echo_offset = if req.echo { req.prompt.len() } else { 0 };

    if req.stream {
        let body = completion_stream(daemon, req, max_tokens, params, extras, qos, hold).await?;
        return Ok(Generation {
            kind: Kind::Completion,
            id,
            model: model_name,
            created,
            warm_header: None,
            body,
        });
    }

    let daemon_lp = Arc::clone(&daemon);
    let model_log = model_name.clone();
    let out = blocking(qos, hold, move || {
        let mut hooks = hooks;
        let t0 = std::time::Instant::now();
        let mut decode = DecodeSpan::default();
        let mut produced = 0u64;
        let session = daemon.create(None, params)?;
        qos.apply(&daemon, session)?;
        daemon.append(session, Some(req.prompt.clone()), Vec::new())?;
        if hooks.is_on() {
            daemon.session_fits(session, None)?;
        }
        hooks.admitted(&daemon, session);
        hooks.check()?;
        let out = daemon.generate_streaming_ex(session, max_tokens, extras, |ev| {
            hooks.cancel_if_gone();
            if let EventBody::Generated { span, .. } = &ev.body {
                if !span.is_empty() {
                    produced += span.len() as u64;
                    decode.stamp(produced);
                }
            }
            Ok(())
        })?;
        let first_tok = decode.first_token();
        decode.finish();
        let c = collect(&out.events);
        let usage = usage(&daemon, session, out.tokens_generated, out.warm_prefix);
        access_log(
            "POST /v1/completions",
            &model_log,
            usage.prompt.unwrap_or(0),
            out.warm_prefix,
            out.tokens_generated as u64,
            GenTiming {
                t0,
                first: first_tok,
                decode,
            },
            finish_reason(out.finish, out.tokens_generated, max_tokens, false),
        );
        let mut text = String::new();
        if req.echo {
            text.push_str(&req.prompt);
        }
        text.push_str(&c.content);
        text.push_str(&c.reasoning);
        text.push_str(&c.unparsed_tool);
        Ok::<_, DaemonError>((out, text, usage))
    })
    .await;
    let (out, text, usage) = out?;
    let logprobs = want_logprobs.then(|| decode_logprobs(&daemon_lp, &out.logprobs, echo_offset));
    let mut frame = Frame::text(
        text,
        logprobs,
        Some(finish_reason(
            out.finish,
            out.tokens_generated,
            max_tokens,
            false,
        )),
    );
    frame.usage = Some(usage);
    Ok(Generation {
        kind: Kind::Completion,
        id,
        model: model_name,
        created,
        warm_header: None,
        body: Body::Complete(frame),
    })
}

#[allow(clippy::too_many_arguments)]
async fn completion_stream(
    daemon: Arc<Daemon>,
    req: CompletionRequest,
    max_tokens: u32,
    params: GenParams,
    extras: crate::scheduler::GenExtras,
    qos: RequestQos,
    hold: Option<crate::keypolicy::ClientKey>,
) -> Result<Body, RunError> {
    let want_logprobs = req.logprobs.is_some();
    let admitted_daemon = Arc::clone(&daemon);
    let (req, session) = blocking(qos, hold.clone(), move || {
        let session = admitted_daemon.create(None, params)?;
        qos.apply(&admitted_daemon, session)?;
        admitted_daemon.append(session, Some(req.prompt.clone()), Vec::new())?;
        admitted_daemon.session_fits(session, None)?;
        Ok((req, session))
    })
    .await?;
    let rx = spawn_stream(qos, hold, move |tx| {
        let _closed = crate::keepalive::cancel_when_closed(&daemon, session, tx);
        let send = |v: Frame| {
            tx.blocking_send(Ok(v))
                .map_err(|_| DaemonError::Protocol("generation consumer gone"))
        };
        let lp_q: std::rc::Rc<
            std::cell::RefCell<std::collections::VecDeque<crate::scheduler::TokenLogprob>>,
        > = std::rc::Rc::new(std::cell::RefCell::new(std::collections::VecDeque::new()));
        let lp_fill = std::rc::Rc::clone(&lp_q);
        let mut offset: usize = 0;
        let run = || -> Result<(), DaemonError> {
            if req.echo {
                offset = req.prompt.len();
                send(Frame::text(&req.prompt, None, None))?;
            }
            let out = daemon.generate_streaming_ex_lp(
                session,
                max_tokens,
                extras,
                |ev| {
                    if let EventBody::Generated {
                        text,
                        channel: ch,
                        span,
                        ..
                    } = &ev.body
                    {
                        let these: Vec<crate::scheduler::TokenLogprob> = if want_logprobs {
                            let mut q = lp_q.borrow_mut();
                            (0..span.len()).filter_map(|_| q.pop_front()).collect()
                        } else {
                            Vec::new()
                        };
                        if !text.is_empty() && *ch != channel::TOOL_CALL {
                            let lp = if want_logprobs {
                                let v = Some(decode_logprobs(&daemon, &these, offset));
                                offset += text.len();
                                v
                            } else {
                                None
                            };
                            return send(Frame::text(text, lp, None));
                        }
                    }
                    Ok(())
                },
                |items| {
                    lp_fill.borrow_mut().extend(items.iter().cloned());
                },
            )?;
            let fin = finish_reason(out.finish, out.tokens_generated, max_tokens, false);
            let mut terminal = Frame::text("", None, Some(fin));
            terminal.usage = Some(usage(
                &daemon,
                session,
                out.tokens_generated,
                out.warm_prefix,
            ));
            send(terminal)
        };
        run()
    });
    Ok(Body::Stream(rx))
}
pub(crate) async fn chat(
    daemon: Arc<Daemon>,
    model_name: String,
    req: ChatRequest,
    policy: RequestPolicy,
    hold: Option<crate::keypolicy::ClientKey>,
    hooks: crate::keepalive::Hooks,
) -> Result<Generation, RunError> {
    let qos = policy.qos;
    let explicit_max = req.max_completion_tokens.or(req.max_tokens);
    let id = next_id();
    let created = unix_now();

    if req.stream {
        let body = chat_stream(daemon, model_name.clone(), req, policy, explicit_max, hold).await?;
        return Ok(Generation {
            kind: Kind::Chat,
            id,
            model: model_name,
            created,
            warm_header: None,
            body,
        });
    }

    let n = req.n.unwrap_or(1).max(1);
    let route = req.route();
    let stops = req.stop.list();
    let model_log = model_name.clone();
    let out = blocking(qos, hold, move || {
        let mut hooks = hooks;
        let t0 = std::time::Instant::now();
        let mut decode = DecodeSpan::default();
        let mut ttft_first: Option<std::time::Instant> = None;
        let mut last_finish = "-";
        let model_defaults = daemon.model_sampling_defaults();
        let mut extras = extras_of(
            req.presence_penalty,
            req.frequency_penalty,
            req.repeat_penalty,
            &policy.sampling,
            &model_defaults,
        );
        let choice = ToolChoice::parse(&req.tool_choice);
        if choice.forces_call() {
            extras.open_channel = crate::wal::channel::TOOL_CALL;
        }
        let tool_schemas = crate::codec::ToolSchemas::of_values(&req.tools);
        extras.tool_schemas = tool_schemas.clone();
        let handles = RequestHandles::new(
            &daemon,
            grammar_handle_for(&daemon, &req, &choice)?,
            logit_bias_handle_of(&daemon, &req.logit_bias),
        );
        handles.apply(&mut extras);
        let want_logprobs = req.logprobs.unwrap_or(false);
        extras.want_logprobs = want_logprobs;
        extras.top_logprobs = req.top_logprobs.unwrap_or(0).min(20) as u8;
        extras.thinking = req.thinking();
        extras.ignore_eos = req.ignore_eos;
        let cancels = daemon.cancel_registry();
        let RequestPolicy {
            limits,
            sampling: op,
            qos,
        } = policy;
        let mut choices = Vec::with_capacity(n as usize);
        let mut completion_tokens = 0u64;
        let mut prompt_tokens = 0u64;
        let mut warm = 0u64;
        let mut warm_first: Option<u64> = None;
        let mut spec = (0u64, 0u64);
        let mut last_session = None;
        for i in 0..n {
            if i > 0 && hooks.gone() {
                break;
            }
            let (session, seed_drawn) = build_session(&daemon, &req, &op, &model_defaults)?;
            last_session = Some(session);
            let mut extras = extras.clone();
            extras.seed_drawn = seed_drawn;
            qos.apply(&daemon, session)?;
            if hooks.is_on() {
                daemon.session_fits(session, req.thinking())?;
            }
            hooks.admitted(&daemon, session);
            hooks.check()?;
            let max_tokens = limits.resolve(explicit_max, || prompt_len(&daemon, session));
            let mut acc = String::new();
            let mut produced = 0u64;
            let stops = stops.clone();
            let cancels = std::sync::Arc::clone(&cancels);
            let mut stopped = false;
            let out = daemon.generate_streaming_ex(session, max_tokens, extras, |ev| {
                hooks.cancel_if_gone();
                if let EventBody::Generated {
                    text,
                    channel: ch,
                    span,
                    ..
                } = &ev.body
                {
                    if !span.is_empty() {
                        produced += span.len() as u64;
                        decode.stamp(produced);
                    }
                    if *ch == channel::TEXT && !stops.is_empty() && !stopped {
                        acc.push_str(text);
                        if stops.iter().any(|s| acc.contains(s.as_str())) {
                            cancels.cancel(session);
                            stopped = true;
                        }
                    }
                }
                Ok(())
            })?;
            ttft_first = ttft_first.or_else(|| decode.first_token());
            decode.finish();
            let mut c = collect(&out.events);
            claim_named_calls(&daemon, &mut c, &out.events, &choice);
            if choice.forces_call() {
                promote_forced_tool_call(&daemon, &mut c, session, tool_schemas.as_deref());
            }
            let (content, hit_stop) = truncate_at_stop(&c.content, &stops);
            c.content = content;
            let usage = usage(&daemon, session, out.tokens_generated, out.warm_prefix);
            completion_tokens += usage.completion;
            spec.0 += out.spec.0;
            spec.1 += out.spec.1;
            prompt_tokens = usage.prompt.unwrap_or(prompt_tokens);
            warm = warm.max(out.warm_prefix);
            warm_first.get_or_insert(out.warm_prefix);
            let finish = if !c.broken_calls.is_empty() {
                "length"
            } else if hit_stop {
                "stop"
            } else {
                finish_reason(
                    out.finish,
                    out.tokens_generated,
                    max_tokens,
                    !c.tool_calls.is_empty() && choice != ToolChoice::None,
                )
            };
            choices.push(Choice {
                index: i,
                message: message(&c, &choice),
                finish: Some(finish),
                logprobs: want_logprobs.then(|| decode_logprobs(&daemon, &out.logprobs, 0)),
                ..Choice::default()
            });
            last_finish = finish;
        }
        access_log(
            route,
            &model_log,
            prompt_tokens,
            warm_first.unwrap_or(warm),
            completion_tokens,
            GenTiming {
                t0,
                first: ttft_first,
                decode,
            },
            last_finish,
        );
        Ok::<_, DaemonError>((choices, prompt_tokens, completion_tokens, warm, spec, last_session))
    })
    .await;
    let (choices, prompt_tokens, completion_tokens, warm, spec, session) = out?;
    let frame = Frame {
        choices,
        usage: Some(Usage::complete(
            prompt_tokens,
            completion_tokens,
            Some(warm.min(prompt_tokens)),
        )),
        spec: (spec.0 > 0).then_some(spec),
        session,
        ..Frame::default()
    };
    Ok(Generation {
        kind: Kind::Chat,
        id,
        model: model_name,
        created,
        warm_header: Some(warm),
        body: Body::Complete(frame),
    })
}

async fn chat_stream(
    daemon: Arc<Daemon>,
    model_name: String,
    req: ChatRequest,
    policy: RequestPolicy,
    explicit_max: Option<u32>,
    hold: Option<crate::keypolicy::ClientKey>,
) -> Result<Body, RunError> {
    let RequestPolicy {
        limits,
        sampling: op,
        qos,
    } = policy;
    let t0 = std::time::Instant::now();
    let admitted_daemon = Arc::clone(&daemon);
    let (req, session, seed_drawn, model_defaults) = blocking(qos, hold.clone(), move || {
        let model_defaults = admitted_daemon.model_sampling_defaults();
        let (session, seed_drawn) = build_session(&admitted_daemon, &req, &op, &model_defaults)?;
        qos.apply(&admitted_daemon, session)?;
        admitted_daemon.session_fits(session, req.thinking())?;
        Ok((req, session, seed_drawn, model_defaults))
    })
    .await?;
    let rx = spawn_stream(qos, hold, move |tx| {
        let _closed = crate::keepalive::cancel_when_closed(&daemon, session, tx);
        let send = |v: Frame| -> Result<(), DaemonError> {
            tx.blocking_send(Ok(v))
                .map_err(|_| DaemonError::Protocol("generation consumer gone"))
        };
        let run = || -> Result<(), DaemonError> {
            send(Frame::chat(
                Message {
                    assistant: true,
                    ..Message::default()
                },
                None,
            ))?;
            let mut tool_index = 0usize;
            let mut orphaned_calls = 0usize;
            let mut extras = extras_of(
                req.presence_penalty,
                req.frequency_penalty,
                req.repeat_penalty,
                &op,
                &model_defaults,
            );
            extras.seed_drawn = seed_drawn;
            let choice = ToolChoice::parse(&req.tool_choice);
            if choice.forces_call() {
                extras.open_channel = crate::wal::channel::TOOL_CALL;
            }
            let tool_schemas = crate::codec::ToolSchemas::of_values(&req.tools);
            extras.tool_schemas = tool_schemas.clone();
            let mut frag =
                crate::tool_fragment::ToolCallStream::new(if choice == ToolChoice::None {
                    crate::tool_fragment::ToolFragmentMode::None
                } else {
                    daemon.tool_fragment_mode()
                })
                .with_schemas(tool_schemas.clone());
            let mut open_call_id = String::new();
            let handles = RequestHandles::new(
                &daemon,
                grammar_handle_for(&daemon, &req, &choice)?,
                logit_bias_handle_of(&daemon, &req.logit_bias),
            );
            handles.apply(&mut extras);
            let want_logprobs = req.logprobs.unwrap_or(false);
            extras.want_logprobs = want_logprobs;
            extras.top_logprobs = req.top_logprobs.unwrap_or(0).min(20) as u8;
            extras.thinking = req.thinking();
            extras.ignore_eos = req.ignore_eos;
            let lp_q: std::rc::Rc<
                std::cell::RefCell<std::collections::VecDeque<crate::scheduler::TokenLogprob>>,
            > = std::rc::Rc::new(std::cell::RefCell::new(std::collections::VecDeque::new()));
            let lp_fill = std::rc::Rc::clone(&lp_q);
            let max_tokens = limits.resolve(explicit_max, || prompt_len(&daemon, session));
            let stops = req.stop.list();
            let cancels = daemon.cancel_registry();
            let delta_mode = !want_logprobs;
            let continuous_usage = req
                .stream_options
                .as_ref()
                .is_some_and(|o| o.continuous_usage_stats);
            struct StreamText {
                usage_now: Option<u32>,
                committed: u32,
                decode: DecodeSpan,
                content_acc: String,
                content_sent: usize,
                stopped: bool,
                dead: bool,
            }
            let st = std::rc::Rc::new(std::cell::RefCell::new(StreamText {
                usage_now: None,
                committed: 0,
                decode: DecodeSpan::default(),
                content_acc: String::new(),
                content_sent: 0,
                stopped: false,
                dead: false,
            }));
            let push_text =
                |s: &mut StreamText, text: &str, lp: Option<Logprobs>| -> Result<(), DaemonError> {
                    if s.stopped {
                        return Ok(());
                    }
                    if choice.forces_call() {
                        s.content_acc.push_str(text);
                        return Ok(());
                    }
                    let start = s.content_sent;
                    s.content_acc.push_str(text);
                    let tail = &s.content_acc[start..];
                    let mut cut: Option<usize> = None;
                    for stop in &stops {
                        if let Some(i) = tail.find(stop.as_str()) {
                            cut = Some(cut.map_or(i, |c| c.min(i)));
                        }
                    }
                    let hit = cut.is_some();
                    let end = match cut {
                        Some(i) => start + i,
                        None => start + tail.len() - stop_prefix_holdback(tail, &stops),
                    };
                    let piece = s.content_acc.get(start..end).unwrap_or("").to_string();
                    s.content_sent = s.content_sent.max(end);
                    if !piece.is_empty() {
                        let mut ck = Frame::chat(Message::content(piece), None);
                        if let Some(lp) = lp {
                            ck.choices[0].logprobs = Some(lp);
                        }
                        if let Some(n) = s.usage_now {
                            ck.usage = Some(Usage::partial(n));
                        }
                        send(ck)?;
                    }
                    if hit {
                        cancels.cancel(session);
                        s.stopped = true;
                    }
                    Ok(())
                };
            let st_delta = std::rc::Rc::clone(&st);
            let on_delta = |ch: u32, text: &str, produced: u32| {
                if !delta_mode || text.is_empty() {
                    return;
                }
                let mut s = st_delta.borrow_mut();
                if continuous_usage {
                    s.usage_now = Some(produced);
                }
                if s.dead {
                    return;
                }
                s.decode.stamp(produced as u64);
                let r = if ch == channel::REASONING {
                    let mut ck = Frame::chat(Message::reasoning(text), None);
                    if let Some(n) = s.usage_now {
                        ck.usage = Some(Usage::partial(n));
                    }
                    send(ck)
                } else {
                    push_text(&mut s, text, None)
                };
                if r.is_err() {
                    cancels.cancel(session);
                    s.dead = true;
                }
            };
            let st_ev = std::rc::Rc::clone(&st);
            let on_event = |ev: &crate::CommittedEvent| {
                match &ev.body {
                    EventBody::Generated {
                        text,
                        channel: ch,
                        span,
                        ..
                    } => {
                        let mut s = st_ev.borrow_mut();
                        let covered_by_delta = delta_mode
                            && (*ch == channel::TEXT || *ch == channel::REASONING)
                            && !text.is_empty();
                        s.committed += span.len() as u32;
                        let committed = s.committed as u64;
                        if !covered_by_delta && !span.is_empty() {
                            s.decode.stamp(committed);
                        }
                        if continuous_usage && !delta_mode {
                            s.usage_now = Some(s.committed);
                        }
                        let these: Vec<crate::scheduler::TokenLogprob> = if want_logprobs {
                            let mut q = lp_q.borrow_mut();
                            (0..span.len()).filter_map(|_| q.pop_front()).collect()
                        } else {
                            Vec::new()
                        };
                        if *ch == channel::TOOL_CALL {
                            for d in frag.feed(text) {
                                if d.opening {
                                    open_call_id = format!("call_{}", ev.event_id);
                                }
                                let json = Message::tool(&d, tool_index, &open_call_id);
                                send(Frame::chat(json, None))?;
                            }
                            return Ok(());
                        }
                        if text.is_empty() {
                            return Ok(());
                        }
                        if delta_mode {
                            return Ok(());
                        }
                        if *ch == channel::REASONING {
                            let mut ck = Frame::chat(Message::reasoning(text), None);
                            if let Some(n) = s.usage_now {
                                ck.usage = Some(Usage::partial(n));
                            }
                            return send(ck);
                        }
                        let lp = want_logprobs.then(|| decode_logprobs(&daemon, &these, 0));
                        push_text(&mut s, text, lp)
                    }
                    EventBody::ToolParseFailure { raw } => {
                        if choice != ToolChoice::None && !frag.opened() {
                            if let Some((name, arguments)) = daemon.named_tool_call(raw) {
                                let delta = Message::complete_tool(
                                    tool_index,
                                    format!("call_{}", ev.event_id),
                                    name,
                                    arguments,
                                );
                                tool_index += 1;
                                orphaned_calls += 1;
                                return send(Frame::chat(delta, None));
                            }
                        }
                        if frag.opened() {
                            orphaned_calls += 1;
                            tool_index += 1;
                        }
                        frag.reset();
                        let mut ck = Frame::chat(Message::content(raw), None);
                        if continuous_usage {
                            ck.usage = Some(Usage::partial(st_ev.borrow().committed));
                        }
                        send(ck)
                    }
                    EventBody::ToolUse { name, arguments } => {
                        if choice == ToolChoice::None {
                            let raw = format!(
                                "{{\"name\":{},\"arguments\":{}}}",
                                serde_json::Value::String(name.clone()),
                                arguments
                            );
                            let mut ck = Frame::chat(Message::content(raw), None);
                            if continuous_usage {
                                ck.usage = Some(Usage::partial(st_ev.borrow().committed));
                            }
                            return send(ck);
                        }
                        if frag.opened() {
                            for d in frag.finish(arguments) {
                                send(Frame::chat(
                                    Message::tool(&d, tool_index, &open_call_id),
                                    None,
                                ))?;
                            }
                            frag.reset();
                            tool_index += 1;
                            return Ok(());
                        }
                        let delta = Message::complete_tool(
                            tool_index,
                            format!("call_{}", ev.event_id),
                            name.clone(),
                            arguments.clone(),
                        );
                        tool_index += 1;
                        send(Frame::chat(delta, None))
                    }
                    _ => Ok(()),
                }
            };
            let on_logprobs = |items: &[crate::scheduler::TokenLogprob]| {
                lp_fill.borrow_mut().extend(items.iter().cloned());
            };
            let out = if delta_mode {
                daemon.generate_streaming_delta(
                    session,
                    max_tokens,
                    extras,
                    on_event,
                    on_logprobs,
                    on_delta,
                )?
            } else {
                daemon.generate_streaming_ex_lp(
                    session,
                    max_tokens,
                    extras,
                    on_event,
                    on_logprobs,
                )?
            };
            let st = st.borrow();
            let (mut decode, stopped) = (st.decode, st.stopped);
            let first_tok = decode.first_token();
            decode.finish();
            let (content_acc, content_sent) = (st.content_acc.clone(), st.content_sent);
            drop(st);
            let mut c = collect(&out.events);
            if !stopped && !choice.forces_call() && content_sent < content_acc.len() {
                let rest = content_acc[content_sent..].to_string();
                let mut ck = Frame::chat(Message::content(rest), None);
                if continuous_usage {
                    ck.usage = Some(Usage::partial(out.tokens_generated));
                }
                send(ck)?;
            }
            if choice.forces_call() {
                c.content = content_acc.clone();
                promote_forced_tool_call(&daemon, &mut c, session, tool_schemas.as_deref());
                if tool_index > 0 {
                } else if let Some((id, name, args)) = c.tool_calls.first() {
                    send(Frame::chat(
                        Message::complete_tool(
                            tool_index,
                            format!("call_{id}"),
                            name.clone(),
                            args.clone(),
                        ),
                        None,
                    ))?;
                } else if !c.content.is_empty() {
                    let mut ck = Frame::chat(Message::content(c.content), None);
                    if continuous_usage {
                        ck.usage = Some(Usage::partial(out.tokens_generated));
                    }
                    send(ck)?;
                }
            }
            let fin = if stopped {
                "stop"
            } else {
                finish_reason(
                    out.finish,
                    out.tokens_generated,
                    max_tokens,
                    !c.tool_calls.is_empty() && choice != ToolChoice::None,
                )
            };
            let fin = if orphaned_calls > 0 { "length" } else { fin };
            let mut terminal = Frame::chat(Message::default(), Some(fin));
            terminal.usage = Some(usage(
                &daemon,
                session,
                out.tokens_generated,
                out.warm_prefix,
            ));
            access_log(
                &format!("{} (stream)", req.route()),
                &model_name,
                terminal.usage.as_ref().and_then(|u| u.prompt).unwrap_or(0),
                out.warm_prefix,
                out.tokens_generated as u64,
                GenTiming {
                    t0,
                    first: first_tok,
                    decode,
                },
                fin,
            );
            terminal.warm = Some(out.warm_prefix);
            terminal.spec = (out.spec.0 > 0).then_some(out.spec);
            terminal.session = Some(session);
            send(terminal)
        };
        run()
    });

    Ok(Body::Stream(rx))
}
const FIM_DEFAULT_MAX_TOKENS: u32 = 128;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn fim_completions(
    daemon: Arc<Daemon>,
    model_name: String,
    req: CompletionRequest,
    limits: TokenLimits,
    sampling: SamplingOverrides,
    mode: u8,
    key: Option<crate::keypolicy::ClientKey>,
) -> Result<Generation, RunError> {
    if !daemon.fim_supported() {
        return Err(DaemonError::NoFimDialect.into());
    }
    let hold = key;
    let max_tokens = req
        .max_tokens
        .or(limits.default_max_tokens)
        .unwrap_or(FIM_DEFAULT_MAX_TOKENS)
        .max(1);
    let serving = daemon
        .fim_satellite()
        .unwrap_or_else(|| Arc::clone(&daemon));
    let model_defaults = {
        let d = Arc::clone(&serving);
        tokio::task::spawn_blocking(move || {
            crate::with_sync_class(Some(crate::qos::INLINE_COMPLETION), || {
                d.model_sampling_defaults()
            })
        })
        .await
        .unwrap_or_default()
    };
    let op = sampling.for_daemon(&serving);
    let mut params = params_of_for(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.seed,
        &model_defaults,
        &op,
    );
    let seed_drawn = draw_seed_if_unseeded(&mut params, req.seed);
    let mut extras = extras_of(
        req.presence_penalty,
        req.frequency_penalty,
        req.repeat_penalty,
        &op,
        &model_defaults,
    );
    extras.seed_drawn = seed_drawn;
    extras.ignore_eos = req.ignore_eos;
    let fim = crate::FimRequest {
        prefix: req.prompt,
        suffix: req.suffix.unwrap_or_default(),
        mode,
        max_tokens,
        params,
        extras,
    };
    let id = format!(
        "cmpl-superfluid-{}",
        next_id().rsplit('-').next().unwrap_or("0")
    );
    let created = unix_now();
    let served_by = daemon
        .fim_satellite_name()
        .unwrap_or_else(|| model_name.clone());
    let log_model = served_by.clone();
    let fin_of = move |c: &crate::Completion| {
        if c.expired {
            "length"
        } else {
            finish_reason(c.finish, c.tokens.len() as u32, max_tokens, false)
        }
    };
    let ext = move |c: &crate::Completion| FimMetadata {
        cached: c.cached,
        expired: c.expired,
        served_by: served_by.clone(),
    };
    if req.stream {
        let admitted_daemon = Arc::clone(&daemon);
        let fim = blocking(
            RequestQos {
                class: crate::qos::INLINE_COMPLETION,
                batch_invariant: false,
            },
            hold.clone(),
            move || {
                admitted_daemon.fim_fits(&fim)?;
                Ok(fim)
            },
        )
        .await?;
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::task::spawn_blocking(move || {
            let _hold = hold;
            let t0 = std::time::Instant::now();
            let mut decode = DecodeSpan::default();
            let mut delivered = 0usize;
            let res = daemon.complete_streaming(&fim, |piece, produced| {
                decode.stamp(produced as u64);
                delivered = delivered.max(produced);
                match tx.blocking_send(Ok(Frame::text(piece, None, None))) {
                    Ok(()) => std::ops::ControlFlow::Continue(()),
                    Err(_) => std::ops::ControlFlow::Break(()),
                }
            });
            match res {
                Ok(c) => {
                    let fin = fin_of(&c);
                    let first = decode.first_token();
                    decode.finish();
                    access_log(
                        "POST /v1/completions (fim)",
                        &log_model,
                        c.prompt_tokens as u64,
                        fim_log_cached(&c),
                        c.tokens.len().max(delivered) as u64,
                        GenTiming { t0, first, decode },
                        fin,
                    );
                    let mut last = Frame::text("", None, Some(fin));
                    last.fim = Some(ext(&c));
                    last.usage = Some(Usage::complete(
                        c.prompt_tokens as u64,
                        c.tokens.len() as u64,
                        None,
                    ));
                    let _ = tx.blocking_send(Ok(last));
                }
                Err(e) => {
                    let client_error = matches!(
                        e,
                        DaemonError::StreamTooLong { .. } | DaemonError::NoFimDialect
                    );
                    let _ = tx.blocking_send(Err(StreamFailure::new(e, client_error)));
                }
            }
        });
        return Ok(Generation {
            kind: Kind::Completion,
            id,
            model: model_name,
            created,
            warm_header: None,
            body: Body::Stream(rx),
        });
    }
    let out = tokio::task::spawn_blocking(move || {
        let _hold = hold;
        let t0 = std::time::Instant::now();
        let mut decode = DecodeSpan::default();
        let c = daemon.complete_streaming(&fim, |_, produced| {
            decode.stamp(produced as u64);
            std::ops::ControlFlow::Continue(())
        })?;
        let first = decode.first_token();
        decode.finish();
        access_log(
            "POST /v1/completions (fim)",
            &log_model,
            c.prompt_tokens as u64,
            fim_log_cached(&c),
            c.tokens.len() as u64,
            GenTiming { t0, first, decode },
            fin_of(&c),
        );
        Ok::<_, DaemonError>(c)
    })
    .await;
    let c = out.map_err(|e| RunError::Worker(e.to_string()))??;
    let mut frame = Frame::text(&c.text, None, Some(fin_of(&c)));
    frame.usage = Some(Usage::complete(
        c.prompt_tokens as u64,
        c.tokens.len() as u64,
        None,
    ));
    frame.fim = Some(ext(&c));
    Ok(Generation {
        kind: Kind::Completion,
        id,
        model: model_name,
        created,
        warm_header: None,
        body: Body::Complete(frame),
    })
}

pub(crate) fn chat_once(
    daemon: &Daemon,
    req: &ChatRequest,
    policy: RequestPolicy,
) -> Result<Frame, DaemonError> {
    let RequestPolicy {
        limits,
        sampling: op,
        qos,
    } = policy;
    let _class = crate::SyncClassScope::enter(Some(qos.class));
    let model_defaults = daemon.model_sampling_defaults();
    let mut extras = extras_of(
        req.presence_penalty,
        req.frequency_penalty,
        req.repeat_penalty,
        &op,
        &model_defaults,
    );
    let choice = ToolChoice::parse(&req.tool_choice);
    if choice.forces_call() {
        extras.open_channel = crate::wal::channel::TOOL_CALL;
    }
    let tool_schemas = crate::codec::ToolSchemas::of_values(&req.tools);
    extras.tool_schemas = tool_schemas.clone();
    let handles = RequestHandles::new(
        daemon,
        grammar_handle_for(daemon, req, &choice)?,
        logit_bias_handle_of(daemon, &req.logit_bias),
    );
    handles.apply(&mut extras);
    let want_logprobs = req.logprobs.unwrap_or(false);
    extras.want_logprobs = want_logprobs;
    extras.top_logprobs = req.top_logprobs.unwrap_or(0).min(20) as u8;
    extras.thinking = req.thinking();
    extras.ignore_eos = req.ignore_eos;
    let stops = req.stop.list();
    let (session, seed_drawn) = build_session(daemon, req, &op, &model_defaults)?;
    extras.seed_drawn = seed_drawn;
    qos.apply(daemon, session)?;
    let max_tokens = limits.resolve(req.max_completion_tokens.or(req.max_tokens), || {
        prompt_len(daemon, session)
    });
    let cancels = daemon.cancel_registry();
    let mut acc = String::new();
    let mut stopped = false;
    let out = daemon.generate_streaming_ex(session, max_tokens, extras, |ev| {
        if let EventBody::Generated {
            text, channel: ch, ..
        } = &ev.body
        {
            if *ch == channel::TEXT && !stops.is_empty() && !stopped {
                acc.push_str(text);
                if stops.iter().any(|s| acc.contains(s.as_str())) {
                    cancels.cancel(session);
                    stopped = true;
                }
            }
        }
        Ok(())
    });
    let out = out?;
    let mut c = collect(&out.events);
    claim_named_calls(daemon, &mut c, &out.events, &choice);
    if choice.forces_call() {
        promote_forced_tool_call(daemon, &mut c, session, tool_schemas.as_deref());
    }
    let (content, hit_stop) = truncate_at_stop(&c.content, &stops);
    c.content = content;
    let usage = usage(daemon, session, out.tokens_generated, out.warm_prefix);
    let finish = if !c.broken_calls.is_empty() {
        "length"
    } else if hit_stop {
        "stop"
    } else {
        finish_reason(
            out.finish,
            out.tokens_generated,
            max_tokens,
            !c.tool_calls.is_empty() && choice != ToolChoice::None,
        )
    };
    Ok(Frame {
        choices: vec![Choice {
            message: message(&c, &choice),
            finish: Some(finish),
            logprobs: want_logprobs.then(|| decode_logprobs(daemon, &out.logprobs, 0)),
            ..Choice::default()
        }],
        usage: Some(usage),
        spec: (out.spec.0 > 0).then_some(out.spec),
        ..Frame::default()
    })
}

pub(crate) async fn blocking<T: Send + 'static>(
    qos: RequestQos,
    hold: Option<crate::keypolicy::ClientKey>,
    work: impl FnOnce() -> Result<T, DaemonError> + Send + 'static,
) -> Result<T, RunError> {
    tokio::task::spawn_blocking(move || {
        let _hold = hold;
        let _class = crate::SyncClassScope::enter(Some(qos.class));
        work()
    })
    .await
    .map_err(|e| RunError::Worker(e.to_string()))?
    .map_err(RunError::Daemon)
}

pub(crate) fn base64_decode(data: &str) -> Result<Vec<u8>, DaemonError> {
    use base64::Engine;
    let cleaned: String = data.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(&cleaned)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(&cleaned))
        .map_err(|_| DaemonError::Protocol("invalid base64 image data"))
}

pub(crate) fn spawn_stream<T: Send + 'static>(
    qos: RequestQos,
    hold: Option<crate::keypolicy::ClientKey>,
    work: impl FnOnce(&tokio::sync::mpsc::Sender<Result<T, StreamFailure>>) -> Result<(), DaemonError>
        + Send
        + 'static,
) -> tokio::sync::mpsc::Receiver<Result<T, StreamFailure>> {
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::task::spawn_blocking(move || {
        let _hold = hold;
        let _class = crate::SyncClassScope::enter(Some(qos.class));
        if let Err(error) = work(&tx) {
            let client_error = matches!(
                error,
                DaemonError::Constraint(_)
                    | DaemonError::TemplateRefused(_)
                    | DaemonError::Unsupported(_)
                    | DaemonError::StreamTooLong { .. }
                    | DaemonError::NoFimDialect
            );
            let _ = tx.blocking_send(Err(StreamFailure::new(error, client_error)));
        }
    });
    rx
}
