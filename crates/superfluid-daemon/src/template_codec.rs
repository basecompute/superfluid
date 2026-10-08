//! `TemplateCodec`.

use superfluid_abi::Status;
use minijinja::{context, Environment, Value};

use crate::codec::{
    learn_delims, Delims, BundleCodec, Channelizer, Segmenter, TextCodec, ToolEnvelope, S_K1, S_K2, S_NAME, S_V1, S_V2,
};
use crate::wal::channel;

enum ToolWire {
    Envelope(ToolEnvelope),
    Delimited(Delims),
    Raw,
}

impl ToolWire {
    fn learn(codec: &TemplateCodec, frame: Option<(&str, &str)>, gemma_frame: bool) -> ToolWire {
        if gemma_frame {
            return ToolWire::Envelope(ToolEnvelope::Gemma);
        }
        ToolWire::learn_from(
            |args: &str| -> Option<String> {
                let user = crate::codec::ChatMessage::new(crate::wal::role::USER, "hi");
                let mut asst = crate::codec::ChatMessage::new(crate::wal::role::ASSISTANT, "");
                asst.tool_calls.push(crate::codec::ToolCallMsg {
                    id: String::new(),
                    name: S_NAME.to_string(),
                    arguments: args.to_string(),
                });
                let msgs = vec![codec.message_value(&user), codec.message_value(&asst)];
                codec.try_render_values(msgs, &[], false, &serde_json::Map::new()).ok()
            },
            frame,
        )
    }

    fn learn_from(render: impl Fn(&str) -> Option<String>, frame: Option<(&str, &str)>) -> ToolWire {
        let two = match render(&format!("{{\"{S_K1}\":\"{S_V1}\",\"{S_K2}\":\"{S_V2}\"}}")) {
            Some(t) => t,
            None => return ToolWire::Raw,
        };
        if two.contains(&format!("\"{S_K1}\"")) && two.contains(&format!("\"{S_NAME}\"")) {
            return ToolWire::Envelope(ToolEnvelope::Json);
        }
        let packed: String = two.chars().filter(|c| !c.is_whitespace()).collect();
        if packed.contains(&format!("{S_NAME}<arg_key>{S_K1}</arg_key><arg_value>{S_V1}</arg_value>")) {
            return ToolWire::Envelope(ToolEnvelope::Glm);
        }
        let zero = render("{}");
        match learn_delims(&two, zero.as_deref(), frame) {
            Some(d) => ToolWire::Delimited(d),
            None => ToolWire::Raw,
        }
    }

    pub(crate) fn parse(&self, raw: &str, schemas: Option<&crate::codec::ToolSchemas>) -> Option<(String, String)> {
        match self {
            ToolWire::Envelope(e) => e.parse_call(raw, schemas),
            ToolWire::Delimited(d) => d.parse(raw, schemas).or_else(|| crate::codec::parse_json_tool_call(raw)),
            ToolWire::Raw => crate::codec::parse_json_tool_call(raw),
        }
    }
}

pub struct TemplateCodec {
    inner: BundleCodec,
    env: Environment<'static>,
    bos: String,
    eos: String,
    markers: Vec<(u32, u32, u32)>,
    channel_names: Vec<(u32, Vec<u32>)>,
    terminators: Vec<u32>,
    tool_wire: ToolWire,
    verified: bool,
    template_digest: u64,
    mentions_reasoning_effort: bool,
    harmony: Option<crate::harmony::HarmonyIds>,
    effort_ladders: Vec<(EffortContext, Option<Vec<&'static str>>)>,
    thinking_switch: bool,
}

impl TemplateCodec {
        #[cfg(feature = "basert")]
    pub fn load(model_path: &std::path::Path) -> Result<TemplateCodec, Status> {
        let tok = superfluid_engine_ffi::TokenizerHandle::load(model_path).map_err(|_| Status::Fatal)?;
        Self::from_tokenizer(std::sync::Arc::new(tok))
    }

    pub fn from_tokenizer(
        tok: std::sync::Arc<dyn superfluid_engine::Tokenizer>,
    ) -> Result<TemplateCodec, Status> {
        let inner = BundleCodec::from_tokenizer(tok);
        let tok = inner.tokenizer();
        let source = tok.chat_template_jinja();
        if source.trim().is_empty() {
            return Err(Status::Unsupported);
        }
        let tool_source = tok.chat_template_named("tool_use").filter(|t| !t.trim().is_empty());
        let specials = tok.special_tokens();

        let by_id = |id: u32| specials.iter().find(|(_, i)| *i == id).map(|(s, _)| s.clone());
        let eos = by_id(tok.eos_token()).unwrap_or_default();
        let bos = tok.bos_token().and_then(by_id).unwrap_or_default();

        let harmony = crate::harmony::HarmonyIds::discover(&specials).filter(|_| source.contains("<|channel|>"));

        let find = |name: &str| specials.iter().find(|(s, _)| s == name).map(|(_, i)| *i);
        let mut markers = Vec::new();
        if let (Some(o), Some(c)) = (find("<think>"), find("</think>")) {
            markers.push((o, c, channel::REASONING));
        }
        let mut channel_names = Vec::new();
        if source.contains("<|channel>thought") {
            if let (Some(o), Some(c)) = (find("<|channel>"), find("<channel|>")) {
                markers.push((o, c, channel::REASONING));
                let mut name = tok.encode("thought\n");
                if tok.bos_token().is_some_and(|bos| name.first() == Some(&bos)) {
                    name.remove(0);
                }
                channel_names.push((o, name));
            }
        }
        let mut gemma_frame = false;
        if let (Some(o), Some(c)) = (find("<tool_call>"), find("</tool_call>")) {
            markers.push((o, c, channel::TOOL_CALL));
        } else if let (Some(o), Some(c)) = (
            find(crate::codec::GEMMA_TOOL_OPEN),
            find(crate::codec::GEMMA_TOOL_CLOSE),
        ) {
            markers.push((o, c, channel::TOOL_CALL));
            gemma_frame = true;
        }

        let mut terminators = Vec::new();
        if let Some(h) = harmony {
            terminators.push(h.ret);
            terminators.push(h.call);
        }
        if let (Some(o), Some(c)) = (
            find(crate::codec::GEMMA_TOOL_RESPONSE_OPEN),
            find(crate::codec::GEMMA_TOOL_RESPONSE_CLOSE),
        ) {
            markers.push((o, c, channel::REASONING));
            terminators.push(o);
        }

        let pinned_date = today_utc();
        let mut template_digest = superfluid_fingerprint::fnv1a64(source.as_bytes(), 0xCBF2_9CE4_8422_2325);
        if let Some(t) = &tool_source {
            template_digest = superfluid_fingerprint::fnv1a64(t.as_bytes(), template_digest);
        }
        let mentions_reasoning_effort = source.contains("reasoning_effort")
            || tool_source.as_ref().is_some_and(|t| t.contains("reasoning_effort"));
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |m: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m))
        });
        env.add_function("strftime_now", move |_fmt: String| pinned_date.clone());
        env.add_filter("tojson", hf_tojson);
        env.add_template_owned("chat", source)
            .map_err(|_| Status::Unsupported)?;
        if let Some(t) = tool_source {
            env.add_template_owned("chat_tools", t)
                .map_err(|_| Status::Unsupported)?;
        }

        let frame = markers
            .iter()
            .find(|(_, _, ch)| *ch == channel::TOOL_CALL)
            .map(|&(o, c, _)| (inner.marker_literal(o), inner.marker_literal(c)));

        let mut codec = TemplateCodec {
            inner,
            env,
            bos,
            eos,
            markers,
            channel_names,
            terminators,
            tool_wire: ToolWire::Raw,
            verified: false,
            template_digest,
            mentions_reasoning_effort,
            harmony,
            effort_ladders: Vec::new(),
            thinking_switch: false,
        };
        codec.tool_wire = if codec.harmony.is_some() {
            ToolWire::Envelope(ToolEnvelope::Harmony)
        } else {
            ToolWire::learn(&codec, frame.as_ref().map(|(o, c)| (o.as_str(), c.as_str())), gemma_frame)
        };
        match codec.try_render(&[("user", "probe")], &[], true) {
            Ok(t) if !t.trim().is_empty() => {}
            _ => return Err(Status::Unsupported),
        }
        codec.thinking_switch = codec.probe_thinking_switch();
        if codec.supports_reasoning_effort() {
            let ladders = learn_effort_ladders(|msgs, tools, generation, kw| {
                let msgs = msgs.iter().map(|m| codec.message_value(m)).collect();
                codec.try_render_values(msgs, tools, generation, kw)
            });
            if ladders.iter().any(|(_, l)| l.as_ref().is_none_or(|l| !l.is_empty())) {
                tracing::info!("TemplateCodec: reasoning_effort levels by request shape {:?}", ladders);
            }
            codec.effort_ladders = ladders;
        }
        codec.verified = codec.validate_append_stability();
        if !codec.verified {
            tracing::info!(
                "TemplateCodec: chat template is not append-stable — each turn renders the whole conversation afresh"
            );
        }
        Ok(codec)
    }

    fn probe_thinking_switch(&self) -> bool {
        crate::codec::thinking_switch_changes(|msgs, tools, on| {
            let mut kw = serde_json::Map::new();
            kw.insert("enable_thinking".into(), serde_json::Value::Bool(on));
            let msgs = msgs.iter().map(|m| self.message_value(m)).collect();
            self.try_render_values(msgs, tools, true, &kw).ok()
        })
    }

    pub fn verified(&self) -> bool {
        self.verified
    }

    pub fn tool_wire_kind(&self) -> &'static str {
        match &self.tool_wire {
            ToolWire::Envelope(e) => e.name(),
            ToolWire::Delimited(_) => "delimited",
            ToolWire::Raw => "raw",
        }
    }

    pub fn tool_body_is_json(&self) -> bool {
        matches!(self.tool_wire, ToolWire::Envelope(ToolEnvelope::Json))
    }

    pub fn template_digest(&self) -> u64 {
        self.template_digest
    }

    pub fn render_identity(&self) -> String {
        let ladders: Vec<Option<&[&str]>> = self.effort_ladders.iter().map(|(_, l)| l.as_deref()).collect();
        render_identity_of(self.template_digest, &ladders)
    }

    pub fn render_turns_structured(
        &self,
        messages: &[crate::codec::ChatMessage],
        tools: &[String],
    ) -> Option<Vec<u32>> {
        self.render_turns_structured_with(messages, tools, &serde_json::Map::new())
    }

    pub fn render_turns_structured_with(
        &self,
        messages: &[crate::codec::ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        self.try_render_turns_structured_with(messages, tools, kwargs).ok()
    }

    pub fn try_render_turns_structured_with(
        &self,
        messages: &[crate::codec::ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Vec<u32>, String> {
        self.try_render_prompt_protected(messages, tools, kwargs, false)
            .map_err(|e| format!("{e:#}"))
    }

    fn message_value(&self, m: &crate::codec::ChatMessage) -> Value {
        self.message_value_with(m, &mut None)
    }

    fn message_value_with(
        &self,
        m: &crate::codec::ChatMessage,
        subs: &mut Option<Vec<(String, String)>>,
    ) -> Value {
        fn stand_in(subs: &mut Option<Vec<(String, String)>>, text: &str) -> String {
            match subs {
                Some(s) => {
                    let key = crate::codec::stand_in_key(s.len());
                    s.push((key.clone(), text.to_string()));
                    key
                }
                None => text.to_string(),
            }
        }
        let protect = m.role != crate::wal::role::ASSISTANT;
        let role = role_name(m.role);
        let mut fields: Vec<(&str, Value)> = vec![("role", Value::from(role))];
        if !m.parts.is_empty() {
            let parts: Vec<serde_json::Value> = m
                .parts
                .iter()
                .map(|p| match p {
                    crate::codec::ContentPart::Text(t) => {
                        let text = if protect { stand_in(subs, t) } else { t.clone() };
                        serde_json::json!({"type": "text", "text": text})
                    }
                    crate::codec::ContentPart::Image { .. } => serde_json::json!({"type": "image"}),
                    crate::codec::ContentPart::Audio { .. } => serde_json::json!({"type": "audio"}),
                })
                .collect();
            fields.push(("content", Value::from_serialize(&parts)));
        } else {
            let content = if protect { stand_in(subs, &m.content) } else { m.content.clone() };
            fields.push(("content", Value::from(content)));
        }
        if let Some(r) = &m.reasoning {
            fields.push(("reasoning_content", Value::from(r.clone())));
            fields.push(("thinking", Value::from(r.clone())));
        }
        let calls: Vec<Value> = m
            .tool_calls
            .iter()
            .map(|c| {
                let args = match serde_json::from_str::<serde_json::Value>(&c.arguments) {
                    Ok(serde_json::Value::Object(o)) => Value::from_serialize(&o),
                    Ok(serde_json::Value::Null) => Value::from_serialize(serde_json::Map::new()),
                    Ok(other) => Value::from_serialize(serde_json::json!({ "value": other })),
                    Err(_) if c.arguments.trim().is_empty() => {
                        Value::from_serialize(serde_json::Map::new())
                    }
                    Err(_) => Value::from_serialize(serde_json::json!({ "value": c.arguments })),
                };
                Value::from_iter([
                    ("id", Value::from(c.id.clone())),
                    ("type", Value::from("function")),
                    ("function", context! { name => c.name.clone(), arguments => args }),
                ])
            })
            .collect();
        if !calls.is_empty() {
            fields.push(("tool_calls", Value::from(calls)));
        }
        if let Some(id) = &m.tool_call_id {
            fields.push(("tool_call_id", Value::from(id.clone())));
        }
        if let Some(name) = &m.name {
            fields.push(("name", Value::from(name.clone())));
        }
        Value::from_iter(fields)
    }

    fn try_render_prompt_protected(
        &self,
        messages: &[crate::codec::ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, minijinja::Error> {
        let reference = {
            let msgs: Vec<Value> = messages.iter().map(|m| self.message_value(m)).collect();
            self.try_render_values(msgs, tools, add_generation_prompt, kwargs)?
        };
        let mut subs = Some(Vec::new());
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| self.message_value_with(m, &mut subs))
            .collect();
        let subs = subs.unwrap_or_default();
        let with_stand_ins = self.try_render_values(msgs, tools, add_generation_prompt, kwargs)?;
        match self.encode_rendered(&with_stand_ins, &subs, &reference) {
            Some(toks) => Ok(toks),
            None => {
                tracing::debug!(
                    "chat template does not carry every message verbatim; \
                     the prompt is encoded as framing"
                );
                Ok(self.encode_framing(&reference))
            }
        }
    }

    fn encode_rendered(
        &self,
        text: &str,
        subs: &[(String, String)],
        reference: &str,
    ) -> Option<Vec<u32>> {
        let owned =
            crate::codec::split_around_stand_ins(text, subs, reference, crate::codec::Placement::KnownFormsOnly)?;
        let pieces: Vec<(&str, bool)> = owned.iter().map(|(t, c)| (t.as_str(), *c)).collect();
        Some(self.inner.encode_pieces(&pieces))
    }

    fn try_render_values(
        &self,
        msgs: Vec<Value>,
        tools: &[String],
        add_generation_prompt: bool,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, minijinja::Error> {
        let tool_vals: Vec<Value> = tools
            .iter()
            .filter_map(|t| serde_json::from_str::<serde_json::Value>(t).ok())
            .map(|j| Value::from_serialize(&j))
            .collect();
        let t = if !tool_vals.is_empty() {
            self.env.get_template("chat_tools").or_else(|_| self.env.get_template("chat"))?
        } else {
            self.env.get_template("chat")?
        };
        let remapped;
        let ladder = EffortContext::of(&msgs, tools, add_generation_prompt, kwargs).best(&self.effort_ladders);
        let kwargs = match kwargs
            .get("reasoning_effort")
            .and_then(|v| v.as_str())
            .and_then(|r| nearest_effort(ladder, r).map(|to| (r, to)))
        {
            Some((from, to)) => {
                tracing::debug!("reasoning_effort {from:?} -> {to:?} (this request shape's levels: {ladder:?})");
                let mut m = kwargs.clone();
                m.insert("reasoning_effort".into(), serde_json::Value::String(to.into()));
                remapped = m;
                &remapped
            }
            None => kwargs,
        };
        let extra = Value::from_serialize(kwargs);
        t.render(context! {
            messages => Value::from(msgs),
            tools => if tool_vals.is_empty() { Value::from(()) } else { Value::from(tool_vals) },
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos.clone(),
            eos_token => self.eos.clone(),
            ..extra
        })
    }

    fn try_render(
        &self,
        messages: &[(&str, &str)],
        tools: &[String],
        add_generation_prompt: bool,
    ) -> Result<String, minijinja::Error> {
        let msgs: Vec<Value> = messages
            .iter()
            .map(|(role, content)| context! { role => *role, content => *content })
            .collect();
        self.try_render_values(msgs, tools, add_generation_prompt, &serde_json::Map::new())
    }

    pub fn render_conversation(
        &self,
        messages: &[(&str, &str)],
        tools: &[String],
        add_generation_prompt: bool,
    ) -> String {
        self.try_render(messages, tools, add_generation_prompt).unwrap_or_default()
    }

    pub fn render_conversation_tokens(
        &self,
        messages: &[(&str, &str)],
        tools: &[String],
        add_generation_prompt: bool,
    ) -> Option<Vec<u32>> {
        let text = self.try_render(messages, tools, add_generation_prompt).ok()?;
        Some(self.encode_framing(&text))
    }

    fn validate_append_stability(&self) -> bool {
        let convo = [
            ("system", "You are a helpful assistant."),
            ("user", "First question about geography."),
            ("assistant", "A helpful answer."),
            ("user", "A follow-up question."),
        ];
        let mut prev: Vec<u32> = Vec::new();
        for n in 1..=convo.len() {
            let Ok(text) = self.try_render(&convo[..n], &[], false) else {
                return false;
            };
            let toks = self.encode_framing(&text);
            if !toks.starts_with(&prev) {
                return false;
            }
            prev = toks;
        }
        true
    }
}

impl TemplateCodec {
    fn encode_framing(&self, text: &str) -> Vec<u32> {
        self.inner.encode_pieces(&[(text, false)])
    }
}

impl TextCodec for TemplateCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }
    fn encode_content(&self, text: &str) -> Vec<u32> {
        self.inner.encode_content(text)
    }
    fn encode_pieces(&self, pieces: &[(&str, bool)]) -> Vec<u32> {
        self.inner.encode_pieces(pieces)
    }
    fn chat_template_source(&self) -> Option<String> {
        self.inner.chat_template_source()
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        self.inner.token_bytes(token)
    }
    fn marker_literal(&self, token: u32) -> String {
        self.inner.marker_literal(token)
    }

    fn render_fim(&self, prefix: &str, suffix: &str, mode: u8) -> Option<Vec<u32>> {
        self.inner.render_fim(prefix, suffix, mode)
    }
    fn behavior_fingerprint(&self) -> Option<superfluid_fingerprint::BehaviorFingerprint> {
        Some(self.inner.fingerprint(&format!("template/{}", self.render_identity()), &self.markers))
    }

    fn renders_per_message(&self) -> bool {
        false
    }

    fn render_prompt_structured(
        &self,
        messages: &[crate::codec::ChatMessage],
        tools: &[String],
    ) -> Option<Vec<u32>> {
        self.try_render_prompt_protected(messages, tools, &serde_json::Map::new(), true)
            .ok()
    }

    fn supports_reasoning_effort(&self) -> bool {
        if self.mentions_reasoning_effort {
            return true;
        }
        let probe = [
            crate::codec::ChatMessage::new(crate::wal::role::SYSTEM, ""),
            crate::codec::ChatMessage::new(crate::wal::role::USER, "x"),
        ];
        let Some(plain) = self.render_turns_structured(&probe, &[]) else {
            return false;
        };
        let mut kw = serde_json::Map::new();
        kw.insert(
            "reasoning_effort".into(),
            serde_json::Value::String("__probe__".into()),
        );
        match self.try_render_turns_structured_with(&probe, &[], &kw) {
            Err(_) => true,
            Ok(toks) => toks != plain,
        }
    }

    fn supports_enable_thinking(&self) -> bool {
        self.thinking_switch
    }

    fn reasoning_effort_levels(&self) -> Vec<&'static str> {
        common_levels(self.effort_ladders.iter().map(|(_, l)| l.as_deref()))
    }

    fn validate_template_kwargs(
        &self,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        if kwargs.is_empty() {
            return Ok(());
        }
        let probe = [
            crate::codec::ChatMessage::new(crate::wal::role::SYSTEM, ""),
            crate::codec::ChatMessage::new(crate::wal::role::USER, "x"),
        ];
        if self
            .try_render_turns_structured_with(&probe, &[], &serde_json::Map::new())
            .is_err()
        {
            return Ok(());
        }
        self.try_render_turns_structured_with(&probe, &[], kwargs)
            .map(|_| ())
    }

    fn render_prompt_structured_with(
        &self,
        messages: &[crate::codec::ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<Vec<u32>> {
        match self.try_render_prompt_protected(messages, tools, kwargs, true) {
            Ok(toks) => Some(toks),
            Err(e) => {
                tracing::warn!("chat template render failed: {e:#}");
                None
            }
        }
    }

    fn trailing_generation_prompt(
        &self,
        messages: &[crate::codec::ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
        span: &[u32],
    ) -> usize {
        let Ok(without) = self.try_render_prompt_protected(messages, tools, kwargs, false) else {
            return 0;
        };
        if without.is_empty() || span.len() <= without.len() {
            return 0;
        }
        if span.starts_with(&without) {
            return span.len() - without.len();
        }
        let bytes_of = |toks: &[u32]| -> Vec<u8> { toks.iter().flat_map(|&t| self.inner.token_bytes(t)).collect() };
        let without_bytes = bytes_of(&without);
        if !bytes_of(span).starts_with(&without_bytes) {
            tracing::debug!(
                "generation prompt not derived: the render without it is no text prefix of the render with it"
            );
            return 0;
        }
        let boundary = without_bytes.len();
        let mut cum = 0usize;
        for (k, &t) in span.iter().enumerate() {
            if cum >= boundary {
                return span.len() - k;
            }
            cum += self.inner.token_bytes(t).len();
        }
        0
    }

    fn template_refusal(
        &self,
        messages: &[crate::codec::ChatMessage],
        tools: &[String],
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Option<String> {
        let msgs: Vec<Value> = messages.iter().map(|m| self.message_value(m)).collect();
        let e = self.try_render_values(msgs, tools, true, kwargs).err()?;
        Some(e.detail().map(str::to_string).unwrap_or_else(|| e.to_string()))
    }

    fn render_prompt(&self, messages: &[(u32, String)], tools: &[String]) -> Option<Vec<u32>> {
        let msgs: Vec<crate::codec::ChatMessage> = messages
            .iter()
            .map(|(role, text)| crate::codec::ChatMessage::new(*role, text.clone()))
            .collect();
        self.try_render_prompt_protected(&msgs, tools, &serde_json::Map::new(), true)
            .ok()
    }

    fn channel_markers(&self) -> Vec<(u32, u32, u32)> {
        self.markers.clone()
    }

    fn channelizer(&self) -> Box<dyn Channelizer> {
        match self.harmony {
            Some(ids) => Box::new(crate::harmony::HarmonyChannelizer::new(self.inner.tokenizer(), ids)),
            None => Box::new(Segmenter::new(self.markers.clone()).with_names(self.channel_names.clone())),
        }
    }

    fn parse_tool_call_with(
        &self,
        raw: &str,
        schemas: Option<&crate::codec::ToolSchemas>,
    ) -> Option<(String, String)> {
        self.tool_wire.parse(raw, schemas)
    }

    fn named_tool_call(&self, raw: &str) -> Option<(String, String)> {
        match &self.tool_wire {
            ToolWire::Envelope(e) => e.named_call(raw),
            ToolWire::Delimited(_) | ToolWire::Raw => None,
        }
    }

    fn call_grammar(&self, tool_jsons: &[String]) -> Option<String> {
        match &self.tool_wire {
            ToolWire::Envelope(e) => e.call_grammar(tool_jsons),
            ToolWire::Delimited(_) | ToolWire::Raw => None,
        }
    }

    fn structural_tag(&self, tool_jsons: &[String], at_least_one: bool) -> Option<String> {
        if let Some(ids) = self.harmony {
            return if at_least_one {
                crate::harmony::forced_call_tag(ids, tool_jsons)
            } else {
                crate::harmony::auto_call_tag(ids, tool_jsons)
            };
        }
        match &self.tool_wire {
            ToolWire::Envelope(e) => {
                let (begin, end) = self.tool_call_delimiters()?;
                e.structural_tag(tool_jsons, &begin, &end, at_least_one)
            }
            ToolWire::Delimited(d) => {
                let (begin, end) = self.tool_call_delimiters()?;
                d.structural_tag(tool_jsons, &begin, &end, at_least_one)
            }
            ToolWire::Raw => None,
        }
    }

    fn response_format_tag(&self, json_schema: &str) -> Option<String> {
        crate::harmony::final_json_tag(self.harmony?, json_schema)
    }

    fn turn_terminators(&self) -> Vec<u32> {
        self.terminators.clone()
    }

    fn tool_envelope(&self) -> crate::codec::ToolEnvelope {
        match &self.tool_wire {
            ToolWire::Envelope(e) => *e,
            ToolWire::Delimited(_) | ToolWire::Raw => crate::codec::ToolEnvelope::Json,
        }
    }

    fn tool_fragment_mode(&self) -> crate::tool_fragment::ToolFragmentMode {
        use crate::tool_fragment::ToolFragmentMode;
        match &self.tool_wire {
            ToolWire::Delimited(d) => ToolFragmentMode::Delimited(d.clone()),
            ToolWire::Envelope(e) => match e {
                crate::codec::ToolEnvelope::Json => ToolFragmentMode::Json,
                _ => ToolFragmentMode::None,
            },
            ToolWire::Raw => ToolFragmentMode::Json,
        }
    }
}

fn role_name(role: u32) -> &'static str {
    match role {
        crate::wal::role::SYSTEM => "system",
        crate::wal::role::ASSISTANT => "assistant",
        crate::wal::role::TOOL => "tool",
        _ => "user",
    }
}

fn today_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    civil_date(secs.div_euclid(86_400))
}

fn civil_date(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

const EFFORT_SCALE: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

const EFFORT_NORMALIZATION_VERSION: u32 = 1;

fn render_identity_of(digest: u64, ladders: &[Option<&[&str]>]) -> String {
    if ladders.iter().all(|l| l.is_some_and(|l| l.is_empty())) {
        return format!("{digest:016x}");
    }
    let mut parts: Vec<String> = ladders.iter().map(|l| l.map_or_else(|| "!".to_string(), |l| l.join(","))).collect();
    parts.dedup();
    format!("{digest:016x}/effort{EFFORT_NORMALIZATION_VERSION}:{}", parts.join("|"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EffortContext {
    generation: bool,
    system: bool,
    tools: bool,
    thinking: Option<bool>,
}

impl EffortContext {
    fn of(
        msgs: &[Value],
        tools: &[String],
        generation: bool,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> EffortContext {
        EffortContext {
            generation,
            system: msgs.iter().any(|m| m.get_attr("role").ok().and_then(|r| r.as_str().map(|r| r == "system")) == Some(true)),
            tools: !tools.is_empty(),
            thinking: kwargs.get("enable_thinking").and_then(|v| v.as_bool()),
        }
    }

    fn best<'a>(&self, ladders: &'a [(EffortContext, Option<Vec<&'static str>>)]) -> &'a [&'static str] {
        ladders
            .iter()
            .rev()
            .max_by_key(|(c, _)| {
                (c.generation == self.generation, c.tools == self.tools, c.system == self.system, c.thinking == self.thinking)
            })
            .and_then(|(_, l)| l.as_deref())
            .unwrap_or(&[])
    }
}

#[allow(clippy::type_complexity)]
fn learn_effort_ladders<E>(
    render: impl Fn(&[crate::codec::ChatMessage], &[String], bool, &serde_json::Map<String, serde_json::Value>) -> Result<String, E>,
) -> Vec<(EffortContext, Option<Vec<&'static str>>)> {
    let mut shapes = crate::codec::thinking_probe_contexts();
    if let Some(tools) = shapes.iter().map(|(_, t)| t.clone()).find(|t| !t.is_empty()) {
        shapes.push((vec![crate::codec::ChatMessage::new(crate::wal::role::USER, "probe")], tools));
    }
    let mut ladders = Vec::new();
    for (msgs, tools) in &shapes {
        for generation in [true, false] {
            for on in [None, Some(true), Some(false)] {
                let ctx = EffortContext {
                    generation,
                    system: msgs.iter().any(|m| m.role == crate::wal::role::SYSTEM),
                    tools: !tools.is_empty(),
                    thinking: on,
                };
                let with = |effort: Option<&str>| {
                    let mut kw = serde_json::Map::new();
                    if let Some(on) = on {
                        kw.insert("enable_thinking".into(), serde_json::Value::Bool(on));
                    }
                    if let Some(e) = effort {
                        kw.insert("reasoning_effort".into(), serde_json::Value::String(e.into()));
                    }
                    render(msgs, tools, generation, &kw)
                };
                let unavailable = with(None).is_ok() && EFFORT_SCALE.iter().all(|e| with(Some(*e)).is_err());
                let ladder = (!unavailable).then(|| learn_effort_ladder(with));
                ladders.push((ctx, ladder));
            }
        }
    }
    ladders
}

fn common_levels<'a>(ladders: impl IntoIterator<Item = Option<&'a [&'static str]>>) -> Vec<&'static str> {
    let ladders: Option<Vec<&[&'static str]>> = ladders.into_iter().collect();
    let Some(ladders) = ladders else { return Vec::new() };
    let mut it = ladders.into_iter().filter(|l| !l.is_empty());
    let Some(first) = it.next() else { return Vec::new() };
    let mut out = first.to_vec();
    for l in it {
        out.retain(|n| l.contains(n));
    }
    out
}

fn effort_rank(name: &str) -> Option<usize> {
    EFFORT_SCALE.iter().position(|n| *n == name)
}

fn contains_word(text: &str, word: &str) -> bool {
    let b = text.as_bytes();
    text.match_indices(word).any(|(i, _)| {
        let before = i == 0 || !b[i - 1].is_ascii_alphanumeric();
        let j = i + word.len();
        let after = j >= b.len() || !b[j].is_ascii_alphanumeric();
        before && after
    })
}

fn learn_effort_ladder<E>(render: impl Fn(Option<&str>) -> Result<String, E>) -> Vec<&'static str> {
    let mut groups: Vec<(String, String, Vec<&'static str>)> = Vec::new();
    let mut refused = 0;
    for &name in EFFORT_SCALE.iter() {
        let Ok(out) = render(Some(name)) else {
            refused += 1;
            continue;
        };
        match groups.iter_mut().find(|(o, _, _)| *o == out) {
            Some((_, _, names)) => names.push(name),
            None => {
                let lower = out.to_lowercase();
                groups.push((out, lower, vec![name]));
            }
        }
    }
    if groups.len() < 2 {
        return match (groups.pop(), refused) {
            (Some((_, _, names)), 1..) => names,
            _ => Vec::new(),
        };
    }
    let mut ladder: Vec<&'static str> = Vec::new();
    for (i, (_, lower, names)) in groups.iter().enumerate() {
        let spelled: Vec<&'static str> = names
            .iter()
            .copied()
            .filter(|n| {
                contains_word(lower, n)
                    && groups.iter().enumerate().all(|(j, (_, l, _))| j == i || !contains_word(l, n))
            })
            .collect();
        match spelled.as_slice() {
            [one] => ladder.push(*one),
            _ => ladder.extend(names.iter().copied()),
        }
    }
    ladder.sort_by_key(|n| effort_rank(n));
    ladder
}

fn nearest_effort(ladder: &[&'static str], requested: &str) -> Option<&'static str> {
    let req = requested.trim().to_ascii_lowercase();
    if ladder.is_empty() {
        return None;
    }
    if let Some(l) = ladder.iter().find(|l| **l == req) {
        return (requested != *l).then_some(*l);
    }
    let r = effort_rank(&req)? as i64;
    ladder.iter().copied().min_by_key(|l| {
        let d = effort_rank(l).unwrap_or(0) as i64 - r;
        (d.abs(), d < 0)
    })
}

#[cfg(test)]
mod effort_ladder_tests {
    use super::*;

    fn ladder_of(src: &str) -> Vec<&'static str> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |m: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m))
        });
        env.add_function("strftime_now", |_f: String| "2026-10-04".to_string());
        env.add_filter("tojson", hf_tojson);
        env.add_template_owned("chat", src.to_string()).expect("template compiles");
        let t = env.get_template("chat").unwrap();
        learn_effort_ladder(|e| {
            let msgs = vec![context! { role => "user", content => "probe" }];
            match e {
                Some(e) => t.render(context! { messages => msgs, add_generation_prompt => true, reasoning_effort => e }),
                None => t.render(context! { messages => msgs, add_generation_prompt => true }),
            }
        })
    }

    #[test]
    fn glm_5_3_learns_low_high_max_and_medium_goes_to_high() {
        let l = ladder_of(include_str!("../../../tests/fixtures/chat_templates/glm_5_3.jinja"));
        assert_eq!(l, ["low", "high", "max"]);
        assert_eq!(nearest_effort(&l, "medium"), Some("high"));
        assert_eq!(nearest_effort(&l, "minimal"), Some("low"));
        assert_eq!(nearest_effort(&l, "none"), Some("low"), "the bottom of the scale, not the top");
        assert_eq!(nearest_effort(&l, "xhigh"), Some("max"));
        for level in ["low", "high", "max"] {
            assert_eq!(nearest_effort(&l, level), None, "{level} is native: untouched");
        }
        assert_eq!(nearest_effort(&l, "High"), Some("high"));
        assert_eq!(nearest_effort(&l, " low "), Some("low"));
    }

    #[test]
    fn glm_5_2_knows_only_high_and_max() {
        let l = ladder_of(include_str!("../../../tests/fixtures/chat_templates/glm_5_2.jinja"));
        assert_eq!(l, ["high", "max"]);
        assert_eq!(nearest_effort(&l, "low"), Some("high"), "low used to mean MAX here");
        assert_eq!(nearest_effort(&l, "medium"), Some("high"));
    }

    #[test]
    fn gpt_oss_passes_every_value_through() {
        let l = ladder_of(include_str!("../../../tests/fixtures/chat_templates/gpt_oss.jinja"));
        assert_eq!(l, EFFORT_SCALE);
        for v in EFFORT_SCALE {
            assert_eq!(nearest_effort(&l, v), None);
        }
    }

    #[test]
    fn a_validating_template_gets_a_value_it_accepts() {
        let src = "{% if reasoning_effort is defined and reasoning_effort not in ['low', 'medium', 'xhigh'] %}\
                   {{ raise_exception('reasoning_effort must be low, medium or xhigh') }}{% endif %}\
                   effort={{ reasoning_effort | default('medium') }}{% for m in messages %}{{ m.content }}{% endfor %}";
        let l = ladder_of(src);
        assert_eq!(l, ["low", "medium", "xhigh"]);
        assert_eq!(nearest_effort(&l, "high"), Some("xhigh"), "high used to 400");
        assert_eq!(nearest_effort(&l, "max"), Some("xhigh"));
        assert_eq!(nearest_effort(&l, "minimal"), Some("low"));
    }

    #[test]
    fn templates_without_the_knob_and_unknown_words_are_left_alone() {
        let l = ladder_of(include_str!("../../../tests/fixtures/chat_templates/nemotron_3_nano.jinja"));
        assert!(l.is_empty(), "{l:?}");
        assert_eq!(nearest_effort(&l, "high"), None);
        let glm = ["low", "high", "max"];
        assert_eq!(nearest_effort(&glm, "auto"), None, "not a scale word: the template decides");
    }

    #[test]
    fn a_distinct_level_without_a_label_keeps_its_names() {
        let src = "{% if reasoning_effort in ['minimal', 'low'] %}short{% elif reasoning_effort == 'high' %}effort high\
                   {% else %}effort max{% endif %}\n{% for m in messages %}{{ m.content }}{% endfor %}";
        let l = ladder_of(src);
        assert_eq!(l, ["minimal", "low", "high", "max"]);
        for native in ["minimal", "low", "high", "max"] {
            assert_eq!(nearest_effort(&l, native), None, "{native} is a level the template renders");
        }
        assert_eq!(nearest_effort(&l, "medium"), Some("high"));
        assert_eq!(nearest_effort(&l, "none"), Some("minimal"));
    }

    #[test]
    fn case_only_differences_are_distinct_levels() {
        let src = "{% if reasoning_effort == 'high' %}Effort: HIGH{% elif reasoning_effort == 'max' %}Effort: high\
                   {% else %}Effort: low{% endif %}\n{% for m in messages %}{{ m.content }}{% endfor %}";
        let l = ladder_of(src);
        assert!(l.contains(&"high") && l.contains(&"max"), "{l:?}");
        assert_eq!(nearest_effort(&l, "max"), None, "max renders its own prompt; never folded into high");
        assert_eq!(nearest_effort(&l, "high"), None);
    }

    #[test]
    fn normalization_is_part_of_the_render_identity() {
        let d = 0x1234_5678_9abc_def0;
        assert_eq!(render_identity_of(d, &[]), "123456789abcdef0", "no ladder: the digest alone, as before");
        assert_eq!(render_identity_of(d, &[Some(&[]), Some(&[])]), "123456789abcdef0");
        let lhm: Option<&[&str]> = Some(&["low", "high", "max"]);
        let glm = render_identity_of(d, &[lhm, lhm, lhm]);
        assert_eq!(glm, format!("123456789abcdef0/effort{EFFORT_NORMALIZATION_VERSION}:low,high,max"));
        assert_ne!(glm, render_identity_of(d, &[Some(&["high", "max"])]));
        assert_ne!(glm, render_identity_of(d, &[lhm, Some(&["low", "high"])]), "a context's ladder is identity too");
        assert_ne!(glm, render_identity_of(d, &[lhm, None]), "so is a shape that refuses every value");
    }

    #[test]
    fn a_template_that_accepts_one_value_gets_it() {
        let src = "{% if reasoning_effort is defined and reasoning_effort != 'high' %}\
                   {{ raise_exception('only high') }}{% endif %}effort={{ reasoning_effort }}\n\
                   {% for m in messages %}{{ m.content }}{% endfor %}";
        let l = ladder_of(src);
        assert_eq!(l, ["high"]);
        assert_eq!(nearest_effort(&l, "low"), Some("high"));
        assert!(ladder_of("{% for m in messages %}{{ m.content }}{% endfor %}").is_empty());
    }

    #[test]
    fn each_request_shape_is_normalized_against_its_own_ladder() {
        let src = "{% if tools %}{% set ok = ['low', 'medium', 'high'] %}{% else %}{% set ok = ['low', 'high'] %}{% endif %}\
                   Effort: {{ reasoning_effort if reasoning_effort in ok else 'high' }}\n\
                   {% for m in messages %}{{ m.content }}{% endfor %}";
        let mut env = Environment::new();
        env.add_template_owned("chat", src.to_string()).unwrap();
        let t = env.get_template("chat").unwrap();
        let learn = |tools: bool| {
            learn_effort_ladder(|e| {
                let msgs = vec![context! { role => "user", content => "probe" }];
                let tools: Vec<Value> = if tools { vec![Value::from("probe")] } else { Vec::new() };
                t.render(context! { messages => msgs, tools => tools, reasoning_effort => e })
            })
        };
        assert_eq!(learn(false), ["low", "high"]);
        assert_eq!(learn(true), ["low", "medium", "high"]);
        let ctx = |system, tools, thinking| EffortContext { generation: true, system, tools, thinking };
        let ladders = vec![
            (ctx(false, false, None), Some(learn(false))),
            (ctx(true, false, None), Some(learn(false))),
            (ctx(true, true, None), Some(learn(true))),
        ];
        let with_tools = ctx(false, true, None).best(&ladders);
        let without = ctx(false, false, None).best(&ladders);
        assert_eq!(nearest_effort(with_tools, "medium"), None, "native with tools: never remapped");
        assert_eq!(nearest_effort(without, "medium"), Some("high"), "not a level without tools");
        assert_eq!(nearest_effort(without, "max"), Some("high"));
        let levels = common_levels(ladders.iter().map(|(_, l)| l.as_deref()));
        assert_eq!(levels, ["low", "high"], "advertised: what every shape has");
        let msgs = vec![context! { role => "system", content => "s" }, context! { role => "user", content => "u" }];
        let mut kw = serde_json::Map::new();
        kw.insert("enable_thinking".into(), serde_json::Value::Bool(false));
        assert_eq!(EffortContext::of(&msgs, &["{}".to_string()], true, &kw), ctx(true, true, Some(false)));
        assert_eq!(EffortContext::of(&msgs[1..], &[], true, &serde_json::Map::new()), ctx(false, false, None));
        let ladders = vec![
            (ctx(true, true, None), Some(vec!["high", "max"])),
            (ctx(false, true, None), Some(vec!["low", "high"])),
        ];
        assert_eq!(ctx(false, true, None).best(&ladders), ["low", "high"]);
        assert_eq!(ctx(true, true, None).best(&ladders), ["high", "max"]);
    }

    #[test]
    fn effort_read_only_under_thinking_is_learned_there() {
        let src = "{% if enable_thinking is defined and enable_thinking and reasoning_effort is defined %}\
                   {% if reasoning_effort not in ['low', 'high'] %}{{ raise_exception('bad effort') }}{% endif %}\
                   effort={{ reasoning_effort }}\n{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}";
        let mut env = Environment::new();
        env.add_function("raise_exception", |m: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m))
        });
        env.add_template_owned("chat", src.to_string()).unwrap();
        let t = env.get_template("chat").unwrap();
        let ladders = learn_effort_ladders(|msgs, tools, _generation, kw| {
            let msgs: Vec<Value> = msgs.iter().map(|m| context! { content => m.content.clone() }).collect();
            let tools: Vec<Value> = tools.iter().map(|t| Value::from(t.clone())).collect();
            t.render(context! { messages => msgs, tools => tools, ..Value::from_serialize(kw) })
        });
        assert_eq!(ladders.len(), 24, "4 conversations x 2 render modes x 3 thinking values");
        let ctx = |system, tools, thinking| EffortContext { generation: true, system, tools, thinking };
        assert_eq!(ctx(false, true, Some(true)).best(&ladders), ["low", "high"]);
        assert!(ctx(false, true, Some(false)).best(&ladders).is_empty(), "ignored when thinking is off");
        assert_eq!(nearest_effort(ctx(false, false, Some(true)).best(&ladders), "max"), Some("high"));
        assert!(ladders.iter().any(|(c, _)| *c == ctx(false, true, None)), "tools without a system turn");
    }

    #[test]
    fn render_modes_and_refusing_shapes() {
        let src = "{% if reasoning_effort is defined %}{% if not add_generation_prompt %}{{ raise_exception('no effort here') }}\
                   {% elif reasoning_effort not in ['low', 'high'] %}{{ raise_exception('bad') }}{% endif %}\
                   effort={{ reasoning_effort }}\n{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}";
        let mut env = Environment::new();
        env.add_function("raise_exception", |m: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m))
        });
        env.add_template_owned("chat", src.to_string()).unwrap();
        let t = env.get_template("chat").unwrap();
        let ladders = learn_effort_ladders(|msgs, _tools, generation, kw| {
            let msgs: Vec<Value> = msgs.iter().map(|m| context! { content => m.content.clone() }).collect();
            t.render(context! { messages => msgs, add_generation_prompt => generation, ..Value::from_serialize(kw) })
        });
        let shape = |generation| EffortContext { generation, system: false, tools: false, thinking: None };
        assert_eq!(shape(true).best(&ladders), ["low", "high"]);
        assert!(ladders.iter().any(|(c, l)| *c == shape(false) && l.is_none()), "refused, not ignored");
        assert!(shape(false).best(&ladders).is_empty(), "nothing to map to: passes through");
        assert_eq!(
            common_levels(ladders.iter().map(|(_, l)| l.as_deref())),
            Vec::<&str>::new(),
            "no level works in every shape"
        );
        assert_eq!(common_levels([Some(&["low", "high"][..]), Some(&[][..])]), ["low", "high"]);
    }

    #[test]
    fn whole_words_only() {
        assert!(contains_word("effort: xhigh", "xhigh"));
        assert!(!contains_word("effort: xhigh", "high"));
        assert!(contains_word("reasoning effort: high", "high"));
    }
}

/// `tojson` as chat templates expect it: Python's `json.dumps` separators and indentation,
/// characters unescaped.
fn hf_tojson(
    value: &Value,
    indent: Option<Value>,
    args: minijinja::value::Kwargs,
) -> Result<Value, minijinja::Error> {
    let ensure_ascii: Option<bool> = args.get("ensure_ascii")?;
    let indent = match indent {
        Some(i) => Some(i),
        None => args.get("indent")?,
    };
    args.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(v) => match bool::try_from(v.clone()).ok() {
            Some(true) => Some(2),
            Some(false) => None,
            None => Some(usize::try_from(v)?),
        },
    };
    let mut out = Vec::<u8>::new();
    let written = match indent {
        Some(n) => {
            let pad = " ".repeat(n);
            let formatter = serde_json::ser::PrettyFormatter::with_indent(pad.as_bytes());
            serde::Serialize::serialize(value, &mut serde_json::Serializer::with_formatter(&mut out, formatter))
        }
        None => serde::Serialize::serialize(value, &mut serde_json::Serializer::with_formatter(&mut out, PyJsonFormatter)),
    };
    let unwritable = |e: &dyn std::fmt::Display| {
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("cannot serialize to JSON: {e}"))
    };
    written.map_err(|e| unwritable(&e))?;
    let s = String::from_utf8(out).map_err(|e| unwritable(&e))?;
    if !ensure_ascii.unwrap_or(false) {
        return Ok(Value::from_safe_string(s));
    }
    let mut ascii = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            ascii.push(c);
        } else {
            let mut buf = [0u16; 2];
            for unit in c.encode_utf16(&mut buf) {
                ascii.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    Ok(Value::from_safe_string(ascii))
}

/// `json.dumps`'s default separators: `", "` between items, `": "` after a key.
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

#[cfg(test)]
mod tojson_tests {
    use super::*;

    fn render(src: &str, v: serde_json::Value) -> Result<String, minijinja::Error> {
        let mut env = Environment::new();
        env.add_filter("tojson", hf_tojson);
        env.add_template("t", src)?;
        env.get_template("t")?.render(context! { v => Value::from_serialize(&v) })
    }

    #[test]
    fn hf_keywords_render_and_match_the_builtin() {
        let v = serde_json::json!({"name": "get_weather", "desc": "Météo <now> & 'later'", "n": [1, 2]});
        let plain = render("{{ v | tojson }}", v.clone()).unwrap();
        assert_eq!(render("{{ v | tojson(ensure_ascii=False) }}", v.clone()).unwrap(), plain);
        assert_eq!(plain, r#"{"name": "get_weather", "desc": "Météo <now> & 'later'", "n": [1, 2]}"#, "json.dumps's layout");
        let ascii = render("{{ v | tojson(ensure_ascii=True) }}", v.clone()).unwrap();
        assert_eq!(ascii, plain.replace('é', "\\u00e9"));
        let emoji = render("{{ v | tojson(ensure_ascii=True) }}", serde_json::json!("🙂")).unwrap();
        assert_eq!(emoji, "\"\\ud83d\\ude42\"");
        assert_eq!(
            render("{{ v | tojson(2) }}", v.clone()).unwrap(),
            render("{{ v | tojson(indent=2, ensure_ascii=False) }}", v.clone()).unwrap()
        );
    }

    #[test]
    fn loop_controls_compile_and_run() {
        let mut env = Environment::new();
        env.add_template("t", "{% for x in v %}{% if x > 2 %}{% break %}{% endif %}{% if x == 1 %}{% continue %}{% endif %}{{ x }}{% endfor %}")
            .expect("break/continue compile");
        let out = env.get_template("t").unwrap().render(context! { v => vec![0, 1, 2, 3, 4] }).unwrap();
        assert_eq!(out, "02");
    }

    #[test]
    fn layout_changing_keywords_are_still_refused() {
        let v = serde_json::json!({"b": 1, "a": 2});
        assert!(render("{{ v | tojson(sort_keys=True) }}", v.clone()).is_err());
        assert!(render("{{ v | tojson(separators=[',', ':']) }}", v).is_err());
    }
}

#[cfg(test)]
mod date_tests {
    use super::*;

    #[test]
    fn civil_dates_are_right_around_the_awkward_days() {
        assert_eq!(civil_date(0), "1970-01-01");
        assert_eq!(civil_date(10_956), "1999-12-31");
        assert_eq!(civil_date(11_016), "2000-02-29");
        assert_eq!(civil_date(20_724), "2026-09-28");
        assert_eq!(civil_date(-1), "1969-12-31");
    }

    #[test]
    fn today_is_a_date_after_the_old_constant() {
        let t = today_utc();
        assert_eq!(t.len(), 10);
        assert!(t.as_str() > "2024-01-01", "{t}");
    }
}

#[cfg(test)]
mod glm_wire_tests {
    use super::*;
    use crate::codec::ToolSchemas;

    const GLM_5_2: &str = include_str!("../../../tests/fixtures/chat_templates/glm_5_2.jinja");
    const GLM_5_3: &str = include_str!("../../../tests/fixtures/chat_templates/glm_5_3.jinja");
    const GLM_4_5_SHAPE: &str = "{%- for m in messages %}{%- if m.tool_calls %}{%- for tc in m.tool_calls %}\
        {%- set tc = tc.function %}<tool_call>{{ tc.name }}\n\
        {% for k, v in tc.arguments.items() %}<arg_key>{{ k }}</arg_key>\n\
        <arg_value>{{ v | tojson(ensure_ascii=False) if v is not string else v }}</arg_value>\n\
        {% endfor %}</tool_call>{% endfor %}{% else %}<|{{ m.role }}|>{{ m.content }}{% endif %}{% endfor %}";

    pub(super) fn env_for(src: &str) -> Environment<'static> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |m: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m))
        });
        env.add_function("strftime_now", |_f: String| "2026-10-04".to_string());
        env.add_filter("tojson", hf_tojson);
        env.add_template_owned("chat", src.to_string()).expect("template compiles");
        env
    }

    pub(super) fn render_call(env: &Environment<'static>, name: &str, args: &serde_json::Value) -> String {
        let msgs = serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": name, "arguments": args}}
            ]}
        ]);
        env.get_template("chat")
            .unwrap()
            .render(context! { messages => Value::from_serialize(&msgs), add_generation_prompt => false })
            .expect("renders a call")
    }

    pub(super) fn call_body(rendered: &str) -> &str {
        let open = rendered.rfind("<tool_call>").expect("a call block") + "<tool_call>".len();
        let close = open + rendered[open..].find("</tool_call>").expect("closed");
        &rendered[open..close]
    }

    #[test]
    fn glm_templates_learn_the_glm_wire() {
        for (label, src) in [("GLM-5.2", GLM_5_2), ("GLM-5.3", GLM_5_3), ("GLM-4.5 shape", GLM_4_5_SHAPE)] {
            let env = env_for(src);
            let wire = ToolWire::learn_from(
                |args| {
                    let a: serde_json::Value = serde_json::from_str(args).ok()?;
                    Some(render_call(&env, S_NAME, &a))
                },
                Some(("<tool_call>", "</tool_call>")),
            );
            assert!(matches!(wire, ToolWire::Envelope(ToolEnvelope::Glm)), "{label}");
        }
    }

    #[test]
    fn calls_the_real_templates_render_read_back_exactly() {
        let tool = serde_json::json!({"type": "function", "function": {"name": "edit_file", "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}, "line": {"type": "string"}, "flag": {"type": "string"},
                "nothing": {"type": "string"}, "content": {"type": "string"},
                "count": {"type": "integer"}, "ratio": {"type": "number"}, "dry_run": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "opts": {"type": "object"}, "maybe": {"type": ["string", "null"]}
            }
        }}});
        let args = serde_json::json!({
            "path": "/tmp/naïve café.txt", "line": "3", "flag": "true", "nothing": "null",
            "content": "\n    def f():\n        return '<x>' & 1\n",
            "count": 3, "ratio": 0.5, "dry_run": false, "tags": ["a", "b c"],
            "opts": {"k": [1, 2], "s": "v"}, "maybe": null
        });
        let schemas = ToolSchemas::from_values([&tool]);
        for (label, src) in [("GLM-5.2", GLM_5_2), ("GLM-5.3", GLM_5_3), ("GLM-4.5 shape", GLM_4_5_SHAPE)] {
            let env = env_for(src);
            let rendered = render_call(&env, "edit_file", &args);
            let body = call_body(&rendered);
            let (name, got) = ToolEnvelope::Glm.parse_call(body, Some(&schemas)).expect(label);
            assert_eq!(name, "edit_file", "{label}");
            let got: serde_json::Value = serde_json::from_str(&got).unwrap();
            assert_eq!(got, args, "{label}: rendered body {body:?}");
            let (_, blind) = ToolEnvelope::Glm.parse_call(body, None).unwrap();
            let blind: serde_json::Value = serde_json::from_str(&blind).unwrap();
            assert_eq!(blind["line"], 3, "{label}");
        }
    }

    #[test]
    fn zero_argument_calls_render_and_read_as_a_bare_name() {
        for src in [GLM_5_2, GLM_5_3] {
            let env = env_for(src);
            let rendered = render_call(&env, "list_files", &serde_json::json!({}));
            assert_eq!(call_body(&rendered), "list_files");
            let parsed = ToolEnvelope::Glm.parse_call(call_body(&rendered), None);
            assert_eq!(parsed, Some(("list_files".to_string(), "{}".to_string())));
        }
    }
}

#[cfg(test)]
mod qwen_wire_tests {
    use super::glm_wire_tests::{call_body, env_for, render_call};
    use super::*;
    use crate::codec::ToolSchemas;

    const QWEN_3_6: &str = include_str!("../../../tests/fixtures/chat_templates/qwen3_6.jinja");
    const QWEN_3_5: &str = include_str!("../../../tests/fixtures/chat_templates/qwen3_5.jinja");

    fn learned(src: &str) -> (Environment<'static>, ToolWire) {
        let env = env_for(src);
        let wire = ToolWire::learn_from(
            |args| {
                let a: serde_json::Value = serde_json::from_str(args).ok()?;
                Some(render_call(&env, S_NAME, &a))
            },
            Some(("<tool_call>", "</tool_call>")),
        );
        (env, wire)
    }

    #[test]
    fn qwen_templates_render_the_wire_the_parser_reads() {
        let tool = serde_json::json!({"type": "function", "function": {"name": "edit_file", "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}, "port": {"type": "string"}, "flag": {"type": "string"},
                "nothing": {"type": "string"}, "content": {"type": "string"}, "blank": {"type": "string"},
                "id": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "count": {"type": "integer"}, "ratio": {"type": "number"}, "dry_run": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "opts": {"type": "object"}
            }
        }}});
        let args = serde_json::json!({
            "path": "/tmp/naïve café.txt", "port": "8080", "flag": "true", "nothing": "null",
            "content": "\n    def f():\n        return '<x>' & 1\n", "blank": "",
            "id": "42",
            "count": 3, "ratio": 0.5, "dry_run": false, "tags": ["a", "b c"],
            "opts": {"k": [1, 2], "s": "v"}
        });
        let schemas = ToolSchemas::from_values([&tool]);
        for (label, src) in [("Qwen3.6", QWEN_3_6), ("Qwen3.5", QWEN_3_5)] {
            let (env, wire) = learned(src);
            assert!(matches!(wire, ToolWire::Delimited(_)), "{label}: learns the XML wire");
            let rendered = render_call(&env, "edit_file", &args);
            let body = call_body(&rendered);
            let (name, got) = wire.parse(body, Some(&schemas)).expect(label);
            assert_eq!(name, "edit_file", "{label}");
            let got: serde_json::Value = serde_json::from_str(&got).unwrap();
            assert_eq!(got, args, "{label}: rendered body {body:?}");
            let (_, blind) = wire.parse(body, None).unwrap();
            let blind: serde_json::Value = serde_json::from_str(&blind).unwrap();
            assert_eq!(blind["port"], 8080, "{label}");
            assert_eq!(blind["content"], args["content"], "{label}: payload whitespace never depended on the schema");
        }
    }

    #[test]
    fn a_string_port_and_a_trailing_newline_survive_the_real_template() {
        let (env, wire) = learned(QWEN_3_6);
        let tool = serde_json::json!({"type": "function", "function": {"name": "write", "parameters": {
            "type": "object",
            "properties": {"port": {"type": "string"}, "content": {"type": "string"}},
            "required": ["port", "content"]
        }}});
        let args = serde_json::json!({"port": "8080", "content": "<!DOCTYPE html>\n<p>hi</p>\n"});
        let rendered = render_call(&env, "write", &args);
        let body = call_body(&rendered);
        assert!(body.contains("<parameter=port>\n8080\n</parameter>\n"), "{body:?}");
        let (_, got) = wire.parse(body, Some(&ToolSchemas::from_values([&tool]))).unwrap();
        assert_eq!(serde_json::from_str::<serde_json::Value>(&got).unwrap(), args);
    }
}
