//! The codec layer over a tokenizer that is NOT libbaseRT's.

use std::sync::Arc;

use superfluid_daemon::codec::{
    auto_codec_from_tokenizer, chatml_codec_from_tokenizer, BundleCodec, ChatMlCodec, TextCodec,
    TurnState,
};
use superfluid_daemon::template_codec::TemplateCodec;
use superfluid_daemon::wal::role;
use superfluid_engine::Tokenizer;

struct FakeTokenizer {
    specials: Vec<(String, u32)>,
    template: String,
    control: bool,
    bos: Option<u32>,
}

const IM_START: u32 = 256;
const IM_END: u32 = 257;

const TEMPLATE: &str = r#"{%- for message in messages %}
    {%- if message.role == "assistant" and message.tool_calls %}
        {{- '<|im_start|>assistant\n' }}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}
            {{- '<tool_call>\n{"name": "' }}{{- tool_call.name }}{{- '", "arguments": ' }}
            {%- if tool_call.arguments is string %}{{- tool_call.arguments }}{%- else %}{{- tool_call.arguments | tojson }}{%- endif %}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"#;

impl FakeTokenizer {
    fn chatml() -> Arc<dyn Tokenizer> {
        Arc::new(FakeTokenizer {
            specials: [
                "<|im_start|>",
                "<|im_end|>",
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<|endoftext|>",
            ]
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), 256 + i as u32))
            .collect(),
            template: TEMPLATE.to_string(),
            control: false,
            bos: None,
        })
    }

    fn chatml_control() -> Arc<dyn Tokenizer> {
        Arc::new(FakeTokenizer {
            specials: [
                "<|im_start|>",
                "<|im_end|>",
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<|endoftext|>",
            ]
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), 256 + i as u32))
            .collect(),
            template: TEMPLATE.to_string(),
            control: true,
            bos: None,
        })
    }

    fn bare() -> Arc<dyn Tokenizer> {
        Arc::new(FakeTokenizer {
            specials: vec![("<|endoftext|>".into(), 256)],
            template: String::new(),
            control: false,
            bos: None,
        })
    }
}

impl Tokenizer for FakeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let b = text.as_bytes();
        let mut out = Vec::new();
        if let Some(bos) = self.bos {
            out.push(bos);
        }
        let mut i = 0;
        while i < b.len() {
            let special = self
                .specials
                .iter()
                .filter(|(s, _)| b[i..].starts_with(s.as_bytes()))
                .max_by_key(|(s, _)| s.len());
            match special {
                Some((s, id)) => {
                    out.push(*id);
                    i += s.len();
                }
                None => {
                    out.push(b[i] as u32);
                    i += 1;
                }
            }
        }
        out
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        if token < 256 {
            vec![token as u8]
        } else if self.control {
            Vec::new()
        } else {
            self.specials
                .iter()
                .find(|(_, id)| *id == token)
                .map(|(s, _)| s.as_bytes().to_vec())
                .unwrap_or_default()
        }
    }
    fn vocab_size(&self) -> u32 {
        256 + self.specials.len() as u32
    }
    fn special_tokens(&self) -> Vec<(String, u32)> {
        self.specials.clone()
    }
    fn bos_token(&self) -> Option<u32> {
        self.bos
    }
    fn eos_token(&self) -> u32 {
        self.specials.last().map(|(_, id)| *id).unwrap_or(0)
    }
    fn chat_template_jinja(&self) -> String {
        self.template.clone()
    }
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

fn user_turn(text: &str) -> Vec<u32> {
    let mut v = vec![IM_START];
    v.extend(bytes("user\n"));
    v.extend(bytes(text));
    v.push(IM_END);
    v.extend(bytes("\n"));
    v
}

#[test]
fn raw_codec_over_any_tokenizer() {
    let tok = FakeTokenizer::chatml();
    let codec = BundleCodec::from_tokenizer(Arc::clone(&tok));
    assert_eq!(
        codec.encode("hi<|im_end|>"),
        vec![b'h' as u32, b'i' as u32, IM_END]
    );
    assert_eq!(codec.decode(&[b'h' as u32, b'i' as u32]), "hi");
    let fp = codec
        .behavior_fingerprint()
        .expect("a tokenizer-backed codec carries an identity");
    assert_eq!(
        fp.tokenizer_hash,
        superfluid_fingerprint::vocab_hash(tok.vocab_size(), |t| tok.token_bytes(t))
    );
}

#[test]
fn curated_chatml_over_any_tokenizer() {
    let codec = ChatMlCodec::from_tokenizer(FakeTokenizer::chatml()).expect("markers are atomic");
    assert_eq!(
        codec.render_message(role::USER, "hi"),
        Some(user_turn("hi"))
    );
    let opener = codec
        .generation_prefix(TurnState::AfterInput)
        .expect("chatml opens the assistant turn");
    let mut expect = vec![IM_START];
    expect.extend(bytes("assistant\n"));
    assert!(opener.starts_with(&expect), "opener {opener:?}");
    assert!(chatml_codec_from_tokenizer(FakeTokenizer::chatml()).is_ok());
}

#[test]
fn harmony_template_routes_channels_and_parses_calls() {
    const START: u32 = 300;
    const CHANNEL: u32 = 301;
    const MESSAGE: u32 = 302;
    const END: u32 = 303;
    const RETURN: u32 = 304;
    const CALL: u32 = 305;
    const CONSTRAIN: u32 = 306;
    let specials = vec![
        ("<|start|>".to_string(), START),
        ("<|channel|>".to_string(), CHANNEL),
        ("<|message|>".to_string(), MESSAGE),
        ("<|end|>".to_string(), END),
        ("<|return|>".to_string(), RETURN),
        ("<|call|>".to_string(), CALL),
        ("<|constrain|>".to_string(), CONSTRAIN),
    ];
    let template = "{%- for message in messages %}{%- if message.role == 'assistant' %}<|start|>assistant<|channel|>final<|message|>{{ message.content }}<|end|>{%- else %}<|start|>{{ message.role }}<|message|>{{ message.content }}<|end|>{%- endif %}{%- endfor %}\
{%- if add_generation_prompt %}<|start|>assistant{%- endif %}";
    let tok = FakeTokenizer { specials, template: template.into(), control: true, bos: None };
    let codec = TemplateCodec::from_tokenizer(std::sync::Arc::new(tok)).expect("template codec");
    assert_eq!(codec.tool_wire_kind(), "harmony");
    assert_eq!(codec.tool_envelope(), superfluid_daemon::codec::ToolEnvelope::Harmony);
    let mut terms = codec.turn_terminators();
    terms.sort_unstable();
    assert_eq!(terms, vec![RETURN, CALL], "the turn ends at <|return|> or <|call|>");
    let prompt = codec.render_conversation_tokens(&[("user", "hi")], &[], true).expect("renders");
    assert_eq!(prompt.last(), Some(&bytes("assistant").last().copied().unwrap()), "the prompt ends inside the assistant's header");

    let mut ch = codec.channelizer();
    let mut toks: Vec<u32> = vec![CHANNEL];
    toks.extend(bytes("analysis"));
    toks.push(MESSAGE);
    toks.extend(bytes("plan"));
    toks.extend([END, START]);
    toks.extend(bytes("assistant"));
    toks.push(CHANNEL);
    toks.extend(bytes("commentary to=functions.lookup "));
    toks.push(CONSTRAIN);
    toks.extend(bytes("json"));
    toks.push(MESSAGE);
    toks.extend(bytes(r#"{"q":"x"}"#));
    toks.push(CALL);
    let runs = ch.split(&toks);
    let decode = |t: &[u32]| String::from_utf8(t.iter().flat_map(|&x| codec.token_bytes(x)).collect()).unwrap();
    let routed: Vec<(u32, String, bool)> = runs.iter().filter(|r| !r.text.is_empty()).map(|r| (r.channel, decode(&r.text), r.closes)).collect();
    assert_eq!(routed.len(), 2);
    assert_eq!(routed[0], (superfluid_daemon::wal::channel::REASONING, "plan".into(), true));
    assert_eq!(routed[1].0, superfluid_daemon::wal::channel::TOOL_CALL);
    assert!(routed[1].2, "<|call|> closes the call");
    assert_eq!(codec.parse_tool_call(&routed[1].1), Some(("lookup".into(), r#"{"q":"x"}"#.into())));
    let mut toks: Vec<u32> = vec![CHANNEL];
    toks.extend(bytes("final"));
    toks.push(MESSAGE);
    toks.extend(bytes("Hello!"));
    toks.push(RETURN);
    let runs = codec.channelizer().split(&toks);
    let visible: Vec<(u32, String, bool)> = runs.iter().filter(|r| !r.text.is_empty()).map(|r| (r.channel, decode(&r.text), r.closes)).collect();
    assert_eq!(visible, vec![(superfluid_daemon::wal::channel::TEXT, "Hello!".into(), true)]);
}

#[test]
fn template_codec_renders_one_bos_for_a_bos_family() {
    const BOS: u32 = 258;
    let tok = FakeTokenizer {
        specials: vec![("<|im_start|>".into(), IM_START), ("<|im_end|>".into(), IM_END), ("<bos>".into(), BOS)],
        template: format!("{{{{ bos_token }}}}{TEMPLATE}"),
        control: true,
        bos: Some(BOS),
    };
    assert_eq!(tok.encode("hi")[0], BOS, "the fake auto-prepends BOS like its families");
    let codec = TemplateCodec::from_tokenizer(std::sync::Arc::new(tok)).expect("template codec");
    let toks = codec.render_conversation_tokens(&[("user", "hi")], &[], true).expect("renders");
    assert_eq!(toks[0], BOS, "the template's BOS leads");
    assert_ne!(toks[1], BOS, "and it is the only one: {toks:?}");
    let tok = FakeTokenizer {
        specials: vec![("<|im_start|>".into(), IM_START), ("<|im_end|>".into(), IM_END), ("<bos>".into(), BOS)],
        template: TEMPLATE.into(),
        control: true,
        bos: Some(BOS),
    };
    let codec = TemplateCodec::from_tokenizer(std::sync::Arc::new(tok)).expect("template codec");
    let toks = codec.render_conversation_tokens(&[("user", "hi")], &[], true).expect("renders");
    assert_eq!(toks[0], BOS);
    assert_ne!(toks[1], BOS);
}

#[test]
fn template_codec_over_any_tokenizer_frames_like_chatml() {
    let tmpl = TemplateCodec::from_tokenizer(FakeTokenizer::chatml()).expect("template loads");
    assert!(tmpl.verified(), "per-message rendering is append-stable");
    assert_eq!(
        tmpl.tool_wire_kind(),
        "json",
        "the template writes JSON call bodies"
    );
    let messages = [(role::USER, "hi".to_string())];
    let rendered = tmpl
        .render_prompt(&messages, &[])
        .expect("template renders a prompt");
    let mut expect = user_turn("hi");
    expect.push(IM_START);
    expect.extend(bytes("assistant\n"));
    assert_eq!(rendered, expect);
    let curated = ChatMlCodec::from_tokenizer(FakeTokenizer::chatml()).unwrap();
    assert_eq!(curated.render_prompt(&messages, &[]), Some(rendered));
}

#[test]
fn auto_selection_over_any_tokenizer() {
    let (_, tag) = auto_codec_from_tokenizer(FakeTokenizer::chatml());
    assert_eq!(tag, "chatml");
    let (raw, tag) = auto_codec_from_tokenizer(FakeTokenizer::bare());
    assert_eq!(tag, "raw");
    assert_eq!(raw.encode("ab"), bytes("ab"));
    assert!(
        raw.render_message(role::USER, "x").is_none(),
        "a raw codec has no dialect"
    );
}

#[test]
fn control_token_markers_keep_their_literal_delimiters() {
    let tool = r#"{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}"#;
    let template = TemplateCodec::from_tokenizer(FakeTokenizer::chatml_control()).expect("template codec");
    assert_eq!(template.decode(&[260]), "", "a control token decodes to nothing");
    assert_eq!(
        template.tool_call_delimiters(),
        Some(("<tool_call>".to_string(), "</tool_call>".to_string()))
    );
    let tag = template.structural_tag(&[tool.to_string()], true).expect("a structural tag");
    assert!(tag.contains("<tool_call>") && tag.contains("</tool_call>"), "{tag}");

    let chatml = chatml_codec_from_tokenizer(FakeTokenizer::chatml_control()).expect("chatml codec");
    assert_eq!(
        chatml.tool_call_delimiters(),
        Some(("<tool_call>".to_string(), "</tool_call>".to_string()))
    );
    let raw = BundleCodec::from_tokenizer(FakeTokenizer::chatml_control());
    assert_eq!(raw.marker_literal(260), "<tool_call>");
    assert_eq!(raw.marker_literal(65), "A", "an ordinary token is its own text");
}

#[test]
fn template_codec_hands_back_reasoning_as_thinking() {
    const START: u32 = 300;
    const CHANNEL: u32 = 301;
    const MESSAGE: u32 = 302;
    const END: u32 = 303;
    const RETURN: u32 = 304;
    const CALL: u32 = 305;
    let specials = vec![
        ("<|start|>".to_string(), START),
        ("<|channel|>".to_string(), CHANNEL),
        ("<|message|>".to_string(), MESSAGE),
        ("<|end|>".to_string(), END),
        ("<|return|>".to_string(), RETURN),
        ("<|call|>".to_string(), CALL),
    ];
    let template = "{%- for message in messages %}{%- if message.role == 'assistant' %}\
{%- if \"thinking\" in message %}<|start|>assistant<|channel|>analysis<|message|>{{ message.thinking }}<|end|>{%- endif %}\
{%- if \"reasoning_content\" in message %}[rc={{ message.reasoning_content }}]{%- endif %}\
<|start|>assistant<|channel|>final<|message|>{{ message.content }}<|end|>\
{%- else %}<|start|>{{ message.role }}<|message|>{{ message.content }}<|end|>{%- endif %}{%- endfor %}\
{%- if add_generation_prompt %}<|start|>assistant{%- endif %}";
    let tok = FakeTokenizer { specials, template: template.into(), control: true, bos: None };
    let codec = TemplateCodec::from_tokenizer(std::sync::Arc::new(tok)).expect("template codec");
    let kwargs = serde_json::Map::new();
    let mut with = superfluid_daemon::codec::ChatMessage::new(superfluid_daemon::wal::role::ASSISTANT, "done");
    with.reasoning = Some("plan".into());
    let without = superfluid_daemon::codec::ChatMessage::new(superfluid_daemon::wal::role::ASSISTANT, "done");
    let user = superfluid_daemon::codec::ChatMessage::new(superfluid_daemon::wal::role::USER, "hi");
    let render = |msgs: &[superfluid_daemon::codec::ChatMessage]| -> Vec<u32> {
        codec.try_render_turns_structured_with(msgs, &[], &kwargs).expect("renders")
    };
    let contains = |hay: &[u32], needle: &[u32]| hay.windows(needle.len()).any(|w| w == needle);
    let yes = render(&[user.clone(), with]);
    assert!(contains(&yes, &bytes("plan")), "the reasoning reaches `thinking`");
    assert!(contains(&yes, &bytes("[rc=plan]")), "and `reasoning_content`");
    let mut analysis = vec![START];
    analysis.extend(bytes("assistant"));
    analysis.push(CHANNEL);
    analysis.extend(bytes("analysis"));
    analysis.push(MESSAGE);
    analysis.extend(bytes("plan"));
    analysis.push(END);
    assert!(contains(&yes, &analysis), "rendered as the analysis message the template writes");
    let no = render(&[user, without]);
    assert!(!contains(&no, &bytes("analysis")), "no reasoning: the key is absent, so no analysis message");
    assert!(!contains(&no, &bytes("[rc=")), "and no reasoning_content either");
    assert!(!contains(&no, &bytes("none")) && !contains(&no, &bytes("None")), "never a rendered none");
}

#[test]
fn template_codec_leaves_the_tool_keys_a_message_lacks_absent() {
    const HEADER: u32 = 300;
    const END_HEADER: u32 = 301;
    const EOT: u32 = 302;
    let specials = vec![
        ("<|start_header_id|>".to_string(), HEADER),
        ("<|end_header_id|>".to_string(), END_HEADER),
        ("<|eot_id|>".to_string(), EOT),
    ];
    let template = r#"{%- for message in messages %}
    {{- '[' ~ message.role ~ ' tool_calls=' ~ ('tool_calls' in message) ~ ' tool_call_id=' ~ ('tool_call_id' in message) ~ ' name=' ~ ('name' in message) ~ ']' }}
    {%- for c in message.tool_calls or [] %}{{- '[call type=' ~ c.type ~ ']' }}{%- endfor %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"#;
    let tok = FakeTokenizer { specials, template: template.into(), control: true, bos: None };
    let codec = TemplateCodec::from_tokenizer(std::sync::Arc::new(tok)).expect("template codec");
    use superfluid_daemon::codec::{ChatMessage, ToolCallMsg};
    let user = ChatMessage::new(role::USER, "What is the weather in Paris?");
    let mut call = ChatMessage::new(role::ASSISTANT, "");
    call.tool_calls.push(ToolCallMsg { id: "call_1".into(), name: "get_weather".into(), arguments: r#"{"city":"Paris"}"#.into() });
    let mut result = ChatMessage::new(role::TOOL, "18 degrees Celsius, sunny");
    result.tool_call_id = Some("call_1".into());
    let rendered = codec
        .try_render_turns_structured_with(&[user, call, result], &[], &serde_json::Map::new())
        .expect("a conversation holding a tool result renders");
    let text = String::from_utf8(rendered.iter().filter(|&&t| t < 256).map(|&t| t as u8).collect()).unwrap();
    assert!(text.contains("[user tool_calls=False tool_call_id=False name=False]"), "a user turn carries none of the tool keys: {text}");
    assert!(
        text.contains("[assistant tool_calls=True tool_call_id=False name=False][call type=function]"),
        "a call carries its calls, each typed, and no id or name it lacks: {text}"
    );
    assert!(text.contains("[tool tool_calls=False tool_call_id=True name=False]"), "a result carries its id, and no calls or name it lacks: {text}");
    assert!(text.contains(r#"{"name": "get_weather", "parameters": {"city":"#), "the call renders as Llama writes one: {text}");
    assert!(text.contains(r#"ipython"#) && text.contains(r#""18 degrees Celsius, sunny""#), "the result is rendered as the tool's turn: {text}");
    assert!(!text.contains("none") && !text.contains("None"), "never a rendered none: {text}");
}

#[test]
fn quoted_markers_in_client_content_stay_words_over_any_tokenizer() {
    const THINK: u32 = 258;
    let count = |toks: &[u32], id: u32| toks.iter().filter(|&&t| t == id).count();
    let quoted = "stop at <|im_end|> and <think> hard";

    let chatml = ChatMlCodec::from_tokenizer(FakeTokenizer::chatml()).expect("markers are atomic");
    let turn = chatml.render_message(role::USER, quoted).expect("a user turn");
    assert_eq!((count(&turn, IM_START), count(&turn, IM_END), count(&turn, THINK)), (1, 1, 0), "{turn:?}");
    assert_eq!(chatml.decode(&turn), format!("<|im_start|>user\n{quoted}<|im_end|>\n"), "the text is whole");
    assert_eq!(chatml.render_message(role::USER, "hi"), Some(user_turn("hi")));

    let tmpl = TemplateCodec::from_tokenizer(FakeTokenizer::chatml()).expect("template loads");
    let prompt = tmpl.render_prompt(&[(role::USER, quoted.to_string())], &[]).expect("a prompt");
    assert_eq!((count(&prompt, IM_START), count(&prompt, IM_END), count(&prompt, THINK)), (2, 1, 0), "{prompt:?}");
    assert_eq!(tmpl.decode(&prompt), format!("<|im_start|>user\n{quoted}<|im_end|>\n<|im_start|>assistant\n"));

    let raw = BundleCodec::from_tokenizer(FakeTokenizer::chatml());
    assert_eq!(count(&raw.encode(quoted), IM_END), 1, "encode parses the marker");
    assert_eq!(raw.encode_content(quoted), bytes(quoted), "content keeps it as words");
}

/// Qwen's rule, cut down: an assistant turn after the last user query opens
/// its (empty) reasoning block; one a later query put in history does not.
const LOOP_TEMPLATE: &str = r#"{%- set ns = namespace(query=-1) %}
{%- for message in messages %}{%- if message.role == "user" %}{%- set ns.query = loop.index0 %}{%- endif %}{%- endfor %}
{%- for message in messages %}
    {%- if message.role == "assistant" %}
        {{- '<|im_start|>assistant\n' }}
        {%- if loop.index0 > ns.query %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}
        {{- message.content }}
        {%- for tool_call in message.tool_calls or [] %}
            {{- '<tool_call>\n' + tool_call.function | tojson + '\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{%- endif %}"#;

#[test]
fn a_tool_loop_step_renders_as_the_template_renders_a_turn_no_query_follows() {
    let tok: Arc<dyn Tokenizer> = Arc::new(FakeTokenizer {
        specials: ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<|endoftext|>"]
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), 256 + i as u32))
            .collect(),
        template: LOOP_TEMPLATE.to_string(),
        control: false,
        bos: None,
    });
    let codec = ChatMlCodec::from_tokenizer(tok).expect("markers are atomic");
    let calls = vec![("lookup_code".to_string(), r#"{"account": "a-1"}"#.to_string())];
    let in_history = codec.render_assistant_with_tool_calls("", &calls).unwrap();
    let step = codec.render_assistant_with_tool_calls_after_query("", &calls).unwrap();
    let (think, end_think, nl) = (256 + 2, 256 + 3, u32::from(b'\n'));
    assert!(!in_history.contains(&think), "a turn a later query put in history has no reasoning block");
    // <|im_start|> and "assistant\n" (one token a byte here), then the opener.
    let head = 1 + "assistant\n".len();
    assert_eq!(step[..head], in_history[..head], "the same turn start");
    assert_eq!(step[head..head + 6], [think, nl, nl, end_think, nl, nl], "a step of a tool loop opens its reasoning block, as the template does");
    assert_eq!(step[head + 6..], in_history[head..], "and the same turn after it");
}

/// Qwen3.8's rule, cut down: every assistant turn opens its (empty)
/// reasoning block, in history as at the end (`preserve_thinking`).
const PRESERVING_TEMPLATE: &str = r#"{%- for message in messages %}
    {%- if message.role == "assistant" %}
        {{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' + message.content }}
        {%- for tool_call in message.tool_calls or [] %}
            {{- '<tool_call>\n' + tool_call.function | tojson + '\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{%- endif %}"#;

fn codec_with(template: &str) -> ChatMlCodec {
    let tok: Arc<dyn Tokenizer> = Arc::new(FakeTokenizer {
        specials: ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<|endoftext|>"]
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), 256 + i as u32))
            .collect(),
        template: template.to_string(),
        control: false,
        bos: None,
    });
    ChatMlCodec::from_tokenizer(tok).expect("markers are atomic")
}

#[test]
fn a_template_that_keeps_the_reasoning_block_in_history_gets_it_on_every_assistant_turn() {
    let codec = codec_with(PRESERVING_TEMPLATE);
    let calls = vec![("lookup_code".to_string(), r#"{"account": "a-1"}"#.to_string())];
    let (think, end_think, nl) = (256 + 2, 256 + 3, u32::from(b'\n'));
    let opener = [think, nl, nl, end_think, nl, nl];
    let head = 1 + "assistant\n".len();
    for turn in [
        codec.render_assistant_with_tool_calls("", &calls).unwrap(),
        codec.render_assistant_with_tool_calls_after_query("", &calls).unwrap(),
        codec.render_message(role::ASSISTANT, "the code is X").unwrap(),
    ] {
        assert_eq!(turn[head..head + 6], opener, "the block the template keeps, in history too");
    }
    let plain = codec_with(LOOP_TEMPLATE).render_message(role::ASSISTANT, "the code is X").unwrap();
    assert!(!plain.contains(&think), "a template that drops it in history: an answer in history has none");
}
