//! HF `tokenizer.json` as a [`superfluid_engine::Tokenizer`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use superfluid_engine::Tokenizer;

const MAX_VOCAB_PADDING: u64 = 1 << 20;

#[derive(Debug, thiserror::Error)]
pub enum HfTokenizerError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("tokenizer.json: {0}")]
    Tokenizer(String),
    #[error("the bundle carries no HF tokenizer: {0}")]
    NotHf(&'static str),
    #[error("tokenizer.json: unsupported decoder `{0}` (per-token decoding needs ByteLevel, Metaspace, Replace, ByteFallback, WordPiece, BPEDecoder, Fuse, Strip or a Sequence of them)")]
    Decoder(String),
    #[error(
        "no EOS token: none of tokenizer_config.json `eos_token`, generation_config.json \
         `eos_token_id` and config.json `eos_token_id` names one"
    )]
    NoEos,
}

#[derive(Debug, Clone, PartialEq)]
enum Step {
    ByteLevel,
    ByteFallback,
    Replace { from: String, to: String },
    WordPiece { prefix: String },
    BpeSuffix { suffix: String },
    Strip { content: char, start: usize, stop: usize },
}

pub struct HfTokenizer {
    tk: tokenizers::Tokenizer,
    vocab_size: u32,
    added: HashMap<u32, (String, bool)>,
    specials: Vec<(String, u32)>,
    bos: Option<u32>,
    eos: u32,
    chat_template: String,
    chat_templates: HashMap<String, String>,
    steps: Vec<Step>,
    unicode_to_byte: HashMap<char, u8>,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTokenizer")
            .field("vocab_size", &self.vocab_size)
            .field("added", &self.added.len())
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .field("steps", &self.steps)
            .field("chat_template_len", &self.chat_template.len())
            .finish()
    }
}

pub fn byte_to_unicode() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!'..=b'~').map(u32::from).collect();
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = vec!['\0'; 256];
    for (b, c) in bs.into_iter().zip(cs) {
        table[b as usize] = char::from_u32(c).expect("byte-level alphabet is valid unicode");
    }
    table
}

fn bundle_header(path: &Path) -> Result<serde_json::Value, HfTokenizerError> {
    use std::io::Read;
    let io = |source| HfTokenizerError::Io { path: path.to_path_buf(), source };
    let mut f = std::fs::File::open(path).map_err(io)?;
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() || &magic != b"BASE" {
        return Err(HfTokenizerError::NotHf("not a .base bundle"));
    }
    let mut head = [0u8; 12];
    f.read_exact(&mut head).map_err(io)?;
    let len = u64::from_le_bytes(head[4..12].try_into().expect("8 bytes"));
    let meta = f.metadata().map_err(io)?;
    if len > meta.len() {
        return Err(HfTokenizerError::NotHf("a header longer than the file"));
    }
    let mut text = vec![0u8; len as usize];
    f.read_exact(&mut text).map_err(io)?;
    serde_json::from_slice(&text).map_err(|source| HfTokenizerError::Json { path: path.to_path_buf(), source })
}

fn read_json(path: &Path) -> Result<Option<serde_json::Value>, HfTokenizerError> {
    match std::fs::read(path) {
        Ok(bytes) => {
            serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|source| HfTokenizerError::Json {
                    path: path.to_path_buf(),
                    source,
                })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(HfTokenizerError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn model_cfg_key<'a>(cfg: Option<&'a serde_json::Value>, key: &str) -> Option<&'a serde_json::Value> {
    let c = cfg?;
    c.get(key).or_else(|| c.get("text_config").and_then(|t| t.get(key)))
}

fn read_text(path: &Path) -> Result<Option<String>, HfTokenizerError> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(HfTokenizerError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn chat_template_files(dir: &Path) -> Result<(Option<String>, HashMap<String, String>), HfTokenizerError> {
    let template = read_text(&dir.join("chat_template.jinja"))?;
    let mut named = HashMap::new();
    let extra = dir.join("additional_chat_templates");
    if extra.is_dir() {
        let entries = std::fs::read_dir(&extra).map_err(|source| HfTokenizerError::Io { path: extra.clone(), source })?;
        for entry in entries {
            let path = entry.map_err(|source| HfTokenizerError::Io { path: extra.clone(), source })?.path();
            if path.extension().and_then(|x| x.to_str()) != Some("jinja") {
                continue;
            }
            if let (Some(name), Some(t)) = (path.file_stem().and_then(|x| x.to_str()), read_text(&path)?) {
                named.insert(name.to_string(), t);
            }
        }
    }
    Ok((template, named))
}

fn token_text(v: Option<&serde_json::Value>) -> Option<String> {
    match v? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}

fn chat_template(cfg: Option<&serde_json::Value>) -> (String, HashMap<String, String>) {
    let Some(v) = cfg.and_then(|c| c.get("chat_template")) else {
        return (String::new(), HashMap::new());
    };
    let ordered: Vec<(String, String)> = match v {
        serde_json::Value::String(s) => return (s.clone(), HashMap::new()),
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|i| {
                let name = i.get("name")?.as_str()?;
                let template = i.get("template")?.as_str()?;
                Some((name.to_string(), template.to_string()))
            })
            .collect(),
        serde_json::Value::Object(map) => map
            .iter()
            .filter_map(|(name, t)| Some((name.clone(), t.as_str()?.to_string())))
            .collect(),
        _ => return (String::new(), HashMap::new()),
    };
    let default = ordered
        .iter()
        .find(|(n, _)| n == "default")
        .or_else(|| ordered.first())
        .map(|(_, t)| t.clone())
        .unwrap_or_default();
    (default, ordered.into_iter().collect())
}

fn eos_id(v: Option<&serde_json::Value>) -> Option<u32> {
    match v? {
        serde_json::Value::Number(n) => n.as_u64().map(|n| n as u32),
        serde_json::Value::Array(a) => a.first().and_then(|n| n.as_u64()).map(|n| n as u32),
        _ => None,
    }
}

fn byte_level_pretokenizer(tokenizer_json: &serde_json::Value) -> bool {
    fn is_byte_level(v: Option<&serde_json::Value>) -> bool {
        let Some(v) = v else { return false };
        match v.get("type").and_then(|t| t.as_str()) {
            Some("ByteLevel") => true,
            Some("Sequence") => v
                .get("pretokenizers")
                .and_then(|a| a.as_array())
                .into_iter()
                .flatten()
                .any(|d| is_byte_level(Some(d))),
            _ => false,
        }
    }
    is_byte_level(tokenizer_json.get("pre_tokenizer"))
}

fn parse_decoder(
    v: Option<&serde_json::Value>,
    out: &mut Vec<Step>,
    fused: &mut bool,
) -> Result<(), HfTokenizerError> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(());
    };
    let unsupported = |t: &str| Err(HfTokenizerError::Decoder(t.to_string()));
    match v.get("type").and_then(|t| t.as_str()) {
        Some("Sequence") => {
            for d in v.get("decoders").and_then(|a| a.as_array()).into_iter().flatten() {
                parse_decoder(Some(d), out, fused)?;
            }
            Ok(())
        }
        Some("Fuse") => {
            *fused = true;
            Ok(())
        }
        Some(_) if *fused => Ok(()),
        Some("ByteLevel") => {
            out.push(Step::ByteLevel);
            Ok(())
        }
        Some("ByteFallback") => {
            out.push(Step::ByteFallback);
            Ok(())
        }
        Some("Metaspace") => {
            let from = v.get("replacement").and_then(|r| r.as_str()).unwrap_or("\u{2581}");
            out.push(Step::Replace { from: from.to_string(), to: " ".to_string() });
            Ok(())
        }
        Some("Replace") => {
            let Some(from) = v.get("pattern").and_then(|p| p.get("String")).and_then(|s| s.as_str()) else {
                return unsupported("Replace(Regex)");
            };
            let to = v.get("content").and_then(|c| c.as_str()).unwrap_or_default();
            out.push(Step::Replace { from: from.to_string(), to: to.to_string() });
            Ok(())
        }
        Some("WordPiece") => {
            let prefix = v.get("prefix").and_then(|p| p.as_str()).unwrap_or("##");
            out.push(Step::WordPiece { prefix: prefix.to_string() });
            Ok(())
        }
        Some("BPEDecoder") => {
            let suffix = v.get("suffix").and_then(|p| p.as_str()).unwrap_or("</w>");
            out.push(Step::BpeSuffix { suffix: suffix.to_string() });
            Ok(())
        }
        Some("Strip") => {
            let content = v.get("content").and_then(|c| c.as_str()).and_then(|c| c.chars().next()).unwrap_or(' ');
            let start = v.get("start").and_then(|n| n.as_u64()).unwrap_or(0) as usize;
            let stop = v.get("stop").and_then(|n| n.as_u64()).unwrap_or(0) as usize;
            out.push(Step::Strip { content, start, stop });
            Ok(())
        }
        Some(other) => unsupported(other),
        None => unsupported("untyped"),
    }
}

fn byte_fallback(piece: &str) -> Option<u8> {
    if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
        u8::from_str_radix(&piece[3..5], 16).ok()
    } else {
        None
    }
}

impl HfTokenizer {
    pub fn load(path: &Path) -> Result<HfTokenizer, HfTokenizerError> {
        let dir = if path.is_file() {
            path.parent().unwrap_or(path).to_path_buf()
        } else {
            path.to_path_buf()
        };
        let tok_path = dir.join("tokenizer.json");
        let tk = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| HfTokenizerError::Tokenizer(e.to_string()))?;
        let tok_json = read_json(&tok_path)?.unwrap_or(serde_json::Value::Null);
        let cfg = read_json(&dir.join("tokenizer_config.json"))?;
        let gen = read_json(&dir.join("generation_config.json"))?;
        let model_cfg = read_json(&dir.join("config.json"))?;
        let (file_template, file_named) = chat_template_files(&dir)?;
        let template_json = read_json(&dir.join("chat_template.json"))?;
        Self::build(tk, tok_json, cfg, gen, model_cfg, file_template, file_named, template_json)
    }

    pub fn from_bundle(path: &Path) -> Result<HfTokenizer, HfTokenizerError> {
        let header = bundle_header(path)?;
        let t = header.get("tokenizer").ok_or(HfTokenizerError::NotHf("the header has no tokenizer"))?;
        if t.get("tokenizer_type").and_then(serde_json::Value::as_str).is_some_and(|k| k != "hf") {
            return Err(HfTokenizerError::NotHf("its tokenizer is not HF's"));
        }
        let tok_json = t.get("tokenizer.json").cloned().ok_or(HfTokenizerError::NotHf("no tokenizer.json"))?;
        let tk = tokenizers::Tokenizer::from_str(&tok_json.to_string())
            .map_err(|e| HfTokenizerError::Tokenizer(e.to_string()))?;
        let cfg = t.get("tokenizer_config.json").cloned();
        let template = t.get("tokenizer.chat_template").and_then(serde_json::Value::as_str).map(str::to_string);
        let model_cfg = header.get("config").cloned();
        Self::build(tk, tok_json, cfg, None, model_cfg, template, HashMap::new(), None)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        tk: tokenizers::Tokenizer,
        tok_json: serde_json::Value,
        cfg: Option<serde_json::Value>,
        gen: Option<serde_json::Value>,
        model_cfg: Option<serde_json::Value>,
        file_template: Option<String>,
        file_named: HashMap<String, String>,
        template_json: Option<serde_json::Value>,
    ) -> Result<HfTokenizer, HfTokenizerError> {
        let mut tk = tk;
        let _ = tk.with_truncation(None);
        tk.with_padding(None);

        let eos = token_text(cfg.as_ref().and_then(|c| c.get("eos_token")))
            .and_then(|s| tk.token_to_id(&s))
            .or_else(|| eos_id(gen.as_ref().and_then(|g| g.get("eos_token_id"))))
            .or_else(|| eos_id(model_cfg_key(model_cfg.as_ref(), "eos_token_id")))
            .ok_or(HfTokenizerError::NoEos)?;
        let bos = token_text(cfg.as_ref().and_then(|c| c.get("bos_token")))
            .and_then(|s| tk.token_to_id(&s));
        let (mut chat_template, mut chat_templates) = chat_template(cfg.as_ref());
        if let Some(t) = file_template {
            chat_template = t;
        } else if chat_template.is_empty() {
            let (t, named) = crate::chat_template(template_json.as_ref());
            if !t.is_empty() {
                chat_template = t;
                chat_templates = named;
            }
        }
        chat_templates.extend(file_named);

        let tokenizer_vocab = tk.get_vocab_size(true) as u32;
        let padded = match model_cfg_key(model_cfg.as_ref(), "vocab_size").and_then(|v| v.as_u64()) {
            Some(v) if v > tokenizer_vocab as u64 + MAX_VOCAB_PADDING => {
                return Err(HfTokenizerError::Tokenizer(format!(
                    "its vocabulary is {tokenizer_vocab} tokens, and config.json's vocab_size says {v}: not that vocabulary padded"
                )));
            }
            Some(v) => v as u32,
            None => 0,
        };
        let vocab_size = tokenizer_vocab.max(padded);

        let mut added_list: Vec<(u32, String, bool)> = tk
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, t)| (id, t.content, t.special))
            .collect();
        added_list.sort_by_key(|(id, _, _)| *id);
        let specials: Vec<(String, u32)> = added_list
            .iter()
            .map(|(id, s, _)| (s.clone(), *id))
            .collect();
        let added = added_list
            .into_iter()
            .map(|(id, s, sp)| (id, (s, sp)))
            .collect();

        let mut steps = Vec::new();
        let mut fused = false;
        parse_decoder(tok_json.get("decoder"), &mut steps, &mut fused)?;
        if steps.is_empty() {
            if byte_level_pretokenizer(&tok_json) {
                steps.push(Step::ByteLevel);
            } else if tok_json.get("decoder").is_none_or(|d| d.is_null()) {
                steps.push(Step::Replace { from: "\u{2581}".to_string(), to: " ".to_string() });
                steps.push(Step::ByteFallback);
            }
        }
        let unicode_to_byte = byte_to_unicode()
            .into_iter()
            .enumerate()
            .map(|(b, c)| (c, b as u8))
            .collect();
        Ok(HfTokenizer {
            tk,
            vocab_size,
            added,
            specials,
            bos,
            eos,
            chat_template,
            chat_templates,
            steps,
            unicode_to_byte,
        })
    }

    pub fn is_byte_level(&self) -> bool {
        self.steps.contains(&Step::ByteLevel)
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match self.tk.encode(text, true) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => Vec::new(),
        }
    }

    fn token_bytes(&self, token: u32) -> Vec<u8> {
        let piece = match self.added.get(&token) {
            Some((_, true)) => return Vec::new(),
            Some((text, false)) => text.clone(),
            None => match self.tk.id_to_token(token) {
                Some(piece) => piece,
                None => return Vec::new(),
            },
        };
        let mut text = piece;
        for step in &self.steps {
            match step {
                Step::ByteLevel => {
                    let mut out = Vec::with_capacity(text.len());
                    for c in text.chars() {
                        match self.unicode_to_byte.get(&c) {
                            Some(b) => out.push(*b),
                            None => out.extend_from_slice(c.to_string().as_bytes()),
                        }
                    }
                    return out;
                }
                Step::ByteFallback => {
                    if let Some(b) = byte_fallback(&text) {
                        return vec![b];
                    }
                }
                Step::Replace { from, to } => text = text.replace(from.as_str(), to),
                Step::WordPiece { prefix } => {
                    text = match text.strip_prefix(prefix.as_str()) {
                        Some(rest) => rest.to_string(),
                        None => format!(" {text}"),
                    }
                }
                Step::BpeSuffix { suffix } => text = text.replace(suffix.as_str(), " "),
                Step::Strip { content, start, stop } => {
                    let lead = text.chars().take(*start).take_while(|c| c == content).count();
                    let mut t: String = text.chars().skip(lead).collect();
                    let trail = t.chars().rev().take(*stop).take_while(|c| c == content).count();
                    if trail > 0 {
                        let keep = t.chars().count() - trail;
                        t = t.chars().take(keep).collect();
                    }
                    text = t;
                }
            }
        }
        text.into_bytes()
    }

    fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    fn special_tokens(&self) -> Vec<(String, u32)> {
        self.specials.clone()
    }

    fn bos_token(&self) -> Option<u32> {
        self.bos
    }

    fn eos_token(&self) -> u32 {
        self.eos
    }

    fn chat_template_jinja(&self) -> String {
        self.chat_template.clone()
    }

    fn chat_template_named(&self, name: &str) -> Option<String> {
        self.chat_templates.get(name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::AddedToken;

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let d = std::env::temp_dir().join(format!(
            "superfluid-tokenizer-hf-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_synthetic(dir: &Path) {
        let alphabet = byte_to_unicode();
        let mut vocab = tokenizers::models::bpe::Vocab::default();
        for (i, c) in alphabet.iter().enumerate() {
            vocab.insert(c.to_string(), i as u32);
        }
        vocab.insert("hi".into(), 256);
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![("h".into(), "i".into())])
            .build()
            .unwrap();
        let mut tk = tokenizers::Tokenizer::new(bpe);
        tk.with_pre_tokenizer(Some(
            ByteLevel::default()
                .add_prefix_space(false)
                .use_regex(false),
        ));
        tk.with_decoder(Some(ByteLevel::default()));
        tk.with_post_processor(Some(ByteLevel::default()));
        tk.add_special_tokens(&[
            AddedToken::from("<|im_start|>", true),
            AddedToken::from("<|im_end|>", true),
        ]);
        tk.add_tokens(&[AddedToken::from("<tool_call>", false)]);
        tk.save(dir.join("tokenizer.json"), false).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({
                "bos_token": null,
                "eos_token": "<|im_end|>",
                "chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}",
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({ "vocab_size": 300 }).to_string(),
        )
        .unwrap();
    }

    fn write_json_checkpoint(dir: &Path, model: serde_json::Value, decoder: serde_json::Value, cfg: serde_json::Value) {
        let tok = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": decoder,
            "model": model,
        });
        std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
    }

    fn bpe_model(pieces: &[&str]) -> serde_json::Value {
        let vocab: serde_json::Map<String, serde_json::Value> =
            pieces.iter().enumerate().map(|(i, p)| (p.to_string(), serde_json::json!(i))).collect();
        serde_json::json!({
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": [],
        })
    }

    fn write_bundle(path: &Path, header: &serde_json::Value) {
        let text = header.to_string();
        let mut bytes = b"BASE".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn a_bundles_embedded_tokenizer_is_the_checkpoints() {
        let dir = scratch_dir("bundle");
        write_synthetic(&dir);
        let from_dir = HfTokenizer::load(&dir).unwrap();
        let read = |f: &str| serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(dir.join(f)).unwrap()).unwrap();
        let bundle = dir.join("model.base");
        write_bundle(
            &bundle,
            &serde_json::json!({
                "schema": 1,
                "config": read("config.json"),
                "tokenizer": {
                    "tokenizer_type": "hf",
                    "tokenizer.json": read("tokenizer.json"),
                    "tokenizer_config.json": read("tokenizer_config.json"),
                    "tokenizer.chat_template": "{{ messages[0].content }}",
                },
            }),
        );
        let from_bundle = HfTokenizer::from_bundle(&bundle).unwrap();
        assert_eq!(from_bundle.encode("hi <|im_end|>"), from_dir.encode("hi <|im_end|>"));
        assert_eq!(from_bundle.vocab_size(), 300, "the header's config pads the vocabulary");
        assert_eq!(from_bundle.eos_token(), from_dir.eos_token());
        assert_eq!(from_bundle.special_tokens(), from_dir.special_tokens());
        assert_eq!(from_bundle.chat_template_jinja(), "{{ messages[0].content }}", "the bundle's template wins");

        std::fs::write(dir.join("not.base"), b"GGUF\x03\x00\x00\x00").unwrap();
        assert!(matches!(HfTokenizer::from_bundle(&dir.join("not.base")), Err(HfTokenizerError::NotHf(_))));
        write_bundle(&bundle, &serde_json::json!({"schema": 1, "tokenizer": {"tokenizer_type": "spm"}}));
        assert!(matches!(HfTokenizer::from_bundle(&bundle), Err(HfTokenizerError::NotHf(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sentencepiece_style_decoder_sequence_yields_exact_bytes() {
        let dir = scratch_dir("sp");
        write_json_checkpoint(
            &dir,
            bpe_model(&["<s>", "▁the", "cat", "<0xE2>", "<0x9C>", "<0x93>", "▁"]),
            serde_json::json!({"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0},
            ]}),
            serde_json::json!({"eos_token": "<s>"}),
        );
        let tok = HfTokenizer::load(&dir).expect("load");
        assert!(!tok.is_byte_level());
        assert_eq!(tok.token_bytes(1), b" the".to_vec(), "a word-initial piece keeps its space");
        assert_eq!(tok.token_bytes(2), b"cat".to_vec());
        assert_eq!(tok.token_bytes(3), vec![0xE2], "byte fallback is the byte, not a lossy char");
        assert_eq!(tok.token_bytes(6), b" ".to_vec(), "Strip after Fuse touches no token");
    }

    #[test]
    fn an_added_token_that_is_not_special_goes_through_the_decoder() {
        let dir = scratch_dir("added");
        let added = |id: u32, content: &str, special: bool| {
            serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": special})
        };
        let tok = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [added(3, "<eos>", true), added(4, "▁▁▁▁", false), added(5, "<table>", false)],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ]},
            "model": bpe_model(&["<s>", "▁the", "cat"]),
        });
        std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), serde_json::json!({"eos_token": "<eos>"}).to_string()).unwrap();
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.token_bytes(4), b"    ".to_vec(), "four spaces of indentation, not four `▁`");
        assert_eq!(tok.token_bytes(5), b"<table>".to_vec(), "an added token the decoder has nothing to say about is itself");
        assert_eq!(tok.token_bytes(3), Vec::<u8>::new(), "a special token is not text");
        assert_eq!(tok.token_bytes(1), b" the".to_vec());
    }

    #[test]
    fn a_stored_truncation_or_padding_does_not_apply_to_prompts() {
        let dir = scratch_dir("trunc");
        let tok = serde_json::json!({
            "version": "1.0",
            "truncation": {"direction": "Right", "max_length": 3, "strategy": "LongestFirst", "stride": 0},
            "padding": {"strategy": {"Fixed": 8}, "direction": "Right", "pad_to_multiple_of": null, "pad_id": 0, "pad_type_id": 0, "pad_token": "<s>"},
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}]},
            "model": {"type": "WordLevel", "vocab": {"<s>": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}, "unk_token": "<s>"},
        });
        std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), serde_json::json!({"eos_token": "<s>"}).to_string()).unwrap();
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.encode("b c d e f"), [1, 2, 3, 4, 5], "all of it, and nothing after it");
        assert_eq!(tok.encode("b"), [1]);
    }

    #[test]
    fn wordpiece_decoder_joins_continuations_and_spaces_words() {
        let dir = scratch_dir("wp");
        let vocab: serde_json::Map<String, serde_json::Value> = ["[UNK]", "play", "##ing", "[SEP]"]
            .iter()
            .enumerate()
            .map(|(i, p)| (p.to_string(), serde_json::json!(i)))
            .collect();
        write_json_checkpoint(
            &dir,
            serde_json::json!({"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": vocab}),
            serde_json::json!({"type": "WordPiece", "prefix": "##", "cleanup": true}),
            serde_json::json!({"eos_token": "[SEP]"}),
        );
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.token_bytes(1), b" play".to_vec(), "a word start carries its space");
        assert_eq!(tok.token_bytes(2), b"ing".to_vec(), "a continuation drops its prefix");
    }

    #[test]
    fn bpe_end_of_word_suffix_becomes_the_space_after_the_word() {
        let dir = scratch_dir("bpew");
        write_json_checkpoint(
            &dir,
            bpe_model(&["<eos>", "the</w>", "th"]),
            serde_json::json!({"type": "BPEDecoder", "suffix": "</w>"}),
            serde_json::json!({"eos_token": "<eos>"}),
        );
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.token_bytes(1), b"the ".to_vec());
        assert_eq!(tok.token_bytes(2), b"th".to_vec());
    }

    #[test]
    fn an_unsupported_decoder_is_refused_at_load() {
        let dir = scratch_dir("ctc");
        write_json_checkpoint(
            &dir,
            bpe_model(&["<eos>", "a"]),
            serde_json::json!({"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "|", "cleanup": true}),
            serde_json::json!({"eos_token": "<eos>"}),
        );
        match HfTokenizer::load(&dir) {
            Err(HfTokenizerError::Decoder(t)) => assert_eq!(t, "CTC"),
            other => panic!("expected a decoder refusal, got {other:?}"),
        }
    }

    #[test]
    fn eos_falls_back_to_the_model_config() {
        let dir = scratch_dir("eos");
        write_json_checkpoint(
            &dir,
            bpe_model(&["<pad>", "<end>", "a"]),
            serde_json::json!({"type": "ByteFallback"}),
            serde_json::json!({}),
        );
        assert!(matches!(HfTokenizer::load(&dir), Err(HfTokenizerError::NoEos)));
        std::fs::write(dir.join("config.json"), serde_json::json!({"eos_token_id": [1, 0]}).to_string()).unwrap();
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.eos_token(), 1, "config.json names the primary EOS");
    }

    #[test]
    fn object_form_named_templates_are_read_too() {
        let dir = scratch_dir("named-obj");
        write_json_checkpoint(
            &dir,
            bpe_model(&["<eos>", "a"]),
            serde_json::json!({"type": "ByteFallback"}),
            serde_json::json!({"eos_token": "<eos>", "chat_template": {"default": "D", "tool_use": "T"}}),
        );
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.chat_template_jinja(), "D");
        assert_eq!(tok.chat_template_named("tool_use").as_deref(), Some("T"));
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({"eos_token": "<eos>", "chat_template": {"chat": "C", "tool_use": "T"}}).to_string(),
        )
        .unwrap();
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.chat_template_jinja(), "C");
        assert_eq!(tok.chat_template_named("tool_use").as_deref(), Some("T"));
    }

    #[test]
    fn template_files_beside_the_config_are_read() {
        let dir = scratch_dir("jinja-files");
        write_json_checkpoint(
            &dir,
            bpe_model(&["<eos>", "a"]),
            serde_json::json!({"type": "ByteFallback"}),
            serde_json::json!({"eos_token": "<eos>"}),
        );
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.chat_template_jinja(), "", "no template anywhere yet");
        std::fs::write(dir.join("chat_template.jinja"), "{{ messages }}").unwrap();
        std::fs::create_dir_all(dir.join("additional_chat_templates")).unwrap();
        std::fs::write(dir.join("additional_chat_templates/tool_use.jinja"), "T").unwrap();
        std::fs::write(dir.join("additional_chat_templates/notes.txt"), "ignored").unwrap();
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.chat_template_jinja(), "{{ messages }}");
        assert_eq!(tok.chat_template_named("tool_use").as_deref(), Some("T"));
        assert_eq!(tok.chat_template_named("notes"), None);
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({"eos_token": "<eos>", "chat_template": "from-config"}).to_string(),
        )
        .unwrap();
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.chat_template_jinja(), "{{ messages }}");
    }

    #[test]
    fn chat_template_json_sidecar_is_a_fallback() {
        let dir = scratch_dir("jinja-json");
        write_json_checkpoint(
            &dir,
            bpe_model(&["<eos>", "a"]),
            serde_json::json!({"type": "ByteFallback"}),
            serde_json::json!({"eos_token": "<eos>"}),
        );
        std::fs::write(
            dir.join("chat_template.json"),
            serde_json::json!({"chat_template": [{"name": "default", "template": "D"}, {"name": "rag", "template": "R"}]}).to_string(),
        )
        .unwrap();
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.chat_template_jinja(), "D");
        assert_eq!(tok.chat_template_named("rag").as_deref(), Some("R"));
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({"eos_token": "<eos>", "chat_template": "C"}).to_string(),
        )
        .unwrap();
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.chat_template_jinja(), "C");
        assert_eq!(tok.chat_template_named("rag"), None);
    }

    #[test]
    fn nested_text_config_widens_the_id_space() {
        let dir = scratch_dir("text-config");
        write_json_checkpoint(
            &dir,
            bpe_model(&["<eos>", "a"]),
            serde_json::json!({"type": "ByteFallback"}),
            serde_json::json!({"eos_token": "<eos>"}),
        );
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({"model_type": "gemma3", "text_config": {"vocab_size": 300}}).to_string(),
        )
        .unwrap();
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.vocab_size(), 300, "the id space is the padded one");
        assert!(tok.token_bytes(299).is_empty(), "padding ids decode to nothing");
        let honest = std::fs::read_to_string(dir.join("config.json")).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({"model_type": "gemma3", "text_config": {"vocab_size": 4_000_000_000u64}}).to_string(),
        )
        .unwrap();
        let why = HfTokenizer::load(&dir).unwrap_err().to_string();
        assert!(why.contains("vocab_size says 4000000000"), "{why}");
        std::fs::write(dir.join("config.json"), honest).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), serde_json::json!({}).to_string()).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({"text_config": {"vocab_size": 300, "eos_token_id": 0}}).to_string(),
        )
        .unwrap();
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.eos_token(), 0);
    }

    #[test]
    fn named_templates_keep_every_name() {
        let dir = scratch_dir("named");
        write_json_checkpoint(
            &dir,
            bpe_model(&["<eos>", "a"]),
            serde_json::json!({"type": "ByteFallback"}),
            serde_json::json!({"eos_token": "<eos>", "chat_template": [
                {"name": "tool_use", "template": "T"},
                {"name": "default", "template": "D"},
            ]}),
        );
        let tok = HfTokenizer::load(&dir).expect("load");
        assert_eq!(tok.chat_template_jinja(), "D");
        assert_eq!(tok.chat_template_named("tool_use").as_deref(), Some("T"));
        assert_eq!(tok.chat_template_named("rag"), None);
    }

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let t = byte_to_unicode();
        let set: std::collections::HashSet<char> = t.iter().copied().collect();
        assert_eq!(set.len(), 256);
        assert_eq!(t[b'a' as usize], 'a');
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
    }

    #[test]
    fn synthetic_byte_level_checkpoint_roundtrips_exact_bytes() {
        let dir = scratch_dir("synthetic");
        write_synthetic(&dir);
        let tok = HfTokenizer::load(&dir).expect("load");
        assert!(tok.is_byte_level());
        assert_eq!(tok.vocab_size(), 300);
        assert!(tok.token_bytes(299).is_empty());
        for b in 0u32..256 {
            assert_eq!(tok.token_bytes(b), vec![b as u8], "byte {b}");
        }
        assert_eq!(tok.token_bytes(256), b"hi".to_vec());
        assert_eq!(
            tok.special_tokens(),
            vec![
                ("<|im_start|>".to_string(), 257),
                ("<|im_end|>".to_string(), 258),
                ("<tool_call>".to_string(), 259)
            ]
        );
        assert_eq!(
            tok.encode("<|im_start|>hi<|im_end|><tool_call>"),
            vec![257, 256, 258, 259]
        );
        assert!(tok.token_bytes(257).is_empty());
        assert_eq!(tok.token_bytes(259), b"<tool_call>".to_vec());
        assert_eq!(tok.bos_token(), None);
        assert_eq!(tok.eos_token(), 258);
        assert!(tok.chat_template_jinja().contains("<|im_start|>"));
        let ids = tok.encode("é");
        let bytes: Vec<u8> = ids.iter().flat_map(|&t| tok.token_bytes(t)).collect();
        assert_eq!(bytes, "é".as_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_eos_is_a_typed_error() {
        let dir = scratch_dir("noeos");
        write_synthetic(&dir);
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        assert!(matches!(
            HfTokenizer::load(&dir),
            Err(HfTokenizerError::NoEos)
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn qwen3_dir() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("SUPERFLUID_TEST_HF_TOKENIZER") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var("HOME").ok()?;
        let snaps =
            PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
        std::fs::read_dir(snaps)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.join("tokenizer.json").is_file())
    }

    #[test]
    fn qwen3_from_the_hf_cache_when_present() {
        let Some(dir) = qwen3_dir() else {
            eprintln!("SKIP: no Qwen3-0.6B tokenizer in the HF cache");
            return;
        };
        let tok = HfTokenizer::load(&dir).expect("load qwen3");
        assert!(tok.is_byte_level());
        assert_eq!(
            tok.encode("<|im_start|>user\nhi<|im_end|>"),
            vec![151644, 872, 198, 6023, 151645]
        );
        assert_eq!(tok.token_bytes(198), b"\n".to_vec());
        assert!(
            tok.token_bytes(151644).is_empty(),
            "control tokens are structure, not text"
        );
        assert_eq!(
            tok.token_bytes(151657),
            b"<tool_call>".to_vec(),
            "non-special added tokens are text"
        );
        assert_eq!(tok.eos_token(), 151645);
        assert_eq!(tok.bos_token(), None);
        assert!(tok
            .special_tokens()
            .contains(&("<|im_start|>".to_string(), 151644)));
        assert!(tok.chat_template_jinja().contains("<|im_start|>"));
        assert!(tok.vocab_size() >= 151669, "vocab {}", tok.vocab_size());
    }
}
