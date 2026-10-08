//! `<worker> check [--model <path>]`.

use std::path::PathBuf;

use superfluid_engine::artifact::{Device, Format, Rule, RuntimeInfo, TokenizerSource};
use serde_json::{json, Value};

pub struct ModelCheck {
    pub path: PathBuf,
    pub reads: Result<(), String>,
    pub facts: Option<Result<String, String>>,
    pub sizing: Option<Result<String, String>>,
}

pub struct Report {
    pub info: RuntimeInfo,
    pub version: Option<String>,
    pub available: Result<(), String>,
    pub devices: Vec<Device>,
    pub capabilities: String,
    pub model: Option<ModelCheck>,
    pub pull: Option<Vec<(String, bool, String)>>,
    pub named: Option<String>,
}

fn verdict(r: &Result<(), String>) -> Value {
    match r {
        Ok(()) => Value::Bool(true),
        Err(why) => Value::String(why.clone()),
    }
}

fn parsed(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or(Value::Null)
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|e| e.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

pub fn format_json(f: &Format) -> Value {
    let mut v = json!({"id": f.id, "describe": f.describe});
    match &f.rule {
        Rule::File { extensions, magic } => {
            v["file"] = json!({"extensions": extensions});
            if let Some(m) = magic {
                v["file"]["magic"] = json!(String::from_utf8_lossy(m));
            }
        }
        Rule::Directory { files, extensions } => v["directory"] = json!({"files": files, "extensions": extensions}),
    }
    v
}

pub fn formats_from(v: &Value) -> Vec<Format> {
    let one = |f: &Value| -> Option<Format> {
        let rule = if let Some(file) = f.get("file") {
            Rule::File {
                extensions: strings(&file["extensions"]),
                magic: file["magic"].as_str().map(|m| m.as_bytes().to_vec()),
            }
        } else {
            let d = f.get("directory")?;
            Rule::Directory { files: strings(&d["files"]), extensions: strings(&d["extensions"]) }
        };
        Some(Format { id: f["id"].as_str()?.to_string(), describe: f["describe"].as_str()?.to_string(), rule })
    };
    v.as_array().map(|a| a.iter().filter_map(one).collect()).unwrap_or_default()
}

pub fn tokenizer_json(t: &TokenizerSource) -> Value {
    match t {
        TokenizerSource::Library(name) => json!({"from": "library", "name": name}),
        TokenizerSource::HuggingFace => json!({"from": "huggingface"}),
        TokenizerSource::Engine => json!({"from": "engine"}),
    }
}

pub fn tokenizer_from(v: &Value) -> Option<TokenizerSource> {
    match v["from"].as_str()? {
        "library" => Some(TokenizerSource::Library(v["name"].as_str()?.to_string())),
        "huggingface" => Some(TokenizerSource::HuggingFace),
        "engine" => Some(TokenizerSource::Engine),
        _ => None,
    }
}

pub fn device_json(d: &Device) -> Value {
    json!({"backend": d.backend, "name": d.name, "memory": d.memory})
}

pub fn devices_from(v: &Value) -> Vec<Device> {
    let one = |d: &Value| -> Option<Device> {
        Some(Device {
            backend: d["backend"].as_str()?.to_string(),
            name: d["name"].as_str().unwrap_or_default().to_string(),
            memory: d["memory"].as_u64().unwrap_or(0),
        })
    };
    v.as_array().map(|a| a.iter().filter_map(one).collect()).unwrap_or_default()
}

impl Report {
    pub fn to_json(&self) -> Value {
        let mut v = json!({
            "runtime": {"id": self.info.id, "version": self.version, "aliases": self.info.aliases},
            "link_w": [crate::server::PROTO_VERSION],
            "available": verdict(&self.available),
            "formats": self.info.formats.iter().map(format_json).collect::<Vec<_>>(),
            "tokenizer": tokenizer_json(&self.info.tokenizer),
            "devices": self.devices.iter().map(device_json).collect::<Vec<_>>(),
            "capabilities": parsed(&self.capabilities),
        });
        if let Some(options) = &self.pull {
            v["pull"] = json!({
                "options": options.iter().map(|(name, value, about)| json!({"name": name, "value": value, "about": about})).collect::<Vec<_>>(),
            });
        }
        if let Some(by) = &self.named {
            v["named"] = json!(by);
        }
        if let Some(m) = &self.model {
            v["model"] = json!({"path": m.path.to_string_lossy(), "reads": verdict(&m.reads)});
            for (key, said) in [("facts", &m.facts), ("sizing", &m.sizing)] {
                match said {
                    Some(Ok(text)) => v["model"][key] = parsed(text),
                    Some(Err(why)) => v["model"][key] = Value::String(why.clone()),
                    None => {}
                }
            }
        }
        v
    }

    pub fn exit_code(&self) -> i32 {
        let reads = self.model.as_ref().is_none_or(|m| m.reads.is_ok());
        if self.available.is_ok() && reads {
            0
        } else {
            1
        }
    }

    pub fn finish(&self) -> ! {
        println!("{}", self.to_json());
        std::process::exit(self.exit_code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> Report {
        Report {
            info: RuntimeInfo {
                id: "llamacpp",
                aliases: &["llama.cpp"],
                formats: vec![Format::file("gguf", "a GGUF file", &["gguf"], Some(b"GGUF"))],
                tokenizer: TokenizerSource::Library("superfluid_tokenizer_llamacpp".into()),
            },
            version: Some("llama.cpp b11284".into()),
            available: Ok(()),
            devices: vec![Device { backend: "Metal".into(), name: "Apple M5 Pro".into(), memory: 1 << 30 }],
            capabilities: r#"{"descriptor_version":1}"#.into(),
            model: None,
            pull: None,
            named: None,
        }
    }

    #[test]
    fn a_usable_runtime_reports_itself_and_exits_zero() {
        let r = report();
        assert_eq!(
            r.to_json(),
            json!({
                "runtime": {"id": "llamacpp", "version": "llama.cpp b11284", "aliases": ["llama.cpp"]},
                "link_w": [crate::server::PROTO_VERSION],
                "available": true,
                "formats": [{"id": "gguf", "describe": "a GGUF file", "file": {"extensions": ["gguf"], "magic": "GGUF"}}],
                "tokenizer": {"from": "library", "name": "superfluid_tokenizer_llamacpp"},
                "devices": [{"backend": "Metal", "name": "Apple M5 Pro", "memory": 1u64 << 30}],
                "capabilities": {"descriptor_version": 1},
            })
        );
        assert_eq!(r.exit_code(), 0);
    }

    #[test]
    fn what_a_report_declares_reads_back() {
        let dir = Format::directory("mlx", "an MLX model directory", &["config.json"], &["safetensors"]);
        let mut r = report();
        r.info.formats.push(dir);
        let v = r.to_json();
        assert_eq!(formats_from(&v["formats"]), r.info.formats);
        assert_eq!(devices_from(&v["devices"]), r.devices);
        assert_eq!(tokenizer_from(&v["tokenizer"]), Some(r.info.tokenizer.clone()));
        for t in [TokenizerSource::HuggingFace, TokenizerSource::Engine] {
            assert_eq!(tokenizer_from(&tokenizer_json(&t)), Some(t));
        }
        assert_eq!(formats_from(&json!([{"id": "x", "describe": "x", "archive": {}}])), vec![]);
        assert_eq!(tokenizer_from(&json!({"from": "sentencepiece"})), None);
    }

    #[test]
    fn what_is_not_usable_says_why_and_exits_one() {
        let mut r = report();
        r.version = None;
        r.available = Err("ModuleNotFoundError: No module named 'mlx'".into());
        assert_eq!(r.to_json()["available"], "ModuleNotFoundError: No module named 'mlx'");
        assert!(r.to_json().get("named").is_none());
        r.named = Some("SUPERFLUID_LLAMA_LIB".into());
        assert_eq!(r.to_json()["named"], "SUPERFLUID_LLAMA_LIB");
        assert_eq!(r.to_json()["runtime"]["version"], Value::Null);
        assert_eq!(r.exit_code(), 1);

        let mut r = report();
        r.model = Some(ModelCheck { path: "/m".into(), reads: Err("/m is not a GGUF file".into()), facts: None, sizing: None });
        assert_eq!(r.to_json()["model"], json!({"path": "/m", "reads": "/m is not a GGUF file"}));
        assert_eq!(r.exit_code(), 1);

        r.model = Some(ModelCheck {
            path: "/m.gguf".into(),
            reads: Ok(()),
            facts: Some(Ok(r#"{"architecture":"qwen3"}"#.into())),
            sizing: Some(Ok(r#"{"trained_context":40960}"#.into())),
        });
        assert_eq!(r.to_json()["model"]["facts"], json!({"architecture": "qwen3"}));
        assert_eq!(r.to_json()["model"]["sizing"], json!({"trained_context": 40960}));
        assert_eq!(r.exit_code(), 0);
        r.model.as_mut().unwrap().sizing = Some(Err("no GPU here to size a window against".into()));
        assert_eq!(r.to_json()["model"]["sizing"], "no GPU here to size a window against");
        assert_eq!(r.exit_code(), 0);
    }
}
