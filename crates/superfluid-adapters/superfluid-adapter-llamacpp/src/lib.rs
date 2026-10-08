//! llama.cpp as a superfluid runtime.

pub mod recipe;
pub mod runtime;
pub mod sys;
pub mod tokenizer;

use std::path::Path;
use std::sync::Arc;

use superfluid_adapter_kit::install::{Host, Plan, Request};
use superfluid_adapter_kit::{Format, Loaded, Probe, Pull, PullOption, Runtime, RuntimeInfo, Sizing, Tokenizer, TokenizerSource, WorkerArgs};

pub use runtime::{devices, model_facts, quiet_logs, LlamaConfig, LlamaError, LlamaRuntime};
pub use tokenizer::LlamaTokenizer;

pub struct Llamacpp;

impl Runtime for Llamacpp {
    fn info(&self) -> RuntimeInfo {
        RuntimeInfo {
            id: "llamacpp",
            aliases: &["llama.cpp", "llama-cpp"],
            formats: vec![Format::file("gguf", "a GGUF file", &["gguf"], Some(b"GGUF"))],
            tokenizer: TokenizerSource::Library("superfluid_tokenizer_llamacpp".into()),
        }
    }

    fn flags(&self) -> &'static [&'static str] {
        &["--llama-lib"]
    }

    fn prepare(&self, args: &WorkerArgs) {
        if let Some(lib) = args.extra("--llama-lib") {
            sys::set_library(lib.into());
        }
    }

    fn named(&self, args: &WorkerArgs) -> Option<String> {
        if args.extra("--llama-lib").is_some() {
            return Some("--llama-lib".into());
        }
        std::env::var_os("SUPERFLUID_LLAMA_LIB").filter(|p| !p.is_empty()).map(|_| "SUPERFLUID_LLAMA_LIB".into())
    }

    fn probe(&self, _args: &WorkerArgs) -> Result<Probe, String> {
        let lib = sys::load()?;
        Ok(Probe { version: lib.version, devices: devices() })
    }

    fn static_capabilities(&self) -> String {
        superfluid_executor::capabilities::static_descriptor("llamacpp", &[superfluid_abi::encoding::LOSSLESS], true).to_string()
    }

    fn reads(&self, path: &Path) -> Result<(), String> {
        let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if !meta.is_file() {
            return Err(format!("{} is not a file; a GGUF model is one file", path.display()));
        }
        let mut magic = [0u8; 4];
        std::fs::File::open(path)
            .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if &magic != b"GGUF" {
            return Err(format!("{} is not a GGUF file (no GGUF magic)", path.display()));
        }
        Ok(())
    }

    fn model_facts(&self, path: &Path) -> Option<Result<String, String>> {
        Some(model_facts(path).map_err(|e| e.to_string()))
    }

    fn sizing(&self, path: &Path, _args: &WorkerArgs) -> Option<Result<Sizing, String>> {
        Some(runtime::sizing(path))
    }

    fn open(&self, args: &WorkerArgs) -> Result<Loaded, String> {
        let ctx = superfluid_adapter_kit::context(args);
        let runtime = LlamaRuntime::open(LlamaConfig {
            model_path: superfluid_adapter_kit::model(args)?.to_path_buf(),
            max_seq_len: ctx,
            max_batch: args.max_batch,
            ..Default::default()
        })
        .map_err(|e| e.to_string())?;
        Ok(Loaded::primitives(runtime, ctx))
    }

    fn tokenizer(&self, model: &Path) -> Option<Result<Arc<dyn Tokenizer>, String>> {
        Some(
            LlamaTokenizer::load(model)
                .map(|t| Arc::new(t) as Arc<dyn Tokenizer>)
                .map_err(|e| format!("tokenizer load failed for {}: {e}", model.display())),
        )
    }

    fn pull_options(&self) -> Option<&'static [PullOption]> {
        Some(&[
            PullOption { name: "file", value: true, about: "the GGUF file in the repository, instead of the quant the :tag names" },
            PullOption { name: "no-mmproj", value: false, about: "skip the multimodal projector a vision model carries" },
        ])
    }

    fn pull(&self, pull: &Pull, _args: &WorkerArgs) -> Result<std::path::PathBuf, String> {
        pull.split()?;
        let lib = sys::load()?;
        let dir = lib.path.parent().map(Path::to_path_buf).unwrap_or_default();
        let tool = dir.join("llama");
        if !tool.is_file() {
            return Err(format!(
                "the llama.cpp at {} has no `llama` tool beside it to pull with (a build from llama.cpp's releases has it: \
                 `superfluid runtime install llamacpp`)",
                dir.display()
            ));
        }
        let mut cmd = std::process::Command::new(&tool);
        cmd.args(["download", "-hf", &pull.id]);
        if pull.offline {
            cmd.arg("--offline");
        }
        if let Some(Some(file)) = pull.option("file") {
            cmd.args(["-hff", file]);
        }
        if pull.option("no-mmproj").is_some() {
            cmd.arg("--no-mmproj");
        }
        superfluid_adapter_kit::run_pull_tool(&mut cmd, &format!("llama download {}", pull.id)).map_err(|why| {
            if pull.offline {
                format!("{why} (offline: only the Hugging Face cache was looked at)")
            } else {
                why
            }
        })
    }

    fn plan(&self, req: &Request, host: &Host) -> Option<Result<Vec<Plan>, String>> {
        Some(recipe::plan(req, host))
    }

    fn install(&self, plan: &Plan, dir: &Path) -> Result<Vec<(String, String)>, String> {
        let lib = dir.join("lib");
        let sources = superfluid_adapter_kit::install::fetch_and_unpack(plan, &lib, dir)?;
        if !sys::file_names("libllama").iter().any(|n| lib.join(n).is_file()) {
            let what = plan.assets.first().map(|a| a.url.as_str()).unwrap_or("the plan (it names no archive)");
            return Err(format!("{what} holds no libllama: not a llama.cpp build this adapter knows"));
        }
        let name = tokenizer_library_file();
        let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default();
        let found = [exe_dir.join(&name), exe_dir.join("..").join("lib").join(&name)].into_iter().find(|p| p.is_file()).ok_or_else(|| {
            format!(
                "the llama.cpp tokenizer library ({name}) is not beside this worker or in ../lib \
                 (a source build makes it with `cargo build -p superfluid-tokenizer-llamacpp`)"
            )
        })?;
        std::fs::copy(&found, lib.join(&name)).map_err(|e| format!("{}: {e}", found.display()))?;
        Ok(sources)
    }
}

fn tokenizer_library_file() -> String {
    if cfg!(target_os = "macos") {
        "libsuperfluid_tokenizer_llamacpp.dylib".into()
    } else {
        "libsuperfluid_tokenizer_llamacpp.so".into()
    }
}

pub fn runtime_version() -> String {
    match sys::load() {
        Ok(lib) => lib.version,
        Err(why) => format!("llama.cpp (not loaded: {why})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_host_has_a_cpu_and_apple_silicon_its_gpu_first() {
        if let Err(why) = sys::load() {
            eprintln!("SKIP: {why}");
            return;
        }
        let d = devices();
        assert!(d.iter().any(|d| d.backend == "CPU"), "{d:?}");
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert_eq!(d[0].backend, "Metal", "{d:?}");
            assert!(d[0].memory > 0 && !d[0].name.is_empty(), "{d:?}");
        }
    }

    #[test]
    fn a_gguf_is_read_by_its_magic_and_anything_else_says_why_not() {
        let dir = std::env::temp_dir().join(format!("superfluid-llama-reads-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("weights.bin");
        std::fs::write(&gguf, b"GGUF\x03\x00\x00\x00").unwrap();
        assert_eq!(Llamacpp.reads(&gguf), Ok(()));
        let zip = dir.join("model.gguf");
        std::fs::write(&zip, b"PK\x03\x04").unwrap();
        assert!(Llamacpp.reads(&zip).unwrap_err().ends_with("is not a GGUF file (no GGUF magic)"));
        assert!(Llamacpp.reads(&dir).unwrap_err().ends_with("is not a file; a GGUF model is one file"));
        assert!(Llamacpp.reads(&dir.join("missing.gguf")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
