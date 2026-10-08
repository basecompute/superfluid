//! baseRT's native engine as a runtime.

pub mod recipe;

use std::path::Path;
use std::sync::Arc;

use superfluid_adapter_kit::install::{Host, Plan, Request};
use superfluid_adapter_kit::{Pull, PullOption};
use superfluid_adapter_kit::{ring_specs_for, Device, Format, Loaded, Probe, Runtime, RuntimeInfo, Tokenizer, TokenizerSource, WorkerArgs};
use superfluid_engine_ffi::{libbasert, FfiEngineConfig, NativeEngine, TokenizerHandle};

pub struct Basert;

impl Runtime for Basert {
    fn info(&self) -> RuntimeInfo {
        RuntimeInfo {
            id: "basert",
            aliases: &["native"],
            formats: vec![Format::file("base", "a .base bundle", &["base"], Some(b"BASE"))],
            tokenizer: TokenizerSource::Engine,
        }
    }

    fn flags(&self) -> &'static [&'static str] {
        &["--kv-bits", "--basert-lib"]
    }

    fn prepare(&self, args: &WorkerArgs) {
        if let Some(lib) = args.extra("--basert-lib") {
            libbasert::set_library(lib.into());
        }
    }

    fn named(&self, args: &WorkerArgs) -> Option<String> {
        if args.extra("--basert-lib").is_some() {
            return Some("--basert-lib".into());
        }
        std::env::var_os("BASERT_LIB").filter(|p| !p.is_empty()).map(|_| "BASERT_LIB".into())
    }

    fn probe(&self, _args: &WorkerArgs) -> Result<Probe, String> {
        let lib = libbasert::load()?;
        Ok(Probe { version: format!("{} ({})", lib.version, lib.path.display()), devices: devices() })
    }

    fn reads(&self, path: &Path) -> Result<(), String> {
        if path.is_dir() {
            return Err(format!("{} is a directory; a .base bundle is one file", path.display()));
        }
        let mut magic = [0u8; 4];
        std::fs::File::open(path)
            .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if &magic != b"BASE" {
            return Err(format!("{} is not a .base bundle (no BASE magic)", path.display()));
        }
        Ok(())
    }

    fn open(&self, args: &WorkerArgs) -> Result<Loaded, String> {
        let model = superfluid_adapter_kit::model(args)?.to_path_buf();
        superfluid_engine_ffi::set_kv_bits(args.kv_bits);
        let engine = NativeEngine::load(FfiEngineConfig {
            model_path: model,
            max_context: args.max_context,
            max_batch_size: args.max_batch,
            seed_ttl_ticks: 64,
        })
        .map_err(|e| e.to_string())?;
        let rings = ring_specs_for(superfluid_adapter_kit::context(args), engine.vocab());
        Ok(Loaded::engine(engine, Some(rings)))
    }

    fn tokenizer(&self, model: &Path) -> Option<Result<Arc<dyn Tokenizer>, String>> {
        Some(
            TokenizerHandle::load(model)
                .map(|t| Arc::new(t) as Arc<dyn Tokenizer>)
                .map_err(|e| format!("tokenizer load failed for {}: {e}", model.display())),
        )
    }

    fn pull_options(&self) -> Option<&'static [PullOption]> {
        Some(&[
            PullOption { name: "target", value: true, about: "the quant scheme a model fetched from its source is converted to (base-q4, base-q8, ...)" },
            PullOption { name: "profile", value: true, about: "a quant profile JSON, instead of the auto-selected one" },
            PullOption { name: "revision", value: true, about: "the Hugging Face revision, branch or tag" },
            PullOption { name: "force", value: false, about: "fetch and convert again, even if installed" },
        ])
    }

    fn pull(&self, pull: &Pull, _args: &WorkerArgs) -> Result<std::path::PathBuf, String> {
        pull.split()?;
        let tool = basert_tool()?;
        let mut cmd = std::process::Command::new(&tool);
        cmd.arg("resolve").arg(&pull.id);
        if pull.offline {
            cmd.arg("--offline");
        }
        for (name, value) in &pull.options {
            cmd.arg(format!("--{name}"));
            cmd.args(value);
        }
        superfluid_adapter_kit::run_pull_tool(&mut cmd, &format!("basert resolve {}", pull.id)).map_err(|why| {
            if why.contains("unrecognized subcommand") {
                format!("the basert tool at {} is older than this adapter (it has no `resolve`): update the runtime", tool.display())
            } else {
                why
            }
        })
    }

    fn plan(&self, req: &Request, host: &Host) -> Option<Result<Vec<Plan>, String>> {
        Some(recipe::plan(req, host))
    }
}

fn basert_tool() -> Result<std::path::PathBuf, String> {
    if let Some(p) = std::env::var_os("BASERT_CLI").filter(|p| !p.is_empty()) {
        return Ok(p.into());
    }
    let beside = libbasert::load().ok().and_then(|lib| lib.path.parent().map(|d| d.join("basert")));
    if let Some(p) = beside.filter(|p| p.is_file()) {
        return Ok(p);
    }
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .map(|d| d.join("basert"))
        .find(|p| p.is_file())
        .ok_or_else(|| {
            "no `basert` tool to pull basert models with: not beside libbaseRT, not on PATH, and BASERT_CLI is not set \
             (`superfluid runtime install basert` brings it)"
                .to_string()
        })
}

pub fn suggest_max_context(
    models: &[std::path::PathBuf],
    speculators: &[(std::path::PathBuf, bool, Option<usize>)],
    lanes: i32,
    kv_bits: i32,
) -> Option<i32> {
    superfluid_engine_ffi::suggest_max_context(models, speculators, lanes, kv_bits)
}

pub fn devices() -> Vec<Device> {
    let Some(memory) = libbasert::device_memory_budget() else {
        return Vec::new();
    };
    let backend = if cfg!(target_os = "macos") { "Metal" } else { "CUDA" };
    vec![Device { backend: backend.into(), name: gpu_name(), memory }]
}

fn gpu_name() -> String {
    #[cfg(target_os = "macos")]
    if let Some(chip) = sysctl_string("machdep.cpu.brand_string") {
        return chip;
    }
    #[cfg(target_os = "linux")]
    if let Some(model) = nvidia_model() {
        return model;
    }
    "GPU".into()
}

#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Option<String> {
    use std::ffi::{c_char, c_int, c_void, CString};
    extern "C" {
        fn sysctlbyname(name: *const c_char, oldp: *mut c_void, oldlenp: *mut usize, newp: *mut c_void, newlen: usize) -> c_int;
    }
    let key = CString::new(name).ok()?;
    let mut buf = [0u8; 256];
    let mut len = buf.len();
    // SAFETY: a read-only query into a buffer of the length passed.
    let rc = unsafe { sysctlbyname(key.as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) };
    if rc != 0 {
        return None;
    }
    let text = String::from_utf8_lossy(&buf[..len]);
    let text = text.trim_end_matches('\0').trim();
    (!text.is_empty()).then(|| text.to_string())
}

#[cfg(target_os = "linux")]
fn nvidia_model() -> Option<String> {
    let from_driver = std::fs::read_dir("/proc/driver/nvidia/gpus").ok().and_then(|dir| {
        dir.flatten().find_map(|gpu| model_line(&std::fs::read_to_string(gpu.path().join("information")).ok()?))
    });
    from_driver.or_else(|| {
        let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=name", "--format=csv,noheader"]).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).lines().next().map(str::trim).map(str::to_string))?
            .filter(|name| named(name))
    })
}

#[cfg(any(target_os = "linux", test))]
fn model_line(information: &str) -> Option<String> {
    information.lines().find_map(|l| l.strip_prefix("Model:")).map(str::trim).filter(|m| named(m)).map(str::to_string)
}

#[cfg(any(target_os = "linux", test))]
fn named(name: &str) -> bool {
    !name.is_empty() && !name.eq_ignore_ascii_case("unknown")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bundle_is_read_by_its_magic_and_its_format_names_it() {
        let dir = std::env::temp_dir().join(format!("superfluid-basert-reads-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bundle = dir.join("m.base");
        std::fs::write(&bundle, b"BASE\x01\x00").unwrap();
        let gguf = dir.join("m.gguf");
        std::fs::write(&gguf, b"GGUF").unwrap();
        assert_eq!(Basert.reads(&bundle), Ok(()));
        assert!(Basert.reads(&gguf).unwrap_err().ends_with("is not a .base bundle (no BASE magic)"));
        assert!(Basert.reads(&dir).unwrap_err().ends_with("is a directory; a .base bundle is one file"));
        let base = &Basert.info().formats[0];
        assert!(base.matches(&bundle) && !base.matches(&gguf));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_driver_that_names_no_model_is_not_a_name() {
        let named = "Model: \t\t NVIDIA GeForce RTX 4090\nIRQ:   \t\t 142\n";
        assert_eq!(model_line(named).as_deref(), Some("NVIDIA GeForce RTX 4090"));
        let spark = "Model: \t\t Unknown\nIRQ:   \t\t 0\nGPU UUID: \t GPU-00000000\n";
        assert_eq!(model_line(spark), None);
        assert_eq!(model_line("IRQ: 1\n"), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_chip_names_the_gpu_on_apple_silicon() {
        assert!(!gpu_name().is_empty());
    }
}
