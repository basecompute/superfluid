//! baseRT's native engine as a runtime.

pub mod recipe;

use std::path::Path;
use std::sync::Arc;

use superfluid_adapter_kit::install::{Candidate, Host, Plan, Request};
use superfluid_adapter_kit::{Pull, PullOption};
use superfluid_adapter_kit::{ring_specs_for, Device, Format, Loaded, Probe, Runtime, RuntimeInfo, Tokenizer, TokenizerSource, WorkerArgs};
use superfluid_engine_ffi::{libbasert, FfiEngineConfig, NativeEngine, TokenizerHandle};

pub struct Basert;

impl Runtime for Basert {
    fn compat(&self) -> serde_json::Value {
        let abi = libbasert::abi_built_for();
        let pins = recipe::pinned();
        // Named `pinned_engines`, not `tested`, as each carries the ABI number it states; `versions` lists them itself.
        serde_json::json!({
            "basert_version": libbasert::BUILT_FOR,
            "abi": abi,
            "unnumbered_minor": abi.and_then(libbasert::unnumbered_minor_for),
            "pinned_engines": recipe::driven(&pins).iter().map(|p| serde_json::json!({"version": p.version, "abi": p.abi})).collect::<Vec<_>>(),
        })
    }

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
        Ok(Probe { version: format!("{} ({})", lib.version, lib.path.display()), devices: devices(), abi: lib.abi.map(u64::from) })
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
            PullOption { name: "file", value: true, about: "the .base file to take from a repo that holds several (e.g. Foo-Medium.base)" },
            PullOption { name: "force", value: false, about: "fetch and convert again, even if installed" },
        ])
    }

    fn pull(&self, pull: &Pull, _args: &WorkerArgs) -> Result<std::path::PathBuf, String> {
        let (id, variant) = pull.split()?;
        let tool = basert_tool()?;
        if !pull.offline {
            let installed = if variant.is_some() { installed_list(&tool).unwrap_or_default() } else { String::new() };
            if needs_pull(&installed, id, variant, pull.option("force").is_some()) {
                let version = if variant.is_some() { tool_version(&tool) } else { String::new() };
                let args = pull_args(pull, id, variant, &version);
                let mut cmd = std::process::Command::new(&tool);
                cmd.arg("pull").args(&args);
                superfluid_adapter_kit::run_tool(&mut cmd, &format!("basert pull {}", args[0]))?;
            }
        }
        let list = installed_list(&tool)?;
        // a file taken by name is served from the install basert records it in
        match (variant, pull.option("file").flatten()) {
            (None, Some(file)) => installed_file(&list, id, file),
            _ => installed_path(&list, id, variant),
        }
        .map_err(|why| if pull.offline { format!("{why} (offline: nothing was pulled)") } else { why })
    }

    fn plan(&self, req: &Request, host: &Host) -> Option<Result<Vec<Plan>, String>> {
        Some(recipe::plan(req, host))
    }

    fn versions(&self, host: &Host) -> Vec<Candidate> {
        // The pins alone, offline: `plan` would consult the release listing when a pin fails.
        recipe::versions(host)
    }

    fn unlisted(&self) -> Option<String> {
        Some(recipe::unlisted())
    }
}

/// What `basert list --json` prints.
fn installed_list(tool: &std::path::Path) -> Result<String, String> {
    let list = std::process::Command::new(tool)
        .args(["list", "--json"])
        .output()
        .map_err(|e| format!("basert list --json: {e}"))?;
    if !list.status.success() {
        return Err(format!("basert list --json failed ({}): {}", list.status, String::from_utf8_lossy(&list.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&list.stdout).into_owned())
}

/// Whether `basert pull` has to run for `id[:variant]`. A named variant that is installed already is
/// used as it is: a basert before 0.4.2 takes no variant, so pulling would fetch another quant, or fail
/// where the repo has no default one (a `file-…` variant taken with `--file`), and a later one leaves
/// it as it is. `--force` always pulls.
fn needs_pull(list_json: &str, id: &str, variant: Option<&str>, force: bool) -> bool {
    force || variant.is_none() || installed_path(list_json, id, variant).is_err()
}

/// What follows `basert pull` for `id[:variant]`. A basert whose `--version` printed `version`
/// takes the variant with the id when [`takes_variant`] says so, with every option but `--file`,
/// `--profile` and `--target`, which it refuses beside a variant as naming it twice; an older one
/// takes the id alone, with every option.
fn pull_args(pull: &Pull, id: &str, variant: Option<&str>, version: &str) -> Vec<String> {
    let variant = variant.filter(|_| takes_variant(version));
    let mut args = vec![variant.map_or_else(|| id.to_string(), |v| format!("{id}:{v}"))];
    for (name, value) in &pull.options {
        if variant.is_some() && ["file", "profile", "target"].contains(&name.as_str()) {
            continue;
        }
        args.push(format!("--{name}"));
        args.extend(value.clone());
    }
    args
}

/// Whether the basert whose `--version` printed `version` (`basert 0.4.2`) takes `pull <id>:<variant>`,
/// as 0.4.2 and later do. A pre-release of 0.4.2 may predate it, and output naming no version is
/// taken as an older basert's.
fn takes_variant(version: &str) -> bool {
    let Some(v) = version.split_whitespace().find(|w| w.starts_with(|c: char| c.is_ascii_digit())) else {
        return false;
    };
    let v = v.split_once('+').map_or(v, |(v, _build)| v);
    let (core, pre) = v.split_once('-').map_or((v, false), |(core, _)| (core, true));
    let parts: Result<Vec<u64>, _> = core.split('.').map(str::parse).collect();
    matches!(parts.as_deref(), Ok(&[major, minor, patch]) if (major, minor, patch, !pre) >= (0, 4, 2, true))
}

/// What `basert --version` prints, or nothing when it fails.
fn tool_version(tool: &std::path::Path) -> String {
    std::process::Command::new(tool)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

/// The `.base` file `basert list --json` holds for `id` that was pulled from `file`, matched
/// as `basert pull --file` matches it: by the whole path when `file` has a directory, else by
/// its name, the `.base` extension optional either way.
fn installed_file(list_json: &str, id: &str, file: &str) -> Result<std::path::PathBuf, String> {
    let entries: Vec<serde_json::Value> = serde_json::from_str(list_json).map_err(|e| format!("basert list --json: {e}"))?;
    let bare = |s: &str| s.strip_suffix(".base").unwrap_or(s).to_string();
    let names = |f: &str| match file.contains('/') {
        true => bare(f) == bare(file),
        false => bare(f.rsplit('/').next().unwrap_or(f)) == bare(file),
    };
    entries
        .iter()
        .find(|m| m["id"] == id && m["installed"] == true && m["file"].as_str().is_some_and(names))
        .and_then(|m| m["path"].as_str())
        .map(std::path::PathBuf::from)
        .ok_or_else(|| format!("basert lists no install of {id} pulled from {file} (a basert whose list reports `file` is needed)"))
}

/// The `.base` file `basert list --json` holds for `id`: the variant named with the id,
/// else the one variant installed, else `default-q4` among several.
fn installed_path(list_json: &str, id: &str, variant: Option<&str>) -> Result<std::path::PathBuf, String> {
    let entries: Vec<serde_json::Value> = serde_json::from_str(list_json).map_err(|e| format!("basert list --json: {e}"))?;
    let mine: Vec<&serde_json::Value> =
        entries.iter().filter(|m| m["id"] == id && m["installed"] == true && m["path"].is_string()).collect();
    let variants = || mine.iter().filter_map(|m| m["variant"].as_str()).collect::<Vec<_>>().join(", ");
    let chosen = match variant {
        Some(v) => mine
            .iter()
            .find(|m| m["variant"] == v)
            .ok_or_else(|| format!("basert has no installed variant {v} of {id} (installed: {})", variants()))?,
        None => match mine.as_slice() {
            [] => return Err(format!("basert has no installed model {id}")),
            [one] => one,
            many => many.iter().find(|m| m["variant"] == "default-q4").unwrap_or(&many[0]),
        },
    };
    Ok(std::path::PathBuf::from(chosen["path"].as_str().unwrap_or_default()))
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
    // baseRT states its budget alone, the share of the device it plans a model against.
    vec![Device { backend: backend.into(), name: gpu_name(), memory, budget: memory }]
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
    fn compat_names_only_the_engines_that_install_without_untested() {
        let named: Vec<String> =
            Basert.compat()["pinned_engines"].as_array().unwrap().iter().map(|p| p["version"].as_str().unwrap().to_string()).collect();
        let installs: Vec<String> = recipe::versions(&Host { os: "macos".into(), arch: "arm64".into(), ..Host::default() })
            .into_iter()
            .filter(|c| c.here.is_ok())
            .map(|c| c.version)
            .collect();
        assert!(!named.is_empty(), "the recipe pins an engine this build drives");
        let mut sorted = named.clone();
        sorted.sort_by_key(|v| std::cmp::Reverse(superfluid_adapter_kit::install::version_key(v)));
        // The recipe also pins 0.2.6, of a line this build does not drive: listed with why, and not named.
        assert_eq!(sorted, installs, "what compat names is what installs on its own");
    }

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

#[cfg(test)]
mod pull_tests {
    use super::{installed_file, installed_path, needs_pull, pull_args, takes_variant, Basert};
    use superfluid_adapter_kit::{Pull, Runtime};

    #[test]
    fn a_named_file_is_served_from_the_install_basert_records_it_in() {
        let l = r#"[
            {"id":"org/a","variant":"default-medium","installed":true,"path":"/m/medium.base","file":"a-Medium.base"},
            {"id":"org/a","variant":"cuda-q4mix","installed":true,"path":"/m/cuda.base","file":"sub/a-cuda-q4mix.base"},
            {"id":"org/a","variant":"default-q4","installed":true,"path":"/m/q4.base"}]"#;
        assert_eq!(installed_file(l, "org/a", "a-Medium.base").unwrap().to_str(), Some("/m/medium.base"));
        assert_eq!(installed_file(l, "org/a", "a-Medium").unwrap().to_str(), Some("/m/medium.base"));
        assert_eq!(installed_file(l, "org/a", "a-cuda-q4mix.base").unwrap().to_str(), Some("/m/cuda.base"));
        assert_eq!(installed_file(l, "org/a", "sub/a-cuda-q4mix").unwrap().to_str(), Some("/m/cuda.base"));
        assert!(installed_file(l, "org/a", "other/a-cuda-q4mix.base").is_err());
        assert!(installed_file(l, "org/a", "a-Small.base").unwrap_err().contains("no install of org/a pulled from a-Small.base"));
    }

    fn list(entries: &[(&str, &str, bool)]) -> String {
        let items: Vec<String> = entries
            .iter()
            .map(|(id, variant, installed)| {
                format!(
                    r#"{{"id":"{id}","variant":"{variant}","installed":{installed},"path":"/m/{id}/{variant}/model.base"}}"#
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    #[test]
    fn the_one_installed_variant_is_the_model() {
        let l = list(&[("org/a", "default-q4", true), ("org/b", "default-q8", true)]);
        assert_eq!(installed_path(&l, "org/b", None).unwrap().to_str().unwrap(), "/m/org/b/default-q8/model.base");
    }

    #[test]
    fn a_named_variant_is_taken_and_a_missing_one_refused() {
        let l = list(&[("org/a", "default-q4", true), ("org/a", "default-q8", true)]);
        assert_eq!(installed_path(&l, "org/a", Some("default-q8")).unwrap().to_str().unwrap(), "/m/org/a/default-q8/model.base");
        let e = installed_path(&l, "org/a", Some("default-f16")).unwrap_err();
        assert!(e.contains("no installed variant default-f16") && e.contains("default-q4, default-q8"), "{e}");
    }

    #[test]
    fn several_variants_fall_back_to_default_q4_then_the_first() {
        let l = list(&[("org/a", "default-q8", true), ("org/a", "default-q4", true)]);
        assert_eq!(installed_path(&l, "org/a", None).unwrap().to_str().unwrap(), "/m/org/a/default-q4/model.base");
        let l = list(&[("org/a", "default-q8", true), ("org/a", "default-f16", true)]);
        assert_eq!(installed_path(&l, "org/a", None).unwrap().to_str().unwrap(), "/m/org/a/default-q8/model.base");
    }

    #[test]
    fn an_installed_named_variant_is_not_pulled_again() {
        let l = list(&[("org/a", "file-a-medium-6583221f", true), ("org/b", "default-q4", false)]);
        assert!(!needs_pull(&l, "org/a", Some("file-a-medium-6583221f"), false));
        assert!(needs_pull(&l, "org/a", Some("file-a-medium-6583221f"), true), "--force pulls");
        assert!(needs_pull(&l, "org/a", None, false), "a bare id pulls (its default quant may be newer)");
        assert!(needs_pull(&l, "org/a", Some("default-q8"), false), "a variant not here pulls");
        assert!(needs_pull(&l, "org/b", Some("default-q4"), false), "a variant listed but not installed pulls");
        assert!(needs_pull("", "org/a", Some("file-a-medium-6583221f"), false), "no list: pull as before");
    }

    #[test]
    fn basert_takes_a_variant_with_the_id_from_0_4_2() {
        for v in ["basert 0.4.2\n", "basert 0.4.3", "basert 0.10.0\n", "basert 1.0.0", "basert 0.4.3-rc.1", "0.4.2"] {
            assert!(takes_variant(v), "{v:?}");
        }
        for v in ["basert 0.4.1\n", "basert 0.3.10", "basert 0.4.2-rc.779", "", "basert", "basert unknown", "basert 0.4", "basert 0.4.x", "garbage"] {
            assert!(!takes_variant(v), "{v:?}");
        }
    }

    /// What follows `basert pull` for the pull `args`, given the basert that printed `version`.
    fn argv(args: &str, version: &str) -> String {
        let args: Vec<String> = args.split(' ').map(str::to_string).collect();
        let pull = Pull::parse(&args, Basert.pull_options().unwrap()).unwrap();
        let (id, variant) = pull.split().unwrap();
        pull_args(&pull, id, variant, version).join(" ")
    }

    #[test]
    fn a_named_variant_is_pulled_as_named_where_basert_takes_one() {
        let every = "--target base-q8 --profile p.json --revision v1 --file a-Medium.base --force";
        let named = format!("org/a:file-a-medium-6583221f {every}");
        let as_named = "org/a:file-a-medium-6583221f --revision v1 --force";
        assert_eq!(argv(&named, "basert 0.4.2\n"), as_named, "--file, --profile and --target would name the variant twice");
        assert_eq!(argv(&named, "basert 0.10.0\n"), as_named);
        assert_eq!(argv(&named, "basert 0.4.1\n"), format!("org/a {every}"), "an older basert takes the id alone");
        assert_eq!(argv(&named, "garbage"), format!("org/a {every}"), "so does one that names no version");
        assert_eq!(argv(&format!("org/a {every}"), "basert 0.4.2\n"), format!("org/a {every}"), "a bare id is pulled as before");
    }

    #[test]
    fn a_model_not_installed_is_refused() {
        let l = list(&[("org/a", "default-q4", false)]);
        assert!(installed_path(&l, "org/a", None).unwrap_err().contains("no installed model org/a"));
        assert!(installed_path("not json", "org/a", None).unwrap_err().contains("basert list --json"));
    }
}

