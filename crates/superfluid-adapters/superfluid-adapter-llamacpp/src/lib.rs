//! llama.cpp as a superfluid runtime.

pub mod options;
pub mod recipe;
pub mod runtime;
pub mod sys;
pub mod tokenizer;

use std::path::Path;
use std::sync::Arc;

use superfluid_adapter_kit::install::{Host, Plan, Request};
use superfluid_adapter_kit::{
    Format, Loaded, Probe, Pull, PullOption, Runtime, RuntimeInfo, ServeOption, Sizing, Tokenizer, TokenizerSource, WorkerArgs,
};

pub use runtime::{devices, model_facts, quiet_logs, LlamaConfig, LlamaError, LlamaRuntime};
pub use tokenizer::LlamaTokenizer;

pub struct Llamacpp;

/// The library the command line names, by the option or its older spelling.
fn lib_given(args: &WorkerArgs) -> Option<&str> {
    superfluid_adapter_kit::option(args, "lib").or_else(|| args.extra("--llama-lib"))
}

impl Runtime for Llamacpp {
    fn compat(&self) -> serde_json::Value {
        serde_json::json!({"tested": crate::recipe::tested().into_iter().map(|(v, _)| v).collect::<Vec<_>>()})
    }

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

    fn options(&self) -> &'static [ServeOption] {
        options::OPTIONS
    }

    fn prepare(&self, args: &WorkerArgs) {
        if let Some(lib) = lib_given(args) {
            sys::set_library(lib.into());
        }
    }

    fn named(&self, args: &WorkerArgs) -> Option<String> {
        if lib_given(args).is_some() {
            return Some("--llamacpp-lib".into());
        }
        std::env::var_os("SUPERFLUID_LLAMA_LIB").filter(|p| !p.is_empty()).map(|_| "SUPERFLUID_LLAMA_LIB".into())
    }

    fn probe(&self, _args: &WorkerArgs) -> Result<Probe, String> {
        let lib = sys::load()?;
        Ok(Probe { version: lib.version, devices: devices(), abi: None })
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

    fn sizing(&self, path: &Path, args: &WorkerArgs) -> Option<Result<Sizing, String>> {
        Some(options::cache_types(args).and_then(|(k, v)| runtime::sizing(path, k, v)))
    }

    fn open(&self, args: &WorkerArgs) -> Result<Loaded, String> {
        let ctx = superfluid_adapter_kit::context(args);
        let mut cfg = LlamaConfig {
            model_path: superfluid_adapter_kit::model(args)?.to_path_buf(),
            max_seq_len: ctx,
            max_batch: args.max_batch,
            ..Default::default()
        };
        options::apply(args, &mut cfg)?;
        let runtime = LlamaRuntime::open(cfg).map_err(|e| e.to_string())?;
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
            PullOption { name: "file", value: true, about: "the GGUF file in the repository, instead of the quant or size the :tag names" },
            PullOption { name: "no-mmproj", value: false, about: "skip the multimodal projector a vision model carries" },
        ])
    }

    fn pull(&self, pull: &Pull, _args: &WorkerArgs) -> Result<std::path::PathBuf, String> {
        let (id, tag) = pull.split()?;
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
        let file = match pull.option("file").flatten() {
            Some(file) => Some(file.to_string()),
            None => sized::file(id, tag, pull.offline)?,
        };
        let mut cmd = std::process::Command::new(&tool);
        cmd.arg("download");
        match &file {
            // a file names what a tag would (llama.cpp reads a size tag as a quant and refuses it)
            Some(file) => cmd.args(["-hf", id, "-hff", file]),
            None => cmd.args(["-hf", &pull.id]),
        };
        if pull.offline {
            cmd.arg("--offline");
        }
        if pull.option("no-mmproj").is_some() {
            cmd.arg("--no-mmproj");
        }
        // `llama download -hf` lands in the Hub cache's directory for the repository; its own
        // cache (LLAMA_CACHE) takes direct URLs and docker models, which a pull never names.
        // The Hub's listing sizes the bar; not under HF_HUB_OFFLINE, which keeps off the Hub.
        let watch = (!pull.offline).then(|| superfluid_adapter_kit::progress::Watch {
            roots: vec![superfluid_adapter_kit::install::hf_repo_cache(id)],
            follow: vec![],
            total: if superfluid_adapter_kit::install::hf_offline() { None } else { expected::bytes(id, tag, file.as_deref(), pull.option("no-mmproj").is_some()) },
        });
        superfluid_adapter_kit::progress::watched(watch, || superfluid_adapter_kit::run_pull_tool(&mut cmd, &format!("llama download {}", pull.id))).map_err(|why| {
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

/// What a pull fetches, in bytes, for its progress bar: the GGUF `llama download` takes (every
/// shard of it) and the projector it takes beside a vision model, less what the Hub cache holds.
/// Best effort: a listing or manifest that fails leaves the bar without a total.
mod expected {
    use superfluid_adapter_kit::install::{fetch_json, hf_endpoint, hf_repo_listing, HfListing};

    pub fn bytes(id: &str, tag: Option<&str>, file: Option<&str>, no_mmproj: bool) -> Option<u64> {
        let listing = hf_repo_listing(id, None).ok()?;
        let (model, mmproj) = match file {
            Some(f) => (f.to_string(), None),
            None => {
                let (model, mmproj) = manifest(id, tag.unwrap_or("latest"))?;
                (path_of(&listing, &model)?, mmproj.and_then(|m| path_of(&listing, &m)))
            }
        };
        Some(of(&listing, &model, mmproj.as_deref(), no_mmproj))
    }

    /// The file the Hub's manifest for `id:tag` names, and the projector beside it: what
    /// `llama download` asks when no file is named, so the same file. Each is the SHA-256 of
    /// its bytes (the manifest's layers name files so; it names them by path only to llama.cpp's
    /// user agent), which the listing knows the path of.
    fn manifest(id: &str, tag: &str) -> Option<(String, Option<String>)> {
        let url = format!("{}/v2/{id}/manifests/{tag}", hf_endpoint().trim_end_matches('/'));
        let v = fetch_json(&url).ok()?;
        let layer = |kind: &str| {
            v["layers"]
                .as_array()?
                .iter()
                .find(|l| l["mediaType"].as_str().is_some_and(|m| m.ends_with(kind)))
                .and_then(|l| l["digest"].as_str())
                .map(|d| d.strip_prefix("sha256:").unwrap_or(d).to_string())
        };
        Some((layer(".model")?, layer(".projector")))
    }

    /// The path in `listing` of the file `key` names, by path or by the SHA-256 of its bytes.
    fn path_of(listing: &HfListing, key: &str) -> Option<String> {
        listing.files.iter().find(|f| f.path == key || f.blob == key).map(|f| f.path.clone())
    }

    /// The bytes of `model` (its shards with it) and of the projector: `mmproj` when the
    /// manifest named one, else the sole projector in the repository, which llama.cpp takes
    /// beside a file named to it.
    pub fn of(listing: &HfListing, model: &str, mmproj: Option<&str>, no_mmproj: bool) -> u64 {
        let stem = super::sized::model(model);
        let projectors: Vec<&str> = listing.files.iter().map(|f| f.path.as_str()).filter(|p| super::sized::is_mmproj(p)).collect();
        let projector = match (no_mmproj, mmproj) {
            (true, _) => None,
            (false, Some(named)) => Some(named.to_string()),
            (false, None) => match projectors.as_slice() {
                [one] => Some(one.to_string()),
                _ => None,
            },
        };
        listing.to_fetch(|p| {
            let gguf = p.to_ascii_lowercase().ends_with(".gguf");
            (gguf && !super::sized::is_mmproj(p) && super::sized::model(p) == stem) || projector.as_deref() == Some(p)
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use superfluid_adapter_kit::install::HfFile;

        fn listing(files: &[(&str, u64)]) -> HfListing {
            HfListing {
                id: "org/m".into(),
                sha: "abc".into(),
                files: files.iter().map(|(p, n)| HfFile { path: p.to_string(), size: *n, blob: format!("blob-{p}") }).collect(),
            }
        }

        /// The real repository, asked of the Hub: run with `--ignored` where the network is.
        #[test]
        #[ignore]
        fn the_hub_names_the_file_a_tag_takes_and_sizes_it() {
            let id = "ggml-org/gemma-3-1b-it-GGUF";
            let listing = hf_repo_listing(id, None).expect("the listing");
            let (model, mmproj) = manifest(id, "latest").expect("the manifest");
            assert_eq!((path_of(&listing, &model).as_deref(), mmproj), (Some("gemma-3-1b-it-Q4_K_M.gguf"), None));
            assert_eq!(path_of(&listing, &manifest(id, "Q8_0").unwrap().0).as_deref(), Some("gemma-3-1b-it-Q8_0.gguf"));
            assert_eq!(manifest(id, "nope"), None, "a tag the repository has no file for");
            let q8 = listing.files.iter().find(|f| f.path == "gemma-3-1b-it-Q8_0.gguf").unwrap();
            assert_eq!(q8.size, 1_069_306_368);
            let (model, projector) = manifest("ggml-org/gemma-3-4b-it-GGUF", "latest").expect("a vision model's manifest");
            let vision = hf_repo_listing("ggml-org/gemma-3-4b-it-GGUF", None).unwrap();
            assert_eq!(path_of(&vision, &model).as_deref(), Some("gemma-3-4b-it-Q4_K_M.gguf"));
            assert_eq!(projector.and_then(|p| path_of(&vision, &p)).as_deref(), Some("mmproj-model-f16.gguf"));
            println!("expected bytes, latest: {:?}; Q8_0: {:?}", bytes(id, None, None, false), bytes(id, Some("Q8_0"), None, false));
        }

        #[test]
        fn a_pull_is_the_file_named_its_shards_and_the_projector_it_takes() {
            let l = listing(&[("README.md", 10), ("m-Q4_K_M.gguf", 800), ("m-Q8_0.gguf", 1000), ("mmproj-F16.gguf", 300)]);
            assert_eq!(of(&l, "m-Q4_K_M.gguf", None, false), 1100, "the sole projector comes along");
            assert_eq!(of(&l, "m-Q4_K_M.gguf", None, true), 800, "--no-mmproj leaves it");
            assert_eq!(of(&l, "m-Q8_0.gguf", Some("mmproj-F16.gguf"), false), 1300);
            let two = listing(&[("m-Q4_K_M.gguf", 800), ("mmproj-F16.gguf", 300), ("mmproj-Q8.gguf", 200)]);
            assert_eq!(of(&two, "m-Q4_K_M.gguf", None, false), 800, "which of two projectors is llama.cpp's call");
            assert_eq!(of(&two, "m-Q4_K_M.gguf", Some("mmproj-Q8.gguf"), false), 1000);
            let shards = listing(&[("m-XL-00001-of-00002.gguf", 500), ("m-XL-00002-of-00002.gguf", 400), ("m-Small.gguf", 100)]);
            assert_eq!(of(&shards, "m-XL-00001-of-00002.gguf", None, false), 900);
            assert_eq!(of(&shards, "m-Small.gguf", None, false), 100);
        }
    }
}

/// A release in sizes: a repository whose GGUFs are named by size, the BaseQuant repos'
/// `-XS`/`-Small`/`-Medium`/`-Large`/`-XL`, and none by quant. Hugging Face's manifest, which
/// `llama download` asks with no file named, reads no size: with the tag `latest` it hands
/// back some file (the Large one, for `basecompute/Qwen3.8-27B-BaseQuant-V1`), and a size
/// tag it refuses as not a quantization. So the file is picked here, from the repository's
/// listing, and named to `llama download` as `-hff`: the size the tag spells, else the
/// default size, else the sole file. The basert runtime's pull reads these releases by the
/// same rule.
pub mod sized {
    /// The sizes, smallest first, and how a release spells each in a file name.
    const SIZES: [(&str, &str); 5] = [("xs", "XS"), ("small", "Small"), ("medium", "Medium"), ("large", "Large"), ("xl", "XL")];
    /// The size a pull takes when nothing names one.
    pub const DEFAULT: &str = "medium";

    /// The size `name` spells, case aside (`Medium` → `medium`).
    pub fn name(name: &str) -> Option<&'static str> {
        let n = name.to_ascii_lowercase();
        SIZES.iter().find(|(s, _)| *s == n).map(|(s, _)| *s)
    }

    /// A file's name without `.gguf` and without the shard suffix llama.cpp splits a large
    /// file into: `Foo-Medium.gguf`, `Foo-Medium-00001-of-00002.gguf` → `Foo-Medium`.
    fn name_stem(name: &str) -> String {
        let stem = name.strip_suffix(".gguf").or_else(|| name.strip_suffix(".GGUF")).unwrap_or(name);
        match stem.rsplit_once("-of-") {
            Some((head, total)) if total.chars().all(|c| c.is_ascii_digit()) && !total.is_empty() => match head.rsplit_once('-') {
                Some((stem, index)) if index.chars().all(|c| c.is_ascii_digit()) && !index.is_empty() => stem.to_string(),
                _ => stem.to_string(),
            },
            _ => stem.to_string(),
        }
    }

    /// The model a file is of, by its whole path in the repository: `sub/Foo-Medium.gguf` and
    /// `sub/Foo-Medium-00001-of-00002.gguf` are one model, `other/Foo-Medium.gguf` another.
    pub(crate) fn model(path: &str) -> String {
        match path.rsplit_once('/') {
            Some((dir, name)) => format!("{dir}/{}", name_stem(name)),
            None => name_stem(path),
        }
    }

    /// The last `-` segment of a file's name (its directories aside), lowercased:
    /// `Foo-Medium.gguf` → `medium`, `Foo-UD-Q4_K_M.gguf` → `q4_k_m`.
    fn tag(path: &str) -> String {
        let stem = name_stem(path.rsplit('/').next().unwrap_or(path));
        stem.rsplit('-').next().unwrap_or(&stem).to_ascii_lowercase()
    }

    /// The models among `files` that `keep` admits, each with its first file in path order
    /// (a sharded model's first shard, which llama.cpp follows to the rest).
    fn models<'a>(files: &[&'a String], keep: impl Fn(&str) -> bool) -> Vec<(String, &'a String)> {
        let mut sorted: Vec<&'a String> = files.to_vec();
        sorted.sort();
        let mut out: Vec<(String, &'a String)> = Vec::new();
        for f in sorted.into_iter().filter(|f| keep(f)) {
            let m = model(f);
            if !out.iter().any(|(seen, _)| *seen == m) {
                out.push((m, f));
            }
        }
        out
    }

    /// The size a GGUF is named by, if any.
    fn size_of(path: &str) -> Option<&'static str> {
        name(&tag(path))
    }

    /// Whether a GGUF is named by quant, as llama.cpp's conversions and the common
    /// publishers name theirs (`Q4_K_M`, `IQ2_S`, `UD-Q4_K_XL`, `BF16`, `TQ1_0`, `MXFP4`).
    fn quant_named(path: &str) -> bool {
        let t = tag(path);
        if ["f16", "bf16", "f32", "f64", "mxfp4", "nvfp4"].contains(&t.as_str()) {
            return true;
        }
        let t = t.strip_prefix('i').or_else(|| t.strip_prefix('t')).unwrap_or(&t);
        t.strip_prefix('q').is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
    }

    /// The multimodal projector a vision model carries beside its weights, not a model file.
    pub(crate) fn is_mmproj(path: &str) -> bool {
        path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase().starts_with("mmproj")
    }

    /// The GGUF a pull of `id` with `tag` takes from the files `files` lists, when the
    /// repository is a release in sizes: the size the tag spells, else [`DEFAULT`], else
    /// the sole file. None when it is not one (a GGUF named by quant, or none by size),
    /// so llama.cpp picks as it does; a quant tag is llama.cpp's too. A size the release
    /// does not come in names the ones it does.
    pub fn pick(id: &str, files: &[String], tag: Option<&str>) -> Result<Option<String>, String> {
        let size = match tag {
            Some(t) => match name(t) {
                Some(s) => Some(s),
                None => return Ok(None),
            },
            None => None,
        };
        let ggufs: Vec<&String> = files.iter().filter(|f| f.to_ascii_lowercase().ends_with(".gguf") && !is_mmproj(f)).collect();
        if ggufs.iter().any(|f| quant_named(f)) || !ggufs.iter().any(|f| size_of(f).is_some()) {
            return match size {
                Some(s) => Err(format!(
                    "{id} has no {s} GGUF: its files are not named by size ({}); name one with --pull-file",
                    ggufs.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(", ")
                )),
                None => Ok(None),
            };
        }
        let offered: Vec<&str> = SIZES.iter().map(|(s, _)| *s).filter(|s| ggufs.iter().any(|f| size_of(f) == Some(s))).collect();
        let want = size.unwrap_or(DEFAULT);
        let hits = models(&ggufs, |f| size_of(f) == Some(want));
        match (hits.as_slice(), size) {
            ([(_, first)], _) => Ok(Some((*first).clone())),
            ([], None) if models(&ggufs, |_| true).len() == 1 => Ok(Some(models(&ggufs, |_| true)[0].1.clone())),
            ([], _) => Err(format!(
                "{id} has no {want} GGUF; it comes in: {} (`{id}:<size>` names one, --pull-file a file)",
                offered.join(", ")
            )),
            _ => Err(format!(
                "{want} names several GGUFs of {id} ({}); name one with --pull-file",
                hits.iter().map(|(_, f)| f.as_str()).collect::<Vec<_>>().join(", ")
            )),
        }
    }

    /// The file a pull of `id[:tag]` names to `llama download`, when the repository is a
    /// release in sizes (see [`pick`]), read from its listing. Offline nothing is listed:
    /// a size spells its file as the releases name them (`<repo>-Small.gguf`), and no tag
    /// is left to llama.cpp's cache. A listing that fails leaves a plain pull to llama.cpp
    /// with a warning, and fails a size, which only the listing can resolve.
    pub fn file(id: &str, tag: Option<&str>, offline: bool) -> Result<Option<String>, String> {
        let size = match tag {
            Some(t) => match name(t) {
                Some(s) => Some(s),
                None => return Ok(None),
            },
            None => None,
        };
        if offline {
            let repo = id.rsplit('/').next().unwrap_or(id);
            return Ok(size.map(|s| format!("{repo}-{}.gguf", SIZES.iter().find(|(n, _)| *n == s).map_or(s, |(_, spelled)| spelled))));
        }
        let files = match superfluid_adapter_kit::install::hf_repo_files(id) {
            Ok(files) => files,
            Err(why) => match size {
                Some(s) => return Err(format!("{id}:{s}: the repository could not be listed to find its {s} file ({why}); name it with --pull-file")),
                None => {
                    eprintln!("warning: {id} could not be listed ({why}); llama.cpp picks the file");
                    return Ok(None);
                }
            },
        };
        let picked = pick(id, &files, tag)?;
        if let (Some(file), None) = (&picked, size) {
            eprintln!("llamacpp: {id} is a release in sizes; taking {file} (`{id}:<size>` or --pull-file takes another)");
        }
        Ok(picked)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn files(names: &[&str]) -> Vec<String> {
            names.iter().map(|n| n.to_string()).collect()
        }

        #[test]
        fn a_release_in_sizes_gives_its_medium_the_size_named_or_its_sole_file() {
            let id = "org/m";
            let release = files(&["README.md", "m-XS.gguf", "m-Small.gguf", "m-Medium.gguf", "m-Large.gguf", "m-XL.gguf", "m-Medium.base"]);
            assert_eq!(pick(id, &release, None).unwrap().as_deref(), Some("m-Medium.gguf"));
            assert_eq!(pick(id, &release, Some("small")).unwrap().as_deref(), Some("m-Small.gguf"));
            assert_eq!(pick(id, &release, Some("XL")).unwrap().as_deref(), Some("m-XL.gguf"));
            // a quant tag is llama.cpp's to read
            assert_eq!(pick(id, &release, Some("Q4_K_M")).unwrap(), None);
            // one file is taken whatever its size; a size the release lacks names the ones it has
            assert_eq!(pick(id, &files(&["m-Small.gguf"]), None).unwrap().as_deref(), Some("m-Small.gguf"));
            let e = pick(id, &files(&["m-Small.gguf", "m-Large.gguf"]), None).unwrap_err();
            assert!(e.contains("has no medium GGUF; it comes in: small, large") && e.contains("`org/m:<size>`"), "{e}");
            let e = pick(id, &files(&["m-Small.gguf"]), Some("large")).unwrap_err();
            assert!(e.contains("it comes in: small"), "{e}");
            // a sharded size is its first shard
            let shards = files(&["m-Medium-00002-of-00002.gguf", "m-Medium-00001-of-00002.gguf", "m-Small.gguf"]);
            assert_eq!(pick(id, &shards, None).unwrap().as_deref(), Some("m-Medium-00001-of-00002.gguf"));
            // the projector beside a vision model is not a model
            let vision = files(&["mmproj-F16.gguf", "v-Medium.gguf"]);
            assert_eq!(pick(id, &vision, None).unwrap().as_deref(), Some("v-Medium.gguf"));
        }

        #[test]
        fn a_repository_named_by_quant_is_left_to_llama_cpp() {
            let id = "org/q";
            for repo in [
                files(&["q-Q4_K_M.gguf", "q-Q8_0.gguf"]),
                files(&["q-UD-Q4_K_XL.gguf", "q-UD-IQ2_S.gguf", "q-BF16.gguf"]),
                files(&["q-TQ1_0.gguf"]),
                files(&["gpt-oss-20b-mxfp4.gguf"]),
                files(&["q-Q4_K_M.gguf", "q-Medium.gguf"]),
                files(&["model.gguf"]),
                files(&["README.md", "config.json"]),
            ] {
                assert_eq!(pick(id, &repo, None).unwrap(), None, "{repo:?}");
                assert_eq!(pick(id, &repo, Some("Q4_K_M")).unwrap(), None, "{repo:?}");
                let e = pick(id, &repo, Some("medium")).unwrap_err();
                assert!(e.contains("not named by size"), "{repo:?}: {e}");
            }
        }

        #[test]
        fn offline_a_size_spells_its_file_and_nothing_named_is_llama_cpps() {
            assert_eq!(file("basecompute/Qwen3.8-27B-BaseQuant-V1", Some("small"), true).unwrap().as_deref(), Some("Qwen3.8-27B-BaseQuant-V1-Small.gguf"));
            assert_eq!(file("org/m", Some("XL"), true).unwrap().as_deref(), Some("m-XL.gguf"));
            assert_eq!(file("org/m", None, true).unwrap(), None);
            assert_eq!(file("org/m", Some("Q4_K_M"), true).unwrap(), None);
        }

        /// Two models of one size in different directories are two, not one; and the sole
        /// model of a repository is sole however many shards it is in.
        #[test]
        fn models_are_told_apart_by_their_whole_path_and_counted_over_their_shards() {
            let id = "org/m";
            let two = files(&["imatrix/m-Medium.gguf", "original/m-Medium.gguf"]);
            let e = pick(id, &two, None).unwrap_err();
            assert!(e.contains("medium names several GGUFs") && e.contains("--pull-file"), "{e}");
            let e = pick(id, &two, Some("medium")).unwrap_err();
            assert!(e.contains("medium names several GGUFs"), "{e}");
            let sharded = files(&["m-XL-00002-of-00002.gguf", "m-XL-00001-of-00002.gguf", "README.md"]);
            assert_eq!(pick(id, &sharded, None).unwrap().as_deref(), Some("m-XL-00001-of-00002.gguf"));
            // a directory's name is not the file's tag
            let dirs = files(&["q4-quants/m-Medium.gguf"]);
            assert_eq!(pick(id, &dirs, None).unwrap().as_deref(), Some("q4-quants/m-Medium.gguf"));
        }

        /// The real repository, listed from the Hub: run with `--ignored` where the network is.
        #[test]
        #[ignore]
        fn the_basequant_repository_gives_its_medium() {
            let id = "basecompute/Qwen3.8-27B-BaseQuant-V1";
            assert_eq!(file(id, None, false).unwrap().as_deref(), Some("Qwen3.8-27B-BaseQuant-V1-Medium.gguf"));
            assert_eq!(file(id, Some("xs"), false).unwrap().as_deref(), Some("Qwen3.8-27B-BaseQuant-V1-XS.gguf"));
            assert_eq!(file("Qwen/Qwen3-0.6B-GGUF", None, false).unwrap(), None);
        }
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
            assert_eq!(d[0].budget, (superfluid_executor::sizing::PLANNING_SHARE * d[0].memory as f64) as u64, "{d:?}");
        }
        assert!(d.iter().filter(|d| d.backend == "CPU").all(|d| d.budget == 0), "nothing is planned on a CPU: {d:?}");
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
