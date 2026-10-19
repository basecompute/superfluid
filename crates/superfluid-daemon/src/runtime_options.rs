//! `--<runtime>-<name> <value>` on `superfluid serve`: an option of one runtime, checked against what its adapter
//! declares and forwarded to its worker as `--<name> <value>`; set once per process, as `BASERT_LIB` is.

use std::path::Path;
use std::sync::OnceLock;

use serde_json::{json, Value};
use superfluid_adapter_kit::{check_value, placeholder, OptionDecl, WorkerArgs};

use crate::runtime_pick::RuntimeId;
use crate::runtimes::{Catalog, Runtime, Source, Worker};

/// One `--<runtime>-<name> <value>` as given, resolved to the runtime it names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Given {
    pub flag: String,
    pub runtime: RuntimeId,
    pub name: String,
    pub value: String,
}

/// Whether `flag` is shaped as a runtime option, `--<x>-<y>`; only then does `serve` keep it for [`resolve`].
pub fn looks_like(flag: &str) -> bool {
    let Some(body) = flag.strip_prefix("--") else { return false };
    let shaped = body.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '.' | '_'));
    shaped && body.split_once('-').is_some_and(|(id, name)| !id.is_empty() && !name.is_empty())
}

/// The runtime among `ids` that `flag` names, the longest id winning (`--mlx-x-y` is mlx's `x-y`
/// unless a runtime `mlx-x` is here), or `None` when no runtime here is named.
pub fn resolve(flag: &str, value: &str, ids: &[RuntimeId]) -> Option<Given> {
    let body = flag.strip_prefix("--")?;
    let mut best: Option<(RuntimeId, &str)> = None;
    for id in ids.iter().map(|id| id.base()) {
        let Some(name) = body.strip_prefix(id.name()).and_then(|rest| rest.strip_prefix('-')) else { continue };
        if !name.is_empty() && best.is_none_or(|(b, _)| id.name().len() > b.name().len()) {
            best = Some((id, name));
        }
    }
    best.map(|(runtime, name)| Given { flag: flag.to_string(), runtime, name: name.to_string(), value: value.to_string() })
}

/// Refuses an option given twice: the worker reads the last value and this process the first (`daemon_environment`),
/// so a repeat would split one knob between two values; serve refuses it as a usage error before anything starts.
pub fn refuse_repeats(given: &[Given]) -> Result<(), String> {
    for (i, g) in given.iter().enumerate() {
        if let Some(first) = given[..i].iter().find(|p| p.runtime == g.runtime && p.name == g.name) {
            return Err(format!("{}: given twice ({:?}, then {:?}); give it once", g.flag, first.value, g.value));
        }
    }
    Ok(())
}

/// The options among `given` for `runtime`, checked against its declarations, as the pairs its worker takes.
pub fn check(given: &[Given], runtime: &Runtime) -> Result<Vec<(String, String)>, String> {
    let id = runtime.id.base();
    let mut out = Vec::new();
    for g in given.iter().filter(|g| g.runtime == id) {
        let Some(decl) = runtime.options.iter().find(|o| o.name == g.name) else {
            let takes: Vec<String> = runtime.options.iter().map(|o| format!("--{}-{}", id.name(), o.name)).collect();
            return Err(if takes.is_empty() {
                format!("{}: the {} runtime takes no option of its own", g.flag, id.name())
            } else {
                format!("{}: the {} runtime takes no such option (it takes: {})", g.flag, id.name(), takes.join(", "))
            });
        };
        check_value(&decl.kind, &decl.choices, decl.least, &g.value).map_err(|e| format!("{}: {e}", g.flag))?;
        out.push((format!("--{}", g.name), g.value.clone()));
    }
    Ok(out)
}

/// The options per runtime, as the `--<name> <value>` pairs its worker takes.
pub type Settled = Vec<(RuntimeId, Vec<(String, String)>)>;

/// The options as given, grouped by runtime, for the first probe of a runtime: an option that makes it usable
/// here (its library, its environment) counts from the start, and the worker refuses one it does not take.
pub fn given_pairs(given: &[Given]) -> Settled {
    let mut out: Settled = Vec::new();
    for g in given {
        let pair = (format!("--{}", g.name), g.value.clone());
        match out.iter_mut().find(|(id, _)| *id == g.runtime) {
            Some((_, pairs)) => pairs.push(pair),
            None => out.push((g.runtime, vec![pair])),
        }
    }
    out
}

/// When a probe failed with options given: the option at fault, named as the daemon names it, when the worker
/// answers without them; `None` when the runtime fails without them too.
pub fn blamed(catalog: &Catalog, id: RuntimeId, given: &[Given]) -> Option<String> {
    if !given.iter().any(|g| g.runtime == id.base()) {
        return None;
    }
    let plain = catalog.locate_plain(id)?;
    let rt = crate::runtimes::check_worker(id, &plain).ok()?;
    check(given, &rt).err()
}

/// The arguments an in-process worker gets, parsed as a worker process parses them, so `--venv` lands in its field.
pub fn worker_args(id: RuntimeId, model: &Path, ctx: i32, max_batch: u32, kv_bits: i32) -> Result<WorkerArgs, String> {
    worker_args_from(&for_runtime(id), model, ctx, max_batch, kv_bits)
}

/// The arguments an in-process pull gets: the runtime's options alone, parsed as a worker process parses them.
pub fn pull_args(id: RuntimeId) -> Result<WorkerArgs, String> {
    let options = for_runtime(id);
    let argv: Vec<String> = options.iter().flat_map(|(f, v)| [f.clone(), v.clone()]).collect();
    WorkerArgs::parse(&argv)
}

/// What this process must carry for the libraries it loads itself: the llama.cpp tokenizer finds libllama by
/// `SUPERFLUID_LLAMA_LIB`, since the worker's `--lib` never reaches it.
pub fn daemon_environment(settled: &Settled) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    for (id, options) in settled {
        if id.base().name() == "llamacpp" {
            if let Some((_, lib)) = options.iter().find(|(flag, _)| flag == "--lib") {
                out.push(("SUPERFLUID_LLAMA_LIB", lib.clone()));
            }
        }
    }
    out
}

/// Sets [`daemon_environment`] on this process, before any such library is loaded.
pub fn export_to_daemon() {
    if let Some(all) = SET.get() {
        for (name, value) in daemon_environment(all) {
            std::env::set_var(name, value);
        }
    }
}

pub fn worker_args_from(options: &[(String, String)], model: &Path, ctx: i32, max_batch: u32, kv_bits: i32) -> Result<WorkerArgs, String> {
    let mut argv: Vec<String> = vec![
        "--model".into(),
        model.to_string_lossy().into_owned(),
        "--max-context".into(),
        ctx.to_string(),
        "--max-batch".into(),
        max_batch.to_string(),
        "--kv-bits".into(),
        kv_bits.to_string(),
    ];
    for (flag, value) in options {
        argv.push(flag.clone());
        argv.push(value.clone());
    }
    WorkerArgs::parse(&argv)
}

/// Every given option checked: against `primary` for its runtime, and against any other runtime named, which must be usable here.
pub fn settle(given: &[Given], primary: &Runtime, catalog: &Catalog) -> Result<Settled, String> {
    let mut ids: Vec<RuntimeId> = given.iter().map(|g| g.runtime).collect();
    ids.sort();
    ids.dedup();
    let mut out = Vec::new();
    for id in ids {
        let checked = if id == primary.id.base() {
            check(given, primary)?
        } else {
            let rt = catalog.require(id).map_err(|e| {
                let flag = given.iter().find(|g| g.runtime == id).map(|g| g.flag.as_str()).unwrap_or_default();
                format!("{flag}: the {} runtime is not usable here, so its options cannot be checked: {e}", id.name())
            })?;
            check(given, &rt)?
        };
        out.push((id, checked));
    }
    Ok(out)
}

static SET: OnceLock<Settled> = OnceLock::new();

/// Keeps the settled options for every worker this process starts; the first call wins.
pub fn set(settled: Settled) {
    let _ = SET.set(settled);
}

/// The `--<name> <value>` pairs a worker of `id` gets.
pub fn for_runtime(id: RuntimeId) -> Vec<(String, String)> {
    SET.get().and_then(|all| all.iter().find(|(r, _)| *r == id.base()).map(|(_, o)| o.clone())).unwrap_or_default()
}

/// What `superfluid runtime options <id>` lists: a usable runtime's declaration, else the one its adapter makes before
/// an install, asked as `runtime versions` asks it (the worker named for it, the linked adapter, the worker that would
/// install it): the options, `--llamacpp-lib` among them, are what one reads before the runtime runs here.
pub fn declared_by(catalog: &Catalog, id: RuntimeId) -> Result<Vec<OptionDecl>, String> {
    let id = id.base();
    let not_usable = match catalog.require(id) {
        Ok(runtime) => return Ok(runtime.options),
        Err(why) => why,
    };
    let named = std::env::var_os(crate::runtimes::worker_env(id)).is_some();
    if let Some(rt) = crate::linked::get(id).filter(|_| !named) {
        return Ok(superfluid_adapter_kit::declared(rt));
    }
    // A worker's report declares its options whether or not the runtime runs here.
    let bin = crate::installs::installer(catalog, id).map_err(|_| not_usable.clone())?;
    let report = crate::runtimes::run_check_model(&Worker { bin, prefix: Vec::new(), source: Source::Shipped }, None).map_err(|_| not_usable)?;
    Ok(superfluid_worker::check::options_from(&report["options"]))
}

/// `superfluid runtime options <id>`, for a person.
pub fn text(id: RuntimeId, options: &[OptionDecl]) -> String {
    let id = id.base().name();
    if options.is_empty() {
        return format!("the {id} runtime takes no option of its own; `superfluid serve --help` lists the flags every runtime takes\n");
    }
    let rows: Vec<(String, String, String)> = options
        .iter()
        .map(|o| {
            let default = if o.default.is_empty() { "-".to_string() } else { o.default.clone() };
            (format!("--{id}-{} {}", o.name, placeholder(&o.kind, &o.choices)), default, o.about.clone())
        })
        .collect();
    let w1 = rows.iter().map(|r| r.0.chars().count()).max().unwrap_or(0);
    let w2 = rows.iter().map(|r| r.1.chars().count()).max().unwrap_or(0).max("DEFAULT".len());
    let mut out = format!("Options of the {id} runtime on `superfluid serve`:\n\n  {:<w1$}  {:<w2$}  \n", "OPTION", "DEFAULT");
    for (flag, default, about) in rows {
        out.push_str(&format!("  {flag:<w1$}  {default:<w2$}  {about}\n"));
    }
    out
}

/// `superfluid runtime options <id> --json`.
pub fn json(id: RuntimeId, options: &[OptionDecl]) -> Value {
    json!({"runtime": id.base().name(), "options": options.iter().map(|o| o.to_json()).collect::<Vec<_>>()})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtimes::Info;

    fn runtime(id: &str, options: Vec<OptionDecl>) -> Runtime {
        Runtime {
            id: RuntimeId::new(id),
            worker: Worker { bin: std::path::PathBuf::from("/nowhere"), prefix: Vec::new(), source: Source::Env },
            version: None,
            abi: None,
            capabilities: None,
            info: Info::default(),
            devices: Vec::new(),
            tokenizer: None,
            pull: None,
            options,
        }
    }

    fn decl(name: &str, kind: &str, choices: &[&str], default: &str) -> OptionDecl {
        OptionDecl {
            name: name.into(),
            kind: kind.into(),
            choices: choices.iter().map(|c| c.to_string()).collect(),
            least: None,
            default: default.into(),
            about: format!("about {name}"),
        }
    }

    #[test]
    fn the_options_as_given_are_grouped_by_runtime_for_the_first_probe() {
        let ids = [RuntimeId::new("llamacpp"), RuntimeId::new("mlx")];
        let given = [
            resolve("--llamacpp-lib", "/custom/lib", &ids).unwrap(),
            resolve("--mlx-venv", "/custom/env", &ids).unwrap(),
            resolve("--llamacpp-n-gpu-layers", "20", &ids).unwrap(),
        ];
        let pairs = given_pairs(&given);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], (ids[0], vec![("--lib".to_string(), "/custom/lib".to_string()), ("--n-gpu-layers".to_string(), "20".to_string())]));
        assert_eq!(pairs[1], (ids[1], vec![("--venv".to_string(), "/custom/env".to_string())]));
        assert!(given_pairs(&[]).is_empty());
    }

    #[test]
    fn an_option_given_twice_is_refused() {
        let ids = [RuntimeId::new("llamacpp"), RuntimeId::new("mlx")];
        let given = |flags: &[(&str, &str)]| flags.iter().map(|(f, v)| resolve(f, v, &ids).unwrap()).collect::<Vec<_>>();
        let twice = given(&[("--llamacpp-lib", "/a"), ("--mlx-venv", "/e"), ("--llamacpp-lib", "/b")]);
        assert_eq!(refuse_repeats(&twice).unwrap_err(), "--llamacpp-lib: given twice (\"/a\", then \"/b\"); give it once");
        let same = given(&[("--llamacpp-threads", "8"), ("--llamacpp-threads", "8")]);
        assert!(refuse_repeats(&same).is_err(), "the same value twice is still a repeat");
        assert!(refuse_repeats(&given(&[("--llamacpp-lib", "/a"), ("--mlx-venv", "/a"), ("--llamacpp-threads", "8")])).is_ok());
        assert!(refuse_repeats(&[]).is_ok());
    }

    #[test]
    fn in_process_worker_arguments_are_parsed_as_a_worker_process_parses_them() {
        let options = [("--venv".to_string(), "/custom/env".to_string()), ("--prefill-chunk".to_string(), "512".to_string())];
        let args = worker_args_from(&options, Path::new("/m"), 4096, 8, 0).unwrap();
        assert_eq!(args.venv.as_deref(), Some(Path::new("/custom/env")), "where the MLX adapter reads its environment");
        assert_eq!(args.extra, [("--prefill-chunk".to_string(), "512".to_string())]);
        assert_eq!((args.model.as_deref(), args.max_context, args.max_batch, args.kv_bits), (Some(Path::new("/m")), 4096, 8, 0));
        let plain = worker_args_from(&[], Path::new("/m"), 2048, 4, 8).unwrap();
        assert!(plain.venv.is_none() && plain.extra.is_empty() && plain.kv_bits == 8);
        assert!(worker_args_from(&[("--lib".to_string(), String::new())], Path::new("/m"), 1, 1, 0).is_ok(), "a value is the parser's to take; the worker judges it");
    }

    #[test]
    fn the_daemon_carries_the_llama_cpp_library_for_its_own_tokenizer() {
        let ids = [RuntimeId::new("llamacpp"), RuntimeId::new("mlx")];
        let given = [resolve("--llamacpp-lib", "/custom/lib", &ids).unwrap(), resolve("--mlx-venv", "/custom/env", &ids).unwrap()];
        assert_eq!(daemon_environment(&given_pairs(&given)), [("SUPERFLUID_LLAMA_LIB", "/custom/lib".to_string())]);
        let other = [resolve("--llamacpp-n-gpu-layers", "20", &ids).unwrap()];
        assert!(daemon_environment(&given_pairs(&other)).is_empty(), "only the library concerns the daemon's own process");
        let args = WorkerArgs::parse(&["--venv".to_string(), "/custom/env".to_string()]).unwrap();
        assert_eq!(args.venv.as_deref(), Some(Path::new("/custom/env")), "what pull_args hands an in-process pull");
    }

    #[test]
    fn a_flag_is_kept_for_resolution_only_when_shaped_as_an_option() {
        assert!(looks_like("--llamacpp-n-gpu-layers"));
        assert!(looks_like("--mlx-venv"));
        assert!(looks_like("--max-context"), "a flag serve knows never reaches here; the shape alone is judged");
        assert!(!looks_like("--port"), "no '-' after the runtime");
        assert!(!looks_like("--llamacpp-"), "an empty name");
        assert!(!looks_like("-ngl") && !looks_like("--Llamacpp-x") && !looks_like("--llamacpp-x=1"));
    }

    #[test]
    fn the_longest_runtime_id_names_the_option() {
        let ids = [RuntimeId::new("mlx"), RuntimeId::new("mlx-pro"), RuntimeId::new("llamacpp@v1-metal")];
        let g = resolve("--mlx-pro-chunk", "4", &ids).unwrap();
        assert_eq!((g.runtime.name(), g.name.as_str(), g.value.as_str()), ("mlx-pro", "chunk", "4"));
        let g = resolve("--mlx-prefill-chunk", "4", &ids).unwrap();
        assert_eq!((g.runtime.name(), g.name.as_str()), ("mlx", "prefill-chunk"));
        let g = resolve("--llamacpp-threads", "8", &ids).unwrap();
        assert_eq!((g.runtime.name(), g.name.as_str()), ("llamacpp", "threads"), "an install's id names its base runtime");
        assert!(resolve("--max-contxt", "8", &ids).is_none(), "no runtime max here");
        assert!(resolve("--mlx-", "8", &ids).is_none());
    }

    #[test]
    fn a_value_is_checked_against_the_declaration_and_the_worker_flag_is_built() {
        let rt = runtime("llamacpp", vec![decl("n-gpu-layers", "int", &[], "-1"), decl("flash-attn", "choice", &["auto", "on", "off"], "auto")]);
        let ids = [rt.id];
        let good = [resolve("--llamacpp-n-gpu-layers", "20", &ids).unwrap(), resolve("--llamacpp-flash-attn", "off", &ids).unwrap()];
        assert_eq!(check(&good, &rt).unwrap(), [("--n-gpu-layers".to_string(), "20".to_string()), ("--flash-attn".to_string(), "off".to_string())]);
        let bad = [resolve("--llamacpp-flash-attn", "yes", &ids).unwrap()];
        assert_eq!(check(&bad, &rt).unwrap_err(), "--llamacpp-flash-attn: one of auto, on, off expected, got \"yes\"");
        let mut floored = runtime("llamacpp", vec![decl("batch-size", "int", &[], "2048")]);
        floored.options[0].least = Some(1);
        let zero = [resolve("--llamacpp-batch-size", "0", &[floored.id]).unwrap()];
        assert_eq!(check(&zero, &floored).unwrap_err(), "--llamacpp-batch-size: at least 1 expected, got 0");
        let unknown = [resolve("--llamacpp-ngl", "20", &ids).unwrap()];
        assert_eq!(
            check(&unknown, &rt).unwrap_err(),
            "--llamacpp-ngl: the llamacpp runtime takes no such option (it takes: --llamacpp-n-gpu-layers, --llamacpp-flash-attn)"
        );
        let none = runtime("basert", Vec::new());
        let given = [resolve("--basert-threads", "2", &[none.id]).unwrap()];
        assert_eq!(check(&given, &none).unwrap_err(), "--basert-threads: the basert runtime takes no option of its own");
        assert!(check(&good, &none).unwrap().is_empty(), "another runtime's options are not its concern");
    }

    #[test]
    fn the_options_are_listed_before_the_runtime_runs_here() {
        let root = std::env::temp_dir().join(format!("superfluid-runtime-options-declared-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (beside, libexec) = (root.join("bin"), root.join("libexec").join("superfluid"));
        std::fs::create_dir_all(&beside).unwrap();
        std::fs::create_dir_all(&libexec).unwrap();
        // A shipped adapter whose library is not here: not usable, and still declaring what it takes.
        let report = r#"{"runtime":{"id":"toyopt","version":"v1"},"available":"no libtoy here","options":[{"name":"lib","kind":"text","default":"the install's","about":"a library of your own"}]}"#;
        crate::runtimes::tests::fake_worker(&libexec, "superfluid-worker-toyopt", report, 1);
        let cat = Catalog::new(root.join("home"), Some(beside.clone()), beside.join("superfluid-workerd"));
        let id = RuntimeId::new("toyopt");
        assert!(cat.require(id).is_err(), "the runtime does not run here");
        let options = declared_by(&cat, id).unwrap();
        assert_eq!(options.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(), ["lib"]);
        assert!(text(id, &options).contains("--toyopt-lib <value>"));
        let nowhere = RuntimeId::new("nort");
        assert_eq!(declared_by(&cat, nowhere).unwrap_err(), cat.require(nowhere).unwrap_err(), "nothing declares it: the runtime's own error");
        for id in crate::linked::ids().into_iter().filter(|id| std::env::var_os(crate::runtimes::worker_env(*id)).is_none()) {
            let linked = superfluid_adapter_kit::declared(crate::linked::get(id).unwrap());
            assert_eq!(declared_by(&cat, id).unwrap(), linked, "{}: a linked adapter answers in process", id.name());
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_listing_shows_each_option_with_its_value_shape_and_default() {
        let rt = runtime("llamacpp", vec![decl("n-gpu-layers", "int", &[], "-1"), decl("flash-attn", "choice", &["auto", "on", "off"], ""), decl("lib", "text", &[], "the install's")]);
        let t = text(rt.id, &rt.options);
        assert!(t.contains("--llamacpp-n-gpu-layers <N>"), "{t}");
        assert!(t.contains("--llamacpp-flash-attn <auto|on|off>"), "{t}");
        assert!(t.contains("--llamacpp-lib <value>") && t.contains("the install's") && t.contains("about lib"), "{t}");
        assert!(text(RuntimeId::new("basert"), &[]).starts_with("the basert runtime takes no option of its own"));
        assert_eq!(json(rt.id, &rt.options)["options"][1]["choices"], json!(["auto", "on", "off"]));
    }
}
