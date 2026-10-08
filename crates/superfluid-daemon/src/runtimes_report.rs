//! `superfluid runtimes [<model>] [--json]`.

use std::path::Path;

use serde_json::{json, Value};

use crate::capabilities::{Cap, Capabilities};
use crate::runtimes::{Catalog, Found, Runtime};

fn short_version(v: Option<&str>) -> String {
    v.map(|v| v.split(" (").next().unwrap_or(v).to_string()).unwrap_or_else(|| "-".into())
}

fn state(f: &Found) -> &'static str {
    match (&f.usable, &f.worker) {
        (Ok(_), _) => "ready",
        (Err(_), None) => "missing",
        (Err(_), Some(w)) if matches!(w.source, crate::runtimes::Source::Installed | crate::runtimes::Source::Env) => "broken",
        (Err(_), Some(_)) => "not installed",
    }
}

fn columns(rows: &[Vec<String>]) -> String {
    let n = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..n).map(|c| rows.iter().filter_map(|r| r.get(c)).map(|s| s.chars().count()).max().unwrap_or(0)).collect();
    let mut out = String::new();
    for r in rows {
        let line: Vec<String> = r
            .iter()
            .enumerate()
            .map(|(c, cell)| if c + 1 == r.len() { cell.clone() } else { format!("{cell:<w$}", w = widths[c]) })
            .collect();
        out.push_str(line.join("  ").trim_end());
        out.push('\n');
    }
    out
}

pub fn with_installs(catalog: &Catalog) -> Vec<Found> {
    let mut all = Vec::new();
    for found in catalog.survey() {
        let id = found.id;
        let default = found.worker.as_ref().and_then(|w| w.install());
        all.push(found);
        for install in crate::installs::list(catalog, id) {
            if Some(&install.name) != default.as_ref() {
                if let Some(at) = crate::runtime_pick::RuntimeId::parse(&format!("{id}@{}", install.name)) {
                    all.push(catalog.inspect(at));
                }
            }
        }
    }
    all
}

fn install_cell(f: &Found, catalog: Option<&Catalog>) -> String {
    let Some(name) = f.worker.as_ref().and_then(|w| w.install()) else { return "-".into() };
    if f.id.install().is_some() {
        return name;
    }
    match catalog.and_then(|c| crate::installs::pinned(c, f.id)) {
        Some(p) if p == name => format!("{name} (pinned)"),
        _ => format!("{name} (default)"),
    }
}

pub fn table(found: &[Found], packages: &Path) -> String {
    table_of(found, packages, None)
}

pub fn table_of(found: &[Found], packages: &Path, catalog: Option<&Catalog>) -> String {
    let mut rows = vec![["RUNTIME", "STATE", "INSTALL", "VERSION", "READS", "DEVICE", "WORKER"].map(String::from).to_vec()];
    let mut problems = Vec::new();
    for f in found {
        let reads = f
            .info
            .as_ref()
            .map(|i| i.formats.iter().map(|x| x.id.as_str()).collect::<Vec<_>>().join(","))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "-".into());
        let (version, device) = match &f.usable {
            Ok(rt) => (short_version(rt.version.as_deref()), rt.device().map(|d| d.summary()).unwrap_or_else(|| "-".into())),
            Err(why) => {
                problems.push(format!("{}: {why}", f.id));
                ("-".into(), "-".into())
            }
        };
        let worker = f.worker.as_ref().map(|w| format!("{}: {}", w.source.name(), w.bin.display())).unwrap_or_else(|| "-".into());
        rows.push(vec![f.id.to_string(), state(f).into(), install_cell(f, catalog), version, reads, device, worker]);
    }
    let mut out = if found.is_empty() { "no runtime is installed here\n".to_string() } else { columns(&rows) };
    for p in problems {
        out.push_str(&p);
        out.push('\n');
    }
    out.push_str(&format!("installed runtimes: {}\n", packages.display()));
    out
}

pub fn table_json(found: &[Found]) -> Value {
    Value::Array(found.iter().map(runtime_json).collect())
}

fn runtime_json(f: &Found) -> Value {
    use superfluid_worker::check::{device_json, format_json, tokenizer_json};
    let mut v = json!({"id": f.id.name(), "state": state(f)});
    if let Some(info) = &f.info {
        v["aliases"] = json!(info.aliases);
        v["formats"] = Value::Array(info.formats.iter().map(format_json).collect());
        v["reads"] = json!(info.reads_described());
        if let Some(t) = &info.tokenizer {
            v["tokenizer_source"] = tokenizer_json(t);
        }
    }
    if let Some(w) = &f.worker {
        v["worker"] = json!({"path": w.bin, "source": w.source.name()});
        if let Some(install) = w.install() {
            v["install"] = json!(install);
        }
    }
    match &f.usable {
        Ok(rt) => {
            v["ready"] = true.into();
            v["version"] = json!(rt.version);
            v["devices"] = Value::Array(rt.devices.iter().map(device_json).collect());
            if let Some(t) = &rt.tokenizer {
                v["tokenizer"] = json!(t);
            }
            if let Some(c) = &rt.capabilities {
                v["capabilities"] = serde_json::from_str(c).unwrap_or_default();
            }
        }
        Err(why) => v["ready"] = why.clone().into(),
    }
    v
}

struct Answer {
    found: Found,
    reads: Result<(), String>,
    facts: Option<Value>,
}

fn ask(catalog: &Catalog, found: Found, model: &Path) -> Answer {
    let _ = catalog;
    let by_format = |f: &Found| -> Result<(), String> {
        match &f.info {
            Some(i) if i.format_of(model).is_some() => Ok(()),
            Some(i) => Err(format!("it reads {}", i.reads_described())),
            None => Err(f.usable.as_ref().err().cloned().unwrap_or_default()),
        }
    };
    let report = found.worker.as_ref().and_then(|w| {
        if w.source == crate::runtimes::Source::BuiltIn && !w.bin.is_file() {
            crate::linked::report(found.id, Some(model))
        } else {
            crate::runtimes::run_check_model(w, Some(model)).ok()
        }
    });
    let (reads, facts) = match report.as_ref().map(|r| &r["model"]) {
        Some(m) if !m.is_null() => {
            let reads = match &m["reads"] {
                Value::Bool(true) => Ok(()),
                Value::String(why) => Err(why.clone()),
                _ => by_format(&found),
            };
            (reads, m.get("facts").filter(|f| f.is_object()).cloned())
        }
        _ => (by_format(&found), None),
    };
    Answer { found, reads, facts }
}

fn facts_line(facts: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(a) = facts["architecture"].as_str().filter(|a| !a.is_empty()) {
        parts.push(a.to_string());
    }
    if let Some(sd) = facts["sampling_defaults"].as_object() {
        let kv: Vec<String> = sd.iter().map(|(k, v)| format!("{k} {v}")).collect();
        parts.push(format!("sampling defaults: {}", kv.join(", ")));
    }
    parts.join("; ")
}

fn serves_lines(rt: &Runtime) -> Vec<(String, String)> {
    let caps: Capabilities = rt.static_capabilities();
    if caps.record().is_none() {
        return vec![("capabilities".into(), format!("the {} runtime describes a model once it is loaded", rt.id))];
    }
    let show = |c: Cap<'_>| match c {
        Cap::Yes => "yes".to_string(),
        Cap::No(why) => format!("no: {why}"),
        Cap::Unknown => "said once the model is loaded".to_string(),
    };
    [
        ("images", "modalities", "image_encode"),
        ("audio", "modalities", "gemma_audio_encode"),
        ("speech to text", "workload", "speech_to_text"),
        ("embeddings", "workload", "embedding"),
        ("grammar", "sampling", "grammar"),
        ("--speculate", "serving", "multi_row_verification"),
        ("--park-lossy", "serving", "park_lossy"),
        ("--kv-bits", "load", "kv_bits"),
    ]
    .into_iter()
    .map(|(what, section, key)| (what.to_string(), show(caps.get(section, key))))
    .collect()
}

pub fn model(catalog: &Catalog, model: &Path) -> (String, Value, bool) {
    let answers: Vec<Answer> = catalog.survey().into_iter().map(|f| ask(catalog, f, model)).collect();
    let format = answers.iter().find_map(|a| a.found.info.as_ref()?.format_of(model).cloned());
    let serving = catalog.reader_of(model).and_then(|id| catalog.require(id));

    let mut out = format!(
        "{}: {}\n\n",
        model.display(),
        format.as_ref().map(|f| f.describe.clone()).unwrap_or_else(|| "no runtime here declares this format".into())
    );
    let mut rows = vec![["RUNTIME", "READS IT", "DEVICE", "MODEL"].map(String::from).to_vec()];
    for a in &answers {
        let device = a.found.usable.as_ref().ok().and_then(|r| r.device()).map(|d| d.summary()).unwrap_or_else(|| "-".into());
        let (reads, detail) = match &a.reads {
            Ok(()) => ("yes".to_string(), a.facts.as_ref().map(facts_line).unwrap_or_default()),
            Err(why) => ("no".to_string(), why.clone()),
        };
        rows.push(vec![a.found.id.to_string(), reads, device, detail]);
    }
    if answers.is_empty() {
        out.push_str("no runtime is installed here\n");
    } else {
        out.push_str(&columns(&rows));
    }
    out.push('\n');
    match &serving {
        Ok(rt) => {
            out.push_str(&format!(
                "{} serves it: the runtime that reads its format (--runtime picks another that reads it)\n",
                rt.id
            ));
            let lines = serves_lines(rt);
            let w = lines.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
            for (what, verdict) in lines {
                out.push_str(&format!("  {what:<w$}  {verdict}\n"));
            }
        }
        Err(why) => out.push_str(&format!("no runtime here serves it: {why}\n")),
    }

    let json = json!({
        "model": model,
        "format": format.as_ref().map(superfluid_worker::check::format_json),
        "runtimes": answers.iter().map(|a| {
            let mut v = runtime_json(&a.found);
            v["reads_model"] = match &a.reads { Ok(()) => true.into(), Err(why) => why.clone().into() };
            if let Some(f) = &a.facts { v["model_facts"] = f.clone(); }
            v
        }).collect::<Vec<_>>(),
        "serves": match &serving { Ok(rt) => json!(rt.id.name()), Err(why) => json!({"none": why}) },
    });
    (out, json, serving.is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_pick::RuntimeId;
    use crate::runtimes::tests::{fake_worker, report};

    const V: u16 = superfluid_agent::client::PROTO_VERSION;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("superfluid-runtimes-report-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn install(home: &Path, id: &str, text: &str, exit: i32) {
        let root = home.join("runtimes").join(id);
        let bin = root.join("v1-metal/bin");
        std::fs::create_dir_all(&bin).unwrap();
        fake_worker(&bin, &format!("superfluid-worker-{id}"), text, exit);
        let lib = root.join("v1-metal/lib");
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::write(lib.join(crate::dylib_tokenizer::library_file("superfluid_tokenizer_llamacpp")), b"").unwrap();
        let _ = std::fs::remove_file(root.join("current"));
        std::os::unix::fs::symlink("v1-metal", root.join("current")).unwrap();
    }

    #[test]
    fn the_table_shows_what_each_runtime_reads_and_runs_on_and_what_is_wrong() {
        if !crate::linked::ids().is_empty() {
            return;
        }
        let home = scratch("table");
        install(&home, "llamacpp", &report("llamacpp", V, "true"), 0);
        install(&home, "mlx", &report("mlx", V, r#""ModuleNotFoundError: No module named 'mlx'""#), 1);
        let cat = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let text = table(&cat.survey(), &home.join("runtimes"));
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("RUNTIME   STATE   INSTALL             VERSION  READS  DEVICE"), "{text}");
        assert!(lines[1].starts_with("llamacpp  ready   v1-metal (default)  v1       gguf   Metal: Apple M5 Pro, 36.0 GB  installed: "), "{text}");
        assert!(lines[2].starts_with("mlx       broken  v1-metal (default)  -        mlx    -"), "{text}");
        assert!(lines[3].starts_with("mlx: the mlx runtime at ") && lines[3].contains("`superfluid runtime repair mlx`"), "{text}");
        let json = table_json(&cat.survey());
        assert_eq!(json[0]["formats"][0]["id"], "gguf");
        assert_eq!(json[0]["reads"], "a GGUF file");
        assert_eq!(json[0]["devices"][0]["backend"], "Metal");
        assert_eq!(json[1]["state"], "broken");
    }

    #[test]
    fn an_adapter_with_no_install_is_listed_as_not_installed() {
        if !crate::linked::ids().is_empty() {
            return;
        }
        let root = scratch("shipped");
        let (beside, libexec) = (root.join("bin"), root.join("libexec").join("superfluid"));
        std::fs::create_dir_all(&beside).unwrap();
        std::fs::create_dir_all(&libexec).unwrap();
        let not_found = r#""llama.cpp not found: install it with `superfluid runtime install llamacpp`""#;
        fake_worker(&libexec, "superfluid-worker-llamacpp", &report("llamacpp", V, not_found), 1);
        let home = root.join("home");
        let cat = Catalog::new(home.clone(), Some(beside.clone()), beside.join("superfluid-workerd"));
        let text = table(&cat.survey(), &home.join("runtimes"));
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[1].starts_with("llamacpp  not installed  -        -        gguf   -       shipped with superfluid: "), "{text}");
        assert_eq!(
            lines[2],
            "llamacpp: the llamacpp runtime is not installed: llama.cpp not found: install it with `superfluid runtime install llamacpp`",
            "{text}"
        );
        let json = table_json(&cat.survey());
        assert_eq!(json[0]["state"], "not installed");
        assert_eq!(json[0]["worker"]["source"], "shipped with superfluid");
        assert_eq!(json[0]["formats"][0]["id"], "gguf");
    }

    #[test]
    fn a_model_is_answered_by_every_runtime_and_served_by_its_formats() {
        if !crate::linked::ids().is_empty() {
            return;
        }
        let home = scratch("model");
        let with_model = |id: &str, reads: &str| {
            let r: Value = serde_json::from_str(&report(id, V, "true")).unwrap();
            let mut r = r;
            r["model"] = json!({"path": "/m", "reads": serde_json::from_str::<Value>(reads).unwrap()});
            if reads == "true" {
                r["model"]["facts"] = json!({"architecture": "qwen3", "sampling_defaults": {"temperature": 0.6}});
            }
            r.to_string()
        };
        install(&home, "llamacpp", &with_model("llamacpp", "true"), 0);
        install(&home, "mlx", &with_model("mlx", r#""/m is not a directory; an MLX model is a directory""#), 1);
        let cat = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let m = scratch("model-file").join("q.gguf");
        std::fs::write(&m, b"GGUF").unwrap();
        let (text, json, ok) = model(&cat, &m);
        assert!(ok, "{text}");
        assert!(text.starts_with(&format!("{}: a GGUF file\n", m.display())), "{text}");
        assert!(text.contains("llamacpp  yes       Metal: Apple M5 Pro, 36.0 GB  qwen3; sampling defaults: temperature 0.6"), "{text}");
        assert!(text.contains("mlx       no        Metal: Apple M5 Pro, 36.0 GB  /m is not a directory; an MLX model is a directory"), "{text}");
        assert!(text.contains("llamacpp serves it"), "{text}");
        assert!(text.contains("--park-lossy    no: no lossy encoding"), "{text}");
        assert!(text.contains("--kv-bits       no: no such knob"), "{text}");
        assert!(text.contains("images          said once the model is loaded"), "{text}");
        assert_eq!(json["serves"], "llamacpp");
        assert_eq!(json["format"]["id"], "gguf");
        assert_eq!(json["runtimes"][1]["reads_model"], "/m is not a directory; an MLX model is a directory");

        let b = m.with_file_name("q.base");
        std::fs::write(&b, b"BASE").unwrap();
        let (text, json, ok) = model(&cat, &b);
        assert!(!ok);
        assert!(text.starts_with(&format!("{}: no runtime here declares this format", b.display())), "{text}");
        assert!(text.contains("no runtime here serves it: no runtime here reads"), "{text}");
        assert!(json["serves"]["none"].as_str().unwrap().contains("runtimes here: llamacpp reads a GGUF file"));
        let _ = RuntimeId::new("llamacpp");
    }
}
