//! The checks every adapter runs on itself, so that a runtime the daemon has never seen behaves as
//! the daemon expects.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use superfluid_worker::check::{devices_from, formats_from, tokenizer_from};
use serde_json::Value;

use crate::{check, valid_id, Format, Rule, Runtime, WorkerArgs};

pub fn runtime(rt: &dyn Runtime) {
    let info = rt.info();
    let id = info.id;
    assert!(valid_id(id), "the id {id:?} is not spelled as a runtime id can be");
    for (i, alias) in info.aliases.iter().enumerate() {
        assert!(
            !alias.is_empty() && !alias.contains(char::is_whitespace) && !alias.contains('='),
            "{id}: the alias {alias:?} cannot be written in --runtime"
        );
        assert_ne!(*alias, id, "{id}: an alias repeats the id");
        assert!(!info.aliases[..i].contains(alias), "{id}: the alias {alias:?} is declared twice");
    }
    for flag in rt.flags() {
        assert!(flag.starts_with("--"), "{id}: the flag {flag:?} is not spelled --name");
    }
    for (i, o) in rt.pull_options().unwrap_or_default().iter().enumerate() {
        assert!(
            !o.name.is_empty() && o.name != "offline" && o.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "{id}: the pull option {:?} cannot be written --pull-<name>",
            o.name
        );
        assert!(!rt.pull_options().unwrap_or_default()[..i].iter().any(|p| p.name == o.name), "{id}: the pull option {} is declared twice", o.name);
    }
    for (i, f) in info.formats.iter().enumerate() {
        assert!(!f.id.is_empty() && !f.describe.is_empty(), "{id}: a format needs an id and words for it");
        assert!(!info.formats[..i].iter().any(|g| g.id == f.id), "{id}: the format {} is declared twice", f.id);
        match &f.rule {
            Rule::File { extensions, magic } => assert!(
                !extensions.is_empty() || magic.as_ref().is_some_and(|m| !m.is_empty()),
                "{id}: the file format {} names neither an extension nor a magic, so it matches nothing",
                f.id
            ),
            Rule::Directory { files, extensions } => assert!(
                !extensions.is_empty(),
                "{id}: the directory format {} names no weights extension, so it matches nothing (files: {files:?})",
                f.id
            ),
        }
    }

    let caps = rt.static_capabilities();
    if !caps.is_empty() {
        let v: Value = serde_json::from_str(&caps).unwrap_or_else(|e| panic!("{id}: the static record is not JSON: {e}"));
        assert!(v.get("descriptor_version").is_some(), "{id}: the static record has no descriptor_version: {v}");
    }

    let report = check(rt, &WorkerArgs::default());
    let json = report.to_json();
    assert_eq!(json["runtime"]["id"], id, "{id}: the report names another runtime: {json}");
    let aliases: Vec<&str> = json["runtime"]["aliases"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert_eq!(aliases, info.aliases, "{id}: the report's aliases");
    assert_eq!(formats_from(&json["formats"]), info.formats, "{id}: the report's formats do not read back");
    assert_eq!(tokenizer_from(&json["tokenizer"]), Some(info.tokenizer.clone()), "{id}: the report's tokenizer");
    assert_eq!(devices_from(&json["devices"]), report.devices, "{id}: the report's devices do not read back");
    assert_eq!(json["available"] == Value::Bool(true), report.available.is_ok(), "{id}: available");

    let dir = scratch(id);
    for f in &info.formats {
        let sample = make(&dir, f);
        assert!(f.matches(&sample), "{id}: {} does not match an artifact made to its own rule", f.id);
        assert_eq!(rt.reads(&sample), Ok(()), "{id}: reads refuses {}, made to its format {}", sample.display(), f.id);
    }
    let stranger = dir.join("notes.txt");
    std::fs::write(&stranger, b"not a model").unwrap();
    assert!(!info.formats.iter().any(|f| f.matches(&stranger)), "{id}: a text file matches a format");
    assert!(rt.reads(&stranger).is_err(), "{id}: reads accepts a text file");
    let missing = dir.join("missing");
    assert!(rt.reads(&missing).is_err(), "{id}: reads accepts a path that does not exist");
    let with_model = check(rt, &WorkerArgs { model: Some(missing.clone()), ..Default::default() });
    assert!(with_model.model.as_ref().is_some_and(|m| m.reads.is_err()), "{id}: check --model <missing> reads it");
    assert_eq!(with_model.exit_code(), 1, "{id}: check --model <missing> must exit 1");
    let _ = std::fs::remove_dir_all(&dir);
}

pub fn worker(bin: &Path, id: &str) {
    let run = |args: &[&str]| {
        let out = Command::new(bin).args(args).output().unwrap_or_else(|e| panic!("{}: {e}", bin.display()));
        (out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
    };
    let report = |stdout: &str| -> Value {
        serde_json::from_str(stdout.trim()).unwrap_or_else(|e| panic!("{id}: check printed no report ({e}): {stdout}"))
    };

    let (code, out, err) = run(&["check"]);
    let usable = match code {
        Some(0) => true,
        Some(1) => false,
        other => panic!("{id}: check exited {other:?}: {err}"),
    };
    let v = report(&out);
    assert_eq!(v["runtime"]["id"], id, "{id}: check names another runtime: {v}");
    assert_eq!(v["available"] == Value::Bool(true), usable, "{id}: the exit code disagrees with available: {v}");
    assert!(
        v["link_w"].as_array().is_some_and(|a| a.contains(&Value::from(superfluid_worker::server::PROTO_VERSION))),
        "{id}: check does not speak this daemon's Link W: {v}"
    );

    let (code, out, err) = run(&["version"]);
    if usable {
        assert_eq!(code, Some(0), "{id}: version failed on a usable runtime: {err}");
        assert_eq!(out.trim(), v["runtime"]["version"], "{id}: version and check disagree");
    } else {
        assert_eq!(code, Some(1), "{id}: version succeeded on a runtime check calls unusable");
        assert!(!err.trim().is_empty(), "{id}: version says nothing about why the runtime is not usable");
    }

    let missing = std::env::temp_dir().join(format!("superfluid-conformance-no-such-model-{}", std::process::id()));
    let (code, out, _) = run(&["check", "--model", &missing.to_string_lossy()]);
    assert_eq!(code, Some(1), "{id}: check --model <missing> must exit 1");
    assert!(report(&out)["model"]["reads"].is_string(), "{id}: check --model <missing> says nothing about the model");

    let (code, _, err) = run(&["check", "--no-such-flag", "x"]);
    assert_eq!(code, Some(2), "{id}: a flag it does not read must be refused");
    assert!(err.contains("unknown flag --no-such-flag"), "{id}: {err}");

    let (code, _, err) = run(&["--model", &missing.to_string_lossy()]);
    assert_eq!(code, Some(2), "{id}: serving without the Link W streams must be refused");
    assert!(err.contains("--frames-fd"), "{id}: {err}");
}

fn scratch(id: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!("superfluid-conformance-{id}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn make(dir: &Path, f: &Format) -> PathBuf {
    match &f.rule {
        Rule::File { extensions, magic } => {
            let name = match extensions.first() {
                Some(ext) => format!("model.{ext}"),
                None => "model".to_string(),
            };
            let path = dir.join(format!("{}-{name}", f.id));
            let mut bytes = magic.clone().unwrap_or_default();
            bytes.extend_from_slice(&[0; 60]);
            std::fs::write(&path, bytes).unwrap();
            path
        }
        Rule::Directory { files, extensions } => {
            let path = dir.join(format!("{}-model", f.id));
            std::fs::create_dir_all(&path).unwrap();
            for file in files {
                std::fs::write(path.join(file), b"{}").unwrap();
            }
            std::fs::write(path.join(format!("weights.{}", extensions[0])), b"").unwrap();
            path
        }
    }
}
