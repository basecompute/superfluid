//! `superfluid models`: the models pulled here, and the hub catalog's.
//!
//! A pull that succeeds leaves a record beside its lock in `$SUPERFLUID_HOME/pulls`, `<stem>.json` beside
//! `<stem>.lock` ([`crate::pulls::stem`]): the id exactly as it was pulled, the runtime that pulled it, where the
//! model is and its size. The record is what says a model is downloaded. A lock with no record beside it was left
//! by a pull from before superfluid kept records, or by one that failed or is still running; it is listed by the
//! id its name gives back, flagged as guessed, since the name does not keep `/`, `:` and `_` apart.

use std::collections::HashSet;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::runtime_pick::RuntimeId;

/// The version of the document `superfluid models --json` prints, and of a pull's record.
pub const SCHEMA: u32 = 1;

/// What a pull that succeeded leaves at `$SUPERFLUID_HOME/pulls/<stem>.json`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Record {
    #[serde(default)]
    pub schema: u32,
    /// The id exactly as it was pulled, its `:tag` included.
    pub id: String,
    /// The name it is served as: the id without its tag.
    pub served_name: String,
    /// The runtime that pulled it (basert, llamacpp, mlx), without an install.
    pub runtime: String,
    /// Where the pull put it, a file or a directory, as the runtime gave it.
    pub path: PathBuf,
    /// Its size in bytes when it was pulled, as [`size_on_disk`] counts it.
    pub size: u64,
    /// When it was pulled, or found already pulled: seconds since the Unix epoch.
    pub pulled_at: u64,
}

/// Records that a pull of `id` with `runtime` gave `path`, in place of what an earlier pull of it recorded. The
/// record is written beside its place and renamed into it, so a reader sees the old record or the new, never half
/// of one. Ids whose stems are the same share one record, as they share one lock: the last pull of them is kept.
pub fn record(home: &Path, id: &str, runtime: RuntimeId, path: &Path) -> Result<Record, String> {
    let record = Record {
        schema: SCHEMA,
        id: id.to_string(),
        served_name: crate::pulls::served_name(id),
        runtime: runtime.base().to_string(),
        path: std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()),
        size: size_on_disk(path),
        pulled_at: now_secs(),
    };
    let dir = home.join("pulls");
    let stem = crate::pulls::stem(id);
    let file = dir.join(format!("{stem}.json"));
    let tmp = dir.join(format!(".{stem}.json.{}", std::process::id()));
    let body = serde_json::to_string_pretty(&record).expect("a record is JSON") + "\n";
    std::fs::create_dir_all(&dir)
        .and_then(|()| std::fs::write(&tmp, body))
        .and_then(|()| std::fs::rename(&tmp, &file))
        .map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("the record of the pull of {id} could not be written to {}: {e}", file.display())
        })?;
    Ok(record)
}

/// The bytes a pulled model takes, by the length of its files (not the blocks they hold on disk).
///
/// A file is its length, through a link to it: a file a pull gives in the Hugging Face cache is a link into the
/// cache's blobs. A directory is the total of the files in it and below, each counted once by its device and inode,
/// so a blob two names link to, or a file with two hard links, counts once. A link to a file is followed wherever it
/// leads, as a snapshot's links name blobs outside the snapshot (`../../blobs/<hash>`, and in a deduplicated cache
/// `hub/blobs/<xx>/<hash>`, outside the repository's own directory). A link to a directory is never followed, so the
/// walk stays inside the model's directory and cannot loop. Nothing else in the cache counts: other revisions, files
/// no link names, partial downloads. What cannot be read counts nothing, and a path that is not there is 0.
pub fn size_on_disk(path: &Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else { return 0 };
    if !meta.is_dir() {
        return meta.len();
    }
    let mut seen = HashSet::new();
    let mut total = 0;
    walk(path, &mut seen, &mut total);
    total
}

fn walk(dir: &Path, seen: &mut HashSet<(u64, u64)>, total: &mut u64) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else { continue };
        let meta = if file_type.is_dir() {
            walk(&entry.path(), seen, total);
            continue;
        } else if file_type.is_symlink() {
            match std::fs::metadata(entry.path()) {
                Ok(m) if m.is_file() => m,
                _ => continue,
            }
        } else {
            match entry.metadata() {
                Ok(m) if m.is_file() => m,
                _ => continue,
            }
        };
        if seen.insert((meta.dev(), meta.ino())) {
            *total += meta.len();
        }
    }
}

/// A model `superfluid models` lists as downloaded: a pull's record, or a lock a pull left with none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Downloaded {
    pub id: String,
    pub served_name: String,
    pub runtime: Option<String>,
    pub path: Option<PathBuf>,
    pub size: Option<u64>,
    pub pulled_at: Option<u64>,
    /// Whether the path is there now; None when no path is known.
    pub exists: Option<bool>,
    /// The id was read back from a lock's name: no record says what it was.
    pub id_guessed: bool,
}

impl Downloaded {
    fn of(record: Record) -> Downloaded {
        Downloaded {
            exists: Some(record.path.exists()),
            id: record.id,
            served_name: record.served_name,
            runtime: Some(record.runtime),
            path: Some(record.path),
            size: Some(record.size),
            pulled_at: Some(record.pulled_at),
            id_guessed: false,
        }
    }

    fn guessed(id: String) -> Downloaded {
        Downloaded {
            served_name: crate::pulls::served_name(&id),
            id,
            runtime: None,
            path: None,
            size: None,
            pulled_at: None,
            exists: None,
            id_guessed: true,
        }
    }

    fn here(&self) -> bool {
        self.exists == Some(true)
    }

    fn json(&self) -> Value {
        json!({
            "id": self.id,
            "served_name": self.served_name,
            "runtime": self.runtime,
            "path": self.path,
            "size": self.size,
            "pulled_at": self.pulled_at,
            "exists": self.exists,
            "id_guessed": self.id_guessed,
        })
    }
}

/// The models pulled under `home`, by id: one per record, and one per lock that has no record beside it, its id
/// guessed with the help of `catalog` ([`guess_id`]). A record that cannot be read counts as none.
pub fn downloaded(home: &Path, catalog: &HubCatalog) -> Vec<Downloaded> {
    let mut records = Vec::new();
    let mut locks = Vec::new();
    for entry in std::fs::read_dir(home.join("pulls")).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if let Some(stem) = name.strip_suffix(".json") {
            let record = std::fs::read(entry.path()).ok().and_then(|b| serde_json::from_slice::<Record>(&b).ok());
            if let Some(record) = record {
                records.push((stem.to_string(), record));
            }
        } else if let Some(stem) = name.strip_suffix(".lock") {
            locks.push(stem.to_string());
        }
    }
    let mut out: Vec<Downloaded> = Vec::new();
    for stem in locks {
        if !records.iter().any(|(s, _)| *s == stem) {
            out.push(Downloaded::guessed(guess_id(&stem, catalog)));
        }
    }
    out.extend(records.into_iter().map(|(_, r)| Downloaded::of(r)));
    out.sort_by(|a, b| a.id.to_ascii_lowercase().cmp(&b.id.to_ascii_lowercase()).then_with(|| a.id.cmp(&b.id)));
    out
}

/// The id a lock's name most likely stands for. A catalog id whose stem is the name, alone or with one of its
/// quants as the tag, is taken first; otherwise the first `_` is read as the `/` every id has, and the rest are
/// left as they are, as a `:` cannot be told from a `_` there.
pub fn guess_id(stem: &str, catalog: &HubCatalog) -> String {
    for entry in &catalog.entries {
        let Some(id) = entry.get("id").and_then(Value::as_str) else { continue };
        if crate::pulls::stem(id) == stem {
            return id.to_string();
        }
        if let Some(quant) = entry.get("quant").and_then(Value::as_str).filter(|q| !q.is_empty()) {
            let tagged = format!("{id}:{quant}");
            if crate::pulls::stem(&tagged) == stem {
                return tagged;
            }
        }
    }
    stem.replacen('_', "/", 1)
}

/// The hub catalog as `$SUPERFLUID_HOME/cache/hub-catalog.json` keeps it: baseRT's converted models, one entry per
/// variant (an id and a quant), several per id.
#[derive(Debug, Clone, Default)]
pub struct HubCatalog {
    /// The entries, each with the fields the cache has.
    pub entries: Vec<Map<String, Value>>,
    /// The catalog's own `updated`: the day it was made.
    pub updated: Option<String>,
    /// When the copy here was fetched (the file's modification time), seconds since the Unix epoch; None when
    /// there is no copy here.
    pub fetched: Option<u64>,
}

impl HubCatalog {
    /// The catalog kept at `file`; empty when there is none, or it cannot be read.
    pub fn load(file: &Path) -> HubCatalog {
        let fetched = std::fs::metadata(file)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        let mut catalog = HubCatalog::from_json(&std::fs::read_to_string(file).unwrap_or_default());
        catalog.fetched = fetched;
        catalog
    }

    pub fn from_json(text: &str) -> HubCatalog {
        let v: Value = serde_json::from_str(text).unwrap_or(Value::Null);
        let entries = v["models"].as_array().into_iter().flatten().filter_map(|e| e.as_object().cloned()).collect();
        HubCatalog { entries, updated: v["updated"].as_str().map(str::to_string), fetched: None }
    }

    fn ids(&self) -> usize {
        self.entries.iter().filter_map(|e| e.get("id").and_then(Value::as_str)).collect::<HashSet<_>>().len()
    }
}

/// Whether a catalog entry is downloaded here. It is when a model listed as downloaded, with a record and a path
/// that is there, has the entry's `id` as its served name (ASCII case aside, as the Hub has it) and names the
/// entry's variant in one of three ways:
/// - its `:tag` is the entry's `quant` (`basecompute/Qwen3-4B:default-q8`);
/// - its path is a `model.base` in a directory named for the `quant`, which is how baseRT's hub cache keeps a
///   variant (`.../basecompute/Qwen3-4B/default-q4/model.base`), wherever a pull without a tag lands;
/// - its path's file name is the entry's `file` (a flat `Qwen3-4B-Q4.base`).
///
/// A model that names none of these (no tag, and a path that says nothing of the variant) matches no entry: its
/// runtime chose the variant, and the record does not say which. A guessed id or a missing path matches nothing.
pub fn is_downloaded(entry: &Map<String, Value>, here: &[Downloaded]) -> bool {
    let field = |k: &str| entry.get(k).and_then(Value::as_str).filter(|s| !s.is_empty());
    let Some(id) = field("id") else { return false };
    let (quant, file) = (field("quant"), field("file"));
    here.iter().filter(|d| !d.id_guessed && d.here() && d.served_name.eq_ignore_ascii_case(id)).any(|d| {
        let tag = d.id.split_once(':').map(|(_, t)| t);
        let path = d.path.as_deref();
        let name = path.and_then(Path::file_name).and_then(|n| n.to_str());
        let variant_dir = path.filter(|_| name == Some("model.base")).and_then(Path::parent).and_then(Path::file_name);
        (quant.is_some() && (tag == quant || variant_dir.and_then(|v| v.to_str()) == quant)) || (file.is_some() && name == file)
    })
}

/// The document `superfluid models --json` prints.
pub fn json(here: &[Downloaded], catalog: &HubCatalog) -> Value {
    let entries: Vec<Value> = catalog
        .entries
        .iter()
        .map(|e| {
            let mut e = e.clone();
            e.insert("downloaded".into(), Value::Bool(is_downloaded(&e, here)));
            Value::Object(e)
        })
        .collect();
    json!({
        "schema": SCHEMA,
        "downloaded": here.iter().map(Downloaded::json).collect::<Vec<_>>(),
        "catalog_updated": catalog.updated,
        "catalog_fetched": catalog.fetched,
        "catalog": entries,
    })
}

/// `superfluid models`: the models downloaded here, then a line on the catalog.
pub fn text(here: &[Downloaded], catalog: &HubCatalog, now: u64, home: Option<&Path>) -> String {
    let mut out = String::new();
    if here.is_empty() {
        out.push_str("No model is pulled here yet: `superfluid serve <org/model[:tag]>` pulls one on first use.\n");
    } else {
        let mut rows = vec![["ID", "RUNTIME", "SIZE", "PATH"].map(String::from).to_vec()];
        for d in here {
            let path = match (&d.path, d.exists) {
                (Some(p), Some(false)) => format!("missing: {}", tilde(p, home)),
                (Some(p), _) => tilde(p, home),
                (None, _) => "-".into(),
            };
            let id = if d.id_guessed { format!("{}*", d.id) } else { d.id.clone() };
            rows.push(vec![id, d.runtime.clone().unwrap_or_else(|| "-".into()), d.size.map(bytes).unwrap_or_else(|| "-".into()), path]);
        }
        out.push_str(&crate::runtimes_report::columns(&rows));
        if here.iter().any(|d| d.exists == Some(false)) {
            out.push_str("missing: the model is no longer where it was pulled to; `superfluid serve <id>` pulls it again\n");
        }
        if here.iter().any(|d| d.id_guessed) {
            out.push_str(
                "*: no record of the pull (one from before superfluid kept them, or one that failed): the id is\n   \
                 read back from its lock's name, where '/', ':' and '_' are all '_'; its next pull records it\n",
            );
        }
    }
    out.push('\n');
    out.push_str(&catalog_line(here, catalog, now));
    out.push_str(if catalog.entries.is_empty() { "\n" } else { "\n`superfluid models --catalog` lists them.\n" });
    out
}

/// `superfluid models --catalog`: the catalog's entries, marking those downloaded here.
pub fn catalog_text(here: &[Downloaded], catalog: &HubCatalog, now: u64) -> String {
    let mut out = String::new();
    if !catalog.entries.is_empty() {
        let mut rows = vec![["MODEL", "SIZE", "ARCH", "BACKEND", "HERE"].map(String::from).to_vec()];
        for e in &catalog.entries {
            let field = |k: &str| e.get(k).and_then(Value::as_str).filter(|s| !s.is_empty());
            let model = match (field("id"), field("quant")) {
                (Some(id), Some(q)) => format!("{id}:{q}"),
                (Some(id), None) => id.to_string(),
                (None, _) => continue,
            };
            let size = e.get("size").and_then(Value::as_u64).map(bytes).unwrap_or_else(|| "-".into());
            let at = |k: &str| field(k).unwrap_or("-").to_string();
            let mark = if is_downloaded(e, here) { "downloaded" } else { "" };
            rows.push(vec![model, size, at("arch"), at("backend"), mark.to_string()]);
        }
        out.push_str(&crate::runtimes_report::columns(&rows));
        out.push('\n');
    }
    out.push_str(&catalog_line(here, catalog, now));
    out.push_str(if catalog.entries.is_empty() { "\n" } else { "\n`superfluid serve <model>` pulls one.\n" });
    out
}

fn catalog_line(here: &[Downloaded], catalog: &HubCatalog, now: u64) -> String {
    let Some(fetched) = catalog.fetched else {
        return "No hub catalog is kept here yet: `superfluid models --refresh` fetches it.".into();
    };
    if catalog.entries.is_empty() {
        return format!("The hub catalog kept here (fetched {}) lists no model: `superfluid models --refresh` fetches it again.", ago(now, fetched));
    }
    let downloaded = catalog.entries.iter().filter(|e| is_downloaded(e, here)).count();
    let dated = catalog.updated.as_deref().map(|d| format!("dated {d}, ")).unwrap_or_default();
    format!(
        "Hub catalog: {} variants of {} models, {downloaded} downloaded here; {dated}fetched {}.",
        catalog.entries.len(),
        catalog.ids(),
        ago(now, fetched)
    )
}

/// `superfluid models [--catalog] [--refresh] [--json]`; the exit status.
pub fn run(args: &[String]) -> i32 {
    let (mut as_json, mut list_catalog, mut refresh) = (false, false, false);
    for a in args {
        match a.as_str() {
            "--json" => as_json = true,
            "--catalog" => list_catalog = true,
            "--refresh" => refresh = true,
            other => {
                eprintln!("superfluid models: no option {other:?}");
                eprint!("{}", crate::cli::MODELS_HELP);
                return 2;
            }
        }
    }
    let home = crate::runtimes::default_home();
    let file = crate::speculate_resolve::catalog_file(&home);
    let mut code = 0;
    if refresh {
        if let Err(e) = crate::speculate_resolve::refresh_catalog(&file) {
            eprintln!("superfluid: the hub catalog could not be fetched ({e}); listing the copy kept here");
            code = 1;
        }
    }
    let catalog = HubCatalog::load(&file);
    let here = downloaded(&home, &catalog);
    if as_json {
        println!("{}", serde_json::to_string_pretty(&json(&here, &catalog)).expect("json"));
    } else if list_catalog {
        print!("{}", catalog_text(&here, &catalog, now_secs()));
    } else {
        print!("{}", text(&here, &catalog, now_secs(), std::env::home_dir().as_deref()));
    }
    code
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn tilde(path: &Path, home: Option<&Path>) -> String {
    crate::runtimes_report::tilde(&path.display().to_string(), home)
}

/// A size as the install plans print theirs: decimal units, one decimal place.
fn bytes(n: u64) -> String {
    let f = n as f64;
    if f >= 1e9 {
        format!("{:.1} GB", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.1} MB", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.1} kB", f / 1e3)
    } else {
        format!("{n} B")
    }
}

fn ago(now: u64, then: u64) -> String {
    let secs = now.saturating_sub(then);
    let (n, unit) = match secs {
        0..60 => return "just now".into(),
        60..3600 => (secs / 60, "minute"),
        3600..86_400 => (secs / 3600, "hour"),
        _ => (secs / 86_400, "day"),
    };
    format!("{n} {unit}{} ago", if n == 1 { "" } else { "s" })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("superfluid-models-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn file(path: &Path, len: usize) -> PathBuf {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, vec![0u8; len]).unwrap();
        path.to_path_buf()
    }

    fn catalog() -> HubCatalog {
        let mut c = HubCatalog::from_json(
            &json!({
                "schema": 1,
                "updated": "2026-11-06",
                "models": [
                    {"id": "basecompute/Qwen3-4B", "file": "Qwen3-4B-Q4.base", "quant": "default-q4", "size": 2, "arch": "qwen3"},
                    {"id": "basecompute/Qwen3-4B", "file": "Qwen3-4B-Q8.base", "quant": "default-q8", "size": 4, "arch": "qwen3"},
                    {"id": "basecompute/Qwen3-4B", "file": "Qwen3-4B-cuda-q8.base", "quant": "cuda-q8", "size": 4, "backend": "cuda"},
                    {"id": "basecompute/Flat", "file": "Flat-Q4.base", "quant": "default-q4", "size": 1},
                    {"id": "basecompute/Qwen3.6-35B-A3B-DFlash2", "file": "x.base", "quant": "default-q4", "speculator_auto": true},
                ],
            })
            .to_string(),
        );
        c.fetched = Some(1_000);
        c
    }

    #[test]
    fn a_pull_is_recorded_beside_its_lock_and_read_back_as_it_was() {
        let home = scratch("record");
        let model = file(&home.join("hub/org/m/default-q4/model.base"), 10);
        let id = "org/m:default-q4";
        let r = record(&home, id, RuntimeId::new("basert@0.5.0-metal"), &model).unwrap();
        assert_eq!((r.schema, r.served_name.as_str(), r.runtime.as_str(), r.size), (1, "org/m", "basert", 10));
        assert!(home.join("pulls/org_m_default-q4.json").is_file());
        let here = downloaded(&home, &HubCatalog::default());
        assert_eq!(
            here,
            vec![Downloaded {
                id: id.into(),
                served_name: "org/m".into(),
                runtime: Some("basert".into()),
                path: Some(model.clone()),
                size: Some(10),
                pulled_at: Some(r.pulled_at),
                exists: Some(true),
                id_guessed: false,
            }]
        );
        // A later pull of it (an offline one that finds it in place, say) replaces the record.
        file(&model, 12);
        record(&home, id, RuntimeId::new("basert"), &model).unwrap();
        assert_eq!(downloaded(&home, &HubCatalog::default())[0].size, Some(12));
        let left: Vec<_> = std::fs::read_dir(home.join("pulls")).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(left, ["org_m_default-q4.json"], "no temporary file is left behind");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_record_keeps_the_id_its_lock_name_cannot_give_back() {
        let home = scratch("ids");
        let model = file(&home.join("m.base"), 1);
        let ids = ["basecompute/Qwen3.6-35B-A3B-DFlash2:default-q4", "org/my_model", "unsloth/Qwen3-0.6B-GGUF:Q4_K_M"];
        for id in ids {
            crate::pulls::Lock::take(&home.join("pulls"), id, "a test").unwrap();
            record(&home, id, RuntimeId::new("llamacpp"), &model).unwrap();
        }
        assert!(home.join("pulls/basecompute_Qwen3.6-35B-A3B-DFlash2_default-q4.lock").is_file());
        let here = downloaded(&home, &HubCatalog::default());
        let got: Vec<(&str, &str, bool)> = here.iter().map(|d| (d.id.as_str(), d.served_name.as_str(), d.id_guessed)).collect();
        assert_eq!(
            got,
            [
                ("basecompute/Qwen3.6-35B-A3B-DFlash2:default-q4", "basecompute/Qwen3.6-35B-A3B-DFlash2", false),
                ("org/my_model", "org/my_model", false),
                ("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", "unsloth/Qwen3-0.6B-GGUF", false),
            ]
        );
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_model_whose_path_is_gone_is_listed_as_missing() {
        let home = scratch("missing");
        let model = file(&home.join("gone.gguf"), 3);
        record(&home, "org/gone-GGUF:Q4_K_M", RuntimeId::new("llamacpp"), &model).unwrap();
        std::fs::remove_file(&model).unwrap();
        let here = downloaded(&home, &HubCatalog::default());
        assert_eq!((here.len(), here[0].exists, here[0].size), (1, Some(false), Some(3)));
        let out = text(&here, &HubCatalog::default(), 0, None);
        assert!(out.contains(&format!("missing: {}", model.display())), "{out}");
        assert!(out.contains("`superfluid serve <id>` pulls it again"), "{out}");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_lock_from_before_records_is_listed_with_its_id_guessed() {
        let home = scratch("legacy");
        let pulls = home.join("pulls");
        for name in ["basecompute_Qwen3.6-35B-A3B-DFlash2_default-q4", "basecompute_Qwen3-4B", "someone_my_model", ".hidden"] {
            file(&pulls.join(format!("{name}.lock")), 0);
        }
        // A record that cannot be read is no record: its lock is listed as guessed.
        file(&pulls.join("basecompute_Qwen3-4B.json"), 0);
        let here = downloaded(&home, &catalog());
        let got: Vec<(&str, bool, Option<bool>)> = here.iter().map(|d| (d.id.as_str(), d.id_guessed, d.exists)).collect();
        assert_eq!(
            got,
            [
                ("basecompute/Qwen3-4B", true, None),
                ("basecompute/Qwen3.6-35B-A3B-DFlash2:default-q4", true, None),
                ("someone/my_model", true, None),
            ]
        );
        assert!(here.iter().all(|d| d.path.is_none() && d.runtime.is_none() && d.size.is_none()));
        let out = text(&here, &catalog(), 1_000, None);
        let row = |id: &str| out.lines().map(|l| l.split_whitespace().collect::<Vec<_>>()).find(|w| w[0] == id);
        assert_eq!(row("basecompute/Qwen3-4B*"), Some(vec!["basecompute/Qwen3-4B*", "-", "-", "-"]), "{out}");
        assert!(out.contains("where '/', ':' and '_' are all '_'"), "{out}");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_catalog_entry_is_downloaded_when_a_record_names_its_variant() {
        let home = scratch("match");
        let hub = home.join("hub");
        let q4 = file(&hub.join("basecompute/Qwen3-4B/default-q4/model.base"), 2);
        let q8 = file(&hub.join("elsewhere/q8.base"), 4);
        let flat = file(&hub.join("Flat-Q4.base"), 1);
        let basert = RuntimeId::new("basert");
        record(&home, "basecompute/Qwen3-4B", basert, &q4).unwrap();
        record(&home, "basecompute/qwen3-4b:default-q8", basert, &q8).unwrap();
        record(&home, "basecompute/Flat", basert, &flat).unwrap();
        // No tag and a path that says nothing of the variant: no entry is claimed.
        let other = file(&hub.join("some.base"), 1);
        record(&home, "basecompute/Qwen3.6-35B-A3B-DFlash2", basert, &other).unwrap();
        let cat = catalog();
        let marks = |here: &[Downloaded]| -> Vec<bool> { cat.entries.iter().map(|e| is_downloaded(e, here)).collect() };
        assert_eq!(marks(&downloaded(&home, &cat)), [true, true, false, true, false]);
        // A path that is gone, or an id guessed from a lock, claims nothing.
        std::fs::remove_file(&q8).unwrap();
        file(&home.join("pulls/basecompute_Qwen3.6-35B-A3B-DFlash2_default-q4.lock"), 0);
        assert_eq!(marks(&downloaded(&home, &cat)), [true, false, false, true, false]);
        let out = catalog_text(&downloaded(&home, &cat), &cat, 1_000 + 2 * 86_400);
        let row = |model: &str| out.lines().map(|l| l.split_whitespace().collect::<Vec<_>>()).find(|w| w[0] == model);
        assert_eq!(row("basecompute/Qwen3-4B:default-q4"), Some(vec!["basecompute/Qwen3-4B:default-q4", "2", "B", "qwen3", "-", "downloaded"]));
        assert_eq!(row("basecompute/Qwen3-4B:cuda-q8"), Some(vec!["basecompute/Qwen3-4B:cuda-q8", "4", "B", "-", "cuda"]));
        assert!(out.ends_with("\nHub catalog: 5 variants of 3 models, 2 downloaded here; dated 2026-11-06, fetched 2 days ago.\n`superfluid serve <model>` pulls one.\n"), "{out}");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn the_json_document_has_both_lists_and_its_version() {
        let home = scratch("json");
        let model = file(&home.join("hub/basecompute/Qwen3-4B/default-q4/model.base"), 2);
        record(&home, "basecompute/Qwen3-4B:default-q4", RuntimeId::new("basert"), &model).unwrap();
        file(&home.join("pulls/someone_x.lock"), 0);
        let cat = catalog();
        let doc = json(&downloaded(&home, &cat), &cat);
        let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys(&doc), ["schema", "downloaded", "catalog_updated", "catalog_fetched", "catalog"]);
        assert_eq!((doc["schema"].clone(), doc["catalog_updated"].clone(), doc["catalog_fetched"].clone()), (json!(1), json!("2026-11-06"), json!(1_000)));
        let fields = ["id", "served_name", "runtime", "path", "size", "pulled_at", "exists", "id_guessed"];
        let rows = doc["downloaded"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| keys(r) == fields));
        assert_eq!(rows[0]["id"], "basecompute/Qwen3-4B:default-q4");
        assert_eq!((rows[0]["path"].as_str(), rows[0]["size"].as_u64(), rows[0]["exists"].as_bool()), (model.to_str(), Some(2), Some(true)));
        assert!(rows[0]["pulled_at"].is_u64());
        assert_eq!(rows[1], json!({"id": "someone/x", "served_name": "someone/x", "runtime": null, "path": null, "size": null,
                                    "pulled_at": null, "exists": null, "id_guessed": true}));
        let entries = doc["catalog"].as_array().unwrap();
        assert_eq!(entries.len(), 5);
        assert_eq!(
            entries[0],
            json!({"id": "basecompute/Qwen3-4B", "file": "Qwen3-4B-Q4.base", "quant": "default-q4", "size": 2, "arch": "qwen3", "downloaded": true}),
            "an entry keeps the fields the cache has, in its order, and says whether it is here"
        );
        assert_eq!(entries[4]["speculator_auto"], true);
        assert!(entries[1..].iter().all(|e| e["downloaded"] == false));
        let none = json(&[], &HubCatalog::default());
        assert_eq!(none, json!({"schema": 1, "downloaded": [], "catalog_updated": null, "catalog_fetched": null, "catalog": []}));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_catalog_is_read_with_its_date_and_when_it_was_fetched() {
        let dir = scratch("catalog");
        let f = dir.join("cache/hub-catalog.json");
        let none = HubCatalog::load(&f);
        assert!(none.entries.is_empty() && none.fetched.is_none() && none.updated.is_none());
        assert!(text(&[], &none, 0, None).contains("No hub catalog is kept here yet: `superfluid models --refresh` fetches it"));
        file(&f, 0);
        std::fs::write(&f, json!({"updated": "2026-11-06", "models": [{"id": "a/b", "quant": "q"}, 7]}).to_string()).unwrap();
        let c = HubCatalog::load(&f);
        assert_eq!((c.entries.len(), c.updated.as_deref()), (1, Some("2026-11-06")));
        assert!(c.fetched.is_some_and(|t| now_secs().abs_diff(t) < 60));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_models_size_counts_each_file_once_and_walks_no_linked_directory() {
        use std::os::unix::fs::symlink;
        let dir = scratch("size");
        let blobs = dir.join("hub/blobs");
        let weights = file(&blobs.join("7f/7f99"), 1000);
        let config = file(&dir.join("hub/models--org--m/blobs/6c97"), 10);
        let snap = dir.join("hub/models--org--m/snapshots/abc");
        std::fs::create_dir_all(snap.join("sub")).unwrap();
        // A snapshot links into blobs, here outside the repository's directory as a deduplicated cache has it.
        symlink(&weights, snap.join("model.safetensors")).unwrap();
        symlink("../../blobs/6c97", snap.join("config.json")).unwrap();
        symlink(&weights, snap.join("sub/same-blob-again")).unwrap();
        file(&snap.join("tokenizer.json"), 5);
        std::fs::hard_link(snap.join("tokenizer.json"), snap.join("sub/tokenizer-hardlink.json")).unwrap();
        symlink(&blobs, snap.join("linked-dir")).unwrap();
        symlink(dir.join("nowhere"), snap.join("dangling")).unwrap();
        file(&blobs.join("other-revision"), 99_999);
        assert_eq!(size_on_disk(&snap), 1000 + 10 + 5);
        assert_eq!(size_on_disk(&snap.join("model.safetensors")), 1000, "a pulled file is the blob it links to");
        assert_eq!(size_on_disk(&config), 10);
        assert_eq!(size_on_disk(&dir.join("nowhere")), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sizes_and_ages_read_the_way_the_rest_of_the_cli_prints_them() {
        assert_eq!([bytes(0), bytes(999), bytes(1_500), bytes(20_515_667_968)], ["0 B", "999 B", "1.5 kB", "20.5 GB"]);
        assert_eq!([ago(100, 70), ago(200, 80), ago(7_300, 100), ago(90_000, 0), ago(0, 50)], ["just now", "2 minutes ago", "2 hours ago", "1 day ago", "just now"]);
    }

    #[test]
    fn models_has_help_that_fits_the_terminal() {
        assert!(crate::cli::COMMANDS.contains(&"models"));
        let help = crate::cli::command_help("models").unwrap();
        assert!(help.starts_with("List the models pulled here"), "{help}");
        for flag in ["--json", "--catalog", "--refresh"] {
            assert!(help.contains(flag), "{flag}");
        }
        assert!(help.lines().all(|l| l.chars().count() <= 100), "{help}");
        assert!(crate::cli::overview().contains("\n  models [--json]      List the models pulled here and the hub catalog's\n"));
    }
}
