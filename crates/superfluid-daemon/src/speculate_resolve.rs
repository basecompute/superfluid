//! `--speculate` without knowing drafter formats.

use std::cmp::Reverse;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

const PATH_STRATEGIES: [&str; 4] = ["dflash", "dspark", "eagle3", "draft-model"];

const AUTO_KINDS: [&str; 3] = ["dspark", "dflash", "eagle3"];

const HUB_DEPTH: usize = 3;

const CATALOG_URL: &str = "https://raw.githubusercontent.com/basecompute/baseRT/main/base-convert/crates/base-hub/catalog.json";

const CATALOG_MAX_AGE_SECS: u64 = 24 * 3600;

static FETCH_CATALOG: AtomicBool = AtomicBool::new(false);

/// Whether a stale catalog may be refreshed over the network (`serve` without `--offline`).
pub fn allow_catalog_fetch(yes: bool) {
    FETCH_CATALOG.store(yes, Ordering::Relaxed);
}

#[derive(serde::Deserialize, Default)]
struct CatalogFile {
    #[serde(default)]
    models: Vec<CatalogRow>,
}

#[derive(serde::Deserialize, Clone)]
struct CatalogRow {
    id: String,
    #[serde(default)]
    file: String,
    #[serde(default)]
    speculator_for: Option<String>,
    #[serde(default)]
    speculator_rank: Option<u32>,
    #[serde(default)]
    quant: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    sha256: Option<String>,
}

#[derive(Default)]
pub struct Pairings {
    rows: Vec<CatalogRow>,
    digests: Option<PathBuf>,
}

impl Pairings {
    pub fn load(hub: Option<&Path>) -> Self {
        let cache = crate::runtimes::default_home().join("cache");
        let mut p = Self::from_json(&catalog_json(&cache.join("hub-catalog.json"), FETCH_CATALOG.load(Ordering::Relaxed)));
        if let Some(theirs) = hub.and_then(|h| std::fs::read_to_string(h.join(".catalog-cache.json")).ok()) {
            p.rows.extend(Self::from_json(&theirs).rows);
        }
        p.digests = Some(cache.join("sha256"));
        p
    }

    pub fn from_json(json: &str) -> Self {
        Self { rows: serde_json::from_str::<CatalogFile>(json).map(|c| c.models).unwrap_or_default(), digests: None }
    }

    fn pinned_row(&self, path: &Path) -> Option<&CatalogRow> {
        let name = path.file_name()?.to_str()?;
        let pinned = |r: &&CatalogRow| r.sha256.as_deref().is_some_and(|d| !d.is_empty());
        if name == "model.base" {
            let id = self.id_of(path)?;
            let variant = path.parent()?.file_name()?.to_str()?;
            return self.rows.iter().filter(pinned).find(|r| r.id == id && r.quant == variant);
        }
        self.rows.iter().filter(pinned).find(|r| r.file == name)
    }

    /// Why a bundle the catalog pins may not be used: its bytes are not the ones the catalog
    /// lists, which is a partial or corrupt download, or another file under a catalog name.
    pub fn digest_refusal(&self, path: &Path) -> Option<String> {
        let row = self.pinned_row(path)?;
        let want = row.sha256.as_deref()?.to_ascii_lowercase();
        let meta = std::fs::metadata(path).ok()?;
        let what = format!("{} {}", row.id, if row.quant.is_empty() { &row.file } else { &row.quant });
        let got = match row.size {
            Some(size) if size != meta.len() => format!("{} bytes, not {size}", meta.len()),
            _ => match file_sha256(path, &meta, self.digests.as_deref())? {
                got if got == want => return None,
                got => format!("sha256 {got}, not {want}"),
            },
        };
        Some(format!(
            "{} is not the catalog's {what} ({got}); pull it again, or rename it if it is a conversion of your own",
            path.display()
        ))
    }

    fn id_of(&self, path: &Path) -> Option<String> {
        let name = path.file_name()?.to_str()?;
        if name == "model.base" {
            let repo = path.parent()?.parent()?;
            let ns = repo.parent()?.file_name()?.to_str()?;
            return Some(format!("{ns}/{}", repo.file_name()?.to_str()?));
        }
        self.rows.iter().find(|r| r.file == name).map(|r| r.id.clone())
    }

    fn rank(&self, drafter: &Path, target_id: &str) -> Option<u32> {
        let id = self.id_of(drafter)?;
        let rows = || self.rows.iter().filter(|r| r.id == id);
        if !rows().any(|r| r.speculator_for.as_deref() == Some(target_id)) {
            return None;
        }
        Some(rows().find_map(|r| r.speculator_rank).unwrap_or(u32::MAX))
    }
}

/// The hub catalog kept at `own`, refreshed from `CATALOG_URL` once a day when `fetch` allows.
fn catalog_json(own: &Path, fetch: bool) -> String {
    let age = std::fs::metadata(own).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok());
    if fetch && age.is_none_or(|a| a.as_secs() >= CATALOG_MAX_AGE_SECS) {
        if let Ok(v) = superfluid_adapter_kit::install::fetch_json(CATALOG_URL) {
            let body = v.to_string();
            if own.parent().is_some_and(|d| std::fs::create_dir_all(d).is_ok()) && std::fs::write(own, &body).is_ok() {
                return body;
            }
        }
    }
    std::fs::read_to_string(own).unwrap_or_default()
}

/// The file's sha256, remembered under `cache` against its device, inode, size, mtime and ctime
/// so a multi-gigabyte bundle is hashed once rather than at every start. `None` if unreadable.
fn file_sha256(path: &Path, meta: &std::fs::Metadata, cache: Option<&Path>) -> Option<String> {
    use sha2::Digest;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;
    let stamp = |m: &std::fs::Metadata| {
        format!(
            "{} {} {} {}.{} {}.{}",
            m.dev(), m.ino(), m.len(), m.mtime(), m.mtime_nsec(), m.ctime(), m.ctime_nsec()
        )
    };
    let hex = |d: &[u8]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let before = stamp(meta);
    let entry = cache.and_then(|dir| {
        let canon = path.canonicalize().ok()?;
        Some(dir.join(hex(&sha2::Sha256::digest(canon.as_os_str().as_bytes()))))
    });
    if let Some(remembered) = entry.as_ref().and_then(|e| std::fs::read_to_string(e).ok()) {
        if let Some((st, digest)) = remembered.trim().split_once('\n') {
            if st == before {
                return Some(digest.to_string());
            }
        }
    }
    let mut f = std::fs::File::open(path).ok()?;
    if meta.len() >= 1 << 30 {
        eprintln!("superfluid: checking {} against the hub catalog's sha256 (once per file)", path.display());
    }
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => h.update(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
    let digest = hex(&h.finalize());
    let unchanged = std::fs::metadata(path).is_ok_and(|m| stamp(&m) == before);
    if let (Some(entry), true) = (entry, unchanged) {
        if let Some(dir) = entry.parent() {
            let tmp = entry.with_extension(format!("tmp{}", std::process::id()));
            let wrote = std::fs::create_dir_all(dir).is_ok()
                && std::fs::write(&tmp, format!("{before}\n{digest}\n")).is_ok()
                && std::fs::rename(&tmp, &entry).is_ok();
            if !wrote {
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }
    Some(digest)
}

pub fn read_base_header(path: &Path) -> Option<serde_json::Value> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut prefix = [0u8; 16];
    f.read_exact(&mut prefix).ok()?;
    if &prefix[..4] != b"BASE" || u32::from_le_bytes(prefix[4..8].try_into().ok()?) != 1 {
        return None;
    }
    let len = u64::from_le_bytes(prefix[8..16].try_into().ok()?);
    if len == 0 || len > (256 << 20) {
        return None;
    }
    let mut json = vec![0u8; len as usize];
    f.read_exact(&mut json).ok()?;
    serde_json::from_slice(&json).ok()
}

pub fn header_speculator_kind(h: &serde_json::Value) -> Option<&str> {
    if let Some(k) = h
        .get("speculator")
        .and_then(|s| s.get("config"))
        .and_then(|c| c.get("kind"))
        .and_then(|k| k.as_str())
    {
        return Some(k);
    }
    if h.get("arch").and_then(|a| a.as_str()) != Some("glm_dsa") {
        return None;
    }
    let layers = h
        .get("config")
        .and_then(|c| c.get("num_hidden_layers"))
        .and_then(|n| n.as_u64())?;
    let want = format!("layers.{layers}.eh_proj.weight");
    h.get("tensors")
        .and_then(|t| t.as_array())?
        .iter()
        .any(|t| t.get("name").and_then(|n| n.as_str()) == Some(want.as_str()))
        .then_some("mtp")
}

pub fn strategy_for_drafter_header(h: &serde_json::Value) -> &'static str {
    match h.get("arch").and_then(|v| v.as_str()).unwrap_or("") {
        "dflash" => "dflash",
        "dspark" => "dspark",
        "eagle3" => "eagle3",
        _ => "draft-model",
    }
}

pub fn resolve(directive: &str, target: &Path) -> Result<Option<String>, String> {
    let d = directive.trim();
    if d == "off" || d.is_empty() {
        return Ok(None);
    }
    if d == "prompt-lookup" {
        return Ok(Some(d.to_string()));
    }
    if d == "mtp-head" {
        if let Some(h) = read_base_header(target) {
            let kind = header_speculator_kind(&h);
            if kind != Some("mtp") {
                let built = h
                    .get("baserT_version")
                    .and_then(|v| v.as_str())
                    .map(|v| format!(" (converted by baseRT {v})"))
                    .unwrap_or_default();
                return Err(match kind {
                    None => format!(
                        "--speculate mtp-head: {} carries no speculator head{built}; reconvert it with \
                         base-convert >= 0.2.5 from a checkpoint that ships an MTP head, or use \
                         --speculate auto to pair an installed drafter",
                        target.display()
                    ),
                    Some(k) => format!(
                        "--speculate mtp-head: {}'s speculator head is `{k}`, not an MTP head",
                        target.display()
                    ),
                });
            }
        }
        return Ok(Some(d.to_string()));
    }
    if let Some((kind, path)) = d.split_once(':') {
        if PATH_STRATEGIES.contains(&kind) {
            let path = expand_tilde(path);
            if let Some(why) = catalog_refusal(Path::new(&path)) {
                return Err(format!("--speculate {d}: {why}"));
            }
            return Ok(Some(format!("{kind}:{path}")));
        }
    }
    if d == "auto" {
        return resolve_all(d, target).map(|ranked| ranked.into_iter().next());
    }
    let d = expand_tilde(d);
    let d = d.as_str();
    let p = Path::new(d);
    if p.is_file() {
        let h = read_base_header(p)
            .ok_or_else(|| format!("--speculate {d}: not a .base bundle (convert the drafter first)"))?;
        if let Some(why) = catalog_refusal(p) {
            return Err(format!("--speculate {d}: {why}"));
        }
        return Ok(Some(format!("{}:{}", strategy_for_drafter_header(&h), d)));
    }
    Err(format!(
        "--speculate {d}: expected auto | prompt-lookup | mtp-head | a drafter .base path | \
         dflash/dspark/eagle3/draft-model:<path>"
    ))
}

pub fn resolve_all(directive: &str, target: &Path) -> Result<Vec<String>, String> {
    let d = directive.trim();
    if d != "auto" {
        return resolve(d, target).map(|r| r.into_iter().collect());
    }
    let h = read_base_header(target)
        .ok_or_else(|| format!("--speculate auto: cannot read the bundle header of {}", target.display()))?;
    let head = header_speculator_kind(&h);
    let mut ranked = Vec::new();
    if head == Some("mtp") {
        ranked.push("mtp-head".to_string());
    }
    let pairings = Pairings::load(hub_roots().first().map(PathBuf::as_path));
    ranked.extend(
        find_drafters(target, &h, &auto_search_dirs(target), &pairings)
            .into_iter()
            .map(|(kind, p)| format!("{kind}:{}", p.display())),
    );
    Ok(ranked)
}

fn catalog_refusal(path: &Path) -> Option<String> {
    if !path.is_file() {
        return None;
    }
    Pairings::load(hub_roots().first().map(PathBuf::as_path)).digest_refusal(path)
}

fn expand_tilde(path: &str) -> String {
    let rest = match path.strip_prefix('~') {
        Some("") => "",
        Some(r) if r.starts_with('/') => r,
        _ => return path.to_string(),
    };
    match std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        Some(home) => format!("{}{rest}", Path::new(&home).display()),
        None => path.to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecLoad {
    EmbeddedHead,
    Sidecar,
    DraftModel,
}

pub fn sizing_path(resolved: &str, target: &Path) -> Option<(PathBuf, SpecLoad)> {
    if resolved == "mtp-head" {
        return Some((target.to_path_buf(), SpecLoad::EmbeddedHead));
    }
    match resolved.split_once(':') {
        Some(("draft-model", path)) => Some((PathBuf::from(path), SpecLoad::DraftModel)),
        Some((kind, path)) if PATH_STRATEGIES.contains(&kind) => Some((PathBuf::from(path), SpecLoad::Sidecar)),
        _ => None,
    }
}

pub fn auto_search_dirs(target: &Path) -> Vec<(PathBuf, usize)> {
    let dir = match target.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let mut dirs = vec![(dir.clone(), 0), (dir.join("drafters"), 0)];
    let up = match dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => Some(p.to_path_buf()),
        Some(_) if matches!(dir.components().next_back(), Some(Component::Normal(_))) => Some(PathBuf::from(".")),
        Some(_) => Some(dir.join("..")),
        None => None,
    };
    if let Some(up) = up {
        dirs.push((up.join("drafters"), 0));
    }
    for hub in hub_roots() {
        dirs.push((hub, HUB_DEPTH));
    }
    dirs
}

fn hub_roots() -> Vec<PathBuf> {
    let env = |k| std::env::var_os(k).filter(|v| !v.is_empty());
    hub_roots_from(env("BASERT_MODELS_DIR"), env("HOME"), env("XDG_CACHE_HOME"))
}

fn hub_roots_from(
    models_dir: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
    xdg_cache: Option<std::ffi::OsString>,
) -> Vec<PathBuf> {
    if let Some(v) = models_dir {
        return vec![PathBuf::from(v)];
    }
    let Some(home) = home.map(PathBuf::from) else { return vec![] };
    let platform = if cfg!(target_os = "macos") {
        home.join("Library/Caches")
    } else {
        xdg_cache.map(PathBuf::from).filter(|p| p.is_absolute()).unwrap_or_else(|| home.join(".cache"))
    };
    let mut roots = vec![platform.join("baseRT/models")];
    let legacy = home.join(".cache/baseRT/models");
    if !roots.contains(&legacy) {
        roots.push(legacy);
    }
    roots
}

pub fn find_drafter(
    target: &Path,
    th: &serde_json::Value,
    dirs: &[(PathBuf, usize)],
    pairings: &Pairings,
) -> Option<(&'static str, PathBuf)> {
    find_drafters(target, th, dirs, pairings).into_iter().next()
}

pub fn find_drafters(
    target: &Path,
    th: &serde_json::Value,
    dirs: &[(PathBuf, usize)],
    pairings: &Pairings,
) -> Vec<(&'static str, PathBuf)> {
    let target_names = bundle_names(target, th);
    let target_id = pairings.id_of(target);
    let target_bits = storage_bits(th);
    let mut files = Vec::new();
    for (dir, depth) in dirs {
        collect_bundles(dir, *depth, &mut files);
    }
    let self_path = target.canonicalize().ok();
    let mut seen = std::collections::HashSet::new();
    let mut ranked: Vec<(_, &'static str, PathBuf)> = Vec::new();
    for path in files {
        let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
        if Some(&canon) == self_path.as_ref() || !seen.insert(canon) {
            continue;
        }
        let Some(dh) = read_base_header(&path) else { continue };
        let kind = strategy_for_drafter_header(&dh);
        let Some(kind_rank) = AUTO_KINDS.iter().position(|k| *k == kind) else { continue };
        if !fits(th, &dh) {
            continue;
        }
        if let Some(why) = pairings.digest_refusal(&path) {
            eprintln!("superfluid: speculation: skipping a drafter: {why}");
            continue;
        }
        let named = bundle_names(&path, &dh)
            .iter()
            .any(|d| target_names.iter().any(|t| d.contains(t.as_str())));
        let same_quant = target_bits.is_some() && storage_bits(&dh) == target_bits;
        let created = dh.get("created").and_then(|v| v.as_str()).map_or(0, created_secs);
        let paired = target_id.as_deref().and_then(|t| pairings.rank(&path, t));
        let key = (paired.is_none(), paired, !named, kind_rank, !same_quant, Reverse(created), path.clone());
        ranked.push((key, kind, path));
    }
    ranked.sort_by(|a, b| a.0.cmp(&b.0));
    ranked.into_iter().map(|(_, kind, p)| (kind, p)).collect()
}

fn created_secs(s: &str) -> u64 {
    let s = s.trim();
    s.parse::<u64>().ok().or_else(|| rfc3339_secs(s)).unwrap_or(0)
}

fn rfc3339_secs(s: &str) -> Option<u64> {
    let num = |r: std::ops::Range<usize>| {
        let t = s.get(r)?;
        t.bytes().all(|b| b.is_ascii_digit()).then(|| t.parse::<i64>().ok())?
    };
    let b = s.as_bytes();
    if b.len() < 19 || b[4] != b'-' || b[7] != b'-' || !matches!(b[10], b'T' | b't' | b' ') || b[13] != b':' || b[16] != b':'
    {
        return None;
    }
    let (y, mo, d, h, mi, sec) = (num(0..4)?, num(5..7)?, num(8..10)?, num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || sec > 60 {
        return None;
    }
    let mut rest = &s[19..];
    if let Some(frac) = rest.strip_prefix('.') {
        rest = frac.trim_start_matches(|c: char| c.is_ascii_digit());
    }
    let offset = match rest {
        "" | "Z" | "z" => 0,
        o if o.len() == 6 && o.as_bytes()[3] == b':' => {
            let sign = match o.as_bytes()[0] {
                b'+' => 1,
                b'-' => -1,
                _ => return None,
            };
            let (oh, om) = (o.get(1..3)?, o.get(4..6)?);
            let digits = |t: &str| t.bytes().all(|b| b.is_ascii_digit()).then(|| t.parse::<i64>().ok())?;
            sign * (digits(oh)? * 3600 + digits(om)? * 60)
        }
        _ => return None,
    };
    let y = if mo <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if mo > 2 { mo - 3 } else { mo + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    u64::try_from(days * 86_400 + h * 3600 + mi * 60 + sec - offset).ok()
}

fn fits(th: &serde_json::Value, dh: &serde_json::Value) -> bool {
    let num = |h: &serde_json::Value, k: &str| h.get("config").and_then(|c| c.get(k)).and_then(|v| v.as_u64());
    let (Some(hidden), Some(vocab)) = (num(th, "hidden_size"), num(th, "vocab_size")) else {
        return false;
    };
    if num(dh, "hidden_size") != Some(hidden) || num(dh, "vocab_size") != Some(vocab) {
        return false;
    }
    let layers = num(th, "num_hidden_layers");
    if let Some(n) = num(dh, "num_target_layers") {
        if layers != Some(n) {
            return false;
        }
    }
    if let Some(layers) = layers {
        for key in ["target_layer_ids", "aux_layer_ids"] {
            let ids = dh.get("config").and_then(|c| c.get(key)).and_then(|v| v.as_array());
            if ids.is_some_and(|ids| ids.iter().any(|i| i.as_u64().is_none_or(|i| i >= layers))) {
                return false;
            }
        }
    }
    let backend = |h: &serde_json::Value| match h.get("target_backend").and_then(|v| v.as_str()).unwrap_or("metal") {
        "cuda" | "cuda_sm121" => "cuda".to_owned(),
        other => other.to_owned(),
    };
    backend(th) == backend(dh)
}

fn bundle_names(path: &Path, h: &serde_json::Value) -> Vec<String> {
    let mut raw = Vec::new();
    if let Some(s) = h.get("source").and_then(|s| s.get("filename")).and_then(|v| v.as_str()) {
        raw.push(s.to_string());
    }
    match path.file_stem().and_then(|s| s.to_str()) {
        Some("model") => {
            if let Some(repo) = path.parent().and_then(Path::parent).and_then(Path::file_name) {
                raw.push(repo.to_string_lossy().into_owned());
            }
        }
        Some(stem) => raw.push(stem.to_string()),
        None => {}
    }
    raw.iter()
        .map(|s| strip_artifact_suffixes(s))
        .map(|s| s.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '.').collect::<String>())
        .filter(|s| s.len() >= 3 && !is_generic_name(s))
        .collect()
}

fn storage_bits(h: &serde_json::Value) -> Option<String> {
    let layer = h.get("tensors").and_then(|t| t.as_array()).and_then(|ts| {
        ts.iter().find(|t| {
            let name = t.get("name").and_then(|n| n.as_str()).unwrap_or("");
            name.starts_with("layers.")
                && name.ends_with(".weight")
                && t.get("shape").and_then(|s| s.as_array()).is_some_and(|s| s.len() == 2)
        })
    });
    let raw = layer
        .and_then(|t| t.get("dtype").and_then(|d| d.as_str()))
        .or_else(|| h.get("quant_scheme").and_then(|q| q.as_str()))?
        .to_ascii_lowercase();
    Some(raw.strip_prefix("base_").map(str::to_string).unwrap_or(raw))
}

fn is_generic_name(s: &str) -> bool {
    let base = s.trim_end_matches(|c: char| c.is_ascii_digit());
    let base = base.strip_suffix("of").unwrap_or(base).trim_end_matches(|c: char| c.is_ascii_digit());
    matches!(
        base,
        "model" | "models" | "weight" | "weights" | "checkpoint" | "checkpoints" | "ckpt" | "pytorchmodel"
            | "consolidated" | "adapter" | "adaptermodel"
    )
}

fn strip_artifact_suffixes(name: &str) -> String {
    let mut s = name.to_ascii_lowercase();
    loop {
        let before = s.len();
        for ext in [".base", ".gguf", ".safetensors"] {
            if let Some(t) = s.strip_suffix(ext) {
                s.truncate(t.len());
            }
        }
        let cut = s
            .char_indices()
            .rev()
            .filter(|&(i, c)| i > 0 && matches!(c, '-' | '_' | '.'))
            .map(|(i, _)| i)
            .find(|&i| is_artifact_tag(&s[i + 1..]));
        if let Some(i) = cut {
            s.truncate(i);
        }
        if s.len() == before {
            return s;
        }
    }
}

fn is_artifact_tag(t: &str) -> bool {
    const WORDS: [&str; 12] =
        ["tiled", "f16", "bf16", "f32", "fp16", "fp8", "int4", "int8", "mxfp4", "nvfp4", "4bit", "8bit"];
    if WORDS.contains(&t) {
        return true;
    }
    let t = t.strip_prefix("mtp").unwrap_or(t);
    let t = t.strip_prefix('i').unwrap_or(t);
    let Some(rest) = t.strip_prefix('q') else { return false };
    let mut parts = rest.split('_');
    let bits = parts.next().unwrap_or("");
    !bits.is_empty()
        && bits.bytes().all(|b| b.is_ascii_digit())
        && parts.all(|p| matches!(p, "0" | "1" | "k" | "s" | "m" | "l" | "xs" | "xxs" | "xl"))
}

fn collect_bundles(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if (ft.is_file() || ft.is_symlink()) && p.extension().is_some_and(|x| x == "base") && p.is_file() {
            out.push(p);
        } else if ft.is_dir() && depth > 0 && !e.file_name().to_string_lossy().starts_with('.') {
            collect_bundles(&p, depth - 1, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn bundle(dir: &Path, name: &str, header: serde_json::Value) -> std::path::PathBuf {
        bundle_v(dir, name, header, 1)
    }

    fn bundle_v(dir: &Path, name: &str, header: serde_json::Value, version: u32) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        let json = serde_json::to_vec(&header).unwrap();
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(b"BASE").unwrap();
        f.write_all(&version.to_le_bytes()).unwrap();
        f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&json).unwrap();
        p
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("spec-resolve-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn target(name: &str, quant: &str) -> serde_json::Value {
        serde_json::json!({"arch": "qwen", "quant_scheme": quant, "target_backend": "metal",
            "source": {"filename": name},
            "config": {"hidden_size": 2560, "vocab_size": 151936, "num_hidden_layers": 36}})
    }

    fn drafter(kind: &str, name: &str, quant: &str, created: u64) -> serde_json::Value {
        serde_json::json!({"arch": kind, "quant_scheme": quant, "target_backend": "metal",
            "created": created.to_string(), "source": {"filename": name},
            "config": {"hidden_size": 2560, "vocab_size": 151936, "num_hidden_layers": 5,
                "num_target_layers": 36, "target_layer_ids": [1, 9, 17, 25, 33], "speculator_kind": kind}})
    }

    fn pick(t: &Path, dirs: &[(PathBuf, usize)]) -> Option<String> {
        pick_with(t, dirs, &Pairings::default())
    }

    fn pick_with(t: &Path, dirs: &[(PathBuf, usize)], cat: &Pairings) -> Option<String> {
        let h = read_base_header(t).unwrap();
        find_drafter(t, &h, dirs, cat).map(|(k, p)| format!("{k}:{}", p.file_name().unwrap().to_string_lossy()))
    }

    #[test]
    fn drafter_kind_comes_from_the_header() {
        let dir = scratch("kind");
        let t = bundle(&dir, "t.base", serde_json::json!({"arch": "qwen35"}));
        for (arch, want) in [("dflash", "dflash"), ("dspark", "dspark"), ("eagle3", "eagle3"), ("qwen3", "draft-model")] {
            let p = bundle(&dir, &format!("{arch}.base"), serde_json::json!({"arch": arch}));
            let got = resolve(p.to_str().unwrap(), &t).unwrap().unwrap();
            assert_eq!(got, format!("{want}:{}", p.display()));
        }
        assert_eq!(resolve("dflash:/x.base", &t).unwrap().as_deref(), Some("dflash:/x.base"));
        assert_eq!(resolve("prompt-lookup", &t).unwrap().as_deref(), Some("prompt-lookup"));
        let home = std::env::var("HOME").unwrap();
        assert_eq!(resolve("dflash:~/d/x.base", &t).unwrap().unwrap(), format!("dflash:{home}/d/x.base"));
        assert_eq!(resolve("draft-model:~", &t).unwrap().unwrap(), format!("draft-model:{home}"));
        assert_eq!(resolve("dspark:~user/x.base", &t).unwrap().as_deref(), Some("dspark:~user/x.base"));
        if let Ok(rel) = dir.join("dflash.base").strip_prefix(&home).map(|r| r.to_path_buf()) {
            let got = resolve(&format!("~/{}", rel.display()), &t).unwrap().unwrap();
            assert_eq!(got, format!("dflash:{home}/{}", rel.display()));
        }
        assert_eq!(resolve("auto", &t).unwrap(), None);
        let m = bundle(&dir, "m.base", serde_json::json!({"arch": "qwen35", "speculator": {"config": {"kind": "mtp"}}}));
        assert_eq!(resolve("auto", &m).unwrap().as_deref(), Some("mtp-head"));
        assert_eq!(resolve("mtp-head", &m).unwrap().as_deref(), Some("mtp-head"));
        let old = bundle(&dir, "old-mtp.base", serde_json::json!({"arch": "qwen35", "baserT_version": "0.2.3"}));
        let err = resolve("mtp-head", &old).unwrap_err();
        assert!(err.contains("carries no speculator head") && err.contains("baseRT 0.2.3"), "{err}");
        let other = bundle(&dir, "eagle.base", serde_json::json!({"arch": "qwen35", "speculator": {"config": {"kind": "eagle3"}}}));
        assert!(resolve("mtp-head", &other).unwrap_err().contains("not an MTP head"));
        assert_eq!(resolve("mtp-head", Path::new("/nonexistent/x.base")).unwrap().as_deref(), Some("mtp-head"));
        let junk = dir.join("junk.bin");
        std::fs::write(&junk, b"not a bundle").unwrap();
        assert!(resolve(junk.to_str().unwrap(), &t).is_err());
        assert!(resolve("medusa", &t).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sizing_counts_what_the_strategy_loads() {
        let t = Path::new("/m/Qwen3.8-27B-Q4.base");
        assert_eq!(sizing_path("mtp-head", t), Some((t.to_path_buf(), SpecLoad::EmbeddedHead)));
        assert_eq!(sizing_path("dflash:/d/x.base", t), Some((PathBuf::from("/d/x.base"), SpecLoad::Sidecar)));
        assert_eq!(sizing_path("dspark:/d/y.base", t), Some((PathBuf::from("/d/y.base"), SpecLoad::Sidecar)));
        assert_eq!(sizing_path("eagle3:/d/z.base", t), Some((PathBuf::from("/d/z.base"), SpecLoad::Sidecar)));
        assert_eq!(sizing_path("draft-model:/d/lm.base", t), Some((PathBuf::from("/d/lm.base"), SpecLoad::DraftModel)));
        assert_eq!(sizing_path("draft-model:/m/b.base", t), Some((PathBuf::from("/m/b.base"), SpecLoad::DraftModel)));
        assert_eq!(sizing_path("prompt-lookup", t), None);
    }

    #[test]
    fn the_hub_cache_is_where_basert_pull_installs() {
        let os = |s: &str| Some(std::ffi::OsString::from(s));
        assert_eq!(hub_roots_from(os("/m"), os("/h"), None), vec![PathBuf::from("/m")]);
        let roots = hub_roots_from(None, os("/h"), os("/xdg"));
        if cfg!(target_os = "macos") {
            assert_eq!(roots, vec![PathBuf::from("/h/Library/Caches/baseRT/models"), PathBuf::from("/h/.cache/baseRT/models")]);
        } else {
            assert_eq!(roots, vec![PathBuf::from("/xdg/baseRT/models"), PathBuf::from("/h/.cache/baseRT/models")]);
            assert_eq!(hub_roots_from(None, os("/h"), os("rel")), vec![PathBuf::from("/h/.cache/baseRT/models")]);
        }
        assert!(hub_roots_from(None, None, None).is_empty());
    }

    #[test]
    fn a_glm_nextn_layer_is_an_mtp_head() {
        let dir = scratch("glm-nextn");
        let tensors = |extra: &str| {
            serde_json::json!([{"name": "layers.0.self_attn.q_a_proj.weight"}, {"name": extra}])
        };
        let with = bundle(
            &dir,
            "glm.base",
            serde_json::json!({"arch": "glm_dsa", "config": {"num_hidden_layers": 2},
                               "tensors": tensors("layers.2.eh_proj.weight")}),
        );
        assert_eq!(resolve("mtp-head", &with).unwrap().as_deref(), Some("mtp-head"));
        assert_eq!(resolve_all("auto", &with).unwrap().first().map(String::as_str), Some("mtp-head"));
        let without = bundle(
            &dir,
            "glm-plain.base",
            serde_json::json!({"arch": "glm_dsa", "config": {"num_hidden_layers": 3},
                               "tensors": tensors("layers.2.eh_proj.weight")}),
        );
        assert!(resolve("mtp-head", &without).is_err());
        let other = bundle(
            &dir,
            "other.base",
            serde_json::json!({"arch": "qwen35", "config": {"num_hidden_layers": 2},
                               "tensors": tensors("layers.2.eh_proj.weight")}),
        );
        assert!(resolve("mtp-head", &other).is_err());
    }

    #[test]
    fn auto_prefers_the_head_over_an_installed_drafter() {
        let root = scratch("head");
        let mut h = target("Qwen3-4B", "base_q4");
        h["speculator"] = serde_json::json!({"config": {"kind": "mtp"}});
        let t = bundle(&root, "qwen3-4b/Qwen3-4B-Q4.base", h);
        bundle(&root, "drafters/Qwen3-4B-DFlash.base", drafter("dflash", "Qwen3-4B-DFlash", "base_q4", 1));
        assert_eq!(resolve("auto", &t).unwrap().as_deref(), Some("mtp-head"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn auto_finds_a_drafter_in_the_models_layout() {
        let root = scratch("layout");
        let t = bundle(&root, "qwen3-4b/Qwen3-4B-Q4.base", target("Qwen3-4B", "base_q4"));
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        bundle(&root, "drafters/Qwen3-4B_eagle3.base", drafter("eagle3", "Qwen3-4B_eagle3", "base_q4", 1));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("eagle3:Qwen3-4B_eagle3.base"));
        bundle(&root, "drafters/Qwen3-4B-DFlash-b16.base", drafter("dflash", "Qwen3-4B-DFlash-b16", "base_q4", 1));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dflash:Qwen3-4B-DFlash-b16.base"));
        bundle(&root, "drafters/dspark_qwen3_4b_block7.base", drafter("dspark", "dspark_qwen3_4b_block7", "base_q4", 1));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dspark:dspark_qwen3_4b_block7.base"));
        bundle(&root, "drafters/dspark_qwen3_4b_block7-q8.base", drafter("dspark", "dspark_qwen3_4b_block7", "base_q8", 9));
        bundle(&root, "drafters/dspark_qwen3_4b_block7-new.base", drafter("dspark", "dspark_qwen3_4b_block7", "base_q4", 5));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dspark:dspark_qwen3_4b_block7-new.base"));
        let mut lm = drafter("qwen", "Qwen3-0.6B", "base_q4", 99);
        lm["config"]["num_hidden_layers"] = serde_json::json!(28);
        bundle(&root, "drafters/Qwen3-0.6B.base", lm);
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dspark:dspark_qwen3_4b_block7-new.base"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn auto_keeps_every_fitting_drafter_in_rank_order_for_fallback() {
        let root = scratch("ranked");
        let t = bundle(&root, "qwen3-4b/Qwen3-4B-Q4.base", target("Qwen3-4B", "base_q4"));
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        bundle(&root, "drafters/Qwen3-4B_eagle3.base", drafter("eagle3", "Qwen3-4B_eagle3", "base_q4", 1));
        bundle(&root, "drafters/Qwen3-4B-DFlash-b16.base", drafter("dflash", "Qwen3-4B-DFlash-b16", "base_q4", 1));
        bundle(&root, "drafters/dspark_qwen3_4b_block7.base", drafter("dspark", "dspark_qwen3_4b_block7", "base_q4", 1));
        let mut wide = drafter("dspark", "dspark_qwen3_4b_wide", "base_q4", 9);
        wide["config"]["hidden_size"] = serde_json::json!(4096);
        bundle(&root, "drafters/dspark_qwen3_4b_wide.base", wide);
        let h = read_base_header(&t).unwrap();
        let ranked: Vec<String> = find_drafters(&t, &h, &dirs, &Pairings::default())
            .into_iter()
            .map(|(k, p)| format!("{k}:{}", p.file_name().unwrap().to_string_lossy()))
            .collect();
        assert_eq!(
            ranked,
            ["dspark:dspark_qwen3_4b_block7.base", "dflash:Qwen3-4B-DFlash-b16.base", "eagle3:Qwen3-4B_eagle3.base"]
        );
        assert_eq!(pick(&t, &dirs).as_deref(), Some(ranked[0].as_str()));
        assert_eq!(resolve_all("prompt-lookup", &t).unwrap(), ["prompt-lookup"]);
        assert!(resolve_all("off", &t).unwrap().is_empty());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn created_reads_epoch_and_rfc3339() {
        assert_eq!(created_secs("1790218976"), 1_790_218_976);
        assert_eq!(created_secs("2026-04-24T12:00:00Z"), 1_777_032_000);
        assert_eq!(created_secs("2026-04-24T12:00:00.250Z"), 1_777_032_000);
        assert_eq!(created_secs("2026-04-24T12:00:00+01:30"), 1_777_026_600);
        assert_eq!(created_secs("1970-01-01T00:00:00Z"), 0);
        assert_eq!(created_secs("2024-02-29T00:00:00Z"), 1_709_164_800);
        for junk in ["", "yesterday", "2026-13-01T00:00:00Z", "2026-04-24T12:00:00+0100", "2026-04-24"] {
            assert_eq!(created_secs(junk), 0, "{junk}");
        }
    }

    #[test]
    fn auto_prefers_the_newest_iso_stamped_conversion() {
        let root = scratch("iso");
        let t = bundle(&root, "m/Qwen3-4B-Q4.base", target("Qwen3-4B", "base_q4"));
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        let stamped = |c: &str| {
            let mut h = drafter("dspark", "Qwen3-4B-DSpark", "base_q4", 0);
            h["created"] = serde_json::json!(c);
            h
        };
        bundle(&root, "drafters/a.base", stamped("2026-04-24T12:00:00Z"));
        bundle(&root, "drafters/b.base", stamped("2026-09-01T00:00:00Z"));
        bundle(&root, "drafters/c.base", stamped("1780000000"));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dspark:b.base"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn auto_skips_a_format_version_the_engine_will_not_open() {
        let root = scratch("fmtver");
        let t = bundle(&root, "m/Qwen3-4B-Q4.base", target("Qwen3-4B", "base_q4"));
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        let v2 = bundle_v(&root, "drafters/a.base", drafter("dspark", "Qwen3-4B-DSpark", "base_q4", 2_000_000_000), 2);
        bundle(&root, "drafters/b.base", drafter("dspark", "Qwen3-4B-DSpark", "base_q4", 1_000_000_000));
        assert!(read_base_header(&v2).is_none());
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dspark:b.base"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn relative_targets_search_one_level_up_too() {
        let up = |t: &str| auto_search_dirs(Path::new(t))[2].0.clone();
        assert_eq!(up("models/m.base"), PathBuf::from("./drafters"));
        assert_eq!(up("a/models/m.base"), PathBuf::from("a/drafters"));
        assert_eq!(up("m.base"), PathBuf::from("./../drafters"));
        assert_eq!(up("/x/models/m.base"), PathBuf::from("/x/drafters"));
    }

    #[test]
    fn auto_prefers_a_drafter_named_for_the_target_over_a_same_shape_one() {
        let root = scratch("named");
        let t = bundle(&root, "qwen38/Qwen3.8-4B-Q4.base", target("Qwen3.8-4B", "base_q4"));
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        bundle(&root, "drafters/Qwen3.6-4B-DSpark.base", drafter("dspark", "Qwen3.6-4B-DSpark", "base_q4", 1));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dspark:Qwen3.6-4B-DSpark.base"));
        bundle(&root, "drafters/Qwen3.8-4B-DFlash.base", drafter("dflash", "Qwen3.8-4B-DFlash", "base_q4", 1));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dflash:Qwen3.8-4B-DFlash.base"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_sidecar_ranks_by_its_own_storage_width_not_its_targets_scheme() {
        let root = scratch("storage");
        let t = bundle(&root, "qwen3-4b/Qwen3-4B-Q4.base", target("Qwen3-4B", "base_q4"));
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        let with_layers = |mut h: serde_json::Value, dtype: &str| {
            h["tensors"] =
                serde_json::json!([{"name": "layers.0.mlp.down_proj.weight", "dtype": dtype, "shape": [2560, 9728]}]);
            h
        };
        bundle(
            &root,
            "drafters/Qwen3-4B-DFlash-b16.base",
            with_layers(drafter("dflash", "Qwen3-4B-DFlash-b16", "base_q4", 9), "f16"),
        );
        bundle(
            &root,
            "drafters/Qwen3-4B-DFlash-b16-q4.base",
            with_layers(drafter("dflash", "Qwen3-4B-DFlash-b16", "base_q4", 1), "base_q4"),
        );
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dflash:Qwen3-4B-DFlash-b16-q4.base"));
        assert_eq!(storage_bits(&read_base_header(&t).unwrap()).as_deref(), Some("q4"));
        assert_eq!(storage_bits(&with_layers(drafter("dflash", "x", "base_q4", 1), "f16")).as_deref(), Some("f16"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_generic_provenance_name_does_not_make_a_sibling_look_named() {
        let root = scratch("generic");
        let t = bundle(&root, "qwen38/Qwen3.8-4B-Q4.base", target("model.safetensors", "base_q4"));
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        bundle(
            &root,
            "drafters/Qwen3.6-4B-DSpark.base",
            drafter("dspark", "model-00001-of-00002.safetensors", "base_q4", 9),
        );
        bundle(&root, "drafters/Qwen3.8-4B-DFlash.base", drafter("dflash", "model.safetensors", "base_q4", 1));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dflash:Qwen3.8-4B-DFlash.base"));
        for (name, generic) in [
            ("model", true),
            ("model00001of00004", true),
            ("pytorchmodel", true),
            ("consolidated00", true),
            ("qwen3.8-4b", false),
            ("modelscope7b", false),
        ] {
            assert_eq!(is_generic_name(name), generic, "{name}");
        }
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn artifact_suffixes_do_not_hide_the_model_name() {
        for (raw, want) in [
            ("Qwen3.8-27B-Q8.base", "qwen3.8-27b"),
            ("Qwen3.8-27B-Q8.gguf", "qwen3.8-27b"),
            ("Qwen3.8-27B-mtpq4-Q4", "qwen3.8-27b"),
            ("Qwen3-4B-Q4_K_M.gguf", "qwen3-4b"),
            ("qwen3-4b-iq4_xs", "qwen3-4b"),
            ("Qwen3.8-27B-DFlash2-Q4-tiled", "qwen3.8-27b-dflash2"),
            ("Qwen3-8B-BF16", "qwen3-8b"),
            ("Qwen3-8B-F16", "qwen3-8b"),
            ("Qwen3-4B-DFlash-b16", "qwen3-4b-dflash-b16"),
            ("Qwen3.5-2B-Base", "qwen3.5-2b-base"),
            ("qwen3-30b-a3b", "qwen3-30b-a3b"),
            ("Q4", "q4"),
        ] {
            assert_eq!(strip_artifact_suffixes(raw), want, "{raw}");
        }
    }

    #[test]
    fn a_quant_tagged_target_still_prefers_its_named_drafter() {
        let root = scratch("quantname");
        let mut th = target("Qwen3.8-27B-Q8.gguf", "base_q8");
        th["source"]["filename"] = serde_json::json!("Qwen3.8-27B-Q8.gguf");
        let t = bundle(&root, "qwen38/Qwen3.8-27B-Q8.base", th);
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        bundle(&root, "drafters/Qwen3.6-27B-DSpark.base", drafter("dspark", "Qwen3.6-27B-DSpark", "base_q8", 1));
        bundle(&root, "drafters/Qwen3.8-27B-DFlash.base", drafter("dflash", "Qwen3.8-27B-DFlash", "base_q8", 1));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dflash:Qwen3.8-27B-DFlash.base"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn auto_rejects_drafters_that_do_not_fit() {
        let root = scratch("fit");
        let t = bundle(&root, "m/Qwen3-4B-Q4.base", target("Qwen3-4B", "base_q4"));
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        let mut wide = drafter("dflash", "Qwen3-4B-DFlash-a", "base_q4", 1);
        wide["config"]["hidden_size"] = serde_json::json!(4096);
        bundle(&root, "drafters/a.base", wide);
        let mut vocab = drafter("dflash", "Qwen3-4B-DFlash-b", "base_q4", 1);
        vocab["config"]["vocab_size"] = serde_json::json!(248320);
        bundle(&root, "drafters/b.base", vocab);
        let mut deep = drafter("dflash", "Qwen3-4B-DFlash-c", "base_q4", 1);
        deep["config"]["num_target_layers"] = serde_json::json!(64);
        bundle(&root, "drafters/c.base", deep);
        let mut taps = drafter("eagle3", "Qwen3-4B-e3", "base_q4", 1);
        taps["config"].as_object_mut().unwrap().remove("num_target_layers");
        taps["config"]["aux_layer_ids"] = serde_json::json!([2, 20, 40]);
        bundle(&root, "drafters/d.base", taps);
        let mut cuda = drafter("dspark", "Qwen3-4B-dspark", "base_q4", 1);
        cuda["target_backend"] = serde_json::json!("cuda");
        bundle(&root, "drafters/e.base", cuda);
        assert_eq!(pick(&t, &dirs), None);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn cuda_tags_a_cuda_build_opens_pair_with_each_other() {
        let root = scratch("cudaclass");
        let mut th = target("Qwen3-4B", "base_q4");
        th["target_backend"] = serde_json::json!("cuda");
        let t = bundle(&root, "m/Qwen3-4B-Q4.base", th);
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        bundle(&root, "drafters/a.base", drafter("dspark", "Qwen3-4B-DSpark", "base_q4", 1));
        assert_eq!(pick(&t, &dirs), None);
        let mut sm = drafter("dflash", "Qwen3-4B-DFlash", "base_q4", 1);
        sm["target_backend"] = serde_json::json!("cuda_sm121");
        bundle(&root, "drafters/b.base", sm);
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dflash:b.base"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_missing_backend_is_metal() {
        let root = scratch("backend");
        let mut th = target("Qwen3-4B", "base_q4");
        th.as_object_mut().unwrap().remove("target_backend");
        let t = bundle(&root, "m/Qwen3-4B-Q4.base", th);
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        let mut cuda = drafter("dspark", "Qwen3-4B-DSpark", "base_q4", 1);
        cuda["target_backend"] = serde_json::json!("cuda_sm121");
        bundle(&root, "drafters/a.base", cuda);
        assert_eq!(pick(&t, &dirs), None);
        bundle(&root, "drafters/b.base", drafter("dflash", "Qwen3-4B-DFlash", "base_q4", 1));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dflash:b.base"));
        let mut old = drafter("dspark", "Qwen3-4B-DSpark", "base_q4", 1);
        old.as_object_mut().unwrap().remove("target_backend");
        bundle(&root, "drafters/c.base", old);
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dspark:c.base"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn auto_walks_the_hub_cache_layout() {
        let hub = scratch("hub");
        let t = bundle(&hub, "basecompute/Qwen3-4B/default-q4/model.base", target("Qwen3-4B", "base_q4"));
        bundle(&hub, "basecompute/Qwen3-4B-EAGLE3/default-q4/model.base", drafter("eagle3", "x", "base_q4", 1));
        bundle(&hub, ".src/hf/staged/model.base", drafter("dspark", "Qwen3-4B-DSpark", "base_q4", 1));
        let got = find_drafter(&t, &read_base_header(&t).unwrap(), &[(hub.clone(), HUB_DEPTH)], &Pairings::default())
            .unwrap();
        assert_eq!(got.0, "eagle3");
        assert!(got.1.ends_with("Qwen3-4B-EAGLE3/default-q4/model.base"), "{}", got.1.display());
        std::fs::remove_dir_all(&hub).ok();
    }

    fn catalog() -> Pairings {
        Pairings::from_json(
            &serde_json::json!({"models": [
                {"id": "basecompute/Qwen3-4B", "file": "Qwen3-4B-Q4.base"},
                {"id": "basecompute/Qwen3-4B-EAGLE3", "file": "Qwen3-4B-EAGLE3-Q4.base",
                 "speculator_for": "basecompute/Qwen3-4B", "speculator_rank": 1},
                {"id": "basecompute/Qwen3-4B-DFlash", "file": "Qwen3-4B-DFlash-Q4.base",
                 "speculator_for": "basecompute/Qwen3-4B", "speculator_rank": 2},
                {"id": "basecompute/Qwen3-4B-DSpark", "file": "Qwen3-4B-DSpark-Q4.base",
                 "speculator_for": "basecompute/Qwen3-4B"},
            ]})
            .to_string(),
        )
    }

    #[test]
    fn auto_takes_the_catalog_pairing_in_rank_order() {
        let root = scratch("catalog");
        let t = bundle(&root, "qwen3-4b/Qwen3-4B-Q4.base", target("Qwen3-4B", "base_q4"));
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        let cat = catalog();
        bundle(&root, "drafters/dspark_qwen3_4b_block7.base", drafter("dspark", "dspark_qwen3_4b_block7", "base_q4", 9));
        bundle(&root, "drafters/Qwen3-4B-DSpark-Q4.base", drafter("dspark", "x", "base_q4", 1));
        assert_eq!(pick_with(&t, &dirs, &cat).as_deref(), Some("dspark:Qwen3-4B-DSpark-Q4.base"));
        bundle(&root, "drafters/Qwen3-4B-DFlash-Q4.base", drafter("dflash", "x", "base_q4", 1));
        assert_eq!(pick_with(&t, &dirs, &cat).as_deref(), Some("dflash:Qwen3-4B-DFlash-Q4.base"));
        bundle(&root, "drafters/Qwen3-4B-EAGLE3-Q4.base", drafter("eagle3", "x", "base_q4", 1));
        assert_eq!(pick_with(&t, &dirs, &cat).as_deref(), Some("eagle3:Qwen3-4B-EAGLE3-Q4.base"));
        assert_eq!(pick(&t, &dirs).as_deref(), Some("dspark:dspark_qwen3_4b_block7.base"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_pairing_for_another_target_does_not_count() {
        let root = scratch("other");
        let t = bundle(&root, "m/Qwen3-4B-Instruct-2507-Q4.base", target("Qwen3-4B-Instruct-2507", "base_q4"));
        let dirs = auto_search_dirs(&t)[..3].to_vec();
        bundle(&root, "drafters/Qwen3-4B-EAGLE3-Q4.base", drafter("eagle3", "Qwen3-4B-EAGLE3", "base_q4", 1));
        bundle(&root, "drafters/Qwen3-4B-DSpark-Q4.base", drafter("dspark", "Qwen3-4B-DSpark", "base_q4", 1));
        assert_eq!(pick_with(&t, &dirs, &catalog()).as_deref(), Some("dspark:Qwen3-4B-DSpark-Q4.base"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn hub_bundles_are_identified_by_their_repo() {
        let hub = scratch("hubcat");
        let t = bundle(&hub, "basecompute/Qwen3-4B/default-q4/model.base", target("Qwen3-4B", "base_q4"));
        bundle(&hub, "basecompute/Qwen3-4B-DSpark/default-q4/model.base", drafter("dspark", "a", "base_q4", 1));
        bundle(&hub, "basecompute/Qwen3-4B-EAGLE3/default-q4/model.base", drafter("eagle3", "b", "base_q4", 1));
        let got = find_drafter(&t, &read_base_header(&t).unwrap(), &[(hub.clone(), HUB_DEPTH)], &catalog()).unwrap();
        assert_eq!(got.0, "eagle3");
        std::fs::remove_dir_all(&hub).ok();
    }

    #[test]
    fn the_catalog_is_read_from_the_cache_and_not_fetched_offline() {
        let root = scratch("catjson");
        let own = root.join("cache").join("hub-catalog.json");
        assert_eq!(catalog_json(&own, false), "");
        std::fs::create_dir_all(own.parent().unwrap()).unwrap();
        std::fs::write(&own, r#"{"models":[{"id":"basecompute/Qwen3-4B","file":"Qwen3-4B-Q4.base"}]}"#).unwrap();
        let p = Pairings::from_json(&catalog_json(&own, true));
        assert_eq!(p.id_of(Path::new("/x/Qwen3-4B-Q4.base")).as_deref(), Some("basecompute/Qwen3-4B"));
        std::fs::remove_dir_all(&root).ok();
    }

    fn sha256_of(path: &Path) -> String {
        use sha2::Digest;
        sha2::Sha256::digest(std::fs::read(path).unwrap()).iter().map(|b| format!("{b:02x}")).collect()
    }

    fn pinned(rows: serde_json::Value, digests: &Path) -> Pairings {
        let mut p = Pairings::from_json(&serde_json::json!({ "models": rows }).to_string());
        p.digests = Some(digests.to_path_buf());
        p
    }

    #[test]
    fn a_drafter_that_is_not_the_catalogs_bytes_is_skipped() {
        let hub = scratch("hubsha");
        let t = bundle(&hub, "basecompute/Qwen3-4B/default-q4/model.base", target("Qwen3-4B", "base_q4"));
        let eagle = bundle(&hub, "basecompute/Qwen3-4B-EAGLE3/default-q4/model.base", drafter("eagle3", "b", "base_q4", 1));
        let spark = bundle(&hub, "basecompute/Qwen3-4B-DSpark/default-q4/model.base", drafter("dspark", "a", "base_q4", 1));
        let flat = bundle(&hub, "drafters/Qwen3-4B-DFlash-Q4.base", drafter("dflash", "c", "base_q4", 1));
        let other = "0".repeat(64);
        let rows = |eagle_sha: &str| {
            serde_json::json!([
                {"id": "basecompute/Qwen3-4B", "quant": "default-q4"},
                {"id": "basecompute/Qwen3-4B-EAGLE3", "quant": "default-q4", "sha256": eagle_sha,
                 "speculator_for": "basecompute/Qwen3-4B", "speculator_rank": 1},
                {"id": "basecompute/Qwen3-4B-DFlash", "file": "Qwen3-4B-DFlash-Q4.base", "quant": "default-q4",
                 "size": 1, "sha256": other, "speculator_for": "basecompute/Qwen3-4B", "speculator_rank": 2},
                {"id": "basecompute/Qwen3-4B-DSpark", "quant": "default-q4", "sha256": sha256_of(&spark),
                 "speculator_for": "basecompute/Qwen3-4B", "speculator_rank": 3},
            ])
        };
        let digests = hub.join(".digests");
        let dirs = [(hub.clone(), HUB_DEPTH), (hub.join("drafters"), 0)];
        let h = read_base_header(&t).unwrap();
        let ranked = |cat: &Pairings| -> Vec<PathBuf> { find_drafters(&t, &h, &dirs, cat).into_iter().map(|(_, p)| p).collect() };

        assert_eq!(ranked(&pinned(rows(&sha256_of(&eagle)), &digests)), vec![eagle.clone(), spark.clone()]);
        assert_eq!(ranked(&pinned(rows(&other), &digests)), vec![spark.clone()], "a mismatched drafter is skipped");
        let why = pinned(rows(&other), &digests).digest_refusal(&flat).unwrap();
        assert!(why.contains("Qwen3-4B-DFlash default-q4") && why.contains("pull it again"), "{why}");
        std::fs::remove_dir_all(&hub).ok();
    }

    #[test]
    fn a_verified_digest_is_remembered_until_the_file_changes() {
        let hub = scratch("shacache");
        let d = bundle(&hub, "basecompute/Qwen3-4B-EAGLE3/default-q4/model.base", drafter("eagle3", "b", "base_q4", 1));
        let good = sha256_of(&d);
        let digests = hub.join(".digests");
        let cat = pinned(serde_json::json!([{"id": "basecompute/Qwen3-4B-EAGLE3", "quant": "default-q4", "sha256": good}]), &digests);
        assert_eq!(cat.digest_refusal(&d), None);
        let entries: Vec<PathBuf> = std::fs::read_dir(&digests).unwrap().map(|e| e.unwrap().path()).collect();
        assert_eq!(entries.len(), 1, "{entries:?}");
        let remembered = std::fs::read_to_string(&entries[0]).unwrap();
        assert!(remembered.ends_with(&format!("{good}\n")), "{remembered}");

        std::fs::write(&entries[0], remembered.replace(&good, &"f".repeat(64))).unwrap();
        assert!(cat.digest_refusal(&d).is_some(), "an unchanged file is not hashed again");
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&d, std::fs::read(&d).unwrap()).unwrap();
        assert_eq!(cat.digest_refusal(&d), None, "a rewritten file is hashed afresh");
        std::fs::remove_dir_all(&hub).ok();
    }

    #[test]
    fn an_explicit_drafter_under_a_catalog_name_must_be_the_catalogs_bytes() {
        let dir = scratch("shaexplicit");
        let t = bundle(&dir, "t.base", target("Qwen3-1.7B", "base_q4"));
        let d = bundle(&dir, "Qwen3-1.7B-EAGLE3-F16.base", drafter("eagle3", "x", "base_f16", 1));
        let rows = serde_json::json!([{"id": "basecompute/Qwen3-1.7B-EAGLE3", "file": "Qwen3-1.7B-EAGLE3-F16.base", "size": 1, "sha256": "ff"}]);
        let cat = pinned(rows, &dir.join("sha256"));
        let err = cat.digest_refusal(&d).unwrap();
        assert!(err.contains("basecompute/Qwen3-1.7B-EAGLE3") && err.contains("bytes, not"), "{err}");
        let own = bundle(&dir, "my-eagle.base", drafter("eagle3", "x", "base_f16", 1));
        assert_eq!(cat.digest_refusal(&own), None, "a name the catalog does not list is not checked");
        assert!(resolve(own.to_str().unwrap(), &t).unwrap().is_some());
        std::fs::remove_dir_all(&dir).ok();
    }
}
