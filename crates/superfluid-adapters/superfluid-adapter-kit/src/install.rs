//! Installing a runtime from where its own project publishes it.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde_json::{json, Value};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Host {
    pub os: String,
    pub arch: String,
    pub nvidia_driver: Option<String>,
    pub cuda_runtime: Option<u32>,
    pub rocm: bool,
    /// The AMD GPUs by LLVM target (`gfx1151`), as HIP numbers them: the first is device 0 ([`hip_view`]). When
    /// `amd_order_unknown`, the driver's GPUs in node order instead.
    pub amd_gpus: Vec<String>,
    /// For each of `amd_gpus`, the index HIP_VISIBLE_DEVICES names it by (its place among the GPUs ROCR_VISIBLE_DEVICES
    /// leaves). Empty: each GPU's own place in `amd_gpus`.
    pub amd_hip_indices: Vec<usize>,
    /// A visibility variable names GPUs by UUID, which is not resolved here: which GPU is device 0 is unknown.
    pub amd_order_unknown: bool,
    /// The ROCm releases installed side by side as `/opt/rocm/core-<version>`, as AMD's ROCm 10 packages lay them out (`10.0`).
    pub rocm_runtimes: Vec<String>,
    pub vulkan: bool,
}

impl Host {
    pub fn detect() -> Host {
        let os = std::env::consts::OS;
        let arch = match std::env::consts::ARCH {
            "aarch64" => "arm64",
            "x86_64" => "x64",
            other => other,
        };
        let mut host = Host { os: os.into(), arch: arch.into(), ..Host::default() };
        if host.os == "linux" {
            host.nvidia_driver = std::fs::read_to_string("/proc/driver/nvidia/version").ok().and_then(|t| nvidia_driver_version(&t));
            host.cuda_runtime = cuda_runtime_major();
            host.rocm = Path::new("/dev/kfd").exists() && Path::new("/opt/rocm").is_dir();
            let view = amd_view();
            host.amd_gpus = view.gpus;
            host.amd_hip_indices = view.hip_indices;
            host.amd_order_unknown = !view.known;
            host.rocm_runtimes = rocm_runtimes(Path::new("/opt/rocm"));
            host.vulkan = linux_vulkan();
        }
        host
    }

    pub fn platform(&self) -> String {
        format!("{}-{}", self.os, self.arch)
    }

    pub fn nvidia_major(&self) -> Option<u32> {
        self.nvidia_driver.as_deref().and_then(|v| v.split('.').next()).and_then(|m| m.parse().ok())
    }

    pub fn describe(&self) -> String {
        let os = match self.os.as_str() {
            "macos" => "macOS",
            "linux" => "Linux",
            other => other,
        };
        let mut facts = vec![format!("{os} {}", self.arch)];
        if let Some(v) = &self.nvidia_driver {
            facts.push(format!("NVIDIA driver {v}"));
        }
        if let Some(m) = self.cuda_runtime {
            facts.push(format!("CUDA runtime {m}"));
        }
        if !self.amd_gpus.is_empty() {
            facts.push(format!("AMD {}", self.amd_gpus.join(" + ")));
        }
        if !self.rocm_runtimes.is_empty() {
            facts.push(format!("ROCm {}", self.rocm_runtimes.join(" + ")));
        } else if self.rocm {
            facts.push("ROCm".into());
        }
        if self.vulkan {
            facts.push("Vulkan".into());
        }
        facts.join(", ")
    }

    pub fn to_json(&self) -> Value {
        json!({
            "os": self.os, "arch": self.arch, "nvidia_driver": self.nvidia_driver,
            "cuda_runtime": self.cuda_runtime, "rocm": self.rocm, "amd_gpus": self.amd_gpus,
            "amd_order_unknown": self.amd_order_unknown, "rocm_runtimes": self.rocm_runtimes, "vulkan": self.vulkan,
        })
    }
}

/// The AMD GPUs HIP numbers as devices 0, 1, ..., by LLVM target ([`hip_view`] of the driver's GPUs and this process's
/// visibility variables). baseRT's ROCm engine runs on device 0. None without the driver (`/dev/kfd`).
pub fn amd_gpus() -> Vec<String> {
    amd_view().gpus
}

/// The driver's GPUs as HIP will number them in this process: [`hip_view`] under ROCR_VISIBLE_DEVICES and HIP's own
/// variable, HIP_VISIBLE_DEVICES, or CUDA_VISIBLE_DEVICES when that is unset or empty.
pub fn amd_view() -> HipView {
    let rocr = std::env::var("ROCR_VISIBLE_DEVICES").ok();
    let set = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    let hip = set("HIP_VISIBLE_DEVICES").or_else(|| set("CUDA_VISIBLE_DEVICES"));
    hip_view(&kfd_gpus(), rocr.as_deref(), hip.as_deref())
}

/// AMD GPUs as HIP numbers them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HipView {
    /// Device 0 first; with `known` false, the driver's GPUs in node order.
    pub gpus: Vec<String>,
    /// For each of `gpus`, the index HIP_VISIBLE_DEVICES names it by: its place among the GPUs ROCR_VISIBLE_DEVICES leaves.
    pub hip_indices: Vec<usize>,
    /// False when a visibility variable reaches an entry naming a GPU by UUID, which this does not resolve.
    pub known: bool,
}

/// The GPUs in `kfd` (the driver's, in node order) as HIP numbers them, under `rocr` (ROCR_VISIBLE_DEVICES; `None` when
/// unset) and then `hip` (HIP's variable; `None` when unset or empty), each read as the ROCm runtime reads it:
///
/// - ROCr (`amd_filter_device.cpp`): set but empty shows no GPU. Entries are trimmed and read up to the number of GPUs;
///   each is an index (C `strtol`, base 0, so `0x1` is 1) into the driver's GPUs. An entry that does not parse, an index
///   out of range, or a repeat ends the list.
/// - HIP (`rocdevice.cpp`): entries index the GPUs ROCr left; only the plain decimal form counts (`01`, `-1` and ` 1`
///   do not). An invalid entry ends the list; a repeat is skipped.
///
/// Either may name a GPU by UUID (`GPU-...`): that needs the GPUs' unique IDs and each runtime's matching rule, so a UUID
/// entry the runtime would reach makes the order unknown (`known` false) rather than guessed.
pub fn hip_view(kfd: &[String], rocr: Option<&str>, hip: Option<&str>) -> HipView {
    let unknown = || HipView { gpus: kfd.to_vec(), hip_indices: Vec::new(), known: false };
    let after_rocr = match rocr {
        None => (0..kfd.len()).collect(),
        Some(list) => match rocr_selection(kfd.len(), list) {
            Some(picked) => picked,
            None => return unknown(),
        },
    };
    let order = match hip {
        None => (0..after_rocr.len()).collect(),
        Some(list) => match hip_selection(after_rocr.len(), list) {
            Some(picked) => picked,
            None => return unknown(),
        },
    };
    HipView { gpus: order.iter().map(|&j| kfd[after_rocr[j]].clone()).collect(), hip_indices: order, known: true }
}

/// A comma list's entries as C's `getline` and a `do { } while (pos < size)` loop see them: a final empty entry after a
/// trailing comma is not one.
fn entries(list: &str) -> Vec<&str> {
    let mut all: Vec<&str> = list.split(',').collect();
    if all.len() > 1 && all.last() == Some(&"") {
        all.pop();
    }
    all
}

/// The driver indices ROCR_VISIBLE_DEVICES selects, in its order; `None` when it reaches a UUID or an empty entry.
fn rocr_selection(gpus: usize, list: &str) -> Option<Vec<usize>> {
    if list.is_empty() {
        return Some(Vec::new());
    }
    let mut picked = Vec::new();
    for entry in entries(list).into_iter().take(gpus) {
        let entry = entry.trim();
        if entry.is_empty() || entry.starts_with(['G', 'g']) {
            return None;
        }
        match strtol0(entry) {
            Some(i) if i >= 0 && (i as u128) < gpus as u128 && !picked.contains(&(i as usize)) => picked.push(i as usize),
            _ => break,
        }
    }
    Some(picked)
}

/// The indices HIP_VISIBLE_DEVICES selects among `gpus`, in its order; `None` when it reaches a UUID.
fn hip_selection(gpus: usize, list: &str) -> Option<Vec<usize>> {
    let mut picked = Vec::new();
    for entry in entries(list) {
        if entry.contains("GPU-") {
            return None;
        }
        match entry.parse::<usize>() {
            Ok(i) if i < gpus && entry == i.to_string() => {
                if !picked.contains(&i) {
                    picked.push(i);
                }
            }
            _ => break,
        }
    }
    Some(picked)
}

/// C's `strtol(s, &end, 0)` when it consumes all of `s`: an optional sign, then hex after `0x`, octal after `0`, else
/// decimal. `None` when anything is left over or there are no digits.
fn strtol0(s: &str) -> Option<i128> {
    let (negative, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let (radix, body) = if let Some(hex) = digits.strip_prefix("0x").or_else(|| digits.strip_prefix("0X")) {
        (16, hex)
    } else if digits.len() > 1 && digits.starts_with('0') {
        (8, &digits[1..])
    } else {
        (10, digits)
    };
    if body.is_empty() || !body.chars().all(|c| c.is_digit(radix)) {
        return None;
    }
    let value = body.chars().try_fold(0i128, |v, c| v.checked_mul(radix as i128)?.checked_add(c.to_digit(radix)? as i128)).unwrap_or(i128::MAX);
    Some(if negative { -value } else { value })
}

/// The AMD GPUs the KFD driver lists, by LLVM target, in node order.
fn kfd_gpus() -> Vec<String> {
    if !Path::new("/dev/kfd").exists() {
        return Vec::new();
    }
    let Ok(dir) = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes") else {
        return Vec::new();
    };
    let mut nodes: Vec<(u32, PathBuf)> = dir.flatten().filter_map(|n| Some((n.file_name().to_str()?.parse().ok()?, n.path()))).collect();
    nodes.sort();
    nodes.into_iter().filter_map(|(_, node)| gfx_target(&std::fs::read_to_string(node.join("properties")).ok()?)).collect()
}

/// The LLVM target a KFD topology node's `properties` state, as `gfx_target_version` major*10000 + minor*100 + stepping:
/// 110501 is gfx1151, 90010 gfx90a. A CPU node states 0.
fn gfx_target(properties: &str) -> Option<String> {
    let v: u32 = properties.lines().find_map(|l| l.strip_prefix("gfx_target_version "))?.trim().parse().ok()?;
    (v != 0).then(|| format!("gfx{}{:x}{:x}", v / 10000, v / 100 % 100, v % 100))
}

/// The versions of the ROCm releases under `root` as `core-<version>/lib`, oldest first. The links beside them (`core`,
/// `core-10`, through /etc/alternatives) name one of them again and are not counted.
fn rocm_runtimes(root: &Path) -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()) && e.path().join("lib").is_dir())
        .filter_map(|e| e.file_name().to_str()?.strip_prefix("core-").map(str::to_string))
        .collect();
    found.sort_by_key(|v| version_key(v));
    found
}

fn nvidia_driver_version(text: &str) -> Option<String> {
    text.lines().next()?.split_whitespace().find(|w| {
        let parts: Vec<&str> = w.split('.').collect();
        parts.len() >= 2 && parts.iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
    })
    .map(str::to_string)
}

fn linux_libraries() -> Vec<String> {
    let mut names: Vec<String> = Command::new("/sbin/ldconfig")
        .arg("-p")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().filter_map(|l| l.split_whitespace().next().map(str::to_string)).collect())
        .unwrap_or_default();
    for dir in ["/usr/local/cuda/lib64", "/usr/local/cuda/targets/sbsa-linux/lib", "/usr/local/cuda/targets/x86_64-linux/lib"] {
        if let Ok(rd) = std::fs::read_dir(dir) {
            names.extend(rd.flatten().filter_map(|e| e.file_name().to_str().map(str::to_string)));
        }
    }
    names
}

fn cuda_runtime_major() -> Option<u32> {
    let libs = linux_libraries();
    let major = |stem: &str| -> Vec<u32> {
        libs.iter()
            .filter_map(|n| n.strip_prefix(stem))
            .filter_map(|rest| rest.split('.').next().and_then(|m| m.parse().ok()))
            .collect()
    };
    let cudart = major("libcudart.so.");
    let cublas = major("libcublas.so.");
    cudart.into_iter().filter(|m| cublas.contains(m)).max()
}

fn linux_vulkan() -> bool {
    let loader = linux_libraries().iter().any(|n| n.starts_with("libvulkan.so.1"));
    let icd = ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d"]
        .iter()
        .any(|d| std::fs::read_dir(d).map(|rd| rd.flatten().any(|e| e.path().extension().is_some_and(|x| x == "json"))).unwrap_or(false));
    loader && icd
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Request {
    pub version: Option<String>,
    pub backend: Option<String>,
    pub from: Option<String>,
    pub sha256: Option<String>,
    pub untested: bool,
}

impl Request {
    pub fn parse(args: &[String]) -> Result<Request, String> {
        let mut r = Request::default();
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            let mut value = || it.next().cloned().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--version" => r.version = Some(value()?),
                "--backend" => r.backend = Some(value()?.to_ascii_lowercase()),
                "--from" => r.from = Some(value()?),
                "--sha256" => r.sha256 = Some(value()?.to_ascii_lowercase()),
                "--untested" => r.untested = true,
                other => return Err(format!("unknown flag {other}")),
            }
        }
        Ok(r)
    }

    pub fn to_args(&self) -> Vec<String> {
        let mut a = Vec::new();
        for (flag, v) in [("--version", &self.version), ("--backend", &self.backend), ("--from", &self.from), ("--sha256", &self.sha256)] {
            if let Some(v) = v {
                a.push(flag.to_string());
                a.push(v.clone());
            }
        }
        if self.untested {
            a.push("--untested".into());
        }
        a
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub sha256: Option<String>,
    pub size: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub id: String,
    pub version: String,
    pub backend: String,
    pub install: String,
    pub assets: Vec<Asset>,
    pub tested: bool,
    pub why: String,
    /// Set when the plan was made without asking what the newest version is: the adapter asks the network for that
    /// (basert, its release listing) and the ask failed, so this is the newest the adapter knows on its own, and the
    /// text says what failed. `superfluid runtime update` then does not take it for the newest there is.
    pub offline: Option<String>,
    pub extra: Value,
}

impl Plan {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id, "version": self.version, "backend": self.backend, "install": self.install,
            "assets": self.assets.iter().map(|a| json!({"name": a.name, "url": a.url, "sha256": a.sha256, "size": a.size})).collect::<Vec<_>>(),
            "tested": self.tested, "why": self.why, "offline": self.offline, "extra": self.extra,
        })
    }

    pub fn from_json(v: &Value) -> Result<Plan, String> {
        let s = |k: &str| v[k].as_str().map(str::to_string).ok_or_else(|| format!("a plan without \"{k}\""));
        let assets = v["assets"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|x| {
                        Ok(Asset {
                            name: x["name"].as_str().ok_or("an asset without a name")?.to_string(),
                            url: x["url"].as_str().ok_or("an asset without a url")?.to_string(),
                            sha256: x["sha256"].as_str().map(str::to_string),
                            size: x["size"].as_u64(),
                        })
                    })
                    .collect::<Result<Vec<_>, &str>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Plan {
            id: s("id")?,
            version: s("version")?,
            backend: s("backend")?,
            install: s("install")?,
            assets,
            tested: v["tested"].as_bool().unwrap_or(false),
            why: v["why"].as_str().unwrap_or_default().to_string(),
            offline: v["offline"].as_str().map(str::to_string),
            extra: v["extra"].clone(),
        })
    }

    pub fn size(&self) -> u64 {
        self.assets.iter().filter_map(|a| a.size).sum()
    }
}

/// A version an adapter installs on its own, planned for a host: the first plan, or why it does not install there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub version: String,
    pub here: Result<Plan, String>,
}

impl Candidate {
    pub fn to_json(&self) -> Value {
        match &self.here {
            Ok(p) => json!({"version": self.version, "plan": p.to_json()}),
            Err(why) => json!({"version": self.version, "error": why}),
        }
    }

    pub fn from_json(v: &Value) -> Result<Candidate, String> {
        let version = v["version"].as_str().ok_or("a listed version without \"version\"")?.to_string();
        let here = match v["error"].as_str() {
            Some(why) => Err(why.to_string()),
            None => Ok(Plan::from_json(&v["plan"])?),
        };
        Ok(Candidate { version, here })
    }

    /// What the install fetches, wheels included.
    pub fn size(&self) -> u64 {
        self.here.as_ref().map(|p| p.size() + p.extra["wheels_size"].as_u64().unwrap_or(0)).unwrap_or(0)
    }
}

/// An adapter's rule for the versions `versions` does not list that install without `--untested` all the same.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unlisted {
    /// Which ones, as a clause of `superfluid runtime versions`'s footer.
    pub describe: String,
    /// What the rule asks of a version (basert: the engine ABI it drives), kept in the manifest of each install of an unlisted
    /// version made under it: another build's rule installs that version again only when it asks the same.
    pub requires: String,
}

impl Unlisted {
    pub fn to_json(&self) -> Value {
        json!({"describe": self.describe, "requires": self.requires})
    }

    pub fn from_json(v: &Value) -> Option<Unlisted> {
        Some(Unlisted { describe: v["describe"].as_str()?.to_string(), requires: v["requires"].as_str()?.to_string() })
    }
}

pub fn backend_rank(backend: &str) -> u32 {
    let family = backend.split(['-', '_']).next().unwrap_or(backend);
    match family {
        "metal" => 0,
        "cuda" => 1,
        "rocm" => 2,
        "vulkan" => 3,
        "sycl" => 4,
        "openvino" => 5,
        "cpu" => 9,
        _ => 8,
    }
}

pub fn version_key(v: &str) -> Vec<u64> {
    v.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(|s| s.parse().unwrap_or(0)).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub id: String,
    pub install: String,
    pub version: String,
    pub backend: String,
    pub link_w: Vec<u16>,
    pub platforms: Vec<String>,
    pub worker: String,
    pub sources: Vec<(String, String)>,
    pub tested: bool,
    /// For a version the adapter that installed it did not list, what its rule for those asked of it ([`Unlisted::requires`]);
    /// none for a listed version, one installed with `--untested`, and an install made before manifests kept it.
    pub requires: Option<String>,
}

impl Manifest {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id, "install": self.install, "version": self.version, "backend": self.backend,
            "link_w": self.link_w, "platforms": self.platforms, "worker": self.worker,
            "sources": self.sources.iter().map(|(u, s)| json!({"url": u, "sha256": s})).collect::<Vec<_>>(),
            "tested": self.tested, "requires": self.requires,
        })
    }

    pub fn from_json(v: &Value) -> Result<Manifest, String> {
        let s = |k: &str| v[k].as_str().map(str::to_string).ok_or_else(|| format!("runtime.json: no \"{k}\""));
        let strings = |x: &Value| -> Vec<String> {
            x.as_array().map(|a| a.iter().filter_map(|e| e.as_str().map(str::to_string)).collect()).unwrap_or_default()
        };
        let version = s("version")?;
        Ok(Manifest {
            id: s("id")?,
            install: v["install"].as_str().map(str::to_string).unwrap_or_else(|| version.clone()),
            version,
            backend: v["backend"].as_str().unwrap_or("unknown").to_string(),
            link_w: v["link_w"].as_array().map(|a| a.iter().filter_map(|e| e.as_u64()).map(|n| n as u16).collect()).unwrap_or_default(),
            platforms: strings(&v["platforms"]),
            worker: s("worker")?,
            sources: v["sources"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| Some((x["url"].as_str()?.to_string(), x["sha256"].as_str().unwrap_or_default().to_string())))
                        .collect()
                })
                .unwrap_or_default(),
            tested: v["tested"].as_bool().unwrap_or(true),
            requires: v["requires"].as_str().map(str::to_string),
        })
    }

    pub fn read(install: &Path) -> Result<Manifest, String> {
        let path = install.join("runtime.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Manifest::from_json(&v)
    }

    pub fn write(&self, install: &Path) -> Result<(), String> {
        let path = install.join("runtime.json");
        std::fs::write(&path, format!("{:#}\n", self.to_json())).map_err(|e| format!("{}: {e}", path.display()))
    }
}

pub fn sha256_of(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn local_path(src: &str) -> Option<&str> {
    src.strip_prefix("file://").or_else(|| (!src.contains("://")).then_some(src))
}

/// What is wrong with fetching `asset`, before anything is fetched: a download is run, so one
/// from anywhere but this machine needs a SHA-256 named in advance.
pub fn unchecked(asset: &Asset) -> Option<String> {
    (asset.sha256.is_none() && local_path(&asset.url).is_none()).then(|| {
        format!(
            "{}: no SHA-256 to check it against, and only a file on this machine is installed unchecked \
             (name the file's SHA-256 with --sha256)",
            asset.url
        )
    })
}

pub fn fetch(asset: &Asset, dir: &Path) -> Result<(PathBuf, String), String> {
    if let Some(why) = unchecked(asset) {
        return Err(why);
    }
    let dest = dir.join(&asset.name);
    let src = asset.url.as_str();
    if let Some(path) = local_path(src) {
        std::fs::copy(path, &dest).map_err(|e| format!("{path}: {e}"))?;
    } else if src.starts_with("https://") {
        let part = dir.join(format!("{}.part", asset.name));
        let out = Command::new("curl")
            .args(["-fSL", "--proto", "=https", "--tlsv1.2", "--retry", "3", "--connect-timeout", "30"])
            .args(["--speed-limit", "1024", "--speed-time", "60", "--progress-bar", "-o"])
            .arg(&part)
            .arg(src)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit())
            .status()
            .map_err(|e| format!("curl: {e} (installs fetch with the system curl)"))?;
        if !out.success() {
            let _ = std::fs::remove_file(&part);
            return Err(format!("fetching {src} failed ({out})"));
        }
        std::fs::rename(&part, &dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    } else {
        return Err(format!("{src}: only https:// URLs and local files are fetched"));
    }
    if let Some(want) = asset.size {
        let have = std::fs::metadata(&dest).map_err(|e| e.to_string())?.len();
        if have != want {
            return Err(format!("{src}: {have} bytes, expected {want}"));
        }
    }
    let sum = sha256_of(&dest)?;
    match &asset.sha256 {
        Some(want) if !want.eq_ignore_ascii_case(&sum) => {
            Err(format!("{src}: SHA-256 {sum}, expected {want} (not the file its project published)"))
        }
        Some(_) => Ok((dest, sum)),
        None => {
            eprintln!("warning: {src} has no SHA-256 to check against: installing a local file unchecked (its SHA-256 is {sum})");
            Ok((dest, sum))
        }
    }
}

pub fn fetch_json(url: &str) -> Result<Value, String> {
    fetch_json_within(url, Duration::from_secs(20))
}

/// A lookup a caller falls back from must end: an answer that stalls fails once `deadline` has passed, retries included.
pub fn fetch_json_within(url: &str, deadline: Duration) -> Result<Value, String> {
    if !url.starts_with("https://") {
        return Err(format!("{url}: only https:// URLs are fetched"));
    }
    let secs = deadline.as_secs().max(1).to_string();
    let mut curl = Command::new("curl");
    curl.args(["-fsSL", "--proto", "=https", "--tlsv1.2", "--retry", "2", "-H", "Accept: application/json"]);
    curl.args(["--connect-timeout", &secs, "--max-time", &secs, "--retry-max-time", &secs]);
    if let Some(token) = token_for(url).and_then(std::env::var_os).filter(|t| !t.is_empty()) {
        curl.arg("-H").arg(format!("Authorization: Bearer {}", token.to_string_lossy()));
    }
    let out = curl.arg(url).output().map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        return Err(format!("fetching {url} failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("{url}: {e}"))
}

/// The environment variable whose token a fetch of `url` sends, if any. GitHub's API allows
/// sixty unauthenticated requests an hour per address, which hosted runners share; a token
/// lifts that, for the API alone. A gated or private Hugging Face repository is listed with
/// `HF_TOKEN`, as `llama download` and `huggingface_hub` read it.
fn token_for(url: &str) -> Option<&'static str> {
    if url.starts_with("https://api.github.com/") {
        return Some("GITHUB_TOKEN");
    }
    let hub = format!("{}/", hf_endpoint().trim_end_matches('/'));
    (url.starts_with("https://huggingface.co/") || url.starts_with(&hub)).then_some("HF_TOKEN")
}

/// The Hugging Face Hub, or the mirror `HF_ENDPOINT` names, as `llama download` and
/// `huggingface_hub` read it.
pub fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT").ok().filter(|e| !e.trim().is_empty()).unwrap_or_else(|| "https://huggingface.co".into())
}

/// The files of the Hub repository `id` (paths within it), from its listing.
pub fn hf_repo_files(id: &str) -> Result<Vec<String>, String> {
    Ok(hf_repo_listing(id, None)?.files.into_iter().map(|f| f.path).collect())
}

/// A file of a Hub repository, as its listing gives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfFile {
    /// Its path within the repository.
    pub path: String,
    pub size: u64,
    /// The name its bytes take in the Hub cache's `blobs`: the LFS SHA-256 of a large file,
    /// the git blob id of a small one, as huggingface_hub and llama.cpp both name them.
    pub blob: String,
}

/// A Hub repository's listing at one revision, with each file's size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfListing {
    pub id: String,
    /// The commit the revision names.
    pub sha: String,
    pub files: Vec<HfFile>,
}

impl HfListing {
    /// Whether the Hub cache already holds `file`'s bytes.
    pub fn cached(&self, file: &HfFile) -> bool {
        hf_repo_cache(&self.id).join("blobs").join(&file.blob).is_file()
    }

    /// The bytes a pull of the files `keep` admits fetches: those not in the cache yet.
    pub fn to_fetch(&self, keep: impl Fn(&str) -> bool) -> u64 {
        self.files.iter().filter(|f| keep(&f.path) && !self.cached(f)).map(|f| f.size).sum()
    }
}

/// The listing of the Hub repository `id` at `revision` (`main` when none), with sizes.
pub fn hf_repo_listing(id: &str, revision: Option<&str>) -> Result<HfListing, String> {
    let endpoint = hf_endpoint();
    let url = match revision {
        Some(rev) => format!("{}/api/models/{id}/revision/{}?blobs=true", endpoint.trim_end_matches('/'), rev.replace('/', "%2F")),
        None => format!("{}/api/models/{id}?blobs=true", endpoint.trim_end_matches('/')),
    };
    let v = fetch_json(&url)?;
    let listing = hf_listing_from(id, &v);
    if listing.files.is_empty() {
        return Err(format!("{url} lists no files for {id}"));
    }
    Ok(listing)
}

/// [`hf_repo_listing`]'s reading of what the Hub answered.
pub fn hf_listing_from(id: &str, v: &Value) -> HfListing {
    let files = v["siblings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| {
            let path = s["rfilename"].as_str()?.to_string();
            let size = s["lfs"]["size"].as_u64().or_else(|| s["size"].as_u64()).unwrap_or(0);
            let blob = s["lfs"]["sha256"].as_str().or_else(|| s["blobId"].as_str()).unwrap_or_default().to_string();
            Some(HfFile { path, size, blob })
        })
        .collect();
    HfListing { id: id.to_string(), sha: v["sha"].as_str().unwrap_or_default().to_string(), files }
}

/// The Hub cache, as huggingface_hub and llama.cpp find it: `HF_HUB_CACHE` (or its older name
/// `HUGGINGFACE_HUB_CACHE`), else `hub` under
/// `HF_HOME`, else `huggingface/hub` under `XDG_CACHE_HOME`, else `~/.cache/huggingface/hub`.
pub fn hf_hub_cache() -> PathBuf {
    let set = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    set("HF_HUB_CACHE")
        .or_else(|| set("HUGGINGFACE_HUB_CACHE"))
        .or_else(|| set("HF_HOME").map(|h| h.join("hub")))
        .or_else(|| set("XDG_CACHE_HOME").map(|c| c.join("huggingface").join("hub")))
        .unwrap_or_else(|| home_dir().join(".cache").join("huggingface").join("hub"))
}

/// Where the Hub cache keeps the repository `id`: `models--org--name`, and where a pull of it
/// lands its bytes. huggingface_hub (from 2.0) then moves a large file into the store it shares
/// between repositories, `blobs/<xx>/<hash>` under the cache itself, and links to it from
/// here; a watch follows the file there by its inode, and nothing another pull lands in that
/// store is counted as this one's.
pub fn hf_repo_cache(id: &str) -> PathBuf {
    hf_hub_cache().join(format!("models--{}", id.replace('/', "--")))
}

/// Whether the Hub is not to be reached, as huggingface_hub reads it: `HF_HUB_OFFLINE` (or
/// the older `TRANSFORMERS_OFFLINE`) set to `1`, `ON`, `YES` or `TRUE`, in any case. A pull
/// under it looks nothing up on the Hub, as `snapshot_download` fetches nothing.
pub fn hf_offline() -> bool {
    ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"].iter().any(|name| {
        std::env::var(name).is_ok_and(|v| ["1", "ON", "YES", "TRUE"].contains(&v.trim().to_ascii_uppercase().as_str()))
    })
}

/// Where this platform keeps caches: `~/Library/Caches` on macOS, else `XDG_CACHE_HOME` or
/// `~/.cache`, as llama.cpp and basert place theirs.
pub fn platform_cache_dir() -> PathBuf {
    if cfg!(target_os = "macos") {
        return home_dir().join("Library").join("Caches");
    }
    std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()).map(PathBuf::from).unwrap_or_else(|| home_dir().join(".cache"))
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME").filter(|v| !v.is_empty()).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

/// Whether `path` matches the shell pattern `pattern`, where `*` stands for any run of
/// characters (directories included, as huggingface_hub's `allow_patterns` reads it).
pub fn glob_matches(pattern: &str, path: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else {
        return path.is_empty();
    };
    if !pattern.contains('*') {
        return path == first;
    }
    let Some(mut rest) = path.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    for (i, part) in parts.iter().enumerate() {
        let last = i + 1 == parts.len();
        if last {
            return rest.ends_with(part);
        }
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    true
}

pub fn unpack(archive: &Path, into: &Path) -> Result<(), String> {
    std::fs::create_dir_all(into).map_err(|e| format!("{}: {e}", into.display()))?;
    let name = archive.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let zip = name.ends_with(".zip");
    let listed = if zip {
        Command::new("unzip").arg("-Z1").arg(archive).output()
    } else {
        Command::new("tar").arg("-tf").arg(archive).output()
    }
    .map_err(|e| format!("listing {name}: {e}"))?;
    if !listed.status.success() {
        return Err(format!("listing {name}: {}", String::from_utf8_lossy(&listed.stderr).trim()));
    }
    if let Some(bad) = String::from_utf8_lossy(&listed.stdout).lines().find(|m| escapes(Path::new(m))) {
        return Err(format!("{name}: its member {bad} would land outside the install"));
    }
    let mut cmd = if zip {
        let mut c = Command::new("unzip");
        c.arg("-q").arg(archive).arg("-d").arg(into);
        c
    } else {
        let mut c = Command::new("tar");
        c.arg("-xf").arg(archive).arg("-C").arg(into).args(["--no-same-owner", "--no-same-permissions"]);
        c
    };
    let out = cmd.output().map_err(|e| format!("unpacking {name}: {e}"))?;
    if !out.status.success() {
        return Err(format!("unpacking {name}: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    contained(into).map_err(|e| format!("{name}: {e}"))
}

fn escapes(member: &Path) -> bool {
    use std::path::Component;
    member.components().any(|c| matches!(c, Component::RootDir | Component::Prefix(_) | Component::ParentDir))
}

/// Every symlink under `root` points inside it, so nothing that later moves or reads the tree
/// reaches past it.
fn contained(root: &Path) -> Result<(), String> {
    let canon = root.canonicalize().map_err(|e| format!("{}: {e}", root.display()))?;
    let mut dirs = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten() {
            let kind = entry.file_type().map_err(|e| format!("{}: {e}", entry.path().display()))?;
            if kind.is_dir() {
                dirs.push((entry.path(), depth + 1));
            } else if kind.is_symlink() {
                let link = entry.path();
                let target = std::fs::read_link(&link).map_err(|e| format!("{}: {e}", link.display()))?;
                let lexical = target.components().try_fold(depth as isize, |at, c| {
                    let at = match c {
                        std::path::Component::Normal(_) => at + 1,
                        std::path::Component::CurDir => at,
                        std::path::Component::ParentDir => at - 1,
                        _ => -1,
                    };
                    (at >= 0).then_some(at)
                });
                let inside = lexical.is_some() && link.canonicalize().map(|p| p.starts_with(&canon)).unwrap_or(true);
                if !inside {
                    return Err(format!("{} links to {}, outside the install", link.display(), target.display()));
                }
            }
        }
    }
    Ok(())
}

pub fn single_top(dir: &Path) -> Result<PathBuf, String> {
    let entries: Vec<std::fs::DirEntry> = std::fs::read_dir(dir)
        .map(|rd| rd.flatten().filter(|e| !e.file_name().to_string_lossy().starts_with('.')).collect())
        .unwrap_or_default();
    match entries.as_slice() {
        [only] => match only.file_type() {
            Ok(t) if t.is_symlink() => Err(format!("{}: an archive whose only entry is a symlink", only.path().display())),
            Ok(t) if t.is_dir() => Ok(only.path()),
            _ => Ok(dir.to_path_buf()),
        },
        _ => Ok(dir.to_path_buf()),
    }
}

pub fn fetch_and_unpack(plan: &Plan, into: &Path, dir: &Path) -> Result<Vec<(String, String)>, String> {
    let fetched = dir.join(".fetch");
    std::fs::create_dir_all(&fetched).map_err(|e| format!("{}: {e}", fetched.display()))?;
    std::fs::create_dir_all(into).map_err(|e| format!("{}: {e}", into.display()))?;
    let mut sources = Vec::new();
    for (i, asset) in plan.assets.iter().enumerate() {
        let (file, sha) = fetch(asset, &fetched)?;
        let unpacked = dir.join(format!(".unpack-{i}"));
        unpack(&file, &unpacked)?;
        let top = innermost(&unpacked)?;
        contained(&top).map_err(|e| format!("{}: {e}", asset.name))?;
        merge(&top, into)?;
        let _ = std::fs::remove_dir_all(&unpacked);
        let _ = std::fs::remove_file(&file);
        sources.push((asset.url.clone(), sha));
    }
    let _ = std::fs::remove_dir_all(&fetched);
    Ok(sources)
}

fn innermost(dir: &Path) -> Result<PathBuf, String> {
    let mut at = dir.to_path_buf();
    loop {
        let next = single_top(&at)?;
        if next == at {
            return Ok(at);
        }
        at = next;
    }
}

fn merge(from: &Path, into: &Path) -> Result<(), String> {
    for entry in std::fs::read_dir(from).map_err(|e| format!("{}: {e}", from.display()))?.flatten() {
        let src = entry.path();
        let dest = into.join(entry.file_name());
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if let Ok(there) = dest.symlink_metadata() {
            if is_dir && there.is_dir() {
                merge(&src, &dest)?;
            }
            continue;
        }
        std::fs::rename(&src, &dest).map_err(|e| format!("{} -> {}: {e}", src.display(), dest.display()))?;
    }
    Ok(())
}

pub fn copy_worker(install: &Path, id: &str) -> Result<PathBuf, String> {
    let bin = install.join("bin");
    std::fs::create_dir_all(&bin).map_err(|e| format!("{}: {e}", bin.display()))?;
    let dest = bin.join(format!("superfluid-worker-{id}"));
    if dest.is_file() {
        return Ok(dest);
    }
    let me = std::env::current_exe().map_err(|e| format!("this worker's own path: {e}"))?;
    std::fs::copy(&me, &dest).map_err(|e| format!("{} -> {}: {e}", me.display(), dest.display()))?;
    Ok(dest)
}

pub fn own_install() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let install = exe.parent()?.parent()?.to_path_buf();
    install.join("runtime.json").is_file().then_some(install)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hub_listing_gives_each_file_its_size_and_the_blob_its_bytes_take() {
        let v = json!({"sha": "f9c2", "siblings": [
            {"rfilename": ".gitattributes", "blobId": "6956", "size": 1700},
            {"rfilename": "m-Q4_K_M.gguf", "blobId": "8e4e", "size": 806058240, "lfs": {"sha256": "8ccc", "size": 806058240}},
        ]});
        let l = hf_listing_from("org/m", &v);
        assert_eq!(l.sha, "f9c2");
        assert_eq!(l.files[0], HfFile { path: ".gitattributes".into(), size: 1700, blob: "6956".into() });
        assert_eq!(l.files[1], HfFile { path: "m-Q4_K_M.gguf".into(), size: 806058240, blob: "8ccc".into() });
        assert_eq!(l.to_fetch(|p| p.ends_with(".gguf")), 806058240, "nothing of it is cached here");
    }

    #[test]
    fn shell_patterns_match_as_huggingface_hub_reads_them() {
        assert!(glob_matches("*.json", "config.json"));
        assert!(glob_matches("*.json", "sub/dir/config.json"));
        assert!(!glob_matches("*.json", "config.jsonl"));
        assert!(glob_matches("model*.safetensors", "model-00001-of-00002.safetensors"));
        assert!(glob_matches("model*.safetensors", "model.safetensors"));
        assert!(!glob_matches("model*.safetensors", "adapter_model.safetensors"));
        assert!(glob_matches("tokenizer.model", "tokenizer.model"));
        assert!(!glob_matches("tokenizer.model", "tokenizer.model.v2"));
        assert!(glob_matches("*", "anything/at/all"));
        assert!(glob_matches("a*b*c", "a-b-c"));
        assert!(!glob_matches("a*b*c", "a-c-b"));
    }

    #[test]
    fn the_hub_cache_is_where_huggingface_hub_and_llama_cpp_look() {
        // the environment is shared with the other tests: read, never set
        let cache = hf_hub_cache();
        assert!(cache.ends_with("hub") || std::env::var_os("HF_HUB_CACHE").is_some(), "{}", cache.display());
        assert!(hf_repo_cache("org/name").ends_with("models--org--name"));
        assert_eq!(hf_repo_cache("org/name").parent(), Some(cache.as_path()));
        if std::env::var_os("HF_HUB_OFFLINE").is_none() && std::env::var_os("TRANSFORMERS_OFFLINE").is_none() {
            assert!(!hf_offline());
        }
    }

    #[test]
    fn a_json_lookup_that_stalls_gives_up_by_its_deadline() {
        // A server that takes the connection and never answers, as a stalled API does.
        let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("https://127.0.0.1:{}/releases", server.local_addr().unwrap().port());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || tx.send(fetch_json_within(&url, Duration::from_secs(2))));
        let got = rx.recv_timeout(Duration::from_secs(15)).expect("the lookup gives up by its deadline");
        assert!(got.is_err(), "{got:?}");
        drop(server);
    }

    #[test]
    fn the_driver_version_is_read_from_the_kernel_module_line() {
        let line = "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  580.95.05  Release Build  (dvs-builder@U16-I1-N08-12-3)  Thu Sep 18 2025\nGCC version: gcc 13";
        assert_eq!(nvidia_driver_version(line).as_deref(), Some("580.95.05"));
        let host = Host { nvidia_driver: Some("580.95.05".into()), ..Host::default() };
        assert_eq!(host.nvidia_major(), Some(580));
        assert_eq!(nvidia_driver_version("no version here"), None);
    }

    #[test]
    fn an_amd_gpu_is_named_by_the_target_its_kfd_node_states() {
        let node = |v: u32| format!("cpu_cores_count 0\nsimd_count 80\nmax_waves_per_simd 16\ngfx_target_version {v}\nvendor_id 4098\n");
        assert_eq!(gfx_target(&node(110501)).as_deref(), Some("gfx1151"), "Strix Halo");
        assert_eq!(gfx_target(&node(110500)).as_deref(), Some("gfx1150"), "Strix Point");
        assert_eq!(gfx_target(&node(110000)).as_deref(), Some("gfx1100"));
        assert_eq!(gfx_target(&node(90010)).as_deref(), Some("gfx90a"), "a stepping past 9 is a hex digit");
        assert_eq!(gfx_target(&node(0)), None, "the CPU's node");
        assert_eq!(gfx_target("cpu_cores_count 16\n"), None);
        let host = Host { os: "linux".into(), arch: "x64".into(), rocm: true, amd_gpus: vec!["gfx1151".into()], rocm_runtimes: vec!["10.0".into()], ..Host::default() };
        assert_eq!(host.describe(), "Linux x64, AMD gfx1151, ROCm 10.0");
        assert_eq!(Host { rocm_runtimes: vec![], ..host }.describe(), "Linux x64, AMD gfx1151, ROCm");
    }

    #[test]
    fn hip_numbers_the_gpus_as_rocr_then_hip_read_their_lists() {
        let kfd: Vec<String> = ["gfx1100", "gfx1151", "gfx1150"].iter().map(|g| g.to_string()).collect();
        let seen = |rocr: Option<&str>, hip: Option<&str>| {
            let v = hip_view(&kfd, rocr, hip);
            (v.known, v.gpus, v.hip_indices)
        };
        let gpus = |v: &[&str]| v.iter().map(|g| g.to_string()).collect::<Vec<_>>();
        assert_eq!(seen(None, None), (true, kfd.clone(), vec![0, 1, 2]), "unset: node order");
        // ROCR_VISIBLE_DEVICES, as amd_filter_device.cpp reads it.
        assert_eq!(seen(Some(""), None), (true, vec![], vec![]), "set but empty: no GPU");
        assert_eq!(seen(Some("2, 0"), None), (true, gpus(&["gfx1150", "gfx1100"]), vec![0, 1]), "trimmed, in its order");
        assert_eq!(seen(Some("0x1"), None).1, gpus(&["gfx1151"]), "strtol base 0");
        assert_eq!(seen(Some("1,1,0"), None).1, gpus(&["gfx1151"]), "a repeat ends the list");
        assert_eq!(seen(Some("1,7,0"), None).1, gpus(&["gfx1151"]), "so does an index out of range");
        assert_eq!(seen(Some("1,x,0"), None).1, gpus(&["gfx1151"]), "and an entry that does not parse");
        assert_eq!(seen(Some("-1"), None).1, gpus(&[]));
        assert_eq!(seen(Some("1,"), None).1, gpus(&["gfx1151"]), "no entry after a final comma");
        assert_eq!(seen(Some("0,1,2,GPU-1a2b"), None).1, kfd, "entries past the GPU count are not read");
        assert!(!seen(Some("GPU-1a2b3c4d5e6f7a8b"), None).0, "a UUID: unknown");
        assert!(!seen(Some(",1"), None).0, "an empty entry: unknown");
        // HIP_VISIBLE_DEVICES, as rocdevice.cpp reads it, counting what ROCr left.
        assert_eq!(seen(None, Some("-1")), (true, vec![], vec![]), "invalid from the first entry: no GPU");
        assert_eq!(seen(None, Some("1,0")), (true, gpus(&["gfx1151", "gfx1100"]), vec![1, 0]));
        assert_eq!(seen(None, Some("1,1,0")).1, gpus(&["gfx1151", "gfx1100"]), "a repeat is skipped, not an end");
        assert_eq!(seen(None, Some("01")).1, gpus(&[]), "only the plain decimal form");
        assert_eq!(seen(None, Some("1, 0")).1, gpus(&["gfx1151"]), "no trimming");
        assert_eq!(seen(None, Some("1,,0")).1, gpus(&["gfx1151"]), "an empty entry ends the list");
        assert_eq!(seen(Some("2,1"), Some("1")), (true, gpus(&["gfx1151"]), vec![1]), "HIP counts ROCr's list");
        assert!(!seen(None, Some("0,GPU-1a2b")).0, "a UUID it reaches: unknown");
        assert!(seen(None, Some("9,GPU-1a2b")).0, "not one past an invalid entry");
    }

    #[test]
    fn rocm_runtimes_are_the_core_trees_with_libraries_under_the_root() {
        let root = std::env::temp_dir().join(format!("superfluid-kit-rocm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for dir in ["core-10.0/lib", "core-9.9/lib", "core-11.0", "lib", "share"] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        // As AMD's ROCm 10 packages leave it (Ubuntu 26.04): `core` and `core-10` link to the tree.
        std::os::unix::fs::symlink(root.join("core-10.0"), root.join("core-10")).unwrap();
        std::os::unix::fs::symlink(root.join("core-10.0"), root.join("core")).unwrap();
        assert_eq!(rocm_runtimes(&root), ["9.9", "10.0"], "by version, only a tree with its lib/, and not a link to one");
        assert!(rocm_runtimes(&root.join("missing")).is_empty());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn gpu_backends_rank_above_the_cpu_and_versions_compare_by_number() {
        assert!(backend_rank("metal") < backend_rank("cuda-13.4"));
        assert!(backend_rank("cuda-12.8") < backend_rank("vulkan"));
        assert!(backend_rank("vulkan") < backend_rank("cpu"));
        assert!(backend_rank("something-new") < backend_rank("cpu"));
        assert!(version_key("b11284") < version_key("b11300"));
        assert!(version_key("0.2.6") < version_key("0.3.0"));
    }

    #[test]
    fn a_request_reads_back_from_its_flags() {
        let r = Request { version: Some("b11284".into()), backend: Some("vulkan".into()), untested: true, ..Request::default() };
        assert_eq!(Request::parse(&r.to_args()).unwrap(), r);
        assert_eq!(Request::parse(&["--backend".into(), "CUDA".into()]).unwrap().backend.as_deref(), Some("cuda"));
        assert!(Request::parse(&["--versions".into(), "x".into()]).unwrap_err().contains("unknown flag"));
    }

    #[test]
    fn a_plan_and_a_manifest_read_back_from_their_json() {
        let plan = Plan {
            id: "llamacpp".into(),
            version: "b11284".into(),
            backend: "metal".into(),
            install: "b11284-metal".into(),
            assets: vec![Asset { name: "a.tar.gz".into(), url: "https://x/a.tar.gz".into(), sha256: Some("ab".into()), size: Some(3) }],
            tested: true,
            why: "macOS arm64: Metal".into(),
            offline: None,
            extra: json!({"index": "https://pypi.org/simple"}),
        };
        assert_eq!(Plan::from_json(&plan.to_json()).unwrap(), plan);
        assert_eq!(plan.size(), 3);
        let offline = Plan { offline: Some("fetching the listing failed".into()), ..plan.clone() };
        assert_eq!(Plan::from_json(&offline.to_json()).unwrap(), offline, "an offline plan says what it could not ask");
        assert_eq!(plan.to_json()["offline"], Value::Null);
        let m = Manifest {
            id: "llamacpp".into(),
            install: "b11284-metal".into(),
            version: "b11284".into(),
            backend: "metal".into(),
            link_w: vec![4],
            platforms: vec!["macos-arm64".into()],
            worker: "bin/superfluid-worker-llamacpp".into(),
            sources: vec![("https://x/a.tar.gz".into(), "ab".into())],
            tested: true,
            requires: Some("engine ABI 1".into()),
        };
        assert_eq!(Manifest::from_json(&m.to_json()).unwrap(), m);
        let older = Manifest { requires: None, ..m };
        assert_eq!(Manifest::from_json(&older.to_json()).unwrap(), older, "an install from before manifests kept a rule's ask");
    }

    #[test]
    fn archives_unpack_through_their_top_directory_and_the_first_keeps_a_shared_name() {
        let dir = std::env::temp_dir().join(format!("superfluid-kit-unpack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let make = |name: &str, files: &[(&str, &str)]| -> PathBuf {
            let src = dir.join(format!("src-{name}"));
            for (f, text) in files {
                let p = src.join(f);
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                std::fs::write(p, text).unwrap();
            }
            let tar = dir.join(format!("{name}.tar.gz"));
            let ok = Command::new("tar").arg("-czf").arg(&tar).arg("-C").arg(&src).arg(".").status().unwrap();
            assert!(ok.success());
            tar
        };
        std::fs::create_dir_all(dir.join("src-main/llama-b1")).unwrap();
        std::os::unix::fs::symlink("libllama.dylib", dir.join("src-main/llama-b1/libllama.0.dylib")).unwrap();
        let main = make("main", &[("llama-b1/libllama.dylib", "llama"), ("llama-b1/LICENSE", "mit")]);
        let rt = make("cudart", &[("libcudart.so.13", "cudart"), ("LICENSE", "nvidia")]);
        let asset = |p: &Path| Asset { name: p.file_name().unwrap().to_string_lossy().into(), url: p.display().to_string(), sha256: None, size: None };
        let plan = Plan {
            id: "x".into(),
            version: "1".into(),
            backend: "cpu".into(),
            install: "1-cpu".into(),
            assets: vec![asset(&main), asset(&rt)],
            tested: true,
            why: String::new(),
            offline: None,
            extra: Value::Null,
        };
        let stage = dir.join("stage");
        std::fs::create_dir_all(&stage).unwrap();
        let sources = fetch_and_unpack(&plan, &stage.join("lib"), &stage).unwrap();
        assert_eq!(sources.len(), 2);
        let lib = stage.join("lib");
        assert_eq!(std::fs::read_to_string(lib.join("libllama.dylib")).unwrap(), "llama");
        assert_eq!(std::fs::read_to_string(lib.join("libcudart.so.13")).unwrap(), "cudart");
        assert_eq!(std::fs::read_to_string(lib.join("LICENSE")).unwrap(), "mit", "the runtime's own archive keeps its name");
        assert_eq!(std::fs::read_to_string(lib.join("libllama.0.dylib")).unwrap(), "llama", "a link inside the archive is kept");
        let left: Vec<String> = std::fs::read_dir(&stage).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(left, ["lib"], "nothing of the fetch is left behind");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_local_file_is_fetched_and_checked_against_its_digest() {
        let dir = std::env::temp_dir().join(format!("superfluid-kit-fetch-{}", std::process::id()));
        let out = dir.join("out");
        std::fs::create_dir_all(&out).unwrap();
        let src = dir.join("src.bin");
        std::fs::write(&src, b"abc").unwrap();
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let asset = |sha256: Option<&str>, size| Asset { name: "f.bin".into(), url: src.display().to_string(), sha256: sha256.map(str::to_string), size };
        assert_eq!(fetch(&asset(Some(sha), Some(3)), &out).unwrap().1, sha);
        assert!(fetch(&asset(Some(&"0".repeat(64)), None), &out).unwrap_err().contains("not the file its project published"));
        assert!(fetch(&asset(None, Some(4)), &out).unwrap_err().contains("3 bytes, expected 4"));
        assert_eq!(fetch(&asset(None, None), &out).unwrap().1, sha, "a local file is installed unchecked, with a warning");
        let http = Asset { name: "x".into(), url: "http://example.com/x".into(), sha256: Some(sha.into()), size: None };
        assert!(fetch(&http, &out).unwrap_err().contains("only https://"));
        let remote = Asset { name: "x.tar.gz".into(), url: "https://example.invalid/x.tar.gz".into(), sha256: None, size: None };
        let e = fetch(&remote, &out).unwrap_err();
        assert!(e.contains("no SHA-256 to check it against") && e.contains("--sha256"), "{e}");
        assert!(!out.join("x.tar.gz").exists(), "nothing is fetched to find out");
        assert_eq!(unchecked(&Asset { url: format!("file://{}", src.display()), ..remote.clone() }), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_archive_reaches_nothing_outside_its_install() {
        let dir = std::env::temp_dir().join(format!("superfluid-kit-escape-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let outside = dir.join("outside");
        std::fs::create_dir_all(outside.join("etc")).unwrap();
        std::fs::write(outside.join("etc/passwd"), "root").unwrap();
        let tar = |name: &str, at: &Path, members: &[&str]| -> PathBuf {
            let out = dir.join(format!("{name}.tar"));
            assert!(Command::new("tar").arg("-cPf").arg(&out).args(members).current_dir(at).status().unwrap().success());
            out
        };
        let run = |archive: PathBuf| -> String {
            let stage = dir.join("stage");
            let _ = std::fs::remove_dir_all(&stage);
            std::fs::create_dir_all(&stage).unwrap();
            let name = archive.file_name().unwrap().to_string_lossy().into_owned();
            let a = Asset { name, url: archive.display().to_string(), sha256: None, size: None };
            let plan = Plan {
                id: "x".into(),
                version: "1".into(),
                backend: "cpu".into(),
                install: "1-cpu".into(),
                assets: vec![a],
                tested: true,
                why: String::new(),
                offline: None,
                extra: Value::Null,
            };
            fetch_and_unpack(&plan, &stage.join("lib"), &stage).unwrap_err()
        };

        let top = dir.join("top");
        std::fs::create_dir_all(&top).unwrap();
        std::os::unix::fs::symlink(&outside, top.join("pkg")).unwrap();
        let e = run(tar("top", &top, &["pkg"]));
        assert!(e.contains("outside the install"), "{e}");

        let rel = dir.join("rel/pkg");
        std::fs::create_dir_all(&rel).unwrap();
        std::fs::write(rel.join("libx.so"), "x").unwrap();
        std::os::unix::fs::symlink("../../../outside/etc", rel.join("etc")).unwrap();
        let e = run(tar("rel", &dir.join("rel"), &["pkg"]));
        assert!(e.contains("pkg/etc links to ../../../outside/etc, outside the install"), "{e}");

        let up = dir.join("up/sub");
        std::fs::create_dir_all(&up).unwrap();
        std::fs::write(dir.join("up/evil"), "x").unwrap();
        let e = run(tar("up", &up, &["../evil"]));
        assert!(e.contains("../evil would land outside the install"), "{e}");

        assert_eq!(std::fs::read_to_string(outside.join("etc/passwd")).unwrap(), "root", "what is outside stays where it was");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_symlinked_top_entry_is_refused() {
        let dir = std::env::temp_dir().join(format!("superfluid-kit-top-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("real")).unwrap();
        std::fs::create_dir_all(dir.join("u")).unwrap();
        std::os::unix::fs::symlink(dir.join("real"), dir.join("u/pkg")).unwrap();
        assert!(single_top(&dir.join("u")).unwrap_err().contains("only entry is a symlink"));
        std::fs::create_dir_all(dir.join("d/pkg")).unwrap();
        assert_eq!(single_top(&dir.join("d")).unwrap(), dir.join("d/pkg"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
