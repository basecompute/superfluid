//! Installing a runtime from where its own project publishes it.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Host {
    pub os: String,
    pub arch: String,
    pub nvidia_driver: Option<String>,
    pub cuda_runtime: Option<u32>,
    pub rocm: bool,
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
        if self.rocm {
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
            "cuda_runtime": self.cuda_runtime, "rocm": self.rocm, "vulkan": self.vulkan,
        })
    }
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
    pub extra: Value,
}

impl Plan {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id, "version": self.version, "backend": self.backend, "install": self.install,
            "assets": self.assets.iter().map(|a| json!({"name": a.name, "url": a.url, "sha256": a.sha256, "size": a.size})).collect::<Vec<_>>(),
            "tested": self.tested, "why": self.why, "extra": self.extra,
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
            extra: v["extra"].clone(),
        })
    }

    pub fn size(&self) -> u64 {
        self.assets.iter().filter_map(|a| a.size).sum()
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
}

impl Manifest {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id, "install": self.install, "version": self.version, "backend": self.backend,
            "link_w": self.link_w, "platforms": self.platforms, "worker": self.worker,
            "sources": self.sources.iter().map(|(u, s)| json!({"url": u, "sha256": s})).collect::<Vec<_>>(),
            "tested": self.tested,
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
        let size = asset.size.map(|s| format!(" ({:.1} MB)", s as f64 / 1e6)).unwrap_or_default();
        eprintln!("fetching {src}{size}");
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
    if !url.starts_with("https://") {
        return Err(format!("{url}: only https:// URLs are fetched"));
    }
    let out = Command::new("curl")
        .args(["-fsSL", "--proto", "=https", "--tlsv1.2", "--retry", "2", "-H", "Accept: application/json"])
        .arg(url)
        .output()
        .map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        return Err(format!("fetching {url} failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("{url}: {e}"))
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
    fn the_driver_version_is_read_from_the_kernel_module_line() {
        let line = "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  580.95.05  Release Build  (dvs-builder@U16-I1-N08-12-3)  Thu Sep 18 2025\nGCC version: gcc 13";
        assert_eq!(nvidia_driver_version(line).as_deref(), Some("580.95.05"));
        let host = Host { nvidia_driver: Some("580.95.05".into()), ..Host::default() };
        assert_eq!(host.nvidia_major(), Some(580));
        assert_eq!(nvidia_driver_version("no version here"), None);
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
            extra: json!({"index": "https://pypi.org/simple"}),
        };
        assert_eq!(Plan::from_json(&plan.to_json()).unwrap(), plan);
        assert_eq!(plan.size(), 3);
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
        };
        assert_eq!(Manifest::from_json(&m.to_json()).unwrap(), m);
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
