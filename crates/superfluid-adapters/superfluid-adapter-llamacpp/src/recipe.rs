//! What installing llama.cpp fetches.

use superfluid_adapter_kit::install::{fetch_json, version_key, Asset, Host, Plan, Request};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Archive {
    pub name: String,
    pub size: Option<u64>,
    pub sha256: Option<String>,
    pub version: String,
    pub platform: String,
    pub backend: String,
    pub cudart: bool,
}

pub fn parse_name(name: &str) -> Option<Archive> {
    let (cudart, rest) = match name.strip_prefix("cudart-") {
        Some(r) => (true, r),
        None => (false, name),
    };
    let rest = rest.strip_prefix("llama-")?.strip_suffix(".tar.gz")?;
    let (version, rest) = rest.split_once("-bin-")?;
    if !version.starts_with('b') || !version[1..].chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (os, rest) = rest.split_once('-')?;
    let os = match os {
        "macos" => "macos",
        "ubuntu" => "linux",
        _ => return None,
    };
    let (backend, arch) = match rest.rsplit_once('-') {
        Some((backend, arch)) => (backend.to_string(), arch),
        None => (String::new(), rest),
    };
    if !matches!(arch, "arm64" | "x64") {
        return None;
    }
    let backend = match (os, backend.as_str()) {
        ("macos", "") if arch == "arm64" => "metal".to_string(),
        (_, "") => "cpu".to_string(),
        _ => backend,
    };
    Some(Archive {
        name: name.to_string(),
        size: None,
        sha256: None,
        version: version.to_string(),
        platform: format!("{os}-{arch}"),
        backend,
        cudart,
    })
}

fn release_url(recipe: &Value, version: &str, name: &str) -> String {
    recipe["release"].as_str().unwrap_or_default().replace("{version}", version).replace("{name}", name)
}

fn recipe() -> Value {
    serde_json::from_str(include_str!("../recipe.json")).expect("recipe.json is JSON")
}

pub fn tested() -> Vec<(String, Vec<Archive>)> {
    let r = recipe();
    let mut builds: Vec<(String, Vec<Archive>)> = r["tested"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|b| {
                    let version = b["version"].as_str()?.to_string();
                    let archives = b["assets"].as_array()?.iter().filter_map(archive_of).collect();
                    Some((version, archives))
                })
                .collect()
        })
        .unwrap_or_default();
    builds.sort_by_key(|(v, _)| std::cmp::Reverse(version_key(v)));
    builds
}

fn archive_of(a: &Value) -> Option<Archive> {
    let mut archive = parse_name(a["name"].as_str()?)?;
    archive.size = a["size"].as_u64();
    archive.sha256 = a["sha256"].as_str().map(str::to_string).or_else(|| digest(&a["digest"]));
    Some(archive)
}

fn digest(v: &Value) -> Option<String> {
    v.as_str()?.strip_prefix("sha256:").map(str::to_string)
}

fn published(version: &str) -> Result<Vec<Archive>, String> {
    let r = recipe();
    let url = r["api"].as_str().unwrap_or_default().replace("{version}", version);
    let release = fetch_json(&url).map_err(|e| format!("llama.cpp {version}: {e}"))?;
    let archives: Vec<Archive> = release["assets"].as_array().map(|a| a.iter().filter_map(archive_of).collect()).unwrap_or_default();
    if archives.is_empty() {
        return Err(format!("llama.cpp {version} publishes no archive this adapter knows"));
    }
    Ok(archives)
}

fn cuda_major_for_driver(driver_major: u32) -> Option<u32> {
    match driver_major {
        580.. => Some(13),
        525.. => Some(12),
        _ => None,
    }
}

fn cuda_version(backend: &str) -> Option<(u32, u32)> {
    let v = backend.strip_prefix("cuda-")?;
    let (major, minor) = v.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn choose(host: &Host, have: &[&Archive]) -> Vec<(String, String)> {
    let builds: Vec<&str> = have.iter().filter(|a| !a.cudart).map(|a| a.backend.as_str()).collect();
    let has = |b: &str| builds.contains(&b);
    let mut out = Vec::new();
    if host.os == "macos" {
        let b = if has("metal") { "metal" } else { "cpu" };
        out.push((b.to_string(), format!("{}: the {} build", host.describe(), if b == "metal" { "Metal" } else { "CPU" })));
        return out;
    }
    if let Some(driver) = host.nvidia_major() {
        match cuda_major_for_driver(driver) {
            Some(max) => {
                let best = builds
                    .iter()
                    .filter_map(|b| cuda_version(b).map(|v| (v, *b)))
                    .filter(|((major, _), _)| *major <= max)
                    .max_by_key(|(v, _)| *v);
                if let Some((_, b)) = best {
                    out.push((b.to_string(), format!("{}: NVIDIA driver {driver} runs CUDA {max}", host.describe())));
                }
            }
            None => out.push((String::new(), format!("NVIDIA driver {driver} is older than any CUDA build here runs"))),
        }
    }
    if host.rocm {
        if let Some(b) = builds.iter().find(|b| b.starts_with("rocm")) {
            out.push((b.to_string(), format!("{}: ROCm", host.describe())));
        }
    }
    if host.vulkan && has("vulkan") {
        out.push(("vulkan".to_string(), format!("{}: a Vulkan driver", host.describe())));
    }
    if has("cpu") {
        out.push(("cpu".to_string(), format!("{}: the CPU build", host.describe())));
    }
    out.retain(|(b, _)| !b.is_empty());
    out
}

pub fn plan(req: &Request, host: &Host) -> Result<Vec<Plan>, String> {
    let r = recipe();
    let builds = tested();
    let tested_versions: Vec<&str> = builds.iter().map(|(v, _)| v.as_str()).collect();
    let untested = |version: &str| -> Result<(), String> {
        if tested_versions.contains(&version) || req.untested {
            return Ok(());
        }
        Err(format!(
            "llama.cpp {version} is not a build this adapter was tested with (tested: {}); its C API may differ — \
             pass --untested to install it anyway",
            tested_versions.join(", ")
        ))
    };

    if let Some(from) = &req.from {
        let name = from.rsplit('/').next().unwrap_or(from).to_string();
        let parsed = parse_name(&name);
        let version = req.version.clone().or_else(|| parsed.as_ref().map(|a| a.version.clone())).unwrap_or_else(|| "custom".into());
        untested(&version)?;
        let backend = req.backend.clone().or_else(|| parsed.as_ref().map(|a| a.backend.clone())).unwrap_or_else(|| "cpu".into());
        let known = builds.iter().find(|(v, _)| *v == version).and_then(|(_, a)| a.iter().find(|a| a.name == name));
        let sha256 = req.sha256.clone().or_else(|| known.and_then(|a| a.sha256.clone()));
        return Ok(vec![Plan {
            id: "llamacpp".into(),
            install: format!("{version}-{backend}"),
            assets: vec![Asset { name, url: from.clone(), sha256, size: known.and_then(|a| a.size) }],
            tested: tested_versions.contains(&version.as_str()),
            why: format!("{from}, as named"),
            version,
            backend,
            extra: Value::Null,
        }]);
    }

    let version = req.version.clone().or_else(|| builds.first().map(|(v, _)| v.clone())).ok_or("the recipe lists no build")?;
    untested(&version)?;
    let archives = match builds.iter().find(|(v, _)| *v == version) {
        Some((_, a)) => a.clone(),
        None => published(&version)?,
    };
    let platform = host.platform();
    let here: Vec<&Archive> = archives.iter().filter(|a| a.platform == platform).collect();
    if here.is_empty() {
        return Err(format!("llama.cpp {version} publishes no build for {platform}"));
    }
    let choices: Vec<(String, String)> = match &req.backend {
        Some(want) => {
            let driver_runs = host.nvidia_major().and_then(cuda_major_for_driver);
            let within = |b: &String| match (cuda_version(b), driver_runs) {
                (Some((major, _)), Some(max)) => b == want || major <= max,
                _ => true,
            };
            let named: Vec<String> = here
                .iter()
                .filter(|a| !a.cudart && (a.backend == *want || a.backend.split('-').next() == Some(want.as_str())))
                .map(|a| a.backend.clone())
                .collect();
            let newest = |of: &[String]| of.iter().max_by_key(|b| cuda_version(b).unwrap_or((0, 0))).cloned();
            let b = newest(&named.iter().filter(|b| within(b)).cloned().collect::<Vec<_>>())
                .or_else(|| newest(&named))
                .ok_or_else(|| {
                    let mut all: Vec<&str> = here.iter().filter(|a| !a.cudart).map(|a| a.backend.as_str()).collect();
                    all.sort();
                    format!("llama.cpp {version} has no {want} build for {platform} (it has: {})", all.join(", "))
                })?;
            vec![(b, format!("{}: {want}, as named", host.describe()))]
        }
        None => choose(host, &here),
    };
    if choices.is_empty() {
        return Err(format!("no llama.cpp {version} build for {} runs here", host.describe()));
    }
    let asset = |a: &Archive| Asset {
        name: a.name.clone(),
        url: release_url(&r, &version, &a.name),
        sha256: a.sha256.clone(),
        size: a.size,
    };
    Ok(choices
        .into_iter()
        .filter_map(|(backend, why)| {
            let build = here.iter().find(|a| !a.cudart && a.backend == backend)?;
            let mut assets = vec![asset(build)];
            let mut why = why;
            if let Some((major, _)) = cuda_version(&backend) {
                if host.cuda_runtime != Some(major) {
                    if let Some(rt) = here.iter().find(|a| a.cudart && a.backend == backend) {
                        assets.push(asset(rt));
                        why.push_str(&format!("; no CUDA {major} runtime on the system, so its archive too"));
                    }
                }
            }
            Some(Plan {
                id: "llamacpp".into(),
                version: version.clone(),
                install: format!("{version}-{backend}"),
                backend,
                assets,
                tested: tested_versions.contains(&version.as_str()),
                why,
                extra: json!({}),
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(os: &str, arch: &str) -> Host {
        Host { os: os.into(), arch: arch.into(), ..Host::default() }
    }

    fn backends(plans: &[Plan]) -> Vec<&str> {
        plans.iter().map(|p| p.backend.as_str()).collect()
    }

    #[test]
    fn archive_names_say_version_platform_and_backend() {
        let a = parse_name("llama-b11284-bin-ubuntu-cuda-13.4-arm64.tar.gz").unwrap();
        assert_eq!((a.version.as_str(), a.platform.as_str(), a.backend.as_str(), a.cudart), ("b11284", "linux-arm64", "cuda-13.4", false));
        let a = parse_name("cudart-llama-b11284-bin-ubuntu-cuda-12.8-x64.tar.gz").unwrap();
        assert_eq!((a.platform.as_str(), a.backend.as_str(), a.cudart), ("linux-x64", "cuda-12.8", true));
        assert_eq!(parse_name("llama-b11284-bin-macos-arm64.tar.gz").unwrap().backend, "metal");
        assert_eq!(parse_name("llama-b11284-bin-macos-x64.tar.gz").unwrap().backend, "cpu");
        assert_eq!(parse_name("llama-b11284-bin-ubuntu-x64.tar.gz").unwrap().backend, "cpu");
        assert_eq!(parse_name("llama-b11284-bin-ubuntu-openvino-2026.4-x64.tar.gz").unwrap().backend, "openvino-2026.4");
        for bad in ["llama-b11284-bin-win-cpu-x64.zip", "llama-b11284-xcframework.zip", "llama-v1-bin-ubuntu-x64.tar.gz", "llama-b1-bin-ubuntu-s390x.tar.gz"] {
            assert_eq!(parse_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_recipe_lists_a_tested_build_with_every_archive_checked() {
        let builds = tested();
        let (version, archives) = &builds[0];
        assert!(version.starts_with('b'), "{version}");
        assert!(archives.iter().all(|a| a.sha256.as_ref().is_some_and(|s| s.len() == 64) && a.size.is_some_and(|s| s > 0)));
        for platform in ["macos-arm64", "linux-x64", "linux-arm64"] {
            assert!(archives.iter().any(|a| a.platform == platform && !a.cudart), "{platform}");
        }
    }

    #[test]
    fn apple_silicon_gets_metal_and_an_intel_mac_the_cpu_build() {
        let plans = plan(&Request::default(), &host("macos", "arm64")).unwrap();
        assert_eq!(backends(&plans), ["metal"]);
        let p = &plans[0];
        assert_eq!(p.install, format!("{}-metal", p.version));
        assert!(p.tested && p.assets.len() == 1 && p.assets[0].url.starts_with("https://github.com/ggml-org/llama.cpp/releases/download/"));
        assert_eq!(backends(&plan(&Request::default(), &host("macos", "x64")).unwrap()), ["cpu"]);
    }

    #[test]
    fn linux_picks_the_newest_cuda_its_driver_runs_then_falls_back() {
        let mut h = host("linux", "x64");
        h.nvidia_driver = Some("580.95.05".into());
        h.vulkan = true;
        let plans = plan(&Request::default(), &h).unwrap();
        assert_eq!(backends(&plans), ["cuda-13.4", "vulkan", "cpu"]);
        assert_eq!(plans[0].assets.len(), 2, "no CUDA runtime on the system: its archive comes too");
        assert!(plans[0].assets[1].name.starts_with("cudart-"));
        h.cuda_runtime = Some(13);
        assert_eq!(plan(&Request::default(), &h).unwrap()[0].assets.len(), 1, "the system's CUDA 13 runtime serves");
        h.nvidia_driver = Some("535.10".into());
        assert_eq!(backends(&plan(&Request::default(), &h).unwrap())[0], "cuda-12.8", "a 12.x driver runs CUDA 12 only");
        let mut arm = host("linux", "arm64");
        arm.nvidia_driver = Some("535.10".into());
        assert_eq!(backends(&plan(&Request::default(), &arm).unwrap()), ["cpu"], "no CUDA 12 build for arm64");
        let mut amd = host("linux", "x64");
        amd.rocm = true;
        assert_eq!(backends(&plan(&Request::default(), &amd).unwrap()), ["rocm-10.0", "cpu"]);
    }

    #[test]
    fn a_named_backend_is_the_only_one_and_must_exist() {
        let mut h = host("linux", "x64");
        h.nvidia_driver = Some("580.1".into());
        let named = |b: &str| plan(&Request { backend: Some(b.into()), ..Request::default() }, &h);
        assert_eq!(backends(&named("vulkan").unwrap()), ["vulkan"]);
        assert_eq!(backends(&named("cuda").unwrap()), ["cuda-13.4"], "a family names its newest");
        assert_eq!(backends(&named("cuda-12.8").unwrap()), ["cuda-12.8"]);
        assert!(named("metal").unwrap_err().contains("has no metal build for linux-x64"));
        h.nvidia_driver = Some("535.10".into());
        let named = |b: &str| plan(&Request { backend: Some(b.into()), ..Request::default() }, &h);
        assert_eq!(backends(&named("cuda").unwrap()), ["cuda-12.8"]);
        assert_eq!(backends(&named("cuda-13.4").unwrap()), ["cuda-13.4"], "a build named in full is the one asked for");
        h.nvidia_driver = None;
        let named = |b: &str| plan(&Request { backend: Some(b.into()), ..Request::default() }, &h);
        assert_eq!(backends(&named("cuda").unwrap()), ["cuda-13.4"]);
    }

    #[test]
    fn an_untested_build_needs_the_operator_to_say_so() {
        let e = plan(&Request { version: Some("b1".into()), ..Request::default() }, &host("macos", "arm64")).unwrap_err();
        assert!(e.contains("not a build this adapter was tested with") && e.contains("--untested"), "{e}");
    }

    #[test]
    fn a_named_archive_of_a_tested_build_is_checked_against_the_recipe() {
        let (version, archives) = &tested()[0];
        let mac = archives.iter().find(|a| a.platform == "macos-arm64").unwrap();
        let req = Request { from: Some(format!("https://mirror.example/llama/{}", mac.name)), ..Request::default() };
        let plans = plan(&req, &host("linux", "x64")).unwrap();
        assert_eq!(plans.len(), 1);
        let p = &plans[0];
        assert_eq!((p.version.as_str(), p.backend.as_str(), p.install.clone()), (version.as_str(), "metal", format!("{version}-metal")));
        assert_eq!(p.assets[0].sha256, mac.sha256, "the recipe's digest, fetched from anywhere");
        let custom = Request { from: Some("/tmp/llama-custom.tar.gz".into()), ..Request::default() };
        assert!(plan(&custom, &host("linux", "x64")).unwrap_err().contains("custom is not a build"));
        let custom = Request { untested: true, backend: Some("vulkan".into()), ..custom };
        let p = &plan(&custom, &host("linux", "x64")).unwrap()[0];
        assert_eq!((p.install.as_str(), p.tested, p.assets[0].sha256.clone()), ("custom-vulkan", false, None));
    }
}
