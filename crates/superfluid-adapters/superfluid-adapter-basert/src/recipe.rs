//! What installing basert fetches.

use superfluid_adapter_kit::install::{fetch_json, version_key, Asset, Host, Plan, Request};
use serde_json::Value;

fn recipe() -> Value {
    serde_json::from_str(include_str!("../recipe.json")).expect("recipe.json is JSON")
}

pub fn pinned() -> Vec<(String, Vec<Asset>)> {
    let r = recipe();
    let url = |version: &str, name: &str| r["release"].as_str().unwrap_or_default().replace("{version}", version).replace("{name}", name);
    r["pinned"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let version = p["version"].as_str()?.to_string();
                    let assets = p["assets"]
                        .as_array()?
                        .iter()
                        .filter_map(|x| {
                            let name = x["name"].as_str()?.to_string();
                            Some(Asset { url: url(&version, &name), sha256: Some(x["sha256"].as_str()?.to_string()), size: x["size"].as_u64(), name })
                        })
                        .collect();
                    Some((version, assets))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn platform(host: &Host) -> Result<(&'static str, &'static str), String> {
    match (host.os.as_str(), host.arch.as_str()) {
        ("macos", "arm64") => Ok(("macos-arm64", "metal")),
        ("linux", "arm64") if host.nvidia_driver.is_some() => Ok(("linux-arm64-cuda", "cuda")),
        _ => Err(format!(
            "baseRT publishes engines for Apple silicon and for Linux arm64 with an NVIDIA GPU; this is {}",
            host.describe()
        )),
    }
}

fn minor(v: &str) -> Vec<u64> {
    version_key(v).into_iter().take(2).collect()
}

pub fn version_of(name: &str) -> Option<String> {
    let v = name.strip_prefix("basert-engine-")?.strip_suffix(".tar.gz")?.rsplit('-').next()?;
    (!v.is_empty() && v.chars().next()?.is_ascii_digit()).then(|| v.to_string())
}

pub fn plan(req: &Request, host: &Host) -> Result<Vec<Plan>, String> {
    let pins = pinned();
    plan_with(req, host, &pins, &Value::Null).or_else(|why| {
        if req.from.is_some() {
            return Err(why);
        }
        let releases = fetch_json(recipe()["releases"].as_str().unwrap_or_default())?;
        plan_with(req, host, &pins, &releases)
    })
}

pub fn plan_with(req: &Request, host: &Host, pins: &[(String, Vec<Asset>)], releases: &Value) -> Result<Vec<Plan>, String> {
    let built_for = superfluid_engine_ffi::libbasert::BUILT_FOR;
    let (part, backend) = platform(host)?;
    if let Some(backend_asked) = &req.backend {
        if backend_asked != backend {
            return Err(format!("baseRT's engine runs on {backend} here, not {backend_asked}"));
        }
    }
    let tested = |v: &str| minor(v) == minor(built_for);
    let refuse_untested = |v: &str| -> Result<(), String> {
        if tested(v) || req.untested {
            return Ok(());
        }
        Err(format!(
            "basert {v} is not a version this adapter drives (it was built for {built_for}: libbaseRT refuses another minor \
             version); pass --untested to install it anyway"
        ))
    };
    let engine = |v: &str| format!("basert-engine-{part}-{v}.tar.gz");
    let pin = |v: &str| pins.iter().find(|(have, _)| have == v).and_then(|(_, a)| a.iter().find(|a| a.name == engine(v))).cloned();
    if let Some(from) = &req.from {
        let name = from.rsplit('/').next().unwrap_or(from).to_string();
        let version = req.version.clone().or_else(|| version_of(&name)).unwrap_or_else(|| "custom".into());
        refuse_untested(&version)?;
        let known = pin(&version).filter(|a| a.name == name);
        return Ok(vec![Plan {
            id: "basert".into(),
            install: format!("{version}-{backend}"),
            assets: vec![Asset {
                name,
                url: from.clone(),
                sha256: req.sha256.clone().or_else(|| known.as_ref().and_then(|a| a.sha256.clone())),
                size: known.and_then(|a| a.size),
            }],
            tested: tested(&version),
            why: format!("{from}, as named"),
            version,
            backend: backend.into(),
            extra: Value::Null,
        }]);
    }
    let mut published: Vec<(String, Asset)> = releases
        .as_array()
        .map(|rs| {
            rs.iter()
                .filter(|r| !r["draft"].as_bool().unwrap_or(false))
                .filter_map(|r| {
                    let version = r["tag_name"].as_str()?.trim_start_matches('v').to_string();
                    let want = engine(&version);
                    let a = r["assets"].as_array()?.iter().find(|a| a["name"] == want.as_str())?;
                    let asset = Asset { name: want, url: a["browser_download_url"].as_str()?.to_string(), sha256: None, size: a["size"].as_u64() };
                    Some((version, asset))
                })
                .collect()
        })
        .unwrap_or_default();
    published.sort_by_key(|(v, _)| std::cmp::Reverse(version_key(v)));
    let newest = published.first().map(|(v, _)| v.clone()).unwrap_or_else(|| "none".into());
    let checked = |version: String, asset: Asset| -> Result<(String, Asset), String> {
        if let Some(pinned) = pin(&version) {
            let sha256 = req.sha256.clone().or(pinned.sha256.clone());
            return Ok((version, Asset { sha256, ..pinned }));
        }
        match &req.sha256 {
            Some(sha) => Ok((version, Asset { sha256: Some(sha.clone()), ..asset })),
            None => Err(format!(
                "basert {version} has no SHA-256 pinned in this superfluid, so it is not installed on what the release page alone \
                 says: check the engine's SHA-256 (baseRT publishes it as {}.sha256) and pass it with --version {version} --sha256 <digest>",
                asset.name
            )),
        }
    };
    let mut pinned_here: Vec<(String, Asset)> = pins.iter().filter_map(|(v, _)| Some((v.clone(), pin(v)?))).collect();
    pinned_here.sort_by_key(|(v, _)| std::cmp::Reverse(version_key(v)));
    let (version, asset) = match &req.version {
        Some(v) => {
            let v = v.trim_start_matches('v');
            refuse_untested(v)?;
            let found = pinned_here.into_iter().chain(published).find(|(have, _)| have == v).ok_or_else(|| {
                format!("baseRT publishes no basert {v} engine for {part} (the newest is {newest})")
            })?;
            checked(found.0, found.1)?
        }
        None => {
            let found = pinned_here.into_iter().find(|(v, _)| tested(v)).or_else(|| published.into_iter().find(|(v, _)| tested(v)));
            let found = found.ok_or_else(|| {
                format!(
                    "baseRT publishes no {}.x engine for {part} yet (the newest is {newest}); this superfluid drives {built_for}. \
                     Install a bundle with --from <path|url> --sha256 <digest>, or another version with --version <v> --untested",
                    minor(built_for).iter().map(u64::to_string).collect::<Vec<_>>().join(".")
                )
            })?;
            checked(found.0, found.1)?
        }
    };
    Ok(vec![Plan {
        id: "basert".into(),
        install: format!("{version}-{backend}"),
        tested: tested(&version),
        why: format!("{}: baseRT's {} engine", host.describe(), if backend == "metal" { "Metal" } else { "CUDA" }),
        assets: vec![asset],
        version,
        backend: backend.into(),
        extra: Value::Null,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mac() -> Host {
        Host { os: "macos".into(), arch: "arm64".into(), ..Host::default() }
    }

    fn release(v: &str) -> Value {
        json!({
            "tag_name": format!("v{v}"), "draft": false,
            "assets": [
                {"name": format!("basert-engine-macos-arm64-{v}.tar.gz"), "size": 10, "digest": "sha256:ab",
                 "browser_download_url": format!("https://github.com/basecompute/baseRT/releases/download/v{v}/basert-engine-macos-arm64-{v}.tar.gz")},
                {"name": format!("basert-engine-macos-arm64-{v}.tar.gz.sha256"), "size": 1, "browser_download_url": "x"},
            ],
        })
    }

    fn built_for() -> String {
        superfluid_engine_ffi::libbasert::BUILT_FOR.to_string()
    }

    fn pin(v: &str, sha: &str) -> (String, Vec<Asset>) {
        let name = format!("basert-engine-macos-arm64-{v}.tar.gz");
        (v.to_string(), vec![Asset { url: format!("https://pinned/{name}"), name, sha256: Some(sha.into()), size: Some(10) }])
    }

    #[test]
    fn the_newest_pinned_release_of_the_built_for_minor_version_is_installed() {
        let b = built_for();
        let mut parts: Vec<u64> = version_key(&b);
        parts.truncate(2);
        let same = format!("{}.{}.9", parts[0], parts[1]);
        let older = format!("{}.{}.0", parts[0], parts[1].saturating_sub(1));
        let releases = json!([release(&older), release(&same)]);
        let pins = [pin(&older, "cd"), pin(&same, "ef")];
        let p = &plan_with(&Request::default(), &mac(), &pins, &releases).unwrap()[0];
        assert_eq!((p.version.as_str(), p.backend.as_str(), p.tested), (same.as_str(), "metal", true));
        assert_eq!(p.install, format!("{same}-metal"));
        assert_eq!(p.assets[0].sha256.as_deref(), Some("ef"), "the pinned digest, not the one the release page shows");
        let p = &plan_with(&Request::default(), &mac(), &pins, &Value::Null).unwrap()[0];
        assert_eq!(p.assets[0].url, format!("https://pinned/basert-engine-macos-arm64-{same}.tar.gz"), "a pin needs no release listing");
        let e = plan_with(&Request { version: Some(older.clone()), ..Request::default() }, &mac(), &pins, &releases).unwrap_err();
        assert!(e.contains("--untested"), "{e}");
        let p = &plan_with(&Request { version: Some(older.clone()), untested: true, ..Request::default() }, &mac(), &pins, &releases).unwrap()[0];
        assert!(!p.tested && p.version == older && p.assets[0].sha256.as_deref() == Some("cd"));
    }

    #[test]
    fn a_release_with_no_pinned_digest_needs_one_named() {
        let b = built_for();
        let parts: Vec<u64> = version_key(&b);
        let same = format!("{}.{}.9", parts[0], parts[1]);
        let releases = json!([release(&same)]);
        let e = plan_with(&Request::default(), &mac(), &[], &releases).unwrap_err();
        assert!(e.contains(&format!("basert {same} has no SHA-256 pinned")) && e.contains("--sha256"), "{e}");
        assert!(e.contains(&format!("basert-engine-macos-arm64-{same}.tar.gz.sha256")), "{e}");
        let e = plan_with(&Request { version: Some(same.clone()), untested: true, ..Request::default() }, &mac(), &[], &releases).unwrap_err();
        assert!(e.contains("has no SHA-256 pinned"), "--untested is about the version, not the digest: {e}");
        let named = Request { version: Some(same.clone()), sha256: Some("12".into()), ..Request::default() };
        let p = &plan_with(&named, &mac(), &[], &releases).unwrap()[0];
        assert_eq!((p.assets[0].sha256.as_deref(), p.assets[0].size), (Some("12"), Some(10)));
        assert!(p.assets[0].url.starts_with("https://github.com/basecompute/baseRT/releases/download/"));
    }

    #[test]
    fn the_pins_are_whole() {
        let pins = pinned();
        assert!(!pins.is_empty());
        for (v, assets) in &pins {
            assert_eq!(assets.len(), 2, "{v}: an engine for each platform baseRT builds");
            for a in assets {
                assert_eq!(version_of(&a.name).as_deref(), Some(v.as_str()));
                assert!(a.sha256.as_ref().is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())), "{a:?}");
                assert!(a.size.is_some_and(|s| s > 0) && a.url == format!("https://github.com/basecompute/baseRT/releases/download/v{v}/{}", a.name));
            }
        }
    }

    #[test]
    fn with_no_release_of_its_version_the_error_says_what_there_is() {
        let releases = json!([release("0.0.1")]);
        let e = plan_with(&Request::default(), &mac(), &[], &releases).unwrap_err();
        assert!(e.contains("the newest is 0.0.1") && e.contains("--from"), "{e}");
    }

    #[test]
    fn a_bundle_names_its_version_and_other_machines_are_refused() {
        assert_eq!(version_of("basert-engine-linux-arm64-cuda-0.2.6.tar.gz").as_deref(), Some("0.2.6"));
        let linux = Host { os: "linux".into(), arch: "x64".into(), ..Host::default() };
        assert!(plan_with(&Request::default(), &linux, &[], &json!([])).unwrap_err().contains("baseRT publishes engines for"));
        let from = Request { from: Some("/x/basert-engine-macos-arm64-0.0.1.tar.gz".into()), untested: true, ..Request::default() };
        let p = &plan_with(&from, &mac(), &[], &Value::Null).unwrap()[0];
        assert_eq!((p.version.as_str(), p.install.as_str(), p.assets[0].sha256.as_deref()), ("0.0.1", "0.0.1-metal", None));
        let pins = [pin("0.0.1", "ab")];
        let p = &plan_with(&Request { from: Some("https://mirror/basert-engine-macos-arm64-0.0.1.tar.gz".into()), ..from }, &mac(), &pins, &Value::Null).unwrap()[0];
        assert_eq!(p.assets[0].sha256.as_deref(), Some("ab"), "a pinned engine is checked fetched from anywhere");
    }
}
