//! Runtime installs.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use superfluid_adapter_kit::install::{backend_rank, unchecked, version_key, Manifest, Plan, Request};
use serde_json::Value;

use crate::runtime_pick::RuntimeId;
use crate::runtimes::{Catalog, Source, Worker};

#[derive(Debug, Clone, PartialEq)]
pub struct Install {
    pub name: String,
    pub dir: PathBuf,
    pub manifest: Manifest,
}

pub fn list(catalog: &Catalog, id: RuntimeId) -> Vec<Install> {
    let root = catalog.package_dir(id.base());
    let mut all: Vec<Install> = std::fs::read_dir(&root)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .filter_map(|e| {
                    let name = e.file_name().to_str()?.to_string();
                    if name.starts_with('.') || name == "current" {
                        return None;
                    }
                    let manifest = Manifest::read(&e.path()).ok()?;
                    Some(Install { name, dir: e.path(), manifest })
                })
                .collect()
        })
        .unwrap_or_default();
    all.sort_by(|a, b| {
        backend_rank(&a.manifest.backend)
            .cmp(&backend_rank(&b.manifest.backend))
            .then_with(|| version_key(&b.manifest.version).cmp(&version_key(&a.manifest.version)))
            .then_with(|| a.name.cmp(&b.name))
    });
    all
}

pub fn pinned(catalog: &Catalog, id: RuntimeId) -> Option<String> {
    let name = std::fs::read_to_string(catalog.package_dir(id.base()).join("pinned")).ok()?.trim().to_string();
    list(catalog, id).into_iter().any(|i| i.name == name).then_some(name)
}

pub fn default(catalog: &Catalog, id: RuntimeId) -> Option<Install> {
    let all = list(catalog, id);
    let pin = pinned(catalog, id);
    all.iter().find(|i| Some(&i.name) == pin.as_ref()).or(all.first()).cloned()
}

pub fn refresh_current(catalog: &Catalog, id: RuntimeId) -> Result<Option<String>, String> {
    let root = catalog.package_dir(id.base());
    let link = root.join("current");
    match default(catalog, id) {
        Some(d) => {
            let tmp = root.join(format!(".current-new-{}", unguessable()?));
            std::os::unix::fs::symlink(&d.name, &tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, &link).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                format!("switching current: {e}")
            })?;
            Ok(Some(d.name))
        }
        None => {
            let _ = std::fs::remove_file(&link);
            let _ = std::fs::remove_file(root.join("pinned"));
            Ok(None)
        }
    }
}

/// A name nobody else sharing the directory can claim first.
fn unguessable() -> Result<u64, String> {
    use std::io::Read;
    let mut buf = [0u8; 8];
    std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut buf)).map_err(|e| format!("/dev/urandom: {e}"))?;
    Ok(u64::from_le_bytes(buf))
}

pub fn pin(catalog: &Catalog, id: RuntimeId, name: &str) -> Result<(), String> {
    let all = list(catalog, id);
    if !all.iter().any(|i| i.name == name) {
        return Err(format!("{} has no install {name} (installed: {})", id.base(), names(&all)));
    }
    let path = catalog.package_dir(id.base()).join("pinned");
    std::fs::write(&path, format!("{name}\n")).map_err(|e| format!("{}: {e}", path.display()))?;
    refresh_current(catalog, id).map(|_| ())
}

pub fn unpin(catalog: &Catalog, id: RuntimeId) -> Result<Option<String>, String> {
    let _ = std::fs::remove_file(catalog.package_dir(id.base()).join("pinned"));
    refresh_current(catalog, id)
}

fn names(all: &[Install]) -> String {
    if all.is_empty() {
        "none".into()
    } else {
        all.iter().map(|i| i.name.as_str()).collect::<Vec<_>>().join(", ")
    }
}

pub fn installer(catalog: &Catalog, id: RuntimeId) -> Result<PathBuf, String> {
    let id = id.base();
    let name = crate::runtimes::worker_name(id);
    if let Some(p) = std::env::var_os(crate::runtimes::worker_env(id)) {
        return Ok(PathBuf::from(p));
    }
    let mut places: Vec<PathBuf> = Vec::new();
    places.extend(catalog.beside().map(|dir| dir.join(&name)));
    places.extend(catalog.shipped(id));
    if let Some(d) = default(catalog, id) {
        places.push(d.dir.join(&d.manifest.worker));
    }
    places.into_iter().find(|p| p.is_file()).ok_or_else(|| {
        let shipped = catalog.shipped_ids();
        if !shipped.is_empty() {
            return format!(
                "this superfluid ships no adapter to install the {id} runtime with (it ships: {})",
                shipped.iter().map(|i| i.name()).collect::<Vec<_>>().join(", ")
            );
        }
        format!(
            "no {name} to install the {id} runtime with (looked beside superfluid, in a release's libexec/superfluid and in its \
             installs; a source build makes it with `cargo build -p superfluid-adapter-{id}`)"
        )
    })
}

#[derive(Debug, Clone)]
pub struct Plans {
    pub host: String,
    pub plans: Vec<Plan>,
}

pub fn plan(installer: &Path, req: &Request) -> Result<Plans, String> {
    let mut cmd = Command::new(installer);
    crate::exec::without_secrets(&mut cmd).arg("plan").args(req.to_args()).stdin(Stdio::null());
    let out = crate::exec::when_not_busy(|| cmd.output()).map_err(|e| format!("{} could not start: {e}", installer.display()))?;
    if !out.status.success() {
        return Err(last_lines(&out.stderr).unwrap_or_else(|| format!("{} plan failed ({})", installer.display(), out.status)));
    }
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("{} plan printed no plan: {e}", installer.display()))?;
    let plans = v["plans"].as_array().map(|a| a.iter().map(Plan::from_json).collect::<Result<Vec<_>, _>>()).transpose()?.unwrap_or_default();
    if plans.is_empty() {
        return Err(format!("{} has nothing to install here", installer.display()));
    }
    Ok(Plans { host: v["describe"].as_str().unwrap_or_default().to_string(), plans })
}

#[derive(Debug, Clone)]
pub struct Installed {
    pub install: Install,
    pub version: String,
    pub device: Option<String>,
    pub fell_back: Vec<(String, String)>,
    pub replaced: bool,
    pub default: Option<String>,
}

pub fn install(catalog: &Catalog, id: RuntimeId, installer: &Path, plans: &[Plan]) -> Result<Installed, String> {
    let id = id.base();
    let root = catalog.package_dir(id);
    std::fs::create_dir_all(&root).map_err(|e| format!("{}: {e}", root.display()))?;
    if let Some(plan) = plans.iter().find(|p| !crate::runtime_pick::valid_install(&p.install)) {
        return Err(format!(
            "installing {id}: '{}' is not a name an install can have (letters, digits, '.', '-' and '_', not starting with '.')",
            plan.install
        ));
    }
    if let Some(why) = plans.iter().flat_map(|p| &p.assets).find_map(unchecked) {
        return Err(format!("installing {id}: {why}"));
    }
    let _one_at_a_time = crate::pulls::Lock::take(&root, ".install", &format!("install of {id}"))?;
    let mut left: Vec<PathBuf> = std::fs::read_dir(&root)
        .map(|entries| entries.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    left.retain(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(".staging-") || n.starts_with(".old-")));
    left.sort();
    let mut put_back = false;
    for stale in left {
        match set_aside_unreplaced(id, &root, &stale) {
            Some(dest) => {
                std::fs::rename(&stale, &dest).map_err(|e| {
                    format!(
                        "{}: an interrupted install set this copy of {} aside and it cannot be put back: {e}",
                        stale.display(),
                        dest.display()
                    )
                })?;
                put_back = true;
            }
            None => {
                let _ = std::fs::remove_dir_all(&stale);
            }
        }
    }
    if put_back {
        refresh_current(catalog, id)?;
    }
    let mut fell_back = Vec::new();
    for plan in plans {
        let staging = root.join(format!(".staging-{}", unguessable()?));
        std::fs::create_dir(&staging).map_err(|e| format!("{}: {e}", staging.display()))?;
        let staged = stage(id, installer, plan, &staging);
        let checked = staged.and_then(|manifest| verify(id, plan, &staging).map(|ok| (manifest, ok)));
        match checked {
            Ok((manifest, Ok((version, device)))) => {
                let dest = root.join(&plan.install);
                let replaced = dest.exists();
                let aside = root.join(format!(".old-{}-{}", plan.install, unguessable()?));
                if replaced {
                    if let Err(e) = std::fs::rename(&dest, &aside) {
                        let _ = std::fs::remove_dir_all(&staging);
                        return Err(format!("{}: {e}", dest.display()));
                    }
                }
                if let Err(e) = std::fs::rename(&staging, &dest) {
                    if replaced {
                        let _ = std::fs::rename(&aside, &dest);
                    }
                    let _ = std::fs::remove_dir_all(&staging);
                    return Err(format!("{}: {e}", dest.display()));
                }
                if replaced {
                    let _ = std::fs::remove_dir_all(&aside);
                }
                let default = refresh_current(catalog, id)?;
                return Ok(Installed {
                    install: Install { name: plan.install.clone(), dir: dest, manifest },
                    version,
                    device,
                    fell_back,
                    replaced,
                    default,
                });
            }
            Ok((_, Err(why))) => {
                let _ = std::fs::remove_dir_all(&staging);
                fell_back.push((plan.install.clone(), why));
            }
            Err(e) => {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(format!("installing {id} {}: {e}", plan.install));
            }
        }
    }
    Err(format!(
        "no {id} build runs here: {}",
        fell_back.iter().map(|(p, why)| format!("{p}: {why}")).collect::<Vec<_>>().join("; ")
    ))
}

fn set_aside_unreplaced(id: RuntimeId, root: &Path, aside: &Path) -> Option<PathBuf> {
    let name = aside.file_name()?.to_str()?.strip_prefix(".old-")?;
    let (install, pid) = name.rsplit_once('-')?;
    if pid.is_empty() || !pid.bytes().all(|b| b.is_ascii_digit()) || !crate::runtime_pick::valid_install(install) {
        return None;
    }
    let manifest = Manifest::read(aside).ok()?;
    if manifest.id != id.name() || manifest.install != install {
        return None;
    }
    let dest = root.join(install);
    dest.symlink_metadata().is_err().then_some(dest)
}

fn stage(id: RuntimeId, installer: &Path, plan: &Plan, staging: &Path) -> Result<Manifest, String> {
    let mut cmd = Command::new(installer);
    crate::exec::without_secrets(&mut cmd).arg("install").arg("--into").arg(staging).arg("--plan").arg(plan.to_json().to_string()).stdin(Stdio::null()).stderr(Stdio::inherit());
    let out = crate::exec::when_not_busy(|| cmd.output()).map_err(|e| format!("{} could not start: {e}", installer.display()))?;
    if !out.status.success() {
        return Err(format!("{} install failed ({})", installer.display(), out.status));
    }
    let manifest = Manifest::read(staging)?;
    if manifest.id != id.name() || manifest.install != plan.install {
        return Err(format!("the install made {} {}, the plan says {id} {}", manifest.id, manifest.install, plan.install));
    }
    Ok(manifest)
}

fn verify(id: RuntimeId, plan: &Plan, staging: &Path) -> Result<Result<(String, Option<String>), String>, String> {
    let worker = Worker { bin: staging.join("bin").join(crate::runtimes::worker_name(id)), prefix: Vec::new(), source: Source::Installed };
    if !worker.bin.is_file() {
        return Err(format!("the install has no {}", worker.bin.display()));
    }
    let runtime = match crate::runtimes::check_worker(id, &worker) {
        Ok(r) => r,
        Err(why) => return Ok(Err(why)),
    };
    let family = plan.backend.split(['-', '_']).next().unwrap_or(&plan.backend).to_ascii_lowercase();
    let device = runtime.devices.iter().find(|d| family == "cpu" || d.backend.eq_ignore_ascii_case(&family));
    match device {
        Some(d) => Ok(Ok((runtime.version.unwrap_or_default(), Some(d.summary())))),
        None => Ok(Err(format!(
            "its check finds no {} device here (it sees: {})",
            plan.backend,
            if runtime.devices.is_empty() {
                "none".to_string()
            } else {
                runtime.devices.iter().map(|d| d.backend.clone()).collect::<Vec<_>>().join(", ")
            }
        ))),
    }
}

pub fn not_an_update(current: &str, planned: &str) -> Option<String> {
    if current == "custom" {
        return Some("it is a build of your own, not one of the adapter's".to_string());
    }
    (version_key(current) > version_key(planned))
        .then(|| format!("it is {current}, newer than {planned}, the newest build this adapter is tested with"))
}

pub fn remove(catalog: &Catalog, id: RuntimeId, name: Option<&str>) -> Result<Vec<String>, String> {
    let root = catalog.package_dir(id.base());
    let all = list(catalog, id);
    match name {
        Some(n) => {
            let Some(i) = all.iter().find(|i| i.name == n) else {
                return Err(format!("{} has no install {n} (installed: {})", id.base(), names(&all)));
            };
            std::fs::remove_dir_all(&i.dir).map_err(|e| format!("{}: {e}", i.dir.display()))?;
            if pinned(catalog, id).is_none() {
                let _ = std::fs::remove_file(root.join("pinned"));
            }
            refresh_current(catalog, id)?;
            Ok(vec![n.to_string()])
        }
        None => {
            if all.is_empty() {
                return Err(format!("{} is not installed", id.base()));
            }
            std::fs::remove_dir_all(&root).map_err(|e| format!("{}: {e}", root.display()))?;
            Ok(all.into_iter().map(|i| i.name).collect())
        }
    }
}

fn last_lines(stderr: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(stderr);
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).rev().take(3).collect();
    (!lines.is_empty()).then(|| lines.into_iter().rev().collect::<Vec<_>>().join(" / "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    const V: u16 = superfluid_agent::client::PROTO_VERSION;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("superfluid-installs-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn plan_for(id: &str, version: &str, backend: &str) -> Plan {
        Plan {
            id: id.into(),
            version: version.into(),
            backend: backend.into(),
            install: format!("{version}-{backend}"),
            assets: Vec::new(),
            tested: true,
            why: "a test".into(),
            extra: Value::Null,
        }
    }

    fn installer(dir: &Path, id: &str, devices: &str, broken: &str) -> PathBuf {
        let p = dir.join(format!("superfluid-worker-{id}"));
        let script = format!(
            r#"#!/bin/sh
# plan: nothing to say; install --into DIR --plan JSON
[ "$1" = install ] || exit 2
dir="$3"; plan="$5"
echo "$dir" >> "$0.staged"
backend=$(printf '%s' "$plan" | sed -E 's/.*"backend":"([^"]*)".*/\1/')
install=$(printf '%s' "$plan" | sed -E 's/.*"install":"([^"]*)".*/\1/')
version=$(printf '%s' "$plan" | sed -E 's/.*"version":"([^"]*)".*/\1/')
case " {broken} " in *" $backend "*) echo "cannot fetch $backend" >&2; exit 1;; esac
devices=$(printf '%s' '{devices}' | sed -E "s/.*\"$backend\":\[([^]]*)\].*/\1/")
mkdir -p "$dir/bin"
cat > "$dir/bin/superfluid-worker-{id}" <<EOF
#!/bin/sh
echo '{{"runtime":{{"id":"{id}","version":"$version"}},"link_w":[{V}],"available":true,"devices":[$devices]}}'
EOF
chmod +x "$dir/bin/superfluid-worker-{id}"
echo '{{"id":"{id}","install":"'$install'","version":"'$version'","backend":"'$backend'","link_w":[{V}],"platforms":[],"worker":"bin/superfluid-worker-{id}","sources":[],"tested":true}}' > "$dir/runtime.json"
cat "$dir/runtime.json"
"#
        );
        std::fs::write(&p, script).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    fn device(backend: &str) -> String {
        format!(r#"{{"backend":"{backend}","name":"x","memory":1}}"#)
    }

    #[test]
    fn installs_live_side_by_side_and_the_best_ranked_is_the_default_until_one_is_pinned() {
        let dir = scratch("side");
        let home = dir.join("home");
        let cat = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let id = RuntimeId::new("toyrt");
        let devices = format!(r#"{{"cpu":[{}],"vulkan":[{},{}]}}"#, device("CPU"), device("Vulkan"), device("CPU"));
        let w = installer(&dir, "toyrt", &devices, "");
        let cpu = install(&cat, id, &w, &[plan_for("toyrt", "b2", "cpu")]).unwrap();
        assert_eq!((cpu.install.name.as_str(), cpu.default.as_deref(), cpu.replaced), ("b2-cpu", Some("b2-cpu"), false));
        let vk = install(&cat, id, &w, &[plan_for("toyrt", "b1", "vulkan")]).unwrap();
        assert_eq!(vk.default.as_deref(), Some("b1-vulkan"), "a GPU backend outranks the CPU, older or not");
        assert_eq!(std::fs::read_link(home.join("runtimes/toyrt/current")).unwrap(), Path::new("b1-vulkan"));
        assert_eq!(list(&cat, id).iter().map(|i| i.name.as_str()).collect::<Vec<_>>(), ["b1-vulkan", "b2-cpu"]);

        pin(&cat, id, "b2-cpu").unwrap();
        assert_eq!(default(&cat, id).unwrap().name, "b2-cpu");
        assert_eq!(install(&cat, id, &w, &[plan_for("toyrt", "b3", "vulkan")]).unwrap().default.as_deref(), Some("b2-cpu"), "a pin holds");
        assert_eq!(unpin(&cat, id).unwrap().as_deref(), Some("b3-vulkan"));
        assert!(pin(&cat, id, "b9-cuda").unwrap_err().contains("has no install b9-cuda"));

        assert_eq!(cat.require(id).unwrap().version.as_deref(), Some("b3"));
        let at = RuntimeId::new("toyrt@b2-cpu");
        assert_eq!(cat.require(at).unwrap().version.as_deref(), Some("b2"));

        assert_eq!(remove(&cat, id, Some("b3-vulkan")).unwrap(), ["b3-vulkan"]);
        assert_eq!(default(&cat, id).unwrap().name, "b1-vulkan");
        assert_eq!(remove(&cat, id, None).unwrap().len(), 2);
        assert!(!home.join("runtimes/toyrt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_build_that_finds_no_device_falls_back_and_a_failed_fetch_stops() {
        let dir = scratch("fallback");
        let home = dir.join("home");
        let cat = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let id = RuntimeId::new("toyrt");
        let devices = format!(r#"{{"cuda-13.4":[{}],"vulkan":[{}],"cpu":[{}]}}"#, device("CPU"), device("Vulkan"), device("CPU"));
        let w = installer(&dir, "toyrt", &devices, "");
        let plans = [plan_for("toyrt", "b1", "cuda-13.4"), plan_for("toyrt", "b1", "vulkan"), plan_for("toyrt", "b1", "cpu")];
        let got = install(&cat, id, &w, &plans).unwrap();
        assert_eq!(got.install.name, "b1-vulkan");
        assert_eq!(got.fell_back.len(), 1);
        assert!(got.fell_back[0].1.contains("finds no cuda-13.4 device here (it sees: CPU)"), "{:?}", got.fell_back);

        let broken = installer(&dir, "toyrt", &devices, "cuda-13.4");
        let e = install(&cat, id, &broken, &plans).unwrap_err();
        assert!(e.contains("installing toyrt b1-cuda-13.4") && e.contains("install failed"), "{e}");
        assert_eq!(list(&cat, id).len(), 1, "what was installed stays");
        let left: Vec<String> =
            std::fs::read_dir(home.join("runtimes/toyrt")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        assert!(left.iter().all(|n| !n.starts_with(".staging")), "no staging left behind: {left:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn an_install_past_the_tested_builds_is_not_updated_back() {
        assert_eq!(not_an_update("b11284", "b11300"), None);
        assert_eq!(not_an_update("0.2.6", "0.3.0"), None);
        assert!(not_an_update("b12000", "b11284").unwrap().contains("newer than b11284"));
        assert!(not_an_update("custom", "b11284").unwrap().contains("a build of your own"));
        assert_eq!(not_an_update("b11284", "b11284"), None);
    }

    #[test]
    fn an_install_is_named_by_a_name_an_install_can_have() {
        let dir = scratch("names");
        let home = dir.join("home");
        let cat = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let id = RuntimeId::new("toyrt");
        let w = installer(&dir, "toyrt", &format!(r#"{{"cpu":[{}]}}"#, device("CPU")), "");
        for bad in ["../x-cpu", "a/b", ".hidden", "b1 cpu", ""] {
            let mut plan = plan_for("toyrt", "b1", "cpu");
            plan.install = bad.to_string();
            let e = install(&cat, id, &w, &[plan]).unwrap_err();
            assert!(e.contains("is not a name an install can have"), "{bad}: {e}");
        }
        assert!(!home.join("runtimes/x-cpu").exists());
        assert!(list(&cat, id).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_install_clears_what_an_interrupted_one_left() {
        let dir = scratch("stale");
        let home = dir.join("home");
        let cat = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let id = RuntimeId::new("toyrt");
        let root = home.join("runtimes/toyrt");
        for stale in [".staging-4242-0", ".old-b1-cpu-4242"] {
            std::fs::create_dir_all(root.join(stale).join("lib")).unwrap();
        }
        let w = installer(&dir, "toyrt", &format!(r#"{{"cpu":[{}]}}"#, device("CPU")), "");
        install(&cat, id, &w, &[plan_for("toyrt", "b1", "cpu")]).unwrap();
        let left: Vec<String> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(".staging-") || n.starts_with(".old-"))
            .collect();
        assert!(left.is_empty(), "{left:?}");
        let again = install(&cat, id, &w, &[plan_for("toyrt", "b1", "cpu")]).unwrap();
        assert!(again.replaced);
        assert!(root.join("b1-cpu/bin/superfluid-worker-toyrt").is_file());
        assert!(!std::fs::read_dir(&root).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with(".old-")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_install_set_aside_by_an_interrupted_replacement_is_put_back() {
        let dir = scratch("aside");
        let home = dir.join("home");
        let cat = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let id = RuntimeId::new("toyrt");
        let root = home.join("runtimes/toyrt");
        let devices = format!(r#"{{"cpu":[{}]}}"#, device("CPU"));
        let plans = [plan_for("toyrt", "b1", "cpu")];
        let w = installer(&dir, "toyrt", &devices, "");
        install(&cat, id, &w, &plans).unwrap();
        std::fs::rename(root.join("b1-cpu"), root.join(".old-b1-cpu-4242")).unwrap();
        assert!(list(&cat, id).is_empty());
        assert_eq!(refresh_current(&cat, id).unwrap(), None);
        let broken = installer(&dir, "toyrt", &devices, "cpu");
        let e = install(&cat, id, &broken, &plans).unwrap_err();
        assert!(e.contains("installing toyrt b1-cpu") && e.contains("install failed"), "{e}");
        assert!(root.join("b1-cpu/bin/superfluid-worker-toyrt").is_file(), "the install that was there is there again");
        assert_eq!(list(&cat, id).iter().map(|i| i.name.as_str()).collect::<Vec<_>>(), ["b1-cpu"]);
        assert_eq!(std::fs::read_link(root.join("current")).unwrap(), Path::new("b1-cpu"), "and it is the default again");
        assert!(!root.join(".old-b1-cpu-4242").exists());

        let copied = Command::new("cp").arg("-R").arg(root.join("b1-cpu")).arg(root.join(".old-b1-cpu-4243")).status().unwrap();
        assert!(copied.success());
        install(&cat, id, &broken, &plans).unwrap_err();
        assert!(!root.join(".old-b1-cpu-4243").exists());
        assert!(root.join("b1-cpu/bin/superfluid-worker-toyrt").is_file());
        std::fs::remove_dir_all(root.join("b1-cpu")).unwrap();
        std::fs::create_dir_all(root.join(".old-b1-cpu-4244/lib")).unwrap();
        install(&cat, id, &broken, &plans).unwrap_err();
        assert!(!root.join("b1-cpu").exists() && !root.join(".old-b1-cpu-4244").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_archive_from_elsewhere_with_no_digest_is_not_installed() {
        let dir = scratch("unchecked");
        let home = dir.join("home");
        let cat = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let id = RuntimeId::new("toyrt");
        let w = installer(&dir, "toyrt", &format!(r#"{{"cpu":[{}]}}"#, device("CPU")), "");
        let asset = |url: &str, sha256: Option<&str>| superfluid_adapter_kit::install::Asset {
            name: "a.tar.gz".into(),
            url: url.into(),
            sha256: sha256.map(str::to_string),
            size: None,
        };
        let mut plan = plan_for("toyrt", "b1", "cpu");
        plan.assets = vec![asset("https://mirror.example/a.tar.gz", None)];
        let e = install(&cat, id, &w, &[plan.clone()]).unwrap_err();
        assert!(e.contains("installing toyrt: https://mirror.example/a.tar.gz: no SHA-256"), "{e}");
        assert!(!w.with_extension("staged").exists() && list(&cat, id).is_empty(), "nothing was staged");
        plan.assets = vec![asset("https://mirror.example/a.tar.gz", Some("ab")), asset("/local/a.tar.gz", None)];
        install(&cat, id, &w, &[plan]).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn staging_names_are_not_predictable_and_a_planted_one_is_not_reused() {
        let dir = scratch("names-unguessable");
        let home = dir.join("home");
        let cat = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let id = RuntimeId::new("toyrt");
        let root = home.join("runtimes/toyrt");
        let w = installer(&dir, "toyrt", &format!(r#"{{"cpu":[{}]}}"#, device("CPU")), "");
        install(&cat, id, &w, &[plan_for("toyrt", "b1", "cpu")]).unwrap();
        install(&cat, id, &w, &[plan_for("toyrt", "b1", "cpu")]).unwrap();
        let staged: Vec<String> = std::fs::read_to_string(w.with_extension("staged")).unwrap().lines().map(str::to_string).collect();
        assert_eq!(staged.len(), 2);
        assert_ne!(staged[0], staged[1], "each install stages somewhere new");
        let by_pid = root.join(format!(".staging-{}-0", std::process::id())).display().to_string();
        assert!(staged.iter().all(|s| *s != by_pid && s.starts_with(&root.join(".staging-").display().to_string())), "{staged:?}");

        std::fs::create_dir_all(root.join(".current-new/x")).unwrap();
        assert_eq!(refresh_current(&cat, id).unwrap().as_deref(), Some("b1-cpu"), "a name someone else made is left alone");
        assert_eq!(std::fs::read_link(root.join("current")).unwrap(), Path::new("b1-cpu"));
        assert!(root.join(".current-new/x").is_dir());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

