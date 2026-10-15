//! `superfluid runtime versions <id> [--json]`, and the INSTALLABLE column of `superfluid runtimes`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{json, Value};
use superfluid_adapter_kit::install::{Candidate, Host};

use crate::installs::{self, Install};
use crate::runtime_pick::RuntimeId;
use crate::runtimes::Catalog;

pub struct Listing {
    pub id: RuntimeId,
    pub host: String,
    /// What the adapter lists, in its order (newest first).
    pub candidates: Vec<Candidate>,
    pub installs: Vec<Install>,
    pub default: Option<String>,
    pub pinned: Option<String>,
}

/// What the adapter lists for this machine: the worker named by `SUPERFLUID_WORKER_<ID>` when one is, else the
/// linked adapter in process, else the worker `runtime install` would use; a worker answers from its recipe alone.
pub fn of(catalog: &Catalog, id: RuntimeId) -> Result<Listing, String> {
    let named = std::env::var_os(crate::runtimes::worker_env(id.base())).map(PathBuf::from);
    of_with(catalog, id, named.as_deref())
}

pub fn of_with(catalog: &Catalog, id: RuntimeId, named: Option<&Path>) -> Result<Listing, String> {
    let id = id.base();
    let (host, candidates) = match (named, crate::linked::get(id)) {
        (Some(worker), _) => asked(worker)?,
        (None, Some(rt)) => {
            let host = Host::detect();
            (host.describe(), rt.versions(&host))
        }
        (None, None) => asked(&installs::installer(catalog, id)?)?,
    };
    Ok(Listing {
        id,
        host,
        candidates,
        installs: installs::list(catalog, id),
        default: installs::default(catalog, id).map(|i| i.name),
        pinned: installs::pinned(catalog, id),
    })
}

fn asked(worker: &Path) -> Result<(String, Vec<Candidate>), String> {
    let mut cmd = Command::new(worker);
    crate::exec::without_secrets(&mut cmd).arg("versions").stdin(Stdio::null());
    let out = crate::exec::when_not_busy(|| cmd.output()).map_err(|e| format!("{} could not start: {e}", worker.display()))?;
    if !out.status.success() {
        return Err(format!("{} lists no versions ({}): an adapter from before the listing existed", worker.display(), out.status));
    }
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("{} versions printed no listing: {e}", worker.display()))?;
    let listed = v["versions"].as_array().ok_or_else(|| format!("{} versions printed no listing", worker.display()))?;
    let candidates = listed.iter().map(Candidate::from_json).collect::<Result<Vec<_>, _>>()?;
    Ok((v["describe"].as_str().unwrap_or_default().to_string(), candidates))
}

/// The versions that install here, newest first (`installable` in `superfluid runtimes --json`).
pub fn here(listing: &Listing) -> Vec<&str> {
    listing.candidates.iter().filter(|c| c.here.is_ok()).map(|c| c.version.as_str()).collect()
}

/// The INSTALLABLE cell: the newest listed version that installs here, marked when an install of it exists.
pub fn cell(listing: &Result<Listing, String>) -> String {
    let Ok(l) = listing else { return "-".into() };
    match l.candidates.iter().find(|c| c.here.is_ok()) {
        None => "none here".into(),
        Some(c) if l.installs.iter().any(|i| i.manifest.version == c.version) => format!("{} (installed)", c.version),
        Some(c) => c.version.clone(),
    }
}

/// A line of the listing: a listed version, or an installed one the adapter does not list.
struct Row<'a> {
    version: &'a str,
    listed: Option<&'a Candidate>,
    installed: Vec<&'a Install>,
}

impl Listing {
    fn rows(&self) -> Vec<Row<'_>> {
        let of_version = |v: &str| self.installs.iter().filter(|i| i.manifest.version == v).collect::<Vec<_>>();
        let mut rows: Vec<Row> =
            self.candidates.iter().map(|c| Row { version: &c.version, listed: Some(c), installed: of_version(&c.version) }).collect();
        for i in &self.installs {
            if !rows.iter().any(|r| r.version == i.manifest.version) {
                rows.push(Row { version: &i.manifest.version, listed: None, installed: of_version(&i.manifest.version) });
            }
        }
        rows
    }

    fn mark(&self, name: &str) -> String {
        if self.pinned.as_deref() == Some(name) {
            format!("{name} (pinned)")
        } else if self.default.as_deref() == Some(name) {
            format!("{name} (default)")
        } else {
            name.to_string()
        }
    }

    pub fn text(&self) -> String {
        let mut out = format!("{}: the versions this adapter installs on its own, newest first; this machine is {}\n", self.id, self.host);
        let rows = self.rows();
        if rows.is_empty() {
            out.push_str("  none: the adapter lists no version, and nothing is installed\n");
        } else {
            let mut table = vec![["VERSION", "INSTALLS HERE AS", "INSTALLED"].map(String::from).to_vec()];
            for r in &rows {
                let here = match r.listed.map(|c| (c, &c.here)) {
                    Some((c, Ok(p))) => match c.size() {
                        0 => p.install.clone(),
                        size => format!("{} ({:.1} MB)", p.install, size as f64 / 1e6),
                    },
                    Some((_, Err(why))) => format!("no: {why}"),
                    None => "(not a version this adapter lists)".to_string(),
                };
                let installed = match r.installed.as_slice() {
                    [] => "-".to_string(),
                    some => some.iter().map(|i| self.mark(&i.name)).collect::<Vec<_>>().join(", "),
                };
                table.push(vec![r.version.to_string(), here, installed]);
            }
            out.push_str(&crate::runtimes_report::columns(&table));
        }
        out.push_str(&format!(
            "`superfluid runtime install {} --version <v>` installs one; a version not listed installs with --untested\n",
            self.id
        ));
        out
    }

    pub fn json(&self) -> Value {
        let versions: Vec<Value> = self
            .rows()
            .iter()
            .map(|r| {
                let mut v = json!({
                    "version": r.version,
                    "listed": r.listed.is_some(),
                    "installed": r.installed.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(),
                });
                match r.listed.map(|c| (c, &c.here)) {
                    Some((c, Ok(p))) => {
                        v["installable"] = true.into();
                        v["install"] = json!(p.install);
                        v["backend"] = json!(p.backend);
                        v["size"] = json!(c.size());
                    }
                    Some((_, Err(why))) => v["installable"] = json!(why),
                    None => {}
                }
                v
            })
            .collect();
        json!({"id": self.id.name(), "host": self.host, "default": self.default, "pinned": self.pinned, "versions": versions})
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    pub(crate) fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("superfluid-versions-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A worker that answers `versions` with `listing` and anything else with `report`.
    pub(crate) fn fake_worker_listing(dir: &Path, name: &str, report: &str, listing: &str) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\nif [ \"$1\" = versions ]; then cat <<'EOF'\n{listing}\nEOF\nexit 0; fi\ncat <<'EOF'\n{report}\nEOF\nexit 0\n"))
            .unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    /// An install as `superfluid runtime install` leaves it, for `installs::list`.
    pub(crate) fn manifest(home: &Path, id: &str, name: &str, version: &str, backend: &str, tested: bool) -> PathBuf {
        let dir = home.join("runtimes").join(id).join(name);
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        let m = json!({"id": id, "install": name, "version": version, "backend": backend, "worker": format!("bin/superfluid-worker-{id}"), "tested": tested});
        std::fs::write(dir.join("runtime.json"), format!("{m:#}\n")).unwrap();
        dir
    }

    pub(crate) const LISTING: &str = r#"{"describe":"macOS arm64","versions":[
        {"version":"t2","error":"toyrt t2 publishes no build for macos-arm64"},
        {"version":"t1","plan":{"id":"toyrt","version":"t1","backend":"metal","install":"t1-metal","assets":[{"name":"a.tgz","url":"https://x/a.tgz","sha256":"ab","size":2500000}],"tested":true,"why":"w","extra":{"wheels_size":500000}}}]}"#;

    #[test]
    fn a_listing_says_which_versions_install_here_and_which_are_installed() {
        let root = scratch("listing");
        let (beside, libexec, home) = (root.join("bin"), root.join("libexec").join("superfluid"), root.join("home"));
        std::fs::create_dir_all(&beside).unwrap();
        std::fs::create_dir_all(&libexec).unwrap();
        fake_worker_listing(&libexec, "superfluid-worker-toyrt", "{}", LISTING);
        manifest(&home, "toyrt", "t1-metal", "t1", "metal", true);
        manifest(&home, "toyrt", "t9-cpu", "t9", "cpu", false);
        let cat = Catalog::new(home.clone(), Some(beside.clone()), beside.join("superfluid-workerd"));
        let id = RuntimeId::new("toyrt");

        let l = of(&cat, id).unwrap();
        assert_eq!((l.host.as_str(), l.default.as_deref(), l.pinned.as_deref()), ("macOS arm64", Some("t1-metal"), None));
        let text = l.text();
        // Columns are padded to the widest cell; the words and their order are what is checked.
        let squeezed: Vec<String> = text.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
        assert_eq!(squeezed[0], "toyrt: the versions this adapter installs on its own, newest first; this machine is macOS arm64");
        assert_eq!(squeezed[1], "VERSION INSTALLS HERE AS INSTALLED");
        assert_eq!(squeezed[2], "t2 no: toyrt t2 publishes no build for macos-arm64 -");
        assert_eq!(squeezed[3], "t1 t1-metal (3.0 MB) t1-metal (default)");
        assert_eq!(squeezed[4], "t9 (not a version this adapter lists) t9-cpu");
        assert_eq!(squeezed[5], "`superfluid runtime install toyrt --version <v>` installs one; a version not listed installs with --untested");
        assert_eq!(squeezed.len(), 6);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[1].starts_with("VERSION  INSTALLS HERE AS") && lines[2].starts_with("t2       no: "), "aligned: {text}");
        assert_eq!(lines[1].find("INSTALLED"), lines[3].find("t1-metal (default)"), "aligned: {text}");
        let j = l.json();
        assert_eq!(j["id"], "toyrt");
        assert_eq!(j["versions"][0]["installable"], "toyrt t2 publishes no build for macos-arm64");
        assert_eq!(j["versions"][0]["installed"], json!([]));
        assert_eq!((j["versions"][1]["installable"].as_bool(), j["versions"][1]["install"].as_str()), (Some(true), Some("t1-metal")));
        assert_eq!((j["versions"][1]["size"].as_u64(), j["versions"][1]["installed"][0].as_str()), (Some(3_000_000), Some("t1-metal")));
        assert_eq!((j["versions"][2]["listed"].as_bool(), j["versions"][2]["installable"].is_null()), (Some(false), true));
        assert_eq!(here(&l), ["t1"]);
        assert_eq!(cell(&Ok(l)), "t1 (installed)");

        std::fs::write(home.join("runtimes").join("toyrt").join("pinned"), "t9-cpu\n").unwrap();
        let l = of(&cat, id).unwrap();
        let text = l.text();
        let squeezed: Vec<String> = text.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
        assert_eq!(squeezed[3], "t1 t1-metal (3.0 MB) t1-metal", "{text}");
        assert_eq!(squeezed[4], "t9 (not a version this adapter lists) t9-cpu (pinned)", "{text}");
        assert_eq!(l.json()["pinned"], "t9-cpu");

        let bare = Listing { installs: vec![], default: None, pinned: None, candidates: vec![Candidate { version: "t2".into(), here: Err("no".into()) }], ..l };
        assert_eq!(cell(&Ok(bare)), "none here");

        crate::runtimes::tests::fake_worker(&libexec, "superfluid-worker-oldrt", "{}", 2);
        let Err(e) = of(&cat, RuntimeId::new("oldrt")) else { panic!("an old worker listed versions") };
        assert!(e.contains("lists no versions"), "{e}");
        assert_eq!(cell(&Err(e)), "-");
        let named = fake_worker_listing(&root.join("elsewhere"), "superfluid-worker-named", "{}", r#"{"describe":"here","versions":[{"version":"n1","error":"not here"}]}"#);
        for id in crate::linked::ids().into_iter().chain([RuntimeId::new("toyrt")]) {
            let l = of_with(&cat, id, Some(&named)).unwrap();
            assert_eq!((l.host.as_str(), l.candidates.len(), l.candidates[0].version.as_str()), ("here", 1, "n1"), "{}", id.name());
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
