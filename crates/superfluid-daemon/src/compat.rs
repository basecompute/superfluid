//! `superfluid compat [--json]`: what this build was built and tested for, per runtime.

use std::process::Command;

use serde_json::{json, Map, Value};

use crate::runtimes::Catalog;

/// Every adapter this build ships: a linked one in process, a shipped worker through `compat --json`.
pub fn json(catalog: &Catalog) -> Value {
    let mut runtimes: Map<String, Value> = Map::new();
    for rt in crate::linked::runtimes() {
        runtimes.insert(rt.info().id.into(), rt.compat());
    }
    for id in catalog.shipped_ids() {
        if runtimes.contains_key(id.name()) {
            continue;
        }
        let Some(worker) = catalog.shipped(id) else { continue };
        let mut cmd = Command::new(&worker);
        crate::exec::without_secrets(&mut cmd).args(["compat", "--json"]);
        let answer = match crate::exec::output_within(&mut cmd, crate::exec::ANSWER_WITHIN) {
            Ok(None) => json!({"error": format!("{} compat did not answer within {}s", worker.display(), crate::exec::ANSWER_WITHIN.as_secs())}),
            Ok(Some(out)) if out.status.success() => serde_json::from_slice::<Value>(&out.stdout)
                .unwrap_or_else(|e| json!({"error": format!("{}: compat is not JSON: {e}", worker.display())})),
            Ok(Some(out)) => json!({"error": format!("{} compat exited {}{}", worker.display(), out.status, crate::exec::stderr_tail(&out.stderr))}),
            Err(e) => json!({"error": format!("{}: {e}", worker.display())}),
        };
        runtimes.insert(id.name().into(), answer);
    }
    json!({"superfluid": env!("CARGO_PKG_VERSION"), "runtimes": runtimes})
}

/// The runtimes whose shipped adapter did not answer. `superfluid compat` exits 1 when there is one, so a release
/// whose adapter is broken fails where it writes compat.json instead of publishing an error for a host to read.
pub fn unanswered(c: &Value) -> Vec<&str> {
    c["runtimes"].as_object().map(|r| r.iter().filter(|(_, v)| v["error"].is_string()).map(|(id, _)| id.as_str()).collect()).unwrap_or_default()
}

/// `superfluid compat`, for a person.
pub fn text(c: &Value) -> String {
    let mut out = format!("{}\n", crate::cli::version());
    let Some(runtimes) = c["runtimes"].as_object() else { return out };
    for (id, r) in runtimes {
        if let Some(e) = r["error"].as_str() {
            out.push_str(&format!("{id}: {e}\n"));
            continue;
        }
        let line = match id.as_str() {
            "basert" => {
                let abi = r["abi"].as_u64().map(|n| format!("engine ABI {n}")).unwrap_or_else(|| "no engine ABI number (a header before 0.4.0)".into());
                let unnumbered = match r["unnumbered_minor"].as_str() {
                    Some(minor) => format!("one that states no number serves if it is baseRT {minor}.x"),
                    None => "one that states no number does not serve".to_string(),
                };
                let pinned: Vec<String> = r["pinned_engines"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|p| p["version"].as_str().map(str::to_string)).collect())
                    .unwrap_or_default();
                format!(
                    "built against baseRT {} headers, {abi}. A libbaseRT with the same ABI number serves, whatever its version; {unnumbered}. \
                     Installs on its own: {}",
                    r["basert_version"].as_str().unwrap_or("?"),
                    if pinned.is_empty() { "nothing".to_string() } else { pinned.join(", ") }
                )
            }
            _ => {
                let tested: Vec<String> =
                    r["tested"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
                let python = r["python"].as_str().map(|p| format!(" on Python {p}")).unwrap_or_default();
                format!("installs on its own{python}: {}", if tested.is_empty() { "nothing".to_string() } else { tested.join(", ") })
            }
        };
        out.push_str(&format!("{id}: {line}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("superfluid-compat-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn every_shipped_adapter_answers_for_its_runtime() {
        let root = scratch("shipped");
        let beside = root.join("bin");
        let libexec = root.join("libexec").join("superfluid");
        std::fs::create_dir_all(&beside).unwrap();
        std::fs::create_dir_all(&libexec).unwrap();
        let toy = libexec.join("superfluid-worker-toyrt");
        // The toy also tells whether a credential reached it.
        std::fs::write(&toy, "#!/bin/sh\n[ \"$1\" = compat ] || exit 2\necho \"{\\\"tested\\\":[\\\"t2\\\",\\\"t1\\\"],\\\"saw\\\":\\\"${SUPERFLUID_COMPAT_SENTINEL_TOKEN:-}\\\"}\"\n").unwrap();
        std::fs::set_permissions(&toy, std::fs::Permissions::from_mode(0o755)).unwrap();
        let broken = libexec.join("superfluid-worker-oldrt");
        std::fs::write(&broken, "#!/bin/sh\nexit 2\n").unwrap();
        std::fs::set_permissions(&broken, std::fs::Permissions::from_mode(0o755)).unwrap();
        let fails = libexec.join("superfluid-worker-failrt");
        std::fs::write(&fails, "#!/bin/sh\necho 'superfluid-worker-failrt: no recipe.json beside the adapter' >&2\nexit 1\n").unwrap();
        std::fs::set_permissions(&fails, std::fs::Permissions::from_mode(0o755)).unwrap();
        let cat = Catalog::new(root.join("home"), Some(beside.clone()), beside.join("superfluid-workerd"));
        std::env::set_var("SUPERFLUID_COMPAT_SENTINEL_TOKEN", "hunter2");
        let c = json(&cat);
        std::env::remove_var("SUPERFLUID_COMPAT_SENTINEL_TOKEN");
        assert_eq!(c["superfluid"], env!("CARGO_PKG_VERSION"));
        assert_eq!(c["runtimes"]["toyrt"]["tested"][0], "t2");
        assert_eq!(c["runtimes"]["toyrt"]["saw"], "", "a credential-named variable is not handed to the worker");
        for id in crate::linked::ids() {
            assert!(c["runtimes"][id.name()].is_object(), "the linked {} adapter answers in process: {c}", id.name());
        }
        assert!(c["runtimes"]["oldrt"]["error"].as_str().unwrap().contains("exited"), "{c}");
        assert_eq!(c["runtimes"]["failrt"]["error"].as_str().unwrap().rsplit(": ").next(), Some("no recipe.json beside the adapter"), "{c}");
        assert_eq!(unanswered(&c), ["failrt", "oldrt"], "{c}");
        #[cfg(feature = "basert")]
        {
            assert!(c["runtimes"]["basert"]["basert_version"].is_string(), "{c}");
            assert!(c["runtimes"]["basert"]["pinned_engines"].is_array(), "{c}");
        }
        let t = text(&c);
        assert!(t.contains("toyrt: installs on its own: t2, t1"), "{t}");
        assert!(t.contains("oldrt: ") && t.contains("exited"), "{t}");
        assert!(unanswered(&json!({"runtimes": {"toyrt": {"tested": []}}})).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
