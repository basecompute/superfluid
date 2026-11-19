//! The Microsoft Visual C++ runtime beside llama.cpp on Windows.
//!
//! llama.cpp's Windows builds are linked against the runtime's shared libraries and carry none.
//! Windows has them only once something has installed the redistributable, which takes an
//! administrator. A superfluid release carries the three (Microsoft lets them ship beside the
//! program that needs them), and an install of llama.cpp is given what the system lacks.

use std::path::{Path, PathBuf};

/// What `llama.dll` and the libraries beside it are linked against.
pub const LIBRARIES: [&str; 3] = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"];

/// Where the redistributable installs itself, for whoever can run it.
pub const REDISTRIBUTABLE: &str = "https://aka.ms/vs/17/release/vc_redist.x64.exe";

/// Puts in `lib`, beside `llama.dll`, each runtime library that neither it nor `system`
/// (System32) holds, from the first of `shipped` that has it. The ones still missing are returned:
/// a source build ships none.
pub fn place(lib: &Path, system: &Path, shipped: &[PathBuf]) -> std::io::Result<Vec<&'static str>> {
    let mut missing = Vec::new();
    for name in LIBRARIES {
        if lib.join(name).is_file() || system.join(name).is_file() {
            continue;
        }
        match shipped.iter().map(|d| d.join(name)).find(|p| p.is_file()) {
            Some(from) => {
                std::fs::copy(&from, lib.join(name))?;
            }
            None => missing.push(name),
        }
    }
    Ok(missing)
}

/// The runtime libraries `lib` would be loaded without: in neither it nor `system`.
pub fn missing(lib: &Path, system: &Path) -> Vec<&'static str> {
    LIBRARIES.into_iter().filter(|name| !lib.join(name).is_file() && !system.join(name).is_file()).collect()
}

/// What to say when llama.cpp is to be loaded without `missing`.
pub fn advice(missing: &[&str]) -> String {
    format!(
        "llama.cpp needs the Microsoft Visual C++ runtime, and this machine has no {}: a superfluid release brings it \
         (scripts/install.ps1), or install the redistributable from {REDISTRIBUTABLE}",
        missing.join(", ")
    )
}

/// System32, where the redistributable puts the runtime.
pub fn system_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into())).join("System32")
}

/// Where a release keeps its copies, from the directory of the worker that installs:
/// `libexec\superfluid\vcruntime`, which is beside a shipped adapter and across from `bin\`.
pub fn shipped(exe_dir: &Path) -> Vec<PathBuf> {
    vec![exe_dir.join("vcruntime"), exe_dir.join("..").join("libexec").join("superfluid").join("vcruntime")]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sf-vcruntime-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn an_install_is_given_only_what_the_system_lacks() {
        let (lib, system, release) = (dir("lib"), dir("system"), dir("release"));
        let shipped = shipped(&release.join("libexec").join("superfluid"));
        std::fs::create_dir_all(&shipped[0]).unwrap();
        for name in LIBRARIES {
            std::fs::write(shipped[0].join(name), name).unwrap();
        }
        // The system has one of the three: that one is the system's to keep current.
        std::fs::write(system.join("vcruntime140.dll"), b"system").unwrap();
        assert_eq!(missing(&lib, &system), ["msvcp140.dll", "vcruntime140_1.dll"]);
        assert!(place(&lib, &system, &shipped).unwrap().is_empty());
        assert_eq!(std::fs::read(lib.join("msvcp140.dll")).unwrap(), b"msvcp140.dll");
        assert!(lib.join("vcruntime140_1.dll").is_file());
        assert!(!lib.join("vcruntime140.dll").exists(), "the system's own is not shadowed");
        assert!(missing(&lib, &system).is_empty());
        // The daemon's own worker runs from bin\, across from libexec\.
        std::fs::create_dir_all(release.join("bin")).unwrap();
        let from_bin = super::shipped(&release.join("bin"));
        assert!(from_bin.iter().any(|d| d.join("msvcp140.dll").is_file()), "{from_bin:?}");
        for d in [lib, system, release] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn a_build_that_ships_none_says_what_is_missing_and_where_it_comes_from() {
        let (lib, system) = (dir("bare-lib"), dir("bare-system"));
        let missing = place(&lib, &system, &[lib.join("nowhere")]).unwrap();
        assert_eq!(missing, LIBRARIES);
        let said = advice(&missing);
        assert!(said.contains("msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll") && said.contains(REDISTRIBUTABLE), "{said}");
        for d in [lib, system] {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}
