//! How the MLX worker links its Python.

use std::path::Path;

fn main() {
    let python = pyo3_build_config::get();
    let version = python.version();
    println!("cargo:rustc-env=SUPERFLUID_MLX_PYTHON={}.{}", version.major, version.minor);
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if !macos || !python.shared() {
        return;
    }
    let Some(lib) = python.lib_name() else { return };
    let file = format!("lib{lib}.dylib");
    let listed = python.lib_dir().is_some_and(|dir| Path::new(dir).join(&file).exists());
    if !listed {
        let beside = python
            .executable()
            .and_then(|exe| Path::new(exe).canonicalize().ok())
            .and_then(|exe| Some(exe.parent()?.parent()?.join("lib")))
            .filter(|dir| dir.join(&file).exists());
        if let Some(dir) = beside {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
    println!("cargo:rustc-link-arg-bins=-Wl,-weak-l{lib}");
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path/../python/lib");
}
