use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.join("..").join("..");
    let header = PathBuf::from(env::var("DEP_BASERT_INCLUDE").expect("baseRT-sys exports its include dir"))
        .join("baseRT")
        .join("baseRT.h");
    println!("cargo:rerun-if-changed={}", header.display());
    let text = std::fs::read_to_string(&header).unwrap_or_else(|e| panic!("{}: {e}", header.display()));
    let field = |name: &str| -> String {
        text.lines()
            .find_map(|l| l.trim().strip_prefix(&format!("#define {name} ")).map(|v| v.trim().to_string()))
            .unwrap_or_else(|| panic!("{} defines no {name}", header.display()))
    };
    let version = format!(
        "{}.{}.{}",
        field("BASERT_VERSION_MAJOR"),
        field("BASERT_VERSION_MINOR"),
        field("BASERT_VERSION_PATCH")
    );
    println!("cargo:rustc-env=SUPERFLUID_BASERT_HEADER_VERSION={version}");

    let lib_dir = env::var("BASERT_LIB_DIR")
        .or_else(|_| env::var("BASERT_LIB_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("build"));
    let lib_dir = lib_dir.canonicalize().unwrap_or(lib_dir).display().to_string();
    println!("cargo:rustc-env=SUPERFLUID_BASERT_LIB_DIR={lib_dir}");
    println!("cargo:rerun-if-env-changed=BASERT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=BASERT_LIB_PATH");
}
