use std::path::PathBuf;
use std::process::Command;

fn main() {
    let include = PathBuf::from(std::env::var("DEP_BASERT_INCLUDE").expect("baseRT-sys exports its include dir")).join("baseRT");
    println!("cargo:rerun-if-changed=layout_probe.c");
    println!("cargo:rerun-if-changed={}", include.join("baseRT_tick.h").display());
    println!("cargo::rustc-check-cfg=cfg(abi_c_layout)");
    cc::Build::new().file("layout_probe.c").include(&include).compile("abi_layout_probe");

    // The probe has to run to report the header's layout, so a cross build
    // leaves the check to `cargo test` on the target.
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let compiler = cc::Build::new().include(&include).define("ABI_LAYOUT_PROBE_MAIN", None).get_compiler();
    if std::env::var("HOST").ok() != std::env::var("TARGET").ok() || compiler.is_like_msvc() {
        return;
    }
    let exe = out.join("abi_layout_probe");
    let built = compiler
        .to_command()
        .arg("layout_probe.c")
        .arg("-o")
        .arg(&exe)
        .status()
        .expect("run the C compiler on layout_probe.c");
    assert!(built.success(), "superfluid: the ABI layout probe did not compile");
    let ran = Command::new(&exe).output().expect("run the ABI layout probe");
    assert!(ran.status.success(), "superfluid: the ABI layout probe failed");
    let stdout = String::from_utf8(ran.stdout).expect("probe output is UTF-8");
    let mut entries: Vec<(&str, &str)> =
        stdout.lines().map(|l| l.split_once('\t').expect("probe line is name<TAB>value")).collect();
    entries.sort();
    let mut table = String::from("pub const C_LAYOUT: &[(&str, u64)] = &[\n");
    for (name, value) in entries {
        table.push_str(&format!("    ({name:?}, {value}),\n"));
    }
    table.push_str("];\n");
    std::fs::write(out.join("c_layout.rs"), table).expect("write c_layout.rs");
    println!("cargo:rustc-cfg=abi_c_layout");
}
