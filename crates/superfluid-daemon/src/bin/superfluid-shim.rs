//! The `superfluid` on PATH on Windows.
//!
//! Releases sit side by side under `versions\<v>`, and `versions\current` says which one runs.
//! On Unix the commands on PATH are symbolic links through `current`. An ordinary Windows user
//! may make no symbolic link, so there each command on PATH is a copy of this program under the
//! command's name (`superfluid.exe`, `superfluid-workerd.exe`, `superfluid-noded.exe`). It reads
//! which version is current, runs that version's program of its own name with its own
//! arguments, console and streams, and exits with its status.
//!
//! `superfluid self install` places the copies; `self update` and `self use` change `current`,
//! and what was started before keeps the version it started with.
//!
//! It uses nothing of the daemon's: it must stay small, and one release's copy runs every
//! later release.

use std::path::{Path, PathBuf};

/// The file beside this program that holds the path of the directory versions are kept in.
const VERSIONS_FILE: &str = "superfluid-versions";
/// What `superfluid self` reads in a launcher on PATH to tell whether a release carries a newer one: the marker and
/// four digits, raised when this program changes what it does. Two builds of one revision differ in their bytes
/// (a timestamp, a build id) and not in what they do, so a launcher is replaced by revision, not by content.
#[used]
static REVISION: [u8; 29] = *b"superfluid-shim-revision:0001";
const INSTALL: &str = "irm https://basecompute.co/sf/install.ps1 | iex";

/// The program `launcher` stands for: `<versions>\<current>\bin\<the launcher's own name>`.
fn target(launcher: &Path) -> Result<PathBuf, String> {
    let (Some(dir), Some(name)) = (launcher.parent(), launcher.file_name()) else {
        return Err(format!("{} has no directory to look beside it in", launcher.display()));
    };
    let noted = dir.join(VERSIONS_FILE);
    let versions = std::fs::read_to_string(&noted).map_err(|e| format!("{}: {e}", noted.display()))?;
    let versions = PathBuf::from(versions.trim());
    let current = versions.join("current");
    let version = std::fs::read_to_string(&current).map_err(|e| format!("{}: {e}", current.display()))?;
    let version = version.trim();
    // One directory beside `current`, never a path out of the versions.
    let one_name = !version.is_empty() && version != "." && version != ".." && !version.contains(['/', '\\', ':']);
    if !one_name {
        return Err(format!("{} does not name a version", current.display()));
    }
    let program = versions.join(version).join("bin").join(name);
    match program.is_file() {
        true => Ok(program),
        false => Err(format!("{} names {version}, which has no {}", current.display(), Path::new(name).display())),
    }
}

#[cfg(windows)]
fn run(program: &Path) -> i32 {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{BOOL, HANDLE};
    use windows_sys::Win32::System::Console::{SetConsoleCtrlHandler, CTRL_BREAK_EVENT, CTRL_C_EVENT};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK,
    };

    /// Ctrl+C and Ctrl+Break reach every program on the console: the one this runs answers
    /// them, and this waits for it. Closing the console ends both.
    unsafe extern "system" fn left_to_the_program(kind: u32) -> BOOL {
        BOOL::from(kind == CTRL_C_EVENT || kind == CTRL_BREAK_EVENT)
    }
    // SAFETY: the handler is a function that lives as long as the process and touches nothing.
    unsafe {
        SetConsoleCtrlHandler(Some(left_to_the_program), 1);
    }
    let mut child = match std::process::Command::new(program).args(std::env::args_os().skip(1)).spawn() {
        Ok(child) => child,
        Err(e) => {
            eprintln!("superfluid: {}: {e}", program.display());
            return 1;
        }
    };
    // Ending this ends the program: whoever ends the command they started means the program.
    // What the program itself starts is left out, so a server `launch` leaves running outlives
    // the command that started it.
    // SAFETY: the job is this process's own and stays open until it exits, which is what ends
    // the program; `limits` is plain data, zeroed and then set, and its size is what is passed;
    // the child's handle is open while `child` lives.
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if !job.is_null() {
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK;
            let size = std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
            if SetInformationJobObject(job, JobObjectExtendedLimitInformation, std::ptr::addr_of!(limits).cast(), size) != 0 {
                AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE);
            }
        }
    }
    match child.wait() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("superfluid: waiting for {}: {e}", program.display());
            1
        }
    }
}

/// Elsewhere the commands on PATH are symbolic links, and nothing places this program.
#[cfg(not(windows))]
fn run(program: &Path) -> i32 {
    match std::process::Command::new(program).args(std::env::args_os().skip(1)).status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("superfluid: {}: {e}", program.display());
            1
        }
    }
}

fn main() {
    // Read, so no linker takes the marker for unused.
    std::hint::black_box(&REVISION);
    let found = std::env::current_exe().map_err(|e| format!("this program's own path: {e}")).and_then(|me| target(&me));
    let status = match found {
        Ok(program) => run(&program),
        Err(e) => {
            eprintln!("superfluid: {e}. The install script puts a release in place: {INSTALL}");
            1
        }
    };
    std::process::exit(status);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_launcher_runs_the_current_versions_program_of_its_own_name() {
        let dir = std::env::temp_dir().join(format!("sf-shim-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (bin, versions) = (dir.join("bin"), dir.join("home").join("versions"));
        std::fs::create_dir_all(&bin).unwrap();
        for v in ["0.2.0", "0.2.1"] {
            std::fs::create_dir_all(versions.join(v).join("bin")).unwrap();
            std::fs::write(versions.join(v).join("bin").join("superfluid.exe"), v).unwrap();
        }
        std::fs::write(versions.join("0.2.1").join("bin").join("superfluid-workerd.exe"), "w").unwrap();
        let launcher = bin.join("superfluid.exe");

        assert!(target(&launcher).unwrap_err().contains(VERSIONS_FILE), "nothing says where versions are");
        std::fs::write(bin.join(VERSIONS_FILE), format!("{}\n", versions.display())).unwrap();
        assert!(target(&launcher).unwrap_err().contains("current"), "no version is current");

        std::fs::write(versions.join("current"), "0.2.0\n").unwrap();
        assert_eq!(target(&launcher).unwrap(), versions.join("0.2.0").join("bin").join("superfluid.exe"));
        let e = target(&bin.join("superfluid-workerd.exe")).unwrap_err();
        assert!(e.contains("0.2.0") && e.contains("superfluid-workerd.exe"), "a version without the program is said: {e}");

        std::fs::write(versions.join("current"), "0.2.1\n").unwrap();
        assert_eq!(target(&launcher).unwrap(), versions.join("0.2.1").join("bin").join("superfluid.exe"));
        assert_eq!(target(&bin.join("superfluid-workerd.exe")).unwrap(), versions.join("0.2.1").join("bin").join("superfluid-workerd.exe"));

        for out in ["..\\..\\bin", "../x", "", "..", "C:evil"] {
            std::fs::write(versions.join("current"), out).unwrap();
            assert!(target(&launcher).unwrap_err().contains("does not name a version"), "{out:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
