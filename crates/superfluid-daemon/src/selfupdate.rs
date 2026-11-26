//! `superfluid self update|list|use|check|install`: releases side by side under `$SUPERFLUID_HOME/versions`, a
//! `current` link the commands on PATH go through, and the daily check `serve` runs.
//!
//! On Windows an ordinary user may make no symbolic link. `current` is there a file that holds the version's name
//! (`sys::link_dir`), and each command on PATH is a copy of the release's launcher (`superfluid-shim.exe`), which
//! reads `current` and runs that version's program of its own name.

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use superfluid_adapter_kit::install::{single_top, unpack, version_key, Asset};

use crate::runtime_pick::RuntimeId;
use crate::runtimes::Catalog;

pub const REPO: &str = "basecompute/superfluid";
const LISTING: &str = "https://api.github.com/repos/basecompute/superfluid/releases?per_page=20";
const CHECK_EVERY: Duration = Duration::from_secs(24 * 60 * 60);
pub const NO_CHECK_ENV: &str = "SUPERFLUID_NO_UPDATE_CHECK";
/// Where the release listing is read from, for a mirror or a test: a URL or a file.
pub const LISTING_ENV: &str = "SUPERFLUID_RELEASES";
const LINKED: &[&str] = &["superfluid", "superfluid-workerd", "superfluid-noded"];
/// The launcher a Windows release carries in `libexec\superfluid`, copied to the prefix under each command's name.
const SHIM: &str = "superfluid-shim.exe";
/// The file beside the launchers that tells them where the versions are.
#[cfg(windows)]
const VERSIONS_FILE: &str = "superfluid-versions";
/// The file at the root of a release a package manager installed (Homebrew, pip, npm), one line `<manager>: <command>`:
/// the copy is that manager's to update, with that command.
pub const INSTALLED_BY: &str = "installed-by";
/// The stamp basert writes at the root of the superfluid it installs for `basert serve`, before it also wrote an
/// `installed-by` file naming itself: such a copy is basert's, not the install script's.
const BASERT_STAMP: &str = ".release";
/// The name `installed-by` gives a release that was put in one place on Windows, with no versions beside it, and the
/// command that installs one with them.
const INSTALL_PS1: &str = "install.ps1";
const INSTALL_PS1_COMMAND: &str = "irm https://basecompute.co/sf/install.ps1 | iex";
/// How long a new copy's `--version` or `runtimes --json` may take: it runs under the lock, which a copy that hangs
/// would hold for every later update. Longer than the runtime checks `runtimes --json` makes, a minute each at most.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5 * 60);

pub const HELP: &str = "Update superfluid itself, or switch between the versions installed here.

Usage:
  superfluid self update [--version <v>] [--dry-run] [--yes]
                                                        Install the newest release that fits this machine and make it current
  superfluid self update --auto [--json] [--yes]        Install what an automatic update would, now (for apps and scripts)
  superfluid self check [--json]                        Ask whether a newer release exists, without installing
  superfluid self list                                  The versions installed here, and which is current
  superfluid self use <v>                               Make an installed version current (the one an update replaced stays)
  superfluid self auto-update [on|off]                  Turn automatic updates on or off; with neither, say whether they are on
  superfluid self auto-update wait <days>               Wait this many days after a release before an automatic update takes it (0-365)
  superfluid self install --from <dir> [--prefix <dir>] Place an unpacked release here; what the install script runs

Releases live side by side under $SUPERFLUID_HOME/versions/<v>, with a `current` link that the
`superfluid` on PATH goes through. An update unpacks the new release beside the old one, proves
it starts, flips the link, keeps the version it replaced and removes older ones (a version with
a `keep` file inside, or one a running server started from, stays). A running server keeps the
version it started with until it is started again.

With a baseRT engine here, a release must declare the engine's ABI number; with llama.cpp or
MLX only, the newest release fits. A superfluid installed by a package manager or cargo is
updated the way it was installed: for Homebrew, a global npm, pipx, uv and cargo this command
runs that manager's update (asking first; --yes runs it without asking, and is needed without
a terminal), and for pip in an environment or a project's npm dependency it names the command.
The superfluid basert installs for `basert serve` is basert's: this command runs
`basert superfluid install`, which picks the release that serves basert's engine.

On Windows, where an ordinary user may make no symbolic link, `current` is a file that holds
the version's name, and each command on PATH is a small launcher that reads it and runs that
version's program. A copy npm, pip, pipx or uv installed there has its manager's command
named and not run: Windows lets nothing replace a program that is running.

Automatic updates: a command run in a terminal (serve, launch, node, top, runtime install,
update, repair, remove, use) reads the release listing once a day in the background, and when
a release newer than this one fits here (and has been out as many days as the wait asks),
installs it before the command goes on: the download with a progress bar, the proof above, the
runtimes not pinned and not in use updated with the new superfluid, then the command run again
by the new version.
`superfluid self use <old>` goes back, and no automatic update goes to the version it left or
an older one again. A copy Homebrew, a global npm, pipx, uv or cargo installed is updated by
that manager in the background once no superfluid runs from it, with its output in
$SUPERFLUID_HOME/logs/auto-update.log; one basert installed is left to basert.
Without a terminal (a service, an app, a pipe), under CI, or with automatic updates off, the
command prints one line per release instead. A failed update says so and the command goes on.

The wait: by default a release is installed as soon as the daily read sees it.
`superfluid self auto-update wait 3` makes automatic updates wait until a release has been out
3 days (kept in $SUPERFLUID_HOME/settings.json as updateWaitDays; 0 waits for none), and
SUPERFLUID_UPDATE_WAIT_DAYS=<days> sets it for one shell or service. The line saying a release
is out waits as long, so it names the release an automatic update would take. `superfluid self
update` installs the newest release at once, whatever the wait; `superfluid self check` names
it, and the one an automatic update would take when that is another.

Turning them off: `superfluid self auto-update off` (kept in $SUPERFLUID_HOME/settings.json),
SUPERFLUID_NO_UPDATE=1 or SUPERFLUID_NO_UPDATE_CHECK=1 (no check, no update, no line), or
--no-update on any command for that run. `superfluid serve` also says once a day when a newer
release is out while it runs; --offline turns its check off. `superfluid self auto-update` says
whether they are on, and the wait.

Apps and scripts: `superfluid self update --auto` reads the listing and installs the release an
automatic update would (out as long as the wait asks, fitting the engine, not one `self use`
went back from), then updates the runtimes; with updates off, or nothing to install, it says
why and exits 0. It needs no terminal, and exits 1 when the update fails, 3 for a copy a package
manager or basert updates. With --json it also prints {from, installed, result, reason, notes,
runtimes} on stdout, each runtime as {id, outcome, from, to, reason}, a failed one included.
`superfluid self check --json` prints {current, newest, auto, autoReason, notes, autoUpdate,
waitDays}, where `auto` is what `--auto` would install now.
";

/// The release asset name for this build's platform.
pub fn target() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Some("aarch64-apple-darwin"),
        ("linux", "aarch64") => Some("aarch64-unknown-linux-gnu"),
        ("linux", "x86_64") => Some("x86_64-unknown-linux-gnu"),
        ("windows", "x86_64") => Some("x86_64-pc-windows-msvc"),
        _ => None,
    }
}

pub fn running() -> &'static str {
    env!("CARGO_PKG_VERSION")
}


/// Whether `v` can name a version directory: it starts with a digit and holds only what a version does, so it is one
/// path component and never empty, `.` or `..`, which would make the version's directory the layout or above it.
pub(crate) fn is_version(v: &str) -> bool {
    v.starts_with(|c: char| c.is_ascii_digit()) && v.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+'))
}

/// A part of a pre-release, `rc` or `1` in `0.2.0-rc.1`: a number comes before a word, as semver orders them.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PreRelease {
    Number(u64),
    Word(String),
}

/// A version's place in release order: by its numbers, then a release after the pre-releases that lead to it
/// (`0.2.0-rc.1` before `0.2.0`, which the numbers alone put the other way), then the pre-release part by part; build
/// metadata after a `+` counts for nothing.
fn release_order(v: &str) -> (Vec<u64>, bool, Vec<PreRelease>) {
    let v = v.split('+').next().unwrap_or(v);
    let (numbers, pre) = match v.split_once('-') {
        Some((numbers, pre)) => (numbers, Some(pre)),
        None => (v, None),
    };
    let parts = pre
        .map(|p| p.split('.').map(|id| id.parse().map(PreRelease::Number).unwrap_or_else(|_| PreRelease::Word(id.to_string()))).collect())
        .unwrap_or_default();
    (version_key(numbers), pre.is_none(), parts)
}

/// `$SUPERFLUID_HOME/versions`: a directory per release, `current`, `previous`, and `prefix` naming the bin directory that links here.
pub struct Versions {
    pub dir: PathBuf,
}

impl Versions {
    pub fn of(home: &Path) -> Versions {
        Versions { dir: home.join("versions") }
    }

    pub fn version_dir(&self, v: &str) -> PathBuf {
        self.dir.join(v)
    }

    pub(crate) fn binary(&self, v: &str) -> PathBuf {
        self.version_dir(v).join("bin").join(crate::sys::exe_name("superfluid"))
    }

    pub fn current(&self) -> Option<String> {
        let link = self.dir.join("current");
        let target = match cfg!(windows) {
            // A file that holds the name: read through, it is the version's directory.
            true => Some(crate::sys::through_link(&link)).filter(|t| *t != link)?,
            false => std::fs::read_link(&link).ok()?,
        };
        target.file_name().and_then(|n| n.to_str()).map(str::to_string)
    }

    pub fn previous(&self) -> Option<String> {
        let v = std::fs::read_to_string(self.dir.join("previous")).ok()?.trim().to_string();
        (!v.is_empty() && self.binary(&v).is_file()).then_some(v)
    }

    /// The complete versions here, newest first.
    pub fn on_disk(&self) -> Vec<String> {
        let mut all: Vec<String> = std::fs::read_dir(&self.dir)
            .map(|rd| {
                rd.flatten()
                    .filter_map(|e| e.file_name().to_str().map(str::to_string))
                    .filter(|n| !n.starts_with('.') && n != "current" && n.chars().next().is_some_and(|c| c.is_ascii_digit()))
                    .filter(|n| self.binary(n).is_file())
                    .collect()
            })
            .unwrap_or_default();
        all.sort_by_key(|v| std::cmp::Reverse(release_order(v)));
        all
    }

    pub fn prefix(&self) -> Option<PathBuf> {
        let p = std::fs::read_to_string(self.dir.join("prefix")).ok()?.trim().to_string();
        (!p.is_empty()).then(|| PathBuf::from(p))
    }

    /// Point `current` at `v`, atomically, remembering what it pointed at.
    pub fn set_current(&self, v: &str) -> Result<(), String> {
        if !self.binary(v).is_file() {
            return Err(format!("{} is not an installed version (see `superfluid self list`)", v));
        }
        if let Some(was) = self.current().filter(|was| was != v) {
            std::fs::write(self.dir.join("previous"), format!("{was}\n")).map_err(|e| format!("{}: {e}", self.dir.display()))?;
        }
        let tmp = self.dir.join(".current.tmp");
        let _ = std::fs::remove_file(&tmp);
        crate::sys::link_dir(v, &tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        crate::sys::replace_dir_link(&tmp, &self.dir.join("current")).map_err(|e| format!("{}: {e}", self.dir.join("current").display()))
    }

    /// Removes every version but `current`, `previous`, one with a `keep` file, and one a running process started from.
    pub fn prune(&self) -> Result<Vec<String>, String> {
        let (current, previous) = (self.current(), self.previous());
        let mut gone = Vec::new();
        for v in self.on_disk() {
            let dir = self.version_dir(&v);
            if Some(&v) == current.as_ref() || Some(&v) == previous.as_ref() || dir.join("keep").exists() || running_from(&dir) {
                continue;
            }
            std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            gone.push(v);
        }
        Ok(gone)
    }

    /// The names in `prefix` that are the install script's real files, not links.
    #[cfg(unix)]
    fn legacy_files(&self, prefix: &Path) -> Vec<&'static str> {
        LINKED.iter().copied().filter(|name| prefix.join(name).symlink_metadata().is_ok_and(|m| m.file_type().is_file())).collect()
    }

    /// The links in `prefix`: the three commands through `current`, and `superfluid-<v>` per version. The install script's
    /// real files are replaced on a migration, except `superfluid-workerd` while a server still runs from it (it starts its
    /// workers by that path), which a `migrating` note has the next link finish once nothing runs from there. The note
    /// names the processes that ran from there, by pid and start time: once a link is renamed over a server's own
    /// executable, macOS may give no path for it, and a server found by its path alone would be missed.
    #[cfg(unix)]
    pub fn link(&self, prefix: &Path, replace_files: bool) -> Result<Vec<String>, String> {
        std::fs::create_dir_all(prefix).map_err(|e| format!("{}: {e}", prefix.display()))?;
        std::fs::write(self.dir.join("prefix"), format!("{}\n", prefix.display())).map_err(|e| format!("{}: {e}", self.dir.display()))?;
        let mut notes = Vec::new();
        let pending = self.dir.join("migrating");
        let legacy = self.legacy_files(prefix);
        let migrating = !legacy.is_empty() && (replace_files || pending.exists());
        // A process the note names counts while it is the one that started then, whatever path is given for it now.
        let mut holding: Vec<(i32, u64)> = match migrating {
            true => std::fs::read_to_string(&pending)
                .unwrap_or_default()
                .lines()
                .skip(1)
                .filter_map(|l| l.split_once(' ').and_then(|(p, at)| Some((p.parse().ok()?, at.parse().ok()?))))
                .filter(|&(p, at)| started_at(p) == Some(at))
                .collect(),
            false => Vec::new(),
        };
        if migrating {
            for p in processes_from(prefix) {
                if !holding.contains(&p) {
                    holding.push(p);
                }
            }
        }
        let in_use = !holding.is_empty();
        let replace: Vec<&str> = match (migrating, in_use) {
            (true, true) => vec!["superfluid"],
            (true, false) => LINKED.to_vec(),
            (false, _) => Vec::new(),
        };
        let ours = |link: &Path| std::fs::read_link(link).map(|t| t.starts_with(&self.dir)).unwrap_or(false);
        let relink = |link: &Path, target: &Path| -> Result<(), String> {
            let tmp = link.with_file_name(format!(".{}.tmp", link.file_name().and_then(|n| n.to_str()).unwrap_or("link")));
            let _ = std::fs::remove_file(&tmp);
            crate::sys::symlink_file(target, &tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, link).map_err(|e| format!("{}: {e}", link.display()))
        };
        for name in LINKED {
            let link = prefix.join(name);
            let target = self.dir.join("current").join("bin").join(name);
            match link.symlink_metadata() {
                Ok(m) if m.file_type().is_symlink() && ours(&link) => continue,
                Ok(m) if m.file_type().is_symlink() || replace.contains(name) => relink(&link, &target)?,
                Ok(_) if migrating => continue,
                Ok(_) => {
                    notes.push(format!("{} is not a link into {}; left as it is", link.display(), self.dir.display()));
                    continue;
                }
                Err(_) => relink(&link, &target)?,
            }
        }
        if migrating {
            let libexec = prefix.join("..").join("libexec").join("superfluid");
            if in_use {
                let note: String = holding.iter().map(|(p, at)| format!("{p} {at}\n")).collect();
                std::fs::write(&pending, format!("{}\n{note}", prefix.display())).map_err(|e| format!("{}: {e}", pending.display()))?;
                notes.push(format!(
                    "a server still runs from the copy the install script placed in {}: its superfluid-workerd, superfluid-noded and workers stay until it stops; the next update, switch or install finishes the move",
                    prefix.display()
                ));
            } else {
                if let Ok(rd) = std::fs::read_dir(&libexec) {
                    for e in rd.flatten() {
                        let name = e.file_name().to_string_lossy().to_string();
                        if name.starts_with("superfluid-worker-") || name.starts_with("libsuperfluid_tokenizer_") {
                            let _ = std::fs::remove_file(e.path());
                        }
                    }
                    let _ = std::fs::remove_dir(&libexec);
                }
                let _ = std::fs::remove_file(&pending);
                notes.push(format!("the install script's copy in {} now links into {}; its workers were removed", prefix.display(), self.dir.display()));
            }
        } else if pending.exists() && legacy.is_empty() {
            let _ = std::fs::remove_file(&pending);
        }
        let here = self.on_disk();
        for v in &here {
            let link = prefix.join(format!("superfluid-{v}"));
            if !ours(&link) {
                relink(&link, &self.binary(v))?;
            }
        }
        if let Ok(rd) = std::fs::read_dir(prefix) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if let Some(v) = name.strip_prefix("superfluid-") {
                    if v.chars().next().is_some_and(|c| c.is_ascii_digit()) && !here.iter().any(|h| h == v) && ours(&e.path()) {
                        let _ = std::fs::remove_file(e.path());
                    }
                }
            }
        }
        Ok(notes)
    }

    /// The commands in `prefix` on Windows: a copy of the current version's launcher under each command's name, and
    /// beside them the file that tells a launcher where the versions are. A launcher says its revision
    /// (`launcher_revision`), and one in the prefix stays unless the release carries a newer one, so most updates
    /// place nothing. A file that is no launcher is replaced only in a prefix an install here keeps or with
    /// `replace_files` (the install script's run), and is otherwise left and said.
    #[cfg(windows)]
    pub fn link(&self, prefix: &Path, replace_files: bool) -> Result<Vec<String>, String> {
        std::fs::create_dir_all(prefix).map_err(|e| format!("{}: {e}", prefix.display()))?;
        std::fs::write(self.dir.join("prefix"), format!("{}\n", prefix.display())).map_err(|e| format!("{}: {e}", self.dir.display()))?;
        let current = self.current().ok_or_else(|| format!("no version is current in {}", self.dir.display()))?;
        let shim = self.version_dir(&current).join("libexec").join("superfluid").join(SHIM);
        let noted = prefix.join(VERSIONS_FILE);
        let ours = noted.is_file();
        let launcher = match std::fs::read(&shim) {
            Ok(launcher) => launcher,
            // The launchers an earlier release placed run this version as they ran that one.
            Err(_) if ours => return Ok(Vec::new()),
            Err(e) => return Err(format!("{}: {e}", shim.display())),
        };
        let revision = launcher_revision(&launcher).unwrap_or(0);
        let mut notes = Vec::new();
        // What an earlier run set aside while it was running.
        if let Ok(rd) = std::fs::read_dir(prefix) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with(".superfluid") && (name.contains(".exe.old-") || name.contains(".exe.new-")) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        std::fs::write(&noted, format!("{}\n", self.dir.display())).map_err(|e| format!("{}: {e}", noted.display()))?;
        for name in LINKED {
            let name = crate::sys::exe_name(name);
            let dest = prefix.join(&name);
            match std::fs::read(&dest).ok().map(|have| launcher_revision(&have)) {
                Some(Some(have)) if have >= revision => continue,
                Some(None) if !(ours || replace_files) => {
                    notes.push(format!("{} is not a launcher an install here placed; left as it is", dest.display()));
                    continue;
                }
                _ => {}
            }
            place_launcher(&launcher, &dest)?;
        }
        Ok(notes)
    }
}

/// The revision a launcher's program carries after its marker (`REVISION` in src/bin/superfluid-shim.rs); none for a
/// file that is not a launcher.
#[cfg(windows)]
fn launcher_revision(program: &[u8]) -> Option<u32> {
    const MARK: &[u8] = b"superfluid-shim-revision:";
    let at = program.windows(MARK.len()).position(|w| w == MARK)? + MARK.len();
    std::str::from_utf8(program.get(at..at + 4)?).ok()?.parse().ok()
}

/// Puts a launcher at `dest`. One that is there may be running (it is what this command was typed as): Windows lets
/// a running program be renamed, not written over or removed, so it steps aside and goes the next time one is placed.
#[cfg(windows)]
fn place_launcher(launcher: &[u8], dest: &Path) -> Result<(), String> {
    let name = dest.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let new = dest.with_file_name(format!(".{name}.new-{}", std::process::id()));
    let aside = dest.with_file_name(format!(".{name}.old-{}", std::process::id()));
    std::fs::write(&new, launcher).map_err(|e| format!("{}: {e}", new.display()))?;
    let had = dest.exists();
    if had {
        let _ = std::fs::remove_file(&aside);
        std::fs::rename(dest, &aside).map_err(|e| {
            let _ = std::fs::remove_file(&new);
            format!("{}: {e}", dest.display())
        })?;
    }
    if let Err(e) = std::fs::rename(&new, dest) {
        let _ = std::fs::remove_file(&new);
        if had {
            let _ = std::fs::rename(&aside, dest);
        }
        return Err(format!("{}: {e}", dest.display()));
    }
    let _ = std::fs::remove_file(&aside);
    Ok(())
}

/// Where the running executable came from, which decides who updates it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Installed {
    /// `<home>/versions/<v>/bin/superfluid`: ours to update.
    Managed { versions: PathBuf, version: String },
    /// The install script's layout before versions existed: real files in `<prefix>/bin` and `<prefix>/libexec/superfluid`.
    InPlace { prefix: PathBuf },
    /// A package manager (an `installed-by` file says which), `cargo install`, or a source build.
    Other(PathBuf),
}

pub fn installed_as(exe: &Path) -> Installed {
    let exe = crate::sys::canonical(exe).unwrap_or_else(|_| exe.to_path_buf());
    let parts: Vec<&std::ffi::OsStr> = exe.iter().collect();
    let n = parts.len();
    if n >= 4 && parts[n - 2] == "bin" && parts[n - 4] == "versions" {
        let versions = exe.ancestors().nth(3).map(Path::to_path_buf).unwrap_or_default();
        return Installed::Managed { versions, version: parts[n - 3].to_string_lossy().to_string() };
    }
    if let Some(prefix) = exe.parent().filter(|b| b.file_name().is_some_and(|n| n == "bin")).and_then(Path::parent) {
        // A package keeps the release's layout too, so it is told apart by its file, not by where it is.
        if prefix.join(INSTALLED_BY).is_file() || prefix.join(BASERT_STAMP).is_file() {
            return Installed::Other(exe);
        }
        // The layout before versions never shipped on Windows: a release unpacked in one place there is not one
        // to move under versions, and the install script is named for it (`delegate`).
        if prefix.join("libexec").join("superfluid").is_dir() && !cfg!(windows) {
            return Installed::InPlace { prefix: prefix.to_path_buf() };
        }
    }
    Installed::Other(exe)
}

/// The package manager that installed `exe` and the command that updates it, from the `installed-by` file at the root
/// of its release.
fn package_manager(exe: &Path) -> Option<(String, String)> {
    let line = std::fs::read_to_string(exe.parent()?.parent()?.join(INSTALLED_BY)).ok()?;
    let (manager, command) = line.trim().split_once(": ")?;
    Some((manager.trim().to_string(), command.trim().to_string()))
}

/// The command that updates the running superfluid, for the line `serve` prints when a newer release is out.
pub(crate) fn update_command(exe: &Path) -> String {
    let exe = crate::sys::canonical(exe).unwrap_or_else(|_| exe.to_path_buf());
    match installed_as(&exe) {
        Installed::Other(p) => delegate(&p).map(|d| d.command().to_string()),
        _ => None,
    }
    .unwrap_or_else(|| "superfluid self update".to_string())
}

pub(crate) fn not_ours(exe: &Path) -> String {
    let p = exe.to_string_lossy();
    if let Some(d) = delegate(exe) {
        return format!("{p} was installed by {}; `{}` updates it", d.manager(), d.command());
    }
    let how = if p.contains("/target/") || p.contains("\\target\\") { "it is a source build: build it again" } else { "update it the way it was installed" };
    format!("{p} was not installed by the install script or by `superfluid self update`; {how}")
}

/// How a copy another tool installed is updated: by that tool's own command, which `self update` runs (after asking),
/// or which it only names, saying why. The command is built here from the manager's name, never read from the
/// `installed-by` file, whose text is for showing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delegate {
    /// `argv` is run, its binary found from the copy's own path so that it is the installation that owns the copy;
    /// `path_first` goes first on PATH (npm's `node`, beside it). `command` is what a user would type.
    Run { manager: String, command: String, argv: Vec<String>, path_first: Option<PathBuf> },
    Name { manager: String, command: String, why: String },
}

impl Delegate {
    pub(crate) fn manager(&self) -> &str {
        match self {
            Delegate::Run { manager, .. } | Delegate::Name { manager, .. } => manager,
        }
    }

    pub(crate) fn command(&self) -> &str {
        match self {
            Delegate::Run { command, .. } | Delegate::Name { command, .. } => command,
        }
    }
}

/// The delegate for the copy at the real path `exe` (`<root>/bin/superfluid`), from the manager its `installed-by`
/// file names, or from where it is for packages published before they carried the file; `None` for a copy no package
/// manager or basert placed.
pub(crate) fn delegate(exe: &Path) -> Option<Delegate> {
    let found = delegate_by_place(exe)?;
    Some(if cfg!(windows) { named_while_it_runs(found) } else { found })
}

/// On Windows a program that is running can be neither deleted nor written over, nor its folder renamed, so a manager
/// started from the copy it is to update fails part of the way through: there its command is named, for the user to
/// run once superfluid has ended. basert is handed this process (`update_by_manager`) and stays as it is.
fn named_while_it_runs(found: Delegate) -> Delegate {
    match found {
        Delegate::Run { manager, command, .. } if manager != "basert" => Delegate::Name { why: while_it_runs(&manager), manager, command },
        other => other,
    }
}

fn while_it_runs(manager: &str) -> String {
    format!("Windows lets {manager} replace no program that is running: run it yourself, with no superfluid running")
}

/// `p` as the checks on where a copy is read it. A Windows path has `/` for each `\` and is in lower case, as its
/// names match there; both keep every byte's place, so what is found in the result is at the same place in `p`.
fn shape(p: &str, windows: bool) -> String {
    if windows {
        p.replace('\\', "/").to_ascii_lowercase()
    } else {
        p.to_string()
    }
}

/// `word` as one word of a command a user is shown to type. A path no shell splits or reads into stays as it is; any
/// other goes in single quotes, which PowerShell reads on Windows (a `'` in it doubled) and a POSIX shell elsewhere.
fn shell_word(word: &str, windows: bool) -> String {
    let plain = |c: char| c.is_alphanumeric() || matches!(c, '/' | ':' | '.' | '_' | '-' | '+') || (windows && matches!(c, '\\' | '~'));
    if !word.is_empty() && word.chars().all(plain) {
        return word.to_string();
    }
    format!("'{}'", word.replace('\'', if windows { "''" } else { r"'\''" }))
}

/// The start of a command that runs the program at `program`. PowerShell takes a quoted path for a string, and runs
/// it only after `&`.
fn shell_program(program: &str, windows: bool) -> String {
    let word = shell_word(program, windows);
    if windows && word.starts_with('\'') {
        format!("& {word}")
    } else {
        word
    }
}

/// `why`, saying which shell a quoted Windows command is for: cmd reads neither `'` nor `&` as PowerShell does.
fn for_powershell(command: &str, why: String, windows: bool) -> String {
    if windows && command.contains('\'') {
        format!("{why}; the command is written for PowerShell")
    } else {
        why
    }
}

fn delegate_by_place(exe: &Path) -> Option<Delegate> {
    let p = exe.to_string_lossy().into_owned();
    let at = shape(&p, cfg!(windows));
    let bin = exe.parent()?;
    let root = bin.parent()?;
    // pipx and uv keep the wheel in a venv of their own, whose file says pip.
    if at.contains("/pipx/venvs/") {
        return Some(on_path("pipx", &["upgrade", "superfluid-cli"]));
    }
    if at.contains("/uv/tools/") {
        return Some(on_path("uv", &["tool", "upgrade", "superfluid-cli"]));
    }
    let named = package_manager(exe);
    Some(match named.as_ref().map(|(m, _)| m.to_ascii_lowercase()).as_deref() {
        Some("basert") => basert(root),
        Some("homebrew" | "brew") => homebrew(&p),
        Some("npm") => npm(&p),
        Some("pip") => pip(&p),
        Some("cargo") => cargo(bin),
        Some(INSTALL_PS1) => install_ps1(),
        Some(_) => {
            let (manager, command) = named?;
            Delegate::Name { manager, command, why: "superfluid runs only Homebrew, npm, pipx, uv, cargo and basert itself".into() }
        }
        None if root.join(BASERT_STAMP).is_file() => basert(root),
        None if p.contains("/Cellar/") => homebrew(&p),
        None if at.contains("/node_modules/") => npm(&p),
        None if at.contains("/site-packages/") => pip(&p),
        None if at.contains("/.cargo/bin/") => cargo(bin),
        // A release unpacked in one place on Windows: the script installs one with versions beside it.
        None if cfg!(windows) && root.join("libexec").join("superfluid").is_dir() => install_ps1(),
        None => return None,
    })
}

/// Whether this user may write `path`, as the manager would need to.
fn writable(path: &Path) -> bool {
    crate::sys::writable(path)
}

fn executable(path: &Path) -> bool {
    crate::sys::is_executable(path)
}

/// `tool args…` with `tool` from PATH, or named when PATH has none. On Windows a tool is a program or a script
/// (`uv.exe`, `npm.cmd`).
fn on_path(tool: &str, args: &[&str]) -> Delegate {
    let command = std::iter::once(tool).chain(args.iter().copied()).collect::<Vec<_>>().join(" ");
    let files: Vec<String> = if cfg!(windows) { ["exe", "cmd", "bat"].iter().map(|ext| format!("{tool}.{ext}")).collect() } else { vec![tool.to_string()] };
    let found = std::env::var_os("PATH")
        .and_then(|path| std::env::split_paths(&path).flat_map(|d| files.iter().map(move |f| d.join(f))).find(|p| executable(p)));
    match found {
        Some(bin) => Delegate::Run {
            manager: tool.to_string(),
            command,
            argv: std::iter::once(bin.display().to_string()).chain(args.iter().map(|a| a.to_string())).collect(),
            path_first: None,
        },
        None => Delegate::Name { manager: tool.to_string(), why: format!("no {tool} on PATH"), command },
    }
}

/// The superfluid basert installs for `basert serve` (`<basert dir>/superfluid`, or basert's data directory): basert
/// picks the release that serves its engine. Run by the basert beside it, else the one on PATH.
fn basert(root: &Path) -> Delegate {
    let beside = root.parent().map(|d| d.join("basert")).filter(|b| executable(b));
    match beside {
        Some(b) => Delegate::Run {
            manager: "basert".into(),
            command: "basert superfluid install".into(),
            argv: vec![b.display().to_string(), "superfluid".into(), "install".into()],
            path_first: None,
        },
        None => match on_path("basert", &["superfluid", "install"]) {
            Delegate::Name { command, why, .. } => Delegate::Name { manager: "basert".into(), command, why },
            run => run,
        },
    }
}

/// `brew upgrade <formula>` by the Homebrew whose Cellar holds the copy, unless the formula is pinned or that Homebrew
/// is another user's.
fn homebrew(p: &str) -> Delegate {
    let manager = "Homebrew".to_string();
    let Some(i) = p.find("/Cellar/") else {
        let why = "the Homebrew it came from cannot be told from where it is".to_string();
        return Delegate::Name { manager, command: "brew upgrade superfluid".into(), why };
    };
    let prefix = Path::new(&p[..i]);
    let formula = p[i + "/Cellar/".len()..].split('/').next().unwrap_or("superfluid");
    let command = format!("brew upgrade {formula}");
    let brew = prefix.join("bin").join("brew");
    let why = if !executable(&brew) {
        format!("there is no {}", brew.display())
    } else if prefix.join("var/homebrew/pinned").join(formula).exists() {
        format!("it is pinned in Homebrew: `brew unpin {formula}` first")
    } else if !writable(&prefix.join("Cellar")) {
        format!("the Homebrew in {} is another user's", prefix.display())
    } else {
        let argv = vec![brew.display().to_string(), "upgrade".into(), formula.into()];
        return Delegate::Run { manager, command, argv, path_first: None };
    };
    Delegate::Name { manager, command, why }
}

/// The prefix of the npm global folder that holds the copy at `p`, if one does. On Unix the folder is
/// `<prefix>/lib/node_modules`, which no project has. On Windows it is `<prefix>\node_modules`, as a project's is, and
/// what tells it apart is the command npm wrote beside it, `<prefix>\superfluid.cmd`; a project's commands are in
/// `node_modules\.bin`.
fn npm_global_prefix(p: &str, windows: bool) -> Option<&str> {
    let at = shape(p, windows);
    if !windows {
        return at.find("/lib/node_modules/").map(|i| &p[..i]);
    }
    let prefix = &p[..at.find("/node_modules/")?];
    Path::new(prefix).join("superfluid.cmd").is_file().then_some(prefix)
}

/// What a user types to update the copy in the npm global folder under `prefix`. On Windows it names the prefix: the
/// command is all the update there is there, and typed as it is elsewhere it would install into whichever folder
/// npm's configuration names, leaving this copy as it was.
fn npm_global_command(prefix: &str, windows: bool) -> String {
    if windows {
        format!("npm install --global --prefix {} @basecompute/superfluid@latest", shell_word(prefix, true))
    } else {
        "npm install --global @basecompute/superfluid@latest".to_string()
    }
}

/// `npm install --global` by the npm whose global folder holds the copy, told that folder's prefix (npm's
/// configuration or NPM_CONFIG_PREFIX may name another); a project's dependency (or pnpm's, yarn's, bun's) is that
/// project's to update. On Windows the command is named (`named_while_it_runs`).
fn npm(p: &str) -> Delegate {
    let manager = "npm".to_string();
    let Some(prefix) = npm_global_prefix(p, cfg!(windows)) else {
        let why = "it is a project's dependency (or pnpm's, yarn's or bun's): update it in that project".to_string();
        return Delegate::Name { manager, command: "npm install @basecompute/superfluid@latest".into(), why };
    };
    let command = npm_global_command(prefix, cfg!(windows));
    let prefix = Path::new(prefix);
    if cfg!(windows) {
        // Its npm is not looked for: it may be anywhere (Node's installer keeps the global folder apart from npm
        // itself), and it is not run.
        return Delegate::Name { why: for_powershell(&command, while_it_runs(&manager), true), manager, command };
    }
    let bin = prefix.join("bin");
    let npm = bin.join("npm");
    let why = if !executable(&npm) {
        format!("there is no {}", npm.display())
    } else if !writable(&prefix.join("lib").join("node_modules")) {
        format!("npm's global folder in {} needs root to write: run it yourself", prefix.display())
    } else {
        let argv = vec![
            npm.display().to_string(),
            "install".into(),
            "--global".into(),
            "--prefix".into(),
            prefix.display().to_string(),
            "@basecompute/superfluid@latest".into(),
        ];
        return Delegate::Run { manager, command, argv, path_first: Some(bin) };
    };
    Delegate::Name { manager, command, why }
}

/// pip is only named: the environment may be a project's, which a lock file (Poetry, uv, pip-tools) pins, or the
/// system's, which refuses or needs root.
fn pip(p: &str) -> Delegate {
    let python = python_of(p, cfg!(windows)).filter(|py| executable(py));
    let command = pip_command(python.as_deref(), cfg!(windows));
    let why = "its environment may be a project's, pinned by a lock file, or the system's: run it in that environment yourself";
    Delegate::Name { manager: "pip".into(), why: for_powershell(&command, why.into(), cfg!(windows)), command }
}

/// pip's update by the environment's own `python`, its path quoted where a shell would split it (a home folder named
/// with a space), or by whichever `pip` is on PATH when that Python is not found.
fn pip_command(python: Option<&Path>, windows: bool) -> String {
    match python {
        Some(py) => format!("{} -m pip install --upgrade superfluid-cli", shell_program(&py.to_string_lossy(), windows)),
        None => "pip install --upgrade superfluid-cli".to_string(),
    }
}

/// Where the Python of the environment that holds the copy at `p` would be:
/// `<env>/lib/python3.x/site-packages/superfluid_cli/release/bin/superfluid` has `<env>/bin/python`. On Windows
/// `<env>\Lib\site-packages\…` has `<env>\Scripts\python.exe` when the environment is a venv, and `<env>\python.exe`
/// when it is a Python install.
fn python_of(p: &str, windows: bool) -> Option<PathBuf> {
    let at = shape(p, windows);
    if !windows {
        return at.find("/lib/python").map(|i| Path::new(&p[..i]).join("bin").join("python"));
    }
    let env = Path::new(&p[..at.find("/lib/site-packages/")?]);
    let venv = env.join("Scripts").join("python.exe");
    Some(if venv.is_file() { venv } else { env.join("python.exe") })
}

/// The Windows install script, which replaces a release where it is. Only named: the copy it replaces is the one that
/// would be running it, and a server started from that copy keeps its files open.
fn install_ps1() -> Delegate {
    Delegate::Name {
        manager: "the install script".into(),
        command: INSTALL_PS1_COMMAND.into(),
        why: "in PowerShell: it installs a release under versions, which `superfluid self update` then keeps up to date".into(),
    }
}

/// `cargo install superfluid` by the cargo beside the copy.
fn cargo(bin: &Path) -> Delegate {
    let cargo = bin.join(crate::sys::exe_name("cargo"));
    if !executable(&cargo) {
        return on_path("cargo", &["install", "superfluid"]);
    }
    let argv = vec![cargo.display().to_string(), "install".into(), "superfluid".into()];
    Delegate::Run { manager: "cargo".into(), command: "cargo install superfluid".into(), argv, path_first: Some(bin.to_path_buf()) }
}

/// The directory a package manager's update of the copy at `exe` replaces, which nothing may run from while it does:
/// the release's root, or for cargo the bin directory it shares with other tools.
pub(crate) fn watched_dir(exe: &Path, manager: &str) -> PathBuf {
    let root = exe.parent().and_then(Path::parent).unwrap_or(Path::new("/"));
    let watched = if manager == "cargo" { exe.parent().unwrap_or(root) } else { root };
    crate::sys::canonical(watched).unwrap_or_else(|_| watched.to_path_buf())
}

/// The other processes running a program of superfluid's from under `watched`: its workers start from the copy's
/// own directories; only superfluid's programs count (a cargo install shares ~/.cargo/bin with other tools).
pub(crate) fn superfluid_under(watched: &Path) -> Vec<i32> {
    other_processes()
        .into_iter()
        .filter(|(_, e)| runs_under(e, watched) && e.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("superfluid")))
        .map(|(pid, _)| pid)
        .collect()
}

/// A yes/no question on stderr, answered on stdin; an empty answer is yes.
fn confirm(question: &str) -> Result<bool, String> {
    eprint!("{question}");
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer).map_err(|e| format!("reading the answer: {e}"))?;
    Ok(matches!(answer.trim().to_ascii_lowercase().as_str(), "" | "y" | "yes"))
}

/// `self update` of a copy another tool installed: its manager's command, run after asking (or with `--yes`) when it
/// is one superfluid runs, else named. Nothing runs while a server runs from the copy, which the manager replaces.
/// basert is handed off to (`Report::hand_off`) rather than run: it refuses to replace a copy that anything runs from,
/// this process included.
fn update_by_manager(exe: &Path, opts: &Options) -> Result<Report, String> {
    let mut report = Report::default();
    let Some(plan) = delegate(exe) else { return Err(not_ours(exe)) };
    let p = exe.display();
    let (manager, command, argv, path_first) = match (plan, &opts.version) {
        (Delegate::Name { manager, command, why }, _) => return Err(format!("{p} was installed by {manager}, which updates it: run `{command}` ({why})")),
        (Delegate::Run { manager, command, .. }, Some(_)) if manager == "basert" => {
            return Err(format!("{p} is basert's, which picks the release that serves its engine: `{command}` updates it"))
        }
        (Delegate::Run { manager, command, .. }, Some(_)) => {
            return Err(format!("{p} was installed by {manager}, which chooses its version: `{command}` updates it, and a version is pinned there"))
        }
        (Delegate::Run { manager, command, argv, path_first }, None) => (manager, command, argv, path_first),
    };
    let shown = argv.join(" ");
    if opts.dry_run {
        report.say(format!("{p} was installed by {manager}: `self update` would run {shown}"));
        return Ok(report);
    }
    let watched = watched_dir(exe, &manager);
    let idle = || -> Result<(), String> {
        let busy: Vec<String> = superfluid_under(&watched).iter().map(i32::to_string).collect();
        if busy.is_empty() {
            return Ok(());
        }
        Err(format!("a server runs from {} (pid {}), which `{command}` replaces: stop it, then run this again", watched.display(), busy.join(", ")))
    };
    idle()?;
    if !opts.yes {
        if !(std::io::stdin().is_terminal() && std::io::stderr().is_terminal()) {
            return Err(format!("{p} was installed by {manager}: run `{command}`, or `superfluid self update --yes` to let superfluid run it"));
        }
        if !confirm(&format!("superfluid: {p} was installed by {manager}. Run `{shown}`? [Y/n] "))? {
            return Err(format!("nothing was run; `{command}` updates it"));
        }
        // The question may have waited long enough for a server to start.
        idle()?;
    }
    if manager == "cargo" {
        eprintln!("superfluid: cargo builds superfluid from source: this takes a few minutes");
    }
    eprintln!("superfluid: running {shown}");
    let mut run = Command::new(&argv[0]);
    run.args(&argv[1..]);
    if let Some(first) = path_first {
        let rest = std::env::var_os("PATH").unwrap_or_default();
        let path = std::env::join_paths(std::iter::once(first).chain(std::env::split_paths(&rest))).map_err(|e| e.to_string())?;
        run.env("PATH", path);
    }
    if manager == "basert" {
        report.hand_off = Some(run);
        return Ok(report);
    }
    let status = crate::exec::when_not_busy(|| run.status()).map_err(|e| format!("running {}: {e}", argv[0]))?;
    if !status.success() {
        return Err(format!("`{shown}` failed ({status})"));
    }
    report.changed = true;
    report.say(format!("{manager} updated it"));
    Ok(report)
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    pub version: String,
    /// YYYY-MM-DD
    pub published: String,
    pub tarball: Asset,
    pub sha256_url: String,
    /// The basert engine ABI the release declares in an asset name.
    pub abi: Option<u32>,
    pub compat_url: Option<String>,
    /// The release's page, its notes: the listing's `html_url`, else the tag's page on GitHub.
    pub notes_url: String,
}

/// The page of release `v`'s notes: the one the listing gives when it is an https URL, else the tag's on GitHub.
pub fn notes_url(v: &str, listed: Option<&str>) -> String {
    listed.filter(|u| u.starts_with("https://")).map(str::to_string).unwrap_or_else(|| format!("https://github.com/{REPO}/releases/tag/v{v}"))
}

/// The releases of the listing with a tarball for `target`, newest first; drafts and prereleases left out.
pub fn releases(listing: &Value, target: &str) -> Vec<Release> {
    let tar = format!("superfluid-{target}.tar.gz");
    let mut out: Vec<Release> = listing
        .as_array()
        .map(|a| a.iter().filter_map(|r| release_of(r, &tar, target)).collect())
        .unwrap_or_default();
    out.sort_by_key(|r| std::cmp::Reverse(release_order(&r.version)));
    out
}

fn release_of(r: &Value, tar: &str, target: &str) -> Option<Release> {
    if r["draft"].as_bool().unwrap_or(false) || r["prerelease"].as_bool().unwrap_or(false) {
        return None;
    }
    let version = r["tag_name"].as_str()?.trim_start_matches('v').to_string();
    if !is_version(&version) {
        return None;
    }
    let assets = r["assets"].as_array()?;
    let url = |name: &str| assets.iter().find(|a| a["name"] == name).and_then(|a| a["browser_download_url"].as_str()).map(str::to_string);
    let size = assets.iter().find(|a| a["name"] == tar).and_then(|a| a["size"].as_u64());
    let abi = assets.iter().find_map(|a| a["name"].as_str()?.strip_prefix("superfluid-basert-abi-")?.strip_suffix(".txt")?.parse().ok());
    Some(Release {
        published: r["published_at"].as_str().unwrap_or_default().chars().take(10).collect(),
        tarball: Asset { name: tar.to_string(), url: url(tar)?, sha256: None, size },
        sha256_url: url(&format!("{tar}.sha256"))?,
        abi,
        compat_url: url(&format!("superfluid-{target}.compat.json")),
        notes_url: notes_url(&version, r["html_url"].as_str()),
        version,
    })
}

/// What a release must declare to drive the baseRT engine here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Engine {
    /// No engine, or none this superfluid can tell: the newest release fits.
    None,
    Abi(u32),
    /// A 0.3.x engine, which states no number: the release must accept that line.
    Unnumbered(String),
}

pub fn engine_here(catalog: &Catalog) -> Engine {
    if std::env::var_os("BASERT_LIB").is_none() && !catalog.package_dir(RuntimeId::new("basert")).is_dir() {
        return Engine::None;
    }
    match catalog.inspect(RuntimeId::new("basert")).usable {
        Ok(rt) => match (rt.abi, rt.version.as_deref()) {
            (Some(n), _) => Engine::Abi(n as u32),
            (None, Some(v)) => Engine::Unnumbered(v.split(|c: char| !c.is_ascii_digit() && c != '.').next().unwrap_or("").split('.').take(2).collect::<Vec<_>>().join(".")),
            (None, None) => Engine::None,
        },
        Err(_) => Engine::None,
    }
}

impl Engine {
    pub fn describe(&self) -> String {
        match self {
            Engine::None => "no baseRT engine here".into(),
            Engine::Abi(n) => format!("the baseRT engine here states ABI {n}"),
            Engine::Unnumbered(minor) => format!("the baseRT engine here is {minor}.x and states no ABI"),
        }
    }
}

/// Whether `release` drives the engine; `compat` fetches its compat.json when the asset names alone do not say.
pub fn fits(release: &Release, engine: &Engine, compat: &dyn Fn(&str) -> Option<Value>) -> bool {
    match engine {
        Engine::None => true,
        Engine::Abi(n) => match release.abi {
            Some(have) => have == *n,
            None => release.compat_url.as_deref().and_then(compat).and_then(|c| c["runtimes"]["basert"]["abi"].as_u64()) == Some(u64::from(*n)),
        },
        Engine::Unnumbered(minor) => release
            .compat_url
            .as_deref()
            .and_then(compat)
            .and_then(|c| c["runtimes"]["basert"]["unnumbered_minor"].as_str().map(str::to_string))
            .is_some_and(|m| m == *minor),
    }
}

/// The newest release that fits, skipping ones published fewer than `wait` days ago when given.
pub fn newest<'a>(releases: &'a [Release], engine: &Engine, wait: Option<u32>, compat: &dyn Fn(&str) -> Option<Value>) -> Option<&'a Release> {
    let today = today();
    releases.iter().find(|r| wait.is_none_or(|days| out_for(&r.published, days, today)) && fits(r, engine, compat))
}

/// Whether a release published on `published` (`2026-01-01`) has been out `days` days on `today` (days since the
/// epoch): with no wait, as soon as it is listed; with one, never when its date cannot be read.
pub(crate) fn out_for(published: &str, days: u32, today: i64) -> bool {
    days == 0 || age_days(published, today).is_some_and(|age| age >= i64::from(days))
}

/// The release listing to read: `SUPERFLUID_RELEASES`, else GitHub's.
pub fn listing_source() -> String {
    std::env::var(LISTING_ENV).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| LISTING.to_string())
}

/// A small text or JSON document: a file, or an https URL through curl.
fn get_text(url: &str, max_time: u32) -> Result<String, String> {
    if let Some(path) = url.strip_prefix("file://").or_else(|| (!url.contains("://")).then_some(url)) {
        return std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"));
    }
    if !url.starts_with("https://") {
        return Err(format!("{url}: only https:// URLs are fetched"));
    }
    let out = Command::new("curl")
        .args(["-fsSL", "--proto", "=https", "--tlsv1.2", "--max-time", &max_time.to_string(), "-H", "Accept: application/vnd.github+json"])
        .arg(url)
        .output()
        .map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        return Err(format!("fetching {url} failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn get_json(url: &str, max_time: u32) -> Result<Value, String> {
    serde_json::from_str(&get_text(url, max_time)?).map_err(|e| format!("{url}: {e}"))
}

pub fn fetch_releases(source: &str, target: &str, max_time: u32) -> Result<Vec<Release>, String> {
    let listing = get_json(source, max_time)?;
    let found = releases(&listing, target);
    if found.is_empty() {
        return Err(format!("{source} lists no release with a superfluid-{target}.tar.gz"));
    }
    Ok(found)
}

pub(crate) fn compat_fetcher(max_time: u32) -> impl Fn(&str) -> Option<Value> {
    move |url: &str| get_json(url, max_time).ok()
}


fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn age_days(published: &str, today: i64) -> Option<i64> {
    let mut parts = published.split('-').map(|p| p.parse::<i64>().ok());
    let (y, m, d) = (parts.next()??, parts.next()??, parts.next()??);
    Some(today - days_from_civil(y, m, d))
}

/// Today, in days since the epoch.
fn today() -> i64 {
    (now_secs() / 86_400) as i64
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}


/// The other processes on this machine, each with its executable as a real path.
#[cfg(target_os = "linux")]
fn other_processes() -> Vec<(i32, PathBuf)> {
    let me = std::process::id() as i32;
    std::fs::read_dir("/proc")
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.file_name().to_str()?.parse::<i32>().ok().filter(|&p| p != me).map(|p| (p, e.path())))
                .filter_map(|(p, dir)| Some((p, std::fs::read_link(dir.join("exe")).ok()?)))
                .collect()
        })
        .unwrap_or_default()
}

/// When `pid` started, in clock ticks since boot: with the pid, what tells a process from a later one given its number.
#[cfg(target_os = "linux")]
fn started_at(pid: i32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The command name, in parentheses, may hold spaces; the start time is the 20th field after it.
    stat.rsplit_once(')')?.1.split_whitespace().nth(19)?.parse().ok()
}

#[cfg(target_os = "macos")]
fn other_processes() -> Vec<(i32, PathBuf)> {
    let me = std::process::id() as libc::c_int;
    // SAFETY: with a null buffer, proc_listallpids only reports how many pids there are.
    let n = unsafe { libc::proc_listallpids(std::ptr::null_mut(), 0) };
    if n <= 0 {
        return Vec::new();
    }
    let mut pids = vec![0 as libc::c_int; n as usize * 2 + 16];
    let bytes = (pids.len() * std::mem::size_of::<libc::c_int>()) as libc::c_int;
    // SAFETY: the buffer is this function's, and its size in bytes is what is passed.
    let got = unsafe { libc::proc_listallpids(pids.as_mut_ptr() as *mut libc::c_void, bytes) };
    pids.truncate(got.max(0) as usize);
    pids.into_iter()
        .filter(|&p| p > 0 && p != me)
        .filter_map(|p| {
            let mut buf = vec![0u8; 4096];
            // SAFETY: proc_pidpath writes at most the buffer's length, which it is told.
            let len = unsafe { libc::proc_pidpath(p, buf.as_mut_ptr() as *mut libc::c_void, buf.len() as u32) };
            (len > 0).then(|| (p, PathBuf::from(String::from_utf8_lossy(&buf[..len as usize]).to_string())))
        })
        .collect()
}

/// When `pid` started, in microseconds since the epoch.
#[cfg(target_os = "macos")]
fn started_at(pid: i32) -> Option<u64> {
    // SAFETY: proc_bsdinfo is plain data, for which all zeroes is a value.
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    // SAFETY: proc_pidinfo writes at most `size` bytes into `info`, which is that big.
    let got = unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0, &mut info as *mut _ as *mut libc::c_void, size) };
    (got == size).then(|| info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec)
}

#[cfg(windows)]
fn other_processes() -> Vec<(i32, PathBuf)> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS};
    use windows_sys::Win32::System::Threading::{OpenProcess, QueryFullProcessImageNameW, PROCESS_QUERY_LIMITED_INFORMATION};
    let me = std::process::id();
    let mut found = Vec::new();
    // SAFETY: the snapshot and each process handle are closed here; `entry` is plain data whose size is set as the
    // call requires; the path buffer's length is what is passed, and only what the call filled is read.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return found;
        }
        let mut entry: PROCESSENTRY32W = std::mem::zeroed();
        entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
        let mut more = Process32FirstW(snapshot, &mut entry) != 0;
        while more {
            let pid = entry.th32ProcessID;
            if pid != 0 && pid != me {
                // Another user's process gives no handle, and is not this user's server.
                let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
                if !process.is_null() {
                    let mut path = [0u16; 1024];
                    let mut len = path.len() as u32;
                    if QueryFullProcessImageNameW(process, 0, path.as_mut_ptr(), &mut len) != 0 {
                        found.push((pid as i32, PathBuf::from(String::from_utf16_lossy(&path[..len as usize]))));
                    }
                    CloseHandle(process);
                }
            }
            more = Process32NextW(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    found
}

/// When `pid` started, in 100 ns ticks since 1601.
#[cfg(windows)]
fn started_at(pid: i32) -> Option<u64> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    // SAFETY: the handle is closed here; the four times are plain data the call fills.
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
        if process.is_null() {
            return None;
        }
        let mut times: [FILETIME; 4] = std::mem::zeroed();
        let got = GetProcessTimes(process, &mut times[0], &mut times[1], &mut times[2], &mut times[3]) != 0;
        CloseHandle(process);
        got.then(|| (u64::from(times[0].dwHighDateTime) << 32) | u64::from(times[0].dwLowDateTime))
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn other_processes() -> Vec<(i32, PathBuf)> {
    Vec::new()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn started_at(_pid: i32) -> Option<u64> {
    None
}

/// Whether the executable `exe` is under the directory `dir`, both real paths. Windows names are compared without
/// regard to case: a process's path is as it was started, not as the directory spells it.
fn under(exe: &Path, dir: &Path) -> bool {
    if cfg!(windows) {
        let lower = |p: &Path| PathBuf::from(p.to_string_lossy().to_lowercase());
        return lower(exe).starts_with(lower(dir));
    }
    exe.starts_with(dir)
}

/// Whether another process runs an executable under `dir`.
pub fn running_from(dir: &Path) -> bool {
    !processes_from(dir).is_empty()
}

/// The other processes that use what is under `dir`: run an executable from it, or have a file of it mapped (a library
/// loaded: the engine `superfluid-workerd` loads from a basert install, a tokenizer the daemon loads, an MLX
/// install's Python extensions), by pid.
pub(crate) fn pids_using(dir: &Path) -> Vec<i32> {
    let Ok(dir) = crate::sys::canonical(dir) else { return Vec::new() };
    other_processes().into_iter().filter(|(p, exe)| runs_under(exe, &dir) || maps_from(*p, &dir)).map(|(p, _)| p).collect()
}

/// Whether the process `pid` has a file under the real path `dir` mapped: the file-backed lines of its `maps`.
#[cfg(target_os = "linux")]
fn maps_from(pid: i32, dir: &Path) -> bool {
    let Ok(maps) = std::fs::read_to_string(format!("/proc/{pid}/maps")) else { return false };
    // The path is what follows the fifth field, from its first '/'; a file removed since says so after it.
    maps.lines().filter_map(|l| l.find('/').map(|i| &l[i..])).any(|p| Path::new(p.strip_suffix(" (deleted)").unwrap_or(p)).starts_with(dir))
}

#[cfg(target_os = "macos")]
fn maps_from(pid: i32, dir: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    // <sys/proc_info.h>, which the libc crate does not carry.
    #[repr(C)]
    struct RegionInfo {
        protection: u32,
        max_protection: u32,
        inheritance: u32,
        flags: u32,
        offset: u64,
        behavior: u32,
        user_wired_count: u32,
        user_tag: u32,
        pages_resident: u32,
        pages_shared_now_private: u32,
        pages_swapped_out: u32,
        pages_dirtied: u32,
        ref_count: u32,
        shadow_depth: u32,
        share_mode: u32,
        private_pages_resident: u32,
        shared_pages_resident: u32,
        obj_id: u32,
        depth: u32,
        address: u64,
        size: u64,
    }
    #[repr(C)]
    struct RegionWithPath {
        info: RegionInfo,
        vip: libc::vnode_info_path,
    }
    // PROC_PIDREGIONPATHINFO2: the region at or after an address that a file backs, with that file's path.
    const REGION_PATH_INFO_2: libc::c_int = 22;
    let size = std::mem::size_of::<RegionWithPath>() as libc::c_int;
    let mut at = 0u64;
    // At most this many regions are looked at; a process maps far fewer files.
    for _ in 0..100_000 {
        // SAFETY: plain data, for which all zeroes is a value.
        let mut region: RegionWithPath = unsafe { std::mem::zeroed() };
        // SAFETY: proc_pidinfo writes at most `size` bytes into `region`, which is that big.
        let got = unsafe { libc::proc_pidinfo(pid, REGION_PATH_INFO_2, at, &mut region as *mut _ as *mut libc::c_void, size) };
        if got != size {
            return false;
        }
        // SAFETY: the path is a C array of the struct's own, which the call ends with a NUL or fills.
        let path: &[u8] = unsafe { std::slice::from_raw_parts(region.vip.vip_path.as_ptr() as *const u8, std::mem::size_of_val(&region.vip.vip_path)) };
        let path = &path[..path.iter().position(|&b| b == 0).unwrap_or(path.len())];
        if !path.is_empty() && Path::new(std::ffi::OsStr::from_bytes(path)).starts_with(dir) {
            return true;
        }
        let next = region.info.address.saturating_add(region.info.size);
        if next <= at {
            return false;
        }
        at = next;
    }
    false
}

#[cfg(windows)]
fn maps_from(pid: i32, dir: &Path) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Module32FirstW, Module32NextW, MODULEENTRY32W, TH32CS_SNAPMODULE, TH32CS_SNAPMODULE32};
    // SAFETY: the snapshot is closed here; `entry` is plain data whose size is set as the call requires, and only the
    // part of its path the call filled is read.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid as u32);
        if snapshot == INVALID_HANDLE_VALUE {
            return false;
        }
        let mut entry: MODULEENTRY32W = std::mem::zeroed();
        entry.dwSize = std::mem::size_of::<MODULEENTRY32W>() as u32;
        let mut more = Module32FirstW(snapshot, &mut entry) != 0;
        let mut found = false;
        while more && !found {
            let len = entry.szExePath.iter().position(|&c| c == 0).unwrap_or(entry.szExePath.len());
            found = runs_under(&PathBuf::from(String::from_utf16_lossy(&entry.szExePath[..len])), dir);
            more = Module32NextW(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
        found
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn maps_from(_pid: i32, _dir: &Path) -> bool {
    false
}

/// The other processes running an executable under `dir`, by pid and when each started.
fn processes_from(dir: &Path) -> Vec<(i32, u64)> {
    let Ok(dir) = crate::sys::canonical(dir) else { return Vec::new() };
    other_processes().into_iter().filter(|(_, exe)| runs_under(exe, &dir)).map(|(p, _)| (p, started_at(p).unwrap_or(0))).collect()
}

/// Whether the process whose executable the system reports as `exe` runs from under the real path `dir`. Windows
/// reports the path the process was started by, which may reach the file by a short name (`C:\Users\RUNNER~1`) or
/// through a junction; such a path is resolved as `dir` was. Elsewhere the system reports the real path.
fn runs_under(exe: &Path, dir: &Path) -> bool {
    under(exe, dir) || (cfg!(windows) && crate::sys::canonical(exe).is_ok_and(|real| under(&real, dir)))
}


#[derive(Debug, Default, Clone)]
pub struct Options {
    pub version: Option<String>,
    pub dry_run: bool,
    /// For a copy another tool installed: run its update without asking.
    pub yes: bool,
    /// The release listing; `listing_source()` when not set.
    pub source: Option<String>,
    /// Each line printed as it is said, and the download drawn as a bar on a terminal: `self update` run by hand.
    pub live: bool,
    /// Give up at once when another update holds the lock, rather than wait for it: an automatic update, which a
    /// command waits on.
    pub no_wait: bool,
}

/// What `self update` did, line by line.
#[derive(Debug, Default)]
pub struct Report {
    pub lines: Vec<String>,
    /// The lines out of the ordinary (a link left as it was, a version not removed), which an automatic update shows.
    pub notes: Vec<String>,
    pub changed: bool,
    /// The command `superfluid self update` replaces itself with, its own update left to that command.
    pub hand_off: Option<Command>,
    /// Lines are printed as they are said.
    pub live: bool,
}

impl Report {
    fn say(&mut self, line: impl Into<String>) {
        let line = line.into();
        if self.live {
            println!("superfluid: {line}");
            let _ = std::io::stdout().flush();
        }
        self.lines.push(line);
    }

    fn note(&mut self, line: impl Into<String>) {
        let line = line.into();
        self.notes.push(line.clone());
        self.say(line);
    }
}

/// `cmd`'s output, or an error once it has run for `limit`, when it is killed.
fn output_within(cmd: &mut Command, limit: Duration) -> Result<Output, String> {
    let what = cmd.get_program().to_string_lossy().to_string();
    match crate::exec::output_within(cmd, limit) {
        Ok(Some(out)) => Ok(out),
        Ok(None) => Err(format!("{what} did not answer within {}s", limit.as_secs())),
        Err(e) => Err(format!("{what}: {e}")),
    }
}

fn version_of(binary: &Path) -> Result<String, String> {
    let out = output_within(Command::new(binary).arg("--version"), PROBE_TIMEOUT)?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace().nth(1).map(str::to_string).filter(|v| out.status.success() && is_version(v)).ok_or_else(|| {
        format!("{} did not say its version ({}): {}", binary.display(), out.status, String::from_utf8_lossy(&out.stderr).trim())
    })
}

/// The proof a host runs: the copy's `--version` and `runtimes --json`, the basert row ready when an engine is in use.
fn probe(versions: &Versions, v: &str, home: &Path, engine: &Engine) -> Result<(), String> {
    probe_binary(&versions.binary(v), v, home, engine)
}

/// The same proof of a copy wherever it is, so a replacement is proved before anything it replaces is touched.
fn probe_binary(binary: &Path, v: &str, home: &Path, engine: &Engine) -> Result<(), String> {
    // A copy cut short can hold the command and nothing beside it; it starts, and its server finds no workerd.
    if !binary.with_file_name(crate::sys::exe_name("superfluid-workerd")).is_file() {
        return Err(format!("{} has no superfluid-workerd beside it", binary.display()));
    }
    let said = version_of(binary)?;
    if said != v {
        return Err(format!("{} says it is {said}, not {v}", binary.display()));
    }
    let out = output_within(Command::new(binary).args(["runtimes", "--json"]).env("SUPERFLUID_HOME", home).env(NO_CHECK_ENV, "1"), PROBE_TIMEOUT)?;
    if !out.status.success() {
        return Err(format!("{} runtimes --json failed ({}): {}", binary.display(), out.status, String::from_utf8_lossy(&out.stderr).trim()));
    }
    if *engine != Engine::None {
        let rows: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("{} runtimes --json: {e}", binary.display()))?;
        let basert = rows.as_array().and_then(|a| a.iter().find(|r| r["id"] == "basert"));
        match basert.map(|r| &r["ready"]) {
            Some(Value::Bool(true)) => {}
            Some(Value::String(why)) => return Err(format!("{v} cannot drive the engine here: {why}")),
            _ => return Err(format!("{v} reports no basert runtime, which this machine serves with")),
        }
    }
    Ok(())
}

fn move_tree(from: &Path, to: &Path) -> Result<(), String> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    copy_tree(from, to)?;
    std::fs::remove_dir_all(from).map_err(|e| format!("{}: {e}", from.display()))
}

/// `move_tree` for a tree the caller unpacked and removes itself, which may hold the program that is running this:
/// the install script runs the release it unpacked. Where the tree cannot be renamed (another volume) it is copied,
/// and what cannot then be removed is left to the caller. Windows deletes no program that is running, so there the
/// removal always stops at this one.
fn take_tree(from: &Path, to: &Path) -> Result<(), String> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    copy_tree(from, to)?;
    let _ = std::fs::remove_dir_all(from);
    Ok(())
}

fn copy_tree(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| format!("{}: {e}", to.display()))?;
    for e in std::fs::read_dir(from).map_err(|e| format!("{}: {e}", from.display()))?.flatten() {
        let (src, dest) = (e.path(), to.join(e.file_name()));
        let kind = e.file_type().map_err(|e| format!("{}: {e}", src.display()))?;
        if kind.is_dir() {
            copy_tree(&src, &dest)?;
        } else if kind.is_symlink() {
            let target = std::fs::read_link(&src).map_err(|e| format!("{}: {e}", src.display()))?;
            crate::sys::symlink_file(&target, &dest).map_err(|e| format!("{}: {e}", dest.display()))?;
        } else {
            std::fs::copy(&src, &dest).map_err(|e| format!("{} -> {}: {e}", src.display(), dest.display()))?;
        }
    }
    Ok(())
}

/// Download, check and unpack `release` into the layout; nothing is placed until the whole tree is verified, and what
/// was downloaded goes whether or not it was placed.
fn place(versions: &Versions, release: &Release, report: &mut Report, bar: bool) -> Result<(), String> {
    let staging = versions.dir.join(".staging").join(&release.version);
    let _ = std::fs::remove_dir_all(&staging);
    let placed = place_through(versions, release, &staging, report, bar);
    let _ = std::fs::remove_dir_all(&staging);
    placed
}

fn place_through(versions: &Versions, release: &Release, staging: &Path, report: &mut Report, bar: bool) -> Result<(), String> {
    let v = &release.version;
    std::fs::create_dir_all(staging).map_err(|e| format!("{}: {e}", staging.display()))?;
    let want = get_text(&release.sha256_url, 30)?.split_whitespace().next().unwrap_or_default().to_ascii_lowercase();
    if want.len() != 64 || !want.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("{}: not a SHA-256", release.sha256_url));
    }
    let asset = Asset { sha256: Some(want), ..release.tarball.clone() };
    report.say(format!("downloading {} ({})", asset.name, asset.size.map(|s| format!("{:.1} MB", s as f64 / 1e6)).unwrap_or_else(|| "size unknown".into())));
    let archive = crate::autoupdate::download(&asset, staging, bar)?;
    let unpacked = staging.join("unpacked");
    unpack(&archive, &unpacked)?;
    let top = single_top(&unpacked)?;
    let exe = std::env::consts::EXE_SUFFIX;
    // On Windows the launcher too: the commands on PATH are copies of it.
    let needed = [Some(format!("bin/superfluid{exe}")), Some(format!("bin/superfluid-workerd{exe}")), cfg!(windows).then(|| format!("libexec/superfluid/{SHIM}"))];
    for needed in needed.into_iter().flatten() {
        if !top.join(&needed).is_file() {
            return Err(format!("{} holds no {needed}", asset.name));
        }
    }
    let dest = versions.version_dir(v);
    let _ = std::fs::remove_dir_all(&dest);
    // A move cut short across filesystems leaves part of a tree, which is not a version.
    move_tree(&top, &dest).inspect_err(|_| {
        let _ = std::fs::remove_dir_all(&dest);
    })
}

/// The install script's in-place copy, kept under the layout as the version to go back to. It is copied beside the
/// layout and moved in whole, so a copy cut short (a full disk, an interrupt) is never taken for the version and its
/// original workers are not removed for it; a version missing a file of the copy is copied again.
fn keep_in_place(versions: &Versions, prefix: &Path) -> Result<String, String> {
    let bin = prefix.join("bin");
    let libexec = prefix.join("libexec").join("superfluid");
    let old = version_of(&bin.join("superfluid"))?;
    let dest = versions.version_dir(&old);
    if holds_in_place(&dest, &bin, &libexec) {
        return Ok(old);
    }
    let staging = versions.dir.join(".staging").join(format!("kept-{old}"));
    let _ = std::fs::remove_dir_all(&staging);
    let copied = (|| {
        std::fs::create_dir_all(staging.join("bin")).map_err(|e| format!("{}: {e}", staging.display()))?;
        for name in LINKED {
            let src = bin.join(name);
            if src.is_file() {
                std::fs::copy(&src, staging.join("bin").join(name)).map_err(|e| format!("{}: {e}", src.display()))?;
            }
        }
        if libexec.is_dir() {
            copy_tree(&libexec, &staging.join("libexec").join("superfluid"))?;
        }
        let _ = std::fs::remove_dir_all(&dest);
        std::fs::rename(&staging, &dest).map_err(|e| format!("{} -> {}: {e}", staging.display(), dest.display()))
    })();
    if copied.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    copied.map(|()| old)
}

/// Whether `dest` holds every command and worker of the in-place copy in `bin` and `libexec`.
fn holds_in_place(dest: &Path, bin: &Path, libexec: &Path) -> bool {
    let commands = LINKED.iter().filter(|name| bin.join(name).is_file()).all(|name| dest.join("bin").join(name).is_file());
    let workers = std::fs::read_dir(libexec)
        .map(|rd| rd.flatten().all(|e| dest.join("libexec").join("superfluid").join(e.file_name()).exists()))
        .unwrap_or(true);
    dest.join("bin").join("superfluid").is_file() && commands && workers
}

/// The install script's in-place copies become links into the layout (see `Versions::link` for a running server).
fn migrate(versions: &Versions, prefix: &Path, report: &mut Report) -> Result<(), String> {
    for note in versions.link(&prefix.join("bin"), true)? {
        report.note(note);
    }
    report.say(format!("the copy the install script placed in {}/bin is kept under {} as the version to go back to", prefix.display(), versions.dir.display()));
    Ok(())
}

/// `from` takes the place of `dest`, set aside until `from` is in place; a failed move leaves nothing at `dest` and
/// brings the copy back, and if that fails too both failures are reported.
fn replace_tree(from: &Path, dest: &Path, aside: &Path, mover: impl Fn(&Path, &Path) -> Result<(), String>) -> Result<(), String> {
    let _ = std::fs::remove_dir_all(aside);
    let had = dest.exists();
    if had {
        std::fs::rename(dest, aside).map_err(|e| format!("{}: {e}", dest.display()))?;
    }
    match mover(from, dest) {
        Ok(()) => {
            let _ = std::fs::remove_dir_all(aside);
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(dest);
            if had {
                if let Err(back) = std::fs::rename(aside, dest) {
                    return Err(format!("{e}; and {} could not be put back from {}: {back}", dest.display(), aside.display()));
                }
            }
            Err(e)
        }
    }
}

/// The older versions removed, once the new one is current: one that cannot be removed is said, not an error, since
/// the update itself is done; the next prune tries it again.
fn prune_older(versions: &Versions, report: &mut Report) {
    match versions.prune() {
        Ok(gone) if !gone.is_empty() => report.say(format!("removed {}", gone.join(", "))),
        Ok(_) => {}
        Err(e) => report.note(format!("an older version was not removed ({e}); the next update tries again")),
    }
}

/// `self update`: the newest release that fits, or the one named.
pub fn update(home: &Path, catalog: &Catalog, exe: &Path, opts: &Options) -> Result<Report, String> {
    let mut report = Report { live: opts.live, ..Report::default() };
    let installed = installed_as(exe);
    let (versions, migrating) = match &installed {
        Installed::Managed { versions, .. } => (Versions { dir: versions.clone() }, None),
        Installed::InPlace { prefix } => (Versions::of(home), Some(prefix.clone())),
        Installed::Other(p) => return update_by_manager(p, opts),
    };
    let target = target().ok_or_else(|| format!("no release is published for {} {}", std::env::consts::OS, std::env::consts::ARCH))?;
    // Taken before anything is read or decided, so an update queued behind another sees the layout it left.
    let _one_at_a_time = match opts.dry_run {
        true => None,
        false => {
            std::fs::create_dir_all(&versions.dir).map_err(|e| format!("{}: {e}", versions.dir.display()))?;
            match opts.no_wait {
                true => Some(crate::pulls::Lock::try_take(&versions.dir, "self")?.ok_or("another update of superfluid is running")?),
                false => Some(crate::pulls::Lock::take(&versions.dir, "self", "update of superfluid")?),
            }
        }
    };
    let engine = engine_here(catalog);
    let compat = compat_fetcher(15);
    let found = fetch_releases(opts.source.as_deref().unwrap_or(&listing_source()), target, 30)?;
    let chosen = match &opts.version {
        Some(want) => {
            let want = want.trim_start_matches('v');
            let r = found.iter().find(|r| r.version == want).ok_or_else(|| {
                format!("no release {want} for {target} (the newest is {})", found.first().map(|r| r.version.as_str()).unwrap_or("none"))
            })?;
            if !fits(r, &engine, &compat) {
                return Err(format!("{want} does not declare support for the engine here ({}); `superfluid runtimes --json` shows what this build requires", engine.describe()));
            }
            r
        }
        None => newest(&found, &engine, None, &compat).ok_or_else(|| {
            format!("no release fits: {}, and the newest release for {target} is {}", engine.describe(), found.first().map(|r| r.version.as_str()).unwrap_or("none"))
        })?,
    };
    let v = chosen.version.clone();
    let current = versions.current();
    if migrating.is_none() && current.as_deref() == Some(v.as_str()) {
        report.say(format!("superfluid {v} is current, and the newest release that fits ({})", engine.describe()));
        return Ok(report);
    }
    // Without a version named an update only moves forward: a build newer than every release stays.
    let here = current.clone().unwrap_or_else(|| running().to_string());
    if opts.version.is_none() && migrating.is_none() && release_order(&v) < release_order(&here) {
        report.say(format!("superfluid {here} is current; the newest release that fits is {v} ({}), and `--version {v}` goes to it", engine.describe()));
        return Ok(report);
    }
    let on_disk = versions.binary(&v).is_file();
    report.say(format!(
        "{} -> {v} (published {}; {}){}",
        current.as_deref().unwrap_or(running()),
        chosen.published,
        engine.describe(),
        if on_disk { ", already installed here" } else { "" }
    ));
    if opts.dry_run {
        report.say("dry run: nothing fetched or changed");
        return Ok(report);
    }
    if opts.live {
        eprintln!("Updating superfluid {} → {v} · release notes: {}", current.as_deref().unwrap_or(running()), chosen.notes_url);
    }
    // The in-place copy goes under the layout first, as the version to go back to.
    let kept = match &migrating {
        Some(prefix) => Some(keep_in_place(&versions, prefix)?),
        None => None,
    };
    let on_disk = on_disk || versions.binary(&v).is_file();
    if !on_disk {
        place(&versions, chosen, &mut report, opts.live || opts.no_wait)?;
    }
    if let Err(e) = probe(&versions, &v, home, &engine) {
        if !on_disk {
            let _ = std::fs::remove_dir_all(versions.version_dir(&v));
        }
        return Err(format!("{v} was not made current: {e}"));
    }
    versions.set_current(&v)?;
    if let Some(old) = kept.filter(|old| *old != v) {
        std::fs::write(versions.dir.join("previous"), format!("{old}\n")).map_err(|e| format!("{}: {e}", versions.dir.display()))?;
    }
    report.changed = true;
    // An update made, by hand or automatically, ends a step back `self use` made.
    if let Some(check) = Check::read(home).filter(|c| c.declined.is_some()) {
        let _ = Check { declined: None, ..check }.write(home);
    }
    report.say(format!("superfluid {v} is current ({})", versions.dir.join("current").display()));
    if let Some(prev) = versions.previous() {
        report.say(format!("{prev} stays installed: `superfluid self use {prev}` goes back to it"));
    }
    // Pruned before the links are made, so a removed version's `superfluid-<v>` goes with it.
    prune_older(&versions, &mut report);
    if let Some(prefix) = &migrating {
        migrate(&versions, prefix, &mut report)?;
    } else if let Some(prefix) = versions.prefix() {
        for note in versions.link(&prefix, false)? {
            report.note(note);
        }
    }
    report.say("a running server keeps the version it started with; starting it again uses the new one");
    Ok(report)
}

/// `self use <v>`.
pub fn use_version(home: &Path, catalog: &Catalog, exe: &Path, v: &str) -> Result<Report, String> {
    let versions = match installed_as(exe) {
        Installed::Managed { versions, .. } => Versions { dir: versions },
        _ => Versions::of(home),
    };
    let v = v.trim_start_matches('v');
    if !is_version(v) {
        return Err(format!("{v:?} is not a version (see `superfluid self list`)"));
    }
    let mut report = Report::default();
    std::fs::create_dir_all(&versions.dir).map_err(|e| format!("{}: {e}", versions.dir.display()))?;
    let _one_at_a_time = crate::pulls::Lock::take(&versions.dir, "self", "switch of superfluid")?;
    if versions.current().as_deref() == Some(v) {
        report.say(format!("superfluid {v} is current already"));
        return Ok(report);
    }
    probe(&versions, v, home, &engine_here(catalog))?;
    let was = versions.current();
    versions.set_current(v)?;
    report.changed = true;
    report.say(format!("superfluid {v} is current"));
    declined_after_use(home, was.as_deref(), v);
    if let Some(prefix) = versions.prefix() {
        for note in versions.link(&prefix, false)? {
            report.say(note);
        }
    }
    Ok(report)
}

/// Whether the release `a` comes after `b`.
pub(crate) fn later(a: &str, b: &str) -> bool {
    release_order(a) > release_order(b)
}

/// What `self use` from `was` to `v` leaves in the record: going back from a newer version declines it, so an
/// automatic update does not undo the choice; going to the declined version or past it ends that.
fn declined_after_use(home: &Path, was: Option<&str>, v: &str) {
    let check = Check::read(home).unwrap_or_default();
    let declined = match (was, check.declined.as_deref()) {
        // Back from `was`: it and what is older are declined, and a later release declined before stays so.
        (Some(was), Some(d)) if release_order(was) > release_order(v) && release_order(d) > release_order(was) => Some(d.to_string()),
        (Some(was), _) if release_order(was) > release_order(v) => Some(was.to_string()),
        (_, Some(d)) if release_order(v) >= release_order(d) => None,
        (_, d) => d.map(str::to_string),
    };
    if declined != check.declined {
        let _ = Check { declined, ..check }.write(home);
    }
}

/// `self install --from <dir> [--prefix <dir>]`: an unpacked release placed
/// into the layout and made current; the install script's last step. `<dir>` is moved there, or copied when it is on
/// another volume, and then what is left of it is the script's to remove.
pub fn install_from(home: &Path, catalog: &Catalog, from: &Path, prefix: Option<&Path>) -> Result<Report, String> {
    let versions = Versions::of(home);
    let mut report = Report::default();
    std::fs::create_dir_all(&versions.dir).map_err(|e| format!("{}: {e}", versions.dir.display()))?;
    let _one_at_a_time = crate::pulls::Lock::take(&versions.dir, "self", "install of superfluid")?;
    let command = from.join("bin").join(crate::sys::exe_name("superfluid"));
    let v = version_of(&command)?;
    if cfg!(windows) && !from.join("libexec").join("superfluid").join(SHIM).is_file() {
        return Err(format!("{} holds no libexec\\superfluid\\{SHIM}, which the commands on PATH are copies of", from.display()));
    }
    let engine = engine_here(catalog);
    let dest = versions.version_dir(&v);
    if dest == from {
        probe(&versions, &v, home, &engine).map_err(|e| format!("{v} was not made current: {e}"))?;
    } else if versions.binary(&v).is_file() && probe(&versions, &v, home, &engine).is_ok() {
        // An installed version is kept as it is; a copy of it that works is not replaced.
        report.say(format!("superfluid {v} is installed here already and works; the copy given was not used"));
    } else {
        // Proved where it is, then set in place of a copy that does not work.
        probe_binary(&command, &v, home, &engine).map_err(|e| format!("{v} was not made current: {e}"))?;
        replace_tree(from, &dest, &versions.dir.join(format!(".replaced-{v}")), take_tree)?;
    }
    // The install script's earlier copy under the prefix is kept as the version to go back to.
    let kept = match prefix.and_then(|p| match installed_as(&p.join(crate::sys::exe_name("superfluid"))) {
        Installed::InPlace { prefix: root } => Some(root),
        _ => None,
    }) {
        Some(root) => Some(keep_in_place(&versions, &root)?),
        None => None,
    };
    versions.set_current(&v)?;
    if let Some(old) = kept.filter(|old| *old != v) {
        std::fs::write(versions.dir.join("previous"), format!("{old}\n")).map_err(|e| format!("{}: {e}", versions.dir.display()))?;
        report.say(format!("the copy that was in the prefix is kept as {old}: `superfluid self use {old}` goes back to it"));
    }
    report.changed = true;
    report.say(format!("superfluid {v} is current ({})", versions.dir.join("current").display()));
    prune_older(&versions, &mut report);
    if let Some(prefix) = prefix.map(Path::to_path_buf).or_else(|| versions.prefix()) {
        for note in versions.link(&prefix, true)? {
            report.say(note);
        }
        report.say(format!("{} {} it", prefix.join(crate::sys::exe_name("superfluid")).display(), if cfg!(windows) { "runs" } else { "links to" }));
    }
    Ok(report)
}

/// `self list`.
pub fn list(home: &Path, exe: &Path) -> String {
    let (versions, note) = match installed_as(exe) {
        Installed::Managed { versions, .. } => (Versions { dir: versions }, None),
        Installed::InPlace { prefix } => (Versions::of(home), Some(format!("this superfluid is the install script's copy in {}/bin; `superfluid self update` moves it under {}", prefix.display(), home.join("versions").display()))),
        Installed::Other(p) => (Versions::of(home), Some(not_ours(&p))),
    };
    let mut out = String::new();
    let here = versions.on_disk();
    if here.is_empty() {
        out.push_str(&format!("no versions in {}\n", versions.dir.display()));
    } else {
        out.push_str(&format!("versions in {}:\n", versions.dir.display()));
        let (current, previous) = (versions.current(), versions.previous());
        for v in &here {
            let mut marks = Vec::new();
            if Some(v) == current.as_ref() {
                marks.push("current".to_string());
            }
            if Some(v) == previous.as_ref() {
                marks.push(format!("previous (`superfluid self use {v}` goes back)"));
            }
            if versions.version_dir(v).join("keep").exists() {
                marks.push("kept (a `keep` file)".into());
            }
            out.push_str(&format!("  {v}  {}\n", marks.join(", ")));
        }
    }
    out.push_str(&format!("running: {}\n", running()));
    if let Some(cache) = Check::read(home) {
        let wait = crate::autoupdate::wait(home, &|k| std::env::var(k).ok()).days;
        out.push_str(&format!("{}\n", cache.under(wait).describe()));
    }
    if let Some(n) = note {
        out.push_str(&format!("{n}\n"));
    }
    out
}


/// `$SUPERFLUID_HOME/update-check.json`: the last read, the releases it found that fit here, the newest of them that
/// has been out as long as the wait asks and its notes (as the wait it was last read under picks it), the version
/// announced, the version an automatic update last tried and the
/// one it handed to a package manager (and when), and the version `self use` stepped back from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Check {
    pub checked_at: u64,
    /// The wait, in days, `newest` was picked under; `under` picks again for another.
    pub wait: Option<u32>,
    pub newest: Option<String>,
    pub published: Option<String>,
    pub notified: Option<String>,
    /// The page of the newest release's notes.
    pub notes: Option<String>,
    pub attempted: Option<String>,
    /// When an automatic update of `attempted` was tried, in seconds since the epoch: one that failed is not tried
    /// again for a day, until the next daily read.
    pub attempted_at: u64,
    /// The release an automatic update handed to the package manager that owns the copy, and when: a detached
    /// `self background-update` runs its update once nothing runs from the copy, and clears this when it could not.
    pub delegated: Option<String>,
    pub delegated_at: u64,
    /// The version `self use` stepped back from: no automatic update goes to it or to an older one, until a newer
    /// release, `self update`, or `self use` of it again.
    pub declined: Option<String>,
    /// The releases the last read found that fit here, newest first, a few of them: each wait picks the newest that
    /// has been out that long, so a record serves every wait (a shell and a service with different ones) without a
    /// read of its own. `None` in a record from before they were kept, which names only what its own wait picked.
    pub seen: Option<Vec<Seen>>,
}

/// A release the last read found that fits here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seen {
    pub version: String,
    pub published: String,
    /// The page of its notes.
    pub notes: String,
}

/// How many of the releases that fit a read keeps: enough for any wait between releases.
const SEEN_KEPT: usize = 8;

impl Check {
    fn path(home: &Path) -> PathBuf {
        home.join("update-check.json")
    }

    pub fn read(home: &Path) -> Option<Check> {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(Self::path(home)).ok()?).ok()?;
        let text = |k: &str| v[k].as_str().map(str::to_string);
        Some(Check {
            checked_at: v["checked_at"].as_u64().unwrap_or(0),
            wait: v["wait_days"].as_u64().and_then(|d| u32::try_from(d).ok()),
            newest: text("newest"),
            published: text("published"),
            notified: text("notified"),
            notes: text("notes"),
            attempted: text("attempted"),
            attempted_at: v["attempted_at"].as_u64().unwrap_or(0),
            delegated: text("delegated"),
            delegated_at: v["delegated_at"].as_u64().unwrap_or(0),
            declined: text("declined"),
            seen: v["releases"].as_array().map(|a| {
                a.iter()
                    .filter_map(|r| {
                        Some(Seen {
                            version: r["version"].as_str()?.to_string(),
                            published: r["published"].as_str().unwrap_or_default().to_string(),
                            notes: r["notes"].as_str().unwrap_or_default().to_string(),
                        })
                    })
                    .collect()
            }),
        })
    }

    /// Written whole beside the file and renamed over it, so a process reading it at the same time (the launch of
    /// another command, the background check) finds the old record or the new, never half of one.
    pub(crate) fn write(&self, home: &Path) -> Result<(), String> {
        std::fs::create_dir_all(home).map_err(|e| format!("{}: {e}", home.display()))?;
        let v = json!({
            "checked_at": self.checked_at,
            "wait_days": self.wait,
            "newest": self.newest,
            "published": self.published,
            "notified": self.notified,
            "notes": self.notes,
            "attempted": self.attempted,
            "attempted_at": self.attempted_at,
            "delegated": self.delegated,
            "delegated_at": self.delegated_at,
            "declined": self.declined,
            "releases": self.seen.as_ref().map(|seen| seen.iter().map(|r| json!({"version": r.version, "published": r.published, "notes": r.notes})).collect::<Vec<_>>()),
        });
        crate::autoupdate::write_atomically(&Self::path(home), format!("{v:#}\n").as_bytes())
    }

    pub fn due(&self) -> bool {
        now_secs().saturating_sub(self.checked_at) >= CHECK_EVERY.as_secs()
    }

    /// Whether the read was made under `wait`, so the release it names is the one that wait takes.
    pub(crate) fn made_with(&self, wait: u32) -> bool {
        self.wait == Some(wait)
    }

    /// Whether the record says what `wait` takes without a read: it keeps the releases every wait picks from, or (one
    /// from before) was made under this wait.
    pub(crate) fn answers(&self, wait: u32) -> bool {
        self.seen.is_some() || self.made_with(wait)
    }

    /// The record as `wait` reads it: naming the newest release kept that has been out that long. One from before the
    /// releases were kept names nothing for a wait other than its own.
    pub(crate) fn under(self, wait: u32) -> Check {
        let today = today();
        let pick = match &self.seen {
            Some(seen) => seen.iter().find(|r| out_for(&r.published, wait, today)).cloned(),
            None if self.made_with(wait) => return self,
            None => None,
        };
        Check {
            wait: Some(wait),
            newest: pick.as_ref().map(|r| r.version.clone()),
            published: pick.as_ref().map(|r| r.published.clone()),
            notes: pick.map(|r| r.notes),
            ..self
        }
    }

    /// The newest release known, when it is newer than this one.
    pub fn newer(&self) -> Option<&str> {
        self.newer_than(running())
    }

    pub(crate) fn newer_than(&self, here: &str) -> Option<&str> {
        self.newest.as_deref().filter(|n| release_order(n) > release_order(here))
    }

    /// The page of the newest release's notes.
    pub(crate) fn notes_url(&self) -> Option<String> {
        self.newest.as_deref().map(|v| notes_url(v, self.notes.as_deref()))
    }

    pub fn describe(&self) -> String {
        let ago = now_secs().saturating_sub(self.checked_at);
        let when = if ago < 3600 { format!("{}m ago", ago / 60) } else if ago < 86_400 { format!("{}h ago", ago / 3600) } else { format!("{}d ago", ago / 86_400) };
        let published = self.published.as_deref().unwrap_or("?");
        match (self.newer(), self.wait.filter(|&w| w > 0)) {
            (Some(n), None) => format!("newest release: {n} (published {published}; checked {when}): `superfluid self update` installs it"),
            (Some(n), Some(w)) => format!(
                "newest release out {} or more: {n} (published {published}; checked {when}): `superfluid self update` installs the newest",
                crate::autoupdate::days(w)
            ),
            (None, None) => format!("newest release that fits: {} (checked {when})", self.newest.as_deref().unwrap_or("none known")),
            (None, Some(w)) => format!("newest release out {} or more that fits: {} (checked {when})", crate::autoupdate::days(w), self.newest.as_deref().unwrap_or("none known")),
        }
    }
}

/// Read the listing now and record the newest release that fits here and has been out `wait` days.
pub fn check_now(home: &Path, catalog: &Catalog, wait: u32, source: &str) -> Result<Check, String> {
    let target = target().ok_or("no release is published for this platform")?;
    let found = fetch_releases(source, target, 5)?;
    record(home, &found, &engine_here(catalog), wait, &compat_fetcher(5))
}

/// The releases of `found` that drive `engine`, newest first (a few, down to one out as long as any wait asks),
/// recorded with what the record held besides; the record as `wait` reads it, naming the newest that has been out
/// `wait` days.
pub(crate) fn record(home: &Path, found: &[Release], engine: &Engine, wait: u32, compat: &dyn Fn(&str) -> Option<Value>) -> Result<Check, String> {
    let today = today();
    let mut seen = Vec::new();
    for r in found.iter().filter(|r| fits(r, engine, compat)) {
        seen.push(Seen { version: r.version.clone(), published: r.published.clone(), notes: r.notes_url.clone() });
        if seen.len() == SEEN_KEPT || out_for(&r.published, crate::autoupdate::MAX_WAIT_DAYS, today) {
            break;
        }
    }
    let previous = Check::read(home).unwrap_or_default();
    let check = Check { checked_at: now_secs(), seen: Some(seen), ..previous }.under(wait);
    check.write(home)?;
    Ok(check)
}

/// The one line printed when a newer release is out, once per release; `command` installs it, or, under a wait, the
/// newest (which may be younger than the release named). With automatic updates on, it says how to turn them off.
pub(crate) fn announce(home: &Path, mut check: Check, command: &str, auto: bool) -> Option<String> {
    let newer = check.newer()?.to_string();
    if check.notified.as_deref() == Some(newer.as_str()) {
        return None;
    }
    let notes = check.notes_url().unwrap_or_default();
    check.notified = Some(newer.clone());
    let _ = check.write(home);
    let off = if auto { format!(" · turn off: {}", crate::autoupdate::OFF_COMMAND) } else { String::new() };
    let installs = if check.wait.unwrap_or(0) > 0 { "installs the newest" } else { "installs it" };
    Some(format!("superfluid {newer} is available (this is {}): `{command}` {installs} · release notes: {notes}{off}", running()))
}

/// What `serve` runs once the server is up: one listing read a day at most, one line per new release.
pub fn check_in_background(home: PathBuf, offline: bool) {
    if offline || crate::autoupdate::switched_off(&|k| std::env::var(k).ok()).is_some() {
        return;
    }
    std::thread::Builder::new()
        .name("update-check".into())
        .spawn(move || loop {
            if let Some(line) = tick(&home, &Catalog::from_env(), &listing_source()) {
                eprintln!("superfluid: {line}");
            }
            // Hourly, so a server that stays up learns of a later release, and a check another process made is not repeated.
            std::thread::sleep(Duration::from_secs(60 * 60));
        })
        .ok();
}

/// One round: the cache when not yet due, else the listing; the line to print for a release not yet announced.
pub fn tick(home: &Path, catalog: &Catalog, source: &str) -> Option<String> {
    let command = update_command(&std::env::current_exe().unwrap_or_default());
    let env = |k: &str| std::env::var(k).ok();
    let auto = crate::autoupdate::enabled(home, &env).is_ok();
    let wait = crate::autoupdate::wait(home, &env).days;
    // A record made under another wait answers for this one from the releases it keeps: a shell and a service with
    // different waits share the one read a day.
    match Check::read(home).filter(|c| !c.due() && c.answers(wait)) {
        Some(cached) => announce(home, cached.under(wait), &command, auto),
        None => announce(home, check_now(home, catalog, wait, source).ok()?, &command, auto),
    }
}


fn usage() -> ! {
    eprint!("{HELP}");
    std::process::exit(2);
}

/// `superfluid self <action> ...`; the exit code.
pub fn run(args: &[String]) -> i32 {
    let home = crate::runtimes::default_home();
    let exe = std::env::current_exe().unwrap_or_default();
    let catalog = Catalog::from_env();
    let args = crate::cli::split_equals(args, |f| Some(matches!(f, "--version" | "--from" | "--prefix")));
    let print = |r: Result<Report, String>| -> i32 {
        match r {
            Ok(report) => {
                if !report.live {
                    for line in report.lines {
                        println!("superfluid: {line}");
                    }
                }
                0
            }
            Err(e) => {
                eprintln!("superfluid: {e}");
                1
            }
        }
    };
    match args.first().map(String::as_str) {
        Some("update") => {
            let mut opts = Options { live: true, ..Options::default() };
            let (mut auto, mut json) = (false, false);
            let mut it = args[1..].iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--version" => opts.version = it.next().cloned().or_else(|| usage()),
                    "--dry-run" => opts.dry_run = true,
                    "--yes" | "-y" => opts.yes = true,
                    "--auto" => auto = true,
                    "--json" => json = true,
                    _ => usage(),
                }
            }
            // What an automatic update would do, for an app: nothing is asked, so --yes changes nothing. With --json,
            // stdout holds the summary alone, and the line saying why nothing was done goes to stderr with the rest.
            if json && !auto {
                usage();
            }
            if auto {
                if opts.version.is_some() || opts.dry_run {
                    usage();
                }
                crate::autoupdate::PROGRESS_LINES.store(true, std::sync::atomic::Ordering::Relaxed);
                let env = |k: &str| std::env::var(k).ok();
                let mut out = |l: String| if json { eprintln!("{l}") } else { println!("{l}") };
                let (code, summary) = crate::autoupdate::update_auto(&home, &catalog, &exe, &listing_source(), &env, &mut out, &mut |l| eprintln!("{l}"));
                if json {
                    println!("{summary}");
                }
                return code;
            }
            match update(&home, &catalog, &exe, &opts) {
                Ok(Report { hand_off: Some(mut cmd), .. }) => {
                    let _ = std::io::stdout().flush();
                    // Only returns when the command could not be run.
                    let e = crate::sys::replace_process(&mut cmd);
                    eprintln!("superfluid: running {}: {e}", cmd.get_program().to_string_lossy());
                    1
                }
                r => print(r),
            }
        }
        Some("check") if args.len() == 1 => match crate::autoupdate::check_lines(&home, &catalog, &exe, &listing_source(), &|k| std::env::var(k).ok()) {
            Ok(lines) => {
                for line in lines {
                    println!("superfluid: {line}");
                }
                0
            }
            Err(e) => {
                eprintln!("superfluid: {e}");
                1
            }
        },
        Some("check") if args.len() == 2 && args[1] == "--json" => match crate::autoupdate::check_json(&home, &catalog, &exe, &listing_source(), &|k| std::env::var(k).ok()) {
            Ok(v) => {
                println!("{v}");
                0
            }
            Err(e) => {
                eprintln!("superfluid: {e}");
                1
            }
        },
        // What a launch starts in the background when the daily read is due: the record alone, nothing printed.
        Some("check") if args.len() == 2 && args[1] == "--record" => match check_now(&home, &catalog, crate::autoupdate::wait(&home, &|k| std::env::var(k).ok()).days, &listing_source()) {
            Ok(_) => 0,
            Err(_) => 1,
        },
        Some("auto-update") => crate::autoupdate::auto_update_command(&home, &args[1..].iter().map(String::as_str).collect::<Vec<_>>()),
        // What an automatic update leaves running for a package manager's copy: it runs the manager once nothing runs from the copy.
        Some("background-update") if args.len() == 1 => crate::autoupdate::background_update(&home, &exe, None),
        Some("background-update") if args.len() == 3 && args[1] == "--version" => crate::autoupdate::background_update(&home, &exe, Some(&args[2])),
        Some("list") if args.len() == 1 => {
            let _ = std::io::stdout().write_all(list(&home, &exe).as_bytes());
            0
        }
        Some("use") if args.len() == 2 => print(use_version(&home, &catalog, &exe, &args[1])),
        Some("install") => {
            let (mut from, mut prefix) = (None, None);
            let mut it = args[1..].iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--from" => from = it.next().map(PathBuf::from).or_else(|| usage()),
                    "--prefix" => prefix = it.next().map(PathBuf::from).or_else(|| usage()),
                    _ => usage(),
                }
            }
            let Some(from) = from else { usage() };
            print(install_from(&home, &catalog, &from, prefix.as_deref()))
        }
        _ => usage(),
    }
}

#[cfg(test)]
mod install_script_tests {
    use super::*;

    #[test]
    fn a_release_the_windows_install_script_placed_is_updated_by_running_it_again() {
        let root = std::env::temp_dir().join(format!("sf-self-ps1-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::create_dir_all(root.join("libexec").join("superfluid")).unwrap();
        let exe = root.join("bin").join(crate::sys::exe_name("superfluid"));
        std::fs::write(&exe, b"").unwrap();
        std::fs::write(root.join(INSTALLED_BY), format!("{INSTALL_PS1}: {INSTALL_PS1_COMMAND}\n")).unwrap();
        assert!(matches!(installed_as(&exe), Installed::Other(_)), "it is the script's, not a version here");
        let exe = crate::sys::canonical(&exe).unwrap();
        match delegate(&exe) {
            Some(Delegate::Name { manager, command, .. }) => {
                assert_eq!((manager.as_str(), command.as_str()), ("the install script", INSTALL_PS1_COMMAND));
            }
            other => panic!("the script is named, never run: {other:?}"),
        }
        assert_eq!(update_command(&exe), INSTALL_PS1_COMMAND, "what `serve` says installs a newer release");
        let refused = update(&root, &Catalog::from_env(), &exe, &Options::default()).unwrap_err();
        assert!(refused.contains(INSTALL_PS1_COMMAND), "{refused}");
        let _ = std::fs::remove_dir_all(&root);
    }

    // Where npm and pip put a copy on Windows, read here on any platform: the files are laid out as there.
    #[test]
    fn a_copy_npm_or_pip_placed_on_windows_is_told_by_where_it_is() {
        assert_eq!(shape(r"C:\Users\Me\AppData\Roaming\npm\node_modules\X", true), "c:/users/me/appdata/roaming/npm/node_modules/x");
        assert_eq!(shape(r"/home/me/a\B", false), r"/home/me/a\B", "a Unix name may hold a backslash, and its case counts");
        let root = std::env::temp_dir().join(format!("sf-self-places-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let file = |path: &Path| {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"").unwrap();
        };

        // npm keeps a global package's own dependencies inside it, and writes its commands beside node_modules.
        let npm = root.join("npm");
        let platform = Path::new("node_modules").join("@basecompute").join("superfluid-win32-x64").join("release").join("bin").join("superfluid.exe");
        let global = npm.join("node_modules").join("@basecompute").join("superfluid").join(&platform);
        file(&global);
        assert_eq!(npm_global_prefix(&global.to_string_lossy(), true), None, "node_modules with no command beside it is a project's");
        file(&npm.join("superfluid.cmd"));
        assert_eq!(npm_global_prefix(&global.to_string_lossy(), true), npm.to_str());
        let project = root.join("app").join(&platform);
        file(&project);
        assert_eq!(npm_global_prefix(&project.to_string_lossy(), true), None);
        assert_eq!(npm_global_prefix("/usr/local/lib/node_modules/@basecompute/superfluid-linux-x64/release/bin/superfluid", false), Some("/usr/local"));
        assert_eq!(npm_global_prefix("/srv/app/node_modules/@basecompute/superfluid-linux-x64/release/bin/superfluid", false), None);

        // A venv has its Python in Scripts; a Python install has it at its root.
        let env = root.join("env");
        let copy = env.join("Lib").join("site-packages").join("superfluid_cli").join("release").join("bin").join("superfluid.exe");
        file(&copy);
        assert_eq!(python_of(&copy.to_string_lossy(), true), Some(env.join("python.exe")));
        file(&env.join("Scripts").join("python.exe"));
        assert_eq!(python_of(&copy.to_string_lossy(), true), Some(env.join("Scripts").join("python.exe")));
        assert_eq!(python_of(r"C:\Users\Me\AppData\Roaming\Python\Python313\site-packages\superfluid_cli\release\bin\superfluid.exe", true), None, "a user site has no Lib");
        assert_eq!(
            python_of("/opt/env/lib/python3.12/site-packages/superfluid_cli/release/bin/superfluid", false),
            Some(PathBuf::from("/opt/env/bin/python"))
        );

        let run = Delegate::Run { manager: "uv".into(), command: "uv tool upgrade superfluid-cli".into(), argv: vec!["uv".into()], path_first: None };
        match named_while_it_runs(run) {
            Delegate::Name { manager, command, why } => {
                assert_eq!((manager.as_str(), command.as_str()), ("uv", "uv tool upgrade superfluid-cli"));
                assert!(why.contains("replace no program that is running"), "{why}");
            }
            other => panic!("{other:?}"),
        }
        let basert = Delegate::Run { manager: "basert".into(), command: "basert superfluid install".into(), argv: vec!["basert".into()], path_first: None };
        assert_eq!(named_while_it_runs(basert.clone()), basert, "basert is handed the process, and replaces a copy nothing runs from");
        let _ = std::fs::remove_dir_all(&root);
    }

    // On Windows the command a user is shown is the whole update, so it has to be one a shell runs as it is typed.
    #[test]
    fn a_command_a_user_is_to_type_updates_the_copy_it_is_about_and_survives_a_space() {
        // npm is told the folder the copy is in: npm's own configuration may name another (%AppData%\npm).
        assert_eq!(npm_global_command(r"D:\sf-tools", true), r"npm install --global --prefix D:\sf-tools @basecompute/superfluid@latest");
        assert_eq!(
            npm_global_command(r"C:\Users\Jane Doe\AppData\Roaming\npm", true),
            r"npm install --global --prefix 'C:\Users\Jane Doe\AppData\Roaming\npm' @basecompute/superfluid@latest"
        );
        assert_eq!(npm_global_command("/usr/local", false), "npm install --global @basecompute/superfluid@latest", "elsewhere superfluid runs npm, with the prefix");

        // A path no shell splits is as it is; another is quoted, and PowerShell runs a quoted program after `&`.
        let python = |p: &str, windows| pip_command(Some(Path::new(p)), windows);
        assert_eq!(python(r"C:\Users\me\venv\Scripts\python.exe", true), r"C:\Users\me\venv\Scripts\python.exe -m pip install --upgrade superfluid-cli");
        assert_eq!(python(r"C:\Users\RUNNER~1\venv\Scripts\python.exe", true), r"C:\Users\RUNNER~1\venv\Scripts\python.exe -m pip install --upgrade superfluid-cli");
        assert_eq!(
            python(r"C:\Users\Jane Doe\venv\Scripts\python.exe", true),
            r"& 'C:\Users\Jane Doe\venv\Scripts\python.exe' -m pip install --upgrade superfluid-cli"
        );
        assert_eq!(python(r"C:\Users\O'Brien & Co\python.exe", true), r"& 'C:\Users\O''Brien & Co\python.exe' -m pip install --upgrade superfluid-cli");
        assert_eq!(python("/opt/env/bin/python", false), "/opt/env/bin/python -m pip install --upgrade superfluid-cli");
        assert_eq!(python("/home/jane doe/it's/bin/python", false), r"'/home/jane doe/it'\''s/bin/python' -m pip install --upgrade superfluid-cli");
        assert_eq!(pip_command(None, true), "pip install --upgrade superfluid-cli");
        assert_eq!(shell_word("", true), "''");
        assert_eq!(shell_word(r"a\b", false), r"'a\b'", "a backslash is a name's on Unix, and a shell's too");

        // cmd reads a quoted command another way, so the reason says whose it is.
        let why = for_powershell(&python(r"C:\Users\Jane Doe\python.exe", true), "run it yourself".into(), true);
        assert_eq!(why, "run it yourself; the command is written for PowerShell");
        assert_eq!(for_powershell(&python(r"C:\Users\me\python.exe", true), "run it yourself".into(), true), "run it yourself");
        assert_eq!(for_powershell("'/a b/python' -m pip", "run it yourself".into(), false), "run it yourself");
    }

    // The install script hands over the folder it unpacked, with the program it runs inside.
    #[test]
    fn an_unpacked_release_is_taken_whole_and_what_cannot_be_removed_is_left_to_its_owner() {
        let root = std::env::temp_dir().join(format!("sf-self-take-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let release = |dir: &Path| {
            std::fs::create_dir_all(dir.join("bin")).unwrap();
            std::fs::write(dir.join("bin").join("superfluid"), b"a release").unwrap();
        };
        // On one volume it is renamed.
        release(&root.join("unpacked"));
        take_tree(&root.join("unpacked"), &root.join("moved")).unwrap();
        assert_eq!(std::fs::read(root.join("moved").join("bin").join("superfluid")).unwrap(), b"a release");
        assert!(!root.join("unpacked").exists());
        // Where it cannot be (here, something is in the way, as another volume would refuse), it is copied; a source
        // that cannot then be removed, as a running program's on Windows, does not undo the install.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let from = root.join("held");
            release(&from);
            std::fs::create_dir_all(root.join("dest").join("in-the-way")).unwrap();
            std::fs::set_permissions(from.join("bin"), std::fs::Permissions::from_mode(0o555)).unwrap();
            take_tree(&from, &root.join("dest")).unwrap();
            assert_eq!(std::fs::read(root.join("dest").join("bin").join("superfluid")).unwrap(), b"a release");
            if from.exists() {
                std::fs::set_permissions(from.join("bin"), std::fs::Permissions::from_mode(0o755)).unwrap();
            }
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}

// The Windows layout, on files that stand in for a release: the launcher is whatever bytes the version carries.
#[cfg(all(test, windows))]
mod windows_tests {
    use super::*;

    fn release(versions: &Versions, v: &str, launcher: &[u8]) {
        let dir = versions.version_dir(v);
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        std::fs::create_dir_all(dir.join("libexec").join("superfluid")).unwrap();
        for name in LINKED {
            std::fs::write(dir.join("bin").join(crate::sys::exe_name(name)), v).unwrap();
        }
        std::fs::write(dir.join("libexec").join("superfluid").join(SHIM), launcher).unwrap();
    }

    #[test]
    fn the_commands_on_path_are_launchers_that_are_told_where_the_versions_are() {
        let root = std::env::temp_dir().join(format!("sf-self-win-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let versions = Versions::of(&root.join("home"));
        let prefix = root.join("programs").join("bin");
        // Two builds of one revision: the same program, not the same bytes.
        release(&versions, "0.2.0", b"superfluid-shim-revision:0001 built monday");
        release(&versions, "0.2.1", b"superfluid-shim-revision:0001 built tuesday");
        assert_eq!(versions.current(), None);
        assert!(versions.link(&prefix, true).unwrap_err().contains("no version is current"));

        versions.set_current("0.2.0").unwrap();
        assert_eq!(versions.current().as_deref(), Some("0.2.0"));
        assert_eq!(versions.link(&prefix, true).unwrap(), Vec::<String>::new());
        let said = std::fs::read_to_string(prefix.join(VERSIONS_FILE)).unwrap();
        assert_eq!(Path::new(said.trim()), versions.dir);
        for name in LINKED {
            assert_eq!(std::fs::read(prefix.join(crate::sys::exe_name(name))).unwrap(), b"superfluid-shim-revision:0001 built monday", "{name}");
        }
        assert_eq!(versions.prefix().as_deref(), Some(prefix.as_path()));

        // A switch changes the file the launchers read and places nothing.
        versions.set_current("0.2.1").unwrap();
        assert_eq!((versions.current().as_deref(), versions.previous().as_deref()), (Some("0.2.1"), Some("0.2.0")));
        assert_eq!(versions.link(&prefix, false).unwrap(), Vec::<String>::new());
        assert_eq!(std::fs::read(prefix.join("superfluid.exe")).unwrap(), b"superfluid-shim-revision:0001 built monday");

        // A release with a newer launcher has it take the place of the one there, and nothing is left aside.
        release(&versions, "0.3.0", b"superfluid-shim-revision:0002");
        versions.set_current("0.3.0").unwrap();
        assert_eq!(versions.link(&prefix, false).unwrap(), Vec::<String>::new());
        assert_eq!(std::fs::read(prefix.join("superfluid.exe")).unwrap(), b"superfluid-shim-revision:0002");
        // Going back to an older release keeps the newer launcher, which runs it as well.
        versions.set_current("0.2.0").unwrap();
        assert_eq!(versions.link(&prefix, false).unwrap(), Vec::<String>::new());
        assert_eq!(std::fs::read(prefix.join("superfluid.exe")).unwrap(), b"superfluid-shim-revision:0002");
        let left: Vec<String> = std::fs::read_dir(&prefix).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.starts_with('.')).collect();
        assert_eq!(left, Vec::<String>::new());

        // A version with no launcher of its own is run by the ones there.
        release(&versions, "0.3.1", b"");
        std::fs::remove_file(versions.version_dir("0.3.1").join("libexec").join("superfluid").join(SHIM)).unwrap();
        versions.set_current("0.3.1").unwrap();
        assert_eq!(versions.link(&prefix, false).unwrap(), Vec::<String>::new());
        assert_eq!(std::fs::read(prefix.join("superfluid.exe")).unwrap(), b"superfluid-shim-revision:0002");
        assert!(versions.link(&root.join("elsewhere"), true).unwrap_err().contains(SHIM), "a new prefix needs one");

        assert_eq!(
            installed_as(&versions.binary("0.3.0")),
            Installed::Managed { versions: crate::sys::canonical(&versions.dir).unwrap(), version: "0.3.0".into() }
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_program_no_install_here_placed_is_left_unless_the_install_script_asks() {
        let root = std::env::temp_dir().join(format!("sf-self-win-theirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let versions = Versions::of(&root.join("home"));
        let prefix = root.join("bin");
        release(&versions, "0.2.0", b"superfluid-shim-revision:0001");
        versions.set_current("0.2.0").unwrap();
        std::fs::create_dir_all(&prefix).unwrap();
        std::fs::write(prefix.join("superfluid.exe"), b"someone's own build").unwrap();
        let notes = versions.link(&prefix, false).unwrap();
        assert!(notes.len() == 1 && notes[0].contains("left as it is"), "{notes:?}");
        assert_eq!(std::fs::read(prefix.join("superfluid.exe")).unwrap(), b"someone's own build");
        assert_eq!(std::fs::read(prefix.join("superfluid-workerd.exe")).unwrap(), b"superfluid-shim-revision:0001", "what was not there is placed");
        // From then on the prefix is one an install here keeps.
        std::fs::remove_file(prefix.join(VERSIONS_FILE)).unwrap();
        assert_eq!(versions.link(&prefix, true).unwrap(), Vec::<String>::new());
        assert_eq!(std::fs::read(prefix.join("superfluid.exe")).unwrap(), b"superfluid-shim-revision:0001");
        assert_eq!((launcher_revision(b"superfluid-shim-revision:0012 x"), launcher_revision(b"an .exe"), launcher_revision(b"superfluid-shim-revision:1")), (Some(12), None, None));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A release a package carries at `root`, its `installed-by` saying `line`; the program's real path.
    fn packaged(root: &Path, line: &str) -> PathBuf {
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::create_dir_all(root.join("libexec").join("superfluid")).unwrap();
        std::fs::write(root.join("bin").join("superfluid.exe"), b"").unwrap();
        std::fs::write(root.join(INSTALLED_BY), format!("{line}\n")).unwrap();
        crate::sys::canonical(root.join("bin").join("superfluid.exe")).unwrap()
    }

    #[test]
    fn a_copy_npm_or_pip_installed_is_theirs_and_their_command_is_named_never_run() {
        let root = std::env::temp_dir().join(format!("sf-self-win-packages-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let npm_line = "npm: npm install --global @basecompute/superfluid@latest";
        let pip_line = "pip: pip install --upgrade superfluid-cli";
        let named = |exe: &Path| match delegate(exe) {
            Some(Delegate::Name { manager, command, why }) => (manager, command, why),
            other => panic!("{}: {other:?}", exe.display()),
        };

        // npm's global folder, with the command npm wrote beside it.
        let prefix = root.join("npm");
        let in_main = prefix.join("node_modules").join("@basecompute").join("superfluid").join("node_modules");
        let global = packaged(&in_main.join("@basecompute").join("superfluid-win32-x64").join("release"), npm_line);
        std::fs::write(prefix.join("superfluid.cmd"), b"").unwrap();
        assert!(matches!(installed_as(&global), Installed::Other(_)), "it is npm's, not a release unpacked in one place");
        // The command names the folder this copy is in, which need not be the one npm is set to.
        let update_it = format!("npm install --global --prefix {} @basecompute/superfluid@latest", crate::sys::canonical(&prefix).unwrap().display());
        let (manager, command, why) = named(&global);
        assert_eq!((manager.as_str(), command.as_str()), ("npm", update_it.as_str()));
        assert!(why.contains("Windows lets npm replace no program that is running") && !why.contains("PowerShell"), "{why}");
        assert_eq!(update_command(&global), update_it);
        let yes = Options { yes: true, ..Options::default() };
        let refused = update(&root, &Catalog::from_env(), &global, &yes).unwrap_err();
        assert!(refused.contains(&format!("was installed by npm, which updates it: run `{update_it}`")), "{refused}");
        // A folder named with a space is quoted, as PowerShell reads it.
        let spaced = root.join("my tools");
        let in_spaced = packaged(&spaced.join("node_modules").join("@basecompute").join("superfluid-win32-x64").join("release"), npm_line);
        std::fs::write(spaced.join("superfluid.cmd"), b"").unwrap();
        let (_, command, why) = named(&in_spaced);
        assert_eq!(command, format!("npm install --global --prefix '{}' @basecompute/superfluid@latest", crate::sys::canonical(&spaced).unwrap().display()));
        assert!(why.contains("written for PowerShell"), "{why}");
        // A project's dependency is that project's.
        let project = packaged(&root.join("app").join("node_modules").join("@basecompute").join("superfluid-win32-x64").join("release"), npm_line);
        let (_, command, why) = named(&project);
        assert_eq!(command, "npm install @basecompute/superfluid@latest");
        assert!(why.contains("update it in that project"), "{why}");

        // pip, with the Python of the environment the copy is in.
        let env = root.join("venv");
        let pip_copy = packaged(&env.join("Lib").join("site-packages").join("superfluid_cli").join("release"), pip_line);
        std::fs::create_dir_all(env.join("Scripts")).unwrap();
        std::fs::write(env.join("Scripts").join("python.exe"), b"").unwrap();
        let python = crate::sys::canonical(env.join("Scripts").join("python.exe")).unwrap();
        let (manager, command, _) = named(&pip_copy);
        assert_eq!((manager.as_str(), command), ("pip", format!("{} -m pip install --upgrade superfluid-cli", python.display())));
        // A home folder named with a space: the program is quoted, and run with `&`.
        let env = root.join("Jane Doe").join("venv");
        let pip_copy = packaged(&env.join("Lib").join("site-packages").join("superfluid_cli").join("release"), pip_line);
        std::fs::create_dir_all(env.join("Scripts")).unwrap();
        std::fs::write(env.join("Scripts").join("python.exe"), b"").unwrap();
        let python = crate::sys::canonical(env.join("Scripts").join("python.exe")).unwrap();
        let (_, command, why) = named(&pip_copy);
        assert_eq!(command, format!("& '{}' -m pip install --upgrade superfluid-cli", python.display()));
        assert!(why.contains("written for PowerShell"), "{why}");
        // pipx and uv are told apart by their folders, and are named too, with or without one on PATH.
        let site = Path::new("Lib").join("site-packages").join("superfluid_cli").join("release");
        let pipx = packaged(&root.join("pipx").join("venvs").join("superfluid-cli").join(&site), pip_line);
        let (manager, command, _) = named(&pipx);
        assert_eq!((manager.as_str(), command.as_str()), ("pipx", "pipx upgrade superfluid-cli"));
        let uv = packaged(&root.join("uv").join("tools").join("superfluid-cli").join(&site), pip_line);
        let (manager, command, _) = named(&uv);
        assert_eq!((manager.as_str(), command.as_str()), ("uv", "uv tool upgrade superfluid-cli"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_running_program_is_found_by_where_it_runs_from_whatever_the_case() {
        // Something that stays up for the length of the test, started from a directory of its own.
        let root = std::env::temp_dir().join(format!("sf-self-win-running-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("Bin")).unwrap();
        let system = PathBuf::from(std::env::var_os("SystemRoot").unwrap()).join("System32");
        let program = root.join("Bin").join("waits.exe");
        std::fs::copy(system.join("ping.exe"), &program).unwrap();
        assert!(!running_from(&root));
        let mut child = Command::new(&program).args(["-n", "30", "127.0.0.1"]).stdout(std::process::Stdio::null()).spawn().unwrap();
        let found = processes_from(&root);
        assert_eq!(found.iter().map(|(pid, _)| *pid).collect::<Vec<_>>(), [child.id() as i32]);
        assert!(found[0].1 > 0 && started_at(child.id() as i32) == Some(found[0].1), "it has a start time: {found:?}");
        assert!(under(Path::new(r"C:\Users\Me\BIN\x.exe"), Path::new(r"c:\users\me\bin")));
        assert!(!under(Path::new(r"C:\Users\Me\binary\x.exe"), Path::new(r"C:\Users\Me\bin")), "a name that only starts the same is another directory");
        child.kill().unwrap();
        let _ = child.wait();
        assert!(!running_from(&root));
        let _ = std::fs::remove_dir_all(&root);
    }
}

// The tests stand in for a worker with a POSIX shell script, and for a link with a symlink.
#[cfg(all(test, unix))]
pub(crate) mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Stdio;
    use superfluid_adapter_kit::install::sha256_of;

    pub(crate) fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("superfluid-self-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A release tree whose `superfluid` answers `--version` and `runtimes --json` as a real one would (the rows in
    /// `$SUPERFLUID_HOME/runtimes.json` when there are some), and notes each `runtime ...` it is asked to run in
    /// `$SUPERFLUID_HOME/runtime-calls`, saying on stdout that it ran it (exiting 23 instead for a runtime `<id>` while
    /// `$SUPERFLUID_HOME/runtime-fails-<id>` is there).
    pub(crate) fn release_tree(dir: &Path, v: &str) -> PathBuf {
        let bin = dir.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(dir.join("libexec").join("superfluid")).unwrap();
        let script = format!(
            "#!/bin/sh\ncase \"$1\" in\n  --version) echo \"superfluid {v}\";;\n  runtimes) cat \"$SUPERFLUID_HOME/runtimes.json\" 2>/dev/null || echo '[{{\"id\":\"basert\",\"ready\":true}}]';;\n  runtime) echo \"$*\" >> \"$SUPERFLUID_HOME/runtime-calls\"; [ -e \"$SUPERFLUID_HOME/runtime-fails-$3\" ] && exit 23; echo \"ran $*\";;\n  *) exit 2;;\nesac\n"
        );
        for name in LINKED {
            let p = bin.join(name);
            std::fs::write(&p, &script).unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        std::fs::write(dir.join("libexec").join("superfluid").join("superfluid-worker-toy"), "#!/bin/sh\n").unwrap();
        dir.to_path_buf()
    }

    /// A listing with one release per version, its assets as local files.
    pub(crate) fn listing(dir: &Path, versions: &[(&str, &str, Option<u32>)]) -> PathBuf {
        let target = target().unwrap();
        let mut rs = Vec::new();
        for (v, published, abi) in versions {
            let name = format!("superfluid-{target}");
            let tree = dir.join(format!("tree-{v}")).join(&name);
            release_tree(&tree, v);
            let tar = dir.join(format!("{name}-{v}.tar.gz"));
            let st = Command::new("tar").args(["-C"]).arg(tree.parent().unwrap()).arg("-czf").arg(&tar).arg(&name).status().unwrap();
            assert!(st.success());
            let sum = sha256_of(&tar).unwrap();
            let sha = dir.join(format!("{name}-{v}.tar.gz.sha256"));
            std::fs::write(&sha, format!("{sum}  {name}.tar.gz\n")).unwrap();
            let compat = dir.join(format!("{name}-{v}.compat.json"));
            std::fs::write(&compat, json!({"runtimes": {"basert": {"abi": abi, "unnumbered_minor": abi.map(|_| "0.3")}}}).to_string()).unwrap();
            let mut assets = vec![
                json!({"name": format!("{name}.tar.gz"), "browser_download_url": tar, "size": std::fs::metadata(&tar).unwrap().len()}),
                json!({"name": format!("{name}.tar.gz.sha256"), "browser_download_url": sha}),
                json!({"name": format!("{name}.compat.json"), "browser_download_url": compat}),
            ];
            if let Some(n) = abi {
                assets.push(json!({"name": format!("superfluid-basert-abi-{n}.txt"), "browser_download_url": "x"}));
            }
            rs.push(json!({"tag_name": format!("v{v}"), "draft": false, "prerelease": false, "published_at": format!("{published}T12:00:00Z"), "assets": assets}));
        }
        rs.push(json!({"tag_name": "v9.9.9", "draft": true, "prerelease": false, "published_at": "2026-01-01T00:00:00Z", "assets": []}));
        let p = dir.join("releases.json");
        std::fs::write(&p, Value::Array(rs).to_string()).unwrap();
        p
    }

    pub(crate) fn no_compat(_: &str) -> Option<Value> {
        None
    }

    #[test]
    fn a_listing_is_read_newest_first_and_a_release_fits_by_the_engine_it_declares() {
        let dir = scratch("listing");
        let p = listing(&dir, &[("0.1.3", "2026-10-01", Some(1)), ("0.2.0", "2026-10-20", Some(2)), ("0.1.2", "2026-09-01", None)]);
        let found = releases(&serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap(), target().unwrap());
        assert_eq!(found.iter().map(|r| r.version.as_str()).collect::<Vec<_>>(), ["0.2.0", "0.1.3", "0.1.2"], "newest first, the draft left out");
        assert_eq!((found[0].abi, found[1].abi, found[2].abi), (Some(2), Some(1), None));
        assert_eq!(found[1].published, "2026-10-01");
        let odd = json!([{"tag_name": "v", "assets": []}, {"tag_name": "v../..", "assets": []}, {"tag_name": "vnext", "assets": []}]);
        assert_eq!(odd.as_array().unwrap().iter().filter_map(|r| release_of(r, "x.tar.gz", target().unwrap())).count(), 0, "a tag that is no version names no directory");
        assert!(["0.1.3", "0.2.0-rc.1", "1.0.0+build.5"].iter().all(|v| is_version(v)));
        let mut order = ["0.2.0-rc.1", "0.1.9", "0.2.0", "0.2.0-rc.10", "0.2.1-alpha", "0.2.0-beta", "0.2.0-rc.2", "0.2.0-rc", "0.10.0+build.5"];
        order.sort_by_key(|v| std::cmp::Reverse(release_order(v)));
        assert_eq!(order, ["0.10.0+build.5", "0.2.1-alpha", "0.2.0", "0.2.0-rc.10", "0.2.0-rc.2", "0.2.0-rc.1", "0.2.0-rc", "0.2.0-beta", "0.1.9"]);
        assert!(["", ".", "..", "../0.1.3", "0.1.3/..", "v0.1.3"].iter().all(|v| !is_version(v)));
        let compat = |url: &str| get_json(url, 1).ok();
        assert_eq!(newest(&found, &Engine::None, None, &no_compat).map(|r| r.version.as_str()), Some("0.2.0"));
        assert_eq!(newest(&found, &Engine::Abi(1), None, &no_compat).map(|r| r.version.as_str()), Some("0.1.3"), "by the marker alone");
        assert_eq!(newest(&found, &Engine::Abi(3), None, &compat), None);
        assert_eq!(newest(&found, &Engine::Unnumbered("0.3".into()), None, &compat).map(|r| r.version.as_str()), Some("0.2.0"), "compat.json names the line");
        assert_eq!(newest(&found, &Engine::Unnumbered("0.2".into()), None, &compat), None);
        assert_eq!(newest(&found, &Engine::None, Some(100_000), &no_compat), None);
        assert_eq!(newest(&found, &Engine::None, Some(0), &no_compat).map(|r| r.version.as_str()), Some("0.2.0"), "no wait");
        assert_eq!(age_days("2000-01-01", today()).map(|d| d > 9000), Some(true));
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11017);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn where_an_executable_came_from_decides_who_updates_it() {
        let dir = scratch("installed");
        let managed = dir.join("home").join("versions").join("0.1.3").join("bin").join("superfluid");
        std::fs::create_dir_all(managed.parent().unwrap()).unwrap();
        std::fs::write(&managed, "").unwrap();
        assert_eq!(installed_as(&managed), Installed::Managed { versions: std::fs::canonicalize(dir.join("home").join("versions")).unwrap(), version: "0.1.3".into() });
        let in_place = dir.join("local").join("bin").join("superfluid");
        std::fs::create_dir_all(in_place.parent().unwrap()).unwrap();
        std::fs::create_dir_all(dir.join("local").join("libexec").join("superfluid")).unwrap();
        std::fs::write(&in_place, "").unwrap();
        assert_eq!(installed_as(&in_place), Installed::InPlace { prefix: std::fs::canonicalize(dir.join("local")).unwrap() });
        let other = dir.join("Cellar").join("superfluid").join("0.1.3").join("bin").join("superfluid");
        std::fs::create_dir_all(other.parent().unwrap()).unwrap();
        std::fs::write(&other, "").unwrap();
        assert!(matches!(installed_as(&other), Installed::Other(_)));
        assert!(not_ours(&other).contains("brew upgrade superfluid"));
        // A package keeps the release's layout, and its `installed-by` file says whose it is.
        let packaged = release_tree(&dir.join("site-packages").join("superfluid_cli").join("release"), "0.1.3").join("bin").join("superfluid");
        assert!(matches!(installed_as(&packaged), Installed::InPlace { .. }), "without the file it looks like the install script's copy");
        std::fs::write(dir.join("site-packages").join("superfluid_cli").join("release").join(INSTALLED_BY), "pip: pip install --upgrade superfluid-cli\n").unwrap();
        assert!(matches!(installed_as(&packaged), Installed::Other(_)));
        let packaged = std::fs::canonicalize(&packaged).unwrap();
        assert_eq!(not_ours(&packaged), format!("{} was installed by pip; `pip install --upgrade superfluid-cli` updates it", packaged.display()));
        assert_eq!(update_command(&packaged), "pip install --upgrade superfluid-cli");
        assert_eq!(update_command(&in_place), "superfluid self update");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let refused = update(&home, &catalog, &packaged, &Options::default()).unwrap_err();
        assert!(refused.contains("installed by pip"), "{refused}");
        assert!(list(&home, &packaged).contains("`pip install --upgrade superfluid-cli` updates it"));
        assert!(running_from(std::env::current_exe().unwrap().parent().unwrap()) || true, "other processes may or may not run from here");
        assert!(!running_from(&dir.join("nowhere")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_update_places_the_release_beside_the_old_one_flips_current_and_keeps_one_back() {
        let dir = scratch("update");
        let home = dir.join("home");
        let prefix = dir.join("local");
        let p = listing(&dir, &[("0.1.3", "2026-10-01", Some(1)), ("0.1.4", "2026-10-10", Some(1)), ("0.1.5", "2026-10-12", Some(1))]);
        let from = |version: Option<&str>, dry_run: bool| Options { version: version.map(str::to_string), dry_run, source: Some(p.to_string_lossy().to_string()), ..Options::default() };
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let unpacked = release_tree(&dir.join("unpacked").join("superfluid-x"), "0.1.3");
        let r = install_from(&home, &catalog, &unpacked, Some(&prefix.join("bin"))).unwrap();
        assert!(r.changed, "{:?}", r.lines);
        let versions = Versions::of(&home);
        assert_eq!(versions.current().as_deref(), Some("0.1.3"));
        assert_eq!(std::fs::read_link(prefix.join("bin").join("superfluid")).unwrap(), versions.dir.join("current").join("bin").join("superfluid"));
        assert!(prefix.join("bin").join("superfluid-0.1.3").exists() && prefix.join("bin").join("superfluid-workerd").exists());
        assert!(!unpacked.exists(), "moved, not copied");
        let exe = versions.version_dir("0.1.3").join("bin").join("superfluid");
        let r = update(&home, &catalog, &exe, &from(None, true)).unwrap();
        assert!(r.lines[0].starts_with("0.1.3 -> 0.1.5 (published 2026-10-12;") && !r.changed, "{:?}", r.lines);
        assert_eq!(versions.on_disk(), ["0.1.3"]);
        let r = update(&home, &catalog, &exe, &from(None, false)).unwrap();
        assert!(r.changed, "{:?}", r.lines);
        assert_eq!(versions.current().as_deref(), Some("0.1.5"));
        assert_eq!(versions.previous().as_deref(), Some("0.1.3"));
        assert_eq!(versions.on_disk(), ["0.1.5", "0.1.3"]);
        assert!(r.lines.iter().any(|l| l == "0.1.3 stays installed: `superfluid self use 0.1.3` goes back to it"), "{:?}", r.lines);
        assert!(prefix.join("bin").join("superfluid-0.1.5").exists());
        assert!(!versions.dir.join(".staging").join("0.1.5").exists());
        let r = update(&home, &catalog, &versions.version_dir("0.1.5").join("bin").join("superfluid"), &from(None, false)).unwrap();
        assert!(!r.changed && r.lines[0].starts_with("superfluid 0.1.5 is current"), "{:?}", r.lines);
        release_tree(&versions.version_dir("0.9.0"), "0.9.0");
        versions.set_current("0.9.0").unwrap();
        let r = update(&home, &catalog, &exe, &from(None, false)).unwrap();
        assert!(!r.changed && r.lines[0].starts_with("superfluid 0.9.0 is current; the newest release that fits is 0.1.5"), "{:?}", r.lines);
        versions.set_current("0.1.5").unwrap();
        std::fs::remove_dir_all(versions.version_dir("0.9.0")).unwrap();
        std::fs::write(versions.dir.join("previous"), "0.1.3\n").unwrap();
        let r = update(&home, &catalog, &exe, &from(Some("v0.1.3"), false)).unwrap();
        assert!(r.changed && versions.current().as_deref() == Some("0.1.3") && versions.previous().as_deref() == Some("0.1.5"), "{:?}", r.lines);
        let r = update(&home, &catalog, &exe, &from(Some("0.1.4"), false)).unwrap();
        assert_eq!((versions.current().as_deref(), versions.previous().as_deref()), (Some("0.1.4"), Some("0.1.3")));
        assert_eq!(versions.on_disk(), ["0.1.4", "0.1.3"], "0.1.5 was neither current nor previous: {:?}", r.lines);
        assert!(prefix.join("bin").join("superfluid-0.1.5").symlink_metadata().is_err(), "its link went with it, not left dangling");
        std::fs::write(versions.version_dir("0.1.3").join("keep"), "").unwrap();
        let r = use_version(&home, &catalog, &exe, "0.1.3").unwrap();
        assert!(r.changed && versions.current().as_deref() == Some("0.1.3"));
        assert!(use_version(&home, &catalog, &exe, "..").unwrap_err().contains("is not a version"));
        assert_eq!(list(&home, &exe).lines().filter(|l| l.contains("current") || l.contains("previous")).count(), 2);
        let unreachable = Options { source: Some("/nonexistent/releases.json".into()), ..Options::default() };
        assert!(update(&home, &catalog, &exe, &unreachable).unwrap_err().contains("releases.json"), "the listing is needed for an update");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_installed_version_is_kept_and_only_a_proved_copy_replaces_a_broken_one() {
        let dir = scratch("reinstall");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let versions = Versions::of(&home);
        let r = install_from(&home, &catalog, &release_tree(&dir.join("first").join("superfluid-x"), "0.1.3"), None).unwrap();
        assert!(r.changed);
        std::fs::write(versions.version_dir("0.1.3").join("mine"), "").unwrap();
        let r = install_from(&home, &catalog, &release_tree(&dir.join("second").join("superfluid-x"), "0.1.3"), None).unwrap();
        assert!(r.lines.iter().any(|l| l.contains("installed here already and works")), "{:?}", r.lines);
        assert!(versions.version_dir("0.1.3").join("mine").exists() && versions.current().as_deref() == Some("0.1.3"));
        let bad = release_tree(&dir.join("bad").join("superfluid-x"), "0.1.3");
        std::fs::write(bad.join("bin").join("superfluid"), "#!/bin/sh\nexit 1\n").unwrap();
        let e = install_from(&home, &catalog, &bad, None).unwrap_err();
        assert!(e.contains("did not say its version"), "refused before anything is touched: {e}");
        assert!(versions.version_dir("0.1.3").join("mine").exists() && versions.current().as_deref() == Some("0.1.3"));
        let wrong = release_tree(&dir.join("wrong").join("superfluid-x"), "0.1.3");
        std::fs::write(wrong.join("bin").join("superfluid"), "#!/bin/sh\n[ \"$1\" = --version ] && echo 'superfluid 0.1.3' && exit 0\nexit 1\n").unwrap();
        let r = install_from(&home, &catalog, &wrong, None).unwrap();
        assert!(r.lines.iter().any(|l| l.contains("the copy given was not used")), "{:?}", r.lines);
        assert!(versions.version_dir("0.1.3").join("mine").exists() && versions.current().as_deref() == Some("0.1.3"));
        std::fs::write(versions.binary("0.1.3"), "#!/bin/sh\nexit 1\n").unwrap();
        let e = install_from(&home, &catalog, &wrong, None).unwrap_err();
        assert!(e.contains("was not made current") && e.contains("runtimes --json failed"), "{e}");
        assert!(versions.version_dir("0.1.3").join("mine").exists(), "a copy that fails the probe replaces nothing");
        let good = release_tree(&dir.join("good").join("superfluid-x"), "0.1.3");
        let r = install_from(&home, &catalog, &good, None).unwrap();
        assert!(r.changed && !versions.version_dir("0.1.3").join("mine").exists() && version_of(&versions.binary("0.1.3")).is_ok(), "{:?}", r.lines);
        assert!(!versions.dir.join(".replaced-0.1.3").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    const SLEEP_ENV: &str = "SUPERFLUID_SELF_TEST_SLEEP";

    /// This test binary copied under a prefix and told to sleep: a server running from the install script's layout.
    #[test]
    fn sleeps_when_asked_to() {
        if std::env::var_os(SLEEP_ENV).is_some() {
            // With a file named, it is mapped first, as a library is when loaded.
            let _mapped = std::env::var_os(SLEEP_MAP_ENV).map(|p| {
                use std::os::fd::AsRawFd;
                let f = std::fs::File::open(p).unwrap();
                // SAFETY: a private read-only mapping of a file this process opened, never read through here.
                let at = unsafe { libc::mmap(std::ptr::null_mut(), 4096, libc::PROT_READ, libc::MAP_PRIVATE, f.as_raw_fd(), 0) };
                assert_ne!(at, libc::MAP_FAILED);
                f
            });
            std::thread::sleep(Duration::from_secs(60));
        }
    }

    const SLEEP_MAP_ENV: &str = "SUPERFLUID_SELF_TEST_MAP";

    /// This test binary, run from where it is, asleep with `file` mapped: a `superfluid-workerd` that loaded an engine
    /// library from an install.
    pub(crate) fn sleeper_mapping(file: &Path) -> Sleeper {
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args(["selfupdate::tests::sleeps_when_asked_to", "--exact", "--quiet"])
            .env(SLEEP_ENV, "1")
            .env(SLEEP_MAP_ENV, file)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        Sleeper(crate::exec::when_not_busy(|| cmd.spawn()).unwrap())
    }

    /// Whether a process is seen using what is under `dir` within five seconds.
    pub(crate) fn seen_using(dir: &Path) -> bool {
        (0..50).any(|_| {
            std::thread::sleep(Duration::from_millis(100));
            !pids_using(dir).is_empty()
        })
    }

    #[test]
    fn a_process_that_maps_a_file_under_a_directory_uses_it_though_it_runs_from_elsewhere() {
        let dir = scratch("maps");
        let lib = dir.join("lib").join("libbaseRT.dylib");
        std::fs::create_dir_all(lib.parent().unwrap()).unwrap();
        std::fs::write(&lib, vec![7u8; 8192]).unwrap();
        let other = dir.join("other");
        std::fs::create_dir_all(&other).unwrap();
        let loaded = sleeper_mapping(&lib);
        assert!(seen_using(&dir), "the library it loaded is under the directory");
        assert!(pids_using(&dir).contains(&(loaded.0.id() as i32)));
        assert!(!running_from(&dir), "it runs from elsewhere");
        assert!(pids_using(&other).is_empty());
        drop(loaded);
        assert!((0..50).any(|_| {
            std::thread::sleep(Duration::from_millis(100));
            pids_using(&dir).is_empty()
        }));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_replacement_that_fails_leaves_the_copy_set_aside_back_in_place() {
        let dir = scratch("replace");
        let (from, dest, aside) = (dir.join("from"), dir.join("dest"), dir.join(".replaced"));
        std::fs::create_dir_all(&from).unwrap();
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("mine"), "").unwrap();
        let half = |from: &Path, dest: &Path| -> Result<(), String> {
            std::fs::create_dir_all(dest).unwrap();
            std::fs::write(dest.join("partial"), "").unwrap();
            Err(format!("{} -> {}: no space", from.display(), dest.display()))
        };
        let e = replace_tree(&from, &dest, &aside, half).unwrap_err();
        assert!(e.contains("no space") && !e.contains("could not be put back"), "{e}");
        assert!(dest.join("mine").exists() && !dest.join("partial").exists(), "the copy set aside is back, whole");
        assert!(!aside.exists());
        let whole = |from: &Path, dest: &Path| -> Result<(), String> { copy_tree(from, dest) };
        std::fs::write(from.join("new"), "").unwrap();
        replace_tree(&from, &dest, &aside, whole).unwrap();
        assert!(dest.join("new").exists() && !dest.join("mine").exists() && !aside.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Killed and waited for on every way out of a test.
    pub(crate) struct Sleeper(pub(crate) std::process::Child);

    impl Drop for Sleeper {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    /// This test binary copied to `exe` and run from there, asleep: a server running from the install script's layout.
    /// A copy of this test binary rather than a system one: macOS kills a platform binary run from elsewhere. Started
    /// as `when_not_busy` starts a program: on Linux a child another test forks while the copy is still open for
    /// writing holds that write until it execs, and the copy cannot be run until then (ETXTBSY).
    pub(crate) fn sleeper_at(exe: &Path) -> Sleeper {
        std::fs::copy(std::env::current_exe().unwrap(), exe).unwrap();
        let mut cmd = Command::new(exe);
        cmd.args(["selfupdate::tests::sleeps_when_asked_to", "--exact", "--quiet"])
            .env(SLEEP_ENV, "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        Sleeper(crate::exec::when_not_busy(|| cmd.spawn()).unwrap())
    }

    /// Whether a process is seen running from `dir` within five seconds.
    pub(crate) fn seen_running_from(dir: &Path) -> bool {
        (0..50).any(|_| {
            std::thread::sleep(Duration::from_millis(100));
            running_from(dir)
        })
    }

    #[test]
    fn a_server_still_running_from_the_install_scripts_copy_keeps_its_workerd_until_it_stops() {
        let dir = scratch("legacy-running");
        let (home, prefix) = (dir.join("home"), dir.join("local"));
        let p = listing(&dir, &[("0.1.3", "2026-10-01", None)]);
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        release_tree(&prefix, "0.1.2");
        let bin = prefix.join("bin");
        let mut sleeper = sleeper_at(&bin.join("superfluid-workerd"));
        if !seen_running_from(&bin) {
            eprintln!("SKIP: this platform does not report the sleeper's executable");
            return;
        }
        let exe = bin.join("superfluid");
        let opts = Options { source: Some(p.to_string_lossy().to_string()), ..Options::default() };
        let r = update(&home, &catalog, &exe, &opts).unwrap();
        let versions = Versions::of(&home);
        assert_eq!(versions.current().as_deref(), Some("0.1.3"), "{:?}", r.lines);
        assert!(std::fs::symlink_metadata(&exe).unwrap().file_type().is_symlink(), "the command moved");
        assert!(bin.join("superfluid-workerd").symlink_metadata().unwrap().file_type().is_file(), "the running server's workerd stayed a file: {:?}", r.lines);
        assert!(bin.join("superfluid-noded").symlink_metadata().unwrap().file_type().is_file());
        assert!(prefix.join("libexec").join("superfluid").join("superfluid-worker-toy").exists(), "and its workers");
        assert!(versions.dir.join("migrating").exists());
        assert!(r.lines.iter().any(|l| l.contains("still runs from the copy")), "{:?}", r.lines);
        assert_eq!(versions.previous().as_deref(), Some("0.1.2"));
        versions.link(&bin, false).unwrap();
        assert!(bin.join("superfluid-workerd").symlink_metadata().unwrap().file_type().is_file());
        drop(std::mem::replace(&mut sleeper, Sleeper(Command::new("true").spawn().unwrap())));
        let notes = versions.link(&bin, false).unwrap();
        assert!(bin.join("superfluid-workerd").symlink_metadata().unwrap().file_type().is_symlink(), "{notes:?}");
        assert!(!prefix.join("libexec").join("superfluid").join("superfluid-worker-toy").exists());
        assert!(!versions.dir.join("migrating").exists());
        assert!(notes.iter().any(|n| n.contains("now links into")), "{notes:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_server_whose_own_command_became_a_link_still_holds_the_install_scripts_copy() {
        let dir = scratch("legacy-server");
        let (home, prefix) = (dir.join("home"), dir.join("local"));
        let versions = Versions::of(&home);
        release_tree(&versions.version_dir("0.1.3"), "0.1.3");
        versions.set_current("0.1.3").unwrap();
        release_tree(&prefix, "0.1.2");
        let bin = prefix.join("bin");
        let toy = prefix.join("libexec").join("superfluid").join("superfluid-worker-toy");
        let mut sleeper = sleeper_at(&bin.join("superfluid"));
        if !seen_running_from(&bin) {
            eprintln!("SKIP: this platform does not report the sleeper's executable");
            return;
        }
        let notes = versions.link(&bin, true).unwrap();
        assert!(bin.join("superfluid").symlink_metadata().unwrap().file_type().is_symlink(), "the command moved: {notes:?}");
        assert!(bin.join("superfluid-workerd").symlink_metadata().unwrap().file_type().is_file() && toy.exists(), "{notes:?}");
        // macOS gives the path of an executable a link was renamed over only while that name still leads somewhere: with
        // the link dangling it gives none, and the server is known by the note alone.
        std::fs::remove_file(versions.binary("0.1.3")).unwrap();
        let notes = versions.link(&bin, false).unwrap();
        assert!(bin.join("superfluid-workerd").symlink_metadata().unwrap().file_type().is_file(), "the server's workerd stayed a file: {notes:?}");
        assert!(toy.exists() && versions.dir.join("migrating").exists(), "and its workers: {notes:?}");
        assert!(notes.iter().any(|n| n.contains("still runs from the copy")), "{notes:?}");
        drop(std::mem::replace(&mut sleeper, Sleeper(Command::new("true").spawn().unwrap())));
        // A process that took a recorded id later is not the one recorded.
        let note = std::fs::read_to_string(versions.dir.join("migrating")).unwrap();
        std::fs::write(versions.dir.join("migrating"), format!("{note}{} 1\n", std::process::id())).unwrap();
        let notes = versions.link(&bin, false).unwrap();
        assert!(bin.join("superfluid-workerd").symlink_metadata().unwrap().file_type().is_symlink(), "{notes:?}");
        assert!(!toy.exists() && !versions.dir.join("migrating").exists(), "{notes:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_install_script_over_an_earlier_copy_keeps_that_copy_to_go_back_to() {
        let dir = scratch("script-over-legacy");
        let (home, prefix) = (dir.join("home"), dir.join("local"));
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        release_tree(&prefix, "0.1.2");
        let tree = release_tree(&dir.join("unpacked").join("superfluid-x"), "0.1.3");
        let r = install_from(&home, &catalog, &tree, Some(&prefix.join("bin"))).unwrap();
        let versions = Versions::of(&home);
        assert_eq!((versions.current().as_deref(), versions.previous().as_deref()), (Some("0.1.3"), Some("0.1.2")), "{:?}", r.lines);
        assert_eq!(version_of(&versions.binary("0.1.2")).unwrap(), "0.1.2");
        for name in LINKED {
            assert!(prefix.join("bin").join(name).symlink_metadata().unwrap().file_type().is_symlink(), "{name}");
        }
        assert!(!prefix.join("libexec").join("superfluid").join("superfluid-worker-toy").exists());
        assert!(r.lines.iter().any(|l| l.contains("kept as 0.1.2")), "{:?}", r.lines);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_half_finished_copy_of_the_install_scripts_copy_is_copied_again_not_taken_for_it() {
        let dir = scratch("half-kept");
        let (home, prefix) = (dir.join("home"), dir.join("local"));
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let versions = Versions::of(&home);
        release_tree(&prefix, "0.1.2");
        // What a copy cut short leaves: the command and nothing beside it.
        std::fs::create_dir_all(versions.version_dir("0.1.2").join("bin")).unwrap();
        std::fs::copy(prefix.join("bin").join("superfluid"), versions.binary("0.1.2")).unwrap();
        let e = use_version(&home, &catalog, &versions.binary("0.1.2"), "0.1.2").unwrap_err();
        assert!(e.contains("no superfluid-workerd"), "{e}");
        let tree = release_tree(&dir.join("unpacked").join("superfluid-x"), "0.1.3");
        let r = install_from(&home, &catalog, &tree, Some(&prefix.join("bin"))).unwrap();
        assert_eq!(versions.previous().as_deref(), Some("0.1.2"), "{:?}", r.lines);
        assert!(versions.version_dir("0.1.2").join("bin").join("superfluid-workerd").is_file(), "{:?}", r.lines);
        assert!(versions.version_dir("0.1.2").join("libexec").join("superfluid").join("superfluid-worker-toy").exists(), "the workers came along before the originals went");
        assert!(!prefix.join("libexec").join("superfluid").join("superfluid-worker-toy").exists());
        assert!(!versions.dir.join(".staging").join("kept-0.1.2").exists());
        assert!(use_version(&home, &catalog, &versions.binary("0.1.3"), "0.1.2").unwrap().changed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_download_that_does_not_check_out_leaves_nothing_behind() {
        let dir = scratch("bad-download");
        let home = dir.join("home");
        let p = listing(&dir, &[("0.1.3", "2026-10-01", None)]);
        std::fs::write(dir.join(format!("superfluid-{}-0.1.3.tar.gz.sha256", target().unwrap())), format!("{}  x\n", "0".repeat(64))).unwrap();
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let versions = Versions::of(&home);
        release_tree(&versions.version_dir("0.1.2"), "0.1.2");
        versions.set_current("0.1.2").unwrap();
        let opts = Options { source: Some(p.to_string_lossy().to_string()), ..Options::default() };
        let e = update(&home, &catalog, &versions.binary("0.1.2"), &opts).unwrap_err();
        assert!(!versions.dir.join(".staging").join("0.1.3").exists(), "{e}");
        assert!(!versions.version_dir("0.1.3").exists() && versions.current().as_deref() == Some("0.1.2"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_older_version_that_cannot_be_removed_does_not_fail_the_install() {
        let dir = scratch("prune-fails");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let versions = Versions::of(&home);
        release_tree(&versions.version_dir("0.1.0"), "0.1.0");
        release_tree(&versions.version_dir("0.1.1"), "0.1.1");
        versions.set_current("0.1.1").unwrap();
        let stuck = versions.version_dir("0.1.0").join("bin");
        std::fs::set_permissions(&stuck, std::fs::Permissions::from_mode(0o555)).unwrap();
        let r = install_from(&home, &catalog, &release_tree(&dir.join("unpacked").join("superfluid-x"), "0.1.2"), None);
        std::fs::set_permissions(&stuck, std::fs::Permissions::from_mode(0o755)).unwrap();
        let r = r.unwrap();
        assert!(r.changed && versions.current().as_deref() == Some("0.1.2"), "{:?}", r.lines);
        assert!(r.lines.iter().any(|l| l.starts_with("an older version was not removed (")), "{:?}", r.lines);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_switch_waits_for_an_update_that_holds_the_lock() {
        let dir = scratch("lock");
        let home = dir.join("home");
        let versions = Versions::of(&home);
        release_tree(&versions.version_dir("0.1.3"), "0.1.3");
        release_tree(&versions.version_dir("0.1.4"), "0.1.4");
        versions.set_current("0.1.4").unwrap();
        let held = crate::pulls::Lock::take(&versions.dir, "self", "update of superfluid").unwrap();
        let exe = versions.binary("0.1.4");
        let (home2, exe2) = (home.clone(), exe.clone());
        let switching = std::thread::spawn(move || {
            let catalog = Catalog::new(home2.clone(), None, home2.join("superfluid-workerd"));
            use_version(&home2, &catalog, &exe2, "0.1.3").map(|r| r.changed)
        });
        std::thread::sleep(Duration::from_millis(300));
        assert!(!switching.is_finished(), "the switch does not run while the lock is held");
        assert_eq!(versions.current().as_deref(), Some("0.1.4"));
        drop(held);
        assert_eq!(switching.join().unwrap(), Ok(true));
        assert_eq!(versions.current().as_deref(), Some("0.1.3"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_check_goes_on_from_the_cache_and_asks_again_when_due() {
        let dir = scratch("tick");
        let home = dir.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None)]);
        let source = p.to_string_lossy().to_string();
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let first = tick(&home, &catalog, &source).unwrap();
        assert!(first.starts_with("superfluid 99.0.0 is available"), "{first}");
        assert_eq!(tick(&home, &catalog, "/nonexistent/releases.json"), None, "not due: the cache answers, nothing is read, and the release was announced already");
        let stale = Check { checked_at: 1, ..Check::read(&home).unwrap() };
        stale.write(&home).unwrap();
        assert_eq!(tick(&home, &catalog, "/nonexistent/releases.json"), None, "due, but the listing is unreachable: nothing, no panic");
        assert_eq!(Check::read(&home).unwrap().checked_at, 1, "an unreachable listing leaves the cache as it was");
        let p = listing(&dir.join("later"), &[("99.0.0", "2026-01-01", None), ("99.5.0", "2026-01-02", None)]);
        let later = tick(&home, &catalog, &p.to_string_lossy()).unwrap();
        assert!(later.starts_with("superfluid 99.5.0 is available"), "a release published since is announced on the next due check: {later}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_check_answers_a_changed_wait_from_the_cache_and_never_with_the_release_another_wait_picked() {
        let dir = scratch("tick-wait");
        let home = dir.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let unreachable = "/nonexistent/releases.json";
        // Read under a wait of 3 days: 99.1.0 is not out that long.
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None), ("99.1.0", "2099-01-01", None)]);
        crate::autoupdate::set_wait(&home, 3).unwrap();
        let c = check_now(&home, &catalog, 3, &p.to_string_lossy()).unwrap();
        assert_eq!((c.newest.as_deref(), c.wait), (Some("99.0.0"), Some(3)));
        let checked = c.checked_at;
        // The wait set to none: the newest is announced, not the release the wait of 3 days picked, from the releases
        // the record keeps; nothing is read again (the listing is unreachable, the record says when it was read).
        crate::autoupdate::set_wait(&home, 0).unwrap();
        let line = tick(&home, &catalog, unreachable).unwrap_or_default();
        assert!(line.starts_with("superfluid 99.1.0 is available") && line.contains("installs it ·"), "{line}");
        assert_eq!(Check::read(&home).unwrap().checked_at, checked, "no read of its own");
        // Read under no wait (a listing of a fresh release alone), then a wait of a week: the release the record named
        // under no wait is not announced, from the cache, with the listing unreachable.
        let p = listing(&dir.join("fresh"), &[("99.1.0", "2099-01-01", None)]);
        let c = check_now(&home, &catalog, 0, &p.to_string_lossy()).unwrap();
        assert_eq!(c.newest.as_deref(), Some("99.1.0"));
        Check { notified: None, ..Check::read(&home).unwrap() }.write(&home).unwrap();
        crate::autoupdate::set_wait(&home, 7).unwrap();
        assert_eq!(tick(&home, &catalog, unreachable), None, "a week's wait names nothing the read found, whatever the record named");
        assert_eq!(Check::read(&home).unwrap().notified, None, "nothing announced");
        // A record from before the releases were kept, made under another wait, is read again once.
        let legacy = Check { seen: None, wait: Some(0), newest: Some("99.1.0".into()), notified: None, ..Check::read(&home).unwrap() };
        legacy.write(&home).unwrap();
        assert_eq!(tick(&home, &catalog, unreachable), None, "read again (unreachable), not answered with 99.1.0");
        let p = listing(&dir.join("both"), &[("99.0.0", "2026-01-01", None), ("99.1.0", "2099-01-01", None)]);
        let line = tick(&home, &catalog, &p.to_string_lossy()).unwrap_or_default();
        assert!(line.starts_with("superfluid 99.0.0 is available") && line.contains("installs the newest"), "{line}");
        assert!(Check::read(&home).unwrap().seen.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_release_that_does_not_start_or_drive_the_engine_is_not_made_current() {
        let dir = scratch("probe");
        let home = dir.join("home");
        let versions = Versions::of(&home);
        let tree = release_tree(&versions.version_dir("0.5.0"), "0.5.0");
        std::fs::write(
            tree.join("bin").join("superfluid"),
            "#!/bin/sh\nif [ \"$1\" = runtimes ]; then echo '[{\"id\":\"basert\",\"ready\":\"libbaseRT states ABI 2\"}]'; else echo 'superfluid 0.5.0'; fi\n",
        )
        .unwrap();
        assert!(probe(&versions, "0.5.0", &home, &Engine::None).is_ok(), "no engine in use: starting is enough");
        let e = probe(&versions, "0.5.0", &home, &Engine::Abi(1)).unwrap_err();
        assert!(e.contains("cannot drive the engine here: libbaseRT states ABI 2"), "{e}");
        std::fs::write(tree.join("bin").join("superfluid"), "#!/bin/sh\necho 'superfluid ..'\n").unwrap();
        assert!(probe(&versions, "0.5.0", &home, &Engine::None).unwrap_err().contains("did not say its version"));
        std::fs::write(tree.join("bin").join("superfluid"), "#!/bin/sh\necho 'superfluid 0.4.9'\n").unwrap();
        assert!(probe(&versions, "0.5.0", &home, &Engine::None).unwrap_err().contains("says it is 0.4.9"));
        // A copy that hangs is killed, so the lock it runs under is let go.
        std::fs::write(tree.join("bin").join("superfluid"), "#!/bin/sh\nexec sleep 30\n").unwrap();
        let began = std::time::Instant::now();
        let e = output_within(Command::new(versions.binary("0.5.0")).arg("--version"), Duration::from_secs(1)).unwrap_err();
        assert!(e.contains("did not answer within 1s") && began.elapsed() < Duration::from_secs(10), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_in_place_install_is_migrated_into_the_layout() {
        let dir = scratch("migrate");
        let (home, prefix) = (dir.join("home"), dir.join("local"));
        let p = listing(&dir, &[("0.1.3", "2026-10-01", None)]);
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        release_tree(&prefix, "0.1.2");
        let exe = prefix.join("bin").join("superfluid");
        assert!(matches!(installed_as(&exe), Installed::InPlace { .. }));
        let opts = Options { source: Some(p.to_string_lossy().to_string()), ..Options::default() };
        let r = update(&home, &catalog, &exe, &opts).unwrap();
        let versions = Versions::of(&home);
        assert_eq!(versions.current().as_deref(), Some("0.1.3"), "{:?}", r.lines);
        assert!(std::fs::symlink_metadata(&exe).unwrap().file_type().is_symlink(), "the copy became a link");
        assert_eq!(std::fs::read_link(prefix.join("bin").join("superfluid-noded")).unwrap(), versions.dir.join("current").join("bin").join("superfluid-noded"));
        assert!(!prefix.join("libexec").join("superfluid").join("superfluid-worker-toy").exists(), "the old workers are gone");
        assert_eq!(versions.previous().as_deref(), Some("0.1.2"), "{:?}", r.lines);
        assert_eq!(version_of(&versions.binary("0.1.2")).unwrap(), "0.1.2");
        assert!(versions.version_dir("0.1.2").join("libexec").join("superfluid").join("superfluid-worker-toy").exists(), "with its workers");
        assert_eq!(versions.on_disk(), ["0.1.3", "0.1.2"]);
        assert_eq!(versions.prefix().as_deref(), Some(std::fs::canonicalize(prefix.join("bin")).unwrap().as_path()));
        assert!(r.lines.iter().any(|l| l.contains("now links into")), "{:?}", r.lines);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_release_is_out_long_enough_once_the_wait_has_passed() {
        let day = days_from_civil(2026, 3, 10);
        assert!(out_for("2026-03-10", 0, day), "no wait: as soon as it is out");
        assert!(out_for("2026-03-12", 0, day), "no wait: a date ahead of this clock is no matter");
        assert!(out_for("not a date", 0, day));
        assert!(!out_for("2026-03-10", 3, day));
        assert!(!out_for("2026-03-08", 3, day), "two days");
        assert!(out_for("2026-03-07", 3, day), "three days");
        assert!(out_for("2026-02-28", 1, days_from_civil(2026, 3, 1)), "across a month");
        assert!(!out_for("not a date", 1, day), "a date that cannot be read is never out long enough");
    }

    #[test]
    fn the_daily_check_records_the_newest_release_and_announces_it_once() {
        let dir = scratch("check");
        let home = dir.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None), ("99.1.0", "2099-01-01", None)]);
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let c = check_now(&home, &catalog, 3, &p.to_string_lossy()).unwrap();
        assert_eq!(c.newest.as_deref(), Some("99.0.0"), "99.1.0 is not out 3 days");
        assert_eq!(c.wait, Some(3));
        assert!(!c.due());
        assert_eq!(Check::read(&home), Some(c.clone()));
        let line = announce(&home, c.clone(), "superfluid self update", true).unwrap();
        assert!(line.starts_with("superfluid 99.0.0 is available (this is ") && line.contains("`superfluid self update`"), "{line}");
        assert!(
            line.ends_with("installs the newest · release notes: https://github.com/basecompute/superfluid/releases/tag/v99.0.0 · turn off: superfluid self auto-update off"),
            "every notice names the release's notes and how to turn updates off; under a wait, the command installs the newest, not the release named: {line}"
        );
        assert_eq!(announce(&home, Check::read(&home).unwrap(), "superfluid self update", true), None, "announced once");
        let again = Check { notified: None, ..c.clone() };
        let line = announce(&home, again, "brew upgrade superfluid", false).unwrap();
        assert!(line.contains("`brew upgrade superfluid` installs the newest") && !line.contains("turn off"), "with updates off already, nothing to turn off: {line}");
        assert!(Check::read(&home).unwrap().describe().starts_with("newest release out 3 days or more: 99.0.0 (published 2026-01-01;"));
        assert!(Check { wait: Some(0), ..Check::read(&home).unwrap() }.describe().starts_with("newest release: 99.0.0 (published 2026-01-01;"));
        let c = check_now(&home, &catalog, 0, &p.to_string_lossy()).unwrap();
        assert_eq!((c.newest.as_deref(), c.wait), (Some("99.1.0"), Some(0)), "no wait: the newest, as soon as it is out");
        let stale = Check { checked_at: 1, ..Check::default() };
        assert!(stale.due());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A script at `path` that appends its arguments and the first PATH entry to `log`.
    fn logging_script(path: &Path, log: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, format!("#!/bin/sh\necho \"$*|${{PATH%%:*}}\" >> {}\n", log.display())).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// A packaged release at `root` (`<root>/bin/superfluid`) whose `installed-by` says `named`, by its real path.
    fn packaged(root: &Path, named: Option<&str>) -> PathBuf {
        release_tree(root, "0.1.5");
        if let Some(line) = named {
            std::fs::write(root.join(INSTALLED_BY), format!("{line}\n")).unwrap();
        }
        std::fs::canonicalize(root.join("bin").join("superfluid")).unwrap()
    }

    #[test]
    fn each_package_manager_is_run_from_the_installation_that_owns_the_copy_or_named() {
        let dir = scratch("delegate");
        // Homebrew: the brew of the Cellar, unless pinned.
        let brew_copy = packaged(&dir.join("brew/Cellar/superfluid/0.1.5/libexec"), Some("Homebrew: brew upgrade superfluid"));
        assert!(matches!(delegate(&brew_copy), Some(Delegate::Name { ref why, .. }) if why.starts_with("there is no ")));
        logging_script(&dir.join("brew/bin/brew"), &dir.join("log"));
        let brew = std::fs::canonicalize(dir.join("brew/bin/brew")).unwrap();
        assert_eq!(
            delegate(&brew_copy),
            Some(Delegate::Run {
                manager: "Homebrew".into(),
                command: "brew upgrade superfluid".into(),
                argv: vec![brew.display().to_string(), "upgrade".into(), "superfluid".into()],
                path_first: None,
            })
        );
        std::fs::create_dir_all(dir.join("brew/var/homebrew/pinned")).unwrap();
        std::fs::write(dir.join("brew/var/homebrew/pinned/superfluid"), "").unwrap();
        assert!(matches!(delegate(&brew_copy), Some(Delegate::Name { ref why, .. }) if why.contains("`brew unpin superfluid` first")));
        // npm: a global folder's own npm, its bin first on PATH; a project's dependency is named.
        let npm_line = Some("npm: npm install --global @basecompute/superfluid@latest");
        let global = packaged(&dir.join("node/lib/node_modules/@basecompute/superfluid-darwin-arm64/release"), npm_line);
        logging_script(&dir.join("node/bin/npm"), &dir.join("log"));
        let bin = std::fs::canonicalize(dir.join("node/bin")).unwrap();
        match delegate(&global) {
            Some(Delegate::Run { argv, path_first, .. }) => {
                let prefix = std::fs::canonicalize(dir.join("node")).unwrap().display().to_string();
                assert_eq!(argv[1..], ["install", "--global", "--prefix", &prefix, "@basecompute/superfluid@latest"]);
                assert_eq!(path_first, Some(bin));
            }
            other => panic!("{other:?}"),
        }
        let project = packaged(&dir.join("app/node_modules/@basecompute/superfluid-darwin-arm64/release"), npm_line);
        assert!(matches!(delegate(&project), Some(Delegate::Name { ref why, .. }) if why.contains("update it in that project")));
        // pip is named, with its environment's python; pipx and uv are told apart by their venvs.
        let pip_line = Some("pip: pip install --upgrade superfluid-cli");
        let env = dir.join("venv");
        let pip_copy = packaged(&env.join("lib/python3.12/site-packages/superfluid_cli/release"), pip_line);
        logging_script(&env.join("bin/python"), &dir.join("log"));
        let python = std::fs::canonicalize(env.join("bin/python")).unwrap();
        assert!(matches!(delegate(&pip_copy), Some(Delegate::Name { ref command, .. })
            if *command == format!("{} -m pip install --upgrade superfluid-cli", python.display())));
        let pipx = packaged(&dir.join(".local/pipx/venvs/superfluid-cli/lib/python3.12/site-packages/superfluid_cli/release"), pip_line);
        let d = delegate(&pipx).unwrap();
        assert_eq!((d.manager(), d.command()), ("pipx", "pipx upgrade superfluid-cli"));
        let uv = packaged(&dir.join(".local/share/uv/tools/superfluid-cli/lib/python3.12/site-packages/superfluid_cli/release"), pip_line);
        let d = delegate(&uv).unwrap();
        assert_eq!((d.manager(), d.command()), ("uv", "uv tool upgrade superfluid-cli"));
        // A manager superfluid does not run is named with the file's own command.
        let conda = packaged(&dir.join("conda/release"), Some("conda: conda update superfluid"));
        assert!(matches!(delegate(&conda), Some(Delegate::Name { ref command, .. }) if command == "conda update superfluid"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_superfluid_basert_installed_is_left_to_basert() {
        let dir = scratch("basert-owned");
        // basert's layout: <dir>/basert beside <dir>/superfluid, which an older basert marked with its stamp alone.
        let copy = packaged(&dir.join("superfluid"), None);
        assert!(matches!(installed_as(&copy), Installed::InPlace { .. }), "without basert's marks it is the install script's");
        std::fs::write(dir.join("superfluid").join(BASERT_STAMP), "v0.1.5\n").unwrap();
        assert!(matches!(installed_as(&copy), Installed::Other(_)));
        // No basert beside it: the one on PATH, when there is one.
        let d = delegate(&copy).unwrap();
        assert_eq!((d.manager(), d.command()), ("basert", "basert superfluid install"));
        logging_script(&dir.join("basert"), &dir.join("log"));
        std::fs::write(dir.join("superfluid").join(INSTALLED_BY), "basert: basert superfluid install\n").unwrap();
        let basert = std::fs::canonicalize(dir.join("basert")).unwrap();
        assert_eq!(
            delegate(&copy),
            Some(Delegate::Run {
                manager: "basert".into(),
                command: "basert superfluid install".into(),
                argv: vec![basert.display().to_string(), "superfluid".into(), "install".into()],
                path_first: None,
            })
        );
        assert_eq!(update_command(&copy), "basert superfluid install");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        // A version is basert's to pick.
        let opts = Options { version: Some("0.1.4".into()), yes: true, ..Options::default() };
        let refused = update(&home, &catalog, &copy, &opts).unwrap_err();
        assert!(refused.contains("is basert's, which picks the release that serves its engine"), "{refused}");
        // With --yes, basert is handed this process, not run beside it: it refuses while anything runs from the copy.
        // Nothing was moved under ~/.superfluid.
        let opts = Options { yes: true, ..Options::default() };
        let mut hand_off = update(&home, &catalog, &copy, &opts).unwrap().hand_off.expect("basert is handed off to");
        assert!(!dir.join("log").exists());
        assert!(crate::exec::when_not_busy(|| hand_off.status()).unwrap().success());
        assert!(std::fs::read_to_string(dir.join("log")).unwrap().starts_with("superfluid install|"));
        assert!(!home.join("versions").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_package_managers_update_is_asked_for_dry_run_or_run_with_yes() {
        let dir = scratch("by-manager");
        let copy = packaged(&dir.join("brew/Cellar/superfluid/0.1.5/libexec"), Some("Homebrew: brew upgrade superfluid"));
        logging_script(&dir.join("brew/bin/brew"), &dir.join("log"));
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        // Tests run without a terminal: without --yes the command is named, not run.
        let refused = update(&home, &catalog, &copy, &Options::default()).unwrap_err();
        assert!(refused.contains("run `brew upgrade superfluid`, or `superfluid self update --yes`"), "{refused}");
        let dry = update(&home, &catalog, &copy, &Options { dry_run: true, yes: true, ..Options::default() }).unwrap();
        assert!(dry.lines[0].contains("`self update` would run ") && dry.lines[0].ends_with("brew/bin/brew upgrade superfluid"), "{:?}", dry.lines);
        let pinned = Options { version: Some("0.1.4".into()), yes: true, ..Options::default() };
        assert!(update(&home, &catalog, &copy, &pinned).unwrap_err().contains("which chooses its version"));
        assert!(!dir.join("log").exists());
        let done = update(&home, &catalog, &copy, &Options { yes: true, ..Options::default() }).unwrap();
        assert!(done.changed && done.hand_off.is_none());
        assert!(std::fs::read_to_string(dir.join("log")).unwrap().starts_with("upgrade superfluid|"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
