//! Automatic updates: the setting and the switches that turn them off, the daily read a launch starts in the
//! background, and the update a command run in a terminal makes before it goes on (`superfluid self` has the rest).
//!
//! A command that serves or installs (`serve`, `launch`, `node`, `top`, `runtime install|update|repair|remove|use`)
//! looks at what the last read of the release listing found. When it is due again, a detached
//! `superfluid self check --record` reads it, so the command never waits on the network. When the record names a
//! release newer than this one (past the hold-off and fitting the engine here, as the daily notice's), a copy under
//! `versions` or the install script's is updated in the foreground, with a progress bar, and the command is run
//! again by the new version; a copy a package manager owns is left to that manager, run in the background once
//! nothing runs from the copy. Without a terminal, under CI, or with updates turned off, one line per release says
//! a newer one is out.

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use superfluid_adapter_kit::install::{sha256_of, Asset};

use crate::installs;
use crate::runtime_pick::RuntimeId;
use crate::runtimes::Catalog;
use crate::selfupdate::{self, Check, Delegate, Installed, Options, Report, Versions};

/// Set to anything but empty, `0` or `false`: no update is made, none is looked for, and none is announced.
pub const NO_UPDATE_ENV: &str = "SUPERFLUID_NO_UPDATE";
/// Set on a command superfluid runs again itself: the version it ran as before an automatic update ran it with the
/// new one, or the version a node starting again runs as. Such a run updates nothing, and the variable is taken out of
/// its environment, so what it starts (a server, an agent) is not kept from updating.
pub const UPDATED_FROM_ENV: &str = "SUPERFLUID_UPDATED_FROM";
/// On any command, before a `--`: no update for that run.
pub const NO_UPDATE_FLAG: &str = "--no-update";
/// What turns automatic updates off, which every line about one names.
pub const OFF_COMMAND: &str = "superfluid self auto-update off";
/// `$SUPERFLUID_HOME/settings.json`: `{"autoUpdate": true|false}`, on when the file or the field is missing.
const SETTINGS: &str = "settings.json";
const AUTO_UPDATE_KEY: &str = "autoUpdate";
/// The package managers whose update an automatic update runs itself, in the background.
const IN_BACKGROUND: &[&str] = &["Homebrew", "npm", "cargo", "pipx", "uv"];
/// How long the background update of a package manager's copy waits for the superfluid running from it to end.
const WAIT_FOR_IDLE: Duration = Duration::from_secs(7 * 24 * 60 * 60);


/// Whether an environment variable is set to something that turns a thing on: not empty, `0` or `false`.
fn set(value: Option<String>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
}

/// `bytes` written beside `path` and renamed over it, so a reader finds the old file or the new, never part of one.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let written = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    written.map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })
}

/// The setting in `settings.json`: on unless it says `false`.
pub fn setting(home: &Path) -> bool {
    let read = std::fs::read_to_string(home.join(SETTINGS)).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok());
    read.and_then(|v| v[AUTO_UPDATE_KEY].as_bool()).unwrap_or(true)
}

/// Sets the setting, keeping whatever else the file holds (Superfluid Desktop's own keys).
pub fn set_setting(home: &Path, on: bool) -> Result<(), String> {
    std::fs::create_dir_all(home).map_err(|e| format!("{}: {e}", home.display()))?;
    let path = home.join(SETTINGS);
    let mut v = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).filter(Value::is_object).unwrap_or_else(|| json!({}));
    v[AUTO_UPDATE_KEY] = Value::Bool(on);
    write_atomically(&path, format!("{v:#}\n").as_bytes())
}

/// The variable that turns updates, the check and its line off, when one does.
pub fn switched_off(env: &dyn Fn(&str) -> Option<String>) -> Option<&'static str> {
    if set(env(NO_UPDATE_ENV)) {
        return Some("SUPERFLUID_NO_UPDATE is set");
    }
    set(env(selfupdate::NO_CHECK_ENV)).then_some("SUPERFLUID_NO_UPDATE_CHECK is set")
}

/// Whether automatic updates are on here; when not, why.
pub fn enabled(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> Result<(), String> {
    if let Some(why) = switched_off(env) {
        return Err(why.to_string());
    }
    if set(env("CI")) {
        return Err("CI is set".to_string());
    }
    match setting(home) {
        true => Ok(()),
        false => Err(OFF_COMMAND.to_string()),
    }
}

/// `automatic updates: on`, or `off (<why>)`.
pub fn state(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> String {
    match enabled(home, env) {
        Ok(()) => "automatic updates: on".to_string(),
        Err(why) => format!("automatic updates: off ({why})"),
    }
}

/// `superfluid self auto-update [on|off]`; the exit code. Superfluid Desktop runs it with `on` or `off` to follow its
/// own switch.
pub fn auto_update_command(home: &Path, arg: Option<&str>) -> i32 {
    let env = |k: &str| std::env::var(k).ok();
    let on = match arg {
        None => {
            println!("{}", state(home, &env));
            return 0;
        }
        Some("on") => true,
        Some("off") => false,
        Some(_) => {
            eprint!("{}", selfupdate::HELP);
            return 2;
        }
    };
    if let Err(e) = set_setting(home, on) {
        eprintln!("superfluid: {e}");
        return 1;
    }
    println!("automatic updates: {}", if on { "on" } else { "off" });
    if let (true, Err(why)) = (on, enabled(home, &env)) {
        eprintln!("superfluid: {why}, which keeps them off here");
    }
    0
}


/// `args` without `--no-update`, and whether it was there: it may come anywhere before a `--`, after which the words
/// are another program's.
pub fn strip_no_update(args: &[String]) -> (Vec<String>, bool) {
    let mut found = false;
    let mut past_separator = false;
    let mut out = Vec::with_capacity(args.len());
    for a in args {
        if !past_separator && a == NO_UPDATE_FLAG {
            found = true;
            continue;
        }
        past_separator |= a == "--";
        out.push(a.clone());
    }
    (out, found)
}

/// What a launch does about updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Nothing: no read of the listing, no line, no update.
    Quiet,
    /// The daily read, and one line per newer release.
    Notice,
    /// The daily read, and the update when the record names a newer release.
    Update,
}

/// The commands that update before they run: those that serve, or install what serves. Help, the version, `self`,
/// `compat`, `stop`, the listings (`runtimes`, `models`, `runtime versions|options`), the clients of a running
/// server (`session`, `media`), `fleet` (its token is read by scripts) and anything asked for as `--json` do not.
fn updates_first(before: &[&str]) -> bool {
    if before.iter().any(|a| matches!(*a, "--json" | "-h" | "--help")) {
        return false;
    }
    match before {
        ["serve", ..] | ["launch", ..] | ["node", ..] | ["top", ..] => true,
        ["runtime", sub, ..] => matches!(*sub, "install" | "update" | "repair" | "remove" | "use"),
        _ => false,
    }
}

/// What the launch of `args` does: `no_update_flag` is `--no-update`, `terminal` whether stderr is one, `setting_on`
/// the setting in `settings.json`.
pub fn mode(args: &[String], no_update_flag: bool, terminal: bool, setting_on: bool, env: &dyn Fn(&str) -> Option<String>) -> Mode {
    let before: Vec<&str> = args.iter().map(String::as_str).take_while(|a| *a != "--").collect();
    let updated = env(UPDATED_FROM_ENV).is_some_and(|v| !v.is_empty());
    if no_update_flag || switched_off(env).is_some() || updated || before.contains(&"--offline") || !updates_first(&before) {
        return Mode::Quiet;
    }
    if !terminal || set(env("CI")) || !setting_on {
        return Mode::Notice;
    }
    Mode::Update
}

/// Whether an automatic update runs the update of `found`, the package manager that owns this copy, itself: one of
/// Homebrew, a global npm, cargo, pipx and uv that superfluid would run (not pinned, not another user's, not a
/// project's), a copy this user `owns`, and not on Windows, where a program that runs cannot be replaced. What runs
/// from the copy (a server, this command) is waited for by the update in the background.
pub fn runs_in_background(found: &Delegate, owns: bool, windows: bool) -> bool {
    matches!(found, Delegate::Run { manager, .. } if IN_BACKGROUND.contains(&manager.as_str())) && owns && !windows
}

const DAY: u64 = 24 * 60 * 60;

/// Whether `newer` was tried within the last day: a failed update is not tried again until the next daily read.
fn tried(check: &Check, newer: &str) -> bool {
    check.attempted.as_deref() == Some(newer) && selfupdate::now_secs().saturating_sub(check.attempted_at) < DAY
}

/// Whether `newer` is handed to the package manager's update in the background, which may still be waiting for the
/// copy to be free; one handed over longer ago than that wait (its process lost, say to a restart) is handed again.
fn delegated(check: &Check, newer: &str) -> bool {
    check.delegated.as_deref() == Some(newer) && selfupdate::now_secs().saturating_sub(check.delegated_at) < WAIT_FOR_IDLE.as_secs() + DAY
}

/// Whether `self use` went back from `newer`, or from a later version: no automatic update undoes that choice.
fn declined(check: &Check, newer: &str) -> bool {
    check.declined.as_deref().is_some_and(|d| !selfupdate::later(newer, d))
}

fn mark_attempt(home: &Path, newer: &str) {
    let check = Check::read(home).unwrap_or_default();
    let _ = Check { attempted: Some(newer.to_string()), attempted_at: selfupdate::now_secs(), ..check }.write(home);
}

/// What a launch does about a newer release, by who owns the copy.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Plan {
    /// Update it here, now.
    Here,
    /// Run its package manager's update in the background.
    Manager(Delegate),
    /// Say a newer release is out, once.
    Notice,
}

fn plan(exe: &Path, mode: Mode, check: &Check, from: &str, newer: &str) -> Plan {
    if mode != Mode::Update || declined(check, newer) {
        return Plan::Notice;
    }
    match selfupdate::installed_as(exe) {
        // A version run by its own name (`superfluid-0.2.0`), not through `current`, is left as it is.
        Installed::Managed { versions, version } if version == from && (Versions { dir: versions.clone() }).current().as_deref() == Some(from) => match tried(check, newer) {
            true => Plan::Notice,
            false => Plan::Here,
        },
        Installed::InPlace { .. } if !tried(check, newer) => Plan::Here,
        // Handed over once per release (the line that says a release is out is another matter), and again a day after
        // an update in the background that did not run it.
        Installed::Other(p) => match selfupdate::delegate(&p) {
            Some(found) if !delegated(check, newer) && !tried(check, newer) => {
                match runs_in_background(&found, crate::sys::writable(&selfupdate::watched_dir(&p, found.manager())), cfg!(windows)) {
                    true => Plan::Manager(found),
                    false => Plan::Notice,
                }
            }
            _ => Plan::Notice,
        },
        _ => Plan::Notice,
    }
}

/// What `main` runs before the command `args` (without `--no-update`, which `no_update_flag` says was there). It
/// returns to go on with the command here; after an update it runs the command with the new version in its place.
pub fn at_launch(args: &[String], no_update_flag: bool) {
    let env = |k: &str| std::env::var(k).ok();
    let home = crate::runtimes::default_home();
    let mode = mode(args, no_update_flag, std::io::stderr().is_terminal(), setting(&home), &env);
    // For this run alone: what it starts (a server, an agent and the commands run in it) updates as any other would.
    std::env::remove_var(UPDATED_FROM_ENV);
    if mode == Mode::Quiet {
        return;
    }
    let Ok(exe) = std::env::current_exe() else { return };
    let exe = crate::sys::canonical(&exe).unwrap_or(exe);
    let check = Check::read(&home).unwrap_or_default();
    let from = selfupdate::running();
    let newer = check.newer_than(from).map(str::to_string);
    let planned = newer.as_deref().map(|n| plan(&exe, mode, &check, from, n));
    // The daily read is started once the plan is made, and not before an update made here: that update reads the
    // listing itself, and the command it runs again in this process's place would be left a child it never waits for.
    let due = check.due();
    if due && planned != Some(Plan::Here) {
        refresh_in_background(&home, &exe);
    }
    let (Some(newer), Some(planned)) = (newer, planned) else { return };
    let notes = check.notes_url().unwrap_or_default();
    match planned {
        Plan::Here => {
            let catalog = Catalog::from_env();
            let mut say = |line: String| eprintln!("{line}");
            match update_here(&home, &catalog, &exe, from, &newer, &notes, None, named_runtime(args), &mut say) {
                Some(bin) => run_again(&bin, args, from, &newer),
                None if due => refresh_in_background(&home, &exe),
                None => {}
            }
        }
        Plan::Manager(found) => match in_background(&home, &exe, &found, from, &newer, &notes) {
            Ok(line) => eprintln!("{line}"),
            Err(e) => {
                eprintln!("superfluid: {} could not be started in the background ({e})", found.command());
                notice(&home, &exe);
            }
        },
        Plan::Notice => notice(&home, &exe),
    }
}

/// The one line per release, from the record as it is now (a launch may have claimed the next read).
fn notice(home: &Path, exe: &Path) {
    let Some(check) = Check::read(home) else { return };
    let auto = enabled(home, &|k| std::env::var(k).ok()).is_ok();
    if let Some(line) = selfupdate::announce(home, check, &selfupdate::update_command(exe), auto) {
        eprintln!("superfluid: {line}");
    }
}

/// The daily read, claimed first (so the launches after this one do not start it again) and made by a detached
/// `superfluid self check --record` in a session of its own, its streams closed: the command goes on at once.
fn refresh_in_background(home: &Path, exe: &Path) {
    // A home that is not there yet is not made for this: a command that fails as it starts writes nothing, and the
    // first command that keeps something there (a server, an install) has the next one read the listing.
    if !home.is_dir() {
        return;
    }
    // The record as it is now, so what was written since this command read it (an attempt) is kept.
    let check = Check::read(home).unwrap_or_default();
    if (Check { checked_at: selfupdate::now_secs(), ..check }).write(home).is_err() {
        return;
    }
    let mut cmd = Command::new(exe);
    cmd.args(["self", "check", "--record"])
        .env(selfupdate::NO_CHECK_ENV, "1")
        .env(NO_UPDATE_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    crate::sys::detached_hidden(&mut cmd);
    if let Ok(mut child) = cmd.spawn() {
        // Waited for, so it leaves no zombie while a long command runs.
        let _ = std::thread::Builder::new().name("update-check".into()).spawn(move || child.wait());
    }
}

/// The runtime a `runtime install|repair|remove <id>` names: the update before it leaves that runtime to the
/// command, which would install it again, or find a build installed beside the one it removes.
fn named_runtime(args: &[String]) -> Option<RuntimeId> {
    match args {
        [runtime, sub, id, ..] if runtime == "runtime" && matches!(sub.as_str(), "install" | "repair" | "remove") => RuntimeId::parse(id).map(RuntimeId::base),
        _ => None,
    }
}

/// The command run again by the new version, in this process's place (on Windows, as a child this process waits
/// for); only returns when it could not be.
fn run_again(bin: &Path, args: &[String], from: &str, newer: &str) {
    let _ = std::io::stdout().flush();
    let mut cmd = Command::new(bin);
    cmd.args(args).env(UPDATED_FROM_ENV, from);
    let e = crate::sys::replace_process(&mut cmd);
    eprintln!("superfluid: superfluid {newer} could not be started ({e}); going on with {from}");
}


/// The automatic update of `exe`, the version `from`, to `newer`, whose notes are at `notes`: the release placed and
/// made current as `self update --version <newer>` does it (without waiting on another update), the runtimes that
/// are neither pinned nor in use (nor `skip`, which the command is about to change) updated by the new version, and a
/// line each for going back and turning updates off. The new program, or `None` when the update failed, which a line
/// says.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_here(
    home: &Path,
    catalog: &Catalog,
    exe: &Path,
    from: &str,
    newer: &str,
    notes: &str,
    source: Option<String>,
    skip: Option<RuntimeId>,
    say: &mut dyn FnMut(String),
) -> Option<PathBuf> {
    say(format!("Updating superfluid {from} → {newer} · release notes: {notes}"));
    mark_attempt(home, newer);
    let (versions, managed) = match selfupdate::installed_as(exe) {
        Installed::Managed { versions, .. } => (Versions { dir: versions }, true),
        _ => (Versions::of(home), false),
    };
    let caught = Interrupts::catch();
    let opts = Options { version: Some(newer.to_string()), no_wait: true, source, ..Options::default() };
    let failed = |why: String| format!("couldn't update superfluid to {newer}: {why}; staying on {from}");
    let bin = versions.binary(newer);
    let report = match selfupdate::update(home, catalog, exe, &opts) {
        Ok(report) if report.changed => report,
        // Another update made it current while this command started: the command goes on with it.
        Ok(_) if versions.current().as_deref() == Some(newer) && bin.is_file() => {
            drop(caught);
            say(format!("superfluid {newer} is in use · `superfluid self use {from}` goes back · `{OFF_COMMAND}` turns automatic updates off"));
            return Some(bin);
        }
        Ok(report) => {
            say(failed(report.lines.last().cloned().unwrap_or_else(|| "nothing was changed".into())));
            return None;
        }
        // `current` moved, and what came after failed (a link in the prefix): a copy under `versions` runs the new
        // version from now on, so it is one, with what failed said. The install script's copy keeps its own files.
        Err(e) if managed && versions.current().as_deref() == Some(newer) && bin.is_file() => {
            say(format!("superfluid: {e}"));
            Report::default()
        }
        Err(e) => {
            say(failed(if interrupted() { "interrupted".to_string() } else { e }));
            return None;
        }
    };
    for note in &report.notes {
        say(format!("superfluid: {note}"));
    }
    update_runtimes(&bin, catalog, from, newer, skip, say);
    drop(caught);
    say(format!("superfluid {newer} is in use · `superfluid self use {from}` goes back · `{OFF_COMMAND}` turns automatic updates off"));
    Some(bin)
}

/// The runtimes installed here, by name.
fn installed_runtimes(catalog: &Catalog) -> Vec<RuntimeId> {
    let mut ids: Vec<RuntimeId> = std::fs::read_dir(catalog.home().join("runtimes"))
        .map(|rd| rd.flatten().filter_map(|e| e.file_name().to_str().and_then(RuntimeId::parse)).collect())
        .unwrap_or_default();
    ids.retain(|id| id.install().is_none() && !installs::list(catalog, *id).is_empty());
    ids.sort();
    ids
}

/// The new superfluid `bin` run with `args` for an update: in `catalog`'s home, updating nothing itself, its stdout on
/// this command's stderr (the command's own stdout may be read by a program: `launch --print`, `serve > log`).
fn by_new_version(bin: &Path, catalog: &Catalog, from: &str, args: &[&str]) -> Command {
    let mut cmd = Command::new(bin);
    cmd.args(args)
        .env("SUPERFLUID_HOME", catalog.home())
        .env(NO_UPDATE_ENV, "1")
        .env(UPDATED_FROM_ENV, from)
        .stdin(Stdio::null())
        .stdout(stderr_for_child());
    cmd
}

/// This process's stderr, for a child to write its stdout to; nowhere when it cannot be had.
fn stderr_for_child() -> Stdio {
    #[cfg(unix)]
    {
        use std::os::fd::AsFd;
        if let Ok(fd) = std::io::stderr().as_fd().try_clone_to_owned() {
            return Stdio::from(fd);
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsHandle;
        if let Ok(handle) = std::io::stderr().as_handle().try_clone_to_owned() {
            return Stdio::from(handle);
        }
    }
    Stdio::null()
}

/// `runtime update <id>` by the new superfluid `bin` for each runtime installed here, so its install has a build and
/// a worker the new version runs: one it still cannot run afterwards (the build did not change, so the install keeps
/// the worker the old version copied in) is installed again by it, `runtime repair <id>`. A runtime pinned to an
/// install, one whose default install a running process uses, and `skip` stay.
fn update_runtimes(bin: &Path, catalog: &Catalog, from: &str, newer: &str, skip: Option<RuntimeId>, say: &mut dyn FnMut(String)) {
    let mut updated = Vec::new();
    for id in installed_runtimes(catalog) {
        if interrupted() {
            say("superfluid: the runtimes were left as they are: interrupted".to_string());
            return;
        }
        if Some(id) == skip {
            continue;
        }
        let Some(current) = installs::default(catalog, id) else { continue };
        if let Some(pin) = installs::pinned(catalog, id) {
            say(format!("superfluid: {id} stays on {pin}, which is pinned (`superfluid runtime update {id}` updates it)"));
            continue;
        }
        if !selfupdate::pids_using(&current.dir).is_empty() {
            say(format!("superfluid: {id} stays on {}: a running server uses it (`superfluid runtime update {id}` updates it once none does)", current.name));
            continue;
        }
        say(format!("superfluid: updating {id} ({}) with superfluid {newer}", current.name));
        let mut cmd = by_new_version(bin, catalog, from, &["runtime", "update", id.name()]);
        match crate::exec::when_not_busy(|| cmd.status()) {
            Ok(status) if status.success() => updated.push(id),
            Ok(status) => say(format!("superfluid: {id} was not updated ({status}); `superfluid runtime update {id}` tries again")),
            Err(e) => say(format!("superfluid: {id} was not updated ({e}); `superfluid runtime update {id}` tries again")),
        }
    }
    if updated.is_empty() || interrupted() {
        return;
    }
    for (id, why) in not_run_by(bin, catalog, from).into_iter().filter(|(id, _)| updated.contains(id)) {
        say(format!("superfluid: {id} does not run with superfluid {newer} ({why}): installing it again with it"));
        let mut cmd = by_new_version(bin, catalog, from, &["runtime", "repair", id.name()]);
        match crate::exec::when_not_busy(|| cmd.status()) {
            Ok(status) if status.success() => {}
            Ok(status) => say(format!("superfluid: {id} was not installed again ({status}); `superfluid runtime repair {id}` tries again")),
            Err(e) => say(format!("superfluid: {id} was not installed again ({e}); `superfluid runtime repair {id}` tries again")),
        }
    }
}

/// The installed runtimes the new superfluid `bin` cannot run, as its `runtimes --json` reports them, and why.
fn not_run_by(bin: &Path, catalog: &Catalog, from: &str) -> Vec<(RuntimeId, String)> {
    let mut cmd = by_new_version(bin, catalog, from, &["runtimes", "--json"]);
    let Ok(Some(out)) = crate::exec::output_within(&mut cmd, Duration::from_secs(5 * 60)) else { return Vec::new() };
    let rows: Value = serde_json::from_slice(&out.stdout).unwrap_or_default();
    let broken = |r: &&Value| r["state"] == "broken" && r["worker"]["source"] == "installed";
    rows.as_array()
        .map(|rows| {
            rows.iter()
                .filter(broken)
                .filter_map(|r| Some((RuntimeId::parse(r["id"].as_str()?)?, r["ready"].as_str().unwrap_or("it reports no reason").to_string())))
                .collect()
        })
        .unwrap_or_default()
}


/// `$SUPERFLUID_HOME/logs/auto-update.log`, where a package manager's update run in the background writes.
fn log_path(home: &Path) -> PathBuf {
    home.join("logs").join("auto-update.log")
}

/// Starts `superfluid self background-update` from the copy, detached, its output appended to the log: it runs the
/// manager's update once nothing runs from the copy (this command included, which runs from it until it ends). The
/// release is marked handed over (and announced, as the line says it is out), so this is done once per release
/// unless that update does not run it. The line to print.
fn in_background(home: &Path, exe: &Path, found: &Delegate, from: &str, newer: &str, notes: &str) -> Result<String, String> {
    let log = log_path(home);
    std::fs::create_dir_all(log.parent().unwrap_or(home)).map_err(|e| format!("{}: {e}", log.display()))?;
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&log).map_err(|e| format!("{}: {e}", log.display()))?;
    let (y, mo, d, h, mi, s) = crate::sys::utc(selfupdate::now_secs());
    let _ = writeln!(out, "{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z superfluid {from} -> {newer}: {} for {}", found.command(), exe.display());
    let err = out.try_clone().map_err(|e| format!("{}: {e}", log.display()))?;
    let mut cmd = Command::new(exe);
    cmd.args(["self", "background-update"])
        .env(selfupdate::NO_CHECK_ENV, "1")
        .env(NO_UPDATE_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err));
    crate::sys::detached_hidden(&mut cmd);
    // Marked before it starts, so that what it writes when it gives up is not written over.
    let check = Check::read(home).unwrap_or_default();
    let before = check.clone();
    let _ = Check { notified: Some(newer.to_string()), delegated: Some(newer.to_string()), delegated_at: selfupdate::now_secs(), ..check }.write(home);
    let mut child = match crate::exec::when_not_busy(|| cmd.spawn()) {
        Ok(child) => child,
        Err(e) => {
            let _ = before.write(home);
            return Err(e.to_string());
        }
    };
    let _ = std::thread::Builder::new().name("auto-update".into()).spawn(move || child.wait());
    Ok(format!(
        "Updating superfluid {from} → {newer} with {} in the background (log: {}) · release notes: {notes} · turn off: {OFF_COMMAND}",
        found.manager(),
        log.display()
    ))
}

/// `superfluid self background-update`: waits until no other superfluid runs from this copy, then runs the update
/// of the package manager that owns it, all of it into the log the launch opened; the exit code. It runs nothing when
/// automatic updates were turned off meanwhile, or the copy was removed or is no longer that manager's; when it runs
/// nothing, or the manager's update fails, the release is no longer marked handed over (`handed_back`).
pub fn background_update(home: &Path, exe: &Path) -> i32 {
    let exe = crate::sys::canonical(exe).unwrap_or_else(|_| exe.to_path_buf());
    let Some(Delegate::Run { manager, argv, path_first, .. }) = selfupdate::delegate(&exe) else {
        eprintln!("superfluid: {} is not a copy superfluid updates with its package manager", exe.display());
        handed_back(home, false);
        return 1;
    };
    if !IN_BACKGROUND.contains(&manager.as_str()) {
        eprintln!("superfluid: {manager}'s update is not run in the background");
        handed_back(home, false);
        return 1;
    }
    let watched = selfupdate::watched_dir(&exe, &manager);
    let began = Instant::now();
    let mut said = false;
    loop {
        // The file's setting: this process's environment turns updates off for itself.
        if !setting(home) {
            eprintln!("superfluid: automatic updates were turned off; {} was not run", argv.join(" "));
            handed_back(home, false);
            return 0;
        }
        let busy = selfupdate::superfluid_under(&watched);
        if busy.is_empty() {
            break;
        }
        if began.elapsed() > WAIT_FOR_IDLE {
            eprintln!("superfluid: gave up: superfluid still runs from {} (pid {busy:?})", watched.display());
            handed_back(home, true);
            return 1;
        }
        if !said {
            eprintln!("superfluid: waiting until no superfluid runs from {} (pid {busy:?})", watched.display());
            said = true;
        }
        std::thread::sleep(Duration::from_secs(15));
    }
    // Looked at again now the wait is over: an uninstall made meanwhile is not undone by an install.
    if let Some(why) = not_to_run(home, &exe, &argv) {
        eprintln!("superfluid: {why}; {} was not run", argv.join(" "));
        handed_back(home, false);
        return 0;
    }
    eprintln!("superfluid: running {}", argv.join(" "));
    let mut run = Command::new(&argv[0]);
    run.args(&argv[1..]).stdin(Stdio::null());
    if let Some(first) = path_first {
        let rest = std::env::var_os("PATH").unwrap_or_default();
        if let Ok(path) = std::env::join_paths(std::iter::once(first).chain(std::env::split_paths(&rest))) {
            run.env("PATH", path);
        }
    }
    match crate::exec::when_not_busy(|| run.status()) {
        Ok(status) if status.success() => {
            eprintln!("superfluid: {manager} updated it");
            0
        }
        Ok(status) => {
            eprintln!("superfluid: `{}` failed ({status})", argv.join(" "));
            handed_back(home, true);
            1
        }
        Err(e) => {
            eprintln!("superfluid: running {}: {e}", argv[0]);
            handed_back(home, true);
            1
        }
    }
}

/// Why the manager's update `argv` for the copy `exe` is not to be run now: automatic updates turned off, the copy
/// gone (uninstalled while a server ran from it), or no longer one that manager updates with that command.
fn not_to_run(home: &Path, exe: &Path, argv: &[String]) -> Option<String> {
    if !setting(home) {
        return Some("automatic updates were turned off".to_string());
    }
    if !exe.is_file() {
        return Some(format!("{} is gone", exe.display()));
    }
    match selfupdate::delegate(exe) {
        Some(Delegate::Run { argv: now, .. }) if now == argv => None,
        _ => Some(format!("{} is no longer updated by `{}`", exe.display(), argv.join(" "))),
    }
}

/// The release handed over no longer marked so, when the update in the background did not run its manager or the
/// manager failed. A failure counts as an attempt, so it is handed over again after a day, and is announced again, so
/// the next launch says the release is still out.
fn handed_back(home: &Path, failed: bool) {
    let Some(check) = Check::read(home) else { return };
    let Some(v) = check.delegated.clone() else { return };
    let check = Check { delegated: None, delegated_at: 0, ..check };
    let check = match failed {
        true => Check { attempted: Some(v), attempted_at: selfupdate::now_secs(), notified: None, ..check },
        false => check,
    };
    let _ = check.write(home);
}


static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Whether Ctrl+C was pressed while an automatic update caught it.
pub(crate) fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

/// Ctrl+C caught while it lives: it stops the update (the download, the proof, a runtime's update all end with it),
/// and the command goes on, rather than ending the command with it.
struct Interrupts;

impl Interrupts {
    fn catch() -> Interrupts {
        INTERRUPTED.store(false, Ordering::Relaxed);
        #[cfg(unix)]
        // SAFETY: the handler only stores to an atomic, which is async-signal-safe.
        unsafe {
            libc::signal(libc::SIGINT, on_interrupt as extern "C" fn(libc::c_int) as libc::sighandler_t);
        }
        #[cfg(windows)]
        // SAFETY: the handler is a function that lives as long as the process and only stores to an atomic.
        unsafe {
            windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(on_ctrl), 1);
        }
        Interrupts
    }
}

impl Drop for Interrupts {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: putting back the default disposition.
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
        }
        #[cfg(windows)]
        // SAFETY: removes the handler `catch` added.
        unsafe {
            windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(on_ctrl), 0);
        }
    }
}

#[cfg(unix)]
extern "C" fn on_interrupt(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

#[cfg(windows)]
unsafe extern "system" fn on_ctrl(kind: u32) -> windows_sys::Win32::Foundation::BOOL {
    use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT};
    let ours = kind == CTRL_C_EVENT || kind == CTRL_BREAK_EVENT;
    if ours {
        INTERRUPTED.store(true, Ordering::Relaxed);
    }
    windows_sys::Win32::Foundation::BOOL::from(ours)
}


/// The width of the terminal on stderr, else `COLUMNS`, else 80.
fn terminal_width() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: winsize is plain data, which TIOCGWINSZ fills.
        let mut size: libc::winsize = unsafe { std::mem::zeroed() };
        // SAFETY: the descriptor is stderr's, and `size` is the struct the request writes.
        if unsafe { libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size) } == 0 && size.ws_col > 0 {
            return size.ws_col as usize;
        }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Console::{GetConsoleScreenBufferInfo, GetStdHandle, CONSOLE_SCREEN_BUFFER_INFO, STD_ERROR_HANDLE};
        // SAFETY: the info is plain data the call fills; the handle is stderr's, which the process does not close.
        unsafe {
            let mut info: CONSOLE_SCREEN_BUFFER_INFO = std::mem::zeroed();
            if GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &mut info) != 0 {
                // The window's width, not the buffer's, which may be wider and scroll.
                let width = i32::from(info.srWindow.Right) - i32::from(info.srWindow.Left) + 1;
                if width > 0 {
                    return width as usize;
                }
            }
        }
    }
    std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok()).filter(|&c| c > 0).unwrap_or(80)
}

/// One line of the progress bar: the file, the bar and the share done when the size is known, and the megabytes;
/// at most `width` columns, the name dropped first when the terminal is narrow.
pub(crate) fn bar_line(name: &str, have: u64, total: Option<u64>, width: usize) -> String {
    let mb = |b: u64| format!("{:.1}", b as f64 / 1e6);
    let Some(total) = total.filter(|&t| t > 0) else {
        return format!("  {name}  {} MB", mb(have));
    };
    let have = have.min(total);
    let percent = (have * 100 / total) as usize;
    // The bytes done as wide as the size, so the bar keeps its length as they grow.
    let tail = format!(" {percent:>3}%  {:>w$}/{} MB", mb(have), mb(total), w = mb(total).len());
    let named = format!("  {name} ");
    // One column short of the width, so the line never wraps.
    let room = |head: &str| width.saturating_sub(head.chars().count() + tail.chars().count() + 3);
    let head = if room(&named) >= 10 { named } else { "  ".to_string() };
    let cells = room(&head).clamp(10, 40);
    let filled = cells * percent / 100;
    format!("{head}[{}{}]{tail}", "#".repeat(filled), ".".repeat(cells - filled))
}

/// The progress of a download, drawn over itself on stderr; nothing when off.
struct Bar {
    name: String,
    total: Option<u64>,
    on: bool,
    width: usize,
    drawn: usize,
    last: Option<Instant>,
}

impl Bar {
    fn new(name: &str, total: Option<u64>, on: bool) -> Bar {
        Bar { name: name.to_string(), total, on, width: if on { terminal_width() } else { 80 }, drawn: 0, last: None }
    }

    fn show(&mut self, have: u64, force: bool) {
        if !self.on || (!force && self.last.is_some_and(|t| t.elapsed() < Duration::from_millis(100))) {
            return;
        }
        self.last = Some(Instant::now());
        let line = bar_line(&self.name, have, self.total, self.width);
        let len = line.chars().count();
        // Spaces over what a longer line before left.
        eprint!("\r{line}{}", " ".repeat(self.drawn.saturating_sub(len)));
        let _ = std::io::stderr().flush();
        self.drawn = len;
    }

    fn done(&mut self, have: Option<u64>) {
        if let Some(have) = have {
            self.show(have, true);
        }
        if self.on && self.drawn > 0 {
            eprintln!();
        }
    }
}

/// Downloads `asset` into `dir`, checking its size and SHA-256, with a progress bar on stderr when `bar` and stderr is
/// a terminal: the size from the listing, against the bytes in the `.part` file curl writes. The file's path.
pub(crate) fn download(asset: &Asset, dir: &Path, bar: bool) -> Result<PathBuf, String> {
    let src = asset.url.as_str();
    let Some(want) = asset.sha256.as_deref() else { return Err(format!("{src}: no SHA-256 to check it against")) };
    let dest = dir.join(&asset.name);
    let part = dir.join(format!("{}.part", asset.name));
    let _ = std::fs::remove_file(&part);
    let mut shown = Bar::new(&asset.name, asset.size, bar && std::io::stderr().is_terminal());
    let local = src.strip_prefix("file://").or_else(|| (!src.contains("://")).then_some(src));
    let got = match local {
        Some(path) => copy_showing(Path::new(path), &part, &mut shown),
        None if src.starts_with("https://") => curl_showing(src, &part, &mut shown),
        None => Err(format!("{src}: only https:// URLs and local files are fetched")),
    };
    shown.done(got.as_ref().ok().copied());
    let placed = got.and_then(|_| std::fs::rename(&part, &dest).map_err(|e| format!("{}: {e}", dest.display())));
    if let Err(e) = placed {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    if let Some(size) = asset.size {
        let have = std::fs::metadata(&dest).map_err(|e| e.to_string())?.len();
        if have != size {
            return Err(format!("{src}: {have} bytes, expected {size}"));
        }
    }
    let sum = sha256_of(&dest)?;
    if !want.eq_ignore_ascii_case(&sum) {
        return Err(format!("{src}: SHA-256 {sum}, expected {want} (not the file its project published)"));
    }
    Ok(dest)
}

fn copy_showing(from: &Path, to: &Path, shown: &mut Bar) -> Result<u64, String> {
    let mut src = std::fs::File::open(from).map_err(|e| format!("{}: {e}", from.display()))?;
    let mut dest = std::fs::File::create(to).map_err(|e| format!("{}: {e}", to.display()))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut have = 0u64;
    loop {
        if interrupted() {
            return Err("interrupted".into());
        }
        let n = src.read(&mut buf).map_err(|e| format!("{}: {e}", from.display()))?;
        if n == 0 {
            return Ok(have);
        }
        dest.write_all(&buf[..n]).map_err(|e| format!("{}: {e}", to.display()))?;
        have += n as u64;
        shown.show(have, false);
    }
}

fn curl_showing(src: &str, part: &Path, shown: &mut Bar) -> Result<u64, String> {
    let mut cmd = Command::new("curl");
    cmd.args(["-fsSL", "--proto", "=https", "--tlsv1.2", "--retry", "3", "--connect-timeout", "30"])
        .args(["--speed-limit", "1024", "--speed-time", "60", "-o"])
        .arg(part)
        .arg(src)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = crate::exec::when_not_busy(|| cmd.spawn()).map_err(|e| format!("curl: {e} (superfluid fetches with the system curl)"))?;
    let have = || std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(e) => return Err(format!("curl: {e}")),
        }
        if interrupted() {
            let _ = child.kill();
            let _ = child.wait();
            return Err("interrupted".into());
        }
        shown.show(have(), false);
        std::thread::sleep(Duration::from_millis(100));
    };
    if !status.success() {
        let mut said = String::new();
        if let Some(mut err) = child.stderr.take() {
            let _ = err.read_to_string(&mut said);
        }
        let said = said.trim();
        return Err(match interrupted() {
            true => "interrupted".into(),
            false if said.is_empty() => format!("fetching {src} failed ({status})"),
            false => format!("fetching {src} failed: {said}"),
        });
    }
    Ok(have())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k: &str| map.get(k).cloned()
    }

    fn words(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn the_setting_is_on_until_turned_off_and_keeps_what_else_the_file_holds() {
        let home = std::env::temp_dir().join(format!("superfluid-auto-settings-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let none = env_of(&[]);
        assert!(setting(&home), "no file: on");
        assert_eq!(state(&home, &none), "automatic updates: on");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(home.join(SETTINGS), r#"{"theme": "dark"}"#).unwrap();
        assert!(setting(&home), "no field: on");
        assert_eq!(auto_update_command(&home, Some("off")), 0);
        assert!(!setting(&home));
        let kept: Value = serde_json::from_str(&std::fs::read_to_string(home.join(SETTINGS)).unwrap()).unwrap();
        assert_eq!(kept, json!({"theme": "dark", "autoUpdate": false}), "Superfluid Desktop's own keys stay");
        assert_eq!(state(&home, &none), "automatic updates: off (superfluid self auto-update off)");
        assert_eq!(auto_update_command(&home, None), 0);
        assert_eq!(auto_update_command(&home, Some("on")), 0);
        assert!(setting(&home));
        assert_eq!(auto_update_command(&home, Some("maybe")), 2);
        assert!(setting(&home), "a word that is neither changes nothing");
        assert!(!std::fs::read_dir(&home).unwrap().flatten().any(|e| e.file_name().to_string_lossy().contains(".tmp-")), "written whole, then renamed");
        // The environment wins over the file, for the run it is set in.
        assert_eq!(state(&home, &env_of(&[(NO_UPDATE_ENV, "1")])), "automatic updates: off (SUPERFLUID_NO_UPDATE is set)");
        assert_eq!(state(&home, &env_of(&[(NO_UPDATE_ENV, "0")])), "automatic updates: on", "0 turns nothing off");
        assert_eq!(state(&home, &env_of(&[(selfupdate::NO_CHECK_ENV, "1")])), "automatic updates: off (SUPERFLUID_NO_UPDATE_CHECK is set)");
        assert_eq!(state(&home, &env_of(&[("CI", "true")])), "automatic updates: off (CI is set)");
        assert_eq!(state(&home, &env_of(&[("CI", "false")])), "automatic updates: on");
        std::fs::write(home.join(SETTINGS), "not json").unwrap();
        assert!(setting(&home), "a file that cannot be read is no switch");
        set_setting(&home, false).unwrap();
        assert!(!setting(&home));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn no_update_is_taken_off_any_command_before_a_separator() {
        assert_eq!(strip_no_update(&words("serve m.gguf --no-update --port 1")), (words("serve m.gguf --port 1"), true));
        assert_eq!(strip_no_update(&words("--no-update serve m.gguf")), (words("serve m.gguf"), true));
        assert_eq!(strip_no_update(&words("serve m.gguf")), (words("serve m.gguf"), false));
        assert_eq!(
            strip_no_update(&words("launch claude -- --no-update")),
            (words("launch claude -- --no-update"), false),
            "after `--` the words are the agent's"
        );
        assert_eq!(strip_no_update(&words("launch --no-update claude -- --no-update")), (words("launch claude -- --no-update"), true));
        assert_eq!(strip_no_update(&[]), (Vec::new(), false));
    }

    #[test]
    fn only_a_command_that_serves_or_installs_in_a_terminal_updates_first() {
        let none = env_of(&[]);
        let at = |line: &str| mode(&words(line), false, true, true, &none);
        for line in ["serve m.gguf", "serve m.gguf --port 8000", "launch claude", "node join", "top", "runtime install llamacpp", "runtime update mlx", "runtime repair basert", "runtime remove llamacpp b1-cpu", "runtime use mlx --auto", "launch claude -- --json"] {
            assert_eq!(at(line), Mode::Update, "{line}");
        }
        // What only says something, or is read by a program, waits on nothing and changes nothing.
        for line in ["", "--help", "-h", "help", "help serve", "-V", "--version", "version", "self update", "self auto-update off", "self check --record", "compat", "compat --json", "stop", "runtimes", "runtimes --json", "models", "models --json", "runtime versions llamacpp", "runtime options mlx --json", "session inspect 1", "media gc", "fleet token", "serve --help", "runtime install llamacpp --help", "serve m.gguf --offline", "nonsense"] {
            assert_eq!(at(line), Mode::Quiet, "{line:?}");
        }
        let args = words("serve m.gguf");
        assert_eq!(mode(&args, true, true, true, &none), Mode::Quiet, "--no-update");
        for (k, v) in [(NO_UPDATE_ENV, "1"), (selfupdate::NO_CHECK_ENV, "1"), (UPDATED_FROM_ENV, "0.2.1")] {
            assert_eq!(mode(&args, false, true, true, &env_of(&[(k, v)])), Mode::Quiet, "{k}={v}");
        }
        assert_eq!(mode(&args, false, true, true, &env_of(&[(NO_UPDATE_ENV, "0")])), Mode::Update);
        // A service, an app, a pipe, CI, or updates turned off: the line at most.
        assert_eq!(mode(&args, false, false, true, &none), Mode::Notice, "no terminal");
        assert_eq!(mode(&args, false, true, true, &env_of(&[("CI", "true")])), Mode::Notice);
        assert_eq!(mode(&args, false, true, false, &none), Mode::Notice, "superfluid self auto-update off");
    }

    #[test]
    fn a_package_managers_update_runs_in_the_background_only_for_a_copy_it_may_replace() {
        let run = |manager: &str| Delegate::Run { manager: manager.into(), command: format!("{manager} up"), argv: vec![manager.into()], path_first: None };
        for manager in ["Homebrew", "npm", "cargo", "pipx", "uv"] {
            assert!(runs_in_background(&run(manager), true, false), "{manager}");
            assert!(!runs_in_background(&run(manager), false, false), "{manager}: another user's copy");
            assert!(!runs_in_background(&run(manager), true, true), "{manager}: Windows replaces no program that runs");
        }
        assert!(!runs_in_background(&run("basert"), true, false), "basert updates its own copy");
        let named = Delegate::Name { manager: "pip".into(), command: "pip install --upgrade superfluid-cli".into(), why: "a project's".into() };
        assert!(!runs_in_background(&named, true, false), "pip, a project's npm, a pinned formula: named, not run");
    }

    #[test]
    fn the_runtime_a_runtime_command_installs_repairs_or_removes_is_left_to_it() {
        assert_eq!(named_runtime(&words("runtime remove llamacpp")), Some(RuntimeId::new("llamacpp")));
        assert_eq!(named_runtime(&words("runtime remove mlx --all")), Some(RuntimeId::new("mlx")));
        assert_eq!(named_runtime(&words("runtime remove llamacpp@b1-cpu")), Some(RuntimeId::new("llamacpp")), "by its base id");
        assert_eq!(named_runtime(&words("runtime install basert --version 0.4")), Some(RuntimeId::new("basert")));
        assert_eq!(named_runtime(&words("runtime repair mlx 0.1-metal")), Some(RuntimeId::new("mlx")));
        // Updating it first changes nothing an update or a pin would not.
        assert_eq!(named_runtime(&words("runtime update llamacpp")), None);
        assert_eq!(named_runtime(&words("runtime use mlx --auto")), None);
        assert_eq!(named_runtime(&words("serve m.gguf")), None);
        assert_eq!(named_runtime(&words("runtime remove")), None);
    }

    #[test]
    fn the_variable_an_update_runs_a_command_again_with_is_for_that_run_alone() {
        std::env::set_var(UPDATED_FROM_ENV, "0.2.1");
        // `self list` updates nothing, so this reads no record and starts nothing.
        at_launch(&words("self list"), false);
        assert_eq!(std::env::var_os(UPDATED_FROM_ENV), None, "what the command starts updates as any other would");
    }

    #[test]
    fn the_progress_bar_fits_the_terminal_and_says_how_far_it_is() {
        let line = bar_line("superfluid-x.tar.gz", 13_500_000, Some(27_000_000), 80);
        assert!(line.starts_with("  superfluid-x.tar.gz [") && line.ends_with("]  50%  13.5/27.0 MB"), "{line}");
        assert!(line.chars().count() < 80, "{line}");
        let bar = &line[line.find('[').unwrap()..=line.find(']').unwrap()];
        assert_eq!(bar.matches('#').count(), bar.matches('.').count(), "half the bar is filled: {line}");
        let narrow = bar_line("superfluid-aarch64-apple-darwin.tar.gz", 27_000_000, Some(27_000_000), 40);
        assert!(narrow.starts_with("  [") && narrow.contains("100%"), "the name goes first: {narrow}");
        assert_eq!(bar_line("x.tar.gz", 1_200_000, None, 80), "  x.tar.gz  1.2 MB", "no size: the bytes alone");
        assert!(bar_line("x", 99, Some(10), 80).contains("100%  0.0/0.0 MB"), "never past the end");
        assert!(bar_line("x", 0, Some(10), 0).contains("[.........."), "a width of none still draws");
    }
}

// The automatic update itself, on release trees that stand in for superfluid (POSIX shell scripts).
#[cfg(all(test, unix))]
mod update_tests {
    use super::*;
    use crate::selfupdate::tests::{listing, no_compat, release_tree, scratch, seen_running_from, sleeper_at};
    use crate::selfupdate::{record, releases, target, Engine, HOLD_OFF_DAYS};
    use superfluid_adapter_kit::install::Manifest;

    fn install_of(catalog: &Catalog, id: &str, name: &str) -> PathBuf {
        let dir = catalog.package_dir(RuntimeId::new(id)).join(name);
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        let (version, backend) = name.split_once('-').unwrap();
        Manifest {
            id: id.into(),
            install: name.into(),
            version: version.into(),
            backend: backend.into(),
            link_w: vec![superfluid_agent::client::PROTO_VERSION],
            platforms: Vec::new(),
            worker: format!("bin/superfluid-worker-{id}"),
            sources: Vec::new(),
            tested: true,
            requires: None,
        }
        .write(&dir)
        .unwrap();
        dir
    }

    #[test]
    fn the_newest_release_past_the_hold_off_that_fits_the_engine_is_installed_and_the_way_back_said() {
        let dir = scratch("auto-update");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let p = listing(&dir, &[("99.0.0", "2026-01-01", Some(1)), ("99.1.0", "2026-01-02", Some(2)), ("99.2.0", "2099-01-01", Some(1))]);
        let found = releases(&serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap(), target().unwrap());
        // The record the daily read leaves: 99.2.0 is inside the hold-off, 99.1.0 drives another engine.
        let check = record(&home, &found, &Engine::Abi(1), Some(HOLD_OFF_DAYS), &no_compat).unwrap();
        assert_eq!(check.newest.as_deref(), Some("99.0.0"));
        assert_eq!(check.notes_url().as_deref(), Some("https://github.com/basecompute/superfluid/releases/tag/v99.0.0"));
        assert_eq!(Check::read(&home), Some(check.clone()), "the record reads back whole");
        assert!(!std::fs::read_dir(&home).unwrap().flatten().any(|e| e.file_name().to_string_lossy().contains(".tmp-")));
        let versions = Versions::of(&home);
        release_tree(&versions.version_dir("98.0.0"), "98.0.0");
        versions.set_current("98.0.0").unwrap();
        let exe = std::fs::canonicalize(versions.binary("98.0.0")).unwrap();
        assert_eq!(plan(&exe, Mode::Update, &check, "98.0.0", "99.0.0"), Plan::Here);
        assert_eq!(plan(&exe, Mode::Notice, &check, "98.0.0", "99.0.0"), Plan::Notice, "no terminal: the line");
        assert_eq!(plan(&exe, Mode::Update, &check, "97.0.0", "99.0.0"), Plan::Notice, "a version run by its own name stays as it is");

        // Three runtimes: one to update, one pinned, one a server runs from.
        install_of(&catalog, "llamacpp", "b1-cpu");
        install_of(&catalog, "mlx", "0.1-metal");
        install_of(&catalog, "mlx", "0.2-metal");
        installs::pin(&catalog, RuntimeId::new("mlx"), "0.1-metal").unwrap();
        let busy = install_of(&catalog, "basert", "0.4-metal");
        let _server = sleeper_at(&busy.join("bin").join("superfluid-worker-basert"));
        let seen = seen_running_from(&busy);
        // The new version cannot run llamacpp's install after its update (the build did not change, and the worker
        // in it is the old version's), nor mlx's, which is pinned and so not updated.
        let broken = |id: &str| json!({"id": id, "state": "broken", "worker": {"source": "installed"}, "ready": format!("the {id} runtime speaks Link W [1] and this superfluid speaks 2")});
        std::fs::write(home.join("runtimes.json"), json!([{"id": "basert", "ready": true}, broken("llamacpp"), broken("mlx")]).to_string()).unwrap();

        let mut lines = Vec::new();
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &check.notes_url().unwrap(), Some(p.to_string_lossy().to_string()), None, &mut |l| lines.push(l));
        assert_eq!(bin.map(|b| std::fs::canonicalize(b).unwrap()), Some(std::fs::canonicalize(versions.binary("99.0.0")).unwrap()), "{lines:?}");
        assert_eq!((versions.current().as_deref(), versions.previous().as_deref()), (Some("99.0.0"), Some("98.0.0")));
        assert_eq!(lines[0], "Updating superfluid 98.0.0 → 99.0.0 · release notes: https://github.com/basecompute/superfluid/releases/tag/v99.0.0");
        assert_eq!(
            lines.last().unwrap(),
            "superfluid 99.0.0 is in use · `superfluid self use 98.0.0` goes back · `superfluid self auto-update off` turns automatic updates off"
        );
        // The new version updated the one runtime free to be updated.
        let calls = std::fs::read_to_string(home.join("runtime-calls")).unwrap();
        let expected = if seen { "runtime update llamacpp\nruntime repair llamacpp\n" } else { "runtime update basert\nruntime update llamacpp\nruntime repair llamacpp\n" };
        assert_eq!(calls, expected, "{lines:?}");
        assert!(lines.contains(&"superfluid: updating llamacpp (b1-cpu) with superfluid 99.0.0".to_string()), "{lines:?}");
        assert!(
            lines.contains(&"superfluid: llamacpp does not run with superfluid 99.0.0 (the llamacpp runtime speaks Link W [1] and this superfluid speaks 2): installing it again with it".to_string()),
            "{lines:?}"
        );
        assert!(lines.contains(&"superfluid: mlx stays on 0.1-metal, which is pinned (`superfluid runtime update mlx` updates it)".to_string()), "{lines:?}");
        if seen {
            assert!(lines.iter().any(|l| l.starts_with("superfluid: basert stays on 0.4-metal: a running server uses it")), "{lines:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_automatic_update_that_fails_says_so_leaves_the_version_and_waits_for_the_next_read() {
        let dir = scratch("auto-update-fails");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None)]);
        let found = releases(&serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap(), target().unwrap());
        let check = record(&home, &found, &Engine::None, Some(HOLD_OFF_DAYS), &no_compat).unwrap();
        let versions = Versions::of(&home);
        release_tree(&versions.version_dir("98.0.0"), "98.0.0");
        versions.set_current("98.0.0").unwrap();
        let exe = std::fs::canonicalize(versions.binary("98.0.0")).unwrap();
        let notes = check.notes_url().unwrap();

        // Offline: the listing cannot be read.
        let mut lines = Vec::new();
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &notes, Some("/nonexistent/releases.json".into()), None, &mut |l| lines.push(l));
        assert_eq!(bin, None);
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[1].starts_with("couldn't update superfluid to 99.0.0: /nonexistent/releases.json: ") && lines[1].ends_with("; staying on 98.0.0"), "{lines:?}");
        assert_eq!(versions.current().as_deref(), Some("98.0.0"));
        let after = Check::read(&home).unwrap();
        assert_eq!(after.attempted.as_deref(), Some("99.0.0"));
        assert_eq!(plan(&exe, Mode::Update, &after, "98.0.0", "99.0.0"), Plan::Notice, "not tried again until the next daily read");
        let a_day_on = Check { attempted_at: after.attempted_at - 24 * 60 * 60, ..after.clone() };
        assert_eq!(plan(&exe, Mode::Update, &a_day_on, "98.0.0", "99.0.0"), Plan::Here);

        // Another update holds the lock: it is not waited for.
        let held = crate::pulls::Lock::take(&versions.dir, "self", "update of superfluid").unwrap();
        let began = Instant::now();
        let mut lines = Vec::new();
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &notes, Some(p.to_string_lossy().to_string()), None, &mut |l| lines.push(l));
        assert!(bin.is_none() && began.elapsed() < Duration::from_secs(5), "{lines:?}");
        assert_eq!(lines[1], "couldn't update superfluid to 99.0.0: another update of superfluid is running; staying on 98.0.0");
        drop(held);

        // A download that does not check out.
        std::fs::write(dir.join(format!("superfluid-{}-99.0.0.tar.gz.sha256", target().unwrap())), format!("{}  x\n", "0".repeat(64))).unwrap();
        let mut lines = Vec::new();
        assert!(update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &notes, Some(p.to_string_lossy().to_string()), None, &mut |l| lines.push(l)).is_none());
        assert!(lines[1].contains("SHA-256") && lines[1].ends_with("; staying on 98.0.0"), "{lines:?}");
        assert_eq!(versions.current().as_deref(), Some("98.0.0"));
        assert!(!versions.version_dir("99.0.0").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A copy Homebrew placed under `dir`, and its `brew`, which notes what it is asked to run in `dir/brew-ran`
    /// and fails while `dir/brew-fails` is there. The copy's real path.
    fn homebrew_copy(dir: &Path) -> PathBuf {
        let root = dir.join("brew/Cellar/superfluid/98.0.0/libexec");
        release_tree(&root, "98.0.0");
        std::fs::write(root.join(selfupdate::INSTALLED_BY), "Homebrew: brew upgrade superfluid\n").unwrap();
        let brew = dir.join("brew/bin/brew");
        std::fs::create_dir_all(brew.parent().unwrap()).unwrap();
        let script = format!("#!/bin/sh\necho \"$*\" >> '{0}/brew-ran'\n[ ! -e '{0}/brew-fails' ]\n", dir.display());
        std::fs::write(&brew, script).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&brew, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::canonicalize(root.join("bin/superfluid")).unwrap()
    }

    #[test]
    fn a_copy_homebrew_owns_is_updated_by_homebrew_in_the_background_once_per_release() {
        let dir = scratch("auto-update-brew");
        let home = dir.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let exe = homebrew_copy(&dir);
        let check = Check { checked_at: selfupdate::now_secs(), newest: Some("99.0.0".into()), ..Check::default() };
        check.write(&home).unwrap();
        match plan(&exe, Mode::Update, &check, "98.0.0", "99.0.0") {
            Plan::Manager(found) => assert_eq!((found.manager(), found.command()), ("Homebrew", "brew upgrade superfluid")),
            other => panic!("{other:?}"),
        }
        // The line that says the release is out (a server's hourly look, a run with no terminal) hands nothing over,
        // and a server running from the copy is waited for in the background, not a reason to hand nothing over.
        assert!(selfupdate::announce(&home, check.clone(), "brew upgrade superfluid", true).is_some());
        let announced = Check::read(&home).unwrap();
        assert_eq!(announced.notified.as_deref(), Some("99.0.0"));
        assert!(matches!(plan(&exe, Mode::Update, &announced, "98.0.0", "99.0.0"), Plan::Manager(_)), "announced is not handed over");

        let Plan::Manager(found) = plan(&exe, Mode::Update, &announced, "98.0.0", "99.0.0") else { unreachable!() };
        let line = in_background(&home, &exe, &found, "98.0.0", "99.0.0", "https://notes").unwrap();
        assert!(line.starts_with("Updating superfluid 98.0.0 → 99.0.0 with Homebrew in the background (log: "), "{line}");
        let handed = Check::read(&home).unwrap();
        assert_eq!((handed.delegated.as_deref(), handed.notified.as_deref()), (Some("99.0.0"), Some("99.0.0")));
        assert_eq!(plan(&exe, Mode::Update, &handed, "98.0.0", "99.0.0"), Plan::Notice, "once per release");
        let lost = Check { delegated_at: handed.delegated_at - WAIT_FOR_IDLE.as_secs() - DAY, ..handed.clone() };
        assert!(matches!(plan(&exe, Mode::Update, &lost, "98.0.0", "99.0.0"), Plan::Manager(_)), "an update in the background lost to a restart is handed over again");
        assert_eq!(plan(&exe, Mode::Notice, &check, "98.0.0", "99.0.0"), Plan::Notice);

        // The update in the background runs brew once nothing runs from the copy.
        assert_eq!(background_update(&home, &exe), 0);
        assert_eq!(std::fs::read_to_string(dir.join("brew-ran")).unwrap(), "upgrade superfluid\n");
        std::fs::remove_file(dir.join("brew-ran")).unwrap();

        // brew fails: tried, so handed over again after a day, and said again.
        handed.write(&home).unwrap();
        std::fs::write(dir.join("brew-fails"), "").unwrap();
        assert_eq!(background_update(&home, &exe), 1);
        let failed = Check::read(&home).unwrap();
        assert_eq!((failed.delegated.as_deref(), failed.attempted.as_deref(), failed.notified.as_deref()), (None, Some("99.0.0"), None));
        assert_eq!(plan(&exe, Mode::Update, &failed, "98.0.0", "99.0.0"), Plan::Notice, "not handed over again within the day");
        assert!(selfupdate::announce(&home, failed.clone(), "brew upgrade superfluid", true).is_some(), "the release is said to be out again");
        let a_day_on = Check { attempted_at: failed.attempted_at - DAY, ..failed };
        assert!(matches!(plan(&exe, Mode::Update, &a_day_on, "98.0.0", "99.0.0"), Plan::Manager(_)));
        std::fs::remove_file(dir.join("brew-fails")).unwrap();
        std::fs::remove_file(dir.join("brew-ran")).unwrap();

        // Turned off while it waited: brew is not run, and the release is no longer handed over.
        handed.write(&home).unwrap();
        set_setting(&home, false).unwrap();
        assert_eq!(background_update(&home, &exe), 0);
        assert!(!dir.join("brew-ran").exists(), "turned off: nothing run");
        assert_eq!(Check::read(&home).unwrap().delegated, None);
        set_setting(&home, true).unwrap();

        // Pinned, or uninstalled while a server ran from it: nothing is run, so nothing removed is put back.
        assert_eq!(not_to_run(&home, &exe, &["x".to_string()]).unwrap(), format!("{} is no longer updated by `x`", exe.display()));
        std::fs::create_dir_all(dir.join("brew/var/homebrew/pinned")).unwrap();
        std::fs::write(dir.join("brew/var/homebrew/pinned/superfluid"), "").unwrap();
        assert_eq!(plan(&exe, Mode::Update, &check, "98.0.0", "99.0.0"), Plan::Notice, "a pinned formula is named");
        let Plan::Manager(Delegate::Run { argv, .. }) = ({
            std::fs::remove_file(dir.join("brew/var/homebrew/pinned/superfluid")).unwrap();
            plan(&exe, Mode::Update, &check, "98.0.0", "99.0.0")
        }) else {
            panic!("unpinned, it is Homebrew's again")
        };
        assert_eq!(not_to_run(&home, &exe, &argv), None);
        std::fs::write(dir.join("brew/var/homebrew/pinned/superfluid"), "").unwrap();
        assert_eq!(not_to_run(&home, &exe, &argv), Some(format!("{} is no longer updated by `{}`", exe.display(), argv.join(" "))), "pinned since");
        std::fs::remove_file(&exe).unwrap();
        assert_eq!(not_to_run(&home, &exe, &argv), Some(format!("{} is gone", exe.display())));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn going_back_with_self_use_is_not_undone_by_an_automatic_update() {
        let dir = scratch("auto-update-declined");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let versions = Versions::of(&home);
        release_tree(&versions.version_dir("98.0.0"), "98.0.0");
        release_tree(&versions.version_dir("99.0.0"), "99.0.0");
        versions.set_current("99.0.0").unwrap();
        Check { checked_at: selfupdate::now_secs(), newest: Some("99.0.0".into()), ..Check::default() }.write(&home).unwrap();
        let exe = std::fs::canonicalize(versions.binary("98.0.0")).unwrap();
        selfupdate::use_version(&home, &catalog, &exe, "98.0.0").unwrap();
        let back = Check::read(&home).unwrap();
        assert_eq!(back.declined.as_deref(), Some("99.0.0"));
        let exe = std::fs::canonicalize(versions.binary("98.0.0")).unwrap();
        assert_eq!(plan(&exe, Mode::Update, &back, "98.0.0", "99.0.0"), Plan::Notice, "a day on or a month on");
        let a_day_on = Check { attempted: Some("99.0.0".into()), attempted_at: 1, ..back.clone() };
        assert_eq!(plan(&exe, Mode::Update, &a_day_on, "98.0.0", "99.0.0"), Plan::Notice);
        assert_eq!(plan(&exe, Mode::Update, &back, "98.0.0", "99.1.0"), Plan::Here, "a later release is updated to");
        // Further back keeps the later version declined; forward again to it ends the step back.
        release_tree(&versions.version_dir("97.0.0"), "97.0.0");
        selfupdate::use_version(&home, &catalog, &exe, "97.0.0").unwrap();
        assert_eq!(Check::read(&home).unwrap().declined.as_deref(), Some("99.0.0"));
        selfupdate::use_version(&home, &catalog, &exe, "98.0.0").unwrap();
        assert_eq!(Check::read(&home).unwrap().declined.as_deref(), Some("99.0.0"));
        selfupdate::use_version(&home, &catalog, &exe, "99.0.0").unwrap();
        assert_eq!(Check::read(&home).unwrap().declined, None);
        // An update ends it too.
        selfupdate::use_version(&home, &catalog, &exe, "98.0.0").unwrap();
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None)]);
        let opts = Options { version: Some("99.0.0".into()), source: Some(p.to_string_lossy().to_string()), ..Options::default() };
        assert!(selfupdate::update(&home, &catalog, &exe, &opts).unwrap().changed);
        assert_eq!(Check::read(&home).unwrap().declined, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_update_whose_links_fail_after_the_switch_goes_on_with_the_new_version() {
        let dir = scratch("auto-update-links");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None)]);
        let versions = Versions::of(&home);
        release_tree(&versions.version_dir("98.0.0"), "98.0.0");
        versions.set_current("98.0.0").unwrap();
        // A prefix this user may no longer write to: `current` moves, the new version's link cannot be made.
        let prefix = dir.join("prefix");
        std::fs::create_dir_all(&prefix).unwrap();
        std::fs::write(versions.dir.join("prefix"), format!("{}\n", prefix.display())).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&prefix, std::fs::Permissions::from_mode(0o555)).unwrap();
        if std::fs::write(prefix.join("probe"), "").is_ok() {
            eprintln!("SKIP: this user writes to a directory whose mode forbids it");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        install_of(&catalog, "llamacpp", "b1-cpu");
        let exe = std::fs::canonicalize(versions.binary("98.0.0")).unwrap();
        let mut lines = Vec::new();
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", "https://notes", Some(p.to_string_lossy().to_string()), None, &mut |l| lines.push(l));
        std::fs::set_permissions(&prefix, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(versions.current().as_deref(), Some("99.0.0"));
        assert_eq!(bin.map(|b| std::fs::canonicalize(b).unwrap()), Some(std::fs::canonicalize(versions.binary("99.0.0")).unwrap()), "{lines:?}");
        assert!(!lines.iter().any(|l| l.contains("staying on")), "{lines:?}");
        assert!(lines.iter().any(|l| l.starts_with("superfluid: ") && l.contains(&prefix.display().to_string())), "what failed is said: {lines:?}");
        assert_eq!(std::fs::read_to_string(home.join("runtime-calls")).unwrap(), "runtime update llamacpp\n", "the runtimes are updated by it");
        assert!(lines.last().unwrap().starts_with("superfluid 99.0.0 is in use"), "{lines:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_runtime_the_command_names_is_left_to_it_and_what_the_new_version_says_goes_to_stderr() {
        let dir = scratch("auto-update-skip");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let bin = release_tree(&dir.join("new"), "99.0.0").join("bin/superfluid");
        install_of(&catalog, "llamacpp", "b1-cpu");
        install_of(&catalog, "mlx", "0.1-metal");
        let mut lines = Vec::new();
        update_runtimes(&bin, &catalog, "98.0.0", "99.0.0", Some(RuntimeId::new("llamacpp")), &mut |l| lines.push(l));
        assert_eq!(std::fs::read_to_string(home.join("runtime-calls")).unwrap(), "runtime update mlx\n", "{lines:?}");
        // The new version's stdout is this command's stderr: a script `launch --print` writes, or a log, is not mixed with it.
        let mut cmd = by_new_version(Path::new("/bin/sh"), &catalog, "98.0.0", &["-c", "[ /dev/fd/1 -ef /dev/fd/2 ]"]);
        assert!(cmd.status().unwrap().success());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
