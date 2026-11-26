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
use crate::selfupdate::{self, Check, Delegate, Installed, Options, Versions};

/// Set to anything but `0`, no update is made, none is looked for, and none is announced.
pub const NO_UPDATE_ENV: &str = "SUPERFLUID_NO_UPDATE";
/// The version a command ran as before an automatic update ran it again with the new one; such a run updates nothing.
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
/// project's), a copy this user `owns`, nothing else of superfluid's running from it, and not on Windows, where a
/// program that runs cannot be replaced.
pub fn runs_in_background(found: &Delegate, owns: bool, others_running: bool, windows: bool) -> bool {
    matches!(found, Delegate::Run { manager, .. } if IN_BACKGROUND.contains(&manager.as_str())) && owns && !others_running && !windows
}

/// Whether `newer` was tried within the last day: a failed update is not tried again until the next daily read.
fn tried(check: &Check, newer: &str) -> bool {
    check.attempted.as_deref() == Some(newer) && selfupdate::now_secs().saturating_sub(check.attempted_at) < 24 * 60 * 60
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
    if mode != Mode::Update {
        return Plan::Notice;
    }
    match selfupdate::installed_as(exe) {
        // A version run by its own name (`superfluid-0.2.0`), not through `current`, is left as it is.
        Installed::Managed { versions, version } if version == from && (Versions { dir: versions.clone() }).current().as_deref() == Some(from) => match tried(check, newer) {
            true => Plan::Notice,
            false => Plan::Here,
        },
        Installed::InPlace { .. } if !tried(check, newer) => Plan::Here,
        Installed::Other(p) => match selfupdate::delegate(&p) {
            Some(found) if check.notified.as_deref() != Some(newer) => {
                let watched = selfupdate::watched_dir(&p, found.manager());
                let others = !selfupdate::superfluid_under(&watched).is_empty();
                match runs_in_background(&found, crate::sys::writable(&watched), others, cfg!(windows)) {
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
    if mode == Mode::Quiet {
        return;
    }
    let Ok(exe) = std::env::current_exe() else { return };
    let exe = crate::sys::canonical(&exe).unwrap_or(exe);
    let check = Check::read(&home).unwrap_or_default();
    if check.due() {
        refresh_in_background(&home, &exe, &check);
    }
    let from = selfupdate::running();
    let Some(newer) = check.newer_than(from).map(str::to_string) else { return };
    let notes = check.notes_url().unwrap_or_default();
    match plan(&exe, mode, &check, from, &newer) {
        Plan::Here => {
            let catalog = Catalog::from_env();
            let mut say = |line: String| eprintln!("{line}");
            if let Some(bin) = update_here(&home, &catalog, &exe, from, &newer, &notes, None, &mut say) {
                run_again(&bin, args, from, &newer);
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
fn refresh_in_background(home: &Path, exe: &Path, check: &Check) {
    // A home that is not there yet is not made for this: a command that fails as it starts writes nothing, and the
    // first command that keeps something there (a server, an install) has the next one read the listing.
    if !home.is_dir() {
        return;
    }
    if (Check { checked_at: selfupdate::now_secs(), ..check.clone() }).write(home).is_err() {
        return;
    }
    let mut cmd = Command::new(exe);
    cmd.args(["self", "check", "--record"])
        .env(selfupdate::NO_CHECK_ENV, "1")
        .env(NO_UPDATE_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    crate::sys::detached(&mut cmd);
    if let Ok(mut child) = cmd.spawn() {
        // Waited for, so it leaves no zombie while a long command runs.
        let _ = std::thread::Builder::new().name("update-check".into()).spawn(move || child.wait());
    }
}

/// The command run again by the new version, in this process's place; only returns when it could not be.
fn run_again(bin: &Path, args: &[String], from: &str, newer: &str) {
    let _ = std::io::stdout().flush();
    let mut cmd = Command::new(bin);
    cmd.args(args).env(UPDATED_FROM_ENV, from);
    let e = crate::sys::replace_process(&mut cmd);
    eprintln!("superfluid: superfluid {newer} could not be started ({e}); going on with {from}");
}


/// The automatic update of `exe`, the version `from`, to `newer`, whose notes are at `notes`: the release placed and
/// made current as `self update --version <newer>` does it (without waiting on another update), the runtimes that
/// are neither pinned nor in use updated by the new version, and a line each for going back and turning updates
/// off. The new program, or `None` when the update failed, which a line says.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_here(home: &Path, catalog: &Catalog, exe: &Path, from: &str, newer: &str, notes: &str, source: Option<String>, say: &mut dyn FnMut(String)) -> Option<PathBuf> {
    say(format!("Updating superfluid {from} → {newer} · release notes: {notes}"));
    mark_attempt(home, newer);
    let versions = match selfupdate::installed_as(exe) {
        Installed::Managed { versions, .. } => Versions { dir: versions },
        _ => Versions::of(home),
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
        Err(e) => {
            say(failed(if interrupted() { "interrupted".to_string() } else { e }));
            return None;
        }
    };
    for note in &report.notes {
        say(format!("superfluid: {note}"));
    }
    update_runtimes(&bin, catalog, from, newer, say);
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

/// `runtime update <id>` by the new superfluid `bin` for each runtime installed here, which brings a build its
/// workers speak to. A runtime pinned to an install, or whose default install a running server uses, stays.
fn update_runtimes(bin: &Path, catalog: &Catalog, from: &str, newer: &str, say: &mut dyn FnMut(String)) {
    for id in installed_runtimes(catalog) {
        if interrupted() {
            say("superfluid: the runtimes were left as they are: interrupted".to_string());
            return;
        }
        let Some(current) = installs::default(catalog, id) else { continue };
        if let Some(pin) = installs::pinned(catalog, id) {
            say(format!("superfluid: {id} stays on {pin}, which is pinned (`superfluid runtime update {id}` updates it)"));
            continue;
        }
        if !selfupdate::pids_from(&current.dir).is_empty() {
            say(format!("superfluid: {id} stays on {}: a running server uses it (`superfluid runtime update {id}` updates it once none does)", current.name));
            continue;
        }
        say(format!("superfluid: updating {id} ({}) with superfluid {newer}", current.name));
        let mut cmd = Command::new(bin);
        cmd.args(["runtime", "update", id.name()])
            .env("SUPERFLUID_HOME", catalog.home())
            .env(NO_UPDATE_ENV, "1")
            .env(UPDATED_FROM_ENV, from)
            .stdin(Stdio::null());
        match crate::exec::when_not_busy(|| cmd.status()) {
            Ok(status) if status.success() => {}
            Ok(status) => say(format!("superfluid: {id} was not updated ({status}); `superfluid runtime update {id}` tries again")),
            Err(e) => say(format!("superfluid: {id} was not updated ({e}); `superfluid runtime update {id}` tries again")),
        }
    }
}


/// `$SUPERFLUID_HOME/logs/auto-update.log`, where a package manager's update run in the background writes.
fn log_path(home: &Path) -> PathBuf {
    home.join("logs").join("auto-update.log")
}

/// Starts `superfluid self background-update` from the copy, detached, its output appended to the log: it runs the
/// manager's update once nothing runs from the copy (this command included, which runs from it until it ends). The
/// release is marked announced, so this is done once per release. The line to print.
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
    crate::sys::detached(&mut cmd);
    let mut child = crate::exec::when_not_busy(|| cmd.spawn()).map_err(|e| e.to_string())?;
    let _ = std::thread::Builder::new().name("auto-update".into()).spawn(move || child.wait());
    let check = Check::read(home).unwrap_or_default();
    let _ = Check { notified: Some(newer.to_string()), ..check }.write(home);
    Ok(format!(
        "Updating superfluid {from} → {newer} with {} in the background (log: {}) · release notes: {notes} · turn off: {OFF_COMMAND}",
        found.manager(),
        log.display()
    ))
}

/// `superfluid self background-update`: waits until no other superfluid runs from this copy, then runs the update
/// of the package manager that owns it, all of it into the log the launch opened; the exit code.
pub fn background_update(exe: &Path) -> i32 {
    let exe = crate::sys::canonical(exe).unwrap_or_else(|_| exe.to_path_buf());
    let Some(Delegate::Run { manager, argv, path_first, .. }) = selfupdate::delegate(&exe) else {
        eprintln!("superfluid: {} is not a copy superfluid updates with its package manager", exe.display());
        return 1;
    };
    if !IN_BACKGROUND.contains(&manager.as_str()) {
        eprintln!("superfluid: {manager}'s update is not run in the background");
        return 1;
    }
    let watched = selfupdate::watched_dir(&exe, &manager);
    let began = Instant::now();
    let mut said = false;
    loop {
        let busy = selfupdate::superfluid_under(&watched);
        if busy.is_empty() {
            break;
        }
        if began.elapsed() > WAIT_FOR_IDLE {
            eprintln!("superfluid: gave up: superfluid still runs from {} (pid {busy:?})", watched.display());
            return 1;
        }
        if !said {
            eprintln!("superfluid: waiting until no superfluid runs from {} (pid {busy:?})", watched.display());
            said = true;
        }
        std::thread::sleep(Duration::from_secs(15));
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
            1
        }
        Err(e) => {
            eprintln!("superfluid: running {}: {e}", argv[0]);
            1
        }
    }
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
            assert!(runs_in_background(&run(manager), true, false, false), "{manager}");
            assert!(!runs_in_background(&run(manager), false, false, false), "{manager}: another user's copy");
            assert!(!runs_in_background(&run(manager), true, true, false), "{manager}: a server runs from the copy");
            assert!(!runs_in_background(&run(manager), true, false, true), "{manager}: Windows replaces no program that runs");
        }
        assert!(!runs_in_background(&run("basert"), true, false, false), "basert updates its own copy");
        let named = Delegate::Name { manager: "pip".into(), command: "pip install --upgrade superfluid-cli".into(), why: "a project's".into() };
        assert!(!runs_in_background(&named, true, false, false), "pip, a project's npm, a pinned formula: named, not run");
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

        let mut lines = Vec::new();
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &check.notes_url().unwrap(), Some(p.to_string_lossy().to_string()), &mut |l| lines.push(l));
        assert_eq!(bin.map(|b| std::fs::canonicalize(b).unwrap()), Some(std::fs::canonicalize(versions.binary("99.0.0")).unwrap()), "{lines:?}");
        assert_eq!((versions.current().as_deref(), versions.previous().as_deref()), (Some("99.0.0"), Some("98.0.0")));
        assert_eq!(lines[0], "Updating superfluid 98.0.0 → 99.0.0 · release notes: https://github.com/basecompute/superfluid/releases/tag/v99.0.0");
        assert_eq!(
            lines.last().unwrap(),
            "superfluid 99.0.0 is in use · `superfluid self use 98.0.0` goes back · `superfluid self auto-update off` turns automatic updates off"
        );
        // The new version updated the one runtime free to be updated.
        let calls = std::fs::read_to_string(home.join("runtime-calls")).unwrap();
        let expected = if seen { "runtime update llamacpp\n" } else { "runtime update basert\nruntime update llamacpp\n" };
        assert_eq!(calls, expected, "{lines:?}");
        assert!(lines.contains(&"superfluid: updating llamacpp (b1-cpu) with superfluid 99.0.0".to_string()), "{lines:?}");
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
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &notes, Some("/nonexistent/releases.json".into()), &mut |l| lines.push(l));
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
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &notes, Some(p.to_string_lossy().to_string()), &mut |l| lines.push(l));
        assert!(bin.is_none() && began.elapsed() < Duration::from_secs(5), "{lines:?}");
        assert_eq!(lines[1], "couldn't update superfluid to 99.0.0: another update of superfluid is running; staying on 98.0.0");
        drop(held);

        // A download that does not check out.
        std::fs::write(dir.join(format!("superfluid-{}-99.0.0.tar.gz.sha256", target().unwrap())), format!("{}  x\n", "0".repeat(64))).unwrap();
        let mut lines = Vec::new();
        assert!(update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &notes, Some(p.to_string_lossy().to_string()), &mut |l| lines.push(l)).is_none());
        assert!(lines[1].contains("SHA-256") && lines[1].ends_with("; staying on 98.0.0"), "{lines:?}");
        assert_eq!(versions.current().as_deref(), Some("98.0.0"));
        assert!(!versions.version_dir("99.0.0").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_copy_homebrew_owns_is_updated_by_homebrew_in_the_background_once_per_release() {
        let dir = scratch("auto-update-brew");
        let root = dir.join("brew/Cellar/superfluid/98.0.0/libexec");
        release_tree(&root, "98.0.0");
        std::fs::write(root.join(selfupdate::INSTALLED_BY), "Homebrew: brew upgrade superfluid\n").unwrap();
        let brew = dir.join("brew/bin/brew");
        std::fs::create_dir_all(brew.parent().unwrap()).unwrap();
        std::fs::write(&brew, "#!/bin/sh\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&brew, std::fs::Permissions::from_mode(0o755)).unwrap();
        let exe = std::fs::canonicalize(root.join("bin/superfluid")).unwrap();
        let check = Check { checked_at: selfupdate::now_secs(), newest: Some("99.0.0".into()), ..Check::default() };
        match plan(&exe, Mode::Update, &check, "98.0.0", "99.0.0") {
            Plan::Manager(found) => assert_eq!((found.manager(), found.command()), ("Homebrew", "brew upgrade superfluid")),
            other => panic!("{other:?}"),
        }
        let announced = Check { notified: Some("99.0.0".into()), ..check.clone() };
        assert_eq!(plan(&exe, Mode::Update, &announced, "98.0.0", "99.0.0"), Plan::Notice, "once per release");
        assert_eq!(plan(&exe, Mode::Notice, &check, "98.0.0", "99.0.0"), Plan::Notice);
        std::fs::create_dir_all(dir.join("brew/var/homebrew/pinned")).unwrap();
        std::fs::write(dir.join("brew/var/homebrew/pinned/superfluid"), "").unwrap();
        assert_eq!(plan(&exe, Mode::Update, &check, "98.0.0", "99.0.0"), Plan::Notice, "a pinned formula is named");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
