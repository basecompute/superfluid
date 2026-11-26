//! Automatic updates: the setting and the switches that turn them off, the daily read a launch starts in the
//! background, and the update a command run in a terminal makes before it goes on (`superfluid self` has the rest).
//!
//! A command that serves or installs (`serve`, `launch`, `node`, `top`, `runtime install|update|repair|remove|use`)
//! looks at what the last read of the release listing found. When it is due again, a detached
//! `superfluid self check --record` reads it, so the command never waits on the network. When the record names a
//! release newer than this one (out as long as the wait in days asks, and fitting the engine here, as the daily
//! notice's), a copy under `versions` or the install script's is updated in the foreground, with a progress bar, and
//! the command is run again by the new version; a copy a package manager owns is left to that manager, run in the
//! background once nothing runs from the copy. Without a terminal, under CI, or with updates turned off, one line
//! per release says a newer one is out.

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
use crate::selfupdate::{self, Check, Delegate, Engine, Installed, Options, Release, Report, Versions};

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
/// `$SUPERFLUID_HOME/settings.json`: `{"autoUpdate": true|false, "updateWaitDays": <days>}`, on and no wait when the
/// file or a field is missing.
const SETTINGS: &str = "settings.json";
const AUTO_UPDATE_KEY: &str = "autoUpdate";
const WAIT_KEY: &str = "updateWaitDays";
/// A whole number of days: the wait for the environment it is set in (a shell, a service), over the setting.
pub const WAIT_ENV: &str = "SUPERFLUID_UPDATE_WAIT_DAYS";
/// The longest wait, in days.
pub const MAX_WAIT_DAYS: u32 = 365;
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

/// `settings.json` as JSON; `None` when it is missing or cannot be read, which leaves every setting as it is by default.
fn settings(home: &Path) -> Option<Value> {
    std::fs::read_to_string(home.join(SETTINGS)).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
}

/// The setting in `settings.json`: on unless it says `false`.
pub fn setting(home: &Path) -> bool {
    settings(home).and_then(|v| v[AUTO_UPDATE_KEY].as_bool()).unwrap_or(true)
}

/// Sets `key` in `settings.json`, keeping whatever else the file holds (Superfluid Desktop's own keys).
fn set_key(home: &Path, key: &str, value: Value) -> Result<(), String> {
    std::fs::create_dir_all(home).map_err(|e| format!("{}: {e}", home.display()))?;
    let path = home.join(SETTINGS);
    let mut v = settings(home).filter(Value::is_object).unwrap_or_else(|| json!({}));
    v[key] = value;
    write_atomically(&path, format!("{v:#}\n").as_bytes())
}

/// Sets the setting, keeping whatever else the file holds.
pub fn set_setting(home: &Path, on: bool) -> Result<(), String> {
    set_key(home, AUTO_UPDATE_KEY, Value::Bool(on))
}

/// Sets the wait, in days, keeping whatever else the file holds.
pub fn set_wait(home: &Path, days: u32) -> Result<(), String> {
    set_key(home, WAIT_KEY, json!(days))
}

/// A wait as `self auto-update wait` or the variable gives it: a whole number of days from 0 to 365.
pub fn wait_of(text: &str) -> Option<u32> {
    text.trim().parse::<u32>().ok().filter(|&d| d <= MAX_WAIT_DAYS)
}

/// How long after a release is published automatic updates wait before they install it, or say it is out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wait {
    pub days: u32,
    /// Where it came from, when not the setting as written (or no setting): the variable, or a value that was not
    /// used and why. The state line says it.
    pub note: Option<String>,
}

/// The wait here: `SUPERFLUID_UPDATE_WAIT_DAYS` when it holds one, else `updateWaitDays` in `settings.json`, else none.
/// A setting that is not a whole number from 0 to 365 counts as missing, as any setting the file holds wrong does;
/// so does a variable, which leaves the setting in force.
pub fn wait(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> Wait {
    let given = env(WAIT_ENV).filter(|v| !v.is_empty());
    if let Some(days) = given.as_deref().and_then(wait_of) {
        return Wait { days, note: Some(format!("{WAIT_ENV} is set")) };
    }
    let mut notes = Vec::new();
    if let Some(v) = given {
        notes.push(format!("{WAIT_ENV}={v} is not a whole number of days from 0 to {MAX_WAIT_DAYS}, so the setting is used"));
    }
    let days = match settings(home).map(|v| v[WAIT_KEY].clone()).unwrap_or(Value::Null) {
        Value::Null => 0,
        v => match v.as_u64().filter(|&d| d <= u64::from(MAX_WAIT_DAYS)) {
            Some(d) => d as u32,
            None => {
                notes.push(format!("{WAIT_KEY} {v} in {} is not a whole number of days from 0 to {MAX_WAIT_DAYS}, so there is none", home.join(SETTINGS).display()));
                0
            }
        },
    };
    Wait { days, note: (!notes.is_empty()).then(|| notes.join("; ")) }
}

/// `1 day`, `3 days`.
pub fn days(n: u32) -> String {
    match n {
        1 => "1 day".to_string(),
        n => format!("{n} days"),
    }
}

/// What the wait means, for the state line.
pub fn wait_phrase(n: u32) -> String {
    match n {
        0 => "a release installs as soon as it is out".to_string(),
        n => format!("a release installs once it has been out {}", days(n)),
    }
}

/// What `self auto-update wait <n>` prints.
pub fn wait_set(n: u32) -> String {
    match n {
        0 => "automatic updates install a release as soon as it is out".to_string(),
        n => format!("automatic updates wait {} after a release", days(n)),
    }
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

/// `automatic updates: on`, or `off (<why>)`, then the wait: `· a release installs as soon as it is out`, or `once it
/// has been out 3 days`, with where it came from when not the setting.
pub fn state(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> String {
    let on = match enabled(home, env) {
        Ok(()) => "automatic updates: on".to_string(),
        Err(why) => format!("automatic updates: off ({why})"),
    };
    let w = wait(home, env);
    let from = w.note.map(|n| format!(" ({n})")).unwrap_or_default();
    format!("{on} · {}{from}", wait_phrase(w.days))
}

/// `superfluid self auto-update [on|off|wait <days>]`, given the words after `auto-update`; the exit code. Superfluid
/// Desktop runs it with `on` or `off`, and `wait <days>`, to follow its own settings.
pub fn auto_update_command(home: &Path, args: &[&str]) -> i32 {
    let env = |k: &str| std::env::var(k).ok();
    let on = match args {
        [] => {
            println!("{}", state(home, &env));
            return 0;
        }
        ["on"] => true,
        ["off"] => false,
        ["wait", n] => return wait_command(home, n, &env),
        _ => {
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

/// `superfluid self auto-update wait <n>`; the exit code.
fn wait_command(home: &Path, n: &str, env: &dyn Fn(&str) -> Option<String>) -> i32 {
    let Some(n) = wait_of(n) else {
        eprintln!("superfluid: the wait is a whole number of days from 0 to {MAX_WAIT_DAYS}, not `{n}`");
        return 2;
    };
    if let Err(e) = set_wait(home, n) {
        eprintln!("superfluid: {e}");
        return 1;
    }
    println!("{}", wait_set(n));
    let here = wait(home, env);
    if here.days != n {
        eprintln!("superfluid: {WAIT_ENV} is set, which makes the wait {} here", days(here.days));
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
    let wait = wait(&home, &env).days;
    let check = Check::read(&home).unwrap_or_default();
    let from = selfupdate::running();
    // The record answers for this wait (the setting changed since, or the variable differs) from the releases it
    // keeps, so environments with different waits share the one read a day; one from before they were kept, made
    // under another wait, names nothing for this one, and is read again once.
    let due = check.due() || !check.answers(wait);
    let check = check.under(wait);
    let newer = check.newer_than(from).map(str::to_string);
    let planned = newer.as_deref().map(|n| plan(&exe, mode, &check, from, n));
    // The daily read is started once the plan is made, and not before an update made here: that update reads the
    // listing itself, and the command it runs again in this process's place would be left a child it never waits for.
    if due && planned != Some(Plan::Here) {
        refresh_in_background(&home, &exe, wait);
    }
    let (Some(newer), Some(planned)) = (newer, planned) else { return };
    let notes = check.notes_url().unwrap_or_default();
    match planned {
        Plan::Here => {
            let catalog = Catalog::from_env();
            let mut say = |line: String| eprintln!("{line}");
            match update_here(&home, &catalog, &exe, from, &newer, &notes, None, named_runtime(args), &mut Vec::new(), &mut say) {
                Some(bin) => run_again(&bin, args, from, &newer),
                None if due => refresh_in_background(&home, &exe, wait),
                None => {}
            }
        }
        Plan::Manager(found) => match in_background(&home, &exe, &found, from, &newer, &notes) {
            Ok(line) => eprintln!("{line}"),
            Err(e) => {
                eprintln!("superfluid: {} could not be started in the background ({e})", found.command());
                notice(&home, &exe, wait);
            }
        },
        Plan::Notice => notice(&home, &exe, wait),
    }
}

/// The one line per release, from the record as it is now (a launch may have claimed the next read), as the wait here
/// reads it: what an automatic update installs and what the line says is out are the same release.
fn notice(home: &Path, exe: &Path, wait: u32) {
    let Some(check) = Check::read(home).map(|c| c.under(wait)) else { return };
    let auto = enabled(home, &|k| std::env::var(k).ok()).is_ok();
    if let Some(line) = selfupdate::announce(home, check, &selfupdate::update_command(exe), auto) {
        eprintln!("superfluid: {line}");
    }
}

/// The daily read, claimed first (so the launches after this one do not start it again) and made by a detached
/// `superfluid self check --record` in a session of its own, its streams closed: the command goes on at once. It
/// inherits this environment, so it reads the listing under the same `wait`.
fn refresh_in_background(home: &Path, exe: &Path, wait: u32) {
    // A home that is not there yet is not made for this: a command that fails as it starts writes nothing, and the
    // first command that keeps something there (a server, an install) has the next one read the listing.
    if !home.is_dir() {
        return;
    }
    // The record as it is now, so what was written since this command read it (an attempt) is kept.
    let check = Check::read(home).unwrap_or_default();
    if claimed(check, wait).write(home).is_err() {
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

/// The record a launch writes as it starts the read: marked read now and, when it cannot answer for this wait (one from
/// before the releases were kept, made under another), naming no release, which that wait chose, until the read says
/// which this one takes.
fn claimed(check: Check, wait: u32) -> Check {
    let check = Check { checked_at: selfupdate::now_secs(), ..check };
    match check.answers(wait) {
        true => check,
        false => Check { wait: Some(wait), newest: None, published: None, notes: None, ..check },
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
/// line each for going back and turning updates off, what became of each runtime added to `ran`. The new program, or
/// `None` when the update failed, which a line says.
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
    ran: &mut Vec<Value>,
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
    ran.extend(update_runtimes(&bin, catalog, from, newer, skip, say));
    drop(caught);
    say(format!("superfluid {newer} is in use · `superfluid self use {from}` goes back · `{OFF_COMMAND}` turns automatic updates off"));
    Some(bin)
}

/// The version `self update --auto` updates from: `current` for a copy under `versions` (an app may run a version by
/// its own name), else this one.
fn version_here(exe: &Path) -> String {
    match selfupdate::installed_as(exe) {
        Installed::Managed { versions, .. } => (Versions { dir: versions }).current().unwrap_or_else(|| selfupdate::running().to_string()),
        _ => selfupdate::running().to_string(),
    }
}

/// Why `self update --auto` leaves the copy `exe` as it is: another tool installed it, and updates it.
fn not_self_managed(exe: &Path) -> Option<String> {
    match selfupdate::installed_as(exe) {
        Installed::Other(p) => Some(format!("{}; `superfluid self update --auto` updates only a copy the install script or `superfluid self update` placed", selfupdate::not_ours(&p))),
        _ => None,
    }
}

/// What an automatic update installs from `found`, the listing just read, over `from`: the newest release that fits
/// the engine here and has been out `wait` days, unless `self use` went back from it; else why none, as a line.
pub(crate) fn eligible<'a>(found: &'a [Release], engine: &Engine, compat: &dyn Fn(&str) -> Option<Value>, check: &Check, from: &str, wait: u32) -> Result<&'a Release, String> {
    let newer = |r: &&Release| selfupdate::later(&r.version, from);
    match selfupdate::newest(found, engine, Some(wait), compat).filter(newer) {
        Some(r) if declined(check, &r.version) => {
            Err(format!("you went back from {} with self use; it stays until a newer release is out", check.declined.as_deref().unwrap_or(&r.version)))
        }
        Some(r) => Ok(r),
        None => match selfupdate::newest(found, engine, None, compat).filter(newer) {
            Some(r) => Err(format!(
                "superfluid {} (published {}) is held off: automatic updates wait {} after a release; `superfluid self update` installs it now",
                r.version,
                r.published,
                days(wait)
            )),
            None => Err(format!("superfluid {from} is up to date for automatic updates")),
        },
    }
}

/// `superfluid self update --auto`: the update a launch makes, for an app or a script to run (Superfluid Desktop),
/// with no terminal and nothing run again. With automatic updates on, the listing is read now, the release `eligible`
/// picks is installed as at a launch, and the runtimes are updated, and repaired, by it. `out` takes the line saying
/// nothing was to be updated, and why; `say` the update's own lines. The exit code: 0 when updated or nothing was to
/// be, 1 when the update failed (the version here stays current), 3 for a copy another tool updates; and what
/// `--json` prints: `{from, installed, result, reason, notes, runtimes}`, `result` one of `updated`, `none`, `off`,
/// `failed` and `not-ours`, `runtimes` what became of each (`update_runtimes`).
pub fn update_auto(home: &Path, catalog: &Catalog, exe: &Path, source: &str, env: &dyn Fn(&str) -> Option<String>, out: &mut dyn FnMut(String), say: &mut dyn FnMut(String)) -> (i32, Value) {
    let from = version_here(exe);
    let summary = |result: &str, installed: Option<&str>, reason: Option<String>, notes: Option<&str>, runtimes: Vec<Value>| {
        json!({"from": from, "installed": installed, "result": result, "reason": reason, "notes": notes, "runtimes": runtimes})
    };
    if let Err(why) = enabled(home, env) {
        let line = format!("automatic updates are off ({why})");
        out(line.clone());
        return (0, summary("off", None, Some(line), None, Vec::new()));
    }
    if let Some(why) = not_self_managed(exe) {
        say(format!("superfluid: {why}"));
        return (3, summary("not-ours", None, Some(why), None, Vec::new()));
    }
    let found = match selfupdate::target().ok_or_else(|| "no release is published for this platform".to_string()).and_then(|t| selfupdate::fetch_releases(source, t, 30)) {
        Ok(found) => found,
        Err(e) => {
            say(format!("superfluid: {e}"));
            return (1, summary("failed", None, Some(e), None, Vec::new()));
        }
    };
    let engine = selfupdate::engine_here(catalog);
    let compat = selfupdate::compat_fetcher(15);
    let wait = wait(home, env).days;
    // What was read is the day's read, too.
    let _ = selfupdate::record(home, &found, &engine, wait, &compat);
    let check = Check::read(home).unwrap_or_default();
    let chosen = match eligible(&found, &engine, &compat, &check, &from, wait) {
        Ok(r) => r,
        Err(why) => {
            out(why.clone());
            return (0, summary("none", None, Some(why), None, Vec::new()));
        }
    };
    let (mut ran, mut last) = (Vec::new(), None);
    let mut saying = |line: String| {
        last = Some(line.clone());
        say(line)
    };
    let notes = Some(chosen.notes_url.as_str());
    match update_here(home, catalog, exe, &from, &chosen.version, &chosen.notes_url, Some(source.to_string()), None, &mut ran, &mut saying) {
        Some(_) => (0, summary("updated", Some(&chosen.version), None, notes, ran)),
        None => (1, summary("failed", None, last, notes, ran)),
    }
}

/// `superfluid self check --json`, from the listing read now: the version here, the newest release that fits here
/// and is newer (`self update` installs it) and its notes, the one `self update --auto` would install (`auto`) or why
/// none (`autoReason`), whether automatic updates are on, and the wait, in days.
pub fn check_json(home: &Path, catalog: &Catalog, exe: &Path, source: &str, env: &dyn Fn(&str) -> Option<String>) -> Result<Value, String> {
    let target = selfupdate::target().ok_or("no release is published for this platform")?;
    let found = selfupdate::fetch_releases(source, target, 30)?;
    let engine = selfupdate::engine_here(catalog);
    let compat = selfupdate::compat_fetcher(15);
    let wait = wait(home, env).days;
    let _ = selfupdate::record(home, &found, &engine, wait, &compat);
    let check = Check::read(home).unwrap_or_default();
    let from = version_here(exe);
    let newest = selfupdate::newest(&found, &engine, None, &compat).filter(|r| selfupdate::later(&r.version, &from));
    let on = enabled(home, env);
    let auto = match (&on, not_self_managed(exe)) {
        (Err(why), _) => Err(format!("automatic updates are off ({why})")),
        (_, Some(why)) => Err(why),
        _ => eligible(&found, &engine, &compat, &check, &from, wait).map(|r| r.version.clone()),
    };
    Ok(json!({
        "current": from,
        "newest": newest.map(|r| &r.version),
        "auto": auto.as_ref().ok(),
        "autoReason": auto.as_ref().err(),
        "notes": newest.map(|r| &r.notes_url),
        "autoUpdate": on.is_ok(),
        "waitDays": wait,
    }))
}

/// `superfluid self check`, from the listing read now (and recorded as the day's read): the newest release that fits
/// here and is newer than this one, which `self update` installs, and, when an automatic update would take another
/// (or none yet), which, or why none, under the wait here.
pub fn check_lines(home: &Path, catalog: &Catalog, exe: &Path, source: &str, env: &dyn Fn(&str) -> Option<String>) -> Result<Vec<String>, String> {
    let target = selfupdate::target().ok_or("no release is published for this platform")?;
    let found = selfupdate::fetch_releases(source, target, 5)?;
    let engine = selfupdate::engine_here(catalog);
    let compat = selfupdate::compat_fetcher(5);
    let wait = wait(home, env).days;
    let check = selfupdate::record(home, &found, &engine, wait, &compat)?;
    Ok(described(&found, &engine, &compat, &check, &version_here(exe), wait))
}

/// The lines of `self check`, from the listing `found`, for the version `from` under `wait`.
pub(crate) fn described(found: &[Release], engine: &Engine, compat: &dyn Fn(&str) -> Option<Value>, check: &Check, from: &str, wait: u32) -> Vec<String> {
    let Some(newest) = selfupdate::newest(found, engine, None, compat).filter(|r| selfupdate::later(&r.version, from)) else {
        return vec![format!("no release newer than {from} fits here")];
    };
    let mut lines = vec![format!("newest release: {} (published {}): `superfluid self update` installs it", newest.version, newest.published)];
    match eligible(found, engine, compat, check, from, wait) {
        Ok(r) if r.version == newest.version => {}
        Ok(r) => lines.push(format!("an automatic update takes {} (published {}): automatic updates wait {} after a release", r.version, r.published, days(wait))),
        Err(why) => lines.push(why),
    }
    lines
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
/// install, one whose default install a running process uses, and `skip` stay. What became of each, as
/// `self update --auto --json` reports it: `{id, outcome, from, to, reason}`, the outcome `updated` (another install),
/// `unchanged`, `repaired`, `pinned`, `in-use` or `failed` (the install it had stays, `reason` says why).
fn update_runtimes(bin: &Path, catalog: &Catalog, from: &str, newer: &str, skip: Option<RuntimeId>, say: &mut dyn FnMut(String)) -> Vec<Value> {
    let row = |id: RuntimeId, outcome: &str, was: &str, to: Option<String>, reason: Option<String>| {
        json!({"id": id.name(), "outcome": outcome, "from": was, "to": to, "reason": reason})
    };
    let mut ran = Vec::new();
    let mut updated = Vec::new();
    for id in installed_runtimes(catalog) {
        if interrupted() {
            say("superfluid: the runtimes were left as they are: interrupted".to_string());
            return ran;
        }
        if Some(id) == skip {
            continue;
        }
        let Some(current) = installs::default(catalog, id) else { continue };
        if let Some(pin) = installs::pinned(catalog, id) {
            let why = format!("{id} stays on {pin}, which is pinned (`superfluid runtime update {id}` updates it)");
            say(format!("superfluid: {why}"));
            ran.push(row(id, "pinned", &current.name, None, Some(why)));
            continue;
        }
        if !selfupdate::pids_using(&current.dir).is_empty() {
            let why = format!("{id} stays on {}: a running server uses it (`superfluid runtime update {id}` updates it once none does)", current.name);
            say(format!("superfluid: {why}"));
            ran.push(row(id, "in-use", &current.name, None, Some(why)));
            continue;
        }
        say(format!("superfluid: updating {id} ({}) with superfluid {newer}", current.name));
        let mut cmd = by_new_version(bin, catalog, from, &["runtime", "update", id.name()]);
        let failed = match crate::exec::when_not_busy(|| cmd.status()) {
            Ok(status) if status.success() => None,
            Ok(status) => Some(status.to_string()),
            Err(e) => Some(e.to_string()),
        };
        if let Some(e) = failed {
            let why = format!("{id} was not updated ({e}); `superfluid runtime update {id}` tries again");
            say(format!("superfluid: {why}"));
            ran.push(row(id, "failed", &current.name, None, Some(why)));
            continue;
        }
        let now = installs::default(catalog, id).map(|i| i.name);
        let outcome = if now.as_deref() == Some(current.name.as_str()) { "unchanged" } else { "updated" };
        ran.push(row(id, outcome, &current.name, now, None));
        updated.push(id);
    }
    if updated.is_empty() || interrupted() {
        return ran;
    }
    for (id, why) in not_run_by(bin, catalog, from).into_iter().filter(|(id, _)| updated.contains(id)) {
        say(format!("superfluid: {id} does not run with superfluid {newer} ({why}): installing it again with it"));
        let mut cmd = by_new_version(bin, catalog, from, &["runtime", "repair", id.name()]);
        let failed = match crate::exec::when_not_busy(|| cmd.status()) {
            Ok(status) if status.success() => None,
            Ok(status) => Some(status.to_string()),
            Err(e) => Some(e.to_string()),
        };
        let Some(r) = ran.iter_mut().find(|r| r["id"] == id.name()) else { continue };
        r["to"] = json!(installs::default(catalog, id).map(|i| i.name));
        match failed {
            None => r["outcome"] = json!("repaired"),
            Some(e) => {
                let why = format!("{id} was not installed again ({e}); `superfluid runtime repair {id}` tries again");
                say(format!("superfluid: {why}"));
                r["outcome"] = json!("failed");
                r["reason"] = json!(why);
            }
        }
    }
    ran
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
    cmd.args(["self", "background-update", "--version", newer])
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

/// `superfluid self background-update --version <v>`: waits until no other superfluid runs from this copy, then runs
/// the update of the package manager that owns it, pinned to `version` (the release the daily read chose, out as long
/// as the wait asks and fitting the engine here; without it, the one the record says was handed over), all of it into
/// the log the launch opened; the exit code. It runs nothing when automatic updates were turned off meanwhile, the
/// copy was removed or is no longer that manager's, or Homebrew offers another version; when it runs nothing, or the
/// manager's update fails, the release is no longer marked handed over (`handed_back`).
pub fn background_update(home: &Path, exe: &Path, version: Option<&str>) -> i32 {
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
    let v = version.map(str::to_string).or_else(|| Check::read(home).and_then(|c| c.delegated));
    let Some(pinned) = v.as_deref().filter(|v| selfupdate::is_version(v)).and_then(|v| pinned(&manager, &argv, v)) else {
        eprintln!("superfluid: no release to update to was handed over; {} was not run", argv.join(" "));
        handed_back(home, false);
        return 1;
    };
    let v = v.unwrap_or_default();
    let watched = selfupdate::watched_dir(&exe, &manager);
    let began = Instant::now();
    let mut said = false;
    loop {
        // The file's setting: this process's environment turns updates off for itself.
        if !setting(home) {
            eprintln!("superfluid: automatic updates were turned off; {} was not run", pinned.join(" "));
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
        eprintln!("superfluid: {why}; {} was not run", pinned.join(" "));
        handed_back(home, false);
        return 0;
    }
    // Homebrew takes no version: it is run only when what it offers is the release chosen, and told not to update its
    // taps first, so it installs what was looked at. Another version is left for a later read.
    if manager == "Homebrew" {
        if let Err(why) = homebrew_offers(&argv, &v) {
            eprintln!("superfluid: {why}; {} was not run, and is tried again later", pinned.join(" "));
            handed_back(home, true);
            return 1;
        }
        // And again after it, which may take minutes: turned off (or uninstalled) meanwhile, nothing is run.
        if let Some(why) = not_to_run(home, &exe, &argv) {
            eprintln!("superfluid: {why}; {} was not run", pinned.join(" "));
            handed_back(home, false);
            return 0;
        }
    }
    eprintln!("superfluid: running {}", pinned.join(" "));
    let mut run = Command::new(&pinned[0]);
    run.args(&pinned[1..]).stdin(Stdio::null());
    if manager == "Homebrew" {
        run.env(BREW_NO_AUTO_UPDATE, "1");
    }
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
            eprintln!("superfluid: `{}` failed ({status})", pinned.join(" "));
            handed_back(home, true);
            1
        }
        Err(e) => {
            eprintln!("superfluid: running {}: {e}", pinned[0]);
            handed_back(home, true);
            1
        }
    }
}

/// Keeps a `brew` command from updating Homebrew's taps before it runs.
const BREW_NO_AUTO_UPDATE: &str = "HOMEBREW_NO_AUTO_UPDATE";

/// The manager's update `argv` (the command `self update` runs for the copy, from `selfupdate::delegate`) pinned to
/// release `v`: the manager's own command takes whatever its registry offers now, which may be a release published
/// since the daily read, one not yet out as long as the wait asks, or one that does not fit the engine here.
/// Homebrew's takes no version, and is run as it is once what it offers was looked at (`homebrew_offers`). `None` for
/// a manager not run in the background.
pub(crate) fn pinned(manager: &str, argv: &[String], v: &str) -> Option<Vec<String>> {
    let program = argv.first()?.clone();
    Some(match manager {
        "npm" => {
            let at = argv.iter().position(|a| a.starts_with("@basecompute/superfluid@"))?;
            let mut out = argv.to_vec();
            out[at] = format!("@basecompute/superfluid@{v}");
            out
        }
        "cargo" => argv.iter().cloned().chain(["--version".to_string(), v.to_string()]).collect(),
        "pipx" => vec![program, "install".into(), "--force".into(), format!("superfluid-cli=={v}")],
        "uv" => vec![program, "tool".into(), "install".into(), "--force".into(), format!("superfluid-cli=={v}")],
        "Homebrew" => argv.to_vec(),
        _ => return None,
    })
}

/// Whether the Homebrew whose `brew upgrade <formula>` is `argv` offers release `v` of the formula now, as `brew info`
/// reads its taps without updating them; when not, what it offers.
fn homebrew_offers(argv: &[String], v: &str) -> Result<(), String> {
    let (Some(brew), Some(formula)) = (argv.first(), argv.last()) else { return Err("no brew to ask".to_string()) };
    let mut cmd = Command::new(brew);
    cmd.args(["info", "--json=v2", formula]).env(BREW_NO_AUTO_UPDATE, "1");
    let out = match crate::exec::output_within(&mut cmd, Duration::from_secs(5 * 60)) {
        Ok(Some(out)) if out.status.success() => out,
        Ok(Some(out)) => return Err(format!("`brew info {formula}` failed ({}): {}", out.status, String::from_utf8_lossy(&out.stderr).trim())),
        Ok(None) => return Err(format!("`brew info {formula}` did not answer within 5 minutes")),
        Err(e) => return Err(format!("running {brew}: {e}")),
    };
    let info: Value = serde_json::from_slice(&out.stdout).unwrap_or_default();
    match info["formulae"][0]["versions"]["stable"].as_str() {
        Some(offered) if offered == v => Ok(()),
        Some(offered) => Err(format!("Homebrew offers {formula} {offered}, not {v}, the release the check chose")),
        None => Err(format!("`brew info {formula}` names no version")),
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
/// A download with no terminal to draw its bar on says how far it is in lines instead: `self update --auto`, which an
/// app runs and reads.
pub(crate) static PROGRESS_LINES: AtomicBool = AtomicBool::new(false);

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

/// The line a download with no terminal prints when it has come another tenth of the way (another 10 MB when its
/// size is not known) since `said`, which it moves on.
pub(crate) fn progress_line(name: &str, have: u64, total: Option<u64>, said: &mut u64) -> Option<String> {
    let (step, line) = match total.filter(|&t| t > 0) {
        Some(total) => {
            let percent = have.min(total) * 100 / total;
            (percent / 10, format!("downloading {name}: {percent}%"))
        }
        None => (have / 10_000_000, format!("downloading {name}: {:.1} MB", have as f64 / 1e6)),
    };
    (step > *said).then(|| {
        *said = step;
        line
    })
}

/// The progress of a download, drawn over itself on stderr, or said in lines (`PROGRESS_LINES`); nothing when off.
struct Bar {
    name: String,
    total: Option<u64>,
    on: bool,
    lines: bool,
    said: u64,
    width: usize,
    drawn: usize,
    last: Option<Instant>,
}

impl Bar {
    fn new(name: &str, total: Option<u64>, wanted: bool) -> Bar {
        let on = wanted && std::io::stderr().is_terminal();
        let lines = wanted && !on && PROGRESS_LINES.load(Ordering::Relaxed);
        Bar { name: name.to_string(), total, on, lines, said: 0, width: if on { terminal_width() } else { 80 }, drawn: 0, last: None }
    }

    fn show(&mut self, have: u64, force: bool) {
        if self.lines {
            if let Some(line) = progress_line(&self.name, have, self.total, &mut self.said) {
                eprintln!("{line}");
            }
            return;
        }
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
/// a terminal (lines, with `PROGRESS_LINES`, when it is not): the size from the listing, against the bytes in the
/// `.part` file curl writes. The file's path.
pub(crate) fn download(asset: &Asset, dir: &Path, bar: bool) -> Result<PathBuf, String> {
    let src = asset.url.as_str();
    let Some(want) = asset.sha256.as_deref() else { return Err(format!("{src}: no SHA-256 to check it against")) };
    let dest = dir.join(&asset.name);
    let part = dir.join(format!("{}.part", asset.name));
    let _ = std::fs::remove_file(&part);
    let mut shown = Bar::new(&asset.name, asset.size, bar);
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
        assert_eq!(state(&home, &none), "automatic updates: on · a release installs as soon as it is out");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(home.join(SETTINGS), r#"{"theme": "dark"}"#).unwrap();
        assert!(setting(&home), "no field: on");
        assert_eq!(auto_update_command(&home, &["off"]), 0);
        assert!(!setting(&home));
        let kept: Value = serde_json::from_str(&std::fs::read_to_string(home.join(SETTINGS)).unwrap()).unwrap();
        assert_eq!(kept, json!({"theme": "dark", "autoUpdate": false}), "Superfluid Desktop's own keys stay");
        assert_eq!(state(&home, &none), "automatic updates: off (superfluid self auto-update off) · a release installs as soon as it is out");
        assert_eq!(auto_update_command(&home, &[]), 0);
        assert_eq!(auto_update_command(&home, &["on"]), 0);
        assert!(setting(&home));
        for wrong in [&["maybe"][..], &["on", "off"], &["wait"], &["wait", "3", "4"]] {
            assert_eq!(auto_update_command(&home, wrong), 2, "{wrong:?}");
        }
        assert!(setting(&home), "a word that is neither changes nothing");
        assert!(!std::fs::read_dir(&home).unwrap().flatten().any(|e| e.file_name().to_string_lossy().contains(".tmp-")), "written whole, then renamed");
        // The environment wins over the file, for the run it is set in.
        let on = "automatic updates: on · a release installs as soon as it is out";
        assert_eq!(state(&home, &env_of(&[(NO_UPDATE_ENV, "1")])), "automatic updates: off (SUPERFLUID_NO_UPDATE is set) · a release installs as soon as it is out");
        assert_eq!(state(&home, &env_of(&[(NO_UPDATE_ENV, "0")])), on, "0 turns nothing off");
        assert_eq!(
            state(&home, &env_of(&[(selfupdate::NO_CHECK_ENV, "1")])),
            "automatic updates: off (SUPERFLUID_NO_UPDATE_CHECK is set) · a release installs as soon as it is out"
        );
        assert_eq!(state(&home, &env_of(&[("CI", "true")])), "automatic updates: off (CI is set) · a release installs as soon as it is out");
        assert_eq!(state(&home, &env_of(&[("CI", "false")])), on);
        std::fs::write(home.join(SETTINGS), "not json").unwrap();
        assert!(setting(&home), "a file that cannot be read is no switch");
        set_setting(&home, false).unwrap();
        assert!(!setting(&home));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn the_wait_is_a_setting_kept_beside_the_others_and_the_environment_sets_it_for_one_shell() {
        let home = std::env::temp_dir().join(format!("superfluid-auto-wait-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let none = env_of(&[]);
        assert_eq!(wait(&home, &none), Wait { days: 0, note: None }, "no file: no wait");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(home.join(SETTINGS), r#"{"theme": "dark", "autoUpdate": false}"#).unwrap();
        assert_eq!(auto_update_command(&home, &["wait", "3"]), 0);
        let kept: Value = serde_json::from_str(&std::fs::read_to_string(home.join(SETTINGS)).unwrap()).unwrap();
        assert_eq!(kept, json!({"theme": "dark", "autoUpdate": false, "updateWaitDays": 3}), "the other keys stay");
        assert!(!std::fs::read_dir(&home).unwrap().flatten().any(|e| e.file_name().to_string_lossy().contains(".tmp-")), "written whole, then renamed");
        assert_eq!(wait(&home, &none).days, 3);
        assert_eq!(state(&home, &none), "automatic updates: off (superfluid self auto-update off) · a release installs once it has been out 3 days");
        set_setting(&home, true).unwrap();
        assert_eq!(state(&home, &none), "automatic updates: on · a release installs once it has been out 3 days");
        assert_eq!(wait(&home, &none).days, 3, "turning updates on keeps the wait");
        // The variable, for one shell or service, over the setting; one that holds no wait leaves the setting.
        assert_eq!(state(&home, &env_of(&[(WAIT_ENV, "1")])), "automatic updates: on · a release installs once it has been out 1 day (SUPERFLUID_UPDATE_WAIT_DAYS is set)");
        assert_eq!(state(&home, &env_of(&[(WAIT_ENV, "0")])), "automatic updates: on · a release installs as soon as it is out (SUPERFLUID_UPDATE_WAIT_DAYS is set)");
        assert_eq!(
            state(&home, &env_of(&[(WAIT_ENV, "soon")])),
            "automatic updates: on · a release installs once it has been out 3 days (SUPERFLUID_UPDATE_WAIT_DAYS=soon is not a whole number of days from 0 to 365, so the setting is used)"
        );
        assert_eq!(wait(&home, &env_of(&[(WAIT_ENV, "")])), Wait { days: 3, note: None }, "empty is unset");
        for wrong in ["-1", "366", "2.5", "1e3"] {
            assert_eq!(wait(&home, &env_of(&[(WAIT_ENV, wrong)])).days, 3, "{wrong}");
        }
        assert_eq!(auto_update_command(&home, &["wait", "0"]), 0);
        assert_eq!(state(&home, &none), "automatic updates: on · a release installs as soon as it is out");
        // A wait that is no whole number of days from 0 to 365 changes nothing.
        for wrong in ["-1", "366", "three", ""] {
            assert_eq!(auto_update_command(&home, &["wait", wrong]), 2, "{wrong}");
        }
        assert_eq!(auto_update_command(&home, &["wait", "365"]), 0);
        assert_eq!(wait(&home, &none).days, 365);
        // A value written wrong in the file counts as missing, as any setting written wrong does, and the line says so.
        for wrong in [json!("3"), json!(-2), json!(400), json!(1.5), json!(true)] {
            std::fs::write(home.join(SETTINGS), json!({"updateWaitDays": wrong}).to_string()).unwrap();
            let w = wait(&home, &none);
            assert_eq!(w.days, 0, "{wrong}");
            assert!(w.note.as_deref().is_some_and(|n| n.starts_with(&format!("updateWaitDays {wrong} in ")) && n.ends_with("is not a whole number of days from 0 to 365, so there is none")), "{w:?}");
        }
        std::fs::write(home.join(SETTINGS), json!({"updateWaitDays": null}).to_string()).unwrap();
        assert_eq!(wait(&home, &none), Wait { days: 0, note: None }, "null is no setting");
        std::fs::write(home.join(SETTINGS), "not json").unwrap();
        assert_eq!(wait(&home, &none), Wait { days: 0, note: None }, "a file that cannot be read leaves every setting at its default");
        set_wait(&home, 7).unwrap();
        assert_eq!(wait(&home, &none).days, 7);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn what_setting_the_wait_says() {
        assert_eq!(wait_set(0), "automatic updates install a release as soon as it is out");
        assert_eq!(wait_set(1), "automatic updates wait 1 day after a release");
        assert_eq!(wait_set(3), "automatic updates wait 3 days after a release");
        assert_eq!(wait_phrase(0), "a release installs as soon as it is out");
        assert_eq!(wait_phrase(14), "a release installs once it has been out 14 days");
        assert_eq!((wait_of("0"), wait_of(" 3 "), wait_of("365"), wait_of("366"), wait_of("-1"), wait_of("x")), (Some(0), Some(3), Some(365), None, None, None));
    }

    #[test]
    fn a_record_that_keeps_the_releases_answers_every_wait_without_a_read() {
        let seen = |v: &str, published: &str| selfupdate::Seen { version: v.into(), published: published.into(), notes: format!("n{v}") };
        let kept = vec![seen("99.2.0", "2099-01-01"), seen("99.1.0", "2026-01-01"), seen("99.0.0", "2020-01-01")];
        let check = Check { checked_at: selfupdate::now_secs(), wait: Some(3), newest: Some("99.1.0".into()), seen: Some(kept.clone()), ..Check::default() };
        assert!(check.answers(0) && check.answers(7) && !check.due(), "made under 3 days, it answers for any wait: no read of its own");
        let now = check.clone().under(0);
        assert_eq!((now.wait, now.newest.as_deref(), now.published.as_deref(), now.notes.as_deref()), (Some(0), Some("99.2.0"), Some("2099-01-01"), Some("n99.2.0")), "no wait: the newest");
        assert_eq!(check.clone().under(3).newest.as_deref(), Some("99.1.0"));
        assert_eq!(check.clone().under(365).newest.as_deref(), Some("99.0.0"));
        let none = Check { seen: Some(vec![seen("99.2.0", "2099-01-01")]), ..check.clone() }.under(7);
        assert_eq!((none.newest, none.published, none.notes), (None, None, None), "none out a week: nothing named, not the one the record was made under");
        let other = claimed(Check { checked_at: 5, ..check.clone() }, 7);
        assert_eq!((other.wait, other.newest.as_deref(), &other.seen), (Some(3), Some("99.1.0"), &Some(kept)), "claimed under another wait: kept whole");
        assert!(!other.due());
    }

    #[test]
    fn a_record_from_before_the_releases_were_kept_names_nothing_for_another_wait_until_it_is_read_again() {
        let check = Check { checked_at: 5, wait: Some(3), newest: Some("0.2.2".into()), published: Some("2026-01-01".into()), notes: Some("n".into()), attempted: Some("0.2.1".into()), ..Check::default() };
        let same = claimed(check.clone(), 3);
        assert_eq!((same.wait, same.newest.as_deref()), (Some(3), Some("0.2.2")), "same wait: the release it names stays while it is read again");
        assert!(!same.due(), "claimed: the launches after it do not start the read again");
        let other = claimed(check.clone(), 0);
        assert_eq!((other.wait, &other.newest, &other.published, &other.notes), (Some(0), &None, &None, &None));
        assert_eq!(other.attempted.as_deref(), Some("0.2.1"), "what else it holds is kept");
        assert!(!other.due() && other.made_with(0) && other.answers(0));
        assert!(!check.answers(0) && check.answers(3));
        assert_eq!(check.clone().under(3), check.clone(), "its own wait: what it names");
        assert_eq!(check.clone().under(0).newest, None);
        assert!(!Check { wait: None, ..check }.answers(0), "a record from before the wait was a setting is read again");
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
    fn a_package_managers_update_in_the_background_is_pinned_to_the_release_the_check_chose() {
        let pin = |manager: &str, argv: &str| pinned(manager, &words(argv), "0.2.3").map(|a| a.join(" "));
        assert_eq!(
            pin("npm", "/n/bin/npm install --global --prefix /n @basecompute/superfluid@latest").as_deref(),
            Some("/n/bin/npm install --global --prefix /n @basecompute/superfluid@0.2.3")
        );
        assert_eq!(pin("cargo", "/c/bin/cargo install superfluid").as_deref(), Some("/c/bin/cargo install superfluid --version 0.2.3"));
        assert_eq!(pin("pipx", "/b/pipx upgrade superfluid-cli").as_deref(), Some("/b/pipx install --force superfluid-cli==0.2.3"));
        assert_eq!(pin("uv", "/b/uv tool upgrade superfluid-cli").as_deref(), Some("/b/uv tool install --force superfluid-cli==0.2.3"));
        // Homebrew takes no version: what it offers is looked at before it runs.
        assert_eq!(pin("Homebrew", "/h/bin/brew upgrade superfluid").as_deref(), Some("/h/bin/brew upgrade superfluid"));
        assert_eq!(pin("basert", "/b/basert superfluid install"), None, "not run in the background");
        assert_eq!(pin("npm", "/n/bin/npm install --global left-pad"), None, "not the command superfluid runs");
        assert_eq!(pinned("cargo", &[], "0.2.3"), None);
    }

    #[test]
    fn a_download_with_no_terminal_says_how_far_it_is_every_tenth() {
        let mut said = 0;
        let mut lines = Vec::new();
        for have in (0..=27_000_000).step_by(1_000_000) {
            lines.extend(progress_line("superfluid-x.tar.gz", have, Some(27_000_000), &mut said));
        }
        assert_eq!(lines.len(), 10, "{lines:?}");
        assert_eq!(lines[0], "downloading superfluid-x.tar.gz: 11%");
        assert_eq!(lines.last().unwrap(), "downloading superfluid-x.tar.gz: 100%");
        assert_eq!(progress_line("x", 27_000_000, Some(27_000_000), &mut said), None, "said once");
        let mut said = 0;
        assert_eq!(progress_line("x", 9_000_000, None, &mut said), None);
        assert_eq!(progress_line("x", 21_000_000, None, &mut said).as_deref(), Some("downloading x: 21.0 MB"), "no size: every 10 MB");
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
    use crate::selfupdate::{record, releases, target, Engine};
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
    fn the_newest_release_out_as_long_as_the_wait_asks_that_fits_the_engine_is_installed_and_the_way_back_said() {
        let dir = scratch("auto-update");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let p = listing(&dir, &[("99.0.0", "2026-01-01", Some(1)), ("99.1.0", "2026-01-02", Some(2)), ("99.2.0", "2099-01-01", Some(1))]);
        let found = releases(&serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap(), target().unwrap());
        // The record the daily read leaves under a wait of 3 days: 99.2.0 is not out that long, 99.1.0 drives another
        // engine.
        let check = record(&home, &found, &Engine::Abi(1), 3, &no_compat).unwrap();
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
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &check.notes_url().unwrap(), Some(p.to_string_lossy().to_string()), None, &mut Vec::new(), &mut |l| lines.push(l));
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
        let check = record(&home, &found, &Engine::None, 0, &no_compat).unwrap();
        let versions = Versions::of(&home);
        release_tree(&versions.version_dir("98.0.0"), "98.0.0");
        versions.set_current("98.0.0").unwrap();
        let exe = std::fs::canonicalize(versions.binary("98.0.0")).unwrap();
        let notes = check.notes_url().unwrap();

        // Offline: the listing cannot be read.
        let mut lines = Vec::new();
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &notes, Some("/nonexistent/releases.json".into()), None, &mut Vec::new(), &mut |l| lines.push(l));
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
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &notes, Some(p.to_string_lossy().to_string()), None, &mut Vec::new(), &mut |l| lines.push(l));
        assert!(bin.is_none() && began.elapsed() < Duration::from_secs(5), "{lines:?}");
        assert_eq!(lines[1], "couldn't update superfluid to 99.0.0: another update of superfluid is running; staying on 98.0.0");
        drop(held);

        // A download that does not check out.
        std::fs::write(dir.join(format!("superfluid-{}-99.0.0.tar.gz.sha256", target().unwrap())), format!("{}  x\n", "0".repeat(64))).unwrap();
        let mut lines = Vec::new();
        assert!(update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", &notes, Some(p.to_string_lossy().to_string()), None, &mut Vec::new(), &mut |l| lines.push(l)).is_none());
        assert!(lines[1].contains("SHA-256") && lines[1].ends_with("; staying on 98.0.0"), "{lines:?}");
        assert_eq!(versions.current().as_deref(), Some("98.0.0"));
        assert!(!versions.version_dir("99.0.0").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A copy Homebrew placed under `dir`, and its `brew`, which notes what it is asked to run in `dir/brew-ran`
    /// (with HOMEBREW_NO_AUTO_UPDATE), offers the version in `dir/brew-offers` (99.0.0 without it) to `brew info`,
    /// and fails while `dir/brew-fails` is there; `brew info` waits while `dir/brew-gate` is there. The copy's real
    /// path.
    fn homebrew_copy(dir: &Path) -> PathBuf {
        let root = dir.join("brew/Cellar/superfluid/98.0.0/libexec");
        release_tree(&root, "98.0.0");
        std::fs::write(root.join(selfupdate::INSTALLED_BY), "Homebrew: brew upgrade superfluid\n").unwrap();
        let brew = dir.join("brew/bin/brew");
        std::fs::create_dir_all(brew.parent().unwrap()).unwrap();
        let script = format!(
            "#!/bin/sh\necho \"$*|$HOMEBREW_NO_AUTO_UPDATE\" >> '{0}/brew-ran'\nif [ \"$1\" = info ]; then\n  while [ -e '{0}/brew-gate' ]; do sleep 0.05; done\n  v=$(cat '{0}/brew-offers' 2>/dev/null || echo 99.0.0)\n  printf '{{\"formulae\":[{{\"name\":\"superfluid\",\"versions\":{{\"stable\":\"%s\"}}}}]}}' \"$v\"\n  exit 0\nfi\n[ ! -e '{0}/brew-fails' ]\n",
            dir.display()
        );
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

        // The update in the background runs brew once nothing runs from the copy, when it offers the release chosen,
        // its taps left as they are so that it installs what it offered.
        assert_eq!(background_update(&home, &exe, Some("99.0.0")), 0);
        assert_eq!(std::fs::read_to_string(dir.join("brew-ran")).unwrap(), "info --json=v2 superfluid|1\nupgrade superfluid|1\n");
        std::fs::remove_file(dir.join("brew-ran")).unwrap();

        // Homebrew offers a release published since (or one not yet in its taps): nothing is upgraded, and it is
        // tried again after the next day's read.
        handed.write(&home).unwrap();
        std::fs::write(dir.join("brew-offers"), "99.1.0").unwrap();
        assert_eq!(background_update(&home, &exe, Some("99.0.0")), 1);
        assert_eq!(std::fs::read_to_string(dir.join("brew-ran")).unwrap(), "info --json=v2 superfluid|1\n", "no upgrade");
        let other = Check::read(&home).unwrap();
        assert_eq!((other.delegated.as_deref(), other.attempted.as_deref()), (None, Some("99.0.0")));
        std::fs::remove_file(dir.join("brew-offers")).unwrap();
        std::fs::remove_file(dir.join("brew-ran")).unwrap();

        // Without --version, the release the record says was handed over.
        handed.write(&home).unwrap();
        assert_eq!(background_update(&home, &exe, None), 0);
        assert!(std::fs::read_to_string(dir.join("brew-ran")).unwrap().ends_with("upgrade superfluid|1\n"));
        std::fs::remove_file(dir.join("brew-ran")).unwrap();

        // brew fails: tried, so handed over again after a day, and said again.
        handed.write(&home).unwrap();
        std::fs::write(dir.join("brew-fails"), "").unwrap();
        assert_eq!(background_update(&home, &exe, Some("99.0.0")), 1);
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
        assert_eq!(background_update(&home, &exe, Some("99.0.0")), 0);
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
    fn automatic_updates_turned_off_while_homebrew_is_asked_what_it_offers_run_nothing() {
        let dir = scratch("auto-update-brew-off");
        let home = dir.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let exe = homebrew_copy(&dir);
        Check { delegated: Some("99.0.0".into()), delegated_at: selfupdate::now_secs(), ..Check::default() }.write(&home).unwrap();
        let gate = dir.join("brew-gate");
        std::fs::write(&gate, "").unwrap();
        let (h, e) = (home.clone(), exe.clone());
        let update = std::thread::spawn(move || background_update(&h, &e, Some("99.0.0")));
        // `brew info` was started, and waits: updates are turned off meanwhile, then it offers the release chosen.
        let began = Instant::now();
        while !std::fs::read_to_string(dir.join("brew-ran")).unwrap_or_default().starts_with("info ") {
            assert!(began.elapsed() < Duration::from_secs(30), "brew info was not run");
            std::thread::sleep(Duration::from_millis(20));
        }
        set_setting(&home, false).unwrap();
        std::fs::remove_file(&gate).unwrap();
        assert_eq!(update.join().unwrap(), 0);
        assert_eq!(std::fs::read_to_string(dir.join("brew-ran")).unwrap(), "info --json=v2 superfluid|1\n", "no upgrade");
        let back = Check::read(&home).unwrap();
        assert_eq!((back.delegated.as_deref(), back.attempted.as_deref()), (None, None), "handed back, not tried");
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
        let bin = update_here(&home, &catalog, &exe, "98.0.0", "99.0.0", "https://notes", Some(p.to_string_lossy().to_string()), None, &mut Vec::new(), &mut |l| lines.push(l));
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
        let ran = update_runtimes(&bin, &catalog, "98.0.0", "99.0.0", Some(RuntimeId::new("llamacpp")), &mut |l| lines.push(l));
        assert_eq!(std::fs::read_to_string(home.join("runtime-calls")).unwrap(), "runtime update mlx\n", "{lines:?}");
        assert_eq!(ran, vec![json!({"id": "mlx", "outcome": "unchanged", "from": "0.1-metal", "to": "0.1-metal", "reason": null})]);
        // The new version's stdout is this command's stderr: a script `launch --print` writes, or a log, is not mixed with it.
        let mut cmd = by_new_version(Path::new("/bin/sh"), &catalog, "98.0.0", &["-c", "[ /dev/fd/1 -ef /dev/fd/2 ]"]);
        assert!(cmd.status().unwrap().success());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A home with superfluid 98.0.0 current under `versions`, and the copy's real path.
    fn managed_at(home: &Path) -> PathBuf {
        let versions = Versions::of(home);
        release_tree(&versions.version_dir("98.0.0"), "98.0.0");
        versions.set_current("98.0.0").unwrap();
        std::fs::canonicalize(versions.binary("98.0.0")).unwrap()
    }

    /// `self update --auto`: the exit code, what it printed on stdout, and its own lines.
    fn auto(home: &Path, catalog: &Catalog, exe: &Path, source: &Path, env: &[(&str, &str)]) -> (i32, Vec<String>, Vec<String>) {
        let (code, out, said, _) = auto_json(home, catalog, exe, source, env);
        (code, out, said)
    }

    /// `self update --auto`, and what `--json` prints.
    fn auto_json(home: &Path, catalog: &Catalog, exe: &Path, source: &Path, env: &[(&str, &str)]) -> (i32, Vec<String>, Vec<String>, Value) {
        let env: Vec<(String, String)> = env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let lookup = |k: &str| env.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
        let (mut out, mut said) = (Vec::new(), Vec::new());
        let (code, summary) = update_auto(home, catalog, exe, &source.to_string_lossy(), &lookup, &mut |l| out.push(l), &mut |l| said.push(l));
        (code, out, said, summary)
    }

    #[test]
    fn self_update_auto_installs_the_release_out_as_long_as_the_wait_asks_not_the_newest() {
        let dir = scratch("auto-command");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let exe = managed_at(&home);
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None), ("99.1.0", "2099-01-01", None)]);
        set_wait(&home, 3).unwrap();
        let (code, out, said) = auto(&home, &catalog, &exe, &p, &[]);
        assert_eq!(code, 0, "{said:?}");
        assert!(out.is_empty(), "{out:?}");
        let versions = Versions::of(&home);
        assert_eq!(versions.current().as_deref(), Some("99.0.0"), "99.1.0 is not out 3 days: {said:?}");
        assert_eq!(said[0], "Updating superfluid 98.0.0 → 99.0.0 · release notes: https://github.com/basecompute/superfluid/releases/tag/v99.0.0");
        assert_eq!(said.last().unwrap(), "superfluid 99.0.0 is in use · `superfluid self use 98.0.0` goes back · `superfluid self auto-update off` turns automatic updates off");
        let recorded = Check::read(&home).unwrap();
        assert_eq!((recorded.newest.as_deref(), recorded.wait), (Some("99.0.0"), Some(3)), "the listing read is recorded as the day's read, with its wait");
        // Run again by the new version: nothing out 3 days is newer.
        let exe = std::fs::canonicalize(versions.binary("99.0.0")).unwrap();
        let (code, out, _) = auto(&home, &catalog, &exe, &p, &[]);
        assert_eq!((code, out[0].as_str()), (0, "superfluid 99.1.0 (published 2099-01-01) is held off: automatic updates wait 3 days after a release; `superfluid self update` installs it now"));
        let (_, out, _) = auto(&home, &catalog, &exe, &p, &[(WAIT_ENV, "1")]);
        assert_eq!(out[0], "superfluid 99.1.0 (published 2099-01-01) is held off: automatic updates wait 1 day after a release; `superfluid self update` installs it now", "the variable sets the wait");
        let p_only = listing(&dir.join("only"), &[("99.0.0", "2026-01-01", None)]);
        let (code, out, _) = auto(&home, &catalog, &exe, &p_only, &[]);
        assert_eq!((code, out), (0, vec!["superfluid 99.0.0 is up to date for automatic updates".to_string()]));
        // With no wait (the default, or the variable for this run), the newest release is installed as soon as it is
        // out.
        let (code, out, said) = auto(&home, &catalog, &exe, &p, &[(WAIT_ENV, "0")]);
        assert!(code == 0 && out.is_empty(), "{out:?} {said:?}");
        assert_eq!(versions.current().as_deref(), Some("99.1.0"), "{said:?}");
        assert_eq!(Check::read(&home).unwrap().wait, Some(0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn self_check_names_the_newest_release_and_the_one_the_wait_lets_an_automatic_update_take() {
        let dir = scratch("check-lines");
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None), ("99.1.0", "2099-01-01", None)]);
        let found = releases(&serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap(), target().unwrap());
        let none = Check::default();
        let lines = |from: &str, wait: u32, check: &Check| described(&found, &Engine::None, &no_compat, check, from, wait);
        assert_eq!(lines("98.0.0", 0, &none), ["newest release: 99.1.0 (published 2099-01-01): `superfluid self update` installs it"], "no wait: one release");
        assert_eq!(
            lines("98.0.0", 3, &none),
            [
                "newest release: 99.1.0 (published 2099-01-01): `superfluid self update` installs it",
                "an automatic update takes 99.0.0 (published 2026-01-01): automatic updates wait 3 days after a release",
            ]
        );
        assert_eq!(
            lines("99.0.0", 3, &none),
            [
                "newest release: 99.1.0 (published 2099-01-01): `superfluid self update` installs it",
                "superfluid 99.1.0 (published 2099-01-01) is held off: automatic updates wait 3 days after a release; `superfluid self update` installs it now",
            ]
        );
        let declined = Check { declined: Some("99.1.0".into()), ..Check::default() };
        assert_eq!(lines("99.0.0", 0, &declined)[1], "you went back from 99.1.0 with self use; it stays until a newer release is out");
        assert_eq!(lines("99.1.0", 3, &none), ["no release newer than 99.1.0 fits here"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn self_update_auto_keeps_a_step_back_and_does_nothing_with_updates_off() {
        let dir = scratch("auto-command-held");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let exe = managed_at(&home);
        let versions = Versions::of(&home);
        release_tree(&versions.version_dir("99.0.0"), "99.0.0");
        versions.set_current("99.0.0").unwrap();
        selfupdate::use_version(&home, &catalog, &exe, "98.0.0").unwrap();
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None)]);
        let (code, out, said) = auto(&home, &catalog, &exe, &p, &[]);
        assert_eq!((code, out), (0, vec!["you went back from 99.0.0 with self use; it stays until a newer release is out".to_string()]), "{said:?}");
        assert_eq!(versions.current().as_deref(), Some("98.0.0"));
        // Off, by the setting or the environment: nothing read, nothing changed.
        set_setting(&home, false).unwrap();
        let (code, out, said) = auto(&home, &catalog, &exe, Path::new("/nonexistent/releases.json"), &[]);
        assert_eq!((code, out), (0, vec!["automatic updates are off (superfluid self auto-update off)".to_string()]), "{said:?}");
        set_setting(&home, true).unwrap();
        let (code, out, _) = auto(&home, &catalog, &exe, Path::new("/nonexistent/releases.json"), &[(NO_UPDATE_ENV, "1")]);
        assert_eq!((code, out), (0, vec!["automatic updates are off (SUPERFLUID_NO_UPDATE is set)".to_string()]));
        assert_eq!(versions.current().as_deref(), Some("98.0.0"));
        // Unreachable listing: one line, and a failure.
        let (code, out, said) = auto(&home, &catalog, &exe, Path::new("/nonexistent/releases.json"), &[]);
        assert!(code == 1 && out.is_empty() && said.len() == 1 && said[0].starts_with("superfluid: /nonexistent/releases.json: "), "{said:?}");
        // A copy a package manager owns is that manager's.
        let brew_copy = {
            let root = dir.join("brew/Cellar/superfluid/98.0.0/libexec");
            release_tree(&root, "98.0.0");
            std::fs::write(root.join(selfupdate::INSTALLED_BY), "Homebrew: brew upgrade superfluid\n").unwrap();
            std::fs::canonicalize(root.join("bin/superfluid")).unwrap()
        };
        let (code, out, said) = auto(&home, &catalog, &brew_copy, &p, &[]);
        assert!(code == 3 && out.is_empty(), "{said:?}");
        assert!(said[0].contains("was installed by Homebrew") && said[0].ends_with("`superfluid self update --auto` updates only a copy the install script or `superfluid self update` placed"), "{said:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn self_update_auto_installs_again_a_runtime_whose_worker_the_new_version_cannot_run() {
        let dir = scratch("auto-command-repair");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let exe = managed_at(&home);
        install_of(&catalog, "llamacpp", "b1-cpu");
        let broken = json!({"id": "llamacpp", "state": "broken", "worker": {"source": "installed"}, "ready": "the llamacpp runtime speaks Link W [1] and this superfluid speaks 2"});
        std::fs::write(home.join("runtimes.json"), json!([broken]).to_string()).unwrap();
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None)]);
        let (code, _, said, summary) = auto_json(&home, &catalog, &exe, &p, &[]);
        assert_eq!(code, 0, "{said:?}");
        assert_eq!(std::fs::read_to_string(home.join("runtime-calls")).unwrap(), "runtime update llamacpp\nruntime repair llamacpp\n", "{said:?}");
        assert_eq!(summary["runtimes"], json!([{"id": "llamacpp", "outcome": "repaired", "from": "b1-cpu", "to": "b1-cpu", "reason": null}]));
        assert!(said.iter().any(|l| l.starts_with("superfluid: llamacpp does not run with superfluid 99.0.0 (")), "{said:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn self_update_auto_json_says_what_it_did_and_a_runtime_that_failed_while_its_install_stays() {
        let dir = scratch("auto-command-json");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let exe = managed_at(&home);
        install_of(&catalog, "llamacpp", "b1-cpu");
        install_of(&catalog, "mlx", "0.1-metal");
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None)]);
        // Off: nothing read, the line on stdout is the reason.
        set_setting(&home, false).unwrap();
        let (code, out, _, summary) = auto_json(&home, &catalog, &exe, &p, &[]);
        assert_eq!(code, 0);
        assert_eq!(summary, json!({"from": "98.0.0", "installed": null, "result": "off", "reason": out[0], "notes": null, "runtimes": []}));
        assert_eq!(out[0], "automatic updates are off (superfluid self auto-update off)");
        set_setting(&home, true).unwrap();
        // Updated, with the update of llamacpp failing: its install stays, and the summary says so.
        std::fs::write(home.join("runtime-fails-llamacpp"), "").unwrap();
        let (code, _, said, summary) = auto_json(&home, &catalog, &exe, &p, &[]);
        assert_eq!(code, 0, "a runtime that failed is in the summary, not the exit code: {said:?}");
        let why = "llamacpp was not updated (exit status: 23); `superfluid runtime update llamacpp` tries again";
        assert_eq!(
            summary,
            json!({
                "from": "98.0.0",
                "installed": "99.0.0",
                "result": "updated",
                "reason": null,
                "notes": "https://github.com/basecompute/superfluid/releases/tag/v99.0.0",
                "runtimes": [
                    {"id": "llamacpp", "outcome": "failed", "from": "b1-cpu", "to": null, "reason": why},
                    {"id": "mlx", "outcome": "unchanged", "from": "0.1-metal", "to": "0.1-metal", "reason": null},
                ],
            }),
            "{said:?}"
        );
        assert!(said.contains(&format!("superfluid: {why}")), "{said:?}");
        assert_eq!(installs::default(&catalog, RuntimeId::new("llamacpp")).map(|i| i.name).as_deref(), Some("b1-cpu"));
        // Nothing newer: none, with the line's reason.
        let exe = std::fs::canonicalize(Versions::of(&home).binary("99.0.0")).unwrap();
        let (code, out, _, summary) = auto_json(&home, &catalog, &exe, &p, &[]);
        assert_eq!(code, 0);
        assert_eq!(summary, json!({"from": "99.0.0", "installed": null, "result": "none", "reason": "superfluid 99.0.0 is up to date for automatic updates", "notes": null, "runtimes": []}));
        assert_eq!(out, vec![summary["reason"].as_str().unwrap().to_string()]);
        // The listing unreachable: failed, with the error, and exit 1.
        let (code, _, said, summary) = auto_json(&home, &catalog, &exe, Path::new("/nonexistent/releases.json"), &[]);
        assert_eq!((code, &summary["result"], &summary["installed"]), (1, &json!("failed"), &Value::Null));
        assert_eq!(format!("superfluid: {}", summary["reason"].as_str().unwrap()), said[0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn self_check_json_says_the_newest_release_and_the_one_an_automatic_update_installs() {
        let dir = scratch("check-json");
        let home = dir.join("home");
        let catalog = Catalog::new(home.clone(), None, home.join("superfluid-workerd"));
        let exe = managed_at(&home);
        let p = listing(&dir, &[("99.0.0", "2026-01-01", None), ("99.1.0", "2099-01-01", None)]);
        let source = p.to_string_lossy().to_string();
        let none = |_: &str| None;
        let three = |k: &str| (k == WAIT_ENV).then(|| "3".to_string());
        assert_eq!(
            check_json(&home, &catalog, &exe, &source, &three).unwrap(),
            json!({
                "current": "98.0.0",
                "newest": "99.1.0",
                "auto": "99.0.0",
                "autoReason": null,
                "notes": "https://github.com/basecompute/superfluid/releases/tag/v99.1.0",
                "autoUpdate": true,
                "waitDays": 3,
            })
        );
        let now = check_json(&home, &catalog, &exe, &source, &none).unwrap();
        assert_eq!((&now["newest"], &now["auto"], &now["waitDays"]), (&json!("99.1.0"), &json!("99.1.0"), &json!(0)), "no wait: the newest");
        set_setting(&home, false).unwrap();
        let off = check_json(&home, &catalog, &exe, &source, &none).unwrap();
        assert_eq!((&off["auto"], &off["autoReason"], &off["autoUpdate"]), (&Value::Null, &json!("automatic updates are off (superfluid self auto-update off)"), &json!(false)));
        assert_eq!(off["newest"], "99.1.0", "what is out is said all the same");
        set_setting(&home, true).unwrap();
        let p = listing(&dir.join("old"), &[("97.0.0", "2026-01-01", None)]);
        let none_newer = check_json(&home, &catalog, &exe, &p.to_string_lossy(), &none).unwrap();
        assert_eq!((&none_newer["newest"], &none_newer["notes"], &none_newer["auto"]), (&Value::Null, &Value::Null, &Value::Null));
        assert_eq!(none_newer["autoReason"], "superfluid 98.0.0 is up to date for automatic updates");
        assert!(check_json(&home, &catalog, &exe, "/nonexistent/releases.json", &none).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
