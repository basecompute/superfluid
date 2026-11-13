//! What `serve` and `node join` say on stderr to a program that runs them.
//!
//! With `--log-format json`, or `SUPERFLUID_LOG_FORMAT=json` in the environment, every line on
//! stderr is one JSON object, `{"timestamp": ..., "event": "<name>", ...fields}`. The steps of a
//! start (a runtime installed, a model pulled and loaded, the server ready, a node joined) are
//! events with names and fields of their own, and every other line is a `log` event that carries
//! it. The names and fields are kept across releases, under the schema number the first line
//! states; docs/serving/events.md lists them. In text mode nothing here writes anything, and
//! stderr is what it always was.
//!
//! The many lines this process and its workers write as prose are not each turned into an
//! event. In JSON mode fd 2 becomes a pipe, and a thread, the relay, reads it and writes one
//! JSON line a line to the real stderr: an event as it is, anything else as a `log` event. A
//! worker inherits fd 2, so what it writes is relayed as well, and an event is written into
//! the same pipe, so it comes out in the order it was written beside the prose. A process exit
//! waits for the relay to write out what was written before it, and a panic is a `log` event of
//! its own.

use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

/// The stream's schema, which the first line states. Removing or renaming an event or a field,
/// or changing what one means, raises it; adding one does not.
pub const SCHEMA: u32 = 1;

/// The variable that asks for a format when no `--log-format` does. A host sets it once for
/// every command it runs; the workers a command starts read it too.
pub const ENV: &str = "SUPERFLUID_LOG_FORMAT";

/// How long an exit waits for the relay to write out what came before it.
const EXIT_DRAIN: Duration = Duration::from_secs(2);

/// A line longer than this is relayed in pieces.
const MAX_LINE: usize = 16 * 1024;

/// What the drain writes into the pipe, followed by its number and a newline; the relay writes
/// nothing for it.
const DRAIN_MARK: &[u8] = b"\0superfluid-events-drain:";

/// What an event's line begins with in the pipe (ASCII RS, as a JSON text sequence's record
/// does): Superfluid's own events, this process's and its workers' (the pull gauge's). A line
/// without it is a `log` event, JSON that names an event or not, so a tool a runtime runs cannot
/// stand for one of Superfluid's.
pub const EVENT_MARK: u8 = 0x1e;

static JSON: AtomicBool = AtomicBool::new(false);
/// The real stderr, which the relay writes to; -1 in text mode.
static OUT: AtomicI32 = AtomicI32::new(-1);
static DRAIN_ASKED: AtomicU64 = AtomicU64::new(0);
static DRAIN_SEEN: AtomicU64 = AtomicU64::new(0);
static DRAINER: OnceLock<Mutex<Sender<u64>>> = OnceLock::new();
/// The pull in flight. Its gauge writes `pull.progress` without the id, and the relay adds it.
static PULLING: Mutex<Option<String>> = Mutex::new(None);

/// How stderr is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Prose, for a person.
    Text,
    /// One JSON event a line, for a program.
    Json,
}

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        match s {
            "text" => Some(Format::Text),
            "json" => Some(Format::Json),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::Text => "text",
            Format::Json => "json",
        }
    }
}

/// The format a command line asks for: its last `--log-format`, else [`ENV`], else text. The
/// error names a value that is neither `text` nor `json`.
pub fn requested(args: &[String]) -> Result<Format, String> {
    let mut flag = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--log-format" {
            if let Some(v) = it.next() {
                flag = Some(v.clone());
            }
        } else if let Some(v) = a.strip_prefix("--log-format=") {
            flag = Some(v.to_string());
        }
    }
    match flag {
        Some(v) => Format::parse(&v).ok_or_else(|| format!("--log-format expects text or json, got {v:?}")),
        None => match std::env::var(ENV) {
            Ok(v) if !v.is_empty() => Format::parse(&v).ok_or_else(|| format!("{ENV} expects text or json, got {v:?}")),
            _ => Ok(Format::Text),
        },
    }
}

/// Starts the format `args` ask for, for `command` (`serve`, `node join`), before anything is
/// written. Call it first, while this process has no other thread. In JSON mode stderr becomes
/// the relay's, and the first line is a `start` event with the schema; the workers started
/// after this read the format from [`ENV`].
pub fn init(command: &str, args: &[String]) -> Result<(), String> {
    let format = requested(args)?;
    // Set for JSON, and for text over an inherited JSON; left alone when nothing asked.
    let inherited = std::env::var(ENV).ok();
    if inherited.as_deref() != Some(format.name()) && (format == Format::Json || inherited.is_some()) {
        std::env::set_var(ENV, format.name());
    }
    if format == Format::Json && !json() {
        start_relay().map_err(|e| format!("--log-format json: stderr cannot be relayed: {e}"))?;
        JSON.store(true, Ordering::SeqCst);
        emit("start", json!({"schema": SCHEMA, "version": env!("CARGO_PKG_VERSION"), "command": command, "pid": std::process::id()}));
    }
    Ok(())
}

/// Whether stderr is JSON events.
pub fn json() -> bool {
    JSON.load(Ordering::SeqCst)
}

/// Writes the event `name` with `fields` (a JSON object) in JSON mode; nothing in text mode, where
/// the line a person reads is written beside it as it always was.
pub fn emit(name: &str, fields: Value) {
    if !json() {
        return;
    }
    // The relay stamps it as it writes it out, as it stamps every line: the timestamps rise down
    // the stream. One write under the lock: no other thread's line lands inside it.
    let mut marked = vec![EVENT_MARK];
    marked.extend_from_slice(line(name, fields, None).as_bytes());
    let _ = std::io::stderr().lock().write_all(&marked);
}

/// Says why this process stops and exits with `code`: `superfluid: <message>` in text mode, an
/// `error` event in JSON mode.
pub fn fail(code: i32, message: impl std::fmt::Display) -> ! {
    if json() {
        emit("error", json!({"message": message.to_string(), "exit_code": code}));
    } else {
        eprintln!("superfluid: {message}");
    }
    std::process::exit(code)
}

/// Where what stands for stderr is written: the real stderr behind the relay in JSON mode, fd 2
/// in text mode. `--log-file` points this at its file, so the events go there.
pub fn output_fd() -> RawFd {
    match OUT.load(Ordering::SeqCst) {
        -1 => libc::STDERR_FILENO,
        fd => fd,
    }
}

/// Waits, up to `within`, until the relay has written out everything this process wrote to
/// stderr before the call; at once in text mode.
pub fn drain(within: Duration) {
    let Some(drainer) = DRAINER.get() else { return };
    let seq = DRAIN_ASKED.fetch_add(1, Ordering::SeqCst) + 1;
    // The mark goes into the pipe from a thread of its own: a writer that holds stderr while the
    // pipe is full holds that thread, not this one, which gives up at `within`.
    if !drainer.lock().is_ok_and(|d| d.send(seq).is_ok()) {
        return;
    }
    let deadline = Instant::now() + within;
    while DRAIN_SEEN.load(Ordering::SeqCst) < seq && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A pull of `id` begins: `pull.start`, and its progress lines get its id.
pub(crate) fn pull_started(id: &str, runtime: &str, offline: bool) {
    if !json() {
        return;
    }
    *PULLING.lock().unwrap_or_else(|e| e.into_inner()) = Some(id.to_string());
    emit("pull.start", json!({"id": id, "runtime": runtime, "offline": offline}));
}

/// The pull of `id` landed at `path`: `pull.done`, with the model's size on disk.
pub(crate) fn pull_done(id: &str, path: &Path) {
    if !json() {
        return;
    }
    PULLING.lock().unwrap_or_else(|e| e.into_inner()).take();
    emit("pull.done", json!({"id": id, "path": path.to_string_lossy(), "bytes": bytes_on_disk(path)}));
}

/// The pull of `id` failed: `pull.failed`, when it had begun and did not land.
pub(crate) fn pull_failed(id: &str, error: &str) {
    if !json() {
        return;
    }
    let mut pulling = PULLING.lock().unwrap_or_else(|e| e.into_inner());
    if pulling.as_deref() == Some(id) {
        pulling.take();
        drop(pulling);
        emit("pull.failed", json!({"id": id, "error": error}));
    }
}

/// The line of event `name`: the timestamp when given, the name, then `fields` as given, apart
/// from any field that would stand for either of the two.
fn line(name: &str, fields: Value, timestamp: Option<Value>) -> String {
    let mut out = Map::new();
    if let Some(t) = timestamp {
        out.insert("timestamp".into(), t);
    }
    out.insert("event".into(), Value::String(name.to_string()));
    if let Value::Object(fields) = fields {
        for (k, v) in fields {
            if k != "timestamp" && k != "event" {
                out.insert(k, v);
            }
        }
    }
    let mut line = Value::Object(out).to_string();
    line.push('\n');
    line
}

/// Now as tracing's JSON lines write it: RFC 3339, UTC, to the microsecond.
fn now() -> Value {
    use tracing_subscriber::fmt::time::FormatTime;
    let mut s = String::with_capacity(32);
    let _ = tracing_subscriber::fmt::time::SystemTime.format_time(&mut tracing_subscriber::fmt::format::Writer::new(&mut s));
    Value::String(s)
}

/// The JSON line the relay writes for one line read from the pipe, or none for a blank one,
/// stamped with `timestamp` (the relay's now: the stream's timestamps rise line by line, and a
/// timestamp a line came with is not kept). An event (a line marked with [`EVENT_MARK`]) passes
/// as it is, a pull's progress with the id of `pulling` when it came without one; a line of
/// tracing's JSON is a `log` event with its level and fields; anything else is a `log` event that
/// carries the line.
fn relayed(raw: &[u8], pulling: Option<&str>, timestamp: impl FnOnce() -> Value) -> Option<String> {
    let (marked, raw) = match raw.split_first() {
        Some((&EVENT_MARK, rest)) => (true, rest),
        _ => (false, raw),
    };
    let text = crate::telemetry::clean_log_line(&String::from_utf8_lossy(raw));
    if text.is_empty() {
        return None;
    }
    if text.starts_with('{') {
        if let Ok(Value::Object(mut fields)) = serde_json::from_str::<Value>(&text) {
            let event = match fields.get("event") {
                Some(Value::String(e)) if marked => Some(e.clone()),
                None if fields.contains_key("level") && fields.contains_key("timestamp") => Some("log".to_string()),
                _ => None,
            };
            if let Some(event) = event {
                if event == "pull.progress" && !fields.contains_key("id") {
                    if let Some(id) = pulling {
                        fields.insert("id".into(), Value::String(id.to_string()));
                    }
                }
                return Some(line(&event, Value::Object(fields), Some(timestamp())));
            }
        }
    }
    let level = if text.contains("WARNING") { "WARN" } else { "INFO" };
    Some(line("log", json!({"level": level, "message": text}), Some(timestamp())))
}

/// Bytes read from the pipe, cut into lines.
#[derive(Default)]
struct Lines {
    carry: Vec<u8>,
}

impl Lines {
    /// The whole lines `bytes` completes, each with its newline. A line that runs past
    /// [`MAX_LINE`] without one keeps only what follows its last carriage return (a bar redrawn
    /// in place), and what is still too long goes in pieces.
    fn push(&mut self, bytes: &[u8]) -> Vec<Vec<u8>> {
        self.carry.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(at) = self.carry.iter().position(|&b| b == b'\n') {
            out.push(self.carry.drain(..=at).collect());
        }
        if self.carry.len() > MAX_LINE {
            if let Some(r) = self.carry.iter().rposition(|&b| b == b'\r') {
                self.carry.drain(..=r);
            }
            while self.carry.len() > MAX_LINE {
                out.push(self.carry.drain(..MAX_LINE).collect());
            }
        }
        out
    }

    /// What is left without a newline at the end.
    fn finish(&mut self) -> Option<Vec<u8>> {
        (!self.carry.is_empty()).then(|| std::mem::take(&mut self.carry))
    }
}

/// The drain number a line carries, and the line before the mark (a writer that left a line
/// without its newline); none for a line without a mark.
fn drain_mark(raw: &[u8]) -> Option<(&[u8], Option<u64>)> {
    let at = raw.windows(DRAIN_MARK.len()).position(|w| w == DRAIN_MARK)?;
    let seq = std::str::from_utf8(&raw[at + DRAIN_MARK.len()..]).ok().and_then(|t| t.trim_end().parse().ok());
    Some((&raw[..at], seq))
}

/// Reads the pipe until every writer has closed it, writing what each line stands for to `out`.
fn relay(mut from: std::fs::File, mut out: std::fs::File) {
    let mut write = |raw: &[u8]| {
        let (raw, drained) = drain_mark(raw).unwrap_or((raw, None));
        let pulling = PULLING.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some(line) = relayed(raw, pulling.as_deref(), now) {
            // A host that stopped reading is no reason to stop: the line is dropped.
            let _ = out.write_all(line.as_bytes());
        }
        if let Some(seq) = drained {
            DRAIN_SEEN.fetch_max(seq, Ordering::SeqCst);
        }
    };
    let mut lines = Lines::default();
    let mut buf = vec![0u8; 8192];
    loop {
        let n = match from.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        for line in lines.push(&buf[..n]) {
            write(&line);
        }
    }
    if let Some(rest) = lines.finish() {
        write(&rest);
    }
}

/// Points fd 2 at a pipe the relay reads, keeping the real stderr for the relay to write to.
fn start_relay() -> std::io::Result<()> {
    // SAFETY: fd syscalls on descriptors this process owns; each one is wrapped in an OwnedFd as
    // soon as it exists, so a failure closes what was opened.
    let (out, read, write) = unsafe {
        let out = libc::fcntl(libc::STDERR_FILENO, libc::F_DUPFD_CLOEXEC, 3);
        if out < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let out = OwnedFd::from_raw_fd(out);
        let mut fds = [0i32; 2];
        if libc::pipe(fds.as_mut_ptr()) < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let (read, write) = (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1]));
        libc::fcntl(read.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
        (out, read, write)
    };
    let out = std::fs::File::from(out);
    let out_fd = out.as_raw_fd();
    std::thread::Builder::new().name("superfluid-events".into()).spawn(move || relay(std::fs::File::from(read), out))?;
    let (tx, rx) = std::sync::mpsc::channel::<u64>();
    std::thread::Builder::new().name("superfluid-events-drain".into()).spawn(move || {
        for seq in rx {
            let mut mark = DRAIN_MARK.to_vec();
            mark.extend_from_slice(format!("{seq}\n").as_bytes());
            let _ = std::io::stderr().lock().write_all(&mark);
        }
    })?;
    // SAFETY: dup2 of the pipe's write end, which this process owns, onto fd 2; dup2 clears
    // close-on-exec, so every worker started from here inherits the pipe as its stderr.
    if unsafe { libc::dup2(write.as_raw_fd(), libc::STDERR_FILENO) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    drop(write);
    OUT.store(out_fd, Ordering::SeqCst);
    let _ = DRAINER.set(Mutex::new(tx));
    extern "C" fn at_exit() {
        drain(EXIT_DRAIN);
    }
    // SAFETY: registers a function that takes nothing and returns nothing, as atexit wants.
    unsafe {
        libc::atexit(at_exit);
    }
    std::panic::set_hook(Box::new(on_panic));
    Ok(())
}

/// A panic in JSON mode: one `log` event at level ERROR, with the backtrace when one is taken
/// (RUST_BACKTRACE), written out before the process can end.
fn on_panic(info: &std::panic::PanicHookInfo<'_>) {
    let thread = std::thread::current();
    let payload = info
        .payload()
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
        .unwrap_or("Box<dyn Any>");
    let at = info.location().map(|l| format!(" at {l}")).unwrap_or_default();
    let mut fields = json!({
        "level": "ERROR",
        "message": format!("thread '{}' panicked{at}: {payload}", thread.name().unwrap_or("<unnamed>")),
    });
    let trace = std::backtrace::Backtrace::capture();
    if trace.status() == std::backtrace::BacktraceStatus::Captured {
        fields["backtrace"] = Value::String(trace.to_string());
    }
    emit("log", fields);
    drain(EXIT_DRAIN);
}

/// The bytes `path` holds: a file's length, or the files under a directory (an MLX model),
/// followed through the links a Hugging Face snapshot is made of.
fn bytes_on_disk(path: &Path) -> u64 {
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.is_dir() => std::fs::read_dir(path)
            .map(|rd| rd.flatten().map(|e| bytes_on_disk(&e.path())).sum())
            .unwrap_or(0),
        Ok(m) if m.is_symlink() => std::fs::metadata(path).ok().filter(|t| t.is_file()).map_or(0, |t| t.len()),
        Ok(m) => m.len(),
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at() -> Value {
        Value::String("2026-11-13T10:00:00.000000Z".into())
    }

    fn parsed(line: &str) -> Value {
        assert!(line.ends_with('\n') && !line[..line.len() - 1].contains('\n'), "one line: {line:?}");
        serde_json::from_str(line).unwrap()
    }

    #[test]
    fn an_event_line_leads_with_its_timestamp_and_name_and_keeps_its_fields() {
        let l = line("pull.done", json!({"id": "org/m:q4", "path": "/m.gguf", "bytes": 17, "event": "spoofed", "timestamp": 0}), Some(at()));
        assert_eq!(l, "{\"timestamp\":\"2026-11-13T10:00:00.000000Z\",\"event\":\"pull.done\",\"id\":\"org/m:q4\",\"path\":\"/m.gguf\",\"bytes\":17}\n");
        assert_eq!(line("start", Value::Null, None), "{\"event\":\"start\"}\n", "emitted unstamped, for the relay to stamp");
        assert_eq!(relayed(b"\x1e{\"event\":7}\n", None, at).map(|l| parsed(&l)["event"].clone()), Some(json!("log")), "a marked line that names no event is a line");
    }

    #[test]
    fn the_timestamp_is_rfc3339_in_utc() {
        let Value::String(t) = now() else { panic!("a string") };
        assert_eq!(t.len(), "2026-11-13T10:00:00.000000Z".len(), "{t}");
        assert!(t.ends_with('Z') && t.as_bytes()[10] == b'T', "{t}");
    }

    #[test]
    fn a_line_of_prose_is_a_log_event_that_carries_it() {
        let l = relayed(b"superfluid: pulling org/m with the llamacpp runtime\n", None, at).unwrap();
        assert_eq!(parsed(&l), json!({"timestamp": at(), "event": "log", "level": "INFO", "message": "superfluid: pulling org/m with the llamacpp runtime"}));
        let l = relayed(b"superfluid: WARNING --api-key not set\n", None, at).unwrap();
        assert_eq!(parsed(&l)["level"], "WARN");
        // a bar redrawn in place leaves its last drawing; escapes and blank lines go
        let l = relayed(b"\x1b[1m 10%\r 55%\r100% done\x1b[0m\r\n", None, at).unwrap();
        assert_eq!(parsed(&l)["message"], "100% done");
        assert_eq!(relayed(b"   \n", None, at), None);
        assert_eq!(relayed(b"\x1b[0m\n", None, at), None);
        let l = relayed(b"bytes \xff that are not utf-8\n", None, at).unwrap();
        assert_eq!(parsed(&l)["message"], "bytes \u{fffd} that are not utf-8");
    }

    #[test]
    fn an_event_passes_through_stamped_as_it_is_relayed() {
        let l = relayed(b"\x1e{\"event\":\"serve.ready\",\"http\":\"127.0.0.1:8453\"}\n", None, at).unwrap();
        assert_eq!(l, "{\"timestamp\":\"2026-11-13T10:00:00.000000Z\",\"event\":\"serve.ready\",\"http\":\"127.0.0.1:8453\"}\n");
        let l = relayed(b"\x1e{\"timestamp\":\"2026-01-01T00:00:00.000000Z\",\"event\":\"serve.ready\"}\n", None, at).unwrap();
        assert_eq!(parsed(&l)["timestamp"], at(), "the relay's, so the stream's timestamps rise");
        // the gauge in a worker writes progress with no timestamp and no id
        let l = relayed(b"\x1e{\"event\":\"pull.progress\",\"done_bytes\":5,\"total_bytes\":null}\n", Some("org/m:q4"), at).unwrap();
        assert_eq!(parsed(&l), json!({"timestamp": at(), "event": "pull.progress", "done_bytes": 5, "total_bytes": null, "id": "org/m:q4"}));
        let l = relayed(b"\x1e{\"event\":\"pull.progress\",\"id\":\"own\",\"done_bytes\":5}\n", Some("org/m:q4"), at).unwrap();
        assert_eq!(parsed(&l)["id"], "own");
    }

    #[test]
    fn tracings_json_is_a_log_event_and_other_json_is_prose() {
        let l = relayed(b"{\"timestamp\":\"2026-01-01T00:00:00.000000Z\",\"level\":\"WARN\",\"message\":\"the head is silent\",\"lane\":0,\"target\":\"superfluid_daemon::nodeagent\"}\n", None, at).unwrap();
        assert_eq!(
            l,
            "{\"timestamp\":\"2026-11-13T10:00:00.000000Z\",\"event\":\"log\",\"level\":\"WARN\",\"message\":\"the head is silent\",\"lane\":0,\"target\":\"superfluid_daemon::nodeagent\"}\n"
        );
        // JSON a tool writes is a line like any other, even one that names an event of ours
        for other in [&b"{\"error\":\"from a tool\"}\n"[..], b"{\"event\":\"error\",\"exit_code\":0}\n", b"{\"event\":7}\n", b"{not json\n", b"[1,2]\n"] {
            let l = relayed(other, None, at).unwrap();
            let v = parsed(&l);
            assert_eq!(v["event"], "log", "{l}");
            assert_eq!(v["message"], String::from_utf8_lossy(other).trim_end(), "{l}");
        }
    }

    #[test]
    fn lines_are_cut_at_newlines_and_a_runaway_line_in_pieces() {
        let mut lines = Lines::default();
        assert!(lines.push(b"no newline yet").is_empty());
        assert_eq!(lines.push(b", now\nsecond\nthi"), vec![b"no newline yet, now\n".to_vec(), b"second\n".to_vec()]);
        assert_eq!(lines.finish(), Some(b"thi".to_vec()));
        assert_eq!(lines.finish(), None);
        // a bar redrawn with carriage returns and never ended: only its last drawing is kept
        let mut bar = Vec::new();
        while bar.len() <= MAX_LINE {
            bar.extend_from_slice(b"\r 42% [=====>      ]");
        }
        assert!(lines.push(&bar).is_empty());
        assert_eq!(lines.push(b"\n"), vec![b" 42% [=====>      ]\n".to_vec()]);
        let long = vec![b'x'; 2 * MAX_LINE + 5];
        let pieces = lines.push(&long);
        assert_eq!(pieces.iter().map(Vec::len).collect::<Vec<_>>(), [MAX_LINE, MAX_LINE]);
        assert_eq!(lines.finish().map(|r| r.len()), Some(5));
    }

    #[test]
    fn a_drain_mark_is_found_after_a_line_left_without_its_newline() {
        assert_eq!(drain_mark(b"plain\n"), None);
        assert_eq!(drain_mark(b"\0superfluid-events-drain:7\n"), Some((&b""[..], Some(7))));
        assert_eq!(drain_mark(b"half a line\0superfluid-events-drain:12\n"), Some((&b"half a line"[..], Some(12))));
    }

    #[test]
    fn the_format_is_the_last_flag_else_the_environment() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(requested(&args(&["m.gguf", "--log-format", "json"])), Ok(Format::Json));
        assert_eq!(requested(&args(&["--log-format=json", "--log-format", "text"])), Ok(Format::Text));
        assert_eq!(requested(&args(&["--log-format", "yaml"])), Err("--log-format expects text or json, got \"yaml\"".into()));
        // a dangling flag is the parser's to report
        if std::env::var_os(ENV).is_none() {
            assert_eq!(requested(&args(&["--log-format"])), Ok(Format::Text));
        }
    }

    #[test]
    fn a_models_size_on_disk_follows_a_snapshots_links() {
        let dir = std::env::temp_dir().join(format!("superfluid-events-bytes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("blobs")).unwrap();
        std::fs::create_dir_all(dir.join("snapshot/sub")).unwrap();
        std::fs::write(dir.join("blobs/a"), vec![0u8; 300]).unwrap();
        std::fs::write(dir.join("snapshot/sub/b"), vec![0u8; 20]).unwrap();
        std::os::unix::fs::symlink(dir.join("blobs/a"), dir.join("snapshot/model.safetensors")).unwrap();
        std::os::unix::fs::symlink(dir.join("blobs"), dir.join("snapshot/linked-dir")).unwrap();
        assert_eq!(bytes_on_disk(&dir.join("snapshot")), 320, "a link to a file counts, one to a directory does not");
        assert_eq!(bytes_on_disk(&dir.join("blobs/a")), 300);
        assert_eq!(bytes_on_disk(&dir.join("nowhere")), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
