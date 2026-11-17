use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, OnceLock};

use tracing::Metadata;
use tracing_subscriber::filter::{filter_fn, FilterExt};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::reload;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, Registry};

pub static DROPPED: AtomicU64 = AtomicU64::new(0);

type ReloadHandle = reload::Handle<EnvFilter, Registry>;
static RELOAD: OnceLock<ReloadHandle> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct LogConfig {
    pub dir: PathBuf,
    pub filter: String,
    pub max_file_bytes: u64,
    pub max_files: usize,
    pub queue: usize,
}

impl Default for LogConfig {
    fn default() -> Self {
        LogConfig {
            dir: PathBuf::from("superfluid-logs"),
            filter: "info".into(),
            max_file_bytes: 64 << 20,
            max_files: 8,
            queue: 8192,
        }
    }
}

struct Rotating {
    dir: PathBuf,
    max_file_bytes: u64,
    max_files: usize,
    seq: u64,
    file: Option<std::fs::File>,
    written: u64,
}

impl Rotating {
    fn open(dir: &Path, max_file_bytes: u64, max_files: usize) -> std::io::Result<Rotating> {
        std::fs::create_dir_all(dir)?;
        let mut seq = 0;
        for e in std::fs::read_dir(dir)?.flatten() {
            if let Some(n) = e
                .file_name()
                .to_str()
                .and_then(|n| n.strip_prefix("superfluid."))
                .and_then(|n| n.strip_suffix(".jsonl"))
                .and_then(|n| n.parse::<u64>().ok())
            {
                seq = seq.max(n + 1);
            }
        }
        let mut r = Rotating {
            dir: dir.to_path_buf(),
            max_file_bytes,
            max_files,
            seq,
            file: None,
            written: 0,
        };
        r.rotate()?;
        Ok(r)
    }

    fn rotate(&mut self) -> std::io::Result<()> {
        let path = self.dir.join(format!("superfluid.{}.jsonl", self.seq));
        let mut opts = std::fs::OpenOptions::new();
        opts.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        self.file = Some(opts.open(path)?);
        self.written = 0;
        self.seq += 1;
        let mut names: Vec<(u64, PathBuf)> = std::fs::read_dir(&self.dir)?
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                let n = p
                    .file_name()?
                    .to_str()?
                    .strip_prefix("superfluid.")?
                    .strip_suffix(".jsonl")?
                    .parse::<u64>()
                    .ok()?;
                Some((n, p))
            })
            .collect();
        names.sort();
        while names.len() > self.max_files {
            let (_, p) = names.remove(0);
            let _ = std::fs::remove_file(p);
        }
        Ok(())
    }

    fn write_line(&mut self, line: &[u8]) {
        if self.written + line.len() as u64 > self.max_file_bytes && self.rotate().is_err() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        match self.file.as_mut().map(|f| f.write_all(line)) {
            Some(Ok(())) => self.written += line.len() as u64,
            _ => {
                DROPPED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[derive(Clone)]
struct QueueWriter {
    tx: SyncSender<Vec<u8>>,
}

struct LineBuf {
    tx: SyncSender<Vec<u8>>,
    buf: Vec<u8>,
}

impl Write for LineBuf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for LineBuf {
    fn drop(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        match self.tx.try_send(std::mem::take(&mut self.buf)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                DROPPED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for QueueWriter {
    type Writer = LineBuf;
    fn make_writer(&'a self) -> Self::Writer {
        LineBuf {
            tx: self.tx.clone(),
            buf: Vec::with_capacity(256),
        }
    }
}

pub type LogRing = std::sync::Arc<std::sync::Mutex<LogLines>>;

const LOG_RING_CAP: usize = 2000;

pub fn new_log_ring() -> LogRing {
    std::sync::Arc::new(std::sync::Mutex::new(LogLines {
        lines: std::collections::VecDeque::with_capacity(256),
        evicted: 0,
    }))
}

/// The lines the TUI log pane shows, oldest first, capped at
/// `LOG_RING_CAP`. `evicted` counts lines dropped off the front, so
/// `evicted + i` numbers line `i` for good: the pane compares numbers across
/// redraws to hold its place while new lines stream in.
#[derive(Debug, Default)]
pub struct LogLines {
    lines: std::collections::VecDeque<String>,
    evicted: u64,
}

impl LogLines {
    /// The number the next line will get.
    pub fn end(&self) -> u64 {
        self.evicted + self.lines.len() as u64
    }

    /// Adds `text` one line per `\n`, cleaned for the terminal; blank lines
    /// are dropped.
    fn push(&mut self, text: &str) {
        for line in text.split('\n') {
            let line = clean_log_line(line);
            if line.is_empty() {
                continue;
            }
            while self.lines.len() >= LOG_RING_CAP {
                self.lines.pop_front();
                self.evicted += 1;
            }
            self.lines.push_back(line);
        }
    }
}

impl std::ops::Deref for LogLines {
    type Target = std::collections::VecDeque<String>;
    fn deref(&self) -> &Self::Target {
        &self.lines
    }
}

/// What a terminal would leave on screen for `line`, with nothing that moves
/// its cursor. The TUI writes cell text straight to the terminal, so a stray
/// `\r` or escape sequence would tear the display. A carriage return keeps
/// only what follows it (progress bars redraw that way), ANSI escape
/// sequences go, tabs become spaces, and other control characters are dropped.
pub(crate) fn clean_log_line(line: &str) -> String {
    let line = line.trim_end();
    let line = line.rsplit('\r').next().unwrap_or(line);
    if !line.contains(char::is_control) {
        return line.to_string();
    }
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        match c {
            '\t' => out.push_str("    "),
            '\x1b' => match chars.next() {
                // CSI: parameters and intermediates, then one final byte.
                Some('[') => {
                    for c in chars.by_ref() {
                        if ('\x40'..='\x7e').contains(&c) {
                            break;
                        }
                    }
                }
                // OSC: up to BEL or ST (ESC \).
                Some(']') => {
                    while let Some(c) = chars.next() {
                        if c == '\x07' {
                            break;
                        }
                        if c == '\x1b' {
                            chars.next();
                            break;
                        }
                    }
                }
                _ => {}
            },
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out.trim_end().to_string()
}

struct RingWriter {
    ring: LogRing,
    buf: Vec<u8>,
}
impl std::io::Write for RingWriter {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
pub fn push_ring_line(ring: &LogRing, text: &str) {
    if let Ok(mut r) = ring.lock() {
        r.push(text);
    }
}

impl Drop for RingWriter {
    fn drop(&mut self) {
        if !self.buf.is_empty() {
            push_ring_line(&self.ring, &String::from_utf8_lossy(&self.buf));
        }
    }
}
struct RingMaker {
    ring: LogRing,
}
impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for RingMaker {
    type Writer = RingWriter;
    fn make_writer(&'a self) -> RingWriter {
        RingWriter {
            ring: self.ring.clone(),
            buf: Vec::new(),
        }
    }
}

pub fn init(cfg: &LogConfig) -> Result<(), String> {
    init_sinks(LogSink::File(cfg.clone()), None, None)
}

pub fn init_stderr(filter: &str) -> Result<(), String> {
    init_sinks(LogSink::Stderr(filter.to_string()), None, None)
}

pub fn init_with_ring(cfg: &LogConfig, tui_ring: Option<LogRing>) -> Result<(), String> {
    init_sinks(LogSink::File(cfg.clone()), tui_ring, None)
}

#[derive(Debug, Clone)]
pub enum LogSink {
    File(LogConfig),
    Stderr(String),
}

type BoxedLayer = Box<dyn Layer<Registry> + Send + Sync>;

pub fn init_sinks(
    sink: LogSink,
    tui_ring: Option<LogRing>,
    otlp: Option<crate::otlp::OtlpConfig>,
) -> Result<(), String> {
    let mut errors: Vec<String> = Vec::new();
    let ring_layer = |ring: LogRing| {
        tracing_subscriber::fmt::layer()
            .compact()
            .with_ansi(false)
            .with_target(false)
            .with_writer(RingMaker { ring })
    };
    let stderr_layer = || -> BoxedLayer {
        // Under --log-format json, tracing's own JSON, its fields beside the message: the relay
        // passes each line on as a `log` event (crate::events).
        if crate::events::json() {
            return Box::new(
                tracing_subscriber::fmt::layer()
                    .json()
                    .flatten_event(true)
                    .with_current_span(false)
                    .with_span_list(false)
                    .with_writer(std::io::stderr),
            );
        }
        Box::new(
            tracing_subscriber::fmt::layer()
                .compact()
                .with_target(false)
                .with_writer(std::io::stderr),
        )
    };
    let (logs, filter): (BoxedLayer, String) = match sink {
        LogSink::File(cfg) => match file_writer(&cfg) {
            Ok(tx) => {
                let fmt = tracing_subscriber::fmt::layer()
                    .json()
                    .with_current_span(true)
                    .with_span_list(false)
                    .with_writer(QueueWriter { tx });
                let layer: BoxedLayer = match tui_ring {
                    Some(ring) => Box::new(fmt.and_then(ring_layer(ring))),
                    None => Box::new(fmt),
                };
                (layer, cfg.filter)
            }
            Err(e) => {
                errors.push(format!(
                    "{e} (logging to {})",
                    if tui_ring.is_some() {
                        "the TUI only"
                    } else {
                        "stderr"
                    }
                ));
                let layer: BoxedLayer = match tui_ring {
                    Some(ring) => Box::new(ring_layer(ring)),
                    None => stderr_layer(),
                };
                (layer, cfg.filter)
            }
        },
        LogSink::Stderr(filter) => (stderr_layer(), filter),
    };
    let filter = EnvFilter::try_new(&filter).unwrap_or_else(|e| {
        errors.push(format!("log filter {filter:?}: {e} (using \"info\")"));
        EnvFilter::new("info")
    });
    let mut otlp_err = None;
    let otlp = otlp.and_then(|cfg| {
        let set_up = EnvFilter::try_new(&cfg.filter)
            .map_err(|e| format!("--otlp-filter: {e}"))
            .and_then(|spans| crate::otlp::start(cfg).map(|started| (spans, started)));
        set_up.map_err(|e| otlp_err = Some(e)).ok()
    });
    let (filter, handle) = reload::Layer::new(filter);
    let reg = tracing_subscriber::registry().with(logs.with_filter(filter));
    match otlp {
        Some((spans, (layer, exporter))) => {
            let only_spans = filter_fn(|m: &Metadata<'_>| m.is_span()).and(spans);
            reg.with(layer.with_filter(only_spans))
                .try_init()
                .map_err(|e| e.to_string())?;
            crate::otlp::set_global(exporter);
        }
        None => reg.try_init().map_err(|e| e.to_string())?,
    }
    let _ = RELOAD.set(handle);
    if let Some(e) = otlp_err {
        errors.push(format!("OTLP export disabled (local logs unaffected): {e}"));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn file_writer(cfg: &LogConfig) -> Result<SyncSender<Vec<u8>>, String> {
    let mut rot = Rotating::open(&cfg.dir, cfg.max_file_bytes, cfg.max_files)
        .map_err(|e| format!("log dir {}: {e}", cfg.dir.display()))?;
    let (tx, rx) = sync_channel::<Vec<u8>>(cfg.queue.max(16));
    std::thread::Builder::new()
        .name("superfluid-log-writer".into())
        .spawn(move || {
            while let Ok(line) = rx.recv() {
                rot.write_line(&line);
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(tx)
}

pub fn set_filter(directive: &str) -> Result<(), String> {
    let handle = RELOAD.get().ok_or("telemetry not initialised")?;
    let filter = EnvFilter::try_new(directive).map_err(|e| e.to_string())?;
    handle.reload(filter).map_err(|e| e.to_string())
}

pub fn dropped_total() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

pub type SharedLogConfig = Arc<LogConfig>;

#[cfg(test)]
mod log_ring_tests {
    use super::*;

    #[test]
    fn lines_are_cleaned_of_anything_that_moves_the_cursor() {
        for (raw, want) in [
            ("plain line  ", "plain line"),
            ("loading 10%\rloading 55%\rloading 100%\r\n", "loading 100%"),
            ("\x1b[1;31mERROR\x1b[0m boom", "ERROR boom"),
            ("\x1b]0;window title\x07after", "after"),
            ("\x1b]8;;https://x.test\x1b\\link\x1b]8;;\x1b\\ text", "link text"),
            ("a\tb", "a    b"),
            ("bell\x07 and \x08backspace", "bell and backspace"),
            ("日本語 \u{fffd} ok", "日本語 \u{fffd} ok"),
        ] {
            assert_eq!(clean_log_line(raw), want, "{raw:?}");
        }
    }

    #[test]
    fn push_splits_lines_drops_blanks_and_numbers_through_eviction() {
        let mut log = LogLines::default();
        log.push("first\n  second\n\n\x1b[0m\nthird\n");
        assert_eq!(log.iter().collect::<Vec<_>>(), ["first", "  second", "third"]);
        assert_eq!(log.end(), 3);

        for i in 0..LOG_RING_CAP {
            log.push(&format!("line {i}"));
        }
        assert_eq!(log.len(), LOG_RING_CAP);
        assert_eq!(log.end(), 3 + LOG_RING_CAP as u64, "evicted lines keep counting");
        assert_eq!(log.front().map(String::as_str), Some("line 0"));
    }
}
