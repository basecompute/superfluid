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

pub type LogRing = std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>;

const LOG_RING_CAP: usize = 2000;

pub fn new_log_ring() -> LogRing {
    std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::with_capacity(256)))
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
pub fn push_ring_line(ring: &LogRing, line: String) {
    if let Ok(mut r) = ring.lock() {
        while r.len() >= LOG_RING_CAP {
            r.pop_front();
        }
        r.push_back(line);
    }
}

impl Drop for RingWriter {
    fn drop(&mut self) {
        let line = String::from_utf8_lossy(&self.buf);
        let line = line.trim_end();
        if line.is_empty() {
            return;
        }
        if let Ok(mut r) = self.ring.lock() {
            while r.len() >= LOG_RING_CAP {
                r.pop_front();
            }
            r.push_back(line.to_string());
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
    let stderr_layer = || {
        tracing_subscriber::fmt::layer()
            .compact()
            .with_target(false)
            .with_writer(std::io::stderr)
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
                    None => Box::new(stderr_layer()),
                };
                (layer, cfg.filter)
            }
        },
        LogSink::Stderr(filter) => (Box::new(stderr_layer()), filter),
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
