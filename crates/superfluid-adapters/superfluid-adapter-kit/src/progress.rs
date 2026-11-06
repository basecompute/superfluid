//! A bar over a pull: the bytes the runtime's fetch tool lands on disk, against what it set
//! out to fetch, with the rate and the time left.
//!
//! The tools draw their own bar only on a terminal (llama.cpp's goes to stdout, which a pull
//! reads for the path; basert draws none), so a pull through superfluid showed nothing until
//! the model was there. This one reads what lands in the tool's cache instead, so it is the
//! same for every runtime and needs nothing of the tool: an adapter names the directories the
//! tool writes into and, when it can tell, how many bytes it will write. The files written to
//! since the watch began are the pull's, what was there already is not, and a file is known by
//! its inode, so it counts once however the tool renames or moves it, and the bytes a tool
//! preallocates (basert sizes its `.partial` before fetching into it) count as they are filled.

use std::collections::{HashMap, VecDeque};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Where a pull lands its bytes, and how many it means to.
#[derive(Debug, Clone, Default)]
pub struct Watch {
    /// Directories whose files, written to since the watch began, are the pull's bytes.
    pub roots: Vec<PathBuf>,
    /// Directories a tool moves a finished file into, where it is followed: a file there
    /// counts when it was seen under `roots` first (it is the same inode), and the last bytes
    /// the tool wrote before moving it are read there. Nothing else there is counted, so a
    /// store shared between pulls (huggingface_hub's, basert's install tree) is safe to name.
    pub follow: Vec<PathBuf>,
    /// The bytes the pull means to fetch, when known: the bar's 100%.
    pub total: Option<u64>,
}

/// How often the disk is read and, on a terminal, the bar redrawn.
const TICK: Duration = Duration::from_millis(250);
/// How often a line is printed where there is no terminal to redraw on.
const LINE_EVERY: Duration = Duration::from_secs(10);
/// The rate is over this much recent history, so it follows the connection, not the average.
const RATE_WINDOW: Duration = Duration::from_secs(5);
/// Nothing is drawn before this much has landed: a tool writes a few bytes of bookkeeping
/// (llama.cpp its `refs`) before any of the model, and a model this small needs no bar.
const DRAW_FROM: u64 = 64 * 1024;

/// Runs `run`, drawing the pull `watch` describes while it does; with no watch, just runs it.
pub fn watched<T>(watch: Option<Watch>, run: impl FnOnce() -> T) -> T {
    let Some(watch) = watch.filter(|w| !w.roots.is_empty()) else {
        return run();
    };
    let stop = Arc::new(AtomicBool::new(false));
    let drawer = {
        let stop = stop.clone();
        std::thread::spawn(move || draw_until(watch, &stop))
    };
    let out = run();
    stop.store(true, Ordering::SeqCst);
    let _ = drawer.join();
    out
}

/// Writes `output` from the tool to stderr, clearing a bar drawn there first so the two do not
/// run into each other; the bar is drawn again on its next tick.
pub fn write_through(output: &[u8]) {
    let mut line = LINE.lock().unwrap_or_else(|e| e.into_inner());
    let mut err = std::io::stderr().lock();
    if line.drawn {
        let _ = err.write_all(b"\r\x1b[K");
        line.drawn = false;
    }
    let _ = err.write_all(output);
    let _ = err.flush();
}

/// Whether a bar is drawn on the terminal's current line.
struct Line {
    drawn: bool,
}

static LINE: Mutex<Line> = Mutex::new(Line { drawn: false });

fn draw_until(watch: Watch, stop: &AtomicBool) {
    let tty = std::io::stderr().is_terminal();
    let began = Instant::now();
    let mut seen = Seen::begin(&watch.roots);
    let mut gauge = Gauge::new(watch.total, began);
    let mut last_line = began;
    let mut shown = false;
    loop {
        let done = stop.load(Ordering::SeqCst);
        let now = Instant::now();
        gauge.sample(now, seen.look(&watch.roots, &watch.follow));
        if gauge.fetched >= DRAW_FROM {
            if tty {
                let mut line = LINE.lock().unwrap_or_else(|e| e.into_inner());
                let mut err = std::io::stderr().lock();
                let _ = err.write_all(format!("\r\x1b[K{}", gauge.bar(now)).as_bytes());
                let _ = err.flush();
                line.drawn = true;
                shown = true;
            } else if !shown || now.duration_since(last_line) >= LINE_EVERY {
                write_through(format!("pull: {}\n", gauge.report(now)).as_bytes());
                last_line = now;
                shown = true;
            }
        }
        if done {
            break;
        }
        std::thread::sleep(TICK);
    }
    if shown {
        write_through(format!("pull: {}\n", gauge.summary()).as_bytes());
    }
}

/// What a watch has seen land: the files under its roots that the pull wrote to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Landed {
    /// The bytes of the files the pull wrote to, what they held before it began included: a
    /// resumed partial is that far along. The bar's place against the total.
    pub present: u64,
    /// The bytes written since the watch began: the rate's and the summary's.
    pub fetched: u64,
}

/// The files under a watch's roots, each known by its inode rather than its path: a file
/// counts once however its tool renames or moves it (huggingface_hub fetches into
/// `<blob>.<id>.incomplete`, renames it to `<blob>`, then moves it into the store it shares
/// between repositories; llama.cpp renames its `.downloadInProgress` away), and what leaves
/// the roots stays counted. What each file held when the watch began is not the pull's: an
/// unrelated download that finished a moment before, or the part a resumed pull found.
pub struct Seen {
    before: HashMap<FileId, u64>,
    /// The most bytes seen in each file since.
    most: HashMap<FileId, u64>,
}

type FileId = (u64, u64);

impl Seen {
    /// Notes what is under `roots` now, as not the pull's.
    pub fn begin(roots: &[PathBuf]) -> Seen {
        Seen { before: files_under(roots), most: HashMap::new() }
    }

    /// Reads what is under `roots` now, and under `follow` what came from there, and says
    /// what has landed since the watch began.
    pub fn look(&mut self, roots: &[PathBuf], follow: &[PathBuf]) -> Landed {
        for (id, bytes) in files_under(roots) {
            let most = self.most.entry(id).or_default();
            *most = (*most).max(bytes);
        }
        if !follow.is_empty() {
            for (id, bytes) in files_under(follow) {
                if let Some(most) = self.most.get_mut(&id) {
                    *most = (*most).max(bytes);
                }
            }
        }
        let mut landed = Landed::default();
        for (id, most) in &self.most {
            let before = self.before.get(id).copied().unwrap_or(0);
            if *most > before {
                landed.present += most;
                landed.fetched += most - before;
            }
        }
        landed
    }
}

/// The regular files under `roots`, by identity, with the bytes each holds.
fn files_under(roots: &[PathBuf]) -> HashMap<FileId, u64> {
    let mut files = HashMap::new();
    for root in roots {
        walk(root, &mut files);
    }
    files
}

fn walk(dir: &Path, files: &mut HashMap<FileId, u64>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        // symlink_metadata: a cache links a snapshot to its blob, and the blob is counted once
        let Ok(meta) = entry.path().symlink_metadata() else {
            continue;
        };
        if meta.is_dir() {
            walk(&entry.path(), files);
        } else if meta.is_file() {
            let bytes = files.entry(identity(&entry.path(), &meta)).or_default();
            *bytes = (*bytes).max(held(&meta));
        }
    }
}

/// What tells a file from another across a rename: its device and inode.
#[cfg(unix)]
fn identity(_path: &Path, meta: &std::fs::Metadata) -> FileId {
    use std::os::unix::fs::MetadataExt;
    (meta.dev(), meta.ino())
}

#[cfg(not(unix))]
fn identity(path: &Path, _meta: &std::fs::Metadata) -> FileId {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut h);
    (0, h.finish())
}

/// The bytes a file holds: what is allocated to it, not its length, as a downloader that sizes
/// its file first (basert's `set_len` before the chunks) has fetched none of it yet.
#[cfg(unix)]
fn held(meta: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    (meta.blocks() * 512).min(meta.len())
}

#[cfg(not(unix))]
fn held(meta: &std::fs::Metadata) -> u64 {
    meta.len()
}

/// What has landed, how fast it is landing, and what that says about the rest.
struct Gauge {
    total: Option<u64>,
    began: Instant,
    /// The most seen at once: a tool that takes its download away when it is done (basert
    /// keeps no source by default) takes nothing off the bar.
    present: u64,
    fetched: u64,
    /// When the bytes first and last grew: the rate a summary gives is over the fetch, not the
    /// lookups a tool does before it or the convert or copy it does after.
    first_growth: Option<Instant>,
    last_growth: Instant,
    samples: VecDeque<(Instant, u64)>,
}

impl Gauge {
    fn new(total: Option<u64>, began: Instant) -> Gauge {
        Gauge { total, began, present: 0, fetched: 0, first_growth: None, last_growth: began, samples: VecDeque::new() }
    }

    fn sample(&mut self, now: Instant, landed: Landed) {
        if landed.fetched > self.fetched {
            if self.first_growth.is_none() {
                // the bytes began landing after the last sample that saw none
                self.first_growth = Some(self.samples.back().map_or(now, |(t, _)| *t));
            }
            self.fetched = landed.fetched;
            self.last_growth = now;
        }
        self.present = self.present.max(landed.present);
        self.samples.push_back((now, self.fetched));
        while self.samples.len() > 2 && self.samples.front().is_some_and(|(t, _)| now.duration_since(*t) > RATE_WINDOW) {
            self.samples.pop_front();
        }
    }

    /// Bytes a second over the recent window, or over the whole watch while it is shorter.
    fn rate(&self) -> f64 {
        let (Some((t0, b0)), Some((t1, b1))) = (self.samples.front(), self.samples.back()) else {
            return 0.0;
        };
        let secs = t1.duration_since(*t0).as_secs_f64();
        if secs < 0.5 {
            return 0.0;
        }
        b1.saturating_sub(*b0) as f64 / secs
    }

    /// The seconds the rest takes at the current rate, when both are known.
    fn left(&self) -> Option<u64> {
        let total = self.total?;
        let rate = self.rate();
        (rate > 0.0).then(|| (total.saturating_sub(self.present) as f64 / rate).ceil() as u64)
    }

    /// Whether all the bytes are here and the tool is still at work: verifying, converting
    /// or copying what it fetched.
    fn finishing(&self) -> bool {
        self.total.is_some_and(|t| t > 0 && self.present >= t)
    }

    /// One terminal line: the percentage and bar when the total is known, the bytes, the rate
    /// and the time left (or elapsed).
    fn bar(&self, now: Instant) -> String {
        let rate = self.rate();
        match self.total {
            Some(total) if self.finishing() => {
                format!("100% [{}] {} / {}  fetched in {}, finishing", bar_cells(100.0, 30), size(self.present), size(total), span(self.fetch_secs()))
            }
            Some(total) if total > 0 => {
                let pct = ((self.present as f64 / total as f64) * 100.0).min(100.0);
                format!(
                    "{:>3}% [{}] {} / {}  {}  {}",
                    pct as u64,
                    bar_cells(pct, 30),
                    size(self.present),
                    size(total),
                    rate_text(rate),
                    match self.left() {
                        Some(s) => format!("{} left", span(s)),
                        None => format!("{} elapsed", span(now.duration_since(self.began).as_secs())),
                    }
                )
            }
            _ => format!(
                "{} fetched  {}  {} elapsed",
                size(self.fetched),
                rate_text(rate),
                span(now.duration_since(self.began).as_secs())
            ),
        }
    }

    /// One log line for where there is no terminal: `362 MB / 806 MB (45%), 41.2 MB/s, 11s left`.
    fn report(&self, now: Instant) -> String {
        let rate = self.rate();
        match self.total {
            Some(total) if self.finishing() => {
                format!("{} / {} (100%), fetched in {}, finishing", size(self.present), size(total), span(self.fetch_secs()))
            }
            Some(total) if total > 0 => {
                let pct = ((self.present as f64 / total as f64) * 100.0).min(100.0) as u64;
                format!(
                    "{} / {} ({pct}%), {}, {}",
                    size(self.present),
                    size(total),
                    rate_text(rate),
                    match self.left() {
                        Some(s) => format!("{} left", span(s)),
                        None => format!("{} elapsed", span(now.duration_since(self.began).as_secs())),
                    }
                )
            }
            _ => format!("{} fetched, {}, {} elapsed", size(self.fetched), rate_text(rate), span(now.duration_since(self.began).as_secs())),
        }
    }

    /// The seconds from the first byte landing to the last, rounded up (a fetch under a
    /// second is one).
    fn fetch_secs(&self) -> u64 {
        let from = self.first_growth.unwrap_or(self.began);
        (self.last_growth.duration_since(from).as_secs_f64().ceil() as u64).max(1)
    }

    /// The line left behind when the pull is done: `806 MB in 20s (40 MB/s)`, the bytes this
    /// pull fetched (a resumed one, the rest).
    fn summary(&self) -> String {
        let from = self.first_growth.unwrap_or(self.began);
        let secs = self.last_growth.duration_since(from).as_secs_f64();
        let rate = if secs >= 0.5 { self.fetched as f64 / secs } else { 0.0 };
        format!("{} in {} ({})", size(self.fetched), span(self.fetch_secs()), rate_text(rate))
    }
}

fn bar_cells(pct: f64, width: usize) -> String {
    let filled = ((pct / 100.0) * width as f64).round() as usize;
    let filled = filled.min(width);
    let mut s = "=".repeat(filled);
    if filled < width {
        s.push('>');
        s.push_str(&" ".repeat(width - filled - 1));
    }
    s
}

fn rate_text(bytes_per_sec: f64) -> String {
    if bytes_per_sec <= 0.0 {
        return "-- MB/s".into();
    }
    format!("{}/s", size(bytes_per_sec as u64))
}

/// Bytes in the unit that reads best: `806 MB`, `4.3 GB`, `512 kB`, `17 B`.
pub fn size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else if value < 10.0 {
        format!("{value:.1} {}", UNITS[unit])
    } else {
        format!("{} {}", value.round() as u64, UNITS[unit])
    }
}

/// Seconds as a span: `45s`, `3m 20s`, `1h 05m`.
pub fn span(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m {:02}s", s / 60, s % 60),
        s => format!("{}h {:02}m", s / 3600, (s % 3600) / 60),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sample of a pull that began from nothing: all that is present was fetched.
    fn all(bytes: u64) -> Landed {
        Landed { present: bytes, fetched: bytes }
    }

    #[test]
    fn sizes_and_spans_read_as_people_write_them() {
        assert_eq!(size(0), "0 B");
        assert_eq!(size(17), "17 B");
        assert_eq!(size(512_000), "512 kB");
        assert_eq!(size(806_058_240), "806 MB");
        assert_eq!(size(4_300_000_000), "4.3 GB");
        assert_eq!(size(18_400_000_000), "18 GB");
        assert_eq!(span(45), "45s");
        assert_eq!(span(200), "3m 20s");
        assert_eq!(span(3900), "1h 05m");
        assert_eq!(rate_text(0.0), "-- MB/s");
        assert_eq!(rate_text(41_200_000.0), "41 MB/s");
        assert_eq!(bar_cells(0.0, 10), ">         ");
        assert_eq!(bar_cells(50.0, 10), "=====>    ");
        assert_eq!(bar_cells(100.0, 10), "==========");
    }

    #[test]
    fn the_gauge_reads_the_rate_from_recent_samples_and_the_rest_from_it() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let mut g = Gauge::new(Some(1000), t0);
        g.sample(at(0), all(0));
        assert_eq!(g.rate(), 0.0, "one sample is no rate");
        assert!(g.bar(at(0)).starts_with("  0% [>"), "{}", g.bar(at(0)));
        // the lookups before the first byte are not the fetch
        g.sample(at(3), all(0));
        g.sample(at(4), all(100));
        g.sample(at(5), all(200));
        assert_eq!(g.rate(), 200.0 / 5.0, "over the window: the wait is in it");
        g.sample(at(6), all(300));
        g.sample(at(7), all(400));
        g.sample(at(8), all(500));
        g.sample(at(9), all(600));
        assert_eq!(g.rate(), 100.0, "the window has moved past the wait");
        assert_eq!(g.summary(), "600 B in 6s (100 B/s)", "the summary is over the fetch alone");

        let mut g = Gauge::new(Some(1000), t0);
        g.sample(at(0), all(0));
        g.sample(at(1), all(100));
        g.sample(at(2), all(200));
        assert_eq!(g.rate(), 100.0);
        assert_eq!(g.left(), Some(8));
        assert_eq!(g.bar(at(2)), " 20% [======>                       ] 200 B / 1.0 kB  100 B/s  8s left");
        assert_eq!(g.report(at(2)), "200 B / 1.0 kB (20%), 100 B/s, 8s left");
        // a tool that takes its download away leaves the bar where it was
        g.sample(at(3), all(0));
        assert_eq!((g.present, g.fetched), (200, 200));
        // the window forgets a burst a while back
        for s in 4..20 {
            g.sample(at(s), all(200));
        }
        assert_eq!(g.rate(), 0.0);
        assert!(g.bar(at(20)).ends_with("-- MB/s  20s elapsed"), "{}", g.bar(at(20)));
        // the summary's rate is over the fetch, not what the tool did after
        assert_eq!(g.summary(), "200 B in 2s (100 B/s)");
        // all here, the tool still at work
        g.sample(at(21), all(1000));
        assert!(g.finishing());
        assert_eq!(g.bar(at(22)), "100% [==============================] 1.0 kB / 1.0 kB  fetched in 21s, finishing");
        assert_eq!(g.report(at(22)), "1.0 kB / 1.0 kB (100%), fetched in 21s, finishing");
        assert_eq!(g.summary(), "1.0 kB in 21s (47 B/s)");

        let mut g = Gauge::new(None, t0);
        g.sample(at(0), all(0));
        g.sample(at(2), all(5_000_000));
        assert_eq!(g.bar(at(2)), "5.0 MB fetched  2.5 MB/s  2s elapsed");
        assert_eq!(g.report(at(2)), "5.0 MB fetched, 2.5 MB/s, 2s elapsed");

        // a resumed pull: the bar starts where the partial was, the rate and the summary are the rest
        let mut g = Gauge::new(Some(1000), t0);
        g.sample(at(0), Landed { present: 0, fetched: 0 });
        g.sample(at(1), Landed { present: 950, fetched: 50 });
        g.sample(at(2), Landed { present: 1000, fetched: 100 });
        assert_eq!(g.rate(), 50.0);
        assert!(g.finishing());
        assert_eq!(g.bar(at(2)), "100% [==============================] 1.0 kB / 1.0 kB  fetched in 2s, finishing");
        assert_eq!(g.summary(), "100 B in 2s (50 B/s)");
    }

    #[test]
    fn what_was_there_is_not_counted_and_a_file_is_known_through_its_renames() {
        let dir = std::env::temp_dir().join(format!("superfluid-progress-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("blobs")).unwrap();
        std::fs::create_dir_all(dir.join("snapshots/abc")).unwrap();
        std::fs::create_dir_all(dir.join("store")).unwrap();
        std::fs::write(dir.join("blobs/old"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.join("blobs/partial"), vec![0u8; 4096]).unwrap();
        let roots = vec![dir.join("blobs"), dir.join("snapshots")];
        let store = vec![dir.join("store")];
        std::fs::write(dir.join("store/other"), vec![0u8; 4096]).unwrap();
        let mut seen = Seen::begin(&roots);
        assert_eq!(seen.look(&roots, &store), Landed::default(), "what was there before is not the pull's");
        // a download finished by someone else a moment ago, rewritten in place: not grown, not counted
        std::fs::write(dir.join("blobs/old"), vec![1u8; 100]).unwrap();
        // a partial this pull resumes: present as a whole, fetched by what it grew
        std::fs::OpenOptions::new().append(true).open(dir.join("blobs/partial")).unwrap().write_all(&[0u8; 4096]).unwrap();
        assert_eq!(seen.look(&roots, &store), Landed { present: 8192, fetched: 4096 });
        std::fs::write(dir.join("blobs/new.incomplete"), vec![0u8; 4096]).unwrap();
        std::os::unix::fs::symlink(dir.join("blobs/new.incomplete"), dir.join("snapshots/abc/model.gguf")).unwrap();
        assert_eq!(seen.look(&roots, &store), Landed { present: 12288, fetched: 8192 }, "a new file once, its link not again");
        // renamed when done, as huggingface_hub and llama.cpp do, it is the same file
        std::fs::rename(dir.join("blobs/new.incomplete"), dir.join("blobs/new")).unwrap();
        assert_eq!(seen.look(&roots, &store), Landed { present: 12288, fetched: 8192 });
        // moved into the store it is followed to, where the bytes written just before the move are read;
        // what another pull lands there is not the pull's
        std::fs::rename(dir.join("blobs/new"), dir.join("store/new")).unwrap();
        std::fs::OpenOptions::new().append(true).open(dir.join("store/new")).unwrap().write_all(&[0u8; 4096]).unwrap();
        std::fs::write(dir.join("store/theirs"), vec![0u8; 8192]).unwrap();
        assert_eq!(seen.look(&roots, &store), Landed { present: 16384, fetched: 12288 });
        assert_eq!(seen.look(&roots, &[]), Landed { present: 16384, fetched: 12288 }, "what left the roots stays counted");
        assert_eq!(seen.look(&[dir.join("nowhere"), dir.join("blobs")], &store), Landed { present: 16384, fetched: 12288 }, "a root not there counts nothing");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_sized_before_it_is_fetched_counts_as_it_fills() {
        let dir = std::env::temp_dir().join(format!("superfluid-progress-sized-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let roots = vec![dir.clone()];
        let mut seen = Seen::begin(&roots);
        // sized as a model is: APFS fills a hole in a file of 16 MB or less whole on its first write
        const SIZE: u64 = 64 << 20;
        let f = std::fs::File::create(dir.join("model.partial")).unwrap();
        f.set_len(SIZE).unwrap();
        drop(f);
        // basert's set_len holds no bytes yet (a filesystem that allocates on set_len shows them all)
        let sized = seen.look(&roots, &[]);
        assert_eq!(sized.present, sized.fetched);
        let mut f = std::fs::OpenOptions::new().write(true).open(dir.join("model.partial")).unwrap();
        f.write_all(&[7u8; 300]).unwrap();
        f.sync_all().unwrap();
        drop(f);
        let filled = seen.look(&roots, &[]);
        assert!(filled.fetched >= sized.fetched.max(300), "{filled:?} after {sized:?}");
        assert!(filled.fetched <= SIZE, "{filled:?}");
        if sized.fetched == 0 {
            assert!(filled.fetched < SIZE, "the 300 bytes written, in whole blocks: {filled:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_watch_runs_what_it_is_given_and_hands_back_its_result() {
        assert_eq!(watched(None, || 7), 7);
        let dir = std::env::temp_dir().join(format!("superfluid-progress-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let watch = Watch { roots: vec![dir.clone()], follow: vec![], total: Some(10) };
        let got = watched(Some(watch), || {
            std::fs::write(dir.join("part"), vec![0u8; 10]).unwrap();
            std::thread::sleep(Duration::from_millis(600));
            "done"
        });
        assert_eq!(got, "done");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
