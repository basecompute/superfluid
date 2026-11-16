//! A prompt carrying images prefills one image a tick, in open-tick-sized
//! chunks that never end inside an image's placeholder run, so the lanes
//! decoding beside it keep getting rounds between its images.

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::wal::role;
use superfluid_daemon::{Daemon, DaemonOptions, EngineHost, GenParams, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine, TickRecord};

const TICK: Duration = Duration::from_millis(5);

type TickLog = Arc<Mutex<Vec<TickRecord>>>;

/// A daemon on a mock that keeps a log of its ticks. `runs_per_chunk` 1 faults a
/// chunk carrying two images, as baseRT through 0.5.0 does; 0 takes any number.
fn daemon(
    runs_per_chunk: u32,
    tokens_per_image: u32,
    prefill_token_delay: Duration,
    prefill_budget: Option<u32>,
) -> (Arc<Daemon>, TickLog) {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "superfluid-media-prefill-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let log: TickLog = Arc::default();
    let mut cfg = EngineConfig {
        image_token_id: 999,
        media_tokens_per_image: tokens_per_image,
        media_runs_per_chunk: runs_per_chunk,
        prefill_token_delay,
        tick_log: Some(Arc::clone(&log)),
        tick_delay: TICK,
        pool_bytes: 1 << 22,
        ..EngineConfig::default()
    };
    // No recurrent space: image lanes share the engine instead of running alone.
    cfg.spaces.retain(|s| s.kind == superfluid_abi::space_kind::PAGED_TOKEN_KV);
    let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).unwrap();
    let mut opts = DaemonOptions { max_lanes: 4, media_dir: Some(dir.join("media")), ..Default::default() };
    if let Some(b) = prefill_budget {
        opts.prefill_budget = b;
    }
    (Arc::new(Daemon::with_options(store, host, Box::new(MockChatCodec), opts).unwrap()), log)
}

/// A session whose user turn carries `n` images.
fn image_session(d: &Daemon, n: usize, tag: &str) -> u64 {
    let s = d.create(None, GenParams::default()).unwrap();
    d.append_message(s, role::SYSTEM, "look".into()).unwrap();
    for i in 0..n {
        let hash = d.put_media(format!("{tag} image {i}").as_bytes()).unwrap();
        d.append_image(s, role::USER, &hash, &format!("picture {i}:"), " ").unwrap();
    }
    s
}

/// A lane decoding in the background, the time of each of its events kept.
struct Decoding {
    session: u64,
    events: Arc<Mutex<Vec<Instant>>>,
    join: std::thread::JoinHandle<()>,
}

fn start_decoding(d: &Arc<Daemon>) -> Decoding {
    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, None, (0..64).map(|t| t % 400 + 50).collect()).unwrap();
    let events: Arc<Mutex<Vec<Instant>>> = Arc::default();
    let (dd, ev) = (Arc::clone(d), Arc::clone(&events));
    let join = std::thread::spawn(move || {
        let _ = dd.generate_streaming(session, 1_000_000, |_| {
            ev.lock().unwrap().push(Instant::now());
            Ok(())
        });
    });
    while events.lock().unwrap().len() < 20 {
        assert!(!join.is_finished(), "the decoding lane ended early");
        std::thread::sleep(Duration::from_millis(1));
    }
    Decoding { session, events, join }
}

impl Decoding {
    /// The longest wait between two of its events inside `[from, to]`.
    fn longest_gap(&self, from: Instant, to: Instant) -> Duration {
        let ev = self.events.lock().unwrap();
        let inside: Vec<Instant> = ev.iter().copied().filter(|&t| t >= from && t <= to).collect();
        inside.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(to - from)
    }

    fn stop(self, d: &Daemon) {
        d.cancel_registry().cancel(self.session);
        self.join.join().unwrap();
    }
}

/// Every prefill chunk in the log ends outside any image run and carries one
/// image at most; the lane whose chunks carried images, with its ticks.
fn image_lane_ticks(log: &TickLog) -> (u64, Vec<TickRecord>) {
    let log = log.lock().unwrap();
    for t in log.iter() {
        for p in &t.prefills {
            for &(rs, re) in &p.runs {
                assert!(p.start <= rs && re <= p.end, "chunk {}..{} cuts the image at {rs}..{re}", p.start, p.end);
            }
            assert!(p.runs.len() <= 1, "chunk {}..{} carries {} images", p.start, p.end, p.runs.len());
        }
    }
    let lanes: HashSet<u64> =
        log.iter().flat_map(|t| &t.prefills).filter(|p| !p.runs.is_empty()).map(|p| p.lane_tag).collect();
    assert_eq!(lanes.len(), 1, "one lane carried images");
    let lane = *lanes.iter().next().unwrap();
    let ticks = log.iter().filter(|t| t.prefills.iter().any(|p| p.lane_tag == lane)).cloned().collect();
    (lane, ticks)
}

#[test]
fn a_multi_image_prompt_prefills_an_image_a_tick_beside_a_decoding_lane() {
    for runs_per_chunk in [0, 1] {
        let (d, log) = daemon(runs_per_chunk, 4, Duration::ZERO, None);
        let running = start_decoding(&d);
        let s = image_session(&d, 8, "beside");
        let out = d.generate(s, 4).unwrap_or_else(|e| panic!("runs_per_chunk {runs_per_chunk}: {e}"));
        assert_eq!(out.tokens_generated, 4);
        running.stop(&d);

        let (image_lane, ticks) = image_lane_ticks(&log);
        let with_image =
            ticks.iter().filter(|t| t.prefills.iter().any(|p| p.lane_tag == image_lane && !p.runs.is_empty()));
        assert_eq!(with_image.count(), 8, "one tick for each image");
        for t in &ticks {
            assert!(
                t.decodes.iter().any(|&l| l != image_lane),
                "the decoding lane gets rounds in every tick of the image prompt's prefill: {t:?}"
            );
        }
    }
}

#[test]
fn a_lone_multi_image_prompt_is_served_one_image_a_chunk() {
    // Nothing else runs, so its tick takes the whole lone budget: the images
    // still go one a chunk, which baseRT through 0.5.0 needs.
    let (d, log) = daemon(1, 4, Duration::ZERO, None);
    let s = image_session(&d, 3, "lone");
    let out = d.generate(s, 4).expect("served");
    assert_eq!(out.tokens_generated, 4);
    let (_, ticks) = image_lane_ticks(&log);
    assert!(ticks.len() >= 3, "{} ticks for three images", ticks.len());
}

#[test]
fn an_image_larger_than_the_tick_budget_prefills_whole() {
    // 200 placeholders against a 64-token budget: the chunk that starts with
    // the image takes it whole, and the plan's budget covers it.
    let (d, log) = daemon(1, 200, Duration::ZERO, Some(64));
    let running = start_decoding(&d);
    let s = image_session(&d, 2, "large");
    let out = d.generate(s, 4).expect("served");
    assert_eq!(out.tokens_generated, 4);
    running.stop(&d);
    let (image_lane, ticks) = image_lane_ticks(&log);
    let whole: Vec<(u32, u32)> = ticks
        .iter()
        .flat_map(|t| &t.prefills)
        .filter(|p| p.lane_tag == image_lane && !p.runs.is_empty())
        .map(|p| (p.start, p.end))
        .collect();
    assert_eq!(whole.len(), 2, "{whole:?}");
    assert!(whole.iter().all(|&(s, e)| e - s == 200), "each image in a chunk of its own: {whole:?}");
}

#[test]
fn a_decoding_lane_waits_one_image_not_the_whole_prompt() {
    // Each prompt token takes 1 ms to prefill: eight 48-token images are
    // ~0.4 s of prefill, which the decoding lane used to wait out at once.
    let per_token = Duration::from_millis(1);
    let (d, _log) = daemon(0, 48, per_token, None);
    let running = start_decoding(&d);
    let s = image_session(&d, 8, "stall");
    let t0 = Instant::now();
    let out = d.generate(s, 4).unwrap();
    let t1 = Instant::now();
    assert_eq!(out.tokens_generated, 4);
    let gap = running.longest_gap(t0, t1);
    running.stop(&d);
    let whole = per_token * 8 * 48;
    eprintln!("[media-prefill] longest decode gap {gap:?}; the images alone prefill in {whole:?}; request {:?}", t1 - t0);
    assert!(gap < whole / 2, "the decoding lane stalled {gap:?} beside {whole:?} of image prefill");
}
