//! A request's images encode between ticks, a slice at a time: the lanes
//! already running keep ticking while they do, and an image that fails to
//! encode fails its own request alone.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::wal::role;
use superfluid_daemon::{Daemon, DaemonOptions, EngineHost, GenParams, SessionStore};
use superfluid_engine::mock::MOCK_UNENCODABLE;
use superfluid_engine::{EngineConfig, MockEngine};

const TICK: Duration = Duration::from_millis(5);

/// The engine's tick count at each image encode.
type EncodeLog = Arc<Mutex<Vec<u64>>>;

fn daemon(encode: Duration, max_live_media: usize) -> Arc<Daemon> {
    daemon_with(encode, max_live_media, false)
}

fn daemon_with(encode: Duration, max_live_media: usize, park: bool) -> Arc<Daemon> {
    daemon_in(encode, max_live_media, park).0
}

fn daemon_in(encode: Duration, max_live_media: usize, park: bool) -> (Arc<Daemon>, std::path::PathBuf) {
    daemon_logging(encode, max_live_media, park, None)
}

fn daemon_logging(
    encode: Duration,
    max_live_media: usize,
    park: bool,
    encode_log: Option<EncodeLog>,
) -> (Arc<Daemon>, std::path::PathBuf) {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "superfluid-media-admission-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let mut cfg = EngineConfig {
        image_token_id: 999,
        media_tokens_per_image: 4,
        tick_delay: TICK,
        media_encode_delay: encode,
        media_encode_log: encode_log,
        max_live_media,
        ..EngineConfig::default()
    };
    cfg.spaces.retain(|s| s.kind == superfluid_abi::space_kind::PAGED_TOKEN_KV);
    let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).unwrap();
    let opts = DaemonOptions {
        max_lanes: 4,
        media_dir: Some(dir.join("media")),
        park_dir: park.then(|| dir.join("park")),
        ..Default::default()
    };
    (Arc::new(Daemon::with_options(store, host, Box::new(MockChatCodec), opts).unwrap()), dir)
}

/// A session whose user turn carries one image per entry of `images`.
fn image_session(d: &Daemon, images: &[Vec<u8>]) -> u64 {
    let s = d.create(None, GenParams::default()).unwrap();
    d.append_message(s, role::SYSTEM, "look".into()).unwrap();
    for (i, bytes) in images.iter().enumerate() {
        let hash = d.put_media(bytes).unwrap();
        d.append_image(s, role::USER, &hash, &format!("picture {i}:"), " ").unwrap();
    }
    s
}

fn images(n: usize, tag: &str) -> Vec<Vec<u8>> {
    (0..n).map(|i| format!("{tag} image {i}").into_bytes()).collect()
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

/// A request carrying `n` images, each taking `encode` to encode, admitted
/// beside a decoding lane: the engine's tick count at each of its encodes,
/// the longest stall the decoding lane saw, and how long the request took.
fn encodes_beside(n: usize, encode: Duration) -> (Vec<u64>, Duration, Duration) {
    let log = EncodeLog::default();
    let (d, _) = daemon_logging(encode, 0, false, Some(Arc::clone(&log)));
    let running = start_decoding(&d);
    let s = image_session(&d, &images(n, "stall"));
    let t0 = Instant::now();
    let out = d.generate(s, 4).unwrap();
    let t1 = Instant::now();
    assert_eq!(out.tokens_generated, 4);
    let gap = running.longest_gap(t0, t1);
    running.stop(&d);
    let at = log.lock().unwrap().clone();
    (at, gap, t1 - t0)
}

#[test]
fn a_decoding_lane_keeps_ticking_while_another_request_encodes_its_images() {
    // Two 150 ms encodes do not fit one open tick, so each round encodes one.
    let encode = Duration::from_millis(150);
    for n in [1usize, 4, 8] {
        let (at, gap, took) = encodes_beside(n, encode);
        // The times are a diagnostic, not a check: the mock's ticks cost CPU
        // time too, which a small runner stretches (160 ms on a laptop, 320-345
        // ms on a 3-core macOS runner, the same for 1, 4 and 8 images).
        eprintln!("[media-admission] images={n} encode={encode:?} longest_gap={gap:?} request={took:?} encodes at ticks {at:?}");
        assert_eq!(at.len(), n, "one encode for each image: {at:?}");
        // A tick ran between every two of the request's encodes: the decoding
        // lane's rounds. Encoded at admission, they all sat at one tick count.
        assert!(
            at.windows(2).all(|w| w[1] > w[0]),
            "{n} images: two encodes with no tick between them (ticks at each encode: {at:?})"
        );
    }
}

#[test]
fn an_image_that_fails_to_encode_fails_its_own_request_and_releases_the_rest() {
    // Two images held at once: a handle the failed request kept would make
    // the next two-image request fail.
    let d = daemon(Duration::from_millis(20), 2);
    let running = start_decoding(&d);
    let mut bad = images(2, "bad");
    bad[1] = [MOCK_UNENCODABLE, b" tower refuses this"].concat();
    let s = image_session(&d, &bad);
    let err = d.generate(s, 4).unwrap_err().to_string();
    assert!(err.contains("media encode/bind failed"), "{err}");
    let good = image_session(&d, &images(2, "good"));
    assert_eq!(d.generate(good, 4).unwrap().tokens_generated, 4);
    let ev = running.events.lock().unwrap().len();
    std::thread::sleep(Duration::from_millis(50));
    assert!(running.events.lock().unwrap().len() > ev, "the decoding lane ran on through both");
    running.stop(&d);
}

#[test]
fn a_request_cancelled_while_its_images_encode_ends_and_frees_its_slot() {
    let log = EncodeLog::default();
    let (d, _) = daemon_logging(Duration::from_millis(100), 3, false, Some(Arc::clone(&log)));
    let running = start_decoding(&d);
    let s = image_session(&d, &images(3, "cancelled"));
    let dd = Arc::clone(&d);
    let gen = std::thread::spawn(move || dd.generate(s, 4));
    // Cancelled once its first image's encode has begun, not after a wait: a
    // cancel that comes before the request has registered is dropped as one
    // meant for the session's previous generation, and one that comes before
    // any encode ends a request that holds no image. The mock notes an encode
    // under the log's lock, so no further encode begins while the cancel goes
    // in.
    let begun = loop {
        assert!(!gen.is_finished(), "the request ended before any of its images encoded");
        let at = log.lock().unwrap();
        if !at.is_empty() {
            d.cancel_registry().cancel(s);
            break at.len();
        }
        drop(at);
        std::thread::sleep(Duration::from_millis(1));
    };
    let out = gen.join().unwrap().unwrap();
    assert_eq!(out.tokens_generated, 0, "cancelled with {begun} of its 3 encodes begun");
    // It stopped encoding: at most the encode already asked for when the
    // cancel went in began after it.
    let encoded = log.lock().unwrap().len();
    assert!(encoded <= begun + 1, "{encoded} encodes, {begun} begun at the cancel");
    // Its encoded images were let go: another three-image request fits.
    let next = image_session(&d, &images(3, "after"));
    assert_eq!(d.generate(next, 4).unwrap().tokens_generated, 4);
    running.stop(&d);
}

#[test]
fn a_resumed_chat_with_many_new_images_is_admitted_before_its_seed_lapses() {
    // The mock's seed lease is 8 ticks; ten new images at one a tick would
    // outlast it.
    let (d, dir) = daemon_in(Duration::from_millis(150), 0, true);
    let s = image_session(&d, &images(1, "first turn"));
    assert_eq!(d.generate(s, 4).unwrap().tokens_generated, 4);
    let park = dir.join("park").join(format!("{s}.park"));
    let t0 = Instant::now();
    while !park.is_file() {
        assert!(t0.elapsed() < Duration::from_secs(5), "never parked");
        std::thread::sleep(Duration::from_millis(2));
    }
    let running = start_decoding(&d);
    for (i, bytes) in images(10, "second turn").iter().enumerate() {
        let hash = d.put_media(bytes).unwrap();
        d.append_image(s, role::USER, &hash, &format!("and {i}:"), " ").unwrap();
    }
    let out = d.generate(s, 4).unwrap();
    assert_eq!(out.tokens_generated, 4);
    assert!(out.warm_prefix > 0, "resumed from its parked first turn");
    running.stop(&d);
}

#[test]
fn a_resumed_chat_taken_behind_another_requests_images_keeps_its_seed() {
    // The mock's seed lease is 8 ticks. A resumed chat with one new image is
    // taken while a request ahead of it is still encoding fourteen, an image a
    // round (two 130 ms encodes do not fit one open tick): waiting its turn it
    // reached the front after its lease ran out, its admission was refused as
    // a stale seed, and the tick failed every request in it.
    let (d, dir) = daemon_in(Duration::from_millis(130), 0, true);
    let s = image_session(&d, &images(1, "first turn"));
    assert_eq!(d.generate(s, 4).unwrap().tokens_generated, 4);
    let park = dir.join("park").join(format!("{s}.park"));
    let t0 = Instant::now();
    while !park.is_file() {
        assert!(t0.elapsed() < Duration::from_secs(5), "never parked");
        std::thread::sleep(Duration::from_millis(2));
    }
    let hash = d.put_media(b"second turn image").unwrap();
    d.append_image(s, role::USER, &hash, "and this:", " ").unwrap();

    let running = start_decoding(&d);
    let ahead = image_session(&d, &images(14, "ahead"));
    let dd = Arc::clone(&d);
    let ahead_gen = std::thread::spawn(move || dd.generate(ahead, 4));
    // Inside the first of the fourteen encodes.
    std::thread::sleep(Duration::from_millis(15));
    let resumed = d.generate(s, 4);
    let ahead_out = ahead_gen.join().unwrap();
    running.stop(&d);
    let resumed = resumed.unwrap_or_else(|e| panic!("the resumed chat failed: {e}"));
    assert_eq!(resumed.tokens_generated, 4);
    assert!(resumed.warm_prefix > 0, "resumed from its parked first turn");
    let ahead_out = ahead_out.unwrap_or_else(|e| panic!("the request ahead of it failed: {e}"));
    assert_eq!(ahead_out.tokens_generated, 4);
}

