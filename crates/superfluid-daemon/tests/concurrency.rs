//! Concurrency stress + correctness for the single-node daemon.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use superfluid_abi::finish;
use superfluid_daemon::{Daemon, EngineHost, EventBody, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};

static DIRS: AtomicU64 = AtomicU64::new(0);

fn fresh_dir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "superfluid-conc-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn daemon_in(dir: &std::path::Path, max_lanes: usize) -> Arc<Daemon> {
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    Arc::new(Daemon::new(store, host, Box::new(MockCodec), max_lanes))
}

fn generated(events: &[superfluid_daemon::CommittedEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Generated { span, .. } => Some(span.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

fn reference(prompt: &[u32], max: u32) -> Vec<u32> {
    let d = daemon_in(&fresh_dir(), 4);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, prompt.to_vec()).unwrap();
    let out = d.generate(s, max).unwrap();
    generated(&out.events)
}

#[test]
fn concurrent_distinct_sessions_match_their_solo_output() {
    const N: usize = 32;
    const MAX: u32 = 24;
    let prompts: Vec<Vec<u32>> = (0..N).map(|i| (i as u32 * 100..i as u32 * 100 + 20).collect()).collect();
    let refs: Vec<Vec<u32>> = prompts.iter().map(|p| reference(p, MAX)).collect();

    let d = daemon_in(&fresh_dir(), 8);
    let barrier = Arc::new(Barrier::new(N));
    let mut handles = Vec::new();
    for prompt in &prompts {
        let d = Arc::clone(&d);
        let b = Arc::clone(&barrier);
        let prompt = prompt.clone();
        handles.push(std::thread::spawn(move || {
            let s = d.create(None, GenParams::default()).unwrap();
            d.append(s, None, prompt).unwrap();
            b.wait();
            let out = d.generate(s, MAX).unwrap();
            generated(&out.events)
        }));
    }
    for (i, h) in handles.into_iter().enumerate() {
        let got = h.join().expect("thread panicked");
        assert_eq!(got, refs[i], "session {i} diverged under concurrency");
    }
}

#[test]
fn concurrent_generate_on_one_session_is_serialized() {
    const K: usize = 12;
    let d = daemon_in(&fresh_dir(), 8);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..16).collect()).unwrap();

    let barrier = Arc::new(Barrier::new(K));
    let ok = Arc::new(AtomicUsize::new(0));
    let busy = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    for _ in 0..K {
        let d = Arc::clone(&d);
        let b = Arc::clone(&barrier);
        let (ok, busy) = (Arc::clone(&ok), Arc::clone(&busy));
        handles.push(std::thread::spawn(move || {
            b.wait();
            match d.generate(s, 4) {
                Ok(_) => ok.fetch_add(1, Ordering::Relaxed),
                Err(superfluid_daemon::DaemonError::SessionBusy(_)) => busy.fetch_add(1, Ordering::Relaxed),
                Err(e) => panic!("unexpected error: {e}"),
            };
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert!(ok.load(Ordering::Relaxed) >= 1);
    assert_eq!(ok.load(Ordering::Relaxed) + busy.load(Ordering::Relaxed), K);
    let store = d.store();
    let store = store.lock().unwrap();
    let s_state = store.session(s).unwrap();
    let ids: Vec<u64> = s_state.events.iter().map(|e| e.event_id).collect();
    for w in ids.windows(2) {
        assert!(w[1] > w[0], "event ids not strictly increasing: {ids:?}");
    }
}

#[test]
fn purge_racing_generation_stays_consistent() {
    const ITERS: usize = 60;
    let dir = fresh_dir();
    let d = daemon_in(&dir, 8);
    for i in 0..ITERS {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..16).collect()).unwrap();
        let gen = {
            let d = Arc::clone(&d);
            std::thread::spawn(move || d.generate(s, 16).map(|_| ()))
        };
        let purge = {
            let d = Arc::clone(&d);
            std::thread::spawn(move || d.purge(s, 0, superfluid_daemon::wal::PurgeMode::Reroot).map(|_| ()))
        };
        let g = gen.join().unwrap();
        let p = purge.join().unwrap();
        for r in [g, p] {
            if let Err(e) = r {
                use superfluid_daemon::DaemonError::*;
                assert!(
                    matches!(e, Purged(_) | SessionBusy(_) | GenerationConflict { .. } | UnknownSession(_) | PurgeTimeout(_) | EmptySession(_)),
                    "iter {i}: unexpected error {e}"
                );
            }
        }
    }
    drop(d);
    let reopened = SessionStore::open(&dir.join("wal.log"));
    assert!(reopened.is_ok(), "WAL failed to replay after purge/generate races");
}

#[test]
fn permission_and_generation_never_overlap() {
    const ITERS: usize = 80;
    let d = daemon_in(&fresh_dir(), 8);
    for i in 0..ITERS {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..16).collect()).unwrap();
        let g = {
            let d = Arc::clone(&d);
            std::thread::spawn(move || d.generate(s, 8))
        };
        let p = {
            let d = Arc::clone(&d);
            std::thread::spawn(move || d.request_permission(s, 0, "may I".into()))
        };
        let gr = g.join().unwrap();
        let pr = p.join().unwrap();
        use superfluid_daemon::DaemonError::*;
        match (&gr, &pr) {
            (Ok(_), Ok(_)) => {  }
            (Ok(_), Err(SessionBusy(_))) => {}
            (Err(PermissionPending(_)), Ok(_)) => {}
            (Err(SessionBusy(_)), Ok(_)) => {}
            other => panic!("iter {i}: illegal race outcome {other:?}"),
        }
    }
}

#[test]
fn cancel_during_generation_is_clean() {
    let d = daemon_in(&fresh_dir(), 4);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..16).collect()).unwrap();

    let cancels = d.cancel_registry();
    let gen = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || d.generate(s, 4096))
    };
    for _ in 0..200 {
        let mut c = cancels.lock().unwrap();
        if d.active_registry().lock().unwrap().contains(&s) {
            c.insert(s);
            break;
        }
        drop(c);
        std::thread::sleep(Duration::from_millis(1));
    }
    let out = gen.join().unwrap().expect("generate errored");
    assert!(out.finish == finish::CANCELLED || out.tokens_generated > 0);
    for _ in 0..500 {
        if !d.active_registry().lock().unwrap().contains(&s) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let again = d.generate(s, 4);
    assert!(again.is_ok(), "session stuck after cancel: {:?}", again.err());
}

#[test]
fn throughput_under_concurrency() {
    const N: usize = 24;
    const MAX: u32 = 32;
    let prompts: Vec<Vec<u32>> = (0..N).map(|i| (i as u32 * 10..i as u32 * 10 + 12).collect()).collect();

    let d = daemon_in(&fresh_dir(), 8);
    let barrier = Arc::new(Barrier::new(N));
    let total_tokens = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let mut handles = Vec::new();
    for p in &prompts {
        let d = Arc::clone(&d);
        let b = Arc::clone(&barrier);
        let tt = Arc::clone(&total_tokens);
        let prompt = p.clone();
        handles.push(std::thread::spawn(move || {
            let s = d.create(None, GenParams::default()).unwrap();
            d.append(s, None, prompt).unwrap();
            b.wait();
            let out = d.generate(s, MAX).unwrap();
            tt.fetch_add(out.tokens_generated as u64, Ordering::Relaxed);
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }
    let elapsed = start.elapsed();
    let toks = total_tokens.load(Ordering::Relaxed);
    let tps = toks as f64 / elapsed.as_secs_f64();
    eprintln!(
        "throughput: {N} sessions x {MAX} tok = {toks} tokens in {:.3}s = {tps:.0} tok/s (8 lanes)",
        elapsed.as_secs_f64()
    );
    assert_eq!(toks, (N as u64) * (MAX as u64), "some tokens went missing");
    assert!(elapsed < Duration::from_secs(30), "throughput run stalled: {elapsed:?}");
}

#[test]
fn mixed_workload_stress_and_wal_replays() {
    const THREADS: usize = 16;
    const OPS: usize = 40;
    let dir = fresh_dir();
    let d = daemon_in(&dir, 8);
    let mut pool = Vec::new();
    for _ in 0..THREADS {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..16).collect()).unwrap();
        pool.push(s);
    }
    let pool = Arc::new(pool);
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for t in 0..THREADS {
        let d = Arc::clone(&d);
        let pool = Arc::clone(&pool);
        let b = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            b.wait();
            for op in 0..OPS {
                let mix = (t * 7 + op * 13) % 6;
                let s = pool[(t + op) % pool.len()];
                match mix {
                    0 | 1 => {
                        let _ = d.generate(s, 4);
                    }
                    2 => {
                        let _ = d.append(s, None, (op as u32..op as u32 + 4).collect());
                    }
                    3 => {
                        let _ = d.fork(s, 1, None);
                    }
                    4 => {
                        d.cancel_registry().lock().unwrap().insert(s);
                    }
                    _ => {
                        let _ = d.request_permission(s, 0, "p".into());
                        let _ = d.respond_permission(s, 0, true);
                    }
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked in mixed workload");
    }
    drop(d);
    let reopened = SessionStore::open(&dir.join("wal.log"));
    assert!(reopened.is_ok(), "WAL failed to replay after the mixed storm");
}

#[test]
fn wall_deadline_expires_a_long_generation_mid_flight() {
    let d = daemon_in(&fresh_dir(), 4);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..16).collect()).unwrap();
    let start = Instant::now();
    let (tokens, expired) = d
        .generate_tokens((0..16).collect(), GenParams::default(), 10_000_000, 25)
        .unwrap();
    let elapsed = start.elapsed();
    assert!(expired, "generation ran to the huge budget instead of expiring");
    assert!(tokens.is_empty(), "an expired generation returns no committed stream");
    assert!(
        elapsed < Duration::from_secs(5),
        "deadline did not bound execution: {elapsed:?}"
    );
}

#[test]
fn back_to_back_generate_same_session_is_never_spuriously_busy() {
    let d = daemon_in(&fresh_dir(), 8);

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut bg = Vec::new();
    for _ in 0..6 {
        let d = Arc::clone(&d);
        let stop = Arc::clone(&stop);
        bg.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Ok(s) = d.create(None, GenParams::default()) {
                    let _ = d.append(s, None, (0..16).collect());
                    let _ = d.generate(s, 4);
                }
            }
        }));
    }

    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..16).collect()).unwrap();
    for i in 0..200 {
        match d.generate(s, 2) {
            Ok(_) => {}
            Err(superfluid_daemon::DaemonError::SessionBusy(_)) => {
                stop.store(true, Ordering::Relaxed);
                panic!("iter {i}: SessionBusy immediately after the previous Done — active cleared after the terminal");
            }
            Err(e) => {
                stop.store(true, Ordering::Relaxed);
                panic!("iter {i}: unexpected error {e}");
            }
        }
    }
    stop.store(true, Ordering::Relaxed);
    for h in bg {
        let _ = h.join();
    }
}
