//! The scheduler follows the model's capability record.

use std::io::Write;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use superfluid_daemon::registry::ModelRegistry;
use superfluid_daemon::{Daemon, DaemonError, DaemonOptions, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_executor::fake::FakeConfig;
use superfluid_executor::{Executor, ExecutorConfig, FakePrimitives};

static DIRS: AtomicU64 = AtomicU64::new(0);

fn scratch() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-record-driven-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn executor_daemon(cfg: FakeConfig, ctx: u32, lanes: usize) -> Daemon {
    executor_daemon_with(cfg, ctx, lanes, superfluid_daemon::pressure::PressureConfig::None)
}

fn executor_daemon_with(cfg: FakeConfig, ctx: u32, lanes: usize, pressure_source: superfluid_daemon::pressure::PressureConfig) -> Daemon {
    let dir = scratch();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || {
        let engine = Executor::new(FakePrimitives::new(cfg), ExecutorConfig::default());
        let vocab = engine.descriptor().vocab_size;
        (engine, Some(superfluid_daemon::runtime::ring_specs_for(ctx, vocab)))
    })
    .expect("spawn");
    Daemon::with_options(
        store,
        host,
        Box::new(MockCodec),
        DaemonOptions { max_lanes: lanes, media_dir: Some(dir.join("media")), pressure_source, ..Default::default() },
    )
    .unwrap()
}

#[test]
fn a_stream_longer_than_the_runtime_holds_is_refused_at_admission() {
    let d = executor_daemon(FakeConfig { max_seq_len: 256, ..Default::default() }, 4096, 4);
    assert_eq!(d.max_stream_tokens(), 256, "the runtime's limit, not the rings'");
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..300).collect()).unwrap();
    match d.generate(s, 4) {
        Err(DaemonError::StreamTooLong { len, max }) => assert_eq!((len, max), (300, 256)),
        other => panic!("expected the context error, got {other:?}"),
    }
    let ok = d.create(None, GenParams::default()).unwrap();
    d.append(ok, None, (0..200).collect()).unwrap();
    assert_eq!(d.generate(ok, 4).unwrap().tokens_generated, 4);
}

#[test]
fn lanes_never_outnumber_the_sequences_the_runtime_holds() {
    let d = Arc::new(executor_daemon(FakeConfig { max_seqs: 2, ..Default::default() }, 4096, 8));
    let done = Arc::new(AtomicUsize::new(0));
    let workers: Vec<_> = (0..5u32)
        .map(|i| {
            let (d, done) = (Arc::clone(&d), Arc::clone(&done));
            std::thread::spawn(move || {
                let s = d.create(None, GenParams::default()).unwrap();
                d.append(s, None, (i * 1000..i * 1000 + 40).collect()).unwrap();
                let out = d.generate(s, 24).expect("every request is served");
                assert_eq!(out.tokens_generated, 24);
                done.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();
    for w in workers {
        w.join().expect("worker");
    }
    assert_eq!(done.load(Ordering::Relaxed), 5);
}

/// Three sessions' first turns, then the first session's second turn: how
/// much of it was served warm, and what it generated.
fn second_turn_after_two_others(cfg: FakeConfig) -> (u64, Vec<superfluid_daemon::EventBody>) {
    let d = executor_daemon(cfg, 4096, 4);
    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, None, (0..200).collect()).unwrap();
    assert_eq!(d.generate(session, 8).unwrap().warm_prefix, 0, "the first turn is cold");
    for base in [1000u32, 2000] {
        let other = d.create(None, GenParams::default()).unwrap();
        d.append(other, None, (base..base + 200).collect()).unwrap();
        d.generate(other, 8).unwrap();
    }
    d.append(session, None, (500..520).collect()).unwrap();
    let out = d.generate(session, 8).unwrap();
    (out.warm_prefix, out.events.into_iter().map(|e| e.body).collect())
}

/// A runtime that affords the cache little of its pool keeps the rest as exports, and a
/// session's next turn is served warm from one.
#[test]
fn a_turn_is_served_warm_from_a_cache_entry_out_of_the_pool() {
    let shared = |resident| FakeConfig { copy_shares_cells: true, cache_resident_cells: resident, cache_exported_cells: 4096, ..Default::default() };
    // No bound: the first turn's entry is a sequence in the pool. A bound of
    // 64 cells: each 208-token turn leaves the pool as it is published.
    let (in_pool, exported) = (second_turn_after_two_others(shared(0)), second_turn_after_two_others(shared(64)));
    assert_eq!(exported.0, 208, "the whole first turn is served from its export");
    assert_eq!(exported, in_pool, "and the turn continues as from an entry in the pool");
}

/// With a buffer per lane, an entry that gives its slot to another session is exported,
/// and its next turn is served warm from the export.
#[test]
fn a_turn_is_served_warm_after_other_sessions_took_every_slot() {
    let streams = |exported| FakeConfig { max_seqs: 2, takeover_preferred: true, cache_exported_cells: exported, ..Default::default() };
    let (evicted, kept) = (second_turn_after_two_others(streams(0)), second_turn_after_two_others(streams(4096)));
    assert_eq!(evicted.0, 0, "with no room for exports the entry was evicted for a slot");
    assert_eq!(kept.0, 208, "with room it is served from its export");
    assert_eq!(kept.1, evicted.1, "and the turn continues as the cold one does");
}

/// When the machine is short of memory the cache gives up what it holds in
/// host memory: a session's first turn, exported as it was published, is
/// gone once a relief round has run, and the session's next turn is cold.
#[test]
fn memory_pressure_on_the_machine_takes_the_caches_exports() {
    use superfluid_daemon::pressure::{ManualPressure, PressureConfig, PressureLevel};
    let cfg = FakeConfig {
        copy_shares_cells: true,
        cache_resident_cells: 64,
        cache_exported_cells: 4096,
        step_delay: std::time::Duration::from_millis(2),
        ..Default::default()
    };
    let manual = ManualPressure::new();
    let d = Arc::new(executor_daemon_with(cfg, 4096, 4, PressureConfig::Manual(manual.clone())));
    let stats = d.sched_stats();
    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, None, (0..200).collect()).unwrap();
    d.generate(session, 8).unwrap();
    // A long generation beside it keeps the scheduler ticking, so the first
    // turn is published and relief rounds run while the pressure lasts.
    let busy = d.create(None, GenParams::default()).unwrap();
    d.append(busy, None, (3000..3100).collect()).unwrap();
    let dd = Arc::clone(&d);
    let running = std::thread::spawn(move || dd.generate(busy, 400));
    std::thread::sleep(std::time::Duration::from_millis(100));
    let before = stats.os_pressure_events.load(Ordering::Relaxed);
    manual.set(PressureLevel::Warning);
    for _ in 0..1000 {
        if stats.os_pressure_events.load(Ordering::Relaxed) >= before + 3 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    manual.set(PressureLevel::Normal);
    assert!(stats.os_pressure_events.load(Ordering::Relaxed) >= before + 3, "relief rounds ran");
    assert_eq!(running.join().unwrap().unwrap().tokens_generated, 400);
    d.append(session, None, (500..520).collect()).unwrap();
    assert_eq!(d.generate(session, 8).unwrap().warm_prefix, 0, "the first turn's export went with the pressure");
}

fn mock_daemon_with_record(record: &str, park: &std::path::Path) -> Daemon {
    let dir = scratch();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let mut cfg = EngineConfig::default();
    cfg.spaces.retain(|s| s.kind == superfluid_abi::space_kind::PAGED_TOKEN_KV);
    let record = record.to_string();
    let host = EngineHost::spawn(move || (MockEngine::new(cfg).with_capability_descriptor(record), None)).expect("spawn");
    Daemon::with_options(
        store,
        host,
        Box::new(MockCodec),
        DaemonOptions { max_lanes: 2, park_dir: Some(park.to_path_buf()), ..Default::default() },
    )
    .unwrap()
}

fn files_in(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir).map(|rd| rd.flatten().count()).unwrap_or(0)
}

#[test]
fn a_runtime_that_cannot_export_a_session_is_not_asked_to_park_one() {
    let run = |record: &str| -> usize {
        let park = scratch().join("park");
        let d = mock_daemon_with_record(record, &park);
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..96).collect()).unwrap();
        assert_eq!(d.generate(s, 16).unwrap().tokens_generated, 16);
        d.shutdown();
        files_in(&park)
    };
    assert!(run(r#"{"descriptor_version":1,"serving":{"park_lossless":true}}"#) > 0, "a runtime that exports parks");
    assert_eq!(run(r#"{"descriptor_version":1,"serving":{"park_lossless":"no state export"}}"#), 0);
}

fn worker_reading(dir: &std::path::Path, file: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let script = dir.join("worker.sh");
    let mut f = std::fs::File::create(&script).unwrap();
    writeln!(
        f,
        "#!/bin/sh\nexec '{}' \"$@\" --mock-record \"$(cat '{}')\"",
        env!("CARGO_BIN_EXE_superfluid-workerd"),
        file.display()
    )
    .unwrap();
    drop(f);
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    script
}

const RECORD_V1: &str = r#"{"descriptor_version":1,"runtime":{"id":"mock","version":"1"},"limits":{"max_seq_len":4096}}"#;
const RECORD_V1_RESIZED: &str = r#"{"descriptor_version":1,"runtime":{"id":"mock","version":"1"},"limits":{"max_seq_len":2048}}"#;
const RECORD_V2: &str = r#"{"descriptor_version":1,"runtime":{"id":"mock","version":"2"},"limits":{"max_seq_len":4096}}"#;

fn process_daemon(dir: &std::path::Path, record: &std::path::Path) -> (Daemon, u32) {
    let host = EngineHost::spawn_process(superfluid_daemon::WorkerSpec {
        bin: worker_reading(dir, record),
        args: vec!["--engine".into(), "mock".into()],
        stderr: None,
    })
    .expect("spawn worker process");
    let pid = host.worker_pid().expect("a process has a pid");
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    (Daemon::new(store, host, Box::new(MockCodec), 4), pid)
}

fn kill(pid: u32) {
    let _ = std::process::Command::new("kill").args(["-9", &pid.to_string()]).status();
    std::thread::sleep(std::time::Duration::from_millis(50));
}

fn generate_after_kill(d: &Daemon) -> Result<u32, DaemonError> {
    let mut last = Err(DaemonError::Config("never ran"));
    for _ in 0..3 {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..24).collect()).unwrap();
        last = d.generate(s, 4).map(|o| o.tokens_generated);
        if last.is_ok() || d.needs_reload().is_some() {
            break;
        }
    }
    last
}

#[test]
fn a_respawn_that_only_resizes_keeps_serving() {
    let dir = scratch();
    let record = dir.join("record.json");
    std::fs::write(&record, RECORD_V1).unwrap();
    let (d, pid) = process_daemon(&dir, &record);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..24).collect()).unwrap();
    assert_eq!(d.generate(s, 4).unwrap().tokens_generated, 4);

    std::fs::write(&record, RECORD_V1_RESIZED).unwrap();
    kill(pid);
    assert_eq!(generate_after_kill(&d).unwrap(), 4);
    assert_eq!(d.needs_reload(), None);
    assert_eq!(d.capabilities().limit("max_seq_len"), Some(2048), "the respawned worker's limits hold");
}

fn long_prompt(d: &Daemon) -> Result<u32, DaemonError> {
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..3000).collect()).unwrap();
    d.generate(s, 4).map(|o| o.tokens_generated)
}

#[test]
fn a_respawn_that_holds_more_raises_the_admission_ceiling() {
    let dir = scratch();
    let record = dir.join("record.json");
    std::fs::write(&record, RECORD_V1_RESIZED).unwrap();
    let (d, pid) = process_daemon(&dir, &record);
    assert_eq!(d.max_stream_tokens(), 2048);
    match long_prompt(&d) {
        Err(DaemonError::StreamTooLong { len, max }) => assert_eq!((len, max), (3000, 2048)),
        other => panic!("expected the context error, got {other:?}"),
    }

    std::fs::write(&record, RECORD_V1).unwrap();
    kill(pid);
    assert_eq!(generate_after_kill(&d).unwrap(), 4);
    assert_eq!(d.capabilities().limit("max_seq_len"), Some(4096));
    assert_eq!(d.max_stream_tokens(), 4096, "the respawned worker's limit is the ceiling");
    assert_eq!(long_prompt(&d).unwrap(), 4, "a prompt the runtime now holds is admitted");
}

#[test]
fn a_respawn_that_holds_less_lowers_the_admission_ceiling() {
    let dir = scratch();
    let record = dir.join("record.json");
    std::fs::write(&record, RECORD_V1).unwrap();
    let (d, pid) = process_daemon(&dir, &record);
    assert_eq!(d.max_stream_tokens(), 4096);
    assert_eq!(long_prompt(&d).unwrap(), 4);

    std::fs::write(&record, RECORD_V1_RESIZED).unwrap();
    kill(pid);
    assert_eq!(generate_after_kill(&d).unwrap(), 4);
    assert_eq!(d.capabilities().limit("max_seq_len"), Some(2048));
    match long_prompt(&d) {
        Err(DaemonError::StreamTooLong { len, max }) => assert_eq!((len, max), (3000, 2048)),
        other => panic!("expected the context error, got {other:?}"),
    }
    assert_eq!(d.max_stream_tokens(), 2048, "the respawned worker's limit is the ceiling");
}

#[test]
fn a_respawned_worker_that_describes_another_runtime_retires_the_model() {
    let dir = scratch();
    let record = dir.join("record.json");
    std::fs::write(&record, RECORD_V1).unwrap();
    let (d, pid) = process_daemon(&dir, &record);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..24).collect()).unwrap();
    assert_eq!(d.generate(s, 4).unwrap().tokens_generated, 4);

    std::fs::write(&record, RECORD_V2).unwrap();
    kill(pid);
    let e = generate_after_kill(&d).unwrap_err().to_string();
    assert!(e.contains("the respawned worker's capability record differs (runtime.version)"), "{e}");
    let why = d.needs_reload().expect("retired");
    assert!(why.contains("runtime.version") && why.contains("loads again on its next request"), "{why}");
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..24).collect()).unwrap();
    assert!(d.generate(s, 4).unwrap_err().to_string().contains("runtime.version"));
}

fn reloader(
    dir: &std::path::Path,
    record: &std::path::Path,
    loads: &Arc<AtomicUsize>,
    worker: &Arc<AtomicU32>,
) -> superfluid_daemon::registry::ModelReloader {
    let (dir, record, loads, worker) = (dir.to_path_buf(), record.to_path_buf(), Arc::clone(loads), Arc::clone(worker));
    Box::new(move |source: &str, _runtime, retired: &Daemon| {
        assert_eq!(source, "/models/m.gguf", "loaded from where it came from");
        loads.fetch_add(1, Ordering::Relaxed);
        retired.reload_with(|| {
            let host = EngineHost::spawn_process(superfluid_daemon::WorkerSpec {
                bin: worker_reading(&dir, &record),
                args: vec!["--engine".into(), "mock".into()],
                stderr: None,
            })?;
            worker.store(host.worker_pid().expect("a process has a pid"), Ordering::Relaxed);
            Ok((host, Box::new(MockCodec) as Box<dyn superfluid_daemon::TextCodec + Send + Sync>))
        })
    })
}

struct Served {
    dir: std::path::PathBuf,
    record: std::path::PathBuf,
    held: Arc<Daemon>,
    registry: Arc<ModelRegistry>,
    pid: u32,
    loads: Arc<AtomicUsize>,
    worker: Arc<AtomicU32>,
}

fn served() -> Served {
    let dir = scratch();
    let record = dir.join("record.json");
    std::fs::write(&record, RECORD_V1).unwrap();
    let (d, pid) = process_daemon(&dir, &record);
    let held = Arc::new(d);
    let loader: superfluid_daemon::registry::ModelLoader = {
        let (dir, record) = (dir.clone(), record.clone());
        Box::new(move |_source: &str, _runtime| {
            let own = dir.join("models").join("m");
            std::fs::create_dir_all(&own).unwrap();
            Ok(Arc::new(process_daemon(&own, &record).0))
        })
    };
    let registry = Arc::new(ModelRegistry::with_initial("m", Arc::clone(&held), loader));
    registry.remember_source("m", "/models/m.gguf");
    let (loads, worker) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicU32::new(0)));
    registry.set_reloader(reloader(&dir, &record, &loads, &worker));
    Served { dir, record, held, registry, pid, loads, worker }
}

impl Served {
    fn retire(&self) {
        std::fs::write(&self.record, RECORD_V2).unwrap();
        kill(self.pid);
        assert!(generate_after_kill(&self.held).is_err());
        assert!(self.held.needs_reload().is_some());
    }

    fn reload(&self) -> Arc<Daemon> {
        self.retire();
        let (_, fresh) = self.registry.resolve(Some("m")).expect("the model resolves");
        assert_eq!(fresh.needs_reload(), None);
        fresh
    }
}

fn signal(pid: u32, sig: &str) {
    let _ = std::process::Command::new("kill").args([&format!("-{sig}"), &pid.to_string()]).status();
}

fn tokens_of(d: &Daemon, session: u64) -> Vec<u32> {
    d.store().lock().unwrap().session(session).expect("the session is in this store").tokens.clone()
}

#[test]
fn the_registry_loads_a_retired_model_again() {
    let m = served();
    let fresh = m.reload();
    assert!(Arc::ptr_eq(&fresh, &m.held), "the same daemon, serving again");
    assert_eq!(m.loads.load(Ordering::Relaxed), 1);
    let s = fresh.create(None, GenParams::default()).unwrap();
    fresh.append(s, None, (0..24).collect()).unwrap();
    assert_eq!(fresh.generate(s, 4).unwrap().tokens_generated, 4);
    let (_, again) = m.registry.resolve(Some("m")).unwrap();
    assert!(Arc::ptr_eq(&again, &fresh));
    assert_eq!(m.loads.load(Ordering::Relaxed), 1);

    let dir = scratch();
    let record = dir.join("record.json");
    std::fs::write(&record, RECORD_V1).unwrap();
    let (d, pid) = process_daemon(&dir, &record);
    let d = Arc::new(d);
    let registry =
        ModelRegistry::with_initial("m", Arc::clone(&d), Box::new(|_: &str, _| Err(DaemonError::Config("no loads"))));
    registry.remember_source("m", "/models/m.gguf");
    std::fs::write(&record, RECORD_V2).unwrap();
    kill(pid);
    assert!(generate_after_kill(&d).is_err());
    let (_, still) = registry.resolve(Some("m")).unwrap();
    assert!(still.needs_reload().is_some_and(|why| why.contains("runtime.version")));
}

#[test]
fn a_model_that_loads_again_keeps_its_sessions_through_the_reload_and_a_restart() {
    let m = served();
    let d = Arc::clone(&m.held);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..24).collect()).unwrap();
    assert_eq!(d.generate(s, 4).unwrap().tokens_generated, 4);
    let before = tokens_of(&d, s);
    assert_eq!(before.len(), 28);

    let fresh = m.reload();

    assert_eq!(tokens_of(&fresh, s), before, "the history carried over");
    fresh.append(s, None, (100..108).collect()).unwrap();
    assert_eq!(fresh.generate(s, 4).unwrap().tokens_generated, 4);
    let after = tokens_of(&fresh, s);
    assert_eq!(after.len(), before.len() + 8 + 4);
    assert_eq!(after[..before.len()], before[..]);
    let later = fresh.create(None, GenParams::default()).unwrap();
    fresh.append(later, None, (0..16).collect()).unwrap();
    assert_eq!(fresh.generate(later, 4).unwrap().tokens_generated, 4);
    let later_tokens = tokens_of(&fresh, later);
    assert_eq!(later_tokens.len(), 20);
    assert!(!m.dir.join("models").exists(), "no second store");

    let wal = m.dir.join("wal.log");
    drop(fresh);
    drop(d);
    let Served { held, registry, .. } = m;
    drop(registry);
    held.shutdown();
    drop(held);
    let reopened = SessionStore::open(&wal).unwrap();
    assert_eq!(reopened.session(s).unwrap().tokens, after);
    assert_eq!(reopened.session(later).unwrap().tokens, later_tokens);
}

#[test]
fn a_session_generating_after_a_reload_is_busy_and_cancellable_through_every_handle() {
    let m = served();
    let fresh = m.reload();
    let worker = m.worker.load(Ordering::Relaxed);
    let s = fresh.create(None, GenParams::default()).unwrap();
    fresh.append(s, None, (0..24).collect()).unwrap();

    signal(worker, "STOP");
    let generating = {
        let fresh = Arc::clone(&fresh);
        std::thread::spawn(move || fresh.generate(s, 2000))
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !fresh.active_registry().lock().unwrap().contains(&s) && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let in_flight = fresh.active_registry().lock().unwrap().contains(&s);
    let appended = m.held.append(s, None, vec![77_777]);
    let cancelled = {
        let cancels = m.held.cancel_registry();
        let mut c = cancels.lock().unwrap();
        let found = m.held.active_registry().lock().unwrap().contains(&s);
        if found {
            c.insert(s);
        }
        found
    };
    signal(worker, "CONT");
    let out = generating.join().unwrap().expect("the generation ends");

    assert!(in_flight, "the generation was in flight");
    assert!(matches!(appended, Err(DaemonError::SessionBusy(id)) if id == s), "an append while it generates: {appended:?}");
    assert!(!tokens_of(&fresh, s).contains(&77_777), "the log holds no input the generation never saw");
    assert!(cancelled, "a cancel through the held handle finds the generation in flight");
    assert_eq!(out.finish, superfluid_abi::finish::CANCELLED, "and stops it ({} tokens)", out.tokens_generated);
    m.held.append(s, None, vec![1, 2, 3]).expect("no longer busy");
}

#[test]
fn a_handle_held_since_before_the_reload_serves_after_it() {
    let m = served();
    m.reload();
    let s = m.held.create(None, GenParams::default()).unwrap();
    m.held.append(s, None, (0..24).collect()).unwrap();
    assert_eq!(m.held.generate(s, 4).expect("the held handle generates").tokens_generated, 4);
    assert_eq!(m.held.needs_reload(), None);
}

#[test]
fn a_native_request_loads_a_retired_model_again() {
    use superfluid_daemon::api::{NativeClient, Request, Response};
    let m = served();
    {
        let (registry, held) = (Arc::downgrade(&m.registry), Arc::clone(&m.held));
        held.on_retired(Box::new(move || {
            if let Some(registry) = registry.upgrade() {
                let _ = registry.resolve(Some("m"));
            }
        }));
    }
    let socket = std::env::temp_dir().join(format!("brd-{}-{}.sock", std::process::id(), DIRS.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_file(&socket);
    let listener = std::os::unix::net::UnixListener::bind(&socket).expect("bind the native socket");
    {
        let held = Arc::clone(&m.held);
        std::thread::spawn(move || superfluid_daemon::api::serve(listener, held));
    }
    let mut client = NativeClient::connect(&socket).expect("connect");
    let generate = |client: &mut NativeClient| -> Result<u32, String> {
        let session = match client.request(&Request::Create { parent: None, params: GenParams::default() }).unwrap() {
            Response::Created { session } => session,
            other => panic!("create: {other:?}"),
        };
        client.request(&Request::Append { session, text: None, span: (0..24).collect() }).unwrap();
        match client.generate_stream(session, 4, |_| {}).unwrap() {
            Response::Generated { tokens_generated, .. } => Ok(tokens_generated),
            Response::Err { message } => Err(message),
            other => panic!("generate: {other:?}"),
        }
    };
    assert_eq!(generate(&mut client), Ok(4));

    std::fs::write(&m.record, RECORD_V2).unwrap();
    kill(m.pid);
    let mut failures = Vec::new();
    let mut served_again = None;
    for _ in 0..4 {
        match generate(&mut client) {
            Ok(n) => {
                served_again = Some(n);
                break;
            }
            Err(why) => failures.push(why),
        }
    }
    assert_eq!(served_again, Some(4), "the same connection generates again: {failures:?}");
    assert!(!failures.is_empty(), "the request that finds the worker gone fails");
    assert!(failures[1..].iter().all(|why| why.contains("runtime.version")), "{failures:?}");
    assert_eq!(m.loads.load(Ordering::Relaxed), 1, "loaded again once, by the native request");
    assert_eq!(m.held.needs_reload(), None);
    let _ = std::fs::remove_file(&socket);
}

#[test]
fn a_worker_that_will_not_come_back_retires_the_model_so_a_request_loads_it_again() {
    let dir = scratch();
    let record = dir.join("record.json");
    std::fs::write(&record, RECORD_V1).unwrap();
    let (d, pid) = process_daemon(&dir, &record);
    let d = Arc::new(d);
    let request = |d: &Daemon| {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..24).collect()).unwrap();
        d.generate(s, 4).map(|o| o.tokens_generated)
    };
    assert_eq!(request(&d).unwrap(), 4);
    {
        let (held, dir, record) = (Arc::downgrade(&d), dir.clone(), record.clone());
        d.on_retired(Box::new(move || {
            let Some(d) = held.upgrade() else { return };
            let _ = d.reload_with(|| {
                let host = EngineHost::spawn_process(superfluid_daemon::WorkerSpec {
                    bin: worker_reading(&dir, &record),
                    args: vec!["--engine".into(), "mock".into()],
                    stderr: None,
                })?;
                Ok((host, Box::new(MockCodec) as Box<dyn superfluid_daemon::TextCodec + Send + Sync>))
            });
        }));
    }

    std::fs::remove_file(dir.join("worker.sh")).unwrap();
    kill(pid);
    let mut failures = Vec::new();
    let mut served_again = None;
    for _ in 0..8 {
        match request(&d) {
            Ok(n) => {
                served_again = Some(n);
                break;
            }
            Err(e) => failures.push(e.to_string()),
        }
    }
    assert_eq!(served_again, Some(4), "a request after the respawns ran out loads the model again: {failures:?}");
    assert!(
        failures.last().is_some_and(|why| why.contains("respawns in a row") && why.contains("so retry")),
        "the request that finds the respawns spent says why: {failures:?}"
    );
    assert_eq!(d.needs_reload(), None);
}

#[test]
fn a_reload_that_fails_leaves_the_model_retired_until_one_succeeds() {
    let dir = scratch();
    let record = dir.join("record.json");
    std::fs::write(&record, RECORD_V1).unwrap();
    let (d, _pid) = process_daemon(&dir, &record);
    let request = |d: &Daemon| {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..24).collect()).unwrap();
        d.generate(s, 4).map(|o| o.tokens_generated)
    };
    assert_eq!(request(&d).unwrap(), 4);

    let failed = d.reload_with(|| Err(DaemonError::Config("no engine to be had")));
    assert!(matches!(failed, Err(DaemonError::Config("no engine to be had"))), "{failed:?}");
    let why = d.needs_reload().expect("retired until a load succeeds");
    assert!(why.contains("no engine to be had") && why.contains("so retry"), "{why}");
    let e = request(&d).unwrap_err().to_string();
    assert!(e.contains("no engine to be had"), "the request says why: {e}");

    d.reload_with(|| {
        let host = EngineHost::spawn_process(superfluid_daemon::WorkerSpec {
            bin: worker_reading(&dir, &record),
            args: vec!["--engine".into(), "mock".into()],
            stderr: None,
        })?;
        Ok((host, Box::new(MockCodec) as Box<dyn superfluid_daemon::TextCodec + Send + Sync>))
    })
    .expect("the model loads again");
    assert_eq!(d.needs_reload(), None);
    assert_eq!(request(&d).unwrap(), 4);
}
