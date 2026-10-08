//! The OTLP exporter can never backpressure a tick.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use superfluid_daemon::otlp::{self, OtlpConfig};
use superfluid_daemon::telemetry::{self, LogConfig, LogSink};
use superfluid_daemon::{Daemon, EngineHost, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use tracing_subscriber::layer::SubscriberExt;

fn test_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("superfluid-otlp-bp-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn kv_only() -> EngineConfig {
    let mut cfg = EngineConfig::default();
    cfg.spaces
        .retain(|s| s.kind == superfluid_abi::space_kind::PAGED_TOKEN_KV);
    cfg
}

fn black_hole() -> String {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for c in l.incoming() {
            held.push(c);
        }
    });
    format!("http://{addr}")
}

#[test]
fn hung_collector_never_stalls_ticks_and_counts_drops() {
    let post_timeout = Duration::from_secs(20);
    telemetry::init_sinks(
        LogSink::File(LogConfig {
            dir: test_dir("logs"),
            filter: "info".into(),
            ..Default::default()
        }),
        None,
        Some(OtlpConfig {
            endpoint: black_hole(),
            queue: 8,
            batch_max: 4,
            interval: Duration::from_millis(10),
            timeout: post_timeout,
            ..Default::default()
        }),
    )
    .expect("init once per process");

    let dir = test_dir("daemon");
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(kv_only()), None)).unwrap();
    let d = Daemon::new(store, host, Box::new(MockCodec), 4);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..32).collect()).unwrap();

    let t0 = Instant::now();
    for _ in 0..4 {
        d.generate(s, 64).unwrap();
    }
    for i in 0..5000u64 {
        tracing::info_span!("flood", i).in_scope(|| {});
    }
    let took = t0.elapsed();
    assert!(
        took < post_timeout / 4,
        "the tick waited on the exporter: {took:?}"
    );

    let queue_full = otlp::DROPPED_QUEUE_FULL.load(Ordering::Relaxed);
    assert!(
        queue_full > 4000,
        "overflow is dropped and counted: {queue_full}"
    );
    assert!(
        telemetry::dropped_total() >= queue_full,
        "rolled into telemetry_dropped_total"
    );
    let text = d.metrics_text();
    let line = text
        .lines()
        .find(|l| l.starts_with("superfluid_otlp_spans_dropped_total{reason=\"queue_full\"}"))
        .expect("labelled drop counter");
    assert!(line.rsplit(' ').next().unwrap().parse::<u64>().unwrap() >= queue_full);
    assert_eq!(otlp::EXPORTED.load(Ordering::Relaxed), 0);

    let t1 = Instant::now();
    assert!(!otlp::shutdown(Duration::from_millis(300)));
    assert!(t1.elapsed() < Duration::from_secs(2), "{:?}", t1.elapsed());
}

#[test]
fn refused_collector_counts_failed_exports_and_flushes_promptly() {
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let before = otlp::DROPPED_EXPORT_FAILED.load(Ordering::Relaxed);
    let (layer, handle) = otlp::start(OtlpConfig {
        endpoint: format!("http://127.0.0.1:{port}"),
        batch_max: 8,
        interval: Duration::from_millis(10),
        timeout: Duration::from_secs(2),
        ..Default::default()
    })
    .unwrap();
    let sub = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(sub, || {
        for i in 0..20u64 {
            tracing::info_span!("x", i).in_scope(|| {});
        }
    });
    assert!(
        handle.shutdown(Duration::from_secs(10)),
        "a refusing collector fails fast"
    );
    let failed = otlp::DROPPED_EXPORT_FAILED.load(Ordering::Relaxed) - before;
    assert!(
        failed >= 20,
        "every span of a failed batch is counted: {failed}"
    );
}
