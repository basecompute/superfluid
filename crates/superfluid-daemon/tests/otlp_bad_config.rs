//! Export accounting edge cases.

use std::time::Duration;

use superfluid_daemon::otlp::{self, OtlpConfig};
use superfluid_daemon::telemetry::{self, LogConfig, LogSink};

#[test]
fn bad_otlp_filter_keeps_local_logging() {
    let dir = std::env::temp_dir().join(format!("superfluid-otlp-bad-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let err = telemetry::init_sinks(
        LogSink::File(LogConfig {
            dir: dir.clone(),
            filter: "info".into(),
            ..Default::default()
        }),
        None,
        Some(OtlpConfig {
            endpoint: "http://127.0.0.1:9".into(),
            filter: "not a directive[".into(),
            ..Default::default()
        }),
    )
    .expect_err("the bad filter is reported");
    assert!(err.contains("OTLP export disabled"), "{err}");
    assert!(err.contains("local logs unaffected"), "{err}");

    tracing::info!(marker = "still-logging", "local sink installed");
    telemetry::set_filter("debug").expect("runtime log levels still work");
    std::thread::sleep(Duration::from_millis(200));
    let all: String = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| std::fs::read_to_string(e.path()).unwrap())
        .collect();
    assert!(all.contains("still-logging"), "{all}");
}

#[test]
fn partial_success_splits_exported_and_rejected() {
    use std::io::{Read, Write};
    use std::sync::atomic::Ordering;
    use tracing_subscriber::layer::SubscriberExt;

    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    std::thread::spawn(move || {
        for c in l.incoming() {
            let Ok(mut c) = c else { continue };
            let mut got = Vec::new();
            let mut buf = [0u8; 8192];
            loop {
                let n = c.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    break;
                }
                got.extend_from_slice(&buf[..n]);
                let t = String::from_utf8_lossy(&got).to_string();
                if let Some(i) = t.find("\r\n\r\n") {
                    let len: usize = t[..i]
                        .lines()
                        .find_map(|h| {
                            h.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse().unwrap())
                        })
                        .unwrap_or(0);
                    if got.len() >= i + 4 + len {
                        break;
                    }
                }
            }
            let body = r#"{"partialSuccess":{"rejectedSpans":"2","errorMessage":"limit"}}"#;
            let _ = write!(
                c,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    let exported0 = otlp::EXPORTED.load(Ordering::Relaxed);
    let rejected0 = otlp::DROPPED_REJECTED.load(Ordering::Relaxed);
    let (layer, handle) = otlp::start(OtlpConfig {
        endpoint: format!("http://{addr}"),
        batch_max: 5,
        interval: Duration::from_secs(60),
        ..Default::default()
    })
    .unwrap();
    let sub = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(sub, || {
        for i in 0..5u64 {
            tracing::info_span!("x", i).in_scope(|| {});
        }
    });
    assert!(handle.shutdown(Duration::from_secs(10)));
    assert_eq!(otlp::EXPORTED.load(Ordering::Relaxed) - exported0, 3);
    assert_eq!(
        otlp::DROPPED_REJECTED.load(Ordering::Relaxed) - rejected0,
        2
    );
}
