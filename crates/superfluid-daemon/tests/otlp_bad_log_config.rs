//! A broken LOCAL log configuration never takes the OTLP export down with it.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use superfluid_daemon::otlp::{self, OtlpConfig};
use superfluid_daemon::telemetry::{self, LogSink};

fn collector(posts: Arc<AtomicU64>) -> String {
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
            if String::from_utf8_lossy(&got).starts_with("POST /v1/traces ") {
                posts.fetch_add(1, Ordering::Relaxed);
            }
            let _ =
                c.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}");
        }
    });
    format!("http://{addr}")
}

#[test]
fn bad_log_filter_keeps_the_otlp_export() {
    let posts = Arc::new(AtomicU64::new(0));
    let endpoint = collector(Arc::clone(&posts));
    let err = telemetry::init_sinks(
        LogSink::Stderr("not a directive[".into()),
        None,
        Some(OtlpConfig {
            endpoint,
            interval: Duration::from_millis(20),
            ..Default::default()
        }),
    )
    .expect_err("the bad log filter is reported");
    assert!(err.contains("log filter"), "{err}");
    assert!(!err.contains("OTLP export disabled"), "{err}");
    for i in 0..5u64 {
        tracing::info_span!("still_traced", i).in_scope(|| {});
    }
    assert!(otlp::shutdown(Duration::from_secs(5)));
    assert!(posts.load(Ordering::Relaxed) >= 1, "spans still exported");
    assert!(otlp::EXPORTED.load(Ordering::Relaxed) >= 5);
    telemetry::set_filter("debug").expect("log sink installed");
}
