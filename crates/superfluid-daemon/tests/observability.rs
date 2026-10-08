use std::path::PathBuf;
use std::sync::Arc;

use superfluid_daemon::export::{self, ExportOptions};
use superfluid_daemon::{api, Daemon, EngineHost, GenParams, MockCodec, SessionStore, Tier};
use superfluid_engine::{EngineConfig, MockEngine};

fn test_dir() -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("superfluid-obs-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn kv_only() -> EngineConfig {
    let mut cfg = EngineConfig::default();
    cfg.spaces
        .retain(|s| s.kind == superfluid_abi::space_kind::PAGED_TOKEN_KV);
    cfg
}

#[test]
fn inspect_reports_state_and_residency_tier() {
    let dir = test_dir();
    let park = dir.join("park");
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(kv_only()), None)).unwrap();
    let d = Arc::new(Daemon::with_park(store, host, Box::new(MockCodec), 4, Some(park.clone())));
    let s = d.create(None, GenParams { seed: 3, ..Default::default() }).unwrap();
    d.append(s, None, (0..32).collect()).unwrap();
    d.set_meta(s, 0, Some("probe".into()), None).unwrap();
    let cold = d.inspect(s).unwrap();
    assert_eq!(cold.tier, Tier::Cold);
    assert_eq!(cold.summary.title.as_deref(), Some("probe"));
    assert_eq!(cold.params.seed, 3);
    assert_eq!(cold.tokens.len(), 32);
    assert_eq!(cold.events.len(), 3);
    assert_eq!(cold.wal_version, 2);
    let dd = Arc::clone(&d);
    let t = std::thread::spawn(move || dd.generate(s, 3000));
    for _ in 0..500 {
        if d.active_registry().lock().unwrap().contains(&s) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(d.inspect(s).unwrap().tier, Tier::Resident);
    t.join().unwrap().unwrap();
    let mut tier = Tier::Cold;
    for _ in 0..200 {
        tier = d.inspect(s).unwrap().tier;
        if matches!(tier, Tier::Parked { .. }) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(matches!(tier, Tier::Parked { covered, .. } if covered >= 16), "{tier:?}");
    let socket = dir.join("sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let serving = Arc::clone(&d);
    std::thread::spawn(move || {
        let _ = api::serve(listener, serving);
    });
    let mut c = api::NativeClient::connect(&socket).unwrap();
    match c.request(&api::Request::Inspect { session: s }).unwrap() {
        api::Response::Inspection { inspection } => {
            assert_eq!(inspection.summary.id, s);
            assert!(inspection.behavior_fingerprint.is_some() || inspection.events.len() > 3);
        }
        other => panic!("unexpected {other:?}"),
    }
    match c.request(&api::Request::Metrics).unwrap() {
        api::Response::Text { text } => assert!(text.contains("superfluid_ticks_total")),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn export_matches_goldens() {
    let dir = test_dir();
    let src = fixtures().join("wal-p2-forked.wal");
    let dst = dir.join("wal-p2-forked.wal");
    std::fs::copy(&src, &dst).unwrap();
    std::fs::copy(
        fixtures().join("wal-p2-forked.wal.store-id"),
        dir.join("wal-p2-forked.wal.store-id"),
    )
    .unwrap();
    let store = SessionStore::open(&dst).unwrap();
    assert_eq!(store.store_id().to_hex(), "b0a5e7d15ea5ed0c0ffee00000000001");
    let events: Vec<api::EventMsg> = store
        .session(1)
        .unwrap()
        .events
        .iter()
        .filter(|e| !matches!(e.body, superfluid_daemon::EventBody::EpochBump))
        .cloned()
        .map(Into::into)
        .collect();
    let opts = ExportOptions {
        model: "fixture-model".into(),
        store_ns: export::store_trace_ns(&store.store_id()),
        ..Default::default()
    };
    let jsonl = export::jsonl(1, &events, &opts);
    let otlp = export::otlp_jsonl(1, &events, &opts);
    let golden_jsonl = fixtures().join("export-forked.jsonl");
    let golden_otlp = fixtures().join("export-forked.traces.otlp.jsonl");
    if std::env::var_os("SUPERFLUID_WRITE_GOLDENS").is_some() {
        std::fs::write(&golden_jsonl, &jsonl).unwrap();
        std::fs::write(&golden_otlp, &otlp).unwrap();
    }
    assert_eq!(jsonl, std::fs::read_to_string(&golden_jsonl).expect("golden jsonl present"));
    assert_eq!(otlp, std::fs::read_to_string(&golden_otlp).expect("golden otlp present"));
    assert!(!jsonl.contains("abc"), "content must not leak without opt-in");
    assert!(jsonl.contains("\"bytes\":3"));
    assert!(otlp.contains(&format!("\"stringValue\":\"{}\"", export::GENAI_SEMCONV_VERSION)));
    assert!(otlp.contains("\"superfluid.timing\""));
    let with = export::jsonl(
        1,
        &events,
        &ExportOptions {
            include_content: true,
            ..opts
        },
    );
    assert!(with.contains("\"text\":\"abc\""));
}

#[test]
fn redaction_replaces_secret_shapes_and_caps() {
    let t = "key sk-abcdefghijklmnopqrstuvwxyz0123 and AKIAABCDEFGHIJKLMNOP then ghp_0123456789abcdefghijklmnopqrstuvwxyz ok";
    let r = export::redact(t, 10_000);
    assert!(!r.contains("sk-abcdefghijklmnopqrstuvwxyz0123"));
    assert!(!r.contains("AKIAABCDEFGHIJKLMNOP"));
    assert!(!r.contains("ghp_0123456789abcdefghijklmnopqrstuvwxyz"));
    assert_eq!(r.matches("[REDACTED]").count(), 3);
    assert!(r.starts_with("key ") && r.ends_with(" ok"));
    let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----";
    let r = export::redact(pem, 10_000);
    assert!(!r.contains("MIIE"));
    assert!(r.contains("[REDACTED]"));
    let long = "é".repeat(100);
    let r = export::redact(&long, 11);
    assert!(r.ends_with("…[truncated]"));
    assert!(r.starts_with("ééééé"));
    assert!(std::str::from_utf8(r.as_bytes()).is_ok());
}

#[test]
fn metrics_render_and_advance() {
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(kv_only()), None)).unwrap();
    let d = Daemon::new(store, host, Box::new(MockCodec), 4);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..32).collect()).unwrap();
    d.generate(s, 40).unwrap();
    d.generate(s, 8).unwrap();
    let text = d.metrics_text();
    for name in [
        "superfluid_ticks_total",
        "superfluid_prefill_tokens_total",
        "superfluid_decode_tokens_total",
        "superfluid_warm_prefix_tokens_total",
        "superfluid_cold_admissions_total",
        "superfluid_preemptions_total",
        "superfluid_starvation_grants_total",
        "superfluid_parks_total{encoding=\"lossless\"}",
        "superfluid_resumes_total",
        "superfluid_worker_respawns_total",
        "superfluid_lanes_active",
        "superfluid_queue_depth{class=\"interactive_chat\"}",
        "superfluid_ttft_seconds_bucket{le=\"+Inf\"}",
        "superfluid_ttft_seconds_count",
        "superfluid_spec_accepted_total",
        "superfluid_pins_held",
        "superfluid_pinned_bytes",
        "superfluid_pins_yielded_total",
        "superfluid_pins_expired_total",
        "superfluid_telemetry_dropped_total",
    ] {
        assert!(text.contains(name), "missing {name}\n{text}");
    }
    let value = |name: &str| -> u64 {
        text.lines()
            .find(|l| l.starts_with(name) && !l.starts_with('#'))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse::<f64>().ok())
            .map(|v| v as u64)
            .unwrap_or_else(|| panic!("no {name}"))
    };
    assert!(value("superfluid_ticks_total ") >= 3);
    assert_eq!(value("superfluid_decode_tokens_total "), 48);
    assert_eq!(value("superfluid_prefill_tokens_total "), 32 + (72 - 64));
    assert_eq!(value("superfluid_warm_prefix_tokens_total "), 64);
    assert_eq!(value("superfluid_cold_admissions_total "), 1);
    assert_eq!(value("superfluid_ttft_seconds_count "), 2);
    assert_eq!(value("superfluid_ttft_seconds_bucket{le=\"+Inf\"} "), 2);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let dd = Arc::new(d);
    std::thread::spawn(move || {
        let _ = superfluid_daemon::openai::serve_blocking(listener, dd, "m".into());
    });
    let mut body = String::new();
    for _ in 0..50 {
        if let Ok(mut s) = std::net::TcpStream::connect(addr) {
            use std::io::{Read, Write};
            s.write_all(b"GET /metrics HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").unwrap();
            s.read_to_string(&mut body).unwrap();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(body.contains("200 OK"), "{body}");
    assert!(body.contains("text/plain; version=0.0.4"));
    assert!(body.contains("superfluid_ticks_total"));
}

#[test]
fn telemetry_sink_writes_rotates_and_reloads() {
    use superfluid_daemon::telemetry::{self, LogConfig};
    let dir = test_dir().join("logs");
    telemetry::init(&LogConfig {
        dir: dir.clone(),
        filter: "info".into(),
        max_file_bytes: 16 * 1024,
        max_files: 3,
        queue: 4096,
    })
    .expect("init once per process");
    tracing::info!(session = 7u64, "hello");
    tracing::debug!("filtered out at info");
    for i in 0..300u32 {
        tracing::info!(i, "a line long enough to make the tiny cap rotate a few times over");
    }
    telemetry::set_filter("observability=debug,info").unwrap();
    tracing::debug!(marker = "after-reload", "now visible");
    assert!(telemetry::set_filter("this is not a directive[").is_err());
    std::thread::sleep(std::time::Duration::from_millis(200));
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .collect();
    files.sort();
    assert!(!files.is_empty() && files.len() <= 3, "rotated and capped: {files:?}");
    let all: String = files
        .iter()
        .map(|p| std::fs::read_to_string(p).unwrap())
        .collect();
    assert!(all.lines().all(|l| l.starts_with('{')), "JSON lines");
    assert!(all.contains("after-reload"), "the reloaded filter admits debug");
    assert!(!all.contains("filtered out at info"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for f in &files {
            assert_eq!(std::fs::metadata(f).unwrap().permissions().mode() & 0o777, 0o600);
        }
    }
    let _ = telemetry::dropped_total();
}

#[test]
fn otlp_push_posts_json_to_v1_traces() {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut buf = vec![0u8; 65536];
        let mut got = Vec::new();
        loop {
            let n = s.read(&mut buf).unwrap();
            got.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&got).to_string();
            if let Some(i) = text.find("\r\n\r\n") {
                let head = &text[..i];
                let len: usize = head
                    .lines()
                    .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap()))
                    .unwrap_or(0);
                if got.len() >= i + 4 + len {
                    break;
                }
            }
            if n == 0 {
                break;
            }
        }
        s.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}").unwrap();
        String::from_utf8_lossy(&got).to_string()
    });
    let events = vec![api::EventMsg {
        event_id: 0,
        epoch: 1,
        ts_unix_ms: 1_700_000_000_000,
        body: superfluid_daemon::EventBody::Created {
            parent: None,
            params: GenParams::default(),
        },
    }];
    let status = export::push_otlp_traces(&format!("http://{addr}"), 5, &events, &ExportOptions::default()).unwrap();
    assert_eq!(status, 200);
    let req = server.join().unwrap();
    assert!(req.starts_with("POST /v1/traces HTTP/1.1"), "{req}");
    assert!(req.to_ascii_lowercase().contains("content-type: application/json"));
    assert!(req.contains("resourceSpans"));
}
