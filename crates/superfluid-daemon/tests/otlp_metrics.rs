//! Operational OTLP metrics push.

use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use superfluid_daemon::export::{otlp_metrics_json, MetricsPusher};
use superfluid_daemon::scheduler::SchedStats;

#[test]
fn otlp_metrics_json_is_well_formed() {
    let s = SchedStats::default();
    s.ticks.store(42, Ordering::Relaxed);
    s.decode_tokens.store(1000, Ordering::Relaxed);
    s.lanes_active.store(3, Ordering::Relaxed);
    s.parks_lossy.store(2, Ordering::Relaxed);
    s.queue_depth[0].store(4, Ordering::Relaxed);

    let body = otlp_metrics_json(&s, 1_000_000_000, 500_000_000);
    let v: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");

    let metrics = v["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        .as_array()
        .expect("metrics array");
    let by_name = |n: &str| metrics.iter().find(|m| m["name"] == n).unwrap();

    let ticks = by_name("superfluid_ticks_total");
    assert_eq!(ticks["sum"]["dataPoints"][0]["asInt"], "42");
    assert_eq!(ticks["sum"]["isMonotonic"], true);
    assert_eq!(ticks["sum"]["aggregationTemporality"], 2);

    assert_eq!(by_name("superfluid_lanes_active")["gauge"]["dataPoints"][0]["asInt"], "3");

    let parks = by_name("superfluid_parks_total");
    let lossy = parks["sum"]["dataPoints"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["attributes"][0]["value"]["stringValue"] == "lossy")
        .unwrap();
    assert_eq!(lossy["asInt"], "2");

    let qd = metrics
        .iter()
        .filter(|m| m["name"] == "superfluid_queue_depth")
        .flat_map(|m| m["gauge"]["dataPoints"].as_array().unwrap())
        .find(|p| p["attributes"][0]["value"]["stringValue"] == "interactive_chat")
        .unwrap();
    assert_eq!(qd["asInt"], "4");

    let h = &by_name("superfluid_ttft_seconds")["histogram"]["dataPoints"][0];
    let bounds = h["explicitBounds"].as_array().unwrap().len();
    let buckets = h["bucketCounts"].as_array().unwrap().len();
    assert_eq!(buckets, bounds + 1, "OTLP histogram: bucketCounts == bounds + 1");

    assert_eq!(
        v["resourceMetrics"][0]["resource"]["attributes"][0]["value"]["stringValue"],
        "superfluid"
    );
}

fn mock_collector(captured: Arc<Mutex<Vec<String>>>) -> std::net::SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).is_err() {
                continue;
            }
            let mut clen = 0usize;
            loop {
                let mut h = String::new();
                if reader.read_line(&mut h).unwrap_or(0) == 0 {
                    break;
                }
                if h == "\r\n" {
                    break;
                }
                if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                    clen = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; clen];
            let _ = reader.read_exact(&mut body);
            captured
                .lock()
                .unwrap()
                .push(format!("{}|{}", request_line.trim_end(), String::from_utf8_lossy(&body)));
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}");
            break;
        }
    });
    addr
}

#[test]
fn metrics_pusher_posts_otlp_to_collector() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let addr = mock_collector(Arc::clone(&captured));

    let s = Arc::new(SchedStats::default());
    s.ticks.store(7, Ordering::Relaxed);
    let pusher = MetricsPusher::start(
        format!("http://{addr}"),
        Duration::from_millis(20),
        Arc::clone(&s),
    );

    let mut got = None;
    for _ in 0..150 {
        if let Some(c) = captured.lock().unwrap().first().cloned() {
            got = Some(c);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(pusher);

    let got = got.expect("collector received a metrics push");
    assert!(got.starts_with("POST /v1/metrics "), "wrong request line: {got}");
    let body = got.split_once('|').unwrap().1;
    let v: serde_json::Value = serde_json::from_str(body).expect("pushed body is valid JSON");
    let names: Vec<String> = v["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(names.iter().any(|n| n == "superfluid_ticks_total"), "metrics missing: {names:?}");
}
