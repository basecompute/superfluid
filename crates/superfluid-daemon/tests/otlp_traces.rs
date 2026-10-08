use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use superfluid_daemon::otlp::{self, OtlpConfig, TraceScope};
use superfluid_daemon::telemetry::{self, LogConfig, LogSink};
use superfluid_daemon::{
    export, Daemon, DaemonOptions, EngineHost, GenParams, MockCodec, SessionStore,
};
use superfluid_engine::{EngineConfig, MockEngine};
use serde_json::Value;

fn test_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("superfluid-otlp-{tag}-{}", std::process::id()));
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

type Captured = Arc<Mutex<Vec<(Option<String>, Option<String>, String)>>>;

fn collector() -> (String, Captured) {
    let got: Captured = Arc::new(Mutex::new(Vec::new()));
    let g = Arc::clone(&got);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let app = axum::Router::new().route(
                "/v1/traces",
                axum::routing::post(move |headers: axum::http::HeaderMap, body: String| {
                    let g = Arc::clone(&g);
                    async move {
                        let h = |k: &str| {
                            headers
                                .get(k)
                                .and_then(|v| v.to_str().ok())
                                .map(String::from)
                        };
                        g.lock()
                            .unwrap()
                            .push((h("x-test-token"), h("content-type"), body));
                        "{}"
                    }
                }),
            );
            let l = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(l, app).await.unwrap();
        });
    });
    (format!("http://{addr}"), got)
}

fn attr<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
    span["attributes"]
        .as_array()?
        .iter()
        .find(|a| a["key"] == key)
        .map(|a| &a["value"])
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit()) && s.bytes().any(|b| b != b'0')
}

#[test]
fn daemon_spans_reach_the_collector_as_valid_otlp() {
    let (endpoint, got) = collector();
    telemetry::init_sinks(
        LogSink::File(LogConfig {
            dir: test_dir("logs"),
            filter: "info".into(),
            ..Default::default()
        }),
        None,
        Some(OtlpConfig {
            endpoint: endpoint.clone(),
            headers: vec![("x-test-token".into(), "t0k".into())],
            resource: vec![("deployment.environment".into(), "test".into())],
            interval: Duration::from_millis(50),
            ..Default::default()
        }),
    )
    .expect("init once per process");

    let dir = test_dir("daemon");
    let daemon = |dir: &std::path::Path, scope: TraceScope| {
        let store = SessionStore::open(&dir.join("wal.log")).unwrap();
        let host = EngineHost::spawn(|| (MockEngine::new(kv_only()), None)).unwrap();
        Daemon::with_options(
            store,
            host,
            Box::new(MockCodec),
            DaemonOptions {
                max_lanes: 4,
                park_dir: Some(dir.join("park")),
                trace_scope: scope,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let d = daemon(&dir, TraceScope::model("mock-model"));
    let s = d.create(None, GenParams::default()).unwrap();
    assert_ne!(s, 0);
    d.append(s, None, (0..32).collect()).unwrap();
    d.generate(s, 40).unwrap();
    d.generate(s, 8).unwrap();
    drop(d);
    let d = daemon(&dir, TraceScope::model("mock-model"));
    let resumed =
        tracing::info_span!("embedder_request", model = "APPDATA-STR", request = ?"APPDATA-DBG")
            .in_scope(|| d.generate(s, 4).unwrap());
    assert!(
        resumed.warm_prefix > 0,
        "restart resumed from the park artifact"
    );

    let other_dir = test_dir("other");
    let other = daemon(&other_dir, TraceScope::model("other-model"));
    let s2 = other.create(None, GenParams::default()).unwrap();
    assert_eq!(s2, s, "both WALs number sessions the same way");
    other.append(s2, None, (0..8).collect()).unwrap();
    other.generate(s2, 4).unwrap();
    let (raw_tokens, _) = d
        .generate_tokens((0..8).collect(), GenParams::default(), 4, 0)
        .unwrap();
    assert!(!raw_tokens.is_empty());

    {
        let _p = tracing::info_span!("probe", session = s, n = 3u64, prompt = "TOPSECRET-CONTENT")
            .entered();
        let _q = tracing::info_span!("probe_child", k = 1u64).entered();
    }
    tracing::info!(marker = "not-a-span", "event");

    assert!(
        otlp::shutdown(Duration::from_secs(5)),
        "flush finished in time"
    );

    let posts = got.lock().unwrap().clone();
    assert!(!posts.is_empty(), "collector received nothing");
    let mut spans: Vec<Value> = Vec::new();
    for (token, ctype, body) in &posts {
        assert_eq!(token.as_deref(), Some("t0k"), "configured header sent");
        assert_eq!(ctype.as_deref(), Some("application/json"));
        assert!(
            !body.contains("TOPSECRET"),
            "string field content leaked: {body}"
        );
        assert!(!body.contains("not-a-span"));
        assert!(
            !body.contains("APPDATA"),
            "application string leaked: {body}"
        );
        let v: Value = serde_json::from_str(body).expect("valid JSON");
        let rs = &v["resourceSpans"][0];
        assert_eq!(
            attr(&rs["resource"], "service.name").unwrap()["stringValue"],
            "superfluid"
        );
        assert_eq!(
            attr(&rs["resource"], "deployment.environment").unwrap()["stringValue"],
            "test"
        );
        assert_eq!(
            attr(&rs["resource"], "superfluid.content_included").unwrap()["boolValue"],
            false
        );
        assert_eq!(rs["scopeSpans"][0]["scope"]["name"], "superfluid.operational");
        spans.extend(
            rs["scopeSpans"][0]["spans"]
                .as_array()
                .unwrap()
                .iter()
                .cloned(),
        );
    }

    for sp in &spans {
        assert!(is_hex(sp["traceId"].as_str().unwrap(), 32), "{sp}");
        assert!(is_hex(sp["spanId"].as_str().unwrap(), 16), "{sp}");
        if let Some(p) = sp.get("parentSpanId") {
            assert!(is_hex(p.as_str().unwrap(), 16), "{sp}");
        }
        let start: u64 = sp["startTimeUnixNano"].as_str().unwrap().parse().unwrap();
        let end: u64 = sp["endTimeUnixNano"].as_str().unwrap().parse().unwrap();
        assert!(start > 0 && start <= end, "{sp}");
        assert_eq!(sp["kind"], 1);
        for a in sp["attributes"].as_array().unwrap() {
            let v = a["value"].as_object().unwrap();
            assert_eq!(v.len(), 1);
            if let Some(i) = v.get("intValue") {
                assert!(i.as_str().unwrap().parse::<i64>().is_ok(), "{a}");
            }
        }
    }
    let by_id: HashMap<&str, &Value> = spans
        .iter()
        .map(|sp| (sp["spanId"].as_str().unwrap(), sp))
        .collect();
    assert_eq!(by_id.len(), spans.len(), "span ids are unique");
    let named = |n: &str| {
        spans
            .iter()
            .filter(move |sp| sp["name"] == n)
            .collect::<Vec<_>>()
    };

    let inspection = d.inspect(s).unwrap();
    let store_ns = export::store_trace_ns(&inspection.store_id.expect("store id reported"));
    let session_trace = export::hex_id(&export::trace_id_bytes_ns(store_ns, s));
    let session_root = export::hex_id(&export::span_id_bytes_ns(store_ns, s, export::ROOT));
    assert_ne!(
        session_trace,
        export::trace_id(s),
        "namespaced, not session-number-only"
    );
    let model_of = |sp: &Value| {
        attr(sp, "superfluid.model")
            .and_then(|v| v["stringValue"].as_str())
            .map(String::from)
    };
    let link_to = |sp: &Value, trace: &str, span: &str| {
        sp["links"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|l| l["traceId"] == trace && l["spanId"] == span)
    };
    let all_gens = named("generate");
    let (raw_gens, all_gens): (Vec<&Value>, Vec<&Value>) = all_gens
        .into_iter()
        .partition(|g| attr(g, "superfluid.raw").is_some());
    let (gens, other_gens): (Vec<&Value>, Vec<&Value>) = all_gens
        .iter()
        .partition(|g| model_of(g).as_deref() == Some("mock-model"));
    assert_eq!(gens.len(), 3, "one span per generate call");
    let embedder = named("embedder_request");
    assert_eq!(embedder.len(), 1, "ambient span exported");
    assert!(model_of(embedder[0]).is_none(), "{}", embedder[0]);
    let mut gen_traces = HashSet::new();
    for g in &gens {
        assert!(g.get("parentSpanId").is_none(), "{g}");
        assert_ne!(g["traceId"], session_trace.as_str());
        assert!(
            gen_traces.insert(g["traceId"].as_str().unwrap()),
            "one trace per call"
        );
        assert!(link_to(g, &session_trace, &session_root), "{g}");
        assert_eq!(
            attr(g, "superfluid.session_id").unwrap()["intValue"],
            s.to_string()
        );
        assert!(attr(g, "superfluid.max_tokens").is_some());
    }
    assert_eq!(other_gens.len(), 1);
    let og = other_gens[0];
    assert_eq!(model_of(og).as_deref(), Some("other-model"));
    let ns = export::store_trace_ns(&other.inspect(s2).unwrap().store_id.unwrap());
    assert_ne!(ns, store_ns, "two stores, two namespaces");
    let other_trace = export::hex_id(&export::trace_id_bytes_ns(ns, s2));
    let other_root = export::hex_id(&export::span_id_bytes_ns(ns, s2, export::ROOT));
    assert_ne!(other_trace, session_trace);
    assert!(link_to(og, &other_trace, &other_root), "{og}");
    assert!(!link_to(og, &session_trace, &session_root));
    assert_eq!(raw_gens.len(), 1, "raw completion traced");
    assert!(raw_gens[0].get("links").is_none());
    assert_eq!(model_of(raw_gens[0]).as_deref(), Some("mock-model"));
    let all_gens: Vec<&Value> = all_gens.into_iter().chain(raw_gens).collect();
    let exported = export::otlp_jsonl(
        s,
        &inspection.events,
        &export::ExportOptions {
            store_ns,
            ..Default::default()
        },
    );
    let exported: Value = serde_json::from_str(exported.trim()).unwrap();
    let root = &exported["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
    assert_eq!(root["name"], "session");
    assert_eq!(
        root["spanId"],
        session_root.as_str(),
        "live spans link to the session export's root"
    );
    assert_eq!(root["traceId"], session_trace.as_str());

    let ticks = named("tick");
    assert!(!ticks.is_empty(), "tick spans exported");
    let gen_trace: HashMap<&str, &str> = all_gens
        .iter()
        .map(|g| {
            (
                g["spanId"].as_str().unwrap(),
                g["traceId"].as_str().unwrap(),
            )
        })
        .collect();
    let mut linked = HashSet::new();
    for t in &ticks {
        assert!(t.get("parentSpanId").is_none(), "tick is a root: {t}");
        assert_ne!(t["traceId"], session_trace.as_str());
        let seq = attr(t, "superfluid.tick_seq");
        let no_tick = attr(t, "superfluid.no_tick").is_some();
        assert!(seq.is_some() != no_tick, "{t}");
        assert!(
            attr(t, "superfluid.lanes").is_some(),
            "lanes recorded after admission"
        );
        let tick_model = model_of(t);
        for l in t["links"].as_array().into_iter().flatten() {
            let id = l["spanId"].as_str().unwrap();
            let trace = gen_trace
                .get(id)
                .unwrap_or_else(|| panic!("link resolves to a generate span: {l}"));
            assert_eq!(l["traceId"], *trace, "link names the generate's trace");
            let g = by_id[id];
            assert_eq!(
                model_of(g),
                tick_model,
                "a tick links its own model's calls"
            );
            linked.insert(id);
        }
    }
    assert_eq!(
        linked.len(),
        all_gens.len(),
        "each generate call is linked from its ticks"
    );

    for sp in &spans {
        let Some(p) = sp.get("parentSpanId").and_then(Value::as_str) else {
            continue;
        };
        let parent = by_id
            .get(p)
            .unwrap_or_else(|| panic!("dangling parent: {sp}"));
        assert_eq!(parent["traceId"], sp["traceId"]);
    }
    let probe = named("probe");
    assert_eq!(probe.len(), 1);
    assert!(probe[0].get("links").is_none(), "{}", probe[0]);
    assert!(attr(probe[0], "superfluid.n").is_some());
    assert!(
        attr(probe[0], "superfluid.prompt").is_none(),
        "string fields are never exported"
    );
    let child = named("probe_child");
    assert_eq!(child[0]["parentSpanId"], probe[0]["spanId"]);
    assert_eq!(child[0]["traceId"], probe[0]["traceId"]);
    let parks = named("park");
    assert!(!parks.is_empty(), "park spans exported");
    let restores = named("restore");
    assert!(!restores.is_empty(), "restore span exported");
    for op in parks.into_iter().chain(restores) {
        assert_eq!(
            attr(op, "superfluid.session_id").unwrap()["intValue"],
            s.to_string()
        );
        let parent = by_id[op["parentSpanId"].as_str().unwrap()];
        assert_eq!(parent["name"], "tick", "{op}");
        assert!(
            model_of(op).is_some(),
            "per-space ops name their model: {op}"
        );
        assert_eq!(model_of(op), model_of(parent));
    }

    let text = d.metrics_text();
    let exported_n = otlp::EXPORTED.load(Ordering::Relaxed);
    assert_eq!(exported_n as usize, spans.len());
    assert!(
        text.contains(&format!("superfluid_otlp_spans_exported_total {exported_n}")),
        "{text}"
    );
    assert!(text.contains("superfluid_otlp_spans_dropped_total{reason=\"queue_full\"}"));
    assert!(text.contains("superfluid_otlp_spans_dropped_total{reason=\"export_failed\"}"));
}
