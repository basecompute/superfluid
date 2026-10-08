use std::fmt::Write as _;

use serde_json::{json, Value};

use crate::api::EventMsg;
use crate::wal::EventBody;

pub const GENAI_SEMCONV_VERSION: &str = "1.36.0";

#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub include_content: bool,
    pub max_field_bytes: usize,
    pub model: String,
    pub store_ns: u64,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions {
            include_content: false,
            max_field_bytes: 64 * 1024,
            model: String::new(),
            store_ns: 0,
        }
    }
}

pub fn redact(text: &str, max: usize) -> String {
    let mut out = String::with_capacity(text.len().min(max));
    for (i, word) in text.split(' ').enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let w = word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_');
        let secret = (w.starts_with("sk-") && w.len() >= 24)
            || (w.starts_with("AKIA") && w.len() == 20 && w.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()))
            || (w.starts_with("ghp_") && w.len() >= 36)
            || (w.starts_with("xox") && w.len() >= 24 && w.as_bytes().get(4) == Some(&b'-'))
            || (w.starts_with("AIza") && w.len() == 39);
        if secret {
            out.push_str("[REDACTED]");
        } else {
            out.push_str(word);
        }
    }
    let mut out = out.replace("PRIVATE KEY-----", "PRIVATE KEY-----[REDACTED]");
    if let Some(i) = out.find("-----BEGIN") {
        if let Some(j) = out[i..].find("-----END") {
            out.replace_range(i..i + j, "-----BEGIN [REDACTED] ");
        }
    }
    if out.len() > max {
        let mut cut = max;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
        out.push_str("…[truncated]");
    }
    out
}

fn content(text: &str, o: &ExportOptions) -> Value {
    if o.include_content {
        json!(redact(text, o.max_field_bytes))
    } else {
        json!({ "bytes": text.len() })
    }
}

fn body_json(body: &EventBody, o: &ExportOptions) -> Value {
    match body {
        EventBody::Created { parent, params } => json!({"type":"created","parent":parent,"params":params_json(params)}),
        EventBody::Appended { text, span } => json!({"type":"appended","text": text.as_deref().map(|t| content(t, o)),"tokens":span.len()}),
        EventBody::Message { role, text, span } => json!({"type":"message","role":role,"text":content(text,o),"tokens":span.len()}),
        EventBody::GenerationPrompt { span } => json!({"type":"generation_prompt","tokens":span.len()}),
        EventBody::Generated { span, text, channel, finish } => json!({"type":"generated","tokens":span.len(),"text":content(text,o),"channel":channel,"finish":finish}),
        EventBody::ToolUse { name, arguments } => json!({"type":"tool_use","name":name,"arguments":content(arguments,o)}),
        EventBody::ToolResult { call_id, content: c, span } => json!({"type":"tool_result","call_id":call_id,"content":content(c,o),"tokens":span.len()}),
        EventBody::EpochBump => json!({"type":"epoch_bump"}),
        EventBody::GenerationFingerprint { digest } => json!({"type":"generation_fingerprint","digest":format!("{digest:016x}")}),
        EventBody::ToolParseFailure { raw } => json!({"type":"tool_parse_failure","raw":content(raw,o)}),
        EventBody::Forked { parent, fork_at, params } => json!({"type":"forked","parent":parent,"fork_at":fork_at,"params":params_json(params)}),
        EventBody::MetaUpdated { version, title, archived } => json!({"type":"meta_updated","version":version,"title":title.as_deref().map(|t| content(t,o)),"archived":archived}),
        EventBody::Rebased { parent, fork_at, params, edits } => json!({"type":"rebased","parent":parent,"fork_at":fork_at,"params":params_json(params),"edits":edits.len()}),
        EventBody::ToolExpired { call_id, reason } => json!({"type":"tool_expired","call_id":call_id,"reason":reason}),
        EventBody::Purged { generation, descendants } => json!({"type":"purged","generation":generation,"descendants":format!("{descendants:?}")}),
        EventBody::Rerooted { purged_parent, prefix } => json!({"type":"rerooted","purged_parent":purged_parent,"prefix_events":prefix.len()}),
        EventBody::QosSet { class, batch_invariant } => json!({"type":"qos_set","class":class,"batch_invariant":batch_invariant}),
        EventBody::Block { role, kind, payload, span, visibility_version } => json!({"type":"block","role":role,"kind":kind,"payload":content(payload,o),"tokens":span.len(),"visibility_version":visibility_version}),
        EventBody::PermissionRequest { call_id, text } => json!({"type":"permission_request","call_id":call_id,"text":content(text,o)}),
        EventBody::PermissionResponse { request_id, granted } => json!({"type":"permission_response","request_id":request_id,"granted":granted}),
        EventBody::ToolCancelRequested { call_id } => json!({"type":"tool_cancel_requested","call_id":call_id}),
        EventBody::ToolOutcome { call_id, outcome, note } => json!({"type":"tool_outcome","call_id":call_id,"outcome":outcome,"note":content(note,o)}),
        EventBody::ToolReconciliation { call_id, note } => json!({"type":"tool_reconciliation","call_id":call_id,"note":content(note,o)}),
        EventBody::ToolLease { call_id, deadline_unix_ms } => json!({"type":"tool_lease","call_id":call_id,"deadline_unix_ms":deadline_unix_ms}),
        EventBody::Pinned { deadline_unix_ms } => json!({"type":"pinned","deadline_unix_ms":deadline_unix_ms}),
    }
}

fn params_json(p: &crate::GenParams) -> Value {
    json!({"temperature":p.temperature,"top_p":p.top_p,"min_p":p.min_p,"top_k":p.top_k,"seed":p.seed})
}

pub fn jsonl(session: u64, events: &[EventMsg], o: &ExportOptions) -> String {
    let mut out = String::new();
    for e in events {
        let line = json!({
            "session": session,
            "event_id": e.event_id,
            "epoch": e.epoch,
            "ts_unix_ms": e.ts_unix_ms,
            "event": body_json(&e.body, o),
        });
        let _ = writeln!(out, "{line}");
    }
    out
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn hex_id(bytes: &[u8]) -> String {
    hex(bytes)
}

pub fn trace_id_bytes(session: u64) -> [u8; 16] {
    let h1 = xxhash_rust::xxh3::xxh3_64(&session.to_le_bytes());
    let h2 = xxhash_rust::xxh3::xxh3_64_with_seed(&session.to_le_bytes(), 0x5e55);
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&h1.to_be_bytes());
    out[8..].copy_from_slice(&h2.to_be_bytes());
    out
}

pub fn store_trace_ns(id: &crate::store::StoreId) -> u64 {
    (xxhash_rust::xxh3::xxh3_64_with_seed(&id.0, 0x5354) >> 1).max(1)
}

pub fn trace_id_bytes_ns(ns: u64, session: u64) -> [u8; 16] {
    if ns == 0 {
        return trace_id_bytes(session);
    }
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&ns.to_le_bytes());
    b[8..].copy_from_slice(&session.to_le_bytes());
    let h1 = xxhash_rust::xxh3::xxh3_64(&b);
    let h2 = xxhash_rust::xxh3::xxh3_64_with_seed(&b, 0x5e55);
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&h1.to_be_bytes());
    out[8..].copy_from_slice(&h2.to_be_bytes());
    out
}

pub fn span_id_bytes_ns(ns: u64, session: u64, event_id: u64) -> [u8; 8] {
    if ns == 0 {
        return span_id_bytes(session, event_id);
    }
    let mut b = [0u8; 24];
    b[..8].copy_from_slice(&ns.to_le_bytes());
    b[8..16].copy_from_slice(&session.to_le_bytes());
    b[16..].copy_from_slice(&event_id.to_le_bytes());
    xxhash_rust::xxh3::xxh3_64(&b).to_be_bytes()
}

pub fn trace_id(session: u64) -> String {
    hex(&trace_id_bytes(session))
}

pub fn span_id_bytes(session: u64, event_id: u64) -> [u8; 8] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&session.to_le_bytes());
    b[8..].copy_from_slice(&event_id.to_le_bytes());
    xxhash_rust::xxh3::xxh3_64(&b).to_be_bytes()
}

pub fn span_id(session: u64, event_id: u64) -> String {
    hex(&span_id_bytes(session, event_id))
}

pub const ROOT: u64 = u64::MAX;

fn ns(ms: u64) -> String {
    (ms.saturating_mul(1_000_000)).to_string()
}

pub(crate) fn attr(key: &str, v: Value) -> Value {
    let value = match v {
        Value::String(s) => json!({"stringValue": s}),
        Value::Bool(b) => json!({"boolValue": b}),
        Value::Number(n) if n.is_i64() || n.is_u64() => json!({"intValue": n.to_string()}),
        Value::Number(n) => json!({"doubleValue": n}),
        other => json!({"stringValue": other.to_string()}),
    };
    json!({"key": key, "value": value})
}

pub fn otlp_jsonl(session: u64, events: &[EventMsg], o: &ExportOptions) -> String {
    let store_ns = o.store_ns;
    let tid = hex(&trace_id_bytes_ns(store_ns, session));
    let span_id =
        |session: u64, event_id: u64| hex(&span_id_bytes_ns(store_ns, session, event_id));
    let first_ts = events.iter().map(|e| e.ts_unix_ms).find(|&t| t > 0).unwrap_or(0);
    let last_ts = events.iter().map(|e| e.ts_unix_ms).max().unwrap_or(0);
    let timing = if first_ts == 0 { "unavailable" } else { "wall_clock_ms" };
    let mut spans: Vec<Value> = Vec::new();
    spans.push(json!({
        "traceId": tid, "spanId": span_id(session, ROOT),
        "name": "session", "kind": 1,
        "startTimeUnixNano": ns(first_ts), "endTimeUnixNano": ns(last_ts),
        "attributes": [
            attr("superfluid.session_id", json!(session)),
            attr("superfluid.event_id", json!(ROOT)),
            attr("superfluid.timing", json!(timing)),
        ],
    }));
    let mut open_gen: Option<(u64, u64, u64, u32)> = None;
    let mut open_tools: std::collections::HashMap<u64, (u64, u64, String, u64)> = Default::default();
    for e in events {
        match &e.body {
            EventBody::GenerationPrompt { .. } => {
                open_gen = Some((e.event_id, e.ts_unix_ms, 0, e.epoch as u32));
            }
            EventBody::Generated { span, finish, .. } => {
                let g = open_gen.get_or_insert((e.event_id, e.ts_unix_ms, 0, e.epoch as u32));
                g.2 += span.len() as u64;
                if *finish != superfluid_abi::finish::NONE {
                    let (start_id, start_ts, tokens, epoch) = *g;
                    spans.push(json!({
                        "traceId": tid, "spanId": span_id(session, start_id),
                        "parentSpanId": span_id(session, ROOT),
                        "name": "generate", "kind": 1,
                        "startTimeUnixNano": ns(start_ts), "endTimeUnixNano": ns(e.ts_unix_ms),
                        "attributes": [
                            attr("superfluid.session_id", json!(session)),
                            attr("superfluid.event_id", json!(start_id)),
                            attr("superfluid.parent_event_id", json!(ROOT)),
                            attr("superfluid.epoch", json!(epoch)),
                            attr("superfluid.finish", json!(finish)),
                            attr("gen_ai.system", json!("superfluid")),
                            attr("gen_ai.request.model", json!(o.model.clone())),
                            attr("gen_ai.usage.output_tokens", json!(tokens)),
                        ],
                    }));
                    open_gen = None;
                }
            }
            EventBody::ToolUse { name, .. } => {
                open_tools.insert(e.event_id, (e.event_id, e.ts_unix_ms, name.clone(), e.epoch));
            }
            EventBody::ToolResult { call_id, .. }
            | EventBody::ToolExpired { call_id, .. }
            | EventBody::ToolOutcome { call_id, .. } => {
                if let Some((start_id, start_ts, name, epoch)) = open_tools.remove(call_id) {
                    let parent = open_gen.map(|g| g.0).unwrap_or(ROOT);
                    let expired = matches!(e.body, EventBody::ToolExpired { .. });
                    let outcome = match &e.body {
                        EventBody::ToolOutcome { outcome, .. } => {
                            if *outcome == crate::wal::tool_outcome::CANCELLED {
                                "cancelled"
                            } else {
                                "failed"
                            }
                        }
                        _ if expired => "expired",
                        _ => "result",
                    };
                    spans.push(json!({
                        "traceId": tid, "spanId": span_id(session, start_id),
                        "parentSpanId": span_id(session, parent),
                        "name": "tool_use", "kind": 3,
                        "startTimeUnixNano": ns(start_ts), "endTimeUnixNano": ns(e.ts_unix_ms),
                        "attributes": [
                            attr("superfluid.session_id", json!(session)),
                            attr("superfluid.event_id", json!(start_id)),
                            attr("superfluid.parent_event_id", json!(parent)),
                            attr("superfluid.epoch", json!(epoch)),
                            attr("gen_ai.tool.name", json!(name)),
                            attr("gen_ai.tool.call.id", json!(start_id.to_string())),
                            attr("superfluid.tool.outcome", json!(outcome)),
                        ],
                    }));
                }
            }
            _ => {}
        }
    }
    for (_, (start_id, start_ts, name, epoch)) in open_tools {
        spans.push(json!({
            "traceId": tid, "spanId": span_id(session, start_id),
            "parentSpanId": span_id(session, open_gen.map(|g| g.0).unwrap_or(ROOT)),
            "name": "tool_use", "kind": 3,
            "startTimeUnixNano": ns(start_ts), "endTimeUnixNano": ns(start_ts),
            "attributes": [
                attr("superfluid.session_id", json!(session)),
                attr("superfluid.event_id", json!(start_id)),
                attr("superfluid.epoch", json!(epoch)),
                attr("gen_ai.tool.name", json!(name)),
                attr("gen_ai.tool.call.id", json!(start_id.to_string())),
                attr("superfluid.tool.outcome", json!("open")),
            ],
        }));
    }
    let req = trace_request(
        vec![
            attr("service.name", json!("superfluid")),
            attr("superfluid.semconv_version", json!(GENAI_SEMCONV_VERSION)),
            attr("superfluid.content_included", json!(o.include_content)),
        ],
        "superfluid.session",
        spans,
    );
    format!("{req}\n")
}

pub(crate) fn trace_request(resource: Vec<Value>, scope: &str, spans: Vec<Value>) -> Value {
    json!({
        "resourceSpans": [{
            "resource": {"attributes": resource},
            "scopeSpans": [{
                "scope": {"name": scope, "version": env!("CARGO_PKG_VERSION")},
                "spans": spans,
            }],
        }]
    })
}

pub fn push_otlp_traces(endpoint: &str, session: u64, events: &[EventMsg], o: &ExportOptions) -> Result<u16, String> {
    let body = otlp_jsonl(session, events, o);
    let url = format!("{}/v1/traces", endpoint.trim_end_matches('/'));
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(10)))
        .build()
        .new_agent();
    let resp = agent
        .post(&url)
        .header("content-type", "application/json")
        .send(body.as_bytes())
        .map_err(|e| format!("otlp push to {url}: {e}"))?;
    Ok(resp.status().as_u16())
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::scheduler::{SchedStats, TTFT_BUCKETS_MS};

pub fn otlp_metrics_json(s: &SchedStats, now_nanos: u64, start_nanos: u64) -> String {
    let g = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
    let now = now_nanos.to_string();
    let start = start_nanos.to_string();

    let sum = |name: &str, value: u64| {
        json!({
            "name": name, "unit": "1",
            "sum": {
                "aggregationTemporality": 2, "isMonotonic": true,
                "dataPoints": [{
                    "asInt": value.to_string(),
                    "startTimeUnixNano": start, "timeUnixNano": now
                }]
            }
        })
    };
    let gauge_pt = |name: &str, value: u64, attrs: Value| {
        json!({
            "name": name, "unit": "1",
            "gauge": { "dataPoints": [{
                "asInt": value.to_string(), "timeUnixNano": now, "attributes": attrs
            }]}
        })
    };
    let attr = |k: &str, v: &str| json!([{ "key": k, "value": { "stringValue": v } }]);

    let mut metrics = vec![
        sum("superfluid_ticks_total", g(&s.ticks)),
        sum("superfluid_prefill_tokens_total", g(&s.prefill_tokens)),
        sum("superfluid_decode_tokens_total", g(&s.decode_tokens)),
        sum("superfluid_warm_prefix_tokens_total", g(&s.warm_prefix_tokens)),
        sum("superfluid_cold_admissions_total", g(&s.cold_admissions)),
        sum("superfluid_unservable_cold_admissions_total", g(&s.unservable_cold_admissions)),
        sum("superfluid_unservable_park_refusals_total", g(&s.unservable_park_refusals)),
        sum("superfluid_preemptions_total", g(&s.preemptions)),
        sum("superfluid_starvation_grants_total", g(&s.starvation_grants)),
        sum("superfluid_os_pressure_events_total", g(&s.os_pressure_events)),
        sum("superfluid_pressure_evictions_total", g(&s.pressure_evictions)),
        sum("superfluid_pressure_bytes_evicted_total", g(&s.pressure_bytes_evicted)),
        sum("superfluid_worker_respawns_total", g(&s.worker_respawns)),
        sum("superfluid_spec_proposed_total", g(&s.spec_proposed)),
        sum("superfluid_spec_accepted_total", g(&s.spec_accepted)),
        sum("superfluid_resumes_total", g(&s.resumes)),
        sum("superfluid_pins_yielded_total", g(&s.pins_yielded)),
        sum("superfluid_pins_expired_total", g(&s.pins_expired)),
    ];

    let park_points: Vec<Value> = [("lossless", g(&s.parks_lossless)), ("lossy", g(&s.parks_lossy))]
        .into_iter()
        .map(|(enc, v)| {
            json!({
                "asInt": v.to_string(), "startTimeUnixNano": start, "timeUnixNano": now,
                "attributes": attr("encoding", enc)
            })
        })
        .collect();
    metrics.push(json!({
        "name": "superfluid_parks_total", "unit": "1",
        "sum": { "aggregationTemporality": 2, "isMonotonic": true, "dataPoints": park_points }
    }));

    metrics.push(gauge_pt("superfluid_lanes_active", g(&s.lanes_active), json!([])));
    metrics.push(gauge_pt("superfluid_pins_held", g(&s.pins_held), json!([])));
    metrics.push(gauge_pt("superfluid_pinned_bytes", g(&s.pinned_bytes), json!([])));
    for (c, name) in ["interactive_chat", "inline_completion", "foreground_agent", "background_agent"]
        .iter()
        .enumerate()
    {
        metrics.push(gauge_pt("superfluid_queue_depth", g(&s.queue_depth[c]), attr("class", name)));
    }

    let bounds: Vec<f64> = TTFT_BUCKETS_MS.iter().map(|b| *b as f64 / 1000.0).collect();
    let bucket_counts: Vec<String> = (0..=TTFT_BUCKETS_MS.len())
        .map(|i| g(&s.ttft_buckets[i]).to_string())
        .collect();
    metrics.push(json!({
        "name": "superfluid_ttft_seconds", "unit": "s",
        "histogram": { "aggregationTemporality": 2, "dataPoints": [{
            "startTimeUnixNano": start, "timeUnixNano": now,
            "count": g(&s.ttft_count).to_string(),
            "sum": g(&s.ttft_sum_ms) as f64 / 1000.0,
            "bucketCounts": bucket_counts,
            "explicitBounds": bounds
        }]}
    }));

    serde_json::to_string(&json!({
        "resourceMetrics": [{
            "resource": { "attributes": attr("service.name", "superfluid") },
            "scopeMetrics": [{
                "scope": { "name": "superfluid", "version": env!("CARGO_PKG_VERSION") },
                "metrics": metrics
            }]
        }]
    }))
    .expect("otlp metrics json")
}

pub fn push_otlp_metrics(endpoint: &str, s: &SchedStats) -> Result<u16, String> {
    let now = crate::wal::now_unix_ms() * 1_000_000;
    let body = otlp_metrics_json(s, now, now);
    let url = format!("{}/v1/metrics", endpoint.trim_end_matches('/'));
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(10)))
        .build()
        .new_agent();
    let resp = agent
        .post(&url)
        .header("content-type", "application/json")
        .send(body.as_bytes())
        .map_err(|e| format!("otlp metrics push to {url}: {e}"))?;
    Ok(resp.status().as_u16())
}

pub struct MetricsPusher {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl MetricsPusher {
    pub const MIN_INTERVAL: Duration = Duration::from_millis(10);

    pub fn start(endpoint: String, interval: Duration, stats: Arc<SchedStats>) -> MetricsPusher {
        let interval = interval.max(Self::MIN_INTERVAL);
        let stop = Arc::new(AtomicBool::new(false));
        let s = Arc::clone(&stop);
        let slice = Duration::from_millis(100).min(interval.max(Duration::from_millis(1)));
        let handle = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let mut waited = Duration::ZERO;
                while waited < interval && !s.load(Ordering::Relaxed) {
                    std::thread::sleep(slice);
                    waited += slice;
                }
                if s.load(Ordering::Relaxed) {
                    break;
                }
                if let Err(e) = push_otlp_metrics(&endpoint, &stats) {
                    tracing::debug!("otlp metrics push failed (retry next interval): {e}");
                }
            }
        });
        MetricsPusher {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for MetricsPusher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
