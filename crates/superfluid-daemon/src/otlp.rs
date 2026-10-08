use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use crate::export;

pub static EXPORTED: AtomicU64 = AtomicU64::new(0);
pub static DROPPED_QUEUE_FULL: AtomicU64 = AtomicU64::new(0);
pub static DROPPED_REJECTED: AtomicU64 = AtomicU64::new(0);
pub static DROPPED_EXPORT_FAILED: AtomicU64 = AtomicU64::new(0);

pub const DEFAULT_FILTER: &str = "info,superfluid_daemon::scheduler=debug";

#[derive(Debug, Clone)]
pub struct OtlpConfig {
    pub endpoint: String,
    pub headers: Vec<(String, String)>,
    pub service_name: String,
    pub resource: Vec<(String, String)>,
    pub filter: String,
    pub queue: usize,
    pub batch_max: usize,
    pub interval: Duration,
    pub timeout: Duration,
}

impl Default for OtlpConfig {
    fn default() -> Self {
        OtlpConfig {
            endpoint: String::new(),
            headers: Vec::new(),
            service_name: "superfluid".into(),
            resource: Vec::new(),
            filter: DEFAULT_FILTER.into(),
            queue: 8192,
            batch_max: 512,
            interval: Duration::from_millis(1000),
            timeout: Duration::from_millis(5000),
        }
    }
}

impl OtlpConfig {
    pub fn traces_url(&self) -> String {
        let e = self.endpoint.trim_end_matches('/');
        if e.ends_with("/v1/traces") {
            e.to_string()
        } else {
            format!("{e}/v1/traces")
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_http_url(&self.endpoint)?;
        for (k, v) in &self.headers {
            ureq::http::HeaderName::from_bytes(k.as_bytes())
                .map_err(|_| format!("--otlp-header: bad header name {k:?}"))?;
            ureq::http::HeaderValue::from_str(v)
                .map_err(|_| format!("--otlp-header: bad value for {k:?}"))?;
        }
        tracing_subscriber::EnvFilter::try_new(&self.filter)
            .map_err(|e| format!("--otlp-filter {:?}: {e}", self.filter))?;
        if self.queue == 0 || self.batch_max == 0 {
            return Err("--otlp-queue and the batch size must be > 0".into());
        }
        if self.interval.is_zero() {
            return Err("--otlp-batch-ms must be > 0".into());
        }
        if self.timeout.is_zero() {
            return Err("--otlp-timeout-ms must be > 0".into());
        }
        Ok(())
    }

    pub fn is_loopback(&self) -> bool {
        let rest = self.endpoint.trim_start_matches("http://");
        let host = rest.split('/').next().unwrap_or("");
        let host = match host.strip_prefix('[') {
            Some(v6) => v6.split(']').next().unwrap_or(""),
            None => host.rsplit_once(':').map_or(host, |(h, _)| h),
        };
        host == "localhost"
            || host
                .parse::<std::net::IpAddr>()
                .map(|ip| ip.is_loopback())
                .unwrap_or(false)
    }
}

pub fn validate_http_url(e: &str) -> Result<(), String> {
    if e.starts_with("https://") {
        return Err(format!(
            "{e}: https is not supported (this build's HTTP client has no TLS) — \
             point it at a local OpenTelemetry collector over http:// and let it \
             forward with TLS"
        ));
    }
    if !e.starts_with("http://") {
        return Err(format!("{e}: expected an http:// URL"));
    }
    let uri: ureq::http::Uri = e
        .parse()
        .map_err(|err| format!("{e}: not a valid URL: {err}"))?;
    let auth = uri
        .authority()
        .ok_or_else(|| format!("{e}: missing host"))?;
    if auth.host().is_empty() {
        return Err(format!("{e}: missing host"));
    }
    if let Some((_, port)) = auth.as_str().rsplit_once(':') {
        if !auth.as_str().ends_with(']') && port.parse::<u16>().is_err() {
            return Err(format!("{e}: bad port {port:?}"));
        }
    }
    if uri.query().is_some() {
        return Err(format!("{e}: a query string is not allowed"));
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct TraceScope {
    pub model: String,
    pub ns: u64,
}

impl TraceScope {
    pub fn model(model: &str) -> TraceScope {
        TraceScope {
            model: model.to_string(),
            ns: 0,
        }
    }
}

const STRING_FIELDS: &[&str] = &["model"];

#[derive(Debug, Clone)]
enum Val {
    I(i64),
    U(u64),
    F(f64),
    B(bool),
    S(String),
}

struct SpanData {
    trace: [u8; 16],
    span: [u8; 8],
    parent: Option<[u8; 8]>,
    start_ns: u64,
    attrs: Vec<(&'static str, Val)>,
    links: Vec<([u8; 16], [u8; 8])>,
}

struct Finished {
    name: &'static str,
    target: &'static str,
    end_ns: u64,
    data: SpanData,
}

struct Fields<'a>(&'a mut Vec<(&'static str, Val)>, bool);

fn daemon_owned(target: &str) -> bool {
    target == "superfluid_daemon" || target.starts_with("superfluid_daemon::")
}

impl Visit for Fields<'_> {
    fn record_i64(&mut self, f: &Field, v: i64) {
        self.0.push((f.name(), Val::I(v)));
    }
    fn record_u64(&mut self, f: &Field, v: u64) {
        self.0.push((f.name(), Val::U(v)));
    }
    fn record_f64(&mut self, f: &Field, v: f64) {
        self.0.push((f.name(), Val::F(v)));
    }
    fn record_bool(&mut self, f: &Field, v: bool) {
        self.0.push((f.name(), Val::B(v)));
    }
    fn record_i128(&mut self, f: &Field, v: i128) {
        if let Ok(v) = i64::try_from(v) {
            self.record_i64(f, v);
        }
    }
    fn record_u128(&mut self, f: &Field, v: u128) {
        if let Ok(v) = u64::try_from(v) {
            self.record_u64(f, v);
        }
    }
    fn record_str(&mut self, f: &Field, v: &str) {
        if self.1 && STRING_FIELDS.contains(&f.name()) {
            self.0.push((f.name(), Val::S(v.to_string())));
        }
    }
    fn record_debug(&mut self, _: &Field, _: &dyn std::fmt::Debug) {}
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn fresh_u64() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    static N: AtomicU64 = AtomicU64::new(0);
    let seed = *SEED.get_or_init(|| now_ns() ^ ((std::process::id() as u64) << 32));
    let n = N.fetch_add(1, Ordering::Relaxed);
    xxhash_rust::xxh3::xxh3_64_with_seed(&n.to_le_bytes(), seed).max(1)
}

fn fresh_trace() -> [u8; 16] {
    let mut t = [0u8; 16];
    t[..8].copy_from_slice(&fresh_u64().to_be_bytes());
    t[8..].copy_from_slice(&fresh_u64().to_be_bytes());
    t
}

pub struct OtlpLayer {
    tx: SyncSender<Finished>,
}

impl<S> Layer<S> for OtlpLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut fields = Vec::new();
        let owned = daemon_owned(span.metadata().target());
        attrs.record(&mut Fields(&mut fields, owned));
        let parent = span
            .parent()
            .and_then(|p| p.extensions().get::<SpanData>().map(|d| (d.trace, d.span)));
        let num = |name: &str| {
            fields.iter().find_map(|(k, v)| match (*k == name, v) {
                (true, Val::U(x)) => Some(*x),
                (true, Val::I(x)) if *x >= 0 => Some(*x as u64),
                _ => None,
            })
        };
        let session = num("session")
            .filter(|s| *s != 0)
            .zip(num("store_ns").filter(|ns| *ns != 0 && owned));
        let (trace, parent, links) = match (parent, session) {
            (Some((t, p)), _) => (t, Some(p), Vec::new()),
            (None, Some((s, ns))) => (
                fresh_trace(),
                None,
                vec![(
                    export::trace_id_bytes_ns(ns, s),
                    export::span_id_bytes_ns(ns, s, export::ROOT),
                )],
            ),
            (None, None) => (fresh_trace(), None, Vec::new()),
        };
        span.extensions_mut().insert(SpanData {
            trace,
            span: fresh_u64().to_be_bytes(),
            parent,
            start_ns: now_ns(),
            attrs: fields,
            links,
        });
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if let Some(d) = span.extensions_mut().get_mut::<SpanData>() {
                let owned = daemon_owned(span.metadata().target());
                values.record(&mut Fields(&mut d.attrs, owned));
            }
        }
    }

    fn on_follows_from(&self, id: &Id, follows: &Id, ctx: Context<'_, S>) {
        if id == follows {
            return;
        }
        let Some(target) = ctx
            .span(follows)
            .and_then(|s| s.extensions().get::<SpanData>().map(|d| (d.trace, d.span)))
        else {
            return;
        };
        if let Some(span) = ctx.span(id) {
            if let Some(d) = span.extensions_mut().get_mut::<SpanData>() {
                d.links.push(target);
            }
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };
        let Some(data) = span.extensions_mut().remove::<SpanData>() else {
            return;
        };
        let meta = span.metadata();
        let fin = Finished {
            name: meta.name(),
            target: meta.target(),
            end_ns: now_ns(),
            data,
        };
        match self.tx.try_send(fin) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                DROPPED_QUEUE_FULL.fetch_add(1, Ordering::Relaxed);
                crate::telemetry::DROPPED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

fn rejected_spans(body: &str) -> u64 {
    let Ok(v) = serde_json::from_str::<Value>(body) else {
        return 0;
    };
    let r = &v["partialSuccess"]["rejectedSpans"];
    r.as_u64()
        .or_else(|| r.as_str().and_then(|s| s.parse().ok()))
        .unwrap_or(0)
}

fn attr_key(field: &str) -> String {
    match field {
        "session" => "superfluid.session_id".into(),
        f => format!("superfluid.{f}"),
    }
}

fn val_json(v: &Val) -> Value {
    match *v {
        Val::I(i) => json!(i),
        Val::U(u) => json!(u as i64),
        Val::F(f) => json!(f),
        Val::B(b) => json!(b),
        Val::S(ref s) => json!(s),
    }
}

fn span_json(s: &Finished) -> Value {
    let d = &s.data;
    let mut attrs = Vec::with_capacity(d.attrs.len() + 1);
    attrs.push(export::attr("superfluid.target", json!(s.target)));
    for (k, v) in &d.attrs {
        attrs.push(export::attr(&attr_key(k), val_json(v)));
    }
    let mut o = json!({
        "traceId": export::hex(&d.trace),
        "spanId": export::hex(&d.span),
        "name": s.name,
        "kind": 1,
        "startTimeUnixNano": d.start_ns.to_string(),
        "endTimeUnixNano": s.end_ns.max(d.start_ns).to_string(),
        "attributes": attrs,
    });
    if let Some(p) = d.parent {
        o["parentSpanId"] = json!(export::hex(&p));
    }
    if !d.links.is_empty() {
        o["links"] = Value::Array(
            d.links
                .iter()
                .map(|(t, sp)| json!({"traceId": export::hex(t), "spanId": export::hex(sp)}))
                .collect(),
        );
    }
    o
}

fn encode(cfg: &OtlpConfig, host: &str, batch: &[Finished]) -> String {
    let mut resource = vec![
        export::attr("service.name", json!(cfg.service_name)),
        export::attr("service.version", json!(env!("CARGO_PKG_VERSION"))),
        export::attr("host.name", json!(host)),
        export::attr("superfluid.signal", json!("operational")),
        export::attr("superfluid.content_included", json!(false)),
    ];
    for (k, v) in &cfg.resource {
        resource.push(export::attr(k, json!(v)));
    }
    let spans = batch.iter().map(span_json).collect();
    export::trace_request(resource, "superfluid.operational", spans).to_string()
}

struct Shared {
    stop: AtomicBool,
    done: Mutex<bool>,
    cv: Condvar,
}

#[derive(Clone)]
pub struct OtlpHandle {
    shared: Arc<Shared>,
}

impl OtlpHandle {
    pub fn shutdown(&self, bound: Duration) -> bool {
        self.shared.stop.store(true, Ordering::SeqCst);
        let done = self.shared.done.lock().unwrap_or_else(|e| e.into_inner());
        let (done, _) = self
            .shared
            .cv
            .wait_timeout_while(done, bound, |d| !*d)
            .unwrap_or_else(|e| e.into_inner());
        *done
    }
}

static GLOBAL: OnceLock<OtlpHandle> = OnceLock::new();

pub(crate) fn set_global(h: OtlpHandle) {
    let _ = GLOBAL.set(h);
}

pub fn shutdown(bound: Duration) -> bool {
    GLOBAL.get().map(|h| h.shutdown(bound)).unwrap_or(true)
}

pub fn start(cfg: OtlpConfig) -> Result<(OtlpLayer, OtlpHandle), String> {
    cfg.validate()?;
    let (tx, rx) = sync_channel::<Finished>(cfg.queue);
    let shared = Arc::new(Shared {
        stop: AtomicBool::new(false),
        done: Mutex::new(false),
        cv: Condvar::new(),
    });
    let sh = Arc::clone(&shared);
    std::thread::Builder::new()
        .name("superfluid-otlp".into())
        .spawn(move || {
            run_exporter(&cfg, &rx, &sh);
            *sh.done.lock().unwrap_or_else(|e| e.into_inner()) = true;
            sh.cv.notify_all();
        })
        .map_err(|e| e.to_string())?;
    Ok((OtlpLayer { tx }, OtlpHandle { shared }))
}

const STOP_POLL: Duration = Duration::from_millis(50);

fn run_exporter(cfg: &OtlpConfig, rx: &Receiver<Finished>, sh: &Shared) {
    let url = cfg.traces_url();
    let host = crate::metrics::hostname();
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(cfg.timeout))
        .build()
        .new_agent();
    let mut failing = false;
    let mut flush = |batch: &mut Vec<Finished>| {
        if batch.is_empty() {
            return;
        }
        let n = batch.len() as u64;
        let body = encode(cfg, &host, batch);
        batch.clear();
        let mut req = agent.post(&url).header("content-type", "application/json");
        for (k, v) in &cfg.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        match req.send(body.as_bytes()) {
            Ok(mut resp) => {
                let text = resp
                    .body_mut()
                    .with_config()
                    .limit(64 * 1024)
                    .read_to_string()
                    .unwrap_or_default();
                let rejected = rejected_spans(&text).min(n);
                EXPORTED.fetch_add(n - rejected, Ordering::Relaxed);
                if rejected > 0 {
                    DROPPED_REJECTED.fetch_add(rejected, Ordering::Relaxed);
                    crate::telemetry::DROPPED.fetch_add(rejected, Ordering::Relaxed);
                }
                if failing {
                    failing = false;
                    tracing::info!(target: "superfluid_daemon::otlp", "otlp trace export recovered");
                }
            }
            Err(e) => {
                DROPPED_EXPORT_FAILED.fetch_add(n, Ordering::Relaxed);
                crate::telemetry::DROPPED.fetch_add(n, Ordering::Relaxed);
                if !failing {
                    failing = true;
                    tracing::warn!(target: "superfluid_daemon::otlp", error = %e, "otlp trace export failing; spans are dropped until it recovers");
                }
            }
        }
    };
    let mut batch: Vec<Finished> = Vec::with_capacity(cfg.batch_max);
    let mut deadline = Instant::now() + cfg.interval;
    loop {
        if sh.stop.load(Ordering::SeqCst) {
            while let Ok(s) = rx.try_recv() {
                batch.push(s);
                if batch.len() >= cfg.batch_max {
                    flush(&mut batch);
                }
            }
            flush(&mut batch);
            return;
        }
        let now = Instant::now();
        let wait = deadline.saturating_duration_since(now).min(STOP_POLL);
        match rx.recv_timeout(wait) {
            Ok(s) => {
                batch.push(s);
                if batch.len() >= cfg.batch_max || Instant::now() >= deadline {
                    flush(&mut batch);
                    deadline = Instant::now() + cfg.interval;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if Instant::now() >= deadline {
                    flush(&mut batch);
                    deadline = Instant::now() + cfg.interval;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                flush(&mut batch);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traces_url_and_validation() {
        let c = |e: &str| OtlpConfig {
            endpoint: e.into(),
            ..Default::default()
        };
        assert_eq!(c("http://h:4318").traces_url(), "http://h:4318/v1/traces");
        assert_eq!(c("http://h:4318/").traces_url(), "http://h:4318/v1/traces");
        assert_eq!(
            c("http://h:4318/v1/traces").traces_url(),
            "http://h:4318/v1/traces"
        );
        assert!(c("http://h:4318").validate().is_ok());
        assert!(c("https://h:4318").validate().unwrap_err().contains("TLS"));
        assert!(c("h:4318").validate().is_err());
        assert!(c("http://").validate().is_err());
        assert!(c("http://localhost:notaport")
            .validate()
            .unwrap_err()
            .contains("port"));
        assert!(c("http://localhost:99999").validate().is_err());
        assert!(c("http://h:4318/v1/traces?x=1").validate().is_err());
        assert!(c("http://[::1]:4318").validate().is_ok());
        assert!(c("http://collector/otlp").validate().is_ok());
        let mut bad = c("http://h:1");
        bad.headers.push(("x-auth".into(), "a\nb".into()));
        assert!(bad.validate().unwrap_err().contains("bad value"));
        assert!(c("http://127.0.0.1:4318").is_loopback());
        assert!(c("http://localhost:4318").is_loopback());
        assert!(c("http://[::1]:4318/x").is_loopback());
        assert!(!c("http://10.0.0.2:4318").is_loopback());
        assert!(!c("http://collector:4318").is_loopback());
        let mut bad = c("http://h:1");
        bad.headers.push(("x auth".into(), "v".into()));
        assert!(bad.validate().is_err());
        let mut bad = c("http://h:1");
        bad.filter = "not a directive[".into();
        assert!(bad.validate().unwrap_err().contains("--otlp-filter"));
        let mut bad = c("http://h:1");
        bad.interval = Duration::ZERO;
        assert!(bad.validate().unwrap_err().contains("--otlp-batch-ms"));
        let mut bad = c("http://h:1");
        bad.timeout = Duration::ZERO;
        assert!(bad.validate().unwrap_err().contains("--otlp-timeout-ms"));
    }

    #[test]
    fn partial_success_counts_rejected_spans() {
        assert_eq!(rejected_spans("{}"), 0);
        assert_eq!(rejected_spans(""), 0);
        assert_eq!(rejected_spans(r#"{"partialSuccess":{}}"#), 0);
        assert_eq!(
            rejected_spans(r#"{"partialSuccess":{"rejectedSpans":"3"}}"#),
            3
        );
        assert_eq!(
            rejected_spans(r#"{"partialSuccess":{"rejectedSpans":4}}"#),
            4
        );
    }

    #[test]
    fn store_namespaces_session_ids() {
        use crate::store::StoreId;
        assert_eq!(export::trace_id_bytes_ns(0, 7), export::trace_id_bytes(7));
        assert_eq!(export::span_id_bytes_ns(0, 7, 3), export::span_id_bytes(7, 3));
        let a = export::store_trace_ns(&StoreId([1; 16]));
        let b = export::store_trace_ns(&StoreId([2; 16]));
        assert!(a != 0 && b != 0 && a != b);
        assert!(a <= i64::MAX as u64 && b <= i64::MAX as u64, "int64 domain");
        assert_eq!(a, export::store_trace_ns(&StoreId([1; 16])), "stable");
        assert_ne!(export::trace_id_bytes_ns(a, 1), export::trace_id_bytes_ns(b, 1));
        assert_ne!(export::trace_id_bytes_ns(a, 1), export::trace_id_bytes(1));
    }
}
