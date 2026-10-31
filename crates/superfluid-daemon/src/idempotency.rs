//! `Idempotency-Key`: a client that sends a POST again with the key it sent before gets the
//! answer the first one got, whole, from its start, however the first one ended. The first
//! request with a key runs to its end on its own and its response is recorded as it is made;
//! every client with that key, the first included, reads the recording, so one whose connection
//! dropped reconnects and reads the rest, and one that comes back after a restart reads a
//! finished answer from disk. A request a restart cut short runs again under its key.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use sha2::{Digest, Sha256};
use tokio_stream::StreamExt;

pub const HEADER: &str = "idempotency-key";

/// Set on an answer that was recorded for an earlier request with the same key.
pub const REPLAYED: &str = "idempotent-replayed";

/// How long a finished answer is kept for its key.
const KEEP: std::time::Duration = std::time::Duration::from_secs(24 * 3600);

/// The largest request body a key is taken with.
const MAX_BODY: usize = 256 << 20;

/// The headers a recording keeps: the body's framing is the replay's own.
fn kept(name: &HeaderName) -> bool {
    !matches!(name.as_str(), "content-length" | "transfer-encoding" | "connection")
}

/// The answers recorded under each key, live while their requests run and on disk after.
pub struct Records {
    dir: Option<PathBuf>,
    live: Mutex<HashMap<String, Arc<Record>>>,
}

struct Record {
    request: [u8; 32],
    state: Mutex<Recording>,
    changed: tokio::sync::Notify,
    /// When it was done, for one kept in memory alone.
    done_at: Mutex<Option<std::time::Instant>>,
}

impl Record {
    fn new(request: [u8; 32], recording: Recording) -> Arc<Record> {
        Arc::new(Record { request, state: Mutex::new(recording), changed: tokio::sync::Notify::new(), done_at: Mutex::new(None) })
    }
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct Recording {
    status: Option<u16>,
    headers: Vec<(String, String)>,
    #[serde(skip)]
    body: Vec<u8>,
    done: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Stored {
    request: String,
    #[serde(flatten)]
    recording: Recording,
}

impl Records {
    /// Records kept under `dir`, which a restart reads back; in memory alone without one.
    pub fn new(dir: Option<PathBuf>) -> Records {
        if let Some(dir) = &dir {
            sweep(dir);
        }
        Records { dir, live: Mutex::new(HashMap::new()) }
    }

    fn paths(&self, id: &str) -> Option<(PathBuf, PathBuf)> {
        self.dir.as_ref().map(|d| (d.join(format!("{id}.json")), d.join(format!("{id}.body"))))
    }

    /// A finished answer a run before this one recorded.
    fn stored(&self, id: &str) -> Option<(String, Recording)> {
        let (meta, body) = self.paths(id)?;
        let stored: Stored = serde_json::from_slice(&std::fs::read(meta).ok()?).ok()?;
        if !stored.recording.done {
            return None;
        }
        let mut recording = stored.recording;
        recording.body = std::fs::read(body).ok()?;
        Some((stored.request, recording))
    }

    fn persist(&self, id: &str, rec: &Record, recording: &Recording) {
        let Some((meta, _)) = self.paths(id) else { return };
        let stored = Stored {
            request: hex(&rec.request),
            recording: Recording { status: recording.status, headers: recording.headers.clone(), body: Vec::new(), done: recording.done },
        };
        if let Ok(bytes) = serde_json::to_vec(&stored) {
            let _ = std::fs::write(meta, bytes);
        }
    }

    fn append(&self, id: &str, bytes: &[u8]) {
        let Some((_, body)) = self.paths(id) else { return };
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(body) {
            let _ = f.write_all(bytes);
        }
    }

    fn start(&self, id: &str) {
        let Some((meta, body)) = self.paths(id) else { return };
        if let Some(dir) = meta.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::remove_file(meta);
        let _ = std::fs::remove_file(body);
    }
}

/// Drops what has been kept past [`KEEP`].
fn sweep(dir: &std::path::Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|age| age > KEEP);
        if old {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A key belongs to the credential it came with, so one client cannot read another's answer.
fn record_id(headers: &HeaderMap, key: &str) -> String {
    let mut h = Sha256::new();
    for name in ["authorization", "x-api-key"] {
        h.update(headers.get(name).map(HeaderValue::as_bytes).unwrap_or_default());
        h.update([0]);
    }
    h.update(key.as_bytes());
    hex(&h.finalize())
}

fn refused(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({"error": {"message": message, "type": "invalid_request_error", "param": HEADER, "code": "idempotency_key"}});
    (status, axum::Json(body)).into_response()
}

pub async fn layer(State(records): State<Arc<Records>>, req: Request, next: Next) -> Response {
    if req.method() != axum::http::Method::POST {
        return next.run(req).await;
    }
    let Some(key) = req.headers().get(HEADER).and_then(|v| v.to_str().ok()).map(str::to_string) else {
        return next.run(req).await;
    };
    if key.is_empty() || key.len() > 255 {
        return refused(StatusCode::BAD_REQUEST, "an Idempotency-Key is 1 to 255 characters");
    }
    let id = record_id(req.headers(), &key);
    let (parts, body) = req.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, MAX_BODY).await else {
        return refused(StatusCode::PAYLOAD_TOO_LARGE, "a request this large is not taken with an Idempotency-Key");
    };
    let request: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(parts.uri.path().as_bytes());
        h.update([0]);
        h.update(&bytes);
        h.finalize().into()
    };
    let other = || refused(StatusCode::UNPROCESSABLE_ENTITY, "this Idempotency-Key was sent with another request; use a new key for it");

    let live = { records.live.lock().expect("records").get(&id).cloned() };
    if let Some(rec) = live {
        return if rec.request == request { replay(rec, true).await } else { other() };
    }
    if let Some((stored, recording)) = records.stored(&id) {
        if stored != hex(&request) {
            return other();
        }
        return replay(Record::new(request, recording), true).await;
    }

    // Another request with this key may have got here first.
    let (rec, first) = {
        let mut live = records.live.lock().expect("records");
        live.retain(|_, r| r.done_at.lock().expect("record").is_none_or(|t| t.elapsed() < KEEP));
        match live.get(&id) {
            Some(rec) => (Arc::clone(rec), false),
            None => {
                let rec = Record::new(request, Recording::default());
                live.insert(id.clone(), Arc::clone(&rec));
                (rec, true)
            }
        }
    };
    if !first {
        return if rec.request == request { replay(rec, true).await } else { other() };
    }
    records.start(&id);
    let req = Request::from_parts(parts, Body::from(bytes));
    let (records2, rec2) = (Arc::clone(&records), Arc::clone(&rec));
    // Runs to its end whoever is reading, so a client that drops can come back for the rest.
    tokio::spawn(async move {
        let resp = next.run(req).await;
        let (parts, body) = resp.into_parts();
        {
            let mut st = rec2.state.lock().expect("recording");
            st.status = Some(parts.status.as_u16());
            st.headers = parts
                .headers
                .iter()
                .filter(|(n, _)| kept(n))
                .filter_map(|(n, v)| Some((n.as_str().to_string(), v.to_str().ok()?.to_string())))
                .collect();
            records2.persist(&id, &rec2, &st);
        }
        rec2.changed.notify_waiters();
        let mut stream = body.into_data_stream();
        while let Some(chunk) = stream.next().await {
            let Ok(chunk) = chunk else { break };
            records2.append(&id, &chunk);
            rec2.state.lock().expect("recording").body.extend_from_slice(&chunk);
            rec2.changed.notify_waiters();
        }
        {
            let mut st = rec2.state.lock().expect("recording");
            st.done = true;
            records2.persist(&id, &rec2, &st);
        }
        rec2.changed.notify_waiters();
        // On disk it is read back from there; in memory alone it is kept until it expires.
        if records2.dir.is_some() {
            records2.live.lock().expect("records").remove(&id);
        } else {
            *rec2.done_at.lock().expect("record") = Some(std::time::Instant::now());
        }
    });
    replay(rec, false).await
}

/// The recording as a response: its status and headers once they are known, then its body
/// from the start, followed as it grows until it is done.
async fn replay(rec: Arc<Record>, again: bool) -> Response {
    let (status, headers) = loop {
        let changed = rec.changed.notified();
        {
            let st = rec.state.lock().expect("recording");
            if let Some(status) = st.status {
                break (status, st.headers.clone());
            }
        }
        changed.await;
    };
    let body = futures_util::stream::unfold((rec, 0usize), |(rec, at)| async move {
        loop {
            let watch = Arc::clone(&rec);
            let changed = watch.changed.notified();
            let ready = {
                let st = rec.state.lock().expect("recording");
                if st.body.len() > at {
                    Some(Some((Bytes::copy_from_slice(&st.body[at..]), st.body.len())))
                } else if st.done {
                    Some(None)
                } else {
                    None
                }
            };
            match ready {
                Some(Some((chunk, next))) => return Some((Ok::<_, std::io::Error>(chunk), (rec, next))),
                Some(None) => return None,
                None => changed.await,
            }
        }
    });
    let mut resp = Response::new(Body::from_stream(body));
    *resp.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    for (n, v) in headers {
        if let (Ok(n), Ok(v)) = (HeaderName::try_from(n), HeaderValue::try_from(v)) {
            resp.headers_mut().append(n, v);
        }
    }
    if again {
        resp.headers_mut().insert(REPLAYED, HeaderValue::from_static("true"));
    }
    resp
}
