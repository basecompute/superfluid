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

/// How often a running server sweeps the answers kept past [`KEEP`] from disk.
const SWEEP_EVERY: std::time::Duration = std::time::Duration::from_secs(3600);

/// Whether what was done at `t` has been kept past [`KEEP`].
fn past_keep(t: std::time::SystemTime) -> bool {
    t.elapsed().is_ok_and(|age| age > KEEP)
}

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
    /// When `dir` was last swept.
    swept: Mutex<Option<std::time::Instant>>,
}

struct Record {
    request: [u8; 32],
    state: Mutex<Recording>,
    changed: tokio::sync::Notify,
    /// When it was done, for one kept in memory alone: by the wall clock, as one on disk is aged
    /// by its file's time.
    done_at: Mutex<Option<std::time::SystemTime>>,
}

impl Record {
    fn new(request: [u8; 32], recording: Recording) -> Arc<Record> {
        Arc::new(Record { request, state: Mutex::new(recording), changed: tokio::sync::Notify::new(), done_at: Mutex::new(None) })
    }

    fn expired(&self) -> bool {
        self.done_at.lock().expect("record").is_some_and(past_keep)
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
        let swept = Mutex::new(dir.is_some().then(std::time::Instant::now));
        let records = Records { dir, live: Mutex::new(HashMap::new()), swept };
        if let Some(dir) = &records.dir {
            sweep(dir, &records.live);
        }
        records
    }

    /// Sweeps `dir` again, off the request's thread, once [`SWEEP_EVERY`] has passed since the
    /// last time, so a server that stays up drops what it has kept past [`KEEP`].
    fn sweep_when_due(self: &Arc<Self>) {
        let Some(dir) = self.dir.clone() else { return };
        {
            let mut swept = self.swept.lock().expect("records");
            if swept.is_some_and(|t| t.elapsed() < SWEEP_EVERY) {
                return;
            }
            *swept = Some(std::time::Instant::now());
        }
        let records = Arc::clone(self);
        tokio::task::spawn_blocking(move || sweep(&dir, &records.live));
    }

    fn paths(&self, id: &str) -> Option<(PathBuf, PathBuf)> {
        self.dir.as_ref().map(|d| (d.join(format!("{id}.json")), d.join(format!("{id}.body"))))
    }

    /// A finished answer recorded on disk, unless it has been kept past [`KEEP`].
    fn stored(&self, id: &str) -> Option<(String, Recording)> {
        let (meta, body) = self.paths(id)?;
        if std::fs::metadata(&meta).and_then(|m| m.modified()).is_ok_and(past_keep) {
            return None;
        }
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

/// Drops what has been kept past [`KEEP`], but not the files of a key in `live`: a request
/// reusing an expired key takes it there before it clears and rewrites them. With `live` held,
/// a file is looked at again before it goes, so one rewritten and answered since the first look
/// (its key out of `live` again) is not taken for the old one.
fn sweep(dir: &std::path::Path, live: &Mutex<HashMap<String, Arc<Record>>>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        if e.metadata().and_then(|m| m.modified()).is_ok_and(past_keep) {
            let path = e.path();
            #[cfg(test)]
            if let Some(between) = AFTER_FIRST_LOOK.with(std::cell::Cell::get) {
                between(&path);
            }
            let live = live.lock().expect("records");
            if path.file_stem().and_then(|s| s.to_str()).is_some_and(|id| live.contains_key(id)) {
                continue;
            }
            if std::fs::metadata(&path).and_then(|m| m.modified()).is_ok_and(past_keep) {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Run by this thread's sweeps between the first look at an old file and the locked one.
    static AFTER_FIRST_LOOK: std::cell::Cell<Option<fn(&std::path::Path)>> = const { std::cell::Cell::new(None) };
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

    let live = { records.live.lock().expect("records").get(&id).filter(|r| !r.expired()).cloned() };
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
        live.retain(|_, r| !r.expired());
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
    records.sweep_when_due();
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
            // On disk it is read back from there, where its age is checked, and leaves the live
            // map before anyone can read that it is done; in memory alone it is kept until it
            // expires.
            if records2.dir.is_some() {
                records2.live.lock().expect("records").remove(&id);
            } else {
                *rec2.done_at.lock().expect("record") = Some(std::time::SystemTime::now());
            }
        }
        rec2.changed.notify_waiters();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A server behind the layer, answering each request that reaches it with how many have.
    fn server(records: Arc<Records>) -> (String, tokio::runtime::Runtime) {
        let calls = Arc::new(AtomicU64::new(0));
        let app = axum::Router::new()
            .route(
                "/x",
                axum::routing::post(move || {
                    let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
                    async move { n.to_string() }
                }),
            )
            .layer(axum::middleware::from_fn_with_state(records, layer));
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        rt.spawn(async move { axum::serve(listener, app).await });
        (addr, rt)
    }

    /// Whether the answer was replayed, and the answer.
    fn post(addr: &str, key: &str) -> (bool, String) {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(20)))
            .http_status_as_error(false)
            .build()
            .new_agent();
        let mut resp = agent.post(&format!("http://{addr}/x")).header(HEADER, key).send("the request").unwrap();
        (resp.headers().contains_key(REPLAYED), resp.body_mut().read_to_string().unwrap())
    }

    /// Waits for `what` to hold, for up to 20 s.
    fn until(what: impl Fn() -> bool) -> bool {
        let t = std::time::Instant::now();
        while !what() {
            if t.elapsed() > std::time::Duration::from_secs(20) {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        true
    }

    #[test]
    fn an_answer_kept_in_memory_past_its_day_is_not_replayed() {
        let records = Arc::new(Records::new(None));
        let (addr, _rt) = server(Arc::clone(&records));
        assert_eq!(post(&addr, "k"), (false, "1".into()));
        let rec = records.live.lock().unwrap().get(&record_id(&HeaderMap::new(), "k")).cloned().unwrap();
        assert!(until(|| rec.done_at.lock().unwrap().is_some()), "the answer is done");
        assert_eq!(post(&addr, "k"), (true, "1".into()));
        // As a day and an hour after it was done.
        *rec.done_at.lock().unwrap() = Some(std::time::SystemTime::now() - KEEP - std::time::Duration::from_secs(3600));
        assert_eq!(post(&addr, "k"), (false, "2".into()), "run again, not the answer kept a day ago");
    }

    #[test]
    fn a_running_server_sweeps_what_it_kept_past_its_day_from_disk() {
        let dir = std::env::temp_dir().join(format!("superfluid-idem-sweep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let records = Arc::new(Records::new(Some(dir.clone())));
        let (addr, _rt) = server(Arc::clone(&records));
        // A record no one asks for again, kept a day and an hour ago.
        let old = [dir.join("old.json"), dir.join("old.body")];
        for f in &old {
            std::fs::write(f, "{}").unwrap();
            let then = std::time::SystemTime::now() - KEEP - std::time::Duration::from_secs(3600);
            std::fs::File::options().write(true).open(f).unwrap().set_modified(then).unwrap();
        }
        // Swept when the server started: not again within the hour.
        post(&addr, "a");
        assert!(old.iter().all(|f| f.exists()));
        // An hour on, the next new key sweeps it.
        *records.swept.lock().unwrap() = None;
        post(&addr, "b");
        assert!(until(|| old.iter().all(|f| !f.exists())), "a record kept past its day is swept while the server runs");
        assert_eq!(post(&addr, "a"), (true, "1".into()), "what is kept within its day stays");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_sweep_leaves_the_files_of_a_key_being_answered_again() {
        let dir = std::env::temp_dir().join(format!("superfluid-idem-live-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let records = Records::new(None);
        // An expired key taken again: in the live map, its files not yet cleared.
        records.live.lock().unwrap().insert("again".into(), Record::new([0; 32], Recording::default()));
        let files = [dir.join("again.json"), dir.join("again.body"), dir.join("gone.json"), dir.join("gone.body")];
        for f in &files {
            std::fs::write(f, "{}").unwrap();
            let then = std::time::SystemTime::now() - KEEP - std::time::Duration::from_secs(3600);
            std::fs::File::options().write(true).open(f).unwrap().set_modified(then).unwrap();
        }
        sweep(&dir, &records.live);
        assert!(files[..2].iter().all(|f| f.exists()), "the key being answered again keeps its files");
        assert!(files[2..].iter().all(|f| !f.exists()), "a key no one holds is swept");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_sweep_leaves_a_record_answered_again_since_it_first_looked() {
        // Between the sweep's first look at an expired key's file and its removal, a request takes
        // the key, rewrites the file, answers and leaves the live map.
        let dir = std::env::temp_dir().join(format!("superfluid-idem-again-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let records = Records::new(None);
        let files = [dir.join("again.json"), dir.join("again.body")];
        for f in &files {
            std::fs::write(f, "{}").unwrap();
            let then = std::time::SystemTime::now() - KEEP - std::time::Duration::from_secs(3600);
            std::fs::File::options().write(true).open(f).unwrap().set_modified(then).unwrap();
        }
        fn answered_again(path: &std::path::Path) {
            std::fs::write(path, "{\"answered\":\"again\"}").unwrap();
        }
        AFTER_FIRST_LOOK.with(|f| f.set(Some(answered_again)));
        sweep(&dir, &records.live);
        AFTER_FIRST_LOOK.with(|f| f.set(None));
        assert!(files.iter().all(|f| f.exists()), "a record answered again since the sweep first looked is kept");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
