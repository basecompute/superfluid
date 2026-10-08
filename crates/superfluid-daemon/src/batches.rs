use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const BATCH_POLICY: crate::inference::RequestPolicy = crate::inference::RequestPolicy {
    limits: BATCH_LIMITS,
    sampling: crate::inference::SamplingOverrides {
        temperature: None,
        top_p: None,
        top_k: None,
        min_p: None,
        repeat_penalty: None,
        spec_max_temperature: None,
        fallback: None,
    },
    qos: crate::inference::RequestQos::AGENT,
};

const BATCH_LIMITS: crate::inference::TokenLimits = crate::inference::TokenLimits {
    default_max_tokens: Some(crate::inference::DEFAULT_MAX_TOKENS),
    max_context: crate::inference::DEFAULT_MAX_CONTEXT,
};

use serde::{Deserialize, Serialize};

use crate::Daemon;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestCounts {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Batch {
    pub id: String,
    #[serde(default = "batch_object")]
    pub object: String,
    pub endpoint: String,
    pub input_file_id: String,
    pub completion_window: String,
    pub status: String,
    pub output_file_id: Option<String>,
    #[serde(default)]
    pub error_file_id: String,
    pub created_at: u64,
    #[serde(default)]
    pub completed_at: u64,
    #[serde(default)]
    pub seq: u64,
    pub request_counts: RequestCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
}

fn batch_object() -> String {
    "batch".to_string()
}
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Batches run on at most this many threads; the rest wait in `validating`, in order.
pub const MAX_RUNNING_BATCHES: usize = 4;
pub const MAX_QUEUED_BATCHES: usize = 256;

type Job = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct Runner {
    running: usize,
    queued: std::collections::VecDeque<Job>,
}

#[derive(Debug)]
pub struct BatchesBusy;

static STATUS: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub struct BatchStore {
    dir: PathBuf,
    runner: std::sync::Mutex<Runner>,
}

impl BatchStore {
    pub fn new(dir: PathBuf) -> BatchStore {
        let _ = std::fs::create_dir_all(&dir);
        BatchStore { dir, runner: std::sync::Mutex::new(Runner::default()) }
    }

    /// Creates a batch and runs `job` with its id on a batch worker, or queues it behind the
    /// running ones. Refuses, creating nothing, when the queue is full.
    pub fn start(
        self: &Arc<Self>,
        endpoint: &str,
        input_file_id: &str,
        completion_window: &str,
        total: u64,
        owner: Option<&str>,
        job: impl FnOnce(String) + Send + 'static,
    ) -> Result<Batch, BatchesBusy> {
        let mut r = self.runner.lock().unwrap_or_else(|e| e.into_inner());
        if r.queued.len() >= MAX_QUEUED_BATCHES {
            return Err(BatchesBusy);
        }
        let batch = self.create(endpoint, input_file_id, completion_window, total, owner);
        let id = batch.id.clone();
        let job: Job = Box::new(move || job(id));
        if r.running >= MAX_RUNNING_BATCHES {
            r.queued.push_back(job);
            return Ok(batch);
        }
        r.running += 1;
        drop(r);
        let store = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("superfluid-batch".into())
            .spawn(move || store.work(job));
        if spawned.is_err() {
            self.runner.lock().unwrap_or_else(|e| e.into_inner()).running -= 1;
            let mut b = batch.clone();
            b.status = "failed".to_string();
            b.completed_at = now_secs();
            self.write(&b);
        }
        Ok(batch)
    }

    fn work(&self, mut job: Job) {
        loop {
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_err() {
                eprintln!("superfluid: a batch worker panicked; its batch is left as it was");
            }
            let mut r = self.runner.lock().unwrap_or_else(|e| e.into_inner());
            match r.queued.pop_front() {
                Some(next) => job = next,
                None => {
                    r.running -= 1;
                    return;
                }
            }
        }
    }

    fn new_id() -> String {
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        format!("batch-{:x}{:x}", nanos, N.fetch_add(1, Ordering::Relaxed))
    }

    fn valid_id(id: &str) -> bool {
        id.starts_with("batch-") && id.len() > 6 && id[6..].chars().all(|c| c.is_ascii_hexdigit())
    }
    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    fn write(&self, b: &Batch) {
        if let Ok(j) = serde_json::to_vec(b) {
            let path = self.path(&b.id);
            let tmp = path.with_extension(format!("json.tmp.{:?}", std::thread::current().id()).replace(['(', ')'], ""));
            if std::fs::write(&tmp, j).is_ok() && std::fs::rename(&tmp, &path).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }

    pub fn create(
        &self,
        endpoint: &str,
        input_file_id: &str,
        completion_window: &str,
        total: u64,
        owner: Option<&str>,
    ) -> Batch {
        let b = Batch {
            id: Self::new_id(),
            object: batch_object(),
            endpoint: endpoint.to_string(),
            input_file_id: input_file_id.to_string(),
            completion_window: completion_window.to_string(),
            status: "validating".to_string(),
            output_file_id: None,
            error_file_id: String::new(),
            created_at: now_secs(),
            completed_at: 0,
            seq: {
                static SEQ: AtomicU64 = AtomicU64::new(1);
                SEQ.fetch_add(1, Ordering::Relaxed)
            },
            request_counts: RequestCounts { total, completed: 0, failed: 0 },
            owner: owner.map(str::to_string),
        };
        self.write(&b);
        b
    }

    pub fn get(&self, id: &str) -> Option<Batch> {
        if !Self::valid_id(id) {
            return None;
        }
        serde_json::from_slice(&std::fs::read(self.path(id)).ok()?).ok()
    }

    pub fn list(&self) -> Vec<Batch> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                if e.path().extension().and_then(|x| x.to_str()) == Some("json") {
                    if let Ok(b) = std::fs::read(e.path()) {
                        if let Ok(m) = serde_json::from_slice::<Batch>(&b) {
                            out.push(m);
                        }
                    }
                }
            }
        }
        out.sort_by_key(|b| std::cmp::Reverse(b.created_at));
        out
    }

    pub fn cancel(&self, id: &str) -> Option<(Batch, bool)> {
        let _one = STATUS.lock().unwrap_or_else(|e| e.into_inner());
        let mut b = self.get(id)?;
        let stops = b.status == "in_progress" || b.status == "validating";
        if b.status == "validating" {
            b.status = "cancelled".to_string();
            b.completed_at = now_secs();
            self.write(&b);
        } else if stops {
            b.status = "cancelling".to_string();
            self.write(&b);
        }
        Some((b, stops))
    }

    fn begin(&self, id: &str) -> Option<Batch> {
        let _one = STATUS.lock().unwrap_or_else(|e| e.into_inner());
        let mut b = self.get(id).filter(|b| b.status == "validating")?;
        b.status = "in_progress".to_string();
        self.write(&b);
        Some(b)
    }
}

#[derive(Deserialize)]
struct BatchLine {
    #[serde(default)]
    custom_id: String,
    #[serde(default)]
    url: String,
    body: serde_json::Value,
}

pub fn process(
    daemon: Arc<Daemon>,
    store: Arc<BatchStore>,
    model_name: String,
    batch_id: String,
    qos: crate::inference::RequestQos,
    pace: Option<Arc<crate::keypolicy::KeyPolicy>>,
) {
    let Some(mut batch) = store.begin(&batch_id) else {
        return;
    };

    let input = match daemon.files().content(&batch.input_file_id) {
        Some(bytes) => bytes,
        None => {
            batch.status = "failed".to_string();
            batch.completed_at = now_secs();
            store.write(&batch);
            return;
        }
    };
    let text = String::from_utf8_lossy(&input);
    let mut out_lines = String::new();
    let (mut completed, mut failed) = (0u64, 0u64);

    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if store.get(&batch_id).map(|b| b.status) == Some("cancelling".to_string()) {
            batch.status = "cancelled".to_string();
            batch.completed_at = now_secs();
            batch.request_counts = RequestCounts { total: batch.request_counts.total, completed, failed };
            store.write(&batch);
            return;
        }
        if let Some(key) = &pace {
            let cancelled = || store.get(&batch_id).map(|b| b.status) == Some("cancelling".to_string());
            if !key.pace(cancelled) {
                batch.status = "cancelled".to_string();
                batch.completed_at = now_secs();
                batch.request_counts = RequestCounts { total: batch.request_counts.total, completed, failed };
                store.write(&batch);
                return;
            }
        }
        let parsed: Result<BatchLine, _> = serde_json::from_str(line);
        let (status_code, body_val): (u16, serde_json::Value) = match parsed {
            Ok(bl) => run_line(&daemon, &model_name, &bl.url, &bl.body, qos),
            Err(e) => (400, serde_json::json!({"error": {"message": format!("bad input line: {e}")}})),
        };
        let custom_id = serde_json::from_str::<BatchLine>(line)
            .map(|b| b.custom_id)
            .unwrap_or_else(|_| format!("line-{i}"));
        if (200..300).contains(&status_code) {
            completed += 1;
        } else {
            failed += 1;
        }
        let row = serde_json::json!({
            "id": format!("batch_req_{i}"),
            "custom_id": custom_id,
            "response": {"status_code": status_code, "body": body_val},
            "error": serde_json::Value::Null,
        });
        out_lines.push_str(&row.to_string());
        out_lines.push('\n');
    }

    let output_file_id = daemon
        .files()
        .put("batch_output.jsonl", "batch_output", out_lines.as_bytes(), batch.owner.as_deref())
        .ok()
        .map(|m| m.id);
    batch.output_file_id = output_file_id;
    batch.request_counts = RequestCounts { total: batch.request_counts.total, completed, failed };
    batch.status = "completed".to_string();
    batch.completed_at = now_secs();
    store.write(&batch);
}

fn run_line(
    daemon: &Daemon,
    model_name: &str,
    url: &str,
    body: &serde_json::Value,
    qos: crate::inference::RequestQos,
) -> (u16, serde_json::Value) {
    match url {
        "/v1/chat/completions" => match serde_json::from_value::<crate::inference::ChatRequest>(body.clone()) {
            Ok(req) => match crate::openai::chat_once(
                daemon,
                &req,
                model_name,
                crate::inference::RequestPolicy { qos, ..BATCH_POLICY },
            ) {
                Ok(v) => (200, v),
                Err(e) => (500, serde_json::json!({"error": {"message": e.to_string()}})),
            },
            Err(e) => (400, serde_json::json!({"error": {"message": format!("bad chat body: {e}")}})),
        },
        "/v1/embeddings" => {
            let input = body.get("input").and_then(|v| v.as_str()).unwrap_or("");
            match daemon.embed_as(input, Some(qos.class)) {
                Ok(v) => (
                    200,
                    serde_json::json!({"object": "list", "data": [{"object":"embedding","index":0,"embedding": v}], "model": model_name}),
                ),
                Err(e) => (500, serde_json::json!({"error": {"message": e.to_string()}})),
            }
        }
        other => (400, serde_json::json!({"error": {"message": format!("unsupported batch url: {other}")}})),
    }
}

pub fn batches_dir_for(sessions_dir: &std::path::Path) -> PathBuf {
    sessions_dir.join("batches")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_run_on_a_bounded_pool_and_a_full_queue_refuses() {
        let dir = std::env::temp_dir().join(format!("superfluid-batch-pool-{}", std::process::id()));
        let store = Arc::new(BatchStore::new(dir.clone()));
        let (release, gate) = std::sync::mpsc::channel::<()>();
        let gate = Arc::new(std::sync::Mutex::new(gate));
        let live = Arc::new(AtomicU64::new(0));
        let peak = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicU64::new(0));
        let total = MAX_RUNNING_BATCHES + MAX_QUEUED_BATCHES;
        for _ in 0..total {
            let (gate, live, peak, done) = (Arc::clone(&gate), Arc::clone(&live), Arc::clone(&peak), Arc::clone(&done));
            store
                .start("/v1/chat/completions", "file-x", "24h", 1, None, move |_id| {
                    peak.fetch_max(live.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                    let _ = gate.lock().unwrap().recv();
                    live.fetch_sub(1, Ordering::SeqCst);
                    done.fetch_add(1, Ordering::SeqCst);
                })
                .expect("room");
        }
        let refused = store.start("/v1/chat/completions", "file-x", "24h", 1, None, |_| {});
        assert!(refused.is_err(), "a full queue refuses");
        assert_eq!(store.list().len(), total, "a refused batch is not created");
        for _ in 0..total {
            release.send(()).unwrap();
        }
        for _ in 0..400 {
            if done.load(Ordering::SeqCst) == total as u64 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(done.load(Ordering::SeqCst), total as u64);
        assert!(peak.load(Ordering::SeqCst) <= MAX_RUNNING_BATCHES as u64, "peak {}", peak.load(Ordering::SeqCst));
        assert!(store.start("/v1/chat/completions", "file-x", "24h", 1, None, |_| {}).is_ok(), "room again");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_batch_cancelled_while_queued_never_starts() {
        let dir = std::env::temp_dir().join(format!("superfluid-batch-queued-cancel-{}", std::process::id()));
        let store = BatchStore::new(dir.clone());
        let b = store.create("/v1/chat/completions", "file-x", "24h", 1, None);
        let (cancelled, stops) = store.cancel(&b.id).unwrap();
        assert!(stops);
        assert_eq!(cancelled.status, "cancelled");
        assert!(store.begin(&b.id).is_none(), "a worker reaching it later does not run it");
        assert_eq!(store.get(&b.id).unwrap().status, "cancelled");
        let _ = std::fs::remove_dir_all(dir);
    }
}
