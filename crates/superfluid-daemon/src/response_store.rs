//! Stored responses for the OpenAI `/v1/responses` API: which session each
//! response ran in, so `previous_response_id` can continue it.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::DaemonError;

/// What a response id names. Small, so a scan of every record stays cheap.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseRecord {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    pub model: String,
    pub session: u64,
    /// The event a continuation forks at: one past the response's last.
    pub end_event: u64,
    /// The response's turn ended in end-of-turn, so a turn appended after
    /// it renders as a new turn rather than inside this one.
    pub turn_closed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Digest of what the session renders first (system text, tools, template
    /// arguments). A continuation with another head is rebuilt, not forked.
    pub head: String,
    /// The text-only system message that opens the conversation's first
    /// input, which leads the rebuilt conversation when no instructions do.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_system: Option<String>,
    pub created_at: u64,
    /// Deleted, but kept because a later response's history includes it.
    #[serde(default)]
    pub deleted: bool,
}

/// The response object as returned, and the input items it was given.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseBody {
    pub response: serde_json::Value,
    pub input_items: Vec<serde_json::Value>,
}

pub struct ResponseStore {
    dir: PathBuf,
    /// Puts and deletes one at a time, so a delete's scan of what continues
    /// a response sees every record.
    lock: Mutex<()>,
}

pub fn responses_dir_for(sessions_dir: &Path) -> PathBuf {
    sessions_dir.join("responses")
}

/// A fresh `resp_` id: 128 random bits, so an id is not guessed from another.
pub fn new_response_id() -> Result<String, DaemonError> {
    use std::io::Read;
    let mut buf = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut buf))
        .map_err(DaemonError::Io)?;
    Ok(format!("resp_{}", buf.iter().map(|b| format!("{b:02x}")).collect::<String>()))
}

fn valid_id(id: &str) -> bool {
    id.strip_prefix("resp_")
        .is_some_and(|h| h.len() == 32 && h.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Writes `bytes` so that a crash leaves the old file or the new one, never a
/// torn one: write aside, flush to disk, rename over, flush the directory.
fn write_durable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(bytes)?;
    crate::wal::durable_flush(&f)?;
    drop(f);
    std::fs::rename(&tmp, path)?;
    if let Some(dir) = path.parent() {
        crate::wal::durable_flush(&std::fs::File::open(dir)?)?;
    }
    Ok(())
}

impl ResponseStore {
    pub fn new(dir: PathBuf) -> ResponseStore {
        let _ = std::fs::create_dir_all(&dir);
        ResponseStore { dir, lock: Mutex::new(()) }
    }

    fn meta_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    fn body_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.items.json"))
    }

    /// Stores a response. The body lands first: a record never names a
    /// body that is not on disk.
    pub fn put(&self, record: &ResponseRecord, body: &ResponseBody) -> Result<(), DaemonError> {
        if !valid_id(&record.id) {
            return Err(DaemonError::Protocol("malformed response id"));
        }
        let _one = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        write_durable(&self.body_path(&record.id), &encode(body)?).map_err(DaemonError::Io)?;
        write_durable(&self.meta_path(&record.id), &encode(record)?).map_err(DaemonError::Io)
    }

    /// The record, deleted or not.
    pub fn record(&self, id: &str) -> Option<ResponseRecord> {
        if !valid_id(id) {
            return None;
        }
        serde_json::from_slice(&std::fs::read(self.meta_path(id)).ok()?).ok()
    }

    /// The record of a response that has not been deleted.
    pub fn get(&self, id: &str) -> Option<ResponseRecord> {
        self.record(id).filter(|r| !r.deleted)
    }

    pub fn body(&self, id: &str) -> Option<ResponseBody> {
        if !valid_id(id) {
            return None;
        }
        serde_json::from_slice(&std::fs::read(self.body_path(id)).ok()?).ok()
    }

    fn records(&self) -> Vec<ResponseRecord> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return Vec::new() };
        rd.flatten()
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let id = name.strip_suffix(".json").filter(|id| valid_id(id))?;
                self.record(id)
            })
            .collect()
    }

    /// Deletes a response; one a later response continues is only marked, and goes with its
    /// deleted ancestors once nothing continues it.
    pub fn delete(&self, id: &str) -> Result<bool, DaemonError> {
        let _one = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut rec) = self.get(id) else { return Ok(false) };
        let records = self.records();
        let referenced = |id: &str| records.iter().any(|r| r.previous_response_id.as_deref() == Some(id) && r.id != id);
        if referenced(id) {
            rec.deleted = true;
            write_durable(&self.meta_path(id), &encode(&rec)?).map_err(DaemonError::Io)?;
            return Ok(true);
        }
        let mut gone: Vec<String> = Vec::new();
        let mut next = Some(rec);
        while let Some(r) = next.take() {
            let _ = std::fs::remove_file(self.meta_path(&r.id));
            let _ = std::fs::remove_file(self.body_path(&r.id));
            gone.push(r.id.clone());
            next = r
                .previous_response_id
                .as_deref()
                .and_then(|p| self.record(p))
                .filter(|p| p.deleted)
                .filter(|p| !records.iter().any(|c| c.previous_response_id.as_deref() == Some(&p.id) && !gone.contains(&c.id)));
        }
        Ok(true)
    }
}

fn encode<T: Serialize>(v: &T) -> Result<Vec<u8>, DaemonError> {
    serde_json::to_vec(v).map_err(|_| DaemonError::Protocol("response record encode"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> ResponseStore {
        let dir = std::env::temp_dir().join(format!(
            "superfluid-responses-store-{}-{}",
            std::process::id(),
            new_response_id().unwrap()
        ));
        ResponseStore::new(dir)
    }

    fn rec(id: &str, prev: Option<&str>) -> ResponseRecord {
        ResponseRecord {
            id: id.into(),
            owner: None,
            model: "m".into(),
            session: 1,
            end_event: 3,
            turn_closed: true,
            previous_response_id: prev.map(str::to_string),
            head: "h".into(),
            root_system: None,
            created_at: 0,
            deleted: false,
        }
    }

    fn body() -> ResponseBody {
        ResponseBody { response: serde_json::json!({"object": "response"}), input_items: vec![] }
    }

    #[test]
    fn ids_are_random_and_only_well_formed_ones_are_read() {
        let (a, b) = (new_response_id().unwrap(), new_response_id().unwrap());
        assert_ne!(a, b);
        assert!(valid_id(&a), "{a}");
        for bad in ["resp_../../etc/passwd", "resp_", "file-abc", &a[..20]] {
            assert!(!valid_id(bad), "{bad}");
        }
    }

    #[test]
    fn a_continued_response_is_kept_deleted_until_nothing_continues_it() {
        let s = store();
        let (a, b) = (new_response_id().unwrap(), new_response_id().unwrap());
        s.put(&rec(&a, None), &body()).unwrap();
        s.put(&rec(&b, Some(&a)), &body()).unwrap();
        assert!(s.delete(&a).unwrap());
        assert!(s.get(&a).is_none(), "deleted for callers");
        assert!(s.record(&a).is_some_and(|r| r.deleted) && s.body(&a).is_some(), "kept for b's history");
        assert!(!s.delete(&a).unwrap(), "a deleted response is not deleted twice");
        assert!(s.delete(&b).unwrap());
        assert!(s.record(&a).is_none() && s.record(&b).is_none(), "both gone once unreferenced");
        assert!(s.body(&a).is_none());
    }
}
