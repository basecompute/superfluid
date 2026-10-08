//! File store for the OpenAI `/v1/files` API (the substrate for `/v1/batches`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::DaemonError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMeta {
    pub id: String,
    #[serde(default = "file_object")]
    pub object: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
}

fn file_object() -> String {
    "file".to_string()
}

pub struct FileStore {
    dir: PathBuf,
    max_bytes: Option<u64>,
    expiry_secs: Option<u64>,
    admit: Mutex<()>,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl FileStore {
    pub fn new(dir: PathBuf) -> FileStore {
        Self::with_limits(dir, None, None)
    }

    pub fn with_limits(
        dir: PathBuf,
        max_bytes: Option<u64>,
        expiry_secs: Option<u64>,
    ) -> FileStore {
        let _ = std::fs::create_dir_all(&dir);
        FileStore { dir, max_bytes, expiry_secs, admit: Mutex::new(()) }
    }

    pub fn stored_bytes(&self) -> u64 {
        self.list().iter().map(|m| m.bytes).sum()
    }

    pub fn sweep(&self) -> usize {
        let Some(max_age) = self.expiry_secs else {
            return 0;
        };
        let now = now_secs();
        let mut removed = 0;
        for m in self.list() {
            if now.saturating_sub(m.created_at) > max_age && self.delete(&m.id) {
                removed += 1;
            }
        }
        removed
    }

    fn new_id() -> String {
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        format!("file-{:x}{:x}", nanos, N.fetch_add(1, Ordering::Relaxed))
    }

    fn meta_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }
    fn blob_path(&self, id: &str) -> PathBuf {
        self.dir.join(id)
    }

    fn valid_id(id: &str) -> bool {
        id.starts_with("file-") && id.len() > 5 && id[5..].chars().all(|c| c.is_ascii_hexdigit())
    }

    pub fn put(
        &self,
        filename: &str,
        purpose: &str,
        bytes: &[u8],
        owner: Option<&str>,
    ) -> Result<FileMeta, DaemonError> {
        let _admit = self.max_bytes.map(|_| self.admit.lock().expect("file store"));
        if let Some(max) = self.max_bytes {
            self.sweep();
            let stored = self.stored_bytes();
            let incoming = bytes.len() as u64;
            if stored.saturating_add(incoming) > max {
                return Err(DaemonError::FilesQuota { stored, incoming, max });
            }
        }
        let id = Self::new_id();
        std::fs::write(self.blob_path(&id), bytes).map_err(DaemonError::Io)?;
        let meta = FileMeta {
            id: id.clone(),
            object: "file".to_string(),
            bytes: bytes.len() as u64,
            created_at: now_secs(),
            filename: filename.to_string(),
            purpose: purpose.to_string(),
            owner: owner.map(str::to_string),
        };
        let json = serde_json::to_vec(&meta).map_err(|_| DaemonError::Protocol("file meta encode"))?;
        std::fs::write(self.meta_path(&id), json).map_err(DaemonError::Io)?;
        Ok(meta)
    }

    pub fn get(&self, id: &str) -> Option<FileMeta> {
        if !Self::valid_id(id) {
            return None;
        }
        let bytes = std::fs::read(self.meta_path(id)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    pub fn content(&self, id: &str) -> Option<Vec<u8>> {
        if !Self::valid_id(id) {
            return None;
        }
        std::fs::read(self.blob_path(id)).ok()
    }

    pub fn list(&self) -> Vec<FileMeta> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("json") {
                    if let Ok(b) = std::fs::read(&p) {
                        if let Ok(m) = serde_json::from_slice::<FileMeta>(&b) {
                            out.push(m);
                        }
                    }
                }
            }
        }
        out.sort_by_key(|m| std::cmp::Reverse(m.created_at));
        out
    }

    pub fn delete(&self, id: &str) -> bool {
        if !Self::valid_id(id) || self.get(id).is_none() {
            return false;
        }
        let _ = std::fs::remove_file(self.blob_path(id));
        let _ = std::fs::remove_file(self.meta_path(id));
        true
    }
}

pub fn files_dir_for(sessions_dir: &Path) -> PathBuf {
    sessions_dir.join("files")
}
