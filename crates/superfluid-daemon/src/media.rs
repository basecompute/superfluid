//! The content-addressed media pool.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub fn content_hash(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

pub struct MediaPool {
    dir: PathBuf,
}

impl MediaPool {
    pub fn new(dir: PathBuf) -> MediaPool {
        MediaPool { dir }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn is_valid_id(id: &str) -> bool {
        id.len() == 64 && id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    pub fn path_of(&self, hash: &str) -> PathBuf {
        self.dir.join(hash)
    }

    pub fn resolve(&self, id: &str) -> Option<PathBuf> {
        if !Self::is_valid_id(id) {
            return None;
        }
        let p = self.path_of(id);
        p.is_file().then_some(p)
    }

    pub fn put(&self, bytes: &[u8]) -> std::io::Result<String> {
        let hash = content_hash(bytes);
        let path = self.path_of(&hash);
        if path.exists() {
            return Ok(hash);
        }
        std::fs::create_dir_all(&self.dir)?;
        let tmp = self.dir.join(format!(".{hash}.tmp{}", std::process::id()));
        {
            let mut o = std::fs::OpenOptions::new();
            o.create(true).write(true).truncate(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                o.mode(0o600);
            }
            let mut f = o.open(&tmp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        if let Ok(d) = std::fs::File::open(&self.dir) {
            let _ = d.sync_all();
        }
        Ok(hash)
    }

    pub fn exists(&self, hash: &str) -> bool {
        Self::is_valid_id(hash) && self.path_of(hash).is_file()
    }

    pub fn sweep(&self, live: &HashSet<String>) -> std::io::Result<(usize, usize)> {
        let mut kept = 0;
        let mut removed = 0;
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return Ok((0, 0));
        };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if live.contains(&name) {
                kept += 1;
            } else {
                let _ = std::fs::remove_file(e.path());
                removed += 1;
            }
        }
        Ok((kept, removed))
    }
}

#[cfg(test)]
mod tests {
    use super::MediaPool;

    #[test]
    fn valid_id_is_64_lowercase_hex() {
        assert!(MediaPool::is_valid_id(&"a".repeat(64)));
        assert!(MediaPool::is_valid_id(&"0123456789abcdef".repeat(4)));
        assert!(!MediaPool::is_valid_id(&"A".repeat(64)));
        assert!(!MediaPool::is_valid_id(&"a".repeat(63)));
        assert!(!MediaPool::is_valid_id("../../etc/passwd"));
        assert!(!MediaPool::is_valid_id("/etc/passwd"));
        assert!(!MediaPool::is_valid_id(&format!("{}g", "a".repeat(63))));
    }

    #[test]
    fn exists_and_resolve_reject_traversal_ids() {
        let dir = std::env::temp_dir().join(format!("superfluid-media-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pool = MediaPool::new(dir);
        assert!(!pool.exists("../../../../etc/hosts"));
        assert!(!pool.exists("/etc/hosts"));
        assert!(pool.resolve("/etc/hosts").is_none());
        assert!(pool.resolve("../secrets").is_none());
    }
}
