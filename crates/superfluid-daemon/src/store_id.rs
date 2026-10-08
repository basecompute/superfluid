//! A session store's persisted identity; see [`StoreId`].

use std::path::Path;

use crate::DaemonError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct StoreId(pub [u8; 16]);

impl StoreId {
    pub fn mint() -> StoreId {
        let mut b = [0u8; 16];
        // SAFETY: `b` is a valid, writable 16-byte buffer; getentropy
        // writes at most `len` (16 <= 256, its per-call limit) bytes.
        let rc = unsafe { libc::getentropy(b.as_mut_ptr().cast(), b.len()) };
        if rc != 0 {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let seed = [
                t.to_le_bytes().as_slice(),
                &std::process::id().to_le_bytes(),
                &N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    .to_le_bytes(),
            ]
            .concat();
            b = xxhash_rust::xxh3::xxh3_128(&seed).to_le_bytes();
        }
        StoreId(b)
    }

    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn from_hex(s: &str) -> Option<StoreId> {
        let s = s.trim();
        if s.len() != 32 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let mut b = [0u8; 16];
        for (i, out) in b.iter_mut().enumerate() {
            *out = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(StoreId(b))
    }

    pub fn to_file_line(&self) -> String {
        format!("{} {:016x}\n", self.to_hex(), Self::checksum(&self.0))
    }

    fn checksum(b: &[u8; 16]) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(b, 0x5354_4f52_4549_4421)
    }

    pub fn from_file_line(text: &str) -> Option<StoreId> {
        let mut parts = text.split_whitespace();
        let (id, sum) = (parts.next()?, parts.next()?);
        if parts.next().is_some() {
            return None;
        }
        let id = StoreId::from_hex(id)?;
        let sum = u64::from_str_radix(sum, 16)
            .ok()
            .filter(|_| sum.len() == 16)?;
        (Self::checksum(&id.0) == sum).then_some(id)
    }

    pub fn load_or_mint(path: &Path) -> Result<StoreId, DaemonError> {
        let lock = StoreId::lock(path)?;
        Self::open(path, false, &lock)
    }

    pub fn lock(path: &Path) -> Result<StoreIdLock, DaemonError> {
        Ok(StoreIdLock::acquire(path, libc::LOCK_EX)?.expect("a blocking flock returns held"))
    }

    pub fn try_lock(path: &Path) -> Result<Option<StoreIdLock>, DaemonError> {
        StoreIdLock::acquire(path, libc::LOCK_EX | libc::LOCK_NB)
    }

    pub fn open(path: &Path, fresh_wal: bool, _lock: &StoreIdLock) -> Result<StoreId, DaemonError> {
        let dir = path
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let existing = Self::load(path)?;
        if let (false, Some(id)) = (fresh_wal, existing) {
            sync_dir(dir)?;
            return Ok(id);
        }
        let id = StoreId::mint();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("store-id");
        let tmp = TempFile::write(dir, name, &id)?;
        let id = if existing.is_some() {
            std::fs::rename(&tmp.path, path)?;
            id
        } else {
            match std::fs::hard_link(&tmp.path, path) {
                Ok(()) => id,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Self::load(path)?
                    .ok_or(DaemonError::Protocol(
                        "store id vanished while being published",
                    ))?,
                Err(e) => return Err(e.into()),
            }
        };
        drop(tmp);
        sync_dir(dir)?;
        Ok(id)
    }

    fn load(path: &Path) -> Result<Option<StoreId>, DaemonError> {
        match std::fs::read_to_string(path) {
            Ok(text) => StoreId::from_file_line(&text)
                .map(Some)
                .ok_or(DaemonError::Config(
                    "the session store's id file is corrupt; it only names the store's \
                     trace ids — delete it to mint a new one (exported trace ids change)",
                )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

pub struct StoreIdLock {
    _file: std::fs::File,
}

impl StoreIdLock {
    fn acquire(id_path: &Path, op: libc::c_int) -> Result<Option<StoreIdLock>, DaemonError> {
        use std::os::fd::AsRawFd;
        let mut name = id_path
            .file_name()
            .map(|n| n.to_os_string())
            .unwrap_or_default();
        name.push(".lock");
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(id_path.with_file_name(name))?;
        loop {
            // SAFETY: `file` is an open descriptor owned by this function
            // for the duration of the call; flock only takes its number.
            let rc = unsafe { libc::flock(file.as_raw_fd(), op) };
            if rc == 0 {
                return Ok(Some(StoreIdLock { _file: file }));
            }
            let err = std::io::Error::last_os_error();
            match err.kind() {
                std::io::ErrorKind::Interrupted => {}
                std::io::ErrorKind::WouldBlock => return Ok(None),
                _ => return Err(err.into()),
            }
        }
    }
}

fn sync_dir(dir: &Path) -> Result<(), DaemonError> {
    std::fs::File::open(dir)?.sync_all()?;
    Ok(())
}

struct TempFile {
    path: std::path::PathBuf,
}

impl TempFile {
    fn write(dir: &Path, name: &str, id: &StoreId) -> Result<TempFile, DaemonError> {
        use std::io::Write;
        let path = dir.join(format!(
            ".{name}.tmp.{}.{}",
            std::process::id(),
            id.to_hex()
        ));
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let tmp = TempFile { path };
        f.write_all(id.to_file_line().as_bytes())?;
        f.sync_all()?;
        Ok(tmp)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn store_id_path(wal: &Path) -> std::path::PathBuf {
    let mut name = wal
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".store-id");
    wal.with_file_name(name)
}
