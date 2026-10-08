use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::DaemonError;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct GenParams {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub seed: u64,
}

pub mod role {
    pub const SYSTEM: u32 = 0;
    pub const USER: u32 = 1;
    pub const ASSISTANT: u32 = 2;
    pub const TOOL: u32 = 3;
}

pub mod channel {
    pub const TEXT: u32 = 0;
    pub const REASONING: u32 = 1;
    pub const TOOL_CALL: u32 = 2;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventBody {
    Created { parent: Option<u64>, params: GenParams },
    Appended { text: Option<String>, span: Vec<u32> },
    Message { role: u32, text: String, span: Vec<u32> },
    GenerationPrompt { span: Vec<u32> },
    Generated { span: Vec<u32>, text: String, channel: u32, finish: u32 },
    ToolUse { name: String, arguments: String },
    ToolResult { call_id: u64, content: String, span: Vec<u32> },
    EpochBump,
    GenerationFingerprint { digest: u64 },
    ToolParseFailure { raw: String },
    Forked { parent: u64, fork_at: u64, params: GenParams },
    MetaUpdated { version: u64, title: Option<String>, archived: Option<bool> },
    Rebased { parent: u64, fork_at: u64, params: GenParams, edits: Vec<RebaseEdit> },
    ToolExpired { call_id: u64, reason: String },
    Purged { generation: u64, descendants: PurgeMode },
    Rerooted { purged_parent: u64, prefix: Vec<(u64, EventBody)> },
    QosSet { class: u8, batch_invariant: bool },
    Block { role: u32, kind: u32, payload: String, span: Vec<u32>, visibility_version: u32 },
    PermissionRequest { call_id: u64, text: String },
    PermissionResponse { request_id: u64, granted: bool },
    ToolCancelRequested { call_id: u64 },
    ToolOutcome { call_id: u64, outcome: u8, note: String },
    ToolReconciliation { call_id: u64, note: String },
    ToolLease { call_id: u64, deadline_unix_ms: u64 },
    Pinned { deadline_unix_ms: u64 },
}

impl EventBody {
    pub fn is_stream_neutral(&self) -> bool {
        matches!(
            self,
            EventBody::EpochBump
                | EventBody::GenerationFingerprint { .. }
                | EventBody::QosSet { .. }
                | EventBody::Pinned { .. }
                | EventBody::MetaUpdated { .. }
        )
    }

    pub fn span(&self) -> Option<&[u32]> {
        match self {
            EventBody::Appended { span, .. }
            | EventBody::Message { span, .. }
            | EventBody::GenerationPrompt { span }
            | EventBody::Generated { span, .. }
            | EventBody::ToolResult { span, .. }
            | EventBody::Block { span, .. } => Some(span),
            _ => None,
        }
    }
}

pub mod block_kind {
    pub const FILE_DIFF: u32 = 1;
    pub const ARTIFACT: u32 = 2;
    pub const CITATION: u32 = 3;
    pub const PROGRESS: u32 = 4;
    pub const DIAGNOSTIC: u32 = 5;
    pub const WORKSPACE_REF: u32 = 6;
    pub const IMAGE: u32 = 7;
}

pub mod tool_outcome {
    pub const FAILED: u8 = 1;
    pub const CANCELLED: u8 = 2;
}

pub mod ledger_state {
    pub const OPEN: u8 = 0;
    pub const CANCEL_REQUESTED: u8 = 1;
    pub const SUCCEEDED: u8 = 2;
    pub const FAILED: u8 = 3;
    pub const CANCELLED: u8 = 4;
    pub const EXPIRED: u8 = 5;
    pub const CANCELLED_WITH_LATE_EFFECT: u8 = 6;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RebaseEdit {
    Drop { from: u64, to: u64 },
    Replace { from: u64, to: u64, role: u32, text: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PurgeMode {
    Cascade,
    Reroot,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WalRecord {
    pub session: u64,
    pub event_id: u64,
    pub epoch: u64,
    pub ts_unix_ms: u64,
    pub body: EventBody,
}

#[derive(Serialize, Deserialize)]
struct WalRecordV1 {
    session: u64,
    event_id: u64,
    epoch: u64,
    body: EventBody,
}

#[derive(Serialize, Deserialize)]
struct WalRecordV2 {
    session: u64,
    event_id: u64,
    epoch: u64,
    ts_unix_ms: u64,
    body: EventBody,
}

pub const WAL_MAGIC_V2: &[u8; 8] = b"BRTWAL02";

pub fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub struct Wal {
    file: Option<File>,
    version: u8,
}

impl Wal {
    pub fn ephemeral() -> Wal {
        Wal {
            file: None,
            version: 2,
        }
    }

    pub fn open(path: &Path, mut apply: impl FnMut(WalRecord)) -> Result<Wal, DaemonError> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let len = file.metadata()?.len();
        let mut reader = BufReader::new(&mut file);
        let mut good_end: u64 = 0;
        let version = if len == 0 {
            2
        } else {
            let mut head = [0u8; 8];
            let mut n = 0;
            while n < 8 {
                let k = reader.read(&mut head[n..])?;
                if k == 0 {
                    break;
                }
                n += k;
            }
            if n == 8 && &head == WAL_MAGIC_V2 {
                good_end = 8;
                2
            } else if n < 8 && head[..n] == WAL_MAGIC_V2[..n] {
                good_end = 0;
                2
            } else {
                drop(reader);
                file.seek(SeekFrom::Start(0))?;
                reader = BufReader::new(&mut file);
                1
            }
        };
        loop {
            if version == 2 && good_end == 0 {
                break;
            }
            match read_record(&mut reader, version) {
                Ok(Frame::Record(record, consumed)) => {
                    good_end += consumed;
                    apply(record);
                }
                Ok(Frame::End) => break,
                // Only the last append can be torn (each is fsynced before the
                // next), so a failed frame is that append only if nothing but
                // zero fill follows it.
                Ok(Frame::Failed) => {
                    if !only_zeros(&mut reader)? {
                        return Err(DaemonError::WalCorrupt);
                    }
                    break;
                }
                Err(DaemonError::Io(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break
                }
                Err(e) => return Err(e),
            }
        }
        drop(reader);
        if good_end < len {
            tracing::warn!(
                wal = %path.display(),
                dropped_bytes = len - good_end,
                "the session log ended in a torn record; dropped it"
            );
            file.set_len(good_end)?;
        }
        if len == 0 {
            file.write_all(WAL_MAGIC_V2)?;
            file.sync_data()?;
        } else if version == 2 && good_end < 8 {
            file.set_len(0)?;
            file.write_all(WAL_MAGIC_V2)?;
            file.sync_data()?;
        }
        file.seek(SeekFrom::End(0))?;
        Ok(Wal { file: Some(file), version })
    }

    pub fn version(&self) -> u8 {
        self.version
    }

    pub fn append(&mut self, record: &WalRecord) -> Result<(), DaemonError> {
        let Some(file) = self.file.as_mut() else {
            return Ok(());
        };
        let payload = if self.version == 1 {
            postcard::to_stdvec(&WalRecordV1 {
                session: record.session,
                event_id: record.event_id,
                epoch: record.epoch,
                body: record.body.clone(),
            })?
        } else {
            postcard::to_stdvec(&WalRecordV2 {
                session: record.session,
                event_id: record.event_id,
                epoch: record.epoch,
                ts_unix_ms: record.ts_unix_ms,
                body: record.body.clone(),
            })?
        };
        let mut frame = Vec::with_capacity(12 + payload.len());
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&xxhash_rust::xxh3::xxh3_64(&payload).to_le_bytes());
        frame.extend_from_slice(&payload);
        file.write_all(&frame)?;
        durable_flush(file)?;
        Ok(())
    }
}

pub(crate) fn durable_flush(file: &std::fs::File) -> std::io::Result<()> {
    if std::env::var_os("SUPERFLUID_FULLFSYNC").is_some() {
        return file.sync_data();
    }
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        // SAFETY: `file` is a live, open WAL handle for the duration of the call.
        let rc = unsafe { libc::fsync(file.as_raw_fd()) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        file.sync_data()
    }
}

enum Frame {
    Record(WalRecord, u64),
    End,
    Failed,
}

fn only_zeros(r: &mut impl Read) -> std::io::Result<bool> {
    let mut buf = [0u8; 8192];
    loop {
        match r.read(&mut buf) {
            Ok(0) => return Ok(true),
            Ok(n) if buf[..n].iter().any(|&b| b != 0) => return Ok(false),
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

fn read_record(r: &mut impl Read, version: u8) -> Result<Frame, DaemonError> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(Frame::End),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_RECORD {
        return Err(DaemonError::WalCorrupt);
    }
    let mut sum_buf = [0u8; 8];
    r.read_exact(&mut sum_buf)?;
    let expect = u64::from_le_bytes(sum_buf);
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    if xxhash_rust::xxh3::xxh3_64(&payload) != expect {
        return Ok(Frame::Failed);
    }
    let record = if version == 1 {
        let r: WalRecordV1 = postcard::from_bytes(&payload)?;
        WalRecord {
            session: r.session,
            event_id: r.event_id,
            epoch: r.epoch,
            ts_unix_ms: 0,
            body: r.body,
        }
    } else {
        let r: WalRecordV2 = postcard::from_bytes(&payload)?;
        WalRecord {
            session: r.session,
            event_id: r.event_id,
            epoch: r.epoch,
            ts_unix_ms: r.ts_unix_ms,
            body: r.body,
        }
    };
    Ok(Frame::Record(record, 12 + len as u64))
}

const MAX_RECORD: usize = 1 << 30;
