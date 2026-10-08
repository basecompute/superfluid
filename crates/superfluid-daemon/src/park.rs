//! Park artifacts.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"BRTPARK2";
const MAGIC_V3: &[u8; 8] = b"BRTPARK3";
const HEADER: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParkedSpace {
    pub space_id: u32,
    pub kind: u32,
    pub encoding: u8,
    pub sealed: Vec<u8>,
}

pub struct Parked {
    pub covered: u64,
    pub stream_digest: u64,
    pub encoding: u8,
    pub sealed: Vec<u8>,
    pub spaces: Vec<ParkedSpace>,
    pub multi_space: bool,
}

pub fn stream_digest(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

fn artifact_path(dir: &Path, session: u64) -> PathBuf {
    dir.join(format!("{session}.park"))
}

pub fn write(
    dir: &Path,
    session: u64,
    covered: u64,
    digest: u64,
    encoding: u8,
    sealed: &[u8],
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!("{session}.park.tmp"));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(MAGIC)?;
        f.write_all(&covered.to_le_bytes())?;
        f.write_all(&digest.to_le_bytes())?;
        f.write_all(&(encoding as u64).to_le_bytes())?;
        f.write_all(sealed)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, artifact_path(dir, session))
}

pub fn write_v3(
    dir: &Path,
    session: u64,
    covered: u64,
    digest: u64,
    spaces: &[ParkedSpace],
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!("{session}.park.tmp"));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(MAGIC_V3)?;
        f.write_all(&covered.to_le_bytes())?;
        f.write_all(&digest.to_le_bytes())?;
        f.write_all(&(spaces.len() as u32).to_le_bytes())?;
        f.write_all(&[0u8; 4])?;
        for sp in spaces {
            f.write_all(&sp.space_id.to_le_bytes())?;
            f.write_all(&sp.kind.to_le_bytes())?;
            f.write_all(&[sp.encoding, 0, 0, 0, 0, 0, 0, 0])?;
            f.write_all(&(sp.sealed.len() as u64).to_le_bytes())?;
            f.write_all(&sp.sealed)?;
        }
        f.sync_all()?;
    }
    std::fs::rename(&tmp, artifact_path(dir, session))
}

pub fn read(dir: &Path, session: u64) -> Option<Parked> {
    let mut f = std::fs::File::open(artifact_path(dir, session)).ok()?;
    let mut head = [0u8; HEADER];
    f.read_exact(&mut head).ok()?;
    let covered = u64::from_le_bytes(head[8..16].try_into().expect("8 bytes"));
    let stream_digest = u64::from_le_bytes(head[16..24].try_into().expect("8 bytes"));
    if &head[0..8] == MAGIC {
        let encoding = u64::from_le_bytes(head[24..32].try_into().expect("8 bytes"));
        let mut sealed = Vec::new();
        f.read_to_end(&mut sealed).ok()?;
        if covered == 0 || sealed.is_empty() || encoding > u8::MAX as u64 {
            return None;
        }
        return Some(Parked {
            covered,
            stream_digest,
            encoding: encoding as u8,
            sealed: sealed.clone(),
            spaces: vec![ParkedSpace {
                space_id: 1,
                kind: superfluid_abi::space_kind::PAGED_TOKEN_KV,
                encoding: encoding as u8,
                sealed,
            }],
            multi_space: false,
        });
    }
    if &head[0..8] != MAGIC_V3 {
        return None;
    }
    let n_spaces = u32::from_le_bytes(head[24..28].try_into().expect("4 bytes"));
    if covered == 0 || n_spaces == 0 || n_spaces > 64 {
        return None;
    }
    let mut rest = Vec::new();
    f.read_to_end(&mut rest).ok()?;
    let mut at = 0usize;
    let mut spaces = Vec::with_capacity(n_spaces as usize);
    for _ in 0..n_spaces {
        let hdr_end = at.checked_add(24)?;
        if rest.len() < hdr_end {
            return None;
        }
        let space_id = u32::from_le_bytes(rest[at..at + 4].try_into().ok()?);
        let kind = u32::from_le_bytes(rest[at + 4..at + 8].try_into().ok()?);
        let encoding = rest[at + 8];
        let len = u64::from_le_bytes(rest[at + 16..at + 24].try_into().ok()?) as usize;
        at = hdr_end;
        let end = at.checked_add(len)?;
        if rest.len() < end || len == 0 {
            return None;
        }
        spaces.push(ParkedSpace {
            space_id,
            kind,
            encoding,
            sealed: rest[at..end].to_vec(),
        });
        at = end;
    }
    if at != rest.len() {
        return None;
    }
    let kv = spaces
        .iter()
        .find(|s| s.kind == superfluid_abi::space_kind::PAGED_TOKEN_KV)
        .cloned();
    Some(Parked {
        covered,
        stream_digest,
        encoding: kv.as_ref().map(|k| k.encoding).unwrap_or(0),
        sealed: kv.map(|k| k.sealed).unwrap_or_default(),
        spaces,
        multi_space: true,
    })
}

pub fn remove(dir: &Path, session: u64) {
    let _ = std::fs::remove_file(artifact_path(dir, session));
    let _ = std::fs::remove_file(dir.join(format!("{session}.park.tmp")));
}

pub fn sweep(dir: &Path, max_bytes: u64) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    let is_artifact = |p: &std::path::Path| {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.ends_with(".park") || name.ends_with(".park.tmp")
    };
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter(|e| is_artifact(&e.path()))
        .filter_map(|e| {
            let md = e.metadata().ok()?;
            Some((md.modified().ok()?, md.len(), e.path()))
        })
        .collect();

    let mut total: u64 = files.iter().map(|(_, len, _)| *len).sum();
    if total <= max_bytes {
        return 0;
    }
    files.sort_by_key(|(mtime, _, _)| *mtime);

    let mut reclaimed = 0u64;
    for (_, len, path) in files {
        if total <= max_bytes {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(len);
            reclaimed += len;
        }
    }
    reclaimed
}
