//! Async state ops.

use superfluid_abi::{
    checksum_kind, encoding as enc, space_kind, StateEnvelope, TokenRange, Status,
    STATE_ENVELOPE_MAGIC,
};

use super::state::{SpaceState, StagedImport};

pub const ENVELOPE_VERSION: u32 = 1;
pub const ENVELOPE_BYTES: usize = std::mem::size_of::<StateEnvelope>();

pub fn compat_identity(kind: u32, version_tag: u64, bundle_fp: u64) -> [u8; 32] {
    let mut input = [0u8; 20];
    input[..4].copy_from_slice(&kind.to_le_bytes());
    input[4..12].copy_from_slice(&version_tag.to_le_bytes());
    input[12..20].copy_from_slice(&bundle_fp.to_le_bytes());
    let h = xxhash_rust::xxh3::xxh3_64(&input);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&h.to_le_bytes());
    out
}

pub fn seal(
    space: &SpaceState,
    bundle_fp: u64,
    range: TokenRange,
    encoding: u8,
    payload: &[u8],
) -> Vec<u8> {
    let env = StateEnvelope {
        magic: STATE_ENVELOPE_MAGIC,
        envelope_version: ENVELOPE_VERSION,
        space_kind: space.kind,
        compat_identity: compat_identity(space.kind, space.version_tag, bundle_fp),
        version_tag: space.version_tag,
        provenance_digest: space.provenance,
        taint_bits: space.taint_bits,
        _pad0: 0,
        range,
        encoding,
        _pad1: [0; 3],
        checksum_kind: checksum_kind::XXH3_64,
        payload_len: payload.len() as u64,
        content_checksum: xxhash_rust::xxh3::xxh3_64(payload),
    };
    let mut out = Vec::with_capacity(ENVELOPE_BYTES + payload.len());
    // SAFETY: StateEnvelope is #[repr(C)] with explicit padding;
    // viewing it as bytes is well-defined.
    let env_bytes =
        unsafe { std::slice::from_raw_parts(&env as *const _ as *const u8, ENVELOPE_BYTES) };
    out.extend_from_slice(env_bytes);
    out.extend_from_slice(payload);
    out
}

pub fn open<'a>(
    bytes: &'a [u8],
    target: &SpaceState,
    bundle_fp: u64,
) -> Result<(StateEnvelope, &'a [u8]), Status> {
    if bytes.len() < ENVELOPE_BYTES {
        return Err(Status::EnvelopeMismatch);
    }
    let mut env = StateEnvelope::default();
    // SAFETY: source has >= ENVELOPE_BYTES readable bytes; the type
    // accepts any bit pattern.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            &mut env as *mut _ as *mut u8,
            ENVELOPE_BYTES,
        );
    }
    if env.magic != STATE_ENVELOPE_MAGIC || env.envelope_version != ENVELOPE_VERSION {
        return Err(Status::EnvelopeMismatch);
    }
    if env.space_kind != target.kind {
        return Err(Status::EnvelopeMismatch);
    }
    if env.compat_identity != compat_identity(target.kind, target.version_tag, bundle_fp)
        || env.version_tag != target.version_tag
    {
        return Err(Status::IdentityMismatch);
    }
    let payload = &bytes[ENVELOPE_BYTES..];
    if payload.len() as u64 != env.payload_len {
        return Err(Status::EnvelopeMismatch);
    }
    match env.checksum_kind {
        checksum_kind::XXH3_64 => {
            if xxhash_rust::xxh3::xxh3_64(payload) != env.content_checksum {
                return Err(Status::Checksum);
            }
        }
        _ => return Err(Status::EnvelopeMismatch),
    }
    Ok((env, payload))
}

pub fn stage_import(env: &StateEnvelope, is_promote: bool) -> StagedImport {
    let mut taint = env.taint_bits;
    if env.encoding != enc::LOSSLESS {
        taint |= superfluid_abi::taint::QUANTIZED_DEMOTION;
    }
    StagedImport {
        valid_len: if is_promote { 0 } else { env.range.end },
        provenance: env.provenance_digest,
        taint_bits: taint,
        boundaries: if !is_promote
            && (env.space_kind == space_kind::RECURRENT_BLOB
                || env.space_kind == space_kind::RING_KV)
        {
            vec![env.range.end]
        } else {
            Vec::new()
        },
        promote_range: is_promote.then_some((env.range.start, env.range.end)),
        encoding: env.encoding,
    }
}

pub fn export_size(
    space: &SpaceState,
    bytes_per_token: u64,
    blob_bytes: u64,
    range: TokenRange,
    encoding: u8,
) -> Result<u64, Status> {
    if space.kind == space_kind::RECURRENT_BLOB && encoding != enc::LOSSLESS {
        return Err(Status::Unsupported);
    }
    if space.kind == space_kind::ENCODER_CACHE {
        return Err(Status::Unsupported);
    }
    if range.start > range.end || range.end > space.valid_len {
        return Err(Status::RejectBounds);
    }
    let payload = if space.is_blob_like() {
        blob_bytes
    } else {
        let tokens = range.end - range.start;
        let scale = match encoding {
            enc::LOSSLESS => 1.0,
            enc::Q8 => 0.5,
            enc::Q4 => 0.25,
            _ => return Err(Status::Unsupported),
        };
        ((tokens * bytes_per_token) as f64 * scale).ceil() as u64
    };
    Ok(ENVELOPE_BYTES as u64 + payload)
}
