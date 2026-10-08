//! The state envelope.

use superfluid_abi::{checksum_kind, StateEnvelope, TokenRange, Status, STATE_ENVELOPE_MAGIC};

pub const ENVELOPE_VERSION: u32 = 1;
pub const ENVELOPE_BYTES: usize = std::mem::size_of::<StateEnvelope>();

pub fn compat_identity(kind: u32, version_tag: u64, runtime: &str, version: &str, weights: &[u8; 32]) -> [u8; 32] {
    let mut input = Vec::with_capacity(64 + runtime.len() + version.len());
    input.extend_from_slice(&kind.to_le_bytes());
    input.extend_from_slice(&version_tag.to_le_bytes());
    input.extend_from_slice(runtime.as_bytes());
    input.push(0);
    input.extend_from_slice(version.as_bytes());
    input.push(0);
    input.extend_from_slice(weights);
    let h = superfluid_fingerprint::fnv1a64(&input, 0xCBF2_9CE4_8422_2325);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&h.to_le_bytes());
    out[8..16].copy_from_slice(&superfluid_fingerprint::fnv1a64(&input, h).to_le_bytes());
    out
}

pub struct SealArgs<'a> {
    pub kind: u32,
    pub version_tag: u64,
    pub compat: [u8; 32],
    pub provenance: [u8; 32],
    pub taint_bits: u32,
    pub range: TokenRange,
    pub encoding: u8,
    pub payload: &'a [u8],
}

pub fn seal(a: SealArgs<'_>) -> Vec<u8> {
    let env = StateEnvelope {
        magic: STATE_ENVELOPE_MAGIC,
        envelope_version: ENVELOPE_VERSION,
        space_kind: a.kind,
        compat_identity: a.compat,
        version_tag: a.version_tag,
        provenance_digest: a.provenance,
        taint_bits: a.taint_bits,
        _pad0: 0,
        range: a.range,
        encoding: a.encoding,
        _pad1: [0; 3],
        checksum_kind: checksum_kind::FNV1A64,
        payload_len: a.payload.len() as u64,
        content_checksum: superfluid_fingerprint::fnv1a64(a.payload, 0xCBF2_9CE4_8422_2325),
    };
    let mut out = Vec::with_capacity(ENVELOPE_BYTES + a.payload.len());
    // SAFETY: StateEnvelope is #[repr(C)] with explicit padding;
    // viewing it as bytes is well-defined.
    let env_bytes = unsafe { std::slice::from_raw_parts(&env as *const _ as *const u8, ENVELOPE_BYTES) };
    out.extend_from_slice(env_bytes);
    out.extend_from_slice(a.payload);
    out
}

pub fn open<'a>(
    bytes: &'a [u8],
    kind: u32,
    version_tag: u64,
    compat: &[u8; 32],
) -> Result<(StateEnvelope, &'a [u8]), Status> {
    if bytes.len() < ENVELOPE_BYTES {
        return Err(Status::EnvelopeMismatch);
    }
    let mut env = StateEnvelope::default();
    // SAFETY: the source holds >= ENVELOPE_BYTES; the type accepts any bit
    // pattern.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), &mut env as *mut _ as *mut u8, ENVELOPE_BYTES) };
    if env.magic != STATE_ENVELOPE_MAGIC || env.envelope_version != ENVELOPE_VERSION {
        return Err(Status::EnvelopeMismatch);
    }
    if env.space_kind != kind {
        return Err(Status::EnvelopeMismatch);
    }
    if env.compat_identity != *compat || env.version_tag != version_tag {
        return Err(Status::IdentityMismatch);
    }
    let payload = &bytes[ENVELOPE_BYTES..];
    if payload.len() as u64 != env.payload_len {
        return Err(Status::EnvelopeMismatch);
    }
    match env.checksum_kind {
        checksum_kind::FNV1A64 => {
            if superfluid_fingerprint::fnv1a64(payload, 0xCBF2_9CE4_8422_2325) != env.content_checksum {
                return Err(Status::Checksum);
            }
        }
        _ => return Err(Status::EnvelopeMismatch),
    }
    Ok((env, payload))
}
