pub const FNV_BASIS: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

pub const COMPAT_RECIPE_VERSION: u32 = 2;

pub fn fnv1a64(bytes: &[u8], basis: u64) -> u64 {
    let mut h = basis;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceCompat<'a> {
    pub weights_identity: u64,
    pub architecture: &'a [u8],
    pub n_layers: u32,
    pub kv_dim: u32,
    pub head_dim: u32,
    pub page_size_tokens: u32,
    pub layout_flags: u32,
}

pub const LAYOUT_DSA_KEYS_IN_V: u32 = 1;

impl SpaceCompat<'_> {
    pub fn version_tag(&self) -> u64 {
        let mut input = COMPAT_RECIPE_VERSION.to_le_bytes().to_vec();
        input.extend_from_slice(&self.weights_identity.to_le_bytes());
        input.extend_from_slice(self.architecture);
        for v in [self.n_layers, self.kv_dim, self.head_dim, self.page_size_tokens] {
            input.extend_from_slice(&v.to_le_bytes());
        }
        if self.layout_flags != 0 {
            input.extend_from_slice(&self.layout_flags.to_le_bytes());
        }
        fnv1a64(&input, FNV_BASIS)
    }
}

pub fn content_digest(tokens: &[u32]) -> [u8; 32] {
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let h1 = fnv1a64(&bytes, FNV_BASIS);
    let h2 = fnv1a64(&bytes, !FNV_BASIS);
    let mut d = [0u8; 32];
    d[0..8].copy_from_slice(&h1.to_le_bytes());
    d[8..16].copy_from_slice(&h2.to_le_bytes());
    d
}

pub mod taint {
    pub const LOSSY_DEMOTION: u32 = 1 << 0;
    pub const DECOMP_CHANGING_RESTORE: u32 = 1 << 1;
}

pub mod op {
    pub const PREFILL: u32 = 1;
    pub const DECODE: u32 = 2;
    pub const SEED: u32 = 3;
    pub const RESTORE: u32 = 4;
    pub const PROMOTE: u32 = 5;
    pub const FORK: u32 = 6;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lineage {
    pub digest: u64,
    pub taint: u32,
}

impl Lineage {
    pub fn genesis() -> Lineage {
        Lineage {
            digest: FNV_BASIS,
            taint: 0,
        }
    }

    pub fn append(&self, op_kind: u32, detail: u64) -> Lineage {
        let mut msg = [0u8; 20];
        msg[0..8].copy_from_slice(&self.digest.to_le_bytes());
        msg[8..12].copy_from_slice(&op_kind.to_le_bytes());
        msg[12..20].copy_from_slice(&detail.to_le_bytes());
        Lineage {
            digest: fnv1a64(&msg, FNV_BASIS),
            taint: self.taint,
        }
    }

    pub fn append_tainted(&self, op_kind: u32, detail: u64, taint_bits: u32) -> Lineage {
        let mut next = self.append(op_kind, detail);
        next.taint |= taint_bits;
        next
    }
}

pub struct RecordHasher {
    h: u64,
}

impl RecordHasher {
    pub fn new() -> RecordHasher {
        RecordHasher { h: FNV_BASIS }
    }

    pub fn field(&mut self, tag: u8, bytes: &[u8]) -> &mut Self {
        self.h = fnv1a64(&[tag], self.h);
        self.h = fnv1a64(&(bytes.len() as u32).to_le_bytes(), self.h);
        self.h = fnv1a64(bytes, self.h);
        self
    }

    pub fn field_u32(&mut self, tag: u8, v: u32) -> &mut Self {
        self.field(tag, &v.to_le_bytes())
    }

    pub fn field_u64(&mut self, tag: u8, v: u64) -> &mut Self {
        self.field(tag, &v.to_le_bytes())
    }

    pub fn finish(&self) -> u64 {
        self.h
    }
}

impl Default for RecordHasher {
    fn default() -> Self {
        RecordHasher::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehaviorFingerprint {
    pub codec_id: String,
    pub codec_version: u32,
    pub tokenizer_hash: u64,
    pub marker_digest: u64,
    pub sampling_impl_version: u32,
    pub block_projection_version: u32,
    pub grammar_version: u32,
    pub speculation: Option<u64>,
}

impl BehaviorFingerprint {
    pub fn marker_digest_of(markers: &[(u32, u32, u32)]) -> u64 {
        let mut h = RecordHasher::new();
        for (i, (open, close, channel)) in markers.iter().enumerate() {
            let mut bytes = [0u8; 12];
            bytes[0..4].copy_from_slice(&open.to_le_bytes());
            bytes[4..8].copy_from_slice(&close.to_le_bytes());
            bytes[8..12].copy_from_slice(&channel.to_le_bytes());
            h.field(i as u8, &bytes);
        }
        h.finish()
    }

    pub fn digest(&self) -> u64 {
        let mut h = RecordHasher::new();
        h.field(1, self.codec_id.as_bytes())
            .field_u32(2, self.codec_version)
            .field_u64(3, self.tokenizer_hash)
            .field_u64(4, self.marker_digest)
            .field_u32(5, self.sampling_impl_version)
            .field_u32(6, self.block_projection_version)
            .field_u32(7, self.grammar_version);
        match self.speculation {
            Some(s) => h.field_u64(8, s),
            None => h.field(8, b""),
        };
        h.finish()
    }
}

pub fn vocab_hash(vocab: u32, mut token_bytes: impl FnMut(u32) -> Vec<u8>) -> u64 {
    let mut h = FNV_BASIS;
    for t in 0..vocab {
        let b = token_bytes(t);
        h = fnv1a64(&(b.len() as u32).to_le_bytes(), h);
        h = fnv1a64(&b, h);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_primitive_is_pinned() {
        assert_eq!(fnv1a64(b"", FNV_BASIS), 0xCBF2_9CE4_8422_2325);
        assert_eq!(fnv1a64(b"a", FNV_BASIS), 0xAF63_DC4C_8601_EC8C);
        assert_eq!(fnv1a64(b"foobar", FNV_BASIS), 0x8594_4171_F739_67E8);
    }

    #[test]
    fn space_compat_golden_vector() {
        let c = SpaceCompat {
            weights_identity: 0x1122_3344_5566_7788,
            architecture: b"qwen3",
            n_layers: 28,
            kv_dim: 1024,
            head_dim: 128,
            page_size_tokens: 16,
            layout_flags: 0,
        };
        assert_eq!(c.version_tag(), 0x311D_A57C_1A4B_B441);
        let unid = SpaceCompat {
            weights_identity: 0,
            ..c.clone()
        };
        assert_eq!(unid.version_tag(), 0x0076_E3BB_B2DA_E839);
        assert_ne!(unid.version_tag(), c.version_tag());
        let dsa = SpaceCompat {
            layout_flags: LAYOUT_DSA_KEYS_IN_V,
            ..c.clone()
        };
        assert_eq!(dsa.version_tag(), 0x00D1_60D4_4E80_1840);
        assert_ne!(dsa.version_tag(), c.version_tag());
    }

    #[test]
    fn content_digest_golden_vector() {
        let d = content_digest(&[1, 2, 3]);
        assert_eq!(u64::from_le_bytes(d[0..8].try_into().unwrap()), 0xFD1F_0F43_81EB_0395);
        assert_eq!(u64::from_le_bytes(d[8..16].try_into().unwrap()), 0x0D9D_6D1C_2346_C3BA);
        assert_eq!(d[16..32], [0u8; 16], "rest is zero (the C++ layout)");
    }

    #[test]
    fn lineage_is_ordered_and_taint_is_sticky() {
        let g = Lineage::genesis();
        assert_eq!(g.append(1, 7).digest, 0xE453_CC11_9B29_9C5A);

        let ab = g.append(op::PREFILL, 10).append(op::DECODE, 20);
        let ba = g.append(op::DECODE, 20).append(op::PREFILL, 10);
        assert_ne!(ab.digest, ba.digest);

        let t = g.append_tainted(op::RESTORE, 1, taint::LOSSY_DEMOTION);
        assert_eq!(t.append(op::DECODE, 1).taint, taint::LOSSY_DEMOTION);
        assert_eq!(Lineage::genesis().taint, 0);
    }

    #[test]
    fn record_hashing_resists_concatenation_ambiguity() {
        let a = RecordHasher::new().field(1, b"ab").field(2, b"c").finish();
        let b = RecordHasher::new().field(1, b"a").field(2, b"bc").finish();
        assert_ne!(a, b);
        let c = RecordHasher::new().field(2, b"ab").field(1, b"c").finish();
        assert_ne!(a, c, "field tags separate positions");
    }

    #[test]
    fn behavior_fingerprint_changes_with_every_field() {
        let base = BehaviorFingerprint {
            codec_id: "chatml".into(),
            codec_version: 1,
            tokenizer_hash: 0x1111,
            marker_digest: 0x2222,
            sampling_impl_version: 1,
            block_projection_version: 1,
            grammar_version: 0,
            speculation: None,
        };
        let d = base.digest();
        let mut m = base.clone();
        m.codec_version = 2;
        assert_ne!(m.digest(), d);
        let mut m = base.clone();
        m.tokenizer_hash = 0x1112;
        assert_ne!(m.digest(), d);
        let mut m = base.clone();
        m.speculation = Some(0);
        assert_ne!(m.digest(), d, "Some(0) is distinct from None");
        assert_eq!(base.clone().digest(), d, "digest is deterministic");
    }

    #[test]
    fn vocab_hash_is_order_and_content_sensitive() {
        let v1 = vocab_hash(2, |t| vec![t as u8]);
        let v2 = vocab_hash(2, |t| vec![1 - t as u8]);
        assert_ne!(v1, v2);
        let a = vocab_hash(2, |t| if t == 0 { b"aa".to_vec() } else { b"b".to_vec() });
        let b = vocab_hash(2, |t| if t == 0 { b"a".to_vec() } else { b"ab".to_vec() });
        assert_ne!(a, b);
    }
}
