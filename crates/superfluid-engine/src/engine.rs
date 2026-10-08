//! The [`Engine`] trait.

use superfluid_abi::{
    MatchResult, OpStatus, StateSpaceDesc, StrategyGrant,
    StrategyRegistration, TickEvents, TickPlan, TokenRange, Status,
};

use crate::rings::Rings;

pub type SeqHandle = u64;

pub type OpHandle = u64;

#[derive(Debug, Clone)]
pub struct SpaceConfig {
    pub space_id: u32,
    pub kind: u32,
    pub name: &'static str,
    pub version_tag: u64,
    pub bytes_per_token: u64,
    pub blob_bytes: u64,
    pub page_size_tokens: u32,
    pub snapshot_cadence: u32,
    pub snapshot_interval_tokens: u32,
    pub flags: u32,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub max_batch_size: u32,
    pub pool_bytes: u64,
    pub spaces: Vec<SpaceConfig>,
    pub seed_ttl_ticks: u64,
    pub vocab: u32,
    pub kernel_caps: u32,
    pub image_token_id: u32,
    pub media_tokens_per_image: u32,
    pub scripted: Vec<u32>,
    pub tick_delay: std::time::Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        use superfluid_abi::space_kind;
        EngineConfig {
            max_batch_size: 8,
            pool_bytes: 1 << 20,
            spaces: vec![
                SpaceConfig {
                    space_id: 1,
                    kind: space_kind::PAGED_TOKEN_KV,
                    name: "kv.full",
                    version_tag: 1,
                    bytes_per_token: 64,
                    blob_bytes: 0,
                    page_size_tokens: 16,
                    snapshot_cadence: 0,
                    snapshot_interval_tokens: 0,
                    flags: 0,
                },
                SpaceConfig {
                    space_id: 2,
                    kind: space_kind::RECURRENT_BLOB,
                    name: "gdn.blob",
                    version_tag: 1,
                    bytes_per_token: 0,
                    blob_bytes: 4096,
                    page_size_tokens: 0,
                    snapshot_cadence: 2,
                    snapshot_interval_tokens: 32,
                    flags: 0,
                },
            ],
            seed_ttl_ticks: 8,
            image_token_id: 0,
            media_tokens_per_image: 4,
            vocab: 32000,
            kernel_caps: 0b111,
            scripted: Vec::new(),
            tick_delay: std::time::Duration::ZERO,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageOutput {
    Hidden(Vec<u8>),
    Token(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptSegment {
    pub start_ms: i32,
    pub end_ms: i32,
    pub text: String,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub compression_ratio: f32,
    pub temperature: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TranscribeParams {
    pub language: Option<String>,
    pub translate: bool,
    pub timestamps: bool,
    pub prompt: Option<String>,
}

pub type SegmentSink<'a> = &'a mut (dyn FnMut(i32, i32, &str) -> bool + 'a);

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Transcription {
    pub text: String,
    pub language: String,
    pub duration_ms: i32,
    pub segments: Vec<TranscriptSegment>,
}

pub trait Engine {
    fn tick<'a>(
        &'a mut self,
        plan: &TickPlan,
        arena: &superfluid_abi::RecordArena,
        rings: &mut dyn Rings,
    ) -> Result<&'a TickEvents, Status>;

    fn pump(&mut self);

    fn attach_ring(&mut self, _ring: &crate::rings::RingAttachment) -> Result<(), Status> {
        Ok(())
    }

    fn state_spaces(&self) -> &[StateSpaceDesc];

    fn kv_bits(&self) -> u32 {
        0
    }

    fn strategy_register(
        &mut self,
        reg: &StrategyRegistration,
    ) -> Result<&StrategyGrant, Status>;

    fn space_match(
        &mut self,
        space_id: u32,
        span_tokens: &[u32],
        media_deps: &[[u8; 32]],
    ) -> Result<&MatchResult, Status>;

    fn space_match_ref(
        &mut self,
        space_id: u32,
        _span: &superfluid_abi::TokenRef,
        span_tokens: &[u32],
        media_deps: &[[u8; 32]],
    ) -> Result<&MatchResult, Status> {
        self.space_match(space_id, span_tokens, media_deps)
    }

    fn seed_acquire(
        &mut self,
        span_tokens: &[u32],
        prefix_len: u64,
        determinism_class: u8,
    ) -> Result<u64, Status>;

    fn seed_acquire_ref(
        &mut self,
        _span: &superfluid_abi::TokenRef,
        span_tokens: &[u32],
        prefix_len: u64,
        determinism_class: u8,
    ) -> Result<u64, Status> {
        self.seed_acquire(span_tokens, prefix_len, determinism_class)
    }

    fn seed_release(&mut self, seed_handle: u64) -> Result<(), Status>;

    fn seed_lease_ticks(&self) -> Option<u64> {
        None
    }

    fn seed_adopt(&mut self, _seq: SeqHandle, _span_tokens: &[u32]) -> Result<u64, Status> {
        Err(Status::Unsupported)
    }

    fn media_probe(&mut self, _image_path: &str) -> Result<superfluid_abi::MediaInfo, Status> {
        Err(Status::Unsupported)
    }

    fn media_encode(
        &mut self,
        _image_path: &str,
    ) -> Result<(u64, superfluid_abi::MediaInfo), Status> {
        Err(Status::Unsupported)
    }

    fn media_release(&mut self, _media_handle: u64) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    fn media_bind(&mut self, _lane_tag: u64, _media_handle: u64, _token_offset: u32) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    fn seed_adopt_ref(
        &mut self,
        seq: SeqHandle,
        _span: &superfluid_abi::TokenRef,
        span_tokens: &[u32],
    ) -> Result<u64, Status> {
        self.seed_adopt(seq, span_tokens)
    }

    fn seq_fork(&mut self, parent: SeqHandle, flags: u32) -> Result<SeqHandle, Status>;

    fn lane_sequence(&self, _lane_tag: u64) -> Option<SeqHandle> {
        None
    }

    fn grammar_create(&mut self, _json_schema: &str) -> Result<u32, Status> {
        Err(Status::Unsupported)
    }

    fn grammar_create_structural(&mut self, _tag_json: &str) -> Result<u32, Status> {
        Err(Status::Unsupported)
    }

    fn grammar_free(&mut self, _handle: u32) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    fn logit_bias_create(&mut self, _tokens: &[i32], _values: &[f32]) -> Result<u32, Status> {
        Err(Status::Unsupported)
    }

    fn logit_bias_free(&mut self, _handle: u32) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    fn embed(&mut self, _tokens: &[u32]) -> Result<Vec<f32>, Status> {
        Err(Status::Unsupported)
    }

    fn transcribe(
        &mut self,
        _audio_path: &str,
        _params: &TranscribeParams,
        _on_segment: Option<SegmentSink<'_>>,
    ) -> Result<Transcription, Status> {
        Err(Status::Unsupported)
    }

    fn lora_load(&mut self, _adapter_path: &str) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    fn lora_unload(&mut self) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    fn lora_id(&self) -> Option<String> {
        None
    }

    fn capability_descriptor(&self) -> Option<String> {
        None
    }

    fn create_sequence(&mut self) -> Result<SeqHandle, Status> {
        Err(Status::Unsupported)
    }

    fn free_sequence(&mut self, _seq: SeqHandle) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    fn publish_sequence(&mut self, _seq: SeqHandle, _tokens: &[u32]) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    /// Drops the prefix-cache entry published under exactly `tokens`, unless
    /// a seed holds it; none there is not an error.
    fn unpublish(&mut self, _tokens: &[u32]) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    fn space_export_size(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        range: TokenRange,
        encoding: u8,
    ) -> Result<(u64, u64), Status>;

    fn space_snapshot(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        boundary_pos: u64,
        dst_len: u64,
        sizing_gen: u64,
    ) -> Result<OpHandle, Status>;

    fn space_snapshot_boundary(
        &mut self,
        _seq: SeqHandle,
        _space_id: u32,
        _cap: u64,
    ) -> Result<u64, Status> {
        Err(Status::Unsupported)
    }

    fn cache_evict_entries(&mut self, _keys: &[([u8; 32], [u8; 32])]) -> Result<(), Status> {
        Err(Status::Unsupported)
    }

    fn space_restore(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        src: &[u8],
    ) -> Result<OpHandle, Status>;

    fn space_trim(&mut self, seq: SeqHandle, space_id: u32, new_len: u64) -> Result<(), Status>;

    fn space_demote(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        range: TokenRange,
        encoding: u8,
        dst_len: u64,
        sizing_gen: u64,
    ) -> Result<OpHandle, Status>;

    fn space_promote(
        &mut self,
        seq: SeqHandle,
        space_id: u32,
        range: TokenRange,
        src: &[u8],
    ) -> Result<OpHandle, Status>;

    fn op_poll(&mut self, op: OpHandle) -> Result<OpStatus, Status>;

    fn take_op_output(&mut self, _op: OpHandle) -> Option<Vec<u8>> {
        None
    }

    fn n_layers(&self) -> u32 {
        0
    }

    fn forward_stage(
        &mut self,
        _tokens: &[u32],
        _start_layer: u32,
        _end_layer: u32,
        _hidden_in: Option<Vec<u8>>,
    ) -> Result<StageOutput, Status> {
        Err(Status::Unsupported)
    }

    fn op_cancel(&mut self, op: OpHandle) -> Result<(), Status>;

    fn cache_evict(
        &mut self,
        evictable_cache_classes: u64,
        protected_quota_bytes: u64,
        max_evict_bytes: u64,
        bytes_target: u64,
    ) -> Result<u64, Status>;
}
