//! The options `superfluid serve` takes for this runtime, as `--llamacpp-<name> <value>`: llama.cpp's model and context
//! parameters that keep their meaning under superfluid, which owns the context window, the lanes, the KV layout and sampling.

use superfluid_adapter_kit::{option, OptionKind, ServeOption, WorkerArgs};

use crate::runtime::LlamaConfig;
use crate::sys;

const TYPES: &[&str] = &["f16", "bf16", "f32", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"];

/// ggml's own ceiling on threads (`GGML_MAX_N_THREADS`, the width of its CPU masks): a number past it is a slip, not a machine.
const THREADS_MOST: i64 = 512;

/// The most tokens a decode call or a micro-batch takes: llama.cpp holds a batch to the context window, so a number
/// past any window here is a slip of the keyboard, and one past `u32` would wrap.
const BATCH_MOST: i64 = 1 << 20;

pub const OPTIONS: &[ServeOption] = &[
    ServeOption {
        name: "lib",
        kind: OptionKind::Text,
        default: "the install's",
        about: "A llama.cpp build of your own: the libllama file or its directory (as SUPERFLUID_LLAMA_LIB).",
    },
    ServeOption { name: "n-gpu-layers", kind: OptionKind::IntFrom(-1), default: "-1", about: "Layers kept on the device; -1 is all of them, 0 runs the model on the CPU." },
    ServeOption { name: "main-gpu", kind: OptionKind::IntFrom(0), default: "0", about: "The device that takes the whole model under split-mode none, and small tensors otherwise." },
    ServeOption {
        name: "split-mode",
        kind: OptionKind::Choice(&["none", "layer", "row", "tensor"]),
        default: "layer",
        about: "How a model is spread over several devices.",
    },
    ServeOption {
        name: "tensor-split",
        kind: OptionKind::Text,
        default: "even",
        about: "The share of the model each device takes, comma-separated (3,1 gives the first three quarters).",
    },
    ServeOption {
        name: "load-mode",
        kind: OptionKind::Choice(&["auto", "none", "mmap", "mlock", "mmap-mlock", "direct-io"]),
        default: "auto",
        about: "How weights are read: mmap maps the file, mlock pins them in RAM, direct-io bypasses the page cache.",
    },
    ServeOption {
        name: "lazy-mode",
        kind: OptionKind::Choice(&["off", "auto", "on"]),
        default: "auto",
        about: "Read the rows of tensors the architecture marks (experts) on demand; needs mmap.",
    },
    ServeOption { name: "repack", kind: OptionKind::OnOff, default: "on", about: "Repack weights into the device's extra buffer types (llama.cpp's --no-repack turns it off)." },
    ServeOption { name: "threads", kind: OptionKind::IntIn(0, THREADS_MOST), default: "0", about: "Threads for decoding; 0 is every core." },
    ServeOption {
        name: "threads-batch",
        kind: OptionKind::IntIn(0, THREADS_MOST),
        default: "0",
        about: "Threads for prompt batches; 0 is as many as threads.",
    },
    ServeOption { name: "batch-size", kind: OptionKind::IntIn(1, BATCH_MOST), default: "2048", about: "Tokens per decode call (n_batch)." },
    ServeOption {
        name: "ubatch-size",
        kind: OptionKind::IntIn(1, BATCH_MOST),
        default: "512",
        about: "Tokens per micro-batch (n_ubatch), at most batch-size.",
    },
    ServeOption {
        name: "flash-attn",
        kind: OptionKind::Choice(&["auto", "on", "off"]),
        default: "auto",
        about: "Flash attention; auto lets llama.cpp decide for the device.",
    },
    ServeOption { name: "cache-type-k", kind: OptionKind::Choice(TYPES), default: "f16", about: "The KV cache's K type." },
    ServeOption { name: "cache-type-v", kind: OptionKind::Choice(TYPES), default: "f16", about: "The KV cache's V type; a quantized V needs flash-attn on." },
    ServeOption { name: "kv-offload", kind: OptionKind::OnOff, default: "on", about: "Keep the KV cache on the device (llama.cpp's --no-kv-offload turns it off)." },
    ServeOption { name: "op-offload", kind: OptionKind::OnOff, default: "on", about: "Offload operations on host tensors to the device." },
    ServeOption { name: "swa-full", kind: OptionKind::OnOff, default: "on", about: "A full-size cache for sliding-window layers; off keeps only the window and saves memory." },
    ServeOption {
        name: "rope-scaling",
        kind: OptionKind::Choice(&["none", "linear", "yarn", "longrope"]),
        default: "model's",
        about: "RoPE scaling for a context past the trained one.",
    },
    // llama.cpp reads 0 (a negative value, for the YaRN factors) as "the model's", so these say so rather than refuse it.
    ServeOption { name: "rope-freq-base", kind: OptionKind::FloatFrom(0), default: "model's", about: "RoPE base frequency; 0 takes the model's." },
    ServeOption { name: "rope-freq-scale", kind: OptionKind::FloatFrom(0), default: "model's", about: "RoPE frequency scale; 0 takes the model's." },
    ServeOption { name: "yarn-ext-factor", kind: OptionKind::Float, default: "model's", about: "YaRN extrapolation mix; a negative value takes the model's." },
    ServeOption { name: "yarn-attn-factor", kind: OptionKind::Float, default: "model's", about: "YaRN magnitude scale; a negative value takes the model's." },
    ServeOption { name: "yarn-beta-fast", kind: OptionKind::Float, default: "model's", about: "YaRN low correction dimension; a negative value takes the model's." },
    ServeOption { name: "yarn-beta-slow", kind: OptionKind::Float, default: "model's", about: "YaRN high correction dimension; a negative value takes the model's." },
    ServeOption {
        name: "yarn-orig-ctx",
        kind: OptionKind::IntIn(0, u32::MAX as i64),
        default: "model's",
        about: "YaRN original context size; 0 takes the model's.",
    },
];

/// The value given for `name`, judged by its declaration: a worker process refuses a bad one at start, and an
/// in-process worker, which skips that, is refused here; the bounds live in one place, `OPTIONS`.
fn declared<'a>(args: &'a WorkerArgs, name: &str) -> Result<Option<&'a str>, String> {
    let Some(v) = option(args, name) else { return Ok(None) };
    let o = OPTIONS.iter().find(|o| o.name == name).ok_or_else(|| format!("--{name}: not an option of this runtime"))?;
    o.kind.check(v).map_err(|e| format!("--{name}: {e}"))?;
    Ok(Some(v))
}

fn int(args: &WorkerArgs, name: &str) -> Result<Option<i64>, String> {
    declared(args, name)?.map(|v| v.parse().map_err(|_| format!("--{name}: an integer expected, got {v:?}"))).transpose()
}

fn float(args: &WorkerArgs, name: &str) -> Result<Option<f32>, String> {
    // A number f64 takes may still be past f32's.
    let Some(v) = declared(args, name)? else { return Ok(None) };
    v.parse::<f32>().ok().filter(|f| f.is_finite()).map(Some).ok_or_else(|| format!("--{name}: a number expected, got {v:?}"))
}

fn on_off(args: &WorkerArgs, name: &str) -> Result<Option<bool>, String> {
    match option(args, name) {
        None => Ok(None),
        Some("on") => Ok(Some(true)),
        Some("off") => Ok(Some(false)),
        Some(v) => Err(format!("--{name}: on or off expected, got {v:?}")),
    }
}

fn choice<T: Copy>(args: &WorkerArgs, name: &str, table: &[(&str, T)]) -> Result<Option<T>, String> {
    let Some(v) = option(args, name) else { return Ok(None) };
    table
        .iter()
        .find(|(k, _)| *k == v)
        .map(|(_, t)| Some(*t))
        .ok_or_else(|| format!("--{name}: one of {} expected, got {v:?}", table.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(", ")))
}

const CACHE_TYPES: &[(&str, sys::ggml_type)] = &[
    ("f16", sys::GGML_TYPE_F16),
    ("bf16", sys::GGML_TYPE_BF16),
    ("f32", sys::GGML_TYPE_F32),
    ("q8_0", sys::GGML_TYPE_Q8_0),
    ("q4_0", sys::GGML_TYPE_Q4_0),
    ("q4_1", sys::GGML_TYPE_Q4_1),
    ("q5_0", sys::GGML_TYPE_Q5_0),
    ("q5_1", sys::GGML_TYPE_Q5_1),
    ("iq4_nl", sys::GGML_TYPE_IQ4_NL),
];

/// The cache types given, for pricing a cell before a model is loaded.
pub fn cache_types(args: &WorkerArgs) -> Result<(Option<sys::ggml_type>, Option<sys::ggml_type>), String> {
    Ok((choice(args, "cache-type-k", CACHE_TYPES)?, choice(args, "cache-type-v", CACHE_TYPES)?))
}

/// Sets every option given in `args` on `cfg`; the rest keep their defaults.
pub fn apply(args: &WorkerArgs, cfg: &mut LlamaConfig) -> Result<(), String> {
    if let Some(n) = int(args, "n-gpu-layers")? {
        cfg.n_gpu_layers = n.min(i32::MAX as i64) as i32;
    }
    if let Some(n) = int(args, "main-gpu")? {
        cfg.main_gpu = n.min(i32::MAX as i64) as i32;
    }
    cfg.split_mode = choice(
        args,
        "split-mode",
        &[
            ("none", sys::LLAMA_SPLIT_MODE_NONE),
            ("layer", sys::LLAMA_SPLIT_MODE_LAYER),
            ("row", sys::LLAMA_SPLIT_MODE_ROW),
            ("tensor", sys::LLAMA_SPLIT_MODE_TENSOR),
        ],
    )?;
    if let Some(v) = option(args, "tensor-split") {
        let shares: Vec<f32> = v
            .split(',')
            .map(|s| s.trim().parse::<f32>().ok().filter(|f| f.is_finite() && *f >= 0.0))
            .collect::<Option<Vec<_>>>()
            .filter(|s| !s.is_empty() && s.iter().any(|f| *f > 0.0))
            .ok_or_else(|| format!("--tensor-split: shares at or above 0, comma-separated, not all 0, expected, got {v:?}"))?;
        cfg.tensor_split = shares;
    }
    cfg.load_mode = choice(
        args,
        "load-mode",
        &[
            ("auto", sys::LLAMA_LOAD_MODE_AUTO),
            ("none", sys::LLAMA_LOAD_MODE_NONE),
            ("mmap", sys::LLAMA_LOAD_MODE_MMAP),
            ("mlock", sys::LLAMA_LOAD_MODE_MLOCK),
            ("mmap-mlock", sys::LLAMA_LOAD_MODE_MMAP_MLOCK),
            ("direct-io", sys::LLAMA_LOAD_MODE_DIRECT_IO),
        ],
    )?;
    cfg.lazy_mode = choice(args, "lazy-mode", &[("off", sys::LLAMA_LAZY_MODE_OFF), ("auto", sys::LLAMA_LAZY_MODE_AUTO), ("on", sys::LLAMA_LAZY_MODE_ON)])?;
    cfg.repack = on_off(args, "repack")?;
    if let Some(n) = int(args, "threads")? {
        cfg.n_threads = n.min(i32::MAX as i64) as i32;
    }
    if let Some(n) = int(args, "threads-batch")? {
        cfg.n_threads_batch = n.min(i32::MAX as i64) as i32;
    }
    if let Some(n) = int(args, "batch-size")? {
        cfg.n_batch = n.min(u32::MAX as i64) as u32;
    }
    if let Some(n) = int(args, "ubatch-size")? {
        cfg.n_ubatch = n.min(u32::MAX as i64) as u32;
    }
    cfg.flash_attn = choice(
        args,
        "flash-attn",
        &[("auto", sys::LLAMA_FLASH_ATTN_TYPE_AUTO), ("on", sys::LLAMA_FLASH_ATTN_TYPE_ENABLED), ("off", sys::LLAMA_FLASH_ATTN_TYPE_DISABLED)],
    )?;
    cfg.type_k = choice(args, "cache-type-k", CACHE_TYPES)?;
    cfg.type_v = choice(args, "cache-type-v", CACHE_TYPES)?;
    if cfg.type_v.is_some_and(|t| t != sys::GGML_TYPE_F16 && t != sys::GGML_TYPE_BF16 && t != sys::GGML_TYPE_F32)
        && cfg.flash_attn == Some(sys::LLAMA_FLASH_ATTN_TYPE_DISABLED)
    {
        return Err("--cache-type-v: a quantized V cache needs flash attention, which --flash-attn off refuses".into());
    }
    cfg.offload_kqv = on_off(args, "kv-offload")?;
    cfg.op_offload = on_off(args, "op-offload")?;
    cfg.swa_full = on_off(args, "swa-full")?;
    cfg.rope_scaling = choice(
        args,
        "rope-scaling",
        &[
            ("none", sys::LLAMA_ROPE_SCALING_TYPE_NONE),
            ("linear", sys::LLAMA_ROPE_SCALING_TYPE_LINEAR),
            ("yarn", sys::LLAMA_ROPE_SCALING_TYPE_YARN),
            ("longrope", sys::LLAMA_ROPE_SCALING_TYPE_LONGROPE),
        ],
    )?;
    cfg.rope_freq_base = float(args, "rope-freq-base")?;
    cfg.rope_freq_scale = float(args, "rope-freq-scale")?;
    cfg.yarn_ext_factor = float(args, "yarn-ext-factor")?;
    cfg.yarn_attn_factor = float(args, "yarn-attn-factor")?;
    cfg.yarn_beta_fast = float(args, "yarn-beta-fast")?;
    cfg.yarn_beta_slow = float(args, "yarn-beta-slow")?;
    if let Some(n) = int(args, "yarn-orig-ctx")? {
        cfg.yarn_orig_ctx = Some(n.min(u32::MAX as i64) as u32);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(pairs: &[(&str, &str)]) -> WorkerArgs {
        let argv: Vec<String> = pairs.iter().flat_map(|(k, v)| [format!("--{k}"), v.to_string()]).collect();
        WorkerArgs::parse(&argv).unwrap()
    }

    fn applied(pairs: &[(&str, &str)]) -> Result<LlamaConfig, String> {
        let mut cfg = LlamaConfig::default();
        apply(&args(pairs), &mut cfg).map(|()| cfg)
    }

    #[test]
    fn the_declaration_is_one_the_daemon_honours() {
        superfluid_adapter_kit::check_declaration(OPTIONS).unwrap();
        // The bounds the daemon checks are the ones `apply` holds.
        for (name, least, most) in [
            ("n-gpu-layers", -1, None),
            ("main-gpu", 0, None),
            ("threads", 0, Some(512)),
            ("threads-batch", 0, Some(512)),
            ("batch-size", 1, Some(1 << 20)),
            ("ubatch-size", 1, Some(1 << 20)),
            ("yarn-orig-ctx", 0, Some(u32::MAX as i64)),
        ] {
            let o = OPTIONS.iter().find(|o| o.name == name).unwrap();
            assert_eq!((o.kind.least(), o.kind.most()), (Some(least), most), "{name}");
            assert!(applied(&[(name, &least.to_string())]).is_ok());
            assert!(applied(&[(name, &(least - 1).to_string())]).is_err());
            if let Some(most) = most {
                assert!(applied(&[(name, &most.to_string())]).is_ok(), "{name}");
                assert!(applied(&[(name, &(most + 1).to_string())]).is_err(), "{name}");
            }
        }
        for name in ["rope-freq-base", "rope-freq-scale"] {
            assert_eq!(OPTIONS.iter().find(|o| o.name == name).unwrap().kind.least(), Some(0), "{name}: 0 is the model's, below it nothing");
        }
        // Every option is read: a declared name `apply` ignores would be accepted and do nothing.
        let src = include_str!("options.rs");
        for o in OPTIONS.iter().filter(|o| o.name != "lib") {
            assert!(src.contains(&format!("\"{}\"", o.name)) && src.matches(&format!("\"{}\"", o.name)).count() >= 2, "{} is declared but not read", o.name);
        }
    }

    #[test]
    fn nothing_given_leaves_every_default() {
        let cfg = applied(&[]).unwrap();
        let d = LlamaConfig::default();
        assert_eq!((cfg.n_gpu_layers, cfg.n_batch, cfg.n_ubatch, cfg.n_threads), (d.n_gpu_layers, d.n_batch, d.n_ubatch, d.n_threads));
        assert!(cfg.flash_attn.is_none() && cfg.type_k.is_none() && cfg.load_mode.is_none() && cfg.tensor_split.is_empty());
    }

    #[test]
    fn every_kind_of_option_lands_on_the_config() {
        let cfg = applied(&[
            ("n-gpu-layers", "20"),
            ("main-gpu", "1"),
            ("split-mode", "row"),
            ("tensor-split", "3, 1"),
            ("load-mode", "mmap-mlock"),
            ("lazy-mode", "on"),
            ("repack", "off"),
            ("threads", "8"),
            ("threads-batch", "16"),
            ("batch-size", "1024"),
            ("ubatch-size", "256"),
            ("flash-attn", "on"),
            ("cache-type-k", "q8_0"),
            ("cache-type-v", "q4_0"),
            ("kv-offload", "off"),
            ("op-offload", "off"),
            ("swa-full", "off"),
            ("rope-scaling", "yarn"),
            ("rope-freq-base", "1000000"),
            ("rope-freq-scale", "0.25"),
            ("yarn-ext-factor", "1"),
            ("yarn-attn-factor", "1.5"),
            ("yarn-beta-fast", "32"),
            ("yarn-beta-slow", "1"),
            ("yarn-orig-ctx", "32768"),
        ])
        .unwrap();
        assert_eq!((cfg.n_gpu_layers, cfg.main_gpu, cfg.split_mode), (20, 1, Some(sys::LLAMA_SPLIT_MODE_ROW)));
        assert_eq!(cfg.tensor_split, vec![3.0, 1.0]);
        assert_eq!((cfg.load_mode, cfg.lazy_mode, cfg.repack), (Some(sys::LLAMA_LOAD_MODE_MMAP_MLOCK), Some(sys::LLAMA_LAZY_MODE_ON), Some(false)));
        assert_eq!((cfg.n_threads, cfg.n_threads_batch, cfg.n_batch, cfg.n_ubatch), (8, 16, 1024, 256));
        assert_eq!((cfg.flash_attn, cfg.type_k, cfg.type_v), (Some(sys::LLAMA_FLASH_ATTN_TYPE_ENABLED), Some(sys::GGML_TYPE_Q8_0), Some(sys::GGML_TYPE_Q4_0)));
        assert_eq!((cfg.offload_kqv, cfg.op_offload, cfg.swa_full), (Some(false), Some(false), Some(false)));
        assert_eq!(cfg.rope_scaling, Some(sys::LLAMA_ROPE_SCALING_TYPE_YARN));
        assert_eq!((cfg.rope_freq_base, cfg.rope_freq_scale, cfg.yarn_ext_factor), (Some(1_000_000.0), Some(0.25), Some(1.0)));
        assert_eq!((cfg.yarn_attn_factor, cfg.yarn_beta_fast, cfg.yarn_beta_slow, cfg.yarn_orig_ctx), (Some(1.5), Some(32.0), Some(1.0), Some(32768)));
    }

    #[test]
    fn a_value_the_option_does_not_take_is_refused_by_name() {
        for (pairs, said) in [
            (vec![("n-gpu-layers", "all")], "--n-gpu-layers: an integer expected"),
            (vec![("n-gpu-layers", "-2")], "--n-gpu-layers: at least -1 expected"),
            (vec![("batch-size", "0")], "--batch-size: 1 to 1048576 expected"),
            (vec![("ubatch-size", "4294967296")], "--ubatch-size: 1 to 1048576 expected"),
            (vec![("threads", "2147483647")], "--threads: 0 to 512 expected, got 2147483647"),
            (vec![("rope-freq-base", "-10000")], "--rope-freq-base: at least 0 expected"),
            (vec![("rope-freq-scale", "-0.5")], "--rope-freq-scale: at least 0 expected"),
            (vec![("rope-freq-scale", "1e300")], "--rope-freq-scale: a number expected"),
            (vec![("flash-attn", "yes")], "--flash-attn: one of auto, on, off expected"),
            (vec![("cache-type-k", "q2_k")], "--cache-type-k: one of f16, bf16, f32, q8_0"),
            (vec![("kv-offload", "true")], "--kv-offload: on or off expected"),
            (vec![("rope-freq-base", "fast")], "--rope-freq-base: a number expected"),
            (vec![("tensor-split", "1,-1")], "--tensor-split: shares at or above 0"),
            (vec![("tensor-split", "0,0")], "--tensor-split: shares at or above 0"),
            (vec![("cache-type-v", "q8_0"), ("flash-attn", "off")], "a quantized V cache needs flash attention"),
        ] {
            let e = applied(&pairs).unwrap_err();
            assert!(e.contains(said), "{pairs:?}: {e}");
        }
    }
}
