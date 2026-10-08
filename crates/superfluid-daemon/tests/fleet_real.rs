//! One node, the REAL engine (feature `basert`).

#![cfg(feature = "basert")]

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

use superfluid_daemon::codec::ChatMlCodec;
use superfluid_daemon::fleet::FleetHead;
use superfluid_daemon::nodeagent::NodeAgent;
use superfluid_daemon::{Daemon, EngineHost, GenParams, SessionStore};
use superfluid_engine_ffi::{FfiEngineConfig, NativeEngine};
use superfluid_proto::linkf::Lease;
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};

static MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn model_path() -> Option<PathBuf> {
    let p = model_path_found()?;
    if p.extension().is_some_and(|e| e == "base") {
        superfluid_engine_ffi::libbasert::require()?;
    }
    Some(p)
}

fn model_path_found() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BASERT_TEST_MODEL") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    ["models/Qwen3-0.6B-Q4_K_M.base", "models/Qwen3-0.6B-Q4_K_M.gguf"]
        .iter()
        .map(|c| root.join(c))
        .find(|p| p.exists())
}

fn wal_path(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fleet-real-{}-{}-{}",
        std::process::id(),
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("wal.log")
}

fn real_daemon(model: PathBuf, tag: &str) -> Daemon {
    let store = SessionStore::open(&wal_path(tag)).expect("open store");
    let model_for_codec = model.clone();
    let host = EngineHost::spawn(move || {
        let engine = NativeEngine::load(FfiEngineConfig {
            model_path: model,
            max_context: 4096,
            max_batch_size: 8,
            seed_ttl_ticks: 64,
        })
        .expect("model load");
        let vocab = engine.vocab();
        let specs = vec![
            linkw::RingSpec {
                ring_id: TOKEN_RING_IN,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + 4096 * 4,
                slots: 64,
            },
            linkw::RingSpec {
                ring_id: TOKEN_RING_OUT,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + 1024 * 4,
                slots: 64,
            },
            linkw::RingSpec {
                ring_id: LOGITS_RING,
                kind: linkw::RingKind::Logits,
                slot_bytes: 16 + vocab * 4,
                slots: 8,
            },
        ];
        (engine, Some(specs))
    })
    .expect("spawn engine worker");
    let codec = ChatMlCodec::load(&model_for_codec).expect("tokenizer");
    Daemon::new(store, host, Box::new(codec), 8)
}

#[test]
fn fleet_real_generation_matches_local_token_for_token() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model (set BASERT_TEST_MODEL or place models/Qwen3-0.6B-Q4_K_M.*)");
        return;
    };

    let prompt: Vec<u32> = vec![9707, 11, 1879, 0, 358, 1079, 264];
    let params = GenParams::default();
    let max = 24u32;

    let local_tokens = {
        let local = real_daemon(model.clone(), "ref");
        let (toks, expired) = local.generate_tokens(prompt.clone(), params, max, 0).unwrap();
        assert!(!expired);
        assert!(!toks.is_empty(), "real engine produced no tokens");
        toks
    };

    let node_daemon = Arc::new(real_daemon(model, "node"));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let node_thread = std::thread::spawn(move || {
        let mut node = NodeAgent::new(node_daemon, "node-real", "qwen");
        for sock in listener.incoming() {
            let Ok(sock) = sock else { break };
            let ep = superfluid_linkf::Endpoint::from_tcp(sock).unwrap();
            if node.serve(ep).is_err() {
                break;
            }
        }
    });

    let lease = Lease {
        duration_ms: 120_000,
        renew_by_ms: 90_000,
    };
    let mut head = FleetHead::connect(&addr, Vec::new()).unwrap();
    let session = 1u64;
    head.assign(session, params, 0, "chatml", lease).unwrap();
    head.append(session, &prompt).unwrap();
    let gen = head.generate(session, max as u64, 120_000).unwrap();

    assert!(!gen.expired);
    assert_eq!(
        gen.tokens, local_tokens,
        "real fleet path diverged from the real local path"
    );
    head.wal().assert_single_contiguous_streams();

    drop(head);
    let _ = node_thread;
}
