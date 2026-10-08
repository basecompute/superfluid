//! Standalone engine-worker.

use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;

use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_worker::{WorkerConfig, WorkerServer};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("usage: superfluid-worker <socket-dir>");
            std::process::exit(2);
        });
    std::fs::create_dir_all(&dir).expect("create socket dir");
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
        .expect("restrict socket dir");

    let frames_path = dir.join("frames.sock");
    let fds_path = dir.join("fds.sock");
    let _ = std::fs::remove_file(&frames_path);
    let _ = std::fs::remove_file(&fds_path);
    let frames_listener = UnixListener::bind(&frames_path).expect("bind frames.sock");
    let fds_listener = UnixListener::bind(&fds_path).expect("bind fds.sock");
    eprintln!("superfluid-worker: listening at {}", dir.display());

    let (frames, _) = frames_listener.accept().expect("accept frames");
    let (fds, _) = fds_listener.accept().expect("accept fds");

    let engine = MockEngine::new(EngineConfig::default());
    let cfg = WorkerConfig {
        engine_bundle_hash: *b"mock-engine-bundle-0001-padding!",
    };
    match WorkerServer::new(engine, cfg, frames, fds).serve() {
        Ok(()) => eprintln!("superfluid-worker: peer disconnected, exiting"),
        Err(e) => {
            eprintln!("superfluid-worker: fatal: {e}");
            std::process::exit(1);
        }
    }
}
