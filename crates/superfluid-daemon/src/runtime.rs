use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::thread::JoinHandle;

use superfluid_agent::WorkerClient;
use superfluid_engine::Engine;
use superfluid_proto::linkw;
use superfluid_worker::{WorkerConfig, WorkerServer};

use crate::DaemonError;

#[derive(Debug, Clone)]
pub struct WorkerSpec {
    pub bin: PathBuf,
    pub args: Vec<String>,
    pub stderr: Option<std::sync::Arc<std::os::fd::OwnedFd>>,
}

pub use superfluid_worker::ring_specs_for;

enum Backend {
    Thread(Option<JoinHandle<()>>),
    Process {
        child: std::process::Child,
        spec: WorkerSpec,
    },
}

pub struct EngineHost {
    client: Option<WorkerClient>,
    backend: Backend,
    generation: u64,
    caps: SharedCapabilities,
    record_change: Option<String>,
    yield_slot: YieldSlot,
}

/// The running worker's yield flag (replaced when the worker respawns):
/// raised, the engine ends its tick at the next step boundary.
pub type YieldSlot = std::sync::Arc<std::sync::RwLock<Option<superfluid_shm::YieldSignal>>>;

fn yield_slot_of(client: &WorkerClient) -> YieldSlot {
    std::sync::Arc::new(std::sync::RwLock::new(client.yield_signal()))
}

/// Tells the worker whether latency-sensitive requests are around.
pub fn set_latency(slot: &YieldSlot, on: bool) {
    if let Some(s) = slot.read().expect("yield slot").as_ref() {
        s.set_latency(on);
    }
}

/// Asks the running tick to end at its next step boundary.
pub fn raise_yield(slot: &YieldSlot) {
    if let Some(s) = slot.read().expect("yield slot").as_ref() {
        s.raise();
    }
}

pub type SharedCapabilities = std::sync::Arc<std::sync::RwLock<crate::capabilities::Capabilities>>;

fn caps_of(client: &WorkerClient) -> SharedCapabilities {
    std::sync::Arc::new(std::sync::RwLock::new(crate::capabilities::Capabilities::parse(&client.hello.capabilities)))
}

impl EngineHost {
    pub fn spawn<E, F>(make_engine: F) -> Result<EngineHost, DaemonError>
    where
        E: Engine + 'static,
        F: FnOnce() -> (E, Option<Vec<linkw::RingSpec>>) + Send + 'static,
    {
        let (frames_a, frames_w) = UnixStream::pair()?;
        let (fds_a, fds_w) = UnixStream::pair()?;
        let handle = std::thread::Builder::new()
            .name("engine-worker".into())
            .spawn(move || {
                let (engine, ring_specs) = make_engine();
                let mut server = WorkerServer::new(
                    engine,
                    WorkerConfig {
                        engine_bundle_hash: [0; 32],
                    },
                    frames_w,
                    fds_w,
                );
                if let Some(specs) = ring_specs {
                    server = server.with_ring_specs(specs);
                }
                if let Err(e) = server.serve() {
                    tracing::error!("engine-worker exited with error: {e}");
                }
            })?;
        let client = WorkerClient::connect(frames_a, fds_a)?;
        Ok(EngineHost {
            caps: caps_of(&client),
            yield_slot: yield_slot_of(&client),
            client: Some(client),
            backend: Backend::Thread(Some(handle)),
            generation: 0,
            record_change: None,
        })
    }

    pub fn try_spawn<E, F>(make_engine: F) -> Result<EngineHost, DaemonError>
    where
        E: Engine + 'static,
        F: FnOnce() -> Result<(E, Option<Vec<linkw::RingSpec>>), String> + Send + 'static,
    {
        EngineHost::try_spawn_loaded(move || make_engine().map(|(engine, rings)| superfluid_adapter_kit::Loaded::engine(engine, rings)))
    }

    pub fn try_spawn_loaded<F>(open: F) -> Result<EngineHost, DaemonError>
    where
        F: FnOnce() -> Result<superfluid_adapter_kit::Loaded, String> + Send + 'static,
    {
        let (frames_a, frames_w) = UnixStream::pair()?;
        let (fds_a, fds_w) = UnixStream::pair()?;
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let handle = std::thread::Builder::new()
            .name("engine-worker".into())
            .spawn(move || {
                let loaded = match open() {
                    Ok(v) => {
                        let _ = tx.send(Ok(()));
                        v
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                };
                if let Err(e) = loaded.serve(frames_w, fds_w) {
                    tracing::error!("engine-worker exited with error: {e}");
                }
            })?;
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(msg)) => {
                let _ = handle.join();
                return Err(DaemonError::EngineLoad(msg));
            }
            Err(_) => {
                let _ = handle.join();
                return Err(DaemonError::EngineLoad(
                    "engine worker exited before reporting a load result".into(),
                ));
            }
        }
        let client = WorkerClient::connect(frames_a, fds_a)?;
        Ok(EngineHost {
            caps: caps_of(&client),
            yield_slot: yield_slot_of(&client),
            client: Some(client),
            backend: Backend::Thread(Some(handle)),
            generation: 0,
            record_change: None,
        })
    }

    pub fn spawn_process(spec: WorkerSpec) -> Result<EngineHost, DaemonError> {
        let (child, client) = launch(&spec)?;
        Ok(EngineHost {
            caps: caps_of(&client),
            yield_slot: yield_slot_of(&client),
            client: Some(client),
            backend: Backend::Process { child, spec },
            generation: 0,
            record_change: None,
        })
    }

    pub fn yield_slot(&self) -> YieldSlot {
        std::sync::Arc::clone(&self.yield_slot)
    }

    /// Lowers the yield flag before a tick is sent.
    pub fn clear_yield(&self) {
        if let Some(s) = self.yield_slot.read().expect("yield slot").as_ref() {
            s.clear();
        }
    }

    pub fn is_process(&self) -> bool {
        matches!(self.backend, Backend::Process { .. })
    }

    pub fn worker_pid(&self) -> Option<u32> {
        match &self.backend {
            Backend::Process { child, .. } => Some(child.id()),
            Backend::Thread(_) => None,
        }
    }

    pub fn respawn(&mut self) -> Result<(), DaemonError> {
        let Backend::Process { child, spec } = &mut self.backend else {
            return Err(DaemonError::Protocol(
                "in-process worker shares the daemon's fate; nothing to respawn",
            ));
        };
        drop(self.client.take());
        let _ = child.kill();
        let _ = child.wait();
        let (new_child, client) = launch(spec)?;
        *child = new_child;
        let fresh = crate::capabilities::Capabilities::parse(&client.hello.capabilities);
        let mut held = self.caps.write().expect("capability record");
        let changed = held.changes(&fresh);
        if !changed.is_empty() {
            tracing::warn!(changed = %changed.join(", "), "the respawned worker describes a different runtime or model");
            self.record_change = Some(format!("the respawned worker's capability record differs ({})", changed.join(", ")));
        }
        *held = fresh;
        drop(held);
        *self.yield_slot.write().expect("yield slot") = client.yield_signal();
        self.client = Some(client);
        self.generation += 1;
        Ok(())
    }

    pub fn take_record_change(&mut self) -> Option<String> {
        self.record_change.take()
    }

    pub fn max_seqs(&self) -> Option<u64> {
        self.caps.read().expect("capability record").limit("max_seqs")
    }

    pub fn exact_seeds_only(&self) -> bool {
        matches!(self.caps.read().expect("capability record").get("state", "rollback"), crate::capabilities::Cap::No(_))
    }

    /// The runtime seeds only from a state covering a whole cached sequence, so a shared
    /// prefix needs a copy kept where it ends.
    pub fn needs_prefix_checkpoints(&self) -> bool {
        matches!(self.caps.read().expect("capability record").get("state", "shared_prefix_seed"), crate::capabilities::Cap::No(_))
    }

    pub fn park_refused(&self) -> Option<String> {
        match self.caps.read().expect("capability record").get("serving", "park_lossless") {
            crate::capabilities::Cap::No(why) => Some(why.to_string()),
            _ => None,
        }
    }

    pub fn round_granular(&self) -> bool {
        matches!(self.caps.read().expect("capability record").get("serving", "round_granular"), crate::capabilities::Cap::Yes)
    }

    pub fn lossy_park_allowed(&self) -> bool {
        !matches!(self.caps.read().expect("capability record").get("serving", "park_lossy"), crate::capabilities::Cap::No(_))
    }

    pub fn capabilities(&self) -> SharedCapabilities {
        std::sync::Arc::clone(&self.caps)
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn is_alive(&self) -> bool {
        self.client.is_some()
    }

    pub fn client(&mut self) -> &mut WorkerClient {
        self.client
            .as_mut()
            .expect("caller must gate on is_alive() after a failed respawn")
    }

    fn client_ref(&self) -> Option<&WorkerClient> {
        self.client.as_ref()
    }

    pub fn kv_bits(&self) -> u32 {
        self.client_ref().map(|c| c.hello.kv_bits).unwrap_or(0)
    }

    pub fn page_size(&self) -> Option<u64> {
        self.client_ref()?
            .hello
            .state_space_descriptors
            .iter()
            .find(|d| d.page_size_tokens > 0)
            .map(|d| d.page_size_tokens as u64)
    }

    pub fn has_unservable_space(&self) -> bool {
        self.client_ref()
            .map(|c| {
                c.hello
                    .state_space_descriptors
                    .iter()
                    .any(|d| d.flags & superfluid_abi::space_flag::OPS_UNAVAILABLE != 0)
            })
            .unwrap_or(false)
    }

    pub fn has_recurrent_space(&self) -> bool {
        self.client_ref()
            .map(|c| {
                c.hello
                    .state_space_descriptors
                    .iter()
                    .any(|d| d.kind == superfluid_abi::space_kind::RECURRENT_BLOB)
            })
            .unwrap_or(false)
    }

    pub fn cache_serves_all_spaces(&self) -> bool {
        self.client_ref()
            .map(|c| {
                c.hello
                    .state_space_descriptors
                    .iter()
                    .all(|d| d.flags & superfluid_abi::space_flag::NO_PREFIX_CACHE == 0)
            })
            .unwrap_or(false)
    }

    pub fn parkable_spaces(&self) -> Vec<(u32, u32, u32, u32)> {
        let mut v: Vec<(u32, u32, u32, u32)> = self
            .client_ref()
            .map(|c| {
                c.hello
                    .state_space_descriptors
                    .iter()
                    .filter(|d| {
                        d.flags & superfluid_abi::space_flag::OPS_UNAVAILABLE == 0
                            && d.kind != superfluid_abi::space_kind::ENCODER_CACHE
                    })
                    .map(|d| {
                        (
                            d.space_id,
                            d.kind,
                            d.snapshot_cadence,
                            d.snapshot_interval_tokens,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        v.sort_by_key(|&(id, kind, _, _)| (kind != superfluid_abi::space_kind::PAGED_TOKEN_KV, id));
        v
    }

    pub fn park_boundary(&self, len: u64) -> u64 {
        let mut b = len;
        for &(_, kind, cadence, interval) in &self.parkable_spaces() {
            if kind == superfluid_abi::space_kind::PAGED_TOKEN_KV {
                continue;
            }
            if cadence == 2 && interval > 0 {
                b = b.min(len / interval as u64 * interval as u64);
            }
        }
        b
    }

    pub fn multi_space(&self) -> bool {
        self.parkable_spaces()
            .iter()
            .any(|&(_, kind, _, _)| kind != superfluid_abi::space_kind::PAGED_TOKEN_KV)
    }

    pub fn kv_block_bytes(&self) -> Option<u64> {
        self.client_ref()?
            .hello
            .state_space_descriptors
            .iter()
            .find(|d| d.page_size_tokens > 0)
            .map(|d| d.bytes_per_token * d.page_size_tokens as u64)
    }

    pub fn paged_spaces(&self) -> Vec<(u32, u64, u64)> {
        let Some(c) = self.client_ref() else {
            return Vec::new();
        };
        c.hello
            .state_space_descriptors
            .iter()
            .filter(|d| d.page_size_tokens > 0)
            .filter(|d| d.kind != superfluid_abi::space_kind::ENCODER_CACHE)
            .map(|d| {
                let page = d.page_size_tokens as u64;
                (d.space_id, page, d.bytes_per_token * page)
            })
            .collect()
    }

    pub fn blob_spaces(&self) -> Vec<(u32, u64)> {
        let Some(c) = self.client_ref() else {
            return Vec::new();
        };
        c.hello
            .state_space_descriptors
            .iter()
            .filter(|d| d.page_size_tokens == 0 && d.blob_bytes > 0)
            .filter(|d| d.kind != superfluid_abi::space_kind::ENCODER_CACHE)
            .map(|d| (d.space_id, d.blob_bytes))
            .collect()
    }

    pub fn space_shape(&self, space_id: u32) -> Option<(u32, u32)> {
        self.client_ref()?
            .hello
            .state_space_descriptors
            .iter()
            .find(|d| d.space_id == space_id)
            .map(|d| (d.kind, d.page_size_tokens))
    }

    pub fn max_stream_tokens(&self) -> u64 {
        stream_ceiling(self.staged_tokens(), &self.caps.read().expect("capability record"))
    }

    pub fn staged_tokens(&self) -> u64 {
        self.client_ref()
            .map(|c| {
                c.hello
                    .limits
                    .ring_specs
                    .iter()
                    .find(|s| s.ring_id == superfluid_shm::TOKEN_RING_IN)
                    .map(|s| ((s.slot_bytes.saturating_sub(16)) / 4) as u64)
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    }
}

pub(crate) fn stream_ceiling(staged: u64, caps: &crate::capabilities::Capabilities) -> u64 {
    match caps.limit("max_seq_len") {
        Some(held) if staged > 0 => staged.min(held),
        _ => staged,
    }
}

fn launch(spec: &WorkerSpec) -> Result<(std::process::Child, WorkerClient), DaemonError> {
    use std::os::unix::io::AsRawFd;
    use std::os::unix::process::CommandExt;

    let (frames_a, frames_w) = UnixStream::pair()?;
    let (fds_a, fds_w) = UnixStream::pair()?;
    let frames_fd = frames_w.as_raw_fd();
    let fdchan_fd = fds_w.as_raw_fd();
    let mut cmd = std::process::Command::new(&spec.bin);
    crate::exec::without_secrets(&mut cmd)
        .args(&spec.args)
        .arg("--frames-fd")
        .arg("3")
        .arg("--fd-channel-fd")
        .arg("4");
    if let Some(fd) = &spec.stderr {
        cmd.stderr(std::process::Stdio::from(fd.try_clone()?));
        cmd.stdout(std::process::Stdio::from(fd.try_clone()?));
    }
    // SAFETY: pre_exec runs in the forked child before exec. The two ends are first
    // moved above 10, since one of them may already be 3 or 4: dup2 onto itself keeps
    // CLOEXEC and the worker would inherit a closed descriptor, and dup2 onto the
    // other's number would clobber it. dup2 onto 3 and 4 then clears CLOEXEC on
    // exactly those two, so they alone survive the exec.
    unsafe {
        cmd.pre_exec(move || {
            let frames = libc::fcntl(frames_fd, libc::F_DUPFD_CLOEXEC, 10);
            let fdchan = libc::fcntl(fdchan_fd, libc::F_DUPFD_CLOEXEC, 10);
            if frames < 0 || fdchan < 0 || libc::dup2(frames, 3) < 0 || libc::dup2(fdchan, 4) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = crate::exec::when_not_busy(|| cmd.spawn())?;
    drop(frames_w);
    drop(fds_w);
    let client = WorkerClient::connect(frames_a, fds_a)?;
    Ok((child, client))
}

impl Drop for EngineHost {
    fn drop(&mut self) {
        drop(self.client.take());
        match &mut self.backend {
            Backend::Thread(handle) => {
                if let Some(h) = handle.take() {
                    let _ = h.join();
                }
            }
            Backend::Process { child, .. } => {
                let _ = child.wait();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use superfluid_engine::{EngineConfig, MockEngine};

    use super::EngineHost;
    use crate::DaemonError;

    struct Unwound(Arc<AtomicBool>);
    impl Drop for Unwound {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn try_spawn_refusal_is_engine_load_and_joins_the_worker() {
        let unwound = Arc::new(AtomicBool::new(false));
        let guard = Unwound(unwound.clone());
        let err = EngineHost::try_spawn(move || {
            let _guard = guard;
            Err::<(MockEngine, _), _>("no paged KV on this backend".to_string())
        })
        .err()
        .expect("refusal must surface as an error");
        match err {
            DaemonError::EngineLoad(msg) => assert_eq!(msg, "no paged KV on this backend"),
            other => panic!("expected EngineLoad, got {other:?}"),
        }
        assert!(unwound.load(Ordering::SeqCst), "worker thread not joined");
    }

    #[test]
    fn try_spawn_panicking_closure_reports_early_exit_without_panicking() {
        let unwound = Arc::new(AtomicBool::new(false));
        let guard = Unwound(unwound.clone());
        let err = EngineHost::try_spawn(move || -> Result<(MockEngine, _), String> {
            let _guard = guard;
            panic!("engine build blew up");
        })
        .err()
        .expect("a dead worker must surface as an error");
        match err {
            DaemonError::EngineLoad(msg) => {
                assert_eq!(msg, "engine worker exited before reporting a load result")
            }
            other => panic!("expected EngineLoad, got {other:?}"),
        }
        assert!(unwound.load(Ordering::SeqCst), "worker thread not joined");
    }

    #[test]
    fn try_spawn_success_builds_a_live_host() {
        let host = EngineHost::try_spawn(|| Ok((MockEngine::new(EngineConfig::default()), None)))
            .expect("mock engine loads");
        assert!(!host.is_process());
        drop(host);
    }
}
