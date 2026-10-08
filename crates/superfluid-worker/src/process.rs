//! The worker as its own process.

use std::os::unix::io::FromRawFd;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

use superfluid_proto::linkw;

#[derive(Debug, Clone, Default)]
pub struct WorkerArgs {
    pub frames_fd: Option<i32>,
    pub fd_channel_fd: Option<i32>,
    pub model: Option<PathBuf>,
    pub max_context: i32,
    pub max_batch: u32,
    pub kv_bits: i32,
    pub venv: Option<PathBuf>,
    pub extra: Vec<(String, String)>,
}

impl WorkerArgs {
    pub fn parse(args: &[String]) -> Result<WorkerArgs, String> {
        let mut a = WorkerArgs { max_context: 4096, max_batch: 8, ..Default::default() };
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            if !flag.starts_with("--") {
                return Err(format!("unexpected argument {flag:?}"));
            }
            let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
            let num = |what: &str| format!("{flag}: {what} expected, got {value:?}");
            match flag.as_str() {
                "--frames-fd" => a.frames_fd = Some(value.parse().map_err(|_| num("a descriptor"))?),
                "--fd-channel-fd" => a.fd_channel_fd = Some(value.parse().map_err(|_| num("a descriptor"))?),
                "--model" => a.model = Some(PathBuf::from(value)),
                "--max-context" => a.max_context = value.parse().map_err(|_| num("a token count"))?,
                "--max-batch" => a.max_batch = value.parse().map_err(|_| num("a lane count"))?,
                "--kv-bits" => a.kv_bits = value.parse().map_err(|_| num("a bit width"))?,
                "--venv" => a.venv = Some(PathBuf::from(value)),
                other => a.extra.push((other.to_string(), value.clone())),
            }
        }
        Ok(a)
    }

    pub fn extra(&self, flag: &str) -> Option<&str> {
        self.extra.iter().rev().find(|(k, _)| k == flag).map(|(_, v)| v.as_str())
    }

    pub fn refuse_unread(&self, reads: &[&str]) -> Result<(), String> {
        let unread = |flag: &str, set: bool| set && !reads.contains(&flag);
        if unread("--kv-bits", self.kv_bits != 0) {
            return Err("--kv-bits: this runtime has no such knob".into());
        }
        if unread("--venv", self.venv.is_some()) {
            return Err("--venv: this runtime has no Python environment".into());
        }
        match self.extra.iter().find(|(k, _)| !reads.contains(&k.as_str())) {
            Some((k, _)) => Err(format!("unknown flag {k}")),
            None => Ok(()),
        }
    }

    pub fn streams(&mut self) -> Result<(UnixStream, UnixStream), String> {
        let (Some(frames), Some(fds)) = (self.frames_fd.take(), self.fd_channel_fd.take()) else {
            return Err("--frames-fd and --fd-channel-fd are required to serve".into());
        };
        // SAFETY: the parent dup2'd the Link W socketpair ends onto exactly
        // these descriptors before exec; this process owns them from here,
        // and `take` leaves no second owner.
        Ok(unsafe { (UnixStream::from_raw_fd(frames), UnixStream::from_raw_fd(fds)) })
    }
}

pub fn ring_specs_for(ctx: u32, vocab: u32) -> Vec<linkw::RingSpec> {
    vec![
        linkw::RingSpec {
            ring_id: superfluid_shm::TOKEN_RING_IN,
            kind: linkw::RingKind::Tokens,
            slot_bytes: superfluid_shm::SharedRing::slot_bytes_for(ctx.max(512) * 4),
            slots: 64,
        },
        linkw::RingSpec {
            ring_id: superfluid_shm::TOKEN_RING_OUT,
            kind: linkw::RingKind::Tokens,
            slot_bytes: 16 + 1024 * 4,
            slots: crate::server::TOKEN_OUT_SLOTS,
        },
        linkw::RingSpec {
            ring_id: superfluid_shm::LOGITS_RING,
            kind: linkw::RingKind::Logits,
            slot_bytes: superfluid_shm::SharedRing::slot_bytes_for(vocab * 4),
            slots: 8,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_odd_context_or_vocab_still_makes_rings_that_attach() {
        for spec in ring_specs_for(4097, 50257) {
            let ring = superfluid_shm::SharedRing::create(spec.ring_id, spec.slots, spec.slot_bytes, superfluid_shm::RingRole::Writer);
            assert!(ring.is_ok(), "{spec:?}: {:?}", ring.err());
        }
        assert_eq!(ring_specs_for(4097, 50257)[0].slot_bytes, 16 + 4098 * 4);
    }

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn the_serve_line_parses_and_keeps_what_it_does_not_own() {
        let a = WorkerArgs::parse(&args(&[
            "--frames-fd", "3", "--fd-channel-fd", "4", "--model", "/m.gguf", "--max-context", "8192",
            "--max-batch", "4", "--engine", "native", "--venv", "/env",
        ]))
        .unwrap();
        assert_eq!((a.frames_fd, a.fd_channel_fd), (Some(3), Some(4)));
        assert_eq!(a.model.as_deref(), Some(std::path::Path::new("/m.gguf")));
        assert_eq!((a.max_context, a.max_batch, a.kv_bits), (8192, 4, 0));
        assert_eq!(a.extra("--engine"), Some("native"));
        assert_eq!(a.venv.as_deref(), Some(std::path::Path::new("/env")));
    }

    #[test]
    fn a_bad_line_says_what_is_wrong() {
        assert_eq!(WorkerArgs::parse(&args(&["--max-batch"])).unwrap_err(), "--max-batch needs a value");
        assert!(WorkerArgs::parse(&args(&["--max-batch", "many"])).unwrap_err().starts_with("--max-batch: a lane count"));
        assert!(WorkerArgs::parse(&args(&["serve"])).unwrap_err().contains("unexpected argument"));
        assert!(WorkerArgs::parse(&[]).unwrap().streams().is_err());
    }

    #[test]
    fn a_flag_the_runtime_does_not_read_is_refused() {
        let a = WorkerArgs::parse(&args(&["--kv-bits", "8", "--venv", "/env", "--engine", "mlx"])).unwrap();
        assert_eq!(a.refuse_unread(&["--kv-bits", "--venv", "--engine"]), Ok(()));
        assert_eq!(a.refuse_unread(&["--venv", "--engine"]).unwrap_err(), "--kv-bits: this runtime has no such knob");
        assert_eq!(a.refuse_unread(&["--kv-bits", "--engine"]).unwrap_err(), "--venv: this runtime has no Python environment");
        assert_eq!(a.refuse_unread(&["--kv-bits", "--venv"]).unwrap_err(), "unknown flag --engine");
        assert_eq!(WorkerArgs::parse(&args(&["--model", "/m"])).unwrap().refuse_unread(&[]), Ok(()));
    }
}
