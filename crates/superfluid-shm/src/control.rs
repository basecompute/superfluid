//! The daemon's control words for one worker, in a small segment beside the rings.

use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{segment::ShmSegment, ShmError};

/// "SFCTRL01", little-endian.
pub const CONTROL_MAGIC: u64 = 0x3130_4C52_5443_4653;
pub const CONTROL_BYTES: usize = 64;
/// The fd tag a control segment travels under (rings use their ids).
pub const CONTROL_TAG: u64 = 0xC0;
/// Raised while a tick runs: something is waiting (a chat to admit, a
/// cancel), and the engine ends the tick at its next step boundary.
const YIELD_WORD: usize = 8;
/// Raised while latency-sensitive requests (a chat, an inline completion)
/// have arrived lately: the engine runs its prefill in short steps, so the
/// next one waits less for a tick to end.
const LATENCY_WORD: usize = 16;

#[derive(Debug)]
pub struct ControlSegment {
    seg: ShmSegment,
}

// SAFETY: the mapping is only touched through atomic words.
unsafe impl Sync for ControlSegment {}

impl ControlSegment {
    pub fn create() -> Result<ControlSegment, ShmError> {
        let seg = ShmSegment::create(CONTROL_BYTES)?;
        seg.write_at(0, &CONTROL_MAGIC.to_le_bytes())?;
        Ok(ControlSegment { seg })
    }

    pub fn attach(seg: ShmSegment) -> Result<ControlSegment, ShmError> {
        if seg.len() < CONTROL_BYTES {
            return Err(ShmError::TooSmall);
        }
        let mut magic = [0u8; 8];
        seg.read_at(0, &mut magic)?;
        if u64::from_le_bytes(magic) != CONTROL_MAGIC {
            return Err(ShmError::BadMagic);
        }
        seg.atomic_u64_at(YIELD_WORD)?;
        seg.atomic_u64_at(LATENCY_WORD)?;
        Ok(ControlSegment { seg })
    }

    pub fn raw_fd(&self) -> RawFd {
        self.seg.raw_fd()
    }

    /// A second mapping of the segment, for raising its words from another
    /// thread while its owner is busy.
    pub fn signal(&self) -> Result<YieldSignal, ShmError> {
        // SAFETY: dup of an fd this segment owns; the duplicate is owned below.
        let fd = unsafe { libc::dup(self.seg.raw_fd()) };
        if fd < 0 {
            return Err(ShmError::Io(std::io::Error::last_os_error()));
        }
        // SAFETY: fd is a fresh duplicate nothing else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        Ok(YieldSignal { ctl: ControlSegment::attach(ShmSegment::from_fd(fd, CONTROL_BYTES)?)? })
    }

    fn word(&self, at: usize) -> &AtomicU64 {
        self.seg.atomic_u64_at(at).expect("checked at attach")
    }

    pub fn yield_requested(&self) -> bool {
        self.word(YIELD_WORD).load(Ordering::Acquire) != 0
    }

    pub fn latency_wanted(&self) -> bool {
        self.word(LATENCY_WORD).load(Ordering::Acquire) != 0
    }
}

/// Raises and clears a worker's control words from any thread.
#[derive(Debug)]
pub struct YieldSignal {
    ctl: ControlSegment,
}

impl YieldSignal {
    pub fn raise(&self) {
        self.ctl.word(YIELD_WORD).store(1, Ordering::Release);
    }

    pub fn clear(&self) {
        self.ctl.word(YIELD_WORD).store(0, Ordering::Release);
    }

    pub fn raised(&self) -> bool {
        self.ctl.yield_requested()
    }

    /// Whether latency-sensitive requests are around.
    pub fn set_latency(&self, on: bool) {
        self.ctl.word(LATENCY_WORD).store(u64::from(on), Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdpass::{recv_fd, send_fd};
    use std::os::unix::net::UnixStream;

    #[test]
    fn words_raised_through_a_signal_reach_the_segment_passed_over_the_channel() {
        let ctl = ControlSegment::create().unwrap();
        let signal = ctl.signal().unwrap();
        let (a, b) = UnixStream::pair().unwrap();
        send_fd(&a, ctl.raw_fd(), CONTROL_TAG).unwrap();
        let (fd, tag) = recv_fd(&b).unwrap();
        assert_eq!(tag, CONTROL_TAG);
        let worker = ControlSegment::attach(ShmSegment::from_fd(fd, CONTROL_BYTES).unwrap()).unwrap();
        assert!(!worker.yield_requested() && !worker.latency_wanted());
        std::thread::scope(|s| {
            s.spawn(|| signal.raise());
        });
        assert!(worker.yield_requested() && signal.raised() && !worker.latency_wanted());
        signal.set_latency(true);
        signal.clear();
        assert!(!worker.yield_requested() && worker.latency_wanted(), "the two words are apart");
        let short = ShmSegment::create(16).unwrap();
        assert!(ControlSegment::attach(short).is_err(), "too small");
        let blank = ShmSegment::create(CONTROL_BYTES).unwrap();
        assert!(matches!(ControlSegment::attach(blank), Err(ShmError::BadMagic)));
    }
}
