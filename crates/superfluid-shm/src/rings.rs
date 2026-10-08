//! The shm ring layout and the [`SharedRings`] implementation of `superfluid_engine::Rings` both link
//! ends use.

use std::sync::atomic::{fence, AtomicU64, Ordering};

use superfluid_abi::{RingRef, TokenRef};
use superfluid_engine::rings::{RingError, Rings};

use crate::{control::ControlSegment, segment::ShmSegment, ShmError};

pub const TOKEN_RING_IN: u32 = 1;
pub const TOKEN_RING_OUT: u32 = 2;
pub const LOGITS_RING: u32 = 3;

const RING_MAGIC: u64 = 0x4252_5452_494E_4731;
const HEADER_BYTES: usize = 64;
const SLOT_HEADER_BYTES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingRole {
    Writer,
    Reader,
}

#[derive(Debug)]
pub struct SharedRing {
    pub ring_id: u32,
    seg: ShmSegment,
    slots: u32,
    slot_bytes: u32,
    role: RingRole,
    next_index: u32,
    next_generation: u64,
}

impl SharedRing {
    pub fn required_len(slots: u32, slot_bytes: u32) -> usize {
        HEADER_BYTES + slots as usize * slot_bytes as usize
    }

    /// The smallest slot that holds `payload_bytes` and keeps every slot's
    /// generation word 8-byte aligned.
    pub fn slot_bytes_for(payload_bytes: u32) -> u32 {
        (SLOT_HEADER_BYTES as u32 + payload_bytes).next_multiple_of(8)
    }

    fn check_slot_bytes(slot_bytes: u32) -> Result<(), ShmError> {
        if slot_bytes.is_multiple_of(8) {
            Ok(())
        } else {
            Err(ShmError::Misaligned(HEADER_BYTES + slot_bytes as usize))
        }
    }

    pub fn create(
        ring_id: u32,
        slots: u32,
        slot_bytes: u32,
        role: RingRole,
    ) -> Result<SharedRing, ShmError> {
        assert!(slot_bytes as usize > SLOT_HEADER_BYTES);
        Self::check_slot_bytes(slot_bytes)?;
        let seg = ShmSegment::create(Self::required_len(slots, slot_bytes))?;
        seg.write_at(0, &RING_MAGIC.to_le_bytes())?;
        seg.write_at(8, &ring_id.to_le_bytes())?;
        seg.write_at(12, &slot_bytes.to_le_bytes())?;
        seg.write_at(16, &slots.to_le_bytes())?;
        Ok(SharedRing {
            ring_id,
            seg,
            slots,
            slot_bytes,
            role,
            next_index: 0,
            next_generation: 1,
        })
    }

    pub fn attach(
        seg: ShmSegment,
        ring_id: u32,
        slots: u32,
        slot_bytes: u32,
        role: RingRole,
    ) -> Result<SharedRing, ShmError> {
        Self::check_slot_bytes(slot_bytes)?;
        let mut hdr = [0u8; 20];
        seg.read_at(0, &mut hdr)?;
        let magic = u64::from_le_bytes(hdr[0..8].try_into().expect("8 bytes"));
        let hdr_ring = u32::from_le_bytes(hdr[8..12].try_into().expect("4 bytes"));
        let hdr_slot_bytes = u32::from_le_bytes(hdr[12..16].try_into().expect("4 bytes"));
        let hdr_slots = u32::from_le_bytes(hdr[16..20].try_into().expect("4 bytes"));
        if magic != RING_MAGIC
            || hdr_ring != ring_id
            || hdr_slot_bytes != slot_bytes
            || hdr_slots != slots
        {
            return Err(ShmError::BadMagic);
        }
        if seg.len() < Self::required_len(slots, slot_bytes) {
            return Err(ShmError::TooSmall);
        }
        Ok(SharedRing {
            ring_id,
            seg,
            slots,
            slot_bytes,
            role,
            next_index: 0,
            next_generation: 1,
        })
    }

    pub fn raw_fd(&self) -> std::os::fd::RawFd {
        self.seg.raw_fd()
    }

    pub fn seg_len(&self) -> usize {
        self.seg.len()
    }

    pub fn base_ptr(&self) -> *mut u8 {
        self.seg.as_ptr()
    }

    fn slot_offset(&self, index: u32) -> usize {
        HEADER_BYTES + index as usize * self.slot_bytes as usize
    }

    fn generation(&self, index: u32) -> Result<&AtomicU64, ShmError> {
        self.seg.atomic_u64_at(self.slot_offset(index))
    }

    fn payload_capacity(&self) -> usize {
        self.slot_bytes as usize - SLOT_HEADER_BYTES
    }

    pub fn push(&mut self, payload: &[u8]) -> Result<(u32, u64), ShmError> {
        assert_eq!(self.role, RingRole::Writer, "push on the reader side");
        if payload.len() > self.payload_capacity() {
            return Err(ShmError::PayloadTooLarge);
        }
        let index = self.next_index;
        let generation = self.next_generation;
        let off = self.slot_offset(index);
        let word = self.generation(index)?;
        word.store(0, Ordering::Relaxed);
        fence(Ordering::Release);
        self.seg
            .write_at(off + 8, &(payload.len() as u32).to_le_bytes())?;
        self.seg.write_at(off + 12, &0u32.to_le_bytes())?;
        self.seg.write_at(off + SLOT_HEADER_BYTES, payload)?;
        word.store(generation.to_le(), Ordering::Release);
        self.next_index = (index + 1) % self.slots;
        self.next_generation += 1;
        Ok((index, generation))
    }

    pub fn get(&self, index: u32, generation: u64) -> Result<Vec<u8>, RingError> {
        if index >= self.slots {
            return Err(RingError::OutOfBounds);
        }
        if generation == 0 {
            return Err(RingError::StaleGeneration {
                reference: 0,
                current: 0,
            });
        }
        let off = self.slot_offset(index);
        let word = self.generation(index).map_err(|_| RingError::OutOfBounds)?;
        let current = u64::from_le(word.load(Ordering::Acquire));
        if current != generation {
            return Err(RingError::StaleGeneration {
                reference: generation,
                current,
            });
        }
        let mut len_bytes = [0u8; 4];
        self.seg
            .read_at(off + 8, &mut len_bytes)
            .map_err(|_| RingError::OutOfBounds)?;
        let len = u32::from_le_bytes(len_bytes) as usize;
        if len > self.payload_capacity() {
            return Err(RingError::OutOfBounds);
        }
        let mut out = vec![0u8; len];
        self.seg
            .read_at(off + SLOT_HEADER_BYTES, &mut out)
            .map_err(|_| RingError::OutOfBounds)?;
        fence(Ordering::Acquire);
        let after = u64::from_le(word.load(Ordering::Relaxed));
        if after != generation {
            return Err(RingError::StaleGeneration {
                reference: generation,
                current: after,
            });
        }
        Ok(out)
    }
}

fn tokens_to_bytes(tokens: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out
}

fn bytes_to_tokens(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

fn floats_to_bytes(row: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(row.len() * 4);
    for f in row {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn bytes_to_floats(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

#[derive(Debug)]
pub struct SharedRings {
    pub token_in: SharedRing,
    pub token_out: SharedRing,
    pub logits: SharedRing,
    /// The daemon's control words, once it has passed them.
    pub control: Option<ControlSegment>,
}

impl SharedRings {
    pub fn stage_prompt(&mut self, tokens: &[u32]) -> Result<TokenRef, ShmError> {
        let (index, generation) = self.token_in.push(&tokens_to_bytes(tokens))?;
        Ok(TokenRef {
            ring_id: TOKEN_RING_IN,
            index,
            count: tokens.len() as u32,
            _pad0: 0,
            generation,
        })
    }

    fn token_ring(&self, ring_id: u32) -> Result<&SharedRing, RingError> {
        match ring_id {
            TOKEN_RING_IN => Ok(&self.token_in),
            TOKEN_RING_OUT => Ok(&self.token_out),
            other => Err(RingError::UnknownRing(other)),
        }
    }
}

impl superfluid_engine::testing::StageRings for SharedRings {
    fn stage_prompt(&mut self, tokens: &[u32]) -> TokenRef {
        SharedRings::stage_prompt(self, tokens).expect("prompt exceeds the token ring's slot capacity")
    }
}

impl Rings for SharedRings {
    fn read_tokens(&self, r: &TokenRef) -> Result<Vec<u32>, RingError> {
        let ring = self.token_ring(r.ring_id)?;
        let tokens = bytes_to_tokens(&ring.get(r.index, r.generation)?);
        if tokens.len() != r.count as usize {
            return Err(RingError::OutOfBounds);
        }
        Ok(tokens)
    }

    fn write_tokens(&mut self, tokens: &[u32]) -> Result<TokenRef, RingError> {
        let (index, generation) = self
            .token_out
            .push(&tokens_to_bytes(tokens))
            .map_err(|_| RingError::OutOfBounds)?;
        Ok(TokenRef {
            ring_id: TOKEN_RING_OUT,
            index,
            count: tokens.len() as u32,
            _pad0: 0,
            generation,
        })
    }

    fn write_logits_row(&mut self, row: &[f32]) -> RingRef {
        let (index, generation) = self
            .logits
            .push(&floats_to_bytes(row))
            .expect("logits row exceeds ring slot capacity");
        RingRef {
            ring_id: LOGITS_RING,
            index,
            generation,
        }
    }

    fn read_logits_row(&self, r: &RingRef) -> Result<Vec<f32>, RingError> {
        if r.ring_id != LOGITS_RING {
            return Err(RingError::UnknownRing(r.ring_id));
        }
        Ok(bytes_to_floats(&self.logits.get(r.index, r.generation)?))
    }

    fn yield_requested(&self) -> bool {
        self.control.as_ref().is_some_and(ControlSegment::yield_requested)
    }

    fn latency_wanted(&self) -> bool {
        self.control.as_ref().is_some_and(ControlSegment::latency_wanted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdpass::{recv_fd, send_fd};
    use crate::segment::ShmSegment;
    use std::os::fd::FromRawFd;
    use std::os::unix::net::UnixStream;

    #[test]
    fn ring_push_get_and_staleness() {
        let mut w = SharedRing::create(7, 4, 64, RingRole::Writer).unwrap();
        let (i0, g0) = w.push(b"abc").unwrap();
        assert_eq!(w.get(i0, g0).unwrap(), b"abc");
        for k in 0..4 {
            w.push(&[k as u8; 8]).unwrap();
        }
        let err = w.get(i0, g0).unwrap_err();
        assert!(matches!(err, RingError::StaleGeneration { .. }));
    }

    #[test]
    fn ring_shared_across_fd_pass() {
        let (a, b) = UnixStream::pair().unwrap();
        let mut writer = SharedRing::create(9, 8, 128, RingRole::Writer).unwrap();
        send_fd(&a, writer.raw_fd(), 9).unwrap();
        let (fd, tag) = recv_fd(&b).unwrap();
        let seg = ShmSegment::from_fd(fd, writer.seg_len()).unwrap();
        let reader = SharedRing::attach(seg, tag as u32, 8, 128, RingRole::Reader).unwrap();

        let (i, g) = writer.push(b"cross-process payload").unwrap();
        assert_eq!(reader.get(i, g).unwrap(), b"cross-process payload");
        assert!(matches!(
            reader.get(i, g + 1).unwrap_err(),
            RingError::StaleGeneration { .. }
        ));
    }

    #[test]
    fn a_slot_that_would_misalign_the_generation_word_is_refused() {
        let odd = 16 + 5 * 4;
        assert!(matches!(
            SharedRing::create(7, 4, odd, RingRole::Writer).unwrap_err(),
            ShmError::Misaligned(100)
        ));
        let seg = ShmSegment::create(SharedRing::required_len(4, odd)).unwrap();
        assert!(matches!(
            SharedRing::attach(seg, 7, 4, odd, RingRole::Reader).unwrap_err(),
            ShmError::Misaligned(100)
        ));

        let fits = SharedRing::slot_bytes_for(5 * 4);
        assert_eq!(fits, 40);
        let mut w = SharedRing::create(7, 4, fits, RingRole::Writer).unwrap();
        for k in 0..6u8 {
            let (i, g) = w.push(&[k; 20]).unwrap();
            assert_eq!(w.get(i, g).unwrap(), [k; 20]);
        }
    }

    #[test]
    fn an_emit_larger_than_a_slot_is_an_error_not_a_panic() {
        let mut rings = SharedRings {
            token_in: SharedRing::create(TOKEN_RING_IN, 2, 32, RingRole::Writer).unwrap(),
            token_out: SharedRing::create(TOKEN_RING_OUT, 2, SharedRing::slot_bytes_for(1024 * 4), RingRole::Writer).unwrap(),
            logits: SharedRing::create(LOGITS_RING, 2, 32, RingRole::Writer).unwrap(),
            control: None,
        };
        let fits = rings.write_tokens(&[7; 1024]).unwrap();
        assert_eq!(rings.read_tokens(&fits).unwrap(), vec![7; 1024]);
        assert_eq!(rings.write_tokens(&[7; 1025]).unwrap_err(), RingError::OutOfBounds);
    }

    #[test]
    fn a_reader_racing_the_writer_sees_whole_payloads_or_staleness() {
        let (slots, rounds) = (4u32, 20_000u64);
        let mut writer = SharedRing::create(5, slots, 16 + 64, RingRole::Writer).unwrap();
        let dup = writer.seg.raw_fd();
        // SAFETY: dup(2) of a live fd; the new descriptor is owned below.
        let fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(libc::dup(dup)) };
        let seg = ShmSegment::from_fd(fd, writer.seg_len()).unwrap();
        let reader = SharedRing::attach(seg, 5, slots, 16 + 64, RingRole::Reader).unwrap();
        let read = std::thread::spawn(move || {
            for g in 1..=rounds {
                let index = ((g - 1) % slots as u64) as u32;
                loop {
                    match reader.get(index, g) {
                        Ok(p) => {
                            assert_eq!(p, vec![g as u8; 64], "generation {g}");
                            break;
                        }
                        Err(RingError::StaleGeneration { current, .. }) if current > g => break,
                        Err(RingError::StaleGeneration { .. }) => std::hint::spin_loop(),
                        Err(e) => panic!("{e}"),
                    }
                }
            }
        });
        for g in 1..=rounds {
            writer.push(&[g as u8; 64]).unwrap();
        }
        read.join().unwrap();
    }
}
