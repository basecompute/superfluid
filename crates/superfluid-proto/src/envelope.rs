//! The wire envelope shared by Link W and Link F..

pub const MAX_FRAME: u32 = 16 * 1024 * 1024;

pub const HEADER_LEN: u32 = 21;

pub const DROPPABLE_BIT: u16 = 0x8000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameClass {
    Required = 0,
    Droppable = 1,
}

impl FrameClass {
    fn from_byte(b: u8) -> Option<FrameClass> {
        match b {
            0 => Some(FrameClass::Required),
            1 => Some(FrameClass::Droppable),
            _ => None,
        }
    }
}

pub fn range_class(msg_type: u16) -> FrameClass {
    if msg_type & DROPPABLE_BIT != 0 {
        FrameClass::Droppable
    } else {
        FrameClass::Required
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub proto_version: u16,
    pub msg_type: u16,
    pub seq: u64,
    pub correlation_id: u64,
    pub class: FrameClass,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EnvelopeError {
    #[error("frame length {len} exceeds the {max} byte cap (schema §4.1)")]
    FrameTooLarge { len: u32, max: u32 },

    #[error(
        "envelope class byte {class_byte} for msg_type 0x{msg_type:04x} disagrees with its range (expected {expected:?})"
    )]
    ClassRangeMismatch {
        msg_type: u16,
        class_byte: u8,
        expected: FrameClass,
    },

    #[error(
        "unknown msg_type 0x{msg_type:04x} in the REQUIRED range: protocol error, fail-closed"
    )]
    UnknownRequiredType { msg_type: u16 },

    #[error("frame length {len} is smaller than the {min} byte header")]
    BadLength { len: u32, min: u32 },

    #[error("stream ended with a truncated frame ({buffered} bytes buffered, incomplete)")]
    Truncated { buffered: usize },
}

pub fn encode_frame(
    proto_version: u16,
    msg_type: u16,
    seq: u64,
    correlation_id: u64,
    class: FrameClass,
    payload: &[u8],
) -> Result<Vec<u8>, EnvelopeError> {
    let expected = range_class(msg_type);
    if class != expected {
        return Err(EnvelopeError::ClassRangeMismatch {
            msg_type,
            class_byte: class as u8,
            expected,
        });
    }

    let len = HEADER_LEN as usize + payload.len();
    let len_u32 = u32::try_from(len).unwrap_or(u32::MAX);
    if len_u32 > MAX_FRAME {
        return Err(EnvelopeError::FrameTooLarge {
            len: len_u32,
            max: MAX_FRAME,
        });
    }

    let mut buf = Vec::with_capacity(4 + len);
    buf.extend_from_slice(&len_u32.to_le_bytes());
    buf.extend_from_slice(&proto_version.to_le_bytes());
    buf.extend_from_slice(&msg_type.to_le_bytes());
    buf.extend_from_slice(&seq.to_le_bytes());
    buf.extend_from_slice(&correlation_id.to_le_bytes());
    buf.push(class as u8);
    buf.extend_from_slice(payload);
    Ok(buf)
}

pub struct FrameDecoder<F: Fn(u16) -> bool> {
    buf: Vec<u8>,
    is_known: F,
}

impl<F: Fn(u16) -> bool> FrameDecoder<F> {
    pub fn new(is_known: F) -> Self {
        Self {
            buf: Vec::new(),
            is_known,
        }
    }

    pub fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    pub fn decode_next(&mut self) -> Result<Option<Frame>, EnvelopeError> {
        loop {
            if self.buf.len() < 4 {
                return Ok(None);
            }
            let len = u32::from_le_bytes(self.buf[0..4].try_into().expect("4 bytes"));
            if len < HEADER_LEN {
                return Err(EnvelopeError::BadLength {
                    len,
                    min: HEADER_LEN,
                });
            }
            if len > MAX_FRAME {
                return Err(EnvelopeError::FrameTooLarge {
                    len,
                    max: MAX_FRAME,
                });
            }

            let total = 4usize + len as usize;
            if self.buf.len() < total {
                return Ok(None);
            }

            let proto_version = u16::from_le_bytes(self.buf[4..6].try_into().expect("2 bytes"));
            let msg_type = u16::from_le_bytes(self.buf[6..8].try_into().expect("2 bytes"));
            let seq = u64::from_le_bytes(self.buf[8..16].try_into().expect("8 bytes"));
            let correlation_id = u64::from_le_bytes(self.buf[16..24].try_into().expect("8 bytes"));
            let class_byte = self.buf[24];
            let payload = self.buf[25..total].to_vec();

            self.buf.drain(0..total);

            let expected = range_class(msg_type);
            let class = match FrameClass::from_byte(class_byte) {
                Some(c) if c == expected => c,
                _ => {
                    return Err(EnvelopeError::ClassRangeMismatch {
                        msg_type,
                        class_byte,
                        expected,
                    });
                }
            };

            if !(self.is_known)(msg_type) {
                match expected {
                    FrameClass::Required => {
                        return Err(EnvelopeError::UnknownRequiredType { msg_type });
                    }
                    FrameClass::Droppable => continue,
                }
            }

            return Ok(Some(Frame {
                proto_version,
                msg_type,
                seq,
                correlation_id,
                class,
                payload,
            }));
        }
    }

    pub fn finish(&self) -> Result<(), EnvelopeError> {
        if self.buf.is_empty() {
            Ok(())
        } else {
            Err(EnvelopeError::Truncated {
                buffered: self.buf.len(),
            })
        }
    }
}
