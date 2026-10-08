//! Envelope-level integration tests.

use superfluid_proto::envelope::{encode_frame, EnvelopeError, FrameClass, FrameDecoder, MAX_FRAME};
use superfluid_proto::linkw;

fn always_known(_: u16) -> bool {
    true
}

#[test]
fn round_trip_single_frame() {
    let payload = b"hello superfluid".to_vec();
    let bytes = encode_frame(
        7,
        linkw::msg_type::PING,
        123,
        456,
        FrameClass::Required,
        &payload,
    )
    .expect("encode");

    let mut dec = FrameDecoder::new(linkw::is_known_type);
    dec.feed(&bytes);
    let frame = dec.decode_next().unwrap().expect("one frame");
    assert_eq!(frame.proto_version, 7);
    assert_eq!(frame.msg_type, linkw::msg_type::PING);
    assert_eq!(frame.seq, 123);
    assert_eq!(frame.correlation_id, 456);
    assert_eq!(frame.class, FrameClass::Required);
    assert_eq!(frame.payload, payload);

    assert_eq!(dec.decode_next().unwrap(), None);
    dec.finish().unwrap();
}

#[test]
fn round_trip_empty_payload() {
    let bytes = encode_frame(1, linkw::msg_type::DRAINED, 1, 0, FrameClass::Required, &[]).unwrap();
    let mut dec = FrameDecoder::new(linkw::is_known_type);
    dec.feed(&bytes);
    let frame = dec.decode_next().unwrap().unwrap();
    assert!(frame.payload.is_empty());
}

#[test]
fn multiple_frames_in_one_feed() {
    let a = encode_frame(1, linkw::msg_type::PING, 1, 0, FrameClass::Required, b"a").unwrap();
    let b = encode_frame(1, linkw::msg_type::PONG, 2, 1, FrameClass::Required, b"bb").unwrap();
    let mut both = a.clone();
    both.extend_from_slice(&b);

    let mut dec = FrameDecoder::new(linkw::is_known_type);
    dec.feed(&both);
    let f1 = dec.decode_next().unwrap().unwrap();
    let f2 = dec.decode_next().unwrap().unwrap();
    assert_eq!(f1.msg_type, linkw::msg_type::PING);
    assert_eq!(f2.msg_type, linkw::msg_type::PONG);
    assert_eq!(dec.decode_next().unwrap(), None);
}

#[test]
fn decoder_handles_byte_at_a_time_feed() {
    let payload: Vec<u8> = (0..200u16).map(|i| i as u8).collect();
    let bytes = encode_frame(
        3,
        linkw::msg_type::TELEMETRY,
        9,
        9,
        FrameClass::Droppable,
        &payload,
    )
    .unwrap();

    let mut dec = FrameDecoder::new(linkw::is_known_type);
    for &b in &bytes[..bytes.len() - 1] {
        dec.feed(&[b]);
        assert_eq!(dec.decode_next().unwrap(), None, "must not decode early");
    }
    dec.feed(&bytes[bytes.len() - 1..]);
    let frame = dec.decode_next().unwrap().expect("frame after final byte");
    assert_eq!(frame.payload, payload);
    assert_eq!(frame.class, FrameClass::Droppable);
}

#[test]
fn cap_enforced_on_encode() {
    let too_big = vec![0u8; MAX_FRAME as usize - 20];
    let err = encode_frame(
        1,
        linkw::msg_type::PING,
        0,
        0,
        FrameClass::Required,
        &too_big,
    )
    .unwrap_err();
    assert!(matches!(err, EnvelopeError::FrameTooLarge { .. }));
}

#[test]
fn cap_enforced_on_decode_without_buffering_full_payload() {
    let claimed_len = MAX_FRAME + 1;
    let mut header = Vec::new();
    header.extend_from_slice(&claimed_len.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&linkw::msg_type::PING.to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());
    header.push(0);

    let mut dec = FrameDecoder::new(always_known);
    dec.feed(&header);
    let err = dec.decode_next().unwrap_err();
    assert_eq!(
        err,
        EnvelopeError::FrameTooLarge {
            len: claimed_len,
            max: MAX_FRAME
        }
    );
}

#[test]
fn bad_length_below_header_size_rejected() {
    let mut header = Vec::new();
    header.extend_from_slice(&20u32.to_le_bytes());
    header.extend_from_slice(&[0u8; 20]);

    let mut dec = FrameDecoder::new(always_known);
    dec.feed(&header);
    let err = dec.decode_next().unwrap_err();
    assert_eq!(err, EnvelopeError::BadLength { len: 20, min: 21 });
}

#[test]
fn class_range_mismatch_rejected() {
    let mut buf = Vec::new();
    let payload = b"x";
    let len = 21u32 + payload.len() as u32;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&linkw::msg_type::PING.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.push(1);
    buf.extend_from_slice(payload);

    let mut dec = FrameDecoder::new(always_known);
    dec.feed(&buf);
    let err = dec.decode_next().unwrap_err();
    assert!(matches!(err, EnvelopeError::ClassRangeMismatch { .. }));
}

#[test]
fn encode_rejects_class_range_mismatch_too() {
    let err = encode_frame(
        1,
        linkw::msg_type::PING,
        0,
        0,
        FrameClass::Droppable,
        b"x",
    )
    .unwrap_err();
    assert!(matches!(err, EnvelopeError::ClassRangeMismatch { .. }));
}

#[test]
fn unknown_required_type_is_a_protocol_error() {
    let unknown_required: u16 = 0x0FFF;
    assert!(!linkw::is_known_type(unknown_required));
    let bytes = encode_frame(1, unknown_required, 0, 0, FrameClass::Required, b"x").unwrap();

    let mut dec = FrameDecoder::new(linkw::is_known_type);
    dec.feed(&bytes);
    let err = dec.decode_next().unwrap_err();
    assert_eq!(
        err,
        EnvelopeError::UnknownRequiredType {
            msg_type: unknown_required
        }
    );
}

#[test]
fn unknown_droppable_type_is_skipped_safely() {
    let unknown_droppable: u16 = 0x8FFF;
    assert!(!linkw::is_known_type(unknown_droppable));
    let skipped = encode_frame(
        1,
        unknown_droppable,
        1,
        0,
        FrameClass::Droppable,
        b"ignored",
    )
    .unwrap();
    let known = encode_frame(
        1,
        linkw::msg_type::PONG,
        2,
        0,
        FrameClass::Required,
        b"kept",
    )
    .unwrap();

    let mut combined = skipped;
    combined.extend_from_slice(&known);

    let mut dec = FrameDecoder::new(linkw::is_known_type);
    dec.feed(&combined);
    let frame = dec.decode_next().unwrap().expect("known frame survives");
    assert_eq!(frame.msg_type, linkw::msg_type::PONG);
    assert_eq!(frame.payload, b"kept");
    assert_eq!(dec.decode_next().unwrap(), None);
}

#[test]
fn truncated_header_reported_on_finish() {
    let bytes = encode_frame(
        1,
        linkw::msg_type::PING,
        0,
        0,
        FrameClass::Required,
        b"hello",
    )
    .unwrap();
    let mut dec = FrameDecoder::new(always_known);
    dec.feed(&bytes[..bytes.len() - 3]);
    assert_eq!(dec.decode_next().unwrap(), None);
    let err = dec.finish().unwrap_err();
    assert!(matches!(err, EnvelopeError::Truncated { .. }));
}

#[test]
fn truncated_within_header_itself_reported_on_finish() {
    let mut dec = FrameDecoder::new(always_known);
    dec.feed(&[1, 2, 3]);
    assert_eq!(dec.decode_next().unwrap(), None);
    let err = dec.finish().unwrap_err();
    assert_eq!(err, EnvelopeError::Truncated { buffered: 3 });
}

#[test]
fn clean_finish_after_all_frames_consumed() {
    let bytes = encode_frame(1, linkw::msg_type::PING, 0, 0, FrameClass::Required, b"ok").unwrap();
    let mut dec = FrameDecoder::new(always_known);
    dec.feed(&bytes);
    dec.decode_next().unwrap().unwrap();
    dec.finish().unwrap();
}
