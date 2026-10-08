//! Link F over an encrypted stream: Noise `NNpsk0` keyed by the fleet token, so a head and a
//! node that share the token read each other and nobody else does.

use std::io::{self, ErrorKind, Read, Write};

const PATTERN: &str = "Noise_NNpsk0_25519_ChaChaPoly_BLAKE2s";
const MAX_MESSAGE: usize = 65535;
const TAG: usize = 16;
const MAX_PLAINTEXT: usize = MAX_MESSAGE - TAG;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Initiator,
    Responder,
}

/// The handshake key a shared secret (the token file's bytes) gives.
pub fn key_from(secret: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    sha2::Sha256::digest(secret).into()
}

/// Each Noise message on the wire is a big-endian `u16` length and the message.
pub struct SecureStream<S> {
    inner: S,
    state: snow::TransportState,
    plain: Vec<u8>,
    plain_at: usize,
    cipher: Vec<u8>,
    scratch: Vec<u8>,
}

fn other<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

impl<S: Read + Write> SecureStream<S> {
    /// Runs the handshake on `inner`, blocking within its read timeout, and wraps it.
    /// A peer with another key fails with `PermissionDenied`.
    pub fn handshake(mut inner: S, key: &[u8; 32], role: Role) -> io::Result<SecureStream<S>> {
        let params: snow::params::NoiseParams = PATTERN.parse().map_err(other)?;
        let builder = snow::Builder::new(params).psk(0, key).map_err(other)?;
        let mut hs = match role {
            Role::Initiator => builder.build_initiator(),
            Role::Responder => builder.build_responder(),
        }
        .map_err(other)?;
        let mut buf = vec![0u8; MAX_MESSAGE];
        let refused = |e: snow::Error| io::Error::new(ErrorKind::PermissionDenied, format!("the peer's fleet token differs ({e})"));
        match role {
            Role::Initiator => {
                let n = hs.write_message(&[], &mut buf).map_err(other)?;
                write_message(&mut inner, &buf[..n])?;
                let msg = read_message(&mut inner)?;
                hs.read_message(&msg, &mut buf).map_err(refused)?;
            }
            Role::Responder => {
                let msg = read_message(&mut inner)?;
                hs.read_message(&msg, &mut buf).map_err(refused)?;
                let n = hs.write_message(&[], &mut buf).map_err(other)?;
                write_message(&mut inner, &buf[..n])?;
            }
        }
        let state = hs.into_transport_mode().map_err(other)?;
        Ok(SecureStream { inner, state, plain: Vec::new(), plain_at: 0, cipher: Vec::new(), scratch: vec![0u8; 2 + MAX_MESSAGE] })
    }

    pub fn get_ref(&self) -> &S {
        &self.inner
    }

    fn decrypt_complete_messages(&mut self) -> io::Result<()> {
        let mut at = 0;
        while self.cipher.len() - at >= 2 {
            let len = u16::from_be_bytes([self.cipher[at], self.cipher[at + 1]]) as usize;
            if self.cipher.len() - at - 2 < len {
                break;
            }
            let n = self
                .state
                .read_message(&self.cipher[at + 2..at + 2 + len], &mut self.scratch)
                .map_err(|e| io::Error::new(ErrorKind::InvalidData, format!("a message that is not ours: {e}")))?;
            self.plain.extend_from_slice(&self.scratch[..n]);
            at += 2 + len;
        }
        self.cipher.drain(..at);
        Ok(())
    }
}

impl<S: Read + Write> Read for SecureStream<S> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let mut tmp = [0u8; 64 * 1024];
        loop {
            if self.plain_at < self.plain.len() {
                let n = out.len().min(self.plain.len() - self.plain_at);
                out[..n].copy_from_slice(&self.plain[self.plain_at..self.plain_at + n]);
                self.plain_at += n;
                if self.plain_at == self.plain.len() {
                    self.plain.clear();
                    self.plain_at = 0;
                }
                return Ok(n);
            }
            let n = self.inner.read(&mut tmp)?;
            if n == 0 {
                return Ok(0);
            }
            self.cipher.extend_from_slice(&tmp[..n]);
            self.decrypt_complete_messages()?;
        }
    }
}

impl<S: Read + Write> Write for SecureStream<S> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        for chunk in buf.chunks(MAX_PLAINTEXT) {
            let n = self.state.write_message(chunk, &mut self.scratch[2..]).map_err(other)?;
            self.scratch[..2].copy_from_slice(&(n as u16).to_be_bytes());
            self.inner.write_all(&self.scratch[..2 + n])?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn write_message<W: Write>(w: &mut W, m: &[u8]) -> io::Result<()> {
    let mut framed = Vec::with_capacity(2 + m.len());
    framed.extend_from_slice(&(m.len() as u16).to_be_bytes());
    framed.extend_from_slice(m);
    w.write_all(&framed)
}

fn read_message<R: Read>(r: &mut R) -> io::Result<Vec<u8>> {
    let mut len = [0u8; 2];
    r.read_exact(&mut len)?;
    let mut m = vec![0u8; u16::from_be_bytes(len) as usize];
    r.read_exact(&mut m)?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixStream;

    fn pair(a: &[u8], b: &[u8]) -> (io::Result<SecureStream<UnixStream>>, io::Result<SecureStream<UnixStream>>) {
        let (x, y) = UnixStream::pair().unwrap();
        let (ka, kb) = (key_from(a), key_from(b));
        let responder = std::thread::spawn(move || SecureStream::handshake(y, &kb, Role::Responder));
        let initiator = SecureStream::handshake(x, &ka, Role::Initiator);
        (initiator, responder.join().unwrap())
    }

    #[test]
    fn one_token_both_ways_carries_small_and_large_messages() {
        let (head, node) = pair(b"token", b"token");
        let (mut head, mut node) = (head.unwrap(), node.unwrap());
        let big: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let sent = big.clone();
        let writer = std::thread::spawn(move || {
            head.write_all(b"hello").unwrap();
            head.write_all(&sent).unwrap();
            head.flush().unwrap();
            let mut reply = [0u8; 3];
            head.read_exact(&mut reply).unwrap();
            reply
        });
        let mut got = vec![0u8; 5 + big.len()];
        node.read_exact(&mut got).unwrap();
        assert_eq!(&got[..5], b"hello");
        assert_eq!(&got[5..], &big[..]);
        node.write_all(b"ack").unwrap();
        assert_eq!(&writer.join().unwrap(), b"ack");
    }

    #[test]
    fn another_token_is_refused_at_the_handshake() {
        let (_head, node) = pair(b"token", b"other");
        match node {
            Err(e) => assert_eq!(e.kind(), ErrorKind::PermissionDenied),
            Ok(_) => panic!("a node with another token accepted the head"),
        }
    }

    /// A stream that keeps a copy of every byte written through it.
    struct Tap<S> {
        inner: S,
        written: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    }

    impl<S: Read> Read for Tap<S> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.inner.read(buf)
        }
    }

    impl<S: Write> Write for Tap<S> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.written.lock().unwrap().extend_from_slice(buf);
            self.inner.write(buf)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.inner.flush()
        }
    }

    #[test]
    fn the_wire_is_not_the_plaintext() {
        let (a, b) = UnixStream::pair().unwrap();
        let key = key_from(b"token");
        let written = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let tap = Tap { inner: a, written: std::sync::Arc::clone(&written) };
        let responder = std::thread::spawn(move || {
            let mut node = SecureStream::handshake(b, &key, Role::Responder).unwrap();
            let mut got = [0u8; 15];
            node.read_exact(&mut got).unwrap();
            got
        });
        let mut head = SecureStream::handshake(tap, &key, Role::Initiator).unwrap();
        head.write_all(b"a secret prompt").unwrap();
        assert_eq!(&responder.join().unwrap(), b"a secret prompt");
        let wire = written.lock().unwrap().clone();
        assert!(wire.len() > 15 + 2 + TAG, "the wire carried the handshake and one framed message");
        assert!(!wire.windows(15).any(|w| w == b"a secret prompt"), "the plaintext is on the wire");
    }
}
