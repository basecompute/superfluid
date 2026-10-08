//! A thin framed endpoint for running the Link F state machines over a byte stream.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use superfluid_proto::envelope::FrameClass;
use superfluid_proto::handshake::{negotiate, Hello, HelloAckF, LinkRole};
use superfluid_proto::linkf::{self, msg_type};
use superfluid_proto::{encode_frame, Frame, FrameDecoder};

use crate::secure::{key_from, Role, SecureStream};
use crate::LinkFError;

pub const PROTO_VERSION: u16 = 2;

enum Transport {
    Unix(UnixStream),
    Tcp(TcpStream),
    Secure(Box<SecureStream<TcpStream>>),
}

impl Transport {
    fn set_read_timeout(&self, dur: Option<Duration>) -> std::io::Result<()> {
        match self {
            Transport::Unix(s) => s.set_read_timeout(dur),
            Transport::Tcp(s) => s.set_read_timeout(dur),
            Transport::Secure(s) => s.get_ref().set_read_timeout(dur),
        }
    }

    fn shutdown(&self) -> std::io::Result<()> {
        match self {
            Transport::Unix(s) => s.shutdown(std::net::Shutdown::Both),
            Transport::Tcp(s) => s.shutdown(std::net::Shutdown::Both),
            Transport::Secure(s) => s.get_ref().shutdown(std::net::Shutdown::Both),
        }
    }
}

impl Read for Transport {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Transport::Unix(s) => s.read(buf),
            Transport::Tcp(s) => s.read(buf),
            Transport::Secure(s) => s.read(buf),
        }
    }
}

impl Write for Transport {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Transport::Unix(s) => s.write(buf),
            Transport::Tcp(s) => s.write(buf),
            Transport::Secure(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Transport::Unix(s) => s.flush(),
            Transport::Tcp(s) => s.flush(),
            Transport::Secure(s) => s.flush(),
        }
    }
}

pub struct Endpoint {
    stream: Transport,
    decoder: FrameDecoder<fn(u16) -> bool>,
    seq: u64,
}

impl Endpoint {
    pub fn new(stream: UnixStream) -> Result<Endpoint, LinkFError> {
        Endpoint::wrap(Transport::Unix(stream))
    }

    pub fn from_tcp(stream: TcpStream) -> Result<Endpoint, LinkFError> {
        stream.set_nodelay(true)?;
        Endpoint::wrap(Transport::Tcp(stream))
    }

    pub fn connect_tcp<A: ToSocketAddrs>(addr: A) -> Result<Endpoint, LinkFError> {
        Endpoint::from_tcp(TcpStream::connect(addr)?)
    }

    pub fn connect_tcp_within<A: ToSocketAddrs>(addr: A, timeout: Duration) -> Result<Endpoint, LinkFError> {
        Endpoint::from_tcp(Endpoint::tcp_connect_within(addr, timeout)?)
    }

    /// An encrypted endpoint keyed by `secret`, the initiator's side; `handshake` bounds the
    /// peer's reply. A peer with another secret is `Unauthorized`.
    pub fn connect_tcp_secure<A: ToSocketAddrs>(addr: A, secret: &[u8], handshake: Duration) -> Result<Endpoint, LinkFError> {
        Endpoint::from_tcp_secure(TcpStream::connect(addr)?, secret, Role::Initiator, handshake)
    }

    pub fn connect_tcp_secure_within<A: ToSocketAddrs>(
        addr: A,
        secret: &[u8],
        connect: Duration,
        handshake: Duration,
    ) -> Result<Endpoint, LinkFError> {
        Endpoint::from_tcp_secure(Endpoint::tcp_connect_within(addr, connect)?, secret, Role::Initiator, handshake)
    }

    /// A TCP connection to `addr` within `timeout`, before any Link F or Noise bytes.
    pub fn tcp_connect_within<A: ToSocketAddrs>(addr: A, timeout: Duration) -> std::io::Result<TcpStream> {
        let mut last = None;
        for a in addr.to_socket_addrs()? {
            match TcpStream::connect_timeout(&a, timeout) {
                Ok(s) => return Ok(s),
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "address resolved to nothing")))
    }

    pub fn from_tcp_secure(stream: TcpStream, secret: &[u8], role: Role, handshake: Duration) -> Result<Endpoint, LinkFError> {
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(handshake))?;
        let secure = SecureStream::handshake(stream, &key_from(secret), role).map_err(|e| match e.kind() {
            std::io::ErrorKind::PermissionDenied => LinkFError::Unauthorized,
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => LinkFError::TimedOut,
            _ => LinkFError::Io(e),
        })?;
        Endpoint::wrap(Transport::Secure(Box::new(secure)))
    }

    fn wrap(stream: Transport) -> Result<Endpoint, LinkFError> {
        stream.set_read_timeout(Some(Duration::from_millis(20)))?;
        Ok(Endpoint {
            stream,
            decoder: FrameDecoder::new(linkf::is_known_type as fn(u16) -> bool),
            seq: 1,
        })
    }

    pub fn send<T: serde::Serialize>(&mut self, msg_type: u16, msg: &T) -> Result<(), LinkFError> {
        let class = if msg_type & 0x8000 != 0 {
            FrameClass::Droppable
        } else {
            FrameClass::Required
        };
        let payload = postcard::to_stdvec(msg)?;
        let bytes = encode_frame(PROTO_VERSION, msg_type, self.seq, 0, class, &payload)?;
        self.seq += 1;
        self.stream.write_all(&bytes)?;
        Ok(())
    }

    pub fn pump(&mut self) -> Result<Vec<Frame>, LinkFError> {
        let mut out = Vec::new();
        let mut buf = [0u8; 64 * 1024];
        match self.stream.read(&mut buf) {
            Ok(0) => return Err(LinkFError::Closed),
            Ok(n) => self.decoder.feed(&buf[..n]),
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(e) => return Err(e.into()),
        }
        while let Some(frame) = self.decoder.decode_next()? {
            out.push(frame);
        }
        Ok(out)
    }

    pub fn wait_frames(&mut self) -> Result<Vec<Frame>, LinkFError> {
        loop {
            let frames = self.pump()?;
            if !frames.is_empty() {
                return Ok(frames);
            }
        }
    }

    pub fn hello_as_head(&mut self, auth: Vec<u8>) -> Result<HelloAckF, LinkFError> {
        self.hello(auth, None)
    }

    pub fn hello_as_head_within(&mut self, auth: Vec<u8>, timeout: Duration) -> Result<HelloAckF, LinkFError> {
        self.hello(auth, Some(Instant::now() + timeout))
    }

    fn hello(&mut self, auth: Vec<u8>, deadline: Option<Instant>) -> Result<HelloAckF, LinkFError> {
        self.send(
            msg_type::HELLO,
            &Hello {
                proto_versions: (PROTO_VERSION, PROTO_VERSION),
                link_role: LinkRole::HeadDaemon,
                auth,
            },
        )?;
        loop {
            let frames = self.pump()?;
            if !frames.is_empty() {
                let ack = frames
                    .iter()
                    .find(|f| f.msg_type == msg_type::HELLO_ACK_F)
                    .ok_or(LinkFError::Protocol("expected HelloAckF"))?;
                return Ok(postcard::from_bytes(&ack.payload)?);
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(LinkFError::TimedOut);
            }
        }
    }

    pub fn shutdown(&self) -> Result<(), LinkFError> {
        self.stream.shutdown()?;
        Ok(())
    }

    pub fn answer_hello_as_agent(
        &mut self,
        ack: &HelloAckF,
        expected_auth: &[u8],
    ) -> Result<(), LinkFError> {
        let frames = self.wait_frames()?;
        let hello_frame = frames
            .iter()
            .find(|f| f.msg_type == msg_type::HELLO)
            .ok_or(LinkFError::Protocol("expected Hello"))?;
        let hello: Hello = postcard::from_bytes(&hello_frame.payload)?;
        self.validate_hello(&hello, expected_auth)?;
        self.send(msg_type::HELLO_ACK_F, ack)?;
        Ok(())
    }

    fn validate_hello(&self, hello: &Hello, expected_auth: &[u8]) -> Result<(), LinkFError> {
        negotiate(hello.proto_versions, (PROTO_VERSION, PROTO_VERSION))
            .ok_or(LinkFError::Protocol("no common proto version"))?;
        if hello.link_role != LinkRole::HeadDaemon {
            return Err(LinkFError::Protocol("peer is not a head daemon"));
        }
        if !expected_auth.is_empty() && !ct_eq(&hello.auth, expected_auth) {
            return Err(LinkFError::Unauthorized);
        }
        Ok(())
    }

    pub fn answer_hello_as_agent_within(
        &mut self,
        ack: &HelloAckF,
        timeout: Duration,
        expected_auth: &[u8],
    ) -> Result<(), LinkFError> {
        let deadline = Instant::now() + timeout;
        let hello_frame = loop {
            if let Some(f) = self
                .pump()?
                .into_iter()
                .find(|f| f.msg_type == msg_type::HELLO)
            {
                break f;
            }
            if Instant::now() >= deadline {
                return Err(LinkFError::TimedOut);
            }
        };
        let hello: Hello = postcard::from_bytes(&hello_frame.payload)?;
        self.validate_hello(&hello, expected_auth)?;
        self.send(msg_type::HELLO_ACK_F, ack)?;
        Ok(())
    }
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use superfluid_proto::handshake::CapacitySummary;

    fn ack() -> HelloAckF {
        HelloAckF {
            chosen_version: PROTO_VERSION,
            node_identity: "node".into(),
            capacity: CapacitySummary {
                gpu_memory_bytes: 0,
                gpu_memory_free_bytes: 0,
                host_memory_bytes: 0,
                active_sessions: 0,
                max_sessions: 0,
                max_context_tokens: 0,
                kv_blocks_used: 0,
                kv_blocks_total: 0,
                lanes_active: 0,
                max_lanes: 0,
                queue_depth: 0,
                decode_tokens_per_s: 0,
                prefill_tokens_per_s: 0,
                sampled_at_ms: 0,
            },
            loadable_models: vec!["mock".into()],
            feature_flags: 0,
            limits: linkf::LinkFLimits { max_frame: 1 << 20, transfer_window: 4, max_transfer: 1 << 30, transfer_budget: 1 << 30 },
        }
    }

    #[test]
    fn a_hello_crosses_an_encrypted_endpoint_and_another_secret_does_not() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let node = std::thread::spawn(move || {
            let mut outcomes = Vec::new();
            for _ in 0..2 {
                let (sock, _) = listener.accept().unwrap();
                let ep = Endpoint::from_tcp_secure(sock, b"token", Role::Responder, Duration::from_secs(5));
                outcomes.push(match ep {
                    Ok(mut ep) => ep.answer_hello_as_agent_within(&ack(), Duration::from_secs(5), b"").map(|_| "served"),
                    Err(LinkFError::Unauthorized) => Ok("refused"),
                    Err(e) => Err(e),
                });
            }
            outcomes
        });
        let mut head = Endpoint::connect_tcp_secure(addr, b"token", Duration::from_secs(5)).unwrap();
        assert_eq!(head.hello_as_head(Vec::new()).unwrap().node_identity, "node");
        let other = Endpoint::connect_tcp_secure(addr, b"other", Duration::from_secs(5)).err().map(|e| e.to_string());
        assert!(other.is_some(), "a head with another secret got an endpoint");
        let outcomes = node.join().unwrap();
        assert_eq!(outcomes[0].as_ref().unwrap(), &"served");
        assert_eq!(outcomes[1].as_ref().unwrap(), &"refused");
    }
}
