//! The two things Link W needs from the operating system, under one name on each.
//!
//! A local stream socket: `AF_UNIX` on both, from std on Unix and from
//! `uds_windows` on Windows, which has had the address family since 1803. The
//! Windows one is wrapped, for its read timeout (see [`UnixStream`] there).
//!
//! A handle to a shared-memory segment that another process can be given: a file
//! descriptor on Unix, a section handle on Windows. The names are the Unix ones,
//! since that is what the rest of the code says.

#[cfg(unix)]
pub use std::os::fd::{OwnedFd, RawFd};
#[cfg(unix)]
pub use std::os::unix::net::{UnixListener, UnixStream};

#[cfg(windows)]
pub use windows::{wait_readable, UnixListener, UnixStream};
/// A section handle this process owns.
#[cfg(windows)]
pub type OwnedFd = std::os::windows::io::OwnedHandle;
/// A section handle, borrowed.
#[cfg(windows)]
pub type RawFd = std::os::windows::io::RawHandle;

#[cfg(windows)]
mod windows {
    use std::io::{self, Read, Write};
    use std::net::Shutdown;
    use std::os::windows::io::{AsRawSocket, RawSocket};
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use windows_sys::Win32::Networking::WinSock::{WSAPoll, POLLRDNORM, WSAPOLLFD};

    /// Whether `socket` has something to read within `timeout`: bytes, its end, or an
    /// error, which the read that follows reports. A wait under a millisecond is one.
    pub fn wait_readable(socket: RawSocket, timeout: Duration) -> io::Result<bool> {
        let mut fd = WSAPOLLFD { fd: socket as usize, events: POLLRDNORM, revents: 0 };
        let ms = timeout.as_millis().clamp(1, i32::MAX as u128) as i32;
        // SAFETY: one WSAPOLLFD, as the count says, alive for the call.
        match unsafe { WSAPoll(&mut fd, 1, ms) } {
            0 => Ok(false),
            n if n > 0 => Ok(true),
            _ => Err(io::Error::last_os_error()),
        }
    }

    /// `uds_windows`' stream with a read timeout that is safe to hit.
    ///
    /// Winsock's own (`SO_RCVTIMEO`) cancels the read when the time is up, and a read
    /// cancelled as bytes arrive fails with `WSAEINTR` or `ERROR_IO_PENDING` and leaves
    /// the socket in a state Microsoft calls indeterminate: a link polled every 5 ms
    /// met it within a minute. So the timeout is kept here, and a read waits for the
    /// socket to be readable for that long before it reads, with nothing to cancel.
    /// A clone shares it, as a duplicated descriptor shares the option on Unix.
    #[derive(Debug)]
    pub struct UnixStream {
        inner: uds_windows::UnixStream,
        /// Microseconds; zero for none.
        read_timeout: Arc<AtomicU64>,
    }

    impl UnixStream {
        fn over(inner: uds_windows::UnixStream) -> UnixStream {
            UnixStream { inner, read_timeout: Arc::new(AtomicU64::new(0)) }
        }

        pub fn connect<P: AsRef<Path>>(path: P) -> io::Result<UnixStream> {
            uds_windows::UnixStream::connect(path).map(UnixStream::over)
        }

        pub fn pair() -> io::Result<(UnixStream, UnixStream)> {
            let (a, b) = uds_windows::UnixStream::pair()?;
            Ok((UnixStream::over(a), UnixStream::over(b)))
        }

        pub fn try_clone(&self) -> io::Result<UnixStream> {
            Ok(UnixStream { inner: self.inner.try_clone()?, read_timeout: Arc::clone(&self.read_timeout) })
        }

        pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
            let micros = match timeout {
                None => 0,
                Some(d) if d.is_zero() => return Err(io::Error::new(io::ErrorKind::InvalidInput, "cannot set a zero duration timeout")),
                Some(d) => d.as_micros().clamp(1, u64::MAX as u128) as u64,
            };
            self.read_timeout.store(micros, Ordering::Relaxed);
            Ok(())
        }

        pub fn read_timeout(&self) -> io::Result<Option<Duration>> {
            Ok(match self.read_timeout.load(Ordering::Relaxed) {
                0 => None,
                micros => Some(Duration::from_micros(micros)),
            })
        }

        pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
            self.inner.set_nonblocking(nonblocking)
        }

        pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
            self.inner.shutdown(how)
        }

        fn read_within(&self, buf: &mut [u8]) -> io::Result<usize> {
            let micros = self.read_timeout.load(Ordering::Relaxed);
            if micros != 0 && !wait_readable(self.inner.as_raw_socket(), Duration::from_micros(micros))? {
                return Err(io::ErrorKind::TimedOut.into());
            }
            (&self.inner).read(buf)
        }
    }

    impl Read for UnixStream {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.read_within(buf)
        }
    }

    impl Read for &UnixStream {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.read_within(buf)
        }
    }

    impl Write for UnixStream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            (&self.inner).write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            (&self.inner).flush()
        }
    }

    impl Write for &UnixStream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            (&self.inner).write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            (&self.inner).flush()
        }
    }

    impl AsRawSocket for UnixStream {
        fn as_raw_socket(&self) -> RawSocket {
            self.inner.as_raw_socket()
        }
    }

    #[derive(Debug)]
    pub struct UnixListener(uds_windows::UnixListener);

    impl UnixListener {
        pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<UnixListener> {
            uds_windows::UnixListener::bind(path).map(UnixListener)
        }

        pub fn accept(&self) -> io::Result<(UnixStream, uds_windows::SocketAddr)> {
            let (stream, addr) = self.0.accept()?;
            Ok((UnixStream::over(stream), addr))
        }

        pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
            self.0.set_nonblocking(nonblocking)
        }

        /// Each connection as it comes, as std's does: it never ends.
        pub fn incoming(&self) -> impl Iterator<Item = io::Result<UnixStream>> + '_ {
            std::iter::repeat_with(move || self.accept().map(|(stream, _)| stream))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::Instant;

        #[test]
        fn a_read_that_times_out_says_so_and_the_next_one_reads() {
            let (mut a, mut b) = UnixStream::pair().unwrap();
            b.set_read_timeout(Some(Duration::from_millis(20))).unwrap();
            assert_eq!(b.read_timeout().unwrap(), Some(Duration::from_millis(20)));
            let started = Instant::now();
            let quiet = b.read(&mut [0u8; 4]).unwrap_err();
            assert_eq!(quiet.kind(), io::ErrorKind::TimedOut);
            assert!(started.elapsed() >= Duration::from_millis(15), "it waited: {:?}", started.elapsed());
            a.write_all(b"ab").unwrap();
            let mut buf = [0u8; 4];
            assert_eq!(b.read(&mut buf).unwrap(), 2);
            assert_eq!(&buf[..2], b"ab");
            let clone = b.try_clone().unwrap();
            assert_eq!(clone.read_timeout().unwrap(), Some(Duration::from_millis(20)), "a clone shares the timeout");
            drop(a);
            assert_eq!(b.read(&mut buf).unwrap(), 0, "the end is read at once, not waited out");
            assert!(b.set_read_timeout(Some(Duration::ZERO)).is_err());
        }

        /// What `SO_RCVTIMEO` did within a minute: a writer at random moments around the
        /// reader's timeout, every byte checked.
        #[test]
        fn timed_reads_lose_nothing_while_bytes_arrive_at_any_moment() {
            let (mut a, mut b) = UnixStream::pair().unwrap();
            b.set_read_timeout(Some(Duration::from_millis(1))).unwrap();
            let reader = std::thread::spawn(move || {
                let (mut next, mut buf) = (0u64, [0u8; 256]);
                loop {
                    match b.read(&mut buf) {
                        Ok(0) => return Ok(next),
                        Ok(n) => {
                            for &byte in &buf[..n] {
                                if byte != (next % 251) as u8 {
                                    return Err(format!("wrong byte at {next}"));
                                }
                                next += 1;
                            }
                        }
                        Err(e) if e.kind() == io::ErrorKind::TimedOut => {}
                        Err(e) => return Err(format!("read failed at {next}: {e}")),
                    }
                }
            });
            let (started, mut sent, mut x) = (Instant::now(), 0u64, 0x9E37_79B9_7F4A_7C15u64);
            while started.elapsed() < Duration::from_secs(3) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let piece: Vec<u8> = (0..1 + x % 48).map(|i| ((sent + i) % 251) as u8).collect();
                a.write_all(&piece).unwrap();
                sent += piece.len() as u64;
                let (wait, t) = (Duration::from_micros((x >> 8) % 2001), Instant::now());
                while t.elapsed() < wait {
                    std::hint::spin_loop();
                }
            }
            drop(a);
            assert_eq!(reader.join().unwrap(), Ok(sent));
        }
    }
}
