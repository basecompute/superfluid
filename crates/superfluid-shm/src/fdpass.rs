//! Passing a shared-memory segment to the process at the other end of a
//! dedicated unix socket.
//!
//! On Unix the descriptor rides the socket as `SCM_RIGHTS`. Windows sockets
//! carry no handles, so there the sender duplicates the handle within itself
//! and writes the duplicate's value with its own process id, and the receiver
//! takes that duplicate out of the sender ([`windows`]).

#[cfg(unix)]
pub use unix::{recv_fd, send_fd};
#[cfg(windows)]
pub use windows::{recv_fd, send_fd};

#[cfg(unix)]
mod unix {
use std::io;
use std::mem;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;

pub fn send_fd(stream: &UnixStream, fd: RawFd, tag: u64) -> io::Result<()> {
    let payload = tag.to_le_bytes();
    let iov = libc::iovec {
        iov_base: payload.as_ptr() as *mut libc::c_void,
        iov_len: payload.len(),
    };
    // SAFETY: CMSG_SPACE is a pure size computation.
    let space = unsafe { libc::CMSG_SPACE(mem::size_of::<RawFd>() as u32) } as usize;
    let mut cmsg_buf = vec![0u8; space];
    // SAFETY: msghdr is a plain C struct; all-zeroes is its valid empty state.
    let mut msg: libc::msghdr = unsafe { mem::zeroed() };
    msg.msg_iov = &iov as *const _ as *mut _;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
    msg.msg_controllen = space as _;

    // SAFETY: msg points at valid iov/control buffers laid out per the
    // cmsg macros; the fd is valid for the duration of the call.
    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(mem::size_of::<RawFd>() as u32) as _;
        std::ptr::copy_nonoverlapping(
            &fd as *const RawFd as *const u8,
            libc::CMSG_DATA(cmsg),
            mem::size_of::<RawFd>(),
        );
        let n = libc::sendmsg(stream.as_raw_fd(), &msg, 0);
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        if n != payload.len() as isize {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "short fd-pass send",
            ));
        }
    }
    Ok(())
}

pub fn recv_fd(stream: &UnixStream) -> io::Result<(OwnedFd, u64)> {
    let mut payload = [0u8; 8];
    let iov = libc::iovec {
        iov_base: payload.as_mut_ptr() as *mut libc::c_void,
        iov_len: payload.len(),
    };
    // SAFETY: CMSG_SPACE is a pure size computation.
    let space = unsafe { libc::CMSG_SPACE(mem::size_of::<RawFd>() as u32) } as usize;
    let mut cmsg_buf = vec![0u8; space];
    // SAFETY: msghdr is a plain C struct; all-zeroes is its valid empty state.
    let mut msg: libc::msghdr = unsafe { mem::zeroed() };
    msg.msg_iov = &iov as *const _ as *mut _;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
    msg.msg_controllen = space as _;

    // SAFETY: msg points at valid buffers; recvmsg fills them, and every
    // descriptor the kernel installed is taken into an OwnedFd so a refused
    // message closes them instead of leaking them.
    let (n, fds) = unsafe {
        let n = libc::recvmsg(stream.as_raw_fd(), &mut msg, 0);
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut fds = Vec::new();
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        if !cmsg.is_null() && (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
            // A truncated message reports the length it was sent with, not the
            // part that arrived, so the count is bounded by the buffer too.
            let start = libc::CMSG_DATA(cmsg) as usize;
            let end = (cmsg as usize + (*cmsg).cmsg_len as usize)
                .min(msg.msg_control as usize + msg.msg_controllen as usize);
            let data = end.saturating_sub(start);
            for i in 0..data / mem::size_of::<RawFd>() {
                let mut fd: RawFd = -1;
                std::ptr::copy_nonoverlapping(
                    libc::CMSG_DATA(cmsg).add(i * mem::size_of::<RawFd>()),
                    &mut fd as *mut RawFd as *mut u8,
                    mem::size_of::<RawFd>(),
                );
                if fd >= 0 {
                    fds.push(OwnedFd::from_raw_fd(fd));
                }
            }
        }
        (n as usize, fds)
    };
    if msg.msg_flags & libc::MSG_CTRUNC != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fd-pass control message truncated (more descriptors than one)",
        ));
    }
    let mut fds = fds.into_iter();
    match (n, fds.next(), fds.next()) {
        (8, Some(fd), None) => Ok((fd, u64::from_le_bytes(payload))),
        (_, None, _) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected SCM_RIGHTS control message",
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "malformed fd-pass message",
        )),
    }
}
}

/// The message is the tag, the sender's process id and the value of a handle
/// in the sender, each eight bytes little-endian. That handle is a duplicate
/// made for this one message: the receiver moves it into itself
/// (`DUPLICATE_CLOSE_SOURCE`), so the sender may close its own handle as soon
/// as the send returns, as it may its descriptor on Unix. A message nobody
/// reads leaves the duplicate open in the sender until it exits.
///
/// The receiver opens the process the message names, so what it trusts is who
/// may write to the socket: the daemon holds the other end of a pair it made,
/// or of a socket in a directory only its user reaches.
#[cfg(windows)]
mod windows {
    use std::io::{self, Read, Write};
    use std::os::windows::io::FromRawHandle;

    use windows_sys::Win32::Foundation::{CloseHandle, DuplicateHandle, DUPLICATE_CLOSE_SOURCE, DUPLICATE_SAME_ACCESS, HANDLE};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentProcessId, OpenProcess, PROCESS_DUP_HANDLE};

    use crate::os::{OwnedFd, RawFd, UnixStream};

    // A handle is a pointer in type only: the system looks it up, and refuses one that is not open.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn send_fd(mut stream: &UnixStream, fd: RawFd, tag: u64) -> io::Result<()> {
        let mut transit: HANDLE = std::ptr::null_mut();
        // SAFETY: duplicates a handle the caller holds open, within this process.
        let ok = unsafe { DuplicateHandle(GetCurrentProcess(), fd, GetCurrentProcess(), &mut transit, 0, 0, DUPLICATE_SAME_ACCESS) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut message = [0u8; 24];
        message[..8].copy_from_slice(&tag.to_le_bytes());
        // SAFETY: takes no arguments and cannot fail.
        message[8..16].copy_from_slice(&u64::from(unsafe { GetCurrentProcessId() }).to_le_bytes());
        message[16..].copy_from_slice(&(transit as usize as u64).to_le_bytes());
        stream.write_all(&message).inspect_err(|_| {
            // SAFETY: the duplicate made above, which nobody was told of.
            unsafe { CloseHandle(transit) };
        })
    }

    pub fn recv_fd(mut stream: &UnixStream) -> io::Result<(OwnedFd, u64)> {
        let mut message = [0u8; 24];
        stream.read_exact(&mut message)?;
        let word = |at: usize| u64::from_le_bytes(message[at..at + 8].try_into().expect("eight bytes"));
        let (tag, pid, transit) = (word(0), word(8), word(16) as usize as HANDLE);
        let pid = u32::try_from(pid).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "fd-pass message names no process"))?;
        // SAFETY: a process handle opened here is closed below; GetCurrentProcess
        // returns a pseudo-handle that needs no closing.
        let (sender, opened) = unsafe {
            if pid == GetCurrentProcessId() {
                (GetCurrentProcess(), false)
            } else {
                let p = OpenProcess(PROCESS_DUP_HANDLE, 0, pid);
                if p.is_null() {
                    return Err(io::Error::last_os_error());
                }
                (p, true)
            }
        };
        let mut mine: HANDLE = std::ptr::null_mut();
        // SAFETY: moves the handle the sender duplicated for this message into
        // this process; a value that names no handle there is an error.
        let ok = unsafe { DuplicateHandle(sender, transit, GetCurrentProcess(), &mut mine, 0, 0, DUPLICATE_SAME_ACCESS | DUPLICATE_CLOSE_SOURCE) };
        let err = io::Error::last_os_error();
        if opened {
            // SAFETY: the process handle opened above.
            unsafe { CloseHandle(sender) };
        }
        if ok == 0 {
            return Err(err);
        }
        // SAFETY: mine is a fresh handle nothing else owns.
        Ok((unsafe { OwnedFd::from_raw_handle(mine) }, tag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::os::UnixStream;
    use crate::segment::ShmSegment;
    #[cfg(unix)]
    use std::{io, mem, os::fd::{AsRawFd, RawFd}};

    #[test]
    fn fd_roundtrip_shares_memory() {
        let (a, b) = UnixStream::pair().unwrap();
        let seg = ShmSegment::create(4096).unwrap();
        seg.write_at(0, b"hello-shm").unwrap();
        send_fd(&a, seg.raw_fd(), 42).unwrap();
        let (fd, tag) = recv_fd(&b).unwrap();
        assert_eq!(tag, 42);
        let seg2 = ShmSegment::from_fd(fd, 4096).unwrap();
        let mut out = [0u8; 9];
        seg2.read_at(0, &mut out).unwrap();
        assert_eq!(&out, b"hello-shm");
        seg2.write_at(100, b"back").unwrap();
        let mut out2 = [0u8; 4];
        seg.read_at(100, &mut out2).unwrap();
        assert_eq!(&out2, b"back");
    }

    #[cfg(unix)]
    fn send_fds(stream: &UnixStream, fds: &[RawFd]) {
        let payload = 7u64.to_le_bytes();
        let iov = libc::iovec { iov_base: payload.as_ptr() as *mut libc::c_void, iov_len: payload.len() };
        let bytes = mem::size_of_val(fds) as u32;
        // SAFETY: as in send_fd, with room for every descriptor.
        unsafe {
            let space = libc::CMSG_SPACE(bytes) as usize;
            let mut cmsg_buf = vec![0u8; space];
            let mut msg: libc::msghdr = mem::zeroed();
            msg.msg_iov = &iov as *const _ as *mut _;
            msg.msg_iovlen = 1;
            msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
            msg.msg_controllen = space as _;
            let cmsg = libc::CMSG_FIRSTHDR(&msg);
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len = libc::CMSG_LEN(bytes) as _;
            std::ptr::copy_nonoverlapping(fds.as_ptr() as *const u8, libc::CMSG_DATA(cmsg), bytes as usize);
            assert_eq!(libc::sendmsg(stream.as_raw_fd(), &msg, 0), 8);
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_message_carrying_more_than_one_descriptor_is_refused() {
        let seg = ShmSegment::create(4096).unwrap();
        for count in [2, 3, 8] {
            let (a, b) = UnixStream::pair().unwrap();
            send_fds(&a, &vec![seg.raw_fd(); count]);
            let err = recv_fd(&b).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidData, "{count} descriptors: {err}");
        }
        let (a, b) = UnixStream::pair().unwrap();
        send_fds(&a, &[seg.raw_fd()]);
        assert_eq!(recv_fd(&b).unwrap().1, 7);
    }

    #[test]
    fn a_segment_shorter_than_announced_is_refused_before_it_is_mapped() {
        let (a, b) = UnixStream::pair().unwrap();
        let seg = ShmSegment::create(4096).unwrap();
        send_fd(&a, seg.raw_fd(), 1).unwrap();
        let (fd, _) = recv_fd(&b).unwrap();
        let err = ShmSegment::from_fd(fd, 1 << 20).unwrap_err();
        assert!(matches!(err, crate::ShmError::Short { announced: 1048576, .. }), "{err}");
    }

    #[test]
    fn the_sender_may_close_its_segment_once_it_is_sent() {
        let (a, b) = UnixStream::pair().unwrap();
        let seg = ShmSegment::create(4096).unwrap();
        seg.write_at(0, b"kept").unwrap();
        send_fd(&a, seg.raw_fd(), 3).unwrap();
        drop(seg);
        let (fd, _) = recv_fd(&b).unwrap();
        let mut out = [0u8; 4];
        ShmSegment::from_fd(fd, 4096).unwrap().read_at(0, &mut out).unwrap();
        assert_eq!(&out, b"kept");
    }
}
