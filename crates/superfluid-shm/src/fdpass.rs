//! `SCM_RIGHTS` fd passing over a dedicated unix socket.

use std::io;
use std::mem;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;

use std::os::fd::AsRawFd;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::ShmSegment;

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
}
