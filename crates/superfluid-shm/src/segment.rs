//! Anonymous shared-memory segments.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::AtomicU64;

use crate::ShmError;

#[derive(Debug)]
pub struct ShmSegment {
    fd: OwnedFd,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the mapping is process-shared memory; concurrent access rules
// are enforced by the ring protocol above (single writer, generation
// validation), not by &mut aliasing.
unsafe impl Send for ShmSegment {}

impl ShmSegment {
    pub fn create(len: usize) -> Result<ShmSegment, ShmError> {
        #[cfg(target_os = "linux")]
        let fd: OwnedFd = {
            // SAFETY: memfd_create with a static name; returns -1 on error.
            let raw = unsafe {
                libc::syscall(
                    libc::SYS_memfd_create,
                    c"superfluid-shm".as_ptr(),
                    libc::MFD_CLOEXEC,
                ) as libc::c_int
            };
            if raw < 0 {
                return Err(ShmError::Io(std::io::Error::last_os_error()));
            }
            // SAFETY: raw is a freshly created, owned fd.
            unsafe { OwnedFd::from_raw_fd(raw) }
        };
        #[cfg(not(target_os = "linux"))]
        let fd: OwnedFd = {
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let name = format!(
                "/superfluid-{}-{:x}\0",
                std::process::id(),
                COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            );
            // SAFETY: name is NUL-terminated; O_EXCL guards collisions.
            let raw = unsafe {
                libc::shm_open(
                    name.as_ptr().cast::<libc::c_char>(),
                    libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                    0o600,
                )
            };
            if raw < 0 {
                return Err(ShmError::Io(std::io::Error::last_os_error()));
            }
            // SAFETY: same NUL-terminated name; unlink only removes the
            // name, the open fd stays valid.
            unsafe {
                libc::shm_unlink(name.as_ptr().cast::<libc::c_char>());
            }
            // SAFETY: raw is a freshly created, owned fd.
            unsafe { OwnedFd::from_raw_fd(raw) }
        };

        // SAFETY: fd is a valid shm fd we own.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } != 0 {
            return Err(ShmError::Io(std::io::Error::last_os_error()));
        }
        Self::map(fd, len)
    }

    pub fn from_fd(fd: OwnedFd, len: usize) -> Result<ShmSegment, ShmError> {
        // SAFETY: all-zero is a valid stat; fstat fills it from an fd we own.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 {
            return Err(ShmError::Io(std::io::Error::last_os_error()));
        }
        // Mapping past the object's end succeeds and then faults on first
        // touch, so an announced length the segment cannot back is refused here.
        let actual = st.st_size.max(0) as u64;
        if actual < len as u64 {
            return Err(ShmError::Short { announced: len, actual });
        }
        Self::map(fd, len)
    }

    fn map(fd: OwnedFd, len: usize) -> Result<ShmSegment, ShmError> {
        // SAFETY: fd is a valid mappable shm fd sized >= len (create
        // ftruncated it; from_fd checked it).
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(ShmError::Io(std::io::Error::last_os_error()));
        }
        Ok(ShmSegment {
            fd,
            ptr: ptr as *mut u8,
            len,
        })
    }

    pub fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn read_at(&self, offset: usize, out: &mut [u8]) -> Result<(), ShmError> {
        if offset
            .checked_add(out.len())
            .is_none_or(|end| end > self.len)
        {
            return Err(ShmError::TooSmall);
        }
        // SAFETY: bounds checked above; mapping is live for &self.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.add(offset), out.as_mut_ptr(), out.len());
        }
        Ok(())
    }

    pub fn write_at(&self, offset: usize, data: &[u8]) -> Result<(), ShmError> {
        if offset
            .checked_add(data.len())
            .is_none_or(|end| end > self.len)
        {
            return Err(ShmError::TooSmall);
        }
        // SAFETY: bounds checked above; single-writer discipline is the
        // caller's (ring/buffer protocol).
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len());
        }
        Ok(())
    }

    pub(crate) fn atomic_u64_at(&self, offset: usize) -> Result<&AtomicU64, ShmError> {
        if offset.checked_add(8).is_none_or(|end| end > self.len) {
            return Err(ShmError::TooSmall);
        }
        if !(self.ptr as usize + offset).is_multiple_of(8) {
            return Err(ShmError::Misaligned(offset));
        }
        // SAFETY: in bounds and 8-aligned per the checks above, and the
        // mapping lives as long as &self.
        Ok(unsafe { AtomicU64::from_ptr(self.ptr.add(offset) as *mut u64) })
    }
}

impl Drop for ShmSegment {
    fn drop(&mut self) {
        // SAFETY: ptr/len came from a successful mmap.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}
