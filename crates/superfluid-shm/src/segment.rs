//! Anonymous shared-memory segments.

use std::sync::atomic::AtomicU64;

use crate::os::{OwnedFd, RawFd};
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

#[cfg(unix)]
impl ShmSegment {
    pub fn create(len: usize) -> Result<ShmSegment, ShmError> {
        use std::os::fd::FromRawFd;
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
        if unsafe { libc::ftruncate(std::os::fd::AsRawFd::as_raw_fd(&fd), len as libc::off_t) } != 0 {
            return Err(ShmError::Io(std::io::Error::last_os_error()));
        }
        Self::map(fd, len)
    }

    pub fn from_fd(fd: OwnedFd, len: usize) -> Result<ShmSegment, ShmError> {
        // SAFETY: all-zero is a valid stat; fstat fills it from an fd we own.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        if unsafe { libc::fstat(std::os::fd::AsRawFd::as_raw_fd(&fd), &mut st) } != 0 {
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
                std::os::fd::AsRawFd::as_raw_fd(&fd),
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
        std::os::fd::AsRawFd::as_raw_fd(&self.fd)
    }

    fn unmap(&mut self) {
        // SAFETY: ptr/len came from a successful mmap.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

#[cfg(windows)]
impl ShmSegment {
    /// A section backed by the paging file, with no name: another process
    /// reaches it only through a handle it is given.
    pub fn create(len: usize) -> Result<ShmSegment, ShmError> {
        use std::os::windows::io::FromRawHandle;
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, PAGE_READWRITE};

        let size = len as u64;
        // SAFETY: no file, no security attributes and no name; returns null on error.
        let raw = unsafe {
            CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, (size >> 32) as u32, size as u32, std::ptr::null())
        };
        if raw.is_null() {
            return Err(ShmError::Io(std::io::Error::last_os_error()));
        }
        // SAFETY: raw is a freshly created, owned handle.
        let fd = unsafe { OwnedFd::from_raw_handle(raw) };
        Self::map(fd, len)
    }

    pub fn from_fd(fd: OwnedFd, len: usize) -> Result<ShmSegment, ShmError> {
        Self::map(fd, len)
    }

    /// Maps the whole section and refuses one shorter than `len`: a view asked
    /// to reach past the section's end is refused by the system as access
    /// denied, which says nothing of why.
    fn map(fd: OwnedFd, len: usize) -> Result<ShmSegment, ShmError> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::Memory::{MapViewOfFile, UnmapViewOfFile, VirtualQuery, FILE_MAP_ALL_ACCESS, MEMORY_BASIC_INFORMATION};

        // SAFETY: fd is a handle this process owns; a handle that is not a
        // section, or not one open for reading and writing, is refused.
        let view = unsafe { MapViewOfFile(fd.as_raw_handle(), FILE_MAP_ALL_ACCESS, 0, 0, 0) };
        if view.Value.is_null() {
            return Err(ShmError::Io(std::io::Error::last_os_error()));
        }
        // SAFETY: all-zero is a valid MEMORY_BASIC_INFORMATION, and VirtualQuery
        // fills it for the view just mapped.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        let got = unsafe { VirtualQuery(view.Value, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
        if got == 0 || info.RegionSize < len {
            let err = if got == 0 {
                ShmError::Io(std::io::Error::last_os_error())
            } else {
                ShmError::Short { announced: len, actual: info.RegionSize as u64 }
            };
            // SAFETY: view came from the MapViewOfFile above.
            unsafe { UnmapViewOfFile(view) };
            return Err(err);
        }
        Ok(ShmSegment { fd, ptr: view.Value as *mut u8, len })
    }

    pub fn raw_fd(&self) -> RawFd {
        std::os::windows::io::AsRawHandle::as_raw_handle(&self.fd)
    }

    fn unmap(&mut self) {
        use windows_sys::Win32::System::Memory::{UnmapViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS};
        // SAFETY: ptr came from a successful MapViewOfFile.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.ptr.cast() });
        }
    }
}

impl ShmSegment {
    /// A second handle to the segment, as `dup` gives one.
    pub fn dup_fd(&self) -> std::io::Result<OwnedFd> {
        self.fd.try_clone()
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
        self.unmap();
    }
}
