//! The two things Link W needs from the operating system, under one name on each.
//!
//! A local stream socket: `AF_UNIX` on both, from std on Unix and from
//! `uds_windows` on Windows, which has had the address family since 1803.
//!
//! A handle to a shared-memory segment that another process can be given: a file
//! descriptor on Unix, a section handle on Windows. The names are the Unix ones,
//! since that is what the rest of the code says.

#[cfg(unix)]
pub use std::os::fd::{OwnedFd, RawFd};
#[cfg(unix)]
pub use std::os::unix::net::{UnixListener, UnixStream};

#[cfg(windows)]
pub use uds_windows::{UnixListener, UnixStream};
/// A section handle this process owns.
#[cfg(windows)]
pub type OwnedFd = std::os::windows::io::OwnedHandle;
/// A section handle, borrowed.
#[cfg(windows)]
pub type RawFd = std::os::windows::io::RawHandle;
