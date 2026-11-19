//! What this crate asks of the operating system beyond std, once for Unix and once for Windows.
//!
//! Each function says what the caller wants done; how far Windows can do it is said on the
//! function. Nothing here needs more than an ordinary user's rights on either.

use std::fs::File;
use std::path::Path;
use std::process::Command;

pub use superfluid_adapter_kit::install::{canonical, through_link};

/// This machine's name, as it gives it.
pub fn hostname() -> Option<String> {
    #[cfg(unix)]
    {
        let mut buf = [0u8; 256];
        // SAFETY: `buf` is writable for its full length, which is what is passed; gethostname
        // NUL-terminates within it or fails, and `buf` is read only after it succeeded.
        if unsafe { libc::gethostname(buf.as_mut_ptr().cast::<libc::c_char>(), buf.len()) } != 0 {
            return None;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Some(String::from_utf8_lossy(&buf[..end]).into_owned())
    }
    #[cfg(windows)]
    {
        std::env::var("COMPUTERNAME").ok().filter(|n| !n.is_empty())
    }
}

/// Fills `buf` from the system's random source.
pub fn random(buf: &mut [u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Read;
        File::open("/dev/urandom")?.read_exact(buf)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Security::Cryptography::{BCryptGenRandom, BCRYPT_USE_SYSTEM_PREFERRED_RNG};
        // SAFETY: writes `buf.len()` bytes into `buf`; no algorithm handle with the system's
        // preferred generator.
        let status = unsafe { BCryptGenRandom(std::ptr::null_mut(), buf.as_mut_ptr(), buf.len() as u32, BCRYPT_USE_SYSTEM_PREFERRED_RNG) };
        if status == 0 {
            Ok(())
        } else {
            Err(std::io::Error::other(format!("BCryptGenRandom: status {status:#x}")))
        }
    }
}

/// Makes a file these options create readable and writable by its owner alone. On Windows a
/// file takes the rights of the directory it is made in, and the ones this crate writes are
/// under the user's own profile, which only that user reaches.
pub fn owner_only(opts: &mut std::fs::OpenOptions) -> &mut std::fs::OpenOptions {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts
}

/// The file's permission bits; none on Windows, which has no such bits.
pub fn mode(meta: &std::fs::Metadata) -> u32 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o777
    }
    #[cfg(windows)]
    {
        let _ = meta;
        0
    }
}

/// Whether `path` is a file this user may run: one with an execute bit on Unix; on Windows,
/// where what runs is told by its name, one named as a program or a command script.
pub fn is_executable(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else { return false };
    if cfg!(windows) {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
        return meta.is_file() && matches!(ext.as_str(), "exe" | "com" | "cmd" | "bat");
    }
    meta.is_file() && mode(&meta) & 0o111 != 0
}

/// The name a program called `name` has on disk: with `.exe` on Windows.
pub fn exe_name(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

/// What tells two names of one file apart from two files: device and inode on Unix. Windows
/// gives no such pair without opening the file, so there it is a hash of the path with its
/// links resolved, which is the same for a blob and every snapshot link to it.
pub fn file_identity(path: &Path, meta: &std::fs::Metadata) -> (u64, u64) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let _ = path;
        (meta.dev(), meta.ino())
    }
    #[cfg(windows)]
    {
        let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        (0, xxhash_rust::xxh3::xxh3_64(canon.as_os_str().as_encoded_bytes()) ^ meta.len())
    }
}

/// A line that changes when the file does, for remembering what was computed from it.
pub fn file_stamp(meta: &std::fs::Metadata) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        format!("{} {} {} {}.{} {}.{}", meta.dev(), meta.ino(), meta.len(), meta.mtime(), meta.mtime_nsec(), meta.ctime(), meta.ctime_nsec())
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        format!("{} {} {}", meta.len(), meta.last_write_time(), meta.creation_time())
    }
}

/// Takes the exclusive lock on `file`, which goes when the file closes. With `wait` it blocks
/// until the lock is free; without, `false` says another holds it.
pub fn lock_exclusive(file: &File, wait: bool) -> std::io::Result<bool> {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let op = if wait { libc::LOCK_EX } else { libc::LOCK_EX | libc::LOCK_NB };
        loop {
            // SAFETY: flock on a descriptor that is open for the duration of the call.
            if unsafe { libc::flock(file.as_raw_fd(), op) } == 0 {
                return Ok(true);
            }
            let err = std::io::Error::last_os_error();
            match err.kind() {
                std::io::ErrorKind::Interrupted => {}
                std::io::ErrorKind::WouldBlock => return Ok(false),
                _ => return Err(err),
            }
        }
    }
    #[cfg(windows)]
    {
        if wait {
            return file.lock().map(|()| true);
        }
        match file.try_lock() {
            Ok(()) => Ok(true),
            Err(std::fs::TryLockError::WouldBlock) => Ok(false),
            Err(std::fs::TryLockError::Error(e)) => Err(e),
        }
    }
}

/// Makes `link` a link to the directory `target`, a name beside it: a symbolic link on Unix, and
/// on Windows a file that holds the name, which [`through_link`] reads (it says why).
pub fn link_dir(target: &str, link: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(windows)]
    {
        std::fs::write(link, format!("{target}\n"))
    }
}

/// Puts the directory link `tmp` where `link` is, in one rename: a reader finds the old target
/// or the new.
pub fn replace_dir_link(tmp: &Path, link: &Path) -> std::io::Result<()> {
    std::fs::rename(tmp, link)
}

/// Removes the directory link at `link`, not what it points to.
pub fn remove_dir_link(link: &Path) {
    let _ = std::fs::remove_file(link);
}

/// A symbolic link to a file. Windows refuses an ordinary user one, and what makes them here
/// (`superfluid self`) is not offered there.
pub fn symlink_file(target: &Path, link: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(target, link)
    }
}

/// Whether this user may write `path`.
pub fn writable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else { return false };
        // SAFETY: `c` is a NUL-terminated path that outlives the call.
        unsafe { libc::access(c.as_ptr(), libc::W_OK) == 0 }
    }
    #[cfg(windows)]
    {
        std::fs::metadata(path).is_ok_and(|m| !m.permissions().readonly())
    }
}

/// Whether a process with this id is running.
pub fn alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
        // SAFETY: signal 0 checks for the process without sending anything.
        unsafe { libc::kill(pid, 0) == 0 }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_TIMEOUT};
        use windows_sys::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE};
        // SAFETY: a handle opened here is closed here; waiting no time on it only asks whether
        // the process has ended.
        unsafe {
            let h = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid as u32);
            if h.is_null() {
                return false;
            }
            let running = WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
            CloseHandle(h);
            running
        }
    }
}

/// Asks the process to stop: SIGTERM on Unix, which a server answers by finishing what it
/// runs. Windows has no such request for a process without a console, so there it is ended
/// outright, and its workers end when their streams close.
pub fn terminate(pid: i32) {
    #[cfg(unix)]
    // SAFETY: plain POSIX kill; the caller names a process it started or found running.
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    #[cfg(windows)]
    kill_hard(pid);
}

/// Ends the process at once.
pub fn kill_hard(pid: i32) {
    #[cfg(unix)]
    // SAFETY: plain POSIX kill on a child this process spawned.
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE};
        // SAFETY: a handle opened here is closed here.
        unsafe {
            let h = OpenProcess(PROCESS_TERMINATE, 0, pid as u32);
            if !h.is_null() {
                TerminateProcess(h, 1);
                CloseHandle(h);
            }
        }
    }
}

/// Has `cmd` start in a session of its own, so it outlives this process and its terminal.
pub fn detached(cmd: &mut Command) -> &mut Command {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: setsid is async-signal-safe and touches only the child.
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }
    cmd
}

/// Runs `cmd` in this process's place, and returns only when it could not be started. Unix
/// execs it. Windows has no exec: the program runs as a child with this process's streams, and
/// this process waits and exits with its status.
pub fn replace_process(cmd: &mut Command) -> std::io::Error {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.exec()
    }
    #[cfg(windows)]
    {
        match cmd.status() {
            Ok(status) => std::process::exit(status.code().unwrap_or(1)),
            Err(e) => e,
        }
    }
}

/// A second handle on this process's standard error, one a child does not inherit.
pub fn stderr_clone() -> std::io::Result<File> {
    #[cfg(unix)]
    {
        use std::os::fd::FromRawFd;
        // SAFETY: a duplicate of descriptor 2, owned by the File from here on.
        let fd = unsafe { libc::fcntl(libc::STDERR_FILENO, libc::F_DUPFD_CLOEXEC, 3) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: `fd` is open and nothing else owns it.
        Ok(unsafe { File::from_raw_fd(fd) })
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsHandle;
        Ok(File::from(std::io::stderr().as_handle().try_clone_to_owned()?))
    }
}

/// A pipe's read end and write end, neither inherited by a child.
pub fn pipe() -> std::io::Result<(File, File)> {
    #[cfg(unix)]
    {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        let mut fds = [0i32; 2];
        // SAFETY: `fds` has room for the two descriptors; each is owned as soon as it exists.
        unsafe {
            if libc::pipe(fds.as_mut_ptr()) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let (read, write) = (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1]));
            libc::fcntl(read.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
            libc::fcntl(write.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
            Ok((File::from(read), File::from(write)))
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::OwnedHandle;
        let (read, write) = std::io::pipe()?;
        Ok((File::from(OwnedHandle::from(read)), File::from(OwnedHandle::from(write))))
    }
}

/// Makes `to` this process's standard error from here on: descriptor 2, and on Windows the
/// handle Rust asks for at each write and a child started from here inherits. What stood for
/// it before stays open on Windows: something may be midway through a write to it.
pub fn set_stderr(to: &File) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: `to` is open for the call; dup2 closes what 2 was and leaves `to` its own.
        if unsafe { libc::dup2(to.as_raw_fd(), libc::STDERR_FILENO) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::IntoRawHandle;
        use windows_sys::Win32::System::Console::{SetStdHandle, STD_ERROR_HANDLE};
        let named = to.try_clone()?.into_raw_handle();
        // SAFETY: `named` is an open handle this process owns and never closes.
        if unsafe { SetStdHandle(STD_ERROR_HANDLE, named) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        if let Ok(for_c) = to.try_clone() {
            // SAFETY: the descriptor takes over the handle it is opened on; dup2 leaves 2 with
            // a duplicate, and the first descriptor is closed. A library in this process that
            // writes through the C runtime follows.
            unsafe {
                let fd = libc::open_osfhandle(for_c.into_raw_handle() as isize, 0);
                if fd >= 0 {
                    libc::dup2(fd, 2);
                    libc::close(fd);
                }
            }
        }
        Ok(())
    }
}

/// `seconds` since the Unix epoch as the UTC date and time of day: (year, month, day, hour,
/// minute, second), by the proleptic Gregorian calendar.
pub fn utc(seconds: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (seconds / 86_400) as i64;
    let rem = (seconds % 86_400) as u32;
    // Days to a date, counting years from March so the leap day is a year's last
    // (Howard Hinnant's `civil_from_days`).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day, rem / 3_600, rem % 3_600 / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_since_the_epoch_are_a_utc_date_and_time() {
        assert_eq!(utc(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(utc(951_782_400), (2000, 2, 29, 0, 0, 0), "a leap day in a century year that has one");
        assert_eq!(utc(1_709_251_199), (2024, 2, 29, 23, 59, 59));
        assert_eq!(utc(1_709_251_200), (2024, 3, 1, 0, 0, 0));
        assert_eq!(utc(4_102_444_800), (2100, 1, 1, 0, 0, 0));
        assert_eq!(utc(4_107_542_400), (2100, 3, 1, 0, 0, 0), "2100 has no leap day");
    }

    #[test]
    fn the_system_gives_random_bytes_a_name_and_a_lock_one_holder_has() {
        let (mut a, mut b) = ([0u8; 16], [0u8; 16]);
        random(&mut a).unwrap();
        random(&mut b).unwrap();
        assert_ne!(a, b);
        assert!(hostname().is_some_and(|n| !n.is_empty()));
        let dir = std::env::temp_dir().join(format!("superfluid-sys-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lock");
        let open = || std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(&path).unwrap();
        let (first, second) = (open(), open());
        assert!(lock_exclusive(&first, false).unwrap());
        assert!(!lock_exclusive(&second, false).unwrap(), "held by the first");
        drop(first);
        assert!(lock_exclusive(&second, true).unwrap());
        assert!(alive(std::process::id() as i32) && !alive(0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_directory_link_reads_through_and_is_replaced_by_another() {
        let dir = std::env::temp_dir().join(format!("superfluid-sys-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for v in ["v1", "v2"] {
            std::fs::create_dir_all(dir.join(v)).unwrap();
            std::fs::write(dir.join(v).join("is"), v).unwrap();
        }
        let link = dir.join("current");
        link_dir("v1", &link).unwrap();
        assert_eq!(std::fs::read_to_string(through_link(&link).join("is")).unwrap(), "v1");
        link_dir("v2", &dir.join(".new")).unwrap();
        replace_dir_link(&dir.join(".new"), &link).unwrap();
        assert_eq!(std::fs::read_to_string(through_link(&link).join("is")).unwrap(), "v2");
        remove_dir_link(&link);
        assert!(link.symlink_metadata().is_err() && dir.join("v2").join("is").is_file(), "the link went, not its target");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
