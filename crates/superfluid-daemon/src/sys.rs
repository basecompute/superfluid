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

/// Opens `path` with `opts`, and has a file this creates readable and writable by its owner
/// alone: mode 0600 on Unix. A Windows file takes the rights of the directory it is made in,
/// which under the user's profile are already the user's, but a home elsewhere may give every
/// user of the machine a reading; so there the file is given a list of its own, with this user,
/// the system and its administrators (who reach any file, as root does) and nobody else. Where
/// the volume keeps no such list the file is as its directory has it, as a mode is on a
/// filesystem without modes.
pub fn open_private(opts: &mut std::fs::OpenOptions, path: &Path) -> std::io::Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let file = opts.open(path)?;
    #[cfg(windows)]
    let _ = private::restrict(path);
    Ok(file)
}

#[cfg(windows)]
mod private {
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;

    use windows_sys::Win32::Foundation::{CloseHandle, LocalFree, HANDLE};
    use windows_sys::Win32::Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, SetNamedSecurityInfoW, SDDL_REVISION_1, SE_FILE_OBJECT,
    };
    use windows_sys::Win32::Security::{
        GetSecurityDescriptorDacl, GetTokenInformation, TokenUser, ACL, DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION,
        PSECURITY_DESCRIPTOR, TOKEN_QUERY, TOKEN_USER,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    /// This process's user, as the text form of its security identifier (`S-1-5-21-...`).
    fn user_sid() -> std::io::Result<String> {
        // SAFETY: the token is closed here; the buffer is 8-aligned and as long as it is said
        // to be, and TOKEN_USER is read from it only after the call filled it; the string the
        // system allocates is copied and freed.
        unsafe {
            let mut token: HANDLE = std::ptr::null_mut();
            if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
                return Err(std::io::Error::last_os_error());
            }
            let mut buf = [0u64; 32];
            let mut len = 0u32;
            let got = GetTokenInformation(token, TokenUser, buf.as_mut_ptr().cast(), std::mem::size_of_val(&buf) as u32, &mut len);
            let err = std::io::Error::last_os_error();
            CloseHandle(token);
            if got == 0 {
                return Err(err);
            }
            let user = &*buf.as_ptr().cast::<TOKEN_USER>();
            let mut text: *mut u16 = std::ptr::null_mut();
            if ConvertSidToStringSidW(user.User.Sid, &mut text) == 0 {
                return Err(std::io::Error::last_os_error());
            }
            let n = (0..).take_while(|&i| *text.add(i) != 0).count();
            let sid = String::from_utf16_lossy(std::slice::from_raw_parts(text, n));
            LocalFree(text.cast());
            Ok(sid)
        }
    }

    /// Gives `path` an access list of its own that inherits nothing: full access for this user,
    /// the system (SY) and the administrators (BA).
    pub fn restrict(path: &Path) -> std::io::Result<()> {
        let wide = |s: &std::ffi::OsStr| s.encode_wide().chain(std::iter::once(0)).collect::<Vec<u16>>();
        let sddl = wide(format!("D:P(A;;FA;;;{})(A;;FA;;;SY)(A;;FA;;;BA)", user_sid()?).as_ref());
        let name = wide(path.as_os_str());
        // SAFETY: both strings are NUL-terminated and outlive the calls; the descriptor the
        // system allocates holds the list it points into, and is freed after the list is set.
        unsafe {
            let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
            if ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.as_ptr(), SDDL_REVISION_1, &mut descriptor, std::ptr::null_mut()) == 0 {
                return Err(std::io::Error::last_os_error());
            }
            let (mut present, mut defaulted) = (0i32, 0i32);
            let mut list: *mut ACL = std::ptr::null_mut();
            let status = if GetSecurityDescriptorDacl(descriptor, &mut present, &mut list, &mut defaulted) == 0 || present == 0 {
                Err(std::io::Error::last_os_error())
            } else {
                match SetNamedSecurityInfoW(
                    name.as_ptr(),
                    SE_FILE_OBJECT,
                    DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    list,
                    std::ptr::null(),
                ) {
                    0 => Ok(()),
                    code => Err(std::io::Error::from_raw_os_error(code as i32)),
                }
            };
            LocalFree(descriptor);
            status
        }
    }
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

/// Asks the server with this process id to drain: stop accepting, finish what it runs, exit.
/// SIGUSR1 on Unix. Windows has no signal to send, so a server there waits on an event named
/// for its process id ([`DrainRequest`]), which this sets; a process that made no such event
/// is not a server that drains, and is left alone.
pub fn ask_to_drain(pid: i32) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        // SAFETY: plain POSIX kill; the caller names a server it started or was given.
        if unsafe { libc::kill(pid, libc::SIGUSR1) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenEventW, SetEvent, EVENT_MODIFY_STATE};
        for name in drain_event_names(pid as u32) {
            // SAFETY: `name` is NUL-terminated and outlives the call; a handle opened here is
            // closed here.
            unsafe {
                let h = OpenEventW(EVENT_MODIFY_STATE, 0, name.as_ptr());
                if h.is_null() {
                    continue;
                }
                let set = SetEvent(h) != 0;
                let err = std::io::Error::last_os_error();
                CloseHandle(h);
                return if set { Ok(()) } else { Err(err) };
            }
        }
        Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no superfluid server that drains has that process id"))
    }
}

/// The names a server's drain event may have, as wide strings: one every session of this user
/// finds (a server started over ssh, asked from the desktop), then one this session alone has,
/// for where the first may not be made.
#[cfg(windows)]
fn drain_event_names(pid: u32) -> [Vec<u16>; 2] {
    let wide = |space: &str| format!("{space}\\superfluid-drain-{pid}\0").encode_utf16().collect::<Vec<u16>>();
    [wide("Global"), wide("Local")]
}

/// A Windows server's standing request to be asked to drain: an event named for this process,
/// which [`ask_to_drain`] sets from another.
#[cfg(windows)]
pub struct DrainRequest(windows_sys::Win32::Foundation::HANDLE);

// SAFETY: an event handle is used from any thread.
#[cfg(windows)]
unsafe impl Send for DrainRequest {}

#[cfg(windows)]
impl DrainRequest {
    /// Makes the event. It lasts as long as this process.
    pub fn listen() -> std::io::Result<DrainRequest> {
        Self::listen_as(std::process::id())
    }

    fn listen_as(pid: u32) -> std::io::Result<DrainRequest> {
        use windows_sys::Win32::System::Threading::CreateEventW;
        let mut last = std::io::Error::other("no event name");
        for name in drain_event_names(pid) {
            // SAFETY: `name` is NUL-terminated and outlives the call; default security gives the
            // event to this user. Manual reset: once asked, it stays asked.
            let h = unsafe { CreateEventW(std::ptr::null(), 1, 0, name.as_ptr()) };
            if !h.is_null() {
                return Ok(DrainRequest(h));
            }
            last = std::io::Error::last_os_error();
        }
        Err(last)
    }

    /// Returns once another process has asked.
    pub fn wait(&self) {
        use windows_sys::Win32::System::Threading::{WaitForSingleObject, INFINITE};
        // SAFETY: the handle is this value's, open for as long as it lives.
        unsafe {
            WaitForSingleObject(self.0, INFINITE);
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
/// this process waits and exits with its status. While it waits it stands for the program:
/// Ctrl+C and Ctrl+Break, which reach every process on the console, are the program's to
/// answer, and ending this process ends the program. What the program itself starts is left
/// out of that, as it would be after an exec.
pub fn replace_process(cmd: &mut Command) -> std::io::Error {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.exec()
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{BOOL, HANDLE};
        use windows_sys::Win32::System::Console::{SetConsoleCtrlHandler, CTRL_BREAK_EVENT, CTRL_C_EVENT};
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK,
        };

        unsafe extern "system" fn left_to_the_program(kind: u32) -> BOOL {
            BOOL::from(kind == CTRL_C_EVENT || kind == CTRL_BREAK_EVENT)
        }
        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(e) => return e,
        };
        // SAFETY: the handler is a function that lives as long as the process and touches
        // nothing. The job is this process's own and stays open until it exits, which is what
        // ends the program; `limits` is plain data, zeroed and then set, and its size is what
        // is passed; the child's handle is open while `child` lives.
        unsafe {
            SetConsoleCtrlHandler(Some(left_to_the_program), 1);
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if !job.is_null() {
                let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK;
                let size = std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
                if SetInformationJobObject(job, JobObjectExtendedLimitInformation, std::ptr::addr_of!(limits).cast(), size) != 0 {
                    AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE);
                }
            }
        }
        match child.wait() {
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

    // Under a process id no process has: this test binary may hold servers of its own, which
    // an event named for it would drain.
    #[cfg(windows)]
    #[test]
    fn a_server_asked_to_drain_hears_it_and_a_process_that_never_listened_is_left_alone() {
        let nobody = 0x7fff_fff0 - (std::process::id() % 1024);
        assert_eq!(ask_to_drain(nobody as i32).unwrap_err().kind(), std::io::ErrorKind::NotFound, "nothing listens yet");
        let request = DrainRequest::listen_as(nobody).unwrap();
        let (heard, hears) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            request.wait();
            let _ = heard.send(());
        });
        assert!(hears.recv_timeout(std::time::Duration::from_millis(100)).is_err(), "not asked yet");
        ask_to_drain(nobody as i32).unwrap();
        hears.recv_timeout(std::time::Duration::from_secs(10)).expect("the request was heard");
    }

    #[test]
    fn a_private_file_is_its_owners_and_takes_no_rights_from_its_directory() {
        let dir = std::env::temp_dir().join(format!("superfluid-sys-private-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("secret");
        let file = open_private(std::fs::OpenOptions::new().write(true).create_new(true), &path).unwrap();
        #[cfg(unix)]
        assert_eq!(mode(&file.metadata().unwrap()), 0o600);
        #[cfg(windows)]
        {
            let listed = Command::new("icacls").arg(&path).output().unwrap();
            let listed = String::from_utf8_lossy(&listed.stdout);
            assert!(!listed.contains("(I)"), "nothing is inherited: {listed}");
            assert_eq!(listed.matches("(F)").count(), 3, "this user, the system and the administrators: {listed}");
        }
        drop(file);
        assert!(std::fs::read(&path).is_ok(), "its owner still reads it");
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
