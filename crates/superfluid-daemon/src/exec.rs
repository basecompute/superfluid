//! Starting another program.

use std::process::{Command, Output, Stdio};
use std::time::Duration;

/// How long a worker gets to answer a question about itself (`check`, `compat`, `versions`): a minute, as one that
/// probes a GPU may take a while, and one that hangs must not hold the command that asked it.
pub(crate) const ANSWER_WITHIN: Duration = Duration::from_secs(60);

pub(crate) fn when_not_busy<T>(mut start: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    let mut tries = 0;
    loop {
        match start() {
            Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy && tries < 100 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            other => return other,
        }
    }
}

/// A program's output, or `None` when it gave none within `limit`: it is killed then, so a worker that hangs stops
/// there and not with the command that asked it. It starts as `when_not_busy` starts it, with stdin closed.
pub(crate) fn output_within(cmd: &mut Command, limit: Duration) -> std::io::Result<Option<Output>> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let child = when_not_busy(|| cmd.spawn())?;
    let pid = child.id() as i32;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });
    match rx.recv_timeout(limit) {
        Ok(out) => out.map(Some),
        Err(_) => {
            kill_hard(pid);
            Ok(None)
        }
    }
}

/// The last lines a program wrote to stderr, as an error quotes them (`: a / b / c`); nothing when it wrote none.
/// Its failure is in its own words there, and the command that ran it says no more than the exit status.
pub(crate) fn stderr_tail(stderr: &[u8]) -> String {
    let stderr = String::from_utf8_lossy(stderr);
    let tail: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).rev().take(3).collect();
    if tail.is_empty() {
        return String::new();
    }
    format!(": {}", tail.into_iter().rev().collect::<Vec<_>>().join(" / "))
}

pub(crate) fn kill_hard(pid: i32) {
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    // SAFETY: plain POSIX kill on a child this process spawned.
    unsafe {
        kill(pid, 9);
    }
}

/// A runtime's programs come from wherever its project publishes them; they get this
/// environment without the credentials in it. A pull is the exception, and does not use this:
/// it fetches with the operator's tokens.
pub(crate) fn without_secrets(cmd: &mut std::process::Command) -> &mut std::process::Command {
    for (name, _) in std::env::vars_os() {
        if is_secret(&name) {
            cmd.env_remove(name);
        }
    }
    cmd
}

fn is_secret(name: &std::ffi::OsStr) -> bool {
    let name = name.to_string_lossy().to_ascii_uppercase();
    ["TOKEN", "SECRET", "SECRET_KEY", "PASSWORD", "PASSWD", "API_KEY", "APIKEY", "ACCESS_KEY", "PRIVATE_KEY", "CREDENTIALS"]
        .iter()
        .any(|end| name.ends_with(end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error, ErrorKind};
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn credentials_are_known_by_their_names() {
        for secret in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY", "hf_token", "PGPASSWORD"] {
            assert!(is_secret(std::ffi::OsStr::new(secret)), "{secret}");
        }
        for kept in ["PATH", "HOME", "HF_HOME", "HF_TOKEN_PATH", "SUPERFLUID_TEST_HF_TOKENIZER", "CUDA_VISIBLE_DEVICES", "AWS_ACCESS_KEY_ID"] {
            assert!(!is_secret(std::ffi::OsStr::new(kept)), "{kept}");
        }
    }

    #[test]
    fn a_busy_file_is_tried_again_and_any_other_failure_is_not() {
        let mut calls = 0;
        let started = when_not_busy(|| {
            calls += 1;
            if calls < 3 {
                Err(Error::from(ErrorKind::ExecutableFileBusy))
            } else {
                Ok(calls)
            }
        });
        assert_eq!(started.unwrap(), 3, "busy twice, then started");
        let mut calls = 0;
        let missing = when_not_busy(|| -> std::io::Result<()> {
            calls += 1;
            Err(Error::from(ErrorKind::NotFound))
        });
        assert_eq!((missing.unwrap_err().kind(), calls), (ErrorKind::NotFound, 1), "not a file that will be there in a moment");
    }

    #[test]
    fn a_program_that_hangs_is_stopped_and_one_that_fails_is_quoted() {
        let dir = std::env::temp_dir().join(format!("superfluid-exec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = |name: &str, body: &str| {
            let p = dir.join(name);
            std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
            p
        };
        // exec, so the kill reaches the sleep itself and not only a shell waiting on it.
        let hangs = script("hangs", "exec sleep 30");
        let started = std::time::Instant::now();
        assert!(output_within(&mut Command::new(&hangs), Duration::from_millis(200)).unwrap().is_none());
        assert!(started.elapsed() < Duration::from_secs(10), "stopped at the limit, not when it finished");

        let fails = script("fails", "echo out; echo one >&2; echo >&2; echo two >&2; echo three >&2; echo four >&2; exit 3");
        let out = output_within(&mut Command::new(&fails), Duration::from_secs(10)).unwrap().unwrap();
        assert_eq!((out.status.code(), out.stdout.as_slice()), (Some(3), b"out\n".as_slice()));
        assert_eq!(stderr_tail(&out.stderr), ": two / three / four", "the last three lines, blank ones skipped");
        assert_eq!(stderr_tail(b""), "");
        let missing = output_within(&mut Command::new(dir.join("missing")), Duration::from_secs(1)).unwrap_err();
        assert_eq!(missing.kind(), ErrorKind::NotFound);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
