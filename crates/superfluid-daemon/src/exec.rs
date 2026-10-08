//! Starting another program.

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
}
