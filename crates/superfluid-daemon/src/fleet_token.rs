//! `superfluid fleet token`: the secret a head and its nodes authenticate and encrypt their
//! link with, kept at `$SUPERFLUID_HOME/fleet/token`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn path(home: &Path) -> PathBuf {
    home.join("fleet").join("token")
}

/// The token at `home`, if one has been made.
pub fn read(home: &Path) -> Option<Vec<u8>> {
    let t = std::fs::read(path(home)).ok()?;
    let t = t.trim_ascii().to_vec();
    (!t.is_empty()).then_some(t)
}

/// Keeps `token` at `home`, owner-readable, replacing one that differs.
pub fn write(home: &Path, token: &[u8]) -> Result<(), String> {
    if read(home).as_deref() == Some(token) {
        return Ok(());
    }
    save(home, token, false).map(|_| ())
}

/// Writes the token file, owner-readable; `fresh` refuses to replace one that exists, and
/// says so with `false`.
fn save(home: &Path, token: &[u8], fresh: bool) -> Result<bool, String> {
    let p = path(home);
    let dir = p.parent().expect("the token has a directory");
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    use std::os::unix::fs::OpenOptionsExt;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).mode(0o600);
    if fresh {
        opts.create_new(true);
    } else {
        opts.create(true).truncate(true);
    }
    let mut f = match opts.open(&p) {
        Ok(f) => f,
        Err(e) if fresh && e.kind() == std::io::ErrorKind::AlreadyExists => return Ok(false),
        Err(e) => return Err(format!("{}: {e}", p.display())),
    };
    f.write_all(token).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(true)
}

/// The token at `home`, made now (32 random bytes as hex, readable by the owner only) if absent.
pub fn read_or_create(home: &Path) -> Result<Vec<u8>, String> {
    if let Some(t) = read(home) {
        return Ok(t);
    }
    let mut raw = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut raw))
        .map_err(|e| format!("/dev/urandom: {e}"))?;
    let token: String = raw.iter().map(|b| format!("{b:02x}")).collect();
    // Made by another process in the meantime: theirs is the token.
    if !save(home, token.as_bytes(), true)? {
        return read(home).ok_or_else(|| format!("{}: made by another process but empty", path(home).display()));
    }
    Ok(token.into_bytes())
}

pub const HELP: &str = "The fleet's shared token.
Usage: superfluid fleet token          Print the token, making it the first time
  The token lives at $SUPERFLUID_HOME/fleet/token. `serve --fleet` presents it to every
  node and `superfluid-noded` expects it (both read that path by default), and the link
  between them is encrypted with it. Copy the file, or the printed value, to each node.
";

pub fn run(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("token") => {
            let home = crate::runtimes::default_home();
            match read_or_create(&home) {
                Ok(t) => {
                    println!("{}", String::from_utf8_lossy(&t));
                    eprintln!("superfluid: fleet token at {}", path(&home).display());
                }
                Err(e) => {
                    eprintln!("superfluid: {e}");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprint!("{HELP}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_is_made_once_and_read_back() {
        let home = std::env::temp_dir().join(format!("superfluid-fleet-token-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        assert_eq!(read(&home), None);
        let first = read_or_create(&home).unwrap();
        assert_eq!(first.len(), 64);
        assert!(first.iter().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(read_or_create(&home).unwrap(), first);
        assert_eq!(read(&home).unwrap(), first);
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(path(&home)).unwrap().permissions().mode() & 0o777, 0o600);
        std::fs::remove_dir_all(&home).ok();
    }
}
