//! What a runtime reads and what it runs on, as plain data.

use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Format {
    pub id: String,
    pub describe: String,
    pub rule: Rule,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rule {
    File { extensions: Vec<String>, magic: Option<Vec<u8>> },
    Directory { files: Vec<String>, extensions: Vec<String> },
}

impl Format {
    pub fn file(id: &str, describe: &str, extensions: &[&str], magic: Option<&[u8]>) -> Format {
        Format {
            id: id.into(),
            describe: describe.into(),
            rule: Rule::File {
                extensions: extensions.iter().map(|e| e.to_string()).collect(),
                magic: magic.map(<[u8]>::to_vec),
            },
        }
    }

    pub fn directory(id: &str, describe: &str, files: &[&str], extensions: &[&str]) -> Format {
        Format {
            id: id.into(),
            describe: describe.into(),
            rule: Rule::Directory {
                files: files.iter().map(|f| f.to_string()).collect(),
                extensions: extensions.iter().map(|e| e.to_string()).collect(),
            },
        }
    }

    pub fn matches(&self, path: &Path) -> bool {
        match &self.rule {
            Rule::File { extensions, magic } => {
                let named = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| extensions.iter().any(|x| x.eq_ignore_ascii_case(e)));
                named || magic.as_deref().is_some_and(|m| starts_with(path, m))
            }
            Rule::Directory { files, extensions } => {
                path.is_dir()
                    && files.iter().all(|f| path.join(f).is_file())
                    && std::fs::read_dir(path)
                        .map(|rd| {
                            rd.flatten().any(|e| {
                                e.path()
                                    .extension()
                                    .and_then(|x| x.to_str())
                                    .is_some_and(|x| extensions.iter().any(|want| want == x))
                            })
                        })
                        .unwrap_or(false)
            }
        }
    }
}

fn starts_with(path: &Path, magic: &[u8]) -> bool {
    if magic.is_empty() || !std::fs::metadata(path).is_ok_and(|m| m.is_file()) {
        return false;
    }
    let mut head = vec![0u8; magic.len()];
    std::fs::File::open(path)
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut head))
        .map(|_| head == magic)
        .unwrap_or(false)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSource {
    Library(String),
    HuggingFace,
    Engine,
}

pub fn valid_id(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('.')
        && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '-' | '_'))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeInfo {
    pub id: &'static str,
    pub aliases: &'static [&'static str],
    pub formats: Vec<Format>,
    pub tokenizer: TokenizerSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    pub backend: String,
    pub name: String,
    pub memory: u64,
}

impl Device {
    pub fn summary(&self) -> String {
        let mut s = format!("{}: {}", self.backend, self.name);
        if self.memory > 0 {
            s.push_str(&format!(", {:.1} GB", self.memory as f64 / 1e9));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("superfluid-artifact-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_file_format_matches_by_extension_in_any_case_or_by_magic() {
        let d = dir("file");
        let gguf = Format::file("gguf", "a GGUF file", &["gguf"], Some(b"GGUF"));
        std::fs::write(d.join("a.gguf"), b"").unwrap();
        std::fs::write(d.join("a.GGUF"), b"").unwrap();
        std::fs::write(d.join("weights.bin"), b"GGUF\x03\x00\x00\x00").unwrap();
        std::fs::write(d.join("a.base"), b"BASE").unwrap();
        assert!(gguf.matches(&d.join("a.gguf")));
        assert!(gguf.matches(&d.join("a.GGUF")));
        assert!(gguf.matches(&d.join("weights.bin")), "the magic names a file its name does not");
        assert!(!gguf.matches(&d.join("a.base")));
        assert!(!gguf.matches(&d.join("missing.bin")));
        assert!(!gguf.matches(&d), "a directory has no magic");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_directory_format_needs_its_files_and_one_weight() {
        let d = dir("dir");
        let mlx = Format::directory("mlx", "an MLX model directory", &["config.json"], &["safetensors"]);
        assert!(!mlx.matches(&d));
        std::fs::write(d.join("config.json"), b"{}").unwrap();
        assert!(!mlx.matches(&d), "no weights");
        std::fs::write(d.join("model.safetensors"), b"").unwrap();
        assert!(mlx.matches(&d));
        assert!(!mlx.matches(&d.join("config.json")), "a file is not a directory");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_magic_probe_opens_only_regular_files() {
        let d = dir("fifo");
        let fifo = d.join("pipe");
        let c = std::ffi::CString::new(fifo.to_string_lossy().as_bytes()).unwrap();
        extern "C" {
            fn mkfifo(path: *const std::ffi::c_char, mode: u32) -> i32;
        }
        // SAFETY: a valid NUL-terminated path; mkfifo creates the node and
        // touches nothing else.
        assert_eq!(unsafe { mkfifo(c.as_ptr(), 0o600) }, 0, "mkfifo");
        let gguf = Format::file("gguf", "a GGUF file", &["gguf"], Some(b"GGUF"));
        assert!(!gguf.matches(&fifo));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_device_sums_itself_up() {
        let d = Device { backend: "Metal".into(), name: "Apple M5 Pro".into(), memory: 36_000_000_000 };
        assert_eq!(d.summary(), "Metal: Apple M5 Pro, 36.0 GB");
        let cpu = Device { backend: "CPU".into(), name: "Apple M5 Pro".into(), memory: 0 };
        assert_eq!(cpu.summary(), "CPU: Apple M5 Pro");
    }
}
