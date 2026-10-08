//! What a Mach-O executable loads, read from its load commands.

use std::path::Path;

const MH_MAGIC_64: u32 = 0xfeed_facf;
const LC_REQ_DYLD: u32 = 0x8000_0000;
const LC_LOAD_DYLIB: u32 = 0xc;
const LC_LOAD_WEAK_DYLIB: u32 = 0x18 | LC_REQ_DYLD;
const LC_RPATH: u32 = 0x1c | LC_REQ_DYLD;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dylib {
    pub name: String,
    pub weak: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Links {
    pub dylibs: Vec<Dylib>,
    pub rpaths: Vec<String>,
}

impl Links {
    pub fn read(path: &Path) -> Result<Links, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Links::parse(&bytes).ok_or_else(|| format!("{} is not a 64-bit Mach-O executable", path.display()))
    }

    fn parse(bytes: &[u8]) -> Option<Links> {
        let u32_at = |at: usize| -> Option<u32> { Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?)) };
        if u32_at(0)? != MH_MAGIC_64 {
            return None;
        }
        let ncmds = u32_at(16)?;
        let mut at = 32usize;
        let mut links = Links::default();
        for _ in 0..ncmds {
            let (cmd, size) = (u32_at(at)?, u32_at(at + 4)? as usize);
            if size < 8 {
                return None;
            }
            let command = bytes.get(at..at + size)?;
            let text = |offset_at: usize| -> Option<String> {
                let offset = u32::from_le_bytes(command.get(offset_at..offset_at + 4)?.try_into().ok()?) as usize;
                let tail = command.get(offset..)?;
                let end = tail.iter().position(|b| *b == 0).unwrap_or(tail.len());
                Some(String::from_utf8_lossy(&tail[..end]).into_owned())
            };
            match cmd {
                LC_LOAD_DYLIB | LC_LOAD_WEAK_DYLIB => links.dylibs.push(Dylib { name: text(8)?, weak: cmd == LC_LOAD_WEAK_DYLIB }),
                LC_RPATH => links.rpaths.push(text(8)?),
                _ => {}
            }
            at += size;
        }
        Some(links)
    }

    pub fn python(&self) -> Option<&Dylib> {
        self.dylibs.iter().find(|d| {
            let file = d.name.rsplit('/').next().unwrap_or(&d.name);
            file.starts_with("libpython3") || file == "Python" || file == "Python3"
        })
    }

    pub fn loads_from(&self, lib: &Dylib, dir: &str) -> bool {
        let file = lib.name.rsplit('/').next().unwrap_or(&lib.name);
        lib.name == format!("{dir}/{file}") || (lib.name == format!("@rpath/{file}") && self.rpaths.iter().any(|r| r == dir))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(commands: &[(u32, &str)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (cmd, text) in commands {
            let fields = if *cmd == LC_RPATH { 0 } else { 12 };
            let offset = 12 + fields;
            let mut size = offset + text.len() + 1;
            size += (8 - size % 8) % 8;
            body.extend_from_slice(&cmd.to_le_bytes());
            body.extend_from_slice(&(size as u32).to_le_bytes());
            body.extend_from_slice(&(offset as u32).to_le_bytes());
            body.extend(vec![0u8; fields]);
            body.extend_from_slice(text.as_bytes());
            body.extend(vec![0u8; size - offset - text.len()]);
        }
        let mut out = Vec::new();
        out.extend_from_slice(&MH_MAGIC_64.to_le_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out.extend_from_slice(&(commands.len() as u32).to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&[0u8; 8]);
        out.extend(body);
        out
    }

    #[test]
    fn the_libraries_and_rpaths_are_read_from_the_load_commands() {
        let bytes = image(&[
            (LC_LOAD_DYLIB, "/usr/lib/libSystem.B.dylib"),
            (0x2, "a symbol table, skipped"),
            (LC_LOAD_WEAK_DYLIB, "@rpath/libpython3.12.dylib"),
            (LC_RPATH, "@executable_path/../python/lib"),
        ]);
        let links = Links::parse(&bytes).unwrap();
        assert_eq!(links.dylibs.len(), 2);
        assert_eq!(links.rpaths, ["@executable_path/../python/lib"]);
        let python = links.python().unwrap();
        assert_eq!((python.name.as_str(), python.weak), ("@rpath/libpython3.12.dylib", true));
        assert!(links.loads_from(python, "@executable_path/../python/lib"), "through the rpath");
        assert!(!links.loads_from(python, "@executable_path/../lib"));

        let direct = Links::parse(&image(&[(LC_LOAD_DYLIB, "@executable_path/../python/lib/libpython3.12.dylib")])).unwrap();
        let python = direct.python().unwrap();
        assert!(!python.weak && direct.loads_from(python, "@executable_path/../python/lib"));

        let framework = "/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/Python";
        let fixed = Links::parse(&image(&[(LC_LOAD_DYLIB, framework), (LC_RPATH, "@executable_path/../python/lib")])).unwrap();
        let python = fixed.python().unwrap();
        assert_eq!(python.name, framework);
        assert!(!fixed.loads_from(python, "@executable_path/../python/lib"));
        assert_eq!(Links::parse(&image(&[(LC_LOAD_DYLIB, "/usr/lib/libSystem.B.dylib")])).unwrap().python(), None);
    }

    #[test]
    fn what_is_not_a_thin_mach_o_image_is_not_read() {
        assert_eq!(Links::parse(b""), None);
        assert_eq!(Links::parse(&[0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 2]), None, "a universal binary");
        let mut cut = image(&[(LC_LOAD_DYLIB, "/usr/lib/libSystem.B.dylib")]);
        cut.truncate(cut.len() - 4);
        assert_eq!(Links::parse(&cut), None, "a command past the end of the file");
        let e = Links::read(Path::new("/nonexistent/worker")).unwrap_err();
        assert!(e.starts_with("/nonexistent/worker: "), "{e}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn this_executable_links_a_python() {
        let me = std::env::current_exe().unwrap();
        let links = Links::read(&me).unwrap();
        assert!(links.dylibs.iter().any(|d| d.name.ends_with("libSystem.B.dylib")), "{links:?}");
        assert!(links.python().is_some(), "{links:?}");
    }
}
