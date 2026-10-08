//! OS memory-pressure sources.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureLevel {
    Normal,
    Warning,
    Critical,
}

pub trait PressureSource {
    fn level(&mut self) -> PressureLevel;
}

#[derive(Clone, Default)]
pub struct ManualPressure {
    level: Arc<AtomicU8>,
}

impl ManualPressure {
    pub fn new() -> ManualPressure {
        ManualPressure::default()
    }
    pub fn set(&self, level: PressureLevel) {
        self.level.store(
            match level {
                PressureLevel::Normal => 0,
                PressureLevel::Warning => 1,
                PressureLevel::Critical => 2,
            },
            Ordering::Release,
        );
    }
}

impl PressureSource for ManualPressure {
    fn level(&mut self) -> PressureLevel {
        match self.level.load(Ordering::Acquire) {
            0 => PressureLevel::Normal,
            1 => PressureLevel::Warning,
            _ => PressureLevel::Critical,
        }
    }
}

#[cfg(target_os = "macos")]
pub struct MacOsPressure;

#[cfg(target_os = "macos")]
impl PressureSource for MacOsPressure {
    fn level(&mut self) -> PressureLevel {
        let name = c"kern.memorystatus_vm_pressure_level";
        let mut value: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        // SAFETY: a well-formed NUL-terminated name, an out buffer of
        // the declared size, no new value.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut value as *mut i32 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            return PressureLevel::Normal;
        }
        match value {
            4 => PressureLevel::Critical,
            2 => PressureLevel::Warning,
            _ => PressureLevel::Normal,
        }
    }
}

#[cfg(target_os = "linux")]
pub struct LinuxPsiPressure {
    pub some_warning_pct: f32,
    pub full_critical_pct: f32,
}

#[cfg(target_os = "linux")]
impl Default for LinuxPsiPressure {
    fn default() -> Self {
        LinuxPsiPressure {
            some_warning_pct: 10.0,
            full_critical_pct: 5.0,
        }
    }
}

#[cfg(target_os = "linux")]
impl PressureSource for LinuxPsiPressure {
    fn level(&mut self) -> PressureLevel {
        let Ok(text) = std::fs::read_to_string("/proc/pressure/memory") else {
            return PressureLevel::Normal;
        };
        let avg10 = |line: &str| -> f32 {
            line.split_whitespace()
                .find_map(|kv| kv.strip_prefix("avg10="))
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0)
        };
        let mut level = PressureLevel::Normal;
        for line in text.lines() {
            if line.starts_with("full") && avg10(line) >= self.full_critical_pct {
                return PressureLevel::Critical;
            }
            if line.starts_with("some") && avg10(line) >= self.some_warning_pct {
                level = PressureLevel::Warning;
            }
        }
        level
    }
}

pub fn os_default() -> Option<Box<dyn PressureSource + Send>> {
    #[cfg(target_os = "macos")]
    {
        Some(Box::new(MacOsPressure))
    }
    #[cfg(target_os = "linux")]
    {
        Some(Box::new(LinuxPsiPressure::default()))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

pub enum PressureConfig {
    None,
    Os,
    Manual(ManualPressure),
}

impl PressureConfig {
    pub(crate) fn take_source(&mut self) -> Option<Box<dyn PressureSource + Send>> {
        match self {
            PressureConfig::None => None,
            PressureConfig::Os => os_default(),
            PressureConfig::Manual(m) => Some(Box::new(m.clone())),
        }
    }
}

impl std::fmt::Debug for PressureConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PressureConfig::None => f.write_str("None"),
            PressureConfig::Os => f.write_str("Os"),
            PressureConfig::Manual(_) => f.write_str("Manual"),
        }
    }
}

impl Clone for PressureConfig {
    fn clone(&self) -> Self {
        match self {
            PressureConfig::None => PressureConfig::None,
            PressureConfig::Os => PressureConfig::Os,
            PressureConfig::Manual(m) => PressureConfig::Manual(m.clone()),
        }
    }
}
