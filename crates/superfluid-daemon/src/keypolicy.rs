//! Per-API-key serving policy (`--key-policy <file.json>`).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use serde::Deserialize;

use crate::qos;
use crate::ratelimit::TokenBucket;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileTable {
    keys: Vec<FileKey>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileKey {
    name: String,
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    key_env: Option<String>,
    #[serde(default)]
    class: Option<String>,
    #[serde(default)]
    max_class: Option<String>,
    #[serde(default)]
    batch_invariant: bool,
    #[serde(default)]
    rate_limit_rpm: u32,
    #[serde(default)]
    max_concurrent: u32,
    #[serde(default)]
    admin: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reject {
    RateLimit,
    Concurrency,
    Qos,
}

#[derive(Debug, Default)]
pub struct KeyStats {
    pub requests: AtomicU64,
    pub rejected_rate: AtomicU64,
    pub rejected_concurrency: AtomicU64,
    pub rejected_qos: AtomicU64,
    pub rejected_admin: AtomicU64,
}

pub struct KeyPolicy {
    pub name: String,
    secret: String,
    digest: [u8; 32],
    pub class: u8,
    pub max_class: u8,
    pub allow_batch_invariant: bool,
    pub rate_limit_rpm: u32,
    pub max_concurrent: u32,
    pub admin: bool,
    bucket: TokenBucket,
    in_flight: Arc<AtomicU32>,
    pub stats: KeyStats,
}

impl std::fmt::Debug for KeyPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyPolicy")
            .field("name", &self.name)
            .field("class", &self.class)
            .field("max_class", &self.max_class)
            .field("allow_batch_invariant", &self.allow_batch_invariant)
            .field("rate_limit_rpm", &self.rate_limit_rpm)
            .field("max_concurrent", &self.max_concurrent)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct InFlight(Option<Arc<AtomicU32>>);

impl Drop for InFlight {
    fn drop(&mut self) {
        if let Some(n) = self.0.take() {
            n.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl KeyPolicy {
    pub fn admit(&self, takes_slot: bool, spends_rate: bool) -> Result<InFlight, Reject> {
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        let slot = if self.max_concurrent == 0 || !takes_slot {
            InFlight(None)
        } else {
            let cap = self.max_concurrent;
            let got = self
                .in_flight
                .try_update(Ordering::AcqRel, Ordering::Acquire, |n| (n < cap).then_some(n + 1));
            if got.is_err() {
                self.stats.rejected_concurrency.fetch_add(1, Ordering::Relaxed);
                return Err(Reject::Concurrency);
            }
            InFlight(Some(Arc::clone(&self.in_flight)))
        };
        if spends_rate && !self.bucket.allow() {
            self.stats.rejected_rate.fetch_add(1, Ordering::Relaxed);
            return Err(Reject::RateLimit);
        }
        Ok(slot)
    }

    pub fn pace(&self, stop: impl Fn() -> bool) -> bool {
        loop {
            if stop() {
                return false;
            }
            if self.bucket.allow() {
                self.stats.requests.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            std::thread::sleep(self.bucket.next_token_in().min(std::time::Duration::from_millis(500)));
        }
    }

    pub fn charge_rate(&self) -> bool {
        if self.bucket.allow() {
            return true;
        }
        self.stats.rejected_rate.fetch_add(1, Ordering::Relaxed);
        false
    }

    pub fn in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::Acquire)
    }

    pub fn note_qos_reject(&self) {
        self.stats.rejected_qos.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_admin_reject(&self) {
        self.stats.rejected_admin.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn sha256(s: &str) -> [u8; 32] {
    use sha2::Digest;
    sha2::Sha256::digest(s.as_bytes()).into()
}

#[derive(Clone, Debug)]
pub struct ClientKey(pub Arc<KeyPolicy>, pub Option<Arc<InFlight>>);

#[derive(Debug)]
pub struct KeyTable {
    keys: Vec<Arc<KeyPolicy>>,
    by_digest: std::collections::HashMap<[u8; 32], usize>,
}

fn parse_class(field: &str, name: &str, v: &str, redact: &dyn Fn(&str) -> String) -> Result<u8, String> {
    qos::parse(v).ok_or_else(|| {
        format!(
            "key {name:?}: {field} {} is not a class (expected one of {})",
            redact(v),
            qos::NAMES.join(", ")
        )
    })
}

impl KeyTable {
    pub fn load(path: &std::path::Path, api_key: Option<&str>) -> Result<KeyTable, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse_with_global(&text, |var| std::env::var(var).ok(), api_key)
            .and_then(|t| t.check_global_key(api_key).map(|()| t))
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(text: &str, env: impl Fn(&str) -> Option<String>) -> Result<KeyTable, String> {
        Self::parse_with_global(text, env, None)
    }

    pub fn parse_with_global(
        text: &str,
        env: impl Fn(&str) -> Option<String>,
        api_key: Option<&str>,
    ) -> Result<KeyTable, String> {
        let file: FileTable = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if file.keys.is_empty() {
            return Err("no keys: a policy file with no keys would lock every client out".into());
        }
        let secrets: Vec<String> = file
            .keys
            .iter()
            .flat_map(|k| k.key.clone().into_iter().chain(k.key_env.as_deref().and_then(&env)))
            .collect();
        let redact = |v: &str| -> String {
            if secrets.iter().any(|s| s == v) || api_key.is_some_and(|g| g == v) {
                "(a value equal to a credential, not shown)".to_string()
            } else {
                format!("{v:?}")
            }
        };
        let mut keys: Vec<Arc<KeyPolicy>> = Vec::with_capacity(file.keys.len());
        for (i, k) in file.keys.into_iter().enumerate() {
            let name = k.name.trim().to_string();
            if secrets.iter().any(|s| *s == name || *s == k.name) {
                return Err(format!(
                    "keys[{i}]: the name equals a key's secret; names are published in metrics and logs"
                ));
            }
            if api_key.is_some_and(|g| g == name || g == k.name) {
                return Err(format!("keys[{i}]: the name equals --api-key; names are published in metrics and logs"));
            }
            if name.is_empty()
                || !name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            {
                return Err(format!("key name {:?} must be non-empty [A-Za-z0-9._-]", k.name));
            }
            let secret = match (k.key, k.key_env) {
                (Some(s), None) => s,
                (None, Some(var)) => {
                    env(&var).ok_or_else(|| format!("key {name:?}: environment variable {} is not set", redact(&var)))?
                }
                _ => return Err(format!("key {name:?}: give exactly one of \"key\" or \"key_env\"")),
            };
            if secret.trim().is_empty() {
                return Err(format!("key {name:?}: the key is empty"));
            }
            if !secret.bytes().all(|b| b.is_ascii_graphic()) {
                return Err(format!(
                    "key {name:?}: the key must be visible ASCII (no spaces, control or non-ASCII characters)"
                ));
            }
            let class = match &k.class {
                Some(c) => parse_class("class", &name, c, &redact)?,
                None => qos::FOREGROUND_AGENT,
            };
            let max_class = match &k.max_class {
                Some(c) => parse_class("max_class", &name, c, &redact)?,
                None => class,
            };
            if max_class > class {
                return Err(format!(
                    "key {name:?}: max_class {} is a lower priority than its default class {}; \
                     max_class is the BEST class the header may ask for",
                    qos::NAMES[max_class as usize],
                    qos::NAMES[class as usize]
                ));
            }
            if keys.iter().any(|o| o.name == name) {
                return Err(format!("key name {name:?} is listed twice"));
            }
            if let Some(o) = keys.iter().find(|o| o.secret == secret) {
                return Err(format!("keys {:?} and {name:?} share a secret", o.name));
            }
            let digest = sha256(&secret);
            keys.push(Arc::new(KeyPolicy {
                name,
                secret,
                digest,
                class,
                max_class,
                allow_batch_invariant: k.batch_invariant,
                rate_limit_rpm: k.rate_limit_rpm,
                max_concurrent: k.max_concurrent,
                admin: k.admin,
                bucket: TokenBucket::new(k.rate_limit_rpm),
                in_flight: Arc::new(AtomicU32::new(0)),
                stats: KeyStats::default(),
            }));
        }
        let by_digest = keys.iter().enumerate().map(|(i, k)| (k.digest, i)).collect();
        Ok(KeyTable { keys, by_digest })
    }

    pub fn check_global_key(&self, api_key: Option<&str>) -> Result<(), String> {
        if let Some(k) = api_key.and_then(|g| self.keys.iter().find(|k| k.secret == g)) {
            return Err(format!("key {:?} has the same secret as --api-key", k.name));
        }
        if let Some(i) = api_key.and_then(|g| self.keys.iter().position(|k| k.name == g)) {
            return Err(format!("keys[{i}]: the name equals --api-key; names are published in metrics and logs"));
        }
        Ok(())
    }

    pub fn lookup(&self, presented: &str) -> Option<Arc<KeyPolicy>> {
        self.by_digest.get(&sha256(presented)).map(|&i| Arc::clone(&self.keys[i]))
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &Arc<KeyPolicy>> {
        self.keys.iter()
    }

    pub fn metrics_text(&self) -> String {
        self.metrics_text_for(None)
    }

    pub fn metrics_text_for(&self, only: Option<&KeyPolicy>) -> String {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let shown: Vec<&Arc<KeyPolicy>> =
            self.keys.iter().filter(|k| only.is_none_or(|o| std::ptr::eq(&***k, o))).collect();
        let mut out = String::new();
        out.push_str("# HELP superfluid_key_requests_total HTTP requests authenticated by a --key-policy key.\n# TYPE superfluid_key_requests_total counter\n");
        for k in &shown {
            out.push_str(&format!("superfluid_key_requests_total{{key=\"{}\"}} {}\n", k.name, g(&k.stats.requests)));
        }
        out.push_str("# HELP superfluid_key_rejected_total Keyed requests refused at the boundary, by reason.\n# TYPE superfluid_key_rejected_total counter\n");
        for k in &shown {
            for (reason, n) in [
                ("rate_limit", &k.stats.rejected_rate),
                ("concurrency", &k.stats.rejected_concurrency),
                ("qos", &k.stats.rejected_qos),
                ("admin", &k.stats.rejected_admin),
            ] {
                out.push_str(&format!(
                    "superfluid_key_rejected_total{{key=\"{}\",reason=\"{reason}\"}} {}\n",
                    k.name,
                    g(n)
                ));
            }
        }
        out.push_str("# HELP superfluid_key_in_flight Requests holding one of a key's max_concurrent slots.\n# TYPE superfluid_key_in_flight gauge\n");
        for k in &shown {
            out.push_str(&format!("superfluid_key_in_flight{{key=\"{}\"}} {}\n", k.name, k.in_flight()));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn parses_defaults_and_env_keys() {
        let t = KeyTable::parse(
            r#"{"keys":[
                {"name":"ui","key":"sk-ui","class":"interactive"},
                {"name":"bg","key_env":"BG_KEY"}
            ]}"#,
            |v| (v == "BG_KEY").then(|| "sk-bg".to_string()),
        )
        .unwrap();
        let ui = t.lookup("sk-ui").unwrap();
        assert_eq!((ui.class, ui.max_class), (qos::INTERACTIVE_CHAT, qos::INTERACTIVE_CHAT));
        let bg = t.lookup("sk-bg").unwrap();
        assert_eq!(bg.name, "bg");
        assert_eq!((bg.class, bg.max_class), (qos::FOREGROUND_AGENT, qos::FOREGROUND_AGENT), "defaults: agent, header may only demote");
        assert!(!bg.allow_batch_invariant);
        assert!(t.lookup("sk-nope").is_none());
        assert!(t.lookup("sk-u").is_none(), "a prefix is not a match");
    }

    #[test]
    fn refuses_bad_tables() {
        let err = |s: &str| KeyTable::parse(s, no_env).unwrap_err();
        assert!(err(r#"{"keys":[]}"#).contains("no keys"));
        assert!(err(r#"{"keys":[{"name":"a","key":"k","clas":"agent"}]}"#).contains("unknown field"));
        assert!(err(r#"{"keys":[{"name":"a"}]}"#).contains("exactly one"));
        assert!(err(r#"{"keys":[{"name":"a","key":"k","key_env":"K"}]}"#).contains("exactly one"));
        assert!(err(r#"{"keys":[{"name":"a","key_env":"K"}]}"#).contains("not set"));
        assert!(err(r#"{"keys":[{"name":"a","key":"  "}]}"#).contains("empty"));
        assert!(err(r#"{"keys":[{"name":"a","key":"sk-é"}]}"#).contains("visible ASCII"));
        assert!(err(r#"{"keys":[{"name":"a","key":"sk a"}]}"#).contains("visible ASCII"));
        assert!(err(r#"{"keys":[{"name":"a b","key":"k"}]}"#).contains("[A-Za-z0-9._-]"));
        assert!(err(r#"{"keys":[{"name":"a","key":"k","class":"urgent"}]}"#).contains("not a class"));
        assert!(err(r#"{"keys":[{"name":"a","key":"k","class":"interactive","max_class":"agent"}]}"#)
            .contains("lower priority"));
        assert!(err(r#"{"keys":[{"name":"a","key":"k"},{"name":"a","key":"j"}]}"#).contains("twice"));
        assert!(err(r#"{"keys":[{"name":"a","key":"k"},{"name":"b","key":"k"}]}"#).contains("share a secret"));
        for bad in [
            r#"{"keys":[{"name":"sk-b","key":"sk-a"},{"name":"b","key":"sk-b"}]}"#,
            r#"{"keys":[{"name":"sk-b","key":"sk-a","class":"urgent"},{"name":"b","key":"sk-b"}]}"#,
            r#"{"keys":[{"name":"b","key":"sk-b"},{"name":"sk-b","key":"sk-a"},{"name":"sk-b","key":"sk-c"}]}"#,
            r#"{"keys":[{"name":"sk-b","key":"sk-b","key_env":"K"}]}"#,
        ] {
            let e = err(bad);
            assert!(e.contains("equals a key's secret") && !e.contains("sk-b"), "{e}");
        }
        let e = KeyTable::parse_with_global(r#"{"keys":[{"name":"sk-g","key":"k","class":"urgent"}]}"#, no_env, Some("sk-g"))
            .unwrap_err();
        assert!(e.contains("keys[0]") && e.contains("--api-key") && !e.contains("sk-g"), "{e}");
        let t = KeyTable::parse(r#"{"keys":[{"name":"a","key":"k"}]}"#, no_env).unwrap();
        assert!(t.check_global_key(Some("k")).unwrap_err().contains("--api-key"));
        let e = t.check_global_key(Some("a")).unwrap_err();
        assert!(e.contains("keys[0]") && !e.contains("\"a\""), "{e}");
        assert!(t.check_global_key(Some("other")).is_ok());
        assert!(t.check_global_key(None).is_ok());
    }

    #[test]
    fn concurrency_slots_release_on_drop() {
        let t = KeyTable::parse(r#"{"keys":[{"name":"a","key":"k","max_concurrent":2}]}"#, no_env).unwrap();
        let k = t.lookup("k").unwrap();
        let a = k.admit(true, true).unwrap();
        let _b = k.admit(true, true).unwrap();
        assert_eq!(k.admit(true, true).unwrap_err(), Reject::Concurrency);
        assert_eq!(k.in_flight(), 2);
        drop(a);
        assert_eq!(k.in_flight(), 1);
        let _c = k.admit(true, true).unwrap();
        assert_eq!(k.stats.requests.load(Ordering::Relaxed), 4);
        assert_eq!(k.stats.rejected_concurrency.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn rate_limit_is_per_key_and_does_not_leak_slots() {
        let t = KeyTable::parse(
            r#"{"keys":[{"name":"a","key":"k","rate_limit_rpm":2,"max_concurrent":8},{"name":"b","key":"j"}]}"#,
            no_env,
        )
        .unwrap();
        let a = t.lookup("k").unwrap();
        let _s1 = a.admit(true, true).unwrap();
        let _s2 = a.admit(true, true).unwrap();
        assert_eq!(a.admit(true, true).unwrap_err(), Reject::RateLimit);
        assert_eq!(a.in_flight(), 2, "a rate-limited request gives its slot back");
        let b = t.lookup("j").unwrap();
        for _ in 0..10 {
            b.admit(true, true).unwrap();
        }
        let m = t.metrics_text();
        assert!(m.contains("superfluid_key_rejected_total{key=\"a\",reason=\"rate_limit\"} 1"), "{m}");
        assert!(m.contains("superfluid_key_requests_total{key=\"b\"} 10"), "{m}");
        assert!(m.contains("superfluid_key_in_flight{key=\"a\"} 2"), "{m}");
        assert!(!m.contains("sk-") && !m.contains("\"k\""), "no secret in metrics: {m}");
    }
}
