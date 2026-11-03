//! The head on the local network, as `_superfluid-fleet._tcp`, so a node finds it with no
//! address given.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use mdns_sd::{DaemonEvent, ServiceDaemon, ServiceEvent, ServiceInfo};

pub const SERVICE: &str = "_superfluid-fleet._tcp.local.";

/// "superfluid-fleet" is 16 bytes, one over RFC 6335's 15, and mdns-sd refuses a longer name
/// unless told otherwise. Released nodes browse for this name, so it stays and the cap rises.
const SERVICE_NAME_LEN_MAX: u8 = 16;

/// How long `advertise` waits for mdns-sd to announce the service or refuse it.
const REGISTERED: Duration = Duration::from_secs(2);

/// Announces this head's node listener, and its HTTP port for `superfluid top` when it has one
/// off loopback; the service stays announced while the daemon lives.
pub fn advertise(port: u16, models: &[String], http: Option<u16>) -> Result<ServiceDaemon, String> {
    let daemon = ServiceDaemon::new().map_err(|e| e.to_string())?;
    daemon.set_service_name_len_max(SERVICE_NAME_LEN_MAX).map_err(|e| e.to_string())?;
    // mdns-sd checks a registration on its own thread and tells only its monitors of a refusal.
    let events = daemon.monitor().map_err(|e| e.to_string())?;
    let host = host_label(&crate::metrics::hostname()).to_string();
    // `model` is the first, as a node from before `models` reads it.
    let mut props: HashMap<String, String> = [
        ("model".to_string(), models.first().cloned().unwrap_or_default()),
        ("models".to_string(), models.join(",")),
    ]
    .into();
    if let Some(http) = http {
        props.insert("http".to_string(), http.to_string());
    }
    let info = ServiceInfo::new(SERVICE, &format!("superfluid on {host}"), &format!("{host}.local."), (), port, props)
        .map_err(|e| e.to_string())?
        .enable_addr_auto();
    daemon.register(info).map_err(|e| e.to_string())?;
    let deadline = Instant::now() + REGISTERED;
    loop {
        match events.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(DaemonEvent::Announce(..)) => return Ok(daemon),
            Ok(DaemonEvent::Error(e)) => {
                let _ = daemon.shutdown();
                return Err(e.to_string());
            }
            Ok(_) => {}
            Err(_) => {
                let _ = daemon.shutdown();
                return Err(format!("no interface announced it within {} s", REGISTERED.as_secs()));
            }
        }
    }
}

/// The host as one DNS label: macOS's host name already ends in `.local`, and a host named in
/// another domain is announced in `.local` by its first label.
fn host_label(host: &str) -> &str {
    host.split('.').next().filter(|l| !l.is_empty()).unwrap_or(host)
}

/// A head announced on the local network.
#[derive(Debug, Clone)]
pub struct Head {
    /// Where its node listener answers, IPv4 first.
    pub nodes: Vec<SocketAddr>,
    /// The models it serves.
    pub models: Vec<String>,
    /// Where its HTTP API answers, IPv4 first; empty when it announced none.
    pub http: Vec<SocketAddr>,
}

/// The first head announced on the local network within `wait`.
pub fn find_head(wait: Duration) -> Result<Head, String> {
    find_head_where(wait, |_| true)
}

/// The first head announced within `wait` that `wanted` takes.
fn find_head_where(wait: Duration, wanted: impl Fn(&Head) -> bool) -> Result<Head, String> {
    let daemon = ServiceDaemon::new().map_err(|e| e.to_string())?;
    let events = daemon.browse(SERVICE).map_err(|e| e.to_string())?;
    let deadline = Instant::now() + wait;
    let found = loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break None;
        }
        match events.recv_timeout(left) {
            Ok(ServiceEvent::ServiceResolved(s)) => {
                let mut ips: Vec<_> = s.addresses.iter().map(|a| a.to_ip_addr()).filter(reachable).collect();
                ips.sort_by_key(|ip| !ip.is_ipv4());
                if !ips.is_empty() {
                    let txt = |k| s.txt_properties.get_property_val_str(k).unwrap_or_default();
                    let http: Option<u16> = txt("http").parse().ok();
                    let head = Head {
                        nodes: ips.iter().map(|&ip| SocketAddr::new(ip, s.port)).collect(),
                        models: match txt("models") {
                            "" => vec![txt("model").to_string()].into_iter().filter(|m| !m.is_empty()).collect(),
                            all => all.split(',').map(str::to_string).collect(),
                        },
                        http: http.map(|p| ips.iter().map(|&ip| SocketAddr::new(ip, p)).collect()).unwrap_or_default(),
                    };
                    if wanted(&head) {
                        break Some(head);
                    }
                }
            }
            Ok(_) => {}
            Err(mdns_sd::RecvTimeoutError::Timeout) => break None,
            Err(e) => return Err(e.to_string()),
        }
    };
    let _ = daemon.shutdown();
    found.ok_or_else(|| format!("no fleet head answered on the local network within {} s", wait.as_secs()))
}

/// A link-local IPv6 address needs the interface it was heard on, which a `SocketAddr` built
/// from it does not carry, so it cannot be dialed.
fn reachable(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(_) => true,
        std::net::IpAddr::V6(v6) => !v6.is_unicast_link_local(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_is_announced_as_one_label() {
        assert_eq!(host_label("studio.local"), "studio");
        assert_eq!(host_label("spark-1"), "spark-1");
        assert_eq!(host_label("box-a.lab.example.com."), "box-a");
        assert_eq!(host_label(".odd"), ".odd");
    }

    #[test]
    fn a_link_local_address_is_not_dialed() {
        let ip = |s: &str| s.parse::<std::net::IpAddr>().unwrap();
        assert!(reachable(&ip("192.168.1.20")));
        assert!(reachable(&ip("fd7a:115c:a1e0::1")));
        assert!(!reachable(&ip("fe80::1c2b:3a4d:5e6f:7081")));
    }

    #[test]
    fn the_service_name_fits_the_cap_advertise_sets() {
        let name = SERVICE.trim_start_matches('_').split("._").next().unwrap();
        assert_eq!(name, "superfluid-fleet");
        assert!(name.len() <= SERVICE_NAME_LEN_MAX as usize);
    }

    #[test]
    #[ignore = "needs multicast on an interface: cargo test -p superfluid --lib fleet_mdns -- --ignored"]
    fn a_node_finds_the_head_it_announced() {
        let _head = advertise(48456, &["test-model".into(), "other-model".into()], Some(48453)).unwrap();
        // Another head on this network may answer first; this one serves its own models.
        let head = find_head_where(Duration::from_secs(10), |h| h.models.contains(&"test-model".to_string())).unwrap();
        assert_eq!(head.models, ["test-model", "other-model"]);
        assert!(head.nodes.iter().all(|a| a.port() == 48456));
        assert!(!head.http.is_empty() && head.http.iter().all(|a| a.port() == 48453));
    }
}
