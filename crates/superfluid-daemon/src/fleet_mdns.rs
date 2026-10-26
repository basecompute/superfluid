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

/// Announces this head's node listener; the service stays announced while the daemon lives.
pub fn advertise(port: u16, model: &str) -> Result<ServiceDaemon, String> {
    let daemon = ServiceDaemon::new().map_err(|e| e.to_string())?;
    daemon.set_service_name_len_max(SERVICE_NAME_LEN_MAX).map_err(|e| e.to_string())?;
    // mdns-sd checks a registration on its own thread and tells only its monitors of a refusal.
    let events = daemon.monitor().map_err(|e| e.to_string())?;
    let host = host_label(&crate::metrics::hostname()).to_string();
    let props: HashMap<String, String> = [("model".to_string(), model.to_string())].into();
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

/// The first head announced on the local network within `wait`: every address it answers on,
/// IPv4 first, and the model it serves.
pub fn find_head(wait: Duration) -> Result<(Vec<SocketAddr>, String), String> {
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
                let mut ips: Vec<_> = s.addresses.iter().map(|a| a.to_ip_addr()).collect();
                ips.sort_by_key(|ip| !ip.is_ipv4());
                if !ips.is_empty() {
                    let model = s.txt_properties.get_property_val_str("model").unwrap_or_default().to_string();
                    break Some((ips.into_iter().map(|ip| SocketAddr::new(ip, s.port)).collect(), model));
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
    fn the_service_name_fits_the_cap_advertise_sets() {
        let name = SERVICE.trim_start_matches('_').split("._").next().unwrap();
        assert_eq!(name, "superfluid-fleet");
        assert!(name.len() <= SERVICE_NAME_LEN_MAX as usize);
    }

    #[test]
    #[ignore = "needs multicast on an interface: cargo test -p superfluid --lib fleet_mdns -- --ignored"]
    fn a_node_finds_the_head_it_announced() {
        let _head = advertise(48456, "test-model").unwrap();
        let (addrs, model) = find_head(Duration::from_secs(10)).unwrap();
        assert_eq!(model, "test-model");
        assert!(addrs.iter().all(|a| a.port() == 48456));
    }
}
