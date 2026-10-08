//! The head on the local network, as `_superfluid-fleet._tcp`, so a node finds it with no
//! address given.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

pub const SERVICE: &str = "_superfluid-fleet._tcp.local.";

/// Announces this head's node listener; the service stays announced while the daemon lives.
pub fn advertise(port: u16, model: &str) -> Result<ServiceDaemon, String> {
    let daemon = ServiceDaemon::new().map_err(|e| e.to_string())?;
    let host = crate::metrics::hostname();
    let props: HashMap<String, String> = [("model".to_string(), model.to_string())].into();
    let info = ServiceInfo::new(SERVICE, &format!("superfluid on {host}"), &format!("{host}.local."), (), port, props)
        .map_err(|e| e.to_string())?
        .enable_addr_auto();
    daemon.register(info).map_err(|e| e.to_string())?;
    Ok(daemon)
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
