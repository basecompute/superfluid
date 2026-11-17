//! The head on the local network, as `_superfluid-fleet._tcp`, so a node finds it with no
//! address given.
//!
//! The announcement's TXT properties: `model`, the head's first model (a node from before
//! `models` reads only it); `models`, every model it serves, comma-separated; `http`, the port
//! of its HTTP API when that is off loopback; and `net`, the id of its fleet token's network
//! ([`network_id`]), when it has a token.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use mdns_sd::{DaemonEvent, ServiceDaemon, ServiceEvent, ServiceInfo};

pub const SERVICE: &str = "_superfluid-fleet._tcp.local.";

/// "superfluid-fleet" is 16 bytes, one over RFC 6335's 15, and mdns-sd refuses a longer name
/// unless told otherwise. Released nodes browse for this name, so it stays and the cap rises.
const SERVICE_NAME_LEN_MAX: u8 = 16;

/// How long `advertise` waits for mdns-sd to announce the service or refuse it.
const REGISTERED: Duration = Duration::from_secs(2);

/// What a fleet token's network is called on the local network: the first 16 hex digits of
/// SHA-256 over a domain-separated token, so nodes take only a head of their own network.
///
/// It reveals nothing practical of a token `superfluid fleet token` made, 256 random bits:
/// finding the token from it is a search of 2^256. A token typed by hand (`--token`) that is
/// short or guessable can be found from it offline, by hashing guesses until one matches, by
/// anyone on the network. Two networks share an id by chance once in 2^64.
pub fn network_id(token: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"superfluid fleet network v1\0");
    h.update(token);
    h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Announces this head's node listener, its HTTP port for `superfluid top` when it has one
/// off loopback, and its network when it has a token; the service stays announced while the
/// daemon lives.
pub fn advertise(port: u16, models: &[String], http: Option<u16>, net: Option<&str>) -> Result<ServiceDaemon, String> {
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
    if let Some(net) = net {
        props.insert("net".to_string(), net.to_string());
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
    /// Its network ([`network_id`]); `None` from a head with no token, or one released before
    /// heads announced it.
    pub net: Option<String>,
}

/// How a head stands to the network a node looks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kin {
    /// Of this network, or any head for a node with no network to look for.
    Ours,
    /// Announces no network, so it cannot be told apart: taken only when no head of this
    /// network answers.
    Unmarked,
    /// Of another network: never taken by a node.
    Foreign,
}

/// How `theirs`, a head's announced network, stands to `ours`, the one looked for.
pub fn kin(ours: Option<&str>, theirs: Option<&str>) -> Kin {
    match (ours, theirs) {
        (None, _) => Kin::Ours,
        (Some(a), Some(b)) if a == b => Kin::Ours,
        (Some(_), None) => Kin::Unmarked,
        (Some(_), Some(_)) => Kin::Foreign,
    }
}

/// A look for the head of one network, across as many waits as it takes: it keeps the first
/// unmarked head heard, and which heads of another network it has already said it saw.
#[derive(Debug, Default)]
pub struct Search {
    net: Option<String>,
    others: bool,
    fallback: Option<Head>,
    foreign: Option<Head>,
    said: HashSet<SocketAddr>,
}

/// What a [`Search`] makes of a head it hears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Heard {
    /// This network's: take it now.
    Take,
    /// Kept, in case no head of this network answers in the wait.
    Kept,
    /// Another network's, heard for the first time in this search.
    Foreign,
    /// Nothing new.
    Nothing,
}

impl Search {
    /// A look for the head of `token`'s network; with no token, for any head.
    pub fn for_token(token: &[u8]) -> Search {
        Search { net: (!token.is_empty()).then(|| network_id(token)), ..Default::default() }
    }

    /// Takes a head of another network too, when no head of this one or unmarked answers: for
    /// `superfluid top`, which only reads. A node never does.
    pub fn or_any(mut self) -> Search {
        self.others = true;
        self
    }

    /// The network looked for.
    pub fn net(&self) -> Option<&str> {
        self.net.as_deref()
    }

    /// Weighs a head heard on the network.
    pub fn hear(&mut self, head: &Head) -> Heard {
        match kin(self.net.as_deref(), head.net.as_deref()) {
            Kin::Ours => Heard::Take,
            Kin::Unmarked => {
                if self.fallback.is_none() {
                    self.fallback = Some(head.clone());
                }
                Heard::Kept
            }
            Kin::Foreign => {
                if self.others && self.foreign.is_none() {
                    self.foreign = Some(head.clone());
                }
                let first = head.nodes.first().is_some_and(|a| self.said.insert(*a));
                if first {
                    Heard::Foreign
                } else {
                    Heard::Nothing
                }
            }
        }
    }

    /// The head a wait that heard none of this network ends with: the first unmarked one, else,
    /// for a search that takes others, the first of another network. `true` with an unmarked one.
    pub fn settle(&mut self) -> Option<(Head, bool)> {
        if let Some(head) = self.fallback.take() {
            return Some((head, true));
        }
        self.foreign.take().filter(|_| self.others).map(|head| (head, false))
    }
}

/// The head of `search`'s network announced within `wait`: the first of it to answer, else an
/// unmarked one heard in the wait. `say` is told of a head of another network the first time
/// the search hears it, and of an unmarked head taken.
pub fn find_head(search: &mut Search, wait: Duration, say: &dyn Fn(String)) -> Result<Head, String> {
    let found = browse(wait, |head| match search.hear(&head) {
        Heard::Take => Some(head),
        Heard::Foreign => {
            let then = if search.others { "taken only if no other answers" } else { "not joining it" };
            say(format!("a fleet head of another network answered at {} (its token is not this machine's); {then}", head.nodes[0]));
            None
        }
        Heard::Kept | Heard::Nothing => None,
    })?;
    if let Some(head) = found {
        return Ok(head);
    }
    match search.settle() {
        Some((head, unmarked)) => {
            if unmarked {
                say(format!(
                    "the head at {} announces no network (it predates network ids), so it cannot be told \
                     from another network's; taking it, as no head of this network answered",
                    head.nodes[0]
                ));
            }
            Ok(head)
        }
        None => Err(format!("no fleet head of this network answered on the local network within {} s", wait.as_secs())),
    }
}

/// Browses for heads for `wait`, handing each one resolved to `take` until it takes one.
fn browse(wait: Duration, mut take: impl FnMut(Head) -> Option<Head>) -> Result<Option<Head>, String> {
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
                let txt = |k: &str| s.txt_properties.get_property_val_str(k).unwrap_or_default().to_string();
                if let Some(head) = head_of(&ips, s.port, txt) {
                    if let Some(head) = take(head) {
                        break Some(head);
                    }
                }
            }
            Ok(_) => {}
            Err(mdns_sd::RecvTimeoutError::Timeout) => break None,
            Err(e) => {
                let _ = daemon.shutdown();
                return Err(e.to_string());
            }
        }
    };
    let _ = daemon.shutdown();
    Ok(found)
}

/// The head an announcement describes, from its addresses, port and TXT properties; none when
/// no address can be dialed.
fn head_of(ips: &[std::net::IpAddr], port: u16, txt: impl Fn(&str) -> String) -> Option<Head> {
    if ips.is_empty() {
        return None;
    }
    let http: Option<u16> = txt("http").parse().ok();
    Some(Head {
        nodes: ips.iter().map(|&ip| SocketAddr::new(ip, port)).collect(),
        models: match txt("models").as_str() {
            "" => Some(txt("model")).filter(|m| !m.is_empty()).into_iter().collect(),
            all => all.split(',').map(str::to_string).collect(),
        },
        http: http.map(|p| ips.iter().map(|&ip| SocketAddr::new(ip, p)).collect()).unwrap_or_default(),
        net: Some(txt("net")).filter(|n| !n.is_empty()),
    })
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

    fn head(port: u16, net: Option<&str>) -> Head {
        Head {
            nodes: vec![SocketAddr::from(([192, 168, 1, 20], port))],
            models: vec!["m".into()],
            http: Vec::new(),
            net: net.map(str::to_string),
        }
    }

    #[test]
    fn a_network_id_is_short_fixed_and_the_tokens_own() {
        let a = network_id(b"0123456789abcdef");
        assert_eq!(a.len(), 16);
        assert!(a.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(a, network_id(b"0123456789abcdef"), "the same token, the same network");
        assert_ne!(a, network_id(b"0123456789abcdeg"));
        assert!(!a.contains("0123456789"), "the id is not the token");
    }

    #[test]
    fn a_node_takes_its_own_network_keeps_an_unmarked_head_and_never_another_networks() {
        let ours = network_id(b"token");
        assert_eq!(kin(Some(&ours), Some(&ours)), Kin::Ours);
        assert_eq!(kin(Some(&ours), None), Kin::Unmarked);
        assert_eq!(kin(Some(&ours), Some("0000000000000000")), Kin::Foreign);
        assert_eq!(kin(None, Some("0000000000000000")), Kin::Ours, "a node with no token takes any head");
        assert_eq!(kin(None, None), Kin::Ours);
    }

    #[test]
    fn a_search_says_another_network_once_and_settles_on_an_unmarked_head() {
        let ours = network_id(b"token");
        let mut search = Search::for_token(b"token");
        assert_eq!(search.net(), Some(ours.as_str()));
        assert_eq!(search.hear(&head(1, Some("0000000000000000"))), Heard::Foreign);
        assert_eq!(search.hear(&head(1, Some("0000000000000000"))), Heard::Nothing, "said once");
        assert_eq!(search.hear(&head(2, None)), Heard::Kept);
        assert_eq!(search.hear(&head(3, None)), Heard::Kept);
        assert_eq!(search.hear(&head(4, Some(&ours))), Heard::Take);
        let (taken, unmarked) = search.settle().expect("an unmarked head was heard");
        assert!(unmarked);
        assert_eq!(taken.nodes[0].port(), 2, "the first unmarked one");
        assert!(search.settle().is_none(), "another network's head is never settled on");
        assert_eq!(search.hear(&head(1, Some("0000000000000000"))), Heard::Nothing, "said once across waits");
    }

    #[test]
    fn a_search_that_takes_others_settles_on_another_network_last() {
        let mut search = Search::for_token(b"token").or_any();
        search.hear(&head(1, Some("0000000000000000")));
        let (taken, unmarked) = search.settle().expect("top takes another network's head when nothing else answers");
        assert_eq!((taken.nodes[0].port(), unmarked), (1, false));
        search.hear(&head(1, Some("0000000000000000")));
        search.hear(&head(2, None));
        assert_eq!(search.settle().map(|(h, _)| h.nodes[0].port()), Some(2), "an unmarked head first");
        let mut none = Search::for_token(b"");
        assert_eq!(none.net(), None);
        assert_eq!(none.hear(&head(1, Some("0000000000000000"))), Heard::Take);
    }

    #[test]
    fn an_announcement_reads_back_as_a_head() {
        let ips = ["192.168.1.20".parse().unwrap(), "fd7a::1".parse().unwrap()];
        let props: HashMap<&str, &str> = [("model", "a"), ("models", "a,b"), ("http", "8453"), ("net", "00112233aabbccdd")].into();
        let head = head_of(&ips, 48456, |k| props.get(k).copied().unwrap_or_default().to_string()).unwrap();
        assert_eq!(head.models, ["a", "b"]);
        assert_eq!(head.net.as_deref(), Some("00112233aabbccdd"));
        assert_eq!(head.http[0], "192.168.1.20:8453".parse().unwrap());
        let old: HashMap<&str, &str> = [("model", "a")].into();
        let head = head_of(&ips, 48456, |k| old.get(k).copied().unwrap_or_default().to_string()).unwrap();
        assert_eq!((head.models, head.net, head.http.len()), (vec!["a".to_string()], None, 0), "a head from before");
        assert!(head_of(&[], 48456, |_| String::new()).is_none());
    }

    #[test]
    #[ignore = "needs multicast on an interface: cargo test -p superfluid --lib fleet_mdns -- --ignored"]
    fn a_node_finds_the_head_it_announced() {
        let net = network_id(b"a_node_finds_the_head_it_announced");
        let _head = advertise(48456, &["test-model".into(), "other-model".into()], Some(48453), Some(&net)).unwrap();
        // Another head on this network may answer first; this one is of its own network.
        let mut search = Search::for_token(b"a_node_finds_the_head_it_announced");
        let head = find_head(&mut search, Duration::from_secs(10), &|_| {}).unwrap();
        assert_eq!(head.models, ["test-model", "other-model"]);
        assert_eq!(head.net.as_deref(), Some(net.as_str()));
        assert!(head.nodes.iter().all(|a| a.port() == 48456));
        assert!(!head.http.is_empty() && head.http.iter().all(|a| a.port() == 48453));
    }
}
