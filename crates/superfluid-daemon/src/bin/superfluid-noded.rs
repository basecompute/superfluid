//! `superfluid-noded` — the node-agent as its own process.

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use superfluid_daemon::nodeagent::{NodeAgent, DEFAULT_HANDSHAKE_TIMEOUT, DEFAULT_IDLE_TIMEOUT};
use superfluid_daemon::DaemonError;
use superfluid_linkf::LinkFError;

const DEFAULT_LISTEN: &str = "0.0.0.0:8454";

const DEFAULT_LANES: u32 = 8;

fn usage() -> ! {
    eprintln!(
        "usage: superfluid-noded --model <path|org/model[:tag]> [--listen <ip:port>] [--auth-file <path>] \
         [--runtime <id>] [--model-name <id>] [--max-context N] [--max-batch N] [--identity <name>] \
         [--offline] [--insecure-no-auth] [--idle-timeout-ms N] [--handshake-timeout-ms N]\n\
         \n\
         A model given by id is pulled, and its runtime installed, as `superfluid serve` does; the\n\
         context window is sized for this device unless --max-context pins it.\n\
         (--listen defaults to 0.0.0.0:8454, the fleet-worker port next to BASE 8453;\n\
          --auth-file holds the fleet token (default $SUPERFLUID_HOME/fleet/token, from\n\
          `superfluid fleet token`); with it the link is encrypted and both ends checked.\n\
          There is no sampling flag here: a fleet generation must match the local path\n\
          token for token, so the node applies only what the request carries.)"
    );
    std::process::exit(2);
}

fn fail(code: i32, msg: impl std::fmt::Display) -> ! {
    eprintln!("superfluid-noded: {msg}");
    std::process::exit(code);
}

/// Handshakes in progress at once; a connection past this is closed before its handshake, so a
/// burst of connects (a port scan) cannot hold a thread each for the handshake timeout.
const MAX_HANDSHAKES: usize = 64;

fn main() {
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    if let Err(e) = superfluid_daemon::telemetry::init_stderr(&filter) {
        eprintln!("superfluid-noded: logging: {e}");
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut listen: Option<String> = None;
    let mut runtime: Option<String> = None;
    let mut model: Option<String> = None;
    let mut max_context: Option<i32> = None;
    let mut max_batch: Option<u32> = None;
    let mut identity: Option<String> = None;
    let mut model_name_override: Option<String> = None;
    let mut handshake_timeout = DEFAULT_HANDSHAKE_TIMEOUT;
    let mut idle_timeout = DEFAULT_IDLE_TIMEOUT;
    let mut auth_file: Option<PathBuf> = None;
    let mut insecure_no_auth = false;
    let mut offline = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_else(|| usage());
        match a.as_str() {
            "--listen" => listen = Some(val()),
            "--runtime" | "--engine" => runtime = Some(val()),
            "--model" => model = Some(val()),
            "--max-context" => max_context = Some(val().parse().unwrap_or_else(|_| usage())),
            "--max-batch" => max_batch = Some(val().parse().unwrap_or_else(|_| usage())),
            "--identity" => identity = Some(val()),
            "--model-name" => model_name_override = Some(val()),
            "--auth-file" => auth_file = Some(PathBuf::from(val())),
            "--insecure-no-auth" => insecure_no_auth = true,
            "--offline" => offline = true,
            "--handshake-timeout-ms" => {
                handshake_timeout = Duration::from_millis(val().parse().unwrap_or_else(|_| usage()))
            }
            "--idle-timeout-ms" => {
                idle_timeout = Duration::from_millis(val().parse().unwrap_or_else(|_| usage()))
            }
            _ => usage(),
        }
    }
    let listen = listen.unwrap_or_else(|| DEFAULT_LISTEN.to_string());
    let identity = identity.unwrap_or_else(superfluid_daemon::metrics::hostname);
    let lanes = max_batch.unwrap_or(DEFAULT_LANES).max(1);
    let mock = runtime.as_deref() == Some("mock");
    if model.is_none() && !mock {
        usage();
    }

    // Whatever can refuse the node does so before a model is pulled or loaded.
    let loopback = superfluid_daemon::fleet_join::loopback(&listen);
    let home = superfluid_daemon::runtimes::default_home();
    let token_file = superfluid_daemon::fleet_token::read(&home);
    let auth = match (&auth_file, token_file) {
        (Some(p), _) => std::fs::read(p)
            .map(|t| t.trim_ascii().to_vec())
            .unwrap_or_else(|e| fail(1, format!("cannot read --auth-file {}: {e}", p.display()))),
        (None, Some(token)) => {
            eprintln!("superfluid-noded: fleet token from {}", superfluid_daemon::fleet_token::path(&home).display());
            token
        }
        (None, None) if !loopback && !insecure_no_auth => fail(
            2,
            format!(
                "refusing to serve {listen} without a fleet token. Run `superfluid fleet token` on the head \
                 and pass it in --auth-file <path>, bind to loopback, or (behind an already-authenticated \
                 transport like Tailscale/WireGuard) pass --insecure-no-auth to accept transport-only trust."
            ),
        ),
        (None, None) => {
            if !loopback {
                eprintln!(
                    "superfluid-noded: WARNING --insecure-no-auth: accepting any peer that reaches {listen} \
                     (transport-only trust)"
                );
            }
            Vec::new()
        }
    };
    // The port is checked now and bound once the model has loaded, so a head that connects
    // during a long pull is refused at once instead of waiting out its handshake.
    drop(TcpListener::bind(&listen).unwrap_or_else(|e| fail(1, format!("bind {listen} failed: {e}"))));

    let loaded = if mock {
        superfluid_daemon::fleet_join::load_mock_node(max_context, lanes, &superfluid_daemon::fleet_join::scratch_wal())
    } else {
        let model = model.expect("checked above");
        superfluid_daemon::fleet_join::load_node(&model, runtime.as_deref(), max_context, lanes, offline, &superfluid_daemon::fleet_join::scratch_wal())
    }
    .unwrap_or_else(|e| fail(3, e));
    let model_name = model_name_override.unwrap_or(loaded.model_name);
    eprintln!("superfluid-noded: {}", loaded.summary);

    let listener = TcpListener::bind(&listen).unwrap_or_else(|e| fail(1, format!("bind {listen} failed: {e}")));
    let addr = listener.local_addr().map(|a| a.to_string()).unwrap_or(listen);
    eprintln!("superfluid-noded: node '{identity}' serving '{model_name}' on {addr} (Link F/TCP)");
    if !auth.is_empty() {
        eprintln!("superfluid-noded: the link is encrypted and both ends are checked against the token");
    }
    let node = NodeAgent::new(Arc::new(loaded.daemon), identity, model_name)
        .with_max_lanes(lanes)
        .with_timeouts(handshake_timeout, idle_timeout)
        .with_auth(auth.clone());
    let handshaking = Arc::new(AtomicUsize::new(0));
    for sock in listener.incoming() {
        let sock = match sock {
            Ok(sock) => sock,
            // Out of descriptors, or a connection aborted before it was taken: the node keeps
            // serving the connections it has.
            Err(e) => {
                eprintln!("superfluid-noded: accept failed: {e}");
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
        };
        let peer = sock.peer_addr().map(|a| a.to_string()).unwrap_or_default();
        if handshaking.fetch_add(1, Ordering::SeqCst) >= MAX_HANDSHAKES {
            handshaking.fetch_sub(1, Ordering::SeqCst);
            tracing::debug!("closed a connection from {peer}: {MAX_HANDSHAKES} handshakes already in progress");
            continue;
        }
        let (mut conn, auth, handshaking) = (node.connection(), auth.clone(), Arc::clone(&handshaking));
        // The handshake runs here, not in the accept loop, so a slow or silent peer holds up
        // only its own connection.
        std::thread::spawn(move || {
            let endpoint = if auth.is_empty() {
                superfluid_linkf::Endpoint::from_tcp(sock)
            } else {
                superfluid_linkf::Endpoint::from_tcp_secure(sock, &auth, superfluid_linkf::Role::Responder, handshake_timeout)
            };
            handshaking.fetch_sub(1, Ordering::SeqCst);
            let endpoint = match endpoint {
                Ok(ep) => ep,
                Err(LinkFError::Unauthorized) => {
                    eprintln!("superfluid-noded: refused {peer}: its fleet token differs");
                    return;
                }
                Err(e) => {
                    eprintln!("superfluid-noded: a connection from {peer} failed its setup: {e}");
                    return;
                }
            };
            match conn.serve(endpoint) {
                Ok(()) => {}
                Err(DaemonError::Fleet(LinkFError::Unauthorized)) => {
                    eprintln!("superfluid-noded: refused {peer}: its fleet token differs")
                }
                // A head that restarts or closes its connections is not news.
                Err(DaemonError::Fleet(LinkFError::Closed)) => tracing::debug!("connection from {peer} closed"),
                Err(e) => eprintln!("superfluid-noded: the connection from {peer} ended: {e}"),
            }
        });
    }
}
