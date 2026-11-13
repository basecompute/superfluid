//! A fleet head's nodes over HTTP: `GET /v1/fleet/nodes` lists them and
//! `DELETE /v1/fleet/nodes/{name}` removes one until the head restarts.
//!
//! They sit beside the server's other resources under `/v1`, as `/v1/files` and
//! `/v1/files/{id}` do, so the API's key and key policy cover them as they cover the rest;
//! removing a node is an administrative act, which a policy key needs `admin` for.

use std::sync::Arc;

use axum::extract::{ConnectInfo, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};

use crate::fleet_manager::{FleetManager, NodeEntry, RemoveError};
use crate::http_state::{error_response_typed, AppState, AuthenticatedCredential};

/// The version of the shape both routes answer with; a field is only ever added within one.
pub const SCHEMA: u32 = 1;

pub(crate) async fn nodes_list(State(st): State<AppState>) -> Response {
    let fleet = match fleet_of(&st) {
        Ok(f) => f,
        Err(resp) => return resp,
    };
    let data: Vec<serde_json::Value> = fleet.nodes().iter().map(entry_json).collect();
    Json(serde_json::json!({"object": "list", "schema": SCHEMA, "data": data})).into_response()
}

pub(crate) async fn nodes_remove(
    State(st): State<AppState>,
    ConnectInfo(peer): ConnectInfo<std::net::SocketAddr>,
    cred: Option<Extension<AuthenticatedCredential>>,
    Path(name): Path<String>,
) -> Response {
    let fleet = match fleet_of(&st) {
        Ok(f) => f,
        Err(resp) => return resp,
    };
    // With a key, the auth layer has checked it; with none, only this machine may.
    if cred.is_none() && !peer.ip().is_loopback() {
        return error_response_typed(
            StatusCode::FORBIDDEN,
            format!(
                "removing a fleet node is allowed from this machine only while the head has no API key \
                 (asked from {}); start the head with --api-key, or --key-policy with an admin key, to \
                 remove nodes from elsewhere",
                peer.ip()
            ),
            "permission_error",
            "admin_required",
        );
    }
    match fleet.remove(&name) {
        Ok(entry) => {
            let by = match cred.as_ref().and_then(|c| c.0 .0.strip_prefix("policy:").map(str::to_string)) {
                Some(key) => format!("{}, key '{key}'", peer.ip()),
                None => peer.ip().to_string(),
            };
            eprintln!("superfluid: fleet node '{}' removed (by {by})", entry.status.name);
            let mut body = entry_json(&entry);
            body["schema"] = SCHEMA.into();
            body["removed"] = true.into();
            Json(body).into_response()
        }
        Err(RemoveError::Unknown) => error_response_typed(
            StatusCode::NOT_FOUND,
            format!("no fleet node named '{name}'; GET /v1/fleet/nodes lists them"),
            "invalid_request_error",
            "node_not_found",
        ),
        Err(RemoveError::Ambiguous(addrs)) => error_response_typed(
            StatusCode::CONFLICT,
            format!("{} fleet nodes are named '{name}', at {}; name one by its address", addrs.len(), addrs.join(", ")),
            "invalid_request_error",
            "node_ambiguous",
        ),
    }
}

/// The head's fleet, or the 404 a server that is not a fleet head answers.
#[allow(clippy::result_large_err)]
fn fleet_of(st: &AppState) -> Result<Arc<FleetManager>, Response> {
    st.default_daemon().fleet_manager().ok_or_else(|| {
        error_response_typed(
            StatusCode::NOT_FOUND,
            "this server is not a fleet head, so it has no nodes; `superfluid serve --fleet-listen <addr>` \
             or `--fleet <host:port>` makes one"
                .into(),
            "invalid_request_error",
            "not_a_fleet_head",
        )
    })
}

/// One node as both routes give it.
fn entry_json(n: &NodeEntry) -> serde_json::Value {
    let s = &n.status;
    let since = n.since.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    serde_json::json!({
        "object": "fleet.node",
        "name": s.name,
        "addr": s.addr,
        "connection": if n.joined { "joined" } else { "dialed" },
        "up": s.up,
        "since": since,
        "models": s.models,
        "sessions": n.sessions,
        "lanes": s.lanes,
        "lanes_active": s.lanes_active,
        "queue_depth": s.queue_depth,
        "kv_blocks_used": s.kv_blocks_used,
        "kv_blocks_total": s.kv_blocks_total,
        "decode_tokens": s.decode_tokens,
        "max_context_tokens": n.max_context_tokens,
        "max_lanes": n.max_lanes,
    })
}

/// Where `superfluid fleet nodes` and `fleet remove` look for the head when given none.
const DEFAULT_HEAD: &str = "127.0.0.1:8453";

/// `superfluid fleet nodes [<head>] [--json]`: the head's nodes, a line each; the exit code.
pub fn nodes_cli(args: &[String]) -> i32 {
    let json = args.iter().any(|a| a == "--json");
    let head = match head_arg(args.iter().filter(|a| *a != "--json")) {
        Ok(h) => h,
        Err(code) => return code,
    };
    let body = match request("GET", &head, "/v1/fleet/nodes") {
        Ok((200, body)) => body,
        Ok((_, body)) => return refused(&head, &body),
        Err(e) => return unreachable(&head, e),
    };
    if json {
        println!("{body}");
        return 0;
    }
    let rows: Vec<[String; 6]> = body["data"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|n| {
            let s = |k: &str| n[k].as_str().unwrap_or_default().to_string();
            let models: Vec<&str> = n["models"].as_array().into_iter().flatten().filter_map(|m| m.as_str()).collect();
            [
                s("name"),
                s("addr"),
                s("connection"),
                if n["up"] == true { "up" } else { "down" }.to_string(),
                format!("{}/{}", n["lanes_active"], n["max_lanes"]),
                if models.is_empty() { "-".to_string() } else { models.join(",") },
            ]
        })
        .collect();
    if rows.is_empty() {
        eprintln!("superfluid: the head at {head} has no nodes");
        return 0;
    }
    let header = ["NAME", "ADDR", "LINK", "STATE", "LANES", "MODELS"].map(String::from);
    let widths: Vec<usize> = (0..6).map(|c| rows.iter().chain([&header]).map(|r| r[c].len()).max().unwrap_or(0)).collect();
    for row in [&header].into_iter().chain(&rows) {
        let cells: Vec<String> = row.iter().zip(&widths).map(|(cell, w)| format!("{cell:w$}")).collect();
        println!("{}", cells.join("  ").trim_end());
    }
    0
}

/// `superfluid fleet remove <name> [<head>]`: removes a node until the head restarts.
pub fn remove_cli(args: &[String]) -> i32 {
    let Some(name) = args.first().filter(|a| !a.starts_with('-')) else {
        eprintln!("superfluid: fleet remove takes the node's name, as `superfluid fleet nodes` lists it");
        return 2;
    };
    let head = match head_arg(args[1..].iter()) {
        Ok(h) => h,
        Err(code) => return code,
    };
    let path = format!("/v1/fleet/nodes/{}", path_segment(name));
    match request("DELETE", &head, &path) {
        Ok((200, body)) => {
            let name = body["name"].as_str().unwrap_or(name);
            eprintln!("superfluid: fleet node '{name}' removed; the head at {head} refuses it until the head restarts");
            0
        }
        Ok((_, body)) => refused(&head, &body),
        Err(e) => unreachable(&head, e),
    }
}

/// The head named among `args` (`host:port`, with or without `http://`), else the default.
fn head_arg<'a>(mut args: impl Iterator<Item = &'a String>) -> Result<String, i32> {
    let head = args.next().map(|h| h.trim_start_matches("http://").trim_end_matches('/').to_string());
    if let Some(extra) = args.next() {
        eprintln!("superfluid: unexpected argument {extra}");
        return Err(2);
    }
    Ok(head.unwrap_or_else(|| DEFAULT_HEAD.to_string()))
}

/// `method` on the head's `path`, with `$SUPERFLUID_API_KEY` when set: the status and body.
fn request(method: &str, head: &str, path: &str) -> Result<(u16, serde_json::Value), ureq::Error> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(10)))
        .http_status_as_error(false)
        .build()
        .new_agent();
    let url = format!("http://{head}{path}");
    let mut req = if method == "DELETE" { agent.delete(&url) } else { agent.get(&url) };
    if let Some(key) = std::env::var("SUPERFLUID_API_KEY").ok().filter(|k| !k.is_empty()) {
        req = req.header("authorization", &format!("Bearer {key}"));
    }
    let mut resp = req.call()?;
    let status = resp.status().as_u16();
    let text = resp.body_mut().read_to_string()?;
    Ok((status, serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text))))
}

fn refused(head: &str, body: &serde_json::Value) -> i32 {
    match body["error"]["message"].as_str() {
        Some(why) => eprintln!("superfluid: {why}"),
        None if body["error"]["code"] == "invalid_api_key" => {
            eprintln!("superfluid: the head at {head} wants a key: set SUPERFLUID_API_KEY")
        }
        None => eprintln!("superfluid: the head at {head} answered {body}"),
    }
    1
}

fn unreachable(head: &str, e: ureq::Error) -> i32 {
    eprintln!("superfluid: no fleet head answers at {head} ({e}); give its HTTP address, e.g. 10.0.0.5:8453");
    1
}

/// `s` percent-encoded for one path segment.
fn path_segment(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}
