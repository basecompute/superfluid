//! api-web.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::api::{handle_web, serve_requests, EventMsg, Request, Response, Sink};
use crate::{Daemon, DaemonError, EventBody};

#[derive(Clone)]
struct WebState {
    daemon: Arc<Daemon>,
    token: Arc<String>,
    origins: Arc<Vec<String>>,
    fim_bucket: Arc<std::sync::Mutex<crate::completion_bucket::TokenBucket>>,
}

fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

fn header_values<'a>(headers: &'a HeaderMap, name: &str) -> Vec<&'a str> {
    headers
        .get_all(name)
        .iter()
        .map(|v| v.to_str().unwrap_or(""))
        .collect()
}

fn bearer_tokens(headers: &HeaderMap) -> Vec<&str> {
    header_values(headers, "authorization")
        .into_iter()
        .map(|v| v.strip_prefix("Bearer ").unwrap_or(""))
        .collect()
}

fn all_match(presented: &[&str], token: &str) -> bool {
    !presented.is_empty() && presented.iter().all(|t| ct_eq(t, token))
}

fn origins_allowed(state: &WebState, headers: &HeaderMap) -> bool {
    header_values(headers, "origin")
        .iter()
        .all(|origin| state.origins.iter().any(|o| o == origin))
}

fn authorize(state: &WebState, headers: &HeaderMap) -> Result<(), (StatusCode, &'static str)> {
    if !all_match(&bearer_tokens(headers), &state.token) {
        return Err((StatusCode::UNAUTHORIZED, "bad or missing bearer token"));
    }
    if !origins_allowed(state, headers) {
        return Err((StatusCode::FORBIDDEN, "origin not allowed"));
    }
    if !all_match(&header_values(headers, "x-superfluid-csrf"), &state.token) {
        return Err((StatusCode::FORBIDDEN, "missing or bad CSRF token"));
    }
    Ok(())
}

async fn rpc(
    State(state): State<WebState>,
    headers: HeaderMap,
    Json(req): Json<Request>,
) -> axum::response::Response {
    if let Err((code, msg)) = authorize(&state, &headers) {
        return (code, Json(Response::Err { message: msg.into() })).into_response();
    }
    if matches!(req, Request::Complete { .. }) {
        let cfg = state.daemon.completion_bucket();
        let taken = state
            .fim_bucket
            .lock()
            .expect("api-web fim bucket")
            .try_take(cfg, std::time::Instant::now());
        if let Err(wait) = taken {
            state.daemon.note_completion_throttled();
            let secs = crate::completion_bucket::retry_after_secs(wait);
            let body = Response::Throttled {
                retry_after_ms: crate::completion_bucket::retry_after_ms(wait),
            };
            return (
                StatusCode::TOO_MANY_REQUESTS,
                [(axum::http::header::RETRY_AFTER, secs.to_string())],
                Json(body),
            )
                .into_response();
        }
    }
    let resp = handle_web(&state.daemon, req);
    (StatusCode::OK, Json(resp)).into_response()
}

async fn health() -> impl IntoResponse {
    StatusCode::OK
}

#[derive(serde::Deserialize)]
struct StreamReq {
    session: u64,
    max_tokens: u32,
    #[serde(default)]
    provisional: bool,
}

async fn stream(
    State(state): State<WebState>,
    headers: HeaderMap,
    Json(req): Json<StreamReq>,
) -> axum::response::Response {
    if let Err((code, msg)) = authorize(&state, &headers) {
        return (code, Json(Response::Err { message: msg.into() })).into_response();
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<serde_json::Value>(256);
    let daemon = Arc::clone(&state.daemon);
    let max_tokens = req.max_tokens;
    let session = req.session;
    let bus = daemon.bus();
    let sub_id = if req.provisional {
        let mut sub = bus.subscribe(session, true);
        let rx_bus = sub.take_receiver();
        let tx_p = tx.clone();
        tokio::task::spawn_blocking(move || {
            while let Ok(msg) = rx_bus.recv() {
                if let crate::bus::BusMsg::Provisional { channel, text } = msg {
                    if tx_p
                        .blocking_send(serde_json::json!({"provisional": {"channel": channel, "text": text}}))
                        .is_err()
                    {
                        break;
                    }
                }
            }
        });
        Some((sub.session, sub.id))
    } else {
        None
    };
    let bus_end = Arc::clone(&daemon);
    tokio::task::spawn_blocking(move || {
        let _closed = crate::keepalive::cancel_when_closed(&daemon, session, &tx);
        let tx2 = tx.clone();
        let out = daemon.generate_streaming(session, max_tokens, move |ev| {
            let msg: EventMsg = ev.clone().into();
            let carry = matches!(
                &ev.body,
                EventBody::Generated { .. }
                    | EventBody::ToolUse { .. }
                    | EventBody::ToolResult { .. }
                    | EventBody::ToolParseFailure { .. }
            );
            if carry {
                tx2.blocking_send(serde_json::json!({ "committed": msg }))
                    .map_err(|_| DaemonError::Protocol("generation consumer gone"))?;
            }
            Ok(())
        });
        if let Some((s, id)) = sub_id {
            bus_end.bus().unsubscribe(s, id);
        }
        match out {
            Ok(o) => {
                let _ = tx.blocking_send(serde_json::json!({
                    "done": {"tokens_generated": o.tokens_generated, "finish": o.finish, "warm_prefix": o.warm_prefix}
                }));
            }
            Err(e) => {
                let _ = tx.blocking_send(serde_json::json!({"error": e.to_string()}));
            }
        }
    });
    let body = ReceiverStream::new(rx)
        .map(|v| Event::default().data(v.to_string()))
        .chain(tokio_stream::once(Event::default().data("[DONE]")))
        .map(Ok::<_, std::convert::Infallible>);
    Sse::new(body).keep_alive(KeepAlive::default()).into_response()
}

const WS_INBOUND_QUEUE: usize = 32;
const WS_INBOUND_BYTES: usize = 64 << 20;

type QueuedRequest = (Result<Request, String>, tokio::sync::OwnedSemaphorePermit);

const WS_OUTBOUND_BYTES: usize = 16 << 20;

#[derive(Default)]
struct OutBudget {
    state: std::sync::Mutex<(usize, bool)>,
    freed: std::sync::Condvar,
    turn: std::sync::Mutex<()>,
}

impl OutBudget {
    fn reserve(&self, len: usize) -> bool {
        let mut st = self.state.lock().expect("ws budget");
        while !st.1 && st.0 > 0 && st.0 + len > WS_OUTBOUND_BYTES {
            st = self.freed.wait(st).expect("ws budget");
        }
        if st.1 {
            return false;
        }
        st.0 += len;
        true
    }

    fn release(&self, len: usize) {
        let mut st = self.state.lock().expect("ws budget");
        st.0 = st.0.saturating_sub(len);
        self.freed.notify_all();
    }

    fn close(&self) {
        self.state.lock().expect("ws budget").1 = true;
        self.freed.notify_all();
    }
}

struct CloseOnDrop(Arc<OutBudget>);

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        self.0.close();
    }
}

const WS_PROTOCOL: &str = "superfluid.v1";
const WS_TOKEN_PREFIX: &str = "superfluid.token.";

fn offered_protocols(headers: &HeaderMap) -> Vec<String> {
    headers
        .get_all(axum::http::header::SEC_WEBSOCKET_PROTOCOL)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

fn authorize_ws(state: &WebState, headers: &HeaderMap) -> Result<(), (StatusCode, &'static str)> {
    let offered = offered_protocols(headers);
    let presented: Vec<&str> = bearer_tokens(headers)
        .into_iter()
        .chain(
            offered
                .iter()
                .filter_map(|p| p.strip_prefix(WS_TOKEN_PREFIX)),
        )
        .collect();
    if !all_match(&presented, &state.token) {
        return Err((StatusCode::UNAUTHORIZED, "bad or missing bearer token"));
    }
    if header_values(headers, "origin").is_empty() {
        return Err((
            StatusCode::FORBIDDEN,
            "missing Origin (required on /web/ws)",
        ));
    }
    if !origins_allowed(state, headers) {
        return Err((StatusCode::FORBIDDEN, "origin not allowed"));
    }
    if !offered.is_empty() && !offered.iter().any(|p| p == WS_PROTOCOL) {
        return Err((StatusCode::BAD_REQUEST, "offer the superfluid.v1 subprotocol"));
    }
    Ok(())
}

async fn ws(
    State(state): State<WebState>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> axum::response::Response {
    if let Err((code, msg)) = authorize_ws(&state, &headers) {
        return (code, Json(Response::Err { message: msg.into() })).into_response();
    }
    let daemon = Arc::clone(&state.daemon);
    upgrade
        .protocols([WS_PROTOCOL])
        .on_upgrade(move |socket| ws_session(socket, daemon))
}

async fn ws_session(mut socket: WebSocket, daemon: Arc<Daemon>) {
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<String>(256);
    let budget = Arc::new(OutBudget::default());
    let _closer = CloseOnDrop(Arc::clone(&budget));
    let producer_budget = Arc::clone(&budget);
    let (req_tx, mut req_rx) = tokio::sync::mpsc::channel::<QueuedRequest>(WS_INBOUND_QUEUE);
    let inbound = Arc::new(tokio::sync::Semaphore::new(WS_INBOUND_BYTES));
    let sink: Sink = Arc::new(move |msg: &Response| {
        let _turn = producer_budget.turn.lock().expect("ws producer turn");
        let text = serde_json::to_string(msg)
            .map_err(|_| DaemonError::Protocol("unserializable api-web response"))?;
        let len = text.len();
        if !producer_budget.reserve(len) {
            return Err(DaemonError::Protocol("websocket closed"));
        }
        out_tx.blocking_send(text).map_err(|_| {
            producer_budget.release(len);
            DaemonError::Protocol("websocket closed")
        })
    });
    let loop_sink = Arc::clone(&sink);
    let spawned = std::thread::Builder::new()
        .name("superfluid-web-ws".into())
        .spawn(move || {
            let cancels = daemon.cancel_registry();
            let active = daemon.active_registry();
            let err_sink = Arc::clone(&loop_sink);
            let requests = std::iter::from_fn(move || loop {
                match req_rx.blocking_recv().map(|(item, _permit)| item) {
                    Some(Ok(req)) => return Some(Ok(req)),
                    Some(Err(message)) => {
                        if err_sink(&Response::Err { message }).is_err() {
                            return None;
                        }
                    }
                    None => return None,
                }
            });
            let _ = serve_requests(requests, loop_sink, daemon, &cancels, &active);
        });
    drop(sink);
    if spawned.is_err() {
        return;
    }
    let mut slot: Option<tokio::sync::mpsc::Permit<'_, QueuedRequest>> = None;
    let mut pending: Option<(Result<Request, String>, u32)> = None;
    loop {
        tokio::select! {
            permit = inbound.clone().acquire_many_owned(pending.as_ref().map_or(0, |p| p.1)),
                if pending.is_some() => {
                let Ok(permit) = permit else { break };
                let (item, _) = pending.take().expect("guarded");
                slot.take().expect("slot precedes the read").send((item, permit));
            }
            reserved = req_tx.reserve(), if slot.is_none() && pending.is_none() => match reserved {
                Ok(p) => slot = Some(p),
                Err(_) => break,
            },
            frame = socket.recv(), if slot.is_some() && pending.is_none() => {
                let (parsed, len) = match frame {
                    Some(Ok(Message::Text(t))) => {
                        (serde_json::from_str::<Request>(t.as_str()), t.len())
                    }
                    Some(Ok(Message::Binary(b))) => {
                        (serde_json::from_slice::<Request>(&b), b.len())
                    }
                    Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                    Some(Ok(Message::Close(_)) | Err(_)) | None => break,
                };
                let item = parsed.map_err(|e| format!("invalid request frame: {e}"));
                pending = Some((item, len.clamp(1, WS_INBOUND_BYTES) as u32));
            }
            out = out_rx.recv() => {
                let Some(text) = out else { break };
                let len = text.len();
                let sent = socket.send(Message::Text(text.into())).await;
                budget.release(len);
                if sent.is_err() {
                    break;
                }
            }
        }
    }
    let _ = futures_util::SinkExt::close(&mut socket).await;
    drop(slot);
    drop(req_tx);
}

pub fn serve_blocking(
    listener: std::net::TcpListener,
    daemon: Arc<Daemon>,
    token: String,
    origins: Vec<String>,
) -> std::io::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        listener.set_nonblocking(true)?;
        let listener = tokio::net::TcpListener::from_std(listener)?;
        let app = Router::new()
            .route("/web/rpc", post(rpc))
            .route("/web/stream", post(stream))
            .route("/web/ws", get(ws))
            .route("/web/health", get(health))
            .with_state(WebState {
                fim_bucket: Arc::new(std::sync::Mutex::new(
                    crate::completion_bucket::TokenBucket::new(
                        daemon.completion_bucket(),
                        std::time::Instant::now(),
                    ),
                )),
                daemon,
                token: Arc::new(token),
                origins: Arc::new(origins),
            });
        axum::serve(listener, app).await
    })
}

pub fn mint_token(path: &std::path::Path) -> std::io::Result<String> {
    let mut buf = [0u8; 32];
    {
        use std::io::Read;
        std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    }
    let token: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    let mut o = std::fs::OpenOptions::new();
    o.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    use std::io::Write;
    let mut f = o.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    f.write_all(token.as_bytes())?;
    Ok(token)
}

#[cfg(test)]
mod token_tests {
    #[cfg(unix)]
    #[test]
    fn mint_token_forces_owner_only_even_over_a_pre_existing_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("superfluid-tok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("web-token");
        std::fs::write(&path, b"old").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let _tok = super::mint_token(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "token file must be owner-only after mint");
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn outbound_budget_backpressures_and_closes() {
        let b = Arc::new(OutBudget::default());
        assert!(
            b.reserve(WS_OUTBOUND_BYTES * 2),
            "an oversized frame alone passes"
        );
        b.release(WS_OUTBOUND_BYTES * 2);
        assert!(b.reserve(WS_OUTBOUND_BYTES - 1));
        let waiter = {
            let b = Arc::clone(&b);
            std::thread::spawn(move || b.reserve(2))
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!waiter.is_finished(), "over budget: the producer waits");
        b.release(WS_OUTBOUND_BYTES - 1);
        assert!(waiter.join().unwrap(), "released bytes let it through");
        let waiter = {
            let b = Arc::clone(&b);
            std::thread::spawn(move || b.reserve(WS_OUTBOUND_BYTES))
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(CloseOnDrop(Arc::clone(&b)));
        assert!(!waiter.join().unwrap(), "a closed socket fails the wait");
    }
}
