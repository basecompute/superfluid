//! `--nonstream-keepalive <secs>`.

use std::collections::HashSet;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::{HeaderValue, StatusCode};
use axum::response::Response;

use crate::{Daemon, DaemonError};

pub const KEEPALIVE_HEADER: &str = "x-superfluid-keepalive";

struct Inflight {
    session: u64,
    cancels: Arc<crate::CancelSet>,
    active: Arc<Mutex<HashSet<u64>>>,
}

#[derive(Default)]
struct Link {
    abandoned: AtomicBool,
    inflight: Mutex<Option<Inflight>>,
}

impl Link {
    fn abandon(&self) {
        self.abandoned.store(true, Ordering::SeqCst);
        let inflight = self.inflight.lock().expect("keepalive inflight");
        if let Some(f) = inflight.as_ref() {
            let mut cancels = f.cancels.lock().expect("cancel registry");
            if f.active.lock().expect("active set").contains(&f.session) {
                cancels.insert(f.session);
                f.cancels.wake();
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct Hooks {
    admitted: Option<tokio::sync::oneshot::Sender<()>>,
    link: Option<Arc<Link>>,
    keepalive: bool,
}

impl Hooks {
    /// Keepalive mode: the route commits a 200 once admitted, so it checks
    /// the session fits first.
    pub(crate) fn is_on(&self) -> bool {
        self.keepalive
    }

    /// Ties the session to the client connection, so a client that goes
    /// away cancels it.
    pub(crate) fn admitted(&mut self, daemon: &Daemon, session: u64) {
        let Some(link) = &self.link else { return };
        *link.inflight.lock().expect("keepalive inflight") = Some(Inflight {
            session,
            cancels: daemon.cancel_registry(),
            active: daemon.active_registry(),
        });
        if let Some(tx) = self.admitted.take() {
            let _ = tx.send(());
        }
    }

    #[cfg(test)]
    fn admitted_sessionless(&mut self) {
        if let Some(tx) = self.admitted.take() {
            let _ = tx.send(());
        }
    }

    pub(crate) fn gone(&self) -> bool {
        self.link
            .as_ref()
            .is_some_and(|l| l.abandoned.load(Ordering::SeqCst))
    }

    pub(crate) fn cancel_if_gone(&self) {
        if let Some(link) = self.link.as_ref().filter(|l| l.abandoned.load(Ordering::SeqCst)) {
            link.abandon();
        }
    }

    pub(crate) fn check(&self) -> Result<(), DaemonError> {
        if self.gone() {
            return Err(DaemonError::Protocol("generation consumer gone"));
        }
        Ok(())
    }
}

/// Cancels a streamed generation the moment its client goes away: the server
/// drops a stream's receiver when the connection closes, and the next delta
/// would only find that out a tick or two later. Stops watching when dropped.
pub(crate) struct ClosedWatch {
    _stop: tokio::sync::oneshot::Sender<()>,
}

pub(crate) fn cancel_when_closed<T: Send + 'static>(
    daemon: &Daemon,
    session: u64,
    tx: &tokio::sync::mpsc::Sender<T>,
) -> ClosedWatch {
    let inflight = Inflight {
        session,
        cancels: daemon.cancel_registry(),
        active: daemon.active_registry(),
    };
    watch_closed(inflight, tx)
}

fn watch_closed<T: Send + 'static>(inflight: Inflight, tx: &tokio::sync::mpsc::Sender<T>) -> ClosedWatch {
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let link = Link::default();
    *link.inflight.lock().expect("keepalive inflight") = Some(inflight);
    let tx = tx.clone();
    tokio::runtime::Handle::current().spawn(async move {
        tokio::select! {
            biased;
            _ = stopped => {}
            _ = tx.closed() => link.abandon(),
        }
    });
    ClosedWatch { _stop: stop }
}

/// Abandons the link when the route's future is dropped before it finishes:
/// the server drops a request's future when its client disconnects.
pub(crate) struct Watch {
    link: Arc<Link>,
    armed: bool,
}

impl Watch {
    pub(crate) fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        if self.armed {
            self.link.abandon();
        }
    }
}

/// Hooks for a route that answers in one piece, and the guard that cancels
/// its generation if the request is dropped before `disarm`.
pub(crate) fn watch() -> (Hooks, Watch) {
    let link = Arc::new(Link::default());
    let hooks = Hooks {
        admitted: None,
        link: Some(Arc::clone(&link)),
        keepalive: false,
    };
    (hooks, Watch { link, armed: true })
}

pub(crate) async fn respond<F, Fut>(every: Option<Duration>, route: F) -> Response
where
    F: FnOnce(Hooks) -> Fut,
    Fut: Future<Output = Response> + Send + 'static,
{
    let Some(every) = every else {
        return watched(route).await;
    };
    let (admit_tx, mut admit_rx) = tokio::sync::oneshot::channel();
    let link = Arc::new(Link::default());
    let mut fut = Box::pin(route(Hooks {
        admitted: Some(admit_tx),
        link: Some(Arc::clone(&link)),
        keepalive: true,
    }));
    tokio::select! {
        biased;
        resp = &mut fut => return resp,
        Ok(()) = &mut admit_rx => {}
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::convert::Infallible>>(4);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                resp = &mut fut => {
                    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                        .await
                        .unwrap_or_default();
                    let _ = tx.send(Ok(body)).await;
                    return;
                }
                _ = tx.closed() => {
                    link.abandon();
                    return;
                }
                _ = tick.tick() => {
                    if tx.send(Ok(axum::body::Bytes::from_static(b" "))).await.is_err() {
                        link.abandon();
                        return;
                    }
                }
            }
        }
    });
    let mut resp = Response::new(axum::body::Body::from_stream(
        tokio_stream::wrappers::ReceiverStream::new(rx),
    ));
    *resp.status_mut() = StatusCode::OK;
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    resp.headers_mut()
        .insert(KEEPALIVE_HEADER, HeaderValue::from_static("whitespace"));
    resp
}

/// The route as is, with its generation cancelled if the client disconnects
/// before the response is ready.
pub(crate) async fn watched<F, Fut>(route: F) -> Response
where
    F: FnOnce(Hooks) -> Fut,
    Fut: Future<Output = Response> + Send + 'static,
{
    let (hooks, guard) = watch();
    let resp = route(hooks).await;
    guard.disarm();
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use std::io::{Read, Write};

    fn serve(
        every: Option<Duration>,
        admit: bool,
        delay: Duration,
        resp: fn() -> Response,
    ) -> std::net::SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                listener.set_nonblocking(true).unwrap();
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                let app = axum::Router::new().route(
                    "/x",
                    axum::routing::post(move || async move {
                        respond(every, move |mut hooks: Hooks| async move {
                            if admit {
                                hooks.admitted_sessionless();
                            }
                            tokio::time::sleep(delay).await;
                            resp()
                        })
                        .await
                    }),
                );
                axum::serve(listener, app).await.unwrap();
            });
        });
        addr
    }

    fn post(addr: std::net::SocketAddr) -> (String, String, Duration) {
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        let t0 = std::time::Instant::now();
        write!(
            s,
            "POST /x HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-length: 0\r\n\r\n"
        )
        .unwrap();
        let mut raw = Vec::new();
        let mut head_at = None;
        let mut buf = [0u8; 256];
        loop {
            let n = s.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            raw.extend_from_slice(&buf[..n]);
            if head_at.is_none() && raw.windows(4).any(|w| w == b"\r\n\r\n") {
                head_at = Some(t0.elapsed());
            }
        }
        let raw = String::from_utf8(raw).unwrap();
        let (head, payload) = raw.split_once("\r\n\r\n").unwrap();
        let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
            dechunk(payload)
        } else {
            payload.to_string()
        };
        (head.to_string(), body, head_at.unwrap())
    }

    fn dechunk(mut rest: &str) -> String {
        let mut out = String::new();
        while let Some((size, tail)) = rest.split_once("\r\n") {
            let size = usize::from_str_radix(size.trim(), 16).unwrap();
            if size == 0 {
                break;
            }
            out.push_str(&tail[..size]);
            rest = &tail[size + 2..];
        }
        out
    }

    fn openai_error() -> Response {
        crate::http_state::error_response(StatusCode::INTERNAL_SERVER_ERROR, "engine fault".into())
    }

    fn anthropic_error() -> Response {
        crate::anthropic::error_response(StatusCode::INTERNAL_SERVER_ERROR, "engine fault".into())
    }

    #[test]
    fn an_error_after_commit_is_the_routes_json_error() {
        let addr = serve(Some(Duration::from_millis(200)), true, Duration::from_millis(900), openai_error);
        let (head, body, head_at) = post(addr);
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        assert!(head_at < Duration::from_millis(600), "head is immediate: {head_at:?}");
        assert!(head.to_ascii_lowercase().contains("x-superfluid-keepalive: whitespace"), "{head}");
        assert!(head.to_ascii_lowercase().contains("content-type: application/json"), "{head}");
        assert!(body.starts_with(' '), "spaces precede the body: {body:?}");
        let want = axum::body::to_bytes(openai_error().into_body(), usize::MAX);
        let want = tokio::runtime::Runtime::new().unwrap().block_on(want).unwrap();
        assert_eq!(body.trim_start().as_bytes(), &want[..], "byte-identical error object");
        let v: serde_json::Value = serde_json::from_str(&body).expect("leading whitespace is valid JSON");
        assert_eq!(v["error"]["message"], "engine fault");
        assert_eq!(v["error"]["type"], "server_error");
    }

    #[test]
    fn an_anthropic_error_after_commit_keeps_its_shape() {
        let addr = serve(Some(Duration::from_millis(200)), true, Duration::from_millis(700), anthropic_error);
        let (head, body, _) = post(addr);
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "api_error");
        assert_eq!(v["error"]["message"], "engine fault");
    }

    #[test]
    fn an_error_before_admission_keeps_its_status() {
        let addr = serve(Some(Duration::from_millis(100)), false, Duration::from_millis(400), openai_error);
        let (head, body, head_at) = post(addr);
        assert!(head.starts_with("HTTP/1.1 500"), "{head}");
        assert!(head_at >= Duration::from_millis(400), "nothing committed early: {head_at:?}");
        assert!(!head.to_ascii_lowercase().contains("x-superfluid-keepalive"), "{head}");
        assert!(body.starts_with('{'), "no whitespace: {body:?}");
    }

    #[test]
    fn a_client_that_disconnects_abandons_the_route_without_keepalive() {
        static GONE: AtomicBool = AtomicBool::new(false);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                listener.set_nonblocking(true).unwrap();
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                let app = axum::Router::new().route(
                    "/x",
                    axum::routing::post(|| async {
                        respond(None, |hooks: Hooks| async move {
                            assert!(!hooks.is_on(), "no keepalive: nothing is committed early");
                            std::thread::spawn(move || {
                                for _ in 0..500 {
                                    if hooks.gone() {
                                        GONE.store(true, Ordering::SeqCst);
                                        return;
                                    }
                                    std::thread::sleep(Duration::from_millis(10));
                                }
                            });
                            tokio::time::sleep(Duration::from_secs(5)).await;
                            StatusCode::OK.into_response()
                        })
                        .await
                    }),
                );
                axum::serve(listener, app).await.unwrap();
            });
        });
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        write!(s, "POST /x HTTP/1.1\r\nhost: localhost\r\ncontent-length: 0\r\n\r\n").unwrap();
        std::thread::sleep(Duration::from_millis(200));
        assert!(!GONE.load(Ordering::SeqCst), "still connected");
        drop(s);
        let t0 = std::time::Instant::now();
        while !GONE.load(Ordering::SeqCst) && t0.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(GONE.load(Ordering::SeqCst), "a dropped connection abandons the generation");
    }

    #[test]
    fn a_stream_whose_receiver_is_dropped_cancels_its_session_while_watched() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let cancels = Arc::new(crate::CancelSet::default());
        let active = Arc::new(Mutex::new(HashSet::from([7u64])));
        let inflight = |session| Inflight { session, cancels: Arc::clone(&cancels), active: Arc::clone(&active) };
        let settle = || std::thread::sleep(Duration::from_millis(50));
        let _rt = rt.enter();

        let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
        let watch = watch_closed(inflight(7), &tx);
        settle();
        assert!(cancels.lock().unwrap().is_empty(), "still connected");
        drop(rx);
        let t0 = std::time::Instant::now();
        while !cancels.lock().unwrap().contains(&7) && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(cancels.lock().unwrap().remove(&7), "a dropped receiver cancels the session");
        drop(watch);

        let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
        drop(watch_closed(inflight(7), &tx));
        drop(rx);
        settle();
        assert!(cancels.lock().unwrap().is_empty(), "the generation ended first: nothing to cancel");

        let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
        let _watch = watch_closed(inflight(8), &tx);
        drop(rx);
        settle();
        assert!(cancels.lock().unwrap().is_empty(), "a session no longer active is not cancelled");
    }

    #[test]
    fn off_is_the_route_itself() {
        fn ok() -> Response {
            (StatusCode::OK, axum::Json(serde_json::json!({"a": 1}))).into_response()
        }
        let addr = serve(None, true, Duration::from_millis(300), ok);
        let (head, body, head_at) = post(addr);
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        assert!(head_at >= Duration::from_millis(300), "{head_at:?}");
        assert!(head.to_ascii_lowercase().contains("content-length: 7"), "{head}");
        assert!(!head.to_ascii_lowercase().contains("x-superfluid-keepalive"), "{head}");
        assert_eq!(body, r#"{"a":1}"#);
    }
}
