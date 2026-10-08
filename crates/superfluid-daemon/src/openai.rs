//! api-openai.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::{Daemon, DaemonError};
pub use crate::inference::{TokenLimits, SamplingOverrides, SamplingFallback, RequestQos};
pub use crate::http_state::{HttpQosPolicy, QOS_HEADER, BATCH_INVARIANT_HEADER};
use crate::inference::{
    advertised_generation_defaults, repeat_penalty_for, next_id, unix_now,
    ChatRequest, CompletionRequest, RequestPolicy, DEFAULT_MAX_CONTEXT,
    Kind, Message,
};
use crate::http_state::{
    AppState, AuthenticatedCredential, error_response, error_response_code,
    error_response_typed, daemon_error,
};
use crate::openai_output::generation_response;

#[derive(Clone)]
pub struct ServeConfig {
    pub api_key: Option<String>,
    pub rate_limit_per_minute: u32,
    pub default_max_tokens: Option<u32>,
    pub max_context: u32,
    pub drain_timeout_secs: u64,
    pub sampling: SamplingOverrides,
    pub qos: HttpQosPolicy,
    pub keys: Option<Arc<crate::keypolicy::KeyTable>>,
    pub nonstream_keepalive_secs: u64,
    pub surface: HttpSurface,
    pub model_naming: ModelNaming,
    pub identity: ServerIdentity,
}

impl Default for ServeConfig {
    fn default() -> ServeConfig {
        ServeConfig {
            api_key: None,
            rate_limit_per_minute: 0,
            default_max_tokens: None,
            max_context: DEFAULT_MAX_CONTEXT,
            drain_timeout_secs: 60,
            sampling: SamplingOverrides::default(),
            qos: HttpQosPolicy::default(),
            keys: None,
            nonstream_keepalive_secs: 0,
            surface: HttpSurface::default(),
            model_naming: ModelNaming::default(),
            identity: ServerIdentity::default(),
        }
    }
}

/// What the server calls itself: `owned_by` in the model list, and `server`
/// and `build.version` in /props. A program embedding superfluid may give
/// its own name and version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerIdentity {
    pub name: String,
    pub version: String,
}

impl Default for ServerIdentity {
    fn default() -> ServerIdentity {
        ServerIdentity { name: "superfluid".into(), version: env!("CARGO_PKG_VERSION").into() }
    }
}

/// How a request's `model` picks the model that serves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelNaming {
    /// As the OpenAI API has it: every request names its model, exactly.
    #[default]
    Exact,
    /// A request that names no model gets the default one; a name that is no
    /// model's matches one by case, then as part of its name.
    Loose,
}

/// The APIs served beside the OpenAI-compatible routes. `superfluid serve`
/// serves all of them; a program embedding it may serve fewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpSurface {
    /// Anthropic's Messages API, `/v1/messages`.
    pub anthropic: bool,
    /// Ollama's API, `/api/*` and `/`.
    pub ollama: bool,
}

impl HttpSurface {
    pub const OPENAI_ONLY: HttpSurface = HttpSurface { anthropic: false, ollama: false };

    /// What the startup line says is served besides the OpenAI API.
    pub fn also_served(&self) -> &'static str {
        match (self.anthropic, self.ollama) {
            (true, true) => " (also Anthropic /v1/messages and Ollama /api/*)",
            (true, false) => " (also Anthropic /v1/messages)",
            (false, true) => " (also Ollama /api/*)",
            (false, false) => "",
        }
    }
}

impl Default for HttpSurface {
    fn default() -> HttpSurface {
        HttpSurface { anthropic: true, ollama: true }
    }
}

pub fn serve_blocking(
    listener: std::net::TcpListener,
    daemon: Arc<Daemon>,
    model_name: String,
) -> std::io::Result<()> {
    let registry = Arc::new(crate::registry::ModelRegistry::single(model_name, daemon));
    serve_blocking_registry(listener, registry)
}

pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[derive(Clone)]
struct AuthState {
    cfg: ServeConfig,
    api_key_digest: Option<[u8; 32]>,
    ip_limiter: Arc<crate::ratelimit::RateLimiter>,
}

async fn auth_layer(
    axum::extract::State(auth): axum::extract::State<AuthState>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let cfg = &auth.cfg;
    if cfg.api_key.is_none() && cfg.keys.is_none() {
        return next.run(req).await;
    }
    let path = req.uri().path();
    if path == "/health" {
        return next.run(req).await;
    }
    let exempt = matches!(path, "/metrics" | "/v1/metrics");
    let m = req.method().clone();
    let is_cancel = is_exact_batch_cancel(&m, path);
    let lightweight = (m == axum::http::Method::GET
        && (path == "/v1/batches"
            || path == "/v1/files"
            || path.strip_prefix("/v1/batches/").is_some_and(|rest| !rest.contains('/'))
            || (path.starts_with("/v1/files/") && !path.ends_with("/content"))
            || path.starts_with("/v1/responses/")))
        || is_cancel
        || (m == axum::http::Method::DELETE && (path.starts_with("/v1/files/") || path.starts_with("/v1/responses/")));
    let takes_slot = !lightweight;
    let spends_rate = !is_cancel;
    let h = req.headers();
    let bearer = h
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let x_api_key = h.get("x-api-key").and_then(|v| v.to_str().ok());
    let keyed = cfg
        .keys
        .as_deref()
        .and_then(|t| bearer.and_then(|v| t.lookup(v)).or_else(|| x_api_key.and_then(|v| t.lookup(v))));
    let admin_route = *req.method() == axum::http::Method::POST
        && matches!(path, "/v1/models/load" | "/v1/models/unload" | "/v1/lora/load" | "/v1/lora/unload")
        || (*req.method() == axum::http::Method::DELETE && path.starts_with("/v1/models/"));
    if let Some(key) = keyed {
        if admin_route && !key.admin {
            if key.admit(false, true).is_err() {
                return error_response_typed(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!("API key '{}' exceeded its rate limit ({}/min)", key.name, key.rate_limit_rpm),
                    "rate_limit_error",
                    "rate_limit_exceeded",
                );
            }
            let ip_ok = key.rate_limit_rpm > 0
                || req
                    .extensions()
                    .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
                    .is_none_or(|c| auth.ip_limiter.allow(c.0.ip()));
            if !ip_ok {
                return error_response_typed(
                    StatusCode::TOO_MANY_REQUESTS,
                    "Rate limit exceeded".to_string(),
                    "rate_limit_error",
                    "rate_limit_exceeded",
                );
            }
            key.note_admin_reject();
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({"error": {
                    "message": format!(
                        "API key '{}' may not administer the server (model and LoRA load/unload); \
                         use the global --api-key or grant \"admin\": true on this key",
                        key.name
                    ),
                    "type": "permission_error",
                    "param": null,
                    "code": "admin_required",
                }})),
            )
                .into_response();
        }
        let slot = if exempt {
            None
        } else {
            match key.admit(takes_slot, spends_rate) {
                Ok(slot) => Some(slot),
                Err(why) => {
                    tracing::debug!(key = %key.name, reason = ?why, "key policy refused request");
                    let (message, code) = match why {
                        crate::keypolicy::Reject::Concurrency => (
                            format!("API key '{}' is at its concurrent-request limit ({})", key.name, key.max_concurrent),
                            "concurrency_limit_exceeded",
                        ),
                        _ => (
                            format!("API key '{}' exceeded its rate limit ({}/min)", key.name, key.rate_limit_rpm),
                            "rate_limit_exceeded",
                        ),
                    };
                    return (
                        StatusCode::TOO_MANY_REQUESTS,
                        Json(serde_json::json!({"error": {
                            "message": message,
                            "type": "rate_limit_error",
                            "param": null,
                            "code": code,
                        }})),
                    )
                        .into_response();
                }
            }
        };
        let slot = slot.map(Arc::new);
        req.extensions_mut().insert(AuthenticatedCredential::policy_key(&key.name));
        req.extensions_mut().insert(crate::keypolicy::ClientKey(key, slot.clone()));
        let resp = match &slot {
            Some(slot) => {
                let held = Arc::clone(slot);
                match tokio::spawn(async move {
                    let resp = next.run(req).await;
                    drop(held);
                    resp
                })
                .await
                {
                    Ok(resp) => resp,
                    Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("handler failed: {e}")),
                }
            }
            None => next.run(req).await,
        };
        return match slot {
            Some(slot) => {
                let (parts, body) = resp.into_parts();
                Response::from_parts(parts, axum::body::Body::new(HeldBody { inner: body, _slot: slot }))
            }
            None => resp,
        };
    }
    let ok = auth.api_key_digest.as_ref().is_some_and(|want| {
        let is = |v: &str| constant_time_eq(&crate::keypolicy::sha256(v), want);
        bearer.is_some_and(is) || x_api_key.is_some_and(is)
    });
    if ok {
        if let Some(key) = cfg.api_key.as_deref() {
            req.extensions_mut().insert(AuthenticatedCredential::global_key(key));
        }
        return next.run(req).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({"error": {
            "message": "Invalid or missing API key",
            "type": "authentication_error",
            "param": null,
            "code": "invalid_api_key",
        }})),
    )
        .into_response()
}

struct HeldBody {
    inner: axum::body::Body,
    _slot: Arc<crate::keypolicy::InFlight>,
}

impl axum::body::HttpBody for HeldBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

fn is_exact_batch_cancel(m: &axum::http::Method, path: &str) -> bool {
    *m == axum::http::Method::POST
        && path
            .strip_prefix("/v1/batches/")
            .and_then(|r| r.strip_suffix("/cancel"))
            .is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'))
}

#[derive(Clone)]
struct DeferredIpCharge(Arc<crate::ratelimit::RateLimiter>, std::net::IpAddr);

async fn rate_limit_layer(
    axum::extract::State(limiter): axum::extract::State<Arc<crate::ratelimit::RateLimiter>>,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let path = req.uri().path();
    let key_metered = req
        .extensions()
        .get::<crate::keypolicy::ClientKey>()
        .is_some_and(|k| k.0.rate_limit_rpm > 0);
    if matches!(path, "/health" | "/metrics" | "/v1/metrics") || key_metered {
        return next.run(req).await;
    }
    let keyed = req.extensions().get::<crate::keypolicy::ClientKey>().is_some();
    if keyed && is_exact_batch_cancel(req.method(), path) {
        let mut req = req;
        req.extensions_mut().insert(DeferredIpCharge(Arc::clone(&limiter), peer.ip()));
        return next.run(req).await;
    }
    if limiter.allow(peer.ip()) {
        return next.run(req).await;
    }
    (
        StatusCode::TOO_MANY_REQUESTS,
        Json(serde_json::json!({"error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "param": null,
            "code": "rate_limit_exceeded",
        }})),
    )
        .into_response()
}

const MAX_BODY_SIZE: usize = 100 * 1024 * 1024;

pub fn serve_blocking_registry(
    listener: std::net::TcpListener,
    registry: Arc<crate::registry::ModelRegistry>,
) -> std::io::Result<()> {
    serve_blocking_config(listener, registry, ServeConfig::default())
}

fn shutdown_registry_bounded(
    registry: Arc<crate::registry::ModelRegistry>,
    budget: std::time::Duration,
) -> bool {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("superfluid-shutdown".into())
        .spawn(move || {
            registry.shutdown_all();
            let _ = tx.send(());
        })
        .is_ok()
        && rx.recv_timeout(budget).is_ok()
}

fn is_loopback_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }
    let bare = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')).unwrap_or(&host);
    bare.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn host_of_authority(authority: &str) -> &str {
    if let Some(rest) = authority.strip_prefix('[') {
        return rest.split_once(']').map_or(authority, |(h, _)| h);
    }
    match authority.rsplit_once(':') {
        Some((h, port)) if !h.contains(':') && port.chars().all(|c| c.is_ascii_digit()) => h,
        _ => authority,
    }
}

/// Which `Host` headers a listener answers to. A DNS-rebound page sends its own domain name as
/// `Host` and cannot present an API key, so a keyless server off loopback still refuses names
/// that are not this machine's.
#[derive(Clone)]
pub(crate) enum HostGuard {
    Loopback,
    ThisMachine(Arc<[String]>),
    Any,
}

impl HostGuard {
    pub(crate) fn for_listener(cfg: &ServeConfig, listens_on_loopback: bool) -> HostGuard {
        if listens_on_loopback {
            HostGuard::Loopback
        } else if cfg.api_key.is_none() && cfg.keys.is_none() {
            HostGuard::ThisMachine(machine_host_names().into())
        } else {
            HostGuard::Any
        }
    }
}

fn machine_host_names() -> Vec<String> {
    let mut buf = [0u8; 256];
    // SAFETY: buf is writable for its full length; gethostname NUL-terminates within it or fails.
    if unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) } != 0 {
        return Vec::new();
    }
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let name = String::from_utf8_lossy(&buf[..len]).trim_end_matches('.').to_ascii_lowercase();
    if name.is_empty() {
        return Vec::new();
    }
    let short = name.strip_suffix(".local").unwrap_or(&name).to_string();
    let mut names = vec![name.clone(), format!("{short}.local")];
    if short != name {
        names.push(short);
    }
    names
}

fn names_this_machine(host: &str, names: &[String]) -> bool {
    if is_loopback_host(host) {
        return true;
    }
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    let bare = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')).unwrap_or(&host);
    bare.parse::<std::net::IpAddr>().is_ok() || names.contains(&host)
}

/// Why a request that a web page could have sent is refused, if it is. A browser names the page
/// in `Origin`; one that is not served from this machine may not drive the API, keyed or not.
/// The `Host` check is what stops a DNS-rebound page, which sends no `Origin` on a GET.
pub(crate) fn browser_refusal(headers: &HeaderMap, guard: &HostGuard) -> Option<String> {
    if let Some(origin) = headers.get(axum::http::header::ORIGIN) {
        let origin = origin.to_str().unwrap_or("");
        let authority = origin.split_once("://").map(|(_, rest)| rest).unwrap_or("");
        let authority = authority.split('/').next().unwrap_or("");
        if !is_loopback_host(host_of_authority(authority)) {
            return Some(format!(
                "requests from web pages on other origins are refused (Origin: {origin}); call the API from a \
                 server-side client, or from a page served on localhost"
            ));
        }
    }
    let host = headers.get(axum::http::header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("");
    if host.is_empty() {
        return None;
    }
    match guard {
        HostGuard::Loopback if !is_loopback_host(host_of_authority(host)) => Some(format!(
            "this server listens on loopback and answers only to a loopback Host (got Host: {host})"
        )),
        HostGuard::ThisMachine(names) if !names_this_machine(host_of_authority(host), names) => Some(format!(
            "this server has no API key and answers only to an IP address or this machine's own name \
             (got Host: {host}); start it with --api-key or --key-policy to serve other host names, or \
             connect by IP address"
        )),
        _ => None,
    }
}

async fn browser_guard_layer(
    axum::extract::State(guard): axum::extract::State<HostGuard>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    match browser_refusal(req.headers(), &guard) {
        None => next.run(req).await,
        Some(why) => error_response_typed(StatusCode::FORBIDDEN, why, "permission_error", "origin_refused"),
    }
}

pub(crate) fn access_layers(
    app: Router,
    cfg: &ServeConfig,
    ip_limiter: Arc<crate::ratelimit::RateLimiter>,
    listens_on_loopback: bool,
) -> Router {
    app.layer(axum::middleware::from_fn_with_state(Arc::clone(&ip_limiter), rate_limit_layer))
        .layer(axum::middleware::from_fn_with_state(
            AuthState {
                api_key_digest: cfg.api_key.as_deref().map(crate::keypolicy::sha256),
                cfg: cfg.clone(),
                ip_limiter,
            },
            auth_layer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            HostGuard::for_listener(cfg, listens_on_loopback),
            browser_guard_layer,
        ))
}

pub fn serve_blocking_config(
    listener: std::net::TcpListener,
    registry: Arc<crate::registry::ModelRegistry>,
    cfg: ServeConfig,
) -> std::io::Result<()> {
    if let Some(keys) = &cfg.keys {
        keys.check_global_key(cfg.api_key.as_deref())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    }
    let listens_on_loopback = listener.local_addr()?.ip().is_loopback();
    let rt = tokio::runtime::Runtime::new()?;
    let shutdown_registry = Arc::clone(&registry);
    rt.block_on(async move {
        listener.set_nonblocking(true)?;
        let listener = tokio::net::TcpListener::from_std(listener)?;
        let ip_limiter = Arc::new(crate::ratelimit::RateLimiter::new(cfg.rate_limit_per_minute));
        let mut app = Router::new()
            .route("/v1/models/load", post(models_load))
            .route("/v1/models/unload", post(models_unload))
            .route("/v1/models/{model}", get(models_retrieve).delete(models_delete))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .route("/v1/embeddings", post(embeddings))
            .route("/v1/rerank", post(rerank))
            .route("/v1/audio/transcriptions", post(audio_transcriptions))
            .route("/v1/audio/translations", post(audio_translations))
            .route("/v1/files", post(files_upload).get(files_list))
            .route("/v1/files/{id}", get(files_get).delete(files_delete))
            .route("/v1/files/{id}/content", get(files_content))
            .route("/v1/batches", post(batches_create).get(batches_list))
            .route("/v1/batches/{id}", get(batches_get))
            .route("/v1/batches/{id}/cancel", post(batches_cancel))
            .route("/v1/responses", post(crate::responses::create))
            .route("/v1/responses/{id}", get(crate::responses::retrieve).delete(crate::responses::delete))
            .route("/v1/responses/{id}/input_items", get(crate::responses::input_items_list))
            .route("/v1/responses/{id}/cancel", post(crate::responses::cancel))
            .route("/v1/lora/load", post(lora_load))
            .route("/v1/lora/unload", post(lora_unload))
            .route("/v1/lora", get(lora_list));
        if cfg.surface.anthropic {
            app = app.route("/v1/messages", post(crate::anthropic::messages));
        }
        if cfg.surface.ollama {
            app = app.merge(crate::ollama::routes());
        }
        let app = app
            .route("/v1/models", get(models))
            .route("/v1/tokenize", post(tokenize))
            .route("/metrics", get(metrics))
            .route("/v1/metrics", get(metrics))
            .route("/health", get(health))
            .route("/props", get(props))
            .route("/slots", get(slots))
            .route("/slots/{id}/save", post(slots_retired))
            .route("/slots/{id}/restore", post(slots_retired))
            .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_SIZE))
            .with_state(AppState {
                registry,
                limits: TokenLimits {
                    default_max_tokens: cfg.default_max_tokens,
                    max_context: cfg.max_context,
                },
                sampling: cfg.sampling,
                qos: cfg.qos,
                keys: cfg.keys.clone(),
                fim_buckets: Arc::new(crate::completion_bucket::ClientBuckets::new()),
                nonstream_keepalive: (cfg.nonstream_keepalive_secs > 0)
                    .then(|| std::time::Duration::from_secs(cfg.nonstream_keepalive_secs)),
                model_naming: cfg.model_naming,
                identity: Arc::new(cfg.identity.clone()),
            });
        let app = access_layers(app, &cfg, ip_limiter, listens_on_loopback);
        let app = if cfg.surface.ollama {
            app.layer(axum::middleware::from_fn(crate::ollama::error_layer))
        } else {
            app
        };
        let drain = cfg.drain_timeout_secs;
        let draining = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let draining_sig = Arc::clone(&draining);
        let drain_deadline = Arc::new(std::sync::OnceLock::<std::time::Instant>::new());
        let drain_deadline_sig = Arc::clone(&drain_deadline);
        let watchdog_registry = Arc::clone(&shutdown_registry);
        let shutdown = async move {
            let mut sig = match tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::user_defined1(),
            ) {
                Ok(s) => s,
                Err(_) => return std::future::pending::<()>().await,
            };
            sig.recv().await;
            let _ = drain_deadline_sig
                .set(std::time::Instant::now() + std::time::Duration::from_secs(drain));
            draining_sig.store(true, std::sync::atomic::Ordering::SeqCst);
            eprintln!(
                "superfluid: SIGUSR1 — no longer accepting connections, draining for up to {drain}s"
            );
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(drain)).await;
                eprintln!("superfluid: drain window elapsed with requests still in flight — exiting");
                if !shutdown_registry_bounded(watchdog_registry, std::time::Duration::from_secs(5))
                {
                    eprintln!(
                        "superfluid: scheduler teardown did not finish in 5s — exiting anyway. A \
                         park artifact of a session that ended during the drain may be left \
                         behind; the next startup sweep reclaims it only if --park is on with \
                         a non-zero --park-budget-gb AND the directory is over that budget"
                    );
                }
                std::process::exit(0);
            });
        };
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown)
        .await?;
        if draining.load(std::sync::atomic::Ordering::SeqCst) {
            eprintln!("superfluid: drained — exiting");
            if !shutdown_registry_bounded(shutdown_registry, std::time::Duration::from_secs(30)) {
                eprintln!(
                    "superfluid: scheduler teardown did not finish in 30s — exiting anyway. A park \
                     artifact of a session that ended during the drain may be left behind; the \
                     next startup sweep reclaims it only if --park is on with a non-zero \
                     --park-budget-gb AND the directory is over that budget"
                );
            }
            let left = drain_deadline
                .get()
                .map(|d| d.saturating_duration_since(std::time::Instant::now()))
                .unwrap_or_default();
            crate::otlp::shutdown(left.min(std::time::Duration::from_secs(2)));
            std::process::exit(0);
        }
        Ok(())
    })
}

async fn slots_retired() -> Response {
    error_response_code(
        StatusCode::NOT_IMPLEMENTED,
        concat!(
            "POST /slots/{id}/save|restore is retired. A request's tokens are ",
            "durable without it: the WAL session log is truth, and a repeated ",
            "prefix is served from the prefix cache rather than re-prefilled. KV ",
            "itself is NOT sealed to disk by default — that is `--park`, it applies ",
            "to sessions you resume by id through the session API, and a session ",
            "minted per HTTP request is never resumable by one. See OPERATIONS.md.",
        )
            .into(),
        "route_retired",
    )
}

async fn metrics(
    State(state): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
) -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        {
            let mut text = state.default_daemon().metrics_text();
            if let Some(keys) = &state.keys {
                let only = key.as_ref().map(|k| &*k.0 .0).filter(|k| !k.admin);
                text.push_str(&keys.metrics_text_for(only));
            }
            text
        },
    )
}

pub(crate) fn chat_once(daemon: &Daemon, req: &ChatRequest, model_name: &str, policy: RequestPolicy) -> Result<serde_json::Value, DaemonError> {
    let frame = crate::inference::chat_once(daemon, req, policy)?;
    Ok(crate::openai_output::frame_json(Kind::Chat, &next_id(), model_name, unix_now(), frame, false))
}

#[derive(Deserialize)]
struct BatchCreateRequest {
    input_file_id: String,
    #[serde(default = "default_endpoint")]
    endpoint: String,
    #[serde(default = "default_window")]
    completion_window: String,
}
fn default_endpoint() -> String {
    "/v1/chat/completions".to_string()
}
fn default_window() -> String {
    "24h".to_string()
}

async fn batches_create(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let req: BatchCreateRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    let qos = match st.qos.resolve(&headers, key.as_ref().map(|k| &*k.0 .0)) {
        Ok(q) => q,
        Err(resp) => return resp,
    };
    let caller = owner_of(&key);
    let visible_input = st.default_daemon().files().get(&req.input_file_id).is_some_and(|m| visible(&m.owner, caller));
    let input = match st.default_daemon().files().content(&req.input_file_id).filter(|_| visible_input) {
        Some(b) => b,
        None => return error_response(StatusCode::BAD_REQUEST, format!("no such input_file_id: {}", req.input_file_id)),
    };
    let total = String::from_utf8_lossy(&input).lines().filter(|l| !l.trim().is_empty()).count() as u64;
    let store = st.default_daemon().batches();
    let daemon = st.default_daemon();
    let model_name = st.default_name();
    let pace = key.as_ref().map(|k| Arc::clone(&k.0 .0)).filter(|k| k.rate_limit_rpm > 0);
    let hold = key.as_ref().map(|k| k.0.clone());
    let worker_store = Arc::clone(&store);
    match store.start(&req.endpoint, &req.input_file_id, &req.completion_window, total, caller, move |id| {
        let _hold = hold;
        crate::batches::process(daemon, worker_store, model_name, id, qos, pace)
    }) {
        Ok(batch) => (StatusCode::OK, Json(public(&batch))).into_response(),
        Err(crate::batches::BatchesBusy) => error_response_typed(
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "{} batches are already waiting to run; retry when some finish or cancel one",
                crate::batches::MAX_QUEUED_BATCHES
            ),
            "rate_limit_error",
            "batch_queue_full",
        ),
    }
}

fn owner_of(key: &Option<axum::Extension<crate::keypolicy::ClientKey>>) -> Option<&str> {
    key.as_ref().map(|k| k.0 .0.name.as_str())
}

fn visible(owner: &Option<String>, caller: Option<&str>) -> bool {
    caller.is_none_or(|c| owner.as_deref() == Some(c))
}

fn public<T: serde::Serialize>(v: &T) -> serde_json::Value {
    let mut j = serde_json::to_value(v).unwrap_or(serde_json::Value::Null);
    if let Some(o) = j.as_object_mut() {
        o.remove("owner");
    }
    j
}

async fn batches_list(State(st): State<AppState>, key: Option<axum::Extension<crate::keypolicy::ClientKey>>) -> Response {
    let caller = owner_of(&key);
    let data: Vec<serde_json::Value> = st
        .default_daemon()
        .batches()
        .list()
        .into_iter()
        .filter(|b| visible(&b.owner, caller))
        .map(|b| public(&b))
        .collect();
    (StatusCode::OK, Json(serde_json::json!({"object": "list", "data": data}))).into_response()
}

async fn batches_get(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match st.default_daemon().batches().get(&id).filter(|b| visible(&b.owner, owner_of(&key))) {
        Some(b) => (StatusCode::OK, Json(public(&b))).into_response(),
        None => error_response(StatusCode::NOT_FOUND, format!("no such batch: {id}")),
    }
}

async fn batches_cancel(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    ip_charge: Option<axum::Extension<DeferredIpCharge>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let charge = || -> Option<Response> {
        let k = key.as_ref().map(|k| &*k.0 .0)?;
        if !k.charge_rate() {
            return Some(error_response_typed(
                StatusCode::TOO_MANY_REQUESTS,
                format!("API key '{}' exceeded its rate limit ({}/min)", k.name, k.rate_limit_rpm),
                "rate_limit_error",
                "rate_limit_exceeded",
            ));
        }
        if let Some(axum::Extension(DeferredIpCharge(limiter, ip))) = &ip_charge {
            if !limiter.allow(*ip) {
                return Some(error_response_typed(
                    StatusCode::TOO_MANY_REQUESTS,
                    "Rate limit exceeded".to_string(),
                    "rate_limit_error",
                    "rate_limit_exceeded",
                ));
            }
        }
        None
    };
    let store = st.default_daemon().batches();
    let not_found = || charge().unwrap_or_else(|| error_response(StatusCode::NOT_FOUND, format!("no such batch: {id}")));
    if !store.get(&id).is_some_and(|b| visible(&b.owner, owner_of(&key))) {
        return not_found();
    }
    match store.cancel(&id) {
        Some((b, running)) => {
            if !running {
                if let Some(refused) = charge() {
                    return refused;
                }
            }
            (StatusCode::OK, Json(public(&b))).into_response()
        }
        None => not_found(),
    }
}

async fn files_upload(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    mut mp: axum::extract::Multipart,
) -> Response {
    let mut filename = String::new();
    let mut purpose = String::new();
    let mut bytes: Option<Vec<u8>> = None;
    loop {
        let field = match mp.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return multipart_error(&e, "invalid multipart"),
        };
        let name = field.name().map(|s| s.to_string());
        let fname = field.file_name().map(|s| s.to_string());
        match name.as_deref() {
            Some("file") => {
                filename = fname.unwrap_or_else(|| "upload".to_string());
                match field.bytes().await {
                    Ok(b) => bytes = Some(b.to_vec()),
                    Err(e) => return multipart_error(&e, "file read"),
                }
            }
            Some("purpose") => purpose = field.text().await.unwrap_or_default(),
            _ => {}
        }
    }
    let Some(bytes) = bytes else {
        return error_response(StatusCode::BAD_REQUEST, "missing `file` part".into());
    };
    match st.default_daemon().files().put(&filename, &purpose, &bytes, owner_of(&key)) {
        Ok(meta) => (StatusCode::OK, Json(public(&meta))).into_response(),
        Err(e @ DaemonError::FilesQuota { .. }) => error_response_typed(
            StatusCode::PAYLOAD_TOO_LARGE,
            e.to_string(),
            "quota_exceeded",
            "files_quota_exceeded",
        ),
        Err(e) => daemon_error(e),
    }
}

async fn files_list(State(st): State<AppState>, key: Option<axum::Extension<crate::keypolicy::ClientKey>>) -> Response {
    let caller = owner_of(&key);
    let data: Vec<serde_json::Value> = st
        .default_daemon()
        .files()
        .list()
        .into_iter()
        .filter(|m| visible(&m.owner, caller))
        .map(|m| public(&m))
        .collect();
    (StatusCode::OK, Json(serde_json::json!({"object": "list", "data": data}))).into_response()
}

async fn files_get(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match st.default_daemon().files().get(&id).filter(|m| visible(&m.owner, owner_of(&key))) {
        Some(m) => (StatusCode::OK, Json(public(&m))).into_response(),
        None => error_response(StatusCode::NOT_FOUND, format!("no such file: {id}")),
    }
}

async fn files_content(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let daemon = st.default_daemon();
    let files = daemon.files();
    let mine = files.get(&id).is_some_and(|m| visible(&m.owner, owner_of(&key)));
    match files.content(&id).filter(|_| mine) {
        Some(bytes) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
            bytes,
        )
            .into_response(),
        None => error_response(StatusCode::NOT_FOUND, format!("no such file: {id}")),
    }
}

async fn files_delete(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let daemon = st.default_daemon();
    let files = daemon.files();
    let mine = files.get(&id).is_some_and(|m| visible(&m.owner, owner_of(&key)));
    let deleted = mine && files.delete(&id);
    let status = if deleted { StatusCode::OK } else { StatusCode::NOT_FOUND };
    (status, Json(serde_json::json!({"id": id, "object": "file", "deleted": deleted}))).into_response()
}

#[derive(Deserialize)]
struct RerankRequest {
    model: Option<String>,
    query: String,
    #[serde(default)]
    documents: Vec<String>,
    top_n: Option<usize>,
    #[serde(default)]
    return_documents: bool,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

async fn rerank(
    State(st): State<AppState>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    body: axum::body::Bytes,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let req: RerankRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    let (model_name, daemon) = match st.resolve(req.model.as_deref()) {
        Ok(pair) => pair,
        Err(resp) => return resp,
    };
    if req.documents.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "documents must be non-empty".into());
    }
    let hold = key.as_ref().map(|k| k.0.clone());
    let out = tokio::task::spawn_blocking(move || {
        let _hold = hold;
        let q = daemon.embed_as(&req.query, class)?;
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(req.documents.len());
        for (i, doc) in req.documents.iter().enumerate() {
            let d = daemon.embed_as(doc, class)?;
            scored.push((i, cosine(&q, &d)));
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(req.top_n.unwrap_or(req.documents.len()));
        let results: Vec<serde_json::Value> = scored
            .into_iter()
            .map(|(i, s)| {
                let mut r = serde_json::json!({"index": i, "relevance_score": s});
                if req.return_documents {
                    r["document"] = serde_json::json!({"text": req.documents[i]});
                }
                r
            })
            .collect();
        Ok::<_, DaemonError>(results)
    })
    .await;
    match out {
        Ok(Ok(results)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": format!("rerank-{}", next_id().rsplit('-').next().unwrap_or("0")),
                "results": results,
                "model": model_name,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            })),
        )
            .into_response(),
        Ok(Err(e)) => daemon_error(e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn audio_transcriptions(
    State(st): State<AppState>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    mp: axum::extract::Multipart,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    audio_op(st, key, class, mp, false).await
}

async fn audio_translations(
    State(st): State<AppState>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    mp: axum::extract::Multipart,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    audio_op(st, key, class, mp, true).await
}

fn ms_to_secs(ms: i32) -> f64 {
    ms as f64 / 1000.0
}

fn subtitle_ts(ms: i32, comma: bool) -> String {
    let ms = ms.max(0);
    let (h, m, s, milli) = (ms / 3_600_000, (ms / 60_000) % 60, (ms / 1000) % 60, ms % 1000);
    let sep = if comma { ',' } else { '.' };
    format!("{h:02}:{m:02}:{s:02}{sep}{milli:03}")
}

async fn audio_stream(
    daemon: std::sync::Arc<Daemon>,
    tmp_path: String,
    params: superfluid_engine::TranscribeParams,
    hold: Option<crate::keypolicy::ClientKey>,
    class: Option<u8>,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<serde_json::Value>(32);
    tokio::task::spawn_blocking(move || {
        let _hold = hold;
        let started = daemon.transcribe_streaming_as(&tmp_path, &params, class);
        let (segs, done) = match started {
            Ok(pair) => pair,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp_path);
                let _ = tx.blocking_send(serde_json::json!({"error": e.to_string()}));
                return;
            }
        };
        let mut full = String::new();
        let mut n = 0usize;
        loop {
            match segs.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(seg) => {
                    let delta = if n == 0 { seg.text } else { format!(" {}", seg.text) };
                    n += 1;
                    full.push_str(&delta);
                    if tx
                        .blocking_send(
                            serde_json::json!({"type": "transcript.text.delta", "delta": delta}),
                        )
                        .is_err()
                    {
                        break;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if tx.is_closed() {
                        break;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        drop(segs);
        let result = done.recv();
        let _ = std::fs::remove_file(&tmp_path);
        match result {
            Ok(Ok(t)) => {
                let _ = tx.blocking_send(serde_json::json!({
                    "type": "transcript.text.done",
                    "text": full,
                    "languages": [{"code": t.language}],
                }));
            }
            Ok(Err(e)) => {
                let _ = tx.blocking_send(serde_json::json!({"error": format!("Transcription failed: {e}")}));
            }
            Err(_) => {
                let _ = tx.blocking_send(serde_json::json!({"error": "transcription worker went away"}));
            }
        }
    });
    let stream = ReceiverStream::new(rx)
        .map(|v| Event::default().data(v.to_string()))
        .chain(tokio_stream::once(Event::default().data("[DONE]")))
        .map(Ok::<_, std::convert::Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

fn audio_extension(filename: &str) -> Option<&str> {
    match std::path::Path::new(filename).extension() {
        None => Some("wav"),
        Some(e) => e
            .to_str()
            .filter(|e| (1..=8).contains(&e.len()) && e.bytes().all(|b| b.is_ascii_alphanumeric())),
    }
}

/// Writes an upload to a fresh owner-only file under `dir`. The name is random and the open is
/// exclusive, so a file or symlink planted in a shared temp dir is never written through.
fn stage_audio(dir: &std::path::Path, ext: &str, audio: &[u8]) -> std::io::Result<std::path::PathBuf> {
    use std::hash::{BuildHasher, Hasher};
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut last = None;
    for _ in 0..8 {
        let nonce = std::collections::hash_map::RandomState::new().build_hasher().finish();
        let path = dir.join(format!("superfluid-audio-{nonce:016x}.{ext}"));
        match std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path) {
            Ok(mut f) => {
                if let Err(e) = f.write_all(audio) {
                    let _ = std::fs::remove_file(&path);
                    return Err(e);
                }
                return Ok(path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("no free staging name")))
}

async fn audio_op(
    st: AppState,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    class: Option<u8>,
    mut mp: axum::extract::Multipart,
    translate: bool,
) -> Response {
    let mut audio: Option<Vec<u8>> = None;
    let mut filename = String::from("audio.wav");
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut language_set = false;
    let mut response_format = String::from("json");
    let mut want_granularities = false;
    let mut prompt = String::new();
    let mut task: Option<String> = None;
    let mut stream = false;
    loop {
        let field = match mp.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return multipart_error(&e, "invalid multipart"),
        };
        let name = field.name().map(|s| s.to_string());
        let fname = field.file_name().map(|s| s.to_string());
        match name.as_deref() {
            Some("file") => {
                if let Some(f) = fname {
                    filename = f;
                }
                match field.bytes().await {
                    Ok(b) => audio = Some(b.to_vec()),
                    Err(e) => return multipart_error(&e, "file read"),
                }
            }
            Some("model") => model = Some(field.text().await.unwrap_or_default()),
            Some("language") => {
                language = Some(field.text().await.unwrap_or_default());
                language_set = true;
            }
            Some("response_format") => response_format = field.text().await.unwrap_or_else(|_| "json".into()),
            Some("prompt") => prompt = field.text().await.unwrap_or_default(),
            Some("task") if !translate => task = Some(field.text().await.unwrap_or_default()),
            Some("stream") => {
                let v = field.text().await.unwrap_or_default();
                stream = v == "true" || v == "1";
            }
            Some("timestamp_granularities") | Some("timestamp_granularities[]") => {
                want_granularities = true;
            }
            _ => {}
        }
    }
    let translate = translate || task.as_deref() == Some("translate");
    if model.is_none() && st.requires_model() {
        return error_response(StatusCode::BAD_REQUEST, "missing required field: model".into());
    }
    let (_model_name, daemon) = match st.resolve(model.as_deref()) {
        Ok(pair) => pair,
        Err(resp) => return resp,
    };
    let Some(audio) = audio else {
        return error_response(StatusCode::BAD_REQUEST, "missing `file` part".into());
    };

    let Some(ext) = audio_extension(&filename) else {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("unsupported audio file name {filename:?}: the extension must be 1-8 ASCII letters or digits"),
        );
    };
    let tmp = match stage_audio(&std::env::temp_dir(), ext, &audio) {
        Ok(p) => p,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("stage audio: {e}")),
    };
    if want_granularities && !matches!(response_format.as_str(), "verbose_json" | "srt" | "vtt") {
        response_format = "verbose_json".into();
    }
    let want_timestamps = matches!(response_format.as_str(), "verbose_json" | "srt" | "vtt");
    let lang = match language {
        Some(l) if language_set => Some(l),
        _ if translate => Some(String::new()),
        _ => None,
    };
    if prompt.len() > 8192 {
        let mut cut = prompt.len() - 8192;
        while cut < prompt.len() && !prompt.is_char_boundary(cut) {
            cut += 1;
        }
        prompt.drain(..cut);
    }
    let params = superfluid_engine::TranscribeParams {
        language: lang,
        translate,
        timestamps: want_timestamps,
        prompt: (!prompt.is_empty()).then_some(prompt),
    };
    let tmp_path = tmp.to_string_lossy().into_owned();

    if stream {
        return audio_stream(daemon, tmp_path, params, key.map(|k| k.0), class).await;
    }

    let hold = key.as_ref().map(|k| k.0.clone());
    let out = tokio::task::spawn_blocking(move || {
        let _hold = hold;
        let r = daemon.transcribe_as(&tmp_path, &params, class);
        let _ = std::fs::remove_file(&tmp_path);
        r
    })
    .await;
    let t = match out {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => return error_response(StatusCode::BAD_REQUEST, e.to_string()),
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };

    match response_format.as_str() {
        "text" => (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")], t.text).into_response(),
        "srt" => {
            let mut s = String::new();
            for (i, seg) in t.segments.iter().enumerate() {
                s.push_str(&format!(
                    "{}\n{} --> {}\n{}\n\n",
                    i + 1,
                    subtitle_ts(seg.start_ms, true),
                    subtitle_ts(seg.end_ms, true),
                    seg.text.trim()
                ));
            }
            (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")], s).into_response()
        }
        "vtt" => {
            let mut s = String::from("WEBVTT\n\n");
            for seg in &t.segments {
                s.push_str(&format!(
                    "{} --> {}\n{}\n\n",
                    subtitle_ts(seg.start_ms, false),
                    subtitle_ts(seg.end_ms, false),
                    seg.text.trim()
                ));
            }
            (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")], s).into_response()
        }
        "verbose_json" => {
            let segments: Vec<serde_json::Value> = t
                .segments
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    serde_json::json!({
                        "id": i,
                        "seek": 0,
                        "start": ms_to_secs(s.start_ms),
                        "end": ms_to_secs(s.end_ms),
                        "text": s.text,
                        "tokens": [],
                        "temperature": s.temperature,
                        "avg_logprob": s.avg_logprob,
                        "compression_ratio": s.compression_ratio,
                        "no_speech_prob": s.no_speech_prob,
                    })
                })
                .collect();
            let joined = t
                .segments
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            Json(serde_json::json!({
                "task": if translate { "translate" } else { "transcribe" },
                "language": t.language,
                "duration": ms_to_secs(t.duration_ms),
                "text": joined,
                "segments": segments,
            }))
            .into_response()
        }
        _ => Json(serde_json::json!({ "text": t.text })).into_response(),
    }
}

#[derive(Deserialize)]
struct LoraLoadRequest {
    #[serde(default)]
    path: String,
    #[serde(default)]
    adapter: Option<String>,
}

async fn lora_load(
    State(st): State<AppState>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    body: axum::body::Bytes,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let req: LoraLoadRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    let path = req.adapter.unwrap_or(req.path);
    if path.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "missing adapter `path`".into());
    }
    let daemon = st.default_daemon();
    match tokio::task::spawn_blocking(move || crate::with_sync_class(class, || daemon.lora_load(&path))).await {
        Ok(Ok(id)) => (StatusCode::OK, Json(serde_json::json!({"loaded": true, "id": id}))).into_response(),
        Ok(Err(e)) => daemon_error(e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn lora_unload(
    State(st): State<AppState>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let daemon = st.default_daemon();
    match tokio::task::spawn_blocking(move || crate::with_sync_class(class, || daemon.lora_unload())).await {
        Ok(Ok(_)) => (StatusCode::OK, Json(serde_json::json!({"unloaded": true}))).into_response(),
        Ok(Err(e)) => daemon_error(e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn lora_list(
    State(st): State<AppState>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let model = st.default_name();
    let daemon = st.default_daemon();
    match tokio::task::spawn_blocking(move || crate::with_sync_class(class, || daemon.lora_id())).await {
        Ok(Ok(id)) => {
            let data = vec![serde_json::json!({
                "model": model,
                "adapter": id.clone().unwrap_or_default(),
                "active": id.is_some(),
            })];
            (StatusCode::OK, Json(serde_json::json!({"object": "list", "data": data}))).into_response()
        }
        Ok(Err(e)) => daemon_error(e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    model: Option<String>,
    input: serde_json::Value,
    #[serde(default)]
    encoding_format: Option<String>,
}

async fn embeddings(
    State(st): State<AppState>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    body: axum::body::Bytes,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let req: EmbeddingRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    if req.model.is_none() && st.requires_model() {
        return error_response(StatusCode::BAD_REQUEST, "missing required field: model".into());
    }
    let (model_name, daemon) = match st.resolve(req.model.as_deref()) {
        Ok(pair) => pair,
        Err(resp) => return resp,
    };
    let inputs: Vec<String> = match &req.input {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(a) => {
            let mut v = Vec::with_capacity(a.len());
            for item in a {
                match item.as_str() {
                    Some(s) => v.push(s.to_string()),
                    None => return error_response(StatusCode::BAD_REQUEST, "input array must be strings".into()),
                }
            }
            v
        }
        _ => return error_response(StatusCode::BAD_REQUEST, "input must be a string or array of strings".into()),
    };
    if inputs.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "input must be non-empty".into());
    }
    let base64 = req.encoding_format.as_deref() == Some("base64");
    let hold = key.as_ref().map(|k| k.0.clone());
    let out = tokio::task::spawn_blocking(move || {
        let _hold = hold;
        let mut data = Vec::with_capacity(inputs.len());
        let mut prompt_tokens = 0u64;
        for (i, text) in inputs.iter().enumerate() {
            prompt_tokens += daemon.tokenize(text).len() as u64;
            let v = daemon.embed_as(text, class)?;
            let embedding = if base64 {
                use base64::Engine;
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for f in &v {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                serde_json::json!(v)
            };
            data.push(serde_json::json!({"object": "embedding", "index": i, "embedding": embedding}));
        }
        Ok::<_, DaemonError>((data, prompt_tokens))
    })
    .await;
    let (data, prompt_tokens) = match out {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return daemon_error(e),
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let json = serde_json::json!({
        "object": "list",
        "data": data,
        "model": *model_name,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    });
    (StatusCode::OK, Json(json)).into_response()
}

async fn completions(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    cred: Option<axum::Extension<AuthenticatedCredential>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let req: CompletionRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    let keepalive = st.nonstream_keepalive.filter(|_| !req.stream && req.suffix.is_none());
    crate::keepalive::respond(keepalive, move |hooks| async move {
        match crate::http_generation::completions(State(st), key, axum::extract::ConnectInfo(peer), cred, headers, req, hooks).await {
            Ok(generation) => generation_response(generation),
            Err(response) => response,
        }
    })
    .await
}

fn multipart_error(e: &axum::extract::multipart::MultipartError, what: &str) -> Response {
    let status = e.status();
    if status == StatusCode::PAYLOAD_TOO_LARGE {
        return error_response_code(status, "Request body too large".into(), "payload_too_large");
    }
    error_response(status, format!("{what}: {e}"))
}

pub(crate) struct ModelObject<'a> {
    pub(crate) id: &'a str,
    pub(crate) created: u64,
    pub(crate) n_ctx: Option<u32>,
    pub(crate) owned_by: &'a str,
    pub(crate) loaded: bool,
    pub(crate) capabilities: Option<serde_json::Value>,
}

impl ModelObject<'_> {
    pub(crate) fn build(self) -> serde_json::Value {
        let mut meta = serde_json::Map::new();
        if let Some(n) = self.n_ctx {
            meta.insert("n_ctx".into(), n.into());
        }
        let mut v = serde_json::json!({
            "id": self.id,
            "object": "model",
            "created": self.created,
            "owned_by": self.owned_by,
            "loaded": self.loaded,
            "meta": meta,
            "architecture": {"input_modalities": input_modalities(self.capabilities.as_ref())},
        });
        if let Some(caps) = self.capabilities {
            if let Some(rt) = caps.get("runtime").filter(|r| r.is_object()) {
                v["runtime"] = rt.clone();
            }
            v["capabilities"] = caps;
        }
        v
    }
}

fn input_modalities(capabilities: Option<&serde_json::Value>) -> Vec<&'static str> {
    let mut m = vec!["text"];
    let Some(mods) = capabilities.and_then(|c| c.get("modalities")) else {
        return m;
    };
    let supported = |k: &str| mods.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    if supported("image_encode") {
        m.push("image");
    }
    if supported("gemma_audio_encode") {
        m.push("audio");
    }
    m
}

fn capabilities_for(st: &AppState, id: &str, loaded: bool) -> Option<serde_json::Value> {
    let resolved = loaded.then(|| st.registry.resolve(Some(id))).flatten();
    let (_, d) = resolved.as_ref()?;
    let dialect = serde_json::json!({
        "enable_thinking": d.supports_enable_thinking(),
        "reasoning_effort": d.supports_reasoning_effort(),
        "reasoning_effort_levels": d.reasoning_effort_levels(),
        "fim": d.fim_supported(),
        "fim_model": d.fim_satellite_name(),
    });
    match d.capability_descriptor() {
        Some(mut c) => {
            match c.as_object_mut() {
                Some(obj) => {
                    obj.insert("dialect".into(), dialect);
                    Some(c)
                }
                None => Some(serde_json::json!({"dialect": dialect})),
            }
        }
        None => Some(serde_json::json!({"dialect": dialect})),
    }
}

async fn models(
    State(st): State<AppState>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let created = unix_now();
    let st2 = st.clone();
    let data = tokio::task::spawn_blocking(move || {
        crate::with_sync_class(class, || models_data(&st2, created))
    })
    .await;
    match data {
        Ok(data) => Json(serde_json::json!({"object": "list", "data": data})).into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

fn models_data(st: &AppState, created: u64) -> Vec<serde_json::Value> {
        st.registry
            .all_names()
            .into_iter()
            .map(|id| {
                let loaded = st.registry.is_loaded(&id);
                let capabilities = capabilities_for(st, &id, loaded);
                ModelObject {
                    id: &id,
                    created,
                    n_ctx: Some(st.limits.max_context),
                    owned_by: &st.identity.name,
                    loaded,
                    capabilities,
                }
                .build()
            })
            .collect()
}

async fn models_retrieve(
    State(st): State<AppState>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    axum::extract::Path(model): axum::extract::Path<String>,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let loaded = st.registry.is_loaded(&model);
    if loaded || st.registry.is_known(&model) {
        let capabilities = {
            let (st, model) = (st.clone(), model.clone());
            match tokio::task::spawn_blocking(move || {
                crate::with_sync_class(class, || capabilities_for(&st, &model, loaded))
            })
            .await
            {
                Ok(c) => c,
                Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            }
        };
        let obj = ModelObject {
            id: &model,
            created: unix_now(),
            n_ctx: Some(st.limits.max_context),
            owned_by: &st.identity.name,
            loaded,
            capabilities,
        }
        .build();
        (StatusCode::OK, Json(obj)).into_response()
    } else {
        error_response(StatusCode::NOT_FOUND, format!("model '{model}' not found"))
    }
}

async fn models_delete(
    State(st): State<AppState>,
    axum::extract::Path(model): axum::extract::Path<String>,
) -> Response {
    unload_response(&st, model).await
}

async fn unload_response(st: &AppState, id: String) -> Response {
    let registry = Arc::clone(&st.registry);
    let name = id.clone();
    match tokio::task::spawn_blocking(move || registry.unload(&name)).await {
        Ok(Ok(removed)) => (
            StatusCode::OK,
            Json(serde_json::json!({"id": id, "object": "model", "deleted": removed})),
        )
            .into_response(),
        Ok(Err(e)) => daemon_error(e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Deserialize)]
struct ModelLoadRequest {
    #[serde(alias = "model", alias = "name")]
    id: String,
    #[serde(default)]
    runtime: Option<String>,
}

async fn models_load(
    State(st): State<AppState>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    body: axum::body::Bytes,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let req: ModelLoadRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    if req.id.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "missing model `id`".into());
    }
    let st2 = st.clone();
    let name = req.id.clone();
    let wanted = req.runtime.clone();
    let loaded = tokio::task::spawn_blocking(move || {
        let runtime = match wanted.as_deref() {
            None => None,
            Some(r) => Some(st2.registry.runtime_named(r).map_err(|why| {
                DaemonError::Unsupported(crate::Refusal::new("unsupported_runtime", Some("runtime"), why))
            })?),
        };
        st2.registry.load_on(&name, runtime)?;
        Ok::<_, DaemonError>(crate::with_sync_class(class, || capabilities_for(&st2, &name, true)))
    })
    .await;
    match loaded {
        Ok(Ok(capabilities)) => (
            StatusCode::OK,
            Json(
                ModelObject {
                    id: &req.id,
                    created: unix_now(),
                    n_ctx: Some(st.limits.max_context),
                    owned_by: &st.identity.name,
                    loaded: true,
                    capabilities,
                }
                .build(),
            ),
        )
            .into_response(),
        Ok(Err(e)) => daemon_error(e),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn models_unload(State(st): State<AppState>, body: axum::body::Bytes) -> Response {
    let req: ModelLoadRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    unload_response(&st, req.id).await
}

async fn health(State(st): State<AppState>) -> Json<serde_json::Value> {
    let loaded = st.registry.names().len();
    let known = st.registry.all_names().len();
    Json(serde_json::json!({"status": "ok", "models_loaded": loaded, "models_known": known}))
}

#[derive(Deserialize, Default)]
struct PropsQuery {
    model: Option<String>,
    autoload: Option<String>,
}

impl PropsQuery {
    fn autoloads(&self) -> bool {
        matches!(self.autoload.as_deref(), Some("1" | "true"))
    }
}

async fn props(
    State(st): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<PropsQuery>,
    headers: HeaderMap,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
) -> Response {
    let class = match st.sync_class(&headers, &key) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let (default_name, default_daemon) = st
        .registry
        .resolve(None)
        .expect("a default model is always loaded");
    let (name, daemon, loaded) = match query.model.as_deref().filter(|m| !m.is_empty()) {
        None => (default_name.clone(), default_daemon, true),
        Some(m) if st.registry.is_loaded(m) || (query.autoloads() && st.registry.is_known(m)) => {
            match st.resolve(Some(m)) {
                Ok((n, d)) => (n, d, true),
                Err(resp) => return resp,
            }
        }
        Some(m) if st.registry.is_known(m) => (m.to_string(), default_daemon, false),
        Some(m) => {
            return error_response_code(
                StatusCode::NOT_FOUND,
                format!("model '{m}' not found; loaded: {}", st.registry.names().join(", ")),
                "model_not_found",
            )
        }
    };
    let sampling_daemon = daemon.clone();
    let (resolved, capabilities) = if loaded {
        tokio::task::spawn_blocking(move || {
            crate::with_sync_class(class, || {
                (sampling_daemon.try_model_sampling_defaults(), sampling_daemon.capability_descriptor())
            })
        })
        .await
        .unwrap_or((None, None))
    } else {
        (None, None)
    };
    let chat_template = if loaded { daemon.chat_template() } else { None };
    let eff_repeat = (resolved.is_some() || st.sampling.repeat_penalty.is_some() || st.sampling.fallback.is_some()).then(|| {
        match repeat_penalty_for(None, &st.sampling, &resolved.clone().unwrap_or_default()) {
            0.0 => 1.0,
            v => v,
        }
    });
    let window = match daemon.max_stream_tokens() {
        n if loaded && n > 0 => u32::try_from(n).unwrap_or(u32::MAX),
        _ => st.limits.max_context,
    };
    let mut settings = serde_json::json!({ "max_context": window });
    if let Some(v) = eff_repeat {
        settings["repeat_penalty"] = v.into();
    }
    let eff = advertised_generation_defaults(&resolved.clone().unwrap_or_default(), &st.sampling.for_daemon(&daemon));
    if resolved.is_some() || st.sampling.temperature.is_some() || st.sampling.fallback.is_some() {
        settings["temperature"] = eff.temperature.into();
    }
    if resolved.is_some() || st.sampling.truncates() {
        settings["top_p"] = eff.top_p.into();
        settings["top_k"] = eff.top_k.into();
        settings["min_p"] = eff.min_p.into();
    }
    let advertised = st.limits.default_max_tokens.unwrap_or(st.limits.max_context);
    settings["n_predict"] = advertised.into();
    settings["max_tokens"] = advertised.into();
    settings["n_ctx"] = window.into();
    let mut body = serde_json::json!({
        "model": name,
        "default_model": default_name,
        "models": st.registry.names(),
        "max_context": window,
        "total_slots": 1,
        "default_generation_settings": settings,
        "build": {"version": st.identity.version, "target": engine_target()},
        "server": st.identity.name,
        "max_request_bytes": MAX_BODY_SIZE,
        "capabilities": capabilities.unwrap_or(serde_json::Value::Null),
    });
    body["loaded"] = loaded.into();
    if let Some(template) = chat_template {
        body["chat_template"] = template.into();
    }
    if loaded {
        let bits = daemon.sched_stats().kv_bits.load(std::sync::atomic::Ordering::Relaxed);
        if let Some(label) = crate::scheduler::kv_bits_label(bits) {
            body["kv_cache"] = serde_json::json!({ "bits": bits, "dtype": label });
        }
    }
    Json(body).into_response()
}

fn engine_target() -> &'static str {
    if cfg!(target_os = "macos") {
        "metal"
    } else {
        "cuda"
    }
}

async fn slots(State(st): State<AppState>) -> Json<serde_json::Value> {
    use std::sync::atomic::Ordering;
    let (name, daemon) = st.registry.resolve(None).expect("a default model is always loaded");
    let s = daemon.sched_stats();
    let active = s.lanes_active.load(Ordering::Relaxed);
    let tick = s.tick_in_flight().unwrap_or_default();
    let in_flight_ms = if tick.since_ms == 0 {
        0
    } else {
        crate::scheduler::uptime_ms().saturating_sub(tick.since_ms)
    };
    Json(serde_json::json!([{
        "id": 0,
        "model": name,
        "is_processing": active > 0,
        "cache_position": s.pool_blocks_used.load(Ordering::Relaxed),
        "reuse_snapshots": 0,
        "last_access": unix_now(),
        "idle_seconds": 0,
        "path": name,
        "lanes_active": active,
        "kv_pool_used": s.pool_blocks_used.load(Ordering::Relaxed),
        "kv_pool_total": s.pool_blocks_total.load(Ordering::Relaxed),
        "tick_wall_ms": s.tick_wall_ms_live.load(Ordering::Relaxed),
        "tick_in_flight_ms": in_flight_ms,
        "tick_plan_prefill_tokens": tick.prefill_tokens,
        "tick_plan_prefill_lanes": tick.prefill_lanes,
        "tick_plan_decode_lanes": tick.decode_lanes,
        "tick_plan_admits": tick.admits,
        "tick_plan_retires": tick.retires,
        "decode_tok_s": s.decode_rate_live.load(Ordering::Relaxed),
        "prefill_tok_s": s.prefill_rate_live.load(Ordering::Relaxed),
        "ttft_last_ms": s.ttft_last_ms.load(Ordering::Relaxed),
    }]))
}

#[derive(Deserialize)]
struct TokenizeRequest {
    #[serde(default)]
    text: String,
    #[serde(default)]
    content: Option<String>,
}

async fn tokenize(State(st): State<AppState>, body: axum::body::Bytes) -> Response {
    let req: TokenizeRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    let text = req.content.unwrap_or(req.text);
    let tokens = st.default_daemon().tokenize(&text);
    let count = tokens.len();
    (StatusCode::OK, Json(serde_json::json!({"tokens": tokens, "count": count}))).into_response()
}

pub(crate) fn tool_call_delta(d: &crate::tool_fragment::ToolDelta, index: usize, id: &str) -> serde_json::Value {
    crate::openai_output::message_json(&Message::tool(d, index, id))
}

async fn chat_completions(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let req: ChatRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("invalid request: {e}")),
    };
    let keepalive = st.nonstream_keepalive.filter(|_| !req.stream);
    crate::keepalive::respond(keepalive, move |hooks| async move {
        match crate::http_generation::chat(State(st), key, headers, req, hooks).await {
            Ok(generation) => generation_response(generation),
            Err(response) => response,
        }
    })
    .await
}

#[cfg(test)]
mod input_modality_tests {
    use super::*;

    fn caps(modalities: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"descriptor_version": 1, "modalities": modalities})
    }

    #[test]
    fn text_is_the_floor_without_a_descriptor() {
        assert_eq!(input_modalities(None), vec!["text"]);
    }

    #[test]
    fn a_descriptor_without_modalities_is_text_only() {
        let c = serde_json::json!({"descriptor_version": 1});
        assert_eq!(input_modalities(Some(&c)), vec!["text"]);
    }

    #[test]
    fn a_vision_bundle_advertises_image() {
        let c = caps(serde_json::json!({"image_encode": true}));
        assert_eq!(input_modalities(Some(&c)), vec!["text", "image"]);
    }

    #[test]
    fn a_refusal_reason_is_not_a_yes() {
        let c = caps(serde_json::json!({
            "image_encode": "no vision tower in bundle",
            "gemma_audio_encode": "the Conformer tower's kernels are not ported to this backend",
        }));
        assert_eq!(input_modalities(Some(&c)), vec!["text"]);
    }

    #[test]
    fn a_gemma_audio_bundle_advertises_audio() {
        let c = caps(serde_json::json!({"image_encode": true, "gemma_audio_encode": true}));
        assert_eq!(input_modalities(Some(&c)), vec!["text", "image", "audio"]);
    }

    #[test]
    fn whisper_never_advertises_audio() {
        let c = caps(serde_json::json!({
            "image_encode": "no vision tower in bundle",
            "gemma_audio_encode": "no audio tower in bundle",
            "whisper_transcribe": true,
            "whisper_translate": true,
        }));
        assert_eq!(input_modalities(Some(&c)), vec!["text"]);
    }

    #[test]
    fn n_ctx_is_omitted_rather_than_guessed() {
        let v = ModelObject {
            id: "m",
            created: 0,
            n_ctx: None,
            owned_by: "superfluid-fleet",
            loaded: true,
            capabilities: None,
        }
        .build();
        assert!(v["meta"].is_object(), "meta stays present: {v}");
        assert!(v["meta"]["n_ctx"].is_null(), "unknown n_ctx is absent: {v}");
        assert_eq!(v["owned_by"], "superfluid-fleet");
    }
}

#[cfg(test)]
mod browser_guard_tests {
    use super::{browser_refusal, HostGuard};
    use axum::http::HeaderMap;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, v.parse().unwrap());
        }
        h
    }

    fn this_machine() -> HostGuard {
        HostGuard::ThisMachine(vec!["box.lan".to_string(), "box.local".to_string()].into())
    }

    #[test]
    fn a_page_from_another_origin_is_refused_and_a_local_one_is_not() {
        for origin in ["https://evil.example", "null", "http://192.168.1.5:3000", "http://127.0.0.1.evil.example"] {
            assert!(browser_refusal(&headers(&[("origin", origin)]), &HostGuard::Any).is_some(), "{origin}");
        }
        for origin in ["http://localhost:3000", "http://127.0.0.1:8453", "http://[::1]:8080", "http://app.localhost"] {
            let h = headers(&[("origin", origin), ("host", "127.0.0.1:8453")]);
            assert!(browser_refusal(&h, &HostGuard::Loopback).is_none(), "{origin}");
        }
        assert!(browser_refusal(&headers(&[("host", "127.0.0.1:8453")]), &HostGuard::Loopback).is_none(), "curl sends no Origin");
    }

    #[test]
    fn a_loopback_server_answers_only_to_a_loopback_host() {
        for host in ["127.0.0.1:8453", "localhost:8453", "localhost", "[::1]:8453", "127.0.0.2"] {
            assert!(browser_refusal(&headers(&[("host", host)]), &HostGuard::Loopback).is_none(), "{host}");
        }
        for host in ["rebound.evil.example:8453", "10.0.0.4:8453", "[2001:db8::1]:8453"] {
            assert!(browser_refusal(&headers(&[("host", host)]), &HostGuard::Loopback).is_some(), "{host}");
            assert!(browser_refusal(&headers(&[("host", host)]), &HostGuard::Any).is_none(), "{host}: keyed, off loopback");
        }
    }

    #[test]
    fn a_keyless_server_off_loopback_answers_only_to_an_ip_or_its_own_name() {
        for host in [
            "10.0.0.4:8453", "192.168.1.5", "[2001:db8::1]:8453", "127.0.0.1:8453", "localhost:8453",
            "app.localhost", "box.lan:8453", "BOX.LOCAL.", "box.local:8453",
        ] {
            assert!(browser_refusal(&headers(&[("host", host)]), &this_machine()).is_none(), "{host}");
        }
        for host in ["rebound.evil.example:8453", "box.lan.evil.example", "box", "10.0.0.4.nip.io:8453"] {
            let why = browser_refusal(&headers(&[("host", host)]), &this_machine());
            assert!(why.as_deref().is_some_and(|w| w.contains("--api-key")), "{host}: {why:?}");
        }
    }

    #[test]
    fn this_machines_names_include_its_mdns_name() {
        let HostGuard::ThisMachine(names) =
            HostGuard::for_listener(&super::ServeConfig::default(), false)
        else {
            panic!("a keyless server off loopback guards its Host")
        };
        assert!(names.iter().any(|n| n.ends_with(".local")), "{names:?}");
        let keyed = super::ServeConfig { api_key: Some("k".into()), ..Default::default() };
        assert!(matches!(HostGuard::for_listener(&keyed, false), HostGuard::Any));
        assert!(matches!(HostGuard::for_listener(&keyed, true), HostGuard::Loopback));
    }
}

#[cfg(test)]
mod audio_staging_tests {
    use super::{audio_extension, stage_audio};
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn the_client_extension_is_kept_only_when_it_is_plain() {
        assert_eq!(audio_extension("clip.mp3"), Some("mp3"));
        assert_eq!(audio_extension("clip"), Some("wav"));
        assert_eq!(audio_extension("a.b.flac"), Some("flac"));
        for bad in ["clip.wav;rm -rf", "clip.", "clip.toolongext", "clip.w\u{e9}v", "x.a b"] {
            assert_eq!(audio_extension(bad), None, "{bad}");
        }
    }

    #[test]
    fn staged_audio_is_owner_only_and_never_reuses_a_path() {
        let dir = std::env::temp_dir().join(format!("superfluid-audio-stage-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = stage_audio(&dir, "wav", b"RIFF").unwrap();
        let b = stage_audio(&dir, "wav", b"RIFF").unwrap();
        assert_ne!(a, b);
        assert_eq!(std::fs::read(&a).unwrap(), b"RIFF");
        let mode = std::fs::metadata(&a).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "{mode:o}");
        let name = a.file_name().unwrap().to_string_lossy().into_owned();
        assert!(!name.contains(&std::process::id().to_string()), "{name}: no pid in the name");
        let _ = std::fs::remove_dir_all(dir);
    }
}
