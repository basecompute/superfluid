//! Shared HTTP admission state, scheduling policy and error responses.

use crate::inference::{RequestQos, SamplingOverrides, TokenLimits};
use crate::{Daemon, DaemonError};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) registry: Arc<crate::registry::ModelRegistry>,
    pub(crate) limits: TokenLimits,
    pub(crate) sampling: SamplingOverrides,
    pub(crate) qos: HttpQosPolicy,
    pub(crate) keys: Option<Arc<crate::keypolicy::KeyTable>>,
    pub(crate) fim_buckets: Arc<crate::completion_bucket::ClientBuckets>,
    pub(crate) nonstream_keepalive: Option<std::time::Duration>,
    pub(crate) model_naming: crate::openai::ModelNaming,
    pub(crate) identity: Arc<crate::openai::ServerIdentity>,
}

impl AppState {
    /// Whether a request that names no model is refused.
    pub(crate) fn requires_model(&self) -> bool {
        self.model_naming == crate::openai::ModelNaming::Exact
    }
    pub(crate) fn default_daemon(&self) -> Arc<Daemon> {
        self.registry
            .resolve(None)
            .map(|(_, d)| d)
            .expect("a default model is always loaded")
    }
    pub(crate) fn default_name(&self) -> String {
        self.registry.default_name()
    }
    #[allow(clippy::result_large_err)]
    pub(crate) fn sync_class(
        &self,
        headers: &HeaderMap,
        key: &Option<axum::Extension<crate::keypolicy::ClientKey>>,
    ) -> Result<Option<u8>, Response> {
        self.qos
            .resolve(headers, key.as_ref().map(|k| &*k.0 .0))
            .map(|q| Some(q.class))
    }
    #[allow(clippy::result_large_err)]
    pub(crate) fn resolve(&self, model: Option<&str>) -> Result<(String, Arc<Daemon>), Response> {
        let loose = match model {
            Some(m) if self.model_naming == crate::openai::ModelNaming::Loose => self.registry.loose_name(m),
            _ => None,
        };
        let model = loose.as_deref().or(model);
        let needs_load = model.is_some_and(|m| !self.registry.is_loaded(m) && self.registry.is_known(m))
            || self.registry.reload_pending(model);
        let many_threads = || {
            tokio::runtime::Handle::try_current().is_ok_and(|h| h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread)
        };
        let hit = if needs_load && many_threads() {
            tokio::task::block_in_place(|| self.registry.try_resolve(model))
        } else {
            self.registry.try_resolve(model)
        };
        hit.map_err(|e| match e {
            crate::registry::ResolveError::Load(e) => load_failed(model.unwrap_or(""), e),
            crate::registry::ResolveError::NotFound => error_response(
                StatusCode::NOT_FOUND,
                format!(
                    "model '{}' not found; loaded: {}",
                    model.unwrap_or(""),
                    self.registry.names().join(", ")
                ),
            ),
        })
    }
}

pub const QOS_HEADER: &str = "x-superfluid-qos";
pub const BATCH_INVARIANT_HEADER: &str = "x-superfluid-batch-invariant";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpQosPolicy {
    pub default: RequestQos,
    pub honor_headers: bool,
    pub allow_batch_invariant: bool,
}

impl Default for HttpQosPolicy {
    fn default() -> HttpQosPolicy {
        HttpQosPolicy {
            default: RequestQos::AGENT,
            honor_headers: true,
            allow_batch_invariant: false,
        }
    }
}

impl HttpQosPolicy {
    #[allow(clippy::result_large_err)]
    pub(crate) fn resolve(
        &self,
        headers: &HeaderMap,
        key: Option<&crate::keypolicy::KeyPolicy>,
    ) -> Result<RequestQos, Response> {
        if let Some(key) = key {
            let q = HttpQosPolicy {
                default: RequestQos { class: key.class, batch_invariant: false },
                honor_headers: true,
                allow_batch_invariant: key.allow_batch_invariant,
            }
            .resolve(headers, None)
            .map_err(|r| {
                if r.status() != StatusCode::FORBIDDEN {
                    return r;
                }
                key.note_qos_reject();
                error_response(
                    StatusCode::FORBIDDEN,
                    format!(
                        "API key '{}' may not request a batch-invariant lane ({BATCH_INVARIANT_HEADER}); \
                         grant it with \"batch_invariant\": true on this key in the --key-policy file",
                        key.name
                    ),
                )
            })?;
            if q.class < key.max_class {
                key.note_qos_reject();
                return Err(error_response(
                    StatusCode::FORBIDDEN,
                    format!(
                        "API key '{}' may request at most the {} class ({QOS_HEADER}: {} asked for more)",
                        key.name,
                        crate::qos::NAMES[key.max_class as usize],
                        crate::qos::NAMES[q.class as usize],
                    ),
                ));
            }
            return Ok(q);
        }
        let mut q = self.default;
        if !self.honor_headers {
            return Ok(q);
        }
        if let Some(v) = headers.get(QOS_HEADER) {
            q.class = v.to_str().ok().and_then(crate::qos::parse).ok_or_else(|| {
                error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "invalid {QOS_HEADER} header {:?}: expected one of {}",
                        String::from_utf8_lossy(v.as_bytes()),
                        crate::qos::NAMES.join(", ")
                    ),
                )
            })?;
        }
        if let Some(v) = headers.get(BATCH_INVARIANT_HEADER) {
            q.batch_invariant = match v.to_str().map(|s| s.trim().to_ascii_lowercase()).as_deref() {
                Ok("1" | "true" | "yes" | "on") => true,
                Ok("0" | "false" | "no" | "off") => false,
                _ => {
                    return Err(error_response(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "invalid {BATCH_INVARIANT_HEADER} header {:?}: expected true or false",
                            String::from_utf8_lossy(v.as_bytes())
                        ),
                    ))
                }
            };
            if q.batch_invariant && !self.allow_batch_invariant {
                return Err(error_response(
                    StatusCode::FORBIDDEN,
                    format!(
                        "{BATCH_INVARIANT_HEADER} is disabled on this server: a batch-invariant lane \
                         runs alone and holds back every other request (start superfluid with \
                         --http-allow-batch-invariant to permit it)"
                    ),
                ));
            }
        }
        Ok(q)
    }
}

#[derive(Clone)]
pub(crate) struct AuthenticatedCredential(pub(crate) Arc<str>);

impl AuthenticatedCredential {
    pub(crate) fn global_key(key: &str) -> AuthenticatedCredential {
        AuthenticatedCredential(Arc::from(format!("api-key:{key}")))
    }

    pub(crate) fn policy_key(name: &str) -> AuthenticatedCredential {
        AuthenticatedCredential(Arc::from(format!("policy:{name}")))
    }
}

pub(crate) fn refusal_response(status: StatusCode, r: &crate::Refusal) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": {"message": r.message, "type": "invalid_request_error", "param": r.param, "code": r.code}
        })),
    )
        .into_response()
}

pub(crate) fn error_response(status: StatusCode, message: String) -> Response {
    error_response_with_code(status, message, serde_json::Value::Null)
}

pub(crate) fn error_response_code(status: StatusCode, message: String, code: &str) -> Response {
    error_response_with_code(status, message, serde_json::Value::String(code.into()))
}

pub(crate) fn error_response_typed(
    status: StatusCode,
    message: String,
    kind: &str,
    code: &str,
) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": {"message": message, "type": kind, "param": null, "code": code}
        })),
    )
        .into_response()
}

fn error_response_with_code(
    status: StatusCode,
    message: String,
    code: serde_json::Value,
) -> Response {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    (
        status,
        Json(serde_json::json!({
            "error": {"message": message, "type": kind, "param": null, "code": code}
        })),
    )
        .into_response()
}

fn load_failed(model: &str, e: DaemonError) -> Response {
    use DaemonError as E;
    let status = match e {
        E::Config(_) | E::Constraint(_) | E::TemplateRefused(_) | E::Unsupported(_) => StatusCode::BAD_REQUEST,
        E::Io(_)
        | E::Codec(_)
        | E::WalCorrupt
        | E::EngineLoad(_)
        | E::Agent(_)
        | E::Fleet(_)
        | E::FleetNode { .. }
        | E::Rejected(_)
        | E::Generation(_)
        | E::Protocol(_)
        | E::CompletionThrottled { .. } => StatusCode::SERVICE_UNAVAILABLE,
        E::UnknownSession(_)
        | E::EmptySession(_)
        | E::StreamTooLong { .. }
        | E::SessionBusy(_)
        | E::UnknownToolCall(_)
        | E::UnknownRole(_)
        | E::InvalidTool { .. }
        | E::SystemNotFirst(_)
        | E::ForkPoint { .. }
        | E::InheritedToolCall(_)
        | E::MetaConflict { .. }
        | E::GenerationConflict { .. }
        | E::Purged(_)
        | E::FilesQuota { .. }
        | E::RebaseSpec(_)
        | E::PurgeTimeout(_)
        | E::UnknownPermission(_)
        | E::PermissionPending(_)
        | E::ToolLeaseExpired(_)
        | E::ToolLateEffect(_)
        | E::UnknownBlockKind(_)
        | E::UnknownMedia(_)
        | E::MediaTooLarge(_)
        | E::NoFimDialect => StatusCode::SERVICE_UNAVAILABLE,
    };
    let message = format!("model '{model}' could not be loaded: {e}");
    if let E::Unsupported(r) = &e {
        return refusal_response(status, &crate::Refusal { message, ..r.clone() });
    }
    error_response(status, message)
}

pub(crate) fn sse_error_frame(e: &DaemonError, kind: &str) -> serde_json::Value {
    let mut err = serde_json::json!({"message": e.to_string(), "type": kind});
    if matches!(e, DaemonError::StreamTooLong { .. }) {
        err["code"] = "context_length_exceeded".into();
    }
    if let DaemonError::Unsupported(r) = e {
        err["code"] = r.code.into();
        err["param"] = serde_json::json!(r.param);
    }
    serde_json::json!({"error": err})
}

pub(crate) fn daemon_error(e: DaemonError) -> Response {
    if matches!(e, DaemonError::StreamTooLong { .. }) {
        return error_response_typed(
            StatusCode::BAD_REQUEST,
            e.to_string(),
            "invalid_request_error",
            "context_length_exceeded",
        );
    }
    if matches!(e, DaemonError::NoFimDialect) {
        return error_response_code(
            StatusCode::BAD_REQUEST,
            "this model has no fill-in-middle dialect and no --fim-model satellite is configured"
                .into(),
            "fim_unsupported",
        );
    }
    if let DaemonError::CompletionThrottled { retry_after_ms } = e {
        return throttled_response(std::time::Duration::from_millis(retry_after_ms));
    }
    if let DaemonError::Unsupported(r) = &e {
        return refusal_response(StatusCode::BAD_REQUEST, r);
    }
    let status = match &e {
        DaemonError::UnknownSession(_)
        | DaemonError::EmptySession(_)
        | DaemonError::UnknownToolCall(_)
        | DaemonError::SessionBusy(_)
        | DaemonError::Protocol(_)
        | DaemonError::Constraint(_)
        | DaemonError::TemplateRefused(_)
        | DaemonError::Unsupported(_)
        | DaemonError::Config(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error_response(status, e.to_string())
}

pub(crate) fn throttled_response(wait: std::time::Duration) -> Response {
    let secs = crate::completion_bucket::retry_after_secs(wait);
    let retry_after_ms = crate::completion_bucket::retry_after_ms(wait);
    let mut resp = error_response_typed(
        StatusCode::TOO_MANY_REQUESTS,
        format!("completion rate limit exceeded; retry in {retry_after_ms} ms"),
        "rate_limit_error",
        "completion_rate_limited",
    );
    if let Ok(v) = axum::http::HeaderValue::from_str(&secs.to_string()) {
        resp.headers_mut()
            .insert(axum::http::header::RETRY_AFTER, v);
    }
    resp
}

#[cfg(test)]
mod sse_error_frame_tests {
    use super::sse_error_frame;
    use crate::generation_output::StreamFailure;
    use crate::{DaemonError, Refusal};

    #[test]
    fn the_frame_carries_the_code_a_client_branches_on() {
        let long = sse_error_frame(&DaemonError::StreamTooLong { len: 9000, max: 8192 }, "invalid_request_error");
        assert_eq!(long["error"]["code"], "context_length_exceeded", "{long}");
        assert_eq!(long["error"]["type"], "invalid_request_error", "{long}");
        assert!(long["error"].get("param").is_none(), "{long}");

        let refusal = Refusal::new("unsupported_input", Some("messages[0].content"), "model 'm' on runtime r does not accept images: no media path");
        let refused = sse_error_frame(&DaemonError::Unsupported(refusal), "invalid_request_error");
        assert_eq!(refused["error"]["code"], "unsupported_input", "{refused}");
        assert_eq!(refused["error"]["param"], "messages[0].content", "{refused}");
        assert_eq!(refused["error"]["message"], "model 'm' on runtime r does not accept images: no media path", "{refused}");

        let other = sse_error_frame(&DaemonError::Protocol("sse client gone"), "server_error");
        assert!(other["error"].get("code").is_none() && other["error"]["type"] == "server_error", "{other}");
    }

    #[test]
    fn a_stream_failure_carries_them_too() {
        let long = StreamFailure::new(DaemonError::StreamTooLong { len: 9000, max: 8192 }, true);
        assert_eq!((long.code, long.param.as_deref()), (Some("context_length_exceeded"), None));

        let refusal = Refusal::new("unsupported_input", Some("messages[0].content"), "model 'm' on runtime r does not accept images: no media path");
        let refused = StreamFailure::new(DaemonError::Unsupported(refusal), true);
        assert_eq!((refused.code, refused.param.as_deref()), (Some("unsupported_input"), Some("messages[0].content")));
        assert_eq!(refused.message, "model 'm' on runtime r does not accept images: no media path");

        let other = StreamFailure::new(DaemonError::Protocol("sse client gone"), false);
        assert_eq!((other.code, other.param), (None, None));
    }
}
