//! Common HTTP request admission for the generation adapters.

use crate::http_state::*;
use crate::inference::{self, ChatRequest, CompletionRequest, Generation, RequestPolicy, RunError};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::Response,
};

pub(crate) fn run_error(error: RunError) -> Response {
    match error {
        RunError::Daemon(e) => daemon_error(e),
        RunError::Worker(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// OpenAI bounds `presence_penalty` and `frequency_penalty` to [-2, 2]. A
/// value past them is a client putting a probability or a repeat penalty in
/// the wrong field, and an embedding that reads a lone frequency penalty as
/// a repeat penalty (1 + value) would make a non-positive multiplier of it,
/// which turns the penalised logits around instead of damping them.
#[allow(clippy::result_large_err)]
pub(crate) fn penalties_in_range(presence: Option<f32>, frequency: Option<f32>) -> Result<(), Response> {
    for (name, value) in [("presence_penalty", presence), ("frequency_penalty", frequency)] {
        if value.is_some_and(|v| !(-2.0..=2.0).contains(&v)) {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                format!("{name} must be in [-2, 2]"),
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(crate) async fn completions(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    cred: Option<axum::Extension<AuthenticatedCredential>>,
    headers: HeaderMap,
    req: CompletionRequest,
    hooks: crate::keepalive::Hooks,
) -> Result<Generation, Response> {
    if req.model.is_none() && st.requires_model() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "missing required field: model".into(),
        ));
    }
    penalties_in_range(req.presence_penalty, req.frequency_penalty)?;
    if req.suffix.is_some() {
        let mode = match fim_mode_of(&req) {
            Ok(m) => m,
            Err(resp) => return Err(resp),
        };
        let client = crate::completion_bucket::http_client_key(
            peer.ip(),
            cred.as_ref().map(|axum::Extension(c)| &*c.0),
        );
        let front = st.default_daemon();
        if let Err(wait) = st.fim_buckets.try_take(client, front.completion_bucket()) {
            front.note_completion_throttled();
            return Err(throttled_response(wait));
        }
        let (model_name, daemon) = match st.resolve(req.model.as_deref()) {
            Ok(pair) => pair,
            Err(resp) => return Err(resp),
        };
        return inference::fim_completions(
            daemon,
            model_name,
            req,
            st.limits,
            st.sampling,
            mode,
            key.map(|k| k.0),
        )
        .await
        .map_err(run_error);
    }
    if req.fim_mode.is_some() {
        return Err(error_response_code(
            StatusCode::BAD_REQUEST,
            "fim_mode is only meaningful with suffix".into(),
            "unsupported_parameter",
        ));
    }
    if req.prompt.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "prompt must be non-empty".into(),
        ));
    }
    let qos = match st.qos.resolve(&headers, key.as_ref().map(|k| &*k.0 .0)) {
        Ok(q) => q,
        Err(resp) => return Err(resp),
    };
    let (model_name, daemon) = match st.resolve(req.model.as_deref()) {
        Ok(pair) => pair,
        Err(resp) => return Err(resp),
    };
    inference::completions(
        daemon,
        model_name,
        req,
        RequestPolicy {
            limits: st.limits,
            sampling: st.sampling,
            qos,
        },
        key.map(|k| k.0),
        hooks,
    )
    .await
    .map_err(run_error)
}
#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(crate) async fn chat(
    State(st): State<AppState>,
    key: Option<axum::Extension<crate::keypolicy::ClientKey>>,
    headers: HeaderMap,
    req: ChatRequest,
    hooks: crate::keepalive::Hooks,
) -> Result<Generation, Response> {
    if req.model.is_none() && st.requires_model() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "missing required field: model".into(),
        ));
    }
    if req.messages.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "messages must be non-empty".into(),
        ));
    }
    penalties_in_range(req.presence_penalty, req.frequency_penalty)?;
    let qos = match st.qos.resolve(&headers, key.as_ref().map(|k| &*k.0 .0)) {
        Ok(q) => q,
        Err(resp) => return Err(resp),
    };
    let (model_name, daemon) = match st.resolve(req.model.as_deref()) {
        Ok(pair) => pair,
        Err(resp) => return Err(resp),
    };
    if !req.stream && req.n.unwrap_or(1).max(1) > 128 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "n must be between 1 and 128".into(),
        ));
    }
    inference::chat(
        daemon,
        model_name,
        req,
        RequestPolicy {
            limits: st.limits,
            sampling: st.sampling,
            qos,
        },
        key.map(|k| k.0),
        hooks,
    )
    .await
    .map_err(run_error)
}
#[allow(clippy::result_large_err)]
fn fim_mode_of(req: &CompletionRequest) -> Result<u8, Response> {
    let unsupported = |what: &str| {
        error_response_code(
            StatusCode::BAD_REQUEST,
            format!("{what} is not supported with suffix (fill-in-middle completions)"),
            "unsupported_parameter",
        )
    };
    if req.echo {
        return Err(unsupported("echo"));
    }
    if req.logprobs.is_some() {
        return Err(unsupported("logprobs"));
    }
    match req
        .fim_mode
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("psm") => Ok(crate::codec::fim_mode::PSM),
        Some("spm") => Ok(crate::codec::fim_mode::SPM),
        Some(other) => Err(error_response_code(
            StatusCode::BAD_REQUEST,
            format!("fim_mode must be \"psm\" or \"spm\", got {other:?}"),
            "invalid_value",
        )),
    }
}
