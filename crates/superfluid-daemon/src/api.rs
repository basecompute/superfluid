//! api-native.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::bus::{BusMsg, EndReason};
use crate::store::CommittedEvent;
use crate::wal::{EventBody, GenParams};
use crate::{Daemon, DaemonError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventMsg {
    pub event_id: u64,
    pub epoch: u64,
    pub ts_unix_ms: u64,
    pub body: EventBody,
}

impl From<CommittedEvent> for EventMsg {
    fn from(e: CommittedEvent) -> EventMsg {
        EventMsg {
            event_id: e.event_id,
            epoch: e.epoch,
            ts_unix_ms: e.ts_unix_ms,
            body: e.body,
        }
    }
}

// postcard encodes a variant as its index on the socket, so new variants of Request and
// Response go at the end.
#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    Create {
        parent: Option<u64>,
        params: GenParams,
    },
    Append {
        session: u64,
        text: Option<String>,
        span: Vec<u32>,
    },
    AppendMessage {
        session: u64,
        role: u32,
        text: String,
    },
    Generate {
        session: u64,
        max_tokens: u32,
    },
    Read {
        session: u64,
        cursor: u64,
    },
    AppendToolResult {
        session: u64,
        call_id: u64,
        content: String,
    },
    OpenToolCalls {
        session: u64,
    },
    List,
    Cancel {
        session: u64,
    },
    Subscribe {
        session: u64,
        cursor: u64,
        provisional: bool,
    },
    Unsubscribe {
        sub: u64,
    },
    Fork {
        session: u64,
        at_event: u64,
        params: Option<GenParams>,
    },
    Tree,
    Sessions {
        include_archived: bool,
    },
    SetMeta {
        session: u64,
        expected_version: u64,
        title: Option<String>,
        archived: Option<bool>,
    },
    Rebase {
        session: u64,
        edits: Vec<crate::wal::RebaseEdit>,
        params: Option<GenParams>,
    },
    Purge {
        session: u64,
        expected_generation: u64,
        mode: crate::wal::PurgeMode,
    },
    SetQos {
        session: u64,
        class: u8,
        batch_invariant: bool,
    },
    SetBackgroundRate {
        divisor: u32,
    },
    Inspect {
        session: u64,
    },
    SetLogLevel {
        directive: String,
    },
    Metrics,
    AppendBlock {
        session: u64,
        role: u32,
        kind: u32,
        payload: String,
    },
    RequestPermission {
        session: u64,
        call_id: u64,
        text: String,
    },
    RespondPermission {
        session: u64,
        request_id: u64,
        granted: bool,
    },
    RequestToolCancel {
        session: u64,
        call_id: u64,
    },
    AppendToolOutcome {
        session: u64,
        call_id: u64,
        outcome: u8,
        note: String,
    },
    Ledger {
        session: u64,
    },
    PutMedia {
        bytes: Vec<u8>,
        mime: String,
    },
    AppendImage {
        session: u64,
        role: u32,
        blob: String,
        pre_text: String,
        post_text: String,
    },
    GcMedia,
    Complete {
        prefix: String,
        suffix: String,
        mode: u8,
        max_tokens: u32,
    },
    Trim {
        session: u64,
        to_event: u64,
    },
    Pin {
        session: u64,
        ttl_ms: u64,
    },
    AppendSystem {
        session: u64,
        text: Option<String>,
        tools: Vec<String>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    Created {
        session: u64,
    },
    Delta {
        event: EventMsg,
    },
    Cancelling,
    Committed {
        event: EventMsg,
    },
    Generated {
        events: Vec<EventMsg>,
        tokens_generated: u32,
        finish: u32,
        warm_prefix: u64,
    },
    Events {
        events: Vec<EventMsg>,
    },
    Sessions {
        ids: Vec<u64>,
    },
    ToolCalls {
        calls: Vec<(u64, String, String)>,
    },
    Subscribed {
        sub: u64,
    },
    SubEvent {
        sub: u64,
        event: EventMsg,
    },
    SubProvisional {
        sub: u64,
        channel: u32,
        text: String,
    },
    SubEnded {
        sub: u64,
        reason: String,
    },
    Unsubscribed,
    Err {
        message: String,
    },
    Tree {
        nodes: Vec<(u64, Option<u64>, u64)>,
    },
    SessionList {
        sessions: Vec<crate::SessionSummary>,
    },
    Meta {
        version: u64,
    },
    Purged {
        sessions: Vec<u64>,
    },
    Ack,
    Inspection {
        inspection: crate::Inspection,
    },
    Text {
        text: String,
    },
    Ledger {
        entries: Vec<(u64, crate::store::LedgerEntry)>,
    },
    Media {
        hash: String,
    },
    Gc {
        kept: u64,
        removed: u64,
    },
    Completion {
        text: String,
        tokens: Vec<u32>,
        cached: bool,
        expired: bool,
    },
    Pinned {
        deadline_unix_ms: u64,
    },
    Throttled {
        retry_after_ms: u64,
    },
}

fn write_frame<T: Serialize>(w: &mut impl Write, msg: &T) -> Result<(), DaemonError> {
    let payload = postcard::to_stdvec(msg)?;
    w.write_all(&(payload.len() as u32).to_le_bytes())?;
    w.write_all(&payload)?;
    Ok(())
}

fn read_frame<T: for<'de> Deserialize<'de>>(r: &mut impl Read) -> Result<Option<T>, DaemonError> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    const MAX_FRAME: usize = 64 << 20;
    if len > MAX_FRAME {
        return Err(DaemonError::Protocol("api frame exceeds limit"));
    }
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    Ok(Some(postcard::from_bytes(&payload)?))
}

pub fn handle_web(d: &Daemon, req: Request) -> Response {
    match req {
        Request::Cancel { session } => cancel(&d.cancel_registry(), &d.active_registry(), session),
        Request::Generate { .. }
        | Request::Subscribe { .. }
        | Request::Unsubscribe { .. } => Response::Err {
            message: "streaming verbs use the WebSocket surface (/web/ws) or SSE (/web/stream), \
                      not /web/rpc"
                .into(),
        },
        other => handle(d, other),
    }
}

fn cancel(cancels: &crate::CancelSet, active: &Mutex<HashSet<u64>>, session: u64) -> Response {
    let mut c = cancels.lock().expect("cancel registry");
    if active.lock().expect("active set").contains(&session) {
        c.insert(session);
        // The running tick ends at its next step, not when it would have.
        cancels.wake();
        Response::Cancelling
    } else {
        Response::Err {
            message: format!("session {session} has no generation in flight"),
        }
    }
}

fn handle(d: &Daemon, req: Request) -> Response {
    let result: Result<Response, DaemonError> = (|| match req {
        Request::Create { parent, params } => {
            let session = d.create(parent, params)?;
            Ok(Response::Created { session })
        }
        Request::Append {
            session,
            text,
            span,
        } => {
            let event = d.append(session, text, span)?;
            Ok(Response::Committed {
                event: event.into(),
            })
        }
        Request::AppendMessage {
            session,
            role,
            text,
        } => {
            let event = d.append_message(session, role, text)?;
            Ok(Response::Committed {
                event: event.into(),
            })
        }
        Request::Read { session, cursor } => {
            let events = d.read(session, cursor)?;
            Ok(Response::Events {
                events: events.into_iter().map(Into::into).collect(),
            })
        }
        Request::AppendToolResult {
            session,
            call_id,
            content,
        } => {
            let event = d.append_tool_result(session, call_id, content)?;
            Ok(Response::Committed {
                event: event.into(),
            })
        }
        Request::OpenToolCalls { session } => Ok(Response::ToolCalls {
            calls: d.open_tool_calls(session)?,
        }),
        Request::List => Ok(Response::Sessions {
            ids: d.session_ids(),
        }),
        Request::Fork {
            session,
            at_event,
            params,
        } => {
            let child = d.fork(session, at_event, params)?;
            Ok(Response::Created { session: child })
        }
        Request::Tree => Ok(Response::Tree { nodes: d.tree() }),
        Request::Sessions { include_archived } => Ok(Response::SessionList {
            sessions: d.sessions(include_archived),
        }),
        Request::SetMeta {
            session,
            expected_version,
            title,
            archived,
        } => Ok(Response::Meta {
            version: d.set_meta(session, expected_version, title, archived)?,
        }),
        Request::Rebase {
            session,
            edits,
            params,
        } => {
            let child = d.rebase(session, edits, params)?;
            Ok(Response::Created { session: child })
        }
        Request::Purge {
            session,
            expected_generation,
            mode,
        } => Ok(Response::Purged {
            sessions: d.purge(session, expected_generation, mode)?,
        }),
        Request::SetQos {
            session,
            class,
            batch_invariant,
        } => Ok(Response::Committed {
            event: d.set_qos(session, class, batch_invariant)?.into(),
        }),
        Request::SetBackgroundRate { divisor } => {
            d.set_background_tick_divisor(divisor);
            Ok(Response::Ack)
        }
        Request::Inspect { session } => Ok(Response::Inspection {
            inspection: d.inspect(session)?,
        }),
        Request::SetLogLevel { directive } => {
            crate::telemetry::set_filter(&directive).map_err(DaemonError::Generation)?;
            Ok(Response::Ack)
        }
        Request::Metrics => Ok(Response::Text {
            text: d.metrics_text(),
        }),
        Request::AppendBlock {
            session,
            role,
            kind,
            payload,
        } => Ok(Response::Committed {
            event: d.append_block(session, role, kind, payload)?.into(),
        }),
        Request::RequestPermission {
            session,
            call_id,
            text,
        } => Ok(Response::Committed {
            event: d.request_permission(session, call_id, text)?.into(),
        }),
        Request::RespondPermission {
            session,
            request_id,
            granted,
        } => Ok(Response::Committed {
            event: d.respond_permission(session, request_id, granted)?.into(),
        }),
        Request::RequestToolCancel { session, call_id } => Ok(Response::Committed {
            event: d.request_tool_cancel(session, call_id)?.into(),
        }),
        Request::AppendToolOutcome {
            session,
            call_id,
            outcome,
            note,
        } => Ok(Response::Committed {
            event: d.append_tool_outcome(session, call_id, outcome, note)?.into(),
        }),
        Request::Ledger { session } => Ok(Response::Ledger {
            entries: d.ledger(session)?,
        }),
        Request::PutMedia { bytes, mime: _ } => Ok(Response::Media {
            hash: d.put_media(&bytes)?,
        }),
        Request::AppendImage {
            session,
            role,
            blob,
            pre_text,
            post_text,
        } => Ok(Response::Committed {
            event: d.append_image(session, role, &blob, &pre_text, &post_text)?.into(),
        }),
        Request::GcMedia => {
            let (kept, removed) = d.gc_media()?;
            Ok(Response::Gc {
                kept: kept as u64,
                removed: removed as u64,
            })
        }
        Request::Trim { session, to_event } => Ok(Response::Created {
            session: d.trim(session, to_event)?,
        }),
        Request::Pin { session, ttl_ms } => Ok(Response::Pinned {
            deadline_unix_ms: d.pin(session, ttl_ms)?,
        }),
        Request::AppendSystem {
            session,
            text,
            tools,
        } => Ok(Response::Committed {
            event: d.append_system(session, text, tools)?.into(),
        }),
        Request::Complete {
            prefix,
            suffix,
            mode,
            max_tokens,
        } => {
            let c = d.complete(&prefix, &suffix, mode, max_tokens)?;
            Ok(Response::Completion {
                text: c.text,
                tokens: c.tokens,
                cached: c.cached,
                expired: c.expired,
            })
        }
        Request::Generate { .. }
        | Request::Cancel { .. }
        | Request::Subscribe { .. }
        | Request::Unsubscribe { .. } => {
            Err(DaemonError::Protocol("request routed to the wrong handler"))
        }
    })();
    result.unwrap_or_else(|e| Response::Err {
        message: e.to_string(),
    })
}

pub fn serve(listener: UnixListener, daemon: Arc<Daemon>) -> Result<(), DaemonError> {
    let cancels = daemon.cancel_registry();
    let active = daemon.active_registry();
    for conn in listener.incoming() {
        let stream = conn?;
        let daemon = Arc::clone(&daemon);
        let cancels = Arc::clone(&cancels);
        let active = Arc::clone(&active);
        std::thread::spawn(move || {
            let _ = serve_conn(stream, daemon, &cancels, &active);
        });
    }
    Ok(())
}

pub(crate) type Sink = Arc<dyn Fn(&Response) -> Result<(), DaemonError> + Send + Sync>;

fn send(w: &Sink, msg: &Response) -> Result<(), DaemonError> {
    w(msg)
}

struct ConnSub {
    session: u64,
    bus_id: u64,
    detached: Arc<AtomicBool>,
}

fn serve_conn(
    mut stream: UnixStream,
    daemon: Arc<Daemon>,
    cancels: &crate::CancelSet,
    active: &Mutex<HashSet<u64>>,
) -> Result<(), DaemonError> {
    let out = Mutex::new(stream.try_clone()?);
    let writer: Sink = Arc::new(move |msg: &Response| {
        write_frame(&mut *out.lock().expect("api writer"), msg)
    });
    let requests = std::iter::from_fn(move || read_frame::<Request>(&mut stream).transpose());
    serve_requests(requests, writer, daemon, cancels, active)
}

pub(crate) fn serve_requests(
    requests: impl Iterator<Item = Result<Request, DaemonError>>,
    writer: Sink,
    daemon: Arc<Daemon>,
    cancels: &crate::CancelSet,
    active: &Mutex<HashSet<u64>>,
) -> Result<(), DaemonError> {
    let mut subs: HashMap<u64, ConnSub> = HashMap::new();
    let bucket_cfg = daemon.completion_bucket();
    let mut fim_bucket =
        crate::completion_bucket::TokenBucket::new(bucket_cfg, std::time::Instant::now());
    let result = (|| -> Result<(), DaemonError> {
        for req in requests {
            match req? {
                Request::Generate {
                    session,
                    max_tokens,
                } => {
                    let w = Arc::clone(&writer);
                    let result = daemon.generate_streaming(session, max_tokens, |ev| {
                        send(
                            &w,
                            &Response::Delta {
                                event: ev.clone().into(),
                            },
                        )
                    });
                    let res = match result {
                        Ok(out) => Response::Generated {
                            events: out.events.into_iter().map(Into::into).collect(),
                            tokens_generated: out.tokens_generated,
                            finish: out.finish,
                            warm_prefix: out.warm_prefix,
                        },
                        Err(e) => Response::Err {
                            message: e.to_string(),
                        },
                    };
                    send(&writer, &res)?;
                }
                Request::Cancel { session } => {
                    send(&writer, &cancel(cancels, active, session))?;
                }
                Request::Subscribe {
                    session,
                    cursor,
                    provisional,
                } => {
                    let res = subscribe(&daemon, &writer, &mut subs, session, cursor, provisional);
                    if let Err(e) = res {
                        send(
                            &writer,
                            &Response::Err {
                                message: e.to_string(),
                            },
                        )?;
                    }
                }
                Request::Unsubscribe { sub } => {
                    let res = match subs.remove(&sub) {
                        Some(cs) => {
                            cs.detached.store(true, Ordering::Release);
                            daemon.bus().unsubscribe(cs.session, cs.bus_id);
                            Response::Unsubscribed
                        }
                        None => Response::Err {
                            message: format!("no subscription {sub} on this connection"),
                        },
                    };
                    send(&writer, &res)?;
                }
                req @ Request::Complete { .. } => {
                    let res = match fim_bucket.try_take(bucket_cfg, std::time::Instant::now()) {
                        Ok(()) => handle(&daemon, req),
                        Err(wait) => {
                            daemon.note_completion_throttled();
                            Response::Throttled {
                                retry_after_ms: crate::completion_bucket::retry_after_ms(wait),
                            }
                        }
                    };
                    send(&writer, &res)?;
                }
                other => {
                    let res = handle(&daemon, other);
                    send(&writer, &res)?;
                }
            }
        }
        Ok(())
    })();
    for (_, cs) in subs.drain() {
        cs.detached.store(true, Ordering::Release);
        daemon.bus().unsubscribe(cs.session, cs.bus_id);
    }
    result
}

fn subscribe(
    daemon: &Arc<Daemon>,
    writer: &Sink,
    subs: &mut HashMap<u64, ConnSub>,
    session: u64,
    cursor: u64,
    provisional: bool,
) -> Result<(), DaemonError> {
    let sub = daemon.bus().subscribe(session, provisional);
    let backlog = match daemon.read(session, cursor) {
        Ok(evs) => evs,
        Err(e) => {
            daemon.bus().unsubscribe(session, sub.id);
            return Err(e);
        }
    };
    let sub_id = sub.id;
    let detached = Arc::new(AtomicBool::new(false));
    subs.insert(
        sub_id,
        ConnSub {
            session,
            bus_id: sub.id,
            detached: Arc::clone(&detached),
        },
    );
    send(writer, &Response::Subscribed { sub: sub_id })?;
    let w = Arc::clone(writer);
    std::thread::Builder::new()
        .name(format!("superfluid-sub-{sub_id}"))
        .spawn(move || {
            let mut next_id = cursor;
            for ev in backlog {
                next_id = ev.event_id + 1;
                if send(
                    &w,
                    &Response::SubEvent {
                        sub: sub_id,
                        event: ev.into(),
                    },
                )
                .is_err()
                {
                    return;
                }
            }
            loop {
                if detached.load(Ordering::Acquire) {
                    let _ = send(
                        &w,
                        &Response::SubEnded {
                            sub: sub_id,
                            reason: "unsubscribed".into(),
                        },
                    );
                    return;
                }
                match sub.rx.recv() {
                    Ok(msg) if detached.load(Ordering::Acquire) => {
                        let _ = msg;
                        let _ = send(
                            &w,
                            &Response::SubEnded {
                                sub: sub_id,
                                reason: "unsubscribed".into(),
                            },
                        );
                        return;
                    }
                    Ok(BusMsg::Committed(ev)) => {
                        if ev.event_id < next_id {
                            continue;
                        }
                        next_id = ev.event_id + 1;
                        if send(
                            &w,
                            &Response::SubEvent {
                                sub: sub_id,
                                event: ev.into(),
                            },
                        )
                        .is_err()
                        {
                            return;
                        }
                    }
                    Ok(BusMsg::Provisional { channel, text }) => {
                        if send(
                            &w,
                            &Response::SubProvisional {
                                sub: sub_id,
                                channel,
                                text,
                            },
                        )
                        .is_err()
                        {
                            return;
                        }
                    }
                    Err(_) => {
                        let reason = if detached.load(Ordering::Acquire) {
                            "unsubscribed"
                        } else {
                            match sub.end_reason() {
                                EndReason::Lagged => "lagged",
                                EndReason::PublisherClosed => "publisher closed",
                                EndReason::Purged => "purged",
                            }
                        };
                        let _ = send(
                            &w,
                            &Response::SubEnded {
                                sub: sub_id,
                                reason: reason.into(),
                            },
                        );
                        return;
                    }
                }
            }
        })
        .expect("spawn subscription delivery");
    Ok(())
}

pub struct NativeClient {
    stream: UnixStream,
}

impl NativeClient {
    pub fn connect(path: &std::path::Path) -> Result<NativeClient, DaemonError> {
        Ok(NativeClient {
            stream: UnixStream::connect(path)?,
        })
    }

    pub fn request(&mut self, req: &Request) -> Result<Response, DaemonError> {
        write_frame(&mut self.stream, req)?;
        read_frame(&mut self.stream)?.ok_or(DaemonError::Protocol("daemon closed the connection"))
    }

    pub fn generate_stream(
        &mut self,
        session: u64,
        max_tokens: u32,
        mut on_delta: impl FnMut(EventMsg),
    ) -> Result<Response, DaemonError> {
        write_frame(
            &mut self.stream,
            &Request::Generate {
                session,
                max_tokens,
            },
        )?;
        loop {
            let res: Response = read_frame(&mut self.stream)?
                .ok_or(DaemonError::Protocol("daemon closed mid-stream"))?;
            match res {
                Response::Delta { event } => on_delta(event),
                terminal => return Ok(terminal),
            }
        }
    }

    pub fn cancel(&mut self, session: u64) -> Result<(), DaemonError> {
        match self.request(&Request::Cancel { session })? {
            Response::Cancelling => Ok(()),
            _ => Err(DaemonError::Protocol("unexpected cancel response")),
        }
    }

    pub fn subscribe(
        &mut self,
        session: u64,
        cursor: u64,
        provisional: bool,
    ) -> Result<u64, DaemonError> {
        match self.request(&Request::Subscribe {
            session,
            cursor,
            provisional,
        })? {
            Response::Subscribed { sub } => Ok(sub),
            Response::Err { message } => Err(DaemonError::Generation(message)),
            _ => Err(DaemonError::Protocol("unexpected subscribe response")),
        }
    }

    pub fn next_frame(&mut self) -> Result<Response, DaemonError> {
        read_frame(&mut self.stream)?.ok_or(DaemonError::Protocol("daemon closed the connection"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn socket_variant_indices_are_stable() {
        let enc = |r: &Request| postcard::to_stdvec(r).unwrap();
        assert_eq!(enc(&Request::List), [7]);
        assert_eq!(
            enc(&Request::AppendMessage { session: 1, role: 1, text: "hi".into() }),
            [2, 1, 1, 2, b'h', b'i']
        );
        assert_eq!(
            enc(&Request::AppendToolResult { session: 1, call_id: 4, content: "x".into() }),
            [5, 1, 4, 1, b'x']
        );
        assert_eq!(enc(&Request::Pin { session: 1, ttl_ms: 5 }), [33, 1, 5]);
        assert_eq!(
            enc(&Request::AppendSystem { session: 1, text: None, tools: vec!["{}".into()] }),
            [34, 1, 0, 1, 2, b'{', b'}']
        );
        let res = |r: &Response| postcard::to_stdvec(r).unwrap();
        assert_eq!(res(&Response::Err { message: "no".into() }), [13, 2, b'n', b'o']);
        assert_eq!(res(&Response::Throttled { retry_after_ms: 9 }), [26, 9]);
        let back: Request = postcard::from_bytes(&[3, 1, 8]).unwrap();
        assert!(matches!(back, Request::Generate { session: 1, max_tokens: 8 }));
    }
}
