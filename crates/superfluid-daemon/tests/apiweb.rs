//! api-web auth surface.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::{apiweb, Daemon, EngineHost, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};

fn spawn() -> (std::net::SocketAddr, String) {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-web-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let token = apiweb::mint_token(&dir.join("web-token")).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let t = token.clone();
    std::thread::spawn(move || {
        let _ = apiweb::serve_blocking(listener, daemon, t, vec!["https://ok.example".into()]);
    });
    (addr, token)
}

fn req(addr: std::net::SocketAddr, headers: &str, body: &str) -> (u16, String) {
    for _ in 0..50 {
        if let Ok(mut s) = TcpStream::connect(addr) {
            write!(
                s,
                "POST /web/rpc HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\n{headers}content-length: {}\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            let mut raw = String::new();
            s.read_to_string(&mut raw).unwrap();
            let (head, payload) = raw.split_once("\r\n\r\n").unwrap();
            let status = head.lines().next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
            return (status, payload.to_string());
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("no connection");
}

#[test]
fn web_requires_token_origin_and_csrf() {
    let (addr, token) = spawn();
    let body = r#"{"List":null}"#;
    let auth = format!("authorization: Bearer {token}\r\n");
    let csrf = format!("x-superfluid-csrf: {token}\r\n");
    assert_eq!(req(addr, &csrf, body).0, 401);
    assert_eq!(req(addr, "authorization: Bearer nope\r\nx-superfluid-csrf: nope\r\n", body).0, 401);
    assert_eq!(req(addr, &auth, body).0, 403);
    let bad_origin = format!("{auth}{csrf}origin: https://evil.example\r\n");
    assert_eq!(req(addr, &bad_origin, body).0, 403);
    let ok_origin = format!("{auth}{csrf}origin: https://ok.example\r\n");
    let (status, resp) = req(addr, &ok_origin, body);
    assert_eq!(status, 200, "{resp}");
    let (status, resp) = req(addr, &format!("{auth}{csrf}"), body);
    assert_eq!(status, 200, "{resp}");
    assert!(resp.contains("Sessions"), "native Response over api-web: {resp}");
    let (status, resp) = req(addr, &format!("{auth}{csrf}"), r#"{"Generate":{"session":1,"max_tokens":1}}"#);
    assert_eq!(status, 200);
    assert!(resp.contains("/web/ws"), "{resp}");
    let bad_auth = "authorization: Bearer nope\r\n";
    for h in [
        format!("{auth}{bad_auth}{csrf}"),
        format!("{bad_auth}{auth}{csrf}"),
    ] {
        assert_eq!(req(addr, &h, body).0, 401, "{h}");
    }
    let bad_csrf = "x-superfluid-csrf: nope\r\n";
    assert_eq!(req(addr, &format!("{auth}{csrf}{bad_csrf}"), body).0, 403);
    let two_origins = "origin: https://ok.example\r\norigin: https://evil.example\r\n";
    assert_eq!(
        req(addr, &format!("{auth}{csrf}{two_origins}"), body).0,
        403
    );
}

#[test]
fn web_stream_delivers_committed_events() {
    let (addr, token) = spawn();
    let auth = format!("authorization: Bearer {token}
x-superfluid-csrf: {token}
");
    let (st, resp) = req(addr, &auth, r#"{"Create":{"parent":null,"params":{"temperature":0.0,"top_p":0.0,"min_p":0.0,"top_k":0,"seed":0}}}"#);
    assert_eq!(st, 200, "{resp}");
    let session: u64 = {
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        v["Created"]["session"].as_u64().unwrap()
    };
    let append = format!(r#"{{"Append":{{"session":{session},"text":null,"span":[1,2,3,4,5,6,7,8]}}}}"#);
    assert_eq!(req(addr, &auth, &append).0, 200);
    let body = format!(r#"{{"session":{session},"max_tokens":4,"provisional":true}}"#);
    let (status, _) = stream_req(addr, "", &body);
    assert_eq!(status, 401);
    let (status, sse) = stream_req(addr, &auth, &body);
    assert_eq!(status, 200, "{sse}");
    assert!(sse.contains("[DONE]"), "terminated: {sse}");
    assert!(sse.contains("\"committed\""), "committed frames: {sse}");
    assert!(sse.contains("\"done\""), "done frame with summary: {sse}");
}

#[test]
fn web_rpc_cancels_a_streaming_generation() {
    let (addr, token) = spawn();
    let auth = format!("authorization: Bearer {token}\r\nx-superfluid-csrf: {token}\r\n");
    let (_, resp) = req(addr, &auth, r#"{"Create":{"parent":null,"params":{"temperature":0.0,"top_p":0.0,"min_p":0.0,"top_k":0,"seed":0}}}"#);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let session = v["Created"]["session"].as_u64().unwrap();
    let append = format!(r#"{{"Append":{{"session":{session},"text":null,"span":[1,2,3,4,5,6,7,8]}}}}"#);
    assert_eq!(req(addr, &auth, &append).0, 200);
    let cancel = format!(r#"{{"Cancel":{{"session":{session}}}}}"#);
    let (status, resp) = req(addr, &auth, &cancel);
    assert_eq!(status, 200);
    assert!(resp.contains("no generation in flight"), "idle session: {resp}");

    let stream_auth = auth.clone();
    let streaming = std::thread::spawn(move || {
        stream_req(addr, &stream_auth, &format!(r#"{{"session":{session},"max_tokens":1000000}}"#))
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let (_, resp) = req(addr, &auth, &cancel);
        if resp.contains("Cancelling") {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "the generation never became cancellable: {resp}");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let (status, sse) = streaming.join().unwrap();
    assert_eq!(status, 200);
    assert!(sse.contains("\"finish\":4"), "ended as cancelled: {sse}");
}

#[test]
fn web_rpc_declares_tools() {
    let (addr, token) = spawn();
    let auth = format!("authorization: Bearer {token}\r\nx-superfluid-csrf: {token}\r\n");
    let (_, resp) = req(addr, &auth, r#"{"Create":{"parent":null,"params":{"temperature":0.0,"top_p":0.0,"min_p":0.0,"top_k":0,"seed":0}}}"#);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let session = v["Created"]["session"].as_u64().unwrap();
    let tool = serde_json::json!({"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}});
    let bad = serde_json::json!({"AppendSystem": {"session": session, "text": "sys", "tools": [tool.to_string(), "{\"type\":\"function\"}"]}});
    let (status, resp) = req(addr, &auth, &bad.to_string());
    assert_eq!(status, 200);
    assert!(resp.contains("tool 1 is not a function declaration"), "{resp}");
    let good = serde_json::json!({"AppendSystem": {"session": session, "text": "sys", "tools": [tool.to_string()]}});
    let (status, resp) = req(addr, &auth, &good.to_string());
    assert_eq!(status, 200);
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    let msg = &v["Committed"]["event"]["body"]["Message"];
    assert_eq!((msg["role"].as_u64(), msg["text"].as_str()), (Some(0), Some("sys")), "{resp}");
    let (_, resp) = req(addr, &auth, &good.to_string());
    assert!(resp.contains("AppendSystem must come before any other turn"), "{resp}");
}

fn stream_req(addr: std::net::SocketAddr, headers: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    let head = format!(
        "POST /web/stream HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\n{headers}content-length: {}\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).unwrap();
    s.write_all(body.as_bytes()).unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let raw = String::from_utf8_lossy(&raw).into_owned();
    let (head, payload) = raw.split_once("\r\n\r\n").unwrap_or((&raw, ""));
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    (status, payload.to_string())
}

use superfluid_daemon::api::{self, Request, Response};
use superfluid_daemon::codec::MockCodec;
use superfluid_daemon::{EventBody, GenParams};
use tungstenite::client::IntoClientRequest;
use tungstenite::stream::MaybeTlsStream;

const OK_ORIGIN: &str = "https://ok.example";

struct Served {
    addr: std::net::SocketAddr,
    token: String,
    socket: std::path::PathBuf,
    daemon: Arc<Daemon>,
}

fn spawn_both() -> Served {
    spawn_ticking(std::time::Duration::ZERO)
}

fn spawn_ticking(tick_delay: std::time::Duration) -> Served {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("bws-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let token = apiweb::mint_token(&dir.join("web-token")).unwrap();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let cfg = EngineConfig {
        vocab: 512,
        tick_delay,
        ..EngineConfig::default()
    };
    let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).unwrap();
    let daemon = Arc::new(Daemon::new(store, host, Box::new(MockCodec), 8));
    let socket = dir.join("d.sock");
    let uds = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let d = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = api::serve(uds, d);
    });
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (d, t) = (Arc::clone(&daemon), token.clone());
    std::thread::spawn(move || {
        let _ = apiweb::serve_blocking(listener, d, t, vec![OK_ORIGIN.into()]);
    });
    Served {
        addr,
        token,
        socket,
        daemon,
    }
}

type Ws = tungstenite::WebSocket<MaybeTlsStream<TcpStream>>;

fn ws_connect(addr: std::net::SocketAddr, headers: &[(&'static str, String)]) -> Result<Ws, u16> {
    let mut req = format!("ws://{addr}/web/ws").into_client_request().unwrap();
    for (k, v) in headers {
        req.headers_mut().append(*k, v.parse().unwrap());
    }
    for _ in 0..50 {
        match tungstenite::connect(req.clone()) {
            Ok((ws, _)) => {
                if let MaybeTlsStream::Plain(s) = ws.get_ref() {
                    s.set_read_timeout(Some(std::time::Duration::from_secs(20))).unwrap();
                }
                return Ok(ws);
            }
            Err(tungstenite::Error::Http(resp)) => return Err(resp.status().as_u16()),
            Err(tungstenite::Error::Io(_)) => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            Err(e) => panic!("ws connect: {e}"),
        }
    }
    panic!("no connection");
}

fn ws_open(s: &Served) -> Ws {
    ws_connect(
        s.addr,
        &[
            ("origin", OK_ORIGIN.into()),
            (
                "sec-websocket-protocol",
                format!("superfluid.v1, superfluid.token.{}", s.token),
            ),
        ],
    )
    .expect("authorized upgrade")
}

fn ws_send(ws: &mut Ws, req: &Request) {
    let text = serde_json::to_string(req).unwrap();
    ws.send(tungstenite::Message::text(text)).unwrap();
}

fn ws_next(ws: &mut Ws) -> Response {
    loop {
        match ws.read().expect("ws frame") {
            tungstenite::Message::Text(t) => return serde_json::from_str(t.as_str()).unwrap(),
            tungstenite::Message::Ping(_) | tungstenite::Message::Pong(_) => continue,
            other => panic!("unexpected ws message {other:?}"),
        }
    }
}

fn ws_request(ws: &mut Ws, req: &Request) -> Response {
    ws_send(ws, req);
    ws_next(ws)
}

#[test]
fn ws_upgrade_requires_token_and_origin() {
    let s = spawn_both();
    let origin = ("origin", OK_ORIGIN.to_string());
    let bearer = ("authorization", format!("Bearer {}", s.token));
    assert_eq!(ws_connect(s.addr, std::slice::from_ref(&origin)).err(), Some(401));
    let bad_sub = ("sec-websocket-protocol", "superfluid.v1, superfluid.token.nope".to_string());
    assert_eq!(ws_connect(s.addr, &[origin.clone(), bad_sub]).err(), Some(401));
    let bad_hdr = ("authorization", "Bearer nope".to_string());
    assert_eq!(ws_connect(s.addr, &[origin.clone(), bad_hdr]).err(), Some(401));
    let good_sub = (
        "sec-websocket-protocol",
        format!("superfluid.v1, superfluid.token.{}", s.token),
    );
    let mixed = [
        origin.clone(),
        ("authorization", "Bearer nope".to_string()),
        good_sub.clone(),
    ];
    assert_eq!(ws_connect(s.addr, &mixed).err(), Some(401));
    assert_eq!(ws_connect(s.addr, std::slice::from_ref(&bearer)).err(), Some(403));
    assert_eq!(ws_connect(s.addr, std::slice::from_ref(&good_sub)).err(), Some(403));
    let evil = ("origin", "https://evil.example".to_string());
    assert_eq!(ws_connect(s.addr, &[evil, good_sub.clone()]).err(), Some(403));
    for list in [
        format!("superfluid.v1, superfluid.token.{}, superfluid.token.nope", s.token),
        format!("superfluid.v1, superfluid.token.nope, superfluid.token.{}", s.token),
    ] {
        let two = [origin.clone(), ("sec-websocket-protocol", list)];
        assert_eq!(ws_connect(s.addr, &two).err(), Some(401));
    }
    let two_auth = [
        origin.clone(),
        bearer.clone(),
        ("authorization", "Bearer nope".into()),
    ];
    assert_eq!(ws_connect(s.addr, &two_auth).err(), Some(401));
    let two_origin = [
        origin.clone(),
        ("origin", "https://evil.example".into()),
        good_sub.clone(),
    ];
    assert_eq!(ws_connect(s.addr, &two_origin).err(), Some(403));
    let no_v1 = ("sec-websocket-protocol", format!("superfluid.token.{}", s.token));
    assert_eq!(ws_connect(s.addr, &[origin.clone(), no_v1]).err(), Some(400));

    let mut req = format!("ws://{}/web/ws", s.addr).into_client_request().unwrap();
    req.headers_mut().insert("origin", OK_ORIGIN.parse().unwrap());
    req.headers_mut()
        .insert("sec-websocket-protocol", good_sub.1.parse().unwrap());
    let (mut ws, resp) = tungstenite::connect(req).expect("subprotocol token upgrades");
    assert_eq!(
        resp.headers().get("sec-websocket-protocol").map(|v| v.to_str().unwrap()),
        Some("superfluid.v1"),
        "only superfluid.v1 is echoed, never the token entry"
    );
    assert!(matches!(ws_request(&mut ws, &Request::List), Response::Sessions { .. }));
    let mut ws = ws_connect(s.addr, &[origin, bearer]).expect("header token upgrades");
    assert!(matches!(ws_request(&mut ws, &Request::List), Response::Sessions { .. }));
}

#[test]
fn ws_carries_rpc_and_streaming_generate() {
    let s = spawn_both();
    let mut ws = ws_open(&s);
    let session = match ws_request(
        &mut ws,
        &Request::Create {
            parent: None,
            params: GenParams::default(),
        },
    ) {
        Response::Created { session } => session,
        other => panic!("{other:?}"),
    };
    let appended = ws_request(
        &mut ws,
        &Request::Append {
            session,
            text: None,
            span: (0..16).collect(),
        },
    );
    assert!(matches!(appended, Response::Committed { .. }), "{appended:?}");

    ws.send(tungstenite::Message::text("not json")).unwrap();
    assert!(matches!(ws_next(&mut ws), Response::Err { message } if message.contains("invalid request frame")));
    ws.send(tungstenite::Message::text(r#"{"NoSuchVerb":{}}"#)).unwrap();
    assert!(matches!(ws_next(&mut ws), Response::Err { .. }));
    ws.send(tungstenite::Message::binary(serde_json::to_vec(&Request::List).unwrap()))
        .unwrap();
    assert!(matches!(ws_next(&mut ws), Response::Sessions { ids } if ids.contains(&session)));

    ws_send(
        &mut ws,
        &Request::Generate {
            session,
            max_tokens: 8,
        },
    );
    let mut deltas = Vec::new();
    let (events, tokens) = loop {
        match ws_next(&mut ws) {
            Response::Delta { event } => deltas.push(event),
            Response::Generated {
                events,
                tokens_generated,
                ..
            } => break (events, tokens_generated),
            other => panic!("unexpected {other:?}"),
        }
    };
    assert!(tokens > 0);
    assert!(!deltas.is_empty(), "committed events stream as Delta frames");
    assert_eq!(deltas, events, "the deltas are the summary's events, in order");
    match ws_request(&mut ws, &Request::Read { session, cursor: 0 }) {
        Response::Events { events: all } => {
            assert_eq!(&all[all.len() - events.len()..], &events[..]);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn ws_close_completes_the_handshake() {
    let s = spawn_both();
    let mut ws = ws_open(&s);
    assert!(matches!(ws_request(&mut ws, &Request::List), Response::Sessions { .. }));
    ws.close(None).unwrap();
    loop {
        match ws.read() {
            Ok(_) => continue,
            Err(tungstenite::Error::ConnectionClosed) => break,
            Err(e) => panic!("abnormal close: {e}"),
        }
    }
}

#[test]
fn ws_pipelined_requests_are_backpressured_not_buffered() {
    let s = spawn_both();
    let mut ws = ws_open(&s);
    let session = match ws_request(
        &mut ws,
        &Request::Create {
            parent: None,
            params: GenParams::default(),
        },
    ) {
        Response::Created { session } => session,
        other => panic!("{other:?}"),
    };
    ws_request(
        &mut ws,
        &Request::Append {
            session,
            text: None,
            span: (0..16).collect(),
        },
    );
    ws_send(
        &mut ws,
        &Request::Generate {
            session,
            max_tokens: 64,
        },
    );
    const N: usize = 200;
    for _ in 0..N {
        ws_send(&mut ws, &Request::Tree);
    }
    let mut generated = false;
    let mut trees = 0;
    while trees < N {
        match ws_next(&mut ws) {
            Response::Delta { .. } => assert!(!generated, "deltas precede the summary"),
            Response::Generated { .. } => generated = true,
            Response::Tree { .. } => {
                assert!(generated, "queued requests run after the Generate");
                trees += 1;
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert!(matches!(ws_request(&mut ws, &Request::List), Response::Sessions { .. }));
}

#[test]
fn ws_large_pipelined_frames_pass_the_byte_budget_in_order() {
    let s = spawn_both();
    let mut ws = ws_open(&s);
    let session = match ws_request(
        &mut ws,
        &Request::Create {
            parent: None,
            params: GenParams::default(),
        },
    ) {
        Response::Created { session } => session,
        other => panic!("{other:?}"),
    };
    ws_request(
        &mut ws,
        &Request::Append {
            session,
            text: None,
            span: (0..16).collect(),
        },
    );
    ws_send(
        &mut ws,
        &Request::Generate {
            session,
            max_tokens: 64,
        },
    );
    let big = format!(r#"{{"NoSuchVerb":"{}"}}"#, "x".repeat(2 << 20));
    const N: usize = 48;
    let writer = {
        let MaybeTlsStream::Plain(s) = ws.get_ref() else {
            panic!("plain socket")
        };
        let s = s.try_clone().unwrap();
        let big = big.clone();
        std::thread::spawn(move || {
            let mut w = tungstenite::WebSocket::from_raw_socket(
                s,
                tungstenite::protocol::Role::Client,
                None,
            );
            for _ in 0..N {
                w.write(tungstenite::Message::text(big.clone())).unwrap();
            }
            w.flush().unwrap();
            std::mem::forget(w);
        })
    };
    let mut generated = false;
    let mut errs = 0;
    while errs < N {
        match ws_next(&mut ws) {
            Response::Delta { .. } => {}
            Response::Generated { .. } => generated = true,
            Response::Err { .. } => {
                assert!(generated, "queued frames run after the Generate");
                errs += 1;
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    writer.join().unwrap();
}

#[test]
fn ws_subscription_matches_uds() {
    let s = spawn_both();
    let session = s.daemon.create(None, GenParams::default()).unwrap();
    s.daemon.append(session, None, (0..16).collect()).unwrap();

    let mut uds = api::NativeClient::connect(&s.socket).unwrap();
    let uds_sub = uds.subscribe(session, 0, false).unwrap();
    let mut ws = ws_open(&s);
    let ws_sub = match ws_request(
        &mut ws,
        &Request::Subscribe {
            session,
            cursor: 0,
            provisional: true,
        },
    ) {
        Response::Subscribed { sub } => sub,
        other => panic!("{other:?}"),
    };

    let mut gen = ws_open(&s);
    ws_send(
        &mut gen,
        &Request::Generate {
            session,
            max_tokens: 8,
        },
    );
    let last_id = loop {
        match ws_next(&mut gen) {
            Response::Delta { .. } => {}
            Response::Generated { events, .. } => break events.last().unwrap().event_id,
            other => panic!("unexpected {other:?}"),
        }
    };

    let mut uds_events = Vec::new();
    while uds_events.last().map(|e: &api::EventMsg| e.event_id) != Some(last_id) {
        match uds.next_frame().unwrap() {
            Response::SubEvent { sub, event } => {
                assert_eq!(sub, uds_sub);
                uds_events.push(event);
            }
            other => panic!("UDS without provisional opt-in got {other:?}"),
        }
    }
    let mut ws_events = Vec::new();
    let mut provisional = String::new();
    while ws_events.last().map(|e: &api::EventMsg| e.event_id) != Some(last_id) {
        match ws_next(&mut ws) {
            Response::SubEvent { sub, event } => {
                assert_eq!(sub, ws_sub);
                ws_events.push(event);
            }
            Response::SubProvisional { sub, text, .. } => {
                assert_eq!(sub, ws_sub);
                provisional.push_str(&text);
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(ws_events, uds_events, "same committed stream, same ids, same order");
    assert_eq!(ws_events[0].event_id, 0, "the backlog from the cursor comes first");
    let committed: String = ws_events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Generated { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(!committed.is_empty(), "inputs must decode to text");
    assert_eq!(provisional, committed, "provisional deltas reconcile with the commits");

    ws_send(&mut ws, &Request::Unsubscribe { sub: ws_sub });
    let (mut acked, mut ended) = (false, false);
    while !(acked && ended) {
        match ws_next(&mut ws) {
            Response::Unsubscribed => acked = true,
            Response::SubEnded { sub, reason } => {
                assert_eq!(sub, ws_sub);
                assert_eq!(reason, "unsubscribed");
                ended = true;
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert!(matches!(
        ws_request(&mut ws, &Request::Unsubscribe { sub: ws_sub }),
        Response::Err { .. }
    ));
    assert!(matches!(ws_request(&mut ws, &Request::List), Response::Sessions { .. }));
}

#[test]
fn web_stream_cancels_its_generation_when_the_client_goes_away() {
    let s = spawn_ticking(std::time::Duration::from_millis(200));
    let session = s.daemon.create(None, GenParams::default()).unwrap();
    s.daemon.append(session, None, (0..16).collect()).unwrap();
    let body = format!(r#"{{"session":{session},"max_tokens":12000}}"#);
    let mut conn = TcpStream::connect(s.addr).unwrap();
    write!(
        conn,
        "POST /web/stream HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-type: application/json\r\nauthorization: Bearer {0}\r\nx-superfluid-csrf: {0}\r\ncontent-length: {1}\r\n\r\n{body}",
        s.token,
        body.len()
    )
    .unwrap();
    conn.set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    let mut seen = Vec::new();
    while !String::from_utf8_lossy(&seen).contains("\"committed\"") {
        let mut buf = [0u8; 4096];
        let n = conn.read(&mut buf).unwrap();
        assert!(
            n > 0,
            "the stream ended early: {}",
            String::from_utf8_lossy(&seen)
        );
        seen.extend_from_slice(&buf[..n]);
    }
    let seen = String::from_utf8_lossy(&seen);
    assert!(seen.starts_with("HTTP/1.1 200"), "{seen}");
    let active = s.daemon.active_registry();
    assert!(
        active.lock().unwrap().contains(&session),
        "still generating when the client hangs up"
    );
    let hung_up = std::time::Instant::now();
    drop(conn);
    while active.lock().unwrap().contains(&session) {
        assert!(
            hung_up.elapsed() < std::time::Duration::from_millis(4500),
            "the generation outlived its client by {:?} (run to its end it takes about 13 s)",
            hung_up.elapsed()
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let finish = s
        .daemon
        .read(session, 0)
        .unwrap()
        .iter()
        .rev()
        .find_map(|e| match e.body {
            EventBody::Generated { finish, .. } => Some(finish),
            _ => None,
        });
    assert_eq!(
        finish,
        Some(superfluid_abi::finish::CANCELLED),
        "cancelled, not run to its end"
    );
}

// The clock alone repeats within a microsecond on macOS, and two tests on one
// log are refused while both stores are open.
fn unique() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    format!("{t}-{}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}
