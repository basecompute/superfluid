use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use superfluid_abi::finish;
use superfluid_daemon::codec::MockChatCodec;
use superfluid_daemon::{api, Daemon, EngineHost, EventBody, GenParams, MockCodec, SessionStore};
use superfluid_engine::{EngineConfig, MockEngine};
use superfluid_proto::linkw;
use superfluid_shm::{LOGITS_RING, TOKEN_RING_IN, TOKEN_RING_OUT};

static DIRS: AtomicU64 = AtomicU64::new(0);

fn test_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "superfluid-daemon-test-{}-{}",
        std::process::id(),
        DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("test dir");
    dir
}

fn mock_daemon(wal: &std::path::Path) -> Daemon {
    let store = SessionStore::open(wal).expect("open store");
    let host =
        EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    Daemon::new(store, host, Box::new(MockCodec), 8)
}

fn generated_tokens(events: &[superfluid_daemon::CommittedEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Generated { span, .. } => Some(span.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

#[test]
fn commit_replay_and_epoch_bump() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let prompt: Vec<u32> = (100..116).collect();

    let (session, first_epoch, gen_tokens) = {
        let d = mock_daemon(&wal);
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, Some("hello".into()), prompt.clone()).unwrap();
        let out = d.generate(session, 8).unwrap();
        assert_eq!(out.tokens_generated, 8);
        assert_eq!(out.finish, finish::NONE);
        assert_eq!(out.warm_prefix, 0, "first generation is cold");
        let store = d.store();
        let store = store.lock().unwrap();
        let s = store.session(session).unwrap();
        for (i, e) in s.events.iter().enumerate() {
            assert_eq!(e.event_id, i as u64);
        }
        (session, s.epoch, generated_tokens(&s.events))
    };

    let store = SessionStore::open(&wal).expect("reopen");
    let s = store.session(session).unwrap();
    let mut expect = prompt.clone();
    expect.extend_from_slice(&gen_tokens);
    assert_eq!(s.tokens, expect, "materialized stream replays verbatim");
    assert_eq!(s.epoch, first_epoch + 1, "recovery bumps the append epoch");
    assert_eq!(gen_tokens.len(), 8);
}

#[test]
fn warm_resume_reuses_the_published_prefix() {
    let dir = test_dir();
    let d = mock_daemon(&dir.join("wal.log"));
    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, None, (0..32).collect()).unwrap();

    let first = d.generate(session, 8).unwrap();
    assert_eq!(first.warm_prefix, 0);

    let second = d.generate(session, 4).unwrap();
    assert!(
        second.warm_prefix >= 16,
        "resume seeded {} tokens from cache",
        second.warm_prefix
    );
    assert_eq!(second.tokens_generated, 4);
}

#[test]
fn same_log_same_seed_same_tokens() {
    let run = |dir: PathBuf| -> Vec<u32> {
        let d = mock_daemon(&dir.join("wal.log"));
        let session = d
            .create(
                None,
                GenParams {
                    seed: 42,
                    ..Default::default()
                },
            )
            .unwrap();
        d.append(session, None, (7..39).collect()).unwrap();
        let out = d.generate(session, 6).unwrap();
        generated_tokens(&out.events)
    };
    assert_eq!(run(test_dir()), run(test_dir()));
}

#[test]
fn torn_tail_is_recovered_not_fatal() {
    use std::io::Write;
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let session = {
        let d = mock_daemon(&wal);
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, vec![1, 2, 3]).unwrap();
        session
    };
    let mut f = std::fs::OpenOptions::new().append(true).open(&wal).unwrap();
    f.write_all(&[0xDE, 0xAD, 0xBE]).unwrap();
    drop(f);

    let mut store = SessionStore::open(&wal).expect("recovers");
    assert_eq!(store.session(session).unwrap().tokens, vec![1, 2, 3]);
    store.append(session, None, vec![4]).unwrap();
    assert_eq!(store.session(session).unwrap().tokens, vec![1, 2, 3, 4]);
}

#[test]
fn cursor_reads_are_total() {
    let dir = test_dir();
    let d = mock_daemon(&dir.join("wal.log"));
    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, None, vec![9, 9]).unwrap();
    let all = d.read(session, 0).unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(d.read(session, 1).unwrap().len(), 1);
    assert_eq!(d.read(session, 999).unwrap().len(), 0);
}

#[test]
fn native_api_round_trip() {
    let dir = test_dir();
    let socket = dir.join("superfluid.sock");
    let daemon = Arc::new(mock_daemon(&dir.join("wal.log")));
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let serving = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = api::serve(listener, serving);
    });

    let mut client = api::NativeClient::connect(&socket).expect("connect");
    let session = match client
        .request(&api::Request::Create {
            parent: None,
            params: GenParams::default(),
        })
        .unwrap()
    {
        api::Response::Created { session } => session,
        other => panic!("unexpected {other:?}"),
    };
    match client
        .request(&api::Request::Append {
            session,
            text: Some("hi".into()),
            span: (0..16).collect(),
        })
        .unwrap()
    {
        api::Response::Committed { event } => assert_eq!(event.event_id, 1),
        other => panic!("unexpected {other:?}"),
    }
    let mut deltas = Vec::new();
    match client
        .generate_stream(session, 4, |ev| deltas.push(ev))
        .unwrap()
    {
        api::Response::Generated {
            tokens_generated,
            events,
            ..
        } => {
            assert_eq!(tokens_generated, 4);
            assert!(!events.is_empty());
            assert_eq!(
                deltas.iter().map(|e| e.event_id).collect::<Vec<_>>(),
                events.iter().map(|e| e.event_id).collect::<Vec<_>>()
            );
        }
        other => panic!("unexpected {other:?}"),
    }
    match client
        .request(&api::Request::Read { session, cursor: 0 })
        .unwrap()
    {
        api::Response::Events { events } => {
            assert!(events.len() >= 3);
            for (i, e) in events.iter().enumerate() {
                assert_eq!(e.event_id, i as u64);
            }
        }
        other => panic!("unexpected {other:?}"),
    }
    match client.request(&api::Request::List).unwrap() {
        api::Response::Sessions { ids } => assert_eq!(ids, vec![session]),
        other => panic!("unexpected {other:?}"),
    }
    match client.generate_stream(999, 1, |_| {}).unwrap() {
        api::Response::Err { message } => assert!(message.contains("unknown session")),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn text_appends_and_decoded_generation() {
    let dir = test_dir();
    let d = mock_daemon(&dir.join("wal.log"));
    let session = d.create(None, GenParams::default()).unwrap();
    let e = d.append(session, Some("héllo".into()), Vec::new()).unwrap();
    let EventBody::Appended { span, text } = &e.body else {
        panic!("wrong body");
    };
    assert_eq!(text.as_deref(), Some("héllo"));
    assert_eq!(span.len(), "héllo".len());
    let out = d.generate(session, 4).unwrap();
    for ev in &out.events {
        let EventBody::Generated { span, text, .. } = &ev.body else {
            panic!("wrong body");
        };
        assert_eq!(*text, superfluid_daemon::TextCodec::decode(&MockCodec, span));
    }
}

#[test]
fn cancel_stops_generation_with_a_committed_event() {
    let dir = test_dir();
    let socket = dir.join("superfluid.sock");
    let daemon = Arc::new(mock_daemon(&dir.join("wal.log")));
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let serving = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = api::serve(listener, serving);
    });

    let mut client = api::NativeClient::connect(&socket).expect("connect");
    let session = match client
        .request(&api::Request::Create {
            parent: None,
            params: GenParams::default(),
        })
        .unwrap()
    {
        api::Response::Created { session } => session,
        other => panic!("unexpected {other:?}"),
    };
    client
        .request(&api::Request::Append {
            session,
            text: None,
            span: (0..16).collect(),
        })
        .unwrap();

    let socket2 = socket.clone();
    let mut cancelled_at = None;
    let res = client
        .generate_stream(session, 1_000_000, |ev| {
            if cancelled_at.is_none() {
                cancelled_at = Some(ev.event_id);
                let mut c2 = api::NativeClient::connect(&socket2).expect("connect 2");
                c2.cancel(session).expect("cancel");
            }
        })
        .unwrap();
    match res {
        api::Response::Generated {
            tokens_generated,
            finish,
            events,
            ..
        } => {
            assert!(tokens_generated < 1_000_000);
            assert_eq!(finish, finish::CANCELLED);
            let last = events.last().unwrap();
            let EventBody::Generated { finish, .. } = &last.body else {
                panic!("wrong body");
            };
            assert_eq!(*finish, superfluid_abi::finish::CANCELLED);
        }
        other => panic!("unexpected {other:?}"),
    }

    let res = client.generate_stream(session, 2, |_| {}).unwrap();
    match res {
        api::Response::Generated {
            tokens_generated, ..
        } => assert_eq!(tokens_generated, 2),
        other => panic!("unexpected {other:?}"),
    }
}

fn mock_daemon_with_lanes(wal: &std::path::Path, max_lanes: usize) -> Daemon {
    let store = SessionStore::open(wal).expect("open store");
    let host =
        EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    Daemon::new(store, host, Box::new(MockCodec), max_lanes)
}

#[test]
fn concurrent_generations_equal_sequential() {
    let concurrent = {
        let d = Arc::new(mock_daemon(&test_dir().join("wal.log")));
        let sessions: Vec<u64> = (0..3)
            .map(|i| {
                let sid = d.create(None, GenParams::default()).unwrap();
                d.append(sid, None, (i * 100..i * 100 + 24).collect()).unwrap();
                sid
            })
            .collect();
        let handles: Vec<_> = sessions
            .iter()
            .map(|&sid| {
                let d = Arc::clone(&d);
                std::thread::spawn(move || {
                    let out = d.generate(sid, 40).unwrap();
                    assert_eq!(out.tokens_generated, 40);
                    generated_tokens(&out.events)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    };

    let sequential = {
        let d = mock_daemon(&test_dir().join("wal.log"));
        (0..3u32)
            .map(|i| {
                let sid = d.create(None, GenParams::default()).unwrap();
                d.append(sid, None, (i * 100..i * 100 + 24).collect()).unwrap();
                generated_tokens(&d.generate(sid, 40).unwrap().events)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(concurrent, sequential, "batching changed a lane's output");
}

#[test]
fn queueing_past_the_lane_cap() {
    let d = Arc::new(mock_daemon_with_lanes(&test_dir().join("wal.log"), 2));
    let handles: Vec<_> = (0..5u32)
        .map(|i| {
            let d = Arc::clone(&d);
            std::thread::spawn(move || {
                let sid = d.create(None, GenParams::default()).unwrap();
                d.append(sid, None, (i * 50..i * 50 + 16).collect()).unwrap();
                let out = d.generate(sid, 24).unwrap();
                assert_eq!(out.tokens_generated, 24);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn busy_guard_while_generating() {
    let d = Arc::new(mock_daemon(&test_dir().join("wal.log")));
    let sid = d.create(None, GenParams::default()).unwrap();
    d.append(sid, None, (0..16).collect()).unwrap();

    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (hold_tx, hold_rx) = std::sync::mpsc::channel::<()>();
    let gen = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || {
            let mut signalled = false;
            d.generate_streaming(sid, 512, |_| {
                if !signalled {
                    signalled = true;
                    started_tx.send(()).unwrap();
                    let _ = hold_rx.recv();
                }
                Ok(())
            })
            .unwrap()
        })
    };
    started_rx.recv().unwrap();
    assert!(matches!(
        d.append(sid, None, vec![1]),
        Err(superfluid_daemon::DaemonError::SessionBusy(_))
    ));
    assert!(matches!(
        d.generate(sid, 1),
        Err(superfluid_daemon::DaemonError::SessionBusy(_))
    ));
    hold_tx.send(()).unwrap();
    let out = gen.join().unwrap();
    assert_eq!(out.tokens_generated, 512);
    d.append(sid, None, vec![1]).unwrap();
}

#[test]
fn cancel_hits_only_its_lane() {
    let d = Arc::new(mock_daemon(&test_dir().join("wal.log")));
    let a = d.create(None, GenParams::default()).unwrap();
    let b = d.create(None, GenParams::default()).unwrap();
    d.append(a, None, (0..16).collect()).unwrap();
    d.append(b, None, (500..516).collect()).unwrap();

    let cancels = d.cancel_registry();
    let (a_started_tx, a_started_rx) = std::sync::mpsc::channel();
    let ha = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || {
            let mut signalled = false;
            d.generate_streaming(a, 100_000, |_| {
                if !signalled {
                    signalled = true;
                    a_started_tx.send(()).unwrap();
                }
                Ok(())
            })
            .unwrap()
        })
    };
    let hb = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || d.generate(b, 64).unwrap())
    };
    a_started_rx.recv().unwrap();
    cancels.lock().unwrap().insert(a);

    let out_a = ha.join().unwrap();
    let out_b = hb.join().unwrap();
    assert_eq!(out_a.finish, finish::CANCELLED);
    assert!(out_a.tokens_generated < 100_000);
    assert_eq!(out_b.tokens_generated, 64, "the other lane was untouched");
    assert_eq!(out_b.finish, finish::NONE);
}

#[test]
fn chat_session_event_flow_and_replay() {
    use superfluid_daemon::codec::{MockChatCodec, TextCodec};
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let (session, tokens_before) = {
        let store = SessionStore::open(&wal).unwrap();
        let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None))
            .expect("spawn");
        let d = Daemon::new(store, host, Box::new(MockChatCodec), 8);
        let session = d.create(None, GenParams::default()).unwrap();
        let e = d.append_message(session, 1, "hello there".into()).unwrap();
        let EventBody::Message { role, text, span } = &e.body else {
            panic!("wrong body");
        };
        assert_eq!(*role, 1);
        assert_eq!(text, "hello there");
        assert_eq!(
            *span,
            MockChatCodec.render_message(1, "hello there").unwrap()
        );

        let out = d.generate(session, 8).unwrap();
        assert_eq!(out.tokens_generated, 8);
        {
            let store = d.store();
            let store = store.lock().unwrap();
            let s = store.session(session).unwrap();
            assert!(matches!(s.events[1].body, EventBody::Message { .. }));
            assert!(matches!(
                s.events[2].body,
                EventBody::GenerationFingerprint { .. }
            ));
            let EventBody::GenerationPrompt { span: opener } = &s.events[3].body else {
                panic!("no opener before generation");
            };
            assert_eq!(
                *opener,
                MockChatCodec
                    .generation_prefix(superfluid_daemon::codec::TurnState::AfterInput)
                    .unwrap()
            );
            assert!(matches!(s.events[4].body, EventBody::Generated { .. }));
        }

        d.generate(session, 4).unwrap();
        let store = d.store();
        let store = store.lock().unwrap();
        let s = store.session(session).unwrap();
        let openers = s
            .events
            .iter()
            .filter(|e| matches!(e.body, EventBody::GenerationPrompt { .. }))
            .count();
        assert_eq!(openers, 1, "mid-turn resume must not re-open the turn");
        (session, s.tokens.clone())
    };

    let store = SessionStore::open(&wal).unwrap();
    assert_eq!(store.session(session).unwrap().tokens, tokens_before);
}

#[test]
fn tool_ledger_open_close_and_replay() {
    use superfluid_daemon::codec::MockChatCodec;
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let (session, call_id) = {
        let store = SessionStore::open(&wal).unwrap();
        let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None))
            .expect("spawn");
        let d = Daemon::new(store, host, Box::new(MockChatCodec), 8);
        let session = d.create(None, GenParams::default()).unwrap();
        d.append_message(session, 1, "use the tool".into()).unwrap();
        let call = {
            let store = d.store();
            let mut store = store.lock().unwrap();
            store
                .commit_tool_use(session, "get_weather".into(), r#"{"city":"Paris"}"#.into())
                .unwrap()
        };
        let open = d.open_tool_calls(session).unwrap();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].0, call.event_id);
        assert_eq!(open[0].1, "get_weather");

        assert!(matches!(
            d.append_tool_result(session, 9999, "x".into()),
            Err(superfluid_daemon::DaemonError::UnknownToolCall(9999))
        ));
        d.append_tool_result(session, call.event_id, r#"{"temp":"18C"}"#.into())
            .unwrap();
        assert!(d.open_tool_calls(session).unwrap().is_empty());
        assert!(matches!(
            d.append_tool_result(session, call.event_id, "again".into()),
            Err(superfluid_daemon::DaemonError::UnknownToolCall(_))
        ));
        (session, call.event_id)
    };
    let store = SessionStore::open(&wal).unwrap();
    assert!(store.session(session).unwrap().open_tool_calls.is_empty());
    let _ = call_id;
}

#[test]
fn open_tool_call_survives_restart() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let session = {
        let mut store = SessionStore::open(&wal).unwrap();
        let session = store.create(None, GenParams::default()).unwrap();
        store
            .commit_tool_use(session, "search".into(), r#"{"q":"x"}"#.into())
            .unwrap();
        session
    };
    let store = SessionStore::open(&wal).unwrap();
    let open = &store.session(session).unwrap().open_tool_calls;
    assert_eq!(open.len(), 1);
    assert_eq!(open.values().next().unwrap().0, "search");
}

#[test]
fn a_cancel_that_lands_while_the_lane_finishes_spares_the_next_generation() {
    let store = SessionStore::open(&test_dir().join("wal.log")).expect("open store");
    let slow = EngineConfig {
        tick_delay: std::time::Duration::from_millis(200),
        ..EngineConfig::default()
    };
    let host = EngineHost::spawn(move || (MockEngine::new(slow), None)).expect("spawn");
    let d = Daemon::new(store, host, Box::new(MockCodec), 8);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..16).collect()).unwrap();

    let (cancels, active) = (d.cancel_registry(), d.active_registry());
    let mut produced = 0;
    let mut landed = false;
    let out = d
        .generate_streaming(s, 4, |ev| {
            if let EventBody::Generated { span, .. } = &ev.body {
                produced += span.len();
            }
            if produced >= 4 && !landed {
                let mut c = cancels.lock().unwrap();
                if active.lock().unwrap().contains(&s) {
                    c.insert(s);
                    landed = true;
                }
            }
            Ok(())
        })
        .unwrap();
    assert!(landed, "the cancel landed while the lane was finishing");
    assert_ne!(out.finish, finish::CANCELLED, "a finishing lane is past cancelling");

    let next = d.generate(s, 4).unwrap();
    assert_ne!(next.finish, finish::CANCELLED, "the stale cancel stopped the next generation");
    assert_eq!(next.tokens_generated, 4);
}

#[test]
fn cancel_requires_active_generation() {
    let dir = test_dir();
    let socket = dir.join("superfluid.sock");
    let daemon = Arc::new(mock_daemon(&dir.join("wal.log")));
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let serving = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = api::serve(listener, serving);
    });
    let mut client = api::NativeClient::connect(&socket).expect("connect");
    let session = match client
        .request(&api::Request::Create {
            parent: None,
            params: GenParams::default(),
        })
        .unwrap()
    {
        api::Response::Created { session } => session,
        other => panic!("unexpected {other:?}"),
    };
    client
        .request(&api::Request::Append {
            session,
            text: None,
            span: (0..8).collect(),
        })
        .unwrap();
    match client.request(&api::Request::Cancel { session }).unwrap() {
        api::Response::Err { message } => assert!(message.contains("no generation")),
        other => panic!("unexpected {other:?}"),
    }
    match client.generate_stream(session, 4, |_| {}).unwrap() {
        api::Response::Generated {
            tokens_generated,
            finish,
            ..
        } => {
            assert_eq!(tokens_generated, 4);
            assert_ne!(finish, finish::CANCELLED);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn cancel_of_queued_job_completes_locally() {
    let d = Arc::new(mock_daemon_with_lanes(&test_dir().join("wal.log"), 1));
    let a = d.create(None, GenParams::default()).unwrap();
    let b = d.create(None, GenParams::default()).unwrap();
    d.append(a, None, (0..16).collect()).unwrap();
    d.append(b, None, (500..516).collect()).unwrap();

    let cancels = d.cancel_registry();
    let (a_started_tx, a_started_rx) = std::sync::mpsc::channel();
    let ha = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || {
            let mut signalled = false;
            d.generate_streaming(a, 2000, |_| {
                if !signalled {
                    signalled = true;
                    a_started_tx.send(()).unwrap();
                }
                Ok(())
            })
            .unwrap()
        })
    };
    a_started_rx.recv().unwrap();
    let hb = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || d.generate(b, 64).unwrap())
    };
    while !d.active_registry().lock().unwrap().contains(&b) {
        std::thread::yield_now();
    }
    cancels.lock().unwrap().insert(b);
    let out_b = hb.join().unwrap();
    assert_eq!(out_b.finish, finish::CANCELLED);
    assert_eq!(out_b.tokens_generated, 0, "queued lane never decoded");
    let out_a = ha.join().unwrap();
    assert_eq!(out_a.tokens_generated, 2000, "A unaffected by B's cancel");
    assert_ne!(out_a.finish, finish::CANCELLED);
}

#[test]
fn subscriber_loss_records_cancellation() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let session = {
        let d = mock_daemon(&wal);
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, (0..16).collect()).unwrap();
        let mut n = 0;
        let _ = d.generate_streaming(session, 3000, |_| {
            n += 1;
            if n >= 2 {
                Err(superfluid_daemon::DaemonError::Protocol("client gone"))
            } else {
                Ok(())
            }
        });
        while d.active_registry().lock().unwrap().contains(&session) {
            std::thread::yield_now();
        }
        session
    };
    let store = SessionStore::open(&wal).unwrap();
    let s = store.session(session).unwrap();
    let last_gen = s
        .events
        .iter()
        .rev()
        .find_map(|e| match &e.body {
            EventBody::Generated { finish, .. } => Some(*finish),
            _ => None,
        })
        .expect("generated events exist");
    assert!(
        last_gen == finish::CANCELLED || last_gen == finish::LENGTH,
        "subscriber loss must leave a terminal event, not an unexplained \
         resumable history (got finish={last_gen})"
    );
}

fn kv_only_config() -> EngineConfig {
    let mut cfg = EngineConfig::default();
    cfg.spaces
        .retain(|s| s.kind == superfluid_abi::space_kind::PAGED_TOKEN_KV);
    cfg
}

fn kv_only_daemon(wal: &std::path::Path, park: Option<&std::path::Path>) -> Daemon {
    let store = SessionStore::open(wal).expect("open store");
    let host = EngineHost::spawn(|| (MockEngine::new(kv_only_config()), None)).expect("spawn");
    Daemon::with_park(
        store,
        host,
        Box::new(MockCodec),
        8,
        park.map(|p| p.to_path_buf()),
    )
}

#[test]
fn park_resume_survives_daemon_restart() {
    let run_life1 = |dir: &std::path::Path, park: Option<&std::path::Path>| -> u64 {
        let d = kv_only_daemon(&dir.join("wal.log"), park);
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, (0..32).collect()).unwrap();
        let out = d.generate(session, 8).unwrap();
        assert_eq!(out.warm_prefix, 0, "first generation is cold");
        assert_eq!(out.tokens_generated, 8);
        session
    };

    let warm_dir = test_dir();
    let park = warm_dir.join("park");
    let session = run_life1(&warm_dir, Some(&park));
    assert!(
        park.join(format!("{session}.park")).exists(),
        "retire parked a sealed artifact"
    );

    let cold_dir = test_dir();
    let cold_session = run_life1(&cold_dir, None);

    let warm = {
        let d = kv_only_daemon(&warm_dir.join("wal.log"), Some(&park));
        d.generate(session, 4).unwrap()
    };
    let cold = {
        let d = kv_only_daemon(&cold_dir.join("wal.log"), None);
        d.generate(cold_session, 4).unwrap()
    };
    assert!(
        warm.warm_prefix >= 16,
        "restart resumed warm from the park artifact (got {})",
        warm.warm_prefix
    );
    assert_eq!(cold.warm_prefix, 0, "control restart is cold");
    assert_eq!(
        generated_tokens(&warm.events),
        generated_tokens(&cold.events),
        "resume-from-park continues the exact stream"
    );
}

#[test]
fn corrupt_park_artifact_resumes_cold() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let park = dir.join("park");
    let session = {
        let d = kv_only_daemon(&wal, Some(&park));
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, (0..32).collect()).unwrap();
        d.generate(session, 8).unwrap();
        session
    };
    let artifact = park.join(format!("{session}.park"));
    let mut bytes = std::fs::read(&artifact).unwrap();
    assert!(bytes.len() > 64);
    let mid = 24 + (bytes.len() - 24) / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&artifact, &bytes).unwrap();

    let d = kv_only_daemon(&wal, Some(&park));
    let out = d.generate(session, 4).unwrap();
    assert_eq!(out.warm_prefix, 0, "corrupt artifact resumes cold");
    assert_eq!(out.tokens_generated, 4);
}

#[test]
fn an_ephemeral_session_writes_no_park_artifact() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let park = dir.join("park");
    let (durable, ephemeral) = {
        let store = SessionStore::open(&wal).unwrap();
        let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
        let d = Daemon::with_park(store, host, Box::new(MockCodec), 8, Some(park.clone()));

        let durable = d.create(None, GenParams::default()).unwrap();
        d.append(durable, None, (0..32).collect()).unwrap();
        d.generate(durable, 8).unwrap();

        let ephemeral = d.create(None, GenParams::default()).unwrap();
        d.append(ephemeral, None, (0..32).collect()).unwrap();
        let extras = superfluid_daemon::scheduler::GenExtras {
            ephemeral: true,
            ..Default::default()
        };
        let out = d
            .generate_streaming_ex(ephemeral, 8, extras, |_| Ok(()))
            .unwrap();
        assert_eq!(out.tokens_generated, 8, "generation itself is unchanged");
        (durable, ephemeral)
    };

    assert!(
        park.join(format!("{durable}.park")).exists(),
        "a session that can be resumed by id still parks"
    );
    assert!(
        !park.join(format!("{ephemeral}.park")).exists(),
        "nothing will ever ask for this session by id — no artifact"
    );
}

#[test]
fn recurrent_space_parks_every_space_and_resumes_by_adoption() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let park = dir.join("park");
    let session = {
        let store = SessionStore::open(&wal).unwrap();
        let host =
            EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
        let d = Daemon::with_park(store, host, Box::new(MockCodec), 8, Some(park.clone()));
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, (0..32).collect()).unwrap();
        d.generate(session, 8).unwrap();
        session
    };
    assert!(park.join(format!("{session}.park")).exists());

    let bytes = std::fs::read(park.join(format!("{session}.park"))).unwrap();
    assert_eq!(&bytes[..8], b"BRTPARK3");
    let parsed = superfluid_daemon::park::read(&park, session).expect("parses");
    assert!(parsed.multi_space);
    assert_eq!(parsed.spaces.len(), 2);
    assert_eq!(parsed.covered, 32, "the blob's last interval boundary ≤ 39");

    let store = SessionStore::open(&wal).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    let d = Daemon::with_park(store, host, Box::new(MockCodec), 8, Some(park));
    let stats = d.sched_stats();
    let out = d.generate(session, 4).unwrap();
    assert_eq!(out.warm_prefix, 32, "adopted at the common boundary; the gap re-prefills");
    assert_eq!(out.tokens_generated, 4);
    assert_eq!(stats.resumes.load(std::sync::atomic::Ordering::Relaxed), 1);
    let fresh = d.create(None, GenParams::default()).unwrap();
    d.append(fresh, None, (0..32).collect()).unwrap();
    d.generate(fresh, 8).unwrap();
    d.generate(fresh, 4).unwrap();
    let store = d.store();
    let store = store.lock().unwrap();
    assert_eq!(store.session(session).unwrap().tokens, store.session(fresh).unwrap().tokens);
}

fn unservable_recurrent_config() -> EngineConfig {
    let mut cfg = EngineConfig::default();
    for s in cfg.spaces.iter_mut() {
        if s.kind == superfluid_abi::space_kind::RECURRENT_BLOB {
            s.flags |= superfluid_abi::space_flag::OPS_UNAVAILABLE;
        }
    }
    cfg
}

#[test]
fn unservable_space_forces_cold_and_refuses_park() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let park = dir.join("park");
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host =
        EngineHost::spawn(|| (MockEngine::new(unservable_recurrent_config()), None)).unwrap();
    let d = Daemon::with_park(store, host, Box::new(MockCodec), 8, Some(park.clone()));
    let stats = d.sched_stats();

    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, None, (0..32).collect()).unwrap();
    let first = d.generate(session, 8).unwrap();
    assert_eq!(first.warm_prefix, 0);
    assert_eq!(first.tokens_generated, 8);
    assert!(
        !park.join(format!("{session}.park")).exists(),
        "no partial artifact may be written"
    );
    assert_eq!(stats.unservable_park_refusals.load(Ordering::Relaxed), 1);
    assert_eq!(stats.unservable_cold_admissions.load(Ordering::Relaxed), 1);

    let second = d.generate(session, 4).unwrap();
    assert_eq!(second.warm_prefix, 0, "warm seed refused: cold by construction");
    assert_eq!(second.tokens_generated, 4);
    assert_eq!(stats.unservable_cold_admissions.load(Ordering::Relaxed), 2);
    assert_eq!(stats.unservable_park_refusals.load(Ordering::Relaxed), 2);

    let fresh = d.create(None, GenParams::default()).unwrap();
    d.append(fresh, None, (0..32).collect()).unwrap();
    let f1 = d.generate(fresh, 8).unwrap();
    let f2 = d.generate(fresh, 4).unwrap();
    let toks = |s: u64| {
        let store = d.store();
        let store = store.lock().unwrap();
        store.session(s).unwrap().tokens.clone()
    };
    assert_eq!(f1.tokens_generated + f2.tokens_generated, 12);
    assert_eq!(toks(session), toks(fresh));
}

#[test]
fn subscription_streams_backlog_then_live() {
    let dir = test_dir();
    let socket = dir.join("superfluid.sock");
    let daemon = Arc::new(mock_daemon(&dir.join("wal.log")));
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let serving = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = api::serve(listener, serving);
    });

    let session = daemon.create(None, GenParams::default()).unwrap();
    daemon.append(session, None, (0..24).collect()).unwrap();
    let first = daemon.generate(session, 4).unwrap();
    let backlog_len = daemon.read(session, 0).unwrap().len();
    assert!(backlog_len > first.events.len());

    let mut obs = api::NativeClient::connect(&socket).expect("connect observer");
    let sub = obs.subscribe(session, 0, false).unwrap();
    let mut seen: Vec<u64> = Vec::new();
    for _ in 0..backlog_len {
        match obs.next_frame().unwrap() {
            api::Response::SubEvent { sub: s, event } => {
                assert_eq!(s, sub);
                seen.push(event.event_id);
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(
        seen,
        (0..backlog_len as u64).collect::<Vec<_>>(),
        "backlog is contiguous from the cursor"
    );

    let second = daemon.generate(session, 4).unwrap();
    for expect in &second.events {
        match obs.next_frame().unwrap() {
            api::Response::SubEvent { sub: s, event } => {
                assert_eq!(s, sub);
                assert_eq!(event.event_id, expect.event_id, "live events in commit order");
                assert_eq!(event.body, expect.body);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    let mut got_ack = false;
    let mut got_end = false;
    obs.request(&api::Request::Unsubscribe { sub }).map(|r| collect_unsub(r, sub, &mut got_ack, &mut got_end)).unwrap();
    collect_unsub(obs.next_frame().unwrap(), sub, &mut got_ack, &mut got_end);
    assert!(got_ack && got_end, "both the ack and the reasoned terminator arrived");
}

fn collect_unsub(r: api::Response, sub: u64, got_ack: &mut bool, got_end: &mut bool) {
    match r {
        api::Response::Unsubscribed => *got_ack = true,
        api::Response::SubEnded { sub: s, reason } => {
            assert_eq!(s, sub);
            assert_eq!(reason, "unsubscribed");
            *got_end = true;
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn provisional_deltas_precede_and_reconcile_with_commits() {
    let dir = test_dir();
    let socket = dir.join("superfluid.sock");
    let daemon = {
        let store = SessionStore::open(&dir.join("wal.log")).unwrap();
        let cfg = EngineConfig {
            vocab: 512,
            ..EngineConfig::default()
        };
        let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).unwrap();
        Arc::new(Daemon::new(store, host, Box::new(MockCodec), 8))
    };
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let serving = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = api::serve(listener, serving);
    });

    let session = daemon.create(None, GenParams::default()).unwrap();
    daemon.append(session, None, (0..16).collect()).unwrap();
    let cursor = daemon.read(session, 0).unwrap().len() as u64;

    let mut obs = api::NativeClient::connect(&socket).expect("connect observer");
    let sub = obs.subscribe(session, cursor, true).unwrap();

    let out = daemon.generate(session, 8).unwrap();
    let committed_texts: Vec<String> = out
        .events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Generated { text, .. } if !text.is_empty() => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert!(
        !committed_texts.is_empty(),
        "chosen inputs must produce decodable text (tune the prompt if not)"
    );

    let mut provisional: Vec<String> = Vec::new();
    let mut committed: Vec<String> = Vec::new();
    let mut events_seen = 0;
    while events_seen < out.events.len() {
        match obs.next_frame().unwrap() {
            api::Response::SubProvisional {
                sub: s,
                channel,
                text,
            } => {
                assert_eq!(s, sub);
                assert_eq!(channel, 0, "mock generates on the TEXT channel");
                provisional.push(text);
            }
            api::Response::SubEvent { sub: s, event } => {
                assert_eq!(s, sub);
                events_seen += 1;
                if let EventBody::Generated { text, .. } = &event.body {
                    if !text.is_empty() {
                        committed.push(text.clone());
                    }
                }
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(
            provisional.len() >= committed.len(),
            "a committed slice arrived before its provisional preview"
        );
    }
    assert_eq!(
        provisional.concat(),
        committed_texts.concat(),
        "provisional text reconciles exactly with the committed record"
    );
    assert!(
        provisional.len() >= committed_texts.len(),
        "token-cadence deltas are at least as fine as tick-cadence commits"
    );
    assert_eq!(committed, committed_texts);
}

#[test]
fn adaptive_prefill_budget_grows_on_short_ticks() {
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    let d = Daemon::with_options(
        store,
        host,
        Box::new(MockCodec),
        superfluid_daemon::DaemonOptions { prefill_budget: 0, ..Default::default() },
    )
    .unwrap();
    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, None, (0..3000).map(|i| 0x100 + (i % 200)).collect()).unwrap();
    let _ = d.generate(session, 4).unwrap();
    let live = d
        .sched_stats()
        .prefill_budget_live
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(live > 1024, "adaptive budget should grow on short ticks, got {live}");
}

#[test]
fn in_flight_tick_gauge_is_set_during_a_tick_and_cleared_after() {
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| {
        (
            MockEngine::new(EngineConfig {
                tick_delay: std::time::Duration::from_millis(300),
                ..Default::default()
            }),
            None,
        )
    })
    .unwrap();
    let d = Arc::new(Daemon::with_options(store, host, Box::new(MockCodec), Default::default()).unwrap());
    let stats = d.sched_stats();
    assert!(stats.tick_in_flight().is_none(), "idle: no tick in flight");
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..64).map(|i| 0x100 + i).collect()).unwrap();
    let dd = Arc::clone(&d);
    let t = std::thread::spawn(move || dd.generate(s, 40).unwrap());
    let mut seen = None;
    for _ in 0..200 {
        if let Some(t) = stats.tick_in_flight() {
            seen = Some(t);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let tick = seen.expect("a tick was in flight at some point during a 300 ms-tick generation");
    assert!(superfluid_daemon::scheduler::uptime_ms() >= tick.since_ms, "stamp is on the uptime clock");
    assert!(
        tick.prefill_tokens > 0 || tick.decode_lanes > 0,
        "an in-flight tick carries a plan: {tick:?}"
    );
    t.join().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while stats.tick_in_flight().is_some() {
        assert!(std::time::Instant::now() < deadline, "in-flight stamp never cleared after the request");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn tick_target_is_short_while_lane_slots_are_free() {
    let target_ms = |max_lanes: usize| {
        let store = SessionStore::open(&test_dir().join("wal.log")).unwrap();
        let host =
            EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
        let d = Arc::new(
            Daemon::with_options(
                store,
                host,
                Box::new(MockCodec),
                superfluid_daemon::DaemonOptions { max_lanes, ..Default::default() },
            )
            .unwrap(),
        );
        let a = d.create(None, GenParams::default()).unwrap();
        d.append(a, None, (0..16).map(|i| 0x100 + i).collect()).unwrap();
        let da = Arc::clone(&d);
        let t = std::thread::spawn(move || da.generate(a, 2_000).unwrap());
        let mut seen = std::collections::BTreeSet::new();
        while !t.is_finished() {
            seen.insert(d.sched_stats().tick_target_live_ms.load(std::sync::atomic::Ordering::Relaxed));
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        t.join().unwrap();
        seen
    };
    assert!(
        target_ms(1).contains(&2000),
        "one lane fills a one-lane scheduler: the configured target while it decodes"
    );
    assert_eq!(
        target_ms(2).into_iter().filter(|&ms| ms != 0).collect::<Vec<_>>(),
        vec![250],
        "one lane leaves a slot free: the open target throughout"
    );
}

#[test]
fn decode_grant_scales_while_prefill_contends() {
    let store = SessionStore::open(&test_dir().join("wal.log")).unwrap();
    let host =
        EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    let d = Arc::new(Daemon::with_options(
        store,
        host,
        Box::new(MockCodec),
        superfluid_daemon::DaemonOptions {
            prefill_budget: 256,
            ..Default::default()
        },
    )
    .unwrap());
    let a = d.create(None, GenParams::default()).unwrap();
    d.append(a, None, (0..16).map(|i| 0x100 + i).collect()).unwrap();
    let da = Arc::clone(&d);
    let ta = std::thread::spawn(move || da.generate(a, 3_000).unwrap());
    std::thread::sleep(std::time::Duration::from_millis(100));
    let solo = d
        .sched_stats()
        .decode_grant_live
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        solo >= 32,
        "decode-only grant must be at least TICK_DECODE, got {solo}"
    );
    let b = d.create(None, GenParams::default()).unwrap();
    d.append(b, None, (0..3_000).map(|i| 0x100 + (i % 200)).collect()).unwrap();
    let db = Arc::clone(&d);
    let tb = std::thread::spawn(move || db.generate(b, 4).unwrap());
    let mut max_grant = 0u64;
    while !tb.is_finished() {
        max_grant = max_grant.max(
            d.sched_stats()
                .decode_grant_live
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        std::thread::yield_now();
    }
    tb.join().unwrap();
    ta.join().unwrap();
    assert!(
        max_grant > 32,
        "prefill-carrying ticks should scale the decode grant above TICK_DECODE, got {max_grant}"
    );
}

#[test]
fn append_message_rejects_unknown_role() {
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    let d = Daemon::new(
        store,
        host,
        Box::new(superfluid_daemon::codec::MockChatCodec),
        8,
    );
    let session = d.create(None, GenParams::default()).unwrap();
    let err = d.append_message(session, 99, "hi".into()).unwrap_err();
    assert!(
        matches!(err, superfluid_daemon::DaemonError::UnknownRole(99)),
        "got {err:?}"
    );
    assert_eq!(d.read(session, 1).unwrap().len(), 0);
    d.append_message(session, superfluid_daemon::wal::role::USER, "hi".into())
        .unwrap();
}

#[test]
fn behavior_fingerprint_is_recorded_on_change_only() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let fp_events = |d: &Daemon, s: u64| -> Vec<u64> {
        d.read(s, 0)
            .unwrap()
            .iter()
            .filter_map(|e| match &e.body {
                EventBody::GenerationFingerprint { digest } => Some(*digest),
                _ => None,
            })
            .collect()
    };
    let session = {
        let store = SessionStore::open(&wal).unwrap();
        let host =
            EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
        let d = Daemon::new(store, host, Box::new(superfluid_daemon::codec::MockChatCodec), 8);
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, (0..24).collect()).unwrap();
        d.generate(session, 4).unwrap();
        d.generate(session, 4).unwrap();
        let fps = fp_events(&d, session);
        assert_eq!(fps.len(), 1, "stable codec records exactly once");
        session
    };
    {
        let store = SessionStore::open(&wal).unwrap();
        let host =
            EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
        let d = Daemon::new(store, host, Box::new(superfluid_daemon::codec::MockChatCodec), 8);
        d.generate(session, 4).unwrap();
        assert_eq!(fp_events(&d, session).len(), 1, "restart does not re-record");
    }
    {
        let mut store = SessionStore::open(&wal).unwrap();
        let ev = store
            .commit_generation_fingerprint(session, 0xDEAD_BEEF)
            .unwrap();
        assert!(ev.is_some(), "changed digest commits");
        let again = store
            .commit_generation_fingerprint(session, 0xDEAD_BEEF)
            .unwrap();
        assert!(again.is_none(), "unchanged digest is change-detected");
    }
}

fn process_daemon(wal: &std::path::Path) -> (Daemon, u32) {
    let host = EngineHost::spawn_process(superfluid_daemon::WorkerSpec {
        bin: env!("CARGO_BIN_EXE_superfluid-workerd").into(),
        args: vec!["--engine".into(), "mock".into()],
        stderr: None,
    })
    .expect("spawn worker process");
    let pid = host.worker_pid().expect("process backend has a pid");
    let store = SessionStore::open(wal).expect("open store");
    (Daemon::new(store, host, Box::new(MockCodec), 8), pid)
}

#[test]
fn process_worker_matches_in_process_output() {
    let run = |d: &Daemon| -> Vec<u32> {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..24).collect()).unwrap();
        generated_tokens(&d.generate(s, 8).unwrap().events)
    };
    let (proc_daemon, _pid) = process_daemon(&test_dir().join("wal.log"));
    let via_process = run(&proc_daemon);
    let via_thread = run(&mock_daemon(&test_dir().join("wal.log")));
    assert_eq!(via_process, via_thread, "the boundary changed an output");
    assert!(!via_process.is_empty());
}

#[test]
fn worker_crash_fails_typed_then_respawns() {
    let dir = test_dir();
    let (d, pid) = process_daemon(&dir.join("wal.log"));
    let d = Arc::new(d);
    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, None, (0..24).collect()).unwrap();

    let (tx, rx) = std::sync::mpsc::channel();
    let gen = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || {
            let mut signalled = false;
            d.generate_streaming(session, 4096, |_| {
                if !signalled {
                    signalled = true;
                    let _ = tx.send(());
                }
                Ok(())
            })
        })
    };
    rx.recv().expect("generation started");
    let _ = std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .expect("kill");
    let result = gen.join().expect("generation thread");
    assert!(result.is_err(), "a murdered engine fails typed, not silently");

    let out = d.generate(session, 8).unwrap();
    assert_eq!(out.tokens_generated, 8);
    assert!(!generated_tokens(&out.events).is_empty());
}

#[test]
fn pressure_relief_evicts_and_park_keeps_it_lossless() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let park = dir.join("park");
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let cfg = EngineConfig {
        pool_bytes: 32 * 1024,
        ..kv_only_config()
    };
    let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).expect("spawn");
    let d = Daemon::with_options(
        store,
        host,
        Box::new(MockCodec),
        superfluid_daemon::DaemonOptions {
            max_lanes: 2,
            park_dir: Some(park),
            ..Default::default()
        },
    )
    .unwrap();

    let first = d.create(None, GenParams::default()).unwrap();
    d.append(first, None, (0..96).collect()).unwrap();
    assert_eq!(d.generate(first, 16).unwrap().tokens_generated, 16);
    for i in 1..8u32 {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (i * 100..i * 100 + 96).collect()).unwrap();
        assert_eq!(d.generate(s, 16).unwrap().tokens_generated, 16);
    }

    let stats = d.sched_stats();
    assert!(
        stats.pressure_evictions.load(Ordering::Relaxed) > 0,
        "the watermark policy ran at least once"
    );
    assert!(stats.pressure_bytes_evicted.load(Ordering::Relaxed) > 0);

    let out = d.generate(first, 8).unwrap();
    assert!(
        out.warm_prefix >= 16,
        "evicted-under-pressure session resumes warm (got {})",
        out.warm_prefix
    );
}

#[test]
fn lossy_park_demotes_and_resumes_warm() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let park = dir.join("park");
    let mk = |park_lossy: bool| -> Daemon {
        let store = SessionStore::open(&wal).unwrap();
        let host = EngineHost::spawn(|| (MockEngine::new(kv_only_config()), None)).unwrap();
        Daemon::with_options(
            store,
            host,
            Box::new(MockCodec),
            superfluid_daemon::DaemonOptions {
                max_lanes: 8,
                park_dir: Some(park.clone()),
                park_lossy,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let session = {
        let d = mk(true);
        let session = d.create(None, GenParams::default()).unwrap();
        d.append(session, None, (0..96).collect()).unwrap();
        assert_eq!(d.generate(session, 16).unwrap().tokens_generated, 16);
        session
    };
    let lossy = superfluid_daemon::park::read(&park, session).expect("artifact");
    assert_eq!(lossy.encoding, superfluid_abi::encoding::Q8, "lossy tier recorded");

    let lossless_size = {
        {
            let d = mk(false);
            d.generate(session, 16).unwrap();
        }
        superfluid_daemon::park::read(&park, session).unwrap().sealed.len()
    };
    assert!(
        lossy.sealed.len() < lossless_size,
        "the lossy tier is smaller ({} vs {})",
        lossy.sealed.len(),
        lossless_size
    );

    {
        let d = mk(true);
        d.generate(session, 8).unwrap();
    }
    let d = mk(true);
    let out = d.generate(session, 8).unwrap();
    assert!(
        out.warm_prefix >= 16,
        "restart resumes warm from the LOSSY artifact (got {})",
        out.warm_prefix
    );
}

#[test]
fn fork_inherits_prefix_by_id_and_siblings_diverge() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let (root, child, fork_at) = {
        let d = mock_daemon(&wal);
        let root = d.create(None, GenParams::default()).unwrap();
        d.append(root, None, (0..32).collect()).unwrap();
        d.generate(root, 8).unwrap();
        let root_events = d.read(root, 0).unwrap();
        let fork_at = root_events.len() as u64;
        let child = d.fork(root, fork_at, None).unwrap();
        assert_ne!(child, root);

        let child_events = d.read(child, 0).unwrap();
        assert_eq!(child_events.len(), fork_at as usize + 1);
        for (a, b) in child_events.iter().zip(root_events.iter()) {
            assert_eq!(a.event_id, b.event_id);
            assert_eq!(a.body, b.body);
        }
        assert!(matches!(
            child_events[fork_at as usize].body,
            EventBody::Forked { parent, fork_at: f, .. } if parent == root && f == fork_at
        ));
        {
            let store = d.store();
            let store = store.lock().unwrap();
            assert_eq!(store.session(child).unwrap().tokens, store.session(root).unwrap().tokens);
            assert_eq!(store.session(child).unwrap().base, fork_at);
            assert_eq!(store.session(child).unwrap().parent, Some(root));
        }
        assert_eq!(
            d.read(child, 1).unwrap()[..(fork_at - 1) as usize].to_vec(),
            d.read(root, 1).unwrap()[..(fork_at - 1) as usize].to_vec()
        );

        let c = d.append(child, None, vec![100, 101]).unwrap();
        let p = d.append(root, None, vec![200, 201]).unwrap();
        assert_eq!(c.event_id, fork_at + 1);
        assert_eq!(p.event_id, fork_at);
        {
            let store = d.store();
            let store = store.lock().unwrap();
            let ct = &store.session(child).unwrap().tokens;
            let pt = &store.session(root).unwrap().tokens;
            assert_eq!(&ct[ct.len() - 2..], &[100, 101]);
            assert_eq!(&pt[pt.len() - 2..], &[200, 201]);
            assert_eq!(ct[..ct.len() - 2], pt[..pt.len() - 2]);
        }
        let tree = d.tree();
        assert!(tree.contains(&(root, None, 0)));
        assert!(tree.contains(&(child, Some(root), fork_at)));
        (root, child, fork_at)
    };
    let store = SessionStore::open(&wal).unwrap();
    let c = store.session(child).unwrap();
    let r = store.session(root).unwrap();
    assert_eq!(c.base, fork_at);
    assert_eq!(c.parent, Some(root));
    assert_eq!(c.tokens[..c.tokens.len() - 2], r.tokens[..r.tokens.len() - 2]);
    assert_eq!(&c.tokens[c.tokens.len() - 2..], &[100, 101]);
    assert_eq!(c.events.len() as u64, fork_at + 3);
    assert_eq!(c.events[fork_at as usize + 1].event_id, fork_at + 1);
    assert!(matches!(c.events[fork_at as usize + 1].body, EventBody::Appended { .. }));
    assert!(matches!(c.events[fork_at as usize + 2].body, EventBody::EpochBump));
}

#[test]
fn fork_refuses_bad_points_and_continues_unfinished_turns() {
    let dir = test_dir();
    let d = mock_daemon(&dir.join("wal.log"));
    let root = d.create(None, GenParams::default()).unwrap();
    d.append(root, None, (0..16).collect()).unwrap();
    let out = d.generate(root, 40).unwrap();
    assert!(out.events.len() >= 2);
    let events = d.read(root, 0).unwrap();
    let first_gen = events
        .iter()
        .position(|e| matches!(e.body, EventBody::Generated { finish, .. } if finish == finish::NONE))
        .expect("an unfinished slice");
    assert!(matches!(
        d.fork(root, 0, None),
        Err(superfluid_daemon::DaemonError::ForkPoint { fork_at: 0, .. })
    ));
    let too_far = events.len() as u64 + 1;
    assert!(matches!(
        d.fork(root, too_far, None),
        Err(superfluid_daemon::DaemonError::ForkPoint { .. })
    ));
    assert!(matches!(
        d.fork(9999, 1, None),
        Err(superfluid_daemon::DaemonError::UnknownSession(9999))
    ));
    assert!(d.fork(root, 2, None).is_ok());
    let mid = d.fork(root, first_gen as u64 + 1, None).unwrap();
    let cont = d.generate(mid, 8).unwrap();
    assert_eq!(cont.tokens_generated, 8);
    let store = d.store();
    let store = store.lock().unwrap();
    let parent_tokens = &store.session(root).unwrap().tokens;
    let child_tokens = &store.session(mid).unwrap().tokens;
    assert_eq!(child_tokens[..], parent_tokens[..child_tokens.len()]);
}

#[test]
fn fork_does_not_inherit_the_tool_ledger() {
    use superfluid_daemon::codec::MockChatCodec;
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host =
        EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    let d = Daemon::new(store, host, Box::new(MockChatCodec), 8);
    let root = d.create(None, GenParams::default()).unwrap();
    d.append_message(root, 1, "use the tool".into()).unwrap();
    let call = {
        let store = d.store();
        let mut store = store.lock().unwrap();
        store
            .commit_tool_use(root, "get_weather".into(), r#"{"city":"Paris"}"#.into())
            .unwrap()
    };
    let fork_at = d.read(root, 0).unwrap().len() as u64;
    let child = d.fork(root, fork_at, None).unwrap();
    assert!(d.open_tool_calls(child).unwrap().is_empty());
    assert!(matches!(
        d.append_tool_result(child, call.event_id, "x".into()),
        Err(superfluid_daemon::DaemonError::InheritedToolCall(id)) if id == call.event_id
    ));
    d.append_tool_result(root, call.event_id, r#"{"temp":"18C"}"#.into())
        .unwrap();
    assert!(d.open_tool_calls(root).unwrap().is_empty());
}

#[test]
fn forked_child_admits_warm_from_the_shared_prefix() {
    let dir = test_dir();
    let d = kv_only_daemon(&dir.join("wal.log"), None);
    let root = d.create(None, GenParams::default()).unwrap();
    d.append(root, None, (0..48).collect()).unwrap();
    let r1 = d.generate(root, 8).unwrap();
    assert_eq!(r1.warm_prefix, 0);
    let fork_at = d.read(root, 0).unwrap().len() as u64;
    let child = d.fork(root, fork_at, None).unwrap();
    d.append(root, None, vec![7, 8, 9]).unwrap();
    d.append(child, None, vec![7, 8, 9]).unwrap();
    let r2 = d.generate(root, 8).unwrap();
    let c2 = d.generate(child, 8).unwrap();
    assert!(r2.warm_prefix > 0, "the parent itself resumes warm");
    assert!(c2.warm_prefix >= r2.warm_prefix, "the fork is at least as warm as its parent");
    let store = d.store();
    let store = store.lock().unwrap();
    assert_eq!(store.session(child).unwrap().tokens, store.session(root).unwrap().tokens);
}

#[test]
fn meta_is_versioned_lww_and_archive_is_soft() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let s = {
        let d = mock_daemon(&wal);
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..8).collect()).unwrap();
        assert_eq!(d.set_meta(s, 0, Some("first".into()), None).unwrap(), 1);
        assert!(matches!(
            d.set_meta(s, 0, Some("stale".into()), None),
            Err(superfluid_daemon::DaemonError::MetaConflict { expected: 0, actual: 1, .. })
        ));
        assert_eq!(d.set_meta(s, 1, None, Some(true)).unwrap(), 2);
        assert!(d.sessions(false).is_empty(), "archived: hidden from the projection");
        let all = d.sessions(true);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].title.as_deref(), Some("first"));
        assert!(all[0].archived);
        assert_eq!(all[0].meta_version, 2);
        assert_eq!(all[0].tokens, 8);
        d.append(s, None, vec![9]).unwrap();
        assert_eq!(d.set_meta(s, 2, None, Some(false)).unwrap(), 3);
        assert_eq!(d.sessions(false).len(), 1);
        s
    };
    let store = SessionStore::open(&wal).unwrap();
    let st = store.session(s).unwrap();
    assert_eq!(st.meta_version, 3);
    assert_eq!(st.title.as_deref(), Some("first"));
    assert!(!st.archived);
    assert_eq!(st.tokens.len(), 9);
}

#[test]
fn rebase_drops_and_replaces_middles_without_touching_the_parent() {
    use superfluid_daemon::codec::MockChatCodec;
    use superfluid_daemon::wal::{role, RebaseEdit};
    let dir = test_dir();
    let mk = |wal: &std::path::Path| {
        let store = SessionStore::open(wal).unwrap();
        let host =
            EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
        Daemon::new(store, host, Box::new(MockChatCodec), 8)
    };
    let d = mk(&dir.join("wal.log"));
    let p = d.create(None, GenParams::default()).unwrap();
    d.append_message(p, role::SYSTEM, "sys".into()).unwrap();
    d.append_message(p, role::USER, "q1".into()).unwrap();
    d.append_message(p, role::ASSISTANT, "a1".into()).unwrap();
    d.append_message(p, role::USER, "q2".into()).unwrap();
    let call = {
        let store = d.store();
        let mut store = store.lock().unwrap();
        store.commit_tool_use(p, "t".into(), "{}".into()).unwrap()
    };
    d.append_tool_result(p, call.event_id, "r".into()).unwrap();
    d.append_message(p, role::USER, "q3".into()).unwrap();
    let parent_before = d.read(p, 0).unwrap();

    let child = d
        .rebase(
            p,
            vec![
                RebaseEdit::Drop { from: 2, to: 4 },
                RebaseEdit::Replace {
                    from: 4,
                    to: 5,
                    role: role::USER,
                    text: "q2 edited".into(),
                },
            ],
            None,
        )
        .unwrap();
    assert_eq!(d.read(p, 0).unwrap(), parent_before);
    let ce = d.read(child, 0).unwrap();
    assert_eq!(ce[0].body, parent_before[0].body);
    assert_eq!(ce[1].body, parent_before[1].body);
    assert!(matches!(ce[2].body, EventBody::Rebased { parent: pp, fork_at: 2, .. } if pp == p));
    assert!(matches!(&ce[3].body, EventBody::Message { role: r, text, .. } if *r == role::USER && text == "q2 edited"));
    let new_call = ce.iter().find(|e| matches!(e.body, EventBody::ToolUse { .. })).unwrap();
    assert_ne!(new_call.event_id, call.event_id, "re-opened under a new id");
    assert!(ce.iter().any(|e| matches!(&e.body, EventBody::ToolResult { call_id, .. } if *call_id == new_call.event_id)));
    assert!(d.open_tool_calls(child).unwrap().is_empty(), "the copied result closed it");
    let fresh = d.create(None, GenParams::default()).unwrap();
    d.append_message(fresh, role::SYSTEM, "sys".into()).unwrap();
    d.append_message(fresh, role::USER, "q2 edited".into()).unwrap();
    let fc = {
        let store = d.store();
        let mut store = store.lock().unwrap();
        store.commit_tool_use(fresh, "t".into(), "{}".into()).unwrap()
    };
    d.append_tool_result(fresh, fc.event_id, "r".into()).unwrap();
    d.append_message(fresh, role::USER, "q3".into()).unwrap();
    let store = d.store();
    let store = store.lock().unwrap();
    assert_eq!(store.session(child).unwrap().tokens, store.session(fresh).unwrap().tokens);
    assert_eq!(store.session(child).unwrap().base, 2);
    drop(store);
    assert!(matches!(
        d.rebase(p, vec![RebaseEdit::Drop { from: 0, to: 2 }], None),
        Err(superfluid_daemon::DaemonError::RebaseSpec(_))
    ));
    assert!(matches!(
        d.rebase(
            p,
            vec![
                RebaseEdit::Drop { from: 3, to: 5 },
                RebaseEdit::Drop { from: 4, to: 6 }
            ],
            None
        ),
        Err(superfluid_daemon::DaemonError::RebaseSpec(_))
    ));
}

#[test]
fn purge_is_cas_quiesced_and_reroots_or_cascades() {
    use superfluid_daemon::codec::MockChatCodec;
    use superfluid_daemon::wal::PurgeMode;
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let socket = dir.join("superfluid.sock");
    let (root, kid, grandkid, other_root, other_kid) = {
        let store = SessionStore::open(&wal).unwrap();
        let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None))
            .expect("spawn");
        let d = Arc::new(Daemon::new(store, host, Box::new(MockChatCodec), 8));
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let serving = Arc::clone(&d);
        std::thread::spawn(move || {
            let _ = api::serve(listener, serving);
        });

        let root = d.create(None, GenParams::default()).unwrap();
        d.append_message(root, 1, "hello".into()).unwrap();
        d.generate(root, 4).unwrap();
        let at = d.read(root, 0).unwrap().len() as u64;
        let kid = d.fork(root, at, None).unwrap();
        d.append_message(kid, 1, "kid".into()).unwrap();
        let kat = d.read(kid, 0).unwrap().len() as u64;
        let grandkid = d.fork(kid, kat, None).unwrap();
        let call = {
            let store = d.store();
            let mut store = store.lock().unwrap();
            store.commit_tool_use(root, "t".into(), "{}".into()).unwrap()
        };
        assert_eq!(d.open_tool_calls(root).unwrap().len(), 1);
        let mut obs = api::NativeClient::connect(&socket).unwrap();
        let cursor = d.read(root, 0).unwrap().len() as u64;
        let sub = obs.subscribe(root, cursor, false).unwrap();

        assert!(matches!(
            d.purge(root, 7, PurgeMode::Reroot),
            Err(superfluid_daemon::DaemonError::GenerationConflict { expected: 7, actual: 0, .. })
        ));
        assert_eq!(d.open_tool_calls(root).unwrap().len(), 1);

        let purged = d.purge(root, 0, PurgeMode::Reroot).unwrap();
        assert_eq!(purged, vec![root]);
        let mut ended = None;
        for _ in 0..8 {
            match obs.next_frame().unwrap() {
                api::Response::SubEvent { .. } => continue,
                api::Response::SubEnded { sub: s, reason } => {
                    assert_eq!(s, sub);
                    ended = Some(reason);
                    break;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(ended.as_deref(), Some("purged"));
        assert!(matches!(d.read(root, 0), Err(superfluid_daemon::DaemonError::Purged(_))));
        assert!(matches!(d.append(root, None, vec![1]), Err(superfluid_daemon::DaemonError::Purged(_))));
        assert!(matches!(d.fork(root, 1, None), Err(superfluid_daemon::DaemonError::Purged(_))));
        assert!(matches!(
            d.set_meta(root, 0, Some("back".into()), None),
            Err(superfluid_daemon::DaemonError::Purged(_))
        ));
        {
            let store = d.store();
            let store = store.lock().unwrap();
            let t = store.tombstone(root).expect("tombstone retained");
            assert_eq!(t.generation, 1);
            assert!(t.events.is_empty() && t.tokens.is_empty());
            assert_eq!(t.tombstone_ledger.get(&call.event_id).map(String::as_str), Some("expired"));
            let k = store.session(kid).unwrap();
            assert!(k.rerooted);
            assert!(k.events.iter().any(|e| matches!(e.body, EventBody::Rerooted { purged_parent, .. } if purged_parent == root)));
        }
        let tree = d.tree();
        assert!(tree.iter().any(|&(id, parent, _)| id == kid && parent.is_none()), "kid is a root now");
        assert!(tree.iter().any(|&(id, parent, _)| id == grandkid && parent == Some(kid)));
        assert!(!tree.iter().any(|&(id, _, _)| id == root));
        assert!(d.generate(kid, 2).unwrap().tokens_generated == 2);

        let other_root = d.create(None, GenParams::default()).unwrap();
        d.append_message(other_root, 1, "x".into()).unwrap();
        let oat = d.read(other_root, 0).unwrap().len() as u64;
        let other_kid = d.fork(other_root, oat, None).unwrap();
        let mut purged = d.purge(other_root, 0, PurgeMode::Cascade).unwrap();
        purged.sort();
        assert_eq!(purged, vec![other_root, other_kid]);
        assert!(matches!(d.read(other_kid, 0), Err(superfluid_daemon::DaemonError::Purged(_))));
        assert!(!d.session_ids().contains(&other_root));
        (root, kid, grandkid, other_root, other_kid)
    };
    // The API thread still holds the daemon, and with it the live log.
    let copy = dir.join("copy.log");
    std::fs::copy(&wal, &copy).unwrap();
    let store = SessionStore::open(&copy).unwrap();
    assert!(store.tombstone(root).is_some());
    assert!(store.tombstone(other_root).is_some());
    assert!(store.tombstone(other_kid).is_some());
    let k = store.session(kid).unwrap();
    assert!(k.rerooted);
    assert!(!k.tokens.is_empty());
    let g = store.session(grandkid).unwrap();
    assert_eq!(g.parent, Some(kid));
    assert_eq!(&g.tokens[..], &k.tokens[..g.tokens.len()]);
    assert!(!store.session_ids().contains(&root));
}

#[test]
fn purge_cancels_and_drains_an_in_flight_lane() {
    use superfluid_daemon::wal::PurgeMode;
    let dir = test_dir();
    let d = Arc::new(kv_only_daemon(&dir.join("wal.log"), None));
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..16).collect()).unwrap();
    let dd = Arc::clone(&d);
    let gen = std::thread::spawn(move || dd.generate(s, 100_000));
    for _ in 0..500 {
        if d.active_registry().lock().unwrap().contains(&s) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(d.active_registry().lock().unwrap().contains(&s), "lane admitted");
    let purged = d.purge(s, 0, PurgeMode::Cascade).unwrap();
    assert_eq!(purged, vec![s]);
    let out = gen.join().unwrap().unwrap();
    assert_eq!(out.finish, finish::CANCELLED, "the lane was cancelled by the purge");
    assert!(matches!(d.read(s, 0), Err(superfluid_daemon::DaemonError::Purged(_))));
    let t = d.create(None, GenParams::default()).unwrap();
    d.append(t, None, (0..16).collect()).unwrap();
    assert_eq!(d.generate(t, 4).unwrap().tokens_generated, 4);
}

fn qos_daemon(dir: &std::path::Path, opts: superfluid_daemon::DaemonOptions) -> Arc<Daemon> {
    qos_daemon_on(dir, opts, kv_only_config())
}

fn qos_daemon_on(dir: &std::path::Path, opts: superfluid_daemon::DaemonOptions, engine: EngineConfig) -> Arc<Daemon> {
    let store = SessionStore::open(&dir.join("wal.log")).expect("open store");
    let host = EngineHost::spawn(move || (MockEngine::new(engine), None)).expect("spawn");
    Arc::new(Daemon::with_options(store, host, Box::new(MockCodec), opts).unwrap())
}

#[test]
fn class_lane_cap_bounds_a_class_and_leaves_room_for_others() {
    use superfluid_daemon::qos;
    let dir = test_dir();
    let mut class_lanes = [0usize; 4];
    class_lanes[qos::BACKGROUND_AGENT as usize] = 1;
    let d = qos_daemon(
        &dir,
        superfluid_daemon::DaemonOptions { max_lanes: 4, class_lanes, ..Default::default() },
    );
    let stats = d.sched_stats();
    const TOKENS: u32 = 2000;
    let mut calls = Vec::new();
    for i in 0..3u32 {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..16 + i).collect()).unwrap();
        d.set_qos(s, qos::BACKGROUND_AGENT, false).unwrap();
        let dd = Arc::clone(&d);
        calls.push(std::thread::spawn(move || dd.generate(s, TOKENS)));
    }
    let agent = d.create(None, GenParams::default()).unwrap();
    d.append(agent, None, (100..116).collect()).unwrap();
    let dd = Arc::clone(&d);
    calls.push(std::thread::spawn(move || dd.generate(agent, TOKENS)));
    let mut peak = 0;
    while !calls.iter().all(|h| h.is_finished()) {
        peak = peak.max(stats.lanes_active.load(std::sync::atomic::Ordering::Relaxed));
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    for h in calls {
        assert_eq!(h.join().unwrap().unwrap().tokens_generated, TOKENS);
    }
    assert!(peak <= 2, "background cap of 1 + one agent lane, saw {peak} lanes at once");
    assert_eq!(peak, 2, "the agent ran beside the capped background class, not behind it");
}

#[test]
fn an_ephemeral_session_leaves_no_preemption_artifact() {
    use superfluid_daemon::qos;
    let dir = test_dir();
    let park = dir.join("park");
    let (ephemeral_ids, durable) = {
        let d = qos_daemon(
            &dir,
            superfluid_daemon::DaemonOptions {
                max_lanes: 2,
                park_dir: Some(park.clone()),
                ..Default::default()
            },
        );
        let stats = d.sched_stats();
        let extras = superfluid_daemon::scheduler::GenExtras {
            ephemeral: true,
            ..Default::default()
        };
        const AGENT_TOKENS: u32 = 6000;
        let mut agents = Vec::new();
        for i in 0..2u32 {
            let s = d.create(None, GenParams::default()).unwrap();
            d.append(s, None, (0..32 + i).collect()).unwrap();
            d.set_qos(s, qos::BACKGROUND_AGENT, false).unwrap();
            let dd = Arc::clone(&d);
            let ex = extras.clone();
            agents.push((
                s,
                std::thread::spawn(move || {
                    dd.generate_streaming_ex(s, AGENT_TOKENS, ex, |_| Ok(()))
                }),
            ));
        }
        let ids: Vec<u64> = agents.iter().map(|(s, _)| *s).collect();
        for _ in 0..4000 {
            if stats
                .lanes_active
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 2
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let chat = d.create(None, GenParams::default()).unwrap();
        d.append(chat, None, (100..140).collect()).unwrap();
        d.set_qos(chat, qos::INTERACTIVE_CHAT, false).unwrap();
        d.generate(chat, 16).unwrap();
        assert!(
            stats.preemptions.load(std::sync::atomic::Ordering::Relaxed) >= 1,
            "the test needs a preemption to have happened"
        );
        for (_, h) in agents {
            let out = h.join().unwrap().unwrap();
            assert_eq!(out.tokens_generated, AGENT_TOKENS, "the call still completes in full");
        }
        let durable = d.create(None, GenParams::default()).unwrap();
        d.append(durable, None, (0..32).collect()).unwrap();
        d.generate(durable, 8).unwrap();
        (ids, durable)
    };
    assert!(
        park.join(format!("{durable}.park")).exists(),
        "control: a session that can be resumed by id still parks"
    );
    for s in ephemeral_ids {
        assert!(
            !park.join(format!("{s}.park")).exists(),
            "session {s} finished for good; nothing can address its artifact"
        );
    }
}

#[test]
fn interactive_preempts_background_agents_and_resumes_exact() {
    use superfluid_daemon::qos;
    let dir = test_dir();
    let d = qos_daemon(
        &dir,
        superfluid_daemon::DaemonOptions {
            max_lanes: 2,
            park_dir: Some(dir.join("park")),
            ..Default::default()
        },
    );
    let stats = d.sched_stats();
    const AGENT_TOKENS: u32 = 6000;
    let mut agents = Vec::new();
    for i in 0..2u32 {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..32 + i).collect()).unwrap();
        d.set_qos(s, qos::BACKGROUND_AGENT, false).unwrap();
        let dd = Arc::clone(&d);
        agents.push((s, std::thread::spawn(move || dd.generate(s, AGENT_TOKENS))));
    }
    let agent_ids: Vec<u64> = agents.iter().map(|(s, _)| *s).collect();
    for _ in 0..4000 {
        if stats
            .lanes_active
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 2
        {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let chat = d.create(None, GenParams::default()).unwrap();
    d.append(chat, None, (100..140).collect()).unwrap();
    d.set_qos(chat, qos::INTERACTIVE_CHAT, false).unwrap();
    let both_agents_live = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = Arc::clone(&both_agents_live);
    let active = d.active_registry();
    let ids = agent_ids.clone();
    let out = d
        .generate_streaming(chat, 16, |_| {
            let a = active.lock().unwrap();
            if ids.iter().all(|s| a.contains(s)) {
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(out.tokens_generated, 16);
    assert!(
        both_agents_live.load(std::sync::atomic::Ordering::Relaxed),
        "the interactive job ran while both agent calls were still in flight"
    );
    assert!(stats.preemptions.load(std::sync::atomic::Ordering::Relaxed) >= 1);
    let mut agent_tokens = Vec::new();
    for (s, h) in agents {
        let out = h.join().unwrap().unwrap();
        assert_eq!(out.tokens_generated, AGENT_TOKENS, "the preempted call completes in full");
        let store = d.store();
        let store = store.lock().unwrap();
        agent_tokens.push((s, store.session(s).unwrap().tokens.clone()));
    }
    for (i, (_, toks)) in agent_tokens.iter().enumerate() {
        let f = d.create(None, GenParams::default()).unwrap();
        d.append(f, None, (0..32 + i as u32).collect()).unwrap();
        d.generate(f, AGENT_TOKENS).unwrap();
        let store = d.store();
        let store = store.lock().unwrap();
        assert_eq!(&store.session(f).unwrap().tokens, toks, "preempt/resume is transparent");
    }
}

#[test]
fn interactive_preempts_an_agent_still_prefilling() {
    use superfluid_daemon::qos;
    let dir = test_dir();
    let mut engine = kv_only_config();
    engine.tick_delay = std::time::Duration::from_millis(5);
    let d = qos_daemon_on(
        &dir,
        superfluid_daemon::DaemonOptions { max_lanes: 2, prefill_budget: 64, ..Default::default() },
        engine,
    );
    let stats = d.sched_stats();
    // Two agents with long prompts: at 64 prompt tokens a tick, both are
    // still prefilling long after the chat arrives.
    const PROMPT: u32 = 3000;
    let first_token: Arc<std::sync::Mutex<Vec<std::time::Instant>>> = Arc::default();
    let mut agents = Vec::new();
    for i in 0..2u32 {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..PROMPT).map(|t| (t + i * 7) % 900 + 50).collect()).unwrap();
        d.set_qos(s, qos::BACKGROUND_AGENT, false).unwrap();
        let dd = Arc::clone(&d);
        let seen = Arc::clone(&first_token);
        agents.push(std::thread::spawn(move || {
            let mut first = true;
            dd.generate_streaming(s, 8, |ev| {
                if first && matches!(&ev.body, EventBody::Generated { span, .. } if !span.is_empty()) {
                    first = false;
                    seen.lock().unwrap().push(std::time::Instant::now());
                }
                Ok(())
            })
        }));
    }
    for _ in 0..4000 {
        if stats.lanes_active.load(std::sync::atomic::Ordering::Relaxed) >= 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let chat = d.create(None, GenParams::default()).unwrap();
    d.append(chat, None, (100..140).collect()).unwrap();
    d.set_qos(chat, qos::INTERACTIVE_CHAT, false).unwrap();
    let out = d.generate(chat, 4).unwrap();
    let chat_done = std::time::Instant::now();
    assert_eq!(out.tokens_generated, 4);
    assert!(
        first_token.lock().unwrap().iter().all(|t| *t > chat_done),
        "the chat finished before either agent left prefill: it did not wait for one to start decoding"
    );
    assert!(stats.preemptions.load(std::sync::atomic::Ordering::Relaxed) >= 1);
    for h in agents {
        let out = h.join().unwrap().unwrap();
        assert_eq!(out.tokens_generated, 8, "the preempted agent completes in full");
    }
}

#[test]
fn a_burst_sharing_a_long_prefix_prefills_it_once() {
    let dir = test_dir();
    let mut engine = kv_only_config();
    engine.tick_delay = std::time::Duration::from_millis(2);
    let d = qos_daemon_on(
        &dir,
        superfluid_daemon::DaemonOptions { max_lanes: 4, prefill_budget: 64, ..Default::default() },
        engine,
    );
    let stats = d.sched_stats();
    const PREFIX: u32 = 2048;
    let sessions: Vec<u64> = (0..5u32)
        .map(|i| {
            let s = d.create(None, GenParams::default()).unwrap();
            let mut prompt: Vec<u32> = (0..PREFIX).map(|t| t % 900 + 50).collect();
            prompt.extend((0..8).map(|t| 1000 + i * 8 + t));
            d.append(s, None, prompt).unwrap();
            s
        })
        .collect();
    let calls: Vec<_> = sessions
        .iter()
        .map(|&s| {
            let dd = Arc::clone(&d);
            std::thread::spawn(move || dd.generate(s, 8))
        })
        .collect();
    for h in calls {
        assert_eq!(h.join().unwrap().unwrap().tokens_generated, 8);
    }
    let cold = stats.cold_admissions.load(Ordering::Relaxed);
    let warm = stats.warm_prefix_tokens.load(Ordering::Relaxed);
    assert_eq!(cold, 1, "one request prefilled the shared prefix; the rest started from it (warm {warm})");
    assert!(warm >= 4 * (PREFIX as u64 - 64), "the other four were seeded with the prefix: {warm}");
}

#[test]
fn a_continuation_preempted_mid_replay_keeps_its_progress() {
    use superfluid_daemon::qos;
    use std::sync::atomic::Ordering::Relaxed;
    let dir = test_dir();
    // Paced, so the agent is still decoding when each chat arrives: 8 rounds
    // a tick of 10 ms leave its last 1500 tokens about 2 s, where the four
    // chats take a few ticks each. Unpaced, a fast machine finished the tail
    // between two chats.
    let engine = EngineConfig { tick_delay: std::time::Duration::from_millis(10), ..kv_only_config() };
    let d = qos_daemon_on(
        &dir,
        superfluid_daemon::DaemonOptions {
            max_lanes: 1,
            park_dir: Some(dir.join("park")),
            tick_decode_budget: 8,
            ..Default::default()
        },
        engine,
    );
    let stats = d.sched_stats();
    const AGENT_TOKENS: u32 = 4000;
    let agent = d.create(None, GenParams::default()).unwrap();
    d.append(agent, None, (0..40).collect()).unwrap();
    d.set_qos(agent, qos::BACKGROUND_AGENT, false).unwrap();
    let dd = Arc::clone(&d);
    let handle = std::thread::spawn(move || dd.generate(agent, AGENT_TOKENS));
    // Paced at 8 rounds a 10 ms tick, 2,500 tokens take over 3 s; a slow
    // runner's ticks run longer, so the wait is generous.
    let wait_until = |pred: &dyn Fn() -> bool| {
        for _ in 0..15_000 {
            if pred() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        false
    };
    assert!(wait_until(&|| stats.lanes_active.load(Relaxed) >= 1), "the agent is admitted");
    assert!(wait_until(&|| stats.decode_tokens.load(Relaxed) >= 2500), "the agent produced a tail worth replaying");
    const CHATS: u64 = 4;
    for k in 0..CHATS {
        let chat = d.create(None, GenParams::default()).unwrap();
        d.append(chat, None, (100 + k as u32..116 + k as u32).collect()).unwrap();
        d.set_qos(chat, qos::INTERACTIVE_CHAT, false).unwrap();
        let out = d.generate(chat, 8).unwrap();
        assert_eq!(out.tokens_generated, 8);
        assert!(
            wait_until(&|| stats.resumes.load(Relaxed) > k),
            "readmission {} resumed from a park artifact (parks {} lossless / {} lossy)",
            k + 1,
            stats.parks_lossless.load(Relaxed),
            stats.parks_lossy.load(Relaxed)
        );
    }
    let out = handle.join().unwrap().unwrap();
    assert_eq!(out.tokens_generated, AGENT_TOKENS, "the twice-preempted call completes in full");
    let preemptions = stats.preemptions.load(Relaxed);
    let parks = stats.parks_lossless.load(Relaxed) + stats.parks_lossy.load(Relaxed);
    assert!(preemptions >= CHATS, "every chat preempted the agent ({preemptions})");
    assert!(parks >= preemptions, "every preemption wrote a park, mid-replay ones included ({parks} of {preemptions})");
    assert!(stats.resumes.load(Relaxed) >= CHATS, "every readmission resumed from its park");
    let agent_tokens = {
        let store = d.store();
        let store = store.lock().unwrap();
        store.session(agent).unwrap().tokens.clone()
    };
    let f = d.create(None, GenParams::default()).unwrap();
    d.append(f, None, (0..40).collect()).unwrap();
    d.generate(f, AGENT_TOKENS).unwrap();
    let store = d.store();
    let store = store.lock().unwrap();
    assert_eq!(store.session(f).unwrap().tokens, agent_tokens, "preempt/resume/preempt is transparent");
}

#[test]
fn a_preempted_constrained_agent_resumes_its_grammar() {
    use superfluid_daemon::qos;
    use superfluid_daemon::scheduler::GenExtras;
    const SCHEMA: &str = r#"{"type":"object"}"#;
    const AGENT_TOKENS: u32 = 6000;
    let dir = test_dir();
    let d = qos_daemon(
        &dir,
        superfluid_daemon::DaemonOptions { max_lanes: 2, park_dir: Some(dir.join("park")), ..Default::default() },
    );
    let stats = d.sched_stats();
    let constrained = |d: &Arc<Daemon>, s: u64| {
        let g = d.grammar_create(SCHEMA).unwrap();
        let out = d.generate_streaming_ex(s, AGENT_TOKENS, GenExtras { grammar_handle: g, ..Default::default() }, |_| Ok(()));
        d.grammar_free(g);
        out
    };
    let mut agents = Vec::new();
    for i in 0..2u32 {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..32 + i).collect()).unwrap();
        d.set_qos(s, qos::BACKGROUND_AGENT, false).unwrap();
        let dd = Arc::clone(&d);
        agents.push((s, std::thread::spawn(move || constrained(&dd, s))));
    }
    for _ in 0..4000 {
        if stats.lanes_active.load(std::sync::atomic::Ordering::Relaxed) >= 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let chat = d.create(None, GenParams::default()).unwrap();
    d.append(chat, None, (100..140).collect()).unwrap();
    d.set_qos(chat, qos::INTERACTIVE_CHAT, false).unwrap();
    assert_eq!(d.generate(chat, 16).unwrap().tokens_generated, 16);
    assert!(stats.preemptions.load(std::sync::atomic::Ordering::Relaxed) >= 1, "an agent was preempted");
    let mut agent_tokens = Vec::new();
    for (s, h) in agents {
        assert_eq!(h.join().unwrap().unwrap().tokens_generated, AGENT_TOKENS, "the preempted call completes");
        agent_tokens.push(d.store().lock().unwrap().session(s).unwrap().tokens.clone());
    }
    for (i, toks) in agent_tokens.iter().enumerate() {
        let f = d.create(None, GenParams::default()).unwrap();
        d.append(f, None, (0..32 + i as u32).collect()).unwrap();
        constrained(&d, f).unwrap();
        let uncontended = d.store().lock().unwrap().session(f).unwrap().tokens.clone();
        assert!(&uncontended == toks, "agent {i}: a preempted constrained call resumes its grammar exactly");
    }
}

#[test]
fn a_classed_sync_call_is_deferred_but_completes() {
    use superfluid_daemon::qos;
    let dir = test_dir();
    let d = qos_daemon(&dir, superfluid_daemon::DaemonOptions { max_lanes: 2, ..Default::default() });
    let stats = d.sched_stats();
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..32).collect()).unwrap();
    d.set_qos(s, qos::INTERACTIVE_CHAT, false).unwrap();
    let dd = Arc::clone(&d);
    let hog = std::thread::spawn(move || dd.generate(s, 6000));
    let t0 = std::time::Instant::now();
    while stats.lanes_active.load(Ordering::Relaxed) < 1 {
        assert!(t0.elapsed() < std::time::Duration::from_secs(20), "the lane never started");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let ticks_before = stats.ticks.load(Ordering::Relaxed);
    let _ = d.embed_as("hello", Some(qos::BACKGROUND_AGENT));
    let waited = stats.ticks.load(Ordering::Relaxed) - ticks_before;
    assert!(waited >= 5, "the background call waited behind the interactive lane ({waited} ticks)");
    let ticks_before = stats.ticks.load(Ordering::Relaxed);
    let _ = d.embed_as("hello", Some(qos::INTERACTIVE_CHAT));
    assert!(stats.ticks.load(Ordering::Relaxed) - ticks_before <= 2, "an equal class runs at once");
    let ticks_before = stats.ticks.load(Ordering::Relaxed);
    let _ = superfluid_daemon::with_sync_class(Some(qos::BACKGROUND_AGENT), || d.lora_id());
    let waited = stats.ticks.load(Ordering::Relaxed) - ticks_before;
    assert!(waited >= 5, "a scoped background lookup waited too ({waited} ticks)");
    let ticks_before = stats.ticks.load(Ordering::Relaxed);
    let _ = d.lora_id();
    assert!(stats.ticks.load(Ordering::Relaxed) - ticks_before <= 2, "outside the scope it runs at once");
    let ticks_before = stats.ticks.load(Ordering::Relaxed);
    let _ = superfluid_daemon::with_sync_class(Some(qos::BACKGROUND_AGENT), || d.capability_descriptor());
    assert!(stats.ticks.load(Ordering::Relaxed) - ticks_before <= 2, "the record is read without the engine");
    assert_eq!(hog.join().unwrap().unwrap().tokens_generated, 6000);
}

#[test]
fn starving_agent_lane_is_served_within_the_bound() {
    use superfluid_daemon::qos;
    let dir = test_dir();
    let d = qos_daemon(
        &dir,
        superfluid_daemon::DaemonOptions {
            max_lanes: 2,
            tick_decode_budget: 32,
            agent_starvation_ticks: 3,
            ..Default::default()
        },
    );
    let stats = d.sched_stats();
    let agent = d.create(None, GenParams::default()).unwrap();
    d.append(agent, None, (0..16).collect()).unwrap();
    d.set_qos(agent, qos::BACKGROUND_AGENT, false).unwrap();
    let chat = d.create(None, GenParams::default()).unwrap();
    d.append(chat, None, (50..66).collect()).unwrap();
    d.set_qos(chat, qos::INTERACTIVE_CHAT, false).unwrap();
    let dd = Arc::clone(&d);
    let a = std::thread::spawn(move || dd.generate(agent, 320));
    let dd = Arc::clone(&d);
    let c = std::thread::spawn(move || dd.generate(chat, 3200));
    let a_out = a.join().unwrap().unwrap();
    assert_eq!(a_out.tokens_generated, 320, "the agent completed under contention");
    assert!(
        stats.starvation_grants.load(std::sync::atomic::Ordering::Relaxed) >= 1,
        "the bound fired at least once"
    );
    let c_out = c.join().unwrap().unwrap();
    assert_eq!(c_out.tokens_generated, 3200);
}

#[test]
fn batch_invariant_lane_runs_alone() {
    let dir = test_dir();
    let d = qos_daemon(
        &dir,
        superfluid_daemon::DaemonOptions {
            max_lanes: 4,
            ..Default::default()
        },
    );
    let bi = d.create(None, GenParams::default()).unwrap();
    d.append(bi, None, (0..16).collect()).unwrap();
    d.set_qos(bi, superfluid_daemon::qos::FOREGROUND_AGENT, true).unwrap();
    let other = d.create(None, GenParams::default()).unwrap();
    d.append(other, None, (20..36).collect()).unwrap();
    let dd = Arc::clone(&d);
    let t = std::thread::spawn(move || dd.generate(bi, 96));
    for _ in 0..500 {
        if d.active_registry().lock().unwrap().contains(&bi) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    // A token of the other job may only land once the batch-invariant job has left the
    // scheduler; the registry says so at the moment the token lands.
    let saw_bi_done_first = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let f2 = Arc::clone(&saw_bi_done_first);
    let registry = d.active_registry();
    let out = d
        .generate_streaming(other, 8, |_| {
            if registry.lock().unwrap().contains(&bi) {
                f2.store(false, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(out.tokens_generated, 8);
    assert!(
        saw_bi_done_first.load(std::sync::atomic::Ordering::SeqCst),
        "no token of the other job landed while the batch-invariant lane ran"
    );
    assert_eq!(t.join().unwrap().unwrap().tokens_generated, 96);
}

#[test]
fn seeded_session_is_never_preempted_lossy() {
    use superfluid_daemon::qos;
    let dir = test_dir();
    let park = dir.join("park");
    let d = qos_daemon(
        &dir,
        superfluid_daemon::DaemonOptions {
            max_lanes: 1,
            park_dir: Some(park.clone()),
            park_lossy: true,
            ..Default::default()
        },
    );
    let stats = d.sched_stats();
    let seeded = d.create(None, GenParams { seed: 9, ..Default::default() }).unwrap();
    d.append(seeded, None, (0..32).collect()).unwrap();
    d.set_qos(seeded, qos::BACKGROUND_AGENT, false).unwrap();
    const SEEDED_TOKENS: u32 = 4000;
    let dd = Arc::clone(&d);
    let t = std::thread::spawn(move || dd.generate(seeded, SEEDED_TOKENS));
    for _ in 0..4000 {
        if stats.lanes_active.load(std::sync::atomic::Ordering::Relaxed) >= 1 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let chat = d.create(None, GenParams::default()).unwrap();
    d.append(chat, None, (100..116).collect()).unwrap();
    d.set_qos(chat, qos::INTERACTIVE_CHAT, false).unwrap();
    d.generate(chat, 4).unwrap();
    let out = t.join().unwrap().unwrap();
    assert_eq!(out.tokens_generated, SEEDED_TOKENS);
    assert!(stats.preemptions.load(std::sync::atomic::Ordering::Relaxed) >= 1);
    let f = d.create(None, GenParams { seed: 9, ..Default::default() }).unwrap();
    d.append(f, None, (0..32).collect()).unwrap();
    d.generate(f, SEEDED_TOKENS).unwrap();
    let store = d.store();
    let store = store.lock().unwrap();
    assert_eq!(store.session(seeded).unwrap().tokens, store.session(f).unwrap().tokens);
}

#[test]
fn os_pressure_signals_trigger_relief_and_background_preemption() {
    use superfluid_daemon::pressure::{ManualPressure, PressureConfig, PressureLevel};
    use superfluid_daemon::qos;
    let dir = test_dir();
    let manual = ManualPressure::new();
    let d = qos_daemon_on(
        &dir,
        superfluid_daemon::DaemonOptions {
            max_lanes: 2,
            pressure_source: PressureConfig::Manual(manual.clone()),
            ..Default::default()
        },
        EngineConfig { tick_delay: std::time::Duration::from_millis(50), ..kv_only_config() },
    );
    let stats = d.sched_stats();
    let agent = d.create(None, GenParams::default()).unwrap();
    d.append(agent, None, (0..32).collect()).unwrap();
    d.set_qos(agent, qos::BACKGROUND_AGENT, false).unwrap();
    let dd = Arc::clone(&d);
    let t = std::thread::spawn(move || dd.generate(agent, 3_000));
    for _ in 0..500 {
        if d.active_registry().lock().unwrap().contains(&agent) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let wait_for = |pred: &dyn Fn() -> bool| {
        for _ in 0..1000 {
            if pred() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        false
    };
    manual.set(PressureLevel::Warning);
    assert!(
        wait_for(&|| stats.os_pressure_events.load(std::sync::atomic::Ordering::Relaxed) >= 1),
        "warning acted on"
    );
    let before = stats.preemptions.load(std::sync::atomic::Ordering::Relaxed);
    manual.set(PressureLevel::Critical);
    assert!(
        wait_for(&|| stats.preemptions.load(std::sync::atomic::Ordering::Relaxed) > before),
        "critical pressure parked the background lane"
    );
    manual.set(PressureLevel::Normal);
    let out = t.join().unwrap().unwrap();
    assert_eq!(out.tokens_generated, 3_000, "the preempted call still completes");
}

#[test]
fn speculation_is_transparent_and_certificate_scoped() {
    let dir = test_dir();
    let mk = |sub: &str, speculate: Option<&str>| {
        let d = dir.join(sub);
        std::fs::create_dir_all(&d).unwrap();
        let store = SessionStore::open(&d.join("wal.log")).unwrap();
        let host = EngineHost::spawn(|| (MockEngine::new(kv_only_config()), None)).unwrap();
        Daemon::with_options(
            store,
            host,
            Box::new(MockCodec),
            superfluid_daemon::DaemonOptions {
                speculate: speculate.map(str::to_string),
                ..Default::default()
            },
        )
        .unwrap()
    };
    let plain = mk("plain", None);
    let spec = mk("spec", Some("prompt-lookup"));
    assert!(plain.speculation().is_none());
    let grant = spec.speculation().expect("registered at start");
    assert_eq!(grant.strategy_id, "prompt-lookup");
    assert_eq!(grant.slot, 1);
    assert_eq!(grant.class, superfluid_abi::exactness::SEED_PATH_INVARIANT);
    assert_eq!(grant.modes, superfluid_abi::cert_mode::GREEDY);

    let mut prompt = Vec::new();
    for _ in 0..8 {
        prompt.extend_from_slice(&[11, 12, 13, 14, 15, 16, 17, 18]);
    }
    let run = |d: &Daemon, params: GenParams| -> Vec<u32> {
        let s = d.create(None, params).unwrap();
        d.append(s, None, prompt.clone()).unwrap();
        d.generate(s, 40).unwrap();
        d.generate(s, 24).unwrap();
        let store = d.store();
        let store = store.lock().unwrap();
        store.session(s).unwrap().tokens.clone()
    };
    let greedy = GenParams::default();
    assert_eq!(run(&plain, greedy), run(&spec, greedy), "speculation never changes tokens");
    let sampled = GenParams {
        temperature: 0.7,
        seed: 5,
        ..Default::default()
    };
    assert_eq!(run(&plain, sampled), run(&spec, sampled), "sampled lanes admit unspeculated");
    let st = spec.sched_stats();
    assert!(
        st.spec_proposed.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "greedy lanes proposed drafts"
    );
    let d3 = dir.join("bad");
    std::fs::create_dir_all(&d3).unwrap();
    let store = SessionStore::open(&d3.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(kv_only_config()), None)).unwrap();
    assert!(Daemon::with_options(
        store,
        host,
        Box::new(MockCodec),
        superfluid_daemon::DaemonOptions {
            speculate: Some("eagle3".into()),
            ..Default::default()
        },
    )
    .is_err());
}

#[test]
fn rich_blocks_project_by_versioned_visibility() {
    use superfluid_daemon::codec::{MockChatCodec, MODEL_VISIBILITY_VERSION};
    use superfluid_daemon::wal::{block_kind, role};
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    let d = Daemon::new(store, host, Box::new(MockChatCodec), 8);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append_message(s, role::USER, "fix it".into()).unwrap();
    let before = d.inspect(s).unwrap().tokens.len();
    let diff = d
        .append_block(
            s,
            role::USER,
            block_kind::FILE_DIFF,
            r#"{"path":"src/x.rs","diff":"-a\n+b"}"#.into(),
        )
        .unwrap();
    match &diff.body {
        EventBody::Block {
            kind,
            span,
            visibility_version,
            ..
        } => {
            assert_eq!(*kind, block_kind::FILE_DIFF);
            assert!(!span.is_empty(), "a diff enters context");
            assert_eq!(*visibility_version, MODEL_VISIBILITY_VERSION);
        }
        other => panic!("unexpected {other:?}"),
    }
    let after_diff = d.inspect(s).unwrap().tokens.len();
    assert!(after_diff > before);
    let progress = d
        .append_block(s, role::ASSISTANT, block_kind::PROGRESS, r#"{"percent":40}"#.into())
        .unwrap();
    match &progress.body {
        EventBody::Block { span, .. } => assert!(span.is_empty(), "progress never enters context"),
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(d.inspect(s).unwrap().tokens.len(), after_diff, "no separator, no tokens");
    assert!(matches!(
        d.append_block(s, role::USER, 999, "{}".into()),
        Err(superfluid_daemon::DaemonError::UnknownBlockKind(999))
    ));
    use superfluid_daemon::TextCodec;
    let fp = MockChatCodec.behavior_fingerprint().unwrap();
    assert_eq!(fp.block_projection_version, MODEL_VISIBILITY_VERSION);
    let tokens = d.inspect(s).unwrap().tokens;
    drop(d);
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    assert_eq!(store.session(s).unwrap().tokens, tokens);
}

#[test]
fn permission_requests_gate_generation() {
    use superfluid_daemon::codec::MockChatCodec;
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    let d = Daemon::new(store, host, Box::new(MockChatCodec), 8);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append_message(s, 1, "rm -rf?".into()).unwrap();
    let call = {
        let store = d.store();
        let mut store = store.lock().unwrap();
        store.commit_tool_use(s, "shell".into(), r#"{"cmd":"rm -rf x"}"#.into()).unwrap()
    };
    assert!(matches!(
        d.request_permission(s, 4242, "?".into()),
        Err(superfluid_daemon::DaemonError::UnknownToolCall(4242))
    ));
    let req = d.request_permission(s, call.event_id, "Allow rm -rf x?".into()).unwrap();
    assert!(matches!(
        d.generate(s, 4),
        Err(superfluid_daemon::DaemonError::PermissionPending(_))
    ));
    assert!(matches!(
        d.respond_permission(s, 9999, true),
        Err(superfluid_daemon::DaemonError::UnknownPermission(9999))
    ));
    d.respond_permission(s, req.event_id, false).unwrap();
    assert!(matches!(
        d.respond_permission(s, req.event_id, true),
        Err(superfluid_daemon::DaemonError::UnknownPermission(_))
    ));
    d.append_tool_outcome(s, call.event_id, superfluid_daemon::wal::tool_outcome::CANCELLED, "denied".into())
        .unwrap();
    assert!(d.generate(s, 4).is_ok());
}

#[test]
fn tool_ledger_states_are_exactly_one_terminal_and_audit_derived() {
    use superfluid_daemon::codec::MockChatCodec;
    use superfluid_daemon::wal::{ledger_state, tool_outcome};
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).unwrap();
    let d = Daemon::new(store, host, Box::new(MockChatCodec), 8);
    let s = d.create(None, GenParams::default()).unwrap();
    d.append_message(s, 1, "go".into()).unwrap();
    let mk_call = |name: &str| {
        let store = d.store();
        let mut store = store.lock().unwrap();
        store.commit_tool_use(s, name.into(), "{}".into()).unwrap().event_id
    };
    let state = |id: u64| {
        d.ledger(s).unwrap().into_iter().find(|(i, _)| *i == id).unwrap().1
    };
    let a = mk_call("slow");
    assert_eq!(state(a).state, ledger_state::OPEN);
    d.request_tool_cancel(s, a).unwrap();
    let e = state(a);
    assert_eq!(e.state, ledger_state::CANCEL_REQUESTED);
    assert!(e.cancel_requested);
    assert_eq!(d.open_tool_calls(s).unwrap().len(), 1, "still open until the terminal");
    d.append_tool_outcome(s, a, tool_outcome::CANCELLED, "killed".into()).unwrap();
    assert_eq!(state(a).state, ledger_state::CANCELLED);
    assert!(d.open_tool_calls(s).unwrap().is_empty());
    let tokens_before = d.inspect(s).unwrap().tokens.len();
    let late = d.append_tool_result(s, a, "it ran anyway".into());
    assert!(matches!(late, Err(superfluid_daemon::DaemonError::ToolLateEffect(id)) if id == a));
    let e = state(a);
    assert_eq!(e.state, ledger_state::CANCELLED_WITH_LATE_EFFECT);
    assert_eq!(e.late_effects, 1);
    assert_eq!(d.inspect(s).unwrap().tokens.len(), tokens_before, "never applied to the stream");
    let events = d.read(s, 0).unwrap();
    assert!(events.iter().any(|e| matches!(&e.body, EventBody::ToolReconciliation { call_id, .. } if *call_id == a)));
    assert!(matches!(
        d.append_tool_outcome(s, a, tool_outcome::FAILED, "again".into()),
        Err(superfluid_daemon::DaemonError::UnknownToolCall(_))
    ));
    let b = mk_call("flaky");
    d.append_tool_outcome(s, b, tool_outcome::FAILED, "boom".into()).unwrap();
    assert_eq!(state(b).state, ledger_state::FAILED);
    let c = mk_call("fine");
    d.append_tool_result(s, c, "ok".into()).unwrap();
    assert_eq!(state(c).state, ledger_state::SUCCEEDED);
    let l = mk_call("leased");
    {
        let store = d.store();
        let mut store = store.lock().unwrap();
        store.commit_tool_lease(s, l, 1).unwrap();
    }
    let tokens_before = d.inspect(s).unwrap().tokens.len();
    assert!(matches!(
        d.append_tool_result(s, l, "too late".into()),
        Err(superfluid_daemon::DaemonError::ToolLeaseExpired(id)) if id == l
    ));
    assert_eq!(state(l).state, ledger_state::EXPIRED);
    assert_eq!(d.inspect(s).unwrap().tokens.len(), tokens_before);
    let ids = (a, b, c, l);
    drop(d);
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let led = &store.session(s).unwrap().ledger;
    assert_eq!(led[&ids.0].state, ledger_state::CANCELLED_WITH_LATE_EFFECT);
    assert_eq!(led[&ids.1].state, ledger_state::FAILED);
    assert_eq!(led[&ids.2].state, ledger_state::SUCCEEDED);
    assert_eq!(led[&ids.3].state, ledger_state::EXPIRED);
    assert!(store.session(s).unwrap().open_tool_calls.is_empty());
}

fn media_config() -> EngineConfig {
    EngineConfig {
        image_token_id: 999,
        media_tokens_per_image: 4,
        ..kv_only_config()
    }
}

fn media_daemon(dir: &std::path::Path, park: bool) -> Daemon {
    use superfluid_daemon::codec::MockChatCodec;
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(media_config()), None)).unwrap();
    Daemon::with_options(
        store,
        host,
        Box::new(MockChatCodec),
        superfluid_daemon::DaemonOptions {
            park_dir: park.then(|| dir.join("park")),
            media_dir: Some(dir.join("media")),
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn image_turns_are_durable_bound_and_cache_excluded() {
    use superfluid_daemon::wal::{block_kind, role};
    let dir = test_dir();
    let d = media_daemon(&dir, true);
    let png = b"not really a png, but bytes are bytes";
    let hash = d.put_media(png).unwrap();
    assert_eq!(hash.len(), 64);
    assert_eq!(d.put_media(png).unwrap(), hash, "idempotent");
    let path = d.media_pool().path_of(&hash);
    assert!(path.is_file());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    }
    let s = d.create(None, GenParams::default()).unwrap();
    d.append_message(s, role::SYSTEM, "look".into()).unwrap();
    assert!(matches!(
        d.append_image(s, role::USER, "deadbeef", "", ""),
        Err(superfluid_daemon::DaemonError::UnknownMedia(_))
    ));
    let before = d.inspect(s).unwrap().tokens.len();
    let ev = d.append_image(s, role::USER, &hash, "what color is", " this?").unwrap();
    let (payload, span) = match &ev.body {
        EventBody::Block { kind, payload, span, .. } => {
            assert_eq!(*kind, block_kind::IMAGE);
            (payload.clone(), span.clone())
        }
        other => panic!("unexpected {other:?}"),
    };
    let p: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(p["blob"], hash);
    assert_eq!(p["n_tokens"], 4);
    assert_eq!(span.iter().filter(|&&t| t == 999).count(), 4, "the run the preprocessor sized");
    let first = span.iter().position(|&t| t == 999).unwrap();
    assert_eq!(p["offset"], (before + first) as u64);
    assert!(p["media_identity"]["preprocessing_fp"].is_string());
    let out = d.generate(s, 8).unwrap();
    assert_eq!(out.tokens_generated, 8);
    let stats = d.sched_stats();
    let s2 = d.create(None, GenParams::default()).unwrap();
    d.append_message(s2, role::SYSTEM, "look".into()).unwrap();
    d.append_image(s2, role::USER, &hash, "what color is", " this?").unwrap();
    let out2 = d.generate(s2, 8).unwrap();
    assert_eq!(out2.warm_prefix, 0, "cache excluded");
    let artifact = dir.join("park").join(format!("{s}.park"));
    for _ in 0..2000 {
        if artifact.is_file() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let out3 = d.generate(s, 4).unwrap();
    assert!(out3.warm_prefix > 0, "adopted from the media session's own artifact");
    assert!(stats.resumes.load(std::sync::atomic::Ordering::Relaxed) >= 1);
    let tokens = d.inspect(s).unwrap().tokens;
    drop(d);
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    assert_eq!(store.session(s).unwrap().tokens, tokens);
}

#[test]
fn media_gc_marks_through_live_logs() {
    use superfluid_daemon::wal::{role, PurgeMode};
    let dir = test_dir();
    let d = media_daemon(&dir, false);
    let a = d.put_media(b"image A").unwrap();
    let b = d.put_media(b"image B").unwrap();
    let orphan = d.put_media(b"never referenced").unwrap();
    let s = d.create(None, GenParams::default()).unwrap();
    d.append_image(s, role::USER, &a, "", "").unwrap();
    let at = d.read(s, 0).unwrap().len() as u64;
    let child = d.fork(s, at, None).unwrap();
    d.append_image(child, role::USER, &b, "", "").unwrap();
    let (kept, removed) = d.gc_media().unwrap();
    assert_eq!((kept, removed), (2, 1), "A and B live; the orphan swept");
    assert!(!d.media_pool().exists(&orphan));
    d.purge(s, 0, PurgeMode::Reroot).unwrap();
    let (kept, removed) = d.gc_media().unwrap();
    assert_eq!((kept, removed), (2, 0));
    d.purge(child, 0, PurgeMode::Cascade).unwrap();
    let (kept, removed) = d.gc_media().unwrap();
    assert_eq!((kept, removed), (0, 2));
}

#[test]
fn oversized_images_are_refused_typed() {
    use superfluid_daemon::codec::MockChatCodec;
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| {
        (
            MockEngine::new(EngineConfig {
                media_tokens_per_image: 5000,
                ..media_config()
            }),
            None,
        )
    })
    .unwrap();
    let d = Daemon::with_options(
        store,
        host,
        Box::new(MockChatCodec),
        superfluid_daemon::DaemonOptions {
            media_dir: Some(dir.join("media")),
            ..Default::default()
        },
    )
    .unwrap();
    let hash = d.put_media(b"huge").unwrap();
    let s = d.create(None, GenParams::default()).unwrap();
    assert!(matches!(
        d.append_image(s, 1, &hash, "", ""),
        Err(superfluid_daemon::DaemonError::MediaTooLarge(5000))
    ));
}

#[test]
fn fim_completions_are_ephemeral_cached_and_isolated() {
    use superfluid_daemon::codec::{fim_mode, MockChatCodec};
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(kv_only_config()), None)).unwrap();
    let d = Daemon::with_options(
        store,
        host,
        Box::new(MockChatCodec),
        superfluid_daemon::DaemonOptions::default(),
    )
    .unwrap();
    let sess = d.create(None, GenParams::default()).unwrap();
    d.append(sess, None, (0..16).collect()).unwrap();
    let durable_ids_before = d.session_ids();

    let c1 = d.complete("fn main() {\n    let x = ", "\n}\n", fim_mode::PSM, 8).unwrap();
    assert!(!c1.cached && !c1.expired);
    assert_eq!(c1.tokens.len(), 8);
    assert_eq!(d.session_ids(), durable_ids_before, "no durable session from a completion");

    let c2 = d.complete("fn main() {\n    let x = ", "\n}\n", fim_mode::PSM, 8).unwrap();
    assert!(c2.cached, "identical FIM request hits the composition cache");
    assert_eq!(c2.tokens, c1.tokens);
    let c3 = d.complete("fn main() {\n    let x = ", "\n    y\n}\n", fim_mode::PSM, 8).unwrap();
    assert!(!c3.cached);
    let c4 = d.complete("fn main() {\n    let x = ", "\n}\n", fim_mode::SPM, 8).unwrap();
    assert!(!c4.cached, "a different boundary mode is a different key");

    assert_eq!(d.generate(sess, 4).unwrap().tokens_generated, 4);
}

#[test]
fn completion_without_fim_dialect_refuses() {
    let dir = test_dir();
    let d = mock_daemon(&dir.join("wal.log"));
    assert!(matches!(
        d.complete("a", "b", 0, 4),
        Err(superfluid_daemon::DaemonError::NoFimDialect)
    ));
}

#[test]
fn trim_keeps_the_head_and_leaves_the_parent() {
    let dir = test_dir();
    let d = mock_daemon(&dir.join("wal.log"));
    let p = d.create(None, GenParams::default()).unwrap();
    d.append(p, None, vec![1, 2, 3]).unwrap();
    d.append(p, None, vec![4, 5]).unwrap();
    d.append(p, None, vec![6]).unwrap();
    let before = d.read(p, 0).unwrap();
    let child = d.trim(p, 2).unwrap();
    assert_eq!(d.read(p, 0).unwrap(), before, "parent untouched");
    let store = d.store();
    let store = store.lock().unwrap();
    assert_eq!(store.session(child).unwrap().tokens, vec![1, 2, 3], "head only");
    assert_eq!(store.session(child).unwrap().base, 2);
    drop(store);
    let max = d.read(p, 0).unwrap().len() as u64;
    let full = d.trim(p, max).unwrap();
    let store = d.store();
    let store = store.lock().unwrap();
    assert_eq!(store.session(full).unwrap().tokens, store.session(p).unwrap().tokens);
    drop(store);
    assert!(matches!(
        d.trim(p, 0),
        Err(superfluid_daemon::DaemonError::ForkPoint { fork_at: 0, .. })
    ));
}

#[test]
fn pin_records_a_ttl_deadline() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let s = {
        let d = mock_daemon(&wal);
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, vec![1]).unwrap();
        let deadline = d.pin(s, 60_000).unwrap();
        assert!(deadline > 1_700_000_000_000);
        assert_eq!(d.inspect(s).unwrap().pin_deadline_unix_ms, deadline);
        assert_eq!(d.pin(s, 0).unwrap(), 0, "ttl 0 clears");
        assert_eq!(d.inspect(s).unwrap().pin_deadline_unix_ms, 0);
        d.pin(s, 120_000).unwrap();
        s
    };
    let store = SessionStore::open(&wal).unwrap();
    assert!(store.session(s).unwrap().pin_deadline_unix_ms > 0);
}

fn pin_daemon(dir: &std::path::Path, blocks: u64, opts: superfluid_daemon::DaemonOptions) -> Daemon {
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let cfg = EngineConfig {
        pool_bytes: blocks * 1024,
        ..kv_only_config()
    };
    let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).expect("spawn");
    Daemon::with_options(store, host, Box::new(MockCodec), opts).unwrap()
}

fn pin_daemon_ttl(dir: &std::path::Path, blocks: u64, ttl: u64) -> Daemon {
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let cfg = EngineConfig {
        pool_bytes: blocks * 1024,
        seed_ttl_ticks: ttl,
        ..kv_only_config()
    };
    let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).expect("spawn");
    Daemon::with_options(
        store,
        host,
        Box::new(MockCodec),
        superfluid_daemon::DaemonOptions {
            max_lanes: 2,
            ..Default::default()
        },
    )
    .unwrap()
}

fn run_session(d: &Daemon, prompt: std::ops::Range<u32>, gen: u32) -> u64 {
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, prompt.collect()).unwrap();
    assert_eq!(d.generate(s, gen).unwrap().tokens_generated, gen);
    s
}

fn cached_prefix(d: &Daemon, prompt: std::ops::Range<u32>) -> u64 {
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, prompt.collect()).unwrap();
    d.generate(s, 1).unwrap().warm_prefix
}

#[test]
fn a_pinned_prefix_survives_pressure_until_its_deadline() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(
        &dir,
        32,
        superfluid_daemon::DaemonOptions {
            max_lanes: 2,
            ..Default::default()
        },
    );
    let stats = d.sched_stats();
    let pinned = d.create(None, GenParams::default()).unwrap();
    d.append(pinned, None, (0..96).collect()).unwrap();
    d.pin(pinned, 60_000).unwrap();
    assert_eq!(d.generate(pinned, 16).unwrap().tokens_generated, 16);
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        1,
        "leased at retire"
    );
    assert_eq!(
        stats.pinned_bytes.load(Ordering::Relaxed),
        7 * 1024,
        "the whole 112-token stream = 7 blocks (a session pin stages a lookahead)"
    );
    run_session(&d, 1000..1096, 16);
    for i in 2..9u32 {
        run_session(&d, i * 1000..i * 1000 + 96, 16);
    }
    assert!(
        stats.pressure_evictions.load(Ordering::Relaxed) > 0,
        "pressure ran"
    );
    assert_eq!(
        cached_prefix(&d, 1000..1096),
        0,
        "the unpinned prefix was evicted"
    );
    assert_eq!(
        cached_prefix(&d, 0..96),
        80,
        "the pinned prefix survived (block-aligned)"
    );

    d.pin(pinned, 200).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(300));
    for i in 10..20u32 {
        run_session(&d, i * 1000..i * 1000 + 96, 16);
    }
    assert!(stats.pins_expired.load(Ordering::Relaxed) >= 1);
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 0);
    assert_eq!(
        cached_prefix(&d, 0..96),
        0,
        "an expired pin is ordinary cache again"
    );
}

#[test]
fn a_shared_prefix_pin_keeps_the_longest_deadline() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(&dir, 64, Default::default());
    let stats = d.sched_stats();
    run_session(&d, 0..96, 16);
    let stream: Vec<u32> = (0..96).collect();
    assert_eq!(d.pin_prefix(None, &stream, 64, 60_000).unwrap(), 64);
    assert_eq!(d.pin_prefix(None, &stream, 64, 50).unwrap(), 64);
    std::thread::sleep(std::time::Duration::from_millis(150));
    assert_eq!(d.pinned_prefix_tokens(&stream, &[64]).unwrap(), 64);
    d.pin_prefix(None, &stream, 32, 60_000).unwrap();
    assert_eq!(
        stats.pins_expired.load(Ordering::Relaxed),
        0,
        "the 60 s deadline held"
    );
    assert_eq!(d.pinned_prefix_tokens(&stream, &[64]).unwrap(), 64);
}

#[test]
fn a_worker_respawn_drops_pin_leases_at_once() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let (d, pid) = process_daemon(&dir.join("wal.log"));
    let stats = d.sched_stats();
    let pinned = run_session(&d, 0..96, 16);
    d.pin(pinned, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    let _ = std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .expect("kill");
    std::thread::sleep(std::time::Duration::from_millis(100));
    let other = d.create(None, GenParams::default()).unwrap();
    d.append(other, None, (500..596).collect()).unwrap();
    let _ = d.generate(other, 4);
    for _ in 0..500 {
        if stats.worker_respawns.load(Ordering::Relaxed) == 1
            && stats.pins_held.load(Ordering::Relaxed) == 0
        {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(stats.worker_respawns.load(Ordering::Relaxed), 1);
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        0,
        "no lease survives a respawn"
    );
    assert_eq!(stats.pinned_bytes.load(Ordering::Relaxed), 0);
    d.generate(pinned, 4).unwrap();
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        1,
        "re-leased at its retire"
    );
}

#[test]
fn a_shared_prefix_pin_keeps_every_anchor() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(&dir, 64, Default::default());
    let stats = d.sched_stats();
    let stream: Vec<u32> = (0..96).collect();
    let first = d.create(None, GenParams::default()).unwrap();
    d.append(first, None, stream.clone()).unwrap();
    let second = d.create(None, GenParams::default()).unwrap();
    d.append(second, None, stream.clone()).unwrap();
    assert_eq!(d.pin_prefix(Some(first), &stream, 64, 60_000).unwrap(), 0);
    assert_eq!(d.pin_prefix(Some(second), &stream, 64, 60_000).unwrap(), 0);
    d.generate(first, 4).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    assert_eq!(d.pinned_prefix_tokens(&stream, &[64]).unwrap(), 64);
}

#[test]
fn purging_a_pinned_session_releases_its_lease() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(&dir, 64, Default::default());
    let stats = d.sched_stats();
    let s = run_session(&d, 0..96, 16);
    d.pin(s, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    d.purge(s, 0, superfluid_daemon::wal::PurgeMode::Cascade)
        .unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 0);
    assert_eq!(stats.pinned_bytes.load(Ordering::Relaxed), 0);
}

#[test]
fn a_page_aligned_session_pin_holds_its_last_page() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(&dir, 64, Default::default());
    let stats = d.sched_stats();
    let s = run_session(&d, 0..12, 4);
    d.pin(s, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    assert_eq!(stats.pinned_bytes.load(Ordering::Relaxed), 1024);
}

#[test]
fn a_whole_stream_prefix_pin_holds_its_last_page() {
    let dir = test_dir();
    let d = pin_daemon(&dir, 64, Default::default());
    let s = run_session(&d, 0..12, 4);
    let stream = d.store().lock().unwrap().session(s).unwrap().tokens.clone();
    assert_eq!(stream.len(), 16);
    assert_eq!(d.pin_prefix(None, &stream, 16, 60_000).unwrap(), 16);
}

#[test]
fn a_max_context_session_pin_still_holds() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(&dir, 1024, Default::default());
    let stats = d.sched_stats();
    let max = d.max_stream_tokens() as u32;
    assert!(max > 64);
    let s = run_session(&d, 0..max - 8, 8);
    assert_eq!(
        d.store().lock().unwrap().session(s).unwrap().tokens.len() as u32,
        max
    );
    d.pin(s, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    let held_blocks = stats.pinned_bytes.load(Ordering::Relaxed) / 1024;
    assert_eq!(
        held_blocks,
        (max as u64 - 1) / 16,
        "up to the last whole page"
    );
}

#[test]
fn a_lease_is_charged_in_every_paged_space() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let mut cfg = EngineConfig {
        pool_bytes: 128 * 1024,
        ..kv_only_config()
    };
    let mut second = cfg.spaces[0].clone();
    second.space_id = 3;
    second.name = "kv.second";
    cfg.spaces.push(second);
    let host = EngineHost::spawn(move || (MockEngine::new(cfg), None)).expect("spawn");
    let d = Daemon::with_options(store, host, Box::new(MockCodec), Default::default()).unwrap();
    let stats = d.sched_stats();
    let s = run_session(&d, 0..96, 16);
    d.pin(s, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    assert_eq!(stats.pinned_bytes.load(Ordering::Relaxed), 2 * 7 * 1024);
}

#[test]
fn a_pinned_lane_skips_the_lossy_park() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(kv_only_config()), None)).expect("spawn");
    let d = Daemon::with_options(
        store,
        host,
        Box::new(MockCodec),
        superfluid_daemon::DaemonOptions {
            park_dir: Some(dir.join("park")),
            park_lossy: true,
            ..Default::default()
        },
    )
    .unwrap();
    let stats = d.sched_stats();
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..96).collect()).unwrap();
    d.pin(s, 60_000).unwrap();
    d.generate(s, 16).unwrap();
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        1,
        "leased at retire"
    );
    run_session(&d, 1000..1096, 16);
    assert!(stats.parks_lossy.load(Ordering::Relaxed) >= 1);
}

#[test]
fn an_unanchored_request_no_longer_leases_at_retire() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(&dir, 64, Default::default());
    let stats = d.sched_stats();
    let stream: Vec<u32> = (0..96).collect();
    let a = d.create(None, GenParams::default()).unwrap();
    d.append(a, None, stream.clone()).unwrap();
    assert_eq!(d.pin_prefix(Some(a), &stream, 64, 60_000).unwrap(), 0);
    d.unanchor_pins(a);
    d.generate(a, 4).unwrap();
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        0,
        "no anchor, no lease"
    );
    assert_eq!(d.pin_prefix(None, &stream, 64, 60_000).unwrap(), 64);
}

#[test]
fn the_pin_table_cap_never_drops_a_session_pin() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(&dir, 64, Default::default());
    let stats = d.sched_stats();
    let s = run_session(&d, 0..96, 16);
    d.pin(s, 60_000).unwrap();
    for i in 0..300u32 {
        let stream: Vec<u32> = (i * 1000..i * 1000 + 40).collect();
        d.pin_prefix(None, &stream, 32, 60_000).unwrap();
    }
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        1,
        "the session pin still holds"
    );
}

#[test]
fn a_disabled_pin_leaves_the_lossy_park_alone() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(|| (MockEngine::new(kv_only_config()), None)).expect("spawn");
    let d = Daemon::with_options(
        store,
        host,
        Box::new(MockCodec),
        superfluid_daemon::DaemonOptions {
            park_dir: Some(dir.join("park")),
            park_lossy: true,
            pin_budget_pct: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let stats = d.sched_stats();
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..96).collect()).unwrap();
    d.pin(s, 60_000).unwrap();
    d.generate(s, 16).unwrap();
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        0,
        "enforcement is off"
    );
    assert_eq!(
        stats.parks_lossy.load(Ordering::Relaxed),
        1,
        "the lossy tier stands"
    );
}

#[test]
fn a_lease_is_charged_for_its_pinned_blob() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host =
        EngineHost::spawn(|| (MockEngine::new(EngineConfig::default()), None)).expect("spawn");
    let d = Daemon::with_options(store, host, Box::new(MockCodec), Default::default()).unwrap();
    let stats = d.sched_stats();
    let s = run_session(&d, 0..96, 16);
    d.pin(s, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    assert_eq!(stats.pinned_bytes.load(Ordering::Relaxed), 6 * 1024 + 4096);
}

#[test]
fn an_idle_daemon_releases_expired_pins() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(&dir, 64, Default::default());
    let stats = d.sched_stats();
    let s = run_session(&d, 0..96, 16);
    d.pin(s, 100).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    for _ in 0..500 {
        if stats.pins_held.load(Ordering::Relaxed) == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 0);
    assert_eq!(stats.pins_expired.load(Ordering::Relaxed), 1);
}

#[test]
fn a_seed_length_is_common_to_every_space() {
    let d = mock_daemon(&test_dir().join("wal.log"));
    let first: Vec<u32> = (0..64).collect();
    let a = d.create(None, GenParams::default()).unwrap();
    d.append(a, None, first.clone()).unwrap();
    d.generate(a, 8).unwrap();
    let b = d.create(None, GenParams::default()).unwrap();
    let mut prompt = first[..56].to_vec();
    prompt.extend(1000..1040);
    d.append(b, None, prompt).unwrap();
    assert_eq!(d.generate(b, 4).unwrap().warm_prefix, 32);
}

#[test]
fn a_short_lived_lease_is_renewed_in_time() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon_ttl(&dir, 64, 1);
    let stats = d.sched_stats();
    let pinned = run_session(&d, 0..96, 16);
    d.pin(pinned, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    for i in 1..40u32 {
        run_session(&d, i * 10_000..i * 10_000 + 600, 2);
    }
    assert_eq!(
        stats.pins_yielded.load(Ordering::Relaxed),
        0,
        "nothing needed the pin to yield"
    );
    assert!(stats.ticks.load(Ordering::Relaxed) > 60);
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1, "still leased");
    assert_eq!(cached_prefix(&d, 0..96), 80, "the pinned prefix survived");
}

#[test]
fn a_zero_tick_lease_leaves_pins_unenforced() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon_ttl(&dir, 64, 0);
    let stats = d.sched_stats();
    let s = run_session(&d, 0..96, 16);
    assert!(d.pin(s, 60_000).unwrap() > 0, "the pin is still recorded");
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 0);
    assert_eq!(stats.pinned_bytes.load(Ordering::Relaxed), 0);
    assert!(d.inspect(s).unwrap().pin_deadline_unix_ms > 0);
    assert_eq!(
        cached_prefix(&d, 0..96),
        80,
        "admission seeds are unaffected"
    );
}

#[test]
fn clearing_a_pin_releases_its_lease() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(&dir, 64, Default::default());
    let stats = d.sched_stats();
    let s = run_session(&d, 0..96, 16);
    d.pin(s, 60_000).unwrap();
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        1,
        "already cached: leased at once"
    );
    d.pin(s, 0).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 0);
    assert_eq!(stats.pinned_bytes.load(Ordering::Relaxed), 0);
}

#[test]
fn pins_past_the_budget_yield_oldest_first() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(
        &dir,
        32,
        superfluid_daemon::DaemonOptions {
            pin_budget_pct: 30,
            ..Default::default()
        },
    );
    let stats = d.sched_stats();
    let old = run_session(&d, 0..96, 16);
    let new = run_session(&d, 1000..1096, 16);
    d.pin(old, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    d.pin(new, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1, "one pin fits");
    assert_eq!(
        stats.pins_yielded.load(Ordering::Relaxed),
        1,
        "the older one yielded"
    );
    assert!(stats.pinned_bytes.load(Ordering::Relaxed) <= 9 * 1024);
}

#[test]
fn a_pressure_shortfall_yields_pins() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(
        &dir,
        32,
        superfluid_daemon::DaemonOptions {
            pressure_high_pct: 50,
            pressure_low_pct: 40,
            ..Default::default()
        },
    );
    let stats = d.sched_stats();
    let pinned = run_session(&d, 0..96, 16);
    d.pin(pinned, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 1);
    run_session(&d, 1000..1200, 16);
    assert!(
        stats.pins_yielded.load(Ordering::Relaxed) >= 1,
        "the pin yielded to live work"
    );
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 0);
    assert_eq!(
        cached_prefix(&d, 0..96),
        0,
        "and its prefix was evictable again"
    );
}

#[test]
fn an_unplaceable_tick_yields_pins_instead_of_failing() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let d = pin_daemon(
        &dir,
        32,
        superfluid_daemon::DaemonOptions {
            pressure_high_pct: 99,
            pressure_low_pct: 98,
            ..Default::default()
        },
    );
    let stats = d.sched_stats();
    let a = run_session(&d, 0..96, 16);
    let b = run_session(&d, 1000..1096, 16);
    d.pin(a, 60_000).unwrap();
    d.pin(b, 60_000).unwrap();
    assert_eq!(stats.pins_held.load(Ordering::Relaxed), 2);
    run_session(&d, 5000..5300, 8);
    assert!(stats.pins_yielded.load(Ordering::Relaxed) >= 1);
}

#[test]
fn a_recorded_pin_is_re_enforced_after_restart() {
    use std::sync::atomic::Ordering;
    let dir = test_dir();
    let s = {
        let d = pin_daemon(&dir, 64, Default::default());
        let s = run_session(&d, 0..96, 16);
        d.pin(s, 60_000).unwrap();
        s
    };
    let d = pin_daemon(&dir, 64, Default::default());
    let stats = d.sched_stats();
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        0,
        "a fresh engine caches nothing"
    );
    d.generate(s, 4).unwrap();
    assert_eq!(
        stats.pins_held.load(Ordering::Relaxed),
        1,
        "re-leased at the session's retire"
    );
}

struct TerminatorCodec;

const TERM: u32 = 0x900;

impl superfluid_daemon::TextCodec for TerminatorCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        MockCodec.encode(text)
    }
    fn token_bytes(&self, token: u32) -> Vec<u8> {
        MockCodec.token_bytes(token)
    }
    fn turn_terminators(&self) -> Vec<u32> {
        vec![TERM]
    }
}

#[test]
fn a_declared_terminator_ends_the_turn_and_stays_in_the_span() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let mut script = vec![0x101, 0x102, 0x103, TERM];
    script.extend(0x200..0x230);

    let store = SessionStore::open(&wal).expect("open store");
    let s2 = script.clone();
    let host = EngineHost::spawn(move || {
        (MockEngine::new(EngineConfig { scripted: s2.clone(), ..Default::default() }), None)
    })
    .expect("spawn");
    let d = Daemon::with_options(
        store,
        host,
        Box::new(TerminatorCodec),
        superfluid_daemon::DaemonOptions { tick_decode_budget: 1, ..Default::default() },
    )
    .expect("daemon");

    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, Some("hi".into()), vec![100, 101]).unwrap();
    let out = d.generate(session, script.len() as u32).unwrap();

    let produced = {
        let store = d.store();
        let store = store.lock().unwrap();
        let s = store.session(session).unwrap();
        generated_tokens(&s.events)
    };

    assert_eq!(
        produced,
        vec![0x101, 0x102, 0x103, TERM],
        "the turn must end ON the terminator, with the terminator kept"
    );
    assert_eq!(out.finish, finish::EOS, "a terminated turn ends like an EOS turn");
}

#[test]
fn termination_is_tick_granular_at_the_default_slice() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let mut script = vec![0x101, 0x102, 0x103, TERM];
    script.extend(0x200..0x230);

    let store = SessionStore::open(&wal).expect("open store");
    let s2 = script.clone();
    let host = EngineHost::spawn(move || {
        (MockEngine::new(EngineConfig { scripted: s2.clone(), ..Default::default() }), None)
    })
    .expect("spawn");
    let d = Daemon::new(store, host, Box::new(TerminatorCodec), 8);

    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, Some("hi".into()), vec![100, 101]).unwrap();
    d.generate(session, script.len() as u32).unwrap();
    let produced = {
        let store = d.store();
        let store = store.lock().unwrap();
        let s = store.session(session).unwrap();
        generated_tokens(&s.events)
    };
    assert!(produced.contains(&TERM), "the terminator was generated: {produced:?}");
    assert!(
        produced.len() < script.len(),
        "the lane must stop before the script's tail, got {} of {}",
        produced.len(),
        script.len()
    );
    assert!(
        produced.len() <= 32,
        "overshoot must be bounded by one tick slice (32), got {}",
        produced.len()
    );
}

#[test]
fn without_a_declared_terminator_the_same_script_runs_on() {
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let mut script = vec![0x101, 0x102, 0x103, TERM];
    script.extend(0x200..0x230);

    let store = SessionStore::open(&wal).expect("open store");
    let s2 = script.clone();
    let host = EngineHost::spawn(move || {
        (MockEngine::new(EngineConfig { scripted: s2.clone(), ..Default::default() }), None)
    })
    .expect("spawn");
    let d = Daemon::new(store, host, Box::new(MockCodec), 8);

    let session = d.create(None, GenParams::default()).unwrap();
    d.append(session, Some("hi".into()), vec![100, 101]).unwrap();
    d.generate(session, script.len() as u32).unwrap();
    let produced = {
        let store = d.store();
        let store = store.lock().unwrap();
        let s = store.session(session).unwrap();
        generated_tokens(&s.events)
    };
    assert_eq!(produced, script, "no terminator declared: the whole script runs");
}

#[test]
fn park_sweep_bounds_the_directory_oldest_first() {
    let dir = std::env::temp_dir().join(format!("superfluid-park-sweep-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");

    let base = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000);
    for (i, age) in [3u64, 2, 1, 0].into_iter().enumerate() {
        let path = dir.join(format!("{i}.park"));
        std::fs::write(&path, vec![0u8; 1024]).expect("write");
        let f = std::fs::OpenOptions::new().write(true).open(&path).expect("open");
        let t = base + std::time::Duration::from_secs(100 - age);
        f.set_times(std::fs::FileTimes::new().set_modified(t)).expect("set mtime");
    }

    let freed = superfluid_daemon::park::sweep(&dir, 2 * 1024);
    assert_eq!(freed, 2 * 1024, "reclaimed the two oldest");

    let mut left: Vec<String> = std::fs::read_dir(&dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    assert_eq!(left, vec!["2.park".to_string(), "3.park".to_string()]);

    assert_eq!(superfluid_daemon::park::sweep(&dir, 2 * 1024), 0, "idempotent");

    superfluid_daemon::park::remove(&dir, 3);
    assert!(!dir.join("3.park").exists(), "remove deletes by session id");

    let _ = std::fs::remove_dir_all(&dir);
}

fn mock_daemon_with_rings(token_in_slots: u32, token_out_slots: u32, lanes: usize) -> Daemon {
    let store = SessionStore::open(&test_dir().join("wal.log")).expect("open store");
    let host = EngineHost::spawn(move || {
        let cfg = EngineConfig {
            tick_delay: std::time::Duration::from_millis(100),
            ..EngineConfig::default()
        };
        let vocab = cfg.vocab;
        let specs = vec![
            linkw::RingSpec {
                ring_id: TOKEN_RING_IN,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + 4096 * 4,
                slots: token_in_slots,
            },
            linkw::RingSpec {
                ring_id: TOKEN_RING_OUT,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + 1024 * 4,
                slots: token_out_slots,
            },
            linkw::RingSpec {
                ring_id: LOGITS_RING,
                kind: linkw::RingKind::Logits,
                slot_bytes: 16 + vocab * 4,
                slots: 8,
            },
        ];
        (MockEngine::new(cfg), Some(specs))
    })
    .expect("spawn");
    Daemon::new(store, host, Box::new(MockCodec), lanes)
}

fn generate_all_at_once(d: Daemon, n: u32) {
    let d = Arc::new(d);
    let first = {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append(s, None, (0..16).collect()).unwrap();
        let d = Arc::clone(&d);
        std::thread::spawn(move || d.generate(s, 8))
    };
    std::thread::sleep(std::time::Duration::from_millis(30));
    let rest: Vec<_> = (1..n)
        .map(|i| {
            let s = d.create(None, GenParams::default()).unwrap();
            d.append(s, None, (i * 100..i * 100 + 16).collect()).unwrap();
            let d = Arc::clone(&d);
            std::thread::spawn(move || d.generate(s, 8))
        })
        .collect();
    for (i, h) in std::iter::once(first).chain(rest).enumerate() {
        let out = h.join().unwrap().unwrap_or_else(|e| panic!("request {i} failed: {e}"));
        assert_eq!(out.tokens_generated, 8, "request {i}");
    }
}

#[test]
fn an_admission_burst_larger_than_the_token_in_ring_waits_a_tick_instead_of_failing() {
    generate_all_at_once(mock_daemon_with_rings(4, 64, 8), 8);
}

#[test]
fn lanes_never_outnumber_the_token_out_slots_a_tick_writes() {
    generate_all_at_once(mock_daemon_with_rings(64, 4, 8), 8);
}

#[test]
fn overlong_prompt_is_a_typed_client_error_and_frees_the_session() {
    const CTX: u32 = 64;
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).expect("open store");
    let host = EngineHost::spawn(|| {
        let cfg = EngineConfig::default();
        let vocab = cfg.vocab;
        let e = MockEngine::new(cfg);
        let specs = vec![
            linkw::RingSpec {
                ring_id: TOKEN_RING_IN,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + CTX * 4,
                slots: 64,
            },
            linkw::RingSpec {
                ring_id: TOKEN_RING_OUT,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + 1024 * 4,
                slots: 64,
            },
            linkw::RingSpec {
                ring_id: LOGITS_RING,
                kind: linkw::RingKind::Logits,
                slot_bytes: 16 + vocab * 4,
                slots: 8,
            },
        ];
        (e, Some(specs))
    })
    .expect("spawn");
    let d = Daemon::new(store, host, Box::new(MockCodec), 8);
    assert_eq!(d.max_stream_tokens(), CTX as u64, "ceiling comes from the ring slot");

    let session = d.create(None, GenParams::default()).unwrap();
    let overlong: Vec<u32> = (0..CTX + 8).collect();
    d.append(session, Some("too much".into()), overlong.clone()).unwrap();

    let err = d.generate(session, 4).expect_err("an overlong prompt cannot be served");
    match err {
        superfluid_daemon::DaemonError::StreamTooLong { len, max } => {
            assert_eq!(len, overlong.len() as u64);
            assert_eq!(max, CTX as u64);
            let msg = superfluid_daemon::DaemonError::StreamTooLong { len, max }.to_string();
            assert!(msg.contains(&len.to_string()), "message names the request size: {msg}");
            assert!(msg.contains(&max.to_string()), "message names the ceiling: {msg}");
        }
        other => panic!("expected StreamTooLong, got {other:?}"),
    }

    let fresh = d.create(None, GenParams::default()).unwrap();
    d.append(fresh, Some("fits".into()), (0..8).collect()).unwrap();
    let out = d.generate(fresh, 4).expect("a prompt within the ceiling still generates");
    assert_eq!(out.tokens_generated, 4);
}

#[test]
fn the_turn_opener_cannot_smuggle_a_stream_past_the_ceiling() {
    const CTX: u32 = 64;
    let dir = test_dir();
    let store = SessionStore::open(&dir.join("wal.log")).expect("open store");
    let host = EngineHost::spawn(|| {
        let cfg = EngineConfig::default();
        let vocab = cfg.vocab;
        let e = MockEngine::new(cfg);
        let specs = vec![
            linkw::RingSpec {
                ring_id: TOKEN_RING_IN,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + CTX * 4,
                slots: 64,
            },
            linkw::RingSpec {
                ring_id: TOKEN_RING_OUT,
                kind: linkw::RingKind::Tokens,
                slot_bytes: 16 + 1024 * 4,
                slots: 64,
            },
            linkw::RingSpec {
                ring_id: LOGITS_RING,
                kind: linkw::RingKind::Logits,
                slot_bytes: 16 + vocab * 4,
                slots: 8,
            },
        ];
        (e, Some(specs))
    })
    .expect("spawn");
    let d = Daemon::new(store, host, Box::new(MockChatCodec), 8);

    let session = d.create(None, GenParams::default()).unwrap();
    d.append_message(session, superfluid_daemon::wal::role::USER, "x".repeat(60)).unwrap();
    {
        let store = d.store();
        let store = store.lock().unwrap();
        assert_eq!(
            store.session(session).unwrap().tokens.len() as u64,
            CTX as u64,
            "the stored stream must sit exactly on the ceiling for this to bite"
        );
    }

    let err = d.generate(session, 4).expect_err("the opener pushes this over the ceiling");
    match err {
        superfluid_daemon::DaemonError::StreamTooLong { len, max } => {
            assert_eq!(max, CTX as u64);
            assert!(len > max, "the reported length is the STAGED total: {len} vs {max}");
        }
        other => panic!("the opener overflow must stay typed, got {other:?}"),
    }

    let store = d.store();
    let store = store.lock().unwrap();
    assert_eq!(
        store.session(session).unwrap().tokens.len() as u64,
        CTX as u64,
        "a refused generation must not leave its opener in the log"
    );
}

#[test]
fn an_unbalanced_marker_in_earlier_input_does_not_prime_the_next_turn() {
    use superfluid_daemon::codec::{MOCK_THINK_OPEN, MOCK_TOOL_CLOSE, MOCK_TOOL_OPEN};
    use superfluid_daemon::wal::{channel, role};
    use superfluid_daemon::TextCodec as _;
    let dir = test_dir();
    let wal = dir.join("wal.log");
    let mut script = MockCodec.encode("Looking.");
    script.push(MOCK_TOOL_OPEN);
    script.extend(MockCodec.encode(r#"{"name": "read", "arguments": {"path": "/a"}}"#));
    script.push(MOCK_TOOL_CLOSE);

    let store = SessionStore::open(&wal).expect("open store");
    let s2 = script.clone();
    let host = EngineHost::spawn(move || {
        (MockEngine::new(EngineConfig { scripted: s2.clone(), ..Default::default() }), None)
    })
    .expect("spawn");
    let d = Daemon::new(store, host, Box::new(MockChatCodec), 8);

    let session = d.create(None, GenParams::default()).unwrap();
    d.append_message(session, role::USER, "read the file".into()).unwrap();
    d.append(session, None, vec![MOCK_THINK_OPEN]).unwrap();
    d.generate(session, 64).unwrap();

    let store = d.store();
    let store = store.lock().unwrap();
    let s = store.session(session).unwrap();
    let channels: Vec<u32> = s
        .events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Generated { channel, .. } => Some(*channel),
            _ => None,
        })
        .collect();
    assert!(
        channels.contains(&channel::TEXT) && !channels.contains(&channel::REASONING),
        "the earlier opener primed the turn: channels={channels:?}"
    );
    assert!(
        s.events.iter().any(|e| matches!(&e.body, EventBody::ToolUse { name, .. } if name == "read")),
        "the tool call was not parsed: {:?}",
        s.events.iter().map(|e| &e.body).collect::<Vec<_>>()
    );
}

fn chat_daemon(dir: &std::path::Path, script: Vec<u32>) -> Daemon {
    let store = SessionStore::open(&dir.join("wal.log")).unwrap();
    let host = EngineHost::spawn(move || {
        (MockEngine::new(EngineConfig { scripted: script.clone(), ..Default::default() }), None)
    })
    .expect("spawn");
    Daemon::new(store, host, Box::new(MockChatCodec), 8)
}

fn system_span(d: &Daemon, session: u64) -> Vec<u32> {
    let store = d.store();
    let store = store.lock().unwrap();
    let s = store.session(session).unwrap();
    match &s.events.last().unwrap().body {
        EventBody::Message { role, span, .. } => {
            assert_eq!(*role, superfluid_daemon::wal::role::SYSTEM);
            span.clone()
        }
        other => panic!("expected a system message, got {other:?}"),
    }
}

#[test]
fn declared_tools_render_as_the_http_routes_render_them() {
    use superfluid_daemon::codec::TextCodec;
    let dir = test_dir();
    let d = chat_daemon(&dir, Vec::new());
    let schema = serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}});
    let openai = serde_json::json!({"type": "function", "function": {"name": "get_weather", "description": "Weather now", "parameters": schema}});
    let http = d.create(None, GenParams::default()).unwrap();
    d.append_system_full(http, Some("Be brief.".into()), vec![openai.to_string()], &serde_json::Map::new())
        .unwrap();
    let want = system_span(&d, http);
    assert!(MockChatCodec.decode(&want).contains("get_weather"));

    let pretty = serde_json::to_string_pretty(&openai).unwrap();
    let bare = serde_json::json!({"name": "get_weather", "description": "Weather now", "parameters": schema}).to_string();
    let anthropic = serde_json::json!({"name": "get_weather", "description": "Weather now", "input_schema": schema}).to_string();
    for tool in [pretty, bare, anthropic] {
        let s = d.create(None, GenParams::default()).unwrap();
        d.append_system(s, Some("Be brief.".into()), vec![tool.clone()]).unwrap();
        assert_eq!(system_span(&d, s), want, "{tool}");
    }
    let s = d.create(None, GenParams::default()).unwrap();
    d.append_system(s, None, vec![openai.to_string()]).unwrap();
    assert_eq!(d.read(s, 0).unwrap().len(), 2);
}

#[test]
fn append_system_refuses_bad_tools_and_a_late_declaration() {
    use superfluid_daemon::DaemonError;
    let dir = test_dir();
    let d = chat_daemon(&dir, Vec::new());
    let s = d.create(None, GenParams::default()).unwrap();
    let ok = r#"{"type":"function","function":{"name":"f"}}"#.to_string();
    for (tools, index) in [
        (vec!["not json".to_string()], 0),
        (vec!["[]".into()], 0),
        (vec![ok.clone(), r#"{"type":"function","function":{"description":"x"}}"#.into()], 1),
        (vec![r#"{"type":"retrieval","function":{"name":"f"}}"#.into()], 0),
        (vec![r#"{"name":"f","parameters":"object"}"#.into()], 0),
        (vec![r#"{"function":"f"}"#.into()], 0),
    ] {
        match d.append_system(s, Some("sys".into()), tools.clone()) {
            Err(DaemonError::InvalidTool { index: i, .. }) => assert_eq!(i, index, "{tools:?}"),
            other => panic!("{tools:?}: {other:?}"),
        }
    }
    assert!(matches!(d.append_system(s, None, Vec::new()), Err(DaemonError::Protocol(_))));
    assert_eq!(d.read(s, 1).unwrap().len(), 0, "a refusal commits nothing");

    d.append_message(s, superfluid_daemon::wal::role::USER, "hi".into()).unwrap();
    assert!(matches!(
        d.append_system(s, Some("sys".into()), vec![ok.clone()]),
        Err(DaemonError::SystemNotFirst(id)) if id == s
    ));
    let child = d.fork(s, 2, None).unwrap();
    assert!(matches!(
        d.append_system(child, None, vec![ok]),
        Err(DaemonError::SystemNotFirst(_))
    ));
}

#[test]
fn a_declared_tool_call_is_answered_over_the_socket() {
    use superfluid_daemon::codec::{TextCodec, MOCK_TOOL_CLOSE, MOCK_TOOL_OPEN};
    let mut script = vec![MOCK_TOOL_OPEN];
    script.extend(MockCodec.encode(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#));
    script.push(MOCK_TOOL_CLOSE);
    let dir = test_dir();
    let socket = dir.join("superfluid.sock");
    let daemon = Arc::new(chat_daemon(&dir, script.clone()));
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let serving = Arc::clone(&daemon);
    std::thread::spawn(move || {
        let _ = api::serve(listener, serving);
    });
    let mut client = api::NativeClient::connect(&socket).expect("connect");
    let session = match client
        .request(&api::Request::Create { parent: None, params: GenParams::default() })
        .unwrap()
    {
        api::Response::Created { session } => session,
        other => panic!("unexpected {other:?}"),
    };
    let tool = r#"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}"#;
    match client
        .request(&api::Request::AppendSystem {
            session,
            text: Some("Use tools.".into()),
            tools: vec![tool.into()],
        })
        .unwrap()
    {
        api::Response::Committed { event } => {
            assert!(matches!(event.body, EventBody::Message { role: 0, .. }), "{event:?}")
        }
        other => panic!("unexpected {other:?}"),
    }
    client
        .request(&api::Request::AppendMessage { session, role: 1, text: "Weather in Paris?".into() })
        .unwrap();
    let call = match client.generate_stream(session, script.len() as u32, |_| {}).unwrap() {
        api::Response::Generated { events, .. } => events
            .into_iter()
            .find_map(|e| match e.body {
                EventBody::ToolUse { name, arguments } => Some((e.event_id, name, arguments)),
                _ => None,
            })
            .expect("a ToolUse event"),
        other => panic!("unexpected {other:?}"),
    };
    assert_eq!(call.1, "get_weather");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&call.2).unwrap(),
        serde_json::json!({"city": "Paris"})
    );
    match client.request(&api::Request::OpenToolCalls { session }).unwrap() {
        api::Response::ToolCalls { calls } => assert_eq!(calls, vec![call.clone()]),
        other => panic!("unexpected {other:?}"),
    }
    match client
        .request(&api::Request::AppendToolResult { session, call_id: call.0, content: "18C".into() })
        .unwrap()
    {
        api::Response::Committed { event } => {
            assert!(matches!(event.body, EventBody::ToolResult { call_id, .. } if call_id == call.0))
        }
        other => panic!("unexpected {other:?}"),
    }
    match client
        .request(&api::Request::AppendSystem { session, text: None, tools: vec!["{".into()] })
        .unwrap()
    {
        api::Response::Err { message } => {
            assert!(message.contains("tool 0 is not a function declaration"), "{message}")
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn inspecting_a_session_while_it_starts_generating_does_not_deadlock() {
    let d = Arc::new(mock_daemon(&test_dir().join("wal.log")));
    let s = d.create(None, GenParams::default()).unwrap();
    d.append(s, None, (0..16).collect()).unwrap();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let gen = {
        let (d, tx) = (Arc::clone(&d), done_tx.clone());
        std::thread::spawn(move || {
            for _ in 0..200 {
                d.generate(s, 2).unwrap();
            }
            tx.send(()).unwrap();
        })
    };
    let look = {
        let (d, tx) = (Arc::clone(&d), done_tx);
        std::thread::spawn(move || {
            for _ in 0..2000 {
                d.inspect(s).unwrap();
            }
            tx.send(()).unwrap();
        })
    };
    for _ in 0..2 {
        done_rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("inspect and generate take the store and active locks in one order");
    }
    gen.join().unwrap();
    look.join().unwrap();
}
