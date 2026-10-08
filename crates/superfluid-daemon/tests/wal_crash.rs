use superfluid_daemon::{EventBody, GenParams, SessionStore};

fn spans_of(events: &[superfluid_daemon::CommittedEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::Appended { span, .. }
            | EventBody::Message { span, .. }
            | EventBody::GenerationPrompt { span }
            | EventBody::ToolResult { span, .. }
            | EventBody::Generated { span, .. } => Some(span.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

#[test]
fn every_crash_point_recovers_with_invariants_intact() {
    let dir = std::env::temp_dir().join(format!("superfluid-walcrash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("wal.log");

    let (full_a, full_b, full_c) = {
        let mut store = SessionStore::open(&wal).unwrap();
        let a = store.create(None, GenParams::default()).unwrap();
        let b = store
            .create(
                Some(a),
                GenParams {
                    seed: 7,
                    ..Default::default()
                },
            )
            .unwrap();
        store.append(a, Some("héllo".into()), vec![1, 2, 3]).unwrap();
        store.append(b, None, vec![9, 8]).unwrap();
        store
            .commit_generated(a, vec![4, 5], "ab".into(), 0, 0)
            .unwrap();
        store
            .commit_generated(b, vec![7], "—".into(), 1, 0)
            .unwrap();
        store.append(a, Some("more".into()), vec![6]).unwrap();
        let call = store
            .commit_tool_use(a, "get_weather".into(), r#"{"city":"Paris"}"#.into())
            .unwrap();
        store
            .commit_tool_result(a, call.event_id, "18C".into(), vec![11, 12])
            .unwrap();
        store
            .commit_generated(a, Vec::new(), String::new(), 0, 4)
            .unwrap();
        let c = store.fork(a, 4, None).unwrap();
        store.append(c, None, vec![21, 22]).unwrap();
        store.append(a, None, vec![31]).unwrap();
        (
            store.session(a).unwrap().events.clone(),
            store.session(b).unwrap().events.clone(),
            store.session(c).unwrap().events.clone(),
        )
    };

    let bytes = std::fs::read(&wal).unwrap();
    assert!(bytes.len() > 100, "log substantial enough to sweep");

    for cut in 0..=bytes.len() {
        let case = dir.join("cut.log");
        std::fs::write(&case, &bytes[..cut]).unwrap();

        let mut store = SessionStore::open(&case)
            .unwrap_or_else(|e| panic!("crash at byte {cut}: recovery failed: {e}"));

        for (sid, full) in [(1u64, &full_a), (2u64, &full_b), (3u64, &full_c)] {
            let Ok(s) = store.session(sid) else {
                continue;
            };
            for (i, e) in s.events.iter().enumerate() {
                assert_eq!(e.event_id, i as u64, "crash at byte {cut}: id gap");
            }
            if sid == 3 {
                assert_eq!(s.base, 4, "crash at byte {cut}: fork base");
                assert_eq!(s.parent, Some(1));
                let parent = store.session(1).expect("a fork implies its parent");
                for i in 0..4 {
                    assert_eq!(
                        s.events[i].body, parent.events[i].body,
                        "crash at byte {cut}: inherited prefix diverged"
                    );
                }
            }
            assert_eq!(
                s.tokens,
                spans_of(&s.events),
                "crash at byte {cut}: stream diverged from spans"
            );
            let recovered: Vec<_> = s
                .events
                .iter()
                .filter(|e| !matches!(e.body, EventBody::EpochBump))
                .collect();
            assert!(
                recovered.len() <= full.len(),
                "crash at byte {cut}: extra events appeared"
            );
            for (rec, truth) in recovered.iter().zip(full.iter()) {
                assert_eq!(
                    rec.body, truth.body,
                    "crash at byte {cut}: history rewritten"
                );
            }
            for e in &s.events {
                if let EventBody::ToolUse { .. } = &e.body {
                    let closed = s.events.iter().any(|r| {
                        matches!(&r.body, EventBody::ToolResult { call_id, .. }
                            if *call_id == e.event_id)
                    });
                    assert_eq!(
                        !closed,
                        s.open_tool_calls.contains_key(&e.event_id),
                        "crash at byte {cut}: ledger diverged from the log"
                    );
                }
            }
            let max_prior = recovered.iter().map(|e| e.epoch).max().unwrap_or(0);
            assert!(
                s.epoch > max_prior,
                "crash at byte {cut}: epoch not bumped past the prior owner"
            );
            let max_all = s.events.iter().map(|e| e.epoch).max().unwrap_or(0);
            assert_eq!(s.epoch, max_all, "crash at byte {cut}: bump not durable");
        }

        if let Ok(s) = store.session(1) {
            let next = s.events.len() as u64;
            let e = store.append(1, None, vec![42]).unwrap();
            assert_eq!(e.event_id, next, "crash at byte {cut}: bad next id");
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_acknowledged_record_fails_closed() {
    let dir = std::env::temp_dir().join(format!("superfluid-walcorrupt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("wal.log");
    {
        let mut store = SessionStore::open(&wal).unwrap();
        let s = store.create(None, GenParams::default()).unwrap();
        store.append(s, None, vec![1, 2, 3]).unwrap();
        store.append(s, None, vec![4, 5, 6]).unwrap();
    }
    let mut bytes = std::fs::read(&wal).unwrap();
    bytes[16] ^= 0xFF;
    std::fs::write(&wal, &bytes).unwrap();
    let Err(err) = SessionStore::open(&wal).map(|_| ()) else {
        panic!("corruption must fail closed");
    };
    assert!(
        matches!(
            err,
            superfluid_daemon::DaemonError::WalCorrupt | superfluid_daemon::DaemonError::Codec(_)
        ),
        "unexpected error: {err:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn epochs_are_durably_monotonic_across_reopens() {
    let dir = std::env::temp_dir().join(format!("superfluid-walepoch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("wal.log");
    let session = {
        let mut store = SessionStore::open(&wal).unwrap();
        let s = store.create(None, GenParams::default()).unwrap();
        store.append(s, None, vec![1]).unwrap();
        s
    };
    let mut seen = Vec::new();
    for _ in 0..3 {
        let store = SessionStore::open(&wal).unwrap();
        seen.push(store.session(session).unwrap().epoch);
    }
    assert!(
        seen.windows(2).all(|w| w[1] > w[0]),
        "epochs reused across recoveries: {seen:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn purge_transaction_is_all_or_nothing_at_every_cut() {
    use superfluid_daemon::wal::PurgeMode;
    let dir = std::env::temp_dir().join(format!("superfluid-purgecrash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("wal.log");
    let (root, kid, kid_tokens) = {
        let mut store = SessionStore::open(&wal).unwrap();
        let root = store.create(None, GenParams::default()).unwrap();
        store.append(root, None, vec![1, 2, 3]).unwrap();
        store
            .commit_generated(root, vec![4, 5], "ab".into(), 0, 1)
            .unwrap();
        let kid = store.fork(root, 3, None).unwrap();
        store.append(kid, None, vec![9]).unwrap();
        let call = store
            .commit_tool_use(root, "t".into(), "{}".into())
            .unwrap();
        store.expire_tool_call(root, call.event_id, "purged").unwrap();
        store.purge(root, 0, PurgeMode::Reroot).unwrap();
        let kid_tokens = store.session(kid).unwrap().tokens.clone();
        (root, kid, kid_tokens)
    };
    assert_eq!(kid_tokens, vec![1, 2, 3, 4, 5, 9]);
    let bytes = std::fs::read(&wal).unwrap();
    let mut saw_alive = false;
    let mut saw_purged = false;
    for cut in 0..=bytes.len() {
        let case = dir.join("cut.log");
        std::fs::write(&case, &bytes[..cut]).unwrap();
        let store = SessionStore::open(&case)
            .unwrap_or_else(|e| panic!("crash at byte {cut}: recovery failed: {e}"));
        let Ok(k) = store.session(kid) else {
            continue;
        };
        assert!(
            kid_tokens.starts_with(&k.tokens) && k.tokens.len() >= 5,
            "crash at byte {cut}: child lost its prefix: {:?}",
            k.tokens
        );
        match store.session(root) {
            Ok(r) => {
                saw_alive = true;
                assert!(!k.rerooted, "crash at byte {cut}: rerooted while the parent lives");
                assert!(!r.tokens.is_empty());
            }
            Err(superfluid_daemon::DaemonError::Purged(_)) => {
                saw_purged = true;
                assert!(k.rerooted, "crash at byte {cut}: purged parent, child not re-rooted");
                assert_eq!(k.tokens, kid_tokens);
                let t = store.tombstone(root).unwrap();
                assert!(t.events.is_empty());
                assert_eq!(t.tombstone_ledger.values().next().map(String::as_str), Some("expired"));
            }
            Err(e) => panic!("crash at byte {cut}: {e}"),
        }
    }
    assert!(saw_alive && saw_purged, "the sweep crossed the commit point");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ledger_interleavings_recover_consistently_at_every_cut() {
    use superfluid_daemon::wal::{ledger_state, tool_outcome};
    let dir = std::env::temp_dir().join(format!("superfluid-ledgercrash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let wal = dir.join("wal.log");
    let (a, b, c, l) = {
        let mut store = SessionStore::open(&wal).unwrap();
        let s = store.create(None, GenParams::default()).unwrap();
        store.append(s, None, vec![1, 2]).unwrap();
        let a = store.commit_tool_use(s, "slow".into(), "{}".into()).unwrap().event_id;
        let b = store.commit_tool_use(s, "flaky".into(), "{}".into()).unwrap().event_id;
        let c = store.commit_tool_use(s, "fine".into(), "{}".into()).unwrap().event_id;
        let l = store.commit_tool_use(s, "leased".into(), "{}".into()).unwrap().event_id;
        store.commit_tool_lease(s, l, 1).unwrap();
        store.commit_tool_cancel_request(s, a).unwrap();
        store.commit_tool_outcome(s, a, tool_outcome::CANCELLED, "killed".into()).unwrap();
        store.commit_tool_reconciliation(s, a, "late result: 13 bytes".into()).unwrap();
        store.commit_tool_outcome(s, b, tool_outcome::FAILED, "boom".into()).unwrap();
        store.commit_tool_result(s, c, "ok".into(), vec![9]).unwrap();
        store.expire_tool_call(s, l, "lease expired").unwrap();
        (a, b, c, l)
    };
    let bytes = std::fs::read(&wal).unwrap();
    for cut in 0..=bytes.len() {
        let case = dir.join("cut.log");
        std::fs::write(&case, &bytes[..cut]).unwrap();
        let mut store = SessionStore::open(&case)
            .unwrap_or_else(|e| panic!("crash at byte {cut}: recovery failed: {e}"));
        let Ok(s) = store.session(1) else { continue };
        for (id, e) in &s.ledger {
            let evs = &s.events;
            let has = |pred: &dyn Fn(&EventBody) -> bool| evs.iter().any(|x| pred(&x.body));
            let cancelled = has(&|b| matches!(b, EventBody::ToolOutcome { call_id, outcome, .. } if call_id == id && *outcome == tool_outcome::CANCELLED));
            let failed = has(&|b| matches!(b, EventBody::ToolOutcome { call_id, outcome, .. } if call_id == id && *outcome == tool_outcome::FAILED));
            let result = has(&|b| matches!(b, EventBody::ToolResult { call_id, .. } if call_id == id));
            let expired = has(&|b| matches!(b, EventBody::ToolExpired { call_id, .. } if call_id == id));
            let late = has(&|b| matches!(b, EventBody::ToolReconciliation { call_id, .. } if call_id == id));
            let cancel_req = has(&|b| matches!(b, EventBody::ToolCancelRequested { call_id } if call_id == id));
            let terminals = [cancelled, failed, result, expired].iter().filter(|&&x| x).count();
            assert!(terminals <= 1, "crash at byte {cut}: two terminals on call {id}");
            let expect = if cancelled && late {
                ledger_state::CANCELLED_WITH_LATE_EFFECT
            } else if cancelled {
                ledger_state::CANCELLED
            } else if failed {
                ledger_state::FAILED
            } else if result {
                ledger_state::SUCCEEDED
            } else if expired {
                ledger_state::EXPIRED
            } else if cancel_req {
                ledger_state::CANCEL_REQUESTED
            } else {
                ledger_state::OPEN
            };
            assert_eq!(e.state, expect, "crash at byte {cut}: call {id} state");
            let open = terminals == 0;
            assert_eq!(s.open_tool_calls.contains_key(id), open, "crash at byte {cut}: open set for {id}");
        }
        assert!(store.append(1, None, vec![42]).is_ok());
    }
    let _ = (a, b, c, l);
    let _ = std::fs::remove_dir_all(&dir);
}

fn frames(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut at = 8;
    let mut out = Vec::new();
    while at + 12 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        out.push((at, at + 12 + len));
        at += 12 + len;
    }
    assert_eq!(at, bytes.len(), "the log parses into whole frames");
    out
}

fn three_appends(dir: &std::path::Path) -> (std::path::PathBuf, Vec<u8>) {
    let wal = dir.join("wal.log");
    {
        let mut store = SessionStore::open(&wal).unwrap();
        let s = store.create(None, GenParams::default()).unwrap();
        store.append(s, None, vec![1, 2, 3]).unwrap();
        store.append(s, None, vec![4, 5, 6]).unwrap();
        store.append(s, None, vec![7, 8, 9]).unwrap();
    }
    let bytes = std::fs::read(&wal).unwrap();
    (wal, bytes)
}

#[test]
fn checksum_failed_final_record_is_a_torn_tail() {
    let dir = std::env::temp_dir().join(format!("superfluid-waltorn-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (wal, bytes) = three_appends(&dir);
    let (start, end) = *frames(&bytes).last().unwrap();
    let damaged: Vec<(&str, Vec<u8>)> = vec![
        ("zero-filled payload", {
            let mut b = bytes.clone();
            b[start + 12..end].fill(0);
            b
        }),
        ("zero-filled checksum and payload", {
            let mut b = bytes.clone();
            b[start + 4..end].fill(0);
            b
        }),
        ("garbage payload of the declared length", {
            let mut b = bytes.clone();
            b[start + 12..end].fill(0xA5);
            b
        }),
        ("whole frame zero-filled", {
            let mut b = bytes.clone();
            b[start..end].fill(0);
            b
        }),
        ("zero-filled payload, then zero fill past it", {
            let mut b = bytes.clone();
            b[start + 12..end].fill(0);
            b.extend_from_slice(&[0; 4096]);
            b
        }),
    ];
    for (what, b) in damaged {
        std::fs::write(&wal, &b).unwrap();
        let tokens = {
            let mut store = SessionStore::open(&wal)
                .unwrap_or_else(|e| panic!("{what}: a torn final record must not brick startup: {e}"));
            let s = store.session(1).unwrap();
            let tokens = s.tokens.clone();
            let next = s.next_event_id();
            assert_eq!(store.append(1, None, vec![42]).unwrap().event_id, next, "{what}");
            tokens
        };
        assert_eq!(tokens, vec![1, 2, 3, 4, 5, 6], "{what}: only the torn record is dropped");
        let reopened = std::fs::read(&wal).unwrap();
        assert_eq!(reopened[..start], bytes[..start], "{what}: history kept");
        frames(&reopened);
        let store = SessionStore::open(&wal).unwrap();
        assert_eq!(store.session(1).unwrap().tokens, vec![1, 2, 3, 4, 5, 6, 42], "{what}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn checksum_failure_with_anything_after_it_fails_closed() {
    let dir = std::env::temp_dir().join(format!("superfluid-walmid-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (wal, bytes) = three_appends(&dir);
    let all = frames(&bytes);
    let (start, end) = all[all.len() - 2];
    let (last, last_end) = *all.last().unwrap();
    let damaged: Vec<(&str, Vec<u8>)> = vec![
        ("zero-filled payload mid-log", {
            let mut b = bytes.clone();
            b[start + 12..end].fill(0);
            b
        }),
        ("garbage payload mid-log", {
            let mut b = bytes.clone();
            b[start + 12..end].fill(0xA5);
            b
        }),
        ("bad final record, then non-zero bytes", {
            let mut b = bytes.clone();
            b[last + 12..last_end].fill(0);
            b.extend_from_slice(&[0, 0, 1]);
            b
        }),
    ];
    for (what, b) in damaged {
        std::fs::write(&wal, &b).unwrap();
        let Err(err) = SessionStore::open(&wal).map(|_| ()) else {
            panic!("{what}: corruption must fail closed");
        };
        assert!(
            matches!(err, superfluid_daemon::DaemonError::WalCorrupt),
            "{what}: unexpected error: {err:?}"
        );
        assert_eq!(std::fs::read(&wal).unwrap(), b, "{what}: the log is left as found");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
