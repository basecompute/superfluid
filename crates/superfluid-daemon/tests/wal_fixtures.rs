//! Committed WAL fixtures.

use std::path::PathBuf;

use superfluid_daemon::wal::{PurgeMode, RebaseEdit};
use superfluid_daemon::{EventBody, GenParams, SessionStore};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn open_copy(name: &str) -> SessionStore {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "superfluid-fixture-{}-{n}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let dst = dir.join(name);
    std::fs::copy(fixtures().join(name), &dst).expect("fixture present");
    SessionStore::open(&dst).expect("fixture replays")
}

#[test]
fn p2_forked_fixture_replays() {
    let store = open_copy("wal-p2-forked.wal");
    let root = store.session(1).unwrap();
    let fork = store.session(2).unwrap();
    assert_eq!(root.parent, None);
    assert_eq!(root.base, 0);
    assert_eq!(fork.parent, Some(1));
    assert_eq!(fork.base, 3);
    assert_eq!(root.tokens, vec![1, 2, 3, 4, 5, 6, 7]);
    assert!(root.open_tool_calls.is_empty());
    assert_eq!(fork.tokens, vec![1, 2, 3, 4, 5, 8, 9]);
    assert!(matches!(
        fork.events[3].body,
        EventBody::Forked { parent: 1, fork_at: 3, .. }
    ));
    assert_eq!(fork.events[4].event_id, 4);
    assert!(matches!(&fork.events[4].body, EventBody::Appended { span, .. } if span == &[8, 9]));
    for i in 0..3 {
        assert_eq!(fork.events[i].body, root.events[i].body);
    }
    assert_eq!(fork.params.seed, 7);
}

#[test]
#[ignore]
fn write_p2_forked_fixture() {
    let path = fixtures().join("wal-p2-forked.wal");
    assert!(!path.exists(), "fixtures are never regenerated");
    std::fs::create_dir_all(fixtures()).unwrap();
    let mut store = SessionStore::open(&path).unwrap();
    let a = store.create(None, GenParams::default()).unwrap();
    store.append(a, Some("abc".into()), vec![1, 2, 3]).unwrap();
    store
        .commit_generated(a, vec![4, 5], "de".into(), 0, 1)
        .unwrap();
    let b = store
        .fork(
            a,
            3,
            Some(GenParams {
                seed: 7,
                ..Default::default()
            }),
        )
        .unwrap();
    let call = store
        .commit_tool_use(a, "get_weather".into(), r#"{"city":"Paris"}"#.into())
        .unwrap();
    store
        .commit_tool_result(a, call.event_id, "18C".into(), vec![6])
        .unwrap();
    store.append(b, None, vec![8, 9]).unwrap();
    store.append(a, None, vec![7]).unwrap();
}

#[test]
fn p2_tree_fixture_replays() {
    let store = open_copy("wal-p2-tree.wal");
    assert!(store.tombstone(1).is_some(), "root 1 purged");
    let t1 = store.tombstone(1).unwrap();
    assert_eq!(t1.generation, 1);
    assert_eq!(t1.tombstone_ledger.get(&2).map(String::as_str), Some("expired"));
    let s2 = store.session(2).unwrap();
    assert!(s2.rerooted);
    assert_eq!(s2.title.as_deref(), Some("kept"));
    assert_eq!(s2.meta_version, 1);
    assert_eq!(s2.tokens, vec![1, 2, 3, 4, 9]);
    assert!(s2.events.iter().any(|e| matches!(e.body, EventBody::Rerooted { purged_parent: 1, .. })));
    let s3 = store.session(3).unwrap();
    assert_eq!(s3.parent, Some(2));
    assert!(matches!(&s3.events[s3.base as usize].body, EventBody::Rebased { parent: 2, fork_at: 2, edits, .. } if edits.len() == 1));
    assert_eq!(s3.tokens, vec![1, 2, 3, 4, 42]);
    assert!(store.tombstone(4).is_some() && store.tombstone(5).is_some(), "cascade");
    let mut live = store.session_ids();
    live.sort();
    assert_eq!(live, vec![2, 3]);
}

#[test]
#[ignore]
fn write_p2_tree_fixture() {
    let path = fixtures().join("wal-p2-tree.wal");
    assert!(!path.exists(), "fixtures are never regenerated");
    let mut store = SessionStore::open(&path).unwrap();
    let a = store.create(None, GenParams::default()).unwrap();
    store.append(a, None, vec![1, 2, 3, 4]).unwrap();
    let b = store.fork(a, 2, None).unwrap();
    store.append(b, None, vec![9]).unwrap();
    store.set_meta(b, 0, Some("kept".into()), None).unwrap();
    let c = store
        .rebase_branch(b, 2, None, vec![RebaseEdit::Drop { from: 2, to: 3 }])
        .unwrap();
    store.append(c, None, vec![42]).unwrap();
    let _call = store.commit_tool_use(a, "t".into(), "{}".into()).unwrap();
    let open: Vec<u64> = store.session(a).unwrap().open_tool_calls.keys().copied().collect();
    for id in open {
        store.expire_tool_call(a, id, "purged").unwrap();
    }
    store.purge(a, 0, PurgeMode::Reroot).unwrap();
    let d = store.create(None, GenParams::default()).unwrap();
    store.append(d, None, vec![7]).unwrap();
    let e = store.fork(d, 2, None).unwrap();
    let _ = e;
    store.purge(d, 0, PurgeMode::Cascade).unwrap();
}

#[test]
fn v1_logs_stay_v1_and_new_logs_are_timestamped_v2() {
    let mut old = open_copy("wal-p2-forked.wal");
    assert_eq!(old.wal_version(), 1);
    assert!(old.session(1).unwrap().events.iter().all(|e| e.ts_unix_ms == 0));
    let e = old.append(1, None, vec![77]).unwrap();
    assert_eq!(e.ts_unix_ms, 0, "a v1 file cannot carry timestamps");
    drop(old);
    let dir = std::env::temp_dir().join(format!("superfluid-fixture-v2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wal.log");
    let s = {
        let mut store = SessionStore::open(&path).unwrap();
        assert_eq!(store.wal_version(), 2);
        let s = store.create(None, GenParams::default()).unwrap();
        let e = store.append(s, None, vec![1]).unwrap();
        assert!(e.ts_unix_ms > 1_700_000_000_000, "commit wall-clock recorded");
        s
    };
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"BRTWAL02");
    let store = SessionStore::open(&path).unwrap();
    assert_eq!(store.wal_version(), 2);
    assert!(store.session(s).unwrap().events.iter().all(|e| e.ts_unix_ms > 0));
}

#[test]
fn p2_ledger_fixture_replays() {
    use superfluid_daemon::wal::ledger_state;
    let store = open_copy("wal-p2-ledger.wal");
    let s = store.session(1).unwrap();
    assert_eq!(s.tokens, vec![1, 2, 7, 8, 9]);
    assert!(s.open_permissions.is_empty());
    let led = &s.ledger;
    assert_eq!(led[&2].state, ledger_state::CANCELLED_WITH_LATE_EFFECT);
    assert_eq!(led[&2].late_effects, 1);
    assert!(led[&2].cancel_requested);
    assert_eq!(led[&3].state, ledger_state::EXPIRED);
    assert_eq!(led[&3].lease_deadline_unix_ms, Some(1));
    assert_eq!(led[&4].state, ledger_state::SUCCEEDED);
    assert!(s.open_tool_calls.is_empty());
    assert!(s.events.iter().any(|e| matches!(&e.body, EventBody::Block { kind: 1, .. })));
}

#[test]
#[ignore]
fn write_p2_ledger_fixture() {
    use superfluid_daemon::wal::tool_outcome;
    let path = fixtures().join("wal-p2-ledger.wal");
    assert!(!path.exists(), "fixtures are never regenerated");
    let mut store = SessionStore::open(&path).unwrap();
    let s = store.create(None, GenParams::default()).unwrap();
    store.append(s, None, vec![1, 2]).unwrap();
    let a = store.commit_tool_use(s, "slow".into(), "{}".into()).unwrap().event_id;
    let l = store.commit_tool_use(s, "leased".into(), "{}".into()).unwrap().event_id;
    let c = store.commit_tool_use(s, "fine".into(), "{}".into()).unwrap().event_id;
    let p = store.commit_permission_request(s, a, "allow?".into()).unwrap().event_id;
    store.commit_permission_response(s, p, true).unwrap();
    store.commit_tool_lease(s, l, 1).unwrap();
    store.commit_tool_cancel_request(s, a).unwrap();
    store.commit_tool_outcome(s, a, tool_outcome::CANCELLED, "killed".into()).unwrap();
    store.commit_tool_reconciliation(s, a, "late result: 3 bytes".into()).unwrap();
    store.expire_tool_call(s, l, "lease expired").unwrap();
    store.commit_tool_result(s, c, "ok".into(), vec![7]).unwrap();
    store
        .commit_block(s, 1, 1, r#"{"path":"x","diff":"-a\n+b"}"#.into(), vec![8, 9], 2)
        .unwrap();
}
