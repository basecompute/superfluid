//! The session store's identity.

use std::path::PathBuf;

use superfluid_daemon::store::{store_id_path, StoreId};
use superfluid_daemon::{export, GenParams, Inspection, SessionStore, Tier};

fn test_dir(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("superfluid-storeid-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn distinct_per_store_and_stable_across_reopen() {
    let a_wal = test_dir("a").join("wal.log");
    let b_wal = test_dir("b").join("wal.log");
    let (a, b) = {
        let mut a = SessionStore::open(&a_wal).unwrap();
        let mut b = SessionStore::open(&b_wal).unwrap();
        assert_eq!(a.create(None, GenParams::default()).unwrap(), 1);
        assert_eq!(b.create(None, GenParams::default()).unwrap(), 1);
        (a.store_id(), b.store_id())
    };
    assert_ne!(a, b, "two stores, two ids");
    let (na, nb) = (export::store_trace_ns(&a), export::store_trace_ns(&b));
    assert_ne!(
        export::trace_id_bytes_ns(na, 1),
        export::trace_id_bytes_ns(nb, 1)
    );
    assert_ne!(
        export::span_id_bytes_ns(na, 1, export::ROOT),
        export::span_id_bytes_ns(nb, 1, export::ROOT)
    );
    let text = std::fs::read_to_string(store_id_path(&a_wal)).unwrap();
    assert_eq!(text, a.to_file_line());
    assert!(text.starts_with(&a.to_hex()));
    let again = SessionStore::open(&a_wal).unwrap().store_id();
    assert_eq!(again, a);
    assert_eq!(
        export::trace_id_bytes_ns(export::store_trace_ns(&again), 1),
        export::trace_id_bytes_ns(na, 1)
    );
    let leftovers: Vec<_> = std::fs::read_dir(a_wal.parent().unwrap())
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn old_store_gets_an_id_minted_and_persisted() {
    let dir = test_dir("old");
    let wal = dir.join("wal-p2-forked.wal");
    std::fs::copy(fixtures().join("wal-p2-forked.wal"), &wal).unwrap();
    assert!(!store_id_path(&wal).exists(), "predates store ids");
    let first = {
        let store = SessionStore::open(&wal).unwrap();
        assert_eq!(store.wal_version(), 1, "an old log stays old");
        assert!(store.session(1).is_ok(), "and still replays");
        store.store_id()
    };
    assert!(store_id_path(&wal).exists(), "minted and persisted");
    assert_eq!(SessionStore::open(&wal).unwrap().store_id(), first);
}

#[test]
fn concurrent_mints_agree() {
    let path = test_dir("race").join("wal.log.store-id");
    let ids: Vec<StoreId> = (0..8)
        .map(|_| {
            let p = path.clone();
            std::thread::spawn(move || StoreId::load_or_mint(&p).unwrap())
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect();
    assert!(ids.iter().all(|i| *i == ids[0]), "{ids:?}");
    assert_eq!(StoreId::load_or_mint(&path).unwrap(), ids[0]);
}

#[test]
fn corrupt_id_is_refused() {
    let wal = test_dir("flip").join("wal.log");
    {
        let mut s = SessionStore::open(&wal).unwrap();
        s.create(None, GenParams::default()).unwrap();
    }
    let good = std::fs::read_to_string(store_id_path(&wal)).unwrap();
    let flipped = match good.strip_prefix('0') {
        Some(rest) => format!("2{rest}"),
        None => format!("0{}", &good[1..]),
    };
    std::fs::write(store_id_path(&wal), &flipped).unwrap();
    let err = SessionStore::open(&wal)
        .err()
        .expect("a flipped digit is refused");
    assert!(
        err.to_string().contains("store's id file is corrupt"),
        "{err}"
    );
    assert_eq!(
        std::fs::read_to_string(store_id_path(&wal)).unwrap(),
        flipped
    );

    let wal = test_dir("corrupt").join("wal.log");
    std::fs::write(store_id_path(&wal), "not-an-id\n").unwrap();
    let err = SessionStore::open(&wal).err().expect("refused");
    assert!(
        err.to_string().contains("store's id file is corrupt"),
        "{err}"
    );
    assert_eq!(
        std::fs::read_to_string(store_id_path(&wal)).unwrap(),
        "not-an-id\n"
    );
}

#[test]
fn reset_store_does_not_reuse_a_stale_id() {
    for truncate in [false, true] {
        let wal = test_dir("reset").join("wal.log");
        let old = {
            let mut s = SessionStore::open(&wal).unwrap();
            s.create(None, GenParams::default()).unwrap();
            s.store_id()
        };
        if truncate {
            std::fs::File::create(&wal).unwrap();
        } else {
            std::fs::remove_file(&wal).unwrap();
        }
        assert!(store_id_path(&wal).exists());
        let fresh = {
            let mut s = SessionStore::open(&wal).unwrap();
            assert_eq!(
                s.create(None, GenParams::default()).unwrap(),
                1,
                "numbering restarted"
            );
            s.store_id()
        };
        assert_ne!(fresh, old, "a new incarnation, a new namespace");
        assert_eq!(
            std::fs::read_to_string(store_id_path(&wal)).unwrap(),
            fresh.to_file_line()
        );
        assert_eq!(SessionStore::open(&wal).unwrap().store_id(), fresh);
    }
}

#[test]
fn crash_between_id_replacement_and_wal_creation_redecides() {
    let wal = test_dir("crashwin").join("wal.log");
    let old = {
        let mut s = SessionStore::open(&wal).unwrap();
        s.create(None, GenParams::default()).unwrap();
        s.store_id()
    };
    std::fs::remove_file(&wal).unwrap();
    let replaced = {
        let lock = StoreId::lock(&store_id_path(&wal)).unwrap();
        StoreId::open(&store_id_path(&wal), true, &lock).unwrap()
    };
    assert_ne!(replaced, old);
    assert!(!wal.exists());
    let after = SessionStore::open(&wal).unwrap().store_id();
    assert_ne!(after, old, "the stale id never comes back");
    assert_eq!(
        SessionStore::open(&wal).unwrap().store_id(),
        after,
        "then stable"
    );
}

#[test]
fn empty_but_existing_store_keeps_its_id() {
    let wal = test_dir("empty").join("wal.log");
    let first = SessionStore::open(&wal).unwrap().store_id();
    assert_eq!(SessionStore::open(&wal).unwrap().store_id(), first);
}

#[test]
fn concurrent_opens_of_a_reset_store_agree() {
    let wal = test_dir("replace-race").join("wal.log");
    let stale = {
        let mut s = SessionStore::open(&wal).unwrap();
        s.create(None, GenParams::default()).unwrap();
        s.store_id()
    };
    std::fs::remove_file(&wal).unwrap();
    let ids: Vec<StoreId> = (0..8)
        .map(|_| {
            let w = wal.clone();
            std::thread::spawn(move || loop {
                match SessionStore::open(&w) {
                    Ok(s) => break s.store_id(),
                    Err(superfluid_daemon::DaemonError::Io(e))
                        if e.kind() == std::io::ErrorKind::WouldBlock =>
                    {
                        std::thread::yield_now()
                    }
                    Err(e) => panic!("{e}"),
                }
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect();
    assert!(ids.iter().all(|i| *i == ids[0]), "{ids:?}");
    assert_ne!(ids[0], stale, "the stale id was replaced");
    assert_eq!(
        StoreId::load_or_mint(&store_id_path(&wal)).unwrap(),
        ids[0],
        "and persisted"
    );
}

#[test]
fn a_live_store_holds_its_log() {
    let wal = test_dir("held").join("wal.log");
    let mut first = SessionStore::open(&wal).unwrap();
    first.create(None, GenParams::default()).unwrap();
    let err = SessionStore::open(&wal)
        .err()
        .expect("a second store on a live log is refused");
    assert!(
        matches!(&err, superfluid_daemon::DaemonError::Io(e) if e.kind() == std::io::ErrorKind::WouldBlock),
        "{err:?}"
    );
    assert!(err.to_string().contains("already open"), "{err}");
    first.append(1, None, vec![1]).unwrap();
    drop(first);
    let reopened = SessionStore::open(&wal).unwrap();
    assert_eq!(reopened.session(1).unwrap().tokens, vec![1]);
}

#[test]
fn a_store_let_go_a_moment_later_is_waited_for() {
    let wal = test_dir("handover").join("wal.log");
    let mut first = SessionStore::open(&wal).unwrap();
    first.create(None, GenParams::default()).unwrap();
    first.append(1, None, vec![7]).unwrap();
    let owner = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        drop(first);
    });
    let reopened = SessionStore::open(&wal).expect("the log is opened once its owner lets go");
    assert_eq!(reopened.session(1).unwrap().tokens, vec![7]);
    owner.join().unwrap();
}

#[cfg(unix)]
#[test]
fn failed_publication_leaves_no_temp_file() {
    let dir = test_dir("failpub");
    let path = dir.join("wal.log.store-id");
    std::os::unix::fs::symlink(dir.join("nowhere"), &path).unwrap();
    let err = StoreId::load_or_mint(&path).expect_err("publication fails");
    assert!(err.to_string().contains("vanished"), "{err}");
    let leftovers: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn hex_round_trip() {
    let id = StoreId::mint();
    assert_eq!(StoreId::from_hex(&id.to_hex()), Some(id));
    assert_eq!(StoreId::from_hex("zz"), None);
    assert_eq!(StoreId::from_hex(&"0".repeat(31)), None);
    assert_eq!(StoreId::from_file_line(&id.to_file_line()), Some(id));
    assert_eq!(StoreId::from_file_line(&id.to_hex()), None, "no checksum");
}

fn sample(store_id: Option<StoreId>) -> Inspection {
    let store = {
        let wal = test_dir("insp").join("wal.log");
        let mut s = SessionStore::open(&wal).unwrap();
        s.create(None, GenParams::default()).unwrap();
        s
    };
    let st = store.session(1).unwrap();
    Inspection {
        summary: superfluid_daemon::SessionSummary {
            id: 1,
            parent: None,
            fork_at: 0,
            title: None,
            archived: false,
            meta_version: 0,
            generation: 0,
            events: st.events.len() as u64,
            tokens: 0,
            open_tool_calls: 0,
        },
        params: GenParams::default(),
        epoch: st.epoch,
        last_finish: 0,
        qos_class: 0,
        batch_invariant: false,
        pin_deadline_unix_ms: 0,
        behavior_fingerprint: None,
        open_tool_calls: Vec::new(),
        tier: Tier::Cold,
        tokens: Vec::new(),
        events: st.events.iter().cloned().map(Into::into).collect(),
        wal_version: 2,
        store_id,
    }
}

#[test]
fn inspection_store_id_is_wire_compatible() {
    let id = StoreId([7; 16]);
    let new = sample(Some(id));
    let bytes = postcard::to_stdvec(&new).unwrap();
    assert_eq!(postcard::from_bytes::<Inspection>(&bytes).unwrap(), new);
    let old_bytes = &bytes[..bytes.len() - 17];
    let decoded: Inspection = postcard::from_bytes(old_bytes).expect("old reply decodes");
    assert_eq!(decoded.store_id, None);
    assert_eq!(decoded.events, new.events);
    assert_eq!(decoded.wal_version, 2);
    let short = &bytes[..bytes.len() - 8];
    assert!(
        postcard::from_bytes::<Inspection>(short).is_err(),
        "short id"
    );
    let mut bad_tag = bytes[..bytes.len() - 17].to_vec();
    bad_tag.push(7);
    assert!(
        postcard::from_bytes::<Inspection>(&bad_tag).is_err(),
        "bad tag"
    );
    let none = sample(None);
    let bytes = postcard::to_stdvec(&none).unwrap();
    assert_eq!(
        postcard::from_bytes::<Inspection>(&bytes).unwrap().store_id,
        None
    );
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct OldInspection {
    summary: superfluid_daemon::SessionSummary,
    params: GenParams,
    epoch: u64,
    last_finish: u32,
    qos_class: u8,
    batch_invariant: bool,
    pin_deadline_unix_ms: u64,
    behavior_fingerprint: Option<u64>,
    open_tool_calls: Vec<(u64, String, String)>,
    tier: Tier,
    tokens: Vec<u32>,
    events: Vec<superfluid_daemon::api::EventMsg>,
    wal_version: u8,
}

#[test]
fn older_client_ignores_the_appended_field() {
    let new = sample(Some(StoreId([9; 16])));
    let bytes = postcard::to_stdvec(&new).unwrap();
    let old: OldInspection = postcard::from_bytes(&bytes).expect("old client decodes");
    assert_eq!(old.wal_version, 2);
    assert_eq!(old.events, new.events);
}
