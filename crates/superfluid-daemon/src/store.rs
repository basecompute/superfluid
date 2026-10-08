use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use crate::bus::SessionBus;
pub use crate::store_id::{store_id_path, StoreId, StoreIdLock};
use crate::wal::{EventBody, GenParams, PurgeMode, RebaseEdit, Wal, WalRecord};
use crate::DaemonError;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LedgerEntry {
    pub name: String,
    pub arguments: String,
    pub state: u8,
    pub cancel_requested: bool,
    pub lease_deadline_unix_ms: Option<u64>,
    pub late_effects: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommittedEvent {
    pub event_id: u64,
    pub epoch: u64,
    pub ts_unix_ms: u64,
    pub body: EventBody,
}

#[derive(Debug)]
pub struct SessionState {
    pub id: u64,
    pub parent: Option<u64>,
    pub base: u64,
    pub params: GenParams,
    pub epoch: u64,
    pub tokens: Vec<u32>,
    pub events: Vec<CommittedEvent>,
    pub last_finish: u32,
    pub open_tool_calls: BTreeMap<u64, (String, String)>,
    pub ledger: BTreeMap<u64, LedgerEntry>,
    pub open_permissions: BTreeMap<u64, (u64, String)>,
    pub behavior_fp: Option<u64>,
    pub meta_version: u64,
    pub title: Option<String>,
    pub archived: bool,
    pub generation: u64,
    pub purged: bool,
    pub tombstone_ledger: BTreeMap<u64, String>,
    pub rerooted: bool,
    pub qos_class: u8,
    pub batch_invariant: bool,
    pub pin_deadline_unix_ms: u64,
}

impl SessionState {
    pub fn next_event_id(&self) -> u64 {
        self.events.len() as u64
    }
}

pub struct SessionStore {
    wal: Wal,
    store_id: StoreId,
    _held: Option<StoreIdLock>,
    sessions: BTreeMap<u64, SessionState>,
    next_session: u64,
    bus: Arc<SessionBus>,
}

impl SessionStore {
    pub fn ephemeral() -> SessionStore {
        SessionStore {
            wal: Wal::ephemeral(),
            store_id: StoreId::mint(),
            _held: None,
            sessions: BTreeMap::new(),
            next_session: 1,
            bus: SessionBus::new(),
        }
    }

    pub fn open(path: &Path) -> Result<SessionStore, DaemonError> {
        let mut sessions: BTreeMap<u64, SessionState> = BTreeMap::new();
        let id_path = store_id_path(path);
        // Held while the store lives; a store being dropped lets go a moment later.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        let id_lock = loop {
            match StoreId::try_lock(&id_path)? {
                Some(lock) => break Some(lock),
                None if std::time::Instant::now() < deadline => std::thread::sleep(std::time::Duration::from_millis(20)),
                None => break None,
            }
        };
        let Some(id_lock) = id_lock else {
            return Err(DaemonError::Io(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!(
                    "the session log {} is already open in another session store (another \
                     server on the same sessions directory, or the same model loaded twice); \
                     stop that one first",
                    path.display()
                ),
            )));
        };
        let fresh_wal = match std::fs::metadata(path) {
            Ok(m) => m.len() == 0,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
            Err(e) => return Err(e.into()),
        };
        let store_id = StoreId::open(&id_path, fresh_wal, &id_lock)?;
        let wal = Wal::open(path, |r| replay(&mut sessions, r))?;
        let mut next_session = 1;
        for id in sessions.keys() {
            next_session = next_session.max(id + 1);
        }
        let mut store = SessionStore {
            wal,
            store_id,
            _held: Some(id_lock),
            sessions,
            next_session,
            bus: SessionBus::new(),
        };
        let ids: Vec<u64> = store
            .sessions
            .values()
            .filter(|s| !s.purged)
            .map(|s| s.id)
            .collect();
        for id in ids {
            let new_epoch = store.sessions[&id].epoch + 1;
            let record = WalRecord {
                session: id,
                event_id: store.sessions[&id].next_event_id(),
                epoch: new_epoch,
                ts_unix_ms: store.stamp(),
                body: EventBody::EpochBump,
            };
            store.wal.append(&record)?;
            replay(&mut store.sessions, record);
        }
        Ok(store)
    }

    pub fn create(
        &mut self,
        parent: Option<u64>,
        params: GenParams,
    ) -> Result<u64, DaemonError> {
        if let Some(p) = parent {
            self.session(p)?;
        }
        let id = self.next_session;
        let record = WalRecord {
            session: id,
            event_id: 0,
            epoch: 1,
            ts_unix_ms: self.stamp(),
            body: EventBody::Created { parent, params },
        };
        self.wal.append(&record)?;
        self.next_session += 1;
        replay(&mut self.sessions, record);
        Ok(id)
    }

    pub fn fork(
        &mut self,
        parent: u64,
        fork_at: u64,
        params: Option<GenParams>,
    ) -> Result<u64, DaemonError> {
        let p = self.session(parent)?;
        let max = p.next_event_id();
        if fork_at == 0 || fork_at > max {
            return Err(DaemonError::ForkPoint {
                session: parent,
                fork_at,
                max,
            });
        }
        let params = params.unwrap_or(p.params);
        let id = self.next_session;
        let record = WalRecord {
            session: id,
            event_id: fork_at,
            epoch: 1,
            ts_unix_ms: self.stamp(),
            body: EventBody::Forked {
                parent,
                fork_at,
                params,
            },
        };
        self.wal.append(&record)?;
        self.next_session += 1;
        replay(&mut self.sessions, record);
        Ok(id)
    }

    pub fn append(
        &mut self,
        session: u64,
        text: Option<String>,
        span: Vec<u32>,
    ) -> Result<CommittedEvent, DaemonError> {
        self.commit(session, EventBody::Appended { text, span })
    }

    pub fn append_message(
        &mut self,
        session: u64,
        role: u32,
        text: String,
        span: Vec<u32>,
    ) -> Result<CommittedEvent, DaemonError> {
        self.commit(session, EventBody::Message { role, text, span })
    }

    pub fn commit_generation_prompt(
        &mut self,
        session: u64,
        span: Vec<u32>,
    ) -> Result<CommittedEvent, DaemonError> {
        self.commit(session, EventBody::GenerationPrompt { span })
    }

    pub fn commit_tool_use(
        &mut self,
        session: u64,
        name: String,
        arguments: String,
    ) -> Result<CommittedEvent, DaemonError> {
        self.commit(session, EventBody::ToolUse { name, arguments })
    }

    pub fn commit_tool_result(
        &mut self,
        session: u64,
        call_id: u64,
        content: String,
        span: Vec<u32>,
    ) -> Result<CommittedEvent, DaemonError> {
        let s = self.session(session)?;
        if !s.open_tool_calls.contains_key(&call_id) {
            return Err(DaemonError::UnknownToolCall(call_id));
        }
        self.commit(
            session,
            EventBody::ToolResult {
                call_id,
                content,
                span,
            },
        )
    }

    pub fn commit_generated(
        &mut self,
        session: u64,
        span: Vec<u32>,
        text: String,
        channel: u32,
        finish: u32,
    ) -> Result<CommittedEvent, DaemonError> {
        self.commit(
            session,
            EventBody::Generated {
                span,
                text,
                channel,
                finish,
            },
        )
    }

    pub fn commit_generation_fingerprint(
        &mut self,
        session: u64,
        digest: u64,
    ) -> Result<Option<CommittedEvent>, DaemonError> {
        if self.session(session)?.behavior_fp == Some(digest) {
            return Ok(None);
        }
        self.commit(session, EventBody::GenerationFingerprint { digest })
            .map(Some)
    }

    pub fn commit_tool_parse_failure(
        &mut self,
        session: u64,
        raw: String,
    ) -> Result<CommittedEvent, DaemonError> {
        self.commit(session, EventBody::ToolParseFailure { raw })
    }

    fn commit(&mut self, session: u64, body: EventBody) -> Result<CommittedEvent, DaemonError> {
        let s = self.session(session)?;
        let record = WalRecord {
            session,
            event_id: s.next_event_id(),
            epoch: s.epoch,
            ts_unix_ms: self.stamp(),
            body,
        };
        self.wal.append(&record)?;
        replay(&mut self.sessions, record.clone());
        let ev = CommittedEvent {
            event_id: record.event_id,
            epoch: record.epoch,
            ts_unix_ms: record.ts_unix_ms,
            body: record.body,
        };
        self.bus.publish_committed(session, &ev);
        Ok(ev)
    }

    pub fn store_id(&self) -> StoreId {
        self.store_id
    }

    pub fn wal_version(&self) -> u8 {
        self.wal.version()
    }

    fn stamp(&self) -> u64 {
        if self.wal.version() == 2 {
            crate::wal::now_unix_ms()
        } else {
            0
        }
    }

    pub fn bus(&self) -> Arc<SessionBus> {
        Arc::clone(&self.bus)
    }

    pub fn session(&self, id: u64) -> Result<&SessionState, DaemonError> {
        let s = self.sessions.get(&id).ok_or(DaemonError::UnknownSession(id))?;
        if s.purged {
            return Err(DaemonError::Purged(id));
        }
        Ok(s)
    }

    pub fn tombstone(&self, id: u64) -> Option<&SessionState> {
        self.sessions.get(&id).filter(|s| s.purged)
    }

    pub fn read(&self, id: u64, cursor: u64) -> Result<&[CommittedEvent], DaemonError> {
        let s = self.session(id)?;
        let start = (cursor as usize).min(s.events.len());
        Ok(&s.events[start..])
    }

    pub fn drop_session(&mut self, id: u64) -> Result<(), DaemonError> {
        self.sessions
            .remove(&id)
            .map(|_| ())
            .ok_or(DaemonError::UnknownSession(id))
    }

    pub fn session_ids(&self) -> Vec<u64> {
        self.sessions
            .values()
            .filter(|s| !s.purged)
            .map(|s| s.id)
            .collect()
    }

    pub fn set_meta(
        &mut self,
        session: u64,
        expected_version: u64,
        title: Option<String>,
        archived: Option<bool>,
    ) -> Result<u64, DaemonError> {
        let s = self.session(session)?;
        if s.meta_version != expected_version {
            return Err(DaemonError::MetaConflict {
                session,
                expected: expected_version,
                actual: s.meta_version,
            });
        }
        let version = expected_version + 1;
        self.commit(
            session,
            EventBody::MetaUpdated {
                version,
                title,
                archived,
            },
        )?;
        Ok(version)
    }

    pub fn set_pin(&mut self, session: u64, deadline_unix_ms: u64) -> Result<CommittedEvent, DaemonError> {
        self.session(session)?;
        self.commit(session, EventBody::Pinned { deadline_unix_ms })
    }

    pub fn set_qos(
        &mut self,
        session: u64,
        class: u8,
        batch_invariant: bool,
    ) -> Result<CommittedEvent, DaemonError> {
        if class > crate::qos::BACKGROUND_AGENT {
            return Err(DaemonError::Protocol("unknown QoS class"));
        }
        self.commit(
            session,
            EventBody::QosSet {
                class,
                batch_invariant,
            },
        )
    }

    pub fn rebase_branch(
        &mut self,
        parent: u64,
        fork_at: u64,
        params: Option<GenParams>,
        edits: Vec<RebaseEdit>,
    ) -> Result<u64, DaemonError> {
        let p = self.session(parent)?;
        let max = p.next_event_id();
        if fork_at == 0 || fork_at > max {
            return Err(DaemonError::ForkPoint {
                session: parent,
                fork_at,
                max,
            });
        }
        let params = params.unwrap_or(p.params);
        let id = self.next_session;
        let record = WalRecord {
            session: id,
            event_id: fork_at,
            epoch: 1,
            ts_unix_ms: self.stamp(),
            body: EventBody::Rebased {
                parent,
                fork_at,
                params,
                edits,
            },
        };
        self.wal.append(&record)?;
        self.next_session += 1;
        replay(&mut self.sessions, record);
        Ok(id)
    }

    pub fn commit_block(
        &mut self,
        session: u64,
        role: u32,
        kind: u32,
        payload: String,
        span: Vec<u32>,
        visibility_version: u32,
    ) -> Result<CommittedEvent, DaemonError> {
        self.commit(
            session,
            EventBody::Block {
                role,
                kind,
                payload,
                span,
                visibility_version,
            },
        )
    }

    pub fn commit_permission_request(
        &mut self,
        session: u64,
        call_id: u64,
        text: String,
    ) -> Result<CommittedEvent, DaemonError> {
        let s = self.session(session)?;
        if call_id != 0 && !s.open_tool_calls.contains_key(&call_id) {
            return Err(DaemonError::UnknownToolCall(call_id));
        }
        self.commit(session, EventBody::PermissionRequest { call_id, text })
    }

    pub fn commit_permission_response(
        &mut self,
        session: u64,
        request_id: u64,
        granted: bool,
    ) -> Result<CommittedEvent, DaemonError> {
        let s = self.session(session)?;
        if !s.open_permissions.contains_key(&request_id) {
            return Err(DaemonError::UnknownPermission(request_id));
        }
        self.commit(
            session,
            EventBody::PermissionResponse {
                request_id,
                granted,
            },
        )
    }

    pub fn commit_tool_cancel_request(
        &mut self,
        session: u64,
        call_id: u64,
    ) -> Result<CommittedEvent, DaemonError> {
        let s = self.session(session)?;
        if !s.open_tool_calls.contains_key(&call_id) {
            return Err(DaemonError::UnknownToolCall(call_id));
        }
        self.commit(session, EventBody::ToolCancelRequested { call_id })
    }

    pub fn commit_tool_outcome(
        &mut self,
        session: u64,
        call_id: u64,
        outcome: u8,
        note: String,
    ) -> Result<CommittedEvent, DaemonError> {
        let s = self.session(session)?;
        if !s.open_tool_calls.contains_key(&call_id) {
            return Err(DaemonError::UnknownToolCall(call_id));
        }
        if outcome != crate::wal::tool_outcome::FAILED && outcome != crate::wal::tool_outcome::CANCELLED {
            return Err(DaemonError::Protocol("unknown tool outcome"));
        }
        self.commit(
            session,
            EventBody::ToolOutcome {
                call_id,
                outcome,
                note,
            },
        )
    }

    pub fn commit_tool_reconciliation(
        &mut self,
        session: u64,
        call_id: u64,
        note: String,
    ) -> Result<CommittedEvent, DaemonError> {
        let s = self.session(session)?;
        if !s.ledger.contains_key(&call_id) {
            return Err(DaemonError::UnknownToolCall(call_id));
        }
        self.commit(session, EventBody::ToolReconciliation { call_id, note })
    }

    pub fn commit_tool_lease(
        &mut self,
        session: u64,
        call_id: u64,
        deadline_unix_ms: u64,
    ) -> Result<CommittedEvent, DaemonError> {
        let s = self.session(session)?;
        if !s.open_tool_calls.contains_key(&call_id) {
            return Err(DaemonError::UnknownToolCall(call_id));
        }
        self.commit(
            session,
            EventBody::ToolLease {
                call_id,
                deadline_unix_ms,
            },
        )
    }

    pub fn expire_tool_call(
        &mut self,
        session: u64,
        call_id: u64,
        reason: &str,
    ) -> Result<CommittedEvent, DaemonError> {
        let s = self.session(session)?;
        if !s.open_tool_calls.contains_key(&call_id) {
            return Err(DaemonError::UnknownToolCall(call_id));
        }
        self.commit(
            session,
            EventBody::ToolExpired {
                call_id,
                reason: reason.to_string(),
            },
        )
    }

    pub fn children_of(&self, session: u64) -> Vec<u64> {
        self.sessions
            .values()
            .filter(|c| c.parent == Some(session) && !c.purged)
            .map(|c| c.id)
            .collect()
    }

    pub fn descendants_of(&self, session: u64) -> Vec<u64> {
        let mut out = Vec::new();
        let mut stack = self.children_of(session);
        while let Some(c) = stack.pop() {
            out.push(c);
            stack.extend(self.children_of(c));
        }
        out
    }

    pub fn purge(
        &mut self,
        session: u64,
        expected_generation: u64,
        mode: PurgeMode,
    ) -> Result<Vec<u64>, DaemonError> {
        let s = self.session(session)?;
        if s.generation != expected_generation {
            return Err(DaemonError::GenerationConflict {
                session,
                expected: expected_generation,
                actual: s.generation,
            });
        }
        if !s.open_tool_calls.is_empty() {
            return Err(DaemonError::Protocol("purge requires a settled tool ledger"));
        }
        let mut purged = Vec::new();
        match mode {
            PurgeMode::Reroot => {
                for c in self.children_of(session) {
                    let child = self.session(c)?;
                    let prefix: Vec<(u64, EventBody)> = child.events[..child.base as usize]
                        .iter()
                        .map(|e| (e.event_id, e.body.clone()))
                        .collect();
                    self.commit(
                        c,
                        EventBody::Rerooted {
                            purged_parent: session,
                            prefix,
                        },
                    )?;
                }
            }
            PurgeMode::Cascade => {
                let mut desc = self.descendants_of(session);
                desc.reverse();
                for d in desc {
                    let g = self.session(d)?.generation;
                    self.commit(
                        d,
                        EventBody::Purged {
                            generation: g + 1,
                            descendants: PurgeMode::Cascade,
                        },
                    )?;
                    purged.push(d);
                }
            }
        }
        self.commit(
            session,
            EventBody::Purged {
                generation: expected_generation + 1,
                descendants: mode,
            },
        )?;
        purged.push(session);
        Ok(purged)
    }
}

pub fn ledger_state_name(state: u8) -> &'static str {
    use crate::wal::ledger_state::*;
    match state {
        OPEN => "open",
        CANCEL_REQUESTED => "cancel_requested",
        SUCCEEDED => "closed",
        FAILED => "failed",
        CANCELLED => "cancelled",
        EXPIRED => "expired",
        CANCELLED_WITH_LATE_EFFECT => "cancelled_with_late_effect",
        _ => "unknown",
    }
}

fn replay(sessions: &mut BTreeMap<u64, SessionState>, r: WalRecord) {
    match &r.body {
        EventBody::Created { parent, params } => {
            sessions.insert(
                r.session,
                SessionState {
                    id: r.session,
                    parent: *parent,
                    base: 0,
                    params: *params,
                    epoch: r.epoch,
                    tokens: Vec::new(),
                    events: Vec::new(),
                    last_finish: 0,
                    open_tool_calls: BTreeMap::new(),
                    ledger: BTreeMap::new(),
                    open_permissions: BTreeMap::new(),
                    behavior_fp: None,
                    meta_version: 0,
                    title: None,
                    archived: false,
                    generation: 0,
                    purged: false,
                    tombstone_ledger: BTreeMap::new(),
                    rerooted: false,
                    qos_class: crate::qos::FOREGROUND_AGENT,
                    batch_invariant: false,
                    pin_deadline_unix_ms: 0,
                },
            );
        }
        EventBody::Forked {
            parent,
            fork_at,
            params,
        }
        | EventBody::Rebased {
            parent,
            fork_at,
            params,
            ..
        } => {
            let inherited: Vec<CommittedEvent> = sessions
                .get(parent)
                .map(|p| p.events[..(*fork_at as usize).min(p.events.len())].to_vec())
                .unwrap_or_default();
            let (qos_class, batch_invariant) = sessions
                .get(parent)
                .map(|p| (p.qos_class, p.batch_invariant))
                .unwrap_or((crate::qos::FOREGROUND_AGENT, false));
            let mut tokens = Vec::new();
            let mut last_finish = 0;
            let mut behavior_fp = None;
            for e in &inherited {
                match &e.body {
                    EventBody::Appended { span, .. }
                    | EventBody::Message { span, .. }
                    | EventBody::GenerationPrompt { span }
                    | EventBody::ToolResult { span, .. } => tokens.extend_from_slice(span),
                    EventBody::Generated { span, finish, .. } => {
                        tokens.extend_from_slice(span);
                        last_finish = *finish;
                    }
                    EventBody::GenerationFingerprint { digest } => behavior_fp = Some(*digest),
                    _ => {}
                }
            }
            sessions.insert(
                r.session,
                SessionState {
                    id: r.session,
                    parent: Some(*parent),
                    base: *fork_at,
                    params: *params,
                    epoch: r.epoch,
                    tokens,
                    events: inherited,
                    last_finish,
                    open_tool_calls: BTreeMap::new(),
                    ledger: BTreeMap::new(),
                    open_permissions: BTreeMap::new(),
                    behavior_fp,
                    meta_version: 0,
                    title: None,
                    archived: false,
                    generation: 0,
                    purged: false,
                    tombstone_ledger: BTreeMap::new(),
                    rerooted: false,
                    qos_class,
                    batch_invariant,
                    pin_deadline_unix_ms: 0,
                },
            );
        }
        EventBody::QosSet {
            class,
            batch_invariant,
        } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.qos_class = *class;
                s.batch_invariant = *batch_invariant;
            }
        }
        EventBody::MetaUpdated {
            version,
            title,
            archived,
        } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.meta_version = *version;
                if let Some(t) = title {
                    s.title = Some(t.clone());
                }
                if let Some(a) = archived {
                    s.archived = *a;
                }
            }
        }
        EventBody::ToolExpired { call_id, .. } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.open_tool_calls.remove(call_id);
                if let Some(e) = s.ledger.get_mut(call_id) {
                    e.state = crate::wal::ledger_state::EXPIRED;
                }
            }
        }
        EventBody::Rerooted { prefix, .. } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                if (s.events.len() as u64) < s.base {
                    let mut evs: Vec<CommittedEvent> = prefix
                        .iter()
                        .map(|(id, body)| CommittedEvent {
                            event_id: *id,
                            epoch: 0,
                            ts_unix_ms: 0,
                            body: body.clone(),
                        })
                        .collect();
                    evs.extend(std::mem::take(&mut s.events));
                    let mut tokens = Vec::new();
                    for e in &evs {
                        match &e.body {
                            EventBody::Appended { span, .. }
                            | EventBody::Message { span, .. }
                            | EventBody::GenerationPrompt { span }
                            | EventBody::ToolResult { span, .. }
                            | EventBody::Generated { span, .. } => tokens.extend_from_slice(span),
                            _ => {}
                        }
                    }
                    s.events = evs;
                    s.tokens = tokens;
                }
            }
        }
        EventBody::Purged {
            generation,
            descendants,
        } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.generation = *generation;
                s.purged = true;
                let mut ledger = BTreeMap::new();
                for (id, e) in &s.ledger {
                    ledger.insert(*id, ledger_state_name(e.state).to_string());
                }
                s.tombstone_ledger = ledger;
                s.events.clear();
                s.tokens.clear();
                s.open_tool_calls.clear();
                s.title = None;
            }
            if *descendants == PurgeMode::Reroot {
                for c in sessions.values_mut() {
                    if c.parent == Some(r.session) && !c.purged {
                        c.rerooted = true;
                    }
                }
            }
            return;
        }
        EventBody::Appended { span, .. }
        | EventBody::Message { span, .. }
        | EventBody::GenerationPrompt { span }
        | EventBody::ToolResult { span, .. }
        | EventBody::Block { span, .. }
        | EventBody::Generated { span, .. } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.tokens.extend_from_slice(span);
                s.epoch = s.epoch.max(r.epoch);
                if let EventBody::Generated { finish, .. } = &r.body {
                    s.last_finish = *finish;
                }
                if let EventBody::ToolResult { call_id, .. } = &r.body {
                    s.open_tool_calls.remove(call_id);
                    if let Some(e) = s.ledger.get_mut(call_id) {
                        e.state = crate::wal::ledger_state::SUCCEEDED;
                    }
                }
            }
        }
        EventBody::ToolUse { name, arguments } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.epoch = s.epoch.max(r.epoch);
                s.open_tool_calls
                    .insert(r.event_id, (name.clone(), arguments.clone()));
                s.ledger.insert(
                    r.event_id,
                    LedgerEntry {
                        name: name.clone(),
                        arguments: arguments.clone(),
                        state: crate::wal::ledger_state::OPEN,
                        cancel_requested: false,
                        lease_deadline_unix_ms: None,
                        late_effects: 0,
                    },
                );
            }
        }
        EventBody::PermissionRequest { call_id, text } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.open_permissions
                    .insert(r.event_id, (*call_id, text.clone()));
            }
        }
        EventBody::PermissionResponse { request_id, .. } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.open_permissions.remove(request_id);
            }
        }
        EventBody::ToolCancelRequested { call_id } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                if let Some(e) = s.ledger.get_mut(call_id) {
                    e.cancel_requested = true;
                    if e.state == crate::wal::ledger_state::OPEN {
                        e.state = crate::wal::ledger_state::CANCEL_REQUESTED;
                    }
                }
            }
        }
        EventBody::ToolOutcome { call_id, outcome, .. } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.open_tool_calls.remove(call_id);
                if let Some(e) = s.ledger.get_mut(call_id) {
                    e.state = if *outcome == crate::wal::tool_outcome::CANCELLED {
                        crate::wal::ledger_state::CANCELLED
                    } else {
                        crate::wal::ledger_state::FAILED
                    };
                }
            }
        }
        EventBody::ToolReconciliation { call_id, .. } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                if let Some(e) = s.ledger.get_mut(call_id) {
                    e.late_effects += 1;
                    if e.state == crate::wal::ledger_state::CANCELLED
                        || e.state == crate::wal::ledger_state::CANCELLED_WITH_LATE_EFFECT
                    {
                        e.state = crate::wal::ledger_state::CANCELLED_WITH_LATE_EFFECT;
                    }
                }
            }
        }
        EventBody::ToolLease {
            call_id,
            deadline_unix_ms,
        } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                if let Some(e) = s.ledger.get_mut(call_id) {
                    e.lease_deadline_unix_ms = Some(*deadline_unix_ms);
                }
            }
        }
        EventBody::Pinned { deadline_unix_ms } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.pin_deadline_unix_ms = *deadline_unix_ms;
            }
        }
        EventBody::EpochBump => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.epoch = s.epoch.max(r.epoch);
            }
        }
        EventBody::GenerationFingerprint { digest } => {
            if let Some(s) = sessions.get_mut(&r.session) {
                s.behavior_fp = Some(*digest);
            }
        }
        EventBody::ToolParseFailure { .. } => {}
    }
    if let Some(s) = sessions.get_mut(&r.session) {
        if s.purged {
            return;
        }
        s.events.push(CommittedEvent {
            event_id: r.event_id,
            epoch: r.epoch,
            ts_unix_ms: r.ts_unix_ms,
            body: r.body,
        });
    }
}
