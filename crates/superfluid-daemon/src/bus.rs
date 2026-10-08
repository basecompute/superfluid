//! The session event bus.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

use crate::store::CommittedEvent;

const SUB_QUEUE: usize = 1024;

#[derive(Debug, Clone)]
pub enum BusMsg {
    Committed(CommittedEvent),
    Provisional { channel: u32, text: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    Lagged,
    PublisherClosed,
    Purged,
}

struct Subscriber {
    id: u64,
    tx: SyncSender<BusMsg>,
    provisional: bool,
    lagged: Arc<AtomicBool>,
    purged: Arc<AtomicBool>,
}

pub struct Subscription {
    pub session: u64,
    pub id: u64,
    pub rx: Receiver<BusMsg>,
    lagged: Arc<AtomicBool>,
    purged: Arc<AtomicBool>,
}

impl Subscription {
    pub fn take_receiver(&mut self) -> Receiver<BusMsg> {
        let (_tx, dummy) = std::sync::mpsc::channel();
        std::mem::replace(&mut self.rx, dummy)
    }

    pub fn end_reason(&self) -> EndReason {
        if self.purged.load(Ordering::Acquire) {
            EndReason::Purged
        } else if self.lagged.load(Ordering::Acquire) {
            EndReason::Lagged
        } else {
            EndReason::PublisherClosed
        }
    }
}

#[derive(Default)]
pub struct SessionBus {
    inner: Mutex<BusInner>,
}

#[derive(Default)]
struct BusInner {
    subs: HashMap<u64, Vec<Subscriber>>,
    next_id: u64,
}

impl SessionBus {
    pub fn new() -> Arc<SessionBus> {
        Arc::new(SessionBus::default())
    }

    pub fn subscribe(&self, session: u64, provisional: bool) -> Subscription {
        let (tx, rx) = std::sync::mpsc::sync_channel(SUB_QUEUE);
        let lagged = Arc::new(AtomicBool::new(false));
        let purged = Arc::new(AtomicBool::new(false));
        let mut inner = self.inner.lock().expect("bus");
        inner.next_id += 1;
        let id = inner.next_id;
        inner.subs.entry(session).or_default().push(Subscriber {
            id,
            tx,
            provisional,
            lagged: Arc::clone(&lagged),
            purged: Arc::clone(&purged),
        });
        Subscription {
            session,
            id,
            rx,
            lagged,
            purged,
        }
    }

    pub fn end_session(&self, session: u64, reason: EndReason) {
        let mut inner = self.inner.lock().expect("bus");
        if let Some(subs) = inner.subs.remove(&session) {
            for s in subs {
                match reason {
                    EndReason::Purged => s.purged.store(true, Ordering::Release),
                    EndReason::Lagged => s.lagged.store(true, Ordering::Release),
                    EndReason::PublisherClosed => {}
                }
            }
        }
    }

    pub fn unsubscribe(&self, session: u64, id: u64) {
        let mut inner = self.inner.lock().expect("bus");
        if let Some(v) = inner.subs.get_mut(&session) {
            v.retain(|s| s.id != id);
            if v.is_empty() {
                inner.subs.remove(&session);
            }
        }
    }

    pub fn has_provisional_subscriber(&self, session: u64) -> bool {
        let inner = self.inner.lock().expect("bus");
        inner
            .subs
            .get(&session)
            .is_some_and(|v| v.iter().any(|s| s.provisional))
    }

    pub fn publish_committed(&self, session: u64, ev: &CommittedEvent) {
        self.publish(session, |_| Some(BusMsg::Committed(ev.clone())));
    }

    pub fn publish_provisional(&self, session: u64, channel: u32, text: &str) {
        self.publish(session, |s| {
            s.provisional.then(|| BusMsg::Provisional {
                channel,
                text: text.to_string(),
            })
        });
    }

    fn publish(&self, session: u64, msg_for: impl Fn(&Subscriber) -> Option<BusMsg>) {
        let mut inner = self.inner.lock().expect("bus");
        let Some(subs) = inner.subs.get_mut(&session) else {
            return;
        };
        subs.retain(|s| {
            let Some(msg) = msg_for(s) else {
                return true;
            };
            match s.tx.try_send(msg) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => {
                    s.lagged.store(true, Ordering::Release);
                    false
                }
                Err(TrySendError::Disconnected(_)) => false,
            }
        });
        if subs.is_empty() {
            inner.subs.remove(&session);
        }
    }
}
