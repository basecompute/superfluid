//! The models a fleet head serves: the ones it started with, and each one a node brings.
//!
//! A node that joins with a model the head does not serve brings it: the head serves the model
//! over its nodes from then on, as it serves the ones it started with, lists it in `/v1/models`
//! and announces it on the local network, with no restart. `/v1/models/load` serves a model a
//! node that is up serves, and is refused as not found when none does; `/v1/models/unload` stops
//! serving one. A model stays listed after its last node leaves, and a request for it waits for
//! a node, as one for a model the head started with does.
//!
//! An unloaded model stays unloaded when a node that serves it joins, or joins again: the head
//! says so and does not serve it. `/v1/models/load` serves it again.
//!
//! The head announces the id a node pulls each model by, where it knows one: the id it was
//! given at start, else the first a node serving the model reported. A node that follows its
//! head pulls the head's first model by it.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::fleet_manager::FleetManager;
use crate::fleet_mdns::Announcement;
use crate::registry::{ModelLoader, ModelRegistry};
use crate::{Daemon, DaemonError, Refusal};

/// Builds the daemon that serves a model over the fleet, under its name, once a node that is up
/// serves it.
pub type Build = Box<dyn Fn(&str) -> Result<Arc<Daemon>, DaemonError> + Send + Sync>;

pub struct HeadModels {
    manager: Arc<FleetManager>,
    build: Build,
    registry: OnceLock<Weak<ModelRegistry>>,
    listed: Mutex<Listed>,
    announcement: Mutex<Option<Announcement>>,
}

#[derive(Default)]
struct Listed {
    /// Served, in the order announced: the ones the head started with, then each as it came.
    order: Vec<String>,
    /// Unloaded through the API, which a join does not undo.
    unloaded: HashSet<String>,
    /// Brought by a node, with its name, before the head's registry was in place.
    pending: Vec<(String, String)>,
    /// The id a node pulls each model by, `org/model:tag`, where the head knows one.
    ids: HashMap<String, String>,
}

impl HeadModels {
    /// The models a head starts with, in its order; `build` serves another.
    pub fn new(manager: Arc<FleetManager>, served: Vec<String>, build: Build) -> Arc<HeadModels> {
        Arc::new(HeadModels {
            manager,
            build,
            registry: OnceLock::new(),
            listed: Mutex::new(Listed { order: served, ..Default::default() }),
            announcement: Mutex::new(None),
        })
    }

    /// A head's models that are never more than it started with, for a head that builds none.
    pub fn fixed(manager: Arc<FleetManager>, served: Vec<String>) -> Arc<HeadModels> {
        HeadModels::new(manager, served, Box::new(|_| Err(DaemonError::Config("this head serves only the models it started with"))))
    }

    /// The models served, in the order they are announced.
    pub fn names(&self) -> Vec<String> {
        self.listed.lock().expect("head models").order.clone()
    }

    /// The id a node pulls each of [`Self::names`] by, in the same order; empty for one the
    /// head knows by path only.
    pub fn ids(&self) -> Vec<String> {
        let listed = self.listed.lock().expect("head models");
        listed.order.iter().map(|m| listed.ids.get(m).cloned().unwrap_or_default()).collect()
    }

    /// Notes the id each of `models` is pulled by, as far as `ids` gives them, for a model the
    /// head knows no id for yet, and announces an id that is new for a model served.
    pub fn know_ids(&self, models: &[String], ids: &[String]) {
        let mut announce = false;
        {
            let mut listed = self.listed.lock().expect("head models");
            for (m, id) in models.iter().zip(ids).filter(|(_, id)| !id.is_empty()) {
                if !listed.ids.contains_key(m) {
                    listed.ids.insert(m.clone(), id.clone());
                    announce |= listed.order.contains(m);
                }
            }
        }
        if announce {
            self.announce();
        }
    }

    /// The loader for the head's registry: a model a node that is up serves is served; any other
    /// is refused as not found.
    pub fn loader(self: &Arc<Self>) -> ModelLoader {
        let this = Arc::downgrade(self);
        Box::new(move |name, _runtime| match this.upgrade() {
            Some(this) => this.load(name),
            None => Err(DaemonError::Config("the fleet head is shutting down")),
        })
    }

    fn load(&self, name: &str) -> Result<Arc<Daemon>, DaemonError> {
        if self.manager.serving(name).0 == 0 {
            return Err(DaemonError::Unsupported(Refusal::new(
                "model_not_found",
                Some("model"),
                format!("no node of this fleet that is up serves '{name}'; a node that joins with it is served here"),
            )));
        }
        let daemon = (self.build)(name)?;
        self.listed.lock().expect("head models").unloaded.remove(name);
        Ok(daemon)
    }

    /// Serves each of `models`, which the fleet node `node` serves, that the head does not
    /// serve yet, unless it was unloaded.
    pub fn bring(&self, node: &str, models: &[String]) {
        let new: Vec<String> = {
            let mut listed = self.listed.lock().expect("head models");
            let mut new = Vec::new();
            for m in models {
                if listed.order.contains(m) || listed.pending.iter().any(|(_, p)| p == m) {
                    continue;
                }
                if listed.unloaded.contains(m) {
                    eprintln!("superfluid: fleet node '{node}' serves {m}, which was unloaded here; `/v1/models/load` serves it again");
                } else if self.registry.get().is_none() {
                    listed.pending.push((node.to_string(), m.clone()));
                } else {
                    new.push(m.clone());
                }
            }
            new
        };
        if let Some(registry) = self.registry.get().and_then(Weak::upgrade) {
            for m in new {
                adopt(&registry, node, &m);
            }
        }
    }

    /// Serves the head's models through `registry` from now on, announcing each change, and the
    /// models nodes brought meanwhile now.
    pub fn attach(self: &Arc<Self>, registry: &Arc<ModelRegistry>) {
        let this = Arc::downgrade(self);
        registry.on_change(Box::new(move |names| {
            if let Some(this) = this.upgrade() {
                this.changed(names);
            }
        }));
        let pending = {
            // Taken with the registry set under the same lock, so a model a node brings is
            // either pending here or brought through the registry.
            let mut listed = self.listed.lock().expect("head models");
            let _ = self.registry.set(Arc::downgrade(registry));
            std::mem::take(&mut listed.pending)
        };
        for (node, m) in pending {
            adopt(registry, &node, &m);
        }
    }

    /// Tells the local network of each change to the models served, through `announcement`.
    pub fn announce_with(&self, announcement: Announcement) {
        *self.announcement.lock().expect("head announcement") = Some(announcement);
    }

    fn changed(&self, names: &[String]) {
        {
            let mut listed = self.listed.lock().expect("head models");
            let gone: Vec<String> = listed.order.iter().filter(|m| !names.contains(m)).cloned().collect();
            listed.unloaded.extend(gone);
            listed.order.retain(|m| names.contains(m));
            for n in names {
                if !listed.order.contains(n) {
                    listed.order.push(n.clone());
                }
            }
        }
        self.announce();
    }

    fn announce(&self) {
        let (names, ids) = (self.names(), self.ids());
        if let Some(a) = self.announcement.lock().expect("head announcement").as_ref() {
            if let Err(e) = a.update(&names, &ids) {
                eprintln!("superfluid: the local network was not told of the models served now ({e})");
            }
        }
    }
}

fn adopt(registry: &ModelRegistry, node: &str, name: &str) {
    match registry.load_as(name, name) {
        Ok(true) => eprintln!("superfluid: {name}: served from now on, as fleet node '{node}' serves it"),
        Ok(false) => {}
        Err(e) => eprintln!("superfluid: {name}: fleet node '{node}' serves it, and this head cannot: {e}"),
    }
}
