//! A daemon whose generations run on fleet nodes. The head keeps every handler, the codec, the
//! session store and what a lane's tokens become; a node runs each generation's decode loop, so
//! no step waits on the network.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use superfluid_abi::finish;
use superfluid_proto::linkf::{GenerateOptsMsg, GrammarSpecMsg, TokenLogprobMsg, BLOB_PREFIX};
use superfluid_proto::linkw::{StateSyncOkMsg, StateSyncReqMsg};

use crate::fleet::{Ask, Blob};
use crate::fleet_manager::FleetManager;
use crate::scheduler::{GenerateJob, LaneEvent, LaneOutput, OutputSink, SchedCmd, SchedStats, SchedulerHandle};
use crate::bus::SessionBus;
use crate::{CancelSet, DaemonError, SessionStore, TextCodec};

/// The longest a generation runs on a node when the request set no deadline.
pub const MAX_WALL_MS: u64 = 600_000;

/// Sessions on the fleet are numbered across every model's daemon on the head, which share
/// one fleet manager.
static NEXT_FLEET_SESSION: AtomicU64 = AtomicU64::new(1);

/// What a fleet-backed daemon's generations need from it.
pub(crate) struct Dispatch {
    pub fleet: Arc<FleetManager>,
    pub model: String,
    pub store: Arc<Mutex<SessionStore>>,
    pub codec: Arc<dyn TextCodec + Send + Sync>,
    pub cancels: Arc<CancelSet>,
    pub active: Arc<Mutex<std::collections::HashSet<u64>>>,
    pub stats: Arc<SchedStats>,
    pub tool_lease_ms: u64,
    pub handles: Handles,
    /// What the daemon says it can do: the first node serving the model's own descriptor, once
    /// one answers.
    pub capabilities: crate::runtime::SharedCapabilities,
}

/// The grammars and logit biases this daemon's handlers made, by the handle each was given:
/// kept here, and compiled on whichever node runs the generation that uses one.
#[derive(Default)]
pub(crate) struct Handles {
    next: std::sync::atomic::AtomicU32,
    made: Mutex<std::collections::HashMap<u32, Made>>,
}

#[derive(Clone)]
enum Made {
    Grammar(GrammarSpecMsg),
    Bias(Vec<i32>, Vec<u32>),
}

impl Handles {
    fn make(&self, what: Made) -> u32 {
        let h = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.made.lock().expect("handles").insert(h, what);
        h
    }

    fn free(&self, h: u32) {
        self.made.lock().expect("handles").remove(&h);
    }

    fn get(&self, h: u32) -> Option<Made> {
        (h != 0).then(|| self.made.lock().expect("handles").get(&h).cloned()).flatten()
    }
}

/// Answers the scheduler's commands for a daemon with no engine of its own: a generation is
/// placed on a node, and its tokens come back through the same lane output a local one has.
pub(crate) fn spawn(dispatch: Dispatch) -> SchedulerHandle {
    let (tx, rx) = std::sync::mpsc::channel();
    let join = std::thread::Builder::new()
        .name("superfluid-fleet-dispatch".into())
        .spawn(move || serve(Arc::new(dispatch), rx))
        .expect("spawn the fleet dispatcher");
    SchedulerHandle::new(tx, join)
}

fn serve(d: Arc<Dispatch>, rx: Receiver<SchedCmd>) {
    let bus = d.store.lock().expect("store").bus();
    learn_capabilities(Arc::downgrade(&d));
    while let Ok(cmd) = rx.recv() {
        match cmd {
            SchedCmd::Submit(job) => {
                let (d, bus) = (Arc::clone(&d), Arc::clone(&bus));
                let spawned = std::thread::Builder::new()
                    .name("superfluid-fleet-job".into())
                    .spawn(move || run(&d, &bus, *job));
                if let Err(e) = spawned {
                    tracing::warn!("fleet: cannot start a generation: {e}");
                }
            }
            SchedCmd::Sync(req, reply) => sync(&d, req, reply),
            SchedCmd::SyncAs(_, inner) => match *inner {
                SchedCmd::Sync(req, reply) => sync(&d, req, reply),
                SchedCmd::SyncStream(req, segments, reply) => sync_stream(&d, req, segments, reply),
                _ => {}
            },
            SchedCmd::SyncStream(req, segments, reply) => sync_stream(&d, req, segments, reply),
            SchedCmd::PinHeld(_, reply) => {
                let _ = reply.send(0);
            }
            SchedCmd::Pin(_) | SchedCmd::PinUnanchor(_) => {}
            SchedCmd::Shutdown => break,
        }
    }
}

pub(crate) const NOT_YET: &str = "this operation does not reach fleet nodes yet";

/// What an error says, without the "generation failed" each daemon it passes through would add.
pub(crate) fn message(e: DaemonError) -> String {
    match e {
        DaemonError::Generation(m) => m,
        e => e.to_string(),
    }
}

/// How often the head asks again for a node's capabilities until one answers.
const CAPABILITIES_EVERY: std::time::Duration = std::time::Duration::from_secs(5);

/// Takes the first node serving the model's capability descriptor as the daemon's, so a route
/// refuses what the nodes cannot do as a local server refuses what its engine cannot.
fn learn_capabilities(d: std::sync::Weak<Dispatch>) {
    let spawned = std::thread::Builder::new().name("superfluid-fleet-capabilities".into()).spawn(move || loop {
        let Some(dispatch) = d.upgrade() else { return };
        if let Ok(StateSyncOkMsg::Capabilities { json }) = dispatch.fleet.state_sync(&dispatch.model, StateSyncReqMsg::Capabilities, &[]) {
            if let Ok(mut v) = serde_json::from_str::<serde_json::Value>(&json) {
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("backend".into(), "fleet".into());
                    *dispatch.capabilities.write().expect("capability record") = crate::capabilities::Capabilities::parse(&v.to_string());
                    return;
                }
            }
        }
        drop(dispatch);
        std::thread::sleep(CAPABILITIES_EVERY);
    });
    if let Err(e) = spawned {
        tracing::warn!("fleet: cannot learn the nodes' capabilities: {e}");
    }
}

type Reply = std::sync::mpsc::Sender<Result<StateSyncOkMsg, String>>;

/// A state operation: a grammar or bias is kept on the head under a handle of its own; one the
/// model's engine answers runs on a node serving it (a LoRA on every one), off this thread, so
/// the commands behind it are not held up.
fn sync(d: &Arc<Dispatch>, req: StateSyncReqMsg, reply: Reply) {
    let answer = match req {
        StateSyncReqMsg::CreateGrammar { json_schema } => {
            StateSyncOkMsg::CreateGrammar { grammar_handle: d.handles.make(Made::Grammar(GrammarSpecMsg::JsonSchema(json_schema))) }
        }
        StateSyncReqMsg::CreateGrammarStructural { tag_json } => {
            StateSyncOkMsg::CreateGrammar { grammar_handle: d.handles.make(Made::Grammar(GrammarSpecMsg::Structural(tag_json))) }
        }
        StateSyncReqMsg::FreeGrammar { grammar_handle } => {
            d.handles.free(grammar_handle);
            StateSyncOkMsg::FreeGrammar
        }
        StateSyncReqMsg::CreateLogitBias { tokens, values } => {
            StateSyncOkMsg::CreateLogitBias { handle: d.handles.make(Made::Bias(tokens, values)) }
        }
        StateSyncReqMsg::FreeLogitBias { handle } => {
            d.handles.free(handle);
            StateSyncOkMsg::FreeLogitBias
        }
        req @ (StateSyncReqMsg::Embed { .. } | StateSyncReqMsg::LoraId | StateSyncReqMsg::Capabilities) => {
            return on_node(d, reply, move |d| d.fleet.state_sync(&d.model, req, &[]));
        }
        req @ StateSyncReqMsg::Transcribe { .. } => {
            return on_node(d, reply, move |d| transcribe(d, req, None));
        }
        StateSyncReqMsg::MediaProbe { image_path } => {
            return on_node(d, reply, move |d| {
                let blob = blob_of(&image_path)?;
                let req = StateSyncReqMsg::MediaProbe { image_path: format!("{BLOB_PREFIX}{}", blob.hash) };
                d.fleet.state_sync(&d.model, req, &[blob])
            });
        }
        req @ (StateSyncReqMsg::LoraLoad { .. } | StateSyncReqMsg::LoraUnload) => {
            return on_node(d, reply, move |d| d.fleet.state_sync_all(&d.model, req));
        }
        _ => {
            let _ = reply.send(Err(NOT_YET.to_string()));
            return;
        }
    };
    let _ = reply.send(Ok(answer));
}

/// A transcription streamed to its caller: its segments are told on as the node has them.
fn sync_stream(
    d: &Arc<Dispatch>,
    req: StateSyncReqMsg,
    segments: std::sync::mpsc::Sender<superfluid_proto::linkw::TranscribeSegmentMsg>,
    reply: Reply,
) {
    if !matches!(req, StateSyncReqMsg::Transcribe { .. }) {
        let _ = reply.send(Err(NOT_YET.to_string()));
        return;
    }
    on_node(d, reply, move |d| transcribe(d, req, Some(segments)));
}

/// Runs a transcription on a node serving the model, its audio sent first.
fn transcribe(
    d: &Dispatch,
    req: StateSyncReqMsg,
    segments: Option<std::sync::mpsc::Sender<superfluid_proto::linkw::TranscribeSegmentMsg>>,
) -> Result<StateSyncOkMsg, DaemonError> {
    let StateSyncReqMsg::Transcribe { audio_path, language, translate, timestamps, prompt, .. } = req else {
        return Err(DaemonError::Protocol("not a transcription"));
    };
    let blob = blob_of(&audio_path)?;
    let req = StateSyncReqMsg::Transcribe {
        audio_path: format!("{BLOB_PREFIX}{}", blob.hash),
        language,
        translate,
        timestamps,
        prompt,
        stream: segments.is_some(),
    };
    match segments {
        Some(segments) => d.fleet.state_sync_streaming(&d.model, req, &[blob], &mut |seg| {
            let _ = segments.send(seg);
        }),
        None => d.fleet.state_sync(&d.model, req, &[blob]),
    }
}

/// The blob a media file on the head is: the media pool names a file by its SHA-256 already;
/// another is hashed.
fn blob_of(path: impl AsRef<std::path::Path>) -> Result<Blob, DaemonError> {
    let path = path.as_ref().to_path_buf();
    let named = path.file_name().and_then(|n| n.to_str()).filter(|n| crate::media::MediaPool::is_valid_id(n)).map(str::to_string);
    let hash = match named {
        Some(hash) => hash,
        None => crate::media::content_hash(&std::fs::read(&path)?),
    };
    Ok(Blob { hash, path })
}

fn on_node(
    d: &Arc<Dispatch>,
    reply: Reply,
    run: impl FnOnce(&Dispatch) -> Result<StateSyncOkMsg, DaemonError> + Send + 'static,
) {
    let d = Arc::clone(d);
    let spawned = std::thread::Builder::new()
        .name("superfluid-fleet-sync".into())
        .spawn(move || {
            let _ = reply.send(run(&d).map_err(message));
        });
    if let Err(e) = spawned {
        tracing::warn!("fleet: cannot run a state operation: {e}");
    }
}

/// What a job asks beyond its sampling, for the node that runs it, with the blobs its media are;
/// none when it asks nothing.
fn options(d: &Dispatch, job: &GenerateJob) -> Result<Option<(GenerateOptsMsg, Vec<Blob>)>, DaemonError> {
    let x = &job.extras;
    let lost = |what: &str| DaemonError::Protocol(if what == "grammar" { "a grammar handle this daemon never made" } else { "a logit bias handle this daemon never made" });
    let grammar = match d.handles.get(x.grammar_handle.max(job.grammar_handle)) {
        Some(Made::Grammar(g)) => Some(g),
        Some(Made::Bias(..)) => return Err(lost("grammar")),
        None if x.grammar_handle != 0 || job.grammar_handle != 0 => return Err(lost("grammar")),
        None => None,
    };
    let logit_bias = match d.handles.get(x.logit_bias_handle) {
        Some(Made::Bias(tokens, values)) => Some((tokens, values)),
        Some(Made::Grammar(_)) => return Err(lost("bias")),
        None if x.logit_bias_handle != 0 => return Err(lost("bias")),
        None => None,
    };
    let plain = x.presence_penalty == 0.0
        && x.frequency_penalty == 0.0
        && (x.repeat_penalty == 0.0 || x.repeat_penalty == 1.0)
        && !x.ignore_eos
        && !x.want_logprobs
        && grammar.is_none()
        && logit_bias.is_none()
        && job.media.is_empty();
    if plain {
        return Ok(None);
    }
    let mut blobs = Vec::with_capacity(job.media.len());
    let mut media = Vec::with_capacity(job.media.len());
    for (offset, path, n) in &job.media {
        let blob = blob_of(path)?;
        media.push((*offset, blob.hash.clone(), *n));
        blobs.push(blob);
    }
    let opts = GenerateOptsMsg {
        session_id: 0,
        generation_id: 0,
        presence_penalty_bits: x.presence_penalty.to_bits(),
        frequency_penalty_bits: x.frequency_penalty.to_bits(),
        repeat_penalty_bits: x.repeat_penalty.to_bits(),
        ignore_eos: x.ignore_eos,
        top_logprobs: x.want_logprobs.then_some(x.top_logprobs),
        grammar,
        logit_bias,
        media,
    };
    Ok(Some((opts, blobs)))
}

/// What a fleet-backed daemon says it can do: generation, and a reason for each thing that does
/// not reach nodes yet, so a route refuses it as the descriptor says.
pub(crate) fn capability_record() -> crate::capabilities::Capabilities {
    crate::capabilities::Capabilities::parse(
        &serde_json::json!({
            "descriptor_version": 1,
            "backend": "fleet",
            "workload": {
                "causal_generation": true,
            },
        })
        .to_string(),
    )
}

fn run(d: &Dispatch, bus: &Arc<SessionBus>, job: GenerateJob) {
    let session = job.session;
    let events = job.events.clone();
    let outcome = generate(d, bus, &job);
    d.active.lock().expect("active set").remove(&session);
    let _ = events.send(match outcome {
        Ok(done) => done,
        Err(e) => LaneEvent::Error(e),
    });
}

/// Places the job on a node, streams its tokens through a lane output, and ends the lane as
/// the scheduler would.
fn generate(d: &Dispatch, bus: &SessionBus, job: &GenerateJob) -> Result<LaneEvent, DaemonError> {
    let opts = options(d, job)?;
    let fs = NEXT_FLEET_SESSION.fetch_add(1, Ordering::Relaxed);
    d.fleet.place(fs, job.params, job.class, "raw", &d.model, &job.stream, job.max_tokens as u64)?;
    // Removes the session from the fleet however this ends.
    struct Placed<'a>(&'a FleetManager, u64);
    impl Drop for Placed<'_> {
        fn drop(&mut self) {
            self.0.finish(self.1);
        }
    }
    let _placed = Placed(&d.fleet, fs);

    let max_wall_ms = match job.deadline {
        Some(at) => at.saturating_duration_since(std::time::Instant::now()).as_millis().max(1) as u64,
        None => MAX_WALL_MS,
    };
    let sink = OutputSink { codec: &d.codec, store: &d.store, bus, tool_lease_ms: d.tool_lease_ms };
    let mut out = LaneOutput::new(&d.codec, job.extras.open_channel, 0);
    let terminators = d.codec.turn_terminators();
    let (mut produced, mut fin) = (0u32, finish::NONE);
    let mut failure: Option<DaemonError> = None;
    let mut gone = false;
    // Timed as a local lane is, batch to batch: (when the first came, tokens by then, when the
    // latest came, tokens by then). The first batch also waited out the prefill.
    let mut batches: Option<(std::time::Instant, u32, std::time::Instant, u32)> = None;

    // A cancel lands in the daemon's cancel set; the node is told while the generation runs.
    let stop = AtomicBool::new(false);
    let generated = std::thread::scope(|scope| {
        let (stop, cancels, fleet, session) = (&stop, &d.cancels, &d.fleet, job.session);
        scope.spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if cancels.lock().expect("cancels").remove(&session) {
                    let _ = fleet.cancel(fs);
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        let events = &job.events;
        let mut on_logprobs = |items: &[TokenLogprobMsg]| {
            let items = items
                .iter()
                .map(|lp| crate::scheduler::TokenLogprob {
                    token: lp.token,
                    logprob: f32::from_bits(lp.logprob_bits),
                    top: lp.top.iter().map(|&(t, p)| (t, f32::from_bits(p))).collect(),
                })
                .collect();
            let _ = events.send(LaneEvent::TokenLogprobs(items));
        };
        let ask = opts.as_ref().map(|(opts, blobs)| Ask { opts, blobs });
        let r = d.fleet.generate_streaming_with(fs, job.max_tokens as u64, max_wall_ms, ask.as_ref(), &mut on_logprobs, |batch| {
            if gone || fin != finish::NONE || failure.is_some() {
                return;
            }
            // The turn ends at a terminator, as it does on a local lane; what a node sent past
            // one is not the model's answer.
            let end = match (job.extras.ignore_eos, batch.iter().position(|t| terminators.contains(t))) {
                (false, Some(i)) => {
                    fin = finish::EOS;
                    let _ = d.fleet.cancel(fs);
                    i + 1
                }
                _ => batch.len(),
            };
            let tokens = &batch[..end];
            produced += tokens.len() as u32;
            let now = std::time::Instant::now();
            batches = Some(match batches {
                None => (now, produced, now, produced),
                Some((at, by, _, _)) => (at, by, now, produced),
            });
            d.stats.decode_tokens.fetch_add(tokens.len() as u64, Ordering::Relaxed);
            if job.completion {
                gone = job.events.send(LaneEvent::CompletionTokens(tokens.to_vec())).is_err();
                return;
            }
            out.provisional(sink.codec, sink.bus, job.session, &job.events, tokens);
            match out.commit(&sink, job.session, &job.events, tokens, finish::NONE, false, job.extras.tool_schemas.as_deref()) {
                Ok(true) => {}
                Ok(false) => gone = true,
                Err(e) => failure = Some(e),
            }
        });
        stop.store(true, Ordering::Relaxed);
        r
    });
    if let Some(e) = failure {
        return Err(e);
    }
    if gone {
        let _ = d.fleet.cancel(fs);
        if !job.completion {
            out.close(&sink, job.session, &job.events, finish::CANCELLED, true)?;
        }
        return Ok(done(produced, finish::CANCELLED, batches));
    }
    let g = generated?;
    if g.expired {
        if job.completion {
            return Ok(LaneEvent::Expired);
        }
        fin = finish::LENGTH;
    } else if fin == finish::NONE {
        fin = g.finish;
    }
    if !job.completion {
        out.commit(&sink, job.session, &job.events, &[], finish::NONE, true, job.extras.tool_schemas.as_deref())?;
        out.close(&sink, job.session, &job.events, fin, false)?;
    }
    Ok(done(produced, fin, batches))
}

fn done(tokens_generated: u32, finish: u32, batches: Option<(std::time::Instant, u32, std::time::Instant, u32)>) -> LaneEvent {
    let decode = batches.map_or((0, 0.0), |(at, by, last, n)| ((n - by) as u64, (last - at).as_secs_f64()));
    LaneEvent::Done { tokens_generated, finish, warm_prefix: 0, spec: (0, 0), decode }
}
