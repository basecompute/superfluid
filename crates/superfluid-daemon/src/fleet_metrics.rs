//! The per-node series a fleet head adds to `/metrics`, which `superfluid top` reads.

use superfluid_proto::linkf::NodeStatsMsg;

use crate::fleet_manager::NodeStatus;

/// A per-node series: its name, its Prometheus type and how it reads a node.
type NodeSeries = (&'static str, &'static str, fn(&NodeStatus) -> u64);

/// A scheduler counter nodes report: the series it is, after `superfluid_` on the head's own
/// and after `superfluid_fleet_node_` on each node's; the label it carries beyond a node's
/// (empty for none); and how a node's status reads it.
pub(crate) type NodeCounter = (&'static str, &'static str, fn(&NodeStatsMsg) -> u64);

/// The counters nodes report, in the order [`NodeCounts`] holds them.
pub(crate) const NODE_COUNTERS: [NodeCounter; 19] = [
    ("prefill_tokens_total", "", |s| s.prefill_tokens),
    ("warm_prefix_tokens_total", "", |s| s.warm_prefix_tokens),
    ("cold_admissions_total", "", |s| s.cold_admissions),
    ("unservable_cold_admissions_total", "", |s| s.unservable_cold_admissions),
    ("unservable_park_refusals_total", "", |s| s.unservable_park_refusals),
    ("preemptions_total", "", |s| s.preemptions),
    ("admits_deferred_total", "", |s| s.admits_deferred),
    ("starvation_grants_total", "", |s| s.starvation_grants),
    ("os_pressure_events_total", "", |s| s.os_pressure_events),
    ("pressure_evictions_total", "", |s| s.pressure_evictions),
    ("pressure_bytes_evicted_total", "", |s| s.pressure_bytes_evicted),
    ("pins_yielded_total", "", |s| s.pins_yielded),
    ("pins_expired_total", "", |s| s.pins_expired),
    ("worker_respawns_total", "", |s| s.worker_respawns),
    ("spec_proposed_total", "", |s| s.spec_proposed),
    ("spec_accepted_total", "", |s| s.spec_accepted),
    ("parks_total", "encoding=\"lossless\"", |s| s.parks_lossless),
    ("parks_total", "encoding=\"lossy\"", |s| s.parks_lossy),
    ("resumes_total", "", |s| s.resumes),
];

/// One count per [`NODE_COUNTERS`].
pub type Counts = [u64; NODE_COUNTERS.len()];

/// The count of `counts` the head's own series `key` adds, as [`crate::metrics`] names it:
/// `superfluid_<series>`, with `{<label>}` where it has one.
pub(crate) fn head_count(counts: &Counts, key: &str) -> u64 {
    let Some(series) = key.strip_prefix("superfluid_") else { return 0 };
    let (name, label) = match series.split_once('{') {
        Some((name, label)) => (name, label.strip_suffix('}').unwrap_or(label)),
        None => (series, ""),
    };
    NODE_COUNTERS.iter().zip(counts).find(|((n, l, _), _)| *n == name && *l == label).map_or(0, |(_, &c)| c)
}

/// A node's counters as the head keeps them: what it counted over every start the head has
/// seen, which only grows. Each status gives the counters since one start of the node
/// ([`NodeStatsMsg::run`]) and adds what it counted beyond the highest the head has of that
/// start: the first status of a start adds its counters whole, and one lower than the highest
/// (read late, from another of the node's connections) adds nothing. Each start is kept by
/// itself, because the order statuses are read in does not say which start is the running one:
/// the first status read of a start that ended can come after the next start's.
#[derive(Debug, Default, Clone)]
pub struct NodeCounts {
    /// The highest of each counter in each start, the start read last at the end.
    runs: Vec<(u64, Counts)>,
    total: Counts,
}

impl NodeCounts {
    /// How many starts are kept, the ones read longest ago dropped first. A status of a
    /// dropped start would add its counters whole again, so this is far more than the starts
    /// a node's connections can still hold a status of.
    const RUNS: usize = 8;

    /// Takes in a status's counters; what they add to the node's totals.
    pub fn absorb(&mut self, stats: &NodeStatsMsg) -> Counts {
        let mut high = match self.runs.iter().position(|(run, _)| *run == stats.run) {
            Some(known) => self.runs.remove(known).1,
            None => {
                if self.runs.len() == Self::RUNS {
                    self.runs.remove(0);
                }
                [0; NODE_COUNTERS.len()]
            }
        };
        let mut added = [0; NODE_COUNTERS.len()];
        for (i, (_, _, read)) in NODE_COUNTERS.iter().enumerate() {
            let now = read(stats);
            added[i] = now.saturating_sub(high[i]);
            high[i] = high[i].max(now);
            self.total[i] = self.total[i].saturating_add(added[i]);
        }
        self.runs.push((stats.run, high));
        added
    }

    /// The node's totals; `None` until it has reported any (a node released before them).
    pub fn total(&self) -> Option<Counts> {
        (!self.runs.is_empty()).then_some(self.total)
    }
}

/// Adds `added` into `into`, as the head sums its nodes.
pub fn add(into: &mut Counts, added: &Counts) {
    for (t, a) in into.iter_mut().zip(added) {
        *t = t.saturating_add(*a);
    }
}

/// The head's info and each node's series. `models` is every model the head serves, its
/// default first: the info carries that one as `model`, as it always has, and all of them as
/// `models`. Each node's series carry the models it serves as `model`, joined by `,` and empty
/// until it has reported them.
pub(crate) fn fleet_metrics(nodes: &[NodeStatus], host: &str, models: &[String]) -> String {
    use std::fmt::Write;
    let esc = crate::metrics::label_escape;
    let mut out = String::new();
    let _ = writeln!(out, "# TYPE superfluid_fleet_head_info gauge");
    let _ = writeln!(
        out,
        "superfluid_fleet_head_info{{host=\"{}\",model=\"{}\",models=\"{}\"}} 1",
        esc(host),
        esc(models.first().map(String::as_str).unwrap_or_default()),
        esc(&models.join(","))
    );
    let series: [NodeSeries; 7] = [
        ("superfluid_fleet_node_up", "gauge", |n| n.up as u64),
        ("superfluid_fleet_node_lanes", "gauge", |n| n.lanes as u64),
        ("superfluid_fleet_node_lanes_active", "gauge", |n| n.lanes_active as u64),
        ("superfluid_fleet_node_queue_depth", "gauge", |n| n.queue_depth as u64),
        ("superfluid_fleet_node_pool_blocks_used", "gauge", |n| n.kv_blocks_used),
        ("superfluid_fleet_node_pool_blocks_total", "gauge", |n| n.kv_blocks_total),
        ("superfluid_fleet_node_decode_tokens_total", "counter", |n| n.decode_tokens),
    ];
    for (name, kind, value) in series {
        let _ = writeln!(out, "# TYPE {name} {kind}");
        for n in nodes {
            let _ = writeln!(
                out,
                "{name}{{node=\"{}\",addr=\"{}\",model=\"{}\"}} {}",
                esc(&n.name),
                esc(&n.addr),
                esc(&n.models.join(",")),
                value(n)
            );
        }
    }
    // The counters each node's scheduler reports, from a node that reports them.
    let mut typed = "";
    for (i, (series, label, _)) in NODE_COUNTERS.iter().enumerate() {
        if *series != typed {
            let _ = writeln!(out, "# TYPE superfluid_fleet_node_{series} counter");
            typed = series;
        }
        let label = if label.is_empty() { String::new() } else { format!(",{label}") };
        for n in nodes {
            if let Some(counts) = &n.counters {
                let _ = writeln!(
                    out,
                    "superfluid_fleet_node_{series}{{node=\"{}\",addr=\"{}\",model=\"{}\"{label}}} {}",
                    esc(&n.name),
                    esc(&n.addr),
                    esc(&n.models.join(",")),
                    counts[i]
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{fleet_metrics, head_count, NodeCounts, NODE_COUNTERS};
    use crate::fleet_manager::NodeStatus;
    use superfluid_proto::linkf::NodeStatsMsg;

    fn node(name: &str, models: &[&str]) -> NodeStatus {
        NodeStatus {
            name: name.to_string(),
            addr: "10.0.0.5".to_string(),
            up: true,
            lanes: 4,
            lanes_active: 0,
            queue_depth: 0,
            kv_blocks_used: 0,
            kv_blocks_total: 0,
            decode_tokens: 0,
            models: models.iter().map(|m| m.to_string()).collect(),
            counters: None,
        }
    }

    fn spec(run: u64, proposed: u64, accepted: u64) -> NodeStatsMsg {
        NodeStatsMsg { run, spec_proposed: proposed, spec_accepted: accepted, ..Default::default() }
    }

    fn at(counts: &super::Counts, key: &str) -> u64 {
        head_count(counts, key)
    }

    #[test]
    fn a_nodes_counts_grow_by_what_each_status_adds_and_only_grow() {
        let mut c = NodeCounts::default();
        assert_eq!(c.total(), None, "nothing before a status that carries counters");
        let added = c.absorb(&spec(1, 10, 6));
        assert_eq!((at(&added, "superfluid_spec_proposed_total"), at(&added, "superfluid_spec_accepted_total")), (10, 6));
        let added = c.absorb(&spec(1, 25, 20));
        assert_eq!(at(&added, "superfluid_spec_proposed_total"), 15, "what it counted since");
        // Another connection's status, read late: older, so lower; not a restart.
        let added = c.absorb(&spec(1, 18, 12));
        assert_eq!(added, [0; NODE_COUNTERS.len()]);
        let total = c.total().unwrap();
        assert_eq!((at(&total, "superfluid_spec_proposed_total"), at(&total, "superfluid_spec_accepted_total")), (25, 20));

        // The node restarted: its counters start from 0 again, and add whole.
        let added = c.absorb(&spec(2, 4, 3));
        assert_eq!(at(&added, "superfluid_spec_proposed_total"), 4);
        // A status of the start that ended, read after the restart, adds what that start
        // counted beyond the highest the head had of it, and leaves the running start's alone.
        let added = c.absorb(&spec(1, 30, 22));
        assert_eq!((at(&added, "superfluid_spec_proposed_total"), at(&added, "superfluid_spec_accepted_total")), (5, 2));
        let added = c.absorb(&spec(2, 9, 7));
        assert_eq!((at(&added, "superfluid_spec_proposed_total"), at(&added, "superfluid_spec_accepted_total")), (5, 4));
        let total = c.total().unwrap();
        assert_eq!((at(&total, "superfluid_spec_proposed_total"), at(&total, "superfluid_spec_accepted_total")), (39, 29));
    }

    #[test]
    fn the_first_status_of_a_start_that_ended_read_late_does_not_stop_the_running_starts() {
        let mut c = NodeCounts::default();
        // The running start is read first; the start before it has a status still unread on
        // another connection, the first the head reads of it.
        assert_eq!(at(&c.absorb(&spec(2, 4, 0)), "superfluid_spec_proposed_total"), 4);
        assert_eq!(at(&c.absorb(&spec(1, 10, 0)), "superfluid_spec_proposed_total"), 10);
        assert_eq!(at(&c.absorb(&spec(2, 9, 0)), "superfluid_spec_proposed_total"), 5, "the running start keeps counting");
        assert_eq!(at(&c.absorb(&spec(1, 10, 0)), "superfluid_spec_proposed_total"), 0);
        assert_eq!(at(&c.total().unwrap(), "superfluid_spec_proposed_total"), 19);
    }

    #[test]
    fn a_start_read_again_is_kept_past_the_starts_read_before_it() {
        let mut c = NodeCounts::default();
        for run in 1..=NodeCounts::RUNS as u64 {
            c.absorb(&spec(run, 10, 0));
        }
        // The first start reports again, then one more start than is kept: the start read
        // longest ago is the one dropped, which is no longer the first.
        c.absorb(&spec(1, 12, 0));
        c.absorb(&spec(100, 1, 0));
        assert_eq!(at(&c.absorb(&spec(1, 12, 0)), "superfluid_spec_proposed_total"), 0, "still known");
        assert_eq!(at(&c.total().unwrap(), "superfluid_spec_proposed_total"), 10 * NodeCounts::RUNS as u64 + 2 + 1);
    }

    #[test]
    fn each_counter_names_a_series_the_head_serves_itself() {
        let s = crate::scheduler::SchedStats::default();
        let own = crate::metrics::render_prometheus(&s);
        let mut counts = [0; NODE_COUNTERS.len()];
        for (i, (series, label, _)) in NODE_COUNTERS.iter().enumerate() {
            let key = if label.is_empty() { format!("superfluid_{series}") } else { format!("superfluid_{series}{{{label}}}") };
            assert!(own.lines().any(|l| l == format!("{key} 0")), "{key} is not a series of the head's own:\n{own}");
            counts[i] = i as u64 + 1;
            assert_eq!(head_count(&counts, &key), i as u64 + 1, "{key}");
            counts[i] = 0;
        }
        assert_eq!(head_count(&counts, "superfluid_decode_tokens_total"), 0, "the head counts its own decode");
    }

    #[test]
    fn a_node_that_reports_counters_has_its_series_and_one_that_does_not_has_none() {
        let mut reporting = node("spark", &["qwen"]);
        let mut c = NodeCounts::default();
        c.absorb(&NodeStatsMsg { run: 1, warm_prefix_tokens: 96, preemptions: 2, parks_lossy: 1, ..Default::default() });
        reporting.counters = c.total();
        let text = fleet_metrics(&[reporting, node("old", &["qwen"])], "head", &["qwen".to_string()]);
        assert!(text.contains("superfluid_fleet_node_warm_prefix_tokens_total{node=\"spark\",addr=\"10.0.0.5\",model=\"qwen\"} 96"), "{text}");
        assert!(text.contains("superfluid_fleet_node_preemptions_total{node=\"spark\",addr=\"10.0.0.5\",model=\"qwen\"} 2"), "{text}");
        assert!(text.contains("superfluid_fleet_node_parks_total{node=\"spark\",addr=\"10.0.0.5\",model=\"qwen\",encoding=\"lossy\"} 1"), "{text}");
        assert_eq!(text.matches("# TYPE superfluid_fleet_node_parks_total counter").count(), 1, "{text}");
        assert!(!text.contains("superfluid_fleet_node_spec_proposed_total{node=\"old\""), "a node from before counters reports none: {text}");
    }

    #[test]
    fn each_node_series_names_the_models_the_node_serves_and_none_before_it_reports() {
        let text = fleet_metrics(&[node("spark", &["qwen", "gemma"]), node("m3", &[])], "head", &["qwen".to_string()]);
        let up: Vec<&str> = text.lines().filter(|l| l.starts_with("superfluid_fleet_node_up{")).collect();
        assert_eq!(
            up,
            [
                "superfluid_fleet_node_up{node=\"spark\",addr=\"10.0.0.5\",model=\"qwen,gemma\"} 1",
                "superfluid_fleet_node_up{node=\"m3\",addr=\"10.0.0.5\",model=\"\"} 1",
            ]
        );
        let labelled = text.lines().filter(|l| l.starts_with("superfluid_fleet_node_") && l.contains("model=\"qwen,gemma\"")).count();
        assert_eq!(labelled, 7, "every per-node series carries the label:\n{text}");
    }

    #[test]
    fn the_head_info_keeps_its_first_model_and_lists_them_all() {
        let text = fleet_metrics(&[], "head", &["qwen".to_string(), "gemma".to_string()]);
        assert!(text.contains("superfluid_fleet_head_info{host=\"head\",model=\"qwen\",models=\"qwen,gemma\"} 1"), "{text}");
    }
}
