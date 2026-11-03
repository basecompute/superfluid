//! The per-node series a fleet head adds to `/metrics`, which `superfluid top` reads.

use crate::fleet_manager::NodeStatus;

/// A per-node series: its name, its Prometheus type and how it reads a node.
type NodeSeries = (&'static str, &'static str, fn(&NodeStatus) -> u64);

pub(crate) fn fleet_metrics(nodes: &[NodeStatus], host: &str, model: &str) -> String {
    use std::fmt::Write;
    let esc = crate::metrics::label_escape;
    let mut out = String::new();
    let _ = writeln!(out, "# TYPE superfluid_fleet_head_info gauge");
    let _ = writeln!(out, "superfluid_fleet_head_info{{host=\"{}\",model=\"{}\"}} 1", esc(host), esc(model));
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
            let _ = writeln!(out, "{name}{{node=\"{}\",addr=\"{}\"}} {}", esc(&n.name), esc(&n.addr), value(n));
        }
    }
    out
}
