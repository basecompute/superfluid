//! The per-node series a fleet head adds to `/metrics`, which `superfluid top` reads.

use crate::fleet_manager::NodeStatus;

/// A per-node series: its name, its Prometheus type and how it reads a node.
type NodeSeries = (&'static str, &'static str, fn(&NodeStatus) -> u64);

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
    out
}

#[cfg(test)]
mod tests {
    use super::fleet_metrics;
    use crate::fleet_manager::NodeStatus;

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
        }
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
