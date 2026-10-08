use std::sync::atomic::Ordering;

use crate::scheduler::{SchedStats, TTFT_BUCKETS_MS};

pub fn render_prometheus(s: &SchedStats) -> String {
    let g = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
    let mut out = String::new();
    let mut counter = |name: &str, help: &str, value: u64| {
        out.push_str(&format!(
            "# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}\n"
        ));
    };
    counter("superfluid_ticks_total", "Scheduler ticks executed.", g(&s.ticks));
    counter("superfluid_prefill_tokens_total", "Prompt tokens prefilled.", g(&s.prefill_tokens));
    counter("superfluid_decode_tokens_total", "Tokens decoded.", g(&s.decode_tokens));
    counter(
        "superfluid_warm_prefix_tokens_total",
        "Tokens served from the reuse cache or park artifacts instead of re-prefill.",
        g(&s.warm_prefix_tokens),
    );
    counter("superfluid_cold_admissions_total", "Admissions that started cold.", g(&s.cold_admissions));
    counter(
        "superfluid_unservable_cold_admissions_total",
        "Admissions forced cold by a state space no op serves (hybrid guard).",
        g(&s.unservable_cold_admissions),
    );
    counter(
        "superfluid_unservable_park_refusals_total",
        "Parks refused because a state space has no ops (no artifact written).",
        g(&s.unservable_park_refusals),
    );
    counter("superfluid_preemptions_total", "Lanes parked mid-generation and re-queued.", g(&s.preemptions));
    counter("superfluid_admits_deferred_total", "Admissions the engine had no room for this tick, re-queued for a later one.", g(&s.admits_deferred));
    counter(
        "superfluid_starvation_grants_total",
        "Ticks in which a starving agent lane was served ahead of class order.",
        g(&s.starvation_grants),
    );
    counter("superfluid_os_pressure_events_total", "OS memory-pressure signals acted on.", g(&s.os_pressure_events));
    counter(
        "superfluid_pressure_evictions_total",
        "Pool-watermark relief rounds that evicted cache.",
        g(&s.pressure_evictions),
    );
    counter(
        "superfluid_pressure_bytes_evicted_total",
        "Bytes freed by pressure relief rounds.",
        g(&s.pressure_bytes_evicted),
    );
    counter(
        "superfluid_pins_yielded_total",
        "Pinned prefixes that yielded their lease (pin budget, pressure shortfall, unplaceable tick).",
        g(&s.pins_yielded),
    );
    counter(
        "superfluid_pins_expired_total",
        "Pinned prefixes released at their deadline.",
        g(&s.pins_expired),
    );
    counter("superfluid_worker_respawns_total", "Engine worker respawns.", g(&s.worker_respawns));
    counter("superfluid_spec_proposed_total", "Speculative tokens proposed.", g(&s.spec_proposed));
    counter("superfluid_spec_accepted_total", "Speculative tokens accepted.", g(&s.spec_accepted));
    out.push_str("# HELP superfluid_parks_total Park artifacts handed to the writer.\n# TYPE superfluid_parks_total counter\n");
    out.push_str(&format!(
        "superfluid_parks_total{{encoding=\"lossless\"}} {}\nsuperfluid_parks_total{{encoding=\"lossy\"}} {}\n",
        g(&s.parks_lossless),
        g(&s.parks_lossy)
    ));
    out.push_str("# HELP superfluid_resumes_total Admissions that restored a park artifact and seeded from it.\n# TYPE superfluid_resumes_total counter\n");
    out.push_str(&format!("superfluid_resumes_total {}\n", g(&s.resumes)));
    out.push_str("# HELP superfluid_lanes_active Lanes resident after the last tick.\n# TYPE superfluid_lanes_active gauge\n");
    out.push_str(&format!("superfluid_lanes_active {}\n", g(&s.lanes_active)));
    out.push_str("# HELP superfluid_pool_blocks_used KV paged blocks in use.\n# TYPE superfluid_pool_blocks_used gauge\n");
    out.push_str(&format!("superfluid_pool_blocks_used {}\n", g(&s.pool_blocks_used)));
    out.push_str(
        "# HELP superfluid_prefill_budget_tokens Live prefill budget per tick (adaptive unless pinned).\n# TYPE superfluid_prefill_budget_tokens gauge\n",
    );
    out.push_str(&format!("superfluid_prefill_budget_tokens {}\n", g(&s.prefill_budget_live)));
    out.push_str(
        "# HELP superfluid_decode_grant_tokens Live decode grant per lane per tick (contention-scaled on prefill-carrying ticks).\n# TYPE superfluid_decode_grant_tokens gauge\n",
    );
    out.push_str(&format!("superfluid_decode_grant_tokens {}\n", g(&s.decode_grant_live)));
    out.push_str("# HELP superfluid_pool_blocks_total KV paged blocks total (pool capacity).\n# TYPE superfluid_pool_blocks_total gauge\n");
    out.push_str(&format!("superfluid_pool_blocks_total {}\n", g(&s.pool_blocks_total)));
    out.push_str("# HELP superfluid_pins_held Pinned prefixes holding an engine lease.\n# TYPE superfluid_pins_held gauge\n");
    out.push_str(&format!("superfluid_pins_held {}\n", g(&s.pins_held)));
    out.push_str("# HELP superfluid_pinned_bytes Distinct KV bytes held by pinned prefixes.\n# TYPE superfluid_pinned_bytes gauge\n");
    out.push_str(&format!("superfluid_pinned_bytes {}\n", g(&s.pinned_bytes)));
    out.push_str("# HELP superfluid_queue_depth Queued jobs per QoS class.\n# TYPE superfluid_queue_depth gauge\n");
    for (c, name) in ["interactive_chat", "inline_completion", "foreground_agent", "background_agent"]
        .iter()
        .enumerate()
    {
        out.push_str(&format!(
            "superfluid_queue_depth{{class=\"{name}\"}} {}\n",
            g(&s.queue_depth[c])
        ));
    }
    out.push_str("# HELP superfluid_ttft_seconds Time from admission to the first committed token.\n# TYPE superfluid_ttft_seconds histogram\n");
    let mut cum = 0u64;
    for (i, b) in TTFT_BUCKETS_MS.iter().enumerate() {
        cum += g(&s.ttft_buckets[i]);
        out.push_str(&format!(
            "superfluid_ttft_seconds_bucket{{le=\"{}\"}} {cum}\n",
            *b as f64 / 1000.0
        ));
    }
    cum += g(&s.ttft_buckets[TTFT_BUCKETS_MS.len()]);
    out.push_str(&format!("superfluid_ttft_seconds_bucket{{le=\"+Inf\"}} {cum}\n"));
    out.push_str(&format!(
        "superfluid_ttft_seconds_sum {}\nsuperfluid_ttft_seconds_count {}\n",
        g(&s.ttft_sum_ms) as f64 / 1000.0,
        g(&s.ttft_count)
    ));

    out.push_str(
        "# HELP superfluid_node_info Node identity (host label).\n# TYPE superfluid_node_info gauge\n",
    );
    out.push_str(&format!("superfluid_node_info{{host=\"{}\"}} 1\n", label_escape(&hostname())));
    out
}

fn label_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

pub fn hostname() -> String {
    if let Ok(name) = std::env::var("SUPERFLUID_NODE_NAME") {
        if !name.trim().is_empty() {
            return name;
        }
    }
    let mut buf = [0u8; 256];
    // SAFETY: `buf` is a valid 256-byte writable buffer and `buf.len()` is its
    // exact capacity; gethostname writes at most that many bytes and NUL-
    // terminates within it. We only read `buf` after checking rc == 0.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast::<libc::c_char>(), buf.len()) };
    if rc == 0 {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    } else {
        "unknown".into()
    }
}
