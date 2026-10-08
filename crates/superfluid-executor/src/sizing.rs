//! Pool and window sizing for primitive runtimes.

pub const POOL_SHARE_OF_FREE: f64 = 0.60;

pub const PLANNING_SHARE: f64 = 0.85;

pub const WINDOW_FLOOR: u64 = 4096;

pub const WINDOW_GRANULARITY: u64 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizing {
    pub budget_bytes: u64,
    pub weight_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub trained_context: u64,
}

impl Sizing {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "budget_bytes": self.budget_bytes,
            "weight_bytes": self.weight_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "trained_context": self.trained_context,
        })
    }

    pub fn from_json(v: &serde_json::Value) -> Option<Sizing> {
        Some(Sizing {
            budget_bytes: v["budget_bytes"].as_u64()?,
            weight_bytes: v["weight_bytes"].as_u64()?,
            kv_bytes_per_token: v["kv_bytes_per_token"].as_u64()?,
            trained_context: v["trained_context"].as_u64()?,
        })
    }
}

pub fn reserve_bytes(budget_bytes: u64) -> u64 {
    (1u64 << 30).max(budget_bytes / 8)
}

pub fn window(models: &[Sizing], lanes: u32) -> Option<u64> {
    let budget = models.iter().map(|m| m.budget_bytes).min()?;
    let trained = models.iter().map(|m| m.trained_context).min()?;
    let per_token: u64 = models.iter().map(|m| m.kv_bytes_per_token).sum::<u64>() * u64::from(lanes.max(1));
    if budget == 0 || trained == 0 || per_token == 0 {
        return None;
    }
    let committed = models.iter().map(|m| m.weight_bytes).sum::<u64>() + reserve_bytes(budget);
    let pool = (POOL_SHARE_OF_FREE * budget.saturating_sub(committed) as f64) as u64;
    let mut n = pool / per_token;
    n -= n % WINDOW_GRANULARITY;
    Some(n.min(trained).max(WINDOW_FLOOR.min(trained)))
}

pub fn headroom_bytes(budget_bytes: u64) -> u64 {
    reserve_bytes(budget_bytes) / 2
}

pub fn cells_over_budget(budget_bytes: u64, free_after: u64, bytes_per_cell: u64) -> u64 {
    if budget_bytes == 0 || bytes_per_cell == 0 {
        return 0;
    }
    headroom_bytes(budget_bytes).saturating_sub(free_after).div_ceil(bytes_per_cell)
}

/// A token count as people read it in a log line: 4096, 906k, 1.2M.
pub fn tokens(n: u64) -> String {
    match n {
        0..=9_999 => n.to_string(),
        10_000..=999_499 => format!("{}k", (n + 500) / 1000),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

pub fn budgeted_cells(desired: u64, one_sequence: u64, bytes_per_cell: u64, free_bytes: u64) -> u64 {
    if free_bytes == 0 || bytes_per_cell == 0 {
        return desired;
    }
    let cap = (POOL_SHARE_OF_FREE * free_bytes as f64) as u64 / bytes_per_cell;
    desired.min(cap).max(one_sequence.min(desired))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counts_read_as_people_say_them() {
        assert_eq!(tokens(4096), "4096");
        assert_eq!(tokens(131_072), "131k");
        assert_eq!(tokens(905_562), "906k");
        assert_eq!(tokens(2_097_152), "2.1M");
        assert_eq!(tokens(999_499), "999k");
        assert_eq!(tokens(999_500), "1.0M", "never 1000k");
    }

    const GIB: u64 = 1 << 30;

    #[test]
    fn an_affordable_shape_is_granted_whole() {
        assert_eq!(budgeted_cells(65_536, 16_384, 144 << 10, 30 * GIB), 65_536);
    }

    #[test]
    fn a_shape_past_the_budget_is_capped_to_it() {
        let cells = budgeted_cells(262_144, 32_768, 96 << 10, 20 * GIB);
        assert_eq!(cells, (0.6 * (20 * GIB) as f64) as u64 / (96 << 10));
        assert!((32_768..262_144).contains(&cells));
    }

    #[test]
    fn one_full_sequence_always_fits() {
        assert_eq!(budgeted_cells(65_536, 16_384, 144 << 10, GIB), 16_384);
    }

    #[test]
    fn no_reported_budget_grants_the_request() {
        assert_eq!(budgeted_cells(65_536, 16_384, 144 << 10, 0), 65_536);
        assert_eq!(budgeted_cells(65_536, 16_384, 0, 30 * GIB), 65_536);
    }

    fn model(budget: u64, weights: u64, price: u64, trained: u64) -> Sizing {
        Sizing { budget_bytes: budget * GIB, weight_bytes: weights * GIB, kv_bytes_per_token: price, trained_context: trained }
    }

    const KV_8B: u64 = 2 * 32 * 8 * 128 * 2;

    #[test]
    fn the_native_engines_cases_size_the_same() {
        let mut m = model(96, 5, KV_8B, 131_072);
        assert_eq!(window(&[m], 1), Some(131_072));
        m.trained_context = 262_144;
        assert_eq!(window(&[m], 1), Some(262_144));
        let many = window(&[m], 32).unwrap();
        assert!((WINDOW_FLOOR..262_144).contains(&many), "{many}");
        let small = window(&[model(6, 2, KV_8B, 131_072)], 1).unwrap();
        assert!((WINDOW_FLOOR..131_072).contains(&small), "{small}");
        let m = model(14, 8, KV_8B, 1 << 20);
        let (one, four) = (window(&[m], 1).unwrap(), window(&[m], 4).unwrap());
        assert!(four * 4 <= one + 4 * WINDOW_GRANULARITY && four * 4 + 4 * WINDOW_GRANULARITY >= one, "{one} {four}");
        let f16 = window(&[model(12, 5, KV_8B, 1 << 20)], 1).unwrap();
        let q8 = window(&[model(12, 5, KV_8B * 34 / 64, 1 << 20)], 1).unwrap();
        assert!(q8 > f16, "{f16} {q8}");
        for budget in [4, 8, 16, 24, 48, 96, 192] {
            let w = window(&[model(budget, 3, KV_8B, 1 << 20)], 1).unwrap();
            assert!(w.is_multiple_of(WINDOW_GRANULARITY) && (WINDOW_FLOOR..=1 << 20).contains(&w), "{budget}: {w}");
        }
    }

    #[test]
    fn a_window_is_what_the_budget_leaves_for_every_lane() {
        let m = model(40, 4, 144 << 10, 1 << 20);
        let raw = (0.6 * (31 * GIB) as f64) as u64 / ((144 << 10) * 4);
        assert_eq!(window(&[m], 4), Some(raw - raw % WINDOW_GRANULARITY));
    }

    #[test]
    fn the_trained_window_caps_it_and_the_floor_holds_it_up() {
        assert_eq!(window(&[model(128, 4, 16 << 10, 40_960)], 4), Some(40_960));
        assert_eq!(window(&[model(16, 30, 96 << 10, 262_144)], 8), Some(WINDOW_FLOOR));
        assert_eq!(window(&[model(16, 30, 96 << 10, 2048)], 8), Some(2048));
    }

    #[test]
    fn models_on_one_device_share_it() {
        let a = model(40, 4, 64 << 10, 1 << 20);
        let b = model(48, 8, 32 << 10, 32_768);
        let both = window(&[a, b], 2).unwrap();
        let left = 40 * GIB - (12 * GIB + reserve_bytes(40 * GIB));
        let raw = (0.6 * left as f64) as u64 / ((96 << 10) * 2);
        assert_eq!(both, (raw - raw % 1024).min(32_768));
        assert!(both < window(&[a], 2).unwrap());
    }

    #[test]
    fn unknown_inputs_size_nothing() {
        assert_eq!(window(&[], 4), None);
        assert_eq!(window(&[model(0, 4, 1024, 32_768)], 4), None);
        assert_eq!(window(&[model(40, 4, 0, 32_768)], 4), None);
        assert_eq!(window(&[model(40, 4, 1024, 0)], 4), None);
    }

    #[test]
    fn sizing_travels_as_json() {
        let m = model(40, 4, 144 << 10, 40_960);
        assert_eq!(Sizing::from_json(&m.to_json()), Some(m));
        assert_eq!(Sizing::from_json(&serde_json::json!({"budget_bytes": 1})), None);
    }

    #[test]
    fn a_context_that_ate_the_headroom_gives_cells_back() {
        let budget = 40 * GIB;
        let headroom = headroom_bytes(budget);
        assert_eq!(headroom, 5 * GIB / 2);
        assert_eq!(headroom_bytes(4 * GIB), GIB / 2, "never under half a GiB");
        assert_eq!(cells_over_budget(budget, headroom, 1024), 0);
        assert_eq!(cells_over_budget(budget, headroom + GIB, 1024), 0);
        assert_eq!(cells_over_budget(budget, headroom - 1024 * 10, 1024), 10);
        assert_eq!(cells_over_budget(budget, headroom - 1, 1024), 1);
        assert_eq!(cells_over_budget(budget, 0, 1 << 20), headroom.div_ceil(1 << 20));
        assert_eq!(cells_over_budget(0, 0, 1024), 0);
        assert_eq!(cells_over_budget(budget, 0, 0), 0);
    }
}
