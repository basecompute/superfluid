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

/// What a context takes on its device: so much whatever its size, and so much
/// a cell. A cell costs more there than its KV where the runtime reserves
/// working memory that grows with the pool, and only the device can say how
/// much: the figures are measured, on two contexts made at different sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCost {
    pub fixed_bytes: u64,
    pub bytes_per_cell: u64,
}

impl ContextCost {
    /// From what two contexts took, each as `(cells, bytes)`. `None` where the
    /// two say nothing about a cell: one size, or the larger took no more.
    pub fn from_trials(small: (u64, u64), large: (u64, u64)) -> Option<ContextCost> {
        let (small, large) = if small.0 <= large.0 { (small, large) } else { (large, small) };
        if small.0 == large.0 || large.1 <= small.1 {
            return None;
        }
        let bytes_per_cell = (large.1 - small.1).div_ceil(large.0 - small.0);
        Some(ContextCost { fixed_bytes: small.1.saturating_sub(small.0 * bytes_per_cell), bytes_per_cell })
    }

    pub fn bytes(&self, cells: u64) -> u64 {
        self.fixed_bytes.saturating_add(cells.saturating_mul(self.bytes_per_cell))
    }
}

/// The cells of a pool by what its whole context takes: the pool's share of
/// the free memory, a whole sequence where the share holds less (as
/// `budgeted_cells`), and never so many that the context leaves less than
/// `headroom_bytes` free, which can cut into the one sequence.
pub fn fitted_cells(desired: u64, one_sequence: u64, cost: ContextCost, free_bytes: u64, headroom_bytes: u64) -> u64 {
    if free_bytes == 0 || cost.bytes_per_cell == 0 {
        return desired;
    }
    let cells_within = |bytes: u64| bytes.saturating_sub(cost.fixed_bytes) / cost.bytes_per_cell;
    let share = cells_within((POOL_SHARE_OF_FREE * free_bytes as f64) as u64);
    let most = cells_within(free_bytes.saturating_sub(headroom_bytes));
    desired.min(share).max(one_sequence.min(desired)).min(most)
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

    /// Qwen3-0.6B on llama.cpp b12503 (Metal) with 32 sequence ids, as an M5
    /// Pro measured it: a cell's KV is 114,688 bytes, and the context reserves
    /// another 136,337 for each cell to gather K and V rows into.
    const KV_06B: u64 = 2 * 28 * 8 * 128 * 2;
    const CELL_06B: u64 = KV_06B + 136_337;
    const FIXED_06B: u64 = 160 << 20;

    #[test]
    fn two_contexts_price_a_cell_and_what_a_context_costs_at_any_size() {
        let took = |cells: u64| FIXED_06B + cells * CELL_06B;
        let cost = ContextCost::from_trials((2048, took(2048)), (4096, took(4096))).unwrap();
        assert_eq!(cost, ContextCost { fixed_bytes: FIXED_06B, bytes_per_cell: CELL_06B });
        assert_eq!(ContextCost::from_trials((4096, took(4096)), (2048, took(2048))), Some(cost), "in either order");
        assert_eq!(cost.bytes(208_384), took(208_384));
        assert_eq!(ContextCost::from_trials((2048, took(2048)), (2048, took(2048))), None, "one size prices nothing");
        assert_eq!(ContextCost::from_trials((2048, took(4096)), (4096, took(2048))), None, "nor a larger one that took less");
        let no_fixed = ContextCost::from_trials((2048, 2048 * 100), (4096, 4096 * 100 + 7)).unwrap();
        assert_eq!((no_fixed.fixed_bytes, no_fixed.bytes_per_cell), (0, 101), "a cell is never priced under what the two took");
    }

    #[test]
    fn a_pool_is_sized_by_what_its_context_takes_not_by_its_kv_alone() {
        // 8 lanes of the trained 40,960 on a 48 GB M4 Pro: 39.8 GB free past the weights.
        let (free, headroom) = (39_800_000_000u64, 2_135_000_000u64);
        let (desired, one) = (8 * 40_960u64, 40_960u64);
        let by_kv = budgeted_cells(desired, one, KV_06B, free);
        let cost = ContextCost { fixed_bytes: FIXED_06B, bytes_per_cell: CELL_06B };
        assert!(cost.bytes(by_kv) > free, "priced by its KV the pool's context is {} bytes, past the {free} free", cost.bytes(by_kv));
        let cells = fitted_cells(desired, one, cost, free, headroom);
        assert!(cells >= one && cells < by_kv, "{cells}");
        let share = (POOL_SHARE_OF_FREE * free as f64) as u64;
        assert!(cost.bytes(cells) <= share && cost.bytes(cells + 1) > share, "the context is the pool's share of what is free");
    }

    #[test]
    fn a_fitted_pool_keeps_one_sequence_while_the_headroom_allows() {
        let cost = ContextCost { fixed_bytes: GIB, bytes_per_cell: 1 << 20 };
        let share = (POOL_SHARE_OF_FREE * (100 * GIB) as f64) as u64;
        assert_eq!(fitted_cells(65_536, 16_384, cost, 100 * GIB, 4 * GIB), (share - GIB) >> 20, "the share, less what a context costs at any size");
        assert_eq!(fitted_cells(8192, 4096, cost, 100 * GIB, 4 * GIB), 8192, "an affordable shape is granted whole");
        assert_eq!(fitted_cells(65_536, 16_384, cost, 20 * GIB, 2 * GIB), 16_384, "one sequence past the share");
        assert_eq!(fitted_cells(65_536, 16_384, cost, 12 * GIB, 2 * GIB), 9 * 1024, "and no more than leaves the headroom");
        assert_eq!(fitted_cells(65_536, 16_384, cost, GIB, 2 * GIB), 0, "nothing where the headroom is all there is");
        assert_eq!(fitted_cells(65_536, 16_384, cost, 0, 2 * GIB), 65_536, "no reported budget grants the request");
        assert_eq!(fitted_cells(65_536, 16_384, ContextCost { fixed_bytes: 0, bytes_per_cell: 0 }, 20 * GIB, GIB), 65_536);
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
