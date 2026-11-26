//! Does the loaded model set fit the device it was sized for?
//!
//! `--max-context auto` asks the runtime for a window before anything loads,
//! and the runtime answers from its own plan: the bundles' file sizes, the KV
//! pool and a reserve. A load can hold more than that plan — baseRT 0.4.5 on
//! Metal also builds fused and tiled copies of a dense model's projections
//! (gate|up, q|k|v, ...) beside the still-mapped originals, and charges none of
//! them, so Qwen3.8-27B with a DFlash drafter at 8 lanes x 8192 planned 30.2 GB
//! of a 34.2 GB budget and held 37.6 GB at decode, and every decode failed with
//! Metal's Insufficient Memory. So once the primary worker has loaded, its real
//! footprint is measured and, while the window and lanes are still the auto
//! sizing's to choose, they are refitted to what the device holds, before
//! anything is served; a set that fits at no window refuses to start.

/// What decode will hold, measured once the primary worker has loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Measured {
    /// The worker process's physical footprint: everything it allocated
    /// (KV pool, recurrent state, scratch, copied and fused weights).
    pub worker_bytes: u64,
    /// The bundles mapped and wrapped for the GPU (the model itself, its
    /// speculators, the other models served beside it). Clean file pages are
    /// not in a footprint, but a command buffer makes the whole wrapped range
    /// resident, released pages included.
    pub mapped_bytes: u64,
    /// What attaches after the measurement: a drafter's own KV, scratch and
    /// copies, charged at its file size (Qwen3.8-27B's 1.46 GB DFlash2 added
    /// 1.1 GB to its worker at 8 lanes x 5120).
    pub pending_bytes: u64,
}

impl Measured {
    pub fn total(&self) -> u64 {
        self.worker_bytes.saturating_add(self.mapped_bytes).saturating_add(self.pending_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// Within the budget as loaded.
    Fits,
    /// Serve this window and lane count instead.
    Refit { ctx: i32, lanes: u32 },
    /// Over the budget even at the floor window on one lane (or on the lanes
    /// that were pinned); `floor_bytes` is the estimate there.
    Refuse { floor_bytes: u64 },
}

/// The window and lanes that bring a measured set within `budget`.
///
/// Only the KV pool scales with them, at `kv_lane_token` bytes per token per
/// lane. The window shrinks by the overshoot while it stays at or above
/// `keep`; below that, lanes halve (unless `lanes_pinned`), since a window
/// too short for a long prompt is worth less than concurrency. On the last
/// lane count the window may go down to `floor`. The window never grows past
/// `ctx`: fewer lanes buy a fit, not a longer window than the one sized.
/// Recurrent state and scratch that also shrink with fewer lanes are left
/// uncounted, which errs towards fitting.
#[allow(clippy::too_many_arguments)]
pub fn refit(
    budget: u64,
    measured: u64,
    ctx: i32,
    lanes: u32,
    kv_lane_token: u64,
    lanes_pinned: bool,
    keep: i32,
    floor: i32,
    granularity: i32,
) -> Fit {
    if measured <= budget {
        return Fit::Fits;
    }
    let over = measured - budget;
    let (ctx_u, lanes) = (ctx.max(0) as u64, lanes.max(1));
    let k = kv_lane_token.max(1);
    let floor = floor.min(ctx).max(1);
    let keep = keep.clamp(floor, ctx.max(floor));
    let gran = granularity.max(1) as u64;
    let mut l = lanes;
    loop {
        let freed_by_lanes = k * ctx_u * u64::from(lanes - l);
        let need = over.saturating_sub(freed_by_lanes);
        let per_token = k * u64::from(l);
        let cut = need.div_ceil(per_token);
        let mut n = ctx_u.saturating_sub(cut);
        n -= n % gran;
        let last = lanes_pinned || l == 1;
        if n >= keep as u64 || (last && n >= floor as u64) {
            return Fit::Refit { ctx: n as i32, lanes: l };
        }
        if last {
            let at_floor = measured.saturating_sub(k * (ctx_u * u64::from(lanes) - floor as u64 * u64::from(l)));
            return Fit::Refuse { floor_bytes: at_floor };
        }
        l = (l / 2).max(1);
    }
}

/// A process's physical footprint, the figure Activity Monitor shows as its
/// memory: on Apple silicon this includes its Metal allocations.
#[cfg(target_os = "macos")]
pub fn phys_footprint(pid: u32) -> Option<u64> {
    // SAFETY: rusage_info_v2 is plain integers, for which all-zero is a valid value.
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    // SAFETY: RUSAGE_INFO_V2 fills exactly a rusage_info_v2, which `info` is.
    let rc = unsafe {
        libc::proc_pid_rusage(pid as libc::c_int, libc::RUSAGE_INFO_V2, &mut info as *mut _ as *mut libc::rusage_info_t)
    };
    (rc == 0).then_some(info.ri_phys_footprint)
}

#[cfg(not(target_os = "macos"))]
pub fn phys_footprint(_pid: u32) -> Option<u64> {
    None
}

pub fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;
    // Qwen3.8-27B on a 48 GB M5 Pro: 65536 B of KV per token per lane.
    const K: u64 = 65536;

    #[test]
    fn a_set_within_its_budget_is_left_alone() {
        assert_eq!(refit(34 * GB, 33 * GB, 8192, 8, K, false, 4096, 4096, 1024), Fit::Fits);
    }

    #[test]
    fn a_small_overshoot_shrinks_the_window_and_keeps_the_lanes() {
        // 0.5 GB over at 8 lanes: 0.5e9 / (8 * 65536) = 954 tokens -> 7168.
        assert_eq!(refit(34 * GB, 34 * GB + GB / 2, 8192, 8, K, false, 4096, 4096, 1024), Fit::Refit { ctx: 7168, lanes: 8 });
    }

    #[test]
    fn an_overshoot_the_window_cannot_absorb_halves_the_lanes() {
        // The measured case: 36.5 GB of 34.2. At 8 lanes the window would drop
        // under 4096; at 4 lanes the freed half of the pool (2.1 GB) leaves
        // 0.15 GB, 583 tokens of 4-lane KV.
        let fit = refit(34_200_000_000, 36_500_000_000, 8192, 8, K, false, 4096, 4096, 1024);
        assert_eq!(fit, Fit::Refit { ctx: 7168, lanes: 4 });
    }

    #[test]
    fn lanes_halve_before_the_window_drops_below_the_one_to_keep() {
        // 1.6 GB over at 8 x 7168: at 8 lanes the window would fall to 4096,
        // under the 4608 kept (half the sized 9216); 4 lanes free 1.9 GB.
        assert_eq!(
            refit(34_200_000_000, 35_800_000_000, 7168, 8, K, false, 4608, 4096, 1024),
            Fit::Refit { ctx: 7168, lanes: 4 }
        );
        // The same overshoot on pinned lanes goes down to the floor instead.
        assert_eq!(
            refit(34_200_000_000, 35_800_000_000, 7168, 8, K, true, 4608, 4096, 1024),
            Fit::Refit { ctx: 4096, lanes: 8 }
        );
    }

    #[test]
    fn pinned_lanes_refuse_instead_of_halving() {
        assert!(matches!(refit(34_200_000_000, 36_500_000_000, 8192, 8, K, true, 4096, 4096, 1024), Fit::Refuse { .. }));
    }

    #[test]
    fn a_set_over_budget_at_one_lane_and_the_floor_refuses_with_the_floor_estimate() {
        // 10 GB over with only 4.3 GB of KV to give back.
        match refit(30 * GB, 40 * GB, 8192, 8, K, false, 4096, 4096, 1024) {
            Fit::Refuse { floor_bytes } => {
                assert_eq!(floor_bytes, 40 * GB - K * (8192 * 8 - 4096));
                assert!(floor_bytes > 30 * GB);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_window_never_grows_past_the_sized_one() {
        // Halving the lanes frees far more than the overshoot.
        assert_eq!(refit(34 * GB, 34 * GB + 1, 4096, 8, K, false, 4096, 4096, 1024), Fit::Refit { ctx: 4096, lanes: 4 });
    }

    #[test]
    fn a_floor_above_the_window_is_the_window() {
        assert_eq!(refit(34 * GB, 34 * GB + 1, 2048, 1, K, false, 4096, 4096, 1024), Fit::Refuse { floor_bytes: 34 * GB + 1 });
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn this_process_has_a_footprint() {
        assert!(phys_footprint(std::process::id()).is_some_and(|b| b > 0));
    }
}
