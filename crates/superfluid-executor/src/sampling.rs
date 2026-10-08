use superfluid_abi::SamplingParams;

pub fn argmax(row: &[f32]) -> u32 {
    if row.first().is_none_or(|v| v.is_nan()) {
        return 0;
    }
    let best = max_of(row);
    const BLOCK: usize = 64;
    for (b, block) in row.chunks(BLOCK).enumerate() {
        if max_of(block) == best {
            let at = block.iter().position(|&v| v == best).unwrap_or(0);
            return (b * BLOCK + at) as u32;
        }
    }
    0
}

fn max_of(row: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmaxnmq_f32, vmaxnmvq_f32};
        let (chunks, rest) = row.as_chunks::<16>();
        // SAFETY: NEON is part of the aarch64 baseline; each load reads
        // four f32 inside a 16-element chunk.
        let mut best = unsafe {
            let low = vdupq_n_f32(f32::NEG_INFINITY);
            let (mut a, mut b, mut c, mut d) = (low, low, low, low);
            for chunk in chunks {
                let p = chunk.as_ptr();
                a = vmaxnmq_f32(a, vld1q_f32(p));
                b = vmaxnmq_f32(b, vld1q_f32(p.add(4)));
                c = vmaxnmq_f32(c, vld1q_f32(p.add(8)));
                d = vmaxnmq_f32(d, vld1q_f32(p.add(12)));
            }
            vmaxnmvq_f32(vmaxnmq_f32(vmaxnmq_f32(a, b), vmaxnmq_f32(c, d)))
        };
        for &v in rest {
            if v > best {
                best = v;
            }
        }
        best
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut best = f32::NEG_INFINITY;
        for &v in row {
            if v > best {
                best = v;
            }
        }
        best
    }
}

pub fn penalizes(p: &SamplingParams) -> bool {
    let repeat = if p.repeat_penalty == 0.0 { 1.0 } else { p.repeat_penalty };
    repeat != 1.0 || p.presence_penalty != 0.0 || p.freq_penalty != 0.0
}

pub fn apply_penalties(row: &mut [f32], recent: &[u32], p: &SamplingParams, exempt: &std::collections::HashSet<u32>) {
    if !penalizes(p) {
        return;
    }
    let repeat = if p.repeat_penalty == 0.0 { 1.0 } else { p.repeat_penalty };
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &t in recent {
        if !exempt.contains(&t) {
            *counts.entry(t).or_insert(0) += 1;
        }
    }
    for (t, n) in counts {
        let Some(l) = row.get_mut(t as usize) else { continue };
        if repeat != 1.0 {
            *l = if *l > 0.0 { *l / repeat } else { *l * repeat };
        }
        *l -= p.presence_penalty + p.freq_penalty * n as f32;
    }
}

pub fn apply_logit_bias(row: &mut [f32], bias: &[(u32, f32)]) {
    for &(t, b) in bias {
        if let Some(l) = row.get_mut(t as usize) {
            *l += b;
        }
    }
}

pub fn uniform(position: u64) -> f64 {
    let mut z = position.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn by_prob_desc(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
}

fn sort_head(cands: &mut [(u32, f32)], head: usize) {
    if head < cands.len() {
        cands.select_nth_unstable_by(head - 1, by_prob_desc);
    }
    cands[..head].sort_unstable_by(by_prob_desc);
}

const INITIAL_HEAD: usize = 128;

fn grow(head: usize, n: usize) -> usize {
    head.saturating_mul(8).clamp(head + 1, n)
}

const DEAD_BELOW: f32 = -36.736_8;

#[inline(always)]
fn exp_neg(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let live = x > DEAD_BELOW;
    let x = if live { x } else { 0.0 };
    let n = (x * LOG2E - 0.5) as i32;
    let nf = n as f32;
    let r = (x - nf * LN2_HI) - nf * LN2_LO;
    let poly = ((((1.987_569_1e-4 * r + 1.398_2e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 1.666_666_5e-1) * r + 0.5;
    let y = poly * (r * r) + r + 1.0;
    let scale = f32::from_bits(((n + 127) as u32) << 23);
    if live {
        y * scale
    } else {
        0.0
    }
}

fn softmax_weights(row: &mut [f32], inv_t: f32) -> f32 {
    for l in row.iter_mut() {
        *l *= inv_t;
    }
    let max = max_of(row);
    let mut acc = [0.0f32; 16];
    let (chunks, rest) = row.as_chunks_mut::<16>();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline; every load and store
    // is four f32 inside a 16-element chunk or the 16-element `acc`.
    unsafe {
        use std::arch::aarch64::*;
        let sums = acc.as_mut_ptr();
        let top = vdupq_n_f32(max);
        for chunk in chunks.iter_mut() {
            let p = chunk.as_mut_ptr();
            let x = [
                vsubq_f32(vld1q_f32(p), top),
                vsubq_f32(vld1q_f32(p.add(4)), top),
                vsubq_f32(vld1q_f32(p.add(8)), top),
                vsubq_f32(vld1q_f32(p.add(12)), top),
            ];
            if vmaxnmvq_f32(vmaxnmq_f32(vmaxnmq_f32(x[0], x[1]), vmaxnmq_f32(x[2], x[3]))) <= DEAD_BELOW {
                for (q, _) in x.iter().enumerate() {
                    vst1q_f32(p.add(4 * q), vdupq_n_f32(0.0));
                }
                continue;
            }
            for (q, xq) in x.iter().enumerate() {
                let e = exp_neg_x4(*xq);
                vst1q_f32(p.add(4 * q), e);
                vst1q_f32(sums.add(4 * q), vaddq_f32(vld1q_f32(sums.add(4 * q)), e));
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for chunk in chunks.iter_mut() {
        for j in 0..16 {
            let e = exp_neg(chunk[j] - max);
            chunk[j] = e;
            acc[j] += e;
        }
    }
    let quarter = |q: usize| (acc[q] + acc[q + 4]) + (acc[q + 8] + acc[q + 12]);
    let mut sum = (quarter(0) + quarter(1)) + (quarter(2) + quarter(3));
    for l in rest {
        let e = exp_neg(*l - max);
        *l = e;
        sum += e;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn exp_neg_x4(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let zero = vdupq_n_f32(0.0);
    let live = vcgtq_f32(x, vdupq_n_f32(DEAD_BELOW));
    let x = vbslq_f32(live, x, zero);
    let n = vcvtq_s32_f32(vsubq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)), vdupq_n_f32(0.5)));
    let nf = vcvtq_f32_s32(n);
    let r = vsubq_f32(vsubq_f32(x, vmulq_f32(nf, vdupq_n_f32(0.693_359_4))), vmulq_f32(nf, vdupq_n_f32(-2.121_944_4e-4)));
    let mut poly = vaddq_f32(vmulq_f32(vdupq_n_f32(1.987_569_1e-4), r), vdupq_n_f32(1.398_2e-3));
    poly = vaddq_f32(vmulq_f32(poly, r), vdupq_n_f32(8.333_452e-3));
    poly = vaddq_f32(vmulq_f32(poly, r), vdupq_n_f32(4.166_579_6e-2));
    poly = vaddq_f32(vmulq_f32(poly, r), vdupq_n_f32(1.666_666_5e-1));
    poly = vaddq_f32(vmulq_f32(poly, r), vdupq_n_f32(0.5));
    let y = vaddq_f32(vaddq_f32(vmulq_f32(poly, vmulq_f32(r, r)), r), vdupq_n_f32(1.0));
    let scale = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n, vdupq_n_s32(127))));
    vbslq_f32(live, vmulq_f32(y, scale), zero)
}

fn top_head(probs: &[f32], head: usize, out: &mut Vec<(u32, f32)>) {
    const BLOCK: usize = 16;
    out.clear();
    let n = probs.len();
    if head >= n / 4 {
        out.extend(probs.iter().enumerate().map(|(i, &p)| (i as u32, p)));
        sort_head(out, head);
        return;
    }
    let mut floor = f32::NEG_INFINITY;
    for (b, block) in probs.chunks(BLOCK).enumerate() {
        if max_of(block) <= floor {
            continue;
        }
        for (j, &p) in block.iter().enumerate() {
            if p > floor {
                out.push(((b * BLOCK + j) as u32, p));
                if out.len() == 2 * head {
                    out.select_nth_unstable_by(head - 1, by_prob_desc);
                    out.truncate(head);
                    floor = out[head - 1].1;
                }
            }
        }
    }
    sort_head(out, head);
}

pub fn sample(row: &mut [f32], p: &SamplingParams, u: f64) -> u32 {
    if p.temperature <= 0.0 {
        return argmax(row);
    }
    if row.is_empty() {
        return 0;
    }
    let sum = softmax_weights(row, 1.0 / p.temperature);
    if sum > 0.0 {
        for w in row.iter_mut() {
            *w /= sum;
        }
    }
    let full_total: f32 = if sum > 0.0 { 1.0 } else { 0.0 };
    let n = row.len();
    let mut cands: Vec<(u32, f32)> = Vec::new();
    let k_cap = if p.top_k == 0 { n } else { (p.top_k as usize).clamp(1, n) };
    let mut head = if p.top_k == 0 { INITIAL_HEAD.min(n) } else { k_cap };
    loop {
        top_head(row, head, &mut cands);
        let front = &cands[..head];
        let mut keep = head.min(k_cap);
        let mut cut = p.top_k != 0;
        if p.min_p > 0.0 {
            let floor = p.min_p * front[0].1;
            let c = front[..keep].iter().take_while(|c| c.1 >= floor).count();
            if c < keep {
                keep = c.max(1);
                cut = true;
            } else if !cut && head < n {
                head = grow(head, n);
                continue;
            }
        }
        if p.top_p < 1.0 {
            let mut acc = 0.0f32;
            let mut found = None;
            for (i, c) in front[..keep].iter().enumerate() {
                acc += c.1;
                if acc >= p.top_p {
                    found = Some(i + 1);
                    break;
                }
            }
            match found {
                Some(k) => {
                    keep = k.max(1);
                    cut = true;
                }
                None if !cut && head < n => {
                    head = grow(head, n);
                    continue;
                }
                None => {}
            }
        }
        let total: f32 = if cut { front[..keep].iter().map(|c| c.1).sum() } else { full_total };
        let mut target = u as f32 * total;
        let mut last_with_mass = front[0].0;
        for c in &front[..keep] {
            if target < c.1 {
                return c.0;
            }
            target -= c.1;
            if c.1 > 0.0 {
                last_with_mass = c.0;
            }
        }
        if cut || head == n {
            return last_with_mass;
        }
        head = grow(head, n);
    }
}

pub fn logprobs(row: &[f32], chosen: u32, k: usize) -> (f32, Vec<(u32, f32)>) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + row.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    let lp = row.get(chosen as usize).map(|&l| l - lse).unwrap_or(f32::NEG_INFINITY);
    let k = k.min(row.len());
    if k == 0 {
        return (lp, Vec::new());
    }
    let mut top: Vec<(u32, f32)> = row.iter().enumerate().map(|(i, &l)| (i as u32, l - lse)).collect();
    sort_head(&mut top, k);
    top.truncate(k);
    (lp, top)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_a_pure_function_of_position() {
        assert_eq!(uniform(7), uniform(7));
        assert_ne!(uniform(7), uniform(8));
        assert!((0.0..1.0).contains(&uniform(u64::MAX)));
    }

    #[test]
    fn greedy_is_lowest_index_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[]), 0);
        let plain = |row: &[f32]| -> u32 {
            let mut best = 0usize;
            for (i, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = i;
                }
            }
            best as u32
        };
        let mut st = 0xA11CE_u64;
        for n in 1..40usize {
            for case in 0..40 {
                let mut row: Vec<f32> = (0..n).map(|_| (lcg(&mut st) * 6.0).floor()).collect();
                match case % 5 {
                    1 => row[n / 2] = f32::NAN,
                    2 => row[0] = f32::NAN,
                    3 => row[n - 1] = f32::INFINITY,
                    4 => row.iter_mut().for_each(|v| *v = f32::NEG_INFINITY),
                    _ => {}
                }
                assert_eq!(argmax(&row), plain(&row), "{row:?}");
            }
        }
    }

    #[test]
    fn top_k_one_is_greedy_at_any_temperature() {
        let p = SamplingParams { temperature: 1.0, top_k: 1, top_p: 1.0, ..Default::default() };
        let mut row = vec![0.1, 0.5, 0.2];
        assert_eq!(sample(&mut row, &p, 0.99), 1);
    }

    #[test]
    fn top_p_cuts_on_the_original_mass_after_top_k() {
        let row: Vec<f32> = [0.4f32, 0.3, 0.3].iter().map(|p| p.ln()).collect();
        let p = SamplingParams { temperature: 1.0, top_k: 2, top_p: 0.5, ..Default::default() };
        assert_eq!(sample(&mut row.clone(), &p, 0.95), 1, "the second candidate survives the cut");
        assert_eq!(sample(&mut row.clone(), &p, 0.1), 0);
        let p1 = SamplingParams { temperature: 1.0, top_k: 2, top_p: 0.39, ..Default::default() };
        assert_eq!(sample(&mut row.clone(), &p1, 0.95), 0, "a threshold under the best probability keeps only it");
    }

    #[test]
    fn a_draw_at_the_top_of_the_range_stays_on_a_token_with_mass() {
        let top = 1.0 - f64::EPSILON / 2.0;
        assert_eq!(top as f32, 1.0, "the draw this is about");
        let mut row = vec![f32::NEG_INFINITY; 64];
        (row[5], row[9]) = (1.0, 1.0);
        let p = SamplingParams { temperature: 1.0, top_k: 40, top_p: 1.0, ..Default::default() };
        assert_eq!(sample(&mut row.clone(), &p, top), 9, "the last token with any mass");
        let p = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0, ..Default::default() };
        assert_eq!(sample(&mut row.clone(), &p, top), 9);
        assert_eq!(sample(&mut row.clone(), &p, 0.0), 5);
    }

    #[test]
    fn top_p_zero_keeps_the_top_token() {
        let row: Vec<f32> = [0.4f32, 0.3, 0.3].iter().map(|p| p.ln()).collect();
        for top_p in [0.0, -1.0] {
            let p = SamplingParams { temperature: 1.0, top_p, ..Default::default() };
            for u in [0.0, 0.5, 0.99] {
                assert_eq!(sample(&mut row.clone(), &p, u), 0, "top_p {top_p}, draw {u}");
            }
        }
        let off = SamplingParams { temperature: 1.0, top_p: 1.0, ..Default::default() };
        assert_eq!(sample(&mut row.clone(), &off, 0.99), 2, "1.0 is the whole distribution");
    }

    #[test]
    fn repeat_penalty_demotes_a_seen_token() {
        let p = SamplingParams { repeat_penalty: 100.0, ..Default::default() };
        let mut row = vec![0.5, 10.0, 0.7];
        apply_penalties(&mut row, &[1], &p, &Default::default());
        assert_eq!(argmax(&row), 2);
    }

    fn reference_sample(row: &[f32], p: &SamplingParams, u: f64) -> u32 {
        let mut weights = row.to_vec();
        let sum = softmax_weights(&mut weights, 1.0 / p.temperature);
        let mut cands: Vec<(u32, f32)> = weights.iter().enumerate().map(|(i, &w)| (i as u32, w / sum)).collect();
        let full_total: f32 = 1.0;
        cands.sort_by(by_prob_desc);
        let mut bounded = false;
        if p.top_k != 0 {
            cands.truncate((p.top_k as usize).clamp(1, cands.len()));
            bounded = true;
        }
        if p.min_p > 0.0 {
            let floor = p.min_p * cands[0].1;
            let keep = cands.iter().take_while(|c| c.1 >= floor).count().max(1);
            if keep < cands.len() {
                bounded = true;
            }
            cands.truncate(keep);
        }
        if p.top_p < 1.0 {
            let mut acc = 0.0f32;
            let mut keep = 0usize;
            for c in &cands {
                acc += c.1;
                keep += 1;
                if acc >= p.top_p {
                    bounded = true;
                    break;
                }
            }
            cands.truncate(keep.max(1));
        }
        let total: f32 = if bounded { cands.iter().map(|c| c.1).sum() } else { full_total };
        let mut target = u as f32 * total;
        for c in &cands {
            if target < c.1 {
                return c.0;
            }
            target -= c.1;
        }
        cands.last().map(|c| c.0).unwrap_or(0)
    }

    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 40) as f32) / ((1u64 << 24) as f32)
    }

    #[test]
    fn head_sorting_matches_the_full_sort_reference() {
        let n = 3001usize;
        let mut st = 0x5EED_u64;
        for shape in 0..3 {
            let mut row: Vec<f32> = (0..n).map(|_| lcg(&mut st) * [12.0, 3.0, 0.02][shape]).collect();
            row[17] += [6.0, 1.0, 0.0][shape];
            for &top_k in &[0u32, 1, 7, 50, 5000] {
                for &top_p in &[1.0f32, 0.9, 0.3] {
                    for &min_p in &[0.0f32, 0.1] {
                        for &temperature in &[0.7f32, 1.5] {
                            let p = SamplingParams { temperature, top_k, top_p, min_p, ..Default::default() };
                            let mut us = vec![0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 0.999_999];
                            us.extend((0..8).map(|_| lcg(&mut st) as f64));
                            for u in us {
                                let want = reference_sample(&row, &p, u);
                                let got = sample(&mut row.clone(), &p, u);
                                assert_eq!(got, want, "shape {shape} top_k {top_k} top_p {top_p} min_p {min_p} t {temperature} u {u}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exp_neg_is_the_exponential_to_two_ulp() {
        let mut worst = 0.0f64;
        let mut x = 0.0f32;
        while x > DEAD_BELOW {
            let (got, want) = (exp_neg(x) as f64, (x as f64).exp());
            worst = worst.max(((got - want) / want).abs());
            x -= 0.000_37;
        }
        assert!(worst < 2.5e-7, "relative error {worst:e}");
        assert_eq!(exp_neg(0.0), 1.0);
        for dead in [DEAD_BELOW, -37.0, -1000.0, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(exp_neg(dead), 0.0, "{dead}");
        }
    }

    #[test]
    fn the_row_pass_is_exp_neg_element_for_element() {
        let mut st = 0xE4B_u64;
        for n in [1usize, 15, 16, 17, 31, 32, 33, 100, 1000] {
            let mut row: Vec<f32> = (0..n).map(|_| lcg(&mut st) * 120.0 - 60.0).collect();
            if n > 20 {
                (row[3], row[18]) = (f32::NAN, f32::NEG_INFINITY);
            }
            let inv_t = 1.3f32;
            let scaled: Vec<f32> = row.iter().map(|l| l * inv_t).collect();
            let max = scaled.iter().copied().filter(|v| !v.is_nan()).fold(f32::NEG_INFINITY, f32::max);
            let want: Vec<f32> = scaled.iter().map(|l| exp_neg(l - max)).collect();
            let mut acc = [0.0f32; 16];
            let whole = n / 16 * 16;
            for (i, e) in want[..whole].iter().enumerate() {
                acc[i % 16] += e;
            }
            let quarter = |q: usize| (acc[q] + acc[q + 4]) + (acc[q + 8] + acc[q + 12]);
            let mut want_sum = (quarter(0) + quarter(1)) + (quarter(2) + quarter(3));
            for e in &want[whole..] {
                want_sum += e;
            }
            let sum = softmax_weights(&mut row, inv_t);
            assert_eq!(row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "n {n}");
            assert_eq!(sum.to_bits(), want_sum.to_bits(), "n {n}");
        }
    }

    #[test]
    fn top_head_is_the_head_of_the_sorted_row() {
        let mut st = 0xBEEF_u64;
        for &n in &[1usize, 15, 16, 17, 400, 5000] {
            for &levels in &[2.0f32, 37.0, 1.0e6] {
                let row: Vec<f32> = (0..n).map(|_| (lcg(&mut st) * levels).floor() / levels).collect();
                let mut whole: Vec<(u32, f32)> = row.iter().enumerate().map(|(i, &p)| (i as u32, p)).collect();
                whole.sort_by(by_prob_desc);
                for &head in &[1usize, 2, 7, 128, 1000] {
                    let head = head.min(n);
                    let mut got = Vec::new();
                    top_head(&row, head, &mut got);
                    assert_eq!(&got[..head], &whole[..head], "n {n} levels {levels} head {head}");
                }
            }
        }
    }

    #[test]
    fn a_row_with_no_mass_answers_the_first_token() {
        let p = SamplingParams { temperature: 0.8, top_k: 20, top_p: 0.9, ..Default::default() };
        for u in [0.0, 0.5, 0.999] {
            assert_eq!(sample(&mut vec![f32::NEG_INFINITY; 500], &p, u), 0);
            assert_eq!(sample(&mut vec![f32::NAN; 500], &p, u), 0);
        }
    }

    #[test]
    #[ignore]
    fn sampler_cost() {
        let n = 151_936usize;
        let mut st = 0xC057_u64;
        let rows: Vec<Vec<f32>> = (0..3)
            .map(|shape| {
                let mut row: Vec<f32> = (0..n).map(|_| lcg(&mut st) * [30.0, 12.0, 45.0][shape]).collect();
                row[4242] += 8.0;
                row
            })
            .collect();
        let time = |what: &str, f: &mut dyn FnMut(&[f32]) -> u32| {
            let mut sink = 0u32;
            let t = std::time::Instant::now();
            for i in 0..300 {
                sink ^= f(&rows[i % 3]);
            }
            eprintln!("{what}: {:.0} us ({sink})", t.elapsed().as_secs_f64() * 1e6 / 300.0);
        };
        let p = SamplingParams { temperature: 0.7, top_k: 20, top_p: 0.8, ..Default::default() };
        let open = SamplingParams { temperature: 1.0, top_k: 0, top_p: 0.95, ..Default::default() };
        time("copy the row", &mut |r| r.to_vec().len() as u32);
        time("argmax", &mut |r| argmax(r));
        time("sample top-k 20 top-p 0.8 (with the copy)", &mut |r| sample(&mut r.to_vec(), &p, 0.37));
        time("sample top-p 0.95, no top-k (with the copy)", &mut |r| sample(&mut r.to_vec(), &open, 0.37));
        time("reference: exp per element and a full sort (with the copy)", &mut |r| reference_sample(r, &p, 0.37));
        let mut w = rows[0].clone();
        let t = std::time::Instant::now();
        for _ in 0..300 {
            w.copy_from_slice(&rows[0]);
            std::hint::black_box(softmax_weights(&mut w, 1.0 / 0.7));
        }
        eprintln!("softmax_weights alone (with the copy): {:.0} us", t.elapsed().as_secs_f64() * 1e6 / 300.0);
        if let Some(path) = std::env::var_os("SUPERFLUID_SAMPLER_ROWS") {
            let bytes = std::fs::read(path).unwrap();
            let all: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_ne_bytes(*b)).collect();
            let real: Vec<&[f32]> = all.chunks_exact(n).collect();
            let mut w = vec![0.0f32; n];
            let mut cands = Vec::new();
            let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
            for i in 0..300 {
                w.copy_from_slice(real[i % real.len()]);
                let t = std::time::Instant::now();
                let sum = softmax_weights(&mut w, 1.0 / 0.7);
                a += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                for x in w.iter_mut() {
                    *x /= sum;
                }
                b += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                top_head(&w, 20, &mut cands);
                c += t.elapsed().as_secs_f64();
            }
            eprintln!("real rows: weights {:.0} us, division {:.0} us, head {:.0} us", a * 1e6 / 300.0, b * 1e6 / 300.0, c * 1e6 / 300.0);
        }
    }

    #[test]
    fn flat_row_draws_deep_in_the_id_order() {
        let row = vec![0.0f32; 4000];
        let p = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0, ..Default::default() };
        let got = sample(&mut row.clone(), &p, 0.9999);
        assert_eq!(got, reference_sample(&row, &p, 0.9999));
        assert!(got >= 3990, "equal probabilities walk the id order: {got}");
    }

    #[test]
    fn logprobs_top_k_is_the_sorted_head() {
        let row = vec![1.0f32, 5.0, 3.0, 5.0, 0.5];
        let (_, top) = logprobs(&row, 0, 3);
        assert_eq!(top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 3, 2]);
        assert!(top[0].1 > top[2].1);
    }

    #[test]
    fn logprobs_are_normalized() {
        let (lp, top) = logprobs(&[0.0, 0.0], 0, 2);
        assert!((lp - (0.5f32).ln()).abs() < 1e-6);
        assert_eq!(top.len(), 2);
    }
}
