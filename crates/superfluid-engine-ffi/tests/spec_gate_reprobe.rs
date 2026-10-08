//! A strategy the throughput gate abandoned (losing at every lane count it measured, twice over on
//! re-check) is suspended, not finished.

#![allow(clippy::too_many_arguments)]

mod common;
use common::*;

fn run(path: &std::path::Path, reprobe: &str, n_req: usize, tokens: u32) -> Vec<(Vec<u32>, u32)> {
    std::env::set_var("SUPERFLUID_SPEC_MIN_SPEEDUP", "100");
    std::env::set_var("SUPERFLUID_SPEC_MIN_YIELD", "0");
    std::env::set_var("SUPERFLUID_SPEC_GATE_REPROBE", reprobe);
    let (mut client, worker) = spawn_native(path.to_path_buf());
    let mut seq = 1u64;
    let mut reg = prompt_lookup_reg();
    reg.strategy_id = "mtp-head".into();
    reg.impl_hash = [0x45; 32];
    client.register_strategy(reg).expect("mtp-head registers");
    let prompt: Vec<u32> = match std::env::var("BASERT_TEST_PROMPT_IDS") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => repetitive_prompt(),
    };
    let mut out = Vec::new();
    for i in 0..n_req {
        let (toks, proposed, _) = generate(
            &mut client,
            &mut seq,
            10 + i as u64,
            &prompt,
            tokens,
            false,
            0,
            1,
        );
        out.push((toks, proposed));
    }
    drop(client);
    worker.join().unwrap();
    out
}

#[test]
fn abandoned_gate_reprobes_after_its_window_and_stays_exact() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    if !bundle_has_speculator(&path) {
        eprintln!("SKIP: bundle has no speculator section (the gate needs a tapped strategy)");
        return;
    }
    let off = {
        let (mut client, worker) = spawn_native(path.clone());
        let mut seq = 1u64;
        let prompt: Vec<u32> = match std::env::var("BASERT_TEST_PROMPT_IDS") {
            Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
            Err(_) => repetitive_prompt(),
        };
        let (t, _, _) = generate(&mut client, &mut seq, 1, &prompt, 640, false, 0, 0);
        drop(client);
        worker.join().unwrap();
        t
    };
    let fin = run(&path, "0", 5, 640);
    for (i, (t, p)) in fin.iter().enumerate() {
        eprintln!("final: request {i} proposed {p} over {} tokens", t.len());
        assert_greedy_matches(
            &format!("mtp-head (final abandonment, request {i})"),
            t,
            &off,
        );
    }
    assert!(
        fin[0].1 > 0,
        "the first request speculated before the verdict"
    );
    assert_eq!(
        fin[4].1, 0,
        "a final abandonment drafts nothing on later requests"
    );
    let re = run(&path, "1", 5, 640);
    for (i, (t, p)) in re.iter().enumerate() {
        eprintln!("reprobe: request {i} proposed {p} over {} tokens", t.len());
        assert_greedy_matches(
            &format!("mtp-head (re-probed abandonment, request {i})"),
            t,
            &off,
        );
    }
    assert!(
        re[3].1 > 0 || re[4].1 > 0,
        "the gate re-probed after its window: a later request drafted again (proposed {:?})",
        re.iter().map(|r| r.1).collect::<Vec<_>>()
    );
}

fn run_overlapping(
    path: &std::path::Path,
    reprobe: &str,
    a_ticks: usize,
    quiet: usize,
    every: usize,
    b_ticks: usize,
) -> (Vec<u32>, Vec<u32>, Vec<Vec<u32>>) {
    std::env::set_var("SUPERFLUID_SPEC_MIN_SPEEDUP", "100");
    std::env::set_var("SUPERFLUID_SPEC_MIN_YIELD", "0");
    std::env::set_var("SUPERFLUID_SPEC_GATE_REPROBE", reprobe);
    std::env::set_var("SUPERFLUID_SPEC_ADAPTIVE", "0");
    let (mut client, worker) = spawn_native(path.to_path_buf());
    let mut seq = 1u64;
    let mut reg = prompt_lookup_reg();
    reg.strategy_id = "mtp-head".into();
    reg.impl_hash = [0x45; 32];
    client.register_strategy(reg).expect("mtp-head registers");
    let prompt = repetitive_prompt();
    const A: u64 = 100;
    let admit_lane = |client: &mut WorkerClient, p: &mut linkw::TickPlanMsg, tag: u64| {
        let pref = client.stage_prompt(&prompt).unwrap();
        p.admits.push(admit(tag, pref, false, 0, 1));
        p.prefills.push(linkw::LanePrefillMsg {
            lane_tag: tag,
            token_offset: 0,
            token_count: prompt.len() as u32,
        });
    };
    let mut p = plan(seq);
    seq += 1;
    admit_lane(&mut client, &mut p, A);
    client.tick(p).unwrap();
    let mut a_out = Vec::new();
    let mut a_prop = Vec::new();
    let mut b_out: Vec<Vec<u32>> = Vec::new();
    let mut b: Option<(u64, usize)> = None;
    for t in 0..a_ticks {
        let mut p = plan(seq);
        seq += 1;
        let mut admitted_now = false;
        if b.is_none() && t >= quiet && (t - quiet) % every == every - 1 {
            let tag = 200 + b_out.len() as u64;
            admit_lane(&mut client, &mut p, tag);
            b_out.push(Vec::new());
            b = Some((tag, b_ticks));
            admitted_now = true;
        }
        if !admitted_now {
            p.decodes.push(linkw::LaneDecodeMsg {
                lane_tag: A,
                max_new_tokens: 32,
                overshoot: 0,
            });
        }
        if let (Some((tag, _)), false) = (b, admitted_now) {
            p.decodes.push(linkw::LaneDecodeMsg {
                lane_tag: tag,
                max_new_tokens: 32,
                overshoot: 0,
            });
        }
        let ev = client.tick(p).unwrap();
        let mut tick_prop = 0;
        for e in &ev.emits {
            let toks = client.read_tokens(&e.token_ref).unwrap();
            if e.lane_tag == A {
                a_out.extend(toks);
                tick_prop += e.spec.proposed;
                assert_eq!(e.finish, superfluid_abi::finish::NONE, "lane A finished early");
            } else {
                b_out.last_mut().unwrap().extend(toks);
            }
        }
        a_prop.push(tick_prop);
        if let Some((tag, left)) = b {
            if !admitted_now {
                if left <= 1 {
                    let mut p = plan(seq);
                    seq += 1;
                    p.retires.push(linkw::LaneRetireMsg {
                        lane_tag: tag,
                        publish_to_cache: false,
                    });
                    client.tick(p).unwrap();
                    b = None;
                } else {
                    b = Some((tag, left - 1));
                }
            }
        }
    }
    drop(client);
    worker.join().unwrap();
    std::env::remove_var("SUPERFLUID_SPEC_ADAPTIVE");
    (a_out, a_prop, b_out)
}

#[test]
fn a_lane_live_across_the_lift_drafts_again() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    if !bundle_has_speculator(&path) {
        eprintln!("SKIP: bundle has no speculator section (the gate needs a tapped strategy)");
        return;
    }
    const A_TICKS: usize = 96;
    let off = {
        let (mut client, worker) = spawn_native(path.clone());
        let mut seq = 1u64;
        let (t, _, _) = generate(
            &mut client,
            &mut seq,
            1,
            &repetitive_prompt(),
            32 * A_TICKS as u32,
            false,
            0,
            0,
        );
        drop(client);
        worker.join().unwrap();
        t
    };
    let late = |prop: &[u32]| prop[A_TICKS * 2 / 3..].iter().sum::<u32>();
    let (fa, fp, fb) = run_overlapping(&path, "0", A_TICKS, 48, 6, 3);
    eprintln!("final: lane A proposals per tick {fp:?}");
    assert_greedy_matches("mtp-head lane A (final)", &fa, &off);
    for (i, t) in fb.iter().enumerate() {
        assert_greedy_matches(&format!("mtp-head short lane {i} (final)"), t, &off);
    }
    assert!(fp.iter().sum::<u32>() > 0, "lane A speculated before the verdict");
    assert_eq!(late(&fp), 0, "a final abandonment: lane A drafts nothing late in its run");
    let (ra, rp, rb) = run_overlapping(&path, "1", A_TICKS, 48, 6, 3);
    eprintln!("reprobe: lane A proposals per tick {rp:?}");
    assert_greedy_matches("mtp-head lane A (re-probed)", &ra, &off);
    for (i, t) in rb.iter().enumerate() {
        assert_greedy_matches(&format!("mtp-head short lane {i} (re-probed)"), t, &off);
    }
    assert!(
        late(&rp) > 0,
        "lane A was live across the lift and drafts again on the re-checks after it"
    );
}
