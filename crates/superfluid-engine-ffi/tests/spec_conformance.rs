//! P2.4 speculation conformance harness against the REAL engine (native tick executor over Link W).

#![allow(clippy::too_many_arguments)]

mod common;
use common::*;

#[test]
fn prompt_lookup_registers_and_is_exact_on_and_off() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    std::env::set_var("BASERTD_NO_GPU_SAMPLE", "1");
    let (mut client, worker) = spawn_native(path);
    let mut seq = 1u64;

    let mut bad = prompt_lookup_reg();
    bad.strategy_id = "eagle3".into();
    assert!(
        client.register_strategy(bad).is_err(),
        "unknown strategy refused"
    );
    let mut bad = prompt_lookup_reg();
    bad.capabilities.push(linkw::CapabilityReqMsg {
        kind_id: 999,
        params: vec![],
    });
    assert!(
        client.register_strategy(bad).is_err(),
        "unknown capability kind refused"
    );
    let mut bad = prompt_lookup_reg();
    bad.kernel_caps_required = 1;
    assert!(
        client.register_strategy(bad).is_err(),
        "uncheckable kernel caps refused"
    );
    let mut bad = prompt_lookup_reg();
    bad.target_archs = vec!["not-this-arch".into()];
    assert!(
        client.register_strategy(bad).is_err(),
        "arch mismatch refused"
    );

    let grant = client
        .register_strategy(prompt_lookup_reg())
        .expect("prompt-lookup registers");
    assert_eq!(grant.strategy_slot, 1);
    assert_eq!(grant.certificates.len(), 1);
    let cert = &grant.certificates[0];
    assert_eq!(cert.exactness, superfluid_abi::exactness::DISTRIBUTION_EXACT);
    assert_eq!(
        cert.sampling_modes,
        superfluid_abi::cert_mode::GREEDY | superfluid_abi::cert_mode::GUMBEL
    );
    assert!(!cert.grammar_allowed);

    let pref = client.stage_prompt(&[1, 2, 3, 4]).unwrap();
    let mut p = plan(seq);
    seq += 1;
    let mut a = admit(77, pref, false, 0, 1);
    a.sampling = 2;
    p.admits.push(a);
    p.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 77,
        token_offset: 0,
        token_count: 4,
    });
    match client.tick(p).unwrap_err() {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -110),
        other => panic!("expected typed rejection, got {other:?}"),
    }
    let mut p = plan(seq);
    seq += 1;
    let mut a = admit(78, pref, false, 0, 1);
    a.minimum_exactness = superfluid_abi::exactness::SEED_PATH_INVARIANT;
    p.admits.push(a);
    p.prefills.push(linkw::LanePrefillMsg {
        lane_tag: 78,
        token_offset: 0,
        token_count: 4,
    });
    match client.tick(p).unwrap_err() {
        superfluid_agent::AgentError::Rejected(status) => assert_eq!(status, -140),
        other => panic!("expected CERT_UNMATCHED, got {other:?}"),
    }

    let prompt = repetitive_prompt();
    let (off, p0, _) = generate(&mut client, &mut seq, 1, &prompt, 48, false, 0, 0);
    assert_eq!(p0, 0, "no strategy: nothing proposed");
    let (on, proposed, accepted) = generate(&mut client, &mut seq, 2, &prompt, 48, false, 0, 1);
    assert_greedy_matches("prompt-lookup", &on, &off);
    assert!(
        proposed > 0,
        "prompt-lookup proposed drafts on a self-similar stream"
    );
    assert!(
        accepted > 0,
        "and some were accepted ({accepted}/{proposed})"
    );
    eprintln!(
        "greedy: proposed {proposed} accepted {accepted} over {} tokens",
        on.len()
    );

    let (soff, _, _) = generate(&mut client, &mut seq, 3, &prompt, 48, true, 4242, 0);
    let (son, sp, sa) = generate(&mut client, &mut seq, 4, &prompt, 48, true, 4242, 1);
    let common = soff
        .iter()
        .zip(son.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "sampled: proposed {sp} accepted {sa}; identical prefix {common}/{}",
        soff.len().min(son.len())
    );
    assert!(
        common >= 8,
        "sampled runs share at least the early positions ({common})"
    );
    let free: Vec<u32> = (0..40).map(|i| 3000 + i * 5).collect();
    let (a_off, _, _) = generate(&mut client, &mut seq, 5, &free, 24, true, 4242, 0);
    let (a_on, _, _) = generate(&mut client, &mut seq, 6, &free, 24, true, 4242, 1);
    let (b_on, _, _) = generate(&mut client, &mut seq, 7, &free, 24, true, 99, 1);
    assert_ne!(a_on, b_on, "the seed is live under speculation");
    let common = a_off
        .iter()
        .zip(a_on.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!("free-form sampled: off/on identical prefix {common}/24");
    assert!(common >= 4);

    drop(client);
    worker.join().unwrap();
}

#[test]
fn draft_model_registers_and_is_exact() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let (mut client, worker) = spawn_native(path.clone());
    let mut seq = 1u64;

    let mut reg = prompt_lookup_reg();
    reg.strategy_id = "draft-model".into();
    reg.impl_hash = [0x33; 32];
    reg.artifacts = vec![linkw::ArtifactMsg {
        role: superfluid_abi::artifact_role::DRAFT_WEIGHTS,
        content_hash: [0u8; 32],
        byte_size: 0,
        load_path: path.to_string_lossy().into_owned(),
    }];
    let mut bad = reg.clone();
    bad.artifacts = vec![];
    assert!(
        client.register_strategy(bad).is_err(),
        "draft-model needs a draft artifact"
    );

    if bundle_is_recurrent(&path) {
        assert!(
            client.register_strategy(reg).is_err(),
            "a recurrent-state draft model must be refused"
        );
        eprintln!("SKIP: self-draft on a recurrent-state bundle (draft refused by design)");
        drop(client);
        worker.join().unwrap();
        return;
    }

    let grant = client
        .register_strategy(reg)
        .expect("draft-model registers");
    assert_eq!(grant.strategy_slot, 1);
    let cert = &grant.certificates[0];
    assert_eq!(cert.exactness, superfluid_abi::exactness::DISTRIBUTION_EXACT);

    let prompt: Vec<u32> = (0..40).map(|i| 3000 + i * 5).collect();
    let (off, p0, _) = generate(&mut client, &mut seq, 1, &prompt, 40, false, 0, 0);
    assert_eq!(p0, 0);
    let (on, proposed, accepted) = generate(&mut client, &mut seq, 2, &prompt, 40, false, 0, 1);
    assert_greedy_matches("draft-model", &on, &off);
    assert!(proposed > 0, "the draft model proposed");
    if on.len() >= 40 {
        assert_eq!(
            accepted, proposed,
            "self-draft greedy: every proposal accepted"
        );
    } else {
        assert!(
            accepted > 0,
            "self-draft greedy: some proposals accepted before EOS"
        );
        assert!(
            accepted + 4 >= proposed,
            "self-draft greedy: only the finishing round's tail may go unaccepted ({accepted}/{proposed})"
        );
    }
    eprintln!(
        "draft-model self-draft: proposed {proposed} accepted {accepted} over {} tokens",
        on.len()
    );

    drop(client);
    worker.join().unwrap();
}

#[test]
fn mtp_head_registers_and_is_exact() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    if !bundle_has_speculator(&path) {
        eprintln!("SKIP: bundle has no speculator section");
        return;
    }
    std::env::set_var("BASERTD_NO_GPU_SAMPLE", "1");
    let (mut client, worker) = spawn_native(path);
    let mut seq = 1u64;

    let mut reg = prompt_lookup_reg();
    reg.strategy_id = "mtp-head".into();
    reg.impl_hash = [0x44; 32];
    let mut bad = reg.clone();
    bad.artifacts = vec![linkw::ArtifactMsg {
        role: superfluid_abi::artifact_role::DRAFT_WEIGHTS,
        content_hash: [0u8; 32],
        byte_size: 0,
        load_path: "/nonexistent".into(),
    }];
    assert!(
        client.register_strategy(bad).is_err(),
        "mtp-head takes no artifact"
    );
    let grant = client.register_strategy(reg).expect("mtp-head registers");
    assert_eq!(grant.strategy_slot, 1);
    assert_eq!(
        grant.certificates[0].exactness,
        superfluid_abi::exactness::DISTRIBUTION_EXACT
    );

    let prompt: Vec<u32> = match std::env::var("BASERT_TEST_PROMPT_IDS") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => repetitive_prompt(),
    };
    let (off, p0, _) = generate(&mut client, &mut seq, 1, &prompt, 48, false, 0, 0);
    assert_eq!(p0, 0, "no strategy: nothing proposed");
    let (on, proposed, accepted) = generate(&mut client, &mut seq, 2, &prompt, 48, false, 0, 1);
    assert_greedy_matches("mtp-head", &on, &off);
    assert!(proposed > 0, "the head proposed drafts");
    assert!(
        accepted > 0,
        "and some were accepted ({accepted}/{proposed})"
    );
    eprintln!(
        "mtp-head greedy: proposed {proposed} accepted {accepted} over {} tokens",
        on.len()
    );

    let (soff, _, _) = generate(&mut client, &mut seq, 3, &prompt, 48, true, 4242, 0);
    let (son, sp, sa) = generate(&mut client, &mut seq, 4, &prompt, 48, true, 4242, 1);
    let common = soff
        .iter()
        .zip(son.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "mtp-head sampled: proposed {sp} accepted {sa}; identical prefix {common}/{}",
        soff.len().min(son.len())
    );
    assert!(
        common >= 8,
        "sampled runs share at least the early positions ({common})"
    );

    drop(client);
    worker.join().unwrap();
}

#[test]
fn dflash_registers_and_is_exact() {
    block_drafter_exact("dflash", "BASERT_TEST_DRAFTER", 0x55);
}

#[test]
fn dspark_registers_and_is_exact() {
    if let (Some(path), Ok(dflash)) = (model_path(), std::env::var("BASERT_TEST_DRAFTER")) {
        let _guard = MODEL_LOCK.lock().unwrap();
        let (mut client, worker) = spawn_native(path);
        let mut reg = prompt_lookup_reg();
        reg.strategy_id = "dspark".into();
        reg.impl_hash = [0x66; 32];
        reg.artifacts = vec![linkw::ArtifactMsg {
            role: superfluid_abi::artifact_role::DRAFT_WEIGHTS,
            content_hash: [0u8; 32],
            byte_size: 0,
            load_path: dflash,
        }];
        assert!(
            client.register_strategy(reg).is_err(),
            "dspark needs a Markov head"
        );
        drop(client);
        worker.join().unwrap();
    }
    block_drafter_exact("dspark", "BASERT_TEST_DSPARK", 0x66);
}

#[test]
fn eagle3_registers_and_is_exact() {
    block_drafter_exact("eagle3", "BASERT_TEST_EAGLE3", 0x77);
}

fn block_drafter_exact(id: &str, env: &str, hash: u8) {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no test model");
        return;
    };
    let Ok(drafter) = std::env::var(env) else {
        eprintln!("SKIP: {env} (a converted block-drafter sidecar) not set");
        return;
    };
    std::env::set_var("BASERTD_NO_GPU_SAMPLE", "1");
    let (mut client, worker) = spawn_native(path);
    let mut seq = 1u64;

    let mut reg = prompt_lookup_reg();
    reg.strategy_id = id.into();
    reg.impl_hash = [hash; 32];
    reg.artifacts = vec![linkw::ArtifactMsg {
        role: superfluid_abi::artifact_role::DRAFT_WEIGHTS,
        content_hash: [0u8; 32],
        byte_size: 0,
        load_path: drafter.clone(),
    }];
    let mut bad = reg.clone();
    bad.artifacts = vec![];
    assert!(
        client.register_strategy(bad).is_err(),
        "{id} needs the sidecar artifact"
    );
    let mut bad = reg.clone();
    bad.artifacts[0].load_path = "/nonexistent-drafter.base".into();
    assert!(
        client.register_strategy(bad).is_err(),
        "a missing sidecar is refused"
    );
    let grant = client
        .register_strategy(reg)
        .unwrap_or_else(|e| panic!("{id} registers: {e:?}"));
    assert_eq!(grant.strategy_slot, 1);
    assert_eq!(
        grant.certificates[0].exactness,
        superfluid_abi::exactness::DISTRIBUTION_EXACT
    );

    let prompt: Vec<u32> = match std::env::var("BASERT_TEST_PROMPT_IDS") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => repetitive_prompt(),
    };
    let n = 64;
    let (off, p0, _) = generate(&mut client, &mut seq, 1, &prompt, n, false, 0, 0);
    assert_eq!(p0, 0, "no strategy: nothing proposed");
    let (on, proposed, accepted) = generate(&mut client, &mut seq, 2, &prompt, n, false, 0, 1);
    assert_greedy_matches(id, &on, &off);
    assert!(proposed > 0, "the drafter proposed drafts");
    assert!(
        accepted > 0,
        "and some were accepted ({accepted}/{proposed})"
    );
    eprintln!(
        "{id} greedy: proposed {proposed} accepted {accepted} over {} tokens",
        on.len()
    );

    let (soff, _, _) = generate(&mut client, &mut seq, 3, &prompt, n, true, 4242, 0);
    let (son, sp, sa) = generate(&mut client, &mut seq, 4, &prompt, n, true, 4242, 1);
    let common = soff
        .iter()
        .zip(son.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "{id} sampled: proposed {sp} accepted {sa}; identical prefix {common}/{}",
        soff.len().min(son.len())
    );
    if common < 8 {
        eprintln!(
            "{id} sampled off[..8]={:?} on[..8]={:?} (prompt {} tokens)",
            &soff[..soff.len().min(8)],
            &son[..son.len().min(8)],
            prompt.len()
        );
    }
    assert!(
        common >= 8,
        "sampled runs share at least the early positions ({common})"
    );

    drop(client);
    worker.join().unwrap();
}

#[test]
fn speculation_timing_probe() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let (Some(path), Ok(id)) = (model_path(), std::env::var("BASERT_TEST_TIMING")) else {
        eprintln!("SKIP: BASERT_TEST_TIMING=<strategy> not set");
        return;
    };
    let lanes: u64 = std::env::var("BASERT_TEST_TIMING_LANES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let n: u32 = std::env::var("BASERT_TEST_TIMING_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let (mut client, worker) = spawn_native(path);
    let mut seq = 1u64;
    let mut reg = prompt_lookup_reg();
    reg.strategy_id = id.clone();
    reg.impl_hash = [0x99; 32];
    if let Ok(art) = std::env::var("BASERT_TEST_TIMING_ARTIFACT") {
        reg.artifacts = vec![linkw::ArtifactMsg {
            role: superfluid_abi::artifact_role::DRAFT_WEIGHTS,
            content_hash: [0u8; 32],
            byte_size: 0,
            load_path: art,
        }];
    }
    let grant = client
        .register_strategy(reg)
        .unwrap_or_else(|e| panic!("{id} registers: {e:?}"));
    let prompt: Vec<u32> = match std::env::var("BASERT_TEST_PROMPT_IDS") {
        Ok(v) => v.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        Err(_) => repetitive_prompt(),
    };
    let _ = generate(&mut client, &mut seq, 900, &prompt, 16, false, 0, 0);
    let _ = generate(
        &mut client,
        &mut seq,
        901,
        &prompt,
        16,
        false,
        0,
        grant.strategy_slot,
    );
    for &b in &[1u64, lanes] {
        for &slot in &[0u32, grant.strategy_slot] {
            let t0 = std::time::Instant::now();
            let (toks, proposed, accepted) = generate_many(
                &mut client,
                &mut seq,
                1000 + b * 10 + slot as u64,
                b,
                &prompt,
                n,
                slot,
            );
            let dt = t0.elapsed().as_secs_f64();
            let total: usize = toks.iter().map(|t| t.len()).sum();
            eprintln!(
                "timing {id}: lanes={b} spec={} tokens={total} {:.2}s → {:.1} tok/s ({:.1} per lane); proposed {proposed} accepted {accepted}",
                if slot == 0 { "off" } else { "on " },
                dt,
                total as f64 / dt,
                total as f64 / dt / b as f64
            );
        }
    }
    drop(client);
    worker.join().unwrap();
}

fn generate_many(
    client: &mut WorkerClient,
    seq: &mut u64,
    tag0: u64,
    b: u64,
    prompt: &[u32],
    n: u32,
    slot: u32,
) -> (Vec<Vec<u32>>, u32, u32) {
    let mut p = plan(*seq);
    *seq += 1;
    for i in 0..b {
        let pref = client.stage_prompt(prompt).unwrap();
        p.admits.push(admit(tag0 + i, pref, false, 0, slot));
        p.prefills.push(linkw::LanePrefillMsg {
            lane_tag: tag0 + i,
            token_offset: 0,
            token_count: prompt.len() as u32,
        });
    }
    let ev = client.tick(p).unwrap();
    for i in 0..b {
        let ar = ev
            .admit_results
            .iter()
            .find(|a| a.lane_tag == tag0 + i)
            .unwrap();
        assert_eq!(
            ar.status,
            superfluid_abi::admit_status::ADMITTED,
            "admit: {ar:?}"
        );
    }
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); b as usize];
    let mut done = vec![false; b as usize];
    let (mut proposed, mut accepted) = (0u32, 0u32);
    while done.iter().any(|d| !d) {
        let mut p = plan(*seq);
        *seq += 1;
        for i in 0..b as usize {
            if done[i] {
                continue;
            }
            let want = (n - out[i].len() as u32).min(32) as u16;
            p.decodes.push(linkw::LaneDecodeMsg {
                lane_tag: tag0 + i as u64,
                max_new_tokens: want,
                overshoot: 0,
        });
        }
        let ev = client.tick(p).unwrap();
        for i in 0..b as usize {
            if done[i] {
                continue;
            }
            if let Some(emit) = ev.emits.iter().find(|e| e.lane_tag == tag0 + i as u64) {
                let toks = client.read_tokens(&emit.token_ref).unwrap();
                out[i].extend(toks);
                proposed += emit.spec.proposed;
                accepted += emit.spec.accepted;
                if emit.finish != superfluid_abi::finish::NONE || out[i].len() as u32 >= n {
                    done[i] = true;
                }
            }
        }
    }
    let mut p = plan(*seq);
    *seq += 1;
    for i in 0..b {
        p.retires.push(linkw::LaneRetireMsg {
            lane_tag: tag0 + i,
            publish_to_cache: false,
        });
    }
    client.tick(p).unwrap();
    (out, proposed, accepted)
}
