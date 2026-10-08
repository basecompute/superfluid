//! A preempted lane's continuation is schedule-exact, on the REAL engine.

mod common;
use common::*;

use superfluid_engine_ffi::TokenizerHandle;

fn run(
    client: &mut WorkerClient,
    seq: &mut u64,
    lane: u64,
    prompt: &[u32],
    prefill_len: u32,
    decode_replay: u32,
    n: u16,
) -> Result<(Vec<u32>, Vec<u32>), i32> {
    let pref = client.stage_prompt(prompt).unwrap();
    let mut a = admit(lane, pref, false, 0, 0);
    a.want_logprobs = true;
    a.decode_replay = decode_replay;
    let mut p = plan(*seq);
    *seq += 1;
    p.admits.push(a);
    if prefill_len > 0 {
        p.prefills.push(linkw::LanePrefillMsg { lane_tag: lane, token_offset: 0, token_count: prefill_len });
    }
    p.decodes.push(linkw::LaneDecodeMsg { lane_tag: lane, max_new_tokens: n + decode_replay as u16, overshoot: 0 });
    let ev = match client.tick(p) {
        Ok(ev) => ev,
        Err(superfluid_agent::AgentError::Rejected(s)) => return Err(s),
        Err(e) => panic!("tick: {e}"),
    };
    let emit = ev.emits.iter().find(|e| e.lane_tag == lane).expect("an emit");
    let out = client.read_tokens(&emit.token_ref).unwrap();
    let lp: Vec<u32> = ev.logprobs.iter().filter(|l| l.lane_tag == lane).map(|l| l.logprob_bits).collect();
    let mut r = plan(*seq);
    *seq += 1;
    r.retires.push(linkw::LaneRetireMsg { lane_tag: lane, publish_to_cache: false });
    client.tick(r).unwrap();
    Ok((out, lp))
}

type Spawn = fn(PathBuf) -> (WorkerClient, thread::JoinHandle<()>);

#[test]
fn a_continuation_with_decode_replay_is_bitwise_the_uninterrupted_lane() {
    continuation_is_bitwise_the_uninterrupted_lane_on(spawn_native);
}

#[test]
fn ffi_continuation_with_decode_replay_is_bitwise_the_uninterrupted_lane() {
    continuation_is_bitwise_the_uninterrupted_lane_on(spawn_ffi);
}

fn continuation_is_bitwise_the_uninterrupted_lane_on(spawn: Spawn) {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model (set BASERT_TEST_MODEL)");
        return;
    };
    let tok = TokenizerHandle::load(&model).expect("tokenizer");
    let (mut client, _worker) = spawn(model);
    let mut seq = 1u64;
    let prompt = tok.encode("The seven wonders of the ancient world are, in order of construction:");
    const N: u16 = 24;
    const CUT: usize = 9;

    let (full, full_lp) = run(&mut client, &mut seq, 1, &prompt, prompt.len() as u32, 0, N).expect("admitted");
    assert_eq!(full.len(), N as usize, "the reference produced N tokens");
    let emits_logprobs = !full_lp.is_empty();
    if emits_logprobs {
        assert_eq!(full_lp.len(), N as usize, "one logprob per token");
    }

    let mut resumed = prompt.clone();
    resumed.extend_from_slice(&full[..CUT]);
    let (cont, cont_lp) =
        run(&mut client, &mut seq, 2, &resumed, prompt.len() as u32, CUT as u32, N - CUT as u16).expect("admitted");
    assert_eq!(cont, full[CUT..].to_vec(), "the continuation produces the uninterrupted lane's tokens");
    if emits_logprobs {
        assert_eq!(
            cont_lp,
            full_lp[CUT..].to_vec(),
            "the continuation's rows ARE the uninterrupted lane's rows (f32 logprob bits)"
        );
    }

    let (slice, slice_lp) =
        run(&mut client, &mut seq, 3, &resumed, resumed.len() as u32, 0, N - CUT as u16).expect("admitted");
    let rows_equal = slice_lp.iter().zip(full_lp.get(CUT..).unwrap_or(&[])).filter(|(a, b)| a == b).count();
    eprintln!(
        "prefill-slice resume: {}/{} tokens agree, {}/{} logprob rows bit-equal (diagnostic)",
        slice.iter().zip(&full[CUT..]).filter(|(a, b)| a == b).count(),
        slice.len(),
        rows_equal,
        slice_lp.len()
    );
}

#[test]
fn decode_replay_bounds_the_prefill() {
    decode_replay_bounds_the_prefill_on(spawn_native);
}

#[test]
fn ffi_decode_replay_bounds_the_prefill() {
    decode_replay_bounds_the_prefill_on(spawn_ffi);
}

fn decode_replay_bounds_the_prefill_on(spawn: Spawn) {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model (set BASERT_TEST_MODEL)");
        return;
    };
    let tok = TokenizerHandle::load(&model).expect("tokenizer");
    let (mut client, _worker) = spawn(model);
    let mut seq = 1u64;
    let prompt = tok.encode("One two three four five six seven eight nine ten eleven twelve");
    let len = prompt.len() as u32;
    assert!(len > 6, "prompt spans several tokens");

    let r = run(&mut client, &mut seq, 1, &prompt, len, 4, 2);
    assert_eq!(r.err(), Some(superfluid_abi::Status::RejectBounds as i32), "prefill past prefill_end is refused");
    let r = run(&mut client, &mut seq, 2, &prompt, len - 6, 4, 2);
    assert_eq!(
        r.err(),
        Some(superfluid_abi::Status::RejectIllegalCombination as i32),
        "decode before prefill_end is refused"
    );
    let r = run(&mut client, &mut seq, 3, &prompt, 0, len, 2);
    assert_eq!(r.err(), Some(superfluid_abi::Status::RejectBounds as i32), "a tail that is the whole prompt is refused");
    let (out, _) = run(&mut client, &mut seq, 4, &prompt, len - 4, 4, 2).expect("admitted");
    assert_eq!(out.len(), 2);
}

#[test]
fn a_long_replay_tail_is_paced_by_the_grant() {
    long_replay_tail_is_paced_by_the_grant_on(spawn_native, true);
}

#[test]
fn ffi_long_replay_tail_is_paced_by_the_grant() {
    long_replay_tail_is_paced_by_the_grant_on(spawn_ffi, false);
}

fn long_replay_tail_is_paced_by_the_grant_on(spawn: Spawn, snapshot_checks: bool) {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model (set BASERT_TEST_MODEL)");
        return;
    };
    let tok = TokenizerHandle::load(&model).expect("tokenizer");
    let (mut client, _worker) = spawn(model);
    let mut seq = 1u64;
    let prompt = tok.encode("A short history of the printing press, in three paragraphs:");
    const N: u16 = 48;
    const CUT: usize = 40;
    const GRANT: u16 = 8;
    let (full, _) = run(&mut client, &mut seq, 1, &prompt, prompt.len() as u32, 0, N).expect("admitted");
    assert_eq!(full.len(), N as usize);

    let mut resumed = prompt.clone();
    resumed.extend_from_slice(&full[..CUT]);
    let lane = 2u64;
    let pref = client.stage_prompt(&resumed).unwrap();
    let mut a = admit(lane, pref, false, 0, 0);
    a.decode_replay = CUT as u32;
    let mut p = plan(seq);
    seq += 1;
    p.admits.push(a);
    p.prefills.push(linkw::LanePrefillMsg { lane_tag: lane, token_offset: 0, token_count: prompt.len() as u32 });
    p.decodes.push(linkw::LaneDecodeMsg { lane_tag: lane, max_new_tokens: GRANT, overshoot: 0 });
    let ev = client.tick(p).unwrap();
    let emit = ev.emits.iter().find(|e| e.lane_tag == lane).expect("an emit");
    let first = client.read_tokens(&emit.token_ref).unwrap();
    assert!(first.is_empty(), "a tick whose grant is inside the tail emits nothing, got {first:?}");

    if snapshot_checks {
        const KV: u32 = 1;
        let kv = client.hello.state_space_descriptors.iter().find(|d| d.space_id == KV).expect("kv desc");
        let page = kv.page_size_tokens as u64;
        let seq_handle = match client.state_sync(linkw::StateSyncReqMsg::LaneSequence { lane_tag: lane }).unwrap() {
            linkw::StateSyncOkMsg::LaneSequence { sequence } => sequence,
            other => panic!("unexpected reply {other:?}"),
        };
        let stream_cut = ((resumed.len() as u64 - 1) / page) * page;
        let boundary = match client
            .state_sync(linkw::StateSyncReqMsg::SnapshotBoundary { sequence: seq_handle, space_id: KV, cap: resumed.len() as u64 - 1 })
            .unwrap()
        {
            linkw::StateSyncOkMsg::SnapshotBoundary { boundary } => boundary,
            other => panic!("unexpected reply {other:?}"),
        };
        let live_max = ((prompt.len() as u64 + GRANT as u64) / page) * page;
        assert!(boundary <= live_max, "the boundary is the LIVE length's aligned prefix ({boundary} <= {live_max})");
        assert!(boundary < stream_cut, "mid-replay, the live cut is short of the stream's ({boundary} < {stream_cut})");
        if stream_cut > 0 {
            let past_live = client.state_sync(linkw::StateSyncReqMsg::ExportSize {
                sequence: seq_handle,
                space_id: KV,
                range: linkw::TokenRangeMsg { start: 0, end: stream_cut },
                encoding: superfluid_abi::encoding::LOSSLESS,
            });
            assert!(past_live.is_err(), "an export cut at the stream's end is refused mid-replay: {past_live:?}");
        }
    }

    let mut out = first;
    let mut ticks = 1;
    while out.len() < (N - CUT as u16) as usize {
        let mut p = plan(seq);
        seq += 1;
        p.decodes.push(linkw::LaneDecodeMsg { lane_tag: lane, max_new_tokens: GRANT, overshoot: 0 });
        let ev = client.tick(p).unwrap();
        let emit = ev.emits.iter().find(|e| e.lane_tag == lane).expect("an emit");
        out.extend(client.read_tokens(&emit.token_ref).unwrap());
        ticks += 1;
        assert!(ticks < 40, "the tail never finished replaying");
    }
    assert!(ticks >= 6, "the tail was replayed inside fewer ticks than its grant allows ({ticks})");
    assert_eq!(&out[..(N - CUT as u16) as usize], &full[CUT..], "the paced continuation is the uninterrupted lane");
    let mut r = plan(seq);
    r.retires.push(linkw::LaneRetireMsg { lane_tag: lane, publish_to_cache: false });
    client.tick(r).unwrap();
}
