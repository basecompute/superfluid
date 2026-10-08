//! A preempted constrained lane's continuation, on the REAL engine.

mod common;
use common::*;

use superfluid_engine_ffi::TokenizerHandle;

const SCHEMA: &str =
    r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}"#;

fn create_grammar(client: &mut WorkerClient) -> u32 {
    match client.state_sync(linkw::StateSyncReqMsg::CreateGrammar { json_schema: SCHEMA.to_string() }).unwrap() {
        linkw::StateSyncOkMsg::CreateGrammar { grammar_handle } => grammar_handle,
        other => panic!("CreateGrammar replied {other:?}"),
    }
}

fn constrained(
    client: &mut WorkerClient,
    seq: &mut u64,
    lane: u64,
    prompt: &[u32],
    grammar: u32,
    replay: u32,
    n: u16,
) -> Result<Vec<u32>, i32> {
    let pref = client.stage_prompt(prompt).unwrap();
    let mut a = admit(lane, pref, false, 0, 0);
    a.grammar_handle = grammar;
    a.grammar_replay = replay;
    let mut p = plan(*seq);
    *seq += 1;
    p.admits.push(a);
    p.prefills.push(linkw::LanePrefillMsg { lane_tag: lane, token_offset: 0, token_count: prompt.len() as u32 });
    p.decodes.push(linkw::LaneDecodeMsg { lane_tag: lane, max_new_tokens: n, overshoot: 0 });
    let ev = match client.tick(p) {
        Ok(ev) => ev,
        Err(superfluid_agent::AgentError::Rejected(s)) => return Err(s),
        Err(e) => panic!("tick: {e}"),
    };
    let emit = ev.emits.iter().find(|e| e.lane_tag == lane).expect("an emit");
    let out = client.read_tokens(&emit.token_ref).unwrap();
    let mut r = plan(*seq);
    *seq += 1;
    r.retires.push(linkw::LaneRetireMsg { lane_tag: lane, publish_to_cache: false });
    client.tick(r).unwrap();
    Ok(out)
}

fn text(tok: &TokenizerHandle, tokens: &[u32]) -> String {
    let bytes: Vec<u8> = tokens.iter().flat_map(|&t| tok.token_bytes(t)).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[test]
fn a_continuation_replays_its_grammar_and_finishes_one_object() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let Some(model) = model_path() else {
        eprintln!("SKIP: no test model (set BASERT_TEST_MODEL)");
        return;
    };
    let tok = TokenizerHandle::load(&model).expect("tokenizer");
    let (mut client, _worker) = spawn_native(model);
    let mut seq = 1u64;
    let g = create_grammar(&mut client);
    let prompt = tok.encode("Output a JSON object with the capital city of France.\n");

    let head = constrained(&mut client, &mut seq, 1, &prompt, g, 0, 5).expect("admitted");
    assert_eq!(head.len(), 5);
    let mut cont = prompt.clone();
    cont.extend_from_slice(&head);

    let resumed = constrained(&mut client, &mut seq, 2, &cont, g, 5, 64).expect("admitted");
    let whole = text(&tok, &[head.clone(), resumed].concat());
    let v: serde_json::Value = serde_json::from_str(whole.trim_end_matches(|c: char| c.is_whitespace() || c == '\u{0}'))
        .unwrap_or_else(|e| panic!("not one JSON object ({e}): {whole:?}"));
    assert!(v["city"].is_string(), "schema-valid: {whole:?}");

    let restarted = constrained(&mut client, &mut seq, 3, &cont, g, 0, 8).expect("admitted");
    assert!(
        text(&tok, &restarted).trim_start().starts_with('{'),
        "without the replay the grammar opens a new object mid-response: {:?}",
        text(&tok, &restarted)
    );

    let illegal = superfluid_abi::Status::RejectIllegalCombination as i32;
    assert_eq!(constrained(&mut client, &mut seq, 4, &prompt, g, 3, 8), Err(illegal));
    assert_eq!(constrained(&mut client, &mut seq, 5, &cont, g, cont.len() as u32 + 1, 8), Err(illegal));
    assert!(constrained(&mut client, &mut seq, 6, &cont, g, 5, 64).is_ok());
}
