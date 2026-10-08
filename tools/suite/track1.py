"""Track 1: serving semantics.

Each scenario takes (system name, box, out dir) and returns a dict with
`verdict` in pass | fail | unsupported | error, a one-line `headline`, the
pass condition it was judged on (`condition`), and its numbers. "unsupported"
means the system has no way to run the scenario at all, and says why; that
is a result, not a gap in the harness.

All requests are greedy (temperature 0, seed 1) with thinking off, against
Qwen3-4B Q4 (GGUF for llama.cpp-based systems, MLX 4-bit for MLX ones).
"""
import json
import os
import random
import re
import time

import systems
from common import chat, get_json, med, pct, post_json, r3, run_parallel

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

QUESTIONS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, and explain its complexity.",
    "Explain how a hash map handles collisions, with one concrete example.",
    "Write a SQL query that finds the second highest salary in an employees table, and explain it.",
    "Describe the difference between a process and a thread in an operating system.",
    "Write a JavaScript function that debounces another function, and explain when to use it.",
    "Explain what a binary search tree is and how insertion works.",
    "Write a Rust function that reverses a singly linked list, and explain ownership in it.",
    "Explain the CAP theorem and give an example of a system choosing AP.",
]

_WORDS = ("cache scheduler lane token prefill decode kernel buffer tensor matrix vector stream batch queue worker "
          "session thread mutex socket packet router shard replica quorum ledger index cursor page block layer "
          "weight gradient epoch sample window budget latency throughput deadline priority class policy").split()


def document(n_tokens, seed):
    """Deterministic prose of about n_tokens tokens (about 1.3 tokens per word
    for this vocabulary), different for every seed."""
    rng = random.Random(seed)
    out, words = [], 0
    target = int(n_tokens / 1.3)
    i = 0
    while words < target:
        i += 1
        k = rng.randint(8, 16)
        s = " ".join(rng.choice(_WORDS) for _ in range(k))
        out.append(f"Note {seed}.{i}: the {s}.")
        words += k + 3
    return " ".join(out)


def body(srv, messages, max_tokens=256, stream=True, **kw):
    b = {"model": srv.model_id, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 1,
         "stream": stream}
    if stream:
        b["stream_options"] = {"include_usage": True}
    b.update(NO_THINK)
    b.update(systems.BODY_EXTRA.get(srv.name, {}))
    b.update(kw)
    return b


def ask(q, system=None):
    m = []
    if system:
        m.append({"role": "system", "content": system})
    m.append({"role": "user", "content": q})
    return m


def cached_tokens(rep):
    u = rep.usage or {}
    d = u.get("prompt_tokens_details") or {}
    if d.get("cached_tokens") is not None:
        return d["cached_tokens"]
    t = (rep.raw or {}).get("timings") or {}
    return t.get("cache_n")


def metric(srv, name):
    """One Prometheus sample (summed over labels) from /metrics, or None."""
    import http.client

    try:
        c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
        c.request("GET", "/metrics")
        txt = c.getresponse().read().decode("utf-8", "replace")
        c.close()
    except OSError:
        return None
    vals = [float(m.group(1)) for m in re.finditer(rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", txt, re.M)]
    return sum(vals) if vals else None


def busy_lanes(srv):
    """How many sequences the server is generating right now, where it says."""
    if srv.name.startswith("superfluid-"):
        v = metric(srv, "superfluid_lanes_active")
        return None if v is None else int(v)
    if srv.name == "llama-server":
        st, v = get_json(srv.url, "/slots")
        if st == 200 and isinstance(v, list):
            return sum(1 for s in v if s.get("is_processing"))
    return None


def wait_ok(srv, timeout=300, prompt="Say OK."):
    """Poll with a tiny request until one succeeds; seconds taken or None."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        r = chat(srv.url, body(srv, ask(prompt), max_tokens=4, stream=False), timeout=60)
        if r.ok:
            return time.perf_counter() - t0
        time.sleep(0.25)
    return None


def unsupported(reason):
    return {"verdict": "unsupported", "reason": reason, "headline": reason}


def started(name, box, out, **kw):
    srv = systems.build(name, box, out, **kw)
    srv.load_s = srv.start()
    # Warm-up: weights paged in, kernels compiled; never measured.
    chat(srv.url, body(srv, ask("Say hello."), max_tokens=8, stream=False))
    return srv


# ------------------------------------------------------------ 1. worker kill


def worker_kill(name, box, out, quick=False):
    """SIGKILL the inference worker mid-generation.

    pass: the interrupted stream ends with a typed error (SSE error event or
    HTTP error) or completes, what it delivered is a clean prefix of the
    uninterrupted reply (nothing dropped or duplicated), the server serves
    again without a restart from outside, and the same prompt then gives the
    reference reply again."""
    if not systems.CAPS[name]["worker"]:
        return unsupported("single process: no separate worker to kill; a crash takes the whole server down (see daemon_restart)")
    srv = started(name, box, out, lanes=4, ctx=4096)
    try:
        req = body(srv, ask(QUESTIONS[0]), max_tokens=400)
        ref = chat(srv.url, req)
        workers = srv.worker_pids()
        if not workers:
            return {"verdict": "error", "error": "no worker process found", "children": srv.children()}
        killed = {}

        def kill_at(rep):
            if len(rep.stamps) == 60 and not killed:
                killed["t"] = time.perf_counter()
                killed["chunks"] = len(rep.stamps)
                for p in srv.worker_pids():
                    os.kill(p, 9)
            return None

        cut = chat(srv.url, req, on_chunk=kill_at)
        recovered = wait_ok(srv, timeout=300)
        recovery_s = None if recovered is None else time.perf_counter() - killed.get("t", time.perf_counter())
        after = chat(srv.url, req) if recovered is not None else None
        prefix_clean = ref.text.startswith(cut.text)
        res = {
            "condition": "typed error or completion; delivered text is a prefix of the reference; serves again unaided; same reply after",
            "killed_after_chunks": killed.get("chunks"), "workers_killed": workers, "server_alive": srv.alive(),
            "cut": cut.summary(), "delivered_chars": len(cut.text), "reference_chars": len(ref.text),
            "prefix_clean": prefix_clean, "recovery_s": r3(recovery_s), "new_workers": srv.worker_pids(),
            "after_identical": after is not None and after.text == ref.text,
        }
        ok_end = cut.typed_error or (cut.ok and cut.text == ref.text)
        res["verdict"] = "pass" if (ok_end and prefix_clean and recovered is not None and res["after_identical"]) else "fail"
        res["headline"] = (f"end={cut.error_kind or 'complete'}, prefix_clean={prefix_clean}, recovery {r3(recovery_s)} s, "
                           f"same reply after={res['after_identical']}")
        return res
    finally:
        srv.stop()


# --------------------------------------------------------- 2. daemon restart


def daemon_restart(name, box, out, quick=False):
    """SIGKILL the whole server mid-conversation, restart it on the same
    state directory, resend the conversation (HTTP clients hold the history).

    pass: the server comes back on its state dir after a crash, and the
    resent conversation gets the identical greedy reply. Numbers: restart to
    ready, TTFT of the resent long conversation after the restart vs warm
    before it. (The session API's resume-by-id from the WAL is not an HTTP
    feature and is not exercised here.)"""
    convo = ask(document(5000, 7) + "\n\nSummarise the notes above in three sentences.",
                system="You are a careful assistant.")
    convo += [{"role": "assistant", "content": "The notes describe a serving system's internals."},
              {"role": "user", "content": "Which component is mentioned most, and why might that be?"}]
    srv = started(name, box, out, lanes=4, ctx=8192)
    try:
        req = body(srv, convo, max_tokens=160)
        ref = chat(srv.url, req)
        warm = chat(srv.url, req)
        cut_state = {}

        def crash(rep):
            if len(rep.stamps) == 20 and not cut_state:
                cut_state["t"] = time.perf_counter()
                srv.kill_main()
            return None

        cut = chat(srv.url, req, on_chunk=crash)
        t0 = time.perf_counter()
        restart_err = None
        try:
            srv2 = systems.build(name, box, out, lanes=4, ctx=8192, fresh_state=False)
            srv2.start()
        except Exception as e:  # noqa: BLE001
            restart_err = repr(e)
            srv2 = None
        ready_s = time.perf_counter() - t0
        if srv2 is None:
            return {"verdict": "fail", "headline": f"did not restart on its state dir: {restart_err}", "restart_error": restart_err,
                    "cut": cut.summary()}
        try:
            first = chat(srv2.url, req)
            second = chat(srv2.url, req)
        finally:
            srv2.stop()
        res = {
            "condition": "restarts on the same state dir after SIGKILL; resent conversation gets the identical reply",
            "prompt_tokens": (ref.usage or {}).get("prompt_tokens"), "cut": cut.summary(),
            "restart_to_ready_s": r3(ready_s), "ttft_before_cold_s": r3(ref.ttft), "ttft_before_warm_s": r3(warm.ttft),
            "ttft_after_restart_s": r3(first.ttft), "ttft_after_restart_warm_s": r3(second.ttft),
            "cached_after_restart": cached_tokens(first), "identical_after": first.text == ref.text,
        }
        res["verdict"] = "pass" if res["identical_after"] and first.ok else "fail"
        res["headline"] = (f"ready {r3(ready_s)} s, TTFT warm {r3(warm.ttft)} -> after restart {r3(first.ttft)} s, "
                           f"identical={res['identical_after']}")
        return res
    finally:
        srv.stop()


# ------------------------------------------- 3. agents + one interactive chat


def agents_interactive(name, box, out, quick=False):
    """8 background agents with long prompts saturate the server; 2 s later
    one short interactive chat arrives.

    pass: interactive TTFT <= 1.0 s, and every agent finishes without error
    and without a stall longer than 30 s. On superfluid the agents are sent
    as class `background` and the chat as `interactive` (x-superfluid-qos);
    a second pass without the header shows what the classes are worth. A
    first pass sends the agents alone (no chat at all, before any in the
    run): what serving a chat costs the agents when none comes."""
    n_agents = 4 if quick else 8
    srv = started(name, box, out, lanes=8, ctx=8192)
    try:
        idle = chat(srv.url, body(srv, ask("What is the capital of France? One word."), max_tokens=16))
        # qos / plain / qos again: a difference that flips with the order is
        # the order, not the classes.
        has_qos = bool(systems.CAPS[name]["qos_header"])
        passes = [("agents_only", has_qos)] + ([("qos", True), ("plain", False), ("qos_again", True)] if has_qos else [("plain", False)])
        out_passes = {}
        for label, use_qos in passes:
            hdr_bg = {systems.CAPS[name]["qos_header"]: "background"} if use_qos else {}
            hdr_fg = {systems.CAPS[name]["qos_header"]: "interactive"} if use_qos else {}
            seed0 = {"agents_only": 50, "qos": 100, "plain": 150, "qos_again": 200}[label]
            agent_reqs = [body(srv, ask(document(6000, seed0 + i) +
                                        "\n\nList five risks in these notes.", system="You are a code agent."),
                               max_tokens=384) for i in range(n_agents)]
            got = {}

            def interactive():
                time.sleep(2.0)
                got["fg"] = chat(srv.url, body(srv, ask("What is the capital of France? One word."), max_tokens=16), headers=hdr_fg)
                return got["fg"]

            with_chat = label != "agents_only"
            t0 = time.perf_counter()
            reps = run_parallel([lambda r=r: chat(srv.url, r, headers=hdr_bg) for r in agent_reqs] + ([interactive] if with_chat else []))
            wall = time.perf_counter() - t0
            agents = reps[:-1] if with_chat else reps
            fg = reps[-1] if with_chat else None
            bad = [a.summary() for a in agents if not hasattr(a, "ok") or not a.ok]
            gaps = [a.max_gap for a in agents if hasattr(a, "max_gap")]
            out_passes[label] = {
                "interactive_ttft_s": r3(fg.ttft) if fg else None, "interactive_ok": fg.ok if fg else None,
                "agents_failed": len(bad), "agent_errors": bad[:3],
                "agent_ttft_p50_s": r3(med([a.ttft for a in agents])), "agent_ttft_max_s": r3(max([a.ttft or 0 for a in agents])),
                "agent_max_stall_s": r3(max([g or 0 for g in gaps])), "wall_s": r3(wall),
            }
        main = out_passes[passes[1][0]]
        res = {"condition": "interactive TTFT <= 1.0 s; all agents finish, no stall > 30 s",
               "agents": n_agents, "agent_prompt_tokens_approx": 6000, "idle_interactive_ttft_s": r3(idle.ttft), "passes": out_passes}
        ok = (main["interactive_ok"] and main["interactive_ttft_s"] is not None and main["interactive_ttft_s"] <= 1.0
              and main["agents_failed"] == 0 and main["agent_max_stall_s"] <= 30)
        res["verdict"] = "pass" if ok else "fail"
        res["headline"] = "; ".join(
            (f"{k}: agent TTFT p50 {v['agent_ttft_p50_s']} s, wall {v['wall_s']} s" if k == "agents_only" else
             f"{k}: interactive TTFT {v['interactive_ttft_s']} s (idle {r3(idle.ttft)}), agents failed "
             f"{v['agents_failed']}, max stall {v['agent_max_stall_s']} s, wall {v['wall_s']} s")
            for k, v in out_passes.items())
        return res
    finally:
        srv.stop()


# --------------------------------------------------------- 4. KV exhaustion


def kv_exhaustion(name, box, out, quick=False):
    """Demand 4x the KV the server was given: 8 lanes x 4096 tokens, then 32
    concurrent requests of ~3000 prompt + 900 generated tokens.

    pass: every request either completes or gets a typed error (HTTP error
    with a body, or an SSE error event) -- no resets, hangs or truncated
    streams -- and the server admits a fresh request right after."""
    n = 16 if quick else 32
    srv = started(name, box, out, lanes=8, ctx=4096)
    try:
        reqs = [body(srv, ask(document(3000, 300 + i) + "\n\nExplain these notes in detail.", system="Be thorough."),
                     max_tokens=900, ignore_eos=True) for i in range(n)]
        t0 = time.perf_counter()
        reps = run_parallel([lambda r=r: chat(srv.url, r, timeout=1800) for r in reqs])
        wall = time.perf_counter() - t0
        kinds = {}
        for r in reps:
            k = "ok" if r.ok else (r.error_kind or "exception")
            kinds[k] = kinds.get(k, 0) + 1
        statuses = sorted({r.status for r in reps})
        t1 = time.perf_counter()
        readmit = chat(srv.url, body(srv, ask("Say OK."), max_tokens=4))
        res = {
            "condition": "each request completes or fails typed; no resets/hangs/truncation; fresh request admitted after",
            "requests": n, "kv_tokens_given": 8 * 4096, "kv_tokens_demanded_approx": n * 3900,
            "outcomes": kinds, "http_statuses": statuses, "wall_s": r3(wall),
            "completion_tokens": sum((r.usage or {}).get("completion_tokens", 0) for r in reps if r.ok),
            "ttft_p50_s": r3(pct([r.ttft for r in reps], 50)), "ttft_p99_s": r3(pct([r.ttft for r in reps], 99)),
            "readmit_ok": readmit.ok, "readmit_ttft_s": r3(readmit.ttft), "server_alive": srv.alive(),
            "error_samples": [r.summary() for r in reps if not r.ok][:3],
            "preemptions": metric(srv, "superfluid_preemptions_total") if name.startswith("superfluid-") else None,
        }
        untyped = sum(v for k, v in kinds.items() if k not in ("ok", "http", "event"))
        res["verdict"] = "pass" if untyped == 0 and readmit.ok else "fail"
        res["headline"] = f"outcomes {kinds}, statuses {statuses}, TTFT p99 {res['ttft_p99_s']} s, wall {r3(wall)} s, readmit {readmit.ok}"
        return res
    finally:
        srv.stop()


# --------------------------------------------------- 5. unload under traffic


def unload_reload(name, box, out, quick=False):
    """Unload the model while 4 streams are mid-generation, then send a new
    request for it.

    pass: every in-flight stream completes or ends with a typed error, and
    the model serves again (on its own, or after the system's load call).
    Number: service gap = unload call to the next request's first token."""
    if not systems.CAPS[name]["unload"]:
        return unsupported("no API to unload a model from a running server (single-model process)")
    extra = []
    if name.startswith("superfluid-"):
        # superfluid refuses to unload its last model: serve a second, small one.
        small = os.path.join(box["root"], "models", "small",
                             "Qwen3-0.6B-Q4_K_M.gguf" if name == "superfluid-llamacpp" else "Qwen3-0.6B-4bit")
        extra = ["--model", small]
    srv = started(name, box, out, lanes=8, ctx=4096, extra=extra)
    try:
        if name.startswith("superfluid-"):
            st, v = get_json(srv.url, "/v1/models")
            ids = [m["id"] for m in v["data"]]
            srv.model_id = next(i for i in ids if "0.6B" not in i)
        reqs = [body(srv, ask(q), max_tokens=500) for q in QUESTIONS[:4]]
        state = {}

        def unload_later():
            time.sleep(1.5)
            state["t"] = time.perf_counter()
            state["unload"] = systems.unload(srv)
            state["unload_s"] = time.perf_counter() - state["t"]
            return None

        reps = run_parallel([lambda r=r: chat(srv.url, r) for r in reqs] + [unload_later])[:-1]
        nxt = chat(srv.url, body(srv, ask("Say OK."), max_tokens=4))
        reloaded_by, first_try, load_reply = "on demand", nxt.summary(), None
        if not nxt.ok:
            id_before = srv.model_id
            load_reply = systems.reload(srv)
            nxt = chat(srv.url, body(srv, ask("Say OK."), max_tokens=4))
            reloaded_by = "explicit load call" + ("" if srv.model_id == id_before else f" (now answers as '{srv.model_id}')")
        gap = None if nxt.ttft is None else (nxt.t0 + nxt.ttft) - state["t"]
        kinds = {}
        for r in reps:
            k = "ok" if r.ok else (r.error_kind or "exception")
            kinds[k] = kinds.get(k, 0) + 1
        res = {
            "condition": "in-flight streams complete or fail typed; model serves again",
            "unload_status": state.get("unload", [None])[0], "unload_body": str(state.get("unload", [None, None])[1])[:300],
            "unload_call_s": r3(state.get("unload_s")), "in_flight": kinds, "next_ok": nxt.ok, "reloaded_by": reloaded_by if nxt.ok else None,
            "service_gap_s": r3(gap), "first_request_after_unload": first_try,
            "load_call": None if load_reply is None else [load_reply[0], str(load_reply[1])[:300]],
            "in_flight_samples": [r.summary() for r in reps][:4],
        }
        untyped = sum(v for k, v in kinds.items() if k not in ("ok", "http", "event"))
        res["verdict"] = "pass" if untyped == 0 and nxt.ok else "fail"
        res["headline"] = f"unload {res['unload_status']}, in-flight {kinds}, next ok={nxt.ok} ({res['reloaded_by']}), gap {r3(gap)} s"
        return res
    finally:
        srv.stop()


# --------------------------------------------------- 6. long tool session

TOOLS = [{"type": "function", "function": {
    "name": "lookup_code", "description": "Return the current access code for a named account. Codes change every call; never guess one.",
    "parameters": {"type": "object", "properties": {"account": {"type": "string"}}, "required": ["account"]}}}]


def tool_session(name, box, out, quick=False):
    """One conversation of N tool-calling turns: the user asks for an
    account's access code, the model calls `lookup_code` (it cannot know the
    code otherwise), the client answers, the model states the code.

    pass: every tool call the model emits comes back parsed in `tool_calls`
    (none leak into content as raw markup), and the conversation reaches N
    turns. Turns where the model answers without calling are counted, not
    failed (that is the model, not the server). Numbers: TTFT trend first-10 vs last-10 turns, cached-prompt ratio
    (prefix reuse), answer accuracy. Stops early at a 20-minute budget."""
    turns = 40 if quick else 200
    budget = 20 * 60
    srv = started(name, box, out, lanes=4, ctx=32768)
    try:
        rng = random.Random(5)
        msgs = [{"role": "system", "content": "You are an account assistant. Look up access codes with the lookup_code tool every time; codes change on every call. Answer in one short sentence."}]
        rows = []
        t_start = time.perf_counter()
        emitted = parsed = leaked = correct = errors = bad_args = 0
        for t in range(turns):
            if time.perf_counter() - t_start > budget:
                break
            account = f"acct-{rng.randint(1000, 9999)}"
            code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
            msgs.append({"role": "user", "content": f"What is the access code for {account}?"})
            r1 = chat(srv.url, body(srv, msgs, max_tokens=96, tools=TOOLS))
            if not r1.ok:
                errors += 1
                rows.append({"turn": t, "error": r1.summary()})
                break
            leak = "<tool_call>" in r1.text or '"name": "lookup_code"' in r1.text
            if r1.tool_calls or leak:
                emitted += 1
            if leak:
                leaked += 1
            if not r1.tool_calls:
                rows.append({"turn": t, "no_call": True, "text": r1.text[:200], "ttft": r1.ttft})
                msgs.append({"role": "assistant", "content": r1.text})
                continue
            parsed += 1
            call = r1.tool_calls[0]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except ValueError:
                args = None
            if not isinstance(args, dict) or args.get("account") != account:
                bad_args += 1
            result = code
            cid = call.get("id") or f"call_{t}"
            msgs.append({"role": "assistant", "content": r1.text or "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": str(result)})
            r2 = chat(srv.url, body(srv, msgs, max_tokens=64, tools=TOOLS))
            if not r2.ok:
                errors += 1
                rows.append({"turn": t, "error": r2.summary()})
                break
            if code in r2.text:
                correct += 1
            msgs.append({"role": "assistant", "content": r2.text})
            for r, phase in ((r1, "call"), (r2, "answer")):
                pt = (r.usage or {}).get("prompt_tokens")
                rows.append({"turn": t, "phase": phase, "ttft": r3(r.ttft), "prompt_tokens": pt, "cached": cached_tokens(r)})
        done = 1 + max([r["turn"] for r in rows] or [-1])
        ok_rows = [r for r in rows if r.get("ttft") is not None and r.get("phase")]
        first, last = ok_rows[:20], ok_rows[-20:]
        cached_rows = [r for r in ok_rows if r.get("cached") is not None and r.get("prompt_tokens")]
        ratio = (sum(r["cached"] for r in cached_rows) / sum(r["prompt_tokens"] for r in cached_rows)) if cached_rows else None
        res = {
            "condition": f"emitted tool calls == parsed, none leaked; reaches {turns} turns",
            "turns_target": turns, "turns_done": done, "elapsed_s": r3(time.perf_counter() - t_start),
            "tool_calls_emitted": emitted, "tool_calls_parsed": parsed, "leaked_as_text": leaked,
            "no_call_turns": sum(1 for r in rows if r.get("no_call")), "bad_arguments": bad_args, "answers_correct": correct, "errors": errors,
            "ttft_first10_s": r3(med([r["ttft"] for r in first])), "ttft_last10_s": r3(med([r["ttft"] for r in last])),
            "final_prompt_tokens": (ok_rows[-1]["prompt_tokens"] if ok_rows else None), "cached_ratio": r3(ratio),
            "rows": rows,
        }
        res["verdict"] = "pass" if (leaked == 0 and emitted == parsed and done >= turns and errors == 0) else "fail"
        res["headline"] = (f"{done}/{turns} turns, calls parsed {parsed}/{emitted} (leaked {leaked}), correct {correct}, "
                           f"TTFT {res['ttft_first10_s']} -> {res['ttft_last10_s']} s, cached ratio {res['cached_ratio']}")
        return res
    finally:
        srv.stop()


# ------------------------------------------------------ 7. client disconnect


def client_disconnect(name, box, out, quick=False):
    """8 long generations, then the client goes away -- streaming (drop the
    socket after 16 chunks) and non-streaming (drop it after 2 s).

    pass: after each disconnect the server stops generating for the gone
    clients: where it reports busy lanes they reach 0 within 1 s; otherwise
    8 fresh requests sent 1 s later get their first token as fast as on an
    idle server (<= 2x idle TTFT + 0.5 s), which they cannot if the
    abandoned work still holds the lanes."""
    lanes = 8
    variants = [("default", [])]
    if name.startswith("superfluid-"):
        variants.append(("nonstream-keepalive", ["--nonstream-keepalive", "5"]))
    passes = {}
    for label, extra in variants:
        srv = started(name, box, out, lanes=lanes, ctx=4096, extra=extra)
        try:
            probe = [body(srv, ask(q + " (probe)"), max_tokens=32) for q in QUESTIONS]
            idle = run_parallel([lambda r=r: chat(srv.url, r) for r in probe])
            idle_ttft = med([r.ttft for r in idle])
            row = {"idle_ttft_p50_s": r3(idle_ttft)}
            for mode in ("stream", "nonstream"):
                long = [body(srv, ask(q + " Be exhaustive.", system=f"Run {mode}."), max_tokens=2000, stream=(mode == "stream"),
                             ignore_eos=True) for q in QUESTIONS]
                if mode == "stream":
                    run_parallel([lambda r=r: chat(srv.url, r, abort_after_chunks=16) for r in long])
                else:
                    run_parallel([lambda r=r: chat(srv.url, r, abort_after_s=2.0) for r in long])
                t0 = time.perf_counter()
                freed_s = None
                while time.perf_counter() - t0 < 30:
                    b = busy_lanes(srv)
                    if b is None:
                        break
                    if b == 0:
                        freed_s = time.perf_counter() - t0
                        break
                    time.sleep(0.05)
                observable = busy_lanes(srv) is not None
                time.sleep(max(0.0, 1.0 - (time.perf_counter() - t0)))
                after = run_parallel([lambda r=r: chat(srv.url, r) for r in probe])
                a_ttft = med([r.ttft for r in after])
                if observable:
                    ok = freed_s is not None and freed_s <= 1.0
                else:
                    ok = a_ttft is not None and a_ttft <= 2 * idle_ttft + 0.5
                row[mode] = {"busy_observable": observable, "freed_s": r3(freed_s), "after_ttft_p50_s": r3(a_ttft),
                             "after_ttft_max_s": r3(max([r.ttft or 0 for r in after])), "freed": ok}
                # Let anything still running finish before the next mode.
                wait_ok(srv, timeout=600)
                deadline = time.time() + 600
                while time.time() < deadline and (busy_lanes(srv) or 0) > 0:
                    time.sleep(0.5)
            passes[label] = row
        finally:
            srv.stop()
    main = passes["default"]
    res = {"condition": "abandoned generations stop: busy lanes 0 within 1 s, or fresh requests at idle TTFT", "passes": passes}
    res["verdict"] = "pass" if main["stream"]["freed"] and main["nonstream"]["freed"] else "fail"
    res["headline"] = "; ".join(f"{k}: stream freed={v['stream']['freed']} ({v['stream']['freed_s'] or v['stream']['after_ttft_p50_s']} s), "
                                f"non-stream freed={v['nonstream']['freed']} ({v['nonstream']['freed_s'] or v['nonstream']['after_ttft_p50_s']} s)"
                                for k, v in passes.items())
    return res


# --------------------------------------------------- 8. shared-prefix burst


def shared_prefix(name, box, out, quick=False):
    """32 requests at once sharing one 4k-token system prompt (distinct user
    questions), against 32 with distinct 4k system prompts.

    pass: the shared prefix is prefilled at most twice over the burst, from the
    cached tokens the server reports (shared-burst TTFT p50 at most half the
    cold burst's where it reports none). Numbers: TTFT p50/p99 and wall for
    both, cached prompt tokens where the server reports them."""
    n = 16 if quick else 32
    srv = started(name, box, out, lanes=8, ctx=8192)
    try:
        shared_sys = document(4000, 900)
        qs = [f"Question {i}: in one sentence, what does note 900.{i + 1} say?" for i in range(n)]
        shared = [body(srv, ask(q, system=shared_sys), max_tokens=32) for q in qs]
        cold = [body(srv, ask(q, system=f"Batch {i}. " + document(4000, 1000 + i)), max_tokens=32) for i, q in enumerate(qs)]
        prefill0 = metric(srv, "superfluid_prefill_tokens_total") if name.startswith("superfluid-") else None
        t0 = time.perf_counter()
        s = run_parallel([lambda r=r: chat(srv.url, r) for r in shared])
        s_wall = time.perf_counter() - t0
        prefill1 = metric(srv, "superfluid_prefill_tokens_total") if name.startswith("superfluid-") else None
        t0 = time.perf_counter()
        c = run_parallel([lambda r=r: chat(srv.url, r) for r in cold])
        c_wall = time.perf_counter() - t0
        sc = [cached_tokens(r) for r in s]
        res = {
            "condition": "shared-burst TTFT p50 <= 0.5x cold-burst TTFT p50",
            "requests": n, "prefix_tokens": (s[0].usage or {}).get("prompt_tokens"),
            "shared": {"ttft_p50_s": r3(pct([r.ttft for r in s], 50)), "ttft_p99_s": r3(pct([r.ttft for r in s], 99)),
                       "wall_s": r3(s_wall), "failed": sum(1 for r in s if not r.ok),
                       "cached_tokens_sum": sum(x for x in sc if x) if any(x is not None for x in sc) else None},
            "cold": {"ttft_p50_s": r3(pct([r.ttft for r in c], 50)), "ttft_p99_s": r3(pct([r.ttft for r in c], 99)),
                     "wall_s": r3(c_wall), "failed": sum(1 for r in c if not r.ok)},
            "server_prefill_tokens_shared_burst": None if prefill0 is None or prefill1 is None else prefill1 - prefill0,
        }
        sp, cp = res["shared"]["ttft_p50_s"], res["cold"]["ttft_p50_s"]
        # The shared part is the prompt minus the short question (~40 tokens).
        prefix = max(0, (res["prefix_tokens"] or 0) - 40)
        cached = res["shared"]["cached_tokens_sum"]
        if cached is not None and prefix:
            times = n - cached / prefix
            res["prefix_prefills"] = r3(times)
            ok = times <= 2.0
            how = f"prefix prefilled {r3(times)}x"
        else:
            ok = sp is not None and cp and sp <= 0.5 * cp
            how = "no cached-token count reported; judged on TTFT"
        res["condition"] = "the shared 4k prefix is prefilled at most twice for the whole burst (from reported cached tokens; TTFT p50 <= 0.5x cold where none are reported)"
        res["verdict"] = "pass" if ok and res["shared"]["failed"] == 0 else "fail"
        res["headline"] = (f"{how}; TTFT p50 shared {sp} s vs cold {cp} s (p99 {res['shared']['ttft_p99_s']} vs {res['cold']['ttft_p99_s']}), "
                           f"wall {r3(s_wall)} vs {r3(c_wall)} s")
        return res
    finally:
        srv.stop()


SCENARIOS = {
    "worker_kill": worker_kill,
    "daemon_restart": daemon_restart,
    "agents_interactive": agents_interactive,
    "kv_exhaustion": kv_exhaustion,
    "unload_reload": unload_reload,
    "tool_session": tool_session,
    "client_disconnect": client_disconnect,
    "shared_prefix": shared_prefix,
}
