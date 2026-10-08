"""Track 2: throughput.

Each scenario starts a system once (8 lanes), drops a warm-up block, then for
each concurrency runs ROUNDS rounds of distinct, nonce-prefixed prompts and
reports the median round: aggregate tok/s, the per-request decode rate,
TTFT, inter-token gaps, prefill rate. Every request is streamed.

Correctness sits beside the numbers: in `short`, the same greedy prompts run
alone and then together at each concurrency, and how many replies are
identical to the solo ones is reported (a difference is a result, not
noise: batch shapes change floating-point sums).

Quant honesty: each system runs its native format of the box's model (GGUF
Q4_K_M for llama.cpp-based systems, MLX 4-bit for MLX ones); rows across
formats compare servers, not kernels.
"""
import time

import systems
from common import chat, r3, thermal
from load import completion_tokens, median_round, run_load, summarize
from track1 import QUESTIONS, ask, body, document, started, unsupported

CONCURRENCY = (1, 2, 4, 8)
ROUNDS = 3
LONG_ANSWER = " Answer in about 300 words."


def _nonce(i, seed):
    return f"[request {seed}-{i}] "


def short_bodies(srv, n, seed, max_tokens=256, **kw):
    return [body(srv, ask(_nonce(i, seed) + QUESTIONS[i % len(QUESTIONS)] + LONG_ANSWER), max_tokens=max_tokens, **kw)
            for i in range(n)]


def sweep(srv, make, conc, rounds, warm):
    """{concurrency: median round} for `make(n, seed)` request lists."""
    run_load(srv, make(warm, 0), min(warm, 8))
    rows = {}
    for c in conc:
        n = max(2 * c, 3)
        rows[c] = median_round([summarize(*run_load(srv, make(n, 1000 * c + r), c)) for r in range(rounds)])
    return rows


def identical_at(srv, conc):
    """Greedy replies to fixed prompts, alone and then `c` at a time: how
    many match the solo reply, per concurrency."""
    prompts = [body(srv, ask(q), max_tokens=128) for q in QUESTIONS]
    solo = [chat(srv.url, b).text for b in prompts]
    out = {}
    for c in conc:
        if c == 1:
            continue
        reps, _ = run_load(srv, prompts, c)
        out[c] = sum(1 for r in reps if r.text in solo)
    return {"prompts": len(prompts), "identical_to_solo": out}


def _family(name, box, out, quick, make, *, ctx=4096, extra=None, check_identity=False, condition=None):
    if not systems.known(box, name):
        return unsupported("this box has no build of the system or its model")
    conc = (1, 8) if quick else CONCURRENCY
    rounds = 1 if quick else ROUNDS
    srv = started(name, box, out, lanes=8, ctx=ctx, extra=extra)
    try:
        rows = sweep(srv, lambda n, seed: make(srv, n, seed), conc, rounds, warm=8)
        ident = identical_at(srv, conc) if check_identity else None
    finally:
        srv.stop()
    failed = sum(r["failed"] for r in rows.values())
    res = {"condition": condition or "every request completes", "concurrency": {str(c): v for c, v in rows.items()},
           "load_s": r3(srv.load_s)}
    if ident:
        res["identity"] = ident
    res["verdict"] = "pass" if failed == 0 else "fail"
    parts = [f"c={c} {v['agg_tok_s']} tok/s (TTFT p50 {v['ttft_p50_s']} s)" for c, v in rows.items()]
    if ident:
        parts.append("identical " + ", ".join(f"c={c} {k}/{ident['prompts']}" for c, k in ident["identical_to_solo"].items()))
    res["headline"] = "; ".join(parts) + (f"; {failed} failed" if failed else "")
    return res


def short(name, box, out, quick=False):
    """Short prompts, about 300-word answers, greedy."""
    return _family(name, box, out, quick, lambda srv, n, seed: short_bodies(srv, n, seed), check_identity=True)


def sampled(name, box, out, quick=False):
    """As `short`, sampled (temperature 0.7, top_p 0.95, a seed per request)."""
    def make(srv, n, seed):
        return [dict(b, temperature=0.7, top_p=0.95, seed=seed + i) for i, b in enumerate(short_bodies(srv, n, seed))]
    return _family(name, box, out, quick, make)


def long(name, box, out, quick=False):
    """About 8k-token prompts (a distinct document each), 128-token answers:
    prefill-bound; TTFT and prefill rate are the numbers that matter."""
    def make(srv, n, seed):
        return [body(srv, ask(_nonce(i, seed) + "Summarise the key points of this text.\n\n" + document(8000, seed * 97 + i)),
                     max_tokens=128) for i in range(n)]
    return _family(name, box, out, quick, make, ctx=12288)


def shared(name, box, out, quick=False):
    """Every request shares one ~4k-token system prompt, then asks its own
    question: what prefix reuse is worth under load."""
    sysdoc = "You answer questions about this reference.\n\n" + document(4000, 4242)

    def make(srv, n, seed):
        return [body(srv, ask(_nonce(i, seed) + QUESTIONS[i % len(QUESTIONS)], system=sysdoc), max_tokens=128)
                for i in range(n)]
    return _family(name, box, out, quick, make, ctx=8192)


EDIT_SOURCE = """def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) < 3:
        return None
    name, age, city = parts[0], parts[1], parts[2]
    return {"name": name.strip(), "age": int(age), "city": city.strip()}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            row = parse_line(line)
            if row is not None:
                rows.append(row)
    return rows


def by_city(rows):
    out = {}
    for row in rows:
        out.setdefault(row["city"], []).append(row)
    return out
"""


def edit_bodies(srv, n, seed):
    """Rewrite-this-code requests: the reply repeats most of the prompt, the
    case prompt-lookup speculation is for."""
    return [body(srv, ask(_nonce(i, seed) + "Rewrite this module with type hints on every function. Output the "
                          "whole module and nothing else.\n\n" + EDIT_SOURCE), max_tokens=400) for i in range(n)]


def speculation(name, box, out, quick=False):
    """Edit-style prompts with the system's speculation off, then on (for
    superfluid `--speculate auto`; for others the box's `spec_args`)."""
    on = ["--speculate", "auto"] if name.startswith("superfluid-") else box.get("spec_args", {}).get(name)
    if on is None:
        return unsupported("no speculation setting for this system on this box (spec_args)")
    if not systems.known(box, name):
        return unsupported("this box has no build of the system or its model")
    conc = (1, 8) if quick else (1, 4, 8)
    rounds = 1 if quick else ROUNDS
    arms = {}
    for arm, extra in (("off", None), ("on", on)):
        srv = started(name, box, out, lanes=8, ctx=4096, extra=extra)
        try:
            arms[arm] = sweep(srv, lambda n, seed: edit_bodies(srv, n, seed), conc, rounds, warm=4)
        finally:
            srv.stop()
    speedup = {str(c): r3(arms["on"][c]["agg_tok_s"] / arms["off"][c]["agg_tok_s"]) for c in conc
               if arms["off"][c]["agg_tok_s"] and arms["on"][c]["agg_tok_s"]}
    failed = sum(r["failed"] for a in arms.values() for r in a.values())
    return {"condition": "every request completes", "verdict": "pass" if failed == 0 else "fail",
            "arms": {a: {str(c): v for c, v in rows.items()} for a, rows in arms.items()}, "speedup": speedup,
            "headline": ", ".join(f"c={c} x{s}" for c, s in speedup.items()) + " aggregate with speculation on"}


def sustained(name, box, out, quick=False):
    """Ten minutes of 8 concurrent clients: the rate per minute, first
    against last (a laptop chip's power cap can reorder rankings)."""
    if not systems.known(box, name):
        return unsupported("this box has no build of the system or its model")
    secs = 60 if quick else 600
    srv = started(name, box, out, lanes=8, ctx=4096)
    therm_before = thermal()
    try:
        import threading

        lock = threading.Lock()
        done = []
        stop_at = time.perf_counter() + secs
        t0 = time.perf_counter()

        def client(k):
            i = 0
            while time.perf_counter() < stop_at:
                r = chat(srv.url, short_bodies(srv, 1, 10_000 * k + i, max_tokens=512)[0])
                with lock:
                    done.append((time.perf_counter() - t0, completion_tokens(r) if r.ok else 0, r.ok))
                i += 1

        threads = [threading.Thread(target=client, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
    finally:
        srv.stop()
    minutes = max(1, int(secs // 60))
    per_min = [0] * minutes
    for at, n, _ in done:
        per_min[min(int(at // 60), minutes - 1)] += n
    rates = [r3(n / 60) for n in per_min]
    failed = sum(1 for _, _, ok in done if not ok)
    ratio = r3(rates[-1] / rates[0]) if rates and rates[0] else None
    return {"condition": "every request completes", "verdict": "pass" if failed == 0 else "fail",
            "seconds": r3(wall), "requests": len(done), "failed": failed, "tok_s_per_minute": rates,
            "last_over_first": ratio, "therm_before": therm_before,
            "headline": f"{rates[0]} tok/s first minute, {rates[-1]} last (x{ratio}); {len(done)} requests, {failed} failed"}


SCENARIOS = {
    "short": short,
    "sampled": sampled,
    "long": long,
    "shared": shared,
    "speculation": speculation,
    "sustained": sustained,
}
