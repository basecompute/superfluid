"""Closed-loop load against one server, and its numbers.

`concurrency` clients each send their next request as soon as the last one
ends. Aggregate tok/s is completion tokens over wall time; the per-request
decode rate is each stream's own (tokens after the first over the time from
first to last chunk). The two differ by TTFT and ragged starts and ends, and
the gap is itself a result. Inter-token gaps are per streamed chunk (a chunk
may carry several tokens when a server speculates).
"""
import threading
import time

from common import chat, med, pct, r3


def completion_tokens(rep):
    u = rep.usage or {}
    if u.get("completion_tokens") is not None:
        return u["completion_tokens"]
    return len(rep.stamps)


def prompt_tokens(rep):
    return (rep.usage or {}).get("prompt_tokens")


def run_load(srv, bodies, concurrency, headers=None, timeout=1800):
    """Send `bodies` (streamed requests) with `concurrency` clients; return
    the replies in completion order and the wall time."""
    queue = list(bodies)
    lock = threading.Lock()
    reps = []

    def client():
        while True:
            with lock:
                if not queue:
                    return
                b = queue.pop(0)
            r = chat(srv.url, b, headers=headers, timeout=timeout)
            r.done_at = time.perf_counter()
            with lock:
                reps.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=client) for _ in range(min(concurrency, len(queue)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    for r in reps:
        r.done_at -= t0
    return reps, wall


def summarize(reps, wall):
    ok = [r for r in reps if r.ok]
    toks = sum(completion_tokens(r) for r in ok)
    per_req = []
    gaps = []
    for r in ok:
        n = completion_tokens(r)
        if n > 1 and r.ttft is not None and r.stamps and r.stamps[-1] > r.ttft:
            per_req.append((n - 1) / (r.stamps[-1] - r.ttft))
        gaps += [b - a for a, b in zip(r.stamps, r.stamps[1:])]
    prefill = [prompt_tokens(r) / r.ttft for r in ok if prompt_tokens(r) and r.ttft]
    return {
        "requests": len(reps),
        "failed": len(reps) - len(ok),
        "errors": sorted({str(r.error)[:160] for r in reps if not r.ok})[:5],
        "wall_s": r3(wall),
        "completion_tokens": toks,
        "mean_completion_tokens": r3(toks / len(ok)) if ok else None,
        "agg_tok_s": r3(toks / wall) if wall > 0 else None,
        "per_request_decode_tok_s_p50": r3(med(per_req)),
        "ttft_p50_s": r3(med([r.ttft for r in ok])),
        "ttft_p95_s": r3(pct([r.ttft for r in ok], 95)),
        "prefill_tok_s_p50": r3(med(prefill)),
        "itl_p50_s": r3(pct(gaps, 50)),
        "itl_p95_s": r3(pct(gaps, 95)),
        "itl_p99_s": r3(pct(gaps, 99)),
    }


def median_round(rounds):
    """The round with the median aggregate rate, so its numbers belong to
    one run, plus the spread of that rate across rounds."""
    rs = sorted(rounds, key=lambda r: r["agg_tok_s"] or 0)
    mid = dict(rs[len(rs) // 2])
    rates = [r["agg_tok_s"] for r in rounds if r["agg_tok_s"]]
    mid["rounds"] = len(rounds)
    mid["agg_tok_s_range"] = [min(rates), max(rates)] if rates else None
    return mid
