"""Live Ollama contract/workload checks; needs httpx and a running server."""

import argparse
import concurrent.futures
import json
import math
import statistics
import time
from pathlib import Path

import httpx

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--model")
p.add_argument("--embedding", action="store_true")
p.add_argument("--output", required=True)
p.add_argument("--key", required=True, help="API key configured on the test server")
a = p.parse_args()
c = httpx.Client(
    base_url=a.url,
    headers={"Authorization": "Bearer " + a.key},
    timeout=90,
    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
)
results = []

def check(name, f):
    start = time.perf_counter()
    try:
        detail = f()
        result = {
            "name": name,
            "status": "PASS",
            "seconds": round(time.perf_counter() - start, 4),
            "detail": detail,
        }
    except Exception as e:
        result = {
            "name": name,
            "status": "FAIL",
            "seconds": round(time.perf_counter() - start, 4),
            "error": repr(e),
        }
    results.append(result)
    print(json.dumps(result), flush=True)
    Path(a.output).write_text(json.dumps(results, indent=2))

def post(path, body, status=200):
    r = c.post(path, json=body)
    assert r.status_code == status, (r.status_code, r.text[:600])
    return r.json()

models = c.get("/api/tags").json()["models"]
model = a.model or models[0]["model"]
print("MODEL", model, flush=True)

def discovery():
    assert any(m["model"] == model for m in models)
    for path in ["/", "/api/tags", "/api/ps", "/api/version"]:
        r = c.get(path)
        assert r.status_code == 200, (path, r.text)
        r = c.head(path)
        assert r.status_code == 200 and not r.content
    assert c.get("/api/version").json()["server"] == "superfluid"
    return post("/api/show", {"model": model, "verbose": True}).get("capabilities", [])

check("discovery and HEAD", discovery)

def errors():
    assert httpx.get(a.url + "/api/tags").status_code == 401
    for path, body, status in [
        ("/api/chat", {}, 400),
        ("/api/generate", {"model": "does-not-exist"}, 404),
        (
            "/api/chat",
            {"model": model, "messages": [{"role": "invalid", "content": "hi"}]},
            400,
        ),
        (
            "/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi", "images": ["!!!"]}],
            },
            400,
        ),
        ("/api/chat", {"model": model, "options": {"top_p": 2}}, 400),
        ("/api/chat", {"model": model, "options": {"num_predict": 0}}, 400),
        ("/api/chat", {"model": model, "options": {"num_ctx": 1}}, 400),
        ("/api/chat", {"model": model, "format": 7}, 400),
        (
            "/api/chat",
            {"model": model, "messages": [{"role": "tool", "content": "orphan"}]},
            400,
        ),
        ("/api/embed", {"model": model, "input": ["hi", 7]}, 400),
        ("/api/pull", {"model": model}, 501),
    ]:
        v = post(path, body, status)
        assert isinstance(v["error"], str)
    for text in ["", "{", "null", "[]"]:
        r = c.post("/api/chat", content=text)
        assert r.status_code == 400 and isinstance(r.json()["error"], str), (
            text,
            r.status_code,
            r.text,
        )
    r = c.get("/api/chat")
    assert r.status_code == 405 and isinstance(r.json()["error"], str)

check("validation auth and error envelopes (no keep_alive)", errors)

def generation(
    kind="chat",
    stream=False,
    prompt="Say hello in one short sentence.",
    budget=64,
    extra=None,
):
    body = {
        "model": model,
        "stream": stream,
        "think": False,
        "options": {"num_predict": budget, "temperature": 0},
    }
    body.update(
        {"messages": [{"role": "user", "content": prompt}]}
        if kind == "chat"
        else {"prompt": prompt}
    )
    body.update(extra or {})
    if stream:
        with c.stream("POST", "/api/" + kind, json=body) as r:
            assert r.status_code == 200, r.read()
            assert "application/x-ndjson" in r.headers["content-type"]
            frames = [json.loads(line) for line in r.iter_lines() if line]
    else:
        frames = [post("/api/" + kind, body)]
    assert frames and not any("error" in v for v in frames), frames
    assert sum(v["done"] for v in frames) == 1 and frames[-1]["done"], frames
    assert frames[-1]["eval_count"] <= budget, frames[-1]
    assert frames[-1]["total_duration"] > 0 and frames[-1]["prompt_eval_count"] > 0
    for v in frames:
        assert v["model"] == model and v["created_at"].endswith("Z"), v
    text = "".join(
        v.get("message", {}).get("content", "") if kind == "chat" else v["response"]
        for v in frames
    )
    return text, frames

if not a.embedding:

    def unsupported_embeddings():
        for path, body in [
            ("/api/embed", {"input": "hello"}),
            ("/api/embeddings", {"prompt": "hello"}),
        ]:
            v = post(path, dict(model=model, **body), 400)
            assert "embedding" in v["error"].lower(), v
        generation(budget=8)

    check(
        "unsupported embeddings are client errors and serving recovers",
        unsupported_embeddings,
    )
    for kind in ["chat", "generate"]:
        for stream in [False, True]:
            check(
                f"{kind} stream={stream}", lambda k=kind, s=stream: generation(k, s)[0]
            )

    def parity():
        prompt = "What is two plus two? Answer briefly."
        o = post(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 64,
                "enable_thinking": False,
            },
        )
        expected = o["choices"][0]["message"]["content"]
        for kind in ["chat", "generate"]:
            for stream in [False, True]:
                actual, _ = generation(kind, stream, prompt)
                assert actual == expected, (kind, stream, expected, actual)
        return expected

    check("OpenAI / chat / generate transport parity", parity)

    def schema():
        fmt = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        for kind in ["chat", "generate"]:
            for stream in [False, True]:
                text, _ = generation(
                    kind, stream, "Return answer 42 as JSON.", 96, {"format": fmt}
                )
                v = json.loads(text)
                assert set(v) == {"answer"} and type(v["answer"]) is int, v
        text, _ = generation(
            "generate",
            False,
            "Return only the JSON array [1,2].",
            64,
            {"format": "json"},
        )
        json.loads(text)

    check("JSON schema and JSON-value mode", schema)

    def thinking():
        _, frames = generation("chat", True, "What is 7 times 8?", 128, {"think": True})
        assert any(v["message"].get("thinking") for v in frames), frames

    check("separate reasoning stream", thinking)

    def tools():
        tool = {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get the current weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
        prompt = "You must call weather for Melbourne. Do not answer from memory."
        for stream in [False, True]:
            _, frames = generation("chat", stream, prompt, 192, {"tools": [tool]})
            calls = [t for v in frames for t in v["message"].get("tool_calls", [])]
            assert len(calls) == 1 and isinstance(
                calls[0]["function"]["arguments"], dict
            ), frames
            history = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "", "tool_calls": calls},
                {
                    "role": "tool",
                    "tool_name": "weather",
                    "content": "Sunny, 22 degrees C.",
                },
            ]
            text, _ = generation("chat", False, extra={"messages": history})
            assert text.strip()

    check("tool round trip / object arguments / stream once", tools)

    def raw_fim():
        for stream in [False, True]:
            for extras in [{"raw": True}, {"suffix": ";"}]:
                extras["think"] = None
                _, frames = generation("generate", stream, "let x = ", 16, extras)
                assert frames[-1]["done"]

    check("raw and FIM transports", raw_fim)
    for size in [1, 64, 256]:
        check(
            f"prompt workload repeated_words={size}",
            lambda n=size: generation(
                "chat", True, "apple pear plum. " * n + " Name one fruit.", 32
            )[1][-1]["prompt_eval_count"],
        )

    def oversized():
        for stream in [False, True]:
            r = c.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "x " * 10000}],
                    "stream": stream,
                    "options": {"num_predict": 16},
                },
            )
            if stream and r.status_code == 200:
                values = [json.loads(s) for s in r.text.splitlines()]
                assert any("error" in v for v in values) and not any(
                    v.get("done") for v in values
                ), values[-1:]
            else:
                assert r.status_code == 400 and isinstance(r.json()["error"], str), (
                    r.status_code,
                    r.text[:400],
                )
        generation(budget=8)

    check("context overflow and subsequent recovery", oversized)
    for parallel in [1, 4, 8]:

        def workload(n=parallel):
            def one(i):
                start = time.perf_counter()
                text, frames = generation(
                    "chat" if i % 2 else "generate",
                    bool(i % 3),
                    "Name three colors.",
                    32,
                )
                assert text.strip()
                return time.perf_counter() - start, frames[-1]["eval_count"]

            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                timings = list(pool.map(one, range(2 * n)))
            elapsed = time.perf_counter() - start
            return {
                "requests": len(timings),
                "wall_seconds": round(elapsed, 4),
                "median_seconds": round(statistics.median(t[0] for t in timings), 4),
                "total_output_tokens": sum(t[1] for t in timings),
            }

        check(f"mixed concurrent requests={parallel}", workload)

    def disconnect():
        for _ in range(6):
            with c.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "Write a very long essay."}
                    ],
                    "options": {"num_predict": 1500},
                },
            ) as r:
                assert r.status_code == 200
                next(r.iter_lines())
        text, _ = generation(prompt="Say hello.", budget=16)
        assert text.strip()

    check("six stream disconnects and recovery", disconnect)

    def aliases():
        v = post("/api/generate", {"model": model + ":latest", "stream": False})
        assert v["done_reason"] == "load"

    check("known model preload and latest alias", aliases)
else:

    def vectors():
        for inputs in [
            "search_query: what is a cat?",
            ["search_document: a cat", "search_document: a dog"],
            [""],
            [],
        ]:
            v = post("/api/embed", {"model": model, "input": inputs})
            expected = len(inputs) if isinstance(inputs, list) else 1
            assert len(v["embeddings"]) == expected, v
            for vec in v["embeddings"]:
                assert vec and all(math.isfinite(x) for x in vec)
                assert abs(sum(x * x for x in vec) - 1) < 1e-4
        return "finite normalized vectors"

    check("embedding strings batches empty inputs", vectors)

    def embed_parity():
        text = "search_query: the weather"
        o = post("/v1/embeddings", {"model": model, "input": text})["data"][0][
            "embedding"
        ]
        v = post("/api/embed", {"model": model, "input": text})["embeddings"][0]
        norm = math.sqrt(sum(x * x for x in o))
        assert max(abs(a - b / norm) for a, b in zip(v, o)) < 1e-5
        small = post("/api/embed", {"model": model, "input": text, "dimensions": 64})[
            "embeddings"
        ][0]
        norm = math.sqrt(sum(x * x for x in v[:64]))
        assert (
            len(small) == 64 and max(abs(a - b / norm) for a, b in zip(small, v)) < 1e-5
        )
        legacy = post("/api/embeddings", {"model": model, "prompt": text})["embedding"]
        assert (
            len(legacy) == len(o) and max(abs(a - b) for a, b in zip(legacy, o)) < 1e-5
        )
        return {"native_dimensions": len(v), "reduced_dimensions": len(small)}

    check("embedding OpenAI legacy and dimensions parity", embed_parity)

    def truncate():
        text = "a word " * 1000
        v = post("/api/embed", {"model": model, "input": text, "truncate": False}, 400)
        assert isinstance(v["error"], str)
        v = post("/api/embed", {"model": model, "input": text})
        assert v["embeddings"] and v["prompt_eval_count"] <= 512

    check("embedding context truncation and refusal", truncate)

    def empty_preloads():
        for body in [{}, {"input": None}, {"input": ""}, {"input": []}]:
            v = post("/api/embed", dict(model=model, **body))
            assert v["embeddings"] == [] and v["prompt_eval_count"] == 0, v
        for body in [{}, {"prompt": ""}]:
            assert post("/api/embeddings", dict(model=model, **body))["embedding"] == []

    check("empty embedding requests preload without inference", empty_preloads)

    def concurrent_embeds():
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            values = list(
                pool.map(
                    lambda i: post(
                        "/api/embed",
                        {"model": model, "input": f"search_query: document {i}"},
                    ),
                    range(16),
                )
            )
        assert all(len(v["embeddings"]) == 1 for v in values)

    check("16 embeddings at concurrency 8", concurrent_embeds)
c.close()
raise SystemExit(any(r["status"] == "FAIL" for r in results))
