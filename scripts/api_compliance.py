#!/usr/bin/env python3
"""superfluid API compliance.

Checks a running superfluid's responses against the published shape of each
API it serves: required fields, value types and enum values, and the order
of streamed events. It does not compare output with another server.

  openai     chat and text completions (non-streamed, streamed, logprobs,
             usage in the stream), tool calls, structured outputs, models,
             files, batches, errors
  anthropic  Messages: non-streamed, every streamed event, tool use, the
             thinking switch, errors
  ollama     discovery, chat and generate (NDJSON streams), tools, JSON
             schema output, errors
  server     the llama.cpp-server routes clients probe: /health, /props,
             /slots, /v1/tokenize, /metrics
  embed      /v1/embeddings, /api/embed and /api/embeddings (an embedding
             model; one that serves none is reported as skipped)
  all        openai, anthropic, ollama and server (the default)

Where the published API has a field superfluid documents it does not serve
(Ollama's model metadata and phase timings), the run notes it and does not
fail. Run it against a decode model:

  api_compliance.py --url http://127.0.0.1:8453 [--model ID] [--mode MODE] [--key KEY]

It prints one line per check and exits 1 if any check fails.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

NUM = (int, float)
OPT_STR = (str, type(None))

# Tool and schema fixtures shared by every API's checks.
WEATHER = {
    "name": "get_weather",
    "description": "Get the current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
PLANET = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False}


class Server:
    def __init__(self, url, key):
        self.url = url.rstrip("/")
        self.key = key

    def call(self, method, path, body=None, headers=None, data=None, timeout=180):
        """`(status, text)`; a body is sent as JSON, `data` as given."""
        h = {"anthropic-version": "2023-06-01"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
            h["x-api-key"] = self.key
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def json(self, method, path, body=None, **kw):
        status, text = self.call(method, path, body, **kw)
        try:
            return status, json.loads(text)
        except ValueError:
            return status, {"_raw": text[:300]}

    def upload(self, path, purpose, filename, content):
        boundary = uuid.uuid4().hex
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\n{purpose}\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/jsonl\r\n\r\n".encode() + content + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        status, text = self.call("POST", path, data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            return status, json.loads(text)
        except ValueError:
            return status, {"_raw": text[:300]}


def sse(raw):
    """Server-sent events as `(event name or None, data)`, and whether the
    stream ended with `[DONE]`."""
    out, event, done = [], None, False
    for ln in raw.split("\n"):
        ln = ln.strip()
        if ln.startswith("event:"):
            event = ln[6:].strip()
        elif ln.startswith("data:"):
            d = ln[5:].strip()
            if d == "[DONE]":
                done = True
            elif d:
                try:
                    out.append((event, json.loads(d)))
                except ValueError:
                    out.append((event, {"_raw": d[:200]}))
            event = None
    return out, done


def ndjson(raw):
    out = []
    for ln in raw.splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except ValueError:
                out.append({"_raw": ln[:200]})
    return out


class V:
    """One check: what it found wrong, what it noted, and why it was skipped."""

    def __init__(self, name):
        self.name, self.errs, self.notes, self.gaps, self.skipped = name, [], [], [], None

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)

    def get(self, obj, path):
        cur = obj
        for k in path.split(".") if path else []:
            if isinstance(cur, list):
                try:
                    cur = cur[int(k)]
                except (ValueError, IndexError):
                    return False, None
            elif isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return False, None
        return True, cur

    def req(self, obj, path, typ=None, enum=None, eq=None):
        ok, v = self.get(obj, path)
        if not ok:
            self.errs.append(f"missing '{path}'")
            return None
        return self._value(path, v, typ, enum, eq)

    def opt(self, obj, path, typ=None, enum=None):
        """A field the published API may leave out; checked when present."""
        ok, v = self.get(obj, path)
        return self._value(path, v, typ, enum, None) if ok else None

    def gap(self, obj, path, where=""):
        """A field the published API has and superfluid documents it does not
        serve: noted while it is missing, not a failure."""
        ok, _ = self.get(obj, path)
        name = f"{where} {path}".strip()
        if not ok and name not in self.gaps:
            self.gaps.append(name)

    def check(self, cond, message):
        if not cond:
            self.errs.append(message)

    def _value(self, path, v, typ, enum, eq):
        if typ and (not isinstance(v, typ) or (isinstance(v, bool) and bool not in _types(typ))):
            names = "/".join(t.__name__ for t in _types(typ))
            self.errs.append(f"'{path}' is {type(v).__name__}, want {names}")
            return None
        if enum is not None and v not in enum:
            self.errs.append(f"'{path}'={v!r} not in {enum}")
        if eq is not None and v != eq:
            self.errs.append(f"'{path}'={v!r} != {eq!r}")
        return v

    def skip(self, why):
        self.skipped = why
        return self

    def status(self, got, want, body):
        if got != want:
            self.errs.append(f"HTTP {got}, want {want}: {json.dumps(body)[:200]}")
        return got == want


def _types(typ):
    return typ if isinstance(typ, tuple) else (typ,)


def parses_as_json(v, path, text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        v.errs.append(f"'{path}' is not JSON: {str(text)[:120]!r}")
        return None


def openai_error(v, status, j, want_status):
    if v.status(status, want_status, j):
        v.req(j, "error.message", str)
        v.req(j, "error.type", str)
        v.req(j, "error.param", OPT_STR)
        v.req(j, "error.code", (str, int, type(None)))


# ---- OpenAI ------------------------------------------------------------

def check_chat(j):
    v = V("chat.completion")
    v.req(j, "object", str, eq="chat.completion")
    v.req(j, "id", str), v.req(j, "created", int), v.req(j, "model", str)
    v.req(j, "choices.0.index", int)
    v.req(j, "choices.0.message.role", str, eq="assistant")
    v.req(j, "choices.0.message.content", OPT_STR)
    v.req(j, "choices.0.finish_reason", str, enum=["stop", "length", "tool_calls", "content_filter", "function_call"])
    v.req(j, "usage.prompt_tokens", int), v.req(j, "usage.completion_tokens", int), v.req(j, "usage.total_tokens", int)
    v.opt(j, "usage.prompt_tokens_details.cached_tokens", int)
    return v


def check_chat_logprobs(j):
    v = V("chat logprobs")
    v.req(j, "choices.0.logprobs.content.0.token", str)
    v.req(j, "choices.0.logprobs.content.0.logprob", NUM)
    v.req(j, "choices.0.logprobs.content.0.bytes", list)
    v.req(j, "choices.0.logprobs.content.0.top_logprobs", list)
    v.req(j, "choices.0.logprobs.content.0.top_logprobs.0.token", str)
    v.req(j, "choices.0.logprobs.content.0.top_logprobs.0.logprob", NUM)
    return v


def check_chat_stream(events, done):
    """Every chunk of a streamed chat completion, not only the first."""
    v = V("chat.completion.chunk (every chunk)")
    v.check(done, "the stream did not end with [DONE]")
    v.check(len(events) >= 2, f"{len(events)} chunks")
    chunks = [d for _, d in events]
    for i, c in enumerate(chunks):
        v.req(c, "object", str, eq="chat.completion.chunk")
        v.req(c, "id", str), v.req(c, "created", int), v.req(c, "model", str)
        if c.get("choices"):
            v.req(c, "choices.0.index", int)
            v.req(c, "choices.0.delta", dict)
            fr = v.req(c, "choices.0.finish_reason", OPT_STR)
            if fr is not None:
                v.check(fr in ["stop", "length", "tool_calls", "content_filter", "function_call"], f"chunk {i}: finish_reason {fr!r}")
    if chunks:
        v.req(chunks[0], "choices.0.delta.role", str, eq="assistant")
    finishing = [c for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
    v.check(len(finishing) == 1, f"{len(finishing)} chunks carry a finish_reason, want 1")
    return v


def check_stream_usage(events, done):
    """`stream_options.include_usage`: one more chunk before [DONE] carries the
    usage, with an empty `choices`."""
    v = V("chat stream usage (include_usage)")
    v.check(done, "the stream did not end with [DONE]")
    if not events:
        v.errs.append("no chunks")
        return v
    last = events[-1][1]
    v.req(last, "usage.prompt_tokens", int), v.req(last, "usage.completion_tokens", int), v.req(last, "usage.total_tokens", int)
    v.check(last.get("choices") == [], f"the usage chunk's 'choices' is {json.dumps(last.get('choices'))[:120]}, want []")
    return v


def check_tool_call(j):
    v = V("chat tool call (tool_choice required)")
    v.req(j, "choices.0.finish_reason", str, eq="tool_calls")
    v.req(j, "choices.0.message.role", str, eq="assistant")
    v.req(j, "choices.0.message.content", OPT_STR)
    v.req(j, "choices.0.message.tool_calls.0.id", str)
    v.req(j, "choices.0.message.tool_calls.0.type", str, eq="function")
    v.req(j, "choices.0.message.tool_calls.0.function.name", str, eq=WEATHER["name"])
    args = v.req(j, "choices.0.message.tool_calls.0.function.arguments", str)
    if args is not None:
        parsed = parses_as_json(v, "function.arguments", args)
        v.check(isinstance(parsed, dict) and isinstance(parsed.get("city"), str), f"arguments {args!r} hold no 'city' string")
    return v


def check_tool_call_stream(events, done):
    v = V("chat tool call streamed (named tool_choice)")
    v.check(done, "the stream did not end with [DONE]")
    calls = [c for _, d in events for c in (d.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or []]
    v.check(bool(calls), "no tool_calls delta")
    if calls:
        v.req(calls[0], "index", int)
        v.req(calls[0], "id", str)
        v.req(calls[0], "type", str, eq="function")
        v.req(calls[0], "function.name", str, eq=WEATHER["name"])
        for i, c in enumerate(calls):
            v.req(c, "index", int)
            v.opt(c, "function.arguments", str)
        text = "".join((c.get("function") or {}).get("arguments") or "" for c in calls)
        parsed = parses_as_json(v, "the arguments' fragments", text)
        v.check(isinstance(parsed, dict) and isinstance(parsed.get("city"), str), f"arguments {text!r} hold no 'city' string")
    reasons = [d["choices"][0].get("finish_reason") for _, d in events if d.get("choices")]
    v.check("tool_calls" in reasons, f"finish_reason {[r for r in reasons if r]}, want tool_calls")
    return v


def check_structured(j, name):
    v = V(name)
    v.req(j, "choices.0.finish_reason", str, enum=["stop", "length"])
    content = v.req(j, "choices.0.message.content", str)
    if content is not None:
        parsed = parses_as_json(v, "message.content", content)
        v.check(isinstance(parsed, dict), f"content {content[:120]!r} is not a JSON object")
        if name.endswith("json_schema") and isinstance(parsed, dict):
            v.check(isinstance(parsed.get("name"), str) and set(parsed) == {"name"}, f"{parsed} does not match the schema")
    return v


def check_completion(j):
    v = V("text_completion")
    v.req(j, "object", str, eq="text_completion")
    v.req(j, "id", str), v.req(j, "created", int), v.req(j, "model", str)
    v.req(j, "choices.0.text", str), v.req(j, "choices.0.index", int)
    v.req(j, "choices.0.finish_reason", OPT_STR)
    v.req(j, "usage.total_tokens", int)
    return v


def check_completion_stream(events, done):
    v = V("text_completion streamed")
    v.check(done, "the stream did not end with [DONE]")
    v.check(bool(events), "no chunks")
    for _, c in events:
        v.req(c, "object", str, eq="text_completion")
        v.req(c, "id", str), v.req(c, "created", int), v.req(c, "model", str)
        if c.get("choices"):
            v.req(c, "choices.0.text", str), v.req(c, "choices.0.index", int)
            v.req(c, "choices.0.finish_reason", OPT_STR)
    return v


def check_completion_logprobs(j):
    v = V("completion logprobs")
    v.req(j, "choices.0.logprobs.tokens", list)
    v.req(j, "choices.0.logprobs.token_logprobs", list)
    v.req(j, "choices.0.logprobs.top_logprobs", list)
    v.req(j, "choices.0.logprobs.text_offset", list)
    return v


def check_models_list(j):
    v = V("models list")
    v.req(j, "object", str, eq="list"), v.req(j, "data.0.id", str)
    v.req(j, "data.0.object", str, eq="model"), v.req(j, "data.0.created", int), v.req(j, "data.0.owned_by", str)
    return v


def check_model_obj(j):
    v = V("model retrieve")
    v.req(j, "id", str), v.req(j, "object", str, eq="model"), v.req(j, "created", int), v.req(j, "owned_by", str)
    return v


def check_file(v, j, prefix=""):
    v.req(j, prefix + "id", str), v.req(j, prefix + "object", str, eq="file")
    v.req(j, prefix + "bytes", int), v.req(j, prefix + "created_at", int)
    v.req(j, prefix + "filename", str), v.req(j, prefix + "purpose", str)
    v.opt(j, prefix + "status", str, enum=["uploaded", "processed", "error"])


BATCH_STATUS = ["validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"]


def check_batch(v, j, prefix=""):
    v.req(j, prefix + "id", str), v.req(j, prefix + "object", str, eq="batch")
    v.req(j, prefix + "endpoint", str), v.req(j, prefix + "input_file_id", str)
    v.req(j, prefix + "completion_window", str), v.req(j, prefix + "created_at", int)
    v.req(j, prefix + "status", str, enum=BATCH_STATUS)
    v.opt(j, prefix + "output_file_id", OPT_STR), v.opt(j, prefix + "error_file_id", OPT_STR)
    v.opt(j, prefix + "completed_at", (int, type(None)))
    for k in ["total", "completed", "failed"]:
        v.opt(j, prefix + "request_counts." + k, int)
    ok, err_file = v.get(j, prefix + "error_file_id")
    if ok and err_file == "":
        v.note("'error_file_id' is \"\" where the published API has null when there is none")
    ok, done_at = v.get(j, prefix + "completed_at")
    if ok and done_at == 0:
        v.note("'completed_at' is 0 where the published API has null until the batch completes")


def files_and_batches(s, m, checks):
    line = {"custom_id": "c1", "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": m, "messages": [{"role": "user", "content": "/no_think Say hello."}], "max_tokens": 4}}
    content = (json.dumps(line) + "\n").encode()
    v, b = V("files: upload, list, retrieve, content, delete"), V("batches: create, retrieve, list, output")
    checks += [v, b]
    status, f = s.upload("/v1/files", "batch", "compliance.jsonl", content)
    if not v.status(status, 200, f):
        b.skip("the input file did not upload")
        return
    check_file(v, f)
    fid = f.get("id", "")
    _, listing = s.json("GET", "/v1/files")
    v.req(listing, "object", str, eq="list")
    v.check(any(x.get("id") == fid for x in listing.get("data") or []), "the upload is not listed")
    _, got = s.json("GET", f"/v1/files/{fid}")
    check_file(v, got)
    status, text = s.call("GET", f"/v1/files/{fid}/content")
    v.check(status == 200 and text.encode() == content, f"content: HTTP {status}, {len(text)} bytes, want the {len(content)} uploaded")

    status, batch = s.json("POST", "/v1/batches", {"input_file_id": fid, "endpoint": "/v1/chat/completions", "completion_window": "24h"})
    if b.status(status, 200, batch):
        check_batch(b, batch)
        deadline = time.time() + 120
        while batch.get("status") not in ("completed", "failed", "expired", "cancelled") and time.time() < deadline:
            time.sleep(0.5)
            _, batch = s.json("GET", f"/v1/batches/{batch.get('id')}")
        check_batch(b, batch)
        b.check(batch.get("status") == "completed", f"status {batch.get('status')!r} after 120 s")
        _, listing = s.json("GET", "/v1/batches")
        b.req(listing, "object", str, eq="list")
        if listing.get("data"):
            check_batch(b, listing, "data.0.")
        out = batch.get("output_file_id")
        if out:
            status, text = s.call("GET", f"/v1/files/{out}/content")
            rows = ndjson(text) if status == 200 else []
            b.check(len(rows) == 1, f"the output file holds {len(rows)} lines, want 1")
            for r in rows:
                b.req(r, "id", str), b.req(r, "custom_id", str, eq="c1")
                b.req(r, "response.status_code", int, eq=200)
                b.req(r, "response.body.object", str, eq="chat.completion")
                b.req(r, "error", type(None))

    status, gone = s.json("DELETE", f"/v1/files/{fid}")
    if v.status(status, 200, gone):
        v.req(gone, "id", str, eq=fid), v.req(gone, "object", str, eq="file"), v.req(gone, "deleted", bool, eq=True)


def openai(s, m, checks):
    no_think = {"chat_template_kwargs": {"enable_thinking": False}}
    chat = lambda **k: {"model": m, "messages": [{"role": "user", "content": "/no_think Say hello briefly."}], "max_tokens": 16,
                        "temperature": 0, **no_think, **k}
    comp = lambda **k: {"model": m, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0, **k}
    tool = {"type": "function", "function": WEATHER}
    ask = [{"role": "user", "content": "/no_think What is the weather in Paris? Use the tool."}]

    _, j = s.json("POST", "/v1/chat/completions", chat()); checks.append(check_chat(j))
    _, j = s.json("POST", "/v1/chat/completions", chat(logprobs=True, top_logprobs=2)); checks.append(check_chat_logprobs(j))
    _, raw = s.call("POST", "/v1/chat/completions", chat(stream=True)); checks.append(check_chat_stream(*sse(raw)))
    _, raw = s.call("POST", "/v1/chat/completions", chat(stream=True, stream_options={"include_usage": True}))
    checks.append(check_stream_usage(*sse(raw)))
    _, j = s.json("POST", "/v1/chat/completions", chat(messages=ask, tools=[tool], tool_choice="required", max_tokens=96))
    checks.append(check_tool_call(j))
    named = {"type": "function", "function": {"name": WEATHER["name"]}}
    _, raw = s.call("POST", "/v1/chat/completions", chat(messages=ask, tools=[tool], tool_choice=named, max_tokens=96, stream=True))
    checks.append(check_tool_call_stream(*sse(raw)))
    planet = [{"role": "user", "content": "/no_think Name one planet of the solar system."}]
    schema = {"type": "json_schema", "json_schema": {"name": "planet", "strict": True, "schema": PLANET}}
    _, j = s.json("POST", "/v1/chat/completions", chat(messages=planet, response_format=schema, max_tokens=48))
    checks.append(check_structured(j, "structured output: json_schema"))
    _, j = s.json("POST", "/v1/chat/completions", chat(messages=planet, response_format={"type": "json_object"}, max_tokens=48))
    checks.append(check_structured(j, "structured output: json_object"))
    _, j = s.json("POST", "/v1/completions", comp()); checks.append(check_completion(j))
    _, raw = s.call("POST", "/v1/completions", comp(stream=True)); checks.append(check_completion_stream(*sse(raw)))
    _, j = s.json("POST", "/v1/completions", comp(logprobs=2)); checks.append(check_completion_logprobs(j))
    _, j = s.json("GET", "/v1/models"); checks.append(check_models_list(j))
    _, j = s.json("GET", f"/v1/models/{m}"); checks.append(check_model_obj(j))
    v = V("errors: unknown model")
    status, j = s.json("POST", "/v1/chat/completions", chat(model="no-such-model"))
    openai_error(v, status, j, 404)
    checks.append(v)
    files_and_batches(s, m, checks)


# ---- Anthropic ---------------------------------------------------------

STOP_REASONS = ["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal", None]
BLOCKS = ["text", "thinking", "redacted_thinking", "tool_use"]
DELTAS = ["text_delta", "thinking_delta", "signature_delta", "input_json_delta"]


def check_anthropic_message(j, name="anthropic message"):
    v = V(name)
    v.req(j, "id", str), v.req(j, "type", str, eq="message"), v.req(j, "role", str, eq="assistant")
    v.req(j, "content", list), v.req(j, "content.0.type", str, enum=BLOCKS)
    v.req(j, "model", str)
    v.req(j, "stop_reason", OPT_STR, enum=STOP_REASONS)
    v.req(j, "usage.input_tokens", int), v.req(j, "usage.output_tokens", int)
    return v


def check_anthropic_stream(events):
    v = V("anthropic streamed events")
    names = [n or d.get("type") for n, d in events]
    need = ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    for n in need:
        v.check(n in names, f"missing event '{n}' (got {names[:8]})")
    if names:
        v.check(names[0] == "message_start" and names[-1] == "message_stop", f"the stream runs {names[0]} .. {names[-1]}")
    input_tokens = final_input = None
    for n, d in events:
        kind = n or d.get("type")
        v.check(d.get("type") == kind, f"event '{kind}' carries type {d.get('type')!r}")
        if kind == "message_start":
            v.req(d, "message.id", str), v.req(d, "message.type", str, eq="message"), v.req(d, "message.role", str, eq="assistant")
            v.req(d, "message.content", list), v.req(d, "message.model", str)
            input_tokens = v.req(d, "message.usage.input_tokens", int)
            v.req(d, "message.usage.output_tokens", int)
        elif kind == "content_block_start":
            v.req(d, "index", int), v.req(d, "content_block.type", str, enum=BLOCKS)
        elif kind == "content_block_delta":
            v.req(d, "index", int), v.req(d, "delta.type", str, enum=DELTAS)
        elif kind == "content_block_stop":
            v.req(d, "index", int)
        elif kind == "message_delta":
            v.req(d, "delta.stop_reason", OPT_STR, enum=STOP_REASONS)
            v.req(d, "usage.output_tokens", int)
            final_input = v.opt(d, "usage.input_tokens", int)
    if input_tokens == 0 and final_input:
        v.errs.append(f"message_start reports 0 input tokens; message_delta reports {final_input}")
    return v


def anthropic_error(v, status, j, want_status, what):
    if not v.status(status, want_status, j):
        return
    if j.get("type") != "error" or not isinstance(j.get("error"), dict):
        v.errs.append(f"{what}: not Anthropic's error shape: {json.dumps(j)[:160]}")
        return
    v.req(j, "error.type", str), v.req(j, "error.message", str)


def anthropic(s, m, checks):
    msg = lambda **k: {"model": m, "messages": [{"role": "user", "content": "/no_think Say hello."}], "max_tokens": 16, **k}
    _, j = s.json("POST", "/v1/messages", msg()); checks.append(check_anthropic_message(j))
    _, j = s.json("POST", "/v1/messages", msg(system="You answer in one word.")); checks.append(check_anthropic_message(j, "anthropic message with a system prompt"))
    _, raw = s.call("POST", "/v1/messages", msg(stream=True)); checks.append(check_anthropic_stream(sse(raw)[0]))

    tool = {"name": WEATHER["name"], "description": WEATHER["description"], "input_schema": WEATHER["parameters"]}
    ask = [{"role": "user", "content": "/no_think What is the weather in Paris? Use the tool."}]
    _, j = s.json("POST", "/v1/messages", msg(messages=ask, tools=[tool], tool_choice={"type": "tool", "name": tool["name"]}, max_tokens=128))
    v = check_anthropic_message(j, "anthropic tool use (tool_choice tool)")
    v.req(j, "stop_reason", str, eq="tool_use")
    uses = [b for b in j.get("content") or [] if b.get("type") == "tool_use"]
    v.check(len(uses) == 1, f"{len(uses)} tool_use blocks, want 1")
    for b in uses:
        v.req(b, "id", str), v.req(b, "name", str, eq=tool["name"]), v.req(b, "input", dict)
        v.check(isinstance((b.get("input") or {}).get("city"), str), f"input {b.get('input')} holds no 'city' string")
    checks.append(v)

    v = V("anthropic thinking: disabled")
    plain = [{"role": "user", "content": "Say hello."}]
    status, j = s.json("POST", "/v1/messages", msg(messages=plain, thinking={"type": "disabled"}, max_tokens=48))
    if v.status(status, 200, j):
        kinds = [b.get("type") for b in j.get("content") or []]
        v.check("thinking" not in kinds and "redacted_thinking" not in kinds, f"content blocks {kinds}: thinking was not disabled")
    checks.append(v)

    v = V("anthropic errors: missing max_tokens, unknown model")
    status, j = s.json("POST", "/v1/messages", {"model": m, "messages": [{"role": "user", "content": "hi"}]})
    anthropic_error(v, status, j, 400, "missing max_tokens")
    status, j = s.json("POST", "/v1/messages", msg(model="no-such-model"))
    anthropic_error(v, status, j, 404, "unknown model")
    checks.append(v)


# ---- Ollama ------------------------------------------------------------

OLLAMA_DONE = ["stop", "length", "load"]


def check_ollama_record(v, r, final, kind):
    v.req(r, "model", str), v.req(r, "created_at", str), v.req(r, "done", bool, eq=final)
    if kind == "chat":
        v.req(r, "message.role", str, eq="assistant"), v.req(r, "message.content", str)
        v.opt(r, "message.thinking", str)
    else:
        v.req(r, "response", str)
        v.opt(r, "thinking", str)
    if final:
        v.req(r, "done_reason", str, enum=OLLAMA_DONE)
        v.req(r, "total_duration", int), v.req(r, "prompt_eval_count", int), v.req(r, "eval_count", int)
        for k in ["load_duration", "prompt_eval_duration", "eval_duration"]:
            v.gap(r, k)


def ollama(s, m, checks):
    v = V("ollama discovery: /, /api/version, /api/tags, /api/ps, /api/show")
    status, text = s.call("GET", "/")
    v.check(status == 200 and "running" in text, f"GET /: HTTP {status} {text[:80]!r}")
    _, j = s.json("GET", "/api/version"); v.req(j, "version", str)
    _, j = s.json("GET", "/api/tags")
    v.req(j, "models", list), v.req(j, "models.0.name", str), v.req(j, "models.0.model", str)
    v.check(any(x.get("model") == m for x in j.get("models") or []), f"/api/tags does not list {m}")
    for k in ["modified_at", "size", "digest", "details"]:
        v.gap(j, f"models.0.{k}", "/api/tags")
    _, j = s.json("GET", "/api/ps")
    v.req(j, "models", list)
    if j.get("models"):
        v.req(j, "models.0.name", str), v.req(j, "models.0.model", str), v.opt(j, "models.0.context_length", int)
        for k in ["size", "digest", "expires_at", "size_vram"]:
            v.gap(j, f"models.0.{k}", "/api/ps")
    _, j = s.json("POST", "/api/show", {"model": m})
    v.req(j, "details", dict), v.req(j, "model_info", dict)
    v.opt(j, "capabilities", list), v.opt(j, "template", str)
    v.gap(j, "parameters", "/api/show")
    checks.append(v)

    opts = {"num_predict": 8, "temperature": 0}
    gen = {"model": m, "prompt": "Say hello.", "think": False, "options": opts}
    chat = {"model": m, "messages": [{"role": "user", "content": "Say hello."}], "think": False, "options": opts}
    for path, body, kind in [("/api/generate", gen, "generate"), ("/api/chat", chat, "chat")]:
        v = V(f"ollama {kind}")
        status, j = s.json("POST", path, {**body, "stream": False})
        if v.status(status, 200, j):
            check_ollama_record(v, j, True, kind)
        checks.append(v)
        v = V(f"ollama {kind} streamed (NDJSON)")
        status, raw = s.call("POST", path, body)
        rows = ndjson(raw) if status == 200 else []
        v.check(len(rows) >= 2, f"HTTP {status}, {len(rows)} records")
        for i, r in enumerate(rows):
            check_ollama_record(v, r, i == len(rows) - 1, kind)
        checks.append(v)

    # Ollama has no tool_choice: a model that answers in text instead says
    # nothing about the shape of a call.
    v = V("ollama chat tools")
    tool = {"type": "function", "function": WEATHER}
    ask = {"model": m, "stream": False, "think": False, "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
           "tools": [tool], "options": {"temperature": 0}}
    status, j = s.json("POST", "/api/chat", ask)
    if v.status(status, 200, j):
        if not (j.get("message") or {}).get("tool_calls"):
            v.skip(f"the model answered without calling the tool: {str((j.get('message') or {}).get('content'))[:80]!r}")
        else:
            v.req(j, "message.tool_calls.0.function.name", str, eq=WEATHER["name"])
            args = v.req(j, "message.tool_calls.0.function.arguments", dict)
            v.check(isinstance((args or {}).get("city"), str), f"arguments {args} hold no 'city' string")
            v.req(j, "done_reason", str, enum=OLLAMA_DONE)
    checks.append(v)

    v = V("ollama chat format (JSON schema)")
    ask = {"model": m, "stream": False, "think": False, "messages": [{"role": "user", "content": "Name one planet of the solar system."}],
           "format": PLANET, "options": {"temperature": 0, "num_predict": 48}}
    status, j = s.json("POST", "/api/chat", ask)
    if v.status(status, 200, j):
        content = v.req(j, "message.content", str)
        parsed = parses_as_json(v, "message.content", content) if content is not None else None
        v.check(isinstance(parsed, dict) and isinstance(parsed.get("name"), str), f"content {content!r} does not match the schema")
    checks.append(v)

    v = V("ollama errors: unknown model, unknown option, model management")
    for path, body, want in [
        ("/api/chat", {"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]}, 404),
        ("/api/chat", {**chat, "options": {"mirostat": 1}}, 400),
        ("/api/pull", {"model": m}, 501),
    ]:
        status, j = s.json("POST", path, body)
        if v.status(status, want, j):
            v.req(j, "error", str)
    checks.append(v)


# ---- llama.cpp-server routes ------------------------------------------

def server(s, m, checks):
    v = V("server: /health, /props, /slots, /v1/tokenize, /metrics")
    status, j = s.json("GET", "/health")
    if v.status(status, 200, j):
        v.req(j, "status", str, eq="ok")
    status, j = s.json("GET", "/props")
    if v.status(status, 200, j):
        v.req(j, "default_generation_settings.n_ctx", int)
        v.req(j, "total_slots", int)
        v.opt(j, "chat_template", str)
    status, j = s.json("GET", "/slots")
    if v.status(status, 200, j):
        v.check(isinstance(j, list) and bool(j), f"/slots is {type(j).__name__}, want a non-empty list")
        if isinstance(j, list) and j:
            v.req(j, "0.id", int), v.req(j, "0.is_processing", bool)
    status, j = s.json("POST", "/v1/tokenize", {"model": m, "content": "hello world"})
    if v.status(status, 200, j):
        tokens = v.req(j, "tokens", list)
        v.check(bool(tokens) and all(isinstance(t, int) for t in tokens or []), f"tokens {tokens}")
    status, text = s.call("GET", "/metrics")
    v.check(status == 200 and "# TYPE " in text and "superfluid_" in text, f"/metrics: HTTP {status}, not Prometheus text")
    checks.append(v)


# ---- embeddings --------------------------------------------------------

def refused_embeddings(status, j):
    text = json.dumps(j).lower()
    return status in (400, 501) and "embedding" in text and ("support" in text or "serve" in text)


def embed(s, m, checks):
    v = V("embeddings")
    status, j = s.json("POST", "/v1/embeddings", {"model": m, "input": "hello"})
    if refused_embeddings(status, j):
        checks.append(v.skip(f"the model serves no embeddings: {json.dumps(j)[:120]}"))
        return
    if v.status(status, 200, j):
        v.req(j, "object", str, eq="list")
        v.req(j, "data.0.object", str, eq="embedding"), v.req(j, "data.0.embedding", list), v.req(j, "data.0.index", int)
        v.req(j, "model", str), v.req(j, "usage.prompt_tokens", int), v.req(j, "usage.total_tokens", int)
    checks.append(v)
    v = V("ollama /api/embed, /api/embeddings")
    status, j = s.json("POST", "/api/embed", {"model": m, "input": ["first", "second"]})
    if v.status(status, 200, j):
        v.req(j, "model", str), v.req(j, "embeddings", list), v.req(j, "embeddings.1", list), v.req(j, "embeddings.0.0", NUM)
        v.opt(j, "total_duration", int), v.opt(j, "prompt_eval_count", int)
    status, j = s.json("POST", "/api/embeddings", {"model": m, "prompt": "first"})
    if v.status(status, 200, j):
        v.req(j, "embedding", list), v.req(j, "embedding.0", NUM)
    checks.append(v)


# ---- report ------------------------------------------------------------

def report(title, checks):
    print(f"\n=== {title} ===")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for v in checks:
        state = "SKIP" if v.skipped else ("FAIL" if v.errs else "PASS")
        counts[state] += 1
        print(f"  {'✅' if state == 'PASS' else '❌' if state == 'FAIL' else '⏭️ '} {v.name}")
        if v.skipped:
            print(f"       • {v.skipped}")
        for e in v.errs[:8]:
            print(f"       • {e}")
        if len(v.errs) > 8:
            print(f"       • ... and {len(v.errs) - 8} more")
        for n in v.notes:
            print(f"       ℹ {n}")
        if v.gaps:
            print(f"       ℹ not served, as documented: {', '.join(v.gaps)}")
    return counts


MODES = {
    "openai": ("OpenAI API compliance", openai),
    "anthropic": ("Anthropic Messages API compliance", anthropic),
    "ollama": ("Ollama API compliance", ollama),
    "server": ("llama.cpp-server routes", server),
    "embed": ("Embeddings compliance", embed),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", help="model id (default: the first /v1/models lists)")
    ap.add_argument("--mode", choices=[*MODES, "all"], default="all")
    ap.add_argument("--key", help="API key, for a server started with --api-key")
    a = ap.parse_args()
    s = Server(a.url, a.key)
    m = a.model
    if not m:
        status, j = s.json("GET", "/v1/models")
        if status != 200 or not j.get("data"):
            sys.exit(f"GET /v1/models: HTTP {status}; name a model with --model")
        m = j["data"][0]["id"]
    total = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for mode in (["openai", "anthropic", "ollama", "server"] if a.mode == "all" else [a.mode]):
        title, run = MODES[mode]
        checks = []
        run(s, m, checks)
        for k, n in report(f"{title} ({m})", checks).items():
            total[k] += n
    print(f"\nPASS {total['PASS']}   FAIL {total['FAIL']}   SKIP {total['SKIP']}")
    sys.exit(1 if total["FAIL"] else 0)


if __name__ == "__main__":
    main()
