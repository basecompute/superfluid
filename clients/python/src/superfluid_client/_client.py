"""A client for superfluid's native session API over the `--web` transport.

Requests are JSON over `POST /web/rpc`; generation streams over `POST /web/stream`
as server-sent events. Only the standard library is used.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import urllib.parse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

SYSTEM, USER, ASSISTANT, TOOL = 0, 1, 2, 3
TEXT, REASONING, TOOL_CALL = 0, 1, 2

DEFAULT_URL = "http://127.0.0.1:8460"

_FINISH = {0: "none", 1: "stop", 2: "length", 3: "grammar", 4: "cancelled", 5: "error"}
_ROLES = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


class SuperfluidError(Exception):
    """The server refused a request; `message` is its reason."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status = status


class SessionBusy(SuperfluidError):
    """The session has a generation in flight."""


def _error(message: str) -> SuperfluidError:
    return SessionBusy(message) if " is generating;" in message else SuperfluidError(message)


@dataclass(frozen=True)
class Sampling:
    """A session's sampling parameters. `temperature=0` is greedy; `top_k=0` disables top-k.

    The random draw for each token is a function of the seed and the token's position in the
    session, so forks that share a seed share their random draws. Give each fork its own seed
    to diverge.
    """

    temperature: float = 0.7
    top_p: float = 0.95
    min_p: float = 0.0
    top_k: int = 20
    seed: int = field(default_factory=lambda: random.getrandbits(63))

    @classmethod
    def greedy(cls) -> "Sampling":
        return cls(temperature=0.0, top_p=1.0, min_p=0.0, top_k=0, seed=0)

    def with_seed(self, seed: int) -> "Sampling":
        return replace(self, seed=seed)

    def to_wire(self) -> dict:
        return {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "min_p": float(self.min_p),
            "top_k": int(self.top_k),
            "seed": int(self.seed),
        }

    @classmethod
    def from_wire(cls, v: dict) -> "Sampling":
        return cls(v["temperature"], v["top_p"], v["min_p"], v["top_k"], v["seed"])


@dataclass(frozen=True)
class Event:
    """One committed event of a session's log."""

    id: int
    kind: str
    data: dict
    ts_ms: int
    epoch: int

    @classmethod
    def from_wire(cls, v: dict) -> "Event":
        body = v["body"]
        if isinstance(body, str):
            kind, data = body, {}
        else:
            ((kind, data),) = body.items()
            data = data if isinstance(data, dict) else {"value": data}
        return cls(v["event_id"], kind, data, v["ts_unix_ms"], v["epoch"])

    @property
    def text(self) -> Optional[str]:
        return self.data.get("text")

    @property
    def channel(self) -> Optional[int]:
        return self.data.get("channel")


@dataclass(frozen=True)
class ToolCall:
    id: int
    name: str
    arguments: str

    def json(self) -> Any:
        return json.loads(self.arguments)


@dataclass(frozen=True)
class Chunk:
    """A piece of streamed output. `kind` is "text", "reasoning" or "tool_call"."""

    kind: str
    text: str


@dataclass
class Reply:
    """The outcome of one generation."""

    text: str
    reasoning: str
    tool_calls: list
    finish: str
    tokens: int
    warm_prefix: int
    events: list

    def __str__(self) -> str:
        return self.text


def _kind(channel: int) -> str:
    return {TEXT: "text", REASONING: "reasoning", TOOL_CALL: "tool_call"}.get(channel, "text")


def _default_token() -> str:
    tok = os.environ.get("SUPERFLUID_WEB_TOKEN")
    if tok:
        return tok.strip()
    home = Path(os.environ.get("SUPERFLUID_HOME", Path.home() / ".superfluid"))
    path = home / "sessions" / "web-token"
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        raise SuperfluidError(
            f"no web token at {path}: start the server with `--web 127.0.0.1:<port>`, "
            "or pass token= / set SUPERFLUID_WEB_TOKEN"
        ) from None


class Client:
    """A connection to a server started with `--web 127.0.0.1:<port>`.

    The token defaults to `$SUPERFLUID_WEB_TOKEN`, else `$SUPERFLUID_HOME/sessions/web-token`
    (`~/.superfluid/sessions/web-token`). The URL defaults to `$SUPERFLUID_WEB_URL`, else
    http://127.0.0.1:8460.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        *,
        token_file: Optional[os.PathLike] = None,
        timeout: float = 600.0,
    ):
        url = url or os.environ.get("SUPERFLUID_WEB_URL") or DEFAULT_URL
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"expected an http://host:port URL, got {url!r}")
        self.url = url
        self._host = parts.hostname
        self._port = parts.port or 80
        if token is None and token_file is not None:
            token = Path(token_file).read_text().strip()
        self._token = token if token is not None else _default_token()
        self.timeout = timeout

    # transport

    def _headers(self) -> dict:
        return {
            "authorization": f"Bearer {self._token}",
            "x-superfluid-csrf": self._token,
            "content-type": "application/json",
        }

    def _post(self, path: str, body: dict) -> tuple:
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)
        try:
            conn.request("POST", path, body=json.dumps(body), headers=self._headers())
            resp = conn.getresponse()
        except BaseException:
            conn.close()
            raise
        if resp.status != 200:
            raw = resp.read().decode("utf-8", "replace")
            conn.close()
            try:
                message = json.loads(raw)["Err"]["message"]
            except (ValueError, KeyError, TypeError):
                message = raw or resp.reason
            raise SuperfluidError(f"{path}: {resp.status} {message}", resp.status)
        return conn, resp

    def call(self, verb: str, args: Optional[dict] = None) -> tuple:
        """Send one native request; returns `(response_variant, payload)`.

        `client.call("Fork", {"session": 1, "at_event": 3, "params": None})` is
        `("Created", {"session": 2})`. A refusal raises `SuperfluidError`.
        """
        conn, resp = self._post("/web/rpc", {verb: args})
        try:
            out = json.loads(resp.read())
        finally:
            conn.close()
        if isinstance(out, str):
            return out, None
        if not isinstance(out, dict) or len(out) != 1:
            raise SuperfluidError(f"{verb}: unexpected response {out!r}")
        ((variant, payload),) = out.items()
        if variant == "Err":
            message = payload["message"]
            raise _error(message)
        return variant, payload

    def _expect(self, want: str, verb: str, args: Optional[dict] = None) -> Any:
        variant, payload = self.call(verb, args)
        if variant != want:
            raise SuperfluidError(f"{verb}: expected {want}, got {variant}")
        return payload

    def _stream(self, session: int, max_tokens: int, provisional: bool) -> Iterator[dict]:
        conn, resp = self._post(
            "/web/stream",
            {"session": session, "max_tokens": max_tokens, "provisional": provisional},
        )
        try:
            for raw in resp:
                line = raw.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                data = line[5:].lstrip(" ")
                if data == "[DONE]":
                    return
                yield json.loads(data)
        finally:
            conn.close()

    # sessions

    def session(
        self,
        sampling: Optional[Sampling] = None,
        *,
        system: Optional[str] = None,
        tools: Optional[list] = None,
    ) -> "Session":
        """Create a session, optionally opening with a system message and tools; see `Session.declare`."""
        sampling = sampling or Sampling()
        created = self._expect("Created", "Create", {"parent": None, "params": sampling.to_wire()})
        s = Session(self, created["session"], sampling)
        if tools:
            s.declare(system=system, tools=tools)
        elif system is not None:
            s.system(system)
        return s

    def get(self, session_id: int) -> "Session":
        """A handle on an existing session, e.g. after a restart."""
        params = self.inspect(session_id)["params"]
        return Session(self, session_id, Sampling.from_wire(params))

    def inspect(self, session_id: int) -> dict:
        return self._expect("Inspection", "Inspect", {"session": session_id})["inspection"]

    def sessions(self, include_archived: bool = False) -> list:
        return self._expect("SessionList", "Sessions", {"include_archived": include_archived})["sessions"]

    def tree(self) -> list:
        """`(session, parent, fork_at)` for every live session."""
        return [tuple(n) for n in self._expect("Tree", "Tree")["nodes"]]

    def complete(self, prefix: str, suffix: str = "", max_tokens: int = 64, mode: int = 0) -> dict:
        """Fill-in-the-middle completion between `prefix` and `suffix`."""
        return self._expect(
            "Completion",
            "Complete",
            {"prefix": prefix, "suffix": suffix, "mode": mode, "max_tokens": max_tokens},
        )

    def metrics(self) -> str:
        return self._expect("Text", "Metrics")["text"]

    def set_background_rate(self, divisor: int) -> None:
        self._expect("Ack", "SetBackgroundRate", {"divisor": divisor})

    def health(self) -> bool:
        conn = http.client.HTTPConnection(self._host, self._port, timeout=5)
        try:
            conn.request("GET", "/web/health")
            return conn.getresponse().status == 200
        except OSError:
            return False
        finally:
            conn.close()


class Session:
    """A durable, append-only conversation on the server.

    Appends and generations extend it; `fork`, `rebase` and `trim` make new sessions that
    share its history by reference, leaving this one unchanged.
    """

    def __init__(self, client: Client, session_id: int, sampling: Sampling):
        self.client = client
        self.id = session_id
        self.sampling = sampling
        self._cursor = 0

    def __repr__(self) -> str:
        return f"Session({self.id})"

    def _commit(self, verb: str, args: dict) -> Event:
        payload = self.client._expect("Committed", verb, {"session": self.id, **args})
        event = Event.from_wire(payload["event"])
        self._cursor = max(self._cursor, event.id + 1)
        return event

    # appending

    def say(self, role: str, text: str) -> Event:
        """Append a message as `role` ("system", "user" or "assistant")."""
        return self._commit("AppendMessage", {"role": _ROLES[role], "text": text})

    def system(self, text: str) -> Event:
        return self.say("system", text)

    def user(self, text: str) -> Event:
        return self.say("user", text)

    def assistant(self, text: str) -> Event:
        return self.say("assistant", text)

    def declare(self, *, system: Optional[str] = None, tools: list = ()) -> Event:
        """Open the session with a system turn declaring `tools`, rendered as the chat template
        renders tools. Each tool is an OpenAI function tool, a bare function or an Anthropic tool,
        as a dict or a JSON string. Only a session with nothing appended yet takes one.
        """
        wire = [t if isinstance(t, str) else json.dumps(t) for t in tools]
        return self._commit("AppendSystem", {"text": system, "tools": wire})

    def append_tokens(self, tokens: list, text: Optional[str] = None) -> Event:
        """Append raw token ids with no chat scaffold."""
        return self._commit("Append", {"text": text, "span": list(tokens)})

    def tool_result(self, call: "ToolCall | int", content: str) -> Event:
        call_id = call.id if isinstance(call, ToolCall) else call
        return self._commit("AppendToolResult", {"call_id": call_id, "content": content})

    def image(self, blob: str, *, before: str = "", after: str = "", role: str = "user") -> Event:
        """Append an image by the hash `PutMedia` returned, with text around it."""
        return self._commit(
            "AppendImage",
            {"role": _ROLES[role], "blob": blob, "pre_text": before, "post_text": after},
        )

    # generating

    def stream(self, max_tokens: int = 1024, *, provisional: bool = True) -> "Stream":
        """Generate, yielding `Chunk`s as they arrive. `.reply` holds the outcome after.

        With `provisional`, text arrives token by token ahead of its commit; without it,
        only committed text arrives. Leaving the loop early cancels the generation.
        """
        return Stream(self, max_tokens, provisional)

    def generate(
        self, max_tokens: int = 1024, *, on_text: Optional[Callable[[Chunk], None]] = None
    ) -> Reply:
        """Generate the next turn and return it. `on_text` sees each chunk as it streams."""
        st = self.stream(max_tokens, provisional=on_text is not None)
        for chunk in st:
            if on_text is not None:
                on_text(chunk)
        return st.reply

    def cancel(self) -> bool:
        """Stop the generation in flight; False when there is none."""
        try:
            self.client._expect("Cancelling", "Cancel", {"session": self.id})
            return True
        except SuperfluidError as e:
            if "no generation in flight" in e.message:
                return False
            raise

    # branching

    @property
    def head(self) -> int:
        """The id the next committed event will take; `fork()` branches here by default."""
        for e in self.events(self._cursor):
            self._cursor = e.id + 1
        return self._cursor

    def fork(
        self,
        at: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        sampling: Optional[Sampling] = None,
    ) -> "Session":
        """A new session holding this one's events before `at` (default: all of them).

        Pass `seed` (or `sampling`) for a branch that draws differently. A fork with the
        parent's seed shares the parent's random draws, so it tends to retrace the parent's
        words; it is not guaranteed to, since the fork recomputes from cached history.
        """
        at = self.head if at is None else at
        if sampling is None and seed is not None:
            sampling = self.sampling.with_seed(seed)
        child = self.client._expect(
            "Created",
            "Fork",
            {"session": self.id, "at_event": at, "params": sampling.to_wire() if sampling else None},
        )["session"]
        return Session(self.client, child, sampling or self.sampling)

    def fork_after(self, event: "Event | int", **kw) -> "Session":
        """Fork keeping everything up to and including `event`."""
        event_id = event.id if isinstance(event, Event) else event
        return self.fork(event_id + 1, **kw)

    def rebase(self, edits: list, *, sampling: Optional[Sampling] = None) -> "Session":
        """A new session with events dropped or replaced; see `drop` and `replace`."""
        child = self.client._expect(
            "Created",
            "Rebase",
            {"session": self.id, "edits": edits, "params": sampling.to_wire() if sampling else None},
        )["session"]
        return Session(self.client, child, sampling or self.sampling)

    def edit(self, event: "Event | int", text: str, *, role: Optional[str] = None) -> "Session":
        """A new session with one message's text replaced and nothing after it."""
        e = event if isinstance(event, Event) else self.event(event)
        if e.kind != "Message":
            raise ValueError(f"event {e.id} is a {e.kind}, not a message")
        role_id = _ROLES[role] if role else e.data["role"]
        return self.rebase([replace_events(e.id, self.head, role_id, text)])

    def trim(self, to_event: int) -> "Session":
        """A new session holding this one's events before `to_event`."""
        child = self.client._expect("Created", "Trim", {"session": self.id, "to_event": to_event})["session"]
        return Session(self.client, child, self.sampling)

    # reading

    def events(self, cursor: int = 0) -> list:
        payload = self.client._expect("Events", "Read", {"session": self.id, "cursor": cursor})
        return [Event.from_wire(e) for e in payload["events"]]

    def event(self, event_id: int) -> Event:
        for e in self.events(event_id):
            if e.id == event_id:
                return e
        raise SuperfluidError(f"session {self.id} has no event {event_id}")

    def messages(self) -> list:
        """The conversation as `{"role", "content"[, "reasoning"]}` dicts, events folded."""
        out: list = []
        for e in self.events():
            if e.kind == "Message":
                role = {SYSTEM: "system", USER: "user", ASSISTANT: "assistant"}[e.data["role"]]
                out.append({"role": role, "content": e.data["text"], "event": e.id})
            elif e.kind == "Generated":
                if not out or out[-1]["role"] != "assistant" or out[-1].get("closed"):
                    out.append({"role": "assistant", "content": "", "event": e.id})
                key = "reasoning" if e.channel == REASONING else "content"
                out[-1][key] = out[-1].get(key, "") + e.text
                if e.data.get("finish"):
                    out[-1]["closed"] = True
            elif e.kind == "ToolUse":
                out.append({"role": "tool_call", "name": e.data["name"], "arguments": e.data["arguments"], "event": e.id})
            elif e.kind == "ToolResult":
                out.append({"role": "tool", "call_id": e.data["call_id"], "content": e.data["content"], "event": e.id})
        for m in out:
            m.pop("closed", None)
        return out

    def inspect(self) -> dict:
        return self.client.inspect(self.id)

    def open_tool_calls(self) -> list:
        calls = self.client._expect("ToolCalls", "OpenToolCalls", {"session": self.id})["calls"]
        return [ToolCall(i, n, a) for i, n, a in calls]

    # scheduling and metadata

    def pin(self, ttl_s: float) -> int:
        """Hold this session's prefix in the cache for `ttl_s` seconds; returns the deadline (ms)."""
        return self.client._expect("Pinned", "Pin", {"session": self.id, "ttl_ms": int(ttl_s * 1000)})[
            "deadline_unix_ms"
        ]

    def set_qos(self, qos_class: int, *, batch_invariant: bool = False) -> Event:
        return self._commit("SetQos", {"class": qos_class, "batch_invariant": batch_invariant})

    def set_meta(self, *, title: Optional[str] = None, archived: Optional[bool] = None) -> int:
        version = self.inspect()["summary"]["meta_version"]
        return self.client._expect(
            "Meta",
            "SetMeta",
            {"session": self.id, "expected_version": version, "title": title, "archived": archived},
        )["version"]

    def purge(self, *, cascade: bool = True) -> list:
        """Remove this session (and, with `cascade`, its forks) from view. Not content erasure."""
        generation = self.inspect()["summary"]["generation"]
        return self.client._expect(
            "Purged",
            "Purge",
            {"session": self.id, "expected_generation": generation, "mode": "Cascade" if cascade else "Reroot"},
        )["sessions"]


class Stream:
    """An iterator over one generation's `Chunk`s; `.reply` is set once it ends."""

    def __init__(self, session: Session, max_tokens: int, provisional: bool):
        self.session = session
        self.reply: Optional[Reply] = None
        self._frames = session.client._stream(session.id, max_tokens, provisional)
        self._provisional = provisional

    def __iter__(self) -> Iterator[Chunk]:
        committed: list = []
        done = None
        started = False
        try:
            for frame in self._frames:
                if "error" in frame:
                    raise _error(frame["error"])
                started = True
                if "provisional" in frame:
                    p = frame["provisional"]
                    if p["text"]:
                        yield Chunk(_kind(p["channel"]), p["text"])
                elif "committed" in frame:
                    e = Event.from_wire(frame["committed"])
                    committed.append(e)
                    self.session._cursor = max(self.session._cursor, e.id + 1)
                    if not self._provisional and e.kind == "Generated" and e.text:
                        yield Chunk(_kind(e.channel), e.text)
                elif "done" in frame:
                    done = frame["done"]
        finally:
            if started and done is None:
                self._stop()
            self._frames.close()
        if done is None:
            raise SuperfluidError("the stream ended without a summary")
        self.reply = _reply(committed, done)


    def _stop(self) -> None:
        """Cancel a generation left early and wait for it to end, so the session is idle."""
        try:
            self.session.cancel()
            for frame in self._frames:
                if "done" in frame or "error" in frame:
                    break
        except (SuperfluidError, OSError, ValueError):
            pass


def _reply(events: list, done: dict) -> Reply:
    text = "".join(e.text for e in events if e.kind == "Generated" and e.channel != REASONING)
    reasoning = "".join(e.text for e in events if e.kind == "Generated" and e.channel == REASONING)
    calls = [ToolCall(e.id, e.data["name"], e.data["arguments"]) for e in events if e.kind == "ToolUse"]
    return Reply(
        text=text,
        reasoning=reasoning,
        tool_calls=calls,
        finish=_FINISH.get(done["finish"], str(done["finish"])),
        tokens=done["tokens_generated"],
        warm_prefix=done["warm_prefix"],
        events=events,
    )


def drop_events(start: int, end: int) -> dict:
    """A rebase edit removing events `start` up to (not including) `end`."""
    return {"Drop": {"from": start, "to": end}}


def replace_events(start: int, end: int, role: int, text: str) -> dict:
    """A rebase edit replacing events `start..end` with one message."""
    return {"Replace": {"from": start, "to": end, "role": role, "text": text}}
