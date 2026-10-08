# Sessions and prefix caching

A session is an append-only log of committed events, each carrying the exact tokens it put on the model's input. The log is the source of truth; the KV cache, the prefix cache and park artifacts are views of it that can be rebuilt. Every record is fsynced before its event is delivered, so a daemon or worker crash loses nothing a client has seen.

HTTP requests each run as a fresh session. The [session API](../serving/session_api.md) lets a client keep, fork and resume sessions by id.

## Durability

| event | what happens |
|---|---|
| worker crash | in-flight requests fail with an error; the worker is respawned and sessions continue from their logs |
| daemon crash or restart | the log is replayed; a torn final record is truncated; sessions continue |
| complete record that fails its checksum | the daemon refuses to start rather than guess; move `wal.log` aside or use `--sessions-cache` |

The log lives at `<sessions>/wal.log` (default `~/.superfluid/sessions`). One server per sessions directory. Each additional model has its own log under `<sessions>/models/<id>/`.

`--sessions-cache` treats the log as a cache: each start keeps the previous log as `wal.log.prev` and begins a fresh one. It cannot be combined with `--park`.

## Prefix cache

A request whose prompt starts with tokens a finished request already processed is seeded from the cache instead of prefilled. The match is on token ids only, so changing tools or sampling settings does not flush it.

- Hits are reported as `usage.prompt_tokens_details.cached_tokens`, the `x-superfluid-warm` header, `cached=` in the access log, and `superfluid_warm_prefix_tokens_total`.
- A prefix reaches the cache when its request finishes. Requests that arrive together sharing at least 1024 prompt tokens with a nearly finished request wait for it and are seeded from it, instead of each prefilling the same prefix.
- Requests with images never use the cache.
- Under KV pressure, cached prefixes are evicted first (see [Scheduling](scheduling.md#memory-pressure)).
- On llama.cpp, a prefix that leaves the KV pool (for a lane's cells or sequence, or past the cache's share of a shared pool) is kept in host memory and imported back when a request reuses it; it is evicted only when that room is full too (see [Runtimes](../models/runtimes.md#llamacpp)).

Pin a prefix against eviction with an Anthropic `cache_control` breakpoint (see [Anthropic API](../serving/anthropic_api.md#prompt-caching-cache_control)) or a session `Pin`. Pins together hold at most `--pin-budget-pct` of the pool (default 50) and yield, oldest first, under pressure.

## Parking

`--park` seals a finishing session's KV to `<sessions>/park/<session>.park`; the next generation on that session id restores it instead of re-prefilling, across restarts.

| flag | default | meaning |
|---|---|---|
| `--park` | off | write park artifacts and resume from them |
| `--park-lossy` | off | lossy Q8 tier, about half the size; implies `--kv-bits 16` |
| `--park-budget-gb` | 20 | size bound per park directory; 0 is unbounded |

- Parking is off by default: sealing copies the whole KV synchronously at the end of every generation, which delays every lane beside it.
- Stateless HTTP requests never park at the end of a request; parking pays off for sessions resumed by id through the session API.
- An artifact covers a digest of its tokens: a rewritten history resumes cold, never wrong. A missing or stale artifact falls back to re-prefill.

## Forks and continuation

- **Fork** a session at any committed event: the child inherits the parent's events by reference.
- **Rebase** creates a branch with events dropped or replaced; **Trim** keeps a prefix. The parent is never rewritten.
- **Continue**: generating on a session after a user message or tool result opens a new assistant turn; after a reply cut by `max_tokens` or a cancel, it continues the open turn (inside a reasoning block, if that is where it stopped).
- Sampling is indexed by stream position, so a seeded session draws the same tokens whether or not it was cut.
- A session with a generation in flight is busy: a second generate or append is refused with `SessionBusy`. Forking at a committed point is allowed.

## The native session API

`--socket` (default `~/.superfluid/superfluid.sock`) serves the session API as postcard frames, and `--web` serves it as JSON over loopback HTTP and WebSocket. [Session API](../serving/session_api.md) has its requests, replies and routes.

## Command-line tools

```sh
superfluid session inspect 7                         # summary and one line per event
superfluid session inspect 7 --json
superfluid session export 7 --format jsonl --out ./exports
superfluid session export 7 --format otlp-jsonl --include-content
superfluid session export 7 --endpoint http://collector:4318 --include-content --allow-content-egress
```

Exports omit content unless `--include-content`; pushing content to an endpoint also needs `--allow-content-egress`. Exported content passes through a redaction pass for known secret shapes.

## What the log stores

> [!WARNING]
> The session log stores content: message text, generated text, tool names, arguments and results. Treat the sessions directory as holding every conversation in plain text.

- Images are referenced by SHA-256; their bytes live in `<sessions>/media/`.
- Per-request extras (penalties, `logit_bias`, logprobs) and fill-in-the-middle completions are not recorded.
- `SetMeta { archived: true }` hides a session and keeps everything.
- `Purge` removes a session logically, but its bytes stay in the single-file log. Purge is not content erasure.
