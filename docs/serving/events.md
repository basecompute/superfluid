# Start-up events

A program that runs `superfluid serve` or `superfluid node join` (a desktop app, a supervisor, a test harness) can follow the start from stderr without reading prose. With `--log-format json`, every line Superfluid writes on stderr is one JSON object, and each step of a start (a runtime installed, a model pulled and loaded, the server ready, a node joined) is an event with a name and fields of its own. Without the flag, stderr is the prose it always was.

```sh
superfluid serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --http 127.0.0.1:0 --log-format json
```

```json
{"timestamp":"2026-11-13T09:41:07.112406Z","event":"start","schema":1,"version":"0.1.13","command":"serve","pid":41230}
{"timestamp":"2026-11-13T09:41:07.118930Z","event":"log","level":"INFO","message":"superfluid: pulling unsloth/Qwen3-0.6B-GGUF:Q4_K_M with the llamacpp runtime"}
{"timestamp":"2026-11-13T09:41:07.119002Z","event":"pull.start","id":"unsloth/Qwen3-0.6B-GGUF:Q4_K_M","runtime":"llamacpp","offline":false}
{"timestamp":"2026-11-13T09:41:08.371551Z","event":"pull.progress","done_bytes":104857600,"total_bytes":396705472,"bytes_per_sec":83886080,"secs_left":4,"id":"unsloth/Qwen3-0.6B-GGUF:Q4_K_M"}
{"timestamp":"2026-11-13T09:41:11.620417Z","event":"pull.done","id":"unsloth/Qwen3-0.6B-GGUF:Q4_K_M","path":"/Users/me/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/1f0c/Qwen3-0.6B-Q4_K_M.gguf","bytes":396705472}
{"timestamp":"2026-11-13T09:41:11.702113Z","event":"model.load.start","model":"unsloth/Qwen3-0.6B-GGUF","runtime":"llamacpp","path":"/Users/me/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/1f0c/Qwen3-0.6B-Q4_K_M.gguf"}
{"timestamp":"2026-11-13T09:41:12.954870Z","event":"model.load.done","model":"unsloth/Qwen3-0.6B-GGUF","runtime":"llamacpp","context":40960,"lanes":8}
{"timestamp":"2026-11-13T09:41:12.955391Z","event":"serve.ready","http":"127.0.0.1:52318","socket":"/Users/me/.superfluid/superfluid.sock","models":["unsloth/Qwen3-0.6B-GGUF"]}
```

(Most of its `log` lines are left out here.)

## Turning it on

- `--log-format json` on `serve` or `node join` (`--log-format text`, the default, is the prose).
- `SUPERFLUID_LOG_FORMAT=json` in the environment, for a host that sets it once for every command it runs. A `--log-format` on the command line wins over it. Only `serve` and `node join` read it; the other commands write what they always did.

In JSON mode the terminal monitor is off (`--tui` included), and the bar a pull draws on a terminal is never drawn: the pull is `pull.progress` events instead. `--log-file <path>` sends the events to that file in place of stderr. `--log-dir <dir>` still sends the tracing logs to their own files, in the shape [Observability](../deployment/observability.md#logs) describes; stderr then carries the events and the prose.

## The stream

Every line is one JSON object with at least:

| field | meaning |
|---|---|
| `timestamp` | When the line was written out, RFC 3339 in UTC to the microsecond (`2026-11-13T09:41:07.112406Z`), the format of the `--log-dir` logs. The timestamps rise down the stream. |
| `event` | The event's name: `start`, `log`, `error`, or a dotted name from the table below. |

The first line is a `start` event, which states the stream's `schema`. Every line that is not one of the named events is a `log` event that carries the line: Superfluid's own prose, a worker's output, a runtime's, and Superfluid's tracing logs, each as it was written. A worker started by `serve` or `node join` writes to the same stream, so what it says is a `log` event too, in the order it was said. Only Superfluid writes the named events: a line a runtime or a tool writes is a `log` event even when it is JSON that names an event.

When a start fails, the last line is an `error` event, and its `exit_code` is the process's exit status ([exit codes](cli.md#exit-codes)). A panic is a `log` event at level `ERROR`; when it ends the process, the exit status is 101.

## Events

| event | fields | when |
|---|---|---|
| `start` | `schema` (1), `version` (Superfluid's), `command` (`serve` or `node join`), `pid` | the first line |
| `log` | `level`, `message`; from tracing also `target` and the log's own fields | every line that is not another event. `level` is `INFO`, or `WARN` for a line that says `WARNING`; a tracing log keeps its own (`ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE`) |
| `error` | `message`, `exit_code` | the last line before the process exits on a failure |
| `runtime.install.start` | `runtime` | the runtime a model needs was never installed here, and is being installed (as `superfluid runtime install <runtime>` would) |
| `runtime.install.done` | `runtime`, `install` (the install's name, e.g. `b6123-metal`), `version` | it is installed |
| `runtime.install.failed` | `runtime`, `error` | installing it failed; an `error` (exit 1) follows |
| `pull.start` | `id`, `runtime`, `offline` | a model given by id is looked for in its runtime's cache and, unless `offline`, fetched |
| `pull.progress` | `id`, `done_bytes`, `total_bytes`, `bytes_per_sec`, `secs_left` | while bytes land: at most four a second while they grow, one every 10 seconds while they do not, and the last as the fetch ends. `total_bytes` and `secs_left` are null when not known; `done_bytes` counts a part fetched by an earlier, interrupted pull. A model already in the cache has none |
| `pull.done` | `id`, `path`, `bytes` (the model's size on disk) | the model is on disk at `path` |
| `pull.failed` | `id`, `error` | the pull failed |
| `model.load.start` | `model` (the name the API serves it under), `runtime`, `path` | a model given at start-up begins loading |
| `model.load.done` | `model`, `runtime`, `context` (its window, in tokens), `lanes` | it is loaded |
| `model.load.failed` | `model`, `error` | it did not load: its engine, its tokenizer or its chat template failed; an `error` follows |
| `serve.ready` | `http` (the address the HTTP API is bound to, or null with `--no-http`), `socket` (the session socket, or null), `models` (the models loaded, the default first) | `serve` answers on every listener; a fleet head, once its nodes have shared their models' tokenizers |
| `node.discover.start` | | `node join` with no head address looks for one on the local network |
| `node.head.found` | `addr`, `models` | it found one |
| `node.dialing` | `head`, `node` (this node's identity), `model` (the name the head matches on), `lanes` | the model is loaded and the node dials the head |
| `node.joined` | `head` | the head took the node |
| `node.dial.failed` | `head`, `error` | the head did not answer |
| `node.link.ended` | `reason` | the link to the head ended |
| `node.retry` | `in_secs` | the node dials again after this many seconds |
| `node.refused` | `reason` | the head refused the node's token; an `error` (exit 2) follows |
| `fleet.node.joined` | `node`, `addr`, `models`, `context` | on a head: a node joined |
| `fleet.node.rejoined` | `node`, `addr` | on a head: a node that had joined came back |
| `fleet.node.refused` | `addr`, `reason` | on a head: a node that dialed in was not taken |

`model.load.*` cover the models a start loads: each `--model`, by path or by id. A start can also stop between `model.load.start` and `model.load.done` for a reason of its own (a session socket that cannot be bound, a damaged session log): it then ends with `error` alone, so a host takes `error` as the end whatever came before it. A model a running server loads later, when a request names it (one from `--model-dir`, or one unloaded while idle), and the `--fim-model` satellite have none. A node dials once per lane; its `node.*` link events speak for the first lane, as its prose does.

These names are reserved for events still to come: `node.removed` (the head removed this node), `node.head.changed` with `models` (the node follows its head to another model) and `fleet.node.removed` with `node`.

## What is kept

The names of the events and their fields, and what they mean, are kept across releases. A release may add an event, or a field to one, and the `schema` stays; removing or renaming one, or changing what it means, raises `schema`, and the changelog says so. A host reads the events it knows, ignores the others and any field it does not know, and checks `schema` on the first line.

The `message` of a `log` event is not kept: it is the prose a person reads, and it changes as the prose does. A host that needs to know something a `log` line says asks for an event instead of matching the text.

A start that fails in a fleet head's own checks (its `--fleet-*` flags, its token file, the flags a head does not take) writes its reason as a `log` line before it exits, not yet as an `error` event; the exit status is still the code.
