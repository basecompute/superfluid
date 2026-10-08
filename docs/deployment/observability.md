# Observability

superfluid exposes an access log, structured logs, Prometheus metrics, OTLP traces and metrics, and two terminal monitors. All of them are content-free: ids, counts, lengths, timings and codes, never prompt or completion text.

## Access log

One line per generation on the chat and completion routes (and the Ollama routes that share them):

```text
<-- 200 POST /v1/chat/completions model=Qwen3-4B prompt=812 cached=640 completion=256 ttft=312ms prefill=551 tok/s decode=74.9 tok/s total=3792ms finish=stop
```

| field | meaning |
|---|---|
| `prompt` | prompt tokens after the chat template |
| `cached` | prompt tokens seeded from the prefix cache or a park artifact |
| `completion` | tokens generated |
| `ttft` | ms from the start of generation to the first token delivered (includes queueing) |
| `prefill` | `(prompt - cached) / ttft`; `-` for a fully warm prompt |
| `decode` | tokens per second between the first and last token delivered |
| `total` | wall ms of the whole request |
| `finish` | `stop`, `length` or `tool_calls` |

Both rates are measured at delivery to the client: a slow consumer lowers `decode`. Engine-side rates are on `/slots` and in the monitor. `/v1/messages` writes no access-log line.

## Logs

| flag | default | effect |
|---|---|---|
| `--log-filter <directive>` | `info` | `EnvFilter` syntax, e.g. `superfluid_daemon::scheduler=debug,info` |
| `-v`, `--verbose` | | `--log-filter debug` |
| `--log-dir <dir>` | stderr | JSON lines in `superfluid.<n>.jsonl`, 64 MiB per file, newest 8 kept, mode 0600 |
| `--log-file <path>` | none | redirect stderr (daemon and worker output) to a file |

- With the monitor on and no `--log-dir`, logs go to `<sessions>/logs`.
- The JSON sink has an 8192-line queue; a full queue drops lines (counted in `superfluid_telemetry_dropped_total`) rather than stall the scheduler.
- `RUST_LOG` is not read. The filter can be changed at run time with the session API's `SetLogLevel`.

## Prometheus metrics

```sh
curl -s http://127.0.0.1:8453/metrics | grep -v '^#'
```

`GET /metrics` and `GET /v1/metrics` serve Prometheus text. They need the API key when one is set, but are exempt from rate limits and key slots. On a multi-model server they describe the default model. All names are prefixed `superfluid_`.

| counter | meaning |
|---|---|
| `ticks_total` | scheduler ticks |
| `prefill_tokens_total`, `decode_tokens_total` | tokens prefilled and decoded |
| `warm_prefix_tokens_total` | tokens served from the prefix cache or park artifacts |
| `cold_admissions_total` | admissions that started cold |
| `preemptions_total` | lanes preempted and requeued |
| `admits_deferred_total` | admissions with no room in their tick, requeued |
| `starvation_grants_total` | ticks where a starving lane was served first |
| `pressure_evictions_total`, `pressure_bytes_evicted_total` | watermark eviction rounds and bytes |
| `os_pressure_events_total` | OS memory-pressure signals acted on |
| `pins_yielded_total`, `pins_expired_total` | pinned prefixes released |
| `worker_respawns_total` | worker restarts |
| `spec_proposed_total`, `spec_accepted_total` | speculative draft tokens |
| `parks_total{encoding}`, `resumes_total` | park artifacts written and restored |
| `telemetry_dropped_total` | log lines and spans dropped |
| `otlp_spans_exported_total`, `otlp_spans_dropped_total{reason}` | OTLP span export |
| `fim_completions_total{result}` | FIM completions: `generated`, `cached`, `expired`, `throttled` |
| `key_requests_total{key}`, `key_rejected_total{key,reason}` | per-key traffic (with `--key-policy`) |

| gauge | meaning |
|---|---|
| `lanes_active` | lanes resident |
| `pool_blocks_used`, `pool_blocks_total` | KV pool occupancy |
| `prefill_budget_tokens`, `decode_grant_tokens` | live prefill budget and decode grant |
| `pins_held`, `pinned_bytes` | pinned prefixes |
| `queue_depth{class}` | queued jobs per QoS class |
| `node_info{host}` | always 1; `host` is `SUPERFLUID_NODE_NAME` or the hostname |
| `key_in_flight{key}` | requests holding a key's slots |

`superfluid_ttft_seconds` is a histogram of admission to first token, with buckets from 5 ms to 40 s. It excludes the wait before admission; the access log's `ttft` includes it.

## OTLP

Both exporters use OTLP/HTTP JSON over plain `http://`; an `https://` URL is refused. Run a local collector and let it forward with TLS.

```sh
superfluid serve <model> --otlp-endpoint http://127.0.0.1:4318 --otlp-metrics http://127.0.0.1:4318
```

| flag | default | meaning |
|---|---|---|
| `--otlp-endpoint <url>` | off | push spans to `<url>/v1/traces` |
| `--otlp-header k=v` | | extra header, repeatable |
| `--otlp-service-name` | `superfluid` | `service.name` |
| `--otlp-filter` | `info,superfluid_daemon::scheduler=debug` | which spans export |
| `--otlp-queue`, `--otlp-batch-ms`, `--otlp-timeout-ms` | 8192, 1000, 5000 | queue, batch interval, request timeout |
| `--otlp-metrics <url>` | off | push metrics to `<url>/v1/metrics` |
| `--otlp-interval-ms` | 15000 | metrics push interval |

- Spans: `generate`, `tick`, `park`, `restore`. Only numeric and boolean fields plus the model id are exported, so a span cannot carry text.
- A slow or dead collector costs dropped spans, never a delayed tick. Failed batches are not retried.
- The metrics push carries a subset of `/metrics` (scheduler counters, lane, pin and queue gauges, the TTFT histogram).
- `superfluid session export --format otlp-jsonl` writes a session as an agent trace (GenAI semantic conventions) whose trace ids match the live exporter's.

## Terminal monitors

Both need a build with the `tui` feature (the default build has it).

### `superfluid serve` monitor

On by default when stdout is a terminal; `--no-tui` turns it off, `--tui` forces it. It shows the tick in flight, lane occupancy, the KV pool, decode and prefill rates, TTFT, prefix-cache reuse, speculative acceptance, queue depth per class, counters and a log pane.

| key | action |
|---|---|
| `q`, `Esc`, `Ctrl-C` | quit (stops the server) |
| `p` | pause sampling |
| `Up`/`Down`, `PgUp`/`PgDn`, wheel | scroll the log |
| `g` / `End` | oldest line / follow |

### `superfluid top`

```sh
superfluid top --api-key "$KEY" 10.0.0.11:8453 10.0.0.12:8453
```

Polls each node's `/metrics` once a second and shows one row per node: state (`up`, `401`, `down`), KV pool, decode tok/s, lanes and queue depth, plus a fleet throughput sparkline. `q` or `Esc` quits.

## Health and status endpoints

| endpoint | returns |
|---|---|
| `GET /health` | `{"status":"ok","models_loaded":N,"models_known":M}`; never authenticated or rate-limited, touches no engine |
| `GET /props` | served model, effective generation defaults, window, chat template, KV type |
| `GET /slots` | one aggregate entry: lanes, KV pool, the tick in flight, engine decode and prefill rates, last TTFT |

## Debug environment variables

These print internals to stderr, unfiltered by `--log-filter`.

| variable | prints |
|---|---|
| `SUPERFLUID_DEBUG_TICK` | one line per tick: phase times, prefill tokens, rounds, learned costs, budget and grant |
| `SUPERFLUID_DEBUG_PRESSURE` | pool counters before each eviction decision |
| `SUPERFLUID_ENGINE_VERBOSE` | the baseRT engine's verbose output |
| `SUPERFLUID_LLAMA_LOG=1` | llama.cpp's own log |
