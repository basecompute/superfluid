# Benchmark suite

One reproducible suite that compares serving systems with pass/fail
scenarios and numbers. Every table is generated from the raw JSON a
run leaves behind, so any number can be traced to the run that produced it.

Standard-library Python 3.9+; nothing to install beyond the servers under test.

## Layout

| file | what |
|---|---|
| `suite.py` | driver: for each (scenario, system) cell start the system fresh, run, stop, cool down, write `<scenario>--<system>.json` + `manifest.json` |
| `systems.py` | how each system starts on a box (same lanes and per-sequence context everywhere), its capabilities (separate worker? unload API? QoS header?) |
| `track1.py` | track 1, serving semantics: eight scenarios, each with a stated pass condition |
| `track2.py` | track 2, throughput: concurrency sweeps over short, long and shared-prefix prompts, sampled, speculation, sustained load |
| `load.py` | closed-loop load generator for track 2: aggregate and per-request rates, TTFT, inter-token gaps |
| `common.py` | HTTP client that times streams and can disconnect mid-stream; server process lifecycle; percentiles |
| `compare.py` | markdown tables from one or more result directories, and nothing else |
| `<box>.json` | the machine under test: paths to binaries and models, and their pinned versions (keys below) |

## Running

```
suite.py track1|track2 --box my-box.json \
    --systems superfluid-llamacpp,llama-server,mlx-lm-server,ollama \
    [--scenarios worker_kill,shared_prefix] [--out DIR] [--cooldown 60] [--quick]
compare.py DIR > report.md
```

A cell whose JSON already exists is skipped, so an interrupted run resumes by
re-running the same command. `--quick` shrinks scenario sizes for a dry run of
the harness; quick results are marked as such and are not for publication.

A box file is JSON with these keys (paths are relative to `root`):

| key | what |
|---|---|
| `box`, `chip` | a name for the machine and its chip; both land in the manifest |
| `root` | the directory everything below lives in; results go to `root/results/` |
| `superfluid`, `superfluid_home` | the `superfluid` binary and the `SUPERFLUID_HOME` it serves from |
| `llama_server`, `llama_lib` | the `llama-server` binary and its library directory (`lib_env` names the loader variable; default by platform) |
| `mlx_python`, `mlx_site` | the Python that runs `mlx_lm.server` and its site-packages |
| `ollama`, `ollama_model` | the `ollama` binary and the model name it creates from the GGUF |
| `models.gguf`, `models.mlx` | the model under test in each format |
| `versions` | `{"llama.cpp": ..., "mlx": ..., "mlx-lm": ..., "ollama": ...}`, copied into the manifest |
| `spec_args` | per-system flags that turn speculation on, for track 2's `speculation` cell |
| `extra_systems`, `hf_model` | systems the box describes itself (see below) and the Hugging Face id they take |

Measure on a dedicated machine, one driver per machine (the driver takes a
lock file and kills stray listeners on its ports before it starts). On
macOS set `ulimit -n 10240` first: non-interactive ssh sessions get 256.

## Track 1 scenarios

Qwen3-4B Q4 (Q4_K_M GGUF for llama.cpp-based systems, 4-bit MLX for MLX
ones), greedy, thinking off.

| scenario | what happens | pass condition |
|---|---|---|
| `worker_kill` | SIGKILL the inference worker mid-stream | typed error or completion; delivered text is a clean prefix of the reference; serves again unaided; same reply after |
| `daemon_restart` | SIGKILL the whole server mid-stream, restart on the same state dir, resend the conversation | restarts; identical reply (numbers: restart-to-ready, TTFT before/after) |
| `agents_interactive` | 8 agents with ~6k-token prompts, then one short chat 2 s later | interactive TTFT <= 1.0 s; every agent finishes, no stall > 30 s |
| `kv_exhaustion` | 32 concurrent ~3.9k-token requests against 8 x 4096 tokens of KV | each request completes or fails typed (no resets, hangs, truncation); a fresh request is admitted after |
| `unload_reload` | unload the model under 4 in-flight streams, then request it | in-flight streams complete or fail typed; the model serves again |
| `tool_session` | 200 tool-calling turns in one conversation | every emitted tool call is parsed, none leak as text; 200 turns reached (numbers: TTFT trend, cached-prompt ratio) |
| `client_disconnect` | drop 8 streaming, then 8 non-streaming, requests mid-generation | busy lanes reach 0 within 1 s, or fresh requests see idle TTFT |
| `shared_prefix` | 32 requests sharing one ~5k-token system prompt (a 4k-token document) vs 32 distinct ones | the shared prefix is prefilled at most twice over the burst, from the cached tokens the server reports (shared-burst TTFT p50 <= 0.5 x cold where it reports none) |

"unsupported" (n/a) means the system has no way to run the scenario, for the
reason given in its result; that is reported, not hidden.

Known limits of this version: `daemon_restart` drives the stateless HTTP API,
where clients resend history; superfluid's session API (resume by session id
from the write-ahead log) is not exercised yet.

## Track 2 scenarios (throughput)

Each system runs its native format of the box's model; rows across formats
compare servers, not kernels. Every cell drops a warm-up block and reports
the median of 3 rounds per concurrency (1, 2, 4, 8), each round 2 x
concurrency distinct nonce-prefixed requests, streamed.

| scenario | requests | numbers |
|---|---|---|
| `short` | short questions, ~300-word answers, greedy | aggregate tok/s, per-request decode rate, TTFT, inter-token gap p50/p95/p99; how many greedy replies at each concurrency are identical to the solo ones |
| `sampled` | as `short`, temperature 0.7 | as `short` |
| `long` | ~8k-token distinct documents, 128-token answers | TTFT, prefill tok/s |
| `shared` | one ~4k-token system prompt, distinct questions | TTFT under load with prefix reuse |
| `speculation` | rewrite-this-code prompts, speculation off then on | speedup per concurrency (superfluid `--speculate auto`; others by the box's `spec_args`) |
| `sustained` | 8 clients for 10 minutes | tok/s per minute, last over first |

Systems a box describes itself go in `extra_systems` with their argv;
placeholders `{gguf} {mlx} {hf} {port} {lanes} {ctx} {kv_tokens} {root}` are
filled in.
