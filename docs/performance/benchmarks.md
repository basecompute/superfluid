# Benchmarks

A serving layer should cost nothing over the runtime it serves. These numbers compare superfluid against each runtime's own server on the same machine, model and runtime build.

## Setup

| | |
|---|---|
| machine | Apple M1 Max (64 GB), macOS |
| model | Qwen3-4B, 4-bit: Q4_K_M GGUF for llama.cpp, `mlx-community/Qwen3-4B-4bit` for MLX |
| runtime builds | llama.cpp b11284; mlx-lm 0.31.3 (mlx 0.32.2), on both sides |
| superfluid | `superfluid serve --max-batch 8 --max-context 4096` |
| baselines | `llama-server -np 8 -c 32768`, `mlx_lm.server` |
| date | 2026-10-06 |

**Workload.** Eight coding and explanation prompts, 256-token replies, at 1, 2, 4 and 8 concurrent requests. Throughput is completion tokens over wall time for the whole batch (`/v1/completions`, raw ChatML, so no chat template is in the comparison), the median of three rounds, each prompt distinct so no prefix cache answers. First-token time and stream smoothness are from `/v1/chat/completions` streams.

## Throughput (aggregate tokens per second)

Greedy (temperature 0):

| | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| superfluid, llama.cpp | 76.7 | 87.6 | 126.5 | 150.8 |
| llama-server | 76.8 | 86.2 | 125.4 | 145.1 |
| superfluid, MLX | 100.2 | 117.5 | 127.8 | 130.2 |
| mlx_lm.server | 96.6 | 96.8 | 96.8 | 96.8 |

Sampled (temperature 0.7, top-p 0.8, top-k 20):

| | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| superfluid, llama.cpp | 76.3 | 87.2 | 125.3 | 148.8 |
| llama-server | 77.7 | 86.3 | 125.9 | 145.3 |
| superfluid, MLX | 97.0 | 114.7 | 125.4 | 128.3 |
| mlx_lm.server | 90.0 | 90.1 | 90.2 | 90.2 |

`mlx_lm.server` serves one request at a time, so its total does not grow with concurrency.

## Chat streams

At eight concurrent chat streams the first token arrives in 0.70 s through superfluid on MLX against 9.42 s from `mlx_lm.server`, which queues requests; on llama.cpp it is 0.57 s against llama-server's 0.58 s.

## Serving semantics

Throughput does not show what happens when a worker is killed, the KV pool is oversubscribed, or a chat arrives behind eight agents. `tools/suite` runs eight such scenarios, each with a stated pass condition (see [its README](https://github.com/basecompute/superfluid/blob/main/tools/suite/README.md)); ✓ met it, ✗ did not, n/a has no way to run the scenario.

These ran on a different machine from the throughput numbers above: Apple M5 Pro (64 GB), Qwen3-4B 4-bit, llama.cpp b11284, mlx-lm 0.31.3, Ollama v0.35.1. The superfluid columns are from 2026-10-07 at commit `f634f9f`; llama-server, mlx_lm.server and Ollama ran on 2026-10-06 with the same builds. Each cell is one run of the scenario.

| scenario | superfluid (llama.cpp) | superfluid (MLX) | llama-server | mlx_lm.server | Ollama |
|---|---|---|---|---|---|
| worker kill | ✓ | ✓ | n/a | n/a | ✗ |
| daemon restart | ✓ | ✓ | ✓ | ✗ | ✓ |
| agents + interactive chat | ✓ | ✓ | ✗ | ✗ | ✗ |
| KV exhaustion | ✓ | ✓ | ✓ | ✗ | ✓ |
| unload under traffic | ✓ | ✓ | n/a | n/a | ✓ |
| 200-turn tool session | ✓ | ✓ | ✓ | ✓ | ✓ |
| client disconnect | ✓ | ✓ | ✗ | ✗ | ✓ |
| shared-prefix burst | ✓ | ✓ | ✗ | ✗ / ✓ ¹ | ✗ |

¹ mlx_lm.server failed this scenario in the 2026-10-06 run (it prefilled the shared prefix about four times) and passed when the scenario alone was run again on 2026-10-07 (about once), when one request of its comparison burst of distinct prompts also failed.

- **Agents + interactive chat:** eight agents send ~6k-token prompts, and a short chat in the `interactive` class arrives two seconds later; it passes if the chat's first token comes within 1.0 s and every agent finishes. The scenario measures the chat twice: 0.79 s and 0.55 s on llama.cpp, 0.31 s and 0.57 s on MLX. Measured once on the others: 67 s on llama-server, 64 s on Ollama, 120 s on mlx_lm.server.
- **Shared-prefix burst:** 32 requests at once share one 4,986-token prompt, differing only in a short question. It passes if the shared part is prefilled at most twice over the burst, counted from the cached tokens each server reports. Superfluid prefilled it about once; llama-server and Ollama about eight times. Median first token was 6.7 s on llama.cpp against llama-server's 30.3 s. Judged on first-token time alone (shared median at most half the median of 32 distinct prompts), llama-server and Ollama would pass: 30.3 s against 65.8 s, and 29.3 s against 64.2 s.
- **KV exhaustion:** 32 concurrent ~3.9k-token requests against 8 × 4,096 tokens of KV. All 32 completed on superfluid, llama-server and Ollama. The burst took 148 s on llama.cpp against llama-server's 165 s, and 224 s on MLX against mlx_lm.server's 435 s; mlx_lm.server reset three of the connections.

## Agreement

At temperature 0 a reply is the same text as the runtime's own server gives until two candidate tokens are close enough for rounding to decide. Against llama-server, 6 of 8 replies were identical with each prompt alone and 8 of 8 at four and eight lanes. On MLX the 4-bit model's logits tie exactly at many steps, and a hand-written loop over mlx-lm's own model call parts ways with `mlx_lm.generate` the same way (1 of 4 identical); the adapter's sampler is checked against the host sampler's rules row by row instead.

## Batch scaling

| lanes | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| llama.cpp aggregate tok/s | 76.7 | 87.6 | 126.5 | 150.8 |
| llama.cpp per-lane tok/s | 76.7 | 43.8 | 31.6 | 18.9 |
| MLX aggregate tok/s | 100.2 | 117.5 | 127.8 | 130.2 |

- Aggregate throughput rises with lanes until the engine is compute-bound; the per-lane rate falls.
- Median TTFT roughly doubles from one to eight lanes on both runtimes.
- A larger `--max-batch` costs nothing per tick, but the KV pool is sized for every lane.

## How to measure

The bench harness is not shipped; the method is.

- **Throughput.** Send N concurrent requests with N distinct prompts and a fixed reply length (`max_tokens`, plus `ignore_eos: true` on `/v1/completions`). Divide the sum of `usage.completion_tokens` by the wall time from the first send to the last response. Take the median of several rounds.
- **Templates.** Use `/v1/completions` with the prompt pre-formatted to exclude template rendering; use chat streams for TTFT and smoothness.
- **Access log rates are per request.** `decode=` is one lane's rate (about an eighth of the aggregate at eight lanes); `prefill=` includes queueing. The engine's rates are on `/slots`.
- **TTFT.** `superfluid_ttft_seconds` on `/metrics` starts at admission; the access log's `ttft` includes the wait before it.
- **Warm-up.** Discard the first round after a model loads. Make prompts distinct to measure prefill; share prefixes to measure agent workloads. Say which.
- **Thermals.** Interleave systems round by round and report medians, not best runs.

## Tuning

| flag | default | effect |
|---|---|---|
| `--max-batch N` | 8 | lanes decoding at once; aggregate throughput rises until compute-bound |
| `--max-context N` | auto | per-lane window and KV pool size; costs memory, not speed, until the pool no longer fits |
| `--prefill-budget N` | 4096 | `0` trades prompt throughput for decode continuity on slow models |
| `--tick-target-ms N` | 2000 | longer ticks amortize per-tick cost; shorter admit and cancel sooner (capped at 250 on llama.cpp and MLX) |
| `--class-lanes` | uncapped | reserve lanes between QoS classes |
| `--speculate` | off | `basert` only; a low-concurrency win the gates turn off under load |
| `--kv-bits` | 0 | `basert` only; narrower KV fits more context |
| `--park` | off | does not raise throughput; sealing copies a finishing session's whole KV on the tick thread |

**basert, sampled vs greedy at one lane.** A lone sampled lane is sampled on the host (cheaper than one GPU sampling pass); from two lanes rows are sampled on the GPU. Expect one sampled stream to run slightly behind one greedy stream, closing at two lanes.

**llama.cpp.** No llama.cpp flags pass through; the adapter chooses:

- A KV buffer of `max-context` cells per sequence for plain-attention models, so a step attends only over its own sequence; one shared pool for recurrent and sliding-window models. Per-sequence buffers number the lanes plus two (at least 4, at most 256), so 8 lanes of 8192 hold 10 x 8192 cells, against llama-server's 8 x 8192 for `-np 8`. (The next power of two above twice the lanes, the earlier choice, was 1.6x the KV for no decode gain: Qwen3-4B Q4_K_M, 8 lanes, M4 Max: 181.3 tok/s at 9 buffers, 180.6 at 12, 178.3 at 16.)
- The shared pool is `max-batch × max-context` cells capped at 60% of the device memory the weights leave free.
- The prefix cache keeps one lane's context of a shared pool and holds the rest in host memory, so its entries cost the lanes nothing while they wait. On gemma-3-4b (M5 Pro, `--max-context 8192 --max-batch 4`), after six 2,200-token documents were cached a cold 28 to 94-token prompt's first token took 110 to 118 ms, and 67 to 77 ms with the cache bounded and the batch filled to a multiple of 8 (the Metal kernel computes a short last group of queries against every cell in use).
- An entry pushed out for a sequence is kept in host memory as well. Six conversations returning to a Qwen3-4B daemon with `--max-batch 2` (four buffers) were all cold, 887 ms to a second turn's first token; with the entries held in host memory all six were warm, 98 ms.
- `n_batch` 2048, `n_ubatch` 512, every layer on the GPU.

**MLX.** The `mlx` and `mlx-lm` versions in the install set the kernels; pin them when comparing. The adapter prefills in 2048-token passes and sizes its pool as the llama.cpp adapter above does.
