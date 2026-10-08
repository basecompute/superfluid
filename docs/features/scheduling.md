# Scheduling and QoS

Every generation is a lane in a continuous-batching tick loop. One scheduler thread per model admits queued requests into free lanes, hands out prefill chunks and decode grants for the tick, and commits what the engine returns. QoS classes order the queue, interactive work preempts background agents, and a pressure ladder evicts cache before the KV pool runs out.

## Ticks

A tick is one engine call. Its length is sized toward a target duration:

| situation | target (`basert`) | target (`llamacpp`, `mlx`) |
|---|---|---|
| a lane is free, or one is finishing | 250 ms | 50 ms |
| every lane busy | `--tick-target-ms` (2000) | at most 250 ms |

A request that arrives mid-tick waits for the tick in flight, so short ticks while a lane is free keep admission fast. llama.cpp and MLX declare that a decode round costs the same however rounds are cut into ticks, so their ticks are a few rounds long.

## Lanes and admission

`--max-batch` (default 8) is the number of concurrent lanes. Admission checks the prompt fits the context window (else `context_length_exceeded`), seeds from the prefix cache or a park artifact, and binds any media.

There is one FIFO queue per QoS class. The next job is the head of the highest-priority class under its lane cap. A job that has waited `--starvation-ticks` ticks (default 8) rises one class per period waited.

## QoS classes

| class | aliases | used for |
|---|---|---|
| `interactive` | `interactive-chat`, `0` | chat UIs |
| `completion` | `inline-completion`, `1` | fill-in-the-middle (always) |
| `agent` (default) | `foreground`, `foreground-agent`, `2` | agents |
| `background` | `background-agent`, `3` | batch work |

Set a request's class with:

- the `x-superfluid-qos: <class>` header (ignored under `--no-http-qos-header`);
- `--http-default-qos <class>` for requests that name none;
- a `--key-policy` entry's `class` (see [Security](../deployment/security.md#key-policy));
- `SetQos` on the session API.

> [!WARNING]
> Without a key policy, honoring the header lets any client claim `interactive` and preempt running agents. Bind loopback, use `--no-http-qos-header`, or use a key policy.

`--class-lanes background=2,agent=6` caps how many lanes a class may hold. Caps must be between 1 and `--max-batch`.

### Preemption

When an `interactive` or `completion` job is queued and every lane is busy, the lowest-class running lane below it is preempted: it stops, parks its KV if parking is on, and requeues at the head of its class. A decoding lane is chosen before a prefilling one; batch-invariant and media lanes are never chosen. The client's stream pauses and resumes; nothing is repeated. Critical OS memory pressure preempts every decoding `background` lane.

### Batch-invariant lanes

`x-superfluid-batch-invariant: true` asks for a lane that ticks alone, so output does not depend on other traffic. While one waits, nothing at or below its class is admitted, so any client allowed to ask can serialize the server. Over HTTP it is a 403 unless `--http-allow-batch-invariant` is set or the key policy grants it.

## Prefill and decode budgets

| flag | default | meaning |
|---|---|---|
| `--prefill-budget N` | 4096 | prompt tokens per tick across lanes, 64-4096. `0` selects the adaptive controller (256-4096, sized so a prefill tick lands near the tick target) |
| `--tick-decode-budget N` | 0 | decode tokens per tick across lanes; 0 scales with the grant |
| `--tick-target-ms N` | 2000 | tick duration while every lane is busy |
| `--starvation-ticks N` | 8 | starvation bound and queue aging period |

- Within a tick, prefill goes to starving lanes first, then by class, then shortest remaining prompt first, so a short request does not wait behind a long prompt.
- A tick in which an `interactive` or `completion` prompt finishes prefills nothing of a lower class beside it.
- While a newcomer could be admitted at the end of the tick, one tick's prefill is capped at about 0.5 s of work (whole 512-token chunks, or 128-token steps on a model too slow for one chunk in that time). A short chat arriving while eight agents prefill 7.5k-token prompts reaches its first token in about 0.7 s (Qwen3-4B, llama.cpp, M4 Max).
- One tick's prefill is otherwise capped at 8 s of work.
- A running tick ends at the engine's next step (a prefill micro-batch or a decode round) when a lane is cancelled or an `interactive` or `completion` request is queued; prefill it did not reach is planned again. Engines run by superfluid's own workers (`llamacpp`, `mlx`) do this; a native engine finishes its tick.
- The pinned 4096 default favors prompt throughput; choose `0` when decode continuity on a slow model matters more.

The live budget and grant are `superfluid_prefill_budget_tokens` and `superfluid_decode_grant_tokens` on `/metrics`.

## Memory pressure

The KV pool is the one bounded resource. After every tick:

1. **Watermarks.** At `--pressure-high` (85%) occupancy, cached prefixes are evicted down to `--pressure-low` (70%). Live lanes are never evicted; pinned prefixes yield, oldest first, only when unpinned cache is not enough. On llama.cpp an evicted prefix leaves the pool for host memory, where the next request that shares it imports it back.
2. **OS pressure** (`--os-pressure`, on). A warning runs one eviction round; critical pressure also preempts decoding `background` lanes. Either drops the prefixes held in host memory first, and none is moved there for a while after. macOS reads `kern.memorystatus_vm_pressure_level`; Linux reads `/proc/pressure/memory`.
3. **Replan.** A plan the pool cannot hold is rejected by the engine and retried with pins released.

## Fill-in-the-middle

Completions run in the `completion` class with no session log, a per-client token bucket (`--completion-rate` 10/s, `--completion-burst` 20) and an optional start deadline (`--completion-deadline-ms`). `--fim-model` runs them on a dedicated model with its own lanes (`--fim-max-batch` 2, `--fim-max-context` up to 8192).

## Fleet head

A fleet head has no local scheduler and refuses `--class-lanes`, `--http-default-qos`, `--no-http-qos-header` and `--http-allow-batch-invariant`; set them on each node's own `serve`. See [Distributed serving](../serving/distributed_serving.md).
