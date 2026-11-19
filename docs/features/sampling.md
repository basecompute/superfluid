# Sampling

Every route resolves sampling parameters the same way: what the request names, then the operator's flags, then the model's published defaults, then built-in constants. Each sampled token is a pure function of its position in the session, so a continuation, fork or resume draws what the uninterrupted request would have.

## Parameters

| field | default | notes |
|---|---|---|
| `temperature` | model default, else 1.0 | `0` is greedy |
| `top_p` | model default, else 1.0 | |
| `top_k` | model default, else 40 when sampling | extension; `0` disables |
| `min_p` | model default, else 0.0 | extension |
| `seed` | fresh per request | honoured when given |
| `presence_penalty` | `--presence-penalty` on `basert`, else 0.0 | -2 to 2 |
| `frequency_penalty` | 0.0 | -2 to 2 |
| `repeat_penalty` (alias `repetition_penalty`) | model default, else 1.0 | extension; 1.0 disables |
| `logit_bias` | none | `{"<token id>": bias}` |
| `logprobs`, `top_logprobs` | off | up to 20 alternatives |
| `stop` | none | excluded from output |
| `ignore_eos` | `false` | extension |

The Anthropic route takes `temperature`, `top_p` and `top_k`. The Ollama route maps its `options` onto the same fields (see [Ollama API](../serving/ollama_api.md#options)).

## Defaults precedence

| rung | source |
|---|---|
| 1 | the request |
| 2 | operator flags: `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`, `--presence-penalty` (`basert` only) |
| 3 | the model's published defaults: GGUF `general.sampling.*`, MLX `generation_config.json`, the `.base` header |
| 4 | constants: temperature 1.0, `top_k` 40 for sampled requests nobody truncated, repeat penalty 1.0 |

> [!NOTE]
> Truncation (`top_p`, `top_k`, `min_p`) is taken as one set from one rung. A request that sends only `top_p: 1.0` gets the full distribution; it does not inherit the model's `top_k` underneath it.

- `do_sample: false` in a model's `generation_config.json` means greedy unless a temperature is named.
- `/props` advertises the effective defaults; a value is omitted while unknown rather than guessed.
- `--spec-max-temperature` caps only a temperature that came from the model's default, on a model that speculates.

### Repetition penalty

The default is 1.0 (off). `--repeat-penalty 1.15` restores loop-breaking: in a measurement continuing a prompt that already held eight copies of one word at temperature 0, penalties of 1.0 to 1.10 never escaped in 80 tokens, 1.15 escaped after 14 tokens and 1.20 after 6. It is not the default because a penalty costs decode throughput on the baseRT engine and lowers speculative acceptance (by roughly a third where measured). A checkpoint's own `repetition_penalty` outranks the constant.

The penalty window is the last 64 tokens, prompt included. Penalties skip the model's special tokens and stop set.

### Presence penalty

The default is 0 (off), and no model publishes one. `--presence-penalty 1.5` gives every request that names no `presence_penalty` that value on a model served by the `basert` runtime, on every route (a `/v1/batches` job runs with no operator flag); a request's own value wins, and `"presence_penalty": 0` turns it off for that request. Use it for a model whose card recommends a presence penalty that clients do not send.

The flag is the `basert` runtime's because its engine applies the penalty itself, which its capability record states (`sampling.engine_penalties`). On llama.cpp and MLX a penalized request leaves the runtime's own sampler for superfluid's, token by token, so the server does not turn that on for every request: a model on another runtime is served without the default (the start says so, and `/props` omits the field), and a request that sends `presence_penalty` is penalized on any runtime, as before. A `basert@<install>` runs the worker its install copied, and one installed by superfluid 0.1.14 or older does not state it; `superfluid runtime repair basert@<install>` installs it again with this superfluid's worker. On a fleet head the flag applies to the model where its nodes' engine states it, as the head learns from the first node serving it; the head resolves it, and `--repeat-penalty`, before a node sees the request. Like the repetition penalty, it lowers speculative acceptance, so a model that recommends it only for some modes (thinking off, say) is better served by the client sending it for those requests.

## Order of operations

Per token:

1. grammar mask (when the request is constrained)
2. `logit_bias`
3. repeat, presence and frequency penalties
4. temperature, then top-k, min-p and top-p against the full softmax
5. the draw, or argmax at temperature 0

Ties resolve to the lowest token id on every runtime.

## Where sampling runs

| runtime | greedy | sampled |
|---|---|---|
| `basert` | on the GPU | on the GPU from two lanes; a lone sampled lane on the host |
| `llamacpp` | llama.cpp's argmax | on the host |
| `mlx` | on the device | on the device, by the host sampler's rules |

A lane with `logit_bias`, `logprobs` or a grammar is sampled on the host on every runtime and never speculates.

## Determinism

- **Greedy** gives the same token on every path of every runtime.
- **Seeded sampling** is deterministic per seed within one execution shape. On `basert`, the GPU and host samplers draw different noise, so seeded text can differ between decoding alone, batched, or speculating.
- **Batch composition** moves logits by rounding and can flip a near-tie. For output independent of other traffic, request a batch-invariant lane (`x-superfluid-batch-invariant: 1`, allowed by `--http-allow-batch-invariant`); it ticks alone.
- **Prefix-cache hits** prefill in a different shape than a cold prompt and can flip a near-tie. Same shape and seed give the same output.

## Logprobs

`logprobs` are computed over the penalized, biased, unscaled logits: the chosen token and up to 20 alternatives. Chat replies carry `choices[].logprobs.content[]` (`token`, `logprob`, `bytes`, `top_logprobs`); completions carry `tokens`, `token_logprobs`, `top_logprobs` and `text_offset`.
