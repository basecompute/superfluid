# OpenAI-compatible server

`superfluid serve` exposes an OpenAI-compatible HTTP API at `http://127.0.0.1:8453` by default, on the same listener as the [Anthropic](anthropic_api.md) and [Ollama](ollama_api.md) APIs. Every request runs as a fresh durable session, so a later request that repeats a conversation prefix is served warm from the prefix cache with no client-side session handling.

```sh
superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8453/v1", api_key="unused")
reply = client.chat.completions.create(
    model="unsloth/Qwen3.8-27B-GGUF",
    messages=[{"role": "user", "content": "Say hello."}],
    max_tokens=32,
)
print(reply.choices[0].message.content)
```

The `model` field names an id from `GET /v1/models`. Unknown fields are ignored. The request body limit is 100 MiB (413 above it).

## Endpoints

| method | path | purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | chat completion, streaming or not |
| `POST` | `/v1/completions` | raw-text completion; with `suffix`, fill-in-the-middle |
| `POST`, `GET`, `DELETE` | `/v1/responses`, `/v1/responses/{id}`, `/v1/responses/{id}/input_items` | [Responses API](responses_api.md), with `previous_response_id` |
| `POST` | `/v1/embeddings` | text embeddings (embedding models only) |
| `POST` | `/v1/rerank` | cosine rerank of documents against a query |
| `POST` | `/v1/audio/transcriptions`, `/v1/audio/translations` | speech to text (Whisper models only) |
| `POST` | `/v1/messages` | [Anthropic Messages API](anthropic_api.md) |
| `GET` | `/v1/models`, `/v1/models/{model}` | known models, loaded or not, with capabilities |
| `POST` | `/v1/models/load`, `/v1/models/unload` | load / unload a model (admin) |
| `DELETE` | `/v1/models/{model}` | unload a model (admin) |
| `POST` | `/v1/tokenize` | token ids of a text, no chat scaffold |
| `POST`, `GET` | `/v1/files`, `/v1/files/{id}`, `/v1/files/{id}/content` | upload, list, read, delete files |
| `POST`, `GET` | `/v1/batches`, `/v1/batches/{id}`, `/v1/batches/{id}/cancel` | batch jobs |
| `POST`, `GET` | `/v1/lora/load`, `/v1/lora/unload`, `/v1/lora` | activate (admin), clear (admin) or show a LoRA adapter; `basert` bundles only |
| `GET` | `/health` | liveness; never authenticated |
| `GET` | `/props` | the served model and its effective generation defaults |
| `GET` | `/slots` | lane and KV-pool snapshot (llama.cpp shape) |
| `GET` | `/metrics`, `/v1/metrics` | Prometheus text; see [Observability](../deployment/observability.md) |
| `*` | `/api/*`, `/` | [Ollama API](ollama_api.md) |

## Chat completions

```sh
curl http://127.0.0.1:8453/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "unsloth/Qwen3.8-27B-GGUF",
  "messages": [{"role": "user", "content": "Three facts about Oslo."}],
  "stream": true,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

### Supported parameters

| field | default | notes |
|---|---|---|
| `model` | required | 400 if absent, 404 if unknown |
| `messages` | required | `system`, `user`, `assistant`, `tool`; content as a string or parts (`text`, `image_url`, `input_audio`) |
| `max_completion_tokens`, `max_tokens` | fill the context, or `--max-tokens` | `max_completion_tokens` wins |
| `stream` | `false` | SSE, ending in `data: [DONE]` |
| `stream_options` | | `include_usage` is always on |
| `n` | 1 | 1-128; streaming uses 1 |
| `temperature`, `top_p` | model default | see [Sampling](../features/sampling.md) |
| `seed` | fresh per request | do not send `0` (read as unseeded) |
| `presence_penalty`, `frequency_penalty` | 0 | |
| `stop` | none | string or array; excluded from output |
| `logprobs`, `top_logprobs` | off | `top_logprobs` 0-20 |
| `logit_bias` | none | `{"<token id>": bias}` |
| `tools`, `tool_choice` | `[]`, `auto` | see [Tool calling](../features/tool_calling.md) |
| `response_format` | none | `json_object` or `json_schema`; see [Structured outputs](../features/structured_outputs.md) |
| `reasoning_effort` | model default | see [Chat templates](../features/chat_templates.md) |

### Extra parameters

| field | notes |
|---|---|
| `top_k`, `min_p` | truncation |
| `repeat_penalty` (alias `repetition_penalty`) | 1.0 disables |
| `ignore_eos` | decode to `max_tokens` |
| `chat_template_kwargs` | template variables such as `enable_thinking`, `reasoning_effort`; a key the template refuses is a 400 |
| `enable_thinking` | top-level shorthand; `chat_template_kwargs` wins |
| `stream_options.continuous_usage_stats` | cumulative `completion_tokens` on every content chunk |

### Response

```json
{
  "id": "chatcmpl-superfluid-17", "object": "chat.completion", "model": "Qwen3-4B",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "...", "reasoning_content": "...", "tool_calls": [...]},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160,
            "prompt_tokens_details": {"cached_tokens": 96}},
  "superfluid": {"spec": {"proposed": 60, "accepted": 38, "acceptance_rate": 0.633}}
}
```

- `reasoning_content` appears when the model produced reasoning; `tool_calls` when calls were parsed.
- `finish_reason` is `stop`, `length` or `tool_calls`. A reply cut by the token cap is `length` even if it holds a tool call; do not run it.
- `cached_tokens` is the prompt prefix served from the prefix cache (also in the `x-superfluid-warm` header).
- `superfluid.spec` appears only when [speculative decoding](../features/speculative_decoding.md) ran.

### Streaming

- The first chunk's delta is `{"role":"assistant"}`; content and `reasoning_content` deltas follow at token cadence.
- Tool calls stream as OpenAI partial deltas: an opening entry with `id` and `function.name`, then `arguments` fragments on the same `index`.
- The terminal chunk carries `finish_reason`, `usage` and `superfluid.warm` (cached prefix tokens).
- Lines starting with `:` are keep-alives.
- A client that disconnects cancels the generation at the next tick.
- An error after the stream began arrives as a `{"error": {...}}` data frame, then `data: [DONE]`.

## Completions and fill-in-the-middle

`POST /v1/completions` tokenizes `prompt` directly (a string; arrays and token ids are not accepted). It reads `prompt`, `suffix`, `fim_mode`, `stream`, `echo`, `max_tokens`, `logprobs` (0-20) and the sampling fields; it does not read `stop`, `n`, `logit_bias`, `response_format` or `tools`.

A request with `suffix` is fill-in-the-middle: `prompt` is the prefix, `suffix` the text after the cursor, the reply the middle.

```sh
curl http://127.0.0.1:8453/v1/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen2.5-Coder-1.5B","prompt":"def add(a, b):\n    ","suffix":"\n\nprint(add(1, 2))","max_tokens":32}'
```

| FIM rule | value |
|---|---|
| scheduling class | `completion`, with `--completion-deadline-ms` |
| per-client token bucket | `--completion-burst` 20, refilled at `--completion-rate` 10/s; over it is 429 `completion_rate_limited` with `Retry-After` |
| `max_tokens` default | `--max-tokens`, else 128 |
| `fim_mode` | `psm` (default) or `spm` (extension) |
| refused | `echo`, `logprobs` (400 `unsupported_parameter`); a model with no FIM support and no `--fim-model` (400 `fim_unsupported`) |

The reply's `superfluid` object carries `cached`, `expired` and `served_by`.

## Embeddings, rerank, audio

| endpoint | request | notes |
|---|---|---|
| `/v1/embeddings` | `model`, `input` (string or array), `encoding_format` (`float` or `base64`) | vectors as the engine pools them (not normalized) |
| `/v1/rerank` | `model`, `query`, `documents`, `top_n`, `return_documents` | cosine similarity of embeddings, sorted descending |
| `/v1/audio/transcriptions`, `/v1/audio/translations` | multipart: `file`, `model`, `language`, `prompt`, `response_format`, `stream` | see [Vision and audio](../features/multimodal.md#speech-to-text) |

These need a model whose capability record allows it (`workload.embedding`, `modalities.whisper_*`); the `llamacpp` and `mlx` runtimes serve neither. Otherwise: 400 `unsupported_model`.

## Models

```sh
curl http://127.0.0.1:8453/v1/models
```

Each entry adds `loaded`, `meta.n_ctx` (the context window), `architecture.input_modalities`, `runtime` and `capabilities` (the runtime's capability record plus a `dialect` object: `enable_thinking`, `reasoning_effort`, `reasoning_effort_levels`, `fim`, `fim_model`). Loading and unloading are in [Serving multiple models](multiple_models.md).

## `/props` and `/slots`

`GET /props` returns the served model, `default_generation_settings` (what a request that names nothing runs with), `max_context`, `chat_template`, `kv_cache` and `capabilities`. `?model=<id>` describes another model; `?autoload=1` loads a known model first. A sampling default is omitted, not guessed, while unknown.

`GET /slots` returns one aggregate entry in llama.cpp's slot shape (lanes, KV pool, tick plan, decode and prefill rates). `POST /slots/{id}/save` and `/restore` are retired: 501 `route_retired`.

## Files and batches

| route | notes |
|---|---|
| `POST /v1/files` | multipart `file` and `purpose`; stored under `<sessions>/files/`; ids `file-<hex>`. `--files-max-bytes` caps the store (413 `files_quota_exceeded`), `--files-expiry` ages files out |
| `POST /v1/batches` | `{"input_file_id", "endpoint"?, "completion_window"?}`; JSONL lines `{"custom_id", "method", "url", "body"}` for `/v1/chat/completions` or `/v1/embeddings` |
| `POST /v1/batches/{id}/cancel` | stops between lines |

A batch runs line by line against the default model. Up to 4 batches run at once; later ones wait in `validating`, in order, and a full queue (256 waiting) refuses a new batch with 429 `batch_queue_full`. Lines without `max_tokens` are capped at 2048 and use the model's defaults, not the operator's sampling flags. Status moves `validating`, `in_progress`, `completed` (or `failed`, `cancelled`). Under `--key-policy`, files and batches are visible only to the key that created them.

## Headers

| header | direction | meaning |
|---|---|---|
| `Authorization: Bearer <key>` or `X-Api-Key` | request | with `--api-key` or `--key-policy` |
| `x-superfluid-qos` | request | `interactive`, `completion`, `agent` or `background`; see [Scheduling](../features/scheduling.md) |
| `x-superfluid-batch-invariant` | request | a lane that ticks alone; needs `--http-allow-batch-invariant` |
| `x-superfluid-warm` | response | prompt tokens served from the prefix cache |
| `x-superfluid-keepalive: whitespace` | response | `--nonstream-keepalive` is padding the body with spaces |

With `--nonstream-keepalive <sec>`, a non-streaming generation commits `200 OK` at admission and writes one space every interval until the JSON body, so an idle-timing proxy does not drop a long prefill. Leading whitespace is valid JSON.

## Errors

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}
```

| status | `code` | when |
|---|---|---|
| 400 | `context_length_exceeded` | the prompt is longer than the window |
| 400 | `unsupported_input` | an image or audio part the model cannot take |
| 400 | `unsupported_model` | embeddings, rerank or audio on a model that does not serve them |
| 400 | `unsupported_runtime` | `/v1/models/load` on a runtime that is missing or does not read the model |
| 400 | | malformed body; `constraint unsatisfiable: ...` (a grammar that cannot compile) |
| 401 | `invalid_api_key` | missing or wrong key |
| 403 | `admin_required` | a non-admin key on an admin route; also a QoS header above the key's ceiling |
| 404 | | unknown model |
| 413 | `payload_too_large`, `files_quota_exceeded` | body over 100 MiB; file store full |
| 429 | `rate_limit_exceeded`, `concurrency_limit_exceeded`, `completion_rate_limited` | per-IP, per-key, FIM limits |
| 500 | | engine or worker failure |
| 501 | `route_retired` | `/slots/{id}/save`, `/restore` |

## Browser transport (`--web`)

`--web 127.0.0.1:<port>` starts a separate loopback-only listener carrying the [session API](session_api.md) (not the OpenAI API) for browser front-ends: `POST /web/rpc`, `POST /web/stream`, `GET /web/ws`, `GET /web/health`. Authentication is described in [Security](../deployment/security.md#browser-transport).
