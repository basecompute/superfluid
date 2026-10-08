# Ollama-compatible API

superfluid serves a subset of Ollama's API (inference, discovery and embeddings) on its HTTP listener, for any runtime. It does not read Ollama's model store, pull models or interpret Modelfiles.

Point an Ollama client at the server root:

```sh
superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --port 11434
curl http://127.0.0.1:11434/api/tags
```

> [!NOTE]
> Use a different port if Ollama itself is listening on 11434.

```python
from ollama import Client

client = Client(host="http://127.0.0.1:11434")
model = client.list().models[0].model
for part in client.chat(model=model, messages=[{"role": "user", "content": "Hello"}], stream=True):
    print(part.message.content, end="", flush=True)
```

`superfluid launch ollama` runs the `ollama` command line against the server; see [Ollama CLI](../integrations/ollama_cli.md).

Model names are superfluid's model ids; `id:latest` is accepted. Load models at startup or through [`POST /v1/models/load`](multiple_models.md#load-and-unload); arbitrary paths and remote names are refused.

## Routes

| method | path | behaviour |
|---|---|---|
| `GET` | `/` | `superfluid is running` |
| `GET` | `/api/version` | `{"version", "server": "superfluid", "compatibility": "ollama-inference"}` |
| `GET` | `/api/tags` | every known model id |
| `GET` | `/api/ps` | loaded models and their context length |
| `POST` | `/api/show` | model info; for a loaded model, capabilities and chat template |
| `POST` | `/api/chat` | chat with history, tools, thinking, images, JSON output |
| `POST` | `/api/generate` | templated prompt; `raw` for plain completion; `suffix` for fill-in-the-middle |
| `POST` | `/api/embed` | normalized embeddings (embedding models only) |
| `POST` | `/api/embeddings` | legacy single-`prompt` form, unnormalized |
| any | `/api/pull`, `/api/push`, `/api/create`, `/api/copy`, `/api/delete`, `/api/blobs/*` | 501 |

Discovery routes never load a cold model.

## Chat and generate

```sh
curl -N http://127.0.0.1:11434/api/chat -d '{
  "model": "unsloth/Qwen3.8-27B-GGUF",
  "messages": [{"role": "user", "content": "Explain prefix caching briefly."}],
  "options": {"num_predict": 128, "temperature": 0.2}
}'

curl http://127.0.0.1:11434/api/chat -d '{
  "model": "unsloth/Qwen3.8-27B-GGUF", "stream": false,
  "messages": [{"role": "user", "content": "Return a planet name as JSON."}],
  "format": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
}'
```

| field | notes |
|---|---|
| `model` | required |
| `messages` (chat) | `system`, `user`, `assistant`, `tool`; `images` (base64), `thinking`, `tool_calls`, `tool_name`, `tool_call_id`. Empty = preload |
| `prompt`, `system`, `images`, `suffix`, `raw` (generate) | empty prompt with nothing else = preload; an empty `suffix` is none |
| `template` (generate) | empty only (the model's own template); any other is a 400 |
| `tools` | Ollama function tools |
| `stream` | default `true` (NDJSON) |
| `format` | `"json"` or a JSON schema |
| `think` | `true`/`false` maps to `enable_thinking`; `minimal`, `low`, `medium`, `high` to `reasoning_effort` |
| `logprobs`, `top_logprobs` | chat only; `top_logprobs` 0-20 |
| `options` | see below; `null` is none |
| `keep_alive` | any non-null value is a 400 |

Any other non-null field is a 400 `unsupported parameter: <name>`. `raw` and `suffix` requests refuse `system`, `images`, `format`, `think` and `options.stop`.

### Options

| option | constraint |
|---|---|
| `num_predict` | positive, or `-1`/`-2` to fill the context |
| `num_ctx` | must equal the server's context window |
| `temperature` | ≥ 0 |
| `top_p`, `min_p` | 0-1 |
| `top_k` | ≥ 0 |
| `seed` | ≥ 0, or `-1` for random |
| `repeat_penalty` | > 0 |
| `presence_penalty`, `frequency_penalty` | -2 to 2 |
| `stop` | up to four non-empty strings |

Any other option (`num_gpu`, `num_thread`, `mirostat`, ...) is a 400 `unsupported option: <name>`.

### Records

Each streamed line is one JSON object. Reasoning comes back separately as `message.thinking` (chat) or `thinking` (generate). The terminal record:

```json
{"model":"Qwen3-4B","created_at":"...","message":{"role":"assistant","content":""},
 "done":true,"done_reason":"stop","total_duration":812345678,
 "prompt_eval_count":31,"prompt_eval_cached_count":16,"eval_count":42}
```

- `done_reason` is `stop`, `length`, or `load` for a preload.
- `prompt_eval_cached_count` (extension) is the prefix served from the cache.
- `load_duration`, `prompt_eval_duration` and `eval_duration` are not reported for generation.
- Tool calls arrive whole on one record with `done: false`, just before the terminal record; `function.arguments` is an object.
- Errors before the body keep their HTTP status with `{"error": "message"}`; an error mid-stream is one `{"error": "..."}` line.

## Embeddings

```sh
curl http://127.0.0.1:11434/api/embed -d '{"model": "nomic-embed", "input": ["first", "second"]}'
```

| field | default | notes |
|---|---|---|
| `input` | | string or array; empty = preload |
| `truncate` | `true` | `false` refuses oversized input |
| `dimensions` | the model's | truncate and re-normalize |

`llamacpp` and `mlx` serve no embeddings.

## Not supported

- `keep_alive` (residency is the server's: `--idle-timeout` and the model routes)
- model distribution and Modelfiles (501)
- Ollama metadata and phase timings: `/api/tags` lists a digest of the model id, size 0 and empty `details`; `/api/show` ignores `system`, `template` and `options`
- the fleet head does not serve these routes
