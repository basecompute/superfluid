# Serving multiple models

One daemon can serve several models, on one runtime or several. Each model gets its own worker process and its own session log, and each request is routed by its `model` field.

```sh
superfluid serve ./qwen3-4b.gguf --model mlx-community/Qwen3-4B-4bit \
    --model-dir ~/models --idle-timeout 600
```

| flag | default | meaning |
|---|---|---|
| `--model <model>` | required, repeatable | load at startup; the first is the default for requests that name no model |
| `--model-dir <dir>` | none | every `.base`, GGUF or MLX directory under `<dir>` becomes a known model, loaded on the first request that names it |
| `--idle-timeout <sec>` | 0 (never) | unload a model unused this long (checked every `<sec>/4`); only models that can be loaded again, never the default |
| `--runtime <model>=<id>` | inferred | the runtime for one model |

- A model under `--model-dir` that no installed runtime reads, or whose runtime refuses the serving flags, is skipped with the reason at startup.
- With `--model-dir` and no `--max-context`, the window is a fixed 8192 tokens so on-demand loads find room.
- When two artifacts would get the same name (`qwen.base` and `qwen.gguf`), the second is registered under its full file name; the startup log says so.

> [!NOTE]
> More than one model, `--fim-model` or `--model-dir` all require `--worker-process` (the default).

## List models

```sh
curl http://127.0.0.1:8453/v1/models
```

```json
{"object": "list", "data": [{
  "id": "Qwen3-4B-Q4_K_M", "object": "model", "owned_by": "superfluid",
  "loaded": true,
  "meta": {"n_ctx": 8192},
  "architecture": {"input_modalities": ["text"]},
  "runtime": {"id": "llamacpp", "version": "llama.cpp b11284"},
  "capabilities": {"descriptor_version": 1, "...": "..."}
}]}
```

| field | meaning |
|---|---|
| `id` | the name requests use |
| `loaded` | `false` for a `--model-dir` model not loaded yet |
| `meta.n_ctx` | the context window |
| `architecture.input_modalities` | `text`, plus `image` or `audio` where the model encodes them |
| `runtime`, `capabilities` | the serving runtime and its capability record (loaded models only) |

`GET /v1/models/{model}` returns one entry (404 if unknown). `/health` reports `models_loaded` and `models_known`.

## Load and unload

Admin routes: they need the global `--api-key`, or a policy key with `admin: true`.

```sh
curl -X POST http://127.0.0.1:8453/v1/models/load -d '{"id": "/models/qwen3-4b.gguf"}'
curl -X POST http://127.0.0.1:8453/v1/models/unload -d '{"id": "qwen3-4b"}'
```

| route | body | effect |
|---|---|---|
| `POST /v1/models/load` | `{"id": "<path or known name>", "runtime"?: "<id>"}` | load and make it the default; returns the model object |
| `POST /v1/models/unload` | `{"id": "<name>"}` | unload and free its worker; the default moves to another model |
| `DELETE /v1/models/{model}` | | same as unload |

- `/v1/models/load` takes a path or an already-known name; it does not pull an id. Pull at startup with `--model <id>`.
- The last remaining model cannot be unloaded (400).
- An unloaded model stays known: the next request for it loads it again.
- Loading a model already loaded on a different runtime is refused (`unsupported_runtime`); unload it first.
- Under `--no-worker-process` the route answers 400 `dynamic model loading is not enabled on this server`.

## Per-model state

Each model beside the first keeps its own session log, parks and media under `<sessions>/models/<id>/`, so models never share durable state. Prometheus `/metrics` and `/slots` describe the default model.
