# Quickstart

Serve a model with one command and call it with any OpenAI client. This assumes `superfluid` is [installed](installation.md).

## Serve a model

```sh
superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

On first run this installs the llama.cpp runtime, downloads the model into the Hugging Face cache, and serves it at `http://127.0.0.1:8453`. The runtime is inferred from the model id:

| model | runtime |
|---|---|
| `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` (repo name contains `gguf`) | `llamacpp` |
| `mlx-community/Qwen3-4B-4bit` (`mlx-community/...` or a repo name containing `mlx`) | `mlx` (Apple silicon) |
| any other `org/model` id | `basert` |
| `./Qwen3-4B-Q4_K_M.gguf`, an MLX directory, a `.base` file | picked by the file's format |

A model id is served under its name without the `:tag`, here `unsloth/Qwen3.8-27B-GGUF`. A path is served under its file stem. Gated or private models need `HF_TOKEN` in the environment.

> [!TIP]
> In a terminal the daemon shows a live monitor instead of scrolling logs. Pass `--no-tui` for plain logs.

## Send a request

```sh
curl http://127.0.0.1:8453/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"unsloth/Qwen3.8-27B-GGUF","messages":[{"role":"user","content":"hello"}]}'
```

With the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8453/v1", api_key="unused")
stream = client.chat.completions.create(
    model="unsloth/Qwen3.8-27B-GGUF",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The same server speaks the [Anthropic Messages API](../serving/anthropic_api.md) at `/v1/messages` and the [Ollama API](../serving/ollama_api.md) at `/api/*`.

## List models and check health

```sh
curl http://127.0.0.1:8453/v1/models
curl http://127.0.0.1:8453/health      # {"status":"ok","models_loaded":1,"models_known":1}
```

## Stop

| action | effect |
|---|---|
| `Ctrl-C` / `SIGTERM` | exit at once; in-flight requests are cut |
| `kill -USR1 <pid>` | stop accepting connections, wait up to `--drain-timeout` (60 s) for in-flight requests, exit 0 |
| `q` in the monitor | quit the monitor and the server |

Every committed token is already in the session log under `~/.superfluid/sessions`, so sessions survive a restart.

## Next

- [OpenAI-compatible server](../serving/openai_compatible_server.md): endpoints and parameters
- [Command-line reference](../serving/cli.md): every flag
- [Serving multiple models](../serving/multiple_models.md)
- [Security](../deployment/security.md): read before binding anything but loopback
