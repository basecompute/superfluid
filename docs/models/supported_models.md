# Supported models

superfluid serves whatever its runtimes read: GGUF files through llama.cpp, MLX model directories through MLX, and `.base` bundles through the baseRT engine. A model is given as a local path or as a Hugging Face-style id that the runtime pulls.

## Formats and runtimes

| runtime | format | recognized by | tokenizer |
|---|---|---|---|
| `llamacpp` | GGUF | `*.gguf`, or a file starting with `GGUF` | the GGUF vocabulary, via the adapter's library |
| `mlx` | MLX directory (Apple silicon) | a directory with `config.json` and `*.safetensors` | the directory's `tokenizer.json` |
| `basert` | `.base` bundle | `*.base`, or a file starting with `BASE` | the engine's own |

What each runtime supports beyond plain generation:

| feature | `basert` | `llamacpp` | `mlx` |
|---|---|---|---|
| continuous batching, prefix cache | yes | yes | yes |
| tool calling, structured outputs | yes | yes | yes |
| `logit_bias`, penalties, logprobs (up to 20) | yes | yes | yes |
| park and resume (`--park`) | yes | yes (lossless) | yes (lossless) |
| lossy park (`--park-lossy`) | per bundle | no | no |
| `--kv-bits` | yes | no | no |
| speculative decoding | per bundle | no | no |
| images and audio in chat | per bundle | no | no |
| embeddings, speech to text, LoRA | per bundle | no | no |
| `--max-context auto` | yes | on Metal, and on CUDA with a dedicated GPU; else 8192 | yes |

"Per bundle" means the bundle's capability record decides once it is loaded. Anything a model cannot do is refused per request with a 400 naming the reason. See [Runtimes](runtimes.md#capability-records).

## Paths and ids

```sh
superfluid serve ./Qwen3-4B-Q4_K_M.gguf                  # a path; format picks the runtime
superfluid serve ~/models/Qwen3-4B-4bit                   # an MLX directory
superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M           # an id, pulled by llama.cpp
superfluid serve mlx-community/Qwen3-4B-4bit              # an id, pulled by MLX
superfluid serve basecompute/Qwen3-1.7B                   # an id, pulled by basert
```

A token is an **id** when nothing exists on disk by that name, it does not start with `.`, `/` or `~`, it does not end in `.base` or `.gguf`, and the part before an optional `:tag` contains a `/`. Anything else is a path, so a missing file is reported as missing rather than looked up on a hub.

| form | served under |
|---|---|
| a file path | its stem (`./model.gguf` → `model`) |
| a directory | the directory name |
| a `.base` bundle in the hub cache (with `hub.json` beside it) | the id `hub.json` records |
| `org/model[:tag]` | `org/model` (the tag is dropped) |

Runtime inference for ids: a repo name containing `gguf` (any case) goes to `llamacpp`; `mlx-community/...` or a repo name containing `mlx` goes to `mlx`; anything else to `basert`. `--runtime` overrides it.

## Pulling models by id

Pulling happens during `superfluid serve`; there is no separate `pull` command.

| runtime | tool | `:tag` means | `--pull-<name>` options | cache |
|---|---|---|---|---|
| `llamacpp` | `llama download -hf` | the quant (`Q4_K_M`) | `--pull-file <name>`, `--pull-no-mmproj` | Hugging Face cache |
| `mlx` | `huggingface_hub.snapshot_download` | a revision | `--pull-revision <rev>` | Hugging Face cache (no `*.py` files are fetched) |
| `basert` | `basert resolve` | a variant | `--pull-target <scheme>`, `--pull-profile <json>`, `--pull-revision <rev>`, `--pull-force` | the baseRT models cache (`BASERT_MODELS_DIR`) |

- `--offline` only looks in the cache and fetches nothing.
- An undeclared `--pull-*` option, or `--pull-*` with no model given by id, is refused before anything is fetched.
- A gated or private model needs `HF_TOKEN`.
- One pull of an id runs at a time across daemons.

## Published sampling defaults

When a request names no sampling parameter, the model author's published defaults apply (after any operator `--temperature` etc.):

| runtime | source |
|---|---|
| `llamacpp` | GGUF `general.sampling.*` metadata (older conversions carry none) |
| `mlx` | `generation_config.json` |
| `basert` | the bundle header |

They appear in `superfluid runtimes <model>`, `/v1/models` and `/props`. See [Sampling](../features/sampling.md).

## Context and batch sizing

| flag | default | meaning |
|---|---|---|
| `--max-context <N>` | auto | per-lane window, sized for the device: `basert` before loading, `llamacpp` on Metal or a dedicated CUDA GPU, `mlx` on Metal; elsewhere 8192 |
| `--max-batch <N>` | 8 | concurrent lanes |
| `--kv-bits <N>` | 0 (auto) | KV type on `basert` only |

On llama.cpp and MLX the KV pool is `--max-batch` × `--max-context` cells, capped by the memory the weights leave free; the cap is printed at load. A prompt that does not fit the window is refused with `context_length_exceeded`.

## Checking a model

```sh
superfluid runtimes ./Qwen3-4B-Q4_K_M.gguf
```

Prints which runtime reads it, the model's architecture and sampling defaults, and what that runtime serves and refuses. Exits 1 when no installed runtime serves it.
