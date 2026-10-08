# OpenCode

OpenCode speaks Chat Completions through an OpenAI-compatible provider passed in for the run.

```sh
curl -fsSL https://opencode.ai/install | bash      # install
superfluid launch opencode --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

## What launch gives it

| given | what for |
|---|---|
| `OPENCODE_CONFIG_CONTENT` | a config with a `superfluid` provider (`@ai-sdk/openai-compatible`, the server's `/v1` and key) and the served model with the server's context limit, selected as `model` |

OpenCode merges that config over your own, so your providers and settings stay as they are.
