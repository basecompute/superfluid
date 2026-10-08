# Hermes Agent

Hermes speaks Chat Completions to its `custom` provider.

```sh
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash     # install
superfluid launch hermes --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

Aliases: `hermes-agent`.

## What launch gives it

| given | what for |
|---|---|
| `CUSTOM_BASE_URL` | the server's `/v1` |
| `--provider custom -m <model>` | the served model |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | only for a server behind `--api-key`: Hermes sends a key only to the host it was issued for |

`launch` warns when the window is under 64k tokens, which Hermes refuses to run below. Serve with `--max-context 65536` or more.
