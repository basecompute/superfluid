# Cline

Cline speaks Chat Completions through its `openai-compatible` provider.

```sh
npm install -g cline                               # install
superfluid launch cline --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

Aliases: `cline-cli`.

## What launch gives it

| given | what for |
|---|---|
| `CLINE_PROVIDER_SETTINGS_PATH` | `$SUPERFLUID_HOME/launch/cline-providers.json`, a provider settings file written for the run: an `openai-compatible` provider with the server's `/v1`, its key and the served model |

Your own providers, MCP servers and sessions stay in use. No `-P` or `-m` is passed: given them, Cline saves them over the file.
