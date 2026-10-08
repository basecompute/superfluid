# Codex

Codex speaks the Responses API, which the server answers at `/v1/responses`.

```sh
npm install -g @openai/codex                       # install
superfluid launch codex --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
superfluid launch codex -- exec "fix the failing test"
```

Aliases: `codex-cli`.

## What launch gives it

| given | what for |
|---|---|
| `-c` overrides | a `superfluid` provider: `base_url` the server's `/v1`, `wire_api = "responses"`, the key read from `SUPERFLUID_LAUNCH_KEY`; `web_search = "disabled"`, since the server runs function tools only |
| `-m <model>` | the served model |
| `$SUPERFLUID_HOME/launch/codex-models.json` | a one-model catalog (`model_catalog_json`) carrying the server's context window |

Codex sends its sub-agent and MCP tools as `namespace` tools, which the Responses route flattens to `namespace.member` functions; see [Responses API](../serving/responses_api.md). The catalog names no freeform `apply_patch` tool, so Codex sends it as a function.
