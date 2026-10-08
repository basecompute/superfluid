# Claude Code

Claude Code speaks the Anthropic Messages API, which the server answers at `/v1/messages`.

```sh
curl -fsSL https://claude.ai/install.sh | bash     # install
superfluid launch claude --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
superfluid launch claude -- --continue             # arguments after -- go to claude
```

Aliases: `claude-code`.

## What launch gives it

| given | what for |
|---|---|
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` | the server and its key (a placeholder on a keyless server) |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL` | the served model, for every tier Claude Code would pick |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | the server's context window, so Claude Code compacts in time instead of assuming 200k |
| `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | keeps the system prompt identical between sessions, so a new session reuses the cached prefix of the last one |
| `CLAUDE_CODE_AUTO_MODE_SERVER=0` | set unless your environment sets it: auto mode's permission classifier otherwise runs on Anthropic's servers |
| `ANTHROPIC_API_KEY` | removed from the environment, so no key is sent beside the token |

`launch` warns when the window is under 32k tokens: Claude Code's first request alone is about 17k. Serve with `--max-context 65536` or more.
