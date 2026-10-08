# superfluid launch

`superfluid launch <agent>` runs a coding agent already pointed at a local server, starting one when none answers:

```sh
superfluid launch claude --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

| flag | meaning |
|---|---|
| `--http <addr:port>` | the server (default `$SUPERFLUID_HTTP`, else `127.0.0.1:8453`); when none answers there, `superfluid serve <model>` is started in the background, waited on until it lists its models, and left running (log: `$SUPERFLUID_HOME/logs/serve.log`) |
| `--model` | the model to serve or, on a running server, the served model to use; default the model `launch` last started a server with, else the server's first |
| `--api-key` | the server's key (or `SUPERFLUID_API_KEY`), given to the agent |
| `--print` | show the environment, arguments and files instead of running |
| `superfluid launch` | list the agents and whether each is installed |
| `superfluid stop` | stop the server `launch` started |

Arguments after `--` go to the agent:

```sh
superfluid launch claude -- --continue
superfluid launch codex -- exec "fix the failing test"
```

## Agents

`launch` does not edit an agent's own configuration. Each is pointed at the server for that run only, by environment, flags, or a file under `$SUPERFLUID_HOME/launch/` that the run is told to read; OpenClaw gets a profile of its own.

| agent | API | page |
|---|---|---|
| `claude` | Anthropic Messages | [Claude Code](claude_code.md) |
| `codex` | Responses | [Codex](codex.md) |
| `pi` | Chat Completions | [pi](pi.md) |
| `opencode` | Chat Completions | [OpenCode](opencode.md) |
| `hermes` | Chat Completions | [Hermes Agent](hermes.md) |
| `cline` | Chat Completions | [Cline](cline.md) |
| `openclaw` | Chat Completions | [OpenClaw](openclaw.md) |
| `ollama` | Ollama | [Ollama CLI](ollama_cli.md) |

## Context windows

Every agent that takes a context window is told the server's. An agent's first request carries its system prompt and tool definitions, often 15k to 20k tokens, so `launch` warns when the window is too small (under 32k for Claude Code, under the 64k Hermes requires). Restart the server with a larger `--max-context`; see [Configuration](../serving/configuration.md).
