# OpenClaw

OpenClaw speaks Chat Completions through a provider in its config. `launch` keeps that in a profile of its own, `~/.openclaw-superfluid`, so your own OpenClaw (its channels, memory and model) is left as it is.

```sh
npm install -g openclaw@latest                     # install
superfluid launch openclaw --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
superfluid launch openclaw -- agent --local -m "summarize this repo"
```

Aliases: `clawdbot`, `moltbot`.

## What launch does

| runs | what for |
|---|---|
| `openclaw --profile superfluid setup --baseline` | the first time, to create the profile |
| `openclaw --profile superfluid config set models.providers.superfluid ...` | a `superfluid` provider (`openai-completions`) with the server's `/v1`, its key and the served model with the server's context window |
| `openclaw --profile superfluid config set agents.defaults.model.primary superfluid/<model>` | the served model as the default |
| `openclaw --profile superfluid tui --local` | the terminal UI on the embedded runtime, which needs no gateway; arguments after `--` replace `tui --local` |
