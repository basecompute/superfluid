# pi

pi speaks Chat Completions through a provider that an extension registers for the run.

```sh
npm install -g @earendil-works/pi-coding-agent     # install
superfluid launch pi --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

Aliases: `pi-coding-agent`.

## What launch gives it

| given | what for |
|---|---|
| `-e $SUPERFLUID_HOME/launch/pi-superfluid.ts` | an extension written for the run that registers a `superfluid` provider (`openai-completions`) with the server's `/v1`, its key and the served model with the server's context window |
| `--provider superfluid --model <model>` | the served model |

pi's own settings and `models.json` are not touched.
