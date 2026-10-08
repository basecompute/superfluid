# Ollama CLI

The `ollama` command line is a client of whatever `OLLAMA_HOST` names, and the server answers the [Ollama API](../serving/ollama_api.md).

```sh
# install: https://ollama.com/download
superfluid launch ollama --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M               # ollama run <model>
superfluid launch ollama -- list
superfluid launch ollama -- show unsloth/Qwen3.8-27B-GGUF
```

## What launch gives it

| given | what for |
|---|---|
| `OLLAMA_HOST` | the server's address |
| `run <model>` | the default command; arguments after `--` replace it |

Model names are the server's model ids; `ollama pull` and Modelfiles are not served (see [Ollama API](../serving/ollama_api.md#not-supported)).
