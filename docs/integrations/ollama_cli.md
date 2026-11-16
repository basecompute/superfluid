# Ollama CLI

The `ollama` command line is a client of whatever `OLLAMA_HOST` names, and the server answers the [Ollama API](../serving/ollama_api.md).

```sh
# install: https://ollama.com/download
superfluid launch ollama --model basecompute/Qwen3.8-27B     # ollama run <model>
superfluid launch ollama -- list
superfluid launch ollama -- show basecompute/Qwen3.8-27B
```

## What launch gives it

| given | what for |
|---|---|
| `OLLAMA_HOST` | the server's address |
| `run <model>` | the default command; arguments after `--` replace it |

Model names are the server's model ids; `ollama pull` and Modelfiles are not served (see [Ollama API](../serving/ollama_api.md#not-supported)).
