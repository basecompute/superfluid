# superfluid

One serving daemon for local LLM inference, across runtimes and machines. superfluid puts one scheduler, one durable session log and one set of APIs (OpenAI, Anthropic, Ollama) in front of llama.cpp, MLX and baseRT.

```sh
pip install superfluid-cli            # or: pipx install superfluid-cli, uv tool install superfluid-cli
superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

This package is the prebuilt superfluid release for macOS on Apple silicon (macOS 15 or later) and Linux on x86-64 or arm64 (glibc 2.28 or later), with the `superfluid` command. It has no Python API and needs nothing from your Python beyond running the command. The runtimes themselves (llama.cpp, MLX, baseRT) are installed by superfluid on first use, under `~/.superfluid`, never into your Python environment.

Update it with pip (`pip install --upgrade superfluid-cli`); `superfluid self update` leaves a pip install to pip and says so.

Documentation: <https://superfluid.sh> · Source: <https://github.com/basecompute/superfluid>

The package is named `superfluid-cli` because `superfluid` on PyPI is an unrelated project.

## License

Apache-2.0; see LICENSE and NOTICE in the package. The baseRT engine superfluid can install is a separate product under its own license.
