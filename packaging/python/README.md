# superfluid

Superfluid is an open-source server that runs LLMs on your own hardware. It serves a model from one machine, or from several machines that act as one server. It supports llama.cpp, MLX, and BaseRT, and exposes OpenAI-, Anthropic-, and Ollama-compatible API endpoints.

```sh
pip install superfluid-cli            # or: pipx install superfluid-cli, uv tool install superfluid-cli
superfluid serve basecompute/Qwen3.8-27B
```

This package is the prebuilt superfluid release for macOS on Apple silicon (macOS 15 or later), Linux on x86-64 or arm64 (glibc 2.28 or later) and Windows on x86-64 (Windows 10 1803 or later, which serves GGUF models through llama.cpp), with the `superfluid` command. It has no Python API and needs nothing from your Python beyond running the command. The runtimes themselves (llama.cpp, MLX, baseRT) are installed by superfluid on first use, under `~/.superfluid`, never into your Python environment.

A pipx or uv tool install keeps itself up to date. Once a release is out (and a few days old), the next `superfluid serve`, `launch`, `node`, `top` or `runtime` command run in a terminal says so, with a link to the release notes, and runs `pipx upgrade superfluid-cli` or `uv tool upgrade superfluid-cli` in the background once no superfluid runs from the install; its output goes to `~/.superfluid/logs/auto-update.log`. `superfluid self update` runs it at once, after asking. superfluid never changes a pip environment itself: once per release it prints the command that updates it (`pip install --upgrade superfluid-cli`, with that environment's Python), as it does for every install on Windows, and `superfluid self update` names it. `superfluid self auto-update off`, `SUPERFLUID_NO_UPDATE=1` or `--no-update` (one run) turn automatic updates off.

Documentation: <https://superfluid.sh> · Source: <https://github.com/basecompute/superfluid>

The package is named `superfluid-cli` because `superfluid` on PyPI is an unrelated project.

## License

Apache-2.0; see LICENSE and NOTICE in the package. The baseRT engine superfluid can install is a separate product under its own license.
