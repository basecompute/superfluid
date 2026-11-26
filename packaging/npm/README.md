# @basecompute/superfluid

Superfluid is an open-source server that runs LLMs on your own hardware. It serves a model from one machine, or from several machines that act as one server. It supports llama.cpp, MLX, and BaseRT, and exposes OpenAI-, Anthropic-, and Ollama-compatible API endpoints.

```sh
npm install --global @basecompute/superfluid
superfluid serve basecompute/Qwen3.8-27B
```

This package installs the `superfluid` command. The prebuilt release comes in one optional dependency per platform, and npm installs only the one for your machine: `@basecompute/superfluid-darwin-arm64` (macOS 15 or later on Apple silicon), `@basecompute/superfluid-linux-x64` or `@basecompute/superfluid-linux-arm64` (glibc 2.28 or later), or `@basecompute/superfluid-win32-x64` (Windows 10 1803 or later, which serves GGUF models through llama.cpp). Don't install with `--omit=optional`. It has no JavaScript API. The runtimes themselves (llama.cpp, MLX, baseRT) are installed by superfluid on first use, under `~/.superfluid`.

A global install keeps itself up to date. Once a release is out (and a few days old), the next `superfluid serve`, `launch`, `node`, `top` or `runtime` command run in a terminal says so, with a link to the release notes, and starts `npm install --global @basecompute/superfluid@latest` in the background. It runs once no superfluid runs from the install, and its output goes to `~/.superfluid/logs/auto-update.log`. `superfluid self update` runs it at once, after asking. A project's dependency, an npm folder that needs root, and every install on Windows get a line naming the command instead. `superfluid self auto-update off`, `SUPERFLUID_NO_UPDATE=1` or `--no-update` (one run) turn automatic updates off.

Documentation: <https://superfluid.sh> · Source: <https://github.com/basecompute/superfluid>

## License

Apache-2.0; see LICENSE and NOTICE. The baseRT engine superfluid can install is a separate product under its own license.
