# @basecompute/superfluid

Superfluid is an open-source server that runs LLMs on your own hardware. It serves a model from one machine, or from several machines that act as one server. It supports llama.cpp, MLX, and BaseRT, and exposes OpenAI-, Anthropic-, and Ollama-compatible API endpoints.

```sh
npm install --global @basecompute/superfluid
superfluid serve basecompute/Qwen3.8-27B
```

This package installs the `superfluid` command. The prebuilt release comes in one optional dependency per platform, and npm installs only the one for your machine: `@basecompute/superfluid-darwin-arm64` (macOS 15 or later on Apple silicon), `@basecompute/superfluid-linux-x64` or `@basecompute/superfluid-linux-arm64` (glibc 2.28 or later). Don't install with `--omit=optional`. It has no JavaScript API. The runtimes themselves (llama.cpp, MLX, baseRT) are installed by superfluid on first use, under `~/.superfluid`.

Update it with npm (`npm install --global @basecompute/superfluid@latest`); `superfluid self update` leaves an npm install to npm and says so.

Documentation: <https://superfluid.sh> · Source: <https://github.com/basecompute/superfluid>

## License

Apache-2.0; see LICENSE and NOTICE. The baseRT engine superfluid can install is a separate product under its own license.
