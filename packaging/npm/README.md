# @basecompute/superfluid

One serving daemon for local LLM inference, across runtimes and machines. superfluid puts one scheduler, one durable session log and one set of APIs (OpenAI, Anthropic, Ollama) in front of llama.cpp, MLX and baseRT.

```sh
npm install --global @basecompute/superfluid
superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

This package installs the `superfluid` command. The prebuilt release comes in one optional dependency per platform, and npm installs only the one for your machine: `@basecompute/superfluid-darwin-arm64` (macOS 15 or later on Apple silicon), `@basecompute/superfluid-linux-x64` or `@basecompute/superfluid-linux-arm64` (glibc 2.28 or later). Don't install with `--omit=optional`. It has no JavaScript API. The runtimes themselves (llama.cpp, MLX, baseRT) are installed by superfluid on first use, under `~/.superfluid`.

Update it with npm (`npm install --global @basecompute/superfluid@latest`); `superfluid self update` leaves an npm install to npm and says so.

Documentation: <https://superfluid.sh> · Source: <https://github.com/basecompute/superfluid>

## License

Apache-2.0; see LICENSE and NOTICE. The baseRT engine superfluid can install is a separate product under its own license.
