# Speculative decoding

A drafter proposes several tokens per round and the target model verifies them in one forward pass. Every draft is verified, so a poor drafter costs speed, never output. Every strategy runs on the `basert` runtime; `prompt-lookup` also runs on `llamacpp` and `mlx`. Measured gates turn speculation off where it does not pay.

```sh
superfluid serve ~/models/Qwen3-4B-Q4.base --speculate auto
superfluid serve ~/models/Qwen3-4B-Q4.base --speculate dflash:~/models/drafters/Qwen3-4B-DFlash-Q4.base
superfluid serve ~/models/GLM-5.2-Q4.base --speculate mtp-head
```

The startup log names the strategy it registered (`superfluid: speculation: mtp-head`).

## Strategies

| directive | drafter | needs |
|---|---|---|
| `off` (default) | none | |
| `auto` | the bundle's own MTP head, else the best installed drafter that fits, else plain decoding | |
| `mtp-head` | the multi-token-prediction head in the target bundle | a bundle converted with the head |
| `prompt-lookup` | n-gram drafting from the lane's own context | nothing; drafts only where text repeats |
| `dflash:<path>` | a DFlash block drafter | a `.base` drafter |
| `dspark:<path>` | a DSpark drafter | a `.base` drafter |
| `eagle3:<path>` | an EAGLE-3 head | a `.base` drafter |
| `draft-model:<path>` | a smaller model with the same tokenizer | a `.base` model |
| `<path>` | whatever the file's header says | a `.base` file |

On `llamacpp` and `mlx`, `prompt-lookup` is the one strategy, and `auto` means `prompt-lookup`. It drafts the tokens that followed the last earlier occurrence of the lane's last three (else two) tokens, so it pays where a reply copies its context: code edits, quoting, structured repeats. Verification is distribution-exact, greedy and sampled. MLX drafts while one lane decodes.

> [!NOTE]
> A model that cannot cut a rejected draft off its KV (recurrent or sliding-window state) refuses `prompt-lookup` at startup in its runtime's words, and `auto` decodes plainly. Any other strategy named on `llamacpp` or `mlx` is refused.

With several models, the directive is resolved per model.

## How `auto` picks a drafter

1. The target bundle's own head, if it has one.
2. Otherwise installed DSpark, DFlash and EAGLE-3 drafters whose header fits the target (same hidden size, vocabulary and backend), searched in the target's directory, a `drafters/` directory beside it or one level up, and the baseRT models cache. Nothing is downloaded. A plain draft model is never picked.
3. Among fitting drafters: one the hub catalog pairs with the target, then one whose name contains the target's, then DSpark, DFlash, EAGLE-3, then the target's quantization, then the newest.

If the engine refuses the first pick, `auto` tries the next and finally decodes plainly. An explicit directive has no fallback: a refusal is a startup error.

The hub catalog is fetched from `https://raw.githubusercontent.com/basecompute/baseRT/main/base-convert/crates/base-hub/catalog.json` into `$SUPERFLUID_HOME/cache/hub-catalog.json` (refreshed daily, never with `--offline`), plus the one baseRT caches at `$BASERT_MODELS_DIR/.catalog-cache.json`. A drafter it lists (a pulled `<org>/<repo>/<variant>/model.base`, or a file under a catalog file name) must match the catalog's size and sha256: `auto` skips one that does not and says so, and an explicit directive naming it is a startup error. Each file is hashed once and its digest kept in `$SUPERFLUID_HOME/cache/sha256/` until the file changes.

## Gates

| gate | flag | default | behaviour |
|---|---|---|---|
| adaptive depth | `--spec-adaptive` | on | picks the fastest draft depth per lane count |
| yield floor | `--spec-min-yield`, `--spec-yield-rounds` | 0.75, 24 | a request whose drafter returns fewer accepted tokens per round stops drafting; two failing requests and none passing abandon the strategy for good |
| throughput gate | `--spec-throughput-gate`, `--spec-min-speedup` | on, 1.08 | measures speculating against plain decoding per lane count and decodes plainly where speculation is not 1.08x faster |
| re-probe | `--spec-gate-reprobe`, `--spec-gate-reprobe-max` | 32, 1024 | after a throughput abandonment, re-measure after 32 requests, doubling up to 1024 |

Speculation is usually a low-concurrency win that can invert under load; the gates keep their verdicts per lane count. All flags are in the [CLI reference](../serving/cli.md#speculative-decoding).

Default draft depth: `mtp-head` lets the engine choose (3; 1 on GLM-DSA), `eagle3` 2, `dflash`/`dspark` the drafter's block width, `prompt-lookup` and `draft-model` 4.

## Interactions

- **Exactness.** Greedy output with speculation equals greedy output without. Sampled output follows the same distribution, but seeded sampled text can differ.
- **Grammars, `logit_bias`, `logprobs`.** These lanes never speculate.
- **Penalties.** Allowed, but drafters predict the unpenalized distribution, so acceptance drops.
- **Temperature.** A flat distribution rejects most drafts. `--spec-max-temperature` caps a temperature that came from the model's default (never one a caller or `--temperature` set).
- **Streaming.** An accepted draft arrives as one chunk carrying several tokens; `stream_options.continuous_usage_stats` reports cumulative token counts per chunk.

## What is reported

- `superfluid.spec` (`proposed`, `accepted`, `acceptance_rate`) on chat replies and the final stream chunk.
- `superfluid_spec_proposed_total` and `superfluid_spec_accepted_total` on `/metrics`.
- `spec accept` in the terminal monitor.

## Expected speedups

With Qwen3-4B q4 at one lane, DSpark, DFlash and EAGLE-3 drafters measured 130, 123 and 98 tokens per second (the order `auto` prefers them in). The Qwen3.8-27B MTP head accepts 41% of its drafts at temperature 1.0 against 59% at temperature 0, which is the measurement behind `--spec-max-temperature`. Measure on your own hardware before relying on a figure.
