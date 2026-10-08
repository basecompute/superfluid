# Chat templates and reasoning

superfluid renders conversations with the model's own chat template and splits the output into text, reasoning and tool-call channels on the model's special tokens. The daemon tokenizes in its own process; workers only see token ids.

## Where the template comes from

| artifact | template |
|---|---|
| GGUF | `tokenizer.chat_template` metadata (and `tokenizer.chat_template.tool_use` for requests with tools) |
| MLX / Hugging Face directory | `chat_template.jinja`, else `tokenizer_config.json`'s `chat_template`, else `chat_template.json` |
| `.base` bundle | the bundle's `chat_template` |

`GET /props` returns the loaded model's raw `chat_template`.

## Dialects

`--dialect` picks the codec that frames the conversation.

| dialect | what it is |
|---|---|
| `auto` (default) | `chatml` if it fits the model, else `template`, else `raw` |
| `chatml` | a curated ChatML codec, used only when the vocabulary has the ChatML markers and the model's own template frames turns the same way |
| `template` | the model's own Jinja template, rendered by the daemon |
| `atem` | the ATEM dialect, for vocabularies with ATEM markers |
| `raw` | no chat scaffold; chat routes refuse (session API only) |

The curated ChatML codec is preferred because it is append-stable: each turn is rendered once, so the prefix cache hits on every turn of an agent loop. Templates are checked at load: one that fails to render falls back to the next dialect; one that is not append-stable still serves correctly but gets no prefix-cache guarantee, and the log says so.

The renderer is `minijinja` with the Hugging Face helpers templates expect (`raise_exception`, `strftime_now`, `tojson`). `strftime_now` returns the date the model was loaded, so renders stay stable within a run.

## Thinking and reasoning effort

```json
{"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
```

| switch | accepted as |
|---|---|
| `enable_thinking` | `chat_template_kwargs.enable_thinking`, top-level `enable_thinking`, Ollama `think: true/false` |
| `reasoning_effort` | `chat_template_kwargs.reasoning_effort`, top-level `reasoning_effort`, Ollama `think: "low"` etc. |

`chat_template_kwargs` wins over the top-level forms. The template decides what each switch means; a value the template rejects is a 400.

`/v1/models` reports what each model supports under `capabilities.dialect`:

| field | meaning |
|---|---|
| `enable_thinking` | the template's output changes with `enable_thinking` |
| `reasoning_effort` | the template reads `reasoning_effort` |
| `reasoning_effort_levels` | the levels it distinguishes |

A requested effort that is not one of the template's levels moves to the nearest one on the scale `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, ties going up.

Reasoning is returned as `reasoning_content` (OpenAI), `thinking` blocks (Anthropic) or `thinking` (Ollama). Send it back on assistant turns as `reasoning_content` or `reasoning`.

## Channels

- `<think>` … `</think>`: reasoning (Qwen3.x, Nemotron, GLM)
- `<|channel>thought` … `<channel|>`: reasoning (Gemma 4)
- `<tool_call>` … `</tool_call>`: tool call (Qwen, Nemotron, GLM)
- `<|tool_call>` … `<tool_call|>`: tool call (Gemma 4)
- Harmony headers (`<|channel|>` … `<|message|>` … `<|end|>`): channel chosen per header (gpt-oss)

A channel is wired only when the model has that marker pair; otherwise its content stays plain text.

## Stop conditions

- Generation ends on any token in the model's stop set (for ChatML, `<|im_end|>` and `<|endoftext|>`), plus dialect terminators (Harmony's `<|return|>` and `<|call|>`, Gemma 4's `<|tool_response>`).
- `stop` strings end generation at, and exclude, the first match. Streams hold back a tail that could still become a stop string.
- `ignore_eos: true` disables all of this and decodes to `max_tokens`.

## Marker text in client content

Marker-shaped text inside client content (`<|im_end|>`, `<think>`, `<tool_call>`) is encoded as ordinary text, so a document that quotes a marker cannot forge a turn boundary. Assistant history is replayed verbatim.

In fleet mode only the head renders; nodes run raw. See [Distributed serving](../serving/distributed_serving.md).
