# Anthropic Messages API

`POST /v1/messages` serves the Anthropic Messages API on the same listener as the [OpenAI](openai_compatible_server.md) and [Ollama](ollama_api.md) APIs. Reasoning comes back as `thinking` blocks, text as `text` blocks and parsed tool calls as `tool_use` blocks.

```sh
curl http://127.0.0.1:8453/v1/messages -H 'content-type: application/json' -d '{
  "model": "unsloth/Qwen3.8-27B-GGUF",
  "max_tokens": 256,
  "system": "Be terse.",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}]
}'
```

With the Anthropic Python SDK:

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8453", api_key="unused")
msg = client.messages.create(
    model="unsloth/Qwen3.8-27B-GGUF",
    max_tokens=256,
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(msg.content)
```

No `anthropic-version` header is required. A key may be sent as `x-api-key` or `Authorization: Bearer`.

## Request

| field | default | notes |
|---|---|---|
| `model` | required | a `/v1/models` id |
| `max_tokens` | required | |
| `messages` | required | `content` a string or an array of blocks |
| `system` | none | a string or `text` blocks (joined with newlines) |
| `tools` | `[]` | `{name, description?, input_schema, cache_control?}` |
| `stream` | `false` | |
| `temperature`, `top_p`, `top_k` | model default | resolved as on the OpenAI routes; see [Sampling](../features/sampling.md) |

Ignored: `stop_sequences`, `tool_choice`, `thinking`, `metadata`, `service_tier`. There is no `seed`; a sampled request draws a fresh one.

### Content blocks

| block | mapping |
|---|---|
| `text` | text |
| `image` with `source.type: "base64"` | an image, on a model that accepts images (else 400 `unsupported_input`); URL sources are refused |
| `tool_use` | a prior call in assistant history |
| `tool_result` | a tool result; `content` a string or text blocks. `is_error` is not read |
| `thinking` | prior reasoning in assistant history |

Other block types are skipped.

## Prompt caching (`cache_control`)

A `cache_control` breakpoint on a tool, a `system` block or a content block pins the prompt prefix up to that turn in the KV cache for its TTL, so pressure eviction cannot drop it.

```json
{"system": [{"type": "text", "text": "<long shared preamble>",
             "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}
```

- `type` must be `ephemeral`; `ttl` is `5m` (default) or `1h`.
- At most 4 breakpoints per request; a `1h` breakpoint may not follow a `5m` one.
- Not allowed on `thinking` blocks or empty `text` blocks.
- Pins together hold at most `--pin-budget-pct` of the KV pool (default 50) and yield, oldest first, under pressure.
- Prompts carrying images pin nothing.

## Response

```json
{
  "id": "msg_superfluid_12", "type": "message", "role": "assistant", "model": "Qwen3-4B",
  "content": [
    {"type": "thinking", "thinking": "...", "signature": ""},
    {"type": "text", "text": "..."},
    {"type": "tool_use", "id": "toolu_418", "name": "get_weather", "input": {"city": "Oslo"}}
  ],
  "stop_reason": "tool_use", "stop_sequence": null,
  "usage": {"input_tokens": 24, "cache_creation_input_tokens": 96,
            "cache_read_input_tokens": 0, "output_tokens": 58}
}
```

- `stop_reason` is `end_turn`, `max_tokens` or `tool_use`. A reply cut by `max_tokens` may hold a truncated `tool_use`; do not run it.
- `cache_read_input_tokens` is the prefix served from the prefix cache; `cache_creation_input_tokens` is what this request's breakpoints pinned beyond that.
- A tool block the parser could not read is returned as a `text` block with the raw text.
- `signature` is always empty.

## Streaming

`stream: true` emits the standard event sequence: `message_start`, then `content_block_start` / `content_block_delta` / `content_block_stop` per block, `message_delta` with `stop_reason` and usage, and `message_stop`.

- Text and thinking deltas are sent once per scheduler tick.
- A tool call is emitted when it closes, as one `input_json_delta` carrying the whole arguments.
- Keep-alives are SSE comment lines; there are no `ping` events.
- An error after the stream began is `event: error`.

## Tools and thinking

Tools are rendered through the model's own chat template, as OpenAI function tools are. Decoding is not grammar-constrained on this route (there is no `tool_choice`). Return a `tool_use` block in the assistant turn and answer it with a `tool_result` block in the next user message. See [Tool calling](../features/tool_calling.md).

Whether the model reasons is its template's default; this route passes no template variables. Use the OpenAI route's `chat_template_kwargs` to switch thinking off.

## Errors

```json
{"type": "error", "error": {"type": "invalid_request_error", "message": "..."}}
```

Every daemon-side refusal on this route is a 400 (`invalid_request_error`); a worker failure is a 500 (`api_error`). Refusals made before the handler runs (401, 403, 404, 413, 429) use the OpenAI error envelope.

## Not supported

- `stop_sequences`, `tool_choice`, `thinking` configuration, `metadata`, `service_tier`
- image URLs, audio blocks, documents, `redacted_thinking`
- token-by-token `input_json_delta`, `ping` events
- message batches, token counting, and the Anthropic Files API
