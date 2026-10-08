# OpenAI Responses API

`POST /v1/responses` serves OpenAI's Responses API on the same listener as the [chat completions API](openai_compatible_server.md). It runs on the chat path (same templates, tool parsing, grammars and sampling) and keeps each stored response's conversation as a durable session, so `previous_response_id` continues a conversation without resending it.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8453/v1", api_key="unused")
first = client.responses.create(
    model="unsloth/Qwen3.8-27B-GGUF",
    instructions="Be terse.",
    input="My name is Ada.",
)
second = client.responses.create(
    model="unsloth/Qwen3.8-27B-GGUF",
    instructions="Be terse.",
    previous_response_id=first.id,
    input="What is my name?",
)
print(second.output_text)
```

## Endpoints

| method | path | purpose |
|---|---|---|
| `POST` | `/v1/responses` | create a response, streaming or not |
| `GET` | `/v1/responses/{id}` | the stored response object |
| `DELETE` | `/v1/responses/{id}` | delete it: `{"id", "object": "response", "deleted": true}` |
| `GET` | `/v1/responses/{id}/input_items` | its input items; `limit` (1-100, default 20), `order` (`desc` default, `asc`), `after` |
| `POST` | `/v1/responses/{id}/cancel` | 400: only background responses can be cancelled, and background mode is not served |

## Request

| field | default | notes |
|---|---|---|
| `model` | required | a `/v1/models` id; with `previous_response_id`, defaults to that response's model |
| `input` | required | a string (one user message) or an array of items, below |
| `instructions` | none | a system message leading this request only |
| `previous_response_id` | none | continue a stored response; see [Conversation state](#conversation-state). Unknown, deleted or not stored: 400 `previous_response_not_found` |
| `store` | `true` | `false`: answered, not kept, not continuable |
| `stream` | `false` | SSE events, below |
| `tools` | `[]` | `function` tools: `{type, name, description?, parameters?, strict?}`; and `namespace` tools, groups of function tools (`{type, name, description?, tools}`) the model sees as `namespace.member` |
| `tool_choice` | `auto` | `auto`, `none`, `required`, or `{"type": "function", "name"}` |
| `text.format` | `text` | `json_object` or `json_schema` (`{name, schema, strict}`); see [Structured outputs](../features/structured_outputs.md) |
| `max_output_tokens` | fill the context, or `--max-tokens` | |
| `temperature`, `top_p`, `seed` | model default | see [Sampling](../features/sampling.md) |
| `reasoning.effort` | model default | the template's `reasoning_effort` when it lists that level, else its thinking switch (off for `none` and `minimal`); ignored by a model with neither |
| `metadata` | `{}` | echoed and stored |
| `parallel_tool_calls`, `truncation` | `true`, `disabled` | echoed; not enforced |

Ignored: `user`, `safety_identifier`, `prompt_cache_key`, `service_tier`, `include`, `top_logprobs`, `max_tool_calls`, `stream_options`.

### Input items

| item | mapping |
|---|---|
| `message`, role `user`, `system`, `developer`, `assistant` | a chat message (`developer` is `system`); content a string or parts |
| part `input_text`, `output_text` | text |
| part `input_image` with `image_url` a `data:` URL | an image, on a model that accepts images (else 400 `unsupported_input`); remote URLs and `file_id` are refused |
| part `input_audio` | audio, as on chat completions |
| `function_call` | a prior call, joined to the assistant turn before it; with `namespace`, a call to that namespace's member |
| `function_call_output` | the call's result (`output` a string or text parts) |
| `reasoning` | prior reasoning (`content` or `summary` text), joined to the assistant turn after it |

Output items are valid input items, so a client that manages its own context can send the previous output back as is. A call to a namespace's member comes out as a `function_call` with the member's `name` and its `namespace`.

## Response

```json
{
  "id": "resp_9f2c...", "object": "response", "created_at": 1760000000, "status": "completed",
  "model": "Qwen3-4B", "previous_response_id": null, "store": true,
  "output": [
    {"id": "rs_9f2c..._0", "type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "..."}], "status": "completed"},
    {"id": "msg_9f2c..._1", "type": "message", "role": "assistant", "status": "completed",
     "content": [{"type": "output_text", "text": "...", "annotations": [], "logprobs": []}]},
    {"id": "fc_9f2c..._2", "type": "function_call", "call_id": "call_9f2c..._2", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}", "status": "completed"}
  ],
  "usage": {"input_tokens": 120, "input_tokens_details": {"cached_tokens": 96},
            "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 12}, "total_tokens": 160},
  "incomplete_details": null
}
```

- The request's `instructions`, `tools`, `tool_choice`, `text`, `temperature`, `top_p`, `max_output_tokens`, `reasoning` and `metadata` are echoed.
- Reasoning is raw chain of thought, in `content` as `reasoning_text`; `summary` is empty.
- `status` is `completed`, or `incomplete` with `incomplete_details.reason: "max_output_tokens"` when the token cap cut the reply. A cut function call is not safe to run.
- `cached_tokens` is the prompt prefix served from the prefix cache (also the `x-superfluid-warm` header); `reasoning_tokens` is the reasoning text's token count.
- Response ids are `resp_` and 128 random bits.

## Streaming

`stream: true` sends `event:` and `data:` lines; every event carries `type` and a `sequence_number` counting from 0. There is no `[DONE]`.

| event | when |
|---|---|
| `response.created`, `response.in_progress` | first; `status: "in_progress"`, no output |
| `response.output_item.added` | an item opens: `reasoning`, `message` or `function_call` |
| `response.content_part.added` | its `output_text` or `reasoning_text` part opens |
| `response.reasoning_text.delta`, `response.output_text.delta` | text at token cadence |
| `response.function_call_arguments.delta` | argument fragments of the open call |
| `response.reasoning_text.done`, `response.output_text.done`, `response.function_call_arguments.done` | the item's full text or arguments |
| `response.content_part.done`, `response.output_item.done` | the part and the item close |
| `response.completed` or `response.incomplete` | last; the full response, already stored |
| `error`, then `response.failed` | the generation failed after the stream began |

A client that disconnects cancels the generation at the next tick, and nothing is stored.

## Conversation state

A stored response records, under `<sessions>/responses/`, the model, the session it ran in, the event after its last, and the API key that created it. The map is written to disk before the response is returned (or `response.completed` is sent) and survives a restart.

`previous_response_id` continues that conversation with the new `input` appended:

- **Forked (warm).** When the model is the same, its codec renders each message on its own, the previous turn ended normally (not cut by `max_output_tokens`), and the conversation's head is the same (the leading system text, the tools when they may be called, and the template arguments such as thinking), the new session is a fork of the stored one after the response's last event. Only the new input is appended and rendered; the stored conversation keeps its exact tokens (including any reasoning) and is served from the prefix cache.
- **Rebuilt.** Otherwise the conversation is rebuilt from the stored items: this request's `instructions`, then each earlier response's input and output, then the new input, rendered as one fresh chat request. Correct, not forked; the prefix cache still serves what matches. This is always the case for chat templates that render whole conversations at once (`renders_per_message` false), whose earlier turns can render differently once a turn follows.

`instructions` are not carried over: each request's own instructions lead its conversation, and a request with other instructions (or none) is rebuilt without the earlier ones. `tools` are not carried over either; send them with every request.

A conversation can switch models; the switch rebuilds it.

## Retention and deletion

- `store: false` returns the response but keeps no record: `GET` is 404, and `previous_response_id` to it is 400 `previous_response_not_found`, as for an unknown or deleted id (`invalid_request_error`, param `previous_response_id`). Its session is still logged in the write-ahead log, as every chat request's is.
- `DELETE` removes the record and purges the response's session (reroot: forks of it keep their own copy of the conversation). A purge is logical removal, not erasure: the events stay in the session log file until it is removed. A deleted response that a later stored response continues stays on disk, hidden, because that response's history is rebuilt from it; it goes when nothing continues it.
- Stored responses are kept until deleted; nothing expires them.

Under `--key-policy`, a key reads, lists, continues and deletes only the responses it created; to it, any other id is unknown (404, or 400 as `previous_response_id`). With only `--api-key`, all responses are shared.

## Not supported

| feature | answer |
|---|---|
| built-in tools (`web_search`, `file_search`, `computer_use`, `code_interpreter`, `image_generation`, `mcp`, custom tools) | 400 `unsupported_tool` |
| `background: true`, `/cancel` | 400 `unsupported_parameter` |
| `conversation`, `prompt` (stored prompts), `item_reference` items, `input_file` parts, `input_image` by `file_id` | 400 `unsupported_parameter` |
| tool choices other than `auto`, `none`, `required`, a function | 400 `unsupported_parameter` |
| `include`, logprobs, `top_logprobs`, annotations, refusals | ignored or empty |
