# Tool calling

superfluid supports OpenAI-style tool calling on every runtime. Tools are rendered into the prompt by the model's own chat template, and calls are parsed out of the output by a parser learned from that same template, so each model is told the format it was trained on.

## Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8453/v1", api_key="unused")
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]
reply = client.chat.completions.create(
    model="unsloth/Qwen3.8-27B-GGUF",
    messages=[{"role": "user", "content": "What is the weather in Oslo?"}],
    tools=tools,
    tool_choice="auto",
)
print(reply.choices[0].message.tool_calls)
```

To continue, append the assistant message with its `tool_calls` and a `tool` message carrying `tool_call_id` and the result as `content`.

The same tools work on the [Anthropic API](../serving/anthropic_api.md) (`tools` with `input_schema`, `tool_use` / `tool_result` blocks) and the [Ollama API](../serving/ollama_api.md).

## `tool_choice`

| value | tools declared | decoding |
|---|---|---|
| `"auto"` (default) | yes | a structural tag holds any call to its schema; free text otherwise |
| `"none"` | no | unconstrained; the model is never told the tools exist |
| `"required"` | yes | must be a call to some declared tool |
| `{"type":"function","function":{"name":"f"}}` | yes | must be a call to `f`; an undeclared name is a 400 |

- Unknown or malformed values fall back to `auto`.
- `tool_choice` wins over `response_format` when both are set.
- A forced choice whose grammar cannot be compiled is a 400, never served unconstrained.
- On the Anthropic route there is no `tool_choice` and decoding is not constrained.

## Supported call formats

The format is learned at load by rendering one synthetic call through the model's template.

| wire | example | families |
|---|---|---|
| `json` | `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` | Qwen3, many ChatML models |
| delimited XML | `<function=f><parameter=k>v</parameter></function>` | Qwen3.5, Qwen3.6, Nemotron 3 Nano |
| `glm` | `f<arg_key>k</arg_key><arg_value>v</arg_value>` | GLM-5.x |
| `harmony` | a Harmony message addressed `to=functions.f` | gpt-oss |
| `gemma` | `call:fn{k:v}` | Gemma 4 |
| `atem` | ATEM's XML frame | ATEM models |

`--tool-call-parser auto|json|atem|gemma|harmony|glm` overrides the learned format; it is accepted only when it agrees with the model's template.

For wires that write values raw (XML, GLM), each argument is typed from the tool's JSON Schema: a `string` property stays a string even when it reads `8080`.

## Streaming

A streamed call opens with an entry carrying `index`, `id` and `function.name` once the name is known, then `function.arguments` fragments on the same index, then `finish_reason: "tool_calls"`. JSON wires stream raw argument bytes; delimited wires stream one closed argument at a time; Gemma and ATEM calls arrive whole. Fragments concatenate to the non-streaming `arguments`.

## Malformed calls

- The JSON parser repairs common faults (a markdown fence, surrounding prose, trailing commas, an unterminated tail) but never invents a call.
- A call that still does not parse is returned as text in `content`, so output is never silently dropped.
- A reply cut by `max_tokens` finishes with `length`; any call in it may be incomplete and should not be run.

## Reasoning with tools

Return the assistant's reasoning with its tool call as `reasoning_content` (or `reasoning`) on the assistant message. Templates that need prior reasoning in the prompt (gpt-oss, Qwen3.6 with `preserve_thinking`) receive it. See [Chat templates](chat_templates.md).
