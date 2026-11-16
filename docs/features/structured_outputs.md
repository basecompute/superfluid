# Structured outputs

Superfluid constrains decoding with a grammar so output matches a JSON schema. It works on every runtime: the baseRT engine compiles grammars itself; llama.cpp, MLX and the mock compile them through llguidance against the model's vocabulary.

## JSON schema

```sh
curl http://127.0.0.1:8453/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "basecompute/Qwen3.8-27B",
  "messages": [{"role": "user", "content": "A planet, as JSON."}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "planet",
    "schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}
}'
```

With the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8453/v1", api_key="unused")
schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
reply = client.chat.completions.create(
    model="basecompute/Qwen3.8-27B",
    messages=[{"role": "user", "content": "A planet, as JSON."}],
    response_format={"type": "json_schema", "json_schema": {"name": "planet", "schema": schema}},
)
print(reply.choices[0].message.content)
```

## `response_format`

| value | effect |
|---|---|
| absent or `{"type": "text"}` | unconstrained |
| `{"type": "json_object"}` | output is a JSON object |
| `{"type": "json_schema", "json_schema": {"schema": {...}}}` | output matches the schema |

`{"type": "json_schema"}` with no `json_schema` object is a 400. A `json_schema` object without a `schema` key is read as the schema itself, so `{"name": "planet"}` alone constrains nothing. On the Responses API, a `json_schema` format without `schema` is a 400. On the Ollama API, `format: "json"` or a schema object does the same. The Anthropic API has no structured-output parameter.

## Rules

- A constraint the client asked for is a contract: a schema that cannot be compiled is a 400 (`response_format grammar did not compile on this engine ...`), never served unconstrained.
- Declared tools win over `response_format`: unless `tool_choice` is `none`, a request with tools is constrained to the tool-call format and `response_format` is not applied. See [Tool calling](tool_calling.md).
- On gpt-oss (Harmony), the schema is applied inside the `final` channel, so the model can still reason first.
- A constrained lane is sampled on the host and never speculates.
- The grammar mask is applied before `logit_bias` and penalties. See [Sampling](sampling.md).
