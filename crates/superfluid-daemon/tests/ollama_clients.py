"""Opt-in smoke test: pip install ollama==0.6.3; OLLAMA_HOST=... python this_file.

Requires a running superfluid with a chat model supporting think=false and JSON
grammars. Set OLLAMA_MODEL to select a model; defaults to the first listed id.
"""

import json
import os

from ollama import Client, ResponseError

headers = (
    {"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]}
    if os.environ.get("OLLAMA_API_KEY")
    else {}
)
client = Client(
    host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    headers=headers,
    timeout=90,
)
models = client.list().models
assert models
model = os.environ.get("OLLAMA_MODEL") or models[0].model
assert model and client.ps().models
assert client.show(model).capabilities
options = {"temperature": 0, "num_predict": 64}

for stream in (False, True):
    for method in ("chat", "generate"):
        args = (
            {"messages": [{"role": "user", "content": "Say hello in one word."}]}
            if method == "chat"
            else {"prompt": "Say hello in one word."}
        )
        response = getattr(client, method)(
            model=model, stream=stream, think=False, options=options, **args
        )
        chunks = list(response) if stream else [response]
        assert sum(bool(c.done) for c in chunks) == 1
        assert chunks[-1].done and chunks[-1].eval_count > 0
        assert chunks[-1].prompt_eval_count > 0 and chunks[-1].total_duration > 0
        text = (
            "".join(c.message.content or "" for c in chunks)
            if method == "chat"
            else "".join(c.response or "" for c in chunks)
        )
        assert text.strip(), (method, stream, chunks)
        print(method, "stream" if stream else "json", repr(text))

schema = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
response = client.chat(
    model=model,
    messages=[{"role": "user", "content": "Return answer 42 as JSON."}],
    think=False,
    format=schema,
    options=options,
)
assert isinstance(json.loads(response.message.content)["answer"], int)
response = client.generate(
    model=model,
    prompt="The capital of France is",
    raw=True,
    options={"temperature": 0, "num_predict": 16},
)
assert response.done and response.response

for args, status in (
    ({"model": "nonexistent-ollama-smoke-model"}, 404),
    ({"model": model, "keep_alive": "5m"}, 400),
):
    try:
        client.generate(prompt="hi", stream=False, **args)
    except ResponseError as error:
        assert error.status_code == status and isinstance(error.error, str)
    else:
        raise AssertionError("expected an Ollama ResponseError")
print(
    "PASS: Python Ollama client discovery, chat/generate JSON+NDJSON, schema, raw and errors"
)
