# superfluid-client

A Python client for superfluid's native session API. Sessions live on the server and survive restarts; you can fork one at any event, edit its history into a new branch, cancel a generation and continue it later. It uses only the standard library.

```sh
pip install ./clients/python
superfluid serve <model> --web 127.0.0.1:8460
```

```python
from superfluid_client import Client, Sampling

sf = Client()
chat = sf.session(Sampling(temperature=0.7, seed=1), system="You are a helpful assistant.")
question = chat.user("Plan a three-day trip to Kyoto.")

for chunk in chat.stream(max_tokens=600):
    print(chunk.text, end="", flush=True)

# Three alternative plans from the same point, each with its own seed.
# The forks share the conversation by reference and start from the cache.
for seed in (2, 3, 4):
    alt = chat.fork_after(question, seed=seed)
    print(alt.generate(600).text)

# Rewrite the question; the original branch is untouched.
budget = chat.edit(question, "Plan a three-day trip to Kyoto on a tight budget.")
print(budget.generate(600).text)

# Any session can be picked up again by id, from another process or after a restart.
print(sf.get(chat.id).messages()[-1]["content"])
```

`Client()` reads the URL from `SUPERFLUID_WEB_URL` (default `http://127.0.0.1:8460`) and the token from `SUPERFLUID_WEB_TOKEN`, else `$SUPERFLUID_HOME/sessions/web-token` (`~/.superfluid/sessions/web-token`). Pass `url=`, `token=` or `token_file=` to override.

## Examples

- `examples/fanout.py --context README.md`: reads a document once, then streams four questions about it from forks at the same time.
- `examples/pull_the_plug.py <model>`: starts its own server, kills it with SIGKILL mid-reply, restarts it and continues the same reply.

## Tests

```sh
python -m unittest discover clients/python/tests                  # offline, against a fake server
SUPERFLUID_WEB_URL=http://127.0.0.1:8460 python -m unittest discover clients/python/tests   # also a live server
```

The protocol is documented in [docs/serving/session_api.md](../../docs/serving/session_api.md).
