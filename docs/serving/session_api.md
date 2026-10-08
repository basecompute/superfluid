# Session API

The session API is superfluid's own protocol. The [OpenAI](openai_compatible_server.md), [Anthropic](anthropic_api.md) and [Ollama](ollama_api.md) routes run each request as a fresh session; the session API keeps sessions in the daemon by id, so a client appends to one, generates on it, forks it and reads it back, across restarts. [Sessions and prefix caching](../features/sessions.md) describes what a session holds.

| transport | flag | encoding | credential |
|---|---|---|---|
| unix socket | `--socket` (default `~/.superfluid/superfluid.sock`) | frames of a `u32` little-endian length and a postcard-encoded `Request` or `Response` | none: the socket's file permissions |
| browser transport | `--web 127.0.0.1:<port>`, off by default | JSON over HTTP and WebSocket | a token minted at every start |

Both carry the same requests and replies. `superfluid session` and `superfluid media` are clients of the socket.

## Over HTTP

```sh
superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --web 127.0.0.1:8455
```

The server prints the listener's address and the path of the token it wrote, `<sessions>/web-token`:

```sh
TOKEN=$(cat ~/.superfluid/sessions/web-token)
rpc() {
  curl -s http://127.0.0.1:8455/web/rpc -H 'content-type: application/json' \
    -H "authorization: Bearer $TOKEN" -H "x-superfluid-csrf: $TOKEN" -d "$1"; echo
}
rpc '{"Create":{"parent":null,"params":{"temperature":0.7,"top_p":0.95,"min_p":0.0,"top_k":0,"seed":1}}}'
# {"Created":{"session":1}}
rpc '{"AppendMessage":{"session":1,"role":1,"text":"Name three rivers."}}'
curl -N http://127.0.0.1:8455/web/stream -H 'content-type: application/json' \
  -H "authorization: Bearer $TOKEN" -H "x-superfluid-csrf: $TOKEN" \
  -d '{"session":1,"max_tokens":256}'
```

A request is the JSON form of one `Request`: its name as the key and its fields as the value, as in `{"Create":{...}}`; a request without fields is a bare string, such as `"Tree"`. A reply is one `Response` in the same form, such as `{"Created":{"session":1}}` or `{"Err":{"message":"..."}}`. Roles are `0` system, `1` user, `2` assistant and `3` tool.

To continue the conversation, append the next user message and stream again: generating after a user message opens a new assistant turn.

## In a browser

A web page can use only `/web/ws`. Browsers apply CORS to `fetch`, and the listener sends no CORS headers, so a page's request to `/web/rpc` or `/web/stream` fails before it is sent. List the page's origin with `--web-origin`, and pass the token in the subprotocol list, since a browser cannot set headers on a WebSocket:

```sh
superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --web 127.0.0.1:8455 --web-origin http://localhost:5173
```

```js
// TOKEN is the contents of web-token, passed in by whatever serves the page.
const ws = new WebSocket("ws://127.0.0.1:8455/web/ws", ["superfluid.v1", `superfluid.token.${TOKEN}`]);
const send = (request) => ws.send(JSON.stringify(request));

ws.onopen = () =>
  send({ Create: { parent: null, params: { temperature: 0.7, top_p: 0.95, min_p: 0, top_k: 0, seed: 1 } } });

ws.onmessage = ({ data }) => {
  const reply = JSON.parse(data);
  if (reply.Created) {
    const session = reply.Created.session;
    send({ AppendMessage: { session, role: 1, text: "Name three rivers." } });
    send({ Generate: { session, max_tokens: 256 } });
  } else if (reply.Delta?.event.body.Generated) {
    document.body.append(reply.Delta.event.body.Generated.text);
  } else if (reply.Generated) {
    console.log("finished", reply.Generated.finish);
  } else if (reply.Err) {
    console.error(reply.Err.message);
  }
};
```

> [!NOTE]
> The token changes at every start and `web-token` is readable only by the daemon's user, so a page cannot load it itself; whatever serves or opens the page has to pass it in.

## Routes

| method | path | request | reply |
|---|---|---|---|
| `POST` | `/web/rpc` | one request | one reply; `Generate`, `Subscribe` and `Unsubscribe` are refused with an `Err` |
| `POST` | `/web/stream` | `{"session", "max_tokens", "provisional"}`; `provisional` defaults to false | server-sent events |
| `GET` | `/web/ws` | a WebSocket with the `superfluid.v1` subprotocol; one request per frame | one reply per text frame; every request, streaming included |
| `GET` | `/web/health` | | `200`, without a token |

`/web/rpc` and `/web/stream` take the token as `Authorization: Bearer` and again as `X-Superfluid-CSRF`; `/web/ws` takes it as `Authorization` or as a `superfluid.token.<token>` subprotocol, and requires an `Origin` listed by `--web-origin`. See [Security](../deployment/security.md#browser-transport).

| status | when |
|---|---|
| `401` | missing or wrong token |
| `403` | `Origin` not listed, or a missing or wrong CSRF header |
| `429` | a `Complete` past the listener's completion bucket, with `Retry-After` and a `Throttled` body |
| `200` | anything else, including a request the daemon refuses, which carries `Err` |

`/web/stream` sends one `data:` line per event:

| data | when |
|---|---|
| `{"committed": <event>}` | a committed `Generated`, `ToolUse`, `ToolResult` or `ToolParseFailure` |
| `{"provisional": {"channel", "text"}}` | text ahead of its commit, with `"provisional": true` |
| `{"done": {"tokens_generated", "finish", "warm_prefix"}}` | the generation ended |
| `{"error": "<message>"}` | the generation failed |
| `[DONE]` | last |

- `channel` is `0` text, `1` reasoning, `2` tool call. `finish` is `1` end of turn, `2` `max_tokens`, `3` grammar, `4` cancelled, `5` error, and `0` on a `Generated` event while its turn is still open.
- `warm_prefix` counts the prompt tokens that came from the cache instead of a prefill.
- Closing the stream ends the generation, as `Cancel` on `/web/rpc` does; it finishes with `4`.

An event is `{"event_id", "epoch", "ts_unix_ms", "body"}`, with `body` one session-log record, such as `{"Generated":{"span":[...],"text":"...","channel":0,"finish":0}}`, or a bare string for a record without fields, such as `"EpochBump"`.

| record | fields |
|---|---|
| `Created` | `parent`, `params` |
| `Forked`, `Rebased` | `parent`, `fork_at`, `params`; `Rebased` also `edits` |
| `Message` | `role`, `text`, `span` |
| `GenerationPrompt` | `span`: the assistant header the turn opened with |
| `Generated` | `span`, `text`, `channel`, `finish` |
| `ToolUse` | `name`, `arguments`; its `event_id` is the `call_id` that `AppendToolResult` answers |
| `ToolResult` | `call_id`, `content`, `span` |
| `Appended` | `text`, `span` |

`span` is the token ids the record put on the model's input.

## Requests

| request | fields | reply |
|---|---|---|
| `Create` | `parent` (a session id or `null`), `params` | `Created {session}` |
| `Append` | `session`, `text` (or `null`), `span` (token ids) | `Committed {event}` |
| `AppendMessage` | `session`, `role`, `text` | `Committed` |
| `AppendSystem` | `session`, `text` (or `null`), `tools` (JSON strings) | `Committed` |
| `AppendBlock` | `session`, `role`, `kind`, `payload` | `Committed` |
| `AppendImage` | `session`, `role`, `blob` (a `PutMedia` hash), `pre_text`, `post_text` | `Committed` |
| `Read` | `session`, `cursor` | `Events {events}`, from event `cursor` on |
| `List` | | `Sessions {ids}` |
| `Sessions` | `include_archived` | `SessionList {sessions}` |
| `Tree` | | `Tree {nodes}`, each `[id, parent, fork_at]` |
| `Inspect` | `session` | `Inspection {inspection}` |
| `SetMeta` | `session`, `expected_version`, `title`, `archived` | `Meta {version}` |
| `Purge` | `session`, `expected_generation`, `mode` (`"Cascade"` or `"Reroot"`) | `Purged {sessions}` |
| `Generate` | `session`, `max_tokens` | `Delta {event}` per committed event, then `Generated {events, tokens_generated, finish, warm_prefix}` |
| `Cancel` | `session` | `Cancelling`, or `Err` when nothing is generating |
| `Subscribe` | `session`, `cursor`, `provisional` | `Subscribed {sub}`, then `SubEvent {sub, event}`, `SubProvisional {sub, channel, text}`, and last `SubEnded {sub, reason}` (`unsubscribed`, `lagged`, `publisher closed` or `purged`) |
| `Unsubscribe` | `sub` | `Unsubscribed` |
| `Complete` | `prefix`, `suffix`, `mode` (`0` PSM, `1` SPM), `max_tokens` | `Completion {text, tokens, cached, expired}`, or `Throttled {retry_after_ms}` |
| `Fork` | `session`, `at_event`, `params` (`null` keeps the parent's) | `Created {session}` |
| `Rebase` | `session`, `edits`, `params` | `Created` |
| `Trim` | `session`, `to_event` | `Created` |
| `AppendToolResult` | `session`, `call_id`, `content` | `Committed` |
| `AppendToolOutcome` | `session`, `call_id`, `outcome` (`1` failed, `2` cancelled), `note` | `Committed` |
| `RequestToolCancel` | `session`, `call_id` | `Committed` |
| `OpenToolCalls` | `session` | `ToolCalls {calls}`, each `[call_id, name, arguments]` |
| `Ledger` | `session` | `Ledger {entries}` |
| `RequestPermission` | `session`, `call_id`, `text` | `Committed` |
| `RespondPermission` | `session`, `request_id`, `granted` | `Committed` |
| `SetQos` | `session`, `class` ([`0` to `3`](../features/scheduling.md)), `batch_invariant` | `Committed` |
| `SetBackgroundRate` | `divisor` | `Ack` |
| `Pin` | `session`, `ttl_ms` (`0` clears) | `Pinned {deadline_unix_ms}` |
| `PutMedia` | `bytes`, `mime` | `Media {hash}` |
| `GcMedia` | | `Gc {kept, removed}` |
| `SetLogLevel` | `directive` | `Ack` |
| `Metrics` | | `Text {text}`, the Prometheus exposition |

Any request can be answered with `Err {message}`. A `Rebase` edit is `{"Drop":{"from","to"}}` or `{"Replace":{"from","to","role","text"}}`. In JSON, `PutMedia`'s `bytes` is an array of numbers.

- A connection runs one request at a time, in order: a `Generate` streams its `Delta` frames and its `Generated` summary before the next request starts. Send `Cancel` on a second connection, or on `/web/rpc`.
- Only committed events carry ids, and a cursor a client holds still resolves after a restart.
- `Subscribe` sends the backlog from the cursor, then live events, deduplicated by id. With `provisional`, text also arrives ahead of its commit as `SubProvisional`.
- `Complete` is metered per connection on the socket and on `/web/ws`, and by one bucket for the whole listener on `/web/rpc`.
- Event ids count up from `0` (`Created`) within a session, and a fork keeps its parent's ids for the events it shares. `Fork {at_event: n}` makes a session holding the parent's events `0` to `n - 1`; the next id of a session is the length of its `Read` from `0`, and forking there keeps everything.
- `params` takes all five fields and fills nothing in from the model's defaults. `temperature` `0` is greedy and `top_k` `0` disables top-k.
- `AppendSystem` opens a session with a system turn declaring `tools`, rendered the way the HTTP routes render the same tools. A tool is an OpenAI function tool (`{"type":"function","function":{"name","description","parameters"}}`), a bare function, or an Anthropic tool with `input_schema`. Templates declare tools only in the first turn, so it is refused once the session has any input, a fork's inherited history included. The model's calls arrive as `ToolUse` events; answer each with `AppendToolResult`.
- Rebase ranges are half-open, sorted and do not overlap, and never include event `0`. Replacing from one message to the end of the session is "edit this message and drop what followed".

## Python client

`clients/python` is a client for `/web/rpc` and `/web/stream` with no dependencies beyond the standard library.

```sh
pip install ./clients/python
superfluid serve <model> --web 127.0.0.1:8460
```

```python
from superfluid_client import Client, Sampling

sf = Client()                       # http://127.0.0.1:8460, token from ~/.superfluid/sessions/web-token
s = sf.session(Sampling(temperature=0.7, seed=1), system="You are a code reviewer.")
q = s.user(open("diff.patch").read())

for chunk in s.stream(max_tokens=512):          # token by token; chunk.kind is "text" or "reasoning"
    print(chunk.text, end="")

security = s.fork_after(q, seed=2)              # the diff is shared by reference and starts warm
security.user("Focus on security.")
print(security.generate(512).text)

fixed = s.edit(q, open("diff-v2.patch").read()) # a new branch with the message replaced
again = sf.get(s.id)                            # the same session, after a restart too
```

```python
weather = {"type": "function", "function": {"name": "get_weather", "description": "Current weather",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
s = sf.session(system="Answer with the tools.", tools=[weather])
s.user("Is it raining in Paris?")
reply = s.generate(256)
for call in reply.tool_calls:                   # ToolCall(id, name, arguments)
    s.tool_result(call, get_weather(**call.json()))   # your function, returning a string
print(s.generate(256).text)
```

| call | request | what it does |
|---|---|---|
| `Client.session(sampling, system=, tools=)` | `Create`; `AppendSystem` with `tools` | new session, opening with the system message and tools given |
| `Session.user(text)`, `.system`, `.assistant` | `AppendMessage` | append a message |
| `Session.declare(system=, tools=)` | `AppendSystem` | open the session with a system turn declaring tools, each a dict or a JSON string |
| `Session.stream(max_tokens)`, `.generate(max_tokens)` | `/web/stream` | generate the next turn; a `Reply` has `text`, `reasoning`, `tool_calls`, `finish`, `tokens`, `warm_prefix` |
| `Session.cancel()` | `Cancel` | stop the generation in flight; leaving a `stream()` loop early does this too |
| `Session.fork(at=, seed=)`, `.fork_after(event)` | `Fork` | new session holding the events before `at` (default: all of them) |
| `Session.edit(event, text)`, `.rebase(edits)` | `Rebase` | new session with a message replaced, or events dropped or replaced |
| `Session.trim(to_event)` | `Trim` | new session holding the events before `to_event` |
| `Session.events(cursor)`, `.messages()` | `Read` | the log, raw or folded into messages |
| `Session.tool_result(call, content)`, `.open_tool_calls()` | `AppendToolResult`, `OpenToolCalls` | answer a tool call the model made |
| `Session.pin(ttl_s)`, `.set_qos(class)` | `Pin`, `SetQos` | hold the prefix in cache; set the scheduling class |
| `Client.call(verb, args)` | any | send a raw request; returns `(variant, payload)` |

`clients/python/examples` has two demos: `fanout.py` reads a document once and streams several questions about it from forks at the same time, and `pull_the_plug.py` kills its own server with SIGKILL mid-reply, restarts it and continues the same reply.

## Behaviour to know

- A session keeps the exact tokens a turn produced, reasoning included. The HTTP routes render each request's history through the chat template, which for some reasoning models (Qwen3) drops earlier turns' reasoning, so a later turn can come out differently from the same conversation sent over HTTP. To give the model a past turn in the template's form, rebase its `Generated` events into one `assistant` message holding the text.
- A generation continued after a cancel or a `max_tokens` cut draws exactly the tokens the uncut generation would have. A fork that keeps its parent's seed shares the parent's random draws, but it computes the end of its prompt in a different batch shape than the parent did, so a sampled fork can drift from the parent's words after a while; give forks their own seeds rather than relying on one to retrace its parent.
- After a crash the history is prefilled again, which can differ from the original decode in the last bits of floating point, so a sampled continuation can drift from what the uninterrupted run would have drawn.
- Provisional text runs ahead of the commit. After a crash, generation resumes from the last committed event, so a client showing provisional text can see the tail after it drawn again.
