# Security

superfluid binds to loopback and asks for no credential unless told to. Nothing in it speaks TLS: the HTTP API is meant for a trusted network or a reverse proxy, the unix socket trusts whoever can open it, and the session log holds every prompt and completion in plain text. To report a vulnerability, see the [security policy](https://github.com/basecompute/superfluid/blob/main/SECURITY.md).

## Listeners

| listener | flag | default | credential |
|---|---|---|---|
| OpenAI / Anthropic / Ollama HTTP | `--http`, `--host`/`--port` | `127.0.0.1:8453` | `--api-key` or `--key-policy`, optional |
| native session API | `--socket` | `~/.superfluid/superfluid.sock` | none: file permissions |
| browser transport | `--web` | off; loopback only | a minted bearer token |
| fleet node agent | `superfluid-noded --listen` | `0.0.0.0:8454` | `--auth-file`; required off loopback |

Without `--api-key` or `--key-policy`, a server on loopback says so in one line; one bound anywhere else warns:

```
superfluid: WARNING --api-key not set. Authentication is OFF — anyone who can reach this port can use the API. Bind loopback or set --api-key.
```

## Requests from web pages

Every HTTP listener refuses, with `403` and code `origin_refused`, a request whose `Origin` header names a page that is not on this machine (`localhost`, `127.0.0.0/8`, `[::1]`). It also checks the `Host` header, which is how a DNS-rebinding page is caught: a listener bound to loopback answers only to a loopback `Host`, and a listener bound elsewhere with no `--api-key` or `--key-policy` answers only to an IP address, `localhost`, or this machine's own hostname (and `<hostname>.local`). To reach a keyless server by another DNS name, set `--api-key`; a keyed server off loopback accepts any `Host`, since a rebound page cannot present the key. Clients that send no `Origin` (curl, the OpenAI and Anthropic SDKs, server-side code) are unaffected, and the server sends no CORS headers.

## API key

```sh
superfluid serve <model> --api-key "$(cat /etc/superfluid/key)"
curl -H 'Authorization: Bearer <key>' http://127.0.0.1:8453/v1/models
curl -H 'x-api-key: <key>' http://127.0.0.1:8453/v1/models
```

- Every route except `GET /health` needs the key, including `/metrics` and the Ollama routes. An unknown path without a key is a 401.
- Keys are compared in constant time against a SHA-256 digest.
- A missing or wrong key is 401 with `type: "authentication_error"`, `code: "invalid_api_key"`.

Admin routes change the server for everyone and need the global key or a policy key with `admin: true`: `POST /v1/models/load`, `/v1/models/unload`, `DELETE /v1/models/{id}`, `POST /v1/lora/load`, `/v1/lora/unload`.

> [!WARNING]
> `POST /v1/models/load` loads any path the daemon's user can read. Never expose an unauthenticated server beyond the host.

## Key policy

`--key-policy <file.json>` loads named keys with per-key scheduling and limits. It turns authentication on: a request must present a listed key or the global `--api-key`.

```json
{
  "keys": [
    {"name": "chat-ui", "key_env": "CHAT_UI_KEY", "class": "interactive", "max_concurrent": 16},
    {"name": "batch", "key": "sk-batch-...", "class": "background", "rate_limit_rpm": 60, "max_concurrent": 2},
    {"name": "ops", "key_env": "OPS_KEY", "class": "agent", "admin": true}
  ]
}
```

| field | default | meaning |
|---|---|---|
| `name` | required | public name in logs and metrics (`[A-Za-z0-9._-]`, unique) |
| `key` / `key_env` | one required | the secret inline, or the environment variable holding it |
| `class` | `agent` | QoS class for the key's requests |
| `max_class` | `class` | best class the `x-superfluid-qos` header may ask for; asking beyond is a 403 |
| `batch_invariant` | `false` | may ask for a batch-invariant lane |
| `rate_limit_rpm` | 0 (unlimited) | per-key token bucket; exempts the key from the per-IP `--rate-limit` |
| `max_concurrent` | 0 (unlimited) | in-flight requests; a stream holds its slot while it streams |
| `admin` | `false` | may use admin routes |

- The file is validated at startup: unknown fields, an empty list, duplicate secrets, a name equal to a secret, or a secret equal to `--api-key` are refused.
- Files and batches are scoped to the key that created them.
- Listing, reading and cancelling one's own files and batches take no concurrency slot.
- A non-admin key scraping `/metrics` sees only its own `superfluid_key_*` rows.

| status | `code` | when |
|---|---|---|
| 429 | `rate_limit_exceeded` | the key's bucket is empty |
| 429 | `concurrency_limit_exceeded` | all `max_concurrent` slots are held |
| 403 | `admin_required` | admin route without `admin` |
| 403 | | QoS header above `max_class`, or batch-invariant without permission |

## Rate limits

`--rate-limit <rpm>` is a token bucket per client IP. `/health`, `/metrics` and `/v1/metrics` are exempt, as are requests under a key with its own `rate_limit_rpm`. Over the limit is 429 `rate_limit_exceeded`.

> [!NOTE]
> The address is the TCP peer's; `X-Forwarded-For` is not read. Behind a reverse proxy every client shares the proxy's address, so rate-limit at the proxy or per key instead.

Fill-in-the-middle completions have their own per-client bucket (`--completion-rate`, `--completion-burst`), keyed on the authenticated credential and peer address.

## QoS headers

| flag | default | meaning |
|---|---|---|
| `--http-default-qos <class>` | `agent` | class for requests that name none |
| `--no-http-qos-header` | honoured | ignore `x-superfluid-qos`, so untrusted clients cannot claim `interactive` and preempt agents |
| `--http-allow-batch-invariant` | refused | allow batch-invariant lanes, which can serialize the server |

## Unix socket

The session API socket has no credential: whoever can connect controls every session and the daemon's log level. The daemon binds it with the process umask. Keep it in a directory private to the daemon's user.

## Browser transport

- Loopback only, enforced at startup.
- A random bearer token is written to `<sessions>/web-token` (mode 0600). `/web/rpc` and `/web/stream` need it in `Authorization: Bearer` and echoed in `X-Superfluid-CSRF`.
- `--web-origin <url>` (repeatable) is the `Origin` allow-list. A request with an unlisted `Origin` is 403; one with no `Origin` relies on the token.
- `/web/ws` requires an allow-listed `Origin`; the token may ride `Sec-WebSocket-Protocol` as `superfluid.token.<token>`, beside `superfluid.v1`.
- `/web/health` answers without the token.
- The listener sends no CORS headers, so a page on another origin can reach only `/web/ws`; see [Session API](../serving/session_api.md#in-a-browser).

## What is stored and logged

| path under `--sessions` | contents | mode |
|---|---|---|
| `wal.log`, `models/<id>/` | every prompt and completion, as tokens and text | umask |
| `files/`, `batches/` | `/v1/files` and `/v1/batches` | umask |
| `media/` | images and audio requests carried | 0600 |
| `park/` | sealed KV, with `--park` | umask |
| `logs/`, `web-token` | JSON logs under the TUI, the `--web` token | 0600 |

Logs, metrics and OTLP traces carry ids, counts, timings and codes, never content. Key secrets are never logged; keys appear by `name`. Session exports include content only with `--include-content`, and send it off the host only with `--allow-content-egress`.

## Code and network egress

- `--basert-lib`, `BASERT_LIB` and `SUPERFLUID_LLAMA_LIB` load a shared library you name.
- An MLX directory that ships its own Python is refused unless `SUPERFLUID_MLX_TRUST_MODEL_CODE=1`. Models pulled by id are fetched without `*.py` files.
- `serve <id>` downloads from the Hugging Face Hub; `--offline` forbids it. `runtime install` downloads over HTTPS with pinned digests.
- OTLP exports and `session export --endpoint` use plain HTTP to the collector you name.

## Fleet links

Head-to-node traffic is unencrypted TCP; prompts and generated tokens cross it in the clear. Run it on a private or already-encrypted network (WireGuard, Tailscale).

- The head sends the bytes of `--fleet-auth <file>`; a node with `--auth-file <path>` compares them in constant time.
- A node bound off loopback refuses to start without `--auth-file`, unless `--insecure-no-auth`.
- `--api-key`, `--key-policy` and `--rate-limit` protect the head's own HTTP listener (a policy's class ceilings do not apply: the head has no scheduler).

## Putting it on a network

1. Keep `--http` on loopback and put a TLS-terminating reverse proxy in front.
2. Set `--api-key`, or `--key-policy` when clients differ in class or limits.
3. Allow long responses through the proxy; `--nonstream-keepalive <sec>` keeps a long non-streaming prefill from being dropped as idle.
4. Do not expose `/metrics` publicly.
5. Do not expose `--web` or the unix socket beyond the host.
6. Leave `--rate-limit` off behind a proxy; limit at the proxy or per key.
7. Keep the fleet node port (8454) and the head's port on the private network.
