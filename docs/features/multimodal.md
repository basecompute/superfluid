# Vision and audio

Images and audio can be passed as parts of a chat request, and audio files can be transcribed with a Whisper model. Media support comes from the runtime and the model: only `basert` bundles with a vision or audio tower take media today.

| capability | `basert` | `llamacpp`, `mlx` |
|---|---|---|
| images in chat (`modalities.image_encode`) | bundles with a vision tower | no |
| audio in chat (`modalities.gemma_audio_encode`) | bundles with Gemma's Conformer audio tower | no |
| speech to text (`modalities.whisper_*`) | Whisper bundles | no |

`/v1/models` lists what a model takes in `architecture.input_modalities`. A part the model cannot take is refused before anything is stored:

```json
{"error": {"message": "model 'Qwen3-4B' on runtime llamacpp does not accept images: the llamacpp runtime has no media path",
           "type": "invalid_request_error", "param": "messages[1].content", "code": "unsupported_input"}}
```

## Images in chat

```sh
IMG=$(base64 < tests/fixtures/images/gradient-64.png | tr -d '\n')
cat > req.json <<EOF
{"model": "gemma-4-E2B-it-vl", "max_tokens": 64,
 "messages": [{"role": "user", "content": [
   {"type": "text", "text": "Describe this image."},
   {"type": "image_url", "image_url": {"url": "data:image/png;base64,$IMG"}}]}]}
EOF
curl http://127.0.0.1:8453/v1/chat/completions -H 'content-type: application/json' -d @req.json
```

| API | form |
|---|---|
| OpenAI | `{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}` |
| Anthropic | `{"type": "image", "source": {"type": "base64", "data": "..."}}` |
| Ollama | `"images": ["<base64>", ...]` on the message |
| session API | `PutMedia` then `AppendImage`; or `superfluid media put <file>` |

> [!NOTE]
> Only inline base64 is accepted. Remote URLs are never fetched (400), and the server never reads a local path.

Several images in one message keep their order and the text around each.

## Audio in chat

```json
{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
```

OpenAI route only. WAV only (16-bit PCM or float32); anything else is a 400.

## Speech to text

```sh
curl http://127.0.0.1:8453/v1/audio/transcriptions \
  -F model=whisper-large-v3 -F file=@tests/fixtures/audio/jfk.wav -F response_format=srt
```

`POST /v1/audio/transcriptions` and `/v1/audio/translations` take a multipart form:

| part | default | notes |
|---|---|---|
| `file` | required | WAV |
| `model` | required | a Whisper model |
| `language` | transcribe: `en`; translate: detect | `""` or `auto` detects |
| `prompt` | none | last 8 KiB kept |
| `response_format` | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `timestamp_granularities[]` | | any value upgrades `json`/`text` to `verbose_json`; segment timestamps only |
| `task` | `transcribe` | `translate` on `/transcriptions` translates to English |
| `stream` | `false` | SSE `transcript.text.delta` events, then `transcript.text.done` |
| `temperature` | | accepted and ignored |

## Caching and limits

- Requests with media never seed from or publish to the prefix cache, so `cached_tokens` is 0 and Anthropic `cache_control` pins nothing. `--park` keeps a media session warm for resume through the session API.
- On hybrid (recurrent-state) bundles a media lane runs alone and is never preempted.

| limit | value |
|---|---|
| request body | 100 MiB (base64 adds about a third) |
| placeholder tokens per image | 1 to 4096 |
| context window | placeholder tokens count toward the prompt |

## Media pool

Media bytes are stored once, by SHA-256, under `<sessions>/media/`, durably before any session references them. Blobs are not reference-counted; reclaim space with:

```sh
superfluid media gc      # kept N, removed M
```
