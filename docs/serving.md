# Serving OmniVoice over HTTP

`omnivoice-serve` turns OmniVoice into a small HTTP service so other programs
(AI agents, chat bots, notification pipelines) can turn **text into speech in a
pre-cloned voice**.

Key design points:

- The model is loaded **once** at startup, not per request.
- A voice is **cloned once** and persisted as a reusable `voice_id`
  (a `VoiceClonePrompt` on disk), so later requests only send text.
- Requests are **serialized** against the single model instance, with a bounded
  queue (`--max-queue`) to avoid unbounded VRAM growth.

## Install

```bash
pip install "omnivoice[serve]"
# or from a clone:
uv sync --extra serve
```

`format=ogg` / `format=mp3` additionally need **ffmpeg** on `PATH`
(`apt install ffmpeg`, `brew install ffmpeg`, `winget install Gyan.FFmpeg`).

## Run

```bash
omnivoice-serve --model k2-fsa/OmniVoice --port 8000
# restrict access:
omnivoice-serve --host 127.0.0.1 --port 8000
# with auth:
OMNIVOICE_API_KEY=secret omnivoice-serve --api-key secret
```

OpenAPI docs are served at `http://localhost:8000/docs`.

Useful flags: `--voices-dir`, `--no-asr` (then `ref_text` is required when
cloning), `--dtype`, `--num-step`, `--opus-bitrate`, `--max-upload-mb`,
`--max-queue`, `--rate-limit`, `--warmup`.

## Workflow

### 1. Clone a voice once

```bash
curl -X POST http://localhost:8000/v1/voices \
  -F "audio=@reference.wav" \
  -F "name=ru-narrator"
```

Optional: `-F "ref_text=Точная расшифровка референса"`. If omitted and ASR is
enabled, Whisper transcribes the clip automatically. Use a **3–10 s** clip.

Response:

```json
{
  "voice_id": "ru-narrator-1a2b3c4d",
  "name": "ru-narrator",
  "ref_text": "…",
  "ref_duration_s": 6.4,
  "created_at": "2026-09-11T10:00:00Z"
}
```

Voices are stored in `--voices-dir` as `<voice_id>.pt` + `<voice_id>.json` and
survive restarts.

### 2. Synthesize (voice message)

```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Привет! Это тестовое голосовое сообщение.","voice_id":"ru-narrator-1a2b3c4d","format":"ogg"}' \
  --output message.ogg
```

`format=ogg` returns **OGG/Opus, mono, 48 kHz** — exactly what Telegram and
WhatsApp expect for a voice message.

Other endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | model, device, dtype, sample rate, ffmpeg availability |
| `GET` | `/v1/voices` | list cloned voices |
| `GET` | `/v1/voices/{voice_id}` | voice metadata |
| `DELETE` | `/v1/voices/{voice_id}` | delete a voice |
| `POST` | `/v1/tts` | text → audio |

## `POST /v1/tts` fields

| Field | Default | Notes |
|---|---|---|
| `text` | — | **required** |
| `voice_id` | — | cloned voice; omit for the auto voice |
| `instruct` | — | voice design, e.g. `"female, british accent"` |
| `language` | auto | name (`"Russian"`) or code (`"ru"`); improves quality |
| `speed` | model default | `>1` faster, `<1` slower |
| `duration` | — | fixed seconds; overrides `speed` |
| `num_step` | `32` | lower = faster |
| `guidance_scale` | `2.0` | CFG scale |
| `denoise` | `true` | |
| `normalize_text` | `false` | expands numbers/dates/currency |
| `format` | `ogg` | `wav` \| `ogg` \| `mp3` |
| `response_format` | `binary` | `binary` (audio bytes) \| `json` (base64) |

Binary responses carry `X-Sample-Rate`, `X-Duration-Seconds`, `X-Audio-Format`
and `X-Voice-Id` headers.

## Example: AI agent (Python)

```python
import httpx

BASE = "http://127.0.0.1:8000"

def clone_voice(path: str, name: str) -> str:
    with open(path, "rb") as f:
        r = httpx.post(
            f"{BASE}/v1/voices", files={"audio": f}, data={"name": name}, timeout=300
        )
    r.raise_for_status()
    return r.json()["voice_id"]

def say(text: str, voice_id: str, out: str = "reply.ogg") -> str:
    r = httpx.post(
        f"{BASE}/v1/tts",
        json={"text": text, "voice_id": voice_id, "format": "ogg"},
        timeout=600,
    )
    r.raise_for_status()
    with open(out, "wb") as f:
        f.write(r.content)
    return out

voice_id = clone_voice("reference.wav", "assistant")   # once
say("Готово, задача выполнена.", voice_id)             # per reply
```

With auth, add `headers={"X-API-Key": "secret"}`.

## Telegram note

Telegram's `sendVoice` requires OGG/Opus **and** a `duration`; it is derived
from the file, so just upload the bytes returned with `format=ogg`.

## Performance and hardware

- **GPU (NVIDIA/recommended):** baseline RTF ≈ 0.09 at batch 1 on an H100, so a
  10 s message takes well under a second. FlashInfer + CUDA graphs push it
  lower — see the README.
- **Apple Silicon (MPS):** works, but slower; the audio tokenizer runs on CPU.
- **CPU:** technically works, but generation is **much** slower (tens of
  seconds for a short sentence). Fine for offline/short messages, not for
  real-time chat. A VPS is a poor fit unless it has a GPU.
- One process = one model. Do **not** scale with multiple uvicorn workers;
  run several processes/ports if you have several GPUs, or scale out across
  machines.

## Security

- The server executes no user code, but it does consume GPU time and can read
  your cloned voices — always set `--api-key` when binding to `0.0.0.0`.
- `voice_id` is validated against `^[A-Za-z0-9_-]{1,64}$` to prevent path
  traversal.
- Voice cloning must only be used with consent and in compliance with the
  project's disclaimer.
