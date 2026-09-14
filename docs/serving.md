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
If you cannot add it to `PATH` (common on Windows), point the server at it
directly with `--ffmpeg-path "C:\ffmpeg\bin\ffmpeg.exe"`.

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
cloning), `--dtype`, `--num-step`, `--opus-bitrate`, `--ffmpeg-path`,
`--max-upload-mb`, `--max-queue`, `--rate-limit`, `--warmup`.

## Workflow

### 0. Prepare a reference clip (recommended)

**A voice message from Telegram/WhatsApp works as-is** — Opus-in-OGG decodes
fine (`libsndfile` ≥ 1.1), and the loader falls back to ffmpeg for `.m4a` and
friends. The helper just makes it *better* and checks it is usable:

```bash
python -m omnivoice.scripts.prepare_reference voice.ogg -o refs/ru_narrator.wav \
  --target-duration 8 \
  --text "Привет! Это пример моего голоса для клонирования."
```

It converts to mono at the model sample rate, trims silence, normalizes the
level, warns if the clip is outside the recommended 3–10 s, and prints a
ready-to-run `curl` / JSONL snippet. Options: `--start` / `--end` (cut a
better window out of a long recording), `--target-duration`,
`--no-silence-trim`, `--target-rms`.

Cutting a window out of a long recording:

```bash
python -m omnivoice.scripts.prepare_reference long.m4a -o refs/olya.wav \
  --start 15.75 --end 24.55 \
  --text "У нас играют и новички и опытные игроки, и вы можете выбрать именно по своему уровню."
```

Use `--start`/`--end` rather than `--target-duration` when the recording has
several sentences: `--end` cuts before silence removal, so it lands on the
boundary you asked for, while `--target-duration` trims *after* silence
removal has already joined the sentences together and can clip the next one.
`ref_text` must then be the transcript of **that window only** — the server
does not verify the alignment, a mismatch degrades the clone.

To find good boundaries, list the pauses:

```bash
ffmpeg -hide_banner -i long.m4a -af "silencedetect=noise=-35dB:d=0.25" -f null -
```

What to record:

- **3–10 s** of continuous, natural speech. Under 3 s cloning is unstable;
  over 10 s is slower with no quality gain.
- **Same language as the target speech.** Cross-lingual cloning keeps the
  reference's accent.
- **No** music, reverb, echo, or background voices. One speaker only.
- **Your normal speaking voice**, not a "presenter" voice — the clone copies
  whatever you do, including any unnatural delivery.
- Read a sentence or two in the *style* you want back: calm narration, lively
  assistant, etc. Cloning copies the emotion and pace too.
- Keep numbers as words ("двадцать три", not "23") if you also supply
  `ref_text`; use `normalize_text` at synthesis time for the target text.

Always pass `ref_text` with the exact transcript when you can — it is more
reliable than Whisper auto-transcription, and it prevents the server from
trimming the audio.

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
WhatsApp expect for a voice message. It requires ffmpeg; check `"ffmpeg"` in
`GET /health` before requesting `ogg`/`mp3`, otherwise the request is processed
and *then* rejected with HTTP 503.

#### Windows / PowerShell

In PowerShell `curl` is an alias for `Invoke-WebRequest`, which does not accept
the `-H`/`-d`/`--output` syntax above. Use `curl.exe` explicitly (it ships with
Windows 10+) or the native cmdlet:

**Do not inline the JSON body.** PowerShell 5.1 rewrites native-command arguments:
plain `"` inside single quotes is stripped (→ HTTP 422), while escaping to `\"`
confuses the argument splitter, so a body containing spaces is cut apart and curl
tries to resolve fragments like `Пора` as hostnames (`URL rejected: Bad hostname`).
Use the cmdlet, or pass the body from a file:

```powershell
# simplest — no quoting at all
Invoke-RestMethod -Uri http://localhost:8000/v1/tts -Method Post `
  -ContentType 'application/json' `
  -Body '{"text":"Пора спать! Завтра на тренировку","voice_id":"ru-narrator-1a2b3c4d","format":"ogg"}' `
  -OutFile message.ogg
```

```powershell
# curl.exe with the body read from a file (UTF-8, no BOM)
$body = '{"text":"Пора спать! Завтра на тренировку","voice_id":"ru-narrator-1a2b3c4d","format":"ogg"}'
[IO.File]::WriteAllText("$env:TEMP\tts.json", $body, [Text.UTF8Encoding]::new($false))
curl.exe -X POST http://localhost:8000/v1/tts `
  -H "X-API-Key: <key>" `
  -H "Content-Type: application/json" `
  -d "@$env:TEMP\tts.json" `
  --output message.ogg
```

`Invoke-RestMethod` handles UTF-8 bodies and non-ASCII `OutputFile` paths correctly;
prefer it. If you must use `curl.exe`, the `-d "@file"` form is the only reliable one
on PowerShell 5.1.

If the server was started in a terminal whose `PATH` predates the ffmpeg install,
`/health` reports `"ffmpeg": false`. Restart it from a fresh shell (or pass
`--ffmpeg-path`) — the **server** resolves ffmpeg once at startup, not the client.

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
| `num_step` | server default | iterative unmasking steps; **the effective default is the server's `--num-step`**, which may be lower than the model's 32 |
| `guidance_scale` | `2.0` | CFG scale |
| `denoise` | `true` | |
| `normalize_text` | `false` | expands numbers/dates/currency |
| `format` | `ogg` | `wav` \| `ogg` \| `mp3` |
| `response_format` | `binary` | `binary` (audio bytes) \| `json` (base64) |

Binary responses carry `X-Sample-Rate`, `X-Duration-Seconds`, `X-Audio-Format`
and `X-Voice-Id` headers.

## Getting the clone to sound like the speaker

If the output does not resemble the reference, check these roughly in order —
the first two account for most "it sounds nothing like her" reports, and both
are *server-side* settings rather than anything wrong with the reference.

1. **`num_step`.** This is the model's quality dial and the most common cause.
   The model default is 32, but the *effective* default is whatever `--num-step`
   the server was started with. If you followed the low-spec advice below and
   started with `--num-step 8`, every request silently runs at a quarter of the
   normal quality unless it overrides `num_step` in the JSON body. Measured on
   the same text and voice: 8 steps → 75% of frames carry speech, 32 steps →
   85%. Prefer starting the server at the default 32 and lowering `num_step`
   **per request** only where latency matters.
2. **`format`.** `ogg`/`mp3` are lossy *and* downsampled to 48 kHz re-encoded
   Opus at `--opus-bitrate` (32k by default, with `-application voip`, which
   optimises for telephone intelligibility rather than timbre). To judge
   similarity, always compare in `format=wav` — the server writes PCM without
   ffmpeg and without loss. Raise `--opus-bitrate` to `64k`/`96k` for delivery.
3. **`language`.** Leave it out and the language is auto-detected; passing
   `"Russian"` explicitly improves quality. Not passing it is a needless risk.
4. **`voice_id`.** Omitting it uses the built-in auto voice — there is nothing
   to sound similar *to*. A clone must be created first via `POST /v1/voices`.
5. **`instruct`.** When it agrees with the reference, it improves cloning
   stability for the attributes it names (see `docs/tips.md`). When it
   conflicts, the reference wins.
6. **The reference's delivery.** Cloning copies emotion, pace and register, not
   just timbre. A bouncy sales-pitch clip produces a bouncy clone; a snippet
   with a flat "presenter" tone produces a flat clone. Pick a window whose
   delivery matches how you want the output to sound, and prefer a neutral
   clip over a theatrical one.

Hardware does **not** affect similarity directly: with identical decoding
parameters the output is the same on a laptop or an H100. What a GPU buys is the
ability to *afford* the settings that do matter — full `num_step`, larger
references, and many candidates to A/B. See FlashInfer in the README.

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

Squeezing it onto a weak machine:

- `--num-step 8` (instead of the default 32) cuts the denoising loop ~4x and
  is the single biggest speed-up. It is also the biggest hit to similarity —
  see [Getting the clone to sound like the speaker](#getting-the-clone-to-sound-like-the-speaker).
  Prefer keeping the server at 32 and passing a lower `num_step` per request
  for the messages where latency matters.
- `--no-asr` skips loading Whisper. Always do this on a low-RAM box — and then
  `ref_text` becomes **required** when cloning; auto-transcription is off.
- `--dtype bfloat16` roughly halves the memory the weights occupy, but on CPUs
  without AVX512-BF16 it is emulated and therefore slower. Try `float32`
  first, fall back to `bfloat16` if you run out of RAM.
- Budget: the weights are 2.45 GB and the audio tokenizer 0.81 GB before any
  activations, so a clone + synthesis needs roughly 5–6 GB of free RAM.

## Security

- The server executes no user code, but it does consume GPU time and can read
  your cloned voices — always set `--api-key` when binding to `0.0.0.0`.
- `voice_id` is validated against `^[A-Za-z0-9_-]{1,64}$` to prevent path
  traversal.
- Voice cloning must only be used with consent and in compliance with the
  project's disclaimer.
