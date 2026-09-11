#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HTTP service for OmniVoice.

Loads the model **once**, exposes a small JSON/multipart API so that other
programs (chat bots, AI agents, notification pipelines, ...) can turn text
into speech in a pre-cloned voice.

Two-step workflow:

1. ``POST /v1/voices`` — upload a short reference clip. The voice is cloned
   once and stored as a reusable ``voice_id`` (a ``VoiceClonePrompt`` on disk).
2. ``POST /v1/tts`` — send ``{"text": "...", "voice_id": "..."}`` and get
   audio back. ``format="ogg"`` returns an OGG/Opus file suitable for voice
   messages in Telegram / WhatsApp.

Usage:
    omnivoice-serve --model k2-fsa/OmniVoice --port 8000
    omnivoice-serve --api-key "$OMNIVOICE_API_KEY"

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field

from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt
from omnivoice.utils.audio import load_audio_bytes
from omnivoice.utils.common import get_best_device

logger = logging.getLogger("omnivoice.serve")

_VOICE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_MIME = {"wav": "audio/wav", "ogg": "audio/ogg", "mp3": "audio/mpeg"}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    model: str = "k2-fsa/OmniVoice"
    voices_dir: Path = field(default_factory=lambda: Path("voices"))
    default_num_step: int = 32
    default_guidance_scale: float = 2.0
    api_key: Optional[str] = None
    ffmpeg: Optional[str] = None
    opus_bitrate: str = "32k"
    max_upload_mb: int = 25
    max_requests_per_second: int = 0  # 0 = unlimited
    max_queue: int = 8


# ---------------------------------------------------------------------------
# Generation gate: one model, so serialize access and bound the queue
# ---------------------------------------------------------------------------


class _BusyError(RuntimeError):
    pass


class _Gate:
    """Serializes access to the single model instance and bounds the queue."""

    def __init__(self, max_queue: int) -> None:
        self._guard = threading.Lock()
        self._gen = threading.Lock()
        self._waiting = 0
        self._max_queue = max(1, max_queue)

    @contextlib.contextmanager
    def slot(self):
        with self._guard:
            if self._waiting >= self._max_queue:
                raise _BusyError(f"server busy ({self._waiting} requests queued)")
            self._waiting += 1
        try:
            with self._gen:
                yield
        finally:
            with self._guard:
                self._waiting -= 1


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize.")
    voice_id: Optional[str] = Field(
        None, description="Cloned voice id from POST /v1/voices. Omit for auto voice."
    )
    instruct: Optional[str] = Field(
        None, description="Voice design instruction, e.g. 'female, british accent'."
    )
    language: Optional[str] = Field(
        None, description="Language name ('English') or code ('en')."
    )
    speed: Optional[float] = Field(None, description=">1 faster, <1 slower.")
    duration: Optional[float] = Field(
        None, description="Fixed output duration in seconds; overrides speed."
    )
    num_step: Optional[int] = None
    guidance_scale: Optional[float] = None
    denoise: Optional[bool] = None
    normalize_text: bool = Field(
        False, description="Normalize numbers/dates/currency before synthesis."
    )
    format: str = Field("ogg", description="wav | ogg | mp3 (ogg = Opus voice msg).")
    response_format: str = Field(
        "binary", description="binary (raw audio) or json (base64)."
    )


class VoiceInfo(BaseModel):
    voice_id: str
    name: Optional[str] = None
    ref_text: Optional[str] = None
    ref_duration_s: Optional[float] = None
    language: Optional[str] = None
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Voice store
# ---------------------------------------------------------------------------


class VoiceStore:
    """Persists cloned voices as ``<voice_id>.pt`` + ``<voice_id>.json``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, VoiceClonePrompt] = {}

    def _pt(self, voice_id: str) -> Path:
        return self.root / f"{voice_id}.pt"

    def _meta(self, voice_id: str) -> Path:
        return self.root / f"{voice_id}.json"

    @staticmethod
    def _check(voice_id: str) -> str:
        if not _VOICE_ID_RE.match(voice_id):
            raise HTTPException(400, f"Invalid voice_id: {voice_id!r}")
        return voice_id

    def new_id(self, name: Optional[str]) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", (name or "voice")).strip("-").lower()
        slug = slug[:40] or "voice"
        return self._check(f"{slug}-{uuid.uuid4().hex[:8]}")

    def save(
        self,
        voice_id: str,
        prompt: VoiceClonePrompt,
        meta: Dict[str, Any],
    ) -> None:
        self._check(voice_id)
        prompt.save(str(self._pt(voice_id)))
        self._meta(voice_id).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._cache[voice_id] = prompt

    def load_prompt(self, voice_id: str) -> VoiceClonePrompt:
        self._check(voice_id)
        cached = self._cache.get(voice_id)
        if cached is not None:
            return cached
        path = self._pt(voice_id)
        if not path.is_file():
            raise HTTPException(404, f"Unknown voice_id: {voice_id!r}")
        prompt = VoiceClonePrompt.load(str(path))
        self._cache[voice_id] = prompt
        return prompt

    def get_meta(self, voice_id: str) -> Dict[str, Any]:
        self._check(voice_id)
        path = self._meta(voice_id)
        if not path.is_file():
            raise HTTPException(404, f"Unknown voice_id: {voice_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[Dict[str, Any]]:
        items: list[Dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover - corrupt sidecar
                continue
            if self._pt(path.stem).is_file():
                items.append(meta)
        return items

    def delete(self, voice_id: str) -> bool:
        self._check(voice_id)
        self._cache.pop(voice_id, None)
        existed = False
        for path in (self._pt(voice_id), self._meta(voice_id)):
            if path.is_file():
                path.unlink()
                existed = True
        return existed


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------


def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _ffmpeg_bytes(
    wav: np.ndarray, sr: int, fmt: str, ffmpeg: str, bitrate: str
) -> bytes:
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-i",
        "pipe:0",
    ]
    if fmt == "ogg":
        # OGG/Opus, mono — what Telegram/WhatsApp expect for voice messages.
        args += [
            "-c:a",
            "libopus",
            "-b:a",
            bitrate,
            "-ar",
            "48000",
            "-ac",
            "1",
            "-application",
            "voip",
        ]
    elif fmt == "mp3":
        args += ["-c:a", "libmp3lame", "-b:a", "128k"]
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unsupported format: {fmt}")
    args += ["-f", fmt, "pipe:1"]

    proc = subprocess.run(
        args, input=np.ascontiguousarray(wav, dtype=np.float32).tobytes(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout


def _encode_audio(
    wav: np.ndarray, sr: int, fmt: str, settings: Settings
) -> tuple[bytes, str]:
    fmt = fmt.lower()
    if fmt not in _MIME:
        raise HTTPException(400, f"Unsupported format {fmt!r}; use wav, ogg or mp3.")
    if fmt == "wav":
        return _wav_bytes(wav, sr), _MIME["wav"]
    if not settings.ffmpeg:
        raise HTTPException(
            503,
            f"format={fmt!r} requires ffmpeg, which was not found on PATH. "
            "Install ffmpeg, or request format='wav'.",
        )
    try:
        data = _ffmpeg_bytes(wav, sr, fmt, settings.ffmpeg, settings.opus_bitrate)
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from e
    return data, _MIME[fmt]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_app(model: OmniVoice, settings: Settings) -> FastAPI:
    store = VoiceStore(settings.voices_dir)
    gate = _Gate(settings.max_queue)
    rate_lock = threading.Lock()
    rate_state = {"window_start": time.monotonic(), "count": 0}

    def _require_key(
        x_api_key: Optional[str] = Header(None),
        authorization: Optional[str] = Header(None),
    ) -> None:
        if not settings.api_key:
            return
        provided = x_api_key
        if not provided and authorization and authorization.lower().startswith("bearer "):
            provided = authorization[7:]
        if not secrets.compare_digest(provided or "", settings.api_key):
            raise HTTPException(401, "Invalid or missing API key.")

    def _check_rate() -> None:
        limit = settings.max_requests_per_second
        if limit <= 0:
            return
        now = time.monotonic()
        with rate_lock:
            if now - rate_state["window_start"] >= 1.0:
                rate_state["window_start"] = now
                rate_state["count"] = 0
            rate_state["count"] += 1
            if rate_state["count"] > limit:
                raise HTTPException(429, "Rate limit exceeded.")

    async def _run_gated(fn):
        def _inner():
            try:
                with gate.slot():
                    return fn()
            except _BusyError as e:
                raise HTTPException(429, str(e)) from e

        return await asyncio.to_thread(_inner)

    app = FastAPI(
        title="OmniVoice API",
        description="Zero-shot multilingual TTS: voice cloning + voice design.",
        version="1.0.0",
    )

    # ---------------------------------------------------------------- health
    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": settings.model,
            "device": str(model.device),
            "dtype": str(next(model.parameters()).dtype),
            "sample_rate": model.sampling_rate,
            "voices": len(store.list()),
            "ffmpeg": bool(settings.ffmpeg),
            "default_format": "ogg",
        }

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "service": "OmniVoice API",
            "docs": "/docs",
            "workflow": [
                "POST /v1/voices  (multipart: audio=@ref.wav, name=my_voice)",
                "POST /v1/tts     (json: {text, voice_id, format: 'ogg'})",
            ],
        }

    # ---------------------------------------------------------------- voices
    @app.get("/v1/voices", dependencies=[Depends(_require_key)])
    def list_voices() -> Dict[str, Any]:
        items = store.list()
        return {"count": len(items), "voices": items}

    @app.post(
        "/v1/voices",
        dependencies=[Depends(_require_key)],
        response_model=VoiceInfo,
    )
    async def create_voice(
        audio: UploadFile = File(..., description="Reference audio, 3-10 s."),
        name: Optional[str] = Form(None, description="Human-readable label."),
        ref_text: Optional[str] = Form(
            None, description="Transcript; omit to auto-transcribe with Whisper."
        ),
        language: Optional[str] = Form(None),
        preprocess_prompt: bool = Form(True),
    ) -> VoiceInfo:
        _check_rate()
        raw = await audio.read()
        if not raw:
            raise HTTPException(400, "Empty audio upload.")
        if len(raw) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(
                413, f"Reference audio exceeds {settings.max_upload_mb} MB."
            )

        voice_id = store.new_id(name)

        def _work() -> Dict[str, Any]:
            wav = load_audio_bytes(raw, model.sampling_rate)  # (1, T)
            ref_seconds = float(wav.shape[-1] / model.sampling_rate)
            prompt = model.create_voice_clone_prompt(
                ref_audio=(torch.from_numpy(wav), model.sampling_rate),
                ref_text=(ref_text or None),
                preprocess_prompt=preprocess_prompt,
            )
            meta = {
                "voice_id": voice_id,
                "name": name or voice_id,
                "ref_text": prompt.ref_text,
                "ref_duration_s": round(ref_seconds, 2),
                "language": language,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            store.save(voice_id, prompt, meta)
            return meta

        logger.info("Cloning voice %s from %s", voice_id, audio.filename)
        meta = await _run_gated(_work)
        return VoiceInfo(**meta)

    @app.get("/v1/voices/{voice_id}", dependencies=[Depends(_require_key)])
    def get_voice(voice_id: str) -> Dict[str, Any]:
        return store.get_meta(voice_id)

    @app.delete("/v1/voices/{voice_id}", dependencies=[Depends(_require_key)])
    def delete_voice(voice_id: str) -> Dict[str, Any]:
        if not store.delete(voice_id):
            raise HTTPException(404, f"Unknown voice_id: {voice_id!r}")
        return {"deleted": voice_id}

    # ------------------------------------------------------------------- tts
    @app.post("/v1/tts", dependencies=[Depends(_require_key)])
    async def tts(req: TTSRequest):
        _check_rate()
        text = req.text.strip()
        if not text:
            raise HTTPException(400, "`text` must not be empty.")

        fmt = req.format.lower()
        if fmt not in _MIME:
            raise HTTPException(
                400, f"Unsupported format {req.format!r}; use wav, ogg or mp3."
            )
        if req.response_format.lower() not in ("binary", "json"):
            raise HTTPException(400, "response_format must be 'binary' or 'json'.")

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(req.num_step or settings.default_num_step),
            guidance_scale=(
                float(req.guidance_scale)
                if req.guidance_scale is not None
                else settings.default_guidance_scale
            ),
            denoise=bool(req.denoise) if req.denoise is not None else True,
        )

        prompt = store.load_prompt(req.voice_id) if req.voice_id else None

        def _work() -> np.ndarray:
            kw: Dict[str, Any] = dict(
                text=text,
                language=req.language or None,
                generation_config=gen_config,
                normalize_text=bool(req.normalize_text),
            )
            if prompt is not None:
                kw["voice_clone_prompt"] = prompt
            if req.instruct and req.instruct.strip():
                kw["instruct"] = req.instruct.strip()
            if req.speed is not None:
                kw["speed"] = float(req.speed)
            if req.duration is not None and float(req.duration) > 0:
                kw["duration"] = float(req.duration)
            audios = model.generate(**kw)
            return np.asarray(audios[0], dtype=np.float32).reshape(-1)

        logger.info(
            "TTS request: voice_id=%s chars=%d format=%s",
            req.voice_id,
            len(text),
            req.format,
        )
        wav = await _run_gated(_work)

        data, mime = _encode_audio(wav, model.sampling_rate, fmt, settings)
        duration = round(len(wav) / model.sampling_rate, 3)
        headers = {
            "X-Sample-Rate": str(model.sampling_rate),
            "X-Duration-Seconds": str(duration),
            "X-Audio-Format": fmt,
        }
        if req.voice_id:
            headers["X-Voice-Id"] = req.voice_id

        if req.response_format.lower() == "json":
            return {
                "audio_base64": base64.b64encode(data).decode("ascii"),
                "mime_type": mime,
                "sample_rate": model.sampling_rate,
                "duration_s": duration,
                "voice_id": req.voice_id,
                "format": fmt,
            }

        return Response(content=data, media_type=mime, headers=headers)

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-serve",
        description="Run the OmniVoice HTTP API server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument(
        "--device", default=None, help="cuda / xpu / mps / cpu. Auto-detected."
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="auto: float16 on CUDA/XPU, float32 elsewhere.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--voices-dir",
        default="voices",
        help="Where cloned voices (<voice_id>.pt + .json) are stored.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Require 'X-API-Key: <key>' or 'Authorization: Bearer <key>'. "
        "Also read from $OMNIVOICE_API_KEY.",
    )
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--opus-bitrate", default="32k")
    parser.add_argument("--max-upload-mb", type=int, default=25)
    parser.add_argument(
        "--max-queue",
        type=int,
        default=8,
        help="Max pending generation requests before returning HTTP 429.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Max requests per second (0 = unlimited).",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        help="Skip Whisper; then ref_text is required when cloning.",
    )
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--asr-device", default=None)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one tiny generation at startup to pay kernel/first-call cost.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    device = args.device or get_best_device()
    if args.dtype == "auto":
        dtype = torch.float16 if device.startswith(("cuda", "xpu")) else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    logger.info("Loading model %s on %s (%s) ...", args.model, device, dtype)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
        asr_device=args.asr_device,
    )
    logger.info("Model loaded. Sample rate: %s Hz", model.sampling_rate)

    settings = Settings(
        model=args.model,
        voices_dir=Path(args.voices_dir),
        default_num_step=args.num_step,
        default_guidance_scale=args.guidance_scale,
        api_key=args.api_key or os.environ.get("OMNIVOICE_API_KEY"),
        ffmpeg=shutil.which("ffmpeg"),
        opus_bitrate=args.opus_bitrate,
        max_upload_mb=args.max_upload_mb,
        max_queue=args.max_queue,
        max_requests_per_second=args.rate_limit,
    )
    if not settings.ffmpeg:
        logger.warning(
            "ffmpeg not found on PATH: format='ogg'/'mp3' will return HTTP 503. "
            "Install ffmpeg for voice messages."
        )
    if args.host not in ("127.0.0.1", "localhost") and not settings.api_key:
        logger.warning(
            "Serving on %s without --api-key. Anyone who can reach the port can "
            "use your GPU and read your voices. Consider --api-key.",
            args.host,
        )

    app = build_app(model, settings)

    if args.warmup:
        logger.info("Warming up ...")
        try:
            model.generate(text="Hello.", generation_config=OmniVoiceGenerationConfig())
        except Exception as e:  # pragma: no cover - best effort
            logger.warning("Warmup failed: %s", e)

    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - depends on install extras
        raise SystemExit(
            "uvicorn is required to run the server. "
            'Install it with: pip install "omnivoice[serve]"'
        ) from e

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
