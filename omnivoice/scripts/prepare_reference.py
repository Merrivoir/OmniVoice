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
"""Prepare a clean voice-cloning reference clip.

Takes any audio (voice message exported from Telegram / WhatsApp, `.ogg`,
`.opus`, `.m4a`, `.mp3`, `.wav`, ...) and turns it into a reference file that
OmniVoice clones well: mono, model sample rate, silence trimmed, level
normalized, and within the recommended 3-10 s range.

Voice messages from messengers are **already fine** as input — this script only
makes them *better* (and tells you whether the clip is usable at all).

Usage:
    # Simplest: just clean it up
    python -m omnivoice.scripts.prepare_reference voice.ogg -o refs/ru_narrator.wav

    # Cut a good window out of a longer recording
    python -m omnivoice.scripts.prepare_reference long.m4a -o ref.wav --start 12.5 --target-duration 8

    # Also print the exact transcript prompt for the API/CLI
    python -m omnivoice.scripts.prepare_reference voice.ogg -o ref.wav \
        --text "Привет! Это пример моего голоса для клонирования."

Then clone it:
    curl -X POST http://localhost:8000/v1/voices \\
      -F "audio=@ref.wav" -F "name=ru_narrator" \\
      -F "ref_text=Привет! Это пример моего голоса для клонирования."
"""

from __future__ import annotations

import argparse
import json
import shlex
from typing import Optional

import numpy as np
import soundfile as sf

from omnivoice.utils.audio import load_audio, remove_silence

MIN_RECOMMENDED_S = 3.0
MAX_RECOMMENDED_S = 10.0

# The model rescales references quieter than this RMS up to 0.1, so matching it
# means preprocess_prompt has nothing left to fix.
DEFAULT_TARGET_RMS = 0.1


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))


def normalize_level(
    audio: np.ndarray, target_rms: float, peak_ceiling: float = 0.99
) -> np.ndarray:
    """Scale *audio* to *target_rms*, backing off if that would clip."""
    rms = _rms(audio)
    if rms <= 1e-8:
        return audio
    gain = target_rms / rms
    peak = float(np.max(np.abs(audio))) * gain
    if peak > peak_ceiling:
        gain = peak_ceiling / float(np.max(np.abs(audio)))
    return audio * gain


def prepare(
    path: str,
    sample_rate: int,
    start: Optional[float],
    end: Optional[float],
    target_duration: Optional[float],
    trim_silence: bool,
    target_rms: Optional[float],
) -> tuple[np.ndarray, dict]:
    """Load, trim and normalize *path*; return (waveform (1, T), stats)."""
    audio = load_audio(path, sample_rate)  # (1, T), mono, resampled
    total = audio.shape[-1]
    stats: dict = {"input_seconds": round(total / sample_rate, 2)}

    if start:
        begin = int(round(start * sample_rate))
        if begin >= total:
            raise SystemExit(
                f"--start {start}s is past the end of the file "
                f"({stats['input_seconds']}s)."
            )
        audio = audio[:, begin:]

    if end is not None:
        # Applied before silence removal so the cut stays on the requested
        # boundaries instead of landing inside the next sentence.
        begin = int(round((start or 0.0) * sample_rate))
        stop = int(round(end * sample_rate))
        if stop <= begin:
            raise SystemExit(f"--end {end}s must be greater than --start.")
        audio = audio[:, : stop - begin]

    stats["window_seconds"] = round(audio.shape[-1] / sample_rate, 2)

    if trim_silence:
        before = audio.shape[-1] / sample_rate
        # The same settings create_voice_clone_prompt uses by default.
        audio = remove_silence(
            audio, sample_rate, mid_sil=200, lead_sil=100, trail_sil=200
        )
        stats["trimmed_seconds"] = round(before - audio.shape[-1] / sample_rate, 2)

    if target_duration and audio.shape[-1] / sample_rate > target_duration:
        audio = audio[:, : int(round(target_duration * sample_rate))]
        stats["cut_to_target"] = True

    if target_rms:
        stats["rms_before"] = round(_rms(audio), 4)
        audio = normalize_level(audio, target_rms)
        stats["rms_after"] = round(_rms(audio), 4)

    stats["output_seconds"] = round(audio.shape[-1] / sample_rate, 2)
    stats["peak"] = round(float(np.max(np.abs(audio))), 3)
    return audio, stats


def report(stats: dict, sample_rate: int, text: Optional[str], out: str) -> None:
    if stats["window_seconds"] != stats["input_seconds"]:
        print(
            f"Input:    {stats['input_seconds']}s file -> "
            f"{stats['window_seconds']}s window"
        )
    else:
        print(f"Input:    {stats['input_seconds']}s")
    if "trimmed_seconds" in stats:
        print(f"Silence:  removed {stats['trimmed_seconds']}s")
    if stats.get("cut_to_target"):
        print("Trimmed to requested duration")
    if "rms_before" in stats:
        print(f"Level:    RMS {stats['rms_before']} -> {stats['rms_after']}")
    print(f"Output:   {stats['output_seconds']}s, {sample_rate} Hz, mono, peak {stats['peak']}")
    print(f"Written:  {out}")

    length = stats["output_seconds"]
    print()
    if length < MIN_RECOMMENDED_S:
        print(
            f"WARNING: only {length}s. Under {MIN_RECOMMENDED_S:.0f}s cloning is "
            "unstable — record a longer take."
        )
    elif length > MAX_RECOMMENDED_S:
        print(
            f"WARNING: {length}s is longer than the recommended "
            f"{MAX_RECOMMENDED_S:.0f}s. Cloning still works, but is slower and "
            "slightly less accurate. Re-run with --target-duration 8."
        )
    else:
        print(f"OK: {length}s is in the recommended "
              f"{MIN_RECOMMENDED_S:.0f}-{MAX_RECOMMENDED_S:.0f}s range.")

    if text:
        print()
        print("Next step — clone it (replace the path/text as needed):")
        print(
            "  curl -X POST http://localhost:8000/v1/voices \\\n"
            f"    -F {shlex.quote(f'audio=@{out}')} \\\n"
            f"    -F {shlex.quote('name=my-voice')} \\\n"
            f"    -F {shlex.quote(f'ref_text={text}')}"
        )
        print()
        print("Or as a batch-inference JSONL line:")
        print(
            json.dumps(
                {"id": "sample_001", "text": "<target text>",
                 "ref_audio": out, "ref_text": text},
                ensure_ascii=False,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_reference",
        description="Clean up a voice-cloning reference clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Any audio file; messenger exports are fine.")
    parser.add_argument("-o", "--output", required=True, help="Output WAV path.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Model sample rate (OmniVoice uses 24000).",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Skip this many seconds before processing (pick a better window).",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Stop at this many seconds. Cut on a silence boundary; "
        "--target-duration is applied later and may split the next sentence.",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Cut to at most this many seconds (8 is a good value).",
    )
    parser.add_argument(
        "--no-silence-trim",
        action="store_true",
        help="Keep leading/trailing silence.",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=DEFAULT_TARGET_RMS,
        help="Normalize level to this RMS (0 disables).",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Exact transcript of the clip; prints ready-to-run commands.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    audio, stats = prepare(
        path=args.input,
        sample_rate=args.sample_rate,
        start=args.start,
        end=args.end,
        target_duration=args.target_duration,
        trim_silence=not args.no_silence_trim,
        target_rms=args.target_rms if args.target_rms > 0 else None,
    )

    sf.write(args.output, audio[0], args.sample_rate, subtype="PCM_16")
    report(stats, args.sample_rate, args.text, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
