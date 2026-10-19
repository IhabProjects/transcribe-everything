#!/usr/bin/env python3
"""Transcribe long audio files (hours) with Groq's hosted Whisper API.

Usage:
    transcribe <input.m4a> [--lang fr] [--model whisper-large-v3-turbo|whisper-large-v3]
        [--out transcript] [--prompt "vocabulary hint"] [--timestamps]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

__version__ = "0.1.0"

CHUNK_SECONDS = 8 * 60
OVERLAP_SECONDS = 3
MAX_CHUNK_BYTES = 24 * 1024 * 1024
MIN_SPLIT_SECONDS = 30
DEFAULT_WORKERS = 2
MAX_RETRIES = 8
COST_PER_HOUR = {"whisper-large-v3-turbo": 0.04, "whisper-large-v3": 0.111}
MIN_BILLED_SECONDS = 10  # Groq bills at least 10 s per request


class TranscribeError(Exception):
    """User-facing error: printed without a traceback."""


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #

def check_ffmpeg() -> None:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise TranscribeError(
            f"{' and '.join(missing)} not found on PATH. Install with: brew install ffmpeg"
        )


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise TranscribeError(f"Command failed: {' '.join(cmd[:3])} ...\n{tail}")
    return proc


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        duration = float(proc.stdout.strip())
    except ValueError:
        duration = 0.0
    if proc.returncode != 0 or duration <= 0:
        raise TranscribeError(f"Could not read audio from '{path}'. Is it a valid audio file?")
    return duration


def convert_to_flac(src: Path, dst: Path) -> None:
    run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(dst)])


def cut_chunk(src: Path, dst: Path, start: float, duration: float) -> None:
    run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-i", str(src), "-c:a", "flac", str(dst)])


# --------------------------------------------------------------------------- #
# Chunk planning
# --------------------------------------------------------------------------- #

def chunk_id(start: float, duration: float) -> str:
    return f"{int(round(start * 1000)):010d}_{int(round(duration * 1000)):07d}"


def base_plan(total: float) -> list[tuple[float, float]]:
    plan = []
    start = 0.0
    while start < total:
        dur = min(CHUNK_SECONDS + OVERLAP_SECONDS, total - start)
        plan.append((start, dur))
        if start + dur >= total:
            break
        start += CHUNK_SECONDS
    return plan


def build_plan(flac: Path, total: float, tmp: Path) -> list[tuple[float, float]]:
    """Cut every chunk once to measure it; re-split any chunk above the size cap."""
    final: list[tuple[float, float]] = []
    queue = base_plan(total)
    while queue:
        start, dur = queue.pop(0)
        path = tmp / f"{chunk_id(start, dur)}.flac"
        cut_chunk(flac, path, start, dur)
        if path.stat().st_size <= MAX_CHUNK_BYTES or dur <= MIN_SPLIT_SECONDS:
            final.append((start, dur))
            continue
        path.unlink()
        half = dur / 2
        # Keep the overlap between the two halves so the merge can de-duplicate.
        queue[:0] = [(start, half + OVERLAP_SECONDS / 2), (start + half - OVERLAP_SECONDS / 2,
                                                          half + OVERLAP_SECONDS / 2)]
    return final


# --------------------------------------------------------------------------- #
# Groq API
# --------------------------------------------------------------------------- #

class RateGate:
    """Shared pause so every worker waits when the account hits a rate limit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until = 0.0

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._until = max(self._until, time.time() + seconds)

    def wait(self) -> None:
        while True:
            with self._lock:
                remaining = self._until - time.time()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 5))


def parse_wait_seconds(err) -> float | None:
    """Read retry-after header, or Groq's 'try again in 7m30.5s' message."""
    headers = getattr(getattr(err, "response", None), "headers", None) or {}
    value = headers.get("retry-after")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(err))
    if m and any(m.groups()):
        h, mnt, s = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mnt * 60 + s
    return None


def is_hourly_audio_limit(err) -> bool:
    text = str(err).lower()
    return "seconds of audio per hour" in text or "(asph)" in text or "per day" in text


def transcribe_chunk(client, path: Path, args, gate: RateGate, log) -> dict:
    import groq

    attempt = 0
    while True:
        gate.wait()
        try:
            kwargs = dict(
                model=args.model,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                temperature=0.0,
            )
            if args.lang:
                kwargs["language"] = args.lang
            if args.prompt:
                kwargs["prompt"] = args.prompt
            with open(path, "rb") as fh:
                resp = client.audio.transcriptions.create(file=(path.name, fh.read()), **kwargs)
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            return {
                "text": data.get("text", ""),
                "language": data.get("language"),
                "segments": [
                    {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]}
                    for s in (data.get("segments") or [])
                ],
            }
        except groq.AuthenticationError as e:
            raise TranscribeError("Groq rejected the API key (401). Check GROQ_API_KEY.") from e
        except groq.APIStatusError as e:
            status = e.status_code
            if status != 429 and status < 500:
                raise TranscribeError(f"Groq API error {status} on {path.name}: {e}") from e
            wait = parse_wait_seconds(e)
            if status == 429 and is_hourly_audio_limit(e):
                wait = (wait or 60) + 2
                log(f"Hourly audio limit reached; pausing {wait / 60:.1f} min then resuming...")
                gate.pause(wait)
                continue  # never give up on quota waits
            attempt += 1
            if attempt > MAX_RETRIES:
                raise TranscribeError(f"Giving up on {path.name} after {MAX_RETRIES} retries: {e}") from e
            wait = max(wait or 0, min(2 ** attempt, 120))
            log(f"{path.name}: HTTP {status}, retry {attempt}/{MAX_RETRIES} in {wait:.0f}s")
            if status == 429:
                gate.pause(wait)
            else:
                time.sleep(wait)
        except (groq.APIConnectionError, groq.APITimeoutError) as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise TranscribeError(f"Network error on {path.name}: {e}") from e
            wait = min(2 ** attempt, 120)
            log(f"{path.name}: network error, retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)


# --------------------------------------------------------------------------- #
# Merge and output
# --------------------------------------------------------------------------- #

def merge(plan: list[tuple[float, float]], results: dict[str, dict]) -> list[dict]:
    merged: list[dict] = []
    prev_end = 0.0
    for start, dur in plan:
        for seg in results[chunk_id(start, dur)]["segments"]:
            s, e = seg["start"] + start, seg["end"] + start
            text = seg["text"].strip()
            if not text or s < prev_end:
                continue
            merged.append({"id": len(merged), "start": round(s, 3), "end": round(e, 3), "text": text})
            prev_end = e
    return merged


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def txt_time(t: float) -> str:
    return srt_time(t)[:8]


def write_outputs(out: Path, segments: list[dict], meta: dict,
                  timestamps: bool = False) -> list[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    txt, srt, js = (out.with_name(out.name + ext) for ext in (".txt", ".srt", ".json"))
    lines = (f"[{txt_time(s['start'])}] {s['text']}" if timestamps else s["text"] for s in segments)
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with srt.open("w", encoding="utf-8") as fh:
        for i, s in enumerate(segments, 1):
            fh.write(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n\n")
    js.write_text(json.dumps({**meta, "text": " ".join(s["text"] for s in segments),
                              "segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
    return [txt, srt, js]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def cache_dir_for(src: Path, args) -> Path:
    st = src.stat()
    key = f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|{args.model}|{args.lang}|{args.prompt}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "transcribe-everything"
    return base / f"{src.stem}_{digest}"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="transcribe", description="Transcribe hours of audio with Groq Whisper, cheaply and fast.")
    parser.add_argument("input", type=Path, help="input audio file (.m4a, .mp3, .wav, ...)")
    parser.add_argument("--lang", help="ISO-639-1 language code, e.g. en, fr, ar (auto-detect if omitted)")
    parser.add_argument("--model", default="whisper-large-v3-turbo", choices=list(COST_PER_HOUR))
    parser.add_argument("--out", default="transcript", help="output path prefix (default: transcript)")
    parser.add_argument("--prompt", help="vocabulary / context hint passed to Whisper")
    parser.add_argument("--timestamps", action="store_true", help="prefix each .txt line with [HH:MM:SS]")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel requests (default 2)")
    parser.add_argument("--keep-cache", action="store_true", help="keep chunk cache after success")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()
    if args.lang and not re.fullmatch(r"[a-z]{2,3}", args.lang):
        raise TranscribeError(f"--lang must be a language code like en, fr or ar (got '{args.lang}')")
    if args.workers < 1:
        raise TranscribeError("--workers must be at least 1")

    try:
        from dotenv import load_dotenv
        from groq import Groq
        from tqdm import tqdm
    except ImportError as e:
        raise TranscribeError(f"Missing dependency ({e.name}). Run: pip install -r requirements.txt") from e

    check_ffmpeg()
    src: Path = args.input.expanduser()
    if not src.is_file():
        raise TranscribeError(f"Input file not found: {src}")
    total = probe_duration(src)

    load_dotenv(Path.cwd() / ".env")
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise TranscribeError("GROQ_API_KEY is not set. Export it or put GROQ_API_KEY=... in a .env file.")
    client = Groq(api_key=api_key, max_retries=0, timeout=600)

    cache = cache_dir_for(src, args)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "plan.json"
    tmp = Path(tempfile.mkdtemp(prefix="transcribe_"))
    gate = RateGate()

    try:
        print(f"Input: {src.name} ({total / 60:.1f} min), model {args.model}, "
              f"language {args.lang or 'auto'}")
        print("Converting to 16 kHz mono FLAC...")
        flac = tmp / "full.flac"
        convert_to_flac(src, flac)

        if manifest.exists():
            plan = [tuple(p) for p in json.loads(manifest.read_text())]
        else:
            print("Splitting into chunks...")
            plan = build_plan(flac, total, tmp)
            manifest.write_text(json.dumps(plan))

        results: dict[str, dict] = {}
        todo = []
        for start, dur in plan:
            cid = chunk_id(start, dur)
            cached = cache / f"{cid}.json"
            if cached.exists():
                results[cid] = json.loads(cached.read_text(encoding="utf-8"))
            else:
                todo.append((start, dur))
        if results:
            print(f"Resuming: {len(results)}/{len(plan)} chunks already cached.")

        with tqdm(total=len(plan), initial=len(results), unit="chunk", desc="Transcribing") as bar:
            def work(start: float, dur: float) -> str:
                cid = chunk_id(start, dur)
                path = tmp / f"{cid}.flac"
                if not path.exists():
                    cut_chunk(flac, path, start, dur)
                result = transcribe_chunk(client, path, args, gate, bar.write)
                result.update(start=start, duration=dur)
                part = cache / f"{cid}.json.part"
                part.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                part.replace(cache / f"{cid}.json")
                results[cid] = result
                path.unlink(missing_ok=True)
                return cid

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(work, s, d) for s, d in todo]
                try:
                    for fut in as_completed(futures):
                        fut.result()
                        bar.update(1)
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise

        segments = merge(plan, results)
        languages = sorted({r["language"] for r in results.values() if r.get("language")})
        billed = sum(max(d, MIN_BILLED_SECONDS) for _, d in plan)
        meta = {"source": src.name, "model": args.model, "language": args.lang or ", ".join(languages),
                "duration": round(total, 3), "chunks": len(plan)}
        files = write_outputs(Path(args.out).expanduser(), segments, meta, args.timestamps)

        cost = billed / 3600 * COST_PER_HOUR[args.model]
        print(f"\nDone: {len(segments)} segments, detected language: {', '.join(languages) or 'n/a'}")
        for f in files:
            print(f"  wrote {f}")
        print(f"Estimated cost: ${cost:.3f} ({billed / 3600:.2f} audio hours incl. overlap "
              f"@ ${COST_PER_HOUR[args.model]}/h)")
        if not args.keep_cache:
            shutil.rmtree(cache, ignore_errors=True)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cli() -> None:
    try:
        sys.exit(main())
    except TranscribeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Finished chunks are cached; re-run the same command to resume.",
              file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    cli()
