# transcribe-everything

**Transcribe a 2.5-hour recording in under a minute, for about $0.10.**

[![CI](https://github.com/IhabProjects/transcribe-everything/actions/workflows/ci.yml/badge.svg)](https://github.com/IhabProjects/transcribe-everything/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)

A single-file Python CLI that turns long audio (meetings, lectures, interviews, podcasts) into clean
`.txt`, `.srt` subtitles and `.json`, using [Groq](https://groq.com)'s hosted Whisper models.
It handles the parts that break on long files: splitting, file size limits, rate limits, the hourly
audio quota, and crashes halfway through.

```console
$ transcribe meeting.m4a --lang fr
Input: meeting.m4a (148.9 min), model whisper-large-v3-turbo, language fr
Converting to 16 kHz mono FLAC...
Splitting into chunks...
Transcribing: 100%|██████████| 19/19 [00:49<00:00,  2.62s/chunk]

Done: 2172 segments, detected language: French
  wrote transcript.txt
  wrote transcript.srt
  wrote transcript.json
Estimated cost: $0.100 (2.50 audio hours incl. overlap @ $0.04/h)
```

That's a real run: **149 minutes of audio, 49 seconds, ten cents.**

## Why

| | Long files | Speed (2.5 h audio) | Cost | Your GPU |
| --- | --- | --- | --- | --- |
| **transcribe-everything** | chunked + resumable | **~1 min** | **~$0.10** | not needed |
| Local Whisper on a laptop | yes | 30 min – hours | free | needed |
| OpenAI Whisper API | 25 MB limit, DIY chunking | minutes | ~$0.90 | not needed |

## Features

- **Hours-long files.** Converts to 16 kHz mono FLAC, splits into 8-minute chunks with 3 s overlap, and re-splits any chunk over 24 MB.
- **Resumable.** Each chunk is cached the moment it finishes. Ctrl-C, Wi-Fi drop, laptop sleep: run the same command again and it picks up where it stopped.
- **Rate-limit aware.** Retries 429/5xx with exponential backoff, honors `retry-after`, and when you hit Groq's hourly audio quota it **waits and resumes** instead of failing.
- **Clean merge.** Timestamps are offset to the original file and overlap duplicates are removed.
- **Three outputs.** Plain text (optionally `[HH:MM:SS]`-stamped), SRT subtitles for VLC / YouTube / Premiere, and JSON segments for scripts.
- **99 languages.** Auto-detect, or force one with `--lang`. `--prompt` helps with names and jargon.
- **Cost shown up front.** Prints the estimated cost of every run.
- **One file, three dependencies.** Easy to read, audit and hack.

## Quick start

You need [ffmpeg](https://ffmpeg.org) and a free [Groq API key](https://console.groq.com/keys).

**macOS**

```bash
brew install ffmpeg
git clone https://github.com/IhabProjects/transcribe-everything.git
cd transcribe-everything
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your key into .env
python transcribe.py recording.m4a
```

**Linux:** same steps, with `sudo apt install ffmpeg`.
**Windows:** `winget install ffmpeg`, then the same steps with `.venv\Scripts\activate`.

**Install as a global command** (any OS):

```bash
pipx install git+https://github.com/IhabProjects/transcribe-everything.git
export GROQ_API_KEY=gsk_...
transcribe recording.m4a
```

## Usage

```bash
transcribe lecture.m4a                                   # auto-detect language
transcribe interview.mp3 --lang ar --out interviews/ep1  # Arabic, custom output path
transcribe meeting.m4a --timestamps                      # [00:12:31] before each .txt line
transcribe talk.wav --model whisper-large-v3 --prompt "Kubernetes, Ifrane, Al Akhawayn"
```

Any format ffmpeg can read works: `m4a`, `mp3`, `wav`, `mp4`, `mov`, `ogg`, `webm`, ...

| Option | Default | Description |
| --- | --- | --- |
| `--lang CODE` | auto | ISO-639-1 code (`en`, `fr`, `ar`, `es`, ...). Forcing it improves accuracy. |
| `--model` | `whisper-large-v3-turbo` | `whisper-large-v3` is more accurate, about 3x the price. |
| `--out PREFIX` | `transcript` | Writes `PREFIX.txt`, `PREFIX.srt`, `PREFIX.json`. |
| `--prompt TEXT` | none | Vocabulary hint: names, brands, technical terms. |
| `--timestamps` | off | Prefix each `.txt` line with `[HH:MM:SS]`. |
| `--workers N` | `2` | Parallel requests. Raise on paid Groq tiers. |
| `--keep-cache` | off | Keep per-chunk results after a successful run. |

The API key is read from `GROQ_API_KEY`, or from a `.env` file in the current directory.

## Cost

| Model | Price per audio hour | 3 h recording |
| --- | --- | --- |
| `whisper-large-v3-turbo` | ~$0.04 | ~$0.12 |
| `whisper-large-v3` | ~$0.111 | ~$0.33 |

Groq's free tier also includes a daily audio allowance. Check [groq.com/pricing](https://groq.com/pricing) for current numbers.

## How it works

```
input.m4a ─► ffmpeg ─► 16 kHz mono FLAC ─► 8-min chunks (+3 s overlap, ≤24 MB)
                                                   │
                         2 workers ◄───────────────┘
                             │  Groq Whisper (verbose_json, temperature 0)
                             │  retry on 429/5xx, wait out hourly quota
                             ▼
                 ~/.cache/transcribe-everything/  (one JSON per finished chunk)
                             │
                             ▼
     merge in order, offset timestamps, drop segments inside the overlap
                             │
                             ▼
              transcript.txt   transcript.srt   transcript.json
```

## FAQ

**Is my audio uploaded?** Yes, chunks are sent to Groq's API. Read [Groq's privacy policy](https://groq.com/privacy-policy) before transcribing confidential recordings.

**It stopped halfway.** Run the exact same command again. Finished chunks are skipped.

**Words are misspelled.** Pass `--lang`, try `--model whisper-large-v3`, and put names in `--prompt`.

**Does it identify speakers?** Not yet. See the roadmap.

## Roadmap

- [ ] Speaker labels (diarization)
- [ ] Word-level timestamps and `.vtt` output
- [ ] Batch mode for whole folders
- [ ] Optional summary of the transcript with an LLM
- [ ] Publish to PyPI

Ideas and PRs welcome.

## Contributing

```bash
pip install -e . pytest ruff
ruff check . && pytest
```

Tests run offline and don't need an API key.

## License

[MIT](LICENSE) © Ihab El Bani
