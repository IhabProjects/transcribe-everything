import json

import pytest

import transcribe as t


def test_base_plan_covers_audio_with_overlap():
    plan = t.base_plan(1000.0)
    assert plan[0] == (0.0, t.CHUNK_SECONDS + t.OVERLAP_SECONDS)
    assert plan[1][0] == t.CHUNK_SECONDS
    last_start, last_dur = plan[-1]
    assert last_start + last_dur == pytest.approx(1000.0)


def test_base_plan_short_audio_is_one_chunk():
    assert t.base_plan(120.0) == [(0.0, 120.0)]


def test_merge_offsets_and_drops_overlap():
    plan = [(0.0, 483.0), (480.0, 100.0)]
    results = {
        t.chunk_id(0.0, 483.0): {"segments": [
            {"start": 0.0, "end": 5.0, "text": " hello"},
            {"start": 478.0, "end": 482.0, "text": " end of chunk one"},
        ]},
        t.chunk_id(480.0, 100.0): {"segments": [
            {"start": 0.5, "end": 2.0, "text": " duplicate"},  # starts at 480.5 < 482
            {"start": 2.5, "end": 6.0, "text": " new speech"},
            {"start": 7.0, "end": 8.0, "text": "   "},
        ]},
    }
    merged = t.merge(plan, results)
    assert [s["text"] for s in merged] == ["hello", "end of chunk one", "new speech"]
    assert merged[-1]["start"] == pytest.approx(482.5)
    assert [s["id"] for s in merged] == [0, 1, 2]


def test_srt_time():
    assert t.srt_time(0) == "00:00:00,000"
    assert t.srt_time(3725.5) == "01:02:05,500"


@pytest.mark.parametrize("msg,expected", [
    ("Please try again in 7m30.5s.", 450.5),
    ("Please try again in 1h2m3s.", 3723.0),
    ("Please try again in 12.25s.", 12.25),
    ("no hint here", None),
])
def test_parse_wait_seconds_from_message(msg, expected):
    assert t.parse_wait_seconds(Exception(msg)) == expected


def test_parse_wait_seconds_prefers_header():
    class Resp:
        headers = {"retry-after": "42"}

    class Err(Exception):
        response = Resp()

    assert t.parse_wait_seconds(Err("try again in 5s")) == 42.0


def test_write_outputs(tmp_path):
    segs = [{"id": 0, "start": 0.0, "end": 1.5, "text": "Bonjour"},
            {"id": 1, "start": 61.0, "end": 62.0, "text": "Merci"}]
    files = t.write_outputs(tmp_path / "x", segs, {"model": "m"}, timestamps=True)
    txt, srt, js = (f.read_text(encoding="utf-8") for f in files)
    assert txt == "[00:00:00] Bonjour\n[00:01:01] Merci\n"
    assert "00:01:01,000 --> 00:01:02,000\nMerci" in srt
    assert json.loads(js)["text"] == "Bonjour Merci"
