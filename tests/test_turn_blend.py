"""Tests for the runner and the Pipecat analyzer.

The model is downloaded on first use. The speech tests need a recording of conversational English:
set TB_TEST_WAV to a 16 kHz, 16-bit WAV file, otherwise they are skipped.
"""

import asyncio
import json
import os
import wave

import numpy as np
import pytest
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

from pipecat_turn_blend import TurnBlend, TurnBlendAnalyzer, load_settings, model_path

WAV = os.environ.get("TB_TEST_WAV")


@pytest.fixture(scope="module")
def detector():
    return TurnBlend(model_path())


@pytest.fixture(scope="module")
def speech():
    if not WAV:
        pytest.skip("set TB_TEST_WAV to a 16 kHz, 16-bit WAV of speech")
    with wave.open(WAV) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())
    return pcm.mean(axis=1).astype(np.int16)


def test_every_setting_is_configurable(detector):
    d = detector.new_stream()
    assert len(d.names) == 16
    d.configure(**{n: d.defaults[n] + 1 for n in d.names})
    assert all(d.config[n] == d.defaults[n] + 1 for n in d.names)
    assert d.describe().count("*") == 16
    d.reset_config()
    assert d.config == {n: float(v) for n, v in d.defaults.items()}
    assert detector.config == detector.defaults          # the stream it came from is untouched


def test_bad_settings_are_rejected(detector):
    d = detector.new_stream()
    with pytest.raises(KeyError):
        d.configure(no_such_setting=1)
    with pytest.raises(ValueError):
        d.configure(blend_threshold="high")
    with pytest.raises(ValueError):
        d.configure(timeout_secs=float("nan"))
    with pytest.raises(ValueError):
        d(np.zeros(100, np.float32))


def test_load_settings_file_then_environment(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"blend_threshold": 0.9, "timeout_secs": 2}))
    settings = load_settings(path, env={"TURN_BLEND_TIMEOUT_SECS": "1.5", "TURN_BLEND_VAD_CONFIDENCE": "0.5", "OTHER": "x"})
    assert settings == {"blend_threshold": 0.9, "timeout_secs": "1.5", "vad_confidence": "0.5"}


def test_silence_never_opens_a_turn(detector):
    d = detector.new_stream()
    results = [d(np.zeros(d.chunk_samples, np.float32)) for _ in range(150)]
    assert not any(r["in_turn"] or r["end_of_turn"] for r in results)


def test_runner_finds_turns_and_settings_change_them(detector, speech):
    def turns(**config):
        d = detector.new_stream()
        d.configure(**config)
        audio, n = speech.astype(np.float32) / 32768.0, d.chunk_samples
        return [r["end_of_turn"] for r in (d(audio[i * n:(i + 1) * n]) for i in range(len(audio) // n)) if r["end_of_turn"]]

    default = turns()
    assert len(default) >= 2 and 1.0 in default
    strict = turns(blend_threshold=2.0, timeout_secs=1.0)      # the blend can never commit
    assert strict and set(strict) == {2.0}


async def test_analyzer_reports_complete_turns(speech):
    ended = []
    analyzer = TurnBlendAnalyzer(on_end_of_turn=ended.append)
    analyzer.set_sample_rate(16000)
    complete = 0
    for i in range(0, len(speech) - 320, 320):                  # 20 ms frames, as a transport delivers them
        analyzer.append_audio(speech[i:i + 320].tobytes(), True)
        if i % 3200 == 0:
            await asyncio.sleep(0.02)                           # let the worker thread keep up
            state, metrics = await analyzer.analyze_end_of_turn()
            if state == EndOfTurnState.COMPLETE:
                complete += 1
                assert metrics.is_complete and 0 <= metrics.probability <= 1
    await analyzer.cleanup()
    analyzer._thread.join(120)                                  # the test feeds audio faster than real time
    assert complete >= 2 and len(ended) >= complete
    assert {"reason", "silence_ms", "detect_ms", "blend"} <= set(analyzer.last_result)
    assert analyzer.params.stop_secs == 3.0
