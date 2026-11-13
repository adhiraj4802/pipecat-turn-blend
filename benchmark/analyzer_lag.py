"""
Measure the delay TurnBlendAnalyzer adds inside Pipecat, with audio arriving in real time.

    python benchmark/analyzer_lag.py recording.wav --json analyzer_lag.json

Audio is fed as 20 ms frames at real-time pace, the way a transport delivers it. For every ended turn the
analyzer reports `silence_ms` (silence the model waited for) and `detect_ms` (last speech to decision,
including queueing and compute). Their difference is the delay added by running the model.
"""

import argparse
import asyncio
import json
import time
import wave

import numpy as np

from pipecat_turn_blend import TurnBlendAnalyzer


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json")
    args = ap.parse_args()
    with wave.open(args.wav) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, "need a 16 kHz, 16-bit WAV file"
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())
    pcm = pcm.mean(axis=1).astype(np.int16)

    turns = []
    analyzer = TurnBlendAnalyzer(threads=args.threads, on_end_of_turn=turns.append)
    analyzer.set_sample_rate(16000)
    start = time.perf_counter()
    for k, i in enumerate(range(0, len(pcm) - 320, 320)):
        await asyncio.sleep(max(0.0, start + k * 0.02 - time.perf_counter()))
        analyzer.append_audio(pcm[i:i + 320].tobytes(), True)
        await analyzer.analyze_end_of_turn()
    await asyncio.sleep(0.5)
    await analyzer.cleanup()

    lag = np.array([t["detect_ms"] - t["silence_ms"] for t in turns])
    report = {"threads": args.threads, "audio_seconds": round(len(pcm) / 16000, 1), "turns": len(turns),
              "committed_by_blend": sum(t["reason"] == "commit" for t in turns),
              "silence_ms_mean": round(float(np.mean([t["silence_ms"] for t in turns]))),
              "detect_ms_mean": round(float(np.mean([t["detect_ms"] for t in turns]))),
              "added_delay_ms": {"mean": round(float(lag.mean()), 1), "p50": float(np.percentile(lag, 50)),
                                 "max": float(lag.max())}}
    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
