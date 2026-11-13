"""
Measure how long turn-blend takes to process each 32 ms chunk on this machine.

    python benchmark/latency.py recording.wav                 # 16 kHz, 16-bit WAV of conversational speech
    python benchmark/latency.py recording.wav --threads 1 2 4 6 --json latency.json

A chunk is 32 ms of audio, so the model keeps up in real time while the time per chunk stays under 32 ms.
Most chunks run only Silero VAD and turn-1-mini. While a turn is open, every `score_every_chunks`-th chunk
also runs Smart Turn, which is the expensive part; those chunks are reported separately.
"""

import argparse
import json
import platform
import time
import wave

import numpy as np
import onnxruntime as ort

from pipecat_turn_blend import TurnBlend, model_path


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


def stats(ms):
    ms = np.asarray(ms)
    if not len(ms):
        return None
    return {"chunks": int(len(ms)), "mean_ms": round(float(ms.mean()), 2), "p50_ms": round(float(np.percentile(ms, 50)), 2),
            "p95_ms": round(float(np.percentile(ms, 95)), 2), "p99_ms": round(float(np.percentile(ms, 99)), 2),
            "max_ms": round(float(ms.max()), 2)}


def measure(audio, threads, **config):
    detector = TurnBlend(model_path(), threads=threads, **config)
    n = detector.chunk_samples
    chunks = [audio[i * n:(i + 1) * n] for i in range(len(audio) // n)]
    for c in chunks[:30]:                    # warm up, then start the stream again
        detector(c)
    detector.reset()
    scored, light, turns = [], [], 0
    for c in chunks:
        t = time.perf_counter()
        r = detector(c)
        ms = (time.perf_counter() - t) * 1000
        (scored if r["smart_turn"] >= 0 else light).append(ms)
        turns += bool(r["end_of_turn"])
    total = sum(scored) + sum(light)
    return {"threads": threads, "all_chunks": stats(scored + light), "chunks_with_smart_turn": stats(scored),
            "other_chunks": stats(light), "turns_ended": turns,
            "cpu_time_per_audio_second_ms": round(total / (len(chunks) * n / detector.sample_rate), 1),
            "x_real_time": round(len(chunks) * n / detector.sample_rate / (total / 1000), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--score-every-chunks", type=int, default=None, help="override the model default (2)")
    ap.add_argument("--json", help="write the results to this file")
    args = ap.parse_args()

    with wave.open(args.wav) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, "need a 16 kHz, 16-bit WAV file"
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())
    audio = audio.mean(axis=1).astype(np.float32) / 32768.0
    config = {} if args.score_every_chunks is None else {"score_every_chunks": args.score_every_chunks}

    report = {"cpu": cpu_name(), "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider",
              "audio_seconds": round(len(audio) / 16000, 1), "chunk_ms": 32, "config": config, "runs": []}
    print(f"{report['cpu']}, onnxruntime {ort.__version__}, {report['audio_seconds']} s of audio\n")
    print("threads   all chunks: mean / p50 / p99 / max      with Smart Turn: p50 / p99      others: p50     x real time")
    for threads in args.threads:
        r = measure(audio, threads, **config)
        report["runs"].append(r)
        a, s, o = r["all_chunks"], r["chunks_with_smart_turn"], r["other_chunks"]
        print(f"{threads:>7}   {a['mean_ms']:>6.2f} / {a['p50_ms']:>5.2f} / {a['p99_ms']:>5.2f} / {a['max_ms']:>6.2f} ms"
              f"        {s['p50_ms']:>6.2f} / {s['p99_ms']:>6.2f} ms       {o['p50_ms']:>5.2f} ms     {r['x_real_time']:>6.1f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
