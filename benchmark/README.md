# Benchmarks

Three things are measured here: how well turn-blend decides that a turn is over (LiveKit's eot-bench), what
it does at its own default settings on the same turns, and how much compute and delay it costs.

Everything on this page is for **turn-blend 0.2.0** unless a row says otherwise. The results published with
0.1.0 (an older, 400-turn revision of the benchmark) are kept in [eot-bench/](eot-bench/) for reference.

## 1. End-of-turn accuracy: eot-bench v1.1

Benchmark: [LiveKit eot-bench](https://github.com/livekit/eot-bench), English, all 2,288 turns of
`livekit/eot-bench-data` at revision `ae834af` (400 general, 800 accented, 1,088 short turns), harness commit
`45ff653`. Run on 14 November 2026.

The harness asks the model, every 0.1 s inside each labelled pause, for its probability that the turn is
over. It then sweeps threshold, minimum silence and timeout itself and reports the best trade-offs: the false
cut-off rate reachable within a latency budget, and the mean latency reachable within a cut-off budget.
2,288 turns, 4,049 pauses (2,305 mid-turn, 1,744 ends of turn), 71,654 decision points. Lower is better
everywhere. "n/a" means the model cannot reach that latency budget at all.

#### All English (2,288 turns)

| Model | False cut-offs @ 300 ms | False cut-offs @ 600 ms | Latency @ 5% cut-offs | Latency @ 10% cut-offs |
|---|---:|---:|---:|---:|
| LiveKit Turn Detector v1.1 | 13.4% | 7.5% | 810 ms | 414 ms |
| p99lab turn-1 (GPU) | 19.3% | 8.7% | 912 ms | 529 ms |
| Deepgram Flux | 22.3% | 13.5% | 1409 ms | 826 ms |
| **turn-blend 0.2.0** | **23.5%** | **9.6%** | **995 ms** | **599 ms** |
| LiveKit Turn Detector v1.1-mini | 25.6% | 12.9% | 1124 ms | 738 ms |
| JoinIn AI Baton | 26.1% | 13.6% | 1111 ms | 749 ms |
| turn-blend 0.1.0 | 32.5% | 14.4% | 1114 ms | 761 ms |
| p99lab turn-1-mini | 39.6% | 18.7% | 1169 ms | 862 ms |
| ultraVAD | 39.7% | 20.8% | 1321 ms | 891 ms |
| Smart Turn v3.2 | 46.8% | 22.9% | 1384 ms | 956 ms |
| VAP (silent agent) | 63.5% | 29.0% | 1724 ms | 1048 ms |
| Gradium | 79.3% | 23.2% | 1112 ms | 785 ms |
| Soniox | n/a | 13.3% | 2000 ms | 789 ms |
| AssemblyAI | n/a | 21.7% | 1381 ms | 906 ms |
| Cartesia Ink 2 | n/a | n/a | 1027 ms | 968 ms |
| OpenAI GPT Realtime 2 | n/a | n/a | 2000 ms | 959 ms |

#### General (400 turns)

| Model | False cut-offs @ 300 ms | False cut-offs @ 600 ms | Latency @ 5% cut-offs | Latency @ 10% cut-offs |
|---|---:|---:|---:|---:|
| LiveKit Turn Detector v1.1 | 7.8% | 4.1% | 462 ms | 250 ms |
| p99lab turn-1 (GPU) | 10.6% | 4.5% | 553 ms | 321 ms |
| JoinIn AI Baton | 12.3% | 4.8% | 577 ms | 350 ms |
| Deepgram Flux | 12.9% | 9.9% | 1151 ms | 548 ms |
| **turn-blend 0.2.0** | **13.0%** | **4.1%** | **512 ms** | **384 ms** |
| LiveKit Turn Detector v1.1-mini | 17.7% | 8.2% | 812 ms | 504 ms |
| turn-blend 0.1.0 | 19.1% | 7.1% | 722 ms | 495 ms |
| p99lab turn-1-mini | 22.6% | 9.6% | 849 ms | 580 ms |
| ultraVAD | 27.7% | 11.9% | 899 ms | 663 ms |
| Smart Turn v3.2 | 35.2% | 14.8% | 1051 ms | 739 ms |
| VAP (silent agent) | 47.0% | 14.6% | 1131 ms | 749 ms |
| AssemblyAI | 49.4% | 14.6% | 1049 ms | 713 ms |
| Gradium | 55.6% | 12.6% | 913 ms | 656 ms |
| Soniox | n/a | 5.5% | 647 ms | 512 ms |
| Cartesia Ink 2 | n/a | n/a | 1056 ms | 911 ms |
| OpenAI GPT Realtime 2 | n/a | n/a | 1143 ms | 824 ms |

#### Accented (800 turns)

| Model | False cut-offs @ 300 ms | False cut-offs @ 600 ms | Latency @ 5% cut-offs | Latency @ 10% cut-offs |
|---|---:|---:|---:|---:|
| LiveKit Turn Detector v1.1 | 12.4% | 6.4% | 700 ms | 360 ms |
| p99lab turn-1 (GPU) | 19.1% | 8.5% | 878 ms | 520 ms |
| Deepgram Flux | 22.6% | 12.4% | 1084 ms | 734 ms |
| **turn-blend 0.2.0** | **23.7%** | **9.8%** | **934 ms** | **588 ms** |
| LiveKit Turn Detector v1.1-mini | 24.3% | 11.2% | 999 ms | 676 ms |
| JoinIn AI Baton | 24.9% | 12.2% | 958 ms | 682 ms |
| turn-blend 0.1.0 | 34.8% | 14.1% | 1012 ms | 737 ms |
| p99lab turn-1-mini | 38.2% | 17.6% | 1027 ms | 817 ms |
| ultraVAD | 40.4% | 21.5% | 1165 ms | 904 ms |
| Smart Turn v3.2 | 51.3% | 24.3% | 1224 ms | 945 ms |
| VAP (silent agent) | 68.5% | 33.0% | 1396 ms | 1011 ms |
| Gradium | 89.5% | 23.4% | 941 ms | 737 ms |
| Soniox | n/a | 12.5% | 1154 ms | 762 ms |
| AssemblyAI | n/a | 22.8% | 1232 ms | 898 ms |
| Cartesia Ink 2 | n/a | n/a | 980 ms | 964 ms |
| OpenAI GPT Realtime 2 | n/a | n/a | 1500 ms | 891 ms |

#### Short turns (1,088 turns)

| Model | False cut-offs @ 300 ms | False cut-offs @ 600 ms | Latency @ 5% cut-offs | Latency @ 10% cut-offs |
|---|---:|---:|---:|---:|
| LiveKit Turn Detector v1.1 | 19.5% | 10.8% | 1096 ms | 635 ms |
| p99lab turn-1 (GPU) | 28.3% | 13.2% | 1181 ms | 759 ms |
| Deepgram Flux | 33.3% | 20.0% | 1995 ms | 1167 ms |
| LiveKit Turn Detector v1.1-mini | 35.1% | 17.6% | 1237 ms | 893 ms |
| **turn-blend 0.2.0** | **36.0%** | **16.7%** | **1428 ms** | **919 ms** |
| turn-blend 0.1.0 | 44.1% | 21.5% | 1808 ms | 1022 ms |
| JoinIn AI Baton | 44.5% | 24.8% | 1689 ms | 1112 ms |
| ultraVAD | 52.8% | 28.7% | 1876 ms | 1152 ms |
| Smart Turn v3.2 | 54.8% | 27.4% | 1761 ms | 1195 ms |
| p99lab turn-1-mini | 60.7% | 26.5% | 1600 ms | 1098 ms |
| VAP (silent agent) | 78.7% | 39.3% | 2256 ms | 1403 ms |
| Gradium | 90.6% | 35.7% | 1525 ms | 1017 ms |
| Soniox | n/a | 23.5% | 2500 ms | 1090 ms |
| AssemblyAI | n/a | 26.8% | 1918 ms | 1064 ms |
| Cartesia Ink 2 | n/a | n/a | 2500 ms | 979 ms |
| OpenAI GPT Realtime 2 | n/a | n/a | 2500 ms | 1500 ms |

### How to read this

- **Against LiveKit's v1.1-mini**, turn-blend 0.2.0 makes fewer false cut-offs overall (23.5% against 25.6% at
  300 ms) and on general speech (13.0% against 17.7%), and it is faster at the 5% cut-off budget overall
  (995 ms against 1124 ms) and on general speech (512 ms against 812 ms).
- **On accented speech the two are level** (23.7% against 24.3%; a difference this small is within noise).
- **On short turns it is level on cut-offs and slower** (36.0% against 35.1%, and 1428 ms against 1237 ms at
  the 5% budget). Short answers such as "yes" or "no thanks" are its weakest case.
- **It is behind** LiveKit's full v1.1 model, p99lab's turn-1 (which needs a GPU) and, at the 300 ms budget,
  Deepgram Flux.
- **Against 0.1.0** every figure in every cohort improves.

### How these numbers were produced

- **This exact file, streamed.** `turn-blend.onnx` 0.2.0 was fed each turn in 32 ms chunks, as in a live
  call, and its blend was read at each decision point ([eot-bench-v1.1/score.py](eot-bench-v1.1/score.py)).
  A decision point where the model had no open turn counts as probability 0.
- **Other rows were not re-run.** They are computed with the same harness code from the predictions committed
  in the eot-bench repository ([eot-bench-v1.1/evaluate.py](eot-bench-v1.1/evaluate.py)) and match the tables
  in that repository's own report. `results.json` keeps the repository's adapter names.
- **Two scoring modes, as in the eot-bench report.** Models that declare a score point (LiveKit's two
  detectors, Baton, Smart Turn and ultraVAD score once, 0.2 s into the silence) are evaluated at that point.
  The others, turn-blend included, are evaluated on every prediction they make during the silence. Scored
  on every prediction instead, LiveKit's v1.1-mini would get 26.0% overall and 32.5% on short turns, so the
  comparison with it does not hinge on the mode overall, but the short-turn order does.
- **Local run.** Nothing was submitted to the public leaderboard.

### Limits

- **Not trained or fitted on the benchmark.** The student model was trained on HarperValleyBank, AMI and
  Smart Turn training audio, and the four blend weights were fitted on held-out windows of HarperValleyBank
  and AMI. For 0.1.0 the blend weights were fitted on the benchmark's 400 general turns, so its general row
  is scored on its own fitting data.
- **Model choices were made with this benchmark in view.** Several student sizes and training mixes were
  compared on it before this one was picked, and the default commit threshold was chosen on it. That is a
  milder form of fitting, but it is not none.
- **One training run.** The student was trained once. Differences of about a point between models are within
  run-to-run noise; the gaps to v1.1-mini overall and on general speech are larger than that, the
  differences on accented speech and on short-turn cut-offs are not.
- **19 short turns are never opened.** In 19 of the 1,088 short turns the model's VAD never opens a turn, so
  it cannot end them. They count against it.
- **The harness picks the policy.** As for every row, threshold, minimum silence and timeout are chosen by
  the harness on the same turns it scores. Section 2 shows what the shipped rule does.
- **English at 16 kHz only.**

## 2. Behaviour at the shipped settings

The model was run over the same 2,288 turns with nothing switched off: it opens and ends turns itself with
its own commit rule ([eot-bench-v1.1/real_run.py](eot-bench-v1.1/real_run.py),
[real_eval.py](eot-bench-v1.1/real_eval.py)). A mid-turn pause counts as interrupted when the model ends the
turn inside it. The delay at a real end of turn runs from the start of the final silence to the model ending
the turn.

| Model and commit threshold | Mid-turn pauses interrupted | Median delay at real ends | Mean delay | Ends reached only by the 3 s timeout |
|---|---:|---:|---:|---:|
| turn-blend 0.1.0, default (0.85) | 15.1% | 400 ms | 578 ms | 1.4% |
| **turn-blend 0.2.0, default (0.75)** | **9.3%** | **376 ms** | **594 ms** | **3.6%** |
| turn-blend 0.2.0, threshold 0.72 | 12.5% | 304 ms | 492 ms | 2.2% |
| turn-blend 0.2.0, threshold 0.81 | 4.6% | 720 ms | 965 ms | 11.3% |

By cohort, at the defaults:

| Cohort | Interrupted pauses, 0.1.0 | Interrupted pauses, 0.2.0 | Median delay, 0.1.0 | Median delay, 0.2.0 |
|---|---:|---:|---:|---:|
| General | 8.5% | 5.9% | 416 ms | 314 ms |
| Accented | 16.0% | 10.1% | 400 ms | 432 ms |
| Short turns | 23.9% | 13.4% | 368 ms | 368 ms |

- At about the same speed as 0.1.0, the 0.2.0 default interrupts roughly 40% fewer mid-turn pauses.
- `blend_threshold` is sensitive: between 0.72 and 0.81 the median delay goes from 304 ms to 720 ms.
- These runs were not paced in real time and scored the student model on every second chunk of a pause. On a
  machine where the student cannot run that often, the Pipecat analyzer scores less often (section 3), which
  can delay a decision by a few tens of milliseconds.
- The default threshold of 0.75 was chosen on these turns, after a threshold tuned on held-out audio turned
  out too cautious here.

## 3. Latency

Measured on an AMD Ryzen 5 5600H laptop (6 cores, 12 threads), CPU only, onnxruntime 1.32.0, on 119 s of
eot-bench speech (8 turns with 2 s of silence between them), with the default settings. The laptop was not
idle during the measurement, so treat the figures as slightly pessimistic.

### Compute per 32 ms chunk

`python benchmark/latency.py recording.wav` ([latency.json](latency.json))

| Threads | Most chunks (p50) | Chunks that run the student (p50 / p99) | All chunks (mean / p99) | Speed |
|---:|---:|---:|---:|---:|
| 1 | 3.1 ms | 185.2 / 188.2 ms | 16.3 / 186.3 ms | 2.0x real time |
| 2 | 2.6 ms | 103.4 / 107.7 ms | 10.0 / 104.6 ms | 3.2x real time |
| 4 | 2.5 ms | 68.2 / 71.7 ms | 7.3 / 69.3 ms | 4.4x real time |
| 6 | 2.7 ms | 60.4 / 63.8 ms | 6.9 / 61.4 ms | 4.6x real time |

Most chunks run only Silero VAD and turn-1-mini. While a turn is open, the student model also runs on up to
8 s of audio during pauses, which is the expensive part: about 68 ms on 4 threads, against 31 ms for Smart
Turn in 0.1.0. At the default of every second chunk its budget is 64 ms, which this laptop only just meets
with 6 threads.

### Keeping up in real time

`TurnBlendAnalyzer` measures how long the student takes and, when one run does not fit in
`score_every_chunks` chunks of audio, runs it less often (with 20% headroom) so that it never falls behind.
On this laptop that means every third or fourth chunk with 4 threads and every fourth with 2 threads.
`analyzer.score_every_chunks` holds the interval in use. If you drive `TurnBlend` directly, set
`score_every_chunks` yourself.

### Delay added inside Pipecat

`python benchmark/analyzer_lag.py recording.wav` ([analyzer_lag.json](analyzer_lag.json))

Audio was fed to `TurnBlendAnalyzer` as 20 ms frames at real-time pace. Over 9 turns (8 ended by the blend,
1 by the timeout):

| | 4 threads | 2 threads |
|---|---:|---:|
| Silence the model waited for before ending the turn (mean) | 700 ms | 761 ms |
| Last speech to the turn being reported, `detect_ms` (mean) | 773 ms | 858 ms |
| Delay added by queueing and compute (mean / max) | 72 / 86 ms | 97 / 105 ms |

The means include the one turn that ended by the 3 s timeout; the eight committed turns waited 288 to 832 ms
with 4 threads. These figures are on 9 turns and are not comparable with section 2.

### Not measured

A GPU or TensorRT run of this file, other CPUs, an idle machine, and the full voice-to-voice latency of a
bot (that depends on the STT, LLM and TTS services; the example bot logs it per turn).

## 4. Files

| Path | What it is |
|---|---|
| `eot-bench-v1.1/results.json` | Headline metrics of every model, per cohort |
| `eot-bench-v1.1/data/scores-0.2.0.tar.gz`, `scores-0.1.0.tar.gz` | turn-blend's output at every decision point of every turn |
| `eot-bench-v1.1/data/events-*.tar.gz` | Every end of turn each model produced at its own settings (section 2) |
| `eot-bench-v1.1/score.py`, `evaluate.py` | Produce the scores and the tables of section 1 |
| `eot-bench-v1.1/real_run.py`, `real_eval.py` | Produce the events and the tables of section 2 |
| `latency.py`, `latency.json`, `analyzer_lag.py`, `analyzer_lag.json` | Section 3 |
| `eot-bench/` | Results published with 0.1.0 on the earlier 400-turn revision of the benchmark |

The score and event files are derived from `livekit/eot-bench-data` (CC BY 4.0). They hold turn ids, labels
and model outputs, not audio.

To reproduce section 1: clone eot-bench, download the English part of `livekit/eot-bench-data`, then

```bash
python benchmark/eot-bench-v1.1/score.py DATA_DIR GRID turn-blend.onnx scores
python benchmark/eot-bench-v1.1/evaluate.py EOT_BENCH_REPO "turn-blend 0.2.0"=scores
```

where `GRID` is any `predictions.parquet` under the eot-bench repository's `output/` folder.
