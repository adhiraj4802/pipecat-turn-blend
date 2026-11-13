# pipecat-turn-blend

End-of-turn detection for [Pipecat](https://github.com/pipecat-ai/pipecat) voice agents, as one local ONNX model.

`TurnBlendAnalyzer` is a Pipecat turn analyzer backed by `turn-blend.onnx` (56 MB). The file contains three
open models and the logic that combines them:

- **Silero VAD v6.2** decides whether each 32 ms chunk is speech.
- **Smart Turn v3.2** (Pipecat / Daily) scores the turn so far: does it sound finished?
- **turn-1-mini** (p99lab) gives a second, streaming end-of-turn score.
- A **logistic blend** of the two scores and the elapsed silence decides when the turn is over.

It runs on the CPU, needs no API key and no GPU, and decides from audio alone (no transcript).
**English only**, because turn-1-mini is English only. Input must be 16 kHz.

This is a community integration. It is not affiliated with or endorsed by Daily, Pipecat, Silero, p99lab,
OpenAI or LiveKit. I am an independent developer and do not work for any of them.

## Results

On LiveKit's [eot-bench](https://github.com/livekit/eot-bench) (English, 400 real user turns), scoring this
exact ONNX file:

| Model | False cut-offs @ 300 ms | False cut-offs @ 600 ms | Latency @ 5% cut-offs | Latency @ 10% cut-offs |
|---|---:|---:|---:|---:|
| **turn-blend, cross-validated** | **20.1%** | **7.8%** | **798 ms** | **487 ms** |
| turn-1-mini alone | 22.6% | 9.6% | 849 ms | 580 ms |
| Smart Turn v3.2 alone | 35.2% | 14.8% | 1051 ms | 739 ms |
| VAD baseline (silence only) | 55.6% | 21.7% | 1600 ms | 1000 ms |

Lower is better. The blend is better than either model inside it, and it is behind the best hosted and
large-GPU models (LiveKit Turn Detector v1 scores 9.9% and 543 ms). Read the limits before relying on these
numbers: the blend's four weights were fitted on this same dataset, and this is a local run, not a
leaderboard submission. Full tables, method and limits: [benchmark/README.md](benchmark/README.md).

Compute cost on a laptop CPU (AMD Ryzen 5 5600H, 4 threads): 3 ms for most 32 ms chunks, 31 ms for the
chunks that also run Smart Turn, 4.8 times faster than real time overall. Inside Pipecat the analyzer adds
about 32 ms between the model's decision point and the turn being reported.

## Installation

```bash
uv pip install git+https://github.com/adhiraj4802/pipecat-turn-blend.git
```

The model is not inside the package. It is downloaded from this repository's
[releases](https://github.com/adhiraj4802/pipecat-turn-blend/releases) on first use, checked against its
SHA-256 and cached in `~/.cache/turn-blend`. To use a copy you already have, pass `model_path=`.

## Usage with a Pipecat pipeline

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat_turn_blend import TurnBlendAnalyzer

turn_blend = TurnBlendAnalyzer()            # or TurnBlendAnalyzer("path/to/turn-blend.onnx", blend_threshold=0.9)

aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
    vad_analyzer=SileroVADAnalyzer(),       # starts turns and detects interruptions
    user_turn_strategies=UserTurnStrategies(stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_blend)]),
))

worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=16000))   # the model needs 16 kHz
```

Pipecat's own VAD still opens the user turn and handles interruptions. The analyzer receives every audio
frame, runs the model on a worker thread, and reports the turn complete when the blend commits or the
silence timeout runs out. Pipecat then waits for the final transcript and starts the LLM.

After each turn, `turn_blend.last_result` holds the details (also passed to the optional `on_end_of_turn`
callback):

```python
{"reason": "commit", "silence_ms": 384, "detect_ms": 416, "blend": 0.892, "smart_turn": 0.988, "mini": 0.172}
```

## Settings

Every setting stored in the model can be changed. Defaults:

| Setting | Default | Meaning |
|---|---:|---|
| `vad_confidence` | 0.7 | a chunk is speech when Silero's probability is at least this |
| `vad_start_secs` | 0.2 | continuous speech needed to open a turn |
| `vad_min_db` | -80 | chunks quieter than this (RMS dBFS) never count as speech |
| `pre_speech_ms` | 500 | audio from before the turn opened that Smart Turn also hears |
| `smart_turn_max_secs` | 8 | how much of the turn Smart Turn hears (at most 8) |
| `score_every_chunks` | 2 | score every N chunks while a turn is open |
| `blend_threshold` | 0.85 | commit: blend above this ... |
| `blend_min_silence_ms` | 320 | ... after at least this much silence ... |
| `blend_confirm_scores` | 2 | ... on this many scores in a row |
| `tentative_threshold` | 0.5 | tentative end of turn: blend above this ... |
| `tentative_min_silence_ms` | 160 | ... after at least this much silence |
| `timeout_secs` | 3 | silence that ends the turn even if the blend never commits |
| `w_bias`, `w_smart_turn`, `w_mini`, `w_log_silence` | 2.8007, 0.2939, 0.4873, 1.288 | the blend's weights |

Three ways to set them:

```python
TurnBlendAnalyzer(blend_threshold=0.9, timeout_secs=2.0)        # keyword arguments
turn_blend.detector.configure(blend_min_silence_ms=256)         # later; takes effect on the next chunk
TurnBlendAnalyzer(**load_settings("settings.json"))             # a JSON file and TURN_BLEND_<SETTING> env vars
```

`python examples/configure_demo.py --list` prints the table from the model, and
`python examples/configure_demo.py recording.wav --blend-threshold 0.95 --compare` shows what a change does
to a recording. Values are not range-checked.

## Running the example

[examples/turn_blend_bot.py](examples/turn_blend_bot.py) is a complete voice bot: Deepgram STT, turn-blend,
OpenAI LLM, Deepgram TTS.

```bash
uv pip install "pipecat-turn-blend[example] @ git+https://github.com/adhiraj4802/pipecat-turn-blend.git"
cp examples/.env.example .env                    # fill in DEEPGRAM_API_KEY and OPENAI_API_KEY
python examples/turn_blend_bot.py -t webrtc      # open http://localhost:7860 and press Connect
```

Each turn logs how long turn-blend took to decide and the full turn latency (you stop speaking to the bot
starts speaking).

## Without Pipecat's pipeline

The model can be driven directly, one 512-sample chunk (32 ms at 16 kHz, float32 in [-1, 1]) at a time:

```python
from pipecat_turn_blend import TurnBlend, model_path

detector = TurnBlend(model_path())
for chunk in chunks_of_512_samples:
    result = detector(chunk)
    if result["end_of_turn"]:      # 1 = the blend committed, 2 = silence timeout
        ...
```

## Compatibility

Tested with Pipecat v1.16.0, Python 3.12, onnxruntime 1.32, on Linux x86-64.

## Tests

```bash
uv pip install -e ".[dev]"
TB_TEST_WAV=some_16khz_speech.wav pytest        # without TB_TEST_WAV the speech tests are skipped
```

## Licence

The code in this repository, the blend weights and the turn logic are under the Apache License 2.0.
`turn-blend.onnx` is a combined work that also contains the unmodified weights of Silero VAD (MIT),
Smart Turn v3.2 (BSD 2-Clause, built on OpenAI's Whisper Tiny encoder, MIT) and turn-1-mini (Apache 2.0).
All of these licences apply to the model file; see [LICENSE](LICENSE), [NOTICE](NOTICE) and
[licenses/](licenses/). If you redistribute the model, keep those files with it.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
