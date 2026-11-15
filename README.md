# pipecat-turn-blend

End-of-turn detection for [Pipecat](https://github.com/pipecat-ai/pipecat) voice agents, as one local ONNX model.

`TurnBlendAnalyzer` is a Pipecat turn analyzer backed by `turn-blend.onnx` (105 MB). The file contains three
models and the logic that combines them:

- **Silero VAD v6.2** decides whether each 32 ms chunk is speech.
- A **student model** scores the turn so far: does it sound finished? It is a Whisper Base encoder trained
  to reproduce the scores of p99lab's turn-1, a much larger GPU model.
- **turn-1-mini** (p99lab) gives a second, streaming end-of-turn score.
- A **logistic blend** of the two scores and the elapsed silence decides when the turn is over.

It runs on the CPU, needs no API key and no GPU, and decides from audio alone (no transcript).
**English only**, because turn-1-mini is English only. Input must be 16 kHz.

This is a community integration. It is not affiliated with or endorsed by Daily, Pipecat, Silero, p99lab,
OpenAI or LiveKit. I am an independent developer and do not work for any of them.

## Results

On LiveKit's [eot-bench](https://github.com/livekit/eot-bench) v1.1, English, all 2,288 turns (400 general,
800 accented, 1,088 short turns), scoring this exact ONNX file. Figures are false cut-offs at a 300 ms
latency budget, then mean latency at a 5% cut-off budget; lower is better.

| Model | All English | General | Accented | Short turns |
|---|---:|---:|---:|---:|
| LiveKit Turn Detector v1.1 | 13.5% / 824 ms | 11.2% / 708 ms | 13.3% / 701 ms | 16.0% / 1082 ms |
| p99lab turn-1 (GPU) | 19.3% / 912 ms | 10.6% / 553 ms | 19.1% / 878 ms | 28.3% / 1181 ms |
| **turn-blend 0.2.0 (CPU)** | **23.5% / 995 ms** | **13.0% / 512 ms** | **23.7% / 934 ms** | **36.0% / 1428 ms** |
| LiveKit Turn Detector v1.1-mini | 26.0% / 1183 ms | 18.4% / 1030 ms | 26.4% / 1025 ms | 32.5% / 1235 ms |
| turn-blend 0.1.0 | 32.5% / 1114 ms | 19.1% / 722 ms | 34.8% / 1012 ms | 44.1% / 1808 ms |
| p99lab turn-1-mini | 39.6% / 1169 ms | 22.6% / 849 ms | 38.2% / 1027 ms | 60.7% / 1600 ms |

- turn-blend 0.2.0 makes fewer false cut-offs than LiveKit's v1.1-mini overall and on general and accented
  speech. It is **behind v1.1-mini on short turns** ("yes", "no thanks"), and behind LiveKit's full v1.1
  model and turn-1 overall.
- Rows other than turn-blend are computed from the predictions committed in the eot-bench repository; those
  models were not re-run here. This is a local run, not a leaderboard submission.
- The blend weights of 0.2.0 were fitted on separate held-out audio, not on this benchmark. The student
  was trained once, so differences of about a point are within noise.

Run exactly as it ships, with its default commit rule, 0.2.0 ends the turn inside 9.3% of mid-turn pauses
(0.1.0: 15.1%) and answers a real end of turn after a median of 376 ms (0.1.0: 400 ms).
Full tables, every published model, method and limits: [benchmark/README.md](benchmark/README.md).

Compute cost on a laptop CPU (AMD Ryzen 5 5600H, 4 threads): 3 ms for most 32 ms chunks and 68 ms for the
chunks of a pause that also run the student model (0.1.0: 31 ms), 4.4 times faster than real time overall.
Inside Pipecat the analyzer adds about 70 ms between the model's decision point and the turn being
reported, and runs the student less often on machines that cannot keep up.

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

turn_blend = TurnBlendAnalyzer()            # or TurnBlendAnalyzer("path/to/turn-blend.onnx", blend_threshold=0.8)

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
callback). `smart_turn` is the student model's score; the name is kept from version 0.1.0, where that slot held
Smart Turn, so existing code keeps working:

```python
{"reason": "commit", "silence_ms": 384, "detect_ms": 416, "blend": 0.805, "smart_turn": 0.831, "mini": 0.242}
```

## Settings

Every setting stored in the model can be changed. Defaults:

| Setting | Default | Meaning |
|---|---:|---|
| `vad_confidence` | 0.7 | a chunk is speech when Silero's probability is at least this |
| `vad_start_secs` | 0.2 | continuous speech needed to open a turn |
| `vad_min_db` | -80 | chunks quieter than this (RMS dBFS) never count as speech |
| `pre_speech_ms` | 500 | audio from before the turn opened that the student model also hears |
| `smart_turn_max_secs` | 8 | how much of the turn the student model hears (at most 8) |
| `score_every_chunks` | 2 | score every N chunks while a turn is open |
| `blend_threshold` | 0.75 | commit: blend above this ... |
| `blend_min_silence_ms` | 160 | ... after at least this much silence ... |
| `blend_confirm_scores` | 2 | ... on this many scores in a row |
| `tentative_threshold` | 0.4 | tentative end of turn: blend above this ... |
| `tentative_min_silence_ms` | 160 | ... after at least this much silence |
| `timeout_secs` | 3 | silence that ends the turn even if the blend never commits |
| `w_bias`, `w_smart_turn`, `w_mini`, `w_log_silence` | 1.479, 0.277, 0.172, 0.32 | the blend's weights (`w_smart_turn` weighs the student model) |

`blend_threshold` is the setting to tune first, and it is sensitive. On eot-bench, 0.72 answers about 70 ms
sooner and interrupts more (12.5% of mid-turn pauses instead of 9.3%); 0.81 interrupts far less (4.6%) but
roughly doubles the delay. Settings names containing `smart_turn` are kept from 0.1.0 and now apply to the
student model.

Three ways to set them:

```python
TurnBlendAnalyzer(blend_threshold=0.8, timeout_secs=2.0)        # keyword arguments
turn_blend.detector.configure(blend_min_silence_ms=256)         # later; takes effect on the next chunk
TurnBlendAnalyzer(**load_settings("settings.json"))             # a JSON file and TURN_BLEND_<SETTING> env vars
```

`python examples/configure_demo.py --list` prints the table from the model, and
`python examples/configure_demo.py recording.wav --blend-threshold 0.8 --compare` shows what a change does
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

## Upgrading from 0.1.0

The API, the model's inputs and outputs and all 16 setting names are unchanged. What changes:

- The model file is 105 MB (was 56 MB) and is downloaded again on first use.
- Scoring a pause costs about twice as much CPU (68 ms against 31 ms on 4 laptop threads). Use 2 or more
  threads. `TurnBlendAnalyzer` now runs the student less often by itself when the machine cannot keep up.
- The blend is on a different scale, so a `blend_threshold` you tuned for 0.1.0 does not carry over. The
  defaults for `blend_threshold`, `blend_min_silence_ms` and `tentative_threshold` changed.
- To stay on the old model: `uv pip install git+https://github.com/adhiraj4802/pipecat-turn-blend.git@v0.1.0`.

## Compatibility

Tested with Pipecat v1.16.0, Python 3.12, onnxruntime 1.32, on Linux x86-64.

## Tests

```bash
uv pip install -e ".[dev]"
TB_TEST_WAV=some_16khz_speech.wav pytest        # without TB_TEST_WAV the speech tests are skipped
```

## Licence

The code in this repository, the blend weights and the turn logic are under the Apache License 2.0.
`turn-blend.onnx` is a combined work that also contains the unmodified weights of Silero VAD (MIT) and
turn-1-mini (Apache 2.0). Its student model was initialised from OpenAI's Whisper Base encoder (MIT) and
trained on scores from p99lab's turn-1 (Apache 2.0), using HarperValleyBank, the AMI Meeting Corpus and
Pipecat's Smart Turn training data (all CC BY 4.0). All of these licences apply to the model file; see [LICENSE](LICENSE), [NOTICE](NOTICE) and
[licenses/](licenses/). If you redistribute the model, keep those files with it.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
