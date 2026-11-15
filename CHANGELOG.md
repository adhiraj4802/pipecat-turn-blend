# Changelog

## 0.2.0 - 2026-11-15

- New model. Smart Turn v3.2 is replaced by a student model: a Whisper Base encoder trained to reproduce the
  end-of-turn scores of p99lab's turn-1 on HarperValleyBank, AMI and Smart Turn training audio. Silero VAD
  and turn-1-mini are unchanged. `turn-blend.onnx` is now 105 MB.
- Blend weights refitted on held-out audio instead of the benchmark.
- eot-bench v1.1, English, 2,288 turns: false cut-offs at a 300 ms budget fall from 32.5% to 23.5%, and
  latency at a 5% cut-off budget from 1114 ms to 995 ms. Per-cohort tables, every published model, the
  per-turn scores and the scripts are in `benchmark/`.
- New defaults: `blend_threshold` 0.75 (was 0.85), `blend_min_silence_ms` 160 (was 320),
  `tentative_threshold` 0.4 (was 0.5). A threshold tuned for 0.1.0 does not carry over.
- Scoring a pause costs about twice as much CPU as in 0.1.0 (68 ms against 31 ms on 4 laptop threads).
- `TurnBlendAnalyzer` measures the student's cost and runs it less often when the machine cannot keep up in
  real time, instead of falling behind. `analyzer.score_every_chunks` holds the interval in use.
- No API change. The `smart_turn` output and the `w_smart_turn` and `smart_turn_max_secs` settings keep
  their names and now refer to the student model.
- Licence files updated: Smart Turn's licence removed, turn-1's NOTICE and the training-data credits added.

## 0.1.0 - 2026-11-13

- First release: `TurnBlendAnalyzer` for Pipecat, the `TurnBlend` runner and `turn-blend.onnx` (Silero VAD
  v6.2, Smart Turn v3.2, turn-1-mini and the blend in one file), published as a release asset.
- All 16 model settings can be set by keyword argument, JSON file or `TURN_BLEND_<SETTING>` environment
  variable.
- eot-bench results and CPU latency measurements in `benchmark/`.
- Tested with Pipecat v1.16.0.
