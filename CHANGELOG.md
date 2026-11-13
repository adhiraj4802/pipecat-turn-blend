# Changelog

## 0.1.0 - 2026-11-13

- First release: `TurnBlendAnalyzer` for Pipecat, the `TurnBlend` runner and `turn-blend.onnx` (Silero VAD
  v6.2, Smart Turn v3.2, turn-1-mini and the blend in one file), published as a release asset.
- All 16 model settings can be set by keyword argument, JSON file or `TURN_BLEND_<SETTING>` environment
  variable.
- eot-bench results and CPU latency measurements in `benchmark/`.
- Tested with Pipecat v1.16.0.
