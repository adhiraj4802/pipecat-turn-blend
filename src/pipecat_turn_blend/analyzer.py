"""Pipecat turn analyzer backed by turn-blend.onnx.

Pipecat's own VAD still opens the user turn and handles interruptions. This analyzer receives every audio
frame, runs the model on a worker thread, and reports the turn COMPLETE when the blend commits (or the
silence timeout runs out). Pipecat then waits for the final transcript and starts the LLM.

Written against pipecat-ai 1.16.
"""

import math
import queue
import threading
import time
from collections.abc import Callable

import numpy as np
from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.metrics.metrics import MetricsData, TurnMetricsData

from .model import model_path as default_model_path
from .turn_blend import TurnBlend


class TurnBlendParams(BaseTurnParams):
    """Parameters reported to Pipecat.

    Parameters:
        stop_secs: The silence timeout, in seconds. Mirrors the ``timeout_secs`` setting.
    """

    stop_secs: float = 3.0


class TurnBlendAnalyzer(BaseTurnAnalyzer):
    """End-of-turn detection with turn-blend, a single local ONNX model.

    The model holds Silero VAD, a student model (a Whisper Base encoder trained on the scores of p99lab's
    turn-1), turn-1-mini and a logistic blend of the two scores with the
    elapsed silence. It runs on the CPU, needs no API key and takes 16 kHz audio. English only.

    Example::

        aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(stop=[
                TurnAnalyzerUserTurnStopStrategy(turn_analyzer=TurnBlendAnalyzer()),
            ]),
        ))
        PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=16000))
    """

    def __init__(self, model_path: str | None = None, *, threads: int = 4,
                 on_end_of_turn: Callable[[dict], None] | None = None, **config):
        """Initialize the analyzer and load the model.

        Args:
            model_path: Path to turn-blend.onnx. If omitted, the model is downloaded on first use and cached.
            threads: CPU threads for the model (2 or more for real-time use). If the student model cannot run
                every ``score_every_chunks`` chunks in real time with these threads, the analyzer runs it less
                often; ``analyzer.score_every_chunks`` holds the interval in use.
            on_end_of_turn: Optional callback with the result of each ended turn. Called on the worker thread.
            **config: Any turn-blend setting, e.g. ``blend_threshold=0.9``. ``analyzer.detector.describe()``
                lists them all.
        """
        super().__init__(sample_rate=16000)
        self.detector = TurnBlend(model_path or default_model_path(), threads=threads, **config)
        self.last_result: dict | None = None      # details of the most recent end of turn
        self._on_end = on_end_of_turn
        self._pcm = np.zeros(0, np.float32)
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._complete: TurnMetricsData | None = None
        self._warned = False
        self.score_every_chunks = self.detector.config["score_every_chunks"]   # the interval in use (see _worker)
        self._thread = threading.Thread(target=self._worker, name="turn-blend", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ Pipecat interface ---
    @property
    def speech_triggered(self) -> bool:
        """Whether a user turn is open.

        Returns:
            True while the model has an open turn.
        """
        return self.detector.in_turn

    @property
    def params(self) -> TurnBlendParams:
        """The parameters Pipecat reads.

        Returns:
            The silence timeout currently configured.
        """
        return TurnBlendParams(stop_secs=self.detector.config["timeout_secs"])

    def set_sample_rate(self, sample_rate: int):
        """Record the pipeline's input sample rate and warn if it is not 16 kHz.

        Args:
            sample_rate: The sample rate the pipeline delivers.
        """
        super().set_sample_rate(sample_rate)
        if sample_rate != 16000 and not self._warned:
            self._warned = True
            logger.error(f"turn-blend needs 16 kHz input audio but the pipeline delivers {sample_rate} Hz. "
                         "Set PipelineParams(audio_in_sample_rate=16000).")

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Queue audio for the model.

        Args:
            buffer: 16-bit mono PCM at 16 kHz.
            is_speech: Pipecat's VAD verdict for this audio. Not used; the model has its own VAD.

        Returns:
            COMPLETE once the model has ended the turn, otherwise INCOMPLETE.
        """
        pcm = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        self._pcm = np.concatenate([self._pcm, pcm])
        n = self.detector.chunk_samples
        while len(self._pcm) >= n:
            self._q.put((self._pcm[:n].copy(), time.time()))
            self._pcm = self._pcm[n:]
        return EndOfTurnState.COMPLETE if self._complete is not None else EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        """Report whether the model has ended the turn.

        Returns:
            The state, and the turn metrics when the turn is complete.
        """
        with self._lock:
            done, self._complete = self._complete, None
        return (EndOfTurnState.COMPLETE, done) if done is not None else (EndOfTurnState.INCOMPLETE, None)

    def clear(self):
        """Tell the model that Pipecat ended the turn."""
        self._q.put(("clear", 0.0))

    async def cleanup(self):
        """Stop the worker thread."""
        self._q.put((None, 0.0))

    # ---------------------------------------------------------------- worker thread ---
    def _worker(self):
        was_in_turn, last = False, None
        # Keeping up with real time: the student model is the expensive part and runs on every
        # `score_every_chunks`-th chunk of a pause. If one run takes longer than that many chunks of audio, the
        # worker would fall behind, so the interval is widened to fit the measured cost on this machine.
        every_index = self.detector.names.index("score_every_chunks")
        chunk_ms = 1000.0 * self.detector.chunk_samples / self.detector.sample_rate
        cost_ms = 0.0                                  # running average of one student run
        while True:
            chunk, arrived = self._q.get()
            if chunk is None:
                return
            if isinstance(chunk, str):
                # Pipecat ended the turn. If ours is still open and the user is not mid-speech, close it too.
                if self.detector.in_turn and last is not None and not last["speech"]:
                    self.detector.close_turn()
                    was_in_turn = False
                continue
            every = max(self.detector.config["score_every_chunks"], math.ceil(1.2 * cost_ms / chunk_ms))
            if self._q.qsize() >= 8:                   # a quarter of a second behind: skip the student and catch up
                every = 1e9
            elif every != self.score_every_chunks:
                self.score_every_chunks = every
                logger.debug(f"turn-blend: the student model runs every {every:g} chunks ({cost_ms:.0f} ms per run)")
            self.detector._config[every_index] = every
            started = time.perf_counter()
            last = r = self.detector(chunk)
            if r["smart_turn"] >= 0:
                took = (time.perf_counter() - started) * 1000
                cost_ms = took if not cost_ms else 0.8 * cost_ms + 0.2 * took
            if r["in_turn"] and not was_in_turn:
                with self._lock:
                    self._complete = None          # a verdict from before this speech is stale
            was_in_turn = bool(r["in_turn"])
            if r["end_of_turn"]:
                lag_ms = (time.time() - arrived) * 1000
                result = {
                    "reason": "commit" if r["end_of_turn"] == 1 else "timeout",
                    "silence_ms": round(r["silence_ms"]),
                    "detect_ms": round(r["silence_ms"] + lag_ms),   # last speech -> decision, including compute
                    "blend": round(r["blend"], 3), "smart_turn": round(r["smart_turn"], 3), "mini": round(r["mini"], 3),
                }
                self.last_result = result
                logger.debug(f"turn-blend end of turn: {result}")
                with self._lock:
                    self._complete = TurnMetricsData(processor="TurnBlendAnalyzer", model="turn-blend", is_complete=True,
                                                     probability=max(r["blend"], 0.0),
                                                     e2e_processing_time_ms=float(result["detect_ms"]))
                if self._on_end:
                    self._on_end(result)
