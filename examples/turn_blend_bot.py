"""
A Pipecat voice bot that uses turn-blend for end-of-turn detection. Runs on Daily or on Pipecat's local WebRTC.

  microphone -> Deepgram streaming STT -> turn-blend decides the turn -> OpenAI LLM -> Deepgram Aura-2 TTS

Setup:
    pip install "pipecat-turn-blend[example] @ git+https://github.com/adhiraj4802/pipecat-turn-blend.git"
    cp examples/.env.example .env      # fill in DEEPGRAM_API_KEY, OPENAI_API_KEY (and DAILY_API_KEY for Daily)
Run:
    python examples/turn_blend_bot.py -t webrtc    # no Daily account needed; open http://localhost:7860, press Connect
    python examples/turn_blend_bot.py -t daily     # creates a Daily room; open http://localhost:7860 to join it

The model (105 MB) is downloaded on the first run. Any setting can be changed in .env as TURN_BLEND_<SETTING>.

Each turn logs how long turn-blend took to decide and the full turn latency (you stop speaking -> the bot
starts speaking). The only turn-blend specific lines are the `TurnBlendAnalyzer` ones below.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat_turn_blend import TurnBlendAnalyzer, load_settings

load_dotenv()

SYSTEM = ("You are a friendly voice assistant in a live phone-style conversation. Your replies are spoken aloud, "
          "so keep them to one or two short sentences, with no lists, markdown or emoji.")


def make_transport(args: RunnerArguments):
    if isinstance(args, DailyRunnerArguments):
        from pipecat.transports.daily.transport import DailyParams, DailyTransport
        return DailyTransport(args.room_url, args.token, "turn-blend bot",
                              DailyParams(audio_in_enabled=True, audio_out_enabled=True, camera_out_enabled=False))
    if isinstance(args, SmallWebRTCRunnerArguments):
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
        return SmallWebRTCTransport(webrtc_connection=args.webrtc_connection,
                                    params=TransportParams(audio_in_enabled=True, audio_out_enabled=True))
    raise SystemExit("Use -t daily or -t webrtc")


async def bot(runner_args: RunnerArguments):
    transport = make_transport(runner_args)
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"],
                             settings=DeepgramSTTService.Settings(model="nova-3", language="en"))
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"],
                           settings=OpenAILLMService.Settings(model="gpt-5.4-nano", extra={"reasoning_effort": "none"}))
    tts = DeepgramTTSService(api_key=os.environ["DEEPGRAM_API_KEY"],
                             settings=DeepgramTTSService.Settings(voice="aura-2-thalia-en"))

    # ---- turn-blend: ends the user's turn. Pipecat's Silero VAD starts it and handles interruptions.
    turn_blend = TurnBlendAnalyzer(**load_settings())   # settings: TURN_BLEND_* from .env
    context = LLMContext([{"role": "system", "content": SYSTEM}])
    aggregators = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2)),
        user_turn_strategies=UserTurnStrategies(stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_blend)]),
    ))

    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_measured")
    async def on_latency(_, seconds):
        r = turn_blend.last_result or {}
        logger.info(f"full turn latency {seconds * 1000:.0f} ms (you stopped -> bot started speaking); "
                    f"turn-blend decided after {r.get('detect_ms', '?')} ms ({r.get('reason', '?')})")

    pipeline = Pipeline([transport.input(), stt, aggregators.user(), llm, tts, transport.output(), aggregators.assistant()])
    worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=16000, enable_metrics=True),
                            observers=[latency])

    @transport.event_handler("on_client_connected")
    async def on_connected(_, client):
        context.add_message({"role": "system", "content": "Greet the caller in one short sentence."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_, client):
        await worker.cancel()

    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(worker)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
