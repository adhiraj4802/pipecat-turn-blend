"""
Configure every turn-blend setting from the command line, a JSON file or the environment, and see the effect.

    python examples/configure_demo.py --list                                   # every setting, its default and meaning
    python examples/configure_demo.py recording.wav                            # run with the defaults
    python examples/configure_demo.py recording.wav --blend-threshold 0.95 --timeout-secs 1.5
    python examples/configure_demo.py recording.wav --config settings.json     # {"blend_threshold": 0.95, ...}
    TURN_BLEND_VAD_CONFIDENCE=0.5 python examples/configure_demo.py recording.wav
    python examples/configure_demo.py recording.wav --blend-threshold 0.95 --compare   # defaults next to your settings
    python examples/configure_demo.py --blend-threshold 0.95 --save settings.json      # write the full settings file

There is one flag per setting; the flags are built from the list stored in the model, so a newer model with
more settings needs no change here. When the same setting is given more than once, the order is:
model defaults < --config file < TURN_BLEND_<SETTING> environment variables < command-line flags.

The WAV file must be 16 kHz, 16-bit PCM (mono, or stereo which is mixed down).
Convert with:  ffmpeg -i input.mp3 -ar 16000 -ac 1 recording.wav

In your own code the same settings are keyword arguments:

    analyzer = TurnBlendAnalyzer(blend_threshold=0.95)                # or TurnBlend(model_path(), ...)
    analyzer.detector.configure(timeout_secs=1.5)                     # takes effect on the next chunk
"""

import argparse
import json
import sys
import wave

import numpy as np

from pipecat_turn_blend import TurnBlend, load_settings, model_path
from pipecat_turn_blend.turn_blend import ENV_PREFIX


def read_wav(path):
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getsampwidth() != 2:
            sys.exit("Need a 16 kHz, 16-bit WAV file (see the ffmpeg line at the top of this script).")
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
    return audio


def find_turns(detector, audio):
    """Run the detector over the audio; one dict per ended turn."""
    detector.reset()
    n, turns, started = detector.chunk_samples, [], None
    for i in range(len(audio) // n):
        r = detector(audio[i * n:(i + 1) * n])
        t = (i + 1) * n / detector.sample_rate
        if r["in_turn"] and started is None:
            started = t
        if r["end_of_turn"]:
            turns.append({"start": started if started is not None else t, "end": t, **r})
            started = None
    return turns


def print_turns(turns):
    for t in turns:
        how = "blend committed" if t["end_of_turn"] == 1 else "silence timeout"
        scores = f"   (blend {t['blend']:.2f})" if t["blend"] >= 0 else ""
        print(f"  turn {t['start']:7.2f} s -> {t['end']:7.2f} s   {how} after {t['silence_ms']:.0f} ms of silence{scores}")
    committed = sum(t["end_of_turn"] == 1 for t in turns)
    silence = np.mean([t["silence_ms"] for t in turns]) if turns else 0.0
    print(f"  {len(turns)} turns: {committed} by the blend, {len(turns) - committed} by silence timeout; "
          f"mean silence before the end {silence:.0f} ms")


def main():
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--model")
    detector = TurnBlend(first.parse_known_args()[0].model or model_path())               # loaded first: the model itself lists its settings

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="?", help="16 kHz, 16-bit WAV file to run")
    ap.add_argument("--model", help="path to turn-blend.onnx (default: downloaded on first use)")
    ap.add_argument("--list", action="store_true", help="print every setting and exit")
    ap.add_argument("--config", metavar="FILE", help="JSON file of settings")
    ap.add_argument("--save", metavar="FILE", help="write the resulting settings to a JSON file")
    ap.add_argument("--compare", action="store_true", help="also run with the defaults and show both")
    group = ap.add_argument_group("settings (also: --config file, or the environment variable shown)")
    for name in detector.names:
        group.add_argument("--" + name.replace("_", "-"), dest=name, type=float, metavar="X",
                           help=f"{detector.meanings[name]} (default {detector.defaults[name]:g}; "
                                f"{ENV_PREFIX}{name.upper()})")
    args = ap.parse_args()

    settings = load_settings(args.config)            # file, then environment
    settings.update({n: getattr(args, n) for n in detector.names if getattr(args, n) is not None})   # then flags
    try:
        detector.configure(**settings)
    except (KeyError, ValueError) as e:
        sys.exit(f"Bad setting: {e.args[0]}")

    print(detector.describe())
    if args.save:
        with open(args.save, "w") as f:
            json.dump(detector.config, f, indent=2)
        print(f"\nSaved to {args.save}")
    if args.list or not args.wav:
        if not (args.list or args.save):
            print("\nGive a WAV file to run these settings on it (--help for everything).")
        return

    audio = read_wav(args.wav)
    print(f"\n{args.wav}: {len(audio) / detector.sample_rate:.1f} s")
    if args.compare:
        defaults = detector.new_stream()
        defaults.reset_config()
        print("\nWith the defaults:")
        print_turns(find_turns(defaults, audio))
        print("\nWith your settings:")
    else:
        print()
    print_turns(find_turns(detector, audio))


if __name__ == "__main__":
    main()
