"""
Run turn-blend over eot-bench turns exactly as it ships, with its own commit rule, and log every end of turn.

    python real_run.py DATA_DIR GRID MODEL OUT_DIR [PART PARTS] [--config '{"blend_threshold": 0.72}']

Arguments as for score.py. Nothing is switched off here: the model opens and ends turns by itself at the
default cadence (the student model runs on every second chunk of a pause).
"""
import glob
import io
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from pipecat_turn_blend import TurnBlend

argv = list(sys.argv[1:])
config = json.loads(argv.pop(argv.index("--config") + 1)) if "--config" in argv else {}
if "--config" in argv:
    argv.remove("--config")
data, grid_path, model, out = argv[:4]
part, parts = (int(argv[4]), int(argv[5])) if len(argv) > 5 else (0, 1)
ids = set(pd.read_parquet(grid_path, columns=["id"]).id)
det = TurnBlend(model, threads=2, **config)
os.makedirs(out, exist_ok=True)
idx = -1
for f in sorted(glob.glob(data + "/*.parquet")):
    for batch in pq.ParquetFile(f).iter_batches(batch_size=16, columns=["id", "audio", "cohort"]):
        for r in batch.to_pylist():
            idx += 1
            dest = os.path.join(out, r["id"].replace("/", "_") + ".json")
            if idx % parts != part or r["id"] not in ids or os.path.exists(dest):
                continue
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            assert sr == 16000, sr
            if a.ndim > 1:
                a = a.mean(1)
            det.reset()
            events, tentative, was = [], [], False
            for i in range(len(a) // 512):
                o = det(a[i * 512:(i + 1) * 512])
                t = round((i + 1) * 0.032, 3)
                if o["end_of_turn"]:                       # [time s, 1 = blend committed / 2 = silence timeout, silence ms]
                    events.append([t, int(o["end_of_turn"]), round(o["silence_ms"])])
                if o["tentative"] and not was:
                    tentative.append(t)
                was = bool(o["tentative"])
            with open(dest, "w") as fh:
                json.dump({"id": r["id"], "cohort": r["cohort"], "audio_sec": len(a) / 16000,
                           "events": events, "tentative": tentative}, fh)
