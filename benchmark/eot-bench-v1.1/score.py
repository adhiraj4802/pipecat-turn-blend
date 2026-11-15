"""
Score eot-bench turns with turn-blend.onnx, streaming each turn chunk by chunk as a live call would.

    python score.py DATA_DIR GRID MODEL OUT_DIR [PART PARTS]

DATA_DIR  folder with the English parquet files of livekit/eot-bench-data (data/en/)
GRID      any predictions.parquet from the eot-bench repository's output/ folder (it lists the decision points)
MODEL     turn-blend.onnx
OUT_DIR   one JSON per turn is written here
PART PARTS  optional: run slice PART of PARTS, to split the work over processes

The commit rule is switched off, so the turn stays open and the blend is read at every decision point of the
benchmark grid (every 0.1 s inside each labelled pause). The student model only runs on the chunks a grid
point reads.
"""
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from pipecat_turn_blend import TurnBlend

data, grid_path, model, out = sys.argv[1:5]
part, parts = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
g = pd.read_parquet(grid_path, columns=["id", "span_index", "timestamp", "silence_dur", "label"])
grid = {i: d for i, d in g.groupby("id", sort=False)}
det = TurnBlend(model, threads=4, score_every_chunks=1, blend_threshold=2.0, timeout_secs=1e9)
os.makedirs(out, exist_ok=True)
idx = -1
for f in sorted(glob.glob(data + "/*.parquet")):
    for batch in pq.ParquetFile(f).iter_batches(batch_size=16, columns=["id", "audio", "cohort"]):
        for r in batch.to_pylist():
            idx += 1
            dest = os.path.join(out, r["id"].replace("/", "_") + ".json")
            if idx % parts != part or r["id"] not in grid or os.path.exists(dest):
                continue
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            assert sr == 16000, sr
            if a.ndim > 1:
                a = a.mean(1)
            n, rows = len(a) // 512, grid[r["id"]]
            ks = [min(n, int(np.floor(t * 16000 + 1e-6)) // 512) - 1 for t in rows.timestamp]   # last whole chunk before each point
            need, res = set(ks), {}
            det.reset()
            for i in range(n):
                det.configure(score_every_chunks=1 if i in need else 1e9)
                o = det(a[i * 512:(i + 1) * 512])
                if i in need:
                    res[i] = o
            pts = []
            for k, row in zip(ks, rows.itertuples()):
                o = res.get(k)
                ok = bool(o and o["blend"] >= 0)          # False when no turn is open: the model cannot end a turn there
                pts.append({"span_index": int(row.span_index), "timestamp": float(row.timestamp),
                            "silence_dur": float(row.silence_dur), "label": row.label,
                            "smart_turn": o["smart_turn"] if ok else 0.0, "mini": o["mini"] if o else 0.0,
                            "silence_s": o["silence_ms"] / 1000 if o else 0.0, "started": bool(o and o["in_turn"]),
                            "decidable": ok, "blend": o["blend"] if ok else 0.0})
            with open(dest, "w") as fh:
                json.dump({"id": r["id"], "cohort": r["cohort"], "audio_sec": len(a) / 16000, "points": pts}, fh)
