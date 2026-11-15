"""
What turn-blend does at its own settings: how many mid-turn pauses it interrupts and how long it takes to
answer at real ends of turn.

    python real_eval.py EOT_BENCH_REPO name=EVENTS_DIR [name=EVENTS_DIR ...]

EVENTS_DIR  output of real_run.py (data/events-*.tar.gz unpacked)

A mid-turn pause ("hold") counts as interrupted when the model ends the turn inside it. At a real end ("eot")
the delay runs from the start of the final silence to the model ending the turn; if it never does within the
labelled silence, the full length of that silence is used for the mean.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

repo = sys.argv[1]
PUB = os.path.join(repo, "output/livekit__eot-bench-data__validation__min_silence_100ms/en/")
g = pd.read_parquet(sorted(glob.glob(PUB + "*/predictions.parquet"))[0])
g["start"] = g.timestamp - g.silence_dur
spans = g.groupby(["id", "span_index"]).agg(label=("label", "first"), start=("start", "median"),
                                             end=("timestamp", "max")).reset_index()
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1)
    events, cohort = {}, {}
    for f in glob.glob(d + "/*.json"):
        j = json.load(open(f))
        events[j["id"]], cohort[j["id"]] = j["events"], j["cohort"]
    assert len(events) == spans.id.nunique(), (len(events), spans.id.nunique())
    rows = []
    for r in spans.itertuples():
        hit = [e for e in events[r.id] if r.start < e[0] <= r.end + 0.033]
        rows.append((cohort[r.id], r.label, hit[0][0] - r.start if hit else np.nan, hit[0][1] if hit else 0, r.end - r.start))
    df = pd.DataFrame(rows, columns=["cohort", "label", "t", "kind", "span"])
    print(f"\n{name}")
    for c in ["general", "accented", "short_turns", "pooled"]:
        q = df if c == "pooled" else df[df.cohort == c]
        hold, eot = q[q.label == "hold"], q[q.label == "eot"]
        print(f"  {c:12s} interrupted pauses {100 * hold.t.notna().mean():5.1f}% of {len(hold):4d} | "
              f"ends detected {100 * eot.t.notna().mean():5.1f}% of {len(eot):4d}, median delay {1000 * eot.t.median():4.0f} ms, "
              f"mean {1000 * eot.t.fillna(eot.span).mean():4.0f} ms, by timeout {100 * (eot.kind == 2).mean():4.1f}%")
