"""
Turn score files into the eot-bench headline metrics, per cohort, next to every model published in eot-bench.

    python evaluate.py EOT_BENCH_REPO name=SCORES_DIR [name=SCORES_DIR ...] [--json results.json]

EOT_BENCH_REPO  a clone of https://github.com/livekit/eot-bench (its metrics code and its committed predictions)
SCORES_DIR      output of score.py (data/scores-*.tar.gz unpacked)

A decision point where turn-blend had no open turn counts as probability 0.
"""
import glob
import json
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
args = [a for a in sys.argv[1:] if a != "--json"]
out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
if out_json:
    args.remove(out_json)
repo, runs = args[0], dict(a.split("=", 1) for a in args[1:])
sys.path.insert(0, repo)
from eot_harness.metrics import compute_metrics_from_predictions  # noqa: E402

PUB = os.path.join(repo, "output/livekit__eot-bench-data__validation__min_silence_100ms/en/")
COLS = ["id", "language", "span_index", "timestamp", "silence_dur", "label", "p_eot"]
COHORTS = ["general", "accented", "short_turns", "pooled"]


def headline(pred):
    table, summary = compute_metrics_from_predictions(pred)
    model = table[table.policy_type == "model"]
    out = {}
    for budget in (0.3, 0.6):
        q = model[model.mean_latency <= budget + 1e-9]
        out[f"cut@{int(budget * 1000)}"] = round(100 * float(q.cutoff_rate.min()), 1) if len(q) else None
    for k in ("5pct", "10pct"):
        op = summary["operating_points"].get(k)
        out[f"lat@{k}"] = round(1000 * op["mean_latency"]) if op else None
    return out


def load(scores_dir):
    rows, cohort = [], {}
    for f in glob.glob(scores_dir + "/*.json"):
        d = json.load(open(f))
        cohort[d["id"]] = d["cohort"]
        rows += [{"id": d["id"], "span_index": p["span_index"], "tr": round(p["timestamp"], 3),
                  "p_eot": p["blend"] if p["decidable"] else 0.0} for p in d["points"]]
    return pd.DataFrame(rows), cohort


results, cohort = {}, {}
grid = pd.read_parquet(sorted(glob.glob(PUB + "*/predictions.parquet"))[0]).drop(columns="p_eot")
grid["tr"] = grid.timestamp.round(3)
preds = {}
for name, d in runs.items():
    scores, cohort = load(d)
    preds[name] = grid.merge(scores, on=["id", "span_index", "tr"])
    assert len(preds[name]) == len(grid), f"{name}: {len(preds[name])} of {len(grid)} decision points"
for f in sorted(glob.glob(PUB + "*/predictions.parquet")):
    preds[os.path.basename(os.path.dirname(f))] = pd.read_parquet(f)
for name, pred in preds.items():
    pred = pred.assign(cohort=pred.id.map(cohort))
    results[name] = {c: headline((pred if c == "pooled" else pred[pred.cohort == c])[COLS]) for c in COHORTS}
    print(f"{name[:60]:60s} " + " | ".join(
        "{} {}/{}/{}/{}".format(c[:5], *results[name][c].values()) for c in COHORTS))
if out_json:
    json.dump(results, open(out_json, "w"), indent=1)
