#!/usr/bin/env python
"""Build reference/rewritten_v2_54_per_task.json: the v2 comparator values at full precision.

aggregate.py pairs every raw-selected cell with v2 values read from this file. It replaces the 10-decimal CSV
(rewritten_v2_54_per_task.csv, kept as a human-readable copy), whose rounding made exact ties look like ±1e-11
differences and miscounted positive seeds.

Precision. LightEval writes its results JSON with Python's json module, which serializes a float as its shortest
round-trip repr, and this script writes the values the same way. A value read from here is therefore the very
IEEE-754 double LightEval computed, and a kys-eval results.json (also written with json, common.write_json) holds the
same doubles. No tolerance is needed anywhere downstream.

Run on the cluster that holds the v2 evaluation outputs (projects/rewrite/08_evaluation/04_results_1p5_v2):

    python reference/build_rewritten_v2_reference.py --v2-root <...>/04_results_1p5_v2/downstream [--check-csv]

For each of the 54 cells (6 settings x seeds 42-44 x steps 4768/9537/14305) it requires exactly one
results_*.json whose model_name ends in /rewrite-1p5b-hf/<run>/<step>, records its sha256, and stores acc and
acc_norm for the 64 task keys of the CSV (63 tasks + rw_mmlu:_average). --check-csv asserts that every CSV value is
the 10-decimal rounding of the stored value.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kys_eval import config as C  # noqa: E402

SCHEMA = "kys-eval/rewritten-v2-reference/v1"
OUT = ROOT / "reference" / "rewritten_v2_54_per_task.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-root", required=True, help="08_evaluation/04_results_1p5_v2/downstream")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check-csv", action="store_true", help="also check the 10-decimal CSV against the new file")
    args = ap.parse_args(argv)

    cells = []
    for setting in C.REWRITTEN_SETTINGS:
        for seed in C.SEEDS:
            for epoch in C.EPOCHS:
                step = C.EPOCH_STEPS[epoch]
                run = f"{C.REWRITTEN_V2_RUN_STEM[setting]}-10B-1.5B-seed{seed}"
                hits = sorted(glob.glob(os.path.join(args.v2_root, run, str(step), "results", "**", "results_*.json"),
                                        recursive=True))
                if len(hits) != 1:
                    sys.exit(f"{run}/{step}: {len(hits)} results files")
                path = Path(hits[0])
                raw = path.read_bytes()
                d = json.loads(raw)
                if not d["config_general"]["model_name"].endswith(f"/rewrite-1p5b-hf/{run}/{step}"):
                    sys.exit(f"{path}: model_name {d['config_general']['model_name']}")
                tasks = {k.split("|")[0]: {"acc": v["acc"], "acc_norm": v["acc_norm"]}
                         for k, v in d["results"].items() if k != "all"}
                missing = [t for t in C.EXPECTED_TASKS if t not in tasks]
                if missing:
                    sys.exit(f"{path}: missing {missing}")
                gpu_f = Path(args.v2_root) / run / str(step) / "gpu.txt"
                cells.append({"setting": setting, "seed": seed, "epoch": epoch, "step": step,
                              "v2_run": f"{run}/{step}",
                              "source": str(path.relative_to(args.v2_root)),
                              "source_sha256": hashlib.sha256(raw).hexdigest(),
                              "lighteval_sha": d["config_general"].get("lighteval_sha"),
                              "gpu": gpu_f.read_text().strip() if gpu_f.is_file() else "",
                              "tasks": dict(sorted(tasks.items()))})
    out = {"schema": SCHEMA,
           "description": "acc and acc_norm of the 54 v2 rewritten-grid checkpoints, copied from their LightEval "
                          "results JSONs without rounding (floats serialized as shortest round-trip repr)",
           "generated_by": "reference/build_rewritten_v2_reference.py", "cells": cells}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}: {len(cells)} cells, {sum(len(c['tasks']) for c in cells)} task entries")

    if args.check_csv:
        by = {(c["setting"], c["seed"], c["epoch"]): c for c in cells}
        n = 0
        for row in csv.DictReader(open(C.REWRITTEN_V2_PER_TASK_CSV)):
            t = by[(row["setting"], int(row["seed"]), row["epoch"])]["tasks"][row["task"]]
            for k in ("acc", "acc_norm"):
                if f"{t[k]:.10f}" != row[k]:
                    sys.exit(f"CSV {row} != {t}")
            n += 1
        print(f"CSV check: {n} rows are the 10-decimal rounding of the JSON values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
