"""Offline self-test of the pipeline logic (no GPU, no network, no evaluation).

  python -m kys_eval.selftest

1. model spec parsing and the 63-cell grid
2. aggregation: builds a synthetic raw-selected results tree in which every raw cell carries the
   scores of its rewritten counterpart, runs kys_eval.aggregate, and asserts that every delta is 0 and
   that the per-epoch Mean6/MMLU means reproduce the v2 tables in reference/
3. reference comparison: expected-vs-itself passes exactly; a changed hash, a large score change and
   a missing task fail; a small score change passes within tolerance
Exit status: 0 all passed.
"""

import csv
import copy
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from kys_eval import aggregate, common, reference_check
from kys_eval import config as C
from kys_eval.run_grid import cell_dir, selected_cells


def fake_raw(task_metrics, hashes=None):
    """A minimal LightEval results_*.json dict."""
    return {
        "config_general": {"max_samples": None, "lighteval_sha": "selftest", "model_config": {}},
        "config_tasks": {f"{t}|0": {"num_fewshots": 0} for t in task_metrics},
        "results": {f"{t}|0": m for t, m in task_metrics.items()},
        "summary_tasks": {f"{t}|0": {"hashes": h} for t, h in (hashes or {}).items()},
    }


def results_from_raw(raw, cell=None, gpu=C.REFERENCE_GPU):
    return common.build_results(
        raw, label="selftest", cell=cell, model={"spec": "selftest"}, environment={"gpu": gpu},
        command=None, started_at=None, finished_at=None, wall_seconds=None, raw_path="selftest",
    )


def test_specs_and_grid():
    s = common.parse_model_spec("hf://blab-jhu/KYS-1.5B-Raw-Selected-Baselines/rewrite-1p5b/seed42/raw_random/ep1/hf@abc123")
    assert s == {"kind": "hf", "repo_id": "blab-jhu/KYS-1.5B-Raw-Selected-Baselines",
                 "subfolder": "rewrite-1p5b/seed42/raw_random/ep1/hf", "revision": "abc123"}, s
    assert common.parse_model_spec("hf://org/repo")["subfolder"] == ""
    assert common.parse_model_spec("/tmp/x")["kind"] == "local"
    cells = C.grid_cells()
    n = len(C.RAW_SETTINGS) * len(C.SEEDS) * len(C.EPOCHS)
    assert n == 63 and len(cells) == n and len({c["hf_subfolder"] for c in cells}) == n
    assert cells[0]["hf_subfolder"] == "rewrite-1p5b/seed42/raw_diversity_oriented/ep1/hf"
    assert len(selected_cells([43], C.RAW_SETTINGS, C.EPOCHS)) == len(C.RAW_SETTINGS) * len(C.EPOCHS) == 21
    print("ok  model specs and grid")


def test_aggregation():
    per_cell = defaultdict(dict)
    with open(C.REWRITTEN_V2_PER_TASK_CSV, newline="") as f:
        for row in csv.DictReader(f):
            per_cell[(row["setting"], int(row["seed"]), row["epoch"])][row["task"]] = {
                "acc": float(row["acc"]), "acc_norm": float(row["acc_norm"])}

    with tempfile.TemporaryDirectory() as tmp:
        root, reports = Path(tmp) / "results", Path(tmp) / "reports"
        rewritten_of = C.RAW_COMPARATOR
        for cell in C.grid_cells():
            source = per_cell[(rewritten_of[cell["setting"]], cell["seed"], cell["epoch"])]
            results = results_from_raw(fake_raw(source), cell=cell)
            assert results["complete"], results["missing_tasks"]
            out = cell_dir(root, cell)
            out.mkdir(parents=True)
            common.write_json(out / "results.json", results)
        assert aggregate.main(["--results-root", str(root), "--out-dir", str(reports)]) == 0

        rows = list(csv.DictReader(open(reports / "raw_vs_rewritten.csv")))
        assert len(rows) == len(C.RAW_SETTINGS) * 3 * len(aggregate.ALL_BENCHMARKS), len(rows)
        for row in rows:
            assert row["n_paired"] == "3" and abs(float(row["delta_mean"])) < 1e-12, row

        by_epoch = {(r["setting"], r["epoch"], r["benchmark"]): r for r in csv.DictReader(open(reports / "raw_selected_by_epoch.csv"))}
        for name, bench in (("mean6", "mean6"), ("mmlu", "mmlu")):
            for v2 in csv.DictReader(open(C.REPO_ROOT / "reference" / f"rewritten_v2_{name}_by_epoch.csv")):
                raw_settings = [r for r, w in rewritten_of.items() if w == v2["setting"]]
                for raw_setting in raw_settings:
                    ours = by_epoch[(raw_setting, v2["epoch"], bench)]
                    assert abs(float(ours["mean"]) - float(v2["mean"])) < 2e-6, (ours, v2)
                    assert abs(float(ours["std"]) - float(v2["std"])) < 2e-6, (ours, v2)
        report = (reports / "raw_selected_report.md").read_text()
        assert f"Coverage: **{len(C.grid_cells())} / {len(C.grid_cells())}**" in report and "## Summary: average accuracy (Mean6)" in report
    print("ok  aggregation reproduces the v2 Mean6/MMLU tables; all 7x3x12 deltas are 0")


def test_reference_compare():
    expected = reference_check.load_expected()
    metrics = {t: {k: v for k, v in e.items() if k in common.METRIC_FIELDS} for t, e in expected["tasks"].items()}
    hashes = {t: e["hashes"] for t, e in expected["tasks"].items()}

    def verdict(mutate=None):
        m, h = copy.deepcopy(metrics), copy.deepcopy(hashes)
        if mutate:
            mutate(m, h)
        return reference_check.compare(results_from_raw(fake_raw(m, h)), expected)["verdict"]

    assert verdict() == "PASS (exact match)", verdict()
    assert verdict(lambda m, h: h["rw_piqa"].update(hash_input_tokens="0" * 16)) == "FAIL"
    assert verdict(lambda m, h: m["rw_openbookqa"].update(acc_norm=m["rw_openbookqa"]["acc_norm"] + 0.02)) == "FAIL"
    assert verdict(lambda m, h: m.pop("rw_mmlu:virology")) == "FAIL"
    assert verdict(lambda m, h: m["rw_mmlu:virology"].update(acc_norm=m["rw_mmlu:virology"]["acc_norm"] + 0.004)) == "PASS (within tolerance)"
    print("ok  reference comparison (exact pass, hash/score/missing-task failures, tolerance pass)")


def main():
    test_specs_and_grid()
    test_aggregation()
    test_reference_compare()
    print("selftest passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
