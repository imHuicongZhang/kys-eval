"""Offline self-test of the pipeline logic (no GPU, no network, no evaluation).

  python -m kys_eval.selftest

1. model spec parsing and the 72-cell grid
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
import statistics
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
    assert n == 72 and len(cells) == n and len({c["hf_subfolder"] for c in cells}) == n
    assert cells[0]["hf_subfolder"] == "rewrite-1p5b/seed42/raw_diversity_oriented/ep1/hf"
    assert cells[21]["hf_subfolder"] == "rewrite-1p5b/seed42/signal-disagreement-lambda0/ep1/hf", cells[21]
    assert C.RAW_COMPARATOR["signal-disagreement-lambda0"] == "disagreement_aware"
    assert C.REWRITTEN_V2_RUN_STEM[C.RAW_COMPARATOR["signal-disagreement-lambda0"]] == "signal-disagreement-lambda05"
    assert {C.comparison_kind(s) for s in C.RAW_SETTINGS} == {"strategy-linked", "global Top-10B", "λ grid (λ = 0 vs 0.5)"}
    assert len(selected_cells([43], C.RAW_SETTINGS, C.EPOCHS)) == len(C.RAW_SETTINGS) * len(C.EPOCHS) == 24
    print("ok  model specs and grid")


def reference_tasks():
    """{(setting, seed, epoch): {task: {"acc", "acc_norm"}}} from the full-precision v2 reference JSON."""
    data = json.loads(C.REWRITTEN_V2_PER_TASK_JSON.read_text())
    return {(c["setting"], int(c["seed"]), c["epoch"]): {t: dict(v) for t, v in c["tasks"].items()} for c in data["cells"]}


def test_pairing_statistics():
    """Pairing at different serialization precision, genuine differences, exact ties, mean / ddof-1 SD / seed counts."""
    ref = reference_tasks()
    # 1. serialization: every reference double survives json round trips (the results.json path) unchanged
    for tasks in ref.values():
        for v in tasks.values():
            for x in v.values():
                assert json.loads(json.dumps(x)) == x and json.loads(json.dumps({"x": x}, indent=1))["x"] == x
    # ... while the 10-decimal CSV does not reproduce them: the hazard the JSON reference removes
    csv_vals = {(r["setting"], int(r["seed"]), r["epoch"], r["task"]): float(r["acc_norm"])
                for r in csv.DictReader(open(C.REWRITTEN_V2_PER_TASK_CSV))}
    rounded_differs = sum(csv_vals[(*k, t)] != v["acc_norm"] for k, tasks in ref.items() for t, v in tasks.items())
    assert rounded_differs > 0 and all(abs(csv_vals[(*k, t)] - v["acc_norm"]) <= 5e-11
                                       for k, tasks in ref.items() for t, v in tasks.items())
    try:
        aggregate.load_rewritten(C.REWRITTEN_V2_PER_TASK_CSV)
        raise AssertionError("the 10-decimal CSV must be refused as a pairing reference")
    except (SystemExit, ValueError):
        pass

    # 2.-4. one raw setting (raw_diversity_oriented vs diversity_oriented) with known per-seed offsets on every task
    setting, comp = "raw_diversity_oriented", C.RAW_COMPARATOR["raw_diversity_oriented"]
    cases = {  # epoch -> per-seed offset added to every acc_norm (fraction); mean, ddof-1 SD, positive seeds
        "ep1": {42: 0.01, 43: -0.02, 44: 0.03},    # genuine mixed signs
        "ep2": {42: 0.0, 43: 0.0, 44: 0.0},        # exact ties
        "ep3": {42: 0.01, 43: 0.0, 44: -0.01},     # one tie, one positive, one negative
    }
    with tempfile.TemporaryDirectory() as tmp:
        root, reports = Path(tmp) / "results", Path(tmp) / "reports"
        for seed in C.SEEDS:
            for ep, offs in cases.items():
                src = ref[(comp, seed, ep)]
                shifted = {t: {"acc": v["acc"], "acc_norm": v["acc_norm"] + offs[seed]} for t, v in src.items()}
                cell = next(c for c in C.grid_cells() if (c["setting"], c["seed"], c["epoch"]) == (setting, seed, ep))
                out = cell_dir(root, cell)
                out.mkdir(parents=True)
                common.write_json(out / "results.json", results_from_raw(fake_raw(shifted), cell=cell))
        aggregate.main(["--results-root", str(root), "--out-dir", str(reports)])
        rows = {(r["raw_setting"], r["comparator_setting"], r["epoch"], r["benchmark"]): r
                for r in csv.DictReader(open(reports / "raw_vs_rewritten.csv"))}
        v2 = aggregate.load_rewritten()
        for ep, offs in cases.items():
            for bench, _ in aggregate.ALL_BENCHMARKS:
                # expected per-seed differences, computed independently from the shifted task values
                d = []
                for seed in C.SEEDS:
                    src = ref[(comp, seed, ep)]
                    new = common.benchmark_scores({t: v["acc_norm"] + offs[seed] for t, v in src.items()})[bench]
                    d.append(new - v2[(comp, seed, ep)][bench])
                r = rows[(setting, comp, ep, bench)]
                assert r["n_paired"] == "3", r
                assert r["n_delta_positive"] == str(sum(x > 0 for x in d)), (r, d)
                assert float(r["delta_mean"]) == float(f"{statistics.fmean(d):.6f}"), (r, d)
                assert float(r["delta_std"]) == float(f"{statistics.stdev(d):.6f}"), (r, d)
                if ep == "ep2":
                    assert float(r["delta_mean"]) == 0.0 and r["n_delta_positive"] == "0", r
                # the offsets themselves (every benchmark is a mean of shifted values, so Δ = offset)
                assert max(abs(x - offs[s]) for x, s in zip(d, C.SEEDS)) < 1e-12, (bench, d)
        want = {"ep1": (0.02 / 3, statistics.stdev([0.01, -0.02, 0.03]), 2), "ep2": (0.0, 0.0, 0),
                "ep3": (0.0, 0.01, 1)}
        for ep, (m, sd, k) in want.items():
            r = rows[(setting, comp, ep, "mean6")]
            assert abs(float(r["delta_mean"]) - m) < 1e-6 and abs(float(r["delta_std"]) - sd) < 1e-6, (ep, r)
            assert r["n_delta_positive"] == str(k), (ep, r)
    print("ok  pairing: full-precision reference round-trips; CSV refused; mixed-sign, tied and partly tied "
          "differences give the right mean, ddof-1 SD and positive-seed counts")


def test_aggregation():
    per_cell = reference_tasks()

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
        n_pairs = sum(len(aggregate.comparisons(r)) for r in C.RAW_SETTINGS)
        assert n_pairs == len(C.RAW_SETTINGS) + 5, n_pairs  # 4 strategy-linked + lambda0 also vs quality_base
        assert len(rows) == n_pairs * 3 * len(aggregate.ALL_BENCHMARKS), len(rows)
        v2 = aggregate.load_rewritten()
        for row in rows:
            assert row["n_paired"] == "3", row
            if row["comparison"] == "primary":  # identical doubles on both sides: exactly 0, no tolerance
                assert row["comparator_setting"] == rewritten_of[row["raw_setting"]], row
                assert float(row["delta_mean"]) == 0.0 and float(row["delta_std"]) == 0.0, row
                assert row["n_delta_positive"] == "0", row
            else:  # the raw cell carries its primary comparator's scores, so Δ = primary − quality_base
                assert row["comparison"] == "quality_base" == row["comparator_setting"], row
                d = [v2[(rewritten_of[row["raw_setting"]], s, row["epoch"])][row["benchmark"]]
                     - v2[("quality_base", s, row["epoch"])][row["benchmark"]] for s in C.SEEDS]
                assert abs(float(row["delta_mean"]) - statistics.fmean(d)) < 2e-6, (row, d)
                assert abs(float(row["delta_std"]) - statistics.stdev(d)) < 2e-6, (row, d)
                assert int(row["n_delta_positive"]) == sum(x > 0 for x in d), (row, d)

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
    print(f"ok  aggregation reproduces the v2 Mean6/MMLU tables; primary-pair deltas are 0 and quality_base-pair "
          f"deltas equal comparator − quality_base ({n_pairs} pairs x 3 x 12)")


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


def test_result_identity():
    """E-F1: complete results are reused only for the checkpoint they belong to; E-F2: --expect-cells."""
    from kys_eval import eval_checkpoint, run_grid

    repo = C.HF_REPO
    c0, c1 = selected_cells([42], C.RAW_SETTINGS, C.EPOCHS)[:2]
    spec = lambda c, rev=None: run_grid.cell_spec(c, repo, rev)  # noqa: E731
    meta = lambda c: run_grid.cell_meta(c, repo)  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cell"
        out.mkdir()
        res = {"schema": common.RESULTS_SCHEMA, "complete": True, "cell": meta(c0),
               "model": {"spec": spec(c0), **common.parse_model_spec(spec(c0)), "commit": "abc123def"}}
        common.write_json(out / "results.json", res)
        assert eval_checkpoint.reusable(out, spec(c0), meta(c0))[0] is not None
        assert eval_checkpoint.reusable(out, spec(c0, "abc123"), meta(c0))[0] is not None  # commit prefix
        _, why = eval_checkpoint.reusable(out, spec(c0, "fff000"), meta(c0))
        assert why and "revision" in why[0], why
        _, why = eval_checkpoint.reusable(out, spec(c1), meta(c1))  # another cell's results in this folder
        assert any("model path" in w for w in why) and any("epoch" in w for w in why), why
        wrong_step = dict(meta(c0), step=meta(c0)["step"] + 1)
        _, why = eval_checkpoint.reusable(out, spec(c0), wrong_step)
        assert why == [f"cell step is {c0['step']}, expected {c0['step'] + 1}"], why
        res["model"] = {"spec": "/some/local/dir", "kind": "local", "path": "/some/local/dir"}
        common.write_json(out / "results.json", res)
        assert eval_checkpoint.reusable(out, "/some/local/dir")[0] is not None
        assert eval_checkpoint.reusable(out, "/other/dir")[1]
        assert eval_checkpoint.reusable(out, spec(c0), meta(c0))[1]
        res["complete"] = False
        common.write_json(out / "results.json", res)
        assert eval_checkpoint.reusable(out, "/some/local/dir") == (None, [])
        r = ["--results-root", tmp, "--settings", "raw_diversity_oriented", "raw_disagreement_aware",
             "signal-disagreement-lambda0"]
        assert run_grid.main(r + ["--expect-cells", "72", "--index", "0"]) == 2
        assert run_grid.main(["--results-root", tmp, "--expect-cells", "27", "--index", "0"]) == 2
        assert run_grid.main(r + ["--expect-cells", "27", "--list"]) == 0
    print("ok  result identity check (E-F1) and array size check (E-F2)")


def main():
    test_specs_and_grid()
    test_result_identity()
    test_aggregation()
    test_pairing_statistics()
    test_reference_compare()
    print("selftest passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
