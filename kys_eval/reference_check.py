"""Reference check: re-score one existing v2 checkpoint and compare against its recorded scores.

Run this once on a new cluster before scoring the raw-selected grid. The reference is
diversity_oriented seed42 ep3 (v2 run diversity-first-10B-1.5B-seed42, step 14305); its expected
scores, LightEval hashes and weight sha256s are in reference/expected_diversity_oriented_seed42_ep3.json.

  python -m kys_eval.reference_check --model /path/to/diversity-first-10B-1.5B-seed42/14305
  python -m kys_eval.reference_check --model hf://<org>/<repo>/<subfolder>
  python -m kys_eval.reference_check --compare-only results/reference_check/results.json

Checks, in order:
  1. weights: model.safetensors sha256 equals the recorded one (skipped with --skip-weight-hash)
  2. hashes: for all 64 tasks, LightEval's hash_examples / hash_full_prompts (data + prompts) and
     hash_input_tokens / hash_cont_tokens (tokenization) equal the recorded ones
  3. scores: per-task acc_norm and acc within --tol-task, Mean6 and MMLU within --tol-aggregate
On an H100 an exact match is expected. Writes reference_check.md / .json next to results.json.
Exit status: 0 PASS, 1 FAIL.
"""

import argparse
import json
import sys
from pathlib import Path

from kys_eval import common
from kys_eval import config as C
from kys_eval.common import EvalError
from kys_eval.eval_checkpoint import add_eval_flags, evaluate

EXACT = 1e-12
AGGREGATES = [("mean6_acc_norm", "mean6", "Mean6"), ("mmlu_macro_acc_norm", "mmlu", "MMLU macro")]


def load_expected(path=C.REFERENCE_EXPECTED_JSON):
    return json.loads(Path(path).read_text())


def results_from_any(path):
    """A kys-eval results.json, or a raw LightEval results_*.json packaged on the fly."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") == common.RESULTS_SCHEMA:
        return data
    if "results" in data and "config_general" in data:
        return common.build_results(
            data, label=f"packaged {path}", cell=None, model={"spec": str(path)},
            environment={"gpu": ""}, command=None, started_at=None, finished_at=None,
            wall_seconds=None, raw_path=str(path),
        )
    raise EvalError(f"{path} is neither a kys-eval results.json nor a LightEval results_*.json")


def compare(results, expected, tol_task=C.REFERENCE_TOL_TASK, tol_aggregate=C.REFERENCE_TOL_AGGREGATE):
    got_tasks = results["tasks"]
    missing, hash_mismatches, task_rows = [], [], []
    for task, exp in expected["tasks"].items():
        got = got_tasks.get(task)
        if got is None:
            missing.append(task)
            continue
        for field in common.HASH_FIELDS:
            want, have = exp.get("hashes", {}).get(field), got.get("hashes", {}).get(field)
            if want is not None and have != want:
                hash_mismatches.append({"task": task, "field": field, "expected": want, "got": have})
        for metric in ("acc_norm", "acc"):
            if metric in exp:
                have = got.get(metric)
                task_rows.append({
                    "task": task, "metric": metric, "expected": exp[metric], "got": have,
                    "diff": None if have is None else have - exp[metric],
                })

    aggregate_rows = []
    for exp_key, score_key, label in AGGREGATES:
        want, have = expected["aggregates"][exp_key], results["scores"].get(score_key)
        aggregate_rows.append({"name": label, "expected": want, "got": have, "diff": None if have is None else have - want})

    task_failures = [r for r in task_rows if r["diff"] is None or abs(r["diff"]) > tol_task]
    aggregate_failures = [r for r in aggregate_rows if r["diff"] is None or abs(r["diff"]) > tol_aggregate]

    expected_weights = {n: h for n, h in expected["checkpoint"]["files_sha256"].items() if n.endswith(".safetensors")}
    got_weights = results.get("model", {}).get("weights_sha256")
    if got_weights is None:
        weights = {"status": "not checked", "detail": "results carry no weight hashes"}
    elif all(got_weights.get(n) == h for n, h in expected_weights.items()):
        weights = {"status": "match", "detail": expected_weights}
    else:
        weights = {"status": "MISMATCH", "detail": {"expected": expected_weights, "got": got_weights}}

    diffs = [r["diff"] for r in task_rows + aggregate_rows]
    exact = all(d is not None and abs(d) <= EXACT for d in diffs)
    passed = (
        not missing and not hash_mismatches and not task_failures and not aggregate_failures
        and weights["status"] != "MISMATCH" and results.get("complete", False)
    )
    verdict = ("PASS (exact match)" if exact else "PASS (within tolerance)") if passed else "FAIL"
    return {
        "verdict": verdict,
        "passed": passed,
        "tolerances": {"task": tol_task, "aggregate": tol_aggregate},
        "gpu": results.get("environment", {}).get("gpu"),
        "expected_gpu": expected["recorded"]["gpu"],
        "weights": weights,
        "missing_tasks": missing,
        "hash_mismatches": hash_mismatches,
        "task_failures": task_failures,
        "aggregates": aggregate_rows,
        "n_task_metrics": len(task_rows),
        "n_exact": sum(1 for r in task_rows if r["diff"] is not None and abs(r["diff"]) <= EXACT),
        "max_abs_task_diff": max((abs(r["diff"]) for r in task_rows if r["diff"] is not None), default=None),
        "task_rows": task_rows,
    }


def write_report_md(report, expected, path):
    ck = expected["checkpoint"]
    lines = [
        "# Reference check", "",
        f"**{report['verdict']}**", "",
        f"Checkpoint: `{ck['setting']}` seed {ck['seed']} {ck['epoch']} (step {ck['step']}, v2 run `{ck['v2_run']}`)", "",
        "| Check | Result |", "|---|---|",
        f"| Weights sha256 | {report['weights']['status']} |",
        f"| Missing tasks | {len(report['missing_tasks'])} |",
        f"| LightEval hash mismatches (data, prompts, tokens) | {len(report['hash_mismatches'])} |",
        f"| Task metrics exactly equal | {report['n_exact']} / {report['n_task_metrics']} |",
        f"| Max abs task difference | {report['max_abs_task_diff']} (tolerance {report['tolerances']['task']}) |",
        f"| Task metrics beyond tolerance | {len(report['task_failures'])} |",
        f"| GPU | {report['gpu'] or 'unknown'} (reference: {report['expected_gpu']}) |",
        "", "## Aggregates (acc_norm, %)", "", "| | Expected | Got | Diff (points) |", "|---|---|---|---|",
    ]
    for row in report["aggregates"]:
        diff = "—" if row["diff"] is None else f"{row['diff'] * 100:+.4f}"
        lines.append(f"| {row['name']} | {common.pct(row['expected'])} | {common.pct(row['got'])} | {diff} |")
    lines += ["", "## Commonsense tasks (acc_norm, %)", "", "| Task | Expected | Got | Diff (points) |", "|---|---|---|---|"]
    by_task = {(r["task"], r["metric"]): r for r in report["task_rows"]}
    for tid, _, label, *_ in C.COMMONSENSE:
        row = by_task.get((tid, "acc_norm"))
        if row:
            diff = "—" if row["diff"] is None else f"{row['diff'] * 100:+.4f}"
            lines.append(f"| {label} | {common.pct(row['expected'])} | {common.pct(row['got'])} | {diff} |")
    if report["hash_mismatches"]:
        lines += ["", "## Hash mismatches", "", "| Task | Field | Expected | Got |", "|---|---|---|---|"]
        lines += [f"| `{m['task']}` | {m['field']} | `{m['expected']}` | `{m['got']}` |" for m in report["hash_mismatches"]]
        lines += ["", "hash_examples / hash_full_prompts differ: dataset contents or prompts differ. "
                  "hash_input_tokens / hash_cont_tokens differ: the tokenizer or tokenization differs."]
    if report["task_failures"]:
        lines += ["", "## Task metrics beyond tolerance", "", "| Task | Metric | Expected | Got | Diff |", "|---|---|---|---|---|"]
        lines += [f"| `{r['task']}` | {r['metric']} | {r['expected']} | {r['got']} | {r['diff']} |" for r in report["task_failures"]]
    if report["missing_tasks"]:
        lines += ["", "## Missing tasks", ""] + [f"- `{t}`" for t in report["missing_tasks"]]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="the reference checkpoint: local HF dir or hf://org/repo/subfolder[@rev]")
    source.add_argument("--compare-only", metavar="RESULTS_JSON", help="compare an existing results.json or LightEval results_*.json")
    parser.add_argument("--out-dir", default=str(C.RESULTS_ROOT / "reference_check"), help="default: %(default)s")
    parser.add_argument("--skip-weight-hash", action="store_true", help="evaluate even if the weights are not the recorded checkpoint")
    parser.add_argument("--tol-task", type=float, default=C.REFERENCE_TOL_TASK)
    parser.add_argument("--tol-aggregate", type=float, default=C.REFERENCE_TOL_AGGREGATE)
    add_eval_flags(parser)
    args = parser.parse_args(argv)

    expected = load_expected()
    out_dir = Path(args.out_dir)
    try:
        if args.compare_only:
            results = results_from_any(args.compare_only)
            out_dir = Path(args.compare_only).resolve().parent if args.out_dir == parser.get_default("out_dir") else out_dir
        else:
            weights = None if args.skip_weight_hash else {
                n: h for n, h in expected["checkpoint"]["files_sha256"].items() if n.endswith(".safetensors")
            }
            results = evaluate(
                args.model, out_dir,
                label="reference check: diversity_oriented seed42 ep3 (v2)",
                model_cache=args.model_cache, allow_any_gpu=args.allow_any_gpu,
                allow_env_mismatch=args.allow_env_mismatch, delete_weights=args.delete_weights,
                force=args.force, expected_weights_sha256=weights,
            )
    except EvalError as exc:
        print(f"[kys] reference check FAILED before comparison: {exc}", file=sys.stderr)
        return 1

    report = compare(results, expected, args.tol_task, args.tol_aggregate)
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(out_dir / "reference_check.json", report)
    write_report_md(report, expected, out_dir / "reference_check.md")
    agg = {r["name"]: r for r in report["aggregates"]}
    print(f"[kys] reference check: {report['verdict']}")
    print(f"[kys]   Mean6 {common.pct(agg['Mean6']['got'])} (expected {common.pct(agg['Mean6']['expected'])}), "
          f"MMLU {common.pct(agg['MMLU macro']['got'])} (expected {common.pct(agg['MMLU macro']['expected'])})")
    print(f"[kys]   {report['n_exact']}/{report['n_task_metrics']} task metrics exact, "
          f"{len(report['hash_mismatches'])} hash mismatches, weights {report['weights']['status']}")
    print(f"[kys]   report: {out_dir / 'reference_check.md'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
