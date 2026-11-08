"""Evaluate one HF checkpoint with the pinned v2 protocol.

Writes <out-dir>/results.json (scores, hashes, provenance) and <out-dir>/summary.md, plus the raw
LightEval output under <out-dir>/lighteval_raw/ and the LightEval log at <out-dir>/lighteval.log.

  python -m kys_eval.eval_checkpoint --model /path/to/hf_checkpoint --out-dir results/adhoc/NAME
  python -m kys_eval.eval_checkpoint \\
      --model hf://blab-jhu/KYS-1.5B-Raw-Selected-Baselines/rewrite-1p5b/seed42/raw_random/ep1/hf \\
      --out-dir results/adhoc/raw_random_seed42_ep1

--model is a local HF checkpoint directory or hf://<org>/<repo>[/<subfolder>][@<revision>].
If <out-dir>/results.json is already complete AND belongs to this checkpoint (same model path, and for grid
cells the same setting, seed, epoch, step and HF subfolder) the checkpoint is skipped (use --force to redo).
Complete results for a different checkpoint are reported and re-evaluated, never reused. A checkpoint
re-uploaded to the same path and revision cannot be told apart this way: use --force for it.
Exit status: 0 complete, 1 failed or incomplete.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from kys_eval import common
from kys_eval import config as C
from kys_eval.common import EvalError


CELL_KEYS = ("setting", "seed", "epoch", "step", "hf_subfolder", "hf_repo")


def _spec_identity(spec):
    """(kind, model path, requested revision) of a --model spec; None if it cannot be parsed."""
    try:
        p = common.parse_model_spec(spec)
    except Exception:  # noqa: BLE001 - an unparsable stored spec simply does not match
        return None
    if p["kind"] == "local":
        return ("local", str(Path(p["path"]).expanduser().resolve()), None)
    return ("hf", f"{p['repo_id']}/{p['subfolder']}".rstrip("/"), p.get("revision"))


def identity_mismatch(results, model, cell=None):
    """Reasons why existing results do not belong to `model` (and `cell`); [] if they do (audit E-F1).

    Compared: the model path (local directory, or HF repo + subfolder); the revision, when `model` pins one
    (it must equal the stored revision or prefix the stored resolved commit); and for grid cells the stored
    cell's setting, seed, epoch, step, HF subfolder and HF repo."""
    reasons = []
    stored_model = results.get("model") or {}
    want, have = _spec_identity(model), _spec_identity(stored_model.get("spec") or "")
    if want is None or have is None:
        reasons.append(f"model spec {stored_model.get('spec')!r} is not comparable with {model!r}")
    else:
        if want[:2] != have[:2]:
            reasons.append(f"model path is {have[1]} ({have[0]}), expected {want[1]} ({want[0]})")
        rev = want[2]
        if rev and rev != stored_model.get("revision") and not str(stored_model.get("commit") or "").startswith(rev):
            reasons.append(f"model revision is {stored_model.get('revision')!r} / commit {stored_model.get('commit')!r}, "
                           f"expected {rev!r}")
    if cell is not None:
        stored_cell = results.get("cell") or {}
        for k in CELL_KEYS:
            if k in cell and stored_cell.get(k) != cell[k]:
                reasons.append(f"cell {k} is {stored_cell.get(k)!r}, expected {cell[k]!r}")
    return reasons


def reusable(out_dir, model, cell=None):
    """(results, reasons): results if complete results in out_dir belong to this checkpoint, else None.
    reasons lists the identity mismatches of complete-but-foreign results ([] otherwise)."""
    results = common.load_results(Path(out_dir) / "results.json")
    if not (results and results.get("complete")):
        return None, []
    reasons = identity_mismatch(results, model, cell)
    return (None, reasons) if reasons else (results, [])


def run_lighteval(model_dir, raw_dir, log_path):
    if "," in str(model_dir):
        raise EvalError(f"model path {model_dir} contains a comma, which LightEval model args cannot express")
    cmd = [
        sys.executable, "-m", "lighteval", C.LIGHTEVAL_BACKEND,
        C.MODEL_ARGS.format(path=model_dir),
        C.TASKS_ARG,
        "--custom-tasks", str(C.TASK_FILE),
        "--output-dir", str(raw_dir),
    ]
    env = {**os.environ, **C.EVAL_ENV}
    with open(log_path, "w") as log:
        log.write("$ " + shlex.join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, env=env, cwd=raw_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return cmd, proc.wait()


def evaluate(
    model,
    out_dir,
    *,
    label=None,
    cell=None,
    model_cache=None,
    allow_any_gpu=False,
    allow_env_mismatch=False,
    delete_weights=False,
    force=False,
    expected_weights_sha256=None,
):
    """Evaluate one checkpoint; return the results dict. Raises EvalError on failure."""
    out_dir = Path(out_dir)
    label = label or model
    if not force:
        existing, reasons = reusable(out_dir, model, cell)
        if existing is not None:
            print(f"[kys] SKIP {label}: complete results for this checkpoint already in {out_dir}")
            return existing
        if reasons:
            print(f"[kys] RERUN {label}: the complete results in {out_dir} belong to another checkpoint "
                  f"({'; '.join(reasons)}); re-evaluating and overwriting them")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = common.environment_info()
    problems = common.environment_problems(env)
    if problems:
        message = "environment does not match the pinned v2 setup:\n  - " + "\n  - ".join(problems)
        if not allow_env_mismatch:
            raise EvalError(message + "\nRun ./install.sh and `python -m kys_eval.check_install`, or pass --allow-env-mismatch.")
        print(f"[kys] WARNING {message}")
    env["gpu"] = common.check_gpu(allow_any_gpu)

    started_at, t0 = common.utcnow(), time.time()
    print(f"[kys] {label}: resolving {model}")
    model_dir, provenance, downloaded = common.resolve_model(model, model_cache or C.MODEL_CACHE)
    try:
        problems, warnings = common.validate_model_dir(model_dir)
        for warning in warnings:
            print(f"[kys] WARNING {warning}")
        if problems:
            raise EvalError(f"{model_dir} is not a usable HF checkpoint: " + "; ".join(problems))
        print(f"[kys] {label}: hashing weights in {model_dir}")
        provenance["local_path"] = str(model_dir)
        provenance["weights_sha256"] = common.weights_sha256(model_dir)
        if expected_weights_sha256:
            wrong = {n: h for n, h in expected_weights_sha256.items() if provenance["weights_sha256"].get(n) != h}
            if wrong:
                raise EvalError(
                    f"weights in {model_dir} are not the expected checkpoint: expected {wrong}, "
                    f"found {provenance['weights_sha256']}"
                )

        raw_dir = out_dir / "lighteval_raw"
        shutil.rmtree(raw_dir, ignore_errors=True)  # partial output of an interrupted run
        raw_dir.mkdir()
        print(f"[kys] {label}: running LightEval on {env['gpu'] or 'unknown GPU'} (v2: ~{C.V2_SECONDS_PER_CHECKPOINT} s on one H100)")
        cmd, returncode = run_lighteval(model_dir, raw_dir, out_dir / "lighteval.log")
        raw_path = common.find_lighteval_results(raw_dir)
        if returncode != 0 or raw_path is None:
            raise EvalError(f"LightEval exited with status {returncode}; see {out_dir / 'lighteval.log'}")
        results = common.build_results(
            json.loads(raw_path.read_text()),
            label=label,
            cell=cell,
            model=provenance,
            environment=env,
            command=cmd,
            started_at=started_at,
            finished_at=common.utcnow(),
            wall_seconds=round(time.time() - t0, 1),
            raw_path=str(raw_path.relative_to(out_dir)),
        )
    finally:
        if downloaded and delete_weights:
            shutil.rmtree(model_dir, ignore_errors=True)

    common.write_json(out_dir / "results.json", results)
    common.write_summary_md(results, out_dir / "summary.md")
    if not results["complete"]:
        raise EvalError(
            f"results in {out_dir} are incomplete: missing {results['missing_tasks']}, "
            f"violations {results['protocol_violations']}"
        )
    s = results["scores"]
    print(f"[kys] DONE {label}: Mean6 {common.pct(s['mean6'])}  MMLU {common.pct(s['mmlu'])}  -> {out_dir}")
    return results


def package_existing(raw_json, out_dir, *, label, model, gpu, cell=None):
    """Build results.json + summary.md from an existing LightEval results file (no evaluation)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = common.build_results(
        json.loads(Path(raw_json).read_text()),
        label=label,
        cell=cell,
        model={"spec": model},
        environment={"source": f"packaged from existing LightEval file {raw_json}", "gpu": gpu},
        command=None,
        started_at=None,
        finished_at=None,
        wall_seconds=None,
        raw_path=str(raw_json),
    )
    common.write_json(out_dir / "results.json", results)
    common.write_summary_md(results, out_dir / "summary.md")
    return results


def add_eval_flags(parser):
    parser.add_argument("--model-cache", default=str(C.MODEL_CACHE), help="where HF checkpoints are downloaded (default: %(default)s)")
    parser.add_argument("--allow-any-gpu", action="store_true", help="run on a non-H100 GPU (recorded and flagged)")
    parser.add_argument("--allow-env-mismatch", action="store_true", help="run even if package versions or the LightEval patch differ from the pins")
    parser.add_argument("--delete-weights", action="store_true", help="delete downloaded HF weights after the checkpoint is scored")
    parser.add_argument("--force", action="store_true", help="re-evaluate even if complete results exist")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local HF checkpoint dir or hf://org/repo/subfolder[@rev]")
    parser.add_argument("--out-dir", required=True, help="directory for results.json, summary.md and raw output")
    parser.add_argument("--label", help="name used in summary.md (default: --model)")
    parser.add_argument("--from-lighteval-json", metavar="PATH", help="package an existing LightEval results_*.json instead of evaluating (no GPU)")
    parser.add_argument("--gpu-name", default="", help="with --from-lighteval-json: GPU the file was produced on")
    add_eval_flags(parser)
    args = parser.parse_args(argv)

    try:
        if args.from_lighteval_json:
            results = package_existing(args.from_lighteval_json, args.out_dir, label=args.label or args.model, model=args.model, gpu=args.gpu_name)
            print(f"[kys] packaged {args.from_lighteval_json} -> {args.out_dir} (complete={results['complete']})")
            return 0 if results["complete"] else 1
        evaluate(
            args.model,
            args.out_dir,
            label=args.label,
            model_cache=args.model_cache,
            allow_any_gpu=args.allow_any_gpu,
            allow_env_mismatch=args.allow_env_mismatch,
            delete_weights=args.delete_weights,
            force=args.force,
        )
    except EvalError as exc:
        print(f"[kys] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
