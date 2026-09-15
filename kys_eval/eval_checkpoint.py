"""Evaluate one HF checkpoint with the pinned v2 protocol.

Writes <out-dir>/results.json (scores, hashes, provenance) and <out-dir>/summary.md, plus the raw
LightEval output under <out-dir>/lighteval_raw/ and the LightEval log at <out-dir>/lighteval.log.

  python -m kys_eval.eval_checkpoint --model /path/to/hf_checkpoint --out-dir results/adhoc/NAME
  python -m kys_eval.eval_checkpoint \\
      --model hf://blab-jhu/KYS-1.5B-Raw-Selected-Baselines/rewrite-1p5b/seed42/raw_random/ep1/hf \\
      --out-dir results/adhoc/raw_random_seed42_ep1

--model is a local HF checkpoint directory or hf://<org>/<repo>[/<subfolder>][@<revision>].
If <out-dir>/results.json is already complete the checkpoint is skipped (use --force to redo).
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


def is_complete(out_dir):
    results = common.load_results(Path(out_dir) / "results.json")
    return bool(results and results.get("complete"))


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
    if not force and is_complete(out_dir):
        print(f"[kys] SKIP {label}: complete results already in {out_dir}")
        return common.load_results(out_dir / "results.json")
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
