"""Verify the environment matches the pinned v2 setup (no evaluation, no GPU required).

  python -m kys_eval.check_install

Checks Python and package versions, the patched LightEval tree (sha256 of the 4 patched files), the
verbatim task file, that the --custom-tasks module loads all 63 tasks with 0-shot, token-normalized
acc_norm and pinned dataset revisions, and reports the visible GPU. Exit status: 0 OK, 1 problems.
"""

import importlib.util
import sys

from kys_eval import common
from kys_eval import config as C


def check_tasks():
    problems = []
    spec = importlib.util.spec_from_file_location("kys_tasks_check", C.TASK_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    by_name = {t.name: t for t in module.TASKS_TABLE}
    for task_id in C.EXPECTED_TASKS:
        cfg = by_name.get(task_id)
        if cfg is None:
            problems.append(f"{task_id}: not defined by {C.TASK_FILE.name}")
            continue
        if cfg.hf_revision != C.DATASET_REVISIONS.get(cfg.hf_repo):
            problems.append(f"{task_id}: dataset revision {cfg.hf_revision} not pinned")
        if cfg.num_fewshots != C.NUM_FEWSHOT:
            problems.append(f"{task_id}: num_fewshots {cfg.num_fewshots}")
        names = [m.metric_name for m in cfg.metrics]
        norms = [type(getattr(m.sample_level_fn, "logprob_normalization", None)).__name__ for m in cfg.metrics]
        if names != ["acc", "acc_norm"] or norms != ["NoneType", "LogProbTokenNorm"]:
            problems.append(f"{task_id}: metrics {names} / {norms}")
    for tid, _, _, repo, subset, split, _ in C.COMMONSENSE:
        cfg = by_name.get(tid)
        if cfg and (cfg.hf_repo, cfg.hf_subset, tuple(cfg.evaluation_splits)) != (repo, subset, (split,)):
            problems.append(f"{tid}: dataset {cfg.hf_repo}/{cfg.hf_subset}/{cfg.evaluation_splits} != config")
    return problems


def main():
    env = common.environment_info()
    print(f"python          {env['python']}  (pinned {C.PYTHON_VERSION})")
    for name, version in env["packages"].items():
        mark = "ok" if version == C.PINNED_VERSIONS[name] else f"MISMATCH (pinned {C.PINNED_VERSIONS[name]})"
        print(f"{name:<15} {version}  {mark}")
    print(f"lighteval root  {env['lighteval_root']}")
    for rel, ok in env["lighteval_patched_files_ok"].items():
        print(f"  {'ok      ' if ok else 'MISMATCH'} {rel}")
    print(f"task file       {'ok' if env['task_file_sha256_ok'] else 'MODIFIED'}  {C.TASK_FILE_VERBATIM}")

    problems = common.environment_problems(env)
    if env["lighteval_root"] is not None:
        try:
            task_problems = check_tasks()
        except Exception as exc:  # import errors inside lighteval or the task file
            task_problems = [f"could not load {C.TASK_FILE}: {exc!r}"]
        print(f"tasks           {len(C.EXPECTED_TASKS) - len(task_problems)}/{len(C.EXPECTED_TASKS)} ok")
        problems += task_problems

    gpu = common.gpu_name()
    gpu_note = "" if C.REQUIRED_GPU_SUBSTRING in gpu else f"  (evaluation needs an H100; v2 used {C.REFERENCE_GPU})"
    print(f"gpu             {gpu or 'none visible'}{gpu_note}")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK: environment matches the pinned v2 setup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
