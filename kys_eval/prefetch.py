"""Download the evaluation datasets at their pinned revisions into HF_HOME (no GPU).

  python -m kys_eval.prefetch

Run once, from a node with internet access, before starting SLURM arrays: parallel jobs then read a
warm cache instead of racing to download and prepare the same datasets. Loads each (repo, subset)
exactly as LightEval does (all splits, pinned revision).
"""

import importlib.util
import sys

from kys_eval import config as C


def main():
    import datasets

    spec = importlib.util.spec_from_file_location("kys_tasks_prefetch", C.TASK_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wanted = set(C.EXPECTED_TASKS)
    configs = sorted({(t.hf_repo, t.hf_subset, t.hf_revision) for t in module.TASKS_TABLE if t.name in wanted})
    for i, (repo, subset, revision) in enumerate(configs, 1):
        dataset = datasets.load_dataset(path=repo, name=subset, revision=revision)
        sizes = ", ".join(f"{split}={len(ds)}" for split, ds in dataset.items())
        print(f"[{i}/{len(configs)}] {repo} {subset} @ {revision[:7]}: {sizes}", flush=True)
    print(f"prefetched {len(configs)} dataset configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
