"""Evaluate the raw-selected grid (or part of it) from the HF model repo.

Cells are ordered (seed, setting, epoch); with no filters the indices are 0..35. Filters keep that
order, so `--list` with the same filters shows the indices a SLURM array should use.

  python -m kys_eval.run_grid --list --check-hub          # show cells, their status, and hub availability
  python -m kys_eval.run_grid                             # evaluate every pending cell, one after another
  python -m kys_eval.run_grid --seeds 42                  # one seed (12 cells)
  python -m kys_eval.run_grid --index 7                   # one cell (for SLURM arrays)

Results go to <results-root>/raw_selected/seed<S>/<setting>/ep<N>/{results.json,summary.md}.
Cells with complete results are skipped, so re-running is safe. Exit status: 0 if every selected
cell is complete, 1 if any failed, 2 on bad arguments.
"""

import argparse
import sys
from pathlib import Path

from kys_eval import common
from kys_eval import config as C
from kys_eval.common import EvalError
from kys_eval.eval_checkpoint import add_eval_flags, evaluate, is_complete


def selected_cells(seeds, settings, epochs):
    return [c for c in C.grid_cells() if c["seed"] in seeds and c["setting"] in settings and c["epoch"] in epochs]


def cell_dir(results_root, cell):
    return Path(results_root) / "raw_selected" / f"seed{cell['seed']}" / cell["setting"] / cell["epoch"]


def cell_spec(cell, hf_repo, hf_revision):
    return f"hf://{hf_repo}/{cell['hf_subfolder']}" + (f"@{hf_revision}" if hf_revision else "")


def cell_label(cell):
    return f"{cell['setting']} seed{cell['seed']} {cell['epoch']} (step {cell['step']})"


def hub_checkpoints(hf_repo, hf_revision):
    from huggingface_hub import HfApi

    files = set(HfApi().list_repo_files(hf_repo, revision=hf_revision))
    return {f.rsplit("/", 1)[0] for f in files if f.endswith("/config.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default=str(C.RESULTS_ROOT), help="default: %(default)s")
    parser.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS, choices=C.SEEDS)
    parser.add_argument("--settings", nargs="+", default=C.RAW_SETTINGS, choices=C.RAW_SETTINGS)
    parser.add_argument("--epochs", nargs="+", default=C.EPOCHS, choices=C.EPOCHS)
    parser.add_argument("--hf-repo", default=C.HF_REPO, help="default: %(default)s")
    parser.add_argument("--hf-revision", help="HF repo branch/tag/commit (default: main at download time)")
    parser.add_argument("--list", action="store_true", help="list the selected cells and exit")
    parser.add_argument("--check-hub", action="store_true", help="with --list: show whether each checkpoint is on the hub")
    parser.add_argument("--index", type=int, help="evaluate only the cell at this index of the selected list")
    parser.add_argument("--stop-on-error", action="store_true", help="stop at the first failed cell")
    add_eval_flags(parser)
    args = parser.parse_args(argv)

    cells = selected_cells(args.seeds, args.settings, args.epochs)
    if args.list:
        on_hub = hub_checkpoints(args.hf_repo, args.hf_revision) if args.check_hub else None
        print(f"{'idx':>3}  {'status':<8}  {'hub':<7}  cell")
        for i, cell in enumerate(cells):
            status = "done" if is_complete(cell_dir(args.results_root, cell)) else "pending"
            hub = "-" if on_hub is None else ("present" if cell["hf_subfolder"] in on_hub else "MISSING")
            print(f"{i:>3}  {status:<8}  {hub:<7}  {cell_label(cell):<45} {cell['hf_subfolder']}")
        print(f"{len(cells)} cells; SLURM: --array=0-{len(cells) - 1}")
        return 0

    if args.index is not None:
        if not 0 <= args.index < len(cells):
            print(f"[kys] --index {args.index} out of range for {len(cells)} selected cells", file=sys.stderr)
            return 2
        cells = [cells[args.index]]

    done, failed = [], []
    for cell in cells:
        try:
            evaluate(
                cell_spec(cell, args.hf_repo, args.hf_revision),
                cell_dir(args.results_root, cell),
                label=cell_label(cell),
                cell={k: cell[k] for k in ("setting", "seed", "epoch", "step", "hf_subfolder")} | {"hf_repo": args.hf_repo},
                model_cache=args.model_cache,
                allow_any_gpu=args.allow_any_gpu,
                allow_env_mismatch=args.allow_env_mismatch,
                delete_weights=args.delete_weights,
                force=args.force,
            )
            done.append(cell)
        except EvalError as exc:
            print(f"[kys] FAILED {cell_label(cell)}: {exc}", file=sys.stderr)
            failed.append(cell)
            if args.stop_on_error:
                break

    print(f"[kys] grid: {len(done)} complete, {len(failed)} failed, {len(cells) - len(done) - len(failed)} not attempted")
    for cell in failed:
        print(f"[kys]   failed: {cell_label(cell)}", file=sys.stderr)
    return 1 if failed or len(done) < len(cells) else 0


if __name__ == "__main__":
    sys.exit(main())
