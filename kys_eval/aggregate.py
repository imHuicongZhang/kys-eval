"""Aggregate raw-selected results into markdown reports (no GPU, no network).

  python -m kys_eval.aggregate [--results-root results] [--out-dir reports]

Reads <results-root>/raw_selected/seed*/*/ep*/results.json (only complete ones) and the v2 rewritten
results in reference/rewritten_v2_54_per_task.csv, and writes into --out-dir:

  raw_selected_report.md           summary table of average accuracy (Mean6), then one table per
                                   benchmark: rows = setting, columns = ep1/ep2/ep3,
                                   cells = mean ± std over seeds
  raw_vs_rewritten.md              each raw setting next to its v2 comparator, with deltas: the rewritten
                                   counterpart for the four strategy-linked controls, the fastText
                                   quality_base arm for the three global Top-10B selections
  raw_selected_per_checkpoint.csv  one row per evaluated checkpoint
  raw_selected_by_epoch.csv        setting x epoch x benchmark: mean, std, n, seeds
  raw_vs_rewritten.csv             pair x epoch x benchmark: raw and comparator stats, paired delta
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from kys_eval import common
from kys_eval import config as C

COMMONSENSE_BENCHMARKS = [(short, label) for _, short, label, *_ in C.COMMONSENSE]
CATEGORY_BENCHMARKS = [(common.category_key(c), f"MMLU {c}") for c in C.MMLU_CATEGORIES]
ALL_BENCHMARKS = [("mean6", "Mean6"), ("mmlu", "MMLU")] + COMMONSENSE_BENCHMARKS + CATEGORY_BENCHMARKS


def load_raw(results_root):
    cells, problems = {}, []
    for path in sorted(Path(results_root).glob("raw_selected/seed*/*/ep*/results.json")):
        results = common.load_results(path)
        if results is None or not results.get("cell"):
            problems.append(f"`{path}`: not a kys-eval grid results file")
            continue
        if not results.get("complete"):
            problems.append(f"`{path}`: incomplete ({len(results.get('missing_tasks', []))} tasks missing)")
            continue
        cell = results["cell"]
        acc_norm = {t: e["acc_norm"] for t, e in results["tasks"].items() if "acc_norm" in e}
        cells[(cell["setting"], int(cell["seed"]), cell["epoch"])] = {
            **common.benchmark_scores(acc_norm),
            "step": cell["step"],
            "gpu": results.get("environment", {}).get("gpu", ""),
        }
    return cells, problems


def load_rewritten(path=C.REWRITTEN_V2_PER_TASK_CSV):
    acc_norm, meta = defaultdict(dict), {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["setting"], int(row["seed"]), row["epoch"])
            acc_norm[key][row["task"]] = float(row["acc_norm"])
            meta[key] = (int(row["step"]), row["gpu"])
    return {k: {**common.benchmark_scores(v), "step": meta[k][0], "gpu": meta[k][1]} for k, v in acc_norm.items()}


def seed_values(cells, setting, epoch, benchmark):
    values = {}
    for seed in C.SEEDS:
        cell = cells.get((setting, seed, epoch))
        if cell and cell.get(benchmark) is not None:
            values[seed] = cell[benchmark]
    return values


def summarize(values):
    values = list(values)
    if not values:
        return None, None, 0
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None), len(values)


def fmt_stat(values, signed=False):
    mean, std, n = summarize(values)
    if n == 0:
        return "—"
    text = common.pct(mean, signed)
    if std is not None:
        text += f" ± {std * 100:.2f}"
    return text if n == len(C.SEEDS) else f"{text} (n={n})"


def epoch_table(cells, benchmark):
    lines = [
        "| Setting | " + " | ".join(f"{e} (step {C.EPOCH_STEPS[e]})" for e in C.EPOCHS) + " |",
        "|---|" + "---|" * len(C.EPOCHS),
    ]
    for setting in C.RAW_SETTINGS:
        row = [fmt_stat(seed_values(cells, setting, e, benchmark).values()) for e in C.EPOCHS]
        lines.append(f"| `{setting}` | " + " | ".join(row) + " |")
    return lines


def gpu_notes(cells):
    gpus = sorted({c["gpu"] or "unknown" for c in cells.values()})
    if not gpus:
        return []
    if gpus == [C.REFERENCE_GPU]:
        return [f"All checkpoints were scored on {C.REFERENCE_GPU}, the device used for the v2 rewritten results.", ""]
    note = f"GPUs used: {', '.join(gpus)}. The v2 rewritten results were all scored on {C.REFERENCE_GPU}."
    if any(C.REQUIRED_GPU_SUBSTRING not in g for g in gpus):
        note += (" **WARNING:** some checkpoints ran on a non-H100 GPU; bf16 near-ties differ across architectures "
                 "(~7e-4 on Mean6), so compare those cells with care.")
    return [note, ""]


def write_raw_report(raw, problems, results_root, out_dir):
    expected = C.grid_cells()
    missing = [c for c in expected if (c["setting"], c["seed"], c["epoch"]) not in raw]
    lines = [
        "# Raw-selected baselines: results", "",
        f"Generated {common.utcnow()} from `{results_root}`.", "",
        "Values are `acc_norm` in percent: 0-shot, full evaluation splits, token-length-normalized "
        "log-likelihood (protocol in `kys_eval/config.py`). Each cell is the mean ± sample standard "
        f"deviation over seeds {', '.join(map(str, C.SEEDS))}; `(n=k)` marks cells with fewer seeds.", "",
        f"Coverage: **{len(expected) - len(missing)} / {len(expected)}** checkpoints.", "",
    ]
    lines += gpu_notes(raw)
    if missing:
        lines += ["<details><summary>Checkpoints without complete results</summary>", ""]
        lines += [f"- `{c['setting']}` seed {c['seed']} {c['epoch']}" for c in missing]
        lines += ["", "</details>", ""]
    if problems:
        lines += ["Result files skipped:", ""] + [f"- {p}" for p in problems] + [""]

    lines += [
        "## Summary: average accuracy (Mean6)", "",
        "Mean6 is the unweighted mean of ARC-Easy, HellaSwag, PIQA, SIQA, OpenBookQA and CommonsenseQA, "
        "the paper's headline commonsense average (Fig. 3a).", "",
    ]
    lines += epoch_table(raw, "mean6") + [""]
    lines += ["## MMLU", "", "Unweighted macro-average over the 57 subjects (Fig. 3b).", ""]
    lines += epoch_table(raw, "mmlu") + [""]
    for key, label in COMMONSENSE_BENCHMARKS:
        lines += [f"## {label}", ""] + epoch_table(raw, key) + [""]
    lines += ["## MMLU by category", "", "Mean over each category's subjects (Table 3 breakdown).", ""]
    for key, label in CATEGORY_BENCHMARKS:
        lines += [f"### {label}", ""] + epoch_table(raw, key) + [""]
    (out_dir / "raw_selected_report.md").write_text("\n".join(lines))

    with open(out_dir / "raw_selected_per_checkpoint.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "seed", "epoch", "step"] + [k for k, _ in ALL_BENCHMARKS] + ["gpu"])
        order = {s: i for i, s in enumerate(C.RAW_SETTINGS)}
        for (setting, seed, epoch), cell in sorted(raw.items(), key=lambda kv: (kv[0][1], order.get(kv[0][0], len(order)), kv[0][0], kv[0][2])):
            writer.writerow([setting, seed, epoch, cell["step"]] + [f"{cell[k]:.6f}" for k, _ in ALL_BENCHMARKS] + [cell["gpu"]])

    with open(out_dir / "raw_selected_by_epoch.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "epoch", "benchmark", "mean", "std", "n_seeds", "seeds"])
        for setting in C.RAW_SETTINGS:
            for epoch in C.EPOCHS:
                for key, _ in ALL_BENCHMARKS:
                    values = seed_values(raw, setting, epoch, key)
                    mean, std, n = summarize(values.values())
                    if n:
                        writer.writerow([setting, epoch, key, f"{mean:.6f}", "" if std is None else f"{std:.6f}", n,
                                         "|".join(map(str, values))])
    return missing


def write_pair_report(raw, rewritten, out_dir):
    lines = [
        "# Raw-selected vs v2 comparators", "",
        "Each raw-selected setting next to its v2 comparator, per epoch. Columns are `acc_norm` in percent, "
        "mean ± std over seeds.", "",
        "Two kinds of pairs. The four **strategy-linked** controls are compared with their rewritten counterpart. "
        "The three **global Top-10B** selections are compared with the existing fastText Quality-Base arm: that "
        "is a comparison between selection scores on raw data, not a rewriting comparison.", "",
        "**Δ = comparator − raw**, in accuracy points: positive means the comparator scores higher. Δ is the mean "
        "of the per-seed differences (the two arms share each seed's initialization), ± the standard deviation of "
        "those differences; *Seeds Δ>0* counts seeds with a positive difference.", "",
        "Comparator side: the v2 evaluation of the 54 rewritten-grid checkpoints (`reference/rewritten_v2_54_per_task.csv`, "
        f"all scored on {C.REFERENCE_GPU}). These are not the paper's printed table values.", "",
        "| Raw-selected | Kind | Comparator |", "|---|---|---|",
    ]
    lines += [f"| `{r}` | {'strategy-linked' if r in C.RAW_TO_REWRITTEN else 'global Top-10B'} | {C.PAPER_LABEL[w]} (`{w}`) |"
              for r, w in C.RAW_COMPARATOR.items()] + [""]
    lines += gpu_notes(raw)

    rows = []
    sections = [("mean6", "Mean6 (commonsense average)"), ("mmlu", "MMLU (57-subject macro-average)")]
    sections += COMMONSENSE_BENCHMARKS + CATEGORY_BENCHMARKS
    for key, label in sections:
        lines += [f"## {label}", "", "| Raw setting | Comparator | Epoch | Raw | Comparator | Δ (comparator − raw) | Seeds Δ>0 |",
                  "|---|---|---|---|---|---|---|"]
        for raw_setting, rewritten_setting in C.RAW_COMPARATOR.items():
            for epoch in C.EPOCHS:
                rv = seed_values(raw, raw_setting, epoch, key)
                wv = seed_values(rewritten, rewritten_setting, epoch, key)
                paired = sorted(set(rv) & set(wv))
                deltas = [wv[s] - rv[s] for s in paired]
                d_mean, d_std, d_n = summarize(deltas)
                if d_n:
                    delta = common.pct(d_mean, signed=True) + ("" if d_std is None else f" ± {d_std * 100:.2f}")
                    wins = f"{sum(d > 0 for d in deltas)}/{d_n}"
                else:
                    delta = wins = "—"
                lines.append(f"| `{raw_setting}` | {C.PAPER_LABEL[rewritten_setting]} | {epoch} | {fmt_stat(rv.values())} | "
                             f"{fmt_stat(wv.values())} | {delta} | {wins} |")
                r_mean, r_std, r_n = summarize(rv.values())
                w_mean, w_std, w_n = summarize(wv.values())
                rows.append([raw_setting, rewritten_setting, epoch, key,
                             *(("" if x is None else f"{x:.6f}") for x in (r_mean, r_std)), r_n,
                             *(("" if x is None else f"{x:.6f}") for x in (w_mean, w_std)), w_n,
                             *(("" if x is None else f"{x:.6f}") for x in (d_mean, d_std)), d_n,
                             sum(d > 0 for d in deltas), "|".join(map(str, paired))])
        lines.append("")
    (out_dir / "raw_vs_rewritten.md").write_text("\n".join(lines))

    with open(out_dir / "raw_vs_rewritten.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # column names kept from the four-setting version; "rewritten_*" = the comparator arm
        writer.writerow(["raw_setting", "rewritten_setting", "epoch", "benchmark",
                         "raw_mean", "raw_std", "raw_n", "rewritten_mean", "rewritten_std", "rewritten_n",
                         "delta_mean", "delta_std", "n_paired", "n_delta_positive", "paired_seeds"])
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default=str(C.RESULTS_ROOT), help="default: %(default)s")
    parser.add_argument("--out-dir", default=str(C.REPORTS_DIR), help="default: %(default)s")
    parser.add_argument("--rewritten-csv", default=str(C.REWRITTEN_V2_PER_TASK_CSV), help="default: %(default)s")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, problems = load_raw(args.results_root)
    rewritten = load_rewritten(args.rewritten_csv)
    missing = write_raw_report(raw, problems, args.results_root, out_dir)
    write_pair_report(raw, rewritten, out_dir)
    print(f"[kys] aggregated {len(raw)}/{len(C.grid_cells())} raw-selected checkpoints ({len(problems)} files skipped)")
    for name in ("raw_selected_report.md", "raw_vs_rewritten.md", "raw_selected_per_checkpoint.csv",
                 "raw_selected_by_epoch.csv", "raw_vs_rewritten.csv"):
        print(f"[kys]   {out_dir / name}")
    if missing:
        print(f"[kys] {len(missing)} checkpoints have no complete results yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
