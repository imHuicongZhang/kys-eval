# kys-eval: evaluation of the Know-Your-Sources raw-selected baselines

This repository scores the 36 raw-selected baseline checkpoints (4 settings × seeds 42/43/44 ×
epochs 1/2/3). It uses exactly the protocol of the **v2 evaluation** of the 54 rewritten-arm
checkpoints: the same LightEval commit and patch, tasks, 0-shot setting, metric, dataset
revisions, model loading and GPU type. It then compares each raw setting with its rewritten
counterpart.

> v2 numbers are not the numbers printed in the paper's tables. The paper's tables come from the
> paper-era checkpoints (steps 4770 / 9540 / 14305, and 14147 for Diversity Oriented). v2 re-scored
> the re-released checkpoints at steps 4768 / 9537 / 14305, with differences of up to about 1 point
> per cell. v2 is the comparator by decision.

Run every command from the repository root. Evaluation needs one NVIDIA H100 per checkpoint.
Everything else (install check, self-test, aggregation) runs on CPU.

## Quick start

```bash
git clone https://github.com/imHuicongZhang/kys-eval.git && cd kys-eval
./install.sh                                # venv + pinned packages + patched LightEval; ends with check_install
source .venv/bin/activate
export HF_HOME=$PWD/hf_cache                # datasets + hub cache (~15 GB); any disk with room
python -m kys_eval.selftest                 # offline logic test, no GPU
python -m kys_eval.prefetch                 # download the 7 datasets at pinned revisions (needs internet)

# 1) reference check on a GPU node: must print PASS before scoring new models
python -m kys_eval.reference_check --model /path/to/diversity-first-10B-1.5B-seed42/14305

# 2) score the grid: all 36 checkpoints in one job ...
python -m kys_eval.run_grid --delete-weights
#    ... or as a SLURM array (edit the placeholders first, see "SLURM")
mkdir -p logs && sbatch --export=ALL,KYS_EVAL_ROOT=$PWD slurm/eval_grid_array.sbatch

# 3) reports
python -m kys_eval.aggregate                # -> reports/raw_selected_report.md, reports/raw_vs_rewritten.md
```

## Install

`./install.sh` performs these steps and is safe to re-run:
1. Create a Python **3.11** venv (default `./.venv`; override with `PYTHON=/path/to/python3.11` or `KYS_VENV=/path`).
2. `pip install -r requirements.txt` (torch 2.11.0+cu128, from the PyTorch cu128 index).
3. Clone `huggingface/lighteval` into `third_party/lighteval` (override with `KYS_LIGHTEVAL_DIR`) and check out `10b9104eb16778b174e9e240140974a039018981`.
4. Apply `patches/lighteval-10b9104-kys-v2.patch`, verify with `patches/PATCHED_FILES.sha256`, and `pip install --no-deps -e` it.
5. Run `python -m kys_eval.check_install`. It must end with `OK: environment matches the pinned v2 setup`.

`check_install` verifies:
- Python and package versions against the pins.
- The four patched LightEval files by sha256.
- The verbatim task file by sha256.
- That the `--custom-tasks` module defines all 63 tasks with 0-shot, `acc`/`acc_norm` (token normalization) and pinned dataset revisions.

It also prints the visible GPU. `eval_checkpoint`, `run_grid` and `reference_check` refuse to run if these checks fail, unless you pass `--allow-env-mismatch`.

The raw-selected HF repo is public. If a repo you point at is private, run `huggingface-cli login` or
set `HF_TOKEN` in the environment. Never write a token into this repository.

## Commands

### One checkpoint: `kys_eval.eval_checkpoint`

```bash
python -m kys_eval.eval_checkpoint --model /path/to/hf_checkpoint --out-dir results/adhoc/NAME
python -m kys_eval.eval_checkpoint \
    --model hf://blab-jhu/KYS-1.5B-Raw-Selected-Baselines/rewrite-1p5b/seed42/raw_random/ep1/hf \
    --out-dir results/adhoc/raw_random_seed42_ep1
```

- `--model` is a local HF checkpoint directory (`config.json`, `*.safetensors`, tokenizer files) or `hf://<org>/<repo>[/<subfolder>][@<revision>]`. HF checkpoints are downloaded at the resolved commit into `model_cache/` (`--model-cache`, or `KYS_MODEL_CACHE`).
- It runs `python -m lighteval accelerate "model_name=<dir>,dtype=bfloat16" <63 tasks> --custom-tasks kys_eval/tasks/kys_tasks.py` with `LIGHTEVAL_NO_CACHE=1` and `TOKENIZERS_PARALLELISM=false`.
- It writes `results.json` and `summary.md` (see "Outputs") and skips the checkpoint if `results.json` is already complete. Use `--force` to redo it.
- It refuses non-H100 GPUs unless you pass `--allow-any-gpu`. `--delete-weights` removes downloaded weights afterwards.
- Exit status: 0 complete, 1 failed or incomplete.

### The grid: `kys_eval.run_grid`

```bash
python -m kys_eval.run_grid --list --check-hub        # index, done/pending, present on hub
python -m kys_eval.run_grid                           # all pending cells, sequentially
python -m kys_eval.run_grid --seeds 43                # one seed (12 cells)
python -m kys_eval.run_grid --settings raw_random --epochs ep3
python -m kys_eval.run_grid --index 5                 # one cell of the (filtered) list
```

- Cells are ordered (seed, setting, epoch), giving indices 0–35 for the full grid. Filters keep that order, so `--list` with the same filters shows the indices an array job should use.
- Complete cells are skipped, and a failed cell does not stop the others (use `--stop-on-error` to change that).
- Flags passed through to every cell: `--delete-weights`, `--force`, `--allow-any-gpu`, `--model-cache`, `--hf-revision`, `--results-root` (default `results/`, or `KYS_RESULTS_ROOT`).
- Exit status is 0 only if every selected cell is complete.

### SLURM

`slurm/eval_grid_array.sbatch` runs one checkpoint per array task. Before submitting, replace the placeholders:

| Placeholder | Meaning |
|---|---|
| `<PARTITION>` | a partition with H100s |
| `<ACCOUNT>` | allocation account (delete the line if unused) |
| `<GPU_TYPE>` | gres name for H100, e.g. `h100`; or use `--gres=gpu:1` plus the cluster's H100 `--constraint` |
| `<TIME>` | per checkpoint: v2 took about 9 min on one H100, plus a ~3 GB download. `01:00:00` is safe |

```bash
mkdir -p logs
python -m kys_eval.prefetch                                                            # once, before any array
sbatch --export=ALL,KYS_EVAL_ROOT=$PWD slurm/eval_grid_array.sbatch                    # 36 tasks (0-35)
sbatch --array=0-11 --export=ALL,KYS_EVAL_ROOT=$PWD,KYS_GRID_ARGS="--seeds 44" slurm/eval_grid_array.sbatch
```

- The script activates `${KYS_VENV:-$KYS_EVAL_ROOT/.venv}`, sets `HF_HOME=${HF_HOME:-$KYS_EVAL_ROOT/hf_cache}`, and runs `run_grid --index $SLURM_ARRAY_TASK_ID --delete-weights $KYS_GRID_ARGS`.
- The `--array` range must equal the index range from `run_grid --list $KYS_GRID_ARGS`.
- Resubmitting is safe, because finished cells exit immediately.
- `slurm/reference_check.sbatch` runs the reference check. It needs `KYS_REFERENCE_MODEL=<dir or hf:// spec>`.

### Reference check: `kys_eval.reference_check`

Run it once per cluster, before scoring the grid. It re-scores the v2 checkpoint **diversity_oriented seed 42 ep3** (v2 run `diversity-first-10B-1.5B-seed42`, step 14305) and compares against `reference/expected_diversity_oriented_seed42_ep3.json`:

```bash
python -m kys_eval.reference_check --model /path/to/diversity-first-10B-1.5B-seed42/14305
python -m kys_eval.reference_check --compare-only results/reference_check/results.json   # re-compare, no GPU
```

1. **Weights:** the `model.safetensors` sha256 must be `733262aec4ca52fe94b69df0acf7ef1faa7cd0fc1f42ee79b91004a022db3f45`. This is checked before any GPU time is spent.
2. **Hashes:** for all 64 task entries, LightEval's `hash_examples` and `hash_full_prompts` (dataset and prompts) and `hash_input_tokens` and `hash_cont_tokens` (tokenization) must equal the recorded values.
3. **Scores:** per-task `acc_norm` and `acc` must be within ±0.005, and Mean6 and MMLU within ±0.001. On an H100 the expected verdict is **PASS (exact match)**.

It writes `results/reference_check/{results.json,summary.md,reference_check.md,reference_check.json}`. Exit status: 0 PASS, 1 FAIL.

| ARC-e | HellaSwag | PIQA | SIQA | OBQA | CSQA | **Mean6** | **MMLU** |
|---|---|---|---|---|---|---|---|
| 51.52 | 49.82 | 72.03 | 39.10 | 35.20 | 31.45 | **46.52** | **30.97** |

These are the expected `acc_norm` scores in percent, from v2 on an NVIDIA H100 80GB HBM3. The reference checkpoint is **not on the public Hub**; get it from the maintainers. It is not the
same checkpoint as `blab-jhu/KYS-1.5B-Diversity-Oriented` `seed42/epoch3` (paper-era step 14147).

If the check fails:
- **Hash mismatch:** the environment or datasets differ. Rerun `install.sh`, clear `HF_HOME`, and prefetch again.
- **Scores off by more than the tolerance with matching hashes:** most likely a different GPU architecture or a modified patch.

### Aggregation: `kys_eval.aggregate`

```bash
python -m kys_eval.aggregate [--results-root results] [--out-dir reports]
```

It reads every complete `results/raw_selected/seed*/*/ep*/results.json`, together with the v2 rewritten results in `reference/rewritten_v2_54_per_task.csv`. It can run at any point; missing cells are listed.

- `reports/raw_selected_report.md`:
  - Coverage and the GPUs used.
  - A **summary table of average accuracy (Mean6)**.
  - One table per benchmark: MMLU, ARC-Easy, HellaSwag, PIQA, SIQA, OpenBookQA, CommonsenseQA, and the four MMLU categories.
  - In every table, rows are settings, columns are ep1/ep2/ep3, and cells are mean ± std over seeds.
- `reports/raw_vs_rewritten.md`:
  - For each benchmark, each raw setting next to its rewritten counterpart at every epoch: raw mean ± std, rewritten mean ± std, **Δ = rewritten − raw**, and the number of seeds with Δ > 0.
  - Δ is paired by seed: the mean ± std of the per-seed differences.
- The same numbers as CSV: `raw_selected_per_checkpoint.csv`, `raw_selected_by_epoch.csv` and `raw_vs_rewritten.csv`.

| Raw-selected | Rewritten counterpart (v2 run stem) |
|---|---|
| `raw_diversity_oriented` | `diversity_oriented` (`diversity-first`) |
| `raw_disagreement_aware` | `disagreement_aware` (`signal-disagreement-lambda05`) |
| `raw_random` | `wrap_inspired` (`wrap`) |
| `raw_rewire_inspired` | `rewire_inspired` (`rewrite`) |

### Other tools

| Command | Purpose |
|---|---|
| `python -m kys_eval.check_install` | environment verification (see Install) |
| `python -m kys_eval.prefetch` | download all datasets at pinned revisions into `HF_HOME` |
| `python -m kys_eval.selftest` | offline test: spec parsing, grid, aggregation (reproduces the v2 tables; all deltas 0 on synthetic data), reference comparison logic |
| `python -m kys_eval.eval_checkpoint --from-lighteval-json F --model NAME --out-dir D` | package an existing LightEval `results_*.json` into `results.json` + `summary.md` without evaluating |

## Checkpoint layout

36 checkpoints in HF model repo `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`:

```
rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/      config.json, *.safetensors, tokenizer.json, tokenizer_config.json
  S       ∈ 42, 43, 44
  setting ∈ raw_diversity_oriented, raw_disagreement_aware, raw_random, raw_rewire_inspired
  N       ∈ 1 (step 4768), 2 (step 9537), 3 (step 14305)
```

Each checkpoint is a 1.5B Llama (28 layers, hidden size 2048, vocab 32000, tied embeddings), about 3 GB in bf16. `run_grid --list --check-hub` shows which checkpoints are uploaded.

## Output layout

```
results/                                   (git-ignored)
  raw_selected/seed<S>/<setting>/ep<N>/
    results.json          scores + provenance (schema kys-eval/results/v1)
    summary.md            per-model markdown summary
    lighteval.log         full LightEval output
    lighteval_raw/**/results_<timestamp>.json   raw LightEval output
  reference_check/
    results.json, summary.md, reference_check.md, reference_check.json
  adhoc/<name>/           anything run with eval_checkpoint directly
model_cache/<org>/<repo>/<subfolder>/      downloaded checkpoints (git-ignored; --delete-weights removes them)
reports/                                   aggregate.py output (generated files git-ignored)
```

`results.json` contains:

| Key | Contents |
|---|---|
| `complete` | true only if all 63 tasks are present and there are no protocol violations |
| `missing_tasks`, `protocol_violations` | why a result is not complete |
| `cell` | setting, seed, epoch, step, HF subfolder |
| `model` | spec, HF repo and commit, local path, sha256 of every `*.safetensors` |
| `environment` | host, GPU, Python and package versions, LightEval root, patch and task-file verification, LightEval-recorded sha and model config |
| `protocol` | task string, few-shot count, metric, `max_samples`, environment variables, dataset revisions, exact command |
| `timing` | start/finish, wall seconds, LightEval seconds |
| `scores` | Mean6, MMLU macro, the six commonsense tasks, the four MMLU categories (all `acc_norm`, fractions) |
| `tasks` | per task: `acc`, `acc_norm`, their stderr, and LightEval's four hashes |

## Protocol

| | |
|---|---|
| LightEval | upstream `huggingface/lighteval@10b9104eb16778b174e9e240140974a039018981` (`0.13.1.dev0`) plus `patches/lighteval-10b9104-kys-v2.patch`. v2 results record `lighteval_sha 0228edc…`, a private fork of the same base carrying part of this patch; the rest was uncommitted but active during the run. |
| Patch contents | adds `Metrics.loglikelihood_acc_norm` (token-length-normalized) and `loglikelihood_acc_norm_nospace`; makes `LogProbTokenNorm` ignore padding (`-1`) tokens; fixes 2D continuation slicing (`[:num_choices, :len_choice]`); adds a `LIGHTEVAL_NO_CACHE=1` switch and a pipeline guard for it |
| Backend | `lighteval accelerate`, one process, one GPU, `model_name=<hf dir>,dtype=bfloat16`, automatic batch size, tokenizer from the checkpoint directory |
| Environment | `LIGHTEVAL_NO_CACHE=1` (required), `TOKENIZERS_PARALLELISM=false` |
| Few-shot | 0 for every task |
| Samples | full evaluation split, no `--max-samples` |
| Scoring | log-likelihood of the full answer text for each choice |
| Metric | `acc_norm` = sum of continuation log-probs ÷ number of continuation tokens, then argmax (`acc` also recorded) |
| GPU | v2 used NVIDIA H100 80GB HBM3 for all 54. Use an H100; other architectures move bf16 near-ties by about 7e-4 on Mean6. |

The benchmarks match what the paper reports (§5.4, Fig. 3, Tables 3, 12 and 13, App. F) and what v2 evaluated:

| Benchmark | LightEval task | HF dataset @ revision | Subset | Split | Examples | Paper group |
|---|---|---|---|---|---|---|
| ARC-Easy | `rw_arc:easy` | `allenai/ai2_arc@210d026` | ARC-Easy | test | 2,376 | commonsense (Mean6) |
| HellaSwag | `rw_hellaswag` | `Rowan/hellaswag@218ec52` | default | validation | 10,042 | commonsense (Mean6) |
| PIQA | `rw_piqa` | `lighteval/piqa@41782e6` | plain_text | validation | 1,838 | commonsense (Mean6) |
| SIQA | `rw_siqa` | `lighteval/siqa@54c6a1f` | default | validation | 1,954 | commonsense (Mean6) |
| OpenBookQA | `rw_openbookqa` | `allenai/openbookqa@388097e` | main | validation | 500 | commonsense (Mean6) |
| CommonsenseQA | `rw_commonsense_qa` | `tau/commonsense_qa@94630fe` | default | validation | 1,221 | commonsense (Mean6) |
| MMLU | `rw_mmlu:<subject>` × 57 | `lighteval/mmlu@31d46ab` | one per subject | test | 14,042 | knowledge |

Aggregates (unweighted, as in the paper and v2):
- **Mean6** is the mean of the six commonsense `acc_norm` values.
- **MMLU** is the macro-average over the 57 subjects. It is not example-weighted and does not use LightEval's `rw_mmlu:_average`.
- **MMLU category** is the mean over member subjects: STEM 18, Humanities 13, Social Sciences 12, Other 14.
- **Seed statistics** are the mean and sample standard deviation (ddof=1).

`rewrite_tasks.py` also defines `rw_arc:challenge`, `rw_winogrande`, `rw_boolq`, `rw_sciq` and `rw_lambada`; the paper does not report them and they are not evaluated.

## Repository contents

| Path | Purpose |
|---|---|
| `kys_eval/config.py` | single source of truth: LightEval pins, tasks, metric, dataset revisions, model args, GPU, grid, pairing, tolerances, output locations |
| `kys_eval/tasks/rewrite_tasks.py` | verbatim task file used by v2 (sha256 `a87d708d…`) |
| `kys_eval/tasks/kys_tasks.py` | `--custom-tasks` entry point: the verbatim file, hash-checked, with dataset revisions pinned |
| `kys_eval/common.py` | environment checks, model resolution, LightEval result parsing, `summary.md` |
| `kys_eval/eval_checkpoint.py`, `run_grid.py`, `reference_check.py`, `aggregate.py` | the commands above |
| `kys_eval/check_install.py`, `prefetch.py`, `selftest.py` | install verification, dataset download, offline test |
| `install.sh`, `requirements.txt`, `patches/` | pinned environment |
| `slurm/` | SLURM templates with placeholders |
| `reference/expected_diversity_oriented_seed42_ep3.json` | reference checkpoint identity, scores and hashes |
| `reference/rewritten_v2_54_per_task.csv` | v2 per-task `acc`/`acc_norm` for the 54 rewritten checkpoints |
| `reference/rewritten_v2_{mean6,mmlu}_by_epoch.csv` | v2 seed-aggregated tables, used by the self-test |

`results/`, `model_cache/`, `hf_cache/`, `third_party/`, `.venv/`, logs and credential files are git-ignored.
