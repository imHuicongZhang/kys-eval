# kys-eval: evaluation configuration for the Know-Your-Sources raw-selected baselines

This repository currently holds the **configuration only**: the pinned protocol, task definitions,
LightEval patch, dependency pins, checkpoint grid, and reference numbers. The runner scripts
(single-checkpoint entry point, grid driver, SLURM template, reference check, aggregation) are not
written yet. Review the config first.

The configuration reproduces the **v2 evaluation** of the 54 rewritten-arm checkpoints (6 settings ×
3 seeds × 3 epochs, HF exports under `rewrite-1p5b-hf`), so that the 36 raw-selected checkpoints are
directly comparable to it.

> v2 numbers are not the numbers printed in the paper's tables. The paper's tables come from the
> paper-era checkpoints (steps 4770 / 9540 / 14305, and 14147 for Diversity Oriented). v2 re-scored
> the re-released checkpoints at steps 4768 / 9537 / 14305, with differences of up to about 1 point
> per cell. v2 is the comparator by decision.

## Files

| Path | What it pins |
|---|---|
| `kys_eval/config.py` | Everything below as Python constants: LightEval commit and patch hashes, task list, few-shot, metric, dataset revisions, model args, GPU, grid, pairing, reference tolerances |
| `kys_eval/tasks/rewrite_tasks.py` | Verbatim copy of the task file used by v2 (sha256 `a87d708d…`) |
| `kys_eval/tasks/kys_tasks.py` | The `--custom-tasks` entry point: the verbatim file, hash-checked, with dataset revisions pinned |
| `patches/lighteval-10b9104-kys-v2.patch` | The patch that, applied to upstream LightEval `10b9104`, reproduces the source tree v2 ran |
| `patches/PATCHED_FILES.sha256` | sha256 of the 4 patched files, for `sha256sum -c` from the LightEval checkout root |
| `requirements.txt` | Python 3.11.9 environment pins (torch 2.11.0+cu128, transformers 5.6.2, datasets 4.8.3, …) |
| `reference/expected_diversity_oriented_seed42_ep3.json` | Reference checkpoint identity (file sha256s) and its v2 scores and LightEval hashes for all 63 tasks |
| `reference/rewritten_v2_54_per_task.csv` | v2 per-task `acc` and `acc_norm` for all 54 rewritten checkpoints (the comparator side) |
| `reference/rewritten_v2_{mean6,mmlu}_by_epoch.csv` | v2 seed-aggregated Mean6 and MMLU, setting × epoch |

## Protocol

| | |
|---|---|
| LightEval | upstream `huggingface/lighteval@10b9104eb16778b174e9e240140974a039018981` (version `0.13.1.dev0`) plus `patches/lighteval-10b9104-kys-v2.patch`. v2 results record `lighteval_sha 0228edc…`, a private fork of the same base carrying part of this patch; the rest was uncommitted but active during the run. |
| Patch contents | adds `Metrics.loglikelihood_acc_norm` (token-length-normalized) and `loglikelihood_acc_norm_nospace`; makes `LogProbTokenNorm` ignore padding (`-1`) tokens; fixes 2D continuation slicing (`[:num_choices, :len_choice]`); adds a `LIGHTEVAL_NO_CACHE=1` switch and a pipeline guard for it |
| Backend | `lighteval accelerate`, one process, one GPU, `model_name=<hf dir>,dtype=bfloat16`, automatic batch size, tokenizer from the checkpoint directory |
| Environment | `LIGHTEVAL_NO_CACHE=1` (required), `TOKENIZERS_PARALLELISM=false` |
| Few-shot | 0 for every task |
| Samples | full evaluation split, no `--max-samples` |
| Scoring | log-likelihood of the full answer text for each choice |
| Metric | `acc_norm` = sum of continuation log-probs ÷ number of continuation tokens, then argmax (`acc` also recorded) |
| GPU | v2 used NVIDIA H100 80GB HBM3 for all 54. Use an H100; other architectures move bf16 near-ties by about 7e-4 on Mean6. |

### Benchmarks

The paper reports these benchmarks (§5.4, Fig. 3, Tables 3, 12 and 13, App. F), and v2 evaluated exactly this set.

| Benchmark | LightEval task | HF dataset @ revision | Subset | Split | Examples | Paper group |
|---|---|---|---|---|---|---|
| ARC-Easy | `rw_arc:easy` | `allenai/ai2_arc@210d026` | ARC-Easy | test | 2,376 | commonsense (Mean6) |
| HellaSwag | `rw_hellaswag` | `Rowan/hellaswag@218ec52` | default | validation | 10,042 | commonsense (Mean6) |
| PIQA | `rw_piqa` | `lighteval/piqa@41782e6` | plain_text | validation | 1,838 | commonsense (Mean6) |
| SIQA | `rw_siqa` | `lighteval/siqa@54c6a1f` | default | validation | 1,954 | commonsense (Mean6) |
| OpenBookQA | `rw_openbookqa` | `allenai/openbookqa@388097e` | main | validation | 500 | commonsense (Mean6) |
| CommonsenseQA | `rw_commonsense_qa` | `tau/commonsense_qa@94630fe` | default | validation | 1,221 | commonsense (Mean6) |
| MMLU | `rw_mmlu:<subject>` × 57 | `lighteval/mmlu@31d46ab` | one per subject | test | 14,042 | knowledge |

Full revision hashes are in `config.DATASET_REVISIONS`. Prompts are the full-answer-text prompts of
`rewrite_tasks.py`. `rewrite_tasks.py` also defines `rw_arc:challenge`, `rw_winogrande`, `rw_boolq`,
`rw_sciq` and `rw_lambada`; the paper does not report them and they are not evaluated.

Aggregates (unweighted, as in the paper and v2):
- **Mean6** is the mean of the six commonsense `acc_norm` values.
- **MMLU** is the macro-average over the 57 subjects. It is not example-weighted and does not use LightEval's `rw_mmlu:_average`.
- **MMLU category** is the mean over member subjects: STEM 18, Humanities 13, Social Sciences 12, Other 14.
- **Seed statistics** are the mean and sample standard deviation (ddof=1) over seeds 42, 43 and 44.

Task string passed to LightEval: `config.TASKS_ARG` (6 commonsense tasks then 57 MMLU subsets, each `|0`).

## Checkpoint grid

36 checkpoints in HF model repo `blab-jhu/KYS-1.5B-Raw-Selected-Baselines`:

```
rewrite-1p5b/seed<S>/<setting>/ep<N>/hf/      # config.json, *.safetensors, tokenizer files
  S       ∈ 42, 43, 44
  setting ∈ raw_diversity_oriented, raw_disagreement_aware, raw_random, raw_rewire_inspired
  N       ∈ 1 (step 4768), 2 (step 9537), 3 (step 14305)
```

`config.grid_cells()` fixes the order as (seed, setting, epoch), giving indices 0–35. Every epoch is
evaluated.

### Pairing with the rewritten (v2) arm

| Raw-selected | Rewritten counterpart (v2 run stem) |
|---|---|
| `raw_diversity_oriented` | `diversity_oriented` (`diversity-first`) |
| `raw_disagreement_aware` | `disagreement_aware` (`signal-disagreement-lambda05`) |
| `raw_random` | `wrap_inspired` (`wrap`) |
| `raw_rewire_inspired` | `rewire_inspired` (`rewrite`) |

## Reference check

Before scoring new models, an operator re-scores one existing v2 checkpoint and compares against
`reference/expected_diversity_oriented_seed42_ep3.json`:
- **Weights:** the file sha256s must match.
- **Data, prompts and tokenization:** all four LightEval hashes per task must match exactly.
- **Scores:** per-task `acc_norm` must be within ±0.005 and Mean6/MMLU within ±0.001. On an H100 an exact match is expected.

Checkpoint `diversity_oriented` seed 42 ep3, step 14305 (v2 run `diversity-first-10B-1.5B-seed42/14305`), scored on NVIDIA H100 80GB HBM3:

| ARC-e | HellaSwag | PIQA | SIQA | OBQA | CSQA | **Mean6** | **MMLU** |
|---|---|---|---|---|---|---|---|
| 51.52 | 49.82 | 72.03 | 39.10 | 35.20 | 31.45 | **46.52** | **30.97** |

`model.safetensors` sha256: `733262aec4ca52fe94b69df0acf7ef1faa7cd0fc1f42ee79b91004a022db3f45`.
This checkpoint is **not on the public Hub** and must be obtained from the maintainers. It is not
the same checkpoint as `blab-jhu/KYS-1.5B-Diversity-Oriented` `seed42/epoch3` (paper-era step 14147).

## Installing the pinned environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/huggingface/lighteval.git third_party/lighteval
git -C third_party/lighteval checkout --detach 10b9104eb16778b174e9e240140974a039018981
git -C third_party/lighteval apply --whitespace=nowarn "$PWD/patches/lighteval-10b9104-kys-v2.patch"
(cd third_party/lighteval && sha256sum -c "$OLDPWD/patches/PATCHED_FILES.sha256")   # all 4 must say OK
pip install --no-deps -e third_party/lighteval
```

`results/`, `model_cache/`, `hf_cache/`, `third_party/`, logs and credential files are git-ignored.
