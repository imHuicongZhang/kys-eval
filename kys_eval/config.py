"""Single source of truth for the KYS raw-selected baseline evaluation.

Everything here is pinned to the "v2" evaluation of the 54 rewritten-arm checkpoints
(6 settings x 3 seeds x 3 epochs, HF exports under rewrite-1p5b-hf, results in
projects/rewrite/08_evaluation/04_results_1p5_v2 on the originating cluster), so that the 36
raw-selected checkpoints scored with this configuration are directly comparable to it.

Note: v2 numbers are NOT the numbers printed in the paper's tables. The paper's tables come from the
paper-era checkpoints (steps 4770 / 9540 / 14305, 14147 for Diversity Oriented); v2 re-evaluated
the re-released branched-schedule checkpoints (steps 4768 / 9537 / 14305). v2 is the comparator by
decision.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================================
# LightEval: upstream commit + the exact patch the v2 evaluation ran with
# =============================================================================================
LIGHTEVAL_GIT_URL = "https://github.com/huggingface/lighteval.git"
LIGHTEVAL_BASE_COMMIT = "10b9104eb16778b174e9e240140974a039018981"
LIGHTEVAL_VERSION = "0.13.1.dev0"
# lighteval_sha recorded in every v2 results JSON: a private fork commit = LIGHTEVAL_BASE_COMMIT plus
# part of the patch below. The rest of the patch was uncommitted in that working tree but active
# during the v2 run (files dated 2026-08-20, run 2026-09-07).
LIGHTEVAL_V2_RECORDED_SHA = "0228edcd06f0f3bcf88baf5948e7746577b2af71"
LIGHTEVAL_DIR = Path(os.environ.get("KYS_LIGHTEVAL_DIR", REPO_ROOT / "third_party" / "lighteval"))
PATCH_FILE = REPO_ROOT / "patches" / "lighteval-10b9104-kys-v2.patch"
# sha256 of the patched files; byte-identical to the source tree that produced the v2 results
# (verified: the patch applied to LIGHTEVAL_BASE_COMMIT reproduces these hashes).
PATCHED_FILE_SHA256 = {
    "src/lighteval/metrics/metrics.py": "a71142c185923f4c49f13548c9b90d69f2e9ff21b0209921db8ce2c46ead4c4f",
    "src/lighteval/metrics/normalizations.py": "f42cbc049fcfc71aa6d4a8a40ae32c73fa8c5d2741708026a8021477e4105661",
    "src/lighteval/models/transformers/transformers_model.py": "bb2cf669ef107e632de8e06b95c1a2a1a3175f81d262226ae43773821bb52833",
    "src/lighteval/pipeline.py": "8bb12ec199569e9fd9b089e6803c3d58c55fe5b3df4c04e27a776fa003c32118",
}

# Versions of the v2 environment (Python 3.11.9). Full pin list: requirements.txt.
PYTHON_VERSION = "3.11.9"
PINNED_VERSIONS = {
    "torch": "2.11.0+cu128",
    "transformers": "5.6.2",
    "datasets": "4.8.3",
    "accelerate": "1.13.0",
    "huggingface_hub": "1.12.0",
    "tokenizers": "0.22.2",
    "numpy": "2.4.4",
    "lighteval": LIGHTEVAL_VERSION,
}

# =============================================================================================
# Tasks
# =============================================================================================
# Verbatim copy of projects/lighteval/community_tasks/rewrite_tasks.py (prompts, splits, metrics).
TASK_FILE_VERBATIM = REPO_ROOT / "kys_eval" / "tasks" / "rewrite_tasks.py"
TASK_FILE_SHA256 = "a87d708dab3d84a4870207eaeeff5e251bf11e8e6f513c42870a4786af5b0b4d"
# What gets passed to --custom-tasks: the verbatim file with dataset revisions pinned.
TASK_FILE = REPO_ROOT / "kys_eval" / "tasks" / "kys_tasks.py"

# HF dataset revisions used by v2 (from its offline HF cache; equal to Hub HEAD on 2026-09-15).
# v2 itself loaded unpinned HEAD; pinning guards against later dataset edits.
DATASET_REVISIONS = {
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "lighteval/piqa": "41782e6bf0ef7de82a2ca8a9feb1dca042837fae",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "allenai/openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
    "lighteval/siqa": "54c6a1f8cb6daf4f5abf24a601852612fb35eb25",
    "lighteval/mmlu": "31d46ab06e6934bb0d95f6918668716d1db6f921",
}

NUM_FEWSHOT = 0
# Every task: 0-shot, full evaluation split (no --max-samples), multiple choice scored by
# log-likelihood of the full answer text. Headline metric acc_norm = LogProbTokenNorm
# (sum of continuation log-probs / number of continuation tokens). acc is also recorded.
METRIC = "acc_norm"

# (LightEval task id, short name, paper label, HF repo, subset, split, n examples). Paper order
# (Section 5.4, Figure 3a, Tables 12 and 13).
COMMONSENSE = [
    ("rw_arc:easy", "arc_easy", "ARC-Easy", "allenai/ai2_arc", "ARC-Easy", "test", 2376),
    ("rw_hellaswag", "hellaswag", "HellaSwag", "Rowan/hellaswag", "default", "validation", 10042),
    ("rw_piqa", "piqa", "PIQA", "lighteval/piqa", "plain_text", "validation", 1838),
    ("rw_siqa", "siqa", "SIQA", "lighteval/siqa", "default", "validation", 1954),
    ("rw_openbookqa", "openbookqa", "OpenBookQA", "allenai/openbookqa", "main", "validation", 500),
    ("rw_commonsense_qa", "commonsense_qa", "CommonsenseQA", "tau/commonsense_qa", "default", "validation", 1221),
]

# MMLU: lighteval/mmlu, test split, 14,042 examples over 57 subjects (Section 5.4, Figure 3b, Table 3).
MMLU_REPO = "lighteval/mmlu"
MMLU_SPLIT = "test"
MMLU_N_EXAMPLES = 14042
MMLU_TASK_PREFIX = "rw_mmlu:"
MMLU_LIGHTEVAL_AVERAGE_TASK = "rw_mmlu:_average"  # written by LightEval; not used for the macro
MMLU_CATEGORIES = ["STEM", "Humanities", "Social Sciences", "Other"]
# Hendrycks et al. (2021) taxonomy; validated in v2 against the paper's Table 3.
MMLU_SUBJECT_CATEGORY = {
    "abstract_algebra": "STEM", "astronomy": "STEM", "college_biology": "STEM",
    "college_chemistry": "STEM", "college_computer_science": "STEM", "college_mathematics": "STEM",
    "college_physics": "STEM", "computer_security": "STEM", "conceptual_physics": "STEM",
    "electrical_engineering": "STEM", "elementary_mathematics": "STEM", "high_school_biology": "STEM",
    "high_school_chemistry": "STEM", "high_school_computer_science": "STEM",
    "high_school_mathematics": "STEM", "high_school_physics": "STEM", "high_school_statistics": "STEM",
    "machine_learning": "STEM",
    "formal_logic": "Humanities", "high_school_european_history": "Humanities",
    "high_school_us_history": "Humanities", "high_school_world_history": "Humanities",
    "international_law": "Humanities", "jurisprudence": "Humanities", "logical_fallacies": "Humanities",
    "moral_disputes": "Humanities", "moral_scenarios": "Humanities", "philosophy": "Humanities",
    "prehistory": "Humanities", "professional_law": "Humanities", "world_religions": "Humanities",
    "econometrics": "Social Sciences", "high_school_geography": "Social Sciences",
    "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences", "high_school_microeconomics": "Social Sciences",
    "high_school_psychology": "Social Sciences", "human_sexuality": "Social Sciences",
    "professional_psychology": "Social Sciences", "public_relations": "Social Sciences",
    "security_studies": "Social Sciences", "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    "anatomy": "Other", "business_ethics": "Other", "clinical_knowledge": "Other",
    "college_medicine": "Other", "global_facts": "Other", "human_aging": "Other", "management": "Other",
    "marketing": "Other", "medical_genetics": "Other", "miscellaneous": "Other", "nutrition": "Other",
    "professional_accounting": "Other", "professional_medicine": "Other", "virology": "Other",
}
MMLU_SUBSETS = sorted(MMLU_SUBJECT_CATEGORY)
assert len(MMLU_SUBSETS) == 57

EXPECTED_TASKS = [t[0] for t in COMMONSENSE] + [MMLU_TASK_PREFIX + s for s in MMLU_SUBSETS]
# Task argument, same order as v2's $TASKS_FIG (6 commonsense then 57 MMLU subsets).
TASKS_ARG = ",".join(
    f"{t}|{NUM_FEWSHOT}"
    for t in ["rw_hellaswag", "rw_piqa", "rw_arc:easy", "rw_openbookqa", "rw_commonsense_qa", "rw_siqa"]
    + [MMLU_TASK_PREFIX + s for s in MMLU_SUBSETS]
)

# Aggregates (both unweighted, as in the paper and v2):
#   Mean6 = mean of the six commonsense acc_norm values
#   MMLU  = macro-average of the 57 subject acc_norm values (not example-weighted, not _average)
#   MMLU category = mean of member-subject acc_norm values
# Seed statistics: mean and sample standard deviation (ddof=1) over seeds 42, 43, 44.

# =============================================================================================
# Model loading and hardware (identical to v2)
# =============================================================================================
# lighteval accelerate, one process, one GPU, bf16, automatic batch size (batch_size unset:
# starts at 512 and halves on OOM). Tokenizer is read from the checkpoint directory.
LIGHTEVAL_BACKEND = "accelerate"
MODEL_ARGS = "model_name={path},dtype=bfloat16"
EVAL_ENV = {
    # Required: LightEval's sample cache round-trips through parquet and mangles the 2D
    # output_tokens that token-normalized acc_norm needs.
    "LIGHTEVAL_NO_CACHE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
# All 54 v2 checkpoints were scored on this device. bf16 near-ties resolve differently across GPU
# architectures (~7e-4 on Mean6, up to 4e-3 on one task, H200 vs L40S); H100 NVL vs H100 SXM was
# negligible (9/10 identical, 1 example flip). Score everything on an H100.
REFERENCE_GPU = "NVIDIA H100 80GB HBM3"
REQUIRED_GPU_SUBSTRING = "H100"
V2_SECONDS_PER_CHECKPOINT = 548  # measured, diversity_oriented seed42 ep3, one H100

# =============================================================================================
# Raw-selected grid: 4 settings x 3 seeds x 3 epochs = 36 checkpoints
# =============================================================================================
HF_REPO = "blab-jhu/KYS-1.5B-Raw-Selected-Baselines"
HF_ROOT_PREFIX = "rewrite-1p5b"
RAW_SETTINGS = ["raw_diversity_oriented", "raw_disagreement_aware", "raw_random", "raw_rewire_inspired"]
SEEDS = [42, 43, 44]
EPOCHS = ["ep1", "ep2", "ep3"]
EPOCH_STEPS = {"ep1": 4768, "ep2": 9537, "ep3": 14305}  # same steps as the v2 rewritten grid
TOKENS_PER_STEP = 2097152


def hf_subfolder(seed: int, setting: str, epoch: str) -> str:
    """Path of one HF checkpoint inside HF_REPO: rewrite-1p5b/seed<S>/<setting>/ep<N>/hf."""
    return f"{HF_ROOT_PREFIX}/seed{seed}/{setting}/{epoch}/hf"


def grid_cells(seeds=SEEDS, settings=RAW_SETTINGS, epochs=EPOCHS):
    """Checkpoints in fixed order (seed, setting, epoch); index 0..35 for the full grid."""
    return [
        {"setting": s, "seed": seed, "epoch": e, "step": EPOCH_STEPS[e], "hf_subfolder": hf_subfolder(seed, s, e)}
        for seed in seeds
        for s in settings
        for e in epochs
    ]


# =============================================================================================
# Rewritten comparator (v2) and pairing
# =============================================================================================
REWRITTEN_SETTINGS = [
    "quality_base", "quality_first", "diversity_oriented", "disagreement_aware", "wrap_inspired", "rewire_inspired",
]
# v2 run-directory stems, for provenance.
REWRITTEN_V2_RUN_STEM = {
    "quality_base": "quality-base",
    "quality_first": "quality-first",
    "diversity_oriented": "diversity-first",
    "disagreement_aware": "signal-disagreement-lambda05",
    "wrap_inspired": "wrap",
    "rewire_inspired": "rewrite",
}
PAPER_LABEL = {
    "quality_base": "Quality Base",
    "quality_first": "Quality First",
    "diversity_oriented": "Diversity Oriented",
    "disagreement_aware": "Disagreement Aware",
    "wrap_inspired": "WRAP Inspired",
    "rewire_inspired": "REWIRE Inspired",
}
RAW_TO_REWRITTEN = {
    "raw_diversity_oriented": "diversity_oriented",
    "raw_disagreement_aware": "disagreement_aware",
    "raw_random": "wrap_inspired",
    "raw_rewire_inspired": "rewire_inspired",
}
# Per-task v2 results for the 54 rewritten checkpoints (acc and acc_norm, all 63 tasks + _average).
REWRITTEN_V2_PER_TASK_CSV = REPO_ROOT / "reference" / "rewritten_v2_54_per_task.csv"

# =============================================================================================
# Reference check: one existing v2 checkpoint an operator re-scores before scoring new models
# =============================================================================================
REFERENCE_EXPECTED_JSON = REPO_ROOT / "reference" / "expected_diversity_oriented_seed42_ep3.json"
# Absolute tolerances on acc_norm (fraction, not points). On an H100 an exact match is expected.
REFERENCE_TOL_TASK = 0.005
REFERENCE_TOL_AGGREGATE = 0.001

# =============================================================================================
# Local output locations (all git-ignored)
# =============================================================================================
RESULTS_ROOT = Path(os.environ.get("KYS_RESULTS_ROOT", REPO_ROOT / "results"))
MODEL_CACHE = Path(os.environ.get("KYS_MODEL_CACHE", REPO_ROOT / "model_cache"))
REPORTS_DIR = REPO_ROOT / "reports"
