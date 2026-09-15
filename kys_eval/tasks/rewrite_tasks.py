"""
Rewrite-project evaluation tasks for lighteval.

Six 0-shot multiple-choice tasks scored by loglikelihood on FULL answer text
(HellaSwag, PIQA, ARC-Easy, OpenBookQA, CommonsenseQA, SIQA).

These reuse the exact prompt functions / HF repos / splits / 0-shot setting from
``community_tasks/fineweb_tasks.py`` (copied verbatim so this file is fully
self-contained and immune to custom-task-loader import fragility).

The ONLY methodological difference from fineweb_tasks.py is the metric:

  fineweb_tasks.py : [loglikelihood_acc, loglikelihood_acc_norm_nospace]   # char-length norm
  rewrite_tasks.py : [loglikelihood_acc, loglikelihood_acc_norm]           # TOKEN-length norm

``Metrics.loglikelihood_acc_norm`` (metric_name "acc_norm") uses
``LoglikelihoodAcc(logprob_normalization=LogProbTokenNorm())``: for each candidate
answer it divides the sum of the continuation log-probs by the number of continuation
tokens (context excluded, padding excluded), then takes the argmax. This is the
"acc_norm normalized by continuation token length" requested for the rewrite pilot.

Task names are prefixed "rw_" to avoid any collision with fineweb_tasks.py.

Run (0-shot, six tasks):
  lighteval accelerate "pretrained=<hf_model>,dtype=bfloat16" \\
      "custom|rw_hellaswag|0|0,custom|rw_piqa|0|0,custom|rw_arc:easy|0|0,\\
custom|rw_openbookqa|0|0,custom|rw_commonsense_qa|0|0,custom|rw_siqa|0|0" \\
      --custom-tasks community_tasks/rewrite_tasks.py
"""

import re

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

# ---------------------------------------------------------------------------
# Shared metrics — token-length-normalized acc_norm is the headline metric.
#   Metrics.loglikelihood_acc       -> raw argmax of summed continuation log-prob (reference)
#   Metrics.loglikelihood_acc_norm  -> LogProbTokenNorm: sum(logprob) / #continuation_tokens
# ---------------------------------------------------------------------------
_METRICS = [Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm]

# ---------------------------------------------------------------------------
# Prompt functions — copied verbatim from community_tasks/fineweb_tasks.py
# ---------------------------------------------------------------------------


def hellaswag_prompt(line, task_name: str = None):
    """Fineweb version: removes WikiHow bracket artefacts, uses raw endings."""

    def preprocess(text):
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,
    )


def piqa_harness_prompt(line, task_name: str = None):
    """Harness-style PIQA: scores log-prob of each full solution text."""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['goal']}\nAnswer:",
        choices=[f" {line['sol1']}", f" {line['sol2']}"],
        gold_index=int(line["label"]),
    )


def arc_fw_prompt(line, task_name: str = None):
    """ARC without inspect_ai: scores log-prob of each full choice text."""
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )


def openbookqa_fw_prompt(line, task_name: str = None):
    """Fineweb-style OpenBookQA: bare question_stem, full choice texts, no letters."""
    return Doc(
        task_name=task_name,
        query=line["question_stem"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=["A", "B", "C", "D", "E"].index(line["answerKey"].strip()),
    )


def commonsense_qa_prompt(line, task_name: str = None):
    """Fineweb version: bare question, full choice texts, gold from answerKey letter."""
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=ord(line["answerKey"].strip()) - ord("A"),
        instruction="",
    )


def siqa_prompt(line, task_name: str = None):
    """Fineweb version: context + question as query, raw answer texts as choices."""
    return Doc(
        task_name=task_name,
        query=line["context"] + " " + line["question"],
        choices=[f" {line['answerA']}", f" {line['answerB']}", f" {line['answerC']}"],
        gold_index=int(line["label"]) - 1,
        instruction="",
    )


def winogrande_prompt(line, task_name: str = None):
    """Built-in lighteval Winogrande: fill-the-blank, two full-sentence completions."""
    query, end_of_target = line["sentence"].split("_")
    end_of_target = end_of_target.strip()
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"{line['option1']} {end_of_target}", f"{line['option2']} {end_of_target}"],
        gold_index=int(line["answer"]) - 1 if line["answer"] != "" else -1,
    )


def boolq_prompt(line, task_name: str = None):
    """Built-in lighteval BoolQ: passage+question, yes/no as loglikelihood choices."""
    question = line["question"][:-1] if line["question"][-2:] == "??" else line["question"]
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {question}\nAnswer:",
        choices=[" Yes", " No"],
        gold_index=["Yes", "No"].index(line["answer"]),
    )


def sciq_prompt(line, task_name: str = None):
    """Built-in lighteval SciQ: support passage + question, four full-text choices (gold last)."""
    return Doc(
        task_name=task_name,
        query=f"{line['support']}\nQuestion: {line['question']}\nAnswer:".strip(),
        choices=[
            f" {c}" for c in [line["distractor1"], line["distractor2"], line["distractor3"], line["correct_answer"]]
        ],
        gold_index=3,
    )


def lambada_prompt(line, task_name: str = None):
    """Built-in lighteval LAMBADA: predict the final word given broad narrative context."""
    query, choice = line["text"].rsplit(" ", 1)
    return Doc(
        task_name=task_name,
        query=query,
        gold_index=0,
        choices=[f" {choice}"],
    )


def mmlu_fw_prompt(line, task_name: str = None):
    """
    Fineweb MMLU prompt: NO letter labels in query, full answer text as choices.

    Built-in lighteval MMLU lists 'A. … B. … C. … D. …' in the query and scores
    the single letter token — random on small models.  This version scores the
    log-prob of each complete answer string, giving real signal even on tiny models.
    """
    topic = line["subject"].replace("_", " ")
    return Doc(
        task_name=task_name,
        query=f"The following are questions about {topic}.\nQuestion: {line['question']}\nAnswer:",
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=f"The following are questions about {topic}.\n",
    )


# ---------------------------------------------------------------------------
# Task configs — identical to fineweb_tasks.py except metrics (token-norm acc_norm)
# ---------------------------------------------------------------------------

hellaswag = LightevalTaskConfig(
    name="rw_hellaswag",
    prompt_function=hellaswag_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=_METRICS,
    version=0,
)

piqa = LightevalTaskConfig(
    name="rw_piqa",
    prompt_function=piqa_harness_prompt,
    hf_repo="lighteval/piqa",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=_METRICS,
    version=0,
)

arc_easy = LightevalTaskConfig(
    name="rw_arc:easy",
    prompt_function=arc_fw_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Easy",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=_METRICS,
    version=0,
)

# ARC-Challenge — harder ARC split; reported as an extra reasoning task (NOT part of
# the locked commonsense Mean6). Same prompt/metric as ARC-Easy.
arc_challenge = LightevalTaskConfig(
    name="rw_arc:challenge",
    prompt_function=arc_fw_prompt,
    hf_repo="allenai/ai2_arc",
    hf_subset="ARC-Challenge",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=_METRICS,
    version=0,
)

openbookqa = LightevalTaskConfig(
    name="rw_openbookqa",
    prompt_function=openbookqa_fw_prompt,
    hf_repo="allenai/openbookqa",
    hf_subset="main",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=_METRICS,
    version=0,
)

commonsense_qa = LightevalTaskConfig(
    name="rw_commonsense_qa",
    prompt_function=commonsense_qa_prompt,
    hf_repo="tau/commonsense_qa",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=_METRICS,
    version=0,
)

siqa = LightevalTaskConfig(
    name="rw_siqa",
    prompt_function=siqa_prompt,
    hf_repo="lighteval/siqa",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=_METRICS,
    version=0,
)

# ---------------------------------------------------------------------------
# Task configs — breadth suite (4 tasks): coreference / reading comprehension /
# long-range discourse / science QA. WinoGrande, BoolQ, SciQ are multiple-choice
# and use the same token-norm acc_norm as the seven. LAMBADA has a SINGLE gold
# continuation (no answer choices), so acc_norm is undefined; it uses
# `acc_golds_likelihood` — the standard LAMBADA last-word accuracy (1 iff the
# model's greedy argmax over the gold tokens equals the gold), via loglikelihood.
# ---------------------------------------------------------------------------

_LAMBADA_METRIC = [Metrics.acc_golds_likelihood]

winogrande = LightevalTaskConfig(
    name="rw_winogrande",
    prompt_function=winogrande_prompt,
    hf_repo="allenai/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=_METRICS,
    version=0,
)

boolq = LightevalTaskConfig(
    name="rw_boolq",
    prompt_function=boolq_prompt,
    hf_repo="lighteval/boolq_helm",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=_METRICS,
    version=0,
)

sciq = LightevalTaskConfig(
    name="rw_sciq",
    prompt_function=sciq_prompt,
    hf_repo="allenai/sciq",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=_METRICS,
    version=0,
)

lambada = LightevalTaskConfig(
    name="rw_lambada",
    prompt_function=lambada_prompt,
    hf_repo="cimec/lambada",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=_LAMBADA_METRIC,
    version=0,
)

# ---------------------------------------------------------------------------
# Task configs — MMLU (57 subsets, full answer text + token-norm acc_norm)
#   Reported as the 7th task per-subject; macro-averaged into a single MMLU
#   number downstream. Same prompt/splits as fineweb_tasks.py, metric swapped.
# ---------------------------------------------------------------------------

_MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

MMLU_TASKS = [
    LightevalTaskConfig(
        name=f"rw_mmlu:{subset}",
        prompt_function=mmlu_fw_prompt,
        hf_repo="lighteval/mmlu",
        hf_subset=subset,
        hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        metrics=_METRICS,
        version=0,
    )
    for subset in _MMLU_SUBSETS
]

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

COMMONSENSE_TASKS = [
    hellaswag,
    piqa,
    arc_easy,
    openbookqa,
    commonsense_qa,
    siqa,
]

# Breadth suite — tests coverage beyond commonsense (coreference, reading
# comprehension, long-range discourse, science knowledge).
BREADTH_TASKS = [
    winogrande,
    boolq,
    sciq,
    lambada,
]

# Extra reasoning task, kept OUT of the locked commonsense Mean6.
EXTRA_TASKS = [
    arc_challenge,
]

TASKS_TABLE = COMMONSENSE_TASKS + BREADTH_TASKS + EXTRA_TASKS + MMLU_TASKS

_task_string = lambda task: f"custom|{task.name}|0|0"  # noqa: E731

TASKS_GROUPS = {
    # Six commonsense tasks (the headline-mean set).
    "rewrite": ",".join(_task_string(t) for t in COMMONSENSE_TASKS),
    # Four breadth tasks (WinoGrande, BoolQ, SciQ, LAMBADA).
    "rewrite_breadth": ",".join(_task_string(t) for t in BREADTH_TASKS),
    # Breadth + ARC-Challenge (the finals-now extra pass).
    "rewrite_extra": ",".join(_task_string(t) for t in BREADTH_TASKS + EXTRA_TASKS),
    # Everything: 6 commonsense + 4 breadth + ARC-Challenge + 57 MMLU subsets.
    "rewrite_all": ",".join(_task_string(t) for t in TASKS_TABLE),
}
