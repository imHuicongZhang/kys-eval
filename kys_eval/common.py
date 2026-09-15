"""Shared helpers: environment checks, model resolution, LightEval result parsing, summaries."""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import socket
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from kys_eval import config as C

RESULTS_SCHEMA = "kys-eval/results/v1"
HF_PREFIX = "hf://"
HASH_FIELDS = ("hash_examples", "hash_full_prompts", "hash_input_tokens", "hash_cont_tokens")
METRIC_FIELDS = ("acc", "acc_stderr", "acc_norm", "acc_norm_stderr")


class EvalError(RuntimeError):
    """A failure that stops one checkpoint; the message is meant for the operator."""


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def pct(value, signed=False):
    if value is None:
        return "—"
    return f"{value * 100:+.2f}" if signed else f"{value * 100:.2f}"


def category_key(category):
    return "mmlu_" + category.lower().replace(" ", "_")


# =============================================================================================
# GPU and software environment
# =============================================================================================
def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = out.stdout.strip().splitlines() if out.returncode == 0 else []
    return lines[0].strip() if lines else ""


def check_gpu(allow_any_gpu):
    name = gpu_name()
    if C.REQUIRED_GPU_SUBSTRING in name or allow_any_gpu:
        return name
    raise EvalError(
        f"GPU is {name or 'not visible'!r}; the v2 results were scored on {C.REFERENCE_GPU}. bf16 near-ties resolve "
        "differently across GPU architectures, so run on an H100, or pass --allow-any-gpu (the GPU is recorded "
        "and flagged in the reports)."
    )


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def lighteval_root():
    """Checkout root of the importable lighteval (<root>/src/lighteval/__init__.py), or None."""
    spec = importlib.util.find_spec("lighteval")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parents[2]


def environment_info():
    root = lighteval_root()
    patched = {}
    for rel, want in C.PATCHED_FILE_SHA256.items():
        path = root / rel if root else None
        patched[rel] = bool(path and path.is_file() and sha256_file(path) == want)
    versions = {name: package_version(name) for name in C.PINNED_VERSIONS}
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "packages": versions,
        "version_mismatches": {
            name: {"pinned": C.PINNED_VERSIONS[name], "installed": v}
            for name, v in versions.items()
            if v != C.PINNED_VERSIONS[name]
        },
        "lighteval_root": str(root) if root else None,
        "lighteval_patched_files_ok": patched,
        "lighteval_patch_ok": bool(patched) and all(patched.values()),
        "task_file_sha256_ok": sha256_file(C.TASK_FILE_VERBATIM) == C.TASK_FILE_SHA256,
    }


def environment_problems(env):
    problems = []
    if env["python"].rsplit(".", 1)[0] != C.PYTHON_VERSION.rsplit(".", 1)[0]:
        problems.append(f"Python {env['python']} (pinned {C.PYTHON_VERSION})")
    if env["lighteval_root"] is None:
        problems.append("lighteval is not importable")
    elif not env["lighteval_patch_ok"]:
        bad = [rel for rel, ok in env["lighteval_patched_files_ok"].items() if not ok]
        problems.append(f"lighteval at {env['lighteval_root']} is not the patched v2 tree (sha256 mismatch: {bad})")
    for name, v in env["version_mismatches"].items():
        problems.append(f"{name} {v['installed']} (pinned {v['pinned']})")
    if not env["task_file_sha256_ok"]:
        problems.append(f"{C.TASK_FILE_VERBATIM} was modified (sha256 != {C.TASK_FILE_SHA256})")
    return problems


# =============================================================================================
# Models: local directories and hf://<org>/<repo>/<subfolder>[@revision]
# =============================================================================================
def parse_model_spec(spec):
    if spec.startswith(HF_PREFIX):
        body, revision = spec[len(HF_PREFIX):], None
        if "@" in body:
            body, revision = body.rsplit("@", 1)
        parts = [p for p in body.split("/") if p]
        if len(parts) < 2:
            raise EvalError(f"bad model spec {spec!r}; expected hf://<org>/<repo>[/<subfolder>][@<revision>]")
        return {"kind": "hf", "repo_id": "/".join(parts[:2]), "subfolder": "/".join(parts[2:]), "revision": revision}
    return {"kind": "local", "path": str(Path(spec).expanduser().resolve())}


def resolve_model(spec, cache_dir):
    """Return (local_dir, provenance, downloaded). HF checkpoints are downloaded at a pinned commit."""
    parsed = parse_model_spec(spec)
    if parsed["kind"] == "local":
        path = Path(parsed["path"])
        if not path.is_dir():
            raise EvalError(f"model directory {path} does not exist")
        return path, {"spec": spec, **parsed}, False

    from huggingface_hub import HfApi, snapshot_download

    repo_id, subfolder = parsed["repo_id"], parsed["subfolder"]
    try:
        commit = HfApi().model_info(repo_id, revision=parsed["revision"]).sha
    except Exception as exc:  # network, auth, missing repo
        raise EvalError(f"cannot reach HF model repo {repo_id}: {exc}") from exc
    local_root = Path(cache_dir) / repo_id
    snapshot_download(
        repo_id=repo_id,
        revision=commit,
        allow_patterns=[f"{subfolder}/*"] if subfolder else None,
        local_dir=local_root,
    )
    path = local_root / subfolder if subfolder else local_root
    if not (path / "config.json").is_file():
        raise EvalError(f"{spec}: no config.json under {subfolder or '/'} at commit {commit}; is this checkpoint uploaded?")
    return path, {"spec": spec, **parsed, "commit": commit}, True


def validate_model_dir(path):
    """Return (problems, warnings) for a HF checkpoint directory."""
    path = Path(path)
    problems, warnings = [], []
    if not (path / "config.json").is_file():
        problems.append("config.json missing")
    if not list(path.glob("*.safetensors")):
        problems.append("no *.safetensors weights")
    if not ((path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file()):
        problems.append("no tokenizer.json / tokenizer.model (LightEval loads the tokenizer from the model directory)")
    if (path / "config.json").is_file():
        cfg = json.loads((path / "config.json").read_text())
        expected = {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "tie_word_embeddings": True,
        }
        for key, want in expected.items():
            if cfg.get(key) != want:
                warnings.append(f"config.json {key}={cfg.get(key)!r}, KYS 1.5B checkpoints have {want!r}")
    return problems, warnings


def weights_sha256(path):
    return {p.name: sha256_file(p) for p in sorted(Path(path).glob("*.safetensors"))}


# =============================================================================================
# LightEval results
# =============================================================================================
def find_lighteval_results(raw_dir):
    """Newest results_<timestamp>.json below raw_dir (LightEval nests it under the model path)."""
    files = sorted(Path(raw_dir).rglob("results_*.json"), key=lambda p: p.name)
    return files[-1] if files else None


def parse_lighteval_results(raw):
    summaries = raw.get("summary_tasks", {})
    tasks = {}
    for key, metrics in raw["results"].items():
        if key == "all":
            continue
        entry = {k: metrics[k] for k in METRIC_FIELDS if k in metrics}
        hashes = summaries.get(key, {}).get("hashes")
        if hashes:
            entry["hashes"] = hashes
        tasks[key.split("|")[0]] = entry
    return tasks


def benchmark_scores(acc_norm):
    """Benchmark-level scores from {task_id: acc_norm}. None where a component is missing."""
    scores = {short: acc_norm.get(tid) for tid, short, *_ in C.COMMONSENSE}
    commonsense = [scores[short] for _, short, *_ in C.COMMONSENSE]
    scores["mean6"] = statistics.fmean(commonsense) if None not in commonsense else None
    subjects = {s: acc_norm.get(C.MMLU_TASK_PREFIX + s) for s in C.MMLU_SUBSETS}
    complete = None not in subjects.values()
    scores["mmlu"] = statistics.fmean(subjects.values()) if complete else None
    for category in C.MMLU_CATEGORIES:
        members = [v for s, v in subjects.items() if C.MMLU_SUBJECT_CATEGORY[s] == category]
        scores[category_key(category)] = statistics.fmean(members) if complete else None
    return scores


def missing_tasks(tasks):
    return [t for t in C.EXPECTED_TASKS if "acc_norm" not in tasks.get(t, {})]


def protocol_violations(raw):
    violations = []
    general = raw.get("config_general", {})
    if general.get("max_samples") is not None:
        violations.append(f"max_samples={general['max_samples']} (protocol: full splits)")
    shots = {v.get("num_fewshots") for v in raw.get("config_tasks", {}).values()}
    if shots - {C.NUM_FEWSHOT}:
        violations.append(f"num_fewshots {sorted(shots, key=str)} (protocol: {C.NUM_FEWSHOT})")
    return violations


def build_results(raw, *, label, cell, model, environment, command, started_at, finished_at, wall_seconds, raw_path):
    tasks = parse_lighteval_results(raw)
    missing = missing_tasks(tasks)
    violations = protocol_violations(raw)
    general = raw.get("config_general", {})
    model_config = general.get("model_config", {})
    lighteval_seconds = general.get("total_evaluation_time_secondes")
    return {
        "schema": RESULTS_SCHEMA,
        "label": label,
        "cell": cell,
        "complete": not missing and not violations,
        "missing_tasks": missing,
        "protocol_violations": violations,
        "model": model,
        "environment": {
            **environment,
            "lighteval_sha_recorded": general.get("lighteval_sha"),
            "lighteval_model_config": {
                k: model_config.get(k) for k in ("model_name", "dtype", "batch_size", "revision", "max_length")
            },
        },
        "protocol": {
            "tasks_arg": C.TASKS_ARG,
            "num_fewshot": C.NUM_FEWSHOT,
            "metric": "acc_norm = LogProbTokenNorm (sum of continuation log-probs / continuation tokens)",
            "max_samples": general.get("max_samples"),
            "eval_env": C.EVAL_ENV,
            "dataset_revisions": C.DATASET_REVISIONS,
            "command": command,
        },
        "timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": wall_seconds,
            "lighteval_seconds": float(lighteval_seconds) if lighteval_seconds is not None else None,
        },
        "lighteval_results_file": raw_path,
        "scores": benchmark_scores({t: e["acc_norm"] for t, e in tasks.items() if "acc_norm" in e}),
        "tasks": tasks,
    }


def load_results(path):
    """Parsed results.json if it is a kys-eval results file, else None."""
    try:
        results = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return results if results.get("schema") == RESULTS_SCHEMA else None


def write_summary_md(results, path):
    scores, tasks = results["scores"], results["tasks"]
    env, model, cell = results.get("environment", {}), results.get("model", {}), results.get("cell")
    packages = env.get("packages", {})

    if results["complete"]:
        status = "complete"
    else:
        status = f"INCOMPLETE: {len(results['missing_tasks'])} tasks missing"
        if results["protocol_violations"]:
            status += "; protocol violations: " + "; ".join(results["protocol_violations"])

    lines = [f"# {results['label']}", "", "| | |", "|---|---|", f"| Status | {status} |", f"| Model | `{model.get('spec')}` |"]
    if model.get("commit"):
        lines.append(f"| HF commit | `{model['commit']}` |")
    if model.get("weights_sha256"):
        lines.append("| Weights sha256 | " + ", ".join(f"`{n}` `{h[:16]}…`" for n, h in model["weights_sha256"].items()) + " |")
    if cell:
        lines.append(f"| Grid cell | `{cell['setting']}` seed {cell['seed']} {cell['epoch']} (step {cell['step']}) |")
    gpu = env.get("gpu") or "unknown"
    flag = "" if C.REQUIRED_GPU_SUBSTRING in gpu else f" (v2 used {C.REFERENCE_GPU})"
    lines.append(f"| GPU | {gpu}{flag} |")
    patch = {True: "verified", False: "NOT verified"}.get(env.get("lighteval_patch_ok"), "unknown")
    lines.append(
        f"| LightEval | {packages.get('lighteval', '?')}, recorded sha `{env.get('lighteval_sha_recorded')}`, patch {patch} |"
    )
    if packages:
        lines.append(f"| torch / transformers / datasets | {packages.get('torch')} / {packages.get('transformers')} / {packages.get('datasets')} |")
    timing = results.get("timing", {})
    if timing.get("finished_at"):
        lines.append(f"| Finished | {timing['finished_at']} (LightEval {timing.get('lighteval_seconds') or 0:.0f} s) |")

    lines += ["", "## Scores", "", "Percent, 0-shot, full evaluation splits. `acc_norm` is the reported metric.", ""]
    lines += ["| Benchmark | acc_norm | acc |", "|---|---|---|"]
    for tid, _, label, *_ in C.COMMONSENSE:
        entry = tasks.get(tid, {})
        lines.append(f"| {label} | {pct(entry.get('acc_norm'))} | {pct(entry.get('acc'))} |")
    lines.append(f"| **Mean6** (commonsense average) | **{pct(scores['mean6'])}** | |")
    mmlu_acc = [tasks.get(C.MMLU_TASK_PREFIX + s, {}).get("acc") for s in C.MMLU_SUBSETS]
    mmlu_acc_macro = statistics.fmean(mmlu_acc) if None not in mmlu_acc else None
    lines.append(f"| **MMLU** (57-subject macro-average) | **{pct(scores['mmlu'])}** | {pct(mmlu_acc_macro)} |")

    lines += ["", "### MMLU by category", "", "| " + " | ".join(C.MMLU_CATEGORIES) + " |", "|" + "---|" * len(C.MMLU_CATEGORIES)]
    lines.append("| " + " | ".join(pct(scores[category_key(c)]) for c in C.MMLU_CATEGORIES) + " |")

    lines += ["", "<details><summary>MMLU per subject (acc_norm)</summary>", "", "| Subject | Category | acc_norm |", "|---|---|---|"]
    for subject in C.MMLU_SUBSETS:
        value = tasks.get(C.MMLU_TASK_PREFIX + subject, {}).get("acc_norm")
        lines.append(f"| {subject} | {C.MMLU_SUBJECT_CATEGORY[subject]} | {pct(value)} |")
    lines += ["", "</details>", ""]
    if results["missing_tasks"]:
        lines += ["## Missing tasks", ""] + [f"- `{t}`" for t in results["missing_tasks"]] + [""]
    Path(path).write_text("\n".join(lines))
