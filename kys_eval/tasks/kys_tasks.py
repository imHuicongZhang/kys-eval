"""--custom-tasks entry point: rewrite_tasks.py (verbatim, hash-checked) with dataset revisions pinned.

Pinning only sets LightevalTaskConfig.hf_revision; prompts, splits and metrics are untouched, and
LightEval's per-task hash_examples / hash_full_prompts are unaffected.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from kys_eval.config import DATASET_REVISIONS, EXPECTED_TASKS, TASK_FILE_SHA256, TASK_FILE_VERBATIM  # noqa: E402

_digest = hashlib.sha256(TASK_FILE_VERBATIM.read_bytes()).hexdigest()
if _digest != TASK_FILE_SHA256:
    raise RuntimeError(f"{TASK_FILE_VERBATIM} sha256 {_digest} != pinned {TASK_FILE_SHA256}; the task file was modified")

_spec = importlib.util.spec_from_file_location("kys_rewrite_tasks_verbatim", TASK_FILE_VERBATIM)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_expected = set(EXPECTED_TASKS)
for _task in _module.TASKS_TABLE:
    if _task.name in _expected:
        _task.hf_revision = DATASET_REVISIONS[_task.hf_repo]

TASKS_TABLE = _module.TASKS_TABLE
