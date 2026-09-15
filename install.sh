#!/usr/bin/env bash
# Build the pinned evaluation environment. Safe to re-run.
#
#   ./install.sh                     # venv in ./.venv, LightEval checkout in ./third_party/lighteval
#   PYTHON=/path/to/python3.11 KYS_VENV=/fast/disk/kys-venv ./install.sh
#
# Steps: create a Python 3.11 venv, install requirements.txt, clone huggingface/lighteval at the
# pinned commit, apply the v2 patch, verify the patched files by sha256, install LightEval without
# dependencies, then run `python -m kys_eval.check_install`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3.11}"
VENV="${KYS_VENV:-$ROOT/.venv}"
LV="${KYS_LIGHTEVAL_DIR:-$ROOT/third_party/lighteval}"
BASE=10b9104eb16778b174e9e240140974a039018981
PATCH="$ROOT/patches/lighteval-10b9104-kys-v2.patch"
SUMS="$ROOT/patches/PATCHED_FILES.sha256"

"$PYTHON" -c 'import sys; v = sys.version_info; sys.exit(0 if v[:2] == (3, 11) else f"need Python 3.11 (v2 used 3.11.9), got {sys.version.split()[0]}; set PYTHON=/path/to/python3.11")'

if [ ! -x "$VENV/bin/python" ]; then
  echo "[install] creating venv $VENV"
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements.txt"

if [ ! -d "$LV/.git" ]; then
  echo "[install] cloning LightEval into $LV"
  mkdir -p "$(dirname "$LV")"
  git clone https://github.com/huggingface/lighteval.git "$LV"
fi

if (cd "$LV" && sha256sum --quiet -c "$SUMS") >/dev/null 2>&1; then
  echo "[install] $LV is already the patched v2 tree"
else
  if ! git -C "$LV" diff --quiet || ! git -C "$LV" diff --cached --quiet; then
    echo "[install] $LV has local changes and is not the patched v2 tree; use a fresh KYS_LIGHTEVAL_DIR" >&2
    exit 1
  fi
  git -C "$LV" fetch --quiet origin || true
  git -C "$LV" checkout --quiet --detach "$BASE"
  git -C "$LV" apply --whitespace=nowarn "$PATCH"
  (cd "$LV" && sha256sum -c "$SUMS")
fi
python -m pip install --no-deps -e "$LV"

cd "$ROOT"
python -m kys_eval.check_install
echo "[install] done. Activate with: source $VENV/bin/activate"
