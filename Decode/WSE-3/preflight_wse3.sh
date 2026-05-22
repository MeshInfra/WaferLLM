#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pick_python_bin() {
    local candidate
    for candidate in "${CB_PYTHON:-}" python python3 cs_python; do
        if [ -z "$candidate" ]; then
            continue
        fi
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi
        if "$candidate" - <<'PY' >/dev/null 2>&1
from cerebras.sdk.client import SdkCompiler, SdkRuntime  # noqa: F401
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType  # noqa: F401
PY
        then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

echo "== Decode/WSE-3 preflight =="
echo "workdir: $SCRIPT_DIR"

echo
echo "[1/6] Checking basic commands..."
for cmd in bash jq; do
    if command -v "$cmd" >/dev/null 2>&1; then
        echo "  ok: $cmd -> $(command -v "$cmd")"
    else
        echo "  missing: $cmd"
        exit 1
    fi
done

echo
echo "[2/6] Checking profiling scripts..."
for file in run_profiling_wse3.sh run_profiling_batch_wse3.sh launch_wse3.py compile.py; do
    if [ -f "$file" ]; then
        echo "  ok: $file"
    else
        echo "  missing: $file"
        exit 1
    fi
done

echo
echo "[3/6] Checking model configs..."
for file in model_config/smoke_p8.json profiling_presets/smoke.txt profiling_presets/llama_p_sweep.txt; do
    if [ -f "$file" ]; then
        echo "  ok: $file"
    else
        echo "  missing: $file"
        exit 1
    fi
done

echo
echo "[4/6] Checking writable directories..."
mkdir -p profiling_runs/preflight_check
touch profiling_runs/preflight_check/.write_test
rm -f profiling_runs/preflight_check/.write_test
echo "  ok: profiling_runs/"

echo
echo "[5/6] Checking Cerebras Python environment..."
if PYTHON_BIN="$(pick_python_bin)"; then
    echo "  ok: using Python interpreter '$PYTHON_BIN'"
else
    echo "  missing: could not find a Python interpreter with Cerebras SDK packages"
    echo "  hint: activate the ALCF/Cerebras environment, or set CB_PYTHON explicitly"
    exit 1
fi

echo
echo "[6/6] Checking Python imports in project scripts..."
"$PYTHON_BIN" -m py_compile launch_wse3.py compile.py
echo "  ok: launch_wse3.py / compile.py syntax"

echo
echo "Preflight passed."
echo "Recommended next step:"
echo "  bash run_profiling_batch_wse3.sh --preset smoke --out profiling_runs/smoke_real"
