#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config.json}"
SIMULATOR="${2:-false}"
ARTIFACT_DIR="${3:-}"

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

if ! PYTHON_BIN="$(pick_python_bin)"; then
    echo "Could not find a Python interpreter with the required Cerebras SDK packages."
    echo "Please run this script in the ALCF/Cerebras software environment, or set CB_PYTHON explicitly."
    exit 1
fi

if [ -z "$ARTIFACT_DIR" ]; then
    config_name=$(basename "$CONFIG" .json)
    target_name="wse3"
    if [ "$SIMULATOR" = "true" ]; then
        target_name="appliance_sim"
    fi
    timestamp=$(date +"%Y%m%d_%H%M%S")
    ARTIFACT_DIR="profiling_runs/${config_name}_${target_name}_profiling_${timestamp}"
fi

mkdir -p "$ARTIFACT_DIR"

if [ -f "$CONFIG" ]; then
    echo "Use config values from $CONFIG."
    P=$(jq -r '.P' "$CONFIG")
    GROUP_NUM=$(jq -r '.group_num' "$CONFIG")
    BSZ=$(jq -r '.bsz' "$CONFIG")
    DIM=$(jq -r '.dim' "$CONFIG")
    N_HEADS=$(jq -r '.n_heads' "$CONFIG")
    N_KV_HEADS=$(jq -r '.n_kv_heads' "$CONFIG")
    HEAD_DIM=$(jq -r '.head_dim' "$CONFIG")
    SEQ_LEN=$(jq -r '.seq_len' "$CONFIG")
    FFN_DIM=$(jq -r '.ffn_dim' "$CONFIG")
else
    echo "Use default test values."
    P=8
    GROUP_NUM=2
    BSZ=1
    DIM=64
    N_HEADS=1
    N_KV_HEADS=1
    HEAD_DIM=64
    SEQ_LEN=64
    FFN_DIM=64
fi

dim_p_pe=$(($DIM / $P))
pes_p_head=$(($P / $N_HEADS))
pes_p_kv_head=$(($P / $N_KV_HEADS))
head_dim_p_pe=$(($HEAD_DIM / $P))
seq_len_p_pe=$(($SEQ_LEN / $P))
ffn_dim_p_pe=$(($FFN_DIM / $P))
pe_num_p_group=$(($P / $GROUP_NUM))

root_1st_phase=$((pe_num_p_group / 2))
root_2nd_phase=$(((($GROUP_NUM / 2) * pe_num_p_group) + root_1st_phase))

echo "Running Decode/WSE-3 profiling..."
echo "Config: $CONFIG"
echo "Simulator: $SIMULATOR"
echo "Artifacts: $ARTIFACT_DIR"

"$PYTHON_BIN" compile.py "$P" "$BSZ" "$dim_p_pe" "$pes_p_head" "$pes_p_kv_head" "$head_dim_p_pe" "$seq_len_p_pe" "$ffn_dim_p_pe" "$pe_num_p_group" "$root_1st_phase" "$root_2nd_phase" "$SIMULATOR"

if [ "$SIMULATOR" = "true" ]; then
    "$PYTHON_BIN" launch_wse3.py --config "$CONFIG" --simulator --artifact-dir "$ARTIFACT_DIR"
else
    "$PYTHON_BIN" launch_wse3.py --config "$CONFIG" --artifact-dir "$ARTIFACT_DIR"
fi

if [ -d simfab_traces ]; then
    mv simfab_traces "$ARTIFACT_DIR"/
fi

for file in wsjob-*.json run_meta.json; do
    if [ -e "$file" ]; then
        mv "$file" "$ARTIFACT_DIR"/
    fi
done

rm -rf wio_flows_tmpdir.*

echo
echo "Profiling finished."
echo "Artifact dir: $ARTIFACT_DIR"
echo "Open decode_profiling.ipynb and set:"
echo "artifact_dir = Path(\"$ARTIFACT_DIR\")"
