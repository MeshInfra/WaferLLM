#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG=${CONFIG:-model_config/offchip_h2d_p32_fit.json}
SIMULATOR=${SIMULATOR:-false}
SAMPLES=${SAMPLES:-10}
NONBLOCK=${NONBLOCK:-false}
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-h2d_bench_runs}

PAYLOAD_BYTES=${PAYLOAD_BYTES:-4KiB,8KiB,16KiB,32KiB,64KiB,128KiB,256KiB,512KiB,1MiB,2MiB,4MiB}
CACHE_SYMBOL_PAYLOAD_BYTES=${CACHE_SYMBOL_PAYLOAD_BYTES:-4MiB}
CACHE_BLOCK_BYTES=${CACHE_BLOCK_BYTES:-8MiB,64MiB,144MiB,288MiB}

ARTIFACT_PATH=${ARTIFACT_PATH:-}
CMADDR=${CMADDR:-}

RUN_ID=${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}
PAYLOAD_OUT="${OUTPUT_ROOT}/h2d_${RUN_ID}_payload_sweep"
CACHE_OUT="${OUTPUT_ROOT}/h2d_${RUN_ID}_cache_blocks"

mkdir -p "$OUTPUT_ROOT"

run_launcher() {
    local output_dir=$1
    shift

    if [[ -n "$ARTIFACT_PATH" ]]; then
        if [[ "$SIMULATOR" != "true" && -z "$CMADDR" ]]; then
            echo "ERROR: ARTIFACT_PATH real-WSE runs require CMADDR=<IP:port>." >&2
            exit 2
        fi

        local cmd=(
            "$PYTHON_BIN" launch_wse3.py
            --config "$CONFIG"
            --artifact-path "$ARTIFACT_PATH"
            --h2d-output-dir "$output_dir"
        )
        if [[ "$SIMULATOR" == "true" ]]; then
            cmd+=(--simulator)
        fi
        if [[ -n "$CMADDR" ]]; then
            cmd+=(--cmaddr "$CMADDR")
        fi
        cmd+=("$@")
        "${cmd[@]}"
    else
        local cmd=(bash ./run_wse3.sh "$CONFIG" "$SIMULATOR" --h2d-output-dir "$output_dir")
        if [[ -n "$CMADDR" ]]; then
            cmd+=(--cmaddr "$CMADDR")
        fi
        cmd+=("$@")
        "${cmd[@]}"
    fi
}

echo "=== H2D payload sweep ==="
run_launcher "$PAYLOAD_OUT" \
    --h2d-bench \
    --h2d-preset payload-sweep \
    --h2d-symbols XKCache,XVCache \
    --h2d-payload-bytes "$PAYLOAD_BYTES" \
    --h2d-loop-counts 1 \
    --h2d-nonblock "$NONBLOCK" \
    --h2d-samples "$SAMPLES"

echo "=== H2D cache block sweep ==="
run_launcher "$CACHE_OUT" \
    --h2d-bench \
    --h2d-preset cache-blocks \
    --h2d-symbols XKCache,XVCache \
    --h2d-cache-symbol-payload-bytes "$CACHE_SYMBOL_PAYLOAD_BYTES" \
    --h2d-cache-block-bytes "$CACHE_BLOCK_BYTES" \
    --h2d-nonblock "$NONBLOCK" \
    --h2d-samples "$SAMPLES"

echo "=== Done ==="
echo "Payload sweep summary: ${PAYLOAD_OUT}/h2d_bench_summary.csv"
echo "Cache block summary:   ${CACHE_OUT}/h2d_bench_summary.csv"
