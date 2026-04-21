#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<'EOF'
Usage:
  bash run_profiling_batch.sh [--preset PRESET] [--out OUTPUT_DIR] [config1.json config2.json ...]

Examples:
  bash run_profiling_batch.sh --preset smoke
  bash run_profiling_batch.sh --preset llama_p_sweep --out profiling_runs/my_batch
  bash run_profiling_batch.sh model_config/llama8B_4k_1_240.json model_config/llama8B_4k_1_360.json
EOF
}

PRESET=""
OUTPUT_DIR=""
declare -a CONFIGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --out)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            CONFIGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$PRESET" ]]; then
    PRESET_FILE="$SCRIPT_DIR/profiling_presets/${PRESET}.txt"
    if [[ ! -f "$PRESET_FILE" ]]; then
        echo "Preset not found: $PRESET_FILE"
        exit 1
    fi

    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"
        if [[ -n "$line" ]]; then
            CONFIGS+=("$line")
        fi
    done < "$PRESET_FILE"
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    PRESET="smoke"
    PRESET_FILE="$SCRIPT_DIR/profiling_presets/${PRESET}.txt"
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"
        if [[ -n "$line" ]]; then
            CONFIGS+=("$line")
        fi
    done < "$PRESET_FILE"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    batch_name="${PRESET:-custom}"
    timestamp=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="profiling_runs/${batch_name}_batch_${timestamp}"
fi

mkdir -p "$OUTPUT_DIR"

RUNS_TSV="$OUTPUT_DIR/runs.tsv"
RUNS_TXT="$OUTPUT_DIR/runs.txt"
COMPLETED_TSV="$OUTPUT_DIR/completed.tsv"
echo -e "config\tartifact_dir" > "$RUNS_TSV"
echo -e "config\tartifact_dir" > "$COMPLETED_TSV"
: > "$RUNS_TXT"

echo "Running batch profiling..."
echo "Output dir: $OUTPUT_DIR"
echo "Config count: ${#CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    if [[ ! -f "$config" ]]; then
        echo "Config not found: $config"
        exit 1
    fi

    config_name=$(basename "$config" .json)
    artifact_dir="$OUTPUT_DIR/$config_name"

    echo
    echo "============================================================"
    echo "Config: $config"
    echo "Artifact dir: $artifact_dir"
    echo "============================================================"

    echo -e "${config}\t${artifact_dir}" >> "$RUNS_TSV"
    echo "$artifact_dir" >> "$RUNS_TXT"

    bash run_profiling.sh "$config" "$artifact_dir"

    echo -e "${config}\t${artifact_dir}" >> "$COMPLETED_TSV"
done

echo
echo "Batch profiling finished."
echo "Summary file: $RUNS_TSV"
echo "Completed file: $COMPLETED_TSV"
echo "Artifact root: $OUTPUT_DIR"
