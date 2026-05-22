#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-config.json}"
ARTIFACT_DIR="${2:-}"

if [ -z "$ARTIFACT_DIR" ]; then
    config_name=$(basename "$CONFIG" .json)
    timestamp=$(date +"%Y%m%d_%H%M%S")
    ARTIFACT_DIR="profiling_runs/${config_name}_profiling_${timestamp}"
fi

echo "Running Decode/WSE-3 profiling..."
echo "Config: $CONFIG"
echo "Artifacts: $ARTIFACT_DIR"

bash run_sim.sh "$CONFIG" "$ARTIFACT_DIR"

echo
echo "Profiling finished."
echo "Artifact dir: $ARTIFACT_DIR"
echo "Open decode_profiling.ipynb and set:"
echo "artifact_dir = Path(\"$ARTIFACT_DIR\")"
