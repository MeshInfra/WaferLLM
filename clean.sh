#!/usr/bin/env bash
# Remove all build / run artifacts across every module.
# Compile/run scripts recreate the directories they need, so wiping everything
# below is safe. This script ALWAYS lists what it will remove first.
#
#   ./clean.sh         list targets, then ask for confirmation before deleting
#   ./clean.sh -n      dry-run: list targets and exit without deleting
#   ./clean.sh -y      skip the confirmation prompt (for CI / scripts)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run) DRY_RUN=1 ;;
    -y|--yes)     ASSUME_YES=1 ;;
    -h|--help)    echo "usage: clean.sh [-n|--dry-run] [-y|--yes]"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; echo "usage: clean.sh [-n|--dry-run] [-y|--yes]" >&2; exit 2 ;;
  esac
done

# Per-config sim build dirs, device staging, sim traces, wio flows, logs.
# -prune so we list/delete the dir itself, not its contents.
find_dirs() {
  find . -path '*/.git/*' -prune -o \
    -type d \( \
        -name 'out' -o -name 'out_*' -o -name 'device_staging_*' \
        -o -name 'simfab_traces' -o -name 'wio_flows' -o -name 'wio_flows_tmpdir.*' \
        -o -name 'compile_out' -o -name 'log' -o -name '__pycache__' \
    \) -prune -print
}

# Stray per-run files in module roots. Prune the artifact dirs above first, so
# files already covered by find_dirs are not listed (or deleted) twice.
find_files() {
  find . -path '*/.git/*' -prune -o \
    -type d \( \
        -name 'out' -o -name 'out_*' -o -name 'device_staging_*' \
        -o -name 'simfab_traces' -o -name 'wio_flows' -o -name 'wio_flows_tmpdir.*' \
        -o -name 'compile_out' -o -name 'log' -o -name '__pycache__' \
    \) -prune -o \
    -type f \( \
        -name '*.log' -o -name 'sim_stats.json' -o -name 'simconfig.json' \
        -o -name 'wio_flow.json' -o -name 'artifact_*.json' \
        -o -name 'wsjob-*.json' -o -name 'run_meta.json' \
    \) -print
}

mapfile -t DIRS  < <(find_dirs  | sort)
mapfile -t FILES < <(find_files | sort)

if [ "$(( ${#DIRS[@]} + ${#FILES[@]} ))" -eq 0 ]; then
  echo "Nothing to clean under $ROOT."
  exit 0
fi

echo "The following will be removed under $ROOT:"
for d in "${DIRS[@]}";  do echo "  [dir]  $d"; done
for f in "${FILES[@]}"; do echo "  [file] $f"; done
echo "Total: ${#DIRS[@]} dir(s), ${#FILES[@]} file(s)."

if [ "$DRY_RUN" -eq 1 ]; then
  echo "(dry-run: nothing deleted)"
  exit 0
fi

if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "Proceed with deletion? [y/N] " ans || ans=""
  case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

if [ "${#DIRS[@]}"  -gt 0 ]; then printf '%s\0' "${DIRS[@]}"  | xargs -0 rm -rf; fi
if [ "${#FILES[@]}" -gt 0 ]; then printf '%s\0' "${FILES[@]}" | xargs -0 rm -f;  fi
echo "Clean complete."
