set -e
# Usage: ./run_sim.sh <config.json>   (small configs only — sim cost scales with P*P)
CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "usage: ./run_sim.sh <config.json>"
    exit 1
fi

python compile.py --mode sim --config "$CONFIG"
cs_python launch_sim.py --config "$CONFIG"
