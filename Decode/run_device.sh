set -e
# Usage: ./run_device.sh <config.json>   (drives a real WSE-3 run via the cloud compiler)
CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "usage: ./run_device.sh <config.json>"
    exit 1
fi

python compile.py --mode device --config "$CONFIG"
python launch_device.py --config "$CONFIG"
