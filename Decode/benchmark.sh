set -e
# Standard device config: 8B-4K-256 scale (P=256 -> 65536 PEs).
mkdir -p log
./run_device.sh model_config/llama8B_4k_1_256.json | tee log/device_4k_256.log
