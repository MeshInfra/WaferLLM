set -e
# Standard device config: 8B-4K-256 scale (P=256 -> 65536 PEs, M=N=4096, group_num=16).
mkdir -p log
./run_device.sh 256 4096 4096 16 | tee log/device_4k_256.log
