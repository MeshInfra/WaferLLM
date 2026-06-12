set -e
# Usage: ./run_device.sh P M K N   (drives a real WSE-3 run via the cloud compiler)
P=$1
M=$2
K=$3
N=$4

Mt=$((M / P))
Kt=$((K / P))
Nt=$((N / P))

echo "P=$P, M=$M, K=$K, N=$N, Mt=$Mt, Kt=$Kt, Nt=$Nt"

python compile.py --mode device --P "$P" --Mt "$Mt" --Kt "$Kt" --Nt "$Nt"
python launch_device.py --P "$P" --M "$M" --K "$K" --N "$N"
