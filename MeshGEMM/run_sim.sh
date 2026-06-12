set -e
# Usage: ./run_sim.sh P M K N   (small grids only — sim cost scales with P*P)
P=$1
M=$2
K=$3
N=$4

Mt=$((M / P))
Kt=$((K / P))
Nt=$((N / P))

echo "P=$P, M=$M, K=$K, N=$N, Mt=$Mt, Kt=$Kt, Nt=$Nt"

python compile.py --mode sim --P "$P" --Mt "$Mt" --Kt "$Kt" --Nt "$Nt"
cs_python launch_sim.py --P "$P" --M "$M" --K "$K" --N "$N"
