set -e
# Usage: ./run_sim.sh P M N group_num   (small grids only — sim cost scales with P*P)
P=$1
M=$2
N=$3
G=$4

echo "P=$P, M=$M, N=$N, group_num=$G"

python compile.py --mode sim --P "$P" --M "$M" --N "$N" --group_num "$G"
cs_python launch_sim.py --P "$P" --M "$M" --N "$N" --group_num "$G"
