set -e
# Usage: ./run_device.sh P M N group_num   (drives a real WSE-3 run via the cloud compiler)
P=$1
M=$2
N=$3
G=$4

echo "P=$P, M=$M, N=$N, group_num=$G"

python compile.py --mode device --P "$P" --M "$M" --N "$N" --group_num "$G"
python launch_device.py --P "$P" --M "$M" --N "$N" --group_num "$G"
