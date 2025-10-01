set -e

P=$1
fabric_w=$(($1 + 7))
fabric_h=$(($1 + 2))

Mt=$(($2 / $1))
Nt=$(($3 / $1))

group_num=$4
pe_num_group=$(($1 / $4))
root_1st_phase=$((pe_num_group / 2))
root_2nd_phase=$(((($4 / 2) * pe_num_group) + root_1st_phase))

simulator=false

if [ -n "$5" ]; then
    simulator=$5
fi

echo "P=$1, M=$2, N=$3, group_num=$4, pe_num_group=$pe_num_group, root_1st_phase=$root_1st_phase, root_2nd_phase=$root_2nd_phase, simulator=$simulator"

python compile.py "$P" "$Mt" "$Nt" "$group_num" "$pe_num_group" "$root_1st_phase" "$root_2nd_phase" "$simulator"

if [ "$simulator" == "true" ]; then
    python launch_wse3.py --P "$1" --M "$2" --N "$3" --group_num "$4" --simulator
else
    python launch_wse3.py --P "$1" --M "$2" --N "$3" --group_num "$4"
fi