set -e
fabric_w=$(($1 + 7))
fabric_h=$(($1 + 2))

Mt=$(($2 / $1))
Nt=$(($3 / $1))
group_num=$4

pe_num_group=$(($1 / $4))
root_1st_phase=$((pe_num_group / 2))
root_2nd_phase=$(((($4 / 2) * pe_num_group) + root_1st_phase))

echo "P=$1, M=$2, N=$3, group_num=$4, pe_num_group=$pe_num_group, root_1st_phase=$root_1st_phase, root_2nd_phase=$root_2nd_phase"

python compile.py "$P" "$Mt"  "$Nt" "$group_num" "$pe_num_group" "$root_1st_phase" "$root_2nd_phase"
python launch_device.py --P "$1" --M "$2" --N "$3" --group_num "$group_num"