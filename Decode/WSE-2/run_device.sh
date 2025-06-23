set -e

# export SINGULARITYENV_SIMFABRIC_DEBUG=router
CONFIG=$1

if [ -z "$CONFIG" ]; then
    CONFIG="config.json"
fi

# if config.json exists
if [ -f $CONFIG ]; then
    echo "Use config values from $CONFIG."
    P=$(jq -r '.P' $CONFIG)
    GROUP_NUM=$(jq -r '.group_num' $CONFIG)
    BSZ=$(jq -r '.bsz' $CONFIG)
    DIM=$(jq -r '.dim' $CONFIG)
    N_HEADS=$(jq -r '.n_heads' $CONFIG)
    N_KV_HEADS=$(jq -r '.n_kv_heads' $CONFIG)
    HEAD_DIM=$(jq -r '.head_dim' $CONFIG)
    SEQ_LEN=$(jq -r '.seq_len' $CONFIG)
    FFN_DIM=$(jq -r '.ffn_dim' $CONFIG)
else
    echo "Use default test values."
    P=8
    GROUP_NUM=2
    BSZ=1
    DIM=64
    N_HEADS=1
    N_KV_HEADS=1
    HEAD_DIM=64
    SEQ_LEN=64
    FFN_DIM=64
fi

FABRIC_W=$(($P + 7))
FABRIC_H=$(($P + 2))

dim_p_pe=$(($DIM / $P))
pes_p_head=$(($P / $N_HEADS))
pes_p_kv_head=$(($P / $N_KV_HEADS))
head_dim_p_pe=$(($HEAD_DIM / $P))
seq_len_p_pe=$(($SEQ_LEN / $P))
ffn_dim_p_pe=$(($FFN_DIM / $P))

pe_num_p_group=$(($P / $GROUP_NUM))
root_1st_phase=$((pe_num_p_group / 2))
root_2nd_phase=$(((($GROUP_NUM / 2) * pe_num_p_group) + root_1st_phase))


echo "P: $P"
echo "BSZ: $BSZ"
echo "DIM: $DIM"
# echo "N_HEADS: $N_HEADS"
# echo "N_KV_HEADS: $N_KV_HEADS"
# echo "HEAD_DIM: $HEAD_DIM"
echo "SEQ_LEN: $SEQ_LEN"
echo "FFN_DIM: $FFN_DIM"

echo "GROUP_NUM: $GROUP_NUM"
echo "PE_NUM_PER_GROUP: $pe_num_p_group"
echo "ROOT_1ST_PHASE: $root_1st_phase"
echo "ROOT_2ND_PHASE: $root_2nd_phase"

python compile.py $P $BSZ $dim_p_pe $pes_p_head $pes_p_kv_head $head_dim_p_pe $seq_len_p_pe $ffn_dim_p_pe $pe_num_p_group $root_1st_phase $root_2nd_phase
python launch_device.py --config $CONFIG