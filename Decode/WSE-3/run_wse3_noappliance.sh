set -e

CONFIG=$1

if [ -z "$CONFIG" ]; then
    CONFIG="config.json"
fi

simulator=false

if [ -n "$2" ]; then
    simulator=$2
fi

CSV_OUTPUT=${3:-}

# if config.json exists
if [ -f $CONFIG ]; then
    echo "Use config values from $CONFIG."
    _P=$(jq -r '.P // empty' $CONFIG)
    _Pw=$(jq -r '.Pw // empty' $CONFIG)
    _Ph=$(jq -r '.Ph // empty' $CONFIG)
    # fallback if Pw and Ph is not given
    if [ -z "$_Pw" ] || [ "$_Pw" = "null" ]; then Pw=$_P; else Pw=$_Pw; fi
    if [ -z "$_Ph" ] || [ "$_Ph" = "null" ]; then Ph=$_P; else Ph=$_Ph; fi
    pe_num_p_h_group=$(jq -r '.pe_num_p_h_group' $CONFIG)
    pe_num_p_v_group=$(jq -r '.pe_num_p_v_group' $CONFIG)
    pe_num_p_group_in_head=$(jq -r '.pe_num_p_group_in_head' $CONFIG)
    BSZ=$(jq -r '.bsz' $CONFIG)
    DIM=$(jq -r '.dim' $CONFIG)
    N_HEADS=$(jq -r '.n_heads' $CONFIG)
    N_KV_HEADS=$(jq -r '.n_kv_heads' $CONFIG)
    HEAD_DIM=$(jq -r '.head_dim' $CONFIG)
    SEQ_LEN=$(jq -r '.seq_len' $CONFIG)
    FFN_DIM=$(jq -r '.ffn_dim' $CONFIG)
    # opt-layout pre-computed per-PE tile sizes (ceil-padded; may not divide DIM/SEQ_LEN/FFN_DIM evenly)
    _v_dim_p_pe=$(jq -r '.v_dim_p_pe // empty' $CONFIG)
    _h_dim_p_pe=$(jq -r '.h_dim_p_pe // empty' $CONFIG)
    _seq_len_p_pe=$(jq -r '.seq_len_p_pe // empty' $CONFIG)
    _ffn_dim_p_pe=$(jq -r '.ffn_dim_p_pe // empty' $CONFIG)
else
    echo "Use default test values."
    Pw=16
    Ph=8
    pe_num_p_h_group=4
    pe_num_p_v_group=4
    pe_num_p_group_in_head=4
    BSZ=2
    DIM=64
    N_HEADS=2
    N_KV_HEADS=1
    HEAD_DIM=32
    SEQ_LEN=96
    FFN_DIM=128
    _v_dim_p_pe=""
    _h_dim_p_pe=""
    _seq_len_p_pe=""
    _ffn_dim_p_pe=""
fi

if [ "$simulator" == "true" ]; then
    FABRIC_W=$(($Pw + 7))
    FABRIC_H=$(($Ph + 2))
else
    FABRIC_W=762
    FABRIC_H=1172
fi

# Use JSON-specified tile sizes if present; otherwise compute ceil-division
if [ -n "$_v_dim_p_pe" ] && [ "$_v_dim_p_pe" != "null" ]; then
    v_dim_p_pe=$_v_dim_p_pe
else
    v_dim_p_pe=$(( ($DIM + $Ph - 1) / $Ph ))
fi

if [ -n "$_h_dim_p_pe" ] && [ "$_h_dim_p_pe" != "null" ]; then
    h_dim_p_pe=$_h_dim_p_pe
else
    h_dim_p_pe=$(( ($DIM + $Pw - 1) / $Pw ))
fi

pes_p_head=$(($Pw / $N_HEADS))
pes_p_kv_head=$(($Pw / $N_KV_HEADS))

if [ -n "$_seq_len_p_pe" ] && [ "$_seq_len_p_pe" != "null" ]; then
    seq_len_p_pe=$_seq_len_p_pe
else
    seq_len_p_pe=$(( ($SEQ_LEN + $Ph - 1) / $Ph ))
fi

if [ -n "$_ffn_dim_p_pe" ] && [ "$_ffn_dim_p_pe" != "null" ]; then
    ffn_dim_p_pe=$_ffn_dim_p_pe
else
    ffn_dim_p_pe=$(( ($FFN_DIM + $Pw - 1) / $Pw ))
fi

echo "Pw: $Pw, Ph: $Ph"
echo "BSZ: $BSZ"
echo "DIM: $DIM, v_dim_p_pe: $v_dim_p_pe, h_dim_p_pe: $h_dim_p_pe"
echo "N_HEADS: $N_HEADS, N_KV_HEADS: $N_KV_HEADS"
echo "HEAD_DIM: $HEAD_DIM"
echo "SEQ_LEN: $SEQ_LEN, seq_len_p_pe: $seq_len_p_pe"
echo "FFN_DIM: $FFN_DIM, ffn_dim_p_pe: $ffn_dim_p_pe"
echo "pe_num_p_h_group: $pe_num_p_h_group, pe_num_p_v_group: $pe_num_p_v_group"
echo "pe_num_p_group_in_head: $pe_num_p_group_in_head"

echo "Simulator: $simulator"

EXEC=""
cs_python="cs_python"

$EXEC cslc --arch=wse3 ./src/layout.csl --fabric-dims="$FABRIC_W","$FABRIC_H" --fabric-offsets=4,1 \
    --params=Pw:"$Pw",Ph:"$Ph",bsz:"$BSZ",v_dim_p_pe:"$v_dim_p_pe",h_dim_p_pe:"$h_dim_p_pe",pes_p_head:"$pes_p_head",pes_p_kv_head:"$pes_p_kv_head",head_dim:"$HEAD_DIM",seq_len_p_pe:"$seq_len_p_pe",ffn_dim_p_pe:"$ffn_dim_p_pe",pe_num_p_h_group:"$pe_num_p_h_group",pe_num_p_v_group:"$pe_num_p_v_group",pe_num_p_group_in_head:"$pe_num_p_group_in_head"\
    -o out --memcpy --channels 1

if [ "$simulator" == "true" ]; then
    if [ -n "$CSV_OUTPUT" ]; then
        $cs_python launch_wse3.py --config $CONFIG --simulator --csv-output "${CSV_OUTPUT}"
    else
        $cs_python launch_wse3.py --config $CONFIG --simulator
    fi
else
    if [ -n "$CSV_OUTPUT" ]; then
        $cs_python launch_wse3.py --config $CONFIG  --csv-output "${CSV_OUTPUT}"
    else
        $cs_python launch_wse3.py --config $CONFIG
    fi
fi

rm -rf simfab_traces
rm -rf wio_flows_tmpdir.*