import os
import signal
import sys
import json
import time
from cerebras.sdk.client import SdkCompiler

P = int(sys.argv[1])
bsz = int(sys.argv[2])
dim_p_pe = int(sys.argv[3])
pes_p_head = int(sys.argv[4])
pes_p_kv_head = int(sys.argv[5])
head_dim_p_pe = int(sys.argv[6])
seq_len_p_pe = int(sys.argv[7])
ffn_dim_p_pe = int(sys.argv[8])

pe_num_p_group = int(sys.argv[9])
root_1st_phase = int(sys.argv[10])
root_2nd_phase = int(sys.argv[11])
simulator = sys.argv[12].lower() == "true"

out_path = "compile_out"

print("Start compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

if simulator:
    ARGS=f"--arch=wse3 --fabric-dims={P+7},{P+2} --fabric-offsets=4,1 -o out --memcpy --channels=1 --params=P:{P},bsz:{bsz},dim_p_pe:{dim_p_pe},pes_p_head:{pes_p_head},pes_p_kv_head:{pes_p_kv_head},head_dim_p_pe:{head_dim_p_pe},seq_len_p_pe:{seq_len_p_pe},ffn_dim_p_pe:{ffn_dim_p_pe},pe_num_p_group:{pe_num_p_group},root_1st_phase:{root_1st_phase},root_2nd_phase:{root_2nd_phase}"
else:
    ARGS=f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 -o out --memcpy --channels=4 --params=P:{P},bsz:{bsz},dim_p_pe:{dim_p_pe},pes_p_head:{pes_p_head},pes_p_kv_head:{pes_p_kv_head},head_dim_p_pe:{head_dim_p_pe},seq_len_p_pe:{seq_len_p_pe},ffn_dim_p_pe:{ffn_dim_p_pe},pe_num_p_group:{pe_num_p_group},root_1st_phase:{root_1st_phase},root_2nd_phase:{root_2nd_phase}"

# Instantiate copmiler
with SdkCompiler(resource_cpu=48000, resource_mem=64<<30) as compiler:

    artifact_id = compiler.compile(
        app_path="src",
        csl_main="layout.csl",
        options=ARGS,
        out_path=out_path,
    )

    # Write the artifact_id to a JSON file
    with open(f"{out_path}/artifact_{P}_{P//pe_num_p_group}.json", "w", encoding="utf-8") as f:
        json.dump({"artifact_id": artifact_id,}, f)

print("End compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)