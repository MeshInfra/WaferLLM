import sys
import json
import time
from cerebras.sdk.client import SdkCompiler

P = int(sys.argv[1])
dim_p_pe = int(sys.argv[2])
pes_p_head = int(sys.argv[3])
pes_p_kv_head = int(sys.argv[4])
head_dim_p_pe = int(sys.argv[5])
seq_len_p_pe = int(sys.argv[6])
ffn_dim_p_pe = int(sys.argv[7])

out_path = "compile_out"

print("Start compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

# Instantiate compiler
compiler = SdkCompiler()

# Launch compile job
artifact_id = compiler.compile(
    "src",
    "layout.csl",
    # f"--fabric-dims={P+7},{P+2} --fabric-offsets=4,1 " \
    "--fabric-dims=757,996 --fabric-offsets=4,1 " \
    f"--params=P:{P},dim_p_pe:{dim_p_pe},pes_p_head:{pes_p_head},pes_p_kv_head:{pes_p_kv_head},head_dim_p_pe:{head_dim_p_pe},seq_len_p_pe:{seq_len_p_pe},ffn_dim_p_pe:{ffn_dim_p_pe} " \
    "--memcpy --channels=1 ",
    "compile_out",
)

# Write the artifact_id to a JSON file
with open(f"{out_path}/artifact_{P}.json", "w", encoding="utf-8") as f:
    json.dump({"artifact_id": artifact_id,}, f)