import sys
import json
import time
from cerebras.sdk.client import SdkCompiler

P = int(sys.argv[1])
Mt = int(sys.argv[2])
Nt = int(sys.argv[3])

group_num = int(sys.argv[4])
pe_num_group = int(sys.argv[5])

root_1st_phase = int(sys.argv[6])
root_2nd_phase = int(sys.argv[7])

out_path = "compile_out"

print("Start compiling: "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

# Instantiate compiler
compiler = SdkCompiler()

# Launch compile job
artifact_id = compiler.compile(
    "src",
    "layout.csl",
    # f"--fabric-dims={W+7},{W+2} --fabric-offsets=4,1 " \
    "--fabric-dims=757,996 --fabric-offsets=4,1 " \
    f"--params=P:{P},Mt:{Mt},Nt:{Nt},pe_num_group:{pe_num_group},root_1st_phase:{root_1st_phase},root_2nd_phase:{root_2nd_phase} " \
    "--memcpy --channels=1 ",
    "compile_out",
)

# Write the artifact_id to a JSON file
with open(f"{out_path}/artifact_{P}_{Mt}_{Nt}_{group_num}.json", "w", encoding="utf-8") as f:
    json.dump({"artifact_id": artifact_id,}, f)