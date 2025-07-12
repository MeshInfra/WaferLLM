import sys
import json
import time
from cerebras.sdk.client import SdkCompiler

P = int(sys.argv[1])
Mt = int(sys.argv[2])
Kt = int(sys.argv[3])
Nt = int(sys.argv[4])

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
    f"--params=P:{P},Mt:{Mt},Kt:{Kt},Nt:{Nt} " \
    "--memcpy --channels=2 ",
    "compile_out",
)

# Write the artifact_id to a JSON file
with open(f"{out_path}/artifact_{P}_{Mt}_{Kt}_{Nt}.json", "w", encoding="utf-8") as f:
    json.dump({"artifact_id": artifact_id,}, f)