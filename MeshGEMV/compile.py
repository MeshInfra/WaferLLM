"""Unified compile entry for MeshGEMV.

  --mode sim     : local `cslc` build into out_<cfg>/
  --mode device  : cloud SdkCompiler build -> compile_out/artifact_<cfg>.json

The 2-tree all-reduce parameters (pe_num_group / root_1st_phase / root_2nd_phase)
are derived here from P / M / N / group_num so callers only pass the high-level dims.
"""
import argparse
import json
import os
import subprocess
import time


def parse_args():
    ap = argparse.ArgumentParser(description="Compile MeshGEMV (WSE-3, SDK 2.10)")
    ap.add_argument("--mode", choices=["sim", "device"], required=True)
    ap.add_argument("--P", type=int, required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--group_num", type=int, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    P, M, N, G = args.P, args.M, args.N, args.group_num
    Mt = M // P
    Nt = N // P
    pe_num_group = P // G
    root_1st_phase = pe_num_group // 2
    root_2nd_phase = (G // 2) * pe_num_group + root_1st_phase
    cfg = f"{P}_{Mt}_{Nt}_{G}"
    params = (
        f"P:{P},Mt:{Mt},Nt:{Nt},pe_num_group:{pe_num_group},"
        f"root_1st_phase:{root_1st_phase},root_2nd_phase:{root_2nd_phase}"
    )

    print("Start compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

    if args.mode == "sim":
        out_dir = f"out_{cfg}"
        subprocess.run(["rm", "-rf", out_dir], check=True)
        cmd = [
            "cslc", "--arch=wse3", "./src/layout.csl",
            f"--fabric-dims={P + 7},{P + 2}", "--fabric-offsets=4,1",
            f"--params={params}", "-o", out_dir, "--memcpy", "--channels", "1",
        ]
        subprocess.run(cmd, check=True)
    else:
        from cerebras.sdk.client import SdkCompiler
        os.makedirs("compile_out", exist_ok=True)
        options = (
            f"--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 "
            f"-o out --memcpy --channels=1 --params={params}"
        )
        with SdkCompiler(resource_cpu=48000, resource_mem=64 << 30, disable_version_check=True) as compiler:
            artifact_id = compiler.compile(
                app_path="src", csl_main="layout.csl", options=options, out_path="compile_out",
            )
        with open(f"compile_out/artifact_{cfg}.json", "w", encoding="utf-8") as f:
            json.dump({"artifact_id": artifact_id}, f)

    print("End compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)


if __name__ == "__main__":
    main()
