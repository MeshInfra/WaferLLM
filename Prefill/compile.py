"""Unified compile entry for Prefill (config-driven).

  --mode sim     : local `cslc` build into out_<cfg>/
  --mode device  : cloud SdkCompiler build -> compile_out/artifact_<cfg>.json

<cfg> is the config-file basename, so concurrent configs never collide.
"""
import argparse
import json
import os
import subprocess
import time


def derive_params(cj):
    P = cj["P"]
    return {
        "P": P,
        "dim_p_pe": cj["dim"] // P,
        "pes_p_head": P // cj["n_heads"],
        "pes_p_kv_head": P // cj["n_kv_heads"],
        "head_dim_p_pe": cj["head_dim"] // P,
        "seq_len_p_pe": cj["seq_len"] // P,
        "ffn_dim_p_pe": cj["ffn_dim"] // P,
    }


def main():
    ap = argparse.ArgumentParser(description="Compile Prefill (WSE-3, SDK 2.10)")
    ap.add_argument("--mode", choices=["sim", "device"], required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    with open(args.config, encoding="utf-8") as f:
        cj = json.load(f)
    d = derive_params(cj)
    P = d["P"]
    params = ",".join(f"{k}:{v}" for k, v in d.items())

    print("Start compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)

    if args.mode == "sim":
        out_dir = f"out_{cfg_name}"
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
            f"-o out --memcpy --channels=4 --params={params}"
        )
        with SdkCompiler(resource_cpu=48000, resource_mem=64 << 30, disable_version_check=True) as compiler:
            artifact_id = compiler.compile(
                app_path="src", csl_main="layout.csl", options=options, out_path="compile_out",
            )
        with open(f"compile_out/artifact_{cfg_name}.json", "w", encoding="utf-8") as f:
            json.dump({"artifact_id": artifact_id}, f)

    print("End compiling: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)


if __name__ == "__main__":
    main()
