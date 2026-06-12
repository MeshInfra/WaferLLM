"""MeshGEMM device launcher (appliance python, cloud artifact_<cfg>)."""
import argparse
import json
import random
import time

import numpy as np

from cerebras.sdk.client import SdkRuntime, sdk_utils
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

import host_common as hc

out_path = "compile_out"


def cast_tensor_u32(tensor):
    return np.uint32(tensor.view(np.uint16))


def parse_args():
    parser = argparse.ArgumentParser(description="MeshGEMM on WSE-3")
    parser.add_argument("--P", required=True, type=int, help="PEs rectangle size: P x P")
    parser.add_argument("--M", required=True, type=int, help="Rows of X")
    parser.add_argument("--K", required=True, type=int, help="Inner dimension")
    parser.add_argument("--N", required=True, type=int, help="Columns of W")
    parser.add_argument("--simulator", action="store_true", help="Run the cloud artifact on the simulator")
    return parser.parse_args()


def main():
    random.seed(2025)
    np.random.seed(2025)

    args = parse_args()
    P, M, K, N = args.P, args.M, args.K, args.N
    Mt, Kt, Nt = M // P, K // P, N // P
    cfg = hc.cfg_name(P, M, K, N)

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    tensor_X, tensor_W, tensor_W_offset = hc.make_inputs(P, M, K, N)

    with open(f"{out_path}/artifact_{cfg}.json", "r", encoding="utf-8") as f:
        artifact_id = json.load(f)["artifact_id"]

    print("Start running: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), flush=True)
    print(f"Running on simulator: {args.simulator}")
    with SdkRuntime(artifact_id, simulator=args.simulator, disable_version_check=True) as runner:
        sym_X = runner.get_id("X")
        sym_W = runner.get_id("W")
        sym_res = runner.get_id("res")
        symbol_time_memcpy = runner.get_id("time_memcpy")
        symbol_time_ref = runner.get_id("time_ref")

        X_u32 = cast_tensor_u32(hc.tile_X(tensor_X, P, Mt, Kt))
        runner.memcpy_h2d(sym_X, X_u32, 0, 0, P, P, Mt * Kt,
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

        W_u32 = cast_tensor_u32(hc.tile_W(tensor_W_offset, P, Kt, Nt))
        runner.memcpy_h2d(sym_W, W_u32, 0, 0, P, P, Kt * Nt,
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

        runner.launch("init_task", nonblock=False)
        runner.launch("meshgemm_host", np.int16(0), np.int16(1), nonblock=False)

        res_1d_u32 = np.zeros(M * N, dtype=np.uint32)
        runner.memcpy_d2h(res_1d_u32, sym_res, 0, 0, P, P, Mt * Nt,
                          streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
        res_1d_fp16 = sdk_utils.memcpy_view(res_1d_u32, np.dtype(np.float16))
        res = hc.untile_res(res_1d_fp16, P, Mt, Nt)

        runner.launch("init_task", nonblock=False)
        total_warmup_times, total_repeat_times = 5, 50
        runner.launch("meshgemm_host", np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)

        time_memcpy_1d_f32 = np.zeros(P * P * 3, dtype=np.float32)
        runner.memcpy_d2h(time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, P, P, 3, streaming=False,
                          order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
        time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order="C")

        time_ref_1d_f32 = np.zeros(P * P * 2, np.float32)
        runner.memcpy_d2h(time_ref_1d_f32, symbol_time_ref, 0, 0, P, P, 2, streaming=False,
                          order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
        time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order="C")

    expected = np.matmul(tensor_X.astype(np.float32), tensor_W.astype(np.float32))
    actual = res.astype(np.float32)
    rel_err = np.abs(actual - expected) / (np.abs(expected) + 1e-3)

    mean_cycle, max_cycle = hc.decode_timing(time_memcpy_hwl, time_ref_hwl, P, total_repeat_times)
    print(f"P: {P}, M: {M}, K: {K}, N: {N}")
    print(f"Mean cycle count: {mean_cycle}")
    print(f"Max Cycle count: {max_cycle}")
    print(f"max rel err: {rel_err.max():.4f}")

    freq_ghz = 1.1
    time_cost = max_cycle / (freq_ghz * 1e6)
    print(f"Time: {time_cost} ms")


if __name__ == "__main__":
    main()
