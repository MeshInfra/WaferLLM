"""MeshGEMV simulator launcher (cs_python, local out_<cfg> artifact)."""
import argparse
import os
import random

import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

import host_common as hc


def parse_args():
    parser = argparse.ArgumentParser(description="MeshGEMV on simulator")
    parser.add_argument("--P", required=True, type=int, help="PEs rectangle size: P x P")
    parser.add_argument("--M", required=True, type=int, help="Left vector dimension: 1 x M")
    parser.add_argument("--N", required=True, type=int, help="Right matrix dimension: M x N")
    parser.add_argument("--group_num", required=True, type=int, help="Reduce group count")
    return parser.parse_args()


def main():
    random.seed(2025)
    np.random.seed(2025)

    args = parse_args()
    P, M, N, G = args.P, args.M, args.N, args.group_num
    Mt, Nt = M // P, N // P
    cfg = hc.cfg_name(P, M, N, G)

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    X, tensor_X, tensor_W = hc.make_inputs(P, M, N)

    # Run from inside out_<cfg> so simfab run artifacts (sim.log, sim_stats.json,
    # simconfig.json, simfab_traces/, out.core, wio_flows_tmpdir.*) land there
    # instead of polluting the module root. SdkRuntime has no output-dir kwarg;
    # it writes these to CWD. compile.py already populated out_<cfg> via `cslc -o`.
    out_dir = os.path.abspath(f"out_{cfg}")
    if not os.path.isdir(out_dir):
        raise SystemExit(f"Host: {out_dir} not found — run compile.py --mode sim first")
    os.chdir(out_dir)
    runner = SdkRuntime(out_dir)
    runner.load()
    runner.run()

    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")
    sym_res = runner.get_id("res")
    symbol_time_memcpy = runner.get_id("time_memcpy")
    symbol_time_ref = runner.get_id("time_ref")

    X_u32 = input_array_to_u32(tensor_X.ravel(), 1, 1)
    runner.memcpy_h2d(sym_X, X_u32, 0, 0, P, P, Mt,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    W_u32 = input_array_to_u32(hc.tile_W(tensor_W, P, Mt, Nt), 1, 1)
    runner.memcpy_h2d(sym_W, W_u32, 0, 0, P, P, Mt * Nt,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.launch("init_task", nonblock=False)
    total_warmup_times, total_repeat_times = 1, 10
    runner.launch("meshgemv_host", np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)

    res_1d_u32 = np.zeros(P * N, dtype=np.uint32)
    runner.memcpy_d2h(res_1d_u32, sym_res, 0, 0, P, P, Nt,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
    res_1d_fp16 = memcpy_view(res_1d_u32, np.dtype(np.float16))
    res = res_1d_fp16.reshape(P, N)

    time_memcpy_1d_f32 = np.zeros(P * P * 3, dtype=np.float32)
    runner.memcpy_d2h(time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, P, P, 3, streaming=False,
                      order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order="C")

    time_ref_1d_f32 = np.zeros(P * P * 2, np.float32)
    runner.memcpy_d2h(time_ref_1d_f32, symbol_time_ref, 0, 0, P, P, 2, streaming=False,
                      order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order="C")

    runner.stop()

    expected = np.matmul(X.astype(np.float32), tensor_W.astype(np.float32))  # (1, N)
    actual = res.astype(np.float32)  # (P, N) — every row is the broadcast result
    exp_b = np.broadcast_to(expected, actual.shape)
    rel_err = np.abs(actual - exp_b) / (np.abs(exp_b) + 1e-3)

    mean_cycle, max_cycle = hc.decode_timing(time_memcpy_hwl, time_ref_hwl, P, total_repeat_times)
    print(f"P: {P}, M: {M}, N: {N}")
    print(f"Mean cycle count: {mean_cycle}")
    print(f"Max Cycle count: {max_cycle}")
    print(f"max rel err: {rel_err.max():.4f}")

    assert rel_err.max() < 0.1, f"result mismatch: max rel err {rel_err.max()}"
    print("SUCCESS!")


if __name__ == "__main__":
    main()
