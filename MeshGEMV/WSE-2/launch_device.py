import sys
import random
import time
import numpy as np
import argparse
import json
import struct

from cerebras.sdk.client import SdkRuntime, sdk_utils
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

out_path = "compile_out"

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

def cast_tensor_u32(tensor):
    return np.uint32(tensor.view(np.uint16))

def parse_args():
    
    parser = argparse.ArgumentParser(description="MeshGEMV on WSE-2")
    parser.add_argument("--P", required=True, type=int, help="PEs rectangle size: P x P")
    parser.add_argument("--M", required=True, type=int, help="Left vector dimension: 1 x M")
    parser.add_argument("--N", required=True, type=int, help="Right matrix dimension: M x N")
    parser.add_argument("--group_num", required=True, type=int, help="Group number")
    
    args = parser.parse_args()
    return args

def main():
    random.seed(2025)
    
    args = parse_args()
    
    P = args.P
    M = args.M
    N = args.N
    
    Mt = M // P
    Nt = N // P
    
    group_num = args.group_num
    
    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR
    
    X = np.random.rand(1, M).astype(np.float16)
    tensor_X = np.tile(X.reshape(P, Mt), reps=(1, P))
    
    tensor_W = np.random.rand(M, N).astype(np.float16)
    
    with open(f"{out_path}/artifact_{P}_{Mt}_{Nt}_{group_num}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        artifact_id = data["artifact_id"]
        
    with SdkRuntime(artifact_id, simulator=False) as runner:
        
        sym_X = runner.get_id("X")
        sym_W = runner.get_id("W")
        
        sym_res = runner.get_id("res")
        
        symbol_time_memcpy = runner.get_id("time_memcpy")
        symbol_time_ref = runner.get_id("time_ref")
        
        X_u32 = cast_tensor_u32(tensor_X.ravel())
        runner.memcpy_h2d(sym_X, X_u32, 0, 0, P, P, Mt, \
                        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
        
        W1 = tensor_W.reshape(P, Mt, P, Nt)
        W2 = W1.transpose(0, 2, 1, 3)
        W3 = W2.reshape(P, P, Mt*Nt)
        W_u32 = cast_tensor_u32(W3.ravel())
        runner.memcpy_h2d(sym_W, W_u32, 0, 0, P, P, Mt*Nt, \
                        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
        
        runner.launch('init_task', nonblock=False)
        total_warmup_times, total_repeat_times = 10, 100
        runner.launch('meshgemv_host', np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)
        
        res3_1d_u32 = np.zeros(P*N, dtype=np.uint32)
        runner.memcpy_d2h(res3_1d_u32, sym_res, 0, 0, P, P, Nt, \
                        streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
        res3_1d_fp16 = sdk_utils.memcpy_view(res3_1d_u32, np.dtype(np.float16))
        res = res3_1d_fp16.reshape(P, N)
        
        time_memcpy_1d_f32 = np.zeros(P*P*3, dtype=np.float32)
        runner.memcpy_d2h(time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, P, P, 3, streaming=False,
                        order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
        time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order='C')
        
        time_ref_1d_f32 = np.zeros(P*P*2, np.float32)
        runner.memcpy_d2h(time_ref_1d_f32, symbol_time_ref, 0, 0, P, P, 2, streaming=False,
                        order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
        time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order='C')
        
    time_start = np.zeros((P, P)).astype(int)
    time_end = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(time_memcpy_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_memcpy_hwl[(h, w, 1)]), base=16)
            hex_t2 = int(float_to_hex(time_memcpy_hwl[(h, w, 2)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_start[(h, w)] = make_u48(word)
            word[0] = (hex_t1 >> 16) & 0x0000ffff
            word[1] = hex_t2 & 0x0000ffff
            word[2] = (hex_t2 >> 16) & 0x0000ffff
            time_end[(h, w)] = make_u48(word)
    
    time_ref = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_ref[(h, w)] = make_u48(word)
            
    for py in range(P):
        for px in range(P):
            time_ref[(py, px)] = time_ref[(py, px)] - (px + py)
            
    time_start = time_start - time_ref
    time_end = time_end - time_ref
    
    min_time_start = time_start.min()
    max_time_end = time_end.max()
    
    print(f"\nRepeat count: {total_repeat_times}")
    print(f"P: {P}, M: {M}, N: {N}")
    print(f"Mean cycle count: {np.mean(time_end - time_start)/total_repeat_times}")
    print(f"Max Cycle count: {(max_time_end - min_time_start)/total_repeat_times}")
    
    freq_ghz = 1.1
    time = (max_time_end - min_time_start) / total_repeat_times / (freq_ghz*1e6)
    print(f"Time: {time} ms")
    
if __name__ == "__main__":
    main()