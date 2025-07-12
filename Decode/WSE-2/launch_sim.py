import json
import os
import struct
import argparse
import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view, calculate_cycles
from cerebras.sdk.debug.debug_util import debug_util
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder

def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

class Config:
    def __init__(self):
        self.P = 8
        self.bsz = 1
        self.group_num = 2
        self.dim = 64
        self.n_heads = 1
        self.n_kv_heads = 1
        self.head_dim = 64
        self.seq_len = 64
        self.ffn_dim = 64
        
def parse_args():
    parser = argparse.ArgumentParser(description="Decode on simulator")
    parser.add_argument("--config", default="config.json", type=str, help="Config file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = Config()
    
    if not os.path.exists(args.config):
        print("Host: Use default test values.")
    else:
        with open(args.config) as f:
            config.__dict__.update(json.load(f))
            
    P = config.P
    bsz = config.bsz
    dim = config.dim
    seq_len = config.seq_len
    ffn_dim = config.ffn_dim
    
    dim_p_pe = dim // P
    seq_len_p_pe = seq_len // P
    ffn_dim_p_pe = ffn_dim // P
    
    _dim_p_pe = dim_p_pe
    if (dim_p_pe % 2) == 1:
        _dim_p_pe = dim_p_pe - 1
    
    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR
    
    X = np.random.rand(1, bsz*dim).astype(np.float16)
    tensor_X = np.tile(X.reshape(P, bsz*dim_p_pe), reps=(1, P))
    
    W = np.random.rand(1, dim).astype(np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))
    
    tensor_q_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_k_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_v_weight = np.random.rand(dim, dim).astype(np.float16)
    
    freqs_sin = np.random.rand(1, P*_dim_p_pe//2).astype(np.float16)
    tensor_freqs_sin = np.tile(freqs_sin.reshape(P, _dim_p_pe//2), reps=(1, P))
    freqs_cos = np.random.rand(1, P*_dim_p_pe//2).astype(np.float16)
    tensor_freqs_cos = np.tile(freqs_cos.reshape(P, _dim_p_pe//2), reps=(1, P))
    
    tensor_XKCache = np.random.rand(dim, seq_len).astype(np.float16)
    tensor_XVCache = np.random.rand(seq_len, dim).astype(np.float16)
    
    tensor_o_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_up_weight = np.random.rand(dim, ffn_dim).astype(np.float16)
    tensor_gate_weight = np.random.rand(dim, ffn_dim).astype(np.float16)
    tensor_down_weight = np.random.rand(ffn_dim, dim).astype(np.float16)
    
    runner = SdkRuntime("out")
    runner.load()
    runner.run()
    
    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")
    sym_Q_weight = runner.get_id("Q_weight")
    sym_K_weight = runner.get_id("K_weight")
    sym_V_weight = runner.get_id("V_weight")
    sym_freqs_sin = runner.get_id("freqs_sin")
    sym_freqs_cos = runner.get_id("freqs_cos")
    sym_XKCache = runner.get_id("XKCache")
    sym_XVCache = runner.get_id("XVCache")
    sym_O_weight = runner.get_id("O_weight")
    sym_UP_weight = runner.get_id("UP_weight")
    sym_GATE_weight = runner.get_id("GATE_weight")
    sym_DOWN_weight = runner.get_id("DOWN_weight")
    
    symbol_time_memcpy = runner.get_id("time_memcpy")
    symbol_time_ref = runner.get_id("time_ref")
    
    
    # -------------------------------------------------------------------------- #
    # ------------------------------ H2D memcpy ------------------------------ #
    # -------------------------------------------------------------------------- #
    
    X_u32 = input_array_to_u32(tensor_X.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_X, X_u32, 0, 0, P, P, bsz*dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    W_u32 = input_array_to_u32(tensor_W.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_W, W_u32, 0, 0, P, P, dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    # Copy Q_weight
    Q_reshape = tensor_q_weight.reshape(P, dim_p_pe, P, dim_p_pe)
    Q_transpose = Q_reshape.transpose(0, 2, 1, 3)
    Q_reshape = Q_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
    Q_u32 = input_array_to_u32(Q_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_Q_weight, Q_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    # Copy K_weight
    K_reshape = tensor_k_weight.reshape(P, dim_p_pe, P, dim_p_pe)
    K_transpose = K_reshape.transpose(0, 2, 1, 3)
    K_reshape = K_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
    K_u32 = input_array_to_u32(K_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_K_weight, K_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    # Copy V_weight
    V_reshape = tensor_v_weight.reshape(P, dim_p_pe, P, dim_p_pe)
    V_transpose = V_reshape.transpose(0, 2, 1, 3)
    V_reshape = V_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
    V_u32 = input_array_to_u32(V_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_V_weight, V_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    freqs_sin_u32 = input_array_to_u32(tensor_freqs_sin.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_freqs_sin, freqs_sin_u32, 0, 0, P, P, _dim_p_pe//2, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy freqs_cos
    freqs_cos_u32 = input_array_to_u32(tensor_freqs_cos.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_freqs_cos, freqs_cos_u32, 0, 0, P, P, _dim_p_pe//2, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy XKCache
    XKCache_reshape = tensor_XKCache.reshape(P, dim_p_pe, P, seq_len_p_pe)
    XKCache_transpose = XKCache_reshape.transpose(0, 2, 1, 3)
    XKCache_reshape = XKCache_transpose.reshape(P, P, dim_p_pe * seq_len_p_pe)
    XKCache_u32 = input_array_to_u32(XKCache_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_XKCache, XKCache_u32, 0, 0, P, P, dim_p_pe * seq_len_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy XVCache
    XVCache_reshape = tensor_XVCache.reshape(P, seq_len_p_pe, P, dim_p_pe)
    XVCache_transpose = XVCache_reshape.transpose(0, 2, 1, 3)
    XVCache_reshape = XVCache_transpose.reshape(P, P, seq_len_p_pe * dim_p_pe)
    XVCache_u32 = input_array_to_u32(XVCache_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_XVCache, XVCache_u32, 0, 0, P, P, seq_len_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy O_weight
    O_reshape = tensor_o_weight.reshape(P, dim_p_pe, P, dim_p_pe)
    O_transpose = O_reshape.transpose(0, 2, 1, 3)
    O_reshape = O_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
    O_u32 = input_array_to_u32(O_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_O_weight, O_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy UP_weight
    UP_reshape = tensor_up_weight.reshape(P, dim_p_pe, P, ffn_dim_p_pe)
    UP_transpose = UP_reshape.transpose(0, 2, 1, 3)
    UP_reshape = UP_transpose.reshape(P, P, dim_p_pe * ffn_dim_p_pe)
    UP_u32 = input_array_to_u32(UP_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_UP_weight, UP_u32, 0, 0, P, P, dim_p_pe * ffn_dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy GATE_weight
    GATE_reshape = tensor_gate_weight.reshape(P, dim_p_pe, P, ffn_dim_p_pe)
    GATE_transpose = GATE_reshape.transpose(0, 2, 1, 3)
    GATE_reshape = GATE_transpose.reshape(P, P, dim_p_pe * ffn_dim_p_pe)
    GATE_u32 = input_array_to_u32(GATE_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_GATE_weight, GATE_u32, 0, 0, P, P, dim_p_pe * ffn_dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    # Copy DOWN_weight
    DOWN_reshape = tensor_down_weight.reshape(P, ffn_dim_p_pe, P, dim_p_pe)
    DOWN_transpose = DOWN_reshape.transpose(0, 2, 1, 3)
    DOWN_reshape = DOWN_transpose.reshape(P, P, ffn_dim_p_pe * dim_p_pe)
    DOWN_u32 = input_array_to_u32(DOWN_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(
        sym_DOWN_weight, DOWN_u32, 0, 0, P, P, ffn_dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
    )
    
    runner.launch('init_task', nonblock=False)
    total_warmup_times, total_repeat_times = 1, 10
    runner.launch('decode_host', np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)
    
    time_memcpy_1d_f32 = np.zeros(P*P*3, dtype=np.float32)
    runner.memcpy_d2h(time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, P, P, 3, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order='C')
    
    time_ref_1d_f32 = np.zeros(P*P*2, np.float32)
    runner.memcpy_d2h(time_ref_1d_f32, symbol_time_ref, 0, 0, P, P, 2, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order='C')
    
    runner.stop()
    
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
    print(f"Mean cycle count: {np.mean(time_end - time_start)/total_repeat_times}")
    print(f"Max Cycle count: {(max_time_end - min_time_start)/total_repeat_times}")
    
if __name__ == "__main__":
    main()