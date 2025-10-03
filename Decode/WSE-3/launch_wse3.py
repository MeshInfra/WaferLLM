import json
import os
import signal
import struct
import argparse
import numpy as np

from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
from cerebras.sdk.client import SdkRuntime, sdk_utils

out_path = "compile_out"

def cast_tensor_u32(tensor):
    return np.uint32(tensor.view(np.uint16))

class Config:
    def __init__(self):
        self.P = 8
        self.group_num = 2
        self.bsz = 1
        self.dim = 64
        self.n_heads = 1
        self.n_kv_heads = 1
        self.head_dim = 64
        self.seq_len = 64
        self.ffn_dim = 64
        self.layer_num = 32

def parse_args():
    parser = argparse.ArgumentParser(description="Move to right unit test")
    parser.add_argument("--config", default="config.json", type=str, help="Config file")
    parser.add_argument("--simulator", action="store_true", help="Runs on simulator")
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
    pe_num_p_group = P // config.group_num
    bsz = config.bsz
    dim = config.dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    seq_len = config.seq_len
    ffn_dim = config.ffn_dim
    layer_num = config.layer_num
    dim_p_pe = dim // P
    pes_p_head = P // n_heads
    pes_p_kv_head = P // n_kv_heads
    head_dim_p_pe = head_dim // P
    seq_len_p_pe = seq_len // P
    ffn_dim_p_pe = ffn_dim // P

    print(f"Host: P: {P}, Batch size: {bsz}, dim_p_pe: {dim_p_pe}, pes_p_head: {pes_p_head}, pes_p_kv_head: {pes_p_kv_head}, head_dim_p_pe: {head_dim_p_pe}, ffn_dim_p_pe: {ffn_dim_p_pe}, simulator: {args.simulator}")
    
    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    X = np.random.rand(1, bsz*dim).astype(np.float16)
    tensor_X = np.tile(X.reshape(P, bsz*dim_p_pe), reps=(1, P))
    
    W = np.random.rand(1, dim).astype(np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))
    
    tensor_q_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_k_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_v_weight = np.random.rand(dim, dim).astype(np.float16)
    
    _dim_p_pe = dim_p_pe
    if (dim_p_pe % 2) == 1:
        _dim_p_pe = dim_p_pe - 1
        
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
    
    with open(f"{out_path}/artifact_{P}_{P//pe_num_p_group}.json", "r", encoding="utf8") as f:
        data = json.load(f)
        artifact_path = data["artifact_id"]

    with SdkRuntime(artifact_path, simulator=args.simulator) as runner:
        
        # -------------------------------------------------------------------------- #
        # ------------------------------ Get symbols ------------------------------ #
        # -------------------------------------------------------------------------- #
        
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
        
        # timer symbol list:
        symbol_timer_buf = runner.get_id("timer_buf")
        symbol_timer_ref = runner.get_id("time_ref")
        sym_debug = runner.get_id("debug")
        
        # -------------------------------------------------------------------------- #
        # ------------------------------ H2D memcpy ------------------------------ #
        # -------------------------------------------------------------------------- #
        
        # X_u32 = sdk_utils.input_array_to_u32(tensor_X.ravel(), 1, 1)
        X_u32 = cast_tensor_u32(tensor_X.ravel())
        runner.memcpy_h2d(
            sym_X, X_u32, 0, 0, P, P, bsz*dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        
        # W_u32 = sdk_utils.input_array_to_u32(tensor_W.ravel(), 1, 1)
        W_u32 = cast_tensor_u32(tensor_W.ravel())
        runner.memcpy_h2d(
            sym_W, W_u32, 0, 0, P, P, dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        
        # Copy Q_weight
        Q_reshape = tensor_q_weight.reshape(P, dim_p_pe, P, dim_p_pe)
        Q_transpose = Q_reshape.transpose(0, 2, 1, 3)
        Q_reshape = Q_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
        # Q_u32 = sdk_utils.input_array_to_u32(Q_reshape.ravel(), 1, 1)
        Q_u32 = cast_tensor_u32(Q_reshape.ravel())
        runner.memcpy_h2d(
            sym_Q_weight, Q_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        
        # Copy K_weight
        K_reshape = tensor_k_weight.reshape(P, dim_p_pe, P, dim_p_pe)
        K_transpose = K_reshape.transpose(0, 2, 1, 3)
        K_reshape = K_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
        # K_u32 = sdk_utils.input_array_to_u32(K_reshape.ravel(), 1, 1)
        K_u32 = cast_tensor_u32(K_reshape.ravel())
        runner.memcpy_h2d(
            sym_K_weight, K_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        
        # Copy V_weight
        V_reshape = tensor_v_weight.reshape(P, dim_p_pe, P, dim_p_pe)
        V_transpose = V_reshape.transpose(0, 2, 1, 3)
        V_reshape = V_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
        # V_u32 = sdk_utils.input_array_to_u32(V_reshape.ravel(), 1, 1)
        V_u32 = cast_tensor_u32(V_reshape.ravel())
        runner.memcpy_h2d(
            sym_V_weight, V_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        
        # freqs_sin_u32 = sdk_utils.input_array_to_u32(tensor_freqs_sin.ravel(), 1, 1)
        freqs_sin_u32 = cast_tensor_u32(tensor_freqs_sin.ravel())
        runner.memcpy_h2d(
            sym_freqs_sin, freqs_sin_u32, 0, 0, P, P, _dim_p_pe//2, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy freqs_cos
        # freqs_cos_u32 = sdk_utils.input_array_to_u32(tensor_freqs_cos.ravel(), 1, 1)
        freqs_cos_u32 = cast_tensor_u32(tensor_freqs_cos.ravel())
        runner.memcpy_h2d(
            sym_freqs_cos, freqs_cos_u32, 0, 0, P, P, _dim_p_pe//2, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy XKCache
        XKCache_reshape = tensor_XKCache.reshape(P, dim_p_pe, P, seq_len_p_pe)
        XKCache_transpose = XKCache_reshape.transpose(0, 2, 1, 3)
        XKCache_reshape = XKCache_transpose.reshape(P, P, dim_p_pe * seq_len_p_pe)
        # XKCache_u32 = sdk_utils.input_array_to_u32(XKCache_reshape.ravel(), 1, 1)
        XKCache_u32 = cast_tensor_u32(XKCache_reshape.ravel())
        runner.memcpy_h2d(
            sym_XKCache, XKCache_u32, 0, 0, P, P, dim_p_pe * seq_len_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy XVCache
        XVCache_reshape = tensor_XVCache.reshape(P, seq_len_p_pe, P, dim_p_pe)
        XVCache_transpose = XVCache_reshape.transpose(0, 2, 1, 3)
        XVCache_reshape = XVCache_transpose.reshape(P, P, seq_len_p_pe * dim_p_pe)
        # XVCache_u32 = sdk_utils.input_array_to_u32(XVCache_reshape.ravel(), 1, 1)
        XVCache_u32 = cast_tensor_u32(XVCache_reshape.ravel())
        runner.memcpy_h2d(
            sym_XVCache, XVCache_u32, 0, 0, P, P, seq_len_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy O_weight
        O_reshape = tensor_o_weight.reshape(P, dim_p_pe, P, dim_p_pe)
        O_transpose = O_reshape.transpose(0, 2, 1, 3)
        O_reshape = O_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
        # O_u32 = sdk_utils.input_array_to_u32(O_reshape.ravel(), 1, 1)
        O_u32 = cast_tensor_u32(O_reshape.ravel())
        runner.memcpy_h2d(
            sym_O_weight, O_u32, 0, 0, P, P, dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy UP_weight
        UP_reshape = tensor_up_weight.reshape(P, dim_p_pe, P, ffn_dim_p_pe)
        UP_transpose = UP_reshape.transpose(0, 2, 1, 3)
        UP_reshape = UP_transpose.reshape(P, P, dim_p_pe * ffn_dim_p_pe)
        # UP_u32 = sdk_utils.input_array_to_u32(UP_reshape.ravel(), 1, 1)
        UP_u32 = cast_tensor_u32(UP_reshape.ravel())
        runner.memcpy_h2d(
            sym_UP_weight, UP_u32, 0, 0, P, P, dim_p_pe * ffn_dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy GATE_weight
        GATE_reshape = tensor_gate_weight.reshape(P, dim_p_pe, P, ffn_dim_p_pe)
        GATE_transpose = GATE_reshape.transpose(0, 2, 1, 3)
        GATE_reshape = GATE_transpose.reshape(P, P, dim_p_pe * ffn_dim_p_pe)
        # GATE_u32 = sdk_utils.input_array_to_u32(GATE_reshape.ravel(), 1, 1)
        GATE_u32 = cast_tensor_u32(GATE_reshape.ravel())
        runner.memcpy_h2d(
            sym_GATE_weight, GATE_u32, 0, 0, P, P, dim_p_pe * ffn_dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        # Copy DOWN_weight
        DOWN_reshape = tensor_down_weight.reshape(P, ffn_dim_p_pe, P, dim_p_pe)
        DOWN_transpose = DOWN_reshape.transpose(0, 2, 1, 3)
        DOWN_reshape = DOWN_transpose.reshape(P, P, ffn_dim_p_pe * dim_p_pe)
        # DOWN_u32 = sdk_utils.input_array_to_u32(DOWN_reshape.ravel(), 1, 1)
        DOWN_u32 = cast_tensor_u32(DOWN_reshape.ravel())
        runner.memcpy_h2d(
            sym_DOWN_weight, DOWN_u32, 0, 0, P, P, ffn_dim_p_pe * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        
        # -------------------------------------------------------------------------- #
        # ------------------------------ Run WSE-3 ---------------------------- #
        # -------------------------------------------------------------------------- #
        runner.launch("init_task", nonblock=False)

        total_warmup_times, total_repeat_times = 5, 50
        runner.launch("decode_host", np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)
        
        # -------------------------------------------------------------------------- #
        # ------------------------------ D2H memcpy ------------------------------ #
        # -------------------------------------------------------------------------- #
        
        debug_1d_u32 = np.zeros(P * bsz * dim, dtype=np.uint32)
        runner.memcpy_d2h(
            debug_1d_u32, sym_debug, 0, 0, P, P, bsz * dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        debug = sdk_utils.memcpy_view(debug_1d_u32, np.dtype(np.float16))
        debug = debug.reshape(P, bsz * dim)
        
        # -------------------------------------------------------------------------- #
        # ------------------------------ Timer Check ------------------------------ #
        # -------------------------------------------------------------------------- #
        # Copy back timer_buf from all width x height PEs
        timer_buf_1d_u32 = np.zeros((P*P*3), dtype=np.uint32)
        runner.memcpy_d2h(
            timer_buf_1d_u32, symbol_timer_buf, 0, 0, P, P, 3, streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False
        )
        timer_buf_time_hwl = timer_buf_1d_u32.view(np.float32).reshape((P, P, 3))
        
    # -------------------------------------------------------------------------- #
    # ------------------------------ Compute time ------------------------------ #
    # -------------------------------------------------------------------------- #
    cycles_count = np.zeros((P, P))
    for pe_x in range(P):
            for pe_y in range(P):
                cycles_count[pe_y, pe_x] = sdk_utils.calculate_cycles(timer_buf_time_hwl[pe_y, pe_x, :])
        
    freq_ghz = 1.1

    cycles_count_mean = cycles_count.mean()
    cycles_count_mean_per_step = cycles_count_mean / total_repeat_times
    print(f"\nRepeat count: {total_repeat_times}")
    print(f"Mean cycles count: {cycles_count_mean_per_step*layer_num}")

    throughput_p_request = 1 / ((cycles_count_mean_per_step*layer_num) / (freq_ghz*1e9))
    print(f"Throughput_p_request: {throughput_p_request}")
    
if __name__ == "__main__":
    main()