import json
import os
import argparse
import numpy as np
import math
import contextlib
import sys
from pathlib import Path
import csv
project_root = Path(__file__).resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
try:
    from cerebras.sdk.client import SdkRuntime as _SdkRuntimeClient, sdk_utils
    _memcpy_view    = sdk_utils.memcpy_view
    _calculate_cycles = sdk_utils.calculate_cycles
    _use_client = True
except ImportError:
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime as _SdkRuntimeSim
    from cerebras.sdk.sdk_utils import memcpy_view as _memcpy_view, calculate_cycles as _calculate_cycles
    _use_client = False

from validate import decode_mha_block
from util import calculate_ulp_distance_fp16

out_path = "compile_out"

def cast_tensor_u32(tensor):
    return np.uint32(tensor.view(np.uint16))

@contextlib.contextmanager
def make_runner(out_path, pe_num_p_h_group, Pw, Ph, simulator, cmaddr):
    if _use_client:
        with open(f"{out_path}/artifact_{Pw}_{Ph}_{pe_num_p_h_group}.json", "r", encoding="utf-8") as f:
            artifact_path = json.load(f)["artifact_id"]
        with _SdkRuntimeClient(artifact_path, simulator=simulator) as runner:
            yield runner
    else:
        runner = _SdkRuntimeSim("out", simfab_numthreads=64, msg_level='INFO', suppress_simfab_trace=True, cmaddr=cmaddr)
        runner.load()
        runner.run()
        try:
            yield runner
        finally:
            runner.stop()

class Config:
    def __init__(self):
        self.Pw = 16
        self.Ph = 8
        self.pe_num_p_h_group = 4
        self.pe_num_p_v_group = 4
        self.pe_num_p_group_in_head = 4
        self.bsz = 2
        self.dim = 64
        self.n_heads = 2
        self.n_kv_heads = 1
        self.head_dim = 32
        self.seq_len = 96
        self.ffn_dim = 128
        self.layer_num = 32

def parse_args():
    parser = argparse.ArgumentParser(description="WSE-3 decode launch")
    parser.add_argument("--config", default="config.json", type=str, help="Config file")
    parser.add_argument("--cmaddr", default=None, type=str, help="CM address")
    parser.add_argument("--simulator", action="store_true", help="Runs on simulator")
    parser.add_argument("--runs", default=10, type=str, help="The experiment will be conducted with X runs")
    parser.add_argument("--csv-output", default="benchmark.csv", type=str, help="Path to output CSV file for results")
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()

    if not os.path.exists(args.config):
        print("Host: Use default test values.")
    else:
        with open(args.config) as f:
            config.__dict__.update(json.load(f))

    Pw = config.Pw
    Ph = config.Ph
    pe_num_p_h_group = config.pe_num_p_h_group
    pe_num_p_v_group = config.pe_num_p_v_group
    pe_num_p_group_in_head = config.pe_num_p_group_in_head
    bsz = config.bsz
    dim = config.dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    seq_len = config.seq_len
    ffn_dim = config.ffn_dim
    layer_num = config.layer_num
    if n_heads * head_dim != dim:
        raise AssertionError(f"n_heads * head_dim must be equal to dim, but is not: {n_heads}*{head_dim}!={dim}")
    if Pw % n_heads != 0:
        raise AssertionError(f"Pw must be the multiple of n_heads, but is not. Pw:{Pw}, n_heads:{n_heads}")

    # -------------------------------------------------------------------------- #
    # Use JSON-specified tile sizes when available; otherwise ceil-division
    # -------------------------------------------------------------------------- #
    v_dim_p_pe  = getattr(config, 'v_dim_p_pe',  None) or math.ceil(dim / Ph)
    h_dim_p_pe  = getattr(config, 'h_dim_p_pe',  None) or math.ceil(dim / Pw)
    seq_len_p_pe = getattr(config, 'seq_len_p_pe', None) or math.ceil(seq_len / Ph)
    ffn_dim_p_pe = getattr(config, 'ffn_dim_p_pe', None) or math.ceil(ffn_dim / Pw)
    
    pes_p_head = Pw // n_heads
    pes_p_kv_head = Pw // n_kv_heads    # not used in CSL currently
    
    # Padded (PE-grid) sizes
    req_dim_v      = Ph * v_dim_p_pe
    req_head_dim_h = pes_p_head * h_dim_p_pe   # per-head padded head_dim
    req_dim_h      = n_heads * req_head_dim_h
    req_seq_len    = Ph * seq_len_p_pe
    req_ffn_dim    = Pw * ffn_dim_p_pe
    _dim_p_pe      = (h_dim_p_pe // 2) * 2     # must be even for RoPE
    req_head_freqs = pes_p_head * (_dim_p_pe // 2)
    
    print(f"Host: (Ph, Pw): ({Ph}, {Pw}), Batch size: {bsz},"
          f" (v_dim_p_pe, h_dim_p_pe): ({v_dim_p_pe}, {h_dim_p_pe}),"
          f" (pe_num_p_v_group, pe_num_p_h_group): ({pe_num_p_v_group}, {pe_num_p_h_group}),"
          f" pe_num_p_group_in_head: {pe_num_p_group_in_head},"
          f" pes_p_head: {pes_p_head}, pes_p_kv_head: {pes_p_kv_head},"
          f" head_dim: {head_dim}, seq_len_p_pe: {seq_len_p_pe},"
          f" ffn_dim_p_pe: {ffn_dim_p_pe}")
    print(f"  padded: req_dim_v={req_dim_v}, req_dim_h={req_dim_h},"
          f" req_seq_len={req_seq_len}, req_ffn_dim={req_ffn_dim}")

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    # -------------------------------------------------------------------------- #
    # Generate tensors with actual (unpadded) dimensions — used for validation
    # -------------------------------------------------------------------------- #
    X = np.random.rand(bsz, dim).astype(np.float16)       # (bsz, dim)
    W = (np.random.rand(dim).astype(np.float16) * (0.8 / np.sqrt(dim))).astype(np.float16)
    
    tensor_q_weight = (np.random.rand(dim, dim).astype(np.float16) * (0.8 / np.sqrt(dim))).astype(np.float16)
    tensor_k_weight = (np.random.rand(dim, dim).astype(np.float16) * (0.8 / np.sqrt(dim))).astype(np.float16)
    tensor_v_weight = (np.random.rand(dim, dim).astype(np.float16) * (0.8 / np.sqrt(dim))).astype(np.float16)
    
    # for multi-head attention (per-head RoPE)
    freqs_sin = np.random.rand(head_dim//2).astype(np.float16)
    freqs_cos = np.random.rand(head_dim//2).astype(np.float16)
    
    tensor_XKCache = (np.random.rand(dim, seq_len).astype(np.float16) * (0.8 / np.sqrt(dim))).astype(np.float16)
    tensor_XVCache = (np.random.rand(seq_len, dim).astype(np.float16) * (0.8 / np.sqrt(dim))).astype(np.float16)
    
    tensor_o_weight = (np.random.rand(dim, dim).astype(np.float16) * (0.8 / np.sqrt(dim))).astype(np.float16)
    tensor_up_weight = (np.random.rand(dim, ffn_dim).astype(np.float16) * (0.8 / np.sqrt(dim))).astype(np.float16)
    tensor_gate_weight = (np.random.rand(dim, ffn_dim).astype(np.float16) * (0.8 / np.sqrt(dim))).astype(np.float16)
    tensor_down_weight = (np.random.rand(ffn_dim, dim).astype(np.float16) * (0.8 / np.sqrt(dim))).astype(np.float16)
    
    # -------------------------------------------------------------------------- #
    # Padding helpers  (mirrors mha_block.py › distribute_inputs)
    # -------------------------------------------------------------------------- #
    def _pad(arr, *req_dims, value=0.0):
        pads = [(0, max(0, req - arr.shape[i])) for i, req in enumerate(req_dims)]
        while len(pads) < arr.ndim:
            pads.append((0, 0))
        return np.pad(arr, pads, constant_values=value)

    def _pad_head_cols(mat, value=0.0):
        """Pad (*, n_heads * head_dim) → (*, n_heads * req_head_dim_h) per-head."""
        leading = mat.shape[:-1]
        shaped = mat.reshape(*leading, n_heads, head_dim)
        padded = np.pad(shaped,
                        [(0, 0)] * (shaped.ndim - 1) + [(0, req_head_dim_h - head_dim)],
                        constant_values=value)
        return padded.reshape(*leading, n_heads * req_head_dim_h)

    def _pad_head_rows(mat, value=0.0):
        """Pad (n_heads * head_dim, *) → (n_heads * req_head_dim_h, *) per-head."""
        trailing = mat.shape[1:]
        shaped = mat.reshape(n_heads, head_dim, *trailing)
        padded = np.pad(shaped,
                        [(0, 0), (0, req_head_dim_h - head_dim)] + [(0, 0)] * len(trailing),
                        constant_values=value)
        return padded.reshape(n_heads * req_head_dim_h, *trailing)
    
    # -------------------------------------------------------------------------- #
    # Pad all tensors to PE-grid sizes
    # -------------------------------------------------------------------------- #
    X_pad      = _pad(X, bsz, req_dim_v)                                     # (bsz, req_dim_v)
    W_pad      = _pad(W, req_dim_v)                                           # (req_dim_v,)
    Q_pad      = _pad(_pad_head_cols(tensor_q_weight), req_dim_v, req_dim_h)  # (req_dim_v, req_dim_h)
    K_pad      = _pad(_pad_head_cols(tensor_k_weight),  req_dim_v, req_dim_h)
    V_pad      = _pad(_pad_head_cols(tensor_v_weight),  req_dim_v, req_dim_h)
    freqs_sin_pad = _pad(freqs_sin, req_head_freqs)                    # (req_head_freqs,)
    freqs_cos_pad = _pad(freqs_cos, req_head_freqs)
    XKCache_pad = _pad(_pad_head_rows(tensor_XKCache), req_dim_h, req_seq_len)  # (req_dim_h, req_seq_len)
    XVCache_pad = _pad(_pad_head_cols(tensor_XVCache), req_seq_len, req_dim_h)  # (req_seq_len, req_dim_h)
    O_pad      = _pad(_pad_head_rows(tensor_o_weight), req_dim_h, req_dim_v)    # (req_dim_h, req_dim_v)
    UP_pad     = _pad(tensor_up_weight,   req_dim_v, req_ffn_dim)
    GATE_pad   = _pad(tensor_gate_weight, req_dim_v, req_ffn_dim)
    DOWN_pad   = _pad(tensor_down_weight, req_ffn_dim, req_dim_v)

    # -------------------------------------------------------------------------- #
    # Build H2D tensors from padded data
    # -------------------------------------------------------------------------- #
    # X: vertical partition — each py row gets (bsz, v_dim_p_pe); same for all px cols
    tensor_X = np.tile(
        X_pad.reshape(bsz, Ph, v_dim_p_pe).transpose(1, 0, 2).reshape(Ph, bsz * v_dim_p_pe),
        reps=(1, Pw))                                    # (Ph, Pw * bsz * v_dim_p_pe)

    # W: same partition as X (v-dim)
    tensor_W = np.tile(W_pad.reshape(Ph, v_dim_p_pe), reps=(1, Pw))

    # Q/K/V_weight: (req_dim_v, req_dim_h) → each PE: (v_dim_p_pe, h_dim_p_pe)
    def _h2d_weight_vh(pad):
        return pad.reshape(Ph, v_dim_p_pe, Pw, h_dim_p_pe).transpose(0, 2, 1, 3).reshape(Ph, Pw, v_dim_p_pe * h_dim_p_pe)

    tensor_Q = _h2d_weight_vh(Q_pad)
    tensor_K = _h2d_weight_vh(K_pad)
    tensor_V = _h2d_weight_vh(V_pad)

    # freqs: per-head slice (px_in_head) × Ph rows; tile for n_heads
    tensor_freqs_sin = np.tile(
        freqs_sin_pad.reshape(1, req_head_freqs), reps=(Ph, n_heads))   # (Ph, Pw * _dim_p_pe//2)
    tensor_freqs_cos = np.tile(
        freqs_cos_pad.reshape(1, req_head_freqs), reps=(Ph, n_heads))
    
    # XKCache: (req_dim_h, req_seq_len) — px cols × py rows → each PE: (h_dim_p_pe, seq_len_p_pe)
    tensor_XKCache_h2d = (XKCache_pad
                          .reshape(Pw, h_dim_p_pe, Ph, seq_len_p_pe)
                          .transpose(2, 0, 1, 3)
                          .reshape(Ph, Pw, h_dim_p_pe * seq_len_p_pe))

    # XVCache: (req_seq_len, req_dim_h) → each PE: (seq_len_p_pe, h_dim_p_pe)
    tensor_XVCache_h2d = (XVCache_pad
                          .reshape(Ph, seq_len_p_pe, Pw, h_dim_p_pe)
                          .transpose(0, 2, 1, 3)
                          .reshape(Ph, Pw, seq_len_p_pe * h_dim_p_pe))

    # O_weight: (req_dim_h, req_dim_v) — px cols × py rows → each PE: (h_dim_p_pe, v_dim_p_pe)
    tensor_O_h2d = (O_pad
                    .reshape(Pw, h_dim_p_pe, Ph, v_dim_p_pe)
                    .transpose(2, 0, 1, 3)
                    .reshape(Ph, Pw, h_dim_p_pe * v_dim_p_pe))

    # UP/GATE_weight: (req_dim_v, req_ffn_dim) → each PE: (v_dim_p_pe, ffn_dim_p_pe)
    def _h2d_weight_vf(pad):
        return pad.reshape(Ph, v_dim_p_pe, Pw, ffn_dim_p_pe).transpose(0, 2, 1, 3).reshape(Ph, Pw, v_dim_p_pe * ffn_dim_p_pe)

    tensor_UP_h2d   = _h2d_weight_vf(UP_pad)
    tensor_GATE_h2d = _h2d_weight_vf(GATE_pad)

    # DOWN_weight: (req_ffn_dim, req_dim_v) → each PE: (ffn_dim_p_pe, v_dim_p_pe)
    tensor_DOWN_h2d = (DOWN_pad
                       .reshape(Pw, ffn_dim_p_pe, Ph, v_dim_p_pe)
                       .transpose(2, 0, 1, 3)
                       .reshape(Ph, Pw, ffn_dim_p_pe * v_dim_p_pe))

    # score_mask: 0 for valid seq positions, -inf for padded positions
    # shape matches score tile: each PE holds (bsz * seq_len_p_pe,)
    score_mask_np = np.zeros((Ph, Pw, seq_len_p_pe), dtype=np.float16)
    s0 = np.arange(Ph) * seq_len_p_pe                  # (Ph,)
    valid = np.clip(seq_len - s0, 0, seq_len_p_pe)     # (Ph,)
    seq_indices = np.arange(seq_len_p_pe)              # (seq_len_p_pe,)
    mask = seq_indices[None, :] >= valid[:, None]      # (Ph, seq_len_p_pe)
    score_mask_np[:] = np.where(mask[:, None, :], np.float16(-np.inf), np.float16(0.0))
    
    # initialize time variables
    cycles_count_mean_per_step = 0.0
    e2e_model_cycles_count = 0.0
    throughput_p_request = 0.0

    with make_runner(out_path, pe_num_p_h_group, Pw, Ph, args.simulator, args.cmaddr) as runner:

        # -------------------------------------------------------------------------- #
        # ------------------------------ Get symbols ------------------------------ #
        # -------------------------------------------------------------------------- #
        sym_X        = runner.get_id("X")
        sym_W        = runner.get_id("W")
        sym_Z        = runner.get_id("Z")
        sym_Q_weight = runner.get_id("Q_weight")
        sym_K_weight = runner.get_id("K_weight")
        sym_V_weight = runner.get_id("V_weight")
        sym_freqs_sin = runner.get_id("freqs_sin")
        sym_freqs_cos = runner.get_id("freqs_cos")
        sym_XKCache  = runner.get_id("XKCache")
        sym_XVCache  = runner.get_id("XVCache")
        sym_O_weight    = runner.get_id("O_weight")
        sym_UP_weight   = runner.get_id("UP_weight")
        sym_GATE_weight = runner.get_id("GATE_weight")
        sym_DOWN_weight = runner.get_id("DOWN_weight")
        sym_score_mask = runner.get_id("score_mask")

        # symbol_timer_buf = runner.get_id("timer_buf")
        # symbol_timer_ref = runner.get_id("time_ref")
        # sym_debug        = runner.get_id("debug")
        symbol_time_buf = runner.get_id("time_buf")

        # -------------------------------------------------------------------------- #
        # ------------------------------ H2D memcpy ------------------------------ #
        # -------------------------------------------------------------------------- #
        X_u32 = cast_tensor_u32(tensor_X.ravel())
        runner.memcpy_h2d(
            sym_X, X_u32, 0, 0, Pw, Ph, bsz * v_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        W_u32 = cast_tensor_u32(tensor_W.ravel())
        runner.memcpy_h2d(
            sym_W, W_u32, 0, 0, Pw, Ph, v_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy Q_weight
        Q_u32 = cast_tensor_u32(tensor_Q.ravel())
        runner.memcpy_h2d(
            sym_Q_weight, Q_u32, 0, 0, Pw, Ph, v_dim_p_pe * h_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy K_weight
        K_u32 = cast_tensor_u32(tensor_K.ravel())
        runner.memcpy_h2d(
            sym_K_weight, K_u32, 0, 0, Pw, Ph, v_dim_p_pe * h_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy V_weight
        V_u32 = cast_tensor_u32(tensor_V.ravel())
        runner.memcpy_h2d(
            sym_V_weight, V_u32, 0, 0, Pw, Ph, v_dim_p_pe * h_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy freqs_sin
        freqs_sin_u32 = cast_tensor_u32(tensor_freqs_sin.ravel())
        runner.memcpy_h2d(
            sym_freqs_sin, freqs_sin_u32, 0, 0, Pw, Ph, _dim_p_pe // 2,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy freqs_cos
        freqs_cos_u32 = cast_tensor_u32(tensor_freqs_cos.ravel())
        runner.memcpy_h2d(
            sym_freqs_cos, freqs_cos_u32, 0, 0, Pw, Ph, _dim_p_pe // 2,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy XKCache
        XKCache_u32 = cast_tensor_u32(tensor_XKCache_h2d.ravel())
        runner.memcpy_h2d(
            sym_XKCache, XKCache_u32, 0, 0, Pw, Ph, h_dim_p_pe * seq_len_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy XVCache
        XVCache_u32 = cast_tensor_u32(tensor_XVCache_h2d.ravel())
        runner.memcpy_h2d(
            sym_XVCache, XVCache_u32, 0, 0, Pw, Ph, seq_len_p_pe * h_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy O_weight
        O_u32 = cast_tensor_u32(tensor_O_h2d.ravel())
        runner.memcpy_h2d(
            sym_O_weight, O_u32, 0, 0, Pw, Ph, h_dim_p_pe * v_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy UP_weight
        UP_u32 = cast_tensor_u32(tensor_UP_h2d.ravel())
        runner.memcpy_h2d(
            sym_UP_weight, UP_u32, 0, 0, Pw, Ph, v_dim_p_pe * ffn_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy GATE_weight
        GATE_u32 = cast_tensor_u32(tensor_GATE_h2d.ravel())
        runner.memcpy_h2d(
            sym_GATE_weight, GATE_u32, 0, 0, Pw, Ph, v_dim_p_pe * ffn_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # Copy DOWN_weight
        DOWN_u32 = cast_tensor_u32(tensor_DOWN_h2d.ravel())
        runner.memcpy_h2d(
            sym_DOWN_weight, DOWN_u32, 0, 0, Pw, Ph, ffn_dim_p_pe * v_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        
        # Copy score_mask — each PE: (seq_len_p_pe,)
        score_mask_u32 = cast_tensor_u32(score_mask_np.ravel())
        runner.memcpy_h2d(
            sym_score_mask, score_mask_u32, 0, 0, Pw, Ph, seq_len_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        # -------------------------------------------------------------------------- #
        # ------------------------------ Run WSE-3 -------------------------------- #
        # -------------------------------------------------------------------------- #
        runner.launch("init_task", nonblock=False)

        total_warmup_times, total_repeat_times = 5, int(args.runs)
        
        runner.launch("decode_host", np.int16(total_warmup_times), np.int16(1), nonblock=False)
        
        cycles_count_per_step_acc = 0
        for r in range(total_repeat_times):
            runner.launch("decode_host", np.int16(0), np.int16(1), nonblock=False)

            # -------------------------------------------------------------------------- #
            # ------------------------------ Timer Check ------------------------------ #
            # -------------------------------------------------------------------------- #
            time_buf_1d = np.zeros((Ph*Pw*6), dtype=np.uint32)
            runner.memcpy_d2h(
                time_buf_1d, symbol_time_buf, 0, 0, Pw, Ph, 6, streaming=False,
                data_type=MemcpyDataType.MEMCPY_16BIT,
                order=MemcpyOrder.COL_MAJOR, nonblock=False
            )
            time_buf_u16_hwl = np.reshape((time_buf_1d & 0x0000FFFF).astype(np.uint16), (Ph, Pw, 6), order='F')
            
            # -------------------------------------------------------------------------- #
            # ------------------------------ Compute time ------------------------------ #
            # -------------------------------------------------------------------------- #
            time_start = (
                time_buf_u16_hwl[:, :, 0].astype(np.int64)
                + (time_buf_u16_hwl[:, :, 1].astype(np.int64) << 16)
                + (time_buf_u16_hwl[:, :, 2].astype(np.int64) << 32)
            )

            time_end = (
                time_buf_u16_hwl[:, :, 3].astype(np.int64)
                + (time_buf_u16_hwl[:, :, 4].astype(np.int64) << 16)
                + (time_buf_u16_hwl[:, :, 5].astype(np.int64) << 32)
            )
            
            cycles_count_max = np.max(time_end) - np.min(time_start)
            cycles_count_per_step_acc += cycles_count_max
            print(f"Host: max cycles count (step {r}): {cycles_count_max}")
            
            if args.csv_output:
                fieldnames = [
                    'Pw', 'Ph', 'pe_num_p_h_group', 'pe_num_p_v_group', 'pe_num_p_group_in_head',
                    'bsz', 'dim', 'n_heads', 'n_kv_heads', 'head_dim', 'seq_len', 'ffn_dim', 'layer_num',
                    'h_dim_p_pe', 'v_dim_p_pe', 'seq_len_p_pe', 'ffn_dim_p_pe',
                    'hardware_cycle'
                ]
                result_row = {
                    'Pw': Pw,
                    'Ph': Ph,
                    'pe_num_p_h_group': pe_num_p_h_group,
                    'pe_num_p_v_group': pe_num_p_v_group,
                    'pe_num_p_group_in_head': pe_num_p_group_in_head,
                    'bsz': bsz,
                    'dim': dim,
                    'n_heads': n_heads,
                    'n_kv_heads': n_kv_heads,
                    'head_dim': head_dim,
                    'seq_len': seq_len,
                    'ffn_dim': ffn_dim,
                    'layer_num': getattr(config, 'layer_num', 32),
                    'h_dim_p_pe': h_dim_p_pe,
                    'v_dim_p_pe': v_dim_p_pe,
                    'seq_len_p_pe': seq_len_p_pe,
                    'ffn_dim_p_pe': ffn_dim_p_pe,
                    'hardware_cycle': cycles_count_max
                }
                file_exists = os.path.isfile(args.csv_output)
                with open(args.csv_output, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(result_row)

        # -------------------------------------------------------------------------- #
        # ------------------------------ D2H memcpy ------------------------------ #
        # -------------------------------------------------------------------------- #
        
        z_1d_u32 = np.zeros(Pw * bsz * Ph * v_dim_p_pe, dtype=np.uint32)
        runner.memcpy_d2h(
            z_1d_u32, sym_Z, 0, 0, Pw, Ph, bsz * v_dim_p_pe, streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        simulated_Z = _memcpy_view(z_1d_u32, np.dtype(np.float16))
        simulated_Z = simulated_Z.reshape(Ph, Pw, bsz, v_dim_p_pe)
        simulated_Z = simulated_Z.transpose(0, 2, 1, 3)
        simulated_Z = simulated_Z[:, :, 0].transpose(1, 0, 2).reshape(bsz, Ph * v_dim_p_pe)[:,:dim] # trim padding
        
        # -------------------------------------------------------------------------- #
        # ------------------------------ Save Time Data ---------------------------- #
        # -------------------------------------------------------------------------- #
        # freq_ghz = 1.1    # WSE-2
        freq_ghz = 0.75     # WSE-3
        cycles_count_mean_per_step = cycles_count_per_step_acc / total_repeat_times
        e2e_model_cycles_count = cycles_count_mean_per_step * layer_num
        print(f"\nRepeat count: {total_repeat_times}")
        print(f"Host: Mean cycles count per Layer: {cycles_count_mean_per_step}")
        print(f"Host: End-to-end cycles count of the model: {e2e_model_cycles_count}")

        throughput_p_request = bsz / ((cycles_count_mean_per_step * layer_num) / (freq_ghz * 1e9))
        print(f"Throughput_p_request: {throughput_p_request}")
            

    # -------------------------------------------------------------------------- #
    # ------------------------------ Debug Check ------------------------------ #
    # -------------------------------------------------------------------------- #
    freqs_cos_np = np.tile(freqs_cos, reps=(1, n_heads)).ravel()             # (dim//2,)  PE 分割を展開
    freqs_sin_np = np.tile(freqs_sin, reps=(1, n_heads)).ravel()             # (dim//2,)  PE 分割を展開
    
    # expected_Z, expected_X_norm, expected_Q, expected_K, expected_V, expected_score, expected_output, expected_h1, expected_Z_norm, expected_z3 = decode_block(
    expected_Z, _, _, _, _, _, _, _, _, _ = decode_mha_block(
        X=X,
        W=W,
        tensor_q_weight=tensor_q_weight,
        tensor_k_weight=tensor_k_weight,
        tensor_v_weight=tensor_v_weight,
        freqs_cos=freqs_cos_np,
        freqs_sin=freqs_sin_np,
        tensor_XKCache=tensor_XKCache,
        tensor_XVCache=tensor_XVCache,
        tensor_o_weight=tensor_o_weight,
        tensor_up_weight=tensor_up_weight,
        tensor_gate_weight=tensor_gate_weight,
        tensor_down_weight=tensor_down_weight,
        n_heads = n_heads,
        head_dim = head_dim,
    )

    ulp = calculate_ulp_distance_fp16(expected_Z, simulated_Z)
    print("ULP distance (fp16):")
    # print(ulp)
    print(f"  max ULP: {ulp.max():.1f},  mean ULP: {ulp.mean():.2f},  median ULP: {np.median(ulp):.1f}")
    
    if args.csv_output:
        fieldnames = [
            'Pw', 'Ph', 'pe_num_p_h_group', 'pe_num_p_v_group', 'pe_num_p_group_in_head',
            'bsz', 'dim', 'n_heads', 'n_kv_heads', 'head_dim', 'seq_len', 'ffn_dim', 'layer_num',
            'h_dim_p_pe', 'v_dim_p_pe', 'seq_len_p_pe', 'ffn_dim_p_pe',
            'hardware_cycle_mean', 'e2e_model_hardware_cycle', 'tpr'
        ]
        result_row = {
            'Pw': Pw,
            'Ph': Ph,
            'pe_num_p_h_group': pe_num_p_h_group,
            'pe_num_p_v_group': pe_num_p_v_group,
            'pe_num_p_group_in_head': pe_num_p_group_in_head,
            'bsz': bsz,
            'dim': dim,
            'n_heads': n_heads,
            'n_kv_heads': n_kv_heads,
            'head_dim': head_dim,
            'seq_len': seq_len,
            'ffn_dim': ffn_dim,
            'layer_num': getattr(config, 'layer_num', 32),
            'h_dim_p_pe': h_dim_p_pe,
            'v_dim_p_pe': v_dim_p_pe,
            'seq_len_p_pe': seq_len_p_pe,
            'ffn_dim_p_pe': ffn_dim_p_pe,
            'hardware_cycle_mean': cycles_count_mean_per_step,
            'e2e_model_hardware_cycle': e2e_model_cycles_count,
            'tpr': throughput_p_request,
        }
        file_exists = os.path.isfile("summary.csv")
        with open("summary.csv", 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_row)
        print(f"Results appended to {args.csv_output}")

if __name__ == "__main__":
    main()
