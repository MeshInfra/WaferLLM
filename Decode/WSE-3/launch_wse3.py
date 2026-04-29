import json
import os
import argparse
import numpy as np

from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
from cerebras.sdk.client import SdkRuntime, sdk_utils

out_path = "compile_out"


def cast_tensor_u32(tensor):
    return np.uint32(tensor.view(np.uint16))


def calculate_cycles_from_words(words_u32):
    w0 = int(words_u32[0])
    w1 = int(words_u32[1])
    w2 = int(words_u32[2])

    start = w0 + ((w1 & 0xFFFF) << 32)
    end = ((w1 >> 16) & 0xFFFF) + ((w2 & 0xFFFF) << 16) + (((w2 >> 16) & 0xFFFF) << 32)

    if end < start:
        end += 1 << 48
    return float(end - start)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


PHASE_NAMES = [
    "rmsnorm_x_local",
    "rmsnorm_x_comm",
    "qkv_local",
    "qkv_comm",
    "rope",
    "score_local",
    "score_comm",
    "softmax_local",
    "softmax_comm",
    "output_local",
    "output_comm",
    "o_local",
    "o_comm",
    "attn_residual",
    "rmsnorm_z_local",
    "rmsnorm_z_comm",
    "upgate_local",
    "upgate_comm",
    "silu_mul",
    "down_local",
    "down_comm",
    "ffn_residual",
]

COMM_SUBPHASE_NAMES = [
    "rmsnorm_x_reduce",
    "rmsnorm_x_broadcast",
    "qkv_reduce",
    "qkv_broadcast",
    "score_reduce",
    "score_broadcast",
    "softmax_reduce",
    "softmax_broadcast",
    "output_reduce",
    "output_broadcast",
    "o_reduce",
    "o_broadcast",
    "rmsnorm_z_reduce",
    "rmsnorm_z_broadcast",
    "upgate_reduce",
    "upgate_broadcast",
    "down_reduce",
    "down_broadcast",
]

GROUP_SPECS = [
    (
        "RMSNorm+QKV",
        {
            "Compute (Dist-GEMV)": ["qkv_local"],
            "KV-cache GEMV": [],
            "Communication": ["rmsnorm_x_comm", "qkv_comm"],
            "Elementwise/Norm": ["rmsnorm_x_local"],
        },
    ),
    (
        "RoPE+Score",
        {
            "Compute (Dist-GEMV)": [],
            "KV-cache GEMV": ["score_local"],
            "Communication": ["score_comm"],
            "Elementwise/Norm": ["rope"],
        },
    ),
    (
        "Softmax+Output",
        {
            "Compute (Dist-GEMV)": [],
            "KV-cache GEMV": ["output_local"],
            "Communication": ["softmax_comm", "output_comm"],
            "Elementwise/Norm": ["softmax_local"],
        },
    ),
    (
        "O+Residual",
        {
            "Compute (Dist-GEMV)": ["o_local"],
            "KV-cache GEMV": [],
            "Communication": ["o_comm"],
            "Elementwise/Norm": ["attn_residual"],
        },
    ),
    (
        "RMSNorm+FFN",
        {
            "Compute (Dist-GEMV)": ["upgate_local", "down_local"],
            "KV-cache GEMV": [],
            "Communication": ["rmsnorm_z_comm", "upgate_comm", "down_comm"],
            "Elementwise/Norm": ["rmsnorm_z_local", "silu_mul", "ffn_residual"],
        },
    ),
]


def aggregate_phase_groups(phase_means, total_cycles_mean):
    phase_lookup = {name: float(phase_means[idx]) for idx, name in enumerate(PHASE_NAMES)}
    bars = []
    measured_total = 0.0
    category_totals = {
        "Compute (Dist-GEMV)": 0.0,
        "KV-cache GEMV": 0.0,
        "Communication": 0.0,
        "Elementwise/Norm": 0.0,
        "Other": 0.0,
    }

    for label, mapping in GROUP_SPECS:
        segments = {
            "Compute (Dist-GEMV)": 0.0,
            "KV-cache GEMV": 0.0,
            "Communication": 0.0,
            "Elementwise/Norm": 0.0,
            "Other": 0.0,
        }
        for category, phase_names in mapping.items():
            value = sum(phase_lookup[name] for name in phase_names)
            segments[category] = value
            category_totals[category] += value
            measured_total += value
        bars.append({"label": label, "segments": segments})

    other_cycles = max(float(total_cycles_mean) - measured_total, 0.0)
    bars.append(
        {
            "label": "Other",
            "segments": {
                "Compute (Dist-GEMV)": 0.0,
                "KV-cache GEMV": 0.0,
                "Communication": 0.0,
                "Elementwise/Norm": 0.0,
                "Other": other_cycles,
            },
        }
    )
    category_totals["Other"] = other_cycles
    return bars, category_totals, measured_total, other_cycles


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
    parser = argparse.ArgumentParser(description="Decode WSE-3 launcher")
    parser.add_argument("--config", default="config.json", type=str, help="Config file")
    parser.add_argument("--simulator", action="store_true", help="Runs on appliance simulator")
    parser.add_argument(
        "--artifact-dir",
        default=None,
        type=str,
        help="Directory to store profiling artifacts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    if not os.path.exists(args.config):
        print("Host: Use default test values.")
    else:
        with open(args.config) as f:
            config.__dict__.update(json.load(f))

    P = config.P
    group_num = config.group_num
    pe_num_p_group = P // group_num
    bsz = config.bsz
    dim = config.dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    seq_len = config.seq_len
    ffn_dim = config.ffn_dim
    layer_num = config.layer_num

    artifact_dir = None
    if args.artifact_dir:
        artifact_dir = os.path.abspath(args.artifact_dir)
        os.makedirs(artifact_dir, exist_ok=True)

    dim_p_pe = dim // P
    pes_p_head = P // n_heads
    pes_p_kv_head = P // n_kv_heads
    head_dim_p_pe = head_dim // P
    seq_len_p_pe = seq_len // P
    ffn_dim_p_pe = ffn_dim // P

    print(
        f"Host: P: {P}, Batch size: {bsz}, dim_p_pe: {dim_p_pe}, "
        f"pes_p_head: {pes_p_head}, pes_p_kv_head: {pes_p_kv_head}, "
        f"head_dim_p_pe: {head_dim_p_pe}, seq_len_p_pe: {seq_len_p_pe}, "
        f"ffn_dim_p_pe: {ffn_dim_p_pe}, simulator: {args.simulator}"
    )

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    X = np.random.rand(1, bsz * dim).astype(np.float16)
    tensor_X = np.tile(X.reshape(P, bsz * dim_p_pe), reps=(1, P))

    W = np.random.rand(1, dim).astype(np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))

    tensor_q_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_k_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_v_weight = np.random.rand(dim, dim).astype(np.float16)

    _dim_p_pe = dim_p_pe
    if (dim_p_pe % 2) == 1:
        _dim_p_pe = dim_p_pe - 1

    freqs_sin = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
    tensor_freqs_sin = np.tile(freqs_sin.reshape(P, _dim_p_pe // 2), reps=(1, P))
    freqs_cos = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
    tensor_freqs_cos = np.tile(freqs_cos.reshape(P, _dim_p_pe // 2), reps=(1, P))

    tensor_XKCache = np.random.rand(dim, seq_len).astype(np.float16)
    tensor_XVCache = np.random.rand(seq_len, dim).astype(np.float16)

    tensor_o_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_up_weight = np.random.rand(dim, ffn_dim).astype(np.float16)
    tensor_gate_weight = np.random.rand(dim, ffn_dim).astype(np.float16)
    tensor_down_weight = np.random.rand(ffn_dim, dim).astype(np.float16)

    with open(f"{out_path}/artifact_{P}_{P // pe_num_p_group}.json", "r", encoding="utf8") as f:
        data = json.load(f)
        artifact_path = data["artifact_id"]

    with SdkRuntime(artifact_path, simulator=args.simulator, disable_version_check=True) as runner:
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

        symbol_timer_buf = runner.get_id("timer_buf")
        symbol_phase_cycles = runner.get_id("phase_cycles")
        symbol_phase_flop_equiv = runner.get_id("phase_flop_equiv")
        symbol_phase_local_mem_bytes = runner.get_id("phase_local_mem_bytes")
        symbol_phase_working_set_bytes = runner.get_id("phase_working_set_bytes")
        symbol_comm_subphase_cycles = runner.get_id("comm_subphase_cycles")
        sym_debug = runner.get_id("debug")

        X_u32 = cast_tensor_u32(tensor_X.ravel())
        runner.memcpy_h2d(
            sym_X, X_u32, 0, 0, P, P, bsz * dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        W_u32 = cast_tensor_u32(tensor_W.ravel())
        runner.memcpy_h2d(
            sym_W, W_u32, 0, 0, P, P, dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        Q_reshape = tensor_q_weight.reshape(P, dim_p_pe, P, dim_p_pe)
        Q_transpose = Q_reshape.transpose(0, 2, 1, 3)
        Q_reshape = Q_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
        Q_u32 = cast_tensor_u32(Q_reshape.ravel())
        runner.memcpy_h2d(
            sym_Q_weight, Q_u32, 0, 0, P, P, dim_p_pe * dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        K_reshape = tensor_k_weight.reshape(P, dim_p_pe, P, dim_p_pe)
        K_transpose = K_reshape.transpose(0, 2, 1, 3)
        K_reshape = K_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
        K_u32 = cast_tensor_u32(K_reshape.ravel())
        runner.memcpy_h2d(
            sym_K_weight, K_u32, 0, 0, P, P, dim_p_pe * dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        V_reshape = tensor_v_weight.reshape(P, dim_p_pe, P, dim_p_pe)
        V_transpose = V_reshape.transpose(0, 2, 1, 3)
        V_reshape = V_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
        V_u32 = cast_tensor_u32(V_reshape.ravel())
        runner.memcpy_h2d(
            sym_V_weight, V_u32, 0, 0, P, P, dim_p_pe * dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        freqs_sin_u32 = cast_tensor_u32(tensor_freqs_sin.ravel())
        runner.memcpy_h2d(
            sym_freqs_sin, freqs_sin_u32, 0, 0, P, P, _dim_p_pe // 2,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        freqs_cos_u32 = cast_tensor_u32(tensor_freqs_cos.ravel())
        runner.memcpy_h2d(
            sym_freqs_cos, freqs_cos_u32, 0, 0, P, P, _dim_p_pe // 2,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        XKCache_reshape = tensor_XKCache.reshape(P, dim_p_pe, P, seq_len_p_pe)
        XKCache_transpose = XKCache_reshape.transpose(0, 2, 1, 3)
        XKCache_reshape = XKCache_transpose.reshape(P, P, dim_p_pe * seq_len_p_pe)
        XKCache_u32 = cast_tensor_u32(XKCache_reshape.ravel())
        runner.memcpy_h2d(
            sym_XKCache, XKCache_u32, 0, 0, P, P, dim_p_pe * seq_len_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        XVCache_reshape = tensor_XVCache.reshape(P, seq_len_p_pe, P, dim_p_pe)
        XVCache_transpose = XVCache_reshape.transpose(0, 2, 1, 3)
        XVCache_reshape = XVCache_transpose.reshape(P, P, seq_len_p_pe * dim_p_pe)
        XVCache_u32 = cast_tensor_u32(XVCache_reshape.ravel())
        runner.memcpy_h2d(
            sym_XVCache, XVCache_u32, 0, 0, P, P, seq_len_p_pe * dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        O_reshape = tensor_o_weight.reshape(P, dim_p_pe, P, dim_p_pe)
        O_transpose = O_reshape.transpose(0, 2, 1, 3)
        O_reshape = O_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
        O_u32 = cast_tensor_u32(O_reshape.ravel())
        runner.memcpy_h2d(
            sym_O_weight, O_u32, 0, 0, P, P, dim_p_pe * dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        UP_reshape = tensor_up_weight.reshape(P, dim_p_pe, P, ffn_dim_p_pe)
        UP_transpose = UP_reshape.transpose(0, 2, 1, 3)
        UP_reshape = UP_transpose.reshape(P, P, dim_p_pe * ffn_dim_p_pe)
        UP_u32 = cast_tensor_u32(UP_reshape.ravel())
        runner.memcpy_h2d(
            sym_UP_weight, UP_u32, 0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        GATE_reshape = tensor_gate_weight.reshape(P, dim_p_pe, P, ffn_dim_p_pe)
        GATE_transpose = GATE_reshape.transpose(0, 2, 1, 3)
        GATE_reshape = GATE_transpose.reshape(P, P, dim_p_pe * ffn_dim_p_pe)
        GATE_u32 = cast_tensor_u32(GATE_reshape.ravel())
        runner.memcpy_h2d(
            sym_GATE_weight, GATE_u32, 0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        DOWN_reshape = tensor_down_weight.reshape(P, ffn_dim_p_pe, P, dim_p_pe)
        DOWN_transpose = DOWN_reshape.transpose(0, 2, 1, 3)
        DOWN_reshape = DOWN_transpose.reshape(P, P, ffn_dim_p_pe * dim_p_pe)
        DOWN_u32 = cast_tensor_u32(DOWN_reshape.ravel())
        runner.memcpy_h2d(
            sym_DOWN_weight, DOWN_u32, 0, 0, P, P, ffn_dim_p_pe * dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

        runner.launch("init_task", nonblock=False)

        warmup_steps = 5
        repeat_steps = 50
        runner.launch("decode_host", np.int16(warmup_steps), np.int16(repeat_steps), nonblock=False)

        debug_1d_u32 = np.zeros(P * bsz * dim, dtype=np.uint32)
        runner.memcpy_d2h(
            debug_1d_u32, sym_debug, 0, 0, P, P, bsz * dim_p_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        debug = sdk_utils.memcpy_view(debug_1d_u32, np.dtype(np.float16))
        debug = debug.reshape(P, bsz * dim)

        timer_buf_1d_u32 = np.zeros((P * P * 3), dtype=np.uint32)
        runner.memcpy_d2h(
            timer_buf_1d_u32, symbol_timer_buf, 0, 0, P, P, 3, streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False
        )
        timer_buf_words_u32 = timer_buf_1d_u32.reshape((P, P, 3))
        timer_buf_time_hwl = timer_buf_words_u32.view(np.float32).reshape((P, P, 3))

        phase_cycles_1d_f32 = np.zeros((P * P * len(PHASE_NAMES)), dtype=np.float32)
        runner.memcpy_d2h(
            phase_cycles_1d_f32,
            symbol_phase_cycles,
            0,
            0,
            P,
            P,
            len(PHASE_NAMES),
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        phase_cycles = phase_cycles_1d_f32.reshape((P, P, len(PHASE_NAMES)))

        phase_flop_equiv_1d_f32 = np.zeros((P * P * len(PHASE_NAMES)), dtype=np.float32)
        runner.memcpy_d2h(
            phase_flop_equiv_1d_f32,
            symbol_phase_flop_equiv,
            0,
            0,
            P,
            P,
            len(PHASE_NAMES),
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        phase_flop_equiv = phase_flop_equiv_1d_f32.reshape((P, P, len(PHASE_NAMES)))

        phase_local_mem_bytes_1d_f32 = np.zeros((P * P * len(PHASE_NAMES)), dtype=np.float32)
        runner.memcpy_d2h(
            phase_local_mem_bytes_1d_f32,
            symbol_phase_local_mem_bytes,
            0,
            0,
            P,
            P,
            len(PHASE_NAMES),
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        phase_local_mem_bytes = phase_local_mem_bytes_1d_f32.reshape((P, P, len(PHASE_NAMES)))

        phase_working_set_bytes_1d_f32 = np.zeros((P * P * len(PHASE_NAMES)), dtype=np.float32)
        runner.memcpy_d2h(
            phase_working_set_bytes_1d_f32,
            symbol_phase_working_set_bytes,
            0,
            0,
            P,
            P,
            len(PHASE_NAMES),
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        phase_working_set_bytes = phase_working_set_bytes_1d_f32.reshape((P, P, len(PHASE_NAMES)))

        comm_subphase_cycles_1d_f32 = np.zeros((P * P * len(COMM_SUBPHASE_NAMES)), dtype=np.float32)
        runner.memcpy_d2h(
            comm_subphase_cycles_1d_f32,
            symbol_comm_subphase_cycles,
            0,
            0,
            P,
            P,
            len(COMM_SUBPHASE_NAMES),
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        comm_subphase_cycles = comm_subphase_cycles_1d_f32.reshape((P, P, len(COMM_SUBPHASE_NAMES)))

    cycles_count = np.zeros((P, P))
    for pe_x in range(P):
        for pe_y in range(P):
            cycles_count[pe_y, pe_x] = calculate_cycles_from_words(timer_buf_words_u32[pe_y, pe_x, :])

    cycles_per_step = cycles_count / repeat_steps
    cycles_flat = cycles_per_step.ravel()
    cycles_count_mean = float(cycles_flat.mean())
    cycles_count_min = float(cycles_flat.min())
    cycles_count_max = float(cycles_flat.max())
    cycles_count_p95 = float(np.percentile(cycles_flat, 95))
    cycles_count_std = float(cycles_flat.std())
    cycles_count_cv = float(cycles_count_std / cycles_count_mean) if cycles_count_mean != 0 else 0.0
    cycles_count_imbalance = float(cycles_count_max / cycles_count_mean) if cycles_count_mean != 0 else 0.0

    phase_cycles_per_step = phase_cycles / repeat_steps
    phase_means = phase_cycles_per_step.mean(axis=(0, 1))
    phase_summary = {name: float(phase_means[idx]) for idx, name in enumerate(PHASE_NAMES)}
    phase_flop_equiv_per_step = phase_flop_equiv / repeat_steps
    phase_local_mem_bytes_per_step = phase_local_mem_bytes / repeat_steps
    phase_working_set_bytes_per_step = phase_working_set_bytes / repeat_steps
    phase_flop_equiv_means = phase_flop_equiv_per_step.mean(axis=(0, 1))
    phase_local_mem_bytes_means = phase_local_mem_bytes_per_step.mean(axis=(0, 1))
    phase_working_set_bytes_means = phase_working_set_bytes_per_step.mean(axis=(0, 1))
    phase_property_summary = {}
    for idx, name in enumerate(PHASE_NAMES):
        flop_equiv = float(phase_flop_equiv_means[idx])
        local_mem_bytes = float(phase_local_mem_bytes_means[idx])
        working_set_bytes = float(phase_working_set_bytes_means[idx])
        phase_property_summary[name] = {
            "flop_equiv": flop_equiv,
            "local_mem_bytes": local_mem_bytes,
            "working_set_bytes": working_set_bytes,
            "intensity_flop_per_byte": float(flop_equiv / local_mem_bytes) if local_mem_bytes else 0.0,
            "sram_fit_pct_of_48kb": float(100.0 * working_set_bytes / 49152.0),
        }
    comm_subphase_cycles_per_step = comm_subphase_cycles / repeat_steps
    comm_subphase_means = comm_subphase_cycles_per_step.mean(axis=(0, 1))
    comm_subphase_summary = {
        name: float(comm_subphase_means[idx]) for idx, name in enumerate(COMM_SUBPHASE_NAMES)
    }
    phase_bars, category_totals, measured_phase_cycles, other_cycles = aggregate_phase_groups(
        phase_means,
        cycles_count_mean,
    )

    freq_ghz = 1.1
    model_cycles_mean = cycles_count_mean * layer_num
    throughput_p_request = 1 / (model_cycles_mean / (freq_ghz * 1e9))

    print(f"\nRepeat count: {repeat_steps}")
    print(f"Mean block cycles count: {cycles_count_mean}")
    print(f"Mean model cycles count: {model_cycles_mean}")
    print(f"Throughput_p_request: {throughput_p_request}")
    print(f"Max block cycles count: {cycles_count_max}")
    print(f"P95 block cycles count: {cycles_count_p95}")
    print(f"Std block cycles count: {cycles_count_std}")
    print(f"Max/mean imbalance: {cycles_count_imbalance}")
    print("Host: mean phase cycles:")
    for name in PHASE_NAMES:
        print(f"  - {name}: {phase_summary[name]:.3f}")
    print("Host: mean local-phase properties:")
    for name in PHASE_NAMES:
        props = phase_property_summary[name]
        print(
            f"  - {name}: flop_equiv={props['flop_equiv']:.3f}, "
            f"local_mem_bytes={props['local_mem_bytes']:.3f}, "
            f"working_set_bytes={props['working_set_bytes']:.3f}"
        )
    print("Host: mean communication subphase cycles:")
    for name in COMM_SUBPHASE_NAMES:
        print(f"  - {name}: {comm_subphase_summary[name]:.3f}")
    print(f"Host: measured phase sum: {measured_phase_cycles:.3f}")
    print(f"Host: residual other cycles: {other_cycles:.3f}")

    if artifact_dir is not None:
        manifest = {
            "config_path": os.path.abspath(args.config),
            "target": "appliance_simulator" if args.simulator else "wse3",
            "config": {
                "P": P,
                "bsz": bsz,
                "group_num": group_num,
                "dim": dim,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "head_dim": head_dim,
                "seq_len": seq_len,
                "ffn_dim": ffn_dim,
                "layer_num": layer_num,
            },
            "derived": {
                "dim_p_pe": dim_p_pe,
                "pes_p_head": pes_p_head,
                "pes_p_kv_head": pes_p_kv_head,
                "head_dim_p_pe": head_dim_p_pe,
                "seq_len_p_pe": seq_len_p_pe,
                "ffn_dim_p_pe": ffn_dim_p_pe,
                "repeat_steps": repeat_steps,
                "warmup_steps": warmup_steps,
                "phase_property_method": "runtime-accounted kernel formulas exported by decode.csl; cycles are measured, flop/byte/working-set properties are code-accounted rather than hardware perf counters",
            },
        }

        metrics = {
            "mean_cycles": cycles_count_mean,
            "min_cycles": cycles_count_min,
            "max_cycles": cycles_count_max,
            "p95_cycles": cycles_count_p95,
            "std_cycles": cycles_count_std,
            "cv_cycles": cycles_count_cv,
            "max_mean_imbalance": cycles_count_imbalance,
            "measured_phase_cycles": float(measured_phase_cycles),
            "other_cycles": float(other_cycles),
            "model_mean_cycles": float(model_cycles_mean),
            "throughput_p_request": float(throughput_p_request),
        }

        write_json(os.path.join(artifact_dir, "manifest.json"), manifest)
        write_json(os.path.join(artifact_dir, "metrics.json"), metrics)
        write_json(os.path.join(artifact_dir, "phase_summary.json"), phase_summary)
        write_json(os.path.join(artifact_dir, "phase_property_summary.json"), phase_property_summary)
        write_json(os.path.join(artifact_dir, "comm_subphase_summary.json"), comm_subphase_summary)
        write_json(os.path.join(artifact_dir, "category_summary.json"), category_totals)
        write_json(os.path.join(artifact_dir, "phase_group_summary.json"), phase_bars)
        np.save(os.path.join(artifact_dir, "cycles_count.npy"), cycles_per_step)
        np.savetxt(
            os.path.join(artifact_dir, "cycles_count.csv"),
            cycles_per_step,
            delimiter=",",
            fmt="%.6f",
        )
        np.save(os.path.join(artifact_dir, "phase_cycles.npy"), phase_cycles_per_step)
        np.save(os.path.join(artifact_dir, "phase_flop_equiv.npy"), phase_flop_equiv_per_step)
        np.save(os.path.join(artifact_dir, "phase_local_mem_bytes.npy"), phase_local_mem_bytes_per_step)
        np.save(os.path.join(artifact_dir, "phase_working_set_bytes.npy"), phase_working_set_bytes_per_step)
        np.save(os.path.join(artifact_dir, "comm_subphase_cycles.npy"), comm_subphase_cycles_per_step)
        np.savetxt(
            os.path.join(artifact_dir, "phase_cycles_mean.csv"),
            phase_means.reshape(1, -1),
            delimiter=",",
            fmt="%.6f",
            header=",".join(PHASE_NAMES),
            comments="",
        )
        np.savetxt(
            os.path.join(artifact_dir, "phase_flop_equiv_mean.csv"),
            phase_flop_equiv_means.reshape(1, -1),
            delimiter=",",
            fmt="%.6f",
            header=",".join(PHASE_NAMES),
            comments="",
        )
        np.savetxt(
            os.path.join(artifact_dir, "phase_local_mem_bytes_mean.csv"),
            phase_local_mem_bytes_means.reshape(1, -1),
            delimiter=",",
            fmt="%.6f",
            header=",".join(PHASE_NAMES),
            comments="",
        )
        np.savetxt(
            os.path.join(artifact_dir, "phase_working_set_bytes_mean.csv"),
            phase_working_set_bytes_means.reshape(1, -1),
            delimiter=",",
            fmt="%.6f",
            header=",".join(PHASE_NAMES),
            comments="",
        )
        np.savetxt(
            os.path.join(artifact_dir, "comm_subphase_cycles_mean.csv"),
            comm_subphase_means.reshape(1, -1),
            delimiter=",",
            fmt="%.6f",
            header=",".join(COMM_SUBPHASE_NAMES),
            comments="",
        )
        np.save(os.path.join(artifact_dir, "timer_buf_time_hwl.npy"), timer_buf_time_hwl)
        np.save(os.path.join(artifact_dir, "expected_input.npy"), X)
        np.save(os.path.join(artifact_dir, "debug_output.npy"), debug)
        print(f"Host: profiling artifact dir: {artifact_dir}")


if __name__ == "__main__":
    main()
