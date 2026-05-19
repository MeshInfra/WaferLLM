import json
import os
import argparse
import csv
import time
from time import perf_counter_ns
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


def parse_size(value):
    raw = value.strip()
    lower = raw.lower()
    multipliers = {
        "kib": 1024,
        "kb": 1000,
        "mib": 1024 * 1024,
        "mb": 1000 * 1000,
        "b": 1,
    }
    for suffix, mult in multipliers.items():
        if lower.endswith(suffix):
            return int(float(raw[: -len(suffix)].strip()) * mult)
    return int(raw)


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_size_list(value):
    return [parse_size(item) for item in value.split(",") if item.strip()]


def parse_bool_list(value):
    if value == "true":
        return [True]
    if value == "false":
        return [False]
    return [False, True]


def fmt_size(num_bytes):
    if num_bytes % (1024 * 1024) == 0:
        return f"{num_bytes // (1024 * 1024)}MiB"
    if num_bytes % 1024 == 0:
        return f"{num_bytes // 1024}KiB"
    return f"{num_bytes}B"


def payload_to_local_len(payload_bytes, P, dtype_bytes=2):
    bytes_per_local_element = P * P * dtype_bytes
    if payload_bytes % bytes_per_local_element != 0:
        raise ValueError(
            f"payload_bytes={payload_bytes} must be divisible by "
            f"P*P*dtype_bytes={bytes_per_local_element}"
        )
    local_len = payload_bytes // bytes_per_local_element
    if local_len < 1:
        raise ValueError("payload maps to less than one local element per PE")
    return local_len


def make_h2d_bench_points(args):
    nonblocks = parse_bool_list(args.h2d_nonblock)
    points = []
    payloads = (
        parse_size_list(args.h2d_payload_bytes)
        if args.h2d_payload_bytes
        else [
            4 * 1024,
            8 * 1024,
            16 * 1024,
            32 * 1024,
            64 * 1024,
            128 * 1024,
            256 * 1024,
            512 * 1024,
            1024 * 1024,
            2 * 1024 * 1024,
            4 * 1024 * 1024,
            8 * 1024 * 1024,
        ]
    )

    if args.h2d_preset == "smoke":
        for nonblock in nonblocks:
            points.append(("smoke_64KiB", 64 * 1024, 1, 1, nonblock))
            points.append(("smoke_1MiB", 1024 * 1024, 1, 1, nonblock))
    elif args.h2d_preset == "payload-sweep":
        for payload in payloads:
            for loop_count in parse_int_list(args.h2d_loop_counts):
                for nonblock in nonblocks:
                    points.append((f"payload_{fmt_size(payload)}", payload, 1, loop_count, nonblock))
    elif args.h2d_preset == "chunk-size":
        same_total_bytes = parse_size(args.h2d_same_total_bytes)
        for payload in [4 * 1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024]:
            if same_total_bytes % payload != 0:
                continue
            for nonblock in nonblocks:
                points.append((f"chunk_{fmt_size(payload)}", payload, 1, same_total_bytes // payload, nonblock))
    elif args.h2d_preset == "multi-stream":
        payload = parse_size(args.h2d_stream_payload_bytes)
        for streams in parse_int_list(args.h2d_streams):
            for nonblock in nonblocks:
                points.append((f"streams_{streams}", payload, streams, 1, nonblock))
    elif args.h2d_preset == "all":
        original = args.h2d_preset
        for preset in ["payload-sweep", "chunk-size", "multi-stream"]:
            args.h2d_preset = preset
            points.extend(make_h2d_bench_points(args))
        args.h2d_preset = original
    return points


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_timer_words(runner, symbol_timer_buf, P):
    timer_buf_1d_u32 = np.zeros((P * P * 3), dtype=np.uint32)
    runner.memcpy_d2h(
        timer_buf_1d_u32,
        symbol_timer_buf,
        0,
        0,
        P,
        P,
        3,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    return timer_buf_1d_u32.reshape((P, P, 3))


def issue_h2d_copies(runner, symbol_ids, data_u32, P, local_len, count, nonblock, io_dtype, memcpy_order):
    for idx in range(count):
        symbol_id = symbol_ids[idx % len(symbol_ids)]
        runner.memcpy_h2d(
            symbol_id,
            data_u32,
            0,
            0,
            P,
            P,
            local_len,
            streaming=False,
            data_type=io_dtype,
            order=memcpy_order,
            nonblock=nonblock,
        )


def run_h2d_bench(runner, symbol_map, symbol_timer_buf, args, P, io_dtype, memcpy_order):
    selected_names = [name.strip() for name in args.h2d_symbols.split(",") if name.strip()]
    missing = [name for name in selected_names if name not in symbol_map]
    if missing:
        raise ValueError(f"Unknown H2D symbols: {missing}. Available: {sorted(symbol_map)}")

    selected = [symbol_map[name] for name in selected_names]
    max_local_len = min(item["local_len"] for item in selected)
    max_payload_bytes = P * P * max_local_len * 2
    points = make_h2d_bench_points(args)
    rows = []

    output_dir = args.h2d_output_dir
    if output_dir is None:
        output_dir = os.path.join("h2d_bench_runs", "h2d_" + time.strftime("%Y%m%d_%H%M%S"))
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    runner.launch("init_task", nonblock=False)

    for case_name, payload_bytes, streams, loop_count, nonblock in points:
        local_len = payload_to_local_len(payload_bytes, P, dtype_bytes=2)
        if local_len > max_local_len:
            raise ValueError(
                f"{case_name}: payload {fmt_size(payload_bytes)} requires local_len={local_len}, "
                f"but selected symbols {selected_names} only support local_len <= {max_local_len} "
                f"({fmt_size(max_payload_bytes)} max payload)."
            )

        total_elements = P * P * local_len
        data_u32 = cast_tensor_u32(np.zeros(total_elements, dtype=np.float16))
        symbol_ids = [item["id"] for item in selected]
        h2d_count = streams * loop_count
        total_bytes = payload_bytes * h2d_count

        if args.h2d_warmup:
            runner.launch("h2d_bench_tic", nonblock=False)
            issue_h2d_copies(runner, symbol_ids, data_u32, P, local_len, args.h2d_warmup, False, io_dtype, memcpy_order)
            runner.launch("h2d_bench_toc", nonblock=False)

        for sample in range(args.h2d_samples):
            runner.launch("h2d_bench_tic", nonblock=False)

            host_issue_start_ns = perf_counter_ns()
            issue_h2d_copies(runner, symbol_ids, data_u32, P, local_len, h2d_count, nonblock, io_dtype, memcpy_order)
            host_issue_end_ns = perf_counter_ns()

            host_wall_start_ns = host_issue_start_ns
            runner.launch("h2d_bench_toc", nonblock=False)
            host_wall_end_ns = perf_counter_ns()

            timer_words_u32 = read_timer_words(runner, symbol_timer_buf, P)
            cycles = np.zeros((P, P), dtype=np.float64)
            for pe_y in range(P):
                for pe_x in range(P):
                    cycles[pe_y, pe_x] = calculate_cycles_from_words(timer_words_u32[pe_y, pe_x, :])

            device_tsc_us = float(cycles.max() / args.h2d_tsc_mhz)
            host_issue_us = float((host_issue_end_ns - host_issue_start_ns) / 1000.0)
            host_wall_us = float((host_wall_end_ns - host_wall_start_ns) / 1000.0)

            row = {
                "case": case_name,
                "sample": sample,
                "P": P,
                "symbols": ",".join(selected_names),
                "payload_bytes": payload_bytes,
                "local_len": local_len,
                "streams": streams,
                "loop_count": loop_count,
                "h2d_count": h2d_count,
                "nonblock": nonblock,
                "total_bytes": total_bytes,
                "host_issue_us": host_issue_us,
                "host_wall_us": host_wall_us,
                "device_tsc_us": device_tsc_us,
                "effective_bandwidth_GBps": total_bytes / host_wall_us / 1000.0 if host_wall_us else 0.0,
                "device_bandwidth_GBps": total_bytes / device_tsc_us / 1000.0 if device_tsc_us else 0.0,
            }
            rows.append(row)
            print(
                f"H2D bench {case_name} sample={sample} payload={fmt_size(payload_bytes)} "
                f"streams={streams} loops={loop_count} nonblock={nonblock} "
                f"host_wall={host_wall_us:.3f}us device={device_tsc_us:.3f}us",
                flush=True,
            )

    metadata = {
        "config": os.path.abspath(args.config),
        "simulator": args.simulator,
        "preset": args.h2d_preset,
        "symbols": selected_names,
        "max_payload_bytes": max_payload_bytes,
        "tsc_mhz": args.h2d_tsc_mhz,
    }
    write_json(os.path.join(output_dir, "h2d_bench_manifest.json"), metadata)
    write_json(os.path.join(output_dir, "h2d_bench_results.json"), {"metadata": metadata, "rows": rows})
    write_csv(os.path.join(output_dir, "h2d_bench_results.csv"), rows)
    print(f"Host: H2D benchmark results: {output_dir}", flush=True)


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
    parser.add_argument("--h2d-bench", action="store_true", help="Run WaferLLM-integrated H2D copy benchmark instead of decode")
    parser.add_argument(
        "--h2d-preset",
        choices=["smoke", "payload-sweep", "chunk-size", "multi-stream", "all"],
        default="smoke",
    )
    parser.add_argument("--h2d-symbols", default="XKCache", help="Comma-separated existing WaferLLM symbols to copy into")
    parser.add_argument("--h2d-payload-bytes", default=None, help="Comma-separated payload sizes, e.g. 4KiB,64KiB,1MiB")
    parser.add_argument("--h2d-loop-counts", default="1,4,16,64", help="Comma-separated loop counts")
    parser.add_argument("--h2d-same-total-bytes", default="8MiB", help="Total bytes for chunk-size preset")
    parser.add_argument("--h2d-stream-payload-bytes", default="8MiB", help="Per-stream payload for multi-stream preset")
    parser.add_argument("--h2d-streams", default="1,2,4,8,16", help="Comma-separated stream counts")
    parser.add_argument("--h2d-nonblock", choices=["false", "true", "both"], default="both")
    parser.add_argument("--h2d-samples", type=int, default=3)
    parser.add_argument("--h2d-warmup", type=int, default=1)
    parser.add_argument("--h2d-tsc-mhz", type=float, default=850.0)
    parser.add_argument("--h2d-output-dir", default=None, type=str)
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
    _dim_p_pe = dim_p_pe
    if (dim_p_pe % 2) == 1:
        _dim_p_pe = dim_p_pe - 1

    with open(f"{out_path}/artifact_{P}_{P // pe_num_p_group}.json", "r", encoding="utf8") as f:
        data = json.load(f)
        artifact_path = data["artifact_id"]

    if args.h2d_bench:
        with SdkRuntime(artifact_path, simulator=args.simulator, disable_version_check=True) as runner:
            h2d_symbol_map = {
                "X": {"id": runner.get_id("X"), "local_len": bsz * dim_p_pe},
                "W": {"id": runner.get_id("W"), "local_len": dim_p_pe},
                "Q_weight": {"id": runner.get_id("Q_weight"), "local_len": dim_p_pe * dim_p_pe},
                "K_weight": {"id": runner.get_id("K_weight"), "local_len": dim_p_pe * dim_p_pe},
                "V_weight": {"id": runner.get_id("V_weight"), "local_len": dim_p_pe * dim_p_pe},
                "freqs_sin": {"id": runner.get_id("freqs_sin"), "local_len": _dim_p_pe // 2},
                "freqs_cos": {"id": runner.get_id("freqs_cos"), "local_len": _dim_p_pe // 2},
                "XKCache": {"id": runner.get_id("XKCache"), "local_len": dim_p_pe * seq_len_p_pe},
                "XVCache": {"id": runner.get_id("XVCache"), "local_len": seq_len_p_pe * dim_p_pe},
                "O_weight": {"id": runner.get_id("O_weight"), "local_len": dim_p_pe * dim_p_pe},
                "UP_weight": {"id": runner.get_id("UP_weight"), "local_len": dim_p_pe * ffn_dim_p_pe},
                "GATE_weight": {"id": runner.get_id("GATE_weight"), "local_len": dim_p_pe * ffn_dim_p_pe},
                "DOWN_weight": {"id": runner.get_id("DOWN_weight"), "local_len": ffn_dim_p_pe * dim_p_pe},
            }
            symbol_timer_buf = runner.get_id("timer_buf")
            run_h2d_bench(runner, h2d_symbol_map, symbol_timer_buf, args, P, io_dtype, memcpy_order)
        return

    X = np.random.rand(1, bsz * dim).astype(np.float16)
    tensor_X = np.tile(X.reshape(P, bsz * dim_p_pe), reps=(1, P))

    W = np.random.rand(1, dim).astype(np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))

    tensor_q_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_k_weight = np.random.rand(dim, dim).astype(np.float16)
    tensor_v_weight = np.random.rand(dim, dim).astype(np.float16)

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
