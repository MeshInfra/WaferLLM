"""Shared host-side numpy helpers for SUMMA (no Cerebras SDK imports).

Imported by both launch_sim.py (cs_python / runtime SDK) and launch_device.py
(appliance python / client SDK). SUMMA uses the same tiling as MeshGEMM but, being
a broadcast-multiply algorithm, needs no weight row-shuffle.
"""
import struct

import numpy as np


def cfg_name(P, M, K, N):
    """Per-config suffix used for compile artifacts (out_<cfg>, artifact_<cfg>.json)."""
    return f"{P}_{M // P}_{K // P}_{N // P}"


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)


def make_inputs(P, M, K, N):
    tensor_X = np.random.rand(M, K).astype(np.float16)
    tensor_W = np.random.rand(K, N).astype(np.float16)
    return tensor_X, tensor_W


def tile_X(tensor_X, P, Mt, Kt):
    X1 = tensor_X.reshape(P, Mt, P, Kt)
    X2 = X1.transpose(0, 2, 3, 1)
    return X2.reshape(P, P, Mt * Kt).ravel()


def tile_W(tensor_W, P, Kt, Nt):
    W1 = tensor_W.reshape(P, Kt, P, Nt)
    W2 = W1.transpose(0, 2, 1, 3)
    return W2.reshape(P, P, Kt * Nt).ravel()


def untile_res(res_1d_fp16, P, Mt, Nt):
    res3 = res_1d_fp16.reshape((P, P, Nt, Mt))
    res2 = res3.transpose(0, 3, 1, 2)
    return res2.reshape(P * Mt, P * Nt)


def decode_timing(time_memcpy_hwl, time_ref_hwl, P, total_repeat_times):
    """Decode the on-chip TSC timestamps into (mean_cycle, max_cycle)."""
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
    mean_cycle = np.mean(time_end - time_start) / total_repeat_times
    max_cycle = (max_time_end - min_time_start) / total_repeat_times
    return mean_cycle, max_cycle
