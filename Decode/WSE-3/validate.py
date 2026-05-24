"""
verify_decode.py

decode.csl の decode_struct() に対応する NumPy 参照実装。
launch_sim.py と同一の変数名を使用し、CSL シミュレーション出力との比較に用いる。
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np


# ============================================================================ #
# fast_exp: decode.csl の fast_exp(x) = (1 + x/256)^4 を再現
#   2 回の自乗演算のみ使用 (注: e^x の近似精度は低い)
# ============================================================================ #
def fast_exp_np(x: np.ndarray) -> np.ndarray:
    tmp = np.float16(1.0) + x.astype(np.float16) / np.float16(256.0)
    tmp = (tmp * tmp).astype(np.float16)
    tmp = (tmp * tmp).astype(np.float16)
    return tmp


# ============================================================================ #
# RMSNorm: decode.csl rmsnorm_x / rmsnorm_z に対応
# 本関数は意図された標準 RMSNorm: x * W / sqrt(Σx²/head_dim + ε) を実装する。
# ============================================================================ #
def rmsnorm(x: np.ndarray, W: np.ndarray, head_dim: int, eps: float = 1e-6) -> np.ndarray:
    # x: (bsz, dim),  W: (dim,)
    sum_sq: np.ndarray = np.sum((x * x).astype(np.float32), axis=-1, keepdims=True)   # (bsz, 1)
    inv_rms: np.ndarray = (np.float32(1.0) / np.sqrt(sum_sq / head_dim + eps)).astype(np.float16)
    return (x * W * inv_rms).astype(np.float16)


# ============================================================================ #
# RoPE: decode.csl xq_rope / xk_rope に対応
#
# CSL コードの変換式 (標準 LLaMA RoPE と even/odd の役割が逆):
#   tmp1 = x_odd  * cos  → X_tmp_1
#   tmp2 = x_even * sin  → X_tmp_2
#   tmp3 = x_even * cos  → X_tmp_3
#   tmp4 = x_odd  * sin  → X_tmp_4
#   new_even = tmp1 - tmp2 = x_odd  * cos - x_even * sin
#   new_odd  = tmp3 + tmp4 = x_even * cos + x_odd  * sin
# ============================================================================ #
def rope(x: np.ndarray, freqs_cos: np.ndarray, freqs_sin: np.ndarray) -> np.ndarray:
    # x: (bsz, dim),  freqs_cos/sin: (dim//2,)
    x_even = x[:, 0::2]                                                     # (bsz, dim//2)
    x_odd  = x[:, 1::2]                                                     # (bsz, dim//2)
    new_even = (x_odd * freqs_cos - x_even * freqs_sin).astype(np.float16)
    new_odd  = (x_even * freqs_cos + x_odd  * freqs_sin).astype(np.float16)
    out = np.empty_like(x)
    out[:, 0::2] = new_even
    out[:, 1::2] = new_odd
    return out


# ============================================================================ #
# Softmax: decode.csl softmax_score に対応 (fast_exp 近似使用)
# ============================================================================ #
def softmax_fast_exp(scores: np.ndarray) -> np.ndarray:
    # scores: (bsz, seq_len)
    max_s   = np.max(scores, axis=-1, keepdims=True)                        # (bsz, 1)
    shifted = (scores - max_s).astype(np.float16)
    # exp_s   = fast_exp_np(shifted)
    exp_s = np.exp(shifted)
    return (exp_s / exp_s.sum(axis=-1, keepdims=True)).astype(np.float16)


# ============================================================================ #
# SiLU (fast_exp 版): decode.csl z2_silu / silu_kernel に対応
# ============================================================================ #
def silu_fast_exp(x: np.ndarray) -> np.ndarray:
    # return (x / (np.float16(1.0) + fast_exp_np(-x.astype(np.float16)))).astype(np.float16)
    return (x / (np.float16(1.0) + np.exp(-x.astype(np.float16)))).astype(np.float16)


# ============================================================================ #
# Llama3 Attention Block デコード参照実装
# ============================================================================ #
def decode_block(
    X: np.ndarray,                   # (bsz, dim)       入力トークン埋め込み
    W: np.ndarray,                   # (dim,)            RMSNorm weights  ← launch_sim: W.reshape(dim)
    tensor_q_weight: np.ndarray,     # (dim, dim)
    tensor_k_weight: np.ndarray,     # (dim, dim)
    tensor_v_weight: np.ndarray,     # (dim, dim)
    freqs_cos: np.ndarray,           # (dim//2,)         ← launch_sim: freqs_cos.ravel()
    freqs_sin: np.ndarray,           # (dim//2,)         ← launch_sim: freqs_sin.ravel()
    tensor_XKCache: np.ndarray,      # (dim, seq_len)    K キャッシュ (現トークンの K は含まず)
    tensor_XVCache: np.ndarray,      # (seq_len, dim)    V キャッシュ (現トークンの V は含まず)
    tensor_o_weight: np.ndarray,     # (dim, dim)
    tensor_up_weight: np.ndarray,    # (dim, ffn_dim)
    tensor_gate_weight: np.ndarray,  # (dim, ffn_dim)
    tensor_down_weight: np.ndarray,  # (ffn_dim, dim)
    dim: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    decode.csl の decode_struct() に対応する NumPy 参照実装。

    処理フロー (decode_struct のコメント順):
      1.  RMSNorm(X)                          → X_norm
      2.  QKV 射影 + Y 方向 all-reduce        → Q, K, V
      3.  RoPE 適用                           → Q, K (in-place)
      4.  score = Q @ XKCache * alpha         → (bsz, seq_len)
      5.  Softmax(score, fast_exp)
      6.  output = score @ XVCache            → (bsz, dim)
      7.  h1 = output @ O_weight              → (bsz, dim)  [X 方向 all-reduce]
      8.  Z = X + h1                          [Attention 残差]
      9.  RMSNorm(Z)                          → z_norm
      10. z1 = z_norm @ UP_weight,  z2 = z_norm @ GATE_weight
          z3 = SiLU(z1) * z2                 [SwiGLU; SiLU は UP 側に適用]
          h2 = z3 @ DOWN_weight              [X 方向 all-reduce]
      11. Z = Z + h2                          [FFN 残差]

    返り値: Z (bsz, dim) float16
    """
    f16 = np.float16

    # ---- 1. RMSNorm on X ------------------------------------------------- #
    X_norm = rmsnorm(X, W, dim, eps)                                    # (bsz, dim)   # CONFIRMED

    # ---- 2. QKV 射影 (分散 GEMV + all-reduce と全体として等価) ------------- #
    Q = (X_norm @ tensor_q_weight).astype(f16)                               # (bsz, dim)   # CONFIRMED
    K = (X_norm @ tensor_k_weight).astype(f16)                               # (bsz, dim)
    V = (X_norm @ tensor_v_weight).astype(f16)                               # (bsz, dim)

    # ---- 3. RoPE 適用 ----------------------------------------------------- #
    Q = rope(Q, freqs_cos, freqs_sin)   # CONFIRMED
    K = rope(K, freqs_cos, freqs_sin)
    # K, V は KV キャッシュ更新用 (test では XKCache/XVCache が事前ロード済み)

    # ---- 4. Attention スコア: Q @ K_cache --------------------------------- #
    alpha = f16(1.0 / np.sqrt(float(dim)))
    score = (Q @ tensor_XKCache).astype(f16) * alpha                         # (bsz, seq_len)   # CONFIRMED

    # ---- 5. Softmax (fast_exp 近似, X 方向 all-reduce → 全 seq 集約と等価) #
    score = softmax_fast_exp(score)                                           # (bsz, seq_len)  # CONFIRMED

    # ---- 6. Attention 出力: score @ V_cache ------------------------------- #
    output = (score @ tensor_XVCache).astype(f16)                            # (bsz, dim)   # CONFIRMED

    # ---- 7. O 射影 -------------------------------------------------------- #
    h1 = (output @ tensor_o_weight).astype(f16)                              # (bsz, dim)

    # ---- 8. Attention 残差加算 -------------------------------------------- #
    Z = (X + h1).astype(f16)                                                 # (bsz, dim)

    # ---- 9. RMSNorm on Z -------------------------------------------------- #
    z_norm = rmsnorm(Z, W, dim, eps)                                         # (bsz, dim)

    # ---- 10. FFN (SwiGLU) ------------------------------------------------- #
    z1 = (z_norm @ tensor_up_weight).astype(f16)                             # (bsz, ffn_dim)
    z2 = (z_norm @ tensor_gate_weight).astype(f16)                           # (bsz, ffn_dim)

    # decode.csl z2_silu: ZZ_tile[bsz*dim_p_pe:2*bsz*dim_p_pe] (= z2 の領域) に SiLU 適用
    z3 = (z1 * silu_fast_exp(z2)).astype(f16)                               # (bsz, ffn_dim)
    h2 = (z3 @ tensor_down_weight).astype(f16)                               # (bsz, dim)

    # ---- 11. FFN 残差加算 ------------------------------------------------- #
    Z = (Z + h2).astype(f16)                                                 # (bsz, dim)

    return Z, X_norm, Q, K, V, score, output, h1, z_norm, z3

# ============================================================================ #
# Llama3 Multi-Head Attention Block デコード参照実装
# ============================================================================ #
def decode_mha_block(
    X: np.ndarray,                   # (bsz, dim)
    W: np.ndarray,                   # (dim,)
    tensor_q_weight: np.ndarray,     # (dim, dim)
    tensor_k_weight: np.ndarray,     # (dim, dim)
    tensor_v_weight: np.ndarray,     # (dim, dim)
    freqs_cos: np.ndarray,           # (dim//2,)
    freqs_sin: np.ndarray,           # (dim//2,)
    tensor_XKCache: np.ndarray,      # (dim, seq_len)
    tensor_XVCache: np.ndarray,      # (seq_len, dim)
    tensor_o_weight: np.ndarray,     # (dim, dim)
    tensor_up_weight: np.ndarray,    # (dim, ffn_dim)
    tensor_gate_weight: np.ndarray,  # (dim, ffn_dim)
    tensor_down_weight: np.ndarray,  # (ffn_dim, dim)
    n_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    decode_block の Multi-Head Attention 版。
    RMSNorm / FFN は decode_block と同一。Attention のみ head 分割。
    """
    f16 = np.float16
    bsz = X.shape[0]
    dim = n_heads * head_dim

    # ---- 1. RMSNorm on X ------------------------------------------------- #
    X_norm = rmsnorm(X, W, dim, eps)                                         # (bsz, dim)

    # ---- 2. QKV projection ----------------------------------------------- #
    Q = (X_norm @ tensor_q_weight).astype(f16)                               # (bsz, dim)
    K = (X_norm @ tensor_k_weight).astype(f16)
    V = (X_norm @ tensor_v_weight).astype(f16)

    # ---- 3. RoPE 適用（head ごとに独立して適用） -------------------------- #
    # Q/K を (bsz, n_heads, head_dim) に reshape してから各 head に rope を適用
    Q = Q.reshape(bsz, n_heads, head_dim)
    K = K.reshape(bsz, n_heads, head_dim)
    freqs_cos_h = freqs_cos[:head_dim // 2]                                  # (head_dim//2,)
    freqs_sin_h = freqs_sin[:head_dim // 2]
    for h in range(n_heads):
        Q[:, h, :] = rope(Q[:, h, :], freqs_cos_h, freqs_sin_h)
        K[:, h, :] = rope(K[:, h, :], freqs_cos_h, freqs_sin_h)
    # (bsz, n_heads, head_dim)

    # ---- 4-6. Per-head Attention ------------------------------------------ #
    # XKCache: (dim, seq_len) → (n_heads, head_dim, seq_len)
    XKCache_heads = tensor_XKCache.reshape(n_heads, head_dim, -1)            # (n_heads, head_dim, seq_len)
    # XVCache: (seq_len, dim) → (n_heads, seq_len, head_dim)
    XVCache_heads = tensor_XVCache.reshape(-1, n_heads, head_dim).transpose(1, 0, 2)  # (n_heads, seq_len, head_dim)

    alpha = f16(1.0 / np.sqrt(float(head_dim)))
    output_heads = []
    all_scores = []
    for h in range(n_heads):
        q_h = Q[:, h, :]                                                     # (bsz, head_dim)
        score_h = (q_h @ XKCache_heads[h]).astype(f16) * alpha              # (bsz, seq_len)
        score_h = softmax_fast_exp(score_h)
        out_h = (score_h @ XVCache_heads[h]).astype(f16)                    # (bsz, head_dim)
        output_heads.append(out_h)
        all_scores.append(score_h)

    output = np.concatenate(output_heads, axis=-1).astype(f16)              # (bsz, dim)
    score = np.stack(all_scores, axis=1).astype(f16)                        # (bsz, n_heads, seq_len)

    # ---- 7. O 射影 -------------------------------------------------------- #
    h1 = (output @ tensor_o_weight).astype(f16)                             # (bsz, dim)

    # ---- 8. Attention 残差加算 -------------------------------------------- #
    Z = (X + h1).astype(f16)

    # ---- 9. RMSNorm on Z -------------------------------------------------- #
    z_norm = rmsnorm(Z, W, dim, eps)

    # ---- 10. FFN (SwiGLU) ------------------------------------------------- #
    z1 = (z_norm @ tensor_up_weight).astype(f16)
    z2 = (z_norm @ tensor_gate_weight).astype(f16)
    z3 = (z1 * silu_fast_exp(z2)).astype(f16)
    h2 = (z3 @ tensor_down_weight).astype(f16)

    # ---- 11. FFN 残差加算 ------------------------------------------------- #
    Z = (Z + h2).astype(f16)

    return Z, X_norm, Q, K, V, score, output, h1, z_norm, z3

# ============================================================================ #
# 検証用 main: launch_sim.py と同一の乱数テンソルで動作確認
# ============================================================================ #
def main():
    import json, os, argparse, struct

    class Config:
        P         = 8
        bsz       = 1
        dim       = 64
        n_heads   = 1
        n_kv_heads = 1
        head_dim  = 64
        seq_len   = 64
        ffn_dim   = 64

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed (If using the same tensor as launch_sim.py, ensure they match)")
    args = parser.parse_args()

    config = Config()
    if os.path.exists(args.config):
        with open(args.config) as f:
            config.__dict__.update(json.load(f))

    P         = config.P
    bsz       = config.bsz
    dim       = config.dim
    head_dim  = config.head_dim
    seq_len   = config.seq_len
    ffn_dim   = config.ffn_dim
    dim_p_pe  = dim // P
    _dim_p_pe = (dim_p_pe // 2) * 2   # decode.csl の const _dim_p_pe と同じ

    if args.seed is not None:
        np.random.seed(args.seed)

    # ---------------------------------------------------------------------- #
    # launch_sim.py と同一の変数名・形状でテンソルを生成
    # ---------------------------------------------------------------------- #
    X                  = np.random.rand(1, bsz * dim).astype(np.float16)
    W                  = np.random.rand(1, dim).astype(np.float16)
    tensor_q_weight    = np.random.rand(dim, dim).astype(np.float16)
    tensor_k_weight    = np.random.rand(dim, dim).astype(np.float16)
    tensor_v_weight    = np.random.rand(dim, dim).astype(np.float16)
    freqs_sin          = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
    freqs_cos          = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
    tensor_XKCache     = np.random.rand(dim, seq_len).astype(np.float16)
    tensor_XVCache     = np.random.rand(seq_len, dim).astype(np.float16)
    tensor_o_weight    = np.random.rand(dim, dim).astype(np.float16)
    tensor_up_weight   = np.random.rand(dim, ffn_dim).astype(np.float16)
    tensor_gate_weight = np.random.rand(dim, ffn_dim).astype(np.float16)
    tensor_down_weight = np.random.rand(ffn_dim, dim).astype(np.float16)

    # ---------------------------------------------------------------------- #
    # H2D で渡したテンソルをグローバル形状に変換してから decode_block へ渡す
    # (PE タイリングは decode_block 内部では不要; 全体 GEMV と等価)
    # ---------------------------------------------------------------------- #
    X_np         = X.reshape(bsz, dim)           # (bsz, dim)
    W_np         = W.reshape(dim)                # (dim,)
    freqs_cos_np = freqs_cos.ravel()             # (dim//2,)  PE 分割を展開
    freqs_sin_np = freqs_sin.ravel()             # (dim//2,)

    Z = decode_block(
        X=X_np,
        W=W_np,
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
        head_dim=head_dim,
    )

    print("NumPy Reference Output Z (shape:", Z.shape, "):")
    print(Z)

if __name__ == "__main__":
    main()