# Decode/WSE-3 Profiling Instructions

目前只考虑 `Decode/WSE-3`，只考虑 `P` 的变化，**不考虑 batch size**。

## Goal

本轮 profiling 的目标是做 **weak scaling**，不是 strong scaling。

重要说明：`llama8B_weak_*` 不是仓库原本就有的一组三个真实
llama8B 模型点。它们是从仓库已有的
`Decode/WSE-3/model_config/llama8B_4k_1_480.json` 派生出来的
controlled weak-scaling microbenchmark。也就是说：

- `P=480` 点等价于仓库已有的 `llama8B_4k_1_480`
- `P=240/360` 点是为了 weak scaling 反推得到的缩小问题规模
- 除了 `controlled` 和 `weak` 这两个实验维度外，比例关系来自已有
  `llama8B_4k_1_480` 配置，而不是任意编造

这里的 weak scaling 定义是：

- `P` 增大时，全局问题也按比例增大
- 每个 PE 的本地工作量尽量保持不变

对当前 `Decode/WSE-3` kernel，我们固定下面这些局部量：

- `dim_p_pe = dim / P = 9`
- `seq_len_p_pe = seq_len / P = 9`
- `ffn_dim_p_pe = ffn_dim / P = 30`
- `pe_num_p_group = P / group_num = 24`

所以本轮 weak-scaling 配置只使用这三组派生配置：

- `model_config/llama8B_weak_1_240.json`
- `model_config/llama8B_weak_1_360.json`
- `model_config/llama8B_weak_1_480.json`

对应的 preset 是：

- `profiling_presets/llama_p_weak_sweep.txt`

如果实验要求“所有 config 都必须是仓库原始已有点”，那么当前仓库里没有
一组严格满足 weak scaling 的 llama8B `P` sweep；只能做原始
`llama8B_4k_1_*` sweep 或 fixed-problem controlled sweep，但那不是 weak
scaling。

## 1. Environment Check

先做环境检查：

```bash
cd ./Decode/WSE-3
bash preflight_wse3.sh
```

## 2. Smoke Run

先跑最小验证，确认环境、编译和 artifact 导出都没问题：

```bash
cd ./Decode/WSE-3
bash run_profiling_batch_wse3.sh --preset smoke --out profiling_runs/smoke_real
```

跑完后请保留目录：

- `profiling_runs/smoke_real`

## 3. Weak-Scaling P Sweep

正式 weak scaling 只跑下面这条命令：

```bash
cd ./Decode/WSE-3
bash run_profiling_batch_wse3.sh --preset llama_p_weak_sweep --out profiling_runs/llama_p_weak_sweep_real
```

跑完后请保留目录：

- `profiling_runs/llama_p_weak_sweep_real`

## 4. Optional: Simulator Check

如果想先在 simulator 上验证同一组 weak-scaling config，也可以运行：

```bash
cd ./Decode/WSE-3
bash run_profiling_batch.sh --preset llama_p_weak_sweep --out profiling_runs/llama_p_weak_sweep_sim
```

## 5. Expected Artifacts

每个 batch 目录里应包含：

- `runs.tsv`
- `completed.tsv`

每个配置子目录里应包含：

- `manifest.json`
- `metrics.json`
- `phase_summary.json`
- `phase_group_summary.json`
- `category_summary.json`
- `cycles_count.npy`
- `phase_cycles.npy`
- `timer_buf_time_hwl.npy`
- `run.log`
