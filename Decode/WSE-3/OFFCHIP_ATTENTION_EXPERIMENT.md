# Decode Attention Off-Chip 实验说明

## 实验目标

本实验用于隔离并量化 decode attention 的 data movement cost，重点关注 long-context 下的 non-fit KV-cache regime：

```text
full K/V cache cannot fit in on-chip SRAM,
so K/V must be streamed from off-chip memory in sequence tiles.
```

核心问题：

```text
当 decode attention 扫描很长 KV cache 时，
latency 主要由 off-chip load、on-wafer communication，还是 local compute 主导？
```

第一版只做：

- single head
- `B = 1`
- attention only
- 暂不考虑 FFN、O projection、multi-head、multi-batch

这不是 full WaferLLM decode latency benchmark。当前 full decode path 假设 KV resident on-chip，而这个实验要测的是 KV 不 fit、必须 off-chip streaming 的情况。

## 当前代码限制

当前 `Decode/WSE-3` 里的 KV cache 是 resident on-chip array：

```text
XKCache_tile: dim_p_pe * seq_len_p_pe fp16
XVCache_tile: seq_len_p_pe * dim_p_pe fp16

dim_p_pe     = dim / P
seq_len_p_pe = seq_len / P
```

完整 KV 大小近似为：

```text
KV = 4 * seq_len * head_dim bytes
```

`4` 来自 K/V 两个 fp16 tensor。

所以，只把现有 config 的 `seq_len` 改大，不会自动得到 off-chip streaming。需要新建 attention-only tiled microbenchmark。

## Non-Fit Config

建议第一个 off-chip case：

```text
P        = 32
B        = 1
dim      = 128
head_dim = 128
seq_len  = 131072   # 128K tokens
seq_tile = 16384
```

容量关系：

```text
KV_full = 4 * 131072 * 128 = 64 MiB
S_mesh  = 32 * 32 * 48 KiB = 48 MiB

KV_full > S_mesh
```

完整 KV 放不下。

单个 tile：

```text
KV_tile = 4 * 16384 * 128 = 8 MiB

dim_p_pe       = 128 / 32 = 4
seq_tile_p_pe  = 16384 / 32 = 512
KV_tile_per_PE = 4 * 4 * 512 = 8 KiB
```

这个 case 的性质：

- full KV 不 fit
- one KV tile fit
- 每个 decode step 有 `131072 / 16384 = 8` 个 tile

## Case Matrix

建议准备多组 case，但每组只改变一类因素，方便解释结果。

### 1. Capacity Cases

用于确认 full KV 是否 fit，以及 tiled path 在不同容量压力下的行为。

| case | P | head_dim | seq_len | seq_tile | KV_full | S_mesh | num_tiles | 目的 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C0 small non-fit | 8 | 128 | 8192 | 2048 | 4 MiB | 3 MiB | 4 | 小规模 off-chip smoke test |
| C1 medium non-fit | 16 | 128 | 32768 | 8192 | 16 MiB | 12 MiB | 4 | 中等规模 off-chip test |
| C2 fit baseline | 32 | 128 | 32768 | 16384 | 16 MiB | 48 MiB | 2 | fit control |
| C3 boundary | 32 | 128 | 98304 | 16384 | 48 MiB | 48 MiB | 6 | 接近 raw SRAM 上限 |
| C4 target non-fit | 32 | 128 | 131072 | 16384 | 64 MiB | 48 MiB | 8 | 主要 off-chip case |
| C5 large non-fit | 32 | 128 | 262144 | 16384 | 128 MiB | 48 MiB | 16 | long-context stress |

其中：

```text
KV_full = 4 * seq_len * head_dim
S_mesh  = P^2 * 48 KiB
```

### 2. Tile-Size Sweep

固定 `P=32, head_dim=128, seq_len=131072`，只改变 `seq_tile`。

| case | seq_tile | KV_tile | KV_tile_per_PE | num_tiles | 目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| T0 | 4096 | 2 MiB | 2 KiB | 32 | 很多小 tile |
| T1 | 8192 | 4 MiB | 4 KiB | 16 | 小 tile |
| T2 | 16384 | 8 MiB | 8 KiB | 8 | 默认 tile |
| T3 | 32768 | 16 MiB | 16 KiB | 4 | 大 tile |

这个 sweep 用来观察：

```text
tile 越小：memcpy call 更多，可能被 fixed overhead 主导
tile 越大：copy 更容易摊销 overhead，但 on-chip buffer 压力更大
```

### 3. Head-Dim Sweep

固定 `P=32, seq_len=131072, seq_tile=16384`，只改变 `head_dim`。

| case | head_dim | KV_full | KV_tile | KV_tile_per_PE | 目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| D0 | 64 | 32 MiB | 4 MiB | 4 KiB | fit/control，较小 per-token work |
| D1 | 128 | 64 MiB | 8 MiB | 8 KiB | 默认 non-fit |
| D2 | 256 | 128 MiB | 16 MiB | 16 KiB | 更高 compute 和 traffic pressure |

这个 sweep 用来区分：

```text
latency 是主要随 bytes 增长，还是随 attention compute 增长？
```

### 4. Multi-Stream Copy Cases

固定单 stream payload，改变同时 load 的 head-like streams 数量。

| case | payload_per_stream | H_active | total_payload | 目的 |
| --- | ---: | ---: | ---: | --- |
| M0 | 8 MiB | 1 | 8 MiB | single-stream baseline |
| M1 | 8 MiB | 2 | 16 MiB | low contention |
| M2 | 8 MiB | 4 | 32 MiB | moderate contention |
| M3 | 8 MiB | 8 | 64 MiB | high contention |
| M4 | 8 MiB | 16 | 128 MiB | channel saturation / stress |

这个 sweep 用来估计 multi-head decode 里的 I/O contention。后续不能直接把 single-head latency 乘以 head 数，需要用这里测到的 effective bandwidth 修正。

## Research Questions

### RQ1: Off-Chip Load Cost

```text
How expensive is it to load one K/V tile from off-chip memory to on-chip SRAM?
```

实验：single-stream copy-only。

记录：

```text
payload_bytes
host_wall_us
device_tsc_us
effective_bandwidth_GBps
```

### RQ2: Multi-Stream Contention

```text
How does off-chip bandwidth scale when multiple head-like streams load K/V tiles?
```

实验：multi-stream copy-only。

Sweep：

```text
H_active = 1, 2, 4, 8, 16
```

目的：估计多 head-like streams 的 I/O contention，避免直接用 single-stream 结果乘 head 数。

### RQ3: Tiled Attention Breakdown

```text
In tiled decode attention, how much time is spent on load, score, softmax,
output accumulation, and on-wafer communication?
```

实验：attention-only tiled compute。

逻辑：

```text
for each K/V tile:
    load K_tile, V_tile
    compute q @ K_tile
    update online softmax
    accumulate output with V_tile
```

记录：

```text
T_load_tile
T_score
T_softmax
T_output
T_onwafer_comm
```

## Host-To-Wafer Load 统计

官方有相关 benchmark：

```text
https://sdk.cerebras.net/csl/code-examples/benchmark-bandwidth-test
```

官方给出的关键信息：

- benchmark 用 device-side TSC 计 H2D / D2H 时间
- host timer 不一定等于真正传输开始时间
- host 和 WSE 之间是 100 Gbps Ethernet
- 小 transaction 会受约 200 us TCP/runtime overhead 影响
- nonblocking commands 可能被 runtime 聚合

本实验先固定 WaferLLM 现有 compile settings，不把 memcpy channels 作为实验变量。

我们需要同时记录两种时间：

```text
host_wall_us:
  Python perf_counter 包住 runner.memcpy_h2d(...)

device_tsc_us:
  wafer 侧在 H2D 前后打 timestamp，再 D2H 读回 timestamp
```

特别要检查绝对 latency 是否非单调：

```text
latency(4 KiB) > latency(64 KiB)
```

如果出现这种情况，说明小 payload 可能触发了额外 runtime / TCP / batching / flush overhead。这个不能只用 bandwidth 曲线解释，必须直接画：

```text
payload_bytes vs latency_us
```

建议 sweep：

```text
payload_bytes = 4 KiB, 8 KiB, 16 KiB, 32 KiB, 64 KiB,
                128 KiB, 256 KiB, 512 KiB,
                1 MiB, 2 MiB, 4 MiB, 8 MiB

loop_count = 1, 4, 16, 64
nonblock = false, true
```

另做一个 same-total-bytes chunk-size 对照：

```text
same_total_bytes = 8 MiB

case A: 2048 x 4 KiB
case B:  128 x 64 KiB
case C:    8 x 1 MiB
case D:    1 x 8 MiB
```

这个实验回答：

```text
同样总 bytes，切成很多小 memcpy 是否比少量大 memcpy 慢？
```

模拟器只能验证流程和 timestamp 逻辑，不能用于判断真实 host-to-wafer latency 或 bandwidth。

## 测量口径

区分两类量：

```text
analytical traffic:
  理论上必须搬多少 bytes

measured latency:
  实际搬这些 bytes 花了多久
```

例如：

```text
KV_tile_bytes = 4 * seq_tile * head_dim
```

这是理论 traffic，可以直接算。latency 则会受到 payload size、memcpy call 数量、runtime overhead、nonblocking/batching、on-wafer routing/reduction 影响。

每个 case 都记录：

```text
traffic_bytes
latency_us
effective_bandwidth_GBps
```

## Data Movement

重点统计：

```text
off-chip -> on-chip:
  K_tile, V_tile load bytes and latency

on-wafer:
  score reduction
  online softmax max/sum reduction
  output partial reduction
```

## 本次最小实验

这次先只统计两张表，不改 attention kernel：

```text
1. H2D load:
   host/off-chip -> XKCache/XVCache

2. on-wafer hop:
   decode attention 里已有 collective profiling / estimated critical-path hops
```

第二项不是 isolated physical-link latency，而是：

```text
effective ns per hop =
  measured collective cycles / estimated critical-path hops / freq_ghz
```

对 attention 只看：

```text
score_reduce / score_broadcast
softmax_reduce / softmax_broadcast
output_reduce / output_broadcast
```

输出文件：

```text
H2D:
  h2d_bench_results.csv
  h2d_bench_results.json

On-wafer:
  comm_subphase_summary.json
  comm_hop_summary.csv
  comm_hop_summary.json
```

## 性能模型

单个 tile：

```text
T_tile = T_load_tile + T_compute_tile + T_onwafer_comm_tile
```

单个 decode step：

```text
T_decode = num_tiles * T_tile
num_tiles = 8
```

扩展到 multi-head 时，不直接把 single-head latency 乘以 head 数；需要结合 multi-stream copy-only 测到的 bandwidth / contention。

## 实现顺序

1. 实现 single-stream copy-only microbenchmark。
2. 实现 multi-stream copy-only microbenchmark。
3. 实现 attention-only tiled compute。

## 当前实现

第一阶段现在集成在 WaferLLM 现有路径中：

```text
compile.py
run_wse3.sh
launch_wse3.py --h2d-bench
src/layout.csl
src/decode.csl
```

实现方式：

- 复用现有 WaferLLM compile artifact
- 复用现有 `XKCache` / `XVCache` 等 symbols
- 复用现有 `timer_buf`
- 在 `decode.csl` 里只新增 `h2d_bench_tic` / `h2d_bench_toc`
- `--h2d-bench` 模式不跑 `decode_host`

注意：这个阶段是 Phase 1 的 H2D copy-only benchmark，用来测 WaferLLM runtime path 下把数据 load 到现有 SRAM-resident symbols 的代价。它还不是 off-chip tiled attention compute，也不能直接代表 load 和 attention compute 交织时的最终 latency。

### 推荐 Config

H2D benchmark 需要选一个 resident buffer 足够大的 WaferLLM config。建议先用 fit baseline：

```text
P=32, dim=128, seq_len=32768
```

此时：

```text
XKCache payload capacity = P^2 * (dim/P) * (seq_len/P) * 2 bytes
                         = 8 MiB
```

刚好可以测目标 `seq_tile=16384, head_dim=128` 对应的 one K tile 或 one V tile：

```text
K_tile = 2 * seq_tile * head_dim = 4 MiB
V_tile = 2 * seq_tile * head_dim = 4 MiB
K_tile + V_tile = 8 MiB
```

### 运行方式

先编译并跑 smoke：

```bash
./run_wse3.sh model_config/offchip_h2d_p32_fit.json false \
  --h2d-bench \
  --h2d-preset smoke \
  --h2d-symbols XKCache \
  --h2d-samples 3
```

payload sweep：

```bash
./run_wse3.sh model_config/offchip_h2d_p32_fit.json false \
  --h2d-bench \
  --h2d-preset payload-sweep \
  --h2d-symbols XKCache \
  --h2d-nonblock both \
  --h2d-loop-counts 1,4,16,64 \
  --h2d-samples 5
```

same-total-bytes chunk-size 对照：

```bash
./run_wse3.sh model_config/offchip_h2d_p32_fit.json false \
  --h2d-bench \
  --h2d-preset chunk-size \
  --h2d-symbols XKCache \
  --h2d-nonblock both \
  --h2d-samples 5
```

multi-stream H2D pressure：

```bash
./run_wse3.sh model_config/offchip_h2d_p32_fit.json false \
  --h2d-bench \
  --h2d-preset multi-stream \
  --h2d-symbols XKCache,XVCache \
  --h2d-streams 1,2,4,8,16 \
  --h2d-samples 5
```

模拟器只用于验证 compile / runtime / timestamp 流程，不用于判断真实 latency：

```bash
./run_wse3.sh model_config/offchip_h2d_p32_fit.json true \
  --h2d-bench \
  --h2d-preset smoke
```

`--h2d-symbols XKCache` 只测 K-like cache symbol。要测更接近 K+V 的 traffic，使用：

```bash
--h2d-symbols XKCache,XVCache
```

每个 H2D round 会拷贝所有 selected symbols。因此 `XKCache,XVCache` 的 measured traffic 是 K+V 两个 symbol 的总量。

on-wafer hop 统计复用正常 decode profiling：

```bash
./run_wse3.sh model_config/offchip_h2d_p32_fit.json false \
  --artifact-dir h2d_bench_runs/comm_p32_fit
```

`comm_hop_summary.csv` 只保留 attention 相关行：

```text
score_reduce
score_broadcast
softmax_reduce
softmax_broadcast
output_reduce
output_broadcast
```

### 输出

结果默认写到：

```text
h2d_bench_runs/h2d_YYYYMMDD_HHMMSS/
```

主要看：

```text
host_wall_us
device_tsc_us
effective_bandwidth_GBps
device_bandwidth_GBps
payload_bytes
loop_count
streams
nonblock
```
