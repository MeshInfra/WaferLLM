# WSE-3 SDK Run Results

记录时间：2026-05-20 22:31 PDT

## 目标

本轮目标是在 `Decode/WSE-3` 下准备并运行两个基础 microbenchmark：

1. H2D load benchmark：测 host/off-chip 到 wafer on-chip SRAM 的 copy latency/bandwidth。
2. On-wafer hop benchmark：测 decode attention collective 的 effective ns per hop。

当前结论：H2D-only P=32 artifact 已经编译成功；full decode P=32 artifact 在当前本地 `cslc` 下仍然 link 失败，原因变为 PE memory 不够。真实运行还需要 CS system 的 `--cmaddr <IP:port>`。

## 当前代码改动

### 1. `launch_wse3.py` SDK runtime 兼容

文件：`Decode/WSE-3/launch_wse3.py`

改动内容：

- 原代码依赖 `cerebras.sdk.client.SdkRuntime` 和 `cerebras.appliance.pb.sdk.sdk_common_pb2`。
- 当前 `sdk-cbcore-202505230211-4-9382352f.sif` 里没有 `cerebras.appliance`，也没有 `cerebras.sdk.client`。
- 已加入 fallback 到 SDK 1.4 本地接口：

```python
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
```

- 对 `sdkruntimepybind` 增加了 `load() / run() / stop()` context wrapper。
- 新增参数：

```bash
--cmaddr <IP:port>
```

- 如果使用 `sdkruntimepybind` 跑真实 WSE 但没有传 `--cmaddr`，现在会直接报：

```text
sdkruntimepybind real WSE runs require --cmaddr <IP:port>.
```

### 2. `simd_max` 替换成 `simd_64`

文件：`Decode/WSE-3/src/comm_lib/comm_pe.csl`

当前本地 `cslc` 不认识：

```csl
.simd_mode = .{ .simd_max = true },
```

已经把 50 处全部替换为：

```csl
.simd_mode = .{ .simd_64 = true },
```

验证：

```bash
rg -n "simd_max" src/comm_lib/comm_pe.csl
```

结果为空，说明没有残留。

## 已生成的 artifact

### H2D-only P=32 real artifact：成功

目录：

```text
compile_out/out_32_8_1_4_32_32_4_1024_4_h2d_only_real
```

当前 `compile_out/artifact_32_8.json` 指向：

```json
{
  "artifact_id": "/nfs/hpc/share/chuxu/stampede3/cerebras/WaferLLM/Decode/WSE-3/compile_out/out_32_8_1_4_32_32_4_1024_4_h2d_only_real"
}
```

这个 artifact 用 `src_h2d_only` 编译，适合实验 1 的 H2D copy-only benchmark。

### Full decode P=8 smoke artifact：成功

目录：

```text
compile_out/out_8_2_1_8_8_8_8_8_8_simd64_smoke
```

意义：

```text
simd_max -> simd_64 替换后，完整 decode 源码可以在小 P=8 smoke 配置下编译通过。
```

### Full decode P=32 real artifact：失败

尝试输出目录：

```text
compile_out/out_32_8_1_4_32_32_4_1024_4_simd64_real
```

失败原因：

```text
ld.lld: error: ran out of PE memory for data (section .bss)
ld.lld: error: ran out of PE memory for task table
ld.lld: error: ran out of PE memory for data (section .data.hi)
```

解释：

```text
simd_64 替换解决了 cslc 语义错误，但 P=32 full decode 在当前 compiler/linker 下仍然超过每个 PE 的可用内存/任务表容量。
```

## 已运行过的命令

### 0. 直接运行原始脚本：失败

```bash
cd /nfs/hpc/share/chuxu/stampede3/cerebras/WaferLLM/Decode/WSE-3

./run_wse3.sh model_config/offchip_h2d_p32_fit.json false \
  --h2d-bench \
  --h2d-preset payload-sweep \
  --h2d-symbols XKCache,XVCache \
  --h2d-samples 5
```

失败原因：

```text
ModuleNotFoundError: No module named 'cerebras'
```

### 1. 用 SIF 运行 `run_wse3.sh`：失败

```bash
APPTAINER_TMPDIR=/tmp timeout 1800s \
apptainer exec --userns --unsquash \
  --bind /nfs:/nfs \
  --pwd /nfs/hpc/share/chuxu/stampede3/cerebras/WaferLLM/Decode/WSE-3 \
  /nfs/hpc/share/chuxu/stampede3/cerebras/cs_sdk/sdk-cbcore-202505230211-4-9382352f.sif \
  ./run_wse3.sh model_config/offchip_h2d_p32_fit.json false \
    --h2d-bench \
    --h2d-preset payload-sweep \
    --h2d-symbols XKCache,XVCache \
    --h2d-samples 5
```

失败原因：

```text
run_wse3.sh: line 21: jq: command not found
```

### 2. 用 SIF 直接跑 `compile.py`：失败

```bash
APPTAINER_TMPDIR=/tmp timeout 900s \
apptainer exec --userns --unsquash \
  --bind /nfs:/nfs \
  --pwd /nfs/hpc/share/chuxu/stampede3/cerebras/WaferLLM/Decode/WSE-3 \
  /nfs/hpc/share/chuxu/stampede3/cerebras/cs_sdk/sdk-cbcore-202505230211-4-9382352f.sif \
  python compile.py 32 1 4 32 32 4 1024 4 4 2 18 false
```

失败原因：

```text
ModuleNotFoundError: No module named 'cerebras.sdk.client'
```

### 3. 编译 H2D-only P=32 real artifact：成功

```bash
APPTAINER_TMPDIR=/tmp timeout 900s \
apptainer exec --userns --unsquash \
  --bind /nfs:/nfs \
  --pwd /nfs/hpc/share/chuxu/stampede3/cerebras/WaferLLM/Decode/WSE-3 \
  /nfs/hpc/share/chuxu/stampede3/cerebras/cs_sdk/sdk-cbcore-202505230211-4-9382352f.sif \
  cslc --arch=wse3 \
    --fabric-dims=762,1172 \
    --fabric-offsets=4,1 \
    -o compile_out/out_32_8_1_4_32_32_4_1024_4_h2d_only_real \
    --memcpy \
    --channels=4 \
    --params=P:32,bsz:1,dim_p_pe:4,pes_p_head:32,pes_p_kv_head:32,head_dim_p_pe:4,seq_len_p_pe:1024,ffn_dim_p_pe:4,pe_num_p_group:4,root_1st_phase:2,root_2nd_phase:18 \
    src_h2d_only/layout.csl
```

### 4. 用 H2D-only artifact 尝试启动 H2D benchmark：失败

```bash
APPTAINER_TMPDIR=/tmp timeout 1800s \
/nfs/hpc/share/chuxu/stampede3/cerebras/cs_sdk/cs_python \
  launch_wse3.py \
    --config model_config/offchip_h2d_p32_fit.json \
    --h2d-bench \
    --h2d-preset payload-sweep \
    --h2d-symbols XKCache,XVCache \
    --h2d-samples 5
```

失败原因：

```text
没有传 --cmaddr，sdkruntimepybind 尝试本地模拟 762x1172 fabric，导致 std::bad_alloc。
```

当前代码已加入早失败检查，之后不会再走到这个 `std::bad_alloc`。

### 5. 替换 `simd_max` 到 `simd_64`

```bash
perl -0pi -e 's/\.simd_max = true/\.simd_64 = true/g' src/comm_lib/comm_pe.csl
```

验证：

```bash
rg -n "simd_max" src/comm_lib/comm_pe.csl
```

结果为空。

### 6. 编译 full decode P=32 real artifact：失败在 PE memory

```bash
APPTAINER_TMPDIR=/tmp timeout 900s \
apptainer exec --userns --unsquash \
  --bind /nfs:/nfs \
  --pwd /nfs/hpc/share/chuxu/stampede3/cerebras/WaferLLM/Decode/WSE-3 \
  /nfs/hpc/share/chuxu/stampede3/cerebras/cs_sdk/sdk-cbcore-202505230211-4-9382352f.sif \
  cslc --arch=wse3 \
    --fabric-dims=762,1172 \
    --fabric-offsets=4,1 \
    -o compile_out/out_32_8_1_4_32_32_4_1024_4_simd64_real \
    --memcpy \
    --channels=4 \
    --params=P:32,bsz:1,dim_p_pe:4,pes_p_head:32,pes_p_kv_head:32,head_dim_p_pe:4,seq_len_p_pe:1024,ffn_dim_p_pe:4,pe_num_p_group:4,root_1st_phase:2,root_2nd_phase:18 \
    src/layout.csl
```

失败摘要：

```text
ld.lld: error: ran out of PE memory for data (section .bss)
ld.lld: error: ran out of PE memory for task table
ld.lld: error: ran out of PE memory for data (section .data.hi)
```

### 7. 编译 full decode P=8 smoke artifact：成功

```bash
APPTAINER_TMPDIR=/tmp timeout 300s \
apptainer exec --userns --unsquash \
  --bind /nfs:/nfs \
  --pwd /nfs/hpc/share/chuxu/stampede3/cerebras/WaferLLM/Decode/WSE-3 \
  /nfs/hpc/share/chuxu/stampede3/cerebras/cs_sdk/sdk-cbcore-202505230211-4-9382352f.sif \
  cslc --arch=wse3 \
    --fabric-dims=15,10 \
    --fabric-offsets=4,1 \
    -o compile_out/out_8_2_1_8_8_8_8_8_8_simd64_smoke \
    --memcpy \
    --channels=1 \
    --params=P:8,bsz:1,dim_p_pe:8,pes_p_head:8,pes_p_kv_head:8,head_dim_p_pe:8,seq_len_p_pe:8,ffn_dim_p_pe:8,pe_num_p_group:4,root_1st_phase:2,root_2nd_phase:6 \
    src/layout.csl
```

## 下一步命令

拿到真实 CS system 地址后，实验 1 可以直接跑：

```bash
cd /nfs/hpc/share/chuxu/stampede3/cerebras/WaferLLM/Decode/WSE-3

APPTAINER_TMPDIR=/tmp timeout 1800s \
/nfs/hpc/share/chuxu/stampede3/cerebras/cs_sdk/cs_python \
  launch_wse3.py \
    --cmaddr <IP:port> \
    --config model_config/offchip_h2d_p32_fit.json \
    --h2d-bench \
    --h2d-preset payload-sweep \
    --h2d-symbols XKCache,XVCache \
    --h2d-samples 5
```

预期输出：

```text
h2d_bench_runs/h2d_YYYYMMDD_HHMMSS/h2d_bench_results.csv
h2d_bench_runs/h2d_YYYYMMDD_HHMMSS/h2d_bench_results.json
```

实验 2 还需要先解决 P=32 full decode 的 PE memory link failure，或者改用能编过完整 P=32 artifact 的 appliance `SdkCompiler` 环境。

