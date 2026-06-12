# Decode

## Overview

This folder contains the implementation of the **Decode** algorithm for transformer model inference on Cerebras WSE-3.

## Platform

- **Cerebras SDK version**: 2.10.0
- **Hardware**: WSE-3 only

## Configuration

The Decode implementation uses JSON configuration files to specify model parameters. Example configuration files can be found in `model_config/` (e.g. `test-sim.json` for a small P=8 sim run, `llama8B_4k_1_256.json` for the device-scale 8B-4K run).

**Configuration Parameters:**
- `P`: Number of PEs in each dimension (creates a P×P PE grid)
- `group_num`: Number of PE groups for the two-phase all-reduce
- `bsz`: Batch size
- `dim`: Model hidden dimension
- `n_heads`: Number of attention heads
- `n_kv_heads`: Number of key-value heads (for grouped-query attention)
- `head_dim`: Dimension per attention head
- `seq_len`: Maximum sequence length
- `ffn_dim`: Feed-forward network hidden dimension
- `layer_num`: Number of transformer layers (used by the device throughput report)

## Run with Simulator

The simulator runs the whole fabric on the host, so cost scales with P×P. Keep `P` small (the provided `test-sim.json` uses P=8). The config argument must be the full path to the JSON file.

```bash
# ./run_sim.sh <full/path/to/config.json>
./run_sim.sh model_config/test-sim.json
```

`run_sim.sh` compiles with `cslc` into `out_<cfg>/` and then launches `launch_sim.py`. It reports the mean per-step cycle count.

## Run with Cerebras WSE-3

Compiles via the cloud SdkCompiler and runs on a real WSE-3 appliance. The config argument must be the full path to the JSON file.

```bash
# ./run_device.sh <full/path/to/config.json>
./run_device.sh model_config/llama8B_4k_1_256.json
```

`run_device.sh` compiles into `compile_out/artifact_<cfg>.json` and then launches `launch_device.py`, which reports the mean per-step cycle count and throughput per request. `benchmark.sh` wraps the standard 8B-4K-256 device run and tees the log into `log/`.

**Prerequisites:**
- Access to a WSE-3 system
- Environment configured with Cerebras SDK 2.10.0
- Permissions to run on the hardware

**Performance Considerations:**
- Optimal performance is achieved when the model dimensions are divisible by `P`
- The `group_num` parameter trades off routing resources against all-reduce latency
