# Prefill

## Overview

This folder contains the implementation of the **Prefill** algorithm for transformer
model inference on the Cerebras WSE-3. The prefill phase processes the initial input
sequence in parallel before transitioning to the decode phase.

## Platform

- **Cerebras SDK version**: 2.10.0
- **Hardware**: WSE-3 only

## Configuration

The Prefill implementation uses JSON configuration files to specify model parameters.
Example configuration files live in `model_config/`.

**Configuration Parameters:**
- `P`: Number of PEs in each dimension (creates a P×P PE grid)
- `dim`: Model hidden dimension
- `n_heads`: Number of attention heads
- `n_kv_heads`: Number of key-value heads (for grouped-query attention)
- `head_dim`: Dimension per attention head
- `seq_len`: Sequence length to prefill
- `ffn_dim`: Feed-forward network hidden dimension

The config argument is the **full path** to the JSON file (e.g. `model_config/test-sim.json`).

## Run with Simulator

The simulator validates the implementation locally without consuming hardware. Cost
scales with P×P, so use small configs (`model_config/test-sim.json` is P=8).

```bash
# ./run_sim.sh <full/path/to/config.json>
bash ./run_sim.sh model_config/test-sim.json
```

`run_sim.sh` compiles with `compile.py --mode sim` into `out_<cfg>/` and then runs
`launch_sim.py`. It reports the per-pass mean and max cycle counts.

## Run with Cerebras

Deploy and execute on real WSE-3 hardware (the cloud compiler builds the artifact and
`launch_device.py` drives the appliance).

```bash
# ./run_device.sh <full/path/to/config.json>
bash ./run_device.sh model_config/llama8B_4k_1_256.json
```

**Prerequisites:**
- Access to a WSE-3 system / CS-3 appliance
- A Cerebras SDK 2.10.0 environment configured for the cloud compiler

**Performance Considerations:**
- The WSE-3 provides massive parallelism across the wafer.
- Optimal performance is achieved when dimensions are divisible by P.
- The harness supports warmup runs and multiple repeat runs for accurate timing.
