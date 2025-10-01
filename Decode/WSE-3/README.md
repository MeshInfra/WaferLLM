# Decode

## Overview

This folder contains the implementation of the **Decode** algorithm for transformer model inference on Cerebras WSE-3.

## Platform

- **Cerebras SDK version**: 1.4
- **Cerebras ML Software version**: 2.5
- **Hardware**: WSE-3 only

## Configuration

The Decode implementation uses JSON configuration files to specify model parameters. Example configuration files can be found in `model_config/`.

**Configuration Parameters:**
- `P`: Number of PEs in each dimension (creates P×P PE grid)
- `group_num`: Number of PE groups for parallel execution
- `bsz`: Batch size
- `dim`: Model hidden dimension
- `n_heads`: Number of attention heads
- `n_kv_heads`: Number of key-value heads (for grouped-query attention)
- `head_dim`: Dimension per attention head
- `seq_len`: Maximum sequence length
- `ffn_dim`: Feed-forward network hidden dimension

## Run with Simulator

The simulator allows you to test and debug your Decode implementation before deploying to actual hardware.

```bash
# bash ./run_sim.sh [config_file]
# If no config file is specified, uses config.json or default values
# Example with test configuration
bash ./run_sim.sh model_config/test.json
```

**Note:** The simulator provides cycle-accurate performance estimates and allows debugging without consuming actual hardware resources.

## Run with Cerebras

Deploy and execute your Decode algorithm on the actual WSE-3 hardware.

```bash
# bash ./run_wse3.sh [config_file] [true for simulator | false for real device]
# If no config file is specified, uses config.json or default values
# Example with test configuration
bash ./run_wse3.sh model_config/test.json true # For appliance simulator
bash ./run_wse3.sh model_config/test.json false # Runing on real WSE-3
```

**Prerequisites:**
- Ensure you have access to a WSE-3 system
- Verify your environment is properly configured with Cerebras SDK
- Check that you have the necessary permissions to run on hardware

**Performance Considerations:**
- The WSE-3 provides massive parallelism with thousands of cores
- Optimal performance is achieved when dimensions are divisible by P
- Consider memory constraints when selecting batch size and sequence length
- Decode phase is memory-bandwidth bound, so efficient data layout is crucial
- The `group_num` parameter allows for a trade-off between routing resources and allreduce latency
