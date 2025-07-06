# Prefill

## Overview

This folder contains the implementation of the **Prefill** algorithm for transformer model inference on Cerebras WSE-2. The prefill phase processes the initial input sequence in parallel before transitioning to the decode phase.

## Platform

- **Cerebras SDK version**: 1.2
- **Cerebras ML Software version**: 2.3
- **Hardware**: WSE-2 only

## Configuration

The Prefill implementation uses JSON configuration files to specify model parameters. Example configuration files can be found in `WSE-2/model_config/`.

**Configuration Parameters:**
- `P`: Number of PEs in each dimension (creates P×P PE grid)
- `dim`: Model hidden dimension
- `n_heads`: Number of attention heads
- `n_kv_heads`: Number of key-value heads (for grouped-query attention)
- `head_dim`: Dimension per attention head
- `seq_len`: Sequence length to prefill
- `ffn_dim`: Feed-forward network hidden dimension

## Run with Simulator

The simulator allows you to test and debug your Prefill implementation before deploying to actual hardware.

```bash
cd ./WSE-2
# ./run_sim.sh [config_file]
# If no config file is specified, uses config.json or default values
# Example with test configuration
bash ./run_sim.sh model_config/test.json
```

**Note:** The simulator provides cycle-accurate performance estimates and allows debugging without consuming actual hardware resources.

## Run with Cerebras

Deploy and execute your Prefill algorithm on the actual WSE-2 hardware.

```bash
cd ./WSE-2
# ./run_device.sh [config_file]
# If no config file is specified, uses config.json or default values
# Example with test configuration
bash ./run_device.sh model_config/test.json
```

**Prerequisites:**
- Ensure you have access to a WSE-2 system
- Verify your environment is properly configured with Cerebras SDK
- Check that you have the necessary permissions to run on hardware

**Performance Considerations:**
- The WSE-2 provides massive parallelism with thousands of cores
- Optimal performance is achieved when dimensions are divisible by P
- Prefill phase benefits from parallel processing of the entire input sequence
- Consider memory constraints when selecting sequence length and model dimensions
- The implementation supports both warmup runs and multiple repeat runs for accurate performance measurement