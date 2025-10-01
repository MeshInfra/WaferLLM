# MeshGEMV

## Overview

This folder contains the implementation of the **MeshGEMV** algorithm, which computes general matrix multiplication of the form $[1,N]=[1,M]@[M,N]$

## Platform

- **Cerebras SDK version**: 1.2
- **Cerebras ML Software version**: 2.3
- **Hardware**: WSE-2 only

## Parameters

- `P`: Number of PEs in each dimension (creates P×P PE grid)
- `M`: Length of the input vector (shared dimension)
- `N`: Number of columns in the matrix (output vector length)
- `group_num`: Number of PE groups for parallel execution

## Run with Simulator

The simulator allows you to test and debug your MeshGEMV implementation before deploying to actual hardware.

```bash
# bash ./run_sim.sh P M N group_num
# Then we run [1, M]@[M, N] on P * P PE cores on cerebras simulator
# Example
bash ./run_sim.sh 64 1024 1024 8
```
**Note:** The simulator provides cycle-accurate performance estimates and allows debugging without consuming actual hardware resources.

## Run with Cerebras

Deploy and execute your MeshGEMV algorithm on the actual WSE-2 hardware.

```bash
# bash ./run_wse2.sh P M N group_num
# Then we run [1, M]@[M, N] on P * P PE cores on cerebras chip
# Example
bash ./run_wse2.sh 64 1024 1024 8
```

**Prerequisites:**
- Ensure you have access to a WSE-2 system
- Verify your environment is properly configured with Cerebras SDK
- Check that you have the necessary permissions to run on hardware

**Performance Considerations:**
- The WSE-2 provides massive parallelism with thousands of cores
- Optimal performance is achieved when matrix dimensions are divisible by P
- Consider memory constraints when selecting matrix and vector sizes
- The `group_num` parameter allows for a trade-off between routing resources and allreduce latency