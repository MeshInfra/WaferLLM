# MeshGEMM

## Overview

This folder contains the implementation of the **MeshGEMM** algorithm, which computes general matrix multiplication of the form $[M,N]=[M,K]@[K,N]$

## Platform

- **Cerebras SDK version**: 2.10.0
- **Architecture**: WSE-3

## Parameters

- `P`: Number of PEs in each dimension (creates P×P PE grid)
- `M`: Number of rows in the first matrix
- `K`: Shared dimension between matrices
- `N`: Number of columns in the second matrix

Matrix dimensions must be divisible by `P` (each PE owns an `Mt×Kt` / `Kt×Nt` tile).

## Run with Simulator

The simulator allows you to test and debug your MeshGEMM implementation before deploying to actual hardware. Keep grids small — simulator cost scales with `P*P`.

```bash
# ./run_sim.sh P M K N
# Runs [M, K]@[K, N] on a P x P PE grid on the Cerebras simulator.
# Example
./run_sim.sh 4 64 64 64
```

The simulator compiles locally with `cslc` and runs a `np.matmul` numeric check.

## Run with Cerebras WSE-3

Deploy and execute the MeshGEMM algorithm on the actual WSE-3 hardware via the cloud compiler.

```bash
# ./run_device.sh P M K N
# Runs [M, K]@[K, N] on a P x P PE grid on the WSE-3 appliance.
# Example
./run_device.sh 256 4096 4096 4096
```

**Prerequisites:**
- Ensure you have access to a WSE-3 system
- Verify your environment is properly configured with Cerebras SDK 2.10.0
- Check that you have the necessary permissions to run on hardware

**Performance Considerations:**
- The WSE-3 provides massive parallelism with thousands of cores
- Optimal performance is achieved when matrix dimensions are divisible by P
- Consider memory constraints when selecting matrix sizes
