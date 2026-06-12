# WaferLLM: Large Language Model Inference at Wafer Scale

[![arXiv](https://img.shields.io/badge/arXiv-2502.04563-b31b1b.svg)](https://arxiv.org/abs/2502.04563)

**Authors:** Congjie He, Yeqi Huang, and Pei Mu, *University of Edinburgh;* Ziming Miao, Jilong Xue, Lingxiao Ma, and Fan Yang, *Microsoft Research;* Luo Mai, *University of Edinburgh*

**OSDI 2025**

Emerging AI accelerators increasingly adopt wafer-scale manufacturing technologies, integrating hundreds of thousands of AI cores in a mesh architecture with large distributed on-chip memory (tens of GB in total) and ultra-high on-chip memory bandwidth (tens of PB/s). However, current LLM inference systems, optimized for shared memory architectures like GPUs, fail to exploit these accelerators fully.

We introduce WaferLLM, the first wafer-scale LLM inference system. WaferLLM is guided by a novel PLMR model (pronounced as "Plummer") that captures the unique hardware characteristics of wafer-scale architectures. Leveraging this model, WaferLLM pioneers wafer-scale LLM parallelism, optimizing the utilization of hundreds of thousands of on-chip cores. It also introduces MeshGEMM and MeshGEMV, the first GEMM and GEMV implementations designed to scale effectively on wafer-scale accelerators.

Evaluations show that WaferLLM achieves up to 200× higher accelerator utilization than state-of-the-art methods. Leveraging a wafer-scale accelerator (Cerebras WSE2), WaferLLM delivers GEMV operations 606× faster and 16× more energy-efficient than on an NVIDIA A100 GPU. For full LLM inference, WaferLLM achieves 10-20× speedups over A100 GPU clusters running SGLang and vLLM. These advantages are expected to grow as wafer-scale AI models, software, and hardware continue to mature.

## Prerequisites

You will need Cerebras SDK to reproduce our results.

- **Download link:** https://www.cerebras.ai/developers/sdk-request
- **SDK Version:** This code is developed and tested on **Cerebras SDK 2.10.0** (WSE-3 only).

### System Requirements

- Access to a Cerebras WSE-3 system, or the Cerebras SDK 2.10.0 simulator for local development
- Python 3.8 or higher
- Sufficient memory for running simulations (32GB+ recommended for simulator)

> **Note:** This repository targets **WSE-3 with Cerebras SDK 2.10.0 only**. The original
> OSDI'25 results were measured on WSE-2 / SDK 1.x; that evaluation is described below and is
> preserved for reference, but the code here has been migrated to WSE-3 / 2.10.0.

## How to Use This Library

### Project Structure

Each unit test folder follows a consistent code structure:

```
.
├── clean.sh                       # Remove all build/run artifacts (repo-wide)
├── <module_name>/                 # MeshGEMM, MeshGEMV, SUMMA, Decode, Prefill
│   ├── compile.py                 # Unified compile entry: --mode sim|device
│   ├── launch_sim.py              # Host driver on the simulator (cs_python)
│   ├── launch_device.py           # Host driver on WSE-3 (appliance python)
│   ├── run_sim.sh                 # Compile + run on the simulator
│   ├── run_device.sh              # Compile + run on a real WSE-3
│   ├── benchmark.sh               # Standard device run (8B-4K-256 scale)
│   ├── model_config/              # JSON configs (Decode / Prefill only)
│   └── src/
│       ├── comm_lib/              # Communication library (SUMMA uses <collectives_2d>)
│       │   ├── comm_layout.csl    # Communication topology / routing
│       │   └── comm_pe.csl        # PE communication primitives
│       ├── layout.csl             # PE grid + color/route configuration
│       └── <module>.csl           # Kernel implementation
```

Build/run artifacts (`out_<cfg>/`, `compile_out/`, `log/`) are created on demand by the
scripts and are git-ignored; run `./clean.sh` from the repo root to remove them all.

### Quick Start

We provide two main execution scripts for each module:

1. **`run_sim.sh`** - Compile and run on the Cerebras SDK 2.10.0 simulator (use small configs)
2. **`run_device.sh`** - Compile and run on actual Cerebras WSE-3 hardware

### Module-Specific Parameters

Each module has specific parameters that can be configured. Please refer to the individual README files in each module directory:

- `MeshGEMV/README.md` - Matrix-vector multiplication parameters
- `MeshGEMM/README.md` - Matrix-matrix multiplication parameters
- `Prefill/README.md` - Prefill phase configuration
- `Decode/README.md` - Decode phase configuration

### Communication Library

The `comm_lib` directory in each module contains our custom communication library optimized for wafer-scale architectures:

- `comm_layout.csl` - Defines the communication topology and routing
- `comm_pe.csl` - Implements the processing element communication primitives

This library enables efficient data movement across the massive mesh of cores on the WSE-3.

## Benchmarking

To reproduce the performance results reported in our paper:

1. Ensure you have access to a Cerebras WSE-3 system
2. Run the `benchmark.sh` script in each module directory (standard 8B-4K-256 device config)
3. Compare results with the baseline GPU implementations

## Citation

If you use WaferLLM in your research, please cite:

```bibtex
@inproceedings{he2025waferlm,
  title={WaferLLM: Large Language Model Inference at Wafer Scale},
  author={He, Congjie and Huang, Yeqi and Mu, Pei and Miao, Ziming and Xue, Jilong and Ma, Lingxiao and Yang, Fan and Mai, Luo},
  booktitle={19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  year={2025},
  organization={USENIX Association}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

## Contact

For questions and support, please open an issue on GitHub or contact the authors directly.

## Acknowledgments

This work is made possible through funding, hardware, and technical support from the University of Edinburgh, the Edinburgh International Data Facility (EIDF), the Edinburgh Parallel Computing Centre (EPCC), and the Cerebras teams.
