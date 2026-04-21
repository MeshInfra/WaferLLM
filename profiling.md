# Profiling Instructions

**mvp**
```bash
cd ./Decode/WSE-3
bash run_profiling_batch.sh --preset smoke --out profiling_runs/smoke_real
```

after running the above command, please keep this directory for you:
- `profiling_runs/smoke_real`

**P sweep**

```bash
cd ./Decode/WSE-3
bash run_profiling_batch.sh --preset llama_p_sweep --out profiling_runs/llama_p_sweep_real
```

after running the above command, please keep this directory:
- `profiling_runs/llama_p_sweep_real`

**batch size**
```bash
cd ./Decode/WSE-3
bash run_profiling_batch.sh --preset llama_bsz_sweep --out profiling_runs/llama_bsz_sweep_real
```

**Results**
For every batch directory, we will have the following files:

- `runs.tsv`：plan to run which configuration
- `completed.tsv`：really ran which configuration
- for every subdirectory, we will have the following files: `metrics.json`、`phase_summary.json`、`category_summary.json`、`cycles_count.npy`
