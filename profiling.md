# Profiling Instructions
**env check**

```bash
cd ./Decode/WSE-3
bash preflight_wse3.sh
```

**MVP smoke test**
```bash
cd ./Decode/WSE-3
bash run_profiling_batch_wse3.sh --preset smoke --out profiling_runs/smoke_real
```

after running the above command, please keep this directory:
- `profiling_runs/smoke_real`

**Controlled P sweep**

This preset is the strict single-variable `P` sweep. It fixes:
- `group_num = 20`
- `bsz = 1`
- `dim = head_dim = seq_len = 4320`
- `ffn_dim = 14400`
- `n_heads = n_kv_heads = 1`
- `layer_num = 32`

Only `P` changes: `240 / 360 / 480 / 720`.

```bash
cd ./Decode/WSE-3
bash run_profiling_batch_wse3.sh --preset llama_p_sweep_controlled --out profiling_runs/llama_p_sweep_controlled_real
```

after running the above command, please keep this directory:
- `profiling_runs/llama_p_sweep_controlled_real`

**Controlled batch-size sweep**

This preset is the strict single-variable batch sweep. It fixes:
- `P = 360`
- `group_num = 20`
- `dim = head_dim = seq_len = 4320`
- `ffn_dim = 14400`
- `n_heads = n_kv_heads = 1`
- `layer_num = 32`

Only `bsz` changes: `1 / 2 / 4 / 8 / 16`.

```bash
cd ./Decode/WSE-3
bash run_profiling_batch_wse3.sh --preset llama_bsz_sweep_controlled --out profiling_runs/llama_bsz_sweep_controlled_real
```

after running the above command, please keep this directory:
- `profiling_runs/llama_bsz_sweep_controlled_real`

**Legacy presets**

- `llama_p_sweep` is not a strict `P`-only sweep because `dim` and `seq_len` also vary across configs.
- `llama_bsz_sweep` is usable, but it uses the older `group_num = 18` setup and larger batch points.

**Results**
For every batch directory, we will have the following files:

- `runs.tsv`：plan to run which configuration
- `completed.tsv`：really ran which configuration
- for every subdirectory, we will have the following files: `metrics.json`、`phase_summary.json`、`category_summary.json`、`cycles_count.npy`
