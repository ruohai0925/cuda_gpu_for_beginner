# Sum Reduction Profiling Guide

## Prerequisites

1. **CUDA Toolkit** (nvcc) installed and available in PATH
2. **Python 3.6+** with `openpyxl` package:
   ```
   pip install openpyxl
   ```
3. **GPU** with CUDA compute capability >= 6.0

## Quick Start

### WSL2 / Linux

```bash
cd cuda_timing_ZDSJTU/03_sum_reduction
python3 profile_reduction.py
```

### Windows PowerShell

```powershell
cd cuda_timing_ZDSJTU\03_sum_reduction
python profile_reduction.py
```

## What the Script Does

1. **Compiles** all 6 reduction variants using `nvcc`
2. **Runs** each variant across 13 array sizes: N = 256, 512, 1024, ..., 1,048,576
3. **Times** each configuration over M=100 kernel launches using `cudaEvent` (excludes memory transfers)
4. **Saves** results to `time.xlsx`

## Variants Profiled

| Column Name | Source Directory | Description |
|---|---|---|
| `diverged` | `diverged/` | Warp-divergent modulo-based reduction |
| `bank_conflicts` | `bank_conflicts/` | Sequential indexing (bank conflicts) |
| `no_conflicts` | `no_conflicts/` | Fold-in strategy (zero bank conflicts) |
| `reduce_idle` | `reduce_idle/` | Double-load, halved grid |
| `device_function` | `device_function/` | Unrolled last warp with device function |
| `cooperative_groups` | `cooperative_groups/` | Grid-stride + int4 vectorized + atomicAdd |

## Array Sizes

N = 256 * 2^i, where i = 0, 1, 2, ..., 12:

| i | N | Description |
|---|---|---|
| 0 | 256 | 1 KB |
| 1 | 512 | 2 KB |
| 2 | 1,024 | 4 KB |
| 3 | 2,048 | 8 KB |
| 4 | 4,096 | 16 KB |
| 5 | 8,192 | 32 KB |
| 6 | 16,384 | 64 KB |
| 7 | 32,768 | 128 KB |
| 8 | 65,536 | 256 KB (original hardcoded size) |
| 9 | 131,072 | 512 KB |
| 10 | 262,144 | 1 MB |
| 11 | 524,288 | 2 MB |
| 12 | 1,048,576 | 4 MB |

**Note:** `reduce_idle` and `device_function` require N >= 512, so N=256 is skipped
for these variants (shown as "N/A" in the output).

## Output Format

`time.xlsx` contains one sheet with columns:

```
N (array size) | diverged | bank_conflicts | no_conflicts | reduce_idle | device_function | cooperative_groups
256            | 0.003200 | 0.003100       | 0.002800     | N/A         | N/A             | 0.002500
512            | 0.003400 | 0.003200       | 0.002900     | 0.002700    | 0.002600        | 0.002400
...
1048576        | ...      | ...            | ...          | ...         | ...             | ...
```

All times are in **milliseconds (ms)**, averaged over M=100 iterations.

## Manual Compilation and Running

You can compile and run individual variants manually:

```bash
# Compile a variant
nvcc diverged/sumReduction.cu -o diverged/sumReduction

# Run: N=65536, M=100 iterations
./diverged/sumReduction 65536 100
# Output: a single float (average kernel time in ms)

# Compile and run all variants
nvcc bank_conflicts/sumReduction.cu -o bank_conflicts/sumReduction
nvcc no_conflicts/sumReduction.cu -o no_conflicts/sumReduction
nvcc reduce_idle/sumReduction.cu -o reduce_idle/sumReduction
nvcc device_function/sumReduction.cu -o device_function/sumReduction
nvcc cooperative_groups/sumReduction.cu -o cooperative_groups/sumReduction

./no_conflicts/sumReduction 65536 100
./reduce_idle/sumReduction 65536 100     # N must be >= 512
./cooperative_groups/sumReduction 65536 100
```

### Windows PowerShell (manual)

```powershell
# Compile
nvcc diverged\sumReduction.cu -o diverged\sumReduction.exe

# Run
.\diverged\sumReduction.exe 65536 100
```

## Customization

Edit the top of `profile_reduction.py` to change parameters:

```python
# Change N range
N_VALUES = [256 * (2 ** i) for i in range(13)]

# Change number of timed iterations
M = 100
```

## Troubleshooting

- **"nvcc not found"**: Ensure CUDA Toolkit bin directory is in your PATH
- **Assertion failure**: The kernel output is verified once during warmup. If this
  fails, ensure N is a valid power-of-2 multiple of 256.
- **"N too small" error**: `reduce_idle` and `device_function` need N >= 512
  (double-load requires at least 2 * TB_SIZE = 512 elements)
- **Large N (1M) runs slowly**: This is expected for the diverged/bank_conflicts
  variants. The script allows up to 5 minutes per run.
