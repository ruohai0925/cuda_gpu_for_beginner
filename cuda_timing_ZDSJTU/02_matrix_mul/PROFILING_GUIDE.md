# Matrix Multiplication Profiling Guide

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
cd cuda_timing_ZDSJTU/02_matrix_mul
python3 profile_matmul.py
```

### Windows PowerShell

```powershell
cd cuda_timing_ZDSJTU\02_matrix_mul
python profile_matmul.py
```

## What the Script Does

1. **Compiles** all 4 source files (6 kernels) using `nvcc`
2. **Runs** each kernel across 30 matrix sizes: N = 256, 384, 512, ..., 4096
3. **Times** each configuration over M=100 kernel launches using `cudaEvent` (excludes memory transfers)
4. **Saves** results to `time.xlsx`

## Variants Profiled

| Column Name | Source File | Description |
|---|---|---|
| `baseline` | `baseline/mmul.cu` | Naive global memory, no tiling |
| `alignment_naive` | `alignment/mmul.cu` (kernel 0) | Same as baseline with boundary checks |
| `alignment_transposed_a` | `alignment/mmul.cu` (kernel 1) | Pre-transposed A for better L2 locality |
| `alignment_misaligned_b` | `alignment/mmul.cu` (kernel 2) | Pre-transposed B causing strided access |
| `tiled_1d` | `tiled/mmul.cu` | Shared memory tiling (1D flattened arrays) |
| `tiled_2d` | `tiled/mmul_2D.cu` | Shared memory tiling (2D arrays) |

## Output Format

`time.xlsx` contains one sheet with columns:

```
N (matrix size) | baseline | alignment_naive | alignment_transposed_a | alignment_misaligned_b | tiled_1d | tiled_2d
256             | 0.0312   | 0.0298          | 0.0287                 | 0.0456                 | 0.0089   | 0.0091
384             | ...      | ...             | ...                    | ...                    | ...      | ...
...
4096            | ...      | ...             | ...                    | ...                    | ...      | ...
```

All times are in **milliseconds (ms)**, averaged over M=100 iterations.

## Manual Compilation and Running

You can compile and run individual variants manually:

```bash
# Compile baseline
nvcc baseline/mmul.cu -o baseline/mmul

# Run: N=1024, M=100 iterations
./baseline/mmul 1024 100
# Output: a single float (average kernel time in ms)

# Compile alignment
nvcc alignment/mmul.cu -o alignment/mmul_align

# Run alignment with kernel selection:
#   0 = Naive, 1 = TransposedA, 2 = MisalignedB
./alignment/mmul_align 1024 100 0    # Naive kernel
./alignment/mmul_align 1024 100 1    # TransposedA kernel
./alignment/mmul_align 1024 100 2    # MisalignedB kernel

# Compile tiled variants
nvcc tiled/mmul.cu -o tiled/mmul_tiled
nvcc tiled/mmul_2D.cu -o tiled/mmul_tiled_2D

# Run tiled
./tiled/mmul_tiled 1024 100
./tiled/mmul_tiled_2D 1024 100
```

### Windows PowerShell (manual)

```powershell
# Compile
nvcc baseline\mmul.cu -o baseline\mmul.exe

# Run
.\baseline\mmul.exe 1024 100
```

## Customization

Edit the top of `profile_matmul.py` to change parameters:

```python
# Change N range (must be multiples of 32)
N_VALUES = list(range(256, 4096 + 1, 128))

# Change number of timed iterations
M = 100
```

## Troubleshooting

- **"nvcc not found"**: Ensure CUDA Toolkit bin directory is in your PATH
- **Timeout errors**: Large N (e.g., 4096) with M=100 can take several minutes. The script allows up to 10 minutes per run.
- **Verification assertion failure**: The kernel output is verified once (during warmup) before timing begins. If this fails, check that N is a multiple of 32.
- **Out of GPU memory**: N=4096 requires ~192MB of GPU memory (3 matrices of 4096x4096 ints). Ensure your GPU has sufficient VRAM.
