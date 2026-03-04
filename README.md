# CUDA GPU Programming for Beginners

[中文版本](README_CN.md)

## Overview

A hands-on CUDA learning repository covering fundamental GPU programming concepts — from vector addition to cuBLAS. This project originated as a fork of [CoffeeBeforeArch's cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming) (Nick's "CUDA Crash Course v3" YouTube series) and has since evolved into an independent, restructured repository with significant additions.

## Relationship to the Original Repository

### Origin

This repository was initially forked from [CoffeeBeforeArch/cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming) (941+ stars, GPL-3.0 licensed). The original repo contains code examples from Nick's excellent YouTube tutorial series on GPGPU programming with CUDA.

An intermediate fork existed at [ruohai0925/cuda_programming](https://github.com/ruohai0925/cuda_programming) before being reorganized into this standalone repository.

### What We Build Upon from Nick's Original

- Core CUDA topics (vector addition, matrix multiplication, sum reduction, histogram, convolution) — all code has been modified and enhanced
- The progressive optimization approach (baseline → optimized variants)
- GPL-3.0 license (see [LICENSE](LICENSE))

### What's New in This Repository

| Area | Original (Nick) | This Repo |
|------|-----------------|-----------|
| **Structure** | Flat numbered directories (01–05) | Same core structure + additional modules (06_cuBLAS, misc, timing) |
| **cuBLAS** | Not included | `06_cuBLAS/`: SAXPY with `cublasSetVector`, batched matrix multiplication with cuRAND |
| **Performance Benchmarking** | No timing framework | `cuda_timing_Nick/`: Timing harness for matrix multiplication (naive, coalesced, prefetch, tmp_var, unroll) and sum reduction |
| **Comparative Study** | N/A | `cuda_timing_ZDSJTU/`: Subset of examples with timing instrumentation for performance comparison |
| **Misc Examples** | N/A | `misc/`: OpenACC matrix multiplication, clock-based timing, device query utility |
| **Learning Notes** | N/A | `learning_notes`: Profiling notes with `nsys profile` commands and example output |
| **Documentation** | Minimal README | Detailed README (EN/CN), CLAUDE.md with build commands, code patterns, and key concepts |
| **Code Comments** | Minimal | Enhanced comments and annotations for self-study |

## Repository Structure

```
cuda_gpu_for_beginner/
├── 01_vector_addition/      # baseline → grid_stride/vectorized → pinned → unified memory
├── 02_matrix_mul/           # baseline → alignment → restrict → rectangular → nonMultiple → tiled (1D, 2D)
├── 03_sum_reduction/        # diverged → bank_conflicts → reduce_idle → no_conflicts → device_function → cooperative_groups
├── 04_histogram/            # global_atomic → shmem_atomic
├── 05_convolution/          # 1d_naive → 1d_cache → 1d_constant_memory → 1d_tiled → 2d_constant_memory
├── 06_cuBLAS/               # SAXPY (vector_add_cublas.cu), batched matmul (matrix_mul_cublas.cu)
├── cuda_timing_Nick/        # Performance benchmarking framework
├── cuda_timing_ZDSJTU/      # Timing instrumentation for comparison study
├── misc/                    # OpenACC, clock, query_device
└── learning_notes           # nsys profiling notes
```

## Key Concepts Covered

- **Grid-stride loops & vectorized memory access** (`int4` for 128-bit loads)
- **Pinned memory** (DMA direct access, eliminates double-copy penalty)
- **Unified memory** (`cudaMallocManaged`, automatic migration)
- **Shared memory tiling** (`__shared__` for data reuse, 1D & 2D variants)
- **Non-square & non-multiple matrices** (ceiling division, boundary checks)
- **Bank conflicts** (sequential vs. strided shared memory access)
- **Warp-level optimization** (unrolled last warp with `warpReduce()`)
- **Cooperative groups** (modern sync API, `int4` vectorized loads, `atomicAdd`)
- **Constant memory** (`__constant__` for read-only mask/coefficient data)
- **cuBLAS v2 API** (column-major layout, `cublasSetVector`, cuRAND)
- **OpenACC** (directive-based GPU programming)

## Build

No build system (Makefile/CMake). Each `.cu` file is compiled individually:

```bash
# Standard compilation
nvcc source.cu -o executable

# cuBLAS projects (requires linking)
nvcc vector_add_cublas.cu -o vector_add -lcublas
nvcc matrix_mul_cublas.cu -o matrix_mul -lcublas -lcurand
```

## License

This repository is released under **both** of the following licenses. All code in this repository (including modifications to the original examples) must comply with **both** licenses simultaneously:

1. **GPL-3.0** — inherited from the upstream [CoffeeBeforeArch/cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming) repository. See [LICENSE](LICENSE).

2. **PolyForm Strict 1.0.0** — additional restriction applied to the entire repository by the maintainer. See [LICENSE-POLYFORM](LICENSE-POLYFORM).

**In summary**: All code in this repository is source-available for **personal and non-commercial use only**. You may study, experiment, and learn from this code, but **commercial use is not permitted** without explicit written permission from the maintainer. Both licenses must be respected when using any part of this repository.

## Acknowledgments

- [CoffeeBeforeArch (Nick)](https://github.com/CoffeeBeforeArch) for the original "CUDA Crash Course (v3)" YouTube series and codebase
- The CUDA community for invaluable learning resources
