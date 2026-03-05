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

## Video Tutorials

Every topic in this repository comes with a **video walkthrough** explaining the code, CUDA concepts, and optimization techniques step by step. Watch the full playlist on YouTube: [**CUDA GPU for Beginners**](https://www.youtube.com/playlist?list=PLxcVy4Q7Iwv3S1cfFRIhRDFGtbkxJfBCm)

| # | Video | Link |
|---|-------|------|
| 1 | vector add | [Watch](https://www.youtube.com/watch?v=Z3kQM9FDnS8) |
| 2 | vectorAdd um baseline | [Watch](https://www.youtube.com/watch?v=m8aVbFzY6O4) |
| 2.5 | vectorAdd grid stride and vectorized memory access | [Watch](https://www.youtube.com/watch?v=Fxby17eycuU) |
| 3 | vectorAdd um prefetch | [Watch](https://www.youtube.com/watch?v=zEVu3Ut9OUg) |
| 4 | vectorAdd pinned | [Watch](https://www.youtube.com/watch?v=IJVyGTvR-K0) |
| 5 | mmul baseline | [Watch](https://www.youtube.com/watch?v=7pgxPTsqX4M) |
| 6 | mmul tile | [Watch](https://www.youtube.com/watch?v=Vdm5gYq_ne4) |
| 7 | gpu architecture | [Watch](https://www.youtube.com/watch?v=Jq-9ek3buyc) |
| 8 | mmul alignment | [Watch](https://www.youtube.com/watch?v=LsaO7Q_Vx4Y) |
| 9 | cublas vectorAdd | [Watch](https://www.youtube.com/watch?v=E34QkpOQUJI) |
| 10 | cublas matrix mul | [Watch](https://www.youtube.com/watch?v=FQ-Cg_YvqPA) |
| 11 | sum reduction diverged | [Watch](https://www.youtube.com/watch?v=VgSSfQchzHo) |
| 12 | sum reduction bank conflicts | [Watch](https://www.youtube.com/watch?v=P4ryYKREKmw) |
| 13 | some questions about bank conflicts | [Watch](https://www.youtube.com/watch?v=xtjdzWk2yz0) |
| 14 | sum reduction no conflicts | [Watch](https://www.youtube.com/watch?v=n4ahYs70UwM) |
| 15 | sum reduction reduce idle | [Watch](https://www.youtube.com/watch?v=nanG70BemNE) |
| 16 | sum reduction device function | [Watch](https://www.youtube.com/watch?v=gJIBeWouLak) |
| 17 | sum reduction cooperative groups | [Watch](https://www.youtube.com/watch?v=XGt-2oWTe9k) |
| 18 | visual studio build cuda project | [Watch](https://www.youtube.com/watch?v=7a7qfDYlGU8) |
| 19 | vectorAdd baseline profiling | [Watch](https://www.youtube.com/watch?v=1LnZ6J0zw2Q) |
| 20 | Nsight Systems vs Nsight Compute | [Watch](https://www.youtube.com/watch?v=ynDAEmC5CJ8) |
| 21 | gpu profiling scripts for mmul and sumReduction | [Watch](https://www.youtube.com/watch?v=hLOyh12YmsM) |
| 21.5 | gpu profiling results for mmul and sumReduction | [Watch](https://www.youtube.com/watch?v=NaoyP2egoSw) |
| 22 | 1d convolution naive | [Watch](https://www.youtube.com/watch?v=RQpw8L7jYfY) |
| 23 | GPU concepts recap | [Watch](https://www.youtube.com/watch?v=Bmti4lUJhB0) |
| 24 | 1d convolution constant memory | [Watch](https://www.youtube.com/watch?v=9f5oMRZ8hXg) |
| 25 | 1d convolution shared memory | [Watch](https://www.youtube.com/watch?v=cqE-FzfNHU4) |
| 26 | 1d convolution cache simplification | [Watch](https://www.youtube.com/watch?v=SXh_hmMzsto) |
| 27 | 2d convolution | [Watch](https://www.youtube.com/watch?v=xfVvLvz_kA0) |
| 28 | short summary thinking spatially | [Watch](https://www.youtube.com/watch?v=X9I9EQorroo) |
| 29 | histogram global atomic | [Watch](https://www.youtube.com/watch?v=qVz8x-8Y14s) |
| 30 | histogram shmem atomic | [Watch](https://www.youtube.com/watch?v=6VoHjZAweCw) |
| 31 | matrix multiplication demo | [Watch](https://www.youtube.com/watch?v=HvIgHc-0_Kw) |
| 32 | OpenACC | [Watch](https://www.youtube.com/watch?v=w5Yrz15oE3Q) |
| 33 | gpu device properties | [Watch](https://www.youtube.com/watch?v=wPNZW_GIlc0) |
| 34 | clock function | [Watch](https://www.youtube.com/watch?v=S0et57vOJt0) |

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
