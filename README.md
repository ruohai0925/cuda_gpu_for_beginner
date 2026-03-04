# CUDA GPU Programming for Beginners

[中文版本](#中文版本)

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

## Environment

| Item | Value |
|------|-------|
| OS | Ubuntu 20.04 |
| GPU | NVIDIA GTX 2060 |
| CUDA | 11 |

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

---

# 中文版本

# CUDA GPU 入门编程

## 概述

一个动手实践的 CUDA 学习仓库，涵盖从向量加法到 cuBLAS 的 GPU 编程基础概念。本项目最初 fork 自 [CoffeeBeforeArch 的 cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming)（Nick 的 "CUDA Crash Course v3" YouTube 系列教程），之后逐步发展为一个独立的、经过重新组织的仓库，并包含大量新增内容。

## 与原始仓库的关系

### 起源

本仓库最初 fork 自 [CoffeeBeforeArch/cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming)（941+ stars，GPL-3.0 许可证）。原仓库包含 Nick 优秀的 GPGPU 编程 YouTube 教程系列的代码示例。

中间阶段的 fork 曾存在于 [ruohai0925/cuda_programming](https://github.com/ruohai0925/cuda_programming)，之后被重新组织为本独立仓库。

### 基于 Nick 原始仓库的内容

- 核心 CUDA 主题（向量加法、矩阵乘法、归约求和、直方图、卷积）— 所有代码均已修改和增强
- 渐进优化的方法论（基线 → 优化变体）
- GPL-3.0 许可证（见 [LICENSE](LICENSE)）

### 本仓库的新增内容

| 方面 | 原始仓库 (Nick) | 本仓库 |
|------|-----------------|--------|
| **结构** | 平铺编号目录 (01–05) | 相同核心结构 + 额外模块 (06_cuBLAS, misc, timing) |
| **cuBLAS** | 未包含 | `06_cuBLAS/`：使用 `cublasSetVector` 的 SAXPY、使用 cuRAND 的批量矩阵乘法 |
| **性能基准测试** | 无计时框架 | `cuda_timing_Nick/`：矩阵乘法（naive, coalesced, prefetch, tmp_var, unroll）和归约求和的计时工具 |
| **对比研究** | 无 | `cuda_timing_ZDSJTU/`：带计时的示例子集，用于性能对比研究 |
| **其他示例** | 无 | `misc/`：OpenACC 矩阵乘法、clock 计时、设备查询工具 |
| **学习笔记** | 无 | `learning_notes`：`nsys profile` 分析命令和示例输出 |
| **文档** | 简单 README | 详细 README（中英双语）、CLAUDE.md 含构建命令、代码模式和关键概念 |
| **代码注释** | 少量注释 | 增强注释和标注，适合自学 |

## 仓库结构

```
cuda_gpu_for_beginner/
├── 01_vector_addition/      # 基线 → 网格步进/向量化 → 锁页内存 → 统一内存
├── 02_matrix_mul/           # 基线 → 对齐 → restrict → 矩形矩阵 → 非整数倍 → 分块 (1D, 2D)
├── 03_sum_reduction/        # 发散 → bank冲突 → 减少空闲线程 → 无冲突 → 设备函数 → 协作组
├── 04_histogram/            # 全局原子操作 → 共享内存原子操作
├── 05_convolution/          # 1d朴素 → 1d缓存 → 1d常量内存 → 1d分块 → 2d常量内存
├── 06_cuBLAS/               # SAXPY (vector_add_cublas.cu)、批量矩阵乘法 (matrix_mul_cublas.cu)
├── cuda_timing_Nick/        # 性能基准测试框架
├── cuda_timing_ZDSJTU/      # 计时对比研究
├── misc/                    # OpenACC、clock、query_device
└── learning_notes           # nsys 分析笔记
```

## 涵盖的关键概念

- **网格步进循环与向量化内存访问**（`int4` 实现 128 位加载）
- **锁页内存**（DMA 直接访问，消除双重拷贝开销）
- **统一内存**（`cudaMallocManaged`，自动迁移）
- **共享内存分块**（`__shared__` 实现数据复用，1D 和 2D 变体）
- **非方阵与非整数倍矩阵**（向上取整除法，边界检查）
- **Bank 冲突**（顺序 vs. 跨步共享内存访问模式）
- **Warp 级优化**（使用 `warpReduce()` 展开最后一个 warp）
- **协作组**（现代同步 API，`int4` 向量化加载，`atomicAdd`）
- **常量内存**（`__constant__` 用于只读掩码/系数数据）
- **cuBLAS v2 API**（列优先布局，`cublasSetVector`，cuRAND）
- **OpenACC**（基于指令的 GPU 编程）

## 环境

| 项目 | 值 |
|------|------|
| 操作系统 | Ubuntu 20.04 |
| GPU | NVIDIA GTX 2060 |
| CUDA 版本 | 11 |

## 构建

无构建系统（Makefile/CMake）。每个 `.cu` 文件单独编译：

```bash
# 标准编译
nvcc source.cu -o executable

# cuBLAS 项目（需要链接）
nvcc vector_add_cublas.cu -o vector_add -lcublas
nvcc matrix_mul_cublas.cu -o matrix_mul -lcublas -lcurand
```

## 许可证

本仓库同时遵循以下**两个**许可证。仓库中的所有代码（包括对原始示例的修改）必须同时满足这两个许可证的要求：

1. **GPL-3.0** — 继承自上游 [CoffeeBeforeArch/cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming) 仓库。见 [LICENSE](LICENSE)。

2. **PolyForm Strict 1.0.0** — 由本仓库维护者对整个仓库施加的附加限制。见 [LICENSE-POLYFORM](LICENSE-POLYFORM)。

**简而言之**：本仓库所有代码为源码可见，仅限**个人和非商业用途**。你可以学习、实验和研究这些代码，但未经维护者明确书面许可，**不允许商业使用**。使用本仓库的任何部分时，必须同时遵守两个许可证。

## 致谢

- [CoffeeBeforeArch (Nick)](https://github.com/CoffeeBeforeArch)，感谢其原始 "CUDA Crash Course (v3)" YouTube 系列教程和代码库
- CUDA 社区提供的宝贵学习资源
