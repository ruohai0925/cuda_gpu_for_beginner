# CUDA GPU 入门编程

[English Version](README.md)

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

## 视频教程

本仓库的每个主题都配有**视频讲解**，逐步解析代码、CUDA 概念和优化技巧。在 YouTube 上观看完整播放列表：[**CUDA GPU for Beginners**](https://www.youtube.com/playlist?list=PLxcVy4Q7Iwv3S1cfFRIhRDFGtbkxJfBCm)

| # | 视频 | 链接 |
|---|------|------|
| 1 | vector add | [观看](https://www.youtube.com/watch?v=Z3kQM9FDnS8) |
| 2 | vectorAdd um baseline | [观看](https://www.youtube.com/watch?v=m8aVbFzY6O4) |
| 2.5 | vectorAdd grid stride and vectorized memory access | [观看](https://www.youtube.com/watch?v=Fxby17eycuU) |
| 3 | vectorAdd um prefetch | [观看](https://www.youtube.com/watch?v=zEVu3Ut9OUg) |
| 4 | vectorAdd pinned | [观看](https://www.youtube.com/watch?v=IJVyGTvR-K0) |
| 5 | mmul baseline | [观看](https://www.youtube.com/watch?v=7pgxPTsqX4M) |
| 6 | mmul tile | [观看](https://www.youtube.com/watch?v=Vdm5gYq_ne4) |
| 7 | gpu architecture | [观看](https://www.youtube.com/watch?v=Jq-9ek3buyc) |
| 8 | mmul alignment | [观看](https://www.youtube.com/watch?v=LsaO7Q_Vx4Y) |
| 9 | cublas vectorAdd | [观看](https://www.youtube.com/watch?v=E34QkpOQUJI) |
| 10 | cublas matrix mul | [观看](https://www.youtube.com/watch?v=FQ-Cg_YvqPA) |
| 11 | sum reduction diverged | [观看](https://www.youtube.com/watch?v=VgSSfQchzHo) |
| 12 | sum reduction bank conflicts | [观看](https://www.youtube.com/watch?v=P4ryYKREKmw) |
| 13 | some questions about bank conflicts | [观看](https://www.youtube.com/watch?v=xtjdzWk2yz0) |
| 14 | sum reduction no conflicts | [观看](https://www.youtube.com/watch?v=n4ahYs70UwM) |
| 15 | sum reduction reduce idle | [观看](https://www.youtube.com/watch?v=nanG70BemNE) |
| 16 | sum reduction device function | [观看](https://www.youtube.com/watch?v=gJIBeWouLak) |
| 17 | sum reduction cooperative groups | [观看](https://www.youtube.com/watch?v=XGt-2oWTe9k) |
| 18 | visual studio build cuda project | [观看](https://www.youtube.com/watch?v=7a7qfDYlGU8) |
| 19 | vectorAdd baseline profiling | [观看](https://www.youtube.com/watch?v=1LnZ6J0zw2Q) |
| 20 | Nsight Systems vs Nsight Compute | [观看](https://www.youtube.com/watch?v=ynDAEmC5CJ8) |
| 21 | gpu profiling scripts for mmul and sumReduction | [观看](https://www.youtube.com/watch?v=hLOyh12YmsM) |
| 21.5 | gpu profiling results for mmul and sumReduction | [观看](https://www.youtube.com/watch?v=NaoyP2egoSw) |
| 22 | 1d convolution naive | [观看](https://www.youtube.com/watch?v=RQpw8L7jYfY) |
| 23 | GPU concepts recap | [观看](https://www.youtube.com/watch?v=Bmti4lUJhB0) |
| 24 | 1d convolution constant memory | [观看](https://www.youtube.com/watch?v=9f5oMRZ8hXg) |
| 25 | 1d convolution shared memory | [观看](https://www.youtube.com/watch?v=cqE-FzfNHU4) |
| 26 | 1d convolution cache simplification | [观看](https://www.youtube.com/watch?v=SXh_hmMzsto) |
| 27 | 2d convolution | [观看](https://www.youtube.com/watch?v=xfVvLvz_kA0) |
| 28 | short summary thinking spatially | [观看](https://www.youtube.com/watch?v=X9I9EQorroo) |
| 29 | histogram global atomic | [观看](https://www.youtube.com/watch?v=qVz8x-8Y14s) |
| 30 | histogram shmem atomic | [观看](https://www.youtube.com/watch?v=6VoHjZAweCw) |
| 31 | matrix multiplication demo | [观看](https://www.youtube.com/watch?v=HvIgHc-0_Kw) |
| 32 | OpenACC | [观看](https://www.youtube.com/watch?v=w5Yrz15oE3Q) |
| 33 | gpu device properties | [观看](https://www.youtube.com/watch?v=wPNZW_GIlc0) |
| 34 | clock function | [观看](https://www.youtube.com/watch?v=S0et57vOJt0) |

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
