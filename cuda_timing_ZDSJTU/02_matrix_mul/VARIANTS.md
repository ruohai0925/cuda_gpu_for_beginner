# Matrix Multiplication Variants

## Overview

This directory contains 4 matrix multiplication implementations (6 kernels total)
demonstrating progressive GPU memory access optimization. All operate on square NxN
integer matrices using 32x32 thread blocks (1024 threads per block).

The optimization progression: **Baseline → Alignment Analysis → Shared Memory Tiling**

---

## Variant 1: Baseline (`baseline/mmul.cu`)

**Kernel:** `matrixMul(const int *a, const int *b, int *c, int N)`

**Optimization Level:** None (naive global memory only)

Each thread computes one element of C via a dot product of a row of A and a column of B.

- **Matrix A access pattern:** Broadcast. All threads in a warp share the same `row`,
  so `a[row * N + k]` reads the same address across the warp. Efficient for a single
  warp, but different warps read addresses separated by stride N, causing L2 cache pressure.
- **Matrix B access pattern:** Coalesced. Threads in a warp have consecutive `col` values,
  so `b[k * N + col]` accesses adjacent addresses. Very efficient.
- **Performance bottleneck:** Every element is read from global memory with no reuse.
  Each output element requires 2N global memory reads (N from A, N from B).

---

## Variant 2: Alignment Analysis (`alignment/mmul.cu`)

Three kernels in one file to compare how memory access alignment affects performance.

### Kernel 0: `matrixMulNaive`
Functionally identical to the baseline, with added boundary checks (`if (row < n && col < n)`).
Serves as the control case for alignment comparisons.

### Kernel 1: `matrixMulTransposedA`
Pre-transposes matrix A on the host before uploading to GPU.

- Reads `a_t[k * n + row]` instead of `a[row * n + k]`.
- **Improvement:** While each warp still broadcasts, the entire block (32 warps) now
  reads a contiguous chunk `A_T[k][0...31]`, significantly improving L2 cache spatial
  locality across warps.

### Kernel 2: `matrixMulMisalignedB`
Pre-transposes matrix B to deliberately create the worst-case access pattern.

- Reads `b_t[col * n + k]` instead of `b[k * n + col]`.
- **Disaster:** Within a warp, `col` varies (0, 1, 2...), creating addresses with
  stride N between them. The memory controller must issue 32 separate transactions
  per warp instead of 1. Expect ~5x or worse slowdown.

**Key insight:** Coalesced vs. strided memory access is the single most important
optimization for GPU memory bandwidth.

---

## Variant 3: Tiled with 1D Shared Memory (`tiled/mmul.cu`)

**Kernel:** `matrixMul(const int *a, const int *b, int *c, int N)`

**Optimization Level:** Shared memory tiling (major speedup)

Decomposes the dot product into 32x32 tiles that are loaded into shared memory
(`__shared__`), which is ~100x faster than global memory.

- **Tile loading:** All 1024 threads collaboratively load one tile of A and one tile of B
  into shared memory using coalesced access patterns.
- **Compute phase:** Dot products are computed entirely from shared memory.
- **Synchronization:** Two `__syncthreads()` barriers per tile iteration (after load,
  after compute) to prevent data races.
- **Data reuse:** Each global memory element is loaded once per tile but used 32 times
  (by all threads in the corresponding row/column of the block).
- **Shared memory layout:** Uses 1D flattened arrays `__shared__ int s_a[1024]`
  with manual index calculation `s_a[threadIdx.y * 32 + k]`.

---

## Variant 4: Tiled with 2D Shared Memory (`tiled/mmul_2D.cu`)

**Kernel:** `matrixMul(const int *a, const int *b, int *c, int N)`

**Optimization Level:** Same as Variant 3, improved code readability

Identical algorithm to Variant 3 but uses 2D shared memory arrays:
```cpp
__shared__ int s_a[32][32];   // vs s_a[1024]
s_a[threadIdx.y][k]           // vs s_a[threadIdx.y * 32 + k]
```

The compiler generates identical machine code. Performance should be equivalent to
Variant 3, but the code is much easier to read and reason about.

---

## Performance Expectations

From slowest to fastest (for large N):

1. **alignment_misaligned_b** - Strided B access destroys bandwidth (~5x slower)
2. **baseline / alignment_naive** - No data reuse, all global memory
3. **alignment_transposed_a** - Slightly better L2 locality for A
4. **tiled_1d / tiled_2d** - Shared memory tiling, major speedup (~10-30x over baseline)
