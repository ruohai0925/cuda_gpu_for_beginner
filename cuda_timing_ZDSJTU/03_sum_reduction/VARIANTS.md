# Sum Reduction Variants

## Overview

This directory contains 6 implementations of parallel sum reduction, demonstrating
progressive optimization from a naive divergent approach to a fully optimized
single-kernel solution. All variants use TB_SIZE=256 threads per block. The input
is an integer array of N elements, reduced to a single sum.

The optimization progression:
**Diverged → Bank Conflicts → No Conflicts → Reduce Idle → Warp Unroll → Cooperative Groups**

All variants except `cooperative_groups` use a multi-stage reduction pattern
(first kernel reduces N elements to partial sums, subsequent kernels reduce further
until a single result remains).

---

## Variant 1: Diverged (`diverged/`)

**Kernel:** `sumReduction`

**Optimization Level:** None (baseline with maximum warp divergence)

- Thread selection uses modulo: `threadIdx.x % (2 * s) == 0`
- **Problem:** Within a warp of 32 threads, active and idle threads alternate.
  At s=1, threads 0, 2, 4... work while 1, 3, 5... idle. The hardware must
  execute both paths, wasting half the cycles.
- Stride doubles each iteration (1, 2, 4, 8...), so fewer threads qualify over time.
- This is the worst-performing variant due to maximum warp divergence.

---

## Variant 2: Bank Conflicts (`bank_conflicts/`)

**Kernel:** `sumReduction` (sequential indexing)

**Optimization Level:** Eliminates warp divergence, introduces bank conflicts

- Thread mapping: `index = 2 * s * threadIdx.x`
- Consecutive threads (0, 1, 2...) are mapped to work items, eliminating divergence.
- **Problem:** The `2 * s` stride creates shared memory bank conflicts. At s=1,
  threads 0 and 16 both access bank 0 (2-way conflict). As s grows, conflicts worsen.
- Improvement over V1 (no divergence), but bank conflicts limit memory throughput.

---

## Variant 3: No Conflicts (`no_conflicts/`)

**Kernel:** `sumReductionNoConflicts`

**Optimization Level:** Zero bank conflicts, zero warp divergence

- "Fold-in" strategy: stride starts at `blockDim.x / 2` and halves each iteration.
- Thread selection: `threadIdx.x < s` — first S threads are active.
- Access pattern: `partial_sum[threadIdx.x]` maps each thread to a unique bank.
  Thread 0 → Bank 0, Thread 1 → Bank 1, ... Thread 31 → Bank 31.
- **Result:** Linear mapping guarantees zero bank conflicts. Maximum shared memory
  bandwidth.

---

## Variant 4: Reduce Idle Threads (`reduce_idle/`)

**Kernel:** `sumReductionIdleThreads`

**Optimization Level:** Halved grid, double-load eliminates idle first iteration

- **Key change:** Each thread loads and adds 2 elements during the load phase:
  `partial_sum[tid] = v[i] + v[i + blockDim.x]`
- Grid size is halved: `N / (TB_SIZE * 2)` blocks instead of `N / TB_SIZE`.
- The first level of reduction is "free" — performed during the load step,
  outside of shared memory.
- **Result:** All 256 threads are productive from the first instruction. No idle
  threads at any point. Requires N >= 512.

---

## Variant 5: Warp Unroll (`device_function/`)

**Kernel:** `sumReductionUnroll` + `warpReduce` device function

**Optimization Level:** Eliminates synchronization overhead in final warp

- Same double-load as Variant 4.
- The reduction loop stops when only 1 warp (32 threads) remains active (`s > 32`).
- The final 64 → 1 reduction is handled by an unrolled `warpReduce()` device function:
  6 sequential additions with no loop, no `if`, no `__syncthreads()`.
- **Critical:** Uses `volatile int*` to force shared memory writes (prevents the
  compiler from caching values in registers, which would break intra-warp communication).
- **Result:** Eliminates 5 `__syncthreads()` barriers from the final iterations,
  reducing synchronization overhead.

---

## Variant 6: Cooperative Groups + Vectorized + Atomic (`cooperative_groups/`)

**Kernel:** `sumReductionV6`

**Optimization Level:** Maximum throughput (grid-stride + vectorized + single-pass)

Three key optimizations combined:

1. **Grid-stride loop:** The kernel can handle any array size with a fixed grid.
   Each thread iterates through its assigned stripe of the array, accumulating
   a local sum.

2. **Vectorized loads (`int4`):** Each iteration loads 128 bits (4 integers) in a
   single instruction, quadrupling memory throughput per instruction.

3. **Atomic aggregation (`atomicAdd`):** Each block's result is atomically added to
   a single scalar output. No second-stage kernel needed — the entire reduction
   completes in a single kernel launch.

- Uses `<cooperative_groups.h>` with `this_thread_block()` for synchronization.
- **Result:** Single kernel launch (vs. 2+ stages for all other variants).
  Most efficient for large arrays.

---

## Performance Expectations

From slowest to fastest (for large N):

1. **diverged** — Warp divergence wastes cycles
2. **bank_conflicts** — Bank conflicts limit shared memory bandwidth
3. **no_conflicts** — Clean access pattern, full bandwidth
4. **reduce_idle** — Fewer blocks, better utilization
5. **device_function** — Eliminates sync overhead in final warp
6. **cooperative_groups** — Vectorized loads + single-pass atomic reduction

## Minimum N Requirements

| Variant | Minimum N | Reason |
|---|---|---|
| diverged, bank_conflicts, no_conflicts | 256 | Need at least 1 block of 256 threads |
| reduce_idle, device_function | 512 | Double-load: each block processes 512 elements |
| cooperative_groups | 256 | Grid-stride loop handles any N >= 4 |
