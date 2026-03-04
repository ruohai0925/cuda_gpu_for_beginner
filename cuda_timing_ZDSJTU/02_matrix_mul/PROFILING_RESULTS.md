# Matrix Multiplication Profiling Results & Analysis

Profiled on GTX 2060, CUDA 11, M=100 iterations per data point.
Two independent runs (time1.xlsx, time2.xlsx) were averaged for reliability.

---

## 1. Overview: All Variants Across N

Times in milliseconds (ms). Averaged from 2 runs.

```
  N     | baseline | align_naive | align_transA | misaligned_b | tiled_1d | tiled_2d
  ------+----------+-------------+--------------+--------------+----------+---------
   256  |    0.055 |       0.053 |        0.054 |        0.387 |    0.065 |    0.045
   512  |    0.210 |       0.207 |        0.208 |        1.569 |    0.190 |    0.127
  1024  |    1.731 |       1.067 |        1.074 |        8.611 |    2.114 |    1.481
  2048  |   11.428 |       8.339 |        8.720 |       65.670 |   13.227 |   10.022
  3072  |   31.622 |      28.507 |       29.781 |      224.929 |   36.457 |   26.368
  3968  |   75.706 |      74.313 |       75.125 |      482.269 |   78.939 |   55.933
```

---

## 2. Visualization: Performance Scaling (log scale, selected N)

```
  Time (ms)   [log scale]
  |
  |  N=3968 comparison (excluding misaligned_b which is off-chart at 482 ms)
  |
  |  baseline       ████████████████████████████████████████████████████████████████████ 75.7
  |  align_naive    ██████████████████████████████████████████████████████████████████   74.3
  |  align_transA   █████████████████████████████████████████████████████████████████    75.1
  |  tiled_1d       ██████████████████████████████████████████████████████████████████   78.9
  |  tiled_2d       █████████████████████████████████████████████████               ▎   55.9  ★ FASTEST
  |
  |  N=2048 comparison
  |
  |  baseline       █████████████████████████████████████████████████████▎               11.4
  |  align_naive    ████████████████████████████████████████▎                             8.3  ★ FASTEST
  |  align_transA   ██████████████████████████████████████████▎                           8.7
  |  tiled_1d       ████████████████████████████████████████████████████████████▎        13.2
  |  tiled_2d       ████████████████████████████████████████████████                     10.0
  |
  |  N=512 comparison
  |
  |  baseline       ██████████████████████████                                           0.210
  |  align_naive    █████████████████████████▎                                           0.207
  |  align_transA   █████████████████████████▌                                           0.208
  |  tiled_1d       ███████████████████████▎                                             0.190
  |  tiled_2d       ███████████████▌                                                     0.127  ★ FASTEST
  └─────────────────────────────────────────────────────────────────────────────── Time (ms)
```

---

## 3. The Misaligned Memory Disaster

The most dramatic result: **misaligned_b is 6.5x-8.1x slower** than the naive kernel.

```
  Penalty of strided (uncoalesced) B access vs. naive coalesced B access:

  N     | align_naive (ms) | misaligned_b (ms) | Slowdown
  ------+------------------+--------------------+---------
   256  |            0.053 |              0.387 |    7.2x
   512  |            0.207 |              1.569 |    7.6x
  1024  |            1.067 |              8.611 |    8.1x  ← worst
  2048  |            8.339 |             65.670 |    7.9x
  3072  |           28.507 |            224.929 |    7.9x
  4096  |           80.891 |            524.988 |    6.5x

  Visualization (N=2048):

  align_naive    ████████                                                    8.3 ms
  misaligned_b   ████████████████████████████████████████████████████████▎   65.7 ms
                 ◄───────── 7.9x slower ──────────►

  WHY: In misaligned_b, matrix B is transposed. Access pattern b_t[col * n + k]
  causes threads in a warp to read addresses with stride=N between them.
  The memory controller must issue 32 separate transactions instead of 1.
  This destroys memory bandwidth utilization.
```

**Key takeaway:** Coalesced memory access is the single most important optimization
for GPU performance. A single bad access pattern can cause ~8x slowdown.

---

## 4. The Surprising Mid-Range: alignment_naive Beats Tiled

The most **unexpected** result: between N=1024 and N=2176, `alignment_naive`
(a global-memory-only kernel) is faster than both tiled variants.

```
  Winner at each N range:

  N=256 - 896    →  tiled_2d wins       (shared memory overhead is small vs. data reuse)
  N=1024 - 2176  →  alignment_naive wins (★ UNEXPECTED)
  N=2304 - 3968  →  tiled_2d wins       (data reuse dominates at large N)

  Speedup of align_naive over tiled_2d in the mid-range:

  N     | align_naive | tiled_2d | align_naive is faster by
  ------+-------------+----------+-------------------------
  1024  |       1.067 |    1.481 |  1.39x
  1280  |       2.040 |    3.324 |  1.63x  ← most pronounced
  1536  |       3.554 |    6.029 |  1.70x  ← most pronounced
  1792  |       5.871 |    9.074 |  1.55x
  2048  |       8.339 |   10.022 |  1.20x  (gap closing)
  2176  |      10.760 |   11.231 |  1.04x  (nearly tied)
  2304  |      11.886 |   11.826 |  tiled_2d takes over
```

**Why does this happen?**

1. **L1/L2 cache effectiveness:** For mid-range N (1024-2048), the working set
   can partially fit in L2 cache (GTX 2060 has 3MB L2). The naive kernel benefits
   from implicit caching — the GPU L2 automatically caches frequently accessed
   data. No explicit shared memory management overhead.

2. **Shared memory overhead:** The tiled kernels pay overhead for:
   - Two `__syncthreads()` barriers per tile iteration
   - Shared memory load instructions (extra instructions vs. direct global loads)
   - Register pressure from managing tile indices

3. **Boundary checks in tiled kernels:** Our modified tiled kernels added boundary
   checks (`if (row < N && ...)`) in the tile loading phase. These extra branches
   add overhead per tile iteration. The original code assumed N was exactly 1024
   with no boundary checks.

4. **Cache-friendly access in naive kernel:** The alignment_naive kernel's B access
   pattern (`b[k * N + col]`) is perfectly coalesced. At mid-range N, the L2 cache
   can hold enough tiles of B to provide significant reuse without explicit tiling.

---

## 5. Crossover Point: Where Tiling Wins

```
  align_naive vs tiled_2d over full N range:

  Time
  (ms)     align_naive
   80 ┤     ╱
      │    ╱     tiled_2d
   60 ┤   ╱    ╱
      │  ╱   ╱
   40 ┤ ╱  ╱
      │╱ ╱             ← tiled_2d grows SLOWER (better scaling)
   20 ┤╱╱
      │╱  ← Crossover around N ≈ 2300
   10 ┤╱
      ├───┬───┬───┬───┬───┬───┬───┬───►  N
      256 512 1K  1.5K 2K  2.5K 3K  4K

  At N=3968:  align_naive = 74.3 ms,  tiled_2d = 55.9 ms  → tiled_2d is 1.33x faster
  At N=2048:  align_naive = 8.3 ms,   tiled_2d = 10.0 ms  → align_naive is 1.20x faster

  The crossover happens because tiling reduces global memory traffic from O(N) to
  O(N/32) per output element. As N grows, this 32x reduction in memory bandwidth
  eventually overcomes the shared memory overhead.
```

---

## 6. tiled_1d vs tiled_2d

The 1D-flattened and 2D-array tiled variants should generate identical code,
but **tiled_2d is consistently faster**:

```
  N     | tiled_1d | tiled_2d | tiled_2d speedup
  ------+----------+----------+-----------------
   512  |    0.190 |    0.127 |  1.50x
  1024  |    2.114 |    1.481 |  1.43x
  2048  |   13.227 |   10.022 |  1.32x
  3072  |   36.457 |   26.368 |  1.38x
  3968  |   78.939 |   55.933 |  1.41x

  tiled_2d is 1.3x-1.5x faster than tiled_1d across all N values.
```

**Why?** The 2D array `s_a[threadIdx.y][k]` is laid out in row-major order
by the compiler, which may result in better shared memory bank access patterns
than the manually computed 1D index `s_a[threadIdx.y * blockDim.x + k]`.
The compiler can also optimize 2D array accesses more aggressively.

---

## 7. Scaling Behavior

Matrix multiplication is O(N^3). When N doubles, time should increase ~8x:

```
  N transition    | baseline ratio | Expected (O(N^3))
  ----------------+----------------+------------------
  256  → 512      |    3.84x       |  8.0x  (GPU not saturated at small N)
  512  → 1024     |    8.25x       |  8.0x  ✓ (matches O(N^3))
  1024 → 2048     |    6.60x       |  8.0x  (some cache benefit)
  2048 → 3072     |    2.77x       |  3.4x  (1.5x size increase → ~3.4x)
```

At small N (256→512), the GPU is not fully saturated — there aren't enough thread
blocks to fill all SMs, so doubling N doesn't double the work proportionally. At
large N, the observed scaling closely matches the theoretical O(N^3) prediction.

---

## 8. Baseline vs align_naive: Boundary Check Cost

The `baseline` and `align_naive` kernels are nearly identical, except `align_naive`
has boundary checks. Interestingly, **align_naive is faster** at N >= 1024:

```
  N     | baseline | align_naive | Difference
  ------+----------+-------------+-----------
   256  |    0.055 |       0.053 |  ~same
   512  |    0.210 |       0.207 |  ~same
  1024  |    1.731 |       1.067 |  align_naive 1.62x faster ★
  2048  |   11.428 |       8.339 |  align_naive 1.37x faster
  3968  |   75.706 |      74.313 |  ~same
```

The baseline kernel was compiled from a different source file (baseline/mmul.cu) vs
the alignment kernels (alignment/mmul.cu). The baseline uses `vector<int>` for host
memory while alignment uses `malloc`. The actual kernel code difference is minimal,
but the nvcc compiler may produce different register allocation or instruction scheduling
for the two files. The large gap at N=1024-2048 is likely due to compiler optimization
differences rather than the boundary check cost itself.

---

## 9. Summary of Key Findings

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                        KEY FINDINGS                                     │
  │                                                                         │
  │  1. MEMORY ALIGNMENT IS CRITICAL                                        │
  │     Strided (uncoalesced) access causes 7-8x slowdown.                  │
  │     This is the single biggest performance factor.                      │
  │                                                                         │
  │  2. TILING WINS AT LARGE N (≥ 2304)                                     │
  │     tiled_2d is the fastest variant for N ≥ 2304,                       │
  │     reaching 1.35x speedup over baseline at N=3968.                     │
  │                                                                         │
  │  3. IMPLICIT CACHING COMPETES AT MID-RANGE N                            │
  │     For 1024 ≤ N ≤ 2176, the L2 cache makes global-memory-only         │
  │     kernels competitive with explicit shared memory tiling.             │
  │                                                                         │
  │  4. 2D SHARED MEMORY > 1D FLATTENED                                     │
  │     tiled_2d is consistently 1.3-1.5x faster than tiled_1d.            │
  │     Prefer 2D shared memory arrays for readability AND performance.     │
  │                                                                         │
  │  5. N=4096 CAUSES OOM FOR SOME VARIANTS                                │
  │     baseline, tiled_1d, tiled_2d returned N/A at N=4096.               │
  │     Three NxN int matrices = 192 MB; may hit allocation limits          │
  │     with additional verification buffers.                               │
  └──────────────────────────────────────────────────────────────────────────┘
```
