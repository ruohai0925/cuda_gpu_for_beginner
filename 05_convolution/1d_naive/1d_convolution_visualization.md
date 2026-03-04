# 1D Convolution — Concept & Computation Visualization

Based on `05_convolution/1d_naive/convolution.cu`

---

## 1. What Is 1D Convolution?

A 1D convolution slides a small **mask** (also called kernel/filter) across an
**input array**, computing a weighted sum at each position to produce an **output array**.

```
    Input Array (n elements)          Mask (m=7)             Output Array (n elements)
  ┌───┬───┬───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┐
  │ a │ b │ c │ d │ e │ f │...│    │w0 │w1 │w2 │w3 │w4 │w5 │w6 │    │r0 │r1 │r2 │r3 │r4 │r5 │...│
  └───┴───┴───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┘
         ▲                               ▲                                  ▲
     Big (n=2^20)                  Small (m=7)                       Same size as input
```

**Core idea:** For each output position `i`, center the mask on input position `i`,
multiply overlapping elements pairwise, and sum the products.

---

## 2. Mask Radius and Centering

```
  Mask size m = 7
  Radius  r = m / 2 = 3

  Mask indices:    [0]  [1]  [2]  [3]  [4]  [5]  [6]
                    │    │    │    ▲    │    │    │
                    │    │    │  CENTER  │    │    │
                    ◄─── r=3 ───►  ◄─── r=3 ───►
                       LEFT            RIGHT
```

When computing `result[i]`, the mask is centered at `array[i]`:
- Left edge reads from `array[i - 3]`
- Right edge reads from `array[i + 3]`
- Total window: `array[i-3], array[i-2], array[i-1], array[i], array[i+1], array[i+2], array[i+3]`

In code: `start = tid - r = i - 3`

---

## 3. The Sliding Window — Step by Step

### Example: input = `[1, 3, 5, 7, 9, 2, 4, 6, 8]`, mask = `[1, 2, 3]` (m=3, r=1)

```
  ═══════════════════════════════════════════════════════════════════════
  STEP 0: Computing result[0]     (tid=0, start = 0-1 = -1)
  ═══════════════════════════════════════════════════════════════════════

  Input:         [  1  ] [  3  ] [  5  ] [  7  ] [  9  ] [  2  ] [  4  ] [  6  ] [  8  ]
                    0       1       2       3       4       5       6       7       8

  Mask position:
            ┌─────┐
       ░░░  │  1  │ [  3  ]         ░░░ = Ghost cell (out of bounds → treat as 0)
       [0]    [1]    [2]
      mask   mask   mask
            ▲
          CENTER at index 0

  Calculation:  0×1  +  1×2  +  3×3  =  0 + 2 + 9  =  11
                 ▲       ▲       ▲
              ghost    arr[0]  arr[1]
              (=0)

  result[0] = 11


  ═══════════════════════════════════════════════════════════════════════
  STEP 1: Computing result[1]     (tid=1, start = 1-1 = 0)
  ═══════════════════════════════════════════════════════════════════════

  Input:         [  1  ] [  3  ] [  5  ] [  7  ] [  9  ] [  2  ] [  4  ] [  6  ] [  8  ]
                    0       1       2       3       4       5       6       7       8

  Mask position:
                 ┌─────┐
            [  1  ] [  3  ] [  5  ]
              [0]     [1]     [2]
             mask    mask    mask
                     ▲
                   CENTER at index 1

  Calculation:  1×1  +  3×2  +  5×3  =  1 + 6 + 15  =  22
               arr[0]  arr[1]  arr[2]

  result[1] = 22


  ═══════════════════════════════════════════════════════════════════════
  STEP 2: Computing result[2]     (tid=2, start = 2-1 = 1)
  ═══════════════════════════════════════════════════════════════════════

  Input:         [  1  ] [  3  ] [  5  ] [  7  ] [  9  ] [  2  ] [  4  ] [  6  ] [  8  ]
                    0       1       2       3       4       5       6       7       8

  Mask position:
                          ┌─────┐
                     [  3  ] [  5  ] [  7  ]
                       [0]     [1]     [2]
                      mask    mask    mask
                              ▲
                            CENTER at index 2

  Calculation:  3×1  +  5×2  +  7×3  =  3 + 10 + 21  =  34
               arr[1]  arr[2]  arr[3]

  result[2] = 34


              . . . mask slides one position right each step . . .


  ═══════════════════════════════════════════════════════════════════════
  STEP 8: Computing result[8]     (tid=8, start = 8-1 = 7)     LAST
  ═══════════════════════════════════════════════════════════════════════

  Input:         [  1  ] [  3  ] [  5  ] [  7  ] [  9  ] [  2  ] [  4  ] [  6  ] [  8  ]
                    0       1       2       3       4       5       6       7       8

  Mask position:
                                                                          ┌─────┐
                                                                     [  6  ] [  8  ]  ░░░
                                                                       [0]     [1]    [2]
                                                                      mask    mask   mask
                                                                              ▲
                                                                            CENTER at index 8

  Calculation:  6×1  +  8×2  +  0×3  =  6 + 16 + 0  =  22
               arr[7]  arr[8]  ghost
                                (=0)

  result[8] = 22
```

---

## 4. Boundary Handling (Ghost Cells / Zero Padding)

When the mask extends beyond the array edges, out-of-bounds positions are treated as **0**.

```
  Array indices:     0   1   2   3   4   ...   n-3  n-2  n-1
                   ┌───┬───┬───┬───┬───┬─────┬───┬───┬───┐
                   │   │   │   │   │   │ ... │   │   │   │
                   └───┴───┴───┴───┴───┴─────┴───┴───┴───┘

  For tid=0, r=3:
         ◄── ghost cells ──►
     ░░░  ░░░  ░░░  [0]  [1]  [2]  [3]
      ↑    ↑    ↑    ↑    ↑    ↑    ↑
     =0   =0   =0   a0   a1   a2   a3      ← The 3 leftmost ghost cells are 0
     ×w0  ×w1  ×w2  ×w3  ×w4  ×w5  ×w6

  For tid=n-1, r=3:
                                          ◄── ghost cells ──►
    [n-4] [n-3] [n-2] [n-1]  ░░░  ░░░  ░░░
      ↑     ↑     ↑     ↑     ↑    ↑    ↑
     a...  a...  a...  a...   =0   =0   =0  ← The 3 rightmost ghost cells are 0
     ×w0   ×w1   ×w2   ×w3   ×w4  ×w5  ×w6

  Code implementation:
  ┌────────────────────────────────────────────────────────────────────┐
  │  if (((start + j) >= 0) && ((start + j) < n)) {                  │
  │      temp += array[start + j] * mask[j];   // valid range        │
  │  }                                                                │
  │  // else: implicitly adds 0 (temp is unchanged)                  │
  └────────────────────────────────────────────────────────────────────┘
```

---

## 5. CUDA Thread Mapping (Parallelism)

Each CUDA thread computes **exactly one output element** — no dependencies between threads.

```
  ┌────────────────────────────── GPU Grid ──────────────────────────────┐
  │                                                                      │
  │   Block 0                  Block 1                  Block 2          │
  │  ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐  │
  │  │ T0  T1  ... T255 │    │ T256 T257 ...T511│    │ T512 ...       │  │
  │  │  ↓   ↓       ↓   │    │  ↓    ↓       ↓  │    │  ↓             │  │
  │  │ r[0]r[1]  r[255] │    │r[256]r[257] r[511]│   │r[512] ...     │  │
  │  └──────────────────┘    └──────────────────┘    └────────────────┘  │
  │                                                                      │
  │  ...                                                                 │
  │                                                                      │
  │   Block 4095 (last)                                                  │
  │  ┌──────────────────┐                                                │
  │  │ ...  T(n-1)      │    n = 2^20 = 1,048,576 elements              │
  │  │       ↓          │    THREADS = 256 per block                     │
  │  │    r[n-1]        │    GRID = (n + 255) / 256 = 4096 blocks       │
  │  └──────────────────┘                                                │
  └──────────────────────────────────────────────────────────────────────┘

  tid = blockIdx.x * blockDim.x + threadIdx.x
      = blockIdx.x * 256       + threadIdx.x
```

---

## 6. Full Numerical Example (m=7, r=3)

```
  Input array:   [ 2,  8,  4,  1,  5,  9,  3,  7,  6 ]
  Mask:          [ 1,  1,  2,  3,  2,  1,  1 ]           (m=7, r=3)

  ──────────────────────────────────────────────────────────────────────
  Computing result[0]:   start = 0 - 3 = -3

    Position:   -3   -2   -1    0    1    2    3
    Input:      [0]  [0]  [0]  [2]  [8]  [4]  [1]     (ghost = 0)
    Mask:       [1]  [1]  [2]  [3]  [2]  [1]  [1]

    Sum = 0×1 + 0×1 + 0×2 + 2×3 + 8×2 + 4×1 + 1×1
        = 0   + 0   + 0   + 6   + 16  + 4   + 1
        = 27

  ──────────────────────────────────────────────────────────────────────
  Computing result[3]:   start = 3 - 3 = 0        (fully inside, no ghosts)

    Position:    0    1    2    3    4    5    6
    Input:      [2]  [8]  [4]  [1]  [5]  [9]  [3]
    Mask:       [1]  [1]  [2]  [3]  [2]  [1]  [1]

    Sum = 2×1 + 8×1 + 4×2 + 1×3 + 5×2 + 9×1 + 3×1
        = 2   + 8   + 8   + 3   + 10  + 9   + 3
        = 43

  ──────────────────────────────────────────────────────────────────────
  Computing result[8]:   start = 8 - 3 = 5

    Position:    5    6    7    8    9   10   11
    Input:      [9]  [3]  [7]  [6]  [0]  [0]  [0]     (ghost = 0)
    Mask:       [1]  [1]  [2]  [3]  [2]  [1]  [1]

    Sum = 9×1 + 3×1 + 7×2 + 6×3 + 0×2 + 0×1 + 0×1
        = 9   + 3   + 14  + 18  + 0   + 0   + 0
        = 44
```

---

## 7. Summary: Data Flow in the CUDA Kernel

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                        For each thread (tid):                    │
  │                                                                  │
  │   1.  r = m / 2                    // mask radius                │
  │   2.  start = tid - r              // left edge of window        │
  │   3.  temp = 0                     // accumulator                │
  │   4.  for j in [0, m):                                           │
  │         if start+j in [0, n):      // boundary check             │
  │           temp += array[start+j]   // input element              │
  │                   × mask[j]        // mask weight                │
  │   5.  result[tid] = temp           // write output               │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

  Memory access pattern:
  ┌────────────┐     ┌────────┐     ┌─────────────┐
  │ Global Mem │────►│ Thread │────►│ Global Mem   │
  │ array[...]│     │ (ALU)  │     │ result[tid]  │
  │ mask[...]  │────►│        │     │              │
  └────────────┘     └────────┘     └──────────────┘

  Problem: Every thread reads 'm' elements from Global Memory (slow).
  This is why later optimizations use:
    - Constant memory for the mask  (05_convolution/1d_constant_memory/)
    - Shared memory tiling          (05_convolution/1d_tiled/)
```
