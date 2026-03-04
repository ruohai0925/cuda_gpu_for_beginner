// CUDA Matrix Multiplication Analysis: Access Patterns & Alignment
// Original By: Nick from CoffeeBeforeArch
// Improved & Annotated by: ZDSJTU

//
// Objective: Compare three memory access scenarios:
// 1. Naive: Standard implementation.
//    - Matrix A: Broadcast (Cache Hit dependent)
//    - Matrix B: Coalesced (Good)
// 2. Transposed A: Pre-transpose Matrix A to improve locality.
//    - Matrix A (Physical): Spatial Locality improved across Warps (Better L2 Cache hits)
//    - Matrix B: Coalesced (Good)
// 3. Misaligned B: Pre-transpose Matrix B to force strided access.
//    - Matrix A: Broadcast
//    - Matrix B (Physical): Strided / Uncoalesced (Performance Disaster)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// =================================================================
// KERNEL 1: Naive Implementation
// =================================================================
__global__ void matrixMulNaive(const int *a, const int *b, int *c, int n) {
    // Standard mapping: Row depends on Y, Col depends on X
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;

    if ((row < n) && (col < n)) {
        for (int k = 0; k < n; k++) {
            // Access Pattern Analysis:
            // ------------------------
            // Matrix A: a[row * n + k]
            // - Within a Warp (32 threads), 'row' is constant (usually).
            // - All threads read the SAME address. This is a BROADCAST.
            // - Issue: Different Warps (rows 0, 1, 2...) read addresses separated by stride N.
            // - Result: Moderate performance, heavy L2 cache pressure.
            int val_a = a[row * n + k];

            // Viewpoint A: Your Perspective (The Single-Thread Logic)
            // Imagine we are tracking Thread (0, 0) and Thread (1, 0).
            // Thread (0, 0) is responsible for calculating $C[0][0]$. 
            // It needs to read Row 0 of Matrix A.
            // At loop $k=0$, it reads $A[0][0]$.
            // At loop $k=1$, it reads $A[0][1]$.
            // Thread (1, 0) is responsible for calculating $C[1][0]$. 
            // It needs to read Row 1 of Matrix A.
            // At loop $k=0$, it reads $A[1][0]$.
            // At loop $k=1$, it reads $A[1][1]$.

            // Viewpoint B: The Memory Controller's Perspective "
            // But the GPU doesn't execute threads one by one. 
            // It is massively parallel.
            // Thread (0, 0) and Thread (1, 0) are executing the code simultaneously.
            // Let's freeze time at the exact moment where $k=0$.
            // What happens?
            // Thread (0, 0) shouts: 'Give me $A[0][0]$!'
            // Thread (1, 0) shouts: 'Give me $A[1][0]$!'

            // Matrix B: b[k * n + col]
            // - Within a Warp, 'col' varies (0, 1, 2...).
            // - Threads read adjacent addresses (Stride = 1).
            // - Result: PERFECT COALESCING. High bandwidth efficiency.
            int val_b = b[k * n + col];

            temp_sum += val_a * val_b;
        }
        c[row * n + col] = temp_sum;
    }
}

// =================================================================
// KERNEL 2: Transposed A (Optimized A)
// =================================================================
// Input 'a' here is physically A_Transpose.
// Logic: To get logical A[row][k], we read physical A_T[k][row].
__global__ void matrixMulTransposedA(const int *a_t, const int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;

    if ((row < n) && (col < n)) {
        for (int k = 0; k < n; k++) {
            // Access Pattern Analysis:
            // ------------------------
            // Matrix A (Transposed): a_t[k * n + row]
            // - Why is this better?
            // - Logic: A[row][k] -> Physical: A_T[k][row]
            // - Warp 0 needs row 0 -> reads A_T[k][0]
            // - Warp 1 needs row 1 -> reads A_T[k][1]
            // - While each Warp still does a broadcast, the ENTIRE BLOCK (32 Warps)
            //   is now reading a contiguous chunk of memory (A_T[k][0...31]).
            // - Result: Massive improvement in L2 Cache Spatial Locality.
            int val_a = a_t[k * n + row];

            // Matrix B: Standard Coalesced (Same as Naive)
            int val_b = b[k * n + col];

            temp_sum += val_a * val_b;
        }
        c[row * n + col] = temp_sum;
    }
}

// =================================================================
// KERNEL 3: Misaligned B (The Disaster Case)
// =================================================================
// Input 'b' here is physically B_Transpose.
// Logic: To get logical B[k][col], we read physical B_T[col][k].
__global__ void matrixMulMisalignedB(const int *a, const int *b_t, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;

    if ((row < n) && (col < n)) {
        for (int k = 0; k < n; k++) {
            // Matrix A: Standard Broadcast (Same as Naive)
            int val_a = a[row * n + k];

            // Access Pattern Analysis:
            // ------------------------
            // Matrix B (Transposed): b_t[col * n + k]
            // - Logic: B[k][col] -> Physical: B_T[col][k]
            // - Within a Warp, 'col' varies (0, 1, 2...).
            // - Accesses: 0*N+k, 1*N+k, 2*N+k...
            // - Stride: N (1024 integers).
            // - Result: UNCOALESCED / STRIDED ACCESS.
            // - The memory controller must issue 32 separate transactions for one warp.
            // - Expect ~5x or worse slowdown.
            int val_b = b_t[col * n + k];

            temp_sum += val_a * val_b;
        }
        c[row * n + col] = temp_sum;
    }
}

// Helper: Initialize Matrix
void matrix_init(int *a, int n) {
    for (int i = 0; i < n * n; i++) {
        a[i] = rand() % 100;
    }
}

// Helper: Transpose Matrix (Host Side)
void transpose_host(const int *src, int *dst, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[j * n + i] = src[i * n + j];
        }
    }
}

// Helper: Verify Result
void check_answer(const int *a, const int *b, const int *c, int n) {
    printf("Verifying result (CPU)... ");
    // Only check a small subset to save time, or check full if fast enough
    // For n=1024, O(N^3) is slow on CPU. Let's check 10 random rows.
    for (int i = 0; i < 10; i++) {
        int row = rand() % n;
        for (int j = 0; j < n; j++) {
            int temp = 0;
            for (int k = 0; k < n; k++) {
                temp += a[row * n + k] * b[k * n + j];
            }
            if (c[row * n + j] != temp) {
                printf("FAILED at [%d][%d]. GPU: %d, CPU: %d\n", row, j, c[row * n + j], temp);
                return;
            }
        }
    }
    printf("PASSED (Random Sample Checked).\n");
}

int main() {
    // 1. Setup Parameters
    int n = 1 << 11; // 1024 * 2
    size_t bytes = n * n * sizeof(int);
    printf("Matrix Size: %d x %d\n", n, n);

    // 2. Host Allocation
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);
    int *h_a_transposed = (int*)malloc(bytes); // For Case 2
    int *h_b_transposed = (int*)malloc(bytes); // For Case 3

    // 3. Initialization
    srand(1337);
    matrix_init(h_a, n);
    matrix_init(h_b, n);
    transpose_host(h_a, h_a_transposed, n);
    transpose_host(h_b, h_b_transposed, n);

    // 4. Device Allocation
    int *d_a, *d_b, *d_c, *d_a_t, *d_b_t;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_a_t, bytes);
    cudaMalloc(&d_b_t, bytes);

    // 5. Host -> Device Copy
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_t, h_a_transposed, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_t, h_b_transposed, bytes, cudaMemcpyHostToDevice);

    // 6. Configuration
    int BLOCK_SIZE = 32; // Use 32 to align with Warp size perfectly
    int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Setup Timing Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // =====================================================
    // TEST CASE 1: Naive (Standard)
    // =====================================================
    printf("\nRunning Case 1: Naive (A: Broadcast, B: Coalesced)...\n");
    // Warmup
    matrixMulNaive<<<grid, threads>>>(d_a, d_b, d_c, n);
    
    cudaEventRecord(start);
    for(int i=0; i<10; i++)
        matrixMulNaive<<<grid, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Avg Time: %.3f ms\n", milliseconds / 10.0f);
    
    // Check correctness using original h_a and h_b
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    check_answer(h_a, h_b, h_c, n);

    // =====================================================
    // TEST CASE 2: Transposed A (Optimized)
    // =====================================================
    printf("\nRunning Case 2: Transposed A (A: Improved Locality, B: Coalesced)...\n");
    // Note: We pass d_a_t (transposed) as the first argument
    matrixMulTransposedA<<<grid, threads>>>(d_a_t, d_b, d_c, n); // Warmup

    cudaEventRecord(start);
    for(int i=0; i<10; i++)
        matrixMulTransposedA<<<grid, threads>>>(d_a_t, d_b, d_c, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Avg Time: %.3f ms\n", milliseconds / 10.0f);

    // Verification logic remains same as result C should be identical
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    // check_answer(h_a, h_b, h_c, n); // Skip to save time

    // =====================================================
    // TEST CASE 3: Misaligned B (Disaster)
    // =====================================================
    printf("\nRunning Case 3: Misaligned B (A: Broadcast, B: Strided/Uncoalesced)...\n");
    // Note: We pass d_b_t (transposed) as the second argument
    matrixMulMisalignedB<<<grid, threads>>>(d_a, d_b_t, d_c, n); // Warmup

    cudaEventRecord(start);
    for(int i=0; i<10; i++)
        matrixMulMisalignedB<<<grid, threads>>>(d_a, d_b_t, d_c, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Avg Time: %.3f ms\n", milliseconds / 10.0f);

    // Verification
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    // check_answer(h_a, h_b, h_c, n);

    // Cleanup
    free(h_a); free(h_b); free(h_c); free(h_a_transposed); free(h_b_transposed);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_a_t); cudaFree(d_b_t);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("\nDONE.\n");
    return 0;
}