// CUDA Matrix Multiplication Analysis: Access Patterns & Alignment
// Original By: Nick from CoffeeBeforeArch
// Improved & Annotated by: ZDSJTU
// Profiling support: accepts N, M, kernel_id from command line
//
// Three kernels to compare memory access scenarios:
// kernel_id=0: Naive (A: Broadcast, B: Coalesced)
// kernel_id=1: Transposed A (A: Improved Locality, B: Coalesced)
// kernel_id=2: Misaligned B (A: Broadcast, B: Strided/Uncoalesced)

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
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;

    if ((row < n) && (col < n)) {
        for (int k = 0; k < n; k++) {
            temp_sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = temp_sum;
    }
}

// =================================================================
// KERNEL 2: Transposed A (Optimized A)
// =================================================================
__global__ void matrixMulTransposedA(const int *a_t, const int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;

    if ((row < n) && (col < n)) {
        for (int k = 0; k < n; k++) {
            temp_sum += a_t[k * n + row] * b[k * n + col];
        }
        c[row * n + col] = temp_sum;
    }
}

// =================================================================
// KERNEL 3: Misaligned B (The Disaster Case)
// =================================================================
__global__ void matrixMulMisalignedB(const int *a, const int *b_t, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;

    if ((row < n) && (col < n)) {
        for (int k = 0; k < n; k++) {
            temp_sum += a[row * n + k] * b_t[col * n + k];
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

// Helper: Verify Result (sample-based for large N)
void check_answer(const int *a, const int *b, const int *c, int n) {
    for (int i = 0; i < 10; i++) {
        int row = rand() % n;
        for (int j = 0; j < n; j++) {
            int temp = 0;
            for (int k = 0; k < n; k++) {
                temp += a[row * n + k] * b[k * n + j];
            }
            if (c[row * n + j] != temp) {
                fprintf(stderr, "FAILED at [%d][%d]. GPU: %d, CPU: %d\n",
                        row, j, c[row * n + j], temp);
                exit(1);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments: N (matrix size), M (iterations), kernel_id
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <N> <M> <kernel_id>\n", argv[0]);
        fprintf(stderr, "  kernel_id: 0=Naive, 1=TransposedA, 2=MisalignedB\n");
        return 1;
    }
    int n = atoi(argv[1]);
    int M = atoi(argv[2]);
    int kernel_id = atoi(argv[3]);

    size_t bytes = (size_t)n * n * sizeof(int);

    // Host allocation
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);
    int *h_a_transposed = (int*)malloc(bytes);
    int *h_b_transposed = (int*)malloc(bytes);

    // Initialization
    srand(1337);
    matrix_init(h_a, n);
    matrix_init(h_b, n);
    transpose_host(h_a, h_a_transposed, n);
    transpose_host(h_b, h_b_transposed, n);

    // Device allocation
    int *d_a, *d_b, *d_c, *d_a_t, *d_b_t;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_a_t, bytes);
    cudaMalloc(&d_b_t, bytes);

    // Host -> Device copy
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_t, h_a_transposed, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_t, h_b_transposed, bytes, cudaMemcpyHostToDevice);

    // Grid/block configuration
    int BLOCK_SIZE = 32;
    int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Warmup launch for the selected kernel
    switch (kernel_id) {
        case 0:
            matrixMulNaive<<<grid, threads>>>(d_a, d_b, d_c, n);
            break;
        case 1:
            matrixMulTransposedA<<<grid, threads>>>(d_a_t, d_b, d_c, n);
            break;
        case 2:
            matrixMulMisalignedB<<<grid, threads>>>(d_a, d_b_t, d_c, n);
            break;
        default:
            fprintf(stderr, "Invalid kernel_id: %d\n", kernel_id);
            return 1;
    }
    cudaDeviceSynchronize();

    // Verify correctness once (using original A and B)
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    check_answer(h_a, h_b, h_c, n);

    // Timed iterations (kernel-only, excluding memcpy)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0.0f;

    for (int i = 0; i < M; i++) {
        cudaEventRecord(start);
        switch (kernel_id) {
            case 0:
                matrixMulNaive<<<grid, threads>>>(d_a, d_b, d_c, n);
                break;
            case 1:
                matrixMulTransposedA<<<grid, threads>>>(d_a_t, d_b, d_c, n);
                break;
            case 2:
                matrixMulMisalignedB<<<grid, threads>>>(d_a, d_b_t, d_c, n);
                break;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    // Output average time in ms (parsed by profiling script)
    printf("%f\n", total_ms / M);

    // Cleanup
    free(h_a); free(h_b); free(h_c);
    free(h_a_transposed); free(h_b_transposed);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_a_t); cudaFree(d_b_t);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
