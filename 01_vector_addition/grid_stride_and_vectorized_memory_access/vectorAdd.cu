// Vector Addition with Grid Stride Loops & Vectorized Memory Access
// Based on NVIDIA Developer Blogs
// Updated by: ZDSJTU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// -------------------------------------------------------------------------
// Kernel 1: Grid Stride Loop (Flexible & Reusable)
// -------------------------------------------------------------------------
// Instead of assuming 1 Thread = 1 Element (Monolithic), we use a loop.
// 
// Concept:
// Imagine the Grid is a "window" sliding over the data array.
// All threads process the current window, then jump ahead by the total 
// number of threads in the grid (Stride) to process the next window.
//
// Benefits:
// 1. Decoupling: Grid size is independent of data size N. 
//    You can process 1 million elements with just 128 threads if needed.
// 2. Debugging: You can run this with <<<1, 1>>> to emulate serial execution.
// 3. Portability: Performance scales automatically with hardware size.
__global__ void vectorAddGridStride(const int *__restrict a, const int *__restrict b,
                                    int *__restrict c, int N) {
    // Calculate the global ID of this thread within the grid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the total number of threads in the grid (The Stride)
    int stride = blockDim.x * gridDim.x;

    // Loop over the array
    // If N > Total Threads, threads will loop back and handle more elements.
    for (int i = tid; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// -------------------------------------------------------------------------
// Kernel 2: Vectorized Memory Access (High Bandwidth)
// -------------------------------------------------------------------------
// Instead of reading 1 integer (4 bytes) at a time, we read 4 integers (16 bytes).
//
// Concept:
// GPU memory controllers work best with wide transactions.
// Loading 'int4' generates a single LD.E.128 instruction (128-bit load),
// reducing the instruction count by 4x and improving bus utilization.
//
// Constraints:
// 1. Alignment: The pointers 'a', 'b', 'c' must be aligned to 16 bytes.
//    (cudaMalloc guarantees 256-byte alignment, so we are safe).
// 2. Data Size: N must be divisible by 4. If not, you need a "peeling loop"
//    to handle the remaining 1, 2, or 3 elements (omitted here for clarity).
__global__ void vectorAddVectorized(const int *__restrict a, const int *__restrict b,
                                    int *__restrict c, int N) {
    // 1. Reinterpret pointers as int4*
    // This tricks the compiler into issuing 128-bit load/store instructions.
    const int4 *a4 = reinterpret_cast<const int4*>(a);
    const int4 *b4 = reinterpret_cast<const int4*>(b);
    int4 *c4 = reinterpret_cast<int4*>(c);

    // 2. Adjust N because we are processing 4 elements at a time
    int num_vectors = N / 4;

    // 3. Use Grid Stride Loop on the vectors
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_vectors; i += stride) {
        int4 val_a = a4[i]; // LD.E.128 (Load 16 bytes)
        int4 val_b = b4[i]; // LD.E.128 (Load 16 bytes)
        int4 res;

        // Perform addition on components
        // CUDA does not define operator+ for int4 by default
        res.x = val_a.x + val_b.x;
        res.y = val_a.y + val_b.y;
        res.z = val_a.z + val_b.z;
        res.w = val_a.w + val_b.w;

        c4[i] = res; // ST.E.128 (Store 16 bytes)
    }
}

void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
    for (size_t i = 0; i < a.size(); i++) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Mismatch at " << i << ": " << c[i] << " != " << a[i] + b[i] << std::endl;
            exit(1);
        }
    }
}

int main() {
    // Use a large N to see performance benefits (2^20 = ~1 million elements)
    // Note: For Vectorized kernel, N must be divisible by 4.
    constexpr int N = 1 << 26; // 67108864 elements
    constexpr size_t bytes = sizeof(int) * N;

    std::vector<int> a;
    a.reserve(N);
    std::vector<int> b;
    b.reserve(N);
    std::vector<int> c(N);

    for (int i = 0; i < N; i++) {
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
        c.push_back(0); // Initialize C
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 1 << 10;  // 1 << 10 = 2^10 = 1024 threads per CTA

    // -----------------------------------------------------------------
    // Configuration for Grid Stride
    // -----------------------------------------------------------------
    // With Grid Stride loops, NUM_BLOCKS is largely a performance tuning parameter.
    // We calculate it based on the GPU's "Occupancy" (how many blocks can run at once).
    // A common heuristic is to have a multiple of the # of SMs on your GPU.
    // For simplicity here, we'll just pick a reasonable fixed number.
    // We don't HAVE to cover N. The loop inside the kernel covers N.
    int NUM_BLOCKS = 128; // e.g., 32 SMs * 4 blocks per SM

    std::cout << "Launching Grid Stride Kernel..." << std::endl;
    std::cout << "NUM_BLOCKS: " << NUM_BLOCKS << std::endl;
    std::cout << "NUM_THREADS: " << NUM_THREADS << std::endl;
    // Launch Kernel 1
    vectorAddGridStride<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
    for (int i = 0; i < 100; i++) {
        vectorAddGridStride<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
    }

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    verify_result(a, b, c);
    std::cout << "Grid Stride Kernel Verified." << std::endl;

    // Reset C on device for next test
    cudaMemset(d_c, 0, bytes);

    std::cout << "\nLaunching Vectorized Kernel..." << std::endl;
    // Launch Kernel 2
    // Note: N must be divisible by 4 for this simplified kernel.
    vectorAddVectorized<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
    for (int i = 0; i < 100; i++) {
        vectorAddVectorized<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
    }
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    verify_result(a, b, c);
    std::cout << "Vectorized Kernel Verified." << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    std::cout << "COMPLETED SUCCESSFULLY\n" << std::endl;

    return 0;
}