// This program shows how to get a number of device properties from API
// calls in CUDA. This is the foundation of "Runtime Auto-Tuning" for 
// cross-platform GPU frameworks (like PyTorch or cuBLAS).
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

// nvcc query_device.cpp -o query_device
// or
// g++ query_device.cpp -o query_device \
//   -I/usr/local/cuda/include \
//   -L/usr/local/cuda/lib64 \
//   -lcudart

#include <iostream>

// =================================================================================
// KNOWLEDGE HINT 1: The Essential Header
// We are not writing a .cu file with __global__ kernels here. This is pure C++.
// To access the CUDA API functions, we MUST include this header.
// Also, when compiling with g++, we MUST link the runtime library: `-lcudart`
// =================================================================================
#include <cuda_runtime.h>

using namespace std;

int main(){
    // =================================================================================
    // KNOWLEDGE HINT 2: The API Holy Trinity - Part 1 (The Census)
    // How many GPUs are plugged into this motherboard? 
    // This is crucial for multi-GPU training distributed across nodes.
    // =================================================================================
    int device_count;
    cudaGetDeviceCount(&device_count);
    cout << "There are " << device_count << " GPU(s) in the system" << endl;

    for(int i = 0; i < device_count; i++){
        // =================================================================================
        // KNOWLEDGE HINT 3: The API Holy Trinity - Part 2 (The Selection)
        // If we have 4 GPUs, we must explicitly tell the CUDA Runtime which one we 
        // are talking to. Any subsequent cudaMalloc or Kernel launch will be routed 
        // to this specific device.
        // =================================================================================
        cudaSetDevice(i);

        // =================================================================================
        // KNOWLEDGE HINT 4: The API Holy Trinity - Part 3 (The Interrogation)
        // We declare a massive struct called `cudaDeviceProp`.
        // Then we pass its memory address to `cudaGetDeviceProperties`. The CUDA driver 
        // will fill this struct with dozens of hardware specifications for Device 'i'.
        // =================================================================================
        
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        
        // Print the commercial name of the GPU (e.g., "GeForce RTX 4090")
        cout << "Device " << i << " is a " << device_prop.name << endl;

        // Get information about the software stack
        int driver;
        int runtime;
        cudaDriverGetVersion(&driver);
        cudaRuntimeGetVersion(&runtime);
        cout << "Driver: " << driver << " Runtime: " << runtime << endl;

        // =================================================================================
        // KNOWLEDGE HINT 5: Architecture Identity (Compute Capability)
        // This is the GPU's DNA. 
        // Major = 6 (Pascal), 7 (Volta/Turing), 8 (Ampere/Ada Lovelace), 9 (Hopper).
        // A framework will use this to dynamically select which pre-compiled kernel to run 
        // (e.g., switching to Tensor Core kernels if Major >= 7).
        // =================================================================================
        
        cout << "CUDA capability: " << device_prop.major << "." <<
            device_prop.minor << endl;

        // =================================================================================
        // KNOWLEDGE HINT 6: Memory Constraints (The Ceilings)
        // totalGlobalMem: Total VRAM. We divide by 2^30 (1 << 30) to convert Bytes to GB.
        // If we want to allocate a matrix larger than this, the program will crash.
        // =================================================================================
        cout << "Global memory in GB: " <<
            device_prop.totalGlobalMem / (1 << 20) << endl;

        // =================================================================================
        // KNOWLEDGE HINT 7: Execution Resources (The Muscle)
        // multiProcessorCount: The number of Streaming Multiprocessors (SMs).
        // This dictates the absolute parallel computing power of the GPU.
        // =================================================================================
        cout << "Number of SMs: " << device_prop.multiProcessorCount <<
            endl;

        // The frequency. clockRate is returned in Kilohertz (KHz).
        // Multiplying by 1e-6 (or dividing by 1,000,000) converts KHz to GHz.
        cout << "Max clock rate: " << device_prop.clockRate * 1e-6 <<
            "GHz" << endl;

        // The L2 cache size. Divided by 2^20 (1 << 20) to convert Bytes to MB.
        // Knowing this helps developers decide how aggressively they can rely on 
        // cache hits for non-coalesced memory accesses.
        cout << "The L2 cache size in MB: " <<
            device_prop.l2CacheSize / (1 << 20) << endl;

        // =================================================================================
        // KNOWLEDGE HINT 8: Shared Memory Limits (Crucial for Tiling)
        // This defines the maximum size of the "Village Ballot Box" or "Scratchpad" 
        // per Thread Block. We divide by 2^10 (1 << 10) to convert Bytes to KB.
        // If your Matrix Multiplication Tile requires 64KB, but this returns 48KB, 
        // your kernel will fail to launch! You must dynamically adjust your Tile Size.
        // =================================================================================
        cout << "Total shared memory per block in KB: " <<
            device_prop.sharedMemPerBlock / (1 << 10) << endl;

        // And much more! (e.g., warpSize, maxThreadsPerBlock, maxGridSize, etc.)
    }

    return 0;
}