// This program shows how to automatically execute on the GPU using OpenACC
// It demonstrates Performance Portability: this code can run on an NVIDIA GPU, 
// an AMD GPU, or a multicore CPU without changing a single line of C++ code!
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

// https://developer.nvidia.com/hpc-sdk/releases/26.1
// nvc++ -acc -gpu=cc89 -Minfo=accel -S matrix_mul.cpp -o matrix_mul.asm
// nvc++ -acc -gpu=cc89 -Minfo=accel matrix_mul.cpp -o matrix_mul
// 2376773 ns = 2.376773 ms for OpenACC vs. 1.71 ms for CUDA

#include <stdlib.h>

// Simple function to init matrices with random numbers
void init_matrix(int *m, int N){
    for(int i = 0; i < N*N; i++){
        m[i] = rand() % 100;
    }
}

int main(){
    // Dimensions of matrices (N = 1024)
    int N = 1 << 10;

    // Allocate space for matrices using standard C++ 'new'
    // NO cudaMalloc is needed here!
    int *a = new int[N * N];
    int *b = new int[N * N];
    int *c = new int[N * N];

    // Init matrices on the Host (CPU)
    init_matrix(a, N);
    init_matrix(b, N);

    // =================================================================================
    // KNOWLEDGE HINT 1: The Accelerator Directive & Automated Data Management
    // ---------------------------------------------------------------------------------
    // "#pragma acc kernels" tells the compiler: "Compile the following block of code 
    // to run on the accelerator (GPU)."
    // 
    // "copyin(a[0:N*N], b[0:N*N])" automates cudaMemcpyHostToDevice. It tells the GPU 
    // to allocate memory and copy matrices A and B from the CPU to the GPU. They are 
    // read-only for the GPU, so they don't need to be copied back.
    // 
    // "copy(c[0:N*N])" automates bidirectional transfer. It allocates C on the GPU, 
    // and crucially, copies the final results BACK to the CPU when the region ends.
    // Notice the array shaping syntax "[0:N*N]" which tells the compiler the array bounds.
    // =================================================================================
    
    #pragma acc kernels copyin(a[0:N*N], b[0:N*N]), copy(c[0:N*N])
    {
        // =================================================================================
        // KNOWLEDGE HINT 2: Loop Parallelization (Grid/Block Generation)
        // ---------------------------------------------------------------------------------
        // "#pragma acc loop independent" explicitly tells the compiler that the iterations 
        // of this 'i' loop (the rows) do not depend on each other. 
        // The compiler will automatically map this loop to GPU Thread Blocks (Blocks).
        // =================================================================================
        #pragma acc loop independent
        for(int i = 0; i < N; i++){
            
            // The 'j' loop (the columns) is also completely independent. 
            // The compiler will typically map this to individual Threads within a Block.
            #pragma acc loop independent
            for(int j = 0; j < N; j++){
                
                // Local variable to accumulate the dot product. 
                // The compiler automatically places this in a fast GPU Register.
                float sum = 0;
                
                // =================================================================================
                // KNOWLEDGE HINT 3: Atomic Avoidance via Reduction Clause
                // ---------------------------------------------------------------------------------
                // In our previous Histogram lesson, multiple threads adding to the same 
                // variable caused a "Data Race". 
                // Here, the 'k' loop constantly updates the shared 'sum' variable. 
                // By adding "reduction(+:sum)", we tell the compiler: "Safely divide the 
                // work among threads, and combine their partial 'sum' values using addition (+), 
                // without corrupting the data!" 
                // The compiler writes the complex synchronization/atomic code for us!
                // =================================================================================
                
                #pragma acc loop independent reduction (+: sum)
                for(int k = 0; k < N; k++){
                    sum += a[i * N + k] * b[k * N + j];
                }
                
                // Write the final accumulated register value back to the global matrix C
                c[i * N + j] = sum;
            }
        }
    } // End of "#pragma acc kernels" block. 
      // The compiler automatically inserts cudaMemcpyDeviceToHost here for matrix 'c'!

    return 0;
}