// This program implements 2D convolution using Constant Memory in CUDA.
// It serves as the baseline for spatial (2D) grid algorithms.
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

#include <cassert>
#include <cstdlib>
#include <iostream>

// =================================================================================
// KNOWLEDGE HINT 1: The 2D Mask (From Line to Matrix)
// Our mask is no longer a 1D array of length 7. It is now a 7x7 matrix.
// Total elements: 49. Total size: 196 bytes.
// Because it is so small and read-only, Constant Memory is still the perfect home!
// =================================================================================
#define MASK_DIM 7

// The radius (amount the mask hangs over the center element)
// Integer division: 7 / 2 = 3.
#define MASK_OFFSET (MASK_DIM / 2)


// Allocate the 7x7 mask in the GPU's ultra-fast Constant Cache.
__constant__ int mask[MASK_DIM * MASK_DIM];

// =================================================================================
// CUDA KERNEL: 2-D Convolution (Naive boundary checks + Constant Memory)
// =================================================================================
__global__ void convolution_2d(int *matrix, int *result, int N) {
  
  // =================================================================================
  // KNOWLEDGE HINT 2: 2D Thread Topology
  // Instead of a single 'tid', we must map our grid to a 2D Cartesian plane (X, Y).
  // x-axis corresponds to columns. y-axis corresponds to rows.
  // =================================================================================
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Starting indices for applying the mask.
  // This calculates the absolute (row, col) coordinate of the TOP-LEFT corner 
  // of where the mask currently sits over the input matrix.
  int start_r = row - MASK_OFFSET;
  int start_c = col - MASK_OFFSET;

  // Temp value for accumulating the local dot product
  int temp = 0;

  // =================================================================================
  // KNOWLEDGE HINT 3: The Double For-Loop (Mask Iteration)
  // We must iterate over the 2D structure of the 7x7 mask.
  // =================================================================================
  for (int i = 0; i < MASK_DIM; i++) {       // i iterates over mask rows
    for (int j = 0; j < MASK_DIM; j++) {     // j iterates over mask columns
      
      // =================================================================================
      // KNOWLEDGE HINT 4: 2D Boundary Checking (The 4-Way Halo)
      // A 2D mask can hang off the Top, Bottom, Left, or Right edges!
      // (start_r + i) >= 0 -> Protects the TOP edge
      // (start_r + i) < N  -> Protects the BOTTOM edge
      // =================================================================================
      
      if ((start_r + i) >= 0 && (start_r + i) < N) {
          
        // (start_c + j) >= 0 -> Protects the LEFT edge
        // (start_c + j) < N  -> Protects the RIGHT edge
        if ((start_c + j) >= 0 && (start_c + j) < N) {
            
          // If safe, perform the multiplication.
          // Note the 2D to 1D flattening math: index = Row * Width + Col
          // matrix access: (current_row) * N + (current_col)
          // mask access:   (mask_row) * MASK_DIM + (mask_col)
          temp += matrix[(start_r + i) * N + (start_c + j)] * mask[i * MASK_DIM + j];
        }
      }
    }
  }

  // Write back the final accumulated pixel to the result matrix
  result[row * N + col] = temp;
}

// Helper function to initialize matrices with random numbers
void init_matrix(int *m, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[n * i + j] = rand() % 100;
    }
  }
}

// =================================================================================
// CPU VERIFICATION FUNCTION
// KNOWLEDGE HINT 5: The "Quadruply" Nested Loop (CPU vs. GPU paradigm)
// Notice how the CPU requires 4 nested loops to perform this operation.
// Loops i & j iterate over every pixel in the image.
// Loops k & l iterate over the mask.
// In our GPU kernel, loops i & j disappear because they are implicitly handled 
// by launching millions of parallel threads in our Grid/Block configuration!
// =================================================================================
void verify_result(int *m, int *mask, int *result, int N) {
  int temp;
  int offset_r;
  int offset_c;

  // CPU must manually iterate over every Row and Column of the image
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      temp = 0;

      // Inner loops for the Mask
      for (int k = 0; k < MASK_DIM; k++) {
        offset_r = i - MASK_OFFSET + k;
        for (int l = 0; l < MASK_DIM; l++) {
          offset_c = j - MASK_OFFSET + l;

          // 4-way boundary check
          if (offset_r >= 0 && offset_r < N) {
            if (offset_c >= 0 && offset_c < N) {
              temp += m[offset_r * N + offset_c] * mask[k * MASK_DIM + l];
            }
          }
        }
      }
      // If the GPU result doesn't match the serial CPU result, crash the program.
      assert(result[i * N + j] == temp);
    }
  }
}

int main() {
  // Dimensions of the image/matrix: 1024 x 1024
  int N = 1 << 10;
  size_t bytes_n = N * N * sizeof(int); // ~4MB matrix

  // Allocate and initialize Host Memory
  int *matrix = new int[N * N];
  int *result = new int[N * N];
  init_matrix(matrix, N);

  size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);

  int *h_mask = new int[MASK_DIM * MASK_DIM];
  init_matrix(h_mask, MASK_DIM);

  // Allocate Device Memory
  int *d_matrix;
  int *d_result;
  cudaMalloc(&d_matrix, bytes_n);
  cudaMalloc(&d_result, bytes_n);

  // Move data to Device
  cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
  
  // Magic API to copy data to the Constant Cache symbol
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  // =================================================================================
  // KNOWLEDGE HINT 6: 2D Launch Configuration
  // We configure our execution using the dim3 struct.
  // block_dim(16, 16) means each Block contains a 16x16 square of 256 threads.
  // grid_dim(64, 64) means our Grid is a 64x64 checkerboard of these Blocks.
  // Total threads = 64 * 16 (X) by 64 * 16 (Y) = 1024 x 1024 threads!
  // =================================================================================
  int THREADS = 16;
  int BLOCKS = (N + THREADS - 1) / THREADS;

  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS, BLOCKS);

  // Launch the 2D Kernel
  convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, N);

  // Retrieve result
  cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Verify
  verify_result(matrix, h_mask, result, N);

  std::cout << "COMPLETED SUCCESSFULLY!";

  // Clean up
  delete[] matrix;
  delete[] result;
  delete[] h_mask;
  cudaFree(d_matrix);
  cudaFree(d_result);

  return 0;
}