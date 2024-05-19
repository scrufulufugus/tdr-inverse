#include "utils.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace helpers;

#define MAX_BLOCK_SIZE 1024

__global__ void pivot(matrix_t *matrix, size_t cols, size_t rows, size_t j) {
  // the ith row of the matrix
  __shared__ matrix_t Ri[MAX_BLOCK_SIZE];
  size_t colId = threadIdx.x;
  Ri[colId] = matrix[cols * j + colId];

  // Pivot
  size_t swapRow = j;
  for (size_t i = j; i < rows; i++) {
    if (abs(matrix[cols*i + j]) > abs(matrix[cols*swapRow + j])) {
      swapRow = i;
    }
  }

  Ri[colId] = matrix[cols * swapRow + colId];
  if (swapRow != j) {
    __syncthreads();
#ifdef DEBUG
    printf("1. swap(M[%lu][%lu], M[%lu][%lu]) = swap(%f, %f)\n", j, colId, swapRow, colId,
           matrix[cols * j + colId], matrix[cols * swapRow + colId]);
#endif
    matrix[cols * swapRow + colId] = matrix[cols * j + colId];
  }
}

__global__ void storeAij(matrix_t *matrix, size_t size, matrix_t *Aij, size_t colId, size_t offset) {
  size_t rowId = threadIdx.x + offset;
  Aij[rowId] = matrix[size*rowId + colId];
#ifdef DEBUG
  printf("0. A[%lu][%lu] = %f\n", rowId, colId, Aij[rowId]);
#endif

  if (rowId == colId)
    matrix[size*rowId + colId] = 1.0;
  else
    matrix[size*rowId + colId] = 0.0;
}

// (c) Sharma 2013
__global__ void fixRow(matrix_t *matrix, size_t size, matrix_t *Aij, size_t rowId, size_t offset) {
  // the ith row of the matrix
  __shared__ matrix_t Ri[MAX_BLOCK_SIZE];
  // The diagonal element for ith row
  __shared__ matrix_t Aii;
  size_t sharedColId = threadIdx.x;
  size_t colId = sharedColId + offset;
  Ri[sharedColId] = matrix[size * rowId + colId];
  Aii = Aij[rowId];

#ifdef DEBUG
  printf("1. matrix[%lu][%lu] = %f\n", rowId, colId, Ri[sharedColId]);
#endif
  __syncthreads();
  // Divide the whole row by the diagonal element making sure it is not 0
  Ri[sharedColId] = Ri[sharedColId] / Aii;
  matrix[size * rowId + colId] = Ri[sharedColId];
#ifdef DEBUG
  printf("2. matrix[%lu][%lu] /= %f = %f\n", rowId, colId, Aii, Ri[sharedColId]);
#endif
}

// (c) Sharma 2013
__global__ void fixColumn(matrix_t *matrix, size_t size, matrix_t *Aij, size_t colId, size_t rowOffset, size_t colOffset) {
  size_t sharedI = threadIdx.x;
  size_t i = sharedI + rowOffset;
  size_t j = blockIdx.x + colOffset;
  // The colId column
  __shared__ matrix_t col[MAX_BLOCK_SIZE];
  // The jth element of the colId row
  __shared__ matrix_t AColIdj;
  // The jth column
  __shared__ matrix_t colj[MAX_BLOCK_SIZE];
  col[sharedI] = Aij[i];
  __syncthreads();
  if (col[sharedI] != 0) {
    colj[sharedI] = matrix[i * size + j];
    AColIdj = matrix[colId * size + j];
    if (i != colId) {
#if   PRECISION == 1
      colj[i] = fmaf(-1. * AColIdj, col[sharedI], colj[sharedI]);
#elif PRECISION == 2
      colj[sharedI] = fma(-1. * AColIdj, col[sharedI], colj[sharedI]);
#endif
      //colj[i] = colj[i] - AColIdj * col[sharedI];
#ifdef DEBUG
      printf("3. matrix[%lu][%lu] -= %f * %f = %f\n", i, j, AColIdj, col[sharedI],
             colj[sharedI]);
#endif
    }
    matrix[i * size + j] = colj[sharedI];
  }
}

int main(int argc, char *argv[]) {

  std::string format;
  std::ifstream matrixFile;
  std::ifstream solnFile;
  get_args(argc, argv, matrixFile, solnFile, format);


  size_t rows, cols;
  std::vector<matrix_t> soln;
  std::vector<matrix_t> data;

  if(format == "csv"){
    readCSV(solnFile, soln, rows, cols);
    readCSV(matrixFile, data, rows, cols);
  } else if (format == "mtx"){
    readMTX(solnFile, soln, rows, cols);
    readMTX(matrixFile, data, rows, cols);
  } else {
    throw std::runtime_error("File format not recognized.");
  }

#ifdef DEBUG
  printMatrix(data.data(), rows, cols);
#endif

  // Timing objects
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  matrix_t *data_gpu = copy_to_gpu<matrix_t>(data.data(), rows * cols);
  matrix_t *Aij;
  auto_throw(cudaMalloc(&Aij, rows * sizeof(matrix_t)));

  cudaEventRecord(start);

  // Get the number of kernel launches we need to do
  size_t row_kernels = 1;
  if (rows > MAX_BLOCK_SIZE) {
    row_kernels = rows / MAX_BLOCK_SIZE;
    if (rows % MAX_BLOCK_SIZE != 0) {
      row_kernels++;
    }
  }
  size_t col_kernels = 1;
  if (cols > MAX_BLOCK_SIZE) {
    col_kernels = rows / MAX_BLOCK_SIZE;
    if (cols % MAX_BLOCK_SIZE != 0) {
      col_kernels++;
    }
  }
  fprintf(stderr, "Kernel count: (%lu, %lu)\n", row_kernels, col_kernels);

  // Main program flow
  for (size_t j = 0; j < rows; j++) {
    // pivot<<<1, cols>>>(data_gpu, cols, rows, j);
    // auto_throw(cudaDeviceSynchronize());

    for (size_t i = 0; i < row_kernels; i++) {
      size_t count = std::min(rows - i * MAX_BLOCK_SIZE, (size_t)MAX_BLOCK_SIZE);
      storeAij<<<1, count>>>(data_gpu, cols, Aij, j, i * MAX_BLOCK_SIZE);
    }
    auto_throw(cudaDeviceSynchronize());

    for (size_t i = 0; i < col_kernels; i++) {
      size_t count = std::min(cols - i * MAX_BLOCK_SIZE, (size_t)MAX_BLOCK_SIZE);
      fixRow<<<1, count>>>(data_gpu, cols, Aij, j, i * MAX_BLOCK_SIZE);
    }
    auto_throw(cudaDeviceSynchronize());

    for (size_t i = 0; i < col_kernels; i++) {
      size_t col_count = std::min(cols - i * MAX_BLOCK_SIZE, (size_t)MAX_BLOCK_SIZE);
      for (size_t k = 0; k < row_kernels; k++) {
        size_t row_count = std::min(rows - k * MAX_BLOCK_SIZE, (size_t)MAX_BLOCK_SIZE);
        fixColumn<<<col_count, row_count>>>(data_gpu, cols, Aij, j, k * MAX_BLOCK_SIZE, i * MAX_BLOCK_SIZE);
      }
    }
    auto_throw(cudaDeviceSynchronize());
  }

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float msec;
  cudaEventElapsedTime(&msec, start, stop);

  printf("Runtime: %f\n", msec);

  copy_from_gpu<matrix_t>(data.data(), data_gpu, rows * cols);

#ifdef DEBUG
  printMatrix(data.data(), rows, cols);
#endif

  printError(data.data(), soln.data(), rows, cols);

  return 0;
}
