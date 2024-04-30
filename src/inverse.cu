#include "utils.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace helpers;

#define MAX_BLOCK_SIZE 1024

__global__ void pivot(matrix_t *matrix, int cols, int rows, int j) {
  // the ith row of the matrix
  __shared__ matrix_t Ri[MAX_BLOCK_SIZE];
  int colId = threadIdx.x;
  Ri[colId] = matrix[cols * j + colId];

  // Pivot
  int swapRow = j;
  for (int i = j; i < rows; i++) {
    if (abs(matrix[cols*i + j]) > abs(matrix[cols*swapRow + j])) {
      swapRow = i;
    }
  }

  Ri[colId] = matrix[cols * swapRow + colId];
  if (swapRow != j) {
    __syncthreads();
#ifdef DEBUG
    printf("1. swap(M[%d][%d], M[%d][%d]) = swap(%f, %f)\n", j, colId, swapRow, colId,
           matrix[cols * j + colId], matrix[cols * swapRow + colId]);
#endif
    matrix[cols * swapRow + colId] = matrix[cols * j + colId];
  }
}

__global__ void storeAij(matrix_t *matrix, int size, matrix_t *Aij, int colId) {
  int rowId = threadIdx.x;
  Aij[rowId] = matrix[size*rowId + colId];
}

// (c) Sharma 2013
__global__ void fixRow(matrix_t *matrix, int size, matrix_t *Aij, int rowId) {
  // the ith row of the matrix
  __shared__ matrix_t Ri[MAX_BLOCK_SIZE];
  // The diagonal element for ith row
  __shared__ matrix_t Aii;
  int colId = threadIdx.x;
  Ri[colId] = matrix[size * rowId + colId];
  Aii = Aij[rowId];

#ifdef DEBUG
  printf("1. matrix[%d][%d] = %f\n", rowId, colId, Ri[colId]);
#endif
  __syncthreads();
  // Divide the whole row by the diagonal element making sure it is not 0
  Ri[colId] = Ri[colId] / Aii;
  matrix[size * rowId + colId] = Ri[colId];
#ifdef DEBUG
  printf("2. matrix[%d][%d] /= %f = %f\n", rowId, colId, Aii, Ri[colId]);
#endif
}

// (c) Sharma 2013
__global__ void fixColumn(matrix_t *matrix, int size, matrix_t *Aij, int colId) {
  int i = threadIdx.x;
  int j = blockIdx.x;
  // The colId column
  __shared__ matrix_t col[MAX_BLOCK_SIZE];
  // The jth element of the colId row
  __shared__ matrix_t AColIdj;
  // The jth column
  __shared__ matrix_t colj[MAX_BLOCK_SIZE];
  col[i] = Aij[i];
  __syncthreads();
  if (col[i] != 0) {
    colj[i] = matrix[i * size + j];
    AColIdj = matrix[colId * size + j];
    if (i != colId) {
#if   PRECISION == 1
      colj[i] = fmaf(-1. * AColIdj, col[i], colj[i]);
#elif PRECISION == 2
      colj[i] = fma(-1. * AColIdj, col[i], colj[i]);
#endif
      //colj[i] = colj[i] - AColIdj * col[i];
#ifdef DEBUG
      printf("3. matrix[%d][%d] -= %f * %f = %f\n", i, j, AColIdj, col[i],
             colj[i]);
#endif
    }
    matrix[i * size + j] = colj[i];
  }
}

int main(int argc, char *argv[]) {

  std::ifstream matrixFile;
  std::ifstream solnFile;
  get_args(argc, argv, matrixFile, solnFile);

  size_t rows, cols;
  std::vector<matrix_t> soln;
  readCSV(solnFile, soln, rows, cols);

  std::vector<matrix_t> data;
  readCSV(matrixFile, data, rows, cols);

#ifdef DEBUG
  printMatrix(data.data(), rows, cols);
#endif

  // Timing objects
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Convert matrix to augmented form
  std::vector<matrix_t> aug;
  size_t aug_cols = 2 * cols;
  matrixToAug(data, aug, rows, cols);

  matrix_t *data_gpu = copy_to_gpu<matrix_t>(aug.data(), rows * aug_cols);
  matrix_t *Aij;
  auto_throw(cudaMalloc(&Aij, rows * sizeof(matrix_t)));

  cudaEventRecord(start);

  // Main program flow
  for (size_t j = 0; j < rows; j++) {
    pivot<<<1, aug_cols>>>(data_gpu, aug_cols, rows, j);
    auto_throw(cudaDeviceSynchronize());

    storeAij<<<1, rows>>>(data_gpu, aug_cols, Aij, j);
    auto_throw(cudaDeviceSynchronize());

    fixRow<<<1, aug_cols>>>(data_gpu, aug_cols, Aij, j);
    auto_throw(cudaDeviceSynchronize());

    fixColumn<<<aug_cols, rows>>>(data_gpu, aug_cols, Aij, j);
    auto_throw(cudaDeviceSynchronize());
  }

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float msec;
  cudaEventElapsedTime(&msec, start, stop);

  printf("Runtime: %f\n", msec);

  copy_from_gpu<matrix_t>(aug.data(), data_gpu, rows * aug_cols);

  // Convert matrix from augmented form
  data.clear();
  augToMatrix(data, aug, rows, cols);

#ifdef DEBUG
  printMatrix(data.data(), rows, cols);
#endif

  printError(data.data(), soln.data(), rows, cols);

  return 0;
}
