#include "utils.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace helpers;

const int MAX_BLOCK_SIZE = 1024;

// (c) Sharma 2013
__global__ void fixRow(matrix_t *matrix, int size, int rowId) {
  // the ith row of the matrix
  __shared__ matrix_t Ri[MAX_BLOCK_SIZE];
  // The diagonal element for ith row
  __shared__ matrix_t Aii;
  int colId = threadIdx.x;
  Ri[colId] = matrix[size * rowId + colId];
  Aii = matrix[size * rowId + rowId];

  // Pivot if row is singular
  int i = 0;
  while (Aii == 0.) {
    if (matrix[size * i + rowId] != 0) {
#ifdef DEBUG
      printf("1. matrix[%d][%d] += matrix[%d][%d]\n", rowId, colId, i, colId);
#endif
      Ri[colId] += matrix[size * i + colId];
      Aii = Ri[rowId];
      break;
    }
    i++;
  }
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
__global__ void fixColumn(matrix_t *matrix, int size, int colId) {
  int i = threadIdx.x;
  int j = blockIdx.x;
  // The colId column
  __shared__ matrix_t col[MAX_BLOCK_SIZE];
  // The jth element of the colId row
  __shared__ matrix_t AColIdj;
  // The jth column
  __shared__ matrix_t colj[MAX_BLOCK_SIZE];
  col[i] = matrix[i * size + colId];
  if (col[i] != 0) {
    colj[i] = matrix[i * size + j];
    AColIdj = matrix[colId * size + j];
    if (i != colId) {
      colj[i] = colj[i] - AColIdj * col[i];
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

  cudaEventRecord(start);

  // Main program flow
  for (size_t j = 0; j < rows; j++) {
    fixRow<<<1, aug_cols>>>(data_gpu, aug_cols, j);
    auto_throw(cudaDeviceSynchronize());

    fixColumn<<<aug_cols, rows>>>(data_gpu, aug_cols, j);
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

  long double error;
  long double mae;
  for (size_t i = 0; i < soln.size(); i++) {
    error = std::abs(data[i] - soln[i]) / std::max(std::abs(soln[i]), std::abs(data[i]));
    mae += std::abs(data[i] - soln[i]);
    if (!std::isfinite(data[i]) || error > 0.0001) {
      fprintf(stderr, "matrix[%zu][%zu] expected % E got % E Error: %LE\n", i / cols,
              i % cols, soln[i], data[i], error);
    }
  }
  mae /= soln.size();
  fprintf(stderr, "Mean Absolute Error: %LE\n", mae);
  // printMatrix(data.data(), rows, cols);

  return 0;
}
