#include "utils.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;
using TimeSpan = std::chrono::duration<double>;

void fixRows(matrix_t *matrix, size_t size, size_t rowId) {
#ifdef DEBUG
  for (size_t i = 0; i < size; i++) {
    printf("1. M[%lu][%lu] = %f\n", rowId, i, matrix[size * rowId + i]);
  }
#endif

  matrix_t Aii = matrix[size * rowId + rowId];
  for (size_t i = 0; i < size; i++) {
    matrix[size * rowId + i] /= Aii;
#ifdef DEBUG
    printf("2. matrix[%lu][%lu] /= %f = %f\n", rowId, i, Aii, matrix[size * rowId + i]);
#endif
  }
}

void fixColumns(matrix_t *matrix, size_t cols, size_t rows, size_t colId) {
  for (size_t i = 0; i < rows; i++) {
    matrix_t Aii = matrix[cols * i + colId];
    for (size_t r = 0; r < cols; r++) {
      if (Aii != 0.0 && i != colId) {
#ifdef DEBUG
        matrix_t Air = matrix[cols * i + r];
#endif
        matrix[cols * i + r] -= matrix[cols * colId + r] * Aii;
#ifdef DEBUG
        printf("3. M[%lu][%lu] = M[%lu][%lu] - M[%lu][%lu] * M[%lu][%lu] = %f - %f * %f = %f\n",
               i, r, i, r, colId, r, i, colId,
               Air, matrix[cols * colId + r], matrix[cols * i + colId], matrix[cols * i + r]);
#endif
      }
    }
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

  // Convert matrix to augmented form
  std::vector<matrix_t> aug;
  size_t aug_cols = 2 * cols;
  matrixToAug(data, aug, rows, cols);

  TimePoint start_time = steady_clock::now();

  // Main program flow
  for (size_t j = 0; j < rows; j++) {
    // TODO: Pivot

    fixRows(aug.data(), aug_cols, j);

    fixColumns(aug.data(), aug_cols, rows, j);
  }

  TimePoint end_time = steady_clock::now();
  TimeSpan span = duration_cast<TimeSpan>(end_time - start_time);
  float msec = span.count();

  printf("Runtime: %f\n", msec);

  // Convert matrix from augmented form
  data.clear();
  augToMatrix(data, aug, rows, cols);

#ifdef DEBUG
  printMatrix(data.data(), rows, cols);
#endif

  printError(data.data(), soln.data(), rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      printf("%.*E,", SIGFIGS, data[i*cols + j]);
    }
    printf("\n");
  }

  return 0;
}
