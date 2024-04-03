#include "harmonize.cpp"
#include "util/host.cpp"
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using namespace util;

typedef float matrix_t;
const int MAX_BLOCK_SIZE = 1024;
std::string const USAGE = "matrixfile [solnfile]";

// desc: Reports incorrect command-line usage and exits with status 1
//  pre: None
// post: In description
void bad_usage(char* exec) {
    fprintf(stderr, "Usage: %s %s\n", exec, USAGE.c_str());
    std::exit(1);
}

// desc: Returns the value and thread counts provided by the supplied
//       command-line arguments.
//  pre: There should be exactly two arguments, both positive integers.
//       If this precondition is not met, the program will exit with
//       status 1
// post: In description
void get_args(int argc, char *argv[], std::ifstream &matrixFile, std::ifstream &solnFile) {
    if (argc <= 2) {
        fprintf(stderr, "Too few args\n");
        bad_usage(argv[0]);
    } else if (argc > 3) {
        fprintf(stderr, "Too many args\n");
        bad_usage(argv[0]);
    }

    std::string matrix_filename(argv[1]);
    // TODO: Allow only one file and guess others name
    std::string soln_filename(argv[2]);

    if (matrix_filename.compare(soln_filename) == 0) {
        bad_usage(argv[0]);
    }

    matrixFile.open(matrix_filename);
    if (!matrixFile) {
        fprintf(stderr, "Cannot read file: %s\n", matrix_filename.c_str());
        bad_usage(argv[0]);
    }
    solnFile.open(soln_filename);
    if (!solnFile) {
        fprintf(stderr, "Cannot read file: %s\n", soln_filename.c_str());
        bad_usage(argv[0]);
    }
}


// Read a comma seperated CSV into memory.
__host__ void readCSV(std::istream &file, std::vector<matrix_t> &data, size_t &rows, size_t &cols) {
    rows = 0;
    cols = 0;
    std::string line;

    while(getline(file, line)) {
        std::stringstream line_s(line);
        std::string element;
        cols = 0;
        while(getline(line_s, element, ',')) {
            data.push_back(std::stof(element));
            cols++;
        }
        rows++;
    }
    printf("Read a %d x %d matrix\n", rows, cols);
}

// Takes a matrix and outputs an augmented form
__host__ void matrixToAug(const std::vector<matrix_t> &data, std::vector<matrix_t> &aug, const size_t &rows, const size_t &cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      aug.push_back(data[i*cols + j]);
    }
    for (size_t j = 0; j < cols; j++) {
      if (i == j) {
        aug.push_back(1);
      } else {
        aug.push_back(0);
      }
    }
  }
}

__host__ void augToMatrix(std::vector<matrix_t> &data, const std::vector<matrix_t> &aug, const size_t &rows, const size_t &cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      data.push_back(aug[i*2*cols + j+cols]);
    }
  }
}

__host__ __device__ void printMatrix(matrix_t *matrix, const size_t &rows, const size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      printf("% 7f  ", matrix[j + cols*i]);
    }
    printf("\n");
  }
}

// desc: Allocates a buffer on gpu and copies cpu buffer to it
template <typename T> T* copy_to_gpu(T *data, size_t size) {
  T *gpu_array;
  auto_throw(cudaMalloc(&gpu_array,size*sizeof(T)));

  auto_throw(cudaMemcpy(
    gpu_array,
    data,
    size*sizeof(T),
    cudaMemcpyHostToDevice
  ));
  auto_throw(cudaDeviceSynchronize());

  return gpu_array;
}

// desc: Copies gpu buffer to cpu
template <typename T> void copy_from_gpu(T *cpu_array, T *gpu_array, size_t size) {
  auto_throw(cudaMemcpy(
    cpu_array,
    gpu_array,
    size*sizeof(T),
    cudaMemcpyDeviceToHost
  ));
  auto_throw(cudaDeviceSynchronize());
}

// (c) Sharma 2013
__global__ void fixRow(matrix_t *matrix, int size, int rowId) {
  // the ith row of the matrix
  __shared__ matrix_t Ri[MAX_BLOCK_SIZE];
  // The diagonal element for ith row
  __shared__ matrix_t Aii;
  int colId = threadIdx.x;
  Ri[colId] = matrix[size * rowId + colId];
  Aii = matrix[size * rowId + rowId]; // TODO: If Aii is zero we need to add a row first
  printf("1. matrix[%d][%d] = %f\n", rowId, colId, Ri[colId]);
  __syncthreads();
  // Divide the whole row by the diagonal element making sure it is not 0
  Ri[colId] = Ri[colId] / Aii;
  matrix[size * rowId + colId] = Ri[colId];
  printf("2. matrix[%d][%d] /= %f = %f\n", rowId, colId, Aii, Ri[colId]);
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
  if(col[i] != 0) {
    colj[i] = matrix[i * size + j];
    AColIdj = matrix[colId * size + j];
    if (i != colId) {
      colj[i] = colj[i] - AColIdj * col[i];
      printf("3. matrix[%d][%d] -= %f * %f = %f\n", i, j, AColIdj, col[i], colj[i]);
    }
    matrix[i * size + j] = colj[i];
  }
}

int main(int argc, char *argv[]) {

  cli::ArgSet args(argc, argv);

  std::ifstream matrixFile;
  std::ifstream solnFile;
  get_args(argc, argv, matrixFile, solnFile);

  size_t rows, cols;
  std::vector<matrix_t> soln;
  readCSV(solnFile, soln, rows, cols);

  std::vector<matrix_t> data;
  readCSV(matrixFile, data, rows, cols);

  printMatrix(data.data(), rows, cols);

  // Convert matrix to augmented form
  std::vector<matrix_t> aug;
  size_t aug_cols = 2*cols;
  matrixToAug(data, aug, rows, cols);

  matrix_t *data_gpu = copy_to_gpu<matrix_t>(aug.data(), rows*aug_cols);

  Stopwatch watch;
  watch.start();

  // Main program flow
  for (size_t j = 0; j < rows; j++) {
    fixRow<<<1, aug_cols>>>(data_gpu, aug_cols, j);
    auto_throw(cudaDeviceSynchronize());

    fixColumn<<<aug_cols, rows>>>(data_gpu, aug_cols, j);
    auto_throw(cudaDeviceSynchronize());
  }

  watch.stop();

  float msec = watch.ms_duration();

  printf("Runtime: %f\n", msec);

  copy_from_gpu<matrix_t>(aug.data(), data_gpu, rows*aug_cols);

  // Convert matrix from augmented form
  data.clear();
  augToMatrix(data, aug, rows, cols);

  for (size_t i = 0; i < soln.size(); i++) {
    if (std::abs(data[i] - soln[i]) > 0.0000001) {
      fprintf(stderr, "matrix[%zu][%zu] expected % 7f got % 7f\n", i / cols, i % cols, soln[i], data[i]);
    }
  }
  //printMatrix(data.data(), rows, cols);

  return 0;
}
