#include <fstream>
#include <sstream>
#include <vector>
#include <cfloat>

#ifndef PRECISION
#define PRECISION 2
#endif

#ifndef UTILS_H_
#define UTILS_H_

#if   PRECISION == 1
typedef float matrix_t;
#define SIGFIGS FLT_DECIMAL_DIG
#elif PRECISION == 2
typedef double matrix_t;
#define SIGFIGS DBL_DECIMAL_DIG
#endif

void bad_usage(char *exec);

void get_args(int argc, char *argv[], std::ifstream &matrixFile, std::ifstream &solnFile);

void readCSV(std::istream &file, std::vector<matrix_t> &data, size_t &rows, size_t &cols);

void matrixToAug(const std::vector<matrix_t> &data, std::vector<matrix_t> &aug, const size_t &rows, const size_t &cols);

void augToMatrix(std::vector<matrix_t> &data, const std::vector<matrix_t> &aug, const size_t &rows, const size_t &cols);

void printMatrix(matrix_t *matrix, size_t rows, size_t cols);

void printError(matrix_t *matrix, matrix_t *soln, size_t rows, size_t cols);

#if defined(__CUDACC__) || defined(__NVCC__)
namespace helpers {
  void auto_throw(cudaError_t value);

  // desc: Allocates a buffer on gpu and copies cpu buffer to it
  template <typename T> T *copy_to_gpu(T *data, size_t size) {
    T *gpu_array;
    auto_throw(cudaMalloc(&gpu_array, size * sizeof(T)));

    auto_throw(
        cudaMemcpy(gpu_array, data, size * sizeof(T), cudaMemcpyHostToDevice));
    auto_throw(cudaDeviceSynchronize());

    return gpu_array;
  }

  // desc: Copies gpu buffer to cpu
  template <typename T>
  void copy_from_gpu(T *cpu_array, T *gpu_array, size_t size) {
    auto_throw(cudaMemcpy(cpu_array, gpu_array, size * sizeof(T),
                          cudaMemcpyDeviceToHost));
    auto_throw(cudaDeviceSynchronize());
  }
};
#endif

#endif // UTILS_H_
