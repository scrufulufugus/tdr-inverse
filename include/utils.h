#include <fstream>
#include <sstream>
#include <vector>

#ifndef UTILS_H_
#define UTILS_H_

typedef float matrix_t;

void bad_usage(char *exec);

void get_args(int argc, char *argv[], std::ifstream &matrixFile, std::ifstream &solnFile);

__host__ void readCSV(std::istream &file, std::vector<matrix_t> &data, size_t &rows, size_t &cols);

__host__ void matrixToAug(const std::vector<matrix_t> &data, std::vector<matrix_t> &aug, const size_t &rows, const size_t &cols);

__host__ void augToMatrix(std::vector<matrix_t> &data, const std::vector<matrix_t> &aug, const size_t &rows, const size_t &cols);

__host__ __device__ void printMatrix(matrix_t *matrix, const size_t &rows, const size_t cols);

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

#endif // UTILS_H_
