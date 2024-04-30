#include "harmonize.cpp"
#include "utils.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace util;

__global__ void storeAij(matrix_t *matrix, int size, matrix_t *Aij, int colId) {
  int rowId = threadIdx.x;
  Aij[rowId] = matrix[size*rowId + colId];
#ifdef DEBUG
  printf("0. A[%d][%d] = %f\n", rowId, colId, Aij[rowId]);
#endif

  if (rowId == colId)
    matrix[size*rowId + colId] = 1.0;
  else
    matrix[size*rowId + colId] = 0.0;
}

//struct Inverse { matrix_t *matrix, size_t size, size_t loc };

struct FixRow;
struct FixCol;

struct FixRow {
  using Type = void(*)(size_t rowId, size_t colId);

  template<typename PROGRAM>
  __device__ static void eval(PROGRAM prog, size_t rowId, size_t colId) {
    size_t size = prog.device.size.row;
    matrix_t Ri  = prog.device.matrix[size*rowId + colId];
    matrix_t Aii = prog.device.Aij[rowId];

#ifdef DEBUG
    printf("1. matrix[%lu][%lu] = %f\n", rowId, colId, Ri);
#endif

    Ri /= Aii;
    prog.device.matrix[size*rowId + colId] = Ri;

#ifdef DEBUG
    printf("2. matrix[%lu][%lu] /= %f = %f\n", rowId, colId, Aii, Ri);
#endif

    for (size_t i = 0; i < prog.device.size.col; i++) {
      matrix_t col = prog.device.Aij[i];
      if (col != 0) {
        prog.template async<FixCol>(rowId, i, colId, col);
      }
    }
  }
};

struct FixCol {
  using Type = void(*)(size_t colId, size_t i, size_t j, matrix_t col);

  template<typename PROGRAM>
  __device__ static void eval(PROGRAM prog, size_t colId, size_t i, size_t j, matrix_t col) {
    size_t size = prog.device.size.row;
    matrix_t colj    = prog.device.matrix[i*size + j];
    matrix_t AColIdj = prog.device.matrix[colId*size + j];
    if (i != colId) {
      colj -= AColIdj * col;

#ifdef DEBUG
      printf("3. matrix[%lu][%lu] -= %f * %f = %f\n", i, j, AColIdj, col, colj);
#endif
    }
    prog.device.matrix[i*size + j] = colj;
  }
};

struct Size2D {
  size_t row;
  size_t col;
};

struct InverseState {
  size_t *j;
  Size2D size;
  matrix_t *matrix;
  matrix_t *Aij;
  iter::AtomicIter<unsigned int>* iterator;
};

struct InverseSpec {
  typedef OpUnion<FixRow,FixCol>       OpSet;
  typedef           InverseState DeviceState;

  static const size_t STASH_SIZE =   16;
  static const size_t FRAME_SIZE = 8191;
  static const size_t  POOL_SIZE = 8191;

  /*
  // Defines the initialization function for programs of type 'ProgType'. This function is called by
  // all threads in all work groups just prior to beginning normal execution. States are accessible
  // through the 'device', 'group', and 'thread' variables, just like when defining async functions.
  //
  // Here, we initialize the work iterator to iterate over a range of integers unique to the work
  // group, distributing the set of integers to process more or less evenly across the set of
  // work groups.
  */
  template<typename PROGRAM>
  __device__ static void initialize(PROGRAM prog){

  }

  /*
  // Defines a function for programs of type 'ProgType' which is called by all threads in all work
  // groups after it is determined that there are no more promises to be processed. To be clear:
  // you should not perform any async calls in here and expect them to be always evaluated, since
  // the program is wrapping up and there is a good chance that no work groups will notice the
  // promises before exiting.
  //
  // Because promises are persistant across execution calls, if you want to queue work for the next
  // execution call, you can check if the current executing work group is the final one to call the
  // finalize function and queue work then. This will guarantee that the queued work will only be
  // evaluated in the next exec call.
  */
  template<typename PROGRAM>
  __device__ static void finalize(PROGRAM prog){

  }

  /*
  // Defines the work making function for programs of type 'ProgType'. This function is called by
  // work groups whenever they notice that they are running out of work to perform. To indicate
  // that there is still more work to perform, return 'true'. To indicate that there is no more
  // work left for the work group to make, return 'false', at which point, the work group will no
  // longer call this function for the remainder of the execution run.
  */
  template<typename PROGRAM>
  __device__ static bool make_work(PROGRAM prog){

      size_t size = prog.device.size.row;

      unsigned int iter_step_length = size;

      iter::Iter<unsigned int> iter = prog.device.iterator->leap(iter_step_length);

      unsigned int index;
      while(iter.step(index)){
        size_t rowId = *(prog.device.j);
        size_t colId = index;
        prog.template async<FixRow>(rowId, colId);
      }

      // TODO: size/2 because aug_matrix
      //if (prog.device.iterator->done() && prog.device.j < (prog.device.size/2)) {
      //  prog.device.iterator->reset(0, prog.device.size);
      //  prog.device.j++;
      //}

      return ! prog.device.iterator->done();

  }

};

typedef  HarmonizeProgram < InverseSpec > ProgType;

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

#ifdef DEBUG
  printMatrix(data.data(), rows, cols);
#endif

  // Timing objects
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  InverseState ds;

  ds.size.row = cols;
  ds.size.col = rows;

  //host::DevBuf<matrix_t> Aij = host::DevBuf<matrix_t>(rows);
  //ds.Aij = Aij;
  cudaMalloc(&(ds.Aij), rows * sizeof(matrix_t));
  host::check_error();

  host::DevBuf<matrix_t> data_gpu = host::DevBuf<matrix_t>(ds.size.row * ds.size.col);
  data_gpu << data;
  // Assign the address of the device-side buffer to the device state so that the program
  // can know where to put its output.
  ds.matrix = data_gpu;

  host::DevBuf<size_t> j = host::DevBuf<size_t>();
  j << 1;
  ds.j = j;

  cudaEventRecord(start);

    iter::AtomicIter<unsigned int> host_iter(0,ds.size.row);
    host::DevBuf<iter::AtomicIter<unsigned int>> iterator;
    iterator << host_iter; // Without this we get an access exception even though we set it later
    ds.iterator = iterator;

    // Declare and instance of type 'ProgType' with an arena size of 2^(20) with a device state
    // initialized to the value of our declared device state struct. The arena size of a
    // program determines how much extra space it has to store work if it cannot store
    // everything inside shared memory. If you are *certain* that no work will spill into
    // main memory, you may get some performance benefits by seting the arena size to zero.
    ProgType::Instance instance = ProgType::Instance(0x10000,ds);
    cudaDeviceSynchronize();
    host::check_error();

    // Initialize the instance using 32 work groups
    init<ProgType>(instance,32);
    cudaDeviceSynchronize();
    host::check_error();

  for (size_t cj = 0; cj < ds.size.col; cj++) {
    j << cj; // Push current row to gpu

    // Reset iter
    iter::AtomicIter<unsigned int> host_iter(0,ds.size.row);
    iterator << host_iter;

    storeAij<<<1, rows>>>(ds.matrix, ds.size.row, ds.Aij, cj);
    cudaDeviceSynchronize();
    host::check_error();

    // Execute the instance using 240 work groups, with each work group performing up to
    // 65536 promise executions per thread before halting. If all promises are exhausted
    // before this, the program exits early.
    exec<ProgType>(instance,240,65536);
    cudaDeviceSynchronize();
    host::check_error();
  }

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float msec;
  cudaEventElapsedTime(&msec, start, stop);

  printf("Runtime: %f\n", msec);


  data.clear();
  data_gpu >> data;

#ifdef DEBUG
  printMatrix(data.data(), rows, cols);
#endif

  printError(data.data(), soln.data(), rows, cols);

  return 0;
}
