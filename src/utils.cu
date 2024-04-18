#include "utils.h"

void helpers::auto_throw(cudaError_t value) {
  if (value != cudaSuccess) { throw value; }
}

std::string const USAGE = "matrixfile [solnfile]";

// desc: Reports incorrect command-line usage and exits with status 1
//  pre: None
// post: In description
void bad_usage(char *exec) {
  fprintf(stderr, "Usage: %s %s\n", exec, USAGE.c_str());
  std::exit(1);
}

// desc: Returns the value and thread counts provided by the supplied
//       command-line arguments.
//  pre: There should be exactly two arguments, both positive integers.
//       If this precondition is not met, the program will exit with
//       status 1
// post: In description
void get_args(int argc, char *argv[], std::ifstream &matrixFile,
              std::ifstream &solnFile) {
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
__host__ void readCSV(std::istream &file, std::vector<matrix_t> &data,
                      size_t &rows, size_t &cols) {
  rows = 0;
  cols = 0;
  std::string line;

  while (getline(file, line)) {
    std::stringstream line_s(line);
    std::string element;
    cols = 0;
    while (getline(line_s, element, ',')) {
      data.push_back(std::stof(element));
      cols++;
    }
    rows++;
  }
  printf("Read a %lu x %lu matrix\n", rows, cols);
}

// Takes a matrix and outputs an augmented form
__host__ void matrixToAug(const std::vector<matrix_t> &data,
                          std::vector<matrix_t> &aug, const size_t &rows,
                          const size_t &cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      aug.push_back(data[i * cols + j]);
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

__host__ void augToMatrix(std::vector<matrix_t> &data,
                          const std::vector<matrix_t> &aug, const size_t &rows,
                          const size_t &cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      data.push_back(aug[i * 2 * cols + j + cols]);
    }
  }
}

__host__ __device__ void printMatrix(matrix_t *matrix, size_t rows,
                                     size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      printf("% *E,", SIGFIGS, matrix[i*cols + j]);
    }
    printf("\n");
  }
}
