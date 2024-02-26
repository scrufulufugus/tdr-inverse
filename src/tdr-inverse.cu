#include "harmonize.cpp"
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using namespace util;

typedef float matrix_t;

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

int main(int argc, char *argv[]) {

  cli::ArgSet args(argc, argv);

  size_t rows, cols;

  std::string filename = "./tests/3x3.csv";
  std::ifstream file(filename);
  std::vector<matrix_t> data;
  if (!file.is_open()) {
      printf("Error opening %s\n", filename);
      return 1;
  }
  readCSV(file, data, rows, cols);

  for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
          printf("%f, ", data[j + cols*i]);
      }
      printf("\n");
  }

  Stopwatch watch;
  watch.start();

  // Main program flow

  watch.stop();

  float msec = watch.ms_duration();

  printf("Runtime: %f\n", msec);

  return 0;
}
