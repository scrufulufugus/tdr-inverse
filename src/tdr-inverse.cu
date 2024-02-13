#include "harmonize.cpp"

using namespace util;

int main(int argc, char *argv[]) {

  cli::ArgSet args(argc, argv);

  Stopwatch watch;
  watch.start();

  // Main program flow

  watch.stop();

  float msec = watch.ms_duration();

  printf("Runtime: %f\n", msec);

  return 0;
}
