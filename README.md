# GPU Matrix Inverse

A collection of three programs developed to demonstrate the usefulness of [Thread-Data Remapping](https://dl.acm.org/doi/10.1145/3242089) for inverting a matrix.

## Usage

``` sh
cpu-inverse [input matrix] [inverse matrix]
inverse     [input matrix] [inverse matrix]
tdr-inverse [input matrix] [inverse matrix]
```

Matrices may either be in comma separated CSV format or [Matrix Market Coordinate Format](https://math.nist.gov/MatrixMarket/formats.html#MMformat).

## About

The code in developed in service of a masters research project. The related documents [can be found here](https://github.com/scrufulufugus/tdr-inverse-materials).

### Contents

There are three programs included in this repository:

- `cpu-inverse`: Inverts a matrix on-CPU.
- `inverse`: Inverts a matrix on-GPU utilizing a standard CUDA approach.
- `tdr-inverse`: Inverts a matrix on-GPU utilizing an asynchronous approach, see [Harmonize](https://github.com/CEMeNT-PSAAP/harmonize).

In addition, a few subdirectories contain tools to aid in comparing each implementation:

- `matrix_gen/`: Python scripts for generating matrices and generating inverses for matrices in Matrix Market format.
- `results/`: Scripts for running benchmarks against all implementations and interpreting results.
- `tests/`: Some basic matrices to confirm programs are working as intended.

## Getting Started

This repository contains submodules, please checkout recursively:

```sh
git clone --recursive https://github.com/scrufulufugus/tdr-inverse.git
```

### Dependencies

The only hard dependencies outside of submodules are [CUDA](https://developer.nvidia.com/cuda-toolkit) and [gnumake](https://www.gnu.org/software/make/). Both should be installed though the correct repositories for your distribution of choice.

The `matrix_gen/` and `results/` subdirectories additionally require [anaconda](https://www.anaconda.com/download/success) or a python environment with the dependencies detailed in their respective environment files. See the README in each subdirectory for more information.

### Building

All three inverse programs can be built by running:

``` sh
make
```

### Running

Once built, the binaries can be found in `bin/`. Each program takes two arguments: an input matrix and its inverse. The program will then perform an inverse and return the error from the second argument.

``` sh
./bin/tdr-inverse ./tests/3x3.csv ./tests/3x3_soln.csv
```
