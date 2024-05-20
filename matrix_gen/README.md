# Matrix Generators & Parsers

Tools to generate and parse matrices.

## Set Up

A [conda](https://docs.anaconda.com/free/anaconda/install/index.html) environment file is included to bootstrap the dependencies of programs in this directory. With conda installed run the following to create and activate the environment:

```sh
conda env create -f environment.yml
conda activate matrix_gen
```

## Usage

### matrix_gen.py

`matrix_gen` will generate a set of random matrices and their inverses.

```sh
python3 matrix_gen.py
```

### matrix_sparse_solve.py

`matrix_sparse_solve` will look in the `./input` directory for matrices in the [Matrix Market Format](https://math.nist.gov/MatrixMarket/formats.html#MMformat) and will output both the matrices and their inverses. Output matrices are kept in the MM format since it is far more efficient for sparse matrices.

```sh
python3 matrix_sparse_solve.py
```