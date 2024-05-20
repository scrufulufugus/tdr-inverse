# Results

Benchmark analysis and visualizations

## Set Up

A [conda](https://docs.anaconda.com/free/anaconda/install/index.html) environment file is included to bootstrap the dependencies of programs in this directory. With conda installed run the following to create and activate the environment:

```sh
conda env create -f environment.yml
conda activate tdr_inverse_results
```

## Usage

### Benchmarks

The following benchmarks can be run by calling them from this directory (e.g. `./example.sh`).

- `record_all_random.sh`: Benchmarks `inverse`, `cpu-inverse`, and `tdr-inverse` with all matrices from `matrix_gen.py`
- `record_ez_random.sh`: Benchmarks `inverse`, `cpu-inverse`, and `tdr-inverse` with the smaller matrices from `matrix_gen.py`