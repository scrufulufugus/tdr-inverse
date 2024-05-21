#!/bin/bash

exec ./bench.sh -i input/all_sparse.csv -i input/programs.csv -o out/all_sparse.csv -r n int '^Read a (?P<n>\d+)' -- ../bin/{program} ../matrix_gen/out/{mat}{,_soln}.mtx
