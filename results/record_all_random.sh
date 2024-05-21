#!/bin/bash

exec ./bench.sh -i input/all_random.csv -i input/programs.csv -o out/all_random.csv -- ../bin/{program} ../matrix_gen/out/{n}x{n}{,_soln}.csv
