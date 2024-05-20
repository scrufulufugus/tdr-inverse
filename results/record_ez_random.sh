#!/bin/bash

exec ./bench.sh -i input/ez_random.csv -o out/ez_random.csv -- ../bin/{program} ../matrix_gen/out/{n}x{n}{,_soln}.csv
