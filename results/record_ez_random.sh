#!/bin/bash

exec ../submodules/bench/bench.py -i input/ez_random.csv -o out/ez_random.csv -r time float '^Runtime\s*:\s*(?P<time>\d*\.?\d*)$' -r mae float '^Mean Absolute Error\s*:\s*(?P<mae>\S+)$' -- ../bin/{program} ../matrix_gen/out/{n}x{n}{,_soln}.csv
