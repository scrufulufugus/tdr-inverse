#!/bin/bash

exec ../submodules/bench/bench.py -r time float '^Runtime\s*:\s*(?P<time>\d*\.?\d*)$' -r mae float '^Mean Absolute Error\s*:\s*(?P<mae>\S+)$' "$@"
