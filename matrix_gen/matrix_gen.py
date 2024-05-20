#!/usr/bin/env python3

import os
import numpy as np
from scipy import linalg

SAVE_DIR = "./out"

# Avoid showing really small numbers
np.set_printoptions(precision=7, suppress=True)

# Sizes of matrices to generate
sizes = [ 3, 4, 6, 10, 50, 100, 150, 200, 256, 512, 1024, 2000, 2048, 3000, 4096 ]
N = 100 # Max element value
np.random.seed(57) # Start with a seed

# Generate matrices
# Random ints from 0 to N
#matrices = [ np.random.randint(N, size=(i, i)) for i in sizes ]
# Random from standard distribution mu = 0 and sigma = N/2
matrices = [ (N/2) * np.random.randn(i,i) for i in sizes ]

# Genenate inverse
inverses = [ linalg.inv(m) for m in matrices ]

# Confirm that we have true inverses (within a tolerance)
for i, m in enumerate(matrices):
    print(sizes[i])
    print(m.dot(inverses[i]))

os.makedirs(SAVE_DIR, exist_ok=True)
# Save matrices
for i in range(len(matrices)):
    np.savetxt(SAVE_DIR + "/{0}x{0}.csv".format(sizes[i]), matrices[i], delimiter=",")
    np.savetxt(SAVE_DIR + "/{0}x{0}_soln.csv".format(sizes[i]), inverses[i], delimiter=",")
    print("Saved {0}x{0}".format(sizes[i]))
