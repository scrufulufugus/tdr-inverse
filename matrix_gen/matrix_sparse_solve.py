#!/usr/bin/env python3

from scipy.io import mmread, mmwrite
from scipy.sparse import linalg
import os
import numpy as np

SAVE_DIR  = "./out"
INPUT_DIR = "./input"

for path in os.listdir(INPUT_DIR):
    path = INPUT_DIR+path
    if not str(path).endswith(".mtx"):
        continue
    print(path)
    mat = mmread(path)
    inv = linalg.inv(mat)
    base = os.path.splitext(os.path.basename(path))[0]
    mmwrite(open(SAVE_DIR + f"/{base}.mtx","w+b"), mat)
    mmwrite(open(SAVE_DIR + f"/{base}_soln.mtx","w+b"), inv)
    print(f"Saved {base}")

