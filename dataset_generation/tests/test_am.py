import numpy as np
import dataset_generation.problemcodeAMDC_opt as opt
N = 5
d_val = 0.1
K = 0.19606
a = 1.0
b = 1.0
res = opt.problemcodeAMDC(N, d_val, K, a, b)
print("Result final:", res)
