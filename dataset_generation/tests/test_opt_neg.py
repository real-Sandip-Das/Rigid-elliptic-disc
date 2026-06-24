import sys
import dataset_generation.problemcodeAMDC_opt as opt
import numpy as np

res_opt, _ = opt.problemcodeAMDC(5, 0.1, 0.42675, 1.0, 1.0)
print(f"Opt Added Mass: {res_opt.real * np.pi}")
