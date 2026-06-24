import problemcodeAMDC_opt as opt
import numpy as np
res = opt.problemcodeAMDC(5, 0.1, 0.14597, 1.0, 1.0)
print(f"{res.real * np.pi=}, {res.imag * np.pi=}")
