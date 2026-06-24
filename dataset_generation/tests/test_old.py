import sys
import dataset_generation.problemcodeAMDC as old
import dataset_generation.problemcodeAMDC_opt as opt
import numpy as np

print("Running OLD:")
res_old = old.problemcodeAMDC(5, 0.1, 0.5, 1.0, 1.0)
print(f"Old final: {res_old}")

print("Running OPT:")
res_opt, _ = opt.problemcodeAMDC(5, 0.1, 0.5, 1.0, 1.0)
print(f"Opt final: {res_opt}")
