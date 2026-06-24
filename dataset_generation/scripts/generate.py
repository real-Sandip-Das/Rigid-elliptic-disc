import os

import dataset_generation.problemcodeAMDC_opt as opt
from dataset_generation.generate_dataset import generate_dataset

opt._N_THREADS = os.cpu_count()
# Generate the full PINN dataset
# 11x16x20 = 3520 configurations
generate_dataset(
    a0=1.0, a1=2.0, n_a=10,       # a/b: 1.0 to 2.0 (11 points)
    d0=0.1, d1=0.4, n_d=15,       # d/b: 0.1 to 0.4 (16 points)
    k0_val=0.1, k1_val=2.0, n_k=19, # K: 0.1 to 2.0 (20 points)
    filename='full_pinn_dataset.csv'
)
