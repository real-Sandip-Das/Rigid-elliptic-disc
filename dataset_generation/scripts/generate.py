import os
import sys

# Ensure we can import dataset_generation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import dataset_generation.problemcodeAMDC_opt as opt
from dataset_generation.generate_dataset import generate_dataset

def main():
    # Set number of threads appropriately
    opt._N_THREADS = max(1, os.cpu_count() // 2)
    
    # Generate the full PINN dataset
    # 11x16x20 = 3520 configurations
    generate_dataset(
        a0=1.0, a1=2.0, n_a=10,       # a/b: 1.0 to 2.0 (11 points)
        d0=0.1, d1=0.4, n_d=15,       # d/b: 0.1 to 0.4 (16 points)
        k0_val=0.1, k1_val=2.0, n_k=19, # K: 0.1 to 2.0 (20 points)
        filename='full_pinn_dataset.csv'
    )

if __name__ == '__main__':
    main()
