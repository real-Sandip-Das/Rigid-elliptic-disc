import sys
import os
import pandas as pd
import numpy as np

import dataset_generation.problemcodeAMDC as old

def main():
    combinations = [
        (1.0, 0.2, 0.5),
        (1.0, 0.2, 2.0),
        (1.75, 0.4, 0.5),
        (1.75, 0.4, 2.0)
    ]
    N = 5
    b = 1.0
    
    csv_file = os.path.join(os.path.dirname(__file__), 'reference_dataset.csv')
    rows = []
    
    for a, d, K in combinations:
        print(f"Running OLD for a={a}, d={d}, K={K}")
        final = old.problemcodeAMDC(N, d, K, a, b)
        am = np.real(np.pi * final * a)
        damp = np.imag(np.pi * final * a)
        rows.append([a, d, K, am, damp])
        
    df = pd.DataFrame(rows, columns=['a_b', 'd_b', 'wave_frequency_K', 'Added_Mass', 'Damping_Coefficient'])
    df.to_csv(csv_file, index=False)

if __name__ == '__main__':
    main()
