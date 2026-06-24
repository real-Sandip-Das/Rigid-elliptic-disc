import sys
import os
import csv
import numpy as np

import dataset_generation.problemcodeAMDC_opt as opt

def main():
    csv_file = os.path.join(os.path.dirname(__file__), 'reference_dataset.csv')
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} does not exist.")
        sys.exit(1)
        
    N = 5
    b = 1.0
    
    success = True
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if not row:
                continue
            a = float(row[0])
            d = float(row[1])
            K = float(row[2])
            ref_am = float(row[3])
            ref_damp = float(row[4])
            
            print(f"Testing Opt implementation against reference for a={a}, d={d}, K={K}")
            
            final_opt, _ = opt.problemcodeAMDC(N, d, K, a, b)
            opt_am = np.real(np.pi * final_opt * a)
            opt_damp = np.imag(np.pi * final_opt * a)
            
            print(f"Ref: AM={ref_am:.6f}, Damp={ref_damp:.6f}")
            print(f"Opt: AM={opt_am:.6f}, Damp={opt_damp:.6f}")
            
            if not np.isclose(ref_am, opt_am, atol=1e-5):
                print(f"  -> Added mass mismatch! Expected {ref_am}, got {opt_am}")
                success = False
            if not np.isclose(ref_damp, opt_damp, atol=1e-5):
                print(f"  -> Damping mismatch! Expected {ref_damp}, got {opt_damp}")
                success = False
            print("-" * 40)
            
    if success:
        print("All Opt implementation tests matched the reference data successfully!")
    else:
        print("Some tests failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()
