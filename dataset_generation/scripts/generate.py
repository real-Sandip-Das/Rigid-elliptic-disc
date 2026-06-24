import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure we can import dataset_generation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import dataset_generation.problemcodeAMDC_opt as opt
from dataset_generation.generate_dataset import generate_symmetric_points, evaluate_phi

def generate_full_dataset():
    N = 5 # Truncation order
    b = 1.0 # Keep semi-minor axis at 1 for non-dimensionalization
    
    # Define a much finer grid for a full PINN dataset
    a0, a1, n_a = 1.0, 2.0, 10      # 11 points (1.0, 1.1, ..., 2.0)
    d0, d1, n_d = 0.1, 0.4, 15      # 16 points
    k0_val, k1_val, n_k = 0.1, 2.0, 19 # 20 points
    
    a_vals = np.linspace(a0, a1, n_a + 1)
    d_vals = np.linspace(d0, d1, n_d + 1)
    K_vals = np.linspace(k0_val, k1_val, n_k + 1)
    
    # Generate points on the disc surface to evaluate potential
    s_pts, alpha_pts = generate_symmetric_points(N_s=3, N_alpha=8)
    num_points = len(s_pts)
    
    header = ['a_b', 'd_b', 'wave_frequency_K']
    for i in range(num_points):
        header.append(f'phi_real_{i}')
        header.append(f'phi_imag_{i}')
    header.extend(['Added_Mass', 'Damping_Coefficient'])
    
    filename = 'full_pinn_dataset.csv'
    
    print(f"Starting batched sweep computation for {len(a_vals)*len(d_vals)*len(K_vals)} configurations...")
    t0 = time.time()
    
    # Set number of threads appropriately
    opt._N_THREADS = max(1, os.cpu_count() // 2)
    
    final_sweep, X_sol_sweep = opt.sweep_problemcodeAMDC(
        a0, a1, n_a, d0, d1, n_d, k0_val, k1_val, n_k, N=N, b=b
    )
    print(f"Sweep completed in {time.time()-t0:.2f}s")
    
    print("Evaluating phi for all configurations...")
    t0 = time.time()
    phi_sweep = evaluate_phi(X_sol_sweep, N, s_pts, alpha_pts)
    print(f"Phi evaluation completed in {time.time()-t0:.2f}s")
    
    print("Writing to CSV...")
    t0 = time.time()
    
    rows = []
    for i_a, a_b in enumerate(a_vals):
        for i_d, d_b in enumerate(d_vals):
            for i_K, K in enumerate(K_vals):
                final = final_sweep[i_a, i_d, i_K]
                
                added_mass = np.real(np.pi * final * a_b)
                damping = np.imag(np.pi * final * a_b)
                
                phi_vals = phi_sweep[i_a, i_d, i_K]
                
                row = [a_b, d_b, K]
                for phi in phi_vals:
                    row.append(phi.real)
                    row.append(phi.imag)
                row.extend([added_mass, damping])
                
                rows.append(row)
                
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(filename, index=False)
                    
    print(f"Dataset successfully written to {filename} in {time.time()-t0:.2f}s")

if __name__ == '__main__':
    generate_full_dataset()
