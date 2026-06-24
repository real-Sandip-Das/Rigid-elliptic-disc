import numpy as np
from . import problemcodeAMDC_opt as opt
import csv
import time
import os

def evaluate_phi(X_sol, N, s_pts, alpha_pts):
    """
    Evaluates the potential difference [phi] at given points (s, alpha)
    using the series expansion and coefficients X_sol.
    X_sol can be a single array of shape (size,) or a batch array of shape (..., size).
    """
    # Create an output array with shape matching the batch dims of X_sol + the number of points
    phi_vals = np.zeros(X_sol.shape[:-1] + s_pts.shape, dtype=np.complex128)
    q_idx = 0
    for k in range(N + 1):
        for m in range(N + 1):
            p1 = opt.alform(k, m, s_pts)
            p2 = np.cos(m * alpha_pts)
            
            # Extract the q_idx coefficient for all batch dimensions
            coeff = X_sol[..., q_idx]
            # Add a new axis to coeff to broadcast against the points array
            phi_vals += coeff[..., np.newaxis] * (p1 * p2)
            q_idx += 1
    return phi_vals

def generate_symmetric_points(N_s=3, N_alpha=8):
    """
    Generates radially symmetric points on the disc (in terms of s and alpha).
    s values will be distributed in rings, with one point at the origin.
    Returns 1D arrays of s and alpha values.
    """
    # Exclude s=0 from linspace to add it explicitly as a single center point
    s_vals = np.linspace(0, 1, N_s + 1)[1:]
    alpha_vals = np.linspace(0, 2*np.pi, N_alpha, endpoint=False)
    
    s_pts = [0.0]
    alpha_pts = [0.0]
    for s in s_vals:
        for alpha in alpha_vals:
            s_pts.append(s)
            alpha_pts.append(alpha)
            
    return np.array(s_pts), np.array(alpha_pts)

def main():
    N = 5 # Truncation order
    b = 1.0 # Keep semi-minor axis at 1 for non-dimensionalization
    
    # Sweep parameters corresponding to previous list:
    # a_b_list = [1.0, 1.25, 1.5, 1.75, 2.0]
    a0, a1, n_a = 1.0, 2.0, 4
    # d_b_list = [0.1, 0.2, 0.3, 0.4]
    d0, d1, n_d = 0.1, 0.4, 3
    # K_list = [0.5, 1.0, 1.5, 2.0]
    k0_val, k1_val, n_k = 0.5, 2.0, 3
    
    a_vals = np.linspace(a0, a1, n_a + 1)
    d_vals = np.linspace(d0, d1, n_d + 1)
    K_vals = np.linspace(k0_val, k1_val, n_k + 1)
    
    # Generate points
    s_pts, alpha_pts = generate_symmetric_points(N_s=3, N_alpha=8)
    num_points = len(s_pts)
    
    header = ['a_b', 'd_b', 'wave_frequency_K']
    for i in range(num_points):
        header.append(f'phi_real_{i}')
        header.append(f'phi_imag_{i}')
    header.extend(['Added_Mass', 'Damping_Coefficient'])
    
    filename = 'dataset.csv'
    
    print("Starting batched sweep computation...")
    t0 = time.time()
    final_sweep, X_sol_sweep = opt.sweep_problemcodeAMDC(
        a0, a1, n_a, d0, d1, n_d, k0_val, k1_val, n_k, N=N, b=b
    )
    print(f"Sweep completed in {time.time()-t0:.2f}s")
    
    print("Evaluating phi for all configurations...")
    t0 = time.time()
    # Vectorized evaluation of phi for all configurations simultaneously
    phi_sweep = evaluate_phi(X_sol_sweep, N, s_pts, alpha_pts)
    print(f"Phi evaluation completed in {time.time()-t0:.2f}s")
    
    print("Writing to CSV...")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
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
                    
                    writer.writerow(row)
                    
    print(f"Dataset successfully written to {filename}")
                    
if __name__ == '__main__':
    # Limit default threads if user wants so solver doesn't overload
    opt._N_THREADS = max(1, os.cpu_count() // 2)
    main()
