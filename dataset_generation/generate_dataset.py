import numpy as np
import problemcodeAMDC_opt as opt
import csv
import time
import os

def evaluate_phi(X_sol, N, s_pts, alpha_pts):
    """
    Evaluates the potential difference [phi] at given points (s, alpha)
    using the series expansion and coefficients X_sol.
    """
    phi_vals = np.zeros_like(s_pts, dtype=np.complex128)
    q_idx = 0
    for k in range(N + 1):
        for m in range(N + 1):
            p1 = opt.alform(k, m, s_pts)
            p2 = np.cos(m * alpha_pts)
            phi_vals += X_sol[q_idx] * p1 * p2
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
    # Grid of parameters to sample
    a_b_list = [1.0, 1.25, 1.5, 1.75, 2.0]
    d_b_list = [0.1, 0.2, 0.3, 0.4]
    K_list = np.linspace(0.5, 2.0, 4)  # Non-dimensional wave number K
    
    N = 5 # Truncation order
    b = 1.0 # Keep semi-minor axis at 1 for non-dimensionalization
    
    # Generate points
    s_pts, alpha_pts = generate_symmetric_points(N_s=3, N_alpha=8)
    num_points = len(s_pts)
    
    header = ['a_b', 'd_b', 'wave_frequency_K']
    for i in range(num_points):
        header.append(f'phi_real_{i}')
        header.append(f'phi_imag_{i}')
    header.extend(['Added_Mass', 'Damping_Coefficient'])
    
    filename = 'dataset.csv'
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for a_b in a_b_list:
            for d_b in d_b_list:
                for K in K_list:
                    print(f"Running a/b={a_b:.2f}, d/b={d_b:.2f}, K={K:.2f}...")
                    t0 = time.time()
                    
                    # problemcodeAMDC returns (final, X_sol)
                    final, X_sol = opt.problemcodeAMDC(N, d_b, K, a_b, b)
                    
                    # Convert 'final' to dimensional-less Added Mass & Damping Coeff
                    added_mass = np.real(np.pi * final * a_b)
                    damping = np.imag(np.pi * final * a_b)
                    
                    # Evaluate the potential phi at the generated points
                    phi_vals = evaluate_phi(X_sol, N, s_pts, alpha_pts)
                    
                    row = [a_b, d_b, K]
                    for phi in phi_vals:
                        row.append(phi.real)
                        row.append(phi.imag)
                    row.extend([added_mass, damping])
                    
                    writer.writerow(row)
                    f.flush()
                    print(f"  -> Done in {time.time()-t0:.2f}s")
                    
if __name__ == '__main__':
    # Limit default threads if user wants so solver doesn't overload
    opt._N_THREADS = max(1, os.cpu_count() // 2)
    main()
