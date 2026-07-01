import numpy as np
from . import problemcodeAMDC_opt as opt
import pandas as pd
import time
import os
import sympy as sp
import functools

# 1. Define the symbolic derivative of the radial basis function A(s)
@functools.lru_cache(maxsize=None)
def _get_alform_ds_func(k, m):
    s = sp.Symbol('s')
    l = m + 2*k + 1
    Pl = sp.legendre(l, s)
    P1 = ((-1)**m) * (1 - s**2)**(sp.Rational(m, 2)) * sp.diff(Pl, s, m)
    P = P1.subs(s, sp.sqrt(1 - s**2))
    dP_ds = sp.diff(P, s)
    return sp.lambdify(s, dP_ds, modules='numpy')

def alform_ds(k, m, s):
    func = _get_alform_ds_func(k, m)
    res = func(s)
    if np.isscalar(res) and np.ndim(s) > 0:
        return np.full_like(s, res, dtype=np.float64)
    return res

def evaluate_phi_derivatives(X_sol, N, s_pts, alpha_pts, a_b, b=1.0):
    a = a_b * b
    
    eps = 1e-8
    s_safe = np.where(s_pts < eps, eps, s_pts)
    # Clip near 1 to avoid the 1/sqrt(1-s**2) edge singularity
    s_safe = np.where(s_safe > 1 - eps, 1 - eps, s_safe)
    
    dphi_ds = np.zeros(X_sol.shape[:-1] + s_pts.shape, dtype=np.complex128)
    dphi_dalpha = np.zeros(X_sol.shape[:-1] + s_pts.shape, dtype=np.complex128)
    
    q_idx = 0
    for k in range(N + 1):
        for m in range(N + 1):
            p1 = opt.alform(k, m, s_safe)
            p1_ds = alform_ds(k, m, s_safe)
            
            p2 = np.cos(m * alpha_pts)
            p2_dalpha = -m * np.sin(m * alpha_pts)
            
            coeff = X_sol[..., q_idx][..., np.newaxis]
            
            dphi_ds += coeff * (p1_ds * p2)
            dphi_dalpha += coeff * (p1 * p2_dalpha)
            
            q_idx += 1
            
    # Apply the chain rule conversion to get Cartesian derivatives
    if isinstance(a, np.ndarray):
        while a.ndim < dphi_ds.ndim:
            a = a[..., np.newaxis]
            
    dphi_dx = (np.cos(alpha_pts) / a) * dphi_ds - (np.sin(alpha_pts) / (a * s_safe)) * dphi_dalpha
    dphi_dy = (np.sin(alpha_pts) / b) * dphi_ds + (np.cos(alpha_pts) / (b * s_safe)) * dphi_dalpha
    
    # Do not record derivative loss on the edge singularity
    edge_mask = s_pts >= 1.0 - 1e-10
    dphi_dx[..., edge_mask] = np.nan + 1j * np.nan
    dphi_dy[..., edge_mask] = np.nan + 1j * np.nan
    
    return dphi_dx, dphi_dy

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

def generate_dataset(a0, a1, n_a, d0, d1, n_d, k0_val, k1_val, n_k, filename='dataset.csv', N=5, b=1.0):
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
        header.append(f'dphi_dx_real_{i}')
        header.append(f'dphi_dx_imag_{i}')
        header.append(f'dphi_dy_real_{i}')
        header.append(f'dphi_dy_imag_{i}')
    header.extend(['Added_Mass', 'Damping_Coefficient'])

    
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
    
    print("Evaluating phi derivatives...")
    t0 = time.time()
    dphi_dx_sweep, dphi_dy_sweep = evaluate_phi_derivatives(
        X_sol_sweep, N, s_pts, alpha_pts, a_b=a_vals[:, None, None], b=b
    )
    print(f"Phi derivatives evaluation completed in {time.time()-t0:.2f}s")
    
    print("Writing to CSV...")
    
    rows = []
    for i_a, a_b in enumerate(a_vals):
        for i_d, d_b in enumerate(d_vals):
            for i_K, K in enumerate(K_vals):
                final = final_sweep[i_a, i_d, i_K]
                
                added_mass = np.real(np.pi * final * a_b)
                damping = np.imag(np.pi * final * a_b)
                
                phi_vals = phi_sweep[i_a, i_d, i_K]
                dphi_dx_vals = dphi_dx_sweep[i_a, i_d, i_K]
                dphi_dy_vals = dphi_dy_sweep[i_a, i_d, i_K]
                
                row = [a_b, d_b, K]
                for phi, ddx, ddy in zip(phi_vals, dphi_dx_vals, dphi_dy_vals):
                    row.append(phi.real)
                    row.append(phi.imag)
                    row.append(ddx.real)
                    row.append(ddx.imag)
                    row.append(ddy.real)
                    row.append(ddy.imag)
                row.extend([added_mass, damping])
                
                rows.append(row)
                
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(filename, index=False)
                    
    print(f"Dataset successfully written to {filename}")
                    
if __name__ == '__main__':
    # Limit default threads if user wants so solver doesn't overload
    opt._N_THREADS = max(1, os.cpu_count() // 2)
    # Generate the small sample dataset by default
    generate_dataset(
        a0=1.0, a1=2.0, n_a=4,
        d0=0.1, d1=0.4, n_d=3,
        k0_val=0.5, k1_val=2.0, n_k=3,
        filename='dataset.csv'
    )
