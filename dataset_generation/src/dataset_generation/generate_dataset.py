import numpy as np
from . import problemcodeAMDC_opt as opt
import pandas as pd
import time
import os

def generate_symmetric_points(N_s=3, N_alpha=8):
    """
    Generates radially symmetric points on the disc (in terms of s and alpha).
    s values will be distributed in rings, with one point at the origin.
    Returns 1D arrays of s and alpha values.
    """
    s_vals = np.linspace(0, 1, N_s + 1)[1:]
    alpha_vals = np.linspace(0, 2 * np.pi, N_alpha, endpoint=False)

    s_pts = [0.0]
    alpha_pts = [0.0]
    for s in s_vals:
        for alpha in alpha_vals:
            s_pts.append(s)
            alpha_pts.append(alpha)

    return np.array(s_pts), np.array(alpha_pts)

def generate_dataset(a0, a1, n_a, d0, d1, n_d, k0_val, k1_val, n_k,
                     filename='dataset.csv', N=5, b=1.0):
    a_vals = np.linspace(a0, a1, n_a + 1)
    d_vals = np.linspace(d0, d1, n_d + 1)
    K_vals = np.linspace(k0_val, k1_val, n_k + 1)

    s_pts, alpha_pts = generate_symmetric_points(N_s=3, N_alpha=8)
    num_points = len(s_pts)

    header = ['a_b', 'd_b', 'wave_frequency_K']
    for i in range(num_points):
        header.extend([
            f'phi_real_{i}', f'phi_imag_{i}',
            f'phi_real_d_da_{i}', f'phi_imag_d_da_{i}',
            f'phi_real_d_dd_{i}', f'phi_imag_d_dd_{i}',
            f'dphi_dx_real_{i}', f'dphi_dx_imag_{i}',
            f'dphi_dy_real_{i}', f'dphi_dy_imag_{i}'
        ])
    header.extend([
        'Added_Mass', 'Added_Mass_d_da', 'Added_Mass_d_dd',
        'Damping_Coefficient', 'Damping_Coefficient_d_da', 'Damping_Coefficient_d_dd'
    ])

    print("Starting exact analytical evaluation via JAX Autograd...")
    t0 = time.time()
    
    out_all, jac_a_all, jac_d_all = opt.generate_batch_data_jax(
        a_vals, d_vals, K_vals, N, b, s_pts, alpha_pts
    )
    
    AM_base, DC_base, phi_base, dphi_dx, dphi_dy = out_all
    dAM_da, dDC_da, dphi_da, _, _ = jac_a_all
    dAM_dd, dDC_dd, dphi_dd, _, _ = jac_d_all

    print(f"JAX evaluation completed in {time.time()-t0:.2f}s")

    print("Writing to CSV...")
    rows = []
    for i_a, a_b in enumerate(a_vals):
        for i_d, d_b in enumerate(d_vals):
            for i_K, K in enumerate(K_vals):
                row = [a_b, d_b, K]
                
                for i_p in range(num_points):
                    row.extend([
                        float(phi_base[i_a, i_d, i_K, i_p].real),
                        float(phi_base[i_a, i_d, i_K, i_p].imag),
                        float(dphi_da[i_a, i_d, i_K, i_p].real),
                        float(dphi_da[i_a, i_d, i_K, i_p].imag),
                        float(dphi_dd[i_a, i_d, i_K, i_p].real),
                        float(dphi_dd[i_a, i_d, i_K, i_p].imag),
                        float(dphi_dx[i_a, i_d, i_K, i_p].real),
                        float(dphi_dx[i_a, i_d, i_K, i_p].imag),
                        float(dphi_dy[i_a, i_d, i_K, i_p].real),
                        float(dphi_dy[i_a, i_d, i_K, i_p].imag),
                    ])
                
                # Global quantities
                row.extend([
                    float(AM_base[i_a, i_d, i_K]),
                    float(dAM_da[i_a, i_d, i_K]),
                    float(dAM_dd[i_a, i_d, i_K]),
                    float(DC_base[i_a, i_d, i_K]),
                    float(dDC_da[i_a, i_d, i_K]),
                    float(dDC_dd[i_a, i_d, i_K])
                ])
                rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    df.to_csv(filename, index=False)
    print(f"Dataset successfully written to {filename}")


if __name__ == '__main__':
    # Generate the small sample dataset by default
    generate_dataset(
        a0=1.0, a1=2.0, n_a=4,
        d0=0.1, d1=0.4, n_d=3,
        k0_val=0.5, k1_val=2.0, n_k=3,
        filename='dataset.csv'
    )
