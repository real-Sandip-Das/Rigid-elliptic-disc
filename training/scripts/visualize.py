import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

# Add dataset_generation to path so we can import the exact physical solver
dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset_generation/src'))
sys.path.append(dataset_dir)

from dataset_generation import problemcodeAMDC_opt as opt
from dataset_generation.generate_dataset import generate_symmetric_points
import jax
import jax.numpy as jnp

from training.config import WaveConfig
from training.model import DeepONetWaveSurrogate

def main():
    cfg = WaveConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Phase 2 model...")
    model = DeepONetWaveSurrogate(
        latent_dim=cfg.latent_dim, 
        subnet_width=cfg.subnet_width,
        fourier_mapping_size=cfg.fourier_mapping_size,
        fourier_scale=cfg.fourier_scale
    ).to(device)
    model.load_state_dict(torch.load(cfg.phase2_model_path, map_location=device, weights_only=True))
    model.eval()
    
    # ---------------------------------------------------------
    # PART 1: AM and DC vs K
    # ---------------------------------------------------------
    print("Generating AM and DC vs K Plot...")
    a_b = 1.55
    d_b = 0.25
    b = 1.0
    a = a_b * b
    d = d_b * b
    
    K_vals = np.linspace(0.5, 2.0, 50)
    
    actual_AM = []
    actual_DC = []
    
    # Calculate actuals using closed form
    pre = opt._precompute_A_data_jax(5)
    for K in K_vals:
        # Signature: problemcodeAMDC_jax(a, d_val, K, N, b, pre)
        final, _ = opt.problemcodeAMDC_jax(a, d, K, 5, b, pre)
        added_mass = np.real(np.pi * final * a_b)
        damping = np.imag(np.pi * final * a_b)
        actual_AM.append(added_mass)
        actual_DC.append(damping)
        
    actual_AM = np.array(actual_AM)
    actual_DC = np.array(actual_DC)
    
    # Predict with Phase 2 model
    wave_params = torch.tensor([[a_b, d_b, K] for K in K_vals], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_coeffs = model.predict_coeffs(wave_params).cpu().numpy()
        
    pred_AM = pred_coeffs[:, 0]
    pred_DC = pred_coeffs[:, 1]
    
    # Plot AM and DC
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_vals, actual_AM, 'k-', label='Actual')
    plt.plot(K_vals, pred_AM, 'r--', label='Predicted')
    plt.title(f'Added Mass vs K (a/b={a_b}, d/b={d_b})')
    plt.xlabel('K')
    plt.ylabel('Added Mass')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(K_vals, actual_DC, 'k-', label='Actual')
    plt.plot(K_vals, pred_DC, 'r--', label='Predicted')
    plt.title(f'Damping Coefficient vs K (a/b={a_b}, d/b={d_b})')
    plt.xlabel('K')
    plt.ylabel('Damping Coefficient')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    am_dc_path = os.path.join(os.path.dirname(__file__), '../am_dc_plot.png')
    plt.savefig(am_dc_path, dpi=300)
    plt.close()
    print(f"Saved AM/DC plot to {am_dc_path}")
    
    # ---------------------------------------------------------
    # PART 2: 3D Phi plot at K = 0.25
    # ---------------------------------------------------------
    print("Generating 3D Phi Plots for K = 0.25...")
    K_3d = 0.25
    pre_3d = opt._precompute_A_data_jax(5)
    final_3d, X_sol_3d = opt.problemcodeAMDC_jax(a, d, K_3d, 5, b, pre_3d)
    
    # Get the 25 actual anchor points
    s_pts, alpha_pts = generate_symmetric_points(N_s=3, N_alpha=8)
    
    # Use JAX evaluation
    s_j = jnp.array(s_pts)
    alpha_j = jnp.array(alpha_pts)
    phi_v = jax.vmap(lambda s, al: opt.phi_series(s, al, X_sol_3d, 5))(s_j, alpha_j)
    phi_actual = np.array(phi_v)
    
    # Convert points to cartesian
    x_actual = a * s_pts * np.cos(alpha_pts)
    y_actual = b * s_pts * np.sin(alpha_pts)
    
    phi_actual_real = phi_actual.real
    phi_actual_imag = phi_actual.imag
    
    # Generate dense grid for predictions over the elliptic disk
    s_dense = np.linspace(0, 1, 30)
    alpha_dense = np.linspace(0, 2*np.pi, 60)
    S, Alpha = np.meshgrid(s_dense, alpha_dense)
    
    X_dense = a * S * np.cos(Alpha)
    Y_dense = b * S * np.sin(Alpha)
    
    x_flat = X_dense.flatten()
    y_flat = Y_dense.flatten()
    
    wp_3d = torch.tensor([[a_b, d_b, K_3d]], dtype=torch.float32).to(device)
    x_t = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(0).to(device)
    y_t = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        latent = model.encode(wp_3d)
        phi_r_pred, phi_i_pred = model.forward_trunk_only(latent, x_t, y_t, wp_3d[:, 0])
        
    phi_r_pred = phi_r_pred.cpu().numpy().reshape(X_dense.shape)
    phi_i_pred = phi_i_pred.cpu().numpy().reshape(X_dense.shape)
    
    # Plot Real Part
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_dense, Y_dense, phi_r_pred, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.scatter(x_actual, y_actual, phi_actual_real, color='red', s=50, label='Actual 25 Points', depthshade=False)
    ax.set_title(f'Phi Real Part at K={K_3d} (a/b={a_b}, d/b={d_b})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Phi Real')
    ax.legend()
    real_3d_path = os.path.join(os.path.dirname(__file__), '../phi_real_3d.png')
    plt.savefig(real_3d_path, dpi=300)
    plt.close()
    print(f"Saved Phi Real 3D plot to {real_3d_path}")
    
    # Plot Imaginary Part
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_dense, Y_dense, phi_i_pred, cmap='plasma', alpha=0.8, edgecolor='none')
    ax.scatter(x_actual, y_actual, phi_actual_imag, color='red', s=50, label='Actual 25 Points', depthshade=False)
    ax.set_title(f'Phi Imaginary Part at K={K_3d} (a/b={a_b}, d/b={d_b})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Phi Imag')
    ax.legend()
    imag_3d_path = os.path.join(os.path.dirname(__file__), '../phi_imag_3d.png')
    plt.savefig(imag_3d_path, dpi=300)
    plt.close()
    print(f"Saved Phi Imag 3D plot to {imag_3d_path}")
    # ---------------------------------------------------------
    # PART 3: Contour plots of AM and DC
    # ---------------------------------------------------------
    print("Generating AM and DC Contour Plots...")
    
    a_b_vals = np.linspace(1.0, 2.0, 50)
    d_b_vals = np.linspace(0.1, 0.4, 50)
    A_B, D_B = np.meshgrid(a_b_vals, d_b_vals)
    
    K_plot_vals = np.linspace(0.1, 2.0, 20)
    
    # 1. Run the model once to get all values
    wp_list = []
    for K_val in K_plot_vals:
        for a, d in zip(A_B.flatten(), D_B.flatten()):
            wp_list.append([a, d, K_val])
            
    wp_contour = torch.tensor(wp_list, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_coeffs_contour = model.predict_coeffs(wp_contour).cpu().numpy()
        
    # Reshape predictions back into separate grids for each K value
    pred_AM_all = pred_coeffs_contour[:, 0].reshape((20, A_B.shape[0], A_B.shape[1]))
    pred_DC_all = pred_coeffs_contour[:, 1].reshape((20, A_B.shape[0], A_B.shape[1]))
    
    global_am_min, global_am_max = np.min(pred_AM_all), np.max(pred_AM_all)
    global_dc_min, global_dc_max = np.min(pred_DC_all), np.max(pred_DC_all)
    
    levels_am_global = np.linspace(global_am_min, global_am_max, 30)
    levels_dc_global = np.linspace(global_dc_min, global_dc_max, 30)
    
    # Helper function to create the figure
    def generate_contour_pdf(consistent_scale=True, filename='output.pdf'):
        with PdfPages(filename) as pdf:
            # We want 2 K values per page. 20 K values = 10 pages.
            for page_idx in range(10):
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                for i in range(2):
                    idx = page_idx * 2 + i
                    if idx >= len(K_plot_vals):
                        break
                        
                    K_val = K_plot_vals[idx]
                    pred_AM = pred_AM_all[idx]
                    pred_DC = pred_DC_all[idx]
                    
                    if consistent_scale:
                        levels_am = levels_am_global
                        levels_dc = levels_dc_global
                        vmin_am, vmax_am = global_am_min, global_am_max
                        vmin_dc, vmax_dc = global_dc_min, global_dc_max
                    else:
                        levels_am = np.linspace(np.min(pred_AM), np.max(pred_AM), 30)
                        levels_dc = np.linspace(np.min(pred_DC), np.max(pred_DC), 30)
                        vmin_am, vmax_am = np.min(pred_AM), np.max(pred_AM)
                        vmin_dc, vmax_dc = np.min(pred_DC), np.max(pred_DC)
                    
                    # AM (Left)
                    ax_am = axes[i, 0]
                    c_am = ax_am.contourf(A_B, D_B, pred_AM, levels=levels_am, cmap='viridis', vmin=vmin_am, vmax=vmax_am)
                    fig.colorbar(c_am, ax=ax_am)
                    ax_am.set_title(f'Added Mass (K={K_val:.2f})')
                    ax_am.set_xlabel('a/b')
                    ax_am.set_ylabel('d/b')
                    
                    # DC (Right)
                    ax_dc = axes[i, 1]
                    c_dc = ax_dc.contourf(A_B, D_B, pred_DC, levels=levels_dc, cmap='plasma', vmin=vmin_dc, vmax=vmax_dc)
                    fig.colorbar(c_dc, ax=ax_dc)
                    ax_dc.set_title(f'Damping Coefficient (K={K_val:.2f})')
                    ax_dc.set_xlabel('a/b')
                    ax_dc.set_ylabel('d/b')
                    
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    # 2. Generate and save consistent scale figure
    print("Saving consistent scale contour plot PDF...")
    path_consistent = os.path.join(os.path.dirname(__file__), '../am_dc_contour_consistent.pdf')
    generate_contour_pdf(consistent_scale=True, filename=path_consistent)
    
    # 3. Generate and save independent scale figure
    print("Saving independent scale contour plot PDF...")
    path_independent = os.path.join(os.path.dirname(__file__), '../am_dc_contour_independent.pdf')
    generate_contour_pdf(consistent_scale=False, filename=path_independent)
    
    print("Saved both contour plots.")

if __name__ == '__main__':
    main()
