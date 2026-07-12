import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add dataset_generation to path so we can import the exact physical solver
dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset_generation/src'))
sys.path.append(dataset_dir)

from dataset_generation import problemcodeAMDC_opt as opt
from dataset_generation.generate_dataset import evaluate_phi, generate_symmetric_points

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
    for K in K_vals:
        # Signature: problemcodeAMDC(N, d_val, K, a, b)
        final, _ = opt.problemcodeAMDC(5, d, K, a, b)
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
    final_3d, X_sol_3d = opt.problemcodeAMDC(5, d, K_3d, a, b)
    
    # Get the 25 actual anchor points
    s_pts, alpha_pts = generate_symmetric_points(N_s=3, N_alpha=8)
    
    # evaluate_phi expects an extra batch dimension, so we add one.
    X_sol_3d_batch = np.expand_dims(X_sol_3d, axis=0)
    phi_actual = evaluate_phi(X_sol_3d_batch, 5, s_pts, alpha_pts)[0]
    
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

if __name__ == '__main__':
    main()
