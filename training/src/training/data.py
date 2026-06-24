import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math

class WaveDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
        # Inputs: a_b, d_b, K
        self.wave_params = self.df[['a_b', 'd_b', 'wave_frequency_K']].values
        
        # Targets: Phi (Real and Imag separated)
        real_cols = [f'phi_real_{i}' for i in range(25)]
        imag_cols = [f'phi_imag_{i}' for i in range(25)]
        self.phi_real = self.df[real_cols].values
        self.phi_imag = self.df[imag_cols].values
        
        # Targets: Global Coefficients
        self.coeffs = self.df[['Added_Mass', 'Damping_Coefficient']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.wave_params[idx], dtype=torch.float32),
            torch.tensor(self.phi_real[idx], dtype=torch.float32),
            torch.tensor(self.phi_imag[idx], dtype=torch.float32),
            torch.tensor(self.coeffs[idx], dtype=torch.float32)
        )

def get_dataloaders(csv_file, batch_size, val_split=0.2, seed=42):
    dataset = WaveDataset(csv_file)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def compute_8_features(x, y, z):
    """Helper to convert cartesian to your custom Fourier spatial features."""
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return torch.stack([
        r, theta, torch.sin(theta), torch.cos(theta),
        z, torch.sin(z), torch.sin(2*z), torch.sin(3*z)
    ], dim=-1)

def generate_anchor_features(a_b, d_b, N_s=3, N_alpha=8):
    """Generates the 8 features for the 25 specific dataset points."""
    batch_size = a_b.shape[0]
    
    s_vals = [0.0]
    alpha_vals = [0.0]
    
    s_rings = np.linspace(0, 1, N_s + 1)[1:]
    alphas = np.linspace(0, 2*np.pi, N_alpha, endpoint=False)
    
    for s in s_rings:
        for alpha in alphas:
            s_vals.append(s); alpha_vals.append(alpha)
            
    s_t = torch.tensor(s_vals, dtype=torch.float32, device=a_b.device).unsqueeze(0).repeat(batch_size, 1)
    alpha_t = torch.tensor(alpha_vals, dtype=torch.float32, device=a_b.device).unsqueeze(0).repeat(batch_size, 1)
    
    x = a_b.unsqueeze(1) * s_t * torch.cos(alpha_t)
    y = 1.0 * s_t * torch.sin(alpha_t)
    z = -d_b.unsqueeze(1).repeat(1, 25)
    
    return compute_8_features(x, y, z)

def generate_collocation_features(a_b, d_b, num_points):
    """Generates random points in the fluid domain for Sobolev/PDE training."""
    batch_size = a_b.shape[0]
    device = a_b.device
    
    # Randomly sample s [0, 3] to go past the disk, alpha [0, 2pi], z from bottom to surface [0, d_b]
    s_rand = torch.rand((batch_size, num_points), device=device) * 3.0
    alpha_rand = torch.rand((batch_size, num_points), device=device) * 2 * math.pi
    
    # Z ranges from -d_b (depth of disk) up to 0 (surface)
    z_rand = -d_b.unsqueeze(1) * torch.rand((batch_size, num_points), device=device)
    
    x_rand = a_b.unsqueeze(1) * s_rand * torch.cos(alpha_rand)
    y_rand = 1.0 * s_rand * torch.sin(alpha_rand)
    
    features = compute_8_features(x_rand, y_rand, z_rand)
    features.requires_grad_(True) # CRITICAL: Required to compute d/dx for PDE loss
    
    return features
