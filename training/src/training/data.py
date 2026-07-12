import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class WaveDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
        # Inputs: a_b, d_b, K
        self.wave_params = torch.tensor(self.df[['a_b', 'd_b', 'wave_frequency_K']].values, dtype=torch.float32)
        
        # Targets: Phi values
        real_cols = [f'phi_real_{i}' for i in range(25)]
        imag_cols = [f'phi_imag_{i}' for i in range(25)]
        self.phi_real = torch.tensor(self.df[real_cols].values, dtype=torch.float32)
        self.phi_imag = torch.tensor(self.df[imag_cols].values, dtype=torch.float32)
        
        # Targets: Phi derivatives
        dx_real_cols = [f'dphi_dx_real_{i}' for i in range(25)]
        dx_imag_cols = [f'dphi_dx_imag_{i}' for i in range(25)]
        dy_real_cols = [f'dphi_dy_real_{i}' for i in range(25)]
        dy_imag_cols = [f'dphi_dy_imag_{i}' for i in range(25)]
        
        self.dphi_dx_real = torch.tensor(self.df[dx_real_cols].values, dtype=torch.float32)
        self.dphi_dx_imag = torch.tensor(self.df[dx_imag_cols].values, dtype=torch.float32)
        self.dphi_dy_real = torch.tensor(self.df[dy_real_cols].values, dtype=torch.float32)
        self.dphi_dy_imag = torch.tensor(self.df[dy_imag_cols].values, dtype=torch.float32)
        
        # Targets: Global Coefficients
        self.coeffs = torch.tensor(self.df[['Added_Mass', 'Damping_Coefficient']].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.wave_params[idx],
            self.phi_real[idx],
            self.phi_imag[idx],
            self.dphi_dx_real[idx],
            self.dphi_dx_imag[idx],
            self.dphi_dy_real[idx],
            self.dphi_dy_imag[idx],
            self.coeffs[idx]
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

def generate_anchor_coords(a_b, N_s=3, N_alpha=8):
    """Generates the x, y coordinates for the 25 specific dataset points."""
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
    
    return x, y
