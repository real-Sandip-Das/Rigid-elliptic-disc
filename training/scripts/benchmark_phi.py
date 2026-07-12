import torch
import torch.nn as nn
from training.config import WaveConfig
from training.model import DeepONetWaveSurrogate
from training.data import get_dataloaders, generate_anchor_coords

def main():
    cfg = WaveConfig()
    cfg.data_path = "data/full_pinn_dataset.csv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {cfg.phase1_model_path} onto {device}")
    model = DeepONetWaveSurrogate(
        latent_dim=cfg.latent_dim, 
        subnet_width=cfg.subnet_width,
        fourier_mapping_size=cfg.fourier_mapping_size,
        fourier_scale=cfg.fourier_scale
    ).to(device)
    model.load_state_dict(torch.load(cfg.phase1_model_path, map_location=device))
    model.eval()
    
    print(f"Loading dataset from {cfg.data_path}")
    train_loader, val_loader = get_dataloaders(cfg.data_path, batch_size=512)
    
    mse_loss = nn.MSELoss()
    
    def evaluate_loader(loader, split_name):
        split_loss_r = 0.0
        split_loss_i = 0.0
        
        abs_err_r = 0.0
        abs_err_i = 0.0
        
        all_true_r = []
        all_pred_r = []
        all_true_i = []
        all_pred_i = []
        
        n = 0
        with torch.no_grad():
            for batch in loader:
                wave_params = batch[0].to(device)
                phi_r_true, phi_i_true = batch[1].to(device), batch[2].to(device)
                
                a_b = wave_params[:, 0]
                x, y = generate_anchor_coords(a_b)
                x = x.to(device)
                y = y.to(device)
                
                latent = model.encode(wave_params)
                phi_r_pred, phi_i_pred = model.forward_trunk_only(latent, x, y, a_b)
                
                loss_r = mse_loss(phi_r_pred, phi_r_true)
                loss_i = mse_loss(phi_i_pred, phi_i_true)
                
                bs = wave_params.size(0)
                split_loss_r += loss_r.item() * bs
                split_loss_i += loss_i.item() * bs
                
                abs_err_r += torch.sum(torch.abs(phi_r_pred - phi_r_true)).item()
                abs_err_i += torch.sum(torch.abs(phi_i_pred - phi_i_true)).item()
                
                all_true_r.append(phi_r_true.view(-1).cpu())
                all_pred_r.append(phi_r_pred.view(-1).cpu())
                all_true_i.append(phi_i_true.view(-1).cpu())
                all_pred_i.append(phi_i_pred.view(-1).cpu())
                
                n += bs
                
        # 25 anchor points per sample
        pts = n * 25
        
        true_r = torch.cat(all_true_r)
        pred_r = torch.cat(all_pred_r)
        true_i = torch.cat(all_true_i)
        pred_i = torch.cat(all_pred_i)

        ss_res_r = torch.sum((true_r - pred_r)**2).item()
        ss_tot_r = torch.sum((true_r - torch.mean(true_r))**2).item()
        r2_r = 1 - (ss_res_r / ss_tot_r) if ss_tot_r != 0 else 0.0

        ss_res_i = torch.sum((true_i - pred_i)**2).item()
        ss_tot_i = torch.sum((true_i - torch.mean(true_i))**2).item()
        r2_i = 1 - (ss_res_i / ss_tot_i) if ss_tot_i != 0 else 0.0
        
        print(f"\n{split_name} Split:")
        print(f"  Samples: {n}")
        print(f"  MSE (Real): {split_loss_r / n:.6f}")
        print(f"  MSE (Imag): {split_loss_i / n:.6f}")
        print(f"  MSE (Total): {(split_loss_r + split_loss_i) / n:.6f}")
        print(f"  Mean Abs Error (Real): {abs_err_r / pts:.6f}")
        print(f"  Mean Abs Error (Imag): {abs_err_i / pts:.6f}")
        print(f"  R^2 Score (Real): {r2_r:.6f}")
        print(f"  R^2 Score (Imag): {r2_i:.6f}")
        return split_loss_r, split_loss_i, abs_err_r, abs_err_i, r2_r, r2_i, n

    train_r, train_i, train_mae_r, train_mae_i, train_r2_r, train_r2_i, train_n = evaluate_loader(train_loader, "Train")
    val_r, val_i, val_mae_r, val_mae_i, val_r2_r, val_r2_i, val_n = evaluate_loader(val_loader, "Validation")
    
    overall_n = train_n + val_n
    pts = overall_n * 25
    overall_r = (train_r + val_r) / overall_n
    overall_i = (train_i + val_i) / overall_n
    overall_mae_r = (train_mae_r + val_mae_r) / pts
    overall_mae_i = (train_mae_i + val_mae_i) / pts
    overall_r2_r = (train_r2_r * train_n + val_r2_r * val_n) / overall_n
    overall_r2_i = (train_r2_i * train_n + val_r2_i * val_n) / overall_n
    
    print(f"\nOverall:")
    print(f"  Samples: {overall_n}")
    print(f"  MSE (Real): {overall_r:.6f}")
    print(f"  MSE (Imag): {overall_i:.6f}")
    print(f"  MSE (Total): {overall_r + overall_i:.6f}")
    print(f"  Mean Abs Error (Real): {overall_mae_r:.6f}")
    print(f"  Mean Abs Error (Imag): {overall_mae_i:.6f}")
    print(f"  R^2 Score (Real): {overall_r2_r:.6f}")
    print(f"  R^2 Score (Imag): {overall_r2_i:.6f}")

if __name__ == '__main__':
    main()
