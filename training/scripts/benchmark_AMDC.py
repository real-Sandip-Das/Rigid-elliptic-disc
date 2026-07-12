import torch
import torch.nn as nn
from training.config import WaveConfig
from training.model import DeepONetWaveSurrogate
from training.data import get_dataloaders

def main():
    cfg = WaveConfig()
    cfg.data_path = "data/full_pinn_dataset.csv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {cfg.phase2_model_path} onto {device}")
    model = DeepONetWaveSurrogate(
        latent_dim=cfg.latent_dim, 
        subnet_width=cfg.subnet_width,
        fourier_mapping_size=cfg.fourier_mapping_size,
        fourier_scale=cfg.fourier_scale
    ).to(device)
    model.load_state_dict(torch.load(cfg.phase2_model_path, map_location=device))
    model.eval()
    
    print(f"Loading dataset from {cfg.data_path}")
    train_loader, val_loader = get_dataloaders(cfg.data_path, batch_size=512)
    
    mse_loss = nn.MSELoss()
    
    def evaluate_loader(loader, split_name):
        split_loss_am = 0.0
        split_loss_dc = 0.0
        
        abs_err_am = 0.0
        abs_err_dc = 0.0
        
        all_true_am = []
        all_pred_am = []
        all_true_dc = []
        all_pred_dc = []
        
        n = 0
        with torch.no_grad():
            for batch in loader:
                wave_params = batch[0].to(device)
                true_coeffs = batch[7].to(device)
                
                pred_coeffs = model.predict_coeffs(wave_params)
                
                loss_am = mse_loss(pred_coeffs[:, 0], true_coeffs[:, 0])
                loss_dc = mse_loss(pred_coeffs[:, 1], true_coeffs[:, 1])
                
                bs = wave_params.size(0)
                split_loss_am += loss_am.item() * bs
                split_loss_dc += loss_dc.item() * bs
                
                abs_err_am += torch.sum(torch.abs(pred_coeffs[:, 0] - true_coeffs[:, 0])).item()
                abs_err_dc += torch.sum(torch.abs(pred_coeffs[:, 1] - true_coeffs[:, 1])).item()
                
                all_true_am.append(true_coeffs[:, 0].view(-1).cpu())
                all_pred_am.append(pred_coeffs[:, 0].view(-1).cpu())
                all_true_dc.append(true_coeffs[:, 1].view(-1).cpu())
                all_pred_dc.append(pred_coeffs[:, 1].view(-1).cpu())
                
                n += bs
                
        true_am = torch.cat(all_true_am)
        pred_am = torch.cat(all_pred_am)
        true_dc = torch.cat(all_true_dc)
        pred_dc = torch.cat(all_pred_dc)

        ss_res_am = torch.sum((true_am - pred_am)**2).item()
        ss_tot_am = torch.sum((true_am - torch.mean(true_am))**2).item()
        r2_am = 1 - (ss_res_am / ss_tot_am) if ss_tot_am != 0 else 0.0

        ss_res_dc = torch.sum((true_dc - pred_dc)**2).item()
        ss_tot_dc = torch.sum((true_dc - torch.mean(true_dc))**2).item()
        r2_dc = 1 - (ss_res_dc / ss_tot_dc) if ss_tot_dc != 0 else 0.0
        
        print(f"\n{split_name} Split:")
        print(f"  Samples: {n}")
        print(f"  MSE (AM): {split_loss_am / n:.6f}")
        print(f"  MSE (DC): {split_loss_dc / n:.6f}")
        print(f"  MSE (Total): {(split_loss_am + split_loss_dc) / n:.6f}")
        print(f"  Mean Abs Error (AM): {abs_err_am / n:.6f}")
        print(f"  Mean Abs Error (DC): {abs_err_dc / n:.6f}")
        print(f"  R^2 Score (AM): {r2_am:.6f}")
        print(f"  R^2 Score (DC): {r2_dc:.6f}")
        return split_loss_am, split_loss_dc, abs_err_am, abs_err_dc, r2_am, r2_dc, n

    train_am, train_dc, train_mae_am, train_mae_dc, train_r2_am, train_r2_dc, train_n = evaluate_loader(train_loader, "Train")
    val_am, val_dc, val_mae_am, val_mae_dc, val_r2_am, val_r2_dc, val_n = evaluate_loader(val_loader, "Validation")
    
    overall_n = train_n + val_n
    overall_am = (train_am + val_am) / overall_n
    overall_dc = (train_dc + val_dc) / overall_n
    overall_mae_am = (train_mae_am + val_mae_am) / overall_n
    overall_mae_dc = (train_mae_dc + val_mae_dc) / overall_n
    overall_r2_am = (train_r2_am * train_n + val_r2_am * val_n) / overall_n
    overall_r2_dc = (train_r2_dc * train_n + val_r2_dc * val_n) / overall_n
    
    print(f"\nOverall:")
    print(f"  Samples: {overall_n}")
    print(f"  MSE (AM): {overall_am:.6f}")
    print(f"  MSE (DC): {overall_dc:.6f}")
    print(f"  MSE (Total): {overall_am + overall_dc:.6f}")
    print(f"  Mean Abs Error (AM): {overall_mae_am:.6f}")
    print(f"  Mean Abs Error (DC): {overall_mae_dc:.6f}")
    print(f"  R^2 Score (AM): {overall_r2_am:.6f}")
    print(f"  R^2 Score (DC): {overall_r2_dc:.6f}")

if __name__ == '__main__':
    main()
