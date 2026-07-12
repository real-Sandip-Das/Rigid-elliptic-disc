import time
import torch
import torch.nn.functional as F
from training.model import DeepONetWaveSurrogate
from training.data import get_dataloaders

def train_phase2(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Phase 2 Training on {device}...")

    train_loader, val_loader = get_dataloaders(config.data_path, config.p2_batch_size)
    model = DeepONetWaveSurrogate(latent_dim=config.latent_dim, subnet_width=config.subnet_width).to(device)
    
    # Load frozen Phase 1 weights
    model.load_state_dict(torch.load(config.phase1_model_path, map_location=device, weights_only=True))
    model.freeze_for_phase2()
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.p2_lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=100, min_lr=1e-7
    )

    for epoch in range(1, config.p2_epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            wave_params = batch[0].to(device)
            true_coeffs = batch[7].to(device)
            
            optimizer.zero_grad()
            pred_coeffs = model.predict_coeffs(wave_params)
            loss = F.mse_loss(pred_coeffs, true_coeffs)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        scheduler.step(avg_train_loss)
        
        if epoch % config.log_every == 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    wave_params = batch[0].to(device)
                    true_coeffs = batch[7].to(device)
                    pred_coeffs = model.predict_coeffs(wave_params)
                    loss = F.mse_loss(pred_coeffs, true_coeffs)
                    total_val_loss += loss.item()
            
            epoch_time = time.time() - t0
            print(f"Epoch {epoch}/{config.p2_epochs} | Train Loss: {total_train_loss/len(train_loader):.6f} | Val Loss: {total_val_loss/max(1, len(val_loader)):.6f} | time: {epoch_time:.2f}s")

    torch.save(model.state_dict(), config.phase2_model_path)
    print(f"Phase 2 complete. Model saved to {config.phase2_model_path}")
