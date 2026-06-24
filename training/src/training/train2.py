import torch
from training.model import DeepONetWaveSurrogate
from training.data import get_dataloader
from training.losses import phase2_mse_loss

def train_phase2(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Phase 2 Training on {device}...")

    dataloader = get_dataloader(config.data_path, config.p2_batch_size)
    model = DeepONetWaveSurrogate(latent_dim=config.latent_dim).to(device)
    
    # Load frozen Phase 1 weights
    model.load_state_dict(torch.load(config.phase1_model_path, map_location=device, weights_only=True))
    model.freeze_for_phase2()
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.p2_lr)

    for epoch in range(1, config.p2_epochs + 1):
        total_loss_epoch = 0.0
        
        for batch in dataloader:
            wave_params, _, _, true_coeffs = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            pred_coeffs = model.predict_coeffs(wave_params)
            loss = phase2_mse_loss(pred_coeffs, true_coeffs)
            
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{config.p2_epochs} | Loss (MSE): {total_loss_epoch/len(dataloader):.6f}")

    torch.save(model.state_dict(), config.phase2_model_path)
    print(f"Phase 2 complete. Model saved to {config.phase2_model_path}")
