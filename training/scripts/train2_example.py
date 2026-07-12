from training.config import WaveConfig
from training.train2 import train_phase2

if __name__ == "__main__":
    cfg = WaveConfig()
    
    cfg.data_path = "data/full_pinn_dataset.csv"
    cfg.p2_batch_size = 1024
    cfg.p2_epochs = 10000
    cfg.p2_lr = 1e-1
    cfg.log_every = 10
    
    train_phase2(cfg)
