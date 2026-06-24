from training.config import WaveConfig
from training.train2 import train_phase2

if __name__ == "__main__":
    cfg = WaveConfig()
    
    # Override configuration for Phase 2
    cfg.p2_epochs = 5000 
    cfg.p2_lr = 1e-4
    
    train_phase2(cfg)
