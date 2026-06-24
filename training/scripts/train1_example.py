from training.config import WaveConfig
from training.train1 import train_phase1

if __name__ == "__main__":
    # Initialize the default configuration
    cfg = WaveConfig()
    
    # Override configuration specifically for this run
    cfg.p1_epochs = 10          # Example: Train longer
    cfg.p1_lr = 5e-4               # Example: Lower learning rate
    cfg.p1_batch_size = 64         # Example: Larger batch size
    
    print(f"Loaded config. Saving checkpoints to: {cfg.checkpoint_dir}")
    
    # Execute the training pipeline
    train_phase1(cfg)
