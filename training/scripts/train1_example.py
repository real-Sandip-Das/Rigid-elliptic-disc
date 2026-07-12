from training.config import WaveConfig
from training.train1 import train_phase1

# Initialize the default configuration
cfg = WaveConfig()

cfg.data_path = "data/full_pinn_dataset.csv"

cfg.grad_clip_norm = 3000.0
cfg.p1_epochs = 2000
cfg.p1_lr = 1e-4
cfg.p1_batch_size = 1024
cfg.lbfgs_max_iter = 50000
cfg.lbfgs_max_eval = 50000
cfg.log_every = 1

# ── Hugging Face Hub Integration (Optional) ───────────────────────────
cfg.hf_repo_id = ""

print(f"Loaded config. Saving checkpoints to: {cfg.checkpoint_dir}")

# Execute the training pipeline
train_phase1(cfg)
