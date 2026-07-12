from training.config import WaveConfig
from training.train1 import train_phase1

# Initialize the default configuration
cfg = WaveConfig()

cfg.data_path = "data/full_pinn_dataset.csv"

cfg.grad_clip_norm = 1000.0
cfg.p1_epochs = 10000
cfg.p1_lr = 3e-3
cfg.p1_batch_size = 512

cfg.lra_warmup_threshold = 0.02
cfg.log_every = 1

cfg.use_wandb = True
cfg.wandb_name = f"BS:{cfg.p1_batch_size} lr:{cfg.p1_lr:.1e}"

# ── Hugging Face Hub Integration (Optional) ───────────────────────────
cfg.hf_repo_id = ""

print(f"Loaded config. Saving checkpoints to: {cfg.checkpoint_dir}")

# Execute the training pipeline
train_phase1(cfg)
