from training.config import WaveConfig
from training.train1 import train_phase1

# Initialize the default configuration
cfg = WaveConfig()

# ── 2 epochs Adam (high LR) → 2 epochs L-BFGS ───────────────────
cfg.p1_epochs               = 8    # 8 epochs total
cfg.p1_lr                   = 1e-2  # High learning rate for Adam exploration
cfg.p1_batch_size           = 512
cfg.colloc_points_per_batch = 75    # Fewer colloc pts → faster 3rd-order AD
cfg.lbfgs_max_iter          = 20
cfg.lbfgs_max_eval          = 25
cfg.log_every               = 1     # Print every epoch so we can see progress

# ── Hugging Face Hub Integration (Optional) ───────────────────────────
# Uncomment and fill to enable pushing/pulling to/from HF Hub
cfg.hf_repo_id = "real-Sandip-Das/test-pinn-checkpoint"
cfg.hf_sync_every = 4           # Sync mid-training every N epochs
cfg.hf_force_initialize = False   # Set True to skip pulling state from Hub

print(f"Loaded config. Saving checkpoints to: {cfg.checkpoint_dir}")

# Execute the training pipeline
train_phase1(cfg)
