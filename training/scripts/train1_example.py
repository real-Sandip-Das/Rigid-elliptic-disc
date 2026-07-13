from training.config import WaveConfig
from training.train1 import train_phase1


cfg = WaveConfig()

cfg.data_path = "data/full_pinn_dataset.csv"

cfg.grad_clip_norm = 100.0
cfg.p1_epochs = 10000
cfg.p1_lr = 3e-3
cfg.p1_batch_size = 512

cfg.lra_warmup_threshold = 0.06
cfg.log_every = 1

cfg.use_wandb = True
cfg.wandb_name = f"BS:{cfg.p1_batch_size} lr:{cfg.p1_lr:.1e}"


cfg.hf_repo_id = ""

print(f"Loaded config. Saving checkpoints to: {cfg.checkpoint_dir}")


train_phase1(cfg)
