from training.config import WaveConfig
from training.train1 import train_phase1

if __name__ == "__main__":
    # Initialize the default configuration
    cfg = WaveConfig()

    # ── 20 epochs Adam (high LR) → 20 epochs L-BFGS ───────────────────
    cfg.p1_epochs               = 40    # 20 Adam + 20 L-BFGS
    cfg.p1_lr                   = 1e-2  # High learning rate for Adam exploration
    cfg.p1_batch_size           = 64
    cfg.colloc_points_per_batch = 50    # Fewer colloc pts → faster 3rd-order AD
    cfg.physics_colloc_points   = 20    # Fewer pts in physics-only epochs too
    cfg.lbfgs_start_epoch       = 21    # Switch to L-BFGS at epoch 21
    cfg.lbfgs_max_iter          = 20
    cfg.lbfgs_max_eval          = 25
    cfg.log_every               = 1     # Print every epoch so we can see progress

    print(f"Loaded config. Saving checkpoints to: {cfg.checkpoint_dir}")

    # Execute the training pipeline
    train_phase1(cfg)
