import os
from dataclasses import dataclass


@dataclass
class WaveConfig:
    base_dir: str = "."
    data_path: str = os.path.join(base_dir, "data", "full_pinn_dataset.csv")
    checkpoint_dir: str = os.path.join(base_dir, "checkpoints")

    phase1_model_path: str = os.path.join(checkpoint_dir, "phase1_pinn.pth")
    phase2_model_path: str = os.path.join(checkpoint_dir, "phase2_final.pth")

    latent_dim: int = 256
    subnet_width: int = 64
    fourier_mapping_size: int = 128
    fourier_scale: float = 2.0

    seed: int = 42
    p1_batch_size: int = 32
    p1_epochs: int = 5000
    p1_lr: float = 1e-3
    colloc_points_per_batch: int = 2000
    grad_clip_norm: float = 1.0
    lra_warmup_threshold: float = 0.5

    w_sob_init: float = 1e-6
    w_sob_max: float = 1e-4
    sob_growth: float = 1.01

    use_sobol: bool = True

    use_compile: bool = True

    log_every: int = 100

    hf_repo_id: str = ""
    hf_checkpoint_file: str = "phase1_training_state.pt"
    hf_force_initialize: bool = False
    hf_sync_every: int = 0

    p2_batch_size: int = 64
    p2_epochs: int = 2000
    p2_lr: float = 1e-3

    use_wandb: bool = False
    wandb_project: str = "rigid-elliptic-disc"
    wandb_name: str = ""

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
