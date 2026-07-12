import os
from dataclasses import dataclass

@dataclass
class WaveConfig:
    # Paths (Relative to the folder from where the script is being run)
    base_dir: str = "."
    data_path: str = os.path.join(base_dir, "data", "full_pinn_dataset.csv")
    checkpoint_dir: str = os.path.join(base_dir, "checkpoints")

    phase1_model_path: str = os.path.join(checkpoint_dir, "phase1_pinn.pth")
    phase2_model_path: str = os.path.join(checkpoint_dir, "phase2_final.pth")

    # Architecture
    latent_dim: int = 256
    subnet_width: int = 64
    fourier_mapping_size: int = 128
    fourier_scale: float = 2.0

    # Phase 1: PINN Training (Branch + Trunk)
    seed: int = 42
    p1_batch_size: int = 32
    p1_epochs: int = 5000
    p1_lr: float = 1e-3
    colloc_points_per_batch: int = 2000
    grad_clip_norm: float = 1.0      # Maximum gradient norm for clipping
    lra_warmup_threshold: float = 0.5 # LRA only activates once rel value error drops below this

    # Phase 1: Sobolev loss weights
    w_sob_init: float = 1e-6         # Initial Sobolev (∇(∇²φ)) loss weight
    w_sob_max: float = 1e-4          # Maximum Sobolev loss weight
    sob_growth: float = 1.01         # Multiplicative growth applied to w_sob each epoch


    # Phase 1: Sobol quasi-random sampling
    use_sobol: bool = True            # Replace uniform random with Sobol sequences


    # Phase 1: torch.compile acceleration (PyTorch >= 2.0)
    use_compile: bool = True          # Wrap model with torch.compile if available

    # Logging
    log_every: int = 100             # Print summary every N epochs

    # Hugging Face Integration
    hf_repo_id: str = ""             # If provided, enables HF Hub synchronization
    hf_checkpoint_file: str = "phase1_training_state.pt" # Name of checkpoint file
    hf_force_initialize: bool = False # If True, bypasses pulling from Hub
    hf_sync_every: int = 0           # Syncs to Hub every N epochs (if > 0)

    # Phase 2: Coefficient Training (MLP Head)
    p2_batch_size: int = 64
    p2_epochs: int = 2000
    p2_lr: float = 1e-3

    # Weights & Biases (W&B) Integration
    use_wandb: bool = False
    wandb_project: str = "rigid-elliptic-disc"
    wandb_name: str = "" # If empty, wandb auto-generates a name

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
