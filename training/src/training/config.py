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
    latent_dim: int = 16

    # Phase 1: PINN Training (Branch + Trunk)
    p1_batch_size: int = 32
    p1_epochs: int = 5000
    p1_lr: float = 1e-3
    colloc_points_per_batch: int = 2000

    # Phase 1: PDE / Sobolev loss weights
    w_pde: float = 0.10              # PDE residual loss weight
    w_sob: float = 0.01              # Sobolev (∇(∇²φ)) loss weight
    sob_decay: float = 0.999         # Multiplicative decay applied to w_sob each epoch

    # Phase 1: Physics-only epoch settings
    physics_colloc_points: int = 100  # Random colloc pts per sample in physics-only epochs

    # Phase 1: Sobol quasi-random sampling
    use_sobol: bool = True            # Replace uniform random with Sobol sequences

    # Phase 1: Optimizer curriculum (Adam → L-BFGS)
    lbfgs_start_epoch: int = 4000     # Switch from Adam to L-BFGS at this epoch
    lbfgs_max_iter: int = 20          # max_iter per L-BFGS step() call
    lbfgs_max_eval: int = 25          # max_eval per L-BFGS step() call

    # Phase 1: torch.compile acceleration (PyTorch >= 2.0)
    use_compile: bool = True          # Wrap model with torch.compile if available

    # Logging
    log_every: int = 100             # Print summary every N epochs

    # Phase 2: Coefficient Training (MLP Head)
    p2_batch_size: int = 64
    p2_epochs: int = 2000
    p2_lr: float = 1e-3

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
