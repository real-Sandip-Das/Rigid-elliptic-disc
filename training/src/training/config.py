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

    # Phase 2: Coefficient Training (MLP Head)
    p2_batch_size: int = 64
    p2_epochs: int = 2000
    p2_lr: float = 1e-3

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
