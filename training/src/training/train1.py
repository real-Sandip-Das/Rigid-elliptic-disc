import math
import time
import os

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download, HfApi

from training.model import DeepONetWaveSurrogate
from training.data import get_dataloaders


# ============================================================
# Sobol quasi-random sequence helper
# ============================================================

class SobolSampler:
    """
    Thin wrapper around torch.quasirandom.SobolEngine.

    Draws quasi-random samples in [0, 1)^d for a given dimensionality.
    Falls back to uniform random if SobolEngine is unavailable.

    Sobol sequences have much lower discrepancy than uniform random, which
    reduces variance of the 3rd-order Sobolev loss estimate significantly.
    """

    def __init__(self, dimension: int, scramble: bool = True, device: torch.device = None):
        self.dimension = dimension
        self.device    = device
        try:
            self.engine = torch.quasirandom.SobolEngine(dimension=dimension, scramble=scramble)
            self.available = True
        except Exception:
            self.available = False

    def draw(self, n: int) -> torch.Tensor:
        """Returns [n, d] tensor in [0, 1)^d on the configured device."""
        if self.available:
            return self.engine.draw(n).to(self.device)
        return torch.rand(n, self.dimension, device=self.device)

    def reset(self):
        """Reset the engine so the next draw starts a fresh Sobol sequence."""
        if self.available:
            self.engine.reset()


# ============================================================
# Coordinate generators
# ============================================================

def generate_data_and_colloc_coords(a_b, d_b, num_colloc_points, device, sobol_sampler=None):
    """
    25 fixed anchor points  +  `num_colloc_points` collocation points.

    Collocation points are drawn from a Sobol quasi-random sequence
    when `sobol_sampler` is provided, otherwise uniform random is used.

    Returns x, y, z of shape [B, 25 + num_colloc_points], each a leaf tensor
    with requires_grad=True so autograd can differentiate φ w.r.t. position.

    Used in DATA+PHYSICS epochs only.
    """
    batch_size = a_b.shape[0]

    # ---- Anchor points: 1 centre + 3 rings × 8 angles = 25 ----
    s_vals, alpha_vals = [0.0], [0.0]
    for s in np.linspace(0, 1, 4)[1:]:                          # s ∈ {1/3, 2/3, 1}
        for alpha in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            s_vals.append(s)
            alpha_vals.append(alpha)

    s_t = (torch.tensor(s_vals,     dtype=torch.float32, device=device)
           .unsqueeze(0).expand(batch_size, -1))                 # [B, 25]
    a_t = (torch.tensor(alpha_vals, dtype=torch.float32, device=device)
           .unsqueeze(0).expand(batch_size, -1))                 # [B, 25]

    x_anchor = a_b.unsqueeze(1) * s_t * torch.cos(a_t)
    y_anchor =             1.0  * s_t * torch.sin(a_t)
    z_anchor = -d_b.unsqueeze(1).expand(batch_size, 25)

    # ---- Collocation points (Sobol or uniform) ----
    if sobol_sampler is not None:
        # Sobol draws: 3 dimensions → [num_colloc_points, 3] in [0,1)
        # Dimensions: (s_rand, alpha_rand, z_frac)
        raw = sobol_sampler.draw(num_colloc_points)              # [P, 3]
        s_rand     = raw[:, 0].unsqueeze(0).expand(batch_size, -1) * 3.0          # [B, P]
        alpha_rand = raw[:, 1].unsqueeze(0).expand(batch_size, -1) * 2 * math.pi  # [B, P]
        z_frac     = raw[:, 2].unsqueeze(0).expand(batch_size, -1)                 # [B, P] in [0,1)
    else:
        s_rand     = torch.rand((batch_size, num_colloc_points), device=device) * 3.0
        alpha_rand = torch.rand((batch_size, num_colloc_points), device=device) * 2 * math.pi
        z_frac     = torch.rand((batch_size, num_colloc_points), device=device)

    z_rand = -d_b.unsqueeze(1) * z_frac

    x_colloc = a_b.unsqueeze(1) * s_rand * torch.cos(alpha_rand)
    y_colloc =             1.0  * s_rand * torch.sin(alpha_rand)

    # Concatenate and mark as autograd leaves
    x = torch.cat([x_anchor, x_colloc], dim=1).requires_grad_(True)  # [B, 25+P]
    y = torch.cat([y_anchor, y_colloc], dim=1).requires_grad_(True)
    z = torch.cat([z_anchor, z_rand  ], dim=1).requires_grad_(True)

    return x, y, z


def generate_colloc_coords(a_b, d_b, num_points, device, sobol_sampler=None):
    """
    `num_points` collocation points — NO anchor points.

    Used in PHYSICS-ONLY epochs.  All points are sampled uniformly inside
    the elliptic disc footprint: (x/a_b)² + y² ≤ 1, at the fixed disc
    plane depth z = -d_b (no z sampling).

    Uniform area sampling over an ellipse requires the sqrt-transform:
        u ~ Uniform[0, 1)  →  r = √u
    so that the density is uniform over the disc area (otherwise points
    cluster near the centre with a naive uniform-r draw).

    Sobol sequences give better low-discrepancy coverage than uniform random,
    which reduces variance of the 3rd-order Sobolev loss estimate.

    Returns x, y, z of shape [B, num_points], each a leaf with requires_grad=True.
    """
    batch_size = a_b.shape[0]

    if sobol_sampler is not None:
        raw        = sobol_sampler.draw(num_points)                          # [P, 2] — only 2 dims now
        # dim 0: u ∈ [0,1) → r = √u for uniform disc-area distribution
        r_frac     = raw[:, 0].sqrt().unsqueeze(0).expand(batch_size, -1)    # [B, P]
        alpha_rand = raw[:, 1].unsqueeze(0).expand(batch_size, -1) * 2 * math.pi
    else:
        u_rand     = torch.rand((batch_size, num_points), device=device)
        r_frac     = u_rand.sqrt()                                            # uniform over disc area
        alpha_rand = torch.rand((batch_size, num_points), device=device) * 2 * math.pi

    # x, y lie on/inside the ellipse with semi-axes (a_b, 1.0)
    # z is fixed at the disc plane depth (-d_b) — not sampled
    x = (a_b.unsqueeze(1) * r_frac * torch.cos(alpha_rand)).requires_grad_(True)
    y = (            1.0  * r_frac * torch.sin(alpha_rand)).requires_grad_(True)
    z = (-d_b.unsqueeze(1).expand(batch_size, num_points).clone()).requires_grad_(True)

    return x, y, z


# ============================================================
# PDE and Sobolev losses
# ============================================================

def calc_laplace(phi, x, y, z):
    """
    ∇²φ = φ_xx + φ_yy + φ_zz  via autograd.

    create_graph=True  — keeps the computational graph so we can
                         differentiate the Laplacian again (Sobolev loss).
    retain_graph=True  — (implied by create_graph) lets us call grad()
                         multiple times on the same graph without freeing it.
    """
    ones = torch.ones_like(phi)

    def second_deriv(scalar, v):
        d1 = torch.autograd.grad(
            scalar, v, grad_outputs=ones,
            create_graph=True, retain_graph=True,
        )[0]
        d2 = torch.autograd.grad(
            d1, v, grad_outputs=torch.ones_like(d1),
            create_graph=True, retain_graph=True,
        )[0]
        return d2

    return second_deriv(phi, x) + second_deriv(phi, y) + second_deriv(phi, z)


def calc_dz(phi, z):
    """Computes ∂φ/∂z."""
    return torch.autograd.grad(
        phi, z,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
    )[0]


def compute_sobolev_loss(laplace_r, laplace_i, x, y, z):
    """
    Sobolev term: ||∇(∇²φ_r)||² + ||∇(∇²φ_i)||²

    Penalises spatial non-smoothness of the PDE residual — if the Laplacian
    is small *and* its gradient is small the solution is globally smooth.
    This is the one extra derivative over the plain Laplace PDE loss.
    """
    def grad_sq_norm(field, wrt):
        g = torch.autograd.grad(
            field.sum(), wrt,
            create_graph=True, retain_graph=True,
        )[0]
        return (g ** 2).mean()

    loss_r = sum(grad_sq_norm(laplace_r, v) for v in (x, y, z))
    loss_i = sum(grad_sq_norm(laplace_i, v) for v in (x, y, z))
    return loss_r + loss_i


# ============================================================
# Training
# ============================================================

def train_phase1(config):
    """
    Alternating-epoch training with the following improvements:

    1. Sobol quasi-random sampling
       Collocation points use Sobol sequences (low-discrepancy) instead of
       uniform random. This reduces variance of the 3rd-order Sobolev loss
       estimate and gives better domain coverage.

    2. Sobolev weight annealing
       w_sob is decayed by `config.sob_decay` each epoch so the 3rd-order
       derivative penalty is strong early (encourages global smoothness) and
       soft later (lets Adam / L-BFGS fine-tune the PDE residual).

    3. Adam → L-BFGS optimizer curriculum
       Epochs < lbfgs_start_epoch : Adam (exploration, large steps).
       Epochs ≥ lbfgs_start_epoch : L-BFGS with strong-Wolfe line search
           (fine-tuning). Sobolev smoothness makes the loss landscape convex
           enough for L-BFGS to converge rapidly at this stage.

    4. torch.compile (PyTorch ≥ 2.0)
       Wraps the model with torch.compile for graph fusion and kernel
       optimisation. Automatically skipped on older PyTorch versions.

    Epoch structure:
      Odd epochs  (1, 3, 5, …) — DATA + PHYSICS
        • Full forward: branch encodes wave_params → latent → trunk evaluates features.
        • Loss = data loss (25 anchor pts) + PDE loss (all pts) + Sobolev loss (all pts).
        • Gradients flow to both branch and trunk.
        • Caches each batch's (latent, a_b, d_b) for the next physics epoch.


    Config fields (all have defaults in WaveConfig):
        config.w_pde                 = 0.10   — PDE residual loss weight
        config.w_sob                 = 0.01   — initial Sobolev loss weight
        config.sob_decay             = 0.999  — per-epoch w_sob decay factor
        config.use_sobol             = True   — use Sobol quasi-random sampling
        config.lbfgs_start_epoch     = 4000   — switch to L-BFGS at this epoch
        config.lbfgs_max_iter        = 20     — L-BFGS max_iter per step()
        config.lbfgs_max_eval        = 25     — L-BFGS max_eval per step()
        config.use_compile           = True   — wrap model with torch.compile
        config.log_every             = 100    — print every N epochs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 1 on {device}  |  every epoch = data+physics")

    # Pull config fields with backward-compatible defaults
    w_pde_init    = getattr(config, 'w_pde',                 1e-4)
    w_sob_init    = getattr(config, 'w_sob_init',            1e-6)
    w_sob_max     = getattr(config, 'w_sob_max',             1e-4)
    sob_growth    = getattr(config, 'sob_growth',            1.01)

    use_sobol     = getattr(config, 'use_sobol',             True)
    lbfgs_maxiter = getattr(config, 'lbfgs_max_iter',          50)
    lbfgs_maxeval = getattr(config, 'lbfgs_max_eval',          25)
    use_compile   = getattr(config, 'use_compile',           True)
    log_every     = getattr(config, 'log_every',              100)

    train_loader, val_loader = get_dataloaders(config.data_path, config.p1_batch_size)
    model      = DeepONetWaveSurrogate(latent_dim=config.latent_dim).to(device)
    model.freeze_for_phase1()

    # ── torch.compile (PyTorch ≥ 2.0) ────────────────────────────────────
    # Graph fusion and kernel optimisation — reduces the overhead of
    # repeated 3rd-order autograd traversals.  Silently skipped on older
    # PyTorch where compile() is unavailable.
    if use_compile and hasattr(torch, 'compile'):
        print("torch.compile: enabled (graph fusion for 3rd-order AD)")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed, continuing without: {e}")
    else:
        print("torch.compile: disabled (requires PyTorch >= 2.0)")

    # ── Sobol sampler ─────────────────────────────────────────────────────
    # Dimension 3: (s, alpha, z_fraction) for each collocation draw.
    if use_sobol:
        print(f"Sobol quasi-random sampling: enabled")
        sobol_data = SobolSampler(dimension=3, scramble=True, device=device)
    else:
        print("Sobol quasi-random sampling: disabled (using uniform random)")
        sobol_data = None

    # ── Optimizer setup ───────────────────────────────────────────────────
    # We start with Adam and switch to L-BFGS automatically when LR drops.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.p1_lr,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        min_lr=1e-7
    )
    
    using_lbfgs = False   # flag to track current optimizer

    def make_lbfgs():
        return torch.optim.LBFGS(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.1,  # Lowered from 1.0 to prevent aggressive steps causing NaNs
            max_iter=lbfgs_maxiter,
            max_eval=lbfgs_maxeval,
            line_search_fn='strong_wolfe',
        )

    mse_loss = nn.MSELoss()

    start_epoch = 1

    # ── Hugging Face Hub Integration (Init/Resume) ────────────────────────
    hf_repo_id = getattr(config, 'hf_repo_id', "")
    hf_checkpoint_file = getattr(config, 'hf_checkpoint_file', "phase1_training_state.pt")
    hf_force_initialize = getattr(config, 'hf_force_initialize', False)
    hf_sync_every = getattr(config, 'hf_sync_every', 0)
    checkpoint_loaded = False

    api = None
    if hf_repo_id:
        api = HfApi()
        
        # Ensure remote repo exists
        try:
            api.create_repo(repo_id=hf_repo_id, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not verify/create remote repository: {e}")
            
        if not hf_force_initialize:
            try:
                if api.repo_info(repo_id=hf_repo_id):
                    print(f"Pulling latest checkpoint from Hugging Face Hub ({hf_repo_id})...")
                    snapshot_download(
                        repo_id=hf_repo_id,
                        local_dir=config.checkpoint_dir,
                        allow_patterns=[hf_checkpoint_file]
                    )
                    
                    checkpoint_path = os.path.join(config.checkpoint_dir, hf_checkpoint_file)
                    if os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        using_lbfgs = checkpoint.get('using_lbfgs', False)
                        
                        if using_lbfgs:
                            optimizer = make_lbfgs()
                        
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        
                        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None and not using_lbfgs:
                            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                            
                        start_epoch = checkpoint['epoch'] + 1
                        checkpoint_loaded = True
                        print(f"Successfully resumed from Hub. Starting from epoch {start_epoch}")
                    else:
                        print("Repository found, but no checkpoint file existed yet.")
            except Exception as e:
                print(f"Could not pull from Hub ({str(e)}). Falling back to initialization.")

    if hf_repo_id and not checkpoint_loaded:
        print("Initializing model and optimizer from scratch (or force initialized).")

    for epoch in range(start_epoch, config.p1_epochs + 1):
        t0 = time.time()

        # ── Sobolev weight scheduling ──────────────────────────────────────
        # Starts small for a better initial loss landscape, then slowly
        # increases to enforce smoothness once a good approximation is found.
        w_sob_epoch = min(w_sob_max, w_sob_init * (sob_growth ** (epoch - 1)))
        w_pde       = w_pde_init

        # ── Optimizer curriculum: Adam → L-BFGS ──────────────────────────
        if not using_lbfgs:
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr <= 1.5e-7:
                print(f"\n[Epoch {epoch}] Learning rate reached {current_lr:.2e}. Switching optimizer: Adam → L-BFGS "
                      f"(max_iter={lbfgs_maxiter}, strong_wolfe line search)")
                optimizer   = make_lbfgs()
                using_lbfgs = True

        # ── Reset Sobol engine each epoch so colloc pts vary per epoch ────
        if use_sobol:
            sobol_data.reset()

        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            wave_params, phi_r_true, phi_i_true, _ = [b.to(device) for b in batch]
            a_b = wave_params[:, 0]
            d_b = wave_params[:, 1]

            # --- Spatial coordinates ---
            x, y, z  = generate_data_and_colloc_coords(
                a_b, d_b, config.colloc_points_per_batch, device,
                sobol_sampler=sobol_data,
            )

            if using_lbfgs:
                # ── L-BFGS path: requires a closure ──────────────────────
                # x, y, z are fixed leaf tensors (same coords for all
                # line-search evaluations — correct L-BFGS behaviour).
                # features, phi and laplacian MUST be recomputed inside
                # the closure because backward() frees saved tensors after
                # the first call; a second call would hit a graph error.
                # We also zero x/y/z grads each call so they don't
                # accumulate across line-search steps (harmless but clean).

                def closure():
                    optimizer.zero_grad()
                    for _t in (x, y, z):
                        if _t.grad is not None:
                            _t.grad.zero_()
                    # Rebuild full graph from fixed x, y, z each call
                    _latent = model.encode(wave_params)
                    phi_r_pred, phi_i_pred = model.forward_trunk_only(_latent, x, y, z, a_b, d_b)

                    _loss_data = (
                        mse_loss(phi_r_pred[:, :25], phi_r_true)
                        + mse_loss(phi_i_pred[:, :25], phi_i_true)
                    )
                    _laplace_r = calc_laplace(phi_r_pred, x, y, z)
                    _laplace_i = calc_laplace(phi_i_pred, x, y, z)
                    _loss_pde  = _laplace_r.pow(2).mean() + _laplace_i.pow(2).mean()

                    # Boundary condition at z = -d_b (anchor points are [:, :25])
                    _d_phi_r_dz = calc_dz(phi_r_pred, z)
                    _d_phi_i_dz = calc_dz(phi_i_pred, z)
                    _loss_bc_r = (_d_phi_r_dz[:, :25] - 1.0).pow(2).mean()
                    _loss_bc_i = _d_phi_i_dz[:, :25].pow(2).mean()
                    _loss_pde += _loss_bc_r + _loss_bc_i

                    _loss_sob  = compute_sobolev_loss(_laplace_r, _laplace_i, x, y, z)

                    _loss = _loss_data + w_pde * _loss_pde + w_sob_epoch * _loss_sob
                    
                    if not torch.isfinite(_loss):
                        # Force line search to shrink step size and reject this step
                        optimizer.zero_grad()
                        return torch.tensor(float('inf'), device=_loss.device)

                    _loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    return _loss

                loss_val = optimizer.step(closure)
                total_loss += loss_val.item() if loss_val is not None else 0.0

            else:
                # ── Adam path ─────────────────────────────────────────────
                optimizer.zero_grad()

                latent = model.encode(wave_params)          # [B, latent_dim]

                # Trunk (gradient live → trunk params updated)
                phi_r_pred, phi_i_pred = model.forward_trunk_only(latent, x, y, z, a_b, d_b)

                # Data loss: anchor points only
                loss_data = (
                    mse_loss(phi_r_pred[:, :25], phi_r_true)
                    + mse_loss(phi_i_pred[:, :25], phi_i_true)
                )

                # PDE loss: all points
                laplace_r = calc_laplace(phi_r_pred, x, y, z)
                laplace_i = calc_laplace(phi_i_pred, x, y, z)
                loss_pde  = laplace_r.pow(2).mean() + laplace_i.pow(2).mean()

                # Boundary condition at z = -d_b (anchor points are [:, :25])
                d_phi_r_dz = calc_dz(phi_r_pred, z)
                d_phi_i_dz = calc_dz(phi_i_pred, z)
                loss_bc_r = (d_phi_r_dz[:, :25] - 1.0).pow(2).mean()
                loss_bc_i = d_phi_i_dz[:, :25].pow(2).mean()
                loss_pde += loss_bc_r + loss_bc_i


                # Sobolev loss: ∇(∇²φ) on all points
                loss_sob  = compute_sobolev_loss(laplace_r, laplace_i, x, y, z)

                loss = loss_data + w_pde * loss_pde + w_sob_epoch * loss_sob

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        
        if not using_lbfgs:
            scheduler.step(avg)

        if epoch % log_every == 0:
            opt_tag = "L-BFGS" if using_lbfgs else "Adam"
            epoch_time = time.time() - t0
            print(
                f"Epoch {epoch:5d}/{config.p1_epochs}  [data+phys]  "
                f"opt={opt_tag}  w_sob={w_sob_epoch:.2e}  avg loss: {avg:.6f}  "
                f"time: {epoch_time:.2f}s"
            )

        if epoch % 100 == 0:
            model.eval()
            total_val_data_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    wave_params, phi_r_true, phi_i_true, _ = [b.to(device) for b in batch]
                    a_b = wave_params[:, 0]
                    d_b = wave_params[:, 1]

                    latent = model.encode(wave_params)
                    x, y, z  = generate_data_and_colloc_coords(
                        a_b, d_b, 0, device,
                        sobol_sampler=None,
                    )
                    phi_r_pred, phi_i_pred = model.forward_trunk_only(latent, x, y, z, a_b, d_b)
                    
                    loss_data = (
                        mse_loss(phi_r_pred[:, :25], phi_r_true)
                        + mse_loss(phi_i_pred[:, :25], phi_i_true)
                    )
                    total_val_data_loss += loss_data.item()
            model.train()
            val_loss_avg = total_val_data_loss / max(1, len(val_loader))
            print(f"Epoch {epoch:5d}/{config.p1_epochs}  [validation]  Val Data Loss: {val_loss_avg:.6f}")

        # ── Mid-training Hugging Face Sync ────────────────────────────────────
        if hf_repo_id and hf_sync_every > 0 and epoch % hf_sync_every == 0:
            save_path = os.path.join(config.checkpoint_dir, hf_checkpoint_file)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if not using_lbfgs else None,
                'using_lbfgs': using_lbfgs
            }, save_path)
            
            print(f"Syncing epoch {epoch} checkpoint to Hub ({hf_repo_id})...")
            api.upload_folder(
                folder_path=config.checkpoint_dir,
                repo_id=hf_repo_id,
                commit_message=f"Checkpoint sync: Epoch {epoch}"
            )

    torch.save(model.state_dict(), config.phase1_model_path)
    print(f"Phase 1 complete. Model saved → {config.phase1_model_path}")

    # ── Post-Training: Save and Upload Final State ───────────────────────────
    if hf_repo_id:
        print("Training finished. Preparing final storage upload to Hugging Face Hub...")
        final_save_path = os.path.join(config.checkpoint_dir, hf_checkpoint_file)
        torch.save({
            'epoch': config.p1_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if not using_lbfgs else None,
            'using_lbfgs': using_lbfgs,
            'status': 'completed'
        }, final_save_path)
        
        api.upload_folder(
            folder_path=config.checkpoint_dir,
            repo_id=hf_repo_id,
            commit_message="Training complete: Final model state stored."
        )
        print("Final checkpoint successfully stored on Hugging Face Hub!")
