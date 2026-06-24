import math

import numpy as np
import torch
import torch.nn as nn

from training.model import DeepONetWaveSurrogate
from training.data import get_dataloader


# ============================================================
# Coordinate generators
# ============================================================

def generate_data_and_colloc_coords(a_b, d_b, num_colloc_points, device):
    """
    25 fixed anchor points  +  `num_colloc_points` random collocation points.

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

    # ---- Random collocation points ----
    s_rand     = torch.rand((batch_size, num_colloc_points), device=device) * 3.0
    alpha_rand = torch.rand((batch_size, num_colloc_points), device=device) * 2 * math.pi
    z_rand     = -d_b.unsqueeze(1) * torch.rand((batch_size, num_colloc_points), device=device)

    x_colloc = a_b.unsqueeze(1) * s_rand * torch.cos(alpha_rand)
    y_colloc =             1.0  * s_rand * torch.sin(alpha_rand)

    # Concatenate and mark as autograd leaves
    x = torch.cat([x_anchor, x_colloc], dim=1).requires_grad_(True)  # [B, 25+P]
    y = torch.cat([y_anchor, y_colloc], dim=1).requires_grad_(True)
    z = torch.cat([z_anchor, z_rand  ], dim=1).requires_grad_(True)

    return x, y, z


def generate_colloc_coords(a_b, d_b, num_points, device):
    """
    `num_points` random collocation points — NO anchor points.

    Used in PHYSICS-ONLY epochs.  We never compute data loss here so the
    fixed anchor locations are irrelevant; purely random sampling is cheaper
    and gives better coverage of the residual landscape.

    Returns x, y, z of shape [B, num_points], each a leaf with requires_grad=True.
    """
    batch_size = a_b.shape[0]

    s_rand     = torch.rand((batch_size, num_points), device=device) * 3.0
    alpha_rand = torch.rand((batch_size, num_points), device=device) * 2 * math.pi
    z_rand     = -d_b.unsqueeze(1) * torch.rand((batch_size, num_points), device=device)

    # Each result is computed from non-grad inputs → becomes a leaf after
    # requires_grad_(True), so autograd roots the graph here.
    x = (a_b.unsqueeze(1) * s_rand * torch.cos(alpha_rand)).requires_grad_(True)
    y = (            1.0  * s_rand * torch.sin(alpha_rand)).requires_grad_(True)
    z = z_rand.requires_grad_(True)

    return x, y, z


# ============================================================
# Feature engineering
# ============================================================

def compute_7_features(x, y, z):
    """
    (x, y, z) of shape [B, P]  →  [B, P, 7] cylindrical Fourier features.

    eps in r prevents NaN in ∂(cos θ)/∂x and ∂(sin θ)/∂y at the origin.
    The computation graph flows through here, so gradients of φ w.r.t.
    x, y, z propagate correctly through these features.
    """
    r     = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
    cos_t = x / r
    sin_t = y / r

    return torch.stack(
        [r, cos_t, sin_t, z, torch.sin(z), torch.sin(2 * z), torch.sin(3 * z)],
        dim=-1,
    )   # [B, P, 7]


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
    Alternating-epoch training:

      Odd epochs  (1, 3, 5, …) — DATA + PHYSICS
        • Full forward: branch encodes wave_params → latent → trunk evaluates features.
        • Loss = data loss (25 anchor pts) + PDE loss (all pts) + Sobolev loss (all pts).
        • Gradients flow to both branch and trunk.
        • Caches each batch's (latent, a_b, d_b) for the next physics epoch.

      Even epochs (2, 4, 6, …) — PHYSICS ONLY
        • Branch is SKIPPED — cached latents from the previous data epoch are reused.
        • 100 fresh random collocation points are drawn (no anchors needed).
        • Loss = PDE loss + Sobolev loss only (no data loss).
        • Gradients flow to trunk and biases ONLY (branch is frozen by detachment).

    Config additions (add to your config object, defaults shown):
        config.physics_colloc_points = 100   # random pts per sample in physics epoch
        config.w_pde                 = 0.10  # PDE loss weight
        config.w_sob                 = 0.01  # Sobolev loss weight
        config.log_every             = 100   # print every N epochs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 1 on {device}  |  odd epochs = data+physics, even epochs = physics-only")

    dataloader = get_dataloader(config.data_path, config.p1_batch_size)
    model      = DeepONetWaveSurrogate(latent_dim=config.latent_dim).to(device)
    model.freeze_for_phase1()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.p1_lr,
    )
    mse_loss = nn.MSELoss()

    # Pull optional config fields (backward-compatible defaults)
    w_pde         = getattr(config, 'w_pde',                 0.10)
    w_sob         = getattr(config, 'w_sob',                 0.01)
    physics_n_pts = getattr(config, 'physics_colloc_points',  100)
    log_every     = getattr(config, 'log_every',              100)

    # ----------------------------------------------------------------
    # Latent cache
    #   Populated at the end of every data epoch.
    #   Each entry: (latent [B, D], a_b [B], d_b [B]) — all detached.
    #   Consumed (read-only) during the following physics epoch.
    # ----------------------------------------------------------------
    latent_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for epoch in range(1, config.p1_epochs + 1):
        is_data_epoch = (epoch % 2 == 1)
        total_loss    = 0.0
        n_batches     = 0

        # ============================================================
        if is_data_epoch:
        # ============================================================
            latent_cache = []   # rebuild cache fresh every data epoch

            for batch in dataloader:
                wave_params, phi_r_true, phi_i_true, _ = [b.to(device) for b in batch]
                a_b = wave_params[:, 0]
                d_b = wave_params[:, 1]

                optimizer.zero_grad()

                # --- Branch (gradient live → branch params updated) ---
                latent = model.encode(wave_params)          # [B, latent_dim]

                # Cache a detached copy so the physics epoch can skip branch
                latent_cache.append((
                    latent.detach().clone(),
                    a_b.detach().clone(),
                    d_b.detach().clone(),
                ))

                # --- Spatial coordinates & features ---
                x, y, z  = generate_data_and_colloc_coords(
                    a_b, d_b, config.colloc_points_per_batch, device,
                )
                features = compute_7_features(x, y, z)     # [B, 25+P, 7]

                # --- Trunk (gradient live → trunk params updated) ---
                phi_r_pred, phi_i_pred = model.forward_trunk_only(latent, features)

                # --- Data loss: anchor points only ---
                loss_data = (
                    mse_loss(phi_r_pred[:, :25], phi_r_true)
                    + mse_loss(phi_i_pred[:, :25], phi_i_true)
                )

                # --- PDE loss: all points ---
                laplace_r = calc_laplace(phi_r_pred, x, y, z)
                laplace_i = calc_laplace(phi_i_pred, x, y, z)
                loss_pde  = laplace_r.pow(2).mean() + laplace_i.pow(2).mean()

                # --- Sobolev loss: ∇(∇²φ) on all points ---
                loss_sob  = compute_sobolev_loss(laplace_r, laplace_i, x, y, z)

                loss = loss_data + w_pde * loss_pde + w_sob * loss_sob

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

        # ============================================================
        else:
        # ============================================================
            if not latent_cache:
                # Guard: can only happen if epoch 1 was not a data epoch
                print(f"[Epoch {epoch}] latent_cache is empty — skipping physics epoch.")
                continue

            for latent, a_b, d_b in latent_cache:
                optimizer.zero_grad()

                # --- Fresh random collocation points (no anchors) ---
                x, y, z  = generate_colloc_coords(a_b, d_b, physics_n_pts, device)
                features = compute_7_features(x, y, z)     # [B, physics_n_pts, 7]

                # --- Trunk only ---
                # `latent` is detached → no gradient reaches branch parameters.
                # Trunk, bias_real, bias_imag ARE updated by this loss.
                phi_r_pred, phi_i_pred = model.forward_trunk_only(latent, features)

                # --- PDE loss ---
                laplace_r = calc_laplace(phi_r_pred, x, y, z)
                laplace_i = calc_laplace(phi_i_pred, x, y, z)
                loss_pde  = laplace_r.pow(2).mean() + laplace_i.pow(2).mean()

                # --- Sobolev loss ---
                loss_sob  = compute_sobolev_loss(laplace_r, laplace_i, x, y, z)

                # No data loss — we have no ground truth at random points
                loss = w_pde * loss_pde + w_sob * loss_sob

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

        if epoch % log_every == 0:
            tag = "data+phys" if is_data_epoch else "phys-only"
            avg = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch:5d}/{config.p1_epochs}  [{tag}]  avg loss: {avg:.6f}")

    torch.save(model.state_dict(), config.phase1_model_path)
    print(f"Phase 1 complete. Model saved → {config.phase1_model_path}")
