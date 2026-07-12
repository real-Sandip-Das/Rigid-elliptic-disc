import math
import time
import os
import dataclasses
import wandb

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download, HfApi

from training.model import DeepONetWaveSurrogate
from training.data import get_dataloaders, generate_anchor_coords

def calc_first_deriv(phi, x, y):
    """Computes ∂φ/∂x and ∂φ/∂y."""
    ones = torch.ones_like(phi)
    
    dphi_dx = torch.autograd.grad(
        phi, x, grad_outputs=ones,
        create_graph=True, retain_graph=True
    )[0]
    
    dphi_dy = torch.autograd.grad(
        phi, y, grad_outputs=ones,
        create_graph=True, retain_graph=True
    )[0]
    
    return dphi_dx, dphi_dy

def train_phase1(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 1 on {device}  |  data values + derivatives")

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            config=dataclasses.asdict(config),
            name=config.wandb_name if config.wandb_name else "phase1"
        )

    lbfgs_maxiter = getattr(config, 'lbfgs_max_iter', 20)
    lbfgs_maxeval = getattr(config, 'lbfgs_max_eval', 25)
    log_every     = getattr(config, 'log_every', 1)

    train_loader, val_loader = get_dataloaders(config.data_path, config.p1_batch_size)
    model = DeepONetWaveSurrogate(
        latent_dim=config.latent_dim, 
        subnet_width=config.subnet_width,
        fourier_mapping_size=config.fourier_mapping_size,
        fourier_scale=config.fourier_scale
    ).to(device)
    model.freeze_for_phase1()


    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.p1_lr,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=1e-5, patience=100, min_lr=1e-5
    )

    mse_loss = nn.MSELoss()
    start_epoch = 1

    alpha_lra = 0.1
    lambda_d = 1.0
    prev_avg_v = float('inf')  # LRA gated until value error is low enough

    for epoch in range(start_epoch, config.p1_epochs + 1):
        t0 = time.time()

        total_loss = 0.0
        total_loss_v = 0.0
        total_loss_d = 0.0
        total_grad_norm = 0.0
        n_batches  = 0

        for batch in train_loader:
            wave_params = batch[0].to(device)
            phi_r_true, phi_i_true = batch[1].to(device), batch[2].to(device)
            dx_r_true, dx_i_true = batch[3].to(device), batch[4].to(device)
            dy_r_true, dy_i_true = batch[5].to(device), batch[6].to(device)
            
            a_b = wave_params[:, 0]

            x, y = generate_anchor_coords(a_b)
            # x and y have shape [B, 25] now (from generate_anchor_coords)
            x = x.to(device)
            y = y.to(device)
            x.requires_grad_(True)
            y.requires_grad_(True)

            optimizer.zero_grad()

            latent = model.encode(wave_params)
            phi_r_pred, phi_i_pred = model.forward_trunk_only(latent, x, y, a_b)

            # Value Loss (Relative/Normalized MSE)
            loss_v = (
                mse_loss(phi_r_pred, phi_r_true) / (torch.mean(phi_r_true**2) + 1e-8) +
                mse_loss(phi_i_pred, phi_i_true) / (torch.mean(phi_i_true**2) + 1e-8)
            )

            dphi_r_dx, dphi_r_dy = calc_first_deriv(phi_r_pred, x, y)
            dphi_i_dx, dphi_i_dy = calc_first_deriv(phi_i_pred, x, y)
            
            mask = torch.isfinite(dx_r_true)
            if mask.any():
                loss_d = (
                    torch.mean((dphi_r_dx[mask] - dx_r_true[mask])**2) / (torch.mean(dx_r_true[mask]**2) + 1e-8) +
                    torch.mean((dphi_r_dy[mask] - dy_r_true[mask])**2) / (torch.mean(dy_r_true[mask]**2) + 1e-8) +
                    torch.mean((dphi_i_dx[mask] - dx_i_true[mask])**2) / (torch.mean(dx_i_true[mask]**2) + 1e-8) +
                    torch.mean((dphi_i_dy[mask] - dy_i_true[mask])**2) / (torch.mean(dy_i_true[mask]**2) + 1e-8)
                )
            else:
                loss_d = 0.0

            if mask.any() and prev_avg_v < config.lra_warmup_threshold:
                last_layer_weights = [model.trunk[-1].weight]
                grads_val = torch.autograd.grad(loss_v, last_layer_weights, retain_graph=True, allow_unused=True)[0]
                grads_der = torch.autograd.grad(loss_d, last_layer_weights, retain_graph=True, allow_unused=True)[0]
                
                max_grad_v = torch.max(torch.abs(grads_val)).item() if grads_val is not None else 0.0
                mean_grad_d = torch.mean(torch.abs(grads_der)).item() if grads_der is not None else 0.0
                
                if mean_grad_d > 0:
                    lambda_hat = max(1.0, max_grad_v / mean_grad_d)
                    lambda_d = (1.0 - alpha_lra) * lambda_d + alpha_lra * lambda_hat

            loss = loss_v + float(lambda_d) * loss_d

            loss.backward()
            b_norm = torch.nn.utils.clip_grad_norm_(model.branch.parameters(), max_norm=config.grad_clip_norm)
            t_norm = torch.nn.utils.clip_grad_norm_(model.trunk.parameters(), max_norm=config.grad_clip_norm)
            batch_grad_norm = float((b_norm**2 + t_norm**2)**0.5)
            optimizer.step()

            step_loss = loss.item()
            step_loss_v = loss_v.item()
            step_loss_d = loss_d.item() if isinstance(loss_d, torch.Tensor) else loss_d
            step_grad_norm = batch_grad_norm

            total_loss += step_loss
            total_loss_v += step_loss_v
            total_loss_d += step_loss_d
            total_grad_norm += step_grad_norm

            n_batches += 1
            
            if config.use_wandb:
                wandb.log({
                    "batch/loss": step_loss,
                    "batch/rel_err": step_loss_v,
                    "batch/grad_norm": step_grad_norm,
                    "batch/lambda_d": float(lambda_d)
                })

        avg = total_loss / max(n_batches, 1)
        avg_v = total_loss_v / max(n_batches, 1)
        avg_d = total_loss_d / max(n_batches, 1)
        avg_grad_norm = total_grad_norm / max(n_batches, 1)
        prev_avg_v = avg_v  # Used to gate LRA next epoch
        
        # Track unweighted sum of losses so LR doesn't drop while model is learning derivatives
        scheduler.step(avg_v + avg_d)

        if epoch % log_every == 0:
            epoch_time = time.time() - t0
            print(
                f"Epoch {epoch:5d}/{config.p1_epochs} | "
                f"opt=Adam train loss: {avg:.6f} | "
                f"train rel err: {avg_v:.6f} | "
                f"grad norm: {avg_grad_norm:.4f} | "
                f"time: {epoch_time:.2f}s"
            )

            if config.use_wandb:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "epoch": epoch,
                    "train/loss": avg,
                    "train/rel_err": avg_v,
                    "train/grad_norm": avg_grad_norm,
                    "train/lambda_d": float(lambda_d),
                    "train/log10_lr": math.log10(current_lr),
                    "train/epoch_time_s": epoch_time
                })

        if epoch % max(1, log_every * 10) == 0:
            model.eval()
            val_loss = 0.0
            val_loss_v = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    wave_params = batch[0].to(device)
                    phi_r_true, phi_i_true = batch[1].to(device), batch[2].to(device)
                    dx_r_true, dx_i_true = batch[3].to(device), batch[4].to(device)
                    dy_r_true, dy_i_true = batch[5].to(device), batch[6].to(device)
                    
                    a_b = wave_params[:, 0]
                    x, y = generate_anchor_coords(a_b)
                    x = x.to(device)
                    y = y.to(device)
                    # We need requires_grad for validation to compute derivatives
                    x.requires_grad_(True)
                    y.requires_grad_(True)

                    latent = model.encode(wave_params)
                    with torch.enable_grad():
                        phi_r_pred, phi_i_pred = model.forward_trunk_only(latent, x, y, a_b)
                        
                        # Value Loss (Relative/Normalized MSE)
                        loss_v = (
                            mse_loss(phi_r_pred, phi_r_true) / (torch.mean(phi_r_true**2) + 1e-8) +
                            mse_loss(phi_i_pred, phi_i_true) / (torch.mean(phi_i_true**2) + 1e-8)
                        )
                        
                        dphi_r_dx, dphi_r_dy = calc_first_deriv(phi_r_pred, x, y)
                        dphi_i_dx, dphi_i_dy = calc_first_deriv(phi_i_pred, x, y)
                        
                    mask = torch.isfinite(dx_r_true)
                    if mask.any():
                        loss_d = (
                            torch.mean((dphi_r_dx[mask] - dx_r_true[mask])**2) / (torch.mean(dx_r_true[mask]**2) + 1e-8) +
                            torch.mean((dphi_r_dy[mask] - dy_r_true[mask])**2) / (torch.mean(dy_r_true[mask]**2) + 1e-8) +
                            torch.mean((dphi_i_dx[mask] - dx_i_true[mask])**2) / (torch.mean(dx_i_true[mask]**2) + 1e-8) +
                            torch.mean((dphi_i_dy[mask] - dy_i_true[mask])**2) / (torch.mean(dy_i_true[mask]**2) + 1e-8)
                        )
                    else:
                        loss_d = 0.0
                        
                    val_loss += (loss_v + float(lambda_d) * loss_d).item()
                    val_loss_v += loss_v.item()

            avg_val_loss = val_loss / max(1, len(val_loader))
            avg_val_loss_v = val_loss_v / max(1, len(val_loader))
            print(f"Epoch {epoch:5d}/{config.p1_epochs} | Validation loss: {avg_val_loss:.6f} | Relative Value Error: {avg_val_loss_v:.6f}")
            
            if config.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "val/loss": avg_val_loss,
                    "val/rel_err": avg_val_loss_v
                })
                
            model.train()

        # Early stop if validation starts plateauing at very low values can be added here if needed

    torch.save(model.state_dict(), config.phase1_model_path)
    print(f"Phase 1 complete. Model saved → {config.phase1_model_path}")
    
    if config.use_wandb:
        wandb.save(config.phase1_model_path)
        wandb.finish()
