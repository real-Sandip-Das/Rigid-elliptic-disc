import torch
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DeepONetWaveSurrogate(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        self.branch = nn.Sequential(
            nn.Linear(3, 16), SiLU(),
            nn.Linear(16, 16), SiLU(),
            nn.Linear(16, latent_dim),
        )

        # Input: 7 features (r, cos_t, sin_t, z, sin(z), sin(2z), sin(3z))
        self.trunk = nn.Sequential(
            nn.Linear(7, 32), SiLU(),
            nn.Linear(32, 32), SiLU(),
            nn.Linear(32, 32), SiLU(),
            nn.Linear(32, latent_dim * 2),
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(latent_dim, 16), SiLU(),
            nn.Linear(16, 2),
        )

        self.bias_real = nn.Parameter(torch.zeros(1))
        self.bias_imag = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Core building blocks (used in both epoch types)
    # ------------------------------------------------------------------

    def encode(self, wave_params):
        """
        Branch forward pass only.
        Returns latent of shape [B, latent_dim].

        Call this explicitly so you can:
          - cache the result (detached) before a physics-only epoch
          - keep the gradient live for a data epoch
        """
        return self.branch(wave_params)

    def forward_trunk_only(self, latent, spatial_features):
        """
        Trunk forward pass using a pre-computed branch latent.

        Args:
            latent          : [B, latent_dim]  — may be detached (physics epoch)
                              or live in the graph (data epoch).
            spatial_features: [B, P, 7]        — must be in the grad graph
                              (i.e. computed from x, y, z that have requires_grad=True).

        Returns:
            phi_real, phi_imag : [B, P] each.

        In physics-only epochs `latent` is detached, so no gradient flows to the
        branch; the trunk and bias parameters are updated via the PDE / Sobolev loss.
        In data epochs `latent` is live, so both branch and trunk are updated.
        """
        basis = self.trunk(spatial_features)                  # [B, P, latent_dim*2]
        basis_real, basis_imag = torch.chunk(basis, 2, dim=-1)

        phi_real = torch.einsum('bd,bpd->bp', latent, basis_real) + self.bias_real
        phi_imag = torch.einsum('bd,bpd->bp', latent, basis_imag) + self.bias_imag

        return phi_real, phi_imag

    # ------------------------------------------------------------------
    # Convenience: full forward (encode + trunk)
    # ------------------------------------------------------------------

    def forward(self, wave_params, spatial_features):
        return self.forward_trunk_only(self.encode(wave_params), spatial_features)

    # ------------------------------------------------------------------
    # Phase-2 coefficient head
    # ------------------------------------------------------------------

    def predict_coeffs(self, wave_params):
        return self.mlp_head(self.encode(wave_params))

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def freeze_for_phase1(self):
        """Phase 1: train branch + trunk + biases; freeze MLP head."""
        for p in self.branch.parameters():   p.requires_grad = True
        for p in self.trunk.parameters():    p.requires_grad = True
        self.bias_real.requires_grad = True
        self.bias_imag.requires_grad = True
        for p in self.mlp_head.parameters(): p.requires_grad = False

    def freeze_for_phase2(self):
        """Phase 2: freeze everything except MLP head."""
        for p in self.branch.parameters():   p.requires_grad = False
        for p in self.trunk.parameters():    p.requires_grad = False
        self.bias_real.requires_grad = False
        self.bias_imag.requires_grad = False
        for p in self.mlp_head.parameters(): p.requires_grad = True
