import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Spatial feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def compute_7_features(x, y, z):
    """
    (x, y, z) of shape [B, P]  →  [B, P, 7] cylindrical Fourier features.

    Expects *normalised* coordinates:
        x  = x_physical / a_b   ∈ [-1, 1] inside the ellipse
        y  = y_physical          ∈ [-1, 1] inside the ellipse  (b = 1)
        z  = z_physical / d_b   ∈ [-1, 0] (disc plane to surface)

    eps in r prevents NaN at the origin when differentiating cos θ / sin θ.
    """
    r     = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
    cos_t = x / r
    sin_t = y / r

    return torch.stack(
        [r, cos_t, sin_t, z, torch.sin(z), torch.sin(2 * z), torch.sin(3 * z)],
        dim=-1,
    )   # [B, P, 7]


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

    def forward_trunk_only(self, latent, x, y, z, a_b, d_b):
        """
        Trunk forward pass using a pre-computed branch latent and raw spatial
        coordinates.

        Normalisation is applied internally:
            x_n = x / a_b,  y_n = y,  z_n = z / d_b
        so the trunk always sees dimensionless inputs in [-1, 1] regardless of
        the disc geometry.  Crucially, the autograd graph still roots at the
        original x, y, z leaves, so calc_laplace(phi, x, y, z) computes ∇²φ
        in physical coordinates without any chain-rule correction needed.

        Args:
            latent : [B, latent_dim]  — may be detached (physics epoch)
                      or live in the graph (data epoch).
            x, y, z: [B, P]           — raw physical coordinates, each a
                      requires_grad=True leaf tensor.
            a_b    : [B]              — semi-axis ratio (normalisation for x)
            d_b    : [B]              — submergence depth  (normalisation for z)

        Returns:
            phi_real, phi_imag : [B, P] each.
        """
        # Normalise: inputs become dimensionless w.r.t. disc geometry.
        # Autograd graph still connects to raw x, y, z leaves.
        x_n = x / a_b.unsqueeze(1)   # [B, P]
        z_n = z / d_b.unsqueeze(1)   # [B, P]

        spatial_features = compute_7_features(x_n, y, z_n)      # [B, P, 7]

        basis = self.trunk(spatial_features)                      # [B, P, latent_dim*2]
        basis_real, basis_imag = torch.chunk(basis, 2, dim=-1)

        phi_real = torch.einsum('bd,bpd->bp', latent, basis_real) + self.bias_real
        phi_imag = torch.einsum('bd,bpd->bp', latent, basis_imag) + self.bias_imag

        return phi_real, phi_imag

    # ------------------------------------------------------------------
    # Convenience: full forward (encode + trunk)
    # ------------------------------------------------------------------

    def forward(self, wave_params, x, y, z):
        a_b = wave_params[:, 0]
        d_b = wave_params[:, 1]
        return self.forward_trunk_only(self.encode(wave_params), x, y, z, a_b, d_b)

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
