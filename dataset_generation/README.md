# PINN Dataset Generation for Submerged Elliptic Disc

This folder contains code to generate a dataset for training a Physics-Informed Neural Network (PINN). The dataset relates physical configurations of an elliptic disc submerged under surface water waves to the resulting hydrodynamic coefficients (Added Mass, Damping) and the potential difference $\phi$ across the disc at a set of radially symmetric points.

## Dataset Format

The generated `dataset.csv` contains the following columns:
- `a_b`: Ratio of the semi-major axis to the semi-minor axis ($a/b$).
- `d_b`: Dimensionless submergence depth of the disc ($d/b$).
- `wave_frequency_K`: Dimensionless wave number $K$.
- `phi_real_0`, `phi_imag_0`, ..., `phi_real_P`, `phi_imag_P`: The real and imaginary components of the potential difference jump across the disc evaluated at $P$ points.
- `Added_Mass`: Non-dimensional Added Mass $\hat{\mathcal{A}}$.
- `Damping_Coefficient`: Non-dimensional Damping Coefficient $\hat{\mathcal{B}}$.

## Determining the $(x, y)$ Coordinates for the $i$-th $\phi$ Value

The $i$-th $\phi$ value (`phi_real_i`, `phi_imag_i`) corresponds to a specific physical point $(x, y)$ on the elliptic disc.

The points are parameterized by $s \in [0, 1]$ and $\alpha \in [0, 2\pi)$. For a given ratio $a/b$, assuming the semi-minor axis $b=1$, the semi-major axis is $a = a/b$.

The Cartesian coordinates of the $i$-th point are given by:
$$x_i = a \cdot s_i \cdot \cos(\alpha_i)$$
$$y_i = b \cdot s_i \cdot \sin(\alpha_i) = s_i \cdot \sin(\alpha_i)$$

### Order of Points

The script `generate_dataset.py` generates the points in a radially symmetric manner:
1. **$i = 0$**: The center of the disc ($s_0 = 0.0, \alpha_0 = 0.0$). Thus $(x_0, y_0) = (0, 0)$.
2. **$i = 1$ to $P$**: The remaining points are organized in $N_s$ concentric "rings", where $s$ takes values from `np.linspace(0, 1, N_s + 1)[1:]`. For each ring, $N_\alpha$ angles are evaluated uniformly from $0$ to $2\pi$ (`np.linspace(0, 2*np.pi, N_alpha, endpoint=False)`).

By default, the script generates $N_s = 3$ rings and $N_\alpha = 8$ angles, resulting in $1 + 3 \times 8 = 25$ total points.

The exact $s_i$ and $\alpha_i$ for index $i$ can be reconstructed as follows:
```python
import numpy as np

def get_point_coordinates(i, a_b):
    if i == 0:
        s, alpha = 0.0, 0.0
    else:
        # Determine the ring index and angle index
        idx = i - 1
        ring_idx = idx // 8
        alpha_idx = idx % 8
        
        # Calculate s and alpha based on default N_s=3, N_alpha=8
        s_vals = np.linspace(0, 1, 4)[1:]
        alpha_vals = np.linspace(0, 2*np.pi, 8, endpoint=False)
        
        s = s_vals[ring_idx]
        alpha = alpha_vals[alpha_idx]
        
    # Calculate Cartesian coordinates (assuming b=1)
    x = a_b * s * np.cos(alpha)
    y = s * np.sin(alpha)
    return x, y
```
