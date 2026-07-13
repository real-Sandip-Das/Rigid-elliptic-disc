import numpy as np
import jax
import jax.numpy as jnp
import sympy as sp
import scipy.special as sps
import functools
from concurrent.futures import ThreadPoolExecutor
import os

# Enable 64-bit precision to match the solver's complex128 output
jax.config.update("jax_enable_x64", True)

@jax.custom_jvp
def j0_jax(x):
    return jax.pure_callback(lambda v: sps.j0(v).astype(v.dtype), jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method="broadcast_all")

@jax.custom_jvp
def j1_jax(x):
    return jax.pure_callback(lambda v: sps.j1(v).astype(v.dtype), jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method="broadcast_all")

@j0_jax.defjvp
def j0_jvp(primals, tangents):
    x, = primals; x_dot, = tangents
    return j0_jax(x), -j1_jax(x) * x_dot

@j1_jax.defjvp
def j1_jvp(primals, tangents):
    x, = primals; x_dot, = tangents
    return j1_jax(x), jnp.zeros_like(x)

@jax.custom_jvp
def y0_jax(x):
    return jax.pure_callback(lambda v: sps.y0(v).astype(v.dtype), jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method="broadcast_all")

@jax.custom_jvp
def y1_jax(x):
    return jax.pure_callback(lambda v: sps.y1(v).astype(v.dtype), jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method="broadcast_all")

@y0_jax.defjvp
def y0_jvp(primals, tangents):
    x, = primals; x_dot, = tangents
    return y0_jax(x), -y1_jax(x) * x_dot

@y1_jax.defjvp
def y1_jvp(primals, tangents):
    x, = primals; x_dot, = tangents
    return y1_jax(x), jnp.zeros_like(x)

@jax.custom_jvp
def k0_jax(x):
    return jax.pure_callback(lambda v: sps.k0(v).astype(v.dtype), jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method="broadcast_all")

@jax.custom_jvp
def k1_jax(x):
    return jax.pure_callback(lambda v: sps.k1(v).astype(v.dtype), jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method="broadcast_all")

@k0_jax.defjvp
def k0_jvp(primals, tangents):
    x, = primals; x_dot, = tangents
    return k0_jax(x), -k1_jax(x) * x_dot

@k1_jax.defjvp
def k1_jvp(primals, tangents):
    x, = primals; x_dot, = tangents
    return k1_jax(x), jnp.zeros_like(x)

def lgwt(N, a, b):
    x, w = np.polynomial.legendre.leggauss(N)
    return 0.5*(a*(1-x) + b*(1+x)), w*0.5*(b-a)

_QUAD_N = 100

_INT_X, _INT_W = lgwt(_QUAD_N, 0.0, np.pi / 2)
_INT_TX = np.tan(_INT_X)   # shape (100,)

_DIC_X1, _DIC_W1 = lgwt(_QUAD_N, 0.0, 1.0)
_DIC_X2, _DIC_W2 = lgwt(_QUAD_N, 0.0, 2.0*np.pi)
_DIC_W = np.outer(_DIC_W1, _DIC_W2)  # (100,100)
_S_grid, _A_grid = np.meshgrid(_DIC_X1, _DIC_X2, indexing='xy')
_DIC_X1_FLAT = _S_grid.flatten(order='F')  # (10000,) — s integration nodes
_DIC_X2_FLAT = _A_grid.flatten(order='F')  # (10000,) — alpha integration nodes

_INT_TX_JAX = jnp.array(_INT_TX)
_INT_W_JAX  = jnp.array(_INT_W)
_DIC_X1_FLAT_JAX = jnp.array(_DIC_X1_FLAT)
_DIC_X2_FLAT_JAX = jnp.array(_DIC_X2_FLAT)
_DIC_W_JAX  = jnp.array(_DIC_W)
_DIC_W1_JAX = jnp.array(_DIC_W1)
_DIC_W2_JAX = jnp.array(_DIC_W2)

@functools.lru_cache(maxsize=None)
def _get_alform_func(k, m):
    r  = sp.Symbol('r')
    Pl = sp.legendre(m + 2*k + 1, r)
    P1 = ((-1)**m) * (1-r**2)**(sp.Rational(m,2)) * sp.diff(Pl, r, m)
    return sp.lambdify(r, P1.subs(r, sp.sqrt(1-r**2)), modules='numpy')

@functools.lru_cache(maxsize=None)
def _get_alform2_func(k, m, l_val):
    r  = sp.Symbol('r')
    l1 = int(m + 2*k + 1);  m1 = int(-m + l_val)
    c  = ((-1.0)**m1) / ((2**l1) * sp.factorial(l1))
    P1 = c * (1-r**2)**(sp.Rational(m1,2)) * sp.diff((r**2-1)**l1, r, l1+m1)
    return sp.lambdify(r, P1.subs(r, sp.sqrt(1-r**2)), modules='numpy')

@functools.lru_cache(maxsize=None)
def _get_alform3_func(k, m, l_val):
    r  = sp.Symbol('r')
    l1 = int(m + 2*k + 1);  m1 = int(m + l_val)
    c  = ((-1.0)**m1) / ((2**l1) * sp.factorial(l1))
    P1 = c * (1-r**2)**(sp.Rational(m1,2)) * sp.diff((r**2-1)**l1, r, l1+m1)
    return sp.lambdify(r, P1.subs(r, sp.sqrt(1-r**2)), modules='numpy')

def _eval_lambdify(func, s):
    res = func(s)
    if np.isscalar(res) and np.ndim(s) > 0:
        return np.full(np.shape(s), float(res), dtype=np.float64)
    return np.asarray(res, dtype=np.float64)

@functools.lru_cache(maxsize=None)
def _precompute_A_data_jax(N):
    km_pairs = [(k, m) for k in range(N+1) for m in range(N+1)]

    n_idx = np.arange(N+1)
    theta_c = (2*n_idx + 1) * np.pi / (2*N + 2)
    r_c = np.cos(theta_c / 2)
    R_grid, T_grid = np.meshgrid(r_c, theta_c, indexing='xy')
    R_flat = R_grid.flatten(order='F')
    THETA_flat = T_grid.flatten(order='F')
    denom_flat = np.sqrt(1.0 - R_flat**2)

    size = (N+1)**2

    WPP_rows = np.zeros((size, _QUAD_N*_QUAD_N), dtype=np.float64)
    OP_rows  = np.zeros((size, _QUAD_N*_QUAD_N), dtype=np.float64)

    cosm_cache = {m: np.cos(m*_DIC_X2) for m in range(N+1)}
    for idx, (k,m) in enumerate(km_pairs):
        p1 = _eval_lambdify(_get_alform_func(k,m), _DIC_X1)*_DIC_X1
        op = np.outer(p1, cosm_cache[m])
        WPP_rows[idx] = (_DIC_W*op).ravel()
        OP_rows[idx]  = op.ravel()

    l_unique = np.arange(-2*N, 2*N+1, 2, dtype=int)
    l_to_row = {int(l): i for i, l in enumerate(l_unique)}
    exp_l_table = np.exp(1j * l_unique[:,None] * THETA_flat[None,:])

    exp_m_pos = np.exp( 1j * np.arange(N+1)[:,None] * THETA_flat[None,:])
    exp_m_neg = np.exp(-1j * np.arange(N+1)[:,None] * THETA_flat[None,:])

    alform2_km = {}; alform3_km = {}; coeff_tbl = {}; exp_l_km = {}
    for k, m in km_pairs:
        l1_arr = np.arange(-k, k+1, dtype=np.float64)
        l_arr  = (2*l1_arr).astype(int)

        a2 = np.stack([_eval_lambdify(_get_alform2_func(k,m,int(lv)), R_flat) for lv in l_arr])
        a3 = np.stack([_eval_lambdify(_get_alform3_func(k,m,int(lv)), R_flat) for lv in l_arr])

        ds = np.sign(l_arr**2 - 1)
        lgkm  = sps.gammaln(k+1.5);   lgkm1 = sps.gammaln(k+m+1)
        lf2k1 = sps.gammaln(2*k+2);   lf2km = sps.gammaln(2*k+2*m+2)

        log_C = (np.log(np.pi) + (l_arr+2)*np.log(2.) - np.log(np.abs(l_arr**2-1))
                 + sps.gammaln(2*k-l_arr+2) - sps.gammaln(2*k+2*m+l_arr+2)
                 + lf2km - lf2k1 + lgkm - lgkm1
                 + sps.gammaln(k+l_arr/2+m+1.5) - sps.gammaln(k-l_arr/2+1))
        Clmk = ds * (-1.0)**(l_arr+2*m) * np.exp(log_C)

        log_E = (np.log(np.pi) + (l_arr-2*m+2)*np.log(2.) - np.log(np.abs(l_arr**2-1))
                 + lgkm - lgkm1 + lf2km - lf2k1
                 + sps.gammaln(k+l_arr/2+1.5) - sps.gammaln(m+k-l_arr/2+1)
                 + sps.gammaln(2*m+2*k-l_arr+2) - sps.gammaln(l_arr+2*k+2))
        Elkm = ds * (-1.0)**(l_arr+m) * np.exp(log_E)

        rows = [l_to_row[int(lv)] for lv in l_arr]
        exp_l = exp_l_table[rows]

        alform2_km[(k,m)] = jnp.array(a2)
        alform3_km[(k,m)] = jnp.array(a3)
        coeff_tbl[(k,m)]  = (jnp.array(Clmk), jnp.array(Elkm), jnp.array(l_arr))
        exp_l_km[(k,m)]   = jnp.array(exp_l)

    ht_mats  = []
    l_arrs   = []
    for k, m in km_pairs:
        Clmk, Elkm, _ = coeff_tbl[(k, m)]
        ht = (
            Clmk[:, None] * alform3_km[(k, m)] * exp_l_km[(k, m)] * exp_m_pos[m][None, :]
            + Elkm[:, None] * alform2_km[(k, m)] * exp_l_km[(k, m)] * exp_m_neg[m][None, :]
        ) / (2.0 * denom_flat[None, :])
        ht_mats.append(ht)
        l_arrs.append(coeff_tbl[(k,m)][2])

    x_nodes, w_nodes = lgwt(_QUAD_N, -np.pi, np.pi)

    return {
        'R_flat':        jnp.array(R_flat),
        'THETA_flat':    jnp.array(THETA_flat),
        'WPP_rows':      jnp.array(WPP_rows),
        'OP_rows':       jnp.array(OP_rows),
        'ht_mats':       ht_mats,       # list[(L,size) complex128] — a-independent!
        'x_nodes':       jnp.array(x_nodes),
        'w_nodes':       jnp.array(w_nodes),
        'l_arrs':        l_arrs,
        'km_pairs':      km_pairs,
    }

def getgl_jax(l_arr, a, b, x_nodes, w_nodes):
    gval1 = a * b * (a**2 * jnp.cos(x_nodes)**2 + b**2 * jnp.sin(x_nodes)**2)**(-1.5) * w_nodes
    gval2 = jnp.exp(-1j * x_nodes[:, None] * l_arr[None, :])
    return (1/(2*np.pi)) * jnp.sum(gval1[:, None] * gval2, axis=0)

def intgralv_jax(Z_scal, X_flat):
    """
    X_flat : (M,) — K*R values for all boundary/integration points.
    Returns : (M,) integral values.
    Avoids the old (100,size,size) 3-D allocation.
    """
    tx  = _INT_TX_JAX          # (100,)
    wq  = _INT_W_JAX           # (100,)
    w_eff = wq * (tx * jnp.sin(Z_scal * tx) + jnp.cos(Z_scal * tx))
    K0 = k0_jax(tx[:, None] * X_flat[None, :])  # (100, M)
    return jnp.sum(w_eff[:, None] * K0, axis=0)

def doubleintC_batch_jax(R_pts, THETA_pts, depth, K, a, b):
    """Compute the BIE kernel integral for all collocation points at once."""
    x  = a * R_pts[:, None]   * jnp.cos(THETA_pts[:, None])
    y  = b * R_pts[:, None]   * jnp.sin(THETA_pts[:, None])
    gi  = a * _DIC_X1_FLAT_JAX[None, :] * jnp.cos(_DIC_X2_FLAT_JAX[None, :])
    eta = b * _DIC_X1_FLAT_JAX[None, :] * jnp.sin(_DIC_X2_FLAT_JAX[None, :])

    R_dist = jnp.sqrt((x - gi)**2 + (y - eta)**2)   # (size, M)
    X_arr  = K * R_dist
    Z_scal = 2.0 * K * depth   # scalar

    denom2 = X_arr**2 + Z_scal**2
    denom  = jnp.sqrt(denom2)

    M_term = (K**3) * ((2*Z_scal - 1)/denom2**1.5 + 3*Z_scal**2/denom2**2.5 + 1/denom) + (K**3)/denom

    H0 = j0_jax(X_arr) + 1j * y0_jax(X_arr)
    N_term = (K**3) * (2*jnp.pi*1j*H0*jnp.exp(-Z_scal)
                        - (4/jnp.pi) * intgralv_jax(Z_scal, X_arr.ravel()).reshape(X_arr.shape))
    return M_term + N_term

def assemble_A_c_jax(a, d_val, K, N, b, pre):
    """Assemble BIE system matrix. a and d_val are differentiable."""
    P4_batch = doubleintC_batch_jax(pre['R_flat'], pre['THETA_flat'], d_val, K, a, b)
    c = (a * b) * (P4_batch @ pre['WPP_rows'].T)   # (size, size)

    A_cols = []
    for i, km in enumerate(pre['km_pairs']):
        l_arr   = pre['l_arrs'][i]
        ht_mat  = pre['ht_mats'][i]           # (L, size) — precomputed, a-independent
        gl      = getgl_jax(l_arr, a, b, pre['x_nodes'], pre['w_nodes'])  # (L,)
        A_cols.append(ht_mat.T @ gl)          # (size,)

    A = jnp.stack(A_cols, axis=1)             # (size, size)
    return A, c

def problemcodeAMDC_jax(a, d_val, K, N, b, pre):
    A, c = assemble_A_c_jax(a, d_val, K, N, b, pre)
    size  = (N+1)**2
    f     = 4 * jnp.pi * jnp.ones(size, dtype=jnp.complex128)
    X_sol = jax.scipy.linalg.solve(A + c, f)

    sum1  = (pre['OP_rows'].T @ X_sol).reshape((_QUAD_N, _QUAD_N))
    final = -jnp.sum(jnp.outer(_DIC_W1_JAX, _DIC_W2_JAX) * sum1) / jnp.pi

    return final, X_sol

@functools.lru_cache(maxsize=None)
def _alform_poly_coeffs(k: int, m: int) -> np.ndarray:
    r = sp.Symbol('r')
    l = m + 2 * k + 1
    dPl  = sp.diff(sp.legendre(l, r), r, m)
    poly = sp.Poly(sp.expand(dPl), r)
    return np.array([float(c) for c in poly.all_coeffs()], dtype=np.float64)

def alform_jax(k: int, m: int, s: jnp.ndarray) -> jnp.ndarray:
    coeffs = jnp.array(_alform_poly_coeffs(k, m))
    u = jnp.sqrt(jnp.clip(1.0 - s * s, 0.0, None))
    return ((-1.0) ** m) * (s ** m) * jnp.polyval(coeffs, u)

def phi_series(s, alpha, X_flat, N):
    result = jnp.zeros((), dtype=jnp.complex128)
    q = 0
    for k in range(N + 1):
        for m in range(N + 1):
            basis = alform_jax(k, m, s) * jnp.cos(jnp.array(m, jnp.float64) * alpha)
            result = result + X_flat[q] * basis
            q += 1
    return result

def _eval_phi_and_derivs(X_sol, dX_da, dX_dd, sa_j, N, a, b):
    """
    Evaluate phi and its parametric derivatives at the sample points.
    phi is linear in X, so d(phi)/da = phi_series(..., dX_da) etc.
    """
    s_j     = sa_j[:, 0]
    alpha_j = sa_j[:, 1]

    phi_v   = jax.vmap(lambda s, al: phi_series(s, al, X_sol, N))(s_j, alpha_j)
    dphi_da = jax.vmap(lambda s, al: phi_series(s, al, dX_da,  N))(s_j, alpha_j)
    dphi_dd = jax.vmap(lambda s, al: phi_series(s, al, dX_dd,  N))(s_j, alpha_j)

    def phi_r(s, al): return jnp.real(phi_series(s, al, X_sol, N))
    def phi_i(s, al): return jnp.imag(phi_series(s, al, X_sol, N))

    gr = jax.vmap(jax.grad(phi_r, argnums=(0, 1)))(s_j, alpha_j)
    gi = jax.vmap(jax.grad(phi_i, argnums=(0, 1)))(s_j, alpha_j)

    dphi_ds     = gr[0] + 1j * gi[0]
    dphi_dalpha = gr[1] + 1j * gi[1]

    s_safe = jnp.where(s_j < 1e-12, 1e-12, s_j)
    cos_a  = jnp.cos(alpha_j)
    sin_a  = jnp.sin(alpha_j)

    dphi_dx = (cos_a / a) * dphi_ds - (sin_a / (a * s_safe)) * dphi_dalpha
    dphi_dy = (sin_a / b) * dphi_ds + (cos_a / (b * s_safe)) * dphi_dalpha

    return phi_v, dphi_da, dphi_dd, dphi_dx, dphi_dy


# ──────────────────────────────────────────────────────────────────────
# IFT: solve once, then get dX/da and dX/dd via cheap triangular solves
# ──────────────────────────────────────────────────────────────────────

_FD_H = 1e-5   # relative step for central-difference on the matrix

def _compute_X_and_derivs(a, d_val, K, N, b, pre):
    """
    Solve the BIE system and compute dX/da, dX/dd via the Implicit Function Theorem.

    C(a,d) X = f  =>  dX/dp = -C^{-1} (dC/dp X)   for p in {a, d}

    dC/dp is estimated by central-difference on just the matrix assembly —
    two cheap matrix builds per parameter, then two triangular solves
    reusing the LU already computed for the forward pass.
    """
    h_a = jnp.maximum(jnp.abs(a),     1.0) * _FD_H
    h_d = jnp.maximum(jnp.abs(d_val), 1.0) * _FD_H

    # ── Forward solve ────────────────────────────────────────────────
    A0, c0 = assemble_A_c_jax(a, d_val, K, N, b, pre)
    C0   = A0 + c0
    size = (N + 1) ** 2
    f    = 4 * jnp.pi * jnp.ones(size, dtype=jnp.complex128)
    lu, piv = jax.scipy.linalg.lu_factor(C0)        # factorise once
    X_sol   = jax.scipy.linalg.lu_solve((lu, piv), f)

    # ── dC/da by central difference ──────────────────────────────────
    Ap, cp = assemble_A_c_jax(a + h_a, d_val, K, N, b, pre)
    Am, cm = assemble_A_c_jax(a - h_a, d_val, K, N, b, pre)
    dC_da  = ((Ap + cp) - (Am + cm)) / (2.0 * h_a)

    # ── dC/dd — only c depends on d ──────────────────────────────────
    _, cp_d = assemble_A_c_jax(a, d_val + h_d, K, N, b, pre)
    _, cm_d = assemble_A_c_jax(a, d_val - h_d, K, N, b, pre)
    dC_dd   = (cp_d - cm_d) / (2.0 * h_d)

    # ── Derivative solves (triangular — very cheap) ──────────────────
    dX_da = jax.scipy.linalg.lu_solve((lu, piv), -(dC_da @ X_sol))
    dX_dd = jax.scipy.linalg.lu_solve((lu, piv), -(dC_dd @ X_sol))

    # ── Scalar outputs ───────────────────────────────────────────────
    W = jnp.outer(_DIC_W1_JAX, _DIC_W2_JAX)

    def _final_from_X(X):
        s = (pre['OP_rows'].T @ X).reshape((_QUAD_N, _QUAD_N))
        return -jnp.sum(W * s) / jnp.pi

    final      = _final_from_X(X_sol)
    dfinal_da  = _final_from_X(dX_da)
    dfinal_dd  = _final_from_X(dX_dd)

    AM     = jnp.real(jnp.pi * final * a)
    DC     = jnp.imag(jnp.pi * final * a)
    dAM_da = jnp.real(jnp.pi * (dfinal_da * a + final))   # chain rule on *a
    dDC_da = jnp.imag(jnp.pi * (dfinal_da * a + final))
    dAM_dd = jnp.real(jnp.pi * dfinal_dd * a)
    dDC_dd = jnp.imag(jnp.pi * dfinal_dd * a)

    return X_sol, dX_da, dX_dd, AM, DC, dAM_da, dDC_da, dAM_dd, dDC_dd



# ──────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _get_jitted_evaluator(N, b, n_pts):
    def _single_eval(a, d_val, K, sa_j, pre):
        (X_sol, dX_da, dX_dd,
         AM, DC, dAM_da, dDC_da, dAM_dd, dDC_dd) = _compute_X_and_derivs(a, d_val, K, N, b, pre)

        phi_v, dphi_da, dphi_dd, dphi_dx, dphi_dy = _eval_phi_and_derivs(
            X_sol, dX_da, dX_dd, sa_j, N, a, b)

        out   = (AM, DC, phi_v, dphi_dx, dphi_dy)
        jac_a = (dAM_da, dDC_da, dphi_da, jnp.zeros_like(dphi_dx), jnp.zeros_like(dphi_dy))
        jac_d = (dAM_dd, dDC_dd, dphi_dd, jnp.zeros_like(dphi_dx), jnp.zeros_like(dphi_dy))
        return out, jac_a, jac_d

    return jax.jit(_single_eval)


# ──────────────────────────────────────────────────────────────────────
# Batch generation — parallel over CPU cores
# ──────────────────────────────────────────────────────────────────────
def generate_batch_data_jax(a_vals, d_vals, K_vals, N, b, s_pts, alpha_pts):
    """
    Evaluates AM, DC, phi + all analytical a/d derivatives over the full grid.

    Strategy: compile one JIT kernel, then dispatch entries concurrently in
    fixed-size chunks so that:
      - Thread count is always bounded (n_workers <= cpu_count * 2).
      - Memory at any moment is bounded to chunk_size results, not the full grid.
      - Progress is reported after each chunk.
    """
    pre   = _precompute_A_data_jax(N)
    sa_j  = jnp.stack([jnp.array(s_pts), jnp.array(alpha_pts)], axis=-1)
    n_pts = len(s_pts)

    jit_fn = _get_jitted_evaluator(N, b, n_pts)

    # Build flat list of all (a, d, K) configs
    grid_a, grid_d, grid_K = np.meshgrid(a_vals, d_vals, K_vals, indexing='ij')
    flat_a = grid_a.ravel()
    flat_d = grid_d.ravel()
    flat_K = grid_K.ravel()
    n_total = len(flat_a)

    n_workers  = min(n_total, (os.cpu_count() or 4) * 2)
    chunk_size = n_workers * 4

    print(f"JIT compiling on first entry...")
    first = jit_fn(float(flat_a[0]), float(flat_d[0]), float(flat_K[0]), sa_j, pre)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), first)
    print(f"Compilation done. Evaluating {n_total} entries "
          f"({n_workers} workers, chunk={chunk_size})...")

    def _eval_one(i):
        return jit_fn(float(flat_a[i]), float(flat_d[i]), float(flat_K[i]), sa_j, pre)

    out_lists   = [[] for _ in range(5)]
    jac_a_lists = [[] for _ in range(5)]
    jac_d_lists = [[] for _ in range(5)]

    def _drain(result):
        out, jac_a, jac_d = result
        for j, v in enumerate(out):       out_lists[j].append(np.asarray(v))
        for j, v in enumerate(jac_a):  jac_a_lists[j].append(np.asarray(v))
        for j, v in enumerate(jac_d):  jac_d_lists[j].append(np.asarray(v))

    _drain(first)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        remaining = list(range(1, n_total))
        done_count = 1

        for chunk_start in range(0, len(remaining), chunk_size):
            chunk = remaining[chunk_start : chunk_start + chunk_size]

            chunk_futures = [pool.submit(_eval_one, i) for i in chunk]

            for fut in chunk_futures:
                result = fut.result()
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
                _drain(result)

            done_count += len(chunk)
            print(f"  {done_count}/{n_total} entries done...")

    print("All evaluations complete. Assembling output arrays...")

    shape = (len(a_vals), len(d_vals), len(K_vals))

    def _stack_and_reshape(lst):
        arr = np.stack(lst, axis=0)
        return arr.reshape(shape + arr.shape[1:])

    out_all   = tuple(_stack_and_reshape(lst) for lst in out_lists)
    jac_a_all = tuple(_stack_and_reshape(lst) for lst in jac_a_lists)
    jac_d_all = tuple(_stack_and_reshape(lst) for lst in jac_d_lists)

    return out_all, jac_a_all, jac_d_all
