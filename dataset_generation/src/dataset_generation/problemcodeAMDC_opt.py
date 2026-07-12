import numpy as np
import jax
import jax.numpy as jnp
import sympy as sp
import scipy.special as sps
import functools

# Enable 64-bit precision to match the solver's complex128 output
jax.config.update("jax_enable_x64", True)

# ──────────────────────────────────────────────────────────────────────
# JAX Custom JVPs for SciPy Bessel Functions
# ──────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────
# Quadrature and Precomputation (Independent of a, d)
# ──────────────────────────────────────────────────────────────────────
def lgwt(N, a, b):
    x, w = np.polynomial.legendre.leggauss(N)
    return 0.5*(a*(1-x) + b*(1+x)), w*0.5*(b-a)

_QUAD_N = 100
_INT_X, _INT_W = lgwt(_QUAD_N, 0.0, np.pi / 2)
_INT_TX = np.tan(_INT_X)

_DIC_X1, _DIC_W1 = lgwt(_QUAD_N, 0.0, 1.0)
_DIC_X2, _DIC_W2 = lgwt(_QUAD_N, 0.0, 2.0*np.pi)
_DIC_W = np.outer(_DIC_W1, _DIC_W2)
S, A = np.meshgrid(_DIC_X1, _DIC_X2, indexing='xy')
_DIC_X1_FLAT = S.flatten(order='F')
_DIC_X2_FLAT = A.flatten(order='F')

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
    OP_rows = np.zeros((size, _QUAD_N*_QUAD_N), dtype=np.float64)
    
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
        lgkm = sps.gammaln(k+1.5);  lgkm1 = sps.gammaln(k+m+1)
        lf2k1 = sps.gammaln(2*k+2);  lf2km = sps.gammaln(2*k+2*m+2)
        
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
        coeff_tbl[(k,m)] = (jnp.array(Clmk), jnp.array(Elkm), jnp.array(l_arr))
        exp_l_km[(k,m)] = jnp.array(exp_l)
        
    ht_mat_precomp = []
    for k, m in km_pairs:
        Clmk, Elkm, _ = coeff_tbl[(k, m)]
        ht = (
            Clmk[:, None] * alform3_km[(k, m)] * exp_l_km[(k, m)] * exp_m_pos[m][None, :]
            + Elkm[:, None] * alform2_km[(k, m)] * exp_l_km[(k, m)] * exp_m_neg[m][None, :]
        ) / (2.0 * denom_flat[None, :])
        ht_mat_precomp.append(ht)
        
    x_nodes, w_nodes = lgwt(_QUAD_N, -np.pi, np.pi)
    
    return {
        'R_flat': jnp.array(R_flat), 
        'THETA_flat': jnp.array(THETA_flat),
        'WPP_rows': jnp.array(WPP_rows), 
        'OP_rows': jnp.array(OP_rows),
        'ht_mat_precomp': ht_mat_precomp,  # list of (L, size) arrays
        'x_nodes': jnp.array(x_nodes),
        'w_nodes': jnp.array(w_nodes),
        'l_arrs': [coeff_tbl[km][2] for km in km_pairs], # list of (L,) arrays
        'km_pairs': km_pairs
    }

# ──────────────────────────────────────────────────────────────────────
# JAX solver functions
# ──────────────────────────────────────────────────────────────────────
def getgl_jax(l_arr, a, b, x_nodes, w_nodes):
    gval1 = a * b * (a**2 * jnp.cos(x_nodes)**2 + b**2 * jnp.sin(x_nodes)**2)**(-1.5) * w_nodes
    gval2 = jnp.exp(-1j * x_nodes[:, None] * l_arr[None, :])
    return (1/(2*np.pi)) * jnp.sum(gval1[:, None] * gval2, axis=0)

def intgralv_jax(Z_scal, X_arr):
    tx = jnp.array(_INT_TX)
    wq = jnp.array(_INT_W)
    w_eff = wq * (tx * jnp.sin(Z_scal * tx) + jnp.cos(Z_scal * tx))
    X_exp = X_arr[None, :, :]
    tx_exp = tx[:, None, None]
    integrand = w_eff[:, None, None] * k0_jax(tx_exp * X_exp)
    return jnp.sum(integrand, axis=0)

def doubleintC_batch_jax(R_pts, THETA_pts, depth, K, a, b):
    x = a * R_pts[:, None] * jnp.cos(THETA_pts[:, None])
    y = b * R_pts[:, None] * jnp.sin(THETA_pts[:, None])
    gi = a * jnp.array(_DIC_X1_FLAT)[None, :] * jnp.cos(jnp.array(_DIC_X2_FLAT)[None, :])
    eta = b * jnp.array(_DIC_X1_FLAT)[None, :] * jnp.sin(jnp.array(_DIC_X2_FLAT)[None, :])
    
    X_arr = K * jnp.sqrt((x - gi)**2 + (y - eta)**2)
    Z_scal = 2.0 * K * depth
    denom2 = X_arr**2 + Z_scal**2
    denom = jnp.sqrt(denom2)
    
    M = (K**3) * ((2*Z_scal - 1)/denom2**1.5 + 3*Z_scal**2/denom2**2.5 + 1/denom) + (K**3)/denom
    N_val = (K**3) * (2*jnp.pi*1j*(j0_jax(X_arr) + 1j*y0_jax(X_arr))*jnp.exp(-Z_scal) 
                      - (4/jnp.pi)*intgralv_jax(Z_scal, X_arr))
    return M + N_val

def assemble_A_c_jax(a, d_val, K, N, b, pre):
    P4_batch = doubleintC_batch_jax(pre['R_flat'], pre['THETA_flat'], d_val, K, a, b)
    c = (a * b) * (P4_batch @ pre['WPP_rows'].T)
    
    A_cols = []
    for i, km in enumerate(pre['km_pairs']):
        l_arr = pre['l_arrs'][i]
        ht_mat = pre['ht_mat_precomp'][i]
        gl = getgl_jax(l_arr, a, b, pre['x_nodes'], pre['w_nodes'])
        col = ht_mat.T @ gl
        A_cols.append(col)
        
    A = jnp.column_stack(A_cols)
    return A, c

def problemcodeAMDC_jax(a, d_val, K, N, b, pre):
    A, c = assemble_A_c_jax(a, d_val, K, N, b, pre)
    size = (N+1)**2
    f = 4 * jnp.pi * jnp.ones(size, dtype=jnp.complex128)
    X_sol = jax.scipy.linalg.solve(A + c, f)
    
    sum1 = (pre['OP_rows'].T @ X_sol).reshape((_QUAD_N, _QUAD_N))
    w1 = jnp.array(_DIC_W1)
    w2 = jnp.array(_DIC_W2)
    final = -jnp.sum(jnp.outer(w1, w2) * sum1) / jnp.pi
    
    return final, X_sol

# ──────────────────────────────────────────────────────────────────────
# JAX phi evaluator
# ──────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _alform_poly_coeffs(k: int, m: int) -> np.ndarray:
    r = sp.Symbol('r')
    l = m + 2 * k + 1
    dPl = sp.diff(sp.legendre(l, r), r, m)
    poly = sp.Poly(sp.expand(dPl), r)
    return np.array([float(c) for c in poly.all_coeffs()], dtype=np.float64)

def alform_jax(k: int, m: int, s: jnp.ndarray) -> jnp.ndarray:
    coeffs = jnp.array(_alform_poly_coeffs(k, m))
    u = jnp.sqrt(jnp.clip(1.0 - s * s, 0.0, None))
    return ((-1.0) ** m) * (s ** m) * jnp.polyval(coeffs, u)

def phi_series(s: jnp.ndarray, alpha: jnp.ndarray, X_flat: jnp.ndarray, N: int) -> jnp.ndarray:
    result = jnp.zeros((), dtype=jnp.complex128)
    q = 0
    for k in range(N + 1):
        for m in range(N + 1):
            basis = alform_jax(k, m, s) * jnp.cos(jnp.array(m, jnp.float64) * alpha)
            result = result + X_flat[q] * basis
            q += 1
    return result

def _eval_one_config(X_j: jnp.ndarray, sa_j: jnp.ndarray, N: int, a_j, b_j):
    s_j     = sa_j[:, 0]
    alpha_j = sa_j[:, 1]

    phi_v = jax.vmap(lambda s, a: phi_series(s, a, X_j, N))(s_j, alpha_j)

    def phi_r(s, alpha): return jnp.real(phi_series(s, alpha, X_j, N))
    def phi_i(s, alpha): return jnp.imag(phi_series(s, alpha, X_j, N))

    gr = jax.vmap(jax.grad(phi_r, argnums=(0, 1)))(s_j, alpha_j)
    gi = jax.vmap(jax.grad(phi_i, argnums=(0, 1)))(s_j, alpha_j)

    dphi_ds     = gr[0] + 1j * gi[0]
    dphi_dalpha = gr[1] + 1j * gi[1]

    s_safe = jnp.where(s_j < 1e-12, 1e-12, s_j)
    cos_a  = jnp.cos(alpha_j)
    sin_a  = jnp.sin(alpha_j)

    dphi_dx = (cos_a / a_j) * dphi_ds - (sin_a / (a_j * s_safe)) * dphi_dalpha
    dphi_dy = (sin_a / b_j) * dphi_ds + (cos_a / (b_j * s_safe)) * dphi_dalpha

    return phi_v, dphi_dx, dphi_dy


# ──────────────────────────────────────────────────────────────────────
# Full analytical evaluation wrapping (Forward pass + AD Jacobians)
# ──────────────────────────────────────────────────────────────────────
def evaluate_all_base(a, d_val, K, N, b, sa_j, pre):
    """Base pure function for a single (a, d, K) configuration."""
    final, X_sol = problemcodeAMDC_jax(a, d_val, K, N, b, pre)
    
    AM = jnp.real(jnp.pi * final * a)
    DC = jnp.imag(jnp.pi * final * a)
    
    phi_v, dphi_dx, dphi_dy = _eval_one_config(X_sol, sa_j, N, a, b)
    
    # We return a single flat tuple of all outputs
    return AM, DC, phi_v, dphi_dx, dphi_dy

@functools.partial(jax.jit, static_argnums=(3, 4))
def evaluate_all_with_derivatives(a, d_val, K, N, b, sa_j, pre):
    """Evaluates base quantities and their analytical derivatives w.r.t `a` and `d`."""
    # Forward evaluation
    out = evaluate_all_base(a, d_val, K, N, b, sa_j, pre)
    
    jac_all = jax.jacfwd(evaluate_all_base, argnums=(0, 1))(a, d_val, K, N, b, sa_j, pre)
    
    jac_a = tuple(jac_out[0] for jac_out in jac_all)
    jac_d = tuple(jac_out[1] for jac_out in jac_all)
    
    return out, jac_a, jac_d


def generate_batch_data_jax(a_vals, d_vals, K_vals, N, b, s_pts, alpha_pts):
    """
    Called by generate_dataset.py to run the full grid using the JAX solver.
    """
    pre = _precompute_A_data_jax(N)
    sa_j = jnp.stack([jnp.array(s_pts), jnp.array(alpha_pts)], axis=-1)
    
    # To process the grid efficiently, we can use jax.lax.map or vmap.
    # We will build a flat array of inputs and vmap over it.
    grid_a, grid_d, grid_K = jnp.meshgrid(
        jnp.array(a_vals), 
        jnp.array(d_vals), 
        jnp.array(K_vals), 
        indexing='ij'
    )
    
    flat_a = grid_a.ravel()
    flat_d = grid_d.ravel()
    flat_K = grid_K.ravel()
    # We use jax.lax.map to iterate over the grid inside XLA without 
    # triggering vmap_method restrictions on the scipy callbacks.
    @jax.jit
    def map_all(indices):
        def map_body(idx):
            return evaluate_all_with_derivatives(
                flat_a[idx], flat_d[idx], flat_K[idx], N, b, sa_j, pre
            )
        return jax.lax.map(map_body, indices)
        
    print(f"JIT compiling and evaluating {len(flat_a)} grid points analytically...")
    indices = jnp.arange(len(flat_a))
    out_all, jac_a_all, jac_d_all = map_all(indices)
    
    # out_all is a tuple of (AM, DC, phi_v, dphi_dx, dphi_dy) 
    # each with shape (n_grid, ...)
    shape = (len(a_vals), len(d_vals), len(K_vals))
    
    def reshape_tuple(tup):
        return tuple(x.reshape(shape + x.shape[1:]) for x in tup)
    
    return reshape_tuple(out_all), reshape_tuple(jac_a_all), reshape_tuple(jac_d_all)

