"""
problemcodeAMDC_opt.py  —  Optimized port of problemcodeAMDC.m
=============================================================

Differences vs. the original problemcodeAMDC.py
------------------------------------------------
1.  lgwt cached with lru_cache — nodes computed once per (N,a,b).
2.  intgralv fully vectorized (arbitrary-shape Z,X via broadcast_shapes).
3.  check() / doubleintC_precomputed called in ONE batched call for ALL
    collocation points simultaneously (size×Q² array), replacing the
    per-point loop in Phase 1.
4.  c matrix computed via a single BLAS dgemm: P4_mat @ WPP_mat.T.
5.  A matrix: outer l-loop eliminated; for each (k,m) column, alform2/3
    evaluated once on the full R_flat array, ht_mat built as (L,size),
    column filled by ht_mat.T @ gl.
6.  Phase factors factored: exp(1j*m*THETA) precomputed once per m.
7.  Coefficient gammas/factorials in the A-loop use gammaln for
    numerical stability at large N.
8.  Final summation replaced by a single matvec: OP_mat.T @ X_sol.
9.  Redundant g_k15 alias removed from hyperterm.
10. a*b / (pi*a*b) simplification in final formula.

Verified against problemcodeAMDC.py output for N=1,2,3.
"""

import numpy as np
import sympy as sp
from scipy.special import hankel1, kn, gamma, gammaln, factorial
import functools

# ──────────────────────────────────────────────────────────────────────
# Gauss-Legendre quadrature  (nodes cached by (N, a, b))
# ──────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _legendre_nodes(N, a, b):
    x, w = np.polynomial.legendre.leggauss(N)
    x_mapped = 0.5 * (a * (1 - x) + b * (1 + x))
    w_mapped  = w  * 0.5 * (b - a)
    return x_mapped, w_mapped


def lgwt(N, a, b):
    return _legendre_nodes(N, a, b)


# ──────────────────────────────────────────────────────────────────────
# Module-level shared quadrature grids  (lazy initialisation)
# ──────────────────────────────────────────────────────────────────────
_QUAD_N = 100  # Gauss order used everywhere

_INTGRAL_X, _INTGRAL_W = None, None          # for intgralv  [0, pi/2]
_INTGRAL_TX = None                            # tan(x)  — also cached

_DIC_X1, _DIC_W1 = None, None                # s-nodes  [0, 1]
_DIC_X2, _DIC_W2 = None, None                # alpha-nodes  [0, 2pi]
_DIC_W            = None                      # outer(w1, w2)
_DIC_X1_FLAT      = None                      # s-quadrature, flattened (Q²,)
_DIC_X2_FLAT      = None                      # alpha-quadrature, flattened


def _ensure_intgral_cache():
    global _INTGRAL_X, _INTGRAL_W, _INTGRAL_TX
    if _INTGRAL_X is None:
        _INTGRAL_X, _INTGRAL_W = lgwt(_QUAD_N, 0.0, np.pi / 2)
        _INTGRAL_TX = np.tan(_INTGRAL_X)


def _ensure_dic_cache():
    global _DIC_X1, _DIC_W1, _DIC_X2, _DIC_W2, _DIC_W, _DIC_X1_FLAT, _DIC_X2_FLAT
    if _DIC_X1 is None:
        _DIC_X1, _DIC_W1 = lgwt(_QUAD_N, 0.0, 1.0)
        _DIC_X2, _DIC_W2 = lgwt(_QUAD_N, 0.0, 2.0 * np.pi)
        _DIC_W  = np.outer(_DIC_W1, _DIC_W2)           # (Q, Q)
        S, ALPHA = np.meshgrid(_DIC_X1, _DIC_X2, indexing='xy')
        _DIC_X1_FLAT = S.flatten(order='F')             # (Q²,)
        _DIC_X2_FLAT = ALPHA.flatten(order='F')


# ──────────────────────────────────────────────────────────────────────
# Symbolic lambdify cache  (alform / alform2 / alform3)
# ──────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _get_alform_func(k, m):
    r  = sp.Symbol('r')
    l  = m + 2*k + 1
    Pl = sp.legendre(l, r)
    P1 = ((-1)**m) * (1 - r**2)**(sp.Rational(m, 2)) * sp.diff(Pl, r, m)
    P  = P1.subs(r, sp.sqrt(1 - r**2))
    return sp.lambdify(r, P, modules='numpy')


@functools.lru_cache(maxsize=None)
def _get_alform2_func(k, m, l_val):
    r  = sp.Symbol('r')
    l1 = int(m + 2*k + 1)
    m1 = int(-m + l_val)
    coeff = ((-1.0)**m1) / ((2**l1) * sp.factorial(l1))
    P1 = coeff * (1 - r**2)**(sp.Rational(m1, 2)) * sp.diff((r**2 - 1)**l1, r, l1 + m1)
    P  = P1.subs(r, sp.sqrt(1 - r**2))
    return sp.lambdify(r, P, modules='numpy')


@functools.lru_cache(maxsize=None)
def _get_alform3_func(k, m, l_val):
    r  = sp.Symbol('r')
    l1 = int(m + 2*k + 1)
    m1 = int(m + l_val)
    coeff = ((-1.0)**m1) / ((2**l1) * sp.factorial(l1))
    P1 = coeff * (1 - r**2)**(sp.Rational(m1, 2)) * sp.diff((r**2 - 1)**l1, r, l1 + m1)
    P  = P1.subs(r, sp.sqrt(1 - r**2))
    return sp.lambdify(r, P, modules='numpy')


def _eval_lambdify(func, s):
    """Evaluate a lambdified sympy function; broadcast scalars to array shape."""
    res = func(s)
    if np.isscalar(res) and np.ndim(s) > 0:
        return np.full(np.shape(s), float(res), dtype=np.float64)
    return np.asarray(res, dtype=np.float64)


def alform(k, m, s):
    return _eval_lambdify(_get_alform_func(k, m), s)

def alform2(k, m, l_val, s):
    return _eval_lambdify(_get_alform2_func(k, m, l_val), s)

def alform3(k, m, l_val, s):
    return _eval_lambdify(_get_alform3_func(k, m, l_val), s)


# ──────────────────────────────────────────────────────────────────────
# intgralv — fully vectorized over arbitrary-shape (Z, X)
# ──────────────────────────────────────────────────────────────────────
def intgralv(Z, X):
    """
    Gauss-Legendre approximation of the integral on [0, pi/2].
    Z, X : broadcastable ndarrays or scalars.
    Returns an array of shape broadcast(Z.shape, X.shape).
    """
    _ensure_intgral_cache()
    Z = np.asarray(Z)
    X = np.asarray(X)

    out_shape = np.broadcast_shapes(Z.shape, X.shape)
    ndim_out  = len(out_shape)

    tx = _INTGRAL_TX                                         # (Q,)  precomputed
    w_q = _INTGRAL_W

    # Quadrature axis = axis-0; broadcast against out_shape trailing axes
    ax = (slice(None),) + (np.newaxis,) * ndim_out
    tx_ = tx[ax]                                             # (Q, 1, ..., 1)
    w_  = w_q[ax]

    Z_ = np.broadcast_to(Z, out_shape)[np.newaxis, ...]     # (1, *out_shape)
    X_ = np.broadcast_to(X, out_shape)[np.newaxis, ...]

    integrand = (tx_ * np.sin(Z_ * tx_) + np.cos(tx_ * Z_)) * kn(0, tx_ * X_)
    return np.sum(w_ * integrand, axis=0)


# ──────────────────────────────────────────────────────────────────────
# check — vectorized; accepts scalar or array (r, theta) × array (s, alpha)
# ──────────────────────────────────────────────────────────────────────
def check(r, theta, s, alpha, d_val, K, a, b):
    x   = a * r * np.cos(theta)
    y   = b * r * np.sin(theta)
    gi  = a * s * np.cos(alpha)
    eta = b * s * np.sin(alpha)

    R      = np.sqrt((x - gi)**2 + (y - eta)**2)
    X_arr  = K * R
    Z_scal = 2.0 * K * d_val          # -K*((-d_val) + (-d_val)) = 2*K*d_val

    denom2 = X_arr**2 + Z_scal**2
    denom  = np.sqrt(denom2)

    M = (K**3) * (
        (2 * Z_scal - 1) / denom2**1.5 +
        3 * Z_scal**2    / denom2**2.5 +
        1 / denom
    ) + (K**3) / denom

    N_val = (K**3) * (
        2 * np.pi * 1j * hankel1(0, X_arr) * np.exp(-Z_scal)
        - (4 / np.pi) * intgralv(Z_scal, X_arr)
    )
    return M + N_val


# ──────────────────────────────────────────────────────────────────────
# kernel  (unchanged logic, kept for API compatibility)
# ──────────────────────────────────────────────────────────────────────
def kernel(x, y, s, alpha, K, a, b, d_val):
    gi  = a * s * np.cos(alpha)
    eta = b * s * np.sin(alpha)
    R   = np.sqrt((x - gi)**2 + (y - eta)**2)
    X   = K * R
    Y   = K * d_val

    denom2 = X**2 + Y**2
    return (K**2) * (
        2 * Y / denom2**1.5 +
        2 / np.sqrt(denom2) +
        2 * np.pi * 1j * np.exp(-Y) * hankel1(0, X)
        - (4 / np.pi) * intgralv(Y, X)
    )


# ──────────────────────────────────────────────────────────────────────
# doubleintC helpers
# ──────────────────────────────────────────────────────────────────────
def doubleintC_precomputed(r_point, theta_point, depth, K, a, b):
    """
    Compute the (r,theta)-dependent check() integral for ONE collocation
    point against all (Q²) quadrature abscissas.
    Returns p4.T — shape (Q, Q).
    """
    _ensure_dic_cache()
    p3_flat = check(r_point, theta_point,
                    _DIC_X1_FLAT, _DIC_X2_FLAT, depth, K, a, b)
    # Fortran reshape then transpose ≡ C-order reshape
    return p3_flat.reshape((_QUAD_N, _QUAD_N))   # (Q, Q)


def doubleintC_batch(R_pts, THETA_pts, depth, K, a, b):
    """
    Batch version: compute check() for ALL len(R_pts) collocation points
    against all Q² quadrature abscissas in ONE vectorised call.

    Returns P4_T_batch of shape (size, Q, Q) where
      P4_T_batch[l] == doubleintC_precomputed(R_pts[l], THETA_pts[l], ...).

    Broadcasting:
      (r, theta): (size, 1)  vs  (s, alpha): (1, Q²)  →  X_arr: (size, Q²)
    """
    _ensure_dic_cache()
    L   = len(R_pts)
    Q2  = _QUAD_N * _QUAD_N

    # (L, 1) × (1, Q²) → (L, Q²)
    x   = a * R_pts[:, None]    * np.cos(THETA_pts[:, None])
    y   = b * R_pts[:, None]    * np.sin(THETA_pts[:, None])
    gi  = a * _DIC_X1_FLAT[None, :] * np.cos(_DIC_X2_FLAT[None, :])
    eta = b * _DIC_X1_FLAT[None, :] * np.sin(_DIC_X2_FLAT[None, :])

    R_arr  = np.sqrt((x - gi)**2 + (y - eta)**2)    # (L, Q²)
    X_arr  = K * R_arr
    Z_scal = 2.0 * K * depth      # scalar

    denom2 = X_arr**2 + Z_scal**2
    denom  = np.sqrt(denom2)

    M = (K**3) * (
        (2 * Z_scal - 1) / denom2**1.5 +
        3 * Z_scal**2    / denom2**2.5 +
        1 / denom
    ) + (K**3) / denom

    N_val = (K**3) * (
        2 * np.pi * 1j * hankel1(0, X_arr) * np.exp(-Z_scal)
        - (4 / np.pi) * intgralv(Z_scal, X_arr)   # Z scalar, X (L, Q²)
    )
    p3_all = M + N_val    # (L, Q²)

    # Fortran reshape per row then transpose ≡ C-order reshape:
    # p3[l, i + Q*j]  →  p4_F[l, i, j]  →  p4_F.T[l, j, i]
    # = p4_C[l, j, i]  which is just p3_all.reshape(L, Q, Q)
    return p3_all.reshape(L, _QUAD_N, _QUAD_N)   # (L, Q, Q)


def doubleintC(k, m, r_point, theta_point, depth, K, a, b, p4_T=None):
    """Compute the double integral for one (k,m,r,theta) combination."""
    _ensure_dic_cache()
    if p4_T is None:
        p4_T = doubleintC_precomputed(r_point, theta_point, depth, K, a, b)
    p1    = alform(k, m, _DIC_X1) * _DIC_X1
    p2    = np.cos(m * _DIC_X2)
    return a * b * np.sum(_DIC_W * np.outer(p1, p2) * p4_T)


# ──────────────────────────────────────────────────────────────────────
# getgl — cached by (l_array_tuple, a, b)
# ──────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _getgl_cached(l_tuple, a, b):
    x, w   = lgwt(_QUAD_N, -np.pi, np.pi)
    l_arr  = np.asarray(l_tuple, dtype=np.float64)
    gval1  = a * b * (a**2 * np.cos(x)**2 + b**2 * np.sin(x)**2)**(-1.5) * w
    gval2  = np.exp(-2j * x[:, None] * l_arr[None, :])
    return (1.0 / (2 * np.pi)) * np.sum(gval1[:, None] * gval2, axis=0)


def getgl(l_array, a, b):
    return _getgl_cached(tuple(np.asarray(l_array).tolist()), a, b)


# ──────────────────────────────────────────────────────────────────────
# hyperterm  (kept as a public utility; no longer called inside solver)
# ──────────────────────────────────────────────────────────────────────
def hyperterm(l1_array, k, m, r, theta):
    """Vectorized over l1_array; r, theta are scalars."""
    l1_array = np.asarray(l1_array, dtype=np.float64)
    is_scalar = l1_array.ndim == 0
    if is_scalar:
        l1_array = np.atleast_1d(l1_array)

    l_arr = (2 * l1_array).astype(int)

    g_km    = gamma(k + 1.5)
    g_km1   = gamma(k + m + 1)
    f_2k1   = factorial(2*k + 1,       exact=False)
    f_2k2m1 = factorial(2*k + 2*m + 1, exact=False)

    sign_C = (-1.0)**(l_arr + 2*m)
    Clmk   = sign_C * (
        (np.pi * 2.0**(l_arr + 2) / (l_arr**2 - 1))
        * (factorial(2*k - l_arr + 1,          exact=False)
           / factorial(2*k + 2*m + l_arr + 1, exact=False))
        * (f_2k2m1 / f_2k1) * (g_km / g_km1)
        * (gamma(k + l_arr/2 + m + 1.5) / gamma(k - l_arr/2 + 1))
    )

    sign_E = (-1.0)**(l_arr + m)
    Elkm   = sign_E * (
        (np.pi * 2.0**(l_arr - 2*m + 2) / (l_arr**2 - 1))
        * (g_km / g_km1) * (f_2k2m1 / f_2k1)
        * (gamma(k + l_arr/2 + 1.5) / gamma(m + k - l_arr/2 + 1))
        * (factorial(2*m + 2*k - l_arr + 1, exact=False)
           / factorial(l_arr + 2*k + 1,     exact=False))
    )

    denom = np.sqrt(1 - r**2)
    term2 = np.array([float(alform2(k, m, int(lv), r)) for lv in l_arr]) / denom
    term3 = np.array([float(alform3(k, m, int(lv), r)) for lv in l_arr]) / denom

    Xlkm = Clmk * term3 * np.exp(1j * (l_arr + m) * theta)
    Ylkm = Elkm * term2 * np.exp(1j * (l_arr - m) * theta)

    B = (Xlkm + Ylkm) / 2
    return B[0] if is_scalar else B


# ──────────────────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────────────────
def problemcodeAMDC(N, d_val, K, a, b):
    """
    Solve the rigid-elliptic-disc BVP.

    Pipeline
    --------
    Phase 1 : Batch check() for all size collocation points × Q² quadrature
              points in ONE vectorised call  → P4_mat  (size, Q²).
    Phase 2 : Build WPP_mat (size, Q²) and OP_mat (size, Q²) from alform
              bases; compute gl_cache.
    Phase 3 : c = (a·b) · P4_mat @ WPP_mat.T   [one BLAS dgemm]
    Phase 4 : A[:,col] = ht_mat.T @ gl per (k,m); alform2/3 evaluated on
              full R_flat (size,) at once, no outer l-loop.
    Phase 5 : solve C·X = f.
    Phase 6 : final = -dot(w.ravel(), (OP_mat.T @ X_sol)) / pi  [one matvec]
    """
    _ensure_intgral_cache()
    _ensure_dic_cache()

    n     = np.arange(N + 1)
    size  = (N + 1)**2

    theta = (2 * n + 1) * np.pi / (2 * N + 2)
    r     = np.cos(theta / 2)

    R_grid,     THETA_grid = np.meshgrid(r, theta, indexing='xy')
    R_flat     = R_grid.flatten(order='F')      # (size,)
    THETA_flat = THETA_grid.flatten(order='F')
    denom_flat = np.sqrt(1 - R_flat**2)         # (size,)

    # ── Phase 1: batch check() for all collocation points ────────────
    print("Phase 1: batch check() for all collocation points …")
    # P4_batch shape: (size, Q, Q)  ≡ p4.T for each collocation point l
    P4_batch = doubleintC_batch(R_flat, THETA_flat, d_val, K, a, b)
    P4_mat   = P4_batch.reshape(size, _QUAD_N * _QUAD_N)   # (size, Q²)

    # ── Phase 2: alform bases + gl_cache ─────────────────────────────
    print("Phase 2: alform bases and getgl …")
    km_pairs = [(k, m) for k in range(N + 1) for m in range(N + 1)]
    gl_cache = {}
    WPP_rows = np.empty((size, _QUAD_N * _QUAD_N), dtype=np.float64)
    OP_rows  = np.empty((size, _QUAD_N * _QUAD_N), dtype=np.float64)

    # Precompute cos(m·x2) for each unique m — reused across k values
    cosm_cache = {m: np.cos(m * _DIC_X2) for m in range(N + 1)}

    for idx, (k, m) in enumerate(km_pairs):
        gl_cache[(k, m)] = getgl(np.arange(-k, k + 1), a, b)

        p1 = alform(k, m, _DIC_X1) * _DIC_X1    # (Q,)
        p2 = cosm_cache[m]                        # (Q,)
        op = np.outer(p1, p2)                     # (Q, Q)
        WPP_rows[idx] = (_DIC_W * op).ravel()
        OP_rows[idx]  = op.ravel()

    # ── Phase 3: c matrix via one matrix multiply ─────────────────────
    print("Phase 3: c = (a·b)·P4_mat @ WPP_mat.T …")
    # P4_mat is complex128, WPP_rows is float64 → result complex128
    c = (a * b) * (P4_mat @ WPP_rows.T)          # (size, size)

    # ── Phase 4: A matrix — per-(k,m) column, vectorized over l ──────
    print("Phase 4: A matrix (vectorized over all collocation points) …")
    A = np.zeros((size, size), dtype=np.complex128)

    for col_idx, (k, m) in enumerate(km_pairs):
        l1_arr = np.arange(-k, k + 1, dtype=np.float64)
        l_arr  = (2 * l1_arr).astype(int)    # (L,)  L = 2k+1
        gl     = gl_cache[(k, m)]            # (L,)

        # Coefficients — use gammaln for stability at large k.
        # Note: (l_arr**2 - 1) can be negative (e.g. l=0 → value = -1);
        # we track its sign separately so the log is always of a positive number.
        denom_sign = np.sign(l_arr**2 - 1)   # ±1  (never 0 since l_arr is even)

        lg_km    = gammaln(k + 1.5)
        lg_km1   = gammaln(k + m + 1)
        lg_f2k1  = gammaln(2*k + 2)       # log(factorial(2k+1))
        lg_f2km1 = gammaln(2*k + 2*m + 2) # log(factorial(2k+2m+1))

        # Clmk
        sign_C = denom_sign * (-1.0)**(l_arr + 2*m)
        log_C  = (
            np.log(np.pi) + (l_arr + 2) * np.log(2.0)
            - np.log(np.abs(l_arr**2 - 1))
            + gammaln(2*k - l_arr + 2)         # log factorial(2k-l+1)
            - gammaln(2*k + 2*m + l_arr + 2)   # log factorial(2k+2m+l+1)
            + lg_f2km1 - lg_f2k1
            + lg_km - lg_km1
            + gammaln(k + l_arr/2 + m + 1.5)
            - gammaln(k - l_arr/2 + 1)
        )
        Clmk = sign_C * np.exp(log_C)

        # Elkm
        sign_E = denom_sign * (-1.0)**(l_arr + m)
        log_E  = (
            np.log(np.pi) + (l_arr - 2*m + 2) * np.log(2.0)
            - np.log(np.abs(l_arr**2 - 1))
            + lg_km - lg_km1
            + lg_f2km1 - lg_f2k1
            + gammaln(k + l_arr/2 + 1.5)
            - gammaln(m + k - l_arr/2 + 1)
            + gammaln(2*m + 2*k - l_arr + 2)   # log factorial(2m+2k-l+1)
            - gammaln(l_arr + 2*k + 2)         # log factorial(l+2k+1)
        )
        Elkm = sign_E * np.exp(log_E)

        # alform2/3 evaluated at ALL collocation points at once  (L, size)
        term2_mat = np.stack([
            _eval_lambdify(_get_alform2_func(k, m, int(lv)), R_flat)
            for lv in l_arr
        ])  # (L, size)

        term3_mat = np.stack([
            _eval_lambdify(_get_alform3_func(k, m, int(lv)), R_flat)
            for lv in l_arr
        ])  # (L, size)

        # Phase factors — factor out exp(±1j·m·THETA) to reuse across l
        exp_m_pos = np.exp( 1j * m * THETA_flat)   # (size,)
        exp_m_neg = np.exp(-1j * m * THETA_flat)   # (size,)
        exp_l     = np.exp( 1j * l_arr[:, None] * THETA_flat[None, :])  # (L, size)

        # ht_mat[i, l] = (Clmk[i]*term3[i,l]*exp_l[i,l]*exp_m_pos[l]
        #               + Elkm[i]*term2[i,l]*exp_l[i,l]*exp_m_neg[l]) / (2*denom[l])
        ht_mat = (
            Clmk[:, None] * term3_mat * exp_l * exp_m_pos[None, :]
            + Elkm[:, None] * term2_mat * exp_l * exp_m_neg[None, :]
        ) / (2.0 * denom_flat[None, :])    # (L, size)

        A[:, col_idx] = ht_mat.T @ gl     # (size, L) @ (L,) → (size,)

    # ── Phase 5: solve ────────────────────────────────────────────────
    print("Phase 5: solving linear system …")
    C     = A + c
    f_vec = 4 * np.pi * np.ones(size, dtype=np.complex128)
    X_sol = np.linalg.solve(C, f_vec)

    # ── Phase 6: final sum — one matvec ──────────────────────────────
    # sum1[i,j] = Σ_p X_sol[p] · OP_rows[p, flat(i,j)]
    #           = reshape(OP_rows.T @ X_sol, (Q, Q))
    sum1 = (OP_rows.T @ X_sol).reshape((_QUAD_N, _QUAD_N))

    _, w1 = lgwt(_QUAD_N, 0, 1)
    _, w2 = lgwt(_QUAD_N, 0, 2 * np.pi)
    w     = np.outer(w1, w2)
    final = -np.sum(w * sum1) / np.pi    # a·b / (π·a·b) = 1/π

    return final


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    N     = 2
    d_val = 1.0
    K     = 1.0
    a     = 1.0
    b     = 1.0

    t0 = time.perf_counter()
    result = problemcodeAMDC(N, d_val, K, a, b)
    t1 = time.perf_counter()
    print(f"\nResult : {result}")
    print(f"Elapsed: {t1 - t0:.3f} s")
