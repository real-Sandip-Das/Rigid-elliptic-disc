"""
problemcodeAMDC_opt.py  —  Optimized port of problemcodeAMDC.m
=============================================================

Key optimisations vs the original problemcodeAMDC.py
------------------------------------------------------
1.  lgwt cached with lru_cache — nodes computed once per (N,a,b).
2.  k0(x) instead of kn(0,x) — specialized K₀ implementation, faster.
3.  j0(x)+1j*y0(x) instead of hankel1(0,x) — specialized Bessel fns.
4.  intgralv scalar-Z fast path: Z-dependent trig weights cached once;
    only k0 evaluations remain in the hot path.
5.  Phase 1: all size collocation points batched into one check() call
    (size×Q² array) — single intgralv call for entire P4_mat.
6.  Phase 2.5 (eager pre-computation before the A-loop):
      a. All lambdified alform / alform2 / alform3 functions compiled
         AND evaluated on R_flat in one pass — sympy diff called here,
         never inside Phase 4.
      b. Clmk / Elkm coefficients pre-computed for every (k,m) pair.
      c. Phase factor table exp(1j·l·θ) for all unique l values.
      d. exp(±1j·m·θ) for each unique m.
7.  Phase 3: c = (a·b)·P4_mat @ WPP_mat.T — one BLAS dgemm.
8.  Phase 4: pure array lookups + matrix ops — no sympy, no exp, no
    gammaln calls inside the loop.
9.  Phase 6: final sum = (OP_mat.T @ X_sol).reshape(Q,Q) — one matvec.
10. a·b / (π·a·b) simplified to 1/π in the final formula.
"""

import numpy as np
import sympy as sp
from scipy.special import k0, j0, y0, gamma, gammaln, factorial
import functools

# ──────────────────────────────────────────────────────────────────────
# Gauss-Legendre quadrature  (cached by (N, a, b))
# ──────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _legendre_nodes(N, a, b):
    x, w = np.polynomial.legendre.leggauss(N)
    return 0.5*(a*(1-x) + b*(1+x)), w*0.5*(b-a)


def lgwt(N, a, b):
    return _legendre_nodes(N, a, b)


# ──────────────────────────────────────────────────────────────────────
# Module-level shared quadrature grids  (lazy init)
# ──────────────────────────────────────────────────────────────────────
_QUAD_N = 100

_INTGRAL_X = _INTGRAL_W = _INTGRAL_TX = None   # intgralv nodes [0, pi/2]
_DIC_X1 = _DIC_W1 = _DIC_X2 = _DIC_W2 = None   # doubleintC nodes
_DIC_W = _DIC_X1_FLAT = _DIC_X2_FLAT = None

# Cache for Z-dependent effective weights in intgralv
# key: float(Z),  value: (Q,) real array
_weff_cache: dict = {}


def _ensure_intgral_cache():
    global _INTGRAL_X, _INTGRAL_W, _INTGRAL_TX
    if _INTGRAL_X is None:
        _INTGRAL_X, _INTGRAL_W = lgwt(_QUAD_N, 0.0, np.pi / 2)
        _INTGRAL_TX = np.tan(_INTGRAL_X)


def _ensure_dic_cache():
    global _DIC_X1, _DIC_W1, _DIC_X2, _DIC_W2, _DIC_W, _DIC_X1_FLAT, _DIC_X2_FLAT
    if _DIC_X1 is None:
        _DIC_X1, _DIC_W1 = lgwt(_QUAD_N, 0.0, 1.0)
        _DIC_X2, _DIC_W2 = lgwt(_QUAD_N, 0.0, 2.0*np.pi)
        _DIC_W = np.outer(_DIC_W1, _DIC_W2)
        S, A = np.meshgrid(_DIC_X1, _DIC_X2, indexing='xy')
        _DIC_X1_FLAT = S.flatten(order='F')
        _DIC_X2_FLAT = A.flatten(order='F')


# ──────────────────────────────────────────────────────────────────────
# Symbolic lambdify cache  (alform / alform2 / alform3)
# ──────────────────────────────────────────────────────────────────────
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
    coeff = ((-1.0)**m1) / ((2**l1) * sp.factorial(l1))
    P1 = coeff * (1-r**2)**(sp.Rational(m1,2)) * sp.diff((r**2-1)**l1, r, l1+m1)
    return sp.lambdify(r, P1.subs(r, sp.sqrt(1-r**2)), modules='numpy')


@functools.lru_cache(maxsize=None)
def _get_alform3_func(k, m, l_val):
    r  = sp.Symbol('r')
    l1 = int(m + 2*k + 1);  m1 = int(m + l_val)
    coeff = ((-1.0)**m1) / ((2**l1) * sp.factorial(l1))
    P1 = coeff * (1-r**2)**(sp.Rational(m1,2)) * sp.diff((r**2-1)**l1, r, l1+m1)
    return sp.lambdify(r, P1.subs(r, sp.sqrt(1-r**2)), modules='numpy')


def _eval_lambdify(func, s):
    """Evaluate a lambdified sympy func; broadcast scalars to array shape."""
    res = func(s)
    if np.isscalar(res) and np.ndim(s) > 0:
        return np.full(np.shape(s), float(res), dtype=np.float64)
    return np.asarray(res, dtype=np.float64)


def alform (k, m,       s): return _eval_lambdify(_get_alform_func (k, m),        s)
def alform2(k, m, l, s):    return _eval_lambdify(_get_alform2_func(k, m, l),     s)
def alform3(k, m, l, s):    return _eval_lambdify(_get_alform3_func(k, m, l),     s)


# ──────────────────────────────────────────────────────────────────────
# intgralv — vectorized; scalar-Z fast path with cached trig weights
# ──────────────────────────────────────────────────────────────────────
def intgralv(Z, X):
    """
    Gauss-Legendre approximation of the integral on [0, π/2].
    Scalar-Z fast path: Z-dependent weights w_eff = w·(tx·sin(Z·tx)+cos(Z·tx))
    cached after first call, so only k0 evaluations remain hot.
    """
    _ensure_intgral_cache()
    Z = np.asarray(Z)
    X = np.asarray(X)
    tx  = _INTGRAL_TX   # (Q,)
    w_q = _INTGRAL_W    # (Q,)

    if Z.ndim == 0:                          # scalar Z — common case
        Z_f = float(Z)
        if Z_f not in _weff_cache:
            _weff_cache[Z_f] = w_q * (tx * np.sin(Z_f * tx) + np.cos(Z_f * tx))
        w_eff = _weff_cache[Z_f]            # (Q,)

        ndim = X.ndim
        tx_ = tx.reshape((len(tx),) + (1,)*ndim)      # (Q, 1, ..., 1)
        w_  = w_eff.reshape((len(w_eff),) + (1,)*ndim)
        return np.sum(w_ * k0(tx_ * X[np.newaxis, ...]), axis=0)
    else:                                    # array Z — general path
        out_shape = np.broadcast_shapes(Z.shape, X.shape)
        ndim_out  = len(out_shape)
        ax  = (slice(None),) + (np.newaxis,)*ndim_out
        tx_ = tx[ax];  w_ = w_q[ax]
        Z_  = np.broadcast_to(Z, out_shape)[np.newaxis, ...]
        X_  = np.broadcast_to(X, out_shape)[np.newaxis, ...]
        return np.sum(w_ * (tx_*np.sin(Z_*tx_) + np.cos(tx_*Z_)) * k0(tx_*X_), axis=0)


# ──────────────────────────────────────────────────────────────────────
# check — uses j0/y0 instead of hankel1, k0 instead of kn
# ──────────────────────────────────────────────────────────────────────
def check(r, theta, s, alpha, d_val, K, a, b):
    x   = a * r * np.cos(theta);   y   = b * r * np.sin(theta)
    gi  = a * s * np.cos(alpha);   eta = b * s * np.sin(alpha)

    R      = np.sqrt((x-gi)**2 + (y-eta)**2)
    X_arr  = K * R
    Z_scal = 2.0 * K * d_val           # scalar: -K*((-d)+(-d))

    denom2 = X_arr**2 + Z_scal**2
    denom  = np.sqrt(denom2)

    M = (K**3) * (
        (2*Z_scal - 1) / denom2**1.5 +
         3*Z_scal**2   / denom2**2.5 +
         1 / denom
    ) + (K**3) / denom

    # hankel1(0,x) = j0(x) + 1j*y0(x);  specialized fns are faster
    N_val = (K**3) * (
        2*np.pi*1j * (j0(X_arr) + 1j*y0(X_arr)) * np.exp(-Z_scal)
        - (4/np.pi) * intgralv(Z_scal, X_arr)
    )
    return M + N_val


# ──────────────────────────────────────────────────────────────────────
# kernel  (kept for API compatibility)
# ──────────────────────────────────────────────────────────────────────
def kernel(x, y, s, alpha, K, a, b, d_val):
    gi  = a*s*np.cos(alpha);   eta = b*s*np.sin(alpha)
    R   = np.sqrt((x-gi)**2 + (y-eta)**2)
    X   = K*R;  Y = K*d_val
    denom2 = X**2 + Y**2
    return (K**2) * (
        2*Y / denom2**1.5 + 2/np.sqrt(denom2) +
        2*np.pi*1j * np.exp(-Y) * (j0(X) + 1j*y0(X))
        - (4/np.pi) * intgralv(Y, X)
    )


# ──────────────────────────────────────────────────────────────────────
# doubleintC helpers
# ──────────────────────────────────────────────────────────────────────
def doubleintC_precomputed(r_pt, theta_pt, depth, K, a, b):
    """check() for one collocation point → (Q,Q) array (≡ p4.T)."""
    _ensure_dic_cache()
    p3 = check(r_pt, theta_pt, _DIC_X1_FLAT, _DIC_X2_FLAT, depth, K, a, b)
    return p3.reshape((_QUAD_N, _QUAD_N))   # C-reshape = Fortran-reshape.T


def doubleintC_batch(R_pts, THETA_pts, depth, K, a, b):
    """
    Batch check() for ALL len(R_pts) collocation points × Q² quadrature pts.
    Returns P4_T_batch of shape (size, Q, Q).
    Broadcasting: (size,1) vs (1,Q²) → X_arr (size,Q²).
    """
    _ensure_dic_cache()
    L  = len(R_pts)
    x   = a * R_pts[:, None]          * np.cos(THETA_pts[:, None])
    y   = b * R_pts[:, None]          * np.sin(THETA_pts[:, None])
    gi  = a * _DIC_X1_FLAT[None, :]   * np.cos(_DIC_X2_FLAT[None, :])
    eta = b * _DIC_X1_FLAT[None, :]   * np.sin(_DIC_X2_FLAT[None, :])

    X_arr  = K * np.sqrt((x-gi)**2 + (y-eta)**2)   # (L, Q²)
    Z_scal = 2.0 * K * depth

    denom2 = X_arr**2 + Z_scal**2
    denom  = np.sqrt(denom2)
    M = (K**3)*((2*Z_scal-1)/denom2**1.5 + 3*Z_scal**2/denom2**2.5 + 1/denom) \
        + (K**3)/denom
    N_val = (K**3) * (
        2*np.pi*1j*(j0(X_arr)+1j*y0(X_arr))*np.exp(-Z_scal)
        - (4/np.pi)*intgralv(Z_scal, X_arr)
    )
    return (M + N_val).reshape(L, _QUAD_N, _QUAD_N)   # (L, Q, Q)


def doubleintC(k, m, r_pt, theta_pt, depth, K, a, b, p4_T=None):
    _ensure_dic_cache()
    if p4_T is None:
        p4_T = doubleintC_precomputed(r_pt, theta_pt, depth, K, a, b)
    p1 = alform(k, m, _DIC_X1) * _DIC_X1
    p2 = np.cos(m * _DIC_X2)
    return a*b * np.sum(_DIC_W * np.outer(p1, p2) * p4_T)


# ──────────────────────────────────────────────────────────────────────
# getgl — cached by (l_tuple, a, b)
# ──────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _getgl_cached(l_tuple, a, b):
    x, w  = lgwt(_QUAD_N, -np.pi, np.pi)
    l_arr = np.asarray(l_tuple, dtype=np.float64)
    gval1 = a*b * (a**2*np.cos(x)**2 + b**2*np.sin(x)**2)**(-1.5) * w
    gval2 = np.exp(-2j * x[:, None] * l_arr[None, :])
    return (1/(2*np.pi)) * np.sum(gval1[:, None] * gval2, axis=0)


def getgl(l_array, a, b):
    return _getgl_cached(tuple(np.asarray(l_array).tolist()), a, b)


# ──────────────────────────────────────────────────────────────────────
# hyperterm  (public utility; not called by the solver)
# ──────────────────────────────────────────────────────────────────────
def hyperterm(l1_array, k, m, r, theta):
    l1_array = np.asarray(l1_array, dtype=np.float64)
    is_scalar = l1_array.ndim == 0
    if is_scalar: l1_array = np.atleast_1d(l1_array)
    l_arr = (2*l1_array).astype(int)

    g_km = gamma(k+1.5);  g_km1 = gamma(k+m+1)
    f_2k1 = factorial(2*k+1, exact=False);  f_2k2m1 = factorial(2*k+2*m+1, exact=False)

    sign_C = (-1.0)**(l_arr+2*m)
    Clmk = sign_C * (
        (np.pi * 2.0**(l_arr+2) / (l_arr**2-1))
        * (factorial(2*k-l_arr+1,exact=False) / factorial(2*k+2*m+l_arr+1,exact=False))
        * (f_2k2m1/f_2k1) * (g_km/g_km1)
        * (gamma(k+l_arr/2+m+1.5) / gamma(k-l_arr/2+1))
    )
    sign_E = (-1.0)**(l_arr+m)
    Elkm = sign_E * (
        (np.pi * 2.0**(l_arr-2*m+2) / (l_arr**2-1))
        * (g_km/g_km1) * (f_2k2m1/f_2k1)
        * (gamma(k+l_arr/2+1.5) / gamma(m+k-l_arr/2+1))
        * (factorial(2*m+2*k-l_arr+1,exact=False) / factorial(l_arr+2*k+1,exact=False))
    )
    denom = np.sqrt(1-r**2)
    term2 = np.array([float(alform2(k, m, int(lv), r)) for lv in l_arr]) / denom
    term3 = np.array([float(alform3(k, m, int(lv), r)) for lv in l_arr]) / denom
    B = (Clmk*term3*np.exp(1j*(l_arr+m)*theta) + Elkm*term2*np.exp(1j*(l_arr-m)*theta)) / 2
    return B[0] if is_scalar else B


# ──────────────────────────────────────────────────────────────────────
# _precompute_A_data
# Build all data needed for Phase 4 BEFORE the A-loop:
#   • all lambdified alform/alform2/alform3 compiled (sympy diff done here)
#   • alform2/3 evaluated on R_flat for every (k,m,l_val)
#   • Clmk, Elkm arrays for every (k,m)
#   • Phase factor table exp(1j·l·θ) for all unique l values
#   • exp(±1j·m·θ) for each unique m
# ──────────────────────────────────────────────────────────────────────
def _precompute_A_data(km_pairs, N, R_flat, THETA_flat, a, b):
    """Returns (gl_cache, alform2_tbl, alform3_tbl, coeff_tbl,
                exp_l_table, l_to_row, exp_m_pos, exp_m_neg,
                WPP_rows, OP_rows, alform_x1_cache, cosm_cache)."""

    size = len(R_flat)

    # ── getgl ────────────────────────────────────────────────────────
    gl_cache = {(k, m): getgl(np.arange(-k, k+1), a, b) for k, m in km_pairs}

    # ── Basis matrices WPP_rows / OP_rows ────────────────────────────
    _ensure_dic_cache()
    cosm_cache     = {m: np.cos(m * _DIC_X2) for m in range(N+1)}
    alform_x1_cache = {}
    WPP_rows = np.empty((len(km_pairs), _QUAD_N * _QUAD_N), dtype=np.float64)
    OP_rows  = np.empty((len(km_pairs), _QUAD_N * _QUAD_N), dtype=np.float64)

    for idx, (k, m) in enumerate(km_pairs):
        p1 = alform(k, m, _DIC_X1) * _DIC_X1
        p2 = cosm_cache[m]
        alform_x1_cache[(k, m)] = p1
        op = np.outer(p1, p2)
        WPP_rows[idx] = (_DIC_W * op).ravel()
        OP_rows[idx]  = op.ravel()

    # ── Phase factor table: exp(1j·l·θ) for all unique l values ──────
    l_unique = np.arange(-2*N, 2*N+1, 2, dtype=int)   # (2N+1,)
    l_to_row = {int(l): i for i, l in enumerate(l_unique)}
    # Compute exp table in one batch — shape (2N+1, size)
    exp_l_table = np.exp(1j * l_unique[:, None] * THETA_flat[None, :])

    # ── exp(±1j·m·θ) for each m ──────────────────────────────────────
    exp_m_pos = np.exp( 1j * np.arange(N+1)[:, None] * THETA_flat[None, :])  # (N+1, size)
    exp_m_neg = np.exp(-1j * np.arange(N+1)[:, None] * THETA_flat[None, :])  # (N+1, size)

    # ── Eagerly compile + evaluate alform2/3 lambdas on R_flat ────────
    # This is where sympy diff() is called — done ONCE here, never in Phase 4.
    alform2_tbl = {}   # (k, m, l_val_int) -> (size,)
    alform3_tbl = {}
    coeff_tbl   = {}   # (k, m) -> (Clmk, Elkm)

    for k, m in km_pairs:
        l1_arr = np.arange(-k, k+1, dtype=np.float64)
        l_arr  = (2*l1_arr).astype(int)

        # Alform2/3 on R_flat — lambdify compiled here (cached for later calls)
        for lv in l_arr:
            key = (k, m, int(lv))
            alform2_tbl[key] = _eval_lambdify(_get_alform2_func(k, m, int(lv)), R_flat)
            alform3_tbl[key] = _eval_lambdify(_get_alform3_func(k, m, int(lv)), R_flat)

        # Clmk, Elkm — gammaln for numerical stability
        ds   = np.sign(l_arr**2 - 1)    # sign of denominator
        lgkm  = gammaln(k+1.5);  lgkm1 = gammaln(k+m+1)
        lf2k1 = gammaln(2*k+2);  lf2km1 = gammaln(2*k+2*m+2)

        log_C = (np.log(np.pi) + (l_arr+2)*np.log(2.0) - np.log(np.abs(l_arr**2-1))
                 + gammaln(2*k-l_arr+2) - gammaln(2*k+2*m+l_arr+2)
                 + lf2km1 - lf2k1 + lgkm - lgkm1
                 + gammaln(k+l_arr/2+m+1.5) - gammaln(k-l_arr/2+1))
        Clmk = ds * (-1.0)**(l_arr+2*m) * np.exp(log_C)

        log_E = (np.log(np.pi) + (l_arr-2*m+2)*np.log(2.0) - np.log(np.abs(l_arr**2-1))
                 + lgkm - lgkm1 + lf2km1 - lf2k1
                 + gammaln(k+l_arr/2+1.5) - gammaln(m+k-l_arr/2+1)
                 + gammaln(2*m+2*k-l_arr+2) - gammaln(l_arr+2*k+2))
        Elkm = ds * (-1.0)**(l_arr+m) * np.exp(log_E)

        coeff_tbl[(k, m)] = (Clmk, Elkm, l_arr)

    return (gl_cache, alform2_tbl, alform3_tbl, coeff_tbl,
            exp_l_table, l_to_row, exp_m_pos, exp_m_neg,
            WPP_rows, OP_rows)


# ──────────────────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────────────────
def problemcodeAMDC(N, d_val, K, a, b):
    """
    Solve the rigid-elliptic-disc BVP.

    Pipeline
    --------
    Phase 1  : Batch check() for ALL size collocation × Q² quadrature pts
               in ONE vectorised call → P4_mat (size, Q²).
    Phase 2  : Eager pre-computation of ALL data needed for Phase 4:
               alform2/3 on R_flat, coefficients, phase table.
    Phase 3  : c = (a·b) · P4_mat @ WPP_mat.T          [one dgemm]
    Phase 4  : A columns — pure array lookups + matvec, NO sympy calls.
    Phase 5  : solve  C·X = f.
    Phase 6  : final = -(w·(OP_mat.T @ X_sol)).sum() / π [one matvec]
    """
    _ensure_intgral_cache()
    _ensure_dic_cache()

    n     = np.arange(N + 1)
    size  = (N + 1)**2

    theta = (2*n + 1) * np.pi / (2*N + 2)
    r     = np.cos(theta / 2)

    R_grid, THETA_grid = np.meshgrid(r, theta, indexing='xy')
    R_flat     = R_grid.flatten(order='F')
    THETA_flat = THETA_grid.flatten(order='F')
    denom_flat = np.sqrt(1 - R_flat**2)

    km_pairs = [(k, m) for k in range(N+1) for m in range(N+1)]

    # ── Phase 1: batch check() ────────────────────────────────────────
    print("Phase 1 : batch check() for all collocation points …")
    P4_batch = doubleintC_batch(R_flat, THETA_flat, d_val, K, a, b)
    P4_mat   = P4_batch.reshape(size, _QUAD_N * _QUAD_N)   # (size, Q²), complex128

    # ── Phase 2: eager pre-computation ───────────────────────────────
    print("Phase 2 : pre-computing alform2/3, coefficients, phase table …")
    (gl_cache, alform2_tbl, alform3_tbl, coeff_tbl,
     exp_l_table, l_to_row,
     exp_m_pos, exp_m_neg,
     WPP_rows, OP_rows) = _precompute_A_data(km_pairs, N, R_flat, THETA_flat, a, b)

    # ── Phase 3: c = (a·b)·P4_mat @ WPP_rows.T ───────────────────────
    print("Phase 3 : c matrix via matrix multiply …")
    c = (a * b) * (P4_mat @ WPP_rows.T)        # (size, size), complex128

    # ── Phase 4: A matrix — pure array ops, no sympy, no exp ─────────
    print("Phase 4 : A matrix (pure array ops) …")
    A = np.zeros((size, size), dtype=np.complex128)

    for col_idx, (k, m) in enumerate(km_pairs):
        Clmk, Elkm, l_arr = coeff_tbl[(k, m)]              # (L,) each
        gl                 = gl_cache[(k, m)]               # (L,)

        # Array lookups — no computation
        term2_mat = np.stack([alform2_tbl[(k, m, int(lv))] for lv in l_arr])  # (L, size)
        term3_mat = np.stack([alform3_tbl[(k, m, int(lv))] for lv in l_arr])  # (L, size)
        rows      = [l_to_row[int(lv)] for lv in l_arr]
        exp_l     = exp_l_table[rows]                       # (L, size)
        emp       = exp_m_pos[m]                            # (size,)
        emn       = exp_m_neg[m]                            # (size,)

        ht_mat = (
            Clmk[:, None] * term3_mat * exp_l * emp[None, :]
            + Elkm[:, None] * term2_mat * exp_l * emn[None, :]
        ) / (2.0 * denom_flat[None, :])                    # (L, size)

        A[:, col_idx] = ht_mat.T @ gl                      # (size,)

    # ── Phase 5: solve ────────────────────────────────────────────────
    print("Phase 5 : solving linear system …")
    C     = A + c
    f_vec = 4 * np.pi * np.ones(size, dtype=np.complex128)
    X_sol = np.linalg.solve(C, f_vec)

    # ── Phase 6: final sum — one matvec ──────────────────────────────
    sum1  = (OP_rows.T @ X_sol).reshape((_QUAD_N, _QUAD_N))
    _, w1 = lgwt(_QUAD_N, 0, 1);  _, w2 = lgwt(_QUAD_N, 0, 2*np.pi)
    w     = np.outer(w1, w2)
    return -np.sum(w * sum1) / np.pi


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    N = 2;  d_val = K = a = b = 1.0
    t0 = time.perf_counter()
    result = problemcodeAMDC(N, d_val, K, a, b)
    t1 = time.perf_counter()
    print(f"\nResult : {result}")
    print(f"Elapsed: {t1-t0:.3f} s")
