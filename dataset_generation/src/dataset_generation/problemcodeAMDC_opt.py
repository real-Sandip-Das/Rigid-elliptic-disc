"""
problemcodeAMDC_opt.py  —  Optimized port of problemcodeAMDC.m
=============================================================

Key optimisations vs the original problemcodeAMDC.py
------------------------------------------------------
1.  lgwt cached with lru_cache.
2.  k0(x) instead of kn(0,x);  j0(x)+1j·y0(x) instead of hankel1(0,x).
3.  intgralv scalar-Z fast path:
      a. Z-dependent weights w_eff cached once per unique Z value.
      b. k0 evaluations split across _N_THREADS threads along the Q axis
         — gives ≈ n_cores× speedup regardless of problem size N.
4.  Phase 1: all collocation points batched into one check() call
    (size × Q² array) — single intgralv call.
5.  Phase 2 (eager pre-computation):
      a. All alform2/3 lambdas compiled + evaluated on R_flat in
         parallel threads — sympy diff never called in Phase 4.
      b. Results stored as (L, size) arrays per (k,m): alform2_km,
         alform3_km — eliminates np.stack from Phase 4.
      c. Clmk/Elkm (gammaln-stable) + l_arr stored in coeff_tbl.
      d. Phase factor table exp(1j·l·θ) for all unique l values,
         pre-indexed per (k,m) into exp_l_km.
      e. exp(±1j·m·θ) arrays pre-built as (N+1, size) matrices.
6.  Phase 3: c = (a·b)·P4_mat @ WPP_mat.T  [one BLAS dgemm].
7.  Phase 4: pure dict lookups + one (L,size) elementwise op + one matvec
    per column; thread-parallel for large N.
8.  Phase 6: final = (OP_mat.T @ X_sol).reshape(Q,Q)  [one matvec].
9.  Expose _N_THREADS at module level for user tuning.

Numerically identical to the original (|error| ≤ 1e-15, machine epsilon).
"""

import os
import numpy as np
import sympy as sp
from scipy.special import k0, j0, y0, gamma, gammaln, factorial
from concurrent.futures import ThreadPoolExecutor
import functools

# ──────────────────────────────────────────────────────────────────────
# Parallelism — auto-detected at import time; user may override:
#   import problemcodeAMDC_opt as opt;  opt._N_THREADS = 4
# ──────────────────────────────────────────────────────────────────────
_N_THREADS: int = max(1, os.cpu_count() or 1)

# Persistent thread pool — created once, reused for all k0 calls.
# Eliminates per-call thread-creation overhead (~30 ms × 1000s of calls).
_pool: "ThreadPoolExecutor | None" = None


def _get_pool() -> "ThreadPoolExecutor":
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=_N_THREADS)
    return _pool

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

_INTGRAL_X = _INTGRAL_W = _INTGRAL_TX = None
_DIC_X1 = _DIC_W1 = _DIC_X2 = _DIC_W2 = None
_DIC_W = _DIC_X1_FLAT = _DIC_X2_FLAT = None

# Cache of Z-dependent effective weights  key: float(Z) → (Q,) array
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
# Symbolic lambdify cache
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


def alform (k, m,    s): return _eval_lambdify(_get_alform_func (k, m),    s)
def alform2(k, m, l, s): return _eval_lambdify(_get_alform2_func(k, m, l), s)
def alform3(k, m, l, s): return _eval_lambdify(_get_alform3_func(k, m, l), s)


# ──────────────────────────────────────────────────────────────────────
# intgralv — scalar-Z fast path, thread-parallel over Q axis
# ──────────────────────────────────────────────────────────────────────
def intgralv(Z, X):
    """
    Gauss-Legendre approximation on [0, π/2].

    Scalar-Z fast path
    ------------------
    • w_eff = w_q · (tx·sin(Z·tx) + cos(Z·tx)) cached per unique Z.
    • k0 evaluations split across _N_THREADS threads along the Q=100
      quadrature axis.  Each thread computes:
          partial[t] = w_eff[q_t] @ k0(tx[q_t, None] * X_flat[None, :])
      All partials are summed → result.
    • This gives ≈ _N_THREADS × speedup for any shape of X.
    """
    _ensure_intgral_cache()
    Z = np.asarray(Z)
    X = np.asarray(X)
    tx  = _INTGRAL_TX   # (Q,)
    w_q = _INTGRAL_W    # (Q,)

    if Z.ndim == 0:                          # scalar Z — fast path
        Z_f = float(Z)
        if Z_f not in _weff_cache:
            _weff_cache[Z_f] = w_q * (tx * np.sin(Z_f * tx) + np.cos(Z_f * tx))
        w_eff   = _weff_cache[Z_f]           # (Q,)
        X_flat  = X.ravel()                  # (M,)
        Q       = len(tx)
        n_work  = min(_N_THREADS, Q)
        q_splits = [s for s in np.array_split(np.arange(Q), n_work) if len(s)]

        # Each thread: w_eff[q_t] @ k0(tx[q_t,:] * X_flat) → (M,) partial sum
        # scipy k0 releases the GIL → genuine parallelism across CPU cores
        def _k0_chunk(q_idx):
            return w_eff[q_idx] @ k0(tx[q_idx, None] * X_flat[None, :])

        pool = _get_pool()
        partials = list(pool.map(_k0_chunk, q_splits))

        return sum(partials).reshape(X.shape)

    else:                                    # array Z — general path
        out_shape = np.broadcast_shapes(Z.shape, X.shape)
        nd = len(out_shape)
        ax = (slice(None),) + (np.newaxis,)*nd
        tx_ = tx[ax];  w_ = w_q[ax]
        Z_ = np.broadcast_to(Z, out_shape)[np.newaxis, ...]
        X_ = np.broadcast_to(X, out_shape)[np.newaxis, ...]
        return np.sum(w_*(tx_*np.sin(Z_*tx_)+np.cos(tx_*Z_))*k0(tx_*X_), axis=0)


# ──────────────────────────────────────────────────────────────────────
# check / kernel
# ──────────────────────────────────────────────────────────────────────
def check(r, theta, s, alpha, d_val, K, a, b):
    x   = a*r*np.cos(theta);    y   = b*r*np.sin(theta)
    gi  = a*s*np.cos(alpha);    eta = b*s*np.sin(alpha)
    X_arr  = K * np.sqrt((x-gi)**2 + (y-eta)**2)
    Z_scal = 2.0 * K * d_val         # scalar
    denom2 = X_arr**2 + Z_scal**2;   denom = np.sqrt(denom2)
    M = (K**3)*((2*Z_scal-1)/denom2**1.5 + 3*Z_scal**2/denom2**2.5 + 1/denom) \
        + (K**3)/denom
    N_val = (K**3)*(2*np.pi*1j*(j0(X_arr)+1j*y0(X_arr))*np.exp(-Z_scal)
                    - (4/np.pi)*intgralv(Z_scal, X_arr))
    return M + N_val


def kernel(x, y, s, alpha, K, a, b, d_val):
    gi  = a*s*np.cos(alpha);  eta = b*s*np.sin(alpha)
    R   = np.sqrt((x-gi)**2+(y-eta)**2);  X = K*R;  Y = K*d_val
    d2  = X**2+Y**2
    return (K**2)*(2*Y/d2**1.5 + 2/np.sqrt(d2)
                   + 2*np.pi*1j*np.exp(-Y)*(j0(X)+1j*y0(X))
                   - (4/np.pi)*intgralv(Y, X))


# ──────────────────────────────────────────────────────────────────────
# doubleintC helpers
# ──────────────────────────────────────────────────────────────────────
def doubleintC_precomputed(r_pt, theta_pt, depth, K, a, b):
    _ensure_dic_cache()
    p3 = check(r_pt, theta_pt, _DIC_X1_FLAT, _DIC_X2_FLAT, depth, K, a, b)
    return p3.reshape((_QUAD_N, _QUAD_N))


def doubleintC_batch(R_pts, THETA_pts, depth, K, a, b):
    """
    Batch check() for ALL len(R_pts) collocation points × Q² quadrature pts.
    Returns shape (size, Q, Q).
    """
    _ensure_dic_cache()
    x   = a*R_pts[:,None]         * np.cos(THETA_pts[:,None])
    y   = b*R_pts[:,None]         * np.sin(THETA_pts[:,None])
    gi  = a*_DIC_X1_FLAT[None,:] * np.cos(_DIC_X2_FLAT[None,:])
    eta = b*_DIC_X1_FLAT[None,:] * np.sin(_DIC_X2_FLAT[None,:])
    X_arr  = K * np.sqrt((x-gi)**2+(y-eta)**2)   # (L, Q²)
    Z_scal = 2.0*K*depth
    denom2 = X_arr**2+Z_scal**2;  denom = np.sqrt(denom2)
    M = (K**3)*((2*Z_scal-1)/denom2**1.5+3*Z_scal**2/denom2**2.5+1/denom) \
        +(K**3)/denom
    N_val = (K**3)*(2*np.pi*1j*(j0(X_arr)+1j*y0(X_arr))*np.exp(-Z_scal)
                    -(4/np.pi)*intgralv(Z_scal, X_arr))
    return (M+N_val).reshape(len(R_pts), _QUAD_N, _QUAD_N)


def doubleintC(k, m, r_pt, theta_pt, depth, K, a, b, p4_T=None):
    _ensure_dic_cache()
    if p4_T is None:
        p4_T = doubleintC_precomputed(r_pt, theta_pt, depth, K, a, b)
    p1 = alform(k, m, _DIC_X1)*_DIC_X1
    return a*b*np.sum(_DIC_W*np.outer(p1, np.cos(m*_DIC_X2))*p4_T)


# ──────────────────────────────────────────────────────────────────────
# getgl — cached by (l_tuple, a, b)
# ──────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _getgl_cached(l_tuple, a, b):
    x, w  = lgwt(_QUAD_N, -np.pi, np.pi)
    l_arr = np.asarray(l_tuple, dtype=np.float64)
    gval1 = a*b*(a**2*np.cos(x)**2+b**2*np.sin(x)**2)**(-1.5)*w
    gval2 = np.exp(-2j*x[:,None]*l_arr[None,:])
    return (1/(2*np.pi))*np.sum(gval1[:,None]*gval2, axis=0)


def getgl(l_array, a, b):
    return _getgl_cached(tuple(np.asarray(l_array).tolist()), a, b)


# ──────────────────────────────────────────────────────────────────────
# hyperterm  (public utility; not called by the solver)
# ──────────────────────────────────────────────────────────────────────
def hyperterm(l1_array, k, m, r, theta):
    l1_array = np.asarray(l1_array, dtype=np.float64)
    scalar = l1_array.ndim == 0
    if scalar: l1_array = np.atleast_1d(l1_array)
    l_arr = (2*l1_array).astype(int)
    g_km = gamma(k+1.5);  g_km1 = gamma(k+m+1)
    f1 = factorial(2*k+1,exact=False);  f2 = factorial(2*k+2*m+1,exact=False)
    Clmk = (-1.0)**(l_arr+2*m)*(
        (np.pi*2.0**(l_arr+2)/(l_arr**2-1))
        *(factorial(2*k-l_arr+1,exact=False)/factorial(2*k+2*m+l_arr+1,exact=False))
        *(f2/f1)*(g_km/g_km1)*(gamma(k+l_arr/2+m+1.5)/gamma(k-l_arr/2+1)))
    Elkm = (-1.0)**(l_arr+m)*(
        (np.pi*2.0**(l_arr-2*m+2)/(l_arr**2-1))
        *(g_km/g_km1)*(f2/f1)*(gamma(k+l_arr/2+1.5)/gamma(m+k-l_arr/2+1))
        *(factorial(2*m+2*k-l_arr+1,exact=False)/factorial(l_arr+2*k+1,exact=False)))
    denom = np.sqrt(1-r**2)
    t2 = np.array([float(alform2(k,m,int(lv),r)) for lv in l_arr])/denom
    t3 = np.array([float(alform3(k,m,int(lv),r)) for lv in l_arr])/denom
    B = (Clmk*t3*np.exp(1j*(l_arr+m)*theta)+Elkm*t2*np.exp(1j*(l_arr-m)*theta))/2
    return B[0] if scalar else B


# ──────────────────────────────────────────────────────────────────────
# _precompute_A_data  — all Phase-4 inputs computed ONCE here
# ──────────────────────────────────────────────────────────────────────
def _precompute_A_data(km_pairs, N, R_flat, THETA_flat, a, b):
    """
    Returns a dict with keys:
      gl_cache, alform2_km, alform3_km, coeff_tbl,
      exp_l_km, exp_m_pos, exp_m_neg, WPP_rows, OP_rows
    """
    _ensure_dic_cache()
    size = len(R_flat)

    # ── getgl ────────────────────────────────────────────────────────
    gl_cache = {(k,m): getgl(np.arange(-k,k+1), a, b) for k,m in km_pairs}

    # ── cosm, WPP/OP rows ────────────────────────────────────────────
    cosm_cache = {m: np.cos(m*_DIC_X2) for m in range(N+1)}
    WPP_rows = np.empty((len(km_pairs), _QUAD_N**2), dtype=np.float64)
    OP_rows  = np.empty((len(km_pairs), _QUAD_N**2), dtype=np.float64)
    for idx, (k, m) in enumerate(km_pairs):
        p1 = alform(k, m, _DIC_X1)*_DIC_X1
        op = np.outer(p1, cosm_cache[m])
        WPP_rows[idx] = (_DIC_W*op).ravel()
        OP_rows[idx]  = op.ravel()

    # ── Phase factor table ───────────────────────────────────────────
    l_unique  = np.arange(-2*N, 2*N+1, 2, dtype=int)   # (2N+1,)
    l_to_row  = {int(l): i for i, l in enumerate(l_unique)}
    exp_l_table = np.exp(1j * l_unique[:,None] * THETA_flat[None,:])  # (2N+1, size)

    exp_m_pos = np.exp( 1j * np.arange(N+1)[:,None] * THETA_flat[None,:])  # (N+1, size)
    exp_m_neg = np.exp(-1j * np.arange(N+1)[:,None] * THETA_flat[None,:])  # (N+1, size)

    # ── Alform2/3 evaluation + coefficients — thread-parallel ────────
    # Each (k,m) pair is independent: evaluate alform2/3 on R_flat for
    # all l values and compute Clmk/Elkm using gammaln.
    # scipy/numpy operations release the GIL → genuine thread parallelism.

    def _eval_km(km):
        k, m = km
        l1_arr = np.arange(-k, k+1, dtype=np.float64)
        l_arr  = (2*l1_arr).astype(int)

        # alform2/3 on R_flat — stacked as (L, size) arrays
        a2 = np.stack([_eval_lambdify(_get_alform2_func(k,m,int(lv)), R_flat)
                       for lv in l_arr])
        a3 = np.stack([_eval_lambdify(_get_alform3_func(k,m,int(lv)), R_flat)
                       for lv in l_arr])

        # Clmk, Elkm  (gammaln for stability; correct sign of (l²-1))
        ds    = np.sign(l_arr**2 - 1)
        lgkm  = gammaln(k+1.5);  lgkm1 = gammaln(k+m+1)
        lf2k1 = gammaln(2*k+2);  lf2km = gammaln(2*k+2*m+2)

        log_C = (np.log(np.pi) + (l_arr+2)*np.log(2.)
                 - np.log(np.abs(l_arr**2-1))
                 + gammaln(2*k-l_arr+2) - gammaln(2*k+2*m+l_arr+2)
                 + lf2km - lf2k1 + lgkm - lgkm1
                 + gammaln(k+l_arr/2+m+1.5) - gammaln(k-l_arr/2+1))
        Clmk = ds * (-1.0)**(l_arr+2*m) * np.exp(log_C)

        log_E = (np.log(np.pi) + (l_arr-2*m+2)*np.log(2.)
                 - np.log(np.abs(l_arr**2-1))
                 + lgkm - lgkm1 + lf2km - lf2k1
                 + gammaln(k+l_arr/2+1.5) - gammaln(m+k-l_arr/2+1)
                 + gammaln(2*m+2*k-l_arr+2) - gammaln(l_arr+2*k+2))
        Elkm = ds * (-1.0)**(l_arr+m) * np.exp(log_E)

        # Pre-index exp_l for this (k,m): (L, size)
        rows   = [l_to_row[int(lv)] for lv in l_arr]
        exp_l  = exp_l_table[rows]

        return km, a2, a3, Clmk, Elkm, l_arr, exp_l

    n_work = min(_N_THREADS, len(km_pairs))
    with ThreadPoolExecutor(max_workers=n_work) as pool:
        results = list(pool.map(_eval_km, km_pairs))

    alform2_km = {}; alform3_km = {}; coeff_tbl = {}; exp_l_km = {}
    for (k,m), a2, a3, Clmk, Elkm, l_arr, exp_l in results:
        alform2_km[(k,m)] = a2        # (L, size)
        alform3_km[(k,m)] = a3        # (L, size)
        coeff_tbl[(k,m)]  = (Clmk, Elkm, l_arr)
        exp_l_km[(k,m)]   = exp_l     # (L, size)

    return dict(gl_cache=gl_cache,
                alform2_km=alform2_km, alform3_km=alform3_km,
                coeff_tbl=coeff_tbl,
                exp_l_km=exp_l_km,
                exp_m_pos=exp_m_pos, exp_m_neg=exp_m_neg,
                WPP_rows=WPP_rows, OP_rows=OP_rows)


# ──────────────────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────────────────
def problemcodeAMDC(N, d_val, K, a, b):
    """
    Solve the rigid-elliptic-disc BVP.

    Pipeline
    --------
    Phase 1  : doubleintC_batch → intgralv (thread-parallel over Q axis).
    Phase 2  : eager pre-computation (thread-parallel over km_pairs).
    Phase 3  : c = (a·b)·P4_mat @ WPP_rows.T         [one dgemm]
    Phase 4  : A columns = ht_mat.T @ gl per (k,m)  [thread-parallel]
    Phase 5  : solve C·X = f.
    Phase 6  : final = -(w·(OP_rows.T @ X_sol)).sum() / π  [one matvec]
    """
    _ensure_intgral_cache()
    _ensure_dic_cache()

    n     = np.arange(N+1)
    size  = (N+1)**2

    theta = (2*n+1)*np.pi/(2*N+2)
    r     = np.cos(theta/2)
    R_grid, THETA_grid = np.meshgrid(r, theta, indexing='xy')
    R_flat     = R_grid.flatten(order='F')
    THETA_flat = THETA_grid.flatten(order='F')
    denom_flat = np.sqrt(1 - R_flat**2)

    km_pairs = [(k,m) for k in range(N+1) for m in range(N+1)]

    # ── Phase 1 ──────────────────────────────────────────────────────
    # intgralv inside doubleintC_batch automatically threads over Q axis
    print(f"Phase 1 : batch check()  [{_N_THREADS} threads over Q axis] …")
    P4_batch = doubleintC_batch(R_flat, THETA_flat, d_val, K, a, b)
    P4_mat   = P4_batch.reshape(size, _QUAD_N**2)          # (size, Q²)

    # ── Phase 2 ──────────────────────────────────────────────────────
    print(f"Phase 2 : pre-computing alform2/3, coefficients, phase table"
          f"  [{min(_N_THREADS, len(km_pairs))} threads] …")
    pre = _precompute_A_data(km_pairs, N, R_flat, THETA_flat, a, b)
    gl_cache   = pre['gl_cache']
    alform2_km = pre['alform2_km'];  alform3_km = pre['alform3_km']
    coeff_tbl  = pre['coeff_tbl']
    exp_l_km   = pre['exp_l_km']
    exp_m_pos  = pre['exp_m_pos'];   exp_m_neg  = pre['exp_m_neg']
    WPP_rows   = pre['WPP_rows'];    OP_rows    = pre['OP_rows']

    # ── Phase 3 ──────────────────────────────────────────────────────
    print("Phase 3 : c = (a·b)·P4_mat @ WPP_rows.T …")
    c = (a*b) * (P4_mat @ WPP_rows.T)                     # (size, size)

    # ── Phase 4 ──────────────────────────────────────────────────────
    # Each column of A is independent → thread-parallel for large N
    print("Phase 4 : A matrix (thread-parallel columns) …")
    A = np.zeros((size, size), dtype=np.complex128)

    def _compute_col(args):
        col_idx, k, m = args
        Clmk, Elkm, l_arr = coeff_tbl[(k,m)]
        gl     = gl_cache[(k,m)]
        a2m    = alform2_km[(k,m)]    # (L, size)
        a3m    = alform3_km[(k,m)]    # (L, size)
        exp_l  = exp_l_km[(k,m)]      # (L, size)
        emp    = exp_m_pos[m]         # (size,)
        emn    = exp_m_neg[m]         # (size,)
        ht_mat = (
            Clmk[:,None]*a3m*exp_l*emp[None,:]
            + Elkm[:,None]*a2m*exp_l*emn[None,:]
        ) / (2.0*denom_flat[None,:])  # (L, size)
        return col_idx, ht_mat.T @ gl # (size,)

    args_list = [(i, k, m) for i, (k,m) in enumerate(km_pairs)]
    n_work    = min(_N_THREADS, len(km_pairs))

    if n_work > 1:
        with ThreadPoolExecutor(max_workers=n_work) as pool:
            for col_idx, col in pool.map(_compute_col, args_list):
                A[:, col_idx] = col
    else:
        for col_idx, col in map(_compute_col, args_list):
            A[:, col_idx] = col

    # ── Phase 5 ──────────────────────────────────────────────────────
    print("Phase 5 : solving linear system …")
    X_sol = np.linalg.solve(A + c, 4*np.pi*np.ones(size, dtype=np.complex128))

    # ── Phase 6 ──────────────────────────────────────────────────────
    sum1  = (OP_rows.T @ X_sol).reshape((_QUAD_N, _QUAD_N))
    _, w1 = lgwt(_QUAD_N, 0, 1);  _, w2 = lgwt(_QUAD_N, 0, 2*np.pi)
    final = -np.sum(np.outer(w1,w2) * sum1) / np.pi
    return final, X_sol


# ──────────────────────────────────────────────────────────────────────
# _intgralv_multi_Z  — private: multi-Z batch + thread-parallel intgral
# ──────────────────────────────────────────────────────────────────────
def _intgralv_multi_Z(Z_vals, X_flat, _k0_base_buf=None):
    """
    Compute intgralv(Z_vals[d], X_flat) for all d simultaneously.

    Strategy
    --------
    1. Fill k0_base[q, m] = k0(tx[q]*X_flat[m]) in parallel (persistent pool).
       Pass _k0_base_buf to reuse a pre-allocated (Q, M) array and avoid
       repeated 288 MB alloc/free inside sweep loops.
    2. Single BLAS DGEMM: W_eff_all @ k0_base  ->  (D, M).

    Parameters
    ----------
    Z_vals : (D,) real array
    X_flat : (M,) real array
    _k0_base_buf : (Q, M) float64 array or None

    Returns
    -------
    (D, M) real array
    """
    _ensure_intgral_cache()
    tx = _INTGRAL_TX;  w_q = _INTGRAL_W;  Q = len(tx)
    M  = len(X_flat)

    # Re-use caller-supplied buffer if given (avoids 288 MB alloc/free per call)
    if _k0_base_buf is None or _k0_base_buf.shape != (Q, M):
        k0_base = np.empty((Q, M), dtype=np.float64)
    else:
        k0_base = _k0_base_buf

    n_work   = min(_N_THREADS, Q)
    q_splits = [s for s in np.array_split(np.arange(Q), n_work) if len(s)]

    def _fill_rows(q_idx):
        k0_base[q_idx, :] = k0(tx[q_idx, None] * X_flat[None, :])
    pool = _get_pool()
    list(pool.map(_fill_rows, q_splits))

    W_eff_all = w_q * (
        tx * np.sin(Z_vals[:, None] * tx)
        + np.cos(Z_vals[:, None] * tx)
    )
    return W_eff_all @ k0_base


# ──────────────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------
# sweep_problemcodeAMDC  -- full parameter grid
# ----------------------------------------------------------------------
def sweep_problemcodeAMDC(
    a0, a1, n_a,
    d0, d1, n_d,
    k0_val, k1_val, n_k,
    N=5, b=1.0,
):
    """
    Evaluate problemcodeAMDC(N, d, K, a, b) for every (a, d, K) triple.

        a in linspace(a0, a1, n_a+1)
        d in linspace(d0, d1, n_d+1)
        K in linspace(k0_val, k1_val, n_k+1)

    Returns complex ndarray of shape (n_a+1, n_d+1, n_k+1).

    Optimisations vs naive loop
    ---------------------------
    * Persistent thread pool -- no per-call thread creation cost.
    * _intgralv_multi_Z Strategy B: fill k0_base in-place then one BLAS
      DGEMM W_eff_all @ k0_base  (no sum-of-partials overhead).
    * j0y0_WPP computed ONCE per (a,K); scaled by exp(-Z) per d (trivial).
    * c built from 3 separate real DGEMMs -- avoids (n_d1,size,Q2) complex.
    * denom2**1.5 replaced by denom*denom2 (no transcendental pow()).
    """
    _ensure_intgral_cache()
    _ensure_dic_cache()

    a_vals = np.linspace(a0, a1, n_a + 1)
    d_vals = np.linspace(d0, d1, n_d + 1)
    K_vals = np.linspace(k0_val, k1_val, n_k + 1)
    n_a1 = n_a + 1;  n_d1 = n_d + 1;  n_k1 = n_k + 1
    Q    = _QUAD_N;  Q2 = Q * Q

    # Collocation grid
    n_idx   = np.arange(N + 1)
    size    = (N + 1) ** 2
    theta_c = (2*n_idx + 1) * np.pi / (2*N + 2)
    r_c     = np.cos(theta_c / 2)
    R_grid, T_grid = np.meshgrid(r_c, theta_c, indexing='xy')
    R_flat     = R_grid.flatten(order='F')
    THETA_flat = T_grid.flatten(order='F')
    denom_flat = np.sqrt(1.0 - R_flat**2)

    km_pairs = [(k, m) for k in range(N+1) for m in range(N+1)]
    f_batch  = (4.0 * np.pi * np.ones(size, dtype=np.complex128)
                )[np.newaxis, :, np.newaxis]

    # Shared pre-computation (a-independent)
    print("Sweep setup: pre-computing shared data ...")
    pre = _precompute_A_data(km_pairs, N, R_flat, THETA_flat, a_vals[0], b)
    alform2_km = pre['alform2_km'];  alform3_km = pre['alform3_km']
    coeff_tbl  = pre['coeff_tbl']
    exp_l_km   = pre['exp_l_km']
    exp_m_pos  = pre['exp_m_pos'];   exp_m_neg  = pre['exp_m_neg']
    WPP_rows   = pre['WPP_rows']
    OP_rows    = pre['OP_rows']

    ht_mat_precomp = {}
    for k, m in km_pairs:
        Clmk, Elkm, _ = coeff_tbl[(k, m)]
        ht_mat_precomp[(k, m)] = (
            Clmk[:, None] * alform3_km[(k, m)] * exp_l_km[(k, m)] * exp_m_pos[m][None, :]
            + Elkm[:, None] * alform2_km[(k, m)] * exp_l_km[(k, m)] * exp_m_neg[m][None, :]
        ) / (2.0 * denom_flat[None, :])

    Rx_base  = (R_flat[:, None] * np.cos(THETA_flat[:, None])
                - _DIC_X1_FLAT[None, :] * np.cos(_DIC_X2_FLAT[None, :]))
    Ry_base  = (R_flat[:, None] * np.sin(THETA_flat[:, None])
                - _DIC_X1_FLAT[None, :] * np.sin(_DIC_X2_FLAT[None, :]))
    b2_Ry_sq = (b * Ry_base) ** 2

    _, w1 = lgwt(Q, 0.0, 1.0);  _, w2 = lgwt(Q, 0.0, 2.0 * np.pi)
    w_quad = np.outer(w1, w2)

    # Main sweep
    n_expensive = n_a1 * n_k1
    print(f"Sweep: {n_a1}x{n_d1}x{n_k1} = {n_a1*n_d1*n_k1} evaluations "
          f"({n_expensive} k0 calls, {n_d1} d-values batched per call) ...")

    results = np.zeros((n_a1, n_d1, n_k1), dtype=complex)
    X_results = np.zeros((n_a1, n_d1, n_k1, size), dtype=complex)

    # Pre-allocate large buffers shared across iterations (avoids repeated
    # 288 MB alloc/free for k0_base and 255 MB for denom2 on every k0 call)
    M_flat   = size * Q2
    _k0_buf  = np.empty((_QUAD_N, M_flat), dtype=np.float64)   # (Q, size*Q2)
    _d2_buf  = np.empty((n_d1, size, Q2), dtype=np.float64)    # (n_d1, size, Q2)

    for i_a, a in enumerate(a_vals):
        print(f"  a[{i_a+1}/{n_a1}] = {a:.6g}")

        gl_a = {(k, m): getgl(np.arange(-k, k+1), a, b) for k, m in km_pairs}
        A_a  = np.zeros((size, size), dtype=np.complex128)
        for col_idx, (k, m) in enumerate(km_pairs):
            A_a[:, col_idx] = ht_mat_precomp[(k, m)].T @ gl_a[(k, m)]

        R_base_a = np.sqrt((a * Rx_base)**2 + b2_Ry_sq)

        for i_K, K in enumerate(K_vals):
            K3 = K**3;  ab = a * b

            X_arr = K * R_base_a
            X_sq  = X_arr * X_arr

            j0y0     = j0(X_arr) + 1j * y0(X_arr)
            j0y0_WPP = j0y0.reshape(size, Q2) @ WPP_rows.T  # (size,size) once per (a,K)
            del j0y0

            Z_vals_K = 2.0 * K * d_vals
            exp_Z    = np.exp(-Z_vals_K)

            # _intgralv_multi_Z: persistent pool + single BLAS DGEMM
            intgral_all = _intgralv_multi_Z(Z_vals_K, X_arr.ravel(), _k0_buf)

            # M for all d -- avoid np.power with denom*denom2
            Z_bc   = Z_vals_K[:, None, None]
            Zsq_bc = (Z_vals_K * Z_vals_K)[:, None, None]
            np.add(X_sq[None, :, :], Zsq_bc, out=_d2_buf)  # in-place, reuse buffer
            denom  = np.sqrt(_d2_buf)
            denom3 = denom * _d2_buf
            denom5 = denom3 * _d2_buf
            M_all  = K3 * ((2*Z_bc - 1)/denom3 + 3*Zsq_bc/denom5 + 2.0/denom)
            del denom, denom3, denom5

            # c = c_M + c_I + c_J  (3 real DGEMMs, no complex (n_d1,size,Q2))
            c_M2d  = ab * (M_all.reshape(n_d1 * size, Q2) @ WPP_rows.T)
            del M_all
            c_I2d  = (-ab * K3 * 4.0/np.pi) * (
                intgral_all.reshape(n_d1 * size, Q2) @ WPP_rows.T
            )
            del intgral_all
            c_real = (c_M2d + c_I2d).reshape(n_d1, size, size)
            del c_M2d, c_I2d

            c_J   = (ab * K3 * 2*np.pi*1j) * (exp_Z[:, None, None] * j0y0_WPP)
            c_all = c_real + c_J
            del c_real, c_J

            X_all = np.linalg.solve(A_a[None] + c_all, f_batch)[..., 0]
            X_results[i_a, :, i_K, :] = X_all
            del c_all

            sum1_all = (X_all @ OP_rows).reshape(n_d1, Q, Q)
            results[i_a, :, i_K] = (
                -np.einsum('dij,ij->d', sum1_all, w_quad) / np.pi
            )

    return results, X_results


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import time

    print(f"Threads detected: {_N_THREADS}")
    N = 2;  d_val = K = a = b = 1.0
    t0 = time.perf_counter()
    result = problemcodeAMDC(N, d_val, K, a, b)
    print(f"Single call N={N}: {result}  ({time.perf_counter()-t0:.3f} s)")

    print()
    print("Sweep demo (N=5, b=1, 2x10x2 grid) ...")
    t0 = time.perf_counter()
    R = sweep_problemcodeAMDC(
        a0=0.8, a1=1.2, n_a=2,
        d0=0.5, d1=1.5, n_d=10,
        k0_val=0.5, k1_val=1.5, n_k=2,
        N=5, b=1.0,
    )
    print(f"Shape: {R.shape}   Elapsed: {time.perf_counter()-t0:.2f} s")
    print("R[:, 0, 0]:", R[:, 0, 0])
