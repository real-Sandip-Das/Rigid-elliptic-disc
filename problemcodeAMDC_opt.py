"""
problemcodeAMDC_opt.py
Optimized version of problemcodeAMDC.py.

Key optimizations:
1. Gauss-Legendre nodes/weights computed ONCE and cached globally.
2. intgralv vectorized over all (X, Z) pairs in one batched call, eliminating
   per-element intgralv calls inside check/kernel.
3. doubleintC pulled OUT of the l-loop — result depends only on (k,m) so it is
   computed once per (k,m) pair and reused for all l values.
4. getgl pulled OUT of the l-loop for the same reason.
5. hyperterm fully vectorized — all l_arr elements processed without a Python loop.
6. alform* functions evaluated once per (k,m) on x1 and stored, rather than
   being re-evaluated inside every inner loop iteration.
7. Final summation kept as a single vectorised outer-product accumulation.
"""

import numpy as np
import sympy as sp
from scipy.special import hankel1, kn, gamma, factorial
import functools

# ==========================================
# Gauss-Legendre quadrature (cached globally)
# ==========================================
@functools.lru_cache(maxsize=None)
def _legendre_nodes(N, a, b):
    """Return (nodes, weights) on [a, b] — cached by (N, a, b)."""
    x, w = np.polynomial.legendre.leggauss(N)
    x_mapped = 0.5 * (a * (1 - x) + b * (1 + x))
    w_mapped  = w * 0.5 * (b - a)
    return x_mapped, w_mapped


def lgwt(N, a, b):
    return _legendre_nodes(N, a, b)


# ==========================================
# Shared quadrature grids (set once at import time)
# We lazily build them on first call to problemcodeAMDC so N is known.
# ==========================================
_QUAD_N = 100  # Gauss order used everywhere

# Cached nodes for intgralv  (interval [0, pi/2])
_INTGRAL_X, _INTGRAL_W = None, None

def _ensure_intgral_cache():
    global _INTGRAL_X, _INTGRAL_W
    if _INTGRAL_X is None:
        _INTGRAL_X, _INTGRAL_W = lgwt(_QUAD_N, 0.0, np.pi / 2)


# ==========================================
# Symbolic Function Caches
# ==========================================
@functools.lru_cache(maxsize=None)
def _get_alform_func(k, m):
    r = sp.Symbol('r')
    l = m + 2*k + 1
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
    """Evaluate a lambdified sympy function, broadcasting scalars if needed."""
    res = func(s)
    if np.isscalar(res) and np.ndim(s) > 0:
        return np.full_like(s, res, dtype=np.float64)
    return res


def alform(k, m, s):
    return _eval_lambdify(_get_alform_func(k, m), s)


def alform2(k, m, l_val, s):
    return _eval_lambdify(_get_alform2_func(k, m, l_val), s)


def alform3(k, m, l_val, s):
    return _eval_lambdify(_get_alform3_func(k, m, l_val), s)


# ==========================================
# intgralv — vectorized over (Z, X) arrays
# ==========================================
def intgralv(Z, X):
    """
    Compute the integral for all (Z, X) pairs simultaneously.
    Z, X : arbitrary broadcastable ndarrays (or scalars).
    Returns an array of the same broadcast shape.
    """
    _ensure_intgral_cache()
    x_q = _INTGRAL_X   # shape (Q,)
    w_q = _INTGRAL_W   # shape (Q,)

    Z = np.asarray(Z)
    X = np.asarray(X)

    # Determine the true output shape by broadcasting Z and X together.
    # This correctly handles all combinations: scalar/scalar, scalar/1-D,
    # 1-D/1-D, 2-D/2-D, etc.
    out_shape = np.broadcast_shapes(Z.shape, X.shape)
    ndim_out  = len(out_shape)

    tx = np.tan(x_q)                               # (Q,)
    # Reshape tx and w so that axis 0 is the quadrature axis and the
    # remaining axes broadcast against out_shape.
    tx_ = tx.reshape((len(tx),) + (1,) * ndim_out)  # (Q, 1, ..., 1)
    w_  = w_q.reshape((len(w_q),) + (1,) * ndim_out)

    # Broadcast Z and X to out_shape, then add leading quadrature axis.
    Z_ = np.broadcast_to(Z, out_shape)[np.newaxis, ...]  # (1, *out_shape)
    X_ = np.broadcast_to(X, out_shape)[np.newaxis, ...]

    integrand = (tx_ * np.sin(Z_ * tx_) + np.cos(tx_ * Z_)) * kn(0, tx_ * X_)
    return np.sum(w_ * integrand, axis=0)


# ==========================================
# check — fully vectorized (no loops)
# ==========================================
def check(r, theta, s, alpha, d_val, K, a, b):
    x   = a * r * np.cos(theta)
    y   = b * r * np.sin(theta)
    gi  = a * s * np.cos(alpha)
    eta = b * s * np.sin(alpha)

    R  = np.sqrt((x - gi)**2 + (y - eta)**2)
    X  = K * R
    z  = -d_val
    nu = -d_val
    Z  = -K * (z + nu)

    denom2 = X**2 + Z**2
    denom  = np.sqrt(denom2)

    M = (K**3) * (
        (2 * Z - 1) / (denom2**1.5) +
        3 * Z**2 / (denom2**2.5) +
        1 / denom
    ) + (K**3) / denom

    N_val = (K**3) * (
        2 * np.pi * 1j * hankel1(0, X) * np.exp(-Z)
        - (4 / np.pi) * intgralv(Z, X)
    )
    return M + N_val


# ==========================================
# kernel
# ==========================================
def kernel(x, y, s, alpha, K, a, b, d_val):
    gi  = a * s * np.cos(alpha)
    eta = b * s * np.sin(alpha)
    R   = np.sqrt((x - gi)**2 + (y - eta)**2)
    X   = K * R
    Y   = K * d_val

    denom2 = X**2 + Y**2
    P = (K**2) * (
        2 * Y / (denom2**1.5) +
        2 / np.sqrt(denom2) +
        2 * np.pi * 1j * np.exp(-Y) * hankel1(0, X)
        - (4 / np.pi) * intgralv(Y, X)
    )
    return P


# ==========================================
# doubleintC — per (k, m) only, no l-dependence
# ==========================================
# Cached quadrature grids for doubleintC
_DIC_X1, _DIC_W1 = None, None
_DIC_X2, _DIC_W2 = None, None
_DIC_W  = None
_DIC_X1_FLAT, _DIC_X2_FLAT = None, None

def _ensure_dic_cache():
    global _DIC_X1, _DIC_W1, _DIC_X2, _DIC_W2, _DIC_W, _DIC_X1_FLAT, _DIC_X2_FLAT
    if _DIC_X1 is None:
        _DIC_X1, _DIC_W1 = lgwt(_QUAD_N, 0, 1)
        _DIC_X2, _DIC_W2 = lgwt(_QUAD_N, 0, 2 * np.pi)
        _DIC_W  = np.outer(_DIC_W1, _DIC_W2)          # (100, 100)
        # Quadrature points for (s, alpha) — Fortran order to match original
        _S, _ALPHA = np.meshgrid(_DIC_X1, _DIC_X2, indexing='xy')
        _DIC_X1_FLAT = _S.flatten(order='F')
        _DIC_X2_FLAT = _ALPHA.flatten(order='F')


def doubleintC_precomputed(r_point, theta_point, depth, K, a, b):
    """
    Compute the (r,theta)-dependent part of doubleintC that does NOT depend on (k,m):
    the 2-D array p4 of shape (100,100).
    """
    _ensure_dic_cache()
    p3_flat = check(r_point, theta_point,
                    _DIC_X1_FLAT, _DIC_X2_FLAT,
                    depth, K, a, b)
    p4 = p3_flat.reshape((_QUAD_N, _QUAD_N), order='F')
    return p4.T   # Transposed as in original (p4.')


def doubleintC(k, m, r_point, theta_point, depth, K, a, b, p4_T=None):
    """
    Compute the double integral.
    If p4_T is supplied (pre-computed), skip the expensive check() call.
    """
    _ensure_dic_cache()
    if p4_T is None:
        p4_T = doubleintC_precomputed(r_point, theta_point, depth, K, a, b)

    p1 = alform(k, m, _DIC_X1) * _DIC_X1   # (100,)
    p2 = np.cos(m * _DIC_X2)               # (100,)
    p  = np.outer(p1, p2)                   # (100, 100)

    value = a * b * np.sum(_DIC_W * p * p4_T)
    return value


# ==========================================
# getgl — cached by (l_array_tuple, a, b)
# ==========================================
@functools.lru_cache(maxsize=None)
def _getgl_cached(l_tuple, a, b):
    """Inner cached implementation; l_array passed as a hashable tuple."""
    x, w = lgwt(_QUAD_N, -np.pi, np.pi)
    l_array = np.asarray(l_tuple, dtype=np.float64)
    gval1 = a * b * (a**2 * np.cos(x)**2 + b**2 * np.sin(x)**2)**(-1.5) * w
    gval2 = np.exp(-2j * x[:, None] * l_array[None, :])
    return (1 / (2 * np.pi)) * np.sum(gval1[:, None] * gval2, axis=0)


def getgl(l_array, a, b):
    l_array = np.asarray(l_array)
    return _getgl_cached(tuple(l_array.tolist()), a, b)


# ==========================================
# hyperterm — vectorized over l_array
# ==========================================
def hyperterm(l1_array, k, m, r, theta):
    """
    Vectorized hyperterm: processes all l-values in l1_array at once.
    r, theta are scalars (single collocation point).
    """
    l1_array = np.asarray(l1_array, dtype=np.float64)
    is_scalar = l1_array.ndim == 0
    if is_scalar:
        l1_array = np.atleast_1d(l1_array)

    l_arr = (2 * l1_array).astype(int)   # shape (L,)

    # --- Scalar precomputes (depend on k, m only) ---
    g_km    = gamma(k + 1.5)
    g_km1   = gamma(k + m + 1)
    f_2k1   = factorial(2*k + 1, exact=False)
    f_2k2m1 = factorial(2*k + 2*m + 1, exact=False)
    g_k15   = gamma(k + 1.5)

    # --- Vectorized coefficient computation over l_arr ---
    # Clmk
    sign_C  = (-1.0)**(l_arr + 2*m)
    numer_C = (np.pi * 2.0**(l_arr + 2) / (l_arr**2 - 1)) \
              * (factorial(2*k - l_arr + 1, exact=False) / factorial(2*k + 2*m + l_arr + 1, exact=False)) \
              * (f_2k2m1 / f_2k1) \
              * (g_km / g_km1) \
              * (gamma(k + l_arr/2 + m + 1.5) / gamma(k - l_arr/2 + 1))
    Clmk = sign_C * numer_C

    # Elkm
    sign_E  = (-1.0)**(l_arr + m)
    numer_E = (np.pi * 2.0**(l_arr - 2*m + 2) / (l_arr**2 - 1)) \
              * (g_k15 / g_km1) \
              * (f_2k2m1 / f_2k1) \
              * (gamma(k + l_arr/2 + 1.5) / gamma(m + k - l_arr/2 + 1)) \
              * (factorial(2*m + 2*k - l_arr + 1, exact=False) / factorial(l_arr + 2*k + 1, exact=False))
    Elkm = sign_E * numer_E

    denom = np.sqrt(1 - r**2)

    # alform2 and alform3 must be evaluated per distinct l_val (still needs sympy)
    term2_vals = np.array([float(alform2(k, m, int(lv), r)) for lv in l_arr], dtype=np.float64)
    term3_vals = np.array([float(alform3(k, m, int(lv), r)) for lv in l_arr], dtype=np.float64)

    term3 = term3_vals / denom
    Xlkm  = Clmk * term3 * np.exp(1j * l_arr * theta) * np.exp(1j * m * theta)

    term2 = term2_vals / denom
    Ylkm  = Elkm * term2 * np.exp(1j * l_arr * theta) * np.exp(-1j * m * theta)

    B = (Xlkm + Ylkm) / 2

    return B[0] if is_scalar else B


# ==========================================
# Main solver — optimized triple loop
# ==========================================
def problemcodeAMDC(N, d_val, K, a, b):
    # Ensure shared caches are ready
    _ensure_intgral_cache()
    _ensure_dic_cache()

    n     = np.arange(N + 1)
    size  = (N + 1)**2

    A = np.zeros((size, size), dtype=np.complex128)
    c = np.zeros((size, size), dtype=np.complex128)

    theta = (2 * n + 1) * np.pi / (2 * N + 2)
    r     = np.cos(theta / 2)

    R_grid, THETA_grid = np.meshgrid(r, theta, indexing='xy')
    R_flat     = R_grid.flatten(order='F')
    THETA_flat = THETA_grid.flatten(order='F')

    # ----------------------------------------------------------------
    # Pre-compute p4_T for every collocation point l — this is the
    # expensive check() call inside doubleintC and is independent of (k,m).
    # ----------------------------------------------------------------
    print("Pre-computing check() for all collocation points …")
    p4_T_list = []
    for l in range(size):
        p4_T_list.append(
            doubleintC_precomputed(R_flat[l], THETA_flat[l], d_val, K, a, b)
        )

    # ----------------------------------------------------------------
    # Pre-compute getgl for every (k,m) — independent of l.
    # ----------------------------------------------------------------
    print("Pre-computing getgl for all (k,m) pairs …")
    gl_cache  = {}   # (k,m) -> gl array
    for k in range(N + 1):
        for m in range(N + 1):
            l_arr = np.arange(-k, k + 1)
            gl_cache[(k, m)] = getgl(l_arr, a, b)

    # ----------------------------------------------------------------
    # Pre-compute alform(k,m, x1)*x1 and cos(m*x2) for final summation
    # ----------------------------------------------------------------
    print("Pre-computing alform for final summation …")
    alform_x1_cache = {}
    cosm_x2_cache   = {}
    for k in range(N + 1):
        for m in range(N + 1):
            alform_x1_cache[(k, m)] = alform(k, m, _DIC_X1) * _DIC_X1
            cosm_x2_cache[m]        = np.cos(m * _DIC_X2)

    # ----------------------------------------------------------------
    # Main fill loop — O(size * (N+1)^2) but inner work is cheap
    # ----------------------------------------------------------------
    print("Filling A and c matrices …")
    for l in range(size):
        p4_T = p4_T_list[l]
        p = 0
        for k in range(N + 1):
            for m in range(N + 1):
                l_arr = np.arange(-k, k + 1)
                gl    = gl_cache[(k, m)]

                # hyperterm at single point (r, theta)
                ht = hyperterm(l_arr, k, m, R_flat[l], THETA_flat[l])
                A[l, p] = np.dot(ht, gl)

                # doubleintC with pre-computed p4_T
                p1 = alform_x1_cache[(k, m)]
                p2 = cosm_x2_cache[m]
                pp = np.outer(p1, p2)
                c[l, p] = a * b * np.sum(_DIC_W * pp * p4_T)

                p += 1

    C = A + c
    f = 4 * np.pi * np.ones(size, dtype=np.complex128)

    print("Solving linear system …")
    X_sol = np.linalg.solve(C, f)

    # ----------------------------------------------------------------
    # Final summation — vectorised outer-product accumulation
    # ----------------------------------------------------------------
    x1, w1 = lgwt(_QUAD_N, 0, 1)
    x2, w2 = lgwt(_QUAD_N, 0, 2 * np.pi)

    q_idx = 0
    sum1  = np.zeros((_QUAD_N, _QUAD_N), dtype=np.complex128)

    for k in range(N + 1):
        for m in range(N + 1):
            p1 = alform_x1_cache[(k, m)]
            p2 = cosm_x2_cache[m]
            sum1 += X_sol[q_idx] * np.outer(p1, p2)
            q_idx += 1

    w      = np.outer(w1, w2)
    final  = -np.sum(w * sum1) / np.pi   # a*b cancels

    return final


# ==========================================
# Entry point
# ==========================================
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
