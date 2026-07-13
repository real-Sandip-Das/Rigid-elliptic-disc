import numpy as np
import sympy as sp
from scipy.special import hankel1, kn, gamma, factorial
import functools


def lgwt(N, a, b):
    """
    Computes the Legendre-Gauss nodes and weights on an interval [a,b]
    with truncation order N.
    """

    x, w = np.polynomial.legendre.leggauss(N)

    x_mapped = 0.5 * (a * (1 - x) + b * (1 + x))
    w_mapped = w * 0.5 * (b - a)

    return x_mapped, w_mapped


@functools.lru_cache(maxsize=None)
def _get_alform_func(k, m):
    r = sp.Symbol("r")
    l = m + 2 * k + 1
    Pl = sp.legendre(l, r)

    P1 = ((-1) ** m) * (1 - r**2) ** (sp.Rational(m, 2)) * sp.diff(Pl, r, m)

    P = P1.subs(r, sp.sqrt(1 - r**2))
    return sp.lambdify(r, P, modules="numpy")


@functools.lru_cache(maxsize=None)
def _get_alform2_func(k, m, l_val):
    r = sp.Symbol("r")

    l1 = int(m + 2 * k + 1)
    m1 = int(-m + l_val)

    coeff = ((-1.0) ** m1) / ((2**l1) * sp.factorial(l1))
    P1 = (
        coeff
        * (1 - r**2) ** (sp.Rational(m1, 2))
        * sp.diff((r**2 - 1) ** l1, r, l1 + m1)
    )
    P = P1.subs(r, sp.sqrt(1 - r**2))
    return sp.lambdify(r, P, modules="numpy")


@functools.lru_cache(maxsize=None)
def _get_alform3_func(k, m, l_val):
    r = sp.Symbol("r")

    l1 = int(m + 2 * k + 1)
    m1 = int(m + l_val)

    coeff = ((-1.0) ** m1) / ((2**l1) * sp.factorial(l1))
    P1 = (
        coeff
        * (1 - r**2) ** (sp.Rational(m1, 2))
        * sp.diff((r**2 - 1) ** l1, r, l1 + m1)
    )
    P = P1.subs(r, sp.sqrt(1 - r**2))
    return sp.lambdify(r, P, modules="numpy")


def alform(k, m, s):
    func = _get_alform_func(k, m)
    res = func(s)

    if np.isscalar(res) and np.ndim(s) > 0:
        return np.full_like(s, res, dtype=np.float64)
    return res


def alform2(k, m, l_val, s):
    func = _get_alform2_func(k, m, l_val)
    res = func(s)
    if np.isscalar(res) and np.ndim(s) > 0:
        return np.full_like(s, res, dtype=np.float64)
    return res


def alform3(k, m, l_val, s):
    func = _get_alform3_func(k, m, l_val)
    res = func(s)
    if np.isscalar(res) and np.ndim(s) > 0:
        return np.full_like(s, res, dtype=np.float64)
    return res


def intgralv(z, X):
    N = 100
    c = 0
    d = np.pi / 2
    x, w = lgwt(N, c, d)

    z = np.asarray(z)
    X = np.asarray(X)

    broadcast_shape = (N,) + (1,) * X.ndim
    x_b = x.reshape(broadcast_shape)
    w_b = w.reshape(broadcast_shape)

    tx = np.tan(x_b)
    imval1 = (tx * np.sin(z * tx) + np.cos(tx * z)) * kn(0, tx * X)

    imval = np.sum(w_b * imval1, axis=0)

    return imval


def check(r, theta, s, alpha, d_val, K, a, b):
    x = a * r * np.cos(theta)
    y = b * r * np.sin(theta)
    gi = a * s * np.cos(alpha)
    eta = b * s * np.sin(alpha)

    R = np.sqrt((x - gi) ** 2 + (y - eta) ** 2)
    X = K * R
    z = -d_val
    nu = -d_val
    Z = -K * (z + nu)

    denom2 = X**2 + Z**2
    denom = np.sqrt(denom2)

    M = (K**3) * (
        ((2 * Z - 1) / (denom2**1.5)) + (3 * Z**2 / (denom2**2.5)) + (1 / denom)
    ) + (K**3) / denom

    N_val = (K**3) * (
        2 * np.pi * 1j * hankel1(0, X) * np.exp(-Z) - (4 / np.pi) * intgralv(Z, X)
    )

    P = M + N_val
    return P


def kernel(x, y, s, alpha, K, a, b, d_val):
    gi = a * s * np.cos(alpha)
    eta = b * s * np.sin(alpha)
    R = np.sqrt((x - gi) ** 2 + (y - eta) ** 2)
    X = K * R
    Y = K * d_val

    denom2 = X**2 + Y**2
    P = (K**2) * (
        2 * Y / (denom2**1.5)
        + 2 / np.sqrt(denom2)
        + 2 * np.pi * 1j * np.exp(-Y) * hankel1(0, X)
        - (4 / np.pi) * intgralv(Y, X)
    )
    return P


def doubleintC(k, m, r, theta, depth, K, a, b):
    x1, w1 = lgwt(100, 0, 1)
    x2, w2 = lgwt(100, 0, 2 * np.pi)

    w = np.outer(w1, w2)

    x, y = np.meshgrid(x1, x2, indexing="xy")

    x_flat = x.flatten(order="F")
    y_flat = y.flatten(order="F")

    p3_flat = check(r, theta, x_flat, y_flat, depth, K, a, b)
    p4 = p3_flat.reshape(x.shape, order="F")

    p1 = alform(k, m, x1) * x1
    p2 = np.cos(m * x2)

    p = np.outer(p1, p2)

    integrand = w * p * p4.T
    value = a * b * np.sum(integrand)

    return value


def getgl(l_array, a, b):
    N = 100
    c = -np.pi
    d = np.pi
    x, w = lgwt(N, c, d)

    l_array = np.asarray(l_array)
    gval1 = a * b * (a**2 * np.cos(x) ** 2 + b**2 * np.sin(x) ** 2) ** (-1.5) * w

    gval2 = np.exp(-2j * x[:, None] * l_array[None, :])
    gval = (1 / (2 * np.pi)) * np.sum(gval1[:, None] * gval2, axis=0)

    return gval


def hyperterm(l1_array, k, m, r, theta):
    l1_array = np.asarray(l1_array)
    is_scalar = l1_array.ndim == 0
    if is_scalar:
        l1_array = np.array([l1_array])

    l_arr = 2 * l1_array
    B = np.zeros_like(l_arr, dtype=np.complex128)

    for i, l_val in enumerate(l_arr):
        l_val = int(l_val)

        Clmk = (
            (-1.0) ** (l_val + 2 * m)
            * (np.pi * 2 ** (l_val + 2) / (l_val**2 - 1))
            * (factorial(2 * k - l_val + 1) / factorial(2 * k + 2 * m + l_val + 1))
            * (factorial(2 * k + 2 * m + 1) / factorial(2 * k + 1))
            * (gamma(k + 1.5) / gamma(k + m + 1))
            * (gamma(k + l_val / 2 + m + 1.5) / gamma(k - l_val / 2 + 1))
        )

        Elkm = (
            (-1.0) ** (l_val + m)
            * (np.pi * 2 ** (l_val - 2 * m + 2) / (l_val**2 - 1))
            * (gamma(k + 1.5) / gamma(k + m + 1))
            * (factorial(2 * m + 2 * k + 1) / factorial(2 * k + 1))
            * (gamma(k + l_val / 2 + 1.5) / gamma(m + k - l_val / 2 + 1))
            * (factorial(2 * m + 2 * k - l_val + 1) / factorial(l_val + 2 * k + 1))
        )

        denom = np.sqrt(1 - r**2)

        term3 = float(alform3(k, m, l_val, r)) / denom
        Xlkm = Clmk * term3 * np.exp(1j * l_val * theta) * np.exp(1j * m * theta)

        term2 = float(alform2(k, m, l_val, r)) / denom
        Ylkm = Elkm * term2 * np.exp(1j * l_val * theta) * np.exp(-1j * m * theta)

        B[i] = (Xlkm + Ylkm) / 2

    return B[0] if is_scalar else B


def problemcodeAMDC(N, d_val, K, a, b):
    n = np.arange(N + 1)
    size = (N + 1) ** 2

    A = np.zeros((size, size), dtype=np.complex128)
    c = np.zeros((size, size), dtype=np.complex128)

    theta = (2 * n + 1) * np.pi / (2 * N + 2)
    r = np.cos(theta / 2)

    R_grid, THETA_grid = np.meshgrid(r, theta, indexing="xy")
    R_flat = R_grid.flatten(order="F")
    THETA_flat = THETA_grid.flatten(order="F")

    for l in range(size):
        p = 0
        for k in range(N + 1):
            for m in range(N + 1):
                l_arr = np.arange(-k, k + 1)

                ht = hyperterm(l_arr, k, m, R_flat[l], THETA_flat[l])
                gl = getgl(l_arr, a, b)

                A[l, p] = np.sum(ht * gl)
                c[l, p] = doubleintC(k, m, R_flat[l], THETA_flat[l], d_val, K, a, b)
                p += 1

    C = A + c
    f = 4 * np.pi * np.ones(size, dtype=np.complex128)

    X = np.linalg.solve(C, f)

    x1, w1 = lgwt(100, 0, 1)
    x2, w2 = lgwt(100, 0, 2 * np.pi)

    q_idx = 0
    sum1 = np.zeros((100, 100), dtype=np.complex128)

    for k in range(N + 1):
        for m in range(N + 1):
            p1 = alform(k, m, x1) * x1
            p2 = np.cos(m * x2)
            sum1 += X[q_idx] * np.outer(p1, p2)
            q_idx += 1

    w = np.outer(w1, w2)
    final = -a * b * np.sum(w * sum1) / (np.pi * a * b)

    return final


if __name__ == "__main__":
    import time

    N = 2
    d_val = 1.0
    K = 1.0
    a = 1.0
    b = 1.0
    t0 = time.perf_counter()
    result = problemcodeAMDC(N, d_val, K, a, b)
    t1 = time.perf_counter()
    print(f"\nResult : {result}")
    print(f"Elapsed: {t1 - t0:.3f} s")
