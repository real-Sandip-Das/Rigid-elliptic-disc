import sys, time
import numpy as np
sys.path.insert(0, '/home/sandip/Downloads/BTP/Rigid-elliptic-disc')
import problemcodeAMDC_opt as mod
mod._ensure_intgral_cache()
mod._ensure_dic_cache()
from scipy.special import k0, j0, y0

tx = mod._INTGRAL_TX
Q  = len(tx); Q2 = Q*Q; N = 5; size = (N+1)**2
n_d1 = 11; K = 1.0; a = 1.0; b = 1.0; K3 = K**3; ab = a*b

np.random.seed(42)
X_arr  = np.random.uniform(0.01, 5.0, (size, Q2))
X_sq   = X_arr * X_arr
WPP_rows = np.random.randn(size, Q2)
intgral_all = np.random.randn(n_d1, size*Q2)  # (11, 360000)
d_vals = np.linspace(0.5, 1.5, n_d1)
Z_vals_K = 2.0 * K * d_vals

def bench(name, fn, reps=5):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    print(f'{name}: {(time.perf_counter()-t0)/reps*1000:.1f}ms')

# j0y0
bench("j0+y0 (size,Q2)", lambda: j0(X_arr) + 1j*y0(X_arr))
j0y0 = j0(X_arr) + 1j*y0(X_arr)
bench("j0y0 @ WPP (size,size)", lambda: j0y0.reshape(size, Q2) @ WPP_rows.T)
j0y0_WPP = j0y0.reshape(size, Q2) @ WPP_rows.T

# M computation
bench("denom2 + denom", lambda: np.sqrt(X_sq[None,:,:] + (Z_vals_K**2)[:,None,None]))
Z_bc   = Z_vals_K[:, None, None]
Zsq_bc = (Z_vals_K*Z_vals_K)[:, None, None]
def m_bench():
    denom2 = X_sq[None,:,:] + Zsq_bc
    denom = np.sqrt(denom2)
    denom3 = denom * denom2
    denom5 = denom3 * denom2
    return K3 * ((2*Z_bc-1)/denom3 + 3*Zsq_bc/denom5 + 2.0/denom)
bench("M_all full (11,36,10000)", m_bench)
M_all = m_bench()

# c_M DGEMM
bench("c_M (396,10000)@(10000,36)", lambda: ab * (M_all.reshape(n_d1*size, Q2) @ WPP_rows.T))
# c_I DGEMM
bench("c_I (396,10000)@(10000,36)", lambda: (-ab*K3*4/np.pi) * (intgral_all.reshape(n_d1*size, Q2) @ WPP_rows.T))
# c_J scaling
exp_Z = np.exp(-Z_vals_K)
bench("c_J scaling (trivial)", lambda: (ab*K3*2*np.pi*1j) * (exp_Z[:,None,None] * j0y0_WPP))
