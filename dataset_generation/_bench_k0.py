import sys, time
import numpy as np
sys.path.insert(0, '/home/sandip/Downloads/BTP/Rigid-elliptic-disc')
import problemcodeAMDC_opt as mod
mod._ensure_intgral_cache()
from scipy.special import k0

tx = mod._INTGRAL_TX
Q  = len(tx)
print(f'Q={Q}, _N_THREADS={mod._N_THREADS}')

np.random.seed(42)
X_flat = np.random.uniform(0.01, 5.0, 36*10000)
M = len(X_flat)
print(f'Total k0 evals per call: {Q*M:,}  ({Q*M/1e6:.1f}M)')

# 1: single-thread
t0 = time.perf_counter()
k0_base = k0(tx[:, None] * X_flat[None, :])
st = time.perf_counter()-t0
print(f'Single-thread k0 ({Q},{M}): {st:.3f}s')

# 2: N_THREADS parallel over Q
pool = mod._get_pool()
k0_base2 = np.empty((Q, M))
q_splits = [s for s in np.array_split(np.arange(Q), mod._N_THREADS) if len(s)]

def fill(q_idx):
    k0_base2[q_idx] = k0(tx[q_idx, None] * X_flat[None, :])

list(pool.map(fill, q_splits))  # warm-up
t0 = time.perf_counter()
for _ in range(3):
    list(pool.map(fill, q_splits))
mt = (time.perf_counter()-t0)/3
print(f'{mod._N_THREADS}-thread k0:  {mt:.3f}s/call  ({st/mt:.1f}x speedup)')

# 3: DGEMM W_eff @ k0_base
D = 11
W = np.random.randn(D, Q)
_ = W @ k0_base2  # warm-up
t0 = time.perf_counter()
for _ in range(10):
    res = W @ k0_base2
dg = (time.perf_counter()-t0)/10
print(f'DGEMM ({D},{Q})@({Q},{M}): {dg*1000:.1f}ms/call')

print(f'\nTotal per (a,K) call (threaded k0 + DGEMM): {mt+dg:.3f}s')
print(f'Vs observed sweep: ~1.4s/call  => overhead = ~{1.4-mt-dg:.2f}s from M/N/c computation')
