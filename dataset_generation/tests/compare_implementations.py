import numpy as np
import dataset_generation.problemcodeAMDC as old
import dataset_generation.problemcodeAMDC_opt as opt

a_b_list = [1.0, 1.25, 1.5, 1.75, 2.0]
d_b_list = [0.1, 0.2, 0.3, 0.4]
K_list = [0.5, 1.0, 1.5, 2.0]

for a in [1.0, 1.75]:
    for d in [0.2, 0.4]:
        for K in [0.5, 2.0]:
            print(f"Testing a={a}, d={d}, K={K}")
            
            # Old version
            final_old = old.problemcodeAMDC(5, d, K, a, 1.0)
            am_old = np.real(np.pi * final_old * a)
            damp_old = np.imag(np.pi * final_old * a)
            
            # Opt version
            final_opt, _ = opt.problemcodeAMDC(5, d, K, a, 1.0)
            am_opt = np.real(np.pi * final_opt * a)
            damp_opt = np.imag(np.pi * final_opt * a)
            
            print(f"Old: AM={am_old:.6f}, Damp={damp_old:.6f}")
            print(f"Opt: AM={am_opt:.6f}, Damp={damp_opt:.6f}")
            print("-" * 40)
