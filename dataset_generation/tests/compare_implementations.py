import sys
import os
import pandas as pd
import numpy as np

import dataset_generation.problemcodeAMDC_opt as opt


def main():
    csv_file = os.path.join(os.path.dirname(__file__), "reference_dataset.csv")
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} does not exist.")
        sys.exit(1)

    N = 5
    b = 1.0

    success = True

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        a = float(row["a_b"])
        d = float(row["d_b"])
        K = float(row["wave_frequency_K"])
        ref_am = float(row["Added_Mass"])
        ref_damp = float(row["Damping_Coefficient"])

        print(f"Testing Opt implementation against reference for a={a}, d={d}, K={K}")

        final_opt, _ = opt.problemcodeAMDC(N, d, K, a, b)
        opt_am = np.real(np.pi * final_opt * a)
        opt_damp = np.imag(np.pi * final_opt * a)

        print(f"Ref: AM={ref_am:.6f}, Damp={ref_damp:.6f}")
        print(f"Opt: AM={opt_am:.6f}, Damp={opt_damp:.6f}")

        if not np.isclose(ref_am, opt_am, atol=1e-5):
            print(f"  -> Added mass mismatch! Expected {ref_am}, got {opt_am}")
            success = False
        if not np.isclose(ref_damp, opt_damp, atol=1e-5):
            print(f"  -> Damping mismatch! Expected {ref_damp}, got {opt_damp}")
            success = False
        print("-" * 40)

    if success:
        print("All Opt implementation tests matched the reference data successfully!")
    else:
        print("Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
