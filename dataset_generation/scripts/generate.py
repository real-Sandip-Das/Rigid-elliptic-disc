from dataset_generation.generate_dataset import generate_dataset

generate_dataset(
    a0=1.0,
    a1=2.0,
    n_a=10,
    d0=0.1,
    d1=0.4,
    n_d=15,
    k0_val=0.1,
    k1_val=2.0,
    n_k=19,
    N=2,
    filename="full_pinn_dataset.csv",
)
