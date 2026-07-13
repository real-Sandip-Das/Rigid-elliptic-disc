import jax
import jax.numpy as jnp
import torch
import optax
import torchax
import os

from training.config import WaveConfig
from training.model import DeepONetWaveSurrogate


class CoeffPredictor(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, wave_params):
        latent = self.model.encode(wave_params)
        return self.model.mlp_head(latent)


def optimize_target_multistart(
    jax_model_fn, weights, target_idx, maximize=True, num_starts=500, steps=200
):

    bounds_low = jnp.array([1.0, 0.1, 0.1])
    bounds_high = jnp.array([2.0, 0.4, 2.0])

    key = jax.random.PRNGKey(42 + target_idx + (10 if maximize else 0))
    initial_inputs = jax.random.uniform(
        key, shape=(num_starts, 3), minval=bounds_low, maxval=bounds_high
    )

    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(initial_inputs)

    @jax.jit
    def step(carry, _):
        x, opt_state = carry

        def loss_fn(p):
            model_output = jax_model_fn(weights, (p,))
            specific_output = model_output[:, target_idx]
            return -jnp.sum(specific_output) if maximize else jnp.sum(specific_output)

        loss, grads = jax.value_and_grad(loss_fn)(x)

        updates, opt_state = optimizer.update(grads, opt_state, x)
        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, bounds_low, bounds_high)

        return (x, opt_state), loss

    (final_inputs, _), _ = jax.lax.scan(
        step, (initial_inputs, opt_state), None, length=steps
    )

    final_outputs = jax_model_fn(weights, (final_inputs,))
    final_target_vals = final_outputs[:, target_idx]

    if maximize:
        best_start_idx = jnp.argmax(final_target_vals)
    else:
        best_start_idx = jnp.argmin(final_target_vals)

    global_best_input = final_inputs[best_start_idx]
    global_best_output = final_outputs[best_start_idx]

    return global_best_input, global_best_output


def main():
    print("Loading PyTorch model...")
    cfg = WaveConfig()
    cfg.phase2_model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../checkpoints/phase2_final.pth")
    )

    device = torch.device("cpu")
    base_model = DeepONetWaveSurrogate(
        latent_dim=cfg.latent_dim,
        subnet_width=cfg.subnet_width,
        fourier_mapping_size=cfg.fourier_mapping_size,
        fourier_scale=cfg.fourier_scale,
    )
    base_model.load_state_dict(
        torch.load(cfg.phase2_model_path, map_location=device, weights_only=True)
    )
    base_model.eval()

    import torch.nn.utils.parametrize as parametrize

    for module in base_model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )

    torch_model = CoeffPredictor(base_model)
    torch_model.eval()

    print("Extracting to JAX...")
    weights, jax_model_fn = torchax.extract_jax(torch_model)

    targets = [
        ("Maximize AM", 0, True),
        ("Minimize AM", 0, False),
        ("Maximize DC", 1, True),
        ("Minimize DC", 1, False),
    ]

    for name, target_idx, maximize in targets:
        print(f"\n--- {name} ---")
        final_x, final_output = optimize_target_multistart(
            jax_model_fn, weights, target_idx, maximize
        )
        print(
            f"Optimal Inputs -> a/b: {final_x[0]:.4f}, d/b: {final_x[1]:.4f}, K: {final_x[2]:.4f}"
        )
        print(
            f"Predicted Coeffs -> AM: {final_output[0]:.4f}, DC: {final_output[1]:.4f}"
        )


if __name__ == "__main__":
    main()
