# This file is bespoke for Practical Biomedical Modelling Assignment
# No license applies to this file.

"""
Logistic regression and Recursive formulation of logistic regression based on Beron 2022

Beron, C. C., Neufeld, S. Q., Linderman, S. W., & Sabatini, B. L. (2022). Mice exhibit stochastic and efficient action switching during probabilistic decision making. 
Proceedings of the National Academy of Sciences, 119(15), e2113961119. 
https://doi.org/10.1073/pnas.2113961119
"""

from src import rnn_utils
import optax
from tqdm.auto import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import logging
from typing import Callable, Optional, Tuple, Dict, Any
from jax.example_libraries import optimizers

import jax
import jax.numpy as jnp
import haiku as hk
import optax

DatasetRNN = rnn_utils.DatasetRNN


def vmap_forward_fn(sequence_input, output_dim=1):
  """
  Applies the MLP over the time dimension using hk.vmap. Equivalent to unroll network,
  Input shape: (batch_size, time_steps, feature_dim)
  """
  if sequence_input.ndim != 3:
      raise ValueError("Input must be 3D (time, batch, features)")

  mlp = hk.Linear(output_dim, name="sliding_mlp", with_bias=True)
  vmapped_mlp = hk.vmap(mlp, in_axes=0, out_axes=0, split_rng=False)
  output_sequence = vmapped_mlp(sequence_input)
  return output_sequence


def lr_window_fn(xs, ys=None, lag=5):
    c = xs[:,:,0]*2-1

    c_r = np.pad(np.lib.stride_tricks.sliding_window_view(c * xs[:,:,1], lag, axis=0),
                 ((lag-1, 0), (0,0),(0,0)), mode='constant', constant_values=0)
    if ys is None:
        return np.concatenate((c[:,:,np.newaxis], c_r), axis=2)    
    return np.concatenate((c[:,:,np.newaxis], c_r), axis=2), ys


def train_gru_network(
    make_network: Callable[[], hk.RNNCore],
    training_dataset: DatasetRNN,
    validation_dataset: DatasetRNN,
    opt: optax.GradientTransformation = optax.adam(1e-3),
    random_key: Optional[jax.random.PRNGKey] = None,
    opt_state: Optional[optax.OptState] = None,
    params: Optional[hk.Params] = None,
    n_steps: int = 1000,
    max_grad_norm: float = 1.0,
    ) -> Tuple[hk.Params, optax.OptState, Dict[str, np.ndarray]]:
    """Trains a standard logistic regression."""

    model = hk.transform(make_network)

    if random_key is None:
        random_key = jax.random.PRNGKey(42)

    if params is None:
        random_key, key1 = jax.random.split(random_key)
    
    sample_xs, _ = next(training_dataset)
    params = model.init(key1, lr_window_fn(sample_xs), output_dim=2)
    opt_state = opt.init(params)

    @jax.jit
    def categorical_log_likelihood(params: hk.Params, xs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """Computes Categorical Cross-Entropy loss, handling negative labels as masks."""
        mask = jnp.logical_not(labels < 0)
        log_odds = model.apply(params, None, xs, output_dim=2)

        ps = jax.nn.log_softmax(log_odds)
        if labels.shape[2] != 1:
            raise ValueError(
            f'Categorical loss requires target labels to have shape (..., 1), '
            f'but got shape {labels.shape}'
        )
        labels_squeezed = labels[..., 0]
        one_hot_labels = jax.nn.one_hot(labels_squeezed, num_classes=2)
        log_liks = one_hot_labels * ps
        masked_log_liks = jnp.multiply(log_liks, mask)

        param_leaves = jax.tree_util.tree_leaves(params)
        # Use standard JAX sum over elements in each array, Python sum over the list of arrays
        l2_penalty_sum_sq = sum(jnp.sum(leaf ** 2) for leaf in param_leaves)

        return -jnp.nansum(masked_log_liks)
    

    @jax.jit
    def train_step(
        params: hk.Params,
        opt_state: optax.OptState,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, hk.Params, optax.OptState]:
        """Performs a single training step including loss, gradients, and update.
        Gradients are clipped."""
        loss_val, grads = jax.value_and_grad(categorical_log_likelihood, argnums=0)(
            params, xs, ys
        )
        grads, opt_state = opt.update(grads, opt_state)
        clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
        params = optax.apply_updates(params, clipped_grads)

        return loss_val, params, opt_state


    training_loss_history = []
    validation_loss_history = []

    logging.info(f"Starting training for {n_steps} steps")
    
    pbar = tqdm(jnp.arange(n_steps), desc="Training Progress", leave=True, position=1)
    for step in pbar:
        xs_train, ys_train = next(training_dataset)
        lr_xs_train, lr_ys_train = lr_window_fn(xs_train, ys_train)

        random_key, _, _ = jax.random.split(random_key, 3)
        
        l_train, params, opt_state = train_step(
            params, opt_state, lr_xs_train, lr_ys_train
        )

        # Validate periodically
        if step % 50 == 0 or step == n_steps - 1:

            training_loss_history.append(l_train / xs_train.shape[1])
            xs_val, ys_val = validation_dataset._xs, validation_dataset._ys
            lr_xs_val, lr_ys_val = lr_window_fn(xs_val, ys_val)
            l_validation = categorical_log_likelihood(params, lr_xs_val, lr_ys_val)
            validation_loss_history.append(l_validation / lr_xs_val.shape[1])

            pbar.set_postfix({
                "Train Loss": f"{l_train / xs_train.shape[1]:.3e}",
                "Val Loss": f"{l_validation / lr_xs_val.shape[1]:.3e}"
            })
            logging.info(
                'Step {} of {}. Training Loss: {:.2e}. Validation Loss: {:.2e}'
                .format(step + 1, n_steps, l_train, l_validation))

    pbar.close()
    logging.info(f"Training finished after {n_steps} steps.")
    logging.info(f"Final Training Loss: {training_loss_history[-1]:.3e}")
    if validation_loss_history:
        logging.info(f"Final Validation Loss: {validation_loss_history[-1]:.3e}")


    #  Prepare Results
    losses_dict = {
        'training_loss': np.array(training_loss_history),
        'validation_loss': np.array(validation_loss_history)
    }

    # Final Checks
    if rnn_utils.nan_in_dict(params):
        raise ValueError('NaN detected in final parameters!')
    if training_loss_history and np.isnan(training_loss_history[-1]):
        raise ValueError('NaN detected in final training loss!')
    if validation_loss_history and np.isnan(validation_loss_history[-1]) and len(validation_loss_history) > 0:
        raise ValueError('NaN detected in final validation loss!')

    return params, opt_state, losses_dict
