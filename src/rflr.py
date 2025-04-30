# This file is bespoke for Practical Biomedical Modelling Assignment
# No license applies to this file.

"""
Recursive Formulation of Logistic Regression based on

Beron, C. C., Neufeld, S. Q., Linderman, S. W., & Sabatini, B. L. (2022). Mice exhibit stochastic and efficient action switching during probabilistic decision making. 
Proceedings of the National Academy of Sciences, 119(15), e2113961119. 
https://doi.org/10.1073/pnas.2113961119
"""

from src import rnn_utils
import haiku as hk
import optax
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any
from tqdm.notebook import tqdm
import logging
import seaborn as sns

DatasetRNN = rnn_utils.DatasetRNN


class RFLR(hk.RNNCore):
  """RFLR implemented using Haiku RNN style."""

  def __init__(
      self,
      name: Optional[str] = None,
  ):
    """
    Initializes the RFLR RNNCore.
    """
    super().__init__(name=name)

  def __call__(self, observations: jnp.ndarray, prev_state: jnp.ndarray):
    """
    Performs one step of the RFLR computation.
    """
    alpha = hk.get_parameter(name='alpha', 
                             shape=[], 
                             init=hk.initializers.RandomNormal(stddev=0.1))
    
    beta = hk.get_parameter(name='beta', 
                             shape=[], 
                             init=hk.initializers.RandomNormal(stddev=0.1))
    
    tau = hk.get_parameter(name='tau',
                           shape=[],
                           init=hk.initializers.Constant(1))
    
    # rectify tau
    tau = jnp.maximum(tau, 1e-6)

    # remember that unroll will unroll over the timestep dimension, but you need to
    # allow for an arbitrary batch size.
    c_bar_t = 2 * observations[..., 0:1] - 1
    r_t = observations[..., 1:2]

    next_state = jnp.exp(-1/tau) * prev_state + beta * jnp.multiply(c_bar_t, r_t)
    y_hat = alpha * c_bar_t + next_state

    return y_hat, next_state

  def initial_state(self, batch_size: Optional[int]):
    """
    Returns the initial hidden state for the RFLR.
    Here, the latent size is just 1 for e^(-1/tau) + beta * c * r, but still need
    the extra dimension
    """
    if batch_size is None:
        state_shape = (1,)
    else:
        state_shape = (batch_size, 1)
    return jnp.zeros(state_shape, jnp.float32)
  


def train_rflr(
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

    def unroll_rflr(xs: jnp.ndarray) -> jnp.ndarray:
        """Applies the RNN core over the time dimension of the input sequence."""
        core = make_network()
        batch_size = jnp.shape(xs)[1]
        initial_state = core.initial_state(batch_size)
        y_hats_sequence, _ = hk.dynamic_unroll(core, xs, initial_state)
        return y_hats_sequence

    model = hk.transform(unroll_rflr)

    if random_key is None:
        random_key = jax.random.PRNGKey(42)

    if params is None:
        random_key, key1 = jax.random.split(random_key)
    
    sample_xs, _ = next(training_dataset)
    params = model.init(key1, sample_xs)
    opt_state = opt.init(params)

    @jax.jit
    def categorical_log_likelihood(params: hk.Params, xs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """Computes Categorical Cross-Entropy loss, handling negative labels as masks."""
        mask = jnp.logical_not(labels < 0)
        log_odds = model.apply(params, None, xs)

        ps = jax.nn.log_sigmoid(jnp.concatenate((-log_odds, log_odds), axis=-1))
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

        return -jnp.nansum(masked_log_liks) + l2_penalty_sum_sq
    

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
        # xs_train, ys_train = next(training_dataset)
        xs_train, ys_train = next(training_dataset)

        random_key, _, _ = jax.random.split(random_key, 3)
        
        l_train, params, opt_state = train_step(
            params, opt_state, xs_train, ys_train
        )

        # Validate periodically
        if step % 50 == 0 or step == n_steps - 1:

            training_loss_history.append(l_train / xs_train.shape[1])
            xs_val, ys_val = validation_dataset._xs, validation_dataset._ys
            lr_xs_val, lr_ys_val = xs_val, ys_val
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
