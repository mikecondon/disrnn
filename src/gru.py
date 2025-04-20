# This file is bespoke for Practical Biomedical Modelling Assignment
# No license applies to this file.

"""
Tiny GRU for Interpretability based on

Ji-An, L., Benna, M. K., & Mattar, M. G. (2023). Discovering Cognitive Strategies with Tiny Recurrent Neural Networks. 
Animal Behavior and Cognition. 
https://doi.org/10.1101/2023.04.12.536629
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


class HkGRU(hk.RNNCore):
  """A standard GRU RNN core implemented using Haiku."""

  def __init__(
      self,
      hidden_size: int,
      target_size: int,
      name: Optional[str] = None,
  ):
    """
    Initializes the GRU RNNCore.

    Args:
      hidden_size: The number of hidden units in the GRU layer.
      target_size: The dimensionality of the output prediction.
      name: Optional name for the Haiku module.
    """
    super().__init__(name=name)
    self._hidden_size = hidden_size
    self._target_size = target_size

    self.gru_core = hk.GRU(hidden_size=self._hidden_size)

    self.output_layer = hk.Linear(output_size=self._target_size, name="gru_output_linear")

  def __call__(self, observations: jnp.ndarray, prev_state: jnp.ndarray):
    """
    Performs one step of the GRU computation.
    """
    gru_output, next_state = self.gru_core(observations, prev_state)
    y_hat = self.output_layer(gru_output) 

    return y_hat, next_state

  def initial_state(self, batch_size: Optional[int]):
    """
    Returns the initial hidden state for the GRU.
    """
    return self.gru_core.initial_state(batch_size)


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
    penalty_scale: float = 0,
    do_plot: bool = False,
    ) -> Tuple[hk.Params, optax.OptState, Dict[str, np.ndarray]]:
    """Trains a standard GRU network (like HkGRU).

    Args:
    make_network: A function that, when called, returns a Haiku RNN (e.g., HkGRU).
    training_dataset: A DatasetRNN, containing the data you wish to train on.
                        Expected to yield (xs, ys) tuples.
    validation_dataset: A DatasetRNN, containing validation data.
    opt: The Optax optimizer to use.
    random_key: A jax random key for initialization.
    opt_state: An optimizer state suitable for opt. If None, initializes a new one.
    params: Network parameters. If None, initializes a new network.
    n_steps: Number of training steps.
    max_grad_norm: Maximum global norm for gradient clipping.
    do_plot: Boolean controlling whether a learning curve is plotted.

    Returns:
    params: Trained parameters.
    opt_state: Optimizer state at the end of training.
    losses: Dictionary containing 'training_loss' and 'validation_loss' arrays.
    """
    sample_xs, _ = next(iter(training_dataset))

    # --- Haiku network setup ---
    # Step 1: Define the function to unroll the network over a sequence
    def unroll_network(xs: jnp.ndarray) -> jnp.ndarray:
        """Applies the RNN core over the time dimension of the input sequence."""
        core = make_network()
        batch_size = jnp.shape(xs)[1]
        initial_state = core.initial_state(batch_size)
        y_hats_sequence, _ = hk.dynamic_unroll(core, xs, initial_state)
        return y_hats_sequence

    # Step 2: Transform the network function into init and apply functions
    model = hk.transform(unroll_network)

    # Initialisation
    if random_key is None:
        random_key = jax.random.PRNGKey(42)

    if params is None:
        random_key, key1 = jax.random.split(random_key)
    params = model.init(key1, sample_xs)
    logging.info("Initialized new network parameters.")

    if opt_state is None:
        opt_state = opt.init(params)
        logging.info("Initialized new optimizer state.")

    def categorical_log_likelihood(labels: jnp.ndarray, output_logits: jnp.ndarray) -> jnp.ndarray:
        """Computes Categorical Cross-Entropy loss, handling negative labels as masks."""
        mask = jnp.logical_not(labels < 0)
        log_probs = jax.nn.log_softmax(output_logits)

        if labels.shape[2] != 1:
            raise ValueError(
            f'Categorical loss requires target labels to have shape (..., 1), '
            f'but got shape {labels.shape}'
        )
        labels_squeezed = labels[..., 0]
        one_hot_labels = jax.nn.one_hot(labels_squeezed, num_classes=output_logits.shape[-1])

        log_liks = one_hot_labels * log_probs
        masked_log_liks = jnp.multiply(log_liks, mask)
        loss = -jnp.nansum(masked_log_liks)

        return loss

    def categorical_loss(params: hk.Params, xs: jnp.ndarray, labels: jnp.ndarray, random_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Calculates Categorical Cross-Entropy loss for the given batch."""
        output_logits = model.apply(params, None, xs)
        loss = categorical_log_likelihood(labels, output_logits)
        return loss

    compute_loss = jax.jit(categorical_loss)

    @jax.jit
    def train_step(
        params: hk.Params,
        opt_state: optax.OptState,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        random_key: jax.random.PRNGKey # Pass key if needed by loss/model
    ) -> Tuple[jnp.ndarray, hk.Params, optax.OptState]:
        """Performs a single training step including loss, gradients, and update.
        Gradients are clipped."""
        loss_val, grads = jax.value_and_grad(compute_loss, argnums=0)(
            params, xs, ys, random_key
        )
        grads, opt_state = opt.update(grads, opt_state)
        clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
        params = optax.apply_updates(params, clipped_grads)

        return loss_val, params, opt_state

    training_loss_history = []
    validation_loss_history = []
    last_val_loss = np.inf

    logging.info(f"Starting training for {n_steps} steps")
    
    pbar = tqdm(jnp.arange(n_steps), desc="Training Progress", leave=True, position=1)
    for step in pbar:
        xs_train, ys_train = next(training_dataset)

        random_key, train_key, val_key = jax.random.split(random_key, 3)
        l_train, params, opt_state = train_step(
            params, opt_state, xs_train, ys_train, train_key
        )

        # Validate periodically
        if step % 50 == 0 or step == n_steps - 1:

            training_loss_history.append(l_train / xs_train.shape[1])
            xs_val, ys_val = validation_dataset._xs, validation_dataset._ys
            l_validation = compute_loss(params, xs_val, ys_val, val_key)
            last_val_loss = float(l_validation)
            validation_loss_history.append(last_val_loss / xs_val.shape[1])

            pbar.set_postfix({
                "Train Loss": f"{l_train / xs_train.shape[1]:.3e}",
                "Val Loss": f"{last_val_loss / xs_val.shape[1]:.3e}"
            })
            logging.info(
                'Step {} of {}. Training Loss: {:.2e}. Validation Loss: {:.2e}'
                .format(step + 1, n_steps, l_train, l_validation))

    pbar.close()
    logging.info(f"Training finished after {n_steps} steps.")
    logging.info(f"Final Training Loss: {training_loss_history[-1]:.3e}")
    if validation_loss_history:
        logging.info(f"Final Validation Loss: {validation_loss_history[-1]:.3e}")


    # Plotting
    if n_steps > 1 and do_plot:
        plt.figure(figsize=(14, 4.2))
        sns.set_theme(style='ticks', font_scale=1.6, rc={'axes.labelsize':18, 'axes.titlesize':18}) 
        val_steps = np.linspace(0, n_steps, len(validation_loss_history))
        plt.semilogy(val_steps, training_loss_history, label='Training Loss', alpha=0.8, color=sns.color_palette()[0])
        plt.semilogy(val_steps, validation_loss_history, label='Validation Loss', linestyle='--', marker='o', markersize=4, alpha=0.8, color=sns.color_palette()[1])
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        sns.despine()
        plt.show()

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
