# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to train, evaluate, and examine Haiku RNNs."""
import json
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
import chex
import haiku as hk
import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm.auto import tqdm
import seaborn as sns


class DatasetRNN():
  """Holds a dataset for training an RNN, consisting of inputs and targets.

     Both inputs and targets are stored as [timestep, episode, feature]
     Serves them up in batches
  """

  def __init__(self,
               xs: np.ndarray,
               ys: np.ndarray,
               batch_size: Optional[int] = None):
    """Do error checking and bin up the dataset into batches.

    Args:
      xs: Values to become inputs to the network.
        Should have dimensionality [timestep, episode, feature]
      ys: Values to become output targets for the RNN.
        Should have dimensionality [timestep, episode, feature]
      batch_size: The size of the batch (number of episodes) to serve up each
        time next() is called. If not specified, all episodes in the dataset
        will be served

    """

    if batch_size is None or batch_size > xs.shape[1]:
      batch_size = xs.shape[1]
    if batch_size == 0:
      batch_size = 1

    # Error checking
    # Do xs and ys have the same number of timesteps?
    if xs.shape[0] != ys.shape[0]:
      msg = ('number of timesteps in xs {} must be equal to number of timesteps'
             ' in ys {}.')
      raise ValueError(msg.format(xs.shape[0], ys.shape[0]))

    # Do xs and ys have the same number of episodes?
    if xs.shape[1] != ys.shape[1]:
      msg = ('number of timesteps in xs {} must be equal to number of timesteps'
             ' in ys {}.')
      raise ValueError(msg.format(xs.shape[0], ys.shape[0]))

    # Is the number of episodes divisible by the batch size?
    # if xs.shape[1] % batch_size != 0:
    #   msg = 'dataset size {} must be divisible by batch_size {}.'
    #   raise ValueError(msg.format(xs.shape[1], batch_size))

    # Property setting
    self._xs = xs
    self._ys = ys
    self._batch_size = batch_size
    self._dataset_size = self._xs.shape[1]
    self._idx = 0
    self.n_batches = self._dataset_size // self._batch_size + 1

  def __iter__(self):
    return self

  def __next__(self):
    """Return a batch of data, including both xs and ys."""

    # Define the chunk we want: from idx to idx + batch_size
    # Check that we're not trying to overshoot the size of the dataset
    # assert end <= self._dataset_size
    # Update the index for next time
    # if end == self._dataset_size:
    if self._idx+self._batch_size >= self._dataset_size:
      # x = jnp.concatenate((self._xs[:, start:], self._xs[:, :end%self._dataset_size]), axis=1)
      # y = jnp.concatenate((self._ys[:, start:], self._ys[:, :end%self._dataset_size]), axis=1)
      self._idx = 0
    start = self._idx
    end = start + self._batch_size
    # else:
    x = self._xs[:, start:end]
    y = self._ys[:, start:end]
    self._idx = end
    if np.shape(x)[1]!=self._batch_size or np.shape(y)[1]!=self._batch_size:
      raise ValueError

    # pad_n = max(0, end - self._dataset_size)
    # x = jnp.pad(self._xs[:, start:end, :], ((0,0), (0,pad_n), (0,0)), 'constant', constant_values=-1.0)
    # y = jnp.pad(self._ys[:, start:end, :], ((0,0), (0,pad_n), (0,0)), 'constant', constant_values=-1.0)
    # x = self._xs[:, start:end]

    return x, y


def nan_in_dict(d: np.ndarray | dict[str, Any]):
  """Check a nested dict (e.g. hk.params) for nans."""
  if not isinstance(d, dict):
    return np.any(np.isnan(d))
  else:
    return any(nan_in_dict(v) for v in d.values())


## Training Loop
def train_network(
    make_network: Callable[[], hk.RNNCore],
    training_dataset: DatasetRNN,
    validation_dataset: DatasetRNN,
    opt: optax.GradientTransformation = optax.adam(1e-3),
    random_key: Optional[chex.PRNGKey] = None,
    opt_state: Optional[optax.OptState] = None,
    params: Optional[hk.Params] = None,
    n_steps: int = 1000,
    max_grad_norm: float = 1e10,
    penalty_scale: float = 0,
    do_plot: bool = False,
) -> Tuple[hk.Params, optax.OptState, Dict[str, np.ndarray]]:
  """Trains a network.

  Args:
    make_network: A function that, when called, returns a Haiku RNN
    training_dataset: A DatasetRNN, containing the data you wish to train on
    validation_dataset: A DatasetRNN, containing the data you wish to use for
      validation
    opt: The optimizer you'd like to use to train the network
    random_key: A jax random key, to be used in initializing the network
    opt_state: An optimzier state suitable for opt
      If not specified, will initialize a new optimizer from scratch
    params:  A set of parameters suitable for the network given by make_network
      If not specified, will begin training a network from scratch
    n_steps: An integer giving the number of steps you'd like to train for
    max_grad_norm:  Gradient clipping. Default to a very high ceiling
    penalty_scale: The magnitude of the information bottleneck penalty (beta)
    loss: A string indicating which loss function to use. mse for mean-squared
      error loss where targets are scalars, categorical for log-likelihood loss
      where targets are integer category labels. 'penalized' assumes that the
      final element of the network output is a penalty that needs to be scaled
      and added to the loss (as in hk.DisRNN), losses without 'penalized' assume
      that all elements of the network output are predictions about the target
      (as in hk.LSTM). Specifically:
        'mse': scalar data, network does not output a penalty
        'penalized_mse': scalar data, network computes a penalty
        'categorical': categorical data, network does not output a penalty
        'penalized_categorical': categorical data, network computes a penalty
    do_plot: Boolean that controls whether a learning curve is plotted

  Returns:
    params: Trained parameters
    opt_state: Optimizer state at the end of training
    losses: Losses on both datasets
  """

  sample_xs, _ = next(training_dataset)  # Get a sample input, for shape

  # Haiku, step one: Define the batched network
  def unroll_network(xs):
    core = make_network()
    batch_size = jnp.shape(xs)[1]
    state = core.initial_state(batch_size)
    ys, _ = hk.dynamic_unroll(core, xs, state)
    return ys

  # Haiku, step two: Transform the network into a pair of functions
  # (model.init and model.apply)
  model = hk.transform(unroll_network)

  # PARSE INPUTS
  if random_key is None:
    random_key = jax.random.PRNGKey(0)
  # If params have not been supplied, start training from scratch
  if params is None:
    random_key, key1 = jax.random.split(random_key)
    params = model.init(key1, sample_xs)
  # It an optimizer state has not been supplied, start optimizer from scratch
  if opt_state is None:
    opt_state = opt.init(params)

  def categorical_log_likelihood(
      labels: np.ndarray, output_logits: np.ndarray
  ) -> float:
    # Mask any errors for which label is negative
    mask = jnp.logical_not(labels < 0)
    log_probs = jax.nn.log_softmax(output_logits)
    if labels.shape[2] != 1:
      raise ValueError(
          'Categorical loss function requires targets to be of dimensionality'
          ' (n_timesteps, n_episodes, 1)'
      )
    one_hot_labels = jax.nn.one_hot(
        labels[:, :, 0], num_classes=output_logits.shape[-1]
    )
    log_liks = one_hot_labels * log_probs
    masked_log_liks = jnp.multiply(log_liks, mask)
    loss = -jnp.nansum(masked_log_liks)
    return loss  # pytype: disable=bad-return-type  # jnp-type

  def penalized_categorical_loss(
      params, xs, targets, random_key, penalty_scale=penalty_scale
  ) -> float:
    """Treats the last element of the model outputs as a penalty."""
    # (n_steps, n_episodes, n_targets)
    model_output = model.apply(params, random_key, xs)
    output_logits = model_output[:, :, :-1]
    penalty = jnp.sum(model_output[:, :, -1])
    loss = (
        categorical_log_likelihood(targets, output_logits)
        + penalty_scale * penalty
    )
    return loss

  compute_loss = jax.jit(penalized_categorical_loss)

  # Define what it means to train a single step
  @jax.jit
  def train_step(
      params, opt_state, xs, ys, random_key
  ) -> Tuple[float, Any, Any]:
    loss, grads = jax.value_and_grad(compute_loss, argnums=0)(
        params, xs, ys, random_key
    )
    grads, opt_state = opt.update(grads, opt_state)
    clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
    params = optax.apply_updates(params, clipped_grads)
    return loss, params, opt_state

  # Train the network!
  training_loss_history = []
  validation_loss_history = []
  pbar = tqdm(jnp.arange(n_steps), desc="Training Progress", leave=False, position=1)
  for step in pbar:
    random_key, key1, key2 = jax.random.split(random_key, 3)
    # Train on training data
    xs_tr, ys_tr = next(training_dataset)
    l_train, params, opt_state = train_step(params, opt_state, xs_tr, ys_tr, key2)

    if step % 50 == 0 or step == n_steps - 1:
      training_loss_history.append(float(l_train) / np.shape(xs_tr)[1])
      xs_va, ys_va = validation_dataset._xs, validation_dataset._ys
      l_validation = compute_loss(params, xs_va, ys_va, key1)
      validation_loss_history.append(float(l_validation) / np.shape(xs_va)[1])
      pbar.set_postfix({
            "Train Loss": f"{l_train / np.shape(xs_tr)[1]:.2e}",
            "Val Loss": f"{l_validation / np.shape(xs_va)[1]:.2e}"
        })
      logging.info(
          'Step {} of {}. Training Loss: {:.2e}. Validation Loss: {:.2e}'
          .format(step + 1, n_steps, l_train, l_validation))
  pbar.close()

  if n_steps > 1 and do_plot:
    plt.figure(figsize=(14, 4.2))
    sns.set_theme(style='ticks', font_scale=1.6, rc={'axes.labelsize':18, 'axes.titlesize':18}) 
    val_steps = np.linspace(0, n_steps, len(validation_loss_history))
    plt.semilogy(val_steps, training_loss_history, label='Training Loss', alpha=0.8, color=sns.color_palette()[0])
    plt.semilogy(val_steps, validation_loss_history, label='Validation Loss', linestyle='--', marker='o', markersize=4, alpha=0.8, color=sns.color_palette()[1])
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend(('Training Set', 'Validation Set'))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    sns.despine()
    plt.show()

  losses = {
      'training_loss': np.array(training_loss_history),
      'validation_loss': np.array(validation_loss_history)
  }

  # Check if anything has become NaN that should not be NaN
  if nan_in_dict(params):
    raise ValueError('NaN in params')
  if training_loss_history and np.isnan(training_loss_history[-1]):
    raise ValueError('NaN in loss')
  if validation_loss_history and np.isnan(validation_loss_history[-1]):
    raise ValueError('NaN in loss')

  return params, opt_state, losses


def eval_network(
    make_network: Callable[[], hk.RNNCore],
    params: hk.Params,
    xs: np.ndarray,
) ->  Tuple[np.ndarray, Any]:
  """Run an RNN with specified params and inputs. Track internal state.

  Args:
    make_network: A Haiku function that defines a network architecture
    params: A set of params suitable for that network
    xs: A batch of inputs [timesteps, episodes, features] suitable for the model

  Returns:
    y_hats: Network outputs at each timestep
    states: Network states at each timestep
  """

  n_steps = jnp.shape(xs)[0]

  def unroll_network(xs):
    core = make_network()
    batch_size = jnp.shape(xs)[1]
    state = core.initial_state(batch_size)

    y_hats = []
    states = []

    for t in range(n_steps):
      states.append(state)
      y_hat, new_state = core(xs[t, :], state)
      state = new_state

      y_hats.append(y_hat)

    return np.asarray(y_hats), np.asarray(states)

  model = hk.transform(unroll_network)
  key = jax.random.PRNGKey(np.random.randint(2**32))
  y_hats, states = model.apply(params, key, xs)

  return np.asarray(y_hats), states


def step_network(
    make_network: Callable[[], hk.RNNCore],
    params: hk.Params,
    state: Any,
    xs: np.ndarray,
) -> Tuple[np.ndarray, Any]:
  """Run an RNN for just a single step on a single input (no batching).

  Args:
    make_network: A Haiku function that defines a network architecture
    params: A set of params suitable for that network
    state: An RNN state suitable for that network
    xs: An input for a single timestep from a single episode, with
      shape [n_features]

  Returns:
    y_hat: The output given by the network, with dimensionality [n_features]
    new_state: The new RNN state of the network
  """

  def step_sub(xs):
    core = make_network()
    y_hat, new_state = core(xs, state)
    return y_hat, new_state

  model = hk.transform(step_sub)
  key = jax.random.PRNGKey(np.random.randint(2**32))
  y_hat, new_state = model.apply(params, key, np.expand_dims(xs, axis=0))

  return y_hat, new_state


def get_initial_state(make_network: Callable[[], hk.RNNCore],
                      params: Optional[Any] = None) -> Any:
  """Get the default initial state for a network architecture.

  Args:
    make_network: A Haiku function that defines a network architecture
    params: Optional parameters for the Hk function. If not passed, will init
      new parameters. For many models this will not affect initial state

  Returns:
    initial_state: An initial state from that network
  """

  # The logic below needs a jax randomy key and a sample input in order to work.
  # But neither of these will affect the initial network state, so its ok to
  # generate throwaways
  random_key = jax.random.PRNGKey(np.random.randint(2**32))

  def unroll_network():
    core = make_network()
    state = core.initial_state(batch_size=1)

    return state

  model = hk.transform(unroll_network)

  if params is None:
    params = model.init(random_key)

  initial_state = model.apply(params, random_key)

  return initial_state


def get_new_params(make_network: Callable[[], hk.RNNCore],
                   random_key: Optional[jax.Array] = None) -> Any:
  """Get a new set of random parameters for a network architecture.

  Args:
    make_network: A Haiku function that defines a network architecture
    random_key: a Jax random key

  Returns:
    params: A set of parameters suitable for the architecture
  """

  # If no key has been supplied, pick a random one.
  if random_key is None:
    random_key = jax.random.PRNGKey(np.random.randint(2**32))

  def unroll_network():
    core = make_network()
    state = core.initial_state(batch_size=1)

    return state

  model = hk.transform(unroll_network)

  params = model.init(random_key)

  return params


def to_jnp(list_dict):
  """Recursively convert a dict of lists into one of jnp arrays.
  """
  jnp_dict = dict()
  for key, value in list_dict.items():
    if isinstance(value, dict):
      jnp_dict[key] = to_jnp(value)
    else:
      jnp_dict[key] = jnp.array(value)
  return jnp_dict


class NpEncoder(json.JSONEncoder):
  """Encode Numpy arrays in a format suitable for json.dump.

  This is useful for saving network params, states, etc.
  """

  def default(self, o: Any):
    if isinstance(o, np.integer):
      return int(o)
    if isinstance(o, np.floating):
      return float(o)
    if isinstance(o, np.ndarray):
      return o.tolist()

    if isinstance(o, jnp.integer):
      return int(o)
    if isinstance(o, jnp.floating):
      return float(o)
    if isinstance(o, jnp.ndarray):
      return o.tolist()
    return super().default(o)
