from typing import Optional

from disentangled_rnns.library import rnn_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import json, os

DatasetRNN = rnn_utils.DatasetRNN

def find(s, ch):
  """Find index of character within string."""
  return [i for i, ltr in enumerate(s) if ltr == ch]

def str2int(x):
  return int(x.split('_')[1])

def get_dataset(df: pd.DataFrame,
                batch_size:Optional[int]=None):
  """
  Create a training, validation and test set for each mouse
  """

  eps = df['Session'].value_counts()
  # create a dataset object for this mouse
  max_session_length = np.max(eps.values)
  n_sess = len(eps)

  rewards_by_session = -1 * np.ones((max_session_length, n_sess, 1))
  choices_by_session = -1 * np.ones((max_session_length, n_sess, 1))

  for j, sess_j in enumerate(eps.index):
    # filter to just this session
    sess_data = df[df['Session']==sess_j].sort_values('Trial')
    sess_len = eps[sess_j]
    rewards_by_session[0:sess_len, j, 0] = sess_data['Reward']
    choices_by_session[0:sess_len, j, 0] = sess_data['Decision']

  choice_and_reward = np.concatenate(
      (choices_by_session, rewards_by_session), axis=2
  )
  xs = np.concatenate(
      (0. * np.ones((1, n_sess, 2)), choice_and_reward), axis=0
  )
  ys = np.concatenate((choices_by_session, -1*np.ones((1, n_sess, 1))), axis=0)

  ds_tr = rnn_utils.DatasetRNN(xs=xs, ys=ys, batch_size=batch_size)

  return ds_tr

def model_saver(params: dict, beta, dt, cv=0,loss: Optional[dict]=None):
  directory = "/Users/michaelcondon/workspaces/pbm_group2/disentangled_rnns/models"
  file_path = os.path.join(directory, f"params_{beta:.0e}_{cv}_{dt}.json")
  with open(file_path, 'w') as f:
    json.dump(params, f, indent=4, cls=rnn_utils.NpEncoder)
  if loss != None:
    file_path = os.path.join(directory, f"loss_{beta:.0e}_{cv}_{dt}.json")
    with open(file_path, 'w') as f:
      json.dump(loss, f, indent=4, cls=rnn_utils.NpEncoder)


def model_loader(params_file, loss_file):
    """
    Loads model parameters and training loss from a JSON file saved with NpEncoder
    and converts lists back to JAX arrays.
    """
    with open(params_file, 'r') as f:
      params_raw = json.load(f)
    params = rnn_utils.to_jnp(params_raw)
    with open(loss_file, 'r') as f:
      loss_raw = json.load(f)
    loss = rnn_utils.to_jnp(loss_raw)

    return params, loss
