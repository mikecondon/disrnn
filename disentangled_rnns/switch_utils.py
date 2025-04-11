from typing import Optional

from disentangled_rnns.library import rnn_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import json, os
import itertools

DatasetRNN = rnn_utils.DatasetRNN

def mean(t1):
  if t1 == [0, 0]:
    return 0
  return t1[1] / (t1[0]+t1[1])

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
  return rnn_utils.DatasetRNN(xs=xs, ys=ys, batch_size=batch_size)

def model_saver(params: dict, beta, dt, test_prop,
                loss: Optional[dict]=None):
  cv = f"{test_prop*100:.0f}-{(1-test_prop)*100:.0f}"
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


def split_saver(tr_eps, va_eps, dt, test_prop):
  out_df = pd.concat((pd.DataFrame({'Session': tr_eps.values,
                                    'Side': 'train'},
                                    index=tr_eps.index.to_list()),
                      pd.DataFrame({'Session': va_eps.values,
                                    'Side': 'validation'},
                                    index=va_eps.index.to_list())))
  cv = f'{test_prop*100:.0f}-{(1-test_prop)*100:.0f}'
  directory = "/Users/michaelcondon/workspaces/pbm_group2/disentangled_rnns/models"
  file_path = os.path.join(directory, f'split_{dt}_{cv}.csv')
  out_df.to_csv(file_path)


def split_loader(file_path):
  df = pd.read_csv(file_path, index_col=0)
  tr_eps = df[df['Side']=='train']['Session']
  va_eps = df[df['Side']=='validation']['Session']
  return tr_eps, va_eps

def sampler(logits, sample_type, key=None):
  """
  The shape is (no_timesteps, no_sessions, 2). The dimensions of axis 2
  are p_est of choosing left or right respectively.
  Here we sample from these probabilities.
  """
  if key == None:
    key = 42
  rng = np.random.default_rng(seed=key)

  if sample_type == 'greedy':
    return np.argmax(logits, axis=2)
  elif sample_type == 'thompson':
    p = logits[:,:,0]/np.sum(logits, axis=2)
    rands = rng.random(size=np.shape(p))
    return (rands < p).astype(int)


def switch_bars(history, switches, h_len=3, symm=True):
  """
  This function generates conditional probabilities of switching given each
  3 letter history.
  """
  chars = 'lrLR'
  seq_dict = {''.join(seq): [0,0] for seq in itertools.product(chars, repeat=3)}

  for session_i in range(np.shape(history._xs)[1]):
    h_session = history._xs[:, session_i]
    s_session = switches._xs[:, session_i]
    for ts_i in range(h_len, np.shape(h_session)[0]):
      # -1 is the padding value. If i encounter a -1, all following vals are -1
      # so can be ignored
      if h_session[ts_i, 0] == -1:
        break
      h_ts = h_session[ts_i-h_len: ts_i]
      key = ''.join([chars[int(a+2*b)] for a, b in h_ts])
      if s_session[ts_i, 0] == h_session[ts_i-1, 0]:
        seq_dict[key][0] += 1
      else:
        seq_dict[key][1] += 1

  p_dict = {key: mean(val) for key, val in seq_dict.items()}
  if symm:
    return symm_switch_bars(p_dict, h_len)
  return p_dict


def symm_switch_bars(p_dict, h_len):
  eq_chars = 'aAbB'
  eqs = list(itertools.product(eq_chars, repeat=h_len))[:len(eq_chars)**h_len//2]
  eq_dict = {''.join(seq): 0 for seq in eqs}
  for seq in eq_dict:
    tran1 = seq.translate(str.maketrans('abAB', 'lrLR'))
    tran2 = seq.translate(str.maketrans('abAB', 'rlRL'))
    eq_dict[seq] = (p_dict[tran1] + p_dict[tran2]) / 2
  return eq_dict
