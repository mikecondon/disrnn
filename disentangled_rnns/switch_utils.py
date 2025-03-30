from typing import NamedTuple, Union

from disentangled_rnns.library import rnn_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

DatasetRNN = rnn_utils.DatasetRNN

def find(s, ch):
  """Find index of character within string."""
  return [i for i, ltr in enumerate(s) if ltr == ch]

def str2int(x):
  return int(x.split('_')[1])

def get_dataset(addr: str, tr_prop: float=1, va_prop: float=0, te_prop:float=0):
  """
  load dataset from filename
  """
  assert math.isclose(sum((tr_prop, va_prop, te_prop)), 1, rel_tol=1e-2)
  assert tr_prop >= 0 and va_prop >= 0 and te_prop >= 0
  df = pd.read_csv(addr)
  mice = df.Mouse.unique()
  ds_list = []
  for m_i in mice:
    # create a dataset object for this mouse
    df_i = df[m_i == df['Mouse']]
    df_i.loc[:, 'Session'] = df_i['Session'].map(str2int)
    eps = df_i['Session'].value_counts().sort_index()
    max_session_length = np.max(eps.values)
    n_sess = len(eps)

    rewards_by_session = -1 * np.ones((max_session_length, n_sess, 1))
    choices_by_session = -1 * np.ones((max_session_length, n_sess, 1))

    for j, sess_j in enumerate(eps.index):
      sess_len = eps[sess_j]
      sess_data = df_i[df_i['Session'] == sess_j].sort_values('blockTrial')
      rewards_by_session[0:sess_len, j, 0] = sess_data['Reward']
      choices_by_session[0:sess_len, j, 0] = sess_data['Decision']

    choice_and_reward = np.concatenate(
        (choices_by_session, rewards_by_session), axis=2
    )
    xs = np.concatenate(
        (0. * np.ones((1, n_sess, 2)), choice_and_reward), axis=0
    )
    ys = np.concatenate((choices_by_session, -1*np.ones((1, n_sess, 1))), axis=0)

    idx = np.random.permutation(np.arange(n_sess))
    xs = xs[:, idx, :]
    ys = ys[:, idx, :]
    tr_end = math.floor(tr_prop * n_sess)
    va_end = math.floor((tr_prop + va_prop) * n_sess)

    ds_m_tr = rnn_utils.DatasetRNN(xs=xs[:, :tr_end, :], ys=ys[:, :tr_end, :])
    ds_m_va = rnn_utils.DatasetRNN(xs=xs[:, tr_end:va_end, :], ys=ys[:, tr_end:va_end, :])
    ds_m_te = rnn_utils.DatasetRNN(xs=xs[:, va_end:, :], ys=ys[:, va_end:, :])

    ds_list.append((m_i, ds_m_tr, ds_m_va, ds_m_te))

  return ds_list
