# This file is bespoke for Practical Biomedical Modelling Assignment
# No license applies to this file.

from typing import Optional
from src import rnn_utils
import numpy as np
import pandas as pd
import json, os
import jax
import jax.numpy as jnp
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
  session_order = df['Session'].unique()
  eps = df.value_counts('Session').reindex(session_order)
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

def model_saver(params: dict, 
                model,
                beta,
                tiny_latent_size,
                out_dir, 
                dt, 
                train_prop,
                loss: Optional[dict]=None):
  
  if model == 'disrnn':
      param = f"{beta:.0e}"
  else:
      param = f"{tiny_latent_size:.0f}"
  cv = f"{train_prop*100:.0f}-{(1-train_prop)*100:.0f}"

  file_path = os.path.join(out_dir, f"params_{model}_{param}_{cv}_{dt}.json")
  with open(file_path, 'w') as f:
    json.dump(params, f, indent=4, cls=rnn_utils.NpEncoder)

  if loss != None:
    out_df = pd.DataFrame(loss)
    file_path = os.path.join(out_dir, f"loss_{model}_{param}_{cv}_{dt}.csv")
    out_df.to_csv(file_path)


def model_loader(params_file, loss_file=None):
  """
  Loads model parameters and training loss from a JSON file saved with NpEncoder
  and converts lists back to JAX arrays.
  """
  with open(params_file, 'r') as f:
    params_raw = json.load(f)
  params = rnn_utils.to_jnp(params_raw)
  if loss_file != None:
     loss_df = pd.read_csv(loss_file, index_col=0)
     return params, loss_df

  return params


def sampler(logits, sample_type, key=None, sigmoid=False):
  """
  The shape is (no_timesteps, no_sessions, 2). The dimensions of axis 2
  are p_est of choosing left or right respectively.
  Here we sample from these probabilities.
  """
  if key == None:
    key = 42
  rng = np.random.default_rng(seed=key)
  if sigmoid:
    p = jax.nn.sigmoid(jnp.concatenate((-logits, logits), axis=2))
  else:   
    p = jax.nn.softmax(logits, axis=2)
  if sample_type == 'greedy':
    out_arr = np.argmax(p, axis=2)
  elif sample_type == 'thompson':
    p_new = p[:,:,0]/np.sum(p, axis=2)
    rands = rng.random(size=np.shape(p_new))
    out_arr = (rands > p_new).astype(int)
  return np.where(np.all(logits==0, axis=2), -1, out_arr)



def switch_bars(history, switches, h_len=3, symm=True, prob=True):
  """
  This function generates conditional probabilities of switching given each
  3 letter history.
  """
  chars = 'lrLR'
  seq_dict = {''.join(seq): [0,0] for seq in itertools.product(chars, repeat=3)}

  for session_i in range(np.shape(history)[1]):
    h_session = history[:, session_i]
    s_session = switches[:, session_i]
    for ts_i in range(h_len, np.shape(h_session)[0]):
      # -1 is the padding value. If i encounter a -1, all following vals are -1
      # so can be ignored
      if h_session[ts_i, 0] == -1:
        break
      h_ts = h_session[ts_i-h_len: ts_i]
      key = ''.join([chars[int(a+2*b)] for a, b in h_ts])
      if s_session[ts_i] == h_session[ts_i-1, 0]:
        seq_dict[key][0] += 1
      else:
        seq_dict[key][1] += 1
  if symm:
    seq_dict = symm_switch_bars(seq_dict, h_len)
  if prob:
    seq_dict = {key: mean(val) for key, val in seq_dict.items()}
  return seq_dict


def symm_switch_bars(p_dict, h_len):
  eq_chars = 'aAbB'
  eqs = list(itertools.product(eq_chars, repeat=h_len))[:len(eq_chars)**h_len//2]
  eq_dict = {''.join(seq): 0 for seq in eqs}
  for seq in eq_dict:
    tran1 = seq.translate(str.maketrans('abAB', 'lrLR'))
    tran2 = seq.translate(str.maketrans('abAB', 'rlRL'))
    p1, p2 = p_dict[tran1], p_dict[tran2]
    eq_dict[seq] = [(p1[0]+p2[0]), (p1[1]+p2[1])]
  return eq_dict


def blocker(df, ds, mask, m=10, n=10):
    """
    Returns an array of shape (2*n timesteps, sessions, blocks, features)
    where the third axis corresponds to which block change you want
    """
    block_changes = []
    block_window = []
    max_bl_len = 0
    eps = df['Session'].unique()
    for i, session_i in enumerate(eps):
        sess_data = df[df['Session']==session_i].sort_values('Trial')
        sess_decis = ds[:,i]
        sess_mask = mask[:, i]
        block_idx = np.squeeze(np.argwhere(np.abs(np.diff(sess_data['Target']))), axis=1)
        block_changes.append(block_idx)
        sess_changes = []
        bl_len = 0
        for j in block_idx:
            if j-m>=0 and j+n<len(sess_data):
                if not sess_mask[j+n]:
                   break
                change_arr = sess_decis[j-m:j+n]
                target_arr = sess_data['Target'].iloc[j-m:j+n]
                sess_changes.append(np.stack((change_arr, target_arr)))
                bl_len += 1
        if bl_len > max_bl_len:
            max_bl_len = bl_len
        block_window.append(sess_changes)
    padded_arrays = []
    for elem in block_window:
        pad_len = max_bl_len - np.shape(elem)[0]
        np_pad_width = [(0, pad_len)] + [(0, 0)] * (len(np.shape(elem))-1)
        
        # Pad the array
        padded_arr = np.pad(
            elem, 
            pad_width=np_pad_width, 
            mode='constant', 
            constant_values=-1
        )
        padded_arrays.append(padded_arr)
    out_arr = np.stack(padded_arrays).transpose((3, 0, 1, 2))
    return out_arr


def log_likelihood_normalised(
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
    loss = jnp.nansum(masked_log_liks)
    return loss / np.sum(labels!=-1)