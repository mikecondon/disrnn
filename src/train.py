import rnn_utils
import disrnn
import gru
import switch_utils
import math
import numpy as np
import jax
import pandas as pd
import optax
from datetime import datetime
import argparse


def load_data(train_path, val_path, batch_size):
    df_tr = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    ds_tr = switch_utils.get_dataset(df_tr, batch_size)
    ds_va = switch_utils.get_dataset(df_val, batch_size)

    return ds_tr, ds_va

def model_factory(model, model_size):
    target_size = model_size['target_size']
    obs_size = model_size['obs_size']
    if model=='disrnn':
        latent_size = model_size['dis_latent_size']
        update_mlp_shape = model_size['update_mlp_shape']
        choice_mlp_shape = model_size['choice_mlp_shape']

        def make_network():
            return disrnn.HkDisRNN(update_mlp_shape=update_mlp_shape,
                                    choice_mlp_shape=choice_mlp_shape,
                                    latent_size=latent_size,
                                    obs_size=obs_size, target_size=target_size)
        train_network = rnn_utils.train_network
        
    elif model=='rnn':
        latent_size = model_size['tiny_latent_size']
        def make_network():
            return gru.HkGRU(hidden_size=latent_size, target_size=target_size)
        train_network = gru.train_gru_network
        
    return make_network, train_network

def main(args):
    """
    Iterate through the mice, and through the beta values, saving the trained
    params and loss for each in a json to disk.
    """
    model_shape = {'dis_latent_size': args.dis_latent_size,
                   'tiny_latent_size': args.tiny_latent_size,
                   'update_mlp_shape': args.update_mlp_shape,
                   'choice_mlp_shape': args.choice_mlp_shape,
                   'obs_size': 2,
                   'target_size': 2}
    
    make_network, train_network = model_factory(args.model, model_shape)
    ds_tr, ds_va = load_data(args.tr_path, args.val_path, args.batch_size)

    params, opt_state, losses = train_network(
        make_network,
        ds_tr,
        ds_va,
        opt = optax.adam(args.learning_rate),
        penalty_scale = args.beta,
        n_steps = args.n_steps,
        do_plot = False)

    if args.model == 'disrnn':
        param = args.beta
    else:
        param = 'args.tiny_latent_size'
    switch_utils.model_saver(params, 
                             args.model, 
                             param, 
                             args.out_dir,
                             dt=args.run_id, 
                             loss=losses, 
                             train_prop=args.train_prop)


def parse_shape(shape_str):
    """
    Converts a comma-separated string of integers into a tuple of integers.
    Used as a type for argparse arguments.
    """
    try:
        # Split the string by commas and remove potential whitespace
        parts = [part.strip() for part in shape_str.split(',') if part.strip()]
        if not parts:
             raise ValueError
        return tuple(int(part) for part in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid shape format: '{shape_str}'. Must be comma-separated integers (e.g., '5,5,5'). Error: {e}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Disentangled RNN model.')

    # Paths and Identifiers
    parser.add_argument('--tr_path', type=str, required=True, help='Path to the bandit_data.csv file.')
    parser.add_argument('--val_path', type=str, required=True, help='Path to the bandit_data.csv file.')
    parser.add_argument('--model', type=str, required=True, help='Which model should be used?')

    parser.add_argument('--out_dir', type=str, default='./models', help='Base directory to save models and splits.')
    parser.add_argument('--run_id', type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help='Unique identifier for this run/sweep (used for grouping outputs). Defaults to current timestamp.')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size for training and validation.')
    parser.add_argument('--train_prop', type=float, default=0.7, help='Proportion of data to use for training')

    # for disRNN
    parser.add_argument('--dis_latent_size', type=int, default=5, help='Size of the latent state in the DisRNN.')
    parser.add_argument('--update_mlp_shape', type=parse_shape, default='5,5,5', help='Shape of the update MLP (comma-separated integers).')
    parser.add_argument('--choice_mlp_shape', type=parse_shape, default='2,2', help='Shape of the choice MLP (comma-separated integers).')

    # for tinyRNN
    parser.add_argument('--tiny_latent_size', type=int, default=1, help='Size of the latent state in the Tiny RNN.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Adam optimizer learning rate.')
    parser.add_argument('--n_steps', type=float, default=1e4, help='Number of training steps.')
    parser.add_argument('--beta', type=float, default=0, help='Penalty scale value (beta) for the penalized loss.')

    args = parser.parse_args()

    main(args)