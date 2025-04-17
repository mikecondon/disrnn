import os
import math
import pandas as pd
import argparse
from  datetime import datetime

def main(args):
    print(f"Saving results to base directory: {args.out_dir}")
    df = pd.read_csv(args.data_path)

    # shuffle the sessions
    eps = df['Session'].value_counts().sample(frac=1)
    tr_eps = eps.iloc[:math.floor(args.train_prop*len(eps))]
    va_eps = eps.iloc[math.floor(args.train_prop*len(eps)):]

    df_tr = df[df['Session'].isin(tr_eps.index)]
    df_va = df[df['Session'].isin(va_eps.index)]

    cv = f'{args.train_prop*100:.0f}-{(1-args.train_prop)*100:.0f}'

    try:
        tr_path = os.path.join(args.out_dir, f"train_df_{cv}_{args.run_id}.csv")
        df_tr.to_csv(tr_path)
        print(f"saved training data: {tr_path}")
        val_path = os.path.join(args.out_dir, f"validation_df_{cv}_{args.run_id}.csv")
        df_va.to_csv(val_path)
        print(f"saved validation data: {val_path}")

    except FileNotFoundError:
        print(f"Error: Could not find python or src.train. Check paths/environment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Disentangled RNN model.')

    parser.add_argument('--data_path', type=str, default='/Users/michaelcondon/workspaces/pbm_group2/disentangled_rnns/data/raw/bandit_data.csv', help='Path to the bandit_data.csv file.')
    parser.add_argument('--out_dir', type=str, default='/Users/michaelcondon/workspaces/pbm_group2/disentangled_rnns/data/processed', help='Base directory to save models and splits.')
    parser.add_argument('--run_id', type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M"))
    parser.add_argument('--train_prop', type=float, default=0.7, help='Proportion of data to use for training (e.g., 0.7 means 70% train, 30% val).')
    args = parser.parse_args()

    main(args)
