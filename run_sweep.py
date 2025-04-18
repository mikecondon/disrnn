import subprocess
import os
from datetime import datetime
from tqdm.auto import tqdm


betas = [1e-3, 3e-3, 1e-2, 3e-2]
units = [1, 2]
data_path = "/Users/michaelcondon/workspaces/pbm_group2/disentangled_rnns/data/processed"
output_dir = "./models"
n_steps = 10000
learning_rate = 1e-3
train_prop = 0.7
latent_size = 5
update_mlp_shape = '5,5,5'
choice_mlp_shape = '2,2'
batch_size = 64
seed = 42
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M")


print(f"Starting Experiment Sweep: {run_id}")
print(f"Saving results to base directory: {output_dir}")
print(f"Running training for betas: {betas}")

with tqdm(total=len(betas)+len(units), desc='Overall Progress') as outer_bar:
    for beta_j in betas:
        outer_bar.set_postfix(beta=f"{beta_j:.0e}")

        # i am using uv because i like uv but you could use python instead of uv run
        command = [
            'python', 'train.py',
            '--tr_path', os.path.join(data_path, 'train_df_70-30_2025-04-17_10-28.csv'),
            '--val_path', os.path.join(data_path, 'validation_df_70-30_2025-04-17_10-28.csv'),
            '--model', 'disrnn',
            '--out_dir', output_dir,
            '--run_id', run_id,
            '--beta', str(beta_j),
            '--n_steps', str(n_steps),
            '--learning_rate', str(learning_rate),
            '--train_prop', str(train_prop),
            '--dis_latent_size', str(latent_size),
            '--update_mlp_shape', update_mlp_shape,
            '--choice_mlp_shape', choice_mlp_shape,
            '--batch_size', str(batch_size),
        ]

        try:
            subprocess.run(command, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running train.py for beta={beta_j:.0e}: {e} !!!")
            break
        except FileNotFoundError:
            print(f"Error: Could not find python or src.train. Check paths/environment.")
            break
        outer_bar.update(1)

    for unit in units:
        outer_bar.set_postfix(latent_size=str(unit))
        command = [
            'python', 'train.py',
            '--tr_path', os.path.join(data_path, 'train_df_70-30_2025-04-17_10-28.csv'),
            '--val_path', os.path.join(data_path, 'validation_df_70-30_2025-04-17_10-28.csv'),
            '--model', 'rnn',
            '--out_dir', output_dir,
            '--run_id', run_id,
            '--beta', str(0),
            '--n_steps', str(n_steps),
            '--learning_rate', str(learning_rate),
            '--train_prop', str(train_prop),
            '--tiny_latent_size', str(unit),
            '--batch_size', str(batch_size),
        ]

        try:
            subprocess.run(command, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running train.py for beta={beta_j:.0e}: {e} !!!")
            break
        except FileNotFoundError:
            print(f"Error: Could not find python or src.train. Check paths/environment.")
            break


        outer_bar.update(1)


print(f"\n--- Experiment Sweep Finished: {run_id} ---")