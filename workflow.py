from disentangled_rnns.library import get_datasets
from disentangled_rnns.library import two_armed_bandits
from disentangled_rnns.library import rnn_utils
from disentangled_rnns.library import disrnn
import optax

# Synthetic dataset from a q-learning agent. See other options above.
agent = two_armed_bandits.AgentQ(alpha=0.3, beta=3)
environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
dataset = two_armed_bandits.create_dataset(
    agent,
    environment,
    n_steps_per_session=200,
    n_sessions=1000,
    batch_size=1000,
)

update_mlp_shape = (5,5,5)
choice_mlp_shape = (2,2)
latent_size = 5

def make_network():
  return disrnn.HkDisRNN(update_mlp_shape=update_mlp_shape,
                        choice_mlp_shape=choice_mlp_shape,
                        latent_size=latent_size,
                        obs_size=2, target_size=2)

learning_rate = 1e-3
opt = optax.adam(learning_rate)


# Train one step to initialize
params, opt_state, losses = rnn_utils.train_network(
   make_network,
    dataset,
    dataset,
    opt = optax.adam(1e-2),
    l_train="penalized_categorical",
    n_steps=0)

# Train additional steps
n_steps = 1000
params, opt_state, losses = rnn_utils.train_network(
make_network,
    dataset,
    dataset,
    l_train="penalized_categorical",
    params=params,
    opt_state=opt_state,
    opt = optax.adam(1e-3),
    penalty_scale = 1e-3,
    n_steps=n_steps,
    do_plot = True)


# Eval mode runs the network with no noise
def make_network_eval():
  return disrnn.HkDisRNN(update_mlp_shape=update_mlp_shape,
                        choice_mlp_shape=choice_mlp_shape,
                        latent_size=latent_size,
                        obs_size=2, target_size=2,
                        eval_mode=True)


disrnn.plot_bottlenecks(params, make_network_eval)
disrnn.plot_update_rules(params, make_network_eval)

xs, ys = next(dataset)
_ , network_states = rnn_utils.eval_network(make_network_eval, params, xs)
